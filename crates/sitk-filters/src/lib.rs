//! Phase-0 image filters for sitk-rs, exposed as SimpleITK-style procedural
//! functions (`add`, `cast`, `binary_threshold`, ...).
//!
//! Arithmetic follows ITK's functors, verified against the ITK v6 source
//! (`itkArithmeticOpsFunctors.h`, `itkStatisticsImageFilter.hxx`):
//!
//! - **image ⊕ image** (`add`/`subtract`/`multiply`/`divide`) reproduces
//!   `static_cast<Output>(A op B)` where `A`, `B` are the input pixel type: for
//!   integers this is 2's-complement **wraparound** (`u8` `250 + 10 == 4`), done
//!   here with the type's `wrapping_*` ops so results match exactly and 64-bit
//!   ints keep full precision; for floats it is IEEE arithmetic. `divide` by
//!   zero returns `NumericTraits<Output>::max()` (the type's largest finite
//!   value), matching ITK's `Div` functor.
//! - **image ⊕ constant** uses SimpleITK's `double` constant, so it accumulates
//!   in `f64` and narrows with a saturating cast ([`Scalar::from_f64`]). The
//!   `f64` accumulate matches SimpleITK's double-constant functor; the final
//!   out-of-range float→int cast is undefined in C++, and we define it as
//!   saturation.
//!
//! The struct-style filter API and the remaining ~290 filters arrive with the
//! yaml codegen in a later phase.

pub mod error;
pub mod recursive_gaussian;
pub mod shrink;
pub mod smoothing;

pub use error::{FilterError, Result};
pub use recursive_gaussian::{GaussianOrder, recursive_gaussian, recursive_gaussian_with_order};
pub use shrink::shrink;
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar};
pub use smoothing::smooth_gaussian;

// ---- image ⊕ image functor arithmetic -------------------------------------

/// The four binary arithmetic operations, matching ITK's arithmetic functors.
#[derive(Clone, Copy)]
enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Per-pixel-type arithmetic with ITK functor semantics. Integer ops wrap on
/// overflow (2's complement), matching C++ `static_cast<T>(promoted A op B)`;
/// float ops use IEEE arithmetic. Division by zero returns the type's largest
/// finite value, as ITK's `Div` functor does (`NumericTraits<T>::max()`).
trait Arith: Scalar {
    fn apply(self, rhs: Self, op: ArithOp) -> Self;
}

macro_rules! impl_arith_int {
    ($($t:ty),+ $(,)?) => {$(
        impl Arith for $t {
            #[inline]
            fn apply(self, rhs: Self, op: ArithOp) -> Self {
                match op {
                    ArithOp::Add => self.wrapping_add(rhs),
                    ArithOp::Sub => self.wrapping_sub(rhs),
                    ArithOp::Mul => self.wrapping_mul(rhs),
                    ArithOp::Div => if rhs == 0 { <$t>::MAX } else { self.wrapping_div(rhs) },
                }
            }
        }
    )+};
}

macro_rules! impl_arith_float {
    ($($t:ty),+ $(,)?) => {$(
        impl Arith for $t {
            #[inline]
            fn apply(self, rhs: Self, op: ArithOp) -> Self {
                match op {
                    ArithOp::Add => self + rhs,
                    ArithOp::Sub => self - rhs,
                    ArithOp::Mul => self * rhs,
                    ArithOp::Div => if rhs == 0.0 { <$t>::MAX } else { self / rhs },
                }
            }
        }
    )+};
}

impl_arith_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl_arith_float!(f32, f64);

// ---- shared helpers -------------------------------------------------------

fn build_from_f64<T: Scalar>(size: &[usize], geom: &Image, vals: &[f64]) -> Result<Image> {
    let out: Vec<T> = vals.iter().map(|&v| T::from_f64(v)).collect();
    let mut img = Image::from_vec(size, out)?;
    img.copy_geometry_from(geom);
    Ok(img)
}

/// Build an image of `target` pixel type from `f64` values, copying `geom`'s
/// geometry.
pub(crate) fn image_from_f64(
    target: PixelId,
    size: &[usize],
    geom: &Image,
    vals: &[f64],
) -> Result<Image> {
    dispatch_scalar!(target, build_from_f64, size, geom, vals)
}

fn require_same_shape(a: &Image, b: &Image) -> Result<()> {
    if a.pixel_id() != b.pixel_id() {
        return Err(FilterError::TypeMismatch {
            a: a.pixel_id(),
            b: b.pixel_id(),
        });
    }
    if a.size() != b.size() {
        return Err(FilterError::SizeMismatch {
            a: a.size().to_vec(),
            b: b.size().to_vec(),
        });
    }
    Ok(())
}

// ---- cast -----------------------------------------------------------------

/// `CastImageFilter`: convert an image to another pixel type (`static_cast`
/// semantics via [`Scalar::from_f64`]).
pub fn cast(img: &Image, target: PixelId) -> Result<Image> {
    let vals = img.to_f64_vec();
    image_from_f64(target, img.size(), img, &vals)
}

// ---- binary arithmetic (image ⊕ image) ------------------------------------

fn binary_apply<T: Arith>(a: &Image, b: &Image, op: ArithOp) -> Result<Image> {
    let sa = a.scalar_slice::<T>().expect("dispatch guarantees type");
    let sb = b.scalar_slice::<T>().expect("dispatch guarantees type");
    let out: Vec<T> = sa
        .iter()
        .zip(sb.iter())
        .map(|(&x, &y)| x.apply(y, op))
        .collect();
    let mut img = Image::from_vec(a.size(), out)?;
    img.copy_geometry_from(a);
    Ok(img)
}

fn binary(a: &Image, b: &Image, op: ArithOp) -> Result<Image> {
    require_same_shape(a, b)?;
    dispatch_scalar!(a.pixel_id(), binary_apply, a, b, op)
}

/// `AddImageFilter`: pixel-wise `a + b` (`static_cast<T>(a + b)`; integers wrap).
pub fn add(a: &Image, b: &Image) -> Result<Image> {
    binary(a, b, ArithOp::Add)
}

/// `SubtractImageFilter`: pixel-wise `a - b` (integers wrap).
pub fn subtract(a: &Image, b: &Image) -> Result<Image> {
    binary(a, b, ArithOp::Sub)
}

/// `MultiplyImageFilter`: pixel-wise `a * b` (integers wrap).
pub fn multiply(a: &Image, b: &Image) -> Result<Image> {
    binary(a, b, ArithOp::Mul)
}

/// `DivideImageFilter`: pixel-wise `a / b`; where `b == 0` yields the output
/// type's largest finite value (`NumericTraits<T>::max()`), matching ITK's `Div`
/// functor.
pub fn divide(a: &Image, b: &Image) -> Result<Image> {
    binary(a, b, ArithOp::Div)
}

// ---- binary arithmetic (image ⊕ constant) ---------------------------------

fn constant_apply<T: Scalar>(a: &Image, c: f64, op: fn(f64, f64) -> f64) -> Result<Image> {
    let sa = a.scalar_slice::<T>().expect("dispatch guarantees type");
    let out: Vec<T> = sa.iter().map(|&x| T::from_f64(op(x.as_f64(), c))).collect();
    let mut img = Image::from_vec(a.size(), out)?;
    img.copy_geometry_from(a);
    Ok(img)
}

fn constant(a: &Image, c: f64, op: fn(f64, f64) -> f64) -> Result<Image> {
    dispatch_scalar!(a.pixel_id(), constant_apply, a, c, op)
}

/// `a + c` for a scalar constant.
pub fn add_constant(a: &Image, c: f64) -> Result<Image> {
    constant(a, c, |x, y| x + y)
}

/// `a - c` for a scalar constant.
pub fn subtract_constant(a: &Image, c: f64) -> Result<Image> {
    constant(a, c, |x, y| x - y)
}

/// `a * c` for a scalar constant.
pub fn multiply_constant(a: &Image, c: f64) -> Result<Image> {
    constant(a, c, |x, y| x * y)
}

// ---- unary ----------------------------------------------------------------

/// `AbsImageFilter`: pixel-wise absolute value, output type follows input.
pub fn abs(img: &Image) -> Result<Image> {
    let vals: Vec<f64> = img.to_f64_vec().iter().map(|v| v.abs()).collect();
    image_from_f64(img.pixel_id(), img.size(), img, &vals)
}

// ---- threshold ------------------------------------------------------------

/// `BinaryThresholdImageFilter`: `inside` where `lower <= v <= upper`, else
/// `outside`. Output pixel type is `UInt8`, matching SimpleITK's default.
pub fn binary_threshold(
    img: &Image,
    lower: f64,
    upper: f64,
    inside: u8,
    outside: u8,
) -> Result<Image> {
    let vals = img.to_f64_vec();
    let out: Vec<u8> = vals
        .iter()
        .map(|&v| {
            if v >= lower && v <= upper {
                inside
            } else {
                outside
            }
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

// ---- rescale --------------------------------------------------------------

/// `RescaleIntensityImageFilter`: linearly remap the actual `[min, max]` of the
/// image onto `[output_min, output_max]`. Output pixel type follows input.
pub fn rescale_intensity(img: &Image, output_min: f64, output_max: f64) -> Result<Image> {
    let vals = img.to_f64_vec();
    if vals.is_empty() {
        return Err(FilterError::DegenerateRange);
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &vals {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo == hi {
        return Err(FilterError::DegenerateRange);
    }
    let scale = (output_max - output_min) / (hi - lo);
    let out: Vec<f64> = vals
        .iter()
        .map(|&v| (v - lo) * scale + output_min)
        .collect();
    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- reductions -----------------------------------------------------------

/// Result of [`statistics`], mirroring `StatisticsImageFilter`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Statistics {
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
    /// Sample variance (divisor `n - 1`), as in ITK's `StatisticsImageFilter`.
    pub variance: f64,
    pub sigma: f64,
    pub sum: f64,
}

/// `StatisticsImageFilter`: min / max / mean / variance / sigma / sum.
pub fn statistics(img: &Image) -> Result<Statistics> {
    let vals = img.to_f64_vec();
    let n = vals.len();
    if n == 0 {
        return Err(FilterError::DegenerateRange);
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &v in &vals {
        minimum = minimum.min(v);
        maximum = maximum.max(v);
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n as f64;
    let variance = if n > 1 {
        (sum_sq - (n as f64) * mean * mean) / ((n - 1) as f64)
    } else {
        0.0
    };
    Ok(Statistics {
        minimum,
        maximum,
        mean,
        variance,
        sigma: variance.max(0.0).sqrt(),
        sum,
    })
}

/// `MinimumMaximumImageFilter`: the `(minimum, maximum)` pixel values.
pub fn minimum_maximum(img: &Image) -> Result<(f64, f64)> {
    let s = statistics(img)?;
    Ok((s.minimum, s.maximum))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn cast_u8_to_f32() {
        let a = img_u8(&[2, 2], vec![0, 1, 2, 255]);
        let b = cast(&a, PixelId::Float32).unwrap();
        assert_eq!(b.pixel_id(), PixelId::Float32);
        assert_eq!(b.scalar_slice::<f32>().unwrap(), &[0.0, 1.0, 2.0, 255.0]);
    }

    #[test]
    fn cast_preserves_geometry() {
        let mut a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        a.set_spacing(&[0.5, 2.0]).unwrap();
        a.set_origin(&[3.0, -1.0]).unwrap();
        let b = cast(&a, PixelId::Int16).unwrap();
        assert_eq!(b.spacing(), a.spacing());
        assert_eq!(b.origin(), a.origin());
    }

    #[test]
    fn add_images() {
        let a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        let b = img_u8(&[2, 2], vec![10, 20, 30, 40]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.scalar_slice::<u8>().unwrap(), &[11, 22, 33, 44]);
    }

    #[test]
    fn add_wraps_on_u8_overflow_like_itk() {
        // ITK's uint8 Add functor: static_cast<uint8>(250 + 10) = 4 (2's-complement
        // wraparound). We compute in the pixel type with wrapping_add to match.
        let a = img_u8(&[1, 1], vec![250]);
        let b = img_u8(&[1, 1], vec![10]);
        assert_eq!(add(&a, &b).unwrap().scalar_slice::<u8>().unwrap(), &[4]);
    }

    #[test]
    fn subtract_multiply_divide() {
        let a = Image::from_vec(&[2, 2], vec![10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let b = Image::from_vec(&[2, 2], vec![2.0f32, 4.0, 0.0, 8.0]).unwrap();
        assert_eq!(
            subtract(&a, &b).unwrap().scalar_slice::<f32>().unwrap(),
            &[8.0, 16.0, 30.0, 32.0]
        );
        assert_eq!(
            multiply(&a, &b).unwrap().scalar_slice::<f32>().unwrap(),
            &[20.0, 80.0, 0.0, 320.0]
        );
        // ITK Div functor: divide by zero yields NumericTraits<T>::max().
        assert_eq!(
            divide(&a, &b).unwrap().scalar_slice::<f32>().unwrap(),
            &[5.0, 5.0, f32::MAX, 5.0]
        );
    }

    #[test]
    fn integer_divide_by_zero_is_type_max() {
        let a = Image::from_vec(&[2, 1], vec![10i32, 20]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![0i32, 5]).unwrap();
        assert_eq!(
            divide(&a, &b).unwrap().scalar_slice::<i32>().unwrap(),
            &[i32::MAX, 4]
        );
    }

    #[test]
    fn multiply_wraps_on_u8_overflow_like_itk() {
        // static_cast<uint8>(16 * 16) = static_cast<uint8>(256) = 0.
        let a = img_u8(&[1, 1], vec![16]);
        let b = img_u8(&[1, 1], vec![16]);
        assert_eq!(
            multiply(&a, &b).unwrap().scalar_slice::<u8>().unwrap(),
            &[0]
        );
    }

    #[test]
    fn mismatched_inputs_error() {
        let a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        let b = Image::from_vec(&[2, 2], vec![1.0f32; 4]).unwrap();
        assert!(matches!(add(&a, &b), Err(FilterError::TypeMismatch { .. })));
        let c = img_u8(&[4, 1], vec![1, 2, 3, 4]);
        assert!(matches!(add(&a, &c), Err(FilterError::SizeMismatch { .. })));
    }

    #[test]
    fn constant_ops() {
        let a = img_u8(&[2, 1], vec![5, 10]);
        assert_eq!(
            add_constant(&a, 3.0).unwrap().scalar_slice::<u8>().unwrap(),
            &[8, 13]
        );
        assert_eq!(
            multiply_constant(&a, 2.0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[10, 20]
        );
    }

    #[test]
    fn abs_negative_values() {
        let a = Image::from_vec(&[3, 1], vec![-3i16, 0, 7]).unwrap();
        assert_eq!(abs(&a).unwrap().scalar_slice::<i16>().unwrap(), &[3, 0, 7]);
    }

    #[test]
    fn binary_threshold_inside_outside() {
        let a = Image::from_vec(&[5, 1], vec![0.0f32, 5.0, 10.0, 15.0, 20.0]).unwrap();
        let t = binary_threshold(&a, 5.0, 15.0, 1, 0).unwrap();
        assert_eq!(t.pixel_id(), PixelId::UInt8);
        assert_eq!(t.scalar_slice::<u8>().unwrap(), &[0, 1, 1, 1, 0]);
    }

    #[test]
    fn rescale_to_0_255() {
        let a = Image::from_vec(&[3, 1], vec![10.0f32, 20.0, 30.0]).unwrap();
        let r = rescale_intensity(&a, 0.0, 255.0).unwrap();
        assert_eq!(r.scalar_slice::<f32>().unwrap(), &[0.0, 127.5, 255.0]);
    }

    #[test]
    fn statistics_values() {
        let a = Image::from_vec(&[4, 1], vec![2.0f64, 4.0, 4.0, 6.0]).unwrap();
        let s = statistics(&a).unwrap();
        assert_eq!(s.minimum, 2.0);
        assert_eq!(s.maximum, 6.0);
        assert_eq!(s.mean, 4.0);
        assert_eq!(s.sum, 16.0);
        // sample variance: ((2-4)^2+(4-4)^2+(4-4)^2+(6-4)^2)/(4-1) = 8/3.
        assert!((s.variance - 8.0 / 3.0).abs() < 1e-12);
        assert!((s.sigma - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn minimum_maximum_pair() {
        let a = Image::from_vec(&[3, 1], vec![-1i32, 5, 3]).unwrap();
        assert_eq!(minimum_maximum(&a).unwrap(), (-1.0, 5.0));
    }
}
