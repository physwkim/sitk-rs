//! Unary math and squared-difference filters for sitk-rs, verified against
//! the ITK v6 source under `Modules/Filtering/ImageIntensity/include/`
//! (`itkAbsImageFilter.h`, `itkAcosImageFilter.h`, `itkAsinImageFilter.h`,
//! `itkAtanImageFilter.h`, `itkCosImageFilter.h`, `itkSinImageFilter.h`,
//! `itkTanImageFilter.h`, `itkExpImageFilter.h`, `itkExpNegativeImageFilter.h`,
//! `itkLogImageFilter.h`, `itkLog10ImageFilter.h`, `itkSqrtImageFilter.h`,
//! `itkSquareImageFilter.h`, `itkBoundedReciprocalImageFilter.h`) and
//! `Modules/Filtering/ImageCompare/include/itkSquaredDifferenceImageFilter.h`.
//!
//! Every filter here except [`squared_difference`] is f64-compute
//! (`functor.rs`'s policy (a)): each ITK functor promotes its input to
//! `double`, applies a `<cmath>` function, and `static_cast`s back down
//! (`itkAcosImageFilter.h`'s `Acos`: `static_cast<TOutput>(std::acos(
//! static_cast<double>(A)))`), matching this crate's [`UnaryFunctor`] /
//! [`functor::unary_functor!`] seam exactly. Out-of-domain inputs (`sqrt` of
//! a negative, `log` of `0`) produce `NaN`/`-inf` in `f64`, which
//! [`sitk_core::Scalar::from_f64`] then narrows: integer outputs saturate
//! (`NaN` maps to `0`, `-inf` maps to the type's minimum), float outputs
//! keep the `NaN`/`-inf` exactly.
//!
//! [`squared_difference`] is also f64-compute but over *two* images
//! (`itkSquaredDifferenceImageFilter.h`'s `SquaredDifference2` functor:
//! `diff = double(A) - double(B); static_cast<TOutput>(diff * diff)`), which
//! is outside both `functor.rs` traits ([`UnaryFunctor`] takes one image;
//! [`BinaryFunctor`] computes in the pixel type rather than `f64`). Doing
//! the subtraction in the pixel type would be wrong for unsigned inputs
//! (`5u8 - 10u8` wraps instead of producing `-5`), so it's implemented
//! directly against `to_f64_vec`/`image_from_f64`, the same way
//! `rescale_intensity` and `statistics` in `lib.rs` handle multi-image or
//! reduction f64 math that falls outside the functor seam.

use crate::functor::{self, UnaryFunctor};
use crate::{Result, image_from_f64, require_same_shape};
use sitk_core::{Image, Scalar, dispatch_scalar};

// ---- abs --------------------------------------------------------------

/// `Abs` functor (`itkAbsImageFilter.h`): `static_cast<TOutput>(itk::Math::Absolute(A))`.
struct Abs;
impl UnaryFunctor for Abs {
    fn apply(&self, x: f64) -> f64 {
        x.abs()
    }
}

functor::unary_functor! {
    /// `AbsImageFilter`: pixel-wise absolute value, output type follows input.
    pub fn abs, abs_in_place() = Abs;
}

// ---- trigonometric ------------------------------------------------------

/// `Acos` functor (`itkAcosImageFilter.h`): `std::acos`.
struct Acos;
impl UnaryFunctor for Acos {
    fn apply(&self, x: f64) -> f64 {
        x.acos()
    }
}

functor::unary_functor! {
    /// `AcosImageFilter`: pixel-wise inverse cosine (`std::acos`); `NaN` outside `[-1, 1]`.
    pub fn acos, acos_in_place() = Acos;
}

/// `Asin` functor (`itkAsinImageFilter.h`): `std::asin`.
struct Asin;
impl UnaryFunctor for Asin {
    fn apply(&self, x: f64) -> f64 {
        x.asin()
    }
}

functor::unary_functor! {
    /// `AsinImageFilter`: pixel-wise inverse sine (`std::asin`); `NaN` outside `[-1, 1]`.
    pub fn asin, asin_in_place() = Asin;
}

/// `Atan` functor (`itkAtanImageFilter.h`): `std::atan`.
struct Atan;
impl UnaryFunctor for Atan {
    fn apply(&self, x: f64) -> f64 {
        x.atan()
    }
}

functor::unary_functor! {
    /// `AtanImageFilter`: pixel-wise one-argument inverse tangent (`std::atan`).
    pub fn atan, atan_in_place() = Atan;
}

/// `Cos` functor (`itkCosImageFilter.h`): `std::cos`.
struct Cos;
impl UnaryFunctor for Cos {
    fn apply(&self, x: f64) -> f64 {
        x.cos()
    }
}

functor::unary_functor! {
    /// `CosImageFilter`: pixel-wise cosine (`std::cos`).
    pub fn cos, cos_in_place() = Cos;
}

/// `Sin` functor (`itkSinImageFilter.h`): `std::sin`.
struct Sin;
impl UnaryFunctor for Sin {
    fn apply(&self, x: f64) -> f64 {
        x.sin()
    }
}

functor::unary_functor! {
    /// `SinImageFilter`: pixel-wise sine (`std::sin`).
    pub fn sin, sin_in_place() = Sin;
}

/// `Tan` functor (`itkTanImageFilter.h`): `std::tan`.
struct Tan;
impl UnaryFunctor for Tan {
    fn apply(&self, x: f64) -> f64 {
        x.tan()
    }
}

functor::unary_functor! {
    /// `TanImageFilter`: pixel-wise tangent (`std::tan`).
    pub fn tan, tan_in_place() = Tan;
}

// ---- exponential / logarithmic ------------------------------------------

/// `Exp` functor (`itkExpImageFilter.h`): `std::exp`.
struct Exp;
impl UnaryFunctor for Exp {
    fn apply(&self, x: f64) -> f64 {
        x.exp()
    }
}

functor::unary_functor! {
    /// `ExpImageFilter`: pixel-wise `e^x` (`std::exp`).
    pub fn exp, exp_in_place() = Exp;
}

/// `ExpNegative` functor (`itkExpNegativeImageFilter.h`): `std::exp(-K * x)`.
struct ExpNegative;
impl UnaryFunctor for ExpNegative {
    fn apply(&self, x: f64) -> f64 {
        (-x).exp()
    }
}

functor::unary_functor! {
    /// `ExpNegativeImageFilter`: pixel-wise `exp(-K * x)`, with `K` fixed at
    /// ITK's default of `1.0`. `itkExpNegativeImageFilter.h`'s `ExpNegative`
    /// functor takes a runtime `Factor`, but SimpleITK's generated wrapper
    /// (`ExpNegativeImageFilter.yaml`, `members: []`) never exposes it, so
    /// this port doesn't either.
    pub fn exp_negative, exp_negative_in_place() = ExpNegative;
}

/// `Log` functor (`itkLogImageFilter.h`): `std::log` (natural log).
struct Log;
impl UnaryFunctor for Log {
    fn apply(&self, x: f64) -> f64 {
        x.ln()
    }
}

functor::unary_functor! {
    /// `LogImageFilter`: pixel-wise natural log (`std::log`); `-inf` at `0`, `NaN` below `0`.
    pub fn log, log_in_place() = Log;
}

/// `Log10` functor (`itkLog10ImageFilter.h`): `std::log10`.
struct Log10;
impl UnaryFunctor for Log10 {
    fn apply(&self, x: f64) -> f64 {
        x.log10()
    }
}

functor::unary_functor! {
    /// `Log10ImageFilter`: pixel-wise base-10 log (`std::log10`); `-inf` at `0`, `NaN` below `0`.
    pub fn log10, log10_in_place() = Log10;
}

// ---- power / reciprocal ---------------------------------------------------

/// `Sqrt` functor (`itkSqrtImageFilter.h`): `std::sqrt`.
struct Sqrt;
impl UnaryFunctor for Sqrt {
    fn apply(&self, x: f64) -> f64 {
        x.sqrt()
    }
}

functor::unary_functor! {
    /// `SqrtImageFilter`: pixel-wise square root (`std::sqrt`); `NaN` below `0`.
    pub fn sqrt, sqrt_in_place() = Sqrt;
}

/// `Square` functor (`itkSquareImageFilter.h`): `ra = RealType(A); ra * ra`.
struct Square;
impl UnaryFunctor for Square {
    fn apply(&self, x: f64) -> f64 {
        x * x
    }
}

functor::unary_functor! {
    /// `SquareImageFilter`: pixel-wise `x^2`.
    pub fn square, square_in_place() = Square;
}

/// `BoundedReciprocal` functor (`itkBoundedReciprocalImageFilter.h`): `1.0 / (1.0 + x)`.
struct BoundedReciprocal;
impl UnaryFunctor for BoundedReciprocal {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + x)
    }
}

functor::unary_functor! {
    /// `BoundedReciprocalImageFilter`: pixel-wise `1 / (1 + x)`.
    pub fn bounded_reciprocal, bounded_reciprocal_in_place() = BoundedReciprocal;
}

// ---- squared difference (two images, f64-compute) --------------------------

/// `SquaredDifferenceImageFilter`: pixel-wise `(a - b)^2`, computed in `f64`
/// (`itkSquaredDifferenceImageFilter.h`'s `SquaredDifference2` functor). See
/// the module docs for why this bypasses the `BinaryFunctor` seam.
pub fn squared_difference(a: &Image, b: &Image) -> Result<Image> {
    require_same_shape(a, b)?;
    let va = a.to_f64_vec();
    let vb = b.to_f64_vec();
    let out: Vec<f64> = va
        .iter()
        .zip(&vb)
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .collect();
    image_from_f64(a.pixel_id(), a.size(), a, &out)
}

fn squared_difference_typed_in_place<T: Scalar>(img: &mut Image, other: &[f64]) -> Result<()> {
    let v = img.scalar_vec_mut::<T>()?;
    for (x, &y) in v.iter_mut().zip(other) {
        let diff = x.as_f64() - y;
        *x = T::from_f64(diff * diff);
    }
    Ok(())
}

/// In-place variant of [`squared_difference`]: reuses `a`'s buffer.
pub fn squared_difference_in_place(mut a: Image, b: &Image) -> Result<Image> {
    require_same_shape(&a, b)?;
    let vb = b.to_f64_vec();
    dispatch_scalar!(a.pixel_id(), squared_difference_typed_in_place, &mut a, &vb)?;
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- abs ----

    #[test]
    fn abs_negative_values() {
        let a = Image::from_vec(&[3, 1], vec![-3i16, 0, 7]).unwrap();
        assert_eq!(abs(&a).unwrap().scalar_slice::<i16>().unwrap(), &[3, 0, 7]);
    }

    #[test]
    fn abs_in_place_matches_allocating() {
        let a = Image::from_vec(&[3, 1], vec![-3i16, 0, 7]).unwrap();
        let allocated = abs(&a).unwrap();
        let in_place = abs_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- sqrt of a negative: NaN under Scalar::from_f64 ----

    #[test]
    fn sqrt_negative_saturates_to_zero_on_integer_output() {
        // f64::sqrt(-1.0) is NaN; Scalar::from_f64 for an integer type maps
        // NaN to 0 (Rust's `as` semantics), so this differs from C++
        // static_cast's undefined behavior by design (see pixel.rs docs).
        let a = img_u8(&[2, 1], vec![4, 0]);
        assert_eq!(sqrt(&a).unwrap().scalar_slice::<u8>().unwrap(), &[2, 0]);

        let neg = Image::from_vec(&[1, 1], vec![-1.0f32]).unwrap();
        let cast_to_u8 = crate::cast(&neg, sitk_core::PixelId::UInt8).unwrap();
        assert_eq!(
            sqrt(&cast_to_u8).unwrap().scalar_slice::<u8>().unwrap(),
            &[0]
        );
    }

    #[test]
    fn sqrt_negative_preserves_nan_on_float_output() {
        let a = Image::from_vec(&[1, 1], vec![-1.0f64]).unwrap();
        let out = sqrt(&a).unwrap();
        assert!(out.scalar_slice::<f64>().unwrap()[0].is_nan());
    }

    // ---- log(0) ----

    #[test]
    fn log_of_zero_is_negative_infinity_on_float_output() {
        let a = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        let out = log(&a).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap()[0], f64::NEG_INFINITY);
    }

    #[test]
    fn log_of_zero_saturates_to_type_min_on_integer_output() {
        // f64::ln(0.0) is -inf; Scalar::from_f64 for i16 maps -inf to
        // i16::MIN (Rust's saturating `as` semantics).
        let a = img_u8(&[1, 1], vec![0]);
        let cast_to_i16 = crate::cast(&a, sitk_core::PixelId::Int16).unwrap();
        assert_eq!(
            log(&cast_to_i16).unwrap().scalar_slice::<i16>().unwrap(),
            &[i16::MIN]
        );
    }

    // ---- bounded_reciprocal at 0 ----

    #[test]
    fn bounded_reciprocal_at_zero_is_one() {
        let a = Image::from_vec(&[2, 1], vec![0.0f64, 1.0]).unwrap();
        assert_eq!(
            bounded_reciprocal(&a)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap(),
            &[1.0, 0.5]
        );
    }

    // ---- square overflow on u8 ----

    #[test]
    fn square_overflow_saturates_on_u8() {
        // 250.0^2 = 62500.0, which does not fit in u8; Scalar::from_f64
        // saturates to 255 (contrast lib.rs's `multiply`, whose
        // pixel-type-compute policy wraps instead).
        let a = img_u8(&[2, 1], vec![250, 10]);
        assert_eq!(
            square(&a).unwrap().scalar_slice::<u8>().unwrap(),
            &[255, 100]
        );
    }

    #[test]
    fn square_in_place_matches_allocating() {
        let a = img_u8(&[2, 1], vec![250, 10]);
        let allocated = square(&a).unwrap();
        let in_place = square_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- other unary math: basic sanity ----

    #[test]
    fn trig_and_exp_basic_values() {
        let zero = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        assert_eq!(
            acos(&zero).unwrap().scalar_slice::<f64>().unwrap()[0],
            0.0f64.acos()
        );
        assert_eq!(asin(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(atan(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(cos(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 1.0);
        assert_eq!(sin(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(tan(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(exp(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 1.0);
        assert_eq!(
            exp_negative(&zero).unwrap().scalar_slice::<f64>().unwrap()[0],
            1.0
        );
        assert_eq!(
            log10(&zero).unwrap().scalar_slice::<f64>().unwrap()[0],
            f64::NEG_INFINITY
        );
    }

    // ---- squared_difference ----

    #[test]
    fn squared_difference_basic() {
        let a = img_u8(&[3, 1], vec![5, 0, 250]);
        let b = img_u8(&[3, 1], vec![10, 0, 240]);
        // (5-10)^2=25, (0-0)^2=0, (250-240)^2=100 -- all fit u8, no wrap
        // (unlike a pixel-type-compute `5u8 - 10u8` which would wrap first).
        assert_eq!(
            squared_difference(&a, &b)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[25, 0, 100]
        );
    }

    #[test]
    fn squared_difference_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![5, 0, 250]);
        let b = img_u8(&[3, 1], vec![10, 0, 240]);
        let allocated = squared_difference(&a, &b).unwrap();
        let in_place = squared_difference_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn squared_difference_mismatched_inputs_error() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        assert!(matches!(
            squared_difference(&a, &b),
            Err(crate::FilterError::TypeMismatch { .. })
        ));
    }
}
