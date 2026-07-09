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
//! Both policies are the two faces of the [`functor`] module's pixel-functor
//! seam (ITK's `UnaryFunctorImageFilter` / `BinaryFunctorImageFilter`), which
//! `add`/`subtract`/`multiply`/`divide`, the `*_constant` ops, and `abs` are
//! built on.
//!
//! The struct-style filter API and the remaining ~290 filters arrive with the
//! yaml codegen in a later phase.

pub mod anisotropic_diffusion;
pub mod binary_morphology;
pub mod canny;
pub mod convolution;
pub mod denoise;
pub mod distance;
pub mod error;
pub mod fast_marching;
mod fft;
pub mod functor;
pub mod geometry;
pub mod gradient;
pub mod grid_utility;
mod histogram;
pub mod histogram_matching;
pub mod intensity;
pub mod label;
pub mod label_shape;
pub mod level_set;
pub mod logic;
pub mod math;
pub mod morphology;
pub mod noise;
pub mod projection;
mod random;
pub mod reconstruction;
pub mod recursive_gaussian;
pub mod region_growing;
pub mod sharpening;
pub mod shrink;
pub mod smoothing;
pub mod threshold;
pub mod watershed;

pub use anisotropic_diffusion::{
    curvature_anisotropic_diffusion, gradient_anisotropic_diffusion, stable_time_step_bound,
};
pub use binary_morphology::{
    binary_fillhole, binary_grind_peak, binary_thinning, voting_binary,
    voting_binary_iterative_hole_filling,
};
pub use canny::{canny_edge_detection, zero_crossing};
pub use convolution::{
    ConvolutionBoundaryCondition, OutputRegionMode, convolution, fft_convolution,
};
pub use denoise::{bilateral, binomial_blur, curvature_flow, discrete_gaussian, mean, median};
pub use distance::{
    danielsson_distance_map, signed_danielsson_distance_map, signed_maurer_distance_map,
};
pub use error::{FilterError, Result};
pub use fast_marching::fast_marching;
pub use functor::{BinaryFunctor, UnaryFunctor};
pub use geometry::{
    constant_pad, crop, extract, flip, mirror_pad, permute_axes, region_of_interest, wrap_pad,
};
pub use gradient::{
    derivative, gradient_magnitude, gradient_magnitude_recursive_gaussian, laplacian,
    laplacian_recursive_gaussian, sobel_edge_detection,
};
pub use grid_utility::{checker_board, paste, tile};
pub use histogram_matching::histogram_matching;
pub use intensity::{
    intensity_windowing, intensity_windowing_in_place, invert_intensity, invert_intensity_in_place,
    normalize, otsu_multiple_thresholds, otsu_threshold, sigmoid, sigmoid_in_place,
    triangle_threshold,
};
pub use label::{LabelStatistics, connected_component, label_statistics, relabel_component};
pub use label_shape::{
    BoundingBox, LabelShapeStatisticsSettings, OrientedBoundingBox, ShapeStatistics,
    label_shape_statistics,
};
pub use level_set::{LevelSetResult, geodesic_active_contour_level_set, shape_detection_level_set};
pub use logic::{
    and, and_in_place, mask, mask_in_place, mask_negated, mask_negated_in_place, maximum,
    maximum_in_place, minimum, minimum_in_place, not, not_in_place, or, or_in_place, xor,
    xor_in_place,
};
pub use math::{
    abs, abs_in_place, acos, acos_in_place, asin, asin_in_place, atan, atan_in_place,
    bounded_reciprocal, bounded_reciprocal_in_place, cos, cos_in_place, exp, exp_in_place,
    exp_negative, exp_negative_in_place, log, log_in_place, log10, log10_in_place, sin,
    sin_in_place, sqrt, sqrt_in_place, square, square_in_place, squared_difference,
    squared_difference_in_place, tan, tan_in_place,
};
pub use morphology::{
    StructuringElement, binary_dilate, binary_erode, binary_morphological_closing,
    binary_morphological_opening, black_top_hat, grayscale_dilate, grayscale_erode,
    grayscale_morphological_closing, grayscale_morphological_opening, white_top_hat,
};
pub use noise::{additive_gaussian_noise, salt_and_pepper_noise, shot_noise, speckle_noise};
pub use projection::{
    binary_projection, maximum_projection, mean_projection, median_projection, minimum_projection,
    standard_deviation_projection, sum_projection,
};
pub use reconstruction::{
    grayscale_fillhole, grayscale_grindpeak, h_concave, h_convex, h_maxima, h_minima,
    reconstruction_by_dilation, reconstruction_by_erosion,
};
pub use recursive_gaussian::{GaussianOrder, recursive_gaussian, recursive_gaussian_with_order};
pub use region_growing::{
    IsolatedConnectedResult, confidence_connected, connected_threshold, isolated_connected,
    neighborhood_connected,
};
pub use sharpening::{laplacian_sharpening, unsharp_mask};
pub use shrink::shrink;
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar};
pub use smoothing::smooth_gaussian;
pub use threshold::{
    huang_threshold, intermodes_threshold, isodata_threshold, kittler_illingworth_threshold,
    li_threshold, maximum_entropy_threshold, moments_threshold, renyi_entropy_threshold,
    shanbhag_threshold, threshold, yen_threshold,
};
pub use watershed::{morphological_watershed, morphological_watershed_from_markers};

// ---- image ⊕ image functor arithmetic -------------------------------------
//
// Add/Sub/Mul/Div are [`BinaryFunctor`] markers plugged into
// [`functor::binary_functor!`] below: pixel-type-compute (the seam's policy
// (b)), matching ITK's `itkArithmeticOpsFunctors.h` (`Add2`, `Sub2`, `Mult`,
// `Div` all evaluate their operator in the pixel type, then
// `static_cast<TOutput>`).

/// Pixel-wise `a + b`; matches ITK's `Add2` functor.
struct AddOp;
/// Pixel-wise `a - b`; matches ITK's `Sub2` functor.
struct SubOp;
/// Pixel-wise `a * b`; matches ITK's `Mult` functor.
struct MulOp;
/// Pixel-wise `a / b`, or the type's max value when `b == 0`; matches ITK's
/// `Div` functor (`NumericTraits<T>::max()` on division by zero).
struct DivOp;

macro_rules! impl_binary_functor_int {
    ($($t:ty),+ $(,)?) => {$(
        impl BinaryFunctor<$t> for AddOp {
            fn apply(&self, a: $t, b: $t) -> $t { a.wrapping_add(b) }
        }
        impl BinaryFunctor<$t> for SubOp {
            fn apply(&self, a: $t, b: $t) -> $t { a.wrapping_sub(b) }
        }
        impl BinaryFunctor<$t> for MulOp {
            fn apply(&self, a: $t, b: $t) -> $t { a.wrapping_mul(b) }
        }
        impl BinaryFunctor<$t> for DivOp {
            fn apply(&self, a: $t, b: $t) -> $t {
                if b == 0 { <$t>::MAX } else { a.wrapping_div(b) }
            }
        }
    )+};
}

macro_rules! impl_binary_functor_float {
    ($($t:ty),+ $(,)?) => {$(
        impl BinaryFunctor<$t> for AddOp {
            fn apply(&self, a: $t, b: $t) -> $t { a + b }
        }
        impl BinaryFunctor<$t> for SubOp {
            fn apply(&self, a: $t, b: $t) -> $t { a - b }
        }
        impl BinaryFunctor<$t> for MulOp {
            fn apply(&self, a: $t, b: $t) -> $t { a * b }
        }
        impl BinaryFunctor<$t> for DivOp {
            fn apply(&self, a: $t, b: $t) -> $t {
                if b == 0.0 { <$t>::MAX } else { a / b }
            }
        }
    )+};
}

impl_binary_functor_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl_binary_functor_float!(f32, f64);

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

fn quantize_to_pixel_type_impl<T: Scalar>(v: f64) -> f64 {
    T::from_f64(v).as_f64()
}

/// `static_cast<PixelType>(v)` narrowed back to `f64`: SimpleITK's `pixeltype:
/// Input` YAML members — `MorphologicalWatershedImageFilter::Level`,
/// `HMinimaImageFilter`/`HMaximaImageFilter`/`HConvexImageFilter`/
/// `HConcaveImageFilter::Height` — are all `InputImagePixelType`-typed at the
/// ITK level (each is set via `itkSetMacro(Level`/`Height, InputImagePixelType)`),
/// so the `double` SimpleITK exposes to callers is cast to the pixel type
/// before the underlying filter ever sees it.
pub(crate) fn quantize_to_pixel_type(target: PixelId, v: f64) -> f64 {
    dispatch_scalar!(target, quantize_to_pixel_type_impl, v)
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

functor::binary_functor! {
    /// `AddImageFilter`: pixel-wise `a + b` (`static_cast<T>(a + b)`; integers wrap).
    pub fn add, add_in_place = AddOp;
}

functor::binary_functor! {
    /// `SubtractImageFilter`: pixel-wise `a - b` (integers wrap).
    pub fn subtract, subtract_in_place = SubOp;
}

functor::binary_functor! {
    /// `MultiplyImageFilter`: pixel-wise `a * b` (integers wrap).
    pub fn multiply, multiply_in_place = MulOp;
}

functor::binary_functor! {
    /// `DivideImageFilter`: pixel-wise `a / b`; where `b == 0` yields the output
    /// type's largest finite value (`NumericTraits<T>::max()`), matching ITK's `Div`
    /// functor.
    pub fn divide, divide_in_place = DivOp;
}

// ---- binary arithmetic (image ⊕ constant) ---------------------------------
//
// The constant is SimpleITK's `double`, so unlike the image ⊕ image ops
// above, these accumulate in `f64` and narrow with a saturating cast: the
// seam's policy (a), matching ITK's math functors (`itkAcosImageFilter.h`)
// rather than its arithmetic functors. Each op is a [`UnaryFunctor`] closing
// over the runtime constant, plugged into [`functor::unary_functor!`].

/// `a + c` for a scalar constant.
struct AddConstant(f64);
impl UnaryFunctor for AddConstant {
    fn apply(&self, x: f64) -> f64 {
        x + self.0
    }
}

/// `a - c` for a scalar constant.
struct SubConstant(f64);
impl UnaryFunctor for SubConstant {
    fn apply(&self, x: f64) -> f64 {
        x - self.0
    }
}

/// `a * c` for a scalar constant.
struct MulConstant(f64);
impl UnaryFunctor for MulConstant {
    fn apply(&self, x: f64) -> f64 {
        x * self.0
    }
}

functor::unary_functor! {
    /// `a + c` for a scalar constant.
    pub fn add_constant, add_constant_in_place(c: f64) = AddConstant(c);
}

functor::unary_functor! {
    /// `a - c` for a scalar constant.
    pub fn subtract_constant, subtract_constant_in_place(c: f64) = SubConstant(c);
}

functor::unary_functor! {
    /// `a * c` for a scalar constant.
    pub fn multiply_constant, multiply_constant_in_place(c: f64) = MulConstant(c);
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
    fn add_constant_saturates_on_overflow_unlike_wrapping_add() {
        // Seam policy (a) (f64-compute, `add_constant`): 250 + 10.0 = 260.0,
        // which does not fit in u8, so `Scalar::from_f64` saturates to 255.
        // Contrast seam policy (b) (pixel-type-compute, `add`): the same
        // 250 + 10 wraps to 4 (see `add_wraps_on_u8_overflow_like_itk`).
        let a = img_u8(&[1, 1], vec![250]);
        assert_eq!(
            add_constant(&a, 10.0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[255]
        );
    }

    #[test]
    fn add_in_place_matches_allocating() {
        let a = img_u8(&[2, 2], vec![1, 2, 250, 4]);
        let b = img_u8(&[2, 2], vec![10, 20, 10, 40]);
        let allocated = add(&a, &b).unwrap();
        let in_place = add_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
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
