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
//! [`unary_minus`] is the unary face of the same pixel-type-compute policy
//! as `add`/`subtract`/`multiply`/`divide`/`modulus` (ITK's `UnaryMinus`
//! functor, `itkArithmeticOpsFunctors.h`): `static_cast<T>(-a)`, computed in
//! the pixel type with no `f64` promotion, so a signed integer's minimum
//! value wraps back to itself (`i8::MIN.wrapping_neg() == i8::MIN`) instead
//! of the C++ overflow being undefined behavior --- the same wraparound
//! policy `add`/`subtract`/`multiply`/`divide`/`modulus` already use.
//! `UnaryMinusImageFilter.yaml` restricts this to signed pixel types
//! (`Functor::UnaryMinus`'s doc comment: "Assumed that the output type is
//! signed"); this port checks [`PixelId::is_signed`] at runtime and returns
//! [`FilterError::RequiresSignedPixelType`] in place of the C++ compile-time
//! restriction.
//!
//! The struct-style filter API and the remaining ~290 filters arrive with the
//! yaml codegen in a later phase.

pub mod adaptive_histogram_equalization;
pub mod anisotropic_diffusion;
pub mod attribute_morphology;
pub mod binary_morphology;
pub mod bspline_decomposition;
pub mod canny;
pub mod chan_vese;
pub mod change_label;
pub mod clamp;
pub mod coherence_enhancing_diffusion;
pub mod colliding_fronts;
pub mod complex;
pub mod contour;
pub mod contour_extractor_2d;
pub mod convolution;
pub mod deconvolution;
pub mod demons;
pub mod denoise;
pub mod dicom_orient;
pub mod displacement_field;
pub mod distance;
pub mod edge;
pub mod error;
pub mod expand;
pub mod fast_marching;
pub mod fast_marching_base;
pub mod fast_marching_upwind_gradient;
mod fft;
pub mod fft_correlation;
pub mod fft_shift;
pub mod functor;
pub mod geodesic_morphology;
pub mod geometry;
pub mod gradient;
pub mod grid_utility;
mod histogram;
pub mod histogram_matching;
pub mod intensity;
pub mod join_series;
pub mod kmeans;
pub mod label;
pub mod label_fusion;
pub mod label_intensity;
pub mod label_map;
pub mod label_map_overlay;
pub mod label_set_morphology;
pub mod label_shape;
pub mod label_to_rgb;
pub mod level_set;
mod linalg;
pub mod logic;
pub mod math;
pub mod min_max_curvature_flow;
pub mod morphology;
pub mod morphology_reconstruction;
pub mod n4_bias_field;
pub mod noise;
pub mod noise_estimate;
pub mod object_morphology;
pub mod objectness;
pub mod overlap;
pub mod patch_based_denoising;
pub mod projection;
mod random;
pub mod rank;
pub mod reconstruction;
pub mod recursive_gaussian;
pub mod region_growing;
pub mod regional_extrema;
pub mod reinitialize_level_set;
pub mod scalar_connected_component;
pub mod scalar_to_rgb_colormap;
pub mod sharpening;
pub mod shrink;
pub mod slic;
pub mod slice;
pub mod smoothing;
pub mod sources;
pub mod stochastic_fractal_dimension;
pub mod threshold;
pub mod threshold_maximum_connected_components;
pub mod toboggan;
pub mod vector;
pub mod vector_connected_component;
pub mod watershed;
pub mod watershed_classic;

pub use adaptive_histogram_equalization::adaptive_histogram_equalization;
pub use anisotropic_diffusion::{
    curvature_anisotropic_diffusion, gradient_anisotropic_diffusion, stable_time_step_bound,
};
pub use attribute_morphology::{area_closing, area_opening};
pub use binary_morphology::{
    binary_fillhole, binary_grind_peak, binary_median, binary_thinning, voting_binary,
    voting_binary_iterative_hole_filling,
};
pub use bspline_decomposition::{bspline_decomposition, bspline_spline_poles};
pub use canny::{canny_edge_detection, zero_crossing};
pub use chan_vese::{
    ChanAndVeseParams, ChanAndVeseResult, HeavisideStepFunction,
    scalar_chan_and_vese_dense_level_set,
};
pub use change_label::change_label;
pub use clamp::clamp;
pub use coherence_enhancing_diffusion::{
    CoherenceEnhancingDiffusionSettings, Enhancement, coherence_enhancing_diffusion,
};
pub use colliding_fronts::colliding_fronts;
pub use contour::{binary_contour, binary_pruning, label_contour, simple_contour_extractor};
pub use contour_extractor_2d::{Contour, contour_extractor_2d};
pub use convolution::{
    ConvolutionBoundaryCondition, OutputRegionMode, convolution, fft_convolution,
};
pub use demons::{
    DemonsParams, DemonsResult, DiffeomorphicDemonsParams, EsmGradient,
    FastSymmetricForcesDemonsParams, LevelSetMotionParams, SymmetricForcesDemonsParams,
    demons_registration, diffeomorphic_demons_registration,
    fast_symmetric_forces_demons_registration, level_set_motion_registration,
    symmetric_forces_demons_registration,
};
pub use denoise::{
    bilateral, binomial_blur, box_mean, box_sigma, curvature_flow, discrete_gaussian,
    discrete_gaussian_derivative, mean, median,
};
pub use dicom_orient::{
    DEFAULT_ORIENTATION, DicomOrientResult, dicom_orient, get_direction_cosines_from_orientation,
    get_orientation_from_direction_cosines,
};
pub use displacement_field::{
    DisplacementFieldJacobianDeterminantSettings, InvertDisplacementFieldResult,
    InvertDisplacementFieldSettings, IterativeInverseDisplacementFieldSettings,
    displacement_field_jacobian_determinant, inverse_displacement_field, invert_displacement_field,
    iterative_inverse_displacement_field,
};
pub use distance::{
    approximate_signed_distance_map, danielsson_distance_map, iso_contour_distance,
    signed_danielsson_distance_map, signed_maurer_distance_map,
};
pub use edge::zero_crossing_based_edge_detection;
pub use error::{FilterError, Result};
pub use expand::{Interpolator, expand};
pub use fast_marching::fast_marching;
pub use fast_marching_upwind_gradient::{
    FastMarchingUpwindGradientResult, FastMarchingUpwindGradientSettings,
    fast_marching_upwind_gradient,
};
pub use fft_correlation::{fft_normalized_correlation, masked_fft_normalized_correlation};
pub use fft_shift::fft_shift;
pub use functor::{BinaryFunctor, ComparisonFunctor, UnaryFunctor, UnaryPixelFunctor};
pub use geodesic_morphology::{grayscale_geodesic_dilate, grayscale_geodesic_erode};
pub use geometry::{
    constant_pad, crop, extract, flip, mirror_pad, permute_axes, region_of_interest, wrap_pad,
    zero_flux_neumann_pad,
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
pub use kmeans::{KmeansResult, scalar_image_kmeans};
pub use label::{LabelStatistics, connected_component, label_statistics, relabel_component};
pub use label_shape::{
    BoundingBox, LabelShapeStatisticsSettings, OrientedBoundingBox, ShapeStatistics,
    label_shape_statistics,
};
pub use label_to_rgb::{label_overlay, label_to_rgb};
pub use level_set::{
    CannyLevelSetResult, LevelSetResult, anti_alias_binary, canny_segmentation_level_set,
    geodesic_active_contour_level_set, laplacian_segmentation_level_set, shape_detection_level_set,
    threshold_segmentation_level_set,
};
pub use logic::{
    and, and_in_place, binary_not, binary_not_in_place, bitwise_not, bitwise_not_in_place,
    greater_equal, less_equal, mask, mask_in_place, mask_negated, mask_negated_in_place,
    masked_assign, masked_assign_constant, masked_assign_constant_in_place, masked_assign_in_place,
    maximum, maximum_in_place, minimum, minimum_in_place, not, not_equal, not_in_place, or,
    or_in_place, xor, xor_in_place,
};
pub use math::{
    abs, abs_in_place, absolute_value_difference, absolute_value_difference_in_place, acos,
    acos_in_place, asin, asin_in_place, atan, atan_in_place, atan2, atan2_in_place,
    binary_magnitude, binary_magnitude_in_place, bounded_reciprocal, bounded_reciprocal_in_place,
    cos, cos_in_place, divide_floor, divide_floor_in_place, divide_real, exp, exp_in_place,
    exp_negative, exp_negative_in_place, log, log_in_place, log10, log10_in_place, nary_add,
    nary_maximum, pow, pow_in_place, sin, sin_in_place, sqrt, sqrt_in_place, square,
    square_in_place, squared_difference, squared_difference_in_place, tan, tan_in_place,
    ternary_add, ternary_add_in_place, ternary_magnitude, ternary_magnitude_in_place,
    ternary_magnitude_squared, ternary_magnitude_squared_in_place,
};
pub use min_max_curvature_flow::{binary_min_max_curvature_flow, min_max_curvature_flow};
pub use morphology::{
    StructuringElement, binary_dilate, binary_erode, binary_morphological_closing,
    binary_morphological_opening, black_top_hat, grayscale_dilate, grayscale_erode,
    grayscale_morphological_closing, grayscale_morphological_opening, morphological_gradient,
    white_top_hat,
};
pub use morphology_reconstruction::{
    binary_closing_by_reconstruction, binary_opening_by_reconstruction,
    binary_reconstruction_by_dilation, binary_reconstruction_by_erosion, closing_by_reconstruction,
    grayscale_connected_closing, grayscale_connected_opening, opening_by_reconstruction,
};
pub use n4_bias_field::{
    N4BiasFieldCorrectionResult, N4BiasFieldCorrectionSettings, n4_bias_field_correction,
    n4_bias_field_correction_with_log_bias_field,
};
pub use noise::{additive_gaussian_noise, salt_and_pepper_noise, shot_noise, speckle_noise};
pub use noise_estimate::noise;
pub use overlap::{
    DirectedHausdorffMeasures, HausdorffMeasures, LabelOverlapMeasures, OverlapMeasures,
    directed_hausdorff_distance, hausdorff_distance, label_overlap_measures, similarity_index,
};
pub use patch_based_denoising::{NoiseModel, PatchBasedDenoisingSettings, patch_based_denoising};
pub use projection::{
    binary_projection, binary_threshold_projection, maximum_projection, mean_projection,
    median_projection, minimum_projection, standard_deviation_projection, sum_projection,
};
pub use rank::fast_approximate_rank;
pub use reconstruction::{
    double_threshold, grayscale_fillhole, grayscale_grindpeak, h_concave, h_convex, h_maxima,
    h_minima, reconstruction_by_dilation, reconstruction_by_erosion,
};
pub use recursive_gaussian::{
    GaussianOrder, recursive_gaussian, recursive_gaussian_with_order, smoothing_recursive_gaussian,
};
pub use region_growing::{
    IsolatedConnectedResult, VectorConfidenceConnectedResult, confidence_connected,
    connected_threshold, isolated_connected, neighborhood_connected, vector_confidence_connected,
};
pub use regional_extrema::{
    ValuedRegionalExtremaResult, regional_maxima, regional_minima, valued_regional_maxima,
    valued_regional_minima,
};
pub use reinitialize_level_set::reinitialize_level_set;
pub use scalar_connected_component::scalar_connected_component;
pub use scalar_to_rgb_colormap::{Colormap, scalar_to_rgb_colormap};
pub use sharpening::{laplacian_sharpening, unsharp_mask};
pub use shrink::{bin_shrink, shrink};
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar};
pub use slic::{SlicResult, SlicSettings, slic};
pub use slice::slice;
pub use smoothing::smooth_gaussian;
pub use sources::{
    GaborSourceSettings, GaussianSourceSettings, GridSourceSettings, SourceGeometry, gabor_source,
    gaussian_source, grid_source, physical_point_source,
};
pub use threshold::{
    huang_threshold, intermodes_threshold, isodata_threshold, kittler_illingworth_threshold,
    li_threshold, maximum_entropy_threshold, moments_threshold, renyi_entropy_threshold,
    shanbhag_threshold, threshold, yen_threshold,
};
pub use threshold_maximum_connected_components::{
    ThresholdMaximumConnectedComponentsResult, threshold_maximum_connected_components,
};
pub use toboggan::toboggan;
pub use vector::{compose, edge_potential, vector_index_selection_cast, vector_magnitude};
pub use vector_connected_component::vector_connected_component;
pub use watershed::{morphological_watershed, morphological_watershed_from_markers};
pub use watershed_classic::{
    IsolatedWatershedResult, IsolatedWatershedSettings, WatershedTree, isolated_watershed,
    watershed,
};

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
///
/// The inverse of [`Image::to_f64_vec`], and scalar-only for the same reason:
/// `vals` is one element per pixel, and [`dispatch_scalar!`] would resolve a
/// non-scalar `target` to its *component* type, quietly producing a scalar
/// image of that component type — or, for a complex `target`, an `N`-element
/// buffer where `assemble` demands `2N`. Rejecting every non-scalar `target`
/// with the same [`sitk_core::Error::RequiresScalarPixelType`] the read side
/// raises keeps the pair symmetric: every scalar filter enters through
/// `to_f64_vec` and leaves through here, and neither end can be handed a
/// non-scalar image. The test is a whitelist on
/// [`PixelId::is_scalar`](sitk_core::PixelId::is_scalar), matching
/// `Image::require_scalar`.
pub(crate) fn image_from_f64(
    target: PixelId,
    size: &[usize],
    geom: &Image,
    vals: &[f64],
) -> Result<Image> {
    if !target.is_scalar() {
        return Err(sitk_core::Error::RequiresScalarPixelType(target).into());
    }
    dispatch_scalar!(target, build_from_f64, size, geom, vals)
}

/// Real-pixel-type mapping: `Float32` stays `Float32`, everything else maps
/// to `Float64`. **Diverges from ITK**:
/// `itk::NumericTraits<PixelType>::RealType` is `double` for every scalar
/// pixel type *including* `float` (itkNumericTraits.h:1349/1356), so the
/// upstream rule always resolves to `double`. Flipping `Float32 → Float64`
/// changes `fast_marching`/`anti_alias_binary` output pixel types — breaking;
/// tracked in the upstream-findings ledger §5.6.
///
/// SimpleITK yamls that declare `output_pixel_type: typename
/// itk::NumericTraits<typename InputImageType::PixelType>::RealType` — among
/// them `FastMarchingImageFilter` and `AntiAliasBinaryImageFilter` — resolve
/// their output pixel type through this rule.
/// A vector pixel type maps to the vector variant of its component's real type
/// (`NumericTraits<VariableLengthVector<T>>::RealType` is
/// `VariableLengthVector<NumericTraits<T>::RealType>`), so the projection never
/// silently drops a pixel type's multi-component-ness.
///
/// A complex pixel type maps to its *component's* real type, dropping the
/// complex-ness — `NumericTraits<std::complex<float>>::RealType` is
/// `std::complex<double>` upstream. No caller reaches that arm: every filter
/// routing through here declares a `pixel_types` list that excludes complex and
/// enters via `to_f64_vec`, which rejects a complex image at the scalar seam.
/// The arm nevertheless yields the right answer for the one place it *is* the
/// rule — `ComplexToReal`'s `output_pixel_type: InputImageType::PixelType::value_type`.
pub(crate) fn real_pixel_id(input: PixelId) -> PixelId {
    let real = match input.component_id() {
        PixelId::Float32 => PixelId::Float32,
        _ => PixelId::Float64,
    };
    if input.is_vector() {
        real.vector_id()
    } else {
        real
    }
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

/// `itk::NumericTraits<T>::max()` (`itkNumericTraits.h`'s
/// `itkNUMERIC_TRAITS_MIN_MAX_MACRO`, `std::numeric_limits<T>::max()` for
/// every basic type with no ITK override -- unlike `min()`, which floating
/// types override to the smallest *positive* normalized value; see
/// [`numeric_traits_min`]).
pub(crate) fn numeric_traits_max(id: PixelId) -> f64 {
    match id.component_id() {
        PixelId::UInt8 => u8::MAX as f64,
        PixelId::Int8 => i8::MAX as f64,
        PixelId::UInt16 => u16::MAX as f64,
        PixelId::Int16 => i16::MAX as f64,
        PixelId::UInt32 => u32::MAX as f64,
        PixelId::Int32 => i32::MAX as f64,
        PixelId::UInt64 => u64::MAX as f64,
        PixelId::Int64 => i64::MAX as f64,
        PixelId::Float32 => f32::MAX as f64,
        PixelId::Float64 => f64::MAX,
        _ => unreachable!("PixelId::component_id() always returns a scalar variant"),
    }
}

/// `itk::NumericTraits<T>::min()` (`itkNumericTraits.h`'s
/// `itkNUMERIC_TRAITS_MIN_MAX_MACRO`): `std::numeric_limits<T>::min()` for
/// every integer type (the most-negative representable value), but for
/// `float`/`double` ITK overrides it to `std::numeric_limits<T>::min()`'s
/// *own* meaning for floating types -- the smallest *positive* normalized
/// value (`FLT_MIN`/`DBL_MIN`), not the most negative representable value.
/// Rust's `f32::MIN_POSITIVE`/`f64::MIN_POSITIVE` are the exact equivalents.
pub(crate) fn numeric_traits_min(id: PixelId) -> f64 {
    match id.component_id() {
        PixelId::UInt8 => u8::MIN as f64,
        PixelId::Int8 => i8::MIN as f64,
        PixelId::UInt16 => u16::MIN as f64,
        PixelId::Int16 => i16::MIN as f64,
        PixelId::UInt32 => u32::MIN as f64,
        PixelId::Int32 => i32::MIN as f64,
        PixelId::UInt64 => u64::MIN as f64,
        PixelId::Int64 => i64::MIN as f64,
        PixelId::Float32 => f32::MIN_POSITIVE as f64,
        PixelId::Float64 => f64::MIN_POSITIVE,
        _ => unreachable!("PixelId::component_id() always returns a scalar variant"),
    }
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
    let vals = img.to_f64_vec()?;
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

/// `Modulus` functor (`itkArithmeticOpsFunctors.h`): `a % b`, or the type's
/// max value when `b == 0` (`NumericTraits<TOutput>::max(static_cast<
/// TOutput>(A))`, which for every scalar type this crate supports ignores
/// its argument and returns the type's plain maximum -- see
/// `itkNumericTraits.h`'s `ITK_NUMERIC_TRAITS_MIN_MAX` macro). Integer pixel
/// types only (`ModulusImageFilter.yaml`'s `pixel_types:
/// IntegerPixelIDTypeList`); floats have no `%` operator in C++.
///
/// C++'s `%` on a `TInput1::MIN % -1` overflows (undefined behavior,
/// typically a SIGFPE trap); this crate uses `wrapping_rem`, matching this
/// module's `AddOp`/`SubOp`/`MulOp`/`DivOp` policy of defining C++'s
/// undefined integer-overflow behavior as 2's-complement wraparound rather
/// than panicking (Rust's plain `%` panics on this same input in debug
/// builds).
struct ModOp;

macro_rules! impl_modulus_int {
    ($($t:ty),+ $(,)?) => {$(
        impl BinaryFunctor<$t> for ModOp {
            fn apply(&self, a: $t, b: $t) -> $t {
                if b == 0 { <$t>::MAX } else { a.wrapping_rem(b) }
            }
        }
    )+};
}

impl_modulus_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl BinaryFunctor<f32> for ModOp {
    fn apply(&self, _a: f32, _b: f32) -> f32 {
        unreachable!("gated to integer pixel types by logic::require_integer_pixel_type")
    }
}
impl BinaryFunctor<f64> for ModOp {
    fn apply(&self, _a: f64, _b: f64) -> f64 {
        unreachable!("gated to integer pixel types by logic::require_integer_pixel_type")
    }
}

/// `ModulusImageFilter`: pixel-wise `a % b`; where `b == 0` yields the
/// output type's largest value. Integer pixel types only; errors with
/// [`FilterError::RequiresIntegerPixelType`] on a floating-point image.
pub fn modulus(a: &Image, b: &Image) -> Result<Image> {
    logic::require_integer_pixel_type(a)?;
    functor::binary_apply(a, b, &ModOp)
}

/// In-place variant of [`modulus`]: reuses `a`'s buffer.
pub fn modulus_in_place(a: Image, b: &Image) -> Result<Image> {
    logic::require_integer_pixel_type(&a)?;
    functor::binary_apply_in_place(a, b, &ModOp)
}

// ---- unary arithmetic (image only) -----------------------------------------

/// `UnaryMinus` functor (`itkArithmeticOpsFunctors.h`): `-a`, computed
/// directly in the pixel type (see the module docs for the wraparound and
/// signed-only-pixel-type notes).
struct UnaryMinusOp;

macro_rules! impl_unary_minus_signed_int {
    ($($t:ty),+ $(,)?) => {$(
        impl UnaryPixelFunctor<$t> for UnaryMinusOp {
            fn apply(&self, x: $t) -> $t { x.wrapping_neg() }
        }
    )+};
}

macro_rules! impl_unary_minus_float {
    ($($t:ty),+ $(,)?) => {$(
        impl UnaryPixelFunctor<$t> for UnaryMinusOp {
            fn apply(&self, x: $t) -> $t { -x }
        }
    )+};
}

impl_unary_minus_signed_int!(i8, i16, i32, i64);
impl_unary_minus_float!(f32, f64);
impl UnaryPixelFunctor<u8> for UnaryMinusOp {
    fn apply(&self, _x: u8) -> u8 {
        unreachable!("gated to signed pixel types by require_signed_pixel_type")
    }
}
impl UnaryPixelFunctor<u16> for UnaryMinusOp {
    fn apply(&self, _x: u16) -> u16 {
        unreachable!("gated to signed pixel types by require_signed_pixel_type")
    }
}
impl UnaryPixelFunctor<u32> for UnaryMinusOp {
    fn apply(&self, _x: u32) -> u32 {
        unreachable!("gated to signed pixel types by require_signed_pixel_type")
    }
}
impl UnaryPixelFunctor<u64> for UnaryMinusOp {
    fn apply(&self, _x: u64) -> u64 {
        unreachable!("gated to signed pixel types by require_signed_pixel_type")
    }
}

fn require_signed_pixel_type(img: &Image) -> Result<()> {
    if !img.pixel_id().is_signed() {
        return Err(FilterError::RequiresSignedPixelType(img.pixel_id()));
    }
    Ok(())
}

/// `UnaryMinusImageFilter`: pixel-wise `-a`. Signed pixel types only (see
/// the module docs); errors with [`FilterError::RequiresSignedPixelType`] on
/// an unsigned image.
pub fn unary_minus(img: &Image) -> Result<Image> {
    require_signed_pixel_type(img)?;
    functor::unary_pixel_apply(img, &UnaryMinusOp)
}

/// In-place variant of [`unary_minus`]: reuses `img`'s buffer.
pub fn unary_minus_in_place(img: Image) -> Result<Image> {
    require_signed_pixel_type(&img)?;
    functor::unary_pixel_apply_in_place(img, &UnaryMinusOp)
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
    let vals = img.to_f64_vec()?;
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
    let vals = img.to_f64_vec()?;
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
    let vals = img.to_f64_vec()?;
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
    fn modulus_basic_and_zero_divisor() {
        let a = Image::from_vec(&[3, 1], vec![10i32, -7, 20]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![3i32, 3, 0]).unwrap();
        // 10 % 3 = 1; -7 % 3 = -1 (Rust `%` and C++ `%` both truncate toward
        // zero, sign follows the dividend); 20 % 0 -> i32::MAX.
        assert_eq!(
            modulus(&a, &b).unwrap().scalar_slice::<i32>().unwrap(),
            &[1, -1, i32::MAX]
        );
    }

    #[test]
    fn modulus_min_dividend_by_negative_one_does_not_panic() {
        // i32::MIN % -1 would overflow (and panic in debug Rust); ModOp uses
        // wrapping_rem, matching this crate's established policy of defining
        // C++'s undefined integer-overflow behavior as 2's-complement wrap.
        let a = Image::from_vec(&[1, 1], vec![i32::MIN]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![-1i32]).unwrap();
        assert_eq!(
            modulus(&a, &b).unwrap().scalar_slice::<i32>().unwrap(),
            &[0]
        );
    }

    #[test]
    fn modulus_rejects_float_pixel_type() {
        let a = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        assert_eq!(
            modulus(&a, &b),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
    }

    #[test]
    fn modulus_in_place_matches_allocating() {
        let a = Image::from_vec(&[3, 1], vec![10i32, -7, 20]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![3i32, 3, 0]).unwrap();
        let allocated = modulus(&a, &b).unwrap();
        let in_place = modulus_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn unary_minus_basic_values() {
        let a = Image::from_vec(&[3, 1], vec![0i32, 5, -5]).unwrap();
        assert_eq!(
            unary_minus(&a).unwrap().scalar_slice::<i32>().unwrap(),
            &[0, -5, 5]
        );
    }

    #[test]
    fn unary_minus_min_value_wraps_to_itself() {
        // -(i8::MIN) = 128 overflows i8 (undefined behavior in C++); this
        // crate's wraparound policy gives i8::MIN.wrapping_neg() == i8::MIN.
        let a = Image::from_vec(&[1, 1], vec![i8::MIN]).unwrap();
        assert_eq!(
            unary_minus(&a).unwrap().scalar_slice::<i8>().unwrap(),
            &[i8::MIN]
        );
    }

    #[test]
    fn unary_minus_float() {
        let a = Image::from_vec(&[2, 1], vec![3.5f32, -2.0]).unwrap();
        assert_eq!(
            unary_minus(&a).unwrap().scalar_slice::<f32>().unwrap(),
            &[-3.5, 2.0]
        );
    }

    #[test]
    fn unary_minus_rejects_unsigned_pixel_type() {
        let a = img_u8(&[1, 1], vec![5]);
        assert_eq!(
            unary_minus(&a),
            Err(FilterError::RequiresSignedPixelType(a.pixel_id()))
        );
    }

    #[test]
    fn unary_minus_in_place_matches_allocating() {
        let a = Image::from_vec(&[3, 1], vec![0i32, 5, -5]).unwrap();
        let allocated = unary_minus(&a).unwrap();
        let in_place = unary_minus_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
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

/// The structural guard that keeps every scalar filter safe against a vector
/// image, proven at the seam rather than filter by filter.
///
/// No *scalar* filter in this crate carries a vector check of its own. Instead
/// the two helpers each scalar filter must use to touch pixel data —
/// [`Image::to_f64_vec`] on the way in and [`image_from_f64`] on the way out —
/// and the three `Image` accessors underneath them ([`Image::scalar_slice`],
/// [`Image::scalar_vec_mut`], [`Image::scalar_view`]) all refuse a vector
/// image with [`sitk_core::Error::RequiresScalarPixelType`]. `dispatch_scalar!`
/// resolves a vector `PixelId` to its component type, so a vector image reaches
/// the typed body and is rejected there, uniformly, with no panic path.
///
/// These cases sample the distinct routes a filter can take to the buffer; a
/// filter that used none of them could not read a pixel at all.
///
/// The vector-consuming filters — [`crate::vector`] and
/// [`crate::displacement_field`] — are the exception, and they say so in their
/// signatures: they reach the buffer through the component-aware accessors
/// ([`Image::component_slice`], [`Image::components_to_f64_vec`]) and check the
/// pixel type themselves, because for them a vector image is the *only* legal
/// input. [`crate::dicom_orient`] is a third case: it accepts *either* scalar
/// or vector, dispatching on [`sitk_core::PixelId::is_vector`] and, for a
/// vector image, decomposing into components with [`Image::extract_component`]
/// before ever reaching the scalar seam.
#[cfg(test)]
mod vector_guard {
    use super::*;
    use crate::morphology::StructuringElement;

    /// A 2x2 two-component vector image; every route below must reject it.
    fn vector_image() -> Image {
        Image::from_vec_vector(&[2, 2], 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()
    }

    fn assert_requires_scalar(err: FilterError, expected: PixelId) {
        match err {
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(id)) => {
                assert_eq!(id, expected)
            }
            other => panic!("expected RequiresScalarPixelType, got {other:?}"),
        }
    }

    /// Math op: `sqrt` reads through `scalar_slice` inside `unary_pixel_apply`.
    #[test]
    fn math_filter_rejects_a_vector_image() {
        let err = crate::math::sqrt(&vector_image()).unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat32);
    }

    /// Binary op: both operands are checked, so a scalar/vector mix is caught
    /// too — `require_same_shape` compares `pixel_id`, which differs.
    #[test]
    fn binary_math_filter_rejects_a_vector_operand() {
        let scalar = Image::new(&[2, 2], PixelId::Float32);
        assert!(crate::math::absolute_value_difference(&scalar, &vector_image()).is_err());
    }

    /// Morphology: `grayscale_dilate` reads through a `NeighborhoodIterator`,
    /// which takes a `ScalarView` at construction.
    #[test]
    fn morphology_filter_rejects_a_vector_image() {
        let kernel = StructuringElement::ball(&[1, 1]);
        let err = crate::morphology::grayscale_dilate(&vector_image(), &kernel).unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat32);
    }

    /// Level set: reads through `to_f64_vec`.
    #[test]
    fn level_set_filter_rejects_a_vector_image() {
        let scalar = Image::new(&[2, 2], PixelId::Float32);
        let err = crate::level_set::threshold_segmentation_level_set(
            &vector_image(),
            &scalar,
            0.0,
            1.0,
            0.02,
            1.0,
            1.0,
            1,
            false,
        )
        .unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat32);
    }

    /// Boundary conditions: `constant_pad` reaches pixels only through
    /// `Image::scalar_view`, so the `BoundaryCondition::get_pixel` loop cannot
    /// be entered with a vector image at all.
    #[test]
    fn pad_filter_rejects_a_vector_image() {
        let err =
            crate::geometry::constant_pad(&vector_image(), &[1, 1], &[1, 1], 0.0).unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat32);
    }

    /// A one-component vector image is still a vector image: SimpleITK keeps
    /// `sitkVectorFloat32` distinct from `sitkFloat32` regardless of length, so
    /// the guard must fire on component count 1 as well.
    #[test]
    fn one_component_vector_image_is_still_rejected() {
        let img = Image::from_vec_vector(&[2, 2], 1, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let err = crate::math::sqrt(&img).unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat32);
    }

    /// The write side of the seam: a vector `target` would otherwise dispatch
    /// to its component type and build a scalar image of `vals.len()` pixels.
    #[test]
    fn image_from_f64_rejects_a_vector_target() {
        let geom = Image::new(&[2, 2], PixelId::Float32);
        let err = image_from_f64(PixelId::VectorFloat64, &[2, 2], &geom, &[0.0; 4]).unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat64);
    }

    /// `patch_based_denoising` read its pixels through `Scalar::buffer_ref` on
    /// the raw `PixelBuffer`, which succeeds for a vector image because the
    /// buffer is tagged with the *component* type — the interleaved
    /// `npixels * ncomponents` elements were then processed against a grid of
    /// `npixels`. It reads through `Image::scalar_slice` now. The image is
    /// sized to clear the patch-fits-in-image check, so this pins the guard
    /// rather than an earlier validation error.
    #[test]
    fn patch_based_denoising_rejects_a_vector_image() {
        let data: Vec<f32> = (0..162).map(|i| i as f32).collect();
        let img = Image::from_vec_vector(&[9, 9], 2, data).unwrap();
        let err = crate::patch_based_denoising(&img, &Default::default()).unwrap_err();
        assert_requires_scalar(err, PixelId::VectorFloat32);
    }

    /// `real_pixel_id` preserves vector-ness, so a filter that projects its
    /// output type through it cannot launder a vector input into a scalar
    /// output pixel type.
    #[test]
    fn real_pixel_id_keeps_a_vector_input_vector() {
        assert_eq!(real_pixel_id(PixelId::VectorUInt8), PixelId::VectorFloat64);
        assert_eq!(
            real_pixel_id(PixelId::VectorFloat32),
            PixelId::VectorFloat32
        );
    }
}

/// The same structural guard, exercised against a **complex** image.
///
/// [`vector_guard`] proves that no scalar filter can read a vector image. That
/// proof rested on `Image::require_scalar` rejecting `is_vector()` — a
/// blacklist, which a complex image (a *basic* pixel type upstream, whose
/// buffer nonetheless holds two components per pixel) would have walked
/// straight through, handing a `2N`-long slice to a consumer that indexes it
/// per pixel. The guard is now a whitelist on
/// [`PixelId::is_scalar`](sitk_core::PixelId::is_scalar), and these cases pin
/// each route to the buffer against the category that used to bypass it.
///
/// The complex-consuming filters — [`crate::complex`] — are the exception, and
/// they say so in their signatures: they reach the buffer through
/// [`Image::complex_components`] and check the pixel type themselves.
#[cfg(test)]
mod complex_guard {
    use super::*;
    use crate::morphology::StructuringElement;

    /// A 2x2 `ComplexFloat32` image: 4 pixels, 8 buffer components.
    fn complex_image() -> Image {
        Image::new(&[2, 2], PixelId::ComplexFloat32)
    }

    fn assert_requires_scalar(err: FilterError, expected: PixelId) {
        match err {
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(id)) => {
                assert_eq!(id, expected)
            }
            other => panic!("expected RequiresScalarPixelType, got {other:?}"),
        }
    }

    /// Math op: `sqrt` reads through `scalar_slice` inside `unary_pixel_apply`.
    #[test]
    fn math_filter_rejects_a_complex_image() {
        let err = crate::math::sqrt(&complex_image()).unwrap_err();
        assert_requires_scalar(err, PixelId::ComplexFloat32);
    }

    /// Binary op: `require_same_shape` compares `pixel_id`, which differs.
    #[test]
    fn binary_math_filter_rejects_a_complex_operand() {
        let scalar = Image::new(&[2, 2], PixelId::Float32);
        assert!(crate::math::absolute_value_difference(&scalar, &complex_image()).is_err());
    }

    /// Morphology: reads through a `NeighborhoodIterator`, which takes a
    /// `ScalarView` at construction.
    #[test]
    fn morphology_filter_rejects_a_complex_image() {
        let kernel = StructuringElement::ball(&[1, 1]);
        let err = crate::morphology::grayscale_dilate(&complex_image(), &kernel).unwrap_err();
        assert_requires_scalar(err, PixelId::ComplexFloat32);
    }

    /// Level set: reads through `to_f64_vec`.
    #[test]
    fn level_set_filter_rejects_a_complex_image() {
        let scalar = Image::new(&[2, 2], PixelId::Float32);
        let err = crate::level_set::threshold_segmentation_level_set(
            &complex_image(),
            &scalar,
            0.0,
            1.0,
            0.02,
            1.0,
            1.0,
            1,
            false,
        )
        .unwrap_err();
        assert_requires_scalar(err, PixelId::ComplexFloat32);
    }

    /// Boundary conditions: `constant_pad` reaches pixels only through
    /// `Image::scalar_view`.
    #[test]
    fn pad_filter_rejects_a_complex_image() {
        let err =
            crate::geometry::constant_pad(&complex_image(), &[1, 1], &[1, 1], 0.0).unwrap_err();
        assert_requires_scalar(err, PixelId::ComplexFloat32);
    }

    /// `patch_based_denoising` reads through `Image::scalar_slice`. Sized to
    /// clear the patch-fits-in-image check, so this pins the guard rather than
    /// an earlier validation error.
    #[test]
    fn patch_based_denoising_rejects_a_complex_image() {
        let img = Image::new(&[9, 9], PixelId::ComplexFloat64);
        let err = crate::patch_based_denoising(&img, &Default::default()).unwrap_err();
        assert_requires_scalar(err, PixelId::ComplexFloat64);
    }

    /// The write side of the seam. A complex `target` dispatches to its
    /// component type, so `build_from_f64` would produce an `N`-element buffer
    /// where `assemble` demands `2N`; the guard rejects it at the seam instead,
    /// symmetric with the read side.
    #[test]
    fn image_from_f64_rejects_a_complex_target() {
        let geom = Image::new(&[2, 2], PixelId::Float32);
        let err = image_from_f64(PixelId::ComplexFloat32, &[2, 2], &geom, &[0.0; 4]).unwrap_err();
        assert_requires_scalar(err, PixelId::ComplexFloat32);
    }

    /// The complex-consuming filters are the exception, and only they.
    #[test]
    fn the_complex_filters_accept_a_complex_image() {
        assert!(crate::complex::complex_to_real(&complex_image()).is_ok());
        assert!(crate::complex::complex_to_modulus(&complex_image()).is_ok());
    }
}
