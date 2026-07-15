//! The `SparseFieldLevelSetImageFilter` family: the five segmentation filters
//! `GeodesicActiveContourLevelSetImageFilter`,
//! `ShapeDetectionLevelSetImageFilter`,
//! `ThresholdSegmentationLevelSetImageFilter`,
//! `LaplacianSegmentationLevelSetImageFilter` and
//! `CannySegmentationLevelSetImageFilter`, plus
//! [`AntiAliasBinaryImageFilter`](anti_alias_binary), which drives the same
//! solver with a curvature-flow function and a per-pixel constraint.
//!
//! Ported from ITK's `Modules/Segmentation/LevelSets`,
//! `Modules/Filtering/AntiAlias` and `Modules/Core/FiniteDifference`:
//!
//! | Layer | ITK source |
//! |---|---|
//! | [`grid`] | `itkSparseFieldLevelSetImageFilter.hxx` (`SparseFieldCityBlockNeighborList`) |
//! | [`function`] | `itkLevelSetFunction.h/.hxx`, `itkSegmentationLevelSetFunction.h/.hxx`, `itkCurvatureFlowFunction.h/.hxx` |
//! | [`sparse_field`] | `itkSparseFieldLevelSetImageFilter.h/.hxx`, `itkFiniteDifferenceImageFilter.hxx` |
//! | [`anti_alias`] | `itkAntiAliasBinaryImageFilter.h/.hxx` |
//! | this module | `itkSegmentationLevelSetImageFilter.h/.hxx` plus the five `itk*LevelSetFunction.h/.hxx` + `itk*LevelSetImageFilter.h/.hxx` pairs |
//!
//! The rest of this doc describes the five *segmentation* filters;
//! [`anti_alias_binary`] documents its own inputs and sign convention.
//!
//! Every filter takes an **initial level set** — a real image whose
//! `iso_surface_value` contour is the starting front, negative inside — and a
//! **feature image**, from which that filter's `CalculateSpeedImage` derives the
//! speed. The output is the evolved level set: negative inside the segmented
//! region, positive outside, with the zero crossing on the front. (ITK's own
//! `ThresholdSegmentationLevelSetImageFilter` header, copied into SimpleITK's
//! yaml, states the opposite sign convention; the code follows the convention
//! stated here, so the doc is simply wrong.)
//!
//! For [`geodesic_active_contour_level_set`] and [`shape_detection_level_set`]
//! the feature image is an edge potential map (near `0` on edges, near `1`
//! inside homogeneous regions), typically `1 / (1 + |grad(G * I)|)`
//! ([`bounded_reciprocal`](crate::filters::bounded_reciprocal) of
//! [`gradient_magnitude_recursive_gaussian`](crate::filters::gradient_magnitude_recursive_gaussian)),
//! and the speed image is a verbatim copy of it. For the other three the feature
//! image is the raw image to segment.
//!
//! SimpleITK's yamls declare all five for `RealPixelIDTypeList`; ITK's
//! `TOutputPixelType` template parameter defaults to `float`, so the output is
//! always [`PixelId::Float32`] regardless of input type.
//!
//! ## Upstream behaviour reproduced here
//!
//! * **The functions' `Initialize` weights never survive.** Each
//!   `SegmentationLevelSetFunction` subclass's `Initialize(radius)` installs its
//!   own default weights — `ThresholdSegmentationLevelSetFunction` and
//!   `LaplacianSegmentationLevelSetFunction` set the propagation weight to
//!   `-1`, `CannySegmentationLevelSetFunction` sets both the propagation and
//!   advection weights to `-1`. But `Initialize` runs from
//!   `SetSegmentationFunction`, i.e. in the filter's constructor
//!   (itkSegmentationLevelSetImageFilter.h:457-467), and SimpleITK's generated
//!   `Execute` then calls `SetPropagationScaling`/`SetCurvatureScaling`/
//!   `SetAdvectionScaling` unconditionally. So the exposed scalings *are* the
//!   weights, and these ports take them directly.
//!
//! * **`CurvatureSpeed` differs between the filters.** Only the geodesic active
//!   contour and shape detection functions override it to return
//!   `PropagationSpeed`; the threshold, Laplacian and Canny functions inherit
//!   `LevelSetFunction::CurvatureSpeed`'s constant `1`, so their curvature term
//!   is not damped by the speed image.
//!
//! * **`ThresholdSegmentationLevelSetFunction`'s `EdgeWeight` branch is dead.**
//!   `CalculateSpeedImage` adds `m_EdgeWeight * Laplacian(GradientAnisotropicDiffusion(feature))`
//!   when `m_EdgeWeight != 0`, controlled by `SmoothingIterations` (5),
//!   `SmoothingConductance` (0.8) and `SmoothingTimeStep` (0.1). SimpleITK's
//!   `ThresholdSegmentationLevelSetImageFilter.yaml` exposes none of the four,
//!   and ITK's constructor sets `m_EdgeWeight = 0`, so the branch is
//!   unreachable through the SimpleITK API and is not ported.
//!
//! ## Where this port is deliberately better defined than ITK
//!
//! ITK generates the speed image only when the propagation weight is non-zero
//! (`SegmentationLevelSetImageFilter::GenerateData`). For
//! `CannySegmentationLevelSetImageFilter` with `propagation_scaling == 0` and
//! `advection_scaling != 0` that leaves `CalculateAdvectionImage` computing its
//! distance map into the *unallocated* speed image's requested region
//! (itkCannySegmentationLevelSetFunction.hxx:95), producing an empty advection
//! field. This port always builds the speed image first, so the advection field
//! is well defined at `propagation_scaling == 0`.

mod anti_alias;
mod function;
mod grid;
mod sparse_field;

pub use anti_alias::anti_alias_binary;

use crate::core::{Image, PixelId};
use crate::filters::canny::{canny_edge_detection, zero_crossing_values};
use crate::filters::distance::danielsson_distance_map;
use crate::filters::error::{FilterError, Result};
use crate::filters::geometry::require_same_physical_space;
use crate::filters::gradient::laplacian;
use crate::filters::image_from_f64;
use crate::filters::recursive_gaussian::{GaussianOrder, recursive_gaussian_f64};
use function::{CurvatureSpeed, DifferenceFunction, LevelSetFunction};
use grid::Grid;
use sparse_field::{SolverSetup, SparseFieldSolver, UpdateRule};

/// `GeodesicActiveContourLevelSetFunction::m_DerivativeSigma`, the sigma of the
/// Gaussian whose gradient forms the advection field. ITK's default is `1.0`
/// and SimpleITK does not expose `SetDerivativeSigma`, so it is a constant
/// here.
const DERIVATIVE_SIGMA: f64 = 1.0;

/// `CannySegmentationLevelSetFunction::CalculateDistanceImage` (hxx:93) sets
/// the Canny detector's `MaximumError` to this on every axis; SimpleITK exposes
/// only `Threshold` and `Variance`.
const CANNY_MAXIMUM_ERROR: f64 = 0.01;

/// `CannyEdgeDetectionImageFilter`'s constructor leaves `m_LowerThreshold` at
/// zero and the Canny level-set function never sets it.
const CANNY_LOWER_THRESHOLD: f64 = 0.0;

/// The evolved level set together with the two measurements SimpleITK reports:
/// `GetElapsedIterations()` and `GetRMSChange()`.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelSetResult {
    /// The output level set. The five segmentation filters produce a
    /// [`PixelId::Float32`] image, negative inside the segmented region;
    /// [`anti_alias_binary`] produces the input's real type with the opposite
    /// sign convention.
    pub image: Image,
    /// Number of iterations actually run before `Halt()` returned true.
    pub elapsed_iterations: u32,
    /// The RMS change of the final iteration.
    pub rms_change: f64,
}

/// `GeodesicActiveContourLevelSetImageFilter`: propagation, curvature and
/// advection.
///
/// The update is
///
/// ```text
/// phi_t = -beta g(x) |grad(phi)| - alpha A(x)·grad(phi) + gamma g(x) kappa |grad(phi)|
/// ```
///
/// with `g` the feature (edge potential) image, `A = -grad(G_1.0 * g)` the
/// advection field, and `beta`/`alpha`/`gamma` the `propagation_scaling`,
/// `advection_scaling` and `curvature_scaling`. The advection term behaves like
/// a doublet that attracts the contour onto the edge, so — unlike
/// [`shape_detection_level_set`] — the initial contour may overlap the shape
/// boundary.
///
/// `propagation_scaling` switches the balloon force outwards (positive) or
/// inwards (negative); `reverse_expansion_direction` instead flips the sign of
/// *both* the propagation and advection weights, so negative feature values
/// expand the surface.
///
/// Iteration stops after `number_of_iterations`, or as soon as an iteration's
/// RMS change falls strictly below `maximum_rms_error`. Argument order follows
/// SimpleITK's `GeodesicActiveContourLevelSetImageFilter.yaml`; its defaults
/// are `maximum_rms_error = 0.01`, all three scalings `1.0`,
/// `number_of_iterations = 1000`, `reverse_expansion_direction = false`.
///
/// Errors if the two images differ in size, or if the recursive Gaussian
/// behind the advection field cannot run (any axis shorter than four pixels).
#[allow(clippy::too_many_arguments)]
pub fn geodesic_active_contour_level_set(
    initial_level_set: &Image,
    feature_image: &Image,
    maximum_rms_error: f64,
    propagation_scaling: f64,
    curvature_scaling: f64,
    advection_scaling: f64,
    number_of_iterations: u32,
    reverse_expansion_direction: bool,
) -> Result<LevelSetResult> {
    check_same_size(initial_level_set, feature_image)?;
    let advection = if advection_scaling == 0.0 {
        Vec::new()
    } else {
        advection_field(feature_image)?
    };
    solve(
        initial_level_set,
        Solve {
            speed: feature_image.to_f64_vec()?,
            advection,
            curvature_speed: CurvatureSpeed::Propagation,
            weights: Weights {
                propagation: propagation_scaling,
                curvature: curvature_scaling,
                advection: advection_scaling,
            },
            iso_surface_value: 0.0,
            maximum_rms_error,
            number_of_iterations,
            reverse_expansion_direction,
        },
    )
}

/// `ShapeDetectionLevelSetImageFilter`: propagation and curvature, no
/// advection.
///
/// `ShapeDetectionLevelSetFunction::Initialize` pins the advection weight to
/// zero, so the update is
///
/// ```text
/// phi_t = -beta g(x) |grad(phi)| + gamma g(x) kappa |grad(phi)|
/// ```
///
/// Without the advection doublet the front only stalls where the edge potential
/// `g` vanishes, so the initial contour must lie wholly inside (or wholly
/// outside) the structure to be segmented. Larger `curvature_scaling` gives a
/// smoother contour; it should be non-negative.
///
/// Argument order follows SimpleITK's
/// `ShapeDetectionLevelSetImageFilter.yaml`; its defaults are
/// `maximum_rms_error = 0.02`, both scalings `1.0`,
/// `number_of_iterations = 1000`, `reverse_expansion_direction = false`.
///
/// Errors if the two images differ in size.
pub fn shape_detection_level_set(
    initial_level_set: &Image,
    feature_image: &Image,
    maximum_rms_error: f64,
    propagation_scaling: f64,
    curvature_scaling: f64,
    number_of_iterations: u32,
    reverse_expansion_direction: bool,
) -> Result<LevelSetResult> {
    check_same_size(initial_level_set, feature_image)?;
    solve(
        initial_level_set,
        Solve {
            speed: feature_image.to_f64_vec()?,
            advection: Vec::new(),
            curvature_speed: CurvatureSpeed::Propagation,
            weights: Weights {
                propagation: propagation_scaling,
                curvature: curvature_scaling,
                advection: 0.0,
            },
            iso_surface_value: 0.0,
            maximum_rms_error,
            number_of_iterations,
            reverse_expansion_direction,
        },
    )
}

/// `ThresholdSegmentationLevelSetImageFilter`: the speed is the feature
/// image's distance to the nearer edge of the intensity window
/// `[lower_threshold, upper_threshold]`, positive inside the window and
/// negative outside, so the front locks onto the window's boundary.
///
/// `ThresholdSegmentationLevelSetFunction::CalculateSpeedImage` (hxx:58-83)
/// splits at the window's midpoint `mid = (U - L) / 2 + L`:
///
/// ```text
/// speed(x) = g(x) - L   if g(x) < mid
///          = U - g(x)   otherwise
/// ```
///
/// The comparison is strict, so a feature value exactly at `mid` takes the
/// upper branch. `lower_threshold > upper_threshold` is not rejected by ITK
/// and is not rejected here; it simply makes the speed negative everywhere.
///
/// The update is `phi_t = -beta P(x) |grad(phi)| + gamma kappa |grad(phi)|`:
/// there is no advection term (`Initialize` pins the advection weight to zero)
/// and, unlike [`geodesic_active_contour_level_set`], the curvature term is
/// *not* modulated by the speed — `SegmentationLevelSetFunction` leaves
/// `CurvatureSpeed` at the base `LevelSetFunction`'s constant `1`.
///
/// Argument order follows SimpleITK's
/// `ThresholdSegmentationLevelSetImageFilter.yaml`; its defaults are
/// `lower_threshold = 0.0`, `upper_threshold = 255.0`,
/// `maximum_rms_error = 0.02`, both scalings `1.0`,
/// `number_of_iterations = 1000`, `reverse_expansion_direction = false`.
///
/// Errors if the two images differ in size.
#[allow(clippy::too_many_arguments)]
pub fn threshold_segmentation_level_set(
    initial_level_set: &Image,
    feature_image: &Image,
    lower_threshold: f64,
    upper_threshold: f64,
    maximum_rms_error: f64,
    propagation_scaling: f64,
    curvature_scaling: f64,
    number_of_iterations: u32,
    reverse_expansion_direction: bool,
) -> Result<LevelSetResult> {
    check_same_size(initial_level_set, feature_image)?;
    solve(
        initial_level_set,
        Solve {
            speed: threshold_speed(feature_image, lower_threshold, upper_threshold)?,
            advection: Vec::new(),
            curvature_speed: CurvatureSpeed::Unit,
            weights: Weights {
                propagation: propagation_scaling,
                curvature: curvature_scaling,
                advection: 0.0,
            },
            iso_surface_value: 0.0,
            maximum_rms_error,
            number_of_iterations,
            reverse_expansion_direction,
        },
    )
}

/// `LaplacianSegmentationLevelSetImageFilter`: the speed is the Laplacian of
/// the feature image, so the front locks onto the feature image's
/// second-derivative zero crossings — its edges.
///
/// `LaplacianSegmentationLevelSetFunction::CalculateSpeedImage` (hxx:28-47)
/// runs `LaplacianImageFilter` on the feature image cast to the level-set's
/// real pixel type; the filter's `UseImageSpacing` is left at its `true`
/// default, so the speed is `sum_d (g(x+e_d) + g(x-e_d) - 2 g(x)) /
/// spacing[d]^2` under a `ZeroFluxNeumannBoundaryCondition`.
///
/// There is no advection term: `LaplacianSegmentationLevelSetFunction::
/// SetAdvectionWeight` (h:81-88) silently ignores any non-zero value, and
/// SimpleITK's yaml exposes no `AdvectionScaling`. The curvature term is not
/// modulated by the speed (`CurvatureSpeed` is the base constant `1`).
///
/// Because the speed changes sign across an edge rather than vanishing on it,
/// ITK's header warns that the initial level set must already be close to the
/// edge to be captured; a coarse segmentation is the intended input.
///
/// Argument order follows SimpleITK's
/// `LaplacianSegmentationLevelSetImageFilter.yaml`; its defaults are
/// `maximum_rms_error = 0.02`, both scalings `1.0`,
/// `number_of_iterations = 1000`, `reverse_expansion_direction = false`.
///
/// Errors if the two images differ in size.
pub fn laplacian_segmentation_level_set(
    initial_level_set: &Image,
    feature_image: &Image,
    maximum_rms_error: f64,
    propagation_scaling: f64,
    curvature_scaling: f64,
    number_of_iterations: u32,
    reverse_expansion_direction: bool,
) -> Result<LevelSetResult> {
    check_same_size(initial_level_set, feature_image)?;
    solve(
        initial_level_set,
        Solve {
            speed: laplacian_speed(feature_image)?,
            advection: Vec::new(),
            curvature_speed: CurvatureSpeed::Unit,
            weights: Weights {
                propagation: propagation_scaling,
                curvature: curvature_scaling,
                advection: 0.0,
            },
            iso_surface_value: 0.0,
            maximum_rms_error,
            number_of_iterations,
            reverse_expansion_direction,
        },
    )
}

/// The evolved level set of [`canny_segmentation_level_set`], which reports one
/// extra measurement beyond [`LevelSetResult`]: `GetCannyImage()`.
#[derive(Clone, Debug, PartialEq)]
pub struct CannyLevelSetResult {
    /// The output level set, [`PixelId::Float32`]. Negative inside the
    /// segmented region, positive outside.
    pub image: Image,
    /// Number of iterations actually run before `Halt()` returned true.
    pub elapsed_iterations: u32,
    /// The RMS change of the final iteration.
    pub rms_change: f64,
    /// `GetCannyImage()`: the Canny edge map the speed and advection images
    /// were built from, [`PixelId::Float32`], `1` on edges and `0` elsewhere.
    pub canny_image: Image,
}

/// `CannySegmentationLevelSetImageFilter`: the speed is the distance to the
/// feature image's Canny edges and the advection field is that distance times
/// its own gradient, so the front is drawn onto the edges and pinned there.
///
/// `CannySegmentationLevelSetFunction::CalculateDistanceImage` (hxx:71-97) runs
/// `CannyEdgeDetectionImageFilter` with `UpperThreshold = threshold`,
/// `Variance = variance` and `MaximumError = 0.01` (its `LowerThreshold` is
/// left at the constructor's `0`), then a `DanielssonDistanceMapImageFilter` at
/// its defaults (unsquared distance, `UseImageSpacing` on). `CalculateSpeedImage`
/// grafts that distance map in; `CalculateAdvectionImage` (hxx:40-68) multiplies
/// `GradientImageFilter`'s central-difference gradient of the distance map by
/// the distance map itself.
///
/// The curvature term is not modulated by the speed (`CurvatureSpeed` is the
/// base constant `1`). `iso_surface_value` selects which isocontour of the
/// initial image seeds the front: `SparseFieldLevelSetImageFilter::
/// CopyInputToOutput` subtracts it from the input before taking zero crossings.
///
/// Argument order follows SimpleITK's
/// `CannySegmentationLevelSetImageFilter.yaml`; its defaults are
/// `threshold = 0.0`, `variance = 0.0`, `maximum_rms_error = 0.02`, all three
/// scalings `1.0`, `number_of_iterations = 1000`,
/// `reverse_expansion_direction = false`, `iso_surface_value = 0.0`.
///
/// Errors if the two images differ in size, or if `variance` is negative
/// (`GaussianOperator`'s constraint, surfaced by
/// [`canny_edge_detection`](crate::filters::canny_edge_detection)).
#[allow(clippy::too_many_arguments)]
pub fn canny_segmentation_level_set(
    initial_level_set: &Image,
    feature_image: &Image,
    threshold: f64,
    variance: f64,
    maximum_rms_error: f64,
    propagation_scaling: f64,
    curvature_scaling: f64,
    advection_scaling: f64,
    number_of_iterations: u32,
    reverse_expansion_direction: bool,
    iso_surface_value: f64,
) -> Result<CannyLevelSetResult> {
    check_same_size(initial_level_set, feature_image)?;

    let (canny_image, distance) = canny_distance_image(feature_image, threshold, variance)?;
    let speed = distance.to_f64_vec()?;
    let advection = if advection_scaling == 0.0 {
        Vec::new()
    } else {
        canny_advection_field(&distance, &speed)
    };

    let result = solve(
        initial_level_set,
        Solve {
            speed,
            advection,
            curvature_speed: CurvatureSpeed::Unit,
            weights: Weights {
                propagation: propagation_scaling,
                curvature: curvature_scaling,
                advection: advection_scaling,
            },
            iso_surface_value,
            maximum_rms_error,
            number_of_iterations,
            reverse_expansion_direction,
        },
    )?;

    Ok(CannyLevelSetResult {
        image: result.image,
        elapsed_iterations: result.elapsed_iterations,
        rms_change: result.rms_change,
        canny_image: image_from_f64(
            PixelId::Float32,
            canny_image.size(),
            &canny_image,
            &canny_image.to_f64_vec()?,
        )?,
    })
}

/// The three term weights, before `ReverseExpansionDirection` is applied.
struct Weights {
    propagation: f64,
    curvature: f64,
    advection: f64,
}

/// Everything the concrete filter hands to
/// `SegmentationLevelSetImageFilter::GenerateData`: the already-generated speed
/// and advection images, the term weights, and the stopping criteria.
struct Solve {
    /// `SegmentationLevelSetFunction::m_SpeedImage`, as each subclass's
    /// `CalculateSpeedImage` produced it.
    speed: Vec<f64>,
    /// `m_AdvectionImage`, one buffer per axis. Empty when the advection weight
    /// is zero — ITK never allocates the image then either.
    advection: Vec<Vec<f64>>,
    curvature_speed: CurvatureSpeed,
    weights: Weights,
    /// `SparseFieldLevelSetImageFilter::m_IsoSurfaceValue`.
    iso_surface_value: f64,
    maximum_rms_error: f64,
    number_of_iterations: u32,
    reverse_expansion_direction: bool,
}

/// `SegmentationLevelSetImageFilter::GenerateData` (hxx:64-101).
///
/// ITK generates the speed image only when the propagation weight is non-zero
/// and the advection image only when the advection weight is non-zero. Every
/// caller here has already generated both (the advection buffer is left empty
/// when the weight is zero), because for the `CurvatureSpeed::Propagation`
/// functions the curvature term samples the speed image even at a zero
/// propagation weight — which is exactly why
/// `GeodesicActiveContourLevelSetImageFilter::GenerateData` and
/// `ShapeDetectionLevelSetImageFilter::GenerateData` override `GenerateData` to
/// force the speed image into existence.
fn solve(initial_level_set: &Image, s: Solve) -> Result<LevelSetResult> {
    // "A positive speed value causes surface expansion, the opposite of the
    // default. Flip the sign of the propagation and advection weights."
    let sign = if s.reverse_expansion_direction {
        -1.0
    } else {
        1.0
    };

    let spacing = initial_level_set.spacing().to_vec();
    let func = LevelSetFunction::new(
        s.speed,
        s.advection,
        sign * s.weights.advection,
        sign * s.weights.propagation,
        s.weights.curvature,
        s.curvature_speed,
        &spacing,
    );

    // `CopyInputToOutput`: shift the input by the iso-surface value, then graft
    // the shifted image's zero-crossing map onto the output as the seed of the
    // active layer.
    let shifted: Vec<f64> = initial_level_set
        .to_f64_vec()?
        .into_iter()
        .map(|v| v - s.iso_surface_value)
        .collect();
    let mut shifted_image = Image::from_vec(initial_level_set.size(), shifted.clone())?;
    shifted_image.copy_geometry_from(initial_level_set);
    let zero_crossings = zero_crossing_values(&shifted_image, 0.0, 1.0)?;

    let solver = SparseFieldSolver::new(
        initial_level_set.size(),
        &spacing,
        SolverSetup {
            shifted,
            zero_crossings,
            func: DifferenceFunction::LevelSet(func),
            // `SegmentationLevelSetImageFilter`'s constructor:
            // `this->SetNumberOfLayers(TInputImage::ImageDimension)`.
            number_of_layers: spacing.len(),
            use_image_spacing: true,
            update_rule: UpdateRule::Unconstrained,
        },
    );
    let out = solver.run(s.maximum_rms_error, s.number_of_iterations);

    Ok(LevelSetResult {
        image: image_from_f64(
            PixelId::Float32,
            initial_level_set.size(),
            initial_level_set,
            &out.values,
        )?,
        elapsed_iterations: out.elapsed_iterations,
        rms_change: out.rms_change,
    })
}

/// `GeodesicActiveContourLevelSetFunction::CalculateAdvectionImage`
/// (hxx:43-92): the *negated* gradient of the feature image, taken with
/// `GradientRecursiveGaussianImageFilter` at `m_DerivativeSigma`. Stored as one
/// `f64` buffer per axis rather than a vector image.
///
/// Like `GradientRecursiveGaussianImageFilter` (hxx:245-250) each axis's
/// derivative is divided by that axis's spacing to leave index space. ITK's
/// `m_UseImageDirection` reorientation is not applied, matching the rest of
/// this crate's gradient filters, which assume identity direction cosines.
fn advection_field(feature_image: &Image) -> Result<Vec<Vec<f64>>> {
    let dim = feature_image.dimension();
    let spacing = feature_image.spacing().to_vec();
    let scratch = scratch_f64(feature_image)?;
    let sigma = vec![DERIVATIVE_SIGMA; dim];

    let mut fields = Vec::with_capacity(dim);
    for d in 0..dim {
        let mut orders = vec![GaussianOrder::ZeroOrder; dim];
        orders[d] = GaussianOrder::FirstOrder;
        // `scratch` is `Float64` (`scratch_f64`), so `recursive_gaussian_with_order`
        // would narrow `Float64 -> Float64` — the identity — into an `Image` that
        // `to_f64_vec` immediately unwraps again. Two full volumes per axis, for
        // no change in value. Take the `f64` core directly.
        let derivative = recursive_gaussian_f64(&scratch, &sigma, &orders, false)?;
        fields.push(derivative.into_iter().map(|v| -v / spacing[d]).collect());
    }
    Ok(fields)
}

/// The two inputs of every `SegmentationLevelSetImageFilter` must occupy the
/// same grid.
fn check_same_size(initial_level_set: &Image, feature_image: &Image) -> Result<()> {
    if initial_level_set.size() != feature_image.size() {
        return Err(FilterError::SizeMismatch {
            a: initial_level_set.size().to_vec(),
            b: feature_image.size().to_vec(),
        });
    }
    // The feature image is a named ProcessObject input, walked by the same
    // iterator the inherited verifier uses (unlike Chan–Vese's deep-copied
    // level set), so its grid must match the initial level set's.
    require_same_physical_space(initial_level_set, feature_image, 1)?;
    Ok(())
}

/// `ThresholdSegmentationLevelSetFunction::CalculateSpeedImage` (hxx:58-83)
/// with `m_EdgeWeight == 0`.
fn threshold_speed(
    feature_image: &Image,
    lower_threshold: f64,
    upper_threshold: f64,
) -> Result<Vec<f64>> {
    let mid = ((upper_threshold - lower_threshold) / 2.0) + lower_threshold;
    Ok(feature_image
        .to_f64_vec()?
        .into_iter()
        .map(|g| {
            if g < mid {
                g - lower_threshold
            } else {
                upper_threshold - g
            }
        })
        .collect())
}

/// `LaplacianSegmentationLevelSetFunction::CalculateSpeedImage` (hxx:28-47):
/// `LaplacianImageFilter` on the cast feature image, `UseImageSpacing` on.
fn laplacian_speed(feature_image: &Image) -> Result<Vec<f64>> {
    Ok(laplacian(&scratch_f64(feature_image)?, true)?.to_f64_vec()?)
}

/// `CannySegmentationLevelSetFunction::CalculateDistanceImage` (hxx:71-97):
/// the Canny edge map of the feature image, and the unsigned distance transform
/// to it. Returned as `(canny, distance)`.
fn canny_distance_image(
    feature_image: &Image,
    threshold: f64,
    variance: f64,
) -> Result<(Image, Image)> {
    let dim = feature_image.dimension();
    let canny = canny_edge_detection(
        &scratch_f64(feature_image)?,
        &vec![variance; dim],
        &vec![CANNY_MAXIMUM_ERROR; dim],
        threshold,
        CANNY_LOWER_THRESHOLD,
    )?;
    // `DanielssonDistanceMapImageFilter` at its defaults: `SquaredDistance` off,
    // `UseImageSpacing` on (itkDanielssonDistanceMapImageFilter.h:209).
    let distance = danielsson_distance_map(&canny, false, true)?;
    Ok((canny, distance))
}

/// `CannySegmentationLevelSetFunction::CalculateAdvectionImage` (hxx:40-68):
/// `GradientImageFilter`'s gradient of the distance map, multiplied pixelwise
/// by the distance map. Stored as one `f64` buffer per axis.
///
/// `GradientImageFilter` (hxx:105-125) applies a first-order `DerivativeOperator`
/// scaled by `1 / spacing[d]`, i.e. the central difference
/// `(f(x+e_d) - f(x-e_d)) / (2 spacing[d])`, under a
/// `ZeroFluxNeumannBoundaryCondition`. Its `m_UseImageDirection` reorientation
/// is not applied, matching the rest of this crate's gradient filters, which
/// assume identity direction cosines.
fn canny_advection_field(distance: &Image, values: &[f64]) -> Vec<Vec<f64>> {
    let grid = Grid::new(distance.size());
    let dim = grid.dim();
    let spacing = distance.spacing();

    let mut fields = vec![vec![0.0; values.len()]; dim];
    let mut coord = vec![0i64; dim];
    for (p, &center) in values.iter().enumerate() {
        coord.copy_from_slice(&grid.coord(p));
        for (d, field) in fields.iter_mut().enumerate() {
            coord[d] += 1;
            let forward = values[grid.clamped_index(&coord)];
            coord[d] -= 2;
            let backward = values[grid.clamped_index(&coord)];
            coord[d] += 1;
            field[p] = (forward - backward) / (2.0 * spacing[d]) * center;
        }
    }
    fields
}

/// An `f64` copy of `img`'s pixels with `img`'s geometry.
fn scratch_f64(img: &Image) -> Result<Image> {
    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec()?)?;
    scratch.copy_geometry_from(img);
    Ok(scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{bounded_reciprocal, gradient_magnitude_recursive_gaussian};
    use std::f64::consts::PI;

    const N: usize = 64;
    const CENTER: f64 = 31.5;

    fn image_from(values: Vec<f64>) -> Image {
        Image::from_vec(&[N, N], values).unwrap()
    }

    fn radius_from(x: usize, y: usize) -> f64 {
        let dx = x as f64 - CENTER;
        let dy = y as f64 - CENTER;
        (dx * dx + dy * dy).sqrt()
    }

    /// A signed distance function to a circle of radius `r`, negative inside.
    fn circle_level_set(r: f64) -> Image {
        let mut values = vec![0.0; N * N];
        for y in 0..N {
            for x in 0..N {
                values[x + N * y] = radius_from(x, y) - r;
            }
        }
        image_from(values)
    }

    /// A constant feature image: every pixel propagates and curves at speed
    /// `value`, so the solver is exercised without any edge structure.
    fn constant_feature(value: f64) -> Image {
        image_from(vec![value; N * N])
    }

    /// The radius of the zero level set, estimated from the area it encloses.
    /// The output is a sparse-field distance transform, so counting negative
    /// pixels is a far more stable estimate than tracing the contour.
    fn enclosed_radius(image: &Image) -> f64 {
        let inside = image
            .to_f64_vec()
            .unwrap()
            .iter()
            .filter(|&&v| v < 0.0)
            .count();
        (inside as f64 / PI).sqrt()
    }

    /// The bright disk phantom's indicator function: `1` inside, `0` outside.
    /// Used directly as the feature (speed) image, the way SimpleITK's
    /// `ShapeDetectionLevelSetImageFilter` regression test feeds it
    /// `LargeWhiteCircle_Float`.
    fn disk_indicator(radius: f64) -> Image {
        let mut values = vec![0.0; N * N];
        for y in 0..N {
            for x in 0..N {
                if radius_from(x, y) <= radius {
                    values[x + N * y] = 1.0;
                }
            }
        }
        image_from(values)
    }

    /// The bright disk phantom and its edge potential map: `1 / (1 + |grad(G *
    /// I)|)`, the feature image both filters document as their input.
    fn edge_potential_of_a_disk(radius: f64) -> Image {
        let mut values = vec![0.0; N * N];
        for y in 0..N {
            for x in 0..N {
                if radius_from(x, y) <= radius {
                    values[x + N * y] = 100.0;
                }
            }
        }
        let disk = image_from(values);
        let magnitude = gradient_magnitude_recursive_gaussian(&disk, 1.0, false).unwrap();
        bounded_reciprocal(&magnitude).unwrap()
    }

    // ---- Solver boundaries, driven through the advection-free filter -------

    /// `number_of_iterations == 0` makes `Halt()` true before the first pass:
    /// no iteration elapses, the RMS change stays at its initial zero, and the
    /// contour has not moved.
    #[test]
    fn zero_iterations_leaves_the_contour_where_it_started() {
        let initial = circle_level_set(8.0);
        let result =
            shape_detection_level_set(&initial, &constant_feature(1.0), 0.0, 1.0, 0.0, 0, false)
                .unwrap();

        assert_eq!(result.elapsed_iterations, 0);
        assert_eq!(result.rms_change, 0.0);
        assert!((enclosed_radius(&result.image) - 8.0).abs() < 0.5);
    }

    /// Pure positive propagation on a unit speed image: the front advances one
    /// `m_WaveDT / max|propagation|` per iteration, which for a 2-D image and a
    /// unit speed is `0.25` pixels.
    #[test]
    fn positive_propagation_grows_a_circle_outward() {
        let initial = circle_level_set(6.0);
        let iterations = 20;
        let result = shape_detection_level_set(
            &initial,
            &constant_feature(1.0),
            0.0,
            1.0,
            0.0,
            iterations,
            false,
        )
        .unwrap();

        assert_eq!(result.elapsed_iterations, iterations);
        let expected = 6.0 + 0.25 * f64::from(iterations);
        let actual = enclosed_radius(&result.image);
        assert!(
            (actual - expected).abs() < 1.0,
            "grew to {actual}, expected about {expected}"
        );
    }

    /// `ReverseExpansionDirection` negates the propagation weight, so the same
    /// positive speed image now contracts the circle by the same step.
    #[test]
    fn reverse_expansion_direction_flips_growth_into_shrinkage() {
        let initial = circle_level_set(12.0);
        let iterations = 20;
        let grown = shape_detection_level_set(
            &initial,
            &constant_feature(1.0),
            0.0,
            1.0,
            0.0,
            iterations,
            false,
        )
        .unwrap();
        let shrunk = shape_detection_level_set(
            &initial,
            &constant_feature(1.0),
            0.0,
            1.0,
            0.0,
            iterations,
            true,
        )
        .unwrap();

        assert!(enclosed_radius(&grown.image) > 12.5);
        assert!(enclosed_radius(&shrunk.image) < 11.5);
    }

    /// A negative `propagation_scaling` contracts the circle exactly as
    /// `reverse_expansion_direction` does — the flag and the sign are two ways
    /// to negate the same weight when there is no advection term.
    #[test]
    fn negative_propagation_scaling_matches_the_reverse_flag() {
        let initial = circle_level_set(12.0);
        let negated =
            shape_detection_level_set(&initial, &constant_feature(1.0), 0.0, -1.0, 0.0, 20, false)
                .unwrap();
        let reversed =
            shape_detection_level_set(&initial, &constant_feature(1.0), 0.0, 1.0, 0.0, 20, true)
                .unwrap();

        assert_eq!(
            negated.image.to_f64_vec().unwrap(),
            reversed.image.to_f64_vec().unwrap()
        );
    }

    /// Pure mean-curvature flow (`propagation_scaling == 0`): `phi_t = kappa
    /// |grad(phi)|` shrinks a circle. The speed image must still exist for
    /// `CurvatureSpeed` to sample, which is what
    /// `ShapeDetectionLevelSetImageFilter::GenerateData`'s override guarantees.
    #[test]
    fn pure_curvature_flow_shrinks_a_circle() {
        let initial = circle_level_set(12.0);
        let result =
            shape_detection_level_set(&initial, &constant_feature(1.0), 0.0, 0.0, 1.0, 30, false)
                .unwrap();

        let actual = enclosed_radius(&result.image);
        assert!(actual < 11.0, "curvature flow left the radius at {actual}");
        assert!(actual > 0.0);
    }

    /// A speed image of zeros makes every update zero, so the RMS change of the
    /// first iteration is zero and `Halt()` stops on the *second* pass — well
    /// before `number_of_iterations`. The `elapsed_iterations == 0` guard in
    /// `Halt` is why one iteration always runs.
    #[test]
    fn maximum_rms_error_stops_before_the_iteration_cap() {
        let initial = circle_level_set(8.0);
        let result =
            shape_detection_level_set(&initial, &constant_feature(0.0), 0.01, 1.0, 0.0, 50, false)
                .unwrap();

        assert_eq!(result.elapsed_iterations, 1);
        assert_eq!(result.rms_change, 0.0);
    }

    /// The same run with `maximum_rms_error == 0.0` can never satisfy
    /// `maximum_rms_error > rms_change`, so it burns the whole iteration cap.
    #[test]
    fn a_zero_maximum_rms_error_never_halts_early() {
        let initial = circle_level_set(8.0);
        let result =
            shape_detection_level_set(&initial, &constant_feature(0.0), 0.0, 1.0, 0.0, 50, false)
                .unwrap();

        assert_eq!(result.elapsed_iterations, 50);
    }

    // ---- End to end --------------------------------------------------------

    /// A geodesic active contour seeded well inside a bright disk locks onto
    /// the disk's edge and stays there: the advection doublet `-grad(g)` pulls
    /// the front into the minimum of the edge potential and holds it against
    /// the outward balloon force. Run at SimpleITK's default scalings.
    ///
    /// The lock is on the contour's *position*: `rms_change` does not reach
    /// `maximum_rms_error` on this phantom, because the edge potential never
    /// quite reaches zero, so the residual propagation force keeps the active
    /// layer's values oscillating at the amplitude `ComputeGlobalTimeStep`
    /// admits. The zero crossing itself is unmoved from iteration 100 to 800.
    #[test]
    fn geodesic_active_contour_locks_onto_a_synthetic_edge() {
        let true_radius = 12.0;
        let feature = edge_potential_of_a_disk(true_radius);

        let early = geodesic_active_contour_level_set(
            &circle_level_set(4.0),
            &feature,
            0.01,
            1.0,
            1.0,
            1.0,
            100,
            false,
        )
        .unwrap();
        let late = geodesic_active_contour_level_set(
            &circle_level_set(4.0),
            &feature,
            0.01,
            1.0,
            1.0,
            1.0,
            400,
            false,
        )
        .unwrap();

        let radius = enclosed_radius(&late.image);
        assert!(
            (radius - true_radius).abs() <= 1.0,
            "contour settled at radius {radius}, expected {true_radius}"
        );
        assert!(
            (radius - enclosed_radius(&early.image)).abs() < 0.1,
            "contour drifted between iteration 100 and 400"
        );
    }

    /// The advection term is what lets a geodesic active contour start *across*
    /// the boundary, which `ShapeDetectionLevelSetImageFilter` documents that
    /// it cannot do. Seeded at radius 16 — outside the disk — and with the
    /// balloon force switched off, the doublet alone pulls the contour back
    /// onto the edge.
    #[test]
    fn geodesic_active_contour_recovers_from_a_contour_outside_the_edge() {
        let true_radius = 12.0;
        let feature = edge_potential_of_a_disk(true_radius);

        let result = geodesic_active_contour_level_set(
            &circle_level_set(16.0),
            &feature,
            0.01,
            0.0,
            1.0,
            1.0,
            400,
            false,
        )
        .unwrap();

        let radius = enclosed_radius(&result.image);
        assert!(
            (radius - true_radius).abs() <= 1.0,
            "contour settled at radius {radius}, expected {true_radius}"
        );
    }

    /// Shape detection on the same bright disk, using the feature image
    /// SimpleITK's own `ShapeDetectionLevelSetImageFilter` regression test uses
    /// — the shape's indicator function, whose speed is exactly zero outside
    /// the structure. The front expands from a seed wholly inside, halts on the
    /// disk boundary because every active pixel's propagation *and* curvature
    /// term carries the vanished speed as a factor, and the RMS change falls to
    /// exactly zero, so `Halt()` fires far short of the iteration cap.
    #[test]
    fn shape_detection_locks_onto_the_edge_where_the_speed_vanishes() {
        let true_radius = 12.0;
        let feature = disk_indicator(true_radius);

        let result =
            shape_detection_level_set(&circle_level_set(4.0), &feature, 0.02, 1.0, 1.0, 400, false)
                .unwrap();

        let radius = enclosed_radius(&result.image);
        assert!(
            (radius - true_radius).abs() <= 1.0,
            "contour settled at radius {radius}, expected {true_radius}"
        );
        assert_eq!(result.rms_change, 0.0);
        assert!(result.elapsed_iterations < 400);
    }

    /// Shape detection on the *edge-potential* phantom — the one
    /// [`geodesic_active_contour_level_set`] locks onto — reaches the edge but
    /// cannot hold it, which is why ITK's header requires the initial contour
    /// to "lie wholly within (or wholly outside) the structure".
    ///
    /// The edge potential does retard the front (it travels less far than under
    /// a constant speed image). But `ComputeGlobalTimeStep` sets
    /// `dt = m_WaveDT / max|propagation term|` over the active layer, so the
    /// fastest active pixel always advances by `m_WaveDT` no matter how small
    /// the speed is. For a contour that reaches a symmetric edge band all at
    /// once, *every* active pixel is slow, the maximum is slow, `dt` grows to
    /// compensate — and the front walks straight through, then accelerates
    /// again where the edge potential recovers to 1. Only the advection doublet
    /// of the geodesic active contour supplies a force that reverses across the
    /// edge and can hold the contour there.
    #[test]
    fn shape_detection_slows_at_an_edge_potential_but_cannot_hold_it() {
        let feature = edge_potential_of_a_disk(12.0);

        let retarded =
            shape_detection_level_set(&circle_level_set(4.0), &feature, 0.0, 1.0, 1.0, 60, false)
                .unwrap();
        let unretarded = shape_detection_level_set(
            &circle_level_set(4.0),
            &constant_feature(1.0),
            0.0,
            1.0,
            1.0,
            60,
            false,
        )
        .unwrap();
        assert!(enclosed_radius(&retarded.image) < enclosed_radius(&unretarded.image));

        let leaked =
            shape_detection_level_set(&circle_level_set(4.0), &feature, 0.0, 1.0, 1.0, 140, false)
                .unwrap();
        let radius = enclosed_radius(&leaked.image);
        assert!(
            radius > 20.0,
            "front held at radius {radius}, expected a leak"
        );
    }

    // ---- Input validation --------------------------------------------------

    #[test]
    fn mismatched_input_sizes_are_rejected() {
        let initial = circle_level_set(8.0);
        let feature = Image::from_vec(&[8, 8], vec![1.0; 64]).unwrap();
        assert_eq!(
            shape_detection_level_set(&initial, &feature, 0.01, 1.0, 1.0, 10, false),
            Err(FilterError::SizeMismatch {
                a: vec![N, N],
                b: vec![8, 8],
            })
        );
    }

    /// The output is always `Float32`, matching ITK's default
    /// `TOutputPixelType`, even when the initial level set is `Float64`.
    #[test]
    fn output_is_always_float32() {
        let initial = circle_level_set(8.0);
        assert_eq!(initial.pixel_id(), PixelId::Float64);
        let result =
            shape_detection_level_set(&initial, &constant_feature(1.0), 0.0, 1.0, 0.0, 1, false)
                .unwrap();
        assert_eq!(result.image.pixel_id(), PixelId::Float32);
    }

    // ======================================================================
    // ThresholdSegmentationLevelSetImageFilter
    // ======================================================================

    /// A `1 x 7` feature image carrying the six interesting positions of the
    /// window `[50, 150]`, whose midpoint is `100`.
    #[test]
    fn threshold_speed_splits_at_the_window_midpoint() {
        let feature =
            Image::from_vec(&[7, 1], vec![0.0, 50.0, 99.0, 100.0, 150.0, 151.0, 200.0]).unwrap();
        let speed = threshold_speed(&feature, 50.0, 150.0).unwrap();

        // g < 100: g - 50.  g >= 100: 150 - g.  Zero on both window edges,
        // negative outside the window, peaking at 50 on the midpoint.
        assert_eq!(speed, vec![-50.0, 0.0, 49.0, 50.0, 0.0, -1.0, -50.0]);
    }

    /// The speed is a tent: it rises as `g - L` up to the midpoint and falls as
    /// `U - g` after it, peaking at `(U - L) / 2`. Both branches evaluate to the
    /// half-width at `mid` itself, which is why the strict `<` in
    /// `CalculateSpeedImage` is unobservable there. For `[0, 10]`, `mid = 5` and
    /// the peak is `5`.
    #[test]
    fn threshold_speed_is_a_tent_peaking_at_the_midpoint() {
        let feature = Image::from_vec(&[3, 1], vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            threshold_speed(&feature, 0.0, 10.0).unwrap(),
            vec![4.0, 5.0, 4.0]
        );
    }

    /// `lower_threshold > upper_threshold` is not rejected by ITK; the tent
    /// inverts and the speed is negative everywhere but the midpoint.
    #[test]
    fn threshold_speed_accepts_an_inverted_window() {
        let feature = Image::from_vec(&[3, 1], vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            threshold_speed(&feature, 10.0, 0.0).unwrap(),
            vec![-6.0, -5.0, -6.0]
        );
    }

    /// The intensity plateau the threshold filter is built to find: a disk of
    /// value `100` on a background of `0`.
    fn plateau(radius: f64) -> Image {
        let mut values = vec![0.0; N * N];
        for y in 0..N {
            for x in 0..N {
                if radius_from(x, y) <= radius {
                    values[x + N * y] = 100.0;
                }
            }
        }
        image_from(values)
    }

    /// End to end: a seed circle of radius `4` inside a plateau of radius `12`
    /// grows until it reaches the plateau boundary and stops there. With the
    /// window `[50, 150]` the speed is `+50` on the plateau (`100 >= mid = 100`,
    /// so `150 - 100`) and `-50` off it (`0 < 100`, so `0 - 50`), a stable
    /// equilibrium exactly on the boundary.
    #[test]
    fn threshold_grows_a_seed_to_fill_an_intensity_plateau() {
        let true_radius = 12.0;
        let result = threshold_segmentation_level_set(
            &circle_level_set(4.0),
            &plateau(true_radius),
            50.0,
            150.0,
            0.0,
            1.0,
            1.0,
            200,
            false,
        )
        .unwrap();

        let radius = enclosed_radius(&result.image);
        assert!(
            (radius - true_radius).abs() <= 1.0,
            "front settled at radius {radius}, expected {true_radius}"
        );
    }

    /// `ReverseExpansionDirection` negates the propagation weight — and there is
    /// no advection weight to negate — so it is indistinguishable from negating
    /// `propagation_scaling`. Pinned as exact pixel equality.
    #[test]
    fn threshold_reverse_expansion_direction_negates_the_propagation_weight() {
        let feature = plateau(12.0);
        let reversed = threshold_segmentation_level_set(
            &circle_level_set(8.0),
            &feature,
            50.0,
            150.0,
            0.0,
            1.0,
            1.0,
            20,
            true,
        )
        .unwrap();
        let negated = threshold_segmentation_level_set(
            &circle_level_set(8.0),
            &feature,
            50.0,
            150.0,
            0.0,
            -1.0,
            1.0,
            20,
            false,
        )
        .unwrap();

        assert_eq!(
            reversed.image.to_f64_vec().unwrap(),
            negated.image.to_f64_vec().unwrap()
        );
        // And it really does reverse: forward growth, reversed shrinkage.
        let forward = threshold_segmentation_level_set(
            &circle_level_set(8.0),
            &feature,
            50.0,
            150.0,
            0.0,
            1.0,
            1.0,
            20,
            false,
        )
        .unwrap();
        assert!(enclosed_radius(&forward.image) > 8.5);
        assert!(enclosed_radius(&reversed.image) < 7.5);
    }

    /// Unlike the geodesic active contour, the threshold function inherits
    /// `LevelSetFunction::CurvatureSpeed`'s constant `1`, so the curvature term
    /// is alive even where the speed image is zero. A seed circle on a feature
    /// image whose speed vanishes everywhere (`g == mid` is impossible; use
    /// `g == lower == upper`) still shrinks under pure curvature flow.
    #[test]
    fn threshold_curvature_is_not_damped_by_a_vanishing_speed() {
        let feature = constant_feature(50.0);
        assert_eq!(
            threshold_speed(&feature, 50.0, 50.0).unwrap(),
            vec![0.0; N * N]
        );

        let result = threshold_segmentation_level_set(
            &circle_level_set(12.0),
            &feature,
            50.0,
            50.0,
            0.0,
            1.0,
            1.0,
            30,
            false,
        )
        .unwrap();

        let radius = enclosed_radius(&result.image);
        assert!(radius < 11.0, "curvature flow left the radius at {radius}");
        assert!(radius > 0.0);
    }

    #[test]
    fn threshold_zero_iterations_leaves_the_contour_where_it_started() {
        let result = threshold_segmentation_level_set(
            &circle_level_set(8.0),
            &plateau(12.0),
            50.0,
            150.0,
            0.02,
            1.0,
            1.0,
            0,
            false,
        )
        .unwrap();

        assert_eq!(result.elapsed_iterations, 0);
        assert_eq!(result.rms_change, 0.0);
        assert!((enclosed_radius(&result.image) - 8.0).abs() < 0.5);
    }

    #[test]
    fn threshold_rejects_mismatched_input_sizes() {
        let feature = Image::from_vec(&[8, 8], vec![1.0; 64]).unwrap();
        assert_eq!(
            threshold_segmentation_level_set(
                &circle_level_set(8.0),
                &feature,
                0.0,
                255.0,
                0.02,
                1.0,
                1.0,
                10,
                false
            ),
            Err(FilterError::SizeMismatch {
                a: vec![N, N],
                b: vec![8, 8],
            })
        );
    }

    // ======================================================================
    // LaplacianSegmentationLevelSetImageFilter
    // ======================================================================

    /// The 3x3 discrete Laplacian of a unit impulse, under
    /// `ZeroFluxNeumannBoundaryCondition` and unit spacing. Center:
    /// `(0 + 0 - 2) + (0 + 0 - 2) == -4`. Edge midpoint `(1, 0)`: the `x` axis
    /// contributes `0 + 0 - 0 == 0`, the `y` axis `1 + 0 - 0 == 1` because the
    /// out-of-image neighbor clamps back onto `(1, 0)` itself. Corners: `0`.
    #[test]
    fn laplacian_speed_is_the_discrete_laplacian_of_the_feature_image() {
        #[rustfmt::skip]
        let feature = Image::from_vec(&[3, 3], vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]).unwrap();

        #[rustfmt::skip]
        let expected = vec![
            0.0,  1.0, 0.0,
            1.0, -4.0, 1.0,
            0.0,  1.0, 0.0,
        ];
        assert_eq!(laplacian_speed(&feature).unwrap(), expected);
    }

    /// `LaplacianImageFilter`'s `UseImageSpacing` is on, so each axis's second
    /// difference is divided by that axis's squared spacing. Halving the `x`
    /// spacing quadruples the `x` contribution: the impulse's center becomes
    /// `-2/0.25 - 2 == -10`.
    #[test]
    fn laplacian_speed_divides_each_axis_by_its_squared_spacing() {
        let mut feature = Image::from_vec(&[3, 3], {
            let mut v = vec![0.0; 9];
            v[4] = 1.0;
            v
        })
        .unwrap();
        feature.set_spacing(&[0.5, 1.0]).unwrap();
        assert_eq!(laplacian_speed(&feature).unwrap()[4], -10.0);
    }

    /// The Laplacian of a locally flat feature image is zero, so every active
    /// pixel's propagation term vanishes. With the curvature weight also zero
    /// the update is identically zero, the first iteration's RMS change is `0`,
    /// and `Halt()` fires on the second pass — the front never moves. This is
    /// the failure mode ITK's header warns about: the Laplacian speed carries no
    /// information away from an edge.
    #[test]
    fn laplacian_front_does_not_move_inside_a_flat_region() {
        let result = laplacian_segmentation_level_set(
            &circle_level_set(4.0),
            &plateau(20.0),
            0.02,
            1.0,
            0.0,
            50,
            false,
        )
        .unwrap();

        assert_eq!(result.elapsed_iterations, 1);
        assert_eq!(result.rms_change, 0.0);
        assert!((enclosed_radius(&result.image) - 4.0).abs() < 0.5);
    }

    /// `ReverseExpansionDirection` negates the propagation weight; there is no
    /// advection weight (the Laplacian function refuses any non-zero value), so
    /// it is exactly equivalent to negating `propagation_scaling`.
    #[test]
    fn laplacian_reverse_expansion_direction_negates_the_propagation_weight() {
        let feature = plateau(12.0);
        let reversed = laplacian_segmentation_level_set(
            &circle_level_set(11.0),
            &feature,
            0.0,
            1.0,
            0.0,
            10,
            true,
        )
        .unwrap();
        let negated = laplacian_segmentation_level_set(
            &circle_level_set(11.0),
            &feature,
            0.0,
            -1.0,
            0.0,
            10,
            false,
        )
        .unwrap();

        assert_eq!(
            reversed.image.to_f64_vec().unwrap(),
            negated.image.to_f64_vec().unwrap()
        );
    }

    #[test]
    fn laplacian_zero_iterations_leaves_the_contour_where_it_started() {
        let result = laplacian_segmentation_level_set(
            &circle_level_set(8.0),
            &plateau(12.0),
            0.02,
            1.0,
            1.0,
            0,
            false,
        )
        .unwrap();

        assert_eq!(result.elapsed_iterations, 0);
        assert_eq!(result.rms_change, 0.0);
        assert!((enclosed_radius(&result.image) - 8.0).abs() < 0.5);
    }

    #[test]
    fn laplacian_rejects_mismatched_input_sizes() {
        let feature = Image::from_vec(&[8, 8], vec![1.0; 64]).unwrap();
        assert_eq!(
            laplacian_segmentation_level_set(
                &circle_level_set(8.0),
                &feature,
                0.02,
                1.0,
                1.0,
                10,
                false
            ),
            Err(FilterError::SizeMismatch {
                a: vec![N, N],
                b: vec![8, 8],
            })
        );
    }

    // ======================================================================
    // CannySegmentationLevelSetImageFilter
    // ======================================================================

    /// A `9 x 7` step edge: `0` for `x < 4`, `100` for `x >= 4`.
    fn step_edge() -> Image {
        let mut values = vec![0.0; 9 * 7];
        for y in 0..7 {
            for x in 4..9 {
                values[x + 9 * y] = 100.0;
            }
        }
        Image::from_vec(&[9, 7], values).unwrap()
    }

    /// The speed image is the unsigned distance to the Canny edges. At
    /// `threshold = 10`, `variance = 0` the detector marks columns `2` and `3`
    /// on every row, so the distance along a row is `min(|x - 2|, |x - 3|)`.
    #[test]
    fn canny_speed_is_the_distance_to_the_canny_edges() {
        let (canny, distance) = canny_distance_image(&step_edge(), 10.0, 0.0).unwrap();

        let edges = canny.to_f64_vec().unwrap();
        let expected_edges: Vec<f64> = (0..9)
            .map(|x| f64::from(u8::from(x == 2 || x == 3)))
            .collect();
        for y in 0..7 {
            assert_eq!(&edges[9 * y..9 * (y + 1)], &expected_edges[..], "row {y}");
        }

        let expected: Vec<f64> = (0..9)
            .map(|x: i32| f64::from((x - 2).abs().min((x - 3).abs())))
            .collect();
        let d = distance.to_f64_vec().unwrap();
        for y in 0..7 {
            assert_eq!(&d[9 * y..9 * (y + 1)], &expected[..], "row {y}");
        }
    }

    /// The advection field is `d * grad(d)` with `grad` the central difference
    /// `(d(x+1) - d(x-1)) / (2 * spacing)` under a zero-flux boundary. Along a
    /// row `d = [2, 1, 0, 0, 1, 2, 3, 4, 5]` that is
    /// `[-1, -1, 0, 0, 1, 2, 3, 4, 2.5]`, and the `y` component is zero because
    /// `d` is constant down every column.
    #[test]
    fn canny_advection_field_is_the_distance_times_its_own_gradient() {
        let (_, distance) = canny_distance_image(&step_edge(), 10.0, 0.0).unwrap();
        let values = distance.to_f64_vec().unwrap();
        let field = canny_advection_field(&distance, &values);

        let expected_x = [-1.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.5];
        for y in 0..7 {
            assert_eq!(&field[0][9 * y..9 * (y + 1)], &expected_x[..], "row {y}");
            assert_eq!(&field[1][9 * y..9 * (y + 1)], &[0.0; 9][..], "row {y}");
        }
    }

    /// `advection_scaling == 0` leaves the advection buffer unallocated, exactly
    /// as `SegmentationLevelSetImageFilter::GenerateData` skips
    /// `GenerateAdvectionImage`.
    #[test]
    fn canny_zero_advection_scaling_runs_without_an_advection_field() {
        let result = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &disk_indicator(12.0),
            0.1,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            5,
            false,
            0.0,
        )
        .unwrap();
        assert_eq!(result.elapsed_iterations, 5);
    }

    /// End to end, at SimpleITK's own `cthead` regression settings
    /// (`Threshold = 5`, `Variance = 1`, `PropagationScaling = 1`,
    /// `CurvatureScaling = 0`, `AdvectionScaling = 0.1`): a seed circle of
    /// radius `4` inside a bright disk of radius `12` grows under the
    /// distance-to-edge speed and freezes on the disk's Canny edge ring, where
    /// the distance — hence both the propagation speed and the advection field
    /// `d grad(d)` — is exactly zero. The update over the whole active layer
    /// then vanishes, `rms_change` is exactly `0`, and `Halt()` fires far short
    /// of the iteration cap.
    #[test]
    fn canny_grows_a_seed_onto_the_disk_edge_and_freezes_there() {
        let result = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &plateau(12.0),
            5.0,
            1.0,
            0.02,
            1.0,
            0.0,
            0.1,
            400,
            false,
            0.0,
        )
        .unwrap();

        // The Canny ring the front is chasing.
        let edges = result.canny_image.to_f64_vec().unwrap();
        let inner_edge = (0..N)
            .flat_map(|y| (0..N).map(move |x| (x, y)))
            .filter(|&(x, y)| edges[x + N * y] == 1.0)
            .map(|(x, y)| radius_from(x, y))
            .fold(f64::MAX, f64::min);
        assert!(
            inner_edge > 11.0 && inner_edge < 12.1,
            "ring at {inner_edge}"
        );

        let radius = enclosed_radius(&result.image);
        assert!(
            (radius - inner_edge).abs() < 0.5,
            "front settled at radius {radius}, Canny ring starts at {inner_edge}"
        );
        assert_eq!(result.rms_change, 0.0);
        assert!(result.elapsed_iterations < 400);
    }

    /// The front is stationary once it reaches the ring: iteration 100 and
    /// iteration 200 give bit-identical level sets. Run with
    /// `maximum_rms_error == 0` so `Halt()` cannot end either run early.
    #[test]
    fn canny_front_does_not_drift_after_reaching_the_edge() {
        let early = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &plateau(12.0),
            5.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            100,
            false,
            0.0,
        )
        .unwrap();
        let late = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &plateau(12.0),
            5.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            200,
            false,
            0.0,
        )
        .unwrap();

        assert_eq!(early.elapsed_iterations, 100);
        assert_eq!(late.elapsed_iterations, 200);
        assert_eq!(
            early.image.to_f64_vec().unwrap(),
            late.image.to_f64_vec().unwrap()
        );
    }

    /// The advection field `d grad(d)` points *away* from the Canny edges (it
    /// is `d` times the ascent direction of the distance map), so with a
    /// positive advection weight it opposes the outward propagation rather than
    /// reinforcing it. At SimpleITK's default `advection_scaling = 1.0` — equal
    /// to the propagation scaling — advection wins on this phantom and the seed
    /// collapses to nothing.
    ///
    /// This is upstream behaviour, not a port artefact:
    /// `CannySegmentationLevelSetFunction::Initialize` sets the propagation and
    /// advection weights to `-1`, but SimpleITK's generated `Execute`
    /// overwrites both with the exposed scalings, whose defaults are `+1`. ITK's
    /// own regression test picks `propagation = 0.5`, `advection = 1.0` on a
    /// binary seed at `iso_surface_value = 0.5`; the balance is
    /// application-specific.
    #[test]
    fn canny_default_advection_scaling_opposes_the_propagation_term() {
        let collapsed = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &plateau(12.0),
            5.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            200,
            false,
            0.0,
        )
        .unwrap();
        assert_eq!(enclosed_radius(&collapsed.image), 0.0);
    }

    /// `ReverseExpansionDirection` negates *both* the propagation and advection
    /// weights, so it equals negating both scalings.
    #[test]
    fn canny_reverse_expansion_direction_negates_propagation_and_advection() {
        let feature = disk_indicator(12.0);
        let reversed = canny_segmentation_level_set(
            &circle_level_set(8.0),
            &feature,
            0.1,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            15,
            true,
            0.0,
        )
        .unwrap();
        let negated = canny_segmentation_level_set(
            &circle_level_set(8.0),
            &feature,
            0.1,
            1.0,
            0.0,
            -1.0,
            1.0,
            -1.0,
            15,
            false,
            0.0,
        )
        .unwrap();

        assert_eq!(
            reversed.image.to_f64_vec().unwrap(),
            negated.image.to_f64_vec().unwrap()
        );
    }

    /// `iso_surface_value` selects which isocontour of the initial image seeds
    /// the front: `CopyInputToOutput` subtracts it before taking zero crossings.
    /// Seeding the `-4` contour of a signed distance to a circle of radius `8`
    /// is the same as seeding the zero contour of a circle of radius `4`.
    #[test]
    fn canny_iso_surface_value_shifts_the_initial_contour() {
        let feature = disk_indicator(12.0);
        let shifted = canny_segmentation_level_set(
            &circle_level_set(8.0),
            &feature,
            0.1,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            10,
            false,
            -4.0,
        )
        .unwrap();
        let seeded_directly = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &feature,
            0.1,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            10,
            false,
            0.0,
        )
        .unwrap();

        assert_eq!(
            shifted.image.to_f64_vec().unwrap(),
            seeded_directly.image.to_f64_vec().unwrap()
        );
    }

    /// `GetCannyImage()` is reported as a measurement, `Float32`, `1` on edges.
    #[test]
    fn canny_reports_the_edge_map_it_used() {
        let result = canny_segmentation_level_set(
            &circle_level_set(4.0),
            &disk_indicator(12.0),
            0.1,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1,
            false,
            0.0,
        )
        .unwrap();

        assert_eq!(result.canny_image.pixel_id(), PixelId::Float32);
        assert_eq!(result.canny_image.size(), &[N, N]);
        let edges = result.canny_image.to_f64_vec().unwrap();
        assert!(edges.iter().all(|&v| v == 0.0 || v == 1.0));
        let on = edges.iter().filter(|&&v| v == 1.0).count();
        assert!(on > 0, "the disk's boundary produced no Canny edges");
    }

    #[test]
    fn canny_zero_iterations_leaves_the_contour_where_it_started() {
        let result = canny_segmentation_level_set(
            &circle_level_set(8.0),
            &disk_indicator(12.0),
            0.1,
            1.0,
            0.02,
            1.0,
            1.0,
            1.0,
            0,
            false,
            0.0,
        )
        .unwrap();

        assert_eq!(result.elapsed_iterations, 0);
        assert_eq!(result.rms_change, 0.0);
        assert!((enclosed_radius(&result.image) - 8.0).abs() < 0.5);
    }

    #[test]
    fn canny_rejects_mismatched_input_sizes() {
        let feature = Image::from_vec(&[8, 8], vec![1.0; 64]).unwrap();
        assert_eq!(
            canny_segmentation_level_set(
                &circle_level_set(8.0),
                &feature,
                0.0,
                0.0,
                0.02,
                1.0,
                1.0,
                1.0,
                10,
                false,
                0.0
            )
            .unwrap_err(),
            FilterError::SizeMismatch {
                a: vec![N, N],
                b: vec![8, 8],
            }
        );
    }

    /// A negative `variance` is `GaussianOperator`'s error, surfaced through
    /// `canny_edge_detection`.
    #[test]
    fn canny_rejects_a_negative_variance() {
        assert_eq!(
            canny_segmentation_level_set(
                &circle_level_set(8.0),
                &disk_indicator(12.0),
                0.0,
                -1.0,
                0.02,
                1.0,
                1.0,
                1.0,
                10,
                false,
                0.0
            )
            .unwrap_err(),
            FilterError::InvalidVariance(vec![-1.0, -1.0])
        );
    }
}
