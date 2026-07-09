//! `ScalarChanAndVeseDenseLevelSetImageFilter`: the region-based, *dense*
//! Chan–Vese active contour without edges.
//!
//! Ported from ITK's `Modules/Nonunit/Review`:
//!
//! | Layer | ITK source |
//! |---|---|
//! | the solver loop | `itkMultiphaseFiniteDifferenceImageFilter.hxx` (`GenerateData`, `Halt`) |
//! | update application + reinitialisation | `itkMultiphaseDenseFiniteDifferenceImageFilter.hxx` (`CalculateChange`, `ApplyUpdate`, `CopyInputToOutput`) |
//! | the PDE | `itkRegionBasedLevelSetFunction.hxx` (`ComputeUpdate`, `ComputeHessian`, `ComputeCurvature`, `ComputeGlobalTerm`) |
//! | `c_in` / `c_out` | `itkScalarChanAndVeseLevelSetFunction.hxx` (`ComputeParameters`, `UpdateSharedDataParameters`) |
//! | the Heaviside variants | `itkAtanRegularizedHeavisideStepFunction.hxx`, `itkSinRegularizedHeavisideStepFunction.hxx`, `itkHeavisideStepFunction.hxx` |
//! | this module's public surface | `ScalarChanAndVeseDenseLevelSetImageFilter.yaml` |
//!
//! # Inputs, output, sign convention
//!
//! `initial_level_set` is a real image, **negative inside** the initial region;
//! `feature_image` is the image being segmented. The **output is not the level
//! set** — `MultiphaseDenseFiniteDifferenceImageFilter::PostProcessOutput` calls
//! `CopyInputToOutput`, which fills the output with `0` and then writes the
//! function's lookup value (`1` for the single level set SimpleITK wires up)
//! wherever the final `phi < 0` strictly. So the returned image is a `0`/`1`
//! label image in the input's pixel type.
//!
//! # The single-phase degenerate case
//!
//! SimpleITK's `custom_itk_cast` for `InitialImage` calls `SetFunctionCount(1)`
//! and `SetLevelSet(0, ...)`, so only one level set ever exists. Every piece of
//! the multiphase container machinery collapses:
//!
//! * `UnconstrainedRegionBasedLevelSetFunctionSharedData::PopulateListImage`
//!   fills the nearest-neighbour list image with `L = {0}` at every pixel, so
//!   `ComputeParameters`' inner loop runs exactly once per pixel and its
//!   background product is just `1 - H(phi)`.
//! * `RegionBasedLevelSetFunction::ComputeGlobalTerm` guards the overlap term
//!   with `if (m_SharedData->m_FunctionCount > 1)`, so `overlapTerm` is `0` and
//!   the external term's `product` stays at its initialiser `1`.
//!   `ScalarRegionBasedLevelSetFunction::ComputeOverlapParameters` is never
//!   called.
//! * `ScalarChanAndVeseDenseLevelSetImageFilter::Initialize` crops the feature
//!   image to the level set's physical footprint with a
//!   `RegionOfInterestImageFilter`. With one level set of the same size as the
//!   feature image — the only shape SimpleITK's two same-sized inputs can take —
//!   the ROI is the whole image and `RegionBasedLevelSetFunctionData::m_Start`
//!   is the zero index, so `GetIndex`/`GetFeatureIndex` are the identity. This
//!   port therefore indexes the feature image directly and requires the two
//!   inputs to have the same size.
//! * The `KdTree` is never set (SimpleITK exposes no setter), so
//!   `m_SharedData->SetKdTree` is skipped.
//!
//! `ScalarRegionBasedLevelSetFunction::UpdatePixel` — the narrow-band
//! incremental `c_in`/`c_out` update — is only called by the *sparse* filter and
//! is not part of this port.
//!
//! # The equation
//!
//! With `H` the Heaviside variant, `H_p = H(-phi_p)` (the convention is inside
//! negative, so `H` is `1` inside), `dh_p = H'(-phi_p)`, and
//!
//! ```text
//! c_in  = sum_p f_p H_p     / sum_p H_p        (0 if the denominator <= eps)
//! c_out = sum_p f_p (1-H_p) / sum_p (1-H_p)    (0 if the denominator <= eps)
//! ```
//!
//! the per-pixel update is
//!
//! ```text
//! update_p = curvature_weight * kappa_p * dh_p                       (if dh_p != 0 and curvature_weight != 0)
//!          + reinit_weight * (laplacian_p - kappa_p)                 (if reinit_weight != 0)
//!          + dh_p * ( lambda1 (f_p - c_in)^2 - lambda2 (f_p - c_out)^2
//!                     + 2 * volume_matching_weight * (sum_p H_p - volume)
//!                     - area_weight )                                (if dh_p != 0)
//! ```
//!
//! `kappa` is `ComputeCurvature`'s mean curvature, `laplacian` is
//! `sum_i d2phi/dx_i^2`. Both `CurvatureSpeed` and `LaplacianSmoothingSpeed`
//! are inherited unoverridden and return `1`.
//!
//! Note the sign of the global term: `interim - outTerm`, *not* the textbook
//! `-lambda1 (f-c_in)^2 + lambda2 (f-c_out)^2`. It comes out right because `H`
//! is evaluated at `-phi`, so `dh > 0` and a pixel whose intensity matches
//! `c_in` gets a negative update, pushing `phi` further negative — i.e. the
//! region grows to swallow it.
//!
//! Each iteration then does `phi += 0.08 * update`, rebuilds `phi` as a signed
//! distance map, and reports the RMS of *that* rebuild (see below).
//!
//! # Upstream behaviour reproduced here
//!
//! * **The time step is a hard-coded `0.08`.**
//!   `MultiphaseDenseFiniteDifferenceImageFilter::CalculateChange`
//!   (itkMultiphaseDenseFiniteDifferenceImageFilter.hxx:153) computes a CFL time
//!   step through `RegionBasedLevelSetFunction::ComputeGlobalTimeStep` and then
//!   throws it away: `timeStep = 0.08; // FIXME !!! After all this work, assign
//!   a constant !!! Why ??`. Consequently `GlobalDataStruct`'s
//!   `m_MaxCurvatureChange` / `m_MaxAdvectionChange` / `m_MaxGlobalChange`
//!   accumulators feed nothing, and this port does not maintain them.
//!
//! * **The level set is replaced by a signed distance map every iteration.**
//!   `m_ReinitializeCounter` defaults to `1`
//!   (itkMultiphaseDenseFiniteDifferenceImageFilter.h:166) and `ApplyUpdate`'s
//!   guard is `GetElapsedIterations() % m_ReinitializeCounter == 0`, evaluated
//!   *before* `m_ElapsedIterations++`. With counter `1` that is true on every
//!   iteration (and would be true on the first one for any counter, since
//!   `0 % k == 0`). So after `phi += dt * update`, `phi` is thresholded to
//!   `[NonpositiveMin, 0]` and overwritten with the
//!   `SignedMaurerDistanceMapImageFilter` of that binary mask
//!   (`InsideIsPositive` off, unsquared). SimpleITK exposes no setter for the
//!   counter, so this port hard-codes the every-iteration path.
//!
//! * **`RMSChange` therefore measures the reinitialisation, not the PDE.**
//!   `ApplyUpdate` accumulates `sum (dt * update)^2`, then — inside the
//!   reinitialisation branch — resets the accumulator to `0` and refills it with
//!   `sum (phi_after_update - distance_map)^2`. Since the branch always runs, the
//!   PDE accumulator is always discarded, and it is not computed here. The
//!   reported value is `sqrt(accumulator / number_of_pixels)`, which is
//!   typically large (SimpleITK's own regression test expects `721.33` after two
//!   iterations) and essentially never drops below the `0.02` default
//!   `maximum_rms_error`. The convergence criterion is effectively inert.
//!
//! * **`use_image_spacing` only reaches the distance map.**
//!   `MultiphaseFiniteDifferenceImageFilter::GenerateData` uses it to build
//!   `coeffs` for `SetScaleCoefficients`, but no class in the Chan–Vese chain
//!   ever reads `m_ScaleCoefficients` or `ComputeNeighborhoodScales()`.
//!   `ComputeHessian` divides by `m_InvSpacing`, which
//!   `RegionBasedLevelSetFunction::SetFeatureImage` sets unconditionally from the
//!   **feature** image's spacing. So the derivatives are always in physical
//!   units, taken from the feature image, and the flag's only live use is
//!   `maurer->SetUseImageSpacing(m_UseImageSpacing)`
//!   (itkMultiphaseDenseFiniteDifferenceImageFilter.hxx:227). The distance map's
//!   own spacing comes from the *level set* image (the threshold filter
//!   `CopyInformation`s from it), so a level set and a feature image with
//!   different spacings drive the two halves of one iteration with different
//!   metrics.
//!
//! * **`laplacian_term` subtracts a curvature that may not have been computed.**
//!   `ComputeUpdate` declares `ScalarValueType curvature{}` and only assigns it
//!   inside `if ((dh != 0.) && (m_CurvatureWeight != 0))`. The reinitialisation
//!   smoothing term is `(ComputeLaplacian(gd) - curvature) * reinit_weight`,
//!   evaluated in a *separate* `if`. With `curvature_weight == 0` (or `dh == 0`)
//!   the subtrahend is the zero initialiser, so the smoothing term degenerates
//!   from `Laplacian(phi) - div(grad phi / |grad phi|)` to the bare Laplacian.
//!   This is reproduced.
//!
//! * **The advection term is dead.** `m_AdvectionWeight` is zero-initialised,
//!   nothing in the Chan–Vese chain sets it, and SimpleITK exposes no setter, so
//!   `ComputeUpdate`'s advection branch never runs and
//!   `RegionBasedLevelSetFunction::AdvectionField`'s zero vector is never read.
//!   `CalculateAdvectionImage()` (called from `Initialize`) is an empty base
//!   implementation. None of it is ported.
//!
//! * **`Heaviside` (the unregularised variant) freezes the PDE.**
//!   `HeavisideStepFunction::EvaluateDerivative` returns `1` only when its
//!   argument is *exactly* zero. Every pixel with `phi != 0` therefore gets
//!   `dh == 0`, hence no curvature term and no global term; with the default
//!   `reinitialization_smoothing_weight == 0` the whole update is zero and each
//!   iteration is a pure re-distancing of `{phi <= 0}`.
//!
//! * **`epsilon` is validated only for the regularised variants.**
//!   `RegularizedHeavisideStepFunction::SetEpsilon` throws unless
//!   `epsilon > NumericTraits<double>::epsilon()`; SimpleITK's `custom_itk_cast`
//!   calls it only on the `Atan`/`Sin` branches, so `Heaviside` accepts any
//!   `epsilon` and ignores it.
//!
//! * **`volume` is in pixels, not physical units.** `ComputeVolumeRegularizationTerm`
//!   returns `2 * (sum_p H_p - volume)`, and `sum_p H_p` is an unweighted pixel
//!   count; the spacing never enters.
//!
//! # Where this port differs
//!
//! ITK instantiates the whole chain on the image's pixel type, so for a
//! `Float32` image `c_in`/`c_out` are `double` (the accumulators in
//! `ScalarChanAndVeseLevelSetFunctionData` are declared `double`) but the
//! Heaviside image, the update buffer, the level set and `ComputeUpdate`'s
//! entire arithmetic are `float`. This port computes in `f64` throughout, as the
//! rest of the crate does, and narrows only at the output image. Results agree
//! to `f32` round-off.

use sitk_core::{Image, PixelId};

use crate::distance::signed_maurer_distance_map;
use crate::error::{FilterError, Result};
use crate::image_from_f64;

/// `itk::Math::eps` (itkMath.h:119), the guard on `c_in`/`c_out`'s denominators
/// and on `ComputeCurvature`'s gradient magnitude.
const EPS: f64 = f64::EPSILON;

/// The time step `CalculateChange` returns after discarding the CFL estimate
/// (itkMultiphaseDenseFiniteDifferenceImageFilter.hxx:153).
const TIME_STEP: f64 = 0.08;

/// Which `HeavisideStepFunctionBase` the difference function's domain function
/// is, per `ScalarChanAndVeseDenseLevelSetImageFilter.yaml`'s
/// `HeavisideStepFunction` enum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeavisideStepFunction {
    /// `AtanRegularizedHeavisideStepFunction`: `H(x) = 1/2 + atan(x/eps)/pi`.
    /// The yaml default.
    #[default]
    AtanRegularized,
    /// `SinRegularizedHeavisideStepFunction`: `H(x) = (1 + sin(pi x / 2 eps))/2`
    /// clamped to `[0, 1]` outside `|x| < eps`.
    SinRegularized,
    /// `HeavisideStepFunction`: the unregularised `H(x) = [x >= 0]`, whose
    /// derivative is `1` only at `x == 0`.
    Heaviside,
}

/// The yaml's parameter surface. [`Default`] carries the yaml defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChanAndVeseParams {
    /// Stop once an iteration's `RMSChange` is `<= maximum_rms_error`. Default
    /// `0.02`. See the module docs: with the every-iteration reinitialisation
    /// this criterion is effectively unreachable.
    pub maximum_rms_error: f64,
    /// Iteration cap. Default `1000`.
    pub number_of_iterations: u32,
    /// `lambda1`, the internal (inside) intensity difference weight. Default `1.0`.
    pub lambda1: f64,
    /// `lambda2`, the external (outside) intensity difference weight. Default `1.0`.
    pub lambda2: f64,
    /// Width of the Heaviside regularisation. Default `1.0`. Must exceed
    /// `f64::EPSILON` for [`HeavisideStepFunction::AtanRegularized`] and
    /// [`HeavisideStepFunction::SinRegularized`]; ignored (and unvalidated) for
    /// [`HeavisideStepFunction::Heaviside`].
    pub epsilon: f64,
    /// `gamma`, scales the curvature (contour length) term. Default `1.0`.
    pub curvature_weight: f64,
    /// `nu`, the area regularisation, *subtracted* from the global term.
    /// Default `0.0`.
    pub area_weight: f64,
    /// Weight of the Laplacian smoothing term. Default `0.0`.
    pub reinitialization_smoothing_weight: f64,
    /// Target number of pixels inside the level set. Default `0.0`. In pixels,
    /// not physical volume.
    pub volume: f64,
    /// `tau`, weight of the volume matching term. Default `0.0`.
    pub volume_matching_weight: f64,
    /// Which Heaviside variant to use. Default
    /// [`HeavisideStepFunction::AtanRegularized`].
    pub heaviside_step_function: HeavisideStepFunction,
    /// Default `true`. Only reaches the per-iteration signed distance map; the
    /// PDE's derivatives always use the feature image's spacing.
    pub use_image_spacing: bool,
}

impl Default for ChanAndVeseParams {
    fn default() -> Self {
        ChanAndVeseParams {
            maximum_rms_error: 0.02,
            number_of_iterations: 1000,
            lambda1: 1.0,
            lambda2: 1.0,
            epsilon: 1.0,
            curvature_weight: 1.0,
            area_weight: 0.0,
            reinitialization_smoothing_weight: 0.0,
            volume: 0.0,
            volume_matching_weight: 0.0,
            heaviside_step_function: HeavisideStepFunction::AtanRegularized,
            use_image_spacing: true,
        }
    }
}

/// The label image together with the two measurements SimpleITK reports.
#[derive(Clone, Debug, PartialEq)]
pub struct ChanAndVeseResult {
    /// `1` where the final `phi < 0`, `0` elsewhere, in the input's pixel type.
    pub image: Image,
    /// `GetElapsedIterations()`.
    pub elapsed_iterations: u32,
    /// `GetRMSChange()`. `f64::MAX` if the solver halted before its first
    /// iteration, mirroring `m_RMSChange = NumericTraits<double>::max()`.
    pub rms_change: f64,
}

/// `HeavisideStepFunctionBase::Evaluate` / `EvaluateDerivative`, resolved to one
/// of the three concrete subclasses.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Domain {
    /// `AtanRegularizedHeavisideStepFunction`.
    Atan { one_over_epsilon: f64 },
    /// `SinRegularizedHeavisideStepFunction`.
    Sin { epsilon: f64, angle_factor: f64 },
    /// `HeavisideStepFunction`.
    Step,
}

impl Domain {
    /// `RegularizedHeavisideStepFunction::SetEpsilon` (hxx:25-35) throws unless
    /// `ieps > NumericTraits<RealType>::epsilon()`. `RealType` is `double` for
    /// both `float` and `double` pixels.
    fn new(kind: HeavisideStepFunction, epsilon: f64) -> Result<Self> {
        match kind {
            HeavisideStepFunction::Heaviside => Ok(Domain::Step),
            HeavisideStepFunction::AtanRegularized => {
                check_epsilon(epsilon)?;
                Ok(Domain::Atan {
                    one_over_epsilon: 1.0 / epsilon,
                })
            }
            HeavisideStepFunction::SinRegularized => {
                check_epsilon(epsilon)?;
                Ok(Domain::Sin {
                    epsilon,
                    angle_factor: 0.5 * std::f64::consts::PI / epsilon,
                })
            }
        }
    }

    fn evaluate(self, x: f64) -> f64 {
        match self {
            Domain::Atan { one_over_epsilon } => {
                0.5 + std::f64::consts::FRAC_1_PI * (x * one_over_epsilon).atan()
            }
            Domain::Sin {
                epsilon,
                angle_factor,
            } => {
                if x >= epsilon {
                    1.0
                } else if x <= -epsilon {
                    0.0
                } else {
                    0.5 * (1.0 + (x * angle_factor).sin())
                }
            }
            Domain::Step => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn evaluate_derivative(self, x: f64) -> f64 {
        match self {
            Domain::Atan { one_over_epsilon } => {
                let t = x * one_over_epsilon;
                std::f64::consts::FRAC_1_PI * one_over_epsilon / (1.0 + t * t)
            }
            Domain::Sin {
                epsilon,
                angle_factor,
            } => {
                if x.abs() >= epsilon {
                    0.0
                } else {
                    0.5 * angle_factor * (x * angle_factor).cos()
                }
            }
            Domain::Step => {
                if x == 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

fn check_epsilon(epsilon: f64) -> Result<()> {
    if epsilon > f64::EPSILON {
        Ok(())
    } else {
        Err(FilterError::InvalidHeavisideEpsilon(epsilon))
    }
}

/// `RegionBasedLevelSetFunction::GlobalDataStruct`, minus the three max-change
/// accumulators whose only consumer (`ComputeGlobalTimeStep`) has its result
/// discarded. Reused across pixels.
struct Derivatives {
    dim: usize,
    /// Central first differences, `0.5 * inv_spacing[i] * (phi(+e_i) - phi(-e_i))`.
    dx: Vec<f64>,
    dx_forward: Vec<f64>,
    dx_backward: Vec<f64>,
    /// Row-major `dim x dim` Hessian.
    dxy: Vec<f64>,
    grad_mag_sqr: f64,
    grad_mag: f64,
}

impl Derivatives {
    fn new(dim: usize) -> Self {
        Derivatives {
            dim,
            dx: vec![0.0; dim],
            dx_forward: vec![0.0; dim],
            dx_backward: vec![0.0; dim],
            dxy: vec![0.0; dim * dim],
            grad_mag_sqr: 0.0,
            grad_mag: 0.0,
        }
    }

    fn dxy(&self, i: usize, j: usize) -> f64 {
        self.dxy[i * self.dim + j]
    }

    fn set_dxy(&mut self, i: usize, j: usize, v: f64) {
        self.dxy[i * self.dim + j] = v;
    }
}

/// The level set's raster geometry plus the `ZeroFluxNeumannBoundaryCondition`
/// that `ConstNeighborhoodIterator` applies by default: an out-of-bounds
/// coordinate is clamped, per axis, to the nearest in-bounds one.
struct Grid<'a> {
    size: &'a [usize],
    strides: Vec<usize>,
}

impl Grid<'_> {
    fn coords_of(&self, p: usize) -> Vec<usize> {
        (0..self.size.len())
            .map(|d| (p / self.strides[d]) % self.size[d])
            .collect()
    }

    fn at_index(&self, coord: &[usize]) -> usize {
        coord.iter().zip(&self.strides).map(|(&c, &s)| c * s).sum()
    }

    /// `phi` at `coord + delta`, with each axis clamped into the image.
    fn at(&self, phi: &[f64], coord: &[usize], delta: &[i64]) -> f64 {
        let mut idx = 0usize;
        for (d, &stride) in self.strides.iter().enumerate() {
            let c = (coord[d] as i64 + delta[d]).clamp(0, self.size[d] as i64 - 1);
            idx += c as usize * stride;
        }
        phi[idx]
    }
}

fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `RegionBasedLevelSetFunction::ComputeCurvature` (hxx:161-190).
fn compute_curvature(gd: &Derivatives) -> f64 {
    let dim = gd.dim;
    let mut curvature = 0.0;

    for i in 0..dim {
        for j in 0..dim {
            if j != i {
                curvature -= gd.dx[i] * gd.dx[j] * gd.dxy(i, j);
                curvature += gd.dxy(j, j) * gd.dx[i] * gd.dx[i];
            }
        }
    }

    if gd.grad_mag > EPS {
        curvature /= gd.grad_mag * gd.grad_mag * gd.grad_mag;
    } else {
        curvature /= 1.0 + gd.grad_mag_sqr;
    }

    curvature
}

/// `RegionBasedLevelSetFunction::ComputeLaplacian` (hxx:318-329).
fn compute_laplacian(gd: &Derivatives) -> f64 {
    (0..gd.dim).map(|i| gd.dxy(i, i)).sum()
}

/// The `c_in`/`c_out` pair plus `sum_p H_p`, which the volume regularisation
/// term reads as `m_WeightedNumberOfPixelsInsideLevelSet`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Constants {
    c_in: f64,
    c_out: f64,
    weighted_pixels_inside: f64,
}

/// `ScalarChanAndVeseLevelSetFunction::ComputeParameters` (hxx:92-144) followed
/// by `UpdateSharedDataParameters` (hxx:29-54), specialised to one level set:
/// the nearest-neighbour list is `{0}` at every pixel, so the inner loop runs
/// once and the background product is `1 - H`.
fn compute_constants(feature: &[f64], heaviside: &[f64]) -> Constants {
    let mut weighted_pixels_inside = 0.0;
    let mut weighted_sum_inside = 0.0;
    let mut weighted_pixels_outside = 0.0;
    let mut weighted_sum_outside = 0.0;

    for (&f, &h) in feature.iter().zip(heaviside) {
        let prod = 1.0 - h;
        weighted_sum_inside += f * h;
        weighted_pixels_inside += h;
        weighted_sum_outside += f * prod;
        weighted_pixels_outside += prod;
    }

    Constants {
        c_in: if weighted_pixels_inside > EPS {
            weighted_sum_inside / weighted_pixels_inside
        } else {
            0.0
        },
        c_out: if weighted_pixels_outside > EPS {
            weighted_sum_outside / weighted_pixels_outside
        } else {
            0.0
        },
        weighted_pixels_inside,
    }
}

/// `RegionBasedLevelSetFunction::ComputeGlobalTerm` (hxx:347-378) with
/// `m_FunctionCount == 1`: `overlapTerm` is zero and `product` stays `1`.
fn compute_global_term(
    params: &ChanAndVeseParams,
    constants: &Constants,
    feature_value: f64,
) -> f64 {
    let interim = params.lambda1 * (feature_value - constants.c_in).powi(2);
    let out_term = params.lambda2 * (feature_value - constants.c_out).powi(2);

    // ComputeVolumeRegularizationTerm(): 2 * (weighted pixels inside - volume).
    let regularization =
        params.volume_matching_weight * 2.0 * (constants.weighted_pixels_inside - params.volume)
            - params.area_weight;

    interim - out_term + regularization
}

/// One iteration's frozen view of `ScalarChanAndVeseLevelSetFunction`: the level
/// set it differentiates, the feature image and its `m_InvSpacing`, the domain
/// function, the weights, and the `c_in`/`c_out` that `InitializeIteration` just
/// refreshed.
struct DifferenceFunction<'a> {
    grid: &'a Grid<'a>,
    phi: &'a [f64],
    feature: &'a [f64],
    /// `m_InvSpacing`, taken from the *feature* image by `SetFeatureImage`.
    inv_spacing: &'a [f64],
    domain: Domain,
    params: &'a ChanAndVeseParams,
    constants: Constants,
}

impl DifferenceFunction<'_> {
    /// `RegionBasedLevelSetFunction::ComputeHessian` (hxx:196-230).
    fn compute_hessian(&self, coord: &[usize], gd: &mut Derivatives) {
        let dim = gd.dim;
        let input_value = self.phi[self.grid.at_index(coord)];
        let mut delta = vec![0i64; dim];

        gd.grad_mag_sqr = 0.0;
        for i in 0..dim {
            delta[i] = 1;
            let a = self.grid.at(self.phi, coord, &delta);
            delta[i] = -1;
            let b = self.grid.at(self.phi, coord, &delta);
            delta[i] = 0;

            let inv = self.inv_spacing[i];
            gd.dx[i] = 0.5 * inv * (a - b);
            gd.dx_forward[i] = inv * (a - input_value);
            gd.dx_backward[i] = inv * (input_value - b);

            gd.grad_mag_sqr += gd.dx[i] * gd.dx[i];

            let dii = inv * (gd.dx_forward[i] - gd.dx_backward[i]);
            gd.set_dxy(i, i, dii);
        }

        for i in 0..dim {
            for j in (i + 1)..dim {
                let mut sample = |si: i64, sj: i64| {
                    delta[i] = si;
                    delta[j] = sj;
                    let v = self.grid.at(self.phi, coord, &delta);
                    delta[i] = 0;
                    delta[j] = 0;
                    v
                };
                // positionAa - positionBa + positionDa - positionCa
                let mixed = sample(-1, -1) - sample(-1, 1) + sample(1, 1) - sample(1, -1);
                let dij = 0.25 * self.inv_spacing[i] * self.inv_spacing[j] * mixed;
                gd.set_dxy(i, j, dij);
                gd.set_dxy(j, i, dij);
            }
        }

        gd.grad_mag = gd.grad_mag_sqr.sqrt();
    }

    /// `RegionBasedLevelSetFunction::ComputeUpdate` (hxx:234-314), minus the dead
    /// advection branch and the discarded max-change bookkeeping.
    fn compute_update(&self, coord: &[usize], gd: &mut Derivatives) -> f64 {
        let p = self.grid.at_index(coord);
        let input_value = self.phi[p];
        let params = self.params;

        self.compute_hessian(coord, gd);

        let dh = self.domain.evaluate_derivative(-input_value);

        // `curvature` is left at its zero initialiser when this branch is
        // skipped, and `laplacian_term` below still subtracts it. See the
        // module docs.
        let mut curvature = 0.0;
        let mut curvature_term = 0.0;
        if dh != 0.0 && params.curvature_weight != 0.0 {
            curvature = compute_curvature(gd);
            // CurvatureSpeed() is the inherited constant 1.
            curvature_term = params.curvature_weight * curvature * dh;
        }

        let mut laplacian_term = 0.0;
        if params.reinitialization_smoothing_weight != 0.0 {
            // LaplacianSmoothingSpeed() is the inherited constant 1.
            laplacian_term =
                (compute_laplacian(gd) - curvature) * params.reinitialization_smoothing_weight;
        }

        let global_term = if dh != 0.0 {
            dh * compute_global_term(params, &self.constants, self.feature[p])
        } else {
            0.0
        };

        curvature_term + laplacian_term + global_term
    }
}

/// `BinaryThresholdImageFilter` with
/// `[NumericTraits<InputPixelType>::NonpositiveMin(), 0]` as the inclusive
/// window, as `ApplyUpdate` configures it. For a float pixel type
/// `NonpositiveMin()` is `-max()`, so a `NaN` or `-inf` level set value falls
/// outside the window and is treated as background.
fn lowest_finite(pixel_id: PixelId) -> f64 {
    match pixel_id {
        PixelId::Float32 => -(f32::MAX as f64),
        _ => -f64::MAX,
    }
}

/// `ScalarChanAndVeseDenseLevelSetImageFilter` restricted to SimpleITK's
/// single-phase wiring.
///
/// `initial_level_set` is negative inside the initial region; `feature_image` is
/// the image being segmented. Both must be `Float32` or `Float64`, of the same
/// pixel type and the same size. The returned image is a `0`/`1` label image in
/// that pixel type — `1` where the final level set is strictly negative — not
/// the evolved level set.
///
/// Errors on a non-real pixel type, mismatched pixel types or sizes, and on a
/// non-positive `epsilon` when the Heaviside variant is regularised.
pub fn scalar_chan_and_vese_dense_level_set(
    initial_level_set: &Image,
    feature_image: &Image,
    params: &ChanAndVeseParams,
) -> Result<ChanAndVeseResult> {
    let pixel_id = initial_level_set.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    if feature_image.pixel_id() != pixel_id {
        return Err(FilterError::TypeMismatch {
            a: pixel_id,
            b: feature_image.pixel_id(),
        });
    }
    if initial_level_set.size() != feature_image.size() {
        return Err(FilterError::SizeMismatch {
            a: initial_level_set.size().to_vec(),
            b: feature_image.size().to_vec(),
        });
    }

    let domain = Domain::new(params.heaviside_step_function, params.epsilon)?;

    let size = initial_level_set.size();
    let grid = Grid {
        size,
        strides: strides(size),
    };
    let n = size.iter().product::<usize>();

    let mut phi = initial_level_set.to_f64_vec();
    let feature = feature_image.to_f64_vec();

    // SetFeatureImage() takes m_InvSpacing from the *feature* image, always.
    let inv_spacing: Vec<f64> = feature_image.spacing().iter().map(|s| 1.0 / s).collect();

    let mut heaviside = vec![0.0; n];
    let mut constants = initialize_iteration(&phi, &feature, domain, &mut heaviside);

    // GenerateData(): m_RMSChange = NumericTraits<double>::max() before the loop.
    let mut rms_change = f64::MAX;
    let mut elapsed_iterations = 0u32;

    let mut update = vec![0.0; n];
    let mut gd = Derivatives::new(size.len());

    // Halt(): elapsed >= NumberOfIterations || MaximumRMSError >= RMSChange.
    while elapsed_iterations < params.number_of_iterations && params.maximum_rms_error < rms_change
    {
        let df = DifferenceFunction {
            grid: &grid,
            phi: &phi,
            feature: &feature,
            inv_spacing: &inv_spacing,
            domain,
            params,
            constants,
        };
        for (p, u) in update.iter_mut().enumerate() {
            let coord = grid.coords_of(p);
            *u = df.compute_update(&coord, &mut gd);
        }

        rms_change = apply_update(
            &mut phi,
            &update,
            initial_level_set,
            pixel_id,
            params.use_image_spacing,
        )?;
        elapsed_iterations += 1;

        constants = initialize_iteration(&phi, &feature, domain, &mut heaviside);
    }

    // PostProcessOutput() -> CopyInputToOutput(): m_Lookup[0] == 1 where phi < 0.
    let labels: Vec<f64> = phi
        .iter()
        .map(|&v| if v < 0.0 { 1.0 } else { 0.0 })
        .collect();
    let image = image_from_f64(pixel_id, size, feature_image, &labels)?;

    Ok(ChanAndVeseResult {
        image,
        elapsed_iterations,
        rms_change,
    })
}

/// `ScalarChanAndVeseDenseLevelSetImageFilter::InitializeIteration`: recompute
/// `H(-phi)` for every level set (`UpdateSharedData(true)` -> `ComputeHImage`),
/// then recompute `c_in`/`c_out` from the fresh `H`
/// (`UpdateSharedData(false)` -> `ComputeParameters` + `UpdateSharedDataParameters`).
fn initialize_iteration(
    phi: &[f64],
    feature: &[f64],
    domain: Domain,
    heaviside: &mut [f64],
) -> Constants {
    for (h, &v) in heaviside.iter_mut().zip(phi) {
        // "Convention is inside of level-set function is negative."
        *h = domain.evaluate(-v);
    }
    compute_constants(feature, heaviside)
}

/// `MultiphaseDenseFiniteDifferenceImageFilter::ApplyUpdate` (hxx:176-249) with
/// `m_ReinitializeCounter == 1`: the reinitialisation branch always runs, so the
/// PDE's own RMS accumulator is always discarded and is not computed.
///
/// Returns the reported `RMSChange`.
fn apply_update(
    phi: &mut [f64],
    update: &[f64],
    level_set_geometry: &Image,
    pixel_id: PixelId,
    use_image_spacing: bool,
) -> Result<f64> {
    for (v, &u) in phi.iter_mut().zip(update) {
        *v += TIME_STEP * u;
    }

    let lower = lowest_finite(pixel_id);
    let inside: Vec<f64> = phi
        .iter()
        .map(|&v| if lower <= v && v <= 0.0 { 1.0 } else { 0.0 })
        .collect();
    let mut mask = Image::from_vec(level_set_geometry.size(), inside)?;
    mask.copy_geometry_from(level_set_geometry);

    let distance = signed_maurer_distance_map(&mask, false, false, use_image_spacing, 0.0)?;
    let distance = distance.to_f64_vec();

    let mut accumulator = 0.0;
    for (v, &d) in phi.iter_mut().zip(&distance) {
        accumulator += (*v - d).powi(2);
        *v = d;
    }

    Ok((accumulator / phi.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    /// `dh` at `phi == +/-1` with `epsilon == 1`: `1/pi * 1 / (1 + 1) `.
    const DH: f64 = 1.0 / (2.0 * PI);

    /// A 4x4 level set that is `-1` on the centred 2x2 block and `+1` elsewhere.
    fn fixture_phi() -> Vec<f64> {
        #[rustfmt::skip]
        let v = vec![
            1.0,  1.0,  1.0, 1.0,
            1.0, -1.0, -1.0, 1.0,
            1.0, -1.0, -1.0, 1.0,
            1.0,  1.0,  1.0, 1.0,
        ];
        v
    }

    /// The matching feature image: `10` on the block, `0` elsewhere.
    fn fixture_feature() -> Vec<f64> {
        #[rustfmt::skip]
        let v = vec![
            0.0,  0.0,  0.0, 0.0,
            0.0, 10.0, 10.0, 0.0,
            0.0, 10.0, 10.0, 0.0,
            0.0,  0.0,  0.0, 0.0,
        ];
        v
    }

    fn fixture_images() -> (Image, Image) {
        let phi = Image::from_vec(&[4, 4], fixture_phi()).unwrap();
        let feature = Image::from_vec(&[4, 4], fixture_feature()).unwrap();
        (phi, feature)
    }

    fn atan_constants() -> Constants {
        let phi = fixture_phi();
        let feature = fixture_feature();
        let domain = Domain::new(HeavisideStepFunction::AtanRegularized, 1.0).unwrap();
        let mut h = vec![0.0; 16];
        initialize_iteration(&phi, &feature, domain, &mut h)
    }

    fn update_at(coord: &[usize], params: &ChanAndVeseParams, inv_spacing: &[f64]) -> f64 {
        let phi = fixture_phi();
        let feature = fixture_feature();
        let domain = Domain::new(params.heaviside_step_function, params.epsilon).unwrap();
        let mut h = vec![0.0; 16];
        let constants = initialize_iteration(&phi, &feature, domain, &mut h);
        let size = [4usize, 4];
        let grid = Grid {
            size: &size,
            strides: strides(&size),
        };
        let df = DifferenceFunction {
            grid: &grid,
            phi: &phi,
            feature: &feature,
            inv_spacing,
            domain,
            params,
            constants,
        };
        df.compute_update(coord, &mut Derivatives::new(2))
    }

    // ---- Heaviside variants -------------------------------------------------

    #[test]
    fn atan_heaviside_matches_closed_form() {
        let d = Domain::new(HeavisideStepFunction::AtanRegularized, 1.0).unwrap();
        // H(1) = 1/2 + atan(1)/pi = 1/2 + (pi/4)/pi = 3/4.
        assert!((d.evaluate(1.0) - 0.75).abs() < 1e-15);
        assert!((d.evaluate(-1.0) - 0.25).abs() < 1e-15);
        assert!((d.evaluate(0.0) - 0.5).abs() < 1e-15);
        // H'(x) = 1/(pi eps (1 + (x/eps)^2)).
        assert!((d.evaluate_derivative(0.0) - 1.0 / PI).abs() < 1e-15);
        assert!((d.evaluate_derivative(1.0) - DH).abs() < 1e-15);
    }

    #[test]
    fn atan_heaviside_scales_with_epsilon() {
        let d = Domain::new(HeavisideStepFunction::AtanRegularized, 2.0).unwrap();
        assert!((d.evaluate(2.0) - 0.75).abs() < 1e-15);
        assert!((d.evaluate_derivative(0.0) - 1.0 / (2.0 * PI)).abs() < 1e-15);
    }

    #[test]
    fn sin_heaviside_matches_closed_form() {
        let d = Domain::new(HeavisideStepFunction::SinRegularized, 1.0).unwrap();
        // Saturates outside |x| < eps, inclusive of the endpoints.
        assert_eq!(d.evaluate(1.0), 1.0);
        assert_eq!(d.evaluate(2.0), 1.0);
        assert_eq!(d.evaluate(-1.0), 0.0);
        assert!((d.evaluate(0.0) - 0.5).abs() < 1e-15);
        // H(1/2) = (1 + sin(pi/4))/2.
        assert!((d.evaluate(0.5) - 0.5 * (1.0 + (PI / 4.0).sin())).abs() < 1e-15);
        // H'(0) = angle_factor / 2 = pi/4; H' vanishes at |x| >= eps.
        assert!((d.evaluate_derivative(0.0) - PI / 4.0).abs() < 1e-15);
        assert_eq!(d.evaluate_derivative(1.0), 0.0);
        assert_eq!(d.evaluate_derivative(-1.0), 0.0);
    }

    #[test]
    fn unregularized_heaviside_derivative_is_a_point_mass() {
        let d = Domain::new(HeavisideStepFunction::Heaviside, 1.0).unwrap();
        assert_eq!(d.evaluate(0.0), 1.0);
        assert_eq!(d.evaluate(1e-300), 1.0);
        assert_eq!(d.evaluate(-1e-300), 0.0);
        assert_eq!(d.evaluate_derivative(0.0), 1.0);
        assert_eq!(d.evaluate_derivative(1e-300), 0.0);
    }

    #[test]
    fn regularized_variants_reject_a_non_positive_epsilon() {
        for kind in [
            HeavisideStepFunction::AtanRegularized,
            HeavisideStepFunction::SinRegularized,
        ] {
            assert_eq!(
                Domain::new(kind, 0.0),
                Err(FilterError::InvalidHeavisideEpsilon(0.0))
            );
            assert_eq!(
                Domain::new(kind, f64::EPSILON),
                Err(FilterError::InvalidHeavisideEpsilon(f64::EPSILON))
            );
            assert!(Domain::new(kind, 2.0 * f64::EPSILON).is_ok());
        }
    }

    #[test]
    fn unregularized_heaviside_ignores_epsilon() {
        // SimpleITK's custom_itk_cast never calls SetEpsilon on this branch.
        assert!(Domain::new(HeavisideStepFunction::Heaviside, 0.0).is_ok());
        assert!(Domain::new(HeavisideStepFunction::Heaviside, -5.0).is_ok());
    }

    // ---- c_in / c_out -------------------------------------------------------

    #[test]
    fn constants_are_the_heaviside_weighted_means() {
        // H = 0.75 on the four block pixels (phi = -1), 0.25 on the twelve
        // others. cnt_in = 4(0.75) + 12(0.25) = 6, sum_in = 4(10)(0.75) = 30,
        // so c_in = 5. cnt_out = 4(0.25) + 12(0.75) = 10, sum_out = 4(10)(0.25)
        // = 10, so c_out = 1.
        let c = atan_constants();
        assert!((c.weighted_pixels_inside - 6.0).abs() < 1e-13);
        assert!((c.c_in - 5.0).abs() < 1e-13);
        assert!((c.c_out - 1.0).abs() < 1e-13);
    }

    #[test]
    fn unregularized_constants_are_the_plain_region_means() {
        // H is the indicator of phi <= 0, so c_in is the mean of the block and
        // c_out the mean of its complement.
        let phi = fixture_phi();
        let feature = fixture_feature();
        let domain = Domain::new(HeavisideStepFunction::Heaviside, 1.0).unwrap();
        let mut h = vec![0.0; 16];
        let c = initialize_iteration(&phi, &feature, domain, &mut h);
        assert_eq!(c.weighted_pixels_inside, 4.0);
        assert_eq!(c.c_in, 10.0);
        assert_eq!(c.c_out, 0.0);
    }

    #[test]
    fn empty_denominators_give_zero_constants() {
        // phi > 0 everywhere with the unregularized step: H == 0, so cnt_in == 0
        // and c_in falls back to 0 rather than dividing.
        let phi = vec![1.0; 4];
        let feature = vec![7.0; 4];
        let domain = Domain::new(HeavisideStepFunction::Heaviside, 1.0).unwrap();
        let mut h = vec![0.0; 4];
        let c = initialize_iteration(&phi, &feature, domain, &mut h);
        assert_eq!(c.c_in, 0.0);
        assert_eq!(c.c_out, 7.0);
    }

    // ---- ComputeUpdate ------------------------------------------------------

    #[test]
    fn update_inside_the_block_matches_the_hand_derived_value() {
        // At (1,1): dx = (-1,-1), dxy_00 = dxy_11 = 2, dxy_01 = -0.5,
        // |grad| = sqrt(2). curvature = 2 * (0.5 + 2) / (2 sqrt(2)) = 5 sqrt(2)/4.
        // curvature_term = 1 * (5 sqrt(2)/4) * dh.
        // global = 1*(10-5)^2 - 1*(10-1)^2 = -56, so global_term = -56 dh.
        // dh = 1/(2 pi).
        let params = ChanAndVeseParams::default();
        let expected = (5.0 * SQRT_2 / 4.0 - 56.0) * DH;
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
        // Sanity: the block's interior is pushed further negative, i.e. grows.
        assert!(got < 0.0);
    }

    #[test]
    fn update_at_a_corner_exercises_the_zero_flux_boundary() {
        // At (0,0) every first difference vanishes under the clamped
        // neighbourhood, so |grad| == 0 and ComputeCurvature takes the
        // `/(1 + gradMagSqr)` branch. The mixed derivative is -0.5 but is
        // multiplied by dx[i] dx[j] == 0, so curvature == 0.
        // global = (0-5)^2 - (0-1)^2 = 24, update = 24 dh = 12/pi.
        let params = ChanAndVeseParams::default();
        let expected = 12.0 / PI;
        let got = update_at(&[0, 0], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
        // Sanity: background pixels are pushed positive, i.e. stay outside.
        assert!(got > 0.0);
    }

    #[test]
    fn hessian_uses_the_feature_images_spacing() {
        // inv_spacing comes from SetFeatureImage. Halving it halves the
        // curvature (which has units of 1/length) and leaves the global term
        // alone: curvature = 5 sqrt(2)/8.
        let params = ChanAndVeseParams::default();
        let expected = (5.0 * SQRT_2 / 8.0 - 56.0) * DH;
        let got = update_at(&[1, 1], &params, &[0.5, 0.5]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn zero_curvature_weight_drops_the_curvature_term() {
        let params = ChanAndVeseParams {
            curvature_weight: 0.0,
            ..Default::default()
        };
        let expected = -56.0 * DH;
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn smoothing_term_subtracts_the_computed_curvature() {
        // laplacian at (1,1) is dxy_00 + dxy_11 = 4; curvature = 5 sqrt(2)/4.
        let params = ChanAndVeseParams {
            reinitialization_smoothing_weight: 1.0,
            ..Default::default()
        };
        let curvature = 5.0 * SQRT_2 / 4.0;
        let expected = curvature * DH + (4.0 - curvature) - 56.0 * DH;
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn smoothing_term_subtracts_zero_when_the_curvature_branch_is_skipped() {
        // The upstream quirk: `curvature` keeps its zero initialiser when
        // m_CurvatureWeight == 0, so the smoothing term is the bare Laplacian
        // (4 here) rather than Laplacian - curvature.
        let params = ChanAndVeseParams {
            curvature_weight: 0.0,
            reinitialization_smoothing_weight: 1.0,
            ..Default::default()
        };
        let expected = 4.0 - 56.0 * DH;
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn smoothing_term_subtracts_zero_when_dh_is_zero() {
        // Same quirk through the other half of the guard: the unregularized
        // step gives dh == 0 at phi == -1, so curvature is never computed and
        // the update is the bare Laplacian, with no global term at all.
        let params = ChanAndVeseParams {
            heaviside_step_function: HeavisideStepFunction::Heaviside,
            reinitialization_smoothing_weight: 1.0,
            ..Default::default()
        };
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - 4.0).abs() < 1e-12, "{got}");
    }

    #[test]
    fn volume_and_area_terms_enter_the_global_term() {
        // regularization = tau * 2 * (cnt_in - volume) - nu
        //               = 0.5 * 2 * (6 - 2) - 3 = 1.
        // global = 25 - 81 + 1 = -55.
        let params = ChanAndVeseParams {
            curvature_weight: 0.0,
            volume: 2.0,
            volume_matching_weight: 0.5,
            area_weight: 3.0,
            ..Default::default()
        };
        let expected = -55.0 * DH;
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn lambdas_scale_the_two_fidelity_terms() {
        let params = ChanAndVeseParams {
            curvature_weight: 0.0,
            lambda1: 2.0,
            lambda2: 0.5,
            ..Default::default()
        };
        // 2*(10-5)^2 - 0.5*(10-1)^2 = 50 - 40.5 = 9.5.
        let expected = 9.5 * DH;
        let got = update_at(&[1, 1], &params, &[1.0, 1.0]);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn unregularized_heaviside_freezes_every_pixel_off_the_zero_level() {
        // dh == 0 wherever phi != 0, and the fixture has no zero pixel.
        let params = ChanAndVeseParams {
            heaviside_step_function: HeavisideStepFunction::Heaviside,
            ..Default::default()
        };
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(update_at(&[x, y], &params, &[1.0, 1.0]), 0.0);
            }
        }
    }

    // ---- end to end ---------------------------------------------------------

    #[test]
    fn zero_iterations_labels_the_initial_level_set_and_reports_max_rms() {
        let (phi, feature) = fixture_images();
        let params = ChanAndVeseParams {
            number_of_iterations: 0,
            ..Default::default()
        };
        let out = scalar_chan_and_vese_dense_level_set(&phi, &feature, &params).unwrap();
        assert_eq!(out.elapsed_iterations, 0);
        assert_eq!(out.rms_change, f64::MAX);
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(out.image.to_f64_vec(), expected);
        assert_eq!(out.image.pixel_id(), PixelId::Float64);
    }

    #[test]
    fn one_frozen_iteration_is_a_pure_re_distancing() {
        // With the unregularized step every update is 0, so phi is unchanged
        // and ApplyUpdate replaces it with the signed distance map of
        // {phi <= 0} = the 2x2 block. All four block pixels touch the
        // background, so each is a boundary seed at distance 0 (negated to
        // -0.0); the eight edge-adjacent background pixels sit at 1 and the
        // four corners at sqrt(2).
        //
        // rms = sqrt( [4 * (-1 - 0)^2 + 4 * (1 - sqrt(2))^2 + 8 * (1 - 1)^2] / 16 )
        let (phi, feature) = fixture_images();
        let params = ChanAndVeseParams {
            heaviside_step_function: HeavisideStepFunction::Heaviside,
            number_of_iterations: 1,
            ..Default::default()
        };
        let out = scalar_chan_and_vese_dense_level_set(&phi, &feature, &params).unwrap();
        assert_eq!(out.elapsed_iterations, 1);

        let expected_rms = ((4.0 + 4.0 * (SQRT_2 - 1.0).powi(2)) / 16.0).sqrt();
        assert!(
            (out.rms_change - expected_rms).abs() < 1e-12,
            "{} vs {expected_rms}",
            out.rms_change
        );

        // CopyInputToOutput tests `phi < 0` strictly, and every surviving block
        // pixel now holds -0.0, which is not less than zero.
        assert_eq!(out.image.to_f64_vec(), vec![0.0; 16]);
    }

    #[test]
    fn use_image_spacing_scales_the_re_distancing() {
        // The same frozen iteration on a 2x2-spaced grid. With the flag on, the
        // distance map is in physical units: edge-adjacent background pixels sit
        // at 2 and corners at 2 sqrt(2). With it off, the map ignores spacing
        // and reproduces the unit-spacing result exactly.
        let (mut phi, mut feature) = fixture_images();
        phi.set_spacing(&[2.0, 2.0]).unwrap();
        feature.set_spacing(&[2.0, 2.0]).unwrap();

        let base = ChanAndVeseParams {
            heaviside_step_function: HeavisideStepFunction::Heaviside,
            number_of_iterations: 1,
            ..Default::default()
        };

        let on = scalar_chan_and_vese_dense_level_set(&phi, &feature, &base).unwrap();
        let expected_on = ((4.0 + 4.0 * (2.0 * SQRT_2 - 1.0).powi(2) + 8.0 * 1.0) / 16.0).sqrt();
        assert!(
            (on.rms_change - expected_on).abs() < 1e-12,
            "{} vs {expected_on}",
            on.rms_change
        );

        let off = scalar_chan_and_vese_dense_level_set(
            &phi,
            &feature,
            &ChanAndVeseParams {
                use_image_spacing: false,
                ..base
            },
        )
        .unwrap();
        let expected_off = ((4.0 + 4.0 * (SQRT_2 - 1.0).powi(2)) / 16.0).sqrt();
        assert!(
            (off.rms_change - expected_off).abs() < 1e-12,
            "{} vs {expected_off}",
            off.rms_change
        );
    }

    #[test]
    fn the_default_rms_threshold_never_halts_the_solver() {
        // The reported RMS measures the re-distancing, not the PDE, so it does
        // not decay towards maximum_rms_error. Ten requested iterations run ten
        // iterations.
        let (phi, feature) = fixture_images();
        let params = ChanAndVeseParams {
            number_of_iterations: 10,
            ..Default::default()
        };
        let out = scalar_chan_and_vese_dense_level_set(&phi, &feature, &params).unwrap();
        assert_eq!(out.elapsed_iterations, 10);
        assert!(out.rms_change > params.maximum_rms_error);
    }

    #[test]
    fn a_large_maximum_rms_error_halts_before_the_first_iteration() {
        let (phi, feature) = fixture_images();
        let params = ChanAndVeseParams {
            maximum_rms_error: f64::MAX,
            ..Default::default()
        };
        let out = scalar_chan_and_vese_dense_level_set(&phi, &feature, &params).unwrap();
        assert_eq!(out.elapsed_iterations, 0);
        assert_eq!(out.rms_change, f64::MAX);
    }

    #[test]
    fn float32_input_gives_a_float32_label_image() {
        let phi = Image::from_vec(
            &[4, 4],
            fixture_phi()
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        )
        .unwrap();
        let feature = Image::from_vec(
            &[4, 4],
            fixture_feature()
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        )
        .unwrap();
        let params = ChanAndVeseParams {
            number_of_iterations: 0,
            ..Default::default()
        };
        let out = scalar_chan_and_vese_dense_level_set(&phi, &feature, &params).unwrap();
        assert_eq!(out.image.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn one_dimensional_images_have_no_curvature() {
        // ComputeCurvature's j != i loop is empty in 1-D, so the curvature is
        // zero regardless of the level set's shape and only the global term
        // survives.
        let phi = vec![1.0, -1.0, -1.0, 1.0];
        let feature = vec![0.0, 10.0, 10.0, 0.0];
        let params = ChanAndVeseParams::default();
        let domain = Domain::new(params.heaviside_step_function, params.epsilon).unwrap();
        let mut h = vec![0.0; 4];
        let constants = initialize_iteration(&phi, &feature, domain, &mut h);
        // cnt_in = 2(0.75) + 2(0.25) = 2, sum_in = 2(10)(0.75) = 15 -> c_in = 7.5.
        // cnt_out = 2(0.25) + 2(0.75) = 2, sum_out = 2(10)(0.25) = 5 -> c_out = 2.5.
        assert!((constants.c_in - 7.5).abs() < 1e-13);
        assert!((constants.c_out - 2.5).abs() < 1e-13);

        let size = [4usize];
        let grid = Grid {
            size: &size,
            strides: strides(&size),
        };
        let df = DifferenceFunction {
            grid: &grid,
            phi: &phi,
            feature: &feature,
            inv_spacing: &[1.0],
            domain,
            params: &params,
            constants,
        };
        let got = df.compute_update(&[1], &mut Derivatives::new(1));
        // global = (10-7.5)^2 - (10-2.5)^2 = 6.25 - 56.25 = -50.
        let expected = -50.0 * DH;
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    // ---- input validation ---------------------------------------------------

    #[test]
    fn mismatched_sizes_are_rejected() {
        let phi = Image::from_vec(&[4, 4], fixture_phi()).unwrap();
        let feature = Image::from_vec(&[2, 2], vec![0.0; 4]).unwrap();
        assert_eq!(
            scalar_chan_and_vese_dense_level_set(&phi, &feature, &ChanAndVeseParams::default()),
            Err(FilterError::SizeMismatch {
                a: vec![4, 4],
                b: vec![2, 2]
            })
        );
    }

    #[test]
    fn mismatched_pixel_types_are_rejected() {
        let phi = Image::from_vec(&[4, 4], fixture_phi()).unwrap();
        let feature = Image::from_vec(&[4, 4], vec![0.0f32; 16]).unwrap();
        assert_eq!(
            scalar_chan_and_vese_dense_level_set(&phi, &feature, &ChanAndVeseParams::default()),
            Err(FilterError::TypeMismatch {
                a: PixelId::Float64,
                b: PixelId::Float32
            })
        );
    }

    #[test]
    fn integer_pixel_types_are_rejected() {
        let phi = Image::from_vec(&[4, 4], vec![0i16; 16]).unwrap();
        let feature = Image::from_vec(&[4, 4], vec![0i16; 16]).unwrap();
        assert_eq!(
            scalar_chan_and_vese_dense_level_set(&phi, &feature, &ChanAndVeseParams::default()),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
    }

    #[test]
    fn a_bad_epsilon_is_rejected_end_to_end() {
        let (phi, feature) = fixture_images();
        let params = ChanAndVeseParams {
            epsilon: -1.0,
            ..Default::default()
        };
        assert_eq!(
            scalar_chan_and_vese_dense_level_set(&phi, &feature, &params),
            Err(FilterError::InvalidHeavisideEpsilon(-1.0))
        );
    }
}
