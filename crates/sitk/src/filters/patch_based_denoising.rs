//! ITK's `PatchBasedDenoisingImageFilter`, ported from
//! `Modules/Filtering/Denoising/include/itkPatchBasedDenoisingImageFilter.h`/`.hxx`
//! over its base class `itkPatchBasedDenoisingBaseImageFilter.h`/`.hxx`, with
//! the patch sampler from
//! `Modules/Numerics/Statistics/include/itkGaussianRandomSpatialNeighborSubsampler.h`/`.hxx`
//! (whose `Search` is inherited verbatim from
//! `itkUniformRandomSpatialNeighborSubsampler.hxx`). The public parameter
//! surface and its defaults come from SimpleITK's
//! `Code/BasicFilters/yaml/PatchBasedDenoisingImageFilter.yaml`.
//!
//! Each pixel is replaced by a step along the gradient of the joint entropy of
//! image patches: for the patch `P(x)` around the pixel and a random subset
//! `{P(sᵢ)}` of patches drawn near it,
//!
//! ```text
//! g(x)  = Σᵢ (I(sᵢ) − I(x)) · exp(−d(P(x), P(sᵢ))² / (2σ²))
//!         ──────────────────────────────────────────────────
//!               Σᵢ exp(−d(P(x), P(sᵢ))² / (2σ²)) + ε
//!
//! I'(x) = I(x) + 0.2 · SmoothingWeight · g(x)   (+ a noise-model fidelity term)
//! ```
//!
//! where `d` is the patch-weight-masked Euclidean distance, `σ` is
//! `kernel_bandwidth_sigma`, and `ε` is `MinProbability`. The whole image is
//! updated out-of-place per iteration (ITK's `m_UpdateBuffer` + `ApplyUpdate`),
//! so within one iteration every patch is read from the previous iteration's
//! image.
//!
//! # Scalar pixels only
//!
//! This crate's [`Image`] is scalar, so `NumPixelComponents` and
//! `NumIndependentComponents` are both 1 and `ComponentSpace` is always
//! `EUCLIDEAN`. Upstream additionally handles `RGBPixel`, `RGBAPixel`,
//! `Vector`, `VectorImage` and `DiffusionTensor3D`; the tensor case switches
//! `DetermineComponentSpace` to `RIEMANNIAN` and replaces the signed Euclidean
//! difference with a log-map geodesic difference
//! (`ComputeLogMapAndWeightedSquaredGeodesicDifference`), the exponential-map
//! update (`AddExponentialMapUpdate`), the threaded Riemannian min/max pass,
//! and the `UseFastTensorComputations` 3x3 eigen-analysis. None of that is
//! reachable here. `always_treat_components_as_euclidean` is therefore
//! accepted (it is part of the yaml's public surface) but cannot change the
//! result: `DetermineComponentSpace` already returns `EUCLIDEAN` for every
//! scalar pixel type. The RIEMANNIAN-only warning path in `EnforceConstraints`
//! that silently zeroes `NoiseModelFidelityWeight` is likewise unreachable.
//!
//! # Determinism and the random sampler
//!
//! SimpleITK hard-wires a `GaussianRandomSpatialNeighborSubsampler`, so the
//! patch subset is random. The seed *is* reachable and fixed:
//! `InitializeIteration` clones the sampler once per work unit and calls
//! `sampler->SetSeed(thread)` (`itkPatchBasedDenoisingImageFilter.hxx:1319`),
//! reseeding that clone's Mersenne Twister at the start of *every* iteration.
//! This port runs the image as a single work unit, so the sampler is reseeded
//! with `0` each iteration and the output is bit-reproducible — and it equals
//! ITK's own **single-work-unit** output. ITK's default multi-threaded run
//! splits the region and seeds each piece with its own work-unit id, giving a
//! different (also machine-dependent, since the split depends on the work-unit
//! count) result; that divergence is inherent to ITK's design and is the same
//! policy [`crate::filters::noise`] follows.
//!
//! Within an iteration the sampler's stream is consumed in a fixed order: the
//! kernel-bandwidth Newton iterations first (when
//! `kernel_bandwidth_estimation` is on), then the image update, walking
//! `ImageBoundaryFacesCalculator`'s face list — interior region first, then the
//! low/high boundary face of each axis in axis order. [`boundary_faces`]
//! reproduces that ordering because it determines which draws each pixel gets.
//!
//! # Boundary handling
//!
//! The `ListAdaptorType` declares a `ZeroFluxNeumannBoundaryCondition`, but no
//! boundary value ever reaches the arithmetic. `ComputeGradientJointEntropy`
//! records which of the query patch's taps are in bounds and sums only those
//! (`itkPatchBasedDenoisingImageFilter.hxx:2246`), and the per-pixel region
//! constraint `[min(x, r), max(x, size−r−1)]` guarantees a sampled patch is at
//! least as in-bounds as the query patch, so `selectedPatch.GetPixel(jj)` is
//! only ever read at taps the query patch also has in bounds. A truncated
//! patch is therefore *not* weighted back up: it simply accumulates fewer
//! squared-difference terms, so its distance to every candidate is smaller and
//! its kernel weight larger. Kernel-bandwidth estimation avoids the boundary
//! entirely by iterating only the interior face.
//!
//! # Reproduced upstream quirks
//!
//! - Integer promotion. `ComputeSignedEuclideanDifferenceAndWeightedSquaredNorm`
//!   and the three fidelity terms subtract/multiply two `PixelValueType`
//!   values *before* widening to `double`. For pixel types narrower than
//!   `int` that promotes to `int` and is exact; for `uint32`/`int32`/`uint64`/
//!   `int64` it wraps in the pixel type. [`PatchPixel::sub_f64`] and
//!   [`PatchPixel::mul_f64`] reproduce this per type rather than computing in
//!   `f64` throughout.
//! - `NoiseSigma` is consumed by the `RICIAN` model only. `GAUSSIAN` and
//!   `POISSON` never read it, so setting it changes nothing for them.
//! - `probJointEntropyFirstDerivative` and `...SecondDerivative` each get
//!   `MinProbability` added to them in `ThreadedComputeSigmaUpdate`, though
//!   they are derivatives and not probabilities.
//! - The sigma-update convergence test compares `|sigmaUpdate|` against the
//!   *already-updated* sigma, because `ResolveSigmaUpdate` writes
//!   `m_KernelBandwidthSigma` before returning the update it is tested against.
//!
//! # Deliberate divergences
//!
//! - `static_cast<PixelType>(result)` on an out-of-range `double` is undefined
//!   behaviour in C++. [`crate::core::Scalar::from_f64`] saturates instead.
//! - `POISSON`'s step size is
//!   `std::min(outVal, static_cast<PixelValueType>(0.99999)) + 0.00001`.
//!   Upstream casts `0.99999` to the pixel type *before* the `min`, so
//!   `static_cast<PixelValueType>(0.99999)` is **0** for every integer pixel
//!   type and the step collapses to `1e-5` regardless of the pixel value.
//!   [`poisson_step_size`] takes the `min` in real arithmetic, so an integer
//!   pixel of value `v ≥ 1` gets the intended saturated step `0.99999 + 0.00001`.
//! - `GaussianRandomSpatialNeighborSubsampler::GetIntegerVariate` casts a
//!   possibly-negative `std::floor(randVar)` to `unsigned int` (undefined
//!   behaviour) and relies on the wrapped value failing the `> upperBound`
//!   test. [`GaussianSampler::integer_variate`] rejects out-of-range variates
//!   directly, which agrees with the wrapping implementations for every
//!   variate a `sqrt(SampleVariance)`-scaled normal can realistically produce.
//! - When `CanSelectQuery` is off (the kernel-bandwidth pass) and the search
//!   box degenerates to the query pixel alone, upstream's
//!   `while (pointsFound < numberOfPoints)` never terminates. This port returns
//!   an empty patch set instead, which makes `ThreadedComputeSigmaUpdate`'s
//!   `0 / 0` produce a `NaN` that its own `probJointEntropy > 0` assertion
//!   rejects: the filter fails with [`FilterError::NoPatchesSampled`] rather
//!   than hanging. See [`GaussianSampler::search`].

use crate::core::{Image, Scalar, dispatch_scalar};
use crate::filters::denoise::{modified_bessel_i0, modified_bessel_i1};
use crate::filters::error::{FilterError, Result};
use crate::filters::random::MersenneTwister;

/// `PatchBasedDenoisingBaseImageFilterEnums::NoiseModel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseModel {
    /// No fidelity term at all. SimpleITK's yaml default.
    #[default]
    NoModel,
    /// `∂/∂u ‖u − f‖²`: `gradient = 2·(in − out)`, step size `0.5`.
    Gaussian,
    /// Rician likelihood, `gradient = (in·I₁(α)/I₀(α) − out)/σ²` with
    /// `α = in·out/σ²` and step size `σ²`. The only model that reads
    /// `noise_sigma`. Result is clamped to be nonnegative.
    Rician,
    /// Poisson likelihood, `gradient = (in − out)/(out + 1e-5)`. Result is
    /// clamped to be at least `1e-5`.
    Poisson,
}

/// Parameters of [`patch_based_denoising`], defaulting to SimpleITK's
/// `PatchBasedDenoisingImageFilter.yaml`.
///
/// The yaml hides `SmoothingWeight`, `UseSmoothDiscPatchWeights`,
/// `ComputeConditionalDerivatives`, `UseFastTensorComputations` and the patch
/// weights array; those stay at their ITK constructor defaults (`1.0`, on,
/// off, on, and the smooth disc respectively) and are not exposed here either.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchBasedDenoisingSettings {
    /// Gaussian kernel bandwidth `σ` in intensity units, the initial estimate
    /// when `kernel_bandwidth_estimation` is on and a fixed value otherwise.
    ///
    /// SimpleITK always calls `SetKernelBandwidthSigma`, so ITK's
    /// `InitializeKernelSigma` (30% of the intensity range) is unreachable and
    /// is not ported.
    pub kernel_bandwidth_sigma: f64,
    /// Patch radius in *physical* units. Converted per axis to
    /// `ceil(max_spacing · patch_radius / spacing[d])` voxels, so the patch is
    /// isotropic in physical space and may be anisotropic in voxel space.
    pub patch_radius: u32,
    /// Denoising iterations. Clamped to at least 1 (`itkSetClampMacro`).
    pub number_of_iterations: u32,
    /// Patches drawn per pixel. Upstream draws *with replacement*, and draws
    /// `min(this, points in the search box)` of them — so a small search box
    /// caps the count regardless of this value.
    pub number_of_sample_patches: u32,
    /// Variance of the Gaussian the sampler draws patch offsets from. Also
    /// fixes the sampler's search radius, which SimpleITK derives as
    /// `floor(sqrt(sample_variance) · 2.5)` voxels on every axis.
    pub sample_variance: f64,
    /// Which fidelity term to add. Only used when
    /// `noise_model_fidelity_weight > 0`.
    pub noise_model: NoiseModel,
    /// Noise standard deviation for [`NoiseModel::Rician`]. `0.0` means "use
    /// ITK's default", 5% of the image intensity range — SimpleITK's
    /// `custom_itk_cast` skips `SetNoiseSigma` entirely when the value is 0.
    pub noise_sigma: f64,
    /// Weight of the fidelity term. Clamped to `[0, 1]` (`itkSetClampMacro`).
    /// Zero (the default) disables the noise model completely.
    pub noise_model_fidelity_weight: f64,
    /// Accepted for parity with the yaml; cannot change a scalar-pixel result.
    /// See the module doc.
    pub always_treat_components_as_euclidean: bool,
    /// Re-estimate `σ` from the data by leave-one-out cross validation.
    /// SimpleITK's yaml default is `false` even though its own doc string says
    /// "Defaults to true"; ITK's constructor calls `KernelBandwidthEstimationOff()`.
    pub kernel_bandwidth_estimation: bool,
    /// Multiplies the estimated `σ`. Clamped to `[0.01, 100]`. Used only when
    /// `kernel_bandwidth_estimation` is on.
    pub kernel_bandwidth_multiplication_factor: f64,
    /// Re-estimate `σ` on iterations whose index is a multiple of this.
    /// Clamped to at least 1.
    pub kernel_bandwidth_update_frequency: u32,
    /// Fraction of interior pixels used for the estimation. Clamped to
    /// `[0.01, 1.0]`; converted to a decimation stride `round(1/fraction)`.
    pub kernel_bandwidth_fraction_pixels_for_estimation: f64,
}

impl Default for PatchBasedDenoisingSettings {
    /// SimpleITK's yaml defaults: `KernelBandwidthSigma = 400.0`,
    /// `PatchRadius = 4`, `NumberOfIterations = 1`,
    /// `NumberOfSamplePatches = 200`, `SampleVariance = 400.0`,
    /// `NoiseModel = NOMODEL`, `NoiseSigma = 0.0`,
    /// `NoiseModelFidelityWeight = 0.0`,
    /// `AlwaysTreatComponentsAsEuclidean = false`,
    /// `KernelBandwidthEstimation = false`,
    /// `KernelBandwidthMultiplicationFactor = 1.0`,
    /// `KernelBandwidthUpdateFrequency = 3`,
    /// `KernelBandwidthFractionPixelsForEstimation = 0.2`.
    fn default() -> Self {
        Self {
            kernel_bandwidth_sigma: 400.0,
            patch_radius: 4,
            number_of_iterations: 1,
            number_of_sample_patches: 200,
            sample_variance: 400.0,
            noise_model: NoiseModel::NoModel,
            noise_sigma: 0.0,
            noise_model_fidelity_weight: 0.0,
            always_treat_components_as_euclidean: false,
            kernel_bandwidth_estimation: false,
            kernel_bandwidth_multiplication_factor: 1.0,
            kernel_bandwidth_update_frequency: 3,
            kernel_bandwidth_fraction_pixels_for_estimation: 0.2,
        }
    }
}

/// `m_MinSigma` / `m_MinProbability`: `NumericTraits<RealValueType>::min() * 100`,
/// with `RealValueType = NumericTraits<PixelValueType>::RealType`, which is
/// `double` for *every* scalar pixel type including `float`
/// (`itkNumericTraits.h:1356`). Present "to avoid divide by zero".
const MIN_SIGMA: f64 = f64::MIN_POSITIVE * 100.0;
const MIN_PROBABILITY: f64 = f64::MIN_POSITIVE * 100.0;

/// `m_SmoothingWeight`, not exposed by the yaml.
const SMOOTHING_WEIGHT: f64 = 1.0;
/// `stepSizeSmoothing` in `ThreadedComputeImageUpdate`.
const STEP_SIZE_SMOOTHING: f64 = 0.2;
/// `m_SigmaUpdateConvergenceTolerance`.
const SIGMA_UPDATE_CONVERGENCE_TOLERANCE: f64 = 0.01;
/// `MaxSigmaUpdateIterations`.
const MAX_SIGMA_UPDATE_ITERATIONS: usize = 20;

// ---- pixel-typed arithmetic -----------------------------------------------

/// The three places ITK's scalar path performs arithmetic in `PixelValueType`
/// rather than in `RealValueType`, plus the `static_cast` that narrows the
/// result back. Implemented per type so that C++'s integer promotion — exact
/// for types narrower than `int`, wrapping for the rest — is reproduced.
trait PatchPixel: Scalar {
    /// `static_cast<double>(a - b)` with `a`, `b` of the pixel type.
    fn sub_f64(a: Self, b: Self) -> f64;

    /// `static_cast<double>(a * b)` with `a`, `b` of the pixel type.
    fn mul_f64(a: Self, b: Self) -> f64;
}

/// Types whose C++ integer promotion is to `int`: the arithmetic is exact for
/// subtraction and, for the 8-bit types, for multiplication too. `u16 * u16`
/// can overflow `int` (undefined in C++); the wrapping `i32` here is what
/// every mainstream compiler emits.
macro_rules! impl_patch_pixel_promotes_to_int {
    ($($t:ty),+ $(,)?) => {$(
        impl PatchPixel for $t {
            fn sub_f64(a: Self, b: Self) -> f64 {
                (i32::from(a) - i32::from(b)) as f64
            }
            fn mul_f64(a: Self, b: Self) -> f64 {
                i32::from(a).wrapping_mul(i32::from(b)) as f64
            }
        }
    )+};
}
impl_patch_pixel_promotes_to_int!(u8, i8, u16, i16);

/// Types at or above `int` rank: C++ arithmetic stays in the pixel type and
/// wraps (modularly for the unsigned types; undefined, but wrapping in
/// practice, for the signed ones).
macro_rules! impl_patch_pixel_wrapping {
    ($($t:ty),+ $(,)?) => {$(
        impl PatchPixel for $t {
            fn sub_f64(a: Self, b: Self) -> f64 {
                a.wrapping_sub(b) as f64
            }
            fn mul_f64(a: Self, b: Self) -> f64 {
                a.wrapping_mul(b) as f64
            }
        }
    )+};
}
impl_patch_pixel_wrapping!(u32, i32, u64, i64);

impl PatchPixel for f32 {
    fn sub_f64(a: Self, b: Self) -> f64 {
        f64::from(a - b)
    }
    fn mul_f64(a: Self, b: Self) -> f64 {
        f64::from(a * b)
    }
}

impl PatchPixel for f64 {
    fn sub_f64(a: Self, b: Self) -> f64 {
        a - b
    }
    fn mul_f64(a: Self, b: Self) -> f64 {
        a * b
    }
}

/// `POISSON`'s step-size clamp, `std::min(outVal, 0.99999) + 0.00001`, taken in
/// real arithmetic. Upstream computes `std::min(outVal,
/// static_cast<PixelValueType>(0.99999))`, casting `0.99999` to the pixel type
/// before the `min`; for every integer pixel type that cast truncates to `0`,
/// so the clamp is `min(outVal, 0) == 0` for all non-negative `outVal` and the
/// step collapses to `1e-5` no matter how bright the pixel is. Clamping the
/// literal `0.99999` in `f64` gives the intended saturated step `1e-5 + 0.99999`
/// for any `outVal ≥ 0.99999`, so an integer pixel of value `v ≥ 1` now steps by
/// the same amount a float pixel of value `1.0` would.
fn poisson_step_size(output: f64) -> f64 {
    output.min(0.99999) + 0.00001
}

// ---- geometry helpers ------------------------------------------------------

/// An image lattice: axis sizes plus the first-index-fastest strides that turn
/// a multi-index into the linear offset ITK calls `ComputeOffset`.
struct Grid {
    size: Vec<usize>,
    strides: Vec<usize>,
}

impl Grid {
    fn new(size: &[usize]) -> Self {
        let mut strides = vec![1usize; size.len()];
        for d in 1..size.len() {
            strides[d] = strides[d - 1] * size[d - 1];
        }
        Grid {
            size: size.to_vec(),
            strides,
        }
    }

    fn dim(&self) -> usize {
        self.size.len()
    }

    fn offset(&self, index: &[isize]) -> usize {
        index
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| *i as usize * s)
            .sum()
    }
}

/// An `itk::ImageRegion`: start index and size.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Region {
    index: Vec<isize>,
    size: Vec<usize>,
}

impl Region {
    /// Every index in the region, in first-index-fastest raster order — the
    /// order `ImageRegionIterator` and `ImageToNeighborhoodSampleAdaptor`'s
    /// `ConstIterator` both walk, which is what pairs a sampled patch with its
    /// output pixel.
    fn indices(&self) -> impl Iterator<Item = Vec<isize>> + '_ {
        let total: usize = self.size.iter().product();
        (0..total).map(move |n| {
            let mut rem = n;
            self.index
                .iter()
                .zip(&self.size)
                .map(|(start, len)| {
                    let c = rem % len;
                    rem /= len;
                    start + c as isize
                })
                .collect()
        })
    }
}

/// `NeighborhoodAlgorithm::ImageBoundaryFacesCalculator` applied to the whole
/// image: the non-boundary (interior) region first, then, for each axis in
/// order, its low face and its high face. Faces do not overlap.
///
/// Transcribed from `itkNeighborhoodAlgorithm.hxx:29` with `regionToProcess`
/// equal to the buffered region. Only the ordering matters to the result, via
/// the order in which pixels consume the sampler's random stream.
fn boundary_faces(size: &[usize], radius: &[usize]) -> Vec<Region> {
    let dim = size.len();
    let mut faces = Vec::new();

    let mut nb_size = size.to_vec();
    let mut nb_start = vec![0isize; dim];
    let mut vr_start = vec![0isize; dim];
    let mut vr_size = size.to_vec();

    for i in 0..dim {
        // rStart == bStart == 0 and rSize == bSize == size, so both overlaps
        // reduce to -radius[i] (the `bSize > 2 * radius` branch is the live
        // one: `initialize` has already rejected any shorter axis).
        let mut overlap_low = -(radius[i] as isize);
        let mut overlap_high = if size[i] > 2 * radius[i] {
            -(radius[i] as isize)
        } else {
            radius[i] as isize - size[i] as isize
        };

        if overlap_low < 0 {
            let mut f_start = vec![0isize; dim];
            let mut f_size = vec![0usize; dim];
            for j in 0..dim {
                f_start[j] = vr_start[j];
                if j == i {
                    if -overlap_low > size[i] as isize {
                        overlap_low = -(size[i] as isize);
                    }
                    f_size[j] = (-overlap_low) as usize;
                    vr_size[j] = (vr_size[j] as isize + overlap_low) as usize;
                    vr_start[j] -= overlap_low;
                } else {
                    f_size[j] = vr_size[j];
                }
                f_size[j] = f_size[j].min(size[j]);
            }
            nb_size[i] = nb_size[i].saturating_sub(f_size[i]);
            nb_start[i] += -overlap_low;
            faces.push(Region {
                index: f_start,
                size: f_size,
            });
        }

        if overlap_high < 0 {
            let mut f_start = vec![0isize; dim];
            let mut f_size = vec![0usize; dim];
            for j in 0..dim {
                if j == i {
                    if -overlap_high > size[i] as isize {
                        overlap_high = -(size[i] as isize);
                    }
                    f_start[j] = size[j] as isize + overlap_high;
                    f_size[j] = (-overlap_high) as usize;
                    vr_size[j] = (vr_size[j] as isize + overlap_high) as usize;
                } else {
                    f_start[j] = vr_start[j];
                    f_size[j] = vr_size[j];
                }
            }
            nb_size[i] = nb_size[i].saturating_sub(f_size[i]);
            faces.push(Region {
                index: f_start,
                size: f_size,
            });
        }
    }

    let mut all = vec![Region {
        index: nb_start,
        size: nb_size,
    }];
    all.extend(faces);
    all
}

/// `GetPatchRadiusInVoxels`: `ceil(maxSpacing · patchRadius / spacing[d])`.
fn patch_radius_in_voxels(patch_radius: u32, spacing: &[f64]) -> Vec<usize> {
    let max_spacing = spacing.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    spacing
        .iter()
        .map(|s| (max_spacing * f64::from(patch_radius) / s).ceil() as usize)
        .collect()
}

// ---- smooth-disc patch weights --------------------------------------------

/// `InitializePatchWeightsSmoothDisc`: build the isotropic disc mask on an
/// isotropic `(2·patch_radius+1)^D` grid whose spacing is the image's *largest*
/// spacing, then resample it — identity transform, linear interpolation, zero
/// outside — onto the anisotropic voxel-space patch grid.
///
/// Values are `float` upstream (`itk::Image<float, D>`), and the interpolator
/// accumulates in `double`; both are mirrored, since the mask is clamped into
/// `[0, 1]` and then divided by its own centre value, which magnifies the last
/// bits.
///
/// With isotropic spacing the resample is the identity and the result is the
/// analytic mask: `1` inside `‖x‖ ≤ patch_radius/2`, `0` outside
/// `‖x‖ ≥ patch_radius+1`, and the cubic Hermite ramp
/// `−2t³/L³ + 3t²/L²` in between, with `t = (patch_radius+1) − ‖x‖` and
/// `L = (patch_radius+1) − patch_radius/2`.
fn smooth_disc_patch_weights(
    patch_radius: u32,
    spacing: &[f64],
    radius_vox: &[usize],
) -> Result<Vec<f32>> {
    let dim = spacing.len();
    let max_spacing = spacing.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let physical_diameter = 2 * patch_radius as usize + 1;
    let physical_size = vec![physical_diameter; dim];
    let physical_grid = Grid::new(&physical_size);

    let disc_radius = patch_radius / 2;
    let interval = f64::from((patch_radius + 1) - disc_radius);
    let outer = f64::from(patch_radius + 1);

    let physical: Vec<f32> = Region {
        index: vec![0; dim],
        size: physical_size.clone(),
    }
    .indices()
    .map(|idx| {
        let sq: f64 = idx
            .iter()
            .map(|c| {
                let v = (*c - patch_radius as isize) as f64;
                v * v
            })
            .sum();
        // `Vector<float, D>::GetNorm()` accumulates and takes the root in
        // `double`, then narrows into the `const float distanceFromCenter`.
        let distance = sq.sqrt() as f32;
        if f64::from(distance) >= outer {
            0.0f32
        } else if f64::from(distance) <= f64::from(disc_radius) {
            1.0f32
        } else {
            // `(patchRadius + 1) - distanceFromCenter` is `unsigned - float`,
            // so the subtraction and both `pow(·, 3.0f)`/`pow(·, 2.0f)` calls
            // are in `float`; the `-2.0 / pow(interval, 3.0)` coefficients are
            // in `double`, and the sum narrows back to `float`.
            let t = (outer as f32) - distance;
            let weight = ((-2.0 / interval.powf(3.0)) * f64::from(t.powf(3.0))
                + (3.0 / interval.powf(2.0)) * f64::from(t.powf(2.0)))
                as f32;
            weight.clamp(0.0, 1.0)
        }
    })
    .collect();

    // Resample onto the voxel-space patch grid.
    let voxel_size: Vec<usize> = radius_vox.iter().map(|r| 2 * r + 1).collect();
    let voxel_region = Region {
        index: vec![0; dim],
        size: voxel_size.clone(),
    };

    let mut weights: Vec<f32> = Vec::with_capacity(voxel_size.iter().product());
    for out_idx in voxel_region.indices() {
        // Identity transform, both images at origin 0 with identity direction:
        // the continuous input index is just the physical point over the
        // physical mask's (isotropic) spacing.
        let cont: Vec<f64> = out_idx
            .iter()
            .zip(spacing)
            .map(|(o, s)| (*o as f64 * s) / max_spacing)
            .collect();

        // `ImageFunction::IsInsideBuffer`: `start - 0.5 <= c < end + 0.5`.
        let inside = cont
            .iter()
            .all(|c| *c >= -0.5 && *c < physical_diameter as f64 - 0.5);
        let value = if inside {
            linear_interpolate(&physical, &physical_grid, &cont)
        } else {
            0.0
        };
        weights.push(value.clamp(0.0, 1.0));
    }

    let length = weights.len();
    let center_weight = weights[(length - 1) / 2];
    if center_weight != 1.0 {
        if center_weight <= 0.0 {
            return Err(FilterError::PatchCenterWeightNotPositive(f64::from(
                center_weight,
            )));
        }
        for w in &mut weights {
            *w /= center_weight;
        }
    }
    Ok(weights)
}

/// `LinearInterpolateImageFunction::Evaluate*`: the `2^D` corner sum with the
/// base index floored (never below the start index) and each upper neighbour
/// clamped to the end index. The dimension-specialized `EvaluateOptimized`
/// overloads short-circuit on a zero distance instead of clamping, which is
/// the same value; and their `distance <= 0` early-return differs from the
/// generic form only for a continuous index below the start, which the caller
/// here never produces (`cont >= 0`).
fn linear_interpolate(image: &[f32], grid: &Grid, cont: &[f64]) -> f32 {
    let dim = grid.dim();
    let base: Vec<isize> = cont.iter().map(|c| c.floor() as isize).collect();
    let dist: Vec<f64> = cont
        .iter()
        .zip(&base)
        .map(|(c, b)| *c - *b as f64)
        .collect();

    let mut value = 0.0f64;
    for corner in 0..(1usize << dim) {
        let mut overlap = 1.0f64;
        let mut index = Vec::with_capacity(dim);
        for d in 0..dim {
            let end = grid.size[d] as isize - 1;
            if corner & (1 << d) != 0 {
                index.push((base[d] + 1).clamp(0, end));
                overlap *= dist[d];
            } else {
                index.push(base[d].clamp(0, end));
                overlap *= 1.0 - dist[d];
            }
        }
        value += f64::from(image[grid.offset(&index)]) * overlap;
    }
    value as f32
}

// ---- the patch sampler -----------------------------------------------------

/// `Statistics::GaussianRandomSpatialNeighborSubsampler`, whose `Search` comes
/// unchanged from `UniformRandomSpatialNeighborSubsampler`. The sample region
/// is always the whole image, so instance identifiers are plain linear offsets.
struct GaussianSampler {
    rng: MersenneTwister,
    variance: f64,
    radius: Vec<usize>,
    number_of_results_requested: usize,
}

impl GaussianSampler {
    /// `SetSeed(seed)` on the clone: re-initializes the Mersenne Twister.
    fn reseed(&mut self, seed: u32) {
        self.rng = MersenneTwister::new(seed);
    }

    /// `GaussianRandomSpatialNeighborSubsampler::GetIntegerVariate`: reject
    /// normal deviates whose floor leaves `[lower, upper]`.
    fn integer_variate(&mut self, lower: isize, upper: isize, mean: isize) -> isize {
        loop {
            let variate = self.rng.get_normal_variate(mean as f64, self.variance);
            let floored = variate.floor();
            if floored >= lower as f64 && floored <= upper as f64 {
                return floored as isize;
            }
        }
    }

    /// `UniformRandomSpatialNeighborSubsampler::Search`. Draws
    /// `min(number_of_results_requested, points in the search box)` patch
    /// centres **with replacement** — duplicates are kept, and each is a
    /// separate term in the joint-entropy sums.
    ///
    /// Upstream loops until it has that many *accepted* draws. With
    /// `can_select_query` off and a search box holding nothing but the query,
    /// no draw is ever accepted and the loop hangs; this port returns an empty
    /// result there instead.
    fn search(
        &mut self,
        grid: &Grid,
        query: &[isize],
        constraint: &Region,
        can_select_query: bool,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let dim = grid.dim();
        let mut start = vec![0isize; dim];
        let mut end = vec![0isize; dim];
        let mut box_points: usize = 1;
        for d in 0..dim {
            let radius = self.radius[d] as isize;
            start[d] = if query[d] < radius {
                constraint.index[d].max(0)
            } else {
                (query[d] - radius).max(constraint.index[d])
            };
            let constraint_end = constraint.index[d] + constraint.size[d] as isize;
            end[d] = if query[d] + radius < constraint_end {
                query[d] + radius
            } else {
                constraint_end - 1
            };
            box_points *= (end[d] - start[d] + 1) as usize;
        }

        if !can_select_query && box_points == 1 && start == query {
            return;
        }

        let wanted = box_points.min(self.number_of_results_requested);
        let mut index = vec![0isize; dim];
        while out.len() < wanted {
            for d in 0..dim {
                index[d] = self.integer_variate(start[d], end[d], query[d]);
            }
            if can_select_query || index != query {
                out.push(grid.offset(&index));
            }
        }
    }
}

// ---- the filter ------------------------------------------------------------

/// Everything `Initialize` computes once and every threaded method then reads.
struct Denoiser<'a, T: PatchPixel> {
    input: &'a [T],
    grid: Grid,
    radius: Vec<usize>,
    /// Linear offset of each patch tap, in first-index-fastest order.
    tap_offset: Vec<isize>,
    /// Per-axis index offset of each patch tap, for the in-bounds test.
    tap_index_offset: Vec<Vec<isize>>,
    /// `m_PatchWeights`, `float` upstream.
    patch_weights: Vec<f32>,
    length_patch: usize,
    center: usize,
    /// `m_IntensityRescaleInvFactor` = `100 / (max − min)`.
    intensity_rescale_inv_factor: f64,
    /// `m_NoiseSigmaSquared`.
    noise_sigma_squared: f64,
    /// `m_SigmaUpdateDecimationFactor`.
    sigma_decimation: usize,
    /// `m_TotalNumberPixels`.
    total_pixels: f64,
    fidelity_weight: f64,
    noise_model: NoiseModel,
    faces: Vec<Region>,
}

impl<T: PatchPixel> Denoiser<'_, T> {
    /// `true` when every tap of the patch centred at `index` lies inside the
    /// image — `ConstNeighborhoodIterator::InBounds`, whose inner bounds are
    /// taken from the image's *buffered* region.
    fn patch_in_bounds(&self, index: &[isize]) -> bool {
        index.iter().enumerate().all(|(d, i)| {
            *i >= self.radius[d] as isize && *i < (self.grid.size[d] - self.radius[d]) as isize
        })
    }

    fn tap_in_bounds(&self, index: &[isize], tap: usize) -> bool {
        self.tap_index_offset[tap].iter().enumerate().all(|(d, o)| {
            let p = index[d] + o;
            p >= 0 && p < self.grid.size[d] as isize
        })
    }

    /// The per-pixel `RegionConstraint`,
    /// `[min(x_d, r_d), max(x_d, size_d − r_d − 1)]`, which keeps every sampled
    /// patch at least as in-bounds as the query patch.
    fn region_constraint(&self, index: &[isize]) -> Region {
        let dim = self.grid.dim();
        let mut r_index = vec![0isize; dim];
        let mut r_size = vec![0usize; dim];
        for d in 0..dim {
            let radius = self.radius[d] as isize;
            r_index[d] = index[d].min(radius);
            let hi = index[d].max(self.grid.size[d] as isize - radius - 1);
            r_size[d] = (hi - r_index[d] + 1) as usize;
        }
        Region {
            index: r_index,
            size: r_size,
        }
    }

    /// `ComputeGradientJointEntropy`. Returns the smoothing update direction,
    /// `Σ (I(sᵢ) − I(x))·gᵢ / (Σ gᵢ + MinProbability)`.
    fn gradient_joint_entropy(
        &self,
        output: &[T],
        index: &[isize],
        kernel_sigma: f64,
        sampler: &mut GaussianSampler,
        selected: &mut Vec<usize>,
    ) -> f64 {
        let constraint = self.region_constraint(index);
        sampler.search(&self.grid, index, &constraint, true, selected);

        let offset = self.grid.offset(index) as isize;
        let in_bounds = self.patch_in_bounds(index);
        let taps_in_bounds: Vec<bool> = if in_bounds {
            Vec::new()
        } else {
            (0..self.length_patch)
                .map(|jj| self.tap_in_bounds(index, jj))
                .collect()
        };

        let mut sum_of_gaussians = 0.0f64;
        let mut gradient = 0.0f64;

        for &selected_offset in selected.iter() {
            let selected_offset = selected_offset as isize;
            let mut squared_norm = 0.0f64;

            // The partial unrolling pairs tap `jj` with tap `center + 1 + jj`,
            // which fixes the summation order and hence the rounding.
            for (jj, kk) in (0..self.center).zip(self.center + 1..) {
                for tap in [jj, kk] {
                    if in_bounds || taps_in_bounds[tap] {
                        squared_norm += self.weighted_squared_norm(
                            output,
                            offset,
                            selected_offset,
                            tap,
                            f64::from(self.patch_weights[tap]),
                        );
                    }
                }
            }

            // The centre tap is always in bounds.
            let center_difference = T::sub_f64(
                output[(selected_offset + self.tap_offset[self.center]) as usize],
                output[(offset + self.tap_offset[self.center]) as usize],
            );
            let center_weight = f64::from(self.patch_weights[self.center]);
            squared_norm += center_weight * center_weight * center_difference * center_difference;

            let distance = squared_norm / (kernel_sigma * kernel_sigma);
            let gaussian = (-distance / 2.0).exp();
            sum_of_gaussians += gaussian;
            gradient += center_difference * gaussian;
        }

        gradient / (sum_of_gaussians + MIN_PROBABILITY)
    }

    /// One term of `ComputeSignedEuclideanDifferenceAndWeightedSquaredNorm`:
    /// `(w · (selected − current))²`.
    fn weighted_squared_norm(
        &self,
        image: &[T],
        offset: isize,
        selected_offset: isize,
        tap: usize,
        weight: f64,
    ) -> f64 {
        let difference = T::sub_f64(
            image[(selected_offset + self.tap_offset[tap]) as usize],
            image[(offset + self.tap_offset[tap]) as usize],
        );
        weight * weight * difference * difference
    }

    /// `ThreadedComputeSigmaUpdate` for a single work unit: the first and
    /// second derivatives of the joint entropy with respect to `σ`, summed over
    /// every `sigma_decimation`-th interior pixel.
    ///
    /// `m_ComputeConditionalDerivatives` is off by default and not exposed by
    /// the yaml, so the neighbourhood-entropy correction is not ported.
    fn compute_sigma_derivatives(
        &self,
        output: &[T],
        kernel_sigma: f64,
        sampler: &mut GaussianSampler,
    ) -> Result<(f64, f64)> {
        let mut first_derivative = 0.0f64;
        let mut second_derivative = 0.0f64;
        let mut selected = Vec::new();
        let length_patch = self.length_patch as f64;
        let weight = self.intensity_rescale_inv_factor;

        // Only the interior face: patches there are entirely in bounds.
        for (sample_num, index) in self.faces[0].indices().enumerate() {
            if sample_num % self.sigma_decimation != 0 {
                continue;
            }
            let constraint = self.region_constraint(&index);
            sampler.search(&self.grid, &index, &constraint, false, &mut selected);
            let num_patches = selected.len() as f64;

            let offset = self.grid.offset(&index) as isize;
            let mut prob = 0.0f64;
            let mut prob_first = 0.0f64;
            let mut prob_second = 0.0f64;

            for &selected_offset in &selected {
                let selected_offset = selected_offset as isize;
                let mut squared_norm = 0.0f64;
                for (jj, kk) in (0..self.center).zip(self.center + 1..) {
                    for tap in [jj, kk] {
                        squared_norm += self.weighted_squared_norm(
                            output,
                            offset,
                            selected_offset,
                            tap,
                            weight,
                        );
                    }
                }
                squared_norm += self.weighted_squared_norm(
                    output,
                    offset,
                    selected_offset,
                    self.center,
                    weight,
                );

                let distance = squared_norm.sqrt();
                let gaussian = (-(distance / kernel_sigma).powi(2) / 2.0).exp();
                prob += gaussian;

                // `pow(sigmaKernel, 3.0)`, not `sigma*sigma*sigma`: libm's `pow`
                // and the repeated product can differ in the last bit.
                let factor = squared_norm / kernel_sigma.powf(3.0) - (length_patch / kernel_sigma);
                prob_first += gaussian * factor;
                prob_second += gaussian
                    * (factor * factor + (length_patch / (kernel_sigma * kernel_sigma))
                        - (3.0 * squared_norm / kernel_sigma.powf(4.0)));
            }

            prob = prob / num_patches + MIN_PROBABILITY;
            prob_first = prob_first / num_patches + MIN_PROBABILITY;
            prob_second = prob_second / num_patches + MIN_PROBABILITY;

            // `itkAssertOrThrowMacro(probJointEntropy[ic] > 0.0)`: a `NaN`, which
            // is what `0 / 0` leaves behind when no patch was drawn, fails it too.
            if prob.is_nan() || prob <= 0.0 {
                return Err(FilterError::NoPatchesSampled);
            }

            first_derivative -= prob_first / prob;
            second_derivative -= prob_second / prob - (prob_first / prob).powi(2);
        }

        Ok((first_derivative, second_derivative))
    }

    /// `ResolveSigmaUpdate`: a damped Newton-Raphson step on `σ`, falling back
    /// to gradient descent when the second derivative is not positive, capped
    /// at 30% of `σ`, and floored at `MinSigma`. Returns the (uncapped-by-the-
    /// floor) update that `ComputeKernelBandwidthUpdate` tests for convergence,
    /// and writes the new `σ`.
    fn resolve_sigma_update(&self, first: f64, second: f64, kernel_sigma: &mut f64) -> f64 {
        let first = first / self.total_pixels;
        let second = second / self.total_pixels;
        let sigma = *kernel_sigma;

        let mut update = if second.abs() == 0.0 || second < 0.0 {
            -sgn(first) * sigma * 0.3
        } else {
            -first / second
        };
        if update.abs() > sigma * 0.3 {
            update = sgn(update) * sigma * 0.3;
        }
        *kernel_sigma = if sigma + update < MIN_SIGMA {
            (sigma + MIN_SIGMA) / 2.0
        } else {
            sigma + update
        };
        update
    }

    /// `ComputeKernelBandwidthUpdate`: rescale `σ` to an intensity range of
    /// 100, iterate Newton-Raphson to convergence (or 20 steps), then undo the
    /// rescale. Every Newton step redraws patches, advancing the sampler.
    fn compute_kernel_bandwidth_update(
        &self,
        output: &[T],
        kernel_sigma: &mut f64,
        multiplication_factor: f64,
        sampler: &mut GaussianSampler,
    ) -> Result<()> {
        *kernel_sigma = *kernel_sigma / multiplication_factor * self.intensity_rescale_inv_factor;

        for _ in 0..MAX_SIGMA_UPDATE_ITERATIONS {
            let (first, second) = self.compute_sigma_derivatives(output, *kernel_sigma, sampler)?;
            let update = self.resolve_sigma_update(first, second, kernel_sigma);
            if update.abs() < *kernel_sigma * SIGMA_UPDATE_CONVERGENCE_TOLERANCE {
                break;
            }
        }

        *kernel_sigma = *kernel_sigma / self.intensity_rescale_inv_factor * multiplication_factor;
        Ok(())
    }

    /// `ThreadedComputeImageUpdate` + `ApplyUpdate`: fill the update buffer for
    /// every pixel, face by face, then swap it in.
    fn compute_image_update(
        &self,
        output: &[T],
        update: &mut [T],
        kernel_sigma: f64,
        sampler: &mut GaussianSampler,
    ) {
        let mut selected = Vec::new();
        for face in &self.faces {
            for index in face.indices() {
                let offset = self.grid.offset(&index);
                let mut result = output[offset].as_f64();

                if SMOOTHING_WEIGHT > 0.0 {
                    let gradient = self.gradient_joint_entropy(
                        output,
                        &index,
                        kernel_sigma,
                        sampler,
                        &mut selected,
                    );
                    result += gradient * (SMOOTHING_WEIGHT * STEP_SIZE_SMOOTHING);
                }

                if self.fidelity_weight > 0.0 {
                    result = self.apply_fidelity(result, self.input[offset], output[offset]);
                }

                update[offset] = T::from_f64(result);
            }
        }
    }

    /// The `switch (this->GetNoiseModel())` block of `ThreadedComputeImageUpdate`.
    fn apply_fidelity(&self, result: f64, input: T, output: T) -> f64 {
        let weight = self.fidelity_weight;
        match self.noise_model {
            NoiseModel::NoModel => result,
            NoiseModel::Gaussian => {
                let gradient = 2.0 * T::sub_f64(input, output);
                result + weight * (0.5 * gradient)
            }
            NoiseModel::Rician => {
                let sigma_squared = self.noise_sigma_squared;
                let alpha = T::mul_f64(input, output) / sigma_squared;
                let gradient = (input.as_f64()
                    * (modified_bessel_i1(alpha) / modified_bessel_i0(alpha))
                    - output.as_f64())
                    / sigma_squared;
                (result + weight * (sigma_squared * gradient)).max(0.0)
            }
            NoiseModel::Poisson => {
                let gradient = T::sub_f64(input, output) / (output.as_f64() + 0.00001);
                let step = poisson_step_size(output.as_f64());
                (result + weight * (step * gradient)).max(0.00001)
            }
        }
    }
}

/// `itk::Math::sgn`: `-1`, `0` or `1`.
fn sgn(v: f64) -> f64 {
    ((0.0 < v) as i32 - (v < 0.0) as i32) as f64
}

/// `Math::Round<int64_t>` == `RoundHalfIntegerUp` == `floor(x + 0.5)`.
fn round_half_up(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

/// Denoise `img` by iterative non-local weighted averaging of image patches
/// (Awate & Whitaker 2005/2006).
///
/// Reproduces ITK's **single-work-unit** output exactly; see the module doc for
/// the RNG policy, the scalar-pixel restriction, and the upstream quirks kept.
///
/// Errors with:
/// - [`FilterError::PatchLargerThanImage`] when an axis is shorter than the
///   patch diameter in voxels (`Initialize`);
/// - [`FilterError::ConstantImage`] for a constant image, whose intensity
///   range would divide by zero (`EnforceConstraints`);
/// - [`FilterError::NegativeIntensityForNoiseModel`] for a negative pixel under
///   [`NoiseModel::Rician`] or [`NoiseModel::Poisson`] (`EnforceConstraints`);
/// - [`FilterError::KernelBandwidthSigmaTooSmall`] when
///   `kernel_bandwidth_sigma <= MinSigma` (`Initialize`);
/// - [`FilterError::InvalidSampleVariance`] for a negative `sample_variance`,
///   whose square root SimpleITK feeds to `Math::Floor<unsigned int>`;
/// - [`FilterError::PatchCenterWeightNotPositive`] when the resampled
///   smooth-disc mask has a nonpositive centre
///   (`InitializePatchWeightsSmoothDisc`);
/// - [`FilterError::NoPatchesSampled`] when kernel-bandwidth estimation runs
///   with `number_of_sample_patches == 0` (`ThreadedComputeSigmaUpdate`'s
///   `itkAssertOrThrowMacro`).
pub fn patch_based_denoising(img: &Image, settings: &PatchBasedDenoisingSettings) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), run, img, settings)
}

fn run<T: PatchPixel>(img: &Image, settings: &PatchBasedDenoisingSettings) -> Result<Image> {
    if settings.sample_variance < 0.0 {
        return Err(FilterError::InvalidSampleVariance(settings.sample_variance));
    }

    // `itkSetClampMacro` clamps silently rather than rejecting.
    let iterations = settings.number_of_iterations.max(1);
    let update_frequency = settings.kernel_bandwidth_update_frequency.max(1);
    let fidelity_weight = settings.noise_model_fidelity_weight.clamp(0.0, 1.0);
    let multiplication_factor = settings
        .kernel_bandwidth_multiplication_factor
        .clamp(0.01, 100.0);
    let fraction = settings
        .kernel_bandwidth_fraction_pixels_for_estimation
        .clamp(0.01, 1.0);

    let size = img.size().to_vec();
    let grid = Grid::new(&size);
    let dim = grid.dim();
    let radius = patch_radius_in_voxels(settings.patch_radius, img.spacing());

    // `Initialize`: the image must hold at least one whole patch on every axis.
    let diameter: Vec<usize> = radius.iter().map(|r| 2 * r + 1).collect();
    if (0..dim).any(|d| size[d] < diameter[d]) {
        return Err(FilterError::PatchLargerThanImage { size, diameter });
    }

    let input: &[T] = img.scalar_slice::<T>()?;
    let (image_min, image_max) = min_max(input);
    if image_max <= image_min {
        return Err(FilterError::ConstantImage(image_max.as_f64()));
    }
    if matches!(
        settings.noise_model,
        NoiseModel::Rician | NoiseModel::Poisson
    ) && image_min.as_f64() < 0.0
    {
        return Err(FilterError::NegativeIntensityForNoiseModel(
            image_min.as_f64(),
        ));
    }
    if settings.kernel_bandwidth_sigma <= MIN_SIGMA {
        return Err(FilterError::KernelBandwidthSigmaTooSmall(
            settings.kernel_bandwidth_sigma,
            MIN_SIGMA,
        ));
    }

    let intensity_rescale_inv_factor = 100.0 / T::sub_f64(image_max, image_min);

    // `SetNoiseSigma` is skipped by SimpleITK when the value is 0, leaving
    // ITK's own default of 5% of the intensity range.
    let noise_sigma = if settings.noise_sigma != 0.0 {
        settings.noise_sigma
    } else {
        5.0 / intensity_rescale_inv_factor
    };

    let total_pixels = img.number_of_pixels();
    let decimation = round_half_up(1.0 / fraction)
        .min(round_half_up(total_pixels as f64 / 100.0))
        .max(1) as usize;

    let patch_weights = smooth_disc_patch_weights(settings.patch_radius, img.spacing(), &radius)?;
    let length_patch = patch_weights.len();

    let mut tap_offset = Vec::with_capacity(length_patch);
    let mut tap_index_offset = Vec::with_capacity(length_patch);
    let tap_region = Region {
        index: vec![0; dim],
        size: diameter.clone(),
    };
    for tap in tap_region.indices() {
        let offsets: Vec<isize> = tap
            .iter()
            .zip(&radius)
            .map(|(t, r)| *t - *r as isize)
            .collect();
        tap_offset.push(
            offsets
                .iter()
                .zip(&grid.strides)
                .map(|(o, s)| *o * *s as isize)
                .sum(),
        );
        tap_index_offset.push(offsets);
    }

    let denoiser = Denoiser::<T> {
        input,
        grid,
        radius: radius.clone(),
        tap_offset,
        tap_index_offset,
        patch_weights,
        length_patch,
        center: (length_patch - 1) / 2,
        intensity_rescale_inv_factor,
        noise_sigma_squared: noise_sigma * noise_sigma,
        sigma_decimation: decimation,
        total_pixels: total_pixels as f64,
        fidelity_weight,
        noise_model: settings.noise_model,
        faces: boundary_faces(&size, &radius),
    };

    let mut sampler = GaussianSampler {
        rng: MersenneTwister::new(0),
        variance: settings.sample_variance,
        // SimpleITK: `SetRadius(Math::Floor<unsigned int>(sqrt(variance) * 2.5))`.
        radius: vec![(settings.sample_variance.sqrt() * 2.5).floor() as usize; dim],
        number_of_results_requested: settings.number_of_sample_patches as usize,
    };

    // `CopyInputToOutput`.
    let mut output: Vec<T> = input.to_vec();
    let mut update: Vec<T> = output.clone();
    let mut kernel_sigma = settings.kernel_bandwidth_sigma;

    for elapsed in 0..iterations {
        // `InitializeIteration` reseeds each work unit's sampler clone with its
        // work-unit id; a single work unit is always id 0.
        sampler.reseed(0);

        if settings.kernel_bandwidth_estimation && elapsed % update_frequency == 0 {
            denoiser.compute_kernel_bandwidth_update(
                &output,
                &mut kernel_sigma,
                multiplication_factor,
                &mut sampler,
            )?;
        }
        denoiser.compute_image_update(&output, &mut update, kernel_sigma, &mut sampler);
        output.copy_from_slice(&update);
    }

    let mut out = Image::from_vec(&denoiser.grid.size, output)?;
    out.copy_geometry_from(img);
    Ok(out)
}

/// `MinimumMaximumImageFilter` over the input, in the pixel type.
fn min_max<T: Scalar>(data: &[T]) -> (T, T) {
    let mut min = data[0];
    let mut max = data[0];
    for &v in &data[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 5x5 `f64` fixture every numeric expectation below is derived from.
    const IMG: [f64; 25] = [
        10.0, 20.0, 30.0, 40.0, 50.0, //
        15.0, 25.0, 35.0, 45.0, 55.0, //
        12.0, 22.0, 32.0, 42.0, 52.0, //
        18.0, 28.0, 38.0, 48.0, 58.0, //
        11.0, 21.0, 31.0, 41.0, 51.0,
    ];

    /// `patch_radius = 1`, `kernel_bandwidth_sigma = 25`,
    /// `number_of_sample_patches = 4`, `sample_variance = 1` (so the sampler
    /// radius is `floor(sqrt(1)*2.5) = 2`).
    fn settings() -> PatchBasedDenoisingSettings {
        PatchBasedDenoisingSettings {
            kernel_bandwidth_sigma: 25.0,
            patch_radius: 1,
            number_of_sample_patches: 4,
            sample_variance: 1.0,
            ..Default::default()
        }
    }

    fn fixture() -> Image {
        Image::from_vec(&[5, 5], IMG.to_vec()).unwrap()
    }

    fn values(img: &Image) -> Vec<f64> {
        img.scalar_slice::<f64>().unwrap().to_vec()
    }

    #[track_caller]
    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tol, "index {i}: got {a:.17}, want {e:.17}");
        }
    }

    // ---- hand-derived, no RNG involved ------------------------------------

    /// `patch_radius = 1` gives `discRadius = 0`, `interval = 2`, so the ramp is
    /// `w(d) = -d̄³/4 + 3d̄²/4` with `d̄ = 2 - d`:
    /// `w(0) = 1`, `w(1) = -1/4 + 3/4 = 1/2`, and
    /// `w(√2) = (2-√2)²·(3 - (2-√2))/4 = (√2 - 1)/2 = 0.20710678…`.
    /// The centre weight is exactly 1, so no renormalization happens. The
    /// diagonal weight lands one `f32` ulp above the exactly-rounded value,
    /// because ITK evaluates `(patchRadius + 1) - distanceFromCenter` and both
    /// `pow` calls in `float`.
    #[test]
    fn smooth_disc_patch_weights_radius_one_isotropic_matches_the_cubic_ramp() {
        let w = smooth_disc_patch_weights(1, &[1.0, 1.0], &[1, 1]).unwrap();
        let c = ((2.0f64.sqrt() - 1.0) / 2.0) as f32;
        assert_eq!(w[4], 1.0);
        for (got, want) in w.iter().zip(&[c, 0.5, c, 0.5, 1.0, 0.5, c, 0.5, c]) {
            assert!((got - want).abs() < 1e-7, "got {got}, want {want}");
        }
    }

    /// 1-D, `patch_radius = 2`: `discRadius = 1`, so `|d| <= 1` is weight 1 and
    /// `|d| = 2` lands on the ramp at `d̄ = 1`, giving `-1/4 + 3/4 = 1/2`.
    #[test]
    fn smooth_disc_patch_weights_radius_two_has_a_flat_inner_disc() {
        let w = smooth_disc_patch_weights(2, &[1.0], &[2]).unwrap();
        assert_eq!(w, vec![0.5, 1.0, 1.0, 1.0, 0.5]);
    }

    /// Anisotropic spacing `[1, 2]` with `patch_radius = 1`: the voxel patch is
    /// `5x3` (`ceil(2·1/1) = 2` and `ceil(2·1/2) = 1`), and the physical `3x3`
    /// mask, whose spacing is `maxSpacing = 2`, is sampled at continuous indices
    /// `x ∈ {0, .5, 1, 1.5, 2}`, `y ∈ {0, 1, 2}`. Every sample is inside the
    /// buffer, so linear interpolation just halves the neighbours along `x`.
    #[test]
    fn smooth_disc_patch_weights_anisotropic_spacing_resamples_the_physical_mask() {
        let radius = patch_radius_in_voxels(1, &[1.0, 2.0]);
        assert_eq!(radius, vec![2, 1]);

        let w = smooth_disc_patch_weights(1, &[1.0, 2.0], &radius).unwrap();
        let c = ((2.0f64.sqrt() - 1.0) / 2.0) as f32;
        let h = (c + 0.5) / 2.0;
        let expected = vec![
            c, h, 0.5, h, c, //
            0.5, 0.75, 1.0, 0.75, 0.5, //
            c, h, 0.5, h, c,
        ];
        assert_eq!(w.len(), 15);
        for (got, want) in w.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-7, "got {got}, want {want}");
        }
    }

    /// `ImageBoundaryFacesCalculator` puts the interior first and then the low
    /// and high face of each axis in axis order, with no overlap at the corners.
    #[test]
    fn boundary_faces_orders_interior_then_low_high_per_axis() {
        let faces = boundary_faces(&[5, 5], &[1, 1]);
        let expect = [
            (vec![1, 1], vec![3, 3]),
            (vec![0, 0], vec![1, 5]),
            (vec![4, 0], vec![1, 5]),
            (vec![1, 0], vec![3, 1]),
            (vec![1, 4], vec![3, 1]),
        ];
        assert_eq!(faces.len(), expect.len());
        for (face, (index, size)) in faces.iter().zip(&expect) {
            assert_eq!(&face.index, index);
            assert_eq!(&face.size, size);
        }
        // The faces tile the image exactly once.
        let covered: usize = faces.iter().map(|f| f.size.iter().product::<usize>()).sum();
        assert_eq!(covered, 25);
    }

    /// `GetPatchRadiusInVoxels` = `ceil(maxSpacing · patchRadius / spacing[d])`.
    #[test]
    fn patch_radius_in_voxels_is_isotropic_in_physical_space() {
        assert_eq!(patch_radius_in_voxels(4, &[1.0, 1.0]), vec![4, 4]);
        assert_eq!(patch_radius_in_voxels(2, &[0.3, 1.0]), vec![7, 2]);
        assert_eq!(patch_radius_in_voxels(1, &[1.0, 2.0, 4.0]), vec![4, 2, 1]);
    }

    /// The centre of a 3x3 image with `patch_radius = 1` has the region
    /// constraint `[min(1,1), max(1, 3-1-1)] = [1, 1]` on both axes, so every
    /// sampled patch *is* the query patch: the centre difference is exactly `0`
    /// and the joint-entropy gradient is exactly `0`. The pixel must come
    /// through bit-identical regardless of how many patches were drawn.
    #[test]
    fn single_point_constraint_leaves_the_centre_pixel_untouched() {
        let img =
            Image::from_vec(&[3, 3], vec![1.0, 2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0, 5.0]).unwrap();
        let out = patch_based_denoising(&img, &settings()).unwrap();
        assert_eq!(values(&out)[4], 9.0);
    }

    /// The centre of a 3x3 image whose `patch_radius` is 1 also fixes the whole
    /// output, hand-checked against the independent transcription of the `.hxx`.
    #[test]
    fn tiny_three_by_three_matches_the_reference_transcription() {
        let img =
            Image::from_vec(&[3, 3], vec![1.0, 2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0, 5.0]).unwrap();
        let settings = PatchBasedDenoisingSettings {
            kernel_bandwidth_sigma: 10.0,
            ..settings()
        };
        let out = patch_based_denoising(&img, &settings).unwrap();
        assert_close(
            &values(&out),
            &[
                1.690_478_808_173_368,
                3.4000000000000004,
                3.1077737232151623,
                4.0,
                9.0,
                6.286_632_018_484_947,
                6.955_004_340_512_749,
                8.095_871_827_933_161,
                5.486_747_490_414_632,
            ],
            1e-12,
        );
    }

    // ---- numeric fixtures --------------------------------------------------

    #[test]
    fn no_noise_model_single_iteration() {
        let out = patch_based_denoising(&fixture(), &settings()).unwrap();
        assert_close(
            &values(&out),
            &[
                12.003074873419152,
                20.0,
                30.112607879897823,
                38.906065507084726,
                48.805474454309476,
                15.774_194_935_800_16,
                24.997942351856423,
                34.522642194831384,
                44.256862366983036,
                53.768_854_638_220_3,
                13.192247583183443,
                23.687142411480714,
                31.144865988611716,
                40.964633023548274,
                50.862972965828924,
                19.413295117230085,
                27.988545276195403,
                37.037218331954456,
                44.542013407739596,
                55.877278010966464,
                11.335140674383567,
                21.0,
                31.052668624141518,
                38.773_899_077_887_44,
                50.575248149265065,
            ],
            1e-12,
        );
    }

    /// Every iteration reseeds the sampler with the work-unit id (`0` here), so
    /// the second iteration replays the same random stream over the updated
    /// image rather than continuing the first iteration's stream.
    #[test]
    fn no_noise_model_two_iterations() {
        let settings = PatchBasedDenoisingSettings {
            number_of_iterations: 2,
            ..settings()
        };
        let out = patch_based_denoising(&fixture(), &settings).unwrap();
        assert_close(
            &values(&out),
            &[
                13.767102768416294,
                20.0,
                30.138_490_646_718_72,
                37.948712194175585,
                47.618684195164334,
                16.506927475519827,
                25.138542822243398,
                34.071_301_876_065_67,
                43.484_282_210_122_21,
                52.550_180_108_740_62,
                14.363053242856614,
                25.045124634497824,
                30.404_522_073_725_77,
                39.840_183_406_430_85,
                49.761_015_614_694_97,
                20.643302623174847,
                28.032704994033523,
                36.193_303_185_335_12,
                41.681_463_952_867_27,
                53.718_500_899_075_66,
                11.718114501061432,
                21.0,
                31.047_027_140_579_9,
                36.890007628199726,
                49.858991282428484,
            ],
            1e-12,
        );
    }

    /// `static_cast<PixelType>(result)` truncates toward zero. Every quantity
    /// the algorithm computes for this fixture is an exact integer difference,
    /// so the `u16` run is the `f64` run truncated.
    #[test]
    fn integer_pixels_truncate_the_double_precision_result() {
        let data: Vec<u16> = IMG.iter().map(|v| *v as u16).collect();
        let img = Image::from_vec(&[5, 5], data).unwrap();
        let out = patch_based_denoising(&img, &settings()).unwrap();
        assert_eq!(
            out.scalar_slice::<u16>().unwrap(),
            &[
                12, 20, 30, 38, 48, //
                15, 24, 34, 44, 53, //
                13, 23, 31, 40, 50, //
                19, 27, 37, 44, 55, //
                11, 21, 31, 38, 50
            ]
        );
    }

    /// On the first iteration `out == in`, so `GAUSSIAN`'s
    /// `gradientFidelity = 2·(in - out)` and `POISSON`'s
    /// `(in - out)/(out + 1e-5)` are both exactly zero: neither model can change
    /// anything until the second iteration. `RICIAN` is not zero at `in == out`
    /// and does change the first iteration.
    #[test]
    fn gaussian_and_poisson_fidelity_are_inert_on_the_first_iteration() {
        let plain = values(&patch_based_denoising(&fixture(), &settings()).unwrap());
        for model in [NoiseModel::Gaussian, NoiseModel::Poisson] {
            let settings = PatchBasedDenoisingSettings {
                noise_model: model,
                noise_model_fidelity_weight: 0.5,
                ..settings()
            };
            let out = values(&patch_based_denoising(&fixture(), &settings).unwrap());
            assert_eq!(out, plain, "{model:?} changed the first iteration");
        }
    }

    #[test]
    fn gaussian_fidelity_two_iterations() {
        let settings = PatchBasedDenoisingSettings {
            number_of_iterations: 2,
            noise_model: NoiseModel::Gaussian,
            noise_model_fidelity_weight: 0.5,
            ..settings()
        };
        let out = patch_based_denoising(&fixture(), &settings).unwrap();
        assert_close(
            &values(&out),
            &[
                12.765565331706718,
                20.0,
                30.082186706769807,
                38.495679440633225,
                48.215_946_968_009_6,
                16.119830007619747,
                25.139571646315186,
                34.309_980_778_649_97,
                43.855_851_026_630_69,
                53.165752789630474,
                13.766929451264893,
                24.201553428757467,
                30.832_089_079_419_91,
                40.357866894656716,
                50.329529131780504,
                19.936655064559805,
                28.038_432_355_935_82,
                36.674_694_019_357_89,
                43.410457248997474,
                54.779861893592425,
                11.550544163869649,
                21.0,
                31.020_692_828_509_14,
                38.003_058_089_256,
                50.071_367_207_795_95,
            ],
            1e-12,
        );
    }

    #[test]
    fn poisson_fidelity_two_iterations() {
        let settings = PatchBasedDenoisingSettings {
            number_of_iterations: 2,
            noise_model: NoiseModel::Poisson,
            noise_model_fidelity_weight: 0.5,
            ..settings()
        };
        let out = patch_based_denoising(&fixture(), &settings).unwrap();
        assert_close(
            &values(&out),
            &[
                13.683662765511084,
                20.0,
                30.136620867738575,
                37.962_770_853_120_97,
                47.630921811070074,
                16.482_387_571_957_3,
                25.138583978577216,
                34.078215565698706,
                43.492_677_941_601_55,
                52.561_628_605_654_8,
                14.317865845033847,
                25.009_511_607_187_23,
                30.418_250_399_561_79,
                39.852_820_730_962_85,
                49.772192967731215,
                20.606_902_455_642_95,
                28.032909626314318,
                36.206_300_670_722_21,
                41.720_281_077_996_56,
                53.737495396344535,
                11.703331253621196,
                21.0,
                31.046_179_087_747_18,
                36.918_713_799_115_34,
                49.863_190_488_386_87,
            ],
            1e-12,
        );
    }

    #[test]
    fn rician_fidelity_single_iteration() {
        let settings = PatchBasedDenoisingSettings {
            noise_model: NoiseModel::Rician,
            noise_model_fidelity_weight: 0.5,
            noise_sigma: 3.0,
            ..settings()
        };
        let out = patch_based_denoising(&fixture(), &settings).unwrap();
        assert_close(
            &values(&out),
            &[
                11.772_476_948_742_07,
                19.886857321583364,
                30.037433698133004,
                38.849_753_166_509_19,
                48.760_450_237_522_19,
                15.622_629_619_071_96,
                24.907_625_075_366_15,
                34.458_254_346_275_24,
                44.206_823_536_284_94,
                53.727930778500706,
                13.001613937254259,
                23.584392988378475,
                31.074_413_674_304_72,
                40.911_010_080_912_61,
                50.819_683_778_207_39,
                19.287_403_695_673_93,
                27.907968869957365,
                36.977_932_118_782_34,
                44.495_109_122_305_54,
                55.838_474_208_637_41,
                11.126_469_900_958_1,
                20.892_305_381_274_42,
                30.979932108327095,
                38.718_964_378_498_37,
                50.531108506424296,
            ],
            1e-12,
        );
    }

    /// `noise_sigma = 0` means "unset", so ITK falls back to 5% of the intensity
    /// range: `5 / (100/(58-10)) = 2.4`. Passing `2.4` explicitly must agree.
    #[test]
    fn zero_noise_sigma_falls_back_to_five_percent_of_the_intensity_range() {
        let base = PatchBasedDenoisingSettings {
            noise_model: NoiseModel::Rician,
            noise_model_fidelity_weight: 0.5,
            ..settings()
        };
        let defaulted = patch_based_denoising(&fixture(), &base).unwrap();
        let explicit = patch_based_denoising(
            &fixture(),
            &PatchBasedDenoisingSettings {
                noise_sigma: 2.4,
                ..base
            },
        )
        .unwrap();
        assert_eq!(values(&defaulted), values(&explicit));
    }

    /// `GAUSSIAN` and `POISSON` never read `noise_sigma`.
    #[test]
    fn noise_sigma_does_not_reach_the_gaussian_or_poisson_fidelity_terms() {
        for model in [NoiseModel::Gaussian, NoiseModel::Poisson] {
            let base = PatchBasedDenoisingSettings {
                number_of_iterations: 2,
                noise_model: model,
                noise_model_fidelity_weight: 0.5,
                ..settings()
            };
            let a = patch_based_denoising(&fixture(), &base).unwrap();
            let b = patch_based_denoising(
                &fixture(),
                &PatchBasedDenoisingSettings {
                    noise_sigma: 17.0,
                    ..base
                },
            )
            .unwrap();
            assert_eq!(values(&a), values(&b), "{model:?} read noise_sigma");
        }
    }

    /// Kernel-bandwidth estimation runs before the image update on iteration 0,
    /// consuming the sampler's stream for its Newton iterations, and drives
    /// `σ` from 25 down to ≈5.6039 before a single pixel is denoised.
    #[test]
    fn kernel_bandwidth_estimation_single_iteration() {
        let settings = PatchBasedDenoisingSettings {
            kernel_bandwidth_estimation: true,
            ..settings()
        };
        let out = patch_based_denoising(&fixture(), &settings).unwrap();
        assert_close(
            &values(&out),
            &[
                10.000_216_081_477_31,
                20.003_564_701_994_81,
                30.0,
                39.670_724_000_853_52,
                49.916_444_116_632_23,
                15.035_221_874_203_49,
                24.779235564086896,
                35.020787734383376,
                42.999_937_435_996_77,
                54.688_001_191_758_3,
                12.280_767_664_846_34,
                22.176111206516342,
                32.179449301988086,
                41.897350351888434,
                52.124_110_565_191_19,
                17.674180051004832,
                27.119504291603608,
                37.760_045_284_620_25,
                47.222_848_622_849_12,
                57.534_216_481_146_71,
                12.488707158626374,
                21.047585941996143,
                31.521518364318155,
                39.918514956406334,
                51.114436444905095,
            ],
            1e-12,
        );
    }

    /// `always_treat_components_as_euclidean` cannot change a scalar result:
    /// `DetermineComponentSpace` already returns `EUCLIDEAN`.
    #[test]
    fn always_treat_components_as_euclidean_is_inert_for_scalar_pixels() {
        let a = patch_based_denoising(&fixture(), &settings()).unwrap();
        let b = patch_based_denoising(
            &fixture(),
            &PatchBasedDenoisingSettings {
                always_treat_components_as_euclidean: true,
                ..settings()
            },
        )
        .unwrap();
        assert_eq!(values(&a), values(&b));
    }

    /// `POISSON`'s step size clamps `min(outVal, 0.99999)` in real arithmetic,
    /// so it no longer depends on the pixel type. Upstream cast `0.99999` to the
    /// pixel type first, which is `0` for every integer type and forced the step
    /// to the `1e-5` floor regardless of the pixel value; the fix lets an integer
    /// pixel reach the same saturated step a float pixel of the same value would.
    #[test]
    fn poisson_step_size_clamps_the_literal_in_real_arithmetic() {
        // Saturated branch: any outVal ≥ 0.99999 clamps to 0.99999, so the step
        // is 0.99999 + 0.00001. A pixel of value 1, 5 or 255 — integer or float
        // — all step by this same amount. Upstream's cast denied integer pixels
        // this branch entirely (their step was stuck at 0.00001).
        let saturated = 0.99999_f64 + 0.00001;
        assert_eq!(poisson_step_size(1.0), saturated);
        assert_eq!(poisson_step_size(5.0), saturated);
        assert_eq!(poisson_step_size(255.0), saturated);
        assert_ne!(poisson_step_size(5.0), 0.00001);

        // A truly dark pixel (outVal == 0) keeps the 1e-5 floor: min(0, …) == 0.
        assert_eq!(poisson_step_size(0.0), 0.00001);

        // A fractional float pixel below the clamp passes through unsaturated.
        assert_eq!(poisson_step_size(0.5), 0.5 + 0.00001);
    }

    /// `b - a` is evaluated in the pixel type. `u8` promotes to `int`, so the
    /// difference is a signed `-245`; `u32` wraps modularly.
    #[test]
    fn pixel_arithmetic_reproduces_cpp_integer_promotion() {
        assert_eq!(<u8 as PatchPixel>::sub_f64(5, 250), -245.0);
        assert_eq!(<u8 as PatchPixel>::mul_f64(200, 200), 40000.0);
        assert_eq!(<u32 as PatchPixel>::sub_f64(5, 250), 4294967051.0);
        assert_eq!(<i32 as PatchPixel>::sub_f64(5, 250), -245.0);
    }

    // ---- error paths -------------------------------------------------------

    #[test]
    fn patch_larger_than_the_image_is_rejected() {
        let img = Image::from_vec(&[3, 3], vec![1.0f64; 9]).unwrap();
        let settings = PatchBasedDenoisingSettings {
            patch_radius: 2,
            ..settings()
        };
        assert!(matches!(
            patch_based_denoising(&img, &settings),
            Err(FilterError::PatchLargerThanImage { .. })
        ));
    }

    #[test]
    fn constant_image_is_rejected() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        assert!(matches!(
            patch_based_denoising(&img, &settings()),
            Err(FilterError::ConstantImage(_))
        ));
    }

    #[test]
    fn negative_intensity_is_rejected_for_rician_and_poisson_only() {
        let mut data = IMG.to_vec();
        data[0] = -1.0;
        let img = Image::from_vec(&[5, 5], data).unwrap();
        for model in [NoiseModel::Rician, NoiseModel::Poisson] {
            let settings = PatchBasedDenoisingSettings {
                noise_model: model,
                ..settings()
            };
            assert!(matches!(
                patch_based_denoising(&img, &settings),
                Err(FilterError::NegativeIntensityForNoiseModel(_))
            ));
        }
        assert!(patch_based_denoising(&img, &settings()).is_ok());
    }

    #[test]
    fn kernel_bandwidth_sigma_at_or_below_min_sigma_is_rejected() {
        let settings = PatchBasedDenoisingSettings {
            kernel_bandwidth_sigma: MIN_SIGMA,
            ..settings()
        };
        assert!(matches!(
            patch_based_denoising(&fixture(), &settings),
            Err(FilterError::KernelBandwidthSigmaTooSmall(_, _))
        ));
    }

    #[test]
    fn negative_sample_variance_is_rejected() {
        let settings = PatchBasedDenoisingSettings {
            sample_variance: -1.0,
            ..settings()
        };
        assert!(matches!(
            patch_based_denoising(&fixture(), &settings),
            Err(FilterError::InvalidSampleVariance(_))
        ));
    }

    /// `itkSetClampMacro` clamps silently: zero iterations become one, and a
    /// fidelity weight above 1 becomes 1.
    #[test]
    fn out_of_range_settings_are_clamped_not_rejected() {
        let zero = PatchBasedDenoisingSettings {
            number_of_iterations: 0,
            ..settings()
        };
        let one = settings();
        assert_eq!(
            values(&patch_based_denoising(&fixture(), &zero).unwrap()),
            values(&patch_based_denoising(&fixture(), &one).unwrap())
        );

        let over = PatchBasedDenoisingSettings {
            number_of_iterations: 2,
            noise_model: NoiseModel::Gaussian,
            noise_model_fidelity_weight: 4.0,
            ..settings()
        };
        let at = PatchBasedDenoisingSettings {
            noise_model_fidelity_weight: 1.0,
            ..over
        };
        assert_eq!(
            values(&patch_based_denoising(&fixture(), &over).unwrap()),
            values(&patch_based_denoising(&fixture(), &at).unwrap())
        );
    }

    /// SimpleITK's yaml defaults.
    #[test]
    fn defaults_match_the_simpleitk_yaml() {
        let d = PatchBasedDenoisingSettings::default();
        assert_eq!(d.kernel_bandwidth_sigma, 400.0);
        assert_eq!(d.patch_radius, 4);
        assert_eq!(d.number_of_iterations, 1);
        assert_eq!(d.number_of_sample_patches, 200);
        assert_eq!(d.sample_variance, 400.0);
        assert_eq!(d.noise_model, NoiseModel::NoModel);
        assert_eq!(d.noise_sigma, 0.0);
        assert_eq!(d.noise_model_fidelity_weight, 0.0);
        assert!(!d.always_treat_components_as_euclidean);
        assert!(!d.kernel_bandwidth_estimation);
        assert_eq!(d.kernel_bandwidth_multiplication_factor, 1.0);
        assert_eq!(d.kernel_bandwidth_update_frequency, 3);
        assert_eq!(d.kernel_bandwidth_fraction_pixels_for_estimation, 0.2);
    }
}
