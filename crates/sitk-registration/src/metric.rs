//! The **mean-squares** similarity metric and its compute backend.
//!
//! This module implements **mean squares**
//! (`itk::MeanSquaresImageToImageMetricv4`); the Mattes mutual-information metric
//! for multi-modality registration lives in the sibling
//! [`mattes`](crate::mattes) module and reuses this module's [`FixedSamples`] /
//! [`MovingImage`] primitives.
//!
//! Mean squares is
//!
//! ```text
//! value = (1/N) Σ ( M(T(xᵢ)) − F(xᵢ) )²
//! ```
//!
//! over the fixed-image sample points `xᵢ` (the *virtual domain*) that map,
//! under transform `T`, inside the moving image `M`. `F` is the fixed image.
//! The derivative with respect to the transform parameters `p` is
//!
//! ```text
//! ∂value/∂pₖ = (2/N) Σ diffᵢ · ( ∇M(T(xᵢ)) · J_T(xᵢ) )ₖ
//! ```
//!
//! where `diffᵢ = M(T(xᵢ)) − F(xᵢ)`, `∇M` is the moving image's spatial
//! gradient, and `J_T` is the transform Jacobian
//! ([`ParametricTransform::jacobian_wrt_parameters`]).
//!
//! `∇M` here is the **exact gradient of the linear interpolant**
//! ([`linear_value_and_gradient`]), so the metric derivative is the true
//! gradient of the (interpolated) metric value — the optimizer's finite
//! difference of the value reproduces it. This is a documented, deliberate
//! difference from ITK, whose `ImageToImageMetricv4` defaults to a
//! Gaussian-smoothed gradient image (or a raw central-difference
//! `CentralDifferenceImageFunction` when `SetUseMovingImageGradientFilter` is
//! off); both are gradient *estimates* not consistent with the interpolated
//! value.
//!
//! ## GPU seam
//!
//! The per-sample reduction is isolated behind [`MetricBackend`]. [`CpuBackend`]
//! runs it on the host and is the only backend that compiles on a machine
//! without a GPU (this one). A future CUDA (`cudarc`) or portable
//! `wgpu`/Metal backend implements the same trait — marshalling the sample
//! arrays, moving buffer, and transform parameters to the device — without any
//! change to [`MeanSquaresMetric`] or the registration method above it.
//!
//! ## Parallelism, and why the numbers do not move
//!
//! [`CpuBackend`] evaluates the samples on every core, and returns **the same
//! bits it returned when it was serial**, at any thread count. It gets that by
//! construction, from [`sitk_core::parallel::map_rows_fold_in_order`]: the
//! expensive per-sample work (transform, interpolation, Jacobian) never touches
//! an accumulator, so it runs in parallel; the accumulators are then fed the
//! per-sample contributions on a single thread, in sample order, executing the
//! identical sequence of additions the serial loop did. Nothing is
//! re-associated, so no float sum is re-rounded — which matters because the
//! optimizer is a feedback loop, and a metric value that shifted by one ulp
//! would walk to a different registration result.
//!
//! One path stays serial: the derivative of a transform with a **sparse**
//! Jacobian (B-spline, displacement field). Its per-sample contribution is a
//! scattered, variable-length entry list; staging that into a fixed-width row
//! would cost `O(nparams)` per sample and defeat the sparsity. Its value-only
//! reduction ([`MetricBackend::mean_squares_value`]) is parallel like every
//! other transform's.

use sitk_core::parallel;
use sitk_core::{Image, Scalar, dispatch_scalar};
use sitk_transform::Interpolator;
use sitk_transform::ParametricTransform;
use sitk_transform::interpolator::{
    SincWindow, bspline_coefficients, bspline_value_and_gradient, gaussian_value_and_gradient,
    index_to_physical_matrix, linear_at, linear_value_and_gradient, nearest_at,
    nearest_value_and_gradient, physical_to_index_matrix, strides,
    windowed_sinc_value_and_gradient,
};

use crate::error::{RegistrationError, Result};
use crate::scales::{ScalesEstimator, ScalesEstimatorKind, VirtualGrid};

/// Fixed-image sampling strategy for the registration virtual domain
/// (`itk::ImageRegistrationMethodv4::MetricSamplingStrategyEnum` / SimpleITK's
/// `MetricSamplingStrategyType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Every voxel (SimpleITK's default).
    None,
    /// Every `ceil(1/percentage)`-th voxel, in scan-line (dim-0-fastest)
    /// traversal order.
    Regular,
    /// `floor(N * percentage)` voxels drawn uniformly with replacement.
    Random,
}

/// A small, deterministic, seeded PRNG (SplitMix64, Vigna 2015, public
/// domain) for reproducible [`SamplingStrategy::Random`] sampling. Not
/// bit-parity with ITK's Mersenne Twister — the task only requires
/// reproducibility for a fixed seed, not ITK-identical draws.
///
/// Shared with [`crate::scales`], whose `Sampling::Random` strategy stands in
/// for ITK's `ImageRandomConstIteratorWithIndex` for the same reason.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index in `[0, n)`. `n` must be nonzero.
    pub(crate) fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// A process-unique identity for one prepared [`FixedSamples`] / [`MovingImage`].
///
/// A compute backend that keeps a *device-resident* copy of these buffers — the
/// CUDA backend uploads them once and reuses them across hundreds of optimizer
/// iterations — needs to answer "is this the same data I already have?" on every
/// call. The [`MetricBackend`] trait hands it a `&FixedSamples` with no identity,
/// and a pointer address is not an identity (a freed allocation's address can be
/// handed back to a different one). So each prepared buffer carries a serial
/// number, minted once at construction and never reused within the process.
///
/// Costs one relaxed increment per `FixedSamples`/`MovingImage` built, which is
/// per pyramid level, not per iteration. The CPU path never reads it — so the
/// counter and the fields that hold it are compiled out entirely when the `cuda`
/// feature is off, and a CPU-only build carries neither the atomic nor the extra
/// eight bytes per buffer.
#[cfg(feature = "cuda")]
fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// The fixed image's sample values, kept in the image's **native pixel type**.
///
/// The port used to widen the whole volume to `f64` up front
/// ([`Image::to_f64_vec`]) and hold that copy for the life of the metric — 134 MB
/// at 256³ for a `u16` CT that is 34 MB on disk, allocated and first-touched
/// before a single iteration runs, and then re-read on every iteration at twice
/// or four times the memory traffic the native buffer would cost.
///
/// The widening itself is not the problem and is not removed: `T::as_f64` is
/// **exactly** the conversion `to_f64_vec` performed, so deferring it to the
/// point of use changes no bit of any metric value. What is removed is the
/// *materialized f64 volume*. This is the same distinction that let
/// `sitk_core::fused::map_pixels` keep bit-parity while deleting the port's
/// dominant filter cost.
///
/// # Where the widen is not lossless
///
/// `u64`/`i64` above 2^53 do not survive `as f64` exactly. That is **not a new
/// loss**: `to_f64_vec` applied the identical `as f64` at construction, so a
/// 64-bit-integer fixed image already fed the metric rounded values, and it now
/// feeds it the same rounded values one sample later. Nothing narrows anywhere —
/// the arithmetic stays `f64` end to end. The only honest statement is that a
/// metric over `u64` intensities beyond 2^53 was, and remains, computed on
/// `f64`-rounded inputs.
macro_rules! sample_values {
    ($($variant:ident($ty:ty)),+ $(,)?) => {
        /// One value per sample, length `N`, in the fixed image's own type.
        #[derive(Clone, Debug, PartialEq)]
        pub(crate) enum SampleValues {
            $($variant(Vec<$ty>),)+
        }

        impl SampleValues {
            pub(crate) fn len(&self) -> usize {
                match self { $(Self::$variant(v) => v.len(),)+ }
            }

            /// Sample `s`, widened to `f64` — the same `as f64` the eager
            /// `to_f64_vec` applied, just not stored.
            #[inline]
            pub(crate) fn get(&self, s: usize) -> f64 {
                match self { $(Self::$variant(v) => v[s].as_f64(),)+ }
            }

            /// The `(min, max)` over every sample, widened. Exact and
            /// order-independent, so this is the sequential scan's answer.
            fn min_max(&self) -> Option<(f64, f64)> {
                match self { $(Self::$variant(v) => parallel::min_max(v),)+ }
            }

            /// Hand the native slice to `w`. **One branch per call** — the
            /// interpolators are generic over the pixel type, so an interpolation
            /// that reads 64 corners resolves the type once, not once per corner.
            #[inline]
            fn with<W: WithBuf>(&self, w: W) -> W::Out {
                match self { $(Self::$variant(v) => w.call(v),)+ }
            }
        }

        /// Lets a `Vec<T>` name its own [`SampleValues`] variant, so the gather
        /// below can be generic over the scalar type and still build the enum.
        trait IntoSampleValues: Scalar {
            fn wrap(v: Vec<Self>) -> SampleValues;
        }

        $(impl IntoSampleValues for $ty {
            fn wrap(v: Vec<Self>) -> SampleValues { SampleValues::$variant(v) }
        })+
    };
}

/// An operation that is generic over a [`SampleValues`] buffer's native pixel
/// type. Rust closures cannot be generic, so the work has to arrive as a trait
/// impl for [`SampleValues::with`] to monomorphize it per type.
trait WithBuf {
    type Out;
    fn call<T: Scalar>(self, buf: &[T]) -> Self::Out;
}

sample_values!(
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
);

/// The selected samples' values, in the image's native type — the whole buffer
/// when the selection is the identity, a gather otherwise.
///
/// Parallel, and not as an optimization of the copy itself: the destination is a
/// fresh allocation, so *whichever* thread writes a page first faults it in. A
/// serial `to_vec` here faults 67 MB of pages on one thread and costs an order of
/// magnitude more than the same copy spread over the pool (measured 232 ms vs
/// 22 ms of setup at 256³). This is the same first-touch cost the eager
/// `Image::to_f64_vec` avoided by going through [`parallel::map_slice`].
fn gather_values<T: IntoSampleValues>(
    img: &Image,
    selected: Option<&[usize]>,
) -> Result<SampleValues> {
    let src = img.scalar_slice::<T>()?;
    Ok(T::wrap(match selected {
        None => parallel::map_slice(src, |&v| v),
        Some(flats) => parallel::map_slice(flats, |&f| src[f]),
    }))
}

/// Where a sample's physical point comes from.
///
/// The unsampled, unmasked default — the SimpleITK default, and what a
/// registration actually runs — samples *every voxel of the virtual grid, in
/// grid order*. For that set the point of sample `s` is a closed-form function
/// of `s` alone ([`VirtualGrid::write_point`]: `dim` divisions and nine flops),
/// so the port used to spend 402 MB (at 256³) and a full page-fault pass
/// memoizing a function it can evaluate in the loop that reads it. Now it does
/// not exist.
///
/// Every other strategy — [`SamplingStrategy::Regular`], `Random`, or any mask —
/// selects an arbitrary subset, for which there is no closed form from `s`, so
/// those points are materialized. That is the *only* case that allocates, it is
/// proportional to the sample count rather than the volume, and it is named here
/// rather than being a flag on a buffer that always exists.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SamplePoints {
    /// Sample `s` is voxel `s` of [`FixedSamples::grid`]; its point is derived.
    Grid,
    /// Physical points, row-major `N × dim`, for a sample set with no closed
    /// form.
    Explicit(Vec<f64>),
}

/// Per-task scratch for [`FixedSamples::point`], so the derivation in the hot
/// loop allocates nothing. One per thread, not one per sample.
pub(crate) struct PointScratch {
    index: Vec<usize>,
    point: Vec<f64>,
}

/// The physical points of an arbitrary sample set, given as flat voxel indices
/// into `grid` — the materialization the derived path exists to avoid, kept for
/// the sampled/masked strategies that genuinely have no closed form.
///
/// Bit-identical to the serial loop it replaces: every component is
/// [`VirtualGrid::write_point`]'s, computed from its own sample index alone.
fn explicit_points(grid: &VirtualGrid, flats: &[usize], dim: usize) -> Vec<f64> {
    parallel::map_indexed_init(
        flats.len() * dim,
        || (vec![0usize; dim], vec![0.0f64; dim]),
        |(index, point), i| {
            grid.write_point(flats[i / dim], index, point);
            point[i % dim]
        },
    )
}

/// The fixed image reduced to its sample set (the registration *virtual
/// domain*): every pixel's value and its physical point, precomputed once.
pub struct FixedSamples {
    pub(crate) dim: usize,
    /// Identity for a device-resident copy of these buffers — see [`next_id`].
    #[cfg(feature = "cuda")]
    pub(crate) id: u64,
    /// One value per sample, length `N`, in the fixed image's native pixel type
    /// — read through [`value`](Self::value), which widens. See [`SampleValues`].
    pub(crate) values: SampleValues,
    /// Where each sample's physical point comes from — derived from the grid for
    /// the full-grid default, materialized only for a sampled or masked subset.
    /// Read through [`point`](Self::point), never directly.
    pub(crate) points: SamplePoints,
    /// Minimum fixed-image spacing (the maximum physical step for optimization).
    min_spacing: f64,
    /// The virtual domain as a grid. The metric never reads it — only the
    /// parameter-scales estimator does, because ITK's estimator draws its own
    /// sample points from the virtual domain rather than reusing the metric's
    /// [`points`](Self::points) (see [`crate::scales`]). Keeping the geometry
    /// rather than the points is what lets the estimator honor its own
    /// sampling strategy, its own `central_region_radius`, and ITK's rule that
    /// neither the metric's sampling percentage nor its fixed mask narrows the
    /// scale estimate.
    ///
    /// `pub(crate)` rather than private because the CPU metric and the device
    /// backend both derive their sample points from this geometry — see
    /// [`SamplePoints`].
    pub(crate) grid: VirtualGrid,
}

impl FixedSamples {
    /// Reduce a fixed image to its full sample set (sampling strategy = None:
    /// every pixel, matching SimpleITK's default).
    ///
    /// Fails on a vector `fixed` image, like every scalar consumer of
    /// [`sitk_core::Image::to_f64_vec`].
    pub fn from_image(fixed: &Image) -> Result<Self> {
        Self::from_image_with(fixed, SamplingStrategy::None, 1.0, 0, None)
    }

    /// Reduce a fixed image to its sample set under an explicit sampling
    /// `strategy`, optionally restricted to a fixed-image mask (ITK
    /// `ImageRegistrationMethodv4::SampleFixedImageDomain` +
    /// `ImageToImageMetricv4::SetFixedImageMask`).
    ///
    /// `percentage` and `seed` are used by [`SamplingStrategy::Regular`]
    /// (stride `= ceil(1/percentage)`) and [`SamplingStrategy::Random`]
    /// (`floor(N * percentage)` draws, uniform with replacement, seeded via a
    /// small deterministic PRNG — see the private `SplitMix64`); both are ignored for
    /// [`SamplingStrategy::None`]. `mask` is a binary image on the same grid
    /// as `fixed` (any nonzero value is "inside"): a candidate sample whose
    /// voxel is zero in the mask is dropped, exactly as ITK's
    /// `IsInsideInWorldSpace` gate on the fixed mask.
    ///
    /// Unlike ITK, this does **not** perturb each sample by a sub-voxel
    /// Gaussian jitter (`randomizer->GetNormalVariate() *
    /// oneThirdVirtualSpacing`) — an intentional, documented deviation: every
    /// sample lands exactly on a voxel center, which keeps `Regular`'s stride
    /// and `Random`'s count exactly reproducible without porting a
    /// normal-variate generator.
    ///
    /// # The sample set is a list of voxels, and that is all it is
    ///
    /// Each strategy is a *selection*: which flat voxel indices, in which order,
    /// with what multiplicity. Everything downstream — the values, the physical
    /// points — is a function of that selection and the grid. Writing it that way
    /// is what lets the default case (every voxel, in grid order) carry **no
    /// index list and no points buffer at all**: its selection is the identity,
    /// and [`VirtualGrid::write_point`] recovers any sample's point from its
    /// index in nine flops. The buffer that used to hold those points was 402 MB
    /// at 256³ and the largest single term in a GPU registration's setup.
    ///
    /// Fails if `mask` does not share `fixed`'s size.
    pub fn from_image_with(
        fixed: &Image,
        strategy: SamplingStrategy,
        percentage: f64,
        seed: u64,
        mask: Option<&Image>,
    ) -> Result<Self> {
        let dim = fixed.dimension();
        let size = fixed.size().to_vec();
        let n = fixed.number_of_pixels();

        let idx_to_phys = index_to_physical_matrix(fixed.direction(), fixed.spacing(), dim);
        let grid = VirtualGrid::new(dim, size, fixed.origin().to_vec(), idx_to_phys);

        let mask_buf = match mask {
            Some(m) => {
                if m.size() != fixed.size() {
                    return Err(RegistrationError::MaskSizeMismatch {
                        which: "fixed",
                        mask: m.size().to_vec(),
                        image: fixed.size().to_vec(),
                    });
                }
                Some(m.to_f64_vec()?)
            }
            None => None,
        };
        let mask_allows = |flat: usize| match &mask_buf {
            None => true,
            Some(m) => m[flat] != 0.0,
        };

        // The selected voxels, as flat indices in sample order. `None` is the
        // identity selection — every voxel, in grid order — and is the one case
        // that needs neither this list nor a points buffer.
        let selected: Option<Vec<usize>> = match strategy {
            SamplingStrategy::None if mask_buf.is_none() => None,
            SamplingStrategy::None => Some((0..n).filter(|&s| mask_allows(s)).collect()),
            SamplingStrategy::Regular => {
                let stride = ((1.0 / percentage).ceil() as usize).max(1);
                Some((0..n).step_by(stride).filter(|&s| mask_allows(s)).collect())
            }
            SamplingStrategy::Random => {
                let sample_count = (n as f64 * percentage) as usize;
                let mut rng = SplitMix64::new(seed);
                Some(
                    (0..sample_count)
                        .map(|_| rng.next_below(n))
                        .filter(|&flat| mask_allows(flat))
                        .collect(),
                )
            }
        };

        // The values never become an `f64` volume: the native buffer is taken (or
        // gathered) as-is, and every read widens one sample.
        let values: SampleValues =
            dispatch_scalar!(fixed.pixel_id(), gather_values, fixed, selected.as_deref())?;
        let points = match &selected {
            None => SamplePoints::Grid,
            Some(flats) => SamplePoints::Explicit(explicit_points(&grid, flats, dim)),
        };

        let min_spacing = fixed
            .spacing()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        Ok(Self {
            dim,
            #[cfg(feature = "cuda")]
            id: next_id(),
            values,
            points,
            min_spacing,
            grid,
        })
    }

    /// Scratch for [`point`](Self::point), one per thread.
    pub(crate) fn scratch(&self) -> PointScratch {
        PointScratch {
            index: vec![0usize; self.dim],
            point: vec![0.0f64; self.dim],
        }
    }

    /// The physical point of sample `s`, length `dim`.
    ///
    /// Derived from the grid for the full-grid sample set (no buffer exists to
    /// read), or read out of the materialized points for a sampled/masked one.
    /// Which it is, is not the caller's business — this is the single accessor,
    /// so the storage can be a closed form without 24 call sites knowing.
    #[inline]
    pub(crate) fn point<'a>(&'a self, s: usize, scratch: &'a mut PointScratch) -> &'a [f64] {
        match &self.points {
            SamplePoints::Explicit(p) => &p[s * self.dim..(s + 1) * self.dim],
            SamplePoints::Grid => {
                self.grid
                    .write_point(s, &mut scratch.index, &mut scratch.point);
                &scratch.point
            }
        }
    }

    /// Number of samples `N`.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.values.len() == 0
    }

    /// The value of sample `s`, widened to `f64`.
    #[inline]
    pub(crate) fn value(&self, s: usize) -> f64 {
        self.values.get(s)
    }

    /// The `(min, max)` of the sampled fixed-image values. `(0, 0)` when empty.
    /// This is the fixed-image intensity range over the analysis region (full
    /// sampling ⇒ the whole image), which the Mattes MI metric uses to size the
    /// joint-histogram fixed axis.
    pub(crate) fn value_range(&self) -> (f64, f64) {
        self.values.min_max().unwrap_or((0.0, 0.0))
    }

    /// Build a scale/learning-rate estimator of `kind` for `transform` over
    /// this metric's **virtual domain** (shared by every metric). See
    /// [`ScalesEstimator`].
    ///
    /// `moving` supplies the physical-to-continuous-index matrix
    /// [`ScalesEstimatorKind::IndexShift`] measures its shifts in; the other
    /// two kinds ignore it.
    pub(crate) fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        moving: &MovingImage,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        ScalesEstimator::new(
            &self.grid,
            transform,
            &moving.phys_to_index,
            self.min_spacing,
            kind,
        )
    }
}

/// Everything a device-resident metric backend needs to reproduce the moving
/// image's sampling, borrowed from a [`MovingImage`]. See
/// [`MovingImage::device_view`].
#[cfg(feature = "cuda")]
pub(crate) struct MovingView<'a> {
    pub(crate) id: u64,
    pub(crate) buf: &'a SampleValues,
    pub(crate) size: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) origin: &'a [f64],
    pub(crate) phys_to_index: &'a [f64],
    pub(crate) interpolator: Interpolator,
    pub(crate) mask: Option<&'a [bool]>,
}

/// The moving image plus the geometry needed to map a physical point to a
/// continuous index and to convert an index-space gradient to a physical-space
/// gradient.
pub struct MovingImage {
    dim: usize,
    /// Identity for a device-resident copy of this buffer — see [`next_id`].
    #[cfg(feature = "cuda")]
    pub(crate) id: u64,
    /// The voxels, in the image's **native** pixel type — the interpolators are
    /// generic over it and widen at each load, so no `f64` copy of the volume is
    /// ever made (see [`SampleValues`]).
    buf: SampleValues,
    size: Vec<usize>,
    strides: Vec<usize>,
    origin: Vec<f64>,
    /// `diag(1/spacing) · D⁻¹`, row-major `dim × dim`: maps a physical
    /// displacement from the origin to a continuous index.
    phys_to_index: Vec<f64>,
    interpolator: Interpolator,
    /// Precomputed cubic B-spline coefficients, present only when
    /// `interpolator == BSpline` (see [`bspline_coefficients`]).
    ///
    /// These are `f64` and volume-sized, and that is not removable the way the
    /// pixel buffers were: a coefficient is not a pixel — it is the result of the
    /// prefilter's recursion and does not fit the source type. **The B-spline path
    /// therefore still materializes one `f64` volume**; every other interpolator
    /// materializes none.
    bspline_coeffs: Option<Vec<f64>>,
    /// Binary moving mask, same size/traversal order as `buf`. `None` = no
    /// mask (every in-buffer point is valid).
    mask: Option<Vec<bool>>,
}

impl MovingImage {
    /// Prepare a moving image with linear interpolation and no mask. Fails if
    /// its direction matrix is singular.
    pub fn from_image(moving: &Image) -> Result<Self> {
        Self::from_image_with_interpolator(moving, Interpolator::Linear)
    }

    /// Prepare a moving image with an explicit interpolator (nearest
    /// neighbor, linear, cubic B-spline, or Gaussian — see
    /// [`sitk_transform::Interpolator`]). Fails if its direction matrix is
    /// singular.
    pub fn from_image_with_interpolator(
        moving: &Image,
        interpolator: Interpolator,
    ) -> Result<Self> {
        let dim = moving.dimension();
        let size = moving.size().to_vec();
        let phys_to_index = physical_to_index_matrix(moving.direction(), moving.spacing(), dim)
            .ok_or(RegistrationError::SingularDirection)?;
        let strides_v = strides(&size);
        let buf: SampleValues = dispatch_scalar!(moving.pixel_id(), gather_values, moving, None)?;
        let bspline_coeffs = matches!(interpolator, Interpolator::BSpline).then(|| {
            buf.with(Coefficients {
                size: &size,
                strides: &strides_v,
            })
        });
        Ok(Self {
            dim,
            #[cfg(feature = "cuda")]
            id: next_id(),
            buf,
            strides: strides_v,
            size,
            origin: moving.origin().to_vec(),
            phys_to_index,
            interpolator,
            bspline_coeffs,
            mask: None,
        })
    }

    /// Restrict this moving image to a binary mask on its own grid (ITK
    /// `ImageToImageMetricv4::SetMovingImageMask`): any nonzero mask voxel is
    /// "inside". A physical point that maps to a zero-mask voxel makes
    /// the private `value_and_physical_gradient`
    /// return `None`, exactly as if it fell outside the buffer. Fails if
    /// `mask` does not share this image's size.
    pub fn with_moving_mask(mut self, mask: &Image) -> Result<Self> {
        if mask.size() != self.size.as_slice() {
            return Err(RegistrationError::MaskSizeMismatch {
                which: "moving",
                mask: mask.size().to_vec(),
                image: self.size.clone(),
            });
        }
        self.mask = Some(mask.to_f64_vec()?.iter().map(|&v| v != 0.0).collect());
        Ok(self)
    }

    /// Spatial dimension.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// The moving-image buffer and the geometry a device-resident backend needs
    /// to reproduce [`value_and_physical_gradient`](Self::value_and_physical_gradient)
    /// on its own. `pub(crate)`: the CUDA backend lives in this crate, so none of
    /// this widens the public API.
    #[cfg(feature = "cuda")]
    pub(crate) fn device_view(&self) -> MovingView<'_> {
        MovingView {
            id: self.id,
            buf: &self.buf,
            size: &self.size,
            strides: &self.strides,
            origin: &self.origin,
            phys_to_index: &self.phys_to_index,
            interpolator: self.interpolator,
            mask: self.mask.as_deref(),
        }
    }

    /// Continuous index of physical point `p`: `M · (p − origin)`.
    fn continuous_index(&self, p: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut c = vec![0.0; dim];
        for (r, cr) in c.iter_mut().enumerate() {
            let row = &self.phys_to_index[r * dim..(r + 1) * dim];
            *cr = row
                .iter()
                .zip(p.iter().zip(self.origin.iter()))
                .map(|(&m, (&pj, &oj))| m * (pj - oj))
                .sum();
        }
        c
    }

    /// Whether continuous index `c` is allowed by the moving mask (always
    /// `true` when there is no mask). Rounds to the nearest voxel, matching
    /// ITK's `ImageMaskSpatialObject` point-in-mask test.
    fn mask_allows(&self, c: &[f64]) -> bool {
        let mask = match &self.mask {
            None => return true,
            Some(m) => m,
        };
        let mut flat = 0usize;
        for (d, &cd) in c.iter().enumerate() {
            let r = cd.round();
            if r < 0.0 || r as usize >= self.size[d] {
                return false;
            }
            flat += r as usize * self.strides[d];
        }
        mask[flat]
    }

    /// Sample and its exact index-space gradient at continuous index `c`
    /// under this image's interpolator, or `None` if outside the buffer.
    fn value_and_gradient(&self, c: &[f64]) -> Option<(f64, Vec<f64>)> {
        self.buf.with(ValueAndGradient { img: self, c })
    }

    /// Sample of physical point `p` under this image's interpolator and its
    /// gradient expressed in **physical space**, or `None` if `p` maps
    /// outside the buffer or onto a zero moving-mask voxel.
    ///
    /// With `cindex = M·(p − origin)`, the physical-space gradient is
    /// `∂M(value)/∂p_d = Σ_j (∂value/∂cindex_j) · M[j][d]`, i.e. the index-space
    /// gradient left-multiplied by the moving image's physical-to-index matrix.
    /// Both the mean-squares and Mattes MI metrics need exactly this, so it lives
    /// here rather than being duplicated in each metric's reduction.
    pub(crate) fn value_and_physical_gradient(&self, p: &[f64]) -> Option<(f64, Vec<f64>)> {
        let dim = self.dim;
        let cidx = self.continuous_index(p);
        if !self.mask_allows(&cidx) {
            return None;
        }
        let (value, grad_index) = self.value_and_gradient(&cidx)?;
        let mut grad_phys = vec![0.0; dim];
        for (d, gp) in grad_phys.iter_mut().enumerate() {
            *gp = grad_index
                .iter()
                .enumerate()
                .map(|(j, &gj)| gj * self.phys_to_index[j * dim + d])
                .sum();
        }
        Some((value, grad_phys))
    }

    /// Sample of physical point `p` under this image's interpolator, or
    /// `None` if `p` maps outside the buffer or onto a zero moving-mask
    /// voxel — a value-only sibling of
    /// [`value_and_physical_gradient`](Self::value_and_physical_gradient)
    /// for a pass that does not need the gradient (e.g. the histogram-only
    /// first pass of a two-pass metric such as Mattes' sparse-support path,
    /// Correlation, or ANTS neighborhood correlation, which would otherwise
    /// pay for a gradient it immediately discards).
    ///
    /// For the nearest-neighbor and linear interpolators this skips the
    /// gradient *computation* itself, not just the return value ([`nearest_at`]
    /// / [`linear_at`] are genuinely value-only primitives). The cubic
    /// B-spline and Gaussian interpolators currently expose only a combined
    /// value-and-gradient primitive, so for those this still computes the
    /// gradient and discards it.
    pub(crate) fn value_at(&self, p: &[f64]) -> Option<f64> {
        let cidx = self.continuous_index(p);
        if !self.mask_allows(&cidx) {
            return None;
        }
        self.buf.with(ValueOnly {
            img: self,
            c: &cidx,
        })
    }

    /// The `(min, max)` of the moving-image buffer. `(0, 0)` when empty. The
    /// Mattes MI metric uses this to size the joint-histogram moving axis.
    pub(crate) fn value_range(&self) -> (f64, f64) {
        self.buf.min_max().unwrap_or((0.0, 0.0))
    }
}

/// The cubic B-spline prefilter, run on the moving image's native buffer.
struct Coefficients<'a> {
    size: &'a [usize],
    strides: &'a [usize],
}

impl WithBuf for Coefficients<'_> {
    type Out = Vec<f64>;

    fn call<T: Scalar>(self, buf: &[T]) -> Vec<f64> {
        bspline_coefficients(buf, self.size, self.strides)
    }
}

/// [`MovingImage::value_and_gradient`], monomorphized over the buffer's type.
struct ValueAndGradient<'a> {
    img: &'a MovingImage,
    c: &'a [f64],
}

impl WithBuf for ValueAndGradient<'_> {
    type Out = Option<(f64, Vec<f64>)>;

    fn call<T: Scalar>(self, buf: &[T]) -> Self::Out {
        let (size, strides, c) = (&self.img.size, &self.img.strides, self.c);
        let sinc = |w| windowed_sinc_value_and_gradient(buf, size, strides, c, w);
        match self.img.interpolator {
            Interpolator::NearestNeighbor => nearest_value_and_gradient(buf, size, strides, c),
            Interpolator::Linear => linear_value_and_gradient(buf, size, strides, c),
            Interpolator::BSpline => bspline_value_and_gradient(
                self.img
                    .bspline_coeffs
                    .as_deref()
                    .expect("bspline_coeffs is Some whenever interpolator == BSpline"),
                size,
                strides,
                c,
            ),
            Interpolator::Gaussian => gaussian_value_and_gradient(buf, size, strides, c),
            Interpolator::HammingWindowedSinc => sinc(SincWindow::Hamming),
            Interpolator::CosineWindowedSinc => sinc(SincWindow::Cosine),
            Interpolator::WelchWindowedSinc => sinc(SincWindow::Welch),
            Interpolator::LanczosWindowedSinc => sinc(SincWindow::Lanczos),
            Interpolator::BlackmanWindowedSinc => sinc(SincWindow::Blackman),
        }
    }
}

/// [`MovingImage::value_at`], monomorphized over the buffer's type.
struct ValueOnly<'a> {
    img: &'a MovingImage,
    c: &'a [f64],
}

impl WithBuf for ValueOnly<'_> {
    type Out = Option<f64>;

    fn call<T: Scalar>(self, buf: &[T]) -> Self::Out {
        let (size, strides, c) = (&self.img.size, &self.img.strides, self.c);
        match self.img.interpolator {
            Interpolator::NearestNeighbor => nearest_at(buf, size, strides, c),
            Interpolator::Linear => linear_at(buf, size, strides, c),
            _ => ValueAndGradient { img: self.img, c }
                .call(buf)
                .map(|(v, _)| v),
        }
    }
}

/// The value and parameter-derivative of a metric at one transform.
#[derive(Clone, Debug)]
pub struct MetricValue {
    /// Metric value (lower is better for mean squares).
    pub value: f64,
    /// `∂value/∂pₖ`, length = number of transform parameters.
    pub derivative: Vec<f64>,
    /// How many fixed samples mapped inside the moving image.
    pub valid_points: usize,
}

/// Compute backend for the mean-squares metric: the isolated, parallelizable
/// per-sample reduction. See the [module docs](self#gpu-seam) for the GPU seam.
///
/// Both reductions are required. A backend must not implement
/// [`mean_squares_value`](Self::mean_squares_value) by discarding
/// [`mean_squares`](Self::mean_squares)'s derivative: the gradient-free
/// optimizers call it once per objective evaluation, and the derivative costs
/// `O(nsamples · nparams)` to accumulate and is never read.
pub trait MetricBackend {
    /// Accumulate the mean-squares value and its parameter-derivative over all
    /// fixed samples for the given transform.
    fn mean_squares(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> MetricValue;

    /// Accumulate the mean-squares value alone, for a caller that does not need
    /// the derivative (the gradient-free optimizers, and the line searches'
    /// golden-section probes). `f64::MAX` when no sample is valid, matching
    /// [`mean_squares`](Self::mean_squares)'s value in that case.
    fn mean_squares_value(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> f64;
}

/// Host (CPU) implementation of [`MetricBackend`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl MetricBackend for CpuBackend {
    fn mean_squares(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> MetricValue {
        let nparams = transform.number_of_parameters();
        let n = fixed.len();

        let mut value_sum = 0.0;
        let mut deriv = vec![0.0; nparams];
        let mut valid = 0usize;

        // Sparseness is a property of the transform *type*, not of a point: a
        // transform that has a sparse Jacobian returns `Some` at every point,
        // empty where the point contributes nothing (see
        // `ParametricTransform::sparse_jacobian_wrt_parameters`). So this reads
        // it once, on the first sample, and picks the loop.
        let mut scratch = fixed.scratch();
        let sparse = n > 0
            && transform
                .sparse_jacobian_wrt_parameters(fixed.point(0, &mut scratch))
                .is_some();

        if sparse {
            // Sequential. A sample's sparse contribution is a scattered,
            // variable-length list of (parameter, column) entries; staging it
            // as a dense `nparams`-wide row for the parallel fold would cost
            // O(nparams) per sample and destroy the very sparsity this path
            // exists for. Left serial deliberately — see the metric's parallel
            // note in the module docs.
            for s in 0..n {
                let fp = fixed.point(s, &mut scratch);
                let mp = transform.transform_point(fp);
                let (mv, grad_phys) = match moving.value_and_physical_gradient(&mp) {
                    Some(vg) => vg,
                    None => continue,
                };

                let diff = mv - fixed.value(s);
                value_sum += diff * diff;

                // deriv_k += 2·diff · Σ_d grad_phys[d] · J[d][k], over the
                // affected parameters only.
                let entries = transform
                    .sparse_jacobian_wrt_parameters(fp)
                    .expect("a sparse transform returns Some at every point");
                for (idx, col) in &entries {
                    let g: f64 = col
                        .iter()
                        .zip(grad_phys.iter())
                        .map(|(&c, &gp)| c * gp)
                        .sum();
                    deriv[*idx] += 2.0 * diff * g;
                }

                valid += 1;
            }
        } else {
            // Parallel, and bit-identical to the loop above. Every sample's
            // contribution — the transform, the interpolation, the dense
            // Jacobian, the `2·diff·g` product for each parameter — is computed
            // in parallel into its own row and touches no accumulator. The
            // accumulators are then fed those rows on one thread, in sample
            // order, performing the exact sequence of `+=` a serial loop would.
            // No float sum is re-associated, so the result does not depend on
            // the thread count. See `sitk_core::parallel::map_rows_fold_in_order`.
            parallel::map_rows_fold_in_order(
                n,
                1 + nparams,
                || fixed.scratch(),
                |scratch, s, row| {
                    let fp = fixed.point(s, scratch);
                    let mp = transform.transform_point(fp);
                    let Some((mv, grad_phys)) = moving.value_and_physical_gradient(&mp) else {
                        return false;
                    };

                    let diff = mv - fixed.value(s);
                    row[0] = diff * diff;

                    let jac = transform.jacobian_wrt_parameters(fp);
                    for (k, slot) in row[1..].iter_mut().enumerate() {
                        let mut g = 0.0;
                        for (d, &gp) in grad_phys.iter().enumerate() {
                            g += gp * jac[d * nparams + k];
                        }
                        *slot = 2.0 * diff * g;
                    }
                    true
                },
                |_, row| {
                    value_sum += row[0];
                    for (dk, &contribution) in deriv.iter_mut().zip(&row[1..]) {
                        *dk += contribution;
                    }
                    valid += 1;
                },
            );
        }

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }
        let inv = 1.0 / valid as f64;
        MetricValue {
            value: value_sum * inv,
            derivative: deriv.iter().map(|d| d * inv).collect(),
            valid_points: valid,
        }
    }

    fn mean_squares_value(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> f64 {
        let n = fixed.len();

        let mut value_sum = 0.0;
        let mut valid = 0usize;

        // Parallel per sample, accumulated on one thread in sample order — the
        // same additions in the same order as a serial loop, so the value is
        // bit-identical at any thread count. No Jacobian here, so this needs no
        // dense/sparse split: it covers every transform.
        parallel::map_rows_fold_in_order(
            n,
            1,
            || fixed.scratch(),
            |scratch, s, row| {
                let fp = fixed.point(s, scratch);
                let mp = transform.transform_point(fp);
                // No gradient, no Jacobian: `value_at` decides validity by exactly
                // the same predicate `value_and_physical_gradient` does, so this
                // walks the identical sample set as `mean_squares`.
                let Some(mv) = moving.value_at(&mp) else {
                    return false;
                };
                let diff = mv - fixed.value(s);
                row[0] = diff * diff;
                true
            },
            |_, row| {
                value_sum += row[0];
                valid += 1;
            },
        );

        if valid == 0 {
            return f64::MAX;
        }
        value_sum / valid as f64
    }
}

/// The mean-squares image-to-image metric. Holds the precomputed fixed samples
/// and moving image; [`evaluate`](Self::evaluate) returns value + derivative for
/// a given transform through the chosen backend.
pub struct MeanSquaresMetric {
    fixed: FixedSamples,
    moving: MovingImage,
}

impl MeanSquaresMetric {
    /// Build the metric from a fixed and moving image. Fails if dimensions
    /// disagree or the moving direction matrix is singular.
    pub fn new(fixed: &Image, moving: &Image) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        Ok(Self {
            fixed: FixedSamples::from_image(fixed)?,
            moving: MovingImage::from_image(moving)?,
        })
    }

    /// Build the metric from an already-configured [`FixedSamples`] and
    /// [`MovingImage`] — the seam for a custom sampling strategy, fixed/moving
    /// mask, or interpolator (see [`FixedSamples::from_image_with`] and
    /// [`MovingImage::from_image_with_interpolator`]). Fails if their spatial
    /// dimensions disagree.
    pub fn from_samples(fixed: FixedSamples, moving: MovingImage) -> Result<Self> {
        if fixed.dim != moving.dim() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dim,
                moving: moving.dim(),
            });
        }
        Ok(Self { fixed, moving })
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// Build a scale/learning-rate estimator of `kind` for `transform` over
    /// this metric's virtual domain (ITK's
    /// `RegistrationParameterScalesEstimator` hierarchy).
    pub fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        self.fixed.scales_estimator(transform, &self.moving, kind)
    }

    /// Evaluate value + derivative for `transform` using `backend`.
    pub fn evaluate(
        &self,
        transform: &dyn ParametricTransform,
        backend: &dyn MetricBackend,
    ) -> MetricValue {
        backend.mean_squares(&self.fixed, &self.moving, transform)
    }

    /// The metric value alone at `transform`, skipping the derivative
    /// accumulation entirely (see [`MetricBackend::mean_squares_value`]).
    pub fn value(&self, transform: &dyn ParametricTransform, backend: &dyn MetricBackend) -> f64 {
        backend.mean_squares_value(&self.fixed, &self.moving, transform)
    }
}

/// The one contiguous parameter block a [local-support] transform's Jacobian
/// touches at `point`, as `(block offset, dim × numberOfLocalParameters
/// row-major block)`.
///
/// A local-support transform is a displacement field (that is exactly what
/// ITK's `DisplacementField` transform category means), so its sparse Jacobian
/// at a sample is its own pixel's `numberOfLocalParameters` *consecutive*
/// parameters: entry `i` is `(offset + i, column i)`. This transposes that
/// entry list back into the dense block, which is the shape every consumer of
/// the local path wants.
///
/// Returns `None` when the sample falls outside the field — the sparse accessor
/// answers with an empty entry list there, meaning "zero Jacobian, no affected
/// parameters", so the sample contributes no shift and every caller drops it.
///
/// # Panics
///
/// Panics if `transform` has no sparse Jacobian at all. Call only when
/// [`ParametricTransform::has_local_support`] is `true`; every such transform
/// implements [`ParametricTransform::sparse_jacobian_wrt_parameters`].
///
/// [local-support]: ParametricTransform::has_local_support
pub(crate) fn local_support_block(
    transform: &dyn ParametricTransform,
    point: &[f64],
) -> Option<(usize, Vec<f64>)> {
    debug_assert!(transform.has_local_support());
    let dim = transform.dimension();
    let num_local = transform.number_of_local_parameters();
    let entries = transform
        .sparse_jacobian_wrt_parameters(point)
        .expect("a local-support transform always has a sparse Jacobian");
    if entries.is_empty() {
        return None;
    }
    debug_assert_eq!(entries.len(), num_local);

    let offset = entries[0].0;
    let mut block = vec![0.0; dim * num_local];
    for (i, (index, column)) in entries.iter().enumerate() {
        debug_assert_eq!(*index, offset + i);
        for (r, &c) in column.iter().enumerate() {
            block[r * num_local + i] = c;
        }
    }
    Some((offset, block))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::{Euler3DTransform, TransformBase, TranslationTransform};

    // A separable ramp f(x,y) = 3x + 5y makes the mean-squares gradient exactly
    // analytic, so we can check the derivative sign and magnitude precisely.
    fn ramp(w: usize, h: usize, ax: f64, ay: f64) -> Image {
        let mut v = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = ax * x as f64 + ay * y as f64;
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    /// The parallel metric must return the **bits** of the serial loop it
    /// replaced, at every thread count — not a value within some tolerance. The
    /// optimizer is a feedback loop, so a metric that shifted by one ulp would
    /// walk to a different registration result. The reference here is the serial
    /// loop itself, written out, not a `with_threads(1)` run of the same code:
    /// that would only prove thread-independence, not that the sum is still the
    /// sequential sum.
    #[test]
    fn mean_squares_is_bit_identical_to_the_serial_loop_at_every_thread_count() {
        // 40³ = 64 000 samples, well past the parallel threshold, with an
        // intensity pattern whose sum is rounding-sensitive (any re-association
        // of the accumulation shows up in the low bits).
        let n = 40usize;
        let wave = |phase: f64| {
            let mut v = vec![0.0f64; n * n * n];
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        let (fx, fy, fz) = (x as f64, y as f64, z as f64);
                        v[(z * n + y) * n + x] =
                            137.0 * (0.7 * fx + 1.3 * fy + 2.1 * fz + phase).sin() + 0.001 * fx;
                    }
                }
            }
            Image::from_vec(&[n, n, n], v).unwrap()
        };
        let metric = MeanSquaresMetric::new(&wave(0.0), &wave(0.35)).unwrap();
        // Rigid Euler3D: 6 dense parameters, the benchmark's transform.
        let t = Euler3DTransform::new(0.11, -0.07, 0.05, [1.5, -2.5, 0.75], [20.0, 20.0, 20.0]);

        // The serial loop, verbatim — the code the parallel path replaced.
        let (fixed, moving) = (&metric.fixed, &metric.moving);
        let nparams = t.number_of_parameters();
        let mut want_value = 0.0f64;
        let mut want_deriv = vec![0.0f64; nparams];
        let mut want_valid = 0usize;
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            let mp = t.transform_point(fp);
            let Some((mv, grad_phys)) = moving.value_and_physical_gradient(&mp) else {
                continue;
            };
            let diff = mv - fixed.value(s);
            want_value += diff * diff;
            let jac = t.jacobian_wrt_parameters(fp);
            for (k, dk) in want_deriv.iter_mut().enumerate() {
                let mut g = 0.0;
                for (d, &gp) in grad_phys.iter().enumerate() {
                    g += gp * jac[d * nparams + k];
                }
                *dk += 2.0 * diff * g;
            }
            want_valid += 1;
        }
        // The primitive runs serially below 1<<14 *samples* (valid or not), so
        // the parallel path must actually be the one under test here — and
        // enough samples must survive the transform to make the sum nontrivial.
        assert!(
            fixed.len() > (1 << 14),
            "the parallel path must be taken: only {} samples",
            fixed.len()
        );
        assert!(want_valid > 1000, "only {want_valid} valid samples");
        let inv = 1.0 / want_valid as f64;
        let want_value = want_value * inv;
        let want_deriv: Vec<f64> = want_deriv.iter().map(|d| d * inv).collect();
        let mut want_only = 0.0f64;
        let mut only_valid = 0usize;
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            let Some(mv) = moving.value_at(&t.transform_point(fp)) else {
                continue;
            };
            let diff = mv - fixed.value(s);
            want_only += diff * diff;
            only_valid += 1;
        }
        let want_only = want_only / only_valid as f64;

        // This fixture has teeth: re-associating the very same contributions
        // into 64-wide chunks and folding the partials in order — a textbook
        // "deterministic" chunked reduction, and exactly what this metric must
        // *not* do — lands on different bits. So the assertions below would
        // catch a re-association, rather than passing because the sum happens to
        // be exact.
        let mut contributions = Vec::new();
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            if let Some(mv) = moving.value_at(&t.transform_point(fp)) {
                let diff = mv - fixed.value(s);
                contributions.push(diff * diff);
            }
        }
        let chunked: f64 = contributions
            .chunks(64)
            .map(|c| c.iter().sum::<f64>())
            .sum::<f64>()
            / only_valid as f64;
        assert_ne!(
            chunked.to_bits(),
            want_only.to_bits(),
            "fixture is not rounding-sensitive, so the bit assertions prove nothing"
        );

        for threads in [1usize, 2, 3, 8, 32] {
            let (got, got_only) = sitk_core::parallel::with_threads(threads, || {
                (
                    metric.evaluate(&t, &CpuBackend),
                    metric.value(&t, &CpuBackend),
                )
            });
            assert_eq!(got.valid_points, want_valid, "{threads} threads");
            assert_eq!(
                got.value.to_bits(),
                want_value.to_bits(),
                "{threads} threads moved the value: {} vs {want_value}",
                got.value
            );
            for (k, (&g, &w)) in got.derivative.iter().zip(&want_deriv).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{threads} threads moved derivative[{k}]: {g} vs {w}"
                );
            }
            assert_eq!(
                got_only.to_bits(),
                want_only.to_bits(),
                "{threads} threads moved the value-only reduction: {got_only} vs {want_only}"
            );
        }
    }

    #[test]
    fn identity_on_equal_images_is_zero_with_zero_gradient() {
        let img = ramp(8, 8, 3.0, 5.0);
        let metric = MeanSquaresMetric::new(&img, &img).unwrap();
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let r = metric.evaluate(&t, &CpuBackend);
        assert!(r.value.abs() < 1e-9, "value {}", r.value);
        assert!(r.derivative[0].abs() < 1e-6, "d/dtx {}", r.derivative[0]);
        assert!(r.derivative[1].abs() < 1e-6, "d/dty {}", r.derivative[1]);
        assert_eq!(r.valid_points, 64);
    }

    #[test]
    fn derivative_matches_finite_difference() {
        // Fixed is the ramp; moving is the same ramp. Evaluate the metric as a
        // function of the translation parameters at a nonzero point and compare
        // the analytic derivative to a central finite difference.
        let fixed = ramp(12, 12, 3.0, 5.0);
        let moving = ramp(12, 12, 3.0, 5.0);
        let metric = MeanSquaresMetric::new(&fixed, &moving).unwrap();

        // Offsets chosen off any half-integer so no sample sits on the
        // is_inside boundary (which would flip validity under ±h and break the
        // finite difference).
        let p0 = [0.3f64, -0.4];
        let eval = |p: &[f64]| {
            let t = TranslationTransform::new(p.to_vec());
            metric.evaluate(&t, &CpuBackend)
        };
        let analytic = eval(&p0).derivative;

        let h = 1e-4;
        for k in 0..2 {
            let mut pp = p0;
            pp[k] += h;
            let mut pm = p0;
            pm[k] -= h;
            let fd = (eval(&pp).value - eval(&pm).value) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-3,
                "param {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    /// A flat-index-valued image: `v[i] = i`, so a sample's value pins down
    /// exactly which flat voxel it came from.
    fn indexed(w: usize, h: usize) -> Image {
        let n = w * h;
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn regular_sampling_gives_expected_count_and_stride() {
        let (w, h) = (10, 10);
        let img = indexed(w, h);

        // percentage 0.1 -> stride = ceil(1/0.1) = 10 -> samples at flat
        // indices 0, 10, 20, ..., 90.
        let samples =
            FixedSamples::from_image_with(&img, SamplingStrategy::Regular, 0.1, 0, None).unwrap();
        assert_eq!(samples.len(), 10);
        let expected: Vec<f64> = (0..10).map(|k| (k * 10) as f64).collect();
        let got: Vec<f64> = (0..samples.len()).map(|s| samples.value(s)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn random_sampling_is_reproducible_and_matches_percentage() {
        let (w, h) = (20, 20);
        let n = w * h;
        let img = indexed(w, h);
        let percentage = 0.25;
        let expected_count = (n as f64 * percentage) as usize;

        let a = FixedSamples::from_image_with(&img, SamplingStrategy::Random, percentage, 42, None)
            .unwrap();
        let b = FixedSamples::from_image_with(&img, SamplingStrategy::Random, percentage, 42, None)
            .unwrap();
        assert_eq!(a.len(), expected_count);
        assert_eq!(a.values, b.values, "same seed must draw the same samples");
        assert_eq!(a.points, b.points);

        let c = FixedSamples::from_image_with(&img, SamplingStrategy::Random, percentage, 43, None)
            .unwrap();
        assert_ne!(
            a.values, c.values,
            "a different seed should draw different samples"
        );
    }

    #[test]
    fn fixed_mask_halves_sample_count() {
        let (w, h) = (10, 10);
        let n = w * h;
        let img = Image::from_vec(&[w, h], vec![1.0; n]).unwrap();
        // Mask the first half of voxels (by flat, dim-0-fastest index) in.
        let mut mv = vec![0.0f64; n];
        mv[..n / 2].fill(1.0);
        let mask = Image::from_vec(&[w, h], mv).unwrap();

        let samples =
            FixedSamples::from_image_with(&img, SamplingStrategy::None, 1.0, 0, Some(&mask))
                .unwrap();
        assert_eq!(samples.len(), n / 2);
    }

    #[test]
    fn moving_mask_invalidates_previously_valid_points() {
        let img = ramp(8, 8, 3.0, 5.0);
        let t = TranslationTransform::new(vec![0.0, 0.0]);

        let unmasked = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&img).unwrap(),
            MovingImage::from_image(&img).unwrap(),
        )
        .unwrap();
        let before = unmasked.evaluate(&t, &CpuBackend).valid_points;
        assert_eq!(before, 64);

        // Mask out the right half of the moving image (x >= 4).
        let mut mv = vec![0.0f64; 64];
        for y in 0..8 {
            for x in 0..4 {
                mv[y * 8 + x] = 1.0;
            }
        }
        let mask = Image::from_vec(&[8, 8], mv).unwrap();
        let masked_moving = MovingImage::from_image(&img)
            .unwrap()
            .with_moving_mask(&mask)
            .unwrap();
        let masked =
            MeanSquaresMetric::from_samples(FixedSamples::from_image(&img).unwrap(), masked_moving)
                .unwrap();
        let after = masked.evaluate(&t, &CpuBackend).valid_points;
        assert!(
            after < before,
            "moving mask should drop previously-valid points: after {after} vs before {before}"
        );
    }

    /// The pre-fast-path mean-squares reduction: always uses the dense
    /// `jacobian_wrt_parameters` contract, never the sparse accessor. Kept
    /// here only as an independent reference to verify `CpuBackend`'s sparse
    /// fast path (which takes the sparse branch automatically for any
    /// transform that implements it) computes the identical result.
    fn mean_squares_dense_reference(
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> MetricValue {
        let nparams = transform.number_of_parameters();
        let n = fixed.len();

        let mut value_sum = 0.0;
        let mut deriv = vec![0.0; nparams];
        let mut valid = 0usize;

        let mut scratch = fixed.scratch();
        for s in 0..n {
            let fp = fixed.point(s, &mut scratch);
            let fv = fixed.value(s);

            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue,
            };

            let diff = mv - fv;
            value_sum += diff * diff;

            let jac = transform.jacobian_wrt_parameters(fp);
            for (k, dk) in deriv.iter_mut().enumerate() {
                let mut g = 0.0;
                for (d, &gp) in grad_phys.iter().enumerate() {
                    g += gp * jac[d * nparams + k];
                }
                *dk += 2.0 * diff * g;
            }

            valid += 1;
        }

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }
        let inv = 1.0 / valid as f64;
        MetricValue {
            value: value_sum * inv,
            derivative: deriv.iter().map(|d| d * inv).collect(),
            valid_points: valid,
        }
    }

    #[test]
    fn bspline_sparse_derivative_matches_dense_reference() {
        use sitk_transform::BSplineTransform;

        let fixed = ramp(16, 16, 3.0, 5.0);
        let moving = ramp(16, 16, 3.0, 5.0);
        let fixed_samples = FixedSamples::from_image(&fixed).unwrap();
        let moving_image = MovingImage::from_image(&moving).unwrap();

        let mut t = BSplineTransform::from_image_domain(&fixed, &[4, 4]).unwrap();
        let n = t.number_of_parameters();
        let params: Vec<f64> = (0..n)
            .map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.05)
            .collect();
        t.set_parameters(&params).unwrap();

        let sparse = CpuBackend.mean_squares(&fixed_samples, &moving_image, &t);
        let dense = mean_squares_dense_reference(&fixed_samples, &moving_image, &t);

        assert_eq!(sparse.valid_points, dense.valid_points);
        assert!((sparse.value - dense.value).abs() < 1e-12);
        assert_eq!(sparse.derivative.len(), dense.derivative.len());
        let max_diff = sparse
            .derivative
            .iter()
            .zip(&dense.derivative)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "max derivative diff {max_diff}");
    }

    #[test]
    fn displacement_field_sparse_derivative_matches_dense_reference() {
        use sitk_transform::DisplacementFieldTransform;

        let img = ramp(8, 8, 3.0, 5.0);
        let fixed_samples = FixedSamples::from_image(&img).unwrap();
        let moving_image = MovingImage::from_image(&img).unwrap();

        let mut t = DisplacementFieldTransform::from_image_domain(&img).unwrap();
        let n = t.number_of_parameters();
        let params: Vec<f64> = (0..n).map(|i| ((i * 13 % 7) as f64 - 3.0) * 0.05).collect();
        t.set_parameters(&params).unwrap();

        let sparse = CpuBackend.mean_squares(&fixed_samples, &moving_image, &t);
        let dense = mean_squares_dense_reference(&fixed_samples, &moving_image, &t);

        assert_eq!(sparse.valid_points, dense.valid_points);
        assert!((sparse.value - dense.value).abs() < 1e-12);
        let max_diff = sparse
            .derivative
            .iter()
            .zip(&dense.derivative)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "max derivative diff {max_diff}");
    }

    #[test]
    fn value_agrees_with_evaluate() {
        // `mean_squares_value` walks the samples with `value_at` where
        // `mean_squares` uses `value_and_physical_gradient`. Both must accept
        // the same samples and sum the same squares.
        let fixed = ramp(9, 7, 3.0, 5.0);
        let moving = ramp(9, 7, 3.0, 5.0);
        let metric = MeanSquaresMetric::new(&fixed, &moving).unwrap();
        for t in [[0.0, 0.0], [1.3, -0.7], [-2.5, 2.5]] {
            let transform = TranslationTransform::new(t.to_vec());
            let full = metric.evaluate(&transform, &CpuBackend).value;
            let value_only = metric.value(&transform, &CpuBackend);
            assert!(
                (full - value_only).abs() <= 1e-12 * full.abs().max(1.0),
                "at {t:?}: evaluate {full} vs value {value_only}"
            );
        }
    }
}
