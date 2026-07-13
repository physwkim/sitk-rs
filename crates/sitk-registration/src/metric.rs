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

use sitk_core::Image;
use sitk_core::parallel;
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

/// The physical point at multi-index `index`, via `origin + idx_to_phys ·
/// index`, written into `out` (length `dim`). Shared by every [`FixedSamples`]
/// sampling strategy.
///
/// Writes rather than returns: at 256³ this runs 16.7 M times, and returning a
/// `Vec` made it 16.7 M heap allocations — which is most of what made building
/// the sample set the dominant cost of a GPU registration run.
fn write_point_at(
    out: &mut [f64],
    idx_to_phys: &[f64],
    origin: &[f64],
    dim: usize,
    index: &[usize],
) {
    for (r, pr) in out.iter_mut().enumerate() {
        let mut acc = origin[r];
        for (c, &idx) in index.iter().enumerate() {
            acc += idx_to_phys[r * dim + c] * idx as f64;
        }
        *pr = acc;
    }
}

/// The physical point of every voxel of `size`, row-major `N × dim`, in the
/// dim-0-fastest traversal order [`increment`] produces.
///
/// The buffer is 402 MB at 256³ and freshly allocated, so *the fill is also the
/// first touch*: Linux faults and zeroes one page per 4 KB as it is written.
/// [`sitk_core::parallel::map_indexed_init`] spreads that fault storm across the
/// pool and, because it writes into uninitialized capacity, it writes each page
/// exactly once — the `vec![0.0; n * dim]` this replaces faulted the whole buffer
/// serially first and then overwrote it.
///
/// Every element is a pure function of its own index — the same `origin[r] + Σ_c
/// idx_to_phys[r][c] · index[c]` accumulated in the same order as
/// [`write_point_at`] — so this is bit-for-bit the serial loop's output at any
/// thread count, and it now honors the caller's rayon pool rather than spawning a
/// hardcoded 16 threads behind [`sitk_core::parallel::with_threads`]'s back.
fn grid_points(
    n: usize,
    size: &[usize],
    idx_to_phys: &[f64],
    origin: &[f64],
    dim: usize,
) -> Vec<f64> {
    parallel::map_indexed_init(
        n * dim,
        || vec![0usize; dim],
        |index, i| {
            write_multi_index(index, i / dim, size);
            let r = i % dim;
            let mut acc = origin[r];
            for (c, &idx) in index.iter().enumerate() {
                acc += idx_to_phys[r * dim + c] * idx as f64;
            }
            acc
        },
    )
}

/// The multi-index (dim-0-fastest) of flat voxel index `flat`, written into
/// `out` (length `size.len()`) — the inverse of the traversal order
/// [`increment`] produces.
fn write_multi_index(out: &mut [usize], mut flat: usize, size: &[usize]) {
    for (d, id) in out.iter_mut().enumerate() {
        *id = flat % size[d];
        flat /= size[d];
    }
}

/// [`write_multi_index`] into a fresh `Vec`, for the sampling strategies that
/// need one index at a time rather than all of them.
fn linear_to_multi(flat: usize, size: &[usize]) -> Vec<usize> {
    let mut index = vec![0usize; size.len()];
    write_multi_index(&mut index, flat, size);
    index
}

/// The fixed image reduced to its sample set (the registration *virtual
/// domain*): every pixel's value and its physical point, precomputed once.
pub struct FixedSamples {
    pub(crate) dim: usize,
    /// Identity for a device-resident copy of these buffers — see [`next_id`].
    #[cfg(feature = "cuda")]
    pub(crate) id: u64,
    /// Whether the sample set is *every* voxel of [`grid`](Self::grid), in that
    /// grid's traversal order — true for the unsampled, unmasked default.
    ///
    /// When it holds, [`points`](Self::points) is a 402 MB memo (at 256³) of a
    /// nine-flop function of the sample index, and a device backend can derive
    /// each point rather than be sent it. The CPU path reads `points` regardless,
    /// so the buffer still exists; what this flag removes is the *upload*.
    #[cfg(feature = "cuda")]
    pub(crate) full_grid: bool,
    /// One value per sample, length `N`.
    pub(crate) values: Vec<f64>,
    /// Physical points, row-major `N × dim`.
    pub(crate) points: Vec<f64>,
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
    /// `pub(crate)` rather than private because a device backend derives its
    /// sample points from this geometry — see [`full_grid`](Self::full_grid).
    pub(crate) grid: VirtualGrid,
}

impl FixedSamples {
    /// Reduce a fixed image to its full sample set (sampling strategy = None:
    /// every pixel, matching SimpleITK's default).
    ///
    /// Fails on a vector `fixed` image, like every scalar consumer of
    /// [`sitk_core::Image::to_f64_vec`].
    pub fn from_image(fixed: &Image) -> Result<Self> {
        let dim = fixed.dimension();
        let size = fixed.size().to_vec();
        let values = fixed.to_f64_vec()?;
        let n = values.len();

        // point = origin + (D · diag(spacing)) · index
        let idx_to_phys = index_to_physical_matrix(fixed.direction(), fixed.spacing(), dim);
        let origin = fixed.origin();

        let points = grid_points(n, &size, &idx_to_phys, origin, dim);

        let min_spacing = fixed
            .spacing()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        Ok(Self {
            dim,
            #[cfg(feature = "cuda")]
            id: next_id(),
            #[cfg(feature = "cuda")]
            full_grid: true,
            values,
            points,
            min_spacing,
            grid: VirtualGrid::new(dim, size, origin.to_vec(), idx_to_phys),
        })
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
        let values_all = fixed.to_f64_vec()?;
        let n = values_all.len();

        let idx_to_phys = index_to_physical_matrix(fixed.direction(), fixed.spacing(), dim);
        let origin = fixed.origin();

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
        // The sample set is the grid itself exactly when nothing filters it.
        #[cfg(feature = "cuda")]
        let full_grid = matches!(strategy, SamplingStrategy::None) && mask_buf.is_none();

        let mask_allows = |flat: usize| match &mask_buf {
            None => true,
            Some(m) => m[flat] != 0.0,
        };

        let mut values;
        let mut points;
        // One scratch point, reused: `write_point_at` writes into it and it is
        // appended. The `Vec` it replaces was allocated once per sample.
        let mut scratch = vec![0.0; dim];

        match strategy {
            // Every voxel, nothing masked out: the sample set *is* the grid, in
            // grid order. There is nothing to filter, so nothing needs pushing —
            // take the values buffer whole and fill the points in parallel.
            SamplingStrategy::None if mask_buf.is_none() => {
                points = grid_points(n, &size, &idx_to_phys, origin, dim);
                values = values_all;
            }
            SamplingStrategy::None => {
                values = Vec::with_capacity(n);
                points = Vec::with_capacity(n * dim);
                let mut index = vec![0usize; dim];
                for (s, &fv) in values_all.iter().enumerate() {
                    if mask_allows(s) {
                        values.push(fv);
                        write_point_at(&mut scratch, &idx_to_phys, origin, dim, &index);
                        points.extend_from_slice(&scratch);
                    }
                    increment(&mut index, &size);
                }
            }
            SamplingStrategy::Regular => {
                let stride = ((1.0 / percentage).ceil() as usize).max(1);
                let expect = n.div_ceil(stride);
                values = Vec::with_capacity(expect);
                points = Vec::with_capacity(expect * dim);
                let mut index = vec![0usize; dim];
                for (s, &fv) in values_all.iter().enumerate() {
                    if s % stride == 0 && mask_allows(s) {
                        values.push(fv);
                        write_point_at(&mut scratch, &idx_to_phys, origin, dim, &index);
                        points.extend_from_slice(&scratch);
                    }
                    increment(&mut index, &size);
                }
            }
            SamplingStrategy::Random => {
                let sample_count = (n as f64 * percentage) as usize;
                values = Vec::with_capacity(sample_count);
                points = Vec::with_capacity(sample_count * dim);
                let mut rng = SplitMix64::new(seed);
                for _ in 0..sample_count {
                    let flat = rng.next_below(n);
                    if mask_allows(flat) {
                        let index = linear_to_multi(flat, &size);
                        values.push(values_all[flat]);
                        write_point_at(&mut scratch, &idx_to_phys, origin, dim, &index);
                        points.extend_from_slice(&scratch);
                    }
                }
            }
        }

        let min_spacing = fixed
            .spacing()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        Ok(Self {
            dim,
            #[cfg(feature = "cuda")]
            id: next_id(),
            #[cfg(feature = "cuda")]
            full_grid,
            values,
            points,
            min_spacing,
            grid: VirtualGrid::new(dim, size, origin.to_vec(), idx_to_phys),
        })
    }

    /// Number of samples `N`.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The `(min, max)` of the sampled fixed-image values. `(0, 0)` when empty.
    /// This is the fixed-image intensity range over the analysis region (full
    /// sampling ⇒ the whole image), which the Mattes MI metric uses to size the
    /// joint-histogram fixed axis.
    pub(crate) fn value_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &self.values {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if self.values.is_empty() {
            (0.0, 0.0)
        } else {
            (lo, hi)
        }
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
    pub(crate) buf: &'a [f64],
    pub(crate) size: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) origin: &'a [f64],
    pub(crate) phys_to_index: &'a [f64],
    pub(crate) interpolator: Interpolator,
    pub(crate) mask: Option<&'a [bool]>,
}

/// The moving image as an `f64` buffer plus the geometry needed to map a
/// physical point to a continuous index and to convert an index-space gradient
/// to a physical-space gradient.
pub struct MovingImage {
    dim: usize,
    /// Identity for a device-resident copy of this buffer — see [`next_id`].
    #[cfg(feature = "cuda")]
    pub(crate) id: u64,
    buf: Vec<f64>,
    size: Vec<usize>,
    strides: Vec<usize>,
    origin: Vec<f64>,
    /// `diag(1/spacing) · D⁻¹`, row-major `dim × dim`: maps a physical
    /// displacement from the origin to a continuous index.
    phys_to_index: Vec<f64>,
    interpolator: Interpolator,
    /// Precomputed cubic B-spline coefficients, present only when
    /// `interpolator == BSpline` (see [`bspline_coefficients`]).
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
        let buf = moving.to_f64_vec()?;
        let bspline_coeffs = matches!(interpolator, Interpolator::BSpline)
            .then(|| bspline_coefficients(&buf, &size, &strides_v));
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
        match self.interpolator {
            Interpolator::NearestNeighbor => {
                nearest_value_and_gradient(&self.buf, &self.size, &self.strides, c)
            }
            Interpolator::Linear => {
                linear_value_and_gradient(&self.buf, &self.size, &self.strides, c)
            }
            Interpolator::BSpline => bspline_value_and_gradient(
                self.bspline_coeffs
                    .as_deref()
                    .expect("bspline_coeffs is Some whenever interpolator == BSpline"),
                &self.size,
                &self.strides,
                c,
            ),
            Interpolator::Gaussian => {
                gaussian_value_and_gradient(&self.buf, &self.size, &self.strides, c)
            }
            Interpolator::HammingWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                c,
                SincWindow::Hamming,
            ),
            Interpolator::CosineWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                c,
                SincWindow::Cosine,
            ),
            Interpolator::WelchWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                c,
                SincWindow::Welch,
            ),
            Interpolator::LanczosWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                c,
                SincWindow::Lanczos,
            ),
            Interpolator::BlackmanWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                c,
                SincWindow::Blackman,
            ),
        }
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
        match self.interpolator {
            Interpolator::NearestNeighbor => {
                nearest_at(&self.buf, &self.size, &self.strides, &cidx)
            }
            Interpolator::Linear => linear_at(&self.buf, &self.size, &self.strides, &cidx),
            Interpolator::BSpline => bspline_value_and_gradient(
                self.bspline_coeffs
                    .as_deref()
                    .expect("bspline_coeffs is Some whenever interpolator == BSpline"),
                &self.size,
                &self.strides,
                &cidx,
            )
            .map(|(v, _)| v),
            Interpolator::Gaussian => {
                gaussian_value_and_gradient(&self.buf, &self.size, &self.strides, &cidx)
                    .map(|(v, _)| v)
            }
            Interpolator::HammingWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                &cidx,
                SincWindow::Hamming,
            )
            .map(|(v, _)| v),
            Interpolator::CosineWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                &cidx,
                SincWindow::Cosine,
            )
            .map(|(v, _)| v),
            Interpolator::WelchWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                &cidx,
                SincWindow::Welch,
            )
            .map(|(v, _)| v),
            Interpolator::LanczosWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                &cidx,
                SincWindow::Lanczos,
            )
            .map(|(v, _)| v),
            Interpolator::BlackmanWindowedSinc => windowed_sinc_value_and_gradient(
                &self.buf,
                &self.size,
                &self.strides,
                &cidx,
                SincWindow::Blackman,
            )
            .map(|(v, _)| v),
        }
    }

    /// The `(min, max)` of the moving-image buffer. `(0, 0)` when empty. The
    /// Mattes MI metric uses this to size the joint-histogram moving axis.
    pub(crate) fn value_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &self.buf {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if self.buf.is_empty() {
            (0.0, 0.0)
        } else {
            (lo, hi)
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
        let dim = fixed.dim;
        let nparams = transform.number_of_parameters();
        let n = fixed.values.len();

        let mut value_sum = 0.0;
        let mut deriv = vec![0.0; nparams];
        let mut valid = 0usize;

        // Sparseness is a property of the transform *type*, not of a point: a
        // transform that has a sparse Jacobian returns `Some` at every point,
        // empty where the point contributes nothing (see
        // `ParametricTransform::sparse_jacobian_wrt_parameters`). So this reads
        // it once, on the first sample, and picks the loop.
        let sparse = n > 0
            && transform
                .sparse_jacobian_wrt_parameters(&fixed.points[..dim])
                .is_some();

        if sparse {
            // Sequential. A sample's sparse contribution is a scattered,
            // variable-length list of (parameter, column) entries; staging it
            // as a dense `nparams`-wide row for the parallel fold would cost
            // O(nparams) per sample and destroy the very sparsity this path
            // exists for. Left serial deliberately — see the metric's parallel
            // note in the module docs.
            for s in 0..n {
                let fp = &fixed.points[s * dim..(s + 1) * dim];
                let mp = transform.transform_point(fp);
                let (mv, grad_phys) = match moving.value_and_physical_gradient(&mp) {
                    Some(vg) => vg,
                    None => continue,
                };

                let diff = mv - fixed.values[s];
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
                || (),
                |(), s, row| {
                    let fp = &fixed.points[s * dim..(s + 1) * dim];
                    let mp = transform.transform_point(fp);
                    let Some((mv, grad_phys)) = moving.value_and_physical_gradient(&mp) else {
                        return false;
                    };

                    let diff = mv - fixed.values[s];
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
        let dim = fixed.dim;
        let n = fixed.values.len();

        let mut value_sum = 0.0;
        let mut valid = 0usize;

        // Parallel per sample, accumulated on one thread in sample order — the
        // same additions in the same order as a serial loop, so the value is
        // bit-identical at any thread count. No Jacobian here, so this needs no
        // dense/sparse split: it covers every transform.
        parallel::map_rows_fold_in_order(
            n,
            1,
            || (),
            |(), s, row| {
                let fp = &fixed.points[s * dim..(s + 1) * dim];
                let mp = transform.transform_point(fp);
                // No gradient, no Jacobian: `value_at` decides validity by exactly
                // the same predicate `value_and_physical_gradient` does, so this
                // walks the identical sample set as `mean_squares`.
                let Some(mv) = moving.value_at(&mp) else {
                    return false;
                };
                let diff = mv - fixed.values[s];
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

/// Increment a multi-index in place (first index fastest).
fn increment(index: &mut [usize], size: &[usize]) {
    for d in 0..index.len() {
        index[d] += 1;
        if index[d] < size[d] {
            return;
        }
        index[d] = 0;
    }
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
        let (dim, nparams) = (fixed.dim, t.number_of_parameters());
        let mut want_value = 0.0f64;
        let mut want_deriv = vec![0.0f64; nparams];
        let mut want_valid = 0usize;
        for s in 0..fixed.values.len() {
            let fp = &fixed.points[s * dim..(s + 1) * dim];
            let mp = t.transform_point(fp);
            let Some((mv, grad_phys)) = moving.value_and_physical_gradient(&mp) else {
                continue;
            };
            let diff = mv - fixed.values[s];
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
            fixed.values.len() > (1 << 14),
            "the parallel path must be taken: only {} samples",
            fixed.values.len()
        );
        assert!(want_valid > 1000, "only {want_valid} valid samples");
        let inv = 1.0 / want_valid as f64;
        let want_value = want_value * inv;
        let want_deriv: Vec<f64> = want_deriv.iter().map(|d| d * inv).collect();
        let mut want_only = 0.0f64;
        let mut only_valid = 0usize;
        for s in 0..fixed.values.len() {
            let fp = &fixed.points[s * dim..(s + 1) * dim];
            let Some(mv) = moving.value_at(&t.transform_point(fp)) else {
                continue;
            };
            let diff = mv - fixed.values[s];
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
        for s in 0..fixed.values.len() {
            let fp = &fixed.points[s * dim..(s + 1) * dim];
            if let Some(mv) = moving.value_at(&t.transform_point(fp)) {
                let diff = mv - fixed.values[s];
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
        assert_eq!(samples.values, expected);
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
        let dim = fixed.dim;
        let nparams = transform.number_of_parameters();
        let n = fixed.values.len();

        let mut value_sum = 0.0;
        let mut deriv = vec![0.0; nparams];
        let mut valid = 0usize;

        for s in 0..n {
            let fp = &fixed.points[s * dim..(s + 1) * dim];
            let fv = fixed.values[s];

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
