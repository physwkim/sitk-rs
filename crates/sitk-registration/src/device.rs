//! [`DeviceMeanSquaresMetric`] — mean squares over two images that are **already
//! on the device**.
//!
//! # The transfer this deletes
//!
//! [`crate::CudaMetricBackend`] takes host images and uploads them. When those
//! images came out of a device filter chain, that upload is a re-upload of voxels
//! that were on the device seconds earlier — 113.7 ms at 256³, the single largest
//! item in the measured chain, larger than both filter D2Hs combined. This type
//! consumes the [`DeviceImage`]s the filters produced, so the registration half of
//! the pipeline moves **nothing** across the bus: the volumes are already there,
//! and each iteration exchanges 96 bytes per point-map stage up and 57 KiB of partials
//! down.
//!
//! # No per-call fallback, on purpose
//!
//! [`CudaMetricBackend`](crate::CudaMetricBackend) is infallible by signature and
//! silently runs the CPU whenever the GPU cannot take a call. That is right for a
//! drop-in backend and wrong here: a metric built from device images has no host
//! buffers to fall back *to*, and a hidden per-call dispatch is exactly what made
//! the bus cost invisible in the first place. So every method here returns
//! [`Result`] and **names** its refusal ([`DeviceMetricError`]). The caller decides
//! once, at [`DeviceImage::upload`], whether this pipeline runs at all; if it does
//! not, the caller runs the host chain.

use std::sync::Mutex;

use sitk_cuda::{
    CudaError, DeviceImage, FixedPoints, MovingGeometry, ResidentCorrelation, ResidentMetric,
};
use sitk_transform::interpolator::{index_to_physical_matrix, physical_to_index_matrix, strides};
use sitk_transform::{Interpolator, ParametricTransform};
use thiserror::Error;

use crate::cuda::{affine_form, contract, contract_correlation};
use crate::metric::MetricValue;
use crate::scales::{ScalesEstimator, ScalesEstimatorKind, VirtualGrid};

/// Why the device metric refused a call. Never a silent CPU fallback — see the
/// [module docs](self).
#[derive(Debug, Error)]
pub enum DeviceMetricError {
    /// The kernels are written for `dim = 3`.
    #[error("the device metric is 3-D only; got {0}-D")]
    NotThreeDimensional(usize),

    /// The correlation metric is **global-transform-only**, on the device exactly as on
    /// the host: `CorrelationMetric::check_transform` refuses a local-support transform
    /// by name (mirroring ITK's constructor, which throws). The device names it the same
    /// way rather than letting it fall through to the affine probe and be reported as a
    /// merely non-affine transform — it is not a kernel gap, it is the metric's rule.
    #[error(
        "the correlation metric is global-transform-only; a local-support transform has no derivative it can form"
    )]
    RequiresGlobalTransform,

    /// The fixed and moving images must share a dimension.
    #[error("fixed image is {fixed}-D but moving image is {moving}-D")]
    DimensionMismatch { fixed: usize, moving: usize },

    /// The moving image's direction matrix has no inverse, so a physical point
    /// cannot be mapped to a continuous index.
    #[error("moving image's direction matrix is singular")]
    SingularDirection,

    /// The moment identity the kernel evaluates holds only for a transform whose
    /// **Jacobian** is affine in the point — every globally affine transform
    /// (translation, rigid, Euler, versor, similarity, affine). A B-spline or
    /// displacement field is not, and this metric says so rather than quietly
    /// evaluating it somewhere else.
    #[error(
        "transform's Jacobian is not affine in the point; the device metric has no kernel for it"
    )]
    NonAffineTransform,

    /// The transform has no point map the device can reproduce **bit for bit**: it
    /// reports no [`point_map_stages`](sitk_transform::TransformBase::point_map_stages).
    ///
    /// The device replays the transform's own stages, so the continuous index it
    /// computes is the host's, bit for bit — which the sampler's *discrete* decisions
    /// (`floor`, `is_inside`, `round`) require. A transform whose `transform_point`
    /// evaluates some other expression — `ScaleTransform` and
    /// `ScaleLogarithmicTransform` (centred: `(p − c)·s + c`), `BSplineTransform`,
    /// `DisplacementFieldTransform`, or a composite containing one — cannot be handed
    /// over as stages, and the alternative (an affine form *probed* out of it, exact
    /// in algebra and wrong in the last bits) is the defect this replaced.
    ///
    /// Refused by name, like [`UnsupportedFixedInitialTransform`]: the CPU evaluates
    /// these correctly, but the *device metric* will not silently substitute a map
    /// that merely approximates the host's.
    ///
    /// [`UnsupportedFixedInitialTransform`]: DeviceRegistrationError::UnsupportedFixedInitialTransform
    #[error(
        "transform has no bitwise point map for the device (scale, B-spline, displacement field, \
         or a composite containing one); it is evaluated on the host"
    )]
    NoBitwisePointMap,

    /// The device declined: no driver, no device, NVRTC failure, out of memory.
    #[error(transparent)]
    Cuda(#[from] CudaError),
}

/// Why the device *registration* path refused a run, decided at the boundary before
/// the first iteration — never mid-run, and never by silently running something
/// else. A caller that gets one of these runs
/// [`ImageRegistrationMethod::execute`](crate::ImageRegistrationMethod::execute) on
/// the host images instead.
#[derive(Debug, Error)]
pub enum DeviceRegistrationError {
    /// Mean squares and correlation are the metrics with a device kernel.
    #[error("the device path has kernels only for the mean-squares and correlation metrics")]
    UnsupportedMetric,

    /// The device metric interpolates the moving image linearly.
    #[error("the device metric interpolates linearly; interpolator {0:?} is host-only")]
    UnsupportedInterpolator(Interpolator),

    /// Building a resolution level on the device failed: no device, an NVRTC
    /// failure, out of memory, or a geometry the pyramid ops have no kernel for (a
    /// non-3-D image, a shrink factor of zero, an axis too short for the recursive
    /// Gaussian's fourth-order recursion — the last of which the CPU filter refuses
    /// as well).
    #[error("building a resolution level on the device failed: {0}")]
    Pyramid(#[source] CudaError),

    /// A **fixed-initial transform** whose point map the device cannot reproduce
    /// **bit for bit**.
    ///
    /// The transform relocates the fixed image's sample points, so the level's fixed
    /// image *and* its in-buffer predicate are resampled *through* it. The device does
    /// that ([`sitk_cuda::resample_linear_through`] /
    /// [`sitk_cuda::resample_nearest_through`]) by replaying the transform's **own
    /// stages** ([`point_map_stages`](sitk_transform::TransformBase::point_map_stages)),
    /// in the transform's own order — so a composite is accepted and is *not* folded
    /// into one matrix, because folding rounds once where the transform rounds per stage.
    ///
    /// What is left to refuse: a transform that reports no stages at all, because its
    /// `transform_point` evaluates some *other* expression. `ScaleTransform` and
    /// `ScaleLogarithmicTransform` evaluate the centred `(p − c)·s + c`, which is
    /// `M·p + b` in exact arithmetic and **not** in the last bits; `BSplineTransform` and
    /// `DisplacementFieldTransform` are not linear at all; a `CompositeTransform`
    /// containing any of them reports no stages either. Each is named here rather than
    /// approximated by a probed matrix that would be *almost* right.
    ///
    /// Why "almost right" is not good enough here — and this is the sharpest form of the
    /// argument anywhere in the device path: the predicate is a 0/1 field whose value at
    /// the buffer border is decided by comparing a continuous index against
    /// `[-0.5, size - 0.5)`, and the mask resample rounds that index with
    /// `floor(c + 0.5)`. One ulp does not perturb an intensity by one ulp; it picks a
    /// **different voxel**, flips a shell of border voxels, and moves the valid-point
    /// count the device path pins as *exactly* equal to the host's. An approximate map is
    /// not a slightly worse map — it is a different sample set.
    #[error(
        "a fixed-initial {0:?} transform reports no bitwise point-map stages ({0:?} does \
         not evaluate `mat_vec(matrix, p) + offset` on its own stored fields); the \
         in-buffer predicate is 0/1 and the mask resample rounds with floor(c + 0.5), so \
         one ulp there is a whole voxel — refused rather than approximated"
    )]
    UnsupportedFixedInitialTransform(sitk_transform::TransformKind),

    /// The metric itself refused — a non-affine transform, a non-3-D problem, no
    /// device, or a CUDA failure.
    #[error(transparent)]
    Metric(#[from] DeviceMetricError),

    /// The optimizer driver's own validation (scales length, optimizer weights,
    /// transform dimension) — the same errors [`ImageRegistrationMethod::execute`]
    /// raises, since it is the same driver.
    #[error(transparent)]
    Registration(#[from] crate::RegistrationError),
}

/// Mean-squares metric over two [`DeviceImage`]s, evaluable against any number of
/// transforms without moving a voxel.
///
/// Samples the **whole fixed grid** in traversal order by default: the fixed points are
/// then a closed-form function of the sample index, so nothing but the geometry (24
/// numbers) is uploaded to describe them.
///
/// A **sampling strategy** ([`from_device_sampled`](Self::from_device_sampled)) replaces
/// that with the list of fixed-grid voxels the host drew — 8 bytes a sample, and the
/// points and values are still derived from the grid, at those voxels. The device never
/// draws the samples; it is told which ones.
///
/// The moving image is sampled with **linear** interpolation, which is what the kernel
/// implements.
pub struct DeviceMeanSquaresMetric {
    /// `Mutex` rather than `&mut self` on every call: the optimizer driver holds
    /// the metric by shared reference (it is the same driver the host path uses),
    /// and the resident buffers it evaluates against are `&mut` on the device side.
    /// Uncontended — one optimizer, one thread — so this is a lock, not a queue.
    resident: Mutex<ResidentMetric>,
    /// The fixed image's grid, for the scale/learning-rate estimators. Derived from
    /// the device image's geometry: the estimators walk the *grid*, not the voxels.
    grid: VirtualGrid,
    /// The moving image's physical-to-index matrix, which
    /// [`ScalesEstimatorKind::IndexShift`] measures its shifts in.
    moving_phys_to_index: Vec<f64>,
    min_spacing: f64,
}

impl DeviceMeanSquaresMetric {
    /// Build the metric from two device-resident images. Nothing crosses the bus:
    /// the `f64` copies the kernel reduces in are made device-to-device.
    pub fn from_device(
        fixed: &DeviceImage,
        moving: &DeviceImage,
    ) -> Result<Self, DeviceMetricError> {
        Self::from_device_masked(fixed, moving, None, None)
    }

    /// [`from_device`](Self::from_device) with a **fixed mask** on the fixed image's
    /// grid: a sample whose mask voxel is zero is not a sample, exactly as on the
    /// host, where a zero voxel of the fixed mask drops that sample from
    /// `FixedSamples`.
    ///
    /// The mask does not change how the reduction is performed. A masked-out sample
    /// is skipped in the kernel's grid-stride loop, and the reduction tree is a
    /// function of the block/grid shape and the *voxel count* — never of how many
    /// samples survived. Skipping a term removes it; it does not reorder the terms
    /// that remain, and `is_inside` has always skipped samples this way. So the
    /// valid-point count stays exactly equal to the host's, and value and derivative
    /// stay within the reduction-rounding band the unmasked path already lives in.
    ///
    /// The mask must be on the fixed image's grid (same voxel count), or
    /// [`DeviceMetricError::Cuda`] carrying `CudaError::DegenerateInput`.
    pub fn from_device_masked(
        fixed: &DeviceImage,
        moving: &DeviceImage,
        fixed_mask: Option<&sitk_cuda::DeviceMask>,
        moving_mask: Option<&[bool]>,
    ) -> Result<Self, DeviceMetricError> {
        Self::from_device_sampled(fixed, moving, fixed_mask, moving_mask, None)
    }

    /// [`from_device_masked`](Self::from_device_masked) over a **sampled** subset of the
    /// fixed grid: `samples[s]` is the flat fixed-grid voxel of sample `s`.
    ///
    /// The device does **not** draw the samples. The list is
    /// [`crate::metric::draw_samples`]'s — the same function, called with the same
    /// percentage and the same seed, that the host path draws with — so the two paths do
    /// not *agree* on a sample set, they *share* one. What the device is told is which
    /// voxels; it derives each one's physical point from the same closed form the
    /// full-grid path uses and reads its value from the resident image, so a sampled run
    /// is the full run restricted to those voxels, bit for bit
    /// (`sitk_cuda`'s `the_identity_index_list_is_the_grid_bit_for_bit`).
    ///
    /// The list is the draw **before** the fixed mask has filtered it: the mask is gated
    /// in the kernel, by grid voxel, which an index list still knows. The host filters the
    /// same draw by the same mask, so the two evaluate the same samples — a masked-out
    /// draw contributes nothing on either side.
    ///
    /// `None` is the whole grid in grid order, which is [`from_device_masked`].
    ///
    /// [`from_device_masked`]: Self::from_device_masked
    pub fn from_device_sampled(
        fixed: &DeviceImage,
        moving: &DeviceImage,
        fixed_mask: Option<&sitk_cuda::DeviceMask>,
        moving_mask: Option<&[bool]>,
        samples: Option<&[usize]>,
    ) -> Result<Self, DeviceMetricError> {
        let (resident, layout) =
            with_device_layout(fixed, moving, moving_mask, samples, |points, geom| {
                ResidentMetric::from_device_masked(fixed, points, fixed_mask, moving, geom)
            })?;
        Ok(Self {
            resident: Mutex::new(resident),
            grid: layout.grid,
            moving_phys_to_index: layout.moving_phys_to_index,
            min_spacing: layout.min_spacing,
        })
    }

    /// Device bytes held by the fixed and moving volumes.
    pub fn volume_bytes(&self) -> usize {
        self.resident
            .lock()
            .expect("resident metric poisoned")
            .volume_bytes()
    }

    /// Number of fixed samples the kernel walks — every voxel of the fixed grid,
    /// masked or not. A mask drops samples *inside* the walk (they never become
    /// valid points); it does not shrink the grid, so this count is unchanged by one.
    pub fn sample_count(&self) -> usize {
        self.lock().sample_count()
    }

    /// The resident metric. The lock is uncontended (one optimizer drives it) and
    /// a poisoned lock means a previous evaluation panicked, which this type has no
    /// way to recover from.
    fn lock(&self) -> std::sync::MutexGuard<'_, ResidentMetric> {
        self.resident
            .lock()
            .expect("the device metric's resident buffers are poisoned by an earlier panic")
    }

    /// Scale/learning-rate estimator of `kind` over the fixed image's grid — the
    /// same estimator the host metric builds, from the same geometry. The estimators
    /// walk the grid and the transform's Jacobian; they never read a voxel, so
    /// nothing here comes off the device.
    pub(crate) fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        ScalesEstimator::new(
            &self.grid,
            transform,
            &self.moving_phys_to_index,
            self.min_spacing,
            kind,
        )
    }

    /// The metric value and its derivative with respect to the transform's
    /// parameters, at `transform`.
    ///
    /// Fails with [`DeviceMetricError::NonAffineTransform`] for a transform the moment
    /// identity does not cover, and with [`DeviceMetricError::NoBitwisePointMap`] for one
    /// whose point map the device cannot reproduce bit for bit (a scale transform has an
    /// affine Jacobian and no bitwise point map, so the two refusals are not the same
    /// refusal and are not reported as one). Deterministic: the reduction order is fixed,
    /// so the same inputs give bit-identical results on every call and every run.
    pub fn evaluate(
        &self,
        transform: &dyn ParametricTransform,
    ) -> Result<MetricValue, DeviceMetricError> {
        if transform.dimension() != sitk_cuda::DIM {
            return Err(DeviceMetricError::NotThreeDimensional(
                transform.dimension(),
            ));
        }
        let form = affine_form(transform)?;
        let moments = self.lock().evaluate(&form.stages)?;
        Ok(contract(&moments, &form))
    }

    /// The metric value alone.
    ///
    /// The kernel computes the interpolant's gradient whether or not the caller
    /// wants the derivative, so this reuses the same reduction — the moment pass is
    /// `O(nsamples)` regardless of the parameter count, so there is nothing
    /// per-parameter to skip.
    pub fn value(&self, transform: &dyn ParametricTransform) -> Result<f64, DeviceMetricError> {
        Ok(self.evaluate(transform)?.value)
    }
}

/// Everything a device metric needs that is *not* its kernel: the fixed grid's
/// index-to-physical map, the moving grid's inverse and strides, and the sample set.
///
/// The two metrics differ in what they reduce, not in where the samples are or how
/// the geometry is derived — so this is computed **once**, here, and both call it.
/// A second copy would be a second chance for the two device metrics to disagree
/// with each other about the sample set, which is precisely the disagreement the
/// index-list design exists to prevent.
struct Layout {
    grid: VirtualGrid,
    moving_phys_to_index: Vec<f64>,
    min_spacing: f64,
}

/// Derive the layout, hand the caller the `FixedPoints` and `MovingGeometry` that
/// borrow from it, and return whatever the caller built alongside the layout.
///
/// The closure exists because `FixedPoints`/`MovingGeometry` borrow the matrices and
/// the index list, which are locals here: the resident metric must be constructed
/// while they are alive.
fn with_device_layout<T>(
    fixed: &DeviceImage,
    moving: &DeviceImage,
    moving_mask: Option<&[bool]>,
    samples: Option<&[usize]>,
    build: impl FnOnce(FixedPoints<'_>, &MovingGeometry<'_>) -> Result<T, CudaError>,
) -> Result<(T, Layout), DeviceMetricError> {
    let f = fixed.geometry();
    let m = moving.geometry();
    if f.dimension() != sitk_cuda::DIM {
        return Err(DeviceMetricError::NotThreeDimensional(f.dimension()));
    }
    if m.dimension() != f.dimension() {
        return Err(DeviceMetricError::DimensionMismatch {
            fixed: f.dimension(),
            moving: m.dimension(),
        });
    }

    let dim = f.dimension();
    let idx_to_phys = index_to_physical_matrix(&f.direction, &f.spacing, dim);
    let phys_to_index = physical_to_index_matrix(&m.direction, &m.spacing, dim)
        .ok_or(DeviceMetricError::SingularDirection)?;
    let mstrides = strides(&m.size);

    let geom = MovingGeometry {
        len: moving.len(),
        size: &m.size,
        strides: &mstrides,
        origin: &m.origin,
        phys_to_index: &phys_to_index,
        // The moving mask, on the moving image's own grid. It is *not* resampled
        // and *not* smoothed — the host does not resample it either
        // (`MovingImage::with_moving_mask` takes the mask the user set, and the
        // moving image is never shrunk), and a level's moving volume is the
        // uploaded one smoothed, so the grid is the same at every level and the
        // indices line up.
        mask: moving_mask,
    };
    // The sample set: the whole grid, or the voxels the host's draw named.
    let idx: Option<Vec<i64>> = samples.map(|s| s.iter().map(|&v| v as i64).collect::<Vec<_>>());
    let points = match &idx {
        None => FixedPoints::Grid {
            size: &f.size,
            origin: &f.origin,
            idx_to_phys: &idx_to_phys,
        },
        Some(idx) => FixedPoints::Indices {
            idx,
            size: &f.size,
            origin: &f.origin,
            idx_to_phys: &idx_to_phys,
        },
    };

    let built = build(points, &geom)?;
    Ok((
        built,
        Layout {
            grid: VirtualGrid::new(dim, f.size.clone(), f.origin.clone(), idx_to_phys),
            moving_phys_to_index: phys_to_index,
            min_spacing: f.spacing.iter().copied().fold(f64::INFINITY, f64::min),
        },
    ))
}

/// Normalized cross-correlation over two [`DeviceImage`]s — `value = −sfm²/(sff·smm)`
/// over the mean-subtracted samples.
///
/// The same sample set, masks and sampling as [`DeviceMeanSquaresMetric`] (it is the
/// same [`Layout`] and the same `Resident` underneath), and the same refusals. What
/// differs is the reduction: **two** device passes, because the sample means must be
/// known before any mean-subtracted term can be formed. The one-pass form that would
/// avoid the second launch is refused — see
/// [`sitk_cuda::ResidentCorrelation`] and the `one_pass_moment_form` tests in
/// [`crate::correlation`] — because it trades the host's stable `Σ(f−f̄)²` for
/// `Σf² − N·f̄²`, whose error is a property of the caller's intensity range rather
/// than of the algorithm.
///
/// # Global transforms only, and not by coincidence
///
/// [`CorrelationMetric::check_transform`](crate::CorrelationMetric::check_transform)
/// refuses a local-support transform by name, mirroring ITK's constructor. The moment
/// factorization the device evaluates requires the Jacobian to be affine in the point,
/// which no local-support transform is. The metric's own precondition and the kernel's
/// requirement are the same set — so this metric declines nothing the host would have
/// accepted.
pub struct DeviceCorrelationMetric {
    /// See [`DeviceMeanSquaresMetric::resident`] — same reason, same shape.
    resident: Mutex<ResidentCorrelation>,
    /// The fixed image's grid and the moving image's inverse map, for the
    /// scale/learning-rate estimators the driver builds.
    layout: Layout,
}

impl DeviceCorrelationMetric {
    /// Build from two device-resident images, over the whole fixed grid.
    pub fn from_device(
        fixed: &DeviceImage,
        moving: &DeviceImage,
    ) -> Result<Self, DeviceMetricError> {
        Self::from_device_sampled(fixed, moving, None, None, None)
    }

    /// [`from_device`](Self::from_device) with masks and an optional **sampled** subset
    /// of the fixed grid — the same contract as
    /// [`DeviceMeanSquaresMetric::from_device_sampled`], because it is the same sample
    /// set: the device does not draw, it is told which voxels.
    pub fn from_device_sampled(
        fixed: &DeviceImage,
        moving: &DeviceImage,
        fixed_mask: Option<&sitk_cuda::DeviceMask>,
        moving_mask: Option<&[bool]>,
        samples: Option<&[usize]>,
    ) -> Result<Self, DeviceMetricError> {
        let (resident, layout) =
            with_device_layout(fixed, moving, moving_mask, samples, |points, geom| {
                ResidentCorrelation::from_device_masked(fixed, points, fixed_mask, moving, geom)
            })?;
        Ok(Self {
            resident: Mutex::new(resident),
            layout,
        })
    }

    /// Device bytes held by the fixed and moving volumes.
    pub fn volume_bytes(&self) -> usize {
        self.lock().volume_bytes()
    }

    /// Number of fixed samples the kernel walks.
    pub fn sample_count(&self) -> usize {
        self.lock().sample_count()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ResidentCorrelation> {
        self.resident
            .lock()
            .expect("the device metric's resident buffers are poisoned by an earlier panic")
    }

    /// Scale/learning-rate estimator of `kind` over the fixed image's grid — the same
    /// estimator the host metric builds, from the same geometry.
    pub fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        ScalesEstimator::new(
            &self.layout.grid,
            transform,
            &self.layout.moving_phys_to_index,
            self.layout.min_spacing,
            kind,
        )
    }

    /// The metric value and its parameter-derivative at `transform`.
    ///
    /// Deterministic: the reduction order is fixed, so the same inputs give
    /// bit-identical results on every call and every run.
    pub fn evaluate(
        &self,
        transform: &dyn ParametricTransform,
    ) -> Result<MetricValue, DeviceMetricError> {
        if transform.dimension() != sitk_cuda::DIM {
            return Err(DeviceMetricError::NotThreeDimensional(
                transform.dimension(),
            ));
        }
        // The host's precondition, on the host's terms. A displacement field would
        // fail the affine probe below anyway, but it would fail it as
        // `NonAffineTransform` — and this metric refuses local support *as a metric*,
        // whichever backend it runs on. One rule, named the same way on both paths.
        if transform.has_local_support() {
            return Err(DeviceMetricError::RequiresGlobalTransform);
        }
        let form = affine_form(transform)?;
        let moments = self.lock().evaluate(&form.stages)?;
        Ok(contract_correlation(&moments, &form))
    }

    /// The metric value alone. Both passes still run — the value needs the means, and
    /// the means need a pass.
    pub fn value(&self, transform: &dyn ParametricTransform) -> Result<f64, DeviceMetricError> {
        Ok(self.evaluate(transform)?.value)
    }
}

/// The metrics that have a device kernel — one variant per kernel, matched
/// exhaustively everywhere, so adding a third is a compile error at every site that
/// has to know about it rather than a silent fallthrough to one of the first two.
pub(crate) enum DeviceMetric {
    MeanSquares(DeviceMeanSquaresMetric),
    Correlation(DeviceCorrelationMetric),
}

impl DeviceMetric {
    fn evaluate(
        &self,
        transform: &dyn ParametricTransform,
    ) -> Result<MetricValue, DeviceMetricError> {
        match self {
            Self::MeanSquares(m) => m.evaluate(transform),
            Self::Correlation(m) => m.evaluate(transform),
        }
    }

    fn value(&self, transform: &dyn ParametricTransform) -> Result<f64, DeviceMetricError> {
        match self {
            Self::MeanSquares(m) => m.value(transform),
            Self::Correlation(m) => m.value(transform),
        }
    }

    fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        match self {
            Self::MeanSquares(m) => m.scales_estimator(transform, kind),
            Self::Correlation(m) => m.scales_estimator(transform, kind),
        }
    }
}

/// A [`DeviceMetric`] wrapped for the optimizer driver, which cannot fail by
/// signature ([`crate::optimizer::Objective`] returns values, not `Result`s).
///
/// The refusals that *matter* — a metric the device has no kernel for, a sampling
/// strategy, a pyramid, a non-affine transform — are all decided at the pipeline
/// boundary in
/// [`ImageRegistrationMethod::execute_on_device`](crate::ImageRegistrationMethod::execute_on_device),
/// before the first iteration runs. What is left is a *device* failure mid-run
/// (the driver falling over, an allocation failing): rare, and impossible to
/// answer honestly from inside an infallible callback. So the first such error is
/// recorded here, the iteration returns the same "no valid samples" value the CPU
/// returns (`f64::MAX`, zero derivative), and `execute_on_device` **discards the
/// run and returns the error**. The caller never receives a result computed after
/// a device failure.
pub(crate) struct DeviceActive {
    metric: DeviceMetric,
    failure: Mutex<Option<DeviceMetricError>>,
}

impl DeviceActive {
    pub(crate) fn new(metric: DeviceMetric) -> Self {
        Self {
            metric,
            failure: Mutex::new(None),
        }
    }

    pub(crate) fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        match self.metric.evaluate(transform) {
            Ok(v) => v,
            Err(e) => {
                self.record(e);
                MetricValue {
                    value: f64::MAX,
                    derivative: vec![0.0; transform.number_of_parameters()],
                    valid_points: 0,
                }
            }
        }
    }

    pub(crate) fn value(&self, transform: &dyn ParametricTransform) -> f64 {
        match self.metric.value(transform) {
            Ok(v) => v,
            Err(e) => {
                self.record(e);
                f64::MAX
            }
        }
    }

    pub(crate) fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        self.metric.scales_estimator(transform, kind)
    }

    /// The boundary probe: evaluate once at the composed initial transform and
    /// hand the caller the real error if the device cannot take it. Nothing is
    /// recorded — a refusal here means the run never starts.
    pub(crate) fn metric_evaluate_probe(
        &self,
        transform: &dyn ParametricTransform,
    ) -> Result<MetricValue, DeviceMetricError> {
        self.metric.evaluate(transform)
    }

    /// The first device failure of the run, if any. Checked by
    /// `execute_on_device` after the optimizer stops.
    pub(crate) fn take_failure(&self) -> Option<DeviceMetricError> {
        self.failure.lock().ok().and_then(|mut f| f.take())
    }

    fn record(&self, e: DeviceMetricError) {
        if let Ok(mut slot) = self.failure.lock() {
            slot.get_or_insert(e);
        }
    }
}
