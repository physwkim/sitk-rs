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
//! and each iteration exchanges 96 bytes up and 57 KiB of partials down.
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

use sitk_cuda::{CudaError, DeviceImage, FixedPoints, MovingGeometry, ResidentMetric};
use sitk_transform::interpolator::{index_to_physical_matrix, physical_to_index_matrix, strides};
use sitk_transform::{Interpolator, ParametricTransform};
use thiserror::Error;

use crate::cuda::{affine_form, contract};
use crate::metric::MetricValue;
use crate::scales::{ScalesEstimator, ScalesEstimatorKind, VirtualGrid};

/// Why the device metric refused a call. Never a silent CPU fallback — see the
/// [module docs](self).
#[derive(Debug, Error)]
pub enum DeviceMetricError {
    /// The kernel is written for `dim = 3`.
    #[error("the device mean-squares metric is 3-D only; got {0}-D")]
    NotThreeDimensional(usize),

    /// The fixed and moving images must share a dimension.
    #[error("fixed image is {fixed}-D but moving image is {moving}-D")]
    DimensionMismatch { fixed: usize, moving: usize },

    /// The moving image's direction matrix has no inverse, so a physical point
    /// cannot be mapped to a continuous index.
    #[error("moving image's direction matrix is singular")]
    SingularDirection,

    /// The moment identity the kernel evaluates holds only for a transform whose
    /// point map *and* Jacobian are affine in the point — every globally affine
    /// transform (translation, rigid, Euler, versor, similarity, affine). A
    /// B-spline or displacement field is not, and this metric says so rather than
    /// quietly evaluating it somewhere else.
    #[error("transform is not affine in the point; the device metric has no kernel for it")]
    NonAffineTransform,

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
    /// Mean squares is the only metric with a device kernel.
    #[error("the device path has a kernel only for the mean-squares metric")]
    UnsupportedMetric,

    /// The device metric interpolates the moving image linearly.
    #[error("the device metric interpolates linearly; interpolator {0:?} is host-only")]
    UnsupportedInterpolator(Interpolator),

    /// The device metric samples every voxel of the fixed grid.
    #[error("the device metric samples every voxel; sampling strategies are host-only")]
    UnsupportedSampling,

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
    /// that now ([`sitk_cuda::resample_linear_through`] /
    /// [`sitk_cuda::resample_nearest_through`]) — but only for a transform whose point
    /// map is *literally* `mat_vec(matrix, p) + offset` on its own stored fields, which
    /// is what [`Transform::matrix_offset_map`](sitk_transform::Transform::matrix_offset_map)
    /// hands over and what its contract guarantees.
    ///
    /// What is refused, and why it is refused rather than approximated: the predicate is
    /// a 0/1 field whose value at the buffer border is decided by comparing a continuous
    /// index against `[-0.5, size - 0.5)`. One ulp in the mapped point flips a shell of
    /// voxels there, moves the valid-point count, and breaks the one property the device
    /// path pins as *exactly* equal to the host's. So an approximate map is not a
    /// slightly worse map — it is a different sample set. `ScaleTransform` and
    /// `ScaleLogarithmicTransform` evaluate `(p − c)·s + c`, which is `M·p + b` in exact
    /// arithmetic and **not** in the last bits; `CompositeTransform` rounds once per
    /// stage, so a composed matrix is not its arithmetic either; `BSplineTransform` and
    /// `DisplacementFieldTransform` are not linear at all. Each is named here rather
    /// than folded into a matrix that would be *almost* right.
    #[error(
        "a fixed-initial {0:?} transform has no point map the device can reproduce bit \
         for bit ({0:?} is not `mat_vec(matrix, p) + offset` on its own stored fields); \
         the in-buffer predicate is 0/1 and one ulp at the border moves the valid-point \
         count, so this is refused rather than approximated"
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
/// Samples the **whole fixed grid** in traversal order — a device image carries no
/// sampling strategy and no mask, and the fixed points are a closed-form function
/// of the sample index, so nothing but the geometry (24 numbers) is uploaded to
/// describe them. The moving image is sampled with **linear** interpolation, which
/// is what the kernel implements.
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
        let points = FixedPoints::Grid {
            size: &f.size,
            origin: &f.origin,
            idx_to_phys: &idx_to_phys,
        };

        let resident =
            ResidentMetric::from_device_masked(fixed, points, fixed_mask, moving, &geom)?;
        Ok(Self {
            resident: Mutex::new(resident),
            grid: VirtualGrid::new(dim, f.size.clone(), f.origin.clone(), idx_to_phys),
            moving_phys_to_index: phys_to_index,
            min_spacing: f.spacing.iter().copied().fold(f64::INFINITY, f64::min),
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
    /// Fails with [`DeviceMetricError::NonAffineTransform`] for a transform the
    /// moment identity does not cover. Deterministic: the reduction order is fixed,
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
        let form = affine_form(transform).ok_or(DeviceMetricError::NonAffineTransform)?;
        let moments = self.lock().evaluate(&form.a, &form.b)?;
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

/// A [`DeviceMeanSquaresMetric`] wrapped for the optimizer driver, which cannot
/// fail by signature ([`crate::optimizer::Objective`] returns values, not
/// `Result`s).
///
/// The refusals that *matter* — a metric the device has no kernel for, a mask, a
/// sampling strategy, a pyramid, a non-affine transform — are all decided at the
/// pipeline boundary in
/// [`ImageRegistrationMethod::execute_on_device`](crate::ImageRegistrationMethod::execute_on_device),
/// before the first iteration runs. What is left is a *device* failure mid-run
/// (the driver falling over, an allocation failing): rare, and impossible to
/// answer honestly from inside an infallible callback. So the first such error is
/// recorded here, the iteration returns the same "no valid samples" value the CPU
/// returns (`f64::MAX`, zero derivative), and `execute_on_device` **discards the
/// run and returns the error**. The caller never receives a result computed after
/// a device failure.
pub(crate) struct DeviceActive {
    metric: DeviceMeanSquaresMetric,
    failure: Mutex<Option<DeviceMetricError>>,
}

impl DeviceActive {
    pub(crate) fn new(metric: DeviceMeanSquaresMetric) -> Self {
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
