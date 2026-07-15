//! CUDA [`MetricBackend`]: the mean-squares metric with both volumes resident on
//! the device across the whole optimizer run.
//!
//! Enabled by the `cuda` feature (default off) and selected explicitly:
//!
//! ```ignore
//! reg.set_metric_backend(Box::new(CudaMetricBackend::new()));
//! ```
//!
//! # It cannot fail
//!
//! [`CudaMetricBackend`] owns a [`CpuBackend`] and falls back to it on *every*
//! condition that is not a GPU success: no driver, no device, NVRTC failure, out
//! of memory, a 2-D problem, a non-linear interpolator, a transform whose point
//! map or Jacobian is not affine in the point. `MetricBackend`'s methods are
//! infallible by signature, and this backend keeps them that way — installing it
//! can slow a registration down, but it cannot turn a working one into a failing
//! or a wrong one.
//!
//! # Which transforms the GPU takes
//!
//! The device is told two things, and they are recovered in **two different ways**,
//! because they are held to two different standards:
//!
//! - **The point map**, as the transform's own stages
//!   ([`TransformBase::point_map_stages`]): the stored `matrix`/`offset` pairs its
//!   `transform_point` evaluates, one per map it applies, in the order it applies
//!   them. Not probed, and not folded. The kernel replays that arithmetic, so the
//!   continuous index it computes is **bit-for-bit** the host's — and it must be,
//!   because `floor(c)`, `is_inside(c)` and `round(c)` are *branches*: a 1-ulp
//!   difference there is not a 1-ulp difference in the answer, it is the other side
//!   of a discontinuity in the interpolant's gradient. A transform that reports no
//!   stages is refused by name and runs on the CPU. That set is exactly the
//!   transforms whose `transform_point` evaluates some other expression:
//!   `ScaleTransform` and `ScaleLogarithmicTransform` (centred, `(p − c)·s + c`),
//!   `BSplineTransform`, `DisplacementFieldTransform`, and any composite containing
//!   one of them.
//!
//! - **The Jacobian's affine decomposition**, still *probed* — `J(0)` and
//!   `C_e = J(e_e) − J(0)` — and verified against `J` at two probe points to a
//!   tolerance. This is the right standard for it: `J` feeds a sum of millions of
//!   products, a continuous function of its inputs with no branch anywhere
//!   downstream, so an ulp in `J` is an ulp in the derivative. Nothing discrete
//!   depends on it, and the probe is what rejects a Jacobian that is not affine in
//!   the point at all.
//!
//! The probe *used* to supply both, and the point map it recovered
//! (`A[d][e] = T(e_e)[d] − b[d]`) was algebraically exact and **bitwise wrong**: the
//! subtraction cancels the offset back off, at a cost of an ulp or two. That is the
//! straddle the stage list closes.
//!
//! [`TransformBase::point_map_stages`]: crate::transform::TransformBase::point_map_stages

use std::sync::Mutex;

use crate::core::parallel;
use crate::cuda::{FixedPoints, MAX_STAGES, MovingGeometry, PointStage, ResidentMetric};
use crate::transform::matrix_offset::replay_stages;
use crate::transform::{Interpolator, ParametricTransform, TransformBase};

use crate::registration::device::DeviceMetricError;
use crate::registration::metric::{
    CpuBackend, FixedSamples, MetricBackend, MetricValue, MovingImage, SamplePoints,
};

/// Relative tolerance for the probe that decides whether a transform's **Jacobian**
/// really is affine in the point.
///
/// Recovering `C_e` by differencing `J` at basis points costs a few ulps, so a
/// transform whose Jacobian is exactly affine in the point reproduces `J(q)` to
/// ~1e-16 relative, and one whose Jacobian is not (B-spline, displacement field)
/// misses by O(1). Any threshold between those separates them; 1e-9 is far above the
/// rounding floor and far below any real nonlinearity, so it is not a tuned constant.
///
/// It applies to the Jacobian **only**. The point map is not probed and is not
/// compared to a tolerance: it is taken from the transform's stages and checked on
/// the bits ([`point_stages`]), because the decisions downstream of it are branches.
const AFFINE_TOL: f64 = 1e-9;

/// Off-axis, irrational-ish, and not near any voxel centre: a probe point chosen
/// so that a transform which is affine only on the lattice, or only near the
/// origin, cannot pass by luck.
const PROBE: [f64; 3] = [1.7, -3.1, 2.3];

/// A second probe, far from the first in magnitude, so that a map which is affine
/// locally but not globally is caught too.
const PROBE_FAR: [f64; 3] = [-137.0, 91.5, 204.25];

/// The device-resident volumes, plus the identities of the buffers they came from.
struct Resident {
    fixed_id: u64,
    moving_id: u64,
    metric: ResidentMetric,
}

/// What the device needs from a transform: its point map as **stages**, and its
/// Jacobian's affine decomposition.
///
/// The two halves are recovered differently and held to different standards — see
/// the [module docs](self). The stages are the transform's own arithmetic, taken
/// bitwise; the Jacobian is probed to a tolerance.
pub(crate) struct AffineForm {
    /// The transform's point map: `1..=MAX_STAGES` stages, applied in this order.
    pub(crate) stages: Vec<PointStage>,
    /// `J(0)`, row-major `dim × nparams`.
    pub(crate) j0: Vec<f64>,
    /// `C_e = J(e_e) − J(0)` for each basis direction `e`, same layout as `j0`.
    pub(crate) c: [Vec<f64>; 3],
    pub(crate) nparams: usize,
}

/// The CUDA mean-squares backend. See the [module docs](self).
///
/// Holds its device buffers across calls: the fixed and moving volumes are
/// uploaded once per pyramid level and reused for every optimizer iteration, and
/// the per-iteration partials buffer and its host destination are allocated once
/// and reused too. Nothing in the iteration loop allocates.
pub struct CudaMetricBackend {
    cpu: CpuBackend,
    resident: Mutex<Option<Resident>>,
}

impl CudaMetricBackend {
    /// Build the backend. Never fails: a machine with no GPU gets a backend that
    /// silently runs every evaluation on the CPU.
    pub fn new() -> Self {
        Self {
            cpu: CpuBackend,
            resident: Mutex::new(None),
        }
    }

    /// The moments for this transform, or `None` if the GPU cannot take it — in
    /// which case the caller runs the CPU path.
    fn gpu_moments(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> Option<(crate::cuda::Moments, AffineForm)> {
        if fixed.dim != crate::cuda::DIM || transform.dimension() != crate::cuda::DIM {
            return None;
        }
        // Cheap structural guard before the probes: a local-support transform has
        // one parameter block per voxel, so probing its dense Jacobian would cost
        // more than the evaluation it is trying to accelerate. The affine probe
        // below would reject it anyway — this just declines without paying.
        if transform.has_local_support() {
            return None;
        }
        let view = moving.device_view();
        if view.interpolator != Interpolator::Linear {
            return None;
        }

        let form = affine_form(transform).ok()?;

        let mut guard = self.resident.lock().ok()?;
        let stale = match guard.as_ref() {
            Some(r) => r.fixed_id != fixed.id || r.moving_id != view.id,
            None => true,
        };
        if stale {
            // A new pyramid level: drop the old volumes and upload the new ones.
            // This is the only large transfer in the run.
            *guard = None;
            let geom = MovingGeometry {
                len: view.buf.len(),
                size: view.size,
                strides: view.strides,
                origin: view.origin,
                phys_to_index: view.phys_to_index,
                mask: view.mask,
            };
            // When the sample set is the whole fixed grid in traversal order, the
            // points are a pure function of the sample index: send the grid (24
            // numbers) instead of the points (402 MB at 256³). The host does not
            // hold them either — `SamplePoints::Grid` derives them — so this is
            // the same closed form on both sides of the bus. A sampled or masked
            // set has no such form, so it uploads the points it materialized.
            let points = match &fixed.points {
                SamplePoints::Grid => {
                    let (size, origin, idx_to_phys) = fixed.grid.parts();
                    FixedPoints::Grid {
                        size,
                        origin,
                        idx_to_phys,
                    }
                }
                // This backend uploads the *values it gathered per sample*, not the fixed
                // grid, so its samples have no grid to index into and the points are what
                // it can send. (The device-resident path holds the whole grid and does
                // send indices — `FixedPoints::Indices` — which is why a fixed mask
                // composes with a sampled set there and not here.)
                SamplePoints::Explicit { points, .. } => FixedPoints::Explicit(points),
            };
            // Both volumes are held in their image's native type, so each is
            // widened straight into its upload, a chunk at a time — there is no
            // `f64` volume on the host to hand over.
            let metric = ResidentMetric::new(
                fixed.len(),
                |start, out| {
                    parallel::for_each_mut(out, |i, o| *o = fixed.value(start + i));
                },
                points,
                &geom,
                |start, out| {
                    parallel::for_each_mut(out, |i, o| *o = view.buf.get(start + i));
                },
            )
            .ok()?;
            *guard = Some(Resident {
                fixed_id: fixed.id,
                moving_id: view.id,
                metric,
            });
        }

        let r = guard.as_mut()?;
        let moments = r.metric.evaluate(&form.stages).ok()?;
        Some((moments, form))
    }
}

impl Default for CudaMetricBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// The transform's point map, as stages the device can replay — or `None` if it has
/// none, which is the refusal that keeps a transform on the CPU.
///
/// The stages come from the transform itself ([`TransformBase::point_map_stages`]);
/// this only converts them to the device's fixed-size form and then **checks the
/// bitwise claim** rather than trusting it: replaying the stages on the host must
/// reproduce `transform_point` with `to_bits()` equality at both probe points. That
/// check is the whole contract in one line — if it holds on the host and the kernel
/// performs the same operations in the same order, the continuous index is bit-identical
/// on both sides.
///
/// Generic over `TransformBase`, not `ParametricTransform`, because it has **two**
/// callers and only one of them optimizes: the metric hands over the transform being
/// optimized, and [`ImageRegistrationMethod`](crate::registration::ImageRegistrationMethod)'s device
/// pyramid hands over the *fixed-initial* transform, which has no parameters to
/// differentiate and only ever needs to map a point. One converter, one bitwise check,
/// both consumers.
pub(crate) fn point_stages<T: TransformBase + ?Sized>(transform: &T) -> Option<Vec<PointStage>> {
    let dim = crate::cuda::DIM;
    let maps = transform.point_map_stages()?;
    if maps.is_empty() || maps.len() > MAX_STAGES {
        return None;
    }

    let mut stages = Vec::with_capacity(maps.len());
    for m in &maps {
        if m.matrix.len() != dim * dim || m.offset.len() != dim {
            return None;
        }

        let mut matrix = [0.0f64; 9];
        let mut offset = [0.0f64; 3];
        matrix.copy_from_slice(&m.matrix);
        offset.copy_from_slice(&m.offset);
        stages.push(PointStage { matrix, offset });
    }

    for probe in [PROBE, PROBE_FAR] {
        let truth = transform.transform_point(&probe);
        if truth.len() != dim {
            return None;
        }
        let replayed = replay_stages(&maps, &probe, dim);
        for (got, want) in replayed.iter().zip(truth.iter()) {
            // Bits, not a tolerance. A stage list that is merely *close* to the
            // transform's own arithmetic is the bug this replaced.
            if got.to_bits() != want.to_bits() {
                return None;
            }
        }
    }
    Some(stages)
}

/// Recover the point map's stages and the Jacobian's affine decomposition from
/// `transform`, or `None` if it has no bitwise point map or its Jacobian is not
/// affine in the point.
///
/// The Jacobian half is probed: `J` affine in the point ⟹
/// `J(x) = J(0) + Σ_e x_e·(J(e_e) − J(0))`, checked against the transform at two
/// probe points to [`AFFINE_TOL`]. The point-map half is not probed at all — see
/// [`point_stages`] and the [module docs](self) for why the two are held to
/// different standards.
pub(crate) fn affine_form(
    transform: &dyn ParametricTransform,
) -> Result<AffineForm, DeviceMetricError> {
    let dim = crate::cuda::DIM;
    let nparams = transform.number_of_parameters();
    if nparams == 0 {
        return Err(DeviceMetricError::NonAffineTransform);
    }

    let zero = [0.0f64; 3];
    let j0 = transform.jacobian_wrt_parameters(&zero);
    if j0.len() != dim * nparams {
        return Err(DeviceMetricError::NonAffineTransform);
    }

    let mut c: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for e in 0..dim {
        let mut basis = [0.0f64; 3];
        basis[e] = 1.0;
        let je = transform.jacobian_wrt_parameters(&basis);
        if je.len() != dim * nparams {
            return Err(DeviceMetricError::NonAffineTransform);
        }
        c[e] = je.iter().zip(j0.iter()).map(|(&x, &y)| x - y).collect();
    }

    // Verify, rather than trust. A Jacobian that is not affine in the point misses this
    // by O(1) and the transform declines to the CPU.
    for probe in [PROBE, PROBE_FAR] {
        let jt = transform.jacobian_wrt_parameters(&probe);
        if jt.len() != dim * nparams {
            return Err(DeviceMetricError::NonAffineTransform);
        }
        for (idx, &want) in jt.iter().enumerate() {
            let predicted = j0[idx] + (0..dim).map(|e| probe[e] * c[e][idx]).sum::<f64>();
            if !close(predicted, want) {
                return Err(DeviceMetricError::NonAffineTransform);
            }
        }
    }

    // The point map is asked for LAST, and its refusal is a DIFFERENT refusal with a
    // different name.
    //
    // What the two refusals actually separate -- MEASURED, not assumed
    // (`mattes_device.rs::a_bspline_transform_is_refused_by_name_and_the_name_is_the_point_maps`):
    //
    //   - `NonAffineTransform` fires for a transform whose Jacobian is affine nowhere near
    //     the probes -- a displacement field.
    //   - `NoBitwisePointMap` fires for the rest, and the rest is larger than it looks. A
    //     `ScaleTransform` has an affine Jacobian and no stages, which is the family this
    //     refusal was written for. But a **B-spline** also lands here, and NOT because it
    //     fails the probe: `PROBE` and `PROBE_FAR` lie outside any realistic B-spline
    //     support region, where its Jacobian is identically zero -- and zero is trivially
    //     affine in the point, so the probe above PASSES for it, vacuously. It is the
    //     point map that catches it.
    //
    // The transform is refused either way, by name, and never approximated -- which is the
    // guarantee that matters. But the probe is not the thing that rejects a B-spline, and
    // the comment that used to sit here said it was.
    let stages = point_stages(transform).ok_or(DeviceMetricError::NoBitwisePointMap)?;

    let form = AffineForm {
        stages,
        j0,
        c,
        nparams,
    };
    Ok(form)
}

/// Relative-with-absolute-floor comparison, so a Jacobian entry that is exactly
/// zero (most of them are) does not fail a purely relative test.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= AFFINE_TOL * (1.0 + a.abs().max(b.abs()))
}

/// Contract the moments with the transform's own Jacobian into value + derivative.
///
/// ```text
/// value  = sq / count
/// ∂/∂pₖ  = (2/count) · ( Σ_d J(0)[d][k]·S0[d] + Σ_d Σ_e C_e[d][k]·S1[d][e] )
/// ```
///
/// Exact: this is the same sum the CPU accumulates per-sample, re-associated. The
/// only difference from the CPU's result is the *order* the millions of per-sample
/// terms were added in, which is a rounding difference, not a modelling one.
pub(crate) fn contract(moments: &crate::cuda::Moments, form: &AffineForm) -> MetricValue {
    let dim = crate::cuda::DIM;
    let nparams = form.nparams;

    if moments.count == 0 {
        // Matches CpuBackend exactly: no valid sample is `f64::MAX`, not zero.
        return MetricValue {
            value: f64::MAX,
            derivative: vec![0.0; nparams],
            valid_points: 0,
        };
    }
    let inv = 1.0 / moments.count as f64;

    let mut derivative = vec![0.0; nparams];
    for (k, dk) in derivative.iter_mut().enumerate() {
        let mut g = 0.0;
        for d in 0..dim {
            g += form.j0[d * nparams + k] * moments.s0[d];
            for e in 0..dim {
                g += form.c[e][d * nparams + k] * moments.s1[d][e];
            }
        }
        *dk = 2.0 * g * inv;
    }

    MetricValue {
        value: moments.sq * inv,
        derivative,
        valid_points: moments.count,
    }
}

/// Contract the correlation moments with the transform's own Jacobian into value +
/// derivative — the host half of the NCC metric, and the **only** place its formula
/// exists on this path.
///
/// ```text
/// value      = −sfm² / (sff·smm)
/// fdmₖ       = Σ_d J(0)[d][k]·F0[d] + Σ_d Σ_e C_e[d][k]·F1[d][e]      (mdmₖ likewise)
/// ∂value/∂pₖ = −2·sfm/(sff·smm) · ( fdmₖ − (sfm/smm)·mdmₖ )
/// ```
///
/// The globals `sfm`/`sff`/`smm` never reached the device: they multiply the finished
/// sums and do not depend on the sample, so they factor out of the reduction and are
/// applied here, in `f64`, exactly as `CorrelationMetric::evaluate` applies them.
///
/// # The degenerate branch is the host's, deliberately
///
/// `CorrelationMetric::evaluate` returns `f64::MAX` when no sample is valid, and again
/// when `smm·sff <= f64::EPSILON` (a constant image has no variance to correlate).
/// Both tests are re-stated here **in the same order, on the same product, against the
/// same constant** — not because the device could not test them, but because a
/// threshold on a *reduced* quantity is a discontinuous branch, and two paths whose
/// sums differ by √N·ε can land on opposite sides of it. Keeping the branch in one
/// implementation means the only way the two paths can disagree is if the reduced
/// values straddle the threshold — which is a measurable property of the data, not a
/// second copy of a rule that might drift.
pub(crate) fn contract_correlation(
    moments: &crate::cuda::CorrelationMoments,
    form: &AffineForm,
) -> MetricValue {
    let dim = crate::cuda::DIM;
    let nparams = form.nparams;

    // `means()` returned `None`: no sample maps inside the moving image.
    if moments.count == 0 {
        return MetricValue {
            value: f64::MAX,
            derivative: vec![0.0; nparams],
            valid_points: 0,
        };
    }

    let (sff, smm, sfm) = (moments.sff, moments.smm, moments.sfm);
    // The host's own product and the host's own test, in the host's order.
    let m2f2 = smm * sff;
    if m2f2 <= f64::EPSILON {
        return MetricValue {
            value: f64::MAX,
            derivative: vec![0.0; nparams],
            valid_points: moments.count,
        };
    }

    // The Jacobian contraction, identical in shape to mean squares' — once with the
    // fixed-side moments and once with the moving-side ones.
    let contract_moment = |m0: &[f64; 3], m1: &[[f64; 3]; 3], k: usize| {
        let mut g = 0.0;
        for d in 0..dim {
            g += form.j0[d * nparams + k] * m0[d];
            for (e, &m1de) in m1[d].iter().enumerate().take(dim) {
                g += form.c[e][d * nparams + k] * m1de;
            }
        }
        g
    };

    let mut derivative = vec![0.0; nparams];
    for (k, dk) in derivative.iter_mut().enumerate() {
        let fdm = contract_moment(&moments.f0, &moments.f1, k);
        let mdm = contract_moment(&moments.m0, &moments.m1, k);
        *dk = -2.0 * sfm / (sff * smm) * (fdm - sfm / smm * mdm);
    }

    MetricValue {
        value: -sfm * sfm / m2f2,
        derivative,
        valid_points: moments.count,
    }
}

/// Contract the twelve pRatio-weighted moments with the transform's own Jacobian into
/// the Mattes derivative — the host half of the device Mattes metric, and the **only**
/// place its derivative formula exists on this path.
///
/// ```text
/// ∂value/∂pₖ = Σ_d J(0)[d][k]·A[d] + Σ_d Σ_e C_e[d][k]·B[d][e]
/// ```
///
/// The value is **not** formed here: it came off the device as a joint histogram that is
/// bit-identical to the host's, and was turned into `−MI` by the host metric's own tail
/// (`mattes_tail`). Only the derivative needs the Jacobian, and only the derivative
/// carries the probe's band — see [`crate::registration::device::DeviceMattesMetric`].
pub(crate) fn contract_mattes(
    moments: &crate::cuda::DerivativeMoments,
    form: &AffineForm,
) -> Vec<f64> {
    let dim = crate::cuda::DIM;
    let nparams = form.nparams;

    let mut derivative = vec![0.0; nparams];
    for (k, dk) in derivative.iter_mut().enumerate() {
        let mut g = 0.0;
        for d in 0..dim {
            g += form.j0[d * nparams + k] * moments.a[d];
            for e in 0..dim {
                g += form.c[e][d * nparams + k] * moments.b[d][e];
            }
        }
        *dk = g;
    }
    derivative
}

impl MetricBackend for CudaMetricBackend {
    fn mean_squares(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> MetricValue {
        match self.gpu_moments(fixed, moving, transform) {
            Some((moments, form)) => contract(&moments, &form),
            None => self.cpu.mean_squares(fixed, moving, transform),
        }
    }

    fn mean_squares_value(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> f64 {
        // The kernel computes the interpolant's gradient whether or not the
        // caller wants the derivative, so this reuses it. That is not the waste
        // the trait warns about: what the trait forbids is paying
        // `O(nsamples · nparams)` to accumulate a derivative nobody reads, and the
        // moment reduction is `O(nsamples)` regardless of the parameter count —
        // there is no per-parameter accumulation on the device to skip.
        match self.gpu_moments(fixed, moving, transform) {
            Some((moments, _)) if moments.count > 0 => moments.sq / moments.count as f64,
            Some(_) => f64::MAX,
            None => self.cpu.mean_squares_value(fixed, moving, transform),
        }
    }
}
