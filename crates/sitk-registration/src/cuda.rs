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
//! # Which transforms the GPU takes, and why it is not a type list
//!
//! The device is told the point map as `x ↦ A·x + b` and nothing else. `A` and `b`
//! are recovered from the transform through the ordinary
//! [`ParametricTransform::transform_point`] — `b = T(0)`, `A[:,e] = T(e_e) − b` —
//! and then *verified* against `T` at an off-axis probe point. The same is done
//! for the Jacobian, which the moment identity requires to be affine in the point.
//!
//! So a transform qualifies by satisfying the algebra the kernel assumes, not by
//! appearing on a whitelist. Every globally affine transform passes (translation,
//! rigid, Euler, versor, similarity, affine); a B-spline or displacement field
//! fails the probe and lands on the CPU, which is exactly where it belongs. A new
//! affine-family transform added later works with no change here.

use std::sync::Mutex;

use sitk_cuda::{MovingGeometry, ResidentMetric};
use sitk_transform::{Interpolator, ParametricTransform};

use crate::metric::{CpuBackend, FixedSamples, MetricBackend, MetricValue, MovingImage};

/// Relative tolerance for the probes that decide whether a transform's point map
/// and Jacobian really are affine in the point.
///
/// Recovering `A` by differencing `T` at basis points costs a few ulps, so an
/// exactly-affine transform reproduces `T(q)` to ~1e-16 relative, and a
/// *non*-affine one (B-spline, displacement field) misses by O(1). Any threshold
/// between those separates them; 1e-9 is far above the rounding floor and far
/// below any real nonlinearity, so it is not a tuned constant.
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

/// The affine point map `x ↦ A·x + b` plus the Jacobian's affine decomposition,
/// all recovered from the transform through its public trait.
struct AffineForm {
    /// Row-major 3 × 3.
    a: [f64; 9],
    b: [f64; 3],
    /// `J(0)`, row-major `dim × nparams`.
    j0: Vec<f64>,
    /// `C_e = J(e_e) − J(0)` for each basis direction `e`, same layout as `j0`.
    c: [Vec<f64>; 3],
    nparams: usize,
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
    ) -> Option<(sitk_cuda::Moments, AffineForm)> {
        if fixed.dim != sitk_cuda::DIM || transform.dimension() != sitk_cuda::DIM {
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

        let form = affine_form(transform)?;

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
                buf: view.buf,
                size: view.size,
                strides: view.strides,
                origin: view.origin,
                phys_to_index: view.phys_to_index,
                mask: view.mask,
            };
            let metric = ResidentMetric::new(&fixed.values, &fixed.points, &geom).ok()?;
            *guard = Some(Resident {
                fixed_id: fixed.id,
                moving_id: view.id,
                metric,
            });
        }

        let r = guard.as_mut()?;
        let moments = r.metric.evaluate(&form.a, &form.b).ok()?;
        Some((moments, form))
    }
}

impl Default for CudaMetricBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Recover `x ↦ A·x + b` and the Jacobian's affine decomposition from `transform`,
/// or `None` if either is not affine in the point.
///
/// `T` affine ⟹ `b = T(0)` and `A[:,e] = T(e_e) − T(0)`; likewise each Jacobian
/// column satisfies `J(x) = J(0) + Σ_e x_e·(J(e_e) − J(0))`. Both reconstructions
/// are then checked against the transform itself at two probe points. This is the
/// whole of the GPU's transform support: pass the check, run on the device.
fn affine_form(transform: &dyn ParametricTransform) -> Option<AffineForm> {
    let dim = sitk_cuda::DIM;
    let nparams = transform.number_of_parameters();
    if nparams == 0 {
        return None;
    }

    let zero = [0.0f64; 3];
    let t0 = transform.transform_point(&zero);
    let j0 = transform.jacobian_wrt_parameters(&zero);
    if t0.len() != dim || j0.len() != dim * nparams {
        return None;
    }

    let mut b = [0.0f64; 3];
    b.copy_from_slice(&t0);

    let mut a = [0.0f64; 9];
    let mut c: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for e in 0..dim {
        let mut basis = [0.0f64; 3];
        basis[e] = 1.0;

        let te = transform.transform_point(&basis);
        for d in 0..dim {
            // Column e of A, stored row-major: A[d][e].
            a[d * dim + e] = te[d] - b[d];
        }

        let je = transform.jacobian_wrt_parameters(&basis);
        if je.len() != dim * nparams {
            return None;
        }
        c[e] = je.iter().zip(j0.iter()).map(|(&x, &y)| x - y).collect();
    }

    let form = AffineForm {
        a,
        b,
        j0,
        c,
        nparams,
    };

    // Verify, rather than trust. A transform that is not affine in the point
    // (B-spline, displacement field) misses these by O(1) and declines to the CPU.
    for probe in [PROBE, PROBE_FAR] {
        let truth = transform.transform_point(&probe);
        if truth.len() != dim {
            return None;
        }
        for (d, &want) in truth.iter().enumerate() {
            let predicted = form.b[d]
                + (0..dim)
                    .map(|e| form.a[d * dim + e] * probe[e])
                    .sum::<f64>();
            if !close(predicted, want) {
                return None;
            }
        }

        let jt = transform.jacobian_wrt_parameters(&probe);
        if jt.len() != dim * nparams {
            return None;
        }
        for (idx, &want) in jt.iter().enumerate() {
            let predicted = form.j0[idx] + (0..dim).map(|e| probe[e] * form.c[e][idx]).sum::<f64>();
            if !close(predicted, want) {
                return None;
            }
        }
    }
    Some(form)
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
fn contract(moments: &sitk_cuda::Moments, form: &AffineForm) -> MetricValue {
    let dim = sitk_cuda::DIM;
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
