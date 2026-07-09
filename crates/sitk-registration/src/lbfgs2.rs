//! Unconstrained limited-memory BFGS with a Moré–Thuente line search
//! (`itk::LBFGS2Optimizerv4`).
//!
//! ITK's `LBFGS2Optimizerv4` wraps [libLBFGS](https://www.chokkan.org/software/liblbfgs/)
//! — Okazaki's C port of Nocedal's L-BFGS FORTRAN code — rather than the older
//! netlib `lbfgs.c`/`lbfgsb.c` that [`crate::LBFGSBOptimizer`] ports. Unlike
//! `LBFGSBOptimizer`, this optimizer has **no bounds**: every variable is free,
//! there is no generalized Cauchy point or subspace minimization, and the
//! two-loop recursion runs directly on the full gradient every iteration.
//!
//! The direction each iteration is Nocedal's two-loop recursion (`Mathematics
//! of Computation` 35(151), 1980; Liu & Nocedal, `Mathematical Programming`
//! B 45(3), 1989) over the last `m` `(s, y)` correction pairs, approximating
//! `-H·g` without forming the Hessian. The step length along that direction is
//! chosen by a line search — by default Moré–Thuente's guaranteed-sufficient-
//! decrease method (`ACM TOMS` 20(3), 1994, ported from libLBFGS's
//! `line_search_morethuente` + `update_trial_interval`), or one of three
//! backtracking variants (Armijo / regular Wolfe / strong Wolfe, Dennis &
//! Schnabel 1983) selected via [`LBFGS2Optimizer::set_line_search_method`].
//!
//! ## Port structure
//!
//! This is a direct port of libLBFGS's `lbfgs()` driver and its line-search
//! routines (`Modules/ThirdParty/libLBFGS/src/itklbfgs/lib/lbfgs.c` in the
//! ITK tree), restructured so the driver calls the caller's `eval` closure
//! inline wherever the C code calls `cd->proc_evaluate`. Function names below
//! mirror the C source: `update_trial_interval`, `cubic_minimizer`,
//! `cubic_minimizer2`, `quad_minimizer`, `quad_minimizer2`.
//!
//! **Not ported**: the Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) L1
//! extension (`orthantwise_c`/`orthantwise_start`/`orthantwise_end`,
//! `line_search_backtracking_owlqn`, `owlqn_*`). SimpleITK's
//! `SetOptimizerAsLBFGS2` convenience method never exposes it either — OWL-QN
//! is for L1-regularized log-linear models, not image registration.
//!
//! Setter names mirror `itk::simple::ImageRegistrationMethod::
//! SetOptimizerAsLBFGS2`'s parameters (`solutionAccuracy`,
//! `numberOfIterations`, `hessianApproximateAccuracy`,
//! `deltaConvergenceDistance`, `deltaConvergenceTolerance`,
//! `lineSearchMaximumEvaluations`, `lineSearchMinimumStep`,
//! `lineSearchMaximumStep`, `lineSearchAccuracy`); the line-search method
//! selection and its Wolfe/gradient-accuracy/machine-precision parameters are
//! native to `itk::LBFGS2Optimizerv4` but not exposed by that convenience
//! method, so they keep libLBFGS's own defaults (`wolfe = 0.9`, `gtol = 0.9`,
//! `xtol = 1e-16`).
//!
//! ## Stopping criteria and their [`StopReason`] mapping
//!
//! - **Gradient test** (always active): stops when `‖g‖ ≤ solution_accuracy ·
//!   max(1, ‖x‖)` → [`StopReason::GradientConverged`]. Checked once before the
//!   first iteration too (an already-stationary start returns immediately with
//!   zero iterations), matching libLBFGS's `LBFGS_ALREADY_MINIMIZED`.
//! - **Delta test** (opt-in via `delta_convergence_distance > 0`): stops when
//!   the objective's relative change over the last `delta_convergence_distance`
//!   iterations falls below `delta_convergence_tolerance` →
//!   [`StopReason::Converged`].
//! - **Iteration cap** (`number_of_iterations`, `0` = unlimited, matching ITK):
//!   → [`StopReason::MaxIterations`].
//! - **Line search failure** (rounding error, step outside `[min_step,
//!   max_step]`, exceeding `line_search_maximum_evaluations`, an interval that
//!   collapsed below machine precision, or a direction that is not a descent
//!   direction) → [`StopReason::LineSearchFailed`]. The crate's `StopReason`
//!   has no finer-grained line-search variants, so every libLBFGS `LBFGSERR_*`
//!   line-search code collapses to this one (see its private `LineSearchError`).
//!
//! On a line-search failure the previous iterate is restored before
//! returning — mirroring the C driver's `veccpy(x, xp, n); veccpy(g, gp, n);`
//! — so `value` always equals `eval` at `parameters` exactly (it is never
//! recomputed, only ever a value returned from a prior `eval` call). Unlike
//! [`LBFGSBOptimizer`](crate::LBFGSBOptimizer), this optimizer does not track
//! a "best point ever seen": libLBFGS's driver returns the **last accepted
//! iterate**, and so does this port.

use crate::optimizer::{OptimizerResult, StopReason};

/// Line search algorithm (libLBFGS `lbfgs_parameter_t::linesearch`).
///
/// `itk::LBFGS2Optimizerv4::SetLineSearch` exposes this selection; SimpleITK's
/// `SetOptimizerAsLBFGS2` convenience method does not, so it always uses the
/// default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineSearchMethod {
    /// Moré–Thuente's guaranteed-sufficient-decrease method (the default).
    #[default]
    MoreThuente,
    /// Backtracking with the sufficient-decrease (Armijo) condition only.
    BacktrackingArmijo,
    /// Backtracking with the regular Wolfe condition (Armijo + curvature).
    BacktrackingWolfe,
    /// Backtracking with the strong Wolfe condition (Armijo + `|dg| ≤ wolfe·|dg₀|`).
    BacktrackingStrongWolfe,
}

/// Why a line search failed to find an acceptable step (libLBFGS's
/// `LBFGSERR_*` codes below `LBFGS_SUCCESS`). Every variant maps to
/// [`StopReason::LineSearchFailed`] in [`LBFGS2Optimizer::optimize`] — the
/// crate's `StopReason` has no finer-grained line-search failure variants.
/// `update_trial_interval`'s own internal failure modes (trial step outside
/// the bracket, or `tmin > tmax`) never escape as a distinct variant here:
/// the Moré–Thuente driver only tests its return value for "nonzero", which
/// this enum surfaces as [`LineSearchError::RoundingError`].
#[derive(Clone, Copy, Debug)]
enum LineSearchError {
    /// The step was not positive at line-search entry.
    InvalidParameters,
    /// The search direction is not a descent direction (`g·s ≥ 0`).
    IncreaseGradient,
    /// The step fell below `line_search_minimum_step`.
    MinimumStep,
    /// The step rose above `line_search_maximum_step`.
    MaximumStep,
    /// Exceeded `line_search_maximum_evaluations`.
    MaximumLineSearch,
    /// Rounding errors prevent further progress, or `update_trial_interval`
    /// rejected the trial point (Moré–Thuente only).
    RoundingError,
    /// The bracketing interval collapsed below machine precision.
    WidthTooSmall,
}

/// Which sufficient-decrease/curvature condition [`line_search_backtracking`]
/// enforces (libLBFGS's three backtracking variants share one function,
/// branching on this exactly as `line_search_backtracking` does on
/// `param->linesearch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BacktrackingCondition {
    Armijo,
    Wolfe,
    StrongWolfe,
}

/// Dot product of two equal-length slices (libLBFGS `vecdot`).
fn vecdot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Euclidean norm (libLBFGS `vec2norm`).
fn vec2norm(a: &[f64]) -> f64 {
    vecdot(a, a).sqrt()
}

/// Minimizer of the cubic interpolating `(u, fu, du)` and `(v, fv, dv)`
/// (libLBFGS `CUBIC_MINIMIZER` macro, used when the two points already
/// bracket a minimizer).
fn cubic_minimizer(u: f64, fu: f64, du: f64, v: f64, fv: f64, dv: f64) -> f64 {
    let d = v - u;
    let theta = (fu - fv) * 3.0 / d + du + dv;
    let s = theta.abs().max(du.abs()).max(dv.abs());
    let a = theta / s;
    let mut gamma = s * (a * a - (du / s) * (dv / s)).sqrt();
    if v < u {
        gamma = -gamma;
    }
    let p = gamma - du + theta;
    let q = gamma - du + gamma + dv;
    u + (p / q) * d
}

/// Safeguarded cubic minimizer used when a bracket is not yet established
/// (libLBFGS `CUBIC_MINIMIZER2` macro): clamps the radicand to `≥ 0` and falls
/// back to `xmin`/`xmax` when the cubic has no finite minimizer in range.
#[allow(clippy::too_many_arguments)]
fn cubic_minimizer2(
    u: f64,
    fu: f64,
    du: f64,
    v: f64,
    fv: f64,
    dv: f64,
    xmin: f64,
    xmax: f64,
) -> f64 {
    let d = v - u;
    let theta = (fu - fv) * 3.0 / d + du + dv;
    let s = theta.abs().max(du.abs()).max(dv.abs());
    let a = theta / s;
    let mut gamma = s * (a * a - (du / s) * (dv / s)).max(0.0).sqrt();
    if u < v {
        gamma = -gamma;
    }
    let p = gamma - dv + theta;
    let q = gamma - dv + gamma + du;
    let r = p / q;
    if r < 0.0 && gamma != 0.0 {
        v - r * d
    } else if d > 0.0 {
        xmax
    } else {
        xmin
    }
}

/// Minimizer of the quadratic interpolating `(u, fu, du)` and `(v, fv)`
/// (libLBFGS `QUAD_MINIMIZER` macro).
fn quad_minimizer(u: f64, fu: f64, du: f64, v: f64, fv: f64) -> f64 {
    let a = v - u;
    u + (du / ((fu - fv) / a + du)) / 2.0 * a
}

/// Minimizer of the quadratic interpolating the derivatives `du` at `u` and
/// `dv` at `v` (the secant step; libLBFGS `QUAD_MINIMIZER2` macro).
fn quad_minimizer2(u: f64, du: f64, v: f64, dv: f64) -> f64 {
    let a = u - v;
    v + (dv / (dv - du)) * a
}

/// Update the safeguarded bracketing interval `[x, y]` (or `[y, x]`) and
/// compute the next trial step in `t` (libLBFGS `update_trial_interval`,
/// itself a C port of MINPACK-2 `dcstep`). `fx`/`dx` and `fy`/`fy` are the
/// function/derivative at the two current endpoints; `ft`/`dt` are read-only
/// values at the trial point `t`. Returns `0` on success, or a nonzero code
/// the caller only tests for `!= 0` (this crate's `line_search_morethuente`
/// mirrors libLBFGS's own use of the return value as a boolean "did this
/// update fail" flag, not a specific code to branch on).
#[allow(clippy::too_many_arguments)]
fn update_trial_interval(
    x: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    y: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    t: &mut f64,
    ft: f64,
    dt: f64,
    tmin: f64,
    tmax: f64,
    brackt: &mut bool,
) -> i32 {
    // fsigndiff(dt, dx): true when dt and dx have opposite signs.
    let dsign = dt * (*dx / dx.abs()) < 0.0;

    if *brackt {
        if *t <= x.min(*y) || x.max(*y) <= *t {
            return 1; // the trial value is out of the interval
        }
        if 0.0 <= *dx * (*t - *x) {
            return 2; // the function must decrease from x
        }
        if tmax < tmin {
            return 3; // incorrect tmin/tmax
        }
    }

    let bound;
    let newt;
    if *fx < ft {
        // Case 1: a higher function value — the minimum is bracketed.
        *brackt = true;
        bound = true;
        let mc = cubic_minimizer(*x, *fx, *dx, *t, ft, dt);
        let mq = quad_minimizer(*x, *fx, *dx, *t, ft);
        newt = if (mc - *x).abs() < (mq - *x).abs() {
            mc
        } else {
            mc + 0.5 * (mq - mc)
        };
    } else if dsign {
        // Case 2: a lower function value, opposite-sign derivatives — bracketed.
        *brackt = true;
        bound = false;
        let mc = cubic_minimizer(*x, *fx, *dx, *t, ft, dt);
        let mq = quad_minimizer2(*x, *dx, *t, dt);
        newt = if (mc - *t).abs() > (mq - *t).abs() {
            mc
        } else {
            mq
        };
    } else if dt.abs() < dx.abs() {
        // Case 3: a lower function value, same-sign derivatives, magnitude decreases.
        bound = true;
        let mc = cubic_minimizer2(*x, *fx, *dx, *t, ft, dt, tmin, tmax);
        let mq = quad_minimizer2(*x, *dx, *t, dt);
        newt = if *brackt {
            if (*t - mc).abs() < (*t - mq).abs() {
                mc
            } else {
                mq
            }
        } else if (*t - mc).abs() > (*t - mq).abs() {
            mc
        } else {
            mq
        };
    } else {
        // Case 4: a lower function value, same-sign derivatives, magnitude does not decrease.
        bound = false;
        newt = if *brackt {
            cubic_minimizer(*t, ft, dt, *y, *fy, *dy)
        } else if *x < *t {
            tmax
        } else {
            tmin
        };
    }

    if *fx < ft {
        *y = *t;
        *fy = ft;
        *dy = dt;
    } else {
        if dsign {
            *y = *x;
            *fy = *fx;
            *dy = *dx;
        }
        *x = *t;
        *fx = ft;
        *dx = dt;
    }

    let mut newt = newt.clamp(tmin, tmax);
    if *brackt && bound {
        let mq = *x + 0.66 * (*y - *x);
        if *x < *y {
            if mq < newt {
                newt = mq;
            }
        } else if newt < mq {
            newt = mq;
        }
    }
    *t = newt;
    0
}

/// Backtracking line search enforcing the Armijo, regular-Wolfe, or
/// strong-Wolfe condition per `condition` (libLBFGS `line_search_backtracking`,
/// which shares this one function across all three via `param->linesearch`).
#[allow(clippy::too_many_arguments)]
fn line_search_backtracking<F>(
    n: usize,
    x: &mut [f64],
    f: &mut f64,
    g: &mut [f64],
    s: &[f64],
    stp: &mut f64,
    xp: &[f64],
    condition: BacktrackingCondition,
    opt: &LBFGS2Optimizer,
    eval: &mut F,
) -> Result<(), LineSearchError>
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    if *stp <= 0.0 {
        return Err(LineSearchError::InvalidParameters);
    }

    let dginit = vecdot(g, s);
    if dginit > 0.0 {
        return Err(LineSearchError::IncreaseGradient);
    }

    let dec = 0.5;
    let inc = 2.1;
    let finit = *f;
    let dgtest = opt.line_search_accuracy * dginit;
    let mut count = 0usize;

    loop {
        for i in 0..n {
            x[i] = xp[i] + *stp * s[i];
        }
        let (fv, gv) = eval(x);
        *f = fv;
        g.copy_from_slice(&gv);
        count += 1;

        let width;
        if *f > finit + *stp * dgtest {
            width = dec;
        } else {
            if condition == BacktrackingCondition::Armijo {
                return Ok(());
            }
            let dg = vecdot(g, s);
            if dg < opt.wolfe_coefficient * dginit {
                width = inc;
            } else {
                if condition == BacktrackingCondition::Wolfe {
                    return Ok(());
                }
                if dg > -opt.wolfe_coefficient * dginit {
                    width = dec;
                } else {
                    return Ok(());
                }
            }
        }

        if *stp < opt.line_search_minimum_step {
            return Err(LineSearchError::MinimumStep);
        }
        if *stp > opt.line_search_maximum_step {
            return Err(LineSearchError::MaximumStep);
        }
        if opt.line_search_maximum_evaluations <= count {
            return Err(LineSearchError::MaximumLineSearch);
        }

        *stp *= width;
    }
}

/// Moré–Thuente line search guaranteeing sufficient decrease and curvature
/// (libLBFGS `line_search_morethuente`).
#[allow(clippy::too_many_arguments)]
fn line_search_morethuente<F>(
    n: usize,
    x: &mut [f64],
    f: &mut f64,
    g: &mut [f64],
    s: &[f64],
    stp: &mut f64,
    xp: &[f64],
    opt: &LBFGS2Optimizer,
    eval: &mut F,
) -> Result<(), LineSearchError>
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    if *stp <= 0.0 {
        return Err(LineSearchError::InvalidParameters);
    }

    let dginit = vecdot(g, s);
    if dginit > 0.0 {
        return Err(LineSearchError::IncreaseGradient);
    }

    let mut brackt = false;
    let mut stage1 = true;
    let finit = *f;
    let dgtest = opt.line_search_accuracy * dginit;
    let mut width = opt.line_search_maximum_step - opt.line_search_minimum_step;
    let mut prev_width = 2.0 * width;

    let mut stx = 0.0f64;
    let mut sty = 0.0f64;
    let mut fx = finit;
    let mut fy = finit;
    let mut dgx = dginit;
    let mut dgy = dginit;

    let mut count = 0usize;
    let mut uinfo = 0i32;

    loop {
        let (stmin, stmax) = if brackt {
            (stx.min(sty), stx.max(sty))
        } else {
            (stx, *stp + 4.0 * (*stp - stx))
        };

        *stp = (*stp).clamp(opt.line_search_minimum_step, opt.line_search_maximum_step);

        // If an unusual termination is imminent, use the best step found so far.
        if brackt
            && ((*stp <= stmin || stmax <= *stp)
                || opt.line_search_maximum_evaluations <= count + 1
                || uinfo != 0
                || stmax - stmin <= opt.machine_precision_tolerance * stmax)
        {
            *stp = stx;
        }

        for i in 0..n {
            x[i] = xp[i] + *stp * s[i];
        }
        let (fv, gv) = eval(x);
        *f = fv;
        g.copy_from_slice(&gv);
        let dg = vecdot(g, s);

        let ftest1 = finit + *stp * dgtest;
        count += 1;

        if brackt && ((*stp <= stmin || stmax <= *stp) || uinfo != 0) {
            return Err(LineSearchError::RoundingError);
        }
        if *stp == opt.line_search_maximum_step && *f <= ftest1 && dg <= dgtest {
            return Err(LineSearchError::MaximumStep);
        }
        if *stp == opt.line_search_minimum_step && (ftest1 < *f || dgtest <= dg) {
            return Err(LineSearchError::MinimumStep);
        }
        if brackt && (stmax - stmin) <= opt.machine_precision_tolerance * stmax {
            return Err(LineSearchError::WidthTooSmall);
        }
        if opt.line_search_maximum_evaluations <= count {
            return Err(LineSearchError::MaximumLineSearch);
        }
        if *f <= ftest1 && dg.abs() <= opt.line_search_gradient_accuracy * (-dginit) {
            return Ok(());
        }

        // Enter the second stage once a step gives a nonpositive modified
        // function value and a nonnegative modified derivative.
        if stage1
            && *f <= ftest1
            && opt
                .line_search_accuracy
                .min(opt.line_search_gradient_accuracy)
                * dginit
                <= dg
        {
            stage1 = false;
        }

        if stage1 && ftest1 < *f && *f <= fx {
            // A modified function predicts the step while stage 1 has not yet
            // found a sufficiently-decreasing, nonnegative-derivative point.
            let fm = *f - *stp * dgtest;
            let mut fxm = fx - stx * dgtest;
            let mut fym = fy - sty * dgtest;
            let dgm = dg - dgtest;
            let mut dgxm = dgx - dgtest;
            let mut dgym = dgy - dgtest;

            uinfo = update_trial_interval(
                &mut stx,
                &mut fxm,
                &mut dgxm,
                &mut sty,
                &mut fym,
                &mut dgym,
                stp,
                fm,
                dgm,
                stmin,
                stmax,
                &mut brackt,
            );

            fx = fxm + stx * dgtest;
            fy = fym + sty * dgtest;
            dgx = dgxm + dgtest;
            dgy = dgym + dgtest;
        } else {
            uinfo = update_trial_interval(
                &mut stx,
                &mut fx,
                &mut dgx,
                &mut sty,
                &mut fy,
                &mut dgy,
                stp,
                *f,
                dg,
                stmin,
                stmax,
                &mut brackt,
            );
        }

        if brackt {
            if 0.66 * prev_width <= (sty - stx).abs() {
                *stp = stx + 0.5 * (sty - stx);
            }
            prev_width = width;
            width = (sty - stx).abs();
        }
    }
}

/// Unconstrained limited-memory BFGS optimizer (`itk::LBFGS2Optimizerv4`).
///
/// See the [module documentation](self) for the algorithm, the port
/// structure, and the [`StopReason`] mapping.
#[derive(Clone, Debug)]
pub struct LBFGS2Optimizer {
    hessian_approximate_accuracy: usize,
    solution_accuracy: f64,
    delta_convergence_distance: usize,
    delta_convergence_tolerance: f64,
    number_of_iterations: usize,
    line_search_method: LineSearchMethod,
    line_search_maximum_evaluations: usize,
    line_search_minimum_step: f64,
    line_search_maximum_step: f64,
    line_search_accuracy: f64,
    wolfe_coefficient: f64,
    line_search_gradient_accuracy: f64,
    machine_precision_tolerance: f64,
}

impl LBFGS2Optimizer {
    /// An optimizer with ITK's `LBFGS2Optimizerv4` defaults, matching
    /// SimpleITK's `SetOptimizerAsLBFGS2()` called with no arguments:
    /// `solution_accuracy = 1e-5`, `number_of_iterations = 0` (unlimited),
    /// `hessian_approximate_accuracy = 6`, `delta_convergence_distance = 0`
    /// (disabled), `delta_convergence_tolerance = 1e-5`,
    /// `line_search_maximum_evaluations = 40`, `line_search_minimum_step =
    /// 1e-20`, `line_search_maximum_step = 1e20`, `line_search_accuracy =
    /// 1e-4`. The line search defaults to Moré–Thuente, with libLBFGS's own
    /// defaults for the parameters SimpleITK's convenience method does not
    /// expose (`wolfe_coefficient = 0.9`, `line_search_gradient_accuracy =
    /// 0.9`, `machine_precision_tolerance = 1e-16`).
    pub fn new() -> Self {
        Self {
            hessian_approximate_accuracy: 6,
            solution_accuracy: 1e-5,
            delta_convergence_distance: 0,
            delta_convergence_tolerance: 1e-5,
            number_of_iterations: 0,
            line_search_method: LineSearchMethod::MoreThuente,
            line_search_maximum_evaluations: 40,
            line_search_minimum_step: 1e-20,
            line_search_maximum_step: 1e20,
            line_search_accuracy: 1e-4,
            wolfe_coefficient: 0.9,
            line_search_gradient_accuracy: 0.9,
            machine_precision_tolerance: 1e-16,
        }
    }

    /// Set `epsilon`: iteration stops when `‖g‖ ≤ solution_accuracy ·
    /// max(1, ‖x‖)`.
    pub fn set_solution_accuracy(&mut self, solution_accuracy: f64) -> &mut Self {
        self.solution_accuracy = solution_accuracy;
        self
    }

    /// Set the iteration cap. `0` means unlimited (continue until convergence
    /// or a line-search failure), matching ITK.
    pub fn set_number_of_iterations(&mut self, number_of_iterations: usize) -> &mut Self {
        self.number_of_iterations = number_of_iterations;
        self
    }

    /// Set `m`, the number of `(s, y)` correction pairs kept for the two-loop
    /// recursion. Must be positive.
    pub fn set_hessian_approximate_accuracy(&mut self, m: usize) -> &mut Self {
        self.hessian_approximate_accuracy = m;
        self
    }

    /// Set `past`: the number of iterations back the delta convergence test
    /// compares against. `0` disables the test (the default).
    pub fn set_delta_convergence_distance(&mut self, past: usize) -> &mut Self {
        self.delta_convergence_distance = past;
        self
    }

    /// Set `delta`: the delta convergence test stops when `|(f_past − f) / f|
    /// < delta`.
    pub fn set_delta_convergence_tolerance(&mut self, delta: f64) -> &mut Self {
        self.delta_convergence_tolerance = delta;
        self
    }

    /// Set the line search algorithm (native to `itk::LBFGS2Optimizerv4`, not
    /// exposed by SimpleITK's convenience method). Defaults to Moré–Thuente.
    pub fn set_line_search_method(&mut self, method: LineSearchMethod) -> &mut Self {
        self.line_search_method = method;
        self
    }

    /// Set the maximum number of function/gradient evaluations per line
    /// search.
    pub fn set_line_search_maximum_evaluations(&mut self, n: usize) -> &mut Self {
        self.line_search_maximum_evaluations = n;
        self
    }

    /// Set the minimum step a line search may take.
    pub fn set_line_search_minimum_step(&mut self, step: f64) -> &mut Self {
        self.line_search_minimum_step = step;
        self
    }

    /// Set the maximum step a line search may take.
    pub fn set_line_search_maximum_step(&mut self, step: f64) -> &mut Self {
        self.line_search_maximum_step = step;
        self
    }

    /// Set `ftol`, the sufficient-decrease (Armijo) coefficient. Must be in
    /// `(0, 0.5)`.
    pub fn set_line_search_accuracy(&mut self, ftol: f64) -> &mut Self {
        self.line_search_accuracy = ftol;
        self
    }

    /// Set the Wolfe coefficient, used only by
    /// [`LineSearchMethod::BacktrackingWolfe`] and
    /// [`LineSearchMethod::BacktrackingStrongWolfe`]. Must be in
    /// `(line_search_accuracy, 1)`.
    pub fn set_wolfe_coefficient(&mut self, wolfe: f64) -> &mut Self {
        self.wolfe_coefficient = wolfe;
        self
    }

    /// Set `gtol`, the curvature-condition accuracy used by the Moré–Thuente
    /// line search.
    pub fn set_line_search_gradient_accuracy(&mut self, gtol: f64) -> &mut Self {
        self.line_search_gradient_accuracy = gtol;
        self
    }

    /// Set `xtol`, the machine-precision tolerance below which the
    /// Moré–Thuente line search reports a rounding-error failure.
    pub fn set_machine_precision_tolerance(&mut self, xtol: f64) -> &mut Self {
        self.machine_precision_tolerance = xtol;
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn line_search<F>(
        &self,
        n: usize,
        x: &mut [f64],
        f: &mut f64,
        g: &mut [f64],
        s: &[f64],
        stp: &mut f64,
        xp: &[f64],
        eval: &mut F,
    ) -> Result<(), LineSearchError>
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        match self.line_search_method {
            LineSearchMethod::MoreThuente => {
                line_search_morethuente(n, x, f, g, s, stp, xp, self, eval)
            }
            LineSearchMethod::BacktrackingArmijo => line_search_backtracking(
                n,
                x,
                f,
                g,
                s,
                stp,
                xp,
                BacktrackingCondition::Armijo,
                self,
                eval,
            ),
            LineSearchMethod::BacktrackingWolfe => line_search_backtracking(
                n,
                x,
                f,
                g,
                s,
                stp,
                xp,
                BacktrackingCondition::Wolfe,
                self,
                eval,
            ),
            LineSearchMethod::BacktrackingStrongWolfe => line_search_backtracking(
                n,
                x,
                f,
                g,
                s,
                stp,
                xp,
                BacktrackingCondition::StrongWolfe,
                self,
                eval,
            ),
        }
    }

    /// Minimize `eval` from `initial`, where `eval(p)` returns `(value,
    /// gradient)`. Returns the last accepted iterate (see the [module
    /// documentation](self) for why this optimizer does not track a
    /// best-ever point the way [`LBFGSBOptimizer`](crate::LBFGSBOptimizer)
    /// does).
    ///
    /// Panics on invalid input (mirroring libLBFGS's `lbfgs()` parameter
    /// validation): empty `initial`, a non-positive
    /// `hessian_approximate_accuracy` or `line_search_maximum_evaluations`, a
    /// negative accuracy/tolerance parameter, `line_search_maximum_step <
    /// line_search_minimum_step`, or — for the two Wolfe backtracking
    /// methods — a `wolfe_coefficient` outside `(line_search_accuracy, 1)`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        let n = initial.len();
        assert!(n > 0, "initial parameters must be non-empty");
        assert!(
            self.hessian_approximate_accuracy > 0,
            "hessian_approximate_accuracy must be positive"
        );
        assert!(
            self.solution_accuracy >= 0.0,
            "solution accuracy must be non-negative"
        );
        assert!(
            self.delta_convergence_tolerance >= 0.0,
            "delta convergence tolerance must be non-negative"
        );
        assert!(
            self.line_search_maximum_evaluations > 0,
            "line search maximum evaluations must be positive"
        );
        assert!(
            self.line_search_minimum_step >= 0.0,
            "line search minimum step must be non-negative"
        );
        assert!(
            self.line_search_maximum_step >= self.line_search_minimum_step,
            "line search maximum step must be >= minimum step"
        );
        assert!(
            self.line_search_accuracy >= 0.0,
            "line search accuracy (ftol) must be non-negative"
        );
        if matches!(
            self.line_search_method,
            LineSearchMethod::BacktrackingWolfe | LineSearchMethod::BacktrackingStrongWolfe
        ) {
            assert!(
                self.wolfe_coefficient > self.line_search_accuracy && self.wolfe_coefficient < 1.0,
                "wolfe coefficient must be in (line_search_accuracy, 1)"
            );
        }
        assert!(
            self.line_search_gradient_accuracy >= 0.0,
            "line search gradient accuracy (gtol) must be non-negative"
        );
        assert!(
            self.machine_precision_tolerance >= 0.0,
            "machine precision tolerance (xtol) must be non-negative"
        );

        let m = self.hessian_approximate_accuracy;
        let mut x = initial;
        let (mut fx, mut g) = eval(&x);

        // Make sure the initial variables are not already a minimizer.
        let xnorm0 = vec2norm(&x).max(1.0);
        let gnorm0 = vec2norm(&g);
        if gnorm0 / xnorm0 <= self.solution_accuracy {
            return OptimizerResult {
                parameters: x,
                value: fx,
                iterations: 0,
                stop_reason: StopReason::GradientConverged,
            };
        }

        let mut s_hist = vec![vec![0.0f64; n]; m];
        let mut y_hist = vec![vec![0.0f64; n]; m];
        let mut ys_hist = vec![0.0f64; m];

        let past = self.delta_convergence_distance;
        let mut pf: Vec<f64> = if past > 0 {
            vec![0.0; past]
        } else {
            Vec::new()
        };
        if !pf.is_empty() {
            pf[0] = fx;
        }

        // Compute the initial direction assuming H_0 = I, then the initial step
        // `1 / ‖d‖` (libLBFGS `vec2norminv`).
        let mut d: Vec<f64> = g.iter().map(|&gi| -gi).collect();
        let mut step = 1.0 / vec2norm(&d);

        let mut end = 0usize;
        let mut iterations = 0usize;
        let mut xp = x.clone();
        let mut gp = g.clone();

        let stop_reason;
        loop {
            xp.copy_from_slice(&x);
            gp.copy_from_slice(&g);
            let fold = fx;

            if self
                .line_search(n, &mut x, &mut fx, &mut g, &d, &mut step, &xp, &mut eval)
                .is_err()
            {
                // Revert to the previous point, matching the C driver.
                x.copy_from_slice(&xp);
                g.copy_from_slice(&gp);
                fx = fold;
                stop_reason = StopReason::LineSearchFailed;
                break;
            }

            iterations += 1;

            let xnorm = vec2norm(&x).max(1.0);
            let gnorm = vec2norm(&g);
            if gnorm / xnorm <= self.solution_accuracy {
                stop_reason = StopReason::GradientConverged;
                break;
            }

            if !pf.is_empty() {
                if iterations >= past {
                    let rate = (pf[iterations % past] - fx) / fx;
                    if rate.abs() < self.delta_convergence_tolerance {
                        stop_reason = StopReason::Converged;
                        break;
                    }
                }
                pf[iterations % past] = fx;
            }

            if self.number_of_iterations != 0 && iterations >= self.number_of_iterations {
                stop_reason = StopReason::MaxIterations;
                break;
            }

            // Update the limited-memory pair at the current slot, then compute
            // the next two-loop-recursion direction (libLBFGS's "Recursive
            // formula to compute dir = -(H·g)").
            for i in 0..n {
                s_hist[end][i] = x[i] - xp[i];
                y_hist[end][i] = g[i] - gp[i];
            }
            let ys = vecdot(&y_hist[end], &s_hist[end]);
            let yy = vecdot(&y_hist[end], &y_hist[end]);
            ys_hist[end] = ys;

            let bound = m.min(iterations);
            end = (end + 1) % m;

            for (di, &gi) in d.iter_mut().zip(g.iter()) {
                *di = -gi;
            }

            let mut alpha = vec![0.0f64; m];
            let mut j = end;
            for _ in 0..bound {
                j = (j + m - 1) % m;
                let a = vecdot(&s_hist[j], &d) / ys_hist[j];
                alpha[j] = a;
                for i in 0..n {
                    d[i] -= a * y_hist[j][i];
                }
            }

            let scale = ys / yy;
            for di in d.iter_mut() {
                *di *= scale;
            }

            for _ in 0..bound {
                let b = vecdot(&y_hist[j], &d) / ys_hist[j];
                for i in 0..n {
                    d[i] += (alpha[j] - b) * s_hist[j][i];
                }
                j = (j + 1) % m;
            }

            // The search direction is ready; try a full step first.
            step = 1.0;
        }

        OptimizerResult {
            parameters: x,
            value: fx,
            iterations,
            stop_reason,
        }
    }
}

impl Default for LBFGS2Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(p) = (p0 − 3)² + (p1 + 2)², unconstrained minimum at (3, −2).
    fn quadratic(p: &[f64]) -> (f64, Vec<f64>) {
        let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
        let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
        (v, g)
    }

    /// Rosenbrock f(p) = (1 − p0)² + 100(p1 − p0²)², minimum 0 at (1, 1).
    fn rosenbrock(p: &[f64]) -> (f64, Vec<f64>) {
        let v = (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
        let g = vec![
            -2.0 * (1.0 - p[0]) - 400.0 * p[0] * (p[1] - p[0] * p[0]),
            200.0 * (p[1] - p[0] * p[0]),
        ];
        (v, g)
    }

    #[test]
    fn unconstrained_quadratic_reaches_the_minimum() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(200);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        assert!((r.parameters[0] - 3.0).abs() < 1e-6, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-6, "{:?}", r.parameters);
        assert!(r.value < 1e-10, "value {}", r.value);
        assert_eq!(r.stop_reason, StopReason::GradientConverged);
    }

    #[test]
    fn rosenbrock_converges_from_a_hard_start() {
        // (-1.2, 1.0) is the standard hard Rosenbrock start: a curved valley
        // the two-loop recursion and Moré–Thuente line search must navigate.
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(200);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert!((r.parameters[0] - 1.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8, "value {}", r.value);
    }

    #[test]
    fn returned_value_matches_eval_at_returned_parameters() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(200);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        let (recomputed, _) = quadratic(&r.parameters);
        assert_eq!(r.value, recomputed);
    }

    #[test]
    fn max_iterations_stops_with_that_reason() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(3);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert_eq!(r.stop_reason, StopReason::MaxIterations);
        assert_eq!(r.iterations, 3);
    }

    #[test]
    fn already_at_the_minimum_converges_without_iterating() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(100);
        let r = opt.optimize(vec![3.0, -2.0], quadratic);
        assert_eq!(r.stop_reason, StopReason::GradientConverged);
        assert_eq!(r.iterations, 0);
        assert_eq!(r.parameters, vec![3.0, -2.0]);
    }

    #[test]
    fn delta_convergence_test_stops_before_the_gradient_test() {
        // Rosenbrock's curved valley makes BFGS crawl: once the search finds
        // the valley floor, the objective barely changes iteration to
        // iteration even though the gradient is still far above
        // solution_accuracy's default (1e-5) and the true minimum is still
        // far away. A generous number_of_iterations budget and a loose delta
        // tolerance let the delta/past test catch that plateau before either
        // the gradient test or the iteration cap does.
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(500)
            .set_delta_convergence_distance(1)
            .set_delta_convergence_tolerance(1e-2);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert_eq!(r.stop_reason, StopReason::Converged);
        assert!(r.iterations < 500);
    }

    #[test]
    fn backtracking_armijo_reaches_the_minimum() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(500)
            .set_line_search_method(LineSearchMethod::BacktrackingArmijo);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
    }

    #[test]
    fn backtracking_wolfe_reaches_the_minimum() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(500)
            .set_line_search_method(LineSearchMethod::BacktrackingWolfe);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
    }

    #[test]
    fn backtracking_strong_wolfe_reaches_the_minimum() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(500)
            .set_line_search_method(LineSearchMethod::BacktrackingStrongWolfe);
        let r = opt.optimize(vec![0.0, 0.0], quadratic);
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
    }

    #[test]
    fn high_dimensional_quadratic_with_limited_memory() {
        // n ≫ m exercises the compact two-loop recursion: a 20-D separable
        // quadratic with minimum at pᵢ = i, using only m = 5 corrections.
        let n = 20;
        let mut opt = LBFGS2Optimizer::new();
        opt.set_number_of_iterations(500)
            .set_hessian_approximate_accuracy(5);
        let r = opt.optimize(vec![0.0; n], |p| {
            let mut v = 0.0;
            let mut g = vec![0.0; n];
            for i in 0..n {
                let d = p[i] - i as f64;
                v += d * d;
                g[i] = 2.0 * d;
            }
            (v, g)
        });
        for i in 0..n {
            assert!(
                (r.parameters[i] - i as f64).abs() < 1e-5,
                "param {i} = {}",
                r.parameters[i]
            );
        }
        assert!(r.value < 1e-8, "value {}", r.value);
    }

    #[test]
    #[should_panic(expected = "hessian_approximate_accuracy must be positive")]
    fn zero_hessian_approximate_accuracy_panics() {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_hessian_approximate_accuracy(0)
            .set_number_of_iterations(10);
        opt.optimize(vec![0.0], |p| (p[0] * p[0], vec![2.0 * p[0]]));
    }
}
