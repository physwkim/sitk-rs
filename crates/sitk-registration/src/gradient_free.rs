//! Gradient-free (derivative-free) optimizers: Amoeba, Powell, 1+1
//! evolutionary strategy, and exhaustive grid search.
//!
//! Ports of `itk::AmoebaOptimizerv4` (wrapping vnl's Nelder–Mead
//! `vnl_amoeba`), `itk::PowellOptimizerv4` (Powell's method with a
//! golden-section/Brent line search), `itk::OnePlusOneEvolutionaryOptimizerv4`
//! (a 1+1 evolution strategy driven by `itk::Statistics::NormalVariateGenerator`),
//! and `itk::ExhaustiveOptimizerv4` (full grid search). Unlike the
//! [`crate::optimizer`] and [`crate::lbfgsb`] optimizers, none of these need a
//! gradient: every `optimize` here takes a **value-only** objective
//! (`FnMut(&[f64]) -> f64`).
//!
//! Constructor and setter names mirror SimpleITK's
//! `ImageRegistrationMethod::SetOptimizerAs{Amoeba,Powell,OnePlusOneEvolutionary,
//! Exhaustive}`, including their defaults. Two deliberate deviations from
//! SimpleITK, both required to keep this crate's "no new external
//! dependencies, seeded and deterministic" rule for randomized optimizers:
//!
//! - [`OnePlusOneEvolutionaryOptimizer`] requires an explicit seed rather than
//!   SimpleITK's `sitkWallClock` sentinel (which seeds from `time()`/`clock()`
//!   when the seed is `0`).
//! - [`AmoebaOptimizer`]'s multi-restart heuristic (`with_restarts`) perturbs
//!   its simplex with a random sign; ITK does this via an unseeded, platform-
//!   dependent libc `rand()` call, which is not reproducible even in ITK
//!   itself. This port substitutes the same seeded generator
//!   [`OnePlusOneEvolutionaryOptimizer`] uses (fixed seed `0`) so the result is
//!   deterministic.
//!
//! `itk::Statistics::NormalVariateGenerator` (C.S. Wallace's 1994 fast
//! generator, the RNG `OnePlusOneEvolutionaryOptimizerv4` requires) is ported
//! from scratch below as a private `NormalVariateGenerator`, including the
//! 32-bit two's-complement integer arithmetic the original assumes; its
//! `goto`-based `FastNorm` is restructured into equivalent straight-line
//! control flow (see the doc comment on its private `fast_norm`).

use crate::optimizer::{OptimizerResult, StopReason};

// ============================================================================
// Amoeba (Nelder–Mead downhill simplex, `itk::AmoebaOptimizerv4` / `vnl_amoeba`)
// ============================================================================

/// One corner of the Nelder–Mead simplex (`vnl_amoeba_SimplexCorner`): a
/// position and its objective value.
#[derive(Clone)]
struct SimplexCorner {
    v: Vec<f64>,
    fv: f64,
}

/// Evaluate the objective at `v`, bump the evaluation counter, and wrap the
/// result as a [`SimplexCorner`] (`vnl_amoebaFit::set_corner`).
fn eval_corner<F: FnMut(&[f64]) -> f64>(v: Vec<f64>, eval: &mut F, cnt: &mut u32) -> SimplexCorner {
    *cnt += 1;
    let fv = eval(&v);
    SimplexCorner { v, fv }
}

/// `(1 − λ)·vbar + λ·v`, evaluated as a new simplex corner
/// (`vnl_amoebaFit::set_corner_a_plus_bl`) — the shared primitive behind
/// reflection (`λ = −1`), expansion (`λ = 2`), and contraction/shrink
/// (`λ = 0.5`).
fn corner_a_plus_bl<F: FnMut(&[f64]) -> f64>(
    vbar: &[f64],
    v: &[f64],
    lambda: f64,
    eval: &mut F,
    cnt: &mut u32,
) -> SimplexCorner {
    let nv: Vec<f64> = vbar
        .iter()
        .zip(v)
        .map(|(&a, &b)| (1.0 - lambda) * a + lambda * b)
        .collect();
    eval_corner(nv, eval, cnt)
}

/// Largest per-component absolute difference between two equal-length points
/// (`vnl_amoeba.cxx`'s free `maxabsdiff`).
fn maxabsdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Simplex diameter: the largest [`maxabsdiff`] between consecutive
/// (sorted-by-value) corners (`vnl_amoeba.cxx`'s free `simplex_diameter`).
fn simplex_diameter(simplex: &[SimplexCorner]) -> f64 {
    let mut max = 0.0;
    for i in 0..simplex.len() - 1 {
        let d = maxabsdiff(&simplex[i].v, &simplex[i + 1].v);
        if d > max {
            max = d;
        }
    }
    max
}

/// Spread of objective values across the (sorted) simplex: worst minus best
/// (`vnl_amoeba.cxx`'s free `sorted_simplex_fdiameter`).
fn sorted_fdiameter(simplex: &[SimplexCorner]) -> f64 {
    simplex[simplex.len() - 1].fv - simplex[0].fv
}

/// Faithful port of `vnl_amoebaFit::amoeba`'s core Nelder–Mead loop, given an
/// already-built starting simplex (`x` plus per-dimension offsets `dx` —
/// `vnl_amoeba`'s "absolute" setup, the only mode `AmoebaOptimizerv4` uses).
/// Runs until the simplex diameter and function-value spread both fall under
/// tolerance, or `max_evaluations` objective calls have been spent. Returns
/// the best corner found and the number of evaluations used.
fn vnl_amoeba_minimize<F: FnMut(&[f64]) -> f64>(
    x: &[f64],
    dx: &[f64],
    x_tolerance: f64,
    f_tolerance: f64,
    max_evaluations: u32,
    eval: &mut F,
) -> (Vec<f64>, f64, u32) {
    let n = x.len();
    let mut cnt = 0u32;

    let mut simplex: Vec<SimplexCorner> = Vec::with_capacity(n + 1);
    simplex.push(eval_corner(x.to_vec(), eval, &mut cnt));
    for j in 0..n {
        let mut v = x.to_vec();
        v[j] += dx[j];
        simplex.push(eval_corner(v, eval, &mut cnt));
    }
    simplex.sort_by(|a, b| a.fv.total_cmp(&b.fv));

    while cnt < max_evaluations {
        if simplex_diameter(&simplex) < x_tolerance && sorted_fdiameter(&simplex) < f_tolerance {
            break;
        }

        let mut vbar = vec![0.0; n];
        for corner in simplex.iter().take(n) {
            for (k, vb) in vbar.iter_mut().enumerate() {
                *vb += corner.v[k];
            }
        }
        for v in vbar.iter_mut() {
            *v /= n as f64;
        }

        let reflect = corner_a_plus_bl(&vbar, &simplex[n].v, -1.0, eval, &mut cnt);

        let next = if reflect.fv < simplex[n - 1].fv {
            if reflect.fv < simplex[0].fv {
                let expand = corner_a_plus_bl(&vbar, &reflect.v, 2.0, eval, &mut cnt);
                if expand.fv < simplex[0].fv {
                    expand
                } else {
                    reflect
                }
            } else {
                reflect
            }
        } else {
            let contract_target: &[f64] = if reflect.fv < simplex[n].fv {
                &reflect.v
            } else {
                &simplex[n].v
            };
            let contract = corner_a_plus_bl(&vbar, contract_target, 0.5, eval, &mut cnt);
            if contract.fv < simplex[0].fv {
                contract
            } else {
                let best_v = simplex[0].v.clone();
                for corner in simplex.iter_mut().take(n).skip(1) {
                    let v = corner.v.clone();
                    *corner = corner_a_plus_bl(&best_v, &v, 0.5, eval, &mut cnt);
                }
                corner_a_plus_bl(&best_v, &simplex[n].v, 0.5, eval, &mut cnt)
            }
        };
        simplex[n] = next;
        simplex.sort_by(|a, b| a.fv.total_cmp(&b.fv));
    }

    (simplex[0].v.clone(), simplex[0].fv, cnt)
}

/// Fixed seed for the deterministic RNG substituted for ITK's unseeded
/// `rand()` in [`AmoebaOptimizer`]'s restart heuristic — see the module doc.
const AMOEBA_RESTART_RNG_SEED: i32 = 0;

/// Nelder–Mead downhill simplex optimizer (`itk::AmoebaOptimizerv4`, wrapping
/// vnl's `vnl_amoeba`).
///
/// Builds a simplex of `n + 1` corners around the initial point (each corner
/// `i` offset from the start by `simplex_delta` along dimension `i`) and
/// repeatedly reflects, expands, contracts, or shrinks it toward lower
/// objective values. Runs in ITK's **internal (scaled) coordinate system**:
/// `internal = external · scales`, matching
/// `SingleValuedVnlCostFunctionAdaptorv4::f` (`external = internal / scales`)
/// — the initial point is scaled up before optimizing and the result scaled
/// back down, but `simplex_delta` itself is **not** scaled (SimpleITK always
/// uses a manually-specified, per-dimension-uniform delta — ITK's
/// `AutomaticInitialSimplex` mode is never reached through
/// `SetOptimizerAsAmoeba` and so is not ported).
///
/// With `with_restarts` enabled, the optimizer reruns after convergence,
/// reseeding the simplex at the best solution found so far with a delta
/// halved (and randomly sign-flipped) each restart, stopping when both the
/// parameters and function value stop changing or the iteration budget is
/// exhausted.
#[derive(Clone, Debug)]
pub struct AmoebaOptimizer {
    simplex_delta: f64,
    number_of_iterations: usize,
    scales: Option<Vec<f64>>,
    parameters_convergence_tolerance: f64,
    function_convergence_tolerance: f64,
    with_restarts: bool,
}

impl AmoebaOptimizer {
    /// An Amoeba optimizer with the given (uniform, per-parameter) initial
    /// simplex delta and iteration cap — mirrors
    /// `SetOptimizerAsAmoeba(simplexDelta, numberOfIterations, ...)`.
    /// Parameters-convergence tolerance defaults to `1e-8`, function-
    /// convergence tolerance to `1e-4` (ITK's defaults), and restarts are
    /// disabled.
    pub fn new(simplex_delta: f64, number_of_iterations: usize) -> Self {
        Self {
            simplex_delta,
            number_of_iterations,
            scales: None,
            parameters_convergence_tolerance: 1e-8,
            function_convergence_tolerance: 1e-4,
            with_restarts: false,
        }
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Set the simplex-diameter tolerance (in internal/scaled coordinates)
    /// below which, together with [`set_function_convergence_tolerance`],
    /// optimization stops (ITK default `1e-8`).
    ///
    /// [`set_function_convergence_tolerance`]: Self::set_function_convergence_tolerance
    pub fn set_parameters_convergence_tolerance(&mut self, tolerance: f64) -> &mut Self {
        self.parameters_convergence_tolerance = tolerance;
        self
    }

    /// Set the function-value-spread tolerance across simplex corners below
    /// which, together with [`set_parameters_convergence_tolerance`],
    /// optimization stops (ITK default `1e-4`).
    ///
    /// [`set_parameters_convergence_tolerance`]: Self::set_parameters_convergence_tolerance
    pub fn set_function_convergence_tolerance(&mut self, tolerance: f64) -> &mut Self {
        self.function_convergence_tolerance = tolerance;
        self
    }

    /// Enable the multi-restart heuristic (ITK default off): rerun the
    /// simplex search from the best solution found, with a halved and
    /// randomly sign-flipped delta, until convergence or the iteration
    /// budget is exhausted.
    pub fn set_with_restarts(&mut self, with_restarts: bool) -> &mut Self {
        self.with_restarts = with_restarts;
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run the Nelder–Mead simplex search from `initial`. `eval(p)` returns
    /// the objective value at `p`. Returns the best point visited.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> f64,
    {
        let n = initial.len();
        let ones = vec![1.0; n];
        let scales: &[f64] = match &self.scales {
            Some(s) => {
                assert_eq!(s.len(), n, "scales length must equal parameter count");
                s
            }
            None => &ones,
        };

        // ITK's internal (scaled) coordinate system: internal = external * scales.
        let internal_initial: Vec<f64> = (0..n).map(|k| initial[k] * scales[k]).collect();
        let delta = vec![self.simplex_delta; n];

        let mut eval_internal = |p: &[f64]| -> f64 {
            let external: Vec<f64> = (0..n).map(|k| p[k] / scales[k]).collect();
            eval(&external)
        };

        let max_eval = self.number_of_iterations as u32;
        let (mut best_internal, mut best_value, mut used) = vnl_amoeba_minimize(
            &internal_initial,
            &delta,
            self.parameters_convergence_tolerance,
            self.function_convergence_tolerance,
            max_eval,
            &mut eval_internal,
        );

        if self.with_restarts {
            let mut rng = NormalVariateGenerator::new(AMOEBA_RESTART_RNG_SEED);
            let mut i = 1u32;
            let mut converged = false;
            while !converged && used < max_eval {
                let remaining = max_eval - used;
                let sign = if rng.next_variate() > 0.0 { 1.0 } else { -1.0 };
                let factor = sign / 2f64.powi(i as i32);
                let restart_delta: Vec<f64> = delta.iter().map(|d| d * factor).collect();
                let (candidate, candidate_value, spent) = vnl_amoeba_minimize(
                    &best_internal,
                    &restart_delta,
                    self.parameters_convergence_tolerance,
                    self.function_convergence_tolerance,
                    remaining,
                    &mut eval_internal,
                );
                used += spent;
                let max_abs = (0..n).fold(0.0f64, |m, k| {
                    m.max((best_internal[k] - candidate[k]).abs())
                });
                converged = (best_value - candidate_value).abs()
                    < self.function_convergence_tolerance
                    && max_abs < self.parameters_convergence_tolerance;
                if candidate_value < best_value {
                    best_value = candidate_value;
                    best_internal = candidate;
                }
                i += 1;
            }
        }

        let external: Vec<f64> = (0..n).map(|k| best_internal[k] / scales[k]).collect();

        OptimizerResult {
            parameters: external,
            value: best_value,
            iterations: used as usize,
            stop_reason: if used < max_eval {
                StopReason::Converged
            } else {
                StopReason::MaxIterations
            },
        }
    }
}

// ============================================================================
// Powell (`itk::PowellOptimizerv4`)
// ============================================================================

/// Position at scaled line-parameter `x` along `direction` from `origin`
/// (`itk::PowellOptimizerv4::GetLineValue`'s position computation): each
/// component of `direction` is divided by its scale, matching `SetLine`'s
/// `m_LineDirection[i] = direction[i] / scales[i]`.
fn line_position(origin: &[f64], direction: &[f64], scales: &[f64], x: f64) -> Vec<f64> {
    (0..origin.len())
        .map(|k| origin[k] + x * direction[k] / scales[k])
        .collect()
}

/// Golden-ratio bracketing of a 1-D minimum along `direction` from `origin`
/// (`itk::PowellOptimizerv4::LineBracket`). `(x1, f1)` is the known value at
/// the line origin (always `x1 = 0`, `f1 = fx`); `x2` is the initial step
/// distance to try. Returns `(ax, xx, bx, fx)`: the outer bracket `ax`, the
/// best interior point `xx` found while extrapolating and its value `fx`, and
/// the far bracket point `bx` — exactly the state ITK's caller threads into
/// `BracketedLineOptimize`.
fn line_bracket<F: FnMut(&[f64]) -> f64>(
    origin: &[f64],
    direction: &[f64],
    scales: &[f64],
    mut x1: f64,
    mut f1: f64,
    mut x2: f64,
    eval: &mut F,
) -> (f64, f64, f64, f64) {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut f2 = eval(&line_position(origin, direction, scales, x2));
    if f2 >= f1 {
        std::mem::swap(&mut x1, &mut x2);
        std::mem::swap(&mut f1, &mut f2);
    }
    let mut x3 = x1 + golden_ratio * (x2 - x1);
    let mut f3 = eval(&line_position(origin, direction, scales, x3));
    while f3 < f2 {
        x2 = x3;
        f2 = f3;
        x3 = x1 + golden_ratio * (x2 - x1);
        f3 = eval(&line_position(origin, direction, scales, x3));
    }
    (x1, x2, x3, f2)
}

/// Brent-style bracketed line minimization (`itk::PowellOptimizerv4::
/// BracketedLineOptimize`, adapted from Numerical Recipes). `(ax, bx, cx)`
/// bracket the minimum with `bx` interior and `function_value_of_b` its
/// (already-known) value. Returns `(x, f(x))` at the located extremum.
#[allow(clippy::too_many_arguments)]
fn bracketed_line_optimize<F: FnMut(&[f64]) -> f64>(
    origin: &[f64],
    direction: &[f64],
    scales: &[f64],
    ax: f64,
    bx: f64,
    cx: f64,
    function_value_of_b: f64,
    step_tolerance: f64,
    maximum_line_iteration: usize,
    eval: &mut F,
) -> (f64, f64) {
    const POWELL_TINY: f64 = 1.0e-20;
    let golden_section_ratio = (3.0 - 5.0_f64.sqrt()) / 2.0;

    let mut a = ax.min(cx);
    let mut b = ax.max(cx);
    let mut x = bx;
    let mut w = bx;
    let mut v = 0.0;

    let mut function_value_of_v = function_value_of_b;
    let mut function_value_of_x = function_value_of_v;
    let mut function_value_of_w = function_value_of_v;

    for _ in 0..maximum_line_iteration {
        let middle_range = (a + b) / 2.0;
        let tolerance1 = step_tolerance * x.abs() + POWELL_TINY;
        let tolerance2 = 2.0 * tolerance1;

        if (x - middle_range).abs() <= (tolerance2 - 0.5 * (b - a))
            || 0.5 * (b - a) < step_tolerance
        {
            return (x, function_value_of_x);
        }

        let mut new_step = golden_section_ratio * (if x < middle_range { b - x } else { a - x });

        if (x - w).abs() >= tolerance1 {
            let t = (x - w) * (function_value_of_x - function_value_of_v);
            let mut q = (x - v) * (function_value_of_x - function_value_of_w);
            let mut p = (x - v) * q - (x - w) * t;
            q = 2.0 * (q - t);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            if p.abs() < (new_step * q).abs()
                && p > q * (a - x + 2.0 * tolerance1)
                && p < q * (b - x - 2.0 * tolerance1)
            {
                new_step = p / q;
            }
        }

        if new_step.abs() < tolerance1 {
            new_step = if new_step > 0.0 {
                tolerance1
            } else {
                -tolerance1
            };
        }

        let t_point = x + new_step;
        let function_value_of_t = eval(&line_position(origin, direction, scales, t_point));

        if function_value_of_t <= function_value_of_x {
            if t_point < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = t_point;
            function_value_of_v = function_value_of_w;
            function_value_of_w = function_value_of_x;
            function_value_of_x = function_value_of_t;
        } else {
            if t_point < x {
                a = t_point;
            } else {
                b = t_point;
            }
            if function_value_of_t <= function_value_of_w || w == x {
                v = w;
                w = t_point;
                function_value_of_v = function_value_of_w;
                function_value_of_w = function_value_of_t;
            } else if function_value_of_t <= function_value_of_v || v == x || v == w {
                v = t_point;
                function_value_of_v = function_value_of_t;
            }
        }
    }

    (x, function_value_of_x)
}

/// Powell's method with a golden-section/Brent line search
/// (`itk::PowellOptimizerv4`).
///
/// For an `n`-dimensional parameter space, each outer iteration minimizes the
/// objective along `n` (initially orthogonal) directions in turn, then tries
/// an extrapolated direction through the net displacement of the iteration —
/// Powell's classic heuristic for building non-orthogonal search directions
/// without derivatives. Scales enter only the line search's step-to-position
/// mapping (`position = origin + x · direction / scales`), matching
/// `PowellOptimizerv4::SetLine`.
///
/// Stops when twice the value change is within `value_tolerance` of the
/// value magnitude, or at `number_of_iterations` — note ITK's own loop bound
/// is `<= number_of_iterations`, i.e. it runs one **more** than
/// `number_of_iterations` outer passes if it never converges; this port
/// preserves that.
#[derive(Clone, Debug)]
pub struct PowellOptimizer {
    number_of_iterations: usize,
    maximum_line_iterations: usize,
    step_length: f64,
    step_tolerance: f64,
    value_tolerance: f64,
    scales: Option<Vec<f64>>,
}

impl PowellOptimizer {
    /// A Powell optimizer with SimpleITK's `SetOptimizerAsPowell` defaults:
    /// `numberOfIterations = 100`, `maximumLineIterations = 100`,
    /// `stepLength = 1`, `stepTolerance = 1e-6`, `valueTolerance = 1e-6`.
    pub fn new() -> Self {
        Self {
            number_of_iterations: 100,
            maximum_line_iterations: 100,
            step_length: 1.0,
            step_tolerance: 1e-6,
            value_tolerance: 1e-6,
            scales: None,
        }
    }

    /// Set the maximum number of outer (all-directions) iterations.
    pub fn set_number_of_iterations(&mut self, number_of_iterations: usize) -> &mut Self {
        self.number_of_iterations = number_of_iterations;
        self
    }

    /// Set the maximum number of Brent line-search iterations per direction.
    pub fn set_maximum_line_iterations(&mut self, maximum_line_iterations: usize) -> &mut Self {
        self.maximum_line_iterations = maximum_line_iterations;
        self
    }

    /// Set the initial (scaled) step distance used when bracketing a line
    /// minimum.
    pub fn set_step_length(&mut self, step_length: f64) -> &mut Self {
        self.step_length = step_length;
        self
    }

    /// Set the line-parameter tolerance below which a line search's bracket
    /// is considered collapsed onto the extremum.
    pub fn set_step_tolerance(&mut self, step_tolerance: f64) -> &mut Self {
        self.step_tolerance = step_tolerance;
        self
    }

    /// Set the outer-loop value-convergence tolerance.
    pub fn set_value_tolerance(&mut self, value_tolerance: f64) -> &mut Self {
        self.value_tolerance = value_tolerance;
        self
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run Powell's method from `initial`. `eval(p)` returns the objective
    /// value at `p`.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> f64,
    {
        let n = initial.len();
        let ones = vec![1.0; n];
        let scales: &[f64] = match &self.scales {
            Some(s) => {
                assert_eq!(s.len(), n, "scales length must equal parameter count");
                s
            }
            None => &ones,
        };
        let identity_scales = scales.iter().all(|&s| s == 1.0);

        // Direction set: xi[i] is the i-th search direction, starting as the
        // identity basis (`itk::PowellOptimizerv4`'s `xi.set_identity()`).
        let mut xi: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0; n];
                v[i] = 1.0;
                v
            })
            .collect();

        let mut p = initial.clone();
        let mut pt = p.clone();
        let mut fx = eval(&p);

        let mut stop_reason = StopReason::MaxIterations;
        let mut taken = 0usize;

        for iter in 0..=self.number_of_iterations {
            taken = iter + 1;
            let fp = fx;
            let mut ibig = 0usize;
            let mut del = 0.0;

            for (i, dir) in xi.iter().enumerate() {
                let direction = dir.clone();
                let fptt = fx;

                let (ax, xx, bx, f_at_xx) =
                    line_bracket(&p, &direction, scales, 0.0, fx, self.step_length, &mut eval);
                let (opt_x, opt_val) = bracketed_line_optimize(
                    &p,
                    &direction,
                    scales,
                    ax,
                    xx,
                    bx,
                    f_at_xx,
                    self.step_tolerance,
                    self.maximum_line_iterations,
                    &mut eval,
                );
                p = line_position(&p, &direction, scales, opt_x);
                fx = opt_val;

                if (fptt - fx).abs() > del {
                    del = (fptt - fx).abs();
                    ibig = i;
                }
            }

            if 2.0 * (fp - fx).abs() <= self.value_tolerance * (fp.abs() + fx.abs()) {
                stop_reason = StopReason::Converged;
                break;
            }

            let mut ptt = vec![0.0; n];
            let mut xit = vec![0.0; n];
            for j in 0..n {
                ptt[j] = 2.0 * p[j] - pt[j];
                xit[j] = if identity_scales {
                    p[j] - pt[j]
                } else {
                    (p[j] - pt[j]) * scales[j]
                };
                pt[j] = p[j];
            }

            let fptt = eval(&ptt);
            if fptt < fp {
                let t = 2.0 * (fp - 2.0 * fx + fptt) * (fp - fx - del).powi(2)
                    - del * (fp - fptt).powi(2);
                if t < 0.0 {
                    let (ax, xx, bx, f_at_xx) =
                        line_bracket(&p, &xit, scales, 0.0, fx, 1.0, &mut eval);
                    let (opt_x, opt_val) = bracketed_line_optimize(
                        &p,
                        &xit,
                        scales,
                        ax,
                        xx,
                        bx,
                        f_at_xx,
                        self.step_tolerance,
                        self.maximum_line_iterations,
                        &mut eval,
                    );
                    p = line_position(&p, &xit, scales, opt_x);
                    fx = opt_val;
                    for j in 0..n {
                        xi[ibig][j] = opt_x * xit[j];
                    }
                }
            }
        }

        OptimizerResult {
            parameters: p,
            value: fx,
            iterations: taken,
            stop_reason,
        }
    }
}

impl Default for PowellOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NormalVariateGenerator (`itk::Statistics::NormalVariateGenerator`)
// ============================================================================

const NVG_ELEN: i32 = 7;
const NVG_LEN: i32 = 128;
const NVG_LMASK: i32 = 4 * (NVG_LEN - 1);
const NVG_TLEN: usize = 8 * NVG_LEN as usize;

/// Deterministic seeded pseudo-random unit-normal variate generator
/// (`itk::Statistics::NormalVariateGenerator`, C.S. Wallace's 1994 fast
/// generator). A from-scratch Rust port of the pooled-rotation method: a pool
/// of `NVG_TLEN` (1024) approximately-normal integer deviates is periodically
/// regenerated from a Lehmer-style congruential stream and then repeatedly
/// "rotated" four at a time (a fast, cheap orthogonal transform) to
/// decorrelate reused pool entries into fresh variates.
struct NormalVariateGenerator {
    scale: f64,
    rscale: f64,
    rcons: f64,
    gaussfaze: i32,
    gscale: f64,
    vec1: Vec<i32>,
    nslew: i32,
    irs: i32,
    lseed: i32,
    chic1: f64,
    chic2: f64,
    actual_rsd: f64,
}

impl NormalVariateGenerator {
    /// A generator seeded exactly as ITK's `Initialize(randomSeed)`.
    fn new(seed: i32) -> Self {
        let scale = 30_000_000.0;
        let rscale = 1.0 / scale;
        let rcons = 1.0 / (2.0 * 1024.0 * 1024.0 * 1024.0);
        let fake = 1.0 + 0.125 / NVG_TLEN as f64;
        let chic2 = (2.0 * NVG_TLEN as f64 - fake * fake).sqrt() / fake;
        let chic1 = fake * (0.5 / NVG_TLEN as f64).sqrt();
        Self {
            scale,
            rscale,
            rcons,
            gaussfaze: 1,
            gscale: rscale,
            vec1: vec![0; NVG_TLEN],
            nslew: 0,
            irs: seed,
            lseed: seed,
            chic1,
            chic2,
            actual_rsd: 0.0,
        }
    }

    /// `NormalVariateGenerator::SignedShiftXOR`.
    fn signed_shift_xor(irs: i32) -> i32 {
        let uirs = irs as u32;
        if irs <= 0 {
            ((uirs << 1) ^ 333_556_017) as i32
        } else {
            (uirs << 1) as i32
        }
    }

    /// One step of the Lehmer-style congruential stream that seeds every
    /// random decision in this generator (`m_Lseed`/`m_Irs` update, inlined
    /// throughout ITK's `FastNorm`).
    fn lehmer_step(&mut self) {
        self.lseed = 69069i32.wrapping_mul(self.lseed).wrapping_add(33331);
        self.irs = Self::signed_shift_xor(self.irs);
    }

    /// `NormalVariateGenerator::GetVariate`: return the next cached variate
    /// from the pool, refilling it via [`Self::fast_norm`] once exhausted.
    fn next_variate(&mut self) -> f64 {
        self.gaussfaze -= 1;
        if self.gaussfaze != 0 {
            return self.gscale * self.vec1[self.gaussfaze as usize] as f64;
        }
        self.fast_norm()
    }

    /// `NormalVariateGenerator::FastNorm`, restructured from ITK's
    /// `goto`-based `renormalize` / `recalcsumsq` / `startpass` cascade into
    /// equivalent straight-line control flow:
    ///
    /// - Every 256th call ([`Self::nslew`] a multiple of `0x100`), the
    ///   Chi-squared correction factor ([`Self::recalc_sumsq`]) is
    ///   refreshed; every 65536th call it's preceded by a full pool
    ///   regeneration ([`Self::regenerate_pool`]).
    /// - Every call performs one rotation pass ([`Self::start_pass`]), which
    ///   both produces the returned variate and refills the pool for
    ///   subsequent [`Self::next_variate`] calls.
    fn fast_norm(&mut self) -> f64 {
        if self.nslew & 0xFF == 0 {
            if self.nslew & 0xFFFF != 0 {
                self.recalc_sumsq();
            } else {
                self.regenerate_pool();
                self.recalc_sumsq();
            }
        }
        self.start_pass()
    }

    /// Refill the whole pool with fresh (approximately) unit-normal integer
    /// deviates from the Lehmer stream via a Box–Muller-like polar method,
    /// then rescale so their sum of squares is exactly `NVG_TLEN`
    /// (`FastNorm`'s `renormalize` full-regeneration path).
    fn regenerate_pool(&mut self) {
        let mut ts = 0.0f64;
        let mut p = 0usize;
        loop {
            let (tx, ty, tr) = loop {
                self.lehmer_step();
                let r = self.irs.wrapping_add(self.lseed);
                let tx = self.rcons * r as f64;
                self.lehmer_step();
                let r = self.irs.wrapping_add(self.lseed);
                let ty = self.rcons * r as f64;
                let tr = tx * tx + ty * ty;
                if (0.1..=1.0).contains(&tr) {
                    break (tx, ty, tr);
                }
            };
            self.lehmer_step();
            let mut r = self.irs.wrapping_add(self.lseed);
            if r < 0 {
                r = !r;
            }
            let tz = -2.0 * ((r as f64 + 0.5) * self.rcons).ln();
            ts += tz;
            let tz = (tz / tr).sqrt();
            self.vec1[p] = (self.scale * tx * tz) as i32;
            p += 1;
            self.vec1[p] = (self.scale * ty * tz) as i32;
            p += 1;
            if p >= NVG_TLEN {
                break;
            }
        }
        ts = NVG_TLEN as f64 / ts;
        let tr = ts.sqrt();
        for v in self.vec1.iter_mut() {
            let tx = *v as f64 * tr;
            *v = if tx < 0.0 {
                (tx - 0.5) as i32
            } else {
                (tx + 0.5) as i32
            };
        }
    }

    /// Recompute `m_ActualRSD`, the reciprocal actual standard deviation used
    /// to correct each pass's returned variate (`FastNorm`'s `recalcsumsq`).
    fn recalc_sumsq(&mut self) {
        let mut ts = 0.0f64;
        for &v in self.vec1.iter() {
            let tx = v as f64;
            ts += tx * tx;
        }
        let ts = (ts / (self.scale * self.scale * NVG_TLEN as f64)).sqrt();
        self.actual_rsd = 1.0 / ts;
    }

    /// One rotation pass over the pool (`FastNorm`'s `startpass`/`scanset`):
    /// pick a pseudo-random scan pattern (`stype`) and rotation matrix
    /// (`mtype`), apply it across the pool, then derive the returned variate
    /// from the last pool slot (`endpass`).
    fn start_pass(&mut self) -> f64 {
        self.nslew = self.nslew.wrapping_add(1);
        self.gaussfaze = NVG_TLEN as i32 - 1;

        self.lehmer_step();
        let mut t = self.irs.wrapping_add(self.lseed);
        if t < 0 {
            t = !t;
        }
        t >>= 29 - 2 * NVG_ELEN;
        let mut skew = (NVG_LEN - 1) & t;
        t >>= NVG_ELEN;
        skew = skew.wrapping_mul(4);
        let mut stride = (NVG_LEN / 2 - 1) & t;
        t >>= NVG_ELEN - 1;
        stride = stride.wrapping_mul(8).wrapping_add(4);
        let mtype = t & 3;
        let stype = self.nslew & 3;

        let len = NVG_LEN;
        let (pa, pb, pc, pd, p0, inc, mask, skew, stride) = match stype {
            0 => (
                0,
                len,
                2 * len,
                3 * len,
                4 * len,
                1i32,
                NVG_LMASK,
                skew,
                stride,
            ),
            1 => (
                4 * len,
                5 * len,
                6 * len,
                7 * len,
                0,
                1i32,
                NVG_LMASK,
                skew,
                stride,
            ),
            2 => (
                1,
                1 + 2 * len,
                1 + 4 * len,
                1 + 6 * len,
                0,
                2i32,
                2 * NVG_LMASK,
                skew.wrapping_mul(2),
                stride.wrapping_mul(2),
            ),
            _ => (
                0,
                2 * len,
                4 * len,
                6 * len,
                1,
                2i32,
                2 * NVG_LMASK,
                skew.wrapping_mul(2),
                stride.wrapping_mul(2),
            ),
        };

        match mtype {
            0 => self.matrix0_pass(pa, pb, pc, pd, p0, inc, mask, skew, stride),
            1 => self.matrix1_pass(pa, pb, pc, pd, p0, inc, mask, skew, stride),
            2 => self.matrix2_pass(pa, pb, pc, pd, p0, inc, mask, skew, stride),
            _ => self.matrix3_pass(pa, pb, pc, pd, p0, inc, mask, skew, stride),
        }

        let ts = self.chic1 * (self.chic2 + self.gscale * self.vec1[NVG_TLEN - 1] as f64);
        self.gscale = self.rscale * ts * self.actual_rsd;
        self.gscale * self.vec1[0] as f64
    }

    /// `FastNorm`'s `matrix0`/`mpass0` rotation (`stype`-selected scan
    /// pattern feeds `pa..pd`/`p0`/`inc`/`mask` in; the sign pattern below is
    /// specific to `mtype == 0`).
    #[allow(clippy::too_many_arguments)]
    fn matrix0_pass(
        &mut self,
        mut pa: i32,
        mut pb: i32,
        mut pc: i32,
        mut pd: i32,
        p0: i32,
        inc: i32,
        mask: i32,
        mut skew: i32,
        stride: i32,
    ) {
        // `pa` walks down to one below its starting base on the final
        // iteration (mirroring the C pointer, which is computed but never
        // dereferenced there) — kept as `i32` rather than `usize` so that
        // transient value doesn't panic on underflow.
        pa += inc * (NVG_LEN - 1);
        let mut i = NVG_LEN;
        loop {
            skew = skew.wrapping_add(stride) & mask;
            let mut pe = p0 + skew;

            let mut p = self.vec1[pa as usize].wrapping_neg();
            let mut q = self.vec1[pb as usize].wrapping_neg();
            let mut r = self.vec1[pc as usize];
            let mut s = self.vec1[pd as usize];
            let mut t = p.wrapping_add(q).wrapping_add(r).wrapping_add(s) >> 1;
            p = t.wrapping_sub(p);
            q = t.wrapping_sub(q);
            r = t.wrapping_sub(r);
            s = t.wrapping_sub(s);

            t = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = p;
            pe += inc;
            p = self.vec1[pe as usize];
            self.vec1[pe as usize] = q;
            pe += inc;
            q = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = r;
            pe += inc;
            r = self.vec1[pe as usize];
            self.vec1[pe as usize] = s;

            s = p.wrapping_add(q).wrapping_add(r).wrapping_add(t) >> 1;
            self.vec1[pa as usize] = s.wrapping_sub(p);
            pa -= inc;
            self.vec1[pb as usize] = s.wrapping_sub(q);
            pb += inc;
            self.vec1[pc as usize] = s.wrapping_sub(r);
            pc += inc;
            self.vec1[pd as usize] = s.wrapping_sub(t);
            pd += inc;

            i -= 1;
            if i == 0 {
                break;
            }
        }
    }

    /// `FastNorm`'s `matrix1`/`mpass1` rotation (`mtype == 1`'s sign
    /// pattern).
    #[allow(clippy::too_many_arguments)]
    fn matrix1_pass(
        &mut self,
        mut pa: i32,
        mut pb: i32,
        mut pc: i32,
        mut pd: i32,
        p0: i32,
        inc: i32,
        mask: i32,
        mut skew: i32,
        stride: i32,
    ) {
        pb += inc * (NVG_LEN - 1);
        let mut i = NVG_LEN;
        loop {
            skew = skew.wrapping_add(stride) & mask;
            let mut pe = p0 + skew;

            let mut p = self.vec1[pa as usize].wrapping_neg();
            let mut q = self.vec1[pb as usize];
            let mut r = self.vec1[pc as usize];
            let mut s = self.vec1[pd as usize].wrapping_neg();
            let mut t = p.wrapping_add(q).wrapping_add(r).wrapping_add(s) >> 1;
            p = t.wrapping_sub(p);
            q = t.wrapping_sub(q);
            r = t.wrapping_sub(r);
            s = t.wrapping_sub(s);

            t = self.vec1[pe as usize];
            self.vec1[pe as usize] = p;
            pe += inc;
            p = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = q;
            pe += inc;
            q = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = r;
            pe += inc;
            r = self.vec1[pe as usize];
            self.vec1[pe as usize] = s;

            s = p.wrapping_add(q).wrapping_add(r).wrapping_add(t) >> 1;
            self.vec1[pa as usize] = s.wrapping_sub(p);
            pa += inc;
            self.vec1[pb as usize] = s.wrapping_sub(t);
            pb -= inc;
            self.vec1[pc as usize] = s.wrapping_sub(q);
            pc += inc;
            self.vec1[pd as usize] = s.wrapping_sub(r);
            pd += inc;

            i -= 1;
            if i == 0 {
                break;
            }
        }
    }

    /// `FastNorm`'s `matrix2`/`mpass2` rotation (`mtype == 2`'s sign
    /// pattern).
    #[allow(clippy::too_many_arguments)]
    fn matrix2_pass(
        &mut self,
        mut pa: i32,
        mut pb: i32,
        mut pc: i32,
        mut pd: i32,
        p0: i32,
        inc: i32,
        mask: i32,
        mut skew: i32,
        stride: i32,
    ) {
        pc += inc * (NVG_LEN - 1);
        let mut i = NVG_LEN;
        loop {
            skew = skew.wrapping_add(stride) & mask;
            let mut pe = p0 + skew;

            let mut p = self.vec1[pa as usize];
            let mut q = self.vec1[pb as usize].wrapping_neg();
            let mut r = self.vec1[pc as usize];
            let mut s = self.vec1[pd as usize].wrapping_neg();
            let mut t = p.wrapping_add(q).wrapping_add(r).wrapping_add(s) >> 1;
            p = t.wrapping_sub(p);
            q = t.wrapping_sub(q);
            r = t.wrapping_sub(r);
            s = t.wrapping_sub(s);

            t = self.vec1[pe as usize];
            self.vec1[pe as usize] = p;
            pe += inc;
            p = self.vec1[pe as usize];
            self.vec1[pe as usize] = q;
            pe += inc;
            q = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = r;
            pe += inc;
            r = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = s;

            s = p.wrapping_add(q).wrapping_add(r).wrapping_add(t) >> 1;
            self.vec1[pa as usize] = s.wrapping_sub(r);
            pa += inc;
            self.vec1[pb as usize] = s.wrapping_sub(p);
            pb += inc;
            self.vec1[pc as usize] = s.wrapping_sub(q);
            pc -= inc;
            self.vec1[pd as usize] = s.wrapping_sub(t);
            pd += inc;

            i -= 1;
            if i == 0 {
                break;
            }
        }
    }

    /// `FastNorm`'s `matrix3`/`mpass3` rotation (`mtype == 3`'s sign
    /// pattern).
    #[allow(clippy::too_many_arguments)]
    fn matrix3_pass(
        &mut self,
        mut pa: i32,
        mut pb: i32,
        mut pc: i32,
        mut pd: i32,
        p0: i32,
        inc: i32,
        mask: i32,
        mut skew: i32,
        stride: i32,
    ) {
        pd += inc * (NVG_LEN - 1);
        let mut i = NVG_LEN;
        loop {
            skew = skew.wrapping_add(stride) & mask;
            let mut pe = p0 + skew;

            let mut p = self.vec1[pa as usize];
            let mut q = self.vec1[pb as usize];
            let mut r = self.vec1[pc as usize].wrapping_neg();
            let mut s = self.vec1[pd as usize].wrapping_neg();
            let mut t = p.wrapping_add(q).wrapping_add(r).wrapping_add(s) >> 1;
            p = t.wrapping_sub(p);
            q = t.wrapping_sub(q);
            r = t.wrapping_sub(r);
            s = t.wrapping_sub(s);

            t = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = p;
            pe += inc;
            p = self.vec1[pe as usize];
            self.vec1[pe as usize] = q;
            pe += inc;
            q = self.vec1[pe as usize];
            self.vec1[pe as usize] = r;
            pe += inc;
            r = self.vec1[pe as usize].wrapping_neg();
            self.vec1[pe as usize] = s;

            s = p.wrapping_add(q).wrapping_add(r).wrapping_add(t) >> 1;
            self.vec1[pa as usize] = s.wrapping_sub(q);
            pa += inc;
            self.vec1[pb as usize] = s.wrapping_sub(r);
            pb += inc;
            self.vec1[pc as usize] = s.wrapping_sub(t);
            pc += inc;
            self.vec1[pd as usize] = s.wrapping_sub(p);
            pd -= inc;

            i -= 1;
            if i == 0 {
                break;
            }
        }
    }
}

// ============================================================================
// OnePlusOneEvolutionary (`itk::OnePlusOneEvolutionaryOptimizerv4`)
// ============================================================================

/// 1+1 evolutionary strategy optimizer
/// (`itk::OnePlusOneEvolutionaryOptimizerv4`).
///
/// Each iteration samples a random displacement `delta = A · f_norm` (`A` an
/// `n × n` search-covariance matrix, `f_norm` a vector of independent unit-
/// normal variates) and evaluates the objective at `parent + delta`. An
/// improving child replaces the parent and the search covariance grows by
/// `growth_factor`; otherwise the parent is kept and the covariance shrinks
/// by `shrink_factor` — a rank-1 update in either case, biasing `A` toward
/// directions that have recently paid off. Stops when the Frobenius norm of
/// `A` collapses to `epsilon` (a converged search radius) or at
/// `number_of_iterations`.
///
/// Requires an explicit seed for the unit-normal generator (see the module
/// doc for why this differs from SimpleITK's wall-clock-seeding default).
#[derive(Clone, Debug)]
pub struct OnePlusOneEvolutionaryOptimizer {
    number_of_iterations: usize,
    epsilon: f64,
    initial_radius: f64,
    growth_factor: Option<f64>,
    shrink_factor: Option<f64>,
    seed: i32,
    scales: Option<Vec<f64>>,
}

impl OnePlusOneEvolutionaryOptimizer {
    /// A 1+1 evolutionary optimizer with the given iteration cap and RNG
    /// seed — mirrors `SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations,
    /// ..., seed)` with `seed` mandatory rather than defaulting to
    /// `sitkWallClock`. `epsilon` defaults to `1.5e-4`, `initial_radius` to
    /// `1.01` (ITK's defaults); growth/shrink factors resolve to ITK's
    /// defaults (`1.05` and `1.05^-0.25`) unless overridden.
    pub fn new(number_of_iterations: usize, seed: i32) -> Self {
        Self {
            number_of_iterations,
            epsilon: 1.5e-4,
            initial_radius: 1.01,
            growth_factor: None,
            shrink_factor: None,
            seed,
            scales: None,
        }
    }

    /// Set the minimal search-radius (Frobenius norm of the covariance
    /// matrix) below which optimization stops.
    pub fn set_epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the initial search radius in parameter space.
    pub fn set_initial_radius(&mut self, initial_radius: f64) -> &mut Self {
        self.initial_radius = initial_radius;
        self
    }

    /// Set the search-radius growth factor applied after an improving step
    /// (ITK default `1.05`, resolved when unset).
    pub fn set_growth_factor(&mut self, growth_factor: f64) -> &mut Self {
        self.growth_factor = Some(growth_factor);
        self
    }

    /// Set the search-radius shrink factor applied after a non-improving
    /// step (ITK default `growth_factor^-0.25`, resolved from the
    /// (possibly overridden) growth factor when unset).
    pub fn set_shrink_factor(&mut self, shrink_factor: f64) -> &mut Self {
        self.shrink_factor = Some(shrink_factor);
        self
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run the 1+1 evolutionary search from `initial`. `eval(p)` returns the
    /// objective value at `p`. Deterministic: two calls with the same seed
    /// and objective produce identical results.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> f64,
    {
        let n = initial.len();
        let ones = vec![1.0; n];
        let scales: &[f64] = match &self.scales {
            Some(s) => {
                assert_eq!(s.len(), n, "scales length must equal parameter count");
                s
            }
            None => &ones,
        };
        let growth_factor = self.growth_factor.unwrap_or(1.05);
        let shrink_factor = self
            .shrink_factor
            .unwrap_or_else(|| growth_factor.powf(-0.25));

        let mut rng = NormalVariateGenerator::new(self.seed);

        // A starts diagonal: initial_radius/scales[i] (itk::OnePlusOneEvolutionaryOptimizerv4
        // builds this via `A.set_identity()` then overwriting the diagonal).
        let mut a = vec![vec![0.0; n]; n];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = self.initial_radius / scales[i];
        }

        let mut parent = initial;
        let mut pvalue = eval(&parent);

        let mut stop_reason = StopReason::MaxIterations;
        let mut taken = 0usize;

        for _ in 0..self.number_of_iterations {
            taken += 1;

            let f_norm: Vec<f64> = (0..n).map(|_| rng.next_variate()).collect();
            let delta: Vec<f64> = (0..n)
                .map(|r| (0..n).map(|c| a[r][c] * f_norm[c]).sum())
                .collect();
            let child: Vec<f64> = (0..n).map(|k| parent[k] + delta[k]).collect();
            let cvalue = eval(&child);

            let adjust = if cvalue < pvalue {
                pvalue = cvalue;
                parent = child;
                growth_factor
            } else {
                shrink_factor
            };

            let frobenius_norm = a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
            if frobenius_norm <= self.epsilon {
                stop_reason = StopReason::StepTooSmall;
                break;
            }

            // Rank-1 covariance update: A += ((adjust - 1) / |f_norm|^2) * delta * f_norm^T.
            let denom: f64 = f_norm.iter().map(|v| v * v).sum();
            let alpha = (adjust - 1.0) / denom;
            for c in 0..n {
                for r in 0..n {
                    a[r][c] += alpha * delta[r] * f_norm[c];
                }
            }
        }

        OptimizerResult {
            parameters: parent,
            value: pvalue,
            iterations: taken,
            stop_reason,
        }
    }
}

// ============================================================================
// Exhaustive (`itk::ExhaustiveOptimizerv4`)
// ============================================================================

/// Full grid search (`itk::ExhaustiveOptimizerv4`).
///
/// Samples the objective on a regular grid centered on the initial position:
/// `2 · number_of_steps[i] + 1` points along dimension `i`, spaced
/// `step_length · scales[i]` apart. Unlike the other gradient-free optimizers
/// here, this always runs to completion (there's no early-convergence
/// check), so [`optimize`](Self::optimize) always reports
/// [`StopReason::MaxIterations`] and its `parameters`/`value` are the exact
/// grid point (not an interpolated point) with the lowest objective value —
/// the "best-value grid point".
#[derive(Clone, Debug)]
pub struct ExhaustiveOptimizer {
    number_of_steps: Vec<usize>,
    step_length: f64,
    scales: Option<Vec<f64>>,
}

impl ExhaustiveOptimizer {
    /// An exhaustive-search optimizer mirroring
    /// `SetOptimizerAsExhaustive(numberOfSteps, stepLength = 1.0)`.
    /// `number_of_steps` gives the per-parameter half-width of the grid, in
    /// steps.
    pub fn new(number_of_steps: Vec<usize>, step_length: f64) -> Self {
        Self {
            number_of_steps,
            step_length,
            scales: None,
        }
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Walk the grid from `initial`. `eval(p)` returns the objective value at
    /// `p`. Returns the exact grid point with the lowest value.
    ///
    /// Panics if `number_of_steps` or configured scales' length differs from
    /// `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> f64,
    {
        let n = initial.len();
        assert_eq!(
            self.number_of_steps.len(),
            n,
            "number_of_steps length must equal parameter count"
        );
        let ones = vec![1.0; n];
        let scales: &[f64] = match &self.scales {
            Some(s) => {
                assert_eq!(s.len(), n, "scales length must equal parameter count");
                s
            }
            None => &ones,
        };

        let total: usize = self.number_of_steps.iter().map(|&s| 2 * s + 1).product();

        // ITK seeds Minimum/MaximumMetricValue from the (unshifted) initial
        // position before walking the grid, which itself revisits that same
        // point at its central index — matched here rather than "corrected".
        let mut min_value = eval(&initial);
        let mut min_position = initial.clone();

        let mut index = vec![0usize; n];
        for _ in 0..total {
            let position: Vec<f64> = (0..n)
                .map(|k| {
                    (index[k] as f64 - self.number_of_steps[k] as f64)
                        * self.step_length
                        * scales[k]
                        + initial[k]
                })
                .collect();
            let value = eval(&position);
            if value < min_value {
                min_value = value;
                min_position = position;
            }

            let mut d = 0;
            while d < n {
                index[d] += 1;
                if index[d] > 2 * self.number_of_steps[d] {
                    index[d] = 0;
                    d += 1;
                } else {
                    break;
                }
            }
        }

        OptimizerResult {
            parameters: min_position,
            value: min_value,
            iterations: total,
            stop_reason: StopReason::MaxIterations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(p: &[f64]) -> f64 {
        (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2)
    }

    #[test]
    fn amoeba_minimizes_a_quadratic_bowl() {
        let opt = AmoebaOptimizer::new(1.0, 500);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2)
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-3, "{:?}", r.parameters);
        assert_eq!(r.stop_reason, StopReason::Converged);
    }

    #[test]
    fn amoeba_minimizes_rosenbrock() {
        let opt = AmoebaOptimizer::new(0.5, 5000);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert!((r.parameters[0] - 1.0).abs() < 1e-2, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-2, "{:?}", r.parameters);
        assert_eq!(r.stop_reason, StopReason::Converged);
    }

    #[test]
    fn amoeba_scales_balance_anisotropic_conditioning() {
        let mut opt = AmoebaOptimizer::new(1.0, 2000);
        opt.set_scales(vec![1.0, 1e3]);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 1.0).powi(2) + 1e6 * (p[1] - 1.0).powi(2)
        });
        assert!((r.parameters[0] - 1.0).abs() < 1e-2, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-2, "{:?}", r.parameters);
    }

    #[test]
    fn amoeba_with_restarts_still_converges() {
        let mut opt = AmoebaOptimizer::new(1.0, 2000);
        opt.set_with_restarts(true);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2)
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-2, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-2, "{:?}", r.parameters);
    }

    #[test]
    fn powell_minimizes_a_quadratic_bowl() {
        let opt = PowellOptimizer::new();
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2)
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8);
        assert_eq!(r.stop_reason, StopReason::Converged);
    }

    #[test]
    fn powell_minimizes_rosenbrock() {
        let mut opt = PowellOptimizer::new();
        opt.set_number_of_iterations(200);
        let r = opt.optimize(vec![-1.2, 1.0], rosenbrock);
        assert!((r.parameters[0] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn powell_scales_balance_anisotropic_conditioning() {
        let mut opt = PowellOptimizer::new();
        opt.set_scales(vec![1.0, 1e6]);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 1.0).powi(2) + 1e6 * (p[1] - 1.0).powi(2)
        });
        assert!((r.parameters[0] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn powell_default_matches_new() {
        let a = PowellOptimizer::default();
        let b = PowellOptimizer::new();
        let f = |p: &[f64]| p[0] * p[0];
        assert_eq!(
            a.optimize(vec![1.0], f).parameters,
            b.optimize(vec![1.0], f).parameters
        );
    }

    #[test]
    fn one_plus_one_minimizes_a_quadratic_bowl() {
        let opt = OnePlusOneEvolutionaryOptimizer::new(3000, 42);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2)
        });
        assert!((r.parameters[0] - 3.0).abs() < 0.05, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 0.05, "{:?}", r.parameters);
    }

    #[test]
    fn one_plus_one_stops_on_shrunk_search_radius() {
        let opt = OnePlusOneEvolutionaryOptimizer::new(100_000, 1);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2)
        });
        assert_eq!(r.stop_reason, StopReason::StepTooSmall);
        assert!(r.iterations < 100_000);
    }

    #[test]
    fn one_plus_one_is_deterministic_across_runs_at_the_same_seed() {
        let opt = OnePlusOneEvolutionaryOptimizer::new(500, 7);
        let f = |p: &[f64]| (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
        let r1 = opt.optimize(vec![0.0, 0.0], f);
        let r2 = opt.optimize(vec![0.0, 0.0], f);
        assert_eq!(r1.parameters, r2.parameters);
        assert_eq!(r1.value, r2.value);
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.stop_reason, r2.stop_reason);
    }

    #[test]
    fn one_plus_one_different_seeds_diverge() {
        let f = |p: &[f64]| (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
        let r1 = OnePlusOneEvolutionaryOptimizer::new(50, 1).optimize(vec![0.0, 0.0], f);
        let r2 = OnePlusOneEvolutionaryOptimizer::new(50, 2).optimize(vec![0.0, 0.0], f);
        assert_ne!(r1.parameters, r2.parameters);
    }

    #[test]
    fn exhaustive_finds_the_exact_grid_minimum() {
        // f(p) = (p0-2)^2 + (p1+1)^2, grid step 1, 5 steps each side -> covers
        // [-5, 5], minimum exactly at grid point (2, -1).
        let opt = ExhaustiveOptimizer::new(vec![5, 5], 1.0);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 2.0).powi(2) + (p[1] + 1.0).powi(2)
        });
        assert_eq!(r.parameters, vec![2.0, -1.0]);
        assert_eq!(r.value, 0.0);
        assert_eq!(r.stop_reason, StopReason::MaxIterations);
        assert_eq!(r.iterations, 11 * 11);
    }

    #[test]
    fn exhaustive_scales_change_the_grid_spacing() {
        let mut opt = ExhaustiveOptimizer::new(vec![4, 2], 1.0);
        opt.set_scales(vec![0.5, 2.0]);
        // Spacing becomes stepLength*scale = [0.5, 2.0]; grid covers
        // dim0 in [-2, 2] step 0.5, dim1 in [-4, 4] step 2 -- both exactly hit
        // the target below.
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            (p[0] - 1.5).powi(2) + (p[1] - 4.0).powi(2)
        });
        assert_eq!(r.parameters, vec![1.5, 4.0]);
        assert_eq!(r.value, 0.0);
    }

    #[test]
    fn exhaustive_zero_steps_evaluates_only_the_center() {
        let opt = ExhaustiveOptimizer::new(vec![0, 0], 1.0);
        let r = opt.optimize(vec![1.0, 2.0], |p| p[0] + p[1]);
        assert_eq!(r.parameters, vec![1.0, 2.0]);
        assert_eq!(r.iterations, 1);
    }
}
