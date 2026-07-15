//! Gradient-descent optimizer (`itk::GradientDescentOptimizerv4`).
//!
//! Minimizes a scalar objective `f(p)` given its value and gradient at each
//! step. The update is
//!
//! ```text
//! p ← p − learning_rate · (gradient ⊘ scales)
//! ```
//!
//! where `scales` (default all-ones) balances parameters of different physical
//! magnitude — e.g. an affine's matrix entries (`≈1`) versus its translation
//! (`≈ image extent`) — exactly as ITK's optimizer scales do. Iteration stops at
//! `number_of_iterations`, or earlier when convergence monitoring is enabled and
//! the windowed metric value plateaus (see [`crate::registration::convergence`]) — matching
//! `itk::GradientDescentOptimizerv4`, whose only early stop is the convergence
//! monitor (it has no minimum-step-length tolerance; that belongs to
//! `itk::RegularStepGradientDescentOptimizerv4`).

use crate::core::compensated::compensated_sum;

use crate::registration::convergence::WindowConvergenceMonitor;

/// The scalar objective the line-search optimizers minimize.
///
/// They probe [`value`](Self::value) many times per iteration — once per
/// golden-section refinement — and read no gradient on those probes. A caller
/// passing a plain `(value, gradient)` closure to `optimize` therefore computes
/// and drops a gradient at every probe, because a closure has no value-only
/// kernel. A caller that *has* one — as every metric in this crate does —
/// implements this trait, overrides `value`, and calls `optimize_objective`.
pub trait Objective {
    /// `(value, gradient)` at `p`.
    fn value_and_gradient(&mut self, p: &[f64]) -> (f64, Vec<f64>);

    /// The value at `p` alone. Override whenever it is cheaper than throwing
    /// away `value_and_gradient`'s gradient.
    fn value(&mut self, p: &[f64]) -> f64 {
        self.value_and_gradient(p).0
    }
}

/// So an objective can be driven by an optimizer and still be read afterwards
/// (mirrors `impl FnMut for &mut F`). Forwards both methods, so a `&mut O`
/// keeps `O`'s value-only kernel.
impl<O: Objective + ?Sized> Objective for &mut O {
    fn value_and_gradient(&mut self, p: &[f64]) -> (f64, Vec<f64>) {
        (**self).value_and_gradient(p)
    }

    fn value(&mut self, p: &[f64]) -> f64 {
        (**self).value(p)
    }
}

/// Adapts a `(value, gradient)` closure to [`Objective`], for the `optimize`
/// entry points that take one. It cannot override [`Objective::value`] — a
/// closure has no value-only kernel — so every line-search probe through this
/// adapter computes a gradient and drops it. That is what
/// `optimize_objective` exists to avoid.
struct FnObjective<F>(F);

impl<F> Objective for FnObjective<F>
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    fn value_and_gradient(&mut self, p: &[f64]) -> (f64, Vec<f64>) {
        (self.0)(p)
    }
}

/// Why the optimizer stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Hit `number_of_iterations`.
    MaxIterations,
    /// A step-size tolerance was reached: `RegularStepGradientDescentOptimizer`'s
    /// `minimum_step_length`, or `OnePlusOneEvolutionaryOptimizer`'s Frobenius
    /// search-radius floor. `itk::GradientDescentOptimizerv4` and its line-search
    /// subclasses have no such stop — they end early only via [`Self::Converged`].
    StepTooSmall,
    /// The windowed metric value plateaued at or below the minimum convergence
    /// value (`itk::WindowConvergenceMonitoringFunction`).
    Converged,
    /// The scaled gradient magnitude fell below the gradient-magnitude tolerance
    /// — a stationary point (`itk::RegularStepGradientDescentOptimizerv4`'s
    /// `GRADIENT_MAGNITUDE_TOLERANCE`).
    GradientConverged,
    /// Reached the maximum number of objective evaluations
    /// (`LBFGSB`'s `MaximumNumberOfFunctionEvaluations`).
    MaxFunctionEvaluations,
    /// The line search could not make progress (`LBFGSB`'s
    /// `ABNORMAL_TERMINATION_IN_LNSRCH`).
    LineSearchFailed,
}

/// Outcome of an optimization run.
#[derive(Clone, Debug)]
pub struct OptimizerResult {
    /// Best parameters found. For the gradient-descent optimizers this is the
    /// last iterate; [`LBFGSBOptimizer`](crate::registration::LBFGSBOptimizer) returns the
    /// lowest-value point ever evaluated, which may be an earlier iterate.
    pub parameters: Vec<f64>,
    /// Objective value at `parameters`.
    pub value: f64,
    /// Number of steps taken.
    pub iterations: usize,
    /// Why iteration ended.
    pub stop_reason: StopReason,
}

/// Fixed-step gradient descent with optional per-parameter scales.
#[derive(Clone, Debug)]
pub struct GradientDescentOptimizer {
    learning_rate: f64,
    number_of_iterations: usize,
    scales: Option<Vec<f64>>,
    /// `(window_size, minimum_convergence_value)` when value-plateau monitoring
    /// is enabled; `None` disables it (the default).
    convergence: Option<(usize, f64)>,
}

impl GradientDescentOptimizer {
    /// A gradient-descent optimizer with the given step size and iteration cap.
    /// Scales default to all-ones and convergence monitoring is disabled.
    pub fn new(learning_rate: f64, number_of_iterations: usize) -> Self {
        Self {
            learning_rate,
            number_of_iterations,
            scales: None,
            convergence: None,
        }
    }

    /// Set the (fixed) learning rate.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Enable value-plateau convergence monitoring
    /// (`itk::WindowConvergenceMonitoringFunction`): stop once the windowed
    /// metric value's trend flattens to at or below `minimum_convergence_value`.
    /// This is the only early stop for gradient descent — ITK's
    /// `GradientDescentOptimizerv4` enables it by default in every learning-rate
    /// mode (fixed, estimate-once, estimate-each-iteration).
    pub fn set_convergence(
        &mut self,
        window_size: usize,
        minimum_convergence_value: f64,
    ) -> &mut Self {
        self.convergence = Some((window_size, minimum_convergence_value));
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run gradient descent from `initial` at the configured fixed learning
    /// rate. `eval(p)` returns `(value, gradient)` of the objective at `p`.
    /// Returns the last iterate.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        let lr = self.learning_rate;
        self.optimize_with_lr(initial, &mut eval, |_grad| lr)
    }

    /// Like [`optimize`](Self::optimize) but the learning rate is recomputed
    /// each iteration by `learning_rate_of(current_gradient)` — used for ITK's
    /// estimate-learning-rate-at-each-iteration mode, whose step size does not
    /// shrink, so it relies on convergence monitoring (see
    /// [`set_convergence`](Self::set_convergence)) to stop.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize_with_lr<F, L>(
        &self,
        initial: Vec<f64>,
        mut eval: F,
        mut learning_rate_of: L,
    ) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
        L: FnMut(&[f64]) -> f64,
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

        let mut monitor = self
            .convergence
            .map(|(window, _)| WindowConvergenceMonitor::new(window));

        let mut p = initial;
        let (mut value, mut grad) = eval(&p);
        let mut stop_reason = StopReason::MaxIterations;
        let mut taken = 0usize;

        loop {
            // Check the value at the current iterate before stepping, matching
            // ITK's per-iteration convergence test.
            if let (Some(mon), Some((_, min_cv))) = (monitor.as_mut(), self.convergence) {
                mon.add_energy_value(value);
                if let Some(cv) = mon.convergence_value()
                    && cv <= min_cv
                {
                    stop_reason = StopReason::Converged;
                    break;
                }
            }

            if taken >= self.number_of_iterations {
                // stop_reason is already MaxIterations.
                break;
            }

            let lr = learning_rate_of(&grad);
            for k in 0..n {
                p[k] -= lr * grad[k] / scales[k];
            }
            taken += 1;

            let (v, g) = eval(&p);
            value = v;
            grad = g;
        }

        OptimizerResult {
            parameters: p,
            value,
            iterations: taken,
            stop_reason,
        }
    }
}

/// Regular-step gradient descent (`itk::RegularStepGradientDescentOptimizerv4`).
///
/// Each step moves a **fixed length** in the (scaled) gradient's unit direction
/// rather than a length proportional to the gradient. Whenever the gradient
/// reverses direction — the sign that a step overshot the minimum — the step
/// length is multiplied by `relaxation_factor` (halved). Iteration stops when
/// the step length falls below `minimum_step_length`, when the scaled gradient
/// magnitude falls below `gradient_magnitude_tolerance` (a stationary point), or
/// at `number_of_iterations`.
///
/// This reaches `minimum_step_length` precision by repeated halving, and — via
/// the gradient-magnitude tolerance — stops cleanly at a level that starts
/// already converged, without the fixed-rate step explosion that
/// [`GradientDescentOptimizer`]'s estimate-once schedule risks on a
/// multi-resolution restart. `learning_rate` is the initial step-length scale
/// (ITK's `m_LearningRate`); the actual first step length is
/// `learning_rate` (relaxation starts at 1).
#[derive(Clone, Debug)]
pub struct RegularStepGradientDescentOptimizer {
    learning_rate: f64,
    number_of_iterations: usize,
    scales: Option<Vec<f64>>,
    relaxation_factor: f64,
    minimum_step_length: f64,
    gradient_magnitude_tolerance: f64,
}

impl RegularStepGradientDescentOptimizer {
    /// A regular-step optimizer with the given initial step-length scale,
    /// minimum step length, and iteration cap. The relaxation factor
    /// (`0.5`) and gradient-magnitude tolerance (`1e-4`) default to ITK's
    /// values; scales default to all-ones.
    pub fn new(learning_rate: f64, minimum_step_length: f64, number_of_iterations: usize) -> Self {
        Self {
            learning_rate,
            number_of_iterations,
            scales: None,
            relaxation_factor: 0.5,
            minimum_step_length,
            gradient_magnitude_tolerance: 1e-4,
        }
    }

    /// Set the initial step-length scale (ITK's `m_LearningRate`).
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Set the step-length relaxation factor applied on a direction reversal
    /// (ITK default `0.5`). Must be in `[0, 1)`.
    pub fn set_relaxation_factor(&mut self, relaxation_factor: f64) -> &mut Self {
        self.relaxation_factor = relaxation_factor;
        self
    }

    /// Set the minimum step length below which iteration stops.
    pub fn set_minimum_step_length(&mut self, minimum_step_length: f64) -> &mut Self {
        self.minimum_step_length = minimum_step_length;
        self
    }

    /// Set the scaled-gradient-magnitude tolerance below which iteration stops
    /// at a stationary point (ITK default `1e-4`).
    pub fn set_gradient_magnitude_tolerance(&mut self, tolerance: f64) -> &mut Self {
        self.gradient_magnitude_tolerance = tolerance;
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run regular-step gradient descent from `initial` at the configured fixed
    /// initial step-length scale. `eval(p)` returns `(value, gradient)`.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, mut eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        let lr = self.learning_rate;
        self.optimize_with_lr(initial, &mut eval, |_scaled_grad| lr)
    }

    /// Like [`optimize`](Self::optimize) but the step-length scale (ITK's
    /// `m_LearningRate`) is recomputed each iteration by
    /// `learning_rate_of(scaled_gradient)` — used for ITK's
    /// estimate-learning-rate-at-each-iteration mode. The closure receives the
    /// **scaled** gradient (`gradient ⊘ scales`), the same vector the step is
    /// taken along.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize_with_lr<F, L>(
        &self,
        initial: Vec<f64>,
        mut eval: F,
        mut learning_rate_of: L,
    ) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
        L: FnMut(&[f64]) -> f64,
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

        let mut p = initial;
        let (mut value, mut grad) = eval(&p);
        // ITK seeds the previous gradient with zeros, so the first iteration's
        // direction test never fires.
        let mut previous_scaled = vec![0.0; n];
        let mut relaxation = 1.0f64;
        let mut stop_reason = StopReason::MaxIterations;
        let mut taken = 0usize;

        loop {
            let scaled: Vec<f64> = (0..n).map(|k| grad[k] / scales[k]).collect();
            // Compensated, because ITK compensates it and because of what it feeds. Both
            // reductions in this loop go through `CompensatedSummation` upstream
            // (`itkRegularStepGradientDescentOptimizerv4.hxx:107-113` for the magnitude,
            // `:126-133` for the scalar product) and both are read by a **branch**: the
            // stop test immediately below, and the direction-reversal test after it. A
            // reduction a discrete decision is taken on is exactly the reduction whose
            // last bits are not free — §2.157 measured this loop amplifying a 1e-12
            // derivative difference ~500× per step, precisely *because* the reversal test
            // is a branch that a rounding difference can flip.
            //
            // Parity, not merely accuracy: upstream sums over the parameters in index
            // order on one thread, and so does this, so the compensated result is
            // upstream's number. The sum is short for an affine (12 terms) and long for a
            // displacement field (3 per voxel), which is the case ITK's compensation is
            // really for.
            let gradient_magnitude = compensated_sum(scaled.iter().map(|g| g * g)).sqrt();

            // A near-zero gradient is a stationary point; stop before stepping.
            if gradient_magnitude < self.gradient_magnitude_tolerance {
                stop_reason = StopReason::GradientConverged;
                break;
            }

            // A negative inner product with the previous step's gradient means
            // the direction reversed — an overshoot — so relax the step length.
            // ITK weights the stored previous gradient by the prior step's
            // learning rate and an extra `1/scale` factor; for the uniform
            // scales of a translation (and the sign that actually matters here)
            // this reduces to the plain reversal test used below.
            let scalar_product = compensated_sum(
                scaled
                    .iter()
                    .zip(previous_scaled.iter())
                    .map(|(&a, &b)| a * b),
            );
            if scalar_product < 0.0 {
                relaxation *= self.relaxation_factor;
            }

            let step_length = relaxation * learning_rate_of(&scaled);
            if step_length < self.minimum_step_length {
                stop_reason = StopReason::StepTooSmall;
                break;
            }

            if taken >= self.number_of_iterations {
                // stop_reason is already MaxIterations.
                break;
            }

            // Move a fixed `step_length` along the scaled gradient's unit
            // direction (descent, so subtract).
            let factor = step_length / gradient_magnitude;
            for k in 0..n {
                p[k] -= factor * scaled[k];
            }
            previous_scaled = scaled;
            taken += 1;

            let (v, g) = eval(&p);
            value = v;
            grad = g;
        }

        OptimizerResult {
            parameters: p,
            value,
            iterations: taken,
            stop_reason,
        }
    }
}

/// Golden ratio `φ` used by the line search (ITK's `m_Phi`).
const GOLDEN_PHI: f64 = 1.618034;
/// `2 − φ`, the golden-section probe fraction (ITK's `m_Resphi`).
const GOLDEN_RESPHI: f64 = 2.0 - GOLDEN_PHI;

/// Golden-section search for the learning rate that minimizes the objective along
/// the descent direction (`itk::GradientDescentLineSearchOptimizerv4::
/// GoldenSectionSearch`, shared by the conjugate-gradient subclass). `a` and `c`
/// bracket the minimum, `b` is an interior point, `epsilon` sets the collapse
/// resolution, and `max_line_search_iterations` caps the recursion depth.
/// `metricb` caches the objective at `b` across the recursion (`None` means "not
/// yet evaluated"); `line_value(x)` returns the objective at learning rate `x`.
///
/// ITK's fallback for a trial that yields no valid metric samples (its
/// `metricx == max()` branch) is unreachable here: the objective evaluator always
/// returns a finite value, with invalid-sample handling upstream in the metric
/// rather than signaled through a sentinel, so it is omitted.
#[allow(clippy::too_many_arguments)]
fn golden_section_search(
    epsilon: f64,
    max_line_search_iterations: u32,
    a: f64,
    b: f64,
    c: f64,
    metricb: Option<f64>,
    line_search_iterations: &mut u32,
    line_value: &mut dyn FnMut(f64) -> f64,
) -> f64 {
    if *line_search_iterations > max_line_search_iterations {
        return (c + a) / 2.0;
    }
    *line_search_iterations += 1;

    let x = if c - b > b - a {
        b + GOLDEN_RESPHI * (c - b)
    } else {
        b - GOLDEN_RESPHI * (b - a)
    };
    if (c - a).abs() < epsilon * (b.abs() + x.abs()) {
        return (c + a) / 2.0;
    }

    let metricx = line_value(x);
    // ITK evaluates the objective at b only when it is not already known, caching
    // it down the recursion to avoid redundant evaluations.
    let metricb = metricb.unwrap_or_else(|| line_value(b));

    if metricx < metricb {
        if c - b > b - a {
            golden_section_search(
                epsilon,
                max_line_search_iterations,
                b,
                x,
                c,
                Some(metricx),
                line_search_iterations,
                line_value,
            )
        } else {
            golden_section_search(
                epsilon,
                max_line_search_iterations,
                a,
                x,
                b,
                Some(metricx),
                line_search_iterations,
                line_value,
            )
        }
    } else if c - b > b - a {
        golden_section_search(
            epsilon,
            max_line_search_iterations,
            a,
            b,
            x,
            Some(metricb),
            line_search_iterations,
            line_value,
        )
    } else {
        golden_section_search(
            epsilon,
            max_line_search_iterations,
            x,
            b,
            c,
            Some(metricb),
            line_search_iterations,
            line_value,
        )
    }
}

/// Gradient descent with a golden-section line search
/// (`itk::GradientDescentLineSearchOptimizerv4`).
///
/// Plain [`GradientDescentOptimizer`] takes a step of a *fixed* learning rate
/// each iteration; this variant instead, at every iteration, runs a **golden
/// section search** over the learning rate to find the one that most reduces the
/// objective along the current descent direction, before stepping:
///
/// ```text
/// p ← p − learning_rate_by_golden_section · (gradient ⊘ scales)
/// ```
///
/// The search brackets the rate in `[learning_rate · lower_limit,
/// learning_rate · upper_limit]` (ITK defaults `0` and `5`) and refines it until
/// the bracket collapses to within `epsilon` (default `0.01`) or
/// `maximum_line_search_iterations` (default `20`) recursion levels are reached.
/// The learning rate found at one iteration seeds the bracket of the next
/// (ITK estimates the base rate only once, at the first iteration; here it is the
/// configured `learning_rate`, which the driver sets from the scales estimator).
///
/// Because the lower bracket bound reaches a near-zero rate (an almost-unchanged
/// position), the search cannot make the objective worse in a well-behaved basin,
/// so the descent is effectively monotonic; matching ITK's
/// `ReturnBestParametersAndValue = true` default, the run still returns the
/// lowest-value iterate actually visited (correctly paired with its parameters,
/// as [`LBFGSBOptimizer`](crate::registration::LBFGSBOptimizer) does) to guard the rare case a
/// bounded search overshoots.
///
/// Iteration stops at `number_of_iterations`, or earlier when convergence
/// monitoring is enabled and the windowed metric value plateaus (see
/// [`crate::registration::convergence`]) — matching `itk::GradientDescentLineSearchOptimizerv4`,
/// which inherits `GradientDescentOptimizerv4`'s convergence-monitor stop and adds
/// no minimum-step tolerance.
#[derive(Clone, Debug)]
pub struct GradientDescentLineSearchOptimizer {
    learning_rate: f64,
    number_of_iterations: usize,
    scales: Option<Vec<f64>>,
    lower_limit: f64,
    upper_limit: f64,
    epsilon: f64,
    maximum_line_search_iterations: u32,
    /// `(window_size, minimum_convergence_value)` when value-plateau monitoring
    /// is enabled; `None` disables it (the default).
    convergence: Option<(usize, f64)>,
}

impl GradientDescentLineSearchOptimizer {
    /// A line-search optimizer with the given base learning rate and iteration
    /// cap. The bracket limits (`0` and `5`), line-search resolution `epsilon`
    /// (`0.01`), and maximum line-search recursion (`20`) default to ITK's
    /// values; scales default to all-ones and convergence monitoring is disabled.
    pub fn new(learning_rate: f64, number_of_iterations: usize) -> Self {
        Self {
            learning_rate,
            number_of_iterations,
            scales: None,
            lower_limit: 0.0,
            upper_limit: 5.0,
            epsilon: 0.01,
            maximum_line_search_iterations: 20,
            convergence: None,
        }
    }

    /// Set the base learning rate (ITK's `m_LearningRate`), which the first
    /// iteration's line search brackets.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Set the lower and upper bracket limits: the line search adjusts the
    /// learning rate within `[learning_rate · lower, learning_rate · upper]`
    /// (ITK defaults `0` and `5`).
    pub fn set_line_search_limits(&mut self, lower: f64, upper: f64) -> &mut Self {
        self.lower_limit = lower;
        self.upper_limit = upper;
        self
    }

    /// Set the line-search resolution `epsilon`: the bracket is refined until
    /// `|c − a| < epsilon · (|b| + |x|)` (ITK default `0.01`). Smaller values
    /// localize the minimum better at the cost of more objective evaluations.
    pub fn set_epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the maximum golden-section recursion depth per iteration (ITK default
    /// `20`).
    pub fn set_maximum_line_search_iterations(&mut self, iterations: u32) -> &mut Self {
        self.maximum_line_search_iterations = iterations;
        self
    }

    /// Enable value-plateau convergence monitoring
    /// (`itk::WindowConvergenceMonitoringFunction`): stop once the windowed
    /// metric value's trend flattens to at or below `minimum_convergence_value`.
    pub fn set_convergence(
        &mut self,
        window_size: usize,
        minimum_convergence_value: f64,
    ) -> &mut Self {
        self.convergence = Some((window_size, minimum_convergence_value));
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run gradient descent with a per-iteration golden-section line search from
    /// `initial`. `eval(p)` returns `(value, gradient)` of the objective at `p`.
    /// Returns the lowest-value iterate visited.
    ///
    /// Each iteration takes one gradient to find the descent direction, then up
    /// to `maximum_line_search_iterations` **value-only** probes along it. A
    /// closure has no value-only kernel, so those probes compute and discard a
    /// gradient; a caller that has one should call
    /// [`optimize_objective`](Self::optimize_objective) instead.
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        self.run(initial, FnObjective(eval))
    }

    /// [`optimize`](Self::optimize) driven by an explicit [`Objective`], so the
    /// line search's value probes reach [`Objective::value`] rather than the
    /// blanket closure implementation's discard-the-gradient fallback.
    pub fn optimize_objective<O: Objective>(&self, initial: Vec<f64>, eval: O) -> OptimizerResult {
        self.run(initial, eval)
    }

    fn run<O: Objective>(&self, initial: Vec<f64>, mut eval: O) -> OptimizerResult {
        let n = initial.len();
        let ones = vec![1.0; n];
        let scales: &[f64] = match &self.scales {
            Some(s) => {
                assert_eq!(s.len(), n, "scales length must equal parameter count");
                s
            }
            None => &ones,
        };

        let mut monitor = self
            .convergence
            .map(|(window, _)| WindowConvergenceMonitor::new(window));

        let mut p = initial;
        let (mut value, mut grad) = eval.value_and_gradient(&p);
        // ITK's line search sets ReturnBestParametersAndValue = true: keep the
        // lowest-value iterate, correctly paired with its parameters.
        let mut best_value = value;
        let mut best_params = p.clone();
        // The base rate carries forward: the rate found at one iteration seeds
        // the next iteration's bracket (ITK's m_LearningRate persists).
        let mut base_lr = self.learning_rate;
        let mut stop_reason = StopReason::MaxIterations;
        let mut taken = 0usize;

        loop {
            if value < best_value {
                best_value = value;
                best_params.copy_from_slice(&p);
            }

            // Per-iteration value-plateau test, matching ITK's convergence check.
            if let (Some(mon), Some((_, min_cv))) = (monitor.as_mut(), self.convergence) {
                mon.add_energy_value(value);
                if let Some(cv) = mon.convergence_value()
                    && cv <= min_cv
                {
                    stop_reason = StopReason::Converged;
                    break;
                }
            }

            if taken >= self.number_of_iterations {
                // stop_reason is already MaxIterations.
                break;
            }

            // Descent direction d = gradient ⊘ scales; the step and every
            // line-search trial move along −d.
            let d: Vec<f64> = (0..n).map(|k| grad[k] / scales[k]).collect();

            // Golden-section search for the learning rate over
            // [base_lr·lower, base_lr, base_lr·upper]. The trial evaluator needs
            // only the objective value at p − x·d.
            let mut line_search_iterations = 0u32;
            let lr = {
                let p_ref = &p;
                let d_ref = &d;
                let mut line_value = |x: f64| -> f64 {
                    let trial: Vec<f64> = (0..n).map(|k| p_ref[k] - x * d_ref[k]).collect();
                    eval.value(&trial)
                };
                golden_section_search(
                    self.epsilon,
                    self.maximum_line_search_iterations,
                    base_lr * self.lower_limit,
                    base_lr,
                    base_lr * self.upper_limit,
                    None,
                    &mut line_search_iterations,
                    &mut line_value,
                )
            };
            base_lr = lr;

            for k in 0..n {
                p[k] -= lr * d[k];
            }
            taken += 1;

            let (v, g) = eval.value_and_gradient(&p);
            value = v;
            grad = g;
        }

        OptimizerResult {
            parameters: best_params,
            value: best_value,
            iterations: taken,
            stop_reason,
        }
    }
}

/// Conjugate gradient descent with a golden-section line search
/// (`itk::ConjugateGradientLineSearchOptimizerv4`).
///
/// A subclass of the golden-section line search
/// ([`GradientDescentLineSearchOptimizer`]) that replaces the steepest-descent
/// direction with a **conjugate** one: each iteration combines the current
/// (scaled) gradient with the previous search direction, so successive steps do
/// not undo one another and an elongated basin is descended far faster than plain
/// gradient descent can manage. The learning rate along that direction is still
/// chosen by the golden-section line search each iteration.
///
/// The direction is the modified Polak–Ribière update
///
/// ```text
/// γ = 〈g − g_prev, g〉 / 〈g_prev, g_prev〉,   reset to 0 if γ ∉ (0, 5]
/// d ← g + γ · d_prev
/// ```
///
/// where `g` is the scaled gradient. `g_prev` and `d` start at zero, so the first
/// step is plain steepest descent, and the restart (`γ = 0` outside `(0, 5]`)
/// drops stale conjugacy — ITK's guard against a direction that is no longer a
/// descent direction. The step is then `p ← p − learning_rate · d`.
///
/// Configuration, scales, stopping (`number_of_iterations`, convergence
/// monitoring), and best-value return match
/// [`GradientDescentLineSearchOptimizer`].
#[derive(Clone, Debug)]
pub struct ConjugateGradientLineSearchOptimizer {
    learning_rate: f64,
    number_of_iterations: usize,
    scales: Option<Vec<f64>>,
    lower_limit: f64,
    upper_limit: f64,
    epsilon: f64,
    maximum_line_search_iterations: u32,
    /// `(window_size, minimum_convergence_value)` when value-plateau monitoring
    /// is enabled; `None` disables it (the default).
    convergence: Option<(usize, f64)>,
}

impl ConjugateGradientLineSearchOptimizer {
    /// A conjugate-gradient line-search optimizer with the given base learning
    /// rate and iteration cap. The bracket limits (`0` and `5`), line-search
    /// resolution `epsilon` (`0.01`), and maximum line-search recursion (`20`)
    /// default to ITK's values; scales default to all-ones and convergence
    /// monitoring is disabled.
    pub fn new(learning_rate: f64, number_of_iterations: usize) -> Self {
        Self {
            learning_rate,
            number_of_iterations,
            scales: None,
            lower_limit: 0.0,
            upper_limit: 5.0,
            epsilon: 0.01,
            maximum_line_search_iterations: 20,
            convergence: None,
        }
    }

    /// Set the base learning rate (ITK's `m_InitialLearningRate`), which the first
    /// iteration's line search brackets.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set per-parameter scales (length must equal the parameter count).
    pub fn set_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales = Some(scales);
        self
    }

    /// Set the lower and upper bracket limits: the line search adjusts the
    /// learning rate within `[learning_rate · lower, learning_rate · upper]`
    /// (ITK defaults `0` and `5`).
    pub fn set_line_search_limits(&mut self, lower: f64, upper: f64) -> &mut Self {
        self.lower_limit = lower;
        self.upper_limit = upper;
        self
    }

    /// Set the line-search resolution `epsilon` (ITK default `0.01`).
    pub fn set_epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the maximum golden-section recursion depth per iteration (ITK default
    /// `20`).
    pub fn set_maximum_line_search_iterations(&mut self, iterations: u32) -> &mut Self {
        self.maximum_line_search_iterations = iterations;
        self
    }

    /// Enable value-plateau convergence monitoring
    /// (`itk::WindowConvergenceMonitoringFunction`): stop once the windowed
    /// metric value's trend flattens to at or below `minimum_convergence_value`.
    pub fn set_convergence(
        &mut self,
        window_size: usize,
        minimum_convergence_value: f64,
    ) -> &mut Self {
        self.convergence = Some((window_size, minimum_convergence_value));
        self
    }

    /// Configured per-parameter scales, or `None` for all-ones.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Run conjugate-gradient descent with a per-iteration golden-section line
    /// search from `initial`. `eval(p)` returns `(value, gradient)` of the
    /// objective at `p`. Returns the lowest-value iterate visited.
    ///
    /// As with the plain line search, each iteration is one gradient plus a run
    /// of value-only probes along the conjugate direction — see
    /// [`optimize_objective`](Self::optimize_objective).
    ///
    /// Panics if configured scales' length differs from `initial.len()`.
    pub fn optimize<F>(&self, initial: Vec<f64>, eval: F) -> OptimizerResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        self.run(initial, FnObjective(eval))
    }

    /// [`optimize`](Self::optimize) driven by an explicit [`Objective`], so the
    /// line search's value probes reach [`Objective::value`].
    pub fn optimize_objective<O: Objective>(&self, initial: Vec<f64>, eval: O) -> OptimizerResult {
        self.run(initial, eval)
    }

    fn run<O: Objective>(&self, initial: Vec<f64>, mut eval: O) -> OptimizerResult {
        let n = initial.len();
        let ones = vec![1.0; n];
        let scales: &[f64] = match &self.scales {
            Some(s) => {
                assert_eq!(s.len(), n, "scales length must equal parameter count");
                s
            }
            None => &ones,
        };

        let mut monitor = self
            .convergence
            .map(|(window, _)| WindowConvergenceMonitor::new(window));

        let mut p = initial;
        let (mut value, mut grad) = eval.value_and_gradient(&p);
        let mut best_value = value;
        let mut best_params = p.clone();
        let mut base_lr = self.learning_rate;

        // Conjugate-gradient state (itk::ConjugateGradientLineSearchOptimizerv4):
        // the previous scaled gradient and the running conjugate direction, both
        // zero-initialized so the first step is plain steepest descent.
        let mut last_scaled = vec![0.0; n];
        let mut conjugate = vec![0.0; n];

        let mut stop_reason = StopReason::MaxIterations;
        let mut taken = 0usize;

        loop {
            if value < best_value {
                best_value = value;
                best_params.copy_from_slice(&p);
            }

            if let (Some(mon), Some((_, min_cv))) = (monitor.as_mut(), self.convergence) {
                mon.add_energy_value(value);
                if let Some(cv) = mon.convergence_value()
                    && cv <= min_cv
                {
                    stop_reason = StopReason::Converged;
                    break;
                }
            }

            if taken >= self.number_of_iterations {
                // stop_reason is already MaxIterations.
                break;
            }

            // Scaled gradient, then the modified Polak–Ribière conjugate
            // direction: γ = 〈g − g_prev, g〉 / 〈g_prev, g_prev〉, reset to 0
            // outside (0, 5], and d ← g + γ·d_prev.
            let scaled: Vec<f64> = (0..n).map(|k| grad[k] / scales[k]).collect();
            let gamma_denom: f64 = last_scaled.iter().map(|&x| x * x).sum();
            let mut gamma = 0.0;
            if gamma_denom > f64::EPSILON {
                let num: f64 = (0..n)
                    .map(|k| (scaled[k] - last_scaled[k]) * scaled[k])
                    .sum();
                gamma = num / gamma_denom;
            }
            if !(0.0..=5.0).contains(&gamma) {
                gamma = 0.0;
            }
            last_scaled.copy_from_slice(&scaled);
            for k in 0..n {
                conjugate[k] = scaled[k] + gamma * conjugate[k];
            }

            // Golden-section search for the learning rate along the conjugate
            // direction, then step p ← p − lr·d.
            let mut line_search_iterations = 0u32;
            let lr = {
                let p_ref = &p;
                let d_ref = &conjugate;
                let mut line_value = |x: f64| -> f64 {
                    let trial: Vec<f64> = (0..n).map(|k| p_ref[k] - x * d_ref[k]).collect();
                    eval.value(&trial)
                };
                golden_section_search(
                    self.epsilon,
                    self.maximum_line_search_iterations,
                    base_lr * self.lower_limit,
                    base_lr,
                    base_lr * self.upper_limit,
                    None,
                    &mut line_search_iterations,
                    &mut line_value,
                )
            };
            base_lr = lr;

            for k in 0..n {
                p[k] -= lr * conjugate[k];
            }
            taken += 1;

            let (v, g) = eval.value_and_gradient(&p);
            value = v;
            grad = g;
        }

        OptimizerResult {
            parameters: best_params,
            value: best_value,
            iterations: taken,
            stop_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The scalar product decides a branch, and a naive walk gets its SIGN wrong.**
    ///
    /// This is the sharpest pin in the compensated-summation family, because it does not
    /// argue about last bits: it exhibits a gradient pair for which the naive sum of
    /// `g·g_prev` is **−1** and the true sum is **+1**. Upstream compensates this reduction
    /// (`itkRegularStepGradientDescentOptimizerv4.hxx:126-133`) and then *tests its sign*
    /// to decide whether the descent direction reversed and the step must be relaxed. So
    /// the accumulator's error is not a small error in a reported number — it flips a
    /// discrete decision, halving a step that should not have been halved.
    ///
    /// The products are `[1e16, 1, 1, −1e16, −1]`, summed left to right. Naively, both
    /// `1`s vanish into `1e16`'s ulp (which is 2), the `−1e16` then cancels to exactly
    /// zero, and the trailing `−1` lands the sum at **−1 → "reversed, relax"**. Kahan
    /// carries the lost `2` across the cancellation and lands at **+1 → "not reversed"**.
    /// §2.157 recorded this exact branch amplifying a 1e-12 difference ~500× per step;
    /// this test is that mechanism in the small.
    ///
    /// Reverting either `compensated_sum` in the loop to `.sum()` relaxes the step and the
    /// final parameters move half as far — which is what the assertion catches.
    #[test]
    fn the_reversal_test_is_compensated_and_a_naive_walk_flips_the_branch() {
        // Iteration 1's gradient. Iteration 2's is all ones, so the elementwise products
        // at iteration 2 are exactly this vector.
        let g1 = [1.0e16, 1.0, 1.0, -1.0e16, -1.0];
        let g2 = [1.0, 1.0, 1.0, 1.0, 1.0];

        let naive: f64 = g1.iter().zip(g2.iter()).map(|(&a, &b)| a * b).sum();
        let compensated = compensated_sum(g1.iter().zip(g2.iter()).map(|(&a, &b)| a * b));
        assert!(
            naive < 0.0 && compensated > 0.0,
            "the fixture must be one where the two walks disagree on the SIGN, or this \
             pin cannot fail: naive {naive}, compensated {compensated}"
        );

        // Two iterations, unit scales, learning rate 1, relaxation 0.5 (the default).
        let mut opt = RegularStepGradientDescentOptimizer::new(1.0, 1.0e-12, 2);
        opt.set_relaxation_factor(0.5);

        let mut call = 0usize;
        let r = opt.optimize(vec![0.0; 5], |_p| {
            call += 1;
            // eval #1 seeds the loop with g1; every later eval returns g2.
            let g = if call == 1 { g1 } else { g2 };
            (0.0, g.to_vec())
        });

        // Each iteration moves `p` by `(step_length / ‖scaled‖) · scaled`, and with the
        // reversal branch correctly NOT taken the step length stays `1 · learning_rate = 1`.
        // Iteration 1 cannot relax either way (the previous gradient is still zero), so
        // parameter 1 — whose component is `+1` in both gradients — isolates iteration 2's
        // step length: it moves by `g1[1]/‖g1‖` (a negligible 7e-17, as ‖g1‖ ≈ 1.4e16) and
        // then by `g2[1]/‖g2‖ = 1/√5 ≈ 0.447`.
        let magnitude_1 = compensated_sum(g1.iter().map(|g| g * g)).sqrt();
        let magnitude_2 = compensated_sum(g2.iter().map(|g| g * g)).sqrt();
        let unrelaxed = -(g1[1] / magnitude_1) - (g2[1] / magnitude_2);
        // What a naive scalar product produces: the branch fires, relaxation halves, and
        // iteration 2's step is half as long. The two land a factor of two apart, so this
        // pin discriminates by a mile rather than by an ulp.
        let relaxed = -(g1[1] / magnitude_1) - 0.5 * (g2[1] / magnitude_2);
        assert!(
            (unrelaxed - relaxed).abs() > 0.2,
            "the two outcomes must be far apart, or this pin is measuring rounding"
        );

        assert_eq!(r.iterations, 2, "the fixture must take both steps");
        assert!(
            (r.parameters[1] - unrelaxed).abs() < 1e-12 * unrelaxed.abs().max(1.0),
            "the step was relaxed, so the reversal branch fired — the scalar product was \
             summed naively and came out negative. got {}, expected {unrelaxed} (a relaxed \
             step lands at {relaxed})",
            r.parameters[1]
        );
    }

    #[test]
    fn minimizes_a_quadratic_bowl() {
        // f(p) = (p0 − 3)² + (p1 + 2)², minimum at (3, −2).
        let opt = GradientDescentOptimizer::new(0.1, 1000);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8);
    }

    #[test]
    fn scales_balance_anisotropic_conditioning() {
        // f(p) = (p0 − 1)² + 1e6·(p1 − 1)². Without scales a single step size
        // cannot serve both axes; scales = [1, 1e6] restores conditioning.
        let mut opt = GradientDescentOptimizer::new(0.4, 500);
        opt.set_scales(vec![1.0, 1e6]);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let v = (p[0] - 1.0).powi(2) + 1e6 * (p[1] - 1.0).powi(2);
            let g = vec![2.0 * (p[0] - 1.0), 2e6 * (p[1] - 1.0)];
            (v, g)
        });
        assert!((r.parameters[0] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn gradient_descent_stops_on_convergence_monitor() {
        // Finding A parity: fixed-rate gradient descent has no minimum-step stop;
        // its only early stop is the value-plateau monitor, matching
        // `itk::GradientDescentOptimizerv4`. On the bowl f(p) = (p0−3)² + (p1+2)²
        // at learning rate 0.1 under SimpleITK's SetOptimizerAsGradientDescent
        // convergence defaults (window 10, minimum convergence value 1e-6), the
        // run stops Converged at iteration 37 — the count the port's own
        // Finding-A measurement recorded, and the count a min-step stop (which
        // fired at 83) would have pre-empted.
        let mut opt = GradientDescentOptimizer::new(0.1, 100_000);
        opt.set_convergence(10, 1e-6);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert_eq!(r.stop_reason, StopReason::Converged);
        assert_eq!(r.iterations, 37);
    }

    #[test]
    fn regular_step_minimizes_a_quadratic_bowl() {
        // f(p) = (p0 − 3)² + (p1 + 2)², minimum at (3, −2). With the default
        // gradient-magnitude tolerance (1e-4) the run stops at a stationary
        // point: `‖grad‖ = 2·‖p − p*‖ < 1e-4` gives `‖p − p*‖ < 5e-5`.
        let opt = RegularStepGradientDescentOptimizer::new(1.0, 1e-8, 1000);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert!((r.parameters[0] - 3.0).abs() < 5e-5, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 5e-5, "{:?}", r.parameters);
        // Stopped at the stationary point, not by running out of iterations.
        assert_eq!(r.stop_reason, StopReason::GradientConverged);
        assert!(r.iterations < 1000);
    }

    #[test]
    fn regular_step_relaxes_on_overshoot_from_a_large_step() {
        // An initial step length far larger than the basin still converges: the
        // first steps overshoot, the direction reverses, and the step halves
        // until it fits. Without relaxation a fixed 50-unit step would oscillate
        // forever around the minimum at 3. A negligible gradient tolerance forces
        // the step-length tolerance to govern the stop, so precision tracks
        // `minimum_step_length`.
        let mut opt = RegularStepGradientDescentOptimizer::new(50.0, 1e-6, 1000);
        opt.set_gradient_magnitude_tolerance(1e-12);
        let r = opt.optimize(vec![0.0], |p| {
            ((p[0] - 3.0).powi(2), vec![2.0 * (p[0] - 3.0)])
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert_eq!(r.stop_reason, StopReason::StepTooSmall);
    }

    #[test]
    fn regular_step_stops_immediately_at_a_stationary_start() {
        // Starting exactly at the minimum, the gradient magnitude is below
        // tolerance, so it stops without stepping — the behavior that makes a
        // near-converged multi-resolution restart safe (no fixed-rate blow-up).
        let opt = RegularStepGradientDescentOptimizer::new(1.0, 1e-6, 1000);
        let r = opt.optimize(vec![3.0, -2.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            (0.0, g)
        });
        assert_eq!(r.stop_reason, StopReason::GradientConverged);
        assert_eq!(r.iterations, 0);
        assert_eq!(r.parameters, vec![3.0, -2.0]);
    }

    #[test]
    fn regular_step_scales_balance_anisotropic_conditioning() {
        // f(p) = (p0 − 1)² + 1e6·(p1 − 1)². Scales = [1, 1e6] restore the
        // conditioning so both axes reach the minimum at (1, 1).
        let mut opt = RegularStepGradientDescentOptimizer::new(1.0, 1e-7, 2000);
        opt.set_scales(vec![1.0, 1e6]);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let v = (p[0] - 1.0).powi(2) + 1e6 * (p[1] - 1.0).powi(2);
            let g = vec![2.0 * (p[0] - 1.0), 2e6 * (p[1] - 1.0)];
            (v, g)
        });
        assert!((r.parameters[0] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn line_search_minimizes_a_quadratic_bowl() {
        // f(p) = (p0 − 3)² + (p1 + 2)², minimum at (3, −2). The line search picks
        // a near-optimal rate each step, reaching the minimum in far fewer
        // iterations than a fixed under-scaled rate would.
        let opt = GradientDescentLineSearchOptimizer::new(1.0, 100);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8);
    }

    #[test]
    fn line_search_tames_an_overshooting_base_rate() {
        // f(p) = (p − 3)². A fixed learning rate of 1.0 gives the step
        // p ← p − 1·2(p − 3) = −p + 6, which oscillates 0 → 6 → 0 forever. The
        // line search instead shrinks the rate toward the quadratic's optimal
        // 0.5 (which lands exactly on 3), so it converges despite the same base
        // rate that makes plain gradient descent diverge.
        let opt = GradientDescentLineSearchOptimizer::new(1.0, 50);
        let r = opt.optimize(vec![0.0], |p| {
            ((p[0] - 3.0).powi(2), vec![2.0 * (p[0] - 3.0)])
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8);
    }

    #[test]
    fn line_search_scales_balance_anisotropic_conditioning() {
        // f(p) = (p0 − 1)² + 1e6·(p1 − 1)². Scales = [1, 1e6] restore the
        // conditioning so the golden-section rate serves both axes at once.
        let mut opt = GradientDescentLineSearchOptimizer::new(1.0, 500);
        opt.set_scales(vec![1.0, 1e6]);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let v = (p[0] - 1.0).powi(2) + 1e6 * (p[1] - 1.0).powi(2);
            let g = vec![2.0 * (p[0] - 1.0), 2e6 * (p[1] - 1.0)];
            (v, g)
        });
        assert!((r.parameters[0] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn line_search_stops_on_convergence_monitor() {
        // With value-plateau monitoring enabled, the run stops on the windowed
        // convergence check once the metric value flattens near the minimum,
        // mirroring the base GradientDescentOptimizerv4 stop condition. Finding A
        // parity: there is no minimum-step stop to pre-empt it.
        let mut opt = GradientDescentLineSearchOptimizer::new(1.0, 100_000);
        opt.set_convergence(5, 1e-8);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert_eq!(r.stop_reason, StopReason::Converged);
        assert!(r.iterations < 100_000);
        assert!((r.parameters[0] - 3.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn conjugate_gradient_minimizes_a_quadratic_bowl() {
        // f(p) = (p0 − 3)² + (p1 + 2)², minimum at (3, −2).
        let opt = ConjugateGradientLineSearchOptimizer::new(1.0, 100);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert!((r.parameters[0] - 3.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-4, "{:?}", r.parameters);
        assert!(r.value < 1e-8);
    }

    #[test]
    fn conjugate_gradient_descends_an_ill_conditioned_valley_faster_than_line_search() {
        // f(p) = (p0 − 3)² + 50·(p1 + 2)² is an elongated valley (condition ~50)
        // where steepest descent zig-zags. The conjugate direction combines
        // successive gradients so it reaches the minimum in far fewer iterations
        // than the plain golden-section line search on the identical problem.
        // Both stop on the value-plateau monitor (Finding A parity: no min-step);
        // a tight convergence value lets each reach the minimum accurately so the
        // iteration-count gap is a fair comparison at equal precision.
        let f = |p: &[f64]| {
            let v = (p[0] - 3.0).powi(2) + 50.0 * (p[1] + 2.0).powi(2);
            let g = vec![2.0 * (p[0] - 3.0), 100.0 * (p[1] + 2.0)];
            (v, g)
        };
        let mut cg_opt = ConjugateGradientLineSearchOptimizer::new(1.0, 1000);
        cg_opt.set_convergence(10, 1e-10);
        let cg = cg_opt.optimize(vec![0.0, 0.0], f);
        let mut ls_opt = GradientDescentLineSearchOptimizer::new(1.0, 1000);
        ls_opt.set_convergence(10, 1e-10);
        let ls = ls_opt.optimize(vec![0.0, 0.0], f);

        assert!((cg.parameters[0] - 3.0).abs() < 1e-3, "{:?}", cg.parameters);
        assert!((cg.parameters[1] + 2.0).abs() < 1e-3, "{:?}", cg.parameters);
        assert!(
            cg.iterations < ls.iterations,
            "conjugate gradient took {} iterations, line search {}",
            cg.iterations,
            ls.iterations
        );
    }

    #[test]
    fn conjugate_gradient_scales_balance_anisotropic_conditioning() {
        // f(p) = (p0 − 1)² + 1e6·(p1 − 1)². Scales = [1, 1e6] restore the
        // conditioning so the conjugate search serves both axes at once.
        let mut opt = ConjugateGradientLineSearchOptimizer::new(1.0, 500);
        opt.set_scales(vec![1.0, 1e6]);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let v = (p[0] - 1.0).powi(2) + 1e6 * (p[1] - 1.0).powi(2);
            let g = vec![2.0 * (p[0] - 1.0), 2e6 * (p[1] - 1.0)];
            (v, g)
        });
        assert!((r.parameters[0] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] - 1.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    #[test]
    fn conjugate_gradient_stops_on_convergence_monitor() {
        // With value-plateau monitoring enabled, the run stops on the windowed
        // convergence check once the metric value flattens near the minimum.
        // Finding A parity: there is no minimum-step stop to pre-empt it.
        let mut opt = ConjugateGradientLineSearchOptimizer::new(1.0, 100_000);
        opt.set_convergence(5, 1e-8);
        let r = opt.optimize(vec![0.0, 0.0], |p| {
            let g = vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)];
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, g)
        });
        assert_eq!(r.stop_reason, StopReason::Converged);
        assert!(r.iterations < 100_000);
        assert!((r.parameters[0] - 3.0).abs() < 1e-3, "{:?}", r.parameters);
        assert!((r.parameters[1] + 2.0).abs() < 1e-3, "{:?}", r.parameters);
    }

    /// Counts how the line searches split their work between the two
    /// `Objective` methods.
    #[derive(Debug)]
    struct CountingObjective {
        gradients: usize,
        values: usize,
    }

    impl Objective for CountingObjective {
        fn value_and_gradient(&mut self, p: &[f64]) -> (f64, Vec<f64>) {
            self.gradients += 1;
            let v = (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2);
            (v, vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)])
        }

        fn value(&mut self, p: &[f64]) -> f64 {
            self.values += 1;
            (p[0] - 3.0).powi(2) + (p[1] + 2.0).powi(2)
        }
    }

    #[test]
    fn line_search_probes_the_value_only_kernel() {
        // Every golden-section probe must reach `Objective::value`. If the
        // probes went through `value_and_gradient` instead, `values` would be 0
        // and each probe would have paid for a gradient it never reads.
        let opt = GradientDescentLineSearchOptimizer::new(1.0, 5);
        let mut obj = CountingObjective {
            gradients: 0,
            values: 0,
        };
        let r = opt.optimize_objective(vec![0.0, 0.0], &mut obj);
        assert!(obj.values > obj.gradients, "{obj:?}");
        // One gradient per iteration (plus the initial evaluation), no more.
        assert!(obj.gradients <= r.iterations + 2, "{obj:?}");

        let opt = ConjugateGradientLineSearchOptimizer::new(1.0, 5);
        let mut obj = CountingObjective {
            gradients: 0,
            values: 0,
        };
        let r = opt.optimize_objective(vec![0.0, 0.0], &mut obj);
        assert!(obj.values > obj.gradients, "{obj:?}");
        assert!(obj.gradients <= r.iterations + 2, "{obj:?}");
    }
}
