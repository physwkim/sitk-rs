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
//! `number_of_iterations`, early when the scaled step is below
//! `min_step_tolerance`, or — when convergence monitoring is enabled — when the
//! metric value plateaus (see [`crate::convergence`]).

use crate::convergence::WindowConvergenceMonitor;

/// Why the optimizer stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Hit `number_of_iterations`.
    MaxIterations,
    /// The scaled step length fell below `min_step_tolerance`.
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
    /// last iterate; [`LBFGSBOptimizer`](crate::LBFGSBOptimizer) returns the
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
    min_step_tolerance: f64,
    /// `(window_size, minimum_convergence_value)` when value-plateau monitoring
    /// is enabled; `None` disables it (the default).
    convergence: Option<(usize, f64)>,
}

impl GradientDescentOptimizer {
    /// A gradient-descent optimizer with the given step size and iteration cap.
    /// Scales default to all-ones, the min-step tolerance to `1e-8`, and
    /// convergence monitoring is disabled.
    pub fn new(learning_rate: f64, number_of_iterations: usize) -> Self {
        Self {
            learning_rate,
            number_of_iterations,
            scales: None,
            min_step_tolerance: 1e-8,
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

    /// Set the minimum scaled-step length below which iteration stops early.
    pub fn set_min_step_tolerance(&mut self, tol: f64) -> &mut Self {
        self.min_step_tolerance = tol;
        self
    }

    /// Enable value-plateau convergence monitoring
    /// (`itk::WindowConvergenceMonitoringFunction`): stop once the windowed
    /// metric value's trend flattens to at or below `minimum_convergence_value`.
    /// Required for a non-shrinking step schedule (learning-rate estimation at
    /// each iteration); fixed-rate runs converge via the min-step tolerance
    /// instead.
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
                if let Some(cv) = mon.convergence_value() {
                    if cv <= min_cv {
                        stop_reason = StopReason::Converged;
                        break;
                    }
                }
            }

            if taken >= self.number_of_iterations {
                // stop_reason is already MaxIterations.
                break;
            }

            let lr = learning_rate_of(&grad);
            let mut step_sq = 0.0;
            for k in 0..n {
                let step = lr * grad[k] / scales[k];
                p[k] -= step;
                step_sq += step * step;
            }
            taken += 1;

            let (v, g) = eval(&p);
            value = v;
            grad = g;

            if step_sq.sqrt() < self.min_step_tolerance {
                stop_reason = StopReason::StepTooSmall;
                break;
            }
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
            let gradient_magnitude = scaled.iter().map(|g| g * g).sum::<f64>().sqrt();

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
            let scalar_product: f64 = scaled
                .iter()
                .zip(previous_scaled.iter())
                .map(|(&a, &b)| a * b)
                .sum();
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn stops_early_when_step_is_tiny() {
        let mut opt = GradientDescentOptimizer::new(0.1, 100_000);
        opt.set_min_step_tolerance(1e-6);
        let r = opt.optimize(vec![0.0], |p| (p[0] * p[0], vec![2.0 * p[0]]));
        assert_eq!(r.stop_reason, StopReason::StepTooSmall);
        assert!(r.iterations < 100_000);
        assert!(r.parameters[0].abs() < 1e-5);
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
}
