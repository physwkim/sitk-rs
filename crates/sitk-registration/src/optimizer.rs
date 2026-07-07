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
}

/// Outcome of an optimization run.
#[derive(Clone, Debug)]
pub struct OptimizerResult {
    /// Best parameters found (the last iterate).
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
}
