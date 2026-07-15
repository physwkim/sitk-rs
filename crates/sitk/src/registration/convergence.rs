//! Value-plateau convergence monitoring
//! (`itk::Function::WindowConvergenceMonitoringFunction`).
//!
//! A gradient-descent run whose step size does *not* shrink toward the optimum
//! — ITK's estimate-learning-rate-at-each-iteration mode holds every step at
//! about one voxel — never triggers a small-step stop; it oscillates around the
//! minimum forever. ITK stops it instead by watching the recent metric values:
//! when their trend flattens, the optimization has converged.
//!
//! The monitor keeps a sliding window of the last `window_size` metric values
//! and a running total of their magnitudes. Each query fits a straight line to
//! the window (positions `xₙ = n/(W−1)`, values `eₙ / totalEnergy`) and returns
//! the negated slope: large while the energy is still falling, ~0 once it
//! plateaus. The optimizer stops when it drops to or below its minimum
//! convergence value.
//!
//! ITK performs the line fit with a `BSplineScatteredDataPointSetToImageFilter`
//! of spline order 1 and 2 control points. That is its single-level multilevel
//! B-spline approximation (Lee–Wolberg–Shin) on the linear basis
//! `B₀(x)=1−x`, `B₁(x)=x`, which reduces to the closed form reproduced here:
//! per data point `n`, with `aₙ=1−xₙ`, `bₙ=xₙ`, `Dₙ=aₙ²+bₙ²`, the two control
//! points are
//!
//! ```text
//! c₀ = (Σ aₙ³·yₙ/Dₙ) / (Σ aₙ²),   c₁ = (Σ bₙ³·yₙ/Dₙ) / (Σ bₙ²)
//! ```
//!
//! the fitted line's slope is `c₁ − c₀`, and the convergence value is
//! `−(c₁ − c₀) = c₀ − c₁` — matching
//! `EvaluateGradientAtParametricPoint` over the single span `[0, 1]`.

use std::collections::VecDeque;

/// Sliding-window convergence monitor over recent metric (energy) values.
pub struct WindowConvergenceMonitor {
    window_size: usize,
    energy_values: VecDeque<f64>,
    total_energy: f64,
}

impl WindowConvergenceMonitor {
    /// A monitor over the last `window_size` values. `window_size` must be at
    /// least 2 (the line fit needs two distinct positions); ITK's default is 50
    /// and SimpleITK's `ImageRegistrationMethod` default is 10.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size >= 2, "convergence window size must be >= 2");
        Self {
            window_size,
            energy_values: VecDeque::with_capacity(window_size),
            total_energy: 0.0,
        }
    }

    /// Record one metric value, sliding the window and accumulating the running
    /// total energy (`+= |value|`, over all values ever added, as ITK does).
    pub fn add_energy_value(&mut self, value: f64) {
        self.energy_values.push_back(value);
        if self.energy_values.len() > self.window_size {
            self.energy_values.pop_front();
        }
        self.total_energy += value.abs();
    }

    /// The convergence value, or `None` while the window is not yet full (ITK
    /// returns `+∞` there — "do not stop"). Small (≤ the optimizer's minimum
    /// convergence value) means the metric has plateaued.
    pub fn convergence_value(&self) -> Option<f64> {
        if self.energy_values.len() < self.window_size {
            return None;
        }
        // All-zero energy is a perfect match: converged.
        if self.total_energy == 0.0 {
            return Some(0.0);
        }

        let w = self.window_size as f64;
        let (mut omega0, mut omega1) = (0.0f64, 0.0f64);
        let (mut delta0, mut delta1) = (0.0f64, 0.0f64);
        for (n, &e) in self.energy_values.iter().enumerate() {
            let x = n as f64 / (w - 1.0);
            let a = 1.0 - x;
            let b = x;
            let y = e / self.total_energy;
            let d = a * a + b * b;
            omega0 += a * a;
            omega1 += b * b;
            delta0 += a * a * a * y / d;
            delta1 += b * b * b * y / d;
        }
        let c0 = if omega0 != 0.0 { delta0 / omega0 } else { 0.0 };
        let c1 = if omega1 != 0.0 { delta1 / omega1 } else { 0.0 };
        // Fitted-line slope is c1 - c0; ITK's convergence value is its negation.
        Some(c0 - c1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_full_window_yields_no_value() {
        let mut m = WindowConvergenceMonitor::new(5);
        for _ in 0..4 {
            m.add_energy_value(1.0);
        }
        assert!(m.convergence_value().is_none());
        m.add_energy_value(1.0);
        assert!(m.convergence_value().is_some());
    }

    #[test]
    fn flat_energy_converges_to_zero() {
        // A constant window has zero slope -> convergence value 0.
        let mut m = WindowConvergenceMonitor::new(10);
        for _ in 0..10 {
            m.add_energy_value(3.5);
        }
        let cv = m.convergence_value().unwrap();
        assert!(cv.abs() < 1e-12, "flat energy convergence value {cv} != 0");
    }

    #[test]
    fn steadily_decreasing_energy_is_positive_and_above_flat() {
        // A clearly-decreasing window has a positive convergence value, far
        // larger than a flattened tail — the property the stop test relies on.
        let mut decreasing = WindowConvergenceMonitor::new(10);
        for n in 0..10 {
            decreasing.add_energy_value(10.0 - n as f64);
        }
        let cv_dec = decreasing.convergence_value().unwrap();
        assert!(
            cv_dec > 1e-3,
            "decreasing convergence value {cv_dec} not large"
        );

        let mut plateau = WindowConvergenceMonitor::new(10);
        for _ in 0..20 {
            plateau.add_energy_value(1.0);
        }
        let cv_flat = plateau.convergence_value().unwrap();
        assert!(cv_dec > cv_flat.abs() * 100.0);
    }
}
