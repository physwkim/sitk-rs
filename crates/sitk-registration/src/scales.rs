//! Automatic optimizer parameter-scale and learning-rate estimation from
//! physical shift (`itk::RegistrationParameterScalesFromPhysicalShift` +
//! `itk::GradientDescentOptimizerv4`'s learning-rate estimation).
//!
//! A gradient-descent step `p ← p − lr·(grad ⊘ scales)` needs two things a user
//! should not have to hand-tune: **scales** that make a unit change in each
//! parameter produce a comparable physical shift (so matrix and translation
//! parameters are optimized together), and a **learning rate** that bounds the
//! first step to about one voxel. ITK derives both from how far a parameter
//! change moves the sampled points in physical space.
//!
//! Formulas (verified against the ITK v6 source
//! `itkRegistrationParameterScalesFromShiftBase.hxx` /
//! `itkRegistrationParameterScalesEstimator.hxx` /
//! `itkGradientDescentOptimizerv4.hxx`):
//!
//! - `maxVoxelShift(Δ) = maxₓ ‖T_{p+Δ}(x) − T_p(x)‖` over the sample points.
//!   For the linear transforms registration optimizes this is exactly
//!   `maxₓ ‖J(x)·Δ‖`, and `J` does not depend on `p`, so the Jacobians are
//!   evaluated once at construction.
//! - `scaleᵢ = (maxVoxelShift(δ·eᵢ) / δ)²`, `δ = 0.01`; a parameter that moves
//!   no voxel falls back to `(minNonZeroShift / δ)²`.
//! - `stepScale(step) = maxVoxelShift(step·factor) / factor`,
//!   `factor = δ / max|step|`.
//! - `learningRate = maxStepSize / stepScale(scaledGradient)`, or `1.0` when the
//!   step scale is ~0. `maxStepSize` is the minimum fixed-image spacing.

use sitk_transform::ParametricTransform;

/// ITK's `m_SmallParameterVariation` default.
const SMALL_PARAMETER_VARIATION: f64 = 0.01;

/// Physical-shift scale/learning-rate estimator. Holds the transform Jacobians
/// at the fixed sample points (evaluated once) and the maximum physical step.
pub struct PhysicalShiftScales {
    dim: usize,
    nparams: usize,
    n: usize,
    /// Per sample, row-major `dim × nparams`; concatenated over samples.
    jacobians: Vec<f64>,
    /// δ, the small parameter variation used to probe shifts.
    small_variation: f64,
    /// Maximum physical step size (ITK: minimum fixed/virtual spacing).
    max_step_size: f64,
}

impl PhysicalShiftScales {
    /// Build from the fixed sample points (row-major `N × dim`), the transform
    /// (for its parameter Jacobian), and the maximum physical step size (the
    /// minimum fixed-image spacing).
    pub fn new(
        sample_points: &[f64],
        dim: usize,
        transform: &dyn ParametricTransform,
        max_step_size: f64,
    ) -> Self {
        let nparams = transform.number_of_parameters();
        let n = if dim == 0 {
            0
        } else {
            sample_points.len() / dim
        };
        let stride = dim * nparams;
        let mut jacobians = vec![0.0; n * stride];
        for s in 0..n {
            let p = &sample_points[s * dim..(s + 1) * dim];
            let j = transform.jacobian_wrt_parameters(p);
            jacobians[s * stride..(s + 1) * stride].copy_from_slice(&j);
        }
        Self {
            dim,
            nparams,
            n,
            jacobians,
            small_variation: SMALL_PARAMETER_VARIATION,
            max_step_size,
        }
    }

    /// `maxₓ ‖J(x)·delta‖` over the sample points. `delta` has length `nparams`.
    fn max_voxel_shift(&self, delta: &[f64]) -> f64 {
        let (dim, np) = (self.dim, self.nparams);
        let stride = dim * np;
        let mut max_sq = 0.0f64;
        for s in 0..self.n {
            let jac = &self.jacobians[s * stride..(s + 1) * stride];
            let mut sq = 0.0;
            for d in 0..dim {
                let row = &jac[d * np..(d + 1) * np];
                let dot: f64 = row.iter().zip(delta.iter()).map(|(&j, &x)| j * x).sum();
                sq += dot * dot;
            }
            if sq > max_sq {
                max_sq = sq;
            }
        }
        max_sq.sqrt()
    }

    /// Estimate per-parameter scales (ITK `EstimateScales`, global transform).
    pub fn estimate_scales(&self) -> Vec<f64> {
        let np = self.nparams;
        let d = self.small_variation;
        let eps = f64::EPSILON;

        let mut shifts = vec![0.0; np];
        let mut min_nonzero: Option<f64> = None;
        for i in 0..np {
            let mut delta = vec![0.0; np];
            delta[i] = d;
            let ms = self.max_voxel_shift(&delta);
            shifts[i] = ms;
            if ms > eps {
                min_nonzero = Some(min_nonzero.map_or(ms, |m| m.min(ms)));
            }
        }

        let Some(min_nz) = min_nonzero else {
            // No parameter moves a voxel; unit scales avoid a divide-by-zero.
            return vec![1.0; np];
        };

        let inv_d2 = 1.0 / (d * d);
        shifts
            .iter()
            .map(|&ms| {
                let base = if ms <= eps { min_nz * min_nz } else { ms * ms };
                base * inv_d2
            })
            .collect()
    }

    /// Estimate the physical shift per unit `step` (ITK `EstimateStepScale`,
    /// global transform).
    pub fn estimate_step_scale(&self, step: &[f64]) -> f64 {
        let max_step = step.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        if max_step <= f64::EPSILON {
            return 0.0;
        }
        let factor = self.small_variation / max_step;
        let small: Vec<f64> = step.iter().map(|&v| v * factor).collect();
        self.max_voxel_shift(&small) / factor
    }

    /// Estimate the learning rate for a step along `scaled_gradient`
    /// (`gradient ⊘ scales`): the rate that moves samples by at most
    /// `max_step_size`. Returns `1.0` when the step scale is ~0.
    pub fn estimate_learning_rate(&self, scaled_gradient: &[f64]) -> f64 {
        let step_scale = self.estimate_step_scale(scaled_gradient);
        if step_scale <= f64::EPSILON {
            1.0
        } else {
            self.max_step_size / step_scale
        }
    }

    /// The maximum physical step size (minimum fixed-image spacing).
    pub fn max_step_size(&self) -> f64 {
        self.max_step_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::{AffineTransform, TranslationTransform};

    // Sample points on a 41-index axis (0..=40) centred at 20, so max |x−center|
    // = 20 in each dimension.
    fn grid_points(n: usize) -> (Vec<f64>, usize) {
        let mut pts = Vec::new();
        for y in 0..n {
            for x in 0..n {
                pts.push(x as f64);
                pts.push(y as f64);
            }
        }
        (pts, 2)
    }

    #[test]
    fn translation_scales_are_unit() {
        let (pts, dim) = grid_points(41);
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let est = PhysicalShiftScales::new(&pts, dim, &t, 1.0);
        let scales = est.estimate_scales();
        assert_eq!(scales.len(), 2);
        for s in scales {
            assert!((s - 1.0).abs() < 1e-9, "translation scale {s} != 1");
        }
    }

    #[test]
    fn affine_matrix_scales_are_squared_extent() {
        // Center (20,20); max |x−c| = 20, so matrix-param scale = 20² = 400,
        // translation-param scale = 1.
        let (pts, dim) = grid_points(41);
        let a = AffineTransform::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![20.0, 20.0],
        );
        let est = PhysicalShiftScales::new(&pts, dim, &a, 1.0);
        let scales = est.estimate_scales();
        assert_eq!(scales.len(), 6);
        for s in &scales[0..4] {
            assert!((s - 400.0).abs() < 1e-6, "matrix scale {s} != 400");
        }
        assert!((scales[4] - 1.0).abs() < 1e-9);
        assert!((scales[5] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn learning_rate_bounds_translation_step_to_one_voxel() {
        // For a translation, scales are 1, so stepScale(grad) = ‖grad‖ and
        // lr = maxStepSize / ‖grad‖. With maxStepSize = 1 and grad = (3,4),
        // lr = 1/5, so the first step ‖lr·grad‖ = 1 voxel.
        let (pts, dim) = grid_points(41);
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let est = PhysicalShiftScales::new(&pts, dim, &t, 1.0);
        let grad = [3.0, 4.0];
        let lr = est.estimate_learning_rate(&grad);
        assert!((lr - 0.2).abs() < 1e-9, "lr {lr} != 0.2");
        let step = (lr * grad[0]).hypot(lr * grad[1]);
        assert!(
            (step - 1.0).abs() < 1e-9,
            "first-step shift {step} != 1 voxel"
        );
    }
}
