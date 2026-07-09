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
//!
//! # Local-support transforms (`DisplacementFieldTransform`)
//!
//! A dense transform's parameter count is small (a handful to a few dozen), so
//! probing every parameter and caching a `dim × nparams` Jacobian per sample is
//! cheap. A [`DisplacementFieldTransform`](sitk_transform::DisplacementFieldTransform)
//! has one parameter vector *per pixel*: for a modest 3-D volume `nparams` is in
//! the millions, so the dense probe (`O(nparams)` perturbations, each scanning
//! `O(nsamples)` points, each needing an `O(nparams)`-wide Jacobian row) is
//! intractable — exactly the dense-Jacobian problem the Mattes metric's
//! `evaluate_local_support` (`mattes.rs`) solves for the metric derivative.
//!
//! ITK keys this off `HasLocalSupport()`/`GetTransformCategory() ==
//! DisplacementField` (this crate's [`ParametricTransform::has_local_support`])
//! and switches to a fundamentally different algorithm in
//! `itkRegistrationParameterScalesFromShiftBase.hxx`:
//!
//! - **`EstimateScales`** (lines 34–119) sizes its output to
//!   `GetNumberOfLocalParameters()`, *not* the full parameter count (line 43:
//!   `parameterScales.SetSize(numLocalPara)`), and probes only those
//!   `numLocalPara` axes at a single representative ("central") parameter block
//!   (lines 54–86) instead of every one of the `nparams` parameters. For
//!   [`DisplacementFieldTransform`](sitk_transform::DisplacementFieldTransform)
//!   the local Jacobian at a grid-aligned sample is always the `dim × dim`
//!   identity (`sparse_jacobian_wrt_parameters` in `sitk-transform/src/displacement.rs`
//!   — the interpolation weight of a pixel at its own grid index is exactly 1,
//!   every other weight 0), so every probe shifts by exactly `δ` and the
//!   ITK formula (lines 102–116: `scaleᵢ = maxShiftᵢ² / δ²`) reduces to
//!   `scaleᵢ = 1`: "every local parameter is a displacement in physical units
//!   already and needs no rebalancing."
//! - **`EstimateStepScale`** (lines 124–155) short-circuits *before* the
//!   dense-path linearization: `if (TransformHasLocalSupportForScalesEstimation())
//!   return this->ComputeMaximumVoxelShift(step);` (lines 132–135) — the full,
//!   unnormalized `step` vector is applied as-is and the shift is the max over
//!   *all* sample points of `‖T_{p+step}(x) − T_p(x)‖`. Local support means each
//!   sample's shift depends only on its own owning parameter block, so this is
//!   `maxₓ ‖local_jacobian(x) · step[offset(x)..offset(x)+numLocal]‖` — no
//!   `nparams`-wide structure, just one small dot product per sample.
//!
//! **A wrinkle this crate does not port.** ITK's optimizer
//! (`itkObjectToObjectOptimizerBase.h`, `m_Scales` doc comment: "Size is
//! expected to be == metric->GetNumberOfLocalParameters()"; enforced in
//! `itkObjectToObjectOptimizerBase.cxx`'s scales-validation block) stores the
//! *short* `numberOfLocalParameters()`-length scales array and broadcasts it
//! itself when scaling each pixel's local gradient block. This crate's
//! optimizers (`optimizer.rs`) do not have that broadcast: they index `scales`
//! 1:1 against the full gradient (`assert_eq!(s.len(), n)`, `n =
//! initial.len()`). So [`PhysicalShiftScales::estimate_scales`] returns the
//! ITK-derived per-local-parameter scale *tiled* across every parameter block —
//! a full `nparams`-length vector whose entries are numerically identical to
//! what a broadcast-aware optimizer would read out of ITK's short array (every
//! entry is the same value, so indexing by absolute position or by
//! local-block-relative position gives the same number). Giving
//! `optimizer.rs` genuine broadcast support (so this can return the literal
//! `numberOfLocalParameters()`-length array) is out of this module's scope.
//!
//! [`ParametricTransform::has_local_support`]: sitk_transform::ParametricTransform::has_local_support

use sitk_transform::ParametricTransform;

use crate::metric::local_support_block;

/// ITK's `m_SmallParameterVariation` default.
const SMALL_PARAMETER_VARIATION: f64 = 0.01;

/// The transform-Jacobian data [`PhysicalShiftScales`] holds, chosen once at
/// construction by [`ParametricTransform::has_local_support`]. This is the
/// structural fix for the local-support fast path: a local-support transform's
/// state (`Local`) has no field that could hold a `dim × nparams` array, so a
/// future method cannot silently fall back to the dense probe it would be
/// intractable to build.
enum JacobianStore {
    /// Global transform: the per-sample `dim × nparams` Jacobian, evaluated
    /// once and concatenated row-major over samples (see
    /// [`PhysicalShiftScales::new`]).
    Dense(Vec<f64>),
    /// Local-support transform (`DisplacementFieldTransform`): one
    /// `(parameter-block offset, dim × numberOfLocalParameters local Jacobian)`
    /// pair per sample that falls inside the transform's domain. Samples
    /// outside are simply absent — ITK's local support means they contribute
    /// no shift no matter what the parameters do. Size is `O(nsamples)`,
    /// independent of `nparams`.
    Local {
        num_local: usize,
        blocks: Vec<(usize, Vec<f64>)>,
    },
}

/// Physical-shift scale/learning-rate estimator. Holds the transform Jacobians
/// at the fixed sample points (evaluated once) and the maximum physical step.
pub struct PhysicalShiftScales {
    dim: usize,
    nparams: usize,
    n: usize,
    jac: JacobianStore,
    /// δ, the small parameter variation used to probe shifts.
    small_variation: f64,
    /// Maximum physical step size (ITK: minimum fixed/virtual spacing).
    max_step_size: f64,
}

impl PhysicalShiftScales {
    /// Build from the fixed sample points (row-major `N × dim`), the transform
    /// (for its parameter Jacobian), and the maximum physical step size (the
    /// minimum fixed-image spacing).
    ///
    /// For a [local-support](ParametricTransform::has_local_support) transform
    /// this never builds a `dim × nparams` structure (see the [module
    /// docs](self)): it caches only each sample's small local Jacobian block.
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

        let jac = if transform.has_local_support() {
            let num_local = transform.number_of_local_parameters();
            let mut blocks = Vec::new();
            for s in 0..n {
                let p = &sample_points[s * dim..(s + 1) * dim];
                if let Some(block) = local_support_block(transform, p) {
                    blocks.push(block);
                }
            }
            JacobianStore::Local { num_local, blocks }
        } else {
            let stride = dim * nparams;
            let mut jacobians = vec![0.0; n * stride];
            for s in 0..n {
                let p = &sample_points[s * dim..(s + 1) * dim];
                let j = transform.jacobian_wrt_parameters(p);
                jacobians[s * stride..(s + 1) * stride].copy_from_slice(&j);
            }
            JacobianStore::Dense(jacobians)
        };

        Self {
            dim,
            nparams,
            n,
            jac,
            small_variation: SMALL_PARAMETER_VARIATION,
            max_step_size,
        }
    }

    /// `maxₓ ‖J(x)·delta‖` over the sample points, for the dense (global
    /// transform) Jacobian. `delta` has length `nparams`.
    fn max_voxel_shift_dense(&self, jacobians: &[f64], delta: &[f64]) -> f64 {
        let (dim, np) = (self.dim, self.nparams);
        let stride = dim * np;
        let mut max_sq = 0.0f64;
        for s in 0..self.n {
            let jac = &jacobians[s * stride..(s + 1) * stride];
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

    /// Estimate per-parameter scales (ITK `EstimateScales`).
    pub fn estimate_scales(&self) -> Vec<f64> {
        match &self.jac {
            JacobianStore::Dense(jacobians) => self.estimate_dense_scales(jacobians),
            JacobianStore::Local { num_local, blocks } => {
                self.estimate_local_scales(*num_local, blocks)
            }
        }
    }

    /// `EstimateScales`, global transform: probe every parameter directly.
    fn estimate_dense_scales(&self, jacobians: &[f64]) -> Vec<f64> {
        let np = self.nparams;
        let d = self.small_variation;
        let eps = f64::EPSILON;

        let mut shifts = vec![0.0; np];
        let mut min_nonzero: Option<f64> = None;
        for i in 0..np {
            let mut delta = vec![0.0; np];
            delta[i] = d;
            let ms = self.max_voxel_shift_dense(jacobians, &delta);
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

    /// `EstimateScales`, local-support transform (ITK
    /// `itkRegistrationParameterScalesFromShiftBase.hxx` lines 34–119, the
    /// non-BSpline branch at lines 100–117): probe only the
    /// `numberOfLocalParameters()` axes of a single representative block —
    /// every grid-aligned sample of a [`DisplacementFieldTransform`] carries
    /// the identical local Jacobian (see the [module docs](self)), so any one
    /// block is exact, not an approximation. The result is broadcast across
    /// every parameter block (see the [module-level wrinkle note](self)).
    fn estimate_local_scales(&self, num_local: usize, blocks: &[(usize, Vec<f64>)]) -> Vec<f64> {
        let d = self.small_variation;
        let eps = f64::EPSILON;

        let mut local_scale = vec![1.0; num_local];
        if let Some((_, jac)) = blocks.first() {
            let dim = self.dim;
            let mut shifts = vec![0.0; num_local];
            let mut min_nonzero: Option<f64> = None;
            for i in 0..num_local {
                let mut sq = 0.0;
                for r in 0..dim {
                    let v = jac[r * num_local + i];
                    sq += v * v;
                }
                let shift = d * sq.sqrt();
                shifts[i] = shift;
                if shift > eps {
                    min_nonzero = Some(min_nonzero.map_or(shift, |m| m.min(shift)));
                }
            }
            if let Some(min_nz) = min_nonzero {
                let inv_d2 = 1.0 / (d * d);
                for i in 0..num_local {
                    let base = if shifts[i] <= eps {
                        min_nz * min_nz
                    } else {
                        shifts[i] * shifts[i]
                    };
                    local_scale[i] = base * inv_d2;
                }
            }
            // else: nothing moves a voxel; `local_scale` keeps its unit fallback.
        }

        let mut out = vec![1.0; self.nparams];
        for chunk in out.chunks_mut(num_local) {
            chunk.copy_from_slice(&local_scale[..chunk.len()]);
        }
        out
    }

    /// Estimate the physical shift per unit `step` (ITK `EstimateStepScale`).
    /// `step` has length `nparams`.
    pub fn estimate_step_scale(&self, step: &[f64]) -> f64 {
        match &self.jac {
            JacobianStore::Dense(jacobians) => self.dense_step_scale(jacobians, step),
            JacobianStore::Local { num_local, blocks } => {
                self.local_step_scale(*num_local, blocks, step)
            }
        }
    }

    /// `EstimateStepScale`, global transform: a linear approximation around a
    /// small scaled-down step, since the shift is nonlinear in `step` in
    /// general (only exactly linear for the transforms this crate probes with
    /// a parameter-independent Jacobian).
    fn dense_step_scale(&self, jacobians: &[f64], step: &[f64]) -> f64 {
        let max_step = step.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        if max_step <= f64::EPSILON {
            return 0.0;
        }
        let factor = self.small_variation / max_step;
        let small: Vec<f64> = step.iter().map(|&v| v * factor).collect();
        self.max_voxel_shift_dense(jacobians, &small) / factor
    }

    /// `EstimateStepScale`, local-support transform (ITK
    /// `itkRegistrationParameterScalesFromShiftBase.hxx` lines 132–135): no
    /// linearization — `step` is applied as-is, and since a local-support
    /// transform's shift at a sample depends only on that sample's own
    /// parameter block, this is `maxₓ ‖local_jacobian(x) ·
    /// step[offset(x)..offset(x)+numLocal]‖` computed from the cached blocks,
    /// never touching the `nparams`-length rest of `step`.
    fn local_step_scale(
        &self,
        num_local: usize,
        blocks: &[(usize, Vec<f64>)],
        step: &[f64],
    ) -> f64 {
        let dim = self.dim;
        let mut max_sq = 0.0f64;
        for (offset, jac) in blocks {
            let local_step = &step[*offset..*offset + num_local];
            let mut sq = 0.0;
            for r in 0..dim {
                let row = &jac[r * num_local..(r + 1) * num_local];
                let dot: f64 = row
                    .iter()
                    .zip(local_step.iter())
                    .map(|(&j, &x)| j * x)
                    .sum();
                sq += dot * dot;
            }
            if sq > max_sq {
                max_sq = sq;
            }
        }
        max_sq.sqrt()
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
    use sitk_transform::{AffineTransform, DisplacementFieldTransform, TranslationTransform};

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

    fn identity_dir(dim: usize) -> Vec<f64> {
        let mut d = vec![0.0; dim * dim];
        for i in 0..dim {
            d[i * dim + i] = 1.0;
        }
        d
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

    #[test]
    fn displacement_field_takes_the_local_support_path() {
        let field =
            DisplacementFieldTransform::new(2, &[6, 6], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let (pts, dim) = grid_points(6);
        let est = PhysicalShiftScales::new(&pts, dim, &field, 1.0);
        assert!(matches!(est.jac, JacobianStore::Local { .. }));
    }

    #[test]
    fn displacement_field_scales_are_unit_and_broadcast_to_every_parameter() {
        // ITK's own EstimateScales returns a numberOfLocalParameters()-length
        // unit array for a displacement field (see the module docs); this
        // crate's optimizer has no broadcast support, so estimate_scales
        // tiles that unit value across the full nparams-length vector — every
        // entry is 1.0, matching what a broadcast-aware optimizer would read.
        let field =
            DisplacementFieldTransform::new(2, &[6, 6], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let (pts, dim) = grid_points(6);
        let nparams = field.number_of_parameters();
        let est = PhysicalShiftScales::new(&pts, dim, &field, 1.0);
        let scales = est.estimate_scales();
        assert_eq!(scales.len(), nparams);
        for (i, &s) in scales.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-9, "scales[{i}] = {s} != 1");
        }
    }

    #[test]
    fn displacement_field_step_scale_matches_itks_local_formula() {
        // ITK's EstimateStepScale for a local-support transform is exactly
        // ComputeMaximumVoxelShift(step) — no linearization — which for a
        // displacement field (identity local Jacobian, disjoint per-pixel
        // support) is the max over pixels of that pixel's own step-block norm.
        let field =
            DisplacementFieldTransform::new(2, &[4, 4], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let (pts, dim) = grid_points(4);
        let est = PhysicalShiftScales::new(&pts, dim, &field, 1.0);

        let mut step = vec![0.0; field.number_of_parameters()];
        // Pixel (1,1): raster 1 + 1*4 = 5, offset 10. Step (3,4), norm 5.
        step[10] = 3.0;
        step[11] = 4.0;
        // Pixel (2,2): raster 2 + 2*4 = 10, offset 20. Step (1,0), norm 1.
        step[20] = 1.0;
        step[21] = 0.0;

        let step_scale = est.estimate_step_scale(&step);
        assert!(
            (step_scale - 5.0).abs() < 1e-12,
            "step scale {step_scale} != 5 (max pixel-block norm)"
        );
    }

    #[test]
    fn local_support_construction_avoids_dense_per_parameter_allocation() {
        // A field with hundreds of thousands of parameters, but only a
        // handful of fixed samples (independent of the field's own size, as
        // in real registration where sampling can subsample the domain). The
        // old dense algorithm would have allocated `n * dim * nparams` floats
        // here; the fix must allocate only `O(n)` — one small local-Jacobian
        // block per sample, never touching `nparams`.
        let size = [64usize, 64, 64];
        let field = DisplacementFieldTransform::new(
            3,
            &size,
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &identity_dir(3),
        )
        .unwrap();
        let nparams = field.number_of_parameters();
        assert!(nparams > 700_000, "test needs a large nparams: {nparams}");

        let sample_points = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 20.0, 30.0];
        let n = sample_points.len() / 3;

        let est = PhysicalShiftScales::new(&sample_points, 3, &field, 1.0);
        match &est.jac {
            JacobianStore::Local { num_local, blocks } => {
                assert_eq!(*num_local, 3);
                assert!(
                    blocks.len() <= n,
                    "blocks {} exceeds sample count {n}",
                    blocks.len()
                );
                for (_, jac) in blocks {
                    assert_eq!(
                        jac.len(),
                        3 * num_local,
                        "local block must be dim*num_local, not proportional to nparams"
                    );
                }
            }
            JacobianStore::Dense(_) => {
                panic!("displacement field must take the local-support path")
            }
        }
    }
}
