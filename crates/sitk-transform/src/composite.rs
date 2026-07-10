//! `itk::CompositeTransform`: an ordered stack of transforms composed
//! together.
//!
//! Transforms are added with [`CompositeTransform::add_transform`], which
//! pushes to the back of the queue (`itkCompositeTransform.h:35-37`,
//! `PushBackTransform`). [`transform_point`] then applies them in **reverse**
//! queue order — the most-recently-added transform runs first, on the raw
//! input point; the first-added transform runs last and produces the output
//! (`itkCompositeTransform.hxx:60-71`):
//!
//! ```text
//! T(x) = T0(T1(...TN-1(x)...))   for transforms added in order T0, T1, ..., TN-1
//! ```
//!
//! so `T0` — added first — is the *outermost* transform, matching ITK's own
//! example: "a user wants to apply an Affine transform followed by a
//! Deformation Field transform. They first add the Affine, then the DF"
//! (`itkCompositeTransform.h:42-51`).
//!
//! [`transform_point`]: TransformBase::transform_point

use crate::erased::Transform;
use crate::error::{Result, TransformError};
use crate::transform::{ParametricTransform, TransformBase, check_len};

/// A stack of transforms composed by `y = T0(T1(...TN-1(x)...))`, where
/// `T0, ..., TN-1` were added in that order (`itk::CompositeTransform`). See
/// the module docs for the add-order / apply-order convention.
///
/// # Parameter and Jacobian composition (crate decision — not a literal port)
///
/// ITK's `CompositeTransform` concatenates only the sub-transforms flagged
/// via `SetNthTransformToOptimize` (`itkCompositeTransform.h:174-237`), and
/// sizes each sub-transform's Jacobian block by its *local* parameter count
/// (`GetNumberOfLocalParameters`) rather than its full parameter count, so a
/// displacement-field-like sub-transform can be queried per point without
/// materializing its (huge) dense Jacobian
/// (`itkCompositeTransform.hxx:450-591`).
///
/// This port implements the simplest faithful subset: **every** sub-transform
/// is always included, and each sub-transform's block uses its *full*
/// [`ParametricTransform::number_of_parameters`] — there is no optimize-flag
/// mechanism. `parameters()` / `set_parameters()` / `jacobian_wrt_parameters()`
/// all concatenate blocks in **reverse add order** (`itkCompositeTransform.hxx`
/// `GetParameters`: "the sub-transforms are read in reverse queue order... the
/// last sub-transform to be added is returned first"). The Jacobian is
/// assembled with ITK's exact chain-rule recursion
/// (`itkCompositeTransform.hxx:464-591`): each sub-transform's own
/// parameter-Jacobian block is inserted unchanged, and every
/// previously-inserted block is then left-multiplied by the current
/// sub-transform's spatial Jacobian `d(transform_point)/dx`, so a perturbation
/// of an earlier (more-recently-added) sub-transform's parameters correctly
/// propagates through every transform applied afterwards.
///
/// That spatial Jacobian comes from [`TransformBase::jacobian_wrt_position`], which
/// every matrix-offset, translation and scale transform answers in closed form
/// (ITK's `ComputeJacobianWithRespectToPosition`); only [`BSplineTransform`] and
/// [`DisplacementFieldTransform`] fall back to the trait's finite-difference
/// default.
///
/// [`BSplineTransform`]: crate::BSplineTransform
/// [`DisplacementFieldTransform`]: crate::DisplacementFieldTransform
///
/// A sub-transform with [`ParametricTransform::has_local_support`] (e.g. a
/// dense displacement field) still composes *correctly* here — its own
/// `jacobian_wrt_parameters` already returns a full (mostly-zero) dense
/// Jacobian — just without the memory savings ITK's local-parameter-offset
/// scheme provides.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeTransform {
    dimension: usize,
    /// Added order: `transforms[0]` is `T0`, applied *last*.
    transforms: Vec<Transform>,
}

impl CompositeTransform {
    /// An empty composite transform (identity) of the given spatial
    /// `dimension`. Every sub-transform added later must share this
    /// dimension.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 1, "dimension must be >= 1");
        Self {
            dimension,
            transforms: Vec::new(),
        }
    }

    /// Push `transform` to the back of the queue (`itk::CompositeTransform::
    /// AddTransform` / `PushBackTransform`). It becomes the new outermost
    /// transform for `set_parameters`/`parameters` ordering, but the new
    /// *innermost* (first-applied) transform for `transform_point` — see the
    /// module docs.
    ///
    /// Takes an erased [`Transform`], as `itk::simple::CompositeTransform::
    /// AddTransform(const Transform &)` does — so the queue can be walked back
    /// out by concrete type, which is what writing a composite transform file
    /// requires.
    ///
    /// Fails with [`TransformError::DimensionMismatch`] when the sub-transform's
    /// dimension disagrees with the composite's, as upstream's
    /// `sitkExceptionMacro("Transform argument has dimension ... does not match
    /// this dimension of ...")` (`sitkCompositeTransform.cxx:194-200`).
    pub fn add_transform(&mut self, transform: Transform) -> Result<()> {
        if transform.dimension() != self.dimension {
            return Err(TransformError::DimensionMismatch);
        }
        self.transforms.push(transform);
        Ok(())
    }

    /// The sub-transforms in add order — `itk::CompositeTransform::GetTransformQueue()`,
    /// front to back.
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }

    /// The `n`-th sub-transform in add order (`GetNthTransform`), or `None` when
    /// `n` is out of range.
    pub fn nth_transform(&self, n: usize) -> Option<&Transform> {
        self.transforms.get(n)
    }

    /// Number of sub-transforms in the queue.
    pub fn number_of_transforms(&self) -> usize {
        self.transforms.len()
    }

    /// The inverse composite: every sub-transform inverted, and the queue
    /// reversed — `itk::CompositeTransform::GetInverse`
    /// (`itkCompositeTransform.hxx:406-436`) walks the queue front to back and
    /// `PushFrontTransform`s each inverse. Errors as soon as any sub-transform
    /// has no inverse, exactly where ITK's `GetInverse` clears the queue and
    /// returns `false`.
    pub fn inverse(&self) -> Result<Self> {
        let mut inverse = Self::new(self.dimension);
        for t in &self.transforms {
            inverse.transforms.insert(0, t.inverse()?);
        }
        Ok(inverse)
    }
}

impl TransformBase for CompositeTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        let mut out = point.to_vec();
        for t in self.transforms.iter().rev() {
            out = t.transform_point(&out);
        }
        out
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    /// `itk::MultiTransform::IsLinear()`: linear iff every sub-transform is
    /// (trivially `true` for an empty queue, matching an all-quantified
    /// `all()` over zero elements).
    fn is_linear(&self) -> bool {
        self.transforms.iter().all(|t| t.is_linear())
    }

    /// The chain rule over the queue in application order (last-added first):
    /// `dT/dx = J_{T₀} · … · J_{T_{N−1}}`, each evaluated at the point its own
    /// sub-transform sees.
    fn jacobian_wrt_position(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dimension;
        let mut acc = diagonal_ones(dim);
        let mut current = point.to_vec();
        for t in self.transforms.iter().rev() {
            let spatial = t.jacobian_wrt_position(&current);
            acc = mat_mul(&spatial, &acc, dim);
            current = t.transform_point(&current);
        }
        acc
    }
}

/// The row-major `n × n` identity.
fn diagonal_ones(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for d in 0..n {
        m[d * n + d] = 1.0;
    }
    m
}

/// Row-major `n × n` matrix product `a · b`.
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[r * n + k] * b[k * n + c];
            }
            out[r * n + c] = sum;
        }
    }
    out
}

impl ParametricTransform for CompositeTransform {
    fn number_of_parameters(&self) -> usize {
        self.transforms
            .iter()
            .map(|t| t.number_of_parameters())
            .sum()
    }

    fn parameters(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.number_of_parameters());
        for t in self.transforms.iter().rev() {
            out.extend(t.parameters());
        }
        out
    }

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, self.number_of_parameters())?;
        let mut offset = 0;
        for t in self.transforms.iter_mut().rev() {
            let n = t.number_of_parameters();
            t.set_parameters(&params[offset..offset + n])?;
            offset += n;
        }
        Ok(())
    }

    /// Each sub-transform's fixed parameters, concatenated in **reverse** queue
    /// order — the same order as [`parameters`] — matching
    /// `itk::CompositeTransform::GetFixedParameters`, which iterates
    /// `transforms.rbegin()..rend()` (`itkCompositeTransform.hxx:694-713`).
    /// Its base class `itk::MultiTransform` concatenates them in *forward* queue
    /// order instead (`itkMultiTransform.hxx:134-153`); `CompositeTransform`
    /// overrides that to agree with its own reverse-order `GetParameters`
    /// (ledger §2.77).
    ///
    /// [`parameters`]: ParametricTransform::parameters
    fn fixed_parameters(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.number_of_fixed_parameters());
        for t in self.transforms.iter().rev() {
            out.extend(t.fixed_parameters());
        }
        out
    }

    fn number_of_fixed_parameters(&self) -> usize {
        self.transforms
            .iter()
            .map(|t| t.number_of_fixed_parameters())
            .sum()
    }

    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        let expected = self.number_of_fixed_parameters();
        if params.len() != expected {
            return Err(TransformError::InvalidFixedParameters {
                got: params.len(),
                expected: format!("{expected} (the sub-transforms' fixed parameters)"),
            });
        }
        let mut offset = 0;
        for t in self.transforms.iter_mut().rev() {
            let n = t.number_of_fixed_parameters();
            t.set_fixed_parameters(&params[offset..offset + n])?;
            offset += n;
        }
        Ok(())
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dimension;
        let total_params = self.number_of_parameters();
        let mut out = vec![0.0; dim * total_params];

        let mut offset = 0usize;
        let mut transformed_point = point.to_vec();
        // Reverse add order: last-added first, matching transform_point's
        // application order (see module docs).
        for transform in self.transforms.iter().rev() {
            let offset_last = offset;
            let n = transform.number_of_parameters();
            if n > 0 {
                let block = transform.jacobian_wrt_parameters(&transformed_point);
                for i in 0..dim {
                    for j in 0..n {
                        out[i * total_params + offset_last + j] = block[i * n + j];
                    }
                }
                offset += n;
            }

            if offset_last > 0 {
                // A perturbation of any earlier (already-inserted) block
                // propagates through this transform too, so left-multiply
                // those columns by this transform's spatial Jacobian.
                let spatial = transform.jacobian_wrt_position(&transformed_point);
                for c in 0..offset_last {
                    let mut col = vec![0.0; dim];
                    for (r, slot) in col.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for k in 0..dim {
                            sum += spatial[r * dim + k] * out[k * total_params + c];
                        }
                        *slot = sum;
                    }
                    for r in 0..dim {
                        out[r * total_params + c] = col[r];
                    }
                }
            }

            transformed_point = transform.transform_point(&transformed_point);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bspline::BSplineTransform;
    use crate::erased::TransformKind;
    use crate::transform::{
        AffineTransform, Euler2DTransform, ScaleTransform, TranslationTransform,
    };

    #[test]
    fn inverse_reverses_the_queue_and_inverts_each_sub_transform() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![5.0, -1.0]).into())
            .unwrap();
        c.add_transform(Euler2DTransform::new(0.7, [1.0, 2.0], [3.0, 4.0]).into())
            .unwrap();

        let inv = c.inverse().unwrap();
        assert_eq!(inv.number_of_transforms(), 2);
        // PushFrontTransform order: the inverted rotation is now first-added.
        assert_eq!(inv.nth_transform(0).unwrap().kind(), TransformKind::Euler);
        assert_eq!(
            inv.nth_transform(1).unwrap().kind(),
            TransformKind::Translation
        );

        for p in [[0.0, 0.0], [2.0, 3.0], [-7.5, 11.0]] {
            let back = inv.transform_point(&c.transform_point(&p));
            for d in 0..2 {
                assert!((back[d] - p[d]).abs() < 1e-9, "{back:?} vs {p:?}");
            }
        }
    }

    #[test]
    fn inverse_fails_when_any_sub_transform_has_none() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        c.add_transform(
            BSplineTransform::new(2, &[0.0, 0.0], &[4.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[2, 2])
                .unwrap()
                .into(),
        )
        .unwrap();
        assert!(matches!(c.inverse(), Err(TransformError::NoInverse(_))));
    }

    #[test]
    fn identity_of_empty_composite() {
        let c = CompositeTransform::new(2);
        assert_eq!(c.number_of_parameters(), 0);
        assert_eq!(c.transform_point(&[3.0, -4.0]), vec![3.0, -4.0]);
    }

    #[test]
    fn two_translations_sum() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        c.add_transform(TranslationTransform::new(vec![3.0, 4.0]).into())
            .unwrap();
        assert_eq!(c.transform_point(&[0.0, 0.0]), vec![4.0, 6.0]);
    }

    #[test]
    fn set_parameters_rejects_wrong_length() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        c.add_transform(TranslationTransform::new(vec![3.0, 4.0]).into())
            .unwrap();
        assert!(matches!(
            c.set_parameters(&[1.0, 2.0, 3.0]),
            Err(TransformError::InvalidParameters {
                got: 3,
                expected: 4
            })
        ));
    }

    #[test]
    fn parameters_concatenate_in_reverse_add_order() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        c.add_transform(ScaleTransform::identity(2).into()).unwrap();
        assert_eq!(c.number_of_parameters(), 4);
        let params = c.parameters();
        // Last-added (Scale) is read first, then first-added (Translation) —
        // see the struct docs.
        assert_eq!(&params[0..2], &[1.0, 1.0]);
        assert_eq!(&params[2..4], &[1.0, 2.0]);
    }

    #[test]
    fn rotate_then_translate_matches_manual_composition() {
        // Added order [translation, rotation] ⇒ rotation (last-added) applies
        // first, translation (first-added) applies last: y = R(x) + t.
        let angle = std::f64::consts::FRAC_PI_2;
        let translation = vec![5.0, -1.0];
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(translation.clone()).into())
            .unwrap();
        c.add_transform(Euler2DTransform::new(angle, [0.0, 0.0], [0.0, 0.0]).into())
            .unwrap();

        let p = [2.0, 3.0];
        let y = c.transform_point(&p);

        let rotated = Euler2DTransform::new(angle, [0.0, 0.0], [0.0, 0.0]).transform_point(&p);
        let expected = [rotated[0] + translation[0], rotated[1] + translation[1]];
        for d in 0..2 {
            assert!(
                (y[d] - expected[d]).abs() < 1e-12,
                "dim {d}: {y:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn adding_a_sub_transform_of_another_dimension_fails() {
        let mut c = CompositeTransform::new(3);
        assert!(matches!(
            c.add_transform(TranslationTransform::new(vec![1.0, 2.0]).into()),
            Err(TransformError::DimensionMismatch)
        ));
    }

    #[test]
    fn jacobian_is_finite_difference_consistent_with_three_subtransforms() {
        // Off-center, non-trivial parameters on every sub-transform: exercises
        // the chain-rule left-multiplication through two "earlier" blocks.
        let mut c = CompositeTransform::new(2);
        c.add_transform(
            AffineTransform::new(
                2,
                vec![1.2, 0.3, -0.1, 0.9],
                vec![0.5, -0.7],
                vec![1.0, 2.0],
            )
            .into(),
        )
        .unwrap();
        c.add_transform(TranslationTransform::new(vec![2.0, -3.0]).into())
            .unwrap();
        c.add_transform(Euler2DTransform::new(0.4, [1.5, -0.5], [0.2, 0.3]).into())
            .unwrap();

        let point = [7.0, -2.0];
        let jac = c.jacobian_wrt_parameters(&point);
        let n = c.number_of_parameters();
        let base = c.parameters();
        let h = 1e-6;
        for k in 0..n {
            let mut pp = base.clone();
            pp[k] += h;
            c.set_parameters(&pp).unwrap();
            let yp = c.transform_point(&point);

            let mut pm = base.clone();
            pm[k] -= h;
            c.set_parameters(&pm).unwrap();
            let ym = c.transform_point(&point);

            for i in 0..2 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-4,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }

    /// The composite's own `jacobian_wrt_position` is a product of its
    /// sub-transforms' spatial Jacobians; check it against a finite difference
    /// of the assembled `transform_point`.
    #[test]
    fn position_jacobian_matches_finite_difference() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![0.3, -0.8]).into())
            .unwrap();
        c.add_transform(Euler2DTransform::new(0.37, [0.5, -0.4], [0.4, 0.9]).into())
            .unwrap();
        c.add_transform(ScaleTransform::new(vec![1.3, 0.7], vec![0.4, 0.9]).into())
            .unwrap();

        let point = [1.7f64, -0.6];
        let analytic = c.jacobian_wrt_position(&point);

        let dim = 2;
        for col in 0..dim {
            let h = 1e-6 * point[col].abs().max(1.0);
            let mut plus = point.to_vec();
            let mut minus = point.to_vec();
            plus[col] += h;
            minus[col] -= h;
            let f_plus = c.transform_point(&plus);
            let f_minus = c.transform_point(&minus);
            for row in 0..dim {
                let fd = (f_plus[row] - f_minus[row]) / (2.0 * h);
                let a = analytic[row * dim + col];
                assert!(
                    (a - fd).abs() < 1e-6,
                    "entry ({row},{col}): analytic {a} vs finite difference {fd}"
                );
            }
        }
    }
}
