//! [`Transform::matrix_offset_map`] — the point map in the one form another backend
//! can reproduce **bit for bit**.
//!
//! # The contract, and why it is stronger than `is_linear`
//!
//! > if [`Transform::matrix_offset_map`] returns `Some(m)`, then for every finite `p`,
//! > `transform.transform_point(p)` **is** `mat_vec(m.matrix, p) + m.offset` — bit for
//! > bit, the same operations, in the same order, on the same operands.
//!
//! [`TransformBase::is_linear`](crate::TransformBase::is_linear) asks only whether the
//! map *is* `x ↦ M·x + b` mathematically. [`ScaleTransform`](crate::ScaleTransform) is
//! linear by that test and refused by this one, because it evaluates `(p − c)·s + c`
//! and that is a **different rounding** from `M·p + b`.
//!
//! # Who needs the last bits, and why
//!
//! The caller is `sitk-registration`'s CUDA path, which resamples the fixed image
//! *and* its in-buffer predicate through a fixed-initial transform. The predicate is a
//! 0/1 field whose value at the buffer border is decided by comparing a continuous
//! index against `[-0.5, size − 0.5)`: one ulp there flips a shell of voxels and moves
//! the valid-point count, which the device path pins as **exactly** equal to the host's.
//! The registration metric already recovers an affine from a transform by *probing* it
//! (`b = T(0)`, `A[:,e] = T(e_e) − T(0)`) — a reconstruction ~1e-12 away from the
//! transform's own arithmetic. That is fine for a metric gated at 1e-9 and fatal for a
//! predicate. This function reconstructs nothing: it hands back the matrix and offset
//! the transform **already multiplies**.
//!
//! # Why the accessors cannot lie
//!
//! Every matrix-offset transform here stores `matrix` and `offset` as struct fields;
//! `matrix()`/`offset()` return `&self.matrix`/`&self.offset` (`transform.rs:448`,
//! `:463` and the nine siblings), and `transform_point` is
//!
//! ```ignore
//! let mx = matrix::mat_vec(&self.matrix, point, dim);
//! (0..dim).map(|d| mx[d] + self.offset[d]).collect()
//! ```
//!
//! (`transform.rs:471-475`, `:774`, `:982`, `:1287`, `:1531`, `:1812`, `:2115`,
//! `:2433`, `:2715`). The accessor and the evaluator read the **same field**, so they
//! cannot disagree — including for the versor family, where the quaternion → matrix
//! conversion happens once, at parameter-set time, into that stored field, and *not* at
//! call time. That is what makes the versors bitwise-eligible: a structural property,
//! not a numerical coincidence. It would survive even a staleness bug in the mutators,
//! since a stale cache moves the accessor and the evaluator equally.
//!
//! # What is refused, and why refusing beats approximating
//!
//! - [`ScaleTransform`](crate::ScaleTransform) /
//!   [`ScaleLogarithmicTransform`](crate::ScaleLogarithmicTransform):
//!   `(p − c)·s + c` (`transform.rs:2848-2853`, `:2971-2973`). That *is* `M·p + b` with
//!   `M = diag(s)`, `b = c − s·c` — in exact arithmetic. Folding it rounds `b` once
//!   where the transform rounds per point, and the two differ in the last bits. The
//!   last bits are the whole reason this function exists, so it does not fold: it
//!   refuses.
//! - [`CompositeTransform`](crate::CompositeTransform) applies its stages in sequence,
//!   each rounding on its own (`composite.rs:144-149`). Multiplying the stage matrices
//!   together is the same error as folding a scale. If a backend ever wants a composite,
//!   it transcribes the stages **in order**; it does not get one matrix from here.
//! - `BSpline` / `DisplacementField` are not linear at all.
//!
//! # The one variant that rests on an argument
//!
//! [`TranslationTransform`](crate::TranslationTransform) has no `matrix`/`offset`
//! fields: it evaluates `p[d] + t[d]` (`transform.rs:329-336`), so the matrix here is
//! *synthesized* as the identity, and the bitwise claim becomes an IEEE-754 argument
//! rather than a shared field — `mat_vec(I, p)[0]` is `0.0 + 1.0·p₀ + 0.0·p₁ + 0.0·p₂`,
//! and adding `±0.0` to a finite value is exact, so it is `p₀` to the bit. The argument
//! is pinned, not trusted: `translation_is_bitwise_the_identity_matrix_form` is the one
//! test here that could genuinely fail, and if it ever does the answer is a translation
//! form of its own in the consumer — not a refusal.

use crate::erased::Transform;

/// A point map a backend can reproduce bit for bit: `x ↦ mat_vec(matrix, x) + offset`.
///
/// `matrix` is row-major `dim × dim`; `offset` has length `dim`.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixOffsetMap {
    /// Row-major `dim × dim`.
    pub matrix: Vec<f64>,
    /// Length `dim`.
    pub offset: Vec<f64>,
}

impl Transform {
    /// The stored matrix-offset form of this transform's point map, or `None` when it
    /// has none that is **bitwise** equal to its own `transform_point`.
    ///
    /// See the [module docs](self) for the contract and for the variants that are
    /// mathematically linear yet refused here.
    ///
    /// The match is exhaustive on purpose, and it lives in this crate for that reason:
    /// [`Transform`] is `#[non_exhaustive]`, so a match in a *downstream* crate is
    /// forced to carry a wildcard and would silently accept a new variant. Here a new
    /// variant is a compile error, and whoever adds it has to decide — in this file,
    /// against this contract — whether its arithmetic is `mat_vec(M, p) + b` on the bit.
    pub fn matrix_offset_map(&self) -> Option<MatrixOffsetMap> {
        fn stored(matrix: &[f64], offset: &[f64]) -> Option<MatrixOffsetMap> {
            Some(MatrixOffsetMap {
                matrix: matrix.to_vec(),
                offset: offset.to_vec(),
            })
        }

        match self {
            // The matrix-offset family: `transform_point` multiplies these very fields.
            Transform::Affine(t) => stored(t.matrix(), t.offset()),
            Transform::Euler2D(t) => stored(t.matrix(), t.offset()),
            Transform::Euler3D(t) => stored(t.matrix(), t.offset()),
            Transform::Similarity2D(t) => stored(t.matrix(), t.offset()),
            Transform::Similarity3D(t) => stored(t.matrix(), t.offset()),
            Transform::Versor(t) => stored(t.matrix(), t.offset()),
            Transform::VersorRigid3D(t) => stored(t.matrix(), t.offset()),
            Transform::ScaleVersor3D(t) => stored(t.matrix(), t.offset()),
            Transform::ScaleSkewVersor3D(t) => stored(t.matrix(), t.offset()),
            Transform::ComposeScaleSkewVersor3D(t) => stored(t.matrix(), t.offset()),

            // Synthesized, and pinned: `p + t` is `mat_vec(I, p) + t` to the bit.
            Transform::Translation(t) => {
                let dim = t.translation().len();
                let mut matrix = vec![0.0; dim * dim];
                for d in 0..dim {
                    matrix[d * dim + d] = 1.0;
                }
                Some(MatrixOffsetMap {
                    matrix,
                    offset: t.translation().to_vec(),
                })
            }

            // Linear, but `(p − c)·s + c` is a different rounding from `M·p + b`.
            Transform::Scale(_) | Transform::ScaleLogarithmic(_) => None,
            // Linear iff every stage is — but a composed matrix is not the stages'
            // arithmetic. Transcribe the stages in order, or refuse. Refuse.
            Transform::Composite(_) => None,
            // Not linear.
            Transform::BSpline(_) | Transform::DisplacementField(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AffineTransform, BSplineTransform, ComposeScaleSkewVersor3DTransform, CompositeTransform,
        DisplacementFieldTransform, Euler2DTransform, Euler3DTransform, ScaleLogarithmicTransform,
        ScaleSkewVersor3DTransform, ScaleTransform, ScaleVersor3DTransform, Similarity2DTransform,
        Similarity3DTransform, TransformBase, TranslationTransform, VersorRigid3DTransform,
        VersorTransform,
    };
    use sitk_core::matrix::mat_vec;

    /// Deterministic probe points over a physical extent an image would actually
    /// occupy (millimetres, off-origin, both signs), plus `±0.0`, a near-underflow and
    /// a large magnitude — the arithmetic that can differ is the one near a
    /// cancellation. An LCG, so there is no `rand` dependency and the points are the
    /// same on every run.
    fn probes(dim: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 400.0 - 200.0
        };
        for _ in 0..256 {
            out.push((0..dim).map(|_| next()).collect());
        }
        out.push(vec![0.0; dim]);
        out.push(vec![-0.0; dim]);
        out.push((0..dim).map(|d| 1e-9 * (d as f64 + 1.0)).collect());
        out.push((0..dim).map(|d| -1e9 * (d as f64 + 1.0)).collect());
        out
    }

    /// The contract, checked on the bits: `transform_point(p)` **is** `mat_vec(M, p) + b`,
    /// not merely close to it.
    fn assert_bitwise(name: &str, t: &Transform, dim: usize) {
        let m = t
            .matrix_offset_map()
            .unwrap_or_else(|| panic!("{name}: expected a matrix-offset map, got none"));
        assert_eq!(m.matrix.len(), dim * dim, "{name}: matrix shape");
        assert_eq!(m.offset.len(), dim, "{name}: offset shape");

        for p in probes(dim) {
            let got = t.transform_point(&p);
            let mx = mat_vec(&m.matrix, &p, dim);
            for d in 0..dim {
                let want = mx[d] + m.offset[d];
                assert_eq!(
                    got[d].to_bits(),
                    want.to_bits(),
                    "{name}: axis {d} at {p:?}: transform_point gave {g}, mat_vec(M,p)+b \
                     gave {want} — the map is NOT bitwise, and the variant must be refused \
                     rather than approximated",
                    g = got[d],
                );
            }
        }
    }

    #[test]
    fn the_matrix_offset_family_is_bitwise_its_stored_matrix_and_offset() {
        let t3 = [3.5, -2.25, 7.125];
        let c3 = [12.0, -4.5, 33.25];

        assert_bitwise(
            "Affine",
            &Transform::Affine(AffineTransform::new(
                3,
                vec![0.97, -0.21, 0.11, 0.19, 0.95, -0.24, -0.14, 0.22, 0.96],
                t3.to_vec(),
                c3.to_vec(),
            )),
            3,
        );
        assert_bitwise(
            "Euler3D",
            &Transform::Euler3D(Euler3DTransform::new(0.31, -0.17, 0.44, t3, c3)),
            3,
        );
        assert_bitwise(
            "VersorRigid3D",
            &Transform::VersorRigid3D(VersorRigid3DTransform::new(0.11, -0.23, 0.07, t3, c3)),
            3,
        );
        assert_bitwise(
            "Versor",
            &Transform::Versor(VersorTransform::new(0.11, -0.23, 0.07, c3)),
            3,
        );
        assert_bitwise(
            "Similarity3D",
            &Transform::Similarity3D(Similarity3DTransform::new(1.37, 0.11, -0.23, 0.07, t3, c3)),
            3,
        );
        assert_bitwise(
            "ScaleVersor3D",
            &Transform::ScaleVersor3D(ScaleVersor3DTransform::new(
                [1.1, 0.9, 1.3],
                0.11,
                -0.23,
                0.07,
                t3,
                c3,
            )),
            3,
        );
        assert_bitwise(
            "ScaleSkewVersor3D",
            &Transform::ScaleSkewVersor3D(ScaleSkewVersor3DTransform::new(
                [1.1, 0.9, 1.3],
                [0.02, -0.03, 0.05, 0.01, -0.04, 0.06],
                0.11,
                -0.23,
                0.07,
                t3,
                c3,
            )),
            3,
        );
        assert_bitwise(
            "ComposeScaleSkewVersor3D",
            &Transform::ComposeScaleSkewVersor3D(ComposeScaleSkewVersor3DTransform::new(
                [1.1, 0.9, 1.3],
                [0.02, -0.03, 0.05],
                0.11,
                -0.23,
                0.07,
                t3,
                c3,
            )),
            3,
        );

        // 2-D: eligible, though the 3-D device resample can never be handed one.
        assert_bitwise(
            "Euler2D",
            &Transform::Euler2D(Euler2DTransform::new(0.4, [3.0, -2.0], [10.0, 20.0])),
            2,
        );
        assert_bitwise(
            "Similarity2D",
            &Transform::Similarity2D(Similarity2DTransform::new(
                1.25,
                0.4,
                [3.0, -2.0],
                [10.0, 20.0],
            )),
            2,
        );
    }

    /// The one eligibility that rests on an IEEE-754 argument instead of a shared
    /// field: `p + t` vs `mat_vec(I, p) + t`.
    #[test]
    fn translation_is_bitwise_the_identity_matrix_form() {
        assert_bitwise(
            "Translation",
            &Transform::Translation(TranslationTransform::new(vec![3.5, -2.25, 7.125])),
            3,
        );
        assert_bitwise(
            "Translation (2-D)",
            &Transform::Translation(TranslationTransform::new(vec![-11.0, 0.5])),
            2,
        );
    }

    /// `ScaleTransform` is refused — and this pins *why*, so the refusal cannot be
    /// waved away later as over-caution: the folded form `M·p + b` with `M = diag(s)`,
    /// `b = c − s·c` disagrees with `(p − c)·s + c` in the last bits. If this ever
    /// stops finding a disagreement, the refusal is still right (a fold is not the
    /// transform's arithmetic) but this test is no longer evidence for it, and it fails
    /// loudly rather than passing vacuously.
    #[test]
    fn folding_a_scale_transform_into_a_matrix_changes_the_bits() {
        let scale = vec![3.0, 0.1, 7.25];
        let center = vec![0.1, 123.456, -7.7];
        let t = ScaleTransform::new(scale.clone(), center.clone());

        let mut disagreements = 0usize;
        for p in probes(3) {
            let got = t.transform_point(&p);
            for d in 0..3 {
                let folded = scale[d] * p[d] + (center[d] - scale[d] * center[d]);
                if got[d].to_bits() != folded.to_bits() {
                    disagreements += 1;
                }
            }
        }
        assert!(
            disagreements > 0,
            "the folded matrix form agreed with ScaleTransform on every probe; this test \
             is no longer evidence that the fold is lossy"
        );
    }

    #[test]
    fn the_transforms_with_no_bitwise_matrix_form_are_refused() {
        let scale = ScaleTransform::new(vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 1.0]);
        assert!(
            Transform::Scale(scale).matrix_offset_map().is_none(),
            "ScaleTransform must be refused: (p - c)*s + c is not mat_vec(M,p) + b"
        );

        let slog = ScaleLogarithmicTransform::new(vec![0.5, 0.5, 0.5], vec![1.0, 1.0, 1.0]);
        assert!(
            Transform::ScaleLogarithmic(slog)
                .matrix_offset_map()
                .is_none(),
            "ScaleLogarithmicTransform delegates to ScaleTransform and is refused with it"
        );

        // A composite of two stages that are *individually* eligible. This is the trap:
        // it is linear, and folding the stages would look correct.
        let mut composite = CompositeTransform::new(3);
        composite
            .add_transform(Transform::Translation(TranslationTransform::new(vec![
                1.0, 2.0, 3.0,
            ])))
            .unwrap();
        composite
            .add_transform(Transform::Affine(AffineTransform::new(
                3,
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                vec![0.0; 3],
                vec![0.0; 3],
            )))
            .unwrap();
        assert!(
            composite.is_linear(),
            "the composite of two linear stages is linear — which is exactly why \
             `is_linear` is the wrong test and this module has its own"
        );
        assert!(
            Transform::Composite(composite)
                .matrix_offset_map()
                .is_none(),
            "a CompositeTransform must be refused, not folded into one matrix"
        );

        let field = DisplacementFieldTransform::new(
            3,
            &[4, 4, 4],
            &[0.0; 3],
            &[1.0; 3],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert!(
            Transform::DisplacementField(field)
                .matrix_offset_map()
                .is_none(),
            "a displacement field is not linear"
        );

        let bspline = BSplineTransform::new(
            3,
            &[0.0; 3],
            &[10.0; 3],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[2, 2, 2],
        )
        .unwrap();
        assert!(
            Transform::BSpline(bspline).matrix_offset_map().is_none(),
            "a B-spline transform is not linear"
        );
    }
}
