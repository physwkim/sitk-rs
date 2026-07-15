//! The single implementation of ITK's `itk::ImageBase` index↔physical
//! coordinate transforms, shared by [`Image`](crate::core::Image) and every filter,
//! transform, and registration consumer that converts between a physical point,
//! a continuous index, and a discrete index.
//!
//! Before this module the port carried four independent re-derivations of the
//! same transform, in three term-associations and two origin folds (see
//! `bench/results/coord-rounding-port-map.md` §7). They agreed for axis-aligned
//! modest-origin geometry and diverged from ITK — and from each other — under
//! oblique directions or large origins, and one path (`d / spacing`) diverged
//! even for a diagonal geometry, flipping a discrete index. This module makes
//! the transform hold by construction, matching `itkImageBase.hxx` exactly.
//!
//! # ITK's construction (itkImageBase.hxx:165-175)
//!
//! ITK precomputes two matrices once per geometry:
//! `m_IndexToPhysicalPoint = Direction · diag(spacing)` and
//! `m_PhysicalPointToIndex = inverse(m_IndexToPhysicalPoint)` — the inverse of
//! the **whole composed** matrix, not the direction alone. Each conversion is
//! then one matrix product plus ITK's origin fold. This module builds the same
//! two matrices ([`index_to_physical_matrix`], [`physical_to_index_matrix`]) and
//! applies them with ITK's exact per-method fold and association.
//!
//! # The two origin folds
//!
//! ITK's integer and continuous index→physical methods disagree in the origin
//! fold, and the difference is observable at large origins:
//! - `TransformIndexToPhysicalPoint` (itkImageBase.h:592-604) seeds the
//!   accumulator with the origin and adds terms after — origin **first**,
//!   `((origin + t0) + t1)` — see [`index_to_physical_point`].
//! - `TransformContinuousIndexToPhysicalPoint` (itkImageBase.h:558-572) sums the
//!   terms then adds the origin — origin **last**, `((t0 + t1) + origin)` — see
//!   [`continuous_index_to_physical_point`].
//!
//! # Inverse algorithm (oblique residual)
//!
//! ITK inverts the composed matrix with an SVD pseudo-inverse
//! (`itk::Matrix::GetInverse` → `Math::SVD(...).PseudoInverse()`, itkMatrix.h:336);
//! this module uses the port's Gauss-Jordan [`matrix::invert`]. For a diagonal
//! geometry both yield exactly `diag(1/spacing)`, so the result is bit-identical
//! (this is what fixes the diagonal index flip). For an **oblique** direction the
//! two inverse algorithms differ by at most a few ULP; the association
//! (compose → invert → multiply) matches ITK, the last bits of the inverse
//! entries may not. Documented, not hidden.

use crate::core::matrix;

/// [`index_to_physical_matrix`] writing into a caller-provided `dim × dim`
/// buffer — the allocation-free form for per-pixel loops (label statistics).
pub fn index_to_physical_matrix_into(
    direction: &[f64],
    spacing: &[f64],
    out: &mut [f64],
    dim: usize,
) {
    debug_assert_eq!(direction.len(), dim * dim);
    debug_assert_eq!(spacing.len(), dim);
    debug_assert!(out.len() >= dim * dim);
    for r in 0..dim {
        for c in 0..dim {
            out[r * dim + c] = direction[r * dim + c] * spacing[c];
        }
    }
}

/// ITK `m_IndexToPhysicalPoint = Direction · diag(spacing)` (itkImageBase.hxx:174).
///
/// Row-major `dim × dim`. Because `scale` is diagonal, the matrix product reduces
/// to one exact product per entry: `m[r][c] = direction[r][c] · spacing[c]`.
pub fn index_to_physical_matrix(direction: &[f64], spacing: &[f64], dim: usize) -> Vec<f64> {
    let mut m = vec![0.0; dim * dim];
    index_to_physical_matrix_into(direction, spacing, &mut m, dim);
    m
}

/// ITK `m_PhysicalPointToIndex = inverse(m_IndexToPhysicalPoint)` (itkImageBase.hxx:175).
///
/// Inverts the **whole composed** `Direction · diag(spacing)` — not the direction
/// alone — so the diagonal case yields `diag(1/spacing)` and a physical→index
/// conversion multiplies by the reciprocal exactly as ITK does. `None` if the
/// composed matrix is singular. See the module docs on the oblique ULP residual.
pub fn physical_to_index_matrix(
    direction: &[f64],
    spacing: &[f64],
    dim: usize,
) -> Option<Vec<f64>> {
    matrix::invert(&index_to_physical_matrix(direction, spacing, dim), dim)
}

/// Origin-**first** apply into a caller buffer,
/// `out[r] = origin[r] + Σ_c m[r][c]·index[c]` accumulated with the origin as the
/// initial term — ITK's `TransformIndexToPhysicalPoint` fold
/// (itkImageBase.h:598-602). The one implementation of the origin-first fold.
pub fn index_to_physical_point_f64_into(
    i2p: &[f64],
    origin: &[f64],
    index: &[f64],
    out: &mut [f64],
    dim: usize,
) {
    debug_assert_eq!(i2p.len(), dim * dim);
    debug_assert!(out.len() >= dim);
    for r in 0..dim {
        let mut acc = origin[r];
        for c in 0..dim {
            acc += i2p[r * dim + c] * index[c];
        }
        out[r] = acc;
    }
}

/// Origin-**last** apply into a caller buffer,
/// `out[r] = (Σ_c i2p[r][c]·index[c]) + origin[r]` — ITK's
/// `TransformContinuousIndexToPhysicalPoint` fold (itkImageBase.h:565-570). The
/// one implementation of the origin-last fold.
pub fn continuous_index_to_physical_point_into(
    i2p: &[f64],
    origin: &[f64],
    index: &[f64],
    out: &mut [f64],
    dim: usize,
) {
    debug_assert_eq!(i2p.len(), dim * dim);
    debug_assert!(out.len() >= dim);
    for r in 0..dim {
        let mut acc = 0.0;
        for c in 0..dim {
            acc += i2p[r * dim + c] * index[c];
        }
        out[r] = acc + origin[r];
    }
}

/// ITK `TransformContinuousIndexToPhysicalPoint` (itkImageBase.h:558-572): the
/// terms are summed first and the origin is added **last** —
/// `p[r] = (Σ_c i2p[r][c]·index[c]) + origin[r]`.
pub fn continuous_index_to_physical_point(
    i2p: &[f64],
    origin: &[f64],
    index: &[f64],
    dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    continuous_index_to_physical_point_into(i2p, origin, index, &mut out, dim);
    out
}

/// ITK `TransformIndexToPhysicalPoint` (itkImageBase.h:592-604): the origin is
/// the initial accumulator term (origin **first**). The integer index is widened
/// to `f64` exactly as ITK's `double · IndexValueType` promotion does.
pub fn index_to_physical_point(i2p: &[f64], origin: &[f64], index: &[i64], dim: usize) -> Vec<f64> {
    let widened: Vec<f64> = index.iter().map(|&i| i as f64).collect();
    let mut out = vec![0.0; dim];
    index_to_physical_point_f64_into(i2p, origin, &widened, &mut out, dim);
    out
}

/// [`index_to_physical_point`] for an index already widened to `f64` (a resample
/// or warp output-grid counter). Same origin-**first** fold — ITK's integer
/// `TransformIndexToPhysicalPoint`, since the output index is discrete.
pub fn index_to_physical_point_f64(
    i2p: &[f64],
    origin: &[f64],
    index: &[f64],
    dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    index_to_physical_point_f64_into(i2p, origin, index, &mut out, dim);
    out
}

/// ITK `TransformPhysicalPointToContinuousIndex` (itkImageBase.h:517-532):
/// `cindex = p2i · (point − origin)`, the difference formed per component first,
/// then one matrix product.
pub fn physical_point_to_continuous_index(
    p2i: &[f64],
    origin: &[f64],
    point: &[f64],
    dim: usize,
) -> Vec<f64> {
    let diff: Vec<f64> = (0..dim).map(|k| point[k] - origin[k]).collect();
    matrix::mat_vec(p2i, &diff, dim)
}

/// ITK `Math::RoundHalfIntegerUp<IndexValueType>` (itkImageBase.h:476,
/// itkMath.h:191, itkMathDetail.h:110-116) — round to nearest integer, halfway
/// cases toward **+∞**: `1.5→2`, `2.5→3`, `-1.5→-1`, `-0.5→0`. This is
/// `floor(x + 0.5)`, **not** Rust `f64::round` (which is half away from zero).
pub fn round_half_integer_up(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    // The load-bearing diagonal index flip verified against ITK: point
    // 1.4999999999999998, spacing 3, origin 0, identity direction. ITK inverts
    // the composed diag(3) to diag(1/3) and multiplies -> continuous
    // 0.4999999999999999 -> RoundHalfIntegerUp -> index 0. The pre-fix port
    // divided (point / spacing = 0.49999999999999994) -> index 1.
    #[test]
    fn diagonal_physical_to_index_reciprocal_multiplies_like_itk() {
        let dir = [1.0, 0.0, 0.0, 1.0];
        let spacing = [3.0, 3.0];
        let origin = [0.0, 0.0];
        let p2i = physical_to_index_matrix(&dir, &spacing, 2).unwrap();
        let point = [1.4999999999999998, 0.0];
        let cindex = physical_point_to_continuous_index(&p2i, &origin, &point, 2);
        // Exactly ITK's reciprocal-multiply, NOT point/spacing.
        assert_eq!(cindex[0], (1.0 / 3.0) * 1.4999999999999998);
        assert_eq!(cindex[0], 0.4999999999999999);
        assert_ne!(cindex[0], 1.4999999999999998 / 3.0);
        assert_eq!(round_half_integer_up(cindex[0]), 0);
        // Non-vacuity: the old divide path would round to 1 here.
        assert_eq!(round_half_integer_up(1.4999999999999998 / 3.0), 1);
    }

    // Origin fold: ITK's integer method (origin-first) and continuous method
    // (origin-last) disagree at large origins, and this module reproduces both.
    // A single output row must accumulate two terms for the fold to bite, so the
    // direction is a shear — a diagonal geometry gives one term per row and the
    // fold is invisible. Row 0 gets terms (1.0, 1.0) at origin 1e16, whose ULP is
    // 2: adding 1.0 one at a time to the origin loses both (origin-first -> 1e16),
    // while summing them first survives (origin-last -> 1e16 + 2ulp).
    #[test]
    fn origin_fold_differs_between_integer_and_continuous_at_large_origin() {
        let dir = [1.0, 1.0, 0.0, 1.0]; // shear, invertible (det 1)
        let spacing = [1.0, 1.0];
        let origin = [1e16, 0.0];
        let i2p = index_to_physical_matrix(&dir, &spacing, 2);
        let integer = index_to_physical_point(&i2p, &origin, &[1, 1], 2);
        let continuous = continuous_index_to_physical_point(&i2p, &origin, &[1.0, 1.0], 2);
        assert_eq!(integer[0], 1e16); // ((origin + 1) + 1) fold
        assert_eq!(continuous[0], 1.0000000000000002e16); // ((1 + 1) + origin) fold
        // Non-vacuity: identical bits would mean the fold order was not preserved.
        assert_ne!(integer[0], continuous[0]);
    }

    // ITK NN / mask / sparse-Jacobian rounding is RoundHalfIntegerUp = floor(x+0.5),
    // half toward +INF; Rust f64::round is half AWAY from zero. They diverge on the
    // exact half at every negative half-integer. The separating input the mask path
    // cares about: a continuous index component of -0.5 -> ITK keeps voxel 0, while
    // f64::round drops to -1 (out of bounds -> the sample is wrongly rejected).
    #[test]
    fn round_half_integer_up_keeps_negative_half_at_zero_unlike_rust_round() {
        assert_eq!(round_half_integer_up(-0.5), 0); // ITK keeps voxel 0
        assert_eq!(round_half_integer_up(-1.5), -1);
        assert_eq!(round_half_integer_up(-2.5), -2);
        assert_eq!(round_half_integer_up(0.5), 1);
        assert_eq!(round_half_integer_up(1.5), 2);
        // Non-vacuity: Rust's half-away-from-zero round is the bug being replaced;
        // it maps every negative half the other way.
        assert_eq!((-0.5f64).round() as i64, -1);
        assert_eq!((-1.5f64).round() as i64, -2);
        assert_ne!(round_half_integer_up(-0.5), (-0.5f64).round() as i64);
    }
}
