//! Small dense linear algebra shared by filters: a fixed-size eigendecomposition
//! of a symmetric 2×2 or 3×3 matrix ([`symmetric_eigen`]), and a run-time-sized
//! symmetric pseudo-inverse solve ([`symmetric_pseudo_inverse_solve`]).

/// The widest matrix [`symmetric_eigen`]'s callers need: ITK's
/// `SymmetricSecondRankTensor` and `ShapeLabelObject` are both at most
/// 3-dimensional.
pub(crate) const MAX_DIM: usize = 3;

pub(crate) type Mat = [[f64; MAX_DIM]; MAX_DIM];

/// Cyclic Jacobi eigendecomposition of a symmetric `n × n` matrix. Returns
/// the eigenvalues in ascending order together with a matrix whose *columns*
/// are the matching eigenvectors — the shape `vnl_symmetric_eigensystem`
/// hands `itkShapeLabelMapFilter.hxx` as `eigen.D` and `eigen.V`, and the same
/// ascending order `itk::SymmetricEigenAnalysis` produces under its default
/// `OrderByValue`.
pub(crate) fn symmetric_eigen(input: &Mat, n: usize) -> ([f64; MAX_DIM], Mat) {
    let mut a = *input;
    let mut v = [[0.0; MAX_DIM]; MAX_DIM];
    for (i, row) in v.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }

    let mut norm2 = 0.0;
    for row in a.iter().take(n) {
        for &x in row.iter().take(n) {
            norm2 += x * x;
        }
    }
    let tol = f64::EPSILON * f64::EPSILON * norm2;

    for _sweep in 0..100 {
        let mut off = 0.0;
        for (p, row) in a.iter().enumerate().take(n) {
            for &x in row.iter().take(n).skip(p + 1) {
                off += x * x;
            }
        }
        if off <= tol {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // A <- A * J, then A <- J^T * A, then V <- V * J.
                for row in a.iter_mut().take(n) {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                {
                    // p < q, so the two rows can be borrowed disjointly.
                    let (head, tail) = a.split_at_mut(q);
                    for (apk, aqk) in head[p].iter_mut().zip(tail[0].iter_mut()).take(n) {
                        let (x, y) = (*apk, *aqk);
                        *apk = c * x - s * y;
                        *aqk = s * x + c * y;
                    }
                }
                for row in v.iter_mut().take(n) {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));

    let mut eigenvalues = [0.0; MAX_DIM];
    let mut vectors = [[0.0; MAX_DIM]; MAX_DIM];
    for (dst, &src) in order.iter().enumerate() {
        eigenvalues[dst] = a[src][src];
        for k in 0..n {
            vectors[k][dst] = v[k][src];
        }
    }
    (eigenvalues, vectors)
}

/// Absolute singular-value floor of `itk::KernelTransform::ComputeWMatrix`
/// (`itkKernelTransform.hxx:148-153`): it passes `rcond = 1e-8 / wmax` to
/// `SVDResult::PseudoInverse`, whose threshold is `rcond * wmax`, so a singular
/// value is treated as zero unless it exceeds `1e-8`. The degenerate `wmax == 0`
/// case takes `rcond = 0` there and zeroes everything, which the `> 0` test
/// below reproduces.
const SINGULAR_VALUE_FLOOR: f64 = 1e-8;

/// Minimum-norm least-squares solution of `A x = rhs` for a **symmetric**
/// `n × n` matrix `a` (row-major), by Moore-Penrose pseudo-inverse with ITK's
/// `1e-8` absolute singular-value floor.
///
/// This is `itk::Math::SVD(A).PseudoInverse(1e-8 / wmax) * rhs`
/// (`itkKernelTransform.hxx:148-153`, `itkMathSVD.h:51-67`) computed through a
/// symmetric eigendecomposition instead of an SVD. The two agree exactly: a
/// symmetric `A = V diag(λ) Vᵀ` has singular values `σᵢ = |λᵢ|` and left vectors
/// `uᵢ = sign(λᵢ) vᵢ`, so `V diag(1/σ) Uᵀ = Σ_{σᵢ > tol} (1/λᵢ) vᵢ vᵢᵀ`. The
/// caller must supply a symmetric `a`; `itk::KernelTransform`'s `L` matrix is,
/// by the way `ComputeL` lays out `K`, `P`, and `Pᵀ` around a zero block.
///
/// Cyclic Jacobi, as [`symmetric_eigen`], but sized at run time.
pub(crate) fn symmetric_pseudo_inverse_solve(mut a: Vec<f64>, rhs: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(rhs.len(), n);

    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // The Frobenius norm is invariant under the rotations, so it is computed once.
    let norm2: f64 = a.iter().map(|x| x * x).sum();
    let convergence = f64::EPSILON * f64::EPSILON * norm2;

    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= convergence {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p * n + q] == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // A <- A * J, then A <- Jᵀ * A, then V <- V * J.
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|k| a[k * n + k]).collect();
    let largest = eigenvalues.iter().fold(0.0f64, |m, &l| m.max(l.abs()));
    let floor = if largest > 0.0 {
        SINGULAR_VALUE_FLOOR
    } else {
        0.0
    };

    let mut x = vec![0.0; n];
    for k in 0..n {
        if eigenvalues[k].abs() <= floor {
            continue;
        }
        let projection: f64 = (0..n).map(|i| v[i * n + k] * rhs[i]).sum();
        let scale = projection / eigenvalues[k];
        for i in 0..n {
            x[i] += scale * v[i * n + k];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix_eigenvalues_come_out_ascending() {
        let mut m = [[0.0; 3]; 3];
        m[0][0] = 3.0;
        m[1][1] = 1.0;
        m[2][2] = 2.0;
        let (d, _) = symmetric_eigen(&m, 3);
        assert_eq!([d[0], d[1], d[2]], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn two_by_two_rotation_is_recovered() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let mut m = [[0.0; 3]; 3];
        m[0][0] = 2.0;
        m[0][1] = 1.0;
        m[1][0] = 1.0;
        m[1][1] = 2.0;
        let (d, v) = symmetric_eigen(&m, 2);
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        // Reconstruct V diag(d) V^T.
        for i in 0..2 {
            for j in 0..2 {
                let r: f64 = (0..2).map(|k| v[i][k] * d[k] * v[j][k]).sum();
                assert!((r - m[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pseudo_inverse_solves_a_nonsingular_symmetric_system() {
        // [[2, 1], [1, 2]] x = [3, 3]  =>  x = [1, 1].
        let x = symmetric_pseudo_inverse_solve(vec![2.0, 1.0, 1.0, 2.0], &[3.0, 3.0], 2);
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    /// A rank-1 matrix: the null direction is dropped, so the answer is the
    /// minimum-norm one rather than any solution with a null-space component.
    #[test]
    fn pseudo_inverse_returns_the_minimum_norm_solution() {
        // [[1, 1], [1, 1]] x = [2, 2] has solutions x = [1 + t, 1 - t];
        // the minimum-norm one is [1, 1].
        let x = symmetric_pseudo_inverse_solve(vec![1.0, 1.0, 1.0, 1.0], &[2.0, 2.0], 2);
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    /// An indefinite matrix exercises the `σ = |λ|` correspondence: a negative
    /// eigenvalue must divide with its sign, not its magnitude.
    #[test]
    fn pseudo_inverse_handles_a_negative_eigenvalue() {
        // diag(2, -4) x = [2, 4]  =>  x = [1, -1].
        let x = symmetric_pseudo_inverse_solve(vec![2.0, 0.0, 0.0, -4.0], &[2.0, 4.0], 2);
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] + 1.0).abs() < 1e-12);
    }

    /// Every singular value at or below the `1e-8` floor is zeroed, so an
    /// all-zero matrix yields an all-zero solution rather than infinities.
    #[test]
    fn pseudo_inverse_of_an_all_zero_matrix_is_zero() {
        let x = symmetric_pseudo_inverse_solve(vec![0.0; 4], &[1.0, 2.0], 2);
        assert_eq!(x, vec![0.0, 0.0]);
    }

    /// A matrix whose only eigenvalue sits under the floor is treated as rank 0.
    #[test]
    fn pseudo_inverse_drops_singular_values_at_or_below_the_floor() {
        let x = symmetric_pseudo_inverse_solve(vec![1e-8, 0.0, 0.0, 1e-9], &[1.0, 1.0], 2);
        assert_eq!(x, vec![0.0, 0.0]);

        let kept = symmetric_pseudo_inverse_solve(vec![1.1e-8, 0.0, 0.0, 0.0], &[1.0, 1.0], 2);
        assert!((kept[0] - 1.0 / 1.1e-8).abs() < 1.0);
        assert_eq!(kept[1], 0.0);
    }
}
