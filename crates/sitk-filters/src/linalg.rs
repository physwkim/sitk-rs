//! Small dense linear algebra shared by filters that need an eigendecomposition
//! of a symmetric 2×2 or 3×3 matrix.

/// The widest matrix any caller needs: ITK's `SymmetricSecondRankTensor` and
/// `ShapeLabelObject` are both at most 3-dimensional.
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
}
