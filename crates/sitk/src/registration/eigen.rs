//! Symmetric eigen-decomposition shared by [`crate::registration::landmark`] (Horn's
//! quaternion profile matrix) and [`crate::registration::centered_versor`] (second
//! central-moment matrices).

use crate::core::matrix;

/// Eigen-decompose a symmetric `n x n` row-major matrix via the classical
/// cyclic Jacobi rotation method. Returns `(eigenvalues, eigenvectors)`,
/// where `eigenvectors` is row-major `n x n` and column `j` is the unit
/// eigenvector for `eigenvalues[j]`.
pub(crate) fn jacobi_eigen_symmetric(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = matrix::identity(n);

    for _sweep in 0..100 {
        let mut off_diag_sq = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diag_sq += a[p * n + q] * a[p * n + q];
            }
        }
        if off_diag_sq.sqrt() < 1e-14 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[k * n + p];
                        let akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[p * n + k] = a[k * n + p];
                        a[k * n + q] = s * akp + c * akq;
                        a[q * n + k] = a[k * n + q];
                    }
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_eigenvalues_of_a_diagonal_matrix() {
        let a = vec![9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0];
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&a, 3);
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 4.0).abs() < 1e-10);
        assert!((sorted[2] - 9.0).abs() < 1e-10);

        // Eigenvectors are orthonormal.
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3)
                    .map(|k| eigenvectors[k * 3 + i] * eigenvectors[k * 3 + j])
                    .sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-10, "not orthonormal at ({i},{j})");
            }
        }
    }

    #[test]
    fn recovers_eigenvector_of_a_general_symmetric_matrix() {
        // A @ v = lambda @ v must hold for every returned (eigenvalue, eigenvector) pair.
        let a = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&a, 3);
        for j in 0..3 {
            let v: Vec<f64> = (0..3).map(|r| eigenvectors[r * 3 + j]).collect();
            let av = matrix::mat_vec(&a, &v, 3);
            for d in 0..3 {
                assert!(
                    (av[d] - eigenvalues[j] * v[d]).abs() < 1e-9,
                    "A*v != lambda*v for eigenvalue {}: {:?} vs {:?}",
                    eigenvalues[j],
                    av,
                    v
                );
            }
        }
    }
}
