//! Minimal dense linear algebra for the physical-space model.
//!
//! Only what the [`Image`](crate::Image) coordinate transforms need: a
//! row-major `n x n` matrix inverse and matrix/vector products. Kept dependency
//! -free for Phase 0; a heavier linear-algebra crate can replace this once the
//! registration framework needs one.

/// Multiply a row-major `n x n` matrix by an `n`-vector: `out = m * v`.
pub fn mat_vec(m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(m.len(), n * n);
    debug_assert_eq!(v.len(), n);
    let mut out = vec![0.0; n];
    for (r, out_r) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (c, &vc) in v.iter().enumerate() {
            acc += m[r * n + c] * vc;
        }
        *out_r = acc;
    }
    out
}

/// Multiply two row-major `n x n` matrices: `out = a * b`.
pub fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * n);
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for k in 0..n {
            let ark = a[r * n + k];
            if ark == 0.0 {
                continue;
            }
            for c in 0..n {
                out[r * n + c] += ark * b[k * n + c];
            }
        }
    }
    out
}

/// Invert a row-major `n x n` matrix via Gauss–Jordan elimination with partial
/// pivoting. Returns `None` if the matrix is singular (to within `eps`).
pub fn invert(m: &[f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(m.len(), n * n);
    // Augmented [ m | I ], row-major with 2n columns.
    let w = 2 * n;
    let mut a = vec![0.0f64; n * w];
    for r in 0..n {
        for c in 0..n {
            a[r * w + c] = m[r * n + c];
        }
        a[r * w + n + r] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: largest magnitude in this column at or below the diagonal.
        let mut pivot = col;
        let mut best = a[col * w + col].abs();
        for r in (col + 1)..n {
            let val = a[r * w + col].abs();
            if val > best {
                best = val;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            for c in 0..w {
                a.swap(col * w + c, pivot * w + c);
            }
        }

        let diag = a[col * w + col];
        for c in 0..w {
            a[col * w + c] /= diag;
        }

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * w + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..w {
                a[r * w + c] -= factor * a[col * w + c];
            }
        }
    }

    let mut inv = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            inv[r * n + c] = a[r * w + n + c];
        }
    }
    Some(inv)
}

/// The row-major `n x n` identity matrix.
pub fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_inverts_to_identity() {
        let i3 = identity(3);
        let inv = invert(&i3, 3).unwrap();
        for (a, b) in inv.iter().zip(i3.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_times_original_is_identity() {
        // A non-trivial 2x2 rotation+scale.
        let m = [1.2, -0.4, 0.3, 0.9];
        let inv = invert(&m, 2).unwrap();
        // m * inv should be identity.
        let mut prod = [0.0; 4];
        for r in 0..2 {
            for c in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += m[r * 2 + k] * inv[k * 2 + c];
                }
                prod[r * 2 + c] = acc;
            }
        }
        let id = identity(2);
        for (a, b) in prod.iter().zip(id.iter()) {
            assert!((a - b).abs() < 1e-12, "prod={prod:?}");
        }
    }

    #[test]
    fn singular_matrix_returns_none() {
        let m = [1.0, 2.0, 2.0, 4.0];
        assert!(invert(&m, 2).is_none());
    }

    #[test]
    fn mat_vec_basic() {
        let m = [2.0, 0.0, 0.0, 3.0];
        let v = [5.0, 7.0];
        assert_eq!(mat_vec(&m, &v, 2), vec![10.0, 21.0]);
    }

    #[test]
    fn matmul_identity_is_noop() {
        let m = [1.2, -0.4, 0.3, 0.9];
        assert_eq!(matmul(&identity(2), &m, 2), m.to_vec());
        assert_eq!(matmul(&m, &identity(2), 2), m.to_vec());
    }

    #[test]
    fn matmul_matches_manual_product() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]].
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(matmul(&a, &b, 2), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_composes_like_mat_vec() {
        // (A*B)*v == A*(B*v) for a 3×3 example.
        let a = [1.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 1.0];
        let b = [2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 3.0, 1.0];
        let v = [1.0, -2.0, 0.5];
        let lhs = mat_vec(&matmul(&a, &b, 3), &v, 3);
        let rhs = mat_vec(&a, &mat_vec(&b, &v, 3), 3);
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            assert!((l - r).abs() < 1e-12, "{lhs:?} vs {rhs:?}");
        }
    }
}
