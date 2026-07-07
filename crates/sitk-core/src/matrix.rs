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
}
