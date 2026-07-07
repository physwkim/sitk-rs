//! Continuous-index image interpolation, shared by resampling and registration.
//!
//! An image's pixels are sampled at a *continuous index* (fractional grid
//! coordinate). All functions operate on a flat `f64` buffer plus its `size` and
//! first-index-fastest `strides`, so the same code serves resampling (cast to a
//! target pixel type afterwards) and metric evaluation (needs values and spatial
//! derivatives of the moving image).
//!
//! Boundary conventions are ITK's, verified against the v6 source:
//! `ImageFunction::IsInsideBuffer` treats a pixel-centred continuous index as
//! inside on `[-0.5, size − 0.5)` per axis (`itkImageFunction.hxx`); linear
//! interpolation clamps each neighbour index into `[0, size − 1]`
//! (`itkLinearInterpolateImageFunction.hxx`); nearest-neighbour rounds half up
//! (`Math::RoundHalfIntegerUp`, `itkImageBase.h`).

use sitk_core::matrix;

/// First-index-fastest strides for a size vector.
pub fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// ITK's continuous-index inside-buffer test: pixel-centred coverage
/// `[-0.5, size − 0.5)` on every axis.
pub fn is_inside(cindex: &[f64], size: &[usize]) -> bool {
    cindex
        .iter()
        .zip(size.iter())
        .all(|(&c, &s)| c >= -0.5 && c < s as f64 - 0.5)
}

/// Nearest-neighbour sample, or `None` if `cindex` is outside the buffer.
pub fn nearest_at(buf: &[f64], size: &[usize], strides: &[usize], cindex: &[f64]) -> Option<f64> {
    if !is_inside(cindex, size) {
        return None;
    }
    let mut offset = 0usize;
    for d in 0..size.len() {
        // Round half up, matching ITK's RoundHalfIntegerUp.
        let mut i = (cindex[d] + 0.5).floor() as isize;
        i = i.clamp(0, size[d] as isize - 1);
        offset += i as usize * strides[d];
    }
    Some(buf[offset])
}

/// N-linear sample, or `None` if `cindex` is outside the buffer. Neighbour
/// indices are clamped into `[0, size − 1]` at the boundary, matching ITK.
pub fn linear_at(buf: &[f64], size: &[usize], strides: &[usize], cindex: &[f64]) -> Option<f64> {
    if !is_inside(cindex, size) {
        return None;
    }
    let dim = size.len();
    let mut base = vec![0isize; dim];
    let mut frac = vec![0.0f64; dim];
    for d in 0..dim {
        let f = cindex[d].floor();
        base[d] = f as isize;
        frac[d] = cindex[d] - f;
    }

    let mut acc = 0.0;
    for corner in 0..(1usize << dim) {
        let mut weight = 1.0;
        let mut offset = 0usize;
        for d in 0..dim {
            let bit = (corner >> d) & 1;
            weight *= if bit == 1 { frac[d] } else { 1.0 - frac[d] };
            let idx = (base[d] + bit as isize).clamp(0, size[d] as isize - 1) as usize;
            offset += idx * strides[d];
        }
        if weight != 0.0 {
            acc += weight * buf[offset];
        }
    }
    Some(acc)
}

/// N-linear sample together with the **exact** gradient of the linear
/// interpolant in continuous-index space, or `None` if `cindex` is outside the
/// buffer.
///
/// The returned `grad[j] = ∂(interpolated value)/∂cindex[j]`, computed by
/// differentiating the multilinear corner-weight product — so it is consistent
/// with the value returned by [`linear_at`] (unlike a finite-difference of raw
/// pixels). Neighbour indices are clamped into `[0, size − 1]`, matching
/// [`linear_at`]; the gradient is piecewise-constant within a grid cell and
/// jumps at cell boundaries, as any linear-interpolant gradient does.
pub fn linear_value_and_gradient(
    buf: &[f64],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
) -> Option<(f64, Vec<f64>)> {
    if !is_inside(cindex, size) {
        return None;
    }
    let dim = size.len();
    let mut frac = vec![0.0f64; dim];
    let mut base = vec![0isize; dim];
    for d in 0..dim {
        let f = cindex[d].floor();
        base[d] = f as isize;
        frac[d] = cindex[d] - f;
    }

    let mut value = 0.0;
    let mut grad = vec![0.0f64; dim];
    for corner in 0..(1usize << dim) {
        let mut offset = 0usize;
        let mut weight = 1.0;
        for d in 0..dim {
            let bit = (corner >> d) & 1;
            weight *= if bit == 1 { frac[d] } else { 1.0 - frac[d] };
            let idx = (base[d] + bit as isize).clamp(0, size[d] as isize - 1) as usize;
            offset += idx * strides[d];
        }
        let b = buf[offset];
        value += weight * b;
        // ∂value/∂cindex[j]: swap axis j's weight factor for its ±1 derivative.
        for (j, gj) in grad.iter_mut().enumerate() {
            let mut w_without_j = 1.0;
            for (d, &fr) in frac.iter().enumerate() {
                if d == j {
                    continue;
                }
                let bit = (corner >> d) & 1;
                w_without_j *= if bit == 1 { fr } else { 1.0 - fr };
            }
            let sign = if (corner >> j) & 1 == 1 { 1.0 } else { -1.0 };
            *gj += sign * w_without_j * b;
        }
    }
    Some((value, grad))
}

/// `D · diag(spacing)`, row-major: maps a continuous index to a physical
/// displacement from the origin.
pub fn index_to_physical_matrix(direction: &[f64], spacing: &[f64], dim: usize) -> Vec<f64> {
    let mut m = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            m[r * dim + c] = direction[r * dim + c] * spacing[c];
        }
    }
    m
}

/// `diag(1/spacing) · D⁻¹`, row-major, or `None` if `D` is singular: maps a
/// physical displacement from the origin to a continuous index.
pub fn physical_to_index_matrix(
    direction: &[f64],
    spacing: &[f64],
    dim: usize,
) -> Option<Vec<f64>> {
    let inv = matrix::invert(direction, dim)?;
    let mut m = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            m[r * dim + c] = inv[r * dim + c] / spacing[r];
        }
    }
    Some(m)
}

/// `origin + M · index`.
pub fn affine_apply(m: &[f64], index: &[f64], origin: &[f64], dim: usize) -> Vec<f64> {
    let mv = matrix::mat_vec(m, index, dim);
    (0..dim).map(|d| origin[d] + mv[d]).collect()
}
