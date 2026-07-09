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

/// Nearest-neighbour sample and its gradient at continuous index `cindex`, or
/// `None` if outside the buffer. The value matches [`nearest_at`]; the
/// gradient is the exact gradient of the piecewise-constant nearest-neighbour
/// interpolant, which is zero almost everywhere (it is formally undefined
/// exactly at the round-half-up cell boundary, where this still returns the
/// interior zero gradient rather than a delta).
pub fn nearest_value_and_gradient(
    buf: &[f64],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let value = nearest_at(buf, size, strides, cindex)?;
    Some((value, vec![0.0; size.len()]))
}

/// Convert raw samples into cubic (order-3) B-spline interpolation
/// coefficients (Unser 1993/1999 causal + anticausal recursive filter, mirror
/// boundary) so that [`bspline_value_and_gradient`] reproduces the samples
/// exactly at integer indices. Ports `itk::BSplineDecompositionImageFilter`
/// with the spline order fixed at 3 (SimpleITK's `sitkBSpline` /
/// `sitkBSpline3` default, matching [`Interpolator::BSpline`]). ITK's
/// tolerance-gated "accelerated" causal-sum short-circuit is skipped in favor
/// of the exact closed-form mirror-boundary initialization it falls back to
/// for short lines — exact (not an approximation) for any length, just not
/// the fast path for long ones.
///
/// [`Interpolator::BSpline`]: crate::resample::Interpolator::BSpline
pub fn bspline_coefficients(buf: &[f64], size: &[usize], strides: &[usize]) -> Vec<f64> {
    let mut coeffs = buf.to_vec();
    let pole = 3.0_f64.sqrt() - 2.0;
    for axis in 0..size.len() {
        if size[axis] == 1 {
            continue;
        }
        for_each_line(&mut coeffs, size, strides, axis, |line| {
            bspline3_filter_1d(line, pole);
        });
    }
    coeffs
}

/// Apply `f` to each 1-D line of `values` along `axis` in place (gather,
/// call, scatter back). Shared by [`bspline_coefficients`]'s per-dimension
/// decomposition pass.
fn for_each_line(
    values: &mut [f64],
    size: &[usize],
    strides: &[usize],
    axis: usize,
    mut f: impl FnMut(&mut [f64]),
) {
    let other: Vec<usize> = (0..size.len()).filter(|&d| d != axis).collect();
    let other_sizes: Vec<usize> = other.iter().map(|&d| size[d]).collect();
    let total_lines: usize = other_sizes.iter().product();
    let n = size[axis];
    let line_stride = strides[axis];

    let mut scratch = vec![0.0f64; n];
    let mut oidx = vec![0usize; other.len()];
    for _ in 0..total_lines {
        let base: usize = other
            .iter()
            .zip(oidx.iter())
            .map(|(&d, &i)| i * strides[d])
            .sum();
        for (k, s) in scratch.iter_mut().enumerate() {
            *s = values[base + k * line_stride];
        }
        f(&mut scratch);
        for (k, &s) in scratch.iter().enumerate() {
            values[base + k * line_stride] = s;
        }
        for j in 0..oidx.len() {
            oidx[j] += 1;
            if oidx[j] < other_sizes[j] {
                break;
            }
            oidx[j] = 0;
        }
    }
}

/// Apply the mirror-boundary cubic B-spline decomposition filter to one 1-D
/// line in place. Ports
/// `itk::BSplineDecompositionImageFilter::DataToCoefficients1D` /
/// `SetInitialCausalCoefficient` / `SetInitialAntiCausalCoefficient` for the
/// single-pole (order-3) case.
fn bspline3_filter_1d(line: &mut [f64], z: f64) {
    let n = line.len();
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in line.iter_mut() {
        *v *= gain;
    }

    // Causal initialization: exact mirror-boundary sum (ITK's non-accelerated
    // "full loop" branch, valid for any line length).
    let iz = 1.0 / z;
    let mut zn = z;
    let mut z2n = z.powi(n as i32 - 1);
    let mut sum = line[0] + z2n * line[n - 1];
    z2n = z2n * z2n * iz;
    for &lk in line.iter().skip(1).take(n.saturating_sub(2)) {
        sum += (zn + z2n) * lk;
        zn *= z;
        z2n *= iz;
    }
    line[0] = sum / (1.0 - zn * zn);

    // Causal recursion.
    for k in 1..n {
        line[k] += z * line[k - 1];
    }

    // Anticausal initialization (mirror boundary).
    line[n - 1] = (z / (z * z - 1.0)) * (z * line[n - 2] + line[n - 1]);

    // Anticausal recursion.
    for k in (0..=(n - 2)).rev() {
        line[k] = z * (line[k + 1] - line[k]);
    }
}

/// Cubic (order-3) B-spline sample and its exact index-space gradient at
/// continuous index `cindex`, from precomputed coefficients
/// ([`bspline_coefficients`]), or `None` if outside the buffer. Ports
/// `itk::BSplineInterpolateImageFunction<..., SplineOrder=3>`'s weight and
/// mirror-boundary logic
/// (`EvaluateValueAndDerivativeAtContinuousIndexInternal`); like
/// [`linear_value_and_gradient`], the returned gradient is in
/// **continuous-index** space — spacing/direction are folded in uniformly by
/// the caller (e.g. `MovingImage::value_and_physical_gradient`), not
/// per-kernel here.
pub fn bspline_value_and_gradient(
    coeffs: &[f64],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
) -> Option<(f64, Vec<f64>)> {
    if !is_inside(cindex, size) {
        return None;
    }
    let dim = size.len();
    let mut idx = vec![[0usize; 4]; dim];
    let mut w = vec![[0.0f64; 4]; dim];
    let mut dw = vec![[0.0f64; 4]; dim];

    for d in 0..dim {
        let base = cindex[d].floor() as isize - 1;
        let n = size[d] as isize;
        for (k, ik) in idx[d].iter_mut().enumerate() {
            let mut i = base + k as isize;
            if n == 1 {
                i = 0;
            } else {
                if i < 0 {
                    i = -i;
                }
                if i > n - 1 {
                    i = 2 * (n - 1) - i;
                }
            }
            *ik = i as usize;
        }

        // Value weights: cubic B-spline basis at the four taps
        // [base, base+1, base+2, base+3], `frac = cindex[d] - (base + 1)`.
        let frac = cindex[d] - (base + 1) as f64;
        let w3 = (1.0 / 6.0) * frac * frac * frac;
        let w0 = 1.0 / 6.0 + 0.5 * frac * (frac - 1.0) - w3;
        let w2 = frac + w0 - 2.0 * w3;
        let w1 = 1.0 - w0 - w2 - w3;
        w[d] = [w0, w1, w2, w3];

        // Derivative weights: the order-2 spline evaluated at `cindex + 0.5`
        // (ITK's `SetDerivativeWeights`, `derivativeSplineOrder == 2` case).
        let a = frac - 0.5;
        let dw2 = 0.75 - a * a;
        let dw3 = 0.5 * (a - dw2 + 1.0);
        let dw1 = 1.0 - dw2 - dw3;
        dw[d] = [-dw1, dw1 - dw2, dw2 - dw3, dw3];
    }

    let mut value = 0.0;
    let mut grad = vec![0.0f64; dim];
    for corner in 0..4usize.pow(dim as u32) {
        let mut rem = corner;
        let mut tap = vec![0usize; dim];
        for t in tap.iter_mut() {
            *t = rem % 4;
            rem /= 4;
        }
        let mut offset = 0usize;
        let mut wprod = 1.0;
        for d in 0..dim {
            offset += idx[d][tap[d]] * strides[d];
            wprod *= w[d][tap[d]];
        }
        let c = coeffs[offset];
        value += wprod * c;
        for (q, gq) in grad.iter_mut().enumerate() {
            let mut gprod = 1.0;
            for d in 0..dim {
                gprod *= if d == q { dw[d][tap[d]] } else { w[d][tap[d]] };
            }
            *gq += gprod * c;
        }
    }
    Some((value, grad))
}

/// Continuous-index width (in pixels) of the [`Interpolator::Gaussian`]
/// kernel, matching SimpleITK's `sitkGaussian` preset (`sigma = 0.8 *
/// spacing`): this module works entirely in continuous-index space, where the
/// spacing cancels out of ITK's `sigma / spacing` scaling factor, leaving
/// this constant regardless of the image's physical spacing.
///
/// [`Interpolator::Gaussian`]: crate::resample::Interpolator::Gaussian
pub const GAUSSIAN_SIGMA: f64 = 0.8;
/// Cutoff distance, in units of [`GAUSSIAN_SIGMA`], beyond which a pixel's
/// contribution is ignored. Matches SimpleITK's `sitkGaussian` preset
/// (`alpha = 4.0`).
pub const GAUSSIAN_ALPHA: f64 = 4.0;

/// `erf(x)` via the Abramowitz & Stegun 7.1.26 rational approximation (max
/// absolute error ≈ 1.5e-7). No external numerics crate is available here,
/// and this is far more precise than the finite-difference tolerances the
/// Gaussian kernel's own tests check against.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Gaussian-weighted sample and its exact index-space gradient at continuous
/// index `cindex`, or `None` if outside the buffer. Ports
/// `itk::GaussianInterpolateImageFunction`'s error-function pixel weighting
/// (`EvaluateAtContinuousIndex`) at the fixed [`GAUSSIAN_SIGMA`] /
/// [`GAUSSIAN_ALPHA`] width; like [`linear_value_and_gradient`], the gradient
/// is in continuous-index space. Unlike the other three kernels, this one is
/// **not interpolating** — it is a local weighted average, so it does not in
/// general reproduce the original samples exactly at integer indices.
pub fn gaussian_value_and_gradient(
    buf: &[f64],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
) -> Option<(f64, Vec<f64>)> {
    if !is_inside(cindex, size) {
        return None;
    }
    let dim = size.len();
    let scaling_factor = 1.0 / (std::f64::consts::SQRT_2 * GAUSSIAN_SIGMA);
    let cutoff = GAUSSIAN_SIGMA * GAUSSIAN_ALPHA;

    let mut lo = vec![0usize; dim];
    let mut region_size = vec![0usize; dim];
    let mut erf_arrays = Vec::with_capacity(dim);
    let mut gerf_arrays = Vec::with_capacity(dim);
    for d in 0..dim {
        let begin = (cindex[d] + 0.5 - cutoff).floor().max(0.0) as usize;
        let end = ((cindex[d] + 0.5 + cutoff).ceil() as usize).min(size[d]);
        lo[d] = begin;
        let n = end.saturating_sub(begin);
        region_size[d] = n;

        let mut t = (begin as f64 - 0.5 - cindex[d]) * scaling_factor;
        let mut e_last = erf(t);
        let mut g_last = std::f64::consts::FRAC_2_SQRT_PI * (-t * t).exp();
        let mut erf_arr = vec![0.0f64; n];
        let mut gerf_arr = vec![0.0f64; n];
        for i in 0..n {
            t += scaling_factor;
            let e_now = erf(t);
            erf_arr[i] = e_now - e_last;
            let g_now = std::f64::consts::FRAC_2_SQRT_PI * (-t * t).exp();
            gerf_arr[i] = g_now - g_last;
            e_last = e_now;
            g_last = g_now;
        }
        erf_arrays.push(erf_arr);
        gerf_arrays.push(gerf_arr);
    }

    let total: usize = region_size.iter().product();
    let mut sum_me = 0.0;
    let mut sum_m = 0.0;
    let mut dsum_me = vec![0.0f64; dim];
    let mut dsum_m = vec![0.0f64; dim];
    let mut ridx = vec![0usize; dim];
    for _ in 0..total {
        let mut w = 1.0;
        let mut offset = 0usize;
        for d in 0..dim {
            w *= erf_arrays[d][ridx[d]];
            offset += (lo[d] + ridx[d]) * strides[d];
        }
        let v = buf[offset];
        sum_me += v * w;
        sum_m += w;
        for q in 0..dim {
            let mut dw = 1.0;
            for d in 0..dim {
                dw *= if d == q {
                    gerf_arrays[d][ridx[d]]
                } else {
                    erf_arrays[d][ridx[d]]
                };
            }
            dsum_me[q] += v * dw;
            dsum_m[q] += dw;
        }
        for d in 0..dim {
            ridx[d] += 1;
            if ridx[d] < region_size[d] {
                break;
            }
            ridx[d] = 0;
        }
    }

    let value = sum_me / sum_m;
    let mut grad = vec![0.0f64; dim];
    for q in 0..dim {
        grad[q] = (value * dsum_m[q] - dsum_me[q]) / sum_m * scaling_factor;
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

#[cfg(test)]
mod tests {
    use super::*;

    // f(x, y) = 3x + 5y, exact for both linear and cubic B-spline
    // interpolation (a degree-1 polynomial is reproduced exactly by any
    // interpolating spline of order >= 1).
    fn ramp(w: usize, h: usize) -> (Vec<f64>, Vec<usize>) {
        let size = vec![w, h];
        let mut buf = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = 3.0 * x as f64 + 5.0 * y as f64;
            }
        }
        (buf, size)
    }

    #[test]
    fn nearest_gradient_is_zero_interior() {
        let (buf, size) = ramp(6, 6);
        let strides = strides(&size);
        // 0.3 is safely away from the 0.5 round-half-up boundary.
        let (value, grad) = nearest_value_and_gradient(&buf, &size, &strides, &[2.3, 3.3]).unwrap();
        assert_eq!(value, buf[3 * 6 + 2]);
        assert_eq!(grad, vec![0.0, 0.0]);
    }

    #[test]
    fn nearest_gradient_matches_finite_difference_interior() {
        let (buf, size) = ramp(6, 6);
        let strides = strides(&size);
        let h = 1e-3;
        let c0 = [2.3, 3.3];
        for k in 0..2 {
            let mut cp = c0;
            cp[k] += h;
            let mut cm = c0;
            cm[k] -= h;
            let vp = nearest_at(&buf, &size, &strides, &cp).unwrap();
            let vm = nearest_at(&buf, &size, &strides, &cm).unwrap();
            let fd = (vp - vm) / (2.0 * h);
            assert!(fd.abs() < 1e-9, "axis {k}: fd {fd}");
        }
    }

    #[test]
    fn bspline_reproduces_ramp_exactly_away_from_boundary() {
        // The coefficient recursion's mirror-boundary correction decays only
        // geometrically (pole |z| ≈ 0.268 per pixel of distance), not to
        // exactly zero, so a straight ramp is reproduced exactly only in the
        // limit of being far (in e-foldings of that decay) from every edge.
        // A 40x40 image with a query point near the center puts ~20 pixels
        // of decay between it and the nearest edge (0.268^20 ≈ 1e-11),
        // pushing the boundary artifact below double-precision tolerance.
        let (buf, size) = ramp(40, 40);
        let strides = strides(&size);
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        let cindex = [20.4, 19.6];
        let (value, _) = bspline_value_and_gradient(&coeffs, &size, &strides, &cindex).unwrap();
        let expected = 3.0 * cindex[0] + 5.0 * cindex[1];
        assert!(
            (value - expected).abs() < 1e-9,
            "value {value} vs {expected}"
        );
    }

    #[test]
    fn bspline_reproduces_samples_at_integer_indices() {
        let (buf, size) = ramp(10, 10);
        let strides = strides(&size);
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        for y in 2..8 {
            for x in 2..8 {
                let cindex = [x as f64, y as f64];
                let (value, _) =
                    bspline_value_and_gradient(&coeffs, &size, &strides, &cindex).unwrap();
                let expected = buf[y * 10 + x];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "({x},{y}): value {value} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn bspline_gradient_matches_finite_difference() {
        let (buf, size) = ramp(12, 12);
        let strides = strides(&size);
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        let c0 = [5.4, 6.7];
        let analytic = bspline_value_and_gradient(&coeffs, &size, &strides, &c0)
            .unwrap()
            .1;
        let h = 1e-4;
        for k in 0..2 {
            let mut cp = c0;
            cp[k] += h;
            let mut cm = c0;
            cm[k] -= h;
            let vp = bspline_value_and_gradient(&coeffs, &size, &strides, &cp)
                .unwrap()
                .0;
            let vm = bspline_value_and_gradient(&coeffs, &size, &strides, &cm)
                .unwrap()
                .0;
            let fd = (vp - vm) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-6,
                "axis {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    fn gaussian_blob(w: usize, h: usize, cx: f64, cy: f64, sigma: f64) -> (Vec<f64>, Vec<usize>) {
        let size = vec![w, h];
        let mut buf = vec![0.0f64; w * h];
        let s2 = 2.0 * sigma * sigma;
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                buf[y * w + x] = (-(dx * dx + dy * dy) / s2).exp();
            }
        }
        (buf, size)
    }

    #[test]
    fn gaussian_matches_analytic_blob_near_the_peak() {
        // Unlike linear/BSpline, the Gaussian kernel is a local weighted
        // average, not an interpolating spline — it does not reproduce
        // samples exactly even at integer indices. For a blob much wider
        // (sigma=20) than the kernel's own width (sigma=0.8), the smoothing
        // bias near the peak is small and consistent (~0.0018 here,
        // verified independently in Python), so the interpolant should still
        // track the analytic Gaussian closely.
        let sigma = 20.0;
        let (buf, size) = gaussian_blob(60, 60, 30.0, 30.0, sigma);
        let strides = strides(&size);
        let s2 = 2.0 * sigma * sigma;
        for &cindex in &[[30.0, 30.0], [30.3, 29.6], [31.1, 30.4]] {
            let (value, _) = gaussian_value_and_gradient(&buf, &size, &strides, &cindex).unwrap();
            let (dx, dy) = (cindex[0] - 30.0, cindex[1] - 30.0);
            let expected = (-(dx * dx + dy * dy) / s2).exp();
            assert!(
                (value - expected).abs() < 3e-3,
                "cindex {cindex:?}: value {value} vs analytic {expected}"
            );
        }
    }

    #[test]
    fn gaussian_gradient_matches_finite_difference() {
        // Avoid a cindex where `cindex + 0.5 + cutoff` lands exactly on an
        // integer: the interpolation region's pixel count is then only
        // one-sided continuous in cindex (adding a vanishingly-weighted
        // pixel on one side but not the other), which is an artifact of the
        // hard cutoff radius, not the analytic gradient — a real central
        // difference straddling that exact tie sees a spurious jump.
        let (buf, size) = gaussian_blob(40, 40, 20.0, 20.0, 6.0);
        let strides = strides(&size);
        let c0 = [21.37, 18.71];
        let analytic = gaussian_value_and_gradient(&buf, &size, &strides, &c0)
            .unwrap()
            .1;
        let h = 1e-4;
        for k in 0..2 {
            let mut cp = c0;
            cp[k] += h;
            let mut cm = c0;
            cm[k] -= h;
            let vp = gaussian_value_and_gradient(&buf, &size, &strides, &cp)
                .unwrap()
                .0;
            let vm = gaussian_value_and_gradient(&buf, &size, &strides, &cm)
                .unwrap()
                .0;
            let fd = (vp - vm) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-5,
                "axis {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    #[test]
    fn gaussian_respects_is_inside_bounds() {
        let (buf, size) = gaussian_blob(10, 10, 5.0, 5.0, 2.0);
        let strides = strides(&size);
        // is_inside is the shared `[-0.5, size - 0.5)` rule: size=10 means
        // the valid range along an axis is `[-0.5, 9.5)`, so 9.5 itself is
        // outside (not merely "far away").
        assert!(gaussian_value_and_gradient(&buf, &size, &strides, &[9.5, 5.0]).is_none());
        assert!(gaussian_value_and_gradient(&buf, &size, &strides, &[5.0, 5.0]).is_some());
    }
}
