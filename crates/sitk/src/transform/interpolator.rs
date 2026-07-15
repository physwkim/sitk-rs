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

use crate::core::{Scalar, coord};

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
pub fn nearest_at<T: Scalar>(
    buf: &[T],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
) -> Option<f64> {
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
    Some(buf[offset].as_f64())
}

/// N-linear sample, or `None` if `cindex` is outside the buffer. Neighbour
/// indices are clamped into `[0, size − 1]` at the boundary, matching ITK.
pub fn linear_at<T: Scalar>(
    buf: &[T],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
) -> Option<f64> {
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
            acc += weight * buf[offset].as_f64();
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
pub fn linear_value_and_gradient<T: Scalar>(
    buf: &[T],
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
        let b = buf[offset].as_f64();
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
pub fn nearest_value_and_gradient<T: Scalar>(
    buf: &[T],
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
/// [`Interpolator::BSpline`]: crate::transform::resample::Interpolator::BSpline
pub fn bspline_coefficients<T: Scalar>(buf: &[T], size: &[usize], strides: &[usize]) -> Vec<f64> {
    let mut coeffs: Vec<f64> = buf.iter().map(|v| v.as_f64()).collect();
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
/// [`Interpolator::Gaussian`]: crate::transform::resample::Interpolator::Gaussian
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
pub fn gaussian_value_and_gradient<T: Scalar>(
    buf: &[T],
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
        let v = buf[offset].as_f64();
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

/// Kernel radius for [`windowed_sinc_value_and_gradient`], matching
/// SimpleITK's `sitk*WindowedSinc` presets (`WindowingRadius` in
/// `sitkCreateInterpolator.hxx`): `2 * `[`WINDOWED_SINC_RADIUS`] taps per axis.
pub const WINDOWED_SINC_RADIUS: usize = 5;

/// Window function paired with the sinc kernel by
/// [`windowed_sinc_value_and_gradient`], one per SimpleITK `sitk*WindowedSinc`
/// preset (`itk::Function::*WindowFunction`, `itkWindowedSincInterpolateImageFunction.h`).
/// All formulas below are exact ports, evaluated at the fixed
/// [`WINDOWED_SINC_RADIUS`] (`m` in the ITK formulas).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SincWindow {
    /// `w(x) = 0.54 + 0.46 cos(pi x / m)` (`sitkHammingWindowedSinc`).
    Hamming,
    /// `w(x) = cos(pi x / (2 m))` (`sitkCosineWindowedSinc`).
    Cosine,
    /// `w(x) = 1 - (x / m)^2` (`sitkWelchWindowedSinc`).
    Welch,
    /// `w(x) = sinc(x / m)`, i.e. `sin(pi x / m) / (pi x / m)`
    /// (`sitkLanczosWindowedSinc`).
    Lanczos,
    /// `w(x) = 0.42 + 0.5 cos(pi x / m) + 0.08 cos(2 pi x / m)`
    /// (`sitkBlackmanWindowedSinc`).
    Blackman,
}

impl SincWindow {
    /// `w(x)`.
    fn weight(self, x: f64) -> f64 {
        let m = WINDOWED_SINC_RADIUS as f64;
        match self {
            SincWindow::Hamming => {
                let factor = std::f64::consts::PI / m;
                0.54 + 0.46 * (x * factor).cos()
            }
            SincWindow::Cosine => {
                let factor = std::f64::consts::PI / (2.0 * m);
                (x * factor).cos()
            }
            SincWindow::Welch => {
                let factor = 1.0 / (m * m);
                1.0 - x * factor * x
            }
            SincWindow::Lanczos => {
                if x == 0.0 {
                    1.0
                } else {
                    let factor = std::f64::consts::PI / m;
                    let z = factor * x;
                    z.sin() / z
                }
            }
            SincWindow::Blackman => {
                let factor1 = std::f64::consts::PI / m;
                let factor2 = 2.0 * std::f64::consts::PI / m;
                0.42 + 0.5 * (x * factor1).cos() + 0.08 * (x * factor2).cos()
            }
        }
    }

    /// `dw/dx`. Not part of ITK (which never differentiates this kernel);
    /// each branch is the plain derivative of the corresponding [`weight`]
    /// formula, needed so [`windowed_sinc_value_and_gradient`] can supply the
    /// same value+gradient contract every other kernel in this module gives.
    ///
    /// [`weight`]: SincWindow::weight
    fn dweight(self, x: f64) -> f64 {
        let m = WINDOWED_SINC_RADIUS as f64;
        match self {
            SincWindow::Hamming => {
                let factor = std::f64::consts::PI / m;
                -0.46 * factor * (x * factor).sin()
            }
            SincWindow::Cosine => {
                let factor = std::f64::consts::PI / (2.0 * m);
                -factor * (x * factor).sin()
            }
            SincWindow::Welch => {
                let factor = 1.0 / (m * m);
                -2.0 * factor * x
            }
            SincWindow::Lanczos => {
                if x == 0.0 {
                    0.0
                } else {
                    let factor = std::f64::consts::PI / m;
                    let z = factor * x;
                    factor * (z.cos() * z - z.sin()) / (z * z)
                }
            }
            SincWindow::Blackman => {
                let factor1 = std::f64::consts::PI / m;
                let factor2 = 2.0 * std::f64::consts::PI / m;
                -0.5 * factor1 * (x * factor1).sin() - 0.08 * factor2 * (x * factor2).sin()
            }
        }
    }
}

/// `itk::WindowedSincInterpolateImageFunction::Sinc`: `sin(pi x) / (pi x)`,
/// `1` at `x == 0`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// `d(sinc)/dx`, `0` at `x == 0` (the removable singularity of `sinc`'s own
/// formula; `sinc` is even, so this is its correct value there, not merely a
/// guard).
fn dsinc(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        std::f64::consts::PI * (px.cos() * px - px.sin()) / (px * px)
    }
}

/// Windowed-sinc sample and its exact index-space gradient at continuous
/// index `cindex`, or `None` if outside the buffer. Ports
/// `itk::WindowedSincInterpolateImageFunction::EvaluateAtContinuousIndex` at
/// the fixed [`WINDOWED_SINC_RADIUS`] radius SimpleITK bakes into its five
/// `sitk*WindowedSinc` presets — `m_Radius` is a compile-time template
/// parameter in ITK, so unlike [`GAUSSIAN_SIGMA`](crate::transform::interpolator::GAUSSIAN_SIGMA)
/// it has no runtime setter to port.
///
/// Per axis, the kernel `K(t) = w(t) sinc(t)` (`w` from `window`) is sampled
/// at the `2 * `[`WINDOWED_SINC_RADIUS`] taps surrounding the query point,
/// exactly as the `.hxx` computes `xWeight`: **except** at `distance == 0`
/// (query exactly on a grid point), where ITK overrides the taps with a hard
/// `0`/`1` delta rather than relying on the general formula — `sinc` at a
/// nonzero integer is mathematically `0` but not exactly so in floating point
/// (`sin(pi n)` for a whole `n` isn't exactly `0`), so the override exists
/// purely to reproduce the sample bit-exactly, not because of a real
/// discontinuity.
///
/// ITK has no analytic gradient for this interpolator; the one computed here
/// always uses the smooth per-tap formula (`w'(t) sinc(t) + w(t) sinc'(t)`,
/// product rule), **without** the delta override, even when `distance == 0`.
/// This is deliberate, not an oversight: `K` is even in `t` (every `w` here is
/// even, and `sinc`/`d(sinc)` are each individually well-defined — see
/// [`sinc`]/[`dsinc`] — at `t == 0`) so `K` is `C^1` straight through
/// `distance == 0` with no kink for a delta-shaped gradient to model; the
/// delta only ever existed to suppress the value's floating-point noise.
///
/// Neighbour indices are clamped into `[0, size − 1]` per axis, matching
/// ITK's default `ZeroFluxNeumannBoundaryCondition` (`GetPixel`, edge
/// replication — verified against `itkZeroFluxNeumannBoundaryCondition.hxx`),
/// the same convention [`linear_at`] and [`linear_value_and_gradient`] use.
pub fn windowed_sinc_value_and_gradient<T: Scalar>(
    buf: &[T],
    size: &[usize],
    strides: &[usize],
    cindex: &[f64],
    window: SincWindow,
) -> Option<(f64, Vec<f64>)> {
    if !is_inside(cindex, size) {
        return None;
    }
    let dim = size.len();
    let radius = WINDOWED_SINC_RADIUS;
    let window_size = 2 * radius;

    let mut base_index = vec![0isize; dim];
    let mut distance = vec![0.0f64; dim];
    for d in 0..dim {
        let f = cindex[d].floor();
        base_index[d] = f as isize;
        distance[d] = cindex[d] - f;
    }

    let mut w = vec![vec![0.0f64; window_size]; dim];
    let mut dw = vec![vec![0.0f64; window_size]; dim];
    for d in 0..dim {
        if distance[d] == 0.0 {
            for (i, wi) in w[d].iter_mut().enumerate() {
                *wi = if i == radius - 1 { 1.0 } else { 0.0 };
            }
        } else {
            let mut x = distance[d] + radius as f64;
            for wi in w[d].iter_mut() {
                x -= 1.0;
                *wi = window.weight(x) * sinc(x);
            }
        }
        let mut x = distance[d] + radius as f64;
        for dwi in dw[d].iter_mut() {
            x -= 1.0;
            *dwi = window.dweight(x) * sinc(x) + window.weight(x) * dsinc(x);
        }
    }

    let mut value = 0.0;
    let mut grad = vec![0.0f64; dim];
    for corner in 0..window_size.pow(dim as u32) {
        let mut rem = corner;
        let mut tap = vec![0usize; dim];
        for t in tap.iter_mut() {
            *t = rem % window_size;
            rem /= window_size;
        }
        let mut offset = 0usize;
        let mut wprod = 1.0;
        for d in 0..dim {
            let k = tap[d] as isize - (radius as isize - 1);
            let idx = (base_index[d] + k).clamp(0, size[d] as isize - 1) as usize;
            offset += idx * strides[d];
            wprod *= w[d][tap[d]];
        }
        let pixel = buf[offset].as_f64();
        value += wprod * pixel;
        for (q, gq) in grad.iter_mut().enumerate() {
            let mut gprod = 1.0;
            for d in 0..dim {
                gprod *= if d == q { dw[d][tap[d]] } else { w[d][tap[d]] };
            }
            *gq += gprod * pixel;
        }
    }
    Some((value, grad))
}

/// ITK `m_IndexToPhysicalPoint = Direction · diag(spacing)`, row-major: maps a
/// continuous index to a physical displacement from the origin. The single
/// implementation lives in [`crate::core::coord`]; this re-exports it so resample,
/// warp, and the metric share the one primitive.
pub fn index_to_physical_matrix(direction: &[f64], spacing: &[f64], dim: usize) -> Vec<f64> {
    coord::index_to_physical_matrix(direction, spacing, dim)
}

/// ITK `m_PhysicalPointToIndex = inverse(Direction · diag(spacing))`, row-major,
/// or `None` if singular: maps a physical displacement from the origin to a
/// continuous index. Inverts the **composed** matrix (via [`crate::core::coord`]),
/// not the direction alone — so a diagonal geometry reciprocal-multiplies as ITK
/// does. Previously this inverted the direction then divided by spacing, which
/// diverged from ITK for oblique directions.
pub fn physical_to_index_matrix(
    direction: &[f64],
    spacing: &[f64],
    dim: usize,
) -> Option<Vec<f64>> {
    coord::physical_to_index_matrix(direction, spacing, dim)
}

/// `origin + M · index` mapping a discrete output-grid index to a physical
/// point, via [`crate::core::coord::index_to_physical_point_f64`] — ITK's integer
/// `TransformIndexToPhysicalPoint` fold, origin **first**. (The callers pass an
/// integer grid counter widened to `f64`.)
pub fn affine_apply(m: &[f64], index: &[f64], origin: &[f64], dim: usize) -> Vec<f64> {
    coord::index_to_physical_point_f64(m, origin, index, dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    // R6 (coord-rounding-port-map.md §4): physical_to_index_matrix now inverts
    // the COMPOSED Direction·diag(spacing), matching ITK. For an oblique
    // direction that differs in the last bits from the pre-fix inverse-then-
    // divide; for a diagonal geometry it is unchanged. Non-vacuity: an
    // axis-aligned direction would make both forms identical, so this uses an
    // oblique D and checks the composed identity holds.
    #[test]
    fn physical_to_index_matrix_inverts_the_composed_matrix() {
        let dir = [0.6, -0.8, 0.8, 0.6];
        let spacing = [3.0, 2.0];
        let p2i = physical_to_index_matrix(&dir, &spacing, 2).unwrap();
        // p2i must be the inverse of index_to_physical_matrix bit-for-bit.
        let i2p = index_to_physical_matrix(&dir, &spacing, 2);
        let prod = crate::core::matrix::matmul(&p2i, &i2p, 2);
        for (k, &v) in prod.iter().enumerate() {
            let want = if k == 0 || k == 3 { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-12, "prod[{k}]={v}");
        }
        // Guard: it is NOT the pre-fix diag(1/spacing)·inv(D).
        let invd = crate::core::matrix::invert(&dir, 2).unwrap();
        let old = [
            invd[0] / spacing[0],
            invd[1] / spacing[0],
            invd[2] / spacing[1],
            invd[3] / spacing[1],
        ];
        assert_ne!(p2i, old.to_vec());
    }

    // R5: affine_apply maps a discrete output index origin-FIRST (ITK integer
    // TransformIndexToPhysicalPoint). A single row must accumulate two terms for
    // the fold to bite, so the matrix is a shear at a large origin.
    #[test]
    fn affine_apply_uses_origin_first_fold() {
        let m = [1.0, 1.0, 0.0, 1.0]; // shear
        let origin = [1e16, 0.0];
        let p = affine_apply(&m, &[1.0, 1.0], &origin, 2);
        assert_eq!(p[0], 1e16); // ((origin + 1) + 1), origin-first
        // Guard: the pre-fix origin-last fold ((1+1)+origin) gives a different bit.
        assert_ne!(p[0], (1.0 + 1.0) + 1e16);
    }

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

    /// A deterministic pseudo-random image — no fixed structure the spline
    /// could reproduce by accident.
    fn noisy(w: usize, h: usize) -> (Vec<f64>, Vec<usize>) {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut buf = vec![0.0f64; w * h];
        for v in buf.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state >> 11) as f64 / (1u64 << 53) as f64 * 200.0 - 100.0;
        }
        (buf, vec![w, h])
    }

    #[test]
    fn bspline_decomposition_of_a_constant_image_is_constant() {
        // The IIR gain `(1 − z)(1 − 1/z)` and the mirror-boundary causal /
        // anticausal initializations are together exactly the inverse of the
        // discrete cubic B-spline kernel, whose taps sum to 1 — so the
        // coefficients of a constant image are that same constant, on the
        // boundary rows as much as in the interior.
        let size = vec![7, 5];
        let strides = strides(&size);
        let buf = vec![7.25f64; 35];
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        for (k, &c) in coeffs.iter().enumerate() {
            assert!((c - 7.25).abs() < 1e-12, "coeff[{k}] = {c}, want 7.25");
        }
        // …and the interpolant reproduces the constant everywhere, including
        // off-lattice at the very edge of the buffer.
        for &cindex in &[[-0.49, -0.49], [0.0, 0.0], [3.7, 2.2], [6.49, 4.49]] {
            let (value, grad) =
                bspline_value_and_gradient(&coeffs, &size, &strides, &cindex).unwrap();
            assert!((value - 7.25).abs() < 1e-12, "{cindex:?}: value {value}");
            assert!(
                grad.iter().all(|g| g.abs() < 1e-12),
                "{cindex:?}: grad {grad:?}"
            );
        }
    }

    #[test]
    fn bspline_reproduces_every_sample_of_a_noisy_image_including_the_edges() {
        // Prefiltering exists exactly so that the spline *interpolates*: at
        // every integer index — corners and edges included, where the mirror
        // fold is active — the value must be the original sample, to
        // round-trip precision rather than a loose "close enough".
        let (buf, size) = noisy(9, 7);
        let strides = strides(&size);
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        for y in 0..7 {
            for x in 0..9 {
                let (value, _) =
                    bspline_value_and_gradient(&coeffs, &size, &strides, &[x as f64, y as f64])
                        .unwrap();
                let expected = buf[y * 9 + x];
                assert!(
                    (value - expected).abs() < 1e-11,
                    "({x},{y}): value {value} vs sample {expected}"
                );
            }
        }
    }

    #[test]
    fn bspline_ramp_is_exact_at_samples_but_bends_off_lattice_at_the_boundary() {
        // The cubic spline reproduces a degree-1 polynomial exactly only where
        // the mirror extension is itself that polynomial — i.e. in the
        // interior. Mirroring a ramp about index 0 produces `|x|`, not `x`, so
        // between the first two samples the spline follows the kink, not the
        // ramp. This is ITK's behaviour (`ApplyMirrorBoundaryConditions`), not
        // an approximation error: it is a half-unit-sized effect, not a
        // rounding-sized one.
        // The kink's influence decays geometrically in the pole |z| ≈ 0.268 per
        // pixel of distance, so "interior" means many e-foldings from both
        // edges: at 20 pixels, 0.268^20 ≈ 1e-11.
        let n = 41usize;
        let size = vec![n];
        let strides = strides(&size);
        let buf: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 + 1.0).collect();
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        let exact = |x: f64| 3.0 * x + 1.0;
        let at = |x: f64| {
            bspline_value_and_gradient(&coeffs, &size, &strides, &[x])
                .unwrap()
                .0
        };

        // Exact at every sample, edges included.
        for i in 0..n {
            let x = i as f64;
            assert!((at(x) - exact(x)).abs() < 1e-11, "sample {i}: {}", at(x));
        }
        // Exact off-lattice in the interior.
        for &x in &[19.5f64, 20.25, 20.5] {
            assert!((at(x) - exact(x)).abs() < 1e-8, "interior {x}: {}", at(x));
        }
        // Not exact off-lattice in the first/last mesh cell: a half-unit error,
        // three orders of magnitude above anything rounding could produce.
        for &x in &[0.5f64, (n - 1) as f64 - 0.5] {
            let err = (at(x) - exact(x)).abs();
            assert!(err > 0.4, "boundary {x}: unexpectedly exact (err {err})");
        }
    }

    #[test]
    fn bspline_index_folding_mirrors_about_the_edge_samples() {
        // `ApplyMirrorBoundaryConditions` folds `i < 0` to `−i` and
        // `i > n − 1` to `2(n − 1) − i`, which makes the reconstructed spline
        // exactly even about index 0 and about index n − 1. Asserting that
        // symmetry pins the fold without hard-coding a tap table: an
        // off-by-one in either fold breaks it immediately.
        let (buf, _) = noisy(11, 1);
        let size = vec![11usize];
        let strides = strides(&size);
        let coeffs = bspline_coefficients(&buf, &size, &strides);
        let at = |x: f64| {
            bspline_value_and_gradient(&coeffs, &size, &strides, &[x])
                .unwrap()
                .0
        };
        let last = 10.0f64;
        for &t in &[0.1f64, 0.25, 0.49] {
            assert!(
                (at(-t) - at(t)).abs() < 1e-12,
                "low fold: v(−{t}) {} vs v({t}) {}",
                at(-t),
                at(t)
            );
            assert!(
                (at(last + t) - at(last - t)).abs() < 1e-12,
                "high fold: v({}) {} vs v({}) {}",
                last + t,
                at(last + t),
                last - t,
                at(last - t)
            );
        }
    }

    #[test]
    fn bspline_folds_a_size_one_axis_onto_index_zero() {
        // ITK's `ApplyMirrorBoundaryConditions` special-cases `m_DataLength == 1`
        // by pinning every tap to index 0 (the generic fold would run off the
        // end). Such an axis is then constant, so a 2-D image with a degenerate
        // second axis must interpolate exactly like the 1-D row it holds.
        let (row, _) = noisy(6, 1);
        let size2 = vec![6usize, 1];
        let strides2 = strides(&size2);
        let coeffs2 = bspline_coefficients(&row, &size2, &strides2);

        let size1 = vec![6usize];
        let strides1 = strides(&size1);
        let coeffs1 = bspline_coefficients(&row, &size1, &strides1);

        for &x in &[0.0f64, 1.3, 2.5, 5.0] {
            let v2 = bspline_value_and_gradient(&coeffs2, &size2, &strides2, &[x, 0.0])
                .unwrap()
                .0;
            let v1 = bspline_value_and_gradient(&coeffs1, &size1, &strides1, &[x])
                .unwrap()
                .0;
            assert!((v2 - v1).abs() < 1e-12, "x={x}: 2-D {v2} vs 1-D {v1}");
        }
    }

    #[test]
    fn bspline_weights_at_a_half_index_are_the_hand_derived_cubic_taps() {
        // At `frac = ½` the cubic B-spline taps are B₃(1.5), B₃(0.5), B₃(−0.5),
        // B₃(−1.5) = 1/48, 23/48, 23/48, 1/48. Feed the coefficient buffer
        // directly (no prefiltering) so the assertion is on the weights alone.
        let coeffs = [2.0f64, -1.0, 5.0, 3.0, -4.0, 0.5];
        let size = vec![6usize];
        let strides = strides(&size);
        // Support of cindex 2.5 is taps 1..=4, entirely interior — no folding.
        let (value, _) = bspline_value_and_gradient(&coeffs, &size, &strides, &[2.5]).unwrap();
        let expected = (coeffs[1] + 23.0 * coeffs[2] + 23.0 * coeffs[3] + coeffs[4]) / 48.0;
        assert!((value - expected).abs() < 1e-14, "{value} vs {expected}");
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

    const ALL_SINC_WINDOWS: [SincWindow; 5] = [
        SincWindow::Hamming,
        SincWindow::Cosine,
        SincWindow::Welch,
        SincWindow::Lanczos,
        SincWindow::Blackman,
    ];

    #[test]
    fn windowed_sinc_reproduces_every_sample_at_grid_points_for_every_window() {
        // At `distance == 0` every non-own tap's weight is the hard `0.0`
        // ITK's delta-branch sets (not merely a near-zero `sinc` residual), so
        // the sum collapses to exactly `1.0 * sample` — bit-exact, not just
        // close.
        let (buf, size) = noisy(9, 7);
        let strides = strides(&size);
        for window in ALL_SINC_WINDOWS {
            for y in 0..7 {
                for x in 0..9 {
                    let (value, _) = windowed_sinc_value_and_gradient(
                        &buf,
                        &size,
                        &strides,
                        &[x as f64, y as f64],
                        window,
                    )
                    .unwrap();
                    let expected = buf[y * 9 + x];
                    assert_eq!(value, expected, "{window:?} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn windowed_sinc_interpolant_is_symmetric_about_a_symmetric_lines_center() {
        // K(t) = w(t) sinc(t) is even for every window here (`w` and `sinc`
        // are each even), so reconstructing a sequence that is itself even
        // about index 20 must give a continuous function that is even about
        // cindex 20, off-lattice included.
        let n = 41usize;
        let mut buf = vec![0.0f64; n];
        let mut state = 0x1234_5678_9abc_def0u64;
        for k in 0..=20usize {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state >> 11) as f64 / (1u64 << 53) as f64 * 100.0;
            buf[20 + k] = v;
            buf[20 - k] = v;
        }
        let size = vec![n];
        let strides = strides(&size);
        for window in ALL_SINC_WINDOWS {
            for &t in &[0.1f64, 0.37, 0.49] {
                let plus =
                    windowed_sinc_value_and_gradient(&buf, &size, &strides, &[20.0 + t], window)
                        .unwrap()
                        .0;
                let minus =
                    windowed_sinc_value_and_gradient(&buf, &size, &strides, &[20.0 - t], window)
                        .unwrap()
                        .0;
                assert!(
                    (plus - minus).abs() < 1e-9,
                    "{window:?} t={t}: v(20+t) {plus} vs v(20-t) {minus}"
                );
            }
        }
    }

    /// Independent, brute-force reference for the theory formula in
    /// `itkWindowedSincInterpolateImageFunction.h`'s class docs — `I(x) =
    /// sum_i I_i K(x - i)` over the `2*radius` taps surrounding `x`, `K(t) =
    /// w(t) sinc(t)` — written with its own index arithmetic (absolute pixel
    /// index `i` and `t = cindex - i`, rather than the production code's
    /// `distance`/tap-offset parameterization) so it cannot share a
    /// transcription bug with [`windowed_sinc_value_and_gradient`].
    fn brute_force_windowed_sinc_1d(buf: &[f64], cindex: f64, window: SincWindow) -> f64 {
        let radius = WINDOWED_SINC_RADIUS as i64;
        let base = cindex.floor() as i64;
        let mut sum = 0.0;
        for i in (base - radius + 1)..=(base + radius) {
            let clamped = i.clamp(0, buf.len() as i64 - 1) as usize;
            let t = cindex - i as f64;
            sum += buf[clamped] * window.weight(t) * sinc(t);
        }
        sum
    }

    #[test]
    fn windowed_sinc_matches_a_brute_force_evaluation_of_the_same_formula() {
        let (buf, _) = noisy(12, 1);
        for window in ALL_SINC_WINDOWS {
            let size = vec![12usize];
            let strides = strides(&size);
            // Fractional points only: at an exact grid point ITK's own
            // delta-branch (a floating-point-noise guard, see
            // `windowed_sinc_value_and_gradient`'s docs) makes production and
            // this naive reference differ by the `sinc(integer) != 0.0`
            // rounding residual the guard exists to suppress — a ~1e-16-scale
            // non-issue that grid-point exactness is already covered by
            // `windowed_sinc_reproduces_every_sample_at_grid_points_for_every_window`.
            for &cindex in &[3.3f64, 5.7, 0.5, 10.9] {
                let got =
                    windowed_sinc_value_and_gradient(&buf, &size, &strides, &[cindex], window)
                        .unwrap()
                        .0;
                let want = brute_force_windowed_sinc_1d(&buf, cindex, window);
                assert!(
                    (got - want).abs() < 1e-9,
                    "{window:?} cindex={cindex}: {got} vs brute-force {want}"
                );
            }
        }
    }

    #[test]
    fn windowed_sinc_gradient_matches_finite_difference_for_every_window() {
        let (buf, size) = noisy(14, 14);
        let strides = strides(&size);
        let c0 = [6.4, 7.6];
        // A tighter step than the other kernels' finite-difference tests use:
        // the sinc/window product's higher-order derivatives scale with
        // `(pi/radius)^n` across `2*radius` taps per axis, so at `h = 1e-4`
        // the O(h^2) truncation error alone is close to 1e-6 — not a
        // correctness margin, just this kernel's curvature. `h = 1e-6` cuts
        // truncation error ~1e4x while `vp - vm` (~1e-4 scale here) stays
        // many orders above f64 round-off.
        let h = 1e-6;
        for window in ALL_SINC_WINDOWS {
            let analytic = windowed_sinc_value_and_gradient(&buf, &size, &strides, &c0, window)
                .unwrap()
                .1;
            for k in 0..2 {
                let mut cp = c0;
                cp[k] += h;
                let mut cm = c0;
                cm[k] -= h;
                let vp = windowed_sinc_value_and_gradient(&buf, &size, &strides, &cp, window)
                    .unwrap()
                    .0;
                let vm = windowed_sinc_value_and_gradient(&buf, &size, &strides, &cm, window)
                    .unwrap()
                    .0;
                let fd = (vp - vm) / (2.0 * h);
                assert!(
                    (fd - analytic[k]).abs() < 1e-6,
                    "{window:?} axis {k}: fd {fd} vs analytic {}",
                    analytic[k]
                );
            }
        }
    }

    #[test]
    fn windowed_sinc_respects_is_inside_bounds() {
        let (buf, size) = noisy(10, 10);
        let strides = strides(&size);
        assert!(
            windowed_sinc_value_and_gradient(
                &buf,
                &size,
                &strides,
                &[9.5, 5.0],
                SincWindow::Hamming
            )
            .is_none()
        );
        assert!(
            windowed_sinc_value_and_gradient(
                &buf,
                &size,
                &strides,
                &[5.0, 5.0],
                SincWindow::Hamming
            )
            .is_some()
        );
    }
}
