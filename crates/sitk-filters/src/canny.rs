//! Canny edge detection and the zero-crossing filter it's built from, porting
//! `itkCannyEdgeDetectionImageFilter.h`(`.hxx`) and
//! `itkZeroCrossingImageFilter.h`(`.hxx`).
//!
//! ITK's Canny is not the textbook 4-way-angle-quantized version. Reading
//! `itkCannyEdgeDetectionImageFilter.hxx` top to bottom, `GenerateData` runs:
//!
//! 1. **Smoothing** (`m_GaussianFilter`, a `DiscreteGaussianImageFilter`) —
//!    [`discrete_gaussian_smooth`], private to this module (see below).
//! 2. **`ComputeCannyEdge`** — the second directional derivative of the
//!    smoothed image along its own gradient direction,
//!    `D_uu f = (∇f)ᵀ H (∇f) / |∇f|²` with `u = ∇f/|∇f|`, evaluated per pixel
//!    over a single radius-1 neighborhood ([`second_directional_derivative_field`]).
//! 3. **`ThreadedCompute2ndDerivativePos`** — the derivative of *that* field
//!    along the smoothed image's gradient direction (approximating the third
//!    directional derivative's sign); wherever it is `<= 0`, the pixel keeps
//!    the smoothed image's gradient magnitude, else `0`
//!    ([`positional_gate_field`]).
//! 4. **Non-maximum suppression** — the zero-crossings of step 2's field
//!    ([`zero_crossing_values`], the engine behind the public [`zero_crossing`]),
//!    multiplied elementwise into step 3's field (`MultiplyImageFilter`).
//! 5. **Hysteresis thresholding** (`HysteresisThresholding`/`FollowEdge`) —
//!    [`hysteresis_threshold`], a flood fill seeded from every pixel above
//!    `upper_threshold`, pulling in any full-neighborhood neighbor above
//!    `lower_threshold` that hasn't been visited yet.
//!
//! There is no 4-way gradient-angle quantization anywhere in this pipeline —
//! suppression falls directly out of the sign of the second and third
//! directional derivatives along the smoothed image's *actual* gradient
//! direction, computed continuously via the neighborhood inner products
//! above.
//!
//! ITK's derivative computations here (`m_ComputeCannyEdge1stDerivativeOper`,
//! `m_ComputeCannyEdge2ndDerivativeOper`) are built with
//! `DerivativeOperator::CreateDirectional()` alone — no `FlipAxes()`, no
//! `ScaleCoefficients()` — so they run in raw index space with the operator's
//! *unflipped* coefficients, unlike this crate's [`crate::gradient::derivative`]
//! (which flips and optionally scales by spacing). The order-1 operator is
//! antisymmetric, so unflipped vs. flipped only changes its global sign; every
//! place it appears in this filter is either squared, multiplied by another
//! instance of itself, or divided by its own vector norm, so that global sign
//! cancels out and the unflipped convention used here is numerically
//! equivalent to the flipped one. The order-2 operator is symmetric, so
//! flipping is a no-op for it regardless. [`derivative_operator_coefficients`]
//! is [`crate::gradient`]'s verified generator for both, reused here
//! `pub(crate)` rather than re-derived.
//!
//! Only the smoothing step honors image spacing (`DiscreteGaussianImageFilter`'s
//! `UseImageSpacing`, on by default: kernel variance is `variance[d] /
//! spacing[d]^2`); every derivative computation after that runs purely in
//! index space, exactly as ITK's source does (no `ScaleCoefficients` call
//! anywhere in `itkCannyEdgeDetectionImageFilter.hxx`).
//!
//! **`DiscreteGaussianImageFilter` is not yet ported elsewhere in this
//! workspace.** [`discrete_gaussian_smooth`] and its Bessel-function kernel
//! generator ([`discrete_gaussian_kernel_1d`], `GaussianOperator::
//! GenerateCoefficients` in `itkGaussianOperator.hxx`) are implemented
//! privately in this module for Canny's own use, matching that filter's
//! default `MaximumKernelWidth` (32) and `ZeroFluxNeumannBoundaryCondition`;
//! they are not exported, to avoid colliding with a future public port.

use crate::error::{FilterError, Result};
use crate::gradient::derivative_operator_coefficients;
use crate::image_from_f64;
use sitk_core::{Image, NeighborhoodIterator, PixelId, ZeroFluxNeumannBoundaryCondition};

// ==== private discrete Gaussian (GaussianOperator + DiscreteGaussianImageFilter) ====
//
// Not exported: a sibling port of `DiscreteGaussianImageFilter` is expected
// elsewhere in the workspace, and this crate must not expose a second,
// colliding `discrete_gaussian` name.

/// `GaussianOperator::ModifiedBesselI0` (itkGaussianOperator.hxx): the
/// Abramowitz & Stegun polynomial approximation of the modified Bessel
/// function `I0`.
fn bessel_i0(y: f64) -> f64 {
    let d = y.abs();
    if d < 3.75 {
        let mut m = y / 3.75;
        m *= m;
        1.0 + m
            * (3.5156229
                + m * (3.0899424
                    + m * (1.2067492 + m * (0.2659732 + m * (0.360768e-1 + m * 0.45813e-2)))))
    } else {
        let m = 3.75 / d;
        (d.exp() / d.sqrt())
            * (0.39894228
                + m * (0.1328592e-1
                    + m * (0.225319e-2
                        + m * (-0.157565e-2
                            + m * (0.916281e-2
                                + m * (-0.2057706e-1
                                    + m * (0.2635537e-1
                                        + m * (-0.1647633e-1 + m * 0.392377e-2))))))))
    }
}

/// `GaussianOperator::ModifiedBesselI1` (itkGaussianOperator.hxx).
fn bessel_i1(y: f64) -> f64 {
    let d = y.abs();
    let accumulator = if d < 3.75 {
        let mut m = y / 3.75;
        m *= m;
        d * (0.5
            + m * (0.87890594
                + m * (0.51498869
                    + m * (0.15084934 + m * (0.2658733e-1 + m * (0.301532e-2 + m * 0.32411e-3))))))
    } else {
        let m = 3.75 / d;
        let mut acc = 0.2282967e-1 + m * (-0.2895312e-1 + m * (0.1787654e-1 - m * 0.420059e-2));
        acc = 0.39894228
            + m * (-0.3988024e-1
                + m * (-0.362018e-2 + m * (0.163801e-2 + m * (-0.1031555e-1 + m * acc))));
        acc * (d.exp() / d.sqrt())
    };
    if y < 0.0 { -accumulator } else { accumulator }
}

/// `GaussianOperator::ModifiedBesselI` (itkGaussianOperator.hxx), `n >= 2`:
/// downward recurrence (Miller's algorithm) for the modified Bessel function
/// of integer order `n`.
fn bessel_i(n: i32, y: f64) -> f64 {
    debug_assert!(n >= 2);
    if y == 0.0 {
        return 0.0;
    }
    const ACCURACY: f64 = 40.0;
    let toy = 2.0 / y.abs();
    let mut qip = 0.0f64;
    let mut accumulator = 0.0f64;
    let mut qi = 1.0f64;

    let mut j = 2 * (n + (ACCURACY * n as f64).sqrt() as i32);
    while j > 0 {
        let qim = qip + j as f64 * toy * qi;
        qip = qi;
        qi = qim;
        if qi.abs() > 1.0e10 {
            accumulator *= 1.0e-10;
            qi *= 1.0e-10;
            qip *= 1.0e-10;
        }
        if j == n {
            accumulator = qip;
        }
        j -= 1;
    }

    accumulator *= bessel_i0(y) / qi;

    if y < 0.0 && (n & 1) == 1 {
        -accumulator
    } else {
        accumulator
    }
}

/// `GaussianOperator::GenerateCoefficients` (itkGaussianOperator.hxx): the
/// symmetric 1-D discrete Gaussian kernel for `variance`, built from the
/// modified-Bessel-function identity `1 = e^-t * sum_{k=-inf}^{inf} I_k(t)`.
/// One-sided taps are added (`I0`, then `I1`, `I2`, ...) until their
/// cumulative (unnormalized) sum reaches `1 - maximum_error`, or the one-sided
/// tap count exceeds `maximum_kernel_width`
/// (`DiscreteGaussianImageFilter`'s default of 32) — whichever comes first.
/// The result is normalized to sum to 1 and mirrored:
/// `kernel[radius + k] == kernel[radius - k]`.
fn discrete_gaussian_kernel_1d(
    variance: f64,
    maximum_error: f64,
    maximum_kernel_width: usize,
) -> Vec<f64> {
    let et = (-variance).exp();
    let mut raw = vec![et * bessel_i0(variance), et * bessel_i1(variance)];
    let mut sum = raw[0] + raw[1] * 2.0;

    let cap = 1.0 - maximum_error;
    let mut i: i32 = 2;
    while sum < cap {
        let c = et * bessel_i(i, variance);
        raw.push(c);
        sum += c * 2.0;
        if c <= 0.0 {
            break;
        }
        if raw.len() > maximum_kernel_width {
            break;
        }
        i += 1;
    }

    for c in &mut raw {
        *c /= sum;
    }

    let radius = raw.len() - 1;
    let mut kernel = vec![0.0; 2 * radius + 1];
    for (k, &c) in raw.iter().enumerate() {
        kernel[radius - k] = c;
        kernel[radius + k] = c;
    }
    kernel
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// Convolve `buf` along axis `d` with `kernel` under a clamped
/// (`ZeroFluxNeumannBoundaryCondition`) boundary — `DiscreteGaussianImageFilter`'s
/// default boundary for both its input and its real-valued intermediate
/// buffers.
fn convolve_axis(
    buf: &[f64],
    size: &[usize],
    strides: &[usize],
    d: usize,
    kernel: &[f64],
    radius: usize,
) -> Vec<f64> {
    let stride = strides[d];
    let size_d = size[d] as isize;
    let mut out = vec![0.0f64; buf.len()];
    for (p, slot) in out.iter_mut().enumerate() {
        let coord_d = (p / stride) % size[d];
        let line_base = p - coord_d * stride;
        let mut acc = 0.0;
        for (ki, &w) in kernel.iter().enumerate() {
            let c =
                (coord_d as isize + ki as isize - radius as isize).clamp(0, size_d - 1) as usize;
            acc += w * buf[line_base + c * stride];
        }
        *slot = acc;
    }
    out
}

/// `DiscreteGaussianImageFilter` (itkDiscreteGaussianImageFilter.h(.hxx)),
/// restricted to what `canny_edge_detection` needs from it: separable
/// convolution with [`discrete_gaussian_kernel_1d`], one axis at a time.
/// `variance` is in physical units (`UseImageSpacing` on by default, the only
/// setting ITK's Canny filter ever uses): each axis's kernel variance is
/// `variance[d] / spacing[d]^2` (`GetKernelVarianceArray`). Returns an
/// always-`f64` image (this crate's scratch-then-narrow-once convention, see
/// [`crate::gradient`]'s module docs); geometry is copied from `img`.
fn discrete_gaussian_smooth(img: &Image, variance: &[f64], maximum_error: &[f64]) -> Result<Image> {
    const MAXIMUM_KERNEL_WIDTH: usize = 32;

    let dim = img.dimension();
    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let strides_v = strides(&size);
    let mut buf = img.to_f64_vec();

    for d in 0..dim {
        let kernel_variance = variance[d] / (spacing[d] * spacing[d]);
        let kernel =
            discrete_gaussian_kernel_1d(kernel_variance, maximum_error[d], MAXIMUM_KERNEL_WIDTH);
        let radius = kernel.len() / 2;
        buf = convolve_axis(&buf, &size, &strides_v, d, &kernel, radius);
    }

    let mut out = Image::from_vec(&size, buf)?;
    out.copy_geometry_from(img);
    Ok(out)
}

// ==== ComputeCannyEdge / ThreadedCompute2ndDerivativePos ====

/// `CannyEdgeDetectionImageFilter::ComputeCannyEdge` (called from
/// `ThreadedCompute2ndDerivative`): the second directional derivative of
/// `smoothed` along its own gradient direction, `(∇f)ᵀ H (∇f) / |∇f|²`
/// (`0.0001` added to the denominator to avoid division by zero at flat
/// pixels), evaluated over a radius-1 neighborhood under
/// [`ZeroFluxNeumannBoundaryCondition`].
fn second_directional_derivative_field(smoothed: &Image) -> Result<Vec<f64>> {
    let dim = smoothed.dimension();
    let radius = vec![1usize; dim];
    let iter =
        NeighborhoodIterator::<f64, _>::new(smoothed, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let d1 = derivative_operator_coefficients(1);
    let d2 = derivative_operator_coefficients(2);
    let taps = [-1i64, 0, 1];

    let out = iter
        .map(|(_, nb)| {
            let mut dx = vec![0.0f64; dim];
            let mut dxx = vec![0.0f64; dim];
            let mut off = vec![0i64; dim];
            for a in 0..dim {
                let mut sx = 0.0;
                let mut sxx = 0.0;
                for (k, &delta) in taps.iter().enumerate() {
                    off[a] = delta;
                    let v = nb.get(&off);
                    sx += d1[k] * v;
                    sxx += d2[k] * v;
                }
                off[a] = 0;
                dx[a] = sx;
                dxx[a] = sxx;
            }

            let mut deriv = 0.0;
            for i in 0..dim {
                for j in (i + 1)..dim {
                    off[i] = -1;
                    off[j] = -1;
                    let m_m = nb.get(&off);
                    off[j] = 1;
                    let m_p = nb.get(&off);
                    off[i] = 1;
                    let p_p = nb.get(&off);
                    off[j] = -1;
                    let p_m = nb.get(&off);
                    off[i] = 0;
                    off[j] = 0;
                    let dxy = 0.25 * m_m - 0.25 * m_p - 0.25 * p_m + 0.25 * p_p;
                    deriv += 2.0 * dx[i] * dx[j] * dxy;
                }
            }

            let mut grad_mag = 0.0001;
            for a in 0..dim {
                deriv += dx[a] * dx[a] * dxx[a];
                grad_mag += dx[a] * dx[a];
            }
            deriv / grad_mag
        })
        .collect();
    Ok(out)
}

/// `CannyEdgeDetectionImageFilter::ThreadedCompute2ndDerivativePos`: the
/// derivative of `deriv_field` (the field [`second_directional_derivative_field`]
/// produced) along `smoothed`'s gradient direction — approximating the sign
/// of the third directional derivative. Where it is `<= 0`, the pixel keeps
/// `smoothed`'s own gradient magnitude; elsewhere it is `0`.
fn positional_gate_field(smoothed: &Image, deriv_field: &Image) -> Result<Vec<f64>> {
    let dim = smoothed.dimension();
    let radius = vec![1usize; dim];
    let iter_s =
        NeighborhoodIterator::<f64, _>::new(smoothed, &radius, ZeroFluxNeumannBoundaryCondition)?;
    let iter_d = NeighborhoodIterator::<f64, _>::new(
        deriv_field,
        &radius,
        ZeroFluxNeumannBoundaryCondition,
    )?;

    let d1 = derivative_operator_coefficients(1);
    let taps = [-1i64, 0, 1];

    let out = iter_s
        .zip(iter_d)
        .map(|((_, nb_s), (_, nb_d))| {
            let mut off = vec![0i64; dim];
            let mut dx = vec![0.0f64; dim];
            let mut dx1 = vec![0.0f64; dim];
            let mut grad_mag_sq = 0.0001;
            for a in 0..dim {
                let mut sx = 0.0;
                let mut sx1 = 0.0;
                for (k, &delta) in taps.iter().enumerate() {
                    off[a] = delta;
                    sx += d1[k] * nb_s.get(&off);
                    sx1 += d1[k] * nb_d.get(&off);
                }
                off[a] = 0;
                dx[a] = sx;
                dx1[a] = sx1;
                grad_mag_sq += sx * sx;
            }
            let grad_mag = grad_mag_sq.sqrt();
            let deriv_pos: f64 = (0..dim).map(|a| dx1[a] * (dx[a] / grad_mag)).sum();
            if deriv_pos <= 0.0 { grad_mag } else { 0.0 }
        })
        .collect();
    Ok(out)
}

// ==== ZeroCrossingImageFilter ====

/// `ZeroCrossingImageFilter::DynamicThreadedGenerateData`
/// (itkZeroCrossingImageFilter.hxx): the sign-comparison rule shared by the
/// public [`zero_crossing`] and `canny_edge_detection`'s own non-maximum
/// suppression step. For each pixel, walk its `2 * dim` axis-aligned
/// ("city-block") neighbors in ITK's own order — all `dim` negative-direction
/// offsets first, then all `dim` positive-direction offsets — and label the
/// pixel `foreground` the moment a sign change is found where it is the
/// closer-to-zero side (`|this| < |that|`); a tie (`|this| == |that|`) only
/// counts on the *positive*-direction pass, so a symmetric sign change marks
/// exactly one of the two pixels, never both. Any other pixel is
/// `background`. Uses [`ZeroFluxNeumannBoundaryCondition`], matching ITK.
fn zero_crossing_values(img: &Image, foreground: f64, background: f64) -> Result<Vec<f64>> {
    let dim = img.dimension();
    let radius = vec![1usize; dim];
    let iter = NeighborhoodIterator::<f64, _>::new(img, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let out = iter
        .map(|(_, nb)| {
            let this_one = nb.center_value();
            let mut off = vec![0i64; dim];
            let mut result = background;
            'search: for (pass, delta) in [(0, -1i64), (1, 1i64)] {
                for d in 0..dim {
                    off[d] = delta;
                    let that = nb.get(&off);
                    off[d] = 0;

                    let sign_change = (this_one < 0.0 && that > 0.0)
                        || (this_one > 0.0 && that < 0.0)
                        || (this_one == 0.0 && that != 0.0)
                        || (this_one != 0.0 && that == 0.0);
                    if !sign_change {
                        continue;
                    }
                    let a = this_one.abs();
                    let b = that.abs();
                    if a < b || (a == b && pass == 1) {
                        result = foreground;
                        break 'search;
                    }
                }
            }
            result
        })
        .collect();
    Ok(out)
}

/// An `f64` copy of `img`'s pixels with `img`'s geometry, used as the working
/// buffer for [`zero_crossing`] (matching [`crate::gradient`]'s own
/// `scratch_f64` helper).
fn scratch_f64(img: &Image) -> Result<Image> {
    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec())?;
    scratch.copy_geometry_from(img);
    Ok(scratch)
}

/// `ZeroCrossingImageFilter`: labels each pixel closest to a sign change
/// among its `2 * dim` axis-aligned neighbors with `foreground_value`; every
/// other pixel gets `background_value`. Output is always
/// [`PixelId::UInt8`], matching SimpleITK's yaml
/// (`output_pixel_type: uint8_t`); SimpleITK's own defaults for
/// `foreground_value` / `background_value` are `1` / `0`.
pub fn zero_crossing(img: &Image, foreground_value: u8, background_value: u8) -> Result<Image> {
    let scratch = scratch_f64(img)?;
    let vals = zero_crossing_values(&scratch, foreground_value as f64, background_value as f64)?;
    image_from_f64(PixelId::UInt8, img.size(), img, &vals)
}

// ==== HysteresisThresholding / FollowEdge ====

/// All `3^dim` offsets in `{-1, 0, 1}^dim`, including the zero vector — the
/// full neighborhood `FollowEdge` walks (`nSize = m_Center * 2 + 1`, which for
/// a radius-1 neighborhood is exactly `3^dim`), unlike [`zero_crossing`]'s
/// city-block `2 * dim` set.
fn full_neighborhood_offsets(dim: usize) -> Vec<Vec<i64>> {
    let mut offsets = vec![vec![]];
    for _ in 0..dim {
        let mut next = Vec::with_capacity(offsets.len() * 3);
        for prefix in &offsets {
            for delta in [-1i64, 0, 1] {
                let mut v = prefix.clone();
                v.push(delta);
                next.push(v);
            }
        }
        offsets = next;
    }
    offsets
}

/// `CannyEdgeDetectionImageFilter::HysteresisThresholding` + `FollowEdge`: a
/// flood fill seeded from every pixel (visited in raster order) whose
/// `edge_strength` exceeds `upper`, pulling in any full-neighborhood neighbor
/// (`3^dim - 1` neighbors, `26`-connected in 3-D / `8`-connected in 2-D, plus
/// the pixel itself as a harmless no-op) whose `edge_strength` exceeds
/// `lower` and hasn't been visited yet. A weak edge connected — even
/// transitively — to a strong seed survives with output value `1`; a pixel
/// that never exceeds `upper` and is never reached by such a chain stays `0`.
fn hysteresis_threshold(edge_strength: &[f64], size: &[usize], lower: f64, upper: f64) -> Vec<f64> {
    let dim = size.len();
    let n = edge_strength.len();
    let strides_v = strides(size);
    let offsets = full_neighborhood_offsets(dim);

    let mut output = vec![0.0f64; n];
    let mut stack: Vec<usize> = Vec::new();

    for seed in 0..n {
        if output[seed] != 0.0 || edge_strength[seed] <= upper {
            continue;
        }
        output[seed] = 1.0;
        stack.push(seed);

        while let Some(current) = stack.pop() {
            let coord: Vec<i64> = (0..dim)
                .map(|d| ((current / strides_v[d]) % size[d]) as i64)
                .collect();

            for off in &offsets {
                let mut linear = 0usize;
                let mut inside = true;
                for d in 0..dim {
                    let v = coord[d] + off[d];
                    if v < 0 || v as usize >= size[d] {
                        inside = false;
                        break;
                    }
                    linear += v as usize * strides_v[d];
                }
                if inside && edge_strength[linear] > lower && output[linear] == 0.0 {
                    output[linear] = 1.0;
                    stack.push(linear);
                }
            }
        }
    }

    output
}

// ==== canny_edge_detection ====

/// `CannyEdgeDetectionImageFilter` (itkCannyEdgeDetectionImageFilter.h(.hxx)):
/// Canny's edge detector — Gaussian smoothing, non-maximum suppression via
/// the zero-crossings of the smoothed image's second directional derivative,
/// then hysteresis thresholding of the resulting edge-strength field. See
/// this module's docs for the full per-stage breakdown.
///
/// `variance` and `maximum_error` are per-dimension (`CannyEdgeDetectionImageFilter`'s
/// `ArrayType`; SimpleITK's yaml exposes both as a vector with a
/// same-for-every-axis scalar convenience, default `variance = 0.0` and
/// `maximum_error = 0.01` in each dimension). `lower_threshold` /
/// `upper_threshold` both default to `0.0` in SimpleITK's yaml. Output pixel
/// type follows `img`'s (SimpleITK's yaml declares `RealPixelIDTypeList` with
/// no `output_pixel_type` override).
///
/// Errors if `variance` or `maximum_error` doesn't have one entry per
/// dimension, if any `variance` entry is negative, or if any
/// `maximum_error` entry is outside the open interval `(0.0, 1.0)`
/// (`GaussianOperator::SetMaximumError`'s own constraint).
pub fn canny_edge_detection(
    img: &Image,
    variance: &[f64],
    maximum_error: &[f64],
    upper_threshold: f64,
    lower_threshold: f64,
) -> Result<Image> {
    let dim = img.dimension();
    if variance.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: variance.len(),
        });
    }
    if maximum_error.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: maximum_error.len(),
        });
    }
    if variance.iter().any(|&v| v < 0.0) {
        return Err(FilterError::InvalidVariance(variance.to_vec()));
    }
    if maximum_error.iter().any(|&e| e <= 0.0 || e >= 1.0) {
        return Err(FilterError::InvalidMaximumError(maximum_error.to_vec()));
    }

    let size = img.size().to_vec();

    // 1. Gaussian smoothing.
    let smoothed = discrete_gaussian_smooth(img, variance, maximum_error)?;

    // 2. Second directional derivative field (ComputeCannyEdge).
    let deriv_vals = second_directional_derivative_field(&smoothed)?;
    let deriv_field = {
        let mut im = Image::from_vec(&size, deriv_vals)?;
        im.copy_geometry_from(img);
        im
    };

    // 3. Positional gate (ThreadedCompute2ndDerivativePos): gradient
    //    magnitude where the 3rd-derivative sign condition holds, else 0.
    let gate_vals = positional_gate_field(&smoothed, &deriv_field)?;

    // 4. Non-maximum suppression: zero-crossings of the directional 2nd
    //    derivative, multiplied elementwise into the gate (MultiplyImageFilter).
    let zc_vals = zero_crossing_values(&deriv_field, 1.0, 0.0)?;
    let edge_strength: Vec<f64> = gate_vals
        .iter()
        .zip(&zc_vals)
        .map(|(&g, &z)| g * z)
        .collect();

    // 5. Hysteresis thresholding.
    let out_vals = hysteresis_threshold(&edge_strength, &size, lower_threshold, upper_threshold);

    image_from_f64(img.pixel_id(), &size, img, &out_vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- discrete_gaussian_kernel_1d ----

    #[test]
    fn discrete_gaussian_kernel_zero_variance_is_identity() {
        let kernel = discrete_gaussian_kernel_1d(0.0, 0.01, 32);
        assert_eq!(kernel, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn discrete_gaussian_kernel_is_symmetric_and_normalized() {
        let kernel = discrete_gaussian_kernel_1d(4.0, 0.01, 32);
        let radius = kernel.len() / 2;
        for k in 0..=radius {
            assert!((kernel[radius - k] - kernel[radius + k]).abs() < 1e-15);
        }
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "kernel sum {sum}");
    }

    // ---- zero_crossing_values (direct, sign-comparison rule) ----

    #[test]
    fn zero_crossing_marks_the_side_closer_to_zero() {
        // 1-D-in-2-D row: -1, 1 -- symmetric sign change (tied magnitude).
        // Pixel 0's *positive*-direction neighbor comparison (i >= dim, so
        // the tie counts) is the one that sees the sign change, so pixel 0 is
        // marked; pixel 1 only sees the same tie on its *negative*-direction
        // pass (i < dim), where a tie never counts.
        let img = Image::from_vec(&[2, 1], vec![-1.0f64, 1.0]).unwrap();
        let out = zero_crossing_values(&img, 1.0, 0.0).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn zero_crossing_marks_the_smaller_magnitude_side() {
        // -1, 3: |{-1}| < |3|, so the -1 pixel (closer to zero) is marked.
        let img = Image::from_vec(&[2, 1], vec![-1.0f64, 3.0]).unwrap();
        let out = zero_crossing_values(&img, 1.0, 0.0).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn zero_crossing_exact_zero_counts_as_a_sign_change() {
        // 0, 5: this_one(0) != 0 is false but the "this==0, that!=0" arm
        // fires; |0| < |5| marks the exact-zero pixel itself.
        let img = Image::from_vec(&[2, 1], vec![0.0f64, 5.0]).unwrap();
        let out = zero_crossing_values(&img, 1.0, 0.0).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn zero_crossing_same_sign_neighbors_is_background() {
        let img = Image::from_vec(&[3, 1], vec![1.0f64, 2.0, 3.0]).unwrap();
        let out = zero_crossing_values(&img, 1.0, 0.0).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_crossing_uses_city_block_neighbors_only_2d() {
        // Center negative, every axis-neighbor (4-connected) positive, but
        // the diagonal neighbors (not checked) are also positive with larger
        // magnitude -- must not affect the result either way here; this pins
        // down that only the 4 face neighbors are examined.
        let (w, h) = (3usize, 3usize);
        let mut data = vec![10.0f64; w * h];
        data[w + 1] = -1.0; // center
        let img = Image::from_vec(&[w, h], data).unwrap();
        let out = zero_crossing_values(&img, 1.0, 0.0).unwrap();
        // center: |-1| < |10| on every one of its 4 face-neighbor comparisons
        // -> foreground.
        assert_eq!(out[w + 1], 1.0);
    }

    #[test]
    fn zero_crossing_public_api_outputs_uint8_with_defaults() {
        let img = Image::from_vec(&[2, 1], vec![-1.0f64, 1.0]).unwrap();
        let out = zero_crossing(&img, 1, 0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 0]);
    }

    #[test]
    fn zero_crossing_custom_foreground_background() {
        let img = Image::from_vec(&[2, 1], vec![-1.0f64, 1.0]).unwrap();
        let out = zero_crossing(&img, 9, 2).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[9, 2]);
    }

    // ---- hysteresis_threshold (direct, flood-fill boundary) ----

    #[test]
    fn hysteresis_weak_edge_connected_to_strong_seed_is_kept() {
        // 1-D line: strong seed (10) next to a weak-but-above-lower pixel
        // (0.6), then a gap (below lower) before an isolated weak pixel.
        let edge_strength = vec![10.0, 0.6, 0.05, 0.6];
        let size = [4usize, 1];
        let out = hysteresis_threshold(&edge_strength, &size, 0.5, 1.0);
        assert_eq!(out, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn hysteresis_isolated_weak_edge_is_dropped() {
        // No pixel exceeds upper, so nothing ever seeds, regardless of how
        // many pixels individually exceed lower.
        let edge_strength = vec![0.6, 0.6, 0.6];
        let size = [3usize, 1];
        let out = hysteresis_threshold(&edge_strength, &size, 0.5, 1.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn hysteresis_lower_equals_upper_degenerates_to_single_threshold() {
        let edge_strength = vec![2.0, 2.0, 0.0, 2.0];
        let size = [4usize, 1];
        let out = hysteresis_threshold(&edge_strength, &size, 1.0, 1.0);
        // Every pixel > 1.0 seeds directly; none needs a neighbor chain.
        assert_eq!(out, vec![1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn hysteresis_connectivity_is_full_neighborhood_diagonal_included_2d() {
        // Strong seed at (0,0), weak-but-above-lower pixel only diagonally
        // adjacent at (1,1): full-neighborhood connectivity must still reach
        // it (unlike zero_crossing's city-block set).
        let (w, h) = (2usize, 2usize);
        let mut edge_strength = vec![0.0f64; w * h];
        edge_strength[0] = 10.0; // (0,0)
        edge_strength[w + 1] = 0.6; // (1,1), diagonal from (0,0)
        let out = hysteresis_threshold(&edge_strength, &[w, h], 0.5, 1.0);
        assert_eq!(out, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn hysteresis_no_seed_leaves_everything_background() {
        let edge_strength = vec![0.9, 0.9, 0.9];
        let size = [3usize, 1];
        let out = hysteresis_threshold(&edge_strength, &size, 0.5, 1.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    // ---- canny_edge_detection: validation ----

    #[test]
    fn canny_wrong_variance_length_is_rejected() {
        let img = Image::from_vec(&[4, 4], vec![0.0f64; 16]).unwrap();
        let err = canny_edge_detection(&img, &[1.0], &[0.01, 0.01], 1.0, 0.5).unwrap_err();
        assert_eq!(
            err,
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn canny_wrong_maximum_error_length_is_rejected() {
        let img = Image::from_vec(&[4, 4], vec![0.0f64; 16]).unwrap();
        let err = canny_edge_detection(&img, &[1.0, 1.0], &[0.01], 1.0, 0.5).unwrap_err();
        assert_eq!(
            err,
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn canny_negative_variance_is_rejected() {
        let img = Image::from_vec(&[4, 4], vec![0.0f64; 16]).unwrap();
        let err = canny_edge_detection(&img, &[-1.0, 1.0], &[0.01, 0.01], 1.0, 0.5).unwrap_err();
        assert!(matches!(err, FilterError::InvalidVariance(_)));
    }

    #[test]
    fn canny_maximum_error_out_of_range_is_rejected() {
        let img = Image::from_vec(&[4, 4], vec![0.0f64; 16]).unwrap();
        let err = canny_edge_detection(&img, &[1.0, 1.0], &[0.0, 0.01], 1.0, 0.5).unwrap_err();
        assert!(matches!(err, FilterError::InvalidMaximumError(_)));
        let err = canny_edge_detection(&img, &[1.0, 1.0], &[1.0, 0.01], 1.0, 0.5).unwrap_err();
        assert!(matches!(err, FilterError::InvalidMaximumError(_)));
    }

    // ---- canny_edge_detection: constant image -> no edges ----

    #[test]
    fn canny_constant_image_has_no_edges_2d() {
        let img = Image::from_vec(&[9, 9], vec![5.0f64; 81]).unwrap();
        let out = canny_edge_detection(&img, &[1.0, 1.0], &[0.01, 0.01], 1.0, 0.5).unwrap();
        assert!(out.to_f64_vec().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn canny_constant_image_has_no_edges_3d() {
        let img = Image::from_vec(&[7, 7, 7], vec![5.0f64; 343]).unwrap();
        let out =
            canny_edge_detection(&img, &[1.0, 1.0, 1.0], &[0.01, 0.01, 0.01], 1.0, 0.5).unwrap();
        assert!(out.to_f64_vec().iter().all(|&v| v == 0.0));
    }

    // ---- canny_edge_detection: clean step edge -> one-pixel-wide edge ----

    fn step_column_image(w: usize, h: usize, step_x: usize) -> Image {
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if x >= step_x { 100.0 } else { 0.0 };
            }
        }
        Image::from_vec(&[w, h], data).unwrap()
    }

    #[test]
    fn canny_step_edge_is_one_pixel_wide_at_the_step_2d() {
        // The step's second-directional-derivative field is point-antisymmetric
        // about the true (continuous) step location x=9.5: deriv(9.5+d) =
        // -deriv(9.5-d), so |deriv[9]| == |deriv[10]| exactly at d=0.5. That
        // is a *tie* in zero_crossing_values' sign-comparison rule, and a tie
        // only counts on the positive-direction pass (see
        // zero_crossing_marks_the_side_closer_to_zero above) -- which is the
        // pass pixel 9 (not pixel 10) sees, so the edge lands one pixel
        // *before* the step, at x = step_x - 1, not x = step_x.
        let (w, h) = (21usize, 9usize);
        let step_x = 10;
        let img = step_column_image(w, h, step_x);
        let out = canny_edge_detection(&img, &[2.0, 2.0], &[0.01, 0.01], 5.0, 1.0).unwrap();
        let vals = out.to_f64_vec();
        for y in 2..h - 2 {
            let row_edges: Vec<usize> = (2..w - 2).filter(|&x| vals[y * w + x] != 0.0).collect();
            assert_eq!(
                row_edges,
                vec![step_x - 1],
                "row {y}: expected a single edge pixel exactly at x={}, got {row_edges:?}",
                step_x - 1
            );
        }
    }

    #[test]
    fn canny_step_edge_is_one_voxel_wide_at_the_step_3d() {
        // As in the 2-D case above, this is a near-tie in
        // zero_crossing_values' sign-comparison rule; here the residual
        // asymmetry from the finite, zero-flux-clamped domain (empirically
        // verified) resolves it to land exactly at x = step_x rather than
        // step_x - 1 -- unlike the 2-D test's w=21/step_x=10, which resolves
        // the other way. Both are deterministic given fixed inputs; which
        // side wins is not a universal rule, only a property of this port's
        // arithmetic on this specific domain.
        let (w, h, d) = (17usize, 7usize, 7usize);
        let step_x = 8;
        let mut data = vec![0.0f64; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = if x >= step_x { 100.0 } else { 0.0 };
                }
            }
        }
        let img = Image::from_vec(&[w, h, d], data).unwrap();
        let out =
            canny_edge_detection(&img, &[2.0, 2.0, 2.0], &[0.01, 0.01, 0.01], 5.0, 1.0).unwrap();
        let vals = out.to_f64_vec();
        for z in 2..d - 2 {
            for y in 2..h - 2 {
                let row_edges: Vec<usize> = (2..w - 2)
                    .filter(|&x| vals[z * w * h + y * w + x] != 0.0)
                    .collect();
                assert_eq!(
                    row_edges,
                    vec![step_x],
                    "z={z} y={y}: expected a single edge voxel exactly at x={step_x}, got {row_edges:?}"
                );
            }
        }
    }

    #[test]
    fn canny_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[9, 9], vec![1.0f32; 81]).unwrap();
        let out = canny_edge_detection(&img, &[1.0, 1.0], &[0.01, 0.01], 1.0, 0.5).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }
}
