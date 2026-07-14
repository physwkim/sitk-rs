//! Canny edge detection and the zero-crossing filter it's built from, porting
//! `itkCannyEdgeDetectionImageFilter.h`(`.hxx`) and
//! `itkZeroCrossingImageFilter.h`(`.hxx`).
//!
//! ITK's Canny is not the textbook 4-way-angle-quantized version. Reading
//! `itkCannyEdgeDetectionImageFilter.hxx` top to bottom, `GenerateData` runs:
//!
//! 1. **Smoothing** (`m_GaussianFilter`, a `DiscreteGaussianImageFilter`) —
//!    [`crate::denoise::discrete_gaussian_f64`], the `f64` core shared with
//!    the public [`crate::denoise::discrete_gaussian`] (see below).
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
//! The smoothing stage delegates to [`crate::denoise`]'s
//! `DiscreteGaussianImageFilter` port via its `f64` core
//! ([`crate::denoise::discrete_gaussian_f64`]), pinned to the only
//! configuration ITK's Canny ever uses: `UseImageSpacing` on and the default
//! `MaximumKernelWidth` of 32, under `ZeroFluxNeumannBoundaryCondition`.

use crate::denoise::discrete_gaussian_f64;
use crate::error::{FilterError, Result};
use crate::gradient::derivative_operator_coefficients;
use crate::image_from_f64;
use sitk_core::{Image, NeighborhoodIterator, PixelId, ZeroFluxNeumannBoundaryCondition};

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
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

    // Every axis of this window has extent 3, so the window slot of a one-pixel
    // step along axis `a` is `3^a` — exactly the enumeration `Neighborhood::get`
    // re-derived from an ND offset vector on every single tap.
    let center_slot = iter.len() / 2;
    let window_stride = window_strides(dim);

    // Parallel over output voxels. `dx` and `dxx` are per-task scratch, fully
    // overwritten for each voxel (every `a` in `0..dim` is assigned), never
    // carried between them. Each voxel's sums run over its own window in the same
    // order as before, and nothing accumulates across voxels, so no thread count
    // can reach the arithmetic.
    let out = iter.par_map_window_init(
        || (vec![0.0f64; dim], vec![0.0f64; dim]),
        |(dx, dxx), _, w| {
            for a in 0..dim {
                let mut sx = 0.0;
                let mut sxx = 0.0;
                for (k, &delta) in taps.iter().enumerate() {
                    let v = w.get_f64((center_slot as i64 + delta * window_stride[a]) as usize);
                    sx += d1[k] * v;
                    sxx += d2[k] * v;
                }
                dx[a] = sx;
                dxx[a] = sxx;
            }

            let mut deriv = 0.0;
            for i in 0..dim {
                for j in (i + 1)..dim {
                    let (si, sj) = (window_stride[i], window_stride[j]);
                    let at =
                        |a: i64, b: i64| w.get_f64((center_slot as i64 + a * si + b * sj) as usize);
                    let m_m = at(-1, -1);
                    let m_p = at(-1, 1);
                    let p_p = at(1, 1);
                    let p_m = at(1, -1);
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
        },
    );
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

    let center_slot = iter_s.len() / 2;
    let window_stride = window_strides(dim);

    // This is the one stencil in the module that reads **two** aligned images at
    // the same voxel, which `par_map_window` alone does not cover: it hands out
    // one image's window. The second image's window is refilled into per-task
    // scratch instead (`window_buffer` once per task, `refill` per voxel), so
    // `deriv_field`'s window is still materialized exactly as the serial
    // `iter_s.zip(iter_d)` materialized it — same values, same order — while
    // `smoothed`'s is borrowed and neither is allocated per voxel.
    //
    // `dx`, `dx1` and the refilled window are scratch that each voxel fully
    // overwrites; nothing is carried between voxels, and nothing accumulates
    // across them.
    let out = iter_s.par_map_window_init(
        || {
            (
                iter_d.window_buffer(),
                vec![0i64; dim],
                vec![0.0f64; dim],
                vec![0.0f64; dim],
            )
        },
        |(nb_d, nd, dx, dx1), center, w_s| {
            iter_d.refill(center, nd, nb_d);

            let mut off = vec![0i64; dim];
            let mut grad_mag_sq = 0.0001;
            for a in 0..dim {
                let mut sx = 0.0;
                let mut sx1 = 0.0;
                for (k, &delta) in taps.iter().enumerate() {
                    let slot = (center_slot as i64 + delta * window_stride[a]) as usize;
                    off[a] = delta;
                    sx += d1[k] * w_s.get_f64(slot);
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
        },
    );
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
pub(crate) fn zero_crossing_values(
    img: &Image,
    foreground: f64,
    background: f64,
) -> Result<Vec<f64>> {
    let dim = img.dimension();
    let radius = vec![1usize; dim];
    let iter = NeighborhoodIterator::<f64, _>::new(img, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let center_slot = iter.len() / 2;
    let window_stride = window_strides(dim);

    // Parallel over output voxels: the label is decided entirely by the center and
    // its `2 * dim` city-block neighbors, walked in ITK's own order (all negative
    // directions, then all positive), with the same short-circuit on the first
    // qualifying sign change. Nothing crosses voxels.
    let out = iter.par_map_window(|_, w| {
        let this_one = w.get_f64(center_slot);
        let mut result = background;
        'search: for (pass, delta) in [(0, -1i64), (1, 1i64)] {
            // Axis order is ITK's own (`d = 0 .. dim`), and it is load-bearing:
            // the first qualifying sign change wins, so visiting the axes in a
            // different order could label a different pixel.
            for &stride in &window_stride {
                let that = w.get_f64((center_slot as i64 + delta * stride) as usize);

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
    });
    Ok(out)
}

/// Slot strides of a radius-1 window: the step, inside the window, of a
/// one-pixel move along image axis `d`.
///
/// Every axis of a radius-1 window has extent 3 and the window is laid out
/// dimension-0-fastest, so the stride along axis `d` is `3^d`. Every stencil in
/// this module has radius 1, and each one used to re-derive this per neighbor
/// access, from an ND offset vector it allocated per voxel — that is what
/// `Neighborhood::get` did in a loop. Computing it once per filter turns a
/// neighbor access into one addition, and is what lets these stencils read a
/// borrowed [`sitk_core::WindowView`] instead of a per-voxel `Neighborhood` copy.
fn window_strides(dim: usize) -> Vec<i64> {
    let mut strides = vec![0i64; dim];
    let mut stride = 1i64;
    for s in strides.iter_mut() {
        *s = stride;
        stride *= 3;
    }
    strides
}

/// An `f64` copy of `img`'s pixels with `img`'s geometry, used as the working
/// buffer for [`zero_crossing`] (matching [`crate::gradient`]'s own
/// `scratch_f64` helper).
fn scratch_f64(img: &Image) -> Result<Image> {
    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec()?)?;
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

    // 1. Gaussian smoothing (m_GaussianFilter): ITK's Canny leaves the
    //    DiscreteGaussianImageFilter at its defaults — MaximumKernelWidth 32,
    //    UseImageSpacing on.
    let smoothed = discrete_gaussian_f64(img, variance, maximum_error, 32, true)?;

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
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn canny_constant_image_has_no_edges_3d() {
        let img = Image::from_vec(&[7, 7, 7], vec![5.0f64; 343]).unwrap();
        let out =
            canny_edge_detection(&img, &[1.0, 1.0, 1.0], &[0.01, 0.01, 0.01], 1.0, 0.5).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v == 0.0));
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
        let vals = out.to_f64_vec().unwrap();
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
        let vals = out.to_f64_vec().unwrap();
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

/// Thread-count parity pins for the three window passes in this module.
///
/// All three are **summing** stencils in `f64` (the zero-crossing labeller sums
/// nothing itself, but the field it labels is one), so the non-vacuity guard is
/// the fold-order one, asserted on the volume the pins actually run on.
///
/// These pins are pass-level on purpose. Each pass is pinned against a hand-kept
/// copy of the exact serial loop that was deleted, on the same input, rather than
/// only end-to-end through [`canny_edge_detection`] — an end-to-end pin would let
/// a re-associated sum inside one pass hide behind the hysteresis threshold that
/// follows it. The public filter is pinned too, on top.
///
/// The three passes are `f64`-only by construction (they iterate an `f64`
/// scratch), so unlike the other stencil pins there is no `Float32` half here:
/// the widening path they would exercise does not exist in them. The public
/// filter is run on both pixel types.
///
/// `-0.0` exposure, per pass:
///
/// * `second_directional_derivative_field` — does not apply. `sx`/`sxx`/`deriv`
///   still start at `0.0` and take the same terms in the same order; no accumulate
///   became a store. (It *is* a second derivative, so it can emit `-0.0` — which is
///   exactly why the first term was left as an accumulation rather than folded into
///   an initializer, as it was in `laplacian_recursive_gaussian`.)
/// * `positional_gate_field` — does not apply, same reason. `grad_mag_sq` still
///   starts at its `0.0001` floor and `deriv_pos` still sums over the same axes in
///   the same order.
/// * `zero_crossing_values` — does not apply. It writes one of two caller-supplied
///   labels and never adds; its comparisons (`this_one == 0.0`) treat `-0.0` and
///   `0.0` alike, as they did before.
#[cfg(test)]
mod stencil_thread_parity {
    use super::*;
    use crate::stencil_test_support::{
        PIXELS, THREADS, assert_bits_eq, volume, window_sum_order_is_observable,
    };
    use sitk_core::{PixelId, parallel};

    // ---- the serial references: the exact loops that were deleted -----------

    fn second_directional_derivative_serial(smoothed: &Image) -> Vec<f64> {
        let dim = smoothed.dimension();
        let radius = vec![1usize; dim];
        let iter = NeighborhoodIterator::<f64, _>::new(
            smoothed,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();
        let d1 = derivative_operator_coefficients(1);
        let d2 = derivative_operator_coefficients(2);
        let taps = [-1i64, 0, 1];

        iter.map(|(_, nb)| {
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
        .collect()
    }

    fn positional_gate_serial(smoothed: &Image, deriv_field: &Image) -> Vec<f64> {
        let dim = smoothed.dimension();
        let radius = vec![1usize; dim];
        let iter_s = NeighborhoodIterator::<f64, _>::new(
            smoothed,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();
        let iter_d = NeighborhoodIterator::<f64, _>::new(
            deriv_field,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();
        let d1 = derivative_operator_coefficients(1);
        let taps = [-1i64, 0, 1];

        iter_s
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
            .collect()
    }

    fn zero_crossing_serial(img: &Image, foreground: f64, background: f64) -> Vec<f64> {
        let dim = img.dimension();
        let radius = vec![1usize; dim];
        let iter =
            NeighborhoodIterator::<f64, _>::new(img, &radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        iter.map(|(_, nb)| {
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
        .collect()
    }

    /// The `f64` working images the three passes consume: `smoothed` is the test
    /// volume itself, and `deriv` is the (serially computed) second-derivative
    /// field, which is what the other two passes are actually fed inside the
    /// filter.
    fn fields() -> (Image, Image) {
        let smoothed = volume(PixelId::Float64);
        let deriv_values = second_directional_derivative_serial(&smoothed);
        let mut deriv = Image::from_vec(smoothed.size(), deriv_values).unwrap();
        deriv.copy_geometry_from(&smoothed);
        (smoothed, deriv)
    }

    // ---- non-vacuity --------------------------------------------------------

    /// Every pass here sums `f64` terms over a radius-1 window. If those sums
    /// could not round on this volume, a re-association would be invisible and the
    /// pins would pass on a filter that summed in any order. This asserts the
    /// order is observable on the volume the pins run on.
    #[test]
    fn the_window_sum_order_is_observable() {
        let img = volume(PixelId::Float64);
        assert!(
            window_sum_order_is_observable(&img, &[1, 1, 1]),
            "no voxel changed bits when its window sum was reversed — this volume cannot \
             observe a re-association, so the pins below would pass even on a pass that \
             summed its taps in a different order"
        );
    }

    /// The zero-crossing labeller sums nothing, so the fold-order guard above does
    /// not speak for it. What it needs instead is that its output is not constant:
    /// an all-background (or all-foreground) label field would match a parallel
    /// pass that read entirely the wrong neighbors, and the pin would pass anyway.
    ///
    /// Both classes must therefore be well represented. On this volume the
    /// derivative field oscillates per voxel, so most voxels do sit next to a sign
    /// change and the marked class is the larger one — that is fine; what would not
    /// be fine is either class being empty or negligible.
    #[test]
    fn the_zero_crossing_output_is_not_constant() {
        let (_, deriv) = fields();
        let labels = zero_crossing_serial(&deriv, 1.0, 0.0);
        let n = labels.len();
        let marked = labels.iter().filter(|&&v| v == 1.0).count();
        let unmarked = n - marked;
        assert!(
            marked > n / 20 && unmarked > n / 20,
            "the zero-crossing pass labelled {marked} foreground and {unmarked} background \
             out of {n} — one class is negligible, so the label field is effectively \
             constant and the pin could not catch a wrong neighbor"
        );
    }

    // ---- the pins -----------------------------------------------------------

    #[test]
    fn second_directional_derivative_is_bit_identical_at_every_thread_count() {
        let (smoothed, _) = fields();
        let expected = second_directional_derivative_serial(&smoothed);
        for threads in THREADS {
            let got =
                parallel::with_threads(threads, || second_directional_derivative_field(&smoothed))
                    .unwrap();
            assert_bits_eq(
                &got,
                &expected,
                &format!("second_directional_derivative_field ({threads} threads)"),
            );
        }
    }

    #[test]
    fn positional_gate_is_bit_identical_at_every_thread_count() {
        let (smoothed, deriv) = fields();
        let expected = positional_gate_serial(&smoothed, &deriv);
        for threads in THREADS {
            let got = parallel::with_threads(threads, || positional_gate_field(&smoothed, &deriv))
                .unwrap();
            assert_bits_eq(
                &got,
                &expected,
                &format!("positional_gate_field ({threads} threads)"),
            );
        }
    }

    #[test]
    fn zero_crossing_values_is_bit_identical_at_every_thread_count() {
        let (_, deriv) = fields();
        let expected = zero_crossing_serial(&deriv, 1.0, 0.0);
        for threads in THREADS {
            let got =
                parallel::with_threads(threads, || zero_crossing_values(&deriv, 1.0, 0.0)).unwrap();
            assert_bits_eq(
                &got,
                &expected,
                &format!("zero_crossing_values ({threads} threads)"),
            );
        }
    }

    /// And the whole filter on top of the three passes, on both pixel types: the
    /// composed output must not move with the thread count either.
    #[test]
    fn canny_edge_detection_is_bit_identical_at_every_thread_count() {
        for pixel in PIXELS {
            let img = volume(pixel);
            let variance = vec![2.0f64, 2.0, 2.0];
            let error = vec![0.01f64, 0.01, 0.01];
            let expected = parallel::with_threads(1, || {
                canny_edge_detection(&img, &variance, &error, 12.0, 3.0)
            })
            .unwrap()
            .to_f64_vec()
            .unwrap();
            let marked = expected.iter().filter(|&&v| v != 0.0).count();
            assert!(
                marked > 100,
                "canny marked only {marked} voxels on {pixel:?} — too few to pin anything"
            );
            for threads in THREADS {
                let got = parallel::with_threads(threads, || {
                    canny_edge_detection(&img, &variance, &error, 12.0, 3.0)
                })
                .unwrap()
                .to_f64_vec()
                .unwrap();
                assert_bits_eq(
                    &got,
                    &expected,
                    &format!("canny_edge_detection({pixel:?}, {threads} threads)"),
                );
            }
        }
    }
}
