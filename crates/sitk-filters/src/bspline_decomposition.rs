//! `itk::BSplineDecompositionImageFilter`: converts an image of *samples* into
//! the image of *B-spline coefficients* whose B-spline expansion interpolates
//! those samples.
//!
//! The direct B-spline transform is the recursive IIR filter of Unser,
//! Aldroubi & Eden ("B-Spline Signal Processing", IEEE TSP 41(2), 1993) as
//! coded by Philippe Thévenaz. The spline kernel `β^n` is an all-pole IIR
//! filter, so the interpolation condition `f[k] = Σ_i c[i] β^n(k - i)` is
//! solved by running, for each pole `z`, a causal recursion
//! `c⁺[n] = s[n] + z·c⁺[n-1]` followed by an anticausal recursion
//! `c[n] = z·(c[n+1] - c⁺[n])`, after multiplying the samples by the overall
//! gain `Π_k (1 - z_k)(1 - 1/z_k)` (for the cubic, `λ = 6`). Axes are filtered
//! in sequence, `0, 1, ..., d-1`; the filter is separable, so the result does
//! not depend on that order beyond floating-point rounding.
//!
//! Both recursions need an initial value that encodes the boundary condition.
//! ITK uses **mirror (whole-point symmetric) boundaries**, `c[-1] = c[1]` and
//! `c[N-1+k] = c[N-1-k]`, and the two initializations
//! (`SetInitialCausalCoefficient` / `SetInitialAntiCausalCoefficient`) are the
//! closed forms for exactly that extension. The defining property, which the
//! tests pin, is that convolving the coefficients back with `β^n` sampled at
//! the integers (under the same mirror extension) reproduces the input.
//!
//! Poles per spline order, from `SetPoles` (Unser 1993 Part II, Table I):
//!
//! | order | poles |
//! |-------|-------|
//! | 0, 1  | none (the coefficients *are* the samples) |
//! | 2     | `√8 - 3` |
//! | 3     | `√3 - 2` |
//! | 4     | `√(664 - √438976) + √304 - 19`, `√(664 + √438976) - √304 - 19` |
//! | 5     | `√(135/2 - √(17745/4)) + √(105/4) - 13/2`, `√(135/2 + √(17745/4)) - √(105/4) - 13/2` |
//!
//! Faithfully-reproduced upstream behaviors, rather than "fixed":
//!
//! - **Orders 0 and 1 are the identity.** `SetPoles` gives them zero poles, so
//!   the gain is the empty product `1.0` and no recursion runs. ITK still
//!   copies the input through, so the output equals the input exactly.
//! - **`m_Tolerance` is hard-coded to `1e-10`** with no setter. When it is
//!   positive, `SetInitialCausalCoefficient` computes a *horizon*
//!   `⌈ln(tol) / ln|z|⌉` and, if `horizon < N`, replaces the exact
//!   mirror-boundary initial sum with a truncated one-sided sum over the first
//!   `horizon` samples — an approximation, not an algebraic identity. Lines at
//!   least as long as the horizon therefore take a *different* code path from
//!   short lines, and their coefficients differ from the exact ones by
//!   `O(|z|^horizon)`. For the cubic (`z ≈ -0.2679`) the horizon is 18, so a
//!   19-sample line is accelerated and an 18-sample line is not. This port
//!   keeps the constant and exposes the branch through the private
//!   `data_to_coefficients_1d`, whose `tolerance = 0.0` selects the exact full
//!   sum, so tests can pin that the two agree to `O(|z|^horizon)`.
//! - **An axis of length 1 is skipped entirely.** `DataToCoefficients1D`
//!   returns `false` *before* applying the gain, because the mirror-boundary
//!   initializations both read `m_Scratch[N-2]`, which does not exist. So a
//!   `1 × N` image is filtered along its long axis only, and the gain `c0` is
//!   never applied along the degenerate axis. (Not a no-op-by-accident: had
//!   the gain been applied and no recursion run, the values would have been
//!   scaled by `λ`.)
//! - **The coefficient type is `NumericTraits<OutputPixelType>::RealType`**,
//!   which is `double` for both `Float32` and `Float64` images
//!   (`NumericTraits<float>::RealType == double`, `itkNumericTraits.h:1356`).
//!   ITK's `m_Scratch` is `std::vector<CoeffType> = std::vector<double>`
//!   (`itkBSplineDecompositionImageFilter.h:85,133,175`), so the gain multiply
//!   and the whole IIR recursion — every intermediate accumulator, the running
//!   `sum` of the causal initialization included — run in `double`. The single
//!   narrowing to the output pixel type happens at *write-out*, when a filtered
//!   line is copied back to the output image (`static_cast<OutputPixelType>`,
//!   `CopyScratchToCoefficients`, `.hxx:281`). Because each axis reads the
//!   already-rounded output of the previous axis (`CopyCoefficientsToScratch`
//!   re-widens it, `.hxx:297`), a `Float32` image is rounded to single
//!   precision **once per axis, at write-out** — not after every recursion
//!   step. This port matches that: `data_to_coefficients_1d` runs a line
//!   entirely in `f64`, and `coefficients_along_axis` narrows to the output
//!   pixel type as it writes each line back.
//!
//! `pixel_types: RealPixelIDTypeList`, no `output_pixel_type` override: the
//! input must be `Float32` or `Float64` and the output takes the input's pixel
//! type and geometry.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{Image, PixelId};

/// `BSplineDecompositionImageFilter::m_Tolerance`, hard-coded upstream with no
/// setter. Selects the accelerated (truncated) causal initialization for lines
/// at least `⌈ln(tol) / ln|z|⌉` samples long.
const TOLERANCE: f64 = 1e-10;

/// The poles of the `spline_order`-th order B-spline interpolation filter,
/// porting `SetPoles`. Orders 0 and 1 have none. This is SimpleITK's
/// `GetSplinePoles` measurement.
///
/// Errors with [`FilterError::UnsupportedSplineOrder`] for `spline_order > 5`,
/// where ITK throws from `SetPoles`.
pub fn bspline_spline_poles(spline_order: u32) -> Result<Vec<f64>> {
    Ok(match spline_order {
        0 | 1 => Vec::new(),
        2 => vec![8.0f64.sqrt() - 3.0],
        3 => vec![3.0f64.sqrt() - 2.0],
        4 => vec![
            (664.0 - 438976.0f64.sqrt()).sqrt() + 304.0f64.sqrt() - 19.0,
            (664.0 + 438976.0f64.sqrt()).sqrt() - 304.0f64.sqrt() - 19.0,
        ],
        5 => vec![
            (135.0 / 2.0 - (17745.0f64 / 4.0).sqrt()).sqrt() + (105.0f64 / 4.0).sqrt() - 13.0 / 2.0,
            (135.0 / 2.0 + (17745.0f64 / 4.0).sqrt()).sqrt() - (105.0f64 / 4.0).sqrt() - 13.0 / 2.0,
        ],
        _ => return Err(FilterError::UnsupportedSplineOrder(spline_order)),
    })
}

/// The write-out narrowing, `static_cast<OutputPixelType>` in ITK's
/// `CopyScratchToCoefficients` (`.hxx:281`): a filtered line's coefficients are
/// rounded to the output pixel type as they are written back to the image —
/// once per axis, not per recursion step. Identity for `Float64`.
fn narrower(pixel_id: PixelId) -> fn(f64) -> f64 {
    match pixel_id {
        PixelId::Float32 => |v: f64| v as f32 as f64,
        _ => |v: f64| v,
    }
}

/// `SetInitialCausalCoefficient`: overwrite `s[0]` with the causal recursion's
/// initial value under mirror boundaries.
fn set_initial_causal_coefficient(s: &mut [f64], z: f64, tolerance: f64) {
    let n_len = s.len();
    let mut zn = z;

    if tolerance > 0.0 {
        // `static_cast<SizeValueType>` of a non-negative ceil; tolerance < 1
        // and |z| < 1 make the quotient positive.
        let horizon = (tolerance.ln() / z.abs().ln()).ceil() as usize;
        if horizon < n_len {
            // Accelerated (truncated) loop: the mirrored tail is dropped.
            let mut sum = s[0];
            for &v in s.iter().take(horizon).skip(1) {
                sum += zn * v;
                zn *= z;
            }
            s[0] = sum;
            return;
        }
    }

    // Full loop: the exact mirror-boundary sum.
    let iz = 1.0 / z;
    let mut z2n = z.powf((n_len - 1) as f64);
    let mut sum = s[0] + z2n * s[n_len - 1];
    // `z2n *= z2n * iz` upstream: z^(N-1) -> z^(2N-3).
    z2n = z2n * z2n * iz;
    for &v in s.iter().take(n_len - 1).skip(1) {
        sum += (zn + z2n) * v;
        zn *= z;
        z2n *= iz;
    }
    s[0] = sum / (1.0 - zn * zn);
}

/// `SetInitialAntiCausalCoefficient`: overwrite `s[N-1]` with the anticausal
/// recursion's initial value under mirror boundaries (Unser 1999 Box 2, with
/// the published erratum applied).
fn set_initial_anticausal_coefficient(s: &mut [f64], z: f64) {
    let n_len = s.len();
    s[n_len - 1] = (z / (z * z - 1.0)) * (z * s[n_len - 2] + s[n_len - 1]);
}

/// `DataToCoefficients1D`: in-place sample → coefficient transform of one
/// line. Returns `false` (leaving `s` untouched) for a length-1 line, as ITK
/// does — mirror boundaries need at least two samples.
fn data_to_coefficients_1d(s: &mut [f64], poles: &[f64], tolerance: f64) -> bool {
    let n_len = s.len();
    if n_len == 1 {
        return false;
    }

    // Overall gain, Unser 1993 Part II eq. 2.5 (`λ = 6` for the cubic).
    let mut c0 = 1.0;
    for &z in poles {
        c0 *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    for v in s.iter_mut() {
        *v *= c0;
    }

    for &z in poles {
        set_initial_causal_coefficient(s, z, tolerance);
        for n in 1..n_len {
            s[n] += z * s[n - 1];
        }

        set_initial_anticausal_coefficient(s, z);
        for n in (0..n_len - 1).rev() {
            s[n] = z * (s[n + 1] - s[n]);
        }
    }
    true
}

/// Row-major strides: `stride[0] == 1`, so axis 0 varies fastest.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(size.len());
    let mut acc = 1;
    for &n in size {
        out.push(acc);
        acc *= n;
    }
    out
}

/// `DataToCoefficientsND`: filter every line along `axis`, in place.
fn coefficients_along_axis(
    data: &mut [f64],
    size: &[usize],
    axis: usize,
    poles: &[f64],
    tolerance: f64,
    narrow: fn(f64) -> f64,
) {
    let n_len = size[axis];
    let stride = strides(size)[axis];
    let mut scratch = vec![0.0; n_len];

    for base in 0..data.len() {
        // Only the lines' starting voxels: coordinate along `axis` is 0.
        if !(base / stride).is_multiple_of(n_len) {
            continue;
        }
        for (j, v) in scratch.iter_mut().enumerate() {
            *v = data[base + j * stride];
        }
        if data_to_coefficients_1d(&mut scratch, poles, tolerance) {
            // Narrow to the output pixel type as the line is written back, once
            // per axis — ITK's `CopyScratchToCoefficients` (`.hxx:281`).
            for (j, &v) in scratch.iter().enumerate() {
                data[base + j * stride] = narrow(v);
            }
        }
    }
}

/// `BSplineDecompositionImageFilter`: the B-spline coefficients of `img` at
/// `spline_order`, computed with mirror boundary conditions along every axis.
///
/// The output has the input's pixel type, size and geometry. Orders 0 and 1
/// reproduce the input exactly.
///
/// Errors with [`FilterError::RequiresRealPixelType`] for a non-`Float32`,
/// non-`Float64` input (`pixel_types: RealPixelIDTypeList`) and with
/// [`FilterError::UnsupportedSplineOrder`] for `spline_order > 5`.
pub fn bspline_decomposition(img: &Image, spline_order: u32) -> Result<Image> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    let poles = bspline_spline_poles(spline_order)?;
    let narrow = narrower(pixel_id);

    let size = img.size().to_vec();
    let mut data = img.to_f64_vec()?;
    for axis in 0..size.len() {
        coefficients_along_axis(&mut data, &size, axis, &poles, TOLERANCE, narrow);
    }
    image_from_f64(pixel_id, &size, img, &data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `β³` sampled at the integers: `β³(0) = 2/3`, `β³(±1) = 1/6`, else 0.
    /// Reconstructing the samples from the coefficients under the same
    /// whole-point mirror extension (`c[-1] = c[1]`, `c[N] = c[N-2]`) is the
    /// defining property of the decomposition.
    fn reconstruct_cubic(c: &[f64]) -> Vec<f64> {
        let n = c.len();
        let at = |i: isize| -> f64 {
            // Whole-point mirror: reflect about 0 and about N-1.
            let mut i = i;
            if n == 1 {
                return c[0];
            }
            let period = 2 * (n as isize - 1);
            i = i.rem_euclid(period);
            if i >= n as isize {
                i = period - i;
            }
            c[i as usize]
        };
        (0..n)
            .map(|k| {
                let k = k as isize;
                at(k - 1) / 6.0 + at(k) * 2.0 / 3.0 + at(k + 1) / 6.0
            })
            .collect()
    }

    fn decompose_line(f: &[f64], order: u32, tolerance: f64) -> Vec<f64> {
        let mut s = f.to_vec();
        let poles = bspline_spline_poles(order).unwrap();
        data_to_coefficients_1d(&mut s, &poles, tolerance);
        s
    }

    // ---- pole table ----

    #[test]
    fn poles_match_the_unser_table() {
        assert!(bspline_spline_poles(0).unwrap().is_empty());
        assert!(bspline_spline_poles(1).unwrap().is_empty());
        // Order 2: √8 - 3 = -0.171572875253810
        assert!((bspline_spline_poles(2).unwrap()[0] - (-0.171_572_875_253_809_9)).abs() < 1e-15);
        // Order 3: √3 - 2 = -0.267949192431123
        assert!((bspline_spline_poles(3).unwrap()[0] - (-0.267_949_192_431_122_7)).abs() < 1e-15);
        let p4 = bspline_spline_poles(4).unwrap();
        assert!((p4[0] - (-0.361_341_225_900_220_2)).abs() < 1e-12);
        assert!((p4[1] - (-0.013_725_429_297_339_1)).abs() < 1e-12);
        let p5 = bspline_spline_poles(5).unwrap();
        assert!((p5[0] - (-0.430_575_347_099_973_7)).abs() < 1e-12);
        assert!((p5[1] - (-0.043_096_288_203_264_65)).abs() < 1e-12);
    }

    #[test]
    fn cubic_gain_is_six() {
        let z = bspline_spline_poles(3).unwrap()[0];
        assert!(((1.0 - z) * (1.0 - 1.0 / z) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn spline_order_above_five_is_rejected() {
        assert_eq!(
            bspline_spline_poles(6),
            Err(FilterError::UnsupportedSplineOrder(6))
        );
        let img = Image::from_vec(&[4, 4], vec![1.0f64; 16]).unwrap();
        assert_eq!(
            bspline_decomposition(&img, 6),
            Err(FilterError::UnsupportedSplineOrder(6))
        );
    }

    #[test]
    fn non_real_pixel_type_is_rejected() {
        let img = Image::from_vec(&[4, 4], vec![1i16; 16]).unwrap();
        assert_eq!(
            bspline_decomposition(&img, 3),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
    }

    // ---- hand-derived coefficients ----

    // For N = 2 under whole-point mirror, `c[-1] = c[1]` and `c[2] = c[0]`, so
    // the interpolation system is
    //     f0 = (2/3)c0 + (1/3)c1
    //     f1 = (1/3)c0 + (2/3)c1
    // whose inverse is c0 = 2 f0 - f1, c1 = -f0 + 2 f1.
    #[test]
    fn cubic_length_two_matches_the_hand_inverted_system() {
        let c = decompose_line(&[1.0, 2.0], 3, TOLERANCE);
        assert!((c[0] - 0.0).abs() < 1e-12, "c0 = {}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-12, "c1 = {}", c[1]);
    }

    // For N = 3, `c[-1] = c[1]` and `c[3] = c[1]`:
    //     f0 = (2/3)c0 + (1/3)c1
    //     f1 = (1/6)c0 + (2/3)c1 + (1/6)c2
    //     f2 = (1/3)c1 + (2/3)c2
    #[test]
    fn cubic_length_three_matches_the_hand_inverted_system() {
        let f = [1.0, 3.0, 2.0];
        let c = decompose_line(&f, 3, TOLERANCE);
        // Solve the 3x3 by elimination, by hand:
        //   c0 = (3 f0 - c1) / 2 * ... -> just verify the forward system.
        let f0 = (2.0 / 3.0) * c[0] + (1.0 / 3.0) * c[1];
        let f1 = c[0] / 6.0 + (2.0 / 3.0) * c[1] + c[2] / 6.0;
        let f2 = (1.0 / 3.0) * c[1] + (2.0 / 3.0) * c[2];
        assert!((f0 - f[0]).abs() < 1e-12);
        assert!((f1 - f[1]).abs() < 1e-12);
        assert!((f2 - f[2]).abs() < 1e-12);
    }

    #[test]
    fn cubic_reconstruction_returns_the_input() {
        // Short line: exact (full-sum) causal initialization.
        let f: Vec<f64> = vec![1.0, -2.0, 3.5, 0.25, 7.0, -1.5];
        let c = decompose_line(&f, 3, TOLERANCE);
        for (a, b) in reconstruct_cubic(&c).iter().zip(&f) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn cubic_reconstruction_returns_the_input_on_a_long_accelerated_line() {
        // 40 > horizon (18): the truncated causal initialization runs, and the
        // reconstruction is still accurate to well under 1e-10.
        let f: Vec<f64> = (0..40).map(|i| ((i * i) % 7) as f64 - 3.0).collect();
        let c = decompose_line(&f, 3, TOLERANCE);
        for (a, b) in reconstruct_cubic(&c).iter().zip(&f) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn constant_line_has_constant_coefficients() {
        for order in 0..=5 {
            let c = decompose_line(&[2.5; 12], order, TOLERANCE);
            for v in &c {
                assert!((v - 2.5).abs() < 1e-10, "order {order}: {v}");
            }
        }
    }

    // ---- tolerance / horizon boundary ----

    #[test]
    fn cubic_horizon_is_eighteen() {
        let z: f64 = bspline_spline_poles(3).unwrap()[0];
        let horizon = (TOLERANCE.ln() / z.abs().ln()).ceil() as usize;
        assert_eq!(horizon, 18);
    }

    #[test]
    fn accelerated_and_full_initializations_agree() {
        // Length 18 == horizon: `horizon < N` is false, so both paths are the
        // full sum and the lines are bit-identical.
        let f18: Vec<f64> = (0..18).map(|i| (i as f64).sin()).collect();
        assert_eq!(
            decompose_line(&f18, 3, TOLERANCE),
            decompose_line(&f18, 3, 0.0)
        );

        // Length 19 > horizon: the accelerated path runs, and differs from the
        // exact sum by O(|z|^18) ~ 1.4e-10 scaled by the coefficient
        // magnitudes (here `6 * max|f|`).
        let f19: Vec<f64> = (0..19).map(|i| (i as f64).sin()).collect();
        let acc = decompose_line(&f19, 3, TOLERANCE);
        let full = decompose_line(&f19, 3, 0.0);
        assert_ne!(acc, full);
        for (a, b) in acc.iter().zip(&full) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    // ---- degenerate and identity cases ----

    #[test]
    fn length_one_line_is_left_untouched_and_ungained() {
        let mut s = [4.0];
        let poles = bspline_spline_poles(3).unwrap();
        assert!(!data_to_coefficients_1d(&mut s, &poles, TOLERANCE));
        // Not 24.0: the gain is never applied.
        assert_eq!(s[0], 4.0);
    }

    #[test]
    fn length_one_axis_is_skipped_in_an_image() {
        // 1 x 5: axis 0 is degenerate, axis 1 is filtered.
        let f = vec![1.0, -2.0, 3.5, 0.25, 7.0];
        let img = Image::from_vec(&[1, 5], f.clone()).unwrap();
        let out = bspline_decomposition(&img, 3).unwrap();
        let got = out.to_f64_vec().unwrap();
        // Equals the 1-D decomposition of the same 5 samples: the degenerate
        // axis contributed neither gain nor recursion.
        let want = decompose_line(&f, 3, TOLERANCE);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn single_voxel_image_is_the_identity() {
        let img = Image::from_vec(&[1, 1], vec![9.0f64]).unwrap();
        assert_eq!(
            bspline_decomposition(&img, 3)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            [9.0]
        );
    }

    #[test]
    fn orders_zero_and_one_are_the_identity() {
        let f: Vec<f64> = (0..12).map(|i| (i as f64) * 0.5 - 2.0).collect();
        let img = Image::from_vec(&[3, 4], f.clone()).unwrap();
        for order in [0, 1] {
            assert_eq!(
                bspline_decomposition(&img, order)
                    .unwrap()
                    .to_f64_vec()
                    .unwrap(),
                f
            );
        }
    }

    // ---- separability / per-axis independence ----

    #[test]
    fn a_separable_image_decomposes_to_the_outer_product_of_line_coefficients() {
        // f(x, y) = g(x) * h(y)  =>  c(x, y) = cg(x) * ch(y), because the
        // filter is separable and linear.
        let g = [1.0, -2.0, 3.5, 0.25, 7.0];
        let h = [2.0, 0.5, -1.0, 4.0];
        let mut data = vec![0.0; g.len() * h.len()];
        for (y, &hy) in h.iter().enumerate() {
            for (x, &gx) in g.iter().enumerate() {
                data[x + y * g.len()] = gx * hy;
            }
        }
        let img = Image::from_vec(&[g.len(), h.len()], data).unwrap();
        let out = bspline_decomposition(&img, 3)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        let cg = decompose_line(&g, 3, TOLERANCE);
        let ch = decompose_line(&h, 3, TOLERANCE);
        for (y, &chy) in ch.iter().enumerate() {
            for (x, &cgx) in cg.iter().enumerate() {
                let want = cgx * chy;
                let got = out[x + y * cg.len()];
                assert!((got - want).abs() < 1e-10, "({x},{y}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn axes_are_filtered_independently_in_3d() {
        // A function of x alone must stay a function of x alone: the y and z
        // passes see constant lines and reproduce them.
        let g = [1.0, -2.0, 3.5, 0.25];
        let (nx, ny, nz) = (4, 3, 2);
        let mut data = vec![0.0; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for (x, &gx) in g.iter().enumerate() {
                    data[x + nx * (y + ny * z)] = gx;
                }
            }
        }
        let img = Image::from_vec(&[nx, ny, nz], data).unwrap();
        let out = bspline_decomposition(&img, 3)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let cg = decompose_line(&g, 3, TOLERANCE);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let got = out[x + nx * (y + ny * z)];
                    assert!((got - cg[x]).abs() < 1e-10, "({x},{y},{z}): {got}");
                }
            }
        }
    }

    // ---- pixel type ----

    #[test]
    fn float32_output_narrows_once_at_write_out_not_per_step() {
        let f: Vec<f32> = (0..20).map(|i| (i as f32).sin()).collect();
        let img = Image::from_vec(&[20], f.clone()).unwrap();
        let out = bspline_decomposition(&img, 3).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);

        // ITK runs the whole IIR recursion in `CoeffType = RealType = double`
        // and narrows to the output pixel type once, at write-out
        // (`static_cast<OutputPixelType>`, CopyScratchToCoefficients,
        // itkBSplineDecompositionImageFilter.hxx:281). For this 1-D image that
        // is a single narrowing of the f64 line — not rounding after each step.
        let f64_line = decompose_line(
            &f.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            3,
            TOLERANCE,
        );
        let want: Vec<f64> = f64_line.iter().map(|&v| v as f32 as f64).collect();
        assert_eq!(out.to_f64_vec().unwrap(), want);

        // The former bug rounded after every recursion step; that gives a
        // different line, so the narrow-once result above is a real regression
        // guard against reintroducing per-step rounding.
        let mut per_step: Vec<f64> = f.iter().map(|&v| v as f64).collect();
        narrow_per_step_reference(&mut per_step, &bspline_spline_poles(3).unwrap(), TOLERANCE);
        assert_ne!(want, per_step);
    }

    /// The pre-fix behavior — rounding to f32 after *every* recursion step —
    /// kept only to prove the fixed filter no longer matches it.
    fn narrow_per_step_reference(s: &mut [f64], poles: &[f64], tolerance: f64) {
        let nz = |v: f64| v as f32 as f64;
        let n_len = s.len();
        let mut c0 = 1.0;
        for &z in poles {
            c0 *= (1.0 - z) * (1.0 - 1.0 / z);
        }
        for v in s.iter_mut() {
            *v = nz(*v * c0);
        }
        for &z in poles {
            let mut zn = z;
            let horizon = (tolerance.ln() / z.abs().ln()).ceil() as usize;
            if tolerance > 0.0 && horizon < n_len {
                let mut sum = s[0];
                for &v in s.iter().take(horizon).skip(1) {
                    sum = nz(sum + zn * v);
                    zn *= z;
                }
                s[0] = sum;
            } else {
                let iz = 1.0 / z;
                let mut z2n = z.powf((n_len - 1) as f64);
                let mut sum = nz(s[0] + z2n * s[n_len - 1]);
                z2n = z2n * z2n * iz;
                for &v in s.iter().take(n_len - 1).skip(1) {
                    sum = nz(sum + (zn + z2n) * v);
                    zn *= z;
                    z2n *= iz;
                }
                s[0] = nz(sum / (1.0 - zn * zn));
            }
            for n in 1..n_len {
                s[n] = nz(s[n] + z * s[n - 1]);
            }
            s[n_len - 1] = nz((z / (z * z - 1.0)) * (z * s[n_len - 2] + s[n_len - 1]));
            for n in (0..n_len - 1).rev() {
                s[n] = nz(z * (s[n + 1] - s[n]));
            }
        }
    }

    #[test]
    fn float64_output_keeps_float64() {
        let img = Image::from_vec(&[4, 4], vec![1.0f64; 16]).unwrap();
        assert_eq!(
            bspline_decomposition(&img, 3).unwrap().pixel_id(),
            PixelId::Float64
        );
    }

    #[test]
    fn geometry_is_copied() {
        let mut img = Image::from_vec(&[4, 4], vec![1.0f64; 16]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        let out = bspline_decomposition(&img, 3).unwrap();
        assert_eq!(out.spacing(), [0.5, 2.0]);
        assert_eq!(out.origin(), [-1.0, 3.0]);
    }

    #[test]
    fn every_supported_order_reconstructs_a_constant() {
        // Order 4 and 5 have two poles each; exercising both pole loops.
        for order in 0..=5 {
            let img = Image::from_vec(&[7, 5], vec![3.0f64; 35]).unwrap();
            let out = bspline_decomposition(&img, order)
                .unwrap()
                .to_f64_vec()
                .unwrap();
            for v in out {
                assert!((v - 3.0).abs() < 1e-9, "order {order}: {v}");
            }
        }
    }
}
