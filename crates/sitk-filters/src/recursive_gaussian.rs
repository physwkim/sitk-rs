//! Bit-exact Gaussian smoothing by the Deriche/Farnebäck recursive IIR filter,
//! porting `itk::RecursiveGaussianImageFilter` (zero-order / smoothing) and the
//! forward+backward recursion of `itk::RecursiveSeparableImageFilter`.
//!
//! Unlike the FIR [`smooth_gaussian`](crate::smooth_gaussian) — which samples a
//! truncated Gaussian kernel — this approximates the continuous Gaussian by a
//! fourth-order recursive filter whose cost is independent of `sigma`, and it
//! reproduces ITK's arithmetic operation-for-operation. The coefficients are
//! the improved Deriche set from Farnebäck & Westin, *"Improving Deriche-style
//! Recursive Gaussian Filters"* (J. Math. Imaging Vis. 2006, Table 3), exactly
//! as ITK's `SetUp` / `ComputeNCoefficients` / `ComputeDCoefficients` /
//! `ComputeRemainingCoefficients` compute them.
//!
//! `sigma` is per dimension in **physical units** (matching ITK's
//! `SmoothingRecursiveGaussianImageFilter` default): along axis `d` the Gaussian
//! standard deviation is `sigma[d]`, so in index units it is
//! `sigmad = sigma[d] / spacing[d]`. Axes are filtered in sequence (separable);
//! an axis with `sigma == 0` is left untouched. The boundary replicates the edge
//! value (the border value "extends to infinity"), matching ITK, so a constant
//! image and the DC component are preserved exactly.
//!
//! The recursion needs at least four pixels along each filtered axis (ITK's
//! `RecursiveSeparableImageFilter` requirement); a shorter filtered axis is an
//! [`AxisTooShortForRecursion`](crate::FilterError::AxisTooShortForRecursion)
//! error.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::Image;

/// Gaussian-smooth `img` with a per-dimension physical-space `sigma`, using the
/// recursive (IIR) filter that ports `itk::RecursiveGaussianImageFilter`.
///
/// Errors if `sigma` has the wrong length, any value is negative, or a filtered
/// axis (`sigma > 0`) has fewer than four pixels.
pub fn recursive_gaussian(img: &Image, sigma: &[f64]) -> Result<Image> {
    let dim = img.dimension();
    if sigma.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: sigma.len(),
        });
    }
    if sigma.iter().any(|&s| s < 0.0) {
        return Err(FilterError::InvalidSigma(sigma.to_vec()));
    }

    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let strides = strides(&size);
    let mut buf = img.to_f64_vec();

    for d in 0..dim {
        if sigma[d] <= 0.0 {
            continue;
        }
        if size[d] < 4 {
            return Err(FilterError::AxisTooShortForRecursion {
                axis: d,
                len: size[d],
            });
        }
        let coeff = Coefficients::zero_order(sigma[d] / spacing[d]);
        filter_axis(&mut buf, &size, &strides, d, &coeff);
    }

    image_from_f64(img.pixel_id(), &size, img, &buf)
}

/// The coefficients of the fourth-order recursion for one `sigmad`
/// (index-space sigma), zero-order (smoothing) case. Field names mirror ITK's
/// members: `n*` numerator (causal), `d*` denominator (shared), `m*` numerator
/// of the symmetric anti-causal pass, `bn*`/`bm*` the boundary coefficients.
#[derive(Clone, Copy, Debug)]
struct Coefficients {
    n0: f64,
    n1: f64,
    n2: f64,
    n3: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    bn1: f64,
    bn2: f64,
    bn3: f64,
    bn4: f64,
    bm1: f64,
    bm2: f64,
    bm3: f64,
    bm4: f64,
}

impl Coefficients {
    /// Zero-order (smoothing) coefficients for `sigmad > 0`, reproducing ITK's
    /// `SetUp(ZeroOrder)` → `ComputeD` → `ComputeN` → `ComputeRemaining(true)`.
    fn zero_order(sigmad: f64) -> Self {
        // Improved Deriche constants (Farnebäck & Westin 2006, Table 3), the
        // zero-order column ITK stores as A1[0], B1[0], W1, L1, A2[0], B2[0],
        // W2, L2 and reads with `order == ZeroOrder`.
        const A1: f64 = 1.3530;
        const B1: f64 = 1.8151;
        const W1: f64 = 0.6681;
        const L1: f64 = -1.3932;
        const A2: f64 = -0.3531;
        const B2: f64 = 0.0902;
        const W2: f64 = 2.0787;
        const L2: f64 = -1.3732;

        let sin1 = (W1 / sigmad).sin();
        let sin2 = (W2 / sigmad).sin();
        let cos1 = (W1 / sigmad).cos();
        let cos2 = (W2 / sigmad).cos();
        let exp1 = (L1 / sigmad).exp();
        let exp2 = (L2 / sigmad).exp();

        // ComputeDCoefficients: the shared denominator (recursion) coefficients.
        let d4 = exp1 * exp1 * exp2 * exp2;
        let d3 = -2.0 * cos1 * exp1 * exp2 * exp2 - 2.0 * cos2 * exp2 * exp1 * exp1;
        let d2 = 4.0 * cos2 * cos1 * exp1 * exp2 + exp1 * exp1 + exp2 * exp2;
        let d1 = -2.0 * (exp2 * cos2 + exp1 * cos1);
        let sd = 1.0 + d1 + d2 + d3 + d4;

        // ComputeNCoefficients: the causal numerator coefficients (pre-scaled).
        let mut n0 = A1 + A2;
        let mut n1 = exp2 * (B2 * sin2 - (A2 + 2.0 * A1) * cos2)
            + exp1 * (B1 * sin1 - (A1 + 2.0 * A2) * cos1);
        let mut n2 =
            ((A1 + A2) * cos2 * cos1 - B1 * cos2 * sin1 - B2 * cos1 * sin2) * 2.0 * exp1 * exp2
                + A2 * exp1 * exp1
                + A1 * exp2 * exp2;
        let mut n3 = exp2 * exp1 * exp1 * (B2 * sin2 - A2 * cos2)
            + exp1 * exp2 * exp2 * (B1 * sin1 - A1 * cos1);
        let sn = n0 + n1 + n2 + n3;

        // ZeroOrder normalization (SetUp): scale the numerator so the filter has
        // unity DC gain. `across_scale_normalization` is 1 for the zero order.
        let alpha0 = 2.0 * sn / sd - n0;
        n0 /= alpha0;
        n1 /= alpha0;
        n2 /= alpha0;
        n3 /= alpha0;

        // ComputeRemainingCoefficients(symmetric = true): the anti-causal
        // numerator M and the causal/anti-causal boundary coefficients. SN is
        // recomputed here from the *normalized* N.
        let m1 = n1 - d1 * n0;
        let m2 = n2 - d2 * n0;
        let m3 = n3 - d3 * n0;
        let m4 = -d4 * n0;

        let sn = n0 + n1 + n2 + n3;
        let sm = m1 + m2 + m3 + m4;

        let bn1 = d1 * sn / sd;
        let bn2 = d2 * sn / sd;
        let bn3 = d3 * sn / sd;
        let bn4 = d4 * sn / sd;

        let bm1 = d1 * sm / sd;
        let bm2 = d2 * sm / sd;
        let bm3 = d3 * sm / sd;
        let bm4 = d4 * sm / sd;

        Coefficients {
            n0,
            n1,
            n2,
            n3,
            d1,
            d2,
            d3,
            d4,
            m1,
            m2,
            m3,
            m4,
            bn1,
            bn2,
            bn3,
            bn4,
            bm1,
            bm2,
            bm3,
            bm4,
        }
    }
}

/// Filter every line of `buf` along axis `d` in place, gathering each line into
/// a contiguous buffer, running the recursion, and scattering it back.
fn filter_axis(buf: &mut [f64], size: &[usize], strides: &[usize], d: usize, coeff: &Coefficients) {
    let stride = strides[d];
    let ln = size[d];
    let mut line = vec![0.0f64; ln];
    let mut outs = vec![0.0f64; ln];
    let mut scratch = vec![0.0f64; ln];

    for p in 0..buf.len() {
        // Each line is identified by its first element (coordinate 0 on axis d).
        if (p / stride) % ln != 0 {
            continue;
        }
        for (k, slot) in line.iter_mut().enumerate() {
            *slot = buf[p + k * stride];
        }
        filter_line(&line, coeff, &mut outs, &mut scratch);
        for (k, &v) in outs.iter().enumerate() {
            buf[p + k * stride] = v;
        }
    }
}

/// One line through the fourth-order causal + anti-causal recursion, porting
/// `RecursiveSeparableImageFilter::FilterDataArray` operation-for-operation. The
/// border value is assumed to extend to infinity on both ends. Requires
/// `data.len() >= 4`; `outs` and `scratch` are caller-provided scratch buffers
/// of the same length as `data` (their prior contents are overwritten).
fn filter_line(data: &[f64], c: &Coefficients, outs: &mut [f64], scratch: &mut [f64]) {
    let ln = data.len();

    // ---- Causal (forward) pass ----
    let out_v1 = data[0];

    outs[0] = out_v1 * c.n0 + out_v1 * c.n1 + out_v1 * c.n2 + out_v1 * c.n3;
    outs[1] = data[1] * c.n0 + out_v1 * c.n1 + out_v1 * c.n2 + out_v1 * c.n3;
    outs[2] = data[2] * c.n0 + data[1] * c.n1 + out_v1 * c.n2 + out_v1 * c.n3;
    outs[3] = data[3] * c.n0 + data[2] * c.n1 + data[1] * c.n2 + out_v1 * c.n3;

    // The border value is multiplied by the boundary coefficients m_BNi.
    outs[0] -= out_v1 * c.bn1 + out_v1 * c.bn2 + out_v1 * c.bn3 + out_v1 * c.bn4;
    outs[1] -= outs[0] * c.d1 + out_v1 * c.bn2 + out_v1 * c.bn3 + out_v1 * c.bn4;
    outs[2] -= outs[1] * c.d1 + outs[0] * c.d2 + out_v1 * c.bn3 + out_v1 * c.bn4;
    outs[3] -= outs[2] * c.d1 + outs[1] * c.d2 + outs[0] * c.d3 + out_v1 * c.bn4;

    for i in 4..ln {
        outs[i] = data[i] * c.n0 + data[i - 1] * c.n1 + data[i - 2] * c.n2 + data[i - 3] * c.n3;
        outs[i] -=
            outs[i - 1] * c.d1 + outs[i - 2] * c.d2 + outs[i - 3] * c.d3 + outs[i - 4] * c.d4;
    }

    // ---- Anti-causal (backward) pass into scratch ----
    let out_v2 = data[ln - 1];

    scratch[ln - 1] = out_v2 * c.m1 + out_v2 * c.m2 + out_v2 * c.m3 + out_v2 * c.m4;
    scratch[ln - 2] = data[ln - 1] * c.m1 + out_v2 * c.m2 + out_v2 * c.m3 + out_v2 * c.m4;
    scratch[ln - 3] = data[ln - 2] * c.m1 + data[ln - 1] * c.m2 + out_v2 * c.m3 + out_v2 * c.m4;
    scratch[ln - 4] =
        data[ln - 3] * c.m1 + data[ln - 2] * c.m2 + data[ln - 1] * c.m3 + out_v2 * c.m4;

    // The border value is multiplied by the boundary coefficients m_BMi.
    scratch[ln - 1] -= out_v2 * c.bm1 + out_v2 * c.bm2 + out_v2 * c.bm3 + out_v2 * c.bm4;
    scratch[ln - 2] -= scratch[ln - 1] * c.d1 + out_v2 * c.bm2 + out_v2 * c.bm3 + out_v2 * c.bm4;
    scratch[ln - 3] -=
        scratch[ln - 2] * c.d1 + scratch[ln - 1] * c.d2 + out_v2 * c.bm3 + out_v2 * c.bm4;
    scratch[ln - 4] -=
        scratch[ln - 3] * c.d1 + scratch[ln - 2] * c.d2 + scratch[ln - 1] * c.d3 + out_v2 * c.bm4;

    // ITK's loop: for (i = ln - 4; i > 0; i--) writes scratch[i - 1].
    let mut i = ln - 4;
    while i > 0 {
        scratch[i - 1] =
            data[i] * c.m1 + data[i + 1] * c.m2 + data[i + 2] * c.m3 + data[i + 3] * c.m4;
        scratch[i - 1] -= scratch[i] * c.d1
            + scratch[i + 1] * c.d2
            + scratch[i + 2] * c.d3
            + scratch[i + 3] * c.d4;
        i -= 1;
    }

    // Roll the anti-causal part into the output.
    for (o, &s) in outs.iter_mut().zip(scratch.iter()) {
        *o += s;
    }
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::{Image, PixelId};

    #[test]
    fn zero_sigma_is_identity() {
        let img = Image::from_vec(&[6, 5], (0..30).map(|v| v as f64).collect()).unwrap();
        let out = recursive_gaussian(&img, &[0.0, 0.0]).unwrap();
        assert_eq!(out.to_f64_vec(), img.to_f64_vec());
    }

    #[test]
    fn constant_image_is_preserved() {
        // Unity DC gain (the alpha0 normalization) plus edge-replicating borders
        // keep a constant exactly.
        let img = Image::from_vec(&[10, 10], vec![5.0; 100]).unwrap();
        let out = recursive_gaussian(&img, &[2.0, 2.0]).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 5.0).abs() < 1e-10, "constant not preserved: {v}");
        }
    }

    #[test]
    fn one_dimensional_constant_is_preserved_exactly() {
        // A single line stresses the boundary coefficients directly.
        let img = Image::from_vec(&[16, 1], vec![3.5; 16]).unwrap();
        let out = recursive_gaussian(&img, &[1.5, 0.0]).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 3.5).abs() < 1e-10, "1-D constant not preserved: {v}");
        }
    }

    #[test]
    fn interior_impulse_conserves_mass_and_is_symmetric() {
        // A single impulse well away from the border keeps its total mass (unity
        // DC gain) and spreads symmetrically about the center on both axes,
        // confirming the separable pass is applied correctly on each axis. The
        // grid is sized so the heavier IIR tails are fully contained (at n=81,
        // sigma=2 the edge leakage is ~1e-10).
        let n = 81;
        let c = n / 2;
        let mut data = vec![0.0f64; n * n];
        data[c * n + c] = 100.0;
        let img = Image::from_vec(&[n, n], data).unwrap();
        let v = recursive_gaussian(&img, &[2.0, 2.0]).unwrap().to_f64_vec();

        let total: f64 = v.iter().sum();
        assert!((total - 100.0).abs() < 1e-6, "mass not conserved: {total}");

        let peak = v[c * n + c];
        assert!(peak < 100.0 && peak > 0.0, "peak not spread: {peak}");
        assert!(
            (v[c * n + (c - 1)] - v[c * n + (c + 1)]).abs() < 1e-9,
            "x asymmetric"
        );
        assert!(
            (v[(c - 1) * n + c] - v[(c + 1) * n + c]).abs() < 1e-9,
            "y asymmetric"
        );
    }

    #[test]
    fn impulse_response_width_matches_the_itk_recursive_gaussian() {
        // The zero-order recursive filter approximates a Gaussian, but its
        // effective width is a *fixed* fraction of the nominal sigma: the second
        // moment of the impulse response is 0.93800 * sigma^2 at every scale
        // (measured identical to 5 digits at sigma = 2, 4, 8), whereas the FIR
        // that samples the true kernel gives ~sigma^2. This ~6.2% narrowing is a
        // genuine property of the Farnebäck ZeroOrder coefficients as ITK uses
        // them, not truncation. Pinning the ratio guards the coefficient math
        // against a transposed constant.
        let n = 201;
        let center = n / 2;
        let mut data = vec![0.0f64; n];
        data[center] = 1.0;
        let img = Image::from_vec(&[n, 1], data).unwrap();

        let sigma = 4.0;
        let out = recursive_gaussian(&img, &[sigma, 0.0])
            .unwrap()
            .to_f64_vec();

        // On this grid (edges at 25*sigma) the tails are fully contained, so the
        // DC gain (unit mass) shows to near machine precision.
        let mass: f64 = out.iter().sum();
        assert!((mass - 1.0).abs() < 1e-7, "impulse mass: {mass}");

        let var: f64 = out
            .iter()
            .enumerate()
            .map(|(i, &w)| w * (i as f64 - center as f64).powi(2))
            .sum();
        let ratio = var / (sigma * sigma);
        assert!(
            (0.935..=0.941).contains(&ratio),
            "variance/sigma^2 ratio {ratio} outside the recursive filter's 0.938 band"
        );
    }

    #[test]
    fn approximates_the_fir_gaussian_on_a_smooth_blob() {
        // The recursive and FIR filters approximate the same continuous Gaussian,
        // so on a smooth signal well inside the borders they agree closely.
        let n = 48;
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f64 - 24.0, y as f64 - 24.0);
                data[y * n + x] = (-(dx * dx + dy * dy) / 60.0).exp();
            }
        }
        let img = Image::from_vec(&[n, n], data).unwrap();

        let rec = recursive_gaussian(&img, &[2.0, 2.0]).unwrap().to_f64_vec();
        let fir = crate::smooth_gaussian(&img, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec();

        // Compare the interior (away from the borders) where both are accurate.
        let mut max_abs = 0.0f64;
        for y in 8..n - 8 {
            for x in 8..n - 8 {
                max_abs = max_abs.max((rec[y * n + x] - fir[y * n + x]).abs());
            }
        }
        assert!(max_abs < 5e-3, "recursive vs FIR interior diff {max_abs}");
    }

    #[test]
    fn physical_sigma_accounts_for_spacing() {
        // With spacing 2, a physical sigma of 2 is only 1 voxel of blur, so an
        // impulse spreads less (higher retained peak) than with spacing 1.
        let n = 41;
        let mut data = vec![0.0f64; n * n];
        data[20 * n + 20] = 100.0;

        let mut fine = Image::from_vec(&[n, n], data.clone()).unwrap();
        fine.set_spacing(&[1.0, 1.0]).unwrap();
        let mut coarse = Image::from_vec(&[n, n], data).unwrap();
        coarse.set_spacing(&[2.0, 2.0]).unwrap();

        let peak_fine = recursive_gaussian(&fine, &[2.0, 2.0]).unwrap().to_f64_vec()[20 * n + 20];
        let peak_coarse = recursive_gaussian(&coarse, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec()[20 * n + 20];
        assert!(
            peak_coarse > peak_fine,
            "coarser spacing should blur less: {peak_coarse} vs {peak_fine}"
        );
    }

    #[test]
    fn wrong_sigma_length_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian(&img, &[1.0]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn negative_sigma_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian(&img, &[-1.0, 1.0]),
            Err(FilterError::InvalidSigma(_))
        ));
    }

    #[test]
    fn short_filtered_axis_is_rejected() {
        // Fewer than four pixels along a filtered axis cannot feed the
        // fourth-order recursion (ITK throws the same requirement).
        let img = Image::new(&[3, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian(&img, &[1.0, 1.0]),
            Err(FilterError::AxisTooShortForRecursion { axis: 0, len: 3 })
        ));
    }

    #[test]
    fn short_axis_is_fine_when_not_filtered() {
        // A short axis is only a problem if it is actually filtered (sigma > 0).
        let img = Image::from_vec(&[3, 8], vec![2.0; 24]).unwrap();
        let out = recursive_gaussian(&img, &[0.0, 1.0]).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }
}
