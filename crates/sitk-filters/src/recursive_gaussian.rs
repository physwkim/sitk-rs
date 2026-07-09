//! Bit-exact Gaussian smoothing and its 1st/2nd derivatives by the
//! Deriche/Farnebäck recursive IIR filter, porting
//! `itk::RecursiveGaussianImageFilter` (all three
//! `RecursiveGaussianImageFilterEnums::GaussianOrder` values — `ZeroOrder`,
//! `FirstOrder`, `SecondOrder`, see [`GaussianOrder`]) and the
//! forward+backward recursion of `itk::RecursiveSeparableImageFilter`.
//!
//! Unlike the FIR [`smooth_gaussian`](crate::smooth_gaussian) — which samples
//! a truncated Gaussian kernel — this approximates the continuous Gaussian
//! (or its derivative) by a fourth-order recursive filter whose cost is
//! independent of `sigma`, and it reproduces ITK's arithmetic
//! operation-for-operation. The coefficients are the improved Deriche set
//! from Farnebäck & Westin, *"Improving Deriche-style Recursive Gaussian
//! Filters"* (J. Math. Imaging Vis. 2006, Table 3): three columns of
//! `A1,B1,A2,B2` — one per [`GaussianOrder`] — sharing the same poles
//! `W1,L1,W2,L2`, exactly as ITK's `SetUp` / `ComputeNCoefficients` /
//! `ComputeDCoefficients` / `ComputeRemainingCoefficients` compute them.
//!
//! The forward/backward recursion itself
//! (`RecursiveSeparableImageFilter::FilterDataArray`, ported here as
//! [`filter_line`]/[`filter_axis`]) does not depend on the order at all —
//! only the coefficients [`Coefficients::new`] builds do, so all three
//! orders share the same recursion code:
//! - **Normalization.** `ZeroOrder` scales its numerator for unity DC gain
//!   (`alpha0`); `FirstOrder`/`SecondOrder` instead scale so the filter's
//!   gain matches the true derivative operator (`alpha1`/`alpha2`, derived
//!   from `SD`/`DD`/`ED` — the recursion denominator and its 1st/2nd
//!   "z-derivative" weighted sums). `SecondOrder` additionally blends the
//!   `ZeroOrder` and `SecondOrder` Farnebäck columns through a `beta`
//!   coefficient so the result has zero net DC gain.
//! - **Symmetry.** `ZeroOrder`/`SecondOrder` use a *symmetric* anti-causal
//!   numerator (`Mi = Ni - Di·N0`, matching a filter whose true impulse
//!   response is even); `FirstOrder` is *anti-symmetric*
//!   (`Mi = -(Ni - Di·N0)`, with `M4`'s sign flipped too), matching an odd
//!   impulse response. This is ITK's `ComputeRemainingCoefficients(symmetric)`
//!   flag.
//!
//! `NormalizeAcrossScale` (Lindeberg scale-space normalization) multiplies
//! `FirstOrder`/`SecondOrder` output by an extra `sigma^order` so a feature's
//! peak derivative response does not depend on the scale it is measured at;
//! it is off by default, matching ITK, and produces a plain (unnormalized)
//! derivative when off.
//!
//! `sigma` is per dimension in **physical units** (matching ITK's
//! `SmoothingRecursiveGaussianImageFilter` default): along axis `d` the
//! Gaussian standard deviation is `sigma[d]`, so in index units it is
//! `sigmad = sigma[d] / spacing[d]`. Axes are filtered in sequence
//! (separable); an axis with `sigma == 0` is left untouched. The boundary
//! replicates the edge value (the border value "extends to infinity"),
//! matching ITK, so a constant image and the DC component are preserved
//! exactly under `ZeroOrder`.
//!
//! The recursion needs at least four pixels along each filtered axis (ITK's
//! `RecursiveSeparableImageFilter` requirement); a shorter filtered axis is
//! an [`AxisTooShortForRecursion`](crate::FilterError::AxisTooShortForRecursion)
//! error.
//!
//! Two public entry points, both taking `sigma` per dimension:
//! - [`recursive_gaussian`] applies [`GaussianOrder::ZeroOrder`] (smoothing)
//!   to every dimension. This is the pre-existing signature, kept exactly as
//!   it was (rather than adding a defaulted parameter, which Rust has no
//!   syntax for) because `sitk-registration`'s multi-resolution pyramid
//!   already calls it and must keep compiling unchanged.
//! - [`recursive_gaussian_with_order`] additionally takes a per-dimension
//!   `orders: &[GaussianOrder]` slice — so a caller differentiates along one
//!   axis while smoothing the others, matching how ITK composes per-axis
//!   `RecursiveGaussianImageFilter`s in `GradientRecursiveGaussianImageFilter`
//!   — and a `normalize_across_scale` flag. `recursive_gaussian` is a thin
//!   wrapper over it (`ZeroOrder` on every axis, `normalize_across_scale =
//!   false`).

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::Image;

/// Order of the Gaussian derivative approximated by the recursive filter,
/// matching ITK's `RecursiveGaussianImageFilterEnums::GaussianOrder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianOrder {
    /// Convolve with the Gaussian itself (smoothing). ITK's `ZeroOrder`.
    ZeroOrder,
    /// Convolve with the first derivative of the Gaussian. ITK's `FirstOrder`.
    FirstOrder,
    /// Convolve with the second derivative of the Gaussian. ITK's
    /// `SecondOrder`.
    SecondOrder,
}

/// Gaussian-smooth `img` with a per-dimension physical-space `sigma`, using
/// the recursive (IIR) filter that ports `itk::RecursiveGaussianImageFilter`
/// with [`GaussianOrder::ZeroOrder`] on every dimension. See
/// [`recursive_gaussian_with_order`] to take a derivative instead.
///
/// Errors if `sigma` has the wrong length, any value is negative, or a filtered
/// axis (`sigma > 0`) has fewer than four pixels.
pub fn recursive_gaussian(img: &Image, sigma: &[f64]) -> Result<Image> {
    let orders = vec![GaussianOrder::ZeroOrder; sigma.len()];
    recursive_gaussian_with_order(img, sigma, &orders, false)
}

/// Gaussian-smooth or -differentiate `img` with a per-dimension physical-space
/// `sigma` and a per-dimension [`GaussianOrder`], using the same recursive
/// (IIR) filter as [`recursive_gaussian`]. To take, say, the x-derivative of a
/// 2-D image while smoothing along y (matching how ITK's
/// `GradientRecursiveGaussianImageFilter` composes per-axis filters), pass
/// `orders = &[GaussianOrder::FirstOrder, GaussianOrder::ZeroOrder]`.
///
/// `normalize_across_scale` applies ITK's Lindeberg scale-space
/// normalization: `FirstOrder`/`SecondOrder` output is scaled by an extra
/// `sigma^order` so a feature's peak derivative response does not depend on
/// the scale it is measured at (`ZeroOrder` is unaffected either way). Off by
/// default in ITK; `false` gives a plain (unnormalized) derivative.
///
/// Errors if `sigma` or `orders` has the wrong length, any `sigma` value is
/// negative, or a filtered axis (`sigma > 0`) has fewer than four pixels.
pub fn recursive_gaussian_with_order(
    img: &Image,
    sigma: &[f64],
    orders: &[GaussianOrder],
    normalize_across_scale: bool,
) -> Result<Image> {
    let dim = img.dimension();
    if sigma.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: sigma.len(),
        });
    }
    if orders.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: orders.len(),
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
        let coeff = Coefficients::new(
            orders[d],
            sigma[d] / spacing[d],
            sigma[d],
            normalize_across_scale,
        );
        filter_axis(&mut buf, &size, &strides, d, &coeff);
    }

    image_from_f64(img.pixel_id(), &size, img, &buf)
}

/// The coefficients of the fourth-order recursion for one `sigmad`
/// (index-space sigma) and [`GaussianOrder`]. Field names mirror ITK's
/// members: `n*` numerator (causal), `d*` denominator (shared), `m*` numerator
/// of the anti-causal pass, `bn*`/`bm*` the boundary coefficients.
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
    /// Coefficients of the fourth-order recursion for one `sigmad`
    /// (index-space sigma) and [`GaussianOrder`], reproducing ITK's
    /// `SetUp(order)` in full: `ComputeDCoefficients` (order-independent) →
    /// `ComputeNCoefficients` (one or two Farnebäck columns, depending on the
    /// order) → the order-specific `alpha0`/`alpha1`/`alpha2` normalization →
    /// `ComputeRemainingCoefficients(symmetric)`.
    ///
    /// `sigma` is the *physical* sigma (used only by the
    /// `normalize_across_scale` scale-space factor, `sigma^order`); `sigmad`
    /// is `sigma / spacing`, the index-space sigma the trigonometric and
    /// exponential terms are actually evaluated at.
    fn new(order: GaussianOrder, sigmad: f64, sigma: f64, normalize_across_scale: bool) -> Self {
        // Improved Deriche constants (Farnebäck & Westin 2006, Table 3): the
        // poles (W1,L1,W2,L2) are shared by all three orders; A1,B1,A2,B2
        // hold one column per order ([ZeroOrder, FirstOrder, SecondOrder]),
        // read by ITK's `SetUp` as e.g. `A1[static_cast<int>(m_Order)]`.
        const W1: f64 = 0.6681;
        const L1: f64 = -1.3932;
        const W2: f64 = 2.0787;
        const L2: f64 = -1.3732;
        const A1: [f64; 3] = [1.3530, -0.6724, -1.3563];
        const B1: [f64; 3] = [1.8151, -3.4327, 5.2318];
        const A2: [f64; 3] = [-0.3531, 0.6724, 0.3446];
        const B2: [f64; 3] = [0.0902, 0.6100, -2.2355];

        // ComputeDCoefficients: the shared denominator (recursion)
        // coefficients, plus SD/DD/ED — order-independent, since W1,L1,W2,L2
        // are shared by every column.
        let (d1, d2, d3, d4, sd, dd, ed) = compute_d_coefficients(sigmad, W1, L1, W2, L2);

        // ComputeNCoefficients plus the order-specific normalization (SetUp's
        // switch on m_Order), producing the final N coefficients and the
        // `symmetric` flag ComputeRemainingCoefficients needs.
        let (n0, n1, n2, n3, symmetric) = match order {
            GaussianOrder::ZeroOrder => {
                let (n0, n1, n2, n3, sn, _dn, _en) =
                    compute_n_coefficients(sigmad, A1[0], B1[0], W1, L1, A2[0], B2[0], W2, L2);
                // Unity DC gain; `across_scale_normalization` is always 1 for
                // the zero order — ITK never rescales it by sigma.
                let alpha0 = 2.0 * sn / sd - n0;
                (n0 / alpha0, n1 / alpha0, n2 / alpha0, n3 / alpha0, true)
            }
            GaussianOrder::FirstOrder => {
                let (n0, n1, n2, n3, sn, dn, _en) =
                    compute_n_coefficients(sigmad, A1[1], B1[1], W1, L1, A2[1], B2[1], W2, L2);
                let alpha1 = 2.0 * (sn * dd - dn * sd) / (sd * sd);
                let across_scale = if normalize_across_scale { sigma } else { 1.0 };
                let scale = across_scale / alpha1;
                (n0 * scale, n1 * scale, n2 * scale, n3 * scale, false)
            }
            GaussianOrder::SecondOrder => {
                let (n0_0, n1_0, n2_0, n3_0, sn0, dn0, en0) =
                    compute_n_coefficients(sigmad, A1[0], B1[0], W1, L1, A2[0], B2[0], W2, L2);
                let (n0_2, n1_2, n2_2, n3_2, sn2, dn2, en2) =
                    compute_n_coefficients(sigmad, A1[2], B1[2], W1, L1, A2[2], B2[2], W2, L2);
                // Blend the ZeroOrder and SecondOrder Farnebäck columns so
                // the result has zero net DC gain.
                let beta = -(2.0 * sn2 - sd * n0_2) / (2.0 * sn0 - sd * n0_0);
                let n0 = n0_2 + beta * n0_0;
                let n1 = n1_2 + beta * n1_0;
                let n2 = n2_2 + beta * n2_0;
                let n3 = n3_2 + beta * n3_0;
                let sn = sn2 + beta * sn0;
                let dn = dn2 + beta * dn0;
                let en = en2 + beta * en0;

                let alpha2 = (en * sd * sd - ed * sn * sd - 2.0 * dn * dd * sd
                    + 2.0 * dd * dd * sn)
                    / (sd * sd * sd);
                let across_scale = if normalize_across_scale {
                    sigma * sigma
                } else {
                    1.0
                };
                let scale = across_scale / alpha2;
                (n0 * scale, n1 * scale, n2 * scale, n3 * scale, true)
            }
        };

        // ComputeRemainingCoefficients(symmetric): the anti-causal numerator
        // M and the causal/anti-causal boundary coefficients.
        let (m1, m2, m3, m4) = if symmetric {
            (n1 - d1 * n0, n2 - d2 * n0, n3 - d3 * n0, -d4 * n0)
        } else {
            (-(n1 - d1 * n0), -(n2 - d2 * n0), -(n3 - d3 * n0), d4 * n0)
        };

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

/// Port of `ComputeDCoefficients`: the shared fourth-order recursion
/// denominator coefficients (`D1..D4`), plus `SD` (the sum `1+D1+D2+D3+D4`),
/// `DD` (the same terms weighted `0,1,2,3,4`) and `ED` (weighted
/// `0,1,4,9,16`) — the sums the order-specific `alpha1`/`alpha2`
/// normalization in [`Coefficients::new`] needs. Order-independent: `W1,L1,
/// W2,L2` (the shared poles) are the only inputs besides `sigmad`.
fn compute_d_coefficients(
    sigmad: f64,
    w1: f64,
    l1: f64,
    w2: f64,
    l2: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let cos1 = (w1 / sigmad).cos();
    let cos2 = (w2 / sigmad).cos();
    let exp1 = (l1 / sigmad).exp();
    let exp2 = (l2 / sigmad).exp();

    let d4 = exp1 * exp1 * exp2 * exp2;
    let d3 = -2.0 * cos1 * exp1 * exp2 * exp2 - 2.0 * cos2 * exp2 * exp1 * exp1;
    let d2 = 4.0 * cos2 * cos1 * exp1 * exp2 + exp1 * exp1 + exp2 * exp2;
    let d1 = -2.0 * (exp2 * cos2 + exp1 * cos1);

    let sd = 1.0 + d1 + d2 + d3 + d4;
    let dd = d1 + 2.0 * d2 + 3.0 * d3 + 4.0 * d4;
    let ed = d1 + 4.0 * d2 + 9.0 * d3 + 16.0 * d4;
    (d1, d2, d3, d4, sd, dd, ed)
}

/// Port of `ComputeNCoefficients`: the causal numerator coefficients
/// (`N0..N3`) for one Farnebäck column (`a1,b1,a2,b2` select the
/// ZeroOrder/FirstOrder/SecondOrder column), plus `SN`/`DN`/`EN` — the same
/// `1`/`1,2,3`/`1,4,9`-weighted sums of `N1..N3` (no `N0` term, since it
/// carries no `z^-1` power) used by the order-specific normalization in
/// [`Coefficients::new`].
#[allow(clippy::too_many_arguments)]
fn compute_n_coefficients(
    sigmad: f64,
    a1: f64,
    b1: f64,
    w1: f64,
    l1: f64,
    a2: f64,
    b2: f64,
    w2: f64,
    l2: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let sin1 = (w1 / sigmad).sin();
    let sin2 = (w2 / sigmad).sin();
    let cos1 = (w1 / sigmad).cos();
    let cos2 = (w2 / sigmad).cos();
    let exp1 = (l1 / sigmad).exp();
    let exp2 = (l2 / sigmad).exp();

    let n0 = a1 + a2;
    let n1 =
        exp2 * (b2 * sin2 - (a2 + 2.0 * a1) * cos2) + exp1 * (b1 * sin1 - (a1 + 2.0 * a2) * cos1);
    let n2 = ((a1 + a2) * cos2 * cos1 - b1 * cos2 * sin1 - b2 * cos1 * sin2) * 2.0 * exp1 * exp2
        + a2 * exp1 * exp1
        + a1 * exp2 * exp2;
    let n3 =
        exp2 * exp1 * exp1 * (b2 * sin2 - a2 * cos2) + exp1 * exp2 * exp2 * (b1 * sin1 - a1 * cos1);

    let sn = n0 + n1 + n2 + n3;
    let dn = n1 + 2.0 * n2 + 3.0 * n3;
    let en = n1 + 4.0 * n2 + 9.0 * n3;
    (n0, n1, n2, n3, sn, dn, en)
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
/// `RecursiveSeparableImageFilter::FilterDataArray` operation-for-operation.
/// This recursion is the same for every [`GaussianOrder`] — only the
/// [`Coefficients`] fed in differ. The border value is assumed to extend to
/// infinity on both ends. Requires `data.len() >= 4`; `outs` and `scratch`
/// are caller-provided scratch buffers of the same length as `data` (their
/// prior contents are overwritten).
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

    // ---- derivative orders -------------------------------------------------

    /// 1-D helper: run [`recursive_gaussian_with_order`] with a single order
    /// on a `[n, 1]` image, returning the filtered line.
    fn filter_1d(data: &[f64], sigma: f64, order: GaussianOrder) -> Vec<f64> {
        let img = Image::from_vec(&[data.len(), 1], data.to_vec()).unwrap();
        recursive_gaussian_with_order(
            &img,
            &[sigma, 0.0],
            &[order, GaussianOrder::ZeroOrder],
            false,
        )
        .unwrap()
        .to_f64_vec()
    }

    #[test]
    fn first_order_of_constant_is_near_zero() {
        // The derivative of a constant is 0; the recursion's border extension
        // ("the border value extends to infinity") makes this exact for an
        // interior/infinite constant, so only floating-point roundoff remains.
        let out = filter_1d(&vec![7.0; 32], 3.0, GaussianOrder::FirstOrder);
        for v in out {
            assert!(v.abs() < 1e-9, "first-order of constant not ~0: {v}");
        }
    }

    #[test]
    fn second_order_of_linear_ramp_is_near_zero() {
        // The second derivative of a line is 0. Gaussian smoothing preserves a
        // linear ramp exactly (away from the border), so SecondOrder should
        // read ~0 in the interior. Near the border the "extends to infinity"
        // assumption is violated by a ramp (its true continuation keeps
        // sloping, not staying flat), so the smoothed ramp deviates from the
        // ideal ramp by a boundary-leakage term that decays geometrically
        // (rate `exp(L/sigmad)` per index step, from the recursion's poles);
        // at margin=60 with sigma=3 (sigmad=3) that is `exp(-1.39/3)^60 ~
        // 1e-12`, far below the 1e-8 tolerance used here.
        let n = 200;
        let margin = 60;
        let ramp: Vec<f64> = (0..n).map(|i| 2.5 * i as f64 - 10.0).collect();
        let out = filter_1d(&ramp, 3.0, GaussianOrder::SecondOrder);
        for &v in &out[margin..n - margin] {
            assert!(v.abs() < 1e-8, "second-order of ramp not ~0: {v}");
        }
    }

    #[test]
    fn first_order_matches_analytic_derivative_of_smoothed_gaussian() {
        // f(x) = exp(-(x-c)^2 / (2*s0^2)), a Gaussian bump of width s0.
        // Convolving with a Gaussian of width `sigma` gives (analytically)
        // another Gaussian of width sqrt(s0^2 + sigma^2), scaled by the ratio
        // of the two Gaussians' normalization (s0 / sqrt(s0^2+sigma^2)) since
        // the input here is unnormalized (peak 1, not unit-area). We compare
        // FirstOrder's output against the analytic derivative of that
        // composite Gaussian.
        //
        // The recursive filter only approximates convolution with a true
        // Gaussian (the zero-order impulse response's variance is ~0.938 *
        // sigma^2, see `impulse_response_width_matches_the_itk_recursive_gaussian`),
        // so some relative deviation from the ideal-Gaussian analytic
        // reference is expected even far from any boundary. Empirically (this
        // test, at sigma = 2 and 5.5) the observed peak relative error is
        // ~0.07%-0.19%; 1% of the peak slope keeps a >5x margin over that
        // while catching a mis-derived coefficient (which mismatches by tens
        // of percent, not fractions of one).
        let n = 401;
        let center = 200.0;
        let s0 = 12.0;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let dx = i as f64 - center;
                (-dx * dx / (2.0 * s0 * s0)).exp()
            })
            .collect();

        for &sigma in &[2.0, 5.5] {
            let out = filter_1d(&data, sigma, GaussianOrder::FirstOrder);
            let s_eff2 = s0 * s0 + sigma * sigma;
            let s_eff = s_eff2.sqrt();
            let amp = s0 / s_eff;
            let peak_slope = amp / (s_eff * (-0.5f64).exp().sqrt()); // slope magnitude at the inflection

            // Sample away from the borders (>|8*sigma|) where boundary leakage
            // is negligible relative to the tolerance below.
            for i in (60..n - 60).step_by(5) {
                let x = i as f64 - center;
                let expected = -amp * x / s_eff2 * (-x * x / (2.0 * s_eff2)).exp();
                let got = out[i];
                assert!(
                    (got - expected).abs() < 0.01 * peak_slope,
                    "sigma={sigma} i={i}: got {got}, expected {expected} (peak_slope {peak_slope})"
                );
            }
        }
    }

    #[test]
    fn second_order_matches_analytic_second_derivative_of_smoothed_gaussian() {
        // Same composite-Gaussian setup as
        // `first_order_matches_analytic_derivative_of_smoothed_gaussian`, but
        // comparing SecondOrder against the analytic second derivative
        // d^2/dx^2 [ amp * exp(-x^2 / (2*s_eff^2)) ]
        //   = amp * (x^2 - s_eff^2) / s_eff^4 * exp(-x^2 / (2*s_eff^2))
        //
        // Empirically (this test, at sigma = 2 and 5.5) the observed peak
        // relative error is ~0.20%-0.40% (larger than FirstOrder's, since the
        // second derivative amplifies the same ~0.938-sigma^2 approximation
        // bias further); 1.5% keeps a >3.5x margin over that.
        let n = 401;
        let center = 200.0;
        let s0 = 12.0;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let dx = i as f64 - center;
                (-dx * dx / (2.0 * s0 * s0)).exp()
            })
            .collect();

        for &sigma in &[2.0, 5.5] {
            let out = filter_1d(&data, sigma, GaussianOrder::SecondOrder);
            let s_eff2 = s0 * s0 + sigma * sigma;
            let amp = s0 / s_eff2.sqrt();
            let peak_curvature = amp / (s_eff2 * s_eff2) * s_eff2; // = amp / s_eff2, curvature scale at x=0

            for i in (60..n - 60).step_by(5) {
                let x = i as f64 - center;
                let expected =
                    amp * (x * x - s_eff2) / (s_eff2 * s_eff2) * (-x * x / (2.0 * s_eff2)).exp();
                let got = out[i];
                assert!(
                    (got - expected).abs() < 0.015 * peak_curvature,
                    "sigma={sigma} i={i}: got {got}, expected {expected} (peak_curvature {peak_curvature})"
                );
            }
        }
    }

    #[test]
    fn first_order_of_ramp_matches_constant_slope() {
        // Gaussian-smoothing a linear ramp reproduces it exactly (in the
        // interior), so its first derivative is the ramp's constant slope.
        // Same boundary-leakage argument (and tolerance) as
        // `second_order_of_linear_ramp_is_near_zero`.
        let n = 200;
        let margin = 60;
        let slope = 2.5;
        let ramp: Vec<f64> = (0..n).map(|i| slope * i as f64 - 10.0).collect();
        let out = filter_1d(&ramp, 3.0, GaussianOrder::FirstOrder);
        for &v in &out[margin..n - margin] {
            assert!((v - slope).abs() < 1e-8, "first-order of ramp: {v}");
        }
    }

    #[test]
    fn two_dimensional_derivative_matches_along_each_axis() {
        // A separable 2-D Gaussian bump f(x,y) = exp(-((x-cx)^2+(y-cy)^2)/(2*s0^2)).
        // Differentiating along one axis while smoothing (ZeroOrder) along the
        // other reproduces the 1-D analytic derivative in that axis, since the
        // bump factors as g(x) * g(y).
        let n = 121;
        let c = 60.0;
        let s0 = 10.0;
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f64 - c, y as f64 - c);
                data[y * n + x] = (-(dx * dx + dy * dy) / (2.0 * s0 * s0)).exp();
            }
        }
        let img = Image::from_vec(&[n, n], data).unwrap();
        let sigma = 3.0;
        let s_eff2 = s0 * s0 + sigma * sigma;
        let amp = s0 / s_eff2.sqrt();
        let peak_slope = amp / s_eff2.sqrt() * (-0.5f64).exp().sqrt();

        // d/dx: FirstOrder on axis 0, ZeroOrder on axis 1.
        let dx_out = recursive_gaussian_with_order(
            &img,
            &[sigma, sigma],
            &[GaussianOrder::FirstOrder, GaussianOrder::ZeroOrder],
            false,
        )
        .unwrap()
        .to_f64_vec();
        // Empirically the observed peak relative error here is ~0.38%; 1.5%
        // (same margin as the 1-D SecondOrder case) covers the 2-D case's
        // extra separable-product rounding.
        for yi in (30..n - 30).step_by(10) {
            for xi in (30..n - 30).step_by(10) {
                let (x, y) = (xi as f64 - c, yi as f64 - c);
                let gy = (-y * y / (2.0 * s_eff2)).exp();
                let expected = -amp * x / s_eff2 * (-x * x / (2.0 * s_eff2)).exp() * amp * gy;
                let got = dx_out[yi * n + xi];
                assert!(
                    (got - expected).abs() < 0.015 * peak_slope * amp,
                    "d/dx at ({xi},{yi}): got {got}, expected {expected}"
                );
            }
        }

        // d/dy: ZeroOrder on axis 0, FirstOrder on axis 1.
        let dy_out = recursive_gaussian_with_order(
            &img,
            &[sigma, sigma],
            &[GaussianOrder::ZeroOrder, GaussianOrder::FirstOrder],
            false,
        )
        .unwrap()
        .to_f64_vec();
        for yi in (30..n - 30).step_by(10) {
            for xi in (30..n - 30).step_by(10) {
                let (x, y) = (xi as f64 - c, yi as f64 - c);
                let gx = (-x * x / (2.0 * s_eff2)).exp();
                let expected = -amp * y / s_eff2 * (-y * y / (2.0 * s_eff2)).exp() * amp * gx;
                let got = dy_out[yi * n + xi];
                assert!(
                    (got - expected).abs() < 0.015 * peak_slope * amp,
                    "d/dy at ({xi},{yi}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn wrong_orders_length_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian_with_order(&img, &[1.0, 1.0], &[GaussianOrder::ZeroOrder], false),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }
}
