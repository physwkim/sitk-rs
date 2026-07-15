//! The coefficients of the fourth-order Deriche/Farnebäck recursive Gaussian —
//! the *one* copy of that math in this workspace.
//!
//! This is the coefficient half of `itk::RecursiveGaussianImageFilter`'s `SetUp`
//! (`ComputeDCoefficients` → `ComputeNCoefficients` → the order-specific
//! `alpha0`/`alpha1`/`alpha2` normalization → `ComputeRemainingCoefficients`),
//! separated from the recursion that consumes them because it has **two**
//! consumers that cannot see each other:
//!
//! - `sitk-filters`' host filter, which runs the forward+backward recursion on
//!   the CPU, and
//! - `sitk-cuda`'s device pyramid, which uploads the coefficients to a kernel
//!   that runs the same recursion on the GPU.
//!
//! `sitk-filters` depends on `sitk-cuda`, so the device path cannot reach into
//! the host filter, and that edge cannot be reversed. Both crates depend on
//! `sitk-core` (`sitk-cuda` optionally, under its `cuda` feature), so this is the
//! one place both can reach. Before this module existed the device carried a
//! second, hand-copied `ZeroOrder` transcription of the same twenty numbers,
//! trusted only because a test pinned it; there is now nothing to drift.
//!
//! The recursion itself is *not* here: it is the same for every [`GaussianOrder`]
//! and each consumer expresses it in its own language (Rust over a line buffer,
//! CUDA C over a strided line). Only the coefficients differ by order, and only
//! the coefficients were duplicated.
//!
//! The improved Deriche constants are from Farnebäck & Westin, *"Improving
//! Deriche-style Recursive Gaussian Filters"* (J. Math. Imaging Vis. 2006,
//! Table 3): three columns of `A1,B1,A2,B2` — one per [`GaussianOrder`] — sharing
//! the same poles `W1,L1,W2,L2`.

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

/// The number of coefficients the fourth-order recursion needs, and the length of
/// [`Coefficients::to_array`].
pub const COEFFICIENT_COUNT: usize = 20;

/// The coefficients of the fourth-order recursion for one `sigmad` (index-space
/// sigma) and [`GaussianOrder`]. Field names mirror ITK's members: `n*` numerator
/// (causal), `d*` denominator (shared by both passes), `m*` numerator of the
/// anti-causal pass, `bn*`/`bm*` the boundary coefficients of each.
///
/// The fields are public because the recursion that consumes them lives in
/// another crate — two of them, in two languages. [`to_array`](Self::to_array)
/// gives the same twenty numbers as a flat array for a consumer that must ship
/// them across a boundary that cannot carry a Rust struct (a device upload).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coefficients {
    /// Causal numerator, `z^0`.
    pub n0: f64,
    /// Causal numerator, `z^-1`.
    pub n1: f64,
    /// Causal numerator, `z^-2`.
    pub n2: f64,
    /// Causal numerator, `z^-3`.
    pub n3: f64,
    /// Denominator (recursion), `z^-1`. Shared by both passes.
    pub d1: f64,
    /// Denominator (recursion), `z^-2`. Shared by both passes.
    pub d2: f64,
    /// Denominator (recursion), `z^-3`. Shared by both passes.
    pub d3: f64,
    /// Denominator (recursion), `z^-4`. Shared by both passes.
    pub d4: f64,
    /// Anti-causal numerator, `z^+1`.
    pub m1: f64,
    /// Anti-causal numerator, `z^+2`.
    pub m2: f64,
    /// Anti-causal numerator, `z^+3`.
    pub m3: f64,
    /// Anti-causal numerator, `z^+4`.
    pub m4: f64,
    /// Causal boundary coefficient, `z^-1`.
    pub bn1: f64,
    /// Causal boundary coefficient, `z^-2`.
    pub bn2: f64,
    /// Causal boundary coefficient, `z^-3`.
    pub bn3: f64,
    /// Causal boundary coefficient, `z^-4`.
    pub bn4: f64,
    /// Anti-causal boundary coefficient, `z^+1`.
    pub bm1: f64,
    /// Anti-causal boundary coefficient, `z^+2`.
    pub bm2: f64,
    /// Anti-causal boundary coefficient, `z^+3`.
    pub bm3: f64,
    /// Anti-causal boundary coefficient, `z^+4`.
    pub bm4: f64,
}

/// The two conjugate pole pairs of the recursion, shared by all three
/// [`GaussianOrder`] columns (Farnebäck & Westin 2006, Table 3).
#[derive(Clone, Copy, Debug)]
struct Poles {
    w1: f64,
    l1: f64,
    w2: f64,
    l2: f64,
}

const POLES: Poles = Poles {
    w1: 0.6681,
    l1: -1.3932,
    w2: 2.0787,
    l2: -1.3732,
};

/// One Farnebäck column — the numerator constants of a single [`GaussianOrder`].
/// ITK reads these as `A1[static_cast<int>(m_Order)]` and friends; here each
/// order names its own column.
#[derive(Clone, Copy, Debug)]
struct Column {
    a1: f64,
    b1: f64,
    a2: f64,
    b2: f64,
}

impl Column {
    /// The Farnebäck column for `order` — Table 3 read down a single row of
    /// `A1,B1,A2,B2` instead of across ITK's three-element arrays.
    const fn of(order: GaussianOrder) -> Self {
        match order {
            GaussianOrder::ZeroOrder => Column {
                a1: 1.3530,
                b1: 1.8151,
                a2: -0.3531,
                b2: 0.0902,
            },
            GaussianOrder::FirstOrder => Column {
                a1: -0.6724,
                b1: -3.4327,
                a2: 0.6724,
                b2: 0.6100,
            },
            GaussianOrder::SecondOrder => Column {
                a1: -1.3563,
                b1: 5.2318,
                a2: 0.3446,
                b2: -2.2355,
            },
        }
    }
}

impl Coefficients {
    /// Coefficients of the fourth-order recursion for one `sigmad` (index-space
    /// sigma) and [`GaussianOrder`], reproducing ITK's `SetUp(order)` in full:
    /// `ComputeDCoefficients` (order-independent) → `ComputeNCoefficients` (one
    /// or two Farnebäck columns, depending on the order) → the order-specific
    /// `alpha0`/`alpha1`/`alpha2` normalization →
    /// `ComputeRemainingCoefficients(symmetric)`.
    ///
    /// - **Normalization.** `ZeroOrder` scales its numerator for unity DC gain
    ///   (`alpha0`); `FirstOrder`/`SecondOrder` instead scale so the filter's
    ///   gain matches the true derivative operator (`alpha1`/`alpha2`, derived
    ///   from `SD`/`DD`/`ED` — the recursion denominator and its 1st/2nd
    ///   "z-derivative" weighted sums). `SecondOrder` additionally blends the
    ///   `ZeroOrder` and `SecondOrder` Farnebäck columns through a `beta`
    ///   coefficient so the result has zero net DC gain.
    /// - **Symmetry.** `ZeroOrder`/`SecondOrder` use a *symmetric* anti-causal
    ///   numerator (`Mi = Ni - Di·N0`, matching a filter whose true impulse
    ///   response is even); `FirstOrder` is *anti-symmetric*
    ///   (`Mi = -(Ni - Di·N0)`, with `M4`'s sign flipped too), matching an odd
    ///   impulse response. This is ITK's
    ///   `ComputeRemainingCoefficients(symmetric)` flag.
    ///
    /// `sigmad` is `sigma / spacing`, the index-space sigma the trigonometric and
    /// exponential terms are actually evaluated at, and must be `> 0`. `sigma` is
    /// the *physical* sigma, used only by the `normalize_across_scale`
    /// scale-space factor (`sigma^order`, ITK's Lindeberg normalization, so a
    /// feature's peak derivative response does not depend on the scale it is
    /// measured at). `ZeroOrder` ignores both `sigma` and
    /// `normalize_across_scale`: ITK never rescales the zero order, so a
    /// smoothing-only caller may pass anything for them.
    pub fn new(
        order: GaussianOrder,
        sigmad: f64,
        sigma: f64,
        normalize_across_scale: bool,
    ) -> Self {
        // ComputeDCoefficients: the shared denominator (recursion) coefficients,
        // plus SD/DD/ED — order-independent, since the poles are shared by every
        // column.
        let (d1, d2, d3, d4, sd, dd, ed) = compute_d_coefficients(sigmad, POLES);

        // ComputeNCoefficients plus the order-specific normalization (SetUp's
        // switch on m_Order), producing the final N coefficients and the
        // `symmetric` flag ComputeRemainingCoefficients needs.
        let (n0, n1, n2, n3, symmetric) = match order {
            GaussianOrder::ZeroOrder => {
                let (n0, n1, n2, n3, sn, _dn, _en) =
                    compute_n_coefficients(sigmad, Column::of(GaussianOrder::ZeroOrder), POLES);
                // Unity DC gain; `across_scale_normalization` is always 1 for the
                // zero order — ITK never rescales it by sigma.
                let alpha0 = 2.0 * sn / sd - n0;
                (n0 / alpha0, n1 / alpha0, n2 / alpha0, n3 / alpha0, true)
            }
            GaussianOrder::FirstOrder => {
                let (n0, n1, n2, n3, sn, dn, _en) =
                    compute_n_coefficients(sigmad, Column::of(GaussianOrder::FirstOrder), POLES);
                let alpha1 = 2.0 * (sn * dd - dn * sd) / (sd * sd);
                let across_scale = if normalize_across_scale { sigma } else { 1.0 };
                let scale = across_scale / alpha1;
                (n0 * scale, n1 * scale, n2 * scale, n3 * scale, false)
            }
            GaussianOrder::SecondOrder => {
                let (n0_0, n1_0, n2_0, n3_0, sn0, dn0, en0) =
                    compute_n_coefficients(sigmad, Column::of(GaussianOrder::ZeroOrder), POLES);
                let (n0_2, n1_2, n2_2, n3_2, sn2, dn2, en2) =
                    compute_n_coefficients(sigmad, Column::of(GaussianOrder::SecondOrder), POLES);
                // Blend the ZeroOrder and SecondOrder Farnebäck columns so the
                // result has zero net DC gain.
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

        // ComputeRemainingCoefficients(symmetric): the anti-causal numerator M and
        // the causal/anti-causal boundary coefficients.
        let (m1, m2, m3, m4) = if symmetric {
            (n1 - d1 * n0, n2 - d2 * n0, n3 - d3 * n0, -d4 * n0)
        } else {
            (-(n1 - d1 * n0), -(n2 - d2 * n0), -(n3 - d3 * n0), d4 * n0)
        };

        let sn = n0 + n1 + n2 + n3;
        let sm = m1 + m2 + m3 + m4;

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
            bn1: d1 * sn / sd,
            bn2: d2 * sn / sd,
            bn3: d3 * sn / sd,
            bn4: d4 * sn / sd,
            bm1: d1 * sm / sd,
            bm2: d2 * sm / sd,
            bm3: d3 * sm / sd,
            bm4: d4 * sm / sd,
        }
    }

    /// The same twenty coefficients as a flat array, in **declaration order**:
    /// `[n0..n3, d1..d4, m1..m4, bn1..bn4, bm1..bm4]`.
    ///
    /// For a consumer that must ship them across a boundary a Rust struct cannot
    /// cross — `sitk-cuda` uploads exactly this array to a device kernel that
    /// names the slots `N0 = c[0] … BM4 = c[19]`. A struct field is reachable by
    /// name from Rust and by nothing at all from CUDA C; this array is the seam,
    /// and its order is part of the contract.
    pub fn to_array(&self) -> [f64; COEFFICIENT_COUNT] {
        [
            self.n0, self.n1, self.n2, self.n3, self.d1, self.d2, self.d3, self.d4, self.m1,
            self.m2, self.m3, self.m4, self.bn1, self.bn2, self.bn3, self.bn4, self.bm1, self.bm2,
            self.bm3, self.bm4,
        ]
    }
}

/// Port of `ComputeDCoefficients`: the shared fourth-order recursion denominator
/// coefficients (`D1..D4`), plus `SD` (the sum `1+D1+D2+D3+D4`), `DD` (the same
/// terms weighted `0,1,2,3,4`) and `ED` (weighted `0,1,4,9,16`) — the sums the
/// order-specific `alpha1`/`alpha2` normalization in [`Coefficients::new`] needs.
/// Order-independent: the poles are the only input besides `sigmad`.
fn compute_d_coefficients(sigmad: f64, p: Poles) -> (f64, f64, f64, f64, f64, f64, f64) {
    let cos1 = (p.w1 / sigmad).cos();
    let cos2 = (p.w2 / sigmad).cos();
    let exp1 = (p.l1 / sigmad).exp();
    let exp2 = (p.l2 / sigmad).exp();

    let d4 = exp1 * exp1 * exp2 * exp2;
    let d3 = -2.0 * cos1 * exp1 * exp2 * exp2 - 2.0 * cos2 * exp2 * exp1 * exp1;
    let d2 = 4.0 * cos2 * cos1 * exp1 * exp2 + exp1 * exp1 + exp2 * exp2;
    let d1 = -2.0 * (exp2 * cos2 + exp1 * cos1);

    let sd = 1.0 + d1 + d2 + d3 + d4;
    let dd = d1 + 2.0 * d2 + 3.0 * d3 + 4.0 * d4;
    let ed = d1 + 4.0 * d2 + 9.0 * d3 + 16.0 * d4;
    (d1, d2, d3, d4, sd, dd, ed)
}

/// Port of `ComputeNCoefficients`: the causal numerator coefficients (`N0..N3`)
/// for one Farnebäck [`Column`], plus `SN`/`DN`/`EN` — the same `1`/`1,2,3`/
/// `1,4,9`-weighted sums of `N1..N3` (no `N0` term, since it carries no `z^-1`
/// power) used by the order-specific normalization in [`Coefficients::new`].
fn compute_n_coefficients(sigmad: f64, c: Column, p: Poles) -> (f64, f64, f64, f64, f64, f64, f64) {
    let sin1 = (p.w1 / sigmad).sin();
    let sin2 = (p.w2 / sigmad).sin();
    let cos1 = (p.w1 / sigmad).cos();
    let cos2 = (p.w2 / sigmad).cos();
    let exp1 = (p.l1 / sigmad).exp();
    let exp2 = (p.l2 / sigmad).exp();

    let n0 = c.a1 + c.a2;
    let n1 = exp2 * (c.b2 * sin2 - (c.a2 + 2.0 * c.a1) * cos2)
        + exp1 * (c.b1 * sin1 - (c.a1 + 2.0 * c.a2) * cos1);
    let n2 =
        ((c.a1 + c.a2) * cos2 * cos1 - c.b1 * cos2 * sin1 - c.b2 * cos1 * sin2) * 2.0 * exp1 * exp2
            + c.a2 * exp1 * exp1
            + c.a1 * exp2 * exp2;
    let n3 = exp2 * exp1 * exp1 * (c.b2 * sin2 - c.a2 * cos2)
        + exp1 * exp2 * exp2 * (c.b1 * sin1 - c.a1 * cos1);

    let sn = n0 + n1 + n2 + n3;
    let dn = n1 + 2.0 * n2 + 3.0 * n3;
    let en = n1 + 4.0 * n2 + 9.0 * n3;
    (n0, n1, n2, n3, sn, dn, en)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DC gain of the causal+anti-causal pair: a constant input `k` comes out
    /// as `k * (SN + SM) / SD`. This is the property `alpha0` normalizes for, and
    /// it is what makes the smoothing filter preserve a constant image exactly.
    fn dc_gain(c: &Coefficients) -> f64 {
        let sn = c.n0 + c.n1 + c.n2 + c.n3;
        let sm = c.m1 + c.m2 + c.m3 + c.m4;
        let sd = 1.0 + c.d1 + c.d2 + c.d3 + c.d4;
        (sn + sm) / sd
    }

    #[test]
    fn zero_order_has_unity_dc_gain() {
        for &sigmad in &[0.5, 1.0, 2.0, 4.0, 16.0] {
            let c = Coefficients::new(GaussianOrder::ZeroOrder, sigmad, sigmad, false);
            let g = dc_gain(&c);
            assert!(
                (g - 1.0).abs() < 1e-12,
                "sigmad {sigmad}: zero-order DC gain {g}, want 1"
            );
        }
    }

    #[test]
    fn derivative_orders_have_zero_dc_gain() {
        // A derivative of a constant is zero, so both derivative orders must have
        // no DC response — for SecondOrder this is exactly what the `beta` blend
        // of the two Farnebäck columns is for.
        for order in [GaussianOrder::FirstOrder, GaussianOrder::SecondOrder] {
            for &sigmad in &[0.5, 1.0, 2.0, 4.0, 16.0] {
                let c = Coefficients::new(order, sigmad, sigmad, false);
                let g = dc_gain(&c);
                assert!(
                    g.abs() < 1e-10,
                    "{order:?} at sigmad {sigmad}: DC gain {g}, want 0"
                );
            }
        }
    }

    #[test]
    fn normalize_across_scale_scales_by_sigma_to_the_order() {
        // Lindeberg normalization multiplies the numerator (and therefore the
        // whole response) by sigma^order; the denominator is untouched.
        let sigma = 3.0;
        for (order, power) in [
            (GaussianOrder::FirstOrder, 1i32),
            (GaussianOrder::SecondOrder, 2),
        ] {
            let off = Coefficients::new(order, sigma, sigma, false);
            let on = Coefficients::new(order, sigma, sigma, true);
            let factor = sigma.powi(power);
            assert!((on.n0 - off.n0 * factor).abs() < 1e-9 * factor.abs());
            assert_eq!(on.d1, off.d1, "{order:?}: denominator must not move");
        }
    }

    #[test]
    fn zero_order_ignores_normalize_across_scale() {
        // ITK never rescales the zero order; the flag must be inert for it.
        let off = Coefficients::new(GaussianOrder::ZeroOrder, 2.0, 2.0, false);
        let on = Coefficients::new(GaussianOrder::ZeroOrder, 2.0, 2.0, true);
        assert_eq!(off, on);
    }

    #[test]
    fn to_array_is_in_the_documented_device_order() {
        // `sitk-cuda`'s kernel names these slots N0 = c[0] .. BM4 = c[19]; the
        // order is a contract with a consumer that cannot see the struct.
        let c = Coefficients::new(GaussianOrder::ZeroOrder, 2.5, 2.5, false);
        let a = c.to_array();
        assert_eq!(
            a,
            [
                c.n0, c.n1, c.n2, c.n3, c.d1, c.d2, c.d3, c.d4, c.m1, c.m2, c.m3, c.m4, c.bn1,
                c.bn2, c.bn3, c.bn4, c.bm1, c.bm2, c.bm3, c.bm4,
            ]
        );
    }
}
