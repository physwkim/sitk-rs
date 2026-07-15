//! [`CompensatedSum`] — Kahan summation with ITK's exact contract.
//!
//! The port of `itk::CompensatedSummation` (`itkCompensatedSummation.h`,
//! `itkCompensatedSummation.hxx`). It exists because ITK reaches for that class in a
//! *specific* set of reductions and not in the others, and where it reaches, this port
//! must too: a naive `f64` accumulation of `n` terms carries up to `n·ε` relative error,
//! which is nothing in a display value and is everything in a normalizer, a histogram's
//! total mass, or a number a branch is about to be taken on.
//!
//! This type is the **single owner** of that recurrence. It was written after the same
//! defect was found twice in two weeks — once in Mattes' joint-PDF sum (§2.161), then in
//! the joint-histogram metric's marginals while reading the adjacent file — and the second
//! find is what says the population is larger than the citations. Five hand-rolled Kahan
//! loops in five files would be five copies of one rule, each free to drift; one type is
//! not.
//!
//! # The contract is ITK's, bit for bit — including where it is lossy
//!
//! Mirroring the recurrence is not enough; the *observable* behaviour has to match, and
//! ITK's has three edges that a from-scratch Kahan type would not have:
//!
//! - **[`sum`](CompensatedSum::sum) does not fold the residual back in.** ITK's `GetSum()`
//!   returns `m_Sum` and ignores `m_Compensation` (`itkCompensatedSummation.hxx:132-135`).
//!   Reproduced as-is. (Folding Kahan's residual back would in any case buy almost nothing
//!   — the recurrence holds it at or below half an ulp of the sum by construction. A
//!   genuinely better answer needs a *different* recurrence: see [`neumaier_sum`], which is
//!   the reference the family's pins measure against and which no port path may call.)
//! - **Kahan's blind spot is reproduced, not fixed.** When a term arrives that is larger
//!   than the running sum, the classic recurrence discards the accumulator's low bits
//!   rather than the term's: `[1.0, 1e100, 1.0, −1e100]` sums to **0.0** under ITK's
//!   algorithm and this one, and to **2.0** under Neumaier's. That is upstream's number, so
//!   it is this port's number.
//! - **Assignment resets the compensation** (`.hxx:120-127`), so `s = x` is a fresh
//!   accumulator seeded at `x`, not `x` added to the running one. `GaussianDerivative-
//!   Operator` depends on this: it assigns a *naive* `std::accumulate` into a
//!   `CompensatedSummation` and keeps summing, so the compensation is discarded there.
//!
//! ITK also offers `operator+=(const Self &)` (merge one accumulator into another) and
//! `operator*=` / `operator/=` (scale sum and compensation together). **Neither is ported,
//! and the first is why.** `operator+=(const Self &)` adds the other's `m_Compensation`
//! (`.hxx:79-83`) — but the recurrence defines the compensation as the *negation* of the
//! part that was lost (`total ≈ m_Sum − m_Compensation`), so a correct merge must
//! **subtract** it. Upstream adds it, i.e. it moves the merged total the wrong way by twice
//! the residual, which leaves the merge *worse than not compensating at all*.
//!
//! This is a **live** ITK defect, not a latent one: three filters merge their per-thread
//! accumulators through it — `StatisticsImageFilter` (`.hxx:130-131`, and its Sum, Mean,
//! Variance and Sigma outputs), `DirectedHausdorffDistanceImageFilter` (`.hxx:188`, the
//! average distance) and `PointSetToPointSetMetricWithIndexv4` (`.hxx:198`, `:373`, the
//! metric value). The error is coherent across work units, so it **grows with the thread
//! count** — the exact property `CompensatedSummation` was introduced to remove. Written up
//! as ledger §1.75; the port-side reasoning is §2.171.
//!
//! This port has no reduction that merges two accumulators, so reproducing the sign error
//! as public API would be shipping a trap that buys parity with nothing, while porting it
//! *corrected* would be a silent divergence from upstream's written code. Ported: neither.
//!
//! # What compensating a reduction buys — and it is not always the same good
//!
//! Two different goods, kept apart because the ledger keeps them apart:
//!
//! - **Parity**, when the port's summation *order* is already ITK's (a serial walk over
//!   the same terms). Compensating then makes the result bit-identical to upstream's, not
//!   merely closer to the true sum.
//! - **Accuracy only**, when the port sums the same terms in a different order than ITK
//!   does (ITK partitions across threads and folds per-thread partials; this port folds in
//!   a fixed serial order for thread-count invariance) or computes a deliberately
//!   different quantity. Bit-parity is unavailable there by construction, and compensation
//!   buys a smaller error against the true sum — a real good, but a different one.
//!
//! A call site should say which one it is claiming. Claiming parity where only accuracy
//! was bought is the quiet overclaim this doc exists to prevent.

/// A Kahan-compensated `f64` accumulator with `itk::CompensatedSummation`'s exact
/// semantics. See the [module docs](self) for why the lossy edges are reproduced rather
/// than improved on.
///
/// ```
/// use sitk::core::compensated::CompensatedSum;
///
/// // The classic defeat of naive summation: a large term, then many tiny ones.
/// let mut s = CompensatedSum::new();
/// s += 1.0e16;
/// for _ in 0..100 {
///     s += 1.0;
/// }
/// assert_eq!(s.sum(), 1.0e16 + 100.0);
///
/// let naive: f64 = std::iter::once(1.0e16).chain(std::iter::repeat_n(1.0, 100)).sum();
/// assert_ne!(naive, 1.0e16 + 100.0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompensatedSum {
    sum: f64,
    compensation: f64,
}

impl CompensatedSum {
    /// A zeroed accumulator (ITK's `ResetToZero`, `.hxx:113-118`).
    pub fn new() -> Self {
        Self::default()
    }

    /// An accumulator seeded at `value`, with **no** carried compensation — ITK's
    /// `operator=(const FloatType &)` (`.hxx:120-127`).
    pub fn seeded(value: f64) -> Self {
        Self {
            sum: value,
            compensation: 0.0,
        }
    }

    /// ITK's `AddElement` (`.hxx:60-65`), the recurrence of
    /// `CompensatedSummationAddElement` (`itkCompensatedSummation.h:40-48`).
    pub fn add(&mut self, element: f64) {
        let compensated_input = element - self.compensation;
        let temp_sum = self.sum + compensated_input;
        self.compensation = (temp_sum - self.sum) - compensated_input;
        self.sum = temp_sum;
    }

    /// The running sum, **without** the residual folded in — ITK's `GetSum()`
    /// (`.hxx:130-135`). This is the value upstream computes, and so the value this port
    /// computes. For a strictly more accurate figure to measure *against*, see
    /// [`neumaier_sum`], which upstream never computes and no port path may call.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// The low-order part this accumulator is carrying and has not yet folded into
    /// [`sum`](Self::sum) — i.e. `−m_Compensation`. Exposed so a test can *see* the state
    /// the recurrence maintains (whether a reseed cleared it, whether a fold carried it).
    ///
    /// **This is not "the correction that would make the sum exact".** Folding it back
    /// (`sum + residual`) is very nearly always a no-op in `f64`, because the Kahan
    /// recurrence keeps `|residual|` at or below half an ulp of `sum` *by construction* —
    /// that is what makes it the residual. An earlier draft of this type shipped a
    /// `corrected()` that did exactly that fold and advertised it as Neumaier's; the gate
    /// caught it, and the three tests it broke are still below. See [`neumaier_sum`] for
    /// what an actually-better reference requires.
    pub fn residual(&self) -> f64 {
        -self.compensation
    }
}

/// **Neumaier's compensated summation — strictly more accurate than ITK's, and therefore
/// forbidden in any port path that owes upstream a number.** It exists to be the *judge* in
/// the family's pins: a test that only compares the compensated walk against the naive one
/// cannot tell which is closer to the truth, so it needs a third, better answer.
///
/// Kahan (what [`CompensatedSum`] and ITK implement) has a known blind spot: it assumes
/// each new term is *smaller* than the running sum. When a term arrives that is **larger**,
/// the low-order bits lost are the ones from the accumulator, not from the term, and the
/// classic recurrence discards them — `[1.0, 1e100, 1.0, −1e100]` sums to **0.0** under
/// Kahan and to **2.0** here. Neumaier's variant branches on the magnitude comparison and
/// keeps the right half in both cases, then folds the residual once at the end (which is
/// sound *here*, unlike in Kahan, precisely because this residual is not bounded by half an
/// ulp).
///
/// Reproducing ITK's blind spot is deliberate: the port owes upstream's reduction, not the
/// best available one. See the [module docs](self).
pub fn neumaier_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut sum = 0.0f64;
    let mut correction = 0.0f64;
    for v in values {
        let t = sum + v;
        correction += if sum.abs() >= v.abs() {
            (sum - t) + v
        } else {
            (v - t) + sum
        };
        sum = t;
    }
    sum + correction
}

impl std::ops::AddAssign<f64> for CompensatedSum {
    fn add_assign(&mut self, rhs: f64) {
        self.add(rhs);
    }
}

impl std::ops::SubAssign<f64> for CompensatedSum {
    /// ITK's `operator-=`, which is `AddElement(-rhs)` (`.hxx:86-92`).
    fn sub_assign(&mut self, rhs: f64) {
        self.add(-rhs);
    }
}

impl FromIterator<f64> for CompensatedSum {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut acc = Self::new();
        for v in iter {
            acc.add(v);
        }
        acc
    }
}

/// The compensated sum of `values`, in iteration order — the common case, where the whole
/// reduction is one walk over one slice.
pub fn compensated_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().collect::<CompensatedSum>().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The pin that would fail if `add` were a naive `+=`.** The fixture is chosen so
    /// that naive summation *provably* loses — asserted first, because a compensated-sum
    /// test whose fixture a naive walk also gets right proves nothing at all.
    #[test]
    fn the_recurrence_recovers_what_a_naive_walk_drops() {
        let mut terms = vec![1.0e17];
        terms.extend(std::iter::repeat_n(1.0, 1000));

        let naive: f64 = terms.iter().sum();
        assert_ne!(
            naive,
            1.0e17 + 1000.0,
            "the fixture must defeat naive summation, or the pin below is vacuous"
        );

        assert_eq!(compensated_sum(terms.iter().copied()), 1.0e17 + 1000.0);
    }

    /// **This port reproduces Kahan's blind spot, because ITK has it.** When a term larger
    /// than the running sum arrives, the classic recurrence throws away the accumulator's
    /// low bits instead of the term's: the canonical `[1, 1e100, 1, −1e100]` comes back as
    /// **0.0**, losing both `1`s. Neumaier's variant gets **2.0**.
    ///
    /// The port owes upstream's number, not the best available one, so `sum()` must stay
    /// 0.0 here. If this test ever starts returning 2.0, someone has "improved" the
    /// accumulator into Neumaier and silently diverged every call site in this family from
    /// ITK — which would be the same defect as the naive walk, only in the other direction.
    #[test]
    fn the_accumulator_keeps_itks_blind_spot_and_does_not_quietly_become_neumaier() {
        let terms = [1.0, 1.0e100, 1.0, -1.0e100];

        assert_eq!(
            compensated_sum(terms),
            0.0,
            "ITK's Kahan loses both 1s here, and this port must lose them too"
        );
        assert_eq!(
            neumaier_sum(terms),
            2.0,
            "the reference must be strictly better, or it cannot judge the pins"
        );
    }

    /// `sum()` is ITK's `GetSum()`: `m_Sum`, with the carried residual left out
    /// (`.hxx:130-135`). The residual is real — the accumulator is holding it — and it is
    /// still not in the reported number.
    #[test]
    fn the_sum_is_itks_lossy_getsum_and_leaves_the_residual_behind() {
        // One add of 1.0 into 1e16 (whose ulp is 2) loses the whole term: the sum does not
        // move, and the accumulator carries the lost 1.0 as its residual.
        let mut s = CompensatedSum::seeded(1.0e16);
        s += 1.0;

        assert_eq!(s.sum(), 1.0e16, "GetSum() returns m_Sum, residual unfolded");
        assert_eq!(s.residual(), 1.0, "the residual must be real and carried");
    }

    /// Assignment resets the compensation (ITK `.hxx:120-127`) — `GaussianDerivative-
    /// Operator` relies on it when it assigns a naive `std::accumulate` into the
    /// accumulator and keeps summing.
    #[test]
    fn seeding_drops_the_carried_compensation() {
        let mut s = CompensatedSum::seeded(1.0e16);
        s += 1.0;
        assert_eq!(s.residual(), 1.0);

        let reseeded = CompensatedSum::seeded(s.sum());
        assert_eq!(reseeded.sum(), 1.0e16);
        assert_eq!(
            reseeded.residual(),
            0.0,
            "a reseeded accumulator carries no residual"
        );
    }
}
