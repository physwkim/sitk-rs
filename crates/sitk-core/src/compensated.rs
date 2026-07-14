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
//!   Folding it (Neumaier's correction) is *strictly more accurate* — and would make this
//!   port disagree with upstream in the last bit at every site. Accuracy is not the goal
//!   here; being the reduction ITK computes is. [`corrected`](CompensatedSum::corrected)
//!   exposes the folded value for tests that want a better reference to measure against,
//!   and is deliberately not what `sum()` returns.
//! - **Assignment resets the compensation** (`.hxx:120-127`), so `s = x` is a fresh
//!   accumulator seeded at `x`, not `x` added to the running one. `GaussianDerivative-
//!   Operator` depends on this: it assigns a *naive* `std::accumulate` into a
//!   `CompensatedSummation` and keeps summing, so the compensation is discarded there.
//! - **`+=` of another accumulator adds the other's compensation first, then its sum**
//!   (`.hxx:76-83`) — not the same as adding `other.sum()`.
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
/// use sitk_core::compensated::CompensatedSum;
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

    /// Fold another accumulator in, ITK's `operator+=(const Self &)` (`.hxx:76-83`): the
    /// other's **compensation first**, then its sum. Not equivalent to `add(other.sum())`.
    pub fn add_sum(&mut self, other: &Self) {
        self.add(other.compensation);
        self.add(other.sum);
    }

    /// Scale both the sum and the carried compensation — ITK's `operator*=`
    /// (`.hxx:94-101`).
    pub fn scale(&mut self, factor: f64) {
        self.sum *= factor;
        self.compensation *= factor;
    }

    /// The running sum, **without** the residual folded in — ITK's `GetSum()`
    /// (`.hxx:130-135`). This is the value upstream computes, and so the value this port
    /// computes. For a strictly more accurate figure, see [`corrected`](Self::corrected),
    /// which upstream never returns.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// The sum with the residual folded back in (Neumaier). **More accurate than ITK, and
    /// therefore not what this port uses in a metric** — it exists so a test can measure
    /// [`sum`](Self::sum) against a better reference than the naive walk.
    pub fn corrected(&self) -> f64 {
        self.sum - self.compensation
    }
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
            naive, 1.0e17 + 1000.0,
            "the fixture must defeat naive summation, or the pin below is vacuous"
        );

        assert_eq!(compensated_sum(terms.iter().copied()), 1.0e17 + 1000.0);
    }

    /// **`sum()` must NOT fold the residual in — it is ITK's `GetSum()`, and ITK's is
    /// lossy.** The two differ exactly when a residual is outstanding, and the port owes
    /// upstream's number, not the better one. If this ever starts holding as an equality,
    /// someone has "improved" the accumulator and silently diverged every call site from
    /// ITK in the last bit.
    #[test]
    fn the_sum_is_itks_lossy_getsum_and_not_the_corrected_one() {
        let mut s = CompensatedSum::new();
        s += 1.0;
        s += 1.0e-20;
        s += 1.0e-20;

        assert_eq!(s.sum(), 1.0, "GetSum() returns m_Sum, residual unfolded");
        assert_ne!(
            s.corrected().to_bits(),
            s.sum().to_bits(),
            "the residual must be real here, or this test cannot detect a fold"
        );
        assert!(s.corrected() > 1.0);
    }

    /// Assignment resets the compensation (ITK `.hxx:120-127`) — `GaussianDerivative-
    /// Operator` relies on it when it assigns a naive `std::accumulate` into the
    /// accumulator and keeps going.
    #[test]
    fn seeding_drops_the_carried_compensation() {
        let mut s = CompensatedSum::new();
        s += 1.0;
        s += 1.0e-20;
        assert!(s.corrected() > 1.0);

        let reseeded = CompensatedSum::seeded(s.sum());
        assert_eq!(reseeded.sum(), 1.0);
        assert_eq!(
            reseeded.corrected(),
            1.0,
            "a reseeded accumulator carries no residual"
        );
    }

    /// Folding one accumulator into another adds the other's **compensation first**, then
    /// its sum (ITK `.hxx:76-83`) — which is not `add(other.sum())`.
    #[test]
    fn folding_an_accumulator_carries_its_compensation() {
        let mut part = CompensatedSum::new();
        part += 1.0;
        part += 1.0e-20;
        part += 1.0e-20;

        let mut folded = CompensatedSum::seeded(1.0e17);
        folded.add_sum(&part);

        let mut dropped = CompensatedSum::seeded(1.0e17);
        dropped += part.sum();

        // Both land on 1e17 in the sum; the difference is the residual each carries, which
        // is what the next addition will see.
        assert_ne!(
            folded.corrected().to_bits(),
            dropped.corrected().to_bits(),
            "add_sum must carry the other's compensation, or it is just add(other.sum())"
        );
    }

    /// Scaling scales the residual too (ITK `operator*=`), so a later `sum()` is still the
    /// compensated one.
    #[test]
    fn scaling_scales_the_compensation() {
        let mut s = CompensatedSum::new();
        s += 1.0;
        s += 1.0e-20;
        let before = s.corrected() - s.sum();

        s.scale(2.0);
        let after = s.corrected() - s.sum();

        assert_eq!(after, before * 2.0);
    }
}
