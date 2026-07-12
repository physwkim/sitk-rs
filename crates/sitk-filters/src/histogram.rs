//! Shared 1-D histogram scaffolding for every histogram-driven threshold
//! calculator in this crate: [`crate::intensity`]'s Otsu/Triangle family and
//! [`crate::threshold`]'s Huang/Intermodes/IsoData/KittlerIllingworth/Li/
//! MaximumEntropy/Moments/RenyiEntropy/Shanbhag/Yen family. All of them are
//! built on `itk::Statistics::Histogram`
//! (`Modules/Numerics/Statistics/include/itkHistogram.h(.hxx)`) as populated
//! by `itk::Statistics::ImageToHistogramFilter`/`SampleToHistogramFilter`.
//!
//! ## Histogram construction
//!
//! [`Histogram`] mirrors the `AutoMinimumMaximum` path shared by
//! `itk::Statistics::ImageToHistogramFilter` (used by every
//! `HistogramThresholdImageFilter` subclass) and
//! `itk::Statistics::SampleToHistogramFilter` (used by
//! `OtsuMultipleThresholdsImageFilter` via `ScalarImageToHistogramGenerator`):
//! both build a single-dimension histogram of equal-width bins spanning
//! `[min, max + margin]`, where `margin = (max - min) / bins /
//! marginalScale` (`marginalScale` defaults to 100 in both) is added only to
//! the upper bound, and both leave `ClipBinsAtEnds` at its default `true`.
//! `itkHistogram.hxx`'s `GetIndex` then assigns a value to the bin whose
//! half-open `[min, max)` contains it, except a value at or past the very
//! last bin's upper edge clips into that last bin. On a constant image
//! `margin == 0`, so every bin edge collapses to the same value and every
//! pixel clips into the *last* bin — not bin 0.
//!
//! ITK computes bin edges in `NumericTraits<T>::RealType`, which is `double`
//! for **every** scalar pixel type including `float`
//! (itkNumericTraits.h:1349/1356) — and `ScalarImageToHistogramGenerator`
//! hardcodes `Histogram<double>` outright — so both upstream Otsu paths run
//! bin edges in `double` for every input. This port computes bin edges in
//! `f64` uniformly for every caller, matching that rule exactly.

use crate::error::{FilterError, Result};
use sitk_core::parallel;

/// See the module docs for the construction convention. Single dimension
/// only (this crate's images are scalar-pixel, so the 1-D case is all
/// callers need).
pub(crate) struct Histogram {
    bin_min: Vec<f64>,
    bin_max: Vec<f64>,
    frequency: Vec<u64>,
    total: u64,
}

impl Histogram {
    pub(crate) fn from_values(vals: &[f64], bins: u32) -> Result<Self> {
        if bins == 0 {
            return Err(FilterError::InvalidHistogramBins(0));
        }
        if vals.is_empty() {
            return Err(FilterError::DegenerateRange);
        }
        let bins = bins as usize;

        // `min`/`max` select an element of the input set: exactly associative, so
        // the chunked scan returns the same bits as the sequential one.
        let (lo, hi) = parallel::min_max(vals).ok_or(FilterError::DegenerateRange)?;

        // `itkImageToHistogramFilter.hxx`'s `ApplyMarginalScale` /
        // `itkSampleToHistogramFilter.hxx`'s equivalent: margin added only to
        // the upper bound, `marginalScale` defaults to 100 in both.
        let margin = (hi - lo) / bins as f64 / 100.0;
        let lower = lo;
        let upper = hi + margin;
        // `itkHistogram.hxx`'s `Initialize(size, lowerBound, upperBound)`.
        let interval = (upper - lower) / bins as f64;

        let mut bin_min = Vec::with_capacity(bins);
        let mut bin_max = Vec::with_capacity(bins);
        for j in 0..bins {
            bin_min.push(lower + j as f64 * interval);
            bin_max.push(if j + 1 == bins {
                upper
            } else {
                lower + (j + 1) as f64 * interval
            });
        }

        let mut hist = Self {
            bin_min,
            bin_max,
            frequency: vec![0; bins],
            total: vals.len() as u64,
        };
        // Parallel integer counting — see `from_bounds` for the argument. Here
        // every value is binned (clipped at the ends), so `total` is `vals.len()`.
        hist.frequency = parallel::bin_counts(vals, bins, |v| Some(hist.bin_index(v)));
        Ok(hist)
    }

    /// `itkHistogram::Initialize(size, lowerBound, upperBound)` as used by
    /// `itkHistogramMatchingImageFilter.hxx`'s `ConstructHistogramFromIntensityRange`:
    /// unlike [`from_values`](Self::from_values), the bin range is caller-supplied
    /// (`[lower, upper]`, no automatic margin) and only values inside that
    /// closed range are counted (matching the `.hxx`'s explicit
    /// `if (value >= lowerBound && value <= upperBound)` guard before
    /// binning), so `total` is the count actually inserted, not `vals.len()`.
    /// `true_min`/`true_max` overwrite the first bin's minimum and the last
    /// bin's maximum after uniform bin edges are laid out, matching the
    /// `.hxx`'s `SetBinMin(0, trueMin)`/`SetBinMax(size-1, trueMax)` override.
    ///
    /// Deviates from ITK the same way [`from_values`](Self::from_values) does: bin
    /// edges are computed uniformly in `f64` rather than ITK's hardcoded
    /// `float`-precision `Initialize`.
    pub(crate) fn from_bounds(
        vals: &[f64],
        bins: u32,
        lower: f64,
        upper: f64,
        true_min: f64,
        true_max: f64,
    ) -> Result<Self> {
        if bins == 0 {
            return Err(FilterError::InvalidHistogramBins(0));
        }
        if vals.is_empty() {
            return Err(FilterError::DegenerateRange);
        }
        let bins = bins as usize;
        let interval = (upper - lower) / bins as f64;

        let mut bin_min = Vec::with_capacity(bins);
        let mut bin_max = Vec::with_capacity(bins);
        for j in 0..bins {
            bin_min.push(lower + j as f64 * interval);
            bin_max.push(lower + (j + 1) as f64 * interval);
        }
        bin_min[0] = true_min;
        bin_max[bins - 1] = true_max;

        let mut hist = Self {
            bin_min,
            bin_max,
            frequency: vec![0; bins],
            total: 0,
        };
        // Parallel integer counting: `bin_index` is a pure function of one value
        // against the bin edges fixed above, and `u64` addition is exactly
        // associative, so the chunk-combined counts equal the sequential ones.
        // `total` is the number actually inserted — the values the range guard
        // rejects land in no bin at all.
        let frequency = parallel::bin_counts(vals, bins, |v| {
            (v >= lower && v <= upper).then(|| hist.bin_index(v))
        });
        hist.total = frequency.iter().sum();
        hist.frequency = frequency;
        Ok(hist)
    }

    /// `itkHistogram.hxx`'s `GetIndex`, specialized to one dimension with
    /// `ClipBinsAtEnds == true` (the default both upstream generators leave
    /// it at whenever the marginal-scale computation doesn't overflow, which
    /// `f64` image intensities never do in practice): a value below the
    /// first bin's minimum or past the last bin's maximum clips to that
    /// bin's edge. The two checks are asymmetric on purpose, matching
    /// upstream's check order (`v < min[0]` first, `v >= max[last]` second):
    /// on a degenerate histogram where `min[0] == max[last]` (e.g. a
    /// constant image, `margin == 0`), a value equal to both must fail the
    /// strict `<` first check and fall through to clip into the *last* bin,
    /// not bin 0.
    pub(crate) fn bin_index(&self, v: f64) -> usize {
        let last = self.bins() - 1;
        if v < self.bin_min[0] {
            return 0;
        }
        if v >= self.bin_max[last] {
            return last;
        }
        match self
            .bin_min
            .binary_search_by(|probe| probe.partial_cmp(&v).unwrap())
        {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }

    pub(crate) fn bins(&self) -> usize {
        self.bin_min.len()
    }

    pub(crate) fn frequency(&self, i: usize) -> u64 {
        self.frequency[i]
    }

    pub(crate) fn total_frequency(&self) -> u64 {
        self.total
    }

    pub(crate) fn bin_min(&self, i: usize) -> f64 {
        self.bin_min[i]
    }

    pub(crate) fn bin_max(&self, i: usize) -> f64 {
        self.bin_max[i]
    }

    /// `itkHistogram.hxx`'s `GetMeasurement`/`GetMeasurementVector`: a bin's
    /// centroid, `(min + max) / 2`.
    pub(crate) fn midpoint(&self, i: usize) -> f64 {
        (self.bin_min[i] + self.bin_max[i]) / 2.0
    }

    /// `itkHistogram.hxx`'s `Quantile`, specialized to one dimension: the
    /// value at cumulative-frequency proportion `p`, found by walking bins
    /// from the low end (`p < 0.5`) or the high end (`p >= 0.5`) and
    /// interpolating within the bin where the cumulative proportion crosses
    /// `p`.
    pub(crate) fn quantile(&self, p: f64) -> f64 {
        let size = self.bins();
        let total = self.total as f64;
        let mut cumulated = 0.0;

        if p < 0.5 {
            let mut n = 0usize;
            let mut p_n = 0.0;
            let (mut p_n_prev, mut f_n);
            loop {
                f_n = self.frequency(n) as f64;
                cumulated += f_n;
                p_n_prev = p_n;
                p_n = cumulated / total;
                n += 1;
                if !(n < size && p_n < p) {
                    break;
                }
            }
            let bin_proportion = f_n / total;
            let min = self.bin_min[n - 1];
            let max = self.bin_max[n - 1];
            min + ((p - p_n_prev) / bin_proportion) * (max - min)
        } else {
            let mut n: i64 = size as i64 - 1;
            let mut m = 0usize;
            let mut p_n = 1.0;
            let (mut p_n_prev, mut f_n);
            loop {
                f_n = self.frequency(n as usize) as f64;
                cumulated += f_n;
                p_n_prev = p_n;
                p_n = 1.0 - cumulated / total;
                n -= 1;
                m += 1;
                if !(m < size && p_n > p) {
                    break;
                }
            }
            let bin_proportion = f_n / total;
            let min = self.bin_min[(n + 1) as usize];
            let max = self.bin_max[(n + 1) as usize];
            max - ((p_n_prev - p) / bin_proportion) * (max - min)
        }
    }

    pub(crate) fn quantile_index(&self, p: f64) -> usize {
        self.bin_index(self.quantile(p))
    }

    /// `itk::Statistics::Histogram::Mean(0)`: the frequency-weighted mean of
    /// every bin's midpoint. Used by
    /// [`crate::threshold::isodata_threshold`] as the fallback when no
    /// isodata crossing is found.
    pub(crate) fn mean(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.bins() {
            sum += self.midpoint(i) * self.frequency(i) as f64;
        }
        sum / self.total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_collapses_into_last_bin() {
        let vals = vec![3.0; 10];
        let hist = Histogram::from_values(&vals, 5).unwrap();
        assert_eq!(hist.frequency(4), 10);
        for i in 0..4 {
            assert_eq!(hist.frequency(i), 0);
        }
        // margin == 0 on a constant image, so every bin edge collapses to
        // the constant value.
        assert_eq!(hist.bin_max(0), 3.0);
    }

    #[test]
    fn single_bin_holds_the_whole_range() {
        let vals = vec![0.0, 5.0, 10.0];
        let hist = Histogram::from_values(&vals, 1).unwrap();
        assert_eq!(hist.bins(), 1);
        assert_eq!(hist.frequency(0), 3);
        assert_eq!(hist.total_frequency(), 3);
    }

    #[test]
    fn zero_bins_errors() {
        let vals = vec![0.0, 1.0];
        assert!(matches!(
            Histogram::from_values(&vals, 0),
            Err(FilterError::InvalidHistogramBins(0))
        ));
    }

    #[test]
    fn empty_image_errors() {
        assert!(matches!(
            Histogram::from_values(&[], 4),
            Err(FilterError::DegenerateRange)
        ));
    }

    #[test]
    fn bimodal_split_across_bins() {
        // 50 pixels at 0.0, 50 pixels at 100.0, 10 bins: margin = 100/10/100 = 0.1,
        // interval = 10.01, so 0.0 falls in bin 0 and 100.0 falls in bin 9.
        let mut vals = vec![0.0; 50];
        vals.extend(vec![100.0; 50]);
        let hist = Histogram::from_values(&vals, 10).unwrap();
        assert_eq!(hist.frequency(0), 50);
        assert_eq!(hist.frequency(9), 50);
        for i in 1..9 {
            assert_eq!(hist.frequency(i), 0);
        }
    }

    #[test]
    fn mean_is_frequency_weighted_midpoint_average() {
        let mut vals = vec![0.0; 3];
        vals.extend(vec![10.0; 1]);
        let hist = Histogram::from_values(&vals, 2).unwrap();
        // bin 0 covers [0, ~5), bin 1 covers [~5, 10]: 3 pixels at midpoint(0),
        // 1 pixel at midpoint(1).
        let expected = (hist.midpoint(0) * 3.0 + hist.midpoint(1) * 1.0) / 4.0;
        assert!((hist.mean() - expected).abs() < 1e-9);
    }
}
