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
use sitk_core::{Image, PixelId, parallel};

/// The mask of the `HistogramThresholdImageFilter` family — **one owner for all
/// twelve** of this crate's histogram-driven thresholds, because twelve local mask
/// branches is twelve chances to disagree about what a mask means.
///
/// ITK routes a masked threshold's histogram through
/// `Statistics::MaskedImageToHistogramFilter` (`itkHistogramThresholdImageFilter.hxx:78-89`),
/// and the mask changes the *threshold*, not just the output: the histogram — and,
/// under `AutoMinimumMaximum`, its **bin range** — is built from the admitted voxels
/// only (`itkMaskedImageToHistogramFilter.hxx`, `ThreadedComputeMinimumAndMaximum` and
/// `ThreadedStreamedGenerateData`, both gated on `maskIt.Get() == maskValue`).
///
/// # Two different mask comparisons, in one filter, on purpose
///
/// Reproduced exactly, because it is not what a reader would guess:
///
/// * **Histogram inclusion** is `mask == mask_value` — an *exact equality*, not
///   `!= 0`. `mask_value` defaults to `NumericTraits<MaskPixelType>::max()`, i.e.
///   **255** (`itkHistogramThresholdImageFilter.hxx`'s ctor; SimpleITK's yaml carries
///   the same `255u` default).
/// * **Output masking**, when [`mask_output`](Self::mask_output) is true (ITK's and
///   SimpleITK's default), runs the thresholded image through `MaskImageFilter`
///   (`.hxx:113-125`), which zeroes where the mask equals its *masking value* — and
///   that is **`0`**, not `mask_value`.
///
/// So with the default `mask_value == 255`, a voxel whose mask is `7` is **excluded
/// from the histogram** and **kept in the output**. That asymmetry is upstream's, it
/// is reachable from SimpleITK, and this port reproduces it rather than tidying it.
pub struct ThresholdMask<'a> {
    image: &'a Image,
    mask_value: u8,
    mask_output: bool,
}

impl<'a> ThresholdMask<'a> {
    /// ITK's and SimpleITK's defaults: `mask_value = 255`, `mask_output = true`.
    pub fn new(image: &'a Image) -> Self {
        Self {
            image,
            mask_value: 255,
            mask_output: true,
        }
    }

    /// The value a mask voxel must **equal** to be admitted to the histogram.
    pub fn with_mask_value(mut self, mask_value: u8) -> Self {
        self.mask_value = mask_value;
        self
    }

    /// When true (the default), output voxels whose mask is **`0`** are set to `0`.
    /// Note the value: `0`, not `mask_value` — see the type docs.
    pub fn with_mask_output(mut self, mask_output: bool) -> Self {
        self.mask_output = mask_output;
        self
    }

    /// The voxels this mask admits to the histogram: `mask == mask_value`, exactly.
    ///
    /// Errors with [`FilterError::MaskAdmitsNoVoxels`] when the selection is empty —
    /// see that variant for why the port refuses instead of reproducing (upstream
    /// throws in eleven calculators and returns a `NaN` threshold in two).
    fn selected(&self, img: &Image, vals: &[f64]) -> Result<Vec<f64>> {
        if self.image.size() != img.size() {
            return Err(FilterError::SizeMismatch {
                a: img.size().to_vec(),
                b: self.image.size().to_vec(),
            });
        }
        // ITK does not *sample* the mask: it is a second `ImageToImageFilter` input, so
        // `VerifyInputInformation` (`itkImageToImageFilter.hxx:148-223`) throws "Inputs do
        // not occupy the same physical space!" unless the mask's origin, spacing and
        // direction agree with the image's. Same grid or refuse; no resampling, no
        // index-aligned-but-physically-elsewhere mask.
        if !crate::geometry::same_physical_space(img, self.image) {
            return Err(FilterError::PhysicalSpaceMismatch { index: 1 });
        }
        let mask = self.image.to_f64_vec()?;
        let wanted = f64::from(self.mask_value);
        let selected: Vec<f64> = vals
            .iter()
            .zip(mask.iter())
            .filter(|&(_, &m)| m == wanted)
            .map(|(&v, _)| v)
            .collect();
        if selected.is_empty() {
            return Err(FilterError::MaskAdmitsNoVoxels {
                mask_value: self.mask_value,
            });
        }
        Ok(selected)
    }

    /// `MaskImageFilter` on the thresholded output: zero where the mask is **`0`**.
    fn apply_to_output(&self, out: &mut [u8]) -> Result<()> {
        if !self.mask_output {
            return Ok(());
        }
        let mask = self.image.to_f64_vec()?;
        for (o, &m) in out.iter_mut().zip(mask.iter()) {
            if m == 0.0 {
                *o = 0;
            }
        }
        Ok(())
    }
}

/// `HistogramThresholdImageFilter`'s constructor turns `AutoMinimumMaximum` **off**
/// for the three 8-bit pixel types and leaves it on for every other
/// (`itkHistogramThresholdImageFilter.hxx:44-53`):
///
/// ```text
/// if (typeid(ValueType) == typeid(signed char) || typeid(ValueType) == typeid(unsigned char) ||
///     typeid(ValueType) == typeid(char))
///   { m_AutoMinimumMaximum = false; }
/// else
///   { m_AutoMinimumMaximum = true; }
/// ```
///
/// With it off, and with no explicit `HistogramBinMinimum`/`Maximum` (the filter never
/// sets one), `ImageToHistogramFilter::InitializeOutputHistogram` bins over
/// `[NonpositiveMin() - 0.5, max() + 0.5]` — the **pixel type's** range, half-integer
/// offsets and all, with no marginal scale (`itkImageToHistogramFilter.hxx:155-175`).
/// So an 8-bit image's threshold is computed against `[-0.5, 255.5]` no matter how
/// narrow its data is, and 128 bins are 2 grey levels wide by construction.
///
/// Returns `None` for every other pixel type, which then bins over the data range.
///
/// **This is the family's rule, not the crate's.** `OtsuMultipleThresholdsImageFilter`
/// is a different ITK class: it builds its histogram through
/// `ScalarImageToHistogramGenerator` → `SampleToHistogramFilter`, whose constructor sets
/// `AutoMinimumMaximum = true` unconditionally, with no pixel-type branch
/// (`itkSampleToHistogramFilter.hxx:35`), and the filter never turns it off. It therefore
/// bins over the *data* range for 8-bit input too — so ITK's two Otsu implementations
/// disagree with each other on an 8-bit image (ledger §2.174). `otsu_multiple_thresholds`
/// calls `Histogram::from_values` directly and must keep doing so; only the twelve that
/// route through [`threshold_histogram`] take this rule.
fn fixed_bin_range(pixel_id: PixelId) -> Option<(f64, f64)> {
    match pixel_id {
        PixelId::UInt8 => Some((-0.5, 255.5)),
        PixelId::Int8 => Some((-128.5, 127.5)),
        _ => None,
    }
}

/// The histogram every one of this crate's twelve histogram thresholds is built from:
/// all voxels, or — when a mask is given — exactly the voxels it admits. The single place
/// both the masked/unmasked choice and the bin-range choice are made.
///
/// The two interact, and not in the direction the flag's name suggests: the mask scopes
/// the bin range **only** on the auto path. `MaskedImageToHistogramFilter` overrides
/// `ThreadedComputeMinimumAndMaximum`, but that scan is called from inside
/// `InitializeOutputHistogram`'s `AutoMinimumMaximum` branch alone
/// (`itkImageToHistogramFilter.hxx:140-152`) — so on an 8-bit image the mask selects
/// which voxels are *counted* and has no say in the range at all.
pub(crate) fn threshold_histogram(
    img: &Image,
    vals: &[f64],
    bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<Histogram> {
    let selected;
    let counted = match mask {
        None => vals,
        Some(m) => {
            selected = m.selected(img, vals)?;
            &selected
        }
    };
    match fixed_bin_range(img.pixel_id()) {
        Some((lower, upper)) => Histogram::from_fixed_range(counted, bins, lower, upper),
        None => Histogram::from_values(counted, bins),
    }
}

/// The output-masking half of the same rule, so no caller hand-rolls it.
pub(crate) fn apply_threshold_mask_output(
    out: &mut [u8],
    mask: Option<&ThresholdMask>,
) -> Result<()> {
    match mask {
        None => Ok(()),
        Some(m) => m.apply_to_output(out),
    }
}

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

        // `min`/`max` select an element of the input set: exactly associative, so
        // the chunked scan returns the same bits as the sequential one.
        let (lo, hi) = parallel::min_max(vals).ok_or(FilterError::DegenerateRange)?;

        // `itkImageToHistogramFilter.hxx`'s `ApplyMarginalScale` /
        // `itkSampleToHistogramFilter.hxx`'s equivalent: margin added only to
        // the upper bound, `marginalScale` defaults to 100 in both.
        let margin = (hi - lo) / f64::from(bins) / 100.0;
        Self::over_range(vals, bins, lo, hi + margin)
    }

    /// The **non-`AutoMinimumMaximum`** path of `itkImageToHistogramFilter`
    /// (`.hxx:155-175`): the bin range is fixed, no marginal scale is applied
    /// ("No marginal scaling is applied in this case"), and the values are
    /// counted into it with `ClipBinsAtEnds` still `true`.
    ///
    /// The range is the *pixel type's*, not the data's — see [`fixed_bin_range`],
    /// which is where the choice between this and [`from_values`](Self::from_values)
    /// is made.
    pub(crate) fn from_fixed_range(
        vals: &[f64],
        bins: u32,
        lower: f64,
        upper: f64,
    ) -> Result<Self> {
        if bins == 0 {
            return Err(FilterError::InvalidHistogramBins(0));
        }
        if vals.is_empty() {
            return Err(FilterError::DegenerateRange);
        }
        Self::over_range(vals, bins, lower, upper)
    }

    /// `itkHistogram.hxx`'s `Initialize(size, lowerBound, upperBound)` with
    /// `ClipBinsAtEnds(true)`: equal-width bins spanning `[lower, upper]`, and
    /// every value counted (clipped into the first/last bin if it falls outside),
    /// so `total` is `vals.len()`. The two range rules above differ only in how
    /// they arrive at `lower`/`upper`; the binning itself is one function.
    fn over_range(vals: &[f64], bins: u32, lower: f64, upper: f64) -> Result<Self> {
        let bins = bins as usize;
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
