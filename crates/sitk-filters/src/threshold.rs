//! ITK's plain-clamp threshold filter and the `HistogramThresholdCalculator`
//! family of automatic-threshold calculators (everything except Otsu and
//! Triangle, which live in [`crate::intensity`] alongside the histogram
//! scaffolding they established first).
//!
//! Verified against ITK's headers:
//!
//! - `Modules/Filtering/Thresholding/include/itkThresholdImageFilter.h(.hxx)`
//! - `Modules/Filtering/Thresholding/include/itkHistogramThresholdCalculator.h`,
//!   `itkHistogramThresholdImageFilter.h(.hxx)` (shared calculator/binarization
//!   scaffolding)
//! - `itkHuangThresholdCalculator.h(.hxx)`, `itkIntermodesThresholdCalculator.h(.hxx)`,
//!   `itkIntermodesThresholdImageFilter.h`, `itkIsoDataThresholdCalculator.h(.hxx)`,
//!   `itkKittlerIllingworthThresholdCalculator.h(.hxx)`, `itkLiThresholdCalculator.h(.hxx)`,
//!   `itkMaximumEntropyThresholdCalculator.h(.hxx)`, `itkMomentsThresholdCalculator.h(.hxx)`,
//!   `itkRenyiEntropyThresholdCalculator.h(.hxx)`, `itkShanbhagThresholdCalculator.h(.hxx)`,
//!   `itkYenThresholdCalculator.h(.hxx)`
//! - SimpleITK's generated-wrapper parameter defaults:
//!   `Code/BasicFilters/yaml/{Threshold,Huang,Intermodes,IsoData,KittlerIllingworth,
//!   Li,MaximumEntropy,Moments,RenyiEntropy,Shanbhag,Yen}ThresholdImageFilter.yaml`
//!
//! Every calculator here shares [`crate::intensity::otsu_threshold`]'s
//! binarization convention: `itkHistogramThresholdImageFilter.hxx` always
//! runs its internal `BinaryThresholdImageFilter` with `LowerThreshold =
//! NonpositiveMin`, `UpperThreshold = <computed threshold>`, so a pixel `<=
//! threshold` gets `inside_value` and everything else gets `outside_value`.
//! SimpleITK's wrapper defaults (`InsideValue = 1`, `OutsideValue = 0` for
//! every filter in this family) are left for the caller to supply explicitly,
//! matching this crate's existing threshold functions.

use crate::error::{FilterError, Result};
use crate::histogram::{
    Histogram, ThresholdMask, apply_threshold_mask_output, threshold_histogram,
};
use sitk_core::Image;

// ---- ThresholdImageFilter --------------------------------------------------

/// `ThresholdImageFilter` (`itkThresholdImageFilter.h(.hxx)`): pixels outside
/// the closed range `[lower, upper]` are replaced with `outside_value`;
/// pixels inside pass through unchanged. Unlike [`crate::binary_threshold`],
/// the output pixel type matches the input's own type rather than always
/// being `UInt8`, matching `ThresholdImageFilter.yaml`'s `in_place: true`.
///
/// `DynamicThreadedGenerateData` compares `lower <= value && value <=
/// upper` in the pixel type; this port compares in `f64` instead (as
/// [`crate::binary_threshold`] already does), which only differs from
/// upstream in low-order bits for integer types wider than `f64`'s 53-bit
/// mantissa.
pub fn threshold(img: &Image, lower: f64, upper: f64, outside_value: f64) -> Result<Image> {
    let vals = img.to_f64_vec()?;
    let out: Vec<f64> = vals
        .iter()
        .map(|&v| {
            if v >= lower && v <= upper {
                v
            } else {
                outside_value
            }
        })
        .collect();
    crate::image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- shared calculator scaffolding -----------------------------------------

/// `itkHistogramThresholdImageFilter.hxx`'s internal `BinaryThresholdImageFilter`
/// call: `<= threshold` gets `inside_value`, everything else gets
/// `outside_value`. See the module docs.
fn binarize_and_finish(
    img: &Image,
    vals: &[f64],
    threshold: f64,
    inside_value: u8,
    outside_value: u8,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let mut out: Vec<u8> = vals
        .iter()
        .map(|&v| {
            if v <= threshold {
                inside_value
            } else {
                outside_value
            }
        })
        .collect();
    // `MaskImageFilter` on the thresholded output, as `HistogramThresholdImageFilter`
    // does when `MaskOutput` is set (`.hxx:113-125`). Note it zeroes where the mask is
    // `0`, not where it differs from `mask_value` — see `ThresholdMask`'s docs.
    apply_threshold_mask_output(&mut out, mask)?;
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok((result, threshold))
}

/// `itk::Math::RoundHalfIntegerUp` (`Modules/Core/Common/include/itkMath.h`),
/// which `itk::Math::Round` is a synonym for: rounds half-integers *up*
/// (toward `+inf`), e.g. `-1.5 -> -1`. This disagrees with Rust's
/// `f64::round` (round half *away* from zero, e.g. `-1.5 -> -2`) for
/// negative half-integers, so [`huang_threshold_value`] uses this instead of
/// `.round()`.
fn round_half_up(x: f64) -> f64 {
    (x + 0.5).floor()
}

// ---- Huang ------------------------------------------------------------

/// `HuangThresholdCalculator::GenerateData` (`itkHuangThresholdCalculator.hxx`):
/// finds the bin minimizing Shannon fuzzy entropy between a "background"
/// partition (`[first_bin, threshold]`) and an "object" partition
/// (`[threshold+1, last_bin]`), where each partition's fuzzy membership is
/// derived from its distance to the *other* partition's [`round_half_up`]
/// weighted mean.
///
/// A single-bin histogram returns that bin's midpoint directly, matching the
/// calculator's own early return. On a constant image with more than one
/// bin, every pixel clips into the last bin (see [`crate::histogram`]), so
/// `first_bin == last_bin` and the search range `first_bin..last_bin` is
/// empty — `best_threshold` never leaves its ITK-initialized `0`. This is
/// harmless: a degenerate histogram's bin edges all collapse to the same
/// value (`margin == 0`), so bin 0's midpoint equals the constant too.
fn huang_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    if size == 1 {
        return hist.midpoint(0);
    }

    let first_bin = (0..size)
        .find(|&i| hist.frequency(i) > 0)
        .expect("Histogram::from_values guarantees a non-empty histogram");
    let mut last_bin = size - 1;
    while last_bin > first_bin && hist.frequency(last_bin) == 0 {
        last_bin -= 1;
    }

    let mut s = vec![0.0f64; last_bin + 1];
    let mut w = vec![0.0f64; last_bin + 1];
    s[0] = hist.frequency(0) as f64;
    for i in 1.max(first_bin)..=last_bin {
        s[i] = s[i - 1] + hist.frequency(i) as f64;
        w[i] = w[i - 1] + hist.midpoint(i) * hist.frequency(i) as f64;
    }

    let c = (last_bin - first_bin) as f64;
    let mut smu = vec![0.0f64; last_bin + 1 - first_bin];
    for (i, smu_i) in smu.iter_mut().enumerate().skip(1) {
        let mu = 1.0 / (1.0 + i as f64 / c);
        *smu_i = -mu * mu.ln() - (1.0 - mu) * (1.0 - mu).ln();
    }

    let mut best_threshold = 0usize;
    let mut best_entropy = f64::MAX;
    for threshold in first_bin..last_bin {
        let mut entropy = 0.0;

        let mu = round_half_up(w[threshold] / s[threshold]);
        let mu_idx = hist.bin_index(mu);
        for i in first_bin..=threshold {
            entropy += smu[i.abs_diff(mu_idx)] * hist.frequency(i) as f64;
        }

        let mu2 = round_half_up((w[last_bin] - w[threshold]) / (s[last_bin] - s[threshold]));
        let mu2_idx = hist.bin_index(mu2);
        for i in (threshold + 1)..=last_bin {
            entropy += smu[i.abs_diff(mu2_idx)] * hist.frequency(i) as f64;
        }

        if best_entropy > entropy {
            best_entropy = entropy;
            best_threshold = threshold;
        }
    }

    hist.midpoint(best_threshold)
}

/// `HuangThresholdCalculator` + `HuangThresholdImageFilter`: SimpleITK
/// defaults `number_of_histogram_bins` to 128
/// (`HuangThresholdImageFilter.yaml`, matching Otsu's default rather than
/// the 256 every other calculator in this family uses).
///
/// Returns the binarized image alongside the computed threshold value.
pub fn huang_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = huang_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- Intermodes ------------------------------------------------------------

/// `IntermodesThresholdCalculator::BimodalTest`
/// (`itkIntermodesThresholdCalculator.hxx`): true when the sequence (a
/// smoothed histogram, treated as plain numbers) has exactly two strict
/// interior local maxima.
fn bimodal_test(h: &[f64]) -> bool {
    let mut modes = 0;
    for k in 1..h.len() - 1 {
        if h[k - 1] < h[k] && h[k + 1] < h[k] {
            modes += 1;
            if modes > 2 {
                return false;
            }
        }
    }
    modes == 2
}

/// `IntermodesThresholdCalculator::GenerateData`
/// (`itkIntermodesThresholdCalculator.hxx`): repeatedly smooths the
/// histogram with a 3-point running mean until it has exactly two modes,
/// then thresholds at the mean of the two peak bins.
/// `maximum_smoothing_iterations` is fixed at 1000, matching
/// `itkIntermodesThresholdImageFilter.h`'s own default (the raw
/// `IntermodesThresholdCalculator` class defaults to 10000, but SimpleITK
/// only ever drives it through `IntermodesThresholdImageFilter`, which
/// overwrites that with 1000 unconditionally — its yaml exposes neither
/// knob). Likewise `use_inter_mode` is always `true` here (the "minimum
/// between peaks" alternative is dead code under SimpleITK's fixed
/// wrapper, so this port doesn't carry it).
///
/// A single-mode histogram (e.g. a constant image, which collapses into a
/// single populated bin — see [`crate::histogram`]) never becomes bimodal no
/// matter how much smoothing, so this hits the iteration cap and returns
/// [`FilterError::ThresholdCalculatorFailed`], matching ITK's own
/// `itkGenericExceptionMacro` here.
fn intermodes_threshold_value(hist: &Histogram) -> Result<f64> {
    const MAXIMUM_SMOOTHING_ITERATIONS: u32 = 1000;

    let size = hist.bins();
    if size == 1 {
        return Ok(hist.midpoint(0));
    }

    let mut smoothed: Vec<f64> = (0..size).map(|i| hist.frequency(i) as f64).collect();
    let mut iterations = 0u32;
    while !bimodal_test(&smoothed) {
        let mut previous;
        let mut current = 0.0;
        let mut next = smoothed[0];
        for i in 0..smoothed.len() - 1 {
            previous = current;
            current = next;
            next = smoothed[i + 1];
            smoothed[i] = (previous + current + next) / 3.0;
        }
        let last = smoothed.len() - 1;
        smoothed[last] = (current + next) / 3.0;

        iterations += 1;
        if iterations > MAXIMUM_SMOOTHING_ITERATIONS {
            return Err(FilterError::ThresholdCalculatorFailed {
                calculator: "Intermodes",
                reason: "exceeded maximum iterations for histogram smoothing",
            });
        }
    }

    let mut peak_sum = 0usize;
    for i in 1..smoothed.len() - 1 {
        if smoothed[i - 1] < smoothed[i] && smoothed[i + 1] < smoothed[i] {
            peak_sum += i;
        }
    }
    Ok(hist.midpoint(peak_sum / 2))
}

/// `IntermodesThresholdCalculator` + `IntermodesThresholdImageFilter`:
/// SimpleITK defaults `number_of_histogram_bins` to 256
/// (`IntermodesThresholdImageFilter.yaml`).
///
/// Returns the binarized image alongside the computed threshold value.
pub fn intermodes_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = intermodes_threshold_value(&hist)?;
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- IsoData ----------------------------------------------------------

/// `IsoDataThresholdCalculator::GenerateData`
/// (`itkIsoDataThresholdCalculator.hxx`): advances a candidate split bin
/// upward from the first non-empty bin, stopping once the split bin's own
/// measurement is at least the average of the low- and high-side weighted
/// means computed from it (isodata convergence). If it runs past the last
/// bin without converging (e.g. a single-mode histogram), it falls back to
/// [`Histogram::mean`].
fn isodata_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    if size == 1 {
        return hist.midpoint(0);
    }

    let mut current_pos = 0usize;
    loop {
        match (current_pos..size).find(|&i| hist.frequency(i) > 0) {
            Some(i) => current_pos = i,
            None => return hist.mean(),
        }

        let mut l = 0.0;
        let mut totl = 0.0;
        for i in 0..=current_pos {
            totl += hist.frequency(i) as f64;
            l += hist.midpoint(i) * hist.frequency(i) as f64;
        }
        let mut h = 0.0;
        let mut toth = 0.0;
        for i in (current_pos + 1)..size {
            toth += hist.frequency(i) as f64;
            h += hist.midpoint(i) * hist.frequency(i) as f64;
        }

        if totl > f64::EPSILON && toth > f64::EPSILON {
            let l = l / totl;
            let h = h / toth;
            if hist.midpoint(current_pos) >= (l + h) * 0.5 {
                return hist.midpoint(current_pos);
            }
        }

        current_pos += 1;
    }
}

/// `IsoDataThresholdCalculator` + `IsoDataThresholdImageFilter`: SimpleITK
/// defaults `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn isodata_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = isodata_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- KittlerIllingworth ----------------------------------------------------

/// `A(j)`/`B(j)`/`C(j)` in `itkKittlerIllingworthThresholdCalculator.hxx`:
/// the raw (`A`), measurement-weighted (`B`), and measurement-squared
/// weighted (`C`) cumulative frequency sums over bins `0..=j`.
fn cumulative_a(hist: &Histogram, j: usize) -> f64 {
    (0..=j).map(|i| hist.frequency(i) as f64).sum()
}

fn cumulative_b(hist: &Histogram, j: usize) -> f64 {
    (0..=j)
        .map(|i| hist.midpoint(i) * hist.frequency(i) as f64)
        .sum()
}

fn cumulative_c(hist: &Histogram, j: usize) -> f64 {
    (0..=j)
        .map(|i| {
            let m = hist.midpoint(i);
            m * m * hist.frequency(i) as f64
        })
        .sum()
}

/// `KittlerIllingworthThresholdCalculator::GenerateData`
/// (`itkKittlerIllingworthThresholdCalculator.hxx`): iterates a minimum
/// classification-error criterion (fitting one Gaussian per side of the
/// threshold) to a fixed point, starting from the bin containing the
/// histogram's overall mean. `itk::Math::eps` is `f64::EPSILON` throughout.
///
/// `As1 = cumulative_a(size - 1)` is the histogram's total frequency, always
/// `> 0` for a non-empty histogram, so this port omits the corresponding
/// upstream degeneracy check (unreachable here). The remaining numerical
/// degeneracies (near-zero denominators the reference algorithm cannot
/// recover from) make ITK throw via `itkGenericExceptionMacro`; this port
/// surfaces the same condition as
/// [`FilterError::ThresholdCalculatorFailed`]. Two "not converging" cases
/// are non-fatal in ITK (`itkWarningMacro`, then it just stops iterating and
/// keeps the last threshold) and are ported as a plain loop `break` — note
/// that in the `w0 ≈ 0` branch, upstream does *not* update its
/// non-convergence sentinel (`Tprev`), a quirk reproduced here exactly.
///
/// Because this fits a Gaussian to each side, a partition with no internal
/// spread (e.g. a histogram built from two pure delta spikes, each side's
/// mass concentrated in a single bin) makes that side's variance exactly
/// `0`, hitting the `"sigma2 <= 0"` error on the very first iteration —
/// this is expected for that input, not a defect.
fn kittler_illingworth_threshold_value(hist: &Histogram) -> Result<f64> {
    fn fail(reason: &'static str) -> FilterError {
        FilterError::ThresholdCalculatorFailed {
            calculator: "KittlerIllingworth",
            reason,
        }
    }

    let size = hist.bins();
    if size == 1 {
        return Ok(hist.midpoint(0));
    }

    let mut threshold = hist.bin_index(hist.mean()) as i64;
    let mut t_prev: i64 = -2;

    let as1 = cumulative_a(hist, size - 1);
    let bs1 = cumulative_b(hist, size - 1);
    let cs1 = cumulative_c(hist, size - 1);

    while threshold != t_prev {
        let idx = threshold as usize;
        let at = cumulative_a(hist, idx);
        let bt = cumulative_b(hist, idx);
        let ct = cumulative_c(hist, idx);

        if at.abs() < f64::EPSILON {
            return Err(fail("At = 0"));
        }
        let mu = bt / at;

        if (as1 - at).abs() < f64::EPSILON {
            break;
        }
        let nu = (bs1 - bt) / (as1 - at);
        let p = at / as1;
        let q = (as1 - at) / as1;
        let sigma2 = ct / at - mu * mu;
        let tau2 = (cs1 - ct) / (as1 - at) - nu * nu;

        if sigma2 < f64::EPSILON {
            return Err(fail("sigma2 <= 0"));
        }
        if tau2.abs() < f64::EPSILON {
            return Err(fail("tau2 = 0"));
        }
        if p.abs() < f64::EPSILON {
            return Err(fail("p = 0"));
        }

        let w0 = 1.0 / sigma2 - 1.0 / tau2;
        let w1 = mu / sigma2 - nu / tau2;
        let w2 = mu * mu / sigma2 - nu * nu / tau2 + (sigma2 * q * q / (tau2 * p * p)).log10();

        let sqterm = w1 * w1 - w0 * w2;
        if sqterm < f64::EPSILON {
            break;
        }

        if w0.abs() < f64::EPSILON {
            let temp = -w2 / w1;
            threshold = hist.bin_index(temp) as i64;
        } else {
            t_prev = threshold;
            let temp = (w1 + sqterm.sqrt()) / w0;
            if temp.is_nan() {
                threshold = t_prev;
                break;
            }
            threshold = hist.bin_index(temp) as i64;
        }
    }

    Ok(hist.midpoint(threshold as usize))
}

/// `KittlerIllingworthThresholdCalculator` + `KittlerIllingworthThresholdImageFilter`:
/// SimpleITK defaults `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn kittler_illingworth_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = kittler_illingworth_threshold_value(&hist)?;
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- Li ---------------------------------------------------------------

/// `LiThresholdCalculator::GenerateData` (`itkLiThresholdCalculator.hxx`):
/// iterates Li's minimum cross-entropy fixed point starting from the overall
/// mean, re-deriving the split bin from the *rounded* candidate threshold
/// each pass (`static_cast<int>(x +/- 0.5)`, C++ truncation-toward-zero, not
/// [`round_half_up`]'s floor-based rounding) and stopping once the candidate
/// stabilizes within `0.5`.
///
/// ITK's `GenerateData` does not special-case a single-bin histogram (unlike
/// most of its siblings here) — it falls into the general loop, which this
/// port also does, since the loop already converges to bin 0's midpoint on
/// its own in that case. On a constant image with more than one bin (all
/// mass in the last bin, see [`crate::histogram`]), the first pass's rounded
/// mean looks up bin 0 (the shifted "object" mean is exactly `0`, so
/// `ln(mean_obj) == -inf` drives the candidate to `0`), and the second
/// pass's split bin (now `0`) is self-consistent — harmless, since a
/// degenerate histogram's bin edges all collapse to the same value
/// (`margin == 0`), so bin 0's midpoint equals the constant too.
fn li_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    let bin_min = hist.bin_min(0).min(0.0);
    let tolerance = 0.5;
    let num_pixels = hist.total_frequency() as f64;

    let mean: f64 = (0..size)
        .map(|i| hist.midpoint(i) * hist.frequency(i) as f64)
        .sum::<f64>()
        / num_pixels;

    let mut new_thresh = mean;
    let mut histthresh;
    loop {
        let old_thresh = new_thresh;
        let candidate = (old_thresh + 0.5).trunc();
        histthresh = hist.bin_index(candidate);

        let mut sum_back = 0.0;
        let mut num_back = 0u64;
        for i in 0..=histthresh {
            sum_back += hist.midpoint(i) * hist.frequency(i) as f64;
            num_back += hist.frequency(i);
        }
        let mean_back = if num_back == 0 {
            0.0
        } else {
            sum_back / num_back as f64
        };

        let mut sum_obj = 0.0;
        let mut num_obj = 0u64;
        for i in (histthresh + 1)..size {
            sum_obj += hist.midpoint(i) * hist.frequency(i) as f64;
            num_obj += hist.frequency(i);
        }
        let mean_obj = if num_obj == 0 {
            0.0
        } else {
            sum_obj / num_obj as f64
        };

        let mean_back = mean_back - bin_min;
        let mean_obj = mean_obj - bin_min;
        let temp = (mean_back - mean_obj) / (mean_back.ln() - mean_obj.ln());

        new_thresh = if temp < -f64::EPSILON {
            (temp - 0.5).trunc()
        } else {
            (temp + 0.5).trunc()
        };
        new_thresh += bin_min;

        if (new_thresh - old_thresh).abs() <= tolerance {
            break;
        }
    }

    hist.midpoint(histthresh)
}

/// `LiThresholdCalculator` + `LiThresholdImageFilter`: SimpleITK defaults
/// `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn li_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = li_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- shared entropy-family scaffolding (MaximumEntropy / RenyiEntropy / Shanbhag) --

/// The `norm_histo`/`P1`/`P2` triple built by `GenerateData` in
/// `itkMaximumEntropyThresholdCalculator.hxx`, `itkRenyiEntropyThresholdCalculator.hxx`,
/// and `itkShanbhagThresholdCalculator.hxx` alike: the normalized histogram
/// and its cumulative sum from each end.
struct CumulativeNormalizedHistogram {
    norm_histo: Vec<f64>,
    p1: Vec<f64>,
    p2: Vec<f64>,
}

impl CumulativeNormalizedHistogram {
    fn new(hist: &Histogram) -> Self {
        let size = hist.bins();
        let total = hist.total_frequency() as f64;
        let norm_histo: Vec<f64> = (0..size)
            .map(|i| hist.frequency(i) as f64 / total)
            .collect();

        let mut p1 = vec![0.0f64; size];
        let mut p2 = vec![0.0f64; size];
        p1[0] = norm_histo[0];
        p2[0] = 1.0 - p1[0];
        for i in 1..size {
            p1[i] = p1[i - 1] + norm_histo[i];
            p2[i] = 1.0 - p1[i];
        }

        Self { norm_histo, p1, p2 }
    }

    /// The first bin whose cumulative-from-the-low-end sum is non-negligible
    /// and the last bin whose cumulative-from-the-high-end sum is
    /// non-negligible, matching every entropy calculator's identical
    /// `first_bin`/`last_bin` scan.
    fn first_last_bin(&self) -> (usize, usize) {
        let size = self.norm_histo.len();
        let first_bin = (0..size)
            .find(|&i| self.p1[i].abs() >= f64::EPSILON)
            .unwrap_or(0);
        let last_bin = (first_bin..size)
            .rev()
            .find(|&i| self.p2[i].abs() >= f64::EPSILON)
            .unwrap_or(size - 1);
        (first_bin, last_bin)
    }
}

// ---- MaximumEntropy ---------------------------------------------------

/// `MaximumEntropyThresholdCalculator::GenerateData`
/// (`itkMaximumEntropyThresholdCalculator.hxx`): maximizes the sum of Shannon
/// entropies of the background/object partitions' normalized histograms.
/// `itk::NumericTraits<double>::min()` — the smallest *positive* double, not
/// the most negative — seeds `max_ent`, matching Rust's `f64::MIN_POSITIVE`.
fn maximum_entropy_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    let cnh = CumulativeNormalizedHistogram::new(hist);
    let (first_bin, last_bin) = cnh.first_last_bin();

    let mut threshold = 0usize;
    let mut max_ent = f64::MIN_POSITIVE;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0;
        for ih in 0..=it {
            if hist.frequency(ih) != 0 {
                let x = cnh.norm_histo[ih] / cnh.p1[it];
                ent_back -= x * x.ln();
            }
        }
        let mut ent_obj = 0.0;
        for ih in (it + 1)..size {
            if hist.frequency(ih) != 0 {
                let x = cnh.norm_histo[ih] / cnh.p2[it];
                ent_obj -= x * x.ln();
            }
        }
        let tot_ent = ent_back + ent_obj;
        if max_ent < tot_ent - 0.00001 {
            max_ent = tot_ent;
            threshold = it;
        }
    }

    hist.midpoint(threshold)
}

/// `MaximumEntropyThresholdCalculator` + `MaximumEntropyThresholdImageFilter`:
/// SimpleITK defaults `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn maximum_entropy_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = maximum_entropy_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- Moments ------------------------------------------------------------

/// `MomentsThresholdCalculator::GenerateData`
/// (`itkMomentsThresholdCalculator.hxx`): matches the first three moments of
/// the normalized histogram against a two-level (background/object) target
/// image, solving for the target's foreground fraction `p0`, then picks the
/// first bin where the cumulative normalized histogram exceeds `p0`.
fn moments_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    let total = hist.total_frequency() as f64;
    let histo: Vec<f64> = (0..size)
        .map(|i| hist.frequency(i) as f64 / total)
        .collect();

    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for (i, &h) in histo.iter().enumerate() {
        let m = hist.midpoint(i);
        m1 += m * h;
        m2 += m * m * h;
        m3 += m * m * m * h;
    }

    let cd = m2 - m1 * m1;
    let c0 = (-m2 * m2 + m1 * m3) / cd;
    let c1 = (-m3 + m2 * m1) / cd;
    let z0 = 0.5 * (-c1 - (c1 * c1 - 4.0 * c0).sqrt());
    let z1 = 0.5 * (-c1 + (c1 * c1 - 4.0 * c0).sqrt());
    let p0 = (z1 - m1) / (z1 - z0);

    let mut threshold = 0usize;
    let mut sum = 0.0;
    for (i, &h) in histo.iter().enumerate() {
        sum += h;
        if sum > p0 {
            threshold = i;
            break;
        }
    }

    hist.midpoint(threshold)
}

/// `MomentsThresholdCalculator` + `MomentsThresholdImageFilter`: SimpleITK
/// defaults `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn moments_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = moments_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- RenyiEntropy -----------------------------------------------------

/// `RenyiEntropyThresholdCalculator::MaxEntropyThresholding`
/// (Shannon entropy, `alpha -> 1`): identical in structure to
/// [`maximum_entropy_threshold_value`]'s single criterion.
fn renyi_max_entropy_1(
    hist: &Histogram,
    cnh: &CumulativeNormalizedHistogram,
    first_bin: usize,
    last_bin: usize,
    size: usize,
) -> usize {
    let mut threshold = 0usize;
    let mut max_ent = f64::MIN_POSITIVE;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0;
        for ih in 0..=it {
            if hist.frequency(ih) != 0 {
                let x = cnh.norm_histo[ih] / cnh.p1[it];
                ent_back -= x * x.ln();
            }
        }
        let mut ent_obj = 0.0;
        for ih in (it + 1)..size {
            if hist.frequency(ih) != 0 {
                let x = cnh.norm_histo[ih] / cnh.p2[it];
                ent_obj -= x * x.ln();
            }
        }
        let tot_ent = ent_back + ent_obj;
        if max_ent < tot_ent {
            max_ent = tot_ent;
            threshold = it;
        }
    }
    threshold
}

/// `RenyiEntropyThresholdCalculator::MaxEntropyThresholding2` (Renyi entropy,
/// `alpha = 0.5`).
fn renyi_max_entropy_2(
    cnh: &CumulativeNormalizedHistogram,
    first_bin: usize,
    last_bin: usize,
    size: usize,
) -> usize {
    const TERM: f64 = 1.0 / (1.0 - 0.5);
    let mut threshold = 0usize;
    let mut max_ent = f64::MIN_POSITIVE;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0;
        for ih in 0..=it {
            ent_back += (cnh.norm_histo[ih] / cnh.p1[it]).sqrt();
        }
        let mut ent_obj = 0.0;
        for ih in (it + 1)..size {
            ent_obj += (cnh.norm_histo[ih] / cnh.p2[it]).sqrt();
        }
        let product = ent_back * ent_obj;
        let tot_ent = if product > 0.0 {
            TERM * product.ln()
        } else {
            0.0
        };
        if tot_ent > max_ent {
            max_ent = tot_ent;
            threshold = it;
        }
    }
    threshold
}

/// `RenyiEntropyThresholdCalculator::MaxEntropyThresholding3` (Renyi entropy,
/// `alpha = 2`). Unlike its two siblings, upstream seeds `max_ent` with
/// plain `0.0` rather than `NumericTraits<double>::min()`.
fn renyi_max_entropy_3(
    cnh: &CumulativeNormalizedHistogram,
    first_bin: usize,
    last_bin: usize,
    size: usize,
) -> usize {
    const TERM: f64 = 1.0 / (1.0 - 2.0);
    let mut threshold = 0usize;
    let mut max_ent = 0.0f64;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0;
        for ih in 0..=it {
            let x = cnh.norm_histo[ih] / cnh.p1[it];
            ent_back += x * x;
        }
        let mut ent_obj = 0.0;
        for ih in (it + 1)..size {
            let x = cnh.norm_histo[ih] / cnh.p2[it];
            ent_obj += x * x;
        }
        let product = ent_back * ent_obj;
        let tot_ent = if product > 0.0 {
            TERM * product.ln()
        } else {
            0.0
        };
        if tot_ent > max_ent {
            max_ent = tot_ent;
            threshold = it;
        }
    }
    threshold
}

/// `RenyiEntropyThresholdCalculator::GenerateData`
/// (`itkRenyiEntropyThresholdCalculator.hxx`): combines three independent
/// entropy-maximizing thresholds (Shannon, and Renyi at `alpha = 0.5, 2`)
/// into one, weighting each by how close together the three candidates land
/// (a single-bin histogram short-circuits to that bin's midpoint, matching
/// the calculator's own early return).
fn renyi_entropy_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    if size == 1 {
        return hist.midpoint(0);
    }

    let cnh = CumulativeNormalizedHistogram::new(hist);
    let (first_bin, last_bin) = cnh.first_last_bin();

    let t2 = renyi_max_entropy_1(hist, &cnh, first_bin, last_bin, size);
    let t1 = renyi_max_entropy_2(&cnh, first_bin, last_bin, size);
    let t3 = renyi_max_entropy_3(&cnh, first_bin, last_bin, size);

    let mut t_star = [t1, t2, t3];
    t_star.sort_unstable();
    let [t_star1, t_star2, t_star3] = t_star;

    let (beta1, beta2, beta3) = if t_star1.abs_diff(t_star2) <= 5 {
        if t_star2.abs_diff(t_star3) <= 5 {
            (1.0, 2.0, 1.0)
        } else {
            (0.0, 1.0, 3.0)
        }
    } else if t_star2.abs_diff(t_star3) <= 5 {
        (3.0, 1.0, 0.0)
    } else {
        (1.0, 2.0, 1.0)
    };

    let omega = cnh.p1[t_star3] - cnh.p1[t_star1];
    let real_opt_threshold = t_star1 as f64 * (cnh.p1[t_star1] + 0.25 * omega * beta1)
        + t_star2 as f64 * 0.25 * omega * beta2
        + t_star3 as f64 * (cnh.p2[t_star3] + 0.25 * omega * beta3);

    hist.midpoint(real_opt_threshold as usize)
}

/// `RenyiEntropyThresholdCalculator` + `RenyiEntropyThresholdImageFilter`:
/// SimpleITK defaults `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn renyi_entropy_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = renyi_entropy_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- Shanbhag -----------------------------------------------------------

/// `ShanbhagThresholdCalculator::GenerateData`
/// (`itkShanbhagThresholdCalculator.hxx`): minimizes the absolute difference
/// between background/object entropy sums built from the cumulative
/// normalized histogram — the same [`CumulativeNormalizedHistogram`]
/// scaffolding as [`maximum_entropy_threshold_value`], but a different
/// entropy formula and minimizing rather than maximizing.
fn shanbhag_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    let cnh = CumulativeNormalizedHistogram::new(hist);
    let (first_bin, last_bin) = cnh.first_last_bin();

    let mut threshold = 0usize;
    let mut min_ent = f64::MAX;
    for it in first_bin..=last_bin {
        let mut ent_back = 0.0;
        let term_back = 0.5 / cnh.p1[it];
        for ih in 1..=it {
            ent_back -= cnh.norm_histo[ih] * (1.0 - term_back * cnh.p1[ih - 1]).ln();
        }
        ent_back *= term_back;

        let mut ent_obj = 0.0;
        let term_obj = 0.5 / cnh.p2[it];
        for ih in (it + 1)..size {
            ent_obj -= cnh.norm_histo[ih] * (1.0 - term_obj * cnh.p2[ih]).ln();
        }
        ent_obj *= term_obj;

        let tot_ent = (ent_back - ent_obj).abs();
        if tot_ent < min_ent {
            min_ent = tot_ent;
            threshold = it;
        }
    }

    hist.midpoint(threshold)
}

/// `ShanbhagThresholdCalculator` + `ShanbhagThresholdImageFilter`: SimpleITK
/// defaults `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn shanbhag_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = shanbhag_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

// ---- Yen ----------------------------------------------------------------

/// `YenThresholdCalculator::GenerateData` (`itkYenThresholdCalculator.hxx`):
/// maximizes a criterion combining a Renyi (`alpha = 2`)-style "energy" term
/// over each side's normalized histogram with a binary-cross-entropy term on
/// the cumulative histogram. `itk::NumericTraits<double>::NonpositiveMin()`
/// is the most negative finite double, matching Rust's `f64::MIN` (unlike
/// [`maximum_entropy_threshold_value`]'s `NumericTraits<double>::min()`,
/// which is the smallest *positive* double).
fn yen_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    let total = hist.total_frequency() as f64;
    let norm_histo: Vec<f64> = (0..size)
        .map(|i| hist.frequency(i) as f64 / total)
        .collect();

    let mut p1 = vec![0.0f64; size];
    p1[0] = norm_histo[0];
    for i in 1..size {
        p1[i] = p1[i - 1] + norm_histo[i];
    }

    let mut p1_sq = vec![0.0f64; size];
    p1_sq[0] = norm_histo[0] * norm_histo[0];
    for i in 1..size {
        p1_sq[i] = p1_sq[i - 1] + norm_histo[i] * norm_histo[i];
    }

    let mut p2_sq = vec![0.0f64; size];
    for i in (0..size - 1).rev() {
        p2_sq[i] = p2_sq[i + 1] + norm_histo[i + 1] * norm_histo[i + 1];
    }

    let mut threshold = 0usize;
    let mut max_crit = f64::MIN;
    for it in 0..size {
        let term1 = if p1_sq[it] * p2_sq[it] > 0.0 {
            (p1_sq[it] * p2_sq[it]).ln()
        } else {
            0.0
        };
        let term2 = if p1[it] * (1.0 - p1[it]) > 0.0 {
            (p1[it] * (1.0 - p1[it])).ln()
        } else {
            0.0
        };
        let crit = -term1 + 2.0 * term2;
        if crit > max_crit {
            max_crit = crit;
            threshold = it;
        }
    }

    hist.midpoint(threshold)
}

/// `YenThresholdCalculator` + `YenThresholdImageFilter`: SimpleITK defaults
/// `number_of_histogram_bins` to 256.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn yen_threshold(
    img: &Image,
    inside_value: u8,
    outside_value: u8,
    number_of_histogram_bins: u32,
    mask: Option<&ThresholdMask>,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec()?;
    let hist = threshold_histogram(img, &vals, number_of_histogram_bins, mask)?;
    let threshold = yen_threshold_value(&hist);
    binarize_and_finish(img, &vals, threshold, inside_value, outside_value, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    fn img_f64(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn bimodal_image(low: f64, high: f64, n_each: usize) -> Image {
        let mut vals = vec![low; n_each];
        vals.extend(vec![high; n_each]);
        img_f64(&[2 * n_each, 1], vals)
    }

    // ---- ThresholdImageFilter ----

    #[test]
    fn threshold_clamps_outside_range_and_preserves_pixel_type() {
        let a = Image::from_vec(&[5, 1], vec![0.0f32, 5.0, 10.0, 15.0, 20.0]).unwrap();
        let out = threshold(&a, 5.0, 15.0, -1.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[-1.0, 5.0, 10.0, 15.0, -1.0]
        );
    }

    #[test]
    fn threshold_boundary_values_are_inclusive() {
        let a = img_f64(&[3, 1], vec![4.999, 5.0, 15.0]);
        let out = threshold(&a, 5.0, 15.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[0.0, 5.0, 15.0]);
    }

    // ---- shared inside/outside plumbing (representative: Huang) ----

    #[test]
    fn inside_outside_values_plumb_through() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (out, _) = huang_threshold(&a, 7, 3, 10, None).unwrap();
        let got = out.scalar_slice::<u8>().unwrap();
        assert!(got[..50].iter().all(|&v| v == 7));
        assert!(got[50..].iter().all(|&v| v == 3));
    }

    // ---- Huang ----

    #[test]
    fn huang_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = huang_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn huang_threshold_constant_image_returns_the_constant_value() {
        // first_bin == last_bin on a constant image (see crate::histogram),
        // so the search range is empty and best_threshold stays at its
        // ITK-initialized 0 -- but every bin's edges collapse to the same
        // value on a degenerate histogram (margin == 0), so bin 0's
        // midpoint equals the constant too.
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = huang_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    #[test]
    fn huang_threshold_bin_count_changes_result() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, t_coarse) = huang_threshold(&a, 1, 0, 4, None).unwrap();
        let (_, t_fine) = huang_threshold(&a, 1, 0, 100, None).unwrap();
        assert_ne!(t_coarse, t_fine);
    }

    // ---- Intermodes ----

    #[test]
    fn intermodes_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = intermodes_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn intermodes_threshold_constant_image_exceeds_smoothing_iterations() {
        // A single-spike histogram never becomes bimodal under repeated
        // 3-point-mean smoothing, so this hits ITK's own iteration cap.
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        assert!(matches!(
            intermodes_threshold(&a, 1, 0, 5, None),
            Err(FilterError::ThresholdCalculatorFailed {
                calculator: "Intermodes",
                ..
            })
        ));
    }

    #[test]
    fn intermodes_threshold_single_bin_returns_that_bin() {
        let a = img_f64(&[3, 1], vec![1.0, 2.0, 3.0]);
        let (_, threshold) = intermodes_threshold(&a, 1, 0, 1, None).unwrap();
        assert!(threshold.is_finite());
    }

    // ---- IsoData ----

    #[test]
    fn isodata_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = isodata_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn isodata_threshold_constant_image_falls_back_to_mean() {
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = isodata_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- KittlerIllingworth ----

    #[test]
    fn kittler_illingworth_threshold_separates_bimodal_clusters() {
        // Unlike the other calculators, KittlerIllingworth fits a Gaussian
        // per side and needs a non-zero within-cluster variance -- as
        // measured through the *histogram's bin midpoints*, not raw pixel
        // values. A pure two-delta-spike image (as `bimodal_image` builds)
        // makes `sigma2 == 0` exactly and correctly errors (see the
        // `..._threshold_value` doc comment); even a spread-out cluster
        // fails the same way if every one of its values still collapses
        // into a single bin. With 10 bins over [0, 100] (bin width ~10.01),
        // each cluster below spans at least two bins.
        let mut vals: Vec<f64> = (0..=10).map(|v| 2.0 * v as f64).cycle().take(50).collect();
        vals.extend((0..=10).map(|v| 80.0 + 2.0 * v as f64).cycle().take(50));
        let a = img_f64(&[100, 1], vals);
        let (_, threshold) = kittler_illingworth_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 20.0 && threshold < 80.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn kittler_illingworth_threshold_single_bin_returns_that_bin() {
        let a = img_f64(&[3, 1], vec![1.0, 2.0, 3.0]);
        let (_, threshold) = kittler_illingworth_threshold(&a, 1, 0, 1, None).unwrap();
        assert!(threshold.is_finite());
    }

    #[test]
    fn kittler_illingworth_threshold_constant_image_returns_the_constant_value() {
        // Mean() lands on the last (only-populated) bin via bin_index's
        // clip-to-last-bin rule, so As1 - At == 0 immediately and the loop
        // breaks non-fatally on its first iteration without updating
        // threshold away from that bin.
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = kittler_illingworth_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- Li ----

    #[test]
    fn li_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = li_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn li_threshold_constant_image_returns_the_constant_value() {
        // Every bin's edges collapse to the same value on a degenerate
        // histogram (margin == 0), so whichever bin the fixed-point search
        // lands on, its midpoint equals the constant.
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = li_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- MaximumEntropy ----

    #[test]
    fn maximum_entropy_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = maximum_entropy_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn maximum_entropy_threshold_constant_image_returns_the_constant_value() {
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = maximum_entropy_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- Moments ----

    #[test]
    fn moments_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = moments_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn moments_threshold_constant_image_returns_the_constant_value() {
        // cd == 0 for a constant image (every bin's GetMeasurement equals
        // the same constant) -> NaN throughout p0, but NaN comparisons are
        // always false, so threshold stays 0 -- whose midpoint is still the
        // constant on this degenerate histogram (margin == 0).
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = moments_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- RenyiEntropy ----

    #[test]
    fn renyi_entropy_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = renyi_entropy_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn renyi_entropy_threshold_single_bin_returns_that_bin() {
        let a = img_f64(&[3, 1], vec![1.0, 2.0, 3.0]);
        let (_, threshold) = renyi_entropy_threshold(&a, 1, 0, 1, None).unwrap();
        assert!(threshold.is_finite());
    }

    #[test]
    fn renyi_entropy_threshold_constant_image_returns_the_constant_value() {
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = renyi_entropy_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- Shanbhag ----

    #[test]
    fn shanbhag_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = shanbhag_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn shanbhag_threshold_constant_image_returns_the_constant_value() {
        // term_obj = 0.5 / P2[last_bin] divides by zero (+inf), and
        // ent_obj *= term_obj then yields 0 * inf = NaN, but the comparison
        // against min_ent is always false for NaN (no panic), so threshold
        // stays 0 -- whose midpoint is still the constant on this
        // degenerate histogram (margin == 0).
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = shanbhag_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- Yen ----

    #[test]
    fn yen_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (_, threshold) = yen_threshold(&a, 1, 0, 10, None).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
    }

    #[test]
    fn yen_threshold_constant_image_returns_the_constant_value() {
        let a = img_f64(&[4, 1], vec![9.0; 4]);
        let (_, threshold) = yen_threshold(&a, 1, 0, 5, None).unwrap();
        assert_eq!(threshold, 9.0);
    }

    // ---- bin-count / empty / zero-bin propagation (shared Histogram::from_values) ----

    #[test]
    fn zero_bins_errors_for_every_calculator() {
        let a = bimodal_image(0.0, 100.0, 5);
        assert!(matches!(
            huang_threshold(&a, 1, 0, 0, None),
            Err(FilterError::InvalidHistogramBins(0))
        ));
        assert!(matches!(
            yen_threshold(&a, 1, 0, 0, None),
            Err(FilterError::InvalidHistogramBins(0))
        ));
    }
}
