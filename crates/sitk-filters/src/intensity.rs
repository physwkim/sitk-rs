//! ITK's intensity-transform filters and automatic-thresholding filters.
//!
//! Verified against ITK's headers:
//!
//! - `Modules/Filtering/ImageIntensity/include/itkSigmoidImageFilter.h`
//! - `Modules/Filtering/ImageIntensity/include/itkIntensityWindowingImageFilter.h(.hxx)`
//! - `Modules/Filtering/ImageIntensity/include/itkInvertIntensityImageFilter.h`
//! - `Modules/Filtering/ImageIntensity/include/itkNormalizeImageFilter.h(.hxx)`,
//!   `itkShiftScaleImageFilter.h(.hxx)` (this crate's [`crate::statistics`] is
//!   already verified against `itkStatisticsImageFilter.hxx`'s sample
//!   variance, divisor `n - 1`, which `normalize` reuses for mean/sigma).
//! - `Modules/Numerics/Statistics/include/itkImageToHistogramFilter.h(.hxx)`,
//!   `itkHistogram.h(.hxx)`, `itkScalarImageToHistogramGenerator.hxx`,
//!   `itkSampleToHistogramFilter.hxx`
//! - `Modules/Filtering/Thresholding/include/itkOtsuThresholdCalculator.h(.hxx)`,
//!   `itkOtsuMultipleThresholdsCalculator.h(.hxx)`, `itkOtsuThresholdImageFilter.h`,
//!   `itkOtsuMultipleThresholdsImageFilter.h(.hxx)`,
//!   `itkHistogramThresholdImageFilter.h(.hxx)`, `itkThresholdLabelerImageFilter.h(.hxx)`,
//!   `itkTriangleThresholdCalculator.h(.hxx)`
//! - SimpleITK's generated-wrapper parameter defaults:
//!   `Code/BasicFilters/yaml/{Sigmoid,IntensityWindowing,InvertIntensity,Normalize,
//!   OtsuThreshold,OtsuMultipleThresholds,TriangleThreshold}ImageFilter.yaml`
//!
//! ## Histogram construction
//!
//! [`Histogram`] mirrors the `AutoMinimumMaximum` path shared by
//! `itk::Statistics::ImageToHistogramFilter` (used by
//! `OtsuThresholdImageFilter`/`TriangleThresholdImageFilter` via
//! `HistogramThresholdImageFilter`) and `itk::Statistics::SampleToHistogramFilter`
//! (used by `OtsuMultipleThresholdsImageFilter` via
//! `ScalarImageToHistogramGenerator`): both build a single-dimension histogram
//! of equal-width bins spanning `[min, max + margin]`, where `margin = (max -
//! min) / bins / marginalScale` (`marginalScale` defaults to 100 in both) is
//! added only to the upper bound, and both leave `ClipBinsAtEnds` at its
//! default `true`. `itkHistogram.hxx`'s `GetIndex` then assigns a value to the
//! bin whose half-open `[min, max)` contains it, except a value at or past the
//! very last bin's upper edge clips into that last bin. On a constant image
//! `margin == 0`, so every bin edge collapses to the same value and every
//! pixel clips into the *last* bin — not bin 0.
//!
//! ITK computes bin edges in `NumericTraits<T>::RealType`: `float` for a
//! `Float32` image via `ImageToHistogramFilter`, but `ScalarImageToHistogramGenerator`
//! hardcodes `Histogram<double>` regardless of pixel type — so the two
//! upstream Otsu filters do not even agree with each other on bin-edge
//! precision for `Float32` inputs. This port computes bin edges in `f64`
//! uniformly for both, which only differs from upstream in low-order bits for
//! `Float32` images.

use crate::error::{FilterError, Result};
use crate::functor;
use crate::functor::UnaryFunctor;
use sitk_core::{Image, PixelId};

// ---- Sigmoid ----------------------------------------------------------

/// `itkSigmoidImageFilter.h`'s `Functor::Sigmoid`: `f(x) = (max - min) /
/// (1 + exp(-(x - beta) / alpha)) + min`.
struct Sigmoid {
    alpha: f64,
    beta: f64,
    output_minimum: f64,
    output_maximum: f64,
}

impl UnaryFunctor for Sigmoid {
    fn apply(&self, x: f64) -> f64 {
        let e = 1.0 / (1.0 + (-(x - self.beta) / self.alpha).exp());
        (self.output_maximum - self.output_minimum) * e + self.output_minimum
    }
}

functor::unary_functor! {
    /// `SigmoidImageFilter` (`itkSigmoidImageFilter.h`): a linear
    /// transform of `x` through a logistic sigmoid, `f(x) = (output_maximum -
    /// output_minimum) * sigmoid((x - beta) / alpha) + output_minimum`.
    /// SimpleITK's defaults (`SigmoidImageFilter.yaml`) are `alpha = 1`,
    /// `beta = 0`, `output_minimum = 0`, `output_maximum = 255`.
    pub fn sigmoid, sigmoid_in_place(
        alpha: f64,
        beta: f64,
        output_minimum: f64,
        output_maximum: f64,
    ) = Sigmoid { alpha, beta, output_minimum, output_maximum };
}

// ---- IntensityWindowing ------------------------------------------------

/// `itkIntensityWindowingImageFilter.h`'s `Functor::IntensityWindowingTransform`,
/// with `scale`/`shift` precomputed once (`BeforeThreadedGenerateData` in the
/// `.hxx`) instead of per pixel.
struct IntensityWindowing {
    window_minimum: f64,
    window_maximum: f64,
    output_minimum: f64,
    output_maximum: f64,
    scale: f64,
    shift: f64,
}

impl IntensityWindowing {
    fn new(
        window_minimum: f64,
        window_maximum: f64,
        output_minimum: f64,
        output_maximum: f64,
    ) -> Self {
        let scale = (output_maximum - output_minimum) / (window_maximum - window_minimum);
        let shift = output_minimum - window_minimum * scale;
        Self {
            window_minimum,
            window_maximum,
            output_minimum,
            output_maximum,
            scale,
            shift,
        }
    }
}

impl UnaryFunctor for IntensityWindowing {
    fn apply(&self, x: f64) -> f64 {
        if x < self.window_minimum {
            self.output_minimum
        } else if x > self.window_maximum {
            self.output_maximum
        } else {
            x * self.scale + self.shift
        }
    }
}

functor::unary_functor! {
    /// `IntensityWindowingImageFilter` (`itkIntensityWindowingImageFilter.h`):
    /// linearly remaps `[window_minimum, window_maximum]` onto
    /// `[output_minimum, output_maximum]`; values outside the window clamp to
    /// the corresponding output bound. Unlike [`crate::rescale_intensity`],
    /// the window is caller-specified rather than the image's actual range.
    pub fn intensity_windowing, intensity_windowing_in_place(
        window_minimum: f64,
        window_maximum: f64,
        output_minimum: f64,
        output_maximum: f64,
    ) = IntensityWindowing::new(window_minimum, window_maximum, output_minimum, output_maximum);
}

// ---- InvertIntensity -----------------------------------------------------

/// `itkInvertIntensityImageFilter.h`'s `Functor::InvertIntensityTransform`:
/// `f(x) = maximum - x`.
struct InvertIntensity {
    maximum: f64,
}

impl UnaryFunctor for InvertIntensity {
    fn apply(&self, x: f64) -> f64 {
        self.maximum - x
    }
}

functor::unary_functor! {
    /// `InvertIntensityImageFilter` (`itkInvertIntensityImageFilter.h`):
    /// `f(x) = maximum - x`. ITK defaults `maximum` to the input pixel type's
    /// own max; this port has no implicit per-type default (consistent with
    /// this crate's other threshold/rescale functions), so callers pass it
    /// explicitly.
    pub fn invert_intensity, invert_intensity_in_place(maximum: f64) = InvertIntensity { maximum };
}

// ---- Normalize -------------------------------------------------------

/// `NumericTraits<T>::RealType` pixel-type mapping used by [`normalize`]:
/// stays `Float32` for a `Float32` input, promotes everything else (every
/// integer type, and `Float64` itself) to `Float64`.
fn real_type(id: PixelId) -> PixelId {
    match id {
        PixelId::Float32 => PixelId::Float32,
        _ => PixelId::Float64,
    }
}

/// `NormalizeImageFilter` (`itkNormalizeImageFilter.h(.hxx)`): shifts and
/// scales the image to zero mean and unit variance, `(x - mean) / sigma`,
/// reusing [`crate::statistics`] for `mean`/`sigma` and
/// `itkShiftScaleImageFilter.hxx`'s `(x + shift) * scale` evaluation order
/// (`shift = -mean`, `scale = 1 / sigma`). The output pixel type is the
/// input's `NumericTraits<T>::RealType` (see [`real_type`]), matching
/// `NormalizeImageFilter.yaml`'s `output_pixel_type` — unlike this crate's
/// other unary intensity filters, the output type is not the input's own.
///
/// On a constant image `sigma == 0`, so `scale` is `+inf` and every pixel
/// computes `(x - mean) * scale == 0.0 * inf == NaN`. ITK does not special-case
/// this either — `NormalizeImageFilter.hxx` divides by `GetSigma()`
/// unconditionally — and since the output pixel type is always
/// floating-point, storing `NaN` is well-defined in both languages, so this
/// port reproduces it as-is rather than guarding against it.
pub fn normalize(img: &Image) -> Result<Image> {
    let stats = crate::statistics(img)?;
    let shift = -stats.mean;
    let scale = 1.0 / stats.sigma;
    let vals: Vec<f64> = img
        .to_f64_vec()
        .iter()
        .map(|&v| (v + shift) * scale)
        .collect();
    crate::image_from_f64(real_type(img.pixel_id()), img.size(), img, &vals)
}

// ---- histogram (drives the Otsu / Triangle calculators) -------------------

/// See the module docs for the construction convention. Single dimension
/// only (this crate's images are scalar-pixel, so the 1-D case is all
/// [`crate::statistics`]-adjacent callers need).
struct Histogram {
    bin_min: Vec<f64>,
    bin_max: Vec<f64>,
    frequency: Vec<u64>,
    total: u64,
}

impl Histogram {
    fn from_values(vals: &[f64], bins: u32) -> Result<Self> {
        if bins == 0 {
            return Err(FilterError::InvalidHistogramBins(0));
        }
        if vals.is_empty() {
            return Err(FilterError::DegenerateRange);
        }
        let bins = bins as usize;

        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in vals {
            lo = lo.min(v);
            hi = hi.max(v);
        }

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
        for &v in vals {
            let idx = hist.bin_index(v);
            hist.frequency[idx] += 1;
        }
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
    fn bin_index(&self, v: f64) -> usize {
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

    fn bins(&self) -> usize {
        self.bin_min.len()
    }

    fn frequency(&self, i: usize) -> u64 {
        self.frequency[i]
    }

    fn total_frequency(&self) -> u64 {
        self.total
    }

    fn bin_max(&self, i: usize) -> f64 {
        self.bin_max[i]
    }

    /// `itkHistogram.hxx`'s `GetMeasurement`/`GetMeasurementVector`: a bin's
    /// centroid, `(min + max) / 2`.
    fn midpoint(&self, i: usize) -> f64 {
        (self.bin_min[i] + self.bin_max[i]) / 2.0
    }

    /// `itkHistogram.hxx`'s `Quantile`, specialized to one dimension: the
    /// value at cumulative-frequency proportion `p`, found by walking bins
    /// from the low end (`p < 0.5`) or the high end (`p >= 0.5`) and
    /// interpolating within the bin where the cumulative proportion crosses
    /// `p`.
    fn quantile(&self, p: f64) -> f64 {
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

    fn quantile_index(&self, p: f64) -> usize {
        self.bin_index(self.quantile(p))
    }
}

// ---- Otsu ---------------------------------------------------------------

/// `itk::Math::FloatAlmostEqual(x1, x2, maxUlps=1)`
/// (`Modules/Core/Common/include/itkMath.h`), specialized to the non-negative
/// `f64` domain of a between-class variance score: the general version
/// distinguishes positive/negative zero and negative magnitudes via
/// signed-magnitude bit tricks, which collapses to a plain integer-bitpattern
/// difference when both inputs are known `>= 0`, as `var_between` always is
/// here (a sum of `frequency * mean^2 / total` terms).
fn ulp1_almost_equal_nonneg(x1: f64, x2: f64) -> bool {
    let max_absolute_difference = 0.1 * f64::EPSILON;
    if (x1 - x2).abs() <= max_absolute_difference {
        return true;
    }
    let ulps = (x1.to_bits() as i64 - x2.to_bits() as i64).abs();
    ulps <= 1
}

/// `itk::OtsuMultipleThresholdsCalculator::IncrementThresholds`
/// (`itkOtsuMultipleThresholdsCalculator.hxx`): advances the threshold
/// configuration to the next one in enumeration order (rightmost movable cut
/// point first, each following cut point repacked immediately after it),
/// updating `class_freq`/`class_mean` incrementally rather than recomputing
/// them from scratch. Returns `false` once every configuration has been
/// visited.
fn increment_thresholds(
    hist: &Histogram,
    threshold_idx: &mut [usize],
    global_mean: f64,
    class_mean: &mut [f64],
    class_freq: &mut [f64],
) -> bool {
    let bins = hist.bins() as i64;
    let num_thresholds = threshold_idx.len();
    let num_classes = class_mean.len();
    let total = hist.total_frequency() as f64;

    for j in (0..num_thresholds).rev() {
        let bound = bins - 2 - (num_thresholds as i64 - 1 - j as i64);
        if (threshold_idx[j] as i64) < bound {
            threshold_idx[j] += 1;

            let mean_old = class_mean[j];
            let freq_old = class_freq[j];
            let f = hist.frequency(threshold_idx[j]) as f64;
            class_freq[j] += f;
            class_mean[j] = if class_freq[j] > 0.0 {
                (mean_old * freq_old + hist.midpoint(threshold_idx[j]) * f) / class_freq[j]
            } else {
                0.0
            };

            for k in (j + 1)..num_thresholds {
                threshold_idx[k] = threshold_idx[k - 1] + 1;
                class_freq[k] = hist.frequency(threshold_idx[k]) as f64;
                class_mean[k] = if class_freq[k] > 0.0 {
                    hist.midpoint(threshold_idx[k])
                } else {
                    0.0
                };
            }

            class_freq[num_classes - 1] = total;
            class_mean[num_classes - 1] = global_mean * total;
            for k in 0..num_classes - 1 {
                class_freq[num_classes - 1] -= class_freq[k];
                class_mean[num_classes - 1] -= class_mean[k] * class_freq[k];
            }
            class_mean[num_classes - 1] = if class_freq[num_classes - 1] > 0.0 {
                class_mean[num_classes - 1] / class_freq[num_classes - 1]
            } else {
                0.0
            };

            return true;
        }
        if j == 0 {
            return false;
        }
    }
    false
}

/// `itk::OtsuMultipleThresholdsCalculator::Compute()`
/// (`itkOtsuMultipleThresholdsCalculator.hxx`): an exhaustive search over
/// every placement of `num_thresholds` cut points (kept in ascending order,
/// enumerated via [`increment_thresholds`]) that maximizes
/// `sum_k class_frequency[k] * class_mean[k]^2 / total` — the between-class
/// variance up to an additive constant, per the upstream comment justifying
/// the simplification. Returns the winning bin index for each threshold, in
/// ascending order.
fn otsu_multiple_threshold_indices(
    hist: &Histogram,
    num_thresholds: usize,
    valley_emphasis: bool,
) -> Vec<usize> {
    let bins = hist.bins();
    let total = hist.total_frequency() as f64;
    let num_classes = num_thresholds + 1;

    let mut global_mean = 0.0;
    for j in 0..bins {
        global_mean += hist.midpoint(j) * hist.frequency(j) as f64;
    }
    global_mean /= total;

    let mut threshold_idx: Vec<usize> = (0..num_thresholds).collect();

    let mut class_freq = vec![0.0f64; num_classes];
    let mut freq_sum = 0.0;
    for j in 0..num_classes - 1 {
        class_freq[j] = hist.frequency(threshold_idx[j]) as f64;
        freq_sum += class_freq[j];
    }
    class_freq[num_classes - 1] = total - freq_sum;

    let img_pdf: Vec<f64> = (0..bins)
        .map(|j| hist.frequency(j) as f64 / total)
        .collect();

    let mut class_mean = vec![0.0f64; num_classes];
    let mut mean_sum = 0.0;
    for j in 0..num_classes - 1 {
        class_mean[j] = if class_freq[j] > 0.0 {
            hist.midpoint(threshold_idx[j])
        } else {
            0.0
        };
        mean_sum += class_mean[j] * class_freq[j];
    }
    class_mean[num_classes - 1] = if class_freq[num_classes - 1] > 0.0 {
        (global_mean * total - mean_sum) / class_freq[num_classes - 1]
    } else {
        0.0
    };

    let between_class_score = |freq: &[f64], mean: &[f64]| -> f64 {
        let mut v = 0.0;
        for j in 0..num_classes {
            v += freq[j] * mean[j] * mean[j];
        }
        v / total
    };

    let mut max_var_between = between_class_score(&class_freq, &class_mean);
    if valley_emphasis {
        // Faithfully-ported quirk of `itkOtsuMultipleThresholdsCalculator.hxx`'s
        // `Compute()`: this *initial* factor loop overwrites rather than
        // accumulates (`valleyEmphasisFactor = imgPDF[...]`, not `+=`), unlike
        // the equivalent loop inside the search below — so only the last
        // class boundary's PDF value survives here. Only observable with
        // `valley_emphasis` and more than one threshold.
        let mut factor = 0.0;
        for j in 0..num_classes - 1 {
            factor = img_pdf[threshold_idx[j]];
        }
        max_var_between *= 1.0 - factor;
    }

    let mut max_threshold_idx = threshold_idx.clone();

    while increment_thresholds(
        hist,
        &mut threshold_idx,
        global_mean,
        &mut class_mean,
        &mut class_freq,
    ) {
        let mut var_between = between_class_score(&class_freq, &class_mean);
        if valley_emphasis {
            let mut factor = 0.0;
            for j in 0..num_classes - 1 {
                factor += img_pdf[threshold_idx[j]];
            }
            var_between *= 1.0 - factor;
        }

        if var_between > max_var_between && !ulp1_almost_equal_nonneg(max_var_between, var_between)
        {
            max_var_between = var_between;
            max_threshold_idx = threshold_idx.clone();
        }
    }

    max_threshold_idx
}

fn require_bins_over_thresholds(bins: u32, thresholds: u32) -> Result<()> {
    if bins <= thresholds {
        return Err(FilterError::InvalidThresholdCount { bins, thresholds });
    }
    Ok(())
}

fn threshold_value(hist: &Histogram, idx: usize, return_bin_midpoint: bool) -> f64 {
    if return_bin_midpoint {
        hist.midpoint(idx)
    } else {
        hist.bin_max(idx)
    }
}

/// `OtsuThresholdCalculator` + `OtsuThresholdImageFilter`
/// (`itkOtsuThresholdCalculator.h(.hxx)`, `itkOtsuThresholdImageFilter.h`):
/// computes Otsu's threshold from a `number_of_histogram_bins`-bin histogram
/// of the image (128 is ITK's/SimpleITK's default,
/// `OtsuThresholdImageFilter.yaml`), then binarizes: pixels `<= threshold`
/// get `inside_value`, the rest get `outside_value` — matching
/// `itkHistogramThresholdImageFilter.hxx`'s internal `BinaryThresholdImageFilter`
/// call (`LowerThreshold = NonpositiveMin`, `UpperThreshold = computed
/// threshold`). `OtsuThresholdCalculator` always runs the underlying
/// `OtsuMultipleThresholdsCalculator` with a single threshold and
/// `ValleyEmphasis` left at its default `false` — it exposes no valley-emphasis
/// option of its own.
///
/// Returns the binarized image alongside the computed threshold value.
pub fn otsu_threshold(
    img: &Image,
    number_of_histogram_bins: u32,
    return_bin_midpoint: bool,
    inside_value: u8,
    outside_value: u8,
) -> Result<(Image, f64)> {
    require_bins_over_thresholds(number_of_histogram_bins, 1)?;
    let vals = img.to_f64_vec();
    let hist = Histogram::from_values(&vals, number_of_histogram_bins)?;
    let idx = otsu_multiple_threshold_indices(&hist, 1, false)[0];
    let threshold = threshold_value(&hist, idx, return_bin_midpoint);

    let out: Vec<u8> = vals
        .iter()
        .map(|&v| {
            if v <= threshold {
                inside_value
            } else {
                outside_value
            }
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok((result, threshold))
}

/// `OtsuMultipleThresholdsCalculator` + `OtsuMultipleThresholdsImageFilter`
/// (`itkOtsuMultipleThresholdsCalculator.h(.hxx)`,
/// `itkOtsuMultipleThresholdsImageFilter.h(.hxx)`): computes
/// `number_of_thresholds` Otsu thresholds and labels each pixel with the
/// index of the class its value falls into, offset by `label_offset` —
/// `itkThresholdLabelerImageFilter.h`'s `Functor::ThresholdLabeler`: the
/// lowest class index `k` such that `pixel <= thresholds[k]` (values above
/// every threshold get the highest class, `thresholds.len()`). Output pixel
/// type is `UInt8` (`OtsuMultipleThresholdsImageFilter.yaml`'s
/// `output_pixel_type: uint8_t`, unconditional — unlike [`otsu_threshold`]'s
/// `inside_value`/`outside_value`, ITK does not let the caller pick label
/// values here beyond the additive `label_offset`).
///
/// Returns the labeled image alongside the computed thresholds, ascending.
pub fn otsu_multiple_thresholds(
    img: &Image,
    number_of_thresholds: u32,
    number_of_histogram_bins: u32,
    valley_emphasis: bool,
    return_bin_midpoint: bool,
    label_offset: u8,
) -> Result<(Image, Vec<f64>)> {
    if number_of_thresholds == 0 {
        return Err(FilterError::InvalidThresholdCount {
            bins: number_of_histogram_bins,
            thresholds: 0,
        });
    }
    require_bins_over_thresholds(number_of_histogram_bins, number_of_thresholds)?;

    let vals = img.to_f64_vec();
    let hist = Histogram::from_values(&vals, number_of_histogram_bins)?;
    let indices =
        otsu_multiple_threshold_indices(&hist, number_of_thresholds as usize, valley_emphasis);
    let thresholds: Vec<f64> = indices
        .iter()
        .map(|&i| threshold_value(&hist, i, return_bin_midpoint))
        .collect();

    let out: Vec<u8> = vals
        .iter()
        .map(|&v| {
            let bucket = thresholds
                .iter()
                .position(|&t| v <= t)
                .unwrap_or(thresholds.len());
            (bucket as u8).wrapping_add(label_offset)
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok((result, thresholds))
}

// ---- Triangle -------------------------------------------------------------

/// `std::max_element`'s tie-break for a non-empty range: the index of the
/// *first* maximum. Rust's `Iterator::max_by`/`max_by_key` keep the *last*
/// on ties, so this can't reuse them directly. Returns `0` for an empty
/// slice (a degenerate-histogram guard this port adds; see
/// [`triangle_threshold_value`]).
fn argmax_first(vals: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in vals.iter().enumerate().skip(1) {
        if v > vals[best] {
            best = i;
        }
    }
    best
}

/// `itk::TriangleThresholdCalculator::GenerateData()`
/// (`itkTriangleThresholdCalculator.hxx`): draws a line from the histogram's
/// peak bin to the further (in bin-index distance) of its 1st/99th
/// percentile bin, then picks the bin immediately after whichever bin
/// maximizes the gap between that line and the histogram.
///
/// ITK's `GenerateData` sets the output for a single-bin histogram
/// (`histogram->GetSize(0) == 1`) but has no early return after doing so, so
/// it falls through into the general search and ultimately indexes one past
/// the last bin — undefined behavior in C++. This port returns early with
/// that same single-bin value instead, since that's what the guarded branch
/// computes before the fall-through; there is no defined "port target" for
/// the UB it falls into. The final index is likewise clamped to the last bin
/// as a safety net for any other histogram degenerate enough to put the peak
/// or percentile bins on top of each other.
fn triangle_threshold_value(hist: &Histogram) -> f64 {
    let size = hist.bins();
    if size <= 1 {
        return hist.midpoint(0);
    }

    let mut mx = f64::MIN_POSITIVE;
    let mut mx_idx = 0usize;
    for j in 0..size {
        let f = hist.frequency(j) as f64;
        if f > mx {
            mx_idx = j;
            mx = f;
        }
    }

    let one_pc_idx = hist.quantile_index(0.01);
    let nn_pc_idx = hist.quantile_index(0.99);

    let mut triangle = vec![0.0f64; size];
    let thresh_idx =
        if (mx_idx as f64 - one_pc_idx as f64).abs() > (mx_idx as f64 - nn_pc_idx as f64).abs() {
            let slope = mx / (mx_idx - one_pc_idx) as f64;
            for (k, t) in triangle
                .iter_mut()
                .enumerate()
                .take(mx_idx)
                .skip(one_pc_idx)
            {
                let line = slope * (k - one_pc_idx) as f64;
                *t = line - hist.frequency(k) as f64;
            }
            one_pc_idx + argmax_first(&triangle[one_pc_idx..mx_idx])
        } else {
            let slope = -mx / (nn_pc_idx - mx_idx) as f64;
            for (k, t) in triangle.iter_mut().enumerate().take(nn_pc_idx).skip(mx_idx) {
                let line = slope * (k - mx_idx) as f64 + mx;
                *t = line - hist.frequency(k) as f64;
            }
            mx_idx + argmax_first(&triangle[mx_idx..nn_pc_idx])
        };

    hist.midpoint((thresh_idx + 1).min(size - 1))
}

/// `TriangleThresholdCalculator` + `TriangleThresholdImageFilter`
/// (`itkTriangleThresholdCalculator.h(.hxx)`, `itkTriangleThresholdImageFilter.h`):
/// same `<= threshold` binarization convention as [`otsu_threshold`]
/// (`itkHistogramThresholdImageFilter.hxx`). `number_of_histogram_bins`
/// defaults to 256 in SimpleITK (`TriangleThresholdImageFilter.yaml`, vs.
/// Otsu's 128).
///
/// Returns the binarized image alongside the computed threshold value.
pub fn triangle_threshold(
    img: &Image,
    number_of_histogram_bins: u32,
    inside_value: u8,
    outside_value: u8,
) -> Result<(Image, f64)> {
    let vals = img.to_f64_vec();
    let hist = Histogram::from_values(&vals, number_of_histogram_bins)?;
    let threshold = triangle_threshold_value(&hist);

    let out: Vec<u8> = vals
        .iter()
        .map(|&v| {
            if v <= threshold {
                inside_value
            } else {
                outside_value
            }
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok((result, threshold))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_f64(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- Sigmoid ----

    #[test]
    fn sigmoid_matches_hand_formula() {
        let a = img_f64(&[3, 1], vec![-10.0, 0.0, 10.0]);
        let out = sigmoid(&a, 2.0, 0.0, 0.0, 255.0).unwrap();
        let expected: Vec<f64> = [-10.0, 0.0, 10.0]
            .iter()
            .map(|&x| {
                let e = 1.0 / (1.0 + (-x / 2.0f64).exp());
                255.0 * e
            })
            .collect();
        let got = out.scalar_slice::<f64>().unwrap();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-9, "{g} vs {e}");
        }
        // x == beta is the sigmoid's midpoint: exactly (max-min)/2 + min.
        assert!((got[1] - 127.5).abs() < 1e-9);
    }

    #[test]
    fn sigmoid_in_place_matches_allocating() {
        let a = img_f64(&[2, 1], vec![-5.0, 5.0]);
        let allocated = sigmoid(&a, 1.0, 0.0, 0.0, 255.0).unwrap();
        let in_place = sigmoid_in_place(a, 1.0, 0.0, 0.0, 255.0).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- IntensityWindowing ----

    #[test]
    fn intensity_windowing_clamps_outside_window_and_maps_linearly_inside() {
        let a = img_f64(&[5, 1], vec![-5.0, 0.0, 50.0, 100.0, 200.0]);
        let out = intensity_windowing(&a, 0.0, 100.0, 0.0, 255.0).unwrap();
        let got = out.scalar_slice::<f64>().unwrap();
        assert_eq!(got[0], 0.0); // below window -> output_minimum
        assert_eq!(got[1], 0.0); // at window_minimum
        assert!((got[2] - 127.5).abs() < 1e-9); // window midpoint
        assert!((got[3] - 255.0).abs() < 1e-9); // at window_maximum (computed via scale/shift, not clamped)
        assert_eq!(got[4], 255.0); // above window -> output_maximum (clamped exactly)
    }

    // ---- InvertIntensity ----

    #[test]
    fn invert_intensity_subtracts_from_maximum() {
        let a = img_f64(&[3, 1], vec![0.0, 100.0, 255.0]);
        let out = invert_intensity(&a, 255.0).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[255.0, 155.0, 0.0]);
    }

    // ---- Normalize ----

    #[test]
    fn normalize_zero_mean_unit_variance() {
        let a = img_f64(&[4, 1], vec![2.0, 4.0, 4.0, 6.0]);
        let out = normalize(&a).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        let got = out.scalar_slice::<f64>().unwrap();
        let mean: f64 = got.iter().sum::<f64>() / got.len() as f64;
        assert!(mean.abs() < 1e-9, "mean {mean}");
        let variance: f64 =
            got.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (got.len() as f64 - 1.0);
        assert!((variance - 1.0).abs() < 1e-9, "variance {variance}");
    }

    #[test]
    fn normalize_promotes_float32_stays_float32() {
        let a = Image::from_vec(&[2, 1], vec![1.0f32, 3.0]).unwrap();
        let out = normalize(&a).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn normalize_constant_image_is_nan_everywhere() {
        // sigma == 0 on a constant image -> (x - mean) * (1/sigma) == 0 * inf == NaN,
        // matching ITK's unguarded division (see the `normalize` doc comment).
        let a = img_f64(&[3, 1], vec![7.0, 7.0, 7.0]);
        let out = normalize(&a).unwrap();
        assert!(
            out.scalar_slice::<f64>()
                .unwrap()
                .iter()
                .all(|v| v.is_nan())
        );
    }

    // ---- Histogram ----

    #[test]
    fn histogram_constant_image_collapses_into_last_bin() {
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
    fn histogram_single_bin_holds_the_whole_range() {
        let vals = vec![0.0, 5.0, 10.0];
        let hist = Histogram::from_values(&vals, 1).unwrap();
        assert_eq!(hist.bins(), 1);
        assert_eq!(hist.frequency(0), 3);
        assert_eq!(hist.total_frequency(), 3);
    }

    #[test]
    fn histogram_zero_bins_errors() {
        let vals = vec![0.0, 1.0];
        assert!(matches!(
            Histogram::from_values(&vals, 0),
            Err(FilterError::InvalidHistogramBins(0))
        ));
    }

    #[test]
    fn histogram_empty_image_errors() {
        assert!(matches!(
            Histogram::from_values(&[], 4),
            Err(FilterError::DegenerateRange)
        ));
    }

    #[test]
    fn histogram_bimodal_split_across_bins() {
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

    // ---- Otsu ----

    fn bimodal_image(low: f64, high: f64, n_each: usize) -> Image {
        let mut vals = vec![low; n_each];
        vals.extend(vec![high; n_each]);
        img_f64(&[2 * n_each, 1], vals)
    }

    #[test]
    fn otsu_threshold_constant_image_returns_the_constant_value() {
        let a = img_f64(&[4, 1], vec![7.0; 4]);
        let (out, threshold) = otsu_threshold(&a, 8, false, 1, 0).unwrap();
        assert_eq!(threshold, 7.0);
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn otsu_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (out, threshold) = otsu_threshold(&a, 10, false, 1, 0).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
        let got = out.scalar_slice::<u8>().unwrap();
        assert!(got[..50].iter().all(|&v| v == 1));
        assert!(got[50..].iter().all(|&v| v == 0));
    }

    #[test]
    fn otsu_threshold_needs_more_than_one_bin() {
        let a = bimodal_image(0.0, 100.0, 5);
        assert!(matches!(
            otsu_threshold(&a, 1, false, 1, 0),
            Err(FilterError::InvalidThresholdCount { .. })
        ));
    }

    #[test]
    fn otsu_multiple_thresholds_separates_three_clusters() {
        let mut vals = vec![0.0; 20];
        vals.extend(vec![50.0; 20]);
        vals.extend(vec![100.0; 20]);
        let a = img_f64(&[60, 1], vals);
        let (out, thresholds) = otsu_multiple_thresholds(&a, 2, 30, false, false, 0).unwrap();
        assert_eq!(thresholds.len(), 2);
        assert!(thresholds[0] < thresholds[1]);
        let got = out.scalar_slice::<u8>().unwrap();
        assert!(got[..20].iter().all(|&v| v == 0));
        assert!(got[20..40].iter().all(|&v| v == 1));
        assert!(got[40..].iter().all(|&v| v == 2));
    }

    #[test]
    fn otsu_multiple_thresholds_applies_label_offset() {
        let a = bimodal_image(0.0, 100.0, 10);
        let (out, _) = otsu_multiple_thresholds(&a, 1, 10, false, false, 5).unwrap();
        let got = out.scalar_slice::<u8>().unwrap();
        assert!(got[..10].iter().all(|&v| v == 5));
        assert!(got[10..].iter().all(|&v| v == 6));
    }

    #[test]
    fn otsu_multiple_thresholds_needs_bins_over_thresholds() {
        let a = bimodal_image(0.0, 100.0, 5);
        assert!(matches!(
            otsu_multiple_thresholds(&a, 3, 3, false, false, 0),
            Err(FilterError::InvalidThresholdCount { .. })
        ));
    }

    // ---- Triangle ----

    #[test]
    fn triangle_threshold_separates_bimodal_deltas() {
        let a = bimodal_image(0.0, 100.0, 50);
        let (out, threshold) = triangle_threshold(&a, 10, 1, 0).unwrap();
        assert!(
            threshold > 0.0 && threshold < 100.0,
            "threshold {threshold}"
        );
        let got = out.scalar_slice::<u8>().unwrap();
        assert!(got[..50].iter().all(|&v| v == 1));
        assert!(got[50..].iter().all(|&v| v == 0));
    }

    #[test]
    fn triangle_threshold_single_bin_does_not_panic() {
        let a = img_f64(&[3, 1], vec![1.0, 2.0, 3.0]);
        let (_, threshold) = triangle_threshold(&a, 1, 1, 0).unwrap();
        assert!(threshold.is_finite());
    }

    #[test]
    fn triangle_threshold_constant_image() {
        let a = img_f64(&[3, 1], vec![9.0, 9.0, 9.0]);
        let (out, threshold) = triangle_threshold(&a, 5, 1, 0).unwrap();
        assert_eq!(threshold, 9.0);
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 1]);
    }
}
