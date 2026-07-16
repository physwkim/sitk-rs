//! `LabelIntensityStatisticsImageFilter`: shape *and* intensity attributes of
//! every label.
//!
//! SimpleITK's `LabelIntensityStatisticsImageFilter` wraps
//! `itk::LabelImageToStatisticsLabelMapFilter`, which chains
//! `LabelImageToLabelMapFilter` → `StatisticsLabelMapFilter`. The latter derives
//! from `ShapeLabelMapFilter`, and its `ThreadedProcessLabelObject`
//! (`itkStatisticsLabelMapFilter.hxx:53-307`) begins by calling the shape
//! filter's. So every measurement of [`crate::filters::label_shape::label_shape_statistics`]
//! is also a measurement here — [`IntensityStatistics::shape`] holds them —
//! with the intensity attributes computed on top, against a second *feature*
//! image.
//!
//! The oriented bounding box is the one exception: `ShapeLabelMapFilter` can
//! compute it, but `LabelIntensityStatisticsImageFilter.yaml` exposes neither
//! the `ComputeOrientedBoundingBox` member nor an `OrientedBoundingBox`
//! measurement, so it stays off.
//!
//! ## The histogram, and where the median comes from
//!
//! `Median` is read off a 1-D `itk::Statistics::Histogram<double>` of
//! `number_of_bins` bins spanning the **whole feature image's** minimum and
//! maximum (`MinimumMaximumImageCalculator` in `BeforeThreadedGenerateData`,
//! `.hxx:35-50`) — not the per-label range. It is therefore quantized to a bin
//! centre, and the yaml says so: "the histogram is used to compute the median
//! value, and […] this option may have an effect on the value of the median".
//!
//! Three details of `itk::Statistics::Histogram` that the median inherits:
//!
//! - **the bin width is computed in `float`**, not `double`
//!   (`itkHistogram.hxx:214-238`, lines 225-226: `float interval = (float(upper) -
//!   float(lower)) / float(size)`), and every bin bound but the last is
//!   `double(lower) + float(j) * interval`. `histogram_bins` reproduces the
//!   narrowing exactly.
//! - `SetClipBinsAtEnds(false)`, so a value below the first bin lands in bin 0
//!   and a value at or above the last bin's max lands in the last bin
//!   (`itkHistogram.hxx:255-290`). With the bounds taken from the image's own
//!   min/max, only the maximum-valued pixels take the second path.
//! - `GetMeasurementVector(i)` returns the bin **centre** `(min_i + max_i) / 2`
//!   (`itkHistogram.hxx:462-472`), which is the value the median reports.
//!
//! ### The integer-padding branch
//!
//! When the feature pixel type is integral, at most 2 bytes wide, and
//! `number_of_bins` is exactly `1 << (8 * sizeof)`, ITK spans
//! `[NumericTraits<T>::min() - 0.5, NumericTraits<T>::max() + 0.5]` instead of
//! the image's own range, "so the center of bins are integers"
//! (`itkStatisticsLabelMapFilter.hxx:74-85`). ITK's *default* `NumberOfBins`
//! is that same `1 << (8 * sizeof)` for those types
//! (`itkStatisticsLabelMapFilter.h:136-140`), so the branch is ITK's normal
//! path for `UInt8`/`Int8`/`UInt16`/`Int16` feature images. SimpleITK's yaml
//! overrides the default to `128` for every type, so through SimpleITK the
//! branch is only reached by explicitly asking for 256 bins on an 8-bit feature
//! image (or 65536 on a 16-bit one). [`LabelIntensityStatisticsSettings::default`]
//! follows the yaml.
//!
//! ## Divergences and quirks
//!
//! - **`minimum_index` / `maximum_index` report the *last* extremum.** The
//!   updates are `if (v <= min)` and `if (v >= max)` (`.hxx:117-127`), so on a
//!   tie the later pixel in raster order overwrites the earlier. Reproduced.
//! - **the weighted moments include the pixel's own second moment; the shape
//!   moments do not.** `StatisticsLabelMapFilter` adds `spacing[i]^2 / 12` to
//!   each diagonal entry of the central moments, commented "the normalized
//!   second order central moment of a pixel" (`.hxx:224-228`).
//!   `ShapeLabelMapFilter` accumulates a raw discrete second moment and adds no
//!   such term (`itkShapeLabelMapFilter.hxx:234-244`, `:260-276`). So even for a
//!   *constant* feature image [`IntensityStatistics::weighted_principal_moments`]
//!   does not equal [`ShapeStatistics::principal_moments`] — under unit isotropic
//!   spacing it exceeds it by `1/12` in every eigenvalue. Both are reproduced as
//!   written; pinned by
//!   `a_uniformly_weighted_object_exceeds_the_shape_moments_by_the_pixels_own`.
//! - **the weighted principal moments are clamped and the ratios are
//!   sign-checked (Fixed in this port).** Upstream `StatisticsLabelMapFilter`
//!   neither clamps the eigenvalues nor checks the ratio's sign (`.hxx:231-237`,
//!   `:261-267`), so a signed or float feature image — where the "mass" `sum`
//!   can be negative and the division by it flips the sign of the covariance,
//!   making it indefinite — leaves `weighted_elongation` / `weighted_flatness`
//!   `NaN`. This port mirrors `ShapeLabelMapFilter` instead: each eigenvalue is
//!   clamped with `.max(0.0)` (`itkShapeLabelMapFilter.hxx:315`) and each square
//!   root is taken only when its ratio is positive
//!   (`itkShapeLabelMapFilter.hxx:344-361`), so both ratios stay finite (`0` for
//!   a non-positive-definite object).
//! - **`weighted_flatness` and `weighted_elongation` are guarded separately
//!   (Fixed in this port).** Upstream gates both on `principalMoments[0]` being
//!   non-zero (`.hxx:261-267`); this port gates `flatness` on
//!   `principalMoments[0]` and `elongation` separately on
//!   `principalMoments[dim-2]` (`itkShapeLabelMapFilter.hxx:344`/`:353`), so a
//!   zero `principalMoments[0]` no longer suppresses a real `weighted_elongation`.
//! - `if constexpr (ImageDimension < 2) { elongation = 1; flatness = 1; }`
//!   (`.hxx:256-260`) is unreachable through SimpleITK, which instantiates 2-D
//!   and 3-D only; [`label_shape_statistics`] rejects any other dimension
//!   before this code runs, so the branch is not ported.
//! - `sum3` / `sum4` accumulate `std::pow(v, 3)` / `std::pow(v, 4)`; this port
//!   calls `f64::powf` with the same exponents, which is the same libm entry
//!   point C++'s `std::pow(double, int)` promotes to.

use std::collections::BTreeMap;

use crate::core::{Image, LabelMap, PixelId};

use crate::filters::error::{FilterError, Result};
use crate::filters::geometry::require_same_physical_space;
use crate::filters::label_shape::{
    LabelShapeStatisticsSettings, ShapeStatistics, determinant, index_to_physical, is_almost_zero,
    label_shape_statistics,
};
use crate::filters::linalg::{MAX_DIM, Mat, symmetric_eigen};

/// The intensity attributes `StatisticsLabelObject` carries, plus the
/// `ShapeLabelObject` attributes it inherits.
#[derive(Clone, Debug, PartialEq)]
pub struct IntensityStatistics {
    /// Everything `ShapeLabelMapFilter` computed first.
    /// [`ShapeStatistics::oriented_bounding_box`] is always `None` here.
    pub shape: ShapeStatistics,
    /// Smallest feature value over the object's pixels.
    pub minimum: f64,
    /// Largest feature value over the object's pixels.
    pub maximum: f64,
    /// Index of the **last** pixel in raster order attaining [`Self::minimum`].
    pub minimum_index: Vec<i64>,
    /// Index of the **last** pixel in raster order attaining [`Self::maximum`].
    pub maximum_index: Vec<i64>,
    /// Σ v.
    pub sum: f64,
    /// `sum / number_of_pixels`.
    pub mean: f64,
    /// The bin centre of the histogram's median bin — see the [module docs](self).
    pub median: f64,
    /// The *sample* variance (`n - 1` denominator); `0` for a single pixel.
    pub variance: f64,
    /// `sqrt(variance)`.
    pub standard_deviation: f64,
    /// `0` when `|variance * sigma| <= f64::MIN_POSITIVE`.
    pub skewness: f64,
    /// Excess kurtosis (`- 3.0` applied); `0` when `|variance| <= f64::MIN_POSITIVE`.
    pub kurtosis: f64,
    /// The intensity-weighted centroid, in physical space. Zeroed when `sum` is
    /// (almost) zero.
    pub center_of_gravity: Vec<f64>,
    /// Eigenvalues of the intensity-weighted second-order central moments,
    /// ascending. Each is clamped to `>= 0` with `.max(0.0)` (mirroring
    /// `ShapeLabelMapFilter`, `itkShapeLabelMapFilter.hxx:315`), since a signed
    /// or float feature image can make the weighted covariance indefinite and a
    /// negative eigenvalue is meaningless as a second-order moment — see the
    /// [module docs](self) (§2.65 fix).
    pub weighted_principal_moments: Vec<f64>,
    /// Row-major `dim × dim`; row `i` is the eigenvector for
    /// `weighted_principal_moments[i]`, last row sign-flipped to make the matrix
    /// a proper rotation.
    pub weighted_principal_axes: Vec<f64>,
    /// `sqrt(pm[dim-1] / pm[dim-2])`, or `0` when `pm[0]` is (almost) zero.
    pub weighted_elongation: f64,
    /// `sqrt(pm[1] / pm[0])`, or `0` when `pm[0]` is (almost) zero.
    pub weighted_flatness: f64,
}

/// The four settings SimpleITK's `LabelIntensityStatisticsImageFilter` exposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelIntensityStatisticsSettings {
    /// Pixel value of the label image that is *not* part of any object.
    pub background_value: f64,
    /// Off by default, "because of the high computation time required".
    pub compute_feret_diameter: bool,
    /// On by default.
    pub compute_perimeter: bool,
    /// Bins of the median's histogram. SimpleITK's default is `128` for every
    /// feature pixel type; ITK's own is `1 << (8 * sizeof)` for the integral
    /// types up to 2 bytes wide.
    pub number_of_bins: u32,
}

impl Default for LabelIntensityStatisticsSettings {
    fn default() -> Self {
        Self {
            background_value: 0.0,
            compute_feret_diameter: false,
            compute_perimeter: true,
            number_of_bins: 128,
        }
    }
}

/// `itk::Statistics::Histogram::Initialize(size, lower, upper)`
/// (`itkHistogram.hxx:213-238`): the `(min, max)` bound of each of `n` bins.
///
/// `interval` is a `float` upstream, and every bound but the last is
/// `double(lower) + float(j) * interval`. The `float` narrowing is not an
/// accident of the pixel type — `lower` and `upper` are already `double` there —
/// so it is reproduced rather than widened.
fn histogram_bins(lower: f64, upper: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let interval = (upper as f32 - lower as f32) / n as f32;
    let mut mins = Vec::with_capacity(n);
    let mut maxs = Vec::with_capacity(n);
    for j in 0..n - 1 {
        mins.push(lower + (j as f32 * interval) as f64);
        maxs.push(lower + ((j as f32 + 1.0) * interval) as f64);
    }
    mins.push(lower + ((n as f32 - 1.0) * interval) as f64);
    maxs.push(upper);
    (mins, maxs)
}

/// `itk::Statistics::Histogram::GetIndex` with `ClipBinsAtEnds` off
/// (`itkHistogram.hxx:243-323`): the bin `j` with `mins[j] <= v < maxs[j]`,
/// saturating at both ends.
fn bin_index(mins: &[f64], maxs: &[f64], v: f64) -> usize {
    let last = mins.len() - 1;
    if v < mins[0] {
        0
    } else if v >= maxs[last] {
        last
    } else {
        mins.partition_point(|&m| m <= v) - 1
    }
}

/// The histogram bounds `StatisticsLabelMapFilter::ThreadedProcessLabelObject`
/// picks (`itkStatisticsLabelMapFilter.hxx:74-85`): the feature image's own
/// min/max, or a half-pixel-padded type range on the integer-bin branch.
fn histogram_bounds(feature_id: PixelId, number_of_bins: u32, min: f64, max: f64) -> (f64, f64) {
    let integral_and_narrow = feature_id.is_integer_scalar() && feature_id.size_in_bytes() <= 2;
    if integral_and_narrow && let Some((type_min, type_max)) = feature_id.integer_scalar_bounds() {
        let bits_shift = 8 * feature_id.size_in_bytes();
        if u64::from(number_of_bins) == 1u64 << bits_shift {
            return (type_min as f64 - 0.5, type_max as f64 + 0.5);
        }
    }
    (min, max)
}

/// Compute the shape and intensity attributes of every label in `label_image`,
/// reading intensities from `feature_image`.
///
/// `label_image` must have an integer pixel type (`IntegerLabelPixelIDTypeList`)
/// and be 2-D or 3-D; `feature_image` must be scalar (`BasicPixelIDTypeList`)
/// and have the same size. The returned map is keyed by label value, ascending.
pub fn label_intensity_statistics(
    label_image: &Image,
    feature_image: &Image,
    settings: &LabelIntensityStatisticsSettings,
) -> Result<BTreeMap<i64, IntensityStatistics>> {
    if !label_image.pixel_id().is_integer_scalar() {
        return Err(FilterError::RequiresIntegerPixelType(
            label_image.pixel_id(),
        ));
    }
    if label_image.size() != feature_image.size() {
        return Err(FilterError::SizeMismatch {
            a: label_image.size().to_vec(),
            b: feature_image.size().to_vec(),
        });
    }
    require_same_physical_space(label_image, feature_image, 1)?;
    if settings.number_of_bins == 0 {
        return Err(FilterError::InvalidHistogramBins(settings.number_of_bins));
    }
    let features = feature_image.to_f64_vec()?;

    let shapes = label_shape_statistics(
        label_image,
        &LabelShapeStatisticsSettings {
            background_value: settings.background_value,
            compute_feret_diameter: settings.compute_feret_diameter,
            compute_perimeter: settings.compute_perimeter,
            compute_oriented_bounding_box: false,
        },
    )?;

    let dim = label_image.dimension();
    let size = label_image.size();
    let spacing = label_image.spacing();
    let origin = label_image.origin();
    let direction = label_image.direction();

    // `MinimumMaximumImageCalculator` over the whole feature image.
    let (image_min, image_max) = features
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let n_bins = settings.number_of_bins as usize;
    let (lower, upper) = histogram_bounds(
        feature_image.pixel_id(),
        settings.number_of_bins,
        image_min,
        image_max,
    );
    let (bin_mins, bin_maxs) = histogram_bins(lower, upper, n_bins);

    let pass = StatisticsPass {
        features: &features,
        size,
        dim,
        spacing,
        origin,
        direction,
        bin_mins: &bin_mins,
        bin_maxs: &bin_maxs,
    };

    let label_map = LabelMap::from_label_image(label_image, settings.background_value as i64)?;
    let mut out = BTreeMap::new();
    for object in label_map.label_objects() {
        let shape = shapes
            .get(&object.label())
            .expect("label_shape_statistics saw the same labels")
            .clone();
        out.insert(
            object.label(),
            pass.run(&object.indices().collect::<Vec<_>>(), shape),
        );
    }
    Ok(out)
}

/// Everything `ThreadedProcessLabelObject` reads that is the same for every
/// label object: the feature buffer, the label map's geometry, and the histogram
/// bin bounds that `BeforeThreadedGenerateData` fixed.
struct StatisticsPass<'a> {
    features: &'a [f64],
    size: &'a [usize],
    dim: usize,
    spacing: &'a [f64],
    origin: &'a [f64],
    direction: &'a [f64],
    bin_mins: &'a [f64],
    bin_maxs: &'a [f64],
}

impl StatisticsPass<'_> {
    /// `StatisticsLabelMapFilter::ThreadedProcessLabelObject`
    /// (`itkStatisticsLabelMapFilter.hxx:53-307`), minus the shape pass its
    /// first line delegates to the superclass.
    fn run(&self, indices: &[Vec<i64>], shape: ShapeStatistics) -> IntensityStatistics {
        let &StatisticsPass {
            features,
            size,
            dim,
            spacing,
            origin,
            direction,
            bin_mins,
            bin_maxs,
        } = self;
        let mut frequency = vec![0u64; bin_mins.len()];
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut min_idx = vec![0i64; dim];
        let mut max_idx = vec![0i64; dim];
        let (mut sum, mut sum2, mut sum3, mut sum4) = (0.0, 0.0, 0.0, 0.0);
        let mut center_of_gravity = [0.0f64; MAX_DIM];
        let mut central_moments: Mat = [[0.0; MAX_DIM]; MAX_DIM];

        for index in indices {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for d in 0..dim {
                offset += index[d] as usize * stride;
                stride *= size[d];
            }
            let v = features[offset];
            frequency[bin_index(bin_mins, bin_maxs, v)] += 1;

            // `<=` and `>=`: the last extremum in raster order wins.
            if v <= min {
                min = v;
                min_idx.copy_from_slice(&index[..dim]);
            }
            if v >= max {
                max = v;
                max_idx.copy_from_slice(&index[..dim]);
            }

            sum += v;
            sum2 += v * v;
            sum3 += v.powf(3.0);
            sum4 += v.powf(4.0);

            let mut padded = [0i64; MAX_DIM];
            padded[..dim].copy_from_slice(&index[..dim]);
            let pp = index_to_physical(&padded, dim, spacing, origin, direction);
            for i in 0..dim {
                center_of_gravity[i] += pp[i] * v;
                central_moments[i][i] += v * pp[i] * pp[i];
                for j in i + 1..dim {
                    let weight = v * pp[i] * pp[j];
                    central_moments[i][j] += weight;
                    central_moments[j][i] += weight;
                }
            }
        }

        let total_freq: u64 = frequency.iter().sum();
        let n = total_freq as f64;
        let mean = sum / n;
        let variance = if total_freq > 1 {
            (sum2 - sum * sum / n) / (n - 1.0)
        } else {
            0.0
        };
        let sigma = variance.sqrt();
        let mean2 = mean * mean;

        let mut skewness = 0.0;
        if (variance * sigma).abs() > f64::MIN_POSITIVE {
            skewness = ((sum3 - 3.0 * mean * sum2) / n + 2.0 * mean * mean2) / (variance * sigma);
        }
        let mut kurtosis = 0.0;
        if variance.abs() > f64::MIN_POSITIVE {
            kurtosis = ((sum4 - 4.0 * mean * sum3 + 6.0 * mean2 * sum2) / n - 3.0 * mean2 * mean2)
                / (variance * variance)
                - 3.0;
        }

        let median = median_of(&frequency, bin_mins, bin_maxs, total_freq);

        let mut principal_moments = [0.0f64; MAX_DIM];
        let mut principal_axes: Mat = [[0.0; MAX_DIM]; MAX_DIM];
        let mut elongation = 0.0;
        let mut flatness = 0.0;

        if !is_almost_zero(sum) {
            for (i, row) in central_moments.iter_mut().enumerate().take(dim) {
                center_of_gravity[i] /= sum;
                for v in row.iter_mut().take(dim) {
                    *v /= sum;
                }
            }
            for i in 0..dim {
                for j in 0..dim {
                    central_moments[i][j] -= center_of_gravity[i] * center_of_gravity[j];
                }
            }
            // The normalized second-order central moment of a single pixel.
            for i in 0..dim {
                central_moments[i][i] += spacing[i] * spacing[i] / 12.0;
            }

            let (eigenvalues, eigenvectors) = symmetric_eigen(&central_moments, dim);
            // Clamp each eigenvalue to >= 0, as `ShapeLabelMapFilter` does
            // (`itkShapeLabelMapFilter.hxx:315`). A signed or floating-point
            // feature image can make the weighted covariance indefinite; a
            // negative eigenvalue is meaningless as a second-order moment and
            // would drive the ratios below to `sqrt` of a negative (NaN).
            for (dst, &ev) in principal_moments[..dim].iter_mut().zip(&eigenvalues[..dim]) {
                *dst = ev.max(0.0);
            }
            for i in 0..dim {
                for j in 0..dim {
                    principal_axes[i][j] = eigenvectors[j][i];
                }
            }
            let det = determinant(&principal_axes, dim);
            for v in principal_axes[dim - 1].iter_mut().take(dim) {
                *v *= det;
            }

            // Guard flatness on `pm[0]` and elongation on `pm[dim-2]`
            // separately, and take each root only when its ratio is positive,
            // as `ShapeLabelMapFilter` does (`itkShapeLabelMapFilter.hxx:344-361`).
            if !is_almost_zero(principal_moments[0]) {
                let ratio = principal_moments[1] / principal_moments[0];
                if ratio > 0.0 {
                    flatness = ratio.sqrt();
                }
            }
            if !is_almost_zero(principal_moments[dim - 2]) {
                let ratio = principal_moments[dim - 1] / principal_moments[dim - 2];
                if ratio > 0.0 {
                    elongation = ratio.sqrt();
                }
            }
        } else {
            center_of_gravity = [0.0; MAX_DIM];
        }

        let mut axes = Vec::with_capacity(dim * dim);
        for row in principal_axes.iter().take(dim) {
            axes.extend_from_slice(&row[..dim]);
        }

        IntensityStatistics {
            shape,
            minimum: min,
            maximum: max,
            minimum_index: min_idx,
            maximum_index: max_idx,
            sum,
            mean,
            median,
            variance,
            standard_deviation: sigma,
            skewness,
            kurtosis,
            center_of_gravity: center_of_gravity[..dim].to_vec(),
            weighted_principal_moments: principal_moments[..dim].to_vec(),
            weighted_principal_axes: axes,
            weighted_elongation: elongation,
            weighted_flatness: flatness,
        }
    }
}

/// `itkStatisticsLabelMapFilter.hxx:173-196`.
///
/// The cumulative frequency is compared against the *integer* `(total + 1) / 2`,
/// and the even-population averaging step tests the *integer* `total / 2`, so
/// for an odd population the second bin is never consulted.
fn median_of(frequency: &[u64], bin_mins: &[f64], bin_maxs: &[f64], total_freq: u64) -> f64 {
    let center = |i: usize| (bin_mins[i] + bin_maxs[i]) / 2.0;
    let mut count: u64 = 0;
    for i in 0..frequency.len() {
        count += frequency[i];
        // `(total_freq + 1) / 2`, integer division, as upstream.
        if count >= total_freq.div_ceil(2) {
            let mut median = center(i);
            if total_freq.is_multiple_of(2) && count == total_freq / 2 {
                for (j, &f) in frequency.iter().enumerate().skip(i + 1) {
                    if f > 0 {
                        median += center(j);
                        median *= 0.5;
                        break;
                    }
                }
            }
            return median;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(
        label: &[u8],
        feature: Vec<f64>,
        size: &[usize],
        settings: &LabelIntensityStatisticsSettings,
    ) -> BTreeMap<i64, IntensityStatistics> {
        let l = Image::from_vec(size, label.to_vec()).unwrap();
        let f = Image::from_vec(size, feature).unwrap();
        label_intensity_statistics(&l, &f, settings).unwrap()
    }

    #[test]
    fn sum_mean_variance_and_standard_deviation_of_one_label() {
        // Label 1 covers the feature values 1, 2, 3, 4.
        let out = stats(
            &[1, 1, 1, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            &[4, 1],
            &Default::default(),
        );
        let s = &out[&1];
        assert_eq!(s.sum, 10.0);
        assert_eq!(s.mean, 2.5);
        // Sample variance with an (n - 1) denominator: 5/3.
        assert!((s.variance - 5.0 / 3.0).abs() < 1e-12);
        assert!((s.standard_deviation - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_single_pixel_object_has_zero_variance_not_a_division_by_zero() {
        let out = stats(
            &[1, 0, 0, 0],
            vec![7.0, 0.0, 0.0, 0.0],
            &[4, 1],
            &Default::default(),
        );
        let s = &out[&1];
        assert_eq!(s.variance, 0.0);
        assert_eq!(s.standard_deviation, 0.0);
        assert_eq!(s.mean, 7.0);
        assert_eq!(s.skewness, 0.0);
        assert_eq!(s.kurtosis, 0.0);
    }

    #[test]
    fn a_constant_object_has_zero_variance_skewness_and_kurtosis() {
        let out = stats(&[1, 1, 1, 1], vec![3.0; 4], &[4, 1], &Default::default());
        let s = &out[&1];
        assert_eq!(s.variance, 0.0);
        assert_eq!(s.skewness, 0.0);
        assert_eq!(s.kurtosis, 0.0);
    }

    #[test]
    fn minimum_and_maximum_indices_report_the_last_tie_in_raster_order() {
        // Both 1 and 5 appear twice; `<=` / `>=` keep the later index.
        let out = stats(
            &[1, 1, 1, 1],
            vec![1.0, 5.0, 1.0, 5.0],
            &[4, 1],
            &Default::default(),
        );
        let s = &out[&1];
        assert_eq!(s.minimum, 1.0);
        assert_eq!(s.maximum, 5.0);
        assert_eq!(s.minimum_index, vec![2, 0]);
        assert_eq!(s.maximum_index, vec![3, 0]);
    }

    #[test]
    fn indices_are_two_dimensional_and_row_major() {
        let out = stats(
            &[0, 0, 0, 1],
            vec![0.0, 0.0, 0.0, 9.0],
            &[2, 2],
            &Default::default(),
        );
        assert_eq!(out[&1].maximum_index, vec![1, 1]);
        assert_eq!(out[&1].minimum_index, vec![1, 1]);
    }

    #[test]
    fn each_label_sees_only_its_own_feature_pixels() {
        let out = stats(
            &[1, 1, 2, 2],
            vec![1.0, 3.0, 10.0, 20.0],
            &[4, 1],
            &Default::default(),
        );
        assert_eq!(out[&1].sum, 4.0);
        assert_eq!(out[&2].sum, 30.0);
        assert_eq!(out[&1].maximum, 3.0);
        assert_eq!(out[&2].minimum, 10.0);
    }

    #[test]
    fn the_background_label_gets_no_entry() {
        let out = stats(&[0, 1, 0, 0], vec![5.0; 4], &[4, 1], &Default::default());
        assert_eq!(out.keys().copied().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn a_non_zero_background_value_selects_a_different_object_set() {
        let settings = LabelIntensityStatisticsSettings {
            background_value: 1.0,
            ..Default::default()
        };
        let out = stats(&[1, 2, 1, 2], vec![5.0; 4], &[4, 1], &settings);
        assert_eq!(out.keys().copied().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn the_shape_attributes_match_label_shape_statistics() {
        let l = Image::from_vec(&[4, 2], vec![1u8, 1, 0, 0, 1, 1, 0, 0]).unwrap();
        let f = Image::from_vec(&[4, 2], vec![1.0f64; 8]).unwrap();
        let stats = label_intensity_statistics(&l, &f, &Default::default()).unwrap();
        let shape = label_shape_statistics(&l, &Default::default()).unwrap();
        assert_eq!(stats[&1].shape, shape[&1]);
        assert!(stats[&1].shape.oriented_bounding_box.is_none());
    }

    // ---- the histogram-backed median ------------------------------------

    #[test]
    fn the_median_is_a_bin_centre_not_a_data_value() {
        // Feature range [0, 4] over 4 bins: bin bounds 0,1,2,3,4; centres
        // 0.5, 1.5, 2.5, 3.5. The values 0..=4 land in bins 0,1,2,3,3.
        // total = 5, (5+1)/2 = 3, reached at bin 2 -> centre 2.5.
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 4,
            ..Default::default()
        };
        let out = stats(
            &[1, 1, 1, 1, 1],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            &[5, 1],
            &settings,
        );
        assert_eq!(out[&1].median, 2.5);
        // The true median of the data is 2.0.
        assert_eq!(out[&1].mean, 2.0);
    }

    #[test]
    fn an_even_population_averages_the_median_bin_with_the_next_occupied_one() {
        // Range [0, 4], 4 bins. Values 0, 1, 3, 4 -> bins 0, 1, 3, 3.
        // total = 4; (4+1)/2 = 2, reached at bin 1 with count == 2 == total/2,
        // so the next occupied bin (3, centre 3.5) is averaged in: (1.5+3.5)/2.
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 4,
            ..Default::default()
        };
        let out = stats(&[1, 1, 1, 1], vec![0.0, 1.0, 3.0, 4.0], &[4, 1], &settings);
        assert_eq!(out[&1].median, 2.5);
    }

    #[test]
    fn an_odd_population_never_averages_two_bins() {
        // total = 3; (3+1)/2 = 2 and total/2 = 1, so `count == total/2` cannot
        // hold at the bin where `count >= 2` first does.
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 4,
            ..Default::default()
        };
        let out = stats(&[1, 1, 1], vec![0.0, 1.0, 4.0], &[3, 1], &settings);
        assert_eq!(out[&1].median, 1.5);
    }

    #[test]
    fn the_maximum_valued_pixel_lands_in_the_last_bin() {
        // `GetIndex` with ClipBinsAtEnds off: v >= max of the last bin -> last bin.
        let (mins, maxs) = histogram_bins(0.0, 4.0, 4);
        assert_eq!(bin_index(&mins, &maxs, 4.0), 3);
        assert_eq!(bin_index(&mins, &maxs, 5.0), 3);
        assert_eq!(bin_index(&mins, &maxs, -1.0), 0);
        assert_eq!(bin_index(&mins, &maxs, 0.0), 0);
        assert_eq!(bin_index(&mins, &maxs, 0.999), 0);
        assert_eq!(bin_index(&mins, &maxs, 1.0), 1);
    }

    #[test]
    fn the_histogram_spans_the_whole_feature_image_not_the_label() {
        // Label 1 covers only 0 and 1, but the histogram's range is [0, 100],
        // so both fall in bin 0 and the median is that bin's centre, 25.
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 2,
            ..Default::default()
        };
        let out = stats(
            &[1, 1, 0, 0],
            vec![0.0, 1.0, 50.0, 100.0],
            &[4, 1],
            &settings,
        );
        assert_eq!(out[&1].median, 25.0);
    }

    #[test]
    fn a_constant_feature_image_puts_every_pixel_in_the_last_bin() {
        // lower == upper, so every bin has zero width and `v >= maxs[last]`.
        let out = stats(&[1, 1], vec![7.0, 7.0], &[2, 1], &Default::default());
        assert_eq!(out[&1].median, 7.0);
    }

    #[test]
    fn one_bin_holds_everything_and_its_centre_is_the_midrange() {
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 1,
            ..Default::default()
        };
        let out = stats(&[1, 1, 1], vec![0.0, 1.0, 10.0], &[3, 1], &settings);
        assert_eq!(out[&1].median, 5.0);
    }

    #[test]
    fn zero_bins_is_rejected() {
        let l = Image::from_vec(&[2, 1], vec![1u8, 1]).unwrap();
        let f = Image::from_vec(&[2, 1], vec![1.0f64, 2.0]).unwrap();
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 0,
            ..Default::default()
        };
        assert_eq!(
            label_intensity_statistics(&l, &f, &settings),
            Err(FilterError::InvalidHistogramBins(0))
        );
    }

    #[test]
    fn the_integer_padding_branch_centres_the_bins_on_the_integers() {
        // UInt8 feature, 256 bins: bounds become [-0.5, 255.5], so bin i is
        // [i - 0.5, i + 0.5) and its centre is exactly i.
        let (lower, upper) = histogram_bounds(PixelId::UInt8, 256, 3.0, 9.0);
        assert_eq!((lower, upper), (-0.5, 255.5));
        let (mins, maxs) = histogram_bins(lower, upper, 256);
        for i in [0usize, 1, 127, 254, 255] {
            assert!(((mins[i] + maxs[i]) / 2.0 - i as f64).abs() < 1e-6);
        }
        // 128 bins on the same type takes the image-range branch instead.
        assert_eq!(histogram_bounds(PixelId::UInt8, 128, 3.0, 9.0), (3.0, 9.0));
        // So does a 4-byte integer type at 256 bins.
        assert_eq!(histogram_bounds(PixelId::UInt32, 256, 3.0, 9.0), (3.0, 9.0));
        // And a float type at any bin count.
        assert_eq!(
            histogram_bounds(PixelId::Float32, 256, 3.0, 9.0),
            (3.0, 9.0)
        );
    }

    #[test]
    fn the_integer_padding_branch_reports_an_exact_integer_median() {
        let l = Image::from_vec(&[3, 1], vec![1u8; 3]).unwrap();
        let f = Image::from_vec(&[3, 1], vec![10u8, 20, 30]).unwrap();
        let settings = LabelIntensityStatisticsSettings {
            number_of_bins: 256,
            ..Default::default()
        };
        let out = label_intensity_statistics(&l, &f, &settings).unwrap();
        assert_eq!(out[&1].median, 20.0);
    }

    #[test]
    fn histogram_bin_bounds_narrow_the_interval_to_f32() {
        // 3 bins over [0, 1]: the exact width is 1/3, but ITK stores it as an
        // f32, so the second bin's lower bound is f32(1/3) widened, not 1/3.
        let (mins, _) = histogram_bins(0.0, 1.0, 3);
        assert_eq!(mins[1], (1.0f32 / 3.0) as f64);
        assert_ne!(mins[1], 1.0 / 3.0);
    }

    // ---- skewness / kurtosis / weighted moments --------------------------

    #[test]
    fn skewness_and_kurtosis_of_a_known_sample() {
        let out = stats(
            &[1, 1, 1, 1],
            vec![1.0, 2.0, 3.0, 10.0],
            &[4, 1],
            &Default::default(),
        );
        let s = &out[&1];
        let n = 4.0f64;
        let (sum, sum2) = (16.0f64, 1.0f64 + 4.0 + 9.0 + 100.0);
        let sum3 = 1.0f64 + 8.0 + 27.0 + 1000.0;
        let sum4 = 1.0f64 + 16.0 + 81.0 + 10000.0;
        let mean = sum / n;
        let variance = (sum2 - sum * sum / n) / (n - 1.0);
        let sigma = variance.sqrt();
        let expected_skew =
            ((sum3 - 3.0 * mean * sum2) / n + 2.0 * mean * mean * mean) / (variance * sigma);
        let expected_kurt = ((sum4 - 4.0 * mean * sum3 + 6.0 * mean * mean * sum2) / n
            - 3.0 * mean.powi(4))
            / (variance * variance)
            - 3.0;
        assert!((s.skewness - expected_skew).abs() < 1e-12);
        assert!((s.kurtosis - expected_kurt).abs() < 1e-12);
    }

    #[test]
    fn the_center_of_gravity_is_the_intensity_weighted_centroid() {
        // Pixels at x = 0, 1, 2, 3 with weights 0, 0, 0, 1.
        let out = stats(
            &[1, 1, 1, 1],
            vec![0.0, 0.0, 0.0, 1.0],
            &[4, 1],
            &Default::default(),
        );
        assert_eq!(out[&1].center_of_gravity, vec![3.0, 0.0]);
        assert_eq!(out[&1].shape.centroid, vec![1.5, 0.0]);
    }

    #[test]
    fn a_zero_intensity_sum_zeroes_the_center_of_gravity_and_the_moments() {
        let out = stats(&[1, 1], vec![0.0, 0.0], &[2, 1], &Default::default());
        let s = &out[&1];
        assert_eq!(s.center_of_gravity, vec![0.0, 0.0]);
        assert_eq!(s.weighted_principal_moments, vec![0.0, 0.0]);
        assert_eq!(s.weighted_principal_axes, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.weighted_elongation, 0.0);
        assert_eq!(s.weighted_flatness, 0.0);
    }

    #[test]
    fn a_mixed_sign_feature_image_clamps_the_moments_and_keeps_the_ratios_finite() {
        // A 1x3 row weighted 3, -1, -1: sum = 1, cog_x = -3, so the raw x central
        // moment is -5 - 9 = -14 before the +spacing^2/12 term and the y one is
        // 0 + 1/12. The eigenvalues straddle zero: sorted ascending they are
        // {-14 + 1/12, 1/12}. This port clamps each to >= 0 as
        // `ShapeLabelMapFilter` does (`itkShapeLabelMapFilter.hxx:315`), so the
        // stored moments are {0, 1/12}, and it takes each ratio's root only when
        // the ratio is positive (`.hxx:344-361`), so both ratios stay finite
        // instead of the upstream NaN.
        let out = stats(
            &[1, 1, 1],
            vec![3.0, -1.0, -1.0],
            &[3, 1],
            &Default::default(),
        );
        let s = &out[&1];
        assert_eq!(s.sum, 1.0);
        assert_eq!(s.center_of_gravity, vec![-3.0, 0.0]);
        let pm = &s.weighted_principal_moments;
        assert_eq!(pm[0], 0.0, "{pm:?}");
        assert!((pm[1] - 1.0 / 12.0).abs() < 1e-9, "{pm:?}");
        // pm[0] is clamped to zero, so both ratios' guards suppress the root: the
        // results are a finite 0, not NaN.
        assert_eq!(s.weighted_elongation, 0.0);
        assert_eq!(s.weighted_flatness, 0.0);
        // The shape filter, on the same object, clamps and stays finite too.
        assert_eq!(s.shape.principal_moments[0], 0.0);
        assert!(s.shape.elongation.is_finite());
    }

    #[test]
    fn the_weighted_principal_axes_form_a_proper_rotation() {
        let out = stats(
            &[1, 1, 1, 1, 1, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &Default::default(),
        );
        let a = &out[&1].weighted_principal_axes;
        let det = a[0] * a[3] - a[1] * a[2];
        assert!((det - 1.0).abs() < 1e-9, "determinant was {det}");
    }

    #[test]
    fn a_uniformly_weighted_object_exceeds_the_shape_moments_by_the_pixels_own() {
        // Under a constant feature the weighted second-order central moments are
        // the shape filter's *plus* spacing^2/12 on the diagonal, which only
        // `StatisticsLabelMapFilter` adds (`.hxx:224-228`). With unit isotropic
        // spacing that shifts every eigenvalue by exactly 1/12.
        let out = stats(
            &[1, 1, 1, 1, 1, 1],
            vec![2.0; 6],
            &[3, 2],
            &Default::default(),
        );
        let s = &out[&1];
        assert_eq!(s.shape.principal_moments.len(), 2);
        for (w, p) in s
            .weighted_principal_moments
            .iter()
            .zip(&s.shape.principal_moments)
        {
            assert!((w - p - 1.0 / 12.0).abs() < 1e-12, "{w} vs {p} + 1/12");
        }
        // 3 pixels across x: variance 2/3; 2 pixels across y: variance 1/4.
        assert!((s.shape.principal_moments[0] - 0.25).abs() < 1e-12);
        assert!((s.shape.principal_moments[1] - 2.0 / 3.0).abs() < 1e-12);
        // The centre of gravity does coincide with the centroid.
        assert_eq!(s.center_of_gravity, s.shape.centroid);
    }

    #[test]
    fn spacing_enters_the_moments_and_the_center_of_gravity() {
        let l = Image::from_vec(&[3, 1], vec![1u8; 3]).unwrap();
        let mut f = Image::from_vec(&[3, 1], vec![0.0f64, 0.0, 1.0]).unwrap();
        f.set_spacing(&[2.0, 1.0]).unwrap();
        let mut l = l;
        l.set_spacing(&[2.0, 1.0]).unwrap();
        let out = label_intensity_statistics(&l, &f, &Default::default()).unwrap();
        assert_eq!(out[&1].center_of_gravity, vec![4.0, 0.0]);
    }

    // ---- input validation ------------------------------------------------

    #[test]
    fn a_float_label_image_is_rejected() {
        let l = Image::from_vec(&[2, 1], vec![1.0f32, 1.0]).unwrap();
        let f = Image::from_vec(&[2, 1], vec![1.0f64, 2.0]).unwrap();
        assert_eq!(
            label_intensity_statistics(&l, &f, &Default::default()),
            Err(FilterError::RequiresIntegerPixelType(PixelId::Float32))
        );
    }

    #[test]
    fn a_vector_feature_image_is_rejected() {
        let l = Image::from_vec(&[2, 1], vec![1u8, 1]).unwrap();
        let f = Image::from_vec_vector(&[2, 1], 2, vec![1u8; 4]).unwrap();
        assert_eq!(
            label_intensity_statistics(&l, &f, &Default::default()),
            Err(FilterError::Core(
                crate::core::Error::RequiresScalarPixelType(PixelId::VectorUInt8)
            ))
        );
    }

    #[test]
    fn a_complex_label_image_is_rejected() {
        let l = Image::new(&[2, 1], PixelId::ComplexFloat32);
        let f = Image::from_vec(&[2, 1], vec![1.0f64, 2.0]).unwrap();
        assert_eq!(
            label_intensity_statistics(&l, &f, &Default::default()),
            Err(FilterError::RequiresIntegerPixelType(
                PixelId::ComplexFloat32
            ))
        );
    }

    #[test]
    fn a_complex_feature_image_is_rejected() {
        let l = Image::from_vec(&[2, 1], vec![1u8, 1]).unwrap();
        let f = Image::new(&[2, 1], PixelId::ComplexFloat32);
        assert_eq!(
            label_intensity_statistics(&l, &f, &Default::default()),
            Err(FilterError::Core(
                crate::core::Error::RequiresScalarPixelType(PixelId::ComplexFloat32)
            ))
        );
    }

    #[test]
    fn a_size_mismatch_is_rejected() {
        let l = Image::from_vec(&[2, 1], vec![1u8, 1]).unwrap();
        let f = Image::from_vec(&[3, 1], vec![1.0f64, 2.0, 3.0]).unwrap();
        assert_eq!(
            label_intensity_statistics(&l, &f, &Default::default()),
            Err(FilterError::SizeMismatch {
                a: vec![2, 1],
                b: vec![3, 1]
            })
        );
    }

    #[test]
    fn a_four_dimensional_label_image_is_rejected() {
        let l = Image::from_vec(&[2, 2, 2, 2], vec![1u8; 16]).unwrap();
        let f = Image::from_vec(&[2, 2, 2, 2], vec![1.0f64; 16]).unwrap();
        assert_eq!(
            label_intensity_statistics(&l, &f, &Default::default()),
            Err(FilterError::UnsupportedShapeDimension(4))
        );
    }

    #[test]
    fn a_three_dimensional_label_image_is_accepted() {
        let l = Image::from_vec(&[2, 2, 2], vec![1u8; 8]).unwrap();
        let f = Image::from_vec(&[2, 2, 2], (0..8).map(|v| v as f64).collect()).unwrap();
        let out = label_intensity_statistics(&l, &f, &Default::default()).unwrap();
        assert_eq!(out[&1].sum, 28.0);
        assert_eq!(out[&1].maximum_index, vec![1, 1, 1]);
        assert_eq!(out[&1].center_of_gravity.len(), 3);
        assert_eq!(out[&1].weighted_principal_axes.len(), 9);
    }
}
