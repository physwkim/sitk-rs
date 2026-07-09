//! `HistogramMatchingImageFilter`: match `source`'s intensity histogram to
//! `reference`'s via quantile (Nyul et al. 2000) matching, ported from
//! `itkHistogramMatchingImageFilter.h(.hxx)`.

use crate::error::{FilterError, Result};
use crate::histogram::Histogram;
use crate::image_from_f64;
use crate::quantize_to_pixel_type;
use sitk_core::Image;

fn min_max_mean(vals: &[f64]) -> (f64, f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &v in vals {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v;
    }
    (lo, hi, sum / vals.len() as f64)
}

/// `itk::Math::NotAlmostEquals(d, 0.0)`, specialized to a comparison against
/// exactly zero: `itkMath.h`'s `FloatAlmostEqual` combines a ULP check with
/// an absolute-difference check (`maxAbsoluteDifference` defaulting to
/// `4 * min positive normal`, negligible here), and against a literal `0.0`
/// comparand the ULP branch never fires, so the comparison collapses to
/// `|d| <= 0.1 * epsilon`. Every denominator this module tests is either a
/// real quantile gap or structurally exactly `0.0`, so this tight threshold
/// is enough to tell them apart.
fn is_almost_zero(d: f64) -> bool {
    d.abs() <= 0.1 * f64::EPSILON
}

/// `HistogramMatchingImageFilter`: remap `source`'s intensities so its
/// histogram matches `reference`'s, via a piecewise-linear map through
/// `number_of_match_points` interior quantile-correspondence points plus the
/// two threshold/max endpoints. `source` and `reference` must share a pixel
/// type but not a size (SimpleITK's `HistogramMatchingImageFilter.yaml`
/// marks `ReferenceImage` `no_size_check: true`). Output pixel type follows
/// `source`.
pub fn histogram_matching(
    source: &Image,
    reference: &Image,
    number_of_histogram_levels: u32,
    number_of_match_points: u32,
    threshold_at_mean_intensity: bool,
) -> Result<Image> {
    if source.pixel_id() != reference.pixel_id() {
        return Err(FilterError::TypeMismatch {
            a: source.pixel_id(),
            b: reference.pixel_id(),
        });
    }
    let source_vals = source.to_f64_vec();
    let reference_vals = reference.to_f64_vec();
    if source_vals.is_empty() || reference_vals.is_empty() {
        return Err(FilterError::DegenerateRange);
    }

    let (source_min, source_max, source_mean_raw) = min_max_mean(&source_vals);
    let (reference_min, reference_max, reference_mean_raw) = min_max_mean(&reference_vals);
    // `ComputeMinMaxMean`'s `static_cast<THistogramMeasurement>(sum / count)`.
    let source_mean = quantize_to_pixel_type(source.pixel_id(), source_mean_raw);
    let reference_mean = quantize_to_pixel_type(reference.pixel_id(), reference_mean_raw);

    let source_threshold = if threshold_at_mean_intensity {
        source_mean
    } else {
        source_min
    };
    let reference_threshold = if threshold_at_mean_intensity {
        reference_mean
    } else {
        reference_min
    };

    let source_hist = Histogram::from_bounds(
        &source_vals,
        number_of_histogram_levels,
        source_threshold,
        source_max,
        source_min,
        source_max,
    )?;
    let reference_hist = Histogram::from_bounds(
        &reference_vals,
        number_of_histogram_levels,
        reference_threshold,
        reference_max,
        reference_min,
        reference_max,
    )?;

    // `m_QuantileTable`: `NumberOfMatchPoints + 2` columns, index 0 the
    // threshold, index `N+1` the max, `1..=N` the interior match points at
    // `p = j / (N + 1)`.
    let match_points = number_of_match_points as usize;
    let mut source_q = vec![0.0f64; match_points + 2];
    let mut reference_q = vec![0.0f64; match_points + 2];
    source_q[0] = source_threshold;
    reference_q[0] = reference_threshold;
    source_q[match_points + 1] = source_max;
    reference_q[match_points + 1] = reference_max;
    let delta = 1.0 / (match_points as f64 + 1.0);
    for j in 1..=match_points {
        source_q[j] = source_hist.quantile(j as f64 * delta);
        reference_q[j] = reference_hist.quantile(j as f64 * delta);
    }

    // `m_Gradients`: `NumberOfMatchPoints + 1` slopes between consecutive
    // quantile-table columns.
    let mut gradients = vec![0.0f64; match_points + 1];
    for (j, g) in gradients.iter_mut().enumerate() {
        let denom = source_q[j + 1] - source_q[j];
        *g = if is_almost_zero(denom) {
            0.0
        } else {
            (reference_q[j + 1] - reference_q[j]) / denom
        };
    }
    let lower_gradient = {
        let denom = source_q[0] - source_min;
        if is_almost_zero(denom) {
            0.0
        } else {
            (reference_q[0] - reference_min) / denom
        }
    };
    // `source_q[match_points + 1]` is always exactly `source_max` by
    // construction above, so this denominator is always exactly `0.0` and
    // `upper_gradient` is always `0.0` — this is `itkHistogramMatchingImageFilter.hxx`'s
    // own behavior (`m_UpperGradient` is computed from the same structurally-zero
    // difference), not a simplification made here: any source pixel equal to
    // `source_max` always maps flatly to exactly `reference_max`.
    let upper_gradient = {
        let denom = source_q[match_points + 1] - source_max;
        if is_almost_zero(denom) {
            0.0
        } else {
            (reference_q[match_points + 1] - reference_max) / denom
        }
    };

    // `DynamicThreadedGenerateData`'s per-pixel `std::lower_bound` walk over
    // `m_QuantileTable[0]`, then linear interpolation from the bracketing
    // column (or extrapolation from an end gradient outside the table).
    let out: Vec<f64> = source_vals
        .iter()
        .map(|&src| {
            let mut j = 0usize;
            while j < source_q.len() {
                if src < source_q[j] {
                    break;
                }
                j += 1;
            }
            if j == 0 {
                reference_min + (src - source_min) * lower_gradient
            } else if j == source_q.len() {
                reference_max + (src - source_max) * upper_gradient
            } else {
                reference_q[j - 1] + (src - source_q[j - 1]) * gradients[j - 1]
            }
        })
        .collect();

    image_from_f64(source.pixel_id(), source.size(), source, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::{Image, PixelId};

    #[test]
    fn shifted_image_maps_back_onto_reference_within_quantization_error() {
        // Reference: 0..100 ramp. Source: the same ramp shifted +50 (51..150,
        // so both share the same shape/spread). Matching should map source
        // back close to the reference's range.
        let reference: Vec<f64> = (0..=100).map(|v| v as f64).collect();
        let source: Vec<f64> = (50..=150).map(|v| v as f64).collect();
        let src_img = Image::from_vec(&[source.len()], source).unwrap();
        let ref_img = Image::from_vec(&[reference.len()], reference).unwrap();

        let out = histogram_matching(&src_img, &ref_img, 256, 7, false)
            .unwrap()
            .to_f64_vec();
        // The matched max should land near the reference max (100), and the
        // matched min near the reference min (0), within a few histogram-bin
        // widths of quantization error.
        let out_min = out.iter().cloned().fold(f64::INFINITY, f64::min);
        let out_max = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((out_min - 0.0).abs() < 2.0, "out_min = {out_min}");
        assert!((out_max - 100.0).abs() < 2.0, "out_max = {out_max}");
    }

    #[test]
    fn source_max_pixel_always_maps_to_reference_max() {
        // Documented consequence of `upper_gradient` being structurally 0.0.
        let reference: Vec<f64> = (0..=50).map(|v| v as f64).collect();
        let source: Vec<f64> = (0..=200).map(|v| v as f64).collect();
        let src_img = Image::from_vec(&[source.len()], source).unwrap();
        let ref_img = Image::from_vec(&[reference.len()], reference).unwrap();
        let out = histogram_matching(&src_img, &ref_img, 100, 5, false)
            .unwrap()
            .to_f64_vec();
        assert_eq!(*out.last().unwrap(), 50.0);
    }

    #[test]
    fn threshold_at_mean_changes_the_mapping() {
        // Skewed, non-proportional distributions: thresholding at the mean
        // excludes a different sub-population from each histogram's
        // quantile table than thresholding at the minimum does, so the two
        // settings must produce different maps for at least one pixel. (A
        // pair of plain uniform ramps, or a two-value cluster pair, is
        // scale-invariant under this kind of range restriction and would
        // map identically either way — this needs a real distribution
        // shape.) `source` skews toward its low end (mean 3, of range
        // 0..=9); `reference` skews toward its high end (mean 6).
        let mut source = Vec::new();
        let mut reference = Vec::new();
        for v in 0..10 {
            source.extend(std::iter::repeat_n(v as f64, 10 - v));
            reference.extend(std::iter::repeat_n(v as f64, v + 1));
        }
        let src_img = Image::from_vec(&[source.len()], source).unwrap();
        let ref_img = Image::from_vec(&[reference.len()], reference).unwrap();

        let without = histogram_matching(&src_img, &ref_img, 32, 5, false)
            .unwrap()
            .to_f64_vec();
        let with = histogram_matching(&src_img, &ref_img, 32, 5, true)
            .unwrap()
            .to_f64_vec();
        assert_ne!(without, with);
    }

    #[test]
    fn constant_reference_maps_every_source_pixel_to_that_constant() {
        // reference_min == reference_max, so reference_q collapses to a
        // single value at every column; every source pixel <= source_max
        // is captured by either an interior segment or the final flat
        // upper-gradient segment, both anchored at that constant.
        let reference = vec![42.0; 10];
        let source: Vec<f64> = (0..=20).map(|v| v as f64).collect();
        let src_img = Image::from_vec(&[source.len()], source).unwrap();
        let ref_img = Image::from_vec(&[reference.len()], reference).unwrap();
        let out = histogram_matching(&src_img, &ref_img, 64, 3, false)
            .unwrap()
            .to_f64_vec();
        for v in out {
            assert!((v - 42.0).abs() < 1e-9, "expected 42.0, got {v}");
        }
    }

    #[test]
    fn pixel_type_mismatch_errors() {
        let a = Image::from_vec(&[4], vec![1u8, 2, 3, 4]).unwrap();
        let b = Image::from_vec(&[4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert!(matches!(
            histogram_matching(&a, &b, 10, 5, false),
            Err(FilterError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn output_pixel_type_follows_source() {
        let a = Image::from_vec(&[4], vec![1u8, 2, 3, 4]).unwrap();
        let b = Image::from_vec(&[4], vec![10u8, 20, 30, 40]).unwrap();
        let out = histogram_matching(&a, &b, 10, 5, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }
}
