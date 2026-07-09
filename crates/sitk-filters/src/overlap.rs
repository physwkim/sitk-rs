//! `LabelOverlapMeasuresImageFilter`, `HausdorffDistanceImageFilter` /
//! `DirectedHausdorffDistanceImageFilter`, and
//! `SimilarityIndexImageFilter`: segmentation-comparison metrics.
//!
//! Verified against ITK's `Modules/Filtering/ImageStatistics/include/`
//! `itkLabelOverlapMeasuresImageFilter.h`/`.hxx` and
//! `itkLabelOverlapLabelSetMeasures.h`,
//! `Modules/Filtering/DistanceMap/include/`
//! `itkHausdorffDistanceImageFilter.h`/`.hxx` and
//! `itkDirectedHausdorffDistanceImageFilter.h`/`.hxx`, and
//! `Modules/Filtering/ImageCompare/include/`
//! `itkSimilarityIndexImageFilter.h`/`.hxx`.
//!
//! ## `label_overlap_measures`
//!
//! Per pixel, accumulate six counters per label value (matching
//! `LabelOverlapLabelSetMeasures`): `source` (pixel count in the source
//! image with this label, TP+FP), `target` (TP+FN), `union` (TP+FN+FP),
//! `intersection` (TP), `source_complement` (FP), `target_complement` (FN).
//! `ThreadedStreamedGenerateData`'s per-pixel update is reproduced exactly:
//! every pixel increments `source[sourceLabel]` and `target[targetLabel]`
//! unconditionally, then either both `intersection`/`union` on the shared
//! label (pixel matches) or `union`/`source_complement`/`target_complement`
//! split across the two distinct labels (pixel mismatches). Because this is
//! a pure per-label sum, iteration order does not affect the result (ITK's
//! own multi-threaded merge is likewise just summation), so this port
//! accumulates in one pass without replicating the threading.
//!
//! Every label value that appears in either image gets a
//! [`LabelOverlapMeasures`] entry, **including background (label 0)** —
//! `m_LabelSetMeasures` is built the same way in the `.hxx`. Only the
//! *total* (whole-image) measures skip label 0, matching every `Get*()`
//! total accessor's `if (mapIt->first == LabelType{}) continue;` guard.
//!
//! **Degenerate-denominator quirk, reproduced verbatim:** every formula
//! below divides two accumulated counts. When a total's denominator is
//! exactly `0.0` (`Math::ExactlyEquals`), or a per-label formula's
//! documented zero-guard fires, ITK returns `NumericTraits<RealType>::max()`
//! — `RealType` is `double` for every integer label type, so this is
//! `f64::MAX`, **not infinity**. This port returns `f64::MAX` in the same
//! spot rather than substituting `f64::INFINITY` or `NaN`. Two formulas have
//! *no* zero-guard at all in the `.hxx` and are ported with none: per-label
//! `volume_similarity` (divides by `source + target` unguarded — `0.0 /
//! 0.0` is `NaN`, reproduced as-is) and per-label `false_positive_error`,
//! whose guard only checks `source == 0`, not the actual denominator
//! (`source_complement + (n_vox - union)`), which can itself be zero while
//! `source != 0`, again yielding `NaN`.
//!
//! `false_positive_error`'s `n_vox` is the *source* image's total pixel
//! count (`GetInput(0)->GetLargestPossibleRegion()->GetNumberOfPixels()`),
//! used identically for the total and every per-label computation; the
//! total's denominator re-adds `n_vox - union[label]` fresh for every label
//! (not `n_vox` shared once across all labels), matching the `.hxx`'s
//! per-iteration `nComplementIntersection` recomputation inside the `for`
//! loop.
//!
//! SimpleITK's yaml (`LabelOverlapMeasuresImageFilter.yaml`) restricts this
//! filter to `IntegerPixelIDTypeList` for the source image; `TargetImage`
//! is cast to the source's label type via `CastImageToITK` in the generated
//! C++, so in principle a non-integer `TargetImage` would silently
//! truncate. This port instead requires **both** images to already be an
//! integer pixel type ([`FilterError::RequiresIntegerPixelType`]) — the
//! common case, and it avoids picking a truncation-vs-rounding convention
//! for a path SimpleITK itself does not exercise in its own tests.
//!
//! ## `hausdorff_distance` / `directed_hausdorff_distance`
//!
//! `DirectedHausdorffDistanceImageFilter::BeforeThreadedGenerateData` builds
//! a [`crate::distance::signed_maurer_distance_map`] of the *second* input
//! (`SetSquaredDistance(false)`, spacing-aware; `InsideIsPositive` left at
//! its filter default of `false`), then for every non-zero pixel of the
//! *first* input takes `max(distanceMap value, 0)` (negative values mean
//! "inside the second image's object", i.e. zero true distance) and folds a
//! running max (`GetDirectedHausdorffDistance`) and mean
//! (`GetAverageHausdorffDistance`) over those clamped values. If the first
//! input has no non-zero pixels, `AfterThreadedGenerateData` throws
//! (`"pixelcount is equal to 0"`); this port returns
//! [`FilterError::EmptyHausdorffForegroundSet`] in that spot.
//! `HausdorffDistanceImageFilter` runs the directed filter both ways and
//! takes the max of the two directed distances (`GetHausdorffDistance`) and
//! the mean of the two average distances (`GetAverageHausdorffDistance`).
//!
//! **`UseImageSpacing`**: both `.h` headers default `m_UseImageSpacing` to
//! `true`, but `HausdorffDistanceImageFilter.yaml` declares `members: []` —
//! SimpleITK never generates a setter for it, so the procedural filter
//! always runs with spacing-aware distances. This port hardcodes
//! `use_image_spacing = true` for the same reason: no public knob exists to
//! turn it off through this filter's actual SimpleITK surface.
//!
//! ITK sums the per-pixel distances with `CompensatedSummation` (Kahan
//! summation) before dividing by the pixel count; this port uses a plain
//! `f64` accumulator, consistent with every other reduction in this crate
//! ([`crate::statistics`], [`crate::label::label_statistics`]) — a
//! deliberate precision simplification, not a formula change.

use crate::distance::signed_maurer_distance_map;
use crate::error::{FilterError, Result};
use sitk_core::Image;
use std::collections::BTreeMap;

fn require_integer_pixel_type(img: &Image) -> Result<()> {
    if img.pixel_id().is_floating_point() {
        return Err(FilterError::RequiresIntegerPixelType(img.pixel_id()));
    }
    Ok(())
}

fn require_same_size(a: &Image, b: &Image) -> Result<()> {
    if a.size() != b.size() {
        return Err(FilterError::SizeMismatch {
            a: a.size().to_vec(),
            b: b.size().to_vec(),
        });
    }
    Ok(())
}

// ---- LabelOverlapMeasuresImageFilter --------------------------------------

/// `NumericTraits<double>::max()` — what ITK's degenerate-denominator
/// branches return in place of the mathematically undefined ratio. Named so
/// every reproduction site below reads the same way as the `.hxx`'s
/// `NumericTraits<RealType>::max()`. Note this is `f64::MAX`, not
/// `f64::INFINITY`.
const REAL_TYPE_MAX: f64 = f64::MAX;

#[derive(Clone, Copy, Debug, Default)]
struct LabelCounts {
    source: u64,
    target: u64,
    union: u64,
    intersection: u64,
    source_complement: u64,
    target_complement: u64,
}

/// Per-label overlap measures, mirroring `LabelOverlapMeasuresImageFilter`'s
/// `Get*(LabelType)` accessors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelOverlapMeasures {
    /// `GetTargetOverlap(label)`: `intersection / target`; [`REAL_TYPE_MAX`]
    /// when `target == 0`.
    pub target_overlap: f64,
    /// `GetUnionOverlap(label)` a.k.a. Jaccard coefficient:
    /// `intersection / union`; [`REAL_TYPE_MAX`] when `union == 0`.
    pub union_overlap: f64,
    /// `GetMeanOverlap(label)` a.k.a. Dice coefficient: `2*uo / (1+uo)`
    /// where `uo` is [`Self::union_overlap`] (no additional zero-guard).
    pub mean_overlap: f64,
    /// `GetVolumeSimilarity(label)`: `2*(source-target) / (source+target)`,
    /// **unguarded** in the `.hxx` — `NaN` when `source == target == 0`.
    pub volume_similarity: f64,
    /// `GetFalseNegativeError(label)`: `target_complement / target`;
    /// [`REAL_TYPE_MAX`] when `target == 0`.
    pub false_negative_error: f64,
    /// `GetFalsePositiveError(label)`: `source_complement /
    /// (source_complement + (n_vox - union))`; [`REAL_TYPE_MAX`] when
    /// `source == 0` — the guard checked in the `.hxx`, not the
    /// denominator itself, so a `source != 0` label whose denominator is
    /// still `0` yields `NaN` (reproduced, not sanitized).
    pub false_positive_error: f64,
    /// `GetFalseDiscoveryRate(label)`: `source_complement / source`;
    /// [`REAL_TYPE_MAX`] when `source == 0`.
    pub false_discovery_rate: f64,
}

/// Result of [`label_overlap_measures`]: whole-image totals (every
/// non-background label pooled) plus the same measures per label,
/// mirroring `LabelOverlapMeasuresImageFilter`'s `Get*()` /
/// `Get*(LabelType)` accessor pairs.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapMeasures {
    /// `GetTotalOverlap()`: `Σ intersection / Σ target` over labels != 0.
    pub total_overlap: f64,
    /// `GetUnionOverlap()` a.k.a. Jaccard coefficient:
    /// `Σ intersection / Σ union` over labels != 0.
    pub union_overlap: f64,
    /// `GetMeanOverlap()` a.k.a. Dice coefficient: `2*uo / (1+uo)`.
    pub mean_overlap: f64,
    /// `GetVolumeSimilarity()`: `2*Σ(source-target) / Σ(source+target)`
    /// over labels != 0.
    pub volume_similarity: f64,
    /// `GetFalseNegativeError()`: `Σ target_complement / Σ target` over
    /// labels != 0.
    pub false_negative_error: f64,
    /// `GetFalsePositiveError()`: `Σ source_complement / Σ
    /// (source_complement + (n_vox - union))` over labels != 0.
    pub false_positive_error: f64,
    /// `GetFalseDiscoveryRate()`: `Σ source_complement / Σ source` over
    /// labels != 0.
    pub false_discovery_rate: f64,
    /// Per-label measures, keyed by label value — including background
    /// (`0`), which the totals above exclude.
    pub per_label: BTreeMap<i64, LabelOverlapMeasures>,
}

fn union_overlap_of(intersection: u64, union: u64) -> f64 {
    if union == 0 {
        REAL_TYPE_MAX
    } else {
        intersection as f64 / union as f64
    }
}

fn mean_overlap_of(union_overlap: f64) -> f64 {
    2.0 * union_overlap / (1.0 + union_overlap)
}

/// `LabelOverlapMeasuresImageFilter`: overlap agreement/error measures
/// between `source` and `target` label images. Background is label `0`.
/// Both images must already be an integer pixel type and share a size (see
/// module docs).
pub fn label_overlap_measures(source: &Image, target: &Image) -> Result<OverlapMeasures> {
    require_integer_pixel_type(source)?;
    require_integer_pixel_type(target)?;
    require_same_size(source, target)?;

    let source_labels: Vec<i64> = source
        .to_f64_vec()
        .iter()
        .map(|&v| v.round() as i64)
        .collect();
    let target_labels: Vec<i64> = target
        .to_f64_vec()
        .iter()
        .map(|&v| v.round() as i64)
        .collect();
    // `GetInput(0)->GetLargestPossibleRegion()->GetNumberOfPixels()` — the
    // source image's pixel count, which `require_same_size` guarantees
    // equals the target's.
    let n_vox = source_labels.len() as u64;

    let mut counts: BTreeMap<i64, LabelCounts> = BTreeMap::new();
    for (&s, &t) in source_labels.iter().zip(&target_labels) {
        if s == t {
            let e = counts.entry(s).or_default();
            e.source += 1;
            e.target += 1;
            e.intersection += 1;
            e.union += 1;
        } else {
            counts.entry(s).or_default().source += 1;
            counts.entry(t).or_default().target += 1;
            counts.entry(s).or_default().union += 1;
            counts.entry(t).or_default().union += 1;
            counts.entry(s).or_default().source_complement += 1;
            counts.entry(t).or_default().target_complement += 1;
        }
    }

    let mut per_label = BTreeMap::new();
    let (mut num_total, mut den_total) = (0.0f64, 0.0f64);
    let (mut num_union, mut den_union) = (0.0f64, 0.0f64);
    let (mut num_vol, mut den_vol) = (0.0f64, 0.0f64);
    let (mut num_fne, mut den_fne) = (0.0f64, 0.0f64);
    let (mut num_fpe, mut den_fpe) = (0.0f64, 0.0f64);
    let (mut num_fdr, mut den_fdr) = (0.0f64, 0.0f64);

    for (&label, c) in &counts {
        let source = c.source as f64;
        let target = c.target as f64;
        let union = c.union as f64;
        let intersection = c.intersection as f64;
        let source_complement = c.source_complement as f64;
        let target_complement = c.target_complement as f64;

        let target_overlap = if target == 0.0 {
            REAL_TYPE_MAX
        } else {
            intersection / target
        };
        let union_overlap = union_overlap_of(c.intersection, c.union);
        let mean_overlap = mean_overlap_of(union_overlap);
        // Unguarded, matching GetVolumeSimilarity(LabelType): 0/0 -> NaN.
        let volume_similarity = 2.0 * (source - target) / (source + target);
        let false_negative_error = if target == 0.0 {
            REAL_TYPE_MAX
        } else {
            target_complement / target
        };
        let n_complement_intersection = n_vox as f64 - union; // TN
        let false_positive_error = if source == 0.0 {
            REAL_TYPE_MAX
        } else {
            source_complement / (source_complement + n_complement_intersection)
        };
        let false_discovery_rate = if source == 0.0 {
            REAL_TYPE_MAX
        } else {
            source_complement / source
        };

        // Totals skip the background label, matching every Get*() total
        // accessor's `if (mapIt->first == LabelType{}) continue;`.
        if label != 0 {
            num_total += intersection;
            den_total += target;
            num_union += intersection;
            den_union += union;
            num_vol += source - target;
            den_vol += source + target;
            num_fne += target_complement;
            den_fne += target;
            num_fpe += source_complement;
            den_fpe += source_complement + n_complement_intersection;
            num_fdr += source_complement;
            den_fdr += source;
        }

        per_label.insert(
            label,
            LabelOverlapMeasures {
                target_overlap,
                union_overlap,
                mean_overlap,
                volume_similarity,
                false_negative_error,
                false_positive_error,
                false_discovery_rate,
            },
        );
    }

    let total_overlap = if den_total == 0.0 {
        REAL_TYPE_MAX
    } else {
        num_total / den_total
    };
    let union_overlap = if den_union == 0.0 {
        REAL_TYPE_MAX
    } else {
        num_union / den_union
    };
    let mean_overlap = mean_overlap_of(union_overlap);
    let volume_similarity = if den_vol == 0.0 {
        REAL_TYPE_MAX
    } else {
        2.0 * num_vol / den_vol
    };
    let false_negative_error = if den_fne == 0.0 {
        REAL_TYPE_MAX
    } else {
        num_fne / den_fne
    };
    let false_positive_error = if den_fpe == 0.0 {
        REAL_TYPE_MAX
    } else {
        num_fpe / den_fpe
    };
    let false_discovery_rate = if den_fdr == 0.0 {
        REAL_TYPE_MAX
    } else {
        num_fdr / den_fdr
    };

    Ok(OverlapMeasures {
        total_overlap,
        union_overlap,
        mean_overlap,
        volume_similarity,
        false_negative_error,
        false_positive_error,
        false_discovery_rate,
        per_label,
    })
}

// ---- HausdorffDistanceImageFilter / DirectedHausdorffDistanceImageFilter --

/// Result of [`directed_hausdorff_distance`], mirroring
/// `DirectedHausdorffDistanceImageFilter`'s two measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectedHausdorffMeasures {
    /// `GetDirectedHausdorffDistance()`: `max_{a in A} min_{b in B} |a-b|`.
    pub directed_hausdorff_distance: f64,
    /// `GetAverageHausdorffDistance()`: the mean, over `a in A`, of
    /// `max(min_{b in B} |a-b|, 0)`.
    pub average_hausdorff_distance: f64,
}

/// `DirectedHausdorffDistanceImageFilter`: the directed Hausdorff distance
/// from `image1`'s non-zero pixels ("A") to `image2`'s ("B"). Not
/// symmetric — see [`hausdorff_distance`] for the symmetric `max` of both
/// directions. Runs with `UseImageSpacing = true` unconditionally (see
/// module docs). Errors with [`FilterError::EmptyHausdorffForegroundSet`]
/// if `image1` has no non-zero pixels (`AfterThreadedGenerateData`'s
/// `"pixelcount is equal to 0"` exception).
pub fn directed_hausdorff_distance(
    image1: &Image,
    image2: &Image,
) -> Result<DirectedHausdorffMeasures> {
    require_same_size(image1, image2)?;

    // BeforeThreadedGenerateData: SignedMaurerDistanceMapImageFilter on
    // image2, SquaredDistance(false), UseImageSpacing(true),
    // InsideIsPositive left at the Maurer filter's own default (false).
    let distance_map = signed_maurer_distance_map(image2, false, false, true, 0.0)?;
    let dist_vals = distance_map.to_f64_vec();
    let vals1 = image1.to_f64_vec();

    let mut max_distance = 0.0f64;
    let mut sum = 0.0f64;
    let mut pixel_count = 0u64;
    for (&v1, &d) in vals1.iter().zip(&dist_vals) {
        if v1 != 0.0 {
            // Negative distance-map values mean "inside image2's object";
            // clamp to 0 (no penalty for overlap).
            let clamped = d.max(0.0);
            max_distance = max_distance.max(clamped);
            sum += clamped;
            pixel_count += 1;
        }
    }

    if pixel_count == 0 {
        return Err(FilterError::EmptyHausdorffForegroundSet);
    }

    Ok(DirectedHausdorffMeasures {
        directed_hausdorff_distance: max_distance,
        average_hausdorff_distance: sum / pixel_count as f64,
    })
}

/// Result of [`hausdorff_distance`], mirroring
/// `HausdorffDistanceImageFilter`'s two measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HausdorffMeasures {
    /// `GetHausdorffDistance()`: `max(directed(1,2), directed(2,1))`.
    pub hausdorff_distance: f64,
    /// `GetAverageHausdorffDistance()`: the mean of both directions'
    /// average Hausdorff distances.
    pub average_hausdorff_distance: f64,
}

/// `HausdorffDistanceImageFilter`: the symmetric Hausdorff distance between
/// `image1` and `image2`'s non-zero pixel sets. Runs
/// [`directed_hausdorff_distance`] both ways and combines them; errors if
/// either image has no non-zero pixels (see that function's docs).
pub fn hausdorff_distance(image1: &Image, image2: &Image) -> Result<HausdorffMeasures> {
    let d12 = directed_hausdorff_distance(image1, image2)?;
    let d21 = directed_hausdorff_distance(image2, image1)?;

    Ok(HausdorffMeasures {
        hausdorff_distance: d12
            .directed_hausdorff_distance
            .max(d21.directed_hausdorff_distance),
        average_hausdorff_distance: (d12.average_hausdorff_distance
            + d21.average_hausdorff_distance)
            * 0.5,
    })
}

// ---- SimilarityIndexImageFilter -------------------------------------------

/// `SimilarityIndexImageFilter`: `2 |A ∩ B| / (|A| + |B|)`, where `A` and `B`
/// are the sets of **non-zero** pixels of `image1` and `image2`.
///
/// A pixel is in `A` when `image1`'s value is not exactly zero, and in `B`
/// when `image2`'s is not exactly zero (`ThreadedGenerateData` tests
/// `it1.Get() != InputImage1PixelType{}` for the first image and
/// `Math::NotExactlyEquals(it2.Get(), InputImage2PixelType{})` for the
/// second — two spellings of the same exact `!= 0` comparison, with no
/// tolerance and no `abs`). A negative pixel is therefore *in* the set, and
/// a floating-point `NaN` is too, since `NaN != 0.0`. `-0.0` is not: it
/// compares equal to zero.
///
/// **Both-empty quirk, reproduced:** when neither image has a single
/// non-zero pixel, `AfterThreadedGenerateData` short-circuits to
/// `RealType{}`, i.e. **`0.0`** — not the `NaN` that `2*0/(0+0)` would give,
/// and not `1.0` for "two identical (empty) sets". A zero-pixel image hits
/// the same branch. Note the guard is `if (!countImage1 && !countImage2)`:
/// only *one* image being empty falls through to the division, which is then
/// `0.0 / (n + 0)` = `0.0` anyway.
///
/// The filter is a pure measurement (`no_return_image: true` in SimpleITK's
/// `SimilarityIndexImageFilter.yaml`, whose sole measurement is named
/// `SimilarityIndex`); ITK grafts input 1 through as the output image, which
/// carries no information and has no analogue here.
///
/// Errors with [`FilterError::SizeMismatch`] when the images differ in size:
/// `GenerateInputRequestedRegion` forces `image2`'s requested region to
/// `image1`'s, which the ITK pipeline rejects downstream.
pub fn similarity_index(image1: &Image, image2: &Image) -> Result<f64> {
    require_same_size(image1, image2)?;

    let vals1 = image1.to_f64_vec();
    let vals2 = image2.to_f64_vec();

    let mut count_image1 = 0u64;
    let mut count_image2 = 0u64;
    let mut count_intersection = 0u64;
    for (&v1, &v2) in vals1.iter().zip(&vals2) {
        let nonzero = v1 != 0.0;
        if nonzero {
            count_image1 += 1;
        }
        if v2 != 0.0 {
            count_image2 += 1;
            if nonzero {
                count_intersection += 1;
            }
        }
    }

    if count_image1 == 0 && count_image2 == 0 {
        return Ok(0.0);
    }

    Ok(2.0 * count_intersection as f64 / (count_image1 as f64 + count_image2 as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- label_overlap_measures ----

    #[test]
    fn identical_images_have_perfect_overlap() {
        #[rustfmt::skip]
        let img = img_u8(&[4, 1], vec![1, 1, 2, 0]);
        let m = label_overlap_measures(&img, &img).unwrap();

        assert_eq!(m.total_overlap, 1.0);
        assert_eq!(m.union_overlap, 1.0);
        assert_eq!(m.mean_overlap, 1.0); // Dice
        assert_eq!(m.volume_similarity, 0.0);
        assert_eq!(m.false_negative_error, 0.0);
        assert_eq!(m.false_positive_error, 0.0);
        assert_eq!(m.false_discovery_rate, 0.0);

        for label in [0i64, 1, 2] {
            let l = &m.per_label[&label];
            assert_eq!(l.target_overlap, 1.0, "label {label}");
            assert_eq!(l.union_overlap, 1.0, "label {label}");
            assert_eq!(l.mean_overlap, 1.0, "label {label}");
            assert_eq!(l.volume_similarity, 0.0, "label {label}");
            assert_eq!(l.false_negative_error, 0.0, "label {label}");
            assert_eq!(l.false_positive_error, 0.0, "label {label}");
            assert_eq!(l.false_discovery_rate, 0.0, "label {label}");
        }
    }

    #[test]
    fn disjoint_labels_have_zero_intersection_but_finite_denominators() {
        // source: label 1 on the left half; target: label 2 on the right
        // half. No pixel ever agrees, but every count involved is nonzero,
        // so this exercises the ordinary (non-degenerate) zero-numerator
        // path, not the f64::MAX quirk.
        let source = img_u8(&[4, 1], vec![1, 1, 0, 0]);
        let target = img_u8(&[4, 1], vec![0, 0, 2, 2]);
        let m = label_overlap_measures(&source, &target).unwrap();

        assert_eq!(m.total_overlap, 0.0);
        assert_eq!(m.union_overlap, 0.0);
        assert_eq!(m.mean_overlap, 0.0);

        let l1 = &m.per_label[&1];
        assert_eq!(l1.union_overlap, 0.0);
        assert_ne!(l1.union_overlap, REAL_TYPE_MAX);
    }

    #[test]
    fn all_background_hits_the_degenerate_denominator_quirk() {
        // No non-background label appears anywhere, so every *total*
        // accessor's denominator sum is exactly 0.0 -> f64::MAX (ITK's
        // NumericTraits<RealType>::max(), not infinity).
        let img = img_u8(&[4, 1], vec![0, 0, 0, 0]);
        let m = label_overlap_measures(&img, &img).unwrap();

        assert_eq!(m.total_overlap, REAL_TYPE_MAX);
        assert_eq!(m.union_overlap, REAL_TYPE_MAX);
        assert_eq!(m.volume_similarity, REAL_TYPE_MAX);
        assert_eq!(m.false_negative_error, REAL_TYPE_MAX);
        assert_eq!(m.false_positive_error, REAL_TYPE_MAX);
        assert_eq!(m.false_discovery_rate, REAL_TYPE_MAX);
        assert!(m.total_overlap.is_finite());
        assert_ne!(m.total_overlap, f64::INFINITY);

        // Label 0 itself: source==target==union==intersection==n, so
        // union/target/mean overlap are all 1.0 and volume_similarity is
        // 0/(2n) = 0.0 -- but false_positive_error's source==0 guard does
        // NOT fire (source == n != 0), and the actual denominator
        // (source_complement + (n_vox - union)) is 0 + (n - n) == 0, so the
        // .hxx's unguarded division yields NaN, reproduced here as-is.
        let bg = &m.per_label[&0];
        assert_eq!(bg.union_overlap, 1.0);
        assert_eq!(bg.target_overlap, 1.0);
        assert_eq!(bg.mean_overlap, 1.0);
        assert_eq!(bg.volume_similarity, 0.0);
        assert_eq!(bg.false_negative_error, 0.0);
        assert!(
            bg.false_positive_error.is_nan(),
            "expected NaN from the .hxx's unguarded 0/0, got {}",
            bg.false_positive_error
        );
    }

    #[test]
    fn empty_label_in_target_hits_the_target_overlap_quirk() {
        // Label 5 appears in source but never in target, so its target
        // count is 0 and GetTargetOverlap(5)/GetFalseNegativeError(5) must
        // return f64::MAX even though the label's union is nonzero (so
        // union_overlap does NOT hit the quirk here).
        let source = img_u8(&[4, 1], vec![5, 5, 0, 0]);
        let target = img_u8(&[4, 1], vec![0, 0, 0, 0]);
        let m = label_overlap_measures(&source, &target).unwrap();

        let l5 = &m.per_label[&5];
        assert_eq!(l5.target_overlap, REAL_TYPE_MAX);
        assert_eq!(l5.false_negative_error, REAL_TYPE_MAX);
        assert_eq!(l5.union_overlap, 0.0); // union=2, intersection=0
        assert_ne!(l5.union_overlap, REAL_TYPE_MAX);
    }

    #[test]
    fn hand_computed_totals_and_per_label_values() {
        // source: 1 1 2 2 0 0
        // target: 1 2 2 0 0 0
        let source = img_i32(&[6, 1], vec![1, 1, 2, 2, 0, 0]);
        let target = img_i32(&[6, 1], vec![1, 2, 2, 0, 0, 0]);
        let m = label_overlap_measures(&source, &target).unwrap();

        // Hand-derived from LabelOverlapLabelSetMeasures counters:
        // label 1: source=2 target=1 union=2 intersection=1 sc=1 tc=0
        // label 2: source=2 target=2 union=3 intersection=1 sc=1 tc=1
        // label 0: source=2 target=3 union=3 intersection=2 sc=0 tc=1
        // n_vox = 6
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;

        assert!(close(m.total_overlap, 2.0 / 3.0));
        assert!(close(m.union_overlap, 2.0 / 5.0));
        assert!(close(m.mean_overlap, 0.8 / 1.4));
        assert!(close(m.volume_similarity, 2.0 / 7.0));
        assert!(close(m.false_negative_error, 1.0 / 3.0));
        assert!(close(m.false_positive_error, 2.0 / 9.0));
        assert!(close(m.false_discovery_rate, 0.5));

        let l1 = &m.per_label[&1];
        assert!(close(l1.target_overlap, 1.0));
        assert!(close(l1.union_overlap, 0.5));
        assert!(close(l1.mean_overlap, 2.0 / 3.0));
        assert!(close(l1.volume_similarity, 2.0 / 3.0));
        assert!(close(l1.false_negative_error, 0.0));
        assert!(close(l1.false_positive_error, 0.2));
        assert!(close(l1.false_discovery_rate, 0.5));

        let l2 = &m.per_label[&2];
        assert!(close(l2.target_overlap, 0.5));
        assert!(close(l2.union_overlap, 1.0 / 3.0));
        assert!(close(l2.mean_overlap, 0.5));
        assert!(close(l2.volume_similarity, 0.0));
        assert!(close(l2.false_negative_error, 0.5));
        assert!(close(l2.false_positive_error, 0.25));
        assert!(close(l2.false_discovery_rate, 0.5));

        let l0 = &m.per_label[&0];
        assert!(close(l0.target_overlap, 2.0 / 3.0));
        assert!(close(l0.union_overlap, 2.0 / 3.0));
        assert!(close(l0.mean_overlap, 0.8));
        assert!(close(l0.volume_similarity, -0.4));
        assert!(close(l0.false_negative_error, 1.0 / 3.0));
        assert!(close(l0.false_positive_error, 0.0));
        assert!(close(l0.false_discovery_rate, 0.0));
    }

    #[test]
    fn requires_integer_pixel_type() {
        let f = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let i = img_u8(&[2, 1], vec![1, 2]);
        assert!(matches!(
            label_overlap_measures(&f, &i),
            Err(FilterError::RequiresIntegerPixelType(_))
        ));
        assert!(matches!(
            label_overlap_measures(&i, &f),
            Err(FilterError::RequiresIntegerPixelType(_))
        ));
    }

    #[test]
    fn size_mismatch_errors() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = img_u8(&[3, 1], vec![1, 2, 3]);
        assert!(matches!(
            label_overlap_measures(&a, &b),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    // ---- hausdorff_distance / directed_hausdorff_distance ----

    fn single_voxel(size: &[usize], on: &[usize]) -> Image {
        let n: usize = size.iter().product();
        let mut strides = vec![1usize; size.len()];
        for d in 1..size.len() {
            strides[d] = strides[d - 1] * size[d - 1];
        }
        let mut data = vec![0u8; n];
        let idx: usize = on.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
        data[idx] = 1;
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn identical_images_have_zero_hausdorff_distance() {
        let img = single_voxel(&[5, 5], &[2, 2]);
        let m = hausdorff_distance(&img, &img).unwrap();
        assert_eq!(m.hausdorff_distance, 0.0);
        assert_eq!(m.average_hausdorff_distance, 0.0);
    }

    #[test]
    fn single_voxel_pair_matches_analytic_point_distance() {
        let size = [5usize, 5];
        let a = single_voxel(&size, &[0, 0]);
        let b = single_voxel(&size, &[4, 4]);
        let expected = ((4.0f64).powi(2) + (4.0f64).powi(2)).sqrt();

        let directed = directed_hausdorff_distance(&a, &b).unwrap();
        assert!((directed.directed_hausdorff_distance - expected).abs() < 1e-9);
        assert!((directed.average_hausdorff_distance - expected).abs() < 1e-9);

        let sym = hausdorff_distance(&a, &b).unwrap();
        assert!((sym.hausdorff_distance - expected).abs() < 1e-9);
        assert!((sym.average_hausdorff_distance - expected).abs() < 1e-9);
    }

    #[test]
    fn anisotropic_spacing_scales_the_distance() {
        let size = [5usize, 5];
        let mut a = single_voxel(&size, &[0, 0]);
        let mut b = single_voxel(&size, &[4, 4]);
        a.set_spacing(&[1.0, 3.0]).unwrap();
        b.set_spacing(&[1.0, 3.0]).unwrap();
        // Physical distance uses image2's spacing inside the Maurer map:
        // dx=4*1.0, dy=4*3.0.
        let expected = ((4.0f64).powi(2) + (12.0f64).powi(2)).sqrt();

        let m = hausdorff_distance(&a, &b).unwrap();
        assert!(
            (m.hausdorff_distance - expected).abs() < 1e-9,
            "got {}, expected {expected}",
            m.hausdorff_distance
        );

        // Sanity: this must differ from the isotropic-spacing result, i.e.
        // spacing is not silently ignored.
        let iso_a = single_voxel(&size, &[0, 0]);
        let iso_b = single_voxel(&size, &[4, 4]);
        let iso = hausdorff_distance(&iso_a, &iso_b).unwrap();
        assert_ne!(m.hausdorff_distance, iso.hausdorff_distance);
    }

    #[test]
    fn directed_hausdorff_is_not_symmetric() {
        // A has two points; B coincides with one of them. A -> B's worst
        // case is the distance to the far point; B -> A is 0 (B's single
        // point sits exactly on an A point).
        let size = [10usize, 1];
        let mut data = vec![0u8; 10];
        data[0] = 1;
        data[9] = 1;
        let a = img_u8(&size, data);
        let b = single_voxel(&size, &[0]);

        let ab = directed_hausdorff_distance(&a, &b).unwrap();
        let ba = directed_hausdorff_distance(&b, &a).unwrap();
        assert_eq!(ab.directed_hausdorff_distance, 9.0);
        assert_eq!(ba.directed_hausdorff_distance, 0.0);
        assert_ne!(
            ab.directed_hausdorff_distance,
            ba.directed_hausdorff_distance
        );

        // Average over A's two points: (9 + 0) / 2 = 4.5.
        assert!((ab.average_hausdorff_distance - 4.5).abs() < 1e-9);
    }

    #[test]
    fn empty_foreground_set_errors() {
        let empty = img_u8(&[3, 1], vec![0, 0, 0]);
        let nonempty = single_voxel(&[3, 1], &[1]);
        assert!(matches!(
            directed_hausdorff_distance(&empty, &nonempty),
            Err(FilterError::EmptyHausdorffForegroundSet)
        ));
        assert!(matches!(
            hausdorff_distance(&empty, &nonempty),
            Err(FilterError::EmptyHausdorffForegroundSet)
        ));
        assert!(matches!(
            hausdorff_distance(&nonempty, &empty),
            Err(FilterError::EmptyHausdorffForegroundSet)
        ));
    }

    #[test]
    fn hausdorff_size_mismatch_errors() {
        let a = single_voxel(&[3, 1], &[0]);
        let b = single_voxel(&[4, 1], &[0]);
        assert!(matches!(
            hausdorff_distance(&a, &b),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    // ---- similarity_index ----

    /// `2|A ∩ A| / (|A| + |A|) == 1` for any image with a non-empty
    /// foreground.
    #[test]
    fn similarity_index_of_identical_images_is_one() {
        let a = img_u8(&[4, 1], vec![0, 1, 7, 0]);
        assert_eq!(similarity_index(&a, &a).unwrap(), 1.0);
    }

    /// Disjoint foregrounds: the intersection is empty, so the index is 0
    /// even though both sets are non-empty.
    #[test]
    fn similarity_index_of_disjoint_images_is_zero() {
        let a = img_u8(&[4, 1], vec![1, 1, 0, 0]);
        let b = img_u8(&[4, 1], vec![0, 0, 1, 1]);
        assert_eq!(similarity_index(&a, &b).unwrap(), 0.0);
    }

    /// Hand-derived half overlap: |A| = 4, |B| = 2, |A ∩ B| = 2, so
    /// `2 * 2 / (4 + 2) = 2/3`.
    #[test]
    fn similarity_index_half_overlap_hand_derived() {
        let a = img_u8(&[6, 1], vec![1, 1, 1, 1, 0, 0]);
        let b = img_u8(&[6, 1], vec![0, 0, 1, 1, 0, 0]);
        assert_eq!(similarity_index(&a, &b).unwrap(), 2.0 / 3.0);
    }

    /// One image empty, the other not: falls through the both-empty guard to
    /// `2 * 0 / (3 + 0)` = 0.0.
    #[test]
    fn similarity_index_with_one_empty_image_is_zero() {
        let a = img_u8(&[4, 1], vec![1, 1, 1, 0]);
        let b = img_u8(&[4, 1], vec![0, 0, 0, 0]);
        assert_eq!(similarity_index(&a, &b).unwrap(), 0.0);
        assert_eq!(similarity_index(&b, &a).unwrap(), 0.0);
    }

    /// Both images empty: ITK's `if (!countImage1 && !countImage2)` returns
    /// `RealType{}` = 0.0, not `NaN` and not 1.0.
    #[test]
    fn similarity_index_with_both_images_empty_is_zero() {
        let a = img_u8(&[4, 1], vec![0, 0, 0, 0]);
        let b = img_u8(&[4, 1], vec![0, 0, 0, 0]);
        let s = similarity_index(&a, &b).unwrap();
        assert_eq!(s, 0.0);
        assert!(!s.is_nan());
    }

    /// A zero-pixel image takes the same both-empty branch.
    #[test]
    fn similarity_index_of_empty_images_is_zero() {
        let a = Image::from_vec(&[0, 0], Vec::<u8>::new()).unwrap();
        assert_eq!(similarity_index(&a, &a).unwrap(), 0.0);
    }

    /// "Non-zero" is an exact `!= 0` test, not `> 0`: negative pixels are
    /// foreground. |A| = 2, |B| = 2, |A ∩ B| = 1 -> `2*1/4` = 0.5.
    #[test]
    fn similarity_index_counts_negative_pixels_as_foreground() {
        let a = img_i32(&[4, 1], vec![-3, -1, 0, 0]);
        let b = img_i32(&[4, 1], vec![0, 5, -2, 0]);
        assert_eq!(similarity_index(&a, &b).unwrap(), 0.5);
    }

    /// `NaN != 0.0`, so a `NaN` pixel is foreground; `-0.0 == 0.0`, so it is
    /// background. Here |A| = 1 (the NaN), |B| = 1 (the 1.0 at index 0),
    /// |A ∩ B| = 0 -> 0.0.
    #[test]
    fn similarity_index_treats_nan_as_foreground_and_negative_zero_as_background() {
        let a = Image::from_vec(&[3, 1], vec![0.0f64, f64::NAN, -0.0]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![1.0f64, 0.0, -0.0]).unwrap();
        assert_eq!(similarity_index(&a, &b).unwrap(), 0.0);
        // And a NaN shared by both images does land in the intersection.
        let c = Image::from_vec(&[3, 1], vec![0.0f64, f64::NAN, -0.0]).unwrap();
        assert_eq!(similarity_index(&a, &c).unwrap(), 1.0);
    }

    #[test]
    fn similarity_index_size_mismatch_errors() {
        let a = img_u8(&[3, 1], vec![1, 0, 0]);
        let b = img_u8(&[4, 1], vec![1, 0, 0, 0]);
        assert!(matches!(
            similarity_index(&a, &b),
            Err(FilterError::SizeMismatch { .. })
        ));
    }
}
