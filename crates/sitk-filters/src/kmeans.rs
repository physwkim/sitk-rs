//! `ScalarImageKmeansImageFilter`: classify every pixel of a scalar image
//! into one of `k` classes via 1-D k-means clustering of the pixel
//! intensities.
//!
//! Verified against:
//!
//! - `Modules/Segmentation/Classifiers/include/itkScalarImageKmeansImageFilter.h(.hxx)`
//! - `Modules/Numerics/Statistics/include/itkKdTreeBasedKmeansEstimator.h(.hxx)`
//!   (the estimator `ScalarImageKmeansImageFilter::GenerateData` drives)
//! - `itkEuclideanDistanceMetric.hxx`, `itkMinimumDecisionRule.cxx`,
//!   `itkDistanceToCentroidMembershipFunction.hxx` (the final classification
//!   step's distance metric and tie-breaking rule)
//! - SimpleITK's `Code/BasicFilters/yaml/ScalarImageKmeansImageFilter.yaml`
//!
//! ## Algorithm-level substitution: brute-force Lloyd's iteration for the KdTree
//!
//! `KdTreeBasedKmeansEstimator::StartOptimization` (`itkKdTreeBasedKmeansEstimator.hxx`)
//! does not implement a different clustering algorithm from textbook batch
//! k-means (Lloyd's algorithm) -- the KdTree is purely a performance
//! accelerator (Kanungo et al.'s "filtering algorithm" for exact k-means).
//! Its recursive `Filter` walks the tree pruning any candidate centroid that
//! provably cannot be the nearest one for any point in a node's bounding box
//! (via `IsFarther`), but for every point that *is* visited, it still runs
//! the exact same nearest-centroid search (`GetClosestCandidate`: linear scan
//! over all live candidates, strict `<` comparison) and accumulates the same
//! weighted-centroid sums (`m_CandidateVector[closest].WeightedCentroid` /
//! `.Size`) that a brute-force pass would. `CandidateVector::UpdateCentroids`
//! -- the actual update rule -- and the loop's termination check
//! (`StartOptimization`'s `while (true)` with the max-iteration and
//! centroid-position-change tests) do not reference the tree at all. So a
//! plain per-pixel nearest-centroid pass (this port's [`run_kmeans`]) and the
//! KdTree filtering pass compute *identical* assignments, sums, and
//! centroids at every iteration -- the tree changes only how fast the
//! assignment is computed, never what it computes. This port uses the
//! brute-force pass, which is the appropriate trade for a 1-D clustering
//! problem (no meaningful gain from spatial pruning) and lets it skip
//! reimplementing `itkKdTree`/`itkWeightedCentroidKdTreeGenerator` entirely.
//!
//! ## Effective iteration parameters
//!
//! `GenerateData` explicitly overrides the estimator's own class defaults
//! (`MaximumIteration{ 100 }` in `itkKdTreeBasedKmeansEstimator.h`) with
//! `estimator->SetMaximumIteration(200)` and
//! `estimator->SetCentroidPositionChangesThreshold(0.0)` (0.0 is also the
//! class default, but is set explicitly). Neither is exposed by
//! `ScalarImageKmeansImageFilter.yaml`, so both are hardcoded here as
//! [`MAXIMUM_ITERATION`] and an exact-zero convergence test.
//!
//! `estimator->SetUseClusterLabels` is never called (defaults to `false`),
//! so the `if (m_UseClusterLabels)` cluster-label-filling pass at the end of
//! `StartOptimization` never runs for this filter -- `GenerateData` only
//! reads `estimator->GetParameters()` (the final means) and reclassifies
//! every pixel itself via a fresh `SampleClassifierFilter`. This port
//! likewise never builds per-sample cluster labels during estimation.
//!
//! ## Off-by-one iteration count
//!
//! `StartOptimization`'s loop runs the assignment+update pass *before*
//! checking `m_CurrentIteration >= m_MaximumIteration`, and only increments
//! `m_CurrentIteration` *after* that check and a failed convergence test:
//!
//! ```text
//! CurrentIteration = 0
//! loop {
//!     pass()                                   // pass #CurrentIteration
//!     if CurrentIteration >= MaximumIteration { break }
//!     if converged() { break }
//!     CurrentIteration += 1
//! }
//! ```
//!
//! so with `MaximumIteration = 200`, passes run for `CurrentIteration =
//! 0, 1, ..., 200` -- **201** passes total if convergence is never reached,
//! not 200. [`run_kmeans`] reproduces this by running `0..=MAXIMUM_ITERATION`
//! (inclusive) and checking convergence after every pass, matching the same
//! total pass count and the same iteration at which an early convergence
//! break can occur.
//!
//! ## `GetSumOfSquaredPositionChanges` is not squared
//!
//! Despite its name and doc comment ("sum of squared difference"), this
//! method sums `m_DistanceMetric->Evaluate(previous[i], current[i])` --
//! `EuclideanDistanceMetric::Evaluate`'s two-vector overload, which returns
//! `sqrt(sum((x1[j] - x2[j])^2))`, i.e. the *unsquared* Euclidean distance
//! (`itkEuclideanDistanceMetric.hxx`). For this filter's 1-D measurement
//! vectors that is exactly `|previous[i] - current[i]|`. [`run_kmeans`]'s
//! convergence sum matches this (plain absolute differences, not squared).
//!
//! ## Frozen empty clusters
//!
//! `CandidateVector::UpdateCentroids` only overwrites a candidate's centroid
//! `if (m_Candidates[i].Size > 0)`; a class assigned zero pixels in a given
//! pass keeps its centroid from the *previous* pass unchanged (not
//! reinitialized, not dropped). If a class never attracts any pixel across
//! every pass, its final mean is exactly its initial mean.
//!
//! ## Tie-breaking: lowest class index wins, in both phases
//!
//! `KdTreeBasedKmeansEstimator::GetClosestCandidate` seeds `closest = 0` and
//! only updates on a strict `<`, so among equidistant candidates the
//! lowest-indexed one wins. The final classification step's
//! `MinimumDecisionRule::Evaluate` (`itkMinimumDecisionRule.cxx`) does the
//! same (`discriminantScores[i] < min`, seeded from index 0), operating on
//! `DistanceToCentroidMembershipFunction::Evaluate`'s plain Euclidean
//! distance to each final mean (`itkDistanceToCentroidMembershipFunction.hxx`).
//! [`nearest_mean_index`] is shared by both phases and reproduces this exact
//! tie rule.
//!
//! ## Empty `class_with_initial_mean`
//!
//! Raw ITK's `VerifyPreconditions` throws when `m_InitialMeans.empty()`, but
//! SimpleITK's generated wrapper never lets that happen:
//! `ScalarImageKmeansImageFilter.yaml`'s `custom_itk_cast` for
//! `ClassWithInitialMean` adds `ZeroValue()` and `OneValue()` of the input
//! pixel type (i.e. plain `0.0`/`1.0`) as the two initial means whenever the
//! caller passes an empty list, and otherwise passes the caller's list
//! through unmodified (uncast -- not quantized to the input pixel type).
//! This port -- ported at the SimpleITK level -- reproduces exactly that
//! substitution, making the raw ITK-level empty-list exception unreachable
//! through [`scalar_image_kmeans`].
//!
//! ## `UseNonContiguousLabels` label spacing
//!
//! `GenerateData` spaces output labels by
//! `labelInterval = NumericTraits<OutputPixelType>::max() / numberOfClasses - 1`
//! (`unsigned int` arithmetic; `OutputPixelType` is always `uint8_t` per the
//! yaml, so this is literally `255u32 / k - 1`) when
//! `UseNonContiguousLabels` is set, or `1` (contiguous `0, 1, 2, ...`)
//! otherwise; class `i`'s label is then `i * labelInterval`, accumulated the
//! same way upstream does (`label = 0; classLabels[k] = label; label +=
//! labelInterval;`) via wrapping `u32` arithmetic to reproduce C++'s defined
//! unsigned-integer overflow, then narrowed to `u8` (`static_cast`-style
//! truncation, matching `ImageRegionIterator<uint8_t>::Set` receiving an
//! `unsigned int` class label). `numberOfClasses > 255` underflows
//! `labelInterval` to `u32::MAX`, an upstream footgun this port reproduces
//! rather than guards against -- the class's own doxygen comment already
//! assumes "less than 256 classes".
//!
//! ## `SetImageRegion` is not ported
//!
//! `ScalarImageKmeansImageFilter::SetImageRegion` lets raw ITK restrict
//! classification to a sub-region (labeling pixels outside it
//! `numberOfClasses` or `labelInterval * numberOfClasses`).
//! `ScalarImageKmeansImageFilter.yaml` exposes no member or custom method
//! for it, so SimpleITK's generated wrapper never calls it --
//! `m_ImageRegionDefined` is always `false` through this API, and the
//! region-restriction branch is dead code from the SimpleITK entry point
//! this module ports. Not implemented here.
//!
//! Output pixel type is always `UInt8` (`output_pixel_type: uint8_t` in the
//! yaml), regardless of the input's pixel type.

use crate::error::Result;
use sitk_core::Image;

/// `estimator->SetMaximumIteration(200)` in `GenerateData` -- see the module
/// doc's "Effective iteration parameters".
const MAXIMUM_ITERATION: u32 = 200;

/// Result of [`scalar_image_kmeans`]: the labeled image plus the converged
/// per-class means (`GetFinalMeans()` in `ScalarImageKmeansImageFilter.yaml`'s
/// `measurements`).
#[derive(Clone, Debug, PartialEq)]
pub struct KmeansResult {
    pub image: Image,
    pub final_means: Vec<f64>,
}

/// Nearest-centroid assignment shared by the k-means estimation pass and the
/// final per-pixel classification -- see the module doc's "Tie-breaking".
fn nearest_mean_index(value: f64, means: &[f64]) -> usize {
    let mut best = 0;
    let mut best_dist = (value - means[0]).abs();
    for (i, &mean) in means.iter().enumerate().skip(1) {
        let dist = (value - mean).abs();
        if dist < best_dist {
            best = i;
            best_dist = dist;
        }
    }
    best
}

/// `KdTreeBasedKmeansEstimator::StartOptimization`'s Lloyd loop -- see the
/// module doc's "Algorithm-level substitution", "Off-by-one iteration
/// count", "`GetSumOfSquaredPositionChanges` is not squared", and "Frozen
/// empty clusters".
fn run_kmeans(values: &[f64], initial_means: &[f64]) -> Vec<f64> {
    let k = initial_means.len();
    let mut means = initial_means.to_vec();

    for _pass in 0..=MAXIMUM_ITERATION {
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0u64; k];
        for &v in values {
            let idx = nearest_mean_index(v, &means);
            sums[idx] += v;
            counts[idx] += 1;
        }

        let mut new_means = means.clone();
        for i in 0..k {
            if counts[i] > 0 {
                new_means[i] = sums[i] / counts[i] as f64;
            }
        }

        let change: f64 = means
            .iter()
            .zip(&new_means)
            .map(|(a, b)| (a - b).abs())
            .sum();
        means = new_means;
        if change <= 0.0 {
            break;
        }
    }

    means
}

/// `ScalarImageKmeansImageFilter`: classify every pixel of `img` into one of
/// `class_with_initial_mean.len()` classes (or 2, if empty -- see the module
/// doc) via 1-D k-means clustering of pixel intensities, then label each
/// pixel by its nearest final mean. See the module doc for every quirk
/// reproduced here.
pub fn scalar_image_kmeans(
    img: &Image,
    class_with_initial_mean: &[f64],
    use_non_contiguous_labels: bool,
) -> Result<KmeansResult> {
    let initial_means: Vec<f64> = if class_with_initial_mean.is_empty() {
        vec![0.0, 1.0]
    } else {
        class_with_initial_mean.to_vec()
    };
    let k = initial_means.len();

    let values = img.to_f64_vec()?;
    let final_means = run_kmeans(&values, &initial_means);

    let label_interval: u32 = if use_non_contiguous_labels {
        (u8::MAX as u32 / k as u32).wrapping_sub(1)
    } else {
        1
    };

    let mut class_labels = vec![0u8; k];
    let mut label: u32 = 0;
    for slot in class_labels.iter_mut() {
        *slot = label as u8;
        label = label.wrapping_add(label_interval);
    }

    let out: Vec<u8> = values
        .iter()
        .map(|&v| class_labels[nearest_mean_index(v, &final_means)])
        .collect();

    let mut result_img = Image::from_vec(img.size(), out)?;
    result_img.copy_geometry_from(img);

    Ok(KmeansResult {
        image: result_img,
        final_means,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    fn img_f64(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn separates_three_well_spaced_clusters_with_hand_placed_means() {
        // Three tight clusters around 0, 50, 100; initial means placed
        // exactly on the cluster centers so this converges immediately.
        let mut vals = vec![0.0, 1.0, -1.0];
        vals.extend([50.0, 49.0, 51.0]);
        vals.extend([100.0, 99.0, 101.0]);
        let n = vals.len();
        let img = img_f64(&[n, 1], vals);

        let result = scalar_image_kmeans(&img, &[0.0, 50.0, 100.0], false).unwrap();
        assert_eq!(result.image.pixel_id(), PixelId::UInt8);
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(&labels[0..3], &[0, 0, 0]);
        assert_eq!(&labels[3..6], &[1, 1, 1]);
        assert_eq!(&labels[6..9], &[2, 2, 2]);

        assert_eq!(result.final_means.len(), 3);
        assert!((result.final_means[0] - 0.0).abs() < 1e-9);
        assert!((result.final_means[1] - 50.0).abs() < 1e-9);
        assert!((result.final_means[2] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn separates_three_clusters_from_offset_initial_means() {
        // Same three clusters, but initial means start away from the true
        // centers so this exercises several Lloyd passes, not just a
        // first-pass fixed point.
        let mut vals = vec![0.0, 2.0, -2.0, 1.0];
        vals.extend([50.0, 48.0, 52.0, 51.0]);
        vals.extend([100.0, 98.0, 102.0, 99.0]);
        let n = vals.len();
        let img = img_f64(&[n, 1], vals);

        let result = scalar_image_kmeans(&img, &[10.0, 40.0, 90.0], false).unwrap();
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(&labels[0..4], &[0, 0, 0, 0]);
        assert_eq!(&labels[4..8], &[1, 1, 1, 1]);
        assert_eq!(&labels[8..12], &[2, 2, 2, 2]);
        assert!((result.final_means[0] - 0.25).abs() < 1e-9);
        assert!((result.final_means[1] - 50.25).abs() < 1e-9);
        assert!((result.final_means[2] - 99.75).abs() < 1e-9);
    }

    #[test]
    fn non_contiguous_labels_are_spaced_by_255_over_k_minus_1() {
        // k=3: label_interval = 255/3 - 1 = 84; labels = 0, 84, 168.
        let vals = vec![0.0, 50.0, 100.0];
        let img = img_f64(&[3, 1], vals);
        let result = scalar_image_kmeans(&img, &[0.0, 50.0, 100.0], true).unwrap();
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(labels, &[0, 84, 168]);
    }

    #[test]
    fn contiguous_labels_are_0_1_2() {
        let vals = vec![0.0, 50.0, 100.0];
        let img = img_f64(&[3, 1], vals);
        let result = scalar_image_kmeans(&img, &[0.0, 50.0, 100.0], false).unwrap();
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(labels, &[0, 1, 2]);
    }

    #[test]
    fn empty_initial_means_defaults_to_zero_and_one() {
        // Mirrors SimpleITK's custom_itk_cast: an empty ClassWithInitialMean
        // becomes [0.0, 1.0], not a precondition error.
        let vals = vec![0.0, 0.1, 0.9, 1.0];
        let img = img_f64(&[4, 1], vals);
        let result = scalar_image_kmeans(&img, &[], false).unwrap();
        assert_eq!(result.final_means.len(), 2);
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(labels, &[0, 0, 1, 1]);
    }

    #[test]
    fn single_class_labels_every_pixel_zero() {
        let vals = vec![-5.0, 0.0, 5.0, 1000.0];
        let img = img_f64(&[4, 1], vals);
        let result = scalar_image_kmeans(&img, &[3.0], false).unwrap();
        assert_eq!(result.final_means.len(), 1);
        // The sole class's mean converges to the overall image mean.
        assert!((result.final_means[0] - 250.0).abs() < 1e-9);
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(labels, &[0, 0, 0, 0]);
    }

    #[test]
    fn tie_breaks_to_the_lower_class_index_at_final_classification() {
        // Constructed so k-means converges in exactly one pass to the
        // initial means themselves: cluster 0 = {5, 15} (mean 10), cluster 1
        // = {20, 20} (mean 20). Value 15 sits exactly midway between the
        // *converged* means 10 and 20, so the final classification pass (not
        // just an intermediate iteration) must resolve that tie to the
        // lower class index.
        let vals = vec![5.0, 15.0, 20.0, 20.0];
        let img = img_f64(&[4, 1], vals);
        let result = scalar_image_kmeans(&img, &[10.0, 20.0], false).unwrap();
        assert_eq!(result.final_means, vec![10.0, 20.0]);
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(labels, &[0, 0, 1, 1]);
    }

    #[test]
    fn empty_cluster_keeps_its_initial_mean() {
        // Every pixel is far closer to mean 0 than to mean 1000; class 1
        // never attracts a single pixel across any pass, so its final mean
        // stays exactly its initial value.
        let vals = vec![0.0, 1.0, -1.0, 2.0];
        let img = img_f64(&[4, 1], vals);
        let result = scalar_image_kmeans(&img, &[0.0, 1000.0], false).unwrap();
        assert_eq!(result.final_means[1], 1000.0);
        let labels = result.image.scalar_slice::<u8>().unwrap();
        assert!(labels.iter().all(|&v| v == 0));
    }

    #[test]
    fn output_pixel_type_is_always_uint8() {
        let img = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let result = scalar_image_kmeans(&img, &[1.0, 2.0], false).unwrap();
        assert_eq!(result.image.pixel_id(), PixelId::UInt8);
    }
}
