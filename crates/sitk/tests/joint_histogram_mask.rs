//! A **moving mask** must narrow the moving axis of the joint-histogram MI metric — the
//! axis is sized from the voxels the mask admits, not from every voxel in the volume.
//!
//! This is the debt §2.162 booked. The moving-axis defect it records lived in
//! `MovingImage::value_range()`, which has two callers: Mattes and this metric. Fixing
//! the owner cured both, but only Mattes was pinned — so the claim that JHMI was cured
//! rested on the owner's test plus a reading of upstream, and not on a measurement of
//! JHMI itself. This file is that measurement.
//!
//! # Why the axis is JHMI's business too
//!
//! Upstream sizes `m_MovingImageTrueMin/TrueMax` over the masked voxels
//! (`itkJointHistogramMutualInformationImageToImageMetricv4.hxx:122`, the
//! `m_MovingImageMask.IsNull() || IsInsideInWorldSpace(...)` guard) and then consumes
//! that range in exactly the two places Mattes does: `ComputeJointPDFPoint` normalizes
//! every sample's moving intensity by it (`.hxx:393-395`), and the threader **rejects**
//! a sample whose moving value falls outside it
//! (`itkJointHistogramMutualInformationGetValueAndDerivativeThreader.hxx:116-122`). So a
//! masked-out voxel that widened the axis would move every sample's joint-PDF point and
//! change which samples are admitted at all — the histogram, the value and the derivative
//! are all its business, exactly as in Mattes.
//!
//! # Why the outliers sit in the outermost layer and the mask excludes two
//!
//! Identical to `mattes_mask.rs`, and for the identical reason. The mask gates *sampling*
//! by rounding a sample's continuous index to the nearest moving voxel, but the value at
//! an admitted sample is **interpolated** from that voxel's eight corners. A mask whose
//! excluded region merely touched the admitted one would let a masked-out intensity leak
//! into an admitted sample through the trilinear corners, and the test would then fail for
//! a reason that has nothing to do with the histogram axis — it would be testing the
//! interpolant, not the axis.
//!
//! The fixture leaves a guard band. The mask excludes a shell **two** voxels deep, so an
//! admitted sample rounds to an index in `[2, n−3]`, its continuous index lies in
//! `[1.5, n−2.5)`, and its eight corners lie in `[1, n−2]`. The outliers are written only
//! into layer **0** and layer `n−1`. No admitted sample can read one, at any pose used
//! here. The two volumes below therefore differ *only* in voxels no sample ever reads.

use sitk::core::Image;
use sitk::registration::JointHistogramMutualInformationMetric;
use sitk::registration::metric::{FixedSamples, MovingImage};
use sitk::transform::TranslationTransform;

const N: usize = 16;
const BINS: usize = 32;
/// ITK's `m_VarianceForJointPDFSmoothing` default.
const SMOOTHING: f64 = 1.5;
/// What the masked-out voxels carry in the "outlier" volume. Far outside the admitted
/// range, so counting it stretches every bin of the moving axis.
const OUTLIER: f64 = 1000.0;
/// What they carry in the "tame" volume — inside the admitted range, so the whole-volume
/// range and the masked range coincide.
const TAME: f64 = 0.5;

/// A blob, plus a value written into the outermost layer of the volume.
fn volume(shift: f64, outermost: f64) -> Image {
    let c = N as f64 / 2.0;
    let mut v = vec![0.0f64; N * N * N];
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let (x, y, z) = (i as f64 - c - shift, j as f64 - c, k as f64 - c);
                let edge = i == 0 || j == 0 || k == 0 || i == N - 1 || j == N - 1 || k == N - 1;
                v[(k * N + j) * N + i] = if edge {
                    outermost
                } else {
                    (-(x * x + y * y + z * z) / 32.0).exp()
                };
            }
        }
    }
    Image::from_vec(&[N, N, N], v).unwrap()
}

/// Admits the interior only: a shell **two** voxels deep is excluded, so no admitted
/// sample's interpolation stencil can reach the outermost layer. See the module docs.
fn interior_mask() -> Image {
    let mut m = vec![0.0f64; N * N * N];
    for k in 2..N - 2 {
        for j in 2..N - 2 {
            for i in 2..N - 2 {
                m[(k * N + j) * N + i] = 1.0;
            }
        }
    }
    Image::from_vec(&[N, N, N], m).unwrap()
}

fn metric(moving: &Image, mask: Option<&Image>) -> JointHistogramMutualInformationMetric {
    let fixed = FixedSamples::from_image(&volume(0.0, TAME)).unwrap();
    let mut m = MovingImage::from_image(moving).unwrap();
    if let Some(mask) = mask {
        m = m.with_moving_mask(mask).unwrap();
    }
    JointHistogramMutualInformationMetric::from_samples(fixed, m, BINS, SMOOTHING).unwrap()
}

fn poses() -> [TranslationTransform; 3] {
    [
        TranslationTransform::new(vec![0.0, 0.0, 0.0]),
        TranslationTransform::new(vec![0.7, -0.4, 0.3]),
        TranslationTransform::new(vec![-1.3, 0.9, -0.6]),
    ]
}

/// **The fixture is real.** Without a mask, the outlier volume and the tame volume give
/// *different* JHMI values — the outliers stretch the moving axis, which moves every
/// sample's joint-PDF point and so the whole smoothed histogram. If this ever stops
/// holding, the pin below is vacuous and proves nothing.
#[test]
fn the_outliers_would_move_the_axis_if_they_were_counted() {
    let outlier = metric(&volume(1.0, OUTLIER), None);
    let tame = metric(&volume(1.0, TAME), None);
    for t in poses() {
        let (a, b) = (outlier.value(&t), tame.value(&t));
        assert_ne!(
            a.to_bits(),
            b.to_bits(),
            "unmasked, the outliers must change the value — otherwise the masked pin \
             below cannot fail: {a} vs {b}"
        );
    }
}

/// **The pin.** With the mask, the two volumes are the same metric: they differ only in
/// voxels the mask excludes, and an excluded voxel must contribute *nothing* — not to the
/// samples (it never did) and not to the histogram's moving axis (it did, until §2.162).
///
/// Bit-for-bit, not to a tolerance: with the mask honoured, the axis is derived from
/// identical numbers and every admitted sample reads identical voxels, so there is no
/// rounding to allow for. Anything less than equality means an excluded voxel got in.
#[test]
fn a_masked_out_voxel_does_not_size_the_moving_axis() {
    let mask = interior_mask();
    let outlier = metric(&volume(1.0, OUTLIER), Some(&mask));
    let tame = metric(&volume(1.0, TAME), Some(&mask));
    for t in poses() {
        let (a, b) = (outlier.value(&t), tame.value(&t));
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "a masked-out voxel sized the moving histogram axis: {a} vs {b}"
        );
    }
}

/// The derivative rides the same histogram — the axis sets each sample's joint-PDF point,
/// and the derivative is the interpolated PDF's gradient at that point — so it is pinned
/// the same way.
#[test]
fn the_masked_derivative_is_the_same_derivative_too() {
    let mask = interior_mask();
    let outlier = metric(&volume(1.0, OUTLIER), Some(&mask));
    let tame = metric(&volume(1.0, TAME), Some(&mask));
    for t in poses() {
        let (a, b) = (outlier.evaluate(&t), tame.evaluate(&t));
        assert_eq!(a.valid_points, b.valid_points, "different sample sets");
        assert_eq!(a.value.to_bits(), b.value.to_bits(), "different values");
        for (k, (&da, &db)) in a.derivative.iter().zip(b.derivative.iter()).enumerate() {
            assert_eq!(
                da.to_bits(),
                db.to_bits(),
                "param {k}: a masked-out voxel reached the derivative: {da} vs {db}"
            );
        }
    }
}
