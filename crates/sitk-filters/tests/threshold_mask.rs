//! A **mask** must narrow the histogram of every `HistogramThresholdImageFilter`-family
//! threshold — the threshold is computed from the voxels the mask admits, not from every
//! voxel in the image.
//!
//! This is the §2.162 shape, one level up. There, a masked-out voxel sized a metric's
//! histogram *axis*; here, a masked-out voxel votes in the histogram that *picks the
//! threshold*, and the threshold decides every output pixel. ITK routes the histogram
//! through `Statistics::MaskedImageToHistogramFilter`
//! (`itkHistogramThresholdImageFilter.hxx:78-89`) and SimpleITK exposes `MaskImage` on all
//! twelve of these filters. This port accepted no mask at all until §2.173.
//!
//! # Two mask comparisons, and both are pinned
//!
//! Upstream uses **two different** mask rules inside one filter, which is exactly the kind
//! of thing a port invents a "sensible" unification for and gets wrong:
//!
//! * histogram inclusion is `mask == mask_value` (an exact equality; `mask_value` defaults
//!   to **255**), and
//! * output masking zeroes where `mask == 0` — *not* where `mask != mask_value`.
//!
//! So a voxel whose mask is neither `0` nor `mask_value` is excluded from the histogram and
//! **kept** in the output. `a_mask_value_that_is_neither_zero_nor_the_mask_value` pins that
//! asymmetry, because unifying the two rules is the obvious wrong move.

use sitk_core::Image;
use sitk_filters::{
    ThresholdMask, huang_threshold, intermodes_threshold, isodata_threshold,
    kittler_illingworth_threshold, li_threshold, maximum_entropy_threshold, moments_threshold,
    otsu_threshold, renyi_entropy_threshold, shanbhag_threshold, triangle_threshold, yen_threshold,
};

const N: usize = 32;

/// All **twelve** of the family, each as `(name, |img, mask| -> threshold)`. The mask goes to
/// one owner (`histogram::threshold_histogram`), but twelve signatures had to be threaded to
/// reach it, and a dropped argument in any one of them is invisible to a pin that names four.
/// `otsu_multiple_thresholds` is absent deliberately: neither ITK nor SimpleITK gives it a
/// mask (ledger §2.173).
type Thresholder = fn(&Image, Option<&ThresholdMask>) -> f64;
const TWELVE: [(&str, Thresholder); 12] = [
    ("otsu", |i, m| {
        otsu_threshold(i, 128, false, 1, 0, m).unwrap().1
    }),
    ("triangle", |i, m| {
        triangle_threshold(i, 128, 1, 0, m).unwrap().1
    }),
    ("huang", |i, m| huang_threshold(i, 1, 0, 128, m).unwrap().1),
    ("intermodes", |i, m| {
        intermodes_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("isodata", |i, m| {
        isodata_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("kittler_illingworth", |i, m| {
        kittler_illingworth_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("li", |i, m| li_threshold(i, 1, 0, 128, m).unwrap().1),
    ("maximum_entropy", |i, m| {
        maximum_entropy_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("moments", |i, m| {
        moments_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("renyi_entropy", |i, m| {
        renyi_entropy_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("shanbhag", |i, m| {
        shanbhag_threshold(i, 1, 0, 128, m).unwrap().1
    }),
    ("yen", |i, m| yen_threshold(i, 1, 0, 128, m).unwrap().1),
];

/// A bimodal image — a dark background and a bright square — plus a block of **outliers**
/// at an intensity far below both modes. The outliers are what the mask will exclude.
///
/// The outliers sit in rows `0..2`; the bright square is in the middle; everything else is
/// background. Counting the outliers drags the histogram's *lower* bin edge down (they set
/// the min), which re-scales every bin and moves the threshold.
///
/// They are below both modes, not above, and that is deliberate: the family binarizes
/// `v <= threshold` to `inside_value`, so an excluded voxel *above* every threshold would
/// carry `outside_value == 0` and be indistinguishable from one `MaskImageFilter` had
/// zeroed — the output-masking pins below would pass against nothing. Below the threshold,
/// an unzeroed excluded voxel reads `1` and a zeroed one reads `0`.
///
/// Every block carries a deterministic spread, and the spread is **wide enough to survive
/// the wild image's bin width**. Kittler-Illingworth fits a Gaussian to each side and refuses
/// an image outright when a side's variance is zero
/// (`ThresholdCalculatorFailed { reason: "sigma2 <= 0" }`). Flat modes do that immediately;
/// so does a spread of a few units once the outliers stretch the range to ~1200 and 128 bins
/// are ~10 wide, which collapses each mode back into a single bin. A fixture that only ten of
/// the twelve can accept is not a fixture for the family.
fn image_with_outliers(outlier: f64) -> Image {
    let mut v: Vec<f64> = (0..N * N).map(|k| 10.0 + (k % 5) as f64 * 20.0).collect();
    for j in 8..24 {
        for i in 8..24 {
            v[j * N + i] = 200.0 + ((j * N + i) % 7) as f64 * 10.0;
        }
    }
    for j in 0..2 {
        for i in 0..N {
            v[j * N + i] = outlier + ((j * N + i) % 3) as f64 * 20.0;
        }
    }
    Image::from_vec(&[N, N], v).unwrap()
}

/// Admits everything except the outlier rows. `255` is ITK's and SimpleITK's default
/// `MaskValue`, and the mask must be compared with `==`, not `!= 0`.
fn mask_excluding_outliers() -> Image {
    let mut m = vec![255.0f64; N * N];
    for j in 0..2 {
        for i in 0..N {
            m[j * N + i] = 0.0;
        }
    }
    Image::from_vec(&[N, N], m).unwrap()
}

/// **The fixture is real.** Without a mask, the outliers move the threshold — they widen
/// the histogram's range, which re-scales every bin. If this ever stops holding, every pin
/// below is vacuous and proves nothing about masking.
///
/// This assertion is written first and deliberately: it has caught a pin that could not
/// fail three times in this sweep.
#[test]
fn the_outliers_would_move_the_threshold_if_they_were_counted() {
    let tame = image_with_outliers(10.0);
    let wild = image_with_outliers(-1000.0);
    for (name, t, w) in unmasked_thresholds(&tame, &wild) {
        assert_ne!(
            t.to_bits(),
            w.to_bits(),
            "{name}: unmasked, the outliers must move the threshold — otherwise the masked \
             pin cannot fail: {t} vs {w}"
        );
    }
}

/// The four calculators, thresholded with **no** mask, on both images.
fn unmasked_thresholds(tame: &Image, wild: &Image) -> [(&'static str, f64, f64); 4] {
    [
        (
            "otsu",
            otsu_threshold(tame, 128, false, 1, 0, None).unwrap().1,
            otsu_threshold(wild, 128, false, 1, 0, None).unwrap().1,
        ),
        (
            "huang",
            huang_threshold(tame, 1, 0, 128, None).unwrap().1,
            huang_threshold(wild, 1, 0, 128, None).unwrap().1,
        ),
        (
            "li",
            li_threshold(tame, 1, 0, 128, None).unwrap().1,
            li_threshold(wild, 1, 0, 128, None).unwrap().1,
        ),
        (
            "yen",
            yen_threshold(tame, 1, 0, 128, None).unwrap().1,
            yen_threshold(wild, 1, 0, 128, None).unwrap().1,
        ),
    ]
}

/// **The pin.** With the mask, the outlier voxels must contribute *nothing* to the
/// histogram, so the threshold is the one the tame image gives — bit for bit. An excluded
/// voxel that still voted would move it.
#[test]
fn a_masked_out_voxel_does_not_vote_in_the_histogram() {
    let tame = image_with_outliers(10.0);
    let wild = image_with_outliers(-1000.0);
    let mask = mask_excluding_outliers();
    let m = ThresholdMask::new(&mask);

    // The tame image's *masked* threshold is the reference: the two images agree on every
    // admitted voxel, so an honest masked histogram cannot tell them apart.
    let cases: [(&str, f64, f64); 4] = [
        (
            "otsu",
            otsu_threshold(&tame, 128, false, 1, 0, Some(&m)).unwrap().1,
            otsu_threshold(&wild, 128, false, 1, 0, Some(&m)).unwrap().1,
        ),
        (
            "huang",
            huang_threshold(&tame, 1, 0, 128, Some(&m)).unwrap().1,
            huang_threshold(&wild, 1, 0, 128, Some(&m)).unwrap().1,
        ),
        (
            "li",
            li_threshold(&tame, 1, 0, 128, Some(&m)).unwrap().1,
            li_threshold(&wild, 1, 0, 128, Some(&m)).unwrap().1,
        ),
        (
            "yen",
            yen_threshold(&tame, 1, 0, 128, Some(&m)).unwrap().1,
            yen_threshold(&wild, 1, 0, 128, Some(&m)).unwrap().1,
        ),
    ];
    for (name, a, b) in cases {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: a masked-out voxel voted in the histogram: {a} vs {b}"
        );
    }
}

/// `MaskOutput` (ITK's and SimpleITK's default, `true`): the thresholded output is run
/// through `MaskImageFilter`, which zeroes where the mask is `0`.
#[test]
fn mask_output_zeroes_the_excluded_voxels_and_can_be_turned_off() {
    let img = image_with_outliers(-1000.0);
    let mask = mask_excluding_outliers();

    let on = ThresholdMask::new(&mask);
    let (out, _) = otsu_threshold(&img, 128, false, 1, 0, Some(&on)).unwrap();
    let vals = out.to_f64_vec().unwrap();
    for (i, &v) in vals.iter().enumerate().take(2 * N) {
        assert_eq!(
            v, 0.0,
            "voxel {i} is masked out and must be 0 in the output"
        );
    }

    // With `mask_output` off, the histogram is still masked but the output is not: the
    // outliers are below every threshold the admitted voxels can produce, so they binarize
    // to `inside_value` (the family binarizes `v <= threshold`) rather than being zeroed.
    // This is the half of the pin that can fail if `mask_output` is ignored and the output
    // is always masked.
    let off = ThresholdMask::new(&mask).with_mask_output(false);
    let (out, _) = otsu_threshold(&img, 128, false, 1, 0, Some(&off)).unwrap();
    let vals = out.to_f64_vec().unwrap();
    for (i, &v) in vals.iter().enumerate().take(2 * N) {
        assert_eq!(
            v, 1.0,
            "with mask_output off, excluded voxel {i} must keep its thresholded value"
        );
    }
}

/// **The asymmetry, pinned.** A voxel whose mask is neither `0` nor `mask_value` is
/// excluded from the histogram (inclusion is `== mask_value`) and **kept** in the output
/// (output masking zeroes only `== 0`). Unifying the two rules is the obvious wrong move,
/// and this is the test that fails when someone makes it.
#[test]
fn a_mask_value_that_is_neither_zero_nor_the_mask_value() {
    let img = image_with_outliers(-1000.0);
    // The outlier rows carry mask value 7: not admitted (7 != 255), not zeroed (7 != 0).
    let mut m = vec![255.0f64; N * N];
    for j in 0..2 {
        for i in 0..N {
            m[j * N + i] = 7.0;
        }
    }
    let mask_img = Image::from_vec(&[N, N], m).unwrap();
    let mask = ThresholdMask::new(&mask_img);

    let (out, threshold) = otsu_threshold(&img, 128, false, 1, 0, Some(&mask)).unwrap();

    // Excluded from the histogram: the threshold matches the one from a mask that zeroes
    // those same voxels, i.e. the outliers did not vote.
    let zeroing = mask_excluding_outliers();
    let (_, reference) =
        otsu_threshold(&img, 128, false, 1, 0, Some(&ThresholdMask::new(&zeroing))).unwrap();
    assert_eq!(
        threshold.to_bits(),
        reference.to_bits(),
        "mask value 7 must be excluded from the histogram, exactly as 0 is"
    );

    // Kept in the output: `MaskImageFilter` zeroes only where the mask is 0, and 7 is not 0.
    // The same voxels under the *zeroing* mask read 0 (pinned above), so this cannot pass by
    // accident: the two masks exclude the same voxels from the histogram and disagree only
    // on the output.
    let vals = out.to_f64_vec().unwrap();
    assert!(
        vals[..2 * N].iter().all(|&v| v == 1.0),
        "mask value 7 is not 0, so MaskImageFilter must NOT zero these voxels — they are \
         below the threshold and must carry inside_value"
    );
}

/// **Every one of the twelve routes through the owner.** The four pins above name four
/// calculators; the other eight took a `mask` argument that a typo could drop on the floor,
/// and no pin would have moved. So: for each of the twelve, the outliers must move the
/// unmasked threshold (the anti-vacuity half, *per calculator* — a calculator whose threshold
/// the outliers do not move proves nothing about masking) and must not move the masked one.
#[test]
fn every_one_of_the_twelve_routes_through_the_owner() {
    let tame = image_with_outliers(10.0);
    let wild = image_with_outliers(-1000.0);
    let mask = mask_excluding_outliers();
    let m = ThresholdMask::new(&mask);

    for (name, f) in TWELVE {
        let (unmasked_tame, unmasked_wild) = (f(&tame, None), f(&wild, None));
        assert_ne!(
            unmasked_tame.to_bits(),
            unmasked_wild.to_bits(),
            "{name}: unmasked, the outliers must move the threshold, or the masked half of \
             this case cannot fail: {unmasked_tame} vs {unmasked_wild}"
        );

        let (masked_tame, masked_wild) = (f(&tame, Some(&m)), f(&wild, Some(&m)));
        assert_eq!(
            masked_tame.to_bits(),
            masked_wild.to_bits(),
            "{name}: a masked-out voxel voted in the histogram — the mask is not reaching \
             this calculator: {masked_tame} vs {masked_wild}"
        );
    }
}

/// A mask that admits no voxels leaves an empty histogram. ITK does not agree with itself
/// here — eleven of its thirteen calculators throw `"Histogram is empty"`, while Otsu and
/// OtsuMultipleThresholds divide by a zero frequency, NaN out every class variance, and so
/// never take the `varBetween > maxVarBetween` branch even once: the search returns its
/// initializer and the caller gets **bin zero's upper edge**, a plausible-looking number no
/// voxel produced, with nothing raised (ledger §1.76). This port refuses uniformly, by name.
#[test]
fn a_mask_that_admits_no_voxels_is_refused_by_name() {
    let img = image_with_outliers(-1000.0);
    let empty = Image::from_vec(&[N, N], vec![0.0f64; N * N]).unwrap();
    let mask = ThresholdMask::new(&empty);

    let err = otsu_threshold(&img, 128, false, 1, 0, Some(&mask)).unwrap_err();
    assert!(
        matches!(
            err,
            sitk_filters::FilterError::MaskAdmitsNoVoxels { mask_value: 255 }
        ),
        "expected MaskAdmitsNoVoxels, got {err:?}"
    );
    let err = huang_threshold(&img, 128, 1, 0, Some(&mask)).unwrap_err();
    assert!(
        matches!(err, sitk_filters::FilterError::MaskAdmitsNoVoxels { .. }),
        "every calculator refuses the same way, unlike upstream: {err:?}"
    );
}

/// The mask must share the image's grid; a different size is an error, not a silent
/// truncation (ITK's pipeline throws "Inputs do not occupy the same physical space").
#[test]
fn a_mask_of_a_different_size_is_an_error() {
    let img = image_with_outliers(-1000.0);
    let small = Image::from_vec(&[N / 2, N / 2], vec![255.0f64; N * N / 4]).unwrap();
    let mask = ThresholdMask::new(&small);
    assert!(matches!(
        otsu_threshold(&img, 128, false, 1, 0, Some(&mask)),
        Err(sitk_filters::FilterError::SizeMismatch { .. })
    ));
}

/// **The grid, not just the size.** ITK does not sample the mask: it is a second
/// `ImageToImageFilter` input, and `VerifyInputInformation` compares origin, spacing and
/// direction before `GenerateData` runs. A same-shaped mask whose origin has been shifted
/// describes a *different* region of physical space, and reading it index-by-index would
/// silently threshold against the wrong voxels — so it is refused, exactly as the port's
/// other masked filter (`normalized_correlation`) refuses it.
#[test]
fn a_mask_on_a_different_grid_is_refused() {
    let img = image_with_outliers(-1000.0);

    // The same mask, un-shifted, must be accepted — otherwise "refused" below would only be
    // saying that this call always fails.
    let aligned = mask_excluding_outliers();
    otsu_threshold(&img, 128, false, 1, 0, Some(&ThresholdMask::new(&aligned)))
        .expect("an aligned mask must be accepted, or the refusal below proves nothing");

    let mut shifted = mask_excluding_outliers();
    shifted.set_origin(&[5.0, 0.0]).unwrap();
    assert!(matches!(
        otsu_threshold(&img, 128, false, 1, 0, Some(&ThresholdMask::new(&shifted))),
        Err(sitk_filters::FilterError::PhysicalSpaceMismatch { index: 1 })
    ));

    let mut rescaled = mask_excluding_outliers();
    rescaled.set_spacing(&[2.0, 1.0]).unwrap();
    assert!(matches!(
        otsu_threshold(&img, 128, false, 1, 0, Some(&ThresholdMask::new(&rescaled))),
        Err(sitk_filters::FilterError::PhysicalSpaceMismatch { index: 1 })
    ));
}
