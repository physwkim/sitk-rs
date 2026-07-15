//! `connected_component`'s optional `MaskImage` — and the **polarity** it does *not* share
//! with the threshold family's mask.
//!
//! ITK implements this mask by running the input through `MaskImageFilter` before it labels
//! anything (`itkConnectedComponentImageFilter.hxx:79-92`). `MaskImageFilter` keeps a voxel
//! where `mask != m_MaskingValue` and replaces it with the outside value (`0`) where it
//! *equals* it — and `m_MaskingValue` defaults to `TMask{}`, i.e. **`0`**
//! (`itkMaskImageFilter.h:55`, `:107`). So a masked-out voxel reaches the labeler as a zero:
//! as background.
//!
//! Two conventions now live in this port because two live in ITK:
//!
//! | filter | upstream class | rule | default |
//! |---|---|---|---|
//! | `connected_component` | `MaskImageFilter` | *exclude* where `mask == masking_value` | `0` |
//! | the twelve thresholds | `MaskedImageToHistogramFilter` | *admit* where `mask == mask_value` | `255` |
//!
//! `a_mask_of_ones_keeps_everything_here_and_admits_nothing_there` puts both on the same mask
//! and asserts they disagree, so that "fixing" them into one rule fails a test rather than a
//! user's image (ledger §2.175).

use sitk_core::Image;
use sitk_filters::{FilterError, ThresholdMask, connected_component, yen_threshold};

const W: usize = 5;
const H: usize = 3;

/// Two 2×3 blocks joined by a **single bridge voxel** at `(2, 1)`:
///
/// ```text
///   1 1 . 1 1
///   1 1 B 1 1      B = the bridge
///   1 1 . 1 1
/// ```
///
/// Face-connected, this is **one** component — but only through `B`. Remove `B` and it is
/// two. That is the whole point of the fixture: a mask that only removed background voxels
/// could not tell a working mask from an ignored one.
fn bridged() -> Image {
    let mut v = vec![1u8; W * H];
    v[at(2, 0)] = 0;
    v[at(2, 2)] = 0;
    Image::from_vec(&[W, H], v).unwrap()
}

/// Flat index of `(x, y)`, first index fastest.
fn at(x: usize, y: usize) -> usize {
    y * W + x
}

/// A `UInt8` mask of `fill` everywhere, `0` at the bridge.
fn mask_cutting_the_bridge(fill: u8) -> Image {
    let mut m = vec![fill; W * H];
    m[at(2, 1)] = 0;
    Image::from_vec(&[W, H], m).unwrap()
}

fn labels(img: &Image) -> Vec<u32> {
    img.scalar_slice::<u32>().unwrap().to_vec()
}

fn object_count(img: &Image) -> u32 {
    labels(img).into_iter().max().unwrap()
}

/// **The anti-vacuity pin.** The masked-out voxel must *change the labeling*: unmasked the
/// bridge joins the two blocks into one component, masked it is background and there are two.
/// If the mask were ignored, both runs would give one label and every other assertion in this
/// file about "the mask excluded a voxel" would be satisfiable by doing nothing.
#[test]
fn the_mask_splits_a_component_that_is_one_component_without_it() {
    let img = bridged();

    let unmasked = connected_component(&img, None, false).unwrap();
    assert_eq!(
        object_count(&unmasked),
        1,
        "the bridge must join the blocks when nothing is masked: {:?}",
        labels(&unmasked)
    );

    let mask = mask_cutting_the_bridge(1);
    let masked = connected_component(&img, Some(&mask), false).unwrap();
    assert_eq!(
        object_count(&masked),
        2,
        "masking the bridge must split the component in two: {:?}",
        labels(&masked)
    );

    // The bridge itself is background in the masked run — `MaskImageFilter` zeroed it before
    // the labeler saw it — and the two halves carry different labels, numbered in raster
    // order of first appearance.
    let out = labels(&masked);
    assert_eq!(out[at(2, 1)], 0, "the masked-out voxel must be background");
    assert_eq!(out[at(0, 1)], 1, "the left block is seen first");
    assert_eq!(out[at(4, 1)], 2, "the right block is seen second");
}

/// **The polarity asymmetry, pinned across both filters at once.** The identical mask image —
/// all `1`s — is a *no-op* for `connected_component` (`MaskImageFilter` excludes only where
/// the mask is `0`) and admits **not one voxel** to a threshold's histogram (which admits only
/// where the mask is `255`). Anyone who unifies the two conventions breaks exactly one of
/// these two assertions.
#[test]
fn a_mask_of_ones_keeps_everything_here_and_admits_nothing_there() {
    let img = bridged();
    let ones = Image::from_vec(&[W, H], vec![1u8; W * H]).unwrap();

    let unmasked = connected_component(&img, None, false).unwrap();
    let with_ones = connected_component(&img, Some(&ones), false).unwrap();
    assert_eq!(
        labels(&with_ones),
        labels(&unmasked),
        "`mask == 1` excludes nothing in connected_component: only `mask == 0` does"
    );

    // The same mask, to the other convention: `mask_value` defaults to 255, inclusion is
    // `mask == mask_value`, so a mask of 1s selects the empty set and the port refuses by name.
    assert!(
        matches!(
            yen_threshold(&img, 1, 0, 128, Some(&ThresholdMask::new(&ones))),
            Err(FilterError::MaskAdmitsNoVoxels { mask_value: 255 })
        ),
        "the threshold family admits on `mask == 255`, so a mask of 1s admits nothing"
    );
}

/// The other half of the same polarity: every nonzero value keeps, not just `255`. A mask of
/// `7`s must be as inert as a mask of `255`s — `MaskImageFilter` tests `!= masking_value`, it
/// does not test `== 255`.
#[test]
fn every_nonzero_mask_value_keeps_the_voxel() {
    let img = bridged();
    let unmasked = labels(&connected_component(&img, None, false).unwrap());

    for fill in [1u8, 7, 128, 255] {
        let mask = Image::from_vec(&[W, H], vec![fill; W * H]).unwrap();
        let out = connected_component(&img, Some(&mask), false).unwrap();
        assert_eq!(
            labels(&out),
            unmasked,
            "a mask of {fill}s must keep every voxel: only 0 excludes"
        );
    }

    // ...and a mask of 0s excludes every voxel, so nothing is labeled at all.
    let zeros = Image::from_vec(&[W, H], vec![0u8; W * H]).unwrap();
    let out = connected_component(&img, Some(&zeros), false).unwrap();
    assert!(
        labels(&out).iter().all(|&l| l == 0),
        "a mask of 0s excludes every voxel: {:?}",
        labels(&out)
    );
}

/// Masking out a whole component removes it from the numbering — the surviving objects are
/// still consecutive from 1, because `CreateConsecutive` numbers what is left.
#[test]
fn masking_out_a_whole_component_renumbers_the_rest() {
    let img = bridged();
    let mut m = vec![255u8; W * H];
    for y in 0..H {
        m[at(2, y)] = 0; // the bridge column
        m[at(3, y)] = 0; // and the right block
        m[at(4, y)] = 0;
    }
    let mask = Image::from_vec(&[W, H], m).unwrap();

    let out = connected_component(&img, Some(&mask), false).unwrap();
    assert_eq!(object_count(&out), 1, "only the left block survives");
    let out = labels(&out);
    assert_eq!(out[at(0, 1)], 1, "and it is label 1, not label 2");
    assert!(
        out[at(3, 1)] == 0 && out[at(4, 1)] == 0,
        "the masked block is background"
    );
}

/// SimpleITK fixes the mask template to `itk::Image<uint8_t, Dim>` and feeds it through
/// `CastImageToITK` — a `dynamic_cast`, not a value cast — so a `UInt16` mask throws upstream.
/// Refused by name here, not silently quantized.
#[test]
fn a_mask_that_is_not_uint8_is_refused_by_name() {
    let img = bridged();
    let mask = Image::from_vec(&[W, H], vec![1u16; W * H]).unwrap();
    assert!(matches!(
        connected_component(&img, Some(&mask), false),
        Err(FilterError::RequiresUInt8MaskPixelType(
            sitk_core::PixelId::UInt16
        ))
    ));
}

/// A mask of the wrong size cannot be iterated with the image.
#[test]
fn a_mask_of_a_different_size_is_an_error() {
    let img = bridged();
    let mask = Image::from_vec(&[W, H + 1], vec![1u8; W * (H + 1)]).unwrap();
    assert!(matches!(
        connected_component(&img, Some(&mask), false),
        Err(FilterError::SizeMismatch { .. })
    ));
}

/// The mask is a pipeline *input*, so `ImageToImageFilter::VerifyInputInformation` compares
/// its origin / spacing / direction against the image's and throws "Inputs do not occupy the
/// same physical space!" on a mismatch. Same grid or refuse — ITK never resamples a mask.
#[test]
fn a_mask_on_a_different_grid_is_refused() {
    let img = bridged();

    // The aligned mask must be *accepted*, or "refused" below would only be saying that this
    // call always fails.
    let aligned = mask_cutting_the_bridge(1);
    connected_component(&img, Some(&aligned), false)
        .expect("an aligned mask must be accepted, or the refusal below proves nothing");

    let mut shifted = mask_cutting_the_bridge(1);
    shifted.set_origin(&[5.0, 0.0]).unwrap();
    assert!(matches!(
        connected_component(&img, Some(&shifted), false),
        Err(FilterError::PhysicalSpaceMismatch { index: 1 })
    ));

    let mut rescaled = mask_cutting_the_bridge(1);
    rescaled.set_spacing(&[2.0, 1.0]).unwrap();
    assert!(matches!(
        connected_component(&img, Some(&rescaled), false),
        Err(FilterError::PhysicalSpaceMismatch { index: 1 })
    ));
}
