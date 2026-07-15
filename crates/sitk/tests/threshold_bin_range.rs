//! An **8-bit** image's histogram thresholds are computed against the *pixel type's*
//! range, not the data's — `HistogramThresholdImageFilter`'s constructor turns
//! `AutoMinimumMaximum` off for `char` / `signed char` / `unsigned char` and leaves it on
//! for everything else (`itkHistogramThresholdImageFilter.hxx:44-53`), and with it off
//! `ImageToHistogramFilter` bins over `[NonpositiveMin() - 0.5, max() + 0.5]` with no
//! marginal scale (`itkImageToHistogramFilter.hxx:155-175`).
//!
//! So for `UInt8` the bins span `[-0.5, 255.5]` — 256 wide — however narrow the data is.
//! With the SimpleITK-default 128 bins that is **exactly 2 grey levels per bin**, so every
//! threshold the family can return lands on the grid `t + 0.5 ∈ {0, 1, ..., 256}`: Otsu on
//! the even points (it returns a bin's upper edge, `GetMaxs`), the other ten on the odd ones
//! (they return the midpoint, `GetMeasurement`). That grid is the fingerprint of the type
//! range, and it is what these tests assert — "the threshold moved" alone would not
//! distinguish the rule from a rounding change.
//!
//! **This changes what every 8-bit caller of the twelve already got** — see ledger §2.174.
//! It is not a mask feature: it applies masked and unmasked alike.

use sitk::core::Image;
use sitk::filters::{ThresholdMask, otsu_multiple_thresholds, otsu_threshold, yen_threshold};

const N: usize = 32;

/// A bimodal image whose data occupies a **narrow** part of the 8-bit range: background
/// around 100, a bright square around 135. Binned over the data's `[100, 140]` the bins are
/// ~0.3 grey levels wide; binned over the type's `[-0.5, 255.5]` they are 2.0. The two
/// cannot agree, which is the point.
fn narrow_u8(mask_rows: bool) -> Vec<u8> {
    let mut v: Vec<u8> = (0..N * N).map(|k| 100 + (k % 5) as u8 * 2).collect();
    for j in 8..24 {
        for i in 8..24 {
            v[j * N + i] = 130 + ((j * N + i) % 5) as u8 * 2;
        }
    }
    if mask_rows {
        for j in 0..2 {
            for i in 0..N {
                v[j * N + i] = 250;
            }
        }
    }
    v
}

fn u8_image() -> Image {
    Image::from_vec(&[N, N], narrow_u8(false)).unwrap()
}

/// The identical *values*, as `Float64` — which takes the auto path, i.e. exactly the rule
/// this port applied to 8-bit images before §2.174.
fn same_values_as_f64() -> Image {
    let v: Vec<f64> = narrow_u8(false).iter().map(|&x| f64::from(x)).collect();
    Image::from_vec(&[N, N], v).unwrap()
}

/// 128 bins over `[-0.5, 255.5]` are exactly 2.0 wide, so a bin **edge** is `-0.5 + 2k` and
/// a bin **midpoint** is `0.5 + 2k`. Together: `t + 0.5` is a whole number in `[0, 256]`.
///
/// Both conventions are in play, and that is upstream's doing: Otsu returns
/// `histogram->GetMaxs()[0][idx]` — the upper edge — while the other ten return
/// `histogram->GetMeasurement(idx, 0)` — the midpoint (`itkYenThresholdCalculator.hxx:88`
/// and its nine siblings). A data-range histogram of this fixture has bins ~0.31 wide, so
/// it cannot land on this grid except by accident.
fn on_the_uint8_type_grid(t: f64) -> bool {
    let k = t + 0.5;
    k.fract() == 0.0 && (0.0..=256.0).contains(&k)
}

/// The stricter form, for Otsu alone: an *edge*, so `k` is even.
fn on_a_uint8_type_bin_edge(t: f64) -> bool {
    on_the_uint8_type_grid(t) && ((t + 0.5) / 2.0).fract() == 0.0
}

/// **The divergence, and the anti-vacuity assertion in one.** The same pixel values
/// threshold *differently* depending on whether they are `UInt8` or `Float64`, because ITK
/// changes the bin range on the pixel type. If these ever agree, every assertion in this
/// file is vacuous — and the port would have silently reverted to binning 8-bit data over
/// its own range.
#[test]
fn the_pixel_type_changes_the_threshold_of_identical_values() {
    for (name, as_u8, as_f64) in [
        (
            "otsu",
            otsu_threshold(&u8_image(), 128, false, 1, 0, None)
                .unwrap()
                .1,
            otsu_threshold(&same_values_as_f64(), 128, false, 1, 0, None)
                .unwrap()
                .1,
        ),
        (
            "yen",
            yen_threshold(&u8_image(), 1, 0, 128, None).unwrap().1,
            yen_threshold(&same_values_as_f64(), 1, 0, 128, None)
                .unwrap()
                .1,
        ),
    ] {
        assert_ne!(
            as_u8.to_bits(),
            as_f64.to_bits(),
            "{name}: the same values as UInt8 and as Float64 must NOT threshold alike — \
             upstream bins the first over the type range and the second over the data \
             range: {as_u8} vs {as_f64}"
        );
        assert!(
            on_the_uint8_type_grid(as_u8),
            "{name}: a UInt8 threshold must sit on the [-0.5, 255.5]/128 grid, got {as_u8}"
        );
        assert!(
            !on_the_uint8_type_grid(as_f64),
            "{name}: the Float64 threshold must come off the data range, not the type \
             grid, got {as_f64}"
        );
    }
}

/// `Int8` takes the same branch with its own range: `[-128.5, 127.5]`, also 256 wide.
#[test]
fn an_int8_image_bins_over_the_int8_type_range() {
    let v: Vec<i8> = narrow_u8(false)
        .iter()
        .map(|&x| (x as i16 - 120) as i8)
        .collect();
    let img = Image::from_vec(&[N, N], v).unwrap();
    let t = otsu_threshold(&img, 128, false, 1, 0, None).unwrap().1;

    // 128 bins over [-128.5, 127.5] are 2.0 wide: every edge is -128.5 + 2k.
    let k = (t + 128.5) / 2.0;
    assert!(
        k.fract() == 0.0 && (0.0..=128.0).contains(&k),
        "an Int8 threshold must be a bin edge of [-128.5, 127.5]/128, got {t}"
    );
    // Anti-vacuity: the same values as Float64 do *not* land on that grid, so this is the
    // type range talking and not an arithmetic coincidence.
    let as_f64: Vec<f64> = narrow_u8(false)
        .iter()
        .map(|&x| f64::from(x as i16 - 120))
        .collect();
    let f = otsu_threshold(
        &Image::from_vec(&[N, N], as_f64).unwrap(),
        128,
        false,
        1,
        0,
        None,
    )
    .unwrap()
    .1;
    assert!(
        ((f + 128.5) / 2.0).fract() != 0.0,
        "the Float64 twin must come off the data range, got {f}"
    );
}

/// **The mask does not scope the bin range on an 8-bit image**, which is the half of this
/// that reading the flag's name would get backwards. `MaskedImageToHistogramFilter`
/// overrides `ThreadedComputeMinimumAndMaximum`, but that scan runs only inside the
/// `AutoMinimumMaximum` branch — so on `UInt8` the mask decides which voxels are *counted*
/// and has no say in the range.
#[test]
fn a_mask_selects_voxels_but_not_the_bin_range_on_an_eight_bit_image() {
    let img = Image::from_vec(&[N, N], narrow_u8(true)).unwrap();
    let mut m = vec![255u8; N * N];
    for j in 0..2 {
        for i in 0..N {
            m[j * N + i] = 0;
        }
    }
    let mask_img = Image::from_vec(&[N, N], m).unwrap();
    let mask = ThresholdMask::new(&mask_img);

    let unmasked = otsu_threshold(&img, 128, false, 1, 0, None).unwrap().1;
    let masked = otsu_threshold(&img, 128, false, 1, 0, Some(&mask))
        .unwrap()
        .1;

    // The mask is not inert: excluding the 250-valued rows moves the threshold. Without
    // this, the assertion below would hold for a mask the code ignored entirely.
    assert_ne!(
        unmasked.to_bits(),
        masked.to_bits(),
        "the mask must still change which voxels are counted: {unmasked} vs {masked}"
    );

    // ...but both thresholds still sit on the *type* lattice: the mask never re-scaled a bin.
    assert!(
        on_a_uint8_type_bin_edge(unmasked) && on_a_uint8_type_bin_edge(masked),
        "a mask must not re-scale an 8-bit histogram's bins: {unmasked}, {masked}"
    );
}

/// **The population boundary, and it is ITK disagreeing with itself.**
/// `OtsuMultipleThresholdsImageFilter` is a different class: it builds its histogram
/// through `ScalarImageToHistogramGenerator` → `SampleToHistogramFilter`, whose constructor
/// sets `AutoMinimumMaximum = true` unconditionally with no pixel-type branch
/// (`itkSampleToHistogramFilter.hxx:35`), and never turns it off. So it bins an 8-bit image
/// over the **data** range while `OtsuThresholdImageFilter` bins it over the **type** range
/// — same algorithm, same image, one threshold, two answers. Reproduced, not reconciled
/// (ledger §2.174).
#[test]
fn otsu_multiple_thresholds_still_bins_an_eight_bit_image_over_the_data_range() {
    let img = u8_image();
    let single = otsu_threshold(&img, 128, false, 1, 0, None).unwrap().1;
    let multiple = otsu_multiple_thresholds(&img, 1, 128, false, false, 0)
        .unwrap()
        .1;
    assert_eq!(multiple.len(), 1);

    assert!(
        on_a_uint8_type_bin_edge(single),
        "OtsuThresholdImageFilter bins over the type range: {single}"
    );
    assert!(
        !on_the_uint8_type_grid(multiple[0]),
        "OtsuMultipleThresholdsImageFilter must still bin over the data range — it reaches \
         SampleToHistogramFilter, which has no pixel-type branch: {}",
        multiple[0]
    );
    assert_ne!(
        single.to_bits(),
        multiple[0].to_bits(),
        "ITK's two Otsu implementations disagree on an 8-bit image, and this port \
         reproduces both: {single} vs {}",
        multiple[0]
    );
}
