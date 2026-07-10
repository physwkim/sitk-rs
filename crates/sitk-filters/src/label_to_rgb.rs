//! `LabelToRGBImageFilter`, ported from
//! `Modules/Filtering/ImageFusion/include/`'s `itkLabelToRGBFunctor.h` and
//! `itkLabelToRGBImageFilter.h`/`.hxx`, plus SimpleITK's shared colormap
//! parsing (`sitkLabelFunctorUtils.hxx`'s `SetLabelFunctorFromColormap`).
//!
//! [`build_color_table`] and [`label_color_indices`] are also shared by
//! `LabelOverlayImageFilter` (`crate::label_overlay`), which uses the same
//! default palette and background/index lookup but blends into a base image
//! instead of filling a fixed `uint8_t` output.
//!
//! ## Upstream findings
//!
//! 1. **`LabelToRGBImageFilter`'s background color is always black, despite
//!    the docstring.** `itkLabelToRGBImageFilter.h:32-33` and this crate's own
//!    yaml (`LabelToRGBImageFilter.yaml`'s `detaileddescription`) both claim
//!    "a gray pixel with the same intensity than the background label is
//!    produced" for the background value. But `m_BackgroundColor` is a
//!    separate member (`itkLabelToRGBImageFilter.h:123`,
//!    `OutputPixelType m_BackgroundColor{}`), default-constructed to all-zero
//!    (`itkLabelToRGBImageFilter.hxx:37-42`) and only ever forwarded to the
//!    functor as-is (`itkLabelToRGBImageFilter.hxx:87-91`,
//!    `BeforeThreadedGenerateData`) -- and `LabelToRGBImageFilter.yaml`
//!    exposes no `BackgroundColor` setter at all, only `BackgroundValue`
//!    (the label to treat as background), so through SimpleITK's surface this
//!    member is unreachable and permanently zero. The gray-matching-intensity
//!    behavior described in the docstring belongs to
//!    `LabelOverlayImageFilter`/`LabelOverlayFunctor`
//!    (`itkLabelOverlayFunctor.h:63-71`, which really does pass the base
//!    image's own value through), not to `LabelToRGBImageFilter`. This port
//!    reproduces the actual (fixed-black) behavior, not the docstring's.
//!
//! 2. **`SetLabelFunctorFromColormap`'s own doc comment describes behavior
//!    its code does not implement.** `sitkLabelFunctorUtils.hxx:26-32` states
//!    a `colormap` whose length isn't a multiple of 3 has its "remainder …
//!    ignored", but the loop at `sitkLabelFunctorUtils.hxx:43-46`
//!    (`for (size_t i = 0; i < colormap.size(); i += 3) { functor.AddColor(
//!    colormap[i], colormap[i + 1], colormap[i + 2]); }`) reads
//!    `colormap[i + 1]`/`colormap[i + 2]` unconditionally on its last
//!    iteration -- an out-of-bounds `std::vector::operator[]` read (undefined
//!    behavior) when the remainder is 1 or 2 bytes, not a graceful skip.
//!    [`build_color_table`] implements the *documented* behavior (via
//!    `chunks_exact(3)`, which drops an incomplete trailing remainder), a
//!    diverge-for-C++-UB case per this crate's porting policy.

use crate::error::{FilterError, Result};
use sitk_core::{Image, PixelId, Scalar};

/// `itkLabelToRGBFunctor.h`'s default-constructor palette (lines 70-76): 30
/// raw `(r, g, b)` bytes, before `AddColor`'s `NumericTraits<ValueType>::max()`
/// scaling.
const DEFAULT_LABEL_COLORS: [[u8; 3]; 30] = [
    [255, 0, 0],
    [0, 205, 0],
    [0, 0, 255],
    [0, 255, 255],
    [255, 0, 255],
    [255, 127, 0],
    [0, 100, 0],
    [138, 43, 226],
    [139, 35, 35],
    [0, 0, 128],
    [139, 139, 0],
    [255, 62, 150],
    [139, 76, 57],
    [0, 134, 139],
    [205, 104, 57],
    [191, 62, 255],
    [0, 139, 69],
    [199, 21, 133],
    [205, 55, 0],
    [32, 178, 170],
    [106, 90, 205],
    [255, 20, 147],
    [69, 139, 116],
    [72, 118, 255],
    [205, 79, 57],
    [0, 0, 205],
    [139, 34, 82],
    [139, 0, 139],
    [238, 130, 238],
    [139, 0, 0],
];

/// `SetLabelFunctorFromColormap` (`sitkLabelFunctorUtils.hxx:38-46`): the
/// default 30-color palette stays in effect unless `colormap` holds at least
/// one full RGB triple, in which case it *replaces* the palette entirely
/// (`ResetColors()`, no merge) with `colormap.len() / 3` colors, silently
/// dropping an incomplete trailing remainder. See the module docs' upstream
/// finding #2 for why this diverges from the literal (UB) C++.
fn build_color_table(colormap: &[u8]) -> Vec<[u8; 3]> {
    if colormap.len() / 3 == 0 {
        return DEFAULT_LABEL_COLORS.to_vec();
    }
    colormap
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Per-pixel lookup shared by both filters: `None` for a background label,
/// `Some(index into a `num_colors`-entry color table)` otherwise
/// (`itkLabelToRGBFunctor.h:90-102`'s `operator()`: `p == m_BackgroundValue`
/// / `m_Colors[p % m_Colors.size()]`).
///
/// `p % m_Colors.size()` in C++ triggers the usual arithmetic conversions:
/// `TLabel` (any of the 8 signed/unsigned integer pixel types) converts to
/// `m_Colors.size()`'s type (`size_t`, unsigned 64-bit) by sign-extending to
/// 64 bits and reinterpreting the bit pattern as unsigned -- exactly
/// `(p as i64) as u64` in Rust for every one of the 8 integer scalar types
/// (verified: zero-extension for the unsigned source types, since a
/// non-negative value's `i64`/`u64` representations already agree; identity
/// for `i64`/`u64` themselves; sign-extension-then-reinterpret for the
/// narrower signed types). This is well-defined C++, not UB, so no
/// divergence is needed here.
///
/// Also the sole validation point for both filters: only the 8 integer
/// scalar `PixelId`s are matched, so a floating-point, vector, or
/// vector-integer label image (`LabelToRGBImageFilter.yaml`'s
/// `pixel_types: IntegerPixelIDTypeList`, `LabelOverlayImageFilter.yaml`'s
/// `pixel_types2: IntegerPixelIDTypeList`) falls through to
/// [`FilterError::RequiresIntegerPixelType`].
fn label_color_indices(
    label_img: &Image,
    background_value: f64,
    num_colors: usize,
) -> Result<Vec<Option<usize>>> {
    let id = label_img.pixel_id();
    let n = num_colors as u64;

    macro_rules! indices_for {
        ($ty:ty) => {{
            let bg = <$ty as Scalar>::from_f64(background_value);
            label_img
                .scalar_slice::<$ty>()?
                .iter()
                .map(|&p| (p != bg).then(|| (((p as i64) as u64) % n) as usize))
                .collect()
        }};
    }

    match id {
        PixelId::UInt8 => Ok(indices_for!(u8)),
        PixelId::Int8 => Ok(indices_for!(i8)),
        PixelId::UInt16 => Ok(indices_for!(u16)),
        PixelId::Int16 => Ok(indices_for!(i16)),
        PixelId::UInt32 => Ok(indices_for!(u32)),
        PixelId::Int32 => Ok(indices_for!(i32)),
        PixelId::UInt64 => Ok(indices_for!(u64)),
        PixelId::Int64 => Ok(indices_for!(i64)),
        _ => Err(FilterError::RequiresIntegerPixelType(id)),
    }
}

/// `LabelToRGBImageFilter`: apply the (default or custom) color palette to a
/// label image, producing a 3-component `uint8_t` (`VectorUInt8`) image.
/// Background labels (`background_value`, default `0.0`) map to fixed black
/// -- see the module docs' upstream finding #1.
pub fn label_to_rgb(label_img: &Image, background_value: f64, colormap: &[u8]) -> Result<Image> {
    let colors = build_color_table(colormap);
    let indices = label_color_indices(label_img, background_value, colors.len())?;

    let mut data = Vec::with_capacity(indices.len() * 3);
    for idx in &indices {
        match idx {
            Some(i) => data.extend_from_slice(&colors[*i]),
            None => data.extend_from_slice(&[0, 0, 0]),
        }
    }

    let mut out = Image::from_vec_vector(label_img.size(), 3, data)?;
    out.copy_geometry_from(label_img);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label_img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn label_to_rgb_default_palette_lookup() {
        let img = label_img_u8(&[3, 1], vec![0, 1, 2]);
        let out = label_to_rgb(&img, 0.0, &[]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[
                0, 0, 0, // label 0 == background -> fixed black
                0, 205, 0, // label 1 -> palette[1]
                0, 0, 255, // label 2 -> palette[2]
            ]
        );
    }

    #[test]
    fn label_to_rgb_wraps_around_the_30_color_palette() {
        // label 30 % 30 == 0 -> the same color as label index 0.
        let img = label_img_u8(&[1, 1], vec![30]);
        let out = label_to_rgb(&img, 255.0, &[]).unwrap();
        assert_eq!(out.component_slice::<u8>().unwrap(), &[255, 0, 0]);
    }

    #[test]
    fn label_to_rgb_negative_label_on_a_signed_type_wraps_via_two_complement() {
        // Int32 label -1: (p as i64) as u64 == u64::MAX; u64::MAX % 30 == 15
        // (hand-computed) -> palette[15] == (191, 62, 255).
        let img = Image::from_vec(&[1, 1], vec![-1i32]).unwrap();
        let out = label_to_rgb(&img, 0.0, &[]).unwrap();
        assert_eq!(out.component_slice::<u8>().unwrap(), &[191, 62, 255]);
    }

    #[test]
    fn label_to_rgb_custom_colormap_replaces_the_default_palette() {
        // The yaml's own `custom_color` test settings: red, white, blue.
        let colormap: [u8; 9] = [255, 0, 0, 255, 255, 255, 0, 0, 255];
        let img = label_img_u8(&[4, 1], vec![0, 1, 2, 3]);
        let out = label_to_rgb(&img, 0.0, &colormap).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[
                0, 0, 0, // label 0 == background -> fixed black
                255, 255, 255, // label 1 % 3 == 1 -> white
                0, 0, 255, // label 2 % 3 == 2 -> blue
                255, 0, 0, // label 3 % 3 == 0 -> red
            ]
        );
    }

    #[test]
    fn label_to_rgb_colormap_not_a_multiple_of_3_drops_the_remainder() {
        // 5 bytes: one full (255,0,0) triple plus an incomplete (10,20)
        // remainder, dropped per the module docs' upstream finding #2.
        let colormap: [u8; 5] = [255, 0, 0, 10, 20];
        let img = label_img_u8(&[2, 1], vec![1, 5]);
        let out = label_to_rgb(&img, 0.0, &colormap).unwrap();
        // Only one color in the table, so every non-background label maps to it.
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[255, 0, 0, 255, 0, 0]
        );
    }

    #[test]
    fn label_to_rgb_rejects_a_floating_point_label_image() {
        let img = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        let err = label_to_rgb(&img, 0.0, &[]).unwrap_err();
        assert!(matches!(
            err,
            FilterError::RequiresIntegerPixelType(PixelId::Float32)
        ));
    }

    #[test]
    fn label_to_rgb_copies_geometry() {
        let mut img = label_img_u8(&[2, 1], vec![0, 1]);
        img.set_spacing(&[2.0, 3.0]).unwrap();
        let out = label_to_rgb(&img, 0.0, &[]).unwrap();
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.size(), &[2, 1]);
    }
}
