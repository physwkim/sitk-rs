//! `LabelToRGBImageFilter` and `LabelOverlayImageFilter`, ported from
//! `Modules/Filtering/ImageFusion/include/`'s `itkLabelToRGBFunctor.h`,
//! `itkLabelToRGBImageFilter.h`/`.hxx`, `itkLabelOverlayFunctor.h` and
//! `itkLabelOverlayImageFilter.h`/`.hxx`, plus SimpleITK's shared colormap
//! parsing (`sitkLabelFunctorUtils.hxx`'s `SetLabelFunctorFromColormap`).
//!
//! [`build_color_table`] and [`label_color_indices`] are shared by both
//! filters, which differ only in output pixel type and per-pixel
//! combination:
//!
//! - [`label_to_rgb`]'s output ValueType is always `uint8_t`
//!   (`LabelToRGBImageFilter.yaml`'s `output_image_type: itk::VectorImage<
//!   uint8_t, ...>`), so `AddColor`'s `byte / 255 * NumericTraits<uint8_t>::max()`
//!   scaling is the identity (`max() == 255`) and the raw palette bytes are
//!   used directly.
//! - [`label_overlay`]'s output ValueType is the *base image's own* pixel
//!   type (`LabelOverlayImageFilter.yaml`'s `output_image_type: itk::VectorImage<
//!   typename InputImageType::PixelType, ...>`), so the same palette is
//!   rescaled to `NumericTraits<ValueType>::max()` per output type -- for an
//!   `Int16` base into `[0, 32767]`, matching ITK. A *floating-point* base
//!   would scale into `[0, f64::MAX]` and swamp the base pixel in the blend;
//!   this port rejects that instead (see finding 3).
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
//!
//! 3. **`LabelOverlayImageFilter`'s color table is rescaled by the *base
//!    image's* `NumericTraits<ValueType>::max()`, not a fixed `255`;
//!    a floating-point base is rejected (Fixed in this port).**
//!    `itkLabelToRGBFunctor.h:112-116`'s `AddColor` scales every palette byte
//!    by `NumericTraits<ValueType>::max()`, and `LabelOverlayFunctor`'s
//!    internal `m_RGBFunctor` (`itkLabelOverlayFunctor.h:137`) is instantiated
//!    with `ValueType` = the *base* image's own pixel component type
//!    (`LabelOverlayImageFilter.yaml`'s `output_image_type`), not `uint8_t`.
//!    For every **integer** base this is reproduced faithfully -- an `Int16`
//!    base scales the palette into `[0, 32767]`, a `UInt16` base into
//!    `[0, 65535]`. But a **floating-point** base scales into `[0, f64::MAX]`,
//!    which swamps the base pixel in the blend `opaque*opacity + p1*(1-opacity)`
//!    by hundreds of orders of magnitude and makes the overlay meaningless.
//!    That is silent wrongness on realistic well-formed input, so this port
//!    rejects a floating-point base with
//!    [`FilterError::FloatingPointBaseLabelOverlay`] rather than emitting the
//!    swamped garbage. Any float scaling rule (fixed `0-255`, data-driven max)
//!    would be invented semantics with no unique answer, so typed refusal is
//!    the honest resolution. Ledger §2.54.

use crate::error::{FilterError, Result};
use crate::quantize_to_pixel_type;
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar};

/// `itkLabelToRGBFunctor.h`'s default-constructor palette (lines 70-76): 30
/// raw `(r, g, b)` bytes, before `AddColor`'s `NumericTraits<ValueType>::max()`
/// scaling.
pub(crate) const DEFAULT_LABEL_COLORS: [[u8; 3]; 30] = [
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
pub(crate) fn build_color_table(colormap: &[u8]) -> Vec<[u8; 3]> {
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

/// Unlike [`crate::require_same_shape`], `image` and `label_image` may
/// legitimately differ in pixel type (`LabelOverlayImageFilter.yaml`'s
/// `pixel_types`/`pixel_types2`), so only their size is checked here.
fn require_same_size(a: &Image, b: &Image) -> Result<()> {
    if a.size() != b.size() {
        return Err(FilterError::SizeMismatch {
            a: a.size().to_vec(),
            b: b.size().to_vec(),
        });
    }
    Ok(())
}

fn build_vector_from_f64<T: Scalar>(
    size: &[usize],
    components_per_pixel: usize,
    geom: &Image,
    vals: &[f64],
) -> Result<Image> {
    let out: Vec<T> = vals.iter().map(|&v| T::from_f64(v)).collect();
    let mut img = Image::from_vec_vector(size, components_per_pixel, out)?;
    img.copy_geometry_from(geom);
    Ok(img)
}

/// The vector-image counterpart of `crate::image_from_f64`: narrows `vals`
/// (`components_per_pixel` interleaved `f64`s per pixel) to `component_id`'s
/// native type and builds a vector image, copying `geom`'s geometry.
pub(crate) fn vector_image_from_f64(
    component_id: PixelId,
    size: &[usize],
    components_per_pixel: usize,
    geom: &Image,
    vals: &[f64],
) -> Result<Image> {
    dispatch_scalar!(
        component_id,
        build_vector_from_f64,
        size,
        components_per_pixel,
        geom,
        vals
    )
}

/// `LabelOverlayImageFilter`: blend the (default or custom) color palette,
/// looked up by `label_image`, over `image` at `opacity` (default `0.5`).
/// Background labels (`background_value`, default `0.0`) pass `image`'s own
/// value through unchanged on all 3 channels (`itkLabelOverlayFunctor.h:63-71`).
///
/// `image` and `label_image` need only agree on size, not pixel type -- the
/// output's component type follows `image`'s, not `label_image`'s.
///
/// Per non-background pixel, `itkLabelOverlayFunctor.h:74-82` computes:
/// `rgbPixel[c] = static_cast<ValueType>(opaque[c] * opacity + p1 * (1 -
/// opacity))`, where `opaque` is the palette color already scaled to
/// `NumericTraits<ValueType>::max()` and narrowed to `ValueType` *once* when
/// the color table is built (`AddColor`, `itkLabelToRGBFunctor.h:104-118`),
/// not per pixel -- so `opaque[c]` can itself lose precision (e.g. truncating
/// a scaled byte to an integer `ValueType`) before it's promoted back to
/// `double` for the blend. This port reproduces that two-step rounding via
/// [`quantize_to_pixel_type`] when building `scaled_colors`.
pub fn label_overlay(
    image: &Image,
    label_image: &Image,
    opacity: f64,
    background_value: f64,
    colormap: &[u8],
) -> Result<Image> {
    require_same_size(image, label_image)?;

    let base_id = image.pixel_id();
    // §2.54: the palette is scaled by the base image's own
    // NumericTraits<ValueType>::max(). For a floating-point base that is
    // f32/f64 MAX, which swamps the base pixel in the blend and produces a
    // meaningless overlay. Refuse rather than emit the swamped garbage; any
    // float scaling rule would be invented semantics with no unique answer.
    if base_id.is_floating_point() {
        return Err(FilterError::FloatingPointBaseLabelOverlay(base_id));
    }
    let base_vals = image.to_f64_vec()?;
    let value_max = crate::numeric_traits_max(base_id);

    let colors = build_color_table(colormap);
    let scaled_colors: Vec<[f64; 3]> = colors
        .iter()
        .map(|c| {
            std::array::from_fn(|i| {
                quantize_to_pixel_type(base_id, c[i] as f64 / 255.0 * value_max)
            })
        })
        .collect();

    let indices = label_color_indices(label_image, background_value, colors.len())?;

    let mut flat = Vec::with_capacity(base_vals.len() * 3);
    for (&p1, idx) in base_vals.iter().zip(&indices) {
        match idx {
            None => flat.extend_from_slice(&[p1, p1, p1]),
            Some(i) => {
                let p1_blend = p1 * (1.0 - opacity);
                let sc = scaled_colors[*i];
                flat.extend_from_slice(&[
                    sc[0] * opacity + p1_blend,
                    sc[1] * opacity + p1_blend,
                    sc[2] * opacity + p1_blend,
                ]);
            }
        }
    }

    vector_image_from_f64(base_id, image.size(), 3, image, &flat)
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

    // ---- label_overlay ----

    #[test]
    fn label_overlay_background_passes_through_the_base_value_unchanged() {
        let base = Image::from_vec(&[1, 1], vec![42u8]).unwrap();
        let label = label_img_u8(&[1, 1], vec![0]);
        let out = label_overlay(&base, &label, 0.5, 0.0, &[]).unwrap();
        assert_eq!(out.component_slice::<u8>().unwrap(), &[42, 42, 42]);
    }

    #[test]
    fn label_overlay_blends_with_default_opacity_on_a_u8_base_image() {
        // u8 base image: NumericTraits<uint8_t>::max() == 255, so AddColor's
        // scaling is the identity and palette[1] == (0, 205, 0) directly.
        // p1 = 100, opacity = 0.5: p1_blend = 50; static_cast truncates
        // toward zero: [50, floor(205*0.5+50), 50] == [50, 152, 50].
        let base = Image::from_vec(&[1, 1], vec![100u8]).unwrap();
        let label = label_img_u8(&[1, 1], vec![1]);
        let out = label_overlay(&base, &label, 0.5, 0.0, &[]).unwrap();
        assert_eq!(out.component_slice::<u8>().unwrap(), &[50, 152, 50]);
    }

    #[test]
    fn label_overlay_scales_the_palette_by_the_base_images_own_numeric_max() {
        // u16 base image: NumericTraits<uint16_t>::max() == 65535, not 255.
        // palette[1] == (0, 205, 0); scaled green = trunc(205/255*65535) ==
        // 52685. p1 = 1000, opacity = 0.5: p1_blend = 500;
        // green = trunc(52685*0.5 + 500) == 26842; red/blue = 500.
        let base = Image::from_vec(&[1, 1], vec![1000u16]).unwrap();
        let label = label_img_u8(&[1, 1], vec![1]);
        let out = label_overlay(&base, &label, 0.5, 0.0, &[]).unwrap();
        assert_eq!(out.component_slice::<u16>().unwrap(), &[500, 26842, 500]);
    }

    #[test]
    fn label_overlay_rejects_mismatched_sizes() {
        let base = Image::from_vec(&[2, 1], vec![1u8, 2]).unwrap();
        let label = label_img_u8(&[1, 1], vec![0]);
        let err = label_overlay(&base, &label, 0.5, 0.0, &[]).unwrap_err();
        assert!(matches!(err, FilterError::SizeMismatch { .. }));
    }

    #[test]
    fn label_overlay_rejects_a_floating_point_label_image() {
        let base = Image::from_vec(&[1, 1], vec![1u8]).unwrap();
        let label = Image::from_vec(&[1, 1], vec![1.0f64]).unwrap();
        let err = label_overlay(&base, &label, 0.5, 0.0, &[]).unwrap_err();
        assert!(matches!(
            err,
            FilterError::RequiresIntegerPixelType(PixelId::Float64)
        ));
    }

    #[test]
    fn label_overlay_allows_differing_integer_pixel_types_between_base_and_label() {
        // Base image (Int16) and label image (UInt8) may legitimately differ
        // in pixel type -- only their size must agree, and an integer base is
        // accepted.
        let base = Image::from_vec(&[1, 1], vec![10i16]).unwrap();
        let label = label_img_u8(&[1, 1], vec![0]);
        let out = label_overlay(&base, &label, 0.5, 0.0, &[]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorInt16);
        assert_eq!(out.component_slice::<i16>().unwrap(), &[10, 10, 10]);
    }

    #[test]
    fn label_overlay_rejects_a_floating_point_base_image() {
        // §2.54: the palette is scaled by the base's NumericTraits::max(),
        // which is f32/f64 MAX for a floating-point base -- meaningless in the
        // blend. Both float bases are refused with the dedicated typed error,
        // whichever label a pixel carries.
        let label = label_img_u8(&[1, 1], vec![1]);

        let base32 = Image::from_vec(&[1, 1], vec![10.0f32]).unwrap();
        assert_eq!(
            label_overlay(&base32, &label, 0.5, 0.0, &[]).unwrap_err(),
            FilterError::FloatingPointBaseLabelOverlay(PixelId::Float32)
        );

        let base64 = Image::from_vec(&[1, 1], vec![10.0f64]).unwrap();
        assert_eq!(
            label_overlay(&base64, &label, 0.5, 0.0, &[]).unwrap_err(),
            FilterError::FloatingPointBaseLabelOverlay(PixelId::Float64)
        );
    }
}
