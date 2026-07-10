//! `ScalarToRGBColormapImageFilter`, ported from
//! `Modules/Filtering/Colormap/include/itkScalarToRGBColormapImageFilter.h(.hxx)`,
//! `itkColormapFunction.h` (the shared `RescaleInputValue`/
//! `RescaleRGBComponentValue` base logic) and the 14 `itk::Function::*ColormapFunction`
//! `.hxx` files it dispatches to (`itkRedColormapFunction.hxx`,
//! `itkGreenColormapFunction.hxx`, `itkBlueColormapFunction.hxx`,
//! `itkGreyColormapFunction.hxx`, `itkHotColormapFunction.hxx`,
//! `itkCoolColormapFunction.hxx`, `itkSpringColormapFunction.hxx`,
//! `itkSummerColormapFunction.hxx`, `itkAutumnColormapFunction.hxx`,
//! `itkWinterColormapFunction.hxx`, `itkCopperColormapFunction.hxx`,
//! `itkJetColormapFunction.hxx`, `itkHSVColormapFunction.hxx`,
//! `itkOverUnderColormapFunction.hxx`), matching SimpleITK's
//! `ScalarToRGBColormapImageFilter.yaml` (`pixel_types: BasicPixelIDTypeList`,
//! `output_image_type: itk::VectorImage<unsigned char, ...>`, `Colormap`
//! default `Grey`, `UseInputImageExtremaForScaling` default `true`).
//!
//! Every colormap function first rescales the input pixel to `value ∈ [0,
//! 1]` via `ColormapFunction::RescaleInputValue` (`itkColormapFunction.h:90-100`):
//! `value = clamp((v - min) / (max - min), 0, 1)`, where `min`/`max` come
//! either from an actual scan of the input image
//! (`UseInputImageExtremaForScaling = true`, the scan in
//! `BeforeThreadedGenerateData`) or from `ColormapFunction`'s own
//! constructor defaults (`itkColormapFunction.h:78-83`). It then computes a
//! per-channel formula of `value` (see [`apply_colormap`]) and narrows each
//! channel back to `uint8_t` via `RescaleRGBComponentValue`
//! (`itkColormapFunction.h:105-112`): `static_cast<uint8_t>(255.0 * c)` --
//! since SimpleITK exposes no setter for `MinimumRGBComponentValue`/
//! `MaximumRGBComponentValue`, those stay at their `NumericTraits<uint8_t>::min()`/
//! `max()` defaults (`0`/`255`) always, reducing the general
//! `d * v + minimumRGBComponentValue` formula to plain `255.0 * v`.
//!
//! ## Upstream findings
//!
//! 1. **The `UseInputImageExtremaForScaling = true` scan's `maximumValue` seed
//!    is `NumericTraits<T>::min()`, which for a floating-point `T` is the
//!    smallest *positive* normalized value, not the most negative one.**
//!    `itkScalarToRGBColormapImageFilter.hxx:74-88`:
//!    ```text
//!    InputImagePixelType minimumValue = NumericTraits<InputImagePixelType>::max();
//!    InputImagePixelType maximumValue = NumericTraits<InputImagePixelType>::min();
//!    for (...) {
//!      if (value < minimumValue) { minimumValue = value; }
//!      if (value > maximumValue) { maximumValue = value; }
//!    }
//!    ```
//!    For an all-non-positive `Float32`/`Float64` input image (every pixel
//!    `<= 0`), `value > maximumValue` never succeeds (no non-positive value
//!    exceeds a tiny positive seed), so `maximumValue` never leaves its
//!    `FLT_MIN`/`DBL_MIN` seed -- the scan silently reports a near-zero
//!    maximum instead of the image's actual (negative) maximum, shrinking
//!    the effective rescale range and pushing every pixel's `value` higher
//!    than it should be (e.g. an image of `[-10, -5, -1]` rescales `-1` to
//!    `0.9`, not the `1.0` a correct scan would give -- pinned by
//!    [`tests::float_all_non_positive_image_leaves_the_scan_maximum_at_the_tiny_positive_seed`]).
//!    This port reproduces the scan verbatim (see [`input_extrema`]) rather
//!    than fixing it, per this crate's porting policy.
//!
//! 2. **`UseInputImageExtremaForScaling = false` does not fall back to the
//!    input type's full native range for floating-point `T`, for the same
//!    reason.** `itkColormapFunction.h:78-83`'s constructor seeds
//!    `m_MinimumInputValue = NumericTraits<TScalar>::min()` and
//!    `m_MaximumInputValue = NumericTraits<TScalar>::max()`. For every
//!    integer `TScalar` this *is* the full native range (`min()` has no
//!    override), but for `Float32`/`Float64` it pairs the smallest positive
//!    normalized value with the true maximum, giving a rescale denominator
//!    of `d ≈ FLT_MAX`/`DBL_MAX` -- so any pixel of ordinary magnitude maps
//!    to `value ≈ 0`, collapsing almost the entire realistic input range to
//!    a single output color instead of spanning it (pinned by
//!    [`tests::extrema_scaling_disabled_on_a_float_image_collapses_ordinary_values_toward_zero`]).
//!
//! Both findings share the same root cause --
//! `NumericTraits<float/double>::min()` meaning "smallest positive", not
//! "most negative" -- surfacing in two independent branches of this one
//! filter.

use crate::error::Result;
use crate::{numeric_traits_max, numeric_traits_min};
use sitk_core::{Image, PixelId};

/// `itk::Function::*ColormapFunction`'s `RGBColormapFilterEnum`
/// (`itkScalarToRGBColormapImageFilterEnums.h`), matching
/// `ScalarToRGBColormapImageFilter.yaml`'s `Colormap` member exactly (14
/// variants, default `Grey`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Colormap {
    Red,
    Green,
    Blue,
    #[default]
    Grey,
    Hot,
    Cool,
    Spring,
    Summer,
    Autumn,
    Winter,
    Copper,
    Jet,
    HSV,
    OverUnder,
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// The 14 `itk::Function::*ColormapFunction::operator()` bodies, each
/// applied to `value` (already `RescaleInputValue`'s output, in `[0, 1]`),
/// producing `[red, green, blue]` before `RescaleRGBComponentValue`
/// narrows each channel to `uint8_t`.
fn apply_colormap(colormap: Colormap, value: f64) -> [f64; 3] {
    match colormap {
        // itkRedColormapFunction.hxx:34-36
        Colormap::Red => [value, 0.0, 0.0],
        // itkGreenColormapFunction.hxx:34-36
        Colormap::Green => [0.0, value, 0.0],
        // itkBlueColormapFunction.hxx:34-36
        Colormap::Blue => [0.0, 0.0, value],
        // itkGreyColormapFunction.hxx:34-36
        Colormap::Grey => [value, value, value],
        // itkHotColormapFunction.hxx:31-38
        Colormap::Hot => [
            clamp01(63.0 / 26.0 * value - 1.0 / 13.0),
            clamp01(63.0 / 26.0 * value - 11.0 / 13.0),
            clamp01(4.5 * value - 3.5),
        ],
        // itkCoolColormapFunction.hxx:31-35
        Colormap::Cool => [value, 1.0 - value, 1.0],
        // itkSpringColormapFunction.hxx:31-35
        Colormap::Spring => [1.0, value, 1.0 - value],
        // itkSummerColormapFunction.hxx:31-35
        Colormap::Summer => [value, 0.5 * value + 0.5, 0.4],
        // itkAutumnColormapFunction.hxx:31-35
        Colormap::Autumn => [1.0, value, 0.0],
        // itkWinterColormapFunction.hxx:31-35
        Colormap::Winter => [0.0, value, 1.0 - 0.5 * value],
        // itkCopperColormapFunction.hxx:31-37
        Colormap::Copper => [(1.2 * value).min(1.0), 0.8 * value, 0.5 * value],
        // itkJetColormapFunction.hxx:31-38
        Colormap::Jet => [
            clamp01(-(3.95 * (value - 0.7460)).abs() + 1.5),
            clamp01(-(3.95 * (value - 0.492)).abs() + 1.5),
            clamp01(-(3.95 * (value - 0.2385)).abs() + 1.5),
        ],
        // itkHSVColormapFunction.hxx:32-39 -- note the red channel's
        // `Absolute(...) - 5/6` is not negated, unlike green/blue.
        Colormap::HSV => [
            clamp01((5.0 * (value - 0.5)).abs() - 5.0 / 6.0),
            clamp01(-(5.0 * (value - 11.0 / 30.0)).abs() + 11.0 / 6.0),
            clamp01(-(5.0 * (value - 19.0 / 30.0)).abs() + 11.0 / 6.0),
        ],
        // itkOverUnderColormapFunction.hxx:31-48
        Colormap::OverUnder => {
            if value == 0.0 {
                [0.0, 0.0, 1.0]
            } else if value == 1.0 {
                [1.0, 0.0, 0.0]
            } else {
                [value, value, value]
            }
        }
    }
}

/// `ColormapFunction::RescaleRGBComponentValue` (`itkColormapFunction.h:105-112`)
/// with `MinimumRGBComponentValue = 0`, `MaximumRGBComponentValue = 255`
/// (SimpleITK exposes no setter to override either): `static_cast<uint8_t>(
/// 255.0 * v)`. `v` is always in `[0, 1]` by construction (every
/// [`apply_colormap`] arm clamps or is bounded within it), so this is a
/// well-defined truncating narrow, not the undefined out-of-range
/// `static_cast` this crate otherwise diverges from -- `as u8` truncates
/// toward zero identically for any in-range value.
fn rescale_rgb_component(v: f64) -> u8 {
    (255.0 * v) as u8
}

/// The `UseInputImageExtremaForScaling` branch of
/// `BeforeThreadedGenerateData`/`ColormapFunction`'s constructor -- see the
/// module docs' upstream findings 1 and 2 for the floating-point quirks
/// both branches carry.
fn input_extrema(
    id: PixelId,
    vals: &[f64],
    use_input_image_extrema_for_scaling: bool,
) -> (f64, f64) {
    if !use_input_image_extrema_for_scaling {
        return (numeric_traits_min(id), numeric_traits_max(id));
    }
    let mut min_v = numeric_traits_max(id);
    let mut max_v = numeric_traits_min(id);
    for &v in vals {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    (min_v, max_v)
}

/// `ScalarToRGBColormapImageFilter`: map each input pixel through `colormap`,
/// rescaled by the input image's extrema (`use_input_image_extrema_for_scaling
/// = true`, the default) or by `colormap`'s native-range default (`= false`),
/// producing a 3-component `uint8_t` (`VectorUInt8`) image.
pub fn scalar_to_rgb_colormap(
    img: &Image,
    colormap: Colormap,
    use_input_image_extrema_for_scaling: bool,
) -> Result<Image> {
    let vals = img.to_f64_vec()?;
    let (min_v, max_v) = input_extrema(img.pixel_id(), &vals, use_input_image_extrema_for_scaling);
    let d = max_v - min_v;

    let mut data = Vec::with_capacity(vals.len() * 3);
    for &v in &vals {
        let value = ((v - min_v) / d).clamp(0.0, 1.0);
        let rgb = apply_colormap(colormap, value);
        data.push(rescale_rgb_component(rgb[0]));
        data.push(rescale_rgb_component(rgb[1]));
        data.push(rescale_rgb_component(rgb[2]));
    }

    let mut out = Image::from_vec_vector(img.size(), 3, data)?;
    out.copy_geometry_from(img);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FilterError;

    #[test]
    fn grey_extrema_scaling_true_on_a_u8_image_rescales_to_the_actual_min_max() {
        // min=0, max=255 (correct scan, no quirk for integer types):
        // value = v/255; 0 -> 0, 128 -> 0.50196... -> trunc(255*0.50196)=128,
        // 255 -> 1.0 -> 255.
        let img = Image::from_vec(&[3, 1], vec![0u8, 128, 255]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Grey, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[0, 0, 0, 128, 128, 128, 255, 255, 255]
        );
    }

    #[test]
    fn hot_colormap_matches_hand_derived_piecewise_values() {
        // Float64 image spanning [0, 1] exactly, so value == v; hand-derived
        // via the exact itkHotColormapFunction.hxx formula.
        let img = Image::from_vec(&[5, 1], vec![0.0f64, 0.25, 0.5, 0.75, 1.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Hot, true).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[
                0, 0, 0, // value 0.0
                134, 0, 0, // value 0.25
                255, 93, 0, // value 0.5
                255, 247, 0, // value 0.75
                255, 255, 255, // value 1.0
            ]
        );
    }

    #[test]
    fn jet_colormap_matches_hand_derived_piecewise_values() {
        let img = Image::from_vec(&[5, 1], vec![0.0f64, 0.25, 0.5, 0.75, 1.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Jet, true).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[
                0, 0, 142, // value 0.0
                0, 138, 255, // value 0.25
                134, 255, 119, // value 0.5
                255, 122, 0, // value 0.75
                126, 0, 0, // value 1.0
            ]
        );
    }

    #[test]
    fn hsv_colormap_matches_hand_derived_piecewise_values() {
        let img = Image::from_vec(&[5, 1], vec![0.0f64, 0.25, 0.5, 0.75, 1.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::HSV, true).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[
                255, 0, 0, // value 0.0
                106, 255, 0, // value 0.25
                0, 255, 255, // value 0.5
                106, 0, 255, // value 0.75
                255, 0, 0, // value 1.0
            ]
        );
    }

    #[test]
    fn copper_colormap_clamps_only_the_red_channels_upper_bound() {
        // green = 0.8*value, blue = 0.5*value never need clamping for
        // value in [0, 1]; red = min(1.0, 1.2*value) is the only clamp.
        let img = Image::from_vec(&[3, 1], vec![0.0f64, 0.5, 1.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Copper, true).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[0, 0, 0, 153, 102, 63, 255, 204, 127]
        );
    }

    #[test]
    fn over_under_saturates_at_the_extreme_values_only() {
        let img = Image::from_vec(&[3, 1], vec![0.0f64, 0.5, 1.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::OverUnder, true).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[
                0, 0, 255, // value == 0.0 -> saturated dark (blue)
                127, 127, 127, // value == 0.5 -> passthrough grey
                255, 0, 0, // value == 1.0 -> saturated white (red)
            ]
        );
    }

    #[test]
    fn extrema_scaling_false_on_an_integer_image_uses_the_full_native_range() {
        // Int16: min=-32768, max=32767 (no quirk for integer types), so
        // value for v=0 is (0-(-32768))/65535 = 0.50000763...
        let img = Image::from_vec(&[1, 1], vec![0i16]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Grey, false).unwrap();
        assert_eq!(out.component_slice::<u8>().unwrap(), &[127, 127, 127]);
    }

    #[test]
    fn float_all_non_positive_image_leaves_the_scan_maximum_at_the_tiny_positive_seed() {
        // Upstream finding 1: min correctly scans to -10.0, but the maximum
        // seed (f32::MIN_POSITIVE) is never exceeded by any non-positive
        // value, so d == 10.0 (not the -1.0 - (-10.0) == 9.0 a correct scan
        // would give). -1.0 rescales to 0.9, not 1.0.
        let img = Image::from_vec(&[3, 1], vec![-10.0f32, -5.0, -1.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Grey, true).unwrap();
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[0, 0, 0, 127, 127, 127, 229, 229, 229]
        );
    }

    #[test]
    fn extrema_scaling_disabled_on_a_float_image_collapses_ordinary_values_toward_zero() {
        // Upstream finding 2: min=f32::MIN_POSITIVE, max=f32::MAX, so
        // d ~= f32::MAX; any ordinary-magnitude value rescales to ~0.
        let img = Image::from_vec(&[2, 1], vec![1.0f32, 100.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Grey, false).unwrap();
        assert_eq!(out.component_slice::<u8>().unwrap(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn rejects_a_vector_input_image() {
        let img = Image::from_vec_vector(&[1, 1], 2, vec![1.0f32, 2.0]).unwrap();
        let err = scalar_to_rgb_colormap(&img, Colormap::Grey, true).unwrap_err();
        assert!(matches!(err, FilterError::Core(_)));
    }

    #[test]
    fn copies_geometry() {
        let mut img = Image::from_vec(&[2, 1], vec![0u8, 255]).unwrap();
        img.set_spacing(&[2.0, 3.0]).unwrap();
        let out = scalar_to_rgb_colormap(&img, Colormap::Grey, true).unwrap();
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.size(), &[2, 1]);
    }
}
