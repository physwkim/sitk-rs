//! The label-image ⇄ [`LabelMap`] converters.
//!
//! Ports of `Modules/Filtering/LabelMap/include/`:
//!
//! - `itkLabelImageToLabelMapFilter.h` / `.hxx`
//! - `itkLabelMapToLabelImageFilter.h` / `.hxx`
//! - `itkBinaryImageToLabelMapFilter.h` / `.hxx`
//!
//! The first two are one-liners over [`LabelMap::from_label_image`] and
//! [`LabelMap::to_label_image`], which is where the run-length encoding and the
//! `LabelObject` invariant live. What these wrappers add is SimpleITK's
//! pixel-type gating and its `double`-typed parameter casts.
//!
//! ## `binary_image_to_label_map`
//!
//! `itkBinaryImageToLabelMapFilter` derives from the same `ScanlineFilterCommon`
//! as `itkConnectedComponentImageFilter`, so it shares
//! [`crate::label::scanline_components`] and [`crate::label::create_consecutive`]
//! with [`crate::label::connected_component`]. The three differences:
//!
//! 1. **Foreground is an equality test, not "nonzero".** `.hxx:167` compares
//!    `pixelValue == this->m_InputForegroundValue`. SimpleITK casts the `double`
//!    it exposes to the input pixel type first (`pixeltype: Input`), so a
//!    foreground of `1.5` on a `UInt8` image matches pixels valued `1`.
//!
//! 2. **The label numbering skips the output background.**
//!    `CreateConsecutive(m_OutputBackgroundValue)`
//!    (`itkBinaryImageToLabelMapFilter.hxx:117`,
//!    `itkScanlineFilterCommon.h:199-228`) starts its counter at `0` and bumps
//!    it once, on the single assignment where it would equal the background. So
//!    with the default background `0` the labels are `1, 2, 3, …`; with a
//!    background of `3` they are `0, 1, 2, 4, 5, …`. This differs from
//!    `connected_component`, whose background is fixed at `0`.
//!
//! 3. **The output is a `LabelMap`, not an image.** Each run becomes one
//!    `SetLine` (`.hxx:128-141`), in raster order, so no line ever merges with
//!    another and the [`LabelObject`](sitk_core::LabelObject) invariant costs
//!    nothing.
//!
//! `BinaryImageToLabelMapFilter.yaml` fixes the label type to `uint32_t`
//! (`filter_type: itk::BinaryImageToLabelMapFilter<InputImageType,
//! itk::LabelMap< itk::LabelObject< uint32_t, ... > > >`), so the returned map's
//! [`LabelMap::pixel_id`] is always [`PixelId::UInt32`] regardless of the input's.
//!
//! ### Defaults
//!
//! ITK's constructor (`.hxx:33-35`) defaults `m_InputForegroundValue` to
//! `NumericTraits<InputPixelType>::max()` and `m_OutputBackgroundValue` to
//! `NumericTraits<OutputPixelType>::NonpositiveMin()`, and the yaml's
//! `detaileddescriptionSet` still says so. SimpleITK overrides both: the yaml's
//! declared defaults are `1.0` and `0.0`. [`BinaryImageToLabelMapSettings::default`]
//! follows the yaml, which is the behaviour a SimpleITK caller sees.

use sitk_core::{Image, LabelMap, PixelId};

use crate::error::{FilterError, Result};
use crate::label::{create_consecutive, scanline_components};
use crate::quantize_to_pixel_type;

/// `itk::LabelImageToLabelMapFilter`: run-length encode an integer label image.
///
/// `LabelImageToLabelMapFilter.yaml` declares
/// `pixel_types: UnsignedIntegerPixelIDTypeList`, so a signed, floating-point or
/// vector image is rejected. `background_value` is a `pixeltype: Output` member
/// — SimpleITK casts it to the label type, which for this filter *is* the input
/// pixel type (`itk::LabelObject< typename InputImageType::PixelType, ... >`) —
/// before ITK compares it against any pixel.
pub fn label_image_to_label_map(img: &Image, background_value: f64) -> Result<LabelMap> {
    if !img.pixel_id().is_integer_scalar() || img.pixel_id().is_signed() {
        return Err(FilterError::RequiresUnsignedIntegerPixelType(
            img.pixel_id(),
        ));
    }
    let background = quantize_to_pixel_type(img.pixel_id(), background_value) as i64;
    Ok(LabelMap::from_label_image(img, background)?)
}

/// `itk::LabelMapToLabelImageFilter`: paint every object's pixels with its
/// label, over a background-filled image.
///
/// The output pixel type is the map's own ([`LabelMap::pixel_id`]), matching the
/// yaml's `output_image_type: itk::Image<typename InputImageType::LabelType, …>`.
pub fn label_map_to_label_image(map: &LabelMap) -> Result<Image> {
    Ok(map.to_label_image()?)
}

/// The three settings `BinaryImageToLabelMapFilter.yaml` exposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryImageToLabelMapSettings {
    /// Face connectivity when `false` (4-connected in 2-D, 6-connected in 3-D);
    /// face+edge+vertex connectivity when `true`.
    pub fully_connected: bool,
    /// The input value that counts as foreground, compared for **equality**
    /// after a cast to the input pixel type. SimpleITK's default is `1.0`;
    /// ITK's own is `NumericTraits<InputPixelType>::max()`.
    pub input_foreground_value: f64,
    /// The `LabelMap`'s background value, and the label the consecutive
    /// numbering skips. Cast to the `uint32_t` label type. SimpleITK's default
    /// is `0.0`; ITK's own is `NumericTraits<OutputPixelType>::NonpositiveMin()`.
    pub output_background_value: f64,
}

impl Default for BinaryImageToLabelMapSettings {
    fn default() -> Self {
        Self {
            fully_connected: false,
            input_foreground_value: 1.0,
            output_background_value: 0.0,
        }
    }
}

/// `itk::BinaryImageToLabelMapFilter`: label the connected components of the
/// pixels equal to `settings.input_foreground_value`.
///
/// Returns the map and its `NumberOfObjects` measurement.
///
/// `BinaryImageToLabelMapFilter.yaml` declares
/// `pixel_types: IntegerPixelIDTypeList`, so a floating-point or vector image is
/// rejected.
pub fn binary_image_to_label_map(
    img: &Image,
    settings: &BinaryImageToLabelMapSettings,
) -> Result<(LabelMap, u64)> {
    if !img.pixel_id().is_integer_scalar() {
        return Err(FilterError::RequiresIntegerPixelType(img.pixel_id()));
    }
    let size = img.size();
    // `itk::LabelObject<uint32_t>` is the label type the yaml pins.
    let background =
        quantize_to_pixel_type(PixelId::UInt32, settings.output_background_value) as i64;
    let mut map = LabelMap::new(size, PixelId::UInt32, background)?;
    map.copy_geometry_from(img)?;

    let total: usize = size.iter().product();
    if total == 0 {
        return Ok((map, 0));
    }

    let foreground = quantize_to_pixel_type(img.pixel_id(), settings.input_foreground_value);
    let is_fg: Vec<bool> = img.to_f64_vec()?.iter().map(|&v| v == foreground).collect();

    let mut components = scanline_components(&is_fg, size, settings.fully_connected);
    let (root_to_output, number_of_objects) = create_consecutive(&mut components, background);

    let dim = size.len();
    let mut idx = vec![0i64; dim];
    for (line, runs) in components.line_map.iter().enumerate() {
        if runs.is_empty() {
            continue;
        }
        let mut t = line;
        for d in 1..dim {
            idx[d] = (t % size[d]) as i64;
            t /= size[d];
        }
        for run in runs {
            let root = components.uf.find(run.label);
            idx[0] = run.start as i64;
            map.set_line(&idx, run.len as i64, root_to_output[root])?;
        }
    }
    Ok((map, number_of_objects))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::Error as CoreError;

    fn labels_of(map: &LabelMap) -> Vec<i64> {
        map.labels().collect()
    }

    /// Every pixel of the map, as a dense label image, for readable assertions.
    fn dense(map: &LabelMap) -> Vec<u32> {
        map.to_label_image()
            .unwrap()
            .scalar_slice::<u32>()
            .unwrap()
            .to_vec()
    }

    // ---- label_image_to_label_map ----------------------------------------

    #[test]
    fn label_image_to_label_map_encodes_runs_and_keeps_the_input_pixel_type() {
        let img = Image::from_vec(&[4, 2], vec![1u16, 1, 0, 2, 2, 2, 2, 0]).unwrap();
        let map = label_image_to_label_map(&img, 0.0).unwrap();
        assert_eq!(labels_of(&map), vec![1, 2]);
        assert_eq!(map.pixel_id(), PixelId::UInt16);
        assert_eq!(map.label_object(1).unwrap().lines().len(), 1);
        assert_eq!(map.label_object(2).unwrap().lines().len(), 2);
        assert_eq!(map.label_object(2).unwrap().size(), 4);
    }

    #[test]
    fn label_image_to_label_map_casts_the_background_to_the_input_pixel_type() {
        // 2.7 truncates to 2 under `static_cast<uint8_t>`, so label 2 is the
        // background and only label 1 survives.
        let img = Image::from_vec(&[3, 1], vec![1u8, 2, 2]).unwrap();
        let map = label_image_to_label_map(&img, 2.7).unwrap();
        assert_eq!(labels_of(&map), vec![1]);
        assert_eq!(map.background(), 2);
    }

    #[test]
    fn label_image_to_label_map_rejects_signed_float_and_vector_images() {
        let signed = Image::from_vec(&[2, 2], vec![0i16; 4]).unwrap();
        assert_eq!(
            label_image_to_label_map(&signed, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int16
            ))
        );
        let float = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            label_image_to_label_map(&float, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Float32
            ))
        );
        let vector = Image::from_vec_vector(&[2, 2], 2, vec![0u8; 8]).unwrap();
        assert_eq!(
            label_image_to_label_map(&vector, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::VectorUInt8
            ))
        );
    }

    #[test]
    fn label_image_to_label_map_rejects_a_four_dimensional_image() {
        let img = Image::from_vec(&[2, 2, 2, 2], vec![0u8; 16]).unwrap();
        assert_eq!(
            label_image_to_label_map(&img, 0.0),
            Err(FilterError::Core(CoreError::UnsupportedLabelMapDimension(
                4
            )))
        );
    }

    // ---- label_map_to_label_image ----------------------------------------

    #[test]
    fn label_map_to_label_image_round_trips() {
        let img = Image::from_vec(&[4, 3], vec![0u8, 1, 1, 0, 0, 0, 2, 2, 3, 3, 0, 0]).unwrap();
        let map = label_image_to_label_map(&img, 0.0).unwrap();
        assert_eq!(label_map_to_label_image(&map).unwrap(), img);
    }

    #[test]
    fn label_map_to_label_image_fills_with_the_maps_background() {
        let img = Image::from_vec(&[3, 1], vec![7u8, 1, 7]).unwrap();
        let map = label_image_to_label_map(&img, 7.0).unwrap();
        assert_eq!(
            label_map_to_label_image(&map)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[7, 1, 7]
        );
    }

    // ---- binary_image_to_label_map ---------------------------------------

    #[test]
    fn binary_image_to_label_map_labels_face_connected_components_from_one() {
        // 1 . 1
        // 1 . 1
        let img = Image::from_vec(&[3, 2], vec![1u8, 0, 1, 1, 0, 1]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(labels_of(&map), vec![1, 2]);
        assert_eq!(map.pixel_id(), PixelId::UInt32);
        assert_eq!(map.background(), 0);
        assert_eq!(dense(&map), vec![1, 0, 2, 1, 0, 2]);
    }

    #[test]
    fn binary_image_to_label_map_full_connectivity_joins_a_diagonal() {
        // 1 .
        // . 1
        let img = Image::from_vec(&[2, 2], vec![1u8, 0, 0, 1]).unwrap();
        let face = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(face.1, 2);

        let settings = BinaryImageToLabelMapSettings {
            fully_connected: true,
            ..Default::default()
        };
        let (map, n) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(n, 1);
        assert_eq!(labels_of(&map), vec![1]);
    }

    #[test]
    fn binary_image_to_label_map_foreground_is_an_equality_test_not_nonzero() {
        // `connected_component` would treat both 1 and 5 as foreground; this
        // filter only accepts pixels equal to `input_foreground_value`.
        let img = Image::from_vec(&[3, 1], vec![1u8, 5, 1]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(dense(&map), vec![1, 0, 2]);

        let settings = BinaryImageToLabelMapSettings {
            input_foreground_value: 5.0,
            ..Default::default()
        };
        let (map, n) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(n, 1);
        assert_eq!(dense(&map), vec![0, 1, 0]);
    }

    #[test]
    fn binary_image_to_label_map_casts_the_foreground_to_the_input_pixel_type() {
        // `static_cast<uint8_t>(1.9)` is 1.
        let img = Image::from_vec(&[2, 1], vec![1u8, 0]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            input_foreground_value: 1.9,
            ..Default::default()
        };
        assert_eq!(binary_image_to_label_map(&img, &settings).unwrap().1, 1);
    }

    #[test]
    fn binary_image_to_label_map_numbering_skips_the_output_background_once() {
        // Three components with `output_background_value = 2`: CreateConsecutive
        // hands out 0, 1, then bumps past 2 to 3.
        let img = Image::from_vec(&[5, 1], vec![1u8, 0, 1, 0, 1]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            output_background_value: 2.0,
            ..Default::default()
        };
        let (map, n) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(n, 3);
        assert_eq!(labels_of(&map), vec![0, 1, 3]);
        assert_eq!(map.background(), 2);
        assert_eq!(dense(&map), vec![0, 2, 1, 2, 3]);
    }

    #[test]
    fn binary_image_to_label_map_numbering_starts_at_zero_for_a_non_zero_background() {
        let img = Image::from_vec(&[3, 1], vec![1u8, 0, 1]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            output_background_value: 9.0,
            ..Default::default()
        };
        let (map, _) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(labels_of(&map), vec![0, 1]);
    }

    #[test]
    fn binary_image_to_label_map_negative_background_saturates_to_zero() {
        // `static_cast<uint32_t>(-1.0)` is C++ UB (an out-of-range float→int
        // conversion). This port routes the value through the same
        // `quantize_to_pixel_type`/`Scalar::from_f64` saturating cast every
        // other `pixeltype:` member uses, so it lands on 0.
        let img = Image::from_vec(&[2, 1], vec![1u8, 0]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            output_background_value: -1.0,
            ..Default::default()
        };
        let (map, _) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(map.background(), 0);
        assert_eq!(labels_of(&map), vec![1]);
    }

    #[test]
    fn binary_image_to_label_map_on_an_all_background_image_has_no_objects() {
        let img = Image::from_vec(&[3, 2], vec![0u8; 6]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 0);
        assert_eq!(map.number_of_label_objects(), 0);
        assert_eq!(dense(&map), vec![0; 6]);
    }

    #[test]
    fn binary_image_to_label_map_labels_in_raster_order_of_first_appearance() {
        // The right-hand component's first pixel appears before the left-hand
        // one's, so it takes label 1.
        //   . 1
        //   1 1
        let img = Image::from_vec(&[2, 2], vec![0u8, 1, 1, 1]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 1);
        assert_eq!(dense(&map), vec![0, 1, 1, 1]);
    }

    #[test]
    fn binary_image_to_label_map_copies_geometry() {
        let mut img = Image::from_vec(&[2, 2], vec![1u8, 0, 0, 0]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(map.spacing(), &[0.5, 2.0]);
        assert_eq!(map.origin(), &[-1.0, 3.0]);
        assert_eq!(map.to_label_image().unwrap().spacing(), &[0.5, 2.0]);
    }

    #[test]
    fn binary_image_to_label_map_rejects_a_float_image() {
        let img = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            binary_image_to_label_map(&img, &Default::default()),
            Err(FilterError::RequiresIntegerPixelType(PixelId::Float32))
        );
    }

    #[test]
    fn binary_image_to_label_map_3d_face_connectivity_crosses_slices() {
        let mut data = vec![0u8; 8];
        data[0] = 1; // (0,0,0)
        data[4] = 1; // (0,0,1)
        let img = Image::from_vec(&[2, 2, 2], data).unwrap();
        let (_, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn binary_image_to_label_map_agrees_with_connected_component_on_a_zero_one_image() {
        let img = Image::from_vec(&[4, 3], vec![1u8, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1]).unwrap();
        let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        let cc = crate::label::connected_component(&img, false).unwrap();
        assert_eq!(dense(&map), cc.scalar_slice::<u32>().unwrap());
    }
}
