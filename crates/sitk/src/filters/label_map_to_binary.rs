//! `itk::LabelMapToBinaryImageFilter`: paint every label object with one
//! foreground value over a background-filled `UInt8` image.
//!
//! Ported from `Modules/Filtering/LabelMap/include/itkLabelMapToBinaryImageFilter.h`
//! / `.hxx` and `Code/BasicFilters/yaml/LabelMapToBinaryImageFilter.yaml`.
//!
//! `GenerateData` (`itkLabelMapToBinaryImageFilter.hxx:62-87`) runs two
//! parallelized passes over the output region:
//!
//! 1. `DynamicThreadedGenerateData` (`.hxx:92-131`) fills the output with
//!    `m_BackgroundValue`;
//! 2. `LabelMapFilter::DynamicThreadedGenerateData` (`itkLabelMapFilter.hxx:83-117`)
//!    hands each label object to `ThreadedProcessLabelObject` (`.hxx:135-144`),
//!    which writes `m_ForegroundValue` at every index of the object.
//!
//! ## Pixel and parameter types
//!
//! The yaml pins `output_image_type: itk::Image<uint8_t, InputImageType::ImageDimension>`,
//! so the output is always [`PixelId::UInt8`] whatever the map's label type is,
//! and both `pixeltype: Output` members are `static_cast<uint8_t>`-ed from the
//! `double` SimpleITK exposes. `pixel_types: LabelPixelIDTypeList`
//! (`sitkPixelIDTypeLists.h:160-167`) is the unsigned integer types only.
//!
//! ### Defaults
//!
//! ITK's constructor (`.hxx:30-36`) defaults `m_BackgroundValue` to
//! `NumericTraits<uint8_t>::NonpositiveMin()` (`0`) and `m_ForegroundValue` to
//! `NumericTraits<uint8_t>::max()` (`255`). The yaml overrides the foreground to
//! `1.0` and the generated wrapper always calls both setters, so a SimpleITK
//! caller sees `0` / `1`. [`label_map_to_binary`] takes both explicitly; the
//! yaml's values are what [`crate::filters::label_map::binary_image_to_label_map`]
//! round-trips against.
//!
//! ## The background image is unreachable
//!
//! `SetBackgroundImage` (`itkLabelMapToBinaryImageFilter.h:99-104`) installs a
//! second indexed input whose pixels replace the flat background fill
//! (`.hxx:99-119`). `LabelMapToBinaryImageFilter.yaml` declares
//! `number_of_inputs: 1` and exposes no such setter, so
//! `GetNumberOfIndexedInputs() == 2` is never true through SimpleITK and only
//! the flat-fill branch (`.hxx:120-130`) runs. This port has no background-image
//! parameter for the same reason.
//!
//! ## Paint order
//!
//! Ledger §4.28 records this port's deliberate ascending-label paint order for
//! the `LabelMapFilter` family, because ITK dispatches
//! `ThreadedProcessLabelObject` from a multithreaded loop
//! (`itkLabelMapFilter.hxx:83-117`) and leaves the winner of an overlap
//! unspecified. Overlapping objects **are** representable in a
//! [`LabelMap`] — that is what
//! [`crate::filters::label_map::label_unique_label_map`] exists to remove — and this
//! filter's input map is whatever the caller hands it, so overlaps are possible
//! here. They are nevertheless *unobservable*: every object is painted with the
//! same `m_ForegroundValue`, so the pixel a two-object overlap ends up with does
//! not depend on which object wrote it last. Order is therefore irrelevant, and
//! this port's ascending-label walk is bit-identical to any thread interleaving
//! ITK can produce. Pinned by
//! `overlapping_objects_are_painted_with_the_same_foreground`.

use crate::core::{Image, LabelMap, PixelId};

use crate::filters::error::Result;
use crate::filters::label_map::{object_offsets, require_label_pixel_id, strides};
use crate::filters::quantize_to_pixel_type;

/// `itk::LabelMapToBinaryImageFilter`: `foreground_value` inside every label
/// object, `background_value` everywhere else, as a [`PixelId::UInt8`] image
/// carrying the map's geometry.
///
/// Both values are cast to `uint8_t` before use, as the yaml's `pixeltype:
/// Output` members are. SimpleITK's defaults are `background_value = 0.0` and
/// `foreground_value = 1.0`.
pub fn label_map_to_binary(
    map: &LabelMap,
    background_value: f64,
    foreground_value: f64,
) -> Result<Image> {
    require_label_pixel_id(map)?;

    let background = quantize_to_pixel_type(PixelId::UInt8, background_value) as u8;
    let foreground = quantize_to_pixel_type(PixelId::UInt8, foreground_value) as u8;

    let size = map.size();
    let total: usize = size.iter().product();
    let mut data = vec![background; total];

    let st = strides(size);
    for object in map.label_objects() {
        for off in object_offsets(object, &st) {
            data[off] = foreground;
        }
    }

    let mut out = Image::from_vec(size, data)?;
    map.apply_geometry_to(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{LabelObject, PixelId};
    use crate::filters::error::FilterError;
    use crate::filters::label_map::{BinaryImageToLabelMapSettings, binary_image_to_label_map};

    /// A 2-D line: `(start index, length)`.
    type Line2 = ([i64; 2], i64);

    fn map_of(size: &[usize], background: i64, objects: &[(i64, &[Line2])]) -> LabelMap {
        let mut map = LabelMap::new(size, PixelId::UInt8, background).unwrap();
        for (label, lines) in objects {
            for (index, length) in *lines {
                map.set_line(index, *length, *label).unwrap();
            }
        }
        map
    }

    fn pixels(img: &Image) -> Vec<u8> {
        img.scalar_slice::<u8>().unwrap().to_vec()
    }

    #[test]
    fn every_object_is_foreground_and_the_rest_is_background() {
        let map = map_of(&[4, 2], 0, &[(1, &[([0, 0], 2)]), (7, &[([2, 1], 2)])]);
        let out = label_map_to_binary(&map, 0.0, 1.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        assert_eq!(pixels(&out), vec![1, 1, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn foreground_and_background_are_cast_to_uint8() {
        // `static_cast<uint8_t>(9.7)` is 9; -1.0 and 300.0 are out-of-range
        // C++ casts, which this port saturates (see `Scalar::from_f64`).
        let map = map_of(&[3, 1], 0, &[(1, &[([1, 0], 1)])]);
        assert_eq!(
            pixels(&label_map_to_binary(&map, 2.9, 9.7).unwrap()),
            vec![2, 9, 2]
        );
        assert_eq!(
            pixels(&label_map_to_binary(&map, -1.0, 300.0).unwrap()),
            vec![0, 255, 0]
        );
    }

    #[test]
    fn itks_own_defaults_differ_from_the_yamls() {
        // ITK: background NonpositiveMin<uint8> == 0, foreground max() == 255.
        // The yaml's foreground default is 1.0.
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        assert_eq!(
            pixels(&label_map_to_binary(&map, 0.0, 255.0).unwrap()),
            vec![255, 0]
        );
        assert_eq!(
            pixels(&label_map_to_binary(&map, 0.0, 1.0).unwrap()),
            vec![1, 0]
        );
    }

    #[test]
    fn an_empty_map_is_all_background() {
        let map = map_of(&[3, 2], 0, &[]);
        assert_eq!(
            pixels(&label_map_to_binary(&map, 4.0, 1.0).unwrap()),
            vec![4; 6]
        );
    }

    #[test]
    fn a_map_holding_only_an_empty_object_is_all_background() {
        // `label_unique_label_map` leaves fully-overlapped objects behind with
        // zero lines; they paint nothing.
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 0).unwrap();
        map.add_label_object(LabelObject::new(1, 2).unwrap())
            .unwrap();
        assert_eq!(map.number_of_label_objects(), 1);
        assert_eq!(
            pixels(&label_map_to_binary(&map, 0.0, 1.0).unwrap()),
            vec![0, 0, 0]
        );
    }

    #[test]
    fn the_maps_background_value_is_not_the_output_background_value() {
        // The map's own background label (7) never reaches the output; the
        // output background is the filter's parameter.
        let map = map_of(&[3, 1], 7, &[(1, &[([0, 0], 1)])]);
        assert_eq!(
            pixels(&label_map_to_binary(&map, 0.0, 1.0).unwrap()),
            vec![1, 0, 0]
        );
    }

    #[test]
    fn overlapping_objects_are_painted_with_the_same_foreground() {
        // Object 1 covers [0, 3); object 2 covers [2, 5). Paint order cannot
        // matter: both write `foreground`.
        let map = map_of(&[6, 1], 0, &[(1, &[([0, 0], 3)]), (2, &[([2, 0], 3)])]);
        assert_eq!(
            pixels(&label_map_to_binary(&map, 0.0, 1.0).unwrap()),
            vec![1, 1, 1, 1, 1, 0]
        );
    }

    #[test]
    fn geometry_comes_from_the_map() {
        let mut map = map_of(&[2, 2], 0, &[(1, &[([0, 0], 1)])]);
        let mut geom = Image::new(&[2, 2], PixelId::UInt8);
        geom.set_spacing(&[0.5, 2.0]).unwrap();
        geom.set_origin(&[-1.0, 3.0]).unwrap();
        map.copy_geometry_from(&geom).unwrap();

        let out = label_map_to_binary(&map, 0.0, 1.0).unwrap();
        assert_eq!(out.spacing(), &[0.5, 2.0]);
        assert_eq!(out.origin(), &[-1.0, 3.0]);
    }

    #[test]
    fn works_in_three_dimensions() {
        let mut map = LabelMap::new(&[2, 2, 2], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 1, 1], 2, 3).unwrap();
        let out = label_map_to_binary(&map, 0.0, 1.0).unwrap();
        assert_eq!(pixels(&out), vec![0, 0, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn rejects_a_signed_label_type() {
        let map = LabelMap::new(&[2, 2], PixelId::Int16, 0).unwrap();
        assert_eq!(
            label_map_to_binary(&map, 0.0, 1.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int16
            ))
        );
    }

    // ---- round trip against `binary_image_to_label_map` --------------------

    #[test]
    fn binary_image_to_label_map_then_back_is_the_identity() {
        // `binary_image_to_label_map`'s defaults are foreground 1 / background 0
        // and it labels the connected components of the pixels equal to 1; every
        // such pixel lands in exactly one object, so painting them all back with
        // `foreground_value = 1` reproduces the input.
        let img = Image::from_vec(&[4, 3], vec![1u8, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1]).unwrap();
        let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(label_map_to_binary(&map, 0.0, 1.0).unwrap(), img);
    }

    #[test]
    fn round_trip_holds_for_an_all_background_and_an_all_foreground_image() {
        for value in [0u8, 1u8] {
            let img = Image::from_vec(&[3, 2], vec![value; 6]).unwrap();
            let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
            assert_eq!(label_map_to_binary(&map, 0.0, 1.0).unwrap(), img);
        }
    }

    #[test]
    fn round_trip_holds_under_full_connectivity_and_a_moved_map_background() {
        // A non-zero `output_background_value` only renumbers the labels, and
        // this filter ignores labels entirely.
        let img = Image::from_vec(&[3, 3], vec![1u8, 0, 1, 0, 1, 0, 1, 0, 1]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            fully_connected: true,
            output_background_value: 5.0,
            ..Default::default()
        };
        let (map, _) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(map.background(), 5);
        assert_eq!(label_map_to_binary(&map, 0.0, 1.0).unwrap(), img);
    }

    #[test]
    fn round_trip_carries_geometry() {
        let mut img = Image::from_vec(&[2, 2], vec![1u8, 0, 0, 1]).unwrap();
        img.set_spacing(&[3.0, 4.0]).unwrap();
        img.set_origin(&[1.0, 2.0]).unwrap();
        let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(label_map_to_binary(&map, 0.0, 1.0).unwrap(), img);
    }
}
