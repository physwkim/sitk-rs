//! `itk::LabelMapMaskImageFilter`: mask a feature image with one label object of
//! a [`LabelMap`].
//!
//! Ported from `Modules/Filtering/LabelMap/include/itkLabelMapMaskImageFilter.h`
//! / `.hxx` and `Code/BasicFilters/yaml/LabelMapMaskImageFilter.yaml`.
//!
//! ## The four `Label` x `Negated` combinations, as one rule
//!
//! Write `B` for `map.background() == label` and `N` for `negated`.
//! `GenerateData` (`itkLabelMapMaskImageFilter.hxx:233-300`) runs two passes:
//!
//! 1. `DynamicThreadedGenerateData` (`.hxx:304-330`, the `B ^ N` test at `.hxx:313`) fills the whole output
//!    region: with the **feature** image when `B ^ N`, otherwise with
//!    `m_BackgroundValue`.
//! 2. Then it repaints one set of pixels with the *other* value:
//!    - when `B` (`.hxx:250-258`), the set is **every** label object, painted by
//!      `LabelMapFilter`'s threaded dispatch into `ThreadedProcessLabelObject`
//!      (`.hxx:334-376`), which writes `m_BackgroundValue` if `!N` and the
//!      feature pixel if `N`;
//!    - when `!B` (`.hxx:259-299`), the set is the **single** object carrying
//!      `m_Label`, painted with the feature pixel if `!N` and
//!      `m_BackgroundValue` if `N`.
//!
//! Both passes collapse to one statement. Let
//!
//! > `fill_is_feature = B ^ N`, and
//! > `object_set = if B { every object } else { the object labelled `label` }`.
//!
//! Then the output is `object_set` painted with `!fill_is_feature`'s value on a
//! background of `fill_is_feature`'s value. The four combinations follow:
//!
//! | `B` | `N` | fill | `object_set` gets | meaning |
//! |-----|-----|------|-------------------|---------|
//! | no  | no  | background | feature | keep only `label`'s pixels |
//! | no  | yes | feature | background | erase only `label`'s pixels |
//! | yes | no  | feature | background | keep only the map's background zone |
//! | yes | yes | background | feature | erase only the map's background zone |
//!
//! `!N` selects, `N` excludes; naming the map's *background* value as `Label`
//! addresses the implicit, object-less zone rather than an object.
//!
//! A missing label is an error. `GetLabelObject` throws "No label object with
//! label L." (`itkLabelMap.hxx:109-126`) from both `GenerateData`'s `!B` branch
//! and, under `Crop`, from `GenerateOutputInformation` — so
//! `label != background && !map.has_label(label)` never produces an image.
//!
//! ## `Crop` and `CropBorder`
//!
//! `GenerateOutputInformation` (`.hxx:60-222`) narrows the output's largest
//! possible region when `Crop` is on. It computes the axis-aligned bounding box
//! of the run-length lines of exactly `object_set` above — `.hxx:105-151` walks
//! the objects whose label is *not* `m_Label` when `N && B`, and `.hxx:164-201`
//! walks the single `m_Label` object when `!N && !B`. Under `N && B` that
//! `!= m_Label` test skips nothing here: [`LabelMap`] enforces
//! `background ∉ objects.keys()` at its insert seam, so it *is* every object.
//! (ITK's own container admits an object keyed on the background value, which
//! this filter would then silently skip and every `GetLabelObject` would throw
//! on — see [`crate::label_map`].) It then applies
//! `ImageRegion::PadByRadius(m_CropBorder)`
//! (`itkImageRegion.hxx`: `index[i] -= border[i]; size[i] += 2*border[i]`) — so
//! the border is **per axis and on both sides** — and clips the result with
//! `ImageRegion::Crop(input->GetLargestPossibleRegion())` (`.hxx:207-208`), which clamps the start
//! to `0` and the end to the image size. The border can therefore never push the
//! region outside the input.
//!
//! The remaining two combinations are the ones where the kept region is
//! unbounded — it contains the object-less background zone — and they are
//! exactly the two where `fill_is_feature` is true. Upstream detects them
//! individually (`.hxx:102-103` for `N && !B`, `.hxx:161-162` for `!N && B`),
//! `itkWarningMacro`s *"Cropping according to background label is not yet
//! implemented. The full image will be used."*, and leaves `cropRegion` at the
//! input's largest possible region. This port has no warning channel; it
//! silently uses the full region, as upstream's own code path does. Pinned by
//! `crop_is_ignored_for_the_two_unbounded_combinations`.
//!
//! The cropped output keeps the input's spacing and direction, and its origin
//! becomes the physical point of the crop region's start index: SimpleITK's
//! `FixNonZeroIndex` (`sitkImageFilter.h:62-89`) rewrites any non-zero region
//! index into the origin before wrapping the ITK image, because "Simple ITK must
//! use a zero based index".
//!
//! ## Pixel and parameter types
//!
//! `pixel_types: LabelPixelIDTypeList` restricts the map to the unsigned integer
//! label types; `pixel_types2: typelist2::append<BasicPixelIDTypeList,
//! ComplexPixelIDTypeList>::type` admits any scalar or complex feature image,
//! and no vector one. The output has the feature image's pixel type and the
//! **label map's** geometry (the map is input 0, so `ImageToImageFilter`'s
//! default `GenerateOutputInformation` copies its spacing/origin/direction).
//! SimpleITK checks the two inputs' dimension and size match.
//!
//! `Label` is a `uint64_t` with `pixeltype: Input`, so it is
//! `static_cast<LabelType>`-ed — a modular truncation, not a clamp: `Label = 300`
//! against a `LabelUInt8` map addresses label `44`. Pinned by
//! `label_is_truncated_modulo_the_label_type`.
//!
//! ## Upstream findings
//!
//! 1. **`BackgroundValue` is cast to the *label* type, not to the output type.**
//!    The yaml marks it `pixeltype: Output`, and
//!    `ExecuteInternalSetITKFilterParameters.cxx.jinja` expands that to
//!    `static_cast<typename OutputImageType::PixelType>(this->m_BackgroundValue)`.
//!    But `LabelMapMaskImageFilter.yaml` declares neither `output_image_type` nor
//!    `output_pixel_type`, so `sitkDualImageFilterTemplate.cxx.jinja`'s fallback
//!    `using OutputImageType = InputImageType;` makes `OutputImageType` the
//!    **`itk::LabelMap`**, whose `PixelType` is its `LabelType`. The `double` a
//!    caller passes is therefore narrowed to `uint8_t`/`uint16_t`/… first, and
//!    only then converted to the feature image's pixel type by
//!    `SetBackgroundValue`'s parameter. The yaml author knew the alias was wrong
//!    — the `FeatureImage` input carries a `custom_itk_cast` spelling
//!    `typename FilterType::OutputImageType` explicitly — but left the member
//!    alone. Consequences: a negative `background_value` cannot reach a signed
//!    feature image, and a `background_value` above the label type's range
//!    cannot reach a wider feature image. This port reproduces the two-stage
//!    cast; pinned by `background_value_is_narrowed_through_the_label_type` and
//!    `a_negative_background_value_cannot_reach_a_signed_feature_image`.
//! 2. **`testIdxIsInside` is dead code.** Both write loops that can leave the
//!    crop region guard themselves with
//!    `m_Crop && (input->GetBackgroundValue() == m_Label) ^ m_Negated`
//!    (`.hxx:281` and `.hxx:346`; `^` binds tighter than `&&`, so it reads
//!    `m_Crop && (B ^ N)`). `B ^ N` is true only in the two combinations where
//!    `GenerateOutputInformation` bailed out to the full image — where the test
//!    can never fail. In the two combinations where `Crop` really does narrow the
//!    region, the guard is `false` and the painted object set is precisely what
//!    the bounding box was computed from, so no write escapes anyway. The guard
//!    never changes an output pixel. This port has no equivalent; the
//!    fill-then-repaint formulation above cannot write out of region.
//! 3. **An empty `object_set` under `Crop` produces a garbage region.** The
//!    bounding-box loops seed `mins` with `NumericTraits<IndexValueType>::max()`
//!    and `maxs` with `::NonpositiveMin()` and never guard against zero lines
//!    (`.hxx:108-109`, `.hxx:169-170`). With no object — an empty map under
//!    `N && B`, or a zero-line object under `!N && !B`, which
//!    `LabelUniqueLabelMapFilter` readily produces — `regionSize[i] =
//!    maxs[i] - mins[i] + 1` is a signed overflow (undefined behaviour; in
//!    practice `2`), `ImageRegion::Crop` finds no intersection and returns
//!    `false` without touching the region, and the filter allocates a 2x2 image
//!    at index `2^63 - 1` with a nonsense origin. This port refuses instead, with
//!    [`FilterError::LabelMapMaskEmptyCropRegion`]; pinned by
//!    `crop_on_an_empty_object_set_is_an_error`.
//!
//! ## Paint order
//!
//! Ledger §4.28's ascending-label paint order for the `LabelMapFilter` family is
//! unobservable here, as it is in [`crate::label_map_to_binary`]. Overlapping
//! objects are representable in a [`LabelMap`], and the `B` branch does paint
//! every object; but every object is painted with the *same* value — either
//! `m_BackgroundValue` or the feature pixel underneath — so which object writes a
//! shared pixel last cannot change it. The `!B` branch paints a single object.
//! Pinned by `overlapping_objects_paint_the_same_value`.

use sitk_core::{Image, LabelMap, LabelObject, PixelId, Scalar, dispatch_scalar};

use crate::error::{FilterError, Result};
use crate::label_map::{object_offsets, require_label_pixel_id, require_same_size, strides};
use crate::quantize_to_pixel_type;

/// The five settings `LabelMapMaskImageFilter.yaml` exposes, with its defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelMapMaskSettings {
    /// The label to keep (`negated == false`) or to erase (`negated == true`),
    /// `static_cast<LabelType>`-ed before it is compared. `1`.
    pub label: u64,
    /// The value written where the feature image is masked out, narrowed first
    /// to the map's label type — see the module docs' upstream finding 1. `0.0`.
    pub background_value: f64,
    /// Erase `label` instead of keeping it. `false`.
    pub negated: bool,
    /// Shrink the output to the bounding box of the kept objects. `false`.
    pub crop: bool,
    /// Per-axis border added on **both** sides of that bounding box before it is
    /// clipped to the input region. `[0, 0, 0]`.
    pub crop_border: Vec<usize>,
}

impl Default for LabelMapMaskSettings {
    fn default() -> Self {
        LabelMapMaskSettings {
            label: 1,
            background_value: 0.0,
            negated: false,
            crop: false,
            crop_border: vec![0, 0, 0],
        }
    }
}

/// `static_cast<LabelType>(uint64_t)`: a modular truncation. `require_label_pixel_id`
/// has already ruled out every id but the four unsigned ones, and their values
/// all fit an `i64` — `UInt64`'s own bound is clamped to `i64::MAX` by
/// [`PixelId::integer_scalar_bounds`], so a `u64` label above `i64::MAX` is
/// unrepresentable in a [`LabelMap`] and lands on a label no object can carry.
fn cast_to_label_type(id: PixelId, label: u64) -> i64 {
    match id {
        PixelId::UInt8 => (label as u8) as i64,
        PixelId::UInt16 => (label as u16) as i64,
        PixelId::UInt32 => (label as u32) as i64,
        _ => label as i64,
    }
}

/// `static_cast<OutputPixelType>(v)` where `v` already went through
/// `static_cast<LabelType>` — a non-negative integer inside an unsigned label
/// type's range. C++ narrows an integer to a narrower integer modulo `2^bits`
/// (well-defined since C++20), and converts it to a float exactly-or-rounded;
/// `as` reproduces both. The complex feature types arrive here as their
/// [`PixelId::component_id`], `Float32`/`Float64`, which is what
/// `std::complex<T>`'s converting constructor sees.
fn cast_label_value_to_component(id: PixelId, v: f64) -> f64 {
    let u = v as u64;
    match id {
        PixelId::UInt8 => (u as u8) as f64,
        PixelId::Int8 => (u as i8) as f64,
        PixelId::UInt16 => (u as u16) as f64,
        PixelId::Int16 => (u as i16) as f64,
        PixelId::UInt32 => (u as u32) as f64,
        PixelId::Int32 => (u as i32) as f64,
        PixelId::UInt64 => u as f64,
        PixelId::Int64 => (u as i64) as f64,
        PixelId::Float32 => (u as f32) as f64,
        PixelId::Float64 => u as f64,
        _ => unreachable!("PixelId::component_id() always returns a scalar variant"),
    }
}

/// `itk::LabelMapMaskImageFilter`: keep (or erase) the pixels of one label
/// object of `map` in `feature`, replacing the rest with
/// `settings.background_value`.
///
/// The output has `feature`'s pixel type and `map`'s geometry. See the [module
/// docs](self) for the `Label` x `Negated` table, `Crop`'s bounding box, and
/// three upstream findings.
///
/// # Errors
///
/// - [`FilterError::RequiresUnsignedIntegerPixelType`] — `map`'s label type is
///   outside `LabelPixelIDTypeList`.
/// - [`FilterError::RequiresNonVectorPixelType`] — `feature` is a vector image;
///   `pixel_types2` admits only the basic and complex types.
/// - [`FilterError::SizeMismatch`] — SimpleITK's `CheckImageMatchingSize`.
/// - [`FilterError::DimensionLength`] — `crop_border` is shorter than the image
///   dimension, which is what `sitkSTLVectorToITK` throws on
///   (`sitkTemplateFunctions.h:96-110`). Extra entries are ignored, so the
///   3-element default works for a 2-D image.
/// - [`FilterError::LabelMapMaskLabelNotFound`] — `settings.label` is neither the
///   map's background value nor the label of an object.
/// - [`FilterError::LabelMapMaskEmptyCropRegion`] — `crop` is on and the object
///   set to bound has no lines.
pub fn label_map_mask(
    map: &LabelMap,
    feature: &Image,
    settings: &LabelMapMaskSettings,
) -> Result<Image> {
    require_label_pixel_id(map)?;
    let feature_id = feature.pixel_id();
    if feature_id.is_vector() {
        return Err(FilterError::RequiresNonVectorPixelType(feature_id));
    }
    require_same_size(map, feature)?;

    let dim = map.dimension();
    if settings.crop_border.len() < dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: settings.crop_border.len(),
        });
    }

    let label = cast_to_label_type(map.pixel_id(), settings.label);
    let background_is_label = map.background() == label;
    let fill_is_feature = background_is_label ^ settings.negated;

    let objects: Vec<&LabelObject> = if background_is_label {
        map.label_objects().collect()
    } else {
        let object = map
            .label_object(label)
            .ok_or(FilterError::LabelMapMaskLabelNotFound { label })?;
        vec![object]
    };

    let st = strides(map.size());
    let offsets: Vec<usize> = objects
        .iter()
        .flat_map(|object| object_offsets(object, &st))
        .collect();

    // Two-stage cast; upstream finding 1.
    let background = cast_label_value_to_component(
        feature_id.component_id(),
        quantize_to_pixel_type(map.pixel_id(), settings.background_value),
    );

    let mut out = dispatch_scalar!(
        feature_id,
        masked_image,
        feature,
        background,
        fill_is_feature,
        &offsets,
    )?;
    map.apply_geometry_to(&mut out)?;

    if settings.crop && !fill_is_feature {
        let (start, size) = crop_region(&objects, map.size(), &settings.crop_border)
            .ok_or(FilterError::LabelMapMaskEmptyCropRegion)?;
        out = dispatch_scalar!(feature_id, crop_image, &out, &start, &size)?;
    }
    Ok(out)
}

/// Pass 1 (fill the whole region) and pass 2 (repaint `offsets` with the other
/// value), fused. Works on interleaved *components*, so a complex feature image
/// carries both of its components across untouched — `static_cast<std::complex<T>>`
/// of the background scalar gives `(background, 0)`.
fn masked_image<T: Scalar>(
    feature: &Image,
    background: f64,
    fill_is_feature: bool,
    offsets: &[usize],
) -> Result<Image> {
    let src = feature.component_slice::<T>()?;
    let stride = feature.buffer_stride();
    let background_pixel = [T::from_f64(background), T::from_f64(0.0)];
    let background_pixel = &background_pixel[..stride];

    let mut data: Vec<T> = if fill_is_feature {
        src.to_vec()
    } else {
        background_pixel.repeat(src.len() / stride)
    };
    for &pixel in offsets {
        let range = pixel * stride..pixel * stride + stride;
        if fill_is_feature {
            data[range].copy_from_slice(background_pixel);
        } else {
            data[range.clone()].copy_from_slice(&src[range]);
        }
    }

    let mut out = Image::new(feature.size(), feature.pixel_id());
    *out.component_vec_mut::<T>()? = data;
    Ok(out)
}

/// The bounding box of `objects`' lines, padded by `border` on both sides of
/// each axis and clipped to `size`. `None` when no object has a line, which is
/// upstream finding 3.
fn crop_region(
    objects: &[&LabelObject],
    size: &[usize],
    border: &[usize],
) -> Option<(Vec<i64>, Vec<usize>)> {
    let dim = size.len();
    let mut mins = vec![i64::MAX; dim];
    let mut maxs = vec![i64::MIN; dim];
    let mut seen = false;
    for object in objects {
        for line in object.lines() {
            seen = true;
            let idx = line.index();
            for d in 0..dim {
                mins[d] = mins[d].min(idx[d]);
                maxs[d] = maxs[d].max(idx[d]);
            }
            maxs[0] = maxs[0].max(idx[0] + line.length() - 1);
        }
    }
    if !seen {
        return None;
    }
    let mut start = vec![0i64; dim];
    let mut extent = vec![0usize; dim];
    for d in 0..dim {
        let radius = i64::try_from(border[d]).unwrap_or(i64::MAX);
        let lo = mins[d].saturating_sub(radius).max(0);
        let hi = maxs[d].saturating_add(radius).min(size[d] as i64 - 1);
        start[d] = lo;
        extent[d] = (hi - lo + 1) as usize;
    }
    Some((start, extent))
}

/// Extract `size` pixels starting at `start`, moving the start index into the
/// origin the way SimpleITK's `FixNonZeroIndex` (`sitkImageFilter.h:62-89`)
/// does.
fn crop_image<T: Scalar>(img: &Image, start: &[i64], size: &[usize]) -> Result<Image> {
    let dim = size.len();
    let stride = img.buffer_stride();
    let src = img.component_slice::<T>()?;
    let src_st = strides(img.size());

    let total: usize = size.iter().product();
    let mut data: Vec<T> = Vec::with_capacity(total * stride);
    let mut idx = vec![0usize; dim];
    for _ in 0..total {
        let pixel: usize = (0..dim)
            .map(|d| (idx[d] + start[d] as usize) * src_st[d])
            .sum();
        data.extend_from_slice(&src[pixel * stride..pixel * stride + stride]);
        for d in 0..dim {
            idx[d] += 1;
            if idx[d] < size[d] {
                break;
            }
            idx[d] = 0;
        }
    }

    let origin_index: Vec<f64> = start.iter().map(|&i| i as f64).collect();
    let origin = img.continuous_index_to_physical_point(&origin_index);

    let mut out = Image::new(size, img.pixel_id());
    *out.component_vec_mut::<T>()? = data;
    out.set_spacing(img.spacing())?;
    out.set_direction(img.direction())?;
    out.set_origin(&origin)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::Complex;

    /// A 2-D line: `(start index, length)`.
    type Line2 = ([i64; 2], i64);

    /// A 2-D map over `size` whose objects are given as `(label, &[(index, length)])`.
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

    /// A 3x2 feature image with distinct values, and a map whose object 1 covers
    /// the first two pixels of row 0.
    ///
    /// feature:  10 20 30 / 40 50 60
    /// object 1: [(0,0), len 2]
    fn fixture() -> (LabelMap, Image) {
        let map = map_of(&[3, 2], 0, &[(1, &[([0, 0], 2)])]);
        let feature = Image::from_vec(&[3, 2], vec![10u8, 20, 30, 40, 50, 60]).unwrap();
        (map, feature)
    }

    fn settings(label: u64, negated: bool) -> LabelMapMaskSettings {
        LabelMapMaskSettings {
            label,
            negated,
            ..Default::default()
        }
    }

    // ---- the four Label x Negated combinations ----------------------------

    #[test]
    fn label_not_background_not_negated_keeps_only_that_object() {
        let (map, feature) = fixture();
        let out = label_map_mask(&map, &feature, &settings(1, false)).unwrap();
        assert_eq!(pixels(&out), vec![10, 20, 0, 0, 0, 0]);
    }

    #[test]
    fn label_not_background_negated_erases_only_that_object() {
        let (map, feature) = fixture();
        let out = label_map_mask(&map, &feature, &settings(1, true)).unwrap();
        assert_eq!(pixels(&out), vec![0, 0, 30, 40, 50, 60]);
    }

    #[test]
    fn label_is_background_not_negated_keeps_only_the_background_zone() {
        let (map, feature) = fixture();
        // Label 0 == map.background(): the kept region is every pixel no object
        // covers, and both objects' pixels become background.
        let out = label_map_mask(&map, &feature, &settings(0, false)).unwrap();
        assert_eq!(pixels(&out), vec![0, 0, 30, 40, 50, 60]);
    }

    #[test]
    fn label_is_background_negated_erases_only_the_background_zone() {
        let (map, feature) = fixture();
        let out = label_map_mask(&map, &feature, &settings(0, true)).unwrap();
        assert_eq!(pixels(&out), vec![10, 20, 0, 0, 0, 0]);
    }

    #[test]
    fn the_background_pair_covers_every_object_not_just_one() {
        // Two objects. `label == background` addresses the object-less zone, so
        // both objects are repainted, unlike the single-object branch.
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (2, &[([2, 0], 1)])]);
        let feature = Image::from_vec(&[4, 1], vec![10u8, 20, 30, 40]).unwrap();
        let keep_bg = label_map_mask(&map, &feature, &settings(0, false)).unwrap();
        assert_eq!(pixels(&keep_bg), vec![0, 20, 0, 40]);
        let erase_bg = label_map_mask(&map, &feature, &settings(0, true)).unwrap();
        assert_eq!(pixels(&erase_bg), vec![10, 0, 30, 0]);
        // ... while the single-object branch touches only object 1.
        let keep_one = label_map_mask(&map, &feature, &settings(1, false)).unwrap();
        assert_eq!(pixels(&keep_one), vec![10, 0, 0, 0]);
    }

    #[test]
    fn a_non_zero_map_background_moves_which_label_addresses_the_zone() {
        let map = map_of(&[3, 1], 7, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[3, 1], vec![10u8, 20, 30]).unwrap();
        let out = label_map_mask(&map, &feature, &settings(7, false)).unwrap();
        assert_eq!(pixels(&out), vec![0, 20, 30]);
    }

    // ---- BackgroundValue ---------------------------------------------------

    #[test]
    fn background_value_fills_the_masked_out_pixels() {
        let (map, feature) = fixture();
        let s = LabelMapMaskSettings {
            background_value: 99.0,
            ..settings(1, false)
        };
        assert_eq!(
            pixels(&label_map_mask(&map, &feature, &s).unwrap()),
            vec![10, 20, 99, 99, 99, 99]
        );
    }

    #[test]
    fn background_value_is_narrowed_through_the_label_type() {
        // Upstream finding 1. The map's label type is `UInt8`, so a
        // `background_value` of 300 cannot reach the `UInt16` feature image: it
        // is `static_cast<uint8_t>`-ed to 255 first (upstream's cast of an
        // out-of-range double is UB; this port saturates), and only then widened.
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[2, 1], vec![1000u16, 2000]).unwrap();
        let s = LabelMapMaskSettings {
            background_value: 300.0,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.scalar_slice::<u16>().unwrap(), &[1000, 255]);
    }

    #[test]
    fn a_negative_background_value_cannot_reach_a_signed_feature_image() {
        // Upstream finding 1 again: `static_cast<uint8_t>(-5.0)` runs before the
        // conversion to `int8_t`. Upstream's cast is UB; this port saturates to 0.
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[2, 1], vec![-7i8, -9]).unwrap();
        let s = LabelMapMaskSettings {
            background_value: -5.0,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.scalar_slice::<i8>().unwrap(), &[-7, 0]);
    }

    #[test]
    fn the_label_typed_background_value_then_wraps_into_a_narrower_output() {
        // `static_cast<uint8_t>(200.0)` is 200; `static_cast<int8_t>(uint8_t(200))`
        // is -56 (well-defined modular narrowing since C++20).
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[2, 1], vec![1i8, 2]).unwrap();
        let s = LabelMapMaskSettings {
            background_value: 200.0,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.scalar_slice::<i8>().unwrap(), &[1, -56]);
    }

    #[test]
    fn background_value_truncates_toward_zero_like_a_static_cast() {
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let s = LabelMapMaskSettings {
            background_value: 3.9,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[1.0, 3.0]);
    }

    // ---- Label's cast ------------------------------------------------------

    #[test]
    fn label_is_truncated_modulo_the_label_type() {
        // `static_cast<uint8_t>(300)` is 44.
        let map = map_of(&[2, 1], 0, &[(44, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[2, 1], vec![10u8, 20]).unwrap();
        let out = label_map_mask(&map, &feature, &settings(300, false)).unwrap();
        assert_eq!(pixels(&out), vec![10, 0]);
    }

    #[test]
    fn a_missing_label_is_an_error() {
        let (map, feature) = fixture();
        for negated in [false, true] {
            assert_eq!(
                label_map_mask(&map, &feature, &settings(9, negated)),
                Err(FilterError::LabelMapMaskLabelNotFound { label: 9 })
            );
        }
    }

    #[test]
    fn a_missing_label_is_an_error_even_with_crop_on() {
        let (map, feature) = fixture();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(9, false)
        };
        assert_eq!(
            label_map_mask(&map, &feature, &s),
            Err(FilterError::LabelMapMaskLabelNotFound { label: 9 })
        );
    }

    // ---- Crop / CropBorder -------------------------------------------------

    /// A 5x5 feature image of `y * 5 + x`, and a map whose object 1 is the
    /// 2x2 block at (1,1)..(2,2).
    fn crop_fixture() -> (LabelMap, Image) {
        let mut map = LabelMap::new(&[5, 5], PixelId::UInt8, 0).unwrap();
        map.set_line(&[1, 1], 2, 1).unwrap();
        map.set_line(&[1, 2], 2, 1).unwrap();
        let data: Vec<u8> = (0..25u8).collect();
        (map, Image::from_vec(&[5, 5], data).unwrap())
    }

    #[test]
    fn crop_without_border_is_the_objects_bounding_box() {
        let (map, feature) = crop_fixture();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(pixels(&out), vec![6, 7, 11, 12]);
        assert_eq!(out.origin(), &[1.0, 1.0]);
    }

    #[test]
    fn crop_with_a_border_grows_the_box_on_both_sides_of_each_axis() {
        let (map, feature) = crop_fixture();
        let s = LabelMapMaskSettings {
            crop: true,
            crop_border: vec![1, 1, 1],
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        // (1,1)..(2,2) padded by 1 -> (0,0)..(3,3), a 4x4 region. The pixels
        // outside the object are the background value.
        assert_eq!(out.size(), &[4, 4]);
        assert_eq!(out.origin(), &[0.0, 0.0]);
        #[rustfmt::skip]
        let expected = vec![
            0, 0, 0, 0,
            0, 6, 7, 0,
            0, 11, 12, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(pixels(&out), expected);
    }

    #[test]
    fn crop_border_is_per_axis() {
        let (map, feature) = crop_fixture();
        let s = LabelMapMaskSettings {
            crop: true,
            crop_border: vec![2, 0, 0],
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        // x: (1..2) padded by 2 -> (-1..4), clipped to (0..4) -> width 5.
        // y: unchanged (1..2) -> height 2.
        assert_eq!(out.size(), &[5, 2]);
        assert_eq!(out.origin(), &[0.0, 1.0]);
        assert_eq!(pixels(&out), vec![0, 6, 7, 0, 0, 0, 11, 12, 0, 0]);
    }

    #[test]
    fn crop_border_is_clamped_at_the_image_edge() {
        let (map, feature) = crop_fixture();
        let s = LabelMapMaskSettings {
            crop: true,
            crop_border: vec![100, 100, 100],
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[5, 5]);
        assert_eq!(out.origin(), &[0.0, 0.0]);
        // Nothing was cropped away, so this is the uncropped mask.
        let uncropped = label_map_mask(&map, &feature, &settings(1, false)).unwrap();
        assert_eq!(out, uncropped);
    }

    #[test]
    fn crop_clamps_a_box_that_already_touches_the_edge() {
        // Object 1 sits in the corner (0,0); a border of 1 cannot go below 0, so
        // the box is (0,0)..(1,1) rather than (-1,-1)..(1,1).
        let mut map = LabelMap::new(&[4, 4], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 0], 1, 1).unwrap();
        let data: Vec<u8> = (1..17u8).collect();
        let feature = Image::from_vec(&[4, 4], data).unwrap();
        let s = LabelMapMaskSettings {
            crop: true,
            crop_border: vec![1, 1, 1],
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.origin(), &[0.0, 0.0]);
        // Only the object's own pixel keeps its feature value (1); the three
        // border pixels are background.
        assert_eq!(pixels(&out), vec![1, 0, 0, 0]);
    }

    #[test]
    fn crop_moves_the_start_index_into_the_origin_under_a_real_geometry() {
        let (map, mut feature) = crop_fixture();
        let mut geom = Image::new(&[5, 5], PixelId::UInt8);
        geom.set_spacing(&[0.5, 2.0]).unwrap();
        geom.set_origin(&[-1.0, 3.0]).unwrap();
        let mut map = map;
        map.copy_geometry_from(&geom).unwrap();
        feature.copy_geometry_from(&geom);

        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.spacing(), &[0.5, 2.0]);
        // origin + spacing * index, identity direction.
        assert_eq!(out.origin(), &[-0.5, 5.0]);
    }

    #[test]
    fn crop_under_negation_bounds_every_object_when_label_is_the_background() {
        // `N && B`: the kept region is every object, so the box spans both.
        let mut map = LabelMap::new(&[5, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[1, 0], 1, 1).unwrap();
        map.set_line(&[3, 0], 1, 2).unwrap();
        let feature = Image::from_vec(&[5, 1], vec![10u8, 20, 30, 40, 50]).unwrap();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(0, true)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[3, 1]);
        assert_eq!(out.origin(), &[1.0, 0.0]);
        assert_eq!(pixels(&out), vec![20, 0, 40]);
    }

    #[test]
    fn crop_is_ignored_for_the_two_unbounded_combinations() {
        // Upstream warns and uses the full image when `fill_is_feature`:
        // (!N && B) and (N && !B).
        let (map, feature) = crop_fixture();
        for (label, negated) in [(0u64, false), (1u64, true)] {
            let mut s = settings(label, negated);
            s.crop_border = vec![1, 1, 1];
            let uncropped = label_map_mask(&map, &feature, &s).unwrap();
            s.crop = true;
            let cropped = label_map_mask(&map, &feature, &s).unwrap();
            assert_eq!(cropped, uncropped, "label={label} negated={negated}");
            assert_eq!(cropped.size(), &[5, 5]);
        }
    }

    #[test]
    fn crop_on_an_empty_object_set_is_an_error() {
        // Upstream finding 3: an empty map under `N && B`.
        let map = map_of(&[3, 2], 0, &[]);
        let feature = Image::from_vec(&[3, 2], vec![1u8; 6]).unwrap();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(0, true)
        };
        assert_eq!(
            label_map_mask(&map, &feature, &s),
            Err(FilterError::LabelMapMaskEmptyCropRegion)
        );
    }

    #[test]
    fn crop_on_a_zero_line_object_is_an_error() {
        // Upstream finding 3: a `LabelUniqueLabelMapFilter`-style empty object
        // under `!N && !B`.
        let mut map = LabelMap::new(&[3, 2], PixelId::UInt8, 0).unwrap();
        map.add_label_object(LabelObject::new(1, 2).unwrap())
            .unwrap();
        let feature = Image::from_vec(&[3, 2], vec![1u8; 6]).unwrap();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(1, false)
        };
        assert_eq!(
            label_map_mask(&map, &feature, &s),
            Err(FilterError::LabelMapMaskEmptyCropRegion)
        );
    }

    #[test]
    fn crop_border_shorter_than_the_dimension_is_an_error() {
        let (map, feature) = fixture();
        let s = LabelMapMaskSettings {
            crop_border: vec![0],
            ..settings(1, false)
        };
        assert_eq!(
            label_map_mask(&map, &feature, &s),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        );
    }

    #[test]
    fn crop_works_in_three_dimensions() {
        let mut map = LabelMap::new(&[3, 3, 3], PixelId::UInt8, 0).unwrap();
        map.set_line(&[1, 1, 1], 1, 1).unwrap();
        let data: Vec<u8> = (0..27u8).collect();
        let feature = Image::from_vec(&[3, 3, 3], data).unwrap();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[1, 1, 1]);
        assert_eq!(pixels(&out), vec![13]);
        assert_eq!(out.origin(), &[1.0, 1.0, 1.0]);
    }

    // ---- degenerate maps ---------------------------------------------------

    #[test]
    fn an_empty_map_without_crop_yields_a_full_background_or_full_feature_image() {
        let map = map_of(&[3, 1], 0, &[]);
        let feature = Image::from_vec(&[3, 1], vec![10u8, 20, 30]).unwrap();
        // label == background, !negated: keep the whole background zone.
        let keep = label_map_mask(&map, &feature, &settings(0, false)).unwrap();
        assert_eq!(pixels(&keep), vec![10, 20, 30]);
        // label == background, negated: erase the whole background zone.
        let erase = label_map_mask(&map, &feature, &settings(0, true)).unwrap();
        assert_eq!(pixels(&erase), vec![0, 0, 0]);
    }

    #[test]
    fn an_empty_map_and_a_non_background_label_is_a_missing_label_error() {
        let map = map_of(&[3, 1], 0, &[]);
        let feature = Image::from_vec(&[3, 1], vec![10u8, 20, 30]).unwrap();
        assert_eq!(
            label_map_mask(&map, &feature, &settings(1, false)),
            Err(FilterError::LabelMapMaskLabelNotFound { label: 1 })
        );
    }

    #[test]
    fn a_zero_line_object_paints_nothing_without_crop() {
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 0).unwrap();
        map.add_label_object(LabelObject::new(1, 2).unwrap())
            .unwrap();
        let feature = Image::from_vec(&[3, 1], vec![10u8, 20, 30]).unwrap();
        let out = label_map_mask(&map, &feature, &settings(1, false)).unwrap();
        assert_eq!(pixels(&out), vec![0, 0, 0]);
    }

    #[test]
    fn overlapping_objects_paint_the_same_value() {
        // Objects 1 and 2 share pixel 2. Under `label == background` both are
        // painted with the background value, so paint order is unobservable.
        let map = map_of(&[5, 1], 0, &[(1, &[([0, 0], 3)]), (2, &[([2, 0], 3)])]);
        let feature = Image::from_vec(&[5, 1], vec![10u8, 20, 30, 40, 50]).unwrap();
        let out = label_map_mask(&map, &feature, &settings(0, false)).unwrap();
        assert_eq!(pixels(&out), vec![0, 0, 0, 0, 0]);
        let erased = label_map_mask(&map, &feature, &settings(0, true)).unwrap();
        assert_eq!(pixels(&erased), vec![10, 20, 30, 40, 50]);
    }

    // ---- pixel types and geometry ------------------------------------------

    #[test]
    fn the_output_pixel_type_and_geometry_come_from_the_right_input() {
        // Pixel type from the feature image, geometry from the label map.
        let mut map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let mut geom = Image::new(&[2, 1], PixelId::UInt8);
        geom.set_spacing(&[7.0, 8.0]).unwrap();
        geom.set_origin(&[1.0, 2.0]).unwrap();
        map.copy_geometry_from(&geom).unwrap();

        let mut feature = Image::from_vec(&[2, 1], vec![1.5f32, 2.5]).unwrap();
        feature.set_spacing(&[100.0, 200.0]).unwrap();
        feature.set_origin(&[-9.0, -9.0]).unwrap();

        let out = label_map_mask(&map, &feature, &settings(1, false)).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[1.5, 0.0]);
        assert_eq!(out.spacing(), &[7.0, 8.0]);
        assert_eq!(out.origin(), &[1.0, 2.0]);
    }

    #[test]
    fn a_complex_feature_image_keeps_both_components() {
        // `pixel_types2` includes `ComplexPixelIDTypeList`.
        // `static_cast<std::complex<float>>(uint8_t(4))` is `(4, 0)`.
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec_complex(
            &[2, 1],
            vec![Complex::new(1.0f32, 2.0), Complex::new(3.0, 4.0)],
        )
        .unwrap();
        let s = LabelMapMaskSettings {
            background_value: 4.0,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.pixel_id(), PixelId::ComplexFloat32);
        assert_eq!(
            out.get_complex::<f32>(&[0, 0]).unwrap(),
            Complex::new(1.0, 2.0)
        );
        assert_eq!(
            out.get_complex::<f32>(&[1, 0]).unwrap(),
            Complex::new(4.0, 0.0)
        );
    }

    #[test]
    fn a_complex_feature_image_survives_a_crop() {
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[1, 0], 1, 1).unwrap();
        let feature = Image::from_vec_complex(
            &[3, 1],
            vec![
                Complex::new(1.0f64, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 6.0),
            ],
        )
        .unwrap();
        let s = LabelMapMaskSettings {
            crop: true,
            ..settings(1, false)
        };
        let out = label_map_mask(&map, &feature, &s).unwrap();
        assert_eq!(out.size(), &[1, 1]);
        assert_eq!(
            out.get_complex::<f64>(&[0, 0]).unwrap(),
            Complex::new(3.0, 4.0)
        );
        assert_eq!(out.origin(), &[1.0, 0.0]);
    }

    #[test]
    fn rejects_a_vector_feature_image() {
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec_vector(&[2, 1], 2, vec![0u8; 4]).unwrap();
        assert_eq!(
            label_map_mask(&map, &feature, &settings(1, false)),
            Err(FilterError::RequiresNonVectorPixelType(
                PixelId::VectorUInt8
            ))
        );
    }

    #[test]
    fn rejects_a_signed_label_type() {
        let map = LabelMap::new(&[2, 1], PixelId::Int16, 0).unwrap();
        let feature = Image::from_vec(&[2, 1], vec![0u8; 2]).unwrap();
        assert_eq!(
            label_map_mask(&map, &feature, &settings(1, false)),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int16
            ))
        );
    }

    #[test]
    fn rejects_a_feature_image_of_a_different_size() {
        let map = map_of(&[2, 1], 0, &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[3, 1], vec![0u8; 3]).unwrap();
        assert_eq!(
            label_map_mask(&map, &feature, &settings(1, false)),
            Err(FilterError::SizeMismatch {
                a: vec![2, 1],
                b: vec![3, 1]
            })
        );
    }
}
