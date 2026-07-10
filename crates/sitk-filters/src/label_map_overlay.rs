//! The three `LabelMap` colouring filters: [`label_map_to_rgb`],
//! [`label_map_overlay`] and [`label_map_contour_overlay`].
//!
//! Ported from `Modules/Filtering/ImageFusion/include/`'s
//! `itkLabelMapToRGBImageFilter.h`/`.hxx`, `itkLabelMapOverlayImageFilter.h`/`.hxx`
//! and `itkLabelMapContourOverlayImageFilter.h`/`.hxx`, which reuse
//! `itkLabelToRGBFunctor.h` / `itkLabelOverlayFunctor.h` — the same functors
//! [`crate::label_to_rgb`] already ports. The palette, the
//! `SetLabelFunctorFromColormap` replacement rule
//! (`sitkLabelFunctorUtils.hxx:38-46`) and its two upstream findings are shared
//! from that module verbatim; only the two facts below are new to the label-map
//! variants.
//!
//! * [`label_map_to_rgb`]'s output component type is always `uint8_t`
//!   (`LabelMapToRGBImageFilter.yaml`'s `output_image_type: itk::VectorImage<
//!   unsigned char, ...>`), so `AddColor`'s
//!   `byte / 255 * NumericTraits<ValueType>::max()` rescaling is the identity.
//! * [`label_map_overlay`] and [`label_map_contour_overlay`] take their output
//!   component type from the **feature** image
//!   (`output_image_type: itk::VectorImage< typename InputImageType2::PixelType,
//!   ... >`), so the palette is rescaled to that type's `max()` exactly as in
//!   [`crate::label_to_rgb::label_overlay`].
//!
//! All three write three components per pixel (`GenerateOutputInformation`
//! forces `SetNumberOfComponentsPerPixel(3)`:
//! `itkLabelMapToRGBImageFilter.hxx:62-80`,
//! `itkLabelMapOverlayImageFilter.hxx:145-163`,
//! `itkLabelMapContourOverlayImageFilter.hxx:273-291`) and take their geometry
//! from the label map, which is the primary input.
//!
//! ## `LabelMapContourOverlayImageFilter`'s contour pipeline
//!
//! `BeforeThreadedGenerateData` (`itkLabelMapContourOverlayImageFilter.hxx:104-214`)
//! builds an `ObjectByObjectLabelMapFilter` whose per-object minipipeline is
//!
//! 1. select the object, auto-crop its bounding box with a `CropBorder` of
//!    `m_DilationRadius + 1` per axis, clipped to the image
//!    (`itkObjectByObjectLabelMapFilter.hxx:198-209`,
//!    `itkAutoCropLabelMapFilter.hxx:100-107`), and rasterize it to a
//!    `[0, 255]` binary image (`m_InternalForegroundValue` is
//!    `NumericTraits<unsigned char>::max()`, `itkObjectByObjectLabelMapFilter.hxx:50`);
//! 2. `BinaryDilateImageFilter` with `FlatStructuringElement::Ball(m_DilationRadius)`
//!    (`m_BoundaryToForeground = false`, `itkBinaryDilateImageFilter.hxx:36`);
//! 3. depending on `Type`: `PLAIN` keeps the dilation; `CONTOUR` subtracts
//!    `BinaryErodeImageFilter` with `Ball(m_ContourThickness)`
//!    (`m_BoundaryToForeground = true`, `itkBinaryErodeImageFilter.hxx:36`) from
//!    it; `SLICE_CONTOUR` does the same subtraction slice by slice along
//!    `m_SliceDimension` with a `(D-1)`-dimensional ball;
//! 4. back to a label object, keeping the input label
//!    (`m_KeepLabels`, `itkObjectByObjectLabelMapFilter.hxx:257-283`).
//!
//! Step 4's "the label has been stolen by a previously split object" branch is
//! unreachable here: `m_BinaryInternalOutput` defaults to `false`
//! (`itkObjectByObjectLabelMapFilter.h:222`), so the result goes through
//! `LabelImageToLabelMapFilter`, which groups by *value* rather than by
//! connectivity — a `{0, 255}` image always yields exactly one label object.
//! This port therefore emits one object per input object directly.
//!
//! Finally `LabelUniqueLabelMapFilter` resolves overlaps, with
//! `ReverseOrdering = (m_Priority == LOW_LABEL_ON_TOP)`
//! (`itkLabelMapContourOverlayImageFilter.hxx:204-207`) — see
//! [`crate::label_map::label_unique_label_map`], which reproduces that filter's
//! own empty-object defect.
//!
//! ## Upstream findings
//!
//! 1. **`SLICE_CONTOUR`'s slice-kernel radius is built with the wrong loop
//!    variable.** `itkLabelMapContourOverlayImageFilter.hxx:156-163` reads
//!    ```c++
//!    for (unsigned int i = 0, j = 0; i < ImageDimension; ++i)
//!      if (j != static_cast<unsigned int>(m_SliceDimension) && (j < (ImageDimension - 1)))
//!        { srad[j] = m_ContourThickness[i]; ++j; }
//!    ```
//!    The guard tests the *output* index `j` against `m_SliceDimension` where it
//!    plainly means the *input* index `i` — `j` is the write cursor into the
//!    `(D-1)`-dimensional radius. Consequences, with `srad` value-initialized to
//!    zeros (`.hxx:155`): for `m_SliceDimension == 0` the condition is false on
//!    the very first iteration and `j` never advances, so `srad` stays all-zero
//!    and `Ball(0)` erosion is the identity — the subtraction yields an empty
//!    image and *no contour at all*. For `m_SliceDimension == 1` in 3-D,
//!    `srad = (thickness[0], 0)` instead of `(thickness[0], thickness[2])`. Only
//!    `m_SliceDimension == ImageDimension - 1`, ITK's own default
//!    (`.hxx:43`), happens to give the right answer. Reproduced verbatim; pinned
//!    by `slice_contour_with_slice_dimension_zero_paints_nothing` and
//!    `slice_contour_borrows_the_wrong_thickness_axis`.
//! 2. **Through SimpleITK, `SLICE_CONTOUR` can never draw anything.**
//!    `LabelMapContourOverlayImageFilter.yaml`'s `SliceDimension` default is
//!    `0u`, not ITK's `ImageDimension - 1`, and the generated code always calls
//!    the setter. Combined with finding 1, every SimpleITK `SLICE_CONTOUR` run
//!    at the default `SliceDimension` produces an all-zero contour map and an
//!    output image identical to the plain grey feature image. The yaml's own
//!    `detaileddescriptionSet` says "defaults to image dimension - 1".
//!    [`LabelMapContourOverlaySettings::default`] follows the yaml, not ITK.
//! 3. **`DilationRadius`'s yaml default contradicts its yaml documentation.**
//!    `LabelMapContourOverlayImageFilter.yaml` sets `default:
//!    std::vector<unsigned int>(3, 1)` while its `detaileddescriptionSet` says
//!    "Set/Get the object dilation radius - 0 by default", which is ITK's own
//!    default (`itkLabelMapContourOverlayImageFilter.hxx:48-49`). SimpleITK
//!    therefore dilates every object by one pixel before contouring where ITK
//!    does not. [`LabelMapContourOverlaySettings::default`] follows the yaml.
//!
//! ## Divergences of this port
//!
//! * **Overlapping label objects are painted in ascending label order.**
//!   `ThreadedProcessLabelObject` is called from a multithreaded region
//!   (`itkLabelMapFilter.hxx:83-113`), so when two objects of the same label map
//!   share a pixel — representable, and what `LabelUniqueLabelMapFilter` exists
//!   to remove — ITK's winner is whichever thread wrote last, which is not
//!   specified. [`label_map_to_rgb`] and [`label_map_overlay`] paint in
//!   ascending label order, so the highest label wins. [`label_map_contour_overlay`]
//!   is unaffected: its temporary map went through `label_unique_label_map`, so
//!   no two objects overlap.

use sitk_core::{Image, LabelMap, LabelObject, PixelId};

use crate::error::{FilterError, Result};
use crate::label_map::{
    label_unique_label_map, object_offsets, require_label_pixel_id, require_same_size, strides,
};
use crate::label_to_rgb::{build_color_table, vector_image_from_f64};
use crate::morphology::{StructuringElement, binary_dilate, binary_erode};
use crate::{numeric_traits_max, quantize_to_pixel_type};

/// `LabelMapContourOverlayImageFilter`'s `Type` member
/// (`itkLabelMapContourOverlayImageFilter.h`'s `PLAIN`/`CONTOUR`/`SLICE_CONTOUR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ContourOverlayType {
    /// Paint the dilated object, no contour extraction.
    Plain,
    /// Paint `dilate(object) - erode(dilate(object))`.
    #[default]
    Contour,
    /// Like [`Self::Contour`], but eroded slice by slice along
    /// [`LabelMapContourOverlaySettings::slice_dimension`].
    SliceContour,
}

/// `LabelMapContourOverlayImageFilter`'s `Priority` member, forwarded to
/// `LabelUniqueLabelMapFilter::SetReverseOrdering`
/// (`itkLabelMapContourOverlayImageFilter.hxx:207`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ContourOverlayPriority {
    /// The greater label wins an overlap.
    #[default]
    HighLabelOnTop,
    /// The smaller label wins an overlap.
    LowLabelOnTop,
}

/// Defaults from `LabelMapContourOverlayImageFilter.yaml`, which differ from
/// ITK's own on `dilation_radius` and `slice_dimension` — see the module docs'
/// upstream findings 2 and 3.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelMapContourOverlaySettings {
    /// Blend weight of the colour against the feature value. `0.5`.
    pub opacity: f64,
    /// Per-axis `Ball` radius the object is dilated by first. `[1, 1, 1]`
    /// (ITK's own default is all-zero).
    pub dilation_radius: Vec<usize>,
    /// Per-axis `Ball` radius of the erosion subtracted from the dilation.
    /// `[1, 1, 1]`.
    pub contour_thickness: Vec<usize>,
    /// Slicing axis for [`ContourOverlayType::SliceContour`]. `0` (ITK's own
    /// default is `dimension - 1`).
    pub slice_dimension: usize,
    /// [`ContourOverlayType::Contour`].
    pub contour_type: ContourOverlayType,
    /// [`ContourOverlayPriority::HighLabelOnTop`].
    pub priority: ContourOverlayPriority,
    /// `SetLabelFunctorFromColormap`'s RGB triples; empty keeps the default
    /// 30-color palette.
    pub colormap: Vec<u8>,
}

impl Default for LabelMapContourOverlaySettings {
    fn default() -> Self {
        LabelMapContourOverlaySettings {
            opacity: 0.5,
            dilation_radius: vec![1, 1, 1],
            contour_thickness: vec![1, 1, 1],
            slice_dimension: 0,
            contour_type: ContourOverlayType::Contour,
            priority: ContourOverlayPriority::HighLabelOnTop,
            colormap: Vec::new(),
        }
    }
}

/// `itkLabelToRGBFunctor.h:101`'s `m_Colors[p % m_Colors.size()]`, with the
/// same `(p as i64) as u64` conversion [`crate::label_to_rgb`] documents.
fn color_index(label: i64, num_colors: usize) -> usize {
    ((label as u64) % num_colors as u64) as usize
}

/// `LabelMapToRGBImageFilter`: paint every label object with its palette colour
/// on a black background, as a 3-component `VectorUInt8` image.
///
/// The background is black rather than "a gray pixel with the same intensity",
/// for the reason [`crate::label_to_rgb`]'s upstream finding 1 gives: the
/// functor's `m_BackgroundColor` (`itkLabelToRGBFunctor.h:86-87`) is
/// zero-filled and SimpleITK exposes no setter. `BeforeThreadedGenerateData`
/// fills the buffer with `function(GetBackgroundValue())`
/// (`itkLabelMapToRGBImageFilter.hxx:36`), which is exactly that member.
pub fn label_map_to_rgb(map: &LabelMap, colormap: &[u8]) -> Result<Image> {
    require_label_pixel_id(map)?;

    let colors = build_color_table(colormap);
    let size = map.size();
    let st = strides(size);
    let total: usize = size.iter().product();

    let mut data = vec![0u8; total * 3];
    for object in map.label_objects() {
        let color = colors[color_index(object.label(), colors.len())];
        for off in object_offsets(object, &st) {
            data[off * 3..off * 3 + 3].copy_from_slice(&color);
        }
    }

    let mut out = Image::from_vec_vector(size, 3, data)?;
    map.apply_geometry_to(&mut out)?;
    Ok(out)
}

/// The palette, rescaled to `feature_id`'s `NumericTraits<ValueType>::max()` and
/// narrowed to it once, as `AddColor` does (`itkLabelToRGBFunctor.h:104-118`).
fn scaled_colors(colors: &[[u8; 3]], feature_id: PixelId) -> Vec<[f64; 3]> {
    let value_max = numeric_traits_max(feature_id);
    colors
        .iter()
        .map(|c| {
            std::array::from_fn(|i| {
                quantize_to_pixel_type(feature_id, c[i] as f64 / 255.0 * value_max)
            })
        })
        .collect()
}

/// `itkLabelOverlayFunctor.h:74-82`: `opaque[c] * opacity + p1 * (1 - opacity)`.
fn blend(opaque: [f64; 3], p1: f64, opacity: f64) -> [f64; 3] {
    let p1_blend = p1 * (1.0 - opacity);
    [
        opaque[0] * opacity + p1_blend,
        opaque[1] * opacity + p1_blend,
        opaque[2] * opacity + p1_blend,
    ]
}

/// Paints `objects` of `map` over a grey copy of `feature`, returning a
/// 3-component image whose component type is `feature`'s.
///
/// The grey pass is `DynamicThreadedGenerateData`
/// (`itkLabelMapOverlayImageFilter.hxx:106-116`), which calls the functor with
/// the map's *background* label at every pixel and so replicates the feature
/// value across all three channels.
fn overlay_onto(
    map: &LabelMap,
    objects: &LabelMap,
    feature: &Image,
    opacity: f64,
    colormap: &[u8],
) -> Result<Image> {
    let feature_id = feature.pixel_id();
    let base = feature.to_f64_vec()?;

    let colors = build_color_table(colormap);
    let scaled = scaled_colors(&colors, feature_id);

    let mut flat = Vec::with_capacity(base.len() * 3);
    for &p1 in &base {
        flat.extend_from_slice(&[p1, p1, p1]);
    }

    let st = strides(map.size());
    for object in objects.label_objects() {
        let opaque = scaled[color_index(object.label(), colors.len())];
        for off in object_offsets(object, &st) {
            let blended = blend(opaque, base[off], opacity);
            flat[off * 3..off * 3 + 3].copy_from_slice(&blended);
        }
    }

    let mut geom = Image::new(map.size(), PixelId::UInt8);
    map.apply_geometry_to(&mut geom)?;
    vector_image_from_f64(feature_id, map.size(), 3, &geom, &flat)
}

/// `LabelMapOverlayImageFilter`: blend the palette over `feature` at `opacity`
/// (default `0.5`). Background pixels keep `feature`'s own value on all three
/// channels (`itkLabelOverlayFunctor.h:63-71`).
pub fn label_map_overlay(
    map: &LabelMap,
    feature: &Image,
    opacity: f64,
    colormap: &[u8],
) -> Result<Image> {
    require_label_pixel_id(map)?;
    require_same_size(map, feature)?;
    overlay_onto(map, map, feature, opacity, colormap)
}

/// `LabelMapContourOverlayImageFilter`: blend the palette over `feature`, but
/// only along each object's contour. See the module docs for the pipeline and
/// for the three upstream findings this reproduces.
pub fn label_map_contour_overlay(
    map: &LabelMap,
    feature: &Image,
    settings: &LabelMapContourOverlaySettings,
) -> Result<Image> {
    require_label_pixel_id(map)?;
    require_same_size(map, feature)?;

    let dim = map.dimension();
    for r in [&settings.dilation_radius, &settings.contour_thickness] {
        if r.len() < dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: r.len(),
            });
        }
    }

    let contours = contour_label_map(map, settings)?;
    overlay_onto(
        map,
        &contours,
        feature,
        settings.opacity,
        &settings.colormap,
    )
}

/// The `ObjectByObjectLabelMapFilter` + `LabelUniqueLabelMapFilter` stack of
/// `BeforeThreadedGenerateData` (`itkLabelMapContourOverlayImageFilter.hxx:104-214`).
fn contour_label_map(
    map: &LabelMap,
    settings: &LabelMapContourOverlaySettings,
) -> Result<LabelMap> {
    let dim = map.dimension();
    let size = map.size();
    let dilation = &settings.dilation_radius[..dim];
    let thickness = &settings.contour_thickness[..dim];

    let mut objects = LabelMap::new(size, map.pixel_id(), map.background())?;

    for object in map.label_objects() {
        if object.is_empty() {
            continue;
        }
        // `AutoCropLabelMapFilter` with `CropBorder = m_DilationRadius + 1`,
        // clipped to the image (`itkObjectByObjectLabelMapFilter.hxx:111-116`,
        // `itkAutoCropLabelMapFilter.hxx:100-107`).
        let (lo, hi) = cropped_region(object, size, dilation);
        let sub_size: Vec<usize> = (0..dim).map(|d| (hi[d] - lo[d] + 1) as usize).collect();
        let sub_st = strides(&sub_size);

        let mut bin = vec![0u8; sub_size.iter().product()];
        for line in object.lines() {
            let idx = line.index();
            let base: usize = (1..dim)
                .map(|d| (idx[d] - lo[d]) as usize * sub_st[d])
                .sum();
            let start = base + (idx[0] - lo[0]) as usize;
            bin[start..start + line.length() as usize].fill(255);
        }

        let bin_img = Image::from_vec(&sub_size, bin)?;
        let dilated = binary_dilate(
            &bin_img,
            &StructuringElement::ball(dilation),
            255.0,
            0.0,
            false,
        )?;
        let result = match settings.contour_type {
            ContourOverlayType::Plain => dilated,
            ContourOverlayType::Contour => {
                let eroded = binary_erode(
                    &dilated,
                    &StructuringElement::ball(thickness),
                    255.0,
                    0.0,
                    true,
                )?;
                subtract(&dilated, &eroded)?
            }
            ContourOverlayType::SliceContour => {
                slice_contour(&dilated, settings.slice_dimension, thickness)?
            }
        };

        // `m_KeepLabels`: the single `{0, 255}` object keeps the input label.
        let mut contour = LabelObject::new(object.label(), dim)?;
        let mut idx = vec![0i64; dim];
        for (off, &v) in result.scalar_slice::<u8>()?.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let mut rest = off;
            for d in 0..dim {
                idx[d] = (rest % sub_size[d]) as i64 + lo[d];
                rest /= sub_size[d];
            }
            contour.add_index(&idx)?;
        }
        if !contour.is_empty() {
            objects.add_label_object(contour)?;
        }
    }

    label_unique_label_map(
        &objects,
        settings.priority == ContourOverlayPriority::LowLabelOnTop,
    )
}

/// The object's bounding box grown by `dilation + 1` per axis and clipped to
/// `[0, size - 1]`.
fn cropped_region(
    object: &LabelObject,
    size: &[usize],
    dilation: &[usize],
) -> (Vec<i64>, Vec<i64>) {
    let dim = size.len();
    let mut lo = vec![i64::MAX; dim];
    let mut hi = vec![i64::MIN; dim];
    for line in object.lines() {
        let idx = line.index();
        for d in 0..dim {
            lo[d] = lo[d].min(idx[d]);
            hi[d] = hi[d].max(idx[d]);
        }
        hi[0] = hi[0].max(idx[0] + line.length() - 1);
    }
    for d in 0..dim {
        let border = dilation[d] as i64 + 1;
        lo[d] = (lo[d] - border).max(0);
        hi[d] = (hi[d] + border).min(size[d] as i64 - 1);
    }
    (lo, hi)
}

/// `SubtractImageFilter` over two `{0, 255}` images of equal size.
fn subtract(a: &Image, b: &Image) -> Result<Image> {
    let out: Vec<u8> = a
        .scalar_slice::<u8>()?
        .iter()
        .zip(b.scalar_slice::<u8>()?)
        .map(|(&x, &y)| x - y)
        .collect();
    Ok(Image::from_vec(a.size(), out)?)
}

/// `itkLabelMapContourOverlayImageFilter.hxx:154-163`, reproduced with its
/// `j != m_SliceDimension` guard — see the module docs' upstream finding 1.
fn slice_erode_radius(thickness: &[usize], slice_dimension: usize, dim: usize) -> Vec<usize> {
    let mut srad = vec![0usize; dim - 1];
    let mut j = 0usize;
    for &t in thickness.iter().take(dim) {
        if j != slice_dimension && j < dim - 1 {
            srad[j] = t;
            j += 1;
        }
    }
    srad
}

/// `SliceBySliceImageFilter` with `scast - serode(scast)` as its per-slice
/// pipeline (`itkLabelMapContourOverlayImageFilter.hxx:142-171`).
fn slice_contour(dilated: &Image, slice_dimension: usize, thickness: &[usize]) -> Result<Image> {
    let size = dilated.size().to_vec();
    let dim = size.len();
    // `itkSliceBySliceImageFilter.hxx:59-63` throws on an out-of-range axis.
    if slice_dimension >= dim {
        return Err(FilterError::InvalidDirection {
            direction: slice_dimension,
            dimension: dim,
        });
    }

    let srad = slice_erode_radius(thickness, slice_dimension, dim);
    let kernel = StructuringElement::ball(&srad);
    let slice_axes: Vec<usize> = (0..dim).filter(|&d| d != slice_dimension).collect();
    let slice_size: Vec<usize> = slice_axes.iter().map(|&d| size[d]).collect();
    let st = strides(&size);

    let src = dilated.scalar_slice::<u8>()?;
    let mut out = vec![0u8; src.len()];
    let mut plane = vec![0u8; slice_size.iter().product()];

    for s in 0..size[slice_dimension] {
        let mut offsets = Vec::with_capacity(plane.len());
        for (p, slot) in plane.iter_mut().enumerate() {
            let mut rest = p;
            let mut off = s * st[slice_dimension];
            for (k, &d) in slice_axes.iter().enumerate() {
                off += (rest % slice_size[k]) * st[d];
                rest /= slice_size[k];
            }
            *slot = src[off];
            offsets.push(off);
        }
        let plane_img = Image::from_vec(&slice_size, plane.clone())?;
        let eroded = binary_erode(&plane_img, &kernel, 255.0, 0.0, true)?;
        let diff = subtract(&plane_img, &eroded)?;
        for (&off, &v) in offsets.iter().zip(diff.scalar_slice::<u8>()?) {
            out[off] = v;
        }
    }

    Ok(Image::from_vec(&size, out)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label_to_rgb::DEFAULT_LABEL_COLORS;

    type Line2 = ([i64; 2], i64);

    /// A `size`-shaped `UInt8` label map whose objects are given as
    /// `(label, &[(start_index, length)])` runs along axis 0.
    fn map_of(size: &[usize], objects: &[(i64, &[Line2])]) -> LabelMap {
        let mut map = LabelMap::new(size, PixelId::UInt8, 0).unwrap();
        for (label, lines) in objects {
            for (index, length) in *lines {
                map.set_line(index, *length, *label).unwrap();
            }
        }
        map
    }

    fn rgb_at(img: &Image, x: usize, y: usize) -> [u8; 3] {
        let w = img.size()[0];
        let s = img.component_slice::<u8>().unwrap();
        let off = (y * w + x) * 3;
        [s[off], s[off + 1], s[off + 2]]
    }

    fn f64_at(img: &Image, x: usize, y: usize) -> [f64; 3] {
        let w = img.size()[0];
        let s = img.components_to_f64_vec();
        let off = (y * w + x) * 3;
        [s[off], s[off + 1], s[off + 2]]
    }

    #[test]
    fn label_map_to_rgb_paints_the_palette_and_a_black_background() {
        let map = map_of(&[3, 1], &[(1, &[([0, 0], 1)]), (2, &[([2, 0], 1)])]);
        let out = label_map_to_rgb(&map, &[]).unwrap();
        assert_eq!(out.number_of_components_per_pixel(), 3);
        assert_eq!(rgb_at(&out, 0, 0), DEFAULT_LABEL_COLORS[1]);
        assert_eq!(rgb_at(&out, 1, 0), [0, 0, 0]);
        assert_eq!(rgb_at(&out, 2, 0), DEFAULT_LABEL_COLORS[2]);
    }

    #[test]
    fn label_map_to_rgb_wraps_the_palette_modulo_its_length() {
        let map = map_of(&[1, 1], &[(31, &[([0, 0], 1)])]);
        let out = label_map_to_rgb(&map, &[]).unwrap();
        assert_eq!(rgb_at(&out, 0, 0), DEFAULT_LABEL_COLORS[1]);
    }

    #[test]
    fn a_custom_colormap_replaces_the_palette_entirely() {
        let map = map_of(&[2, 1], &[(1, &[([0, 0], 1)]), (2, &[([1, 0], 1)])]);
        let out = label_map_to_rgb(&map, &[9, 8, 7, 6, 5, 4]).unwrap();
        // Two colors now, so label 2 wraps to index 0.
        assert_eq!(rgb_at(&out, 0, 0), [6, 5, 4]);
        assert_eq!(rgb_at(&out, 1, 0), [9, 8, 7]);
    }

    #[test]
    fn an_incomplete_trailing_triple_is_dropped() {
        let map = map_of(&[1, 1], &[(1, &[([0, 0], 1)])]);
        let out = label_map_to_rgb(&map, &[9, 8, 7, 6]).unwrap();
        assert_eq!(rgb_at(&out, 0, 0), [9, 8, 7]);
    }

    #[test]
    fn a_higher_label_wins_an_overlap_in_label_map_to_rgb() {
        let mut map = LabelMap::new(&[1, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 0], 1, 1).unwrap();
        map.set_line(&[0, 0], 1, 2).unwrap();
        let out = label_map_to_rgb(&map, &[]).unwrap();
        assert_eq!(rgb_at(&out, 0, 0), DEFAULT_LABEL_COLORS[2]);
    }

    #[test]
    fn a_signed_label_map_is_rejected() {
        let map = LabelMap::new(&[2, 2], PixelId::Int16, 0).unwrap();
        assert!(matches!(
            label_map_to_rgb(&map, &[]),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int16
            ))
        ));
    }

    /// The three entry points reject a complex feature image: the label map's
    /// own pixel type cannot be complex (`LabelMap::new` refuses it), and the
    /// feature image goes through `Image::to_f64_vec`, whose `require_scalar`
    /// whitelist excludes the two complex ids.
    #[test]
    fn a_complex_feature_image_is_rejected_by_both_overlays() {
        let map = map_of(&[2, 1], &[(1, &[([0, 0], 1)])]);
        let feature = Image::new(&[2, 1], PixelId::ComplexFloat32);
        assert_eq!(
            label_map_overlay(&map, &feature, 0.5, &[]),
            Err(FilterError::Core(
                sitk_core::Error::RequiresScalarPixelType(PixelId::ComplexFloat32)
            ))
        );
        let settings = LabelMapContourOverlaySettings {
            dilation_radius: vec![0, 0],
            ..Default::default()
        };
        assert_eq!(
            label_map_contour_overlay(&map, &feature, &settings),
            Err(FilterError::Core(
                sitk_core::Error::RequiresScalarPixelType(PixelId::ComplexFloat32)
            ))
        );
    }

    #[test]
    fn a_complex_label_map_cannot_be_constructed() {
        assert!(matches!(
            LabelMap::new(&[2, 1], PixelId::ComplexFloat32, 0),
            Err(sitk_core::Error::RequiresIntegerPixelType(
                PixelId::ComplexFloat32
            ))
        ));
    }

    #[test]
    fn label_map_overlay_passes_the_background_through_as_grey() {
        let map = map_of(&[2, 1], &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[2, 1], vec![10u8, 200u8]).unwrap();
        let out = label_map_overlay(&map, &feature, 0.5, &[]).unwrap();
        assert_eq!(rgb_at(&out, 1, 0), [200, 200, 200]);
    }

    #[test]
    fn label_map_overlay_blends_the_color_at_the_given_opacity() {
        let map = map_of(&[1, 1], &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[1, 1], vec![100u8]).unwrap();
        let out = label_map_overlay(&map, &feature, 0.25, &[]).unwrap();
        // Palette color 1 is (0, 205, 0); base type is uint8 so `max() == 255`
        // and the scaling is the identity.
        let c = DEFAULT_LABEL_COLORS[1];
        let expect: Vec<u8> = c
            .iter()
            .map(|&v| (v as f64 * 0.25 + 100.0 * 0.75) as u8)
            .collect();
        assert_eq!(rgb_at(&out, 0, 0), [expect[0], expect[1], expect[2]]);
    }

    #[test]
    fn label_map_overlay_rescales_the_palette_to_the_feature_types_max() {
        let map = map_of(&[1, 1], &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        let out = label_map_overlay(&map, &feature, 1.0, &[]).unwrap();
        let got = f64_at(&out, 0, 0);
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], 205.0 / 255.0 * f64::MAX);
        assert_eq!(got[2], 0.0);
    }

    #[test]
    fn label_map_overlay_output_takes_the_feature_component_type() {
        let map = map_of(&[1, 1], &[(1, &[([0, 0], 1)])]);
        let feature = Image::from_vec(&[1, 1], vec![7i16]).unwrap();
        let out = label_map_overlay(&map, &feature, 0.5, &[]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorInt16);
        assert_eq!(out.number_of_components_per_pixel(), 3);
    }

    #[test]
    fn label_map_overlay_rejects_a_size_mismatch() {
        let map = map_of(&[2, 2], &[]);
        let feature = Image::new(&[3, 2], PixelId::UInt8);
        assert!(matches!(
            label_map_overlay(&map, &feature, 0.5, &[]),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    /// The 3x3 solid square, dilated by 0 and eroded by 1, leaves the 8-pixel
    /// ring; the centre keeps its grey feature value.
    #[test]
    fn contour_of_a_square_is_its_ring() {
        let map = map_of(&[5, 5], &[(1, &[([1, 1], 3), ([1, 2], 3), ([1, 3], 3)])]);
        let feature = Image::new(&[5, 5], PixelId::UInt8);
        let settings = LabelMapContourOverlaySettings {
            dilation_radius: vec![0, 0],
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &settings).unwrap();
        let c = DEFAULT_LABEL_COLORS[1];
        assert_eq!(rgb_at(&out, 1, 1), c);
        assert_eq!(rgb_at(&out, 2, 1), c);
        assert_eq!(rgb_at(&out, 3, 3), c);
        assert_eq!(rgb_at(&out, 2, 2), [0, 0, 0]);
        assert_eq!(rgb_at(&out, 0, 0), [0, 0, 0]);
    }

    #[test]
    fn plain_paints_the_whole_dilated_object() {
        let map = map_of(&[5, 5], &[(1, &[([2, 2], 1)])]);
        let feature = Image::new(&[5, 5], PixelId::UInt8);
        let settings = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::Plain,
            dilation_radius: vec![1, 1],
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &settings).unwrap();
        let c = DEFAULT_LABEL_COLORS[1];
        // `Ball([1, 1])` is the 3x3 box: (x/1.5)^2 + (y/1.5)^2 <= 1 holds at
        // the corners too (0.888).
        assert_eq!(rgb_at(&out, 2, 2), c);
        assert_eq!(rgb_at(&out, 1, 1), c);
        assert_eq!(rgb_at(&out, 3, 3), c);
        assert_eq!(rgb_at(&out, 0, 2), [0, 0, 0]);
    }

    #[test]
    fn the_dilation_radius_grows_the_contour_outward() {
        let map = map_of(&[7, 7], &[(1, &[([3, 3], 1)])]);
        let feature = Image::new(&[7, 7], PixelId::UInt8);
        let plain0 = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::Plain,
            dilation_radius: vec![0, 0],
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &plain0).unwrap();
        assert_eq!(rgb_at(&out, 3, 3), DEFAULT_LABEL_COLORS[1]);
        assert_eq!(rgb_at(&out, 2, 3), [0, 0, 0]);

        let plain1 = LabelMapContourOverlaySettings {
            dilation_radius: vec![1, 1],
            ..plain0
        };
        let out = label_map_contour_overlay(&map, &feature, &plain1).unwrap();
        assert_eq!(rgb_at(&out, 2, 3), DEFAULT_LABEL_COLORS[1]);
    }

    #[test]
    fn high_label_on_top_wins_a_contour_overlap() {
        // Two 1-pixel objects one apart; dilated by 1 they share the pixel
        // between them.
        let map = map_of(&[5, 1], &[(1, &[([1, 0], 1)]), (2, &[([3, 0], 1)])]);
        let feature = Image::new(&[5, 1], PixelId::UInt8);
        let high = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::Plain,
            dilation_radius: vec![1, 1],
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &high).unwrap();
        assert_eq!(rgb_at(&out, 2, 0), DEFAULT_LABEL_COLORS[2]);

        let low = LabelMapContourOverlaySettings {
            priority: ContourOverlayPriority::LowLabelOnTop,
            ..high
        };
        let out = label_map_contour_overlay(&map, &feature, &low).unwrap();
        assert_eq!(rgb_at(&out, 2, 0), DEFAULT_LABEL_COLORS[1]);
    }

    #[test]
    fn contour_overlay_blends_against_the_feature_image() {
        let map = map_of(&[3, 1], &[(1, &[([1, 0], 1)])]);
        let feature = Image::from_vec(&[3, 1], vec![100u8, 100, 100]).unwrap();
        let settings = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::Plain,
            dilation_radius: vec![0, 0],
            opacity: 0.5,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &settings).unwrap();
        let c = DEFAULT_LABEL_COLORS[1];
        let expect: Vec<u8> = c
            .iter()
            .map(|&v| (v as f64 * 0.5 + 100.0 * 0.5) as u8)
            .collect();
        assert_eq!(rgb_at(&out, 1, 0), [expect[0], expect[1], expect[2]]);
        assert_eq!(rgb_at(&out, 0, 0), [100, 100, 100]);
    }

    /// Upstream finding 1 / 2: `j != m_SliceDimension` on the first iteration
    /// with `m_SliceDimension == 0` blocks the loop forever, leaving a zero
    /// radius, an identity erosion and an empty difference.
    #[test]
    fn slice_contour_with_slice_dimension_zero_paints_nothing() {
        assert_eq!(slice_erode_radius(&[1, 1], 0, 2), vec![0]);
        assert_eq!(slice_erode_radius(&[1, 1, 1], 0, 3), vec![0, 0]);

        let map = map_of(&[5, 5], &[(1, &[([1, 1], 3), ([1, 2], 3), ([1, 3], 3)])]);
        let feature = Image::from_vec(&[5, 5], vec![42u8; 25]).unwrap();
        let settings = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::SliceContour,
            dilation_radius: vec![0, 0],
            slice_dimension: 0,
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &settings).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(rgb_at(&out, x, y), [42, 42, 42], "at ({x}, {y})");
            }
        }
    }

    /// Upstream finding 1: in 3-D the second slice radius should come from
    /// `thickness[2]`; the `j`-guard makes it `0`.
    #[test]
    fn slice_contour_borrows_the_wrong_thickness_axis() {
        assert_eq!(slice_erode_radius(&[2, 3, 4], 1, 3), vec![2, 0]);
        // Only ITK's own default `SliceDimension == D - 1` is right.
        assert_eq!(slice_erode_radius(&[2, 3, 4], 2, 3), vec![2, 3]);
    }

    #[test]
    fn slice_contour_at_the_last_axis_erodes_within_each_row() {
        // 2-D, slice_dimension = 1: each row is eroded on its own, so a
        // 3-pixel-wide, 3-row square keeps its left and right columns only.
        let map = map_of(&[5, 5], &[(1, &[([1, 1], 3), ([1, 2], 3), ([1, 3], 3)])]);
        let feature = Image::new(&[5, 5], PixelId::UInt8);
        let settings = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::SliceContour,
            dilation_radius: vec![0, 0],
            contour_thickness: vec![1, 1],
            slice_dimension: 1,
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &settings).unwrap();
        let c = DEFAULT_LABEL_COLORS[1];
        for y in 1..4 {
            assert_eq!(rgb_at(&out, 1, y), c);
            assert_eq!(rgb_at(&out, 2, y), [0, 0, 0]);
            assert_eq!(rgb_at(&out, 3, y), c);
        }
    }

    #[test]
    fn slice_contour_rejects_an_out_of_range_slice_dimension() {
        let map = map_of(&[3, 3], &[(1, &[([1, 1], 1)])]);
        let feature = Image::new(&[3, 3], PixelId::UInt8);
        let settings = LabelMapContourOverlaySettings {
            contour_type: ContourOverlayType::SliceContour,
            slice_dimension: 2,
            dilation_radius: vec![0, 0],
            ..Default::default()
        };
        assert!(matches!(
            label_map_contour_overlay(&map, &feature, &settings),
            Err(FilterError::InvalidDirection {
                direction: 2,
                dimension: 2
            })
        ));
    }

    #[test]
    fn a_too_short_radius_is_rejected() {
        let map = map_of(&[3, 3, 3], &[]);
        let feature = Image::new(&[3, 3, 3], PixelId::UInt8);
        let settings = LabelMapContourOverlaySettings {
            dilation_radius: vec![1, 1],
            ..Default::default()
        };
        assert!(matches!(
            label_map_contour_overlay(&map, &feature, &settings),
            Err(FilterError::DimensionLength {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn erosion_treats_the_outside_of_the_cropped_region_as_foreground() {
        // A 2x2 block in a 3x3 image. The crop border cannot leave the image,
        // and `BinaryErodeImageFilter` sets `m_BoundaryToForeground = true`
        // (`itkBinaryErodeImageFilter.hxx:36`), so the corner (0, 0) — whose
        // only background neighbours lie outside the image — survives the
        // erosion and is *subtracted out* of the contour. The other three
        // pixels of the block have an in-image background neighbour and stay.
        let map = map_of(&[3, 3], &[(1, &[([0, 0], 2), ([0, 1], 2)])]);
        let feature = Image::new(&[3, 3], PixelId::UInt8);
        let settings = LabelMapContourOverlaySettings {
            dilation_radius: vec![0, 0],
            opacity: 1.0,
            ..Default::default()
        };
        let out = label_map_contour_overlay(&map, &feature, &settings).unwrap();
        let c = DEFAULT_LABEL_COLORS[1];
        assert_eq!(rgb_at(&out, 0, 0), [0, 0, 0]);
        assert_eq!(rgb_at(&out, 1, 0), c);
        assert_eq!(rgb_at(&out, 0, 1), c);
        assert_eq!(rgb_at(&out, 1, 1), c);
        assert_eq!(rgb_at(&out, 2, 2), [0, 0, 0]);
    }
}
