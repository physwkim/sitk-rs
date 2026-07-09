//! Morphological opening/closing by reconstruction, connected opening/
//! closing, and binary reconstruction by dilation/erosion.
//!
//! Ports of (ITK `Modules/Filtering/{MathematicalMorphology,
//! BinaryMathematicalMorphology,LabelMap}/include/`):
//!
//! - [`opening_by_reconstruction`] / [`closing_by_reconstruction`] —
//!   `itkOpeningByReconstructionImageFilter.h` / `.hxx`,
//!   `itkClosingByReconstructionImageFilter.h` / `.hxx`.
//! - [`grayscale_connected_opening`] / [`grayscale_connected_closing`] —
//!   `itkGrayscaleConnectedOpeningImageFilter.h` / `.hxx`,
//!   `itkGrayscaleConnectedClosingImageFilter.h` / `.hxx`.
//! - [`binary_reconstruction_by_dilation`] / [`binary_reconstruction_by_erosion`] —
//!   `itkBinaryReconstructionByDilationImageFilter.h` / `.hxx`,
//!   `itkBinaryReconstructionByErosionImageFilter.h` / `.hxx`, built from the
//!   `LabelMap` machinery (`itkBinaryImageToLabelMapFilter`,
//!   `itkBinaryReconstructionLabelMapFilter`, `itkAttributeOpeningLabelMapFilter`,
//!   `itkLabelMapToBinaryImageFilter`, `itkLabelMapMaskImageFilter`,
//!   `itkBinaryNotImageFilter`).
//! - [`binary_opening_by_reconstruction`] / [`binary_closing_by_reconstruction`] —
//!   `itkBinaryOpeningByReconstructionImageFilter.h` / `.hxx`,
//!   `itkBinaryClosingByReconstructionImageFilter.h` / `.hxx`.
//!
//! ## Reconstruction engine reuse
//!
//! ITK implements grayscale reconstruction two different ways depending on
//! the caller. [`crate::reconstruction`]'s `reconstruction_by_dilation` /
//! `reconstruction_by_erosion` (imported below) already port
//! `itkReconstructionImageFilter.hxx`'s raster / anti-raster / FIFO
//! three-pass fast hybrid algorithm, and [`crate::geodesic_morphology`]'s own
//! module docs prove that algorithm is behaviorally identical (for
//! `fully_connected = false`; see that module for the `fully_connected =
//! true` caveat) to running elementary-kernel geodesic dilation/erosion to
//! convergence. Every grayscale filter below (`OpeningByReconstruction`,
//! `ClosingByReconstruction`, `GrayscaleConnectedOpening`,
//! `GrayscaleConnectedClosing`) is a thin `.hxx` wrapper that builds a marker
//! image and hands it straight to `ReconstructionByDilationImageFilter` /
//! `ReconstructionByErosionImageFilter` with no logic of its own beyond
//! that — so this port builds each one directly on `reconstruction_by_dilation`
//! / `reconstruction_by_erosion` rather than re-deriving the three-pass
//! algorithm, relying on the same equivalence [`crate::geodesic_morphology`]
//! already states and tests.
//!
//! ## Opening/closing by reconstruction
//!
//! `OpeningByReconstructionImageFilter::GenerateData`: erode `image` by
//! `kernel` ([`crate::morphology::grayscale_erode`]), then reconstruct that
//! erosion by dilation under `image` as the mask
//! (`reconstruction_by_dilation`) — restoring every eroded object's
//! *original* shape in full, rather than the rounded-off shape a plain
//! `dilate(erode(f))` opening would give (see [`opening_by_reconstruction`]'s
//! own test for a worked example).
//!
//! `PreserveIntensities` (default `false`) changes what happens to pixels the
//! reconstruction didn't need to raise. With it off, the reconstruction above
//! alone is the output — `.hxx` calls this `dilate` (a
//! `ReconstructionByDilationImageFilter` instance), and even a wide, tall
//! peak can get leveled down to its eroded height wherever the flood raises
//! it no further. With it on, a second marker (`.hxx`'s `tempImage`) holds
//! `image`'s *original* value at every pixel where `erode(image)` already
//! equals that first reconstruction's own output, and
//! `NumericTraits<T>::NonpositiveMin()` everywhere else, then reconstructs by
//! dilation under `image` a second time — recovering the true peak height at
//! every pixel the first pass didn't need to move (see
//! [`opening_by_reconstruction`]'s test for a hand-derived example of exactly
//! this effect).
//!
//! `ClosingByReconstructionImageFilter` is the exact dual: dilate, then
//! reconstruct by erosion under the input; `PreserveIntensities`'s second
//! marker compares against that first reconstruction's own output too, using
//! `NumericTraits<T>::max()` as its "affected" fill instead.
//!
//! ## Grayscale connected opening/closing
//!
//! `GrayscaleConnectedOpeningImageFilter::GenerateData`: read `seedValue =
//! inputImage->GetPixel(m_Seed)`. If `seedValue` equals the image's global
//! minimum, `.hxx` fills the *entire* output with that minimum (a degenerate
//! case: a marker of all-minimum reconstructs to all-minimum, since
//! reconstruction-by-dilation's floor is the marker itself) — this is the
//! "seed outside any object" edge case: a seed sitting on the background
//! never touches anything, so nothing survives. Otherwise `.hxx` builds a
//! marker holding the image's global minimum everywhere except `seedValue` at
//! `m_Seed`, and reconstructs by dilation under the input — keeping only the
//! object touching the seed, at its original height, and suppressing every
//! other object to the background minimum. `GrayscaleConnectedClosingImageFilter`
//! is the dual (global maximum, reconstruct by erosion).
//!
//! `.hxx`'s `GetPixel(m_Seed)` performs no bounds check at all — an
//! out-of-range `Seed` is undefined behavior in C++ (it may alias an
//! arbitrary offset via pointer arithmetic on the underlying buffer). This
//! port checks instead ([`FilterError::InvalidSeedIndex`] /
//! [`FilterError::DimensionLength`]), since the same out-of-range multi-index
//! could otherwise alias a different in-bounds flat offset over this crate's
//! linear pixel buffer rather than simply crash.
//!
//! ## Binary reconstruction by dilation/erosion
//!
//! Both are `LabelMap` pipelines, not instantiations of the grayscale
//! `TCompare`-generalized reconstruction engine above; they operate on
//! *connected components*, not pixel-by-pixel flooding. Traced end to end
//! from `itkBinaryReconstructionByDilationImageFilter.hxx` /
//! `itkBinaryReconstructionByErosionImageFilter.hxx`:
//!
//! - **Dilation**: label the connected components of `mask_image`'s
//!   `foreground_value` pixels ([`crate::label::connected_component`], which
//!   this port reuses directly per the task's instruction —
//!   `itkBinaryImageToLabelMapFilter` is the same scanline labeling
//!   algorithm, just parametrized on an arbitrary foreground value rather
//!   than fixed to nonzero). `BinaryReconstructionLabelMapFilter::ThreadedProcessLabelObject`
//!   marks each label object "kept" iff `marker_image` equals
//!   `foreground_value` at *any* of that object's pixels.
//!   `AttributeOpeningLabelMapFilter` drops every object not marked kept.
//!   `LabelMapToBinaryImageFilter` renders the result: kept-object pixels
//!   become `foreground_value`; every other pixel (dropped objects and
//!   originally-non-foreground pixels alike) keeps `mask_image`'s own value
//!   verbatim (`BackgroundImage = mask_image`). Net effect: **kept
//!   mask-foreground components → `foreground_value`; dropped
//!   mask-foreground components → `background_value`; every other pixel →
//!   mask's own value unchanged** (label-preserving pass-through).
//! - **Erosion**: works on the *complement*. `notMask =
//!   BinaryNotImageFilter(mask_image)`, `notMarker =
//!   BinaryNotImageFilter(marker_image)` (both fixed to `ForegroundValue =
//!   foreground_value`; see `itkBinaryNotImageFilter.h`'s functor: `!(a ==
//!   foreground) ? foreground : background` — every non-foreground input
//!   value collapses to the *single* sentinel `foreground_value`, not merely
//!   toggling two values). The same labeler/opener pipeline as dilation then
//!   runs on `notMask`/`notMarker`, and the final `LabelMapMaskImageFilter`
//!   (`Negated = true`, `Label = background_value`, its own `BackgroundValue`
//!   parameter set to `foreground_value`, `FeatureImage = mask_image`)
//!   renders: every pixel starts at `foreground_value`
//!   (`DynamicThreadedGenerateData`'s `(BackgroundValue == Label) ^ Negated`
//!   is `true ^ true = false`, so it fills with its own `m_BackgroundValue`
//!   param, i.e. `foreground_value`), then every *kept* not-mask-foreground
//!   object's pixels are overwritten with `mask_image`'s own verbatim value
//!   (`ThreadedProcessLabelObject`'s `Negated` branch). Net effect: **every
//!   `mask_image`-foreground pixel → `foreground_value` always; a
//!   `mask_image`-background component whose marker is non-`foreground_value`
//!   anywhere → mask's own value unchanged; every other (marker-all-`foreground_value`)
//!   background component → `foreground_value`**. `background_value` never
//!   appears in this derivation's output formula at all — it is inert for
//!   the erosion variant except in the degenerate case `background_value ==
//!   foreground_value`, where `BinaryNotImageFilter`'s single-sentinel
//!   collapse (above) changes which pixels compare equal to
//!   `foreground_value` going into the labeler. This port builds the
//!   not-mask/not-marker transform literally the way the functor does
//!   (rather than a `background`/`foreground`-swap shortcut) so that quirk
//!   reproduces itself rather than being silently fixed.
//!
//! ## Equivalence to geodesic binary reconstruction
//!
//! For a *strictly binary* marker/mask pair (every pixel exactly
//! `foreground_value` or `background_value`, no other labels, and — for
//! dilation — `marker <= mask` pointwise), [`binary_reconstruction_by_dilation`]
//! agrees pixel-for-pixel with running `reconstruction_by_dilation` directly
//! on that same binary pair: a two-valued image only has "flood reached this
//! pixel" or "not yet", so growing the marker's foreground within the mask
//! one geodesic step at a time saturates every touched connected component to
//! fully foreground exactly when a connected-components pass would mark it
//! kept. [`binary_reconstruction_by_erosion`] is the same equivalence run on
//! the complement pair. This crate's binary entry points don't reduce to that
//! call because they additionally support multi-valued (label-preserving)
//! masks and skip the `marker <= mask` precondition entirely (see the
//! derivation above) — properties the grayscale engine doesn't have — but the
//! connected-components algorithm they're built on is the one the label-map
//! `.hxx` files actually specify, so this port implements it at that level
//! per the task's instruction rather than calling through the grayscale
//! engine.
//!
//! ## Binary opening/closing by reconstruction
//!
//! `BinaryOpeningByReconstructionImageFilter::GenerateData`: binary-erode
//! ([`crate::morphology::binary_erode`]) then
//! [`binary_reconstruction_by_dilation`], both with the caller's own
//! `foreground_value`/`background_value` passed straight through unchanged,
//! and no `SafeBorder` padding. `BinaryClosingByReconstructionImageFilter` is
//! the dual — binary-dilate then [`binary_reconstruction_by_erosion`],
//! likewise unpadded — except its internal erode/dilate `background_value`
//! is chosen exactly the way
//! [`crate::morphology::binary_morphological_closing`] chooses its own (`0`
//! unless `foreground_value == 0`, in which case
//! `NumericTraits<T>::max()`): the `.hxx` has no user-facing
//! `BackgroundValue` member at all.

use crate::error::{FilterError, Result};
use crate::label::connected_component;
use crate::morphology::{
    StructuringElement, binary_dilate, binary_erode, bounds_for, grayscale_dilate, grayscale_erode,
};
use crate::reconstruction::{reconstruction_by_dilation, reconstruction_by_erosion};
use crate::{image_from_f64, require_same_shape};
use sitk_core::Image;

// ---- shared helpers --------------------------------------------------

/// First-index-fastest strides for a size vector (see [`seed_flat_index`]).
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// Validates `seed` against `size` and returns its flat (dimension-0-fastest)
/// offset. `GrayscaleConnectedOpeningImageFilter`/`...ClosingImageFilter`'s
/// `.hxx` never checks `m_Seed` before dereferencing it (see the module
/// docs); this port returns [`FilterError::DimensionLength`] for a
/// wrong-length seed and [`FilterError::InvalidSeedIndex`] for one that's
/// the right length but out of range on some axis.
fn seed_flat_index(seed: &[usize], size: &[usize]) -> Result<usize> {
    if seed.len() != size.len() {
        return Err(FilterError::DimensionLength {
            expected: size.len(),
            got: seed.len(),
        });
    }
    if seed.iter().zip(size).any(|(&s, &n)| s >= n) {
        return Err(FilterError::InvalidSeedIndex {
            seed: seed.to_vec(),
            size: size.to_vec(),
        });
    }
    Ok(seed
        .iter()
        .zip(&strides(size))
        .map(|(&s, &st)| s * st)
        .sum())
}

/// The `tempImage` `OpeningByReconstructionImageFilter::GenerateData` /
/// `ClosingByReconstructionImageFilter::GenerateData` build for
/// `PreserveIntensities`: `source`'s own value wherever `a == b`, `else_value`
/// everywhere else.
fn select_where_equal(a: &Image, b: &Image, source: &Image, else_value: f64) -> Result<Image> {
    let a_vals = a.to_f64_vec()?;
    let b_vals = b.to_f64_vec()?;
    let source_vals = source.to_f64_vec()?;
    let out: Vec<f64> = a_vals
        .iter()
        .zip(&b_vals)
        .zip(&source_vals)
        .map(|((&x, &y), &s)| if x == y { s } else { else_value })
        .collect();
    image_from_f64(source.pixel_id(), source.size(), source, &out)
}

// ---- opening / closing by reconstruction ----------------------------

/// `OpeningByReconstructionImageFilter` (see module docs): erode `image` by
/// `kernel`, then reconstruct that erosion by dilation under `image`,
/// restoring every surviving object's original shape rather than a
/// dilate-of-erosion rounding.
pub fn opening_by_reconstruction(
    image: &Image,
    kernel: &StructuringElement,
    fully_connected: bool,
    preserve_intensities: bool,
) -> Result<Image> {
    let eroded = grayscale_erode(image, kernel)?;
    let opened = reconstruction_by_dilation(&eroded, image, fully_connected)?;
    if !preserve_intensities {
        return Ok(opened);
    }
    let (_, nonpositive_min) = bounds_for(image.pixel_id());
    let marker = select_where_equal(&eroded, &opened, image, nonpositive_min)?;
    reconstruction_by_dilation(&marker, image, fully_connected)
}

/// `ClosingByReconstructionImageFilter` (see module docs): the dual of
/// [`opening_by_reconstruction`] — dilate `image` by `kernel`, then
/// reconstruct that dilation by erosion under `image`.
pub fn closing_by_reconstruction(
    image: &Image,
    kernel: &StructuringElement,
    fully_connected: bool,
    preserve_intensities: bool,
) -> Result<Image> {
    let dilated = grayscale_dilate(image, kernel)?;
    let closed = reconstruction_by_erosion(&dilated, image, fully_connected)?;
    if !preserve_intensities {
        return Ok(closed);
    }
    let (max_value, _) = bounds_for(image.pixel_id());
    let marker = select_where_equal(&dilated, &closed, image, max_value)?;
    reconstruction_by_erosion(&marker, image, fully_connected)
}

// ---- grayscale connected opening / closing ---------------------------

/// `GrayscaleConnectedOpeningImageFilter` (see module docs): keep only the
/// object touching `seed`, at its original height, and suppress every other
/// object to `image`'s global minimum.
pub fn grayscale_connected_opening(
    image: &Image,
    seed: &[usize],
    fully_connected: bool,
) -> Result<Image> {
    let flat = seed_flat_index(seed, image.size())?;
    let (min_value, _) = crate::minimum_maximum(image)?;
    let vals = image.to_f64_vec()?;
    let seed_value = vals[flat];
    if seed_value == min_value {
        let out = vec![min_value; vals.len()];
        return image_from_f64(image.pixel_id(), image.size(), image, &out);
    }
    let mut marker_vals = vec![min_value; vals.len()];
    marker_vals[flat] = seed_value;
    let marker_image = image_from_f64(image.pixel_id(), image.size(), image, &marker_vals)?;
    reconstruction_by_dilation(&marker_image, image, fully_connected)
}

/// `GrayscaleConnectedClosingImageFilter` (see module docs): the dual of
/// [`grayscale_connected_opening`] — keep only the object touching `seed`,
/// suppressing every other object to `image`'s global maximum.
pub fn grayscale_connected_closing(
    image: &Image,
    seed: &[usize],
    fully_connected: bool,
) -> Result<Image> {
    let flat = seed_flat_index(seed, image.size())?;
    let (_, max_value) = crate::minimum_maximum(image)?;
    let vals = image.to_f64_vec()?;
    let seed_value = vals[flat];
    if seed_value == max_value {
        let out = vec![max_value; vals.len()];
        return image_from_f64(image.pixel_id(), image.size(), image, &out);
    }
    let mut marker_vals = vec![max_value; vals.len()];
    marker_vals[flat] = seed_value;
    let marker_image = image_from_f64(image.pixel_id(), image.size(), image, &marker_vals)?;
    reconstruction_by_erosion(&marker_image, image, fully_connected)
}

// ---- binary reconstruction by dilation / erosion ----------------------

/// Labels `indicator`'s `true` pixels into connected components
/// ([`connected_component`]) and marks each component `1..=max` "kept" iff
/// `touch` is `true` at any of its pixels — the shared core of
/// `BinaryReconstructionLabelMapFilter::ThreadedProcessLabelObject`, reused
/// by both directions of binary reconstruction (see the module docs).
fn label_and_mark_kept(
    indicator: &[bool],
    size: &[usize],
    touch: &[bool],
    fully_connected: bool,
) -> Result<(Vec<u32>, Vec<bool>)> {
    let bytes: Vec<u8> = indicator.iter().map(|&b| u8::from(b)).collect();
    let indicator_image = Image::from_vec(size, bytes)?;
    let labels = connected_component(&indicator_image, fully_connected)?;
    let label_vals = labels.scalar_slice::<u32>()?.to_vec();

    let max_label = label_vals.iter().copied().max().unwrap_or(0) as usize;
    let mut kept = vec![false; max_label + 1];
    for (&l, &t) in label_vals.iter().zip(touch) {
        if l != 0 && t {
            kept[l as usize] = true;
        }
    }
    Ok((label_vals, kept))
}

/// The `BinaryReconstructionByDilationImageFilter` half (see module docs):
/// components of `mask`'s `foreground_value` pixels that `marker` touches
/// (equals `foreground_value` anywhere on the component) render as
/// `foreground_value`; every other such component renders as
/// `background_value`; every pixel that was never `foreground_value` in
/// `mask` passes through verbatim.
fn reconstruct_by_dilation_components(
    marker: &[f64],
    mask: &[f64],
    size: &[usize],
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Vec<f64>> {
    let indicator: Vec<bool> = mask.iter().map(|&v| v == foreground_value).collect();
    let touch: Vec<bool> = marker.iter().map(|&v| v == foreground_value).collect();
    let (label_vals, kept) = label_and_mark_kept(&indicator, size, &touch, fully_connected)?;
    Ok(mask
        .iter()
        .zip(&label_vals)
        .map(|(&mv, &l)| {
            if l == 0 {
                mv
            } else if kept[l as usize] {
                foreground_value
            } else {
                background_value
            }
        })
        .collect())
}

/// The `BinaryReconstructionByErosionImageFilter` half (see module docs): the
/// dual of [`reconstruct_by_dilation_components`], derived by working on the
/// complement per `BinaryNotImageFilter`'s functor (`v == foreground_value ?
/// background_value : foreground_value` — every non-`foreground_value` value
/// collapses to the single sentinel `foreground_value`). Every
/// `mask`-`foreground_value` pixel renders as `foreground_value`; every other
/// pixel's connected component (of `mask`'s non-`foreground_value` pixels)
/// that `marker` never departs from `foreground_value` on also renders as
/// `foreground_value`; every component `marker` does leave (`!=
/// foreground_value` somewhere) keeps `mask`'s own verbatim value.
/// `background_value` never appears in this output formula — see the module
/// docs on its inertness.
fn reconstruct_by_erosion_components(
    marker: &[f64],
    mask: &[f64],
    size: &[usize],
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Vec<f64>> {
    let not = |v: f64| {
        if v == foreground_value {
            background_value
        } else {
            foreground_value
        }
    };
    let not_mask: Vec<f64> = mask.iter().map(|&v| not(v)).collect();
    let not_marker: Vec<f64> = marker.iter().map(|&v| not(v)).collect();

    let indicator: Vec<bool> = not_mask.iter().map(|&v| v == foreground_value).collect();
    let touch: Vec<bool> = not_marker.iter().map(|&v| v == foreground_value).collect();
    let (label_vals, kept) = label_and_mark_kept(&indicator, size, &touch, fully_connected)?;

    Ok(mask
        .iter()
        .zip(&label_vals)
        .map(|(&mv, &l)| {
            if l != 0 && kept[l as usize] {
                mv
            } else {
                foreground_value
            }
        })
        .collect())
}

/// `BinaryReconstructionByDilationImageFilter` (see module docs).
/// `marker_image`/`mask_image` must share size and pixel type
/// ([`FilterError::SizeMismatch`] / [`FilterError::TypeMismatch`]
/// otherwise) — unlike `reconstruction_by_dilation`, there is no
/// marker-`<=`-mask precondition, since this filter reasons about connected
/// components rather than a pointwise geodesic flood.
pub fn binary_reconstruction_by_dilation(
    marker_image: &Image,
    mask_image: &Image,
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    require_same_shape(marker_image, mask_image)?;
    let out = reconstruct_by_dilation_components(
        &marker_image.to_f64_vec()?,
        &mask_image.to_f64_vec()?,
        mask_image.size(),
        foreground_value,
        background_value,
        fully_connected,
    )?;
    image_from_f64(mask_image.pixel_id(), mask_image.size(), mask_image, &out)
}

/// `BinaryReconstructionByErosionImageFilter` (see module docs).
/// `marker_image`/`mask_image` must share size and pixel type.
/// `background_value` never influences the output except in the degenerate
/// case `background_value == foreground_value` (see the module docs).
pub fn binary_reconstruction_by_erosion(
    marker_image: &Image,
    mask_image: &Image,
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    require_same_shape(marker_image, mask_image)?;
    let out = reconstruct_by_erosion_components(
        &marker_image.to_f64_vec()?,
        &mask_image.to_f64_vec()?,
        mask_image.size(),
        foreground_value,
        background_value,
        fully_connected,
    )?;
    image_from_f64(mask_image.pixel_id(), mask_image.size(), mask_image, &out)
}

// ---- binary opening / closing by reconstruction -----------------------

/// `BinaryOpeningByReconstructionImageFilter` (see module docs):
/// binary-erode `image` by `kernel`, then keep only the mask-foreground
/// connected components erosion left touched anywhere
/// ([`binary_reconstruction_by_dilation`]). `foreground_value`/
/// `background_value` are passed through to both stages unchanged; the
/// erode step keeps its own class default `boundary_to_foreground = true`.
pub fn binary_opening_by_reconstruction(
    image: &Image,
    kernel: &StructuringElement,
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    let eroded = binary_erode(image, kernel, foreground_value, background_value, true)?;
    binary_reconstruction_by_dilation(
        &eroded,
        image,
        foreground_value,
        background_value,
        fully_connected,
    )
}

/// `BinaryClosingByReconstructionImageFilter` (see module docs):
/// binary-dilate `image` by `kernel` (own class default
/// `boundary_to_foreground = false`), then
/// [`binary_reconstruction_by_erosion`] under `image` as the mask. The
/// internal erode/dilate `background_value` is chosen the same way
/// [`crate::morphology::binary_morphological_closing`] chooses its own — `0`
/// unless `foreground_value == 0`, in which case
/// `NumericTraits<T>::max()` — since the `.hxx` has no user-facing
/// `BackgroundValue` member; unlike grayscale closing, there is no
/// `SafeBorder` padding here (`BinaryReconstructionByErosionImageFilter` has
/// no boundary condition to pad against).
pub fn binary_closing_by_reconstruction(
    image: &Image,
    kernel: &StructuringElement,
    foreground_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    let (max_value, _) = bounds_for(image.pixel_id());
    let background_value = if foreground_value == 0.0 {
        max_value
    } else {
        0.0
    };
    let dilated = binary_dilate(image, kernel, foreground_value, background_value, false)?;
    binary_reconstruction_by_erosion(
        &dilated,
        image,
        foreground_value,
        background_value,
        fully_connected,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- opening_by_reconstruction / closing_by_reconstruction ----

    /// `image = [0,5,9,5,0]`, kernel radius 1: `erode = [0,0,5,0,0]`, and the
    /// first reconstruction converges to `[0,5,5,5,0]` — the true peak height
    /// 9 is leveled down to 5, since reconstruction only needs to raise
    /// pixel 2 as far as its neighbors' mask ceiling to reach a fixed point.
    #[test]
    fn opening_by_reconstruction_levels_a_peak_without_preserve_intensities() {
        let image = img_i32(&[5], vec![0, 5, 9, 5, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let out = opening_by_reconstruction(&image, &kernel, false, false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[0, 5, 5, 5, 0]);
    }

    /// Same fixture with `preserve_intensities = true`: pixel 2 is exactly
    /// where `erode` (5) equals the first reconstruction's own output (5), so
    /// the second marker restores its true value (9) there and `MIN`
    /// elsewhere, and the second reconstruction recovers the original image
    /// exactly.
    #[test]
    fn opening_by_reconstruction_recovers_true_height_with_preserve_intensities() {
        let image = img_i32(&[5], vec![0, 5, 9, 5, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let out = opening_by_reconstruction(&image, &kernel, false, true).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[0, 5, 9, 5, 0]);
    }

    /// `D(marker,mask) == -E(-marker,-mask)` at the reconstruction-engine
    /// level makes closing-by-reconstruction the negated dual of
    /// opening-by-reconstruction: `closing_by_reconstruction(image) ==
    /// -opening_by_reconstruction(-image)`.
    #[test]
    fn closing_by_reconstruction_is_the_negated_dual_of_opening() {
        let image = img_i32(&[5], vec![0, 5, 9, 5, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let closed = closing_by_reconstruction(&image, &kernel, false, false).unwrap();

        let negated = img_i32(&[5], vec![0, -5, -9, -5, 0]);
        let opened_of_negated = opening_by_reconstruction(&negated, &kernel, false, false).unwrap();
        let expected: Vec<i32> = opened_of_negated
            .scalar_slice::<i32>()
            .unwrap()
            .iter()
            .map(|&v| -v)
            .collect();

        assert_eq!(closed.scalar_slice::<i32>().unwrap(), expected.as_slice());
    }

    /// A "dumbbell": two 3x3 squares joined by a single-pixel bridge, box
    /// kernel radius 1. Erosion strips the bridge and each square down to its
    /// own left/right rim (hand-verified via the binary-erode/dilate
    /// equivalence for a strictly two-valued image), but since the *mask*
    /// (the original dumbbell) is one connected component, reconstruction
    /// floods the marker back out to the full original shape, bridge
    /// included — unlike a plain `dilate(erode(f))` open, which only reaches
    /// as far as the kernel radius from each surviving point and never
    /// reconnects across the 1-pixel gap.
    #[test]
    fn opening_by_reconstruction_restores_a_bridge_a_plain_open_would_lose() {
        #[rustfmt::skip]
        let image = img_u8(&[7, 3], vec![
            9, 9, 9, 0, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 0, 9, 9, 9,
        ]);
        let kernel = StructuringElement::box_(&[1, 1]);

        let opened = opening_by_reconstruction(&image, &kernel, false, false).unwrap();
        assert_eq!(
            opened.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );

        let naive = grayscale_dilate(&grayscale_erode(&image, &kernel).unwrap(), &kernel).unwrap();
        #[rustfmt::skip]
        assert_eq!(naive.scalar_slice::<u8>().unwrap(), &[
            9, 9, 9, 0, 9, 9, 9,
            9, 9, 9, 0, 9, 9, 9,
            9, 9, 9, 0, 9, 9, 9,
        ]);
        assert_ne!(
            opened.scalar_slice::<u8>().unwrap(),
            naive.scalar_slice::<u8>().unwrap()
        );
    }

    /// A `from_mask` kernel that excludes its own origin can make
    /// `grayscale_erode(f, kernel) > f` at some pixel, violating
    /// `reconstruction_by_dilation`'s marker-`<=`-mask precondition; that
    /// error must surface through [`opening_by_reconstruction`] unchanged.
    #[test]
    fn opening_by_reconstruction_surfaces_reconstruction_marker_error() {
        let image = img_i32(&[3], vec![5, 0, 5]);
        let kernel = StructuringElement::from_mask(&[1], vec![true, false, true]).unwrap();
        assert_eq!(
            opening_by_reconstruction(&image, &kernel, false, false).unwrap_err(),
            FilterError::InvalidReconstructionMarker { relation: "<=" }
        );
    }

    // ---- grayscale_connected_opening / grayscale_connected_closing ----

    /// Two isolated peaks; seeding the left one keeps it at its true height
    /// and suppresses the unrelated right one to the global minimum.
    #[test]
    fn grayscale_connected_opening_keeps_only_the_seeded_object() {
        let image = img_i32(&[9], vec![0, 0, 9, 0, 0, 0, 9, 0, 0]);
        let out = grayscale_connected_opening(&image, &[2], false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            &[0, 0, 9, 0, 0, 0, 0, 0, 0]
        );
    }

    /// Dual: two isolated valleys; seeding the left one keeps it at its true
    /// depth and fills the unrelated right one up to the global maximum.
    #[test]
    fn grayscale_connected_closing_keeps_only_the_seeded_object() {
        let image = img_i32(&[9], vec![9, 9, 0, 9, 9, 9, 0, 9, 9]);
        let out = grayscale_connected_closing(&image, &[2], false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            &[9, 9, 0, 9, 9, 9, 9, 9, 9]
        );
    }

    /// A seed on the background (the global minimum) never touches any
    /// object; `.hxx`'s degenerate branch fires and the whole output is
    /// filled with the minimum, suppressing every object including the one
    /// nearest the seed.
    #[test]
    fn grayscale_connected_opening_seed_outside_any_object_fills_the_minimum() {
        let image = img_i32(&[9], vec![0, 0, 9, 0, 0, 0, 9, 0, 0]);
        let out = grayscale_connected_opening(&image, &[4], false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[0; 9]);
    }

    #[test]
    fn grayscale_connected_opening_rejects_out_of_bounds_seed() {
        let image = img_i32(&[3], vec![0, 5, 0]);
        assert_eq!(
            grayscale_connected_opening(&image, &[3], false).unwrap_err(),
            FilterError::InvalidSeedIndex {
                seed: vec![3],
                size: vec![3],
            }
        );
    }

    #[test]
    fn grayscale_connected_opening_rejects_wrong_length_seed() {
        let image = img_i32(&[3], vec![0, 5, 0]);
        assert_eq!(
            grayscale_connected_opening(&image, &[1, 0, 0], false).unwrap_err(),
            FilterError::DimensionLength {
                expected: 1,
                got: 3
            }
        );
    }

    // ---- binary_reconstruction_by_dilation / binary_reconstruction_by_erosion ----

    /// `mask = [1,1,0,1,1,1,0,9,1]`, `marker` touches only index 4 (inside
    /// the `{3,4,5}` component): that component survives as foreground; the
    /// `{0,1}` and `{8}` components drop to background; the plain-background
    /// pixels (index 2, 6) and the distinct label `9` at index 7 pass through
    /// `mask`'s own value verbatim (label-preserving, not merely
    /// foreground/background).
    #[test]
    fn binary_reconstruction_by_dilation_keeps_touched_component_and_preserves_labels() {
        let mask = img_i32(&[9, 1], vec![1, 1, 0, 1, 1, 1, 0, 9, 1]);
        let marker = img_i32(&[9, 1], vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let out = binary_reconstruction_by_dilation(&marker, &mask, 1.0, 0.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            &[0, 0, 0, 1, 1, 1, 0, 9, 0]
        );
    }

    /// The same shape and roles as above, restated with non-default
    /// `foreground_value`/`background_value` to pin that the algorithm is
    /// value-agnostic rather than hardwired to `1.0`/`0.0`.
    #[test]
    fn binary_reconstruction_by_dilation_plumbs_arbitrary_foreground_background_values() {
        let mask = img_i32(&[9, 1], vec![200, 200, 50, 200, 200, 200, 50, 9, 200]);
        let marker = img_i32(&[9, 1], vec![50, 50, 50, 50, 200, 50, 50, 50, 50]);
        let out = binary_reconstruction_by_dilation(&marker, &mask, 200.0, 50.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            &[50, 50, 50, 200, 200, 200, 50, 9, 50]
        );
    }

    /// `. X / X .`: under face connectivity the two foreground pixels are
    /// separate components, so touching only one keeps it and drops the
    /// other; under full connectivity they're one diagonally-joined
    /// component, so touching either keeps both.
    #[test]
    fn binary_reconstruction_by_dilation_connectivity_changes_which_components_merge() {
        let mask = img_u8(&[2, 2], vec![0, 1, 1, 0]);
        let marker = img_u8(&[2, 2], vec![0, 1, 0, 0]);

        let face = binary_reconstruction_by_dilation(&marker, &mask, 1.0, 0.0, false).unwrap();
        assert_eq!(face.scalar_slice::<u8>().unwrap(), &[0, 1, 0, 0]);

        let full = binary_reconstruction_by_dilation(&marker, &mask, 1.0, 0.0, true).unwrap();
        assert_eq!(full.scalar_slice::<u8>().unwrap(), &[0, 1, 1, 0]);
    }

    #[test]
    fn binary_reconstruction_by_dilation_rejects_mismatched_sizes() {
        let marker = img_u8(&[3, 1], vec![1, 1, 1]);
        let mask = img_u8(&[4, 1], vec![1, 1, 1, 1]);
        assert!(matches!(
            binary_reconstruction_by_dilation(&marker, &mask, 1.0, 0.0, false),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    /// `mask = [0,0,1,3,3,1,0,0,1,0]` (foreground = 1): the not-mask
    /// components are `{0,1}`, `{3,4}` (values `3,3`, a distinct label),
    /// `{6,7}`, `{9}`. `marker` is non-foreground (`0`) only at index 3 and
    /// index 9, so only the `{3,4}` and `{9}` components survive (keep
    /// `mask`'s own value); `{0,1}` and `{6,7}` collapse to
    /// `foreground_value`; every `mask`-foreground pixel (indices 2, 5, 8) is
    /// `foreground_value` unconditionally.
    #[test]
    fn binary_reconstruction_by_erosion_keeps_untouched_background_components() {
        let mask = img_i32(&[10, 1], vec![0, 0, 1, 3, 3, 1, 0, 0, 1, 0]);
        let marker = img_i32(&[10, 1], vec![1, 1, 1, 0, 1, 1, 1, 1, 1, 0]);
        let out = binary_reconstruction_by_erosion(&marker, &mask, 1.0, 0.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            &[1, 1, 1, 3, 3, 1, 1, 1, 1, 0]
        );
    }

    /// Same fixture, `background_value` changed from `0` to `99`:
    /// `background_value` never appears in [`reconstruct_by_erosion_components`]'s
    /// output formula (see the module docs), so the result must be identical.
    #[test]
    fn binary_reconstruction_by_erosion_output_is_independent_of_background_value() {
        let mask = img_i32(&[10, 1], vec![0, 0, 1, 3, 3, 1, 0, 0, 1, 0]);
        let marker = img_i32(&[10, 1], vec![1, 1, 1, 0, 1, 1, 1, 1, 1, 0]);
        let out = binary_reconstruction_by_erosion(&marker, &mask, 1.0, 99.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            &[1, 1, 1, 3, 3, 1, 1, 1, 1, 0]
        );
    }

    // ---- binary_opening_by_reconstruction / binary_closing_by_reconstruction ----

    /// The same dumbbell as [`opening_by_reconstruction_restores_a_bridge_a_plain_open_would_lose`],
    /// exercised through the binary entry point end to end (erode + label
    /// reconstruction) instead of the grayscale engine, contrasted against
    /// the existing [`crate::morphology::binary_morphological_opening`].
    #[test]
    fn binary_opening_by_reconstruction_restores_a_bridge_a_plain_open_would_lose() {
        #[rustfmt::skip]
        let image = img_u8(&[7, 3], vec![
            1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1,
        ]);
        let kernel = StructuringElement::box_(&[1, 1]);

        let opened = binary_opening_by_reconstruction(&image, &kernel, 1.0, 0.0, false).unwrap();
        assert_eq!(
            opened.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );

        let naive =
            crate::morphology::binary_morphological_opening(&image, &kernel, 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(naive.scalar_slice::<u8>().unwrap(), &[
            1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1,
        ]);
        assert_ne!(
            opened.scalar_slice::<u8>().unwrap(),
            naive.scalar_slice::<u8>().unwrap()
        );
    }

    /// A gap of one background pixel (index 4) between two foreground runs,
    /// comfortably clear of the image border so no boundary condition
    /// participates: dilating by radius 1 bridges the gap, and since the
    /// bridging marker pixel is non-foreground nowhere the gap's own
    /// not-mask component is left untouched, the reconstruction-by-erosion
    /// stage fills the gap in the final output.
    #[test]
    fn binary_closing_by_reconstruction_fills_a_gap_narrower_than_the_kernel() {
        let image = img_u8(&[9], vec![0, 0, 9, 9, 0, 9, 9, 0, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let out = binary_closing_by_reconstruction(&image, &kernel, 9.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            &[0, 0, 9, 9, 9, 9, 9, 0, 0]
        );
    }

    /// `foreground_value == 0.0` forces the internal `background_value` to
    /// `NumericTraits<T>::max()` instead of `0.0` (see the module docs); an
    /// already-closed run (no internal gap narrower than the kernel) must
    /// come back unchanged, confirming that branch computes correctly rather
    /// than colliding `0` with itself.
    #[test]
    fn binary_closing_by_reconstruction_foreground_zero_uses_max_as_internal_background() {
        let image = img_u8(&[7], vec![200, 200, 0, 0, 0, 200, 200]);
        let kernel = StructuringElement::box_(&[1]);
        let out = binary_closing_by_reconstruction(&image, &kernel, 0.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );
    }
}
