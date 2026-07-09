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
//!
//! (`BinaryOpeningByReconstruction`/`BinaryClosingByReconstruction` and
//! `BinaryReconstructionByDilation`/`Erosion` land in this module in later
//! commits.)
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
//! convergence. `OpeningByReconstructionImageFilter`/
//! `ClosingByReconstructionImageFilter` (and, in later commits, the other
//! filters in this module) are thin `.hxx` wrappers that build a marker
//! image and hand it straight to `ReconstructionByDilationImageFilter` /
//! `ReconstructionByErosionImageFilter` with no logic of their own beyond
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

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::morphology::{StructuringElement, bounds_for, grayscale_dilate, grayscale_erode};
use crate::reconstruction::{reconstruction_by_dilation, reconstruction_by_erosion};
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
    let a_vals = a.to_f64_vec();
    let b_vals = b.to_f64_vec();
    let source_vals = source.to_f64_vec();
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
    let vals = image.to_f64_vec();
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
    let vals = image.to_f64_vec();
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
}
