//! Morphological opening/closing by reconstruction, connected opening/
//! closing, and binary reconstruction by dilation/erosion.
//!
//! Ports of (ITK `Modules/Filtering/{MathematicalMorphology,
//! BinaryMathematicalMorphology,LabelMap}/include/`):
//!
//! - [`opening_by_reconstruction`] / [`closing_by_reconstruction`] —
//!   `itkOpeningByReconstructionImageFilter.h` / `.hxx`,
//!   `itkClosingByReconstructionImageFilter.h` / `.hxx`.
//!
//! (`GrayscaleConnectedOpening`/`Closing`, `BinaryOpeningByReconstruction`/
//! `BinaryClosingByReconstruction`, and `BinaryReconstructionByDilation`/
//! `Erosion` land in this module in later commits.)
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

use crate::error::Result;
use crate::image_from_f64;
use crate::morphology::{StructuringElement, bounds_for, grayscale_dilate, grayscale_erode};
use crate::reconstruction::{reconstruction_by_dilation, reconstruction_by_erosion};
use sitk_core::Image;

// ---- shared helpers --------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FilterError;

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
}
