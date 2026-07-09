//! Geodesic grayscale morphology: run one elementary iteration, or to
//! convergence.
//!
//! Ports of (ITK `Modules/Filtering/MathematicalMorphology/include/`):
//! `itkGrayscaleGeodesicDilateImageFilter.hxx` /
//! `itkGrayscaleGeodesicErodeImageFilter.hxx`.
//!
//! ## Elementary iteration
//!
//! Each call to `DynamicThreadedGenerateData` dilates (erodes) the marker
//! image by one "elementary" structuring element — a radius-one
//! `ConstShapedNeighborhoodIterator` over the marker, with
//! `ZeroFluxNeumannBoundaryCondition` (edge-*clamped*, unlike
//! [`crate::reconstruction`]'s engine, which simply skips out-of-bounds
//! neighbors because its own padding sentinel can never win a comparison —
//! that reasoning does not apply here, since this filter's boundary supplies
//! a real, possibly-winning replicated pixel value) — then clips the result
//! to the mask (min against the mask for dilate, max for erode).
//! `RunOneIteration = false` (the yaml default) repeats this until the
//! output stops changing (compared against that iteration's own input, i.e.
//! the previous output); `true` performs exactly one pass.
//!
//! Ported here with [`sitk_core::NeighborhoodIterator`] +
//! [`sitk_core::ZeroFluxNeumannBoundaryCondition`] +
//! [`crate::morphology::StructuringElement`]'s on-mask, the same
//! reduce-over-kernel-on-positions shape [`crate::morphology::grayscale_dilate`]
//! / [`crate::morphology::grayscale_erode`] already use, just against a
//! different boundary condition and kernel.
//!
//! ## `FullyConnected` selects a genuinely different elementary kernel — not
//! just a bigger one
//!
//! The `.hxx` activates a different active-offset set depending on
//! `FullyConnected`:
//!
//! - `FullyConnected = false` (yaml default): the elementary structuring
//!   element is the *center pixel plus its face-connected neighbors*
//!   (`ActivateOffset` on the zero offset, then each `±1` axis offset) —
//!   [`crate::morphology::StructuringElement::cross`] at radius 1.
//! - `FullyConnected = true`: every offset in the 3^dim radius-1 box is
//!   activated, **then the center offset is explicitly deactivated**
//!   (`DeactivateOffset` on the zero offset, after activating everything) —
//!   the center pixel's own value never directly participates in that
//!   iteration's max/min; only its face+diagonal neighbors do.
//!
//! So `FullyConnected = true` is not simply "more neighbors than
//! `FullyConnected = false`" — it drops the one thing `FullyConnected =
//! false` always includes (the center itself). This is upstream's own
//! asymmetry, reproduced unchanged (not a deliberate divergence): see
//! [`elementary_kernel`].
//!
//! One consequence, verified in the tests below: a converged
//! (`RunOneIteration = false`), `FullyConnected = false` geodesic
//! dilation/erosion equals
//! [`crate::reconstruction::reconstruction_by_dilation`] /
//! [`reconstruction_by_erosion`](crate::reconstruction::reconstruction_by_erosion)
//! at `fully_connected = false` — both engines' per-pixel update always
//! includes the pixel's own current value alongside its face-connected
//! neighbors, so they share the same fixed-point equation. A converged
//! `FullyConnected = true` run does **not** agree with
//! `reconstruction_by_dilation`/`reconstruction_by_erosion` at
//! `fully_connected = true` in general, even though both are nominally
//! "reconstruction" per the class docs: [`crate::reconstruction::reconstruct`]'s
//! per-pixel update *always* starts from the pixel's own current value
//! (`let mut v = out[f];`) regardless of `fully_connected`, so it is
//! extensive (never drops below the marker) by construction, while this
//! filter's `FullyConnected = true` elementary step never lets the center
//! participate — an isolated marker pixel with no support from its
//! neighbors collapses on the very first iteration instead of holding. See
//! `fully_connected_converged_geodesic_dilate_diverges_from_reconstruction_on_an_isolated_marker`
//! below for a confirmed minimal counterexample (found by exhaustively
//! brute-forcing every marker/mask pair on a 3x3 binary grid, not merely
//! theorized).
//!
//! ## No marker/mask precondition check
//!
//! Unlike [`crate::reconstruction::reconstruction_by_erosion`] /
//! [`reconstruction_by_dilation`](crate::reconstruction::reconstruction_by_dilation)
//! (which port `itkReconstructionImageFilter.hxx`'s explicit
//! `itkExceptionMacro` precondition check), neither
//! `GrayscaleGeodesicDilateImageFilter.hxx` nor
//! `GrayscaleGeodesicErodeImageFilter.hxx` validates marker-vs-mask anywhere
//! in `GenerateData`/`DynamicThreadedGenerateData` — only their class docs
//! *say* the marker must be `<=`/`>=` the mask. This port matches that:
//! [`grayscale_geodesic_dilate`]/[`grayscale_geodesic_erode`] never check the
//! relation and never error over it; garbage in is garbage out, exactly as
//! upstream. (`marker_image`/`mask_image` must still share size and pixel
//! type — that structural requirement comes from this being a Rust port of a
//! filter whose two inputs share one C++ template parameter, not from any
//! runtime check upstream performs.)

use crate::error::Result;
use crate::morphology::{StructuringElement, bounds_for};
use crate::require_same_shape;
use sitk_core::{
    Image, NeighborhoodIterator, Scalar, ZeroFluxNeumannBoundaryCondition, dispatch_scalar,
};

/// The elementary structuring element `FullyConnected` selects (see module
/// docs): [`StructuringElement::cross`] at radius 1 (center included) when
/// `false`, or the radius-1 box with its center offset switched off when
/// `true`.
fn elementary_kernel(dim: usize, fully_connected: bool) -> StructuringElement {
    let radius = vec![1usize; dim];
    if !fully_connected {
        StructuringElement::cross(&radius)
    } else {
        let total = 3usize.pow(dim as u32);
        let mut mask = vec![true; total];
        mask[total / 2] = false; // the center offset, deactivated
        StructuringElement::from_mask(&radius, mask)
            .expect("mask length matches radius by construction")
    }
}

fn geodesic_dilate_iteration_typed<T: Scalar>(
    marker: &Image,
    mask: &Image,
    kernel: &StructuringElement,
) -> Result<Image> {
    let (_, nonpositive_min) = bounds_for(marker.pixel_id());
    let init = T::from_f64(nonpositive_min);
    let mask_vals = mask.scalar_slice::<T>()?;
    let iter =
        NeighborhoodIterator::new(marker, kernel.radius(), ZeroFluxNeumannBoundaryCondition)?;
    let mut out = Vec::with_capacity(marker.number_of_pixels());
    for (i, (_, nb)) in iter.enumerate() {
        let mut v = init;
        for (&on, &val) in kernel.on().iter().zip(nb.values()) {
            if on && val > v {
                v = val;
            }
        }
        let mv = mask_vals[i];
        out.push(if mv < v { mv } else { v });
    }
    let mut result = Image::from_vec(marker.size(), out)?;
    result.copy_geometry_from(marker);
    Ok(result)
}

fn geodesic_erode_iteration_typed<T: Scalar>(
    marker: &Image,
    mask: &Image,
    kernel: &StructuringElement,
) -> Result<Image> {
    let (max_value, _) = bounds_for(marker.pixel_id());
    let init = T::from_f64(max_value);
    let mask_vals = mask.scalar_slice::<T>()?;
    let iter =
        NeighborhoodIterator::new(marker, kernel.radius(), ZeroFluxNeumannBoundaryCondition)?;
    let mut out = Vec::with_capacity(marker.number_of_pixels());
    for (i, (_, nb)) in iter.enumerate() {
        let mut v = init;
        for (&on, &val) in kernel.on().iter().zip(nb.values()) {
            if on && val < v {
                v = val;
            }
        }
        let mv = mask_vals[i];
        out.push(if mv > v { mv } else { v });
    }
    let mut result = Image::from_vec(marker.size(), out)?;
    result.copy_geometry_from(marker);
    Ok(result)
}

/// `GrayscaleGeodesicDilateImageFilter::GenerateData`: run
/// [`geodesic_dilate_iteration_typed`] once (`run_one_iteration`) or until
/// the output no longer changes.
fn geodesic_dilate_typed<T: Scalar>(
    marker: &Image,
    mask: &Image,
    kernel: &StructuringElement,
    run_one_iteration: bool,
) -> Result<Image> {
    let mut current = geodesic_dilate_iteration_typed::<T>(marker, mask, kernel)?;
    if run_one_iteration {
        return Ok(current);
    }
    loop {
        let next = geodesic_dilate_iteration_typed::<T>(&current, mask, kernel)?;
        if next.scalar_slice::<T>()? == current.scalar_slice::<T>()? {
            return Ok(next);
        }
        current = next;
    }
}

/// `GrayscaleGeodesicErodeImageFilter::GenerateData`: the erosion dual of
/// [`geodesic_dilate_typed`].
fn geodesic_erode_typed<T: Scalar>(
    marker: &Image,
    mask: &Image,
    kernel: &StructuringElement,
    run_one_iteration: bool,
) -> Result<Image> {
    let mut current = geodesic_erode_iteration_typed::<T>(marker, mask, kernel)?;
    if run_one_iteration {
        return Ok(current);
    }
    loop {
        let next = geodesic_erode_iteration_typed::<T>(&current, mask, kernel)?;
        if next.scalar_slice::<T>()? == current.scalar_slice::<T>()? {
            return Ok(next);
        }
        current = next;
    }
}

/// `GrayscaleGeodesicDilateImageFilter`: geodesic dilation of `marker_image`
/// under `mask_image` — one elementary radius-1 dilation clipped to the mask
/// (`run_one_iteration = true`), or run to convergence
/// (`run_one_iteration = false`, the yaml default). See the module docs for
/// the `fully_connected` elementary-kernel asymmetry and the (deliberately
/// unenforced) marker/mask precondition.
pub fn grayscale_geodesic_dilate(
    marker_image: &Image,
    mask_image: &Image,
    run_one_iteration: bool,
    fully_connected: bool,
) -> Result<Image> {
    require_same_shape(marker_image, mask_image)?;
    let kernel = elementary_kernel(marker_image.dimension(), fully_connected);
    dispatch_scalar!(
        marker_image.pixel_id(),
        geodesic_dilate_typed,
        marker_image,
        mask_image,
        &kernel,
        run_one_iteration
    )
}

/// `GrayscaleGeodesicErodeImageFilter`: the erosion dual of
/// [`grayscale_geodesic_dilate`].
pub fn grayscale_geodesic_erode(
    marker_image: &Image,
    mask_image: &Image,
    run_one_iteration: bool,
    fully_connected: bool,
) -> Result<Image> {
    require_same_shape(marker_image, mask_image)?;
    let kernel = elementary_kernel(marker_image.dimension(), fully_connected);
    dispatch_scalar!(
        marker_image.pixel_id(),
        geodesic_erode_typed,
        marker_image,
        mask_image,
        &kernel,
        run_one_iteration
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction::{reconstruction_by_dilation, reconstruction_by_erosion};

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- dilate: one iteration vs convergence ----

    #[test]
    fn geodesic_dilate_one_iteration_spreads_by_one_pixel_only() {
        let marker = img_i32(&[7, 1], vec![0, 0, 0, 9, 0, 0, 0]);
        let mask = img_i32(&[7, 1], vec![9; 7]);
        let one = grayscale_geodesic_dilate(&marker, &mask, true, false).unwrap();
        assert_eq!(one.scalar_slice::<i32>().unwrap(), &[0, 0, 9, 9, 9, 0, 0]);
    }

    #[test]
    fn geodesic_dilate_converges_by_flooding_up_to_the_mask_ceiling() {
        let marker = img_i32(&[7, 1], vec![0, 0, 0, 9, 0, 0, 0]);
        let mask = img_i32(&[7, 1], vec![9; 7]);
        let converged = grayscale_geodesic_dilate(&marker, &mask, false, false).unwrap();
        assert_eq!(converged.scalar_slice::<i32>().unwrap(), &[9; 7]);
    }

    // ---- erode: one iteration vs convergence (dual of the above) ----

    #[test]
    fn geodesic_erode_one_iteration_spreads_by_one_pixel_only() {
        let marker = img_i32(&[7, 1], vec![0, 0, 0, -9, 0, 0, 0]);
        let mask = img_i32(&[7, 1], vec![-9; 7]);
        let one = grayscale_geodesic_erode(&marker, &mask, true, false).unwrap();
        assert_eq!(
            one.scalar_slice::<i32>().unwrap(),
            &[0, 0, -9, -9, -9, 0, 0]
        );
    }

    #[test]
    fn geodesic_erode_converges_by_flooding_down_to_the_mask_floor() {
        let marker = img_i32(&[7, 1], vec![0, 0, 0, -9, 0, 0, 0]);
        let mask = img_i32(&[7, 1], vec![-9; 7]);
        let converged = grayscale_geodesic_erode(&marker, &mask, false, false).unwrap();
        assert_eq!(converged.scalar_slice::<i32>().unwrap(), &[-9; 7]);
    }

    // ---- FullyConnected = false converges to plain reconstruction ----

    #[test]
    fn converged_face_connected_geodesic_dilate_matches_reconstruction_by_dilation() {
        let marker = img_i32(&[5, 1], vec![3, 0, 2, 0, 3]);
        let mask = img_i32(&[5, 1], vec![5, 2, 4, 2, 5]);
        let geodesic = grayscale_geodesic_dilate(&marker, &mask, false, false).unwrap();
        let reconstruction = reconstruction_by_dilation(&marker, &mask, false).unwrap();
        assert_eq!(
            geodesic.scalar_slice::<i32>().unwrap(),
            reconstruction.scalar_slice::<i32>().unwrap()
        );
    }

    #[test]
    fn converged_face_connected_geodesic_erode_matches_reconstruction_by_erosion() {
        let marker = img_i32(&[5, 1], vec![2, 5, 3, 5, 2]);
        let mask = img_i32(&[5, 1], vec![0, 3, 1, 3, 0]);
        let geodesic = grayscale_geodesic_erode(&marker, &mask, false, false).unwrap();
        let reconstruction = reconstruction_by_erosion(&marker, &mask, false).unwrap();
        assert_eq!(
            geodesic.scalar_slice::<i32>().unwrap(),
            reconstruction.scalar_slice::<i32>().unwrap()
        );
    }

    // ---- no marker/mask precondition check (unlike reconstruction) ----

    #[test]
    fn geodesic_dilate_does_not_reject_a_marker_above_the_mask() {
        // marker[0] = 9 > mask[0] = 1: `reconstruction_by_dilation` would
        // error on this; the geodesic filter never checks and just clips the
        // offending pixel down to the mask on its very first iteration.
        let marker = img_i32(&[3, 1], vec![9, 0, 0]);
        let mask = img_i32(&[3, 1], vec![1, 1, 1]);
        let out = grayscale_geodesic_dilate(&marker, &mask, true, false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[1, 1, 0]);
    }

    #[test]
    fn geodesic_erode_does_not_reject_a_marker_below_the_mask() {
        let marker = img_i32(&[3, 1], vec![-9, 0, 0]);
        let mask = img_i32(&[3, 1], vec![-1, -1, -1]);
        let out = grayscale_geodesic_erode(&marker, &mask, true, false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[-1, -1, 0]);
    }

    // ---- FullyConnected excludes the center: same-mask comparison ----

    #[test]
    fn fully_connected_one_iteration_can_lower_a_pixel_below_its_own_marker_value() {
        // FullyConnected excludes the center pixel from its own elementary
        // dilation, so a value strictly above both its neighbors can drop —
        // impossible under FullyConnected = false, whose elementary kernel
        // always includes the center (see module docs).
        //
        // This must be a genuinely 1-D image (`&[3]`, not `&[3, 1]`): on a
        // 2-D image with a height-1 axis, ZeroFluxNeumann clamps that axis's
        // `dy = ±1` offsets back onto the same row, silently reintroducing
        // the center's own value through those offsets and defeating the
        // center exclusion this test means to exercise.
        let marker = img_i32(&[3], vec![0, 5, 0]);
        let mask = img_i32(&[3], vec![9, 9, 9]);
        let full = grayscale_geodesic_dilate(&marker, &mask, true, true).unwrap();
        assert_eq!(full.scalar_slice::<i32>().unwrap(), &[5, 0, 5]);
        let face = grayscale_geodesic_dilate(&marker, &mask, true, false).unwrap();
        assert_eq!(face.scalar_slice::<i32>().unwrap(), &[5, 5, 5]);
    }

    // ---- ZeroFluxNeumann boundary: edge pixels see a replicated neighbor ----

    #[test]
    fn boundary_pixels_use_a_replicated_edge_neighbor_not_a_skipped_one() {
        // Single-pixel image: its only "neighbors" are itself, replicated by
        // ZeroFluxNeumann on both sides. If out-of-bounds neighbors were
        // instead skipped (`crate::reconstruction`'s convention), the result
        // would be identical here since the center is always in-kernel for
        // FullyConnected = false — this instead exercises FullyConnected =
        // true, where the only active offsets are the (replicated) left/right
        // neighbors, both of which equal the sole pixel's own value.
        let marker = img_i32(&[1], vec![4]);
        let mask = img_i32(&[1], vec![9]);
        let out = grayscale_geodesic_dilate(&marker, &mask, true, true).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[4]);
    }

    // ---- FullyConnected = true does NOT converge to reconstruction ----

    #[test]
    fn fully_connected_converged_geodesic_dilate_diverges_from_reconstruction_on_an_isolated_marker()
     {
        // A single interior marker pixel with no support from its neighbors
        // (everything else, marker and mask alike, is 0): FullyConnected =
        // true's elementary step never lets a pixel see its own value, so
        // the lone `1` has nothing to sustain it and collapses to `0` on the
        // very first iteration — it stays there, since a field of zeros is
        // already a fixed point. Standard reconstruction always folds in the
        // pixel's own current value (`crate::reconstruction::reconstruct`'s
        // `let mut v = out[f];`), so that same pixel is trivially its own
        // fixed point and reconstruction leaves it at `1`. Confirmed as an
        // actual (not merely theoretical) divergence by exhaustively
        // brute-forcing every marker/mask pair on a 3x3 binary grid; this is
        // the minimal case found. See the module docs.
        let marker = img_i32(&[3, 3], vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let mask = img_i32(&[3, 3], vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let geodesic = grayscale_geodesic_dilate(&marker, &mask, false, true).unwrap();
        assert_eq!(geodesic.scalar_slice::<i32>().unwrap(), &[0; 9]);
        let reconstruction = reconstruction_by_dilation(&marker, &mask, true).unwrap();
        assert_eq!(reconstruction.scalar_slice::<i32>().unwrap()[4], 1);
    }
}
