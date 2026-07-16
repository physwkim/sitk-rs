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
//! [`crate::filters::reconstruction`]'s engine, which simply skips out-of-bounds
//! neighbors because its own padding sentinel can never win a comparison —
//! that reasoning does not apply here, since this filter's boundary supplies
//! a real, possibly-winning replicated pixel value) — then clips the result
//! to the mask (min against the mask for dilate, max for erode).
//! `RunOneIteration = false` (the yaml default) repeats this until the
//! output stops changing (compared against that iteration's own input, i.e.
//! the previous output); `true` performs exactly one pass.
//!
//! Ported here with [`crate::core::NeighborhoodIterator`] +
//! [`crate::core::ZeroFluxNeumannBoundaryCondition`] +
//! [`crate::filters::morphology::StructuringElement`]'s on-mask, the same
//! reduce-over-kernel-on-positions shape [`crate::filters::morphology::grayscale_dilate`]
//! / [`crate::filters::morphology::grayscale_erode`] already use, just against a
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
//!   [`crate::filters::morphology::StructuringElement::cross`] at radius 1.
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
//! `elementary_kernel`.
//!
//! One consequence, verified in the tests below: a converged
//! (`RunOneIteration = false`), `FullyConnected = false` geodesic
//! dilation/erosion equals
//! [`crate::filters::reconstruction::reconstruction_by_dilation`] /
//! [`reconstruction_by_erosion`](crate::filters::reconstruction::reconstruction_by_erosion)
//! at `fully_connected = false` — both engines' per-pixel update always
//! includes the pixel's own current value alongside its face-connected
//! neighbors, so they share the same fixed-point equation. A converged
//! `FullyConnected = true` run does **not** agree with
//! `reconstruction_by_dilation`/`reconstruction_by_erosion` at
//! `fully_connected = true` in general, even though both are nominally
//! "reconstruction" per the class docs: `crate::filters::reconstruction::reconstruct`'s
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
//! Unlike [`crate::filters::reconstruction::reconstruction_by_erosion`] /
//! [`reconstruction_by_dilation`](crate::filters::reconstruction::reconstruction_by_dilation)
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

use crate::core::{
    Image, NeighborhoodIterator, Scalar, ZeroFluxNeumannBoundaryCondition, dispatch_scalar,
};
use crate::filters::error::Result;
use crate::filters::geometry::require_same_physical_space;
use crate::filters::morphology::{StructuringElement, bounds_for};
use crate::filters::require_same_shape;

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

/// Image strides, dimension 0 fastest — the same order `Image::linear_index`
/// uses.
///
/// The serial walks below took their linear voxel index from `iter.enumerate()`.
/// A parallel window map has no counterpart for that, so the index is recovered
/// from the ND window center instead. It is only ever used to read the mask at
/// the *same* voxel; it enters no accumulation.
fn linear_strides(size: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; size.len()];
    let mut stride = 1usize;
    for (s, &extent) in strides.iter_mut().zip(size) {
        *s = stride;
        stride *= extent;
    }
    strides
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
    let strides = linear_strides(marker.size());

    // Parallel over output voxels. `kernel.on()` is in the window's own
    // dimension-0-fastest slot order, so it zips straight onto the borrowed
    // window; the scan visits one voxel's own window in the same order, and the
    // strict `val > v` keeps the first maximum on ties exactly as before. The
    // pointwise `min` against the mask reads the *same* voxel, so nothing crosses
    // voxels and no thread count can reach the result.
    let out: Vec<T> = iter.par_map_window(|center, w| {
        let mut v = init;
        for (&on, val) in kernel.on().iter().zip(w.iter()) {
            if on && val > v {
                v = val;
            }
        }
        let i: usize = center.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
        let mv = mask_vals[i];
        if mv < v { mv } else { v }
    });
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
    let strides = linear_strides(marker.size());

    // Parallel over output voxels — see `geodesic_dilate_iteration_typed`; this is
    // the same scan with the comparison and the mask clamp inverted.
    let out: Vec<T> = iter.par_map_window(|center, w| {
        let mut v = init;
        for (&on, val) in kernel.on().iter().zip(w.iter()) {
            if on && val < v {
                v = val;
            }
        }
        let i: usize = center.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
        let mv = mask_vals[i];
        if mv > v { mv } else { v }
    });
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
    require_same_physical_space(marker_image, mask_image, 1)?;
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
    require_same_physical_space(marker_image, mask_image, 1)?;
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
    use crate::filters::reconstruction::{reconstruction_by_dilation, reconstruction_by_erosion};

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
        // instead skipped (`crate::filters::reconstruction`'s convention), the result
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
        // pixel's own current value (`crate::filters::reconstruction::reconstruct`'s
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

/// Thread-count parity pins for the two geodesic iteration stencils.
///
/// A **comparison** stencil (max/min over the window, then a pointwise clamp
/// against the mask), so the non-vacuity guard is slot-dependence, not fold
/// order — see [`crate::filters::morphology`]'s pin module for why a fold-order guard on a
/// max would be a vacuous assertion.
///
/// These pins also cover the one thing that *was* newly derived here: the serial
/// walk got the mask's linear index from `enumerate()`, and the parallel pass
/// reconstructs it from the window's ND center against the image strides. The
/// reference below still uses `enumerate()`, so if that reconstruction were off by
/// even one voxel — a swapped axis, a wrong stride — the mask would be read at the
/// wrong place and these pins would fail.
///
/// `-0.0` does not apply: no value is added or accumulated. Every output is one of
/// the input values, selected by comparison, and a strict `>` / `<` keeps the first
/// extremum on ties exactly as the serial scan did — so `-0.0` and `0.0`, which
/// compare equal, resolve the same way they did before.
#[cfg(test)]
mod stencil_thread_parity {
    use super::*;
    use crate::core::{PixelId, parallel};
    use crate::filters::stencil_test_support::{PIXELS, THREADS, assert_bits_eq, volume};

    /// A marker strictly below the mask, so the dilation has somewhere to grow and
    /// the mask clamp actually bites.
    fn marker_of(mask: &Image) -> Image {
        let vals: Vec<f64> = mask.to_f64_vec().unwrap().iter().map(|v| v - 9.5).collect();
        let mut m = match mask.pixel_id() {
            PixelId::Float64 => Image::from_vec(mask.size(), vals).unwrap(),
            PixelId::Float32 => {
                let d: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
                Image::from_vec(mask.size(), d).unwrap()
            }
            other => panic!("pin does not cover {other:?}"),
        };
        m.copy_geometry_from(mask);
        m
    }

    // ---- the serial references: the exact loops that were deleted -----------

    fn dilate_iteration_serial<T: Scalar>(
        marker: &Image,
        mask: &Image,
        kernel: &StructuringElement,
    ) -> Vec<f64> {
        let (_, nonpositive_min) = bounds_for(marker.pixel_id());
        let init = T::from_f64(nonpositive_min);
        let mask_vals = mask.scalar_slice::<T>().unwrap();
        let iter = NeighborhoodIterator::<T, _>::new(
            marker,
            kernel.radius(),
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();
        iter.enumerate()
            .map(|(i, (_, nb))| {
                let mut v = init;
                for (&on, &val) in kernel.on().iter().zip(nb.values()) {
                    if on && val > v {
                        v = val;
                    }
                }
                let mv = mask_vals[i];
                if mv < v { mv.as_f64() } else { v.as_f64() }
            })
            .collect()
    }

    fn erode_iteration_serial<T: Scalar>(
        marker: &Image,
        mask: &Image,
        kernel: &StructuringElement,
    ) -> Vec<f64> {
        let (max_value, _) = bounds_for(marker.pixel_id());
        let init = T::from_f64(max_value);
        let mask_vals = mask.scalar_slice::<T>().unwrap();
        let iter = NeighborhoodIterator::<T, _>::new(
            marker,
            kernel.radius(),
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();
        iter.enumerate()
            .map(|(i, (_, nb))| {
                let mut v = init;
                for (&on, &val) in kernel.on().iter().zip(nb.values()) {
                    if on && val < v {
                        v = val;
                    }
                }
                let mv = mask_vals[i];
                if mv > v { mv.as_f64() } else { v.as_f64() }
            })
            .collect()
    }

    fn serial(marker: &Image, mask: &Image, fully_connected: bool, dilate: bool) -> Vec<f64> {
        let kernel = elementary_kernel(marker.dimension(), fully_connected);
        match (marker.pixel_id(), dilate) {
            (PixelId::Float64, true) => dilate_iteration_serial::<f64>(marker, mask, &kernel),
            (PixelId::Float64, false) => erode_iteration_serial::<f64>(marker, mask, &kernel),
            (PixelId::Float32, true) => dilate_iteration_serial::<f32>(marker, mask, &kernel),
            (PixelId::Float32, false) => erode_iteration_serial::<f32>(marker, mask, &kernel),
            (other, _) => panic!("pin does not cover {other:?}"),
        }
    }

    // ---- non-vacuity --------------------------------------------------------

    /// The pin means nothing unless the iteration actually moves voxels, and
    /// unless the *mask* read — the part whose index is now reconstructed rather
    /// than counted — actually decides some of them. Both are asserted: the
    /// dilation must change a large share of the marker, and the mask clamp must
    /// bind on a large share (otherwise a mask read at the wrong index would
    /// still produce the right answer).
    #[test]
    fn the_iteration_moves_voxels_and_the_mask_clamp_binds() {
        let mask = volume(PixelId::Float64);
        let marker = marker_of(&mask);
        let out = grayscale_geodesic_dilate(&marker, &mask, true, true).unwrap();

        let (m0, mk, o) = (
            marker.to_f64_vec().unwrap(),
            mask.to_f64_vec().unwrap(),
            out.to_f64_vec().unwrap(),
        );
        let moved = o.iter().zip(&m0).filter(|(a, b)| a != b).count();
        assert!(
            moved > o.len() / 2,
            "the geodesic dilation moved only {moved}/{} voxels — too static to pin anything",
            o.len()
        );

        let clamped = o.iter().zip(&mk).filter(|(a, b)| a == b).count();
        assert!(
            clamped > o.len() / 10,
            "the mask clamp bound on only {clamped}/{} voxels — a mask read at the wrong \
             index would mostly not show, so this pin could not catch a bad index \
             reconstruction",
            o.len()
        );
    }

    // ---- the pins -----------------------------------------------------------

    #[test]
    fn geodesic_dilate_iteration_is_bit_identical_at_every_thread_count() {
        for pixel in PIXELS {
            let mask = volume(pixel);
            let marker = marker_of(&mask);
            for fully_connected in [false, true] {
                let expected = serial(&marker, &mask, fully_connected, true);
                for threads in THREADS {
                    let got = parallel::with_threads(threads, || {
                        grayscale_geodesic_dilate(&marker, &mask, true, fully_connected)
                    })
                    .unwrap()
                    .to_f64_vec()
                    .unwrap();
                    assert_bits_eq(
                        &got,
                        &expected,
                        &format!(
                            "grayscale_geodesic_dilate({pixel:?}, \
                             fully_connected={fully_connected}, {threads} threads)"
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn geodesic_erode_iteration_is_bit_identical_at_every_thread_count() {
        for pixel in PIXELS {
            let mask = volume(pixel);
            // For the erosion the marker must sit *above* the mask, so use the mask
            // as the marker's floor rather than its ceiling.
            let marker = {
                let vals: Vec<f64> = mask.to_f64_vec().unwrap().iter().map(|v| v + 9.5).collect();
                let mut m = match pixel {
                    PixelId::Float64 => Image::from_vec(mask.size(), vals).unwrap(),
                    PixelId::Float32 => Image::from_vec(
                        mask.size(),
                        vals.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
                    )
                    .unwrap(),
                    other => panic!("pin does not cover {other:?}"),
                };
                m.copy_geometry_from(&mask);
                m
            };
            for fully_connected in [false, true] {
                let expected = serial(&marker, &mask, fully_connected, false);
                for threads in THREADS {
                    let got = parallel::with_threads(threads, || {
                        grayscale_geodesic_erode(&marker, &mask, true, fully_connected)
                    })
                    .unwrap()
                    .to_f64_vec()
                    .unwrap();
                    assert_bits_eq(
                        &got,
                        &expected,
                        &format!(
                            "grayscale_geodesic_erode({pixel:?}, \
                             fully_connected={fully_connected}, {threads} threads)"
                        ),
                    );
                }
            }
        }
    }
}
