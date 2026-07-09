//! `ScalarConnectedComponentImageFilter`: label pixels whose values chain
//! together within a distance threshold.
//!
//! Port of ITK `Modules/Segmentation/ConnectedComponents/include/`:
//! `itkScalarConnectedComponentImageFilter.h` is a thin instantiation of
//! `itkConnectedComponentFunctorImageFilter.hxx` with
//! `Functor::SimilarPixelsFunctor` as its join predicate: two pixels join
//! when `|a - b| <= DistanceThreshold` (`static_cast<TInput>` of the
//! absolute difference, i.e. rounded/truncated to the input pixel type the
//! same way [`crate::quantize_to_pixel_type`]'s other `pixeltype: Input`
//! callers are -- `DistanceThreshold` itself gets the same treatment before
//! comparison, matching `ScalarConnectedComponentImageFilter.yaml`'s
//! `pixeltype: Input`).
//!
//! **This is not [`crate::label::connected_component`]**: that filter joins
//! same-valued *foreground* runs and treats `0` as permanent background.
//! `ScalarConnectedComponentImageFilter` has no such notion -- *every*
//! non-masked pixel gets a label, including pixels valued `0`; a completely
//! uniform image (every pixel identical, even all-zero) becomes one single
//! component covering the whole image.
//!
//! ## The sweep
//!
//! `ConnectedComponentFunctorImageFilter::GenerateData()` makes one raster
//! pass, examining only each pixel's "previous" neighbors (already visited:
//! face-connected `-1` steps, or every earlier-in-neighborhood-order offset
//! when `FullyConnected`) -- [`Half::Previous`], the same connectivity half
//! [`crate::label::connected_component`] and [`crate::watershed`] use for
//! their own raster sweeps. A pixel adopts the label of (is unioned with)
//! every previous neighbor within threshold; a pixel with no such neighbor
//! starts a new component. An `EquivalencyTable` records merges made mid-scan
//! and flattens them in a second pass. This port collapses both passes into
//! a single union-find over pixel indices (join iff `|a - b| <=
//! DistanceThreshold`, restricted to [`Half::Previous`] neighbor pairs so
//! the total edge set searched matches upstream exactly) -- group-theoretically
//! identical to the two-pass eqTable scan, since both reduce to the same
//! union-find closure regardless of merge bookkeeping order.
//!
//! `EquivalencyTable`'s specific label numbering is explicitly documented
//! upstream as arbitrary ("The final object labels are in no particular
//! order... you can reorder the labels..."), so this port does not chase
//! ITK's own internal table-indexing quirks for the numeric label *values* --
//! it assigns contiguous labels `1..=N` in ascending raster order of first
//! appearance, exactly like [`crate::label::connected_component`]'s own
//! documented convention. Which pixels *share* a label matches upstream
//! exactly; the numeric label assigned to a given component may not.
//!
//! `MaskImage` (optional; `TMaskImage` is fixed to `uint8_t` in
//! `ScalarConnectedComponentImageFilter.yaml`'s `filter_type`) excludes a
//! pixel from labeling entirely when the mask value quantizes to `0`
//! (`MaskPixelType{}`): it is set to output `0` and is skipped both as a
//! sweep target and as a candidate neighbor, matching the boundary
//! condition's `ConstantBoundaryCondition<TOutputImage>` of `0` for
//! out-of-frame neighbors (never satisfies `neighborLabel != 0`, so this
//! port simply skips out-of-bounds neighbors, the same boundary-elision
//! pattern used throughout this crate's morphology ports).
//!
//! Output pixel type is fixed `uint32_t`
//! (`output_pixel_type: uint32_t`).

use crate::error::{FilterError, Result};
use crate::quantize_to_pixel_type;
use crate::reconstruction::{Half, NeighborWalker};
use sitk_core::{Image, PixelId};

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while cur != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// `ScalarConnectedComponentImageFilter`: labels pixels whose values chain
/// together within `distance_threshold` (`|a - b| <= distance_threshold`,
/// transitively -- see the module docs for why every non-masked pixel gets
/// a label, unlike [`crate::label::connected_component`]). `mask`, if
/// given, excludes pixels quantizing to `0` from labeling (they become
/// output `0`); it must match `image`'s size.
pub fn scalar_connected_component(
    image: &Image,
    mask: Option<&Image>,
    distance_threshold: f64,
    fully_connected: bool,
) -> Result<Image> {
    if let Some(m) = mask {
        if m.size() != image.size() {
            return Err(FilterError::SizeMismatch {
                a: image.size().to_vec(),
                b: m.size().to_vec(),
            });
        }
    }

    let size = image.size();
    let total: usize = size.iter().product();
    let vals = image.to_f64_vec();
    let id = image.pixel_id();
    let threshold = quantize_to_pixel_type(id, distance_threshold);

    let included: Vec<bool> = match mask {
        None => vec![true; total],
        Some(m) => m
            .to_f64_vec()
            .into_iter()
            .map(|v| quantize_to_pixel_type(PixelId::UInt8, v) != 0.0)
            .collect(),
    };

    let mut parent: Vec<usize> = (0..total).collect();
    let mut walker = NeighborWalker::new(size, fully_connected, Half::Previous);
    for pos in 0..total {
        if !included[pos] {
            continue;
        }
        for &neigh in walker.at(pos, size) {
            if !included[neigh] {
                continue;
            }
            let diff = quantize_to_pixel_type(id, (vals[pos] - vals[neigh]).abs());
            if diff <= threshold {
                union(&mut parent, pos, neigh);
            }
        }
    }

    let mut root_to_output: Vec<Option<u32>> = vec![None; total];
    let mut next_label = 1u32;
    let mut out = vec![0u32; total];
    for pos in 0..total {
        if !included[pos] {
            continue;
        }
        let root = find(&mut parent, pos);
        let label = *root_to_output[root].get_or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        out[pos] = label;
    }

    let mut out_image = Image::from_vec(size, out)?;
    out_image.copy_geometry_from(image);
    Ok(out_image)
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

    /// `diff == threshold` joins (the functor's `<=`); `diff == threshold +
    /// 1` does not.
    #[test]
    fn threshold_boundary_is_inclusive() {
        let joins = img_i32(&[2, 1], vec![10, 15]);
        let out = scalar_connected_component(&joins, None, 5.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1]);

        let splits = img_i32(&[2, 1], vec![10, 16]);
        let out = scalar_connected_component(&splits, None, 5.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 2]);
    }

    /// Similarity chains transitively through union-find: `0-5` and `5-10`
    /// each join at threshold 5, so all three end up in one component even
    /// though `|0 - 10| = 10` exceeds the threshold directly.
    #[test]
    fn similarity_chains_transitively() {
        let image = img_i32(&[3, 1], vec![0, 5, 10]);
        let out = scalar_connected_component(&image, None, 5.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 1]);
    }

    /// Every non-masked pixel gets a label, including an all-zero image --
    /// there is no background exclusion by value, unlike
    /// `crate::label::connected_component`.
    #[test]
    fn zero_valued_pixels_are_not_background() {
        let image = img_i32(&[2, 2], vec![0, 0, 0, 0]);
        let out = scalar_connected_component(&image, None, 0.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 1, 1]);
    }

    /// Two single-pixel components (threshold 0, no shared value) that
    /// touch only diagonally: face connectivity keeps them separate labels;
    /// full connectivity merges them into one.
    #[test]
    fn fully_connected_merges_diagonal_pixels() {
        #[rustfmt::skip]
        let image = img_i32(&[3, 3], vec![
            5, 9, 9,
            9, 5, 9,
            9, 9, 9,
        ]);

        let face = scalar_connected_component(&image, None, 0.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u32>().unwrap(), &[
            1, 2, 2,
            2, 3, 2,
            2, 2, 2,
        ]);

        let full = scalar_connected_component(&image, None, 0.0, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.scalar_slice::<u32>().unwrap(), &[
            1, 2, 2,
            2, 1, 2,
            2, 2, 2,
        ]);
    }

    /// A masked-out pixel (mask value `0`) always outputs `0`, regardless
    /// of how similar its value is to its neighbors, and cannot bridge two
    /// otherwise-separate similar regions.
    #[test]
    fn mask_excludes_pixels_from_labeling() {
        let image = img_i32(&[3, 1], vec![5, 5, 5]);
        let mask = img_u8(&[3, 1], vec![1, 0, 1]);
        let out = scalar_connected_component(&image, Some(&mask), 0.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 0, 2]);
    }

    /// A mask whose size does not match the image is a size-mismatch error.
    #[test]
    fn mask_size_mismatch_is_an_error() {
        let image = img_i32(&[3, 1], vec![1, 2, 3]);
        let mask = img_u8(&[2, 1], vec![1, 1]);
        assert_eq!(
            scalar_connected_component(&image, Some(&mask), 0.0, false),
            Err(FilterError::SizeMismatch {
                a: vec![3, 1],
                b: vec![2, 1],
            })
        );
    }

    /// Labels are contiguous `1..=N` in ascending raster order of first
    /// appearance (this port's documented convention -- see the module
    /// docs for why upstream's own numbering is not chased).
    #[test]
    fn labels_are_contiguous_by_raster_order_of_first_appearance() {
        let image = img_i32(&[4, 1], vec![100, 100, 0, 0]);
        let out = scalar_connected_component(&image, None, 0.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 2, 2]);
    }
}
