//! Morphological watershed segmentation.
//!
//! Ports of:
//!
//! - `itk::MorphologicalWatershedFromMarkersImageFilter`
//!   (`itkMorphologicalWatershedFromMarkersImageFilter.h` / `.hxx`).
//! - `itk::MorphologicalWatershedImageFilter`
//!   (`itkMorphologicalWatershedImageFilter.h` / `.hxx`), whose mini-pipeline
//!   is `HMinimaImageFilter` (only when `level != 0`) →
//!   `RegionalMinimaImageFilter` → `ConnectedComponentImageFilter` →
//!   `MorphologicalWatershedFromMarkersImageFilter`.
//!
//! The mini-pipeline's `HMinimaImageFilter` stage —
//! [`crate::reconstruction::h_minima`], built on
//! [`crate::reconstruction::reconstruction_by_erosion`]
//! (`itkReconstructionImageFilter.hxx` with `TCompare = std::less`) — is a
//! public filter in its own right, ported fully in [`crate::reconstruction`].
//! Only [`regional_minima`] is ported here as a module-private helper,
//! because only the watershed needs it so far:
//!
//! - [`regional_minima`] — `itkValuedRegionalMinimaImageFilter.h` driving
//!   `itkValuedRegionalExtremaImageFilter.hxx`, thresholded as
//!   `itkRegionalMinimaImageFilter.hxx` does (`FlatIsMinima` defaults to
//!   `true`, so a completely flat image is one big minimum).
//!
//! The connected-component labeling of the minima reuses the crate's
//! [`crate::connected_component`] — it is the same
//! `ConnectedComponentImageFilter`, takes the same `fully_connected` flag,
//! treats 0 as background, and numbers objects `1..=N` in raster order of
//! first appearance, which is exactly what the `.hxx` pipeline feeds to the
//! from-markers filter.
//!
//! ## Flooding
//!
//! `MorphologicalWatershedFromMarkersImageFilter::GenerateData` holds two
//! algorithms behind `mark_watershed_line`: Meyer's (watershed lines marked)
//! and Beucher's (no lines). Both flood from the markers through a
//! hierarchical FIFO queue (`std::map<InputPixelType, std::queue<Index>>`)
//! keyed on the input gray level, always draining the smallest key first.
//!
//! Because a pixel is only ever enqueued at a gray level strictly above the
//! level currently being drained (a lower or equal level goes onto the
//! *current* queue instead), the map's smallest key never decreases. This port
//! therefore replaces the `std::map` with a bucket per distinct input value
//! ([`value_ranks`] maps each pixel's value onto its rank in the sorted set
//! of distinct values) and drains the buckets in ascending rank — the same
//! sequence of pops in the same order, without the per-pop `O(log n)`.
//!
//! Both the background label of the marker image and the watershed-line
//! label of the output are `LabelImagePixelType{}`, i.e. **0**.
//!
//! ## Boundary handling
//!
//! Every shaped neighborhood iterator in the `.hxx` is given a
//! `ConstantBoundaryCondition` chosen so that out-of-bounds neighbors are
//! inert:
//!
//! - marker iterator ← `NumericTraits<Label>::max()`, so an outside pixel is
//!   never `bgLabel` and is never enqueued;
//! - status iterator ← `true`, so an outside pixel is never enqueued;
//! - output iterator ← `wsLabel` (Meyer) so it contributes no label to the
//!   collision test, or `NumericTraits<Label>::max()` (Beucher) so it is
//!   never `wsLabel` and is never written.
//!
//! In every one of those roles the outside pixel is skipped, so this port
//! simply skips out-of-bounds neighbors. The same reasoning collapses the
//! `ConstantPadImageFilter`-by-`m_MarkerValue` padding in
//! `itkReconstructionImageFilter.hxx`: its padded border holds
//! `NumericTraits<T>::max()` in *both* the marker and the mask copy, so
//! every one of the three passes' guards (`VN < V`, `V < VN && iN < VN`,
//! `V < VN && iN != VN`) is false at the border, and the border is never
//! written. `regional_minima` likewise uses `max()` on both its input and
//! output iterators, and both of its guards (`Adjacent < Cent`,
//! `NVal == V`) are false there.
//!
//! ## Neighbor order
//!
//! Meyer's flooding is order-dependent: which basin claims a pixel that two
//! basins reach on the same gray level depends on the FIFO insertion order,
//! which depends on the order neighbors are visited. `ShapedNeighborhoodIterator`
//! visits its active offsets in ascending *neighborhood index* — the offset
//! `o ∈ {-1,0,1}^dim` at index `Σ_d (o[d] + 1) · 3^d`, axis 0 fastest
//! (`ConstShapedNeighborhoodIterator::ActivateIndex` keeps the active list
//! sorted). `crate::reconstruction`'s private `neighbor_offsets` reproduces
//! that order, and `itkConnectedComponentAlgorithm.h`'s `setConnectivity` /
//! `setConnectivityPrevious` / `setConnectivityLater` become its three
//! [`crate::reconstruction::Half`] variants, reused here via
//! [`crate::reconstruction::NeighborWalker`].

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::label::connected_component;
use crate::reconstruction::{Half, NeighborWalker};
use sitk_core::Image;
use std::collections::VecDeque;

/// The marker image's background label and the output image's watershed-line
/// label — both `LabelImagePixelType{}` in the `.hxx`.
const BG_LABEL: i64 = 0;
/// See [`BG_LABEL`]; named separately to mirror the `.hxx`'s `wsLabel`.
const WS_LABEL: i64 = 0;

// ---- hierarchical queue keys ---------------------------------------------

/// Map every pixel value onto its rank in the sorted set of distinct values,
/// returning the ranks and the number of distinct values.
///
/// The from-markers flooding only ever compares gray levels and uses them as
/// `std::map` keys, so replacing each value by its rank leaves every decision
/// unchanged while turning the map into a bucket array (see the module docs).
/// `f64::total_cmp` gives a total order even in the presence of `NaN`.
fn value_ranks(vals: &[f64]) -> (Vec<u32>, usize) {
    let mut distinct: Vec<f64> = vals.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let ranks = vals
        .iter()
        .map(|v| {
            distinct
                .binary_search_by(|probe| probe.total_cmp(v))
                .expect("every value is present in the distinct set") as u32
        })
        .collect();
    (ranks, distinct.len())
}

// ---- morphological_watershed_from_markers --------------------------------

/// `MorphologicalWatershedFromMarkersImageFilter`: flood `image`'s gray
/// levels outward from the labeled regions of `marker_image`, so every pixel
/// joins the catchment basin of the marker it is reachable from along the
/// lowest ascending path.
///
/// `marker_image` must have `image`'s size; its label 0 is background (no
/// marker). With `mark_watershed_line`, pixels equidistant between two
/// basins are left at 0 — Meyer's algorithm; without it, Beucher's algorithm
/// assigns every pixel to a basin. `fully_connected` selects face
/// connectivity (`false`, the default) or face+edge+vertex connectivity.
///
/// The output has `marker_image`'s pixel type, matching SimpleITK's
/// `MorphologicalWatershedFromMarkersImageFilter.yaml`
/// (`itk::MorphologicalWatershedFromMarkersImageFilter<InputImageType, InputImageType2>`).
pub fn morphological_watershed_from_markers(
    image: &Image,
    marker_image: &Image,
    mark_watershed_line: bool,
    fully_connected: bool,
) -> Result<Image> {
    // "Marker and input must have the same size." — the `.hxx` compares only
    // sizes, not pixel types, which differ by design here.
    if image.size() != marker_image.size() {
        return Err(FilterError::SizeMismatch {
            a: image.size().to_vec(),
            b: marker_image.size().to_vec(),
        });
    }

    let size = image.size();
    let total = image.number_of_pixels();
    let markers: Vec<i64> = marker_image
        .to_f64_vec()
        .iter()
        .map(|&v| v.round() as i64)
        .collect();

    let out = if total == 0 {
        Vec::new()
    } else {
        let (ranks, num_ranks) = value_ranks(&image.to_f64_vec());
        if mark_watershed_line {
            flood_meyer(size, &ranks, num_ranks, &markers, fully_connected)
        } else {
            flood_beucher(size, &ranks, num_ranks, &markers, fully_connected)
        }
    };

    let vals: Vec<f64> = out.iter().map(|&l| l as f64).collect();
    image_from_f64(marker_image.pixel_id(), size, image, &vals)
}

/// Meyer's algorithm (`m_MarkWatershedLine == true`): a pixel whose already
/// labeled neighbors disagree is a watershed line and stays [`WS_LABEL`],
/// and — being already marked "processed" — never propagates.
fn flood_meyer(
    size: &[usize],
    ranks: &[u32],
    num_ranks: usize,
    markers: &[i64],
    fully_connected: bool,
) -> Vec<i64> {
    let total = markers.len();
    let mut out = vec![WS_LABEL; total];
    // The `.hxx`'s status image: "already processed, or already in the FAH".
    let mut status = vec![false; total];
    let mut fah: Vec<VecDeque<usize>> = vec![VecDeque::new(); num_ranks];
    let mut walker = NeighborWalker::new(size, fully_connected, Half::Full);

    // First stage: markers are processed and copied to the output; their
    // background neighbors seed the hierarchical queue at their own gray level.
    for f in 0..total {
        let marker = markers[f];
        if marker == BG_LABEL {
            continue;
        }
        status[f] = true;
        out[f] = marker;
        for &g in walker.at(f, size) {
            if !status[g] && markers[g] == BG_LABEL {
                fah[ranks[g] as usize].push_back(g);
                status[g] = true;
            }
        }
    }

    // Flooding. A neighbor at a gray level `<= current` joins the current
    // queue; a higher one waits in its own bucket. Nothing is ever added to a
    // bucket at or below `rank`, so draining buckets in ascending rank
    // reproduces `while (!fah.empty()) { ... fah.erase(fah.begin()); }`.
    for rank in 0..num_ranks {
        let mut queue = std::mem::take(&mut fah[rank]);
        while let Some(f) = queue.pop_front() {
            let mut marker = WS_LABEL;
            let mut collision = false;
            for &g in walker.at(f, size) {
                let o = out[g];
                if o != WS_LABEL {
                    if marker != WS_LABEL && o != marker {
                        collision = true;
                        break;
                    }
                    marker = o;
                }
            }
            if collision {
                continue;
            }
            out[f] = marker;
            for &g in walker.at(f, size) {
                if status[g] {
                    continue;
                }
                let g_rank = ranks[g] as usize;
                if g_rank <= rank {
                    queue.push_back(g);
                } else {
                    fah[g_rank].push_back(g);
                }
                status[g] = true;
            }
        }
    }
    out
}

/// Beucher's algorithm (`m_MarkWatershedLine == false`): the first basin to
/// reach an unlabeled pixel claims it, so no watershed lines are produced.
fn flood_beucher(
    size: &[usize],
    ranks: &[u32],
    num_ranks: usize,
    markers: &[i64],
    fully_connected: bool,
) -> Vec<i64> {
    let total = markers.len();
    let mut out = vec![WS_LABEL; total];
    let mut fah: Vec<VecDeque<usize>> = vec![VecDeque::new(); num_ranks];
    let mut walker = NeighborWalker::new(size, fully_connected, Half::Full);

    // First stage: copy the markers to the output, and seed the queue with the
    // marker pixels that touch background — the basins' outer rims. Here the
    // seed's own gray level is the queue key, not the neighbor's.
    for f in 0..total {
        let marker = markers[f];
        if marker == BG_LABEL {
            continue;
        }
        out[f] = marker;
        let has_bg_neighbor = walker.at(f, size).iter().any(|&g| markers[g] == BG_LABEL);
        if has_bg_neighbor {
            fah[ranks[f] as usize].push_back(f);
        }
    }

    for rank in 0..num_ranks {
        let mut queue = std::mem::take(&mut fah[rank]);
        while let Some(f) = queue.pop_front() {
            let current_marker = out[f];
            for &g in walker.at(f, size) {
                if out[g] != WS_LABEL {
                    continue;
                }
                out[g] = current_marker;
                let g_rank = ranks[g] as usize;
                if g_rank <= rank {
                    queue.push_back(g);
                } else {
                    fah[g_rank].push_back(g);
                }
            }
        }
    }
    out
}

// ---- regional minima -------------------------------------------------------

/// `RegionalMinimaImageFilter` (via `ValuedRegionalMinimaImageFilter`): mark
/// every pixel belonging to a regional minimum — a flat zone all of whose
/// neighbors are strictly greater.
///
/// A completely flat image is one big regional minimum, matching
/// `m_FlatIsMinima`'s `true` default.
///
/// `itkValuedRegionalExtremaImageFilter.hxx` records "not a regional minimum"
/// by overwriting the output pixel with `NumericTraits<T>::max()`, and then
/// skips any pixel already holding that value. This port keeps that flag in a
/// separate `marked` buffer instead of aliasing it onto a pixel value, so a
/// real pixel of value `T::max()` cannot be confused for the sentinel.
fn regional_minima(vals: &[f64], size: &[usize], fully_connected: bool) -> Vec<bool> {
    let total = vals.len();
    if total == 0 {
        return Vec::new();
    }
    if vals.iter().all(|&v| v == vals[0]) {
        return vec![true; total];
    }

    let mut walker = NeighborWalker::new(size, fully_connected, Half::Full);
    // `marked[f]`: `f` is known not to belong to a regional minimum.
    let mut marked = vec![false; total];
    let mut stack: Vec<usize> = Vec::new();

    for f in 0..total {
        if marked[f] {
            continue;
        }
        let v = vals[f];
        let has_lower_neighbor = walker.at(f, size).iter().any(|&g| vals[g] < v);
        if !has_lower_neighbor {
            continue;
        }
        // Flood the flat zone of value `v` containing `f`: none of it can be a
        // regional minimum.
        marked[f] = true;
        stack.push(f);
        while let Some(i) = stack.pop() {
            for &g in walker.at(i, size) {
                if !marked[g] && vals[g] == v {
                    marked[g] = true;
                    stack.push(g);
                }
            }
        }
    }

    marked.iter().map(|&m| !m).collect()
}

// ---- morphological_watershed ---------------------------------------------

/// `MorphologicalWatershedImageFilter`: segment `image` into the catchment
/// basins of its regional minima.
///
/// `level` (in input-pixel-type units) merges every basin shallower than it,
/// via an h-minima pre-step; `level == 0` skips that step entirely, as the
/// `.hxx` does. `mark_watershed_line` and `fully_connected` are passed
/// through to [`morphological_watershed_from_markers`], which floods the
/// **original** `image` — the h-minima result only ever feeds the regional
/// minima extraction.
///
/// Output pixel type is `UInt32` (`output_pixel_type: uint32_t` in
/// SimpleITK's `MorphologicalWatershedImageFilter.yaml`), inherited from the
/// connected-component labeling of the minima.
pub fn morphological_watershed(
    image: &Image,
    level: f64,
    mark_watershed_line: bool,
    fully_connected: bool,
) -> Result<Image> {
    let height = crate::quantize_to_pixel_type(image.pixel_id(), level);
    if height < 0.0 {
        return Err(FilterError::InvalidWatershedLevel(level));
    }

    let size = image.size();
    let vals = image.to_f64_vec();
    let minima_source = if height != 0.0 {
        crate::reconstruction::h_minima(image, height, fully_connected)?.to_f64_vec()
    } else {
        vals
    };

    let minima = regional_minima(&minima_source, size, fully_connected);
    let mask: Vec<u8> = minima.iter().map(|&m| u8::from(m)).collect();
    let mut mask_image = Image::from_vec(size, mask)?;
    mask_image.copy_geometry_from(image);

    let markers = connected_component(&mask_image, fully_connected)?;
    morphological_watershed_from_markers(image, &markers, mark_watershed_line, fully_connected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn labels(img: &Image) -> Vec<u32> {
        img.scalar_slice::<u32>().unwrap().to_vec()
    }

    // ---- morphological_watershed_from_markers ----

    /// Two basins of a 1-D ramp, markers at their minima: the watershed line
    /// must land exactly on the ridge pixel (index 4, the only local maximum
    /// between them).
    #[test]
    fn from_markers_1d_ramp_puts_line_on_the_ridge() {
        let image = img_u8(&[9, 1], vec![3, 2, 1, 2, 3, 2, 1, 2, 3]);
        let markers = img_u8(&[9, 1], vec![0, 0, 1, 0, 0, 0, 2, 0, 0]);
        let out = morphological_watershed_from_markers(&image, &markers, true, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 1, 0, 2, 2, 2, 2]
        );
    }

    /// Same input with `mark_watershed_line = false` (Beucher): the ridge
    /// pixel is claimed by whichever basin reaches it first instead of being
    /// left at 0. Every pixel is labeled.
    #[test]
    fn mark_watershed_line_off_labels_the_ridge() {
        let image = img_u8(&[9, 1], vec![3, 2, 1, 2, 3, 2, 1, 2, 3]);
        let markers = img_u8(&[9, 1], vec![0, 0, 1, 0, 0, 0, 2, 0, 0]);
        let out = morphological_watershed_from_markers(&image, &markers, false, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 1, 1, 2, 2, 2, 2]
        );
        assert!(out.scalar_slice::<u8>().unwrap().iter().all(|&v| v != 0));
    }

    /// A 3x3 image whose anti-diagonal is a high ridge. Under face
    /// connectivity the ridge separates the two corners' basins and becomes
    /// the watershed line; under full connectivity the low regions touch
    /// through the ridge's diagonal gaps, and Meyer's flooding resolves the
    /// contact differently.
    #[test]
    fn fully_connected_changes_a_diagonal_ridge() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            0, 0, 5,
            0, 5, 0,
            5, 0, 0,
        ]);
        #[rustfmt::skip]
        let markers = img_u8(&[3, 3], vec![
            1, 0, 0,
            0, 0, 0,
            0, 0, 2,
        ]);

        let face = morphological_watershed_from_markers(&image, &markers, true, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u8>().unwrap(), &[
            1, 1, 0,
            1, 0, 2,
            0, 2, 2,
        ]);

        let full = morphological_watershed_from_markers(&image, &markers, true, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.scalar_slice::<u8>().unwrap(), &[
            1, 1, 1,
            1, 0, 0,
            1, 0, 2,
        ]);
        assert_ne!(
            face.scalar_slice::<u8>().unwrap(),
            full.scalar_slice::<u8>().unwrap()
        );
    }

    /// An explicit marker layout with non-consecutive labels, on a 2-D image
    /// whose rows are identical ramps: the labels are carried through
    /// verbatim and each column of the ridge becomes watershed line.
    #[test]
    fn from_markers_carries_explicit_label_values() {
        #[rustfmt::skip]
        let image = img_u8(&[7, 2], vec![
            2, 1, 2, 3, 2, 1, 2,
            2, 1, 2, 3, 2, 1, 2,
        ]);
        #[rustfmt::skip]
        let markers = Image::from_vec(&[7, 2], vec![
            0u16, 5, 0, 0, 0, 7, 0,
            0,    5, 0, 0, 0, 7, 0,
        ]).unwrap();
        let out = morphological_watershed_from_markers(&image, &markers, true, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt16);
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u16>().unwrap(), &[
            5, 5, 5, 0, 7, 7, 7,
            5, 5, 5, 0, 7, 7, 7,
        ]);
    }

    #[test]
    fn from_markers_rejects_mismatched_sizes() {
        let image = img_u8(&[4, 1], vec![1, 2, 3, 4]);
        let markers = img_u8(&[3, 1], vec![1, 0, 2]);
        assert!(matches!(
            morphological_watershed_from_markers(&image, &markers, true, false),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    // ---- morphological_watershed ----

    /// A flat ridge (plateau) between two basins: the regional-minima step
    /// finds exactly the two end pixels, and the watershed line lands on the
    /// middle of the plateau rather than at one of its edges.
    #[test]
    fn plateau_ridge_splits_in_the_middle() {
        let image = img_u8(&[5, 1], vec![1, 2, 2, 2, 1]);
        let out = morphological_watershed(&image, 0.0, true, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt32);
        assert_eq!(labels(&out), vec![1, 1, 0, 2, 2]);
    }

    /// `level` merges basins shallower than it. The middle minimum of
    /// `[0, 3, 1, 3, 0]` has depth 2, so it survives `level = 1` and is
    /// merged away at `level = 2`.
    #[test]
    fn level_merges_basins_shallower_than_it() {
        let image = img_u8(&[5, 1], vec![0, 3, 1, 3, 0]);

        let none = morphological_watershed(&image, 0.0, true, false).unwrap();
        assert_eq!(labels(&none), vec![1, 0, 2, 0, 3]);

        let below = morphological_watershed(&image, 1.0, true, false).unwrap();
        assert_eq!(labels(&below), vec![1, 0, 2, 0, 3]);

        let merged = morphological_watershed(&image, 2.0, true, false).unwrap();
        assert_eq!(labels(&merged), vec![1, 1, 0, 2, 2]);
    }

    /// `level` is `static_cast<InputImagePixelType>`-ed before the `!= 0`
    /// test, so a sub-unit level on an integer image is exactly `level = 0`.
    #[test]
    fn fractional_level_truncates_to_the_input_pixel_type() {
        let image = img_u8(&[5, 1], vec![0, 3, 1, 3, 0]);
        let truncated = morphological_watershed(&image, 0.75, true, false).unwrap();
        assert_eq!(labels(&truncated), vec![1, 0, 2, 0, 3]);

        // The same 0.75 on a float image does shift the marker, but 0.75 < 2
        // still leaves the depth-2 minimum standing.
        let float = Image::from_vec(&[5, 1], vec![0.0f32, 3.0, 1.0, 3.0, 0.0]).unwrap();
        assert_eq!(
            labels(&morphological_watershed(&float, 0.75, true, false).unwrap()),
            vec![1, 0, 2, 0, 3]
        );
        assert_eq!(
            labels(&morphological_watershed(&float, 2.5, true, false).unwrap()),
            vec![1, 1, 0, 2, 2]
        );
    }

    /// `RegionalMinimaImageFilter`'s `FlatIsMinima` defaults to `true`: a
    /// constant image is a single regional minimum, hence a single basin.
    #[test]
    fn flat_image_is_one_basin() {
        let image = img_u8(&[3, 3], vec![7; 9]);
        let out = morphological_watershed(&image, 0.0, true, false).unwrap();
        assert_eq!(labels(&out), vec![1; 9]);
    }

    /// Without watershed lines every pixel of a two-basin ramp is labeled.
    #[test]
    fn watershed_without_lines_leaves_no_background() {
        let image = img_u8(&[9, 1], vec![3, 2, 1, 2, 3, 2, 1, 2, 3]);
        let lined = morphological_watershed(&image, 0.0, true, false).unwrap();
        assert_eq!(labels(&lined), vec![1, 1, 1, 1, 0, 2, 2, 2, 2]);

        let solid = morphological_watershed(&image, 0.0, false, false).unwrap();
        assert_eq!(labels(&solid), vec![1, 1, 1, 1, 1, 2, 2, 2, 2]);
    }

    /// Diagonal connectivity changes which minima are distinct: two pixels
    /// touching only at a corner are one minimum under full connectivity and
    /// two under face connectivity.
    #[test]
    fn fully_connected_merges_diagonally_touching_minima() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            0, 9, 9,
            9, 0, 9,
            9, 9, 9,
        ]);
        let face = morphological_watershed(&image, 0.0, true, false).unwrap();
        let face_labels = labels(&face);
        assert_ne!(face_labels[0], face_labels[4]);

        let full = morphological_watershed(&image, 0.0, true, true).unwrap();
        let full_labels = labels(&full);
        assert_eq!(full_labels[0], full_labels[4]);
        assert!(full_labels.iter().all(|&v| v == 1));
    }

    #[test]
    fn negative_level_is_rejected() {
        let image = Image::from_vec(&[3, 1], vec![0i16, 5, 0]).unwrap();
        assert!(matches!(
            morphological_watershed(&image, -1.0, true, false),
            Err(FilterError::InvalidWatershedLevel(_))
        ));
    }

    // ---- regional_minima ----

    #[test]
    fn regional_minima_flood_covers_whole_plateau() {
        // The flat zone {1,2,3} has a lower neighbor at either end, so none of
        // it is a minimum; the two end pixels are.
        let vals = [1.0, 2.0, 2.0, 2.0, 1.0];
        assert_eq!(
            regional_minima(&vals, &[5, 1], false),
            vec![true, false, false, false, true]
        );
    }

    #[test]
    fn regional_minima_of_a_flat_image_is_everything() {
        assert_eq!(regional_minima(&[4.0; 4], &[4, 1], false), vec![true; 4]);
    }

    #[test]
    fn value_ranks_order_distinct_values() {
        let (ranks, n) = value_ranks(&[3.0, 1.0, 2.0, 1.0]);
        assert_eq!(n, 3);
        assert_eq!(ranks, vec![2, 0, 1, 0]);
    }
}
