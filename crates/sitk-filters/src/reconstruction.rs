//! Grayscale geodesic morphological reconstruction and the filters built on
//! it.
//!
//! Ports of (ITK `Modules/Filtering/MathematicalMorphology/include/`):
//!
//! - [`reconstruction_by_erosion`] / [`reconstruction_by_dilation`] —
//!   `itkReconstructionImageFilter.hxx`, Luc Vincent's raster / anti-raster /
//!   FIFO three-pass algorithm, generalized here over [`ReconstructionKind`]
//!   exactly as the `.hxx` is generalized over its `TCompare` template
//!   parameter: `itkReconstructionByErosionImageFilter.h` fixes
//!   `TCompare = std::less`, `itkReconstructionByDilationImageFilter.h` fixes
//!   `TCompare = std::greater`. Both are thin instantiations; neither adds any
//!   logic of its own beyond the comparator and the `MarkerValue` used for
//!   boundary padding — see [`reconstruct`]'s docs for why this port needs
//!   neither.
//! - [`grayscale_fillhole`] / [`grayscale_grindpeak`] —
//!   `itkGrayscaleFillholeImageFilter.hxx` / `itkGrayscaleGrindPeakImageFilter.hxx`:
//!   build a marker from the input's own border and its global max (fillhole)
//!   or min (grindpeak), then reconstruct by erosion/dilation under the input
//!   as the mask.
//! - [`h_minima`] / [`h_maxima`] — `itkHMinimaImageFilter.hxx` /
//!   `itkHMaximaImageFilter.hxx`: reconstruct by erosion/dilation under a
//!   marker of `input ± height`, suppressing every regional extremum shallower
//!   than `height`.
//! - [`h_convex`] / [`h_concave`] — `itkHConvexImageFilter.hxx` /
//!   `itkHConcaveImageFilter.hxx`: `input - h_maxima(input)` /
//!   `h_minima(input) - input`, extracting exactly the extrema `h_maxima` /
//!   `h_minima` suppressed.
//!
//! ## The `reconstruct` engine
//!
//! [`reconstruct`] is [`crate::watershed`]'s former private
//! `reconstruction_by_erosion` (`TCompare = std::less`), generalized to
//! [`ReconstructionKind`] so it also serves `TCompare = std::greater`. The
//! generalization is mechanical: every comparison in the `.hxx`
//! (`compare(VN, V)` picking a neighbor, `compare(V, iV)` clamping to the
//! mask, `compare(V, VN) && compare(iN, VN)` seeding the FIFO,
//! `compare(V, VN) && iN != VN` / `compare(iN, V)` driving the FIFO) is
//! literally `TCompare::operator()`, so substituting [`ReconstructionKind::compare`]
//! for each one reproduces both instantiations from one function body.
//!
//! The `.hxx`'s `UseInternalCopy` (default on) pads `marker`/`mask` by one
//! pixel with `m_MarkerValue` (`NumericTraits<T>::max()` for erosion,
//! `NonpositiveMin()` for dilation) purely to let a single
//! `ShapedNeighborhoodIterator` pass cover the whole image without per-pixel
//! bounds checks; the `false` branch instead gives the same iterators a
//! `ConstantBoundaryCondition` of the same `m_MarkerValue`, which is exactly
//! equivalent (`GenerateData` sets that boundary condition on `outNIt`
//! regardless of the branch). Either way, the padded/boundary value is
//! identical in **both** the marker/output iterator and the mask iterator, so
//! every guard above is false there: `compare(VN, V)` never fires because the
//! boundary already equals the type's extreme in the reconstruction's own
//! direction; the FIFO seed guard needs `compare(iN, VN)` where `iN == VN` at
//! the boundary; the FIFO driver needs `iN != VN` where again `iN == VN`. So a
//! boundary neighbor never wins a comparison and never gets written — this
//! port simply skips out-of-bounds neighbors instead of materializing any
//! padding, for either [`ReconstructionKind`].
//!
//! ## Neighbor order and connectivity
//!
//! [`NeighborWalker`], [`Half`], and the private `Connectivity`/`neighbor_offsets`/
//! `strides`/`multi_index` helpers below are shared with
//! [`crate::watershed`]'s own flooding and regional-minima code, which needs
//! the identical `ShapedNeighborhoodIterator` active-offset order (ascending
//! neighborhood index, ties resolved by `itkConnectedComponentAlgorithm.h`'s
//! `setConnectivity`/`setConnectivityPrevious`/`setConnectivityLater` split —
//! [`Half`]'s three variants) for its own order-dependent flooding. This is
//! the shared home for that geometry so neither module carries its own copy.

use crate::error::{FilterError, Result};
use crate::geometry::require_same_physical_space;
use crate::{image_from_f64, quantize_to_pixel_type, require_same_shape};
use sitk_core::Image;
use std::collections::VecDeque;

// ---- N-D indexing and connectivity (shared with `crate::watershed`) ------

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// Decode a flat (first-index-fastest) offset into a multi-index.
fn multi_index(flat: usize, size: &[usize], strides: &[usize], out: &mut Vec<usize>) {
    out.clear();
    out.extend((0..size.len()).map(|d| (flat / strides[d]) % size[d]));
}

/// Which side of the center a connectivity's offsets lie on, matching the
/// three activation helpers in `itkConnectedComponentAlgorithm.h`.
#[derive(Clone, Copy)]
pub(crate) enum Half {
    /// `setConnectivity`: every neighbor.
    Full,
    /// `setConnectivityPrevious`: neighbors earlier in raster order.
    Previous,
    /// `setConnectivityLater`: neighbors later in raster order.
    Later,
}

/// The active offsets of a shaped neighborhood iterator, in ascending
/// neighborhood index (see the module docs). `fully_connected = false` keeps
/// only the axis-aligned unit steps (face connectivity); `true` keeps every
/// nonzero offset in `{-1,0,1}^dim`.
fn neighbor_offsets(dim: usize, fully_connected: bool, half: Half) -> Vec<Vec<i64>> {
    let total = 3usize.pow(dim as u32);
    let center = total / 2;
    let mut offsets = Vec::new();
    for code in 0..total {
        let mut rem = code;
        let mut offset = vec![0i64; dim];
        let mut nonzero_axes = 0usize;
        for d in offset.iter_mut() {
            let digit = rem % 3;
            rem /= 3;
            *d = digit as i64 - 1; // 0,1,2 -> -1,0,1
            if *d != 0 {
                nonzero_axes += 1;
            }
        }
        if nonzero_axes == 0 || (!fully_connected && nonzero_axes > 1) {
            continue;
        }
        let keep = match half {
            Half::Full => true,
            Half::Previous => code < center,
            Half::Later => code > center,
        };
        if keep {
            offsets.push(offset);
        }
    }
    offsets
}

/// A shaped neighborhood: the active offsets plus their flat-index deltas.
struct Connectivity {
    offsets: Vec<Vec<i64>>,
    deltas: Vec<i64>,
}

impl Connectivity {
    fn new(size: &[usize], fully_connected: bool, half: Half) -> Self {
        let st = strides(size);
        let offsets = neighbor_offsets(size.len(), fully_connected, half);
        let deltas = offsets
            .iter()
            .map(|o| o.iter().zip(&st).map(|(&a, &s)| a * s as i64).sum())
            .collect();
        Connectivity { offsets, deltas }
    }

    /// Append the in-bounds neighbors of the pixel at `flat` (multi-index
    /// `idx`) to `out`, in ascending neighborhood index. Out-of-bounds
    /// neighbors are skipped — see the module docs on boundary handling.
    fn collect(&self, idx: &[usize], flat: usize, size: &[usize], out: &mut Vec<usize>) {
        out.clear();
        'offset: for (offset, &delta) in self.offsets.iter().zip(&self.deltas) {
            for (d, &o) in offset.iter().enumerate() {
                let v = idx[d] as i64 + o;
                if v < 0 || v as usize >= size[d] {
                    continue 'offset;
                }
            }
            out.push((flat as i64 + delta) as usize);
        }
    }
}

/// Walks the in-bounds neighbors of successive pixels, reusing one scratch
/// buffer per role so the flooding/reconstruction loops allocate nothing per
/// pixel.
pub(crate) struct NeighborWalker {
    conn: Connectivity,
    idx: Vec<usize>,
    neighbors: Vec<usize>,
    strides: Vec<usize>,
}

impl NeighborWalker {
    pub(crate) fn new(size: &[usize], fully_connected: bool, half: Half) -> Self {
        NeighborWalker {
            conn: Connectivity::new(size, fully_connected, half),
            idx: Vec::with_capacity(size.len()),
            neighbors: Vec::with_capacity(3usize.pow(size.len() as u32)),
            strides: strides(size),
        }
    }

    /// The in-bounds neighbors of `flat`, valid until the next call.
    pub(crate) fn at(&mut self, flat: usize, size: &[usize]) -> &[usize] {
        multi_index(flat, size, &self.strides, &mut self.idx);
        self.conn
            .collect(&self.idx, flat, size, &mut self.neighbors);
        &self.neighbors
    }
}

// ---- the reconstruction engine --------------------------------------------

/// `TCompare` in `itkReconstructionImageFilter.hxx`: `std::less` for
/// `ReconstructionByErosionImageFilter`, `std::greater` for
/// `ReconstructionByDilationImageFilter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReconstructionKind {
    Erosion,
    Dilation,
}

impl ReconstructionKind {
    /// `TCompare::operator()(a, b)`. Also literally the `.hxx`'s per-pixel
    /// precondition check (`if (compare(V, iV)) { throw }`): `compare(marker,
    /// mask)` is true exactly when the marker violates its required relation
    /// to the mask.
    fn compare(self, a: f64, b: f64) -> bool {
        match self {
            ReconstructionKind::Erosion => a < b,
            ReconstructionKind::Dilation => a > b,
        }
    }

    /// The relation the marker must satisfy against the mask everywhere,
    /// reported in [`FilterError::InvalidReconstructionMarker`].
    fn required_relation(self) -> &'static str {
        match self {
            ReconstructionKind::Erosion => ">=",
            ReconstructionKind::Dilation => "<=",
        }
    }
}

/// `ReconstructionImageFilter<..., TCompare>::GenerateData`: three passes over
/// `marker` (in place) — a raster scan taking the best (min for erosion, max
/// for dilation) of the earlier neighbors then clamping to `mask`; an
/// anti-raster scan doing the same over the later neighbors and queueing
/// every pixel that can still improve a later neighbor; and a FIFO
/// propagation over the full neighborhood until the queue drains.
///
/// Callers must guarantee `kind.compare(marker[i], mask[i])` is false for
/// every `i` (the `.hxx`'s stated precondition); the public entry points
/// below check this and return [`FilterError::InvalidReconstructionMarker`]
/// rather than reproducing the `.hxx`'s `itkExceptionMacro`.
fn reconstruct(
    marker: &[f64],
    mask: &[f64],
    size: &[usize],
    fully_connected: bool,
    kind: ReconstructionKind,
) -> Vec<f64> {
    let total = marker.len();
    let mut out = marker.to_vec();
    let mut previous = NeighborWalker::new(size, fully_connected, Half::Previous);
    let mut later = NeighborWalker::new(size, fully_connected, Half::Later);
    let mut full = NeighborWalker::new(size, fully_connected, Half::Full);

    // Raster scan.
    for f in 0..total {
        let mut v = out[f];
        for &g in previous.at(f, size) {
            let vn = out[g];
            if kind.compare(vn, v) {
                v = vn;
            }
        }
        out[f] = if kind.compare(v, mask[f]) { mask[f] } else { v };
    }

    // Anti-raster scan, collecting the seeds of the FIFO pass.
    let mut fifo: VecDeque<usize> = VecDeque::new();
    for f in (0..total).rev() {
        let mut v = out[f];
        for &g in later.at(f, size) {
            let vn = out[g];
            if kind.compare(vn, v) {
                v = vn;
            }
        }
        if kind.compare(v, mask[f]) {
            v = mask[f];
        }
        out[f] = v;
        // A later neighbor this pixel can still improve, but which the mask
        // does not already hold at that neighbor's own value.
        if later
            .at(f, size)
            .iter()
            .any(|&g| kind.compare(v, out[g]) && kind.compare(mask[g], out[g]))
        {
            fifo.push_back(f);
        }
    }

    // FIFO propagation over the full neighborhood.
    while let Some(f) = fifo.pop_front() {
        let v = out[f];
        for &g in full.at(f, size) {
            let vn = out[g];
            let mask_n = mask[g];
            if kind.compare(v, vn) && mask_n != vn {
                // `!kind.compare(out, mask)` is invariant, so `mask_n != vn`
                // means `kind.compare(mask_n, vn)`: improving `g` to
                // `kind`'s-best(v, mask_n) is always a strict improvement.
                out[g] = if kind.compare(mask_n, v) { v } else { mask_n };
                fifo.push_back(g);
            }
        }
    }
    out
}

/// Image-level wrapper shared by [`reconstruction_by_erosion`] and
/// [`reconstruction_by_dilation`]: validate shape/type and the marker/mask
/// precondition, then run [`reconstruct`].
fn reconstruct_images(
    marker_image: &Image,
    mask_image: &Image,
    fully_connected: bool,
    kind: ReconstructionKind,
) -> Result<Image> {
    require_same_shape(marker_image, mask_image)?;
    require_same_physical_space(marker_image, mask_image, 1)?;

    let marker = marker_image.to_f64_vec()?;
    let mask = mask_image.to_f64_vec()?;
    if marker.iter().zip(&mask).any(|(&m, &k)| kind.compare(m, k)) {
        return Err(FilterError::InvalidReconstructionMarker {
            relation: kind.required_relation(),
        });
    }

    let out = reconstruct(&marker, &mask, marker_image.size(), fully_connected, kind);
    image_from_f64(
        marker_image.pixel_id(),
        marker_image.size(),
        marker_image,
        &out,
    )
}

/// `ReconstructionByErosionImageFilter` (`itkReconstructionByErosionImageFilter.h`,
/// a `TCompare = std::less` instantiation of `itkReconstructionImageFilter.hxx`):
/// the largest image `<= marker_image` that is `>= mask_image` and has no
/// regional minimum that `mask_image` does not have.
///
/// `marker_image` and `mask_image` must share size and pixel type; the marker
/// must be pixelwise `>=` the mask everywhere
/// ([`FilterError::InvalidReconstructionMarker`] otherwise).
pub fn reconstruction_by_erosion(
    marker_image: &Image,
    mask_image: &Image,
    fully_connected: bool,
) -> Result<Image> {
    reconstruct_images(
        marker_image,
        mask_image,
        fully_connected,
        ReconstructionKind::Erosion,
    )
}

/// `ReconstructionByDilationImageFilter` (`itkReconstructionByDilationImageFilter.h`,
/// a `TCompare = std::greater` instantiation of `itkReconstructionImageFilter.hxx`):
/// the smallest image `>= marker_image` that is `<= mask_image` and has no
/// regional maximum that `mask_image` does not have.
///
/// `marker_image` and `mask_image` must share size and pixel type; the marker
/// must be pixelwise `<=` the mask everywhere
/// ([`FilterError::InvalidReconstructionMarker`] otherwise).
pub fn reconstruction_by_dilation(
    marker_image: &Image,
    mask_image: &Image,
    fully_connected: bool,
) -> Result<Image> {
    reconstruct_images(
        marker_image,
        mask_image,
        fully_connected,
        ReconstructionKind::Dilation,
    )
}

// ---- fillhole / grindpeak --------------------------------------------------

/// `ImageRegionExclusionConstIteratorWithIndex::SetExclusionRegionToInsetRegion`
/// (`itkImageRegionExclusionConstIteratorWithIndex.hxx`): the region
/// `GrayscaleFillholeImageFilter`/`GrayscaleGrindPeakImageFilter` treat as
/// "interior" is the image shrunk by one pixel from the boundary on every
/// axis — empty (so every pixel counts as "border") along any axis shorter
/// than 3.
fn is_interior(idx: &[usize], size: &[usize]) -> bool {
    idx.iter()
        .zip(size)
        .all(|(&i, &s)| s >= 3 && i >= 1 && i <= s - 2)
}

/// The marker `GrayscaleFillholeImageFilter`/`GrayscaleGrindPeakImageFilter`
/// build from `vals`: the input's own value on the one-pixel border shell,
/// `interior_fill` everywhere else (the input's max for fillhole, its min for
/// grindpeak).
fn border_marker(vals: &[f64], size: &[usize], interior_fill: f64) -> Vec<f64> {
    let st = strides(size);
    let mut idx = Vec::with_capacity(size.len());
    vals.iter()
        .enumerate()
        .map(|(f, &v)| {
            multi_index(f, size, &st, &mut idx);
            if is_interior(&idx, size) {
                interior_fill
            } else {
                v
            }
        })
        .collect()
}

/// `GrayscaleFillholeImageFilter` (`itkGrayscaleFillholeImageFilter.hxx`): fill
/// every local minimum not connected to the image border, leaving
/// border-touching minima untouched.
///
/// The marker holds `image`'s own values on its one-pixel border shell and
/// `image`'s maximum everywhere else, then reconstructs by erosion under
/// `image` as the mask: an interior minimum has no path of equal-or-lower
/// pixels reaching the border to hold it down, so it gets raised to its
/// surrounding rim; a border-touching one starts at the input's own (low)
/// value on the border pixel itself, which the erosion can never raise past.
pub fn grayscale_fillhole(image: &Image, fully_connected: bool) -> Result<Image> {
    let (_, max_value) = crate::minimum_maximum(image)?;
    let marker_vals = border_marker(&image.to_f64_vec()?, image.size(), max_value);
    let marker_image = image_from_f64(image.pixel_id(), image.size(), image, &marker_vals)?;
    reconstruction_by_erosion(&marker_image, image, fully_connected)
}

/// `GrayscaleGrindPeakImageFilter` (`itkGrayscaleGrindPeakImageFilter.hxx`):
/// the dual of [`grayscale_fillhole`] — remove every local maximum not
/// connected to the image border, by reconstructing by dilation under a
/// marker that holds `image`'s minimum on its interior and `image`'s own
/// border values on the shell.
pub fn grayscale_grindpeak(image: &Image, fully_connected: bool) -> Result<Image> {
    let (min_value, _) = crate::minimum_maximum(image)?;
    let marker_vals = border_marker(&image.to_f64_vec()?, image.size(), min_value);
    let marker_image = image_from_f64(image.pixel_id(), image.size(), image, &marker_vals)?;
    reconstruction_by_dilation(&marker_image, image, fully_connected)
}

// ---- h-minima / h-maxima / h-convex / h-concave ---------------------------

/// `HMinimaImageFilter` (`itkHMinimaImageFilter.hxx`): suppress every regional
/// minimum of `image` whose depth is at most `height`, by reconstructing
/// `image` by erosion under the marker `image + height`.
///
/// `height` is quantized to `image`'s pixel type before use, matching
/// [`quantize_to_pixel_type`]'s doc; the marker is then formed the way
/// `ShiftScaleImageFilter` forms it — accumulate `image + height` in a real
/// type, then saturate into the pixel type, which is what [`image_from_f64`]
/// does via `Scalar::from_f64`.
pub fn h_minima(image: &Image, height: f64, fully_connected: bool) -> Result<Image> {
    let height = quantize_to_pixel_type(image.pixel_id(), height);
    let vals = image.to_f64_vec()?;
    let shifted: Vec<f64> = vals.iter().map(|&v| v + height).collect();
    let marker_image = image_from_f64(image.pixel_id(), image.size(), image, &shifted)?;
    reconstruction_by_erosion(&marker_image, image, fully_connected)
}

/// `HMaximaImageFilter` (`itkHMaximaImageFilter.hxx`): the dual of
/// [`h_minima`] — suppress every regional maximum whose height above the
/// local background is at most `height`, by reconstructing `image` by
/// dilation under the marker `image - height`.
pub fn h_maxima(image: &Image, height: f64, fully_connected: bool) -> Result<Image> {
    let height = quantize_to_pixel_type(image.pixel_id(), height);
    let vals = image.to_f64_vec()?;
    let shifted: Vec<f64> = vals.iter().map(|&v| v - height).collect();
    let marker_image = image_from_f64(image.pixel_id(), image.size(), image, &shifted)?;
    reconstruction_by_dilation(&marker_image, image, fully_connected)
}

/// `HConvexImageFilter` (`itkHConvexImageFilter.hxx`): `image -
/// h_maxima(image, height)`, extracting every local maximum more than
/// `height` above its background.
pub fn h_convex(image: &Image, height: f64, fully_connected: bool) -> Result<Image> {
    let hmax = h_maxima(image, height, fully_connected)?;
    crate::subtract(image, &hmax)
}

/// `HConcaveImageFilter` (`itkHConcaveImageFilter.hxx`): `h_minima(image,
/// height) - image`, extracting every local minimum more than `height` below
/// its background.
pub fn h_concave(image: &Image, height: f64, fully_connected: bool) -> Result<Image> {
    let hmin = h_minima(image, height, fully_connected)?;
    crate::subtract(&hmin, image)
}

// ---- double_threshold -------------------------------------------------

/// `DoubleThresholdImageFilter` (`itkDoubleThresholdImageFilter.hxx`):
/// binarize `image` using a *narrow* threshold range `[threshold2,
/// threshold3]` as a marker and a *wide* range `[threshold1, threshold4]` as
/// a mask, then reconstruct the marker by dilation under the mask — keeping
/// only the wide-range objects that contain at least one narrow-range
/// (marker) pixel, while dropping wide-range objects that never touch the
/// narrow range.
///
/// `GenerateData()` builds this from two `BinaryThresholdImageFilter`s
/// (`narrowThreshold` on `[threshold2, threshold3]`, `wideThreshold` on
/// `[threshold1, threshold4]`, both with `InsideValue`/`OutsideValue`) and
/// one `ReconstructionByDilationImageFilter` (`MarkerImage = narrowThreshold`,
/// `MaskImage = wideThreshold`, `FullyConnected = fully_connected`) — reused
/// here as [`crate::binary_threshold`] (inclusive `[lower, upper]`, matching
/// `BinaryThresholdImageFilter`'s own comparison) and
/// [`reconstruction_by_dilation`].
///
/// The `.hxx` documents `Threshold1 <= Threshold2 <= Threshold3 <=
/// Threshold4` as a caller contract but never checks the *cross* relation
/// (`Threshold1 <= Threshold2`, `Threshold3 <= Threshold4`) itself — that
/// relation only matters indirectly, through whether the narrow range ends up
/// a pixelwise subset of the wide range; when it isn't,
/// [`reconstruction_by_dilation`]'s own marker-`<=`-mask precondition surfaces
/// as [`FilterError::InvalidReconstructionMarker`], exactly mirroring
/// `itkReconstructionImageFilter.hxx`'s own runtime check — this port adds no
/// upfront cross-range validation of its own to match.
///
/// What *is* checked upfront, independently for each pair, is each
/// `BinaryThresholdImageFilter`'s own `BeforeThreadedGenerateData`
/// precondition (`Threshold2 <= Threshold3`, `Threshold1 <= Threshold4`):
/// violating either is [`FilterError::InvalidThresholdRange`].
///
/// `thresholds` is `[Threshold1, Threshold2, Threshold3, Threshold4]` in the
/// `.hxx`'s own order, grouped into one array to keep the parameter count
/// down.
pub fn double_threshold(
    image: &Image,
    thresholds: [f64; 4],
    inside_value: u8,
    outside_value: u8,
    fully_connected: bool,
) -> Result<Image> {
    let [threshold1, threshold2, threshold3, threshold4] = thresholds;
    if threshold2 > threshold3 {
        return Err(FilterError::InvalidThresholdRange {
            lower: threshold2,
            upper: threshold3,
        });
    }
    if threshold1 > threshold4 {
        return Err(FilterError::InvalidThresholdRange {
            lower: threshold1,
            upper: threshold4,
        });
    }

    let narrow =
        crate::binary_threshold(image, threshold2, threshold3, inside_value, outside_value)?;
    let wide = crate::binary_threshold(image, threshold1, threshold4, inside_value, outside_value)?;
    reconstruction_by_dilation(&narrow, &wide, fully_connected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- reconstruction_by_erosion / reconstruction_by_dilation ----

    /// mask `[0,3,1,3,0]`: the middle minimum has depth 2 (bounded by 3 on
    /// each side), the two end minima have depth 3.
    #[test]
    fn reconstruction_by_erosion_fills_a_shallow_minimum() {
        let mask = img_i32(&[5, 1], vec![0, 3, 1, 3, 0]);

        // marker = mask + 2: the depth-2 middle minimum is erased, the two
        // depth-3 end minima survive.
        let marker2 = img_i32(&[5, 1], vec![2, 5, 3, 5, 2]);
        assert_eq!(
            reconstruction_by_erosion(&marker2, &mask, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[2, 3, 3, 3, 2]
        );

        // marker = mask + 1: the middle minimum is only flattened to its own
        // depth, and still reads as a regional minimum.
        let marker1 = img_i32(&[5, 1], vec![1, 4, 2, 4, 1]);
        assert_eq!(
            reconstruction_by_erosion(&marker1, &mask, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[1, 3, 2, 3, 1]
        );
    }

    /// `D(marker, mask) == -E(-marker, -mask)`, since `a > b <=> -a < -b`
    /// makes `ReconstructionKind::Dilation`'s comparator the exact negation
    /// of `Erosion`'s. Exercises the dilation engine independently of any
    /// hand-derived array.
    #[test]
    fn reconstruction_by_dilation_is_the_negated_dual_of_erosion() {
        let mask = img_i32(&[5, 1], vec![5, 2, 4, 2, 5]);
        let marker = img_i32(&[5, 1], vec![3, 0, 2, 0, 3]);
        let dilated = reconstruction_by_dilation(&marker, &mask, true).unwrap();

        let neg_mask = img_i32(&[5, 1], vec![-5, -2, -4, -2, -5]);
        let neg_marker = img_i32(&[5, 1], vec![-3, 0, -2, 0, -3]);
        let eroded = reconstruction_by_erosion(&neg_marker, &neg_mask, true).unwrap();
        let expected: Vec<i32> = eroded
            .scalar_slice::<i32>()
            .unwrap()
            .iter()
            .map(|&v| -v)
            .collect();

        assert_eq!(dilated.scalar_slice::<i32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn erosion_rejects_marker_below_mask() {
        let marker = img_i32(&[3, 1], vec![0, 0, 0]);
        let mask = img_i32(&[3, 1], vec![1, 1, 1]);
        assert!(matches!(
            reconstruction_by_erosion(&marker, &mask, false),
            Err(FilterError::InvalidReconstructionMarker { relation: ">=" })
        ));
    }

    #[test]
    fn dilation_rejects_marker_above_mask() {
        let marker = img_i32(&[3, 1], vec![2, 2, 2]);
        let mask = img_i32(&[3, 1], vec![1, 1, 1]);
        assert!(matches!(
            reconstruction_by_dilation(&marker, &mask, false),
            Err(FilterError::InvalidReconstructionMarker { relation: "<=" })
        ));
    }

    #[test]
    fn reconstruction_rejects_mismatched_sizes() {
        let marker = img_i32(&[3, 1], vec![1, 1, 1]);
        let mask = img_i32(&[4, 1], vec![1, 1, 1, 1]);
        assert!(matches!(
            reconstruction_by_erosion(&marker, &mask, false),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    // ---- grayscale_fillhole / grayscale_grindpeak ----

    #[test]
    fn grayscale_fillhole_fills_enclosed_minimum_but_not_border_touching_one() {
        #[rustfmt::skip]
        let image = img_i32(&[5, 5], vec![
            5, 5, 5, 5, 5,
            5, 5, 5, 5, 5,
            5, 5, 1, 5, 5,
            5, 5, 5, 5, 5,
            1, 5, 5, 5, 5,
        ]);
        let out = grayscale_fillhole(&image, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[
            5, 5, 5, 5, 5,
            5, 5, 5, 5, 5,
            5, 5, 5, 5, 5,
            5, 5, 5, 5, 5,
            1, 5, 5, 5, 5,
        ]);
    }

    #[test]
    fn grayscale_grindpeak_removes_interior_peak_but_not_border_touching_plateau() {
        #[rustfmt::skip]
        let image = img_i32(&[5, 5], vec![
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 9, 0, 0,
            0, 0, 0, 0, 0,
            9, 9, 0, 0, 0,
        ]);
        let out = grayscale_grindpeak(&image, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            9, 9, 0, 0, 0,
        ]);
    }

    /// The only interior pixel of a 3x3 image is its center; the corners are
    /// themselves border pixels, so a low corner-to-corner diagonal is a real
    /// bridge to the border under full connectivity but not under face
    /// connectivity.
    #[test]
    fn fully_connected_changes_whether_a_diagonal_bridge_reaches_the_border() {
        #[rustfmt::skip]
        let image = img_i32(&[3, 3], vec![
            1, 5, 5,
            5, 1, 5,
            5, 5, 1,
        ]);

        let face = grayscale_fillhole(&image, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<i32>().unwrap(), &[
            1, 5, 5,
            5, 5, 5,
            5, 5, 1,
        ]);

        let full = grayscale_fillhole(&image, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.scalar_slice::<i32>().unwrap(), &[
            1, 5, 5,
            5, 1, 5,
            5, 5, 1,
        ]);
    }

    #[test]
    fn flat_image_is_unchanged_by_fillhole_and_grindpeak() {
        let image = img_i32(&[3, 3], vec![4; 9]);
        assert_eq!(
            grayscale_fillhole(&image, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[4; 9]
        );
        assert_eq!(
            grayscale_grindpeak(&image, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[4; 9]
        );
    }

    // ---- h_minima / h_maxima ----

    #[test]
    fn h_minima_removes_minima_shallower_than_height_keeps_deeper_ones() {
        // [0,3,1,3,0]: the middle minimum has depth 2, the end minima depth 3.
        let image = img_i32(&[5, 1], vec![0, 3, 1, 3, 0]);

        // height = 2 == the middle minimum's depth: it is erased (raised to
        // 3); the deeper end minima survive, each raised only by height.
        assert_eq!(
            h_minima(&image, 2.0, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[2, 3, 3, 3, 2]
        );

        // height = 1 < the middle minimum's depth: it survives as a regional
        // minimum, merely raised by height like everything else.
        assert_eq!(
            h_minima(&image, 1.0, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[1, 3, 2, 3, 1]
        );
    }

    /// Dual of the `h_minima` case above via `image = -[0,3,1,3,0]`: every
    /// minimum becomes a maximum at the same index with the same depth, so
    /// `h_maxima`'s output is the negation of `h_minima`'s.
    #[test]
    fn h_maxima_removes_peaks_shallower_than_height_keeps_deeper_ones() {
        let peaks = img_i32(&[5, 1], vec![0, -3, -1, -3, 0]);

        assert_eq!(
            h_maxima(&peaks, 2.0, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[-2, -3, -3, -3, -2]
        );
        assert_eq!(
            h_maxima(&peaks, 1.0, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[-1, -3, -2, -3, -1]
        );
    }

    #[test]
    fn flat_image_h_minima_and_h_maxima_shift_uniformly() {
        // A flat image has no reachable boundary within the grid to hold a
        // reconstruction down, so the marker (image +/- height) passes
        // through unchanged.
        let image = img_i32(&[3, 1], vec![4, 4, 4]);
        assert_eq!(
            h_minima(&image, 2.0, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[6, 6, 6]
        );
        assert_eq!(
            h_maxima(&image, 2.0, false)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[2, 2, 2]
        );
    }

    // ---- h_convex / h_concave ----

    #[test]
    fn h_convex_and_h_concave_match_their_hxx_identities() {
        let image = img_i32(&[5, 1], vec![0, 3, 1, 3, 0]);

        let convex = h_convex(&image, 1.0, false).unwrap();
        let expected_convex =
            crate::subtract(&image, &h_maxima(&image, 1.0, false).unwrap()).unwrap();
        assert_eq!(convex, expected_convex);

        let concave = h_concave(&image, 1.0, false).unwrap();
        let expected_concave =
            crate::subtract(&h_minima(&image, 1.0, false).unwrap(), &image).unwrap();
        assert_eq!(concave, expected_concave);
    }

    #[test]
    fn h_convex_extracts_the_peak_height_above_background() {
        let peaks = img_i32(&[5, 1], vec![0, -3, -1, -3, 0]);
        let out = h_convex(&peaks, 2.0, false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[2, 0, 2, 0, 2]);
    }

    #[test]
    fn h_concave_extracts_the_valley_depth_below_background() {
        let mask = img_i32(&[5, 1], vec![0, 3, 1, 3, 0]);
        let out = h_concave(&mask, 2.0, false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[2, 0, 2, 0, 2]);
    }

    // ---- double_threshold ----

    /// Two bumps: `[0,5,10,5,0]` peaks at `10` (reaches the narrow range
    /// `[8,10]`), `[0,6,6,6,0]` peaks at `6` (inside the wide range `[1,10]`
    /// but never the narrow one). Only the first bump is a "narrow-seeded"
    /// wide component, so only it survives reconstruction; the second is
    /// dropped in full even though every one of its pixels passed the wide
    /// threshold.
    #[test]
    fn double_threshold_keeps_inner_seeded_component_drops_outer_only_one() {
        let image = img_i32(&[10, 1], vec![0, 5, 10, 5, 0, 0, 6, 6, 6, 0]);
        let out = double_threshold(&image, [1.0, 8.0, 10.0, 10.0], 1, 0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            &[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn double_threshold_output_is_always_uint8() {
        let image = img_i32(&[3, 1], vec![0, 5, 0]);
        let out = double_threshold(&image, [1.0, 4.0, 10.0, 10.0], 1, 0, false).unwrap();
        assert_eq!(out.pixel_id(), sitk_core::PixelId::UInt8);
    }

    /// `BinaryThresholdImageFilter::BeforeThreadedGenerateData` throws when a
    /// pair's own lower threshold exceeds its own upper threshold — checked
    /// independently for the narrow pair (`Threshold2`/`Threshold3`) and the
    /// wide pair (`Threshold1`/`Threshold4`), before reconstruction ever runs.
    #[test]
    fn double_threshold_rejects_an_inverted_pair() {
        let image = img_i32(&[3, 1], vec![0, 5, 0]);
        assert_eq!(
            double_threshold(&image, [0.0, 9.0, 8.0, 10.0], 1, 0, false).unwrap_err(),
            FilterError::InvalidThresholdRange {
                lower: 9.0,
                upper: 8.0
            }
        );
        assert_eq!(
            double_threshold(&image, [9.0, 1.0, 2.0, 8.0], 1, 0, false).unwrap_err(),
            FilterError::InvalidThresholdRange {
                lower: 9.0,
                upper: 8.0
            }
        );
    }

    /// The `.hxx` never validates `Threshold1 <= Threshold2` /
    /// `Threshold3 <= Threshold4` up front — a narrow range that is *not* a
    /// pixelwise subset of the wide range only surfaces once
    /// [`reconstruction_by_dilation`]'s own marker-`<=`-mask precondition
    /// rejects it. Here `threshold2 = 0 < threshold1 = 5` makes the narrow
    /// range mark pixels (value `3`, at index 1) the wide range excludes.
    #[test]
    fn double_threshold_surfaces_reconstruction_marker_error_for_non_subset_ranges() {
        let image = img_i32(&[5, 1], vec![0, 3, 7, 3, 0]);
        assert_eq!(
            double_threshold(&image, [5.0, 0.0, 10.0, 10.0], 1, 0, false).unwrap_err(),
            FilterError::InvalidReconstructionMarker { relation: "<=" }
        );
    }

    // ---- connectivity helpers ----

    #[test]
    fn neighbor_offsets_are_in_ascending_neighborhood_index() {
        // 2-D face connectivity, ascending index: (0,-1), (-1,0), (1,0), (0,1).
        assert_eq!(
            neighbor_offsets(2, false, Half::Full),
            vec![vec![0, -1], vec![-1, 0], vec![1, 0], vec![0, 1]]
        );
        // Previous / later split the same list at the center.
        assert_eq!(
            neighbor_offsets(2, false, Half::Previous),
            vec![vec![0, -1], vec![-1, 0]]
        );
        assert_eq!(
            neighbor_offsets(2, false, Half::Later),
            vec![vec![1, 0], vec![0, 1]]
        );
        // Full connectivity keeps all 8, still ascending.
        assert_eq!(neighbor_offsets(2, true, Half::Full).len(), 8);
        assert_eq!(neighbor_offsets(2, true, Half::Previous).len(), 4);
        assert_eq!(neighbor_offsets(2, true, Half::Later).len(), 4);
        assert_eq!(neighbor_offsets(3, true, Half::Full).len(), 26);
        assert_eq!(neighbor_offsets(3, false, Half::Full).len(), 6);
    }
}
