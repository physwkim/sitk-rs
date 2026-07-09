//! Label/binary boundary filters: `BinaryContourImageFilter`,
//! `LabelContourImageFilter`, `BinaryPruningImageFilter`, and
//! `SimpleContourExtractorImageFilter`.
//!
//! [`simple_contour_extractor`] is the odd one out: a plain box-neighborhood
//! scan (`Modules/Filtering/ImageFeature/include/itkSimpleContourExtractorImageFilter.hxx`),
//! not a scanline RLE, and it emits a `UInt8` mask rather than the input's own
//! values. See its own docs.
//!
//! ## `binary_contour` / `label_contour`
//!
//! Verified against `Modules/Filtering/ImageLabel/include/`:
//! `itkBinaryContourImageFilter.hxx`, `itkLabelContourImageFilter.hxx`, and
//! the shared `itkScanlineFilterCommon.h` (`SetupLineOffsets`,
//! `CheckNeighbors`, `CompareLines`) both inherit from. Both filters are
//! wrapped for every dimension `>= 2` (`itk_wrap_image_filter(..., 2, 2+)`),
//! and this port is N-D throughout, not just 2-D.
//!
//! These filters are *not* a per-pixel neighborhood scan; they run-length
//! encode (RLE) every scanline along axis 0 into maximal same-classification
//! runs, then compare each line's runs against the runs of every "line
//! neighbor" (a line offset by one step along any combination of the other
//! axes) to find where a run of one classification directly abuts or
//! diagonally touches a run of a different classification. A "line
//! neighbor" is chosen the same way [`crate::label::connected_component`]
//! chooses one (see that module's `line_neighbor_offsets`): axis-aligned
//! only when `fully_connected` is `false`, every combination of `{-1,0,1}`
//! across the outer axes when `true` — plus, uniquely to these two filters,
//! the *same* line is always included as its own "neighbor" too (mirroring
//! `SetupLineOffsets(wholeNeighborhood=true)` appending offset `0`), because
//! a run and the very next run on the same scanline are exactly where most
//! contour pixels are found.
//!
//! Within one line-pair comparison, `CompareLines` finds each overlapping
//! `[start, end]` span between a "current" run and a "neighbour" run by
//! padding the **neighbour's** span (not the current run's) by a tolerance
//! of `0` or `1` pixels before checking overlap ([`overlap_with_tolerance`]
//! mirrors its four overlap cases exactly). That padding is `1` whenever the
//! two lines being compared are literally the same line (an fg/bg run pair
//! that RLE-abut on one line are adjacent-but-not-overlapping index ranges,
//! so *some* tolerance is unconditionally needed there to detect the touch
//! at all — this holds regardless of `fully_connected`), and otherwise `1`
//! only when `fully_connected` is set. Because the padding is applied
//! one-sidedly to the neighbour argument, and in `binary_contour` the
//! neighbour argument is always the *background* line map, `fully_connected`
//! in this algorithm concretely means "how far the background classification
//! is allowed to reach, diagonally, to touch a foreground run" — the
//! opposite framing from how `fully_connected` reads in
//! [`crate::label::connected_component`] (there it directly extends
//! *foreground*-object connectivity). The two knobs happen to agree on 2-D
//! diagonal touches only because the "same classification" runs on the
//! *other* line get filtered out by the classification check before the
//! padding is ever applied to them.
//!
//! Both filters run a strict two-pass structure: pass 1 RLEs every line
//! (recording each run's `[start, start+len-1]` span) and fills the output
//! with each filter's *default* value; pass 2 walks the run maps built in
//! pass 1 (untouched by pass 1's output writes) and restores the non-default
//! value on every span found touching a differently-classified neighbour.
//! The two filters differ in what "default" and "restore" mean:
//!
//! - `binary_contour`: pass 1 sets *foreground* (`== foreground_value`)
//!   pixels to `background_value`, but background pixels keep their
//!   original input value verbatim (`outLineIt.Set(PVal)` — *not*
//!   normalized to `background_value`, a genuine quirk: an input with
//!   several distinct non-foreground values keeps them all in the output).
//!   Pass 2 restores `foreground_value` on every foreground span touching
//!   *any* background run.
//! - `label_contour`: pass 1 sets **every** pixel, foreground or
//!   background alike, to `background_value`. Pass 2 restores each run's
//!   *own original value* (not just `foreground_value`) on every non-
//!   background span touching a run of a *different* value — which
//!   includes background, but also a same-line/same-neighbor run carrying a
//!   different label, so label-vs-label borders are marked from both
//!   labels' own sides.
//!
//! In both filters the marked spans lie on the *inside* of the original
//! object (the outer shell of foreground/labeled pixels), not in the
//! surrounding background — an interior pixel with no differently-
//! classified neighbor within tolerance reverts to the default value.
//!
//! Both filters restrict to `IntegerPixelIDTypeList`
//! (`BinaryContourImageFilter.yaml`/`LabelContourImageFilter.yaml`); a
//! floating-point image is rejected with
//! [`FilterError::RequiresIntegerPixelType`], matching how [`crate::logic`]
//! gates its bitwise filters.
//!
//! ## `binary_pruning`
//!
//! Verified against
//! `Modules/Filtering/BinaryMathematicalMorphology/include/itkBinaryPruningImageFilter.hxx`.
//! `ComputePruneImage` hardcodes 2-element neighbor offsets exactly like
//! `itkBinaryThinningImageFilter.hxx`'s `ComputeThinImage`
//! ([`crate::binary_morphology::binary_thinning`]'s module docs), and
//! `itkBinaryPruningImageFilter.wrap` only instantiates the filter for
//! 2-D images, so this port rejects other dimensions with
//! [`FilterError::UnsupportedPruningDimension`].
//! `BinaryPruningImageFilter.yaml`'s `pixel_types` further narrows the
//! *SimpleITK* wrapper to `UInt8` alone (unlike the underlying ITK template,
//! which the `.wrap` file shows also being instantiated for unsigned-int and
//! real pixel types) — this port matches SimpleITK's narrower surface and
//! rejects any other pixel type with
//! [`FilterError::RequiresUInt8PixelType`].
//!
//! Unlike `binary_thinning`, which collects a whole sub-pass's deletions and
//! applies them only after the sub-pass finishes scanning, `ComputePruneImage`
//! mutates its *single* buffer in place as the raster scan proceeds, with no
//! deferred-delete buffering at all: `NeighborhoodIteratorType ot(...,
//! pruneImage, region)` both reads neighbor pixels and writes
//! `ot.SetCenterPixel(0)` through the same iterator over the same image. Of
//! the 8 offsets, `(-1,-1)`, `(-1,0)`, `(1,-1)`, `(0,-1)` land on pixels
//! *earlier* in raster (x-fastest) order than the current pixel — so within
//! one pass, a pixel's genus sum can see neighbors already zeroed earlier in
//! *that same pass*, not just prior passes. This port reproduces that
//! sequential single-buffer mutation exactly (no double buffering), which is
//! what makes a straight spur recede by exactly one pixel per `iteration`
//! rather than potentially cascading within a single pass (see the module's
//! tests for a hand-traced example). Each `genus` accumulation is a
//! `u8::wrapping_add`, matching `PixelType genus` (`uint8_t`) truncating on
//! every `+=` in the C++.

use crate::error::{FilterError, Result};
use crate::quantize_to_pixel_type;
use sitk_core::{
    Image, NeighborhoodIterator, PixelId, Scalar, ZeroFluxNeumannBoundaryCondition, dispatch_scalar,
};

fn require_integer_pixel_type(image: &Image) -> Result<()> {
    if image.pixel_id().is_floating_point() {
        return Err(FilterError::RequiresIntegerPixelType(image.pixel_id()));
    }
    Ok(())
}

// ---- shared N-D line indexing helpers ----------------------------------
//
// Same construction as `crate::label::connected_component`'s helpers of the
// same names: axis 0 is always the scanline axis and is excluded from the
// "outer" (line-identifying) index space.

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// Decode a flat (first-index-fastest) offset into a multi-index.
fn multi_index(flat: usize, size: &[usize], strides: &[usize]) -> Vec<usize> {
    (0..size.len())
        .map(|d| (flat / strides[d]) % size[d])
        .collect()
}

/// All neighbor-*line* offset vectors over the outer axes, within Chebyshev
/// distance 1 and excluding the all-zero (self) offset: face-only when
/// `!fully_connected` (`setConnectivity` in
/// `itkConnectedComponentAlgorithm.h`), every nonzero combination when
/// `fully_connected`. The "self" (all-zero) line is handled separately by
/// callers, since it always participates regardless of `fully_connected`
/// (see module docs).
fn line_neighbor_offsets(outer_dim: usize, fully_connected: bool) -> Vec<Vec<i64>> {
    if outer_dim == 0 {
        return Vec::new();
    }
    let total = 3usize.pow(outer_dim as u32);
    let mut offsets = Vec::new();
    for code in 0..total {
        let mut rem = code;
        let mut offset = vec![0i64; outer_dim];
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
        offsets.push(offset);
    }
    offsets
}

/// The flat line index of `idx + offset` in the outer (non-scanline) index
/// space, or `None` if it falls outside `outer_size`.
fn neighbor_line_index(
    idx: &[usize],
    offset: &[i64],
    outer_size: &[usize],
    outer_strides: &[usize],
) -> Option<usize> {
    let mut flat = 0usize;
    for d in 0..idx.len() {
        let v = idx[d] as i64 + offset[d];
        if v < 0 || v as usize >= outer_size[d] {
            return None;
        }
        flat += v as usize * outer_strides[d];
    }
    Some(flat)
}

/// `CompareLines`' overlap test: the inclusive `[start, end]` overlap between
/// `[cs, ce]` and `[ns - tol, ne + tol]`, or `None` if they don't touch. The
/// four branches are exactly `itkScanlineFilterCommon.h`'s four overlap
/// cases (case numbers kept in the comments below to match the `.hxx`).
fn overlap_with_tolerance(cs: i64, ce: i64, ns: i64, ne: i64, tol: i64) -> Option<(i64, i64)> {
    let ss1 = ns - tol;
    let ee2 = ne + tol;
    if ss1 >= cs && ee2 <= ce {
        Some((ss1, ee2)) // case 1: neighbour span strictly inside current
    } else if ss1 <= cs && ee2 >= ce {
        Some((cs, ce)) // case 4: neighbour span strictly contains current
    } else if ss1 <= ce && ee2 >= ce {
        Some((ss1, ce)) // case 2: neighbour overlaps current's tail
    } else if ss1 <= cs && ee2 >= cs {
        Some((cs, ee2)) // case 3: neighbour overlaps current's head
    } else {
        None
    }
}

// ---- binary_contour ------------------------------------------------------

/// (start_x, length) of one run along axis 0.
type Span = (i64, i64);

fn mark_touches<T: Copy>(
    current: &[Span],
    neighbor: &[Span],
    tol: i64,
    line_id: usize,
    xsize: usize,
    output: &mut [T],
    mark_value: T,
) {
    let base = line_id * xsize;
    for &(cs, clen) in current {
        let ce = cs + clen - 1;
        for &(ns, nlen) in neighbor {
            let ne = ns + nlen - 1;
            if let Some((os, ol)) = overlap_with_tolerance(cs, ce, ns, ne, tol) {
                for x in os..=ol {
                    output[base + x as usize] = mark_value;
                }
            }
        }
    }
}

fn binary_contour_typed<T: Scalar>(
    image: &Image,
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    let size = image.size();
    let total: usize = size.iter().product();
    if total == 0 {
        let mut result = Image::from_vec(size, Vec::<T>::new())?;
        result.copy_geometry_from(image);
        return Ok(result);
    }
    let xsize = size[0];
    let linecount = total / xsize;

    let foreground = T::from_f64(foreground_value);
    let background = T::from_f64(background_value);
    let input = image.scalar_slice::<T>()?;

    // Pass 1: RLE every line into foreground/background run maps; fill the
    // default output (foreground -> background_value, background pixels
    // pass through their original value untouched -- see module docs).
    let mut output = input.to_vec();
    let mut fg_map: Vec<Vec<Span>> = Vec::with_capacity(linecount);
    let mut bg_map: Vec<Vec<Span>> = Vec::with_capacity(linecount);
    for line in 0..linecount {
        let base = line * xsize;
        let row = &input[base..base + xsize];
        let mut fg = Vec::new();
        let mut bg = Vec::new();
        let mut x = 0usize;
        while x < xsize {
            let start = x;
            let is_fg = row[x] == foreground;
            while x < xsize && (row[x] == foreground) == is_fg {
                x += 1;
            }
            if is_fg {
                for i in start..x {
                    output[base + i] = background;
                }
                fg.push((start as i64, (x - start) as i64));
            } else {
                bg.push((start as i64, (x - start) as i64));
            }
        }
        fg_map.push(fg);
        bg_map.push(bg);
    }

    // Pass 2: restore foreground_value on every foreground span touching a
    // background run, on the same line (tolerance always 1) or a neighbor
    // line (tolerance 1 only when fully_connected).
    let outer_size = &size[1..];
    let outer_dim = outer_size.len();
    let outer_strides = strides(outer_size);
    let neighbor_offsets = line_neighbor_offsets(outer_dim, fully_connected);
    let tol = i64::from(fully_connected);

    for line_id in 0..linecount {
        if fg_map[line_id].is_empty() {
            continue;
        }
        if !bg_map[line_id].is_empty() {
            mark_touches(
                &fg_map[line_id],
                &bg_map[line_id],
                1,
                line_id,
                xsize,
                &mut output,
                foreground,
            );
        }
        if outer_dim == 0 {
            continue;
        }
        let idx = multi_index(line_id, outer_size, &outer_strides);
        for offset in &neighbor_offsets {
            let Some(neighbor_line) = neighbor_line_index(&idx, offset, outer_size, &outer_strides)
            else {
                continue;
            };
            if bg_map[neighbor_line].is_empty() {
                continue;
            }
            mark_touches(
                &fg_map[line_id],
                &bg_map[neighbor_line],
                tol,
                line_id,
                xsize,
                &mut output,
                foreground,
            );
        }
    }

    let mut result = Image::from_vec(size, output)?;
    result.copy_geometry_from(image);
    Ok(result)
}

/// `BinaryContourImageFilter`: keep only the foreground pixels
/// (`== foreground_value`) that touch a background pixel (per
/// `fully_connected`'s tolerance -- see module docs), replacing every other
/// foreground pixel with `background_value`. Background pixels always keep
/// their original input value. Integer pixel types only; output pixel type
/// matches input.
pub fn binary_contour(
    image: &Image,
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    require_integer_pixel_type(image)?;
    dispatch_scalar!(
        image.pixel_id(),
        binary_contour_typed,
        image,
        foreground_value,
        background_value,
        fully_connected
    )
}

// ---- label_contour --------------------------------------------------------

/// (start_x, length, value) of one run along axis 0.
type LabelSpan<T> = (i64, i64, T);

fn rle_line_by_value<T: Scalar>(row: &[T]) -> Vec<LabelSpan<T>> {
    let xsize = row.len();
    let mut runs = Vec::new();
    let mut x = 0usize;
    while x < xsize {
        let start = x;
        let v = row[x];
        while x < xsize && row[x] == v {
            x += 1;
        }
        runs.push((start as i64, (x - start) as i64, v));
    }
    runs
}

fn mark_label_touches<T: Scalar>(
    current: &[LabelSpan<T>],
    neighbor: &[LabelSpan<T>],
    tol: i64,
    line_id: usize,
    xsize: usize,
    output: &mut [T],
    background: T,
) {
    let base = line_id * xsize;
    for &(cs, clen, cv) in current {
        if cv == background {
            continue;
        }
        let ce = cs + clen - 1;
        for &(ns, nlen, nv) in neighbor {
            if nv == cv {
                continue;
            }
            let ne = ns + nlen - 1;
            if let Some((os, ol)) = overlap_with_tolerance(cs, ce, ns, ne, tol) {
                for x in os..=ol {
                    output[base + x as usize] = cv;
                }
            }
        }
    }
}

fn label_contour_typed<T: Scalar>(
    image: &Image,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    let size = image.size();
    let total: usize = size.iter().product();
    let background = T::from_f64(background_value);
    if total == 0 {
        let mut result = Image::from_vec(size, Vec::<T>::new())?;
        result.copy_geometry_from(image);
        return Ok(result);
    }
    let xsize = size[0];
    let linecount = total / xsize;
    let input = image.scalar_slice::<T>()?;

    // Pass 1: every pixel defaults to background_value, whatever its input
    // value was (see module docs).
    let mut output = vec![background; total];
    let mut line_map: Vec<Vec<LabelSpan<T>>> = Vec::with_capacity(linecount);
    for line in 0..linecount {
        let base = line * xsize;
        line_map.push(rle_line_by_value(&input[base..base + xsize]));
    }

    // Pass 2: restore each run's own value on every non-background span
    // touching a differently-valued run (including background itself), on
    // the same line (tolerance always 1) or a neighbor line (tolerance 1
    // only when fully_connected).
    let outer_size = &size[1..];
    let outer_dim = outer_size.len();
    let outer_strides = strides(outer_size);
    let neighbor_offsets = line_neighbor_offsets(outer_dim, fully_connected);
    let tol = i64::from(fully_connected);

    for line_id in 0..linecount {
        if line_map[line_id].is_empty() {
            continue;
        }
        mark_label_touches(
            &line_map[line_id],
            &line_map[line_id],
            1,
            line_id,
            xsize,
            &mut output,
            background,
        );

        if outer_dim == 0 {
            continue;
        }
        let idx = multi_index(line_id, outer_size, &outer_strides);
        for offset in &neighbor_offsets {
            let Some(neighbor_line) = neighbor_line_index(&idx, offset, outer_size, &outer_strides)
            else {
                continue;
            };
            if line_map[neighbor_line].is_empty() {
                continue;
            }
            mark_label_touches(
                &line_map[line_id],
                &line_map[neighbor_line],
                tol,
                line_id,
                xsize,
                &mut output,
                background,
            );
        }
    }

    let mut result = Image::from_vec(size, output)?;
    result.copy_geometry_from(image);
    Ok(result)
}

/// `LabelContourImageFilter`: keep only the labeled pixels
/// (`!= background_value`) that touch a differently-labeled pixel, including
/// background, per `fully_connected`'s tolerance (see module docs);
/// everything else becomes `background_value`. Each kept pixel retains its
/// own original label. Integer pixel types only; output pixel type matches
/// input.
pub fn label_contour(image: &Image, background_value: f64, fully_connected: bool) -> Result<Image> {
    require_integer_pixel_type(image)?;
    dispatch_scalar!(
        image.pixel_id(),
        label_contour_typed,
        image,
        background_value,
        fully_connected
    )
}

// ---- binary_pruning ---------------------------------------------------------

/// The pixel at `(x, y)`, `ZeroFluxNeumannBoundaryCondition`-clamped to
/// `[0, w) x [0, h)`.
fn clamped_get(data: &[u8], w: i64, h: i64, x: i64, y: i64) -> u8 {
    let cx = x.clamp(0, w - 1) as usize;
    let cy = y.clamp(0, h - 1) as usize;
    data[cx + w as usize * cy]
}

/// `ComputePruneImage`'s 8 neighbor offsets, in the `.hxx`'s own
/// `offset1`..`offset8` order (the sum is commutative, so order does not
/// affect the result -- kept for traceability against the source).
const PRUNE_OFFSETS: [(i64, i64); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
];

/// `BinaryPruningImageFilter::ComputePruneImage`: `iteration` sequential
/// raster-order passes over a single shared buffer (see module docs for why
/// this must not be double-buffered). Each pass deletes a foreground pixel
/// (sets it to 0) when the wrapping `u8` sum of its 8 neighbors' values,
/// read live from the same in-progress buffer, is less than 2.
fn compute_prune_image(indicator: &[u8], w: i64, h: i64, iteration: u32) -> Vec<u8> {
    let mut data = indicator.to_vec();
    for _ in 0..iteration {
        for y in 0..h {
            for x in 0..w {
                let idx = (x + w * y) as usize;
                if data[idx] == 0 {
                    continue;
                }
                let mut genus: u8 = 0;
                for &(dx, dy) in &PRUNE_OFFSETS {
                    genus = genus.wrapping_add(clamped_get(&data, w, h, x + dx, y + dy));
                }
                if genus < 2 {
                    data[idx] = 0;
                }
            }
        }
    }
    data
}

/// `BinaryPruningImageFilter`: erode `iteration`-pixel-long spurs (1-pixel-
/// wide protrusions) from a binary image, 2-D `UInt8` only (see module
/// docs). Output pixel type matches input (`UInt8`).
pub fn binary_pruning(image: &Image, iteration: u32) -> Result<Image> {
    if image.pixel_id() != PixelId::UInt8 {
        return Err(FilterError::RequiresUInt8PixelType(image.pixel_id()));
    }
    let size = image.size();
    if size.len() != 2 {
        return Err(FilterError::UnsupportedPruningDimension(size.len()));
    }
    let w = size[0] as i64;
    let h = size[1] as i64;
    let input = image.scalar_slice::<u8>()?;
    let pruned = compute_prune_image(input, w, h, iteration);
    let mut result = Image::from_vec(size, pruned)?;
    result.copy_geometry_from(image);
    Ok(result)
}

// ---- simple_contour_extractor ---------------------------------------------

fn simple_contour_extractor_typed<T: Scalar>(
    image: &Image,
    input_foreground_value: f64,
    input_background_value: f64,
    radius: &[usize],
    output_foreground_value: u8,
    output_background_value: u8,
) -> Result<Image> {
    let foreground = T::from_f64(input_foreground_value);
    let background = T::from_f64(input_background_value);

    let iter = NeighborhoodIterator::<T, _>::new(image, radius, ZeroFluxNeumannBoundaryCondition)?;
    let output: Vec<u8> = iter
        .map(|(_, nb)| {
            if nb.center_value() != foreground {
                return output_background_value;
            }
            if nb.values().contains(&background) {
                output_foreground_value
            } else {
                output_background_value
            }
        })
        .collect();

    let mut result = Image::from_vec(image.size(), output)?;
    result.copy_geometry_from(image);
    Ok(result)
}

/// `SimpleContourExtractorImageFilter`: mark every pixel that *is* foreground
/// (`== input_foreground_value`) and has at least one pixel equal to
/// `input_background_value` inside its box neighborhood of `radius`, under
/// [`ZeroFluxNeumannBoundaryCondition`] (`bit.OverrideBoundaryCondition(&nbc)`
/// in the `.hxx`, so the image border replicates rather than supplying
/// background — a foreground blob running off the edge has no contour there).
/// Marked pixels get `output_foreground_value`, everything else — including
/// every non-foreground pixel — gets `output_background_value`.
///
/// Output is always [`PixelId::UInt8`] (`output_pixel_type: uint8_t` in
/// `SimpleContourExtractorImageFilter.yaml`), and the four values are cast to
/// their respective pixel types before use (`pixeltype: Input` / `Output` in the
/// yaml, which SimpleITK renders as `static_cast<…ImageType::PixelType>`), so
/// `input_foreground_value = 1.5` on an integer image really does become `1`.
///
/// Two literal details from the `.hxx`, both load-bearing:
///
/// * the "at least one background neighbour" scan runs over the **whole**
///   neighbourhood, `for (i = 0; i < neighborhoodSize; ++i)` — the *center*
///   pixel is included. So when `input_foreground_value ==
///   input_background_value`, every foreground pixel is its own background
///   neighbour and the entire foreground is marked.
/// * `radius` is per axis, and a `radius` of all zeros leaves only the center
///   in the window — again yielding the whole foreground under the equal-values
///   case above, and nothing otherwise.
///
/// Errors if `radius.len() != image.dimension()`.
pub fn simple_contour_extractor(
    image: &Image,
    input_foreground_value: f64,
    input_background_value: f64,
    radius: &[usize],
    output_foreground_value: f64,
    output_background_value: f64,
) -> Result<Image> {
    let dim = image.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }
    let output_foreground_value = u8::from_f64(output_foreground_value);
    let output_background_value = u8::from_f64(output_background_value);
    let input_foreground_value = quantize_to_pixel_type(image.pixel_id(), input_foreground_value);
    let input_background_value = quantize_to_pixel_type(image.pixel_id(), input_background_value);
    dispatch_scalar!(
        image.pixel_id(),
        simple_contour_extractor_typed,
        image,
        input_foreground_value,
        input_background_value,
        radius,
        output_foreground_value,
        output_background_value
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn img_u16(size: &[usize], data: Vec<u16>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- binary_contour ----

    /// A solid 3x3 block: the contour is the block's own outer ring (inner
    /// boundary), and the fully-interior center pixel drops to background.
    #[test]
    fn binary_contour_single_blob_is_its_inner_one_pixel_boundary() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]);
        let out = binary_contour(&image, 1.0, 0.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]);
    }

    /// A "plus" shape: the four arm tips are always face-adjacent to
    /// background (contour regardless of connectivity), but the center
    /// pixel's only background neighbors are diagonal (the four corners) --
    /// hand-traced against `CompareLines`: `fully_connected=false` gives the
    /// center pixel tolerance 0 on the cross-line comparison to the row
    /// above/below (whose only background runs are those corners, not
    /// aligned with the center column), so it never gets marked;
    /// `fully_connected=true` pads those background runs by 1, which now
    /// reaches the center column.
    #[test]
    fn binary_contour_fully_connected_changes_a_diagonal_touch() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            0, 1, 0,
            1, 1, 1,
            0, 1, 0,
        ]);
        let face = binary_contour(&image, 1.0, 0.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u8>().unwrap(), &[
            0, 1, 0,
            1, 0, 1,
            0, 1, 0,
        ]);

        let full = binary_contour(&image, 1.0, 0.0, true).unwrap();
        assert_eq!(
            full.scalar_slice::<u8>().unwrap(),
            &[0, 1, 0, 1, 1, 1, 0, 1, 0]
        );
    }

    /// Background pixels keep their own original value verbatim; only
    /// foreground pixels are ever rewritten (module docs' asymmetric-output
    /// quirk).
    #[test]
    fn binary_contour_background_pixels_pass_through_unchanged() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 1], vec![5, 1, 9]);
        let out = binary_contour(&image, 1.0, 0.0, false).unwrap();
        // The single foreground pixel is face-adjacent to both neighbors,
        // so it's a contour pixel; the two background pixels keep 5 and 9,
        // not background_value (0).
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[5, 1, 9]);
    }

    /// A foreground blob that fills the entire image has no background
    /// pixels to compare against at all, so nothing is ever marked --
    /// output is entirely `background_value` (a genuine upstream quirk: the
    /// image border is never treated as an implicit background neighbor).
    #[test]
    fn binary_contour_all_foreground_image_has_no_contour() {
        let image = img_u8(&[3, 3], vec![1; 9]);
        let out = binary_contour(&image, 1.0, 7.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[7u8; 9]);
    }

    /// N-D smoke test: a solid 3x3x3 cube in a 5x5x5 volume keeps its outer
    /// shell and drops only the fully-interior center voxel.
    #[test]
    fn binary_contour_3d_solid_cube_keeps_shell_drops_center() {
        let idx3 = |x: usize, y: usize, z: usize| x + 5 * (y + 5 * z);
        let size = [5usize, 5, 5];
        let total = 125;
        let mut data = vec![0u8; total];
        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    data[idx3(x, y, z)] = 1;
                }
            }
        }
        let image = img_u8(&size, data);
        let out = binary_contour(&image, 1.0, 0.0, false).unwrap();
        let out_data = out.scalar_slice::<u8>().unwrap();
        // center voxel (2,2,2): fully interior, becomes background.
        assert_eq!(out_data[idx3(2, 2, 2)], 0);
        // a face-center shell voxel, e.g. (2,2,1): still foreground.
        assert_eq!(out_data[idx3(2, 2, 1)], 1);
        // a corner shell voxel of the cube, (1,1,1): still foreground.
        assert_eq!(out_data[idx3(1, 1, 1)], 1);
    }

    #[test]
    fn binary_contour_rejects_float_pixel_type() {
        let image = Image::from_vec(&[2, 2], vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(
            binary_contour(&image, 1.0, 0.0, false).unwrap_err(),
            FilterError::RequiresIntegerPixelType(PixelId::Float32)
        );
    }

    // ---- label_contour ----

    /// Two adjacent 3x3-thick label blocks: each block's edge pixels keep
    /// their own label, each block's single fully-interior column becomes
    /// background, and the surrounding background stays background.
    #[test]
    fn label_contour_keeps_each_labels_own_id_on_the_contour() {
        #[rustfmt::skip]
        let image = img_u16(&[8, 5], vec![
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 2, 2, 2, 0,
            0, 1, 1, 1, 2, 2, 2, 0,
            0, 1, 1, 1, 2, 2, 2, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let out = label_contour(&image, 0.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u16>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 2, 2, 2, 0,
            0, 1, 0, 1, 2, 0, 2, 0,
            0, 1, 1, 1, 2, 2, 2, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]);
    }

    #[test]
    fn label_contour_rejects_float_pixel_type() {
        let image = Image::from_vec(&[2, 2], vec![1.0f64, 0.0, 0.0, 2.0]).unwrap();
        assert_eq!(
            label_contour(&image, 0.0, false).unwrap_err(),
            FilterError::RequiresIntegerPixelType(PixelId::Float64)
        );
    }

    // ---- binary_pruning ----

    /// A 3x3 blob with a 6-pixel horizontal spur; each iteration removes
    /// exactly one pixel from the spur's free tip (hand-traced against
    /// `ComputePruneImage`'s sequential, single-buffer raster-order
    /// mutation -- see module docs), so 3 iterations shortens the spur by
    /// exactly 3 pixels and leaves the blob untouched.
    #[test]
    fn binary_pruning_shortens_a_spur_by_exactly_iteration_pixels() {
        #[rustfmt::skip]
        let image = img_u8(&[10, 3], vec![
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let out = binary_pruning(&image, 3).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        ]);
    }

    /// A spur-free solid blob is a fixed point of pruning at any iteration
    /// count -- every pixel always has at least 2 active neighbors.
    #[test]
    fn binary_pruning_fixed_point_on_spur_free_shape() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ]);
        let few = binary_pruning(&image, 3).unwrap();
        let many = binary_pruning(&image, 100).unwrap();
        assert_eq!(few.scalar_slice::<u8>().unwrap(), &[1u8; 9]);
        assert_eq!(many.scalar_slice::<u8>().unwrap(), &[1u8; 9]);
    }

    #[test]
    fn binary_pruning_zero_iterations_is_identity() {
        let image = img_u8(&[3, 1], vec![1, 0, 1]);
        let out = binary_pruning(&image, 0).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn binary_pruning_rejects_non_2d_input() {
        let image = img_u8(&[2, 2, 2], vec![1; 8]);
        assert_eq!(
            binary_pruning(&image, 3).unwrap_err(),
            FilterError::UnsupportedPruningDimension(3)
        );
    }

    #[test]
    fn binary_pruning_rejects_non_uint8_pixel_type() {
        let image = img_u16(&[2, 2], vec![1, 0, 0, 1]);
        assert_eq!(
            binary_pruning(&image, 3).unwrap_err(),
            FilterError::RequiresUInt8PixelType(PixelId::UInt16)
        );
    }

    // ---- simple_contour_extractor ----

    /// A 9x9 image whose `[2, 6] x [2, 6]` block is foreground.
    fn filled_square() -> Image {
        let mut data = vec![0u8; 81];
        for y in 2..7 {
            for x in 2..7 {
                data[y * 9 + x] = 1;
            }
        }
        img_u8(&[9, 9], data)
    }

    /// `radius = 1`: the contour of a filled 5x5 square is its one-pixel border
    /// ring -- a foreground pixel is marked exactly when its 3x3 box reaches a
    /// background pixel, i.e. when `x` or `y` is on the block's edge. Every
    /// pixel pinned.
    #[test]
    fn simple_contour_extractor_of_a_filled_square_is_its_border_ring() {
        let out = simple_contour_extractor(&filled_square(), 1.0, 0.0, &[1, 1], 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
    }

    /// `radius = 2` thickens the ring inward by one pixel: only the block's
    /// exact centre `(4, 4)` keeps an all-foreground 5x5 box.
    #[test]
    fn simple_contour_extractor_radius_two_thickens_the_ring() {
        let out = simple_contour_extractor(&filled_square(), 1.0, 0.0, &[2, 2], 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
    }

    /// A per-axis `radius` reaches only along the axes it is nonzero on: with
    /// `[1, 0]` the 5x5 block's marked pixels are its left and right columns.
    #[test]
    fn simple_contour_extractor_radius_is_per_axis() {
        let out = simple_contour_extractor(&filled_square(), 1.0, 0.0, &[1, 0], 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
    }

    /// "Foreground" and "background" are two independent value tests, not a
    /// binary partition: `7` is neither, so it never marks a neighbour and is
    /// itself written as output background. The `1`-column touching only `7`s
    /// stays unmarked; the one touching a `0` is marked.
    #[test]
    fn simple_contour_extractor_foreground_and_background_values_are_independent() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 3], vec![
            7, 1, 7, 1, 0,
            7, 1, 7, 1, 0,
            7, 1, 7, 1, 0,
        ]);
        let out = simple_contour_extractor(&image, 1.0, 0.0, &[1, 1], 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 1, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 1, 0,
        ]);
    }

    /// The output values are plumbed through verbatim (cast to `uint8_t`), and
    /// they are not required to be `1`/`0`.
    #[test]
    fn simple_contour_extractor_output_values_are_plumbed_through() {
        let out =
            simple_contour_extractor(&filled_square(), 1.0, 0.0, &[1, 1], 255.0, 3.0).unwrap();
        let data = out.scalar_slice::<u8>().unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        assert_eq!(data[2 * 9 + 2], 255); // ring corner
        assert_eq!(data[4 * 9 + 4], 3); // block interior
        assert_eq!(data[0], 3); // outside the block
    }

    /// Output is `UInt8` whatever the input pixel type, and the *input* values
    /// are cast to the input pixel type first (`pixeltype: Input`), so `1.9`
    /// and `1.0` name the same foreground on an integer image.
    #[test]
    fn simple_contour_extractor_casts_input_values_to_the_input_pixel_type() {
        let image = img_u16(&[5, 3], vec![0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]);
        let exact = simple_contour_extractor(&image, 1.0, 0.0, &[1, 1], 1.0, 0.0).unwrap();
        let truncated = simple_contour_extractor(&image, 1.9, 0.4, &[1, 1], 1.0, 0.0).unwrap();
        assert_eq!(exact.pixel_id(), PixelId::UInt8);
        assert_eq!(
            exact.scalar_slice::<u8>().unwrap(),
            truncated.scalar_slice::<u8>().unwrap()
        );
    }

    /// The boundary is `ZeroFluxNeumannBoundaryCondition`, not a background
    /// halo: a foreground band flush against the left edge sees only replicated
    /// foreground there, so its own left column is *not* a contour. Only the
    /// column facing the real background is marked.
    #[test]
    fn simple_contour_extractor_border_replicates_rather_than_supplying_background() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            1, 1, 0,
            1, 1, 0,
            1, 1, 0,
        ]);
        let out = simple_contour_extractor(&image, 1.0, 0.0, &[1, 1], 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
        ]);
    }

    /// An all-foreground image has no background pixel anywhere and the border
    /// replicates, so nothing is ever marked.
    #[test]
    fn simple_contour_extractor_all_foreground_image_has_no_contour() {
        let image = img_u8(&[3, 3], vec![1; 9]);
        let out = simple_contour_extractor(&image, 1.0, 0.0, &[1, 1], 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0u8; 9]);
    }

    /// An all-background image has no foreground pixel, so every pixel takes the
    /// `else` branch and is written as output background.
    #[test]
    fn simple_contour_extractor_all_background_image_has_no_contour() {
        let image = img_u8(&[3, 3], vec![0; 9]);
        let out = simple_contour_extractor(&image, 1.0, 0.0, &[1, 1], 1.0, 9.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[9u8; 9]);
    }

    /// The neighbour scan includes the centre pixel (`i` runs over the whole
    /// neighbourhood), so `input_foreground == input_background` marks every
    /// foreground pixel -- even one whose entire box is foreground, and even at
    /// `radius = 0`, where the box *is* the centre.
    #[test]
    fn simple_contour_extractor_equal_foreground_and_background_marks_all_foreground() {
        let image = img_u8(&[3, 3], vec![1, 0, 1, 0, 1, 0, 1, 0, 1]);
        for radius in [[1, 1], [0, 0]] {
            let out = simple_contour_extractor(&image, 1.0, 1.0, &radius, 1.0, 0.0).unwrap();
            assert_eq!(
                out.scalar_slice::<u8>().unwrap(),
                &[1, 0, 1, 0, 1, 0, 1, 0, 1],
                "radius {radius:?}"
            );
        }
    }

    /// With distinct values and `radius = 0` the window holds only the centre,
    /// which is foreground and therefore never background: nothing is marked.
    #[test]
    fn simple_contour_extractor_zero_radius_marks_nothing() {
        let out = simple_contour_extractor(&filled_square(), 1.0, 0.0, &[0, 0], 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0u8; 81]);
    }

    #[test]
    fn simple_contour_extractor_rejects_wrong_radius_length() {
        let image = img_u8(&[3, 3], vec![1; 9]);
        assert_eq!(
            simple_contour_extractor(&image, 1.0, 0.0, &[1], 1.0, 0.0).unwrap_err(),
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }
}
