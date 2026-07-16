//! Connected-component labeling and its downstream label filters.
//!
//! Bit-faithful (in the sense of "same equivalence classes, same output
//! numbering") ports of:
//!
//! - `itk::ConnectedComponentImageFilter` (`itkConnectedComponentImageFilter.h`
//!   / `.hxx`, plus the scanline machinery in `itkScanlineFilterCommon.h` and
//!   the neighbor-activation helpers in `itkConnectedComponentAlgorithm.h`).
//! - `itk::RelabelComponentImageFilter` (`itkRelabelComponentImageFilter.h` /
//!   `.hxx`).
//! - `itk::LabelStatisticsImageFilter` (`itkLabelStatisticsImageFilter.h` /
//!   `.hxx`).
//!
//! ## `connected_component`
//!
//! ITK's filter is a two-pass scanline algorithm: pass 1 run-length-encodes
//! every scanline (a maximal run of nonzero pixels along axis 0) and assigns
//! each run a temporary label, in the order runs are discovered — i.e. line
//! by line, left to right within a line. Pass 2 walks every line's "previous"
//! neighbor lines (within Chebyshev distance 1 over the non-scanline axes)
//! and unions the temporary labels of any two runs whose x-extents overlap —
//! exactly, for face connectivity, or within a 1-pixel tolerance, for full
//! connectivity (the tolerance is what lets diagonally-touching pixels join).
//! `ConnectedComponentImageFilter::CreateConsecutive` then walks temporary
//! labels `1..=N` in increasing order and assigns output label `k` to the
//! `k`-th *new* union-find root encountered, which is exactly the object
//! whose first pixel (in raster-scan order) appears earliest.
//!
//! ITK's own union-find (`ScanlineFilterCommon::LookupSet`/`LinkLabels`) does
//! no path compression and unions by always making the *smaller* temporary
//! label the new root, which happens to make "root" and "earliest-appearing
//! member" coincide. This port instead uses a standard union-find with path
//! compression and union by rank (no recursion — a flood fill over a real
//! volume would blow the stack). This produces the identical *partition* of
//! runs into components (union-find implementations agree on connectivity
//! regardless of internal bookkeeping), and the final output-label
//! assignment is done the same way ITK's is: scan temporary labels `0..N` in
//! increasing order and number each newly-seen component root as it is first
//! encountered. Because that scan order is exactly the run-discovery
//! (raster-scan) order, the result is ITK's "labels assigned in raster-scan
//! order of first appearance" regardless of which run ends up as the
//! union-find's internal root.
//!
//! Background is always pixel value 0 in and out (SimpleITK's
//! `ConnectedComponentImageFilter.yaml` does not expose `BackgroundValue`,
//! matching ITK's default). Output pixel type is `UInt32`
//! (`output_pixel_type: uint32_t` in the yaml).
//!
//! ## `relabel_component`
//!
//! Counts pixels per nonzero label. With `sort_by_object_size == true`
//! (ITK's default) it sorts descending by count with ties broken by ascending
//! original label value (`itkRelabelComponentImageFilter.hxx`'s `std::sort`
//! comparator, applied to a `std::map`-ordered — i.e. ascending-label —
//! initial vector) and remaps the largest object to label 1, the second
//! largest to label 2, etc. With `false`, ITK skips that `std::sort`, so the
//! objects keep the `std::map` ascending-original-label order. Objects smaller
//! than
//! `minimum_object_size` map to background (0) without consuming an output
//! label. `minimum_object_size == 0` means no minimum, matching ITK's
//! `MinimumObjectSize` default. Output pixel type matches the input's
//! (`RelabelComponentImageFilter.yaml` has no `output_pixel_type`, so
//! SimpleITK's codegen leaves `OutputImageType = InputImageType`).
//!
//! ## `label_statistics`
//!
//! Per-label min/max/mean/variance/sigma/sum/count/bounding-box, matching
//! `LabelStatisticsImageFilter::AfterStreamedGenerateData`. That filter
//! computes variance as `(sumOfSquares - sum²/count) / (count - 1)` — the
//! **sample** variance (divisor `n - 1`), the same convention already used by
//! [`crate::filters::Statistics`]. Label 0 is not special here (unlike
//! `RelabelComponentImageFilter`, this filter has no notion of "background");
//! every label value present in `label_img` gets an entry.

use crate::core::{Image, PixelId};
use crate::filters::error::{FilterError, Result};
use crate::filters::geometry::require_same_physical_space;
use crate::filters::image_from_f64;
use std::collections::{BTreeMap, HashMap};

// ---- union-find -------------------------------------------------------

/// A disjoint-set structure over `0..n`, with path compression and union by
/// rank. Iterative throughout, so it cannot stack-overflow the way a
/// recursive flood fill over a large image's neighbor graph would.
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// The representative of `x`'s set, compressing the path just walked.
    pub(crate) fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while cur != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    /// Merge the sets containing `a` and `b`.
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

// ---- shared N-D indexing helpers --------------------------------------

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

/// All neighbor-line offset vectors over the "outer" axes (every axis except
/// the scanline axis 0), within Chebyshev distance 1 and excluding the
/// all-zero (same-line) offset. For face connectivity, restricted to
/// single-axis offsets — matching `setConnectivity`/`setConnectivityPrevious`
/// in `itkConnectedComponentAlgorithm.h`, which activate only axis-aligned
/// neighbors when `!FullyConnected`.
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

// ---- the shared scanline connected-components core -----------------------

/// One maximal run of foreground pixels along axis 0.
pub(crate) struct Run {
    pub(crate) start: usize,
    pub(crate) len: usize,
    /// 0-based temporary label (== this run's union-find index).
    pub(crate) label: usize,
}

/// The output of `itkScanlineFilterCommon`'s two passes: every scanline's runs,
/// and the union-find whose classes are the connected components.
///
/// Shared by [`connected_component`] and
/// [`crate::filters::label_map::binary_image_to_label_map`], which are the same ITK
/// algorithm (`itkConnectedComponentImageFilter` and
/// `itkBinaryImageToLabelMapFilter` both derive from `ScanlineFilterCommon`)
/// differing only in how the foreground mask is derived and how the resulting
/// classes are numbered and materialised.
pub(crate) struct ScanlineComponents {
    /// `line_map[line_id]` — the runs of scanline `line_id`, left to right.
    pub(crate) line_map: Vec<Vec<Run>>,
    pub(crate) uf: UnionFind,
    pub(crate) num_runs: usize,
}

/// `ScanlineFilterCommon`'s pass 1 (`DynamicThreadedGenerateData`) and pass 2
/// (`ComputeEquivalence`) over a precomputed foreground mask.
///
/// Pass 1 run-length encodes every scanline; temporary labels are handed out in
/// raster order of first appearance (line by line, left to right within a
/// line), matching `ScanlineFilterCommon::InitUnion`. Pass 2 unions runs on
/// neighboring lines whose x-extents overlap — exactly, for face connectivity;
/// within a 1-pixel tolerance, for full connectivity (`itkScanlineFilterCommon.h`'s
/// `CompareLines`).
pub(crate) fn scanline_components(
    is_fg: &[bool],
    size: &[usize],
    fully_connected: bool,
) -> ScanlineComponents {
    let dim = size.len();
    let total: usize = size.iter().product();
    let xsize = size[0];
    let linecount = if total == 0 { 0 } else { total / xsize };

    let mut line_map: Vec<Vec<Run>> = Vec::with_capacity(linecount);
    let mut num_runs = 0usize;
    for line in 0..linecount {
        let base = line * xsize;
        let mut runs = Vec::new();
        let mut x = 0usize;
        while x < xsize {
            if is_fg[base + x] {
                let start = x;
                while x < xsize && is_fg[base + x] {
                    x += 1;
                }
                runs.push(Run {
                    start,
                    len: x - start,
                    label: num_runs,
                });
                num_runs += 1;
            } else {
                x += 1;
            }
        }
        line_map.push(runs);
    }

    let mut uf = UnionFind::new(num_runs);
    if dim > 1 {
        let outer_size = &size[1..];
        let outer_dim = outer_size.len();
        let outer_strides = strides(outer_size);
        let neighbor_offsets = line_neighbor_offsets(outer_dim, fully_connected);
        let tol: i64 = if fully_connected { 1 } else { 0 };

        for line_id in 0..linecount {
            if line_map[line_id].is_empty() {
                continue;
            }
            let idx = multi_index(line_id, outer_size, &outer_strides);
            for offset in &neighbor_offsets {
                let mut neighbor_idx = Vec::with_capacity(outer_dim);
                let mut in_bounds = true;
                for d in 0..outer_dim {
                    let v = idx[d] as i64 + offset[d];
                    if v < 0 || v as usize >= outer_size[d] {
                        in_bounds = false;
                        break;
                    }
                    neighbor_idx.push(v as usize);
                }
                if !in_bounds {
                    continue;
                }
                let neighbor_line: usize = neighbor_idx
                    .iter()
                    .zip(outer_strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                // Only process each unordered line pair once; correctness
                // does not depend on which side does the processing.
                if neighbor_line >= line_id || line_map[neighbor_line].is_empty() {
                    continue;
                }
                for a in &line_map[line_id] {
                    let a_start = a.start as i64;
                    let a_end = (a.start + a.len - 1) as i64;
                    for b in &line_map[neighbor_line] {
                        let b_start = b.start as i64;
                        let b_end = (b.start + b.len - 1) as i64;
                        if a_start <= b_end + tol && b_start <= a_end + tol {
                            uf.union(a.label, b.label);
                        }
                    }
                }
            }
        }
    }

    ScanlineComponents {
        line_map,
        uf,
        num_runs,
    }
}

/// `ScanlineFilterCommon::CreateConsecutive` (`itkScanlineFilterCommon.h:199-228`):
/// number the components `0, 1, 2, …` in ascending temporary-label order,
/// skipping `background` exactly once when the counter reaches it. Returns the
/// per-root output label (only entries at union-find roots are meaningful) and
/// the number of components.
///
/// Because temporary labels are handed out in raster order of first appearance,
/// this numbers each component by where its first pixel appears. ITK's own
/// union-find always points the larger label at the smaller, making its roots
/// the earliest-appearing member and its `m_UnionFind[i] == i` root test
/// equivalent to the `find`-based first-seen test used here.
pub(crate) fn create_consecutive(
    components: &mut ScanlineComponents,
    background: i64,
) -> (Vec<i64>, u64) {
    let mut root_to_output: Vec<i64> = vec![0; components.num_runs];
    let mut assigned = vec![false; components.num_runs];
    let mut consecutive: i64 = 0;
    let mut count: u64 = 0;
    for i in 0..components.num_runs {
        let root = components.uf.find(i);
        if !assigned[root] {
            if consecutive == background {
                consecutive += 1;
            }
            root_to_output[root] = consecutive;
            assigned[root] = true;
            consecutive += 1;
            count += 1;
        }
    }
    (root_to_output, count)
}

// ---- connected_component -----------------------------------------------

/// `ConnectedComponentImageFilter`: label the connected components of the
/// nonzero pixels in `img`. `fully_connected = false` is face connectivity
/// (4-connected in 2-D, 6-connected in 3-D); `true` is full connectivity
/// (8-connected in 2-D, 26-connected in 3-D). Output pixel type is `UInt32`;
/// background stays 0 and object labels are consecutive starting at 1,
/// assigned in raster-scan order of first appearance.
///
/// # The optional mask
///
/// SimpleITK exposes an optional `MaskImage` input (`UInt8`), and ITK implements
/// it by running the *input* through `MaskImageFilter` before labeling anything
/// (`itkConnectedComponentImageFilter.hxx:79-92`). `MaskImageFilter` replaces a
/// voxel with its outside value (`0`) where the mask **equals** the masking value
/// — which is `TMask{}`, i.e. **`0`** (`itkMaskImageFilter.h:55`, `:107`) — so a
/// masked-out voxel arrives at the labeler as a zero, i.e. as background.
///
/// Hence: **a voxel is foreground iff it is nonzero *and* its mask voxel is
/// nonzero.** Note the polarity — every mask value except `0` keeps the voxel, so
/// a mask of all `1`s is a no-op here. That is the opposite of the threshold
/// family's mask ([`crate::filters::histogram::ThresholdMask`]), which admits a voxel only
/// where the mask **equals** `mask_value` (default **255**), and a mask of all
/// `1`s admits nothing. Two upstream classes, two conventions; see
/// `crate::filters::mask_input` and ledger §2.175.
///
/// The mask must be `UInt8`, the image's size, and on the image's grid — ITK's
/// three preconditions for a mask *input*, enforced by
/// `crate::filters::mask_input::uint8_mask_voxels`.
pub fn connected_component(
    img: &Image,
    mask: Option<&Image>,
    fully_connected: bool,
) -> Result<Image> {
    let size = img.size();
    let total: usize = size.iter().product();

    // Before the empty-image shortcut: ITK validates a mask input in
    // `VerifyInputInformation`, i.e. before `GenerateData` looks at a single voxel.
    let mask_voxels = match mask {
        None => None,
        Some(m) => Some(crate::filters::mask_input::uint8_mask_voxels(img, m)?),
    };

    if total == 0 {
        let mut result = Image::from_vec(size, Vec::<u32>::new())?;
        result.copy_geometry_from(img);
        return Ok(result);
    }

    let xsize = size[0];
    let vals = img.to_f64_vec()?;
    let is_fg: Vec<bool> = match mask_voxels {
        None => vals.iter().map(|&v| v != 0.0).collect(),
        Some(m) => vals
            .iter()
            .zip(m)
            .map(|(&v, &m)| v != 0.0 && m != 0)
            .collect(),
    };

    let mut components = scanline_components(&is_fg, size, fully_connected);
    // `CreateConsecutive(0)` skips 0 on its first assignment, so the object
    // labels run `1..=N` — this filter's documented numbering.
    let (root_to_output, _) = create_consecutive(&mut components, 0);

    let mut out = vec![0u32; total];
    for (line, runs) in components.line_map.iter().enumerate() {
        let base = line * xsize;
        for run in runs {
            let root = components.uf.find(run.label);
            let label = root_to_output[root] as u32;
            out[base + run.start..base + run.start + run.len].fill(label);
        }
    }

    let mut result = Image::from_vec(size, out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

// ---- relabel_component ---------------------------------------------------

/// The largest label value representable by an integer pixel type — the
/// bound `RelabelComponentImageFilter` checks kept-object counts against
/// (`NumericTraits<OutputPixelType>::max()`) before it would need to emit
/// another output label.
fn pixel_id_max_label(id: PixelId) -> u64 {
    match id {
        PixelId::UInt8 | PixelId::VectorUInt8 => u8::MAX as u64,
        PixelId::Int8 | PixelId::VectorInt8 => i8::MAX as u64,
        PixelId::UInt16 | PixelId::VectorUInt16 => u16::MAX as u64,
        PixelId::Int16 | PixelId::VectorInt16 => i16::MAX as u64,
        PixelId::UInt32 | PixelId::VectorUInt32 => u32::MAX as u64,
        PixelId::Int32 | PixelId::VectorInt32 => i32::MAX as u64,
        PixelId::UInt64 | PixelId::VectorUInt64 => u64::MAX,
        PixelId::Int64 | PixelId::VectorInt64 => i64::MAX as u64,
        PixelId::Float32 | PixelId::ComplexFloat32 | PixelId::VectorFloat32 => u64::MAX,
        PixelId::Float64 | PixelId::ComplexFloat64 | PixelId::VectorFloat64 => u64::MAX,
    }
}

/// `RelabelComponentImageFilter`: relabel the (background-0) objects in `img`.
///
/// With `sort_by_object_size == true` (ITK's default), label 1 is the largest
/// object, label 2 the second largest, etc., descending by pixel count; ties
/// are broken by ascending original label value. With `false`, ITK skips the
/// size sort and the objects keep their ascending original-label order.
/// Objects with fewer than `minimum_object_size` pixels are dropped to
/// background; `minimum_object_size == 0` means no minimum.
pub fn relabel_component(
    img: &Image,
    minimum_object_size: u64,
    sort_by_object_size: bool,
) -> Result<Image> {
    let labels: Vec<i64> = img
        .to_f64_vec()?
        .iter()
        .map(|&v| v.round() as i64)
        .collect();

    // `std::map<LabelType, ...>` iterates in ascending-key order, so the
    // pre-sort vector ITK builds is already in ascending original-label
    // order; a `BTreeMap` reproduces that here.
    let mut size_map: BTreeMap<i64, u64> = BTreeMap::new();
    for &label in &labels {
        if label != 0 {
            *size_map.entry(label).or_insert(0) += 1;
        }
    }

    let mut size_vec: Vec<(i64, u64)> = size_map.into_iter().collect();
    // ITK sorts descending by size only when `m_SortByObjectSize`; ties keep
    // ascending original-label order (`a.second.m_SizeInPixels >
    // b.second.m_SizeInPixels || (sizes equal && a.first < b.first)` in
    // `itkRelabelComponentImageFilter.hxx`). When false, ITK leaves the
    // `std::map` ascending-original-label order untouched — reproduced here by
    // the `BTreeMap` order `size_vec` already carries.
    if sort_by_object_size {
        size_vec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    }

    let max_output_label = pixel_id_max_label(img.pixel_id());
    let mut relabel_map: HashMap<i64, u64> = HashMap::new();
    let mut next_output = 0u64;
    for (label, size) in size_vec {
        if minimum_object_size > 0 && size < minimum_object_size {
            relabel_map.insert(label, 0);
        } else {
            if next_output == max_output_label {
                return Err(FilterError::TooManyObjects {
                    max: max_output_label,
                });
            }
            next_output += 1;
            relabel_map.insert(label, next_output);
        }
    }

    let out_vals: Vec<f64> = labels
        .iter()
        .map(|&l| {
            if l == 0 {
                0.0
            } else {
                *relabel_map.get(&l).expect("every nonzero label was sized") as f64
            }
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out_vals)
}

// ---- label_statistics ------------------------------------------------

/// Per-label statistics, mirroring `LabelStatisticsImageFilter`.
///
/// Variance is the **sample** variance (divisor `n - 1`), the same
/// convention as [`crate::filters::Statistics`] and matching this filter's own
/// `(sumOfSquares - sum²/count) / (count - 1)` in
/// `itkLabelStatisticsImageFilter.hxx`.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelStatistics {
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
    pub variance: f64,
    pub sigma: f64,
    pub sum: f64,
    pub count: u64,
    /// Inclusive `(min_index, max_index)` per axis, in `img`'s axis order.
    pub bounding_box: Vec<(i64, i64)>,
    /// The **histogram** median: the centre of the bin in which the label's
    /// cumulative frequency first passes half its count
    /// (`itkLabelStatisticsImageFilter.hxx:410-441`). It is *not* the exact median
    /// of the values, except for 8-bit input — see [`label_statistics`] for the bin
    /// range, which is what decides the difference.
    pub median: f64,
}

struct Accum {
    count: u64,
    min: f64,
    max: f64,
    sum: f64,
    sum_sq: f64,
    bbox: Vec<(i64, i64)>,
    histogram: Vec<u64>,
}

/// SimpleITK's `LabelStatisticsImageFilter.yaml` `custom_itk_cast` calls
/// `SetHistogramParameters(256, ...)` on both branches, and ITK's own default is 256
/// (`itkLabelStatisticsImageFilter.hxx:36`).
const HISTOGRAM_BINS: usize = 256;

/// The bin range SimpleITK hands `SetHistogramParameters`, and the whole reason the
/// median is exact for one pixel type and approximate for every other.
///
/// `LabelStatisticsImageFilter.yaml`'s `custom_itk_cast` branches on the *intensity*
/// image's pixel type: for `uint8_t`/`int8_t` it pads the **pixel type's** range by half
/// a unit — `[min - 0.5, max + 0.5]` — which with 256 bins puts exactly one integer in
/// each bin, so the "histogram median" is the true median. ("NOTE: This is a heuristic
/// that works exact median only for (unsigned) char images", says the yaml.) For every
/// other type it runs `MinimumMaximumImageFilter` over the **whole intensity image** —
/// not the label's voxels — and spreads 256 bins over that data range, so a label whose
/// values occupy a narrow part of the image's range lands in a handful of bins and the
/// median is quantized to a bin centre.
fn histogram_bounds(img: &Image) -> Result<(f64, f64)> {
    match img.pixel_id() {
        PixelId::UInt8 => Ok((-0.5, 255.5)),
        PixelId::Int8 => Ok((-128.5, 127.5)),
        _ => {
            let vals = img.to_f64_vec()?;
            Ok(crate::core::parallel::min_max(&vals).unwrap_or((0.0, 0.0)))
        }
    }
}

/// `itk::Statistics::Histogram::Initialize(size, lb, ub)`'s uniform bins — **computed in
/// `float`, not `double`** (`itkHistogram.hxx:222-234`: `float interval = (float(ub) -
/// float(lb)) / float(size)`, and every edge is `lb + float(j) * interval`). The last
/// bin's max is the upper bound verbatim. The `f32` is upstream's, is visible in the bin
/// centres a median reports, and is reproduced here rather than "cleaned up" to `f64`.
fn bin_edges(lower: f64, upper: f64) -> (Vec<f64>, Vec<f64>) {
    let interval = (upper as f32 - lower as f32) / HISTOGRAM_BINS as f32;
    let mut mins = Vec::with_capacity(HISTOGRAM_BINS);
    let mut maxs = Vec::with_capacity(HISTOGRAM_BINS);
    for j in 0..HISTOGRAM_BINS - 1 {
        mins.push(lower + (j as f32 * interval) as f64);
        maxs.push(lower + ((j as f32 + 1.0) * interval) as f64);
    }
    mins.push(lower + ((HISTOGRAM_BINS as f32 - 1.0) * interval) as f64);
    maxs.push(upper);
    (mins, maxs)
}

/// `Histogram::GetIndex` (`itkHistogram.hxx:243-321`) with `m_ClipBinsAtEnds` at its
/// default `true` (`itkHistogram.h:514`): below the first bin's min, or above the last
/// bin's max by more than the last endpoint itself, the measurement is **dropped** — it
/// is counted in no bin. Neither can happen for the two ranges `histogram_bounds`
/// produces (the data range contains every value by construction; the padded 8-bit range
/// contains every 8-bit value), so this returns `None` only for a caller that does not
/// exist yet — and it drops rather than clamps, which is what upstream does.
fn bin_of(v: f64, mins: &[f64], maxs: &[f64]) -> Option<usize> {
    let last = mins.len() - 1;
    if v < mins[0] {
        return None;
    }
    if v >= maxs[last] {
        return if v == maxs[last] { Some(last) } else { None };
    }
    // The bins are contiguous and monotone (`maxs[j] == mins[j + 1]` by construction),
    // so a floor followed by a correcting step lands on the same bin ITK's binary search
    // over these same edges would.
    let width = maxs[0] - mins[0];
    let mut b = if width > 0.0 {
        (((v - mins[0]) / width) as usize).min(last)
    } else {
        0
    };
    while b > 0 && v < mins[b] {
        b -= 1;
    }
    while b < last && v >= maxs[b] {
        b += 1;
    }
    Some(b)
}

/// `GetMedian` (`itkLabelStatisticsImageFilter.hxx:410-441`): walk bins while the running
/// total is `<= count / 2` (**integer** division — `m_Count` is an `IdentifierType`), step
/// back one, and return that bin's **centre**. Not an interpolated quantile, and not a
/// value that need occur in the image.
fn histogram_median(hist: &[u64], count: u64, mins: &[f64], maxs: &[f64]) -> f64 {
    let half = count / 2;
    let mut total: u64 = 0;
    let mut bin: usize = 0;
    while total <= half && bin < HISTOGRAM_BINS {
        total += hist[bin];
        bin += 1;
    }
    bin -= 1;
    mins[bin] + (maxs[bin] - mins[bin]) / 2.0
}

/// `LabelStatisticsImageFilter`: per-label min/max/mean/variance/sigma/sum/
/// count/bounding-box/median of `img`'s intensities, grouped by the integer labels
/// in `label_img` (same size as `img`). Every label value present in
/// `label_img` gets an entry, including 0 — this filter has no notion of
/// "background", unlike [`relabel_component`].
///
/// **The median is a 256-bin histogram median, and its bin range decides whether it is
/// exact.** SimpleITK sets `UseHistograms: true` by *default* and configures the bins in
/// `LabelStatisticsImageFilter.yaml`'s `custom_itk_cast`, so this filter always builds
/// them (there is no "histograms off" mode to mirror — the getter would simply return
/// `0.0`). The range is `histogram_bounds`: the padded **pixel-type** range for
/// `UInt8`/`Int8`, which puts one integer per bin and makes the median exact; the **whole
/// intensity image's** data range for every other type, which quantizes it to a bin
/// centre — and note *whole image*, not *this label's voxels*, so a label confined to a
/// narrow band of a wide-ranged image gets a coarse median. That is upstream's rule, and
/// it is the same class of pixel-type-dependent bin range as the 8-bit thresholds
/// (§2.174).
pub fn label_statistics(img: &Image, label_img: &Image) -> Result<BTreeMap<i64, LabelStatistics>> {
    if img.size() != label_img.size() {
        return Err(FilterError::SizeMismatch {
            a: img.size().to_vec(),
            b: label_img.size().to_vec(),
        });
    }
    require_same_physical_space(img, label_img, 1)?;

    let size = img.size();
    let dim = size.len();
    let strides_ = strides(size);
    let vals = img.to_f64_vec()?;
    let labels: Vec<i64> = label_img
        .to_f64_vec()?
        .iter()
        .map(|&v| v.round() as i64)
        .collect();

    let (lower, upper) = histogram_bounds(img)?;
    let (bin_mins, bin_maxs) = bin_edges(lower, upper);

    let mut acc: BTreeMap<i64, Accum> = BTreeMap::new();
    for flat in 0..vals.len() {
        let label = labels[flat];
        let v = vals[flat];
        let entry = acc.entry(label).or_insert_with(|| Accum {
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            bbox: vec![(i64::MAX, i64::MIN); dim],
            histogram: vec![0; HISTOGRAM_BINS],
        });
        entry.count += 1;
        entry.sum += v;
        entry.sum_sq += v * v;
        if let Some(bin) = bin_of(v, &bin_mins, &bin_maxs) {
            entry.histogram[bin] += 1;
        }
        entry.min = entry.min.min(v);
        entry.max = entry.max.max(v);
        let idx = multi_index(flat, size, &strides_);
        for (&i, bb) in idx.iter().zip(entry.bbox.iter_mut()) {
            let i = i as i64;
            *bb = (bb.0.min(i), bb.1.max(i));
        }
    }

    let mut out = BTreeMap::new();
    for (label, a) in acc {
        let mean = a.sum / a.count as f64;
        let variance = if a.count > 1 {
            (a.sum_sq - a.sum * a.sum / a.count as f64) / (a.count as f64 - 1.0)
        } else {
            0.0
        };
        out.insert(
            label,
            LabelStatistics {
                minimum: a.min,
                maximum: a.max,
                mean,
                variance,
                sigma: variance.max(0.0).sqrt(),
                sum: a.sum,
                count: a.count,
                bounding_box: a.bbox,
                median: histogram_median(&a.histogram, a.count, &bin_mins, &bin_maxs),
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn img_u32(size: &[usize], data: Vec<u32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- connected_component ----

    #[test]
    fn diagonal_touch_differs_by_connectivity_2d() {
        // . X
        // X .
        #[rustfmt::skip]
        let img = img_u8(&[2, 2], vec![
            0, 1,
            1, 0,
        ]);
        let face = connected_component(&img, None, false).unwrap();
        assert_eq!(face.pixel_id(), PixelId::UInt32);
        let face_labels = face.scalar_slice::<u32>().unwrap();
        assert_ne!(face_labels[1], face_labels[2]); // the two 1s, separate components
        assert_eq!(face_labels.iter().filter(|&&v| v != 0).count(), 2);

        let full = connected_component(&img, None, true).unwrap();
        let full_labels = full.scalar_slice::<u32>().unwrap();
        assert_eq!(full_labels[1], full_labels[2]); // joined diagonally
    }

    #[test]
    fn diagonal_touch_differs_by_connectivity_3d() {
        // Opposite corners of a 2x2x2 cube, touching only at a vertex.
        let mut data = vec![0u8; 8];
        data[0] = 1; // (0,0,0)
        data[7] = 1; // (1,1,1)
        let img = img_u8(&[2, 2, 2], data);

        let face = connected_component(&img, None, false).unwrap();
        let face_labels = face.scalar_slice::<u32>().unwrap();
        assert_ne!(face_labels[0], face_labels[7]);
        assert_eq!(
            face_labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3 // background(0) + 2 distinct object labels
        );

        let full = connected_component(&img, None, true).unwrap();
        let full_labels = full.scalar_slice::<u32>().unwrap();
        assert_eq!(full_labels[0], full_labels[7]);
        assert_ne!(full_labels[0], 0);
    }

    #[test]
    fn u_shape_merges_across_scanlines() {
        // Two vertical bars only connected by a bottom row that bridges the
        // gap between them — a naive one-pass algorithm assigns the left and
        // right bars different labels on rows 0-1 and never reconciles them.
        // X X . X X
        // X X . X X
        // X X X X X
        #[rustfmt::skip]
        let data = vec![
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 1, 1, 1,
        ];
        // Row-major (y-major) source above; convert to x-fastest storage.
        let w = 5;
        let h = 3;
        let mut xfastest = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                xfastest[y * w + x] = data[y * w + x];
            }
        }
        let img = img_u8(&[w, h], xfastest);
        let out = connected_component(&img, None, false).unwrap();
        let labels = out.scalar_slice::<u32>().unwrap();
        let distinct: std::collections::HashSet<u32> =
            labels.iter().copied().filter(|&v| v != 0).collect();
        assert_eq!(
            distinct.len(),
            1,
            "left and right bars must merge into one component via the bottom row"
        );
        // top-left pixel and top-right pixel end up with the same label
        assert_eq!(labels[0], labels[3]);
    }

    #[test]
    fn background_only_image_has_no_objects() {
        let img = img_u8(&[3, 3], vec![0; 9]);
        let out = connected_component(&img, None, false).unwrap();
        assert!(out.scalar_slice::<u32>().unwrap().iter().all(|&v| v == 0));
    }

    #[test]
    fn labels_assigned_in_raster_order_of_first_appearance() {
        // Two separate single-pixel objects; the one appearing earlier in
        // raster order (smaller flat index) must get the smaller label.
        #[rustfmt::skip]
        let img = img_u8(&[3, 2], vec![
            0, 0, 1,
            1, 0, 0,
        ]);
        let out = connected_component(&img, None, false).unwrap();
        let labels = out.scalar_slice::<u32>().unwrap();
        // flat index 2 = (x=2,y=0) comes before flat index 3 = (x=0,y=1)
        assert_eq!(labels[2], 1);
        assert_eq!(labels[3], 2);
    }

    // ---- relabel_component ----

    #[test]
    fn relabel_orders_by_descending_size_with_tie_break() {
        // labels: 1 (size 5), 2 (size 3), 3 (size 3, tie with 2), 4 (size 7)
        let mut data = vec![0u32; 18];
        for v in data.iter_mut().take(5) {
            *v = 1;
        }
        for v in data[5..8].iter_mut() {
            *v = 2;
        }
        for v in data[8..11].iter_mut() {
            *v = 3;
        }
        for v in data[11..18].iter_mut() {
            *v = 4;
        }
        let img = img_u32(&[18, 1], data);
        let out = relabel_component(&img, 0, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt32);
        let relabeled = out.scalar_slice::<u32>().unwrap();

        // expected: 4(7)->1, 1(5)->2, 2(3,tie,orig 2<3)->3, 3(3)->4
        assert!(relabeled[0..5].iter().all(|&v| v == 2)); // orig label 1
        assert!(relabeled[5..8].iter().all(|&v| v == 3)); // orig label 2
        assert!(relabeled[8..11].iter().all(|&v| v == 4)); // orig label 3
        assert!(relabeled[11..18].iter().all(|&v| v == 1)); // orig label 4
    }

    #[test]
    fn relabel_without_sort_keeps_ascending_original_label_order() {
        // Same objects as the sorted test: 1(5), 2(3), 3(3), 4(7).
        let mut data = vec![0u32; 18];
        for v in data.iter_mut().take(5) {
            *v = 1;
        }
        for v in data[5..8].iter_mut() {
            *v = 2;
        }
        for v in data[8..11].iter_mut() {
            *v = 3;
        }
        for v in data[11..18].iter_mut() {
            *v = 4;
        }
        let img = img_u32(&[18, 1], data);

        // SortByObjectSize = false: objects keep ascending original-label
        // order, so each label maps to itself (1->1, 2->2, 3->3, 4->4).
        let out = relabel_component(&img, 0, false).unwrap();
        let relabeled = out.scalar_slice::<u32>().unwrap();
        assert!(relabeled[0..5].iter().all(|&v| v == 1));
        assert!(relabeled[5..8].iter().all(|&v| v == 2));
        assert!(relabeled[8..11].iter().all(|&v| v == 3));
        assert!(relabeled[11..18].iter().all(|&v| v == 4));

        // And this differs from the size-sorted relabeling.
        let sorted = relabel_component(&img, 0, true).unwrap();
        assert_ne!(relabeled, sorted.scalar_slice::<u32>().unwrap(),);
    }

    #[test]
    fn relabel_drops_objects_below_minimum_size() {
        // label 1: size 2 (dropped, minimum 3); label 2: size 4 (kept -> 1)
        let mut data = vec![0u32; 6];
        data[0] = 1;
        data[1] = 1;
        data[2] = 2;
        data[3] = 2;
        data[4] = 2;
        data[5] = 2;
        let img = img_u32(&[6, 1], data);
        let out = relabel_component(&img, 3, true).unwrap();
        let relabeled = out.scalar_slice::<u32>().unwrap();
        assert_eq!(&relabeled[0..2], &[0, 0]);
        assert_eq!(&relabeled[2..6], &[1, 1, 1, 1]);
    }

    #[test]
    fn relabel_preserves_input_pixel_type() {
        let img = img_u8(&[4, 1], vec![1, 1, 0, 2]);
        let out = relabel_component(&img, 0, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    // ---- label_statistics ----

    #[test]
    fn label_statistics_hand_computed() {
        // intensities:  2  4  4  6  10  20
        // labels:       1  1  1  1   2   2
        let intensities =
            Image::from_vec(&[6, 1], vec![2.0f64, 4.0, 4.0, 6.0, 10.0, 20.0]).unwrap();
        let labels = img_u32(&[6, 1], vec![1, 1, 1, 1, 2, 2]);

        let stats = label_statistics(&intensities, &labels).unwrap();
        assert_eq!(stats.len(), 2);

        let l1 = &stats[&1];
        assert_eq!(l1.count, 4);
        assert_eq!(l1.minimum, 2.0);
        assert_eq!(l1.maximum, 6.0);
        assert_eq!(l1.mean, 4.0);
        assert_eq!(l1.sum, 16.0);
        // sample variance: ((2-4)^2+(4-4)^2+(4-4)^2+(6-4)^2)/(4-1) = 8/3
        assert!((l1.variance - 8.0 / 3.0).abs() < 1e-12);
        assert!((l1.sigma - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(l1.bounding_box, vec![(0, 3), (0, 0)]);

        let l2 = &stats[&2];
        assert_eq!(l2.count, 2);
        assert_eq!(l2.minimum, 10.0);
        assert_eq!(l2.maximum, 20.0);
        assert_eq!(l2.mean, 15.0);
        assert_eq!(l2.sum, 30.0);
        // sample variance: ((10-15)^2+(20-15)^2)/(2-1) = 50
        assert!((l2.variance - 50.0).abs() < 1e-12);
        assert_eq!(l2.bounding_box, vec![(4, 5), (0, 0)]);
    }

    #[test]
    fn label_statistics_2d_bounding_box() {
        #[rustfmt::skip]
        let intensities = Image::from_vec(&[3, 3], vec![
            1.0f64, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        #[rustfmt::skip]
        let labels = img_u32(&[3, 3], vec![
            1, 1, 0,
            1, 0, 0,
            0, 0, 2,
        ]);
        let stats = label_statistics(&intensities, &labels).unwrap();
        let l1 = &stats[&1];
        assert_eq!(l1.count, 3);
        assert_eq!(l1.sum, 1.0 + 2.0 + 4.0);
        assert_eq!(l1.bounding_box, vec![(0, 1), (0, 1)]);

        let l2 = &stats[&2];
        assert_eq!(l2.count, 1);
        assert_eq!(l2.bounding_box, vec![(2, 2), (2, 2)]);

        let bg = &stats[&0];
        assert_eq!(bg.count, 5);
    }

    /// 8-bit intensities get the padded **pixel-type** range `[-0.5, 255.5]`, so each of
    /// the 256 bins holds exactly one integer and the histogram median *is* the median.
    ///
    /// The even-count case also pins ITK's convention: `GetMedian` walks bins while the
    /// running total is `<= count / 2` and then steps back one
    /// (`itkLabelStatisticsImageFilter.hxx:427-434`), so for `[10, 20, 30, 40]` it reports
    /// the **upper** middle value, 30 — not the interpolated 25 a textbook median gives.
    #[test]
    fn label_statistics_median_is_exact_for_8_bit_and_takes_the_upper_middle() {
        let intensities = img_u8(&[5, 1], vec![10, 20, 30, 40, 100]);
        let labels = img_u32(&[5, 1], vec![1, 1, 1, 1, 1]);
        assert_eq!(
            label_statistics(&intensities, &labels).unwrap()[&1].median,
            30.0
        );

        // Even count: 10, 20, 30, 40 -> 30, not 25.
        let intensities = img_u8(&[4, 1], vec![10, 20, 30, 40]);
        let labels = img_u32(&[4, 1], vec![1, 1, 1, 1]);
        assert_eq!(
            label_statistics(&intensities, &labels).unwrap()[&1].median,
            30.0
        );
    }

    /// For any non-8-bit type the 256 bins are spread over the **whole intensity image's**
    /// data range — not the label's own values — so a label confined to a narrow band gets
    /// a median quantized to a coarse bin centre.
    ///
    /// Label 1 holds `0.0, 1.0, 2.0` (true median `1.0`) while label 2 holds a single
    /// `1000.0`, which is what stretches the range. Bin width is `1000/256 = 3.90625`, all
    /// three of label 1's values fall in bin 0, and the reported median is that bin's
    /// centre, `1.953125`. Scope the range to the label's own voxels instead and the
    /// answer moves to ≈`1.0` — which is the mutation this pin exists to catch.
    #[test]
    fn label_statistics_median_for_float_is_quantized_by_the_whole_images_range() {
        let intensities = Image::from_vec(&[4, 1], vec![0.0f64, 1.0, 2.0, 1000.0]).unwrap();
        let labels = img_u32(&[4, 1], vec![1, 1, 1, 2]);
        let stats = label_statistics(&intensities, &labels).unwrap();

        assert_eq!(
            stats[&1].median, 1.953125,
            "bin 0's centre, not the true median 1.0"
        );
        assert_ne!(stats[&1].median, 1.0);
        // The single 1000.0 sits at the upper bound, which `GetIndex` folds into the last
        // bin rather than dropping (`itkHistogram.hxx:275-284`).
        assert_eq!(stats[&2].median, (996.09375 + 1000.0) / 2.0);
    }

    /// ITK computes the histogram's bin interval and every bin edge in **`float`**
    /// (`itkHistogram.hxx:222-234`), so the edges — and therefore the bin centres a median
    /// reports — carry `f32` rounding even for a `double` histogram. Reproduced, not
    /// cleaned up: this asserts the edge differs from the `f64` computation.
    #[test]
    fn the_histogram_bin_edges_carry_itks_float_rounding() {
        let (lower, upper) = (0.0f64, 1.0f64 / 3.0);
        let (mins, _) = bin_edges(lower, upper);

        let interval = (upper as f32 - lower as f32) / HISTOGRAM_BINS as f32;
        let itk_edge = lower + (128.0f32 * interval) as f64;
        let f64_edge = lower + 128.0 * (upper - lower) / HISTOGRAM_BINS as f64;

        assert_eq!(mins[128], itk_edge);
        assert_ne!(
            mins[128], f64_edge,
            "the f32 interval is upstream's and must survive: {itk_edge} vs {f64_edge}"
        );
    }

    #[test]
    fn label_statistics_size_mismatch_errors() {
        let intensities = Image::from_vec(&[2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let labels = img_u32(&[3, 1], vec![1, 1, 1]);
        assert!(matches!(
            label_statistics(&intensities, &labels),
            Err(FilterError::SizeMismatch { .. })
        ));
    }
}
