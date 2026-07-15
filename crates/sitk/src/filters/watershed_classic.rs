//! `itk::WatershedImageFilter`: the classic segmentation-tree watershed, and
//! `itk::IsolatedWatershedImageFilter` ([`isolated_watershed`]), the one
//! SimpleITK filter built on it.
//!
//! Port of `Modules/Segmentation/Watersheds/include/itkWatershedImageFilter.h`
//! / `.hxx` and the three `itk::watershed::` component filters its
//! mini-pipeline drives, in the configuration that `WatershedImageFilter`'s
//! constructor pins them to:
//!
//! - `itkWatershedSegmenter.h` / `.hxx` with `DoBoundaryAnalysis(false)` and
//!   `SortEdgeLists(true)` — [`segment`];
//! - `itkWatershedSegmentTreeGenerator.h` / `.hxx` with `Merge(false)` and
//!   `ConsumeInput(false)` — [`generate_segment_tree`];
//! - `itkWatershedRelabeler.h` / `.hxx` — [`WatershedTree::relabel`].
//!
//! Because `DoBoundaryAnalysis` is off, the streaming machinery
//! (`itkWatershedBoundary`, `itkWatershedBoundaryResolver`,
//! `itkWatershedEquivalenceRelabeler`, `Segmenter::AnalyzeBoundaryFlow`,
//! `Segmenter::CollectBoundaryInformation`) is unreachable and is not ported.
//! The `flat_region_t::is_on_boundary` flag it would set is therefore always
//! `false`; the one place that reads it ([`descend_flat_regions`]) keeps the
//! check so the transcription lines up with the `.hxx`.
//!
//! ## Not a SimpleITK filter
//!
//! SimpleITK ships no `WatershedImageFilter.yaml` — the only way to reach this
//! algorithm through SimpleITK is [`isolated_watershed`], which composes it
//! internally. [`watershed`] is therefore an **ITK-only surface**: its
//! parameter defaults come from `itkWatershedImageFilter.h`
//! (`m_Threshold{0.0}`, `m_Level{0.0}`) rather than from a yaml, and it does
//! not correspond to any `sitk::` function. It is `pub` because the engine is
//! useful on its own and because [`isolated_watershed`] must be able to
//! document what it composes.
//!
//! Note that [`watershed`] is **a different algorithm** from
//! [`crate::filters::watershed::morphological_watershed`], despite the shared word.
//! `Level` here is a *fraction* of the input's height range that sets a
//! saliency flood level on a merge tree; there, it is an h-minima height in
//! input-pixel-type units. `Threshold` has no counterpart there at all.
//!
//! ## Parameters
//!
//! Both `threshold` and `level` are percentages of the input's height range,
//! and both are clamped to `[0.0, 1.0]` (`WatershedImageFilter::SetThreshold`
//! / `SetLevel`, and again by `itkSetClampMacro` on the components).
//!
//! - `threshold` floors the input at `L = min + threshold * (max - min)`
//!   before segmentation, erasing shallow minima.
//! - `level` is the saliency flood level: basins whose depth is below it merge.
//!
//! ## Pipeline, stage by stage
//!
//! ### 1. `Segmenter::GenerateData` — the initial segmentation
//!
//! The input is copied into a **padded** buffer, one pixel larger on every
//! face, with the interior thresholded and the one-pixel border filled with
//! `max + 1` (`BuildRetainingWall`). The wall is what lets the gradient
//! descent below run without any boundary check: no interior pixel can ever
//! descend into it. Labels live in a parallel padded buffer, `0` meaning
//! `NULL_LABEL`; the wall's labels stay `0` forever, which is also what keeps
//! the wall out of the adjacency table.
//!
//! Connectivity is **face-connected only** (`GenerateConnectivity`: 4-neighbors
//! in 2-D, 6 in 3-D), and the order the neighbors are visited in is
//! load-bearing for every tie-break below. `GenerateConnectivity` builds it as
//! `-e_{D-1}, ..., -e_0, +e_0, ..., +e_{D-1}` — for 2-D that is up, left,
//! right, down, exactly the `.hxx`'s own diagram.
//!
//! `LabelMinima` sweeps the interior in raster order. For an unlabeled pixel it
//! scans the connectivity list *in order* and stops at the **first** neighbor
//! of equal value (`Math::AlmostEquals`, see below): that makes it a flat
//! region, and it takes that neighbor's label if it has one, else opens a new
//! flat region. If no neighbor is equal and none is smaller, it is a
//! single-pixel local minimum and takes a fresh label. Two passes over the
//! image plus an `EquivalencyTable` merge connected flat regions and record,
//! per flat region, the lowest-valued differently-labeled neighbor touching it
//! (`bounds_min`) and *where* that neighbor is (`min_label_ptr`, a raw pointer
//! in ITK — a position here, dereferenced later, after the descent has
//! relabeled it).
//!
//! `GradientDescent` then walks every still-unlabeled pixel down to a labeled
//! one, pushing the path on a stack and painting it all with the label it
//! lands on. Its steepest-descent choice seeds the running minimum with
//! **connectivity neighbor 0** rather than with the pixel itself, then takes a
//! strict `<`: the first neighbor in connectivity order that attains the
//! minimum wins.
//!
//! `DescendFlatRegions` merges each flat region whose `bounds_min` is strictly
//! below its own value into whatever label now sits at `min_label_ptr`; a flat
//! region with no lower neighbor is a flat *basin* and survives as its own
//! segment.
//!
//! `UpdateSegmentTable` then records, per surviving label, its minimum value
//! and its adjacency list: for each pair of differently-labeled adjacent
//! pixels, the edge's height is the **maximum** of the two pixel values, and
//! the recorded height is the **minimum** such height over the whole shared
//! boundary. `SortEdgeLists` sorts each list by ascending height.
//!
//! ### 2. `SegmentTreeGenerator::GenerateData` — the merge tree
//!
//! `CompileMergeList` proposes, for each segment, the single merge across its
//! lowest edge, with `saliency = lowest_edge_height - segment_min` (the
//! segment's depth), keeping it if `saliency < threshold` where `threshold =
//! level * maximum_depth` and `maximum_depth = max - min` of the input.
//! `ExtractMergeHierarchy` then drains those proposals from a min-heap on
//! saliency, applying each merge that is still valid, recording it in the
//! output tree, and pushing the merged segment's new lowest-edge proposal back
//! on the heap.
//!
//! ### 3. `Relabeler::GenerateData` — the flood
//!
//! Applies every tree entry whose saliency is `<= level * tree.back().saliency`
//! to the initial labeling, via an `EquivalencyTable`, and crops the padding.
//!
//! ## Upstream quirks reproduced verbatim
//!
//! - **`level` is applied twice.** The tree generator keeps merges with
//!   `saliency < level * maximum_depth`; the relabeler then applies those with
//!   `saliency <= level * tree.back().saliency`, and `tree.back().saliency` is
//!   itself bounded by the first cut. For a fresh filter the effective flood
//!   level is therefore roughly `level²` of the maximum depth, not `level`.
//!   This is exactly what the `.hxx` files compute, and
//!   [`isolated_watershed`]'s bisection depends on the *relabeler* stage
//!   behaving this way against a tree built once at the upper limit.
//!
//! - **`PruneEdgeLists` keeps one edge too many.** `itkWatershedSegmentTable.hxx`
//!   finds the first edge with `height - min > maximum_saliency`, then does
//!   `++e; erase(e, end())` — so the offending edge itself survives the prune,
//!   and only the ones after it are dropped. [`prune_edge_lists`] reproduces
//!   the off-by-one.
//!
//! - **A single-segment image is a hard error.** `CompileMergeList` dereferences
//!   `edge_list.front()` after only checking `empty()` to throw
//!   `itkGenericExceptionMacro`. A flat image, or one whose `threshold` floods
//!   away every minimum but one, has a single segment with no adjacencies and
//!   ITK throws. This port returns
//!   [`FilterError::WatershedSegmentWithoutEdges`] there rather than inventing
//!   a "one label everywhere" answer ITK never produces.
//!
//! - **`Segmenter::Threshold` mutates the maximum for integral inputs.** Any
//!   pixel exactly at the pixel type's maximum is lowered by one, and the
//!   `maximum` used for the wall is capped the same way, so that `max + 1`
//!   cannot overflow. This port applies the integral branch whenever the input
//!   image's pixel type is an integer type, and the floating branch otherwise —
//!   matching `NumericTraits<InputPixelType>::is_integer`, even though this
//!   port's arithmetic is `f64` throughout.
//!
//! - **`static_cast<InputPixelType>` on the threshold value truncates.** The
//!   threshold `min + threshold * (max - min)` is cast back to the input pixel
//!   type before it is used, so on an integer image it truncates toward zero.
//!
//! ## Determinism, where ITK has none
//!
//! Four loops in the `.hxx` iterate a `std::unordered_map`, whose order is
//! unspecified and differs between standard libraries:
//!
//! - `MergeFlatRegions` over the `EquivalencyTable` — the merged region keeps
//!   the `min_label_ptr` of the *first* source region attaining the strict
//!   minimum `bounds_min`, so ties are resolved by hash order;
//! - `DescendFlatRegions` over the flat-region table — order-independent, since
//!   each region overwrites a disjoint set of pixels;
//! - `CompileMergeList` over the `SegmentTable` — the initial heap's contents
//!   are pushed in hash order, which decides which of several equal-saliency
//!   merges ends up at the heap root;
//! - `PruneEdgeLists` over the `SegmentTable` — order-independent, since each
//!   segment is pruned in isolation.
//!
//! This port iterates all four in **ascending label order** (`BTreeMap`), so
//! its output is deterministic. Where saliencies are distinct the result is
//! identical to ITK's; where several merges share a saliency exactly, which
//! one is recorded first can differ from a given ITK build. The final labeling
//! is less sensitive than that suggests: `Relabeler` feeds the tree into an
//! `EquivalencyTable`, whose `Add` always maps the larger label onto the
//! smaller, so a merged group is named by its minimum label regardless of the
//! direction or order the merges were recorded in.
//!
//! The binary heap itself is *not* left to chance: [`make_heap`], [`push_heap`]
//! and [`pop_heap`] transcribe libstdc++'s `__adjust_heap` / `__push_heap`
//! exactly, so given the same heap contents the same element is popped.
//!
//! ## `Math::AlmostEquals`
//!
//! Flat-region detection compares pixel values with `itk::Math::AlmostEquals`,
//! not `==`. Its overload set makes that an exact comparison for integer pixel
//! types and a 4-ULP comparison (with a `0.1 * epsilon` absolute-difference
//! escape hatch near zero) for `float` and `double`. [`AlmostEquals`] selects
//! the same three behaviors off the input image's [`PixelId`], comparing in
//! `f32` ULPs for a `Float32` input even though this port's working buffer is
//! `f64` — a `Float32` image's values are exactly representable in `f64`, so
//! narrowing back for the comparison is lossless and reproduces ITK's
//! `FloatAlmostEqual<float>`.

use crate::core::{Image, PixelId};
use crate::filters::error::{FilterError, Result};
use crate::filters::gradient::gradient_magnitude_values;
use crate::filters::image_from_f64;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ---- Math::AlmostEquals ---------------------------------------------------

/// The `itk::Math::AlmostEquals` overload selected by the input pixel type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlmostEquals {
    /// Integer pixel types: `AlmostEquals` is `operator==`.
    Exact,
    /// `float` pixel type: `FloatAlmostEqual<float>`.
    F32,
    /// `double` pixel type: `FloatAlmostEqual<double>`.
    F64,
}

impl AlmostEquals {
    fn for_pixel_type(pixel_id: PixelId) -> Self {
        match pixel_id {
            PixelId::Float32 => AlmostEquals::F32,
            PixelId::Float64 => AlmostEquals::F64,
            _ => AlmostEquals::Exact,
        }
    }

    fn eq(self, a: f64, b: f64) -> bool {
        match self {
            AlmostEquals::Exact => a == b,
            AlmostEquals::F32 => f32_almost_equal(a as f32, b as f32),
            AlmostEquals::F64 => f64_almost_equal(a, b),
        }
    }
}

/// `itk::Math::Detail::FloatIEEE<float>::AsULP()`.
fn f32_as_ulp(x: f32) -> i32 {
    let bits = x.to_bits();
    if bits >> 31 != 0 {
        (!(u32::MAX >> 1)).wrapping_sub(bits) as i32
    } else {
        bits as i32
    }
}

/// `itk::Math::Detail::FloatIEEE<double>::AsULP()`.
fn f64_as_ulp(x: f64) -> i64 {
    let bits = x.to_bits();
    if bits >> 63 != 0 {
        (!(u64::MAX >> 1)).wrapping_sub(bits) as i64
    } else {
        bits as i64
    }
}

/// `itk::Math::FloatAlmostEqual<float>` with its default `maxUlps = 4` and
/// `maxAbsoluteDifference = 0.1 * epsilon`.
fn f32_almost_equal(x1: f32, x2: f32) -> bool {
    if (x1 - x2).abs() <= 0.1 * f32::EPSILON {
        return true;
    }
    if x1.is_sign_negative() != x2.is_sign_negative() {
        return false;
    }
    f32_as_ulp(x1).wrapping_sub(f32_as_ulp(x2)).abs() <= 4
}

/// `itk::Math::FloatAlmostEqual<double>`, same defaults.
fn f64_almost_equal(x1: f64, x2: f64) -> bool {
    if (x1 - x2).abs() <= 0.1 * f64::EPSILON {
        return true;
    }
    if x1.is_sign_negative() != x2.is_sign_negative() {
        return false;
    }
    f64_as_ulp(x1).wrapping_sub(f64_as_ulp(x2)).abs() <= 4
}

// ---- itk::EquivalencyTable ------------------------------------------------

/// `itk::EquivalencyTable` (`itkEquivalencyTable.h` / `.cxx`): every entry maps
/// a **larger** label onto a **smaller** one, which is what makes a merged
/// group end up named by its minimum label and makes cycles impossible.
///
/// Backed by a `BTreeMap` rather than ITK's `std::unordered_map` so the one
/// caller that iterates it ([`merge_flat_regions`]) is deterministic.
#[derive(Debug, Default)]
struct EquivalencyTable {
    map: BTreeMap<u64, u64>,
}

impl EquivalencyTable {
    /// `EquivalencyTable::Add`. Normalizes so the key is the larger label, and
    /// on a conflicting existing entry recurses to merge the two chains.
    fn add(&mut self, a: u64, b: u64) {
        let (mut a, mut b) = (a, b);
        loop {
            if a == b {
                return;
            }
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }
            match self.map.get(&a) {
                None => {
                    self.map.insert(a, b);
                    return;
                }
                Some(&existing) => {
                    if existing == b {
                        return;
                    }
                    a = existing; // tail-recurse into Add(existing, b)
                }
            }
        }
    }

    /// `EquivalencyTable::RecursiveLookup`.
    fn recursive_lookup(&self, a: u64) -> u64 {
        let mut ans = a;
        let mut last_ans = a;
        while let Some(&next) = self.map.get(&ans) {
            ans = next;
            if ans == a {
                return last_ans; // about to cycle again
            }
            last_ans = ans;
        }
        ans
    }

    /// `EquivalencyTable::Flatten`.
    fn flatten(&mut self) {
        let keys: Vec<u64> = self.map.keys().copied().collect();
        for k in keys {
            let v = self.map[&k];
            let flat = self.recursive_lookup(v);
            self.map.insert(k, flat);
        }
    }

    /// `EquivalencyTable::Lookup` — non-recursive, identity on a miss.
    fn lookup(&self, a: u64) -> u64 {
        self.map.get(&a).copied().unwrap_or(a)
    }
}

/// `Segmenter::RelabelImage`, restricted to `positions`. The Segmenter always
/// passes the padded image's interior, never the retaining wall.
fn relabel_image(labels: &mut [u64], positions: &[usize], table: &mut EquivalencyTable) {
    table.flatten();
    for &p in positions {
        let temp = table.lookup(labels[p]);
        if temp != labels[p] {
            labels[p] = temp;
        }
    }
}

/// `Relabeler::GenerateData`'s call, which covers the whole buffer.
fn relabel_slice(labels: &mut [u64], table: &mut EquivalencyTable) {
    table.flatten();
    for label in labels.iter_mut() {
        let temp = table.lookup(*label);
        if temp != *label {
            *label = temp;
        }
    }
}

// ---- itk::OneWayEquivalencyTable ------------------------------------------

/// `itk::OneWayEquivalencyTable` (`itkOneWayEquivalencyTable.h` / `.cxx`):
/// unlike [`EquivalencyTable`] the direction of an equivalence is meaningful
/// and `Add` never reorders its arguments.
#[derive(Debug, Default)]
struct OneWayEquivalencyTable {
    map: HashMap<u64, u64>,
}

impl OneWayEquivalencyTable {
    fn add(&mut self, a: u64, b: u64) {
        if a == b {
            return;
        }
        self.map.entry(a).or_insert(b);
    }

    fn recursive_lookup(&self, a: u64) -> u64 {
        let mut ans = a;
        let mut last_ans = a;
        while let Some(&next) = self.map.get(&ans) {
            ans = next;
            if ans == a {
                return last_ans;
            }
            last_ans = ans;
        }
        ans
    }

    /// `OneWayEquivalencyTable::Flatten` — note it looks up `first`, not
    /// `second`, which lands on the same answer. Semantically a no-op given
    /// [`Self::recursive_lookup`]; kept because the `.hxx` calls it and it
    /// bounds the chain length.
    fn flatten(&mut self) {
        let keys: Vec<u64> = self.map.keys().copied().collect();
        for k in keys {
            let flat = self.recursive_lookup(k);
            self.map.insert(k, flat);
        }
    }
}

// ---- itk::watershed::SegmentTable -----------------------------------------

/// One entry of a segment's adjacency list: `SegmentTable::edge_pair_t`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Edge {
    label: u64,
    height: f64,
}

/// `SegmentTable::segment_t`.
#[derive(Clone, Debug)]
struct Segment {
    min: f64,
    edge_list: VecDeque<Edge>,
}

/// `itk::watershed::SegmentTable`. `BTreeMap` for deterministic iteration; see
/// the module docs.
#[derive(Clone, Debug, Default)]
struct SegmentTable {
    segments: BTreeMap<u64, Segment>,
    maximum_depth: f64,
}

/// `SegmentTable::SortEdgeLists`: `std::list::sort` is a stable merge sort and
/// `edge_pair_t::operator<` compares `height` only, so equal-height edges keep
/// the ascending-label order the `std::map` in `UpdateSegmentTable` gave them.
fn sort_edge_lists(table: &mut SegmentTable) {
    for segment in table.segments.values_mut() {
        let mut edges: Vec<Edge> = segment.edge_list.iter().copied().collect();
        // `Ordering::Equal` on an unordered pair matches `operator<` returning
        // false both ways, which is how `std::list::sort` treats a NaN height.
        edges.sort_by(|a, b| {
            a.height
                .partial_cmp(&b.height)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        segment.edge_list = edges.into();
    }
}

/// `SegmentTable::PruneEdgeLists`, including the off-by-one that keeps the
/// first offending edge (see the module docs).
fn prune_edge_lists(table: &mut SegmentTable, maximum_saliency: f64) {
    for segment in table.segments.values_mut() {
        let cut = segment
            .edge_list
            .iter()
            .position(|e| e.height - segment.min > maximum_saliency);
        if let Some(i) = cut {
            segment.edge_list.truncate(i + 1);
        }
    }
}

// ---- itk::watershed::SegmentTree ------------------------------------------

/// `SegmentTree::merge_t`: a merge of `from` into `to` at `saliency`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Merge {
    from: u64,
    to: u64,
    saliency: f64,
}

/// `SegmentTree::merge_comp` — `b.saliency < a.saliency`, which turns the
/// standard max-heap into a min-heap on saliency.
fn merge_comp(a: &Merge, b: &Merge) -> bool {
    b.saliency < a.saliency
}

/// libstdc++'s `std::__push_heap`.
fn push_heap_hole(heap: &mut [Merge], mut hole: usize, top: usize, value: Merge) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !merge_comp(&heap[parent], &value) {
            break;
        }
        heap[hole] = heap[parent];
        hole = parent;
    }
    heap[hole] = value;
}

/// libstdc++'s `std::__adjust_heap`.
fn adjust_heap(heap: &mut [Merge], mut hole: usize, len: usize, value: Merge) {
    let top = hole;
    let mut second_child = hole;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if merge_comp(&heap[second_child], &heap[second_child - 1]) {
            second_child -= 1;
        }
        heap[hole] = heap[second_child];
        hole = second_child;
    }
    if len.is_multiple_of(2) && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        heap[hole] = heap[second_child - 1];
        hole = second_child - 1;
    }
    push_heap_hole(heap, hole, top, value);
}

/// `std::make_heap` with [`merge_comp`].
fn make_heap(heap: &mut [Merge]) {
    let len = heap.len();
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = heap[parent];
        adjust_heap(heap, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::push_heap` with [`merge_comp`]: sifts up the element just appended.
fn push_heap(heap: &mut [Merge]) {
    let last = heap.len() - 1;
    let value = heap[last];
    push_heap_hole(heap, last, 0, value);
}

/// `std::pop_heap` with [`merge_comp`]: moves the root to the back. ITK then
/// calls `PopBack()` to drop it.
fn pop_heap(heap: &mut [Merge]) {
    let len = heap.len();
    if len > 1 {
        let value = heap[len - 1];
        heap[len - 1] = heap[0];
        adjust_heap(&mut heap[..len - 1], 0, len - 1, value);
    }
}

// ---- N-D indexing over the padded buffer ----------------------------------

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// The padded geometry the segmenter works on: the input region grown by one
/// pixel on every face, which the retaining wall then occupies.
struct Padded {
    /// Padded size, `input_size[d] + 2`.
    size: Vec<usize>,
    /// `GenerateConnectivity`'s offsets, as signed flat deltas:
    /// `-stride[D-1], ..., -stride[0], +stride[0], ..., +stride[D-1]`.
    connectivity: Vec<isize>,
    /// Flat padded indices of the interior (the original image region), in
    /// raster order — `ImageRegionIterator` order.
    interior: Vec<usize>,
}

impl Padded {
    fn new(input_size: &[usize]) -> Self {
        let dim = input_size.len();
        let size: Vec<usize> = input_size.iter().map(|s| s + 2).collect();
        let padded_strides = strides(&size);

        let mut connectivity = Vec::with_capacity(2 * dim);
        for &stride in padded_strides.iter().take(dim).rev() {
            connectivity.push(-(stride as isize));
        }
        for &stride in padded_strides.iter().take(dim) {
            connectivity.push(stride as isize);
        }

        let total: usize = input_size.iter().product();
        let in_strides = strides(input_size);
        let interior = (0..total)
            .map(|flat| {
                (0..dim)
                    .map(|d| ((flat / in_strides[d]) % input_size[d] + 1) * padded_strides[d])
                    .sum()
            })
            .collect();

        Padded {
            size,
            connectivity,
            interior,
        }
    }

    fn total(&self) -> usize {
        self.size.iter().product()
    }

    fn neighbor(&self, p: usize, i: usize) -> usize {
        p.wrapping_add_signed(self.connectivity[i])
    }
}

// ---- itk::watershed::Segmenter --------------------------------------------

/// `Segmenter::flat_region_t`. `min_label_ptr` is a raw `IdentifierType *` in
/// ITK; here it is the padded flat index it points at, and the label is read
/// out of the label buffer at the moment ITK would dereference the pointer.
#[derive(Clone, Copy, Debug)]
struct FlatRegion {
    min_label_at: usize,
    bounds_min: f64,
    value: f64,
    /// Only ever set by `AnalyzeBoundaryFlow`, which `DoBoundaryAnalysis(false)`
    /// makes unreachable. Kept so [`descend_flat_regions`] reads like the `.hxx`.
    is_on_boundary: bool,
}

/// `Segmenter::MergeFlatRegions`. Both members of every equivalence recorded by
/// [`label_minima`] are labels it minted for flat regions, so both are keys of
/// `regions`; ITK throws ("Unexpected error.") when they are not, and the two
/// `expect`s below stand in for that. `eq_table` is flattened first so no key is
/// also a value, which is what lets each `a` be removed after a single visit.
fn merge_flat_regions(regions: &mut BTreeMap<u64, FlatRegion>, eq_table: &mut EquivalencyTable) {
    eq_table.flatten();
    let pairs: Vec<(u64, u64)> = eq_table.map.iter().map(|(&a, &b)| (a, b)).collect();
    for (a, b) in pairs {
        let region_a = *regions.get(&a).expect("equivalence key is a flat region");
        let region_b = regions
            .get_mut(&b)
            .expect("equivalence value is a flat region");
        if region_a.bounds_min < region_b.bounds_min {
            region_b.bounds_min = region_a.bounds_min;
            region_b.min_label_at = region_a.min_label_at;
        }
        regions.remove(&a);
    }
}

/// `Segmenter::LabelMinima`: labels every local minimum and every flat region.
///
/// `current_label` enters at `1` (`Segmenter::GenerateData`'s
/// `SetCurrentLabel(1)`) and is bumped once per new region.
fn label_minima(
    values: &[f64],
    labels: &mut [u64],
    padded: &Padded,
    current_label: &mut u64,
    wall: f64,
    eq: AlmostEquals,
) -> BTreeMap<u64, FlatRegion> {
    let n_conn = padded.connectivity.len();
    let mut flat_regions: BTreeMap<u64, FlatRegion> = BTreeMap::new();
    let mut equivalent = EquivalencyTable::default();

    for &p in &padded.interior {
        if labels[p] != 0 {
            continue;
        }

        let current_value = values[p];
        let mut found_single_pixel_minimum = true;
        let mut found_flat_region = false;
        let mut i = 0;
        while i < n_conn {
            let n = padded.neighbor(p, i);
            if eq.eq(current_value, values[n]) {
                found_flat_region = true;
                break;
            }
            if current_value > values[n] {
                found_single_pixel_minimum = false;
            }
            i += 1;
        }

        if found_flat_region {
            let flat_neighbor = padded.neighbor(p, i);
            if labels[flat_neighbor] != 0 {
                labels[p] = labels[flat_neighbor];
            } else {
                labels[p] = *current_label;
                // `nPos = m_Connectivity.index[0]` — the pointer is seeded with
                // connectivity neighbor 0, not with the flat neighbor found above.
                flat_regions.insert(
                    *current_label,
                    FlatRegion {
                        min_label_at: padded.neighbor(p, 0),
                        bounds_min: wall,
                        value: current_value,
                        is_on_boundary: false,
                    },
                );
                *current_label += 1;
            }

            // Did we just link two flat regions of the same height?
            for j in i + 1..n_conn {
                let n = padded.neighbor(p, j);
                if eq.eq(values[p], values[n]) && labels[n] != 0 && labels[n] != labels[p] {
                    equivalent.add(labels[p], labels[n]);
                }
            }
        } else if found_single_pixel_minimum {
            labels[p] = *current_label;
            *current_label += 1;
        }
    }

    merge_flat_regions(&mut flat_regions, &mut equivalent);
    relabel_image(labels, &padded.interior, &mut equivalent);
    equivalent = EquivalencyTable::default();

    // Second pass: establish each flat region's lowest boundary value.
    for &p in &padded.interior {
        let label_p = labels[p];
        let Some(region) = flat_regions.get_mut(&label_p) else {
            continue;
        };
        for i in 0..n_conn {
            let n = padded.neighbor(p, i);
            if labels[n] != label_p && values[n] < region.bounds_min {
                region.bounds_min = values[n];
                region.min_label_at = n;
            }
            if eq.eq(values[p], values[n]) && labels[n] != 0 {
                equivalent.add(label_p, labels[n]);
            }
        }
    }

    merge_flat_regions(&mut flat_regions, &mut equivalent);
    relabel_image(labels, &padded.interior, &mut equivalent);

    flat_regions
}

/// `Segmenter::GradientDescent`: trace every unlabeled pixel down to a label.
fn gradient_descent(values: &[f64], labels: &mut [u64], padded: &Padded) {
    let n_conn = padded.connectivity.len();
    let mut update_stack: Vec<usize> = Vec::new();

    for &p in &padded.interior {
        if labels[p] != 0 {
            continue;
        }
        update_stack.clear();
        let mut current = p;
        let new_label = loop {
            update_stack.push(current);
            // Seeded with connectivity neighbor 0, then strict `<`: the first
            // neighbor in connectivity order attaining the minimum wins.
            let mut min_val = values[padded.neighbor(current, 0)];
            let mut move_to = padded.neighbor(current, 0);
            for i in 1..n_conn {
                let n = padded.neighbor(current, i);
                if values[n] < min_val {
                    min_val = values[n];
                    move_to = n;
                }
            }
            current = move_to;
            if labels[current] != 0 {
                break labels[current];
            }
        };
        for &q in &update_stack {
            labels[q] = new_label;
        }
    }
}

/// `Segmenter::DescendFlatRegions`: a flat region with a strictly lower
/// neighbor merges into whatever label now sits at that neighbor; one without
/// is a flat basin and survives.
fn descend_flat_regions(
    labels: &mut [u64],
    padded: &Padded,
    flat_regions: &BTreeMap<u64, FlatRegion>,
) {
    let mut equivalent = EquivalencyTable::default();
    for (&label, region) in flat_regions {
        if region.bounds_min < region.value && !region.is_on_boundary {
            equivalent.add(label, labels[region.min_label_at]);
        }
    }
    equivalent.flatten();
    relabel_image(labels, &padded.interior, &mut equivalent);
}

/// `Segmenter::UpdateSegmentTable`: per-segment minimum and adjacency list.
fn update_segment_table(values: &[f64], labels: &[u64], padded: &Padded) -> SegmentTable {
    let n_conn = padded.connectivity.len();
    let mut segments: BTreeMap<u64, Segment> = BTreeMap::new();
    // `std::map<IdentifierType, InputPixelType>` per segment: ascending label.
    let mut edge_hash: BTreeMap<u64, BTreeMap<u64, f64>> = BTreeMap::new();

    for &p in &padded.interior {
        let segment_label = labels[p];

        match segments.get_mut(&segment_label) {
            None => {
                segments.insert(
                    segment_label,
                    Segment {
                        min: values[p],
                        edge_list: VecDeque::new(),
                    },
                );
                edge_hash.insert(segment_label, BTreeMap::new());
            }
            Some(segment) => {
                if values[p] < segment.min {
                    segment.min = values[p];
                }
            }
        }

        let edges = edge_hash.get_mut(&segment_label).expect("just inserted");
        for i in 0..n_conn {
            let n = padded.neighbor(p, i);
            if labels[n] == segment_label || labels[n] == 0 {
                continue;
            }
            // The edge height is the max of the two adjacent pixels; the
            // recorded height is the min over the shared boundary.
            let lowest_edge = if values[n] < values[p] {
                values[p]
            } else {
                values[n]
            };
            edges
                .entry(labels[n])
                .and_modify(|h| {
                    if lowest_edge < *h {
                        *h = lowest_edge;
                    }
                })
                .or_insert(lowest_edge);
        }
    }

    for (label, edges) in edge_hash {
        let segment = segments.get_mut(&label).expect("every label has a segment");
        segment.edge_list = edges
            .into_iter()
            .map(|(label, height)| Edge { label, height })
            .collect();
    }

    SegmentTable {
        segments,
        maximum_depth: 0.0,
    }
}

/// The initial segmentation: `itk::watershed::Segmenter::GenerateData` with
/// `DoBoundaryAnalysis(false)` and `SortEdgeLists(true)`.
///
/// Returns the padded label buffer, the padded geometry, and the segment table.
fn segment(
    input_values: &[f64],
    input_size: &[usize],
    pixel_id: PixelId,
    threshold: f64,
) -> (Vec<u64>, Padded, SegmentTable) {
    let eq = AlmostEquals::for_pixel_type(pixel_id);
    let is_integer = !pixel_id.is_floating_point();
    let padded = Padded::new(input_size);

    // `Self::MinMax(input, regionToProcess, minimum, maximum)`.
    let mut minimum = input_values[0];
    let mut maximum = input_values[0];
    for &v in input_values {
        if v > maximum {
            maximum = v;
        }
        if v < minimum {
            minimum = v;
        }
    }

    // "cap the maximum in the image so that we can always define a pixel value
    // that is one greater than the maximum value in the image."
    let type_max = integer_pixel_type_max(pixel_id);
    if is_integer && type_max.is_some_and(|m| maximum == m) {
        maximum -= 1.0;
    }

    // `static_cast<InputPixelType>((m_Threshold * (maximum - minimum)) + minimum)`.
    let threshold_value = {
        let raw = threshold * (maximum - minimum) + minimum;
        if is_integer { raw.trunc() } else { raw }
    };

    // `Self::Threshold(thresholdImage, input, ...)` into the padded interior,
    // then `BuildRetainingWall` over the whole padded border.
    let wall = maximum + 1.0;
    let mut values = vec![wall; padded.total()];
    for (&p, &v) in padded.interior.iter().zip(input_values) {
        values[p] = if v < threshold_value {
            threshold_value
        } else if is_integer && type_max.is_some_and(|m| v == m) {
            v - 1.0
        } else {
            v
        };
    }

    let mut labels = vec![0u64; padded.total()];
    let mut current_label: u64 = 1;

    let flat_regions = label_minima(&values, &mut labels, &padded, &mut current_label, wall, eq);
    gradient_descent(&values, &mut labels, &padded);
    descend_flat_regions(&mut labels, &padded, &flat_regions);

    let mut table = update_segment_table(&values, &labels, &padded);
    sort_edge_lists(&mut table);
    table.maximum_depth = maximum - minimum;

    (labels, padded, table)
}

/// `NumericTraits<InputPixelType>::max()` as `f64`, for the integer pixel
/// types only. `None` for `Float32`/`Float64`, where the cap does not apply.
///
/// `u64::MAX` and `i64::MAX` are not exactly representable in `f64`; the
/// nearest `f64` is what `static_cast<double>` would produce, and it is what a
/// `u64::MAX`-valued pixel widens to, so the equality still fires.
fn integer_pixel_type_max(pixel_id: PixelId) -> Option<f64> {
    match pixel_id {
        PixelId::UInt8 | PixelId::VectorUInt8 => Some(u8::MAX as f64),
        PixelId::Int8 | PixelId::VectorInt8 => Some(i8::MAX as f64),
        PixelId::UInt16 | PixelId::VectorUInt16 => Some(u16::MAX as f64),
        PixelId::Int16 | PixelId::VectorInt16 => Some(i16::MAX as f64),
        PixelId::UInt32 | PixelId::VectorUInt32 => Some(u32::MAX as f64),
        PixelId::Int32 | PixelId::VectorInt32 => Some(i32::MAX as f64),
        PixelId::UInt64 | PixelId::VectorUInt64 => Some(u64::MAX as f64),
        PixelId::Int64 | PixelId::VectorInt64 => Some(i64::MAX as f64),
        PixelId::Float32
        | PixelId::ComplexFloat32
        | PixelId::VectorFloat32
        | PixelId::Float64
        | PixelId::ComplexFloat64
        | PixelId::VectorFloat64 => None,
    }
}

// ---- itk::watershed::SegmentTreeGenerator ---------------------------------

/// `SegmentTreeGenerator::MergeSegments`: fold `from`'s adjacency list into
/// `to`'s, eliminating redundant edges, then erase `from`.
fn merge_segments(
    table: &mut SegmentTable,
    eq_table: &mut OneWayEquivalencyTable,
    from: u64,
    to: u64,
) {
    let from_seg = table.segments.remove(&from).expect("FROM must exist");
    let to_seg = table.segments.get(&to).expect("TO must exist").clone();

    let mut to_min = to_seg.min;
    if from_seg.min < to_min {
        to_min = from_seg.min;
    }

    let to_list: Vec<Edge> = to_seg.edge_list.into_iter().collect();
    let from_list: Vec<Edge> = from_seg.edge_list.into_iter().collect();
    let mut out: VecDeque<Edge> = VecDeque::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let (mut i, mut j) = (0usize, 0usize);

    while i < to_list.len() && j < from_list.len() {
        let label_to = eq_table.recursive_lookup(to_list[i].label);
        let label_from = eq_table.recursive_lookup(from_list[j].label);

        // Ignore any label already in this list, and any pointer back to
        // ourself. This is what keeps the edge lists from growing
        // exponentially as segments merge.
        if seen.contains(&label_to) || label_to == from {
            i += 1; // `to_seg->edge_list.erase(edgeTOi)`
            continue;
        }
        if seen.contains(&label_from) || label_from == to {
            j += 1;
            continue;
        }

        // Which edge is next in the (sorted) list?
        if from_list[j].height < to_list[i].height {
            out.push_back(Edge {
                label: label_from,
                height: from_list[j].height,
            });
            seen.insert(label_from);
            j += 1;
        } else {
            out.push_back(Edge {
                label: label_to,
                height: to_list[i].height,
            });
            seen.insert(label_to);
            i += 1;
        }
    }

    while j < from_list.len() {
        let label_from = eq_table.recursive_lookup(from_list[j].label);
        if !(seen.contains(&label_from) || label_from == to) {
            out.push_back(Edge {
                label: label_from,
                height: from_list[j].height,
            });
            seen.insert(label_from);
        }
        j += 1;
    }

    while i < to_list.len() {
        let label_to = eq_table.recursive_lookup(to_list[i].label);
        if !(seen.contains(&label_to) || label_to == from) {
            out.push_back(Edge {
                label: label_to,
                height: to_list[i].height,
            });
            seen.insert(label_to);
        }
        i += 1;
    }

    let to_seg = table.segments.get_mut(&to).expect("TO must exist");
    to_seg.min = to_min;
    to_seg.edge_list = out;

    eq_table.add(from, to);
}

/// `SegmentTreeGenerator::CompileMergeList`: one candidate merge per segment,
/// across its lowest edge, heapified on saliency.
fn compile_merge_list(
    table: &mut SegmentTable,
    merged: &mut OneWayEquivalencyTable,
    flood_level: f64,
) -> Result<Vec<Merge>> {
    let threshold = flood_level * table.maximum_depth;
    merged.flatten();
    prune_edge_lists(table, threshold);

    let mut merge_list: Vec<Merge> = Vec::new();
    for (&label_from, segment) in table.segments.iter_mut() {
        if segment.edge_list.is_empty() {
            return Err(FilterError::WatershedSegmentWithoutEdges { label: label_from });
        }
        // `m_MergedSegmentsTable` is empty on entry (GenerateData clears it),
        // so the recursive lookups are the identity and the `while` below never
        // spins. Transcribed anyway.
        let mut label_to = merged.recursive_lookup(segment.edge_list[0].label);
        while label_to == label_from {
            segment.edge_list.pop_front();
            let Some(front) = segment.edge_list.front() else {
                return Err(FilterError::WatershedSegmentWithoutEdges { label: label_from });
            };
            label_to = merged.recursive_lookup(front.label);
        }

        let front = segment.edge_list[0];
        let saliency = front.height - segment.min;
        if saliency < threshold {
            merge_list.push(Merge {
                from: label_from,
                to: label_to,
                saliency,
            });
        }
    }

    make_heap(&mut merge_list);
    Ok(merge_list)
}

/// `SegmentTreeGenerator::ExtractMergeHierarchy`: drain the heap, applying and
/// recording each still-valid merge and re-proposing the merged segment.
fn extract_merge_hierarchy(
    table: &mut SegmentTable,
    merged: &mut OneWayEquivalencyTable,
    heap: &mut Vec<Merge>,
    flood_level: f64,
) -> Vec<Merge> {
    let threshold = flood_level * table.maximum_depth;
    let mut tree: Vec<Merge> = Vec::new();
    if heap.is_empty() {
        return tree;
    }

    let mut counter: u32 = 0;
    let mut top_merge = heap[0];

    while !heap.is_empty() && top_merge.saliency <= threshold {
        counter += 1;
        if counter == 10_000 {
            counter = 0;
            prune_edge_lists(table, threshold);
        }
        if counter.is_multiple_of(10_000) {
            merged.flatten();
        }

        pop_heap(heap);
        heap.pop();

        let from_seg_label = merged.recursive_lookup(top_merge.from);
        let to_seg_label = merged.recursive_lookup(top_merge.to);

        if from_seg_label == top_merge.from && from_seg_label != to_seg_label {
            tree.push(Merge {
                from: from_seg_label,
                to: to_seg_label,
                saliency: top_merge.saliency,
            });

            merge_segments(table, merged, from_seg_label, to_seg_label);

            // Propose the merged segment's new lowest-edge merge.
            let to_seg = table.segments.get_mut(&to_seg_label).expect("TO survives");
            if !to_seg.edge_list.is_empty() {
                // ITK's `while (tempMerge.to == tempMerge.from) pop_front()`
                // dereferences `front()` without re-checking `empty()`. Guard
                // it: an emptied list simply proposes nothing.
                let mut proposal = None;
                while let Some(front) = to_seg.edge_list.front().copied() {
                    let target = merged.recursive_lookup(front.label);
                    if target != to_seg_label {
                        proposal = Some(Merge {
                            from: to_seg_label,
                            to: target,
                            saliency: front.height - to_seg.min,
                        });
                        break;
                    }
                    to_seg.edge_list.pop_front();
                }
                if let Some(proposal) = proposal {
                    heap.push(proposal);
                    push_heap(heap);
                }
            }
        }

        if !heap.is_empty() {
            top_merge = heap[0];
        }
    }

    tree
}

/// `SegmentTreeGenerator::GenerateData` with `Merge(false)`, `ConsumeInput(false)`.
fn generate_segment_tree(table: &SegmentTable, flood_level: f64) -> Result<Vec<Merge>> {
    let mut merged = OneWayEquivalencyTable::default(); // `m_MergedSegments->Clear()`

    let mut seg = table.clone(); // `seg->Copy(*input)`
    sort_edge_lists(&mut seg); // already sorted; `std::list::sort` is stable

    let mut heap = compile_merge_list(&mut seg, &mut merged, flood_level)?;
    Ok(extract_merge_hierarchy(
        &mut seg,
        &mut merged,
        &mut heap,
        flood_level,
    ))
}

// ---- the assembled filter -------------------------------------------------

/// The output of the segmenter and the tree generator: everything
/// `itk::watershed::Relabeler` needs, for any flood level up to the one the
/// tree was built at.
///
/// `WatershedImageFilter` caches exactly this much between updates: raising
/// `Level` above `SegmentTreeGenerator::HighestCalculatedFloodLevel` re-runs
/// the tree generator, while lowering it re-runs only the relabeler. Since
/// the relabeler's own cut is `level * tree.back().saliency`, a tree built at
/// a *higher* level gives a *different* answer at the same `level` than a
/// freshly built one would. [`isolated_watershed`] depends on that; [`watershed`]
/// hides it by building the tree at the level it relabels at.
pub struct WatershedTree {
    /// The original (unpadded) image size.
    size: Vec<usize>,
    /// The initial segmentation, cropped back to `size`.
    labels: Vec<u64>,
    /// The merge tree, in the order `ExtractMergeHierarchy` recorded it.
    tree: Vec<Merge>,
}

impl WatershedTree {
    /// `itk::watershed::Relabeler::GenerateData`: apply every merge whose
    /// saliency is at most `level * tree.back().saliency`.
    ///
    /// `level` is clamped to `[0.0, 1.0]` (`itkSetClampMacro(FloodLevel, ...)`).
    /// An empty tree relabels nothing ("Empty input. No relabeling was done.").
    fn relabel(&self, level: f64) -> Vec<u64> {
        let mut out = self.labels.clone();
        let Some(back) = self.tree.last() else {
            return out;
        };
        let merge_limit = level.clamp(0.0, 1.0) * back.saliency;

        let mut eq = EquivalencyTable::default();
        for merge in &self.tree {
            if merge.saliency > merge_limit {
                break;
            }
            eq.add(merge.from, merge.to);
        }

        relabel_slice(&mut out, &mut eq);
        out
    }
}

/// Run the segmenter and the tree generator. `threshold` and `level` are
/// clamped to `[0, 1]`, as `WatershedImageFilter::SetThreshold` / `SetLevel` do.
fn watershed_tree(image: &Image, threshold: f64, level: f64) -> Result<WatershedTree> {
    let size = image.size().to_vec();
    let total: usize = size.iter().product();
    if total == 0 {
        return Ok(WatershedTree {
            size,
            labels: Vec::new(),
            tree: Vec::new(),
        });
    }

    let values = image.to_f64_vec()?;
    let (padded_labels, padded, table) =
        segment(&values, &size, image.pixel_id(), threshold.clamp(0.0, 1.0));
    let tree = generate_segment_tree(&table, level.clamp(0.0, 1.0))?;

    // `Relabeler`'s output requested region is the *input's* largest possible
    // region, so copying it out of the segmenter's padded buffer crops the wall.
    let labels = padded.interior.iter().map(|&p| padded_labels[p]).collect();

    Ok(WatershedTree { size, labels, tree })
}

/// `itk::WatershedImageFilter`: watershed segmentation via a merge tree.
///
/// `threshold` and `level` are both fractions of the input's height range and
/// are clamped to `[0.0, 1.0]`; both default to `0.0` in
/// `itkWatershedImageFilter.h`. Output is a `UInt64` label image
/// (`Image<IdentifierType, D>`) carrying the input's geometry. Labels start at
/// `1`, are **not** contiguous, and `0` never appears.
///
/// See the module docs: this is not [`crate::filters::watershed::morphological_watershed`],
/// it is not exposed by SimpleITK, and `level` ends up applied twice.
///
/// Errors with [`FilterError::WatershedSegmentWithoutEdges`] when the initial
/// segmentation yields a single segment (a flat image, or one over-thresholded
/// to a single minimum) — ITK throws in the same place.
pub fn watershed(image: &Image, threshold: f64, level: f64) -> Result<Image> {
    let tree = watershed_tree(image, threshold, level)?;
    let labels = tree.relabel(level.clamp(0.0, 1.0));
    let mut result = Image::from_vec(&tree.size, labels)?;
    result.copy_geometry_from(image);
    Ok(result)
}

// ---- itk::IsolatedWatershedImageFilter ------------------------------------

/// The non-seed parameters of [`isolated_watershed`], mirroring SimpleITK's
/// `IsolatedWatershedImageFilter.yaml` members and their defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolatedWatershedSettings {
    /// `Threshold`, the watershed threshold. Default `0.0`.
    pub threshold: f64,
    /// `UpperValueLimit`, where the bisection over the waterlevel starts.
    /// Default `1.0`.
    pub upper_value_limit: f64,
    /// `IsolatedValueTolerance`, the bisection's stopping precision. Default
    /// `0.001`.
    pub isolated_value_tolerance: f64,
    /// `ReplaceValue1`, painted over Seed1's basin. SimpleITK defaults it to
    /// `1`; `itkIsolatedWatershedImageFilter.hxx`'s constructor agrees
    /// (`NumericTraits<OutputImagePixelType>::OneValue()`).
    pub replace_value1: u8,
    /// `ReplaceValue2`, painted over Seed2's basin. SimpleITK's yaml defaults
    /// it to `2`, **overriding** ITK's own constructor default of `0`
    /// (`OutputImagePixelType{}`). This port follows SimpleITK.
    pub replace_value2: u8,
}

impl Default for IsolatedWatershedSettings {
    fn default() -> Self {
        IsolatedWatershedSettings {
            threshold: 0.0,
            upper_value_limit: 1.0,
            isolated_value_tolerance: 0.001,
            replace_value1: 1,
            replace_value2: 2,
        }
    }
}

/// The result of [`isolated_watershed`].
#[derive(Clone, Debug, PartialEq)]
pub struct IsolatedWatershedResult {
    /// `UInt8` image: `replace_value1` on Seed1's basin, `replace_value2` on
    /// Seed2's, `0` elsewhere.
    pub image: Image,
    /// `GetIsolatedValue()`: the waterlevel the bisection settled on. This is
    /// always the final `lower` bound, on every path through the filter,
    /// including the ones where the seeds were never separated.
    pub isolated_value: f64,
}

/// Validates `seed` against `size` and returns its flat offset.
/// `IsolatedWatershedImageFilter::VerifyInputInformation` throws
/// `"Seed1 is not within the input image!"` for an out-of-range seed.
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
    let strides = strides(size);
    Ok(seed.iter().zip(&strides).map(|(&s, &st)| s * st).sum())
}

/// `NumericTraits<InputPixelType>::RealType` — the pixel type of the
/// `RealImageType` that `IsolatedWatershedImageFilter` runs its gradient
/// magnitude and its watershed on. `double` for **every** scalar input type:
/// `NumericTraits<float>::RealType` is `double` (itkNumericTraits.h:1349/1356).
fn real_pixel_type(_pixel_id: PixelId) -> PixelId {
    PixelId::Float64
}

/// `itk::IsolatedWatershedImageFilter`: bisect the watershed waterlevel until
/// `seed1` and `seed2` fall in different basins, then paint those two basins.
///
/// The pipeline is `GradientMagnitudeImageFilter` → `WatershedImageFilter`,
/// with **no** smoothing stage — no Gaussian blur, no anisotropic diffusion.
/// The gradient magnitude is computed at `NumericTraits<InputPixelType>::RealType`
/// precision (`double` for **every** scalar input, `float` included), not at
/// the `float` that SimpleITK's standalone `gradient_magnitude` would give.
///
/// The bisection starts at `guess = upper_value_limit` and halves
/// `[lower, upper]` while `lower + isolated_value_tolerance < guess`, where
/// `lower` starts at `threshold`. Each iteration re-runs the watershed at
/// `guess` and moves `upper` down when the seeds share a basin, `lower` up when
/// they do not. Crucially, ITK reuses **one** `WatershedImageFilter` instance
/// across the whole bisection, so its merge tree is computed once — at the
/// first `guess`, which is the largest one — and every later iteration only
/// re-runs the relabeler against that tree. This port does the same
/// ([`WatershedTree`]), because the relabeler's cut is
/// `level * tree.back().saliency` and a per-iteration tree would move that
/// denominator.
///
/// ## When the seeds cannot be separated
///
/// If the loop never ran (`upper_value_limit <= threshold + tolerance`, so the
/// watershed's buffered region never matched the output's — ITK's
/// `GetBufferedRegion() != region` short-circuit), or if it ran and the seeds
/// still share a basin, ITK re-runs the watershed at `lower`. The two seeds
/// then carry the same label, and the painting loop's `value == seed1Label`
/// branch wins first: the shared basin is painted entirely with
/// `replace_value1` and **nothing** receives `replace_value2`.
/// [`IsolatedWatershedResult::isolated_value`] is `lower` on every path.
///
/// Errors with [`FilterError::InvalidSeedIndex`] / [`FilterError::DimensionLength`]
/// for a bad seed, and propagates [`FilterError::WatershedSegmentWithoutEdges`]
/// when the gradient magnitude has a single watershed segment.
pub fn isolated_watershed(
    image: &Image,
    seed1: &[usize],
    seed2: &[usize],
    settings: IsolatedWatershedSettings,
) -> Result<IsolatedWatershedResult> {
    let size = image.size().to_vec();
    let seed1_offset = seed_flat_index(seed1, &size)?;
    let seed2_offset = seed_flat_index(seed2, &size)?;

    // `m_GradientMagnitude->SetInput(inputImage)`, output at RealType.
    let gm_values = gradient_magnitude_values(image, true)?;
    let gradient = image_from_f64(real_pixel_type(image.pixel_id()), &size, image, &gm_values)?;

    let mut lower = settings.threshold;
    let mut upper = settings.upper_value_limit;
    let mut guess = upper;

    let mut tree: Option<WatershedTree> = None;
    let mut labels: Option<Vec<u64>> = None;

    // Binary search for an upper waterlevel that separates the two seeds.
    while lower + settings.isolated_value_tolerance < guess {
        // `SetFloodLevel` only marks the tree generator modified when the new
        // level exceeds the highest already computed, and the first `guess` is
        // the largest one, so the tree is built exactly once.
        if tree.is_none() {
            tree = Some(watershed_tree(&gradient, settings.threshold, guess)?);
        }
        let current = tree.as_ref().expect("just built").relabel(guess);
        if current[seed1_offset] == current[seed2_offset] {
            upper = guess;
        } else {
            lower = guess;
        }
        guess = (upper + lower) / 2.0;
        labels = Some(current);
    }

    // "If the watershed basins are not separated or if the upper/lower
    // threshold were not valid, then use lower." The buffered-region test is
    // true exactly when the loop never ran, and then no tree exists yet.
    let final_labels = match labels {
        Some(l) if l[seed1_offset] != l[seed2_offset] => l,
        _ => {
            let tree = match tree {
                Some(t) => t,
                None => watershed_tree(&gradient, settings.threshold, lower)?,
            };
            tree.relabel(lower)
        }
    };

    let seed1_label = final_labels[seed1_offset];
    let seed2_label = final_labels[seed2_offset];
    let painted: Vec<u8> = final_labels
        .iter()
        .map(|&value| {
            if value == seed1_label {
                settings.replace_value1
            } else if value == seed2_label {
                settings.replace_value2
            } else {
                0
            }
        })
        .collect();

    let mut out = Image::from_vec(&size, painted)?;
    out.copy_geometry_from(image);
    Ok(IsolatedWatershedResult {
        image: out,
        isolated_value: lower,
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    fn img(size: &[usize], data: Vec<f32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn labels(image: &Image) -> Vec<u64> {
        image.scalar_slice::<u64>().unwrap().to_vec()
    }

    /// `Image<IdentifierType, D>` — `IdentifierType` is `uint64_t`.
    #[test]
    fn output_is_uint64_with_input_geometry() {
        let mut input = img(&[5, 1], vec![0.0, 2.0, 4.0, 2.0, 0.0]);
        input.set_spacing(&[2.0, 3.0]).unwrap();
        let out = watershed(&input, 0.0, 0.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt64);
        assert_eq!(out.size(), &[5, 1]);
        assert_eq!(out.spacing(), &[2.0, 3.0]);
    }

    /// `GradientDescent` seeds `minVal` from connectivity neighbor 0 and only
    /// lowers it on a **strict** `<`, so among equal-valued neighbors the one
    /// earliest in connectivity order — the most negative offset — wins.
    ///
    /// `[0, 2, 4, 2, 0]`, padded to 7x3 with a retaining wall of `max+1 = 5`.
    /// `LabelMinima` finds two single-pixel minima, at x=1 (label 1) and x=5
    /// (label 2). `GradientDescent` then walks the unlabeled pixels: x=2 falls
    /// left to label 1; x=3 (the ridge, value 4) sees x=2 and x=4 both at 2 and
    /// takes **x=2**, the -x neighbor, so it joins basin 1; x=4 falls right to
    /// label 2. Had the tie gone the other way the ridge would read `2`.
    #[test]
    fn gradient_descent_tie_prefers_the_negative_neighbor() {
        let input = img(&[5, 1], vec![0.0, 2.0, 4.0, 2.0, 0.0]);
        let out = watershed(&input, 0.0, 0.0).unwrap();
        assert_eq!(labels(&out), vec![1, 1, 1, 2, 2]);
    }

    /// Both basins of `[0, 2, 4, 2, 0]` bottom out at `0` and share a ridge at
    /// the image maximum, so each proposes `saliency = 4 - 0 = 4` while
    /// `CompileMergeList`'s cut is `level * maximumDepth = 1.0 * (4 - 0) = 4`.
    /// The test is `saliency < threshold`, strict, so neither merge is pushed:
    /// the tree is empty and even `level = 1.0` leaves two basins.
    #[test]
    fn symmetric_ridge_never_merges_even_at_level_one() {
        let input = img(&[5, 1], vec![0.0, 2.0, 4.0, 2.0, 0.0]);
        let out = watershed(&input, 0.0, 1.0).unwrap();
        assert_eq!(labels(&out), vec![1, 1, 1, 2, 2]);
    }

    /// `[0, 2, 2, 1, 5]`. `LabelMinima` labels x=1 (value 0) as minimum 1, then
    /// the plateau x=2,x=3 (both 2) as flat region 2, then x=4 (value 1) as
    /// minimum 3. The flat region's `bounds_min` is 0 — the value of x=1, its
    /// lowest neighbor — which is below the region's own value of 2, so
    /// `DescendFlatRegions` folds it into whatever label sits at that neighbor:
    /// label 1.
    ///
    /// Label 2 is thereby consumed and never appears in the output. ITK's
    /// labels are not contiguous, and this pins that.
    #[test]
    fn flat_region_descends_into_the_lower_basin() {
        let input = img(&[5, 1], vec![0.0, 2.0, 2.0, 1.0, 5.0]);
        let out = watershed(&input, 0.0, 0.0).unwrap();
        assert_eq!(labels(&out), vec![1, 1, 1, 3, 3]);
    }

    /// `Level` is consulted twice, with different denominators, and this is the
    /// single most surprising thing about `itk::WatershedImageFilter`.
    ///
    /// For `[0, 2, 2, 1, 5]` the initial segmentation is `{1: min 0}` and
    /// `{3: min 1}`, joined by one edge of height 2, with
    /// `maximumDepth = 5 - 0 = 5`. `CompileMergeList` keeps a merge when
    /// `saliency < level * 5`; `ExtractMergeHierarchy` applies it when
    /// `saliency <= level * 5`; but then `Relabeler` re-thresholds the finished
    /// tree at `level * tree.back().saliency`, a completely different scale.
    ///
    /// - `level = 0.5`: the tree generator's cut is 2.5, so both proposals
    ///   (saliency 2 for segment 1, saliency 1 for segment 3) survive and the
    ///   tree records `3 -> 1` at saliency 1. The relabeler's cut is
    ///   `0.5 * 1.0 = 0.5`, below that saliency — so nothing merges.
    /// - `level = 1.0`: the same one-entry tree, but the relabeler's cut is
    ///   `1.0 * 1.0 = 1.0`, and `saliency <= mergeLimit` now holds. Everything
    ///   collapses to one basin.
    #[test]
    fn level_is_applied_twice_once_per_stage() {
        let input = img(&[5, 1], vec![0.0, 2.0, 2.0, 1.0, 5.0]);
        assert_eq!(
            labels(&watershed(&input, 0.0, 0.5).unwrap()),
            vec![1, 1, 1, 3, 3],
            "the tree holds the merge, the relabeler declines to apply it"
        );
        assert_eq!(
            labels(&watershed(&input, 0.0, 1.0).unwrap()),
            vec![1, 1, 1, 1, 1],
            "the relabeler's cut finally reaches the tree's only saliency"
        );
    }

    /// `SegmentTable::PruneEdgeLists` advances the iterator *past* the first
    /// edge that exceeds `maximum_saliency` before erasing, so it keeps one
    /// edge too many. `CompileMergeList` calls it and then unconditionally
    /// dereferences `edge_list.front()`, so the off-by-one is load-bearing: it
    /// is the only reason an aggressively pruned table does not throw.
    ///
    /// At `level = 0.19` on `[0, 2, 2, 1, 5]` the prune threshold is
    /// `0.19 * 5 = 0.95`. Segment 1's only edge is at `2 - 0 = 2 > 0.95` and
    /// segment 3's at `2 - 1 = 1 > 0.95`, so a faithful `erase(e, end())` would
    /// empty **both** lists. The `++e` saves them; no merge is proposed
    /// (`saliency < 0.95` fails for both) and the two basins survive.
    #[test]
    fn prune_edge_lists_keeps_the_first_offending_edge() {
        let input = img(&[5, 1], vec![0.0, 2.0, 2.0, 1.0, 5.0]);
        let out = watershed(&input, 0.0, 0.19).unwrap();
        assert_eq!(labels(&out), vec![1, 1, 1, 3, 3]);
    }

    /// Directly: an edge at saliency 2 against a limit of 0.95 is kept, and
    /// everything after it is dropped.
    #[test]
    fn prune_edge_lists_truncates_after_the_offender() {
        let mut table = SegmentTable {
            segments: BTreeMap::from([(
                1,
                Segment {
                    min: 0.0,
                    edge_list: VecDeque::from(vec![
                        Edge {
                            label: 2,
                            height: 0.5,
                        },
                        Edge {
                            label: 3,
                            height: 2.0,
                        },
                        Edge {
                            label: 4,
                            height: 3.0,
                        },
                    ]),
                },
            )]),
            maximum_depth: 5.0,
        };
        prune_edge_lists(&mut table, 0.95);
        let kept: Vec<u64> = table.segments[&1]
            .edge_list
            .iter()
            .map(|e| e.label)
            .collect();
        assert_eq!(
            kept,
            vec![2, 3],
            "the height-2.0 offender is kept, 3.0 is not"
        );
    }

    /// `[3, 0, 0, 0, 3]`: the three zeros form one flat region whose
    /// `bounds_min` (3) is *above* its own value (0), so it is a genuine flat
    /// basin and `DescendFlatRegions` leaves it alone. The two 3s then slide
    /// into it, and the whole image is one segment with an empty adjacency
    /// list. `CompileMergeList` dereferences `edge_list.front()` on it and ITK
    /// throws; so does this port.
    #[test]
    fn flat_basin_leaves_one_segment_and_errors() {
        let input = img(&[5, 1], vec![3.0, 0.0, 0.0, 0.0, 3.0]);
        let err = watershed(&input, 0.0, 0.0).unwrap_err();
        assert!(
            matches!(err, FilterError::WatershedSegmentWithoutEdges { label: 1 }),
            "{err:?}"
        );
    }

    /// A constant image is one flat region covering everything: same error.
    #[test]
    fn flat_image_errors() {
        let input = img(&[4, 1], vec![7.0; 4]);
        let err = watershed(&input, 0.0, 0.0).unwrap_err();
        assert!(
            matches!(err, FilterError::WatershedSegmentWithoutEdges { label: 1 }),
            "{err:?}"
        );
    }

    /// `Threshold` clips the image from below at
    /// `threshold * (maximum - minimum) + minimum` before any labeling, which
    /// removes shallow minima and can move a basin boundary.
    ///
    /// `[0, 2, 1, 2, 0]` has three minima (x=1, x=3, x=5) and segments to
    /// `[1, 1, 2, 3, 3]`: at x=4 the descent sees x=3 at 1 and x=5 at 0 and
    /// takes the strictly lower x=5.
    ///
    /// At `threshold = 0.5` the clip level is `0.5 * (2 - 0) + 0 = 1`, so both
    /// zeros rise to 1. The three minima remain, but now x=4 sees x=3 at 1 and
    /// x=5 at 1 — a tie — and the `<` in `GradientDescent` hands it to x=3,
    /// the -x neighbor. The middle basin gains a pixel.
    #[test]
    fn threshold_clips_from_below_and_can_move_a_boundary() {
        let input = img(&[5, 1], vec![0.0, 2.0, 1.0, 2.0, 0.0]);
        assert_eq!(
            labels(&watershed(&input, 0.0, 0.0).unwrap()),
            vec![1, 1, 2, 3, 3]
        );
        assert_eq!(
            labels(&watershed(&input, 0.5, 0.0).unwrap()),
            vec![1, 1, 2, 2, 3]
        );
    }

    /// A `Threshold` high enough to flood every minimum but one leaves a single
    /// segment, which is the same fatal error as a flat image. For
    /// `[0, 2, 2, 1, 5]` the clip level at `threshold = 0.5` is 2.5, which
    /// swallows all of `0, 2, 2, 1` into one plateau below the lone peak.
    #[test]
    fn threshold_can_flood_the_image_into_one_segment() {
        let input = img(&[5, 1], vec![0.0, 2.0, 2.0, 1.0, 5.0]);
        let err = watershed(&input, 0.5, 0.0).unwrap_err();
        assert!(
            matches!(err, FilterError::WatershedSegmentWithoutEdges { label: 1 }),
            "{err:?}"
        );
    }

    /// `itkSetClampMacro(Level, 0.0, 1.0)`.
    #[test]
    fn level_is_clamped_to_the_unit_interval() {
        let input = img(&[5, 1], vec![0.0, 2.0, 2.0, 1.0, 5.0]);
        assert_eq!(
            labels(&watershed(&input, 0.0, 5.0).unwrap()),
            vec![1, 1, 1, 1, 1],
            "level 5.0 clamps to 1.0"
        );
        assert_eq!(
            labels(&watershed(&input, 0.0, -3.0).unwrap()),
            vec![1, 1, 1, 3, 3],
            "level -3.0 clamps to 0.0"
        );
    }

    /// `itkSetClampMacro(Threshold, 0.0, 1.0)`. A clamped threshold of 1.0
    /// clips every pixel up to the image maximum, which is the flat-image error.
    #[test]
    fn threshold_is_clamped_to_the_unit_interval() {
        let input = img(&[5, 1], vec![0.0, 2.0, 2.0, 1.0, 5.0]);
        assert_eq!(
            labels(&watershed(&input, -1.0, 0.5).unwrap()),
            labels(&watershed(&input, 0.0, 0.5).unwrap()),
            "threshold -1.0 clamps to 0.0"
        );
        for threshold in [1.0, 2.0] {
            let err = watershed(&input, threshold, 0.0).unwrap_err();
            assert!(
                matches!(err, FilterError::WatershedSegmentWithoutEdges { label: 1 }),
                "threshold {threshold} floods the image: {err:?}"
            );
        }
    }

    #[test]
    fn empty_image() {
        let input = img(&[0, 0], vec![]);
        let out = watershed(&input, 0.0, 0.0).unwrap();
        assert_eq!(out.size(), &[0, 0]);
        assert!(labels(&out).is_empty());
    }

    /// `Segmenter::Threshold` writes through `static_cast<InputPixelType>`, so
    /// on an integer image the clip level truncates toward zero.
    ///
    /// `[0, 2, 1, 2, 0]` at `threshold = 0.95` gives a clip level of
    /// `0.95 * 2 = 1.9`, which an integer image rounds down to `1` while a
    /// float image keeps. Both segment identically — the topology is the same —
    /// but the clipped pixels differ, and the segment minima record it.
    #[test]
    fn integer_input_truncates_the_clip_level() {
        let values = [0.0, 2.0, 1.0, 2.0, 0.0];
        let (_, _, int_table) = segment(&values, &[5, 1], PixelId::UInt8, 0.95);
        let (_, _, float_table) = segment(&values, &[5, 1], PixelId::Float32, 0.95);
        assert_eq!(int_table.segments[&1].min, 1.0);
        assert_eq!(float_table.segments[&1].min, 1.9);
    }

    /// "Cap the maximum in the image so that we can always define a pixel value
    /// that is one greater than the maximum value in the image": an integer
    /// image containing its own type maximum has that maximum decremented, and
    /// `maximumDepth` — the denominator of every saliency test — shrinks by one
    /// with it. A float image of the same values is untouched.
    #[test]
    fn integer_input_caps_the_maximum_at_type_max() {
        let values = [0.0, 255.0, 128.0];
        let (_, _, int_table) = segment(&values, &[3, 1], PixelId::UInt8, 0.0);
        let (_, _, float_table) = segment(&values, &[3, 1], PixelId::Float32, 0.0);
        assert_eq!(int_table.maximum_depth, 254.0);
        assert_eq!(float_table.maximum_depth, 255.0);
    }

    /// `EquivalencyTable::Add` swaps its arguments so the key is always the
    /// larger label. A group therefore ends up named by its minimum, whichever
    /// direction the merges were recorded in.
    #[test]
    fn equivalency_table_names_a_group_by_its_minimum_label() {
        let mut table = EquivalencyTable::default();
        table.add(1, 7); // recorded "1 merges into 7"
        table.add(7, 3);
        table.flatten();
        assert_eq!(table.lookup(7), 1);
        assert_eq!(table.lookup(3), 1);
        assert_eq!(table.lookup(1), 1, "the minimum is its own representative");
    }

    /// `OneWayEquivalencyTable::Add` keeps the direction it is given and never
    /// overwrites an existing entry.
    #[test]
    fn one_way_equivalency_table_keeps_its_direction() {
        let mut table = OneWayEquivalencyTable::default();
        table.add(1, 7);
        table.add(1, 3); // ignored: 1 already resolves
        assert_eq!(table.recursive_lookup(1), 7);
        assert_eq!(table.recursive_lookup(7), 7);
    }

    /// `SegmentTree::merge_comp` is `b.saliency < a.saliency`, which turns
    /// `std::make_heap`'s max-heap into a **min**-heap on saliency: `Front()`
    /// is the least-salient merge, the one that happens first.
    #[test]
    fn merge_heap_is_a_min_heap_on_saliency() {
        let mut heap = vec![
            Merge {
                from: 1,
                to: 2,
                saliency: 5.0,
            },
            Merge {
                from: 3,
                to: 4,
                saliency: 1.0,
            },
            Merge {
                from: 5,
                to: 6,
                saliency: 3.0,
            },
        ];
        make_heap(&mut heap);
        assert_eq!(heap[0].saliency, 1.0);
        pop_heap(&mut heap);
        heap.pop();
        assert_eq!(heap[0].saliency, 3.0);
        pop_heap(&mut heap);
        heap.pop();
        assert_eq!(heap[0].saliency, 5.0);
    }

    /// `Math::AlmostEquals` resolves to exact `==` when either argument is an
    /// integer type, and to `FloatAlmostEqual` (4 ULPs) for float and double.
    /// `LabelMinima` uses it to decide flat-region membership, so on a float
    /// image two pixels four ULPs apart are the *same* plateau.
    #[test]
    fn almost_equals_is_exact_for_integers_and_ulp_based_for_floats() {
        let one = 1.0f32;
        let four_ulps = f32::from_bits(one.to_bits() + 4);
        assert!(f32_almost_equal(one, four_ulps));
        assert!(!f32_almost_equal(one, f32::from_bits(one.to_bits() + 5)));

        assert!(AlmostEquals::for_pixel_type(PixelId::Float32).eq(one as f64, four_ulps as f64));
        assert!(
            !AlmostEquals::for_pixel_type(PixelId::UInt8).eq(one as f64, four_ulps as f64),
            "integer comparison is exact `==`"
        );
    }

    // ---- itk::IsolatedWatershedImageFilter --------------------------------

    fn u8_img(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    /// `[0, 0, 10, 10, 10, 0, 0]`, spacing 1. Central differences under a
    /// zero-flux Neumann boundary give a gradient magnitude of
    /// `[0, 5, 5, 0, 5, 5, 0]` (the y-derivative vanishes on a single row).
    ///
    /// That surface has three minima at 0 and two plateaus at 5. Each plateau
    /// descends into the basin on its left, so the initial segmentation is
    /// `[1, 1, 1, 3, 3, 3, 5]`. Every segment's lowest edge is at height 5 over
    /// a minimum of 0, so every saliency equals `maximumDepth = 5 - 0`, and
    /// `saliency < level * maximumDepth` is false for every `level <= 1`: the
    /// merge tree is empty at any waterlevel and the three basins never merge.
    fn two_blob_gradient_basins() -> Image {
        u8_img(&[7, 1], vec![0, 0, 10, 10, 10, 0, 0])
    }

    /// The bisection's first `guess` is `upper_value_limit`, which already
    /// separates the seeds, so `lower` rises to it and the loop exits after one
    /// iteration. `GetIsolatedValue()` is that `lower`.
    #[test]
    fn separable_seeds_are_painted_with_both_replace_values() {
        let out = isolated_watershed(
            &two_blob_gradient_basins(),
            &[0, 0],
            &[6, 0],
            IsolatedWatershedSettings::default(),
        )
        .unwrap();
        assert_eq!(out.image.pixel_id(), PixelId::UInt8);
        assert_eq!(
            out.image.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 0, 0, 0, 2],
            "seed1's basin takes ReplaceValue1, seed2's ReplaceValue2,              the middle basin neither"
        );
        assert_eq!(out.isolated_value, 1.0);
    }

    /// Both seeds sit in basin 1, and no waterlevel splits it, so the loop
    /// halves `guess` down to the tolerance without ever raising `lower`.
    ///
    /// `seed1Label == seed2Label` then makes the painting loop's
    /// `value == seed1Label` branch win first: the shared basin is painted
    /// entirely with `ReplaceValue1` and **nothing** receives `ReplaceValue2`.
    #[test]
    fn inseparable_seeds_paint_only_replace_value1() {
        let out = isolated_watershed(
            &two_blob_gradient_basins(),
            &[0, 0],
            &[1, 0],
            IsolatedWatershedSettings::default(),
        )
        .unwrap();
        assert_eq!(
            out.image.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 0, 0, 0, 0],
            "ReplaceValue2 never appears"
        );
        assert_eq!(out.isolated_value, 0.0, "lower never moved");
    }

    /// `ReplaceValue1` and `ReplaceValue2` are plumbed straight through, and
    /// everything outside the two seeded basins gets `OutputImagePixelType{}`.
    #[test]
    fn replace_values_are_plumbed_through() {
        let settings = IsolatedWatershedSettings {
            replace_value1: 7,
            replace_value2: 9,
            ..IsolatedWatershedSettings::default()
        };
        let out =
            isolated_watershed(&two_blob_gradient_basins(), &[0, 0], &[6, 0], settings).unwrap();
        assert_eq!(
            out.image.scalar_slice::<u8>().unwrap(),
            &[7, 7, 7, 0, 0, 0, 9]
        );
    }

    /// When `upper_value_limit <= threshold + tolerance` the `while` never runs.
    /// ITK detects this by the watershed's buffered region still being empty and
    /// re-runs it at `lower`; the seeds may well be separated there anyway, but
    /// `GetIsolatedValue()` still reports `lower` — the bisection never refined
    /// anything.
    #[test]
    fn a_bisection_that_never_iterates_still_segments_at_lower() {
        let settings = IsolatedWatershedSettings {
            threshold: 0.0,
            upper_value_limit: 0.001,
            isolated_value_tolerance: 0.001,
            ..IsolatedWatershedSettings::default()
        };
        let out =
            isolated_watershed(&two_blob_gradient_basins(), &[0, 0], &[6, 0], settings).unwrap();
        assert_eq!(
            out.image.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 0, 0, 0, 2]
        );
        assert_eq!(out.isolated_value, 0.0);
    }

    /// `VerifyInputInformation`: "Seed1 is not within the input image!"
    #[test]
    fn a_seed_outside_the_image_is_an_error() {
        let input = two_blob_gradient_basins();
        let settings = IsolatedWatershedSettings::default();
        for seed in [[7, 0], [0, 1]] {
            let err = isolated_watershed(&input, &seed, &[6, 0], settings).unwrap_err();
            assert!(
                matches!(err, FilterError::InvalidSeedIndex { .. }),
                "seed {seed:?}: {err:?}"
            );
        }
        let err = isolated_watershed(&input, &[0, 0], &[6, 6], settings).unwrap_err();
        assert!(
            matches!(err, FilterError::InvalidSeedIndex { .. }),
            "{err:?}"
        );
    }

    /// A seed with the wrong number of components never reaches ITK's bounds
    /// check, because `IndexType` is dimension-typed.
    #[test]
    fn a_seed_of_the_wrong_dimension_is_an_error() {
        let input = two_blob_gradient_basins();
        let err = isolated_watershed(&input, &[0], &[6, 0], IsolatedWatershedSettings::default())
            .unwrap_err();
        assert!(
            matches!(
                err,
                FilterError::DimensionLength {
                    expected: 2,
                    got: 1
                }
            ),
            "{err:?}"
        );
    }

    /// The gradient magnitude runs at `NumericTraits<InputPixelType>::RealType`,
    /// which is `double` for **every** scalar input — `float` included
    /// (`NumericTraits<float>::RealType` is `double`). SimpleITK's standalone
    /// `GradientMagnitudeImageFilter` yaml instead fixes the output at `float`,
    /// so this is not the same seam.
    #[test]
    fn the_gradient_magnitude_runs_at_real_type() {
        assert_eq!(real_pixel_type(PixelId::UInt8), PixelId::Float64);
        assert_eq!(real_pixel_type(PixelId::Int16), PixelId::Float64);
        assert_eq!(real_pixel_type(PixelId::Float64), PixelId::Float64);
        assert_eq!(real_pixel_type(PixelId::Float32), PixelId::Float64);
    }

    /// A constant image has a zero gradient magnitude everywhere, which is the
    /// flat-image single-segment error, propagated out of the watershed.
    #[test]
    fn a_constant_image_propagates_the_single_segment_error() {
        let input = u8_img(&[4, 1], vec![3; 4]);
        let err = isolated_watershed(
            &input,
            &[0, 0],
            &[3, 0],
            IsolatedWatershedSettings::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, FilterError::WatershedSegmentWithoutEdges { label: 1 }),
            "{err:?}"
        );
    }

    /// `FloatAlmostEqual` tests `|x1 - x2| <= maxAbsoluteDifference` **before**
    /// it tests the signbit, so the sign-mismatch rejection never sees a pair
    /// that close to zero: `0.0` and `-0.0` are equal, and so is any pair
    /// within `0.1 * epsilon` however many ULPs apart.
    #[test]
    fn float_almost_equals_short_circuits_on_the_absolute_difference() {
        assert!(f32_almost_equal(0.0, -0.0));
        assert!(f64_almost_equal(0.0, -0.0));

        // 1e-9 and 2e-9 are millions of ULPs apart, but `0.1 * f32::EPSILON`
        // is ~1.19e-8, which swallows their difference.
        assert!((f32_as_ulp(2e-9) - f32_as_ulp(1e-9)).abs() > 4);
        assert!(f32_almost_equal(1e-9, 2e-9));
    }

    /// The signbit rejection does bite once the pair is far enough from zero
    /// for the absolute-difference shortcut to miss.
    #[test]
    fn float_almost_equals_rejects_a_sign_mismatch() {
        assert!(!f32_almost_equal(1.0, -1.0));
        assert!(!f64_almost_equal(1.0, -1.0));
    }
}
