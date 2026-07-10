//! Fast marching on ITK's *newer* framework, with topology constraints.
//!
//! Port of `itk::FastMarchingImageFilterBase` on `itk::FastMarchingBase`
//! (`itkFastMarchingBase.h/.hxx`, `itkFastMarchingImageFilterBase.h/.hxx`,
//! `itkFastMarchingTraits.h`, `itkNodePair.h`) driven by
//! `itk::FastMarchingThresholdStoppingCriterion`
//! (`itkFastMarchingThresholdStoppingCriterion.h`), with the API surface
//! `FastMarchingBaseImageFilter.yaml` declares: `TrialPoints`,
//! `NormalizationFactor`, `StoppingValue`, `TopologyCheck`,
//! `InitialTrialValues`.
//!
//! The input is the *speed* image; the output is the arrival-time field. The
//! yaml pins `output_pixel_type: float`, so the output is **always**
//! [`PixelId::Float32`] regardless of the speed image's pixel type — this
//! filter does not route through the crate's `real_pixel_id`, and none of the
//! `NumericTraits<T>::RealType` question (doc/upstream-findings.md §5.6)
//! applies to it. Every value the march writes therefore passes through
//! `static_cast<float>`, which this port reproduces by narrowing through
//! `f32` before storing.
//!
//! This is *not* [`crate::fast_marching`]'s filter with extra options: the two
//! class hierarchies differ in the constants they use, in where they test the
//! stopping value, and in how they walk neighbors. The differences that change
//! output are called out below.
//!
//! ## The scheme
//!
//! Every pixel carries a label (`FastMarchingTraitsBase::LabelType`,
//! `itkFastMarchingTraits.h:78-93`) and a value, initialized to
//! [`LARGE_VALUE`]. Trial points seed a min-heap. Each round pops the smallest
//! trial value, freezes it as *alive*, and recomputes every non-alive face
//! neighbor with the upwind quadratic `Solve`
//! (`itkFastMarchingImageFilterBase.hxx:246-302`):
//!
//! For each axis `j`, take the smallest value among that axis's *alive*
//! neighbors (or [`LARGE_VALUE`] when it has none), sort those `dim`
//! candidates ascending, and accumulate them one at a time while the running
//! solution still dominates the candidate:
//!
//! ```text
//! aa += 1/h_j^2                  bb += v_j/h_j^2                cc += v_j^2/h_j^2
//! solution = (sqrt(bb^2 - aa*cc) + bb) / aa       // the larger root
//! ```
//!
//! A solution that does not come out below [`LARGE_VALUE`] is not written
//! back, so the pixel stays far (`.hxx:181-189`).
//!
//! ## Where this filter differs from `FastMarchingImageFilter`
//!
//! [`crate::fast_marching`] ports the *old* `itk::FastMarchingImageFilter`.
//! Four differences change results:
//!
//! 1. **`m_LargeValue` is the pixel type's max, not half of it.**
//!    `FastMarchingBase`'s constructor takes
//!    `NumericTraits<OutputPixelType>::max()` (`itkFastMarchingBase.hxx:35`);
//!    the old filter takes `max()/2` (`itkFastMarchingImageFilter.hxx:37`).
//!    Unreached pixels here hold `f32::MAX`, there `f32::MAX / 2`.
//! 2. **The stopping test is `>=`, and it runs before the point is accepted.**
//!    `FastMarchingThresholdStoppingCriterion::IsSatisfied` is
//!    `m_CurrentValue >= m_Threshold`
//!    (`itkFastMarchingThresholdStoppingCriterion.h:60-64`), evaluated on the
//!    popped node *before* `CheckTopology` / `SetLabelValueForGivenNode`
//!    (`itkFastMarchingBase.hxx:148-153`). The old filter breaks on
//!    `value > m_StoppingValue`. A trial value exactly equal to
//!    `stopping_value` therefore ends the march here and is accepted there.
//! 3. **Zero speed no longer blocks the front.** `Solve` guards the
//!    reciprocal: when `speed/normalization_factor` is `FloatAlmostEqual` to
//!    `0.0` it uses `-sqr(1/(cc + itk::Math::eps))` instead of `-sqr(1/cc)`
//!    (`.hxx:262-269`). `itk::Math::eps` is `DBL_EPSILON` (`itkMath.h:119`)
//!    and `FloatAlmostEqual`'s `maxAbsoluteDifference` defaults to
//!    `0.1 * DBL_EPSILON` (`itkMath.h:329-336`), so a zero-speed pixel gets
//!    `cc = -2^104` and an arrival time of `2^52`, finite and below
//!    `f32::MAX`. It is written, pushed, and goes alive — the front crosses.
//!    The old filter's `-sqr(1/0)` is `-inf`, giving `+inf` and a pixel that
//!    never enters the heap.
//! 4. **The negative-discriminant guard is `< itk::Math::eps`, not `< 0.0`**
//!    (`.hxx:287-291` vs `itkFastMarchingImageFilter.hxx:428-435`). A
//!    discriminant that underflows to exactly `0.0` raises
//!    [`FilterError::NegativeDiscriminant`] here and does not there.
//!
//! ## Upstream defect: a seed on the image border cannot leave it
//!
//! `UpdateNeighbors` guards the *assignment* of the neighbor index, not the
//! read (`itkFastMarchingImageFilterBase.hxx:145-168`):
//!
//! ```text
//! const IndexValueType v = iNode[j];
//! NodeType neighIndex = iNode;
//! for (int s = -1; s < 2; s += 2) {
//!   if ((v > start) && (v < last)) { neighIndex[j] = v + s; }   // :154-157
//!   const unsigned char label = m_LabelImage->GetPixel(neighIndex);
//!   if (label != Alive && label != InitialTrial && label != Forbidden)
//!     this->UpdateValue(oImage, neighIndex);
//! }
//! ```
//!
//! When `v` sits on either end of axis `j`, the condition is false for **both**
//! values of `s`, `neighIndex` stays equal to `iNode`, and the label read is
//! the center's own — which `GenerateData` has just set to `Alive`
//! (`itkFastMarchingBase.hxx:163-166`). So the in-bounds neighbor at `v ± 1`
//! is never updated: **a node on the boundary of axis `j` never propagates
//! along axis `j`.**
//!
//! The consequences are not cosmetic. A single trial point on a face of the
//! image confines the entire march to that face (a seed at `x == 0` in 2-D
//! never reaches `x == 1`); a lone corner seed marches nowhere at all. For an
//! interior seed the defect is invisible, because the only neighbor a border
//! node fails to update along its normal axis is the interior node it was
//! reached from, which is already alive.
//!
//! Two things establish this as a defect rather than a design:
//! `GetInternalNodesUsed`, in the same file, bounds-checks per `s`
//! (`.hxx:212-232`); and the old `FastMarchingImageFilter::UpdateNeighbors`
//! guards the two neighbors separately, so it does reach `start + 1` from
//! `start` (`itkFastMarchingImageFilter.hxx:308-341`). It is reproduced here
//! bit-for-bit and pinned by
//! `a_border_seed_never_propagates_along_the_border_normal_axis`.
//!
//! ## Upstream defect: `NoHandles` is `Strict`
//!
//! `CheckTopology` (`.hxx:306-391`) computes two predicates and dispatches:
//!
//! - `Strict` rejects the node when either `DoesVoxelChangeViolateWellComposedness`
//!   or `DoesVoxelChangeViolateStrictTopology` holds (`.hxx:316-322`).
//! - `NoHandles` rejects on a well-composedness violation (`.hxx:326-331`);
//!   on a *strict* violation it instead asks whether the merge would create a
//!   handle, by comparing the connected-component labels of the two alive
//!   neighbors that face each other (`.hxx:346-363`). Equal labels mean the
//!   two sides already belong to one component, so joining them adds a handle
//!   and the node is rejected (`.hxx:364-369`); different labels mean two
//!   distinct components merge, which is allowed, and the labels are unified
//!   (`.hxx:371-380`).
//!
//! `m_ConnectedComponentImage` is seeded **only** from `m_AlivePoints`
//! (`.hxx:449-452`) and is never written again as the front advances — the one
//! remaining write is the label-unification loop above, which is downstream of
//! the comparison. `FastMarchingBaseImageFilter.yaml` declares no `AlivePoints`
//! member, so through SimpleITK the image is allocated zero-initialized
//! (`.hxx:418-423`), passed through `ConnectedComponentImageFilter` +
//! `RelabelComponentImageFilter` on an all-background input (`.hxx:489-507`),
//! and stays identically zero for the whole march.
//!
//! Therefore `ItC.GetNext(d) == ItC.GetPrevious(d)` is always `0 == 0`,
//! `doesChangeCreateHandle` is unconditionally true, and `NoHandles` rejects
//! exactly the nodes `Strict` rejects. This port implements the two modes with
//! the one predicate `wellComposednessViolation || strictTopologyViolation`
//! rather than transcribing an unreachable branch, and pins the equality with
//! `no_handles_matches_strict_on_a_merge` and
//! `no_handles_matches_strict_in_3d`.
//!
//! ## Topology: what the two predicates test
//!
//! Both read the label image through a radius-1 `NeighborhoodIterator` whose
//! default boundary condition is `ZeroFluxNeumannBoundaryCondition`, i.e. an
//! out-of-image stencil position reads the nearest in-image pixel. (The
//! iterator's inner bounds come from the *image's* buffered region, not from
//! the region it is constructed over — `itkConstNeighborhoodIterator.hxx:621-641`
//! — which is why `IsChangeWellComposed3D` passing `GetRequestedRegion()`
//! (`.hxx:779`) where its 2-D sibling passes `GetBufferedRegion()` (`.hxx:619`)
//! is a harmless inconsistency: `m_LabelImage`'s requested region is the
//! default-constructed empty one, but it only feeds `m_Bound`, which
//! `SetLocation` + `GetPixel` never read.)
//!
//! At the moment `CheckTopology` runs, the center pixel is not yet alive; both
//! predicates model the *change* by flipping the center bit.
//!
//! - `DoesVoxelChangeViolateStrictTopology` (`.hxx:579-611`) counts alive face
//!   neighbors and axes whose two face neighbors are both alive. The node
//!   violates strict topology when at least one such axis exists and *every*
//!   alive face neighbor belongs to one — i.e. the node is exactly a junction
//!   of two fronts.
//! - `DoesVoxelChangeViolateWellComposedness` (`.hxx:562-575`) is the negation
//!   of `IsChangeWellComposed2D` / `3D`. In 2-D (`.hxx:615-696`) it tests four
//!   critical configurations over 4 rotations and 2 reflections of the 3×3
//!   label neighborhood; the C1 case is the classic diagonal contact (the two
//!   opposite diagonal neighbors alive, the faces they share not alive). In
//!   3-D (`.hxx:773-859`) it tests 12 C1 index quadruples and 8 C2 index
//!   octuples of the 3×3×3 neighborhood. Note the two dimensions use opposite
//!   bit conventions — 2-D bits are "not alive" (`.hxx:630`), 3-D bits are
//!   "alive" (`.hxx:788`) — with the center bit flipped in both.
//!
//! For `ImageDimension` outside `{2, 3}` `CheckTopology` only warns and accepts
//! (`.hxx:386-387`), so `topology_check` is inert on 1-D and 4-D images.
//!
//! A rejected node takes `m_TopologyValue`, which is `m_LargeValue`
//! (`itkFastMarchingBase.hxx:36`), and the `Topology` label (`.hxx:319-321`).
//! `Topology` is not in `UpdateNeighbors`'s exclusion set, so a later alive
//! neighbor re-runs `UpdateValue` on it, overwriting the value and relabelling
//! it `Trial`; it is then re-checked when it pops again.
//!
//! ## Heap staleness and tie order
//!
//! `UpdateValue` cannot decrease a node already in the heap, so ITK pushes a
//! *new* node and leaves the defunct one behind. A popped node is accepted only
//! if its value still equals the value stored in the output image
//! (`Math::ExactlyEquals`, `itkFastMarchingBase.hxx:143`) and the pixel is not
//! already alive. This port keeps that, storing values already narrowed to
//! `f32` so the exact-equality test compares like for like.
//!
//! Upstream's heap is `std::priority_queue<NodePairType, std::vector<...>,
//! std::greater<NodePairType>>` (`itkFastMarchingBase.h:259-262`) over a
//! comparison that reads the value alone (`itkNodePair.h:96-100`). Equal-valued
//! nodes therefore pop in an order fixed by `std::push_heap`/`std::pop_heap`'s
//! sift pattern, which the C++ standard does not specify and which differs
//! between libstdc++ and libc++. **This port breaks value ties by ascending
//! flat index**, a documented divergence from an upstream behavior that is not
//! portable enough to have a bit-for-bit target. Without a topology constraint
//! the tie order does not reach the output — a node made alive at value `v`
//! only ever updates neighbors with a newly-alive value of exactly `v`, and
//! admitting an alive neighbor whose value equals a node's current solution
//! leaves that solution unchanged. Under `Strict`/`NoHandles` it can: which of
//! two equal-valued nodes goes alive first decides which one is later seen as
//! a junction.
//!
//! ## Other upstream observations, not reachable through this API
//!
//! - `FastMarchingBase::m_InverseSpeed` is initialized to `-1.0`
//!   (`itkFastMarchingBase.hxx:32`) and never recomputed from
//!   `m_SpeedConstant`, unlike the old filter's
//!   `SetSpeedConstant` (`itkFastMarchingImageFilter.h:305`). `SetSpeedConstant`
//!   is therefore inert in the new framework. It only matters when no input
//!   image is set (`.hxx:259`); SimpleITK always sets one
//!   (`number_of_inputs: 1`).
//! - `Initialize` throws "No Trial Nodes" only when the container pointer is
//!   null (`itkFastMarchingBase.hxx:66-69`). SimpleITK's cast always builds a
//!   container, so an **empty** trial list is not an error: the output is
//!   [`LARGE_VALUE`] everywhere.
//! - Out-of-image trial points are silently dropped by
//!   `InitializeOutput` (`.hxx:522`), but they still occupy a position in the
//!   container that `InitialTrialValues` indexes against (the yaml's
//!   `custom_itk_cast` walks `filter->GetTrialPoints()`), so
//!   `initial_trial_values` is positional against the **full**
//!   `trial_points` list.
//!
//! ## Divergences
//!
//! - Value ties in the heap, as described above.
//! - `criterion->SetThreshold(m_StoppingValue)` narrows a `double` to `float`.
//!   A `stopping_value` outside `float`'s range makes that conversion
//!   undefined in C++; here `as f32` yields an infinity.
//! - A `NaN` speed makes `Solve` return `NaN`, which fails `< m_LargeValue` and
//!   is not written — the same outcome as upstream, reached without a
//!   `NaN`-ordered comparison.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use sitk_core::{Image, PixelId};

use crate::error::{FilterError, Result};
use crate::fast_marching::{check_normalization_factor, strides};
use crate::image_from_f64;

/// `m_LargeValue`: `NumericTraits<OutputPixelType>::max()` for the yaml's
/// `output_pixel_type: float` (`itkFastMarchingBase.hxx:35`). Pixels the front
/// never reaches, and pixels a topology constraint rejected, hold this value.
///
/// Note it is the pixel type's **full** maximum, where
/// [`crate::fast_marching::large_value`] is half of it.
pub const LARGE_VALUE: f64 = f32::MAX as f64;

/// `itk::Math::eps` (`itkMath.h:119`), `std::numeric_limits<double>::epsilon()`.
const ITK_MATH_EPS: f64 = f64::EPSILON;

/// `FloatAlmostEqual`'s default `maxAbsoluteDifference` (`itkMath.h:334-335`).
/// `Solve`'s only use of `FloatAlmostEqual` compares against `0.0`, where the
/// ULP arm can never fire, so this bound decides the branch on its own.
const FLOAT_ALMOST_EQUAL_ZERO: f64 = 0.1 * f64::EPSILON;

/// `FastMarchingTraitsEnums::TopologyCheck` (`itkFastMarchingBase.h:43-48`).
///
/// `NoHandles` and `Strict` behave identically through this API; see the
/// module doc's "`NoHandles` is `Strict`".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TopologyCheck {
    /// No constraint: every popped node is accepted.
    #[default]
    Nothing,
    /// Fronts may merge, but must not create a handle.
    NoHandles,
    /// Fronts must not merge.
    Strict,
}

/// `FastMarchingTraitsBase::LabelType` (`itkFastMarchingTraits.h:78-93`), minus
/// `Forbidden`: `FastMarchingBaseImageFilter.yaml` declares no `ForbiddenPoints`
/// member, so `InitializeOutput`'s forbidden branch (`.hxx:462-483`) never runs
/// and the label never arises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Far,
    Alive,
    Trial,
    InitialTrial,
    Topology,
}

/// The scalar members `FastMarchingBaseImageFilter.yaml` declares, besides the
/// two point lists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastMarchingBaseSettings {
    /// Divides the speed image, so integer images can carry a speed. Must be
    /// `>= f64::EPSILON` (`itkFastMarchingBase.hxx:74-77`).
    pub normalization_factor: f64,
    /// `FastMarchingThresholdStoppingCriterion::SetThreshold`. The march ends
    /// when the smallest trial value is `>=` this.
    pub stopping_value: f64,
    /// `m_TopologyCheck`.
    pub topology_check: TopologyCheck,
}

impl Default for FastMarchingBaseSettings {
    /// The yaml's defaults: `NormalizationFactor: 1.0`,
    /// `StoppingValue: std::numeric_limits<float>::max()/2.0`,
    /// `TopologyCheck: Nothing`.
    fn default() -> Self {
        Self {
            normalization_factor: 1.0,
            stopping_value: LARGE_VALUE / 2.0,
            topology_check: TopologyCheck::Nothing,
        }
    }
}

/// A node on the trial heap: `(value, flat index)`.
///
/// [`Ord`] is reversed so [`BinaryHeap`]'s max-heap pops the smallest value,
/// matching `std::priority_queue<_, _, std::greater<NodePairType>>`. Equal
/// values pop smallest-flat-index first; see the module doc's "Heap staleness
/// and tie order" for why that is a divergence and when it can be observed.
#[derive(Debug)]
struct TrialNode {
    value: f64,
    index: usize,
}

impl Ord for TrialNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .total_cmp(&self.value)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for TrialNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TrialNode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for TrialNode {}

fn sqr(x: f64) -> f64 {
    x * x
}

/// `sitkSTLVectorToITK<NodeType>`: fewer than `dim` elements is "Unable to
/// convert vector to ITK type". The yaml emits its member casts in declaration
/// order and `TrialPoints` is first, so this rejection precedes
/// `SetNormalizationFactor` and hence `Initialize`'s own check.
fn check_point_lengths(points: &[Vec<u32>], dim: usize) -> Result<()> {
    for point in points {
        if point.len() < dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: point.len(),
            });
        }
    }
    Ok(())
}

/// The flat index of a trial point, or `None` when it falls outside the
/// buffered region (`InitializeOutput`'s `m_BufferedRegion.IsInside(idx)`).
fn flat_index(point: &[u32], dim: usize, size: &[usize], strides: &[usize]) -> Option<usize> {
    point
        .iter()
        .take(dim)
        .zip(size)
        .all(|(&c, &e)| (c as usize) < e)
        .then(|| {
            point
                .iter()
                .take(dim)
                .zip(strides)
                .map(|(&c, &s)| c as usize * s)
                .sum()
        })
}

/// The 3×3(×3) stencil offset of neighborhood index `n`, first axis fastest —
/// `itk::Neighborhood`'s linear ordering.
fn neighborhood_offset(n: usize, dim: usize) -> [i64; 3] {
    let mut offset = [0i64; 3];
    let mut stride = 1usize;
    for slot in offset.iter_mut().take(dim) {
        *slot = ((n / stride) % 3) as i64 - 1;
        stride *= 3;
    }
    offset
}

/// `InitializeIndices2D` (`itkFastMarchingImageFilterBase.hxx:700-745`): the
/// identity and the three 90° rotations of the 3×3 neighborhood.
const ROTATION_INDICES: [[usize; 9]; 4] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [8, 7, 6, 5, 4, 3, 2, 1, 0],
    [6, 3, 0, 7, 4, 1, 8, 5, 2],
];

/// `InitializeIndices2D` (`itkFastMarchingImageFilterBase.hxx:747-768`).
const REFLECTION_INDICES: [[usize; 9]; 2] =
    [[6, 7, 8, 3, 4, 5, 0, 1, 2], [2, 1, 0, 5, 4, 3, 8, 7, 6]];

/// `InitializeIndices3D`'s `m_C1Indices` (`itkFastMarchingImageFilterBase.hxx:874-932`).
const C1_INDICES: [[usize; 4]; 12] = [
    [1, 13, 4, 10],
    [9, 13, 10, 12],
    [3, 13, 4, 12],
    [4, 14, 5, 13],
    [12, 22, 13, 21],
    [13, 23, 14, 22],
    [4, 16, 7, 13],
    [13, 25, 16, 22],
    [10, 22, 13, 19],
    [12, 16, 13, 15],
    [13, 17, 14, 16],
    [10, 14, 11, 13],
];

/// `InitializeIndices3D`'s `m_C2Indices` (`itkFastMarchingImageFilterBase.hxx:934-964`),
/// generated by the same recurrence rather than transcribed.
const fn c2_indices() -> [[usize; 8]; 8] {
    let mut c = [[0usize; 8]; 8];
    c[0] = [0, 13, 1, 12, 3, 10, 4, 9];
    c[4] = [9, 22, 10, 21, 12, 19, 13, 18];

    let mut i = 1;
    while i < 4 {
        let addend = if i == 2 { 2 } else { 1 };
        let mut j = 0;
        while j < 8 {
            c[i][j] = c[i - 1][j] + addend;
            c[i + 4][j] = c[i + 3][j] + addend;
            j += 1;
        }
        i += 1;
    }
    c
}

const C2_INDICES: [[usize; 8]; 8] = c2_indices();

/// `IsCriticalC1Configuration2D` (`itkFastMarchingImageFilterBase.hxx:669-672`).
fn is_critical_c1_2d(n: &[bool; 9]) -> bool {
    !n[0] && n[1] && n[3] && !n[4] && !n[8]
}

/// `IsCriticalC2Configuration2D` (`itkFastMarchingImageFilterBase.hxx:674-680`).
fn is_critical_c2_2d(n: &[bool; 9]) -> bool {
    !n[0] && n[1] && n[3] && !n[4] && n[8] && (n[5] || n[7])
}

/// `IsCriticalC3Configuration2D` (`itkFastMarchingImageFilterBase.hxx:682-688`).
fn is_critical_c3_2d(n: &[bool; 9]) -> bool {
    !n[0] && n[1] && n[3] && !n[4] && !n[5] && n[6] && !n[7] && n[8]
}

/// `IsCriticalC4Configuration2D` (`itkFastMarchingImageFilterBase.hxx:690-696`).
fn is_critical_c4_2d(n: &[bool; 9]) -> bool {
    !n[0] && n[1] && n[3] && !n[4] && !n[5] && !n[6] && !n[7] && n[8]
}

/// `IsCriticalC1Configuration3D` (`itkFastMarchingImageFilterBase.hxx:820-826`).
fn is_critical_c1_3d(n: &[bool; 4]) -> bool {
    (n[0] && n[1] && !n[2] && !n[3]) || (!n[0] && !n[1] && n[2] && n[3])
}

/// `IsCriticalC2Configuration3D` (`itkFastMarchingImageFilterBase.hxx:828-859`).
/// Upstream returns `1` or `2` to name the type; the only caller tests `!= 0`.
fn is_critical_c2_3d(n: &[bool; 8]) -> bool {
    (0..4).any(|i| {
        n[2 * i] == n[2 * i + 1]
            && !(0..8).any(|j| n[j] == n[2 * i] && j != 2 * i && j != 2 * i + 1)
    })
}

/// `FastMarchingImageFilterBase`: solve the Eikonal equation
/// `|grad T| * speed = 1` outward from `trial_points`, returning the
/// arrival-time field `T`.
///
/// - `speed` is the speed image; its geometry (spacing included — the update is
///   anisotropic through `1/h_axis^2`) carries to the output.
/// - `trial_points` are image *indices*: at least `dim` elements, optionally a
///   `dim + 1`-th that the yaml's cast reads as the seed's initial arrival time
///   (`if (m_TrialPoints[i].size() > NodeType::Dimension) node.SetValue(...)`).
///   Points outside the image are silently dropped.
/// - `initial_trial_values` overrides the seed values positionally, against the
///   **full** `trial_points` list, dropped points included; missing entries
///   leave the value the point itself carried.
///
/// The output pixel type is always [`PixelId::Float32`]; unreached pixels hold
/// [`LARGE_VALUE`].
pub fn fast_marching_base(
    speed: &Image,
    trial_points: &[Vec<u32>],
    initial_trial_values: &[f64],
    settings: &FastMarchingBaseSettings,
) -> Result<Image> {
    let size = speed.size();
    let dim = size.len();

    check_point_lengths(trial_points, dim)?;
    check_normalization_factor(settings.normalization_factor)?;

    let strides = strides(size);
    let n: usize = size.iter().product();

    let mut marcher = Marcher {
        size: size.to_vec(),
        strides,
        spacing: speed.spacing().to_vec(),
        speed: speed.to_f64_vec()?,
        normalization_factor: settings.normalization_factor,
        threshold: narrow(settings.stopping_value),
        topology_check: settings.topology_check,
        output: vec![LARGE_VALUE; n],
        labels: vec![Label::Far; n],
        heap: BinaryHeap::new(),
    };

    for (i, point) in trial_points.iter().enumerate() {
        let Some(index) = flat_index(point, dim, size, &marcher.strides) else {
            continue;
        };
        let seed_value = point.get(dim).map_or(0.0, |&v| f64::from(v));
        let value = narrow(initial_trial_values.get(i).copied().unwrap_or(seed_value));
        marcher.labels[index] = Label::InitialTrial;
        marcher.output[index] = value;
        marcher.heap.push(TrialNode { value, index });
    }

    marcher.march()?;

    image_from_f64(PixelId::Float32, size, speed, &marcher.output)
}

/// `static_cast<OutputPixelType>(v)` for the yaml's `float` output. Values are
/// stored already narrowed so the heap's exact-equality staleness test compares
/// the same quantity `Math::ExactlyEquals` does.
fn narrow(v: f64) -> f64 {
    v as f32 as f64
}

struct Marcher {
    size: Vec<usize>,
    strides: Vec<usize>,
    spacing: Vec<f64>,
    speed: Vec<f64>,
    normalization_factor: f64,
    /// `m_Threshold`, already through `static_cast<float>`.
    threshold: f64,
    topology_check: TopologyCheck,
    output: Vec<f64>,
    labels: Vec<Label>,
    heap: BinaryHeap<TrialNode>,
}

impl Marcher {
    fn dim(&self) -> usize {
        self.size.len()
    }

    fn coords_of(&self, index: usize) -> Vec<usize> {
        (0..self.dim())
            .map(|d| (index / self.strides[d]) % self.size[d])
            .collect()
    }

    fn flat(&self, coords: &[usize]) -> usize {
        coords.iter().zip(&self.strides).map(|(&c, &s)| c * s).sum()
    }

    /// `FastMarchingBase::GenerateData()`'s heap loop
    /// (`itkFastMarchingBase.hxx:128-171`).
    fn march(&mut self) -> Result<()> {
        while let Some(node) = self.heap.pop() {
            let current_value = self.output[node.index];

            // `Math::ExactlyEquals`: a defunct entry left behind by a later
            // `UpdateValue` (or by a topology rejection) is discarded.
            if current_value != node.value {
                continue;
            }
            if self.labels[node.index] == Label::Alive {
                continue;
            }
            // `FastMarchingThresholdStoppingCriterion::IsSatisfied`.
            if current_value >= self.threshold {
                break;
            }
            if self.check_topology(node.index) {
                self.labels[node.index] = Label::Alive;
                self.update_neighbors(node.index)?;
            }
        }
        Ok(())
    }

    /// `UpdateNeighbors` (`itkFastMarchingImageFilterBase.hxx:143-169`),
    /// including the border defect the module doc describes: on either end of
    /// axis `j` the neighbor index is left at the center, whose label is the
    /// `Alive` just written, so nothing along `j` is updated.
    fn update_neighbors(&mut self, index: usize) -> Result<()> {
        let coords = self.coords_of(index);
        for j in 0..self.dim() {
            let v = coords[j];
            let last = self.size[j] - 1;
            let mut neighbor = coords.clone();

            for s in [-1i64, 1] {
                if v > 0 && v < last {
                    neighbor[j] = (v as i64 + s) as usize;
                }
                let ni = self.flat(&neighbor);
                if !matches!(self.labels[ni], Label::Alive | Label::InitialTrial) {
                    self.update_value(ni)?;
                }
            }
        }
        Ok(())
    }

    /// `UpdateValue` (`itkFastMarchingImageFilterBase.hxx:173-190`) over the
    /// per-axis minima `GetInternalNodesUsed` collects (`.hxx:194-242`).
    fn update_value(&mut self, index: usize) -> Result<()> {
        let coords = self.coords_of(index);
        let mut nodes: Vec<(f64, usize)> = Vec::with_capacity(self.dim());

        for j in 0..self.dim() {
            let mut best = LARGE_VALUE;
            let mut neighbor = coords.clone();
            for s in [-1i64, 1] {
                let t = coords[j] as i64 + s;
                if t >= 0 && t <= (self.size[j] - 1) as i64 {
                    neighbor[j] = t as usize;
                    let ni = self.flat(&neighbor);
                    if self.labels[ni] == Label::Alive {
                        let value = self.output[ni];
                        if best > value {
                            best = value;
                        }
                    }
                }
            }
            nodes.push((best, j));
        }

        let solution = narrow(self.solve(index, &mut nodes)?);
        if solution < LARGE_VALUE {
            self.output[index] = solution;
            self.labels[index] = Label::Trial;
            self.heap.push(TrialNode {
                value: solution,
                index,
            });
        }
        Ok(())
    }

    /// `Solve` (`itkFastMarchingImageFilterBase.hxx:246-302`). `m_InputCache` is
    /// always the speed image here, so the `m_InverseSpeed` branch is dead.
    fn solve(&self, index: usize, nodes: &mut [(f64, usize)]) -> Result<f64> {
        nodes.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut solution = f64::MAX;
        let mut aa = 0.0;
        let mut bb = 0.0;

        let speed = self.speed[index] / self.normalization_factor;
        let mut cc = if speed.abs() <= FLOAT_ALMOST_EQUAL_ZERO {
            -sqr(1.0 / (speed + ITK_MATH_EPS))
        } else {
            -sqr(1.0 / speed)
        };

        for &(value, axis) in nodes.iter() {
            if solution >= value {
                let space_factor = sqr(1.0 / self.spacing[axis]);
                aa += space_factor;
                bb += value * space_factor;
                cc += sqr(value) * space_factor;

                let discriminant = sqr(bb) - aa * cc;
                if discriminant < ITK_MATH_EPS {
                    return Err(FilterError::NegativeDiscriminant);
                }
                solution = (discriminant.sqrt() + bb) / aa;
            } else {
                break;
            }
        }
        Ok(solution)
    }

    /// The label at `coords + offset`, clamped into the image the way the
    /// neighborhood iterator's `ZeroFluxNeumannBoundaryCondition` clamps it.
    fn label_at(&self, coords: &[usize], offset: &[i64]) -> Label {
        let index: usize = (0..self.dim())
            .map(|d| {
                let c = (coords[d] as i64 + offset[d]).clamp(0, self.size[d] as i64 - 1);
                c as usize * self.strides[d]
            })
            .sum();
        self.labels[index]
    }

    /// `It.GetPixel(n)` over the label image's 3×3(×3) neighborhood.
    fn is_alive_at(&self, coords: &[usize], n: usize) -> bool {
        let offset = neighborhood_offset(n, self.dim());
        self.label_at(coords, &offset[..self.dim()]) == Label::Alive
    }

    /// `It.GetNext(axis)` / `It.GetPrevious(axis)`.
    fn is_alive_along(&self, coords: &[usize], axis: usize, step: i64) -> bool {
        let mut offset = [0i64; 3];
        offset[axis] = step;
        self.label_at(coords, &offset[..self.dim()]) == Label::Alive
    }

    /// `CheckTopology` (`itkFastMarchingImageFilterBase.hxx:306-391`), with
    /// `NoHandles` folded onto `Strict`; see the module doc.
    fn check_topology(&mut self, index: usize) -> bool {
        if self.topology_check == TopologyCheck::Nothing {
            return true;
        }
        // `itkWarningMacro("CheckTopology has not be implemented for
        // Dimension != 2 and != 3.")` — and then accepts.
        if self.dim() != 2 && self.dim() != 3 {
            return true;
        }

        let coords = self.coords_of(index);
        let well_composedness_violation = !self.is_change_well_composed(&coords);
        let strict_topology_violation = self.violates_strict_topology(&coords);

        if well_composedness_violation || strict_topology_violation {
            self.output[index] = LARGE_VALUE; // `m_TopologyValue`
            self.labels[index] = Label::Topology;
            return false;
        }
        true
    }

    /// `DoesVoxelChangeViolateStrictTopology`
    /// (`itkFastMarchingImageFilterBase.hxx:579-611`).
    fn violates_strict_topology(&self, coords: &[usize]) -> bool {
        let mut critical_c3_configurations = 0u32;
        let mut faces = 0u32;

        for d in 0..self.dim() {
            let next = self.is_alive_along(coords, d, 1);
            let previous = self.is_alive_along(coords, d, -1);
            if next {
                faces += 1;
            }
            if previous {
                faces += 1;
            }
            if next && previous {
                critical_c3_configurations += 1;
            }
        }

        critical_c3_configurations > 0 && faces % 2 == 0 && critical_c3_configurations * 2 == faces
    }

    /// `DoesVoxelChangeViolateWellComposedness`'s inner call
    /// (`itkFastMarchingImageFilterBase.hxx:562-575`).
    fn is_change_well_composed(&self, coords: &[usize]) -> bool {
        if self.dim() == 2 {
            self.is_change_well_composed_2d(coords)
        } else {
            self.is_change_well_composed_3d(coords)
        }
    }

    /// `IsChangeWellComposed2D` (`itkFastMarchingImageFilterBase.hxx:615-665`).
    /// Bits are "not alive", with the center bit flipped so the node reads as
    /// the alive it is about to become.
    fn is_change_well_composed_2d(&self, coords: &[usize]) -> bool {
        let bits = |permutation: &[usize; 9]| {
            let mut neighborhood = [false; 9];
            for (j, slot) in neighborhood.iter_mut().enumerate() {
                *slot = !self.is_alive_at(coords, permutation[j]);
                if permutation[j] == 4 {
                    *slot = !*slot;
                }
            }
            neighborhood
        };

        for permutation in &ROTATION_INDICES {
            let n = bits(permutation);
            if is_critical_c1_2d(&n)
                || is_critical_c2_2d(&n)
                || is_critical_c3_2d(&n)
                || is_critical_c4_2d(&n)
            {
                return false;
            }
        }

        // The C1 and C2 reflections are already covered by the rotations.
        for permutation in &REFLECTION_INDICES {
            let n = bits(permutation);
            if is_critical_c3_2d(&n) || is_critical_c4_2d(&n) {
                return false;
            }
        }
        true
    }

    /// `IsChangeWellComposed3D` (`itkFastMarchingImageFilterBase.hxx:773-818`).
    /// Bits are "alive" here — the opposite of the 2-D convention — with the
    /// center bit flipped.
    fn is_change_well_composed_3d(&self, coords: &[usize]) -> bool {
        for c1 in &C1_INDICES {
            let mut n = [false; 4];
            for (j, slot) in n.iter_mut().enumerate() {
                *slot = self.is_alive_at(coords, c1[j]);
                if c1[j] == 13 {
                    *slot = !*slot;
                }
            }
            if is_critical_c1_3d(&n) {
                return false;
            }
        }

        for c2 in &C2_INDICES {
            let mut n = [false; 8];
            for (j, slot) in n.iter_mut().enumerate() {
                *slot = self.is_alive_at(coords, c2[j]);
                if c2[j] == 13 {
                    *slot = !*slot;
                }
            }
            if is_critical_c2_3d(&n) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corner value of a unit-speed, unit-spacing march seeded at the
    /// center of a 3×3: `(sqrt(2) + 2) / 2`.
    const CORNER: f64 = 1.707_106_781_186_547_5;

    fn speed(size: &[usize], fill: f64) -> Image {
        Image::from_vec(size, vec![fill; size.iter().product()]).unwrap()
    }

    fn march(image: &Image, points: &[Vec<u32>], settings: &FastMarchingBaseSettings) -> Vec<f64> {
        fast_marching_base(image, points, &[], settings)
            .unwrap()
            .to_f64_vec()
            .unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-6, "pixel {i}: {a} != {e}");
        }
    }

    #[test]
    fn defaults_match_the_yaml() {
        let settings = FastMarchingBaseSettings::default();
        assert_eq!(settings.normalization_factor, 1.0);
        assert_eq!(settings.stopping_value, f64::from(f32::MAX) / 2.0);
        assert_eq!(settings.topology_check, TopologyCheck::Nothing);
        assert_eq!(LARGE_VALUE, f64::from(f32::MAX));
    }

    #[test]
    fn an_interior_seed_marches_outward_in_1d() {
        let out = march(
            &speed(&[5], 1.0),
            &[vec![2]],
            &FastMarchingBaseSettings::default(),
        );
        assert_close(&out, &[2.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn an_interior_seed_marches_outward_in_2d() {
        let out = march(
            &speed(&[3, 3], 1.0),
            &[vec![1, 1]],
            &FastMarchingBaseSettings::default(),
        );
        assert_close(
            &out,
            &[
                CORNER, 1.0, CORNER, //
                1.0, 0.0, 1.0, //
                CORNER, 1.0, CORNER,
            ],
        );
    }

    /// `UpdateNeighbors`'s `(v > start) && (v < last)` guard: a seed on the
    /// `x == 0` face never updates anything at `x == 1`, so the march is
    /// trapped in that column.
    #[test]
    fn a_border_seed_never_propagates_along_the_border_normal_axis() {
        let out = march(
            &speed(&[5, 5], 1.0),
            &[vec![0, 2]],
            &FastMarchingBaseSettings::default(),
        );

        let column: Vec<f64> = (0..5).map(|y| out[y * 5]).collect();
        assert_close(&column, &[2.0, 1.0, 0.0, 1.0, 2.0]);

        for y in 0..5 {
            for x in 1..5 {
                assert_eq!(out[y * 5 + x], LARGE_VALUE, "pixel ({x}, {y}) was reached");
            }
        }
    }

    /// The 1-D face of the same defect: a corner seed marches nowhere.
    #[test]
    fn a_corner_seed_marches_nowhere() {
        let out = march(
            &speed(&[5], 1.0),
            &[vec![0]],
            &FastMarchingBaseSettings::default(),
        );
        assert_close(
            &out,
            &[0.0, LARGE_VALUE, LARGE_VALUE, LARGE_VALUE, LARGE_VALUE],
        );
    }

    /// `IsSatisfied` is `>=`, and it fires before the node is accepted, so the
    /// value-1 ring keeps the trial values `UpdateValue` wrote but never goes
    /// alive.
    #[test]
    fn the_stopping_value_is_inclusive_and_stops_before_accepting() {
        let settings = FastMarchingBaseSettings {
            stopping_value: 1.0,
            ..Default::default()
        };
        let out = march(&speed(&[5], 1.0), &[vec![2]], &settings);
        assert_close(&out, &[LARGE_VALUE, 1.0, 0.0, 1.0, LARGE_VALUE]);
    }

    /// `if (m_TrialPoints[i].size() > NodeType::Dimension) node.SetValue(...)`.
    #[test]
    fn a_trailing_element_is_the_seed_value() {
        let out = march(
            &speed(&[5], 1.0),
            &[vec![2, 3]],
            &FastMarchingBaseSettings::default(),
        );
        assert_close(&out, &[5.0, 4.0, 3.0, 4.0, 5.0]);
    }

    /// `InitialTrialValues`'s cast runs last and overrides positionally.
    #[test]
    fn initial_trial_values_override_the_trailing_element() {
        let out = fast_marching_base(
            &speed(&[5], 1.0),
            &[vec![2, 3]],
            &[10.0],
            &FastMarchingBaseSettings::default(),
        )
        .unwrap()
        .to_f64_vec()
        .unwrap();
        assert_close(&out, &[12.0, 11.0, 10.0, 11.0, 12.0]);
    }

    /// `InitializeOutput` gates each seed on `m_BufferedRegion.IsInside(idx)`,
    /// but the yaml's `InitialTrialValues` cast indexes the *unfiltered*
    /// container.
    #[test]
    fn out_of_bounds_trial_points_are_dropped_but_keep_their_value_slot() {
        let out = fast_marching_base(
            &speed(&[5], 1.0),
            &[vec![9], vec![2]],
            &[100.0, 3.0],
            &FastMarchingBaseSettings::default(),
        )
        .unwrap()
        .to_f64_vec()
        .unwrap();
        assert_close(&out, &[5.0, 4.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn an_empty_trial_list_leaves_the_output_untouched() {
        let out = march(&speed(&[3], 1.0), &[], &FastMarchingBaseSettings::default());
        assert_close(&out, &[LARGE_VALUE; 3]);
    }

    #[test]
    fn a_trial_point_shorter_than_the_dimension_is_an_error() {
        let err = fast_marching_base(
            &speed(&[3, 3], 1.0),
            &[vec![1]],
            &[],
            &FastMarchingBaseSettings::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    /// The `TrialPoints` cast is the first member setter SimpleITK emits, so it
    /// outranks `Initialize`'s normalization-factor check.
    #[test]
    fn the_trial_point_length_check_outranks_the_normalization_factor_check() {
        let settings = FastMarchingBaseSettings {
            normalization_factor: 0.0,
            ..Default::default()
        };
        let err = fast_marching_base(&speed(&[3, 3], 1.0), &[vec![1]], &[], &settings).unwrap_err();
        assert_eq!(
            err,
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn a_normalization_factor_below_eps_is_an_error() {
        let settings = FastMarchingBaseSettings {
            normalization_factor: 0.0,
            ..Default::default()
        };
        let err = fast_marching_base(&speed(&[5], 1.0), &[vec![2]], &[], &settings).unwrap_err();
        assert_eq!(err, FilterError::InvalidNormalizationFactor(0.0));
    }

    #[test]
    fn the_normalization_factor_divides_the_speed() {
        let settings = FastMarchingBaseSettings {
            normalization_factor: 2.0,
            ..Default::default()
        };
        let out = march(&speed(&[5], 2.0), &[vec![2]], &settings);
        assert_close(&out, &[2.0, 1.0, 0.0, 1.0, 2.0]);
    }

    /// `Solve`'s `FloatAlmostEqual(cc, 0.0)` guard turns `-sqr(1/0)` into
    /// `-sqr(1/DBL_EPSILON) == -2^104`, so the arrival time is `sqrt(2^104)`,
    /// exactly `2^52` — finite, below `f32::MAX`, and therefore written.
    /// [`crate::fast_marching`]'s older `Solve` leaves it `+inf` and the pixel
    /// stays unreached.
    #[test]
    fn zero_speed_no_longer_blocks_the_front() {
        let image = Image::from_vec(&[3], vec![1.0f64, 1.0, 0.0]).unwrap();

        let out = march(&image, &[vec![1]], &FastMarchingBaseSettings::default());
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 4_503_599_627_370_496.0); // 2^52
    }

    /// Topology checking is a no-op for `ImageDimension` outside `{2, 3}`:
    /// `(2)` merges the two fronts even under `Strict`.
    #[test]
    fn topology_check_is_inert_outside_2d_and_3d() {
        let settings = FastMarchingBaseSettings {
            topology_check: TopologyCheck::Strict,
            ..Default::default()
        };
        let out = march(&speed(&[5], 1.0), &[vec![1], vec![3]], &settings);
        assert_eq!(out[2], 1.0);
    }

    /// Two fronts meeting head-on at `(2, 1)`: `DoesVoxelChangeViolateStrictTopology`
    /// sees `faces == 2`, `criticalC3 == 1`. Well-composedness is *not*
    /// violated there, so this fixture drives `Strict`'s second predicate and
    /// `NoHandles`'s handle branch. `(2, 1)` is never re-updated afterwards,
    /// because its only remaining non-alive neighbors sit on the `y` border and
    /// so cannot propagate along `y`.
    fn merging_fronts(topology_check: TopologyCheck) -> Vec<f64> {
        march(
            &speed(&[5, 3], 1.0),
            &[vec![1, 1], vec![3, 1]],
            &FastMarchingBaseSettings {
                topology_check,
                ..Default::default()
            },
        )
    }

    #[test]
    fn nothing_lets_two_fronts_merge() {
        let out = merging_fronts(TopologyCheck::Nothing);
        assert_eq!(out[5 + 2], 1.0);
    }

    #[test]
    fn strict_rejects_a_merge() {
        let out = merging_fronts(TopologyCheck::Strict);
        assert_eq!(out[5 + 2], LARGE_VALUE);
        // The two seeds still went alive.
        assert_eq!(out[5 + 1], 0.0);
        assert_eq!(out[5 + 3], 0.0);
    }

    #[test]
    fn no_handles_matches_strict_on_a_merge() {
        assert_eq!(
            merging_fronts(TopologyCheck::NoHandles),
            merging_fronts(TopologyCheck::Strict)
        );
    }

    /// `(2, 2)` touches the alive `(1, 1)` and `(3, 3)` only through their
    /// shared corners, while the faces between them are still trial — the C1
    /// critical configuration of `IsChangeWellComposed2D`. No axis has both
    /// face neighbors alive, so `DoesVoxelChangeViolateStrictTopology` is
    /// false and only well-composedness rejects.
    ///
    /// `stopping_value = 0.6` ends the march right after `(2, 2)` is judged, so
    /// a later `UpdateValue` cannot overwrite the rejection.
    fn diagonal_contact_2d(topology_check: TopologyCheck) -> Vec<f64> {
        fast_marching_base(
            &speed(&[5, 5], 1.0),
            &[vec![1, 1], vec![3, 3], vec![2, 2]],
            &[0.0, 0.0, 0.5],
            &FastMarchingBaseSettings {
                stopping_value: 0.6,
                topology_check,
                ..Default::default()
            },
        )
        .unwrap()
        .to_f64_vec()
        .unwrap()
    }

    #[test]
    fn nothing_accepts_a_diagonal_contact() {
        let out = diagonal_contact_2d(TopologyCheck::Nothing);
        assert_eq!(out[2 * 5 + 2], 0.5);
    }

    #[test]
    fn well_composedness_rejects_a_diagonal_contact() {
        let out = diagonal_contact_2d(TopologyCheck::Strict);
        assert_eq!(out[2 * 5 + 2], LARGE_VALUE);
        assert_eq!(out[5 + 1], 0.0);
        assert_eq!(out[3 * 5 + 3], 0.0);
    }

    #[test]
    fn no_handles_matches_strict_on_a_diagonal_contact() {
        assert_eq!(
            diagonal_contact_2d(TopologyCheck::NoHandles),
            diagonal_contact_2d(TopologyCheck::Strict)
        );
    }

    /// The 3-D C1 quadruple `[1, 13, 4, 10]` reads the stencil offsets
    /// `(0,-1,-1)`, center, `(0,0,-1)`, `(0,-1,0)`. With `(2,1,1)` alive and
    /// the two faces it shares with `(2,2,2)` merely trial, the first arm of
    /// `IsCriticalC1Configuration3D` fires and `(2,2,2)` is rejected. No axis
    /// has both face neighbors alive, so the strict predicate is false.
    fn diagonal_contact_3d(topology_check: TopologyCheck) -> Vec<f64> {
        fast_marching_base(
            &speed(&[5, 5, 5], 1.0),
            &[vec![2, 1, 1], vec![2, 2, 2]],
            &[0.0, 0.5],
            &FastMarchingBaseSettings {
                stopping_value: 0.6,
                topology_check,
                ..Default::default()
            },
        )
        .unwrap()
        .to_f64_vec()
        .unwrap()
    }

    #[test]
    fn nothing_accepts_a_diagonal_contact_in_3d() {
        let out = diagonal_contact_3d(TopologyCheck::Nothing);
        assert_eq!(out[2 + 2 * 5 + 2 * 25], 0.5);
    }

    #[test]
    fn well_composedness_rejects_a_diagonal_contact_in_3d() {
        let out = diagonal_contact_3d(TopologyCheck::Strict);
        assert_eq!(out[2 + 2 * 5 + 2 * 25], LARGE_VALUE);
        assert_eq!(out[2 + 5 + 25], 0.0);
    }

    #[test]
    fn no_handles_matches_strict_in_3d() {
        assert_eq!(
            diagonal_contact_3d(TopologyCheck::NoHandles),
            diagonal_contact_3d(TopologyCheck::Strict)
        );
    }

    /// A marcher whose only meaningful state is the label image, for testing
    /// `CheckTopology`'s two predicates in isolation.
    fn labelled(size: &[usize], alive: &[usize]) -> Marcher {
        let n: usize = size.iter().product();
        let mut labels = vec![Label::Far; n];
        for &index in alive {
            labels[index] = Label::Alive;
        }
        Marcher {
            size: size.to_vec(),
            strides: strides(size),
            spacing: vec![1.0; size.len()],
            speed: vec![1.0; n],
            normalization_factor: 1.0,
            threshold: LARGE_VALUE,
            topology_check: TopologyCheck::Strict,
            output: vec![LARGE_VALUE; n],
            labels,
            heap: BinaryHeap::new(),
        }
    }

    /// Each fixture above must reject through exactly one of `CheckTopology`'s
    /// two predicates; otherwise a predicate wired to a constant would still
    /// pass every test in this module.
    #[test]
    fn the_merge_fixture_violates_strict_topology_only() {
        // `(1, 1)` and `(3, 1)` alive in a 5x3; judging `(2, 1)`.
        let marcher = labelled(&[5, 3], &[5 + 1, 5 + 3]);
        let node = [2usize, 1];
        assert!(marcher.violates_strict_topology(&node));
        assert!(marcher.is_change_well_composed(&node));
    }

    #[test]
    fn the_diagonal_fixture_violates_well_composedness_only() {
        // `(1, 1)` and `(3, 3)` alive in a 5x5; judging `(2, 2)`.
        let marcher = labelled(&[5, 5], &[5 + 1, 3 * 5 + 3]);
        let node = [2usize, 2];
        assert!(!marcher.violates_strict_topology(&node));
        assert!(!marcher.is_change_well_composed(&node));
    }

    #[test]
    fn the_3d_diagonal_fixture_violates_well_composedness_only() {
        // `(2, 1, 1)` alive in a 5x5x5; judging `(2, 2, 2)`.
        let marcher = labelled(&[5, 5, 5], &[2 + 5 + 25]);
        let node = [2usize, 2, 2];
        assert!(!marcher.violates_strict_topology(&node));
        assert!(!marcher.is_change_well_composed(&node));
    }

    /// A node with no alive neighbor is always accepted — every critical
    /// configuration needs at least one alive stencil pixel.
    #[test]
    fn an_isolated_node_passes_both_predicates() {
        for size in [vec![5, 5], vec![5, 5, 5]] {
            let marcher = labelled(&size, &[]);
            let node = vec![2usize; size.len()];
            assert!(!marcher.violates_strict_topology(&node));
            assert!(marcher.is_change_well_composed(&node));
        }
    }

    /// `InitializeIndices3D`'s recurrence, pinned against the values it
    /// generates for `m_C2Indices`.
    #[test]
    fn the_3d_c2_index_recurrence_matches_upstream() {
        assert_eq!(C2_INDICES[0], [0, 13, 1, 12, 3, 10, 4, 9]);
        assert_eq!(C2_INDICES[1], [1, 14, 2, 13, 4, 11, 5, 10]);
        assert_eq!(C2_INDICES[2], [3, 16, 4, 15, 6, 13, 7, 12]);
        assert_eq!(C2_INDICES[3], [4, 17, 5, 16, 7, 14, 8, 13]);
        assert_eq!(C2_INDICES[4], [9, 22, 10, 21, 12, 19, 13, 18]);
        assert_eq!(C2_INDICES[5], [10, 23, 11, 22, 13, 20, 14, 19]);
        assert_eq!(C2_INDICES[6], [12, 25, 13, 24, 15, 22, 16, 21]);
        assert_eq!(C2_INDICES[7], [13, 26, 14, 25, 16, 23, 17, 22]);
    }

    #[test]
    fn the_output_is_always_float32() {
        let image = Image::from_vec(&[3], vec![1u8, 1, 1]).unwrap();
        let out = fast_marching_base(
            &image,
            &[vec![1]],
            &[],
            &FastMarchingBaseSettings::default(),
        )
        .unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }
}
