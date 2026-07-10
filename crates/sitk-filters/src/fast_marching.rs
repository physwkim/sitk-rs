//! Fast marching: an Eikonal solver on a Cartesian grid.
//!
//! Port of `itk::FastMarchingImageFilter`
//! (`itkFastMarchingImageFilter.h` / `.hxx`, `itkLevelSetNode.h`), with the
//! API surface SimpleITK's `FastMarchingImageFilter.yaml` declares:
//! trial points, per-point initial values, `normalization_factor` and
//! `stopping_value`. The input is the *speed* image; the output is the
//! arrival-time field.
//!
//! ## The scheme
//!
//! Every pixel carries a label — far / trial / initial-trial / alive
//! (`FastMarchingImageFilterEnums::Label`) — and a value, initialized to
//! [`large_value`] (`NumericTraits<PixelType>::max() / 2`). The trial points
//! seed a min-heap. Each round pops the smallest trial value, freezes it as
//! *alive*, and recomputes every non-alive face neighbor with the upwind
//! quadratic `UpdateValue`:
//!
//! For each axis `j`, take the smallest value among that axis's *alive*
//! neighbors (or `large_value` when it has none), sort those `dim`
//! candidates ascending, and accumulate them one at a time while the running
//! solution still dominates the candidate:
//!
//! ```text
//! aa += 1/h_j^2                  bb += v_j/h_j^2                cc += v_j^2/h_j^2
//! solution = (sqrt(bb^2 - aa*cc) + bb) / aa       // the larger root
//! ```
//!
//! `cc` starts at `-(normalization_factor / speed[index])^2`, i.e. ITK's
//! `cc = speed/F; cc = -sqr(1/cc)` — the speed is *divided* by
//! `normalization_factor`, which is what lets integer images carry a speed.
//! The `solution >= value` guard is what makes the stencil upwind: a
//! candidate larger than the current solution (and every candidate after it,
//! since they are sorted) is dropped. A solution that stays at
//! `large_value` is not written back, so the pixel stays far.
//!
//! Consequences of the `.hxx` worth spelling out:
//!
//! - **Zero speed blocks the front.** `cc` becomes `-inf`, the discriminant
//!   `+inf`, and the solution `+inf`, which fails `solution < large_value`.
//!   The pixel is never written, never enters the heap, never goes alive —
//!   so it also never propagates. Pixels behind a zero-speed barrier keep
//!   [`large_value`].
//! - **Negative speed acts as its magnitude.** `-(F/speed)^2` squares the
//!   sign away. ITK documents the speed as non-negative but does not check.
//! - **`stopping_value` truncates, it does not clear.** The loop breaks when
//!   the popped value exceeds it, leaving already-computed *trial* values in
//!   the output. Only pixels still labelled far hold [`large_value`].
//! - **Out-of-bounds trial points are silently dropped** — `Initialize()`
//!   gates every seed on `m_BufferedRegion.IsInside(idx)` and no error is
//!   raised. An empty (or entirely dropped) seed list yields an output that
//!   is [`large_value`] everywhere.
//!
//! ## Heap staleness
//!
//! `UpdateValue` cannot decrease a node already in the heap, so ITK pushes a
//! *new* node and leaves the defunct one behind (the class doc calls this
//! out). A popped node is therefore accepted only if its value still equals
//! the value stored in the output image (`Math::ExactlyEquals`) *and* the
//! pixel is not already alive; otherwise it is discarded. This port keeps
//! that exactly — including storing values already narrowed to the output
//! pixel type, so the exact-equality test compares like for like — and uses
//! [`std::collections::BinaryHeap`] with a reversed [`Ord`] rather than
//! `Reverse<_>`, because the node needs a hand-written `Ord` anyway (`f64`
//! is not `Ord`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use sitk_core::{Image, PixelId};

use crate::error::{FilterError, Result};
use crate::{image_from_f64, real_pixel_id};

/// `itk::Math::eps` (`itkMath.h`), the bound `GenerateData` rejects
/// `m_NormalizationFactor` below.
const ITK_MATH_EPS: f64 = f64::EPSILON;

/// First-index-fastest strides for a size vector.
pub(crate) fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `FastMarchingImageFilterEnums::Label`, minus `OutsidePoint`: SimpleITK's
/// `FastMarchingImageFilter` exposes neither `SetOutsidePoints` /
/// `SetBinaryMask` nor `SetAlivePoints`, so those two seed containers are
/// always empty here and only `AlivePoint` arises at run time.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Label {
    Far,
    Alive,
    Trial,
    InitialTrial,
}

/// A node on the trial heap: `(value, flat index)`.
///
/// [`Ord`] is reversed so [`BinaryHeap`]'s max-heap pops the smallest value,
/// matching ITK's `std::priority_queue<AxisNodeType, _, std::greater<>>` over
/// `LevelSetNode::operator>` (which compares the value field alone). Equal
/// values are broken by flat index — ITK leaves that order unspecified, but
/// the arrival field does not depend on it: a node made alive at value `v`
/// only ever updates neighbors with a newly-alive value of exactly `v`, and
/// admitting an alive neighbor whose value equals a node's current solution
/// leaves that solution unchanged.
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

/// The arrival-time pixel type for a given speed pixel type. SimpleITK's
/// `output_pixel_type` is `NumericTraits<InputPixelType>::RealType`, which is
/// `double` for **every** scalar input (`NumericTraits<float>::RealType` is
/// `double`, itkNumericTraits.h:1349/1356); this port's [`real_pixel_id`]
/// currently keeps `Float32 → Float32` — a documented divergence tracked in
/// the upstream-findings ledger §5.6.
pub fn output_pixel_id(speed: PixelId) -> PixelId {
    real_pixel_id(speed)
}

/// The value unreached (far) pixels hold: ITK's `m_LargeValue`,
/// `static_cast<PixelType>(NumericTraits<PixelType>::max() / 2)` for the
/// arrival-time pixel type [`output_pixel_id`] selects.
///
/// It is also SimpleITK's `StoppingValue` default (`double::max() / 2`) when
/// the output type is `double`.
pub fn large_value(speed: PixelId) -> f64 {
    match output_pixel_id(speed) {
        PixelId::Float32 => (f32::MAX / 2.0) as f64,
        _ => f64::MAX / 2.0,
    }
}

/// `FastMarchingImageFilter`: solve the Eikonal equation `|grad T| * speed = 1`
/// outward from `trial_points`, returning the arrival-time field `T`.
///
/// - `speed` is the speed image; its geometry (spacing included — the update
///   is anisotropic through `1/h_axis^2`) carries to the output.
/// - `trial_points` are image *indices*, one `Vec` of length `dim` per point
///   (SimpleITK's `std::vector<std::vector<unsigned int>>`). Points outside
///   the image are silently dropped, as in `Initialize()`.
/// - `initial_trial_values` are the seeds' arrival times, matched to
///   `trial_points` by position (SimpleITK's `InitialTrialValues`); missing
///   entries default to `0.0`.
/// - `normalization_factor` divides the speed; must be `>= f64::EPSILON`.
/// - `stopping_value` ends the march once the smallest trial value exceeds
///   it. SimpleITK's default is `f64::MAX / 2.0`; see [`large_value`].
///
/// The output pixel type is [`output_pixel_id`]; unreached pixels hold
/// [`large_value`].
pub fn fast_marching(
    speed: &Image,
    trial_points: &[Vec<u32>],
    initial_trial_values: &[f64],
    normalization_factor: f64,
    stopping_value: f64,
) -> Result<Image> {
    // ITK checks this before it ever looks at the seeds, and a wrong-length
    // seed is not an error it can express (its indices are dimension-typed),
    // so the normalization factor takes precedence when both are invalid.
    check_normalization_factor(normalization_factor)?;

    let size = speed.size();
    let dim = size.len();
    for point in trial_points {
        if point.len() != dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: point.len(),
            });
        }
    }

    let out_id = output_pixel_id(speed.pixel_id());
    let strides = strides(size);

    // `Initialize()` drops seeds outside the buffered region; the survivors
    // keep their `initial_trial_values` entry, defaulting to `0.0`.
    let trial: Vec<(usize, f64)> = trial_points
        .iter()
        .enumerate()
        .filter(|(_, point)| point.iter().zip(size).all(|(&c, &e)| (c as usize) < e))
        .map(|(i, point)| {
            let index = point
                .iter()
                .zip(&strides)
                .map(|(&c, &s)| c as usize * s)
                .sum();
            (index, initial_trial_values.get(i).copied().unwrap_or(0.0))
        })
        .collect();

    let result = march_flat(
        MarchInput {
            size,
            spacing: speed.spacing(),
            speed: &speed.to_f64_vec()?,
            narrow_to_f32: out_id == PixelId::Float32,
            normalization_factor,
            stopping_value,
            collect_points: false,
            upwind: None,
        },
        &trial,
    )?;

    image_from_f64(out_id, size, speed, &result.values)
}

/// The march's inputs, as `FastMarchingImageFilter`'s members hold them, on a
/// bare buffer rather than an `Image`. `narrow_to_f32` stands for the level-set
/// `PixelType`: it selects both `m_LargeValue` and the `static_cast<PixelType>`
/// every written value passes through.
pub(crate) struct MarchInput<'a> {
    pub(crate) size: &'a [usize],
    pub(crate) spacing: &'a [f64],
    /// The speed image. ITK's no-speed-image branch (`cc = m_InverseSpeed`,
    /// from `m_SpeedConstant`) is not modelled: pass a constant buffer, which
    /// is arithmetically identical.
    pub(crate) speed: &'a [f64],
    pub(crate) narrow_to_f32: bool,
    pub(crate) normalization_factor: f64,
    pub(crate) stopping_value: f64,
    /// `m_CollectPoints`. When set, [`MarchResult::processed`] records
    /// `GetProcessedPoints()`.
    pub(crate) collect_points: bool,
    /// `FastMarchingUpwindGradientImageFilter`'s per-accepted-point work.
    /// `None` runs the plain `FastMarchingImageFilter`.
    pub(crate) upwind: Option<UpwindInput<'a>>,
}

/// `FastMarchingUpwindGradientImageFilterEnums::TargetCondition`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TargetCondition {
    NoTargets,
    OneTarget,
    SomeTargets,
    AllTargets,
}

/// The subclass state `FastMarchingUpwindGradientImageFilter::UpdateNeighbors`
/// reads. Its `VerifyPreconditions` must already have passed: whenever
/// `target_mode` is not [`TargetCondition::NoTargets`], `targets` is non-empty.
pub(crate) struct UpwindInput<'a> {
    /// `m_GenerateGradientImage`. When clear, `ComputeGradient` never runs and
    /// [`MarchResult::gradient`] stays empty.
    pub(crate) generate_gradient: bool,
    /// `m_TargetPoints`, in container order. `None` marks an entry outside the
    /// buffered region: ITK keeps such nodes in the container (so they count
    /// towards `m_TargetPoints->Size()`) but their index never equals an
    /// accepted one, so they are never reached.
    pub(crate) targets: &'a [Option<usize>],
    pub(crate) target_mode: TargetCondition,
    /// `m_NumberOfTargets`, read only by [`TargetCondition::SomeTargets`].
    pub(crate) number_of_targets: usize,
    /// `m_TargetOffset`.
    pub(crate) target_offset: f64,
}

pub(crate) struct MarchResult {
    /// The arrival-time field; unreached pixels hold `m_LargeValue`.
    pub(crate) values: Vec<f64>,
    /// `m_ProcessedPoints`, the flat indices in the order they went alive.
    /// Empty unless [`MarchInput::collect_points`] was set.
    pub(crate) processed: Vec<usize>,
    /// `m_GradientImage`, decomposed into one buffer per axis. Empty unless
    /// [`UpwindInput::generate_gradient`] was set; otherwise `dim` buffers of
    /// `size.product()` values each, zero at every never-accepted pixel
    /// (`FillBuffer(GradientPixelType{})`).
    pub(crate) gradient: Vec<Vec<f64>>,
    /// `m_TargetValue`, `0.0` when no upwind state was given.
    pub(crate) target_value: f64,
}

/// itkFastMarchingImageFilter.hxx `GenerateData()`: "Normalization Factor is
/// null or negative".
pub(crate) fn check_normalization_factor(normalization_factor: f64) -> Result<()> {
    if normalization_factor < ITK_MATH_EPS {
        return Err(FilterError::InvalidNormalizationFactor(
            normalization_factor,
        ));
    }
    Ok(())
}

/// `FastMarchingImageFilter::GenerateData()` over flat `(index, value)` trial
/// seeds. Seeds must already be inside the image.
pub(crate) fn march_flat(input: MarchInput<'_>, trial: &[(usize, f64)]) -> Result<MarchResult> {
    check_normalization_factor(input.normalization_factor)?;

    let large = if input.narrow_to_f32 {
        (f32::MAX / 2.0) as f64
    } else {
        f64::MAX / 2.0
    };
    let n: usize = input.size.iter().product();
    let dim = input.size.len();

    let generate_gradient = input.upwind.as_ref().is_some_and(|u| u.generate_gradient);
    let upwind = input.upwind.map(|u| UpwindState {
        generate_gradient: u.generate_gradient,
        targets: u.targets.to_vec(),
        mode: u.target_mode,
        number_of_targets: u.number_of_targets,
        target_offset: u.target_offset,
        reached: 0,
    });

    let mut solver = FastMarching {
        strides: strides(input.size),
        spacing: input.spacing.to_vec(),
        speed: input.speed.to_vec(),
        size: input.size.to_vec(),
        normalization_factor: input.normalization_factor,
        stopping_value: input.stopping_value,
        large,
        narrow_to_f32: input.narrow_to_f32,
        collect_points: input.collect_points,
        output: vec![large; n],
        labels: vec![Label::Far; n],
        heap: BinaryHeap::new(),
        processed: Vec::new(),
        upwind,
        // `Initialize()`: `m_GradientImage->FillBuffer(GradientPixelType{})`.
        gradient: if generate_gradient {
            vec![vec![0.0; n]; dim]
        } else {
            Vec::new()
        },
        // `Initialize()`: "Need to reset the target value."
        target_value: 0.0,
    };

    solver.seed(trial);
    solver.march()?;

    Ok(MarchResult {
        values: solver.output,
        processed: solver.processed,
        gradient: solver.gradient,
        target_value: solver.target_value,
    })
}

/// [`UpwindInput`] plus the running `m_ReachedTargetPoints->Size()`.
struct UpwindState {
    generate_gradient: bool,
    targets: Vec<Option<usize>>,
    mode: TargetCondition,
    number_of_targets: usize,
    target_offset: f64,
    /// `m_ReachedTargetPoints->Size()`. Only the *count* is tracked: nothing
    /// downstream of `UpdateNeighbors` reads the nodes themselves, and
    /// SimpleITK exposes no `GetReachedTargetPoints`.
    reached: usize,
}

struct FastMarching {
    size: Vec<usize>,
    strides: Vec<usize>,
    spacing: Vec<f64>,
    speed: Vec<f64>,
    normalization_factor: f64,
    /// `m_StoppingValue`. Mutable: a reached target lowers it mid-march.
    stopping_value: f64,
    large: f64,
    narrow_to_f32: bool,
    collect_points: bool,
    output: Vec<f64>,
    labels: Vec<Label>,
    heap: BinaryHeap<TrialNode>,
    processed: Vec<usize>,
    upwind: Option<UpwindState>,
    gradient: Vec<Vec<f64>>,
    target_value: f64,
}

impl FastMarching {
    /// `static_cast<PixelType>(v)` for the arrival-time pixel type. Values are
    /// stored already narrowed so the heap's exact-equality staleness test
    /// compares the same quantity ITK's does.
    fn narrow(&self, v: f64) -> f64 {
        if self.narrow_to_f32 {
            v as f32 as f64
        } else {
            v
        }
    }

    fn coords_of(&self, p: usize) -> Vec<usize> {
        (0..self.size.len())
            .map(|d| (p / self.strides[d]) % self.size[d])
            .collect()
    }

    fn flat(&self, coord: &[usize]) -> usize {
        coord.iter().zip(&self.strides).map(|(&c, &s)| c * s).sum()
    }

    /// `Initialize()`, trial-point half: each seed gets its value and the
    /// `InitialTrialPoint` label and goes on the heap.
    fn seed(&mut self, trial: &[(usize, f64)]) {
        for &(index, value) in trial {
            let value = self.narrow(value);
            self.labels[index] = Label::InitialTrial;
            self.output[index] = value;
            self.heap.push(TrialNode { value, index });
        }
    }

    /// `GenerateData()`'s heap loop.
    fn march(&mut self) -> Result<()> {
        while let Some(node) = self.heap.pop() {
            // Does this node still carry the pixel's current value? If not it
            // is a defunct entry left behind by a later `UpdateValue`.
            if node.value != self.output[node.index] {
                continue;
            }
            if self.labels[node.index] == Label::Alive {
                continue;
            }
            if node.value > self.stopping_value {
                break;
            }
            if self.collect_points {
                self.processed.push(node.index);
            }
            self.labels[node.index] = Label::Alive;
            self.update_neighbors(node.index)?;
            self.on_accepted(node.index);
        }
        Ok(())
    }

    /// The tail of `FastMarchingUpwindGradientImageFilter::UpdateNeighbors`,
    /// which runs after `Superclass::UpdateNeighbors` for the point just made
    /// alive: record its upwind gradient, then test the target condition.
    fn on_accepted(&mut self, index: usize) {
        let Some(mut upwind) = self.upwind.take() else {
            return;
        };
        if upwind.generate_gradient {
            for (axis, value) in self.compute_gradient(index).into_iter().enumerate() {
                self.gradient[axis][index] = value;
            }
        }
        self.check_targets(&mut upwind, index);
        self.upwind = Some(upwind);
    }

    /// `ComputeGradient()`: one-sided differences taken only against *alive*
    /// neighbors — "the front can only come from there" — then the upwind pick
    /// between them.
    ///
    /// A neighbor that is outside the image, or inside but not yet alive,
    /// contributes a difference of exactly `0`, and a pair of zero differences
    /// falls through to `gradientPixel[j] = dx_forward`, i.e. `0`. That is why
    /// the seed itself (accepted before any neighbor is alive) has a zero
    /// gradient.
    fn compute_gradient(&self, index: usize) -> Vec<f64> {
        let center = self.output[index];
        let coord = self.coords_of(index);

        coord
            .iter()
            .enumerate()
            .map(|(j, &c)| {
                let base = index - c * self.strides[j];
                let alive_value = |neighbor: usize| {
                    let ni = base + neighbor * self.strides[j];
                    (self.labels[ni] == Label::Alive).then(|| self.output[ni])
                };

                let dx_backward = match c.checked_sub(1).and_then(alive_value) {
                    Some(back) => self.narrow(center - back),
                    None => 0.0,
                };
                let dx_forward = match (c + 1 < self.size[j]).then(|| alive_value(c + 1)).flatten()
                {
                    Some(forward) => self.narrow(forward - center),
                    None => 0.0,
                };

                // `std::max<LevelSetPixelType>(dx_backward, -dx_forward)`,
                // spelled out so a `-0.0` forward difference cannot flip which
                // argument `f64::max` returns.
                let upwind = if dx_backward < -dx_forward {
                    -dx_forward
                } else {
                    dx_backward
                };
                let gradient = if upwind < 0.0 {
                    0.0
                } else if dx_backward > -dx_forward {
                    dx_backward
                } else {
                    dx_forward
                };
                self.narrow(gradient / self.spacing[j])
            })
            .collect()
    }

    /// The target half of `UpdateNeighbors`, for the point just made alive.
    ///
    /// Three upstream behaviors this reproduces rather than smooths over:
    ///
    /// - With no targets, `m_TargetValue` is overwritten at *every* accepted
    ///   point, so it ends as the last (largest) Eikonal value — which is what
    ///   `GetTargetValue`'s doc promises.
    /// - Under `SomeTargets`/`AllTargets` the count test sits outside the
    ///   target lookup, so once the count is met every subsequent accepted
    ///   point — target or not — re-triggers `targetReached` and moves
    ///   `m_TargetValue` forward. Only the *stopping value* is latched, since
    ///   `newStoppingValue` then only ever grows.
    /// - Under `OneTarget` a later accepted target likewise moves
    ///   `m_TargetValue` forward, so it is the last reached target's arrival
    ///   time, not the first's.
    fn check_targets(&mut self, upwind: &mut UpwindState, index: usize) {
        // `m_TargetReachedMode != NoTargets && m_TargetPoints`: SimpleITK
        // always installs a (possibly empty) target container, and a non-empty
        // one is what `VerifyPreconditions` demands of every other mode, so the
        // pointer test collapses into the mode test.
        if upwind.mode == TargetCondition::NoTargets {
            self.target_value = self.output[index];
            return;
        }

        let matched = upwind.targets.contains(&Some(index));
        if matched {
            upwind.reached += 1;
        }
        let target_reached = match upwind.mode {
            TargetCondition::OneTarget => matched,
            TargetCondition::SomeTargets => upwind.reached == upwind.number_of_targets,
            TargetCondition::AllTargets => upwind.reached == upwind.targets.len(),
            TargetCondition::NoTargets => unreachable!("returned above"),
        };

        if target_reached {
            self.target_value = self.output[index];
            let new_stopping_value = self.target_value + upwind.target_offset;
            if new_stopping_value < self.stopping_value {
                self.stopping_value = new_stopping_value;
            }
        }
    }

    /// `UpdateNeighbors()`. Transcribed with its index bookkeeping intact:
    /// when `index[j]` sits on the low edge the "left" neighbor stays the
    /// center pixel (already alive, hence skipped), and on the high edge the
    /// "right" neighbor stays the left one — a redundant second `UpdateValue`
    /// that recomputes the same solution.
    fn update_neighbors(&mut self, index: usize) -> Result<()> {
        let coord = self.coords_of(index);
        let mut neigh = coord.clone();

        for j in 0..self.size.len() {
            if coord[j] > 0 {
                neigh[j] = coord[j] - 1;
            }
            self.update_if_open(&neigh)?;

            if coord[j] + 1 < self.size[j] {
                neigh[j] = coord[j] + 1;
            }
            self.update_if_open(&neigh)?;

            neigh[j] = coord[j];
        }
        Ok(())
    }

    /// The label gate `UpdateNeighbors` applies before every `UpdateValue`:
    /// alive, initial-trial and outside points are frozen.
    fn update_if_open(&mut self, coord: &[usize]) -> Result<()> {
        let index = self.flat(coord);
        match self.labels[index] {
            Label::Alive | Label::InitialTrial => Ok(()),
            Label::Far | Label::Trial => self.update_value(index, coord),
        }
    }

    /// `UpdateValue()`: solve the upwind quadratic for `index` and, if it
    /// admits a finite arrival time, write it and push a trial node.
    fn update_value(&mut self, index: usize, coord: &[usize]) -> Result<()> {
        let dim = self.size.len();

        // Per axis: the smallest value among that axis's alive neighbors.
        let mut nodes_used: Vec<(f64, usize)> = Vec::with_capacity(dim);
        for (j, &c) in coord.iter().enumerate() {
            let mut value = self.large;
            let base = index - c * self.strides[j];
            for neighbor in [c.checked_sub(1), Some(c + 1)].into_iter().flatten() {
                if neighbor >= self.size[j] {
                    continue;
                }
                let ni = base + neighbor * self.strides[j];
                if self.labels[ni] == Label::Alive && value > self.output[ni] {
                    value = self.output[ni];
                }
            }
            nodes_used.push((value, j));
        }
        // `std::sort` over `LevelSetNode::operator<`, which orders on value
        // alone. Ties are order-independent here (both tied axes are always
        // admitted, and the accumulation is a sum).
        nodes_used.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut solution = self.large;
        let mut aa = 0.0;
        let mut bb = 0.0;
        // cc = speed/F; cc = -1 * sqr(1/cc)
        let scaled_speed = self.speed[index] / self.normalization_factor;
        let inverse_speed = 1.0 / scaled_speed;
        let mut cc = -(inverse_speed * inverse_speed);

        for &(value, axis) in &nodes_used {
            if solution < value {
                break;
            }
            let inverse_spacing = 1.0 / self.spacing[axis];
            let space_factor = inverse_spacing * inverse_spacing;
            aa += space_factor;
            bb += value * space_factor;
            cc += value * value * space_factor;

            let discrim = bb * bb - aa * cc;
            if discrim < 0.0 {
                return Err(FilterError::NegativeDiscriminant);
            }
            solution = (discrim.sqrt() + bb) / aa;
        }

        if solution < self.large {
            let value = self.narrow(solution);
            self.output[index] = value;
            self.labels[index] = Label::Trial;
            self.heap.push(TrialNode { value, index });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(2 + sqrt(2)) / 2`: the exact diagonal arrival time one step out from
    /// a unit-spacing, unit-speed seed (two admitted axes, both at value 1).
    const DIAG: f64 = 1.707_106_781_186_547_6;

    fn speed_f64(size: &[usize], fill: f64) -> Image {
        Image::from_vec(size, vec![fill; size.iter().product()]).unwrap()
    }

    fn march(speed: &Image, seeds: &[Vec<u32>]) -> Vec<f64> {
        fast_marching(speed, seeds, &[], 1.0, f64::MAX / 2.0)
            .unwrap()
            .to_f64_vec()
            .unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "pixel {i}: {a} != {e}");
        }
    }

    #[test]
    fn constant_speed_center_seed_matches_hand_solved_grid() {
        // 3x3, spacing 1, speed 1, seed (1,1). Face neighbors solve one axis:
        // 0 + h/F = 1. Corners admit both axes at value 1: aa=2, bb=2,
        // cc=-1+1+1=1, discrim=2, solution=(sqrt(2)+2)/2.
        let out = march(&speed_f64(&[3, 3], 1.0), &[vec![1, 1]]);
        assert_close(&out, &[DIAG, 1.0, DIAG, 1.0, 0.0, 1.0, DIAG, 1.0, DIAG]);
    }

    #[test]
    fn anisotropic_spacing_scales_each_axis() {
        // spacing = (2, 1). Axis-0 neighbors: 0 + 2 = 2. Axis-1: 0 + 1 = 1.
        // Corners admit (1, axis 1) then (2, axis 0): aa=1/4, bb=1/4,
        // cc=-3/4 -> solution 3; then aa=5/4, bb=9/4, cc=13/4, discrim=1,
        // solution=(1+9/4)/(5/4)=2.6.
        let mut speed = speed_f64(&[3, 3], 1.0);
        speed.set_spacing(&[2.0, 1.0]).unwrap();
        let out = march(&speed, &[vec![1, 1]]);
        assert_close(&out, &[2.6, 1.0, 2.6, 2.0, 0.0, 2.0, 2.6, 1.0, 2.6]);
    }

    #[test]
    fn normalization_factor_divides_the_speed() {
        // speed 2 / F 2 == speed 1 / F 1, pixel for pixel.
        let scaled = fast_marching(&speed_f64(&[3, 3], 2.0), &[vec![1, 1]], &[], 2.0, f64::MAX)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_close(&scaled, &march(&speed_f64(&[3, 3], 1.0), &[vec![1, 1]]));

        // ...and F 4 on speed 2 halves the speed, doubling every arrival time.
        let halved = fast_marching(&speed_f64(&[3, 3], 2.0), &[vec![1, 1]], &[], 4.0, f64::MAX)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_close(
            &halved,
            &[
                2.0 * DIAG,
                2.0,
                2.0 * DIAG,
                2.0,
                0.0,
                2.0,
                2.0 * DIAG,
                2.0,
                2.0 * DIAG,
            ],
        );
    }

    #[test]
    fn two_seeds_give_the_min_of_the_two_fields() {
        // 7x1, seeds at x=0 and x=6: T(x) = min(x, 6-x).
        let out = march(&speed_f64(&[7, 1], 1.0), &[vec![0, 0], vec![6, 0]]);
        assert_close(&out, &[0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn initial_trial_values_offset_their_seed() {
        // Seed x=6 starts at 2.0: T(x) = min(x, 2 + (6-x)).
        let out = fast_marching(
            &speed_f64(&[7, 1], 1.0),
            &[vec![0, 0], vec![6, 0]],
            &[0.0, 2.0],
            1.0,
            f64::MAX / 2.0,
        )
        .unwrap()
        .to_f64_vec()
        .unwrap();
        assert_close(&out, &[0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn stopping_value_truncates_the_alive_region() {
        // 5x5, seed (2,2), stop at 1.0: the four face neighbors (value 1) go
        // alive, the pop of the first diagonal (DIAG > 1) breaks the loop.
        // Trial values already written stay; far pixels hold `large_value`.
        let speed = speed_f64(&[5, 5], 1.0);
        let large = large_value(speed.pixel_id());
        let out = fast_marching(&speed, &[vec![2, 2]], &[], 1.0, 1.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let at = |x: usize, y: usize| out[x + 5 * y];

        assert_eq!(at(2, 2), 0.0);
        assert_eq!(at(2, 1), 1.0);
        // Written as a trial point by an alive face neighbor, never popped.
        assert!((at(1, 1) - DIAG).abs() < 1e-12);
        assert_eq!(at(2, 0), 2.0);
        // Never touched: still far.
        assert_eq!(at(0, 0), large);
        assert_eq!(at(4, 0), large);
        assert_eq!(at(1, 4), large);
    }

    #[test]
    fn zero_speed_blocks_the_front_and_stays_large() {
        // 5x3 with a zero-speed barrier at x=2; seed at (0,1).
        let mut data = vec![1.0f64; 15];
        for y in 0..3 {
            data[2 + 5 * y] = 0.0;
        }
        let speed = Image::from_vec(&[5, 3], data).unwrap();
        let large = large_value(speed.pixel_id());
        let out = march(&speed, &[vec![0, 1]]);
        let at = |x: usize, y: usize| out[x + 5 * y];

        assert_eq!(at(0, 1), 0.0);
        assert_eq!(at(1, 1), 1.0);
        for y in 0..3 {
            for x in 2..5 {
                assert_eq!(at(x, y), large, "({x},{y}) should be unreached");
            }
        }
    }

    #[test]
    fn negative_speed_acts_as_its_magnitude() {
        let negative = march(&speed_f64(&[3, 3], -1.0), &[vec![1, 1]]);
        assert_close(&negative, &march(&speed_f64(&[3, 3], 1.0), &[vec![1, 1]]));
    }

    #[test]
    fn out_of_bounds_trial_points_are_dropped() {
        let speed = speed_f64(&[7, 1], 1.0);
        let large = large_value(speed.pixel_id());

        // One valid seed, one past the edge: only the valid one marches.
        let mixed = march(&speed, &[vec![0, 0], vec![7, 0]]);
        assert_close(&mixed, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Every seed dropped: nothing is ever alive.
        let none = march(&speed, &[vec![7, 0]]);
        assert_close(&none, &[large; 7]);

        // No seeds at all: same.
        assert_close(&march(&speed, &[]), &[large; 7]);
    }

    #[test]
    fn output_pixel_type_is_the_speed_type_real_type() {
        let f32_speed = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        let out = fast_marching(&f32_speed, &[vec![1, 1]], &[], 1.0, f64::MAX / 2.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.scalar_slice::<f32>().unwrap()[0], DIAG as f32);
        // Far pixels hold the float32 large value, not the float64 one.
        assert_eq!(large_value(PixelId::Float32), (f32::MAX / 2.0) as f64);

        let u8_speed = Image::from_vec(&[3, 3], vec![1u8; 9]).unwrap();
        let out = fast_marching(&u8_speed, &[vec![1, 1]], &[], 1.0, f64::MAX / 2.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        assert_eq!(large_value(PixelId::UInt8), f64::MAX / 2.0);
    }

    #[test]
    fn geometry_carries_to_the_output() {
        let mut speed = speed_f64(&[3, 3], 1.0);
        speed.set_spacing(&[2.0, 1.0]).unwrap();
        speed.set_origin(&[-1.0, 4.0]).unwrap();
        let out = fast_marching(&speed, &[vec![1, 1]], &[], 1.0, f64::MAX / 2.0).unwrap();
        assert_eq!(out.spacing(), speed.spacing());
        assert_eq!(out.origin(), speed.origin());
    }

    #[test]
    fn non_positive_normalization_factor_is_rejected() {
        let speed = speed_f64(&[3, 3], 1.0);
        for factor in [0.0, -1.0, ITK_MATH_EPS / 2.0] {
            assert_eq!(
                fast_marching(&speed, &[vec![1, 1]], &[], factor, 1.0).err(),
                Some(FilterError::InvalidNormalizationFactor(factor))
            );
        }
        // The boundary itself is accepted.
        assert!(fast_marching(&speed, &[vec![1, 1]], &[], ITK_MATH_EPS, 1.0).is_ok());
    }

    #[test]
    fn trial_point_of_wrong_length_is_an_error() {
        let speed = speed_f64(&[3, 3], 1.0);
        assert_eq!(
            fast_marching(&speed, &[vec![1, 1, 1]], &[], 1.0, 1.0).err(),
            Some(FilterError::DimensionLength {
                expected: 2,
                got: 3
            })
        );
    }

    /// Both invalid: the normalization factor is checked first, as in ITK.
    #[test]
    fn the_normalization_factor_outranks_the_seed_length() {
        let speed = speed_f64(&[3, 3], 1.0);
        assert_eq!(
            fast_marching(&speed, &[vec![1, 1, 1]], &[], 0.0, 1.0).err(),
            Some(FilterError::InvalidNormalizationFactor(0.0))
        );
    }
}
