//! The layered sparse-field level-set solver, ported from
//! `itkSparseFieldLevelSetImageFilter.h/.hxx`, driven by the iteration and
//! convergence loop of `itkFiniteDifferenceImageFilter.hxx` (`GenerateData`,
//! `Halt`).
//!
//! # The sparse field
//!
//! Rather than evolving `phi` everywhere, the solver keeps `2 * L + 1` lists of
//! pixel indices — the *active layer* (`m_Layers[0]`, the pixels straddling the
//! zero crossing) and `L` layers on each side of it. Odd layer numbers lie
//! inside (negative `phi`), even ones outside. Every non-layer pixel is
//! `STATUS_NULL` in the status image and is never touched.
//!
//! Each iteration:
//!
//! 1. [`calculate_change`](SparseFieldSolver::calculate_change) evaluates the
//!    level-set function at every active-layer index and asks it for a stable
//!    global time step.
//! 2. [`apply_update`](SparseFieldSolver::apply_update) adds `dt * change` to
//!    each active value. A value that leaves the active band `[-g/2, +g/2)` —
//!    where `g` is `m_ConstantGradientValue`, the minimum image spacing — is
//!    demoted out of the active layer, and neighbors from the adjacent layers
//!    are promoted in to replace it. The promotion cascade then propagates
//!    outward through the layers, one layer per round of `ProcessStatusList`.
//! 3. `PropagateAllLayerValues` rebuilds the distance transform in the
//!    non-active layers: each layer's value is its closest `from`-layer
//!    neighbor's value offset by `+/- g`, and a node with no such neighbor is
//!    promoted a layer outward (or deleted past the last layer).
//!
//! # Boundary handling
//!
//! ITK carries an `m_BoundsCheckingActive` flag: it calls
//! `NeedToUseBoundaryConditionOff()` on its neighborhood iterators whenever the
//! sparse field is proven to stay more than `L` pixels away from the image
//! border, which makes out-of-buffer reads return adjacent memory instead of a
//! boundary value. This port has no such flag: it always applies the
//! `ZeroFluxNeumannBoundaryCondition` on reads and always declines
//! out-of-image writes. The two agree, because ITK only turns bounds checking
//! off in exactly the case where no layer node — and therefore no neighborhood
//! center — can be within one pixel of the border.
//!
//! With that boundary condition, an out-of-image neighbor of a layer node
//! clamps back onto the node itself. Every neighbor scan in this file is driven
//! by a status comparison whose target status is never the center's own status,
//! so those scans skip out-of-image neighbors instead of reading the clamped
//! value. Value reads (the derivative stencils,
//! `InitializeActiveLayerValues`) do clamp.

use std::collections::VecDeque;

use super::function::{DifferenceFunction, GlobalData};
use super::grid::{Grid, city_block_neighbors};

/// `SparseFieldLevelSetImageFilter::m_StatusNull`:
/// `NumericTraits<signed char>::NonpositiveMin()`. A pixel outside every layer.
const STATUS_NULL: i32 = -128;
/// A pixel already queued into the next `ProcessStatusList` output, so it is
/// not queued twice.
const STATUS_CHANGING: i32 = -1;
/// An active-layer pixel leaving the active layer outward this iteration.
const STATUS_ACTIVE_CHANGING_UP: i32 = -2;
/// An active-layer pixel leaving the active layer inward this iteration.
const STATUS_ACTIVE_CHANGING_DOWN: i32 = -3;
/// The one-pixel rim of the image.
const STATUS_BOUNDARY_PIXEL: i32 = -4;

/// `SparseFieldLevelSetImageFilter::CalculateUpdateValue`
/// (itkSparseFieldLevelSetImageFilter.h:346-353) and the one override of it
/// among the ported subclasses. It is the sole hook between
/// `UpdateActiveLayerValues`' finite-difference step and the new active-layer
/// value.
pub(super) enum UpdateRule {
    /// The base implementation: `value + dt * change`, with no constraint.
    Unconstrained,
    /// `AntiAliasBinaryImageFilter::CalculateUpdateValue`
    /// (itkAntiAliasBinaryImageFilter.hxx:59-75): the surface may flow under
    /// curvature but never across the interface of the original binary image,
    /// so a pixel whose binary value is the input's maximum is floored at zero
    /// and every other pixel is capped at zero.
    BinaryConstrained {
        /// `m_InputImage`: the *unshifted* binary input, sampled at the
        /// active-layer index.
        input: Vec<f64>,
        /// `m_UpperBinaryValue`, the input's maximum. ITK's test is
        /// `Math::ExactlyEquals`, so a pixel matching neither binary value
        /// takes the `min(.., 0)` branch alongside the lower one.
        upper_binary_value: f64,
    },
}

impl UpdateRule {
    fn calculate_update_value(&self, index: usize, dt: f64, value: f64, change: f64) -> f64 {
        let new_value = value + dt * change;
        match self {
            UpdateRule::Unconstrained => new_value,
            UpdateRule::BinaryConstrained {
                input,
                upper_binary_value,
            } => {
                if input[index] == *upper_binary_value {
                    new_value.max(0.0)
                } else {
                    new_value.min(0.0)
                }
            }
        }
    }
}

/// Everything the concrete `SparseFieldLevelSetImageFilter` subclass fixes
/// before `GenerateData` runs.
pub(super) struct SolverSetup {
    /// `m_ShiftedImage`: the input level set minus `m_IsoSurfaceValue`.
    pub(super) shifted: Vec<f64>,
    /// The `ZeroCrossingImageFilter` map of `shifted`, foreground `0` and
    /// background `1`, which `CopyInputToOutput` grafts onto the output.
    pub(super) zero_crossings: Vec<f64>,
    pub(super) func: DifferenceFunction,
    /// `m_NumberOfLayers`: layers on *one* side of the active layer.
    pub(super) number_of_layers: usize,
    /// `FiniteDifferenceImageFilter::m_UseImageSpacing`, which drives both
    /// `m_ConstantGradientValue` and `InitializeActiveLayerValues`' `MIN_NORM`.
    /// The `SegmentationLevelSetImageFilter`s leave it at its `true` default;
    /// `AntiAliasBinaryImageFilter`'s constructor turns it off.
    pub(super) use_image_spacing: bool,
    pub(super) update_rule: UpdateRule,
}

/// The solver's inputs and evolving state.
pub(super) struct SparseFieldSolver {
    grid: Grid,
    func: DifferenceFunction,
    update_rule: UpdateRule,
    /// The level-set image being evolved; also the filter's output.
    output: Vec<f64>,
    /// `m_ShiftedImage`: the input level set minus `m_IsoSurfaceValue`. Read by
    /// `ConstructActiveLayer`, `InitializeActiveLayerValues` and
    /// `InitializeBackgroundPixels`.
    shifted: Vec<f64>,
    status: Vec<i32>,
    layers: Vec<VecDeque<usize>>,
    update_buffer: Vec<f64>,
    /// `m_NumberOfLayers`: layers on *one* side of the active layer.
    number_of_layers: usize,
    /// `m_ConstantGradientValue`: with `UseImageSpacing` on this is the minimum
    /// spacing, otherwise `1.0`. It is the assumed `|grad(phi)|` used to space
    /// the layers one unit of distance apart.
    constant_gradient_value: f64,
    /// `InitializeActiveLayerValues`' `MIN_NORM`: `1.0e-6`, multiplied by the
    /// minimum spacing only when `UseImageSpacing` is on.
    min_norm: f64,
    neighbors: Vec<(usize, i64)>,
    rms_change: f64,
    elapsed_iterations: u32,
}

/// The output of a solver run: `SparseFieldLevelSetImageFilter`'s level-set
/// image alongside `FiniteDifferenceImageFilter`'s `GetElapsedIterations()`
/// and `GetRMSChange()`.
pub(super) struct SolverOutput {
    pub(super) values: Vec<f64>,
    pub(super) elapsed_iterations: u32,
    pub(super) rms_change: f64,
}

impl SparseFieldSolver {
    pub(super) fn new(size: &[usize], spacing: &[f64], setup: SolverSetup) -> Self {
        let grid = Grid::new(size);
        let dim = grid.dim();
        let min_spacing = spacing.iter().copied().fold(f64::INFINITY, f64::min);
        let layers = 2 * setup.number_of_layers + 1;

        // `Initialize()`: `m_ConstantGradientValue = minSpacing` under
        // `GetUseImageSpacing()`, else `1.0`. `InitializeActiveLayerValues()`
        // scales `MIN_NORM` by the same minimum spacing under the same flag.
        let (constant_gradient_value, min_norm) = if setup.use_image_spacing {
            (min_spacing, 1.0e-6 * min_spacing)
        } else {
            (1.0, 1.0e-6)
        };

        let mut solver = SparseFieldSolver {
            output: setup.zero_crossings,
            shifted: setup.shifted,
            status: vec![STATUS_NULL; grid.number_of_pixels()],
            layers: (0..layers).map(|_| VecDeque::new()).collect(),
            update_buffer: Vec::new(),
            number_of_layers: setup.number_of_layers,
            constant_gradient_value,
            min_norm,
            neighbors: city_block_neighbors(dim),
            rms_change: 0.0,
            elapsed_iterations: 0,
            grid,
            func: setup.func,
            update_rule: setup.update_rule,
        };
        solver.initialize();
        solver
    }

    /// `FiniteDifferenceImageFilter::GenerateData` + `Halt`.
    pub(super) fn run(mut self, maximum_rms_error: f64, number_of_iterations: u32) -> SolverOutput {
        while !self.halt(maximum_rms_error, number_of_iterations) {
            let dt = self.calculate_change();
            self.apply_update(dt);
            self.elapsed_iterations += 1;
        }
        self.post_process_output();
        SolverOutput {
            values: self.output,
            elapsed_iterations: self.elapsed_iterations,
            rms_change: self.rms_change,
        }
    }

    /// `FiniteDifferenceImageFilter::Halt` (itkFiniteDifferenceImageFilter.hxx:210-233).
    /// The RMS test never fires on the first pass, and
    /// `number_of_iterations == 0` halts immediately.
    ///
    /// **Precision divergence from ITK (deliberate, ledger §4.126).** ITK's
    /// `m_RMSChange` field is `double`, but the value is *computed* in
    /// `ValueType` — the level-set pixel type, `float` for a `Float32` level
    /// set: `SparseFieldLevelSetImageFilter::CalculateChange` accumulates
    /// `rms_change_accumulator += sqr(new − old)` and divides by the counter in
    /// `ValueType`, casting to `double` only for the final `sqrt`
    /// (`itkSparseFieldLevelSetImageFilter.hxx:303,344,443`). This port computes
    /// `rms_change` in `f64` throughout. When the true RMS lies within `float`
    /// rounding of `maximum_rms_error`, `float(rms) > thresh` and `f64(rms) >
    /// thresh` can disagree, so the port may run one more or one fewer iteration
    /// than ITK near the threshold. The port keeps `f64` (uniform crate
    /// precision, strictly more accurate).
    fn halt(&self, maximum_rms_error: f64, number_of_iterations: u32) -> bool {
        if self.elapsed_iterations >= number_of_iterations {
            return true;
        }
        if self.elapsed_iterations == 0 {
            return false;
        }
        maximum_rms_error > self.rms_change
    }

    /// The `i`-th city-block neighbor of `coord`, or `None` when it leaves the
    /// image. Returns the linear index only; the caller keeps `coord` intact.
    fn neighbor(&self, coord: &mut [i64], i: usize) -> Option<usize> {
        let (axis, delta) = self.neighbors[i];
        coord[axis] += delta;
        let neighbor = self.grid.in_bounds_index(coord);
        coord[axis] -= delta;
        neighbor
    }

    // ---- Initialization ---------------------------------------------------

    /// `SparseFieldLevelSetImageFilter::Initialize` (hxx:479-580).
    fn initialize(&mut self) {
        // The one-pixel rim, which ITK finds with `ImageBoundaryFacesCalculator`
        // at radius 1.
        for index in 0..self.grid.number_of_pixels() {
            let coord = self.grid.coord(index);
            let on_rim = (0..self.grid.dim())
                .any(|d| coord[d] == 0 || coord[d] == self.grid.size()[d] as i64 - 1);
            if on_rim {
                self.status[index] = STATUS_BOUNDARY_PIXEL;
            }
        }

        self.construct_active_layer();

        // Inside layers are odd, outside layers are even.
        for i in 1..self.layers.len() - 2 {
            self.construct_layer(i as i32, (i + 2) as i32);
        }

        self.initialize_active_layer_values();
        self.propagate_all_layer_values();
        self.initialize_background_pixels();
    }

    /// `ConstructActiveLayer` (hxx:619-701). The active layer is every pixel
    /// the zero-crossing filter marked (`output == 0`); each of its non-marked
    /// city-block neighbors seeds the first inside (odd, `phi < 0`) or first
    /// outside (even) layer.
    ///
    /// ITK guards this push with neither a status check nor a duplicate check,
    /// so a pixel adjacent to two active pixels is pushed onto its layer twice.
    /// Reproduced here: the duplicates are harmless (`PropagateLayerValues` is
    /// idempotent per node) but they are part of the layer contents ITK builds.
    fn construct_active_layer(&mut self) {
        for index in 0..self.grid.number_of_pixels() {
            if self.output[index] != 0.0 {
                continue;
            }
            let mut coord = self.grid.coord(index);

            self.status[index] = 0;
            self.layers[0].push_front(index);

            for i in 0..self.neighbors.len() {
                // An out-of-image neighbor clamps back onto this pixel, whose
                // output value is 0, so ITK's `NotExactlyEquals(.., 0)` test
                // rejects it; skipping is equivalent.
                let Some(neighbor) = self.neighbor(&mut coord, i) else {
                    continue;
                };
                if self.output[neighbor] == 0.0 {
                    continue;
                }

                let layer_number = if self.shifted[neighbor] < 0.0 { 1 } else { 2 };
                self.status[neighbor] = layer_number;
                self.layers[layer_number as usize].push_front(neighbor);
            }
        }
    }

    /// `ConstructLayer` (hxx:705-733): every `STATUS_NULL` neighbor of the
    /// `from` layer joins the `to` layer.
    fn construct_layer(&mut self, from: i32, to: i32) {
        let from_layer = std::mem::take(&mut self.layers[from as usize]);
        for &index in &from_layer {
            let mut coord = self.grid.coord(index);
            for i in 0..self.neighbors.len() {
                // Out-of-image clamps onto the center, whose status is `from`,
                // never `STATUS_NULL`.
                let Some(neighbor) = self.neighbor(&mut coord, i) else {
                    continue;
                };
                if self.status[neighbor] == STATUS_NULL {
                    self.status[neighbor] = to;
                    self.layers[to as usize].push_front(neighbor);
                }
            }
        }
        self.layers[from as usize] = from_layer;
    }

    /// `InitializeActiveLayerValues` (hxx:737-792): seed each active pixel with
    /// the shifted input's distance to the zero crossing, `phi / |grad(phi)|`,
    /// clamped into the active band. The gradient uses the *larger* of the two
    /// one-sided differences per axis.
    fn initialize_active_layer_values(&mut self) {
        let dim = self.grid.dim();
        let change_factor = self.constant_gradient_value / 2.0;
        let min_norm = self.min_norm;
        let scales = self.func.neighborhood_scales().to_vec();

        let active = std::mem::take(&mut self.layers[0]);
        for &index in &active {
            let mut coord = self.grid.coord(index);
            let center = self.shifted[index];

            let mut length = 0.0;
            for (i, &scale) in scales.iter().enumerate().take(dim) {
                coord[i] += 1;
                let forward = self.shifted[self.grid.clamped_index(&coord)];
                coord[i] -= 2;
                let backward = self.shifted[self.grid.clamped_index(&coord)];
                coord[i] += 1;

                let dx_forward = (forward - center) * scale;
                let dx_backward = (center - backward) * scale;
                if dx_forward.abs() > dx_backward.abs() {
                    length += dx_forward * dx_forward;
                } else {
                    length += dx_backward * dx_backward;
                }
            }
            let length = length.sqrt() + min_norm;
            let distance = center / length;

            self.output[index] = distance.clamp(-change_factor, change_factor);
        }
        self.layers[0] = active;
    }

    /// `InitializeBackgroundPixels` (hxx:584-615): everything outside the
    /// sparse field gets a constant one layer beyond the outermost, signed by
    /// the shifted input.
    fn initialize_background_pixels(&mut self) {
        let magnitude = (self.number_of_layers as f64 + 1.0) * self.constant_gradient_value;

        for index in 0..self.grid.number_of_pixels() {
            if self.status[index] == STATUS_NULL || self.status[index] == STATUS_BOUNDARY_PIXEL {
                self.output[index] = if self.shifted[index] > 0.0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
        }
    }

    /// `PostProcessOutput` (hxx:1047-1076): the same flattening, but signed by
    /// the *output* rather than the shifted input, and leaving the rim
    /// (`STATUS_BOUNDARY_PIXEL`) alone.
    fn post_process_output(&mut self) {
        let magnitude = (self.number_of_layers as f64 + 1.0) * self.constant_gradient_value;

        for index in 0..self.grid.number_of_pixels() {
            if self.status[index] == STATUS_NULL {
                self.output[index] = if self.output[index] > 0.0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
        }
    }

    // ---- Iteration --------------------------------------------------------

    /// `CalculateChange` (hxx:809-916), minus the `InterpolateSurfaceLocation`
    /// branch: both ported filters turn that flag off in their constructors.
    fn calculate_change(&mut self) -> f64 {
        let mut gd = GlobalData::new(self.grid.dim());
        let mut buffer = std::mem::take(&mut self.update_buffer);
        buffer.clear();
        buffer.reserve(self.layers[0].len());

        for &index in &self.layers[0] {
            buffer.push(
                self.func
                    .compute_update(&self.output, &self.grid, index, &mut gd),
            );
        }
        self.update_buffer = buffer;

        self.func.compute_global_time_step(&mut gd)
    }

    /// `ApplyUpdate` (hxx:130-199): update the active layer, then walk the
    /// promotion/demotion cascade outward one layer at a time, then rebuild
    /// every layer's distance value.
    fn apply_update(&mut self, dt: f64) {
        let (up0, down0) = self.update_active_layer_values(dt);
        let mut up = [up0, VecDeque::new()];
        let mut down = [down0, VecDeque::new()];

        // First process the status lists generated on the active layer.
        self.process_status_list(&mut up, 0, 2, 1);
        self.process_status_list(&mut down, 0, 1, 2);

        let mut down_to = 0i32;
        let mut up_to = 0i32;
        let mut up_search = 3i32;
        let mut down_search = 4i32;
        let mut j = 1usize;
        let mut k = 0usize;
        while down_search < self.layers.len() as i32 {
            self.process_status_list(&mut up, j, up_to, up_search);
            self.process_status_list(&mut down, j, down_to, down_search);

            if up_to == 0 {
                up_to += 1;
            } else {
                up_to += 2;
            }
            down_to += 2;

            up_search += 2;
            down_search += 2;

            // Swap the lists so the emptied one can be re-used.
            std::mem::swap(&mut j, &mut k);
        }

        // The outermost inside/outside layers of the sparse field.
        self.process_status_list(&mut up, j, up_to, STATUS_NULL);
        self.process_status_list(&mut down, j, down_to, STATUS_NULL);

        // Whatever is left must be brought into the outermost layers: the "up"
        // remainder into the last inside layer, the "down" remainder into the
        // last outside layer.
        let last = self.layers.len() as i32;
        self.process_outside_list(&mut up[k], last - 2);
        self.process_outside_list(&mut down[k], last - 1);

        self.propagate_all_layer_values();
    }

    /// `UpdateActiveLayerValues` (hxx:274-445). Scales the update buffer by
    /// `dt`, adds it to the active layer, and records the RMS change. A value
    /// that leaves `[-g/2, g/2)` demotes its pixel out of the active layer and
    /// drags a neighbor from the adjacent layer in behind it.
    ///
    /// Returns the "up" (outward) and "down" (inward) status lists.
    fn update_active_layer_values(&mut self, dt: f64) -> (VecDeque<usize>, VecDeque<usize>) {
        let lower_active_threshold = -(self.constant_gradient_value / 2.0);
        let upper_active_threshold = self.constant_gradient_value / 2.0;

        let mut up_list = VecDeque::new();
        let mut down_list = VecDeque::new();

        let active = std::mem::take(&mut self.layers[0]);
        let mut survivors = VecDeque::new();
        let mut counter = 0usize;
        let mut rms_change_accumulator = 0.0f64;

        for (n, &index) in active.iter().enumerate() {
            let center = self.output[index];
            let new_value =
                self.update_rule
                    .calculate_update_value(index, dt, center, self.update_buffer[n]);
            let mut coord = self.grid.coord(index);

            if new_value >= upper_active_threshold {
                // Never demote across a neighbor moving the opposite way; that
                // would punch a hole in the active layer.
                if self.has_neighbor_with_status(&mut coord, STATUS_ACTIVE_CHANGING_DOWN) {
                    survivors.push_back(index);
                    continue;
                }
                rms_change_accumulator += (new_value - center).powi(2);

                // Pull the closest first-inside-layer neighbor up into the
                // active layer, keeping the value closest to the zero level set.
                let temp_value = new_value - self.constant_gradient_value;
                self.promote_neighbors(&mut coord, 1, temp_value, |old, new| {
                    old < lower_active_threshold || new.abs() < old.abs()
                });

                up_list.push_front(index);
                self.status[index] = STATUS_ACTIVE_CHANGING_UP;
            } else if new_value < lower_active_threshold {
                if self.has_neighbor_with_status(&mut coord, STATUS_ACTIVE_CHANGING_UP) {
                    survivors.push_back(index);
                    continue;
                }
                rms_change_accumulator += (new_value - center).powi(2);

                let temp_value = new_value + self.constant_gradient_value;
                self.promote_neighbors(&mut coord, 2, temp_value, |old, new| {
                    old >= upper_active_threshold || new.abs() < old.abs()
                });

                down_list.push_front(index);
                self.status[index] = STATUS_ACTIVE_CHANGING_DOWN;
            } else {
                rms_change_accumulator += (new_value - center).powi(2);
                self.output[index] = new_value;
                survivors.push_back(index);
            }
            counter += 1;
        }

        self.layers[0] = survivors;
        self.rms_change = if counter == 0 {
            0.0
        } else {
            (rms_change_accumulator / counter as f64).sqrt()
        };

        (up_list, down_list)
    }

    /// Does any city-block neighbor of `coord` carry `status`? Out-of-image
    /// neighbors clamp onto the center, whose status is `0` at this point in
    /// `UpdateActiveLayerValues` and therefore never one of the
    /// `STATUS_ACTIVE_CHANGING_*` values asked about here.
    fn has_neighbor_with_status(&self, coord: &mut [i64], status: i32) -> bool {
        (0..self.neighbors.len()).any(|i| {
            self.neighbor(coord, i)
                .is_some_and(|neighbor| self.status[neighbor] == status)
        })
    }

    /// Assign `value` to every neighbor in layer `layer` whose current value
    /// `keep` accepts as replaceable. Out-of-image neighbors clamp onto the
    /// center (status `0`), so they never match `layer` (`1` or `2`); skipping
    /// them also keeps the write inside the image, as ITK's bounds-checked
    /// `SetPixel` does.
    fn promote_neighbors(
        &mut self,
        coord: &mut [i64],
        layer: i32,
        value: f64,
        replace: impl Fn(f64, f64) -> bool,
    ) {
        for i in 0..self.neighbors.len() {
            let Some(neighbor) = self.neighbor(coord, i) else {
                continue;
            };
            if self.status[neighbor] == layer && replace(self.output[neighbor], value) {
                self.output[neighbor] = value;
            }
        }
    }

    /// `ProcessStatusList` (hxx:219-270): move every index of `lists[j]` into
    /// layer `change_to`, and queue onto `lists[1 - j]` each of their neighbors
    /// that currently sits in layer `search_for`.
    fn process_status_list(
        &mut self,
        lists: &mut [VecDeque<usize>; 2],
        j: usize,
        change_to: i32,
        search_for: i32,
    ) {
        while let Some(index) = lists[j].pop_front() {
            self.status[index] = change_to;
            self.layers[change_to as usize].push_front(index);

            let mut coord = self.grid.coord(index);
            for i in 0..self.neighbors.len() {
                // Out-of-image clamps onto the center, whose status is now
                // `change_to`; `change_to < search_for` at every call site, so
                // it can match neither `search_for` nor `STATUS_BOUNDARY_PIXEL`.
                let Some(neighbor) = self.neighbor(&mut coord, i) else {
                    continue;
                };
                if self.status[neighbor] == search_for {
                    // Mark it so it is not queued twice.
                    self.status[neighbor] = STATUS_CHANGING;
                    lists[1 - j].push_front(neighbor);
                }
            }
        }
    }

    /// `ProcessOutsideList` (hxx:203-215): drain a list straight into a layer.
    fn process_outside_list(&mut self, list: &mut VecDeque<usize>, change_to: i32) {
        while let Some(index) = list.pop_front() {
            self.status[index] = change_to;
            self.layers[change_to as usize].push_front(index);
        }
    }

    /// `PropagateAllLayerValues` (hxx:920-933).
    fn propagate_all_layer_values(&mut self) {
        self.propagate_layer_values(0, 1, 3, 1); // first inside
        self.propagate_layer_values(0, 2, 4, 2); // first outside

        for i in 1..self.layers.len() - 2 {
            let i = i as i32;
            self.propagate_layer_values(i, i + 2, i + 4, (i + 2) % 2);
        }
    }

    /// `PropagateLayerValues` (hxx:937-1043): re-derive the `to` layer's values
    /// from its `from` neighbors, one `m_ConstantGradientValue` further from
    /// the zero level set. A `to` node with no `from` neighbor is promoted to
    /// the `promote` layer, or dropped from the sparse field when `promote`
    /// runs past the last layer.
    fn propagate_layer_values(&mut self, from: i32, to: i32, promote: i32, in_or_out: i32) {
        // Inward means "more negative".
        let delta = if in_or_out == 1 {
            -self.constant_gradient_value
        } else {
            self.constant_gradient_value
        };
        let past_end = self.layers.len() as i32 - 1;

        let to_layer = std::mem::take(&mut self.layers[to as usize]);
        let mut survivors = VecDeque::new();

        for &index in &to_layer {
            // A node another layer has already claimed is stale; drop it.
            if self.status[index] != to {
                continue;
            }

            let mut coord = self.grid.coord(index);
            let mut value = 0.0f64;
            let mut found_neighbor = false;
            for i in 0..self.neighbors.len() {
                // Out-of-image clamps onto the center, whose status is `to`,
                // never `from`.
                let Some(neighbor) = self.neighbor(&mut coord, i) else {
                    continue;
                };
                if self.status[neighbor] != from {
                    continue;
                }

                let candidate = self.output[neighbor];
                if !found_neighbor {
                    value = candidate;
                } else if in_or_out == 1 {
                    // The largest (least negative) neighbor.
                    value = value.max(candidate);
                } else {
                    // The smallest (least positive) neighbor.
                    value = value.min(candidate);
                }
                found_neighbor = true;
            }

            if found_neighbor {
                self.output[index] = value + delta;
                survivors.push_back(index);
            } else if promote > past_end {
                self.status[index] = STATUS_NULL;
            } else {
                self.layers[promote as usize].push_front(index);
                self.status[index] = promote;
            }
        }

        self.layers[to as usize] = survivors;
    }
}
