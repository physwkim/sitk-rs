//! `IterativeInverseDisplacementFieldImageFilter`
//! (`itkIterativeInverseDisplacementFieldImageFilter.h(.hxx)`): invert a
//! displacement field by a per-pixel coordinate-descent search for the point
//! that maps back onto the pixel.
//!
//! # The algorithm
//!
//! Write `u` for the input field and `y` for an output lattice point. The
//! filter looks for a point `x` with `x + u(x) = y`; then the inverse
//! displacement at `y` is `x − y`.
//!
//! 1. **First guess** (`hxx:46-78`). The negated field `−u` is warped by
//!    *itself* through an `itk::WarpVectorImageFilter`, so the first guess at
//!    `y` is `−u(y − u(y))`, or the warper's `EdgePaddingValue` — the zero
//!    vector (`itkWarpVectorImageFilter.hxx:39-42`) — when `y − u(y)` falls
//!    outside the input's buffer (`itkWarpVectorImageFilter.hxx:170-183`). With
//!    `NumberOfIterations == 0` that warped field *is* the output (`hxx:82-85`).
//!
//! 2. **Coordinate descent** (`hxx:99-216`). Starting from `x = y + guess(y)`,
//!    each of `NumberOfIterations` sweeps probes `x ± step` along every physical
//!    axis in turn, keeping the single probe that most reduces
//!    `‖x + u(x) − y‖`. A probe outside the buffer is skipped, not scored.
//!
//! 3. **Step halving** (`hxx:139-142`). Each axis's `step` starts at that axis's
//!    spacing (see the divergence below) and halves at the *start* of any sweep
//!    that follows a sweep which moved nothing. It is never reset.
//!
//! 4. **Stopping** (`hxx:199-202`). After each sweep, `smallestError <
//!    StopValue` breaks. The default `StopValue` is `0.0`, and the error is a
//!    norm, so the default never stops the loop early — a zero error is not
//!    *less than* zero.
//!
//! # Fixed in this port
//!
//! - **The probe step is each axis's own spacing (fixed here, upstream bug
//!   §2.34).** Upstream reads `const double spacing = inputPtr->GetSpacing()[0];`
//!   (`hxx:89`) — the *first* axis's spacing — and walks every physical axis by
//!   it (`mappedPoint[k] += step`, `hxx:146`), so on an anisotropic field the
//!   coordinate-descent step is wrong on every axis but the first: axis `k` is
//!   probed at a physical distance unrelated to its own sampling. This port
//!   keeps one step per axis, initialized to `spacing[k]`, so each axis is
//!   probed at its own scale. The steps still halve together on a stalled sweep,
//!   so the halving schedule is unchanged. See
//!   `the_probe_step_uses_each_axiss_own_spacing`.
//!
//! # Faithfully reproduced upstream behaviors
//!
//! - **`smallestError` is reset per pixel (fixed here, upstream bug §1.32).**
//!   Upstream declares `double smallestError = 0;` (`hxx:96`) *outside* the
//!   per-pixel loop and only reassigns it when the pixel's initial mapped point
//!   lies inside the buffer (`hxx:122-132`); a pixel whose mapped point starts
//!   outside therefore inherits the previous pixel's error as the value its
//!   probes must beat, making the output depend on the raster order of its
//!   neighbours. This port resets `smallest_error` to `f64::MAX` at the top of
//!   every pixel — the upstream fix PR InsightSoftwareConsortium/ITK#6576 — so
//!   an outside-start pixel searches from a neutral bar (the first in-buffer
//!   probe wins) rather than inheriting a neighbour's residual. See
//!   `smallest_error_is_reset_per_pixel`.
//!
//! - **The search is in physical space, but the probe axes are the physical
//!   axes**, not the lattice axes: `mappedPoint[k] += step` moves along world
//!   axis `k` regardless of the field's direction cosines.
//!
//! - **The first sweep never halves `step`**, because `stillSamePoint` is
//!   initialized to `0` (`hxx:106`).
//!
//! - **`newPoint` records a whole point, not one coordinate.** Within a sweep
//!   the probe on axis `k` starts from the *unperturbed* `mappedPoint`
//!   (`hxx:186` restores it), so a sweep applies exactly one axis's
//!   perturbation — the best-scoring one — and never a combination.
//!
//! # Divergences
//!
//! None. The upstream `InputIt` iterator is advanced but never read
//! (`hxx:97, 212`), so it has no counterpart here.

use sitk_core::Image;

use super::{Field, field_to_image};
use crate::Result;

/// Parameters of `IterativeInverseDisplacementFieldImageFilter`, with the
/// defaults its member initializers give
/// (`itkIterativeInverseDisplacementFieldImageFilter.h:123-125`) and
/// `IterativeInverseDisplacementFieldImageFilter.yaml` repeats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeInverseDisplacementFieldSettings {
    /// Sweeps of the coordinate-descent search, per pixel. `0` outputs the
    /// warped first guess unchanged.
    pub number_of_iterations: u32,
    /// The search stops once the residual `‖x + u(x) − y‖`, in millimetres,
    /// falls strictly below this. The default `0.0` never stops it.
    pub stop_value: f64,
}

impl Default for IterativeInverseDisplacementFieldSettings {
    fn default() -> Self {
        IterativeInverseDisplacementFieldSettings {
            number_of_iterations: 5,
            stop_value: 0.0,
        }
    }
}

/// `‖point + u(point) − original‖`, or `None` when `point` is outside the
/// field's buffer (`hxx:147-155`).
fn residual(forward: &Field, point: &[f64], original: &[f64]) -> Option<f64> {
    let value = forward.evaluate_at_point(point)?;
    Some(
        (0..forward.dim)
            .map(|l| (point[l] + value[l] - original[l]).powi(2))
            .sum::<f64>()
            .sqrt(),
    )
}

/// The first guess: `−u` warped by itself (`hxx:46-78`), an interleaved buffer
/// on the input's own lattice.
fn warp_negated_field_by_itself(forward: &Field) -> Vec<f64> {
    let dim = forward.dim;
    let mut negated = Field::zeros_like(forward);
    for (dst, &src) in negated.data.iter_mut().zip(&forward.data) {
        *dst = -src;
    }

    let mut guess = vec![0.0f64; forward.data.len()];
    for pixel in 0..forward.number_of_pixels() {
        let point = forward.index_to_point(&forward.multi_index(pixel));
        let displacement = negated.vector(pixel);
        let mapped: Vec<f64> = (0..dim).map(|j| point[j] + displacement[j]).collect();
        if let Some(value) = negated.evaluate_at_point(&mapped) {
            guess[pixel * dim..(pixel + 1) * dim].copy_from_slice(&value);
        }
    }
    guess
}

/// Compute the inverse of `displacement_field` by iterative refinement.
///
/// The output is a displacement field on the input's lattice, with the input's
/// geometry and component type.
///
/// Errors: [`super::require_displacement_field`]'s, on an input that is not a
/// real-valued vector image with one component per dimension.
pub fn iterative_inverse_displacement_field(
    displacement_field: &Image,
    settings: &IterativeInverseDisplacementFieldSettings,
) -> Result<Image> {
    let forward = Field::from_image(displacement_field)?;
    let dim = forward.dim;
    let mut values = warp_negated_field_by_itself(&forward);

    if settings.number_of_iterations > 0 {
        for pixel in 0..forward.number_of_pixels() {
            let original = forward.index_to_point(&forward.multi_index(pixel));
            let displacement = &values[pixel * dim..(pixel + 1) * dim];

            let mut mapped: Vec<f64> = (0..dim).map(|j| original[j] + displacement[j]).collect();
            let mut new_point = mapped.clone();

            // Reset per pixel (upstream fix PR #6576): a pixel whose initial
            // mapped point is unevaluable must not inherit the previous pixel's
            // error bar. Upstream declared this once outside the loop and only
            // reassigned it inside the `IsInsideBuffer` branch, making the
            // output depend on raster order (bug §1.32). `f64::MAX` starts every
            // pixel with no bar, so the first in-buffer probe always wins.
            let mut smallest_error = f64::MAX;
            if let Some(error) = residual(&forward, &mapped, &original) {
                smallest_error = error;
            }

            let mut still_same_point = false;
            // One probe step per axis, each initialized to that axis's own
            // spacing (see the module doc: upstream reads only `spacing[0]` and
            // walks every axis by it; this port steps axis `k` by `spacing[k]`).
            // All steps halve together whenever a sweep moved nothing, so the
            // step-halving schedule is unchanged.
            let mut step = forward.spacing.clone();

            for _ in 0..settings.number_of_iterations {
                if still_same_point {
                    for s in step.iter_mut() {
                        *s /= 2.0;
                    }
                }

                for k in 0..dim {
                    for signed in [step[k], -2.0 * step[k]] {
                        mapped[k] += signed;
                        if let Some(error) = residual(&forward, &mapped, &original)
                            && error < smallest_error
                        {
                            smallest_error = error;
                            new_point.copy_from_slice(&mapped);
                        }
                    }
                    // `mappedPoint[k] += step;` (`hxx:186`) restores the axis.
                    mapped[k] += step[k];
                }

                still_same_point = true;
                for j in 0..dim {
                    if new_point[j] != mapped[j] {
                        still_same_point = false;
                    }
                    mapped[j] = new_point[j];
                }

                if smallest_error < settings.stop_value {
                    break;
                }
            }

            for k in 0..dim {
                values[pixel * dim + k] = mapped[k] - original[k];
            }
        }
    }

    field_to_image(
        &forward.size,
        values,
        displacement_field.pixel_id().component_id(),
        &forward.spacing,
        &forward.origin,
        &forward.direction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilterError;
    use sitk_core::PixelId;

    fn field_1d(values: &[f64], spacing: f64) -> Image {
        let mut img = Image::from_vec_vector(&[values.len()], 1, values.to_vec()).unwrap();
        img.set_spacing(&[spacing]).unwrap();
        img
    }

    fn components(img: &Image) -> Vec<f64> {
        img.components_to_f64_vec()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "component {i}: {a} != {e}");
        }
    }

    /// The zero field maps every point to itself, so the residual at the
    /// starting point is already zero and no probe (whose residual is the probe
    /// distance) can beat it.
    #[test]
    fn a_zero_field_inverts_to_the_zero_field() {
        let out = iterative_inverse_displacement_field(
            &field_1d(&[0.0; 5], 1.0),
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        assert_eq!(components(&out), vec![0.0; 5]);
    }

    /// `u ≡ 2` on a unit-spaced 9-point lattice. For `y ≥ 2` the first guess is
    /// `−u(y − 2) = −2`, the mapped point `y − 2` is inside, and the residual
    /// `‖(y−2) + 2 − y‖` is exactly zero — so no probe improves on it and the
    /// output is the exact inverse `−2`.
    ///
    /// The two lattice points at the low edge cannot reach their preimage: it
    /// lies outside the buffer. Their hand-derived values are pinned below.
    #[test]
    fn a_constant_translation_field_inverts_to_the_negated_translation_in_the_interior() {
        let out = iterative_inverse_displacement_field(
            &field_1d(&[2.0; 9], 1.0),
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        assert_close(
            &components(&out),
            &[-0.5, -1.5, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        );
    }

    /// `u = (1, 0)` on a 5×3 unit lattice. Every column `x ≥ 1` has its preimage
    /// `(x−1, j)` inside the buffer, giving a zero residual and the exact
    /// inverse `(−1, 0)`.
    #[test]
    fn a_two_dimensional_translation_field_inverts_exactly_in_the_interior() {
        let mut data = Vec::new();
        for _ in 0..3 {
            for _ in 0..5 {
                data.push(1.0);
                data.push(0.0);
            }
        }
        let img = Image::from_vec_vector(&[5, 3], 2, data).unwrap();
        let out = iterative_inverse_displacement_field(
            &img,
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        let got = components(&out);
        for j in 0..3 {
            for x in 1..5 {
                let pixel = j * 5 + x;
                assert_close(&got[pixel * 2..pixel * 2 + 2], &[-1.0, 0.0]);
            }
        }
    }

    /// `NumberOfIterations == 0` outputs the warped first guess, `−u(y − u(y))`,
    /// zero where `y − u(y)` leaves the buffer (`hxx:82-85`).
    ///
    /// `u = [2, 1.5, 0]` on a unit lattice: `y=0 ↦ −2` is outside, so the guess
    /// is `0`; `y=1 ↦ −0.5` is inside the half-pixel skirt and clamps to
    /// `−u(0) = −2`; `y=2 ↦ 2` gives `−u(2) = 0`.
    #[test]
    fn zero_iterations_outputs_the_negated_field_warped_by_itself() {
        let settings = IterativeInverseDisplacementFieldSettings {
            number_of_iterations: 0,
            ..Default::default()
        };
        let out = iterative_inverse_displacement_field(&field_1d(&[2.0, 1.5, 0.0], 1.0), &settings)
            .unwrap();
        assert_close(&components(&out), &[0.0, -2.0, 0.0]);
    }

    /// `smallest_error` is reset to `f64::MAX` at the top of every pixel (fixed
    /// here, §1.32); upstream declared it once outside the loop, so a pixel
    /// whose initial mapped point is outside the buffer inherited whatever bar
    /// the previous pixel left behind.
    ///
    /// `u = [−1, 1, 0]`, unit spacing, points `x = 0, 1, 2`; the clamped linear
    /// interpolant is `u(x) = −1 + 2x` on `[0, 1]`, `2 − x` on `[1, 2]`, and
    /// flat past each edge inside the `[−0.5, 2.5)` skirt.
    ///
    /// **Pixel 0 is the first pixel and its initial mapped point is outside the
    /// buffer.** Its first guess (`−u` warped by itself) is `−u(1) = −1`, so the
    /// mapped point is `x = 0 + (−1) = −1`, whose continuous index `−1 < −0.5`
    /// is outside; the residual is not computed and `smallest_error` keeps its
    /// reset value. With the fix it is `f64::MAX`, so the first in-buffer probe
    /// always wins and the coordinate descent runs. Sweeping (step `1`, then
    /// halving whenever the point does not move), each accepted probe strictly
    /// the best-so-far: sweep 1 (step 1) probes `x = 0`, residual `|0 + u(0)| =
    /// 1` < MAX, moving to `0`; sweep 2 (step 1, the point moved so no halving)
    /// beats nothing; sweep 3 (step 0.5) probes `x = 0.5`, `u = 0`, residual
    /// `0.5` < `1`, moving to `0.5`; sweep 4 beats nothing; sweep 5 (step 0.25)
    /// probes `x = 0.25`, `u = −0.5`, residual `0.25` < `0.5`, moving to `0.25`.
    /// Five sweeps exhaust `NumberOfIterations`, so the mapped point is `0.25`
    /// and the output displacement is `0.25 − 0 = 0.25`, closing on the true
    /// preimage `x = 1/3` of `y = 0`.
    ///
    /// Under the bug, pixel 0 (the very first pixel) inherits the sentinel
    /// `smallest_error = 0` declared before the loop; no probe residual is ever
    /// `< 0`, so it never moves and outputs its first guess `−1`.
    #[test]
    fn smallest_error_is_reset_per_pixel() {
        let out = iterative_inverse_displacement_field(
            &field_1d(&[-1.0, 1.0, 0.0], 1.0),
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        let got = components(&out);
        assert_close(&got[..1], &[0.25]);
    }

    /// The probe step on axis 0 is axis 0's spacing (`hxx:89`), in physical
    /// units. On a 1-D field the §2.34 fix is a no-op — there is only one axis
    /// and its own spacing is what upstream already read.
    ///
    /// `u = [4, 0, 0]` on a lattice of spacing `2`, so the points are `0, 2, 4`.
    /// Pixel 0's guess is zero (its preimage is outside), giving residual
    /// `|0 + 4 − 0| = 4`. The first probe steps a full `2.0` in physical space
    /// to `x = 2`, where `u = 0` and the residual is `2 < 4`, so the point moves
    /// there and never improves again: the output is `+2`.
    ///
    /// A step of `1.0` — the value a spacing-agnostic reading would use — would
    /// probe `x = 1`, where the interpolated `u = 2` and the residual is `3`,
    /// also an improvement, and the answer would differ.
    #[test]
    fn the_probe_step_is_the_axis_spacing_in_physical_units() {
        let out = iterative_inverse_displacement_field(
            &field_1d(&[4.0, 0.0, 0.0], 2.0),
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        assert_close(&components(&out)[..1], &[2.0]);
    }

    /// Each axis is probed by its *own* spacing (fixed here, §2.34); upstream
    /// stepped every axis by `spacing[0]`.
    ///
    /// A 2×3 field of spacing `(2, 1)` with `u(i, j) = (0, v_j)`,
    /// `v = [3, 3, 0]`. Pixel `(0, 0)` starts at the origin with residual
    /// `‖(0, 3)‖ = 3`. With the fix its axis-1 probe steps by `spacing[1] = 1`:
    /// `(0, 1)` gives `u = (0, 3)`, residual `4` (worse), and `(0, −1)` is
    /// outside, so the first sweep moves nothing. The steps then halve to
    /// `(1, 0.5)`; the axis-1 probe `(0, −0.5)` sits in the half-pixel skirt
    /// where `u` is held at `(0, 3)`, residual `2.5 < 3`, so the point moves
    /// there. From `(0, −0.5)` no further probe (halving `0.25`, `0.125`, …)
    /// improves — the `+j` neighbour is worse and `−j` is outside — so the
    /// coordinate descent settles at the reachable local minimum `(0, −0.5)`,
    /// the genuine best point in the buffer's skirt along axis 1. Output
    /// `(0, −0.5) − (0, 0) = (0, −0.5)`.
    ///
    /// Under the upstream bug axis 1 stepped by `spacing[0] = 2`, jumping the
    /// first sweep straight to `(0, 2)` — where `u = (0, 0)`, residual `2`, and
    /// nothing later beats it — for a different output of `(0, 2)`.
    #[test]
    fn the_probe_step_uses_each_axiss_own_spacing() {
        let v = [3.0, 3.0, 0.0];
        let mut data = Vec::new();
        for &value in &v {
            for _ in 0..2 {
                data.push(0.0);
                data.push(value);
            }
        }
        let mut img = Image::from_vec_vector(&[2, 3], 2, data).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();

        let out = iterative_inverse_displacement_field(
            &img,
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        assert_close(&components(&out)[..2], &[0.0, -0.5]);
    }

    /// A `StopValue` above the residual breaks out of the sweep loop (`hxx:199`).
    ///
    /// On `u ≡ 2` pixel 0's residual after the first sweep is `2` and it has not
    /// moved. With `StopValue = 2.5` the loop breaks there and the output is
    /// `0`; with the default `StopValue = 0.0` four more sweeps run, the step
    /// halves, and the point reaches `−0.5` (see the translation test above).
    /// The default cannot break the loop at all: the residual is a norm, and
    /// `0.0 < 0.0` is false.
    #[test]
    fn a_stop_value_above_the_residual_breaks_the_sweep_loop() {
        let field = field_1d(&[2.0; 9], 1.0);
        let stopped = iterative_inverse_displacement_field(
            &field,
            &IterativeInverseDisplacementFieldSettings {
                number_of_iterations: 5,
                stop_value: 2.5,
            },
        )
        .unwrap();
        let running = iterative_inverse_displacement_field(
            &field,
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();

        assert_close(&components(&stopped)[..1], &[0.0]);
        assert_close(&components(&running)[..1], &[-0.5]);
        // The interior is exact either way: its residual is already zero.
        assert_close(&components(&stopped)[4..5], &[-2.0]);
        assert_close(&components(&running)[4..5], &[-2.0]);
    }

    #[test]
    fn the_output_keeps_the_inputs_geometry_and_component_type() {
        let mut img = Image::from_vec_vector(&[2, 2], 2, vec![0.0f32; 8]).unwrap();
        img.set_spacing(&[0.5, 0.25]).unwrap();
        img.set_origin(&[1.0, -2.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let out = iterative_inverse_displacement_field(
            &img,
            &IterativeInverseDisplacementFieldSettings::default(),
        )
        .unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.spacing(), &[0.5, 0.25]);
        assert_eq!(out.origin(), &[1.0, -2.0]);
        assert_eq!(out.direction(), &[0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn a_scalar_input_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            iterative_inverse_displacement_field(
                &img,
                &IterativeInverseDisplacementFieldSettings::default()
            )
            .unwrap_err(),
            FilterError::Core(sitk_core::Error::RequiresVectorPixelType(PixelId::Float64))
        ));
    }

    #[test]
    fn the_defaults_match_the_yaml() {
        let settings = IterativeInverseDisplacementFieldSettings::default();
        assert_eq!(settings.number_of_iterations, 5);
        assert_eq!(settings.stop_value, 0.0);
    }
}
