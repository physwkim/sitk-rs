//! Fast marching that also records the upwind gradient of the arrival-time
//! field, and can stop once target points are reached.
//!
//! Port of `itk::FastMarchingUpwindGradientImageFilter`
//! (`itkFastMarchingUpwindGradientImageFilter.h` / `.hxx`), with the API
//! surface `FastMarchingUpwindGradientImageFilter.yaml` declares. It is a
//! subclass of the [`crate::fast_marching`] solver and reuses its heap through
//! that module's `march_flat` seam; only the per-accepted-point tail of
//! `UpdateNeighbors` is new.
//!
//! ## The gradient output is a vector image upstream
//!
//! `m_GradientImage` is an `Image<CovariantVector<PixelType, Dimension>>`.
//! This crate has no vector pixel type, so [`FastMarchingUpwindGradientResult`]
//! carries the **scalar decomposition** of that vector output: one scalar image
//! per axis, `gradient[j]` holding the upstream `gradientPixel[j]`, in the
//! output's own pixel type ([`crate::fast_marching::output_pixel_id`]). The
//! `gradient` vector is empty when `generate_gradient_image` is clear, matching
//! ITK, which then never allocates the image at all.
//!
//! SimpleITK's yaml exposes `GradientImage` as a measurement but declares no
//! `GenerateGradientImage` member and never turns the flag on, so
//! `sitk::FastMarchingUpwindGradientImageFilter::GetGradientImage()` always
//! returns an unallocated image. This port exposes the flag instead of
//! reproducing that dead end; its default is ITK's (`false`).
//!
//! ## `ComputeGradient`
//!
//! Run for the point just made alive, after the base class has updated its
//! neighbors. Per axis `j`, with `centerPixel = T(index)`:
//!
//! ```text
//! dx_backward = (back  in image && back  is ALIVE) ? center - T(back)    : 0
//! dx_forward  = (fwd   in image && fwd   is ALIVE) ? T(fwd)  - center    : 0
//!
//! g[j] = max(dx_backward, -dx_forward) < 0 ? 0
//!      : dx_backward > -dx_forward        ? dx_backward
//!      :                                    dx_forward
//! g[j] /= spacing[j]
//! ```
//!
//! Only *alive* neighbors enter — "the front can only come from there". A
//! neighbor that is far, trial, or outside contributes a difference of exactly
//! zero, and two zero differences fall through the `else` to `dx_forward`, so
//! the seed point (accepted before any neighbor is alive) gets a zero gradient,
//! as does any point accepted with no alive neighbor along `j`.
//!
//! ## Targets
//!
//! The mode is not exposed directly; SimpleITK derives it from
//! `number_of_targets`, and this port copies that mapping exactly:
//!
//! | `number_of_targets` | mode | `m_NumberOfTargets` |
//! |---|---|---|
//! | `0` | `NoTargets` | 0 |
//! | `1` | `OneTarget` | 1 |
//! | `n > 1` | `SomeTargets` | `min(n, target_points.len())` |
//!
//! ITK's fourth mode, `AllTargets`, is unreachable from this API — the clamp
//! makes `SomeTargets` with `n >= target_points.len()` behave identically
//! anyway, since each in-bounds target is accepted at most once.
//!
//! Every mode but `NoTargets` requires at least one target point
//! (`VerifyTargetReachedModeConditions`); an empty list is
//! [`FilterError::NoTargetPoints`]. That check runs in `VerifyPreconditions`,
//! ahead of `GenerateData`'s normalization-factor check, and this port keeps
//! that order.
//!
//! When a target is reached, `m_TargetValue` takes the accepted point's arrival
//! time and the stopping value drops to `m_TargetValue + target_offset` — but
//! only if that *lowers* it. Three upstream behaviors follow, reproduced here
//! rather than fixed:
//!
//! - **`NoTargets` overwrites `m_TargetValue` at every accepted point**, so it
//!   ends as the last (largest) Eikonal value generated. SimpleITK's
//!   `GetTargetValue` documents this.
//! - **`SomeTargets` latches on the count, not on the point.** Once
//!   `reached == number_of_targets`, the test outside the target lookup keeps
//!   returning true, so every subsequent accepted point — target or not —
//!   moves `m_TargetValue` forward. The stopping value does not move with it,
//!   because `m_TargetValue + offset` only grows from there.
//! - **`OneTarget` reports the *last* reached target**, not the first: a second
//!   target accepted before the march stops overwrites `m_TargetValue`.
//!
//! Target points are never bounds-filtered upstream: an out-of-image target
//! node stays in the container (so it counts towards the size `AllTargets`
//! compares against) and simply never matches an accepted index. Trial points
//! *are* dropped when out of bounds, by the base class's `Initialize()`.
//!
//! ## Seed values
//!
//! A trial or target point is `dim` unsigned indices, optionally followed by a
//! `dim + 1`-th element that SimpleITK's cast reads as the node's initial value
//! (`if (m_TrialPoints[i].size() > SetDimension) node.SetValue(...)`); target
//! nodes always get value `0.0` and ignore it. `initial_trial_values`, applied
//! last by the yaml's member order, overrides that value positionally against
//! the *full* trial list, out-of-bounds points included.

use sitk_core::{Image, PixelId};

use crate::error::{FilterError, Result};
use crate::fast_marching::{
    MarchInput, TargetCondition, UpwindInput, check_normalization_factor, large_value, march_flat,
    output_pixel_id, strides,
};
use crate::image_from_f64;

/// The scalar members `FastMarchingUpwindGradientImageFilter.yaml` declares,
/// plus ITK's `GenerateGradientImage` flag, which SimpleITK never sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastMarchingUpwindGradientSettings {
    /// Selects the target-reached mode; see the module doc's table.
    pub number_of_targets: u32,
    /// `m_TargetOffset`: how far past the reached target the front runs.
    pub target_offset: f64,
    /// Divides the speed image; must be `>= f64::EPSILON`.
    pub normalization_factor: f64,
    /// `m_GenerateGradientImage`. When clear,
    /// [`FastMarchingUpwindGradientResult::gradient`] is empty.
    pub generate_gradient_image: bool,
}

impl Default for FastMarchingUpwindGradientSettings {
    /// The constructor defaults: `m_TargetOffset = 1.0`,
    /// `m_TargetReachedMode = NoTargets` (`number_of_targets = 0`),
    /// `m_NormalizationFactor = 1.0`, `m_GenerateGradientImage = false`.
    fn default() -> Self {
        Self {
            number_of_targets: 0,
            target_offset: 1.0,
            normalization_factor: 1.0,
            generate_gradient_image: false,
        }
    }
}

pub struct FastMarchingUpwindGradientResult {
    /// The arrival-time field `T`, in [`output_pixel_id`]; unreached pixels
    /// hold [`large_value`].
    pub arrival_time: Image,
    /// The upwind gradient of `T`, one scalar image per axis. Empty unless
    /// [`FastMarchingUpwindGradientSettings::generate_gradient_image`] is set.
    pub gradient: Vec<Image>,
    /// `m_TargetValue`; see the module doc for what it holds in each mode.
    /// `0.0` when the march accepts no point at all.
    pub target_value: f64,
}

/// SimpleITK's `TargetPoints` cast: the mode and `m_NumberOfTargets` that
/// `number_of_targets` selects, given how many target points were supplied.
fn target_mode(number_of_targets: u32, available: usize) -> (TargetCondition, usize) {
    match number_of_targets {
        0 => (TargetCondition::NoTargets, 0),
        1 => (TargetCondition::OneTarget, 1),
        n => (TargetCondition::SomeTargets, (n as usize).min(available)),
    }
}

/// `sitkSTLVectorToITK<IndexType>`: the first `dim` elements are the index, and
/// fewer than `dim` is "Unable to convert vector to ITK type".
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

/// `FastMarchingUpwindGradientImageFilter`: the [`crate::fast_marching`] march,
/// plus the upwind gradient of the arrival-time field and an optional target
/// stopping condition.
///
/// - `speed` is the speed image; its geometry carries to every output.
/// - `trial_points` are image indices (`dim` elements, optionally a `dim + 1`-th
///   giving the seed's initial value). Points outside the image are silently
///   dropped, as in `Initialize()`.
/// - `initial_trial_values` overrides the seed values positionally, against the
///   full `trial_points` list.
/// - `target_points` are image indices; out-of-image targets are kept and never
///   matched.
///
/// The stopping value is not exposed: SimpleITK's yaml omits it, leaving ITK's
/// constructor default of `double(m_LargeValue)` — see [`large_value`] — which
/// a reached target then lowers.
pub fn fast_marching_upwind_gradient(
    speed: &Image,
    trial_points: &[Vec<u32>],
    initial_trial_values: &[f64],
    target_points: &[Vec<u32>],
    settings: &FastMarchingUpwindGradientSettings,
) -> Result<FastMarchingUpwindGradientResult> {
    let size = speed.size();
    let dim = size.len();

    // `VerifyPreconditions()` runs before `GenerateData()`, so the target-mode
    // check outranks the normalization-factor check.
    let (mode, number_of_targets) = target_mode(settings.number_of_targets, target_points.len());
    if mode != TargetCondition::NoTargets && target_points.is_empty() {
        return Err(FilterError::NoTargetPoints);
    }
    check_normalization_factor(settings.normalization_factor)?;
    check_point_lengths(trial_points, dim)?;
    check_point_lengths(target_points, dim)?;

    let out_id = output_pixel_id(speed.pixel_id());
    let strides = strides(size);

    let trial: Vec<(usize, f64)> = trial_points
        .iter()
        .enumerate()
        .filter_map(|(i, point)| {
            let index = flat_index(point, dim, size, &strides)?;
            let seed_value = point.get(dim).map_or(0.0, |&v| f64::from(v));
            let value = initial_trial_values.get(i).copied().unwrap_or(seed_value);
            Some((index, value))
        })
        .collect();
    let targets: Vec<Option<usize>> = target_points
        .iter()
        .map(|point| flat_index(point, dim, size, &strides))
        .collect();

    let result = march_flat(
        MarchInput {
            size,
            spacing: speed.spacing(),
            speed: &speed.to_f64_vec(),
            narrow_to_f32: out_id == PixelId::Float32,
            normalization_factor: settings.normalization_factor,
            // `m_StoppingValue`'s constructor default; a reached target lowers it.
            stopping_value: large_value(speed.pixel_id()),
            collect_points: false,
            upwind: Some(UpwindInput {
                generate_gradient: settings.generate_gradient_image,
                targets: &targets,
                target_mode: mode,
                number_of_targets,
                target_offset: settings.target_offset,
            }),
        },
        &trial,
    )?;

    Ok(FastMarchingUpwindGradientResult {
        arrival_time: image_from_f64(out_id, size, speed, &result.values)?,
        gradient: result
            .gradient
            .iter()
            .map(|axis| image_from_f64(out_id, size, speed, axis))
            .collect::<Result<Vec<_>>>()?,
        target_value: result.target_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(2 + sqrt(2)) / 2 - 1 == 1 / sqrt(2)`: the one-sided difference between
    /// a diagonal pixel and its face neighbor in a unit-speed, unit-spacing
    /// march.
    const DIAG_DROP: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn speed_f64(size: &[usize], fill: f64) -> Image {
        Image::from_vec(size, vec![fill; size.iter().product()]).unwrap()
    }

    fn with_gradient() -> FastMarchingUpwindGradientSettings {
        FastMarchingUpwindGradientSettings {
            generate_gradient_image: true,
            ..Default::default()
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "pixel {i}: {a} != {e}");
        }
    }

    /// A 1-D ramp: the upwind gradient of the arrival time is `1 / speed`
    /// everywhere the front passed through a backward neighbor, and `0` at the
    /// seed, which is accepted before any neighbor is alive.
    #[test]
    fn one_dimensional_ramp_gradient_is_the_inverse_speed() {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[5, 1], 2.0),
            &[vec![0, 0]],
            &[],
            &[],
            &with_gradient(),
        )
        .unwrap();

        assert_close(&out.arrival_time.to_f64_vec(), &[0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(out.gradient.len(), 2);
        assert_close(&out.gradient[0].to_f64_vec(), &[0.0, 0.5, 0.5, 0.5, 0.5]);
        // Axis 1 has extent 1: no neighbor, so both differences stay zero.
        assert_close(&out.gradient[1].to_f64_vec(), &[0.0; 5]);
    }

    /// The stencil admits only *alive* neighbors. A face pixel of a 3x3 march
    /// is accepted while both of its cross-axis neighbors are still far, so
    /// that axis's gradient is exactly zero rather than the centered slope; the
    /// corners, accepted last, see alive neighbors on both faces.
    #[test]
    fn gradient_at_a_front_corner_uses_only_alive_neighbors() {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[3, 3], 1.0),
            &[vec![1, 1]],
            &[],
            &[],
            &with_gradient(),
        )
        .unwrap();

        #[rustfmt::skip]
        assert_close(
            &out.gradient[0].to_f64_vec(),
            &[
                -DIAG_DROP, 0.0, DIAG_DROP,
                -1.0,       0.0, 1.0,
                -DIAG_DROP, 0.0, DIAG_DROP,
            ],
        );
        #[rustfmt::skip]
        assert_close(
            &out.gradient[1].to_f64_vec(),
            &[
                -DIAG_DROP, -1.0, -DIAG_DROP,
                 0.0,        0.0,  0.0,
                 DIAG_DROP,  1.0,  DIAG_DROP,
            ],
        );
    }

    /// `(1, 0)` is accepted with `(1, 1)` alive and `(0, 0)` / `(2, 0)` far, so
    /// its axis-0 gradient falls through both zero differences to `dx_forward`.
    #[test]
    fn a_gradient_axis_with_no_alive_neighbor_is_zero() {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[3, 3], 1.0),
            &[vec![1, 1]],
            &[],
            &[],
            &with_gradient(),
        )
        .unwrap();
        assert_eq!(out.gradient[0].to_f64_vec()[1], 0.0);
        assert_eq!(out.gradient[1].to_f64_vec()[1], -1.0);
        // The seed sees no alive neighbor on either axis.
        assert_eq!(out.gradient[0].to_f64_vec()[4], 0.0);
        assert_eq!(out.gradient[1].to_f64_vec()[4], 0.0);
    }

    #[test]
    fn the_gradient_is_absent_unless_requested() {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[3, 3], 1.0),
            &[vec![1, 1]],
            &[],
            &[],
            &FastMarchingUpwindGradientSettings::default(),
        )
        .unwrap();
        assert!(out.gradient.is_empty());
    }

    #[test]
    fn the_gradient_axis_is_divided_by_that_axis_spacing() {
        let mut speed = speed_f64(&[5, 1], 1.0);
        speed.set_spacing(&[2.0, 1.0]).unwrap();
        let out = fast_marching_upwind_gradient(&speed, &[vec![0, 0]], &[], &[], &with_gradient())
            .unwrap();
        // T(x) = 2x; dx_backward = 2, divided by spacing 2.
        assert_close(&out.arrival_time.to_f64_vec(), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_close(&out.gradient[0].to_f64_vec(), &[0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    /// A 9-pixel line seeded at `x = 4`, with targets at `x = 3` (reached at
    /// `T = 1`) and `x = 8` (reached at `T = 4`).
    fn asymmetric_targets(number_of_targets: u32, target_offset: f64) -> (Vec<f64>, f64) {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[9, 1], 1.0),
            &[vec![4, 0]],
            &[],
            &[vec![3, 0], vec![8, 0]],
            &FastMarchingUpwindGradientSettings {
                number_of_targets,
                target_offset,
                ..Default::default()
            },
        )
        .unwrap();
        (out.arrival_time.to_f64_vec(), out.target_value)
    }

    /// `OneTarget` stops one offset past the *nearest* target; `SomeTargets(2)`
    /// must wait for the far one, so it marches the whole line.
    #[test]
    fn one_target_and_some_targets_stop_at_different_values() {
        let large = large_value(PixelId::Float64);

        let (one, one_value) = asymmetric_targets(1, 1.0);
        assert_eq!(one_value, 1.0);
        assert_close(&one, &[large, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, large]);

        let (some, some_value) = asymmetric_targets(2, 1.0);
        assert_eq!(some_value, 4.0);
        assert_close(&some, &[4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);

        // `number_of_targets` beyond the supplied count clamps to it, so the
        // stop is the same as `SomeTargets(2)` — SimpleITK's `min()`.
        let (clamped, clamped_value) = asymmetric_targets(7, 1.0);
        assert_eq!(clamped_value, 4.0);
        assert_close(&clamped, &some);
    }

    /// A larger `target_offset` runs the front further past the target.
    #[test]
    fn target_offset_widens_the_marched_region() {
        let large = large_value(PixelId::Float64);

        let (tight, _) = asymmetric_targets(1, 0.0);
        assert_close(
            &tight,
            &[large, large, 2.0, 1.0, 0.0, 1.0, 2.0, large, large],
        );

        let (wide, _) = asymmetric_targets(1, 2.0);
        assert_close(&wide, &[4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    /// `SomeTargets` tests the reached *count* outside the target lookup, so
    /// once the count is met every later accepted point moves `m_TargetValue`
    /// forward. `OneTarget` only moves it on an accepted target.
    fn near_targets(number_of_targets: u32) -> (Vec<f64>, f64) {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[9, 1], 1.0),
            &[vec![4, 0]],
            &[],
            &[vec![3, 0], vec![5, 0]],
            &FastMarchingUpwindGradientSettings {
                number_of_targets,
                ..Default::default()
            },
        )
        .unwrap();
        (out.arrival_time.to_f64_vec(), out.target_value)
    }

    #[test]
    fn some_targets_keeps_advancing_the_target_value_after_the_count_is_met() {
        let large = large_value(PixelId::Float64);
        let marched = [large, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, large];

        // Both targets accepted at T = 1, so the stop is 1 + 1 = 2 either way.
        let (one, one_value) = near_targets(1);
        assert_close(&one, &marched);
        assert_eq!(one_value, 1.0);

        // ...but the count stays met while `x = 2` and `x = 6` (T = 2) are
        // accepted, and each of them overwrites `m_TargetValue`.
        let (some, some_value) = near_targets(2);
        assert_close(&some, &marched);
        assert_eq!(some_value, 2.0);
    }

    /// With no targets, `m_TargetValue` is the last (largest) Eikonal value.
    #[test]
    fn no_targets_reports_the_largest_eikonal_value() {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[5, 1], 1.0),
            &[vec![0, 0]],
            &[],
            &[],
            &FastMarchingUpwindGradientSettings::default(),
        )
        .unwrap();
        assert_eq!(out.target_value, 4.0);
    }

    /// A target list is kept whole: an out-of-image target simply never matches
    /// an accepted index, so the front never stops and `m_TargetValue` keeps
    /// its `Initialize()` value of `0.0`.
    #[test]
    fn an_out_of_bounds_target_is_never_reached() {
        let out = fast_marching_upwind_gradient(
            &speed_f64(&[5, 1], 1.0),
            &[vec![0, 0]],
            &[],
            &[vec![9, 0]],
            &FastMarchingUpwindGradientSettings {
                number_of_targets: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_close(&out.arrival_time.to_f64_vec(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.target_value, 0.0);
    }

    /// Trial points are bounds-filtered by `Initialize()`; a march with no
    /// surviving seed accepts nothing.
    #[test]
    fn out_of_bounds_trial_points_are_dropped() {
        let speed = speed_f64(&[5, 1], 1.0);
        let large = large_value(speed.pixel_id());
        let out = fast_marching_upwind_gradient(&speed, &[vec![9, 0]], &[], &[], &with_gradient())
            .unwrap();
        assert_close(&out.arrival_time.to_f64_vec(), &[large; 5]);
        assert_close(&out.gradient[0].to_f64_vec(), &[0.0; 5]);
        assert_eq!(out.target_value, 0.0);
    }

    /// A `dim + 1`-th element of a trial point is its initial value, and
    /// `initial_trial_values` — applied last by the yaml's member order —
    /// overrides it positionally.
    #[test]
    fn a_trailing_seed_value_sets_the_initial_arrival_time() {
        let settings = FastMarchingUpwindGradientSettings::default();
        let speed = speed_f64(&[5, 1], 1.0);

        let out =
            fast_marching_upwind_gradient(&speed, &[vec![0, 0, 3]], &[], &[], &settings).unwrap();
        assert_close(&out.arrival_time.to_f64_vec(), &[3.0, 4.0, 5.0, 6.0, 7.0]);

        let out = fast_marching_upwind_gradient(&speed, &[vec![0, 0, 3]], &[1.0], &[], &settings)
            .unwrap();
        assert_close(&out.arrival_time.to_f64_vec(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn every_target_mode_but_no_targets_needs_a_target_point() {
        let speed = speed_f64(&[5, 1], 1.0);
        for number_of_targets in [1, 2, 7] {
            assert_eq!(
                fast_marching_upwind_gradient(
                    &speed,
                    &[vec![0, 0]],
                    &[],
                    &[],
                    &FastMarchingUpwindGradientSettings {
                        number_of_targets,
                        ..Default::default()
                    },
                )
                .err(),
                Some(FilterError::NoTargetPoints),
            );
        }
        // `NoTargets` never looks at the container.
        assert!(
            fast_marching_upwind_gradient(
                &speed,
                &[vec![0, 0]],
                &[],
                &[],
                &FastMarchingUpwindGradientSettings::default(),
            )
            .is_ok()
        );
    }

    /// `VerifyPreconditions()` runs before `GenerateData()`.
    #[test]
    fn the_target_mode_outranks_the_normalization_factor() {
        assert_eq!(
            fast_marching_upwind_gradient(
                &speed_f64(&[5, 1], 1.0),
                &[vec![0, 0]],
                &[],
                &[],
                &FastMarchingUpwindGradientSettings {
                    number_of_targets: 1,
                    normalization_factor: 0.0,
                    ..Default::default()
                },
            )
            .err(),
            Some(FilterError::NoTargetPoints),
        );
    }

    #[test]
    fn a_non_positive_normalization_factor_is_rejected() {
        assert_eq!(
            fast_marching_upwind_gradient(
                &speed_f64(&[5, 1], 1.0),
                &[vec![0, 0]],
                &[],
                &[],
                &FastMarchingUpwindGradientSettings {
                    normalization_factor: 0.0,
                    ..Default::default()
                },
            )
            .err(),
            Some(FilterError::InvalidNormalizationFactor(0.0)),
        );
    }

    #[test]
    fn a_point_shorter_than_the_image_dimension_is_an_error() {
        let speed = speed_f64(&[5, 5], 1.0);
        let settings = FastMarchingUpwindGradientSettings {
            number_of_targets: 1,
            ..Default::default()
        };
        let short = || {
            Some(FilterError::DimensionLength {
                expected: 2,
                got: 1,
            })
        };
        assert_eq!(
            fast_marching_upwind_gradient(&speed, &[vec![0]], &[], &[vec![1, 1]], &settings).err(),
            short(),
        );
        assert_eq!(
            fast_marching_upwind_gradient(&speed, &[vec![0, 0]], &[], &[vec![1]], &settings).err(),
            short(),
        );
    }

    #[test]
    fn geometry_and_pixel_type_carry_to_every_output() {
        let mut speed = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        speed.set_spacing(&[2.0, 1.0]).unwrap();
        speed.set_origin(&[-1.0, 4.0]).unwrap();
        let out = fast_marching_upwind_gradient(&speed, &[vec![1, 1]], &[], &[], &with_gradient())
            .unwrap();

        assert_eq!(out.arrival_time.pixel_id(), PixelId::Float32);
        assert_eq!(out.arrival_time.spacing(), speed.spacing());
        assert_eq!(out.arrival_time.origin(), speed.origin());
        for axis in &out.gradient {
            assert_eq!(axis.pixel_id(), PixelId::Float32);
            assert_eq!(axis.spacing(), speed.spacing());
            assert_eq!(axis.origin(), speed.origin());
        }
    }
}
