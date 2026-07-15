//! Every expectation here is hand-derived from
//! itkSymmetricForcesDemonsRegistrationFunction.hxx:92-289.
//!
//! The images are `7 x 5` with unit spacing, identity direction and zero origin,
//! so the normalizer `K = (1² + 1²)/2 = 1`. The metric's two-pixel border leaves
//! exactly three pixels — `(2,2)`, `(3,2)`, `(4,2)` — while the squared change is
//! summed over all `35`.

use super::*;
use crate::core::Image;
use crate::filters::demons::{DemonsParams, demons_registration};

const NX: usize = 7;
const NY: usize = 5;
/// `index[j] >= 2 && index[j] <= size[j] - 3` on both axes.
const METRIC_PIXELS: f64 = 3.0;

/// A `7 x 5` `Float64` image whose value at `(x, y)` is `f(x, y)`.
fn grid(f: impl Fn(usize, usize) -> f64) -> Image {
    let mut data = Vec::with_capacity(NX * NY);
    for y in 0..NY {
        for x in 0..NX {
            data.push(f(x, y));
        }
    }
    Image::from_vec(&[NX, NY], data).unwrap()
}

/// One iteration, no smoothing — the raw update field.
fn bare() -> SymmetricForcesDemonsParams {
    SymmetricForcesDemonsParams {
        number_of_iterations: 1,
        smooth_displacement_field: false,
        ..Default::default()
    }
}

/// A `VectorFloat64` field with the same constant vector at every pixel.
fn constant_field(vector: [f64; 2]) -> Image {
    let data: Vec<f64> = (0..NX * NY).flat_map(|_| vector).collect();
    Image::from_vec_vector(&[NX, NY], 2, data).unwrap()
}

fn components(image: &Image, component: usize) -> Vec<f64> {
    let dim = image.number_of_components_per_pixel();
    image
        .component_slice::<f64>()
        .unwrap()
        .iter()
        .skip(component)
        .step_by(dim)
        .copied()
        .collect()
}

/// The same seven `x`-components on each of the five rows.
fn every_row(values: [f64; NX]) -> Vec<f64> {
    values.iter().copied().cycle().take(NX * NY).collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-12, "at {i}: {g} vs {w}");
    }
}

#[test]
fn identical_images_produce_a_zero_field_and_halt_after_one_iteration() {
    let image = grid(|x, _| x as f64);
    let params = SymmetricForcesDemonsParams {
        smooth_displacement_field: false,
        ..Default::default()
    };
    let result = symmetric_forces_demons_registration(&image, &image, None, &params).unwrap();

    assert_eq!(result.elapsed_iterations, 1);
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.rms_change, 0.0);
    assert_close(&components(&result.displacement_field, 0), &[0.0; NX * NY]);
}

/// `fixed = x`, `moving = 0`. The moving gradient vanishes, so the force is
/// `2·speed·∇f / (speed² + |∇f|²)` with `∇f = [0, 1, 1, 1, 1, 1, 0]`:
///
/// | x | speed | denominator | update |
/// |---|---|---|---|
/// | 0 | 0 | — | 0 (below the intensity threshold) |
/// | 1 | 1 | 1 + 1 = 2 | 2/2 = 1 |
/// | 2 | 2 | 4 + 1 = 5 | 4/5 |
/// | 3 | 3 | 9 + 1 = 10 | 6/10 |
/// | 4 | 4 | 16 + 1 = 17 | 8/17 |
/// | 5 | 5 | 25 + 1 = 26 | 10/26 |
/// | 6 | 6 | 36 + 0 = 36 | 0 (the gradient is zero) |
#[test]
fn a_ramp_against_a_constant_gives_the_hand_computed_symmetric_force() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let result = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    let expected = [0.0, 1.0, 0.8, 0.6, 8.0 / 17.0, 5.0 / 13.0, 0.0];
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row(expected),
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; NX * NY]);

    // The moving image is zero everywhere, so the post-update sample is zero too
    // and the metric reduces to the mean of `fixed²` over x = 2, 3, 4.
    assert!(
        (result.metric - (4.0 + 9.0 + 16.0) / METRIC_PIXELS).abs() < 1e-12,
        "{}",
        result.metric
    );
}

/// `m_SumOfSquaredChange` runs over all 35 pixels while
/// `m_NumberOfPixelsProcessed` counts only the 3 that survive the metric's
/// two-pixel border, and `ReleaseGlobalDataPointer` divides the one by the
/// other. The RMS "change" is therefore not an RMS of anything.
#[test]
fn the_rms_change_divides_the_whole_image_sum_by_the_interior_pixel_count() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let result = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    let per_row: f64 = [0.0, 1.0, 0.8, 0.6, 8.0 / 17.0, 5.0 / 13.0, 0.0]
        .iter()
        .map(|u| u * u)
        .sum();
    let expected = (per_row * NY as f64 / METRIC_PIXELS).sqrt();
    assert!(
        (result.rms_change - expected).abs() < 1e-12,
        "{} vs {expected}",
        result.rms_change
    );
    // It even exceeds the largest update, which no true RMS could.
    assert!(result.rms_change > 1.0);
}

/// With the moving gradient gone, this force is exactly twice
/// `DemonsRegistrationFunction`'s on the same inputs — same gradient, same
/// normalizer, same denominator, and a leading `2`.
#[test]
fn the_force_is_twice_the_plain_demons_force_when_the_moving_gradient_vanishes() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);

    let symmetric = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let plain = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            number_of_iterations: 1,
            smooth_displacement_field: false,
            ..Default::default()
        },
    )
    .unwrap();

    let doubled: Vec<f64> = components(&plain.displacement_field, 0)
        .iter()
        .map(|u| 2.0 * u)
        .collect();
    assert_close(&components(&symmetric.displacement_field, 0), &doubled);
}

/// `fixed = x`, `moving = x`, initial field `(-1, 0)`.
///
/// The centre maps to `x - 1`, so `speed = 1` wherever that is in the buffer.
/// The moving gradient is a central difference of the moving image at the
/// *neighbours'* mapped points: at `x = 1` the backward neighbour maps to `-1`,
/// outside the buffer, and is **not subtracted** — leaving `0.5 · m(1) = 0.5`
/// rather than the one-sided `m(1) - m(0) = 1`.
///
/// | x | ∇f | ∇(m∘φ) | denominator | update |
/// |---|---|---|---|---|
/// | 0 | 0 | 0 | — | 0 (speed is 0: the centre maps outside) |
/// | 1 | 1 | 0.5 | 1 + 2.25 = 3.25 | 2·1.5/3.25 = 12/13 |
/// | 2..5 | 1 | 1 | 1 + 4 = 5 | 2·2/5 = 0.8 |
/// | 6 | 0 | 0 | 1 + 0 = 1 | 0 |
fn shifted_setup() -> DemonsResult {
    let image = grid(|x, _| x as f64);
    symmetric_forces_demons_registration(
        &image,
        &image,
        Some(&constant_field([-1.0, 0.0])),
        &bare(),
    )
    .unwrap()
}

#[test]
fn an_out_of_buffer_backward_neighbour_is_not_subtracted() {
    let result = shifted_setup();
    // The field started at -1, so the update is the difference.
    let updates: Vec<f64> = components(&result.displacement_field, 0)
        .iter()
        .map(|d| d + 1.0)
        .collect();
    assert_close(
        &updates,
        &every_row([0.0, 12.0 / 13.0, 0.8, 0.8, 0.8, 0.8, 0.0]),
    );
}

/// The metric samples the moving image at `mappedCenterPoint + update`, not at
/// `mappedCenterPoint`. Each of the three interior pixels lands at `x - 0.2`
/// after its `0.8` update, leaving a residual of `0.2` — where the pre-update
/// residual was `1.0`.
#[test]
fn the_metric_samples_the_moving_image_after_the_update() {
    let result = shifted_setup();
    assert!((result.metric - 0.04).abs() < 1e-12, "{}", result.metric);
}

#[test]
fn the_rms_change_on_the_shifted_pair_sums_over_every_pixel() {
    let result = shifted_setup();
    let per_row: f64 = [0.0, 12.0 / 13.0, 0.8, 0.8, 0.8, 0.8, 0.0]
        .iter()
        .map(|u| u * u)
        .sum();
    let expected = (per_row * NY as f64 / METRIC_PIXELS).sqrt();
    assert!(
        (result.rms_change - expected).abs() < 1e-12,
        "{}",
        result.rms_change
    );
}

/// A centre that maps outside the moving image is *not* skipped: it takes
/// `movingValue = 0`, produces an update, and contributes `fixedValue²` to the
/// metric. `DemonsRegistrationFunction` and `ESMDemonsRegistrationFunction` both
/// return early there, leaving the metric at `f64::MAX`.
#[test]
fn a_centre_mapped_outside_takes_a_zero_moving_value_rather_than_being_skipped() {
    let image = grid(|_, _| 5.0);
    let result = symmetric_forces_demons_registration(
        &image,
        &image,
        Some(&constant_field([100.0, 0.0])),
        &bare(),
    )
    .unwrap();

    // Both images are constant, so every gradient is zero and no update moves.
    assert_close(
        &components(&result.displacement_field, 0),
        &[100.0; NX * NY],
    );
    assert_eq!(result.rms_change, 0.0);
    // 5² over the three interior pixels — not `f64::MAX`.
    assert!((result.metric - 25.0).abs() < 1e-12, "{}", result.metric);
}

/// The metric drops two pixels from each end of every axis, so nothing at all
/// survives on an image whose axes are shorter than five. `m_Metric` and
/// `m_RMSChange` then keep their constructor value.
#[test]
fn an_image_thinner_than_five_pixels_has_no_metric_pixels() {
    let params = SymmetricForcesDemonsParams {
        number_of_iterations: 1,
        smooth_displacement_field: false,
        ..Default::default()
    };

    let small = Image::from_vec(&[4, 4], (0..16).map(|i| f64::from(i % 4)).collect()).unwrap();
    let starved = symmetric_forces_demons_registration(&small, &small, None, &params).unwrap();
    assert_eq!(starved.metric, f64::MAX);
    assert_eq!(starved.rms_change, f64::MAX);

    let big = Image::from_vec(&[5, 5], (0..25).map(|i| f64::from(i % 5)).collect()).unwrap();
    let fed = symmetric_forces_demons_registration(&big, &big, None, &params).unwrap();
    assert_eq!(fed.metric, 0.0);
    assert_eq!(fed.rms_change, 0.0);
}

/// `GetRMSChange()` is overridden to return the difference function's
/// `NumericTraits<double>::max()`, where `demons_registration` reports the
/// filter's `0.0`.
#[test]
fn a_zero_iteration_run_reports_the_functions_initial_rms_change() {
    let image = grid(|x, _| x as f64);
    let result = symmetric_forces_demons_registration(
        &image,
        &image,
        None,
        &SymmetricForcesDemonsParams {
            number_of_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.elapsed_iterations, 0);
    assert_eq!(result.rms_change, f64::MAX);
    assert_eq!(result.metric, f64::MAX);
}

/// `InitializeIteration` smooths the field *before* the update is computed from
/// it, so a one-iteration run from a zero field is unaffected — the opposite of
/// `FastSymmetricForcesDemonsRegistrationFilter`, which smooths afterwards.
#[test]
fn smoothing_the_displacement_field_happens_before_the_update_is_computed() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);

    let unsmoothed = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let smoothed = symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &SymmetricForcesDemonsParams {
            number_of_iterations: 1,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(
        unsmoothed.displacement_field, smoothed.displacement_field,
        "smoothing a zero field must be a no-op"
    );
}

#[test]
fn smoothing_the_update_field_changes_the_result() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);

    let plain = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let viscous = symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &SymmetricForcesDemonsParams {
            smooth_update_field: true,
            ..bare()
        },
    )
    .unwrap();

    assert_ne!(plain.displacement_field, viscous.displacement_field);
    let peak = |image: &Image| {
        components(image, 0)
            .into_iter()
            .fold(f64::MIN, |a, b| a.max(b))
    };
    assert!(peak(&viscous.displacement_field) < peak(&plain.displacement_field));
}

/// Scaling both images by `1e-9` leaves the force scale-invariant, but the
/// denominator falls below the hard-coded `m_DenominatorThreshold`.
#[test]
fn the_denominator_threshold_zeroes_a_scale_invariant_update() {
    let moving = grid(|_, _| 0.0);
    let params = SymmetricForcesDemonsParams {
        intensity_difference_threshold: 0.0,
        ..bare()
    };

    let full = grid(|x, _| x as f64);
    let unscaled = symmetric_forces_demons_registration(&full, &moving, None, &params).unwrap();
    assert!((components(&unscaled.displacement_field, 0)[1] - 1.0).abs() < 1e-12);

    // denominator = (1e-9)²/1 + (1e-9)² = 2e-18 < 1e-9.
    let tiny = grid(|x, _| x as f64 * 1e-9);
    let scaled = symmetric_forces_demons_registration(&tiny, &moving, None, &params).unwrap();
    assert_close(&components(&scaled.displacement_field, 0), &[0.0; NX * NY]);
}

#[test]
fn the_intensity_difference_threshold_zeroes_a_matched_pixel() {
    let moving = grid(|_, _| 0.0);
    let fixed = grid(|x, _| x as f64);
    let params = SymmetricForcesDemonsParams {
        intensity_difference_threshold: 3.5,
        ..bare()
    };
    let result = symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    // Only |speed| = 4, 5 and 6 clear the threshold; x = 6 has a zero gradient.
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 0.0, 0.0, 0.0, 8.0 / 17.0, 5.0 / 13.0, 0.0]),
    );
}

#[test]
fn use_image_spacing_is_inert() {
    let mut fixed = grid(|x, _| x as f64);
    let mut moving = grid(|_, _| 0.0);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    moving.set_spacing(&[2.0, 3.0]).unwrap();

    let on = symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &SymmetricForcesDemonsParams {
            use_image_spacing: true,
            ..bare()
        },
    )
    .unwrap();
    let off = symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &SymmetricForcesDemonsParams {
            use_image_spacing: false,
            ..bare()
        },
    )
    .unwrap();
    assert_eq!(on.displacement_field, off.displacement_field);
}

#[test]
fn iterating_reduces_the_metric() {
    let fixed = Image::from_vec(
        &[8, 8],
        (0..64)
            .map(|i| {
                let (x, y) = (i % 8, i / 8);
                f64::from((2..5).contains(&x) && (2..5).contains(&y))
            })
            .collect(),
    )
    .unwrap();
    let moving = Image::from_vec(
        &[8, 8],
        (0..64)
            .map(|i| {
                let (x, y) = (i % 8, i / 8);
                f64::from((3..6).contains(&x) && (2..5).contains(&y))
            })
            .collect(),
    )
    .unwrap();

    let one = symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &SymmetricForcesDemonsParams {
            number_of_iterations: 1,
            maximum_rms_error: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let ten = symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &SymmetricForcesDemonsParams {
            number_of_iterations: 10,
            maximum_rms_error: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(ten.elapsed_iterations, 10);
    assert!(ten.metric < one.metric, "{} vs {}", ten.metric, one.metric);
}

#[test]
fn a_three_dimensional_run_produces_a_three_component_field() {
    let data: Vec<f64> = (0..125).map(|i| f64::from(i as u32 % 5)).collect();
    let fixed = Image::from_vec(&[5, 5, 5], data).unwrap();
    let moving = Image::from_vec(&[5, 5, 5], vec![0.0; 125]).unwrap();
    let result = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert_eq!(
        result.displacement_field.number_of_components_per_pixel(),
        3
    );
    assert_eq!(result.displacement_field.size(), &[5, 5, 5]);
    // The 5³ image has exactly one interior pixel, at (2, 2, 2).
    assert!(result.metric.is_finite());
}

#[test]
fn integer_inputs_are_accepted() {
    let fixed = Image::from_vec(&[NX, NY], (0..NX * NY).map(|i| (i % NX) as u8).collect()).unwrap();
    let moving = Image::from_vec(&[NX, NY], vec![0u8; NX * NY]).unwrap();
    let result = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 1.0, 0.8, 0.6, 8.0 / 17.0, 5.0 / 13.0, 0.0]),
    );
}

#[test]
fn the_output_takes_the_fixed_images_geometry() {
    let mut fixed = grid(|x, _| x as f64);
    let mut moving = grid(|_, _| 0.0);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    fixed.set_origin(&[10.0, 20.0]).unwrap();
    moving.set_spacing(&[2.0, 3.0]).unwrap();
    moving.set_origin(&[10.0, 20.0]).unwrap();

    let result = symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    assert_eq!(result.displacement_field.spacing(), &[2.0, 3.0]);
    assert_eq!(result.displacement_field.origin(), &[10.0, 20.0]);
}

#[test]
fn the_output_takes_the_initial_fields_geometry_when_one_is_given() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let mut initial = constant_field([0.0, 0.0]);
    initial.set_origin(&[7.0, 8.0]).unwrap();

    let result =
        symmetric_forces_demons_registration(&fixed, &moving, Some(&initial), &bare()).unwrap();
    assert_eq!(result.displacement_field.origin(), &[7.0, 8.0]);
}

#[test]
fn a_vector_fixed_image_is_rejected() {
    let image = Image::from_vec_vector(&[NX, NY], 2, vec![0.0; NX * NY * 2]).unwrap();
    assert!(symmetric_forces_demons_registration(&image, &image, None, &bare()).is_err());
}

#[test]
fn mismatched_pixel_types_are_rejected() {
    let fixed = grid(|_, _| 0.0);
    let moving = Image::from_vec(&[NX, NY], vec![0.0f32; NX * NY]).unwrap();
    assert!(symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).is_err());
}

#[test]
fn a_maximum_error_outside_the_open_unit_interval_is_rejected() {
    let image = grid(|_, _| 0.0);
    for maximum_error in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let params = SymmetricForcesDemonsParams {
            maximum_error,
            ..bare()
        };
        assert!(
            symmetric_forces_demons_registration(&image, &image, None, &params).is_err(),
            "accepted maximum_error {maximum_error}"
        );
    }
}

#[test]
fn standard_deviations_shorter_than_the_image_dimension_are_rejected() {
    let image = grid(|_, _| 0.0);
    let short = SymmetricForcesDemonsParams {
        standard_deviations: vec![1.0],
        ..bare()
    };
    assert!(symmetric_forces_demons_registration(&image, &image, None, &short).is_err());

    let short_update = SymmetricForcesDemonsParams {
        update_field_standard_deviations: vec![1.0],
        ..bare()
    };
    assert!(symmetric_forces_demons_registration(&image, &image, None, &short_update).is_err());
}

#[test]
fn an_initial_field_of_the_wrong_size_is_rejected() {
    let image = grid(|_, _| 0.0);
    let initial = Image::from_vec_vector(&[2, 2], 2, vec![0.0; 8]).unwrap();
    assert!(symmetric_forces_demons_registration(&image, &image, Some(&initial), &bare()).is_err());
}
