//! Every expectation here is hand-derived from
//! itkLevelSetMotionRegistrationFilter.hxx:25-89, 218-242 and
//! itkLevelSetMotionRegistrationFunction.hxx:196-384.
//!
//! The images are `5 x 5` — the recursive Gaussian that smooths the moving
//! image needs at least four pixels on every axis it filters — with unit
//! spacing, identity direction and zero origin.
//!
//! # The workhorse image
//!
//! `moving(x, y) = |x - 2|`, a symmetric V that is constant along `y`.
//! Smoothing is zero-phase and separable, so its output `S` is still symmetric
//! about `x = 2` and still constant along `y`. Therefore, with a zero field:
//!
//! | x | forward | backward | minmod |
//! |---|---|---|---|
//! | 0 | `S(1) - S(0) < 0` | outside the buffer, so `0` | `0` |
//! | 1 | `S(2) - S(1) < 0` | `S(1) - S(0) < 0` | `-g`, some `g > 0` |
//! | 2 | `S(3) - S(2) > 0` | `S(2) - S(1) < 0` | `0` (product `< 0`) |
//! | 3 | `S(4) - S(3) > 0` | `S(3) - S(2) > 0` | `+g` by symmetry |
//! | 4 | outside the buffer, so `0` | `S(4) - S(3) > 0` | `0` |
//!
//! and the `y` gradient vanishes everywhere. So only `x = 1` and `x = 3` move,
//! antisymmetrically, and their update is
//! `speed · (∓g) / (g + alpha)`.
//!
//! `fixed = moving + 1` makes `speed = 1` at every pixel exactly (the field is
//! zero, so the interpolator samples pixel centres). Hence:
//!
//! * `metric = SSD/N = 25/25 = 1`,
//! * `|update| = g / (g + alpha)` at ten pixels and `0` at the other fifteen,
//! * `max_L1 = |update|` (unit spacing, one nonzero component),
//! * `dt = 1 / |update|`, so the **applied** displacement is exactly `∓1` —
//!   independent of `alpha` and of the unknown `g`,
//! * `rms_change = sqrt(10 · |update|² / 25) = |update| · sqrt(0.4)`, measured
//!   on the *unscaled* update.

use super::*;
use sitk_core::Image;

const N: usize = 5;

/// A `5 x 5` `Float64` image whose value at `(x, y)` is `f(x, y)`.
fn grid(f: impl Fn(usize, usize) -> f64) -> Image {
    let mut data = Vec::with_capacity(N * N);
    for y in 0..N {
        for x in 0..N {
            data.push(f(x, y));
        }
    }
    Image::from_vec(&[N, N], data).unwrap()
}

/// `moving(x, y) = |x - 2|`.
fn valley() -> Image {
    grid(|x, _| (x as f64 - 2.0).abs())
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

/// The same five `x`-components on each of the five rows.
fn every_row(values: [f64; N]) -> Vec<f64> {
    values.iter().copied().cycle().take(N * N).collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-9, "at {i}: {g} vs {w}");
    }
}

/// One iteration of the default filter.
fn once() -> LevelSetMotionParams {
    LevelSetMotionParams {
        number_of_iterations: 1,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Defaults and the public surface
// ---------------------------------------------------------------------------

/// itkLevelSetMotionRegistrationFilter.hxx:31-34 turns off the base's
/// displacement-field smoother: "no regularization of the deformation field is
/// performed in LevelSetMotionRegistration".
#[test]
fn neither_smoother_is_on_by_default() {
    let params = LevelSetMotionParams::default();
    assert!(!params.smooth_displacement_field);
    assert!(!params.smooth_update_field);
}

#[test]
fn the_yaml_defaults_are_the_struct_defaults() {
    let params = LevelSetMotionParams::default();
    assert_eq!(params.gradient_smoothing_standard_deviations, 1.0);
    assert_eq!(params.number_of_iterations, 10);
    assert_eq!(params.maximum_rms_error, 0.02);
    assert_eq!(params.standard_deviations, vec![1.0; 3]);
    assert_eq!(params.update_field_standard_deviations, vec![1.0; 3]);
    assert_eq!(params.maximum_kernel_width, 30);
    assert_eq!(params.maximum_error, 0.1);
    assert_eq!(params.alpha, 0.1);
    assert_eq!(params.intensity_difference_threshold, 0.001);
    assert_eq!(params.gradient_magnitude_threshold, 1e-9);
    assert!(params.use_image_spacing);
}

// ---------------------------------------------------------------------------
// minmod
// ---------------------------------------------------------------------------

#[test]
fn minmod_takes_the_smaller_magnitude_when_the_signs_agree() {
    assert_eq!(minmod(2.0, 3.0), 2.0);
    assert_eq!(minmod(3.0, 2.0), 2.0);
    assert_eq!(minmod(-2.0, -3.0), -2.0);
    assert_eq!(minmod(-3.0, -2.0), -2.0);
}

#[test]
fn minmod_is_zero_when_the_signs_disagree_or_either_vanishes() {
    assert_eq!(minmod(2.0, -3.0), 0.0);
    assert_eq!(minmod(-2.0, 3.0), 0.0);
    assert_eq!(minmod(0.0, 3.0), 0.0);
    assert_eq!(minmod(3.0, 0.0), 0.0);
    assert_eq!(minmod(0.0, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// The workhorse image
// ---------------------------------------------------------------------------

/// `dt = 1 / max_L1` normalises the largest update to one pixel in L1, so the
/// applied displacement at `x = 1` and `x = 3` is exactly `∓1` however large
/// the smoothed gradient turns out to be.
#[test]
fn the_largest_applied_displacement_is_exactly_one_pixel_in_l1() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -1.0, 0.0, 1.0, 0.0]),
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; N * N]);
    assert_eq!(result.elapsed_iterations, 1);
}

/// `speed = 1` at all 25 pixels, none of which maps outside the buffer.
#[test]
fn the_metric_is_the_mean_squared_intensity_difference() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    assert!((result.metric - 1.0).abs() < 1e-12, "{}", result.metric);
}

/// `alpha` shrinks the raw update, and `dt` divides it straight back out. The
/// applied field is therefore identical, while the RMS change — accumulated
/// from the raw update at hxx:325-334, before `dt` ever touches it — is
/// strictly smaller. Setting `alpha = 0` makes `|update| = 1` exactly, so
/// `rms_change = sqrt(10/25)`.
#[test]
fn alpha_cancels_out_of_the_applied_field_but_not_out_of_the_rms_change() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let sharp = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            alpha: 0.0,
            ..once()
        },
    )
    .unwrap();
    let blunt = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();

    assert_close(
        &components(&sharp.displacement_field, 0),
        &components(&blunt.displacement_field, 0),
    );
    assert!(
        (sharp.rms_change - 0.4_f64.sqrt()).abs() < 1e-9,
        "{}",
        sharp.rms_change
    );
    assert!(blunt.rms_change < sharp.rms_change);
    // update = g/(g + 0.1) < 1, and rms scales with it.
    assert!(blunt.rms_change > 0.0);
}

/// A one-sided difference whose sample leaves the buffer is `0`, and then
/// `forward · backward > 0` cannot hold, so the border slice has no derivative
/// along the border axis. Here that zeroes `x = 0` and `x = 4` — and the whole
/// `y` axis is flat, so *no* pixel gets a `y` component at all.
#[test]
fn a_border_pixel_has_no_derivative_along_the_border_axis() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    let x = components(&result.displacement_field, 0);

    for row in 0..N {
        assert_eq!(x[row * N], 0.0);
        assert_eq!(x[row * N + 4], 0.0);
        // x = 2 sits at the bottom of the V: the two differences disagree in
        // sign, so minmod is zero there too.
        assert_eq!(x[row * N + 2], 0.0);
    }
}

// ---------------------------------------------------------------------------
// The guards, the metric, and the fallback time step
// ---------------------------------------------------------------------------

/// The gradient of a constant image is below `gradient_magnitude_threshold`
/// everywhere, so every update is zero — but `m_SumOfSquaredDifference` and
/// `m_NumberOfPixelsProcessed` were already incremented (hxx:311-316), so the
/// metric still reports `speed² = 4`. `m_MaxL1Norm` keeps its negative initial
/// value, so `dt` falls back to `1.0`, and the zero RMS change halts the filter
/// after one iteration.
#[test]
fn a_thresholded_pixel_still_counts_toward_the_metric() {
    let moving = grid(|_, _| 3.0);
    let fixed = grid(|_, _| 5.0);

    let result =
        level_set_motion_registration(&fixed, &moving, None, &LevelSetMotionParams::default())
            .unwrap();

    assert_eq!(components(&result.displacement_field, 0), vec![0.0; N * N]);
    assert_eq!(components(&result.displacement_field, 1), vec![0.0; N * N]);
    assert!((result.metric - 4.0).abs() < 1e-12, "{}", result.metric);
    assert_eq!(result.rms_change, 0.0);
    assert_eq!(result.elapsed_iterations, 1);
}

/// `|speed| < intensity_difference_threshold` also returns a zero update after
/// the metric has counted the pixel.
#[test]
fn identical_images_produce_no_update_and_a_zero_metric() {
    let moving = valley();
    let result =
        level_set_motion_registration(&moving, &moving, None, &LevelSetMotionParams::default())
            .unwrap();

    assert_eq!(components(&result.displacement_field, 0), vec![0.0; N * N]);
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.rms_change, 0.0);
}

/// The default `intensity_difference_threshold` of `1e-3` swallows a `5e-4`
/// difference; lowering the threshold lets it through.
#[test]
fn the_intensity_difference_threshold_zeroes_a_matched_pixel() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 5e-4);

    let hushed = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    assert_eq!(components(&hushed.displacement_field, 0), vec![0.0; N * N]);
    assert!((hushed.metric - 2.5e-7).abs() < 1e-18, "{}", hushed.metric);

    let heard = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            intensity_difference_threshold: 1e-6,
            ..once()
        },
    )
    .unwrap();
    // Still normalised to one pixel by dt, however faint the speed.
    assert_close(
        &components(&heard.displacement_field, 0),
        &every_row([0.0, -1.0, 0.0, 1.0, 0.0]),
    );
}

/// Raising `gradient_magnitude_threshold` above the smoothed gradient zeroes
/// every update while leaving the metric intact.
#[test]
fn the_gradient_magnitude_threshold_zeroes_every_update() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            gradient_magnitude_threshold: 1e9,
            ..once()
        },
    )
    .unwrap();

    assert_eq!(components(&result.displacement_field, 0), vec![0.0; N * N]);
    assert!((result.metric - 1.0).abs() < 1e-12);
    assert_eq!(result.rms_change, 0.0);
}

/// A pixel whose mapped point leaves the moving image's buffer is skipped
/// before the metric counts it. Pushing the whole field far to the left leaves
/// no pixel inside, so `m_NumberOfPixelsProcessed == 0` and
/// `ReleaseGlobalDataPointer` leaves the metric at its initial `f64::MAX`.
#[test]
fn a_pixel_mapping_outside_the_moving_image_counts_for_nothing() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);
    let field = Image::from_vec_vector(&[N, N], 2, [-100.0, 0.0].repeat(N * N)).unwrap();

    let result = level_set_motion_registration(&fixed, &moving, Some(&field), &once()).unwrap();

    assert_eq!(result.metric, f64::MAX);
    // No update anywhere, and dt falls back to 1.0, so the field is untouched.
    assert_eq!(
        components(&result.displacement_field, 0),
        vec![-100.0; N * N]
    );
}

// ---------------------------------------------------------------------------
// Halting
// ---------------------------------------------------------------------------

/// `Halt()` (hxx:76-89) adds "an RMS change of exactly zero stops the filter",
/// which fires even when the base rule's strict `maximum_rms_error > rms_change`
/// never can.
#[test]
fn a_zero_rms_change_halts_even_at_a_zero_maximum_rms_error() {
    let moving = valley();
    let result = level_set_motion_registration(
        &moving,
        &moving,
        None,
        &LevelSetMotionParams {
            number_of_iterations: 10,
            maximum_rms_error: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(result.elapsed_iterations, 1);
}

/// `GetRMSChange()` is not overridden, so a zero-iteration run reports the
/// *filter's* `0.0`, while `GetMetric()` forwards to the function, which has
/// computed nothing.
#[test]
fn a_zero_iteration_run_reports_the_filters_rms_change_and_the_functions_metric() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            number_of_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(result.elapsed_iterations, 0);
    assert_eq!(result.rms_change, 0.0);
    assert_eq!(result.metric, f64::MAX);
    assert_eq!(components(&result.displacement_field, 0), vec![0.0; N * N]);
}

/// The base rule still applies: a large `maximum_rms_error` stops after the
/// first iteration.
#[test]
fn a_large_maximum_rms_error_stops_after_one_iteration() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            number_of_iterations: 10,
            maximum_rms_error: 100.0,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(result.elapsed_iterations, 1);
}

/// Two iterations converge exactly, and the extra `Halt` clause stops the
/// filter even though `maximum_rms_error` is `0.0`.
///
/// After the first iteration the field is `[0, -1, 0, 1, 0]`, so the second
/// iteration samples `moving = [2, 1, 0, 1, 2]` at `[0, 0, 2, 4, 4]`, giving
/// `moving_at_mapped = [2, 2, 0, 2, 2]` against `fixed = [3, 2, 1, 2, 3]` and
/// `speed = [1, 0, 1, 0, 1]`. The three pixels with a speed have no gradient —
/// `x = 0` and `x = 4` are borders, `x = 2` is the bottom of the V — and the
/// two with a gradient have no speed. Every update is zero, so
/// `rms_change = 0`, `dt` falls back to `1.0`, the field is untouched, and
/// `metric = 15/25 = 0.6`.
#[test]
fn the_registration_converges_to_a_zero_rms_change_after_two_iterations() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            number_of_iterations: 10,
            maximum_rms_error: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(result.elapsed_iterations, 2);
    assert_eq!(result.rms_change, 0.0);
    assert!((result.metric - 0.6).abs() < 1e-12, "{}", result.metric);
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -1.0, 0.0, 1.0, 0.0]),
    );
}

// ---------------------------------------------------------------------------
// use_image_spacing is live here
// ---------------------------------------------------------------------------

/// Every other filter in this family ignores `UseImageSpacing` under a unit
/// spacing. This one uses the *moving* image's spacing twice — as the physical
/// step of the one-sided differences (hxx:252) and as the divisor of the L1
/// norm (hxx:336) — so under a non-unit spacing the two settings disagree.
#[test]
fn use_image_spacing_changes_the_result() {
    let mut moving = valley();
    let mut fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);
    moving.set_spacing(&[2.0, 1.0]).unwrap();
    fixed.set_spacing(&[2.0, 1.0]).unwrap();

    let with = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    let without = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            use_image_spacing: false,
            ..once()
        },
    )
    .unwrap();

    assert_ne!(
        components(&with.displacement_field, 0),
        components(&without.displacement_field, 0)
    );
    // With spacing on, L1 = |u|/2, so dt = 2/|u| and the applied step is 2 mm —
    // still exactly one pixel.
    assert_close(
        &components(&with.displacement_field, 0),
        &every_row([0.0, -2.0, 0.0, 2.0, 0.0]),
    );
    // With spacing off, the L1 norm divides by 1 and the step is 1 mm.
    assert_close(
        &components(&without.displacement_field, 0),
        &every_row([0.0, -1.0, 0.0, 1.0, 0.0]),
    );
}

// ---------------------------------------------------------------------------
// Smoothing
// ---------------------------------------------------------------------------

/// `InitializeIteration` smooths the field *before* the update is computed
/// (hxx:64-73), so a single iteration starting from a zero field cannot tell
/// the smoother apart from nothing — the zero field smooths to itself.
/// Starting from a rough field, it can.
#[test]
fn the_displacement_field_is_smoothed_before_the_update_is_computed() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);
    let mut data = vec![0.0; 2 * N * N];
    data[2 * (2 * N + 2)] = 3.0; // a spike at (2, 2)

    let field = Image::from_vec_vector(&[N, N], 2, data).unwrap();
    let smoothed = level_set_motion_registration(
        &fixed,
        &moving,
        Some(&field),
        &LevelSetMotionParams {
            smooth_displacement_field: true,
            ..once()
        },
    )
    .unwrap();
    let raw = level_set_motion_registration(&fixed, &moving, Some(&field), &once()).unwrap();

    let smoothed_x = components(&smoothed.displacement_field, 0);
    let raw_x = components(&raw.displacement_field, 0);
    assert_ne!(smoothed_x, raw_x);
    // The spike survives undiminished when nothing smooths it, minus whatever
    // update lands on (2, 2) — which is zero, at the bottom of the V.
    assert_eq!(raw_x[2 * N + 2], 3.0);
    assert!(smoothed_x[2 * N + 2] < 3.0);
}

/// `ApplyUpdate` smooths the update field before the superclass applies it
/// (hxx:224-232). The smoother spreads the antisymmetric `±1` pair, so the
/// zeroed pixels at `x = 0, 2, 4` pick up a displacement.
#[test]
fn smoothing_the_update_field_spreads_the_displacement() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            smooth_update_field: true,
            ..once()
        },
    )
    .unwrap();

    let x = components(&result.displacement_field, 0);
    assert!(x[2 * N].abs() > 0.0);
    assert!(x[2 * N + 2].abs() < 1e-12); // still zero by antisymmetry
    assert!(x[2 * N + 1] < 0.0);
    assert!(x[2 * N + 3] > 0.0);
    // Smoothing cannot amplify, and the RMS change was measured before it.
    assert!(x[2 * N + 3] < 1.0);
}

/// `smooth_field` only reads `standard_deviations` when the corresponding flag
/// is set, so a short vector is only an error for the smoother that runs.
#[test]
fn the_smoothers_standard_deviations_must_cover_every_axis() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    // per_axis validates both vectors up front, whichever flag is set.
    assert!(
        level_set_motion_registration(
            &fixed,
            &moving,
            None,
            &LevelSetMotionParams {
                standard_deviations: vec![1.0],
                ..once()
            },
        )
        .is_err()
    );
    assert!(
        level_set_motion_registration(
            &fixed,
            &moving,
            None,
            &LevelSetMotionParams {
                update_field_standard_deviations: vec![1.0],
                ..once()
            },
        )
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// Sigma, pixel types, dimensions, errors
// ---------------------------------------------------------------------------

/// A larger smoothing sigma flattens the V, changing the gradient's magnitude —
/// though not, thanks to `dt`, the applied displacement's magnitude. What it
/// does change is the RMS change, which sees the raw update.
#[test]
fn a_larger_smoothing_sigma_shrinks_the_raw_update() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let tight = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    let loose = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            gradient_smoothing_standard_deviations: 2.0,
            ..once()
        },
    )
    .unwrap();

    // g/(g + alpha) grows with g, and heavier smoothing gives a smaller g.
    assert!(loose.rms_change < tight.rms_change);
    // Both still normalise to one pixel.
    assert_close(
        &components(&tight.displacement_field, 0),
        &components(&loose.displacement_field, 0),
    );
}

/// `sigma == 0` leaves an axis untouched in `recursive_gaussian`; the gradient
/// is then taken on the raw V, whose one-sided differences are `∓1`, so
/// `minmod` gives `∓1` at `x = 1, 3` and the applied field is unchanged.
#[test]
fn a_zero_sigma_takes_the_gradient_of_the_unsmoothed_image() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let result = level_set_motion_registration(
        &fixed,
        &moving,
        None,
        &LevelSetMotionParams {
            gradient_smoothing_standard_deviations: 0.0,
            alpha: 0.0,
            ..once()
        },
    )
    .unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -1.0, 0.0, 1.0, 0.0]),
    );
    // |update| = 1 exactly, so the RMS change is sqrt(10/25).
    assert!((result.rms_change - 0.4_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn a_negative_sigma_is_an_error() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    assert!(
        level_set_motion_registration(
            &fixed,
            &moving,
            None,
            &LevelSetMotionParams {
                gradient_smoothing_standard_deviations: -1.0,
                ..once()
            },
        )
        .is_err()
    );
}

/// `RecursiveSeparableImageFilter` throws when a filtered axis has fewer than
/// four pixels, and the moving image is always filtered on every axis.
#[test]
fn an_axis_shorter_than_four_pixels_is_an_error() {
    let moving = Image::from_vec(&[5, 3], vec![0.0; 15]).unwrap();
    let fixed = Image::from_vec(&[5, 3], vec![1.0; 15]).unwrap();

    let error = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap_err();
    assert!(
        matches!(
            error,
            crate::FilterError::AxisTooShortForRecursion { axis: 1, len: 3 }
        ),
        "{error:?}"
    );
}

/// `SmoothingRecursiveGaussianImageFilter`'s output pixel type is the *moving*
/// image's, so an integer moving image has its smoothed copy quantised before
/// its gradient is taken. The V `50·|x - 2|` stays far enough apart to survive
/// rounding, so the normalised update is still `∓1`.
#[test]
fn an_integer_moving_image_has_its_smoothed_copy_quantised() {
    let profile = |i: usize| ((i % N) as i32 - 2).unsigned_abs() as u8 * 50;
    let moving = Image::from_vec(&[N, N], (0..N * N).map(profile).collect::<Vec<u8>>()).unwrap();
    let fixed = Image::from_vec(
        &[N, N],
        (0..N * N).map(|i| profile(i) + 1).collect::<Vec<u8>>(),
    )
    .unwrap();

    let result = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    assert_eq!(
        result.displacement_field.pixel_id(),
        sitk_core::PixelId::VectorFloat64
    );
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -1.0, 0.0, 1.0, 0.0]),
    );
    assert!((result.metric - 1.0).abs() < 1e-12);
}

/// A 3-D smoke test: the V runs along `x` and the other two axes are flat, so
/// the same `∓1` displacement appears on every `(y, z)` line.
#[test]
fn the_filter_runs_in_three_dimensions() {
    let mut moving = Vec::new();
    let mut fixed = Vec::new();
    for _ in 0..4 {
        for _ in 0..4 {
            for x in 0..5 {
                let v = (x as f64 - 2.0).abs();
                moving.push(v);
                fixed.push(v + 1.0);
            }
        }
    }
    let moving = Image::from_vec(&[5, 4, 4], moving).unwrap();
    let fixed = Image::from_vec(&[5, 4, 4], fixed).unwrap();

    let result = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    assert_eq!(
        result.displacement_field.number_of_components_per_pixel(),
        3
    );
    let x = components(&result.displacement_field, 0);
    assert_close(
        &x,
        &[0.0, -1.0, 0.0, 1.0, 0.0]
            .iter()
            .copied()
            .cycle()
            .take(5 * 16)
            .collect::<Vec<_>>(),
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; 80]);
    assert_close(&components(&result.displacement_field, 2), &[0.0; 80]);
}

/// The output field carries the fixed image's geometry.
#[test]
fn the_output_field_takes_the_fixed_images_geometry() {
    let mut moving = valley();
    let mut fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);
    moving.set_spacing(&[0.5, 0.5]).unwrap();
    fixed.set_spacing(&[0.5, 0.5]).unwrap();
    fixed.set_origin(&[-1.0, 2.0]).unwrap();

    let result = level_set_motion_registration(&fixed, &moving, None, &once()).unwrap();
    assert_eq!(result.displacement_field.spacing(), &[0.5, 0.5]);
    assert_eq!(result.displacement_field.origin(), &[-1.0, 2.0]);
    assert_eq!(result.displacement_field.size(), &[N, N]);
}

#[test]
fn mismatched_pixel_types_and_dimensions_are_errors() {
    let moving = valley();
    let other_type = Image::from_vec(&[N, N], vec![0u8; N * N]).unwrap();
    assert!(level_set_motion_registration(&moving, &other_type, None, &once()).is_err());

    let other_dim = Image::from_vec(&[N, N, N], vec![0.0; N * N * N]).unwrap();
    assert!(level_set_motion_registration(&moving, &other_dim, None, &once()).is_err());
}

#[test]
fn a_vector_input_is_an_error() {
    let moving = valley();
    let vector = Image::from_vec_vector(&[N, N], 2, vec![0.0; 2 * N * N]).unwrap();
    assert!(level_set_motion_registration(&vector, &moving, None, &once()).is_err());
    assert!(level_set_motion_registration(&moving, &vector, None, &once()).is_err());
}

#[test]
fn a_maximum_error_outside_the_open_unit_interval_is_an_error() {
    let moving = valley();
    for maximum_error in [0.0, 1.0, -0.5, 1.5] {
        assert!(
            level_set_motion_registration(
                &moving,
                &moving,
                None,
                &LevelSetMotionParams {
                    maximum_error,
                    ..once()
                },
            )
            .is_err(),
            "{maximum_error}"
        );
    }
}

#[test]
fn a_malformed_initial_field_is_an_error() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);

    let scalar = Image::from_vec(&[N, N], vec![0.0; N * N]).unwrap();
    assert!(level_set_motion_registration(&fixed, &moving, Some(&scalar), &once()).is_err());

    let wrong_components = Image::from_vec_vector(&[N, N], 3, vec![0.0; 3 * N * N]).unwrap();
    assert!(
        level_set_motion_registration(&fixed, &moving, Some(&wrong_components), &once()).is_err()
    );

    let wrong_size = Image::from_vec_vector(&[4, 4], 2, vec![0.0; 32]).unwrap();
    assert!(level_set_motion_registration(&fixed, &moving, Some(&wrong_size), &once()).is_err());
}

/// An initial field is composed additively: the update lands on top of it.
#[test]
fn the_initial_field_is_the_starting_point() {
    let moving = valley();
    let fixed = grid(|x, _| (x as f64 - 2.0).abs() + 1.0);
    // A small constant shift, small enough that no pixel leaves the buffer and
    // the V's shape still drives the same signs.
    let field = Image::from_vec_vector(&[N, N], 2, [0.0, 0.25].repeat(N * N)).unwrap();

    let result = level_set_motion_registration(&fixed, &moving, Some(&field), &once()).unwrap();
    // The y-component is untouched: the smoothed image is flat along y.
    assert_close(&components(&result.displacement_field, 1), &[0.25; N * N]);
}
