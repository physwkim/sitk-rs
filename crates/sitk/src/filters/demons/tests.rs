use super::*;
use crate::core::PixelId;
use crate::filters::FilterError;

/// A 5x1 `Float64` image. `dim == 2`, so the normalizer is `(1² + 1²)/2 == 1`
/// and every y-derivative is a boundary zero (the y axis has extent 1).
fn row(values: &[f64]) -> Image {
    Image::from_vec(&[5, 1], values.to_vec()).unwrap()
}

fn one_iteration() -> DemonsParams {
    DemonsParams {
        number_of_iterations: 1,
        ..DemonsParams::default()
    }
}

/// The x-components of the field, one per pixel.
fn x_components(result: &DemonsResult) -> Vec<f64> {
    result
        .displacement_field
        .component_slice::<f64>()
        .unwrap()
        .chunks_exact(2)
        .map(|v| v[0])
        .collect()
}

fn y_components(result: &DemonsResult) -> Vec<f64> {
    result
        .displacement_field
        .component_slice::<f64>()
        .unwrap()
        .chunks_exact(2)
        .map(|v| v[1])
        .collect()
}

/// Identical images give a zero speed everywhere, so every update is the zero
/// vector: the demons force's numerator vanishes. The RMS change is then `0.0`,
/// which is strictly below the default `maximum_rms_error` of `0.02`, so the
/// solver halts after exactly one iteration of the ten it was allowed.
#[test]
fn identical_images_produce_a_zero_field_and_halt_after_one_iteration() {
    let image = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = demons_registration(&image, &image, None, &DemonsParams::default()).unwrap();

    assert!(
        result
            .displacement_field
            .component_slice::<f64>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert_eq!(result.elapsed_iterations, 1);
    assert_eq!(result.rms_change, 0.0);
    assert_eq!(result.metric, 0.0);
}

/// The output is a `VectorFloat64` field with one component per dimension.
#[test]
fn the_output_is_a_vector_float64_field_with_one_component_per_dimension() {
    let image = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = demons_registration(&image, &image, None, &DemonsParams::default()).unwrap();
    assert_eq!(result.displacement_field.pixel_id(), PixelId::VectorFloat64);
    assert_eq!(
        result.displacement_field.number_of_components_per_pixel(),
        2
    );
    assert_eq!(result.displacement_field.size(), &[5, 1]);
}

/// One iteration on a fixed ramp against a zero moving image, hand-derived.
///
/// With `K = 1`, `∇f = (1, 0)` at every interior pixel and `0` at the two
/// x-borders, `m = 0`, and `speed = f`:
///
/// | pixel | f | ∇f_x | denominator = f²/K + ∇f_x² | update_x = f ∇f_x / denom |
/// |---|---|---|---|---|
/// | 0 | 0 | 0 | — | `0` (|speed| < 0.001, thresholded) |
/// | 1 | 1 | 1 | 1 + 1 = 2 | 1/2 = `0.5` |
/// | 2 | 2 | 1 | 4 + 1 = 5 | 2/5 = `0.4` |
/// | 3 | 3 | 1 | 9 + 1 = 10 | 3/10 = `0.3` |
/// | 4 | 4 | 0 | 16 + 0 = 16 | 0/16 = `0` |
///
/// `metric = Σ speed² / N = (0+1+4+9+16)/5 = 6`, and
/// `rms = sqrt(Σ|update|² / N) = sqrt((0.25+0.16+0.09)/5) = sqrt(0.1)`.
///
/// The displacement field is zero when it is smoothed at the top of the
/// iteration, so `smooth_displacement_field` cannot perturb this.
#[test]
fn one_iteration_against_a_zero_moving_image_is_hand_computable() {
    let fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let moving = row(&[0.0; 5]);
    let result = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();

    let x = x_components(&result);
    assert_eq!(x[0], 0.0);
    assert!((x[1] - 0.5).abs() < 1e-15, "{}", x[1]);
    assert!((x[2] - 0.4).abs() < 1e-15, "{}", x[2]);
    assert!((x[3] - 0.3).abs() < 1e-15, "{}", x[3]);
    assert_eq!(x[4], 0.0);
    assert_eq!(y_components(&result), vec![0.0; 5]);

    assert_eq!(result.elapsed_iterations, 1);
    assert!((result.metric - 6.0).abs() < 1e-15);
    assert!((result.rms_change - 0.1f64.sqrt()).abs() < 1e-15);
}

/// The gradient choice is a real fork, not a cosmetic one. At pixel `(2, 0)`,
/// with `fixed = [0,1,2,3,4]` and `moving = [0,0,4,4,4]`:
///
/// * `speed = f(2) - m(2) = 2 - 4 = -2`, so `speed² = 4` and `K = 1`.
/// * **Fixed gradient** (`EvaluateAtIndex`): `(f(3) - f(1))/2 = 1`.
///   `denominator = 4 + 1 = 5`, `update_x = -2 * 1 / 5 = -0.4`.
/// * **Moving gradient** (`Evaluate` at the mapped point, central difference
///   over `±0.5 * spacing`): `m(2.5) = 4`, `m(1.5) = (0 + 4)/2 = 2`, so the
///   derivative is `(4 - 2)/1 = 2`. `denominator = 4 + 4 = 8`,
///   `update_x = -2 * 2 / 8 = -0.5`.
#[test]
fn use_moving_image_gradient_switches_the_demons_force() {
    let fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let moving = row(&[0.0, 0.0, 4.0, 4.0, 4.0]);

    let with_fixed = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    assert!((x_components(&with_fixed)[2] - -0.4).abs() < 1e-15);

    let with_moving = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            use_moving_image_gradient: true,
            ..one_iteration()
        },
    )
    .unwrap();
    assert!((x_components(&with_moving)[2] - -0.5).abs() < 1e-15);
}

/// `|speed| < intensity_difference_threshold` returns the zero vector. With
/// `moving = fixed - delta` the speed is exactly `delta` at every pixel and the
/// fixed gradient is `1` at the interior ones.
#[test]
fn intensity_difference_threshold_zeroes_a_matched_pixel() {
    let fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);

    // Just below the default 0.001 threshold: every update is zero.
    let delta = 0.0005;
    let moving = row(&[-delta, 1.0 - delta, 2.0 - delta, 3.0 - delta, 4.0 - delta]);
    let matched = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    assert!(
        matched
            .displacement_field
            .component_slice::<f64>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert_eq!(matched.rms_change, 0.0);
    // The metric still sees these pixels: they are counted before the threshold.
    assert!((matched.metric - delta * delta).abs() < 1e-18);

    // Just above it: `update_x = delta * 1 / (delta² + 1)`.
    let delta = 0.002;
    let moving = row(&[-delta, 1.0 - delta, 2.0 - delta, 3.0 - delta, 4.0 - delta]);
    let unmatched = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    let expected = delta / (delta * delta + 1.0);
    assert!((x_components(&unmatched)[2] - expected).abs() < 1e-15);
}

/// The `m_DenominatorThreshold` guard (`1e-9`, hard-coded and unsettable) is
/// what stops `speed * ∇f / denominator` from amplifying numerical dust into a
/// full-size displacement.
///
/// The two images below are the same ramp scaled by `1e-9`. Scaling `f` by `s`
/// scales `speed` by `s` and `∇f` by `s`, so the ratio `speed ∇f / (speed² +
/// |∇f|²)` is *scale-invariant*: at pixel `(2, 0)` it is `0.4` for both. But the
/// scaled denominator is `5e-18`, below `1e-9`, so the guard fires and the
/// update is zero. With `intensity_difference_threshold` set to `0` the
/// denominator guard is the only thing that can zero it.
#[test]
fn the_denominator_threshold_zeroes_a_scale_invariant_update() {
    let params = DemonsParams {
        intensity_difference_threshold: 0.0,
        ..one_iteration()
    };
    let moving = row(&[0.0; 5]);

    let unscaled = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let big = demons_registration(&unscaled, &moving, None, &params).unwrap();
    assert!((x_components(&big)[2] - 0.4).abs() < 1e-15);

    let scaled = row(&[0.0, 1e-9, 2e-9, 3e-9, 4e-9]);
    let small = demons_registration(&scaled, &moving, None, &params).unwrap();
    // Scale-invariant force, but denominator == 5e-18 < 1e-9.
    assert_eq!(x_components(&small)[2], 0.0);
    assert_eq!(small.rms_change, 0.0);
}

/// An initial field that already aligns the images leaves the field alone.
/// With `fixed(x) = x + 1` and `moving(x) = x`, sampling the moving image at
/// `x + 1` reproduces the fixed image, so every speed is exactly zero.
///
/// The last pixel maps to `x = 5`, outside the moving image's continuous buffer
/// bound of `4.5`, so it returns the zero update *without* being counted —
/// `metric` averages over four pixels, not five.
#[test]
fn an_initial_displacement_field_is_the_starting_point() {
    let fixed = row(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let moving = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);

    let mut initial = Image::new_vector(&[5, 1], PixelId::VectorFloat64, 2).unwrap();
    for pixel in 0..5 {
        initial.set_vector(&[pixel, 0], &[1.0f64, 0.0]).unwrap();
    }

    let params = DemonsParams {
        smooth_displacement_field: false,
        ..DemonsParams::default()
    };
    let result = demons_registration(&fixed, &moving, Some(&initial), &params).unwrap();

    assert_eq!(x_components(&result), vec![1.0; 5]);
    assert_eq!(y_components(&result), vec![0.0; 5]);
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.rms_change, 0.0);
    assert_eq!(result.elapsed_iterations, 1);
}

/// Without the initial field the same pair does move: `speed` is `1` at every
/// pixel and the fixed gradient is nonzero in the interior.
#[test]
fn the_same_pair_without_an_initial_field_produces_a_nonzero_update() {
    let fixed = row(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let moving = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    assert!(x_components(&result).iter().any(|&v| v != 0.0));
    // `speed == 1` everywhere inside the buffer, so `metric == 1`.
    assert!((result.metric - 1.0).abs() < 1e-15);
}

/// `number_of_iterations == 0` halts before the first iteration. The field is
/// the initial one, and `rms_change` is `FiniteDifferenceImageFilter`'s
/// `m_RMSChange{}` — `0.0`, not the function's `f64::MAX`. `metric` never got
/// written, so it *is* `f64::MAX`.
#[test]
fn zero_iterations_returns_the_initial_field_and_an_unset_metric() {
    let fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let moving = row(&[0.0; 5]);
    let result = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            number_of_iterations: 0,
            ..DemonsParams::default()
        },
    )
    .unwrap();

    assert_eq!(result.elapsed_iterations, 0);
    assert_eq!(result.rms_change, 0.0);
    assert_eq!(result.metric, f64::MAX);
    assert!(
        result
            .displacement_field
            .component_slice::<f64>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
}

/// The RMS halting test is a strict `maximum_rms_error > rms_change`. Setting
/// `maximum_rms_error` to `0.0` makes it unsatisfiable even for a converged
/// pair, so all ten iterations run.
#[test]
fn a_zero_maximum_rms_error_never_halts_early() {
    let image = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = demons_registration(
        &image,
        &image,
        None,
        &DemonsParams {
            maximum_rms_error: 0.0,
            number_of_iterations: 10,
            ..DemonsParams::default()
        },
    )
    .unwrap();
    assert_eq!(result.elapsed_iterations, 10);
    assert_eq!(result.rms_change, 0.0);
}

/// `use_image_spacing` reaches only `SetScaleCoefficients`, which no PDE
/// deformable registration function reads. Toggling it must change nothing.
#[test]
fn use_image_spacing_is_inert() {
    let mut fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    // Non-unit spacing, so a live `use_image_spacing` would have something to do.
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    let mut moving = row(&[0.0, 0.0, 4.0, 4.0, 4.0]);
    moving.set_spacing(&[2.0, 3.0]).unwrap();

    let on = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            use_image_spacing: true,
            ..DemonsParams::default()
        },
    )
    .unwrap();
    let off = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            use_image_spacing: false,
            ..DemonsParams::default()
        },
    )
    .unwrap();
    assert_eq!(on, off);
}

/// Smoothing the update field (viscous) is a distinct regularisation from
/// smoothing the displacement field (elastic), and both are reachable.
#[test]
fn smoothing_the_update_field_changes_the_result() {
    let fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let moving = row(&[0.0; 5]);

    let plain = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    let viscous = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            smooth_update_field: true,
            ..one_iteration()
        },
    )
    .unwrap();

    assert_ne!(x_components(&plain), x_components(&viscous));
    // The update was `[0, 0.5, 0.4, 0.3, 0]`; smoothing preserves its mean but
    // spreads it, so the peak drops.
    assert!(x_components(&viscous)[1] < x_components(&plain)[1]);
}

/// The output takes its geometry from the fixed image when no initial field is
/// given, per `GenerateOutputInformation`'s `else if (this->GetFixedImage())`.
#[test]
fn the_output_geometry_comes_from_the_fixed_image() {
    let mut fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    fixed.set_origin(&[7.0, 8.0]).unwrap();
    let moving = row(&[0.0; 5]);

    let result = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    assert_eq!(result.displacement_field.spacing(), &[2.0, 3.0]);
    assert_eq!(result.displacement_field.origin(), &[7.0, 8.0]);
}

/// ...and from the initial displacement field when one is given, per the
/// `if (this->GetInput(0))` branch.
#[test]
fn the_output_geometry_comes_from_the_initial_field_when_set() {
    let mut fixed = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    let moving = row(&[0.0; 5]);

    let mut initial = Image::new_vector(&[5, 1], PixelId::VectorFloat64, 2).unwrap();
    initial.set_spacing(&[0.5, 0.25]).unwrap();
    initial.set_origin(&[-1.0, -2.0]).unwrap();

    let result = demons_registration(&fixed, &moving, Some(&initial), &one_iteration()).unwrap();
    assert_eq!(result.displacement_field.spacing(), &[0.5, 0.25]);
    assert_eq!(result.displacement_field.origin(), &[-1.0, -2.0]);
}

#[test]
fn a_vector_fixed_image_is_rejected() {
    let fixed = Image::from_vec_vector(&[5, 1], 2, vec![0.0f64; 10]).unwrap();
    let moving = Image::from_vec_vector(&[5, 1], 2, vec![0.0f64; 10]).unwrap();
    assert!(matches!(
        demons_registration(&fixed, &moving, None, &DemonsParams::default()).unwrap_err(),
        FilterError::Core(crate::core::Error::RequiresScalarPixelType(
            PixelId::VectorFloat64
        ))
    ));
}

#[test]
fn mismatched_pixel_types_are_rejected() {
    let fixed = row(&[0.0; 5]);
    let moving = Image::from_vec(&[5, 1], vec![0.0f32; 5]).unwrap();
    assert!(matches!(
        demons_registration(&fixed, &moving, None, &DemonsParams::default()).unwrap_err(),
        FilterError::TypeMismatch { .. }
    ));
}

#[test]
fn mismatched_dimensions_are_rejected() {
    let fixed = row(&[0.0; 5]);
    let moving = Image::from_vec(&[5], vec![0.0f64; 5]).unwrap();
    assert!(matches!(
        demons_registration(&fixed, &moving, None, &DemonsParams::default()).unwrap_err(),
        FilterError::TypeMismatch { .. } | FilterError::ImageDimensionMismatch { .. }
    ));
}

/// `sitkSTLVectorToITK` throws when the vector is shorter than the image
/// dimension. A 2D image needs at least two standard deviations; the yaml's
/// default of three is fine, and its third entry is truncated away.
#[test]
fn standard_deviations_shorter_than_the_dimension_are_rejected() {
    let image = row(&[0.0; 5]);
    assert!(matches!(
        demons_registration(
            &image,
            &image,
            None,
            &DemonsParams {
                standard_deviations: vec![1.0],
                ..DemonsParams::default()
            }
        )
        .unwrap_err(),
        FilterError::DimensionLength {
            expected: 2,
            got: 1
        }
    ));
    assert!(matches!(
        demons_registration(
            &image,
            &image,
            None,
            &DemonsParams {
                update_field_standard_deviations: vec![1.0],
                smooth_update_field: true,
                ..DemonsParams::default()
            }
        )
        .unwrap_err(),
        FilterError::DimensionLength {
            expected: 2,
            got: 1
        }
    ));
}

/// The default three-entry vector is truncated to the image dimension, as
/// `sitkSTLVectorToITK` does, rather than rejected.
#[test]
fn extra_standard_deviations_are_truncated() {
    let image = row(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let with_three = demons_registration(&image, &image, None, &DemonsParams::default()).unwrap();
    let with_two = demons_registration(
        &image,
        &image,
        None,
        &DemonsParams {
            standard_deviations: vec![1.0, 1.0],
            ..DemonsParams::default()
        },
    )
    .unwrap();
    assert_eq!(with_three, with_two);
}

/// `GaussianOperator::SetMaximumError` rejects anything outside the open
/// interval `(0, 1)`, including both endpoints.
#[test]
fn a_maximum_error_outside_the_open_unit_interval_is_rejected() {
    let image = row(&[0.0; 5]);
    for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        assert!(
            matches!(
                demons_registration(
                    &image,
                    &image,
                    None,
                    &DemonsParams {
                        maximum_error: bad,
                        ..DemonsParams::default()
                    }
                )
                .unwrap_err(),
                FilterError::GaussianMaximumErrorOutOfRange(_)
            ),
            "maximum_error {bad} should be rejected"
        );
    }
}

#[test]
fn an_initial_field_of_the_wrong_pixel_type_is_rejected() {
    let image = row(&[0.0; 5]);
    let initial = Image::from_vec_vector(&[5, 1], 2, vec![0.0f32; 10]).unwrap();
    assert!(matches!(
        demons_registration(&image, &image, Some(&initial), &DemonsParams::default()).unwrap_err(),
        FilterError::TypeMismatch {
            a: PixelId::VectorFloat64,
            b: PixelId::VectorFloat32
        }
    ));
}

#[test]
fn an_initial_field_with_the_wrong_component_count_is_rejected() {
    let image = row(&[0.0; 5]);
    let initial = Image::from_vec_vector(&[5, 1], 3, vec![0.0f64; 15]).unwrap();
    assert!(matches!(
        demons_registration(&image, &image, Some(&initial), &DemonsParams::default()).unwrap_err(),
        FilterError::DimensionLength {
            expected: 2,
            got: 3
        }
    ));
}

#[test]
fn an_initial_field_of_the_wrong_size_is_rejected() {
    let image = row(&[0.0; 5]);
    let initial = Image::from_vec_vector(&[4, 1], 2, vec![0.0f64; 8]).unwrap();
    assert!(matches!(
        demons_registration(&image, &image, Some(&initial), &DemonsParams::default()).unwrap_err(),
        FilterError::SizeMismatch { .. }
    ));
}

/// A 3D smoke test: the field is the right shape, and an aligned pair still
/// converges to zero. Exercises the `dim == 3` paths of the smoother, the
/// normalizer, and the trilinear interpolator.
#[test]
fn three_dimensional_registration_of_an_aligned_pair() {
    let size = [3usize, 3, 3];
    let data: Vec<f64> = (0..27).map(f64::from).collect();
    let image = Image::from_vec(&size, data).unwrap();
    let result = demons_registration(&image, &image, None, &DemonsParams::default()).unwrap();

    assert_eq!(result.displacement_field.size(), &size);
    assert_eq!(
        result.displacement_field.number_of_components_per_pixel(),
        3
    );
    assert!(
        result
            .displacement_field
            .component_slice::<f64>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.elapsed_iterations, 1);
}

/// Integer inputs are accepted (`BasicPixelIDTypeList`) and widened to `f64`.
#[test]
fn integer_inputs_are_accepted() {
    let fixed = Image::from_vec(&[5, 1], vec![0u8, 1, 2, 3, 4]).unwrap();
    let moving = Image::from_vec(&[5, 1], vec![0u8; 5]).unwrap();
    let result = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    // Same arithmetic as the `f64` ramp above.
    assert!((x_components(&result)[2] - 0.4).abs() < 1e-15);
}

/// Registration reduces the metric: ten iterations on a shifted pair leave the
/// mean square intensity difference below its value after one.
#[test]
fn iterating_reduces_the_metric() {
    let fixed = Image::from_vec(
        &[9, 9],
        (0..81)
            .map(|i| {
                let (x, y) = (i % 9, i / 9);
                if (3..6).contains(&x) && (3..6).contains(&y) {
                    100.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    let moving = Image::from_vec(
        &[9, 9],
        (0..81)
            .map(|i| {
                let (x, y) = (i % 9, i / 9);
                if (4..7).contains(&x) && (3..6).contains(&y) {
                    100.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>(),
    )
    .unwrap();

    let after_one = demons_registration(&fixed, &moving, None, &one_iteration()).unwrap();
    let after_ten = demons_registration(
        &fixed,
        &moving,
        None,
        &DemonsParams {
            number_of_iterations: 10,
            maximum_rms_error: 0.0,
            ..DemonsParams::default()
        },
    )
    .unwrap();

    assert_eq!(after_ten.elapsed_iterations, 10);
    assert!(
        after_ten.metric < after_one.metric,
        "metric did not decrease: {} -> {}",
        after_one.metric,
        after_ten.metric
    );
}
