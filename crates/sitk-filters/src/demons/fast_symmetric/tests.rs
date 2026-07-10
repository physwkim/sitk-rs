//! Every expectation here is hand-derived from
//! itkESMDemonsRegistrationFunction.hxx:166-391 and
//! itkFastSymmetricForcesDemonsRegistrationFilter.hxx:37-220.
//!
//! The images are `5 x 3` with unit spacing, identity direction and zero origin,
//! so `m_Normalizer = (1² + 1²) · 0.5² / 2 = 0.25` at the default
//! `maximum_update_step_length`, and every `y` gradient vanishes (both images
//! are constant along `y`).

use super::*;
use crate::demons::field::{Field, Smoothing, smooth_field};
use crate::demons::{DemonsParams, demons_registration};
use sitk_core::Image;

const NX: usize = 5;
const NY: usize = 3;

/// A `5 x 3` `Float64` image whose value at `(x, y)` is `f(x, y)`.
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
fn bare() -> FastSymmetricForcesDemonsParams {
    FastSymmetricForcesDemonsParams {
        number_of_iterations: 1,
        smooth_displacement_field: false,
        ..Default::default()
    }
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

/// The same five `x`-components on each of the three rows.
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

/// `speed == 0` everywhere, so every update falls under the intensity-difference
/// threshold. The RMS change is `0`, strictly below the default
/// `maximum_rms_error`, so the solver halts after the first iteration.
#[test]
fn identical_images_produce_a_zero_field_and_halt_after_one_iteration() {
    let image = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        smooth_displacement_field: false,
        ..Default::default()
    };
    let result = fast_symmetric_forces_demons_registration(&image, &image, None, &params).unwrap();

    assert_eq!(result.elapsed_iterations, 1);
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.rms_change, 0.0);
    assert_close(&components(&result.displacement_field, 0), &[0.0; 15]);
    assert_close(&components(&result.displacement_field, 1), &[0.0; 15]);
}

/// `fixed = x`, `moving = 0`. The warped moving image is constant, so its
/// gradient is zero and the symmetric force reduces to `∇f`, which
/// `CentralDifferenceImageFunction` zeroes on the first and last columns:
/// `∇f = [0, 1, 1, 1, 0]`.
///
/// With `speed = x` and `denominator = |∇f|² + speed²/0.25`:
///
/// | x | speed | denominator | update = 2·speed·∇f/denominator |
/// |---|---|---|---|
/// | 0 | 0 | — | 0 (below the intensity threshold) |
/// | 1 | 1 | 1 + 4 = 5 | 2/5 = 0.4 |
/// | 2 | 2 | 1 + 16 = 17 | 4/17 |
/// | 3 | 3 | 1 + 36 = 37 | 6/37 |
/// | 4 | 4 | 0 + 64 = 64 | 0 (the gradient is zero) |
#[test]
fn a_ramp_against_a_constant_gives_the_hand_computed_symmetric_force() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    let expected = [0.0, 0.4, 4.0 / 17.0, 6.0 / 37.0, 0.0];
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row(expected),
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; 15]);

    // Every pixel maps inside, so the metric divides by all 15 of them.
    assert!((result.metric - 6.0).abs() < 1e-12, "{}", result.metric);
    let sum_of_squares: f64 = expected.iter().map(|u| u * u).sum();
    assert!((result.rms_change - (sum_of_squares / 5.0).sqrt()).abs() < 1e-12);
}

/// `GradientEnum::Fixed` doubles the fixed image's gradient, and the doubled
/// gradient appears in both the numerator and the denominator:
/// `update = 2·speed·2∇f / (4|∇f|² + speed²/0.25)`. At `x = 1` that is
/// `4 / 8 = 0.5`, against the symmetric force's `0.4`.
#[test]
fn the_fixed_gradient_type_doubles_the_fixed_images_gradient() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let params = FastSymmetricForcesDemonsParams {
        use_gradient_type: EsmGradient::Fixed,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 0.5, 0.4, 0.3, 0.0]),
    );
}

/// `fixed = 0`, `moving = x`. Now the *warped moving* gradient carries the
/// signal, and unlike `CentralDifferenceImageFunction` it takes a one-sided
/// difference at the borders: `∇(m∘φ) = [1, 1, 1, 1, 1]`. The fixed gradient is
/// zero, so the symmetric force is `∇(m∘φ)`.
///
/// | x | speed | denominator | update |
/// |---|---|---|---|
/// | 0 | 0 | — | 0 |
/// | 1 | -1 | 1 + 4 = 5 | -2/5 |
/// | 2 | -2 | 1 + 16 = 17 | -4/17 |
/// | 3 | -3 | 1 + 36 = 37 | -6/37 |
/// | 4 | -4 | 1 + 64 = 65 | -8/65 |
///
/// The last column is nonzero precisely because of the backward difference.
#[test]
fn the_warped_moving_gradient_uses_one_sided_differences_at_the_border() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -0.4, -4.0 / 17.0, -6.0 / 37.0, -8.0 / 65.0]),
    );
    assert!((result.metric - 6.0).abs() < 1e-12);
}

/// On the same pair, `GradientEnum::Fixed` sees only the constant fixed image:
/// the force is identically zero, yet the metric still reports the mismatch.
#[test]
fn the_fixed_gradient_type_cannot_move_a_constant_fixed_image() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        use_gradient_type: EsmGradient::Fixed,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(&components(&result.displacement_field, 0), &[0.0; 15]);
    assert!((result.metric - 6.0).abs() < 1e-12);
    assert_eq!(result.rms_change, 0.0);
}

/// `GradientEnum::WarpedMoving` doubles `∇(m∘φ) = [1, 1, 1, 1, 1]`, so
/// `update = 2·speed·2 / (4 + speed²/0.25)`. The last column keeps its one-sided
/// difference: `-8 / 68 · 2 = -4/17`.
#[test]
fn the_warped_moving_gradient_type_doubles_the_warped_gradient() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        use_gradient_type: EsmGradient::WarpedMoving,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -0.5, -0.4, -0.3, -4.0 / 17.0]),
    );
}

/// `GradientEnum::MappedMoving` differentiates the *unwarped* moving image at
/// the mapped point with `CentralDifferenceImageFunction::Evaluate`, whose
/// samples sit at `x ± 0.5`. At `x = 4` the upper sample lands on the buffer's
/// half-open `4.5` bound and is rejected, zeroing that component — where
/// `WarpedMoving` reported `-4/17`. Everywhere else the two agree.
#[test]
fn the_mapped_moving_gradient_type_zeroes_the_last_column() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        use_gradient_type: EsmGradient::MappedMoving,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -0.5, -0.4, -0.3, 0.0]),
    );
}

/// `maximum_update_step_length <= 0` sets `m_Normalizer = -1`, and the
/// denominator drops its intensity term: `update = 2·speed·g / |g|²`. With
/// `g = 1` that is `2·speed`.
#[test]
fn a_non_positive_step_length_removes_the_intensity_term() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        maximum_update_step_length: 0.0,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, -2.0, -4.0, -6.0, -8.0]),
    );
}

/// With no gradient the unrestricted denominator is `0`, below
/// `m_DenominatorThreshold = 1e-9`, and the update is zero even though the
/// intensity difference is large.
#[test]
fn a_zero_denominator_zeroes_the_update() {
    let fixed = grid(|_, _| 5.0);
    let moving = grid(|_, _| 0.0);
    let params = FastSymmetricForcesDemonsParams {
        maximum_update_step_length: 0.0,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(&components(&result.displacement_field, 0), &[0.0; 15]);
    assert!((result.metric - 25.0).abs() < 1e-12);
}

/// The intensity-difference threshold suppresses the *update*, not the *metric*:
/// `m_SumOfSquaredDifference` and `m_NumberOfPixelsProcessed` are accumulated
/// after the threshold test, for every pixel that mapped inside
/// (itkESMDemonsRegistrationFunction.hxx:383-388).
#[test]
fn the_intensity_difference_threshold_suppresses_updates_but_not_the_metric() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        intensity_difference_threshold: 3.5,
        ..bare()
    };
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &params).unwrap();

    // Only |speed| = 4, at x = 4, clears the threshold.
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 0.0, 0.0, 0.0, -8.0 / 65.0]),
    );
    assert!((result.metric - 6.0).abs() < 1e-12);
    let expected_rms = ((8.0f64 / 65.0).powi(2) / 5.0).sqrt();
    assert!((result.rms_change - expected_rms).abs() < 1e-12);
}

/// A displacement that carries every pixel out of the moving image's buffer
/// leaves `m_NumberOfPixelsProcessed` at zero, so the metric and the RMS change
/// keep their constructor values and the field never moves.
#[test]
fn a_field_that_maps_everything_outside_leaves_the_metric_untouched() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|x, _| x as f64);
    let initial = Image::from_vec_vector(
        &[NX, NY],
        2,
        every_row([10.0; NX])
            .iter()
            .flat_map(|&x| [x, 0.0])
            .collect(),
    )
    .unwrap();

    let result =
        fast_symmetric_forces_demons_registration(&fixed, &moving, Some(&initial), &bare())
            .unwrap();

    assert_eq!(result.elapsed_iterations, 1);
    assert_eq!(result.metric, f64::MAX);
    assert_eq!(result.rms_change, f64::MAX);
    assert_close(&components(&result.displacement_field, 0), &[10.0; 15]);
}

/// `GetRMSChange()` is overridden to return the difference function's value,
/// which starts at `NumericTraits<double>::max()` — where
/// `demons_registration`, which does not override it, reports the filter's
/// `0.0`.
#[test]
fn a_zero_iteration_run_reports_the_functions_initial_rms_change() {
    let image = grid(|x, _| x as f64);
    let params = FastSymmetricForcesDemonsParams {
        number_of_iterations: 0,
        ..Default::default()
    };
    let result = fast_symmetric_forces_demons_registration(&image, &image, None, &params).unwrap();
    assert_eq!(result.elapsed_iterations, 0);
    assert_eq!(result.rms_change, f64::MAX);
    assert_eq!(result.metric, f64::MAX);

    let demons = demons_registration(
        &image,
        &image,
        None,
        &DemonsParams {
            number_of_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(demons.rms_change, 0.0);
}

/// `WarpImageFilter`'s edge padding value is `NumericTraits<MovingPixelType>::max()`,
/// so a `UInt8` moving image that legitimately holds `255` is read as
/// out-of-buffer at every such pixel: no update, and no contribution to the
/// metric. The identical `UInt16` image, where `255` is nowhere near
/// `NumericTraits<uint16_t>::max()`, reports the mismatch.
#[test]
fn a_moving_pixel_equal_to_the_type_maximum_reads_as_out_of_buffer() {
    let params = FastSymmetricForcesDemonsParams {
        number_of_iterations: 1,
        smooth_displacement_field: false,
        ..Default::default()
    };

    let fixed_u8 = Image::from_vec(&[3, 3], vec![0u8; 9]).unwrap();
    let moving_u8 = Image::from_vec(&[3, 3], vec![255u8; 9]).unwrap();
    let swallowed =
        fast_symmetric_forces_demons_registration(&fixed_u8, &moving_u8, None, &params).unwrap();
    assert_eq!(swallowed.metric, f64::MAX);
    assert_eq!(swallowed.rms_change, f64::MAX);

    let fixed_u16 = Image::from_vec(&[3, 3], vec![0u16; 9]).unwrap();
    let moving_u16 = Image::from_vec(&[3, 3], vec![255u16; 9]).unwrap();
    let seen =
        fast_symmetric_forces_demons_registration(&fixed_u16, &moving_u16, None, &params).unwrap();
    assert!((seen.metric - 65025.0).abs() < 1e-9, "{}", seen.metric);
    assert_eq!(seen.rms_change, 0.0);
}

/// `ApplyUpdate` smooths the displacement field *after* adding the update, so a
/// one-iteration run from a zero field returns the smoothed update — not the raw
/// one, as `DemonsRegistrationFilter` would.
#[test]
fn smoothing_the_displacement_field_happens_after_the_update_is_added() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);

    let raw = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let smoothed = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &FastSymmetricForcesDemonsParams {
            number_of_iterations: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let mut expected = Field {
        data: raw
            .displacement_field
            .component_slice::<f64>()
            .unwrap()
            .to_vec(),
        size: vec![NX, NY],
    };
    let before = expected.data.clone();
    smooth_field(
        &mut expected,
        &Smoothing {
            standard_deviations: vec![1.0, 1.0],
            maximum_error: 0.1,
            maximum_kernel_width: 30,
        },
    );
    assert_ne!(expected.data, before, "the smoother must change this field");

    assert_close(
        smoothed
            .displacement_field
            .component_slice::<f64>()
            .unwrap(),
        &expected.data,
    );
}

/// With both smoothers on, the update is smoothed and then the field it lands in
/// is smoothed again.
#[test]
fn both_smoothers_apply_in_turn() {
    let fixed = grid(|_, _| 0.0);
    let moving = grid(|x, _| x as f64);

    let raw = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let both = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &FastSymmetricForcesDemonsParams {
            number_of_iterations: 1,
            smooth_update_field: true,
            ..Default::default()
        },
    )
    .unwrap();

    let smoothing = Smoothing {
        standard_deviations: vec![1.0, 1.0],
        maximum_error: 0.1,
        maximum_kernel_width: 30,
    };
    let mut expected = Field {
        data: raw
            .displacement_field
            .component_slice::<f64>()
            .unwrap()
            .to_vec(),
        size: vec![NX, NY],
    };
    smooth_field(&mut expected, &smoothing);
    smooth_field(&mut expected, &smoothing);

    assert_close(
        both.displacement_field.component_slice::<f64>().unwrap(),
        &expected.data,
    );
}

/// An axis of extent `1` gets a zero warped-moving derivative rather than the
/// out-of-buffer read ITK performs there. The `x` results are unchanged from the
/// `5 x 3` case, whose `y` gradient was zero anyway.
#[test]
fn a_degenerate_axis_yields_a_zero_derivative_instead_of_reading_out_of_bounds() {
    let fixed = Image::from_vec(&[NX, 1], (0..NX).map(|x| x as f64).collect()).unwrap();
    let moving = Image::from_vec(&[NX, 1], vec![0.0; NX]).unwrap();
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &[0.0, 0.4, 4.0 / 17.0, 6.0 / 37.0, 0.0],
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; NX]);
}

/// `use_image_spacing` reaches `SetScaleCoefficients`, which no function in
/// `Modules/Registration/PDEDeformable` reads.
#[test]
fn use_image_spacing_is_inert() {
    let mut fixed = grid(|x, _| x as f64);
    let mut moving = grid(|_, _| 0.0);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    moving.set_spacing(&[2.0, 3.0]).unwrap();

    let on = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &FastSymmetricForcesDemonsParams {
            use_image_spacing: true,
            ..bare()
        },
    )
    .unwrap();
    let off = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &FastSymmetricForcesDemonsParams {
            use_image_spacing: false,
            ..bare()
        },
    )
    .unwrap();
    assert_eq!(on.displacement_field, off.displacement_field);
}

/// The registration reduces the mean square intensity difference on a shifted
/// block.
#[test]
fn the_metric_falls_over_ten_iterations_on_a_shifted_block() {
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

    let one = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &FastSymmetricForcesDemonsParams {
            number_of_iterations: 1,
            maximum_rms_error: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let ten = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        None,
        &FastSymmetricForcesDemonsParams {
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
    let data: Vec<f64> = (0..27).map(|i| f64::from(i % 3)).collect();
    let fixed = Image::from_vec(&[3, 3, 3], data).unwrap();
    let moving = Image::from_vec(&[3, 3, 3], vec![0.0; 27]).unwrap();
    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert_eq!(
        result.displacement_field.number_of_components_per_pixel(),
        3
    );
    assert_eq!(result.displacement_field.size(), &[3, 3, 3]);
}

#[test]
fn the_output_takes_the_fixed_images_geometry() {
    let mut fixed = grid(|x, _| x as f64);
    let mut moving = grid(|_, _| 0.0);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    fixed.set_origin(&[10.0, 20.0]).unwrap();
    moving.set_spacing(&[2.0, 3.0]).unwrap();
    moving.set_origin(&[10.0, 20.0]).unwrap();

    let result = fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    assert_eq!(result.displacement_field.spacing(), &[2.0, 3.0]);
    assert_eq!(result.displacement_field.origin(), &[10.0, 20.0]);
}

#[test]
fn the_output_takes_the_initial_fields_geometry_when_one_is_given() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let mut initial = Image::from_vec_vector(&[NX, NY], 2, vec![0.0; NX * NY * 2]).unwrap();
    initial.set_origin(&[7.0, 8.0]).unwrap();

    let result =
        fast_symmetric_forces_demons_registration(&fixed, &moving, Some(&initial), &bare())
            .unwrap();
    assert_eq!(result.displacement_field.origin(), &[7.0, 8.0]);
}

#[test]
fn a_vector_fixed_image_is_rejected() {
    let fixed = Image::from_vec_vector(&[NX, NY], 2, vec![0.0; NX * NY * 2]).unwrap();
    let moving = Image::from_vec_vector(&[NX, NY], 2, vec![0.0; NX * NY * 2]).unwrap();
    assert!(fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).is_err());
}

#[test]
fn mismatched_pixel_types_are_rejected() {
    let fixed = grid(|_, _| 0.0);
    let moving = Image::from_vec(&[NX, NY], vec![0.0f32; NX * NY]).unwrap();
    assert!(fast_symmetric_forces_demons_registration(&fixed, &moving, None, &bare()).is_err());
}

#[test]
fn a_maximum_error_outside_the_open_unit_interval_is_rejected() {
    let image = grid(|_, _| 0.0);
    for maximum_error in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let params = FastSymmetricForcesDemonsParams {
            maximum_error,
            ..bare()
        };
        assert!(
            fast_symmetric_forces_demons_registration(&image, &image, None, &params).is_err(),
            "accepted maximum_error {maximum_error}"
        );
    }
}

#[test]
fn standard_deviations_shorter_than_the_image_dimension_are_rejected() {
    let image = grid(|_, _| 0.0);
    let short = FastSymmetricForcesDemonsParams {
        standard_deviations: vec![1.0],
        ..bare()
    };
    assert!(fast_symmetric_forces_demons_registration(&image, &image, None, &short).is_err());

    let short_update = FastSymmetricForcesDemonsParams {
        update_field_standard_deviations: vec![1.0],
        ..bare()
    };
    assert!(
        fast_symmetric_forces_demons_registration(&image, &image, None, &short_update).is_err()
    );
}

#[test]
fn an_initial_field_of_the_wrong_size_is_rejected() {
    let image = grid(|_, _| 0.0);
    let initial = Image::from_vec_vector(&[2, 2], 2, vec![0.0; 8]).unwrap();
    assert!(
        fast_symmetric_forces_demons_registration(&image, &image, Some(&initial), &bare()).is_err()
    );
}

#[test]
fn the_default_gradient_type_is_symmetric() {
    assert_eq!(EsmGradient::default(), EsmGradient::Symmetric);
    assert_eq!(
        FastSymmetricForcesDemonsParams::default().use_gradient_type,
        EsmGradient::Symmetric
    );
}
