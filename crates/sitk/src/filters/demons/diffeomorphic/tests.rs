//! Every expectation here is hand-derived from
//! itkDiffeomorphicDemonsRegistrationFilter.hxx:74-283,
//! itkExponentialDisplacementFieldImageFilter.hxx:60-209 and
//! itkESMDemonsRegistrationFunction.hxx:166-391.
//!
//! The images are `5 x 3` with unit spacing, identity direction and zero origin,
//! so `m_Normalizer = (1² + 1²) · 0.5² / 2 = 0.25` at the default
//! `maximum_update_step_length`, and every `y` gradient vanishes.

use super::*;
use crate::core::Image;
use crate::filters::demons::field::{Field, Smoothing, smooth_field};
use crate::filters::demons::{
    FastSymmetricForcesDemonsParams, fast_symmetric_forces_demons_registration,
};

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

/// One iteration, no smoothing, gradient from the fixed image alone — so the
/// raw update is the hand-computable `[0, 0.5, 0.4, 0.3, 0]`.
fn bare() -> DiffeomorphicDemonsParams {
    DiffeomorphicDemonsParams {
        number_of_iterations: 1,
        smooth_displacement_field: false,
        use_gradient_type: EsmGradient::Fixed,
        ..Default::default()
    }
}

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

#[test]
fn the_imposed_squaring_count_follows_the_step_length() {
    // numiterfloat = 2 + log2(step), rounded up, and zero once it is not positive.
    assert_eq!(squaring_steps(2.0), (false, 3));
    assert_eq!(squaring_steps(1.0), (false, 2));
    assert_eq!(squaring_steps(0.75), (false, 2));
    assert_eq!(squaring_steps(0.5), (false, 1));
    // 2 + log2(0.25) == 0.0, which is not > 0.
    assert_eq!(squaring_steps(0.25), (false, 0));
    assert_eq!(squaring_steps(0.2), (false, 0));
    // Unrestricted: the exponentiator picks its own count, capped at 2000.
    assert_eq!(squaring_steps(0.0), (true, 2000));
    assert_eq!(squaring_steps(-1.0), (true, 2000));
}

#[test]
fn identical_images_produce_a_zero_field_and_halt_after_one_iteration() {
    let image = grid(|x, _| x as f64);
    let params = DiffeomorphicDemonsParams {
        smooth_displacement_field: false,
        ..Default::default()
    };
    let result = diffeomorphic_demons_registration(&image, &image, None, &params).unwrap();

    assert_eq!(result.elapsed_iterations, 1);
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.rms_change, 0.0);
    assert_close(&components(&result.displacement_field, 0), &[0.0; 15]);
    assert_close(&components(&result.displacement_field, 1), &[0.0; 15]);
}

/// `fixed = x`, `moving = 0`, `UseGradientType = Fixed`. The warped moving image
/// is constant, `∇f = [0, 1, 1, 1, 0]`, and `usedGradientTimes2 = 2∇f`:
///
/// | x | speed | `\|g₂\|²` | speed²/0.25 | update = 2·speed·g₂/denominator |
/// |---|---|---|---|---|
/// | 0 | 0 | 4 | 0 | 0 (below the intensity threshold) |
/// | 1 | 1 | 4 | 4 | 2·1·2/8 = 0.5 |
/// | 2 | 2 | 4 | 16 | 2·2·2/20 = 0.4 |
/// | 3 | 3 | 4 | 36 | 2·3·2/40 = 0.3 |
/// | 4 | 4 | 0 | 64 | 0 (the gradient is zero) |
///
/// With `UseFirstOrderExp` the field is composed with `Id + u` from a zero
/// start, which is just `u`.
#[test]
fn the_first_order_update_from_a_zero_field_is_the_raw_demons_force() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let params = DiffeomorphicDemonsParams {
        use_first_order_exp: true,
        ..bare()
    };
    let result = diffeomorphic_demons_registration(&fixed, &moving, None, &params).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 0.5, 0.4, 0.3, 0.0]),
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; 15]);
}

/// The same run through the exponential. At the default step length of `0.5`
/// the filter imposes exactly one squaring step, so `e = w + w ∘ (Id + w)` with
/// `w = u/2 = [0, 0.25, 0.2, 0.15, 0]`:
///
/// | x | w(x) | samples w at | value there | e(x) |
/// |---|---|---|---|---|
/// | 0 | 0 | 0.00 | 0 | 0 |
/// | 1 | 0.25 | 1.25 | 0.75·0.25 + 0.25·0.2 = 0.2375 | 0.4875 |
/// | 2 | 0.2 | 2.20 | 0.8·0.2 + 0.2·0.15 = 0.19 | 0.39 |
/// | 3 | 0.15 | 3.15 | 0.85·0.15 + 0.15·0 = 0.1275 | 0.2775 |
/// | 4 | 0 | 4.00 | 0 | 0 |
///
/// The field starts at zero, so `s ∘ exp(u) = 0 + e = e`.
#[test]
fn the_exponential_update_from_a_zero_field_is_the_hand_computed_exponential() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let result = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 0.4875, 0.39, 0.2775, 0.0]),
    );
    assert_close(&components(&result.displacement_field, 1), &[0.0; 15]);
}

/// The exponential is applied to the field, but the metric and the RMS change
/// were accumulated over the raw update `u` — "We compute the global data
/// without taking into account the current update step"
/// (itkESMDemonsRegistrationFunction.hxx:375-381).
///
/// `Σ speed² = 3·(0 + 1 + 4 + 9 + 16) = 90` over 15 processed pixels, and
/// `Σ |u|² = 3·(0.25 + 0.16 + 0.09) = 1.5`.
#[test]
fn the_metric_and_rms_change_measure_the_raw_update_not_the_exponential() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let result = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert!((result.metric - 6.0).abs() < 1e-12, "{}", result.metric);
    let expected = (1.5f64 / 15.0).sqrt();
    assert!(
        (result.rms_change - expected).abs() < 1e-12,
        "{} vs {expected}",
        result.rms_change
    );
}

/// `2 + log2(0.25) == 0`, so the imposed count is zero, the exponentiator takes
/// its caster branch, and `exp(u) == u` — the update rule is first-order whether
/// or not `UseFirstOrderExp` is set.
#[test]
fn a_short_enough_step_length_makes_the_exponential_the_identity() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);

    let with_exp = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            maximum_update_step_length: 0.25,
            ..bare()
        },
    )
    .unwrap();
    let first_order = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            maximum_update_step_length: 0.25,
            use_first_order_exp: true,
            ..bare()
        },
    )
    .unwrap();
    assert_eq!(with_exp.displacement_field, first_order.displacement_field);

    // At 0.5 the squaring step runs and the two part ways.
    let squared = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let squared_first_order = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            use_first_order_exp: true,
            ..bare()
        },
    )
    .unwrap();
    assert_ne!(
        squared.displacement_field,
        squared_first_order.displacement_field
    );
}

/// `maximum_update_step_length <= 0` sets `m_Normalizer = -1`, which drops the
/// `speed²/K` term from the denominator: `update = 2·speed·g₂/|g₂|² = speed·g₂/2`
/// with `|g₂| = 2`, i.e. `speed` itself. The last column's zero gradient now
/// takes the denominator below `m_DenominatorThreshold` instead.
#[test]
fn an_unrestricted_step_length_drops_the_intensity_term_from_the_denominator() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let result = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            maximum_update_step_length: 0.0,
            use_first_order_exp: true,
            ..bare()
        },
    )
    .unwrap();

    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 1.0, 2.0, 3.0, 0.0]),
    );
}

/// Warping a *constant* field is the identity — nearest-neighbour extrapolation
/// keeps every sample at the same value — so `s ∘ (Id + u) + u` collapses to
/// `s + u`, exactly what `FastSymmetricForcesDemonsRegistrationFilter` does with
/// the same difference function.
#[test]
fn with_a_constant_initial_field_the_first_order_composition_reduces_to_addition() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);
    let initial = constant_field([1.0, 0.0]);

    let composed = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        Some(&initial),
        &DiffeomorphicDemonsParams {
            use_first_order_exp: true,
            ..bare()
        },
    )
    .unwrap();
    let added = fast_symmetric_forces_demons_registration(
        &fixed,
        &moving,
        Some(&initial),
        &FastSymmetricForcesDemonsParams {
            number_of_iterations: 1,
            smooth_displacement_field: false,
            use_gradient_type: EsmGradient::Fixed,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(composed.displacement_field, added.displacement_field);
    assert_eq!(composed.metric, added.metric);
    assert_eq!(composed.rms_change, added.rms_change);
}

/// A zero update composes to `s(p_i) + 0`, and sampling a field at its own pixel
/// centres returns it unchanged.
#[test]
fn a_zero_update_leaves_the_field_exactly_where_it_was() {
    let image = grid(|_, _| 5.0);
    let initial = constant_field([1.0, 0.0]);
    let result =
        diffeomorphic_demons_registration(&image, &image, Some(&initial), &bare()).unwrap();

    assert_close(&components(&result.displacement_field, 0), &[1.0; 15]);
    assert_close(&components(&result.displacement_field, 1), &[0.0; 15]);
    assert_eq!(result.metric, 0.0);
    assert_eq!(result.rms_change, 0.0);
}

/// `ApplyUpdate` smooths the field *after* the composition, so a one-iteration
/// run from a zero field returns the smoothed exponential — not the exponential
/// of a smoothed field, and not the raw exponential.
#[test]
fn smoothing_the_displacement_field_happens_after_the_composition() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);

    let raw = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let smoothed = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            smooth_displacement_field: true,
            ..bare()
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
    smooth_field(
        &mut expected,
        &Smoothing {
            standard_deviations: vec![1.0, 1.0],
            maximum_error: 0.1,
            maximum_kernel_width: 30,
        },
    );

    assert_close(
        smoothed
            .displacement_field
            .component_slice::<f64>()
            .unwrap(),
        &expected.data,
    );
}

#[test]
fn smoothing_the_update_field_changes_the_result() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|_, _| 0.0);

    let plain = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();
    let viscous = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            smooth_update_field: true,
            ..bare()
        },
    )
    .unwrap();

    assert_ne!(plain.displacement_field, viscous.displacement_field);
}

/// `GetRMSChange()` is overridden to return the difference function's
/// `NumericTraits<double>::max()`, as in the other ESM-based filter.
#[test]
fn a_zero_iteration_run_reports_the_functions_initial_rms_change() {
    let image = grid(|x, _| x as f64);
    let result = diffeomorphic_demons_registration(
        &image,
        &image,
        None,
        &DiffeomorphicDemonsParams {
            number_of_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.elapsed_iterations, 0);
    assert_eq!(result.rms_change, f64::MAX);
    assert_eq!(result.metric, f64::MAX);
}

/// `Symmetric` averages the two gradients, `Fixed` doubles the fixed one, and
/// `WarpedMoving` doubles the warped moving one — three different forces on a
/// moving image that actually has a gradient.
#[test]
fn the_symmetric_fixed_and_warped_gradients_give_different_fields() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|x, _| if x >= 2 { 1.0 } else { 0.0 });

    let run = |use_gradient_type| {
        diffeomorphic_demons_registration(
            &fixed,
            &moving,
            None,
            &DiffeomorphicDemonsParams {
                use_gradient_type,
                ..bare()
            },
        )
        .unwrap()
        .displacement_field
    };

    assert_ne!(run(EsmGradient::Symmetric), run(EsmGradient::Fixed));
    assert_ne!(run(EsmGradient::Symmetric), run(EsmGradient::WarpedMoving));
    assert_ne!(run(EsmGradient::Fixed), run(EsmGradient::WarpedMoving));
}

/// `WarpedMoving` central-differences the warped moving image on the grid;
/// `MappedMoving` evaluates `CentralDifferenceImageFunction` on the *unwarped*
/// moving image at the mapped point, sampling it at `±0.5 · spacing`
/// (itkCentralDifferenceImageFunction.hxx:263-274).
///
/// At a zero field the mapped point is the pixel centre, and linear
/// interpolation makes `(m(x + 0.5) - m(x - 0.5)) / 1` identical to the
/// index-space central difference — so the two gradients coincide. They part
/// as soon as the field displaces the sample points off the grid.
#[test]
fn the_mapped_moving_gradient_matches_the_warped_one_only_while_the_field_is_zero() {
    let fixed = grid(|x, _| x as f64);
    let moving = grid(|x, _| if x >= 2 { 1.0 } else { 0.0 });

    let run = |use_gradient_type, initial: Option<&Image>| {
        diffeomorphic_demons_registration(
            &fixed,
            &moving,
            initial,
            &DiffeomorphicDemonsParams {
                use_gradient_type,
                ..bare()
            },
        )
        .unwrap()
        .displacement_field
    };

    assert_eq!(
        run(EsmGradient::WarpedMoving, None),
        run(EsmGradient::MappedMoving, None)
    );

    let shifted = constant_field([0.5, 0.0]);
    assert_ne!(
        run(EsmGradient::WarpedMoving, Some(&shifted)),
        run(EsmGradient::MappedMoving, Some(&shifted))
    );
}

#[test]
fn use_image_spacing_is_inert() {
    let mut fixed = grid(|x, _| x as f64);
    let mut moving = grid(|_, _| 0.0);
    fixed.set_spacing(&[2.0, 3.0]).unwrap();
    moving.set_spacing(&[2.0, 3.0]).unwrap();

    let on = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            use_image_spacing: true,
            ..bare()
        },
    )
    .unwrap();
    let off = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
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

    let one = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
            number_of_iterations: 1,
            maximum_rms_error: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let ten = diffeomorphic_demons_registration(
        &fixed,
        &moving,
        None,
        &DiffeomorphicDemonsParams {
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
    let data: Vec<f64> = (0..27).map(|i| f64::from(i as u32 % 3)).collect();
    let fixed = Image::from_vec(&[3, 3, 3], data).unwrap();
    let moving = Image::from_vec(&[3, 3, 3], vec![0.0; 27]).unwrap();
    let result = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();

    assert_eq!(
        result.displacement_field.number_of_components_per_pixel(),
        3
    );
    assert_eq!(result.displacement_field.size(), &[3, 3, 3]);
}

#[test]
fn integer_inputs_are_accepted() {
    let fixed = Image::from_vec(&[NX, NY], (0..NX * NY).map(|i| (i % NX) as u8).collect()).unwrap();
    let moving = Image::from_vec(&[NX, NY], vec![0u8; NX * NY]).unwrap();
    let params = DiffeomorphicDemonsParams {
        use_first_order_exp: true,
        ..bare()
    };
    let result = diffeomorphic_demons_registration(&fixed, &moving, None, &params).unwrap();
    assert_close(
        &components(&result.displacement_field, 0),
        &every_row([0.0, 0.5, 0.4, 0.3, 0.0]),
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

    let result = diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).unwrap();
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
        diffeomorphic_demons_registration(&fixed, &moving, Some(&initial), &bare()).unwrap();
    assert_eq!(result.displacement_field.origin(), &[7.0, 8.0]);
}

#[test]
fn a_vector_fixed_image_is_rejected() {
    let image = Image::from_vec_vector(&[NX, NY], 2, vec![0.0; NX * NY * 2]).unwrap();
    assert!(diffeomorphic_demons_registration(&image, &image, None, &bare()).is_err());
}

#[test]
fn mismatched_pixel_types_are_rejected() {
    let fixed = grid(|_, _| 0.0);
    let moving = Image::from_vec(&[NX, NY], vec![0.0f32; NX * NY]).unwrap();
    assert!(diffeomorphic_demons_registration(&fixed, &moving, None, &bare()).is_err());
}

#[test]
fn a_maximum_error_outside_the_open_unit_interval_is_rejected() {
    let image = grid(|_, _| 0.0);
    for maximum_error in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let params = DiffeomorphicDemonsParams {
            maximum_error,
            ..bare()
        };
        assert!(
            diffeomorphic_demons_registration(&image, &image, None, &params).is_err(),
            "accepted maximum_error {maximum_error}"
        );
    }
}

#[test]
fn standard_deviations_shorter_than_the_image_dimension_are_rejected() {
    let image = grid(|_, _| 0.0);
    let short = DiffeomorphicDemonsParams {
        standard_deviations: vec![1.0],
        ..bare()
    };
    assert!(diffeomorphic_demons_registration(&image, &image, None, &short).is_err());

    let short_update = DiffeomorphicDemonsParams {
        update_field_standard_deviations: vec![1.0],
        ..bare()
    };
    assert!(diffeomorphic_demons_registration(&image, &image, None, &short_update).is_err());
}

#[test]
fn an_initial_field_of_the_wrong_size_is_rejected() {
    let image = grid(|_, _| 0.0);
    let initial = Image::from_vec_vector(&[2, 2], 2, vec![0.0; 8]).unwrap();
    assert!(diffeomorphic_demons_registration(&image, &image, Some(&initial), &bare()).is_err());
}
