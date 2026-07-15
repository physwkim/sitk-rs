use super::*;
use crate::core::Image;

/// A geometry over `size` with unit spacing, zero origin and identity
/// direction, so a physical point equals its continuous index.
fn unit_geometry(size: &[usize]) -> Geometry {
    let pixels: usize = size.iter().product();
    Geometry::new(&Image::from_vec(size, vec![0.0; pixels]).unwrap()).unwrap()
}

/// A `5 x 3` field, `x`-component from `values` on every row, `y`-component 0.
fn ramp_field(values: [f64; 5]) -> Field {
    let mut data = Vec::new();
    for _ in 0..3 {
        for &v in &values {
            data.push(v);
            data.push(0.0);
        }
    }
    Field {
        data,
        size: vec![5, 3],
    }
}

fn x_components(field: &Field) -> Vec<f64> {
    field.data.iter().step_by(2).copied().collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-12, "at {i}: {g} vs {w}");
    }
}

#[test]
fn warping_by_a_zero_displacement_returns_the_input() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([0.0, 1.0, 2.0, 3.0, 4.0]);
    let zero = Field::zeros(&[5, 3]);
    assert_eq!(warp(&input, &zero, &geometry), input);
}

/// Nearest-neighbour extrapolation keeps a constant field constant no matter
/// how far the displacement reaches outside the buffer — the property the
/// exponential's squaring relies on.
#[test]
fn warping_a_constant_field_gives_the_same_constant_everywhere() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([7.0; 5]);
    let displacement = ramp_field([-100.0, -2.0, 0.5, 3.0, 100.0]);
    assert_eq!(warp(&input, &displacement, &geometry), input);
}

/// `WarpVectorImageFilter`'s `m_EdgePaddingValue` (zero) is unreachable: the
/// interpolator's `IsInsideBuffer` always answers `true`, and its base index is
/// clamped into `[0, size - 1]` with a zero fractional distance.
#[test]
fn a_displacement_leaving_the_buffer_extrapolates_the_nearest_pixel() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([0.0, 1.0, 2.0, 3.0, 4.0]);

    let far_left = warp(&input, &ramp_field([-50.0; 5]), &geometry);
    assert_close(&x_components(&far_left), &[0.0; 15]);

    let far_right = warp(&input, &ramp_field([50.0; 5]), &geometry);
    assert_close(&x_components(&far_right), &[4.0; 15]);
}

/// Half a pixel to the right on a unit ramp: `v(x + 0.5) = x + 0.5`, except at
/// the last pixel, whose base index is clamped and whose distance is forced to
/// zero.
#[test]
fn a_fractional_displacement_interpolates_linearly() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([0.0, 1.0, 2.0, 3.0, 4.0]);
    let warped = warp(&input, &ramp_field([0.5; 5]), &geometry);

    let expected: Vec<f64> = [0.5, 1.5, 2.5, 3.5, 4.0]
        .iter()
        .copied()
        .cycle()
        .take(15)
        .collect();
    assert_close(&x_components(&warped), &expected);
}

/// `maxnorm2 == 0` takes the `NumericTraits<double>::min()` branch — `DBL_MIN`,
/// a positive denormal — so `numiterfloat >= 0.0` holds and the count truncates
/// to `1`, not `0`.
#[test]
fn a_zero_field_still_runs_one_squaring_step() {
    let geometry = unit_geometry(&[5, 3]);
    let zero = Field::zeros(&[5, 3]);
    assert_eq!(automatic_number_of_iterations(&zero, &geometry, 2000), 1);
}

/// `numiterfloat = 2 + 0.5·log2(1) = 2.0` exactly, and the code truncates
/// `numiterfloat + 1.0` rather than taking the ceiling its comment claims.
#[test]
fn an_exact_integer_iteration_count_is_rounded_up_anyway() {
    let geometry = unit_geometry(&[5, 3]);
    let unit = ramp_field([1.0; 5]);
    assert_eq!(automatic_number_of_iterations(&unit, &geometry, 2000), 3);
}

/// A displacement far below the pixel size drives `numiterfloat` negative and
/// the count to zero, making the exponential the identity.
#[test]
fn a_tiny_field_needs_no_squaring_step() {
    let geometry = unit_geometry(&[5, 3]);
    let tiny = ramp_field([0.001; 5]);
    // 2 + 0.5 * log2(1e-6) = -7.97 < 0.
    assert_eq!(automatic_number_of_iterations(&tiny, &geometry, 2000), 0);
}

#[test]
fn the_automatic_count_is_thresholded_by_the_maximum() {
    let geometry = unit_geometry(&[5, 3]);
    let unit = ramp_field([1.0; 5]);
    assert_eq!(automatic_number_of_iterations(&unit, &geometry, 2), 2);
}

/// The squared norm is divided by the square of the *minimum* spacing, so a
/// finer grid demands more squaring steps for the same displacement.
#[test]
fn the_automatic_count_scales_with_the_minimum_pixel_spacing() {
    let mut image = Image::from_vec(&[5, 3], vec![0.0; 15]).unwrap();
    image.set_spacing(&[1.0, 0.25]).unwrap();
    let geometry = Geometry::new(&image).unwrap();

    // maxnorm2 = 1 / 0.0625 = 16 → 2 + 0.5*log2(16) = 4.0 → 5.
    let unit = ramp_field([1.0; 5]);
    assert_eq!(automatic_number_of_iterations(&unit, &geometry, 2000), 5);
}

#[test]
fn the_exponential_of_a_zero_field_is_zero() {
    let geometry = unit_geometry(&[5, 3]);
    let zero = Field::zeros(&[5, 3]);
    assert_eq!(exponential(&zero, &geometry, true, 2000), zero);
}

/// Warping a constant field is the identity, so each squaring step exactly
/// doubles it: `N` steps undo the division by `2^N`. The equality is exact,
/// not approximate — halving and doubling are exact in binary floating point.
#[test]
fn the_exponential_of_a_constant_field_is_that_field() {
    let geometry = unit_geometry(&[5, 3]);
    let constant = ramp_field([1.0; 5]);
    // The automatic count is 3 here; the identity holds for any count.
    assert_eq!(exponential(&constant, &geometry, true, 2000), constant);
    for numiter in 0..6 {
        assert_eq!(exponential(&constant, &geometry, false, numiter), constant);
    }
}

/// `numiter == 0` takes the caster branch, which copies the input — the
/// first-order approximation `exp(u) = u`.
#[test]
fn the_exponential_with_no_squaring_step_is_the_identity() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([0.0, 0.5, 0.4, 0.3, 0.0]);
    assert_eq!(exponential(&input, &geometry, false, 0), input);
}

/// One squaring step on `u = [0, 0.5, 0.4, 0.3, 0]`, hand-computed.
///
/// `w = u/2 = [0, 0.25, 0.2, 0.15, 0]`, then `w + w ∘ (Id + w)`:
///
/// | x | w(x) | sample at | w there | w + sample |
/// |---|---|---|---|---|
/// | 0 | 0 | 0.00 | 0 | 0 |
/// | 1 | 0.25 | 1.25 | 0.75·0.25 + 0.25·0.2 = 0.2375 | 0.4875 |
/// | 2 | 0.2 | 2.20 | 0.8·0.2 + 0.2·0.15 = 0.19 | 0.39 |
/// | 3 | 0.15 | 3.15 | 0.85·0.15 + 0.15·0 = 0.1275 | 0.2775 |
/// | 4 | 0 | 4.00 | 0 | 0 |
#[test]
fn one_squaring_step_matches_the_hand_computed_composition() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([0.0, 0.5, 0.4, 0.3, 0.0]);
    let result = exponential(&input, &geometry, false, 1);

    let expected: Vec<f64> = [0.0, 0.4875, 0.39, 0.2775, 0.0]
        .iter()
        .copied()
        .cycle()
        .take(15)
        .collect();
    assert_close(&x_components(&result), &expected);
    // The y-component of a purely horizontal field stays zero.
    assert_close(
        &result
            .data
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>(),
        &[0.0; 15],
    );
}

/// More squaring steps converge on the true exponential, which for this field
/// is strictly less displaced than the first-order `u` where `u` shrinks along
/// its own direction of travel.
#[test]
fn more_squaring_steps_change_the_result() {
    let geometry = unit_geometry(&[5, 3]);
    let input = ramp_field([0.0, 0.5, 0.4, 0.3, 0.0]);

    let one = exponential(&input, &geometry, false, 1);
    let two = exponential(&input, &geometry, false, 2);
    let three = exponential(&input, &geometry, false, 3);

    assert_ne!(one, two);
    assert_ne!(two, three);
    // Successive refinements move less and less.
    let distance = |a: &Field, b: &Field| -> f64 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    };
    assert!(distance(&two, &three) < distance(&one, &two));
}
