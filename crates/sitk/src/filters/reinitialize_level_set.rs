//! `ReinitializeLevelSetImageFilter`: recompute a signed distance function from
//! a level set's zero (or `level_set_value`) crossing.
//!
//! Ported from `itkReinitializeLevelSetImageFilter.h/.hxx` and the
//! `itkLevelSetNeighborhoodExtractor.h/.hxx` it drives. The fast-marching half
//! reuses [`crate::filters::fast_marching`]'s solver through its `march_flat` seam.
//!
//! # The pipeline
//!
//! 1. `LevelSetNeighborhoodExtractor::Locate` walks every pixel and, wherever
//!    the shifted level set `phi - level_set_value` changes sign along a grid
//!    line, linearly interpolates the distance to the crossing:
//!    `d_j = center / (center - neighbor) * spacing[j]`, keeping the smaller of
//!    the two sides of axis `j`. The pixel's distance is then the distance to
//!    the plane through those per-axis crossings,
//!    `sqrt(1 / sum_j d_j^-2)`, and the pixel joins the *inside* list when
//!    `phi - level_set_value <= 0` and the *outside* list otherwise.
//! 2. `FastMarchingImageFilter` marches outward from the outside list at unit
//!    speed; every pixel strictly above the level set takes its arrival time.
//! 3. The same marcher restarts from the inside list; every pixel at or below
//!    the level set takes the *negation* of its arrival time.
//!
//! The result is negative inside, positive outside, and `|grad| == 1` to the
//! accuracy of the first-order upwind scheme. Note that the output is the
//! distance to the `level_set_value` isocontour, not that distance re-offset by
//! `level_set_value`.
//!
//! # Fixed upstream bug
//!
//! * **A pixel that lies exactly on the level set no longer neutralises its
//!   neighbours.** `CalculateDistance` still short-circuits a pixel whose
//!   shifted value is exactly zero into the *inside* list at distance `0` —
//!   that much is correct, and consistent with the `inside = (center <= 0)`
//!   convention used everywhere else. Upstream's bug was in the *neighbour*
//!   sign test, which was strict (`neighValue > 0` / `neighValue < 0`): a
//!   neighbour sitting exactly on the contour is geometrically a crossing no
//!   matter which side the center is on (`d = center / (center - 0) * spacing
//!   == center`, always in range), so this port's sign test is non-strict
//!   (`neighValue >= 0` / `neighValue <= 0`) and accepts it. A level set whose
//!   zero contour lands exactly on grid pixels now seeds the march from every
//!   pixel adjacent to the contour, recovering the exact distance instead of
//!   starving the outward march and leaving the whole outside at the
//!   marcher's `m_LargeValue`.
//!
//! # Upstream behaviour reproduced here
//!
//! * **A pixel with no sign change anywhere joins neither list.** Its
//!   accumulator stays zero, `CalculateDistance` returns early, and the pixel is
//!   simply not a seed. This is what confines the seeds to the crossing's
//!   immediate neighbourhood.
//!
//! * **`InputNarrowBandwidth` is inert.** It reaches the locator only through
//!   `SetInputNarrowBand`, which SimpleITK does not expose, so
//!   `GenerateDataNarrowBand` always takes the `m_Locator->NarrowBandingOff()`
//!   branch. The parameter is kept for API parity with the yaml.
//!
//! * **Narrow banding leaves the far field at `+/- NumericTraits::max()`,** not
//!   at the marcher's large value: `GenerateDataNarrowBand` pre-fills the output
//!   with `max()` outside and `NonpositiveMin()` inside and then overwrites only
//!   the points the truncated march actually processed.

use crate::core::{Image, PixelId};

use crate::filters::error::{FilterError, Result};
use crate::filters::fast_marching::{MarchInput, large_value, march_flat, strides};
use crate::filters::image_from_f64;

/// `FastMarchingImageFilter::m_NormalizationFactor`, left at its constructor
/// default: `ReinitializeLevelSetImageFilter` never calls the setter.
const NORMALIZATION_FACTOR: f64 = 1.0;

/// `FastMarchingImageFilter::m_SpeedConstant`, likewise untouched. The marcher
/// is given no speed image, so it solves `|grad T| = 1`.
const SPEED_CONSTANT: f64 = 1.0;

/// `static_cast<PixelType>(v)` for the level set's pixel type. The extractor
/// stores its interpolated distances in `PixelType` variables, so the narrowing
/// is observable through the comparisons that follow.
fn cast(v: f64, to_f32: bool) -> f64 {
    if to_f32 { v as f32 as f64 } else { v }
}

/// `NumericTraits<PixelType>::max()`, which is both the extractor's
/// `m_LargeValue` and the narrow-band output's `posInfinity`. (The marcher's
/// own large value is *half* this — see [`large_value`].)
fn type_max(id: PixelId) -> f64 {
    if id == PixelId::Float32 {
        f32::MAX as f64
    } else {
        f64::MAX
    }
}

/// `ReinitializeLevelSetImageFilter`: replace the input level set with the
/// approximated signed distance function to its `level_set_value` isocontour,
/// negative inside and positive outside.
///
/// - `level_set_value` selects the isocontour to reinitialise about (ITK's
///   default `0.0`).
/// - `narrow_banding` restricts the output to a band around that isocontour.
///   Pixels the truncated march never processes keep
///   `+/- NumericTraits<PixelType>::max()`.
/// - `input_narrow_bandwidth` is accepted for parity with SimpleITK's yaml but
///   has no effect: it is only consulted when an *input* narrow band container
///   is supplied, which SimpleITK's API cannot do.
/// - `output_narrow_bandwidth` sets the marcher's stopping value to
///   `output_narrow_bandwidth / 2 + 2` when `narrow_banding` is on; it is
///   ignored otherwise. ITK's default is `12.0`.
///
/// The output has the input's pixel type and geometry. `RealPixelIDTypeList`:
/// the input must be [`PixelId::Float32`] or [`PixelId::Float64`].
///
/// See the [module docs](self) for the pipeline, for the upstream bug fixed
/// here (an isocontour that passes exactly through grid pixels no longer
/// starves the outward march), and for the upstream quirks still reproduced.
pub fn reinitialize_level_set(
    image: &Image,
    level_set_value: f64,
    narrow_banding: bool,
    input_narrow_bandwidth: f64,
    output_narrow_bandwidth: f64,
) -> Result<Image> {
    let pixel_id = image.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    // `m_InputNarrowBandwidth` only ever reaches `m_Locator` alongside
    // `SetInputNarrowBand`, which SimpleITK does not expose.
    let _ = input_narrow_bandwidth;

    let size = image.size();
    let spacing = image.spacing();
    let input = image.to_f64_vec()?;
    let to_f32 = pixel_id == PixelId::Float32;
    let large = large_value(pixel_id);

    let seeds = locate(size, spacing, &input, level_set_value, pixel_id);

    // The marcher is reused for both directions; only its trial points change.
    let speed = vec![SPEED_CONSTANT; input.len()];
    let stopping_value = if narrow_banding {
        // `GenerateDataNarrowBand`: `(m_OutputNarrowBandwidth / 2.0) + 2.0`.
        (output_narrow_bandwidth / 2.0) + 2.0
    } else {
        // `m_StoppingValue`'s constructor default, `double(m_LargeValue)`.
        large
    };
    let march = |trial: &[(usize, f64)]| {
        march_flat(
            MarchInput {
                size,
                spacing,
                speed: &speed,
                narrow_to_f32: to_f32,
                normalization_factor: NORMALIZATION_FACTOR,
                stopping_value,
                collect_points: narrow_banding,
                upwind: None,
            },
            trial,
        )
    };

    let outward = march(&seeds.outside)?;
    let inward = march(&seeds.inside)?;

    let output = if narrow_banding {
        let mut output: Vec<f64> = input
            .iter()
            .map(|&v| {
                // `negInfinity` is `NonpositiveMin()`, i.e. `-max()`.
                if v - level_set_value <= 0.0 {
                    -type_max(pixel_id)
                } else {
                    type_max(pixel_id)
                }
            })
            .collect();
        for &p in &outward.processed {
            if input[p] - level_set_value > 0.0 {
                output[p] = outward.values[p];
            }
        }
        for &p in &inward.processed {
            if input[p] - level_set_value <= 0.0 {
                output[p] = -inward.values[p];
            }
        }
        output
    } else {
        // `GenerateDataFull` writes every pixel through exactly one of the two
        // complementary branches, so no pre-fill is needed.
        input
            .iter()
            .enumerate()
            .map(|(p, &v)| {
                if v - level_set_value > 0.0 {
                    outward.values[p]
                } else {
                    -inward.values[p]
                }
            })
            .collect()
    };

    image_from_f64(pixel_id, size, image, &output)
}

/// `LevelSetNeighborhoodExtractor`'s two output containers, as flat
/// `(index, interpolated distance)` trial points in raster order.
struct Seeds {
    inside: Vec<(usize, f64)>,
    outside: Vec<(usize, f64)>,
}

/// `LevelSetNeighborhoodExtractor::GenerateDataFull`, i.e.
/// `CalculateDistance` at every pixel in raster order.
fn locate(
    size: &[usize],
    spacing: &[f64],
    input: &[f64],
    level_set_value: f64,
    pixel_id: PixelId,
) -> Seeds {
    let dim = size.len();
    let strides = strides(size);
    let to_f32 = pixel_id == PixelId::Float32;
    // `m_LargeValue = NumericTraits<PixelType>::max()` — the extractor's, which
    // is twice the marcher's.
    let large = type_max(pixel_id);

    let mut seeds = Seeds {
        inside: Vec::new(),
        outside: Vec::new(),
    };
    let mut nodes_used = vec![0.0f64; dim];

    for index in 0..input.len() {
        // `centerValue` is a `PixelType` variable: the subtraction narrows.
        let center = cast(input[index] - level_set_value, to_f32);

        if center == 0.0 {
            seeds.inside.push((index, 0.0));
            continue;
        }
        let inside = center <= 0.0;

        for (j, node) in nodes_used.iter_mut().enumerate() {
            *node = large;
            let coord = (index / strides[j]) % size[j];
            let base = index - coord * strides[j];

            for neighbor in [coord.checked_sub(1), Some(coord + 1)]
                .into_iter()
                .flatten()
            {
                if neighbor >= size[j] {
                    continue;
                }
                let neighbor_value = cast(
                    input[base + neighbor * strides[j]] - level_set_value,
                    to_f32,
                );

                if (neighbor_value >= 0.0 && inside) || (neighbor_value <= 0.0 && !inside) {
                    // Both the numerator and the denominator carry the center's
                    // sign, so the interpolated distance is positive.
                    let distance = center / (center - neighbor_value) * spacing[j];
                    if *node > distance {
                        // `neighNode.SetValue(distance)` narrows to `PixelType`.
                        *node = cast(distance, to_f32);
                    }
                }
            }
        }

        // "The final distance is given by the minimum distance to the plane
        // crossing formed by the zero set crossing points."
        nodes_used.sort_by(f64::total_cmp);
        let mut accumulator = 0.0f64;
        for &node in &nodes_used {
            if node >= large {
                break;
            }
            accumulator += 1.0 / (node * node);
        }
        if accumulator == 0.0 {
            // No sign change on any axis: the pixel seeds neither march.
            continue;
        }

        let distance = cast((1.0 / accumulator).sqrt(), to_f32);
        if inside {
            seeds.inside.push((index, distance));
        } else {
            seeds.outside.push((index, distance));
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `phi(x, y) = x - offset` on a `w x 1` grid: a vertical zero contour, so
    /// every axis-1 neighbour is out of bounds and the pipeline is exactly 1-D.
    fn ramp(w: usize, offset: f64) -> Image {
        let data: Vec<f64> = (0..w).map(|x| x as f64 - offset).collect();
        Image::from_vec(&[w, 1], data).unwrap()
    }

    fn reinit(image: &Image, level_set_value: f64) -> Vec<f64> {
        reinitialize_level_set(image, level_set_value, false, 12.0, 12.0)
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

    // ---- The interpolated distance, hand-derived ---------------------------

    /// A ramp whose zero crossing falls midway between `x = 2` and `x = 3` is
    /// already a signed distance function, and reinitialising it reproduces it
    /// exactly.
    ///
    /// The two straddling pixels have `center = -/+0.5` and a neighbour of the
    /// opposite sign at `+/-0.5`, so `d = center / (center - neighbor) * 1 =
    /// 0.5` for both. Marching outward from `x = 3` at unit speed gives
    /// `0.5, 1.5, 2.5`; inward from `x = 2` gives the same, negated.
    #[test]
    fn an_exact_signed_distance_ramp_is_reproduced_exactly() {
        assert_close(
            &reinit(&ramp(6, 2.5), 0.0),
            &[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        );
    }

    /// Cubing the level set moves every value but not the zero set, and the
    /// interpolation at the crossing is symmetric (`-0.125` against `+0.125`),
    /// so it recovers exactly the same `0.5` and the output is unchanged.
    /// The distorted values at `x <= 1` and `x >= 4` never enter: those pixels
    /// have no sign change and seed nothing.
    #[test]
    fn a_cubed_level_set_recovers_the_true_distances() {
        let cubed = Image::from_vec(
            &[6, 1],
            (0..6).map(|x| (x as f64 - 2.5).powi(3)).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_close(&reinit(&cubed, 0.0), &[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]);
    }

    /// The interpolation is asymmetric when the crossing is not midway: a
    /// crossing a quarter-pixel from `x = 2` gives `d(2) = 0.25` and
    /// `d(3) = 0.75`, and the march steps by the spacing from there.
    #[test]
    fn an_off_center_crossing_splits_the_pixel_asymmetrically() {
        let image = ramp(6, 2.25);
        // phi = [-2.25, -1.25, -0.25, 0.75, 1.75, 2.75]
        // d(2) = -0.25 / (-0.25 - 0.75) = 0.25;  d(3) = 0.75 / (0.75 + 0.25) = 0.75
        assert_close(
            &reinit(&image, 0.0),
            &[-2.25, -1.25, -0.25, 0.75, 1.75, 2.75],
        );
    }

    /// `d_j` scales with `spacing[j]`, and so does every march step.
    #[test]
    fn spacing_scales_the_interpolation_and_the_march() {
        let mut image = ramp(6, 2.5);
        image.set_spacing(&[2.0, 1.0]).unwrap();
        // d = 0.5 * 2 = 1.0, then the march steps by h = 2.
        assert_close(&reinit(&image, 0.0), &[-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]);
        assert_eq!(
            reinitialize_level_set(&image, 0.0, false, 12.0, 12.0)
                .unwrap()
                .spacing(),
            &[2.0, 1.0]
        );
    }

    // ---- LevelSetValue ------------------------------------------------------

    /// A non-zero `level_set_value` relocates the contour, and the output is the
    /// distance to *that* contour — not shifted back by `level_set_value`.
    #[test]
    fn a_non_zero_level_set_value_shifts_the_contour() {
        // phi - 1 = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5]; the crossing is between
        // x = 3 and x = 4.
        assert_close(
            &reinit(&ramp(6, 2.5), 1.0),
            &[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5],
        );
    }

    /// Sign is decided by `phi - level_set_value` alone, at every pixel.
    #[test]
    fn the_sign_of_the_output_follows_the_shifted_input() {
        for lsv in [-1.0, 0.0, 1.0] {
            let image = ramp(6, 2.5);
            let input = image.to_f64_vec().unwrap();
            for (i, &v) in reinit(&image, lsv).iter().enumerate() {
                if input[i] - lsv > 0.0 {
                    assert!(v > 0.0, "lsv {lsv}, pixel {i}: {v}");
                } else {
                    assert!(v <= 0.0, "lsv {lsv}, pixel {i}: {v}");
                }
            }
        }
    }

    /// The signed distance to a circle of radius `radius` centred at `(c, c)` on
    /// an `n x n` grid.
    fn circle(n: usize, c: f64, radius: f64) -> Vec<f64> {
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f64 - c, y as f64 - c);
                data[x + n * y] = (dx * dx + dy * dy).sqrt() - radius;
            }
        }
        data
    }

    /// The largest `|before - after|` among pixels within two pixels of the
    /// contour, after asserting that no pixel changed side.
    fn worst_deviation_near_the_contour(n: usize, data: &[f64]) -> f64 {
        let image = Image::from_vec(&[n, n], data.to_vec()).unwrap();
        let out = reinit(&image, 0.0);

        let mut worst = 0.0f64;
        for (i, (&before, &after)) in data.iter().zip(&out).enumerate() {
            assert_eq!(before > 0.0, after > 0.0, "pixel {i} changed side");
            if before.abs() <= 2.0 {
                worst = worst.max((before - after).abs());
            }
        }
        worst
    }

    /// A 2-D circle that is already a signed distance function: reinitialising
    /// it is near-identity near the contour, to the first-order upwind scheme's
    /// accuracy.
    ///
    /// The centre is offset by half a pixel so that no grid pixel lands exactly
    /// on the contour — `(2m+1)^2 + (2n+1)^2 == 256` has no solution, since a
    /// sum of two odd squares is `2 mod 4`. The observed worst deviation is
    /// `0.1414`; see
    /// [`exact_zeros_on_the_contour_recover_their_neighbours_exactly`] for what
    /// happens when a pixel *does* land on it.
    #[test]
    fn a_circle_is_near_identity_inside_the_band() {
        let worst = worst_deviation_near_the_contour(34, &circle(34, 16.5, 8.0));
        assert!(
            (0.14..0.15).contains(&worst),
            "worst deviation near the contour: {worst}"
        );
    }

    /// The same circle centred on a pixel puts four pixels exactly on the
    /// contour, one per axis. `CalculateDistance` files each as an inside point
    /// at distance `0`, and its non-strict sign test now accepts that `0` as a
    /// crossing for every neighbour straddling it, at `d = center /
    /// (center - 0) * 1 == center` — i.e. exactly the true distance, since each
    /// of those neighbours sits one grid step from the axis point. All twelve
    /// pixels adjacent to (and including) the four axis points recover their
    /// true distance bit-for-bit.
    ///
    /// The worst deviation near the contour drops from `0.7699` (the old
    /// starved value at `(16, 7)`) to `0.1581`, at `(11, 9)` — a generic
    /// first-order upwind discretisation artefact unrelated to any exact-zero
    /// pixel, in the same magnitude class as
    /// [`a_circle_is_near_identity_inside_the_band`]'s `0.1414` baseline for a
    /// contour with no exact zeros at all.
    #[test]
    fn exact_zeros_on_the_contour_recover_their_neighbours_exactly() {
        let n = 33usize;
        let data = circle(n, 16.0, 8.0);
        assert_eq!(data.iter().filter(|&&v| v == 0.0).count(), 4);

        let worst = worst_deviation_near_the_contour(n, &data);
        assert!(
            (0.15..0.17).contains(&worst),
            "worst deviation near the contour: {worst}"
        );

        // The four axis points and their immediate neighbours recover their
        // true distance exactly.
        let out = reinit(&Image::from_vec(&[n, n], data.clone()).unwrap(), 0.0);
        for &(x, y) in &[
            (24usize, 16usize),
            (23, 16),
            (25, 16),
            (8, 16),
            (7, 16),
            (9, 16),
            (16, 24),
            (16, 23),
            (16, 25),
            (16, 8),
            (16, 7),
            (16, 9),
        ] {
            let i = x + n * y;
            assert_eq!(out[i], data[i], "pixel ({x}, {y})");
        }
    }

    // ---- Narrow banding ------------------------------------------------------

    /// `stopping_value = output_narrow_bandwidth / 2 + 2`. With a bandwidth of
    /// `6` the march halts on the first popped value above `5.0`, so pixels
    /// whose arrival time would be `5.5` or more are never *processed* and keep
    /// the pre-fill — `+/- NumericTraits<double>::max()`, not the marcher's
    /// large value.
    #[test]
    fn narrow_banding_leaves_the_far_field_at_the_type_maximum() {
        let image = ramp(16, 7.5);
        let out = reinitialize_level_set(&image, 0.0, true, 12.0, 6.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        // Inside: x = 7 seeds at 0.5, marching down to x = 3 at 4.5.
        assert_close(&out[3..8], &[-4.5, -3.5, -2.5, -1.5, -0.5]);
        // Outside: x = 8 seeds at 0.5, marching up to x = 12 at 4.5.
        assert_close(&out[8..13], &[0.5, 1.5, 2.5, 3.5, 4.5]);
        // Beyond the band: untouched pre-fill.
        assert_eq!(&out[0..3], &[-f64::MAX; 3]);
        assert_eq!(&out[13..16], &[f64::MAX; 3]);
    }

    /// A wide enough band reaches the whole image, and then narrow banding and
    /// the full pipeline agree pixel for pixel.
    #[test]
    fn a_band_wider_than_the_image_matches_the_full_pipeline() {
        let image = ramp(16, 7.5);
        let banded = reinitialize_level_set(&image, 0.0, true, 12.0, 100.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_close(&banded, &reinit(&image, 0.0));
    }

    /// `output_narrow_bandwidth` is ignored when `narrow_banding` is off, and
    /// `input_narrow_bandwidth` is ignored either way.
    #[test]
    fn the_inert_bandwidths_do_not_change_the_output() {
        let image = ramp(16, 7.5);
        let baseline = reinit(&image, 0.0);
        for bandwidth in [0.0, 1.0, 12.0, 1.0e6] {
            let full = reinitialize_level_set(&image, 0.0, false, bandwidth, bandwidth)
                .unwrap()
                .to_f64_vec()
                .unwrap();
            assert_close(&full, &baseline);
        }
        let banded_a = reinitialize_level_set(&image, 0.0, true, 0.0, 6.0).unwrap();
        let banded_b = reinitialize_level_set(&image, 0.0, true, 1.0e6, 6.0).unwrap();
        assert_close(
            &banded_a.to_f64_vec().unwrap(),
            &banded_b.to_f64_vec().unwrap(),
        );
    }

    // ---- Contour-on-grid handling ----------------------------------------------

    /// A contour landing exactly on grid pixels: `CalculateDistance` files the
    /// zero pixel as *inside* at distance `0`. Its non-strict sign test now
    /// accepts that zero as a sign change for the outside neighbour too, so
    /// `x = 4` (`center = 1`) seeds the outward march at
    /// `d = 1 / (1 - 0) * 1 = 1`, and marching outward from there at unit speed
    /// recovers the exact distance for the rest of the ramp.
    #[test]
    fn a_contour_on_the_grid_recovers_the_exact_outward_distance() {
        let out = reinit(&ramp(7, 3.0), 0.0);

        assert_close(&out[0..4], &[-3.0, -2.0, -1.0, 0.0]);
        assert_close(&out[4..7], &[1.0, 2.0, 3.0]);
    }

    /// The same recovery seen from the `level_set_value` side: shifting the
    /// contour onto a pixel reproduces the same shifted ramp.
    #[test]
    fn a_level_set_value_that_lands_on_a_pixel_recovers_the_exact_distance_too() {
        let out = reinit(&ramp(7, 2.5), 0.5);

        // phi - 0.5 = [-3, -2, -1, 0, 1, 2, 3]
        assert_close(&out[0..4], &[-3.0, -2.0, -1.0, 0.0]);
        assert_close(&out[4..7], &[1.0, 2.0, 3.0]);
    }

    /// A level set with no sign change anywhere seeds neither march, so both
    /// arrival fields stay at the large value and the output is `+/- large`
    /// according to the input's sign.
    #[test]
    fn a_constant_level_set_leaves_both_fields_at_the_large_value() {
        let large = large_value(PixelId::Float64);

        let positive = Image::from_vec(&[4, 1], vec![3.0f64; 4]).unwrap();
        assert_eq!(reinit(&positive, 0.0), vec![large; 4]);

        let negative = Image::from_vec(&[4, 1], vec![-3.0f64; 4]).unwrap();
        assert_eq!(reinit(&negative, 0.0), vec![-large; 4]);
    }

    // ---- Types and error paths -------------------------------------------------

    /// The output keeps the input's pixel type, and a `Float32` level set
    /// narrows every stored distance — including the marcher's large value.
    #[test]
    fn float32_input_stays_float32_throughout() {
        let data: Vec<f32> = (0..6).map(|x| x as f32 - 2.5).collect();
        let image = Image::from_vec(&[6, 1], data).unwrap();
        let out = reinitialize_level_set(&image, 0.0, false, 12.0, 12.0).unwrap();

        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_close(
            &out.to_f64_vec().unwrap(),
            &[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        );

        // A level set with no sign change seeds no march on either side, so it
        // keeps the large value — the `float` one, not the `double` one — and
        // the narrow-band pre-fill (below) is `float`'s maximum.
        let no_crossing = Image::from_vec(&[4, 1], vec![3.0f32; 4]).unwrap();
        let starved = reinitialize_level_set(&no_crossing, 0.0, false, 12.0, 12.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(starved, vec![large_value(PixelId::Float32); 4]);

        let banded = reinitialize_level_set(&image, 0.0, true, 12.0, 0.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(banded[0], -(f32::MAX as f64));
        assert_eq!(banded[5], f32::MAX as f64);
    }

    /// `pixel_types: RealPixelIDTypeList`.
    #[test]
    fn an_integer_level_set_is_rejected() {
        let image = Image::from_vec(&[4, 1], vec![0u8, 1, 2, 3]).unwrap();
        assert_eq!(
            reinitialize_level_set(&image, 0.0, false, 12.0, 12.0).err(),
            Some(FilterError::RequiresRealPixelType(PixelId::UInt8))
        );

        let signed = Image::from_vec(&[4, 1], vec![-1i16, 0, 1, 2]).unwrap();
        assert_eq!(
            reinitialize_level_set(&signed, 0.0, false, 12.0, 12.0).err(),
            Some(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
    }
}
