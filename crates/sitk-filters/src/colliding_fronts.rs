//! Two fast-marching fronts run at each other; the output marks where they
//! collide.
//!
//! Port of `itk::CollidingFrontsImageFilter`
//! (`itkCollidingFrontsImageFilter.h` / `.hxx`), with the API surface
//! `CollidingFrontsImageFilter.yaml` declares.
//!
//! ## The pipeline
//!
//! Two [`crate::fast_marching_upwind_gradient`] marches over the *same* speed
//! image (the filter's input), each seeded on one point set and targeted at the
//! other, both with `GenerateGradientImage` on. The output is the pointwise
//! **dot product** of the two upwind gradient fields — upstream this is
//! `MultiplyImageFilter<GradientImage, GradientImage, OutputImage>`, and
//! `CovariantVector::operator*(const Self &)` is the scalar product. Where the
//! dot product is negative the two fronts travel in opposite directions, i.e.
//! the pixel lies between the seed sets.
//!
//! Both marches run with the level-set pixel type `float` (the yaml's
//! `output_pixel_type`) whatever the input's type, so the arrival times,
//! gradients and their product are all `float`; the output image is
//! [`PixelId::Float32`].
//!
//! `stop_on_targets` switches both marches from `NoTargets` to `AllTargets`,
//! each front stopping one `TargetOffset` (ITK's untouched default, `1.0`)
//! after every seed of the *other* front has been accepted. It is a speed
//! optimization: the fronts still meet, but the far field beyond the seeds is
//! left unmarched, with a zero gradient and hence a zero dot product there.
//!
//! ## Seeds, epsilon and connectivity
//!
//! After the multiply, *both* seed sets are overwritten in the dot-product
//! image with `negative_epsilon` (`-1e-6` by default). A seed's own gradient is
//! zero — it is accepted before any neighbor is alive — so its dot product
//! would otherwise be `0` and fall outside the negative region it anchors.
//!
//! With `apply_connectivity` (the default) the output keeps only the negative
//! component reachable from **`seed_points1`** — `seedList` is built from
//! `m_SeedPoints1` alone, so a negative region touching only `seed_points2`,
//! including the forced `negative_epsilon` at those very seeds, is dropped.
//! The flood fill is `FloodFilledImageFunctionConditionalConstIterator` over a
//! `BinaryThresholdImageFunction::ThresholdBelow(negative_epsilon)`, i.e. face
//! connectivity (`±1` per axis, no diagonals) over the *closed* condition
//!
//! ```text
//! NumericTraits<float>::NonpositiveMin() <= v && v <= negative_epsilon
//! ```
//!
//! so `v == negative_epsilon` is inside. Everything the fill does not reach
//! stays `0` (`AllocateInitialized`), including the positive regions. Without
//! `apply_connectivity` the dot-product image is grafted out whole, seed
//! overwrite included.
//!
//! ## Deviations
//!
//! - **Out-of-image seeds are an error** ([`FilterError::InvalidSeedIndex`]).
//!   `GenerateData` calls `multipliedImage->SetPixel(seedIndex, ...)` with no
//!   bounds check, which is undefined behavior in C++; over this crate's linear
//!   pixel buffer an out-of-range multi-index would alias a different in-bounds
//!   pixel rather than crash. The two marches, by contrast, would have silently
//!   dropped such a seed in `Initialize()`.
//! - **A seed shorter than the image dimension is an error**
//!   ([`FilterError::DimensionLength`]), matching `sitkSTLVectorToITK`'s
//!   "Unable to convert vector to ITK type". A `dim + 1`-th element is the
//!   seed's initial arrival time, as in
//!   [`crate::fast_marching_upwind_gradient`]; `CollidingFrontsImageFilter`
//!   exposes no `InitialTrialValues`, so that is the only way to offset a seed.
//! - With `stop_on_targets`, an empty *other* seed list is
//!   [`FilterError::NoTargetPoints`], from the marches'
//!   `VerifyTargetReachedModeConditions`. Front 1 is verified first, so an
//!   empty `seed_points2` is reported before an empty `seed_points1`. The two
//!   port-added seed checks above run ahead of both, since they gate the index
//!   conversion.

use std::collections::VecDeque;

use sitk_core::{Image, PixelId};

use crate::error::{FilterError, Result};
use crate::fast_marching::{
    MarchInput, TargetCondition, UpwindInput, large_value, march_flat, strides,
};
use crate::image_from_f64;

/// `m_TargetOffset`'s constructor default. `CollidingFrontsImageFilter` never
/// touches it, so the `AllTargets` marches run one time unit past the last
/// reached seed.
const TARGET_OFFSET: f64 = 1.0;

/// `m_NormalizationFactor`'s constructor default; likewise never touched.
const NORMALIZATION_FACTOR: f64 = 1.0;

/// The seed points of one front: their flat indices, and their initial arrival
/// times paired with them for the trial heap.
struct Seeds {
    trial: Vec<(usize, f64)>,
    targets: Vec<Option<usize>>,
}

fn prepare_seeds(points: &[Vec<u32>], size: &[usize], strides: &[usize]) -> Result<Seeds> {
    let dim = size.len();
    let mut trial = Vec::with_capacity(points.len());
    for point in points {
        if point.len() < dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: point.len(),
            });
        }
        if !point
            .iter()
            .take(dim)
            .zip(size)
            .all(|(&c, &e)| (c as usize) < e)
        {
            return Err(FilterError::InvalidSeedIndex {
                seed: point.iter().map(|&c| c as usize).collect(),
                size: size.to_vec(),
            });
        }
        let index: usize = point
            .iter()
            .take(dim)
            .zip(strides)
            .map(|(&c, &s)| c as usize * s)
            .sum();
        trial.push((index, point.get(dim).map_or(0.0, |&v| f64::from(v))));
    }
    let targets = trial.iter().map(|&(index, _)| Some(index)).collect();
    Ok(Seeds { trial, targets })
}

/// One `FastMarchingUpwindGradientImageFilter` of the pair: march `trial`,
/// target `targets`, and return the per-axis gradient buffers.
fn march_front(
    speed: &[f64],
    image: &Image,
    seeds: &Seeds,
    targets: &[Option<usize>],
    stop_on_targets: bool,
) -> Result<Vec<Vec<f64>>> {
    if stop_on_targets && targets.is_empty() {
        return Err(FilterError::NoTargetPoints);
    }
    let result = march_flat(
        MarchInput {
            size: image.size(),
            spacing: image.spacing(),
            speed,
            // The level-set type is the filter's `float` output type.
            narrow_to_f32: true,
            normalization_factor: NORMALIZATION_FACTOR,
            stopping_value: large_value(PixelId::Float32),
            collect_points: false,
            upwind: Some(UpwindInput {
                generate_gradient: true,
                targets,
                target_mode: if stop_on_targets {
                    TargetCondition::AllTargets
                } else {
                    TargetCondition::NoTargets
                },
                number_of_targets: 0,
                target_offset: TARGET_OFFSET,
            }),
        },
        &seeds.trial,
    )?;
    Ok(result.gradient)
}

/// `FloodFilledImageFunctionConditionalConstIterator` over
/// `BinaryThresholdImageFunction::ThresholdBelow(negative_epsilon)`, seeded on
/// front 1: the visited pixels keep their dot product, the rest stay `0`.
fn connected_negative_region(
    dot: &[f32],
    seeds: &[(usize, f64)],
    negative_epsilon: f32,
    size: &[usize],
    strides: &[usize],
) -> Vec<f64> {
    // `m_Lower <= value && value <= m_Upper`, with `m_Lower` left at
    // `NumericTraits<float>::NonpositiveMin()`.
    let included = |v: f32| -f32::MAX <= v && v <= negative_epsilon;

    let mut output = vec![0.0f64; dot.len()];
    let mut temporary = vec![0u8; dot.len()];
    let mut stack: VecDeque<usize> = VecDeque::new();

    // `GoToBegin()`: a seed enters the queue only if it is itself inside.
    for &(index, _) in seeds {
        if included(dot[index]) {
            stack.push_back(index);
            temporary[index] = 2;
        }
    }

    while let Some(index) = stack.pop_front() {
        output[index] = f64::from(dot[index]);
        for (axis, (&stride, &extent)) in strides.iter().zip(size).enumerate() {
            let coord = (index / stride) % size[axis];
            let neighbors = [
                (coord > 0).then(|| index - stride),
                (coord + 1 < extent).then_some(index + stride),
            ];
            for neighbor in neighbors.into_iter().flatten() {
                if temporary[neighbor] != 0 {
                    continue;
                }
                if included(dot[neighbor]) {
                    stack.push_back(neighbor);
                    temporary[neighbor] = 2;
                } else {
                    temporary[neighbor] = 1;
                }
            }
        }
    }
    output
}

/// `CollidingFrontsImageFilter`: the dot product of the upwind gradients of two
/// fast-marching fronts, negative between the two seed sets.
///
/// - `image` is the speed image shared by both marches; its geometry carries to
///   the output, whose pixel type is always [`PixelId::Float32`].
/// - `seed_points1` / `seed_points2` are image indices (`dim` elements,
///   optionally a `dim + 1`-th giving the seed's initial arrival time). Both
///   must be inside the image.
/// - `apply_connectivity` (ITK's default: `true`) keeps only the negative
///   region face-connected to `seed_points1`, zeroing everything else.
/// - `negative_epsilon` (ITK's default: `-1e-6`) is both the value forced onto
///   every seed and the inclusive upper bound of the connectivity threshold.
/// - `stop_on_targets` (ITK's default: `false`) stops each front once every
///   seed of the other front has been reached.
pub fn colliding_fronts(
    image: &Image,
    seed_points1: &[Vec<u32>],
    seed_points2: &[Vec<u32>],
    apply_connectivity: bool,
    negative_epsilon: f64,
    stop_on_targets: bool,
) -> Result<Image> {
    let size = image.size();
    let strides = strides(size);

    let seeds1 = prepare_seeds(seed_points1, size, &strides)?;
    let seeds2 = prepare_seeds(seed_points2, size, &strides)?;

    let speed = image.to_f64_vec();
    let gradient1 = march_front(&speed, image, &seeds1, &seeds2.targets, stop_on_targets)?;
    let gradient2 = march_front(&speed, image, &seeds2, &seeds1.targets, stop_on_targets)?;

    // `MultiplyImageFilter` over `CovariantVector::operator*`: the scalar
    // product, accumulated in the gradient's own `float` pixel type.
    let mut dot: Vec<f32> = (0..speed.len())
        .map(|i| {
            gradient1
                .iter()
                .zip(&gradient2)
                .fold(0.0f32, |sum, (a, b)| sum + (a[i] as f32) * (b[i] as f32))
        })
        .collect();

    let negative_epsilon32 = negative_epsilon as f32;
    for &(index, _) in seeds1.trial.iter().chain(&seeds2.trial) {
        dot[index] = negative_epsilon32;
    }

    let values = if apply_connectivity {
        connected_negative_region(&dot, &seeds1.trial, negative_epsilon32, size, &strides)
    } else {
        dot.iter().map(|&v| f64::from(v)).collect()
    };
    image_from_f64(PixelId::Float32, size, image, &values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ITK's constructor defaults.
    const NEGATIVE_EPSILON: f64 = -1e-6;

    fn speed(size: &[usize], fill: f64) -> Image {
        Image::from_vec(size, vec![fill; size.iter().product()]).unwrap()
    }

    fn run(image: &Image, s1: &[Vec<u32>], s2: &[Vec<u32>], connectivity: bool) -> Vec<f64> {
        colliding_fronts(image, s1, s2, connectivity, NEGATIVE_EPSILON, false)
            .unwrap()
            .to_f64_vec()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "pixel {i}: {a} != {e}");
        }
    }

    /// A straight seven-pixel corridor seeded at both ends. Front 1 has
    /// `grad T1 = +1` on every accepted pixel but its own seed; front 2 has
    /// `grad T2 = -1` on every accepted pixel but its own. The dot product is
    /// therefore `-1` strictly between the seeds and `0` at each seed (whose
    /// own gradient is zero) — which is exactly why both seeds are then forced
    /// to `negative_epsilon`.
    #[test]
    fn a_corridor_collides_between_its_two_seeds() {
        let eps = NEGATIVE_EPSILON;
        let expected = [eps, -1.0, -1.0, -1.0, -1.0, -1.0, eps];

        let image = speed(&[7, 1], 1.0);
        let seeds1 = [vec![0, 0]];
        let seeds2 = [vec![6, 0]];

        // Every pixel is negative, so connectivity keeps all of them.
        assert_close(&run(&image, &seeds1, &seeds2, true), &expected);
        assert_close(&run(&image, &seeds1, &seeds2, false), &expected);
    }

    /// Seeds inside the corridor: the fronts oppose only between them, and both
    /// gradients point the same way outside, giving `+1`.
    #[test]
    fn outside_the_seeds_the_fronts_agree_and_the_dot_product_is_positive() {
        let eps = NEGATIVE_EPSILON;
        let image = speed(&[7, 1], 1.0);
        let seeds1 = [vec![2, 0]];
        let seeds2 = [vec![4, 0]];

        assert_close(
            &run(&image, &seeds1, &seeds2, false),
            &[1.0, 1.0, eps, -1.0, eps, 1.0, 1.0],
        );
        // Connectivity keeps only the negative run reachable from seed 1.
        assert_close(
            &run(&image, &seeds1, &seeds2, true),
            &[0.0, 0.0, eps, -1.0, eps, 0.0, 0.0],
        );
    }

    /// A zero-speed wall keeps each front inside its own half, so every
    /// gradient product is zero and the only negative pixels are the two forced
    /// seeds. `seed_points2`'s speckle is not connected to `seed_points1`, and
    /// the flood fill — seeded on front 1 alone — drops it.
    #[test]
    fn apply_connectivity_drops_a_speckle_not_reachable_from_seed_points_1() {
        let eps = NEGATIVE_EPSILON;
        let mut data = vec![1.0f64; 5];
        data[2] = 0.0;
        let image = Image::from_vec(&[5, 1], data).unwrap();
        let seeds1 = [vec![0, 0]];
        let seeds2 = [vec![4, 0]];

        assert_close(
            &run(&image, &seeds1, &seeds2, false),
            &[eps, 0.0, 0.0, 0.0, eps],
        );
        assert_close(
            &run(&image, &seeds1, &seeds2, true),
            &[eps, 0.0, 0.0, 0.0, 0.0],
        );
    }

    /// The threshold is `value <= negative_epsilon`, closed at the boundary:
    /// an interior dot product of exactly `-1` is inside for
    /// `negative_epsilon == -1.0` and outside for anything below it.
    #[test]
    fn the_negative_epsilon_boundary_is_inclusive() {
        let image = speed(&[7, 1], 1.0);
        let seeds1 = [vec![0, 0]];
        let seeds2 = [vec![6, 0]];

        let at_boundary = colliding_fronts(&image, &seeds1, &seeds2, true, -1.0, false)
            .unwrap()
            .to_f64_vec();
        assert_close(&at_boundary, &[-1.0; 7]);

        // Just past it, only the seeds themselves (forced to -1.5) qualify, and
        // the fill cannot cross the -1 interior to reach seed 2.
        let past_boundary = colliding_fronts(&image, &seeds1, &seeds2, true, -1.5, false)
            .unwrap()
            .to_f64_vec();
        assert_close(&past_boundary, &[-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let ungated = colliding_fronts(&image, &seeds1, &seeds2, false, -1.5, false)
            .unwrap()
            .to_f64_vec();
        assert_close(&ungated, &[-1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.5]);
    }

    /// `StopOnTargets` leaves the far field beyond the seeds unmarched, so its
    /// gradient — and the dot product there — stays zero. The connected
    /// negative region is unchanged, which is the point: it is a speed
    /// optimization.
    #[test]
    fn stop_on_targets_truncates_the_far_field_but_not_the_collision() {
        let eps = NEGATIVE_EPSILON;
        let image = speed(&[9, 1], 1.0);
        let seeds1 = [vec![0, 0]];
        let seeds2 = [vec![4, 0]];

        let run_stop = |stop, connectivity| {
            colliding_fronts(&image, &seeds1, &seeds2, connectivity, eps, stop)
                .unwrap()
                .to_f64_vec()
        };

        // Front 1 stops one TargetOffset past x = 4, so x >= 6 keeps a zero
        // gradient; front 2 reaches everything either way.
        assert_close(
            &run_stop(false, false),
            &[eps, -1.0, -1.0, -1.0, eps, 1.0, 1.0, 1.0, 1.0],
        );
        assert_close(
            &run_stop(true, false),
            &[eps, -1.0, -1.0, -1.0, eps, 1.0, 0.0, 0.0, 0.0],
        );

        let connected = [eps, -1.0, -1.0, -1.0, eps, 0.0, 0.0, 0.0, 0.0];
        assert_close(&run_stop(false, true), &connected);
        assert_close(&run_stop(true, true), &connected);
    }

    /// A `dim + 1`-th element is the seed's initial arrival time, not a third
    /// index. It shifts one front's arrival field by a constant, which the
    /// gradients — and so the output — cannot see.
    #[test]
    fn a_trailing_seed_value_is_accepted_and_shifts_only_the_arrival_field() {
        let image = speed(&[7, 1], 1.0);
        let plain = run(&image, &[vec![0, 0]], &[vec![6, 0]], false);
        let offset = run(&image, &[vec![0, 0, 2]], &[vec![6, 0]], false);
        assert_close(&offset, &plain);
    }

    #[test]
    fn empty_seed_lists_march_nothing_and_leave_a_zero_image() {
        let image = speed(&[5, 1], 1.0);
        assert_close(&run(&image, &[], &[], true), &[0.0; 5]);
        assert_close(&run(&image, &[], &[], false), &[0.0; 5]);

        // Only front 2 seeded: its gradient multiplies front 1's zero field, so
        // the sole negative pixel is the forced seed.
        let one_sided = run(&image, &[], &[vec![4, 0]], false);
        assert_close(&one_sided, &[0.0, 0.0, 0.0, 0.0, NEGATIVE_EPSILON]);
        // ...and with no front-1 seed there is nothing to flood from.
        assert_close(&run(&image, &[], &[vec![4, 0]], true), &[0.0; 5]);
    }

    /// `AllTargets` demands at least one target point, and front 1 — whose
    /// targets are `seed_points2` — is verified first.
    #[test]
    fn stop_on_targets_needs_both_seed_lists() {
        let image = speed(&[5, 1], 1.0);
        for (s1, s2) in [
            (vec![vec![0, 0]], vec![]),
            (vec![], vec![vec![4, 0]]),
            (vec![], vec![]),
        ] {
            assert_eq!(
                colliding_fronts(&image, &s1, &s2, true, NEGATIVE_EPSILON, true).err(),
                Some(FilterError::NoTargetPoints),
            );
        }
        assert!(
            colliding_fronts(
                &image,
                &[vec![0, 0]],
                &[vec![4, 0]],
                true,
                NEGATIVE_EPSILON,
                true,
            )
            .is_ok()
        );
    }

    #[test]
    fn an_out_of_bounds_seed_is_an_error() {
        let image = speed(&[5, 3], 1.0);
        let expected = |seed: Vec<usize>| {
            Some(FilterError::InvalidSeedIndex {
                seed,
                size: vec![5, 3],
            })
        };
        assert_eq!(
            colliding_fronts(&image, &[vec![5, 0]], &[vec![0, 0]], true, -1e-6, false).err(),
            expected(vec![5, 0]),
        );
        assert_eq!(
            colliding_fronts(&image, &[vec![0, 0]], &[vec![0, 3]], true, -1e-6, false).err(),
            expected(vec![0, 3]),
        );
    }

    #[test]
    fn a_seed_shorter_than_the_image_dimension_is_an_error() {
        let image = speed(&[5, 3], 1.0);
        let short = || {
            Some(FilterError::DimensionLength {
                expected: 2,
                got: 1,
            })
        };
        assert_eq!(
            colliding_fronts(&image, &[vec![0]], &[vec![0, 0]], true, -1e-6, false).err(),
            short(),
        );
        assert_eq!(
            colliding_fronts(&image, &[vec![0, 0]], &[vec![0]], true, -1e-6, false).err(),
            short(),
        );
    }

    #[test]
    fn the_output_is_float32_with_the_input_geometry() {
        let mut image = speed(&[5, 1], 1.0);
        image.set_spacing(&[2.0, 1.0]).unwrap();
        image.set_origin(&[-1.0, 4.0]).unwrap();
        let out = colliding_fronts(
            &image,
            &[vec![0, 0]],
            &[vec![4, 0]],
            true,
            NEGATIVE_EPSILON,
            false,
        )
        .unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.spacing(), image.spacing());
        assert_eq!(out.origin(), image.origin());
    }
}
