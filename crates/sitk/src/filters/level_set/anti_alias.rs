//! `AntiAliasBinaryImageFilter`, ported from
//! `itkAntiAliasBinaryImageFilter.h/.hxx`.

use crate::core::Image;

use super::function::{CurvatureFlowFunction, DifferenceFunction};
use super::{LevelSetResult, SolverSetup, SparseFieldSolver, UpdateRule};
use crate::filters::canny::zero_crossing_values;
use crate::filters::error::{FilterError, Result};
use crate::filters::{image_from_f64, real_pixel_id};

/// `AntiAliasBinaryImageFilter`: fit a smooth surface to a binary volume by
/// letting its interface flow under curvature without ever crossing the
/// original binary interface.
///
/// The isosurface value is halfway between the input's minimum and maximum,
/// `max - (max - min) / 2`, so it lands between the two binary values whatever
/// they are. The level set starts at the input shifted by that value and
/// evolves under `CurvatureFlowFunction` — a pure `kappa |grad(phi)|` update at
/// the constant time step `0.05` — inside the sparse-field solver. Each active
/// pixel's new value is then clamped by
/// `AntiAliasBinaryImageFilter::CalculateUpdateValue` (hxx:59-75):
///
/// ```text
/// u^{n+1} = max(u^n + dt * H, 0)   where the input pixel equals the maximum
///         = min(u^n + dt * H, 0)   otherwise
/// ```
///
/// so a pixel that started inside can never end up outside, and vice versa.
/// The comparison against the maximum is exact, so a pixel matching neither
/// binary value of a not-really-binary input takes the `min(.., 0)` branch
/// along with the minimum.
///
/// The output is the **level set**, not a re-thresholded binary image: its
/// zero crossings are the fitted surface, values *inside* it are positive and
/// values *outside* are negative. (This is the opposite of the sign convention
/// the five [segmentation level-set filters](crate::filters::level_set) use.) Pixels
/// away from the sparse field are flattened to `+/- (number_of_layers + 1)`.
///
/// Iteration stops after `number_of_iterations`, or as soon as an iteration's
/// RMS change falls strictly below `maximum_rms_error`. Argument order follows
/// SimpleITK's `AntiAliasBinaryImageFilter.yaml`; its defaults are
/// `maximum_rms_error = 0.07` and `number_of_iterations = 1000`.
///
/// # Upstream behaviour reproduced here
///
/// * **`UseImageSpacing` is off.** The constructor calls
///   `SetUseImageSpacing(false)`, so the derivatives are taken in index space
///   and `m_ConstantGradientValue` is `1.0` regardless of the image's spacing.
///
/// * **The constructor's layer ladder is the identity.** It writes
///   `SetNumberOfLayers(2)` for a 2-D input, `(3)` for 3-D and
///   `(ImageDimension)` otherwise, which is `ImageDimension` in all three arms
///   — the same count `SegmentationLevelSetImageFilter` picks. (`GenerateData`
///   nevertheless warns for `dim > 3` that "only 3 layers are being used".)
///
/// * **A constant input has no interface to preserve.** Minimum and maximum
///   coincide, the shifted level set is zero everywhere, and the zero-crossing
///   filter finds no active layer. `InitializeBackgroundPixels` signs the whole
///   image on `shifted > 0`, which a zero never satisfies, so the output is
///   uniformly *negative* — even for pixels whose value is the maximum. The
///   solver then halts after one no-op iteration with an RMS change of zero.
///
/// * **The output pixel type is `NumericTraits<InputPixelType>::RealType`** —
///   `Float64` for every pixel type this filter accepts. The solver itself
///   runs in `f64`.
///
/// # Errors
///
/// [`FilterError::RequiresIntegerPixelType`] for a floating-point input:
/// `AntiAliasBinaryImageFilter.yaml` declares `pixel_types:
/// IntegerPixelIDTypeList`, so SimpleITK never instantiates this filter for
/// `Float32`/`Float64`.
pub fn anti_alias_binary(
    image: &Image,
    maximum_rms_error: f64,
    number_of_iterations: u32,
) -> Result<LevelSetResult> {
    if image.pixel_id().is_floating_point() {
        return Err(FilterError::RequiresIntegerPixelType(image.pixel_id()));
    }
    let input = image.to_f64_vec()?;
    let dim = image.dimension();

    // `MinimumMaximumImageCalculator` over the input, then
    // "IsoSurface value is halfway between minimum and maximum."
    let min = input.iter().copied().fold(f64::INFINITY, f64::min);
    let max = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let iso_surface_value = max - ((max - min) / 2.0);

    // `CopyInputToOutput`: shift by the iso-surface value, then graft the
    // shifted image's zero-crossing map onto the output.
    let shifted: Vec<f64> = input.iter().map(|&v| v - iso_surface_value).collect();
    let mut shifted_image = Image::from_vec(image.size(), shifted.clone())?;
    shifted_image.copy_geometry_from(image);
    let zero_crossings = zero_crossing_values(&shifted_image, 0.0, 1.0)?;

    let solver = SparseFieldSolver::new(
        image.size(),
        image.spacing(),
        SolverSetup {
            shifted,
            zero_crossings,
            // `SetUseImageSpacing(false)` leaves every `ScaleCoefficient` at
            // `1.0`, and `CurvatureFlowFunction`'s radius is `1` on every axis.
            func: DifferenceFunction::CurvatureFlow(CurvatureFlowFunction::new(vec![1.0; dim])),
            // `SetNumberOfLayers`: `2` in 2-D, `3` in 3-D, `ImageDimension`
            // otherwise — i.e. `ImageDimension` in every arm.
            number_of_layers: dim,
            use_image_spacing: false,
            update_rule: UpdateRule::BinaryConstrained {
                input,
                upper_binary_value: max,
            },
        },
    );
    let out = solver.run(maximum_rms_error, number_of_iterations);

    Ok(LevelSetResult {
        image: image_from_f64(
            real_pixel_id(image.pixel_id()),
            image.size(),
            image,
            &out.values,
        )?,
        elapsed_iterations: out.elapsed_iterations,
        rms_change: out.rms_change,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    const N: usize = 24;

    /// A jagged, axis-aligned diamond (`|x - c| + |y - c| <= r`) on a 24x24
    /// grid: `1` inside, `0` outside. Its four faces are staircases, which is
    /// exactly the aliasing the filter exists to remove.
    fn diamond(radius: i64) -> Image {
        let c = (N / 2) as i64;
        let mut data = vec![0u8; N * N];
        for y in 0..N as i64 {
            for x in 0..N as i64 {
                if (x - c).abs() + (y - c).abs() <= radius {
                    data[(x + N as i64 * y) as usize] = 1;
                }
            }
        }
        Image::from_vec(&[N, N], data).unwrap()
    }

    /// `sign(output) == sign(input - iso)` at every pixel, with zero counted as
    /// belonging to the inside (the `max(.., 0)` branch's floor).
    fn assert_sides_preserved(input: &Image, output: &Image, iso: f64) {
        for (i, (&b, &u)) in input
            .to_f64_vec()
            .unwrap()
            .iter()
            .zip(&output.to_f64_vec().unwrap())
            .enumerate()
        {
            if b - iso > 0.0 {
                assert!(u >= 0.0, "pixel {i}: inside pixel escaped to {u}");
            } else {
                assert!(u <= 0.0, "pixel {i}: outside pixel escaped to {u}");
            }
        }
    }

    // ---- The defining constraint -------------------------------------------

    /// No pixel changes side, and the surface actually moves: the staircase
    /// corners of the diamond relax toward the true edge.
    #[test]
    fn every_pixel_keeps_the_side_the_binary_input_put_it_on() {
        let input = diamond(8);
        let result = anti_alias_binary(&input, 0.07, 1000).unwrap();
        assert_sides_preserved(&input, &result.image, 0.5);
        assert!(result.elapsed_iterations > 0);
    }

    /// The clamp is what preserves the sides. With it removed the solver is
    /// plain curvature flow, and the diamond's staircase corners cross zero.
    /// Assert the fact the clamp guards: the value at a corner pixel of the
    /// staircase never turns negative even though curvature there is strongly
    /// negative (the surface is locally convex outward).
    #[test]
    fn the_constraint_pins_a_staircase_corner_at_zero() {
        let input = diamond(8);
        let result = anti_alias_binary(&input, 0.0, 200).unwrap();
        let out = result.image.to_f64_vec().unwrap();

        // (12, 4) is the diamond's top vertex, a single inside pixel with three
        // outside face neighbors: curvature flow pulls it hard outward.
        let vertex = 12 + N * 4;
        assert_eq!(input.to_f64_vec().unwrap()[vertex], 1.0);
        assert!(
            out[vertex] >= 0.0,
            "the vertex escaped its side: {}",
            out[vertex]
        );
        // Its outside neighbor directly above is pinned on the other side.
        let above = 12 + N * 3;
        assert_eq!(input.to_f64_vec().unwrap()[above], 0.0);
        assert!(out[above] <= 0.0, "the neighbor escaped: {}", out[above]);
    }

    /// Smoothing is real: the diamond's zero crossing pulls in at the vertices
    /// and pushes out at the flat faces, so the vertex pixel's level-set value
    /// drops below the value of a pixel the same city-block distance away on a
    /// face.
    #[test]
    fn the_vertex_relaxes_further_than_the_face() {
        let result = anti_alias_binary(&diamond(8), 0.0, 200).unwrap();
        let out = result.image.to_f64_vec().unwrap();
        let vertex = 12 + N * 4; // on the interface, at the tip
        let face = 8 + N * 8; // on the interface, mid-face
        assert!(
            out[vertex] < out[face],
            "vertex {} should relax below face {}",
            out[vertex],
            out[face]
        );
    }

    // ---- Halting ------------------------------------------------------------

    /// `Halt` (itkFiniteDifferenceImageFilter.hxx): zero iterations halts before
    /// the first `CalculateChange`, so the output is `Initialize()`'s level set
    /// after `PostProcessOutput` and the RMS change is still its `0.0` default.
    #[test]
    fn zero_iterations_returns_the_initialized_level_set() {
        let input = diamond(8);
        let result = anti_alias_binary(&input, 0.07, 0).unwrap();
        assert_eq!(result.elapsed_iterations, 0);
        assert_eq!(result.rms_change, 0.0);
        assert_sides_preserved(&input, &result.image, 0.5);

        // `InitializeActiveLayerValues` clamps the active layer into
        // `[-g/2, g/2]` with `g == 1`; the layers step by `g` from there, and
        // `PostProcessOutput` flattens the rest to `(2 + 1) * 1`.
        let out = result.image.to_f64_vec().unwrap();
        assert_eq!(out[0], -3.0);
        assert_eq!(out[12 + N * 12], 3.0); // deep inside
        assert!(out.iter().all(|&v| (-3.0..=3.0).contains(&v)));
    }

    /// A large disc is already smooth, so the RMS change of the first iteration
    /// is small: with `maximum_rms_error = 0.07` the solver halts after exactly
    /// one iteration, long before the 1000-iteration cap.
    #[test]
    fn rms_convergence_stops_early_on_an_already_smooth_shape() {
        let c = (N / 2) as f64 - 0.5;
        let mut data = vec![0u8; N * N];
        for y in 0..N {
            for x in 0..N {
                let (dx, dy) = (x as f64 - c, y as f64 - c);
                if (dx * dx + dy * dy).sqrt() <= 9.0 {
                    data[x + N * y] = 1;
                }
            }
        }
        let disc = Image::from_vec(&[N, N], data).unwrap();

        let result = anti_alias_binary(&disc, 0.07, 1000).unwrap();
        assert_eq!(result.elapsed_iterations, 1);
        assert!(result.rms_change < 0.07, "rms {}", result.rms_change);

        // The same shape with the RMS gate closed runs the full budget.
        let forced = anti_alias_binary(&disc, 0.0, 5).unwrap();
        assert_eq!(forced.elapsed_iterations, 5);
    }

    /// `Halt` never applies the RMS test on the first pass (`elapsed == 0`), so
    /// even an `maximum_rms_error` above every attainable change runs once.
    #[test]
    fn the_rms_test_never_fires_before_the_first_iteration() {
        let result = anti_alias_binary(&diamond(8), 1.0e9, 1000).unwrap();
        assert_eq!(result.elapsed_iterations, 1);
    }

    // ---- Degenerate inputs ---------------------------------------------------

    /// A constant image has no zero crossing, hence no active layer.
    /// `InitializeBackgroundPixels` signs every pixel on `shifted > 0`, and the
    /// shift is exactly the constant, so the whole output is negative — the
    /// upstream quirk documented on [`anti_alias_binary`].
    #[test]
    fn a_constant_image_collapses_to_the_negative_background() {
        for fill in [0u8, 1, 255] {
            let image = Image::from_vec(&[8, 8], vec![fill; 64]).unwrap();
            let result = anti_alias_binary(&image, 0.07, 1000).unwrap();
            assert_eq!(result.image.to_f64_vec().unwrap(), vec![-3.0; 64]);
            // One no-op iteration: the active layer is empty, so `counter == 0`
            // and the RMS change is zero, which trips `Halt` on the next pass.
            assert_eq!(result.elapsed_iterations, 1);
            assert_eq!(result.rms_change, 0.0);
        }
    }

    /// The two binary values need not be `0` and `1`: the isosurface tracks the
    /// midpoint of whatever min/max the input carries.
    #[test]
    fn arbitrary_binary_values_shift_the_isosurface_to_their_midpoint() {
        let mut data = vec![7i16; N * N];
        let c = (N / 2) as i64;
        for y in 0..N as i64 {
            for x in 0..N as i64 {
                if (x - c).abs() + (y - c).abs() <= 8 {
                    data[(x + N as i64 * y) as usize] = 93;
                }
            }
        }
        let input = Image::from_vec(&[N, N], data).unwrap();
        let result = anti_alias_binary(&input, 0.07, 1000).unwrap();
        assert_sides_preserved(&input, &result.image, 50.0);
    }

    // ---- Types and geometry --------------------------------------------------

    /// `NumericTraits<InputPixelType>::RealType` is `double` for every integer
    /// pixel type, and those are the only ones the yaml admits.
    #[test]
    fn the_output_pixel_type_is_the_inputs_real_type() {
        assert_eq!(
            anti_alias_binary(&diamond(6), 0.07, 3)
                .unwrap()
                .image
                .pixel_id(),
            PixelId::Float64
        );
    }

    /// `pixel_types: IntegerPixelIDTypeList`.
    #[test]
    fn a_floating_point_input_is_rejected() {
        for id in [PixelId::Float32, PixelId::Float64] {
            let err = anti_alias_binary(&Image::new(&[4, 4], id), 0.07, 3).unwrap_err();
            assert!(matches!(err, FilterError::RequiresIntegerPixelType(got) if got == id));
        }
    }

    /// Spacing carries to the output but never into the solver: `UseImageSpacing`
    /// is off, so a stretched grid yields the identical level set.
    #[test]
    fn spacing_is_carried_but_not_used() {
        let plain = diamond(8);
        let mut stretched = diamond(8);
        stretched.set_spacing(&[3.0, 0.5]).unwrap();
        stretched.set_origin(&[-2.0, 5.0]).unwrap();

        let a = anti_alias_binary(&plain, 0.07, 20).unwrap();
        let b = anti_alias_binary(&stretched, 0.07, 20).unwrap();

        assert_eq!(b.image.spacing(), &[3.0, 0.5]);
        assert_eq!(b.image.origin(), &[-2.0, 5.0]);
        assert_eq!(a.image.to_f64_vec().unwrap(), b.image.to_f64_vec().unwrap());
        assert_eq!(a.elapsed_iterations, b.elapsed_iterations);
    }

    /// `PostProcessOutput` flattens the background to
    /// `(number_of_layers + 1) * m_ConstantGradientValue`, and
    /// `number_of_layers` is the image dimension: `3` in 2-D (pinned by
    /// [`zero_iterations_returns_the_initialized_level_set`]), `4` in 3-D.
    ///
    /// The ball's *interior* does not reach that magnitude: it is only 3.5
    /// pixels deep, so every inside pixel still sits inside the four-layer
    /// sparse field and carries a propagated layer value below `+4`. The
    /// background magnitude shows up on the outside, which is deep.
    #[test]
    fn the_background_magnitude_follows_the_layer_count() {
        let n = 12usize;
        let mut data = vec![0u8; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d = |v: usize| (v as f64 - 5.5).powi(2);
                    if (d(x) + d(y) + d(z)).sqrt() <= 3.5 {
                        data[x + n * (y + n * z)] = 1;
                    }
                }
            }
        }
        let ball = Image::from_vec(&[n, n, n], data).unwrap();
        let out = anti_alias_binary(&ball, 0.07, 5)
            .unwrap()
            .image
            .to_f64_vec()
            .unwrap();

        assert_eq!(out[0], -4.0);
        assert!(out.iter().all(|&v| (-4.0..=4.0).contains(&v)));
        // The ball's center is a layer node, one gradient step per layer in
        // from the active band's `[-0.5, 0.5]`, hence strictly below `+4`.
        let center = out[5 + n * (5 + n * 5)];
        assert!((2.0..4.0).contains(&center), "ball center {center}");
    }
}
