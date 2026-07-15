//! `InvertDisplacementFieldImageFilter`
//! (`itkInvertDisplacementFieldImageFilter.h(.hxx)`): Tustison and Avants'
//! fixed-point inversion of a displacement field, the routine SyN uses to keep
//! an explicit inverse field.
//!
//! # The iteration
//!
//! Write `u` for the forward field and `v` for the inverse estimate. One
//! iteration, over every lattice point `y` of the inverse field:
//!
//! 1. **Compose** (`ComposeDisplacementFieldsImageFilter::DynamicThreadedGenerateData`,
//!    `itkComposeDisplacementFieldsImageFilter.hxx:89-114`). With `p₁` the
//!    physical point of `y`, `p₂ = p₁ + v(y)`, and `u(p₂)` the linear
//!    interpolation of the forward field (zero if `p₂` is outside its buffer),
//!    the composed field is `c(y) = (p₂ + u(p₂)) − p₁`. That is the residual of
//!    the fixed-point condition `v(y) = −u(y + v(y))`.
//!
//! 2. **Norms** (`itkInvertDisplacementFieldImageFilter.hxx:221-247`). The
//!    per-pixel error is the **voxel-unit** norm
//!    `n(y) = ‖ c(y) ⊘ spacing ‖`, *not* the physical norm — the composed vector
//!    is divided component-wise by the forward field's spacing before the norm
//!    is taken. `MaxErrorNorm` is `maxᵧ n(y)` and `MeanErrorNorm` is the mean of
//!    `n` over the whole region. The composed field is then negated in place.
//!
//! 3. **Update** (`itkInvertDisplacementFieldImageFilter.hxx:187-210`) with
//!    `ε = 0.75` on the first iteration and `ε = 0.5` afterwards:
//!
//!    ```text
//!    update = −c(y)
//!    if n(y) > ε · MaxErrorNorm:  update *= ε · MaxErrorNorm / n(y)
//!    v(y) += ε · update
//!    ```
//!
//!    Note the mixed units, reproduced as written: the *scaled* (voxel-unit)
//!    norm `n(y)` gates and scales a *physical* update vector. On an anisotropic
//!    field the clamp is therefore not a bound on the physical step length.
//!
//! 4. **Boundary** (`itkInvertDisplacementFieldImageFilter.hxx:199-209`). When
//!    `enforce_boundary_condition`, any `y` with `y[d] == 0` or
//!    `y[d] == size[d] − 1` for some `d` is overwritten with the zero vector,
//!    *after* the update was already computed and stored.
//!
//! The loop runs while `iteration < MaximumNumberOfIterations` **and**
//! `MaxErrorNorm > MaxErrorToleranceThreshold` **and**
//! `MeanErrorNorm > MeanErrorToleranceThreshold`
//! (`itkInvertDisplacementFieldImageFilter.hxx:115-117`). The two norms start at
//! `NumericTraits<RealType>::max()`, so the first iteration always runs; they
//! hold the values measured *during the last iteration that ran*, which are the
//! residuals of the estimate from *before* that iteration's update. Both norm
//! conditions must hold to continue, so the loop stops as soon as *either*
//! tolerance is met.
//!
//! # Faithfully reproduced upstream behaviors
//!
//! - **`MaximumNumberOfIterations` defaults to 10, not ITK's 20.** ITK's member
//!   initializer is `{ 20 }` (`itkInvertDisplacementFieldImageFilter.h:233`) but
//!   `InvertDisplacementFieldImageFilter.yaml` declares `default: 10u`, and
//!   SimpleITK's generated wrapper always calls the setter. This port follows
//!   the yaml, which is the public parameter surface.
//!
//! - **The boundary test uses the composed field's start index.** Upstream reads
//!   `index[d] == startIndex[d] || index[d] == size[d] - startIndex[d] - 1`.
//!   With the zero start index every `Image` in this crate has, the second
//!   disjunct is `size[d] - 1`, the last lattice plane. (Upstream's expression
//!   is not the last plane for a nonzero start index, where it would need to be
//!   `startIndex[d] + size[d] - 1`; see the module report.)
//!
//! - **The initial estimate becomes the output.** `GenerateData` duplicates
//!   `InverseFieldInitialEstimate` and installs it as output 0
//!   (`itkInvertDisplacementFieldImageFilter.hxx:82-92`), so the result carries
//!   the *estimate's* geometry, and it is the estimate's lattice that supplies
//!   the physical points `p₁` fed to the composition. This port does the same.
//!
//! # Divergence
//!
//! Upstream never checks that the initial estimate's lattice matches the
//! forward field's: the composed field and the scaled-norm image are sized from
//! the forward field while the output iterator walks the estimate's region, and
//! a size mismatch reads past the end of one of them. That is C++ undefined
//! behavior, so this port rejects it with [`FilterError::SizeMismatch`] instead.
//!
//! With zero iterations run, upstream returns `NumericTraits<RealType>::max()`
//! for both norms. This port computes in `f64` (see [`super`]) and so returns
//! `f64::MAX`, where an ITK `VectorFloat32` field would report `f32::MAX`.

use crate::core::Image;

use super::{Field, field_to_image};
use crate::filters::geometry::require_same_physical_space;
use crate::filters::{FilterError, Result};

/// The four `InvertDisplacementFieldImageFilter.yaml` members, with its
/// defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvertDisplacementFieldSettings {
    /// `MaximumNumberOfIterations`, yaml default `10u`.
    pub maximum_number_of_iterations: u32,
    /// `MaxErrorToleranceThreshold`, yaml default `0.1`, in voxel units.
    pub max_error_tolerance_threshold: f64,
    /// `MeanErrorToleranceThreshold`, yaml default `0.001`, in voxel units.
    pub mean_error_tolerance_threshold: f64,
    /// `EnforceBoundaryCondition`, yaml default `true`: clamp the inverse to
    /// zero on the outermost lattice plane of every axis.
    pub enforce_boundary_condition: bool,
}

impl Default for InvertDisplacementFieldSettings {
    fn default() -> Self {
        InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 10,
            max_error_tolerance_threshold: 0.1,
            mean_error_tolerance_threshold: 0.001,
            enforce_boundary_condition: true,
        }
    }
}

/// The inverse field together with the yaml's two `measurements`.
#[derive(Clone, Debug, PartialEq)]
pub struct InvertDisplacementFieldResult {
    /// The estimated inverse displacement field, same pixel type as the input.
    pub inverse_displacement_field: Image,
    /// `GetMaxErrorNorm()`: the largest voxel-unit residual norm seen in the
    /// last iteration that ran, or `f64::MAX` if none ran.
    pub max_error_norm: f64,
    /// `GetMeanErrorNorm()`: the mean voxel-unit residual norm over the region
    /// in the last iteration that ran, or `f64::MAX` if none ran.
    pub mean_error_norm: f64,
}

/// Estimate the inverse of `displacement_field` by fixed-point composition.
///
/// `inverse_field_initial_estimate` is the yaml's optional second input; `None`
/// starts from the zero field, as `FillBuffer(zeroVector)` does upstream.
///
/// Errors on a `displacement_field` that is not a real-component vector image
/// with one component per dimension (see
/// [`require_displacement_field`](super::require_displacement_field)), on the
/// same for the initial estimate, and with [`FilterError::SizeMismatch`] when
/// the estimate's lattice differs from the field's.
pub fn invert_displacement_field(
    displacement_field: &Image,
    inverse_field_initial_estimate: Option<&Image>,
    settings: &InvertDisplacementFieldSettings,
) -> Result<InvertDisplacementFieldResult> {
    let forward = Field::from_image(displacement_field)?;

    // Upstream duplicates the initial estimate and installs it as the output, so
    // the inverse estimate keeps that image's geometry; without one it is a zero
    // field on the forward lattice.
    let mut inverse = match inverse_field_initial_estimate {
        Some(estimate) => {
            require_same_physical_space(displacement_field, estimate, 1)?;
            let field = Field::from_image(estimate)?;
            if field.size != forward.size {
                return Err(FilterError::SizeMismatch {
                    a: forward.size.clone(),
                    b: field.size,
                });
            }
            field
        }
        None => Field::zeros_like(&forward),
    };

    let dim = forward.dim;
    let pixels = forward.number_of_pixels();
    let inverse_spacing: Vec<f64> = forward.spacing.iter().map(|&s| 1.0 / s).collect();

    // `v` is the inverse estimate under iteration; `inverse` keeps only the
    // lattice and geometry that supply the physical points `p₁`.
    let mut v = std::mem::take(&mut inverse.data);
    let mut composed = vec![0.0f64; pixels * dim];
    let mut scaled_norm = vec![0.0f64; pixels];

    let mut max_error_norm = f64::MAX;
    let mut mean_error_norm = f64::MAX;
    let mut iteration = 0u32;

    while iteration < settings.maximum_number_of_iterations
        && max_error_norm > settings.max_error_tolerance_threshold
        && mean_error_norm > settings.mean_error_tolerance_threshold
    {
        iteration += 1;

        // Compose: c(y) = (p₂ + u(p₂)) − p₁, with p₂ = p₁ + v(y).
        for pixel in 0..pixels {
            let point1 = inverse.index_to_point(&inverse.multi_index(pixel));
            let warp = &v[pixel * dim..(pixel + 1) * dim];
            let point2: Vec<f64> = (0..dim).map(|d| point1[d] + warp[d]).collect();
            let displacement = forward
                .evaluate_at_point(&point2)
                .unwrap_or_else(|| vec![0.0; dim]);
            for d in 0..dim {
                composed[pixel * dim + d] = point2[d] + displacement[d] - point1[d];
            }
        }

        // Voxel-unit norms of the composed field, which is then negated.
        let mut mean = 0.0f64;
        let mut max = 0.0f64;
        for pixel in 0..pixels {
            let mut norm = 0.0f64;
            for d in 0..dim {
                norm += (composed[pixel * dim + d] * inverse_spacing[d]).powi(2);
            }
            let norm = norm.sqrt();

            mean += norm;
            if max < norm {
                max = norm;
            }
            scaled_norm[pixel] = norm;

            for d in 0..dim {
                composed[pixel * dim + d] = -composed[pixel * dim + d];
            }
        }
        mean_error_norm = mean / pixels as f64;
        max_error_norm = max;

        let epsilon = if iteration == 1 { 0.75 } else { 0.5 };

        // Estimate the inverse from the scaled, clamped update.
        for pixel in 0..pixels {
            let norm = scaled_norm[pixel];
            let scale = if norm > epsilon * max_error_norm {
                epsilon * max_error_norm / norm
            } else {
                1.0
            };
            for d in 0..dim {
                v[pixel * dim + d] += composed[pixel * dim + d] * scale * epsilon;
            }

            if settings.enforce_boundary_condition {
                let index = forward.multi_index(pixel);
                if (0..dim).any(|d| index[d] == 0 || index[d] == forward.size[d] - 1) {
                    v[pixel * dim..(pixel + 1) * dim].fill(0.0);
                }
            }
        }
    }

    let inverse_displacement_field = field_to_image(
        &inverse.size,
        v,
        displacement_field.pixel_id().component_id(),
        &inverse.spacing,
        &inverse.origin,
        &inverse.direction,
    )?;

    Ok(InvertDisplacementFieldResult {
        inverse_displacement_field,
        max_error_norm,
        mean_error_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    /// The smallest legal displacement field: 1-D image, one component.
    fn field_1d(values: &[f64]) -> Image {
        Image::from_vec_vector(&[values.len()], 1, values.to_vec()).unwrap()
    }

    fn components(img: &Image) -> Vec<f64> {
        img.components_to_f64_vec()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    /// A zero field is its own inverse, and the first iteration measures zero
    /// error — so the `max > tol` guard stops the loop right after it.
    #[test]
    fn zero_field_inverts_to_the_zero_field() {
        let field = Image::from_vec_vector(&[4, 4], 2, vec![0.0f64; 32]).unwrap();
        let out =
            invert_displacement_field(&field, None, &InvertDisplacementFieldSettings::default())
                .unwrap();

        assert_eq!(
            out.inverse_displacement_field.pixel_id(),
            PixelId::VectorFloat64
        );
        assert_eq!(components(&out.inverse_displacement_field), vec![0.0; 32]);
        assert_eq!(out.max_error_norm, 0.0);
        assert_eq!(out.mean_error_norm, 0.0);
    }

    /// One iteration of a constant translation field, derived by hand.
    ///
    /// `u ≡ 0.1`, unit spacing. Iteration 1: `v = 0`, so `p₂ = p₁` and
    /// `c = 0.1` everywhere; `n = 0.1`, `max = mean = 0.1`; `ε = 0.75`, and
    /// `n > ε·max = 0.075`, so `update = −0.1 · (0.075 / 0.1) = −0.075` and
    /// `v = 0.75 · (−0.075) = −0.05625`.
    #[test]
    fn constant_translation_field_takes_one_scaled_step_toward_its_inverse() {
        let field = field_1d(&[0.1; 5]);
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 1,
            enforce_boundary_condition: false,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, None, &settings).unwrap();

        assert_close(&components(&out.inverse_displacement_field), &[-0.05625; 5]);
        assert_close(&[out.max_error_norm, out.mean_error_norm], &[0.1, 0.1]);
    }

    /// The loop stops as soon as *either* tolerance is met, because upstream's
    /// `while` conjoins `max > maxTol` **and** `mean > meanTol`.
    ///
    /// Both cases below take exactly the one iteration derived above (`max` and
    /// `mean` both land on `0.1`), each stopped by a different clause, and so
    /// both reproduce `v = −0.05625` rather than iterating further.
    #[test]
    fn either_tolerance_alone_stops_the_loop() {
        let field = field_1d(&[0.1; 5]);

        let stopped_by_mean = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 10,
            max_error_tolerance_threshold: 1e-12,
            mean_error_tolerance_threshold: 0.5,
            enforce_boundary_condition: false,
        };
        let stopped_by_max = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 10,
            max_error_tolerance_threshold: 0.5,
            mean_error_tolerance_threshold: 1e-12,
            enforce_boundary_condition: false,
        };

        for settings in [stopped_by_mean, stopped_by_max] {
            let out = invert_displacement_field(&field, None, &settings).unwrap();
            assert_close(&components(&out.inverse_displacement_field), &[-0.05625; 5]);
        }
    }

    /// Same field, two iterations forced by a tighter tolerance.
    ///
    /// Iteration 2 starts from `v = −0.05625`: `p₂ = x − 0.05625` is inside the
    /// buffer for every `x` (the half-pixel skirt reaches `−0.5`), so
    /// `u(p₂) = 0.1` and `c = 0.04375`. Now `ε = 0.5`, `n = max = 0.04375`, and
    /// `n > ε·max` holds, so `update = −0.04375 · 0.5 = −0.021875` and
    /// `v = −0.05625 + 0.5 · (−0.021875) = −0.0671875`.
    #[test]
    fn a_second_iteration_uses_epsilon_one_half() {
        let field = field_1d(&[0.1; 5]);
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 2,
            max_error_tolerance_threshold: 1e-9,
            mean_error_tolerance_threshold: 1e-9,
            enforce_boundary_condition: false,
        };
        let out = invert_displacement_field(&field, None, &settings).unwrap();

        assert_close(
            &components(&out.inverse_displacement_field),
            &[-0.0671875; 5],
        );
        assert_close(
            &[out.max_error_norm, out.mean_error_norm],
            &[0.04375, 0.04375],
        );
    }

    /// The exact inverse of a constant translation is a fixed point: the
    /// residual is zero, so nothing moves and both norms come out zero.
    #[test]
    fn the_exact_inverse_of_a_translation_field_is_a_fixed_point() {
        let field = field_1d(&[0.1; 5]);
        let estimate = field_1d(&[-0.1; 5]);
        let settings = InvertDisplacementFieldSettings {
            enforce_boundary_condition: false,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, Some(&estimate), &settings).unwrap();

        assert_close(&components(&out.inverse_displacement_field), &[-0.1; 5]);
        assert!(out.max_error_norm < 1e-15);
        assert!(out.mean_error_norm < 1e-15);
    }

    /// The boundary condition zeroes the outermost lattice plane *after* the
    /// update is applied, so only the interior keeps its step.
    #[test]
    fn enforce_boundary_condition_zeroes_the_outermost_lattice_plane() {
        let field = field_1d(&[0.1; 5]);
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 1,
            enforce_boundary_condition: true,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, None, &settings).unwrap();

        assert_close(
            &components(&out.inverse_displacement_field),
            &[0.0, -0.05625, -0.05625, -0.05625, 0.0],
        );
    }

    /// In 2-D the boundary is every pixel on an outer plane of *any* axis, so a
    /// 3×3 field keeps only its centre.
    #[test]
    fn enforce_boundary_condition_zeroes_every_outer_plane_in_two_dimensions() {
        let field = Image::from_vec_vector(&[3, 3], 2, vec![0.1f64; 18]).unwrap();
        let settings = InvertDisplacementFieldSettings {
            enforce_boundary_condition: true,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, None, &settings).unwrap();

        let v = components(&out.inverse_displacement_field);
        for pixel in 0..9 {
            let interior = pixel == 4;
            for d in 0..2 {
                if interior {
                    assert!(v[pixel * 2 + d] != 0.0, "centre pixel was zeroed");
                } else {
                    assert_eq!(v[pixel * 2 + d], 0.0, "boundary pixel {pixel} kept a value");
                }
            }
        }
    }

    /// The error norms are in **voxel** units: halving the spacing doubles them,
    /// while the update itself is unchanged because the scale factor
    /// `ε·max / n` cancels the units on a constant field.
    #[test]
    fn the_error_norms_are_measured_in_voxel_units() {
        let mut field = field_1d(&[0.1; 5]);
        field.set_spacing(&[0.5]).unwrap();
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 1,
            enforce_boundary_condition: false,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, None, &settings).unwrap();

        assert_close(&[out.max_error_norm, out.mean_error_norm], &[0.2, 0.2]);
        assert_close(&components(&out.inverse_displacement_field), &[-0.05625; 5]);
    }

    /// A probe point outside the forward field's buffer contributes a zero
    /// forward displacement, so `c(y) = v(y)`.
    ///
    /// `u ≡ 0`, estimate `v = [−1, 0, 0]`. Pixel 0 probes continuous index `−1`,
    /// past the `−0.5` skirt, so `u = 0` and `c = −1`; `n = [1, 0, 0]`,
    /// `max = 1`, `mean = 1/3`. With `ε = 0.75`: pixel 0 has `n > 0.75`, so
    /// `update = 1 · 0.75 = 0.75` and `v₀ = −1 + 0.75 · 0.75 = −0.4375`.
    #[test]
    fn a_probe_point_outside_the_forward_buffer_reads_zero() {
        let field = field_1d(&[0.0; 3]);
        let estimate = field_1d(&[-1.0, 0.0, 0.0]);
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 1,
            enforce_boundary_condition: false,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, Some(&estimate), &settings).unwrap();

        assert_close(
            &components(&out.inverse_displacement_field),
            &[-0.4375, 0.0, 0.0],
        );
        assert_close(
            &[out.max_error_norm, out.mean_error_norm],
            &[1.0, 1.0 / 3.0],
        );
    }

    /// Both branches of the update clamp on one field, plus the interpolator's
    /// neighbour clamp inside the half-pixel skirt.
    ///
    /// `u = [1, 5, 9]`, estimate `v = [−0.25, 0, 0]`. Pixel 0 probes continuous
    /// index `−0.25`, inside the skirt, where the clamp gives `u = 1`; so
    /// `c = [0.75, 5, 9]` and `n = c`. `max = 9`, `mean = 14.75/3`, `ε = 0.75`,
    /// threshold `ε·max = 6.75`.
    ///
    /// - pixel 0: `0.75 ≤ 6.75`, unscaled: `v = −0.25 + 0.75·(−0.75) = −0.8125`
    /// - pixel 1: `5 ≤ 6.75`, unscaled: `v = 0.75·(−5) = −3.75`
    /// - pixel 2: `9 > 6.75`, scaled to `−6.75`: `v = 0.75·(−6.75) = −5.0625`
    #[test]
    fn the_update_clamp_scales_only_pixels_above_epsilon_times_the_max_norm() {
        let field = field_1d(&[1.0, 5.0, 9.0]);
        let estimate = field_1d(&[-0.25, 0.0, 0.0]);
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 1,
            enforce_boundary_condition: false,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, Some(&estimate), &settings).unwrap();

        assert_close(
            &components(&out.inverse_displacement_field),
            &[-0.8125, -3.75, -5.0625],
        );
        assert_close(
            &[out.max_error_norm, out.mean_error_norm],
            &[9.0, 14.75 / 3.0],
        );
    }

    /// Zero iterations means the loop body never runs, so the initial estimate
    /// passes through unchanged and both norms keep their sentinel value.
    #[test]
    fn zero_iterations_returns_the_initial_estimate_and_the_sentinel_norms() {
        let field = field_1d(&[0.1; 4]);
        let estimate = field_1d(&[1.0, 2.0, 3.0, 4.0]);
        let settings = InvertDisplacementFieldSettings {
            maximum_number_of_iterations: 0,
            ..Default::default()
        };
        let out = invert_displacement_field(&field, Some(&estimate), &settings).unwrap();

        assert_close(
            &components(&out.inverse_displacement_field),
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(out.max_error_norm, f64::MAX);
        assert_eq!(out.mean_error_norm, f64::MAX);
    }

    /// The output carries the *initial estimate's* geometry, because upstream
    /// installs the duplicated estimate as output 0. ITK registers the estimate
    /// via itkSetInputMacro(InverseFieldInitialEstimate), so the base
    /// VerifyInputInformation requires it to occupy the same physical space as the
    /// field (a mismatch is refused — pinned in
    /// tests/physical_space_precondition.rs); under that congruent contract the
    /// estimate's geometry equals the field's, and the output carries it.
    #[test]
    fn the_output_carries_the_initial_estimates_geometry() {
        let mut field = field_1d(&[0.0; 3]);
        field.set_spacing(&[0.5]).unwrap();
        field.set_origin(&[-3.0]).unwrap();

        let mut estimate = field_1d(&[0.0; 3]);
        estimate.set_spacing(&[0.5]).unwrap();
        estimate.set_origin(&[-3.0]).unwrap();

        let out = invert_displacement_field(
            &field,
            Some(&estimate),
            &InvertDisplacementFieldSettings::default(),
        )
        .unwrap();

        assert_eq!(out.inverse_displacement_field.spacing(), &[0.5]);
        assert_eq!(out.inverse_displacement_field.origin(), &[-3.0]);
    }

    /// A `VectorFloat32` field stays `VectorFloat32`.
    #[test]
    fn a_float32_field_keeps_its_component_type() {
        let field = Image::from_vec_vector(&[3], 1, vec![0.0f32; 3]).unwrap();
        let out =
            invert_displacement_field(&field, None, &InvertDisplacementFieldSettings::default())
                .unwrap();
        assert_eq!(
            out.inverse_displacement_field.pixel_id(),
            PixelId::VectorFloat32
        );
    }

    #[test]
    fn a_scalar_input_is_rejected() {
        let field = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            invert_displacement_field(&field, None, &InvertDisplacementFieldSettings::default())
                .unwrap_err(),
            FilterError::Core(crate::core::Error::RequiresVectorPixelType(
                PixelId::Float64
            ))
        ));
    }

    #[test]
    fn an_initial_estimate_of_a_different_size_is_rejected() {
        let field = field_1d(&[0.0; 4]);
        let estimate = field_1d(&[0.0; 3]);
        assert!(matches!(
            invert_displacement_field(
                &field,
                Some(&estimate),
                &InvertDisplacementFieldSettings::default()
            )
            .unwrap_err(),
            FilterError::SizeMismatch { .. }
        ));
    }

    #[test]
    fn an_initial_estimate_that_is_not_a_displacement_field_is_rejected() {
        let field = field_1d(&[0.0; 4]);
        let estimate = Image::new(&[4], PixelId::Float64);
        assert!(matches!(
            invert_displacement_field(
                &field,
                Some(&estimate),
                &InvertDisplacementFieldSettings::default()
            )
            .unwrap_err(),
            FilterError::Core(crate::core::Error::RequiresVectorPixelType(
                PixelId::Float64
            ))
        ));
    }
}
