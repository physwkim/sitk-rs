//! `DisplacementFieldJacobianDeterminantFilter`
//! (`itkDisplacementFieldJacobianDeterminantFilter.h(.hxx)`): the determinant of
//! the Jacobian of the *transformation* a displacement field encodes.
//!
//! # `det(I + du/dx)`, not `det(du/dx)`
//!
//! The transformation is `T(x) = x + u(x)`, so `dT/dx = I + du/dx` and the
//! filter returns `det(I + du/dx)` — `physicalGrad[row][row] +=
//! NumericTraits::OneValue()` at `hxx:250`, just before the determinant. The
//! class doc spells out why: "the determinant of a zero vector field is also
//! zero, whereas the Jacobian determinant of the corresponding identity warp
//! transformation is 1.0" (`.h:41-45`). A zero displacement field therefore maps
//! to the constant `1.0`, not `0.0`.
//!
//! This is what separates the filter from its sibling
//! `itkDeformationFieldJacobianDeterminantFilter`, which omits the `+ I` and
//! returns `det(du/dx)`.
//!
//! # The gradient
//!
//! `EvaluateAtNeighborhood` (`hxx:216-255`) builds two matrices at each pixel.
//!
//! - `localGrad[row][col] = halfWeight[col] * (u(i + e_col)[row] − u(i −
//!   e_col)[row])` — a central difference along *lattice* axis `col` of the
//!   `row`-th (world) component of `u`, on a radius-1 neighborhood.
//!
//! - `physicalGrad[row][col] = Σ_j Direction[col][j] * localGrad[row][j]`, i.e.
//!   `physicalGrad = localGrad · Directionᵀ`. Upstream writes this as the
//!   matrix-vector product `m_InputDirection * localComponentGrad_d` where
//!   `localComponentGrad_d` is *row* `row` of `localGrad`, then scatters the
//!   result across *column* `col` (`hxx:243-248`) — the transpose is implicit in
//!   that scatter. For an orthonormal direction matrix this is the chain rule:
//!   `∂i_j/∂x_col = Direction[col][j] / spacing[j]`, and the `1/spacing[j]` is
//!   already folded into `halfWeight[j]`.
//!
//! Then `1` is added to the diagonal and `vnl_determinant` (`vnl_determinant.hxx`)
//! is taken. See [`determinant`].
//!
//! # Boundary
//!
//! A radius-1 `ConstNeighborhoodIterator` under
//! `ZeroFluxNeumannBoundaryCondition` (`hxx:183, 203`), which returns the value
//! of the nearest in-bounds pixel (`itkZeroFluxNeumannBoundaryCondition.hxx:33-40`
//! clamps the index). At an edge lattice point the central difference therefore
//! degenerates to *half* a one-sided difference: at `i = 0` it is `0.5 · w ·
//! (u(1) − u(0))`, not `w · (u(1) − u(0))`. The face calculator (`hxx:186-201`)
//! only partitions the region for speed; it does not change any value.
//!
//! # Weights, and the order SimpleITK sets them in
//!
//! ITK holds `m_DerivativeWeights` (default `1.0`) and derives
//! `m_HalfDerivativeWeights = 0.5 * m_DerivativeWeights`. Two setters touch it:
//!
//! - `SetUseImageSpacing(false)` resets the weights to `1.0`, but *only* when
//!   they were previously `true` (`hxx:79-86`); at `Update` time
//!   `SetUseImageSpacing(true)` overwrites them with `1/spacing` (`hxx:155-168`).
//! - `SetDerivativeWeights(w)` assigns `w` **and silently sets
//!   `m_UseImageSpacing = false`** (`hxx:55`).
//!
//! SimpleITK's generated `ExecuteInternal` emits the setters in yaml member
//! order (`ExecuteInternalSetITKFilterParameters.cxx.jinja:1-13`), so
//! `SetUseImageSpacing` runs first and `SetDerivativeWeights` — guarded by
//! `if (!this->m_DerivativeWeights.empty())` — runs second and wins. A non-empty
//! `DerivativeWeights` thus disables `UseImageSpacing` no matter what the
//! `UseImageSpacing` member says, and `sitk::…::GetUseImageSpacing()` keeps
//! reporting the stale value. [`DisplacementFieldJacobianDeterminantSettings`]
//! reproduces exactly that precedence.
//!
//! # Divergences
//!
//! - ITK computes in `TRealType`, which SimpleITK fixes to the field's own
//!   component type — `float` for a `VectorFloat32` field. This port computes in
//!   `f64` and narrows on store, as the rest of this module does.
//!
//! - `BeforeThreadedGenerateData` throws when a spacing is zero (`hxx:161-164`).
//!   [`crate::core::Image::set_spacing`] rejects a non-positive spacing, so no
//!   `Image` can carry one and the check has nothing to guard.
//!
//! - Above dimension four `vnl_determinant` decomposes with `vnl_qr`
//!   (`vnl_determinant.hxx:68-100`); [`determinant`] uses Gaussian elimination
//!   with partial pivoting instead. The two agree up to rounding. SimpleITK
//!   images are at most 4-D, so this path is unreachable through the yaml's
//!   public surface.

use crate::core::{Image, Scalar, dispatch_scalar};

use super::Field;
use crate::filters::{FilterError, Result};

/// Parameters of `DisplacementFieldJacobianDeterminantFilter`, as
/// `DisplacementFieldJacobianDeterminantFilter.yaml` declares them.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplacementFieldJacobianDeterminantSettings {
    /// Scale each partial derivative by `1/spacing[i]`, taking the derivative in
    /// world coordinates. Ignored when `derivative_weights` is non-empty.
    pub use_image_spacing: bool,
    /// Explicit per-axis derivative weights. Empty means "unset"; a non-empty
    /// vector overrides `use_image_spacing` and must have one entry per
    /// dimension.
    pub derivative_weights: Vec<f64>,
}

impl Default for DisplacementFieldJacobianDeterminantSettings {
    fn default() -> Self {
        DisplacementFieldJacobianDeterminantSettings {
            use_image_spacing: true,
            derivative_weights: Vec::new(),
        }
    }
}

/// `vnl_determinant` of a row-major `n × n` matrix
/// (`vnl_determinant.hxx:15-46, 51-68`).
///
/// The `n ≤ 4` cases reproduce vnl's explicit cofactor expansions term for
/// term, including their summation order, so the floating-point result is
/// identical. Larger `n` — unreachable from SimpleITK, whose images are at most
/// 4-D — falls back to Gaussian elimination where vnl falls back to `vnl_qr`.
fn determinant(m: &[f64], n: usize) -> f64 {
    let a = |r: usize, c: usize| m[r * n + c];
    match n {
        1 => a(0, 0),
        2 => a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0),
        3 => {
            a(0, 0) * a(1, 1) * a(2, 2) - a(0, 0) * a(2, 1) * a(1, 2) - a(1, 0) * a(0, 1) * a(2, 2)
                + a(1, 0) * a(2, 1) * a(0, 2)
                + a(2, 0) * a(0, 1) * a(1, 2)
                - a(2, 0) * a(1, 1) * a(0, 2)
        }
        4 => {
            a(0, 0) * a(1, 1) * a(2, 2) * a(3, 3)
                - a(0, 0) * a(1, 1) * a(3, 2) * a(2, 3)
                - a(0, 0) * a(2, 1) * a(1, 2) * a(3, 3)
                + a(0, 0) * a(2, 1) * a(3, 2) * a(1, 3)
                + a(0, 0) * a(3, 1) * a(1, 2) * a(2, 3)
                - a(0, 0) * a(3, 1) * a(2, 2) * a(1, 3)
                - a(1, 0) * a(0, 1) * a(2, 2) * a(3, 3)
                + a(1, 0) * a(0, 1) * a(3, 2) * a(2, 3)
                + a(1, 0) * a(2, 1) * a(0, 2) * a(3, 3)
                - a(1, 0) * a(2, 1) * a(3, 2) * a(0, 3)
                - a(1, 0) * a(3, 1) * a(0, 2) * a(2, 3)
                + a(1, 0) * a(3, 1) * a(2, 2) * a(0, 3)
                + a(2, 0) * a(0, 1) * a(1, 2) * a(3, 3)
                - a(2, 0) * a(0, 1) * a(3, 2) * a(1, 3)
                - a(2, 0) * a(1, 1) * a(0, 2) * a(3, 3)
                + a(2, 0) * a(1, 1) * a(3, 2) * a(0, 3)
                + a(2, 0) * a(3, 1) * a(0, 2) * a(1, 3)
                - a(2, 0) * a(3, 1) * a(1, 2) * a(0, 3)
                - a(3, 0) * a(0, 1) * a(1, 2) * a(2, 3)
                + a(3, 0) * a(0, 1) * a(2, 2) * a(1, 3)
                + a(3, 0) * a(1, 1) * a(0, 2) * a(2, 3)
                - a(3, 0) * a(1, 1) * a(2, 2) * a(0, 3)
                - a(3, 0) * a(2, 1) * a(0, 2) * a(1, 3)
                + a(3, 0) * a(2, 1) * a(1, 2) * a(0, 3)
        }
        _ => gaussian_determinant(m.to_vec(), n),
    }
}

fn gaussian_determinant(mut a: Vec<f64>, n: usize) -> f64 {
    let mut det = 1.0f64;
    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            det = -det;
        }
        det *= a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / a[col * n + col];
            for c in col..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
        }
    }
    det
}

/// The weights `BeforeThreadedGenerateData` leaves in `m_DerivativeWeights`
/// after SimpleITK's setter order, and their halves.
fn resolve_weights(
    field: &Field,
    settings: &DisplacementFieldJacobianDeterminantSettings,
) -> Result<Vec<f64>> {
    if !settings.derivative_weights.is_empty() {
        if settings.derivative_weights.len() != field.dim {
            return Err(FilterError::DimensionLength {
                expected: field.dim,
                got: settings.derivative_weights.len(),
            });
        }
        return Ok(settings.derivative_weights.clone());
    }
    if settings.use_image_spacing {
        return Ok(field.spacing.iter().map(|&s| 1.0 / s).collect());
    }
    Ok(vec![1.0; field.dim])
}

/// Compute `det(I + du/dx)` at every pixel of `displacement_field`.
///
/// The output is a scalar image on the input's lattice with the input's geometry
/// and the input's *component* type: a `VectorFloat32` field yields a `Float32`
/// image, a `VectorFloat64` field a `Float64` one.
///
/// Errors:
///
/// - [`super::require_displacement_field`]'s, on an input that is not a
///   real-valued vector image with one component per dimension;
/// - [`FilterError::DimensionLength`] when `derivative_weights` is non-empty and
///   does not have one entry per dimension.
pub fn displacement_field_jacobian_determinant(
    displacement_field: &Image,
    settings: &DisplacementFieldJacobianDeterminantSettings,
) -> Result<Image> {
    let field = Field::from_image(displacement_field)?;
    let dim = field.dim;
    let half: Vec<f64> = resolve_weights(&field, settings)?
        .iter()
        .map(|w| 0.5 * w)
        .collect();

    let mut determinants = vec![0.0f64; field.number_of_pixels()];
    let mut local = vec![0.0f64; dim * dim];
    let mut physical = vec![0.0f64; dim * dim];

    for (pixel, out) in determinants.iter_mut().enumerate() {
        let index = field.multi_index(pixel);

        for col in 0..dim {
            // `it.GetNext(col)` / `it.GetPrevious(col)` under ZeroFluxNeumann:
            // the neighbour index is clamped into the lattice.
            let last = field.size[col] - 1;
            let mut next = index.clone();
            next[col] = (index[col] + 1).min(last);
            let mut previous = index.clone();
            previous[col] = index[col].saturating_sub(1);

            let next = field.vector(field.linear_index(&next));
            let previous = field.vector(field.linear_index(&previous));
            for row in 0..dim {
                local[row * dim + col] = half[col] * (next[row] - previous[row]);
            }
        }

        // physicalGrad = localGrad · Directionᵀ, then `+ I` on the diagonal.
        for row in 0..dim {
            for col in 0..dim {
                physical[row * dim + col] = (0..dim)
                    .map(|j| field.direction[col * dim + j] * local[row * dim + j])
                    .sum();
            }
            physical[row * dim + row] += 1.0;
        }

        *out = determinant(&physical, dim);
    }

    fn build<T: Scalar>(size: &[usize], data: &[f64]) -> Result<Image> {
        let narrowed: Vec<T> = data.iter().map(|&x| T::from_f64(x)).collect();
        Ok(Image::from_vec(size, narrowed)?)
    }

    let component_id = displacement_field.pixel_id().component_id();
    let mut out = dispatch_scalar!(component_id, build, &field.size, &determinants)?;
    out.copy_geometry_from(displacement_field);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-12, "pixel {i}: {a} != {e}");
        }
    }

    /// The identity warp: `u ≡ 0`, so `dT/dx = I` and the determinant is `1`
    /// everywhere, boundary included.
    #[test]
    fn the_identity_displacement_has_jacobian_determinant_one() {
        let img = Image::from_vec_vector(&[4, 4], 2, vec![0.0f64; 32]).unwrap();
        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        assert_close(&out.to_f64_vec().unwrap(), &[1.0; 16]);
    }

    /// A constant field is a pure translation: its gradient vanishes and the
    /// determinant is `1` everywhere.
    #[test]
    fn a_constant_translation_field_has_jacobian_determinant_one() {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.push(3.0f64);
            data.push(-7.0f64);
        }
        let img = Image::from_vec_vector(&[4, 4], 2, data).unwrap();
        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        assert_close(&out.to_f64_vec().unwrap(), &[1.0; 16]);
    }

    /// `u(x, y) = (a·x, b·y)` on a unit lattice. In the interior the central
    /// difference is exact: `du/dx = diag(a, b)`, so the determinant is
    /// `(1 + a)(1 + b)`. With `a = 0.5` and `b = −0.25` that is `0.5625`.
    #[test]
    fn a_linear_field_has_the_analytic_jacobian_determinant_in_the_interior() {
        let (a, b) = (0.5f64, -0.25f64);
        let mut data = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                data.push(a * x as f64);
                data.push(b * y as f64);
            }
        }
        let img = Image::from_vec_vector(&[5, 5], 2, data).unwrap();
        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        let got = out.to_f64_vec().unwrap();
        for y in 1..4 {
            for x in 1..4 {
                assert!(
                    (got[y * 5 + x] - (1.0 + a) * (1.0 + b)).abs() < 1e-12,
                    "({x}, {y}): {}",
                    got[y * 5 + x]
                );
            }
        }
    }

    /// ZeroFluxNeumann clamps the out-of-bounds neighbour, so an edge pixel's
    /// central difference spans one lattice step but is still divided by two:
    /// it is *half* the interior derivative for a linear field.
    ///
    /// On `u(x, y) = (0.5x, −0.25y)` the corner `(0, 0)` sees `du/dx =
    /// diag(0.25, −0.125)` and the determinant is `1.25 · 0.875 = 1.09375`. The
    /// edge `(0, 2)` sees `diag(0.25, −0.25)` for `0.9375`.
    #[test]
    fn the_zero_flux_boundary_halves_the_edge_derivative() {
        let (a, b) = (0.5f64, -0.25f64);
        let mut data = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                data.push(a * x as f64);
                data.push(b * y as f64);
            }
        }
        let img = Image::from_vec_vector(&[5, 5], 2, data).unwrap();
        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        let got = out.to_f64_vec().unwrap();
        assert!((got[0] - 1.25 * 0.875).abs() < 1e-12, "corner: {}", got[0]);
        assert!(
            (got[2 * 5] - 1.25 * 0.75).abs() < 1e-12,
            "left edge: {}",
            got[2 * 5]
        );
    }

    /// `UseImageSpacing` scales each partial by `1/spacing[i]`. Doubling the
    /// x-spacing halves `du_x/dx`, so `(1 + a)` becomes `(1 + a/2)`.
    #[test]
    fn use_image_spacing_scales_each_partial_by_the_inverse_spacing() {
        let (a, b) = (0.5f64, -0.25f64);
        let mut data = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                data.push(a * x as f64);
                data.push(b * y as f64);
            }
        }
        let mut img = Image::from_vec_vector(&[5, 5], 2, data).unwrap();
        img.set_spacing(&[2.0, 4.0]).unwrap();

        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        let got = out.to_f64_vec().unwrap();
        let expected = (1.0 + a / 2.0) * (1.0 + b / 4.0);
        assert!(
            (got[2 * 5 + 2] - expected).abs() < 1e-12,
            "{}",
            got[2 * 5 + 2]
        );
    }

    /// `use_image_spacing = false` with no explicit weights uses `1.0` on every
    /// axis, so the spacing is ignored entirely.
    #[test]
    fn use_image_spacing_off_ignores_the_spacing() {
        let (a, b) = (0.5f64, -0.25f64);
        let mut data = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                data.push(a * x as f64);
                data.push(b * y as f64);
            }
        }
        let mut img = Image::from_vec_vector(&[5, 5], 2, data).unwrap();
        img.set_spacing(&[2.0, 4.0]).unwrap();

        let settings = DisplacementFieldJacobianDeterminantSettings {
            use_image_spacing: false,
            derivative_weights: Vec::new(),
        };
        let out = displacement_field_jacobian_determinant(&img, &settings).unwrap();
        let got = out.to_f64_vec().unwrap();
        assert!((got[2 * 5 + 2] - (1.0 + a) * (1.0 + b)).abs() < 1e-12);
    }

    /// Non-empty `DerivativeWeights` wins over `UseImageSpacing`, which
    /// SimpleITK leaves at its default `true`: `SetDerivativeWeights` runs second
    /// and sets `m_UseImageSpacing = false` (`hxx:55`). The yaml's `2d_weights`
    /// test passes exactly these weights.
    #[test]
    fn explicit_derivative_weights_override_use_image_spacing() {
        let (a, b) = (0.5f64, -0.25f64);
        let mut data = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                data.push(a * x as f64);
                data.push(b * y as f64);
            }
        }
        let mut img = Image::from_vec_vector(&[5, 5], 2, data).unwrap();
        img.set_spacing(&[2.0, 4.0]).unwrap();

        let settings = DisplacementFieldJacobianDeterminantSettings {
            use_image_spacing: true,
            derivative_weights: vec![0.1, 10.0],
        };
        let out = displacement_field_jacobian_determinant(&img, &settings).unwrap();
        let got = out.to_f64_vec().unwrap();
        // The weights, not 1/spacing = (0.5, 0.25), scale the partials.
        let expected = (1.0 + 0.1 * a) * (1.0 + 10.0 * b);
        assert!(
            (got[2 * 5 + 2] - expected).abs() < 1e-12,
            "{}",
            got[2 * 5 + 2]
        );
    }

    /// The direction cosines enter as `physicalGrad = localGrad · Directionᵀ`.
    /// A 90° rotation `D = [[0, −1], [1, 0]]` maps the interior gradient
    /// `diag(a, b)` to `[[0, a], [−b, 0]]`, whose `+ I` determinant is
    /// `1 + a·b`. With `a = 0.5`, `b = −0.25` that is `0.875`, against the
    /// identity-direction value `(1 + a)(1 + b) = 0.5625`.
    #[test]
    fn the_direction_cosines_rotate_the_local_gradient() {
        let (a, b) = (0.5f64, -0.25f64);
        let mut data = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                data.push(a * x as f64);
                data.push(b * y as f64);
            }
        }
        let mut img = Image::from_vec_vector(&[5, 5], 2, data).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        let got = out.to_f64_vec().unwrap();
        assert!(
            (got[2 * 5 + 2] - (1.0 + a * b)).abs() < 1e-12,
            "{}",
            got[2 * 5 + 2]
        );
    }

    /// A 3-D shear `u(x, y, z) = (c·y, 0, 0)`: `du/dx` has a single off-diagonal
    /// entry, so `det(I + du/dx) = 1` — a shear preserves volume.
    #[test]
    fn a_three_dimensional_shear_preserves_volume() {
        let c = 0.75f64;
        let mut data = Vec::new();
        for _z in 0..4 {
            for y in 0..4 {
                for _x in 0..4 {
                    data.push(c * y as f64);
                    data.push(0.0);
                    data.push(0.0);
                }
            }
        }
        let img = Image::from_vec_vector(&[4, 4, 4], 3, data).unwrap();
        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        assert_close(&out.to_f64_vec().unwrap(), &[1.0; 64]);
    }

    #[test]
    fn a_float32_field_yields_a_float32_scalar_image() {
        let img = Image::from_vec_vector(&[3, 3], 2, vec![0.0f32; 18]).unwrap();
        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[1.0f32; 9]);
    }

    #[test]
    fn the_output_keeps_the_inputs_geometry() {
        let mut img = Image::from_vec_vector(&[3, 2], 2, vec![0.0f64; 12]).unwrap();
        img.set_spacing(&[0.5, 0.25]).unwrap();
        img.set_origin(&[1.0, -2.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let out = displacement_field_jacobian_determinant(&img, &Default::default()).unwrap();
        assert_eq!(out.size(), &[3, 2]);
        assert_eq!(out.spacing(), &[0.5, 0.25]);
        assert_eq!(out.origin(), &[1.0, -2.0]);
        assert_eq!(out.direction(), &[0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn a_scalar_input_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            displacement_field_jacobian_determinant(&img, &Default::default()).unwrap_err(),
            FilterError::Core(crate::core::Error::RequiresVectorPixelType(
                PixelId::Float64
            ))
        ));
    }

    #[test]
    fn derivative_weights_of_the_wrong_length_are_rejected() {
        let img = Image::from_vec_vector(&[3, 3], 2, vec![0.0f64; 18]).unwrap();
        let settings = DisplacementFieldJacobianDeterminantSettings {
            use_image_spacing: true,
            derivative_weights: vec![1.0, 1.0, 1.0],
        };
        assert!(matches!(
            displacement_field_jacobian_determinant(&img, &settings).unwrap_err(),
            FilterError::DimensionLength {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn the_defaults_match_the_yaml() {
        let settings = DisplacementFieldJacobianDeterminantSettings::default();
        assert!(settings.use_image_spacing);
        assert!(settings.derivative_weights.is_empty());
    }

    #[test]
    fn the_determinant_matches_vnl_for_each_direct_size() {
        assert_eq!(determinant(&[3.0], 1), 3.0);
        assert_eq!(determinant(&[1.0, 2.0, 3.0, 4.0], 2), -2.0);
        // det [[6,1,1],[4,-2,5],[2,8,7]] = -306.
        assert_eq!(
            determinant(&[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0], 3),
            -306.0
        );
        // A 4×4 upper-triangular matrix: the product of its diagonal.
        let mut m = vec![0.0; 16];
        for (i, d) in [2.0, 3.0, 4.0, 5.0].into_iter().enumerate() {
            m[i * 4 + i] = d;
            for c in i + 1..4 {
                m[i * 4 + c] = 1.0;
            }
        }
        assert_eq!(determinant(&m, 4), 120.0);
    }

    /// The `n > 4` fallback agrees with the direct formulas on a matrix whose
    /// determinant is known: a 5×5 upper-triangular one.
    #[test]
    fn the_gaussian_fallback_agrees_above_dimension_four() {
        let mut m = vec![0.0; 25];
        for (i, d) in [2.0, 3.0, 4.0, 5.0, 6.0].into_iter().enumerate() {
            m[i * 5 + i] = d;
            for c in i + 1..5 {
                m[i * 5 + c] = 1.0;
            }
        }
        assert!((determinant(&m, 5) - 720.0).abs() < 1e-9);
    }
}
