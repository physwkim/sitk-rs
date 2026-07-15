//! `InverseDisplacementFieldImageFilter`
//! (`itkInverseDisplacementFieldImageFilter.h(.hxx)`): invert a displacement
//! field by fitting a thin-plate-spline kernel transform to a subsampled set of
//! landmark correspondences and resampling it on the output lattice.
//!
//! # The algorithm
//!
//! 1. **Subsample** (`PrepareKernelBaseSpline`,
//!    `itkInverseDisplacementFieldImageFilter.hxx:113-143`). The input field is
//!    resampled onto a grid of size `size[i] / SubsamplingFactor` (integer
//!    division) with spacing `spacing[i] * SubsamplingFactor`, keeping the
//!    input's origin **and direction** (see the fixed defect below), through an
//!    `itk::ResampleImageFilter` with the default identity transform and linear
//!    interpolation. A sample point outside the input's buffer takes the
//!    resampler's default pixel value, the zero vector.
//!
//! 2. **Landmarks** (`hxx:162-181`). For each subsampled pixel, the *target*
//!    landmark is its physical point `t` and the *source* landmark is the
//!    deformed point `s = t + u(t)`. So the transform is fitted to carry each
//!    deformed point back to where it came from.
//!
//! 3. **Fit** a `ThinPlateSplineKernelTransform<double, dim>` (the default
//!    `m_KernelTransform`, `hxx:43-45`) through `ComputeWMatrix`
//!    (`itkKernelTransform.hxx:144-156`) — see [`solve_kernel_transform`].
//!
//! 4. **Resample** (`GenerateData`, `hxx:231-249`). For every output lattice
//!    point `y`, the inverse displacement is `T(y) − y`.
//!
//! # Output geometry
//!
//! `GenerateOutputInformation` (`hxx:284-304`) sets the output's size, spacing,
//! and origin from `Size`/`OutputSpacing`/`OutputOrigin`, and leaves the
//! **direction** as `Superclass::GenerateOutputInformation` copied it from the
//! input. SimpleITK exposes no direction setter for this filter, so the output
//! always carries the input field's direction cosines. This port does the same.
//!
//! # Fixed here (upstream bug §1.34): the subsampling grid honours the direction
//!
//! `PrepareKernelBaseSpline` sets the resampler's size, start index, spacing,
//! and origin, but never its `OutputDirection`, which
//! `ResampleImageFilter`'s constructor leaves as the **identity**
//! (`itkResampleImageFilter.hxx:51`, `m_OutputDirection.SetIdentity()`).
//!
//! Two things follow when the input field's direction is not the identity.
//! The resampler walks an axis-aligned grid through physical space rather than
//! the input's own lattice, so sample points that should be interior fall
//! outside the buffer and silently pick up the zero default pixel value. And
//! `sampledInput->TransformIndexToPhysicalPoint` (`hxx:168`) then reads those
//! axis-aligned points back out as the *target* landmarks, so the landmark set
//! is not a subset of the input lattice at all.
//!
//! This port builds the subsampled grid with the input field's own direction
//! cosines — the upstream fix PR InsightSoftwareConsortium/ITK#6577, which sets
//! `resampler->SetOutputDirection(inputImage->GetDirection())`. The subsampled
//! physical points are then exactly every `SubsamplingFactor`-th input lattice
//! point, so the interior samples stay in the buffer and the target landmarks
//! are genuine input points, for any direction. With an identity direction —
//! the overwhelmingly common case, and the only one SimpleITK's own test covers
//! — the fix is a no-op. See
//! `the_subsampling_grid_honours_the_input_direction`.
//!
//! # Divergences
//!
//! - A `subsampling_factor` of zero is an integer division by zero upstream,
//!   which is undefined behavior in C++. This port rejects it with
//!   [`FilterError::InvalidSubsamplingFactor`].
//!
//! - `ComputeWMatrix` takes an SVD of the `L` matrix and multiplies by its
//!   pseudo-inverse. `L` is symmetric by construction, so this port solves
//!   through a symmetric eigendecomposition, which is exactly equivalent — see
//!   [`crate::filters::linalg::symmetric_pseudo_inverse_solve`].
//!
//! - The resampler's fast path for linear transforms interpolates the input
//!   continuous index along each output scan line
//!   (`itkResampleImageFilter.hxx:429-513`) instead of mapping each point
//!   independently. The two are the same affine map; this port maps each point,
//!   so results differ only by floating-point rounding.

use crate::core::Image;

use super::{Field, field_to_image};
use crate::filters::linalg::symmetric_pseudo_inverse_solve;
use crate::filters::{FilterError, Result};

/// The affine and deformable parts of a fitted `ThinPlateSplineKernelTransform`,
/// as `ReorganizeW` (`itkKernelTransform.hxx:287-317`) splits the solution
/// vector.
struct KernelTransform {
    dim: usize,
    /// The source landmarks the kernel is centred on, `dim` coordinates each.
    source: Vec<f64>,
    /// `m_DMatrix`, `dim × landmarks`, row-major: the deformable coefficients.
    d_matrix: Vec<f64>,
    /// `m_AMatrix`, `dim × dim`, row-major: the affine rotation part.
    a_matrix: Vec<f64>,
    /// `m_BVector`, length `dim`: the affine translation part.
    b_vector: Vec<f64>,
}

impl KernelTransform {
    fn landmarks(&self) -> usize {
        self.source.len() / self.dim
    }

    /// `KernelTransform::TransformPoint` (`itkKernelTransform.hxx:322-348`) with
    /// `ThinPlateSplineKernelTransform::ComputeDeformationContribution`
    /// (`itkThinPlateSplineKernelTransform.hxx:37-56`), whose `G(x) = ‖x‖ · I`
    /// collapses the general double loop to `result[o] += r · D(o, lnd)`.
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut result = vec![0.0f64; dim];

        for lnd in 0..self.landmarks() {
            let source = &self.source[lnd * dim..(lnd + 1) * dim];
            let r = (0..dim)
                .map(|d| (point[d] - source[d]).powi(2))
                .sum::<f64>()
                .sqrt();
            for (o, value) in result.iter_mut().enumerate() {
                *value += r * self.d_matrix[o * self.landmarks() + lnd];
            }
        }

        for (i, value) in result.iter_mut().enumerate() {
            for (j, &x) in point.iter().enumerate().take(dim) {
                *value += self.a_matrix[i * dim + j] * x;
            }
            *value += self.b_vector[i] + point[i];
        }
        result
    }
}

/// `KernelTransform::ComputeWMatrix` (`itkKernelTransform.hxx:144-156`) for the
/// thin-plate-spline kernel, whose `ComputeG(x) = ‖x‖ · I`.
///
/// With `N` landmarks in `d` dimensions the system is the classic bordered TPS
/// matrix, laid out by `ComputeL`/`ComputeK`/`ComputeP` (`hxx:161-256`) as
///
/// ```text
/// L = [ K   P ]        K(i,j) = ‖sᵢ − sⱼ‖ · I   (i ≠ j)
///     [ Pᵀ  0 ]        K(i,i) = Stiffness · I   (Stiffness defaults to 0)
///                      Pᵢ     = [ sᵢ[0]·I … sᵢ[d−1]·I  I ]
/// ```
///
/// solved against `Y = [ t₀ − s₀ … t_{N−1} − s_{N−1}  0 … 0 ]ᵀ` (`ComputeY`,
/// `hxx:261-282`; the displacements come from `ComputeD`, `hxx:121-139`). The
/// solution is split by `ReorganizeW` into the deformable `D`, the affine `A`,
/// and the translation `B`.
///
/// `N == 0` leaves `L` an all-zero `d(d+1)` square, whose pseudo-inverse is
/// zero, so the fitted transform is the identity — the same well-defined result
/// upstream reaches through `wmax == 0 ⟹ rcond == 0`.
fn solve_kernel_transform(dim: usize, source: Vec<f64>, target: &[f64]) -> KernelTransform {
    let landmarks = source.len() / dim;
    let deformable = dim * landmarks;
    let affine = dim * (dim + 1);
    let n = deformable + affine;

    let mut l = vec![0.0f64; n * n];

    // K: the reflexive blocks stay zero (Stiffness == 0); the off-diagonal
    // blocks are ‖sᵢ − sⱼ‖ on the diagonal of each d × d block.
    for i in 0..landmarks {
        for j in i + 1..landmarks {
            let r = (0..dim)
                .map(|d| (source[i * dim + d] - source[j * dim + d]).powi(2))
                .sum::<f64>()
                .sqrt();
            for a in 0..dim {
                l[(i * dim + a) * n + (j * dim + a)] = r;
                l[(j * dim + a) * n + (i * dim + a)] = r;
            }
        }
    }

    // P in the top-right block and Pᵀ in the bottom-left.
    for i in 0..landmarks {
        for j in 0..dim {
            for a in 0..dim {
                let row = i * dim + a;
                let col = deformable + j * dim + a;
                l[row * n + col] = source[i * dim + j];
                l[col * n + row] = source[i * dim + j];
            }
        }
        for a in 0..dim {
            let row = i * dim + a;
            let col = deformable + dim * dim + a;
            l[row * n + col] = 1.0;
            l[col * n + row] = 1.0;
        }
    }

    let mut y = vec![0.0f64; n];
    for i in 0..deformable {
        y[i] = target[i] - source[i];
    }

    let w = symmetric_pseudo_inverse_solve(l, &y, n);

    let mut d_matrix = vec![0.0f64; dim * landmarks];
    for lnd in 0..landmarks {
        for d in 0..dim {
            d_matrix[d * landmarks + lnd] = w[lnd * dim + d];
        }
    }
    let mut a_matrix = vec![0.0f64; dim * dim];
    for j in 0..dim {
        for i in 0..dim {
            a_matrix[i * dim + j] = w[deformable + j * dim + i];
        }
    }
    let b_vector = w[deformable + dim * dim..n].to_vec();

    KernelTransform {
        dim,
        source,
        d_matrix,
        a_matrix,
        b_vector,
    }
}

/// Compute the inverse of `displacement_field` on the lattice described by
/// `size`, `output_origin`, and `output_spacing`.
///
/// The yaml's defaults are `Size = (0, 0, 0)`, `OutputOrigin = (0, 0, 0)`,
/// `OutputSpacing = (1, 1, 1)`, and `SubsamplingFactor = 16`; there is no useful
/// dimension-agnostic default for the three `dim_vec` members, so this port asks
/// for all four. SimpleITK's `SetReferenceImage(img)` is
/// `(img.size(), img.origin(), img.spacing())`.
///
/// The output's direction cosines are the input field's, which is the only
/// direction upstream can produce (see the module docs).
///
/// Errors when `displacement_field` is not a real-component vector image with
/// one component per dimension, when any of the three per-axis parameters has
/// the wrong length, or when `subsampling_factor` is zero.
pub fn inverse_displacement_field(
    displacement_field: &Image,
    size: &[usize],
    output_origin: &[f64],
    output_spacing: &[f64],
    subsampling_factor: u32,
) -> Result<Image> {
    let forward = Field::from_image(displacement_field)?;
    let dim = forward.dim;

    for length in [size.len(), output_origin.len(), output_spacing.len()] {
        if length != dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: length,
            });
        }
    }
    if subsampling_factor == 0 {
        return Err(FilterError::InvalidSubsamplingFactor(subsampling_factor));
    }
    let factor = subsampling_factor as usize;

    // The subsampled landmark grid, sharing the input field's direction cosines
    // (upstream fix PR #6577 sets the resampler's OutputDirection; see the
    // module docs). Its physical points are therefore a subset of the input
    // lattice, so `TransformIndexToPhysicalPoint` reads back genuine input
    // points and the interior samples stay inside the buffer.
    let sub_size: Vec<usize> = forward.size.iter().map(|&s| s / factor).collect();
    let sub_spacing: Vec<f64> = forward.spacing.iter().map(|&s| s * factor as f64).collect();
    let landmarks: usize = sub_size.iter().product();

    let mut target = vec![0.0f64; landmarks * dim];
    let mut source = vec![0.0f64; landmarks * dim];
    for landmark in 0..landmarks {
        let mut rest = landmark;
        let mut index = vec![0.0f64; dim];
        for d in 0..dim {
            index[d] = (rest % sub_size[d]) as f64;
            rest /= sub_size[d];
        }

        // p = origin + Direction * (sub_spacing ⊙ index), with the input's
        // direction — matching the output-grid mapping below.
        let scaled: Vec<f64> = (0..dim).map(|d| index[d] * sub_spacing[d]).collect();
        let rotated = crate::core::matrix::mat_vec(&forward.direction, &scaled, dim);
        let point: Vec<f64> = (0..dim).map(|d| forward.origin[d] + rotated[d]).collect();

        // The resampler's identity transform, then linear interpolation of the
        // input field, or its zero default pixel value outside the buffer.
        let cindex = forward.point_to_continuous_index(&point);
        let value = if forward.is_inside_buffer(&cindex) {
            forward.evaluate_at_continuous_index(&cindex)
        } else {
            vec![0.0; dim]
        };

        for d in 0..dim {
            target[landmark * dim + d] = point[d];
            source[landmark * dim + d] = point[d] + value[d];
        }
    }

    let transform = solve_kernel_transform(dim, source, &target);

    let pixels: usize = size.iter().product();
    let mut data = vec![0.0f64; pixels * dim];
    for pixel in 0..pixels {
        let mut rest = pixel;
        let mut index = vec![0.0f64; dim];
        for d in 0..dim {
            index[d] = (rest % size[d]) as f64;
            rest /= size[d];
        }

        // p = origin + Direction * (spacing ⊙ index), with the input's direction.
        let scaled: Vec<f64> = (0..dim).map(|d| index[d] * output_spacing[d]).collect();
        let rotated = crate::core::matrix::mat_vec(&forward.direction, &scaled, dim);
        let point: Vec<f64> = (0..dim).map(|d| output_origin[d] + rotated[d]).collect();

        let mapped = transform.transform_point(&point);
        for d in 0..dim {
            data[pixel * dim + d] = mapped[d] - point[d];
        }
    }

    field_to_image(
        size,
        data,
        displacement_field.pixel_id().component_id(),
        output_spacing,
        output_origin,
        &forward.direction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    fn field_1d(values: &[f64]) -> Image {
        Image::from_vec_vector(&[values.len()], 1, values.to_vec()).unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < tolerance, "{actual:?} != {expected:?}");
        }
    }

    /// Every displacement is zero, so `Y = 0` and the whole solution vector is
    /// zero: `T` is the identity and the inverse field is zero.
    #[test]
    fn a_zero_field_inverts_to_the_zero_field() {
        let field = Image::from_vec_vector(&[4, 4], 2, vec![0.0f64; 32]).unwrap();
        let out = inverse_displacement_field(&field, &[4, 4], &[0.0, 0.0], &[1.0, 1.0], 2).unwrap();

        assert_eq!(out.pixel_id(), PixelId::VectorFloat64);
        assert_close(&out.components_to_f64_vec(), &[0.0; 32], 1e-12);
    }

    /// A constant translation `u ≡ t` gives every landmark the displacement
    /// `−t`, and the unique solution of the bordered system is `D = 0`, `A = 0`,
    /// `B = −t`: the fitted `T(y) = y − t` reproduces the analytic inverse
    /// **everywhere**, not just inside the landmark hull.
    ///
    /// Landmarks here: subsampled size `4 / 2 = 2` at physical `0` and `2`, with
    /// `u = 0.5`, so `s = (0.5, 2.5)` and `t = (0, 2)`.
    /// `Pᵀd = 0` forces `d₀ + d₁ = 0` and `0.5d₀ + 2.5d₁ = 0`, hence `d = 0`;
    /// the two remaining rows read `0.5a + b = −0.5` and `2.5a + b = −0.5`, so
    /// `a = 0` and `b = −0.5`.
    #[test]
    fn a_constant_translation_field_inverts_to_the_negated_translation() {
        let field = field_1d(&[0.5; 4]);
        let out = inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 2).unwrap();
        assert_close(&out.components_to_f64_vec(), &[-0.5; 4], 1e-9);
    }

    /// The same in 2-D: `u ≡ (0.5, −0.25)` on a 4×4 field subsampled by 2 gives
    /// four landmarks and the unique affine solution `B = −t`.
    #[test]
    fn a_constant_translation_field_inverts_to_the_negated_translation_in_two_dimensions() {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.push(0.5);
            data.push(-0.25);
        }
        let field = Image::from_vec_vector(&[4, 4], 2, data).unwrap();
        let out = inverse_displacement_field(&field, &[4, 4], &[0.0, 0.0], &[1.0, 1.0], 2).unwrap();

        let expected: Vec<f64> = (0..16).flat_map(|_| [-0.5, 0.25]).collect();
        assert_close(&out.components_to_f64_vec(), &expected, 1e-9);
    }

    /// A linear field `u(x) = c·x` maps `x ↦ (1 + c)x`, whose inverse is
    /// `y ↦ y / (1 + c)`, an affine map the spline reproduces exactly. So the
    /// inverse displacement is `−c·y / (1 + c)`.
    ///
    /// With `c = 0.25` on a size-4 field subsampled by 2: `t = (0, 2)`,
    /// `u = (0, 0.5)`, `s = (0, 2.5)`, displacements `(0, −0.5)`. `Pᵀd = 0`
    /// reads `2.5d₁ = 0` and `d₀ + d₁ = 0`, so `d = 0`; then `b = 0` and
    /// `2.5a = −0.5`, i.e. `a = −0.2`. `T(y) = 0.8y`, so the inverse
    /// displacement is `−0.2y`, matching `−0.25y / 1.25`.
    #[test]
    fn a_linear_field_inverts_to_the_analytic_affine_inverse() {
        let field = field_1d(&[0.0, 0.25, 0.5, 0.75]);
        let out = inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 2).unwrap();
        assert_close(&out.components_to_f64_vec(), &[0.0, -0.2, -0.4, -0.6], 1e-9);
    }

    /// The fix, pinned. With `direction = [−1]` the input lattice (origin `0`,
    /// spacing `1`, size `4`) occupies physical points `0, −1, −2, −3`.
    ///
    /// The subsampled grid now shares that direction, so its two points
    /// (`sub_spacing = 2`) are `origin + Direction·(2·index)`, i.e. `t = (0,
    /// −2)` — genuine input-lattice points, both interior. The constant field
    /// samples `0.5` at each, so `s = t + u = (0.5, −1.5)` and every
    /// displacement `t − s = −0.5`: a pure translation, fitted exactly as
    /// `T(y) = y − 0.5` (`D = 0`, `A = 0`, `B = −0.5`, the argument of
    /// [`a_constant_translation_field_inverts_to_the_negated_translation`]).
    /// The inverse displacement is therefore the constant `−0.5` everywhere.
    ///
    /// Under the bug the subsampled grid used an identity direction, so its
    /// second point landed at physical `+2` — outside the `[−3, 0]` lattice —
    /// and took the zero default, giving the raster-order-dependent, direction-
    /// blind field `y/3 − 2/3` instead.
    #[test]
    fn the_subsampling_grid_honours_the_input_direction() {
        let mut field = field_1d(&[0.5; 4]);
        field.set_direction(&[-1.0]).unwrap();

        let out = inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 2).unwrap();
        // Output points y = origin + Direction·(spacing·index) = −index; the
        // inverse displacement is the constant −0.5 at every one of them.
        assert_close(&out.components_to_f64_vec(), &[-0.5; 4], 1e-9);
    }

    /// `size[i] / SubsamplingFactor` is integer division, so a factor larger
    /// than the field leaves **zero** landmarks. `L` is then an all-zero
    /// `d(d+1)` square, its pseudo-inverse is zero, and the fitted transform is
    /// the identity — a zero inverse field.
    #[test]
    fn a_subsampling_factor_larger_than_the_field_leaves_no_landmarks() {
        let field = field_1d(&[0.5; 4]);
        let out = inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 8).unwrap();
        assert_eq!(out.components_to_f64_vec(), vec![0.0; 4]);
    }

    /// The transform interpolates its landmarks exactly: `T(sᵢ) = tᵢ`. Sampling
    /// the output at the source landmarks `0.5` and `2.5` of the constant-`0.5`
    /// fixture must give back `−0.5` (the displacement carrying `s` to `t`).
    #[test]
    fn the_fitted_transform_interpolates_its_landmarks() {
        let field = field_1d(&[0.5; 4]);
        // Output lattice {0.5, 2.5}: origin 0.5, spacing 2.
        let out = inverse_displacement_field(&field, &[2], &[0.5], &[2.0], 2).unwrap();
        assert_close(&out.components_to_f64_vec(), &[-0.5, -0.5], 1e-9);
    }

    /// The output takes `Size`, `OutputSpacing`, and `OutputOrigin` from the
    /// parameters, and its direction from the input field.
    #[test]
    fn the_output_geometry_comes_from_the_parameters_and_the_inputs_direction() {
        let mut field = Image::from_vec_vector(&[4, 4], 2, vec![0.0f64; 32]).unwrap();
        field.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let out =
            inverse_displacement_field(&field, &[3, 5], &[1.5, -2.0], &[0.25, 4.0], 2).unwrap();

        assert_eq!(out.size(), &[3, 5]);
        assert_eq!(out.spacing(), &[0.25, 4.0]);
        assert_eq!(out.origin(), &[1.5, -2.0]);
        assert_eq!(out.direction(), &[0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn a_float32_field_keeps_its_component_type() {
        let field = Image::from_vec_vector(&[4], 1, vec![0.5f32; 4]).unwrap();
        let out = inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 2).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_close(
            &out.components_to_f64_vec(),
            &[-0.5; 4],
            1e-6, // f32 storage
        );
    }

    #[test]
    fn a_scalar_input_is_rejected() {
        let field = Image::new(&[4], PixelId::Float64);
        assert!(matches!(
            inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 2).unwrap_err(),
            FilterError::Core(crate::core::Error::RequiresVectorPixelType(
                PixelId::Float64
            ))
        ));
    }

    #[test]
    fn a_zero_subsampling_factor_is_rejected() {
        let field = field_1d(&[0.5; 4]);
        assert!(matches!(
            inverse_displacement_field(&field, &[4], &[0.0], &[1.0], 0).unwrap_err(),
            FilterError::InvalidSubsamplingFactor(0)
        ));
    }

    #[test]
    fn a_per_axis_parameter_of_the_wrong_length_is_rejected() {
        let field = field_1d(&[0.5; 4]);
        assert!(matches!(
            inverse_displacement_field(&field, &[4, 4], &[0.0], &[1.0], 2).unwrap_err(),
            FilterError::DimensionLength {
                expected: 1,
                got: 2
            }
        ));
        assert!(matches!(
            inverse_displacement_field(&field, &[4], &[0.0, 0.0], &[1.0], 2).unwrap_err(),
            FilterError::DimensionLength {
                expected: 1,
                got: 2
            }
        ));
        assert!(matches!(
            inverse_displacement_field(&field, &[4], &[0.0], &[1.0, 1.0], 2).unwrap_err(),
            FilterError::DimensionLength {
                expected: 1,
                got: 2
            }
        ));
    }
}
