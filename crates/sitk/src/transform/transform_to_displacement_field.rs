//! `itk::TransformToDisplacementFieldFilter`
//! (`Modules/Filtering/DisplacementField/include/itkTransformToDisplacementFieldFilter.h(.hxx)`):
//! sample a [`TransformBase`] onto a grid, storing at each grid point the
//! displacement `T(p) − p` that carries that point's physical position `p` to
//! its transformed position.
//!
//! The output is a vector image whose component count is the spatial dimension
//! — the field a [`DisplacementFieldTransform`] consumes.
//!
//! [`DisplacementFieldTransform`]: crate::transform::DisplacementFieldTransform

use crate::core::{Image, PixelId};

use crate::transform::error::{Result, TransformError};
use crate::transform::interpolator::{affine_apply, index_to_physical_matrix};
use crate::transform::resample::increment;
use crate::transform::transform::TransformBase;

/// `TransformToDisplacementFieldFilter`: `displacement(p) = T(p) − p` sampled
/// on an explicit output grid.
///
/// The grid (size / spacing / origin / direction) defaults to SimpleITK's:
/// `Size` 64 per axis (`TransformToDisplacementFieldFilter.yaml`'s
/// `std::vector<unsigned int>(3, 64)`, taken here per the transform's own
/// dimension), unit spacing, zero origin, identity direction. ITK's own default
/// `m_Size` is zero-filled, which produces an empty image; SimpleITK's member
/// default is what a caller of this API actually sees, so it is what this port
/// defaults to.
///
/// ITK's `UseReferenceImage` / `SetReferenceImage` input pair is collapsed, as
/// in SimpleITK, into [`set_reference_image`](Self::set_reference_image), which
/// copies the four grid parameters eagerly.
pub struct TransformToDisplacementFieldFilter {
    size: Option<Vec<usize>>,
    spacing: Option<Vec<f64>>,
    origin: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    output_pixel_type: PixelId,
}

impl Default for TransformToDisplacementFieldFilter {
    fn default() -> Self {
        Self {
            size: None,
            spacing: None,
            origin: None,
            direction: None,
            output_pixel_type: PixelId::VectorFloat64,
        }
    }
}

impl TransformToDisplacementFieldFilter {
    /// A filter with SimpleITK's default grid and a
    /// [`PixelId::VectorFloat64`] output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the whole output grid (size, spacing, origin, direction) from a
    /// reference image — SimpleITK's `SetReferenceImage` custom method.
    pub fn set_reference_image(&mut self, reference: &Image) -> &mut Self {
        self.size = Some(reference.size().to_vec());
        self.spacing = Some(reference.spacing().to_vec());
        self.origin = Some(reference.origin().to_vec());
        self.direction = Some(reference.direction().to_vec());
        self
    }

    /// Override the output size (default: 64 per axis).
    pub fn set_size(&mut self, size: Vec<usize>) -> &mut Self {
        self.size = Some(size);
        self
    }

    /// Override the output spacing (default: 1 per axis).
    pub fn set_output_spacing(&mut self, spacing: Vec<f64>) -> &mut Self {
        self.spacing = Some(spacing);
        self
    }

    /// Override the output origin (default: 0 per axis).
    pub fn set_output_origin(&mut self, origin: Vec<f64>) -> &mut Self {
        self.origin = Some(origin);
        self
    }

    /// Override the output direction (row-major `dim x dim`; default:
    /// identity).
    pub fn set_output_direction(&mut self, direction: Vec<f64>) -> &mut Self {
        self.direction = Some(direction);
        self
    }

    /// Choose the output pixel type. Only [`PixelId::VectorFloat32`] and
    /// [`PixelId::VectorFloat64`] are supported, as SimpleITK's
    /// `OutputPixelType` documents; anything else is rejected by
    /// [`execute`](Self::execute) with
    /// [`TransformError::UnsupportedDisplacementFieldPixelType`].
    pub fn set_output_pixel_type(&mut self, id: PixelId) -> &mut Self {
        self.output_pixel_type = id;
        self
    }

    /// Sample `transform` onto the output grid.
    ///
    /// # Linear fast path
    ///
    /// `DynamicThreadedGenerateData` (itkTransformToDisplacementFieldFilter.hxx:151-168)
    /// branches on `transform->IsLinear()`. The linear branch
    /// (`LinearThreadedGenerateData`, hxx:213-273) evaluates the transform only
    /// at the two ends of each scan line — continuous indices
    /// `(0, j, k, …)` and `(size[0], j, k, …)`, both taken from the *largest
    /// possible region* so a split region does not change the numbers — and
    /// linearly interpolates between the two displacements with
    /// `alpha = i / size[0]`. Because `T(p) − p` is affine in `p` exactly when
    /// `T` is linear, and `p` is affine in the index, that interpolation is
    /// exact rather than an approximation; ITK takes it for speed and for
    /// scan-line-consistent rounding. This port reproduces both branches so the
    /// floating-point results match, not merely the mathematics.
    pub fn execute<T: TransformBase>(&self, transform: &T) -> Result<Image> {
        let dim = transform.dimension();

        if !matches!(
            self.output_pixel_type,
            PixelId::VectorFloat32 | PixelId::VectorFloat64
        ) {
            return Err(TransformError::UnsupportedDisplacementFieldPixelType(
                self.output_pixel_type,
            ));
        }

        let size = self.size.clone().unwrap_or_else(|| vec![64; dim]);
        let spacing = self.spacing.clone().unwrap_or_else(|| vec![1.0; dim]);
        let origin = self.origin.clone().unwrap_or_else(|| vec![0.0; dim]);
        let direction = self
            .direction
            .clone()
            .unwrap_or_else(|| crate::core::matrix::identity(dim));

        if size.len() != dim
            || spacing.len() != dim
            || origin.len() != dim
            || direction.len() != dim * dim
        {
            return Err(TransformError::DimensionMismatch);
        }

        let index_to_phys = index_to_physical_matrix(&direction, &spacing, dim);
        let phys = |index: &[f64]| affine_apply(&index_to_phys, index, &origin, dim);
        let displacement_at = |index: &[f64]| {
            let p = phys(index);
            let t = transform.transform_point(&p);
            let d: Vec<f64> = (0..dim).map(|k| t[k] - p[k]).collect();
            d
        };

        let n_pixels: usize = size.iter().product();
        let mut components = vec![0.0f64; n_pixels * dim];

        if transform.is_linear() {
            self.fill_linear(&mut components, &size, dim, displacement_at);
        } else {
            let mut index = vec![0usize; dim];
            for pixel in components.chunks_exact_mut(dim) {
                let index_f: Vec<f64> = index.iter().map(|&i| i as f64).collect();
                pixel.copy_from_slice(&displacement_at(&index_f));
                increment(&mut index, &size);
            }
        }

        let mut field = match self.output_pixel_type {
            PixelId::VectorFloat32 => Image::from_vec_vector(
                &size,
                dim,
                components.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            ),
            // `set_output_pixel_type`'s contract, enforced above.
            _ => Image::from_vec_vector(&size, dim, components),
        }
        .map_err(TransformError::Core)?;

        field.set_spacing(&spacing).map_err(TransformError::Core)?;
        field.set_origin(&origin).map_err(TransformError::Core)?;
        field
            .set_direction(&direction)
            .map_err(TransformError::Core)?;
        Ok(field)
    }

    /// `LinearThreadedGenerateData`: two transform evaluations per scan line,
    /// linearly interpolated along the fastest axis.
    fn fill_linear(
        &self,
        components: &mut [f64],
        size: &[usize],
        dim: usize,
        displacement_at: impl Fn(&[f64]) -> Vec<f64>,
    ) {
        let size0 = size[0];
        let n_lines: usize = size[1..].iter().product();
        let mut higher = vec![0usize; dim - 1];
        let mut index_f = vec![0.0f64; dim];

        for line in 0..n_lines {
            for (d, &i) in higher.iter().enumerate() {
                index_f[d + 1] = i as f64;
            }
            index_f[0] = 0.0;
            let start = displacement_at(&index_f);
            index_f[0] = size0 as f64;
            let end = displacement_at(&index_f);

            let base = line * size0 * dim;
            for i in 0..size0 {
                let alpha = i as f64 / size0 as f64;
                let one_minus_alpha = 1.0 - alpha;
                let pixel = &mut components[base + i * dim..base + (i + 1) * dim];
                for (k, c) in pixel.iter_mut().enumerate() {
                    *c = one_minus_alpha * start[k] + alpha * end[k];
                }
            }
            increment(&mut higher, &size[1..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::displacement::DisplacementFieldTransform;
    use crate::transform::transform::{
        AffineTransform, Euler2DTransform, ParametricTransform, TranslationTransform,
    };

    fn field_2d(f: &TransformToDisplacementFieldFilter, t: &impl TransformBase) -> Vec<f64> {
        f.execute(t).unwrap().components_to_f64_vec()
    }

    /// `T(x) = x + t` gives `d(x) = t` everywhere, whatever the grid.
    #[test]
    fn translation_transform_yields_a_constant_field() {
        let t = TranslationTransform::new(vec![2.0, -3.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![3, 2]);
        let img = f.execute(&t).unwrap();

        assert_eq!(img.pixel_id(), PixelId::VectorFloat64);
        assert_eq!(img.number_of_components_per_pixel(), 2);
        assert_eq!(img.size(), &[3, 2]);
        assert_eq!(
            img.components_to_f64_vec(),
            vec![
                2.0, -3.0, 2.0, -3.0, 2.0, -3.0, 2.0, -3.0, 2.0, -3.0, 2.0, -3.0
            ]
        );
    }

    /// The identity transform displaces nothing.
    #[test]
    fn identity_transform_yields_a_zero_field() {
        let t = AffineTransform::identity(2);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![2, 2]);
        assert_eq!(field_2d(&f, &t), vec![0.0; 8]);
    }

    /// SimpleITK's `Size` default is 64 per axis, taken here per the
    /// transform's dimension rather than always three-long.
    #[test]
    fn default_size_is_sixty_four_per_axis() {
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let img = TransformToDisplacementFieldFilter::new()
            .execute(&t)
            .unwrap();
        assert_eq!(img.size(), &[64, 64]);
        assert_eq!(img.spacing(), &[1.0, 1.0]);
        assert_eq!(img.origin(), &[0.0, 0.0]);
        assert_eq!(img.direction(), &[1.0, 0.0, 0.0, 1.0]);
    }

    /// A scale about the origin: `T(x) = 2x`, so `d(x) = x`. With unit spacing
    /// and zero origin the physical point equals the index, so pixel `(i, j)`
    /// holds `(i, j)`. This exercises the linear scan-line path with a
    /// non-constant displacement.
    #[test]
    fn linear_transform_displacement_varies_along_the_scan_line() {
        let t = AffineTransform::new(2, vec![2.0, 0.0, 0.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![3, 2]);
        assert_eq!(
            field_2d(&f, &t),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, // j = 0
                0.0, 1.0, 1.0, 1.0, 2.0, 1.0, // j = 1
            ]
        );
    }

    /// Spacing and origin enter through the physical point: pixel `(i, j)` sits
    /// at `p = origin + spacing ⊙ (i, j)`, and `d(p) = p` for `T(x) = 2x`.
    #[test]
    fn grid_geometry_enters_through_the_physical_point() {
        let t = AffineTransform::new(2, vec![2.0, 0.0, 0.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![2, 2])
            .set_output_spacing(vec![0.5, 2.0])
            .set_output_origin(vec![10.0, -1.0]);
        let img = f.execute(&t).unwrap();
        assert_eq!(img.spacing(), &[0.5, 2.0]);
        assert_eq!(img.origin(), &[10.0, -1.0]);
        assert_eq!(
            img.components_to_f64_vec(),
            vec![
                10.0, -1.0, 10.5, -1.0, // j = 0: x = 10, 10.5 ; y = -1
                10.0, 1.0, 10.5, 1.0, // j = 1: y = -1 + 2 = 1
            ]
        );
    }

    /// A non-identity direction matrix rotates the index axes into physical
    /// space before the transform sees them. `D = [[0,-1],[1,0]]` maps index
    /// `(i, j)` to `p = (−j, i)`; with `T(x) = x + (1, 0)` the field is the
    /// constant `(1, 0)` — the displacement is a *physical* vector and is not
    /// rotated back into index space.
    #[test]
    fn non_identity_direction_places_the_grid_but_not_the_displacement() {
        let t = TranslationTransform::new(vec![1.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![2, 2])
            .set_output_direction(vec![0.0, -1.0, 1.0, 0.0]);
        let img = f.execute(&t).unwrap();
        assert_eq!(img.direction(), &[0.0, -1.0, 1.0, 0.0]);
        assert_eq!(
            img.components_to_f64_vec(),
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }

    /// The same rotated grid with a position-dependent transform, so the
    /// direction matrix actually shows up in the values. `D = [[0,-1],[1,0]]`,
    /// `T(x) = 2x` so `d(p) = p = (−j, i)`.
    #[test]
    fn non_identity_direction_with_a_position_dependent_transform() {
        let t = AffineTransform::new(2, vec![2.0, 0.0, 0.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![2, 2])
            .set_output_direction(vec![0.0, -1.0, 1.0, 0.0]);
        let got = field_2d(&f, &t);
        // (i, j) -> p = (-j, i)
        let want = vec![
            0.0, 0.0, // (0,0)
            0.0, 1.0, // (1,0)
            -1.0, 0.0, // (0,1)
            -1.0, 1.0, // (1,1)
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}");
        }
    }

    /// A 90-degree rotation about the origin is linear, so it takes the
    /// scan-line path; `T(x, y) = (−y, x)` gives `d = (−y − x, x − y)`.
    #[test]
    fn rotation_is_linear_and_matches_the_closed_form() {
        let t = Euler2DTransform::new(std::f64::consts::FRAC_PI_2, [0.0, 0.0], [0.0, 0.0]);
        assert!(t.is_linear());
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![3, 3]);
        let got = field_2d(&f, &t);
        for j in 0..3usize {
            for i in 0..3usize {
                let (x, y) = (i as f64, j as f64);
                let base = (j * 3 + i) * 2;
                assert!((got[base] - (-y - x)).abs() < 1e-12, "x at ({i},{j})");
                assert!((got[base + 1] - (x - y)).abs() < 1e-12, "y at ({i},{j})");
            }
        }
    }

    /// A displacement-field transform is not linear, so it takes the per-pixel
    /// path; sampling it back onto its own grid recovers its own field.
    #[test]
    fn nonlinear_transform_takes_the_per_pixel_path() {
        let mut t = DisplacementFieldTransform::new(
            2,
            &[2, 2],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let field = vec![1.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, -2.0];
        t.set_parameters(&field).unwrap();
        assert!(!t.is_linear());

        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![2, 2]);
        let got = field_2d(&f, &t);
        for (g, w) in got.iter().zip(&field) {
            assert!((g - w).abs() < 1e-12, "{got:?} vs {field:?}");
        }
    }

    /// `sitkVectorFloat32` narrows every component with a `static_cast`.
    #[test]
    fn float32_output_pixel_type() {
        let t = TranslationTransform::new(vec![0.1, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![1, 1])
            .set_output_pixel_type(PixelId::VectorFloat32);
        let img = f.execute(&t).unwrap();
        assert_eq!(img.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(img.component_slice::<f32>().unwrap(), &[0.1f64 as f32, 0.0]);
    }

    #[test]
    fn a_non_float_vector_output_pixel_type_is_rejected() {
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_output_pixel_type(PixelId::VectorUInt8);
        assert_eq!(
            f.execute(&t),
            Err(TransformError::UnsupportedDisplacementFieldPixelType(
                PixelId::VectorUInt8
            ))
        );
    }

    /// A scalar output pixel type cannot hold a displacement vector either.
    #[test]
    fn a_scalar_output_pixel_type_is_rejected() {
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_output_pixel_type(PixelId::Float64);
        assert_eq!(
            f.execute(&t),
            Err(TransformError::UnsupportedDisplacementFieldPixelType(
                PixelId::Float64
            ))
        );
    }

    #[test]
    fn grid_parameters_of_the_wrong_length_are_rejected() {
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![2, 2, 2]);
        assert_eq!(f.execute(&t), Err(TransformError::DimensionMismatch));
    }

    /// `set_reference_image` copies all four grid parameters.
    #[test]
    fn reference_image_supplies_the_whole_grid() {
        let mut reference = Image::new(&[3, 2], PixelId::UInt8);
        reference.set_spacing(&[0.5, 4.0]).unwrap();
        reference.set_origin(&[-2.0, 7.0]).unwrap();
        reference.set_direction(&[0.0, 1.0, 1.0, 0.0]).unwrap();

        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_reference_image(&reference);
        let img = f.execute(&t).unwrap();

        assert_eq!(img.size(), reference.size());
        assert_eq!(img.spacing(), reference.spacing());
        assert_eq!(img.origin(), reference.origin());
        assert_eq!(img.direction(), reference.direction());
    }

    /// One-dimensional grids have a single scan line and no higher indices.
    #[test]
    fn one_dimensional_grid() {
        let t = TranslationTransform::new(vec![5.0]);
        let mut f = TransformToDisplacementFieldFilter::new();
        f.set_size(vec![3]);
        let img = f.execute(&t).unwrap();
        assert_eq!(img.number_of_components_per_pixel(), 1);
        assert_eq!(img.components_to_f64_vec(), vec![5.0, 5.0, 5.0]);
    }
}
