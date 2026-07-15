//! ITK's `ITKDisplacementField` module: filters whose input is a *displacement
//! field*, a vector image whose pixel is an `itk::Vector<T, ImageDimension>`.
//!
//! # The pixel type these filters accept
//!
//! Every `DisplacementField` yaml in SimpleITK declares `pixel_types:
//! RealVectorPixelIDTypeList` — `typelist<VectorPixelID<float>,
//! VectorPixelID<double>>` (`sitkPixelIDTypeLists.h:143`) — and then passes the
//! input through `GetImageFromVectorImage`, which reinterprets the
//! `itk::VectorImage<T, N>` buffer as an `itk::Image<itk::Vector<T, N>, N>`
//! after checking
//!
//! ```text
//! if (img->GetNumberOfComponentsPerPixel() != VectorImageType::ImageDimension)
//!   sitkExceptionMacro("Expected number of elements in vector image to be the same as the dimension!");
//! ```
//!
//! (`sitkImageConvert.hxx:38-42`). So a displacement field is exactly: a vector
//! image, floating-point components, and **one component per image dimension**.
//! [`require_displacement_field`] is that check, and every filter in this module
//! goes through it.
//!
//! # Compute precision
//!
//! ITK templates these filters on the field's own component type: the
//! interpolator's `TCoordinate` is `Vector::ComponentType`, so a `VectorFloat32`
//! field has its continuous indices and interpolation weights computed in
//! `float`. This port computes in `f64` throughout and narrows only when storing
//! into the output buffer — the same deliberate divergence
//! [`crate::filters::n4_bias_field`] documents. For a `VectorFloat64` field the two agree
//! exactly; for `VectorFloat32` this port is the more accurate of the two, and
//! differs from ITK at the `f32` round-off level.
//!
//! # Interpolation
//!
//! [`Field::evaluate_at_point`] is `VectorLinearInterpolateImageFunction`
//! (`itkVectorLinearInterpolateImageFunction.hxx:33-107`) behind
//! `ImageFunction::IsInsideBuffer` (`itkImageFunction.h:158-184`). Two details
//! are load-bearing and are reproduced literally:
//!
//! - the "inside" test is on the *continuous index*, and admits the half-pixel
//!   skirt `[-0.5, size - 0.5)` (`itkImageFunction.hxx:60-65` sets
//!   `m_StartContinuousIndex = start - 0.5`, `m_EndContinuousIndex = end + 0.5`;
//!   the test is `>= start && < end`, so the upper bound is exclusive and the
//!   lower inclusive);
//! - inside that skirt, the neighbour indices are **clamped** into
//!   `[0, size-1]`, so the interpolant is constant across the outer half pixel
//!   rather than extrapolated.

use crate::core::{Image, PixelId, Scalar, dispatch_scalar, matrix};

use crate::filters::{FilterError, Result};

mod inverse;
mod invert;
mod iterative_inverse;
mod jacobian_determinant;

pub use inverse::inverse_displacement_field;
pub use invert::{
    InvertDisplacementFieldResult, InvertDisplacementFieldSettings, invert_displacement_field,
};
pub use iterative_inverse::{
    IterativeInverseDisplacementFieldSettings, iterative_inverse_displacement_field,
};
pub use jacobian_determinant::{
    DisplacementFieldJacobianDeterminantSettings, displacement_field_jacobian_determinant,
};

/// Check that `img` is what SimpleITK's `GetImageFromVectorImage` accepts as a
/// displacement field, and return its dimension.
///
/// Errors:
///
/// - [`crate::core::Error::RequiresVectorPixelType`] on a scalar image
///   (`pixel_types: RealVectorPixelIDTypeList` admits only vector pixel ids);
/// - [`FilterError::RequiresRealPixelType`] on integer components (same list);
/// - [`FilterError::DisplacementFieldComponentMismatch`] when the component
///   count differs from the image dimension (`sitkImageConvert.hxx:38-42`).
pub(crate) fn require_displacement_field(img: &Image) -> Result<usize> {
    let id = img.pixel_id();
    if !id.is_vector() {
        return Err(crate::core::Error::RequiresVectorPixelType(id).into());
    }
    if !id.is_floating_point() {
        return Err(FilterError::RequiresRealPixelType(id));
    }
    let dimension = img.dimension();
    let components = img.number_of_components_per_pixel();
    if components != dimension {
        return Err(FilterError::DisplacementFieldComponentMismatch {
            components,
            dimension,
        });
    }
    Ok(dimension)
}

/// A displacement field widened to `f64`, together with the geometry needed to
/// map between its lattice and physical space.
///
/// `data` is the interleaved component buffer read through
/// [`Image::components_to_f64_vec`] — `dim` components per pixel, pixels in
/// first-index-fastest order.
pub(crate) struct Field {
    pub(crate) dim: usize,
    pub(crate) size: Vec<usize>,
    pub(crate) spacing: Vec<f64>,
    pub(crate) origin: Vec<f64>,
    pub(crate) direction: Vec<f64>,
    inverse_direction: Vec<f64>,
    pub(crate) data: Vec<f64>,
}

impl Field {
    /// Read `img` as a displacement field, after [`require_displacement_field`].
    pub(crate) fn from_image(img: &Image) -> Result<Self> {
        let dim = require_displacement_field(img)?;
        let inverse_direction =
            matrix::invert(img.direction(), dim).ok_or(crate::core::Error::SingularDirection)?;
        Ok(Field {
            dim,
            size: img.size().to_vec(),
            spacing: img.spacing().to_vec(),
            origin: img.origin().to_vec(),
            direction: img.direction().to_vec(),
            inverse_direction,
            data: img.components_to_f64_vec(),
        })
    }

    /// A zero field on `other`'s lattice and geometry.
    pub(crate) fn zeros_like(other: &Field) -> Self {
        Field {
            dim: other.dim,
            size: other.size.clone(),
            spacing: other.spacing.clone(),
            origin: other.origin.clone(),
            direction: other.direction.clone(),
            inverse_direction: other.inverse_direction.clone(),
            data: vec![0.0; other.data.len()],
        }
    }

    pub(crate) fn number_of_pixels(&self) -> usize {
        self.size.iter().product()
    }

    /// The `dim` components of pixel `pixel`, whose linear index is
    /// first-index-fastest.
    pub(crate) fn vector(&self, pixel: usize) -> &[f64] {
        &self.data[pixel * self.dim..(pixel + 1) * self.dim]
    }

    /// The multi-index of linear pixel index `pixel`.
    pub(crate) fn multi_index(&self, pixel: usize) -> Vec<usize> {
        let mut rest = pixel;
        let mut index = Vec::with_capacity(self.dim);
        for &s in &self.size {
            index.push(rest % s);
            rest /= s;
        }
        index
    }

    /// `p = origin + Direction * (spacing ⊙ index)`, ITK's
    /// `TransformIndexToPhysicalPoint`.
    pub(crate) fn index_to_point(&self, index: &[usize]) -> Vec<f64> {
        let scaled: Vec<f64> = (0..self.dim)
            .map(|d| index[d] as f64 * self.spacing[d])
            .collect();
        let rotated = matrix::mat_vec(&self.direction, &scaled, self.dim);
        (0..self.dim).map(|d| self.origin[d] + rotated[d]).collect()
    }

    /// ITK's `TransformPhysicalPointToContinuousIndex`.
    pub(crate) fn point_to_continuous_index(&self, point: &[f64]) -> Vec<f64> {
        let diff: Vec<f64> = (0..self.dim).map(|d| point[d] - self.origin[d]).collect();
        let unrotated = matrix::mat_vec(&self.inverse_direction, &diff, self.dim);
        (0..self.dim)
            .map(|d| unrotated[d] / self.spacing[d])
            .collect()
    }

    /// `ImageFunction::IsInsideBuffer(ContinuousIndex)`
    /// (`itkImageFunction.h:158-170`): `index[j] >= -0.5 && index[j] < size[j] -
    /// 0.5` for every `j`. Written as the negation of a conjunction so a `NaN`
    /// coordinate reports "outside", exactly as the upstream comment
    /// ("Test for negative of a positive so we can catch NaN's") intends.
    pub(crate) fn is_inside_buffer(&self, cindex: &[f64]) -> bool {
        (0..self.dim).all(|d| cindex[d] >= -0.5 && cindex[d] < self.size[d] as f64 - 0.5)
    }

    /// `VectorLinearInterpolateImageFunction::EvaluateAtContinuousIndex`
    /// (`itkVectorLinearInterpolateImageFunction.hxx:33-107`), including the
    /// neighbour clamp into `[0, size-1]`, the `if (overlap)` skip of
    /// zero-weight corners, and the `totalOverlap == 1.0` early exit.
    ///
    /// The clamp makes this total: a `cindex` far outside the lattice reads the
    /// nearest corner pixel rather than out of bounds, as upstream's
    /// `std::min`/`std::max` on the neighbour index also does. Callers that need
    /// the upstream "outside means zero" behaviour gate on
    /// [`Field::is_inside_buffer`] first — [`Field::evaluate_at_point`] does.
    pub(crate) fn evaluate_at_continuous_index(&self, cindex: &[f64]) -> Vec<f64> {
        let mut base = vec![0i64; self.dim];
        let mut distance = vec![0.0f64; self.dim];
        for d in 0..self.dim {
            base[d] = cindex[d].floor() as i64;
            distance[d] = cindex[d] - base[d] as f64;
        }

        let mut output = vec![0.0f64; self.dim];
        let mut total_overlap = 0.0f64;
        let neighbors = 1usize << self.dim;

        for counter in 0..neighbors {
            let mut overlap = 1.0f64;
            let mut upper = counter;
            let mut neighbor = vec![0usize; self.dim];
            for d in 0..self.dim {
                let end = self.size[d] as i64 - 1;
                if upper & 1 == 1 {
                    neighbor[d] = (base[d] + 1).clamp(0, end) as usize;
                    overlap *= distance[d];
                } else {
                    neighbor[d] = base[d].clamp(0, end) as usize;
                    overlap *= 1.0 - distance[d];
                }
                upper >>= 1;
            }

            if overlap != 0.0 {
                let pixel = self.linear_index(&neighbor);
                let input = self.vector(pixel);
                for k in 0..self.dim {
                    output[k] += overlap * input[k];
                }
                total_overlap += overlap;
            }

            if total_overlap == 1.0 {
                break;
            }
        }

        output
    }

    /// `VectorInterpolateImageFunction::Evaluate` behind `IsInsideBuffer`:
    /// `None` when `point` falls outside the half-pixel skirt.
    pub(crate) fn evaluate_at_point(&self, point: &[f64]) -> Option<Vec<f64>> {
        let cindex = self.point_to_continuous_index(point);
        self.is_inside_buffer(&cindex)
            .then(|| self.evaluate_at_continuous_index(&cindex))
    }

    pub(crate) fn linear_index(&self, index: &[usize]) -> usize {
        let mut offset = 0usize;
        let mut stride = 1usize;
        for (&i, &s) in index.iter().zip(&self.size) {
            offset += i * stride;
            stride *= s;
        }
        offset
    }
}

/// Build a `dim`-component vector image of `component_id`'s type from an
/// interleaved `f64` buffer, giving it `size` and the geometry
/// `(spacing, origin, direction)`.
///
/// `component_id` must be a scalar pixel id; the output's pixel id is its vector
/// variant, so a `Float64` component type yields a `VectorFloat64` image.
pub(crate) fn field_to_image(
    size: &[usize],
    data: Vec<f64>,
    component_id: PixelId,
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
) -> Result<Image> {
    fn build<T: Scalar>(size: &[usize], data: &[f64]) -> Result<Image> {
        let narrowed: Vec<T> = data.iter().map(|&x| T::from_f64(x)).collect();
        Ok(Image::from_vec_vector(size, size.len(), narrowed)?)
    }

    let mut img = dispatch_scalar!(component_id, build, size, &data)?;
    img.set_spacing(spacing)?;
    img.set_origin(origin)?;
    img.set_direction(direction)?;
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-D `VectorFloat64` field with one component per pixel: the smallest
    /// legal displacement field.
    fn field_1d(values: &[f64]) -> Image {
        Image::from_vec_vector(&[values.len()], 1, values.to_vec()).unwrap()
    }

    #[test]
    fn require_displacement_field_rejects_a_scalar_image() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            require_displacement_field(&img).unwrap_err(),
            FilterError::Core(crate::core::Error::RequiresVectorPixelType(
                PixelId::Float64
            ))
        ));
    }

    #[test]
    fn require_displacement_field_rejects_integer_components() {
        let img = Image::from_vec_vector(&[2, 2], 2, vec![0u8; 8]).unwrap();
        assert!(matches!(
            require_displacement_field(&img).unwrap_err(),
            FilterError::RequiresRealPixelType(PixelId::VectorUInt8)
        ));
    }

    /// `GetImageFromVectorImage` throws unless components == dimension.
    #[test]
    fn require_displacement_field_rejects_a_component_count_that_is_not_the_dimension() {
        let img = Image::from_vec_vector(&[2, 2], 3, vec![0.0f64; 12]).unwrap();
        assert!(matches!(
            require_displacement_field(&img).unwrap_err(),
            FilterError::DisplacementFieldComponentMismatch {
                components: 3,
                dimension: 2
            }
        ));
    }

    #[test]
    fn require_displacement_field_accepts_a_two_component_two_dimensional_field() {
        let img = Image::from_vec_vector(&[2, 2], 2, vec![0.0f64; 8]).unwrap();
        assert_eq!(require_displacement_field(&img).unwrap(), 2);
    }

    /// The half-pixel skirt: `[-0.5, size - 0.5)`, lower bound inclusive and
    /// upper bound exclusive.
    #[test]
    fn is_inside_buffer_admits_the_half_pixel_skirt() {
        let f = Field::from_image(&field_1d(&[1.0, 2.0, 3.0])).unwrap();
        assert!(f.is_inside_buffer(&[-0.5]));
        assert!(!f.is_inside_buffer(&[-0.5 - f64::EPSILON]));
        assert!(f.is_inside_buffer(&[2.4999]));
        assert!(!f.is_inside_buffer(&[2.5]));
        assert!(!f.is_inside_buffer(&[f64::NAN]));
    }

    /// Inside the skirt the neighbours clamp, so the interpolant is flat across
    /// the outer half pixel rather than extrapolating.
    #[test]
    fn evaluate_clamps_neighbors_inside_the_skirt() {
        let f = Field::from_image(&field_1d(&[1.0, 5.0, 9.0])).unwrap();
        assert_eq!(f.evaluate_at_continuous_index(&[-0.25]), vec![1.0]);
        assert_eq!(f.evaluate_at_continuous_index(&[-0.5]), vec![1.0]);
        assert_eq!(f.evaluate_at_continuous_index(&[2.25]), vec![9.0]);
    }

    #[test]
    fn evaluate_interpolates_linearly_between_lattice_points() {
        let f = Field::from_image(&field_1d(&[1.0, 5.0, 9.0])).unwrap();
        assert_eq!(f.evaluate_at_continuous_index(&[0.0]), vec![1.0]);
        assert_eq!(f.evaluate_at_continuous_index(&[0.25]), vec![2.0]);
        assert_eq!(f.evaluate_at_continuous_index(&[1.5]), vec![7.0]);
        assert_eq!(f.evaluate_at_continuous_index(&[2.0]), vec![9.0]);
    }

    /// Bilinear on a 2-D field, one component per axis: pixel `(x, y)` holds
    /// `(x, 10y)`, so the interpolant at `(0.5, 0.5)` is `(0.5, 5)`.
    #[test]
    fn evaluate_interpolates_bilinearly() {
        let mut data = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                data.push(x as f64);
                data.push(10.0 * y as f64);
            }
        }
        let f = Field::from_image(&Image::from_vec_vector(&[2, 2], 2, data).unwrap()).unwrap();
        assert_eq!(f.evaluate_at_continuous_index(&[0.5, 0.5]), vec![0.5, 5.0]);
        assert_eq!(f.evaluate_at_continuous_index(&[1.0, 0.0]), vec![1.0, 0.0]);
    }

    #[test]
    fn evaluate_at_point_maps_through_spacing_and_origin() {
        let mut img = field_1d(&[1.0, 5.0, 9.0]);
        img.set_spacing(&[2.0]).unwrap();
        img.set_origin(&[-1.0]).unwrap();
        let f = Field::from_image(&img).unwrap();

        // Physical 1.0 is continuous index 1.0.
        assert_eq!(f.evaluate_at_point(&[1.0]), Some(vec![5.0]));
        // Physical -2.0 is continuous index -0.5: the inclusive lower bound.
        assert_eq!(f.evaluate_at_point(&[-2.0]), Some(vec![1.0]));
        // Physical -2.1 is continuous index -0.55: outside.
        assert_eq!(f.evaluate_at_point(&[-2.1]), None);
    }

    #[test]
    fn multi_index_is_first_index_fastest() {
        let f = Field::from_image(&Image::from_vec_vector(&[2, 3], 2, vec![0.0f64; 12]).unwrap())
            .unwrap();
        assert_eq!(f.multi_index(0), vec![0, 0]);
        assert_eq!(f.multi_index(1), vec![1, 0]);
        assert_eq!(f.multi_index(2), vec![0, 1]);
        assert_eq!(f.multi_index(5), vec![1, 2]);
    }

    #[test]
    fn field_to_image_narrows_to_the_component_type_and_keeps_geometry() {
        let img = field_to_image(
            &[2, 2],
            vec![1.5; 8],
            PixelId::Float32,
            &[0.5, 0.25],
            &[1.0, 2.0],
            &[1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(img.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(img.number_of_components_per_pixel(), 2);
        assert_eq!(img.spacing(), &[0.5, 0.25]);
        assert_eq!(img.origin(), &[1.0, 2.0]);
        assert_eq!(img.component_slice::<f32>().unwrap(), &[1.5f32; 8]);
    }
}
