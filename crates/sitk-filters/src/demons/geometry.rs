//! `itk::ImageBase`'s index/point transforms, shared by the scalar images the
//! demons functions read ([`super::image_function::RealImage`]) and by the
//! vector fields the diffeomorphic filter composes
//! ([`super::compose`]).

use sitk_core::{Error, Image, matrix};

use crate::Result;

/// An image's grid: size, spacing, origin, direction cosines, and the
/// precomputed inverse of the direction matrix.
///
/// The inverse is hoisted out of the per-pixel loops; `Image`'s own
/// `physical_point_to_continuous_index` inverts on every call and returns a
/// `Result`, neither of which belongs inside `ComputeUpdate`.
pub(crate) struct Geometry {
    pub(crate) size: Vec<usize>,
    pub(crate) spacing: Vec<f64>,
    pub(crate) origin: Vec<f64>,
    /// Row-major `dim x dim`, as `Image::direction`.
    pub(crate) direction: Vec<f64>,
    /// Row-major inverse of `direction`.
    pub(crate) inverse_direction: Vec<f64>,
}

/// Row `i` of a row-major `dim x dim` matrix.
pub(crate) fn row(matrix: &[f64], i: usize, dim: usize) -> &[f64] {
    &matrix[i * dim..i * dim + dim]
}

impl Geometry {
    /// Errors with `SingularDirection` when the direction cosine matrix cannot
    /// be inverted — the same condition `TransformPhysicalPointToContinuousIndex`
    /// relies on.
    pub(crate) fn new(image: &Image) -> Result<Self> {
        let dim = image.dimension();
        let inverse_direction =
            matrix::invert(image.direction(), dim).ok_or(Error::SingularDirection)?;
        Ok(Geometry {
            size: image.size().to_vec(),
            spacing: image.spacing().to_vec(),
            origin: image.origin().to_vec(),
            direction: image.direction().to_vec(),
            inverse_direction,
        })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.size.len()
    }

    /// `ImageBase::TransformIndexToPhysicalPoint`:
    /// `p = origin + Direction * (spacing ⊙ index)`.
    pub(crate) fn index_to_physical_point(&self, index: &[usize]) -> Vec<f64> {
        let dim = self.dimension();
        self.origin
            .iter()
            .enumerate()
            .map(|(i, &origin)| {
                let rotated: f64 = row(&self.direction, i, dim)
                    .iter()
                    .zip(index)
                    .zip(&self.spacing)
                    .map(|((&d, &idx), &spacing)| d * idx as f64 * spacing)
                    .sum();
                origin + rotated
            })
            .collect()
    }

    /// `ImageBase::TransformPhysicalPointToContinuousIndex`.
    pub(crate) fn physical_point_to_continuous_index(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        let diff: Vec<f64> = point
            .iter()
            .zip(&self.origin)
            .map(|(&p, &o)| p - o)
            .collect();
        self.spacing
            .iter()
            .enumerate()
            .map(|(i, &spacing)| {
                let unrotated: f64 = row(&self.inverse_direction, i, dim)
                    .iter()
                    .zip(&diff)
                    .map(|(&m, &d)| m * d)
                    .sum();
                unrotated / spacing
            })
            .collect()
    }

    /// `ImageFunction::IsInsideBuffer(PointType)` (itkImageFunction.h:176-184),
    /// which forwards to the continuous-index overload at line 159-170.
    ///
    /// The buffer's continuous bounds are `[start - 0.5, end + 0.5)` per
    /// `ImageFunction::SetInputImage` (itkImageFunction.hxx:63-64), with `end =
    /// size - 1`. Note the asymmetry: the lower bound is inclusive and the
    /// upper is *exclusive*, so a point exactly half a pixel past the last
    /// pixel centre is outside while its mirror at the first is inside. The
    /// comparison is written as the negation of a positive test so that a
    /// `NaN` coordinate reports outside.
    pub(crate) fn is_inside_buffer(&self, point: &[f64]) -> bool {
        let cindex = self.physical_point_to_continuous_index(point);
        (0..self.dimension()).all(|d| {
            let end = self.size[d] as f64 - 1.0;
            cindex[d] >= -0.5 && cindex[d] < end + 0.5
        })
    }

    /// `ImageBase::TransformLocalVectorToPhysicalVector`
    /// (itkImageBase.h:636-654): `out = Direction * in`.
    pub(crate) fn local_vector_to_physical_vector(&self, local: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        (0..dim)
            .map(|i| {
                row(&self.direction, i, dim)
                    .iter()
                    .zip(local)
                    .map(|(&d, &l)| d * l)
                    .sum()
            })
            .collect()
    }

    /// `ImageBase::TransformPhysicalVectorToLocalVector`
    /// (itkImageBase.h:685-702): `out = Direction⁻¹ * in`.
    pub(crate) fn physical_vector_to_local_vector(&self, physical: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        (0..dim)
            .map(|i| {
                row(&self.inverse_direction, i, dim)
                    .iter()
                    .zip(physical)
                    .map(|(&m, &p)| m * p)
                    .sum()
            })
            .collect()
    }
}
