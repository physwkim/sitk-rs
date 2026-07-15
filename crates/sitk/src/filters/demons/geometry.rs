//! `itk::ImageBase`'s index/point transforms, shared by the scalar images the
//! demons functions read ([`super::image_function::RealImage`]) and by the
//! vector fields the diffeomorphic filter composes
//! ([`super::compose`]).

use crate::core::{Error, Image, coord, matrix};

use crate::filters::Result;

/// An image's grid: size, spacing, origin, direction cosines, and the two
/// precomputed `itk::ImageBase` matrices, hoisted out of the per-pixel loops.
///
/// The point transforms route through the shared [`crate::core::coord`] primitive
/// — the single implementation of `itk::ImageBase`'s index↔physical maps — so
/// they cannot re-diverge from `Image`'s own methods. `Image`'s versions build
/// the matrices per call and return a `Result`, neither of which belongs inside
/// `ComputeUpdate`; the cached matrices here are the same ones it would build.
pub(crate) struct Geometry {
    pub(crate) size: Vec<usize>,
    pub(crate) spacing: Vec<f64>,
    pub(crate) origin: Vec<f64>,
    /// Row-major `dim x dim`, as `Image::direction`.
    pub(crate) direction: Vec<f64>,
    /// Row-major inverse of `direction` alone — for the vector transforms, which
    /// ITK maps by `m_Direction`/`m_InverseDirection` with no spacing or origin.
    pub(crate) inverse_direction: Vec<f64>,
    /// ITK `m_IndexToPhysicalPoint = Direction · diag(spacing)`.
    index_to_physical: Vec<f64>,
    /// ITK `m_PhysicalPointToIndex = inverse(Direction · diag(spacing))`.
    physical_to_index: Vec<f64>,
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
        let index_to_physical =
            coord::index_to_physical_matrix(image.direction(), image.spacing(), dim);
        let physical_to_index =
            coord::physical_to_index_matrix(image.direction(), image.spacing(), dim)
                .ok_or(Error::SingularDirection)?;
        Ok(Geometry {
            size: image.size().to_vec(),
            spacing: image.spacing().to_vec(),
            origin: image.origin().to_vec(),
            direction: image.direction().to_vec(),
            inverse_direction,
            index_to_physical,
            physical_to_index,
        })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.size.len()
    }

    /// `ImageBase::TransformIndexToPhysicalPoint` (integer method, origin-first),
    /// via the shared [`crate::core::coord`] primitive.
    pub(crate) fn index_to_physical_point(&self, index: &[usize]) -> Vec<f64> {
        let dim = self.dimension();
        let widened: Vec<i64> = index.iter().map(|&i| i as i64).collect();
        coord::index_to_physical_point(&self.index_to_physical, &self.origin, &widened, dim)
    }

    /// `ImageBase::TransformPhysicalPointToContinuousIndex`, via the shared
    /// [`crate::core::coord`] primitive — `PhysicalPointToIndex · (p − origin)`
    /// with the composed inverse, so a diagonal geometry reciprocal-multiplies
    /// exactly as ITK does rather than dividing by spacing.
    pub(crate) fn physical_point_to_continuous_index(&self, point: &[f64]) -> Vec<f64> {
        coord::physical_point_to_continuous_index(
            &self.physical_to_index,
            &self.origin,
            point,
            self.dimension(),
        )
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
