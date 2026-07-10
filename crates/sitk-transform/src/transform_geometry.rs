//! `itk::TransformGeometryImageFilter`
//! (`Modules/Core/Transform/include/itkTransformGeometryImageFilter.h(.hxx)`):
//! re-point an image's origin/spacing/direction so that its physical space is
//! related to the original by `transform`, without resampling or touching a
//! single pixel value (`GenerateData` is a straight buffer copy once
//! `GenerateOutputInformation` has already rewritten the geometry; there is no
//! pixel-value computation to port).
//!
//! `VerifyPreconditions` requires `transform->IsLinear()`
//! ([`TransformBase::is_linear`]); this port checks the same thing and returns
//! [`TransformError::NonLinearTransform`] rather than throwing.
//!
//! ## Deriving the new geometry
//!
//! `GenerateOutputInformation` calls `transform->ApplyToImageMetadata(output)`
//! (`itkTransform.hxx`'s `ApplyToImageMetadata`, the generic implementation
//! every linear transform inherits):
//!
//! ```text
//! new_origin = Tinv(origin)
//! for each axis i:
//!     v = spacing[i] * direction[:, i]      // physical extent of one voxel step
//!     v' = Tinv_linear(v)                   // linear part of Tinv, no offset
//!     new_spacing[i] = |v'|
//!     new_direction[:, i] = v' / |v'|
//! ```
//!
//! where `Tinv` is `transform`'s inverse. Rather than requiring every
//! [`TransformBase`] implementor to expose an `inverse()` method, this port
//! derives `Tinv` from the two trait methods already guaranteed exact for a
//! linear transform: `T(x) = M·x + b` with `M = jacobian_wrt_position(_)`
//! (constant, independent of the point, precisely because `is_linear()` is
//! `true`) and `b = transform_point(0)` (since `M·0 = 0`). `Tinv(y) =
//! M⁻¹·(y − b)`, and `Tinv`'s linear part (what `TransformVector` applies,
//! ignoring the offset) is just `M⁻¹`. `M` failing to invert
//! (`sitk_core::matrix::invert` returning `None`) surfaces as
//! [`TransformError::NonInvertibleTransform`] -- see that variant's docs for
//! why this port errors instead of reproducing ITK's null-pointer-deref
//! crash on the same input.

use sitk_core::{Image, matrix};

use crate::error::{Result, TransformError};
use crate::transform::TransformBase;

/// `TransformGeometryImageFilter`: rewrite `image`'s origin, spacing, and
/// direction so that its physical space relates to the original by
/// `transform`, leaving every pixel value unchanged. See the module docs for
/// the exact derivation and the linearity precondition.
pub fn transform_geometry<T: TransformBase>(image: &Image, transform: &T) -> Result<Image> {
    let dim = image.dimension();
    if transform.dimension() != dim {
        return Err(TransformError::DimensionMismatch);
    }
    if !transform.is_linear() {
        return Err(TransformError::NonLinearTransform);
    }

    let zero = vec![0.0; dim];
    let offset = transform.transform_point(&zero);
    let m = transform.jacobian_wrt_position(&zero);
    let m_inv = matrix::invert(&m, dim).ok_or(TransformError::NonInvertibleTransform)?;

    let origin = image.origin();
    let spacing = image.spacing();
    let direction = image.direction();

    let diff: Vec<f64> = (0..dim).map(|d| origin[d] - offset[d]).collect();
    let new_origin = matrix::mat_vec(&m_inv, &diff, dim);

    let mut new_spacing = vec![0.0; dim];
    let mut new_direction = vec![0.0; dim * dim];
    for i in 0..dim {
        let axis_vector: Vec<f64> = (0..dim)
            .map(|k| direction[k * dim + i] * spacing[i])
            .collect();
        let transformed = matrix::mat_vec(&m_inv, &axis_vector, dim);
        let norm = transformed.iter().map(|v| v * v).sum::<f64>().sqrt();
        new_spacing[i] = norm;
        for k in 0..dim {
            new_direction[k * dim + i] = transformed[k] / norm;
        }
    }

    let mut out = image.clone();
    out.set_origin(&new_origin).map_err(TransformError::Core)?;
    out.set_spacing(&new_spacing)
        .map_err(TransformError::Core)?;
    out.set_direction(&new_direction)
        .map_err(TransformError::Core)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::displacement::DisplacementFieldTransform;
    use crate::transform::{AffineTransform, Euler2DTransform, TranslationTransform};
    use sitk_core::PixelId;

    fn img_2d() -> Image {
        let mut img = Image::new(&[4, 4], PixelId::Float64);
        img.set_spacing(&[1.0, 1.0]).unwrap();
        img.set_origin(&[0.0, 0.0]).unwrap();
        img
    }

    /// A pure translation moves the origin by the inverse translation
    /// (`Tinv(y) = y - t`), and leaves spacing/direction untouched (the
    /// linear part of a translation is the identity).
    #[test]
    fn translation_shifts_origin_by_the_inverse() {
        let img = img_2d();
        let t = TranslationTransform::new(vec![3.0, -2.0]);
        let out = transform_geometry(&img, &t).unwrap();
        assert_eq!(out.origin(), &[-3.0, 2.0]);
        assert_eq!(out.spacing(), &[1.0, 1.0]);
        assert_eq!(out.direction(), &[1.0, 0.0, 0.0, 1.0]);
    }

    /// Pixel values are copied through unchanged -- this filter only ever
    /// touches geometry.
    #[test]
    fn pixel_values_are_unchanged() {
        let img = Image::from_vec(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let t = TranslationTransform::new(vec![5.0, 5.0]);
        let out = transform_geometry(&img, &t).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// A 90-degree rotation about the origin: `M = [[0,-1],[1,0]]`, `b = 0`.
    /// `M⁻¹ = [[0,1],[-1,0]]`. With `origin = (2, 0)`:
    /// `new_origin = M⁻¹·(2, 0) = (0, -2)`. The direction columns (initially
    /// the identity, spacing 1) rotate the same way:
    /// `new_direction[:,0] = M⁻¹·(1,0) = (0,-1)`,
    /// `new_direction[:,1] = M⁻¹·(0,1) = (1, 0)`; spacing stays 1 (rotation
    /// preserves length).
    #[test]
    fn rotation_composes_origin_and_direction_by_the_inverse_rotation() {
        let mut img = img_2d();
        img.set_origin(&[2.0, 0.0]).unwrap();
        let t = Euler2DTransform::new(std::f64::consts::FRAC_PI_2, [0.0, 0.0], [0.0, 0.0]);
        let out = transform_geometry(&img, &t).unwrap();

        assert!((out.origin()[0] - 0.0).abs() < 1e-12);
        assert!((out.origin()[1] - -2.0).abs() < 1e-12);
        assert_eq!(out.spacing(), &[1.0, 1.0]);
        // Row-major 2x2: d[row * 2 + col].
        let d = out.direction();
        assert!((d[0] - 0.0).abs() < 1e-12, "[0][0]");
        assert!((d[2] - -1.0).abs() < 1e-12, "[1][0]");
        assert!((d[1] - 1.0).abs() < 1e-12, "[0][1]");
        assert!((d[3] - 0.0).abs() < 1e-12, "[1][1]");
    }

    /// A transform of the wrong spatial dimension is rejected before any
    /// geometry math runs.
    #[test]
    fn dimension_mismatch_errors() {
        let img = img_2d();
        let t = TranslationTransform::new(vec![1.0, 1.0, 1.0]);
        assert_eq!(
            transform_geometry(&img, &t),
            Err(TransformError::DimensionMismatch)
        );
    }

    /// `VerifyPreconditions` (`transform->IsLinear()`): a displacement-field
    /// transform is rejected outright, before any geometry math runs.
    #[test]
    fn non_linear_transform_errors() {
        let img = img_2d();
        let t = DisplacementFieldTransform::new(
            2,
            &[4, 4],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            transform_geometry(&img, &t),
            Err(TransformError::NonLinearTransform)
        );
    }

    /// `AffineTransform` with a singular (rank-deficient) matrix has no
    /// inverse: `VerifyPreconditions` would pass (it's still linear), but
    /// deriving `Tinv` fails.
    #[test]
    fn singular_transform_matrix_errors() {
        let img = img_2d();
        // Both rows identical -> rank 1, not invertible.
        let t = AffineTransform::new(2, vec![1.0, 1.0, 1.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
        assert_eq!(
            transform_geometry(&img, &t),
            Err(TransformError::NonInvertibleTransform)
        );
    }
}
