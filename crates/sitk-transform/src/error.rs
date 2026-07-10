//! Transform / resampling error type.

use sitk_core::PixelId;

/// Errors produced by transforms and resampling.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TransformError {
    /// Input image, output geometry, and transform dimensions disagree.
    #[error("dimension mismatch between input, output geometry, and transform")]
    DimensionMismatch,

    /// The input direction matrix could not be inverted for resampling.
    #[error("input direction matrix is singular")]
    SingularDirection,

    /// A B-spline transform was constructed with a zero mesh size in some
    /// dimension, or with per-dimension arguments of inconsistent length. The
    /// control-point grid spacing (`physicalDimensions / meshSize`) is then
    /// undefined.
    #[error("invalid B-spline transform domain (mesh size must be ≥ 1 in every dimension)")]
    InvalidBSplineDomain,

    /// A displacement-field transform was constructed with per-dimension
    /// arguments (size, origin, spacing, direction) of inconsistent length, or
    /// an empty field grid.
    #[error("invalid displacement-field domain (inconsistent per-dimension geometry)")]
    InvalidDisplacementFieldDomain,

    /// `TransformGeometryImageFilter::VerifyPreconditions` requires
    /// `Transform::IsLinear()`; a B-spline, displacement-field, or
    /// not-fully-linear composite transform fails this precondition.
    #[error("transform must be linear for TransformGeometryImageFilter")]
    NonLinearTransform,

    /// The transform's own linear map could not be inverted.
    /// `itkTransformGeometryImageFilter.hxx` calls `GetInverseTransform()`,
    /// which for a singular `MatrixOffsetTransformBase` returns a null
    /// pointer in C++ that the filter then dereferences unchecked (a latent
    /// crash upstream); this port returns an error instead of reproducing
    /// that undefined behavior.
    #[error("transform matrix is singular; cannot invert for TransformGeometryImageFilter")]
    NonInvertibleTransform,

    /// `TransformToDisplacementFieldFilter`'s `OutputPixelType` member accepts
    /// only `sitkVectorFloat32` and `sitkVectorFloat64`
    /// (`TransformToDisplacementFieldFilter.yaml`'s `briefdescriptionSet`:
    /// "only sitkVectorFloat32 and sitkVectorFloat64 are supported").
    #[error(
        "displacement-field output pixel type must be VectorFloat32 or VectorFloat64, got {0:?}"
    )]
    UnsupportedDisplacementFieldPixelType(PixelId),

    /// `WarpImageFilter::VerifyInputInformation` (itkWarpImageFilter.hxx:103-108)
    /// throws "Expected number of components of displacement field to match
    /// image dimensions!" when the field's pixels are not `ImageDimension`
    /// components long.
    #[error(
        "displacement field has {got} components per pixel; expected {expected} (the image dimension)"
    )]
    DisplacementFieldComponentMismatch { expected: usize, got: usize },

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for transform results.
pub type Result<T> = std::result::Result<T, TransformError>;
