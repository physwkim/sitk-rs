//! Transform / resampling error type.

use crate::core::PixelId;

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

    /// `WarpImageFilter::VerifyInputInformation` (itkWarpImageFilter.hxx:103-109)
    /// throws "Expected number of components of displacement field to match
    /// image dimensions!" when the field's pixels are not `ImageDimension`
    /// components long.
    #[error(
        "displacement field has {got} components per pixel; expected {expected} (the image dimension)"
    )]
    DisplacementFieldComponentMismatch { expected: usize, got: usize },

    /// `VersorRigid3DTransform::set_matrix` was given a matrix that is not a
    /// proper rotation (orthonormal, `det ≥ 0`) to within `itk::Versor<double>`'s
    /// own tolerance (`Versor::Epsilon() = 1e-10`; `1e-7` is only the `float`
    /// specialization, `itkVersor.h:305-309`). Mirrors the
    /// `itkGenericExceptionMacro` guard in `Versor<T>::Set(const MatrixType&)`.
    #[error("matrix is not a proper rotation (must be orthonormal with determinant >= 0)")]
    NotARotationMatrix,

    /// [`ParametricTransform::set_fixed_parameters`] was handed an array this
    /// transform cannot accept. ITK's `SetFixedParameters` overrides throw an
    /// `itkExceptionMacro` in the same situation
    /// (`itkMatrixOffsetTransformBase.hxx:458-465`,
    /// `itkEuler3DTransform.hxx:131-139`).
    ///
    /// [`ParametricTransform::set_fixed_parameters`]: crate::transform::ParametricTransform::set_fixed_parameters
    #[error("invalid fixed parameters: got {got} value(s), expected {expected}")]
    InvalidFixedParameters { got: usize, expected: String },

    /// [`ParametricTransform::set_parameters`] was handed an array whose length
    /// is not [`ParametricTransform::number_of_parameters`]. ITK's
    /// `SetParameters` overrides throw in the same situation, though upstream's
    /// own strictness is per-class and inconsistent (`MatrixOffsetTransformBase`
    /// and `TranslationTransform` throw only when the vector is *shorter*,
    /// `VersorTransform` checks nothing at all, `BSplineTransform` demands exact
    /// equality); this port requires exact equality everywhere (ledger §4.47).
    ///
    /// [`ParametricTransform::set_parameters`]: crate::transform::ParametricTransform::set_parameters
    /// [`ParametricTransform::number_of_parameters`]: crate::transform::ParametricTransform::number_of_parameters
    #[error("invalid parameters: got {got} value(s), expected {expected}")]
    InvalidParameters { got: usize, expected: usize },

    /// [`Transform::inverse`] was called on a transform that has no inverse,
    /// either because the transform class does not define one or because this
    /// instance's linear map is singular. Mirrors `itk::simple::Transform::
    /// GetInverse`, which throws when `SetInverse` fails
    /// (`sitkTransform.cxx:542-552`).
    ///
    /// [`Transform::inverse`]: crate::transform::Transform::inverse
    #[error("transform has no inverse: {0}")]
    NoInverse(&'static str),

    /// [`WarpImageFilter::set_output_size`] was given a size with a zero extent
    /// on some axis. Upstream `WarpImageFilter` treats `m_OutputSize[0] == 0` as
    /// the "size unset, inherit the field's" sentinel — on the *first* axis
    /// alone — so an explicit `[0, 5, 5]` silently discards the `5`s
    /// (itkWarpImageFilter.hxx:428). This port keys "unset" on the `Option`
    /// being `None` instead, so an explicitly-set size is honored; a size that
    /// actually has a zero-pixel axis is a malformed request and is rejected
    /// here rather than silently ignored (ledger §2.37).
    ///
    /// [`WarpImageFilter::set_output_size`]: crate::transform::WarpImageFilter::set_output_size
    #[error("invalid output size {0:?}: every axis must have a non-zero extent")]
    InvalidOutputSize(Vec<usize>),

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] crate::core::Error),
}

/// Convenience alias for transform results.
pub type Result<T> = std::result::Result<T, TransformError>;
