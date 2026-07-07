//! Transform / resampling error type.

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

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for transform results.
pub type Result<T> = std::result::Result<T, TransformError>;
