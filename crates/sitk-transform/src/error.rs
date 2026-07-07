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

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for transform results.
pub type Result<T> = std::result::Result<T, TransformError>;
