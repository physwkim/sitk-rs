//! Error type shared across the sitk-rs core.

use crate::pixel::PixelId;

/// Errors produced by core image operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// A typed buffer length did not match the product of the image size.
    #[error("buffer size mismatch: expected {expected} pixels, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    /// A geometry vector (spacing/origin/direction) had the wrong length for the
    /// image dimension.
    #[error("geometry vector length does not match image dimension {dimension}")]
    GeometryMismatch { dimension: usize },

    /// A spacing component was zero or negative.
    #[error("spacing components must be strictly positive")]
    NonPositiveSpacing,

    /// A typed accessor was called with the wrong pixel type.
    #[error("pixel type mismatch: image is {expected:?}, requested {requested:?}")]
    PixelTypeMismatch {
        expected: PixelId,
        requested: PixelId,
    },

    /// The direction cosine matrix could not be inverted.
    #[error("direction matrix is singular and cannot be inverted")]
    SingularDirection,

    /// A neighborhood radius did not have one entry per image dimension.
    #[error("radius length does not match image dimension {dimension}")]
    RadiusMismatch { dimension: usize },
}

/// Convenience alias for core results.
pub type Result<T> = std::result::Result<T, Error>;
