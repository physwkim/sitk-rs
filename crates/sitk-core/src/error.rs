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

    /// A scalar-only accessor, filter, or writer was handed a vector
    /// (multi-component) image.
    ///
    /// This is the single guard that keeps a scalar consumer from reading an
    /// interleaved vector buffer as if it were one value per pixel. SimpleITK
    /// expresses the same restriction at compile time: a filter whose yaml
    /// declares `pixel_types: BasicPixelIDTypeList` is never instantiated for a
    /// `VectorPixelID`, so calling it on a vector image throws from the
    /// generated wrapper's member-function factory.
    #[error(
        "this operation requires a scalar pixel type, got {pixel_id:?} with {components_per_pixel} components per pixel"
    )]
    RequiresScalarPixelType {
        pixel_id: PixelId,
        components_per_pixel: usize,
    },

    /// A vector-only accessor or filter was handed a scalar image.
    #[error("this operation requires a vector pixel type, got {0:?}")]
    RequiresVectorPixelType(PixelId),

    /// `Image::AllocateInternal` (sitkImage.hxx:63-67) throws "Specified number
    /// of components as N but did not specify pixelID as a vector type!" when a
    /// scalar pixel id is paired with a component count other than 1, and a
    /// vector image needs at least one component.
    #[error("pixel type {pixel_id:?} cannot have {components_per_pixel} components per pixel")]
    InvalidComponentCount {
        pixel_id: PixelId,
        components_per_pixel: usize,
    },

    /// `Image::from_component_images` was given no component images, so the
    /// vector image's component count and pixel type would both be undefined.
    #[error("composing a vector image requires at least one component image")]
    EmptyComponentImageList,

    /// A component index was `>= number_of_components_per_pixel`.
    #[error("component index {index} is out of range for {components_per_pixel} components")]
    ComponentIndexOutOfRange {
        index: usize,
        components_per_pixel: usize,
    },
}

/// Convenience alias for core results.
pub type Result<T> = std::result::Result<T, Error>;
