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

    /// A scalar-only accessor, filter, or writer was handed a non-scalar image
    /// — a vector (multi-component) or complex one.
    ///
    /// This is the single guard that keeps a scalar consumer from reading an
    /// interleaved buffer as if it were one value per pixel. SimpleITK
    /// expresses the same restriction at compile time: a filter whose yaml
    /// declares `pixel_types: BasicPixelIDTypeList` is never instantiated for a
    /// `VectorPixelID`, so calling it on a vector image throws from the
    /// generated wrapper's member-function factory.
    #[error("this operation requires a scalar pixel type, got {0:?}")]
    RequiresScalarPixelType(PixelId),

    /// A vector-only accessor or filter was handed a non-vector image.
    #[error("this operation requires a vector pixel type, got {0:?}")]
    RequiresVectorPixelType(PixelId),

    /// A complex-only accessor or filter was handed a non-complex image —
    /// SimpleITK's `pixel_types: ComplexPixelIDTypeList`
    /// (sitkPixelIDTypeLists.h:104), which instantiates the wrapper for
    /// `sitkComplexFloat32`/`sitkComplexFloat64` and no other pixel type.
    #[error("this operation requires a complex pixel type, got {0:?}")]
    RequiresComplexPixelType(PixelId),

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

    /// A pixel accessor was given an index shorter than the image dimension.
    ///
    /// SimpleITK's `sitkSTLVectorToITK` (sitkTemplateFunctions.h:100-105) throws
    /// "Expected vector of length D but only got N elements." when
    /// `idx.size() < D`; a *longer* index is accepted and its extra elements are
    /// ignored (`sitkImage.h:499-501`).
    #[error("pixel index needs at least {dimension} elements, got {actual}")]
    IndexDimensionMismatch { dimension: usize, actual: usize },

    /// A pixel accessor was given an index outside the image.
    ///
    /// SimpleITK's `PimpleImage::GetIndex` (sitkPimpleImageBase.hxx:788-797)
    /// throws "index out of bounds" when the index leaves the largest possible
    /// region: "Boundary checking is performed on idx, if it is out of bounds an
    /// exception will be thrown" (sitkImage.h:501-502).
    #[error("pixel index {index:?} is outside an image of size {size:?}")]
    IndexOutOfBounds { index: Vec<usize>, size: Vec<usize> },

    /// A label-only operation was handed a floating-point or vector image.
    ///
    /// SimpleITK expresses the same restriction at compile time:
    /// `LabelImageToLabelMapFilter.yaml` declares
    /// `pixel_types: UnsignedIntegerPixelIDTypeList`, so the filter is never
    /// instantiated for a float or vector pixel id.
    #[error("this operation requires an integer scalar pixel type, got {0:?}")]
    RequiresIntegerPixelType(PixelId),

    /// A [`LabelMap`](crate::LabelMap) was asked for a dimension outside
    /// `1..=`[`MAX_DIM`](crate::label_map::MAX_DIM).
    #[error("label maps support dimensions 1..=3, got {0}")]
    UnsupportedLabelMapDimension(usize),

    /// A [`LabelObjectLine`](crate::LabelObjectLine) would have covered no
    /// pixels. Upstream stores such a line; the "optimized" invariant of
    /// [`LabelObject`](crate::LabelObject) makes it unrepresentable.
    #[error("a label object line must cover at least one pixel, got length {0}")]
    NonPositiveLineLength(i64),

    /// A [`LabelObject`](crate::LabelObject) carrying the map's background
    /// value was inserted into a [`LabelMap`](crate::LabelMap). Upstream admits
    /// it into the container and then throws at every `GetLabelObject` and
    /// `RemoveLabel` (`itkLabelMap.hxx:110-116`, `:453-459`).
    #[error("label {0} is the label map's background value")]
    LabelIsBackground(i64),

    /// A label (or a background value) outside the `NumericTraits` range of the
    /// [`LabelMap`](crate::LabelMap)'s `pixel_id` was offered to it. ITK cannot
    /// represent such a state at all — its `LabelType` *is* the label image's
    /// pixel type, so the conversion happens in the caller's `static_cast`. Here
    /// a label is an `i64` throughout, so the map enforces the range itself.
    #[error("label {label} is not representable in a {pixel_id:?} label image")]
    LabelOutOfRange { label: i64, pixel_id: PixelId },

    /// [`LabelMap::push_label_object`](crate::LabelMap::push_label_object) found
    /// no free label. Upstream's `itkExceptionStringMacro("Can't push the label
    /// object: the label map is full.")` (`itkLabelMap.hxx:431-434`).
    #[error("can't push the label object: the label map is full")]
    LabelMapFull,
}

/// Convenience alias for core results.
pub type Result<T> = std::result::Result<T, Error>;
