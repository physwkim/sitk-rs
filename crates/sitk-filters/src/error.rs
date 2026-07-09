//! Filter error type.

use sitk_core::PixelId;

/// Errors produced by filters.
///
/// `PartialEq` (but not `Eq`) because sigma-carrying variants hold `f64`.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum FilterError {
    /// Two-input filter given images of differing size.
    #[error("input images have different sizes: {a:?} vs {b:?}")]
    SizeMismatch { a: Vec<usize>, b: Vec<usize> },

    /// Two-input filter given images of differing pixel type.
    #[error("input images have different pixel types: {a:?} vs {b:?}")]
    TypeMismatch { a: PixelId, b: PixelId },

    /// A rescale/threshold could not proceed because the input range is empty.
    #[error("input intensity range is degenerate (min == max) or image is empty")]
    DegenerateRange,

    /// A per-dimension parameter (shrink factors, sigmas) had the wrong length.
    #[error("expected {expected} values (one per dimension), got {got}")]
    DimensionLength { expected: usize, got: usize },

    /// `StructuringElement::from_mask` was given a mask whose length doesn't
    /// match `radius`'s implied window size (`Π (2*radius[d]+1)`).
    #[error("structuring element mask expected {expected} values (Π 2*radius[d]+1), got {got}")]
    MaskLengthMismatch { expected: usize, got: usize },

    /// A shrink factor was zero (must be a positive integer).
    #[error("shrink factors must be >= 1, got {0:?}")]
    InvalidShrinkFactor(Vec<usize>),

    /// A smoothing sigma was negative.
    #[error("smoothing sigmas must be >= 0, got {0:?}")]
    InvalidSigma(Vec<f64>),

    /// The recursive Gaussian was asked to filter an axis shorter than the
    /// four pixels its fourth-order recursion needs (matching ITK's
    /// `RecursiveSeparableImageFilter` requirement).
    #[error("axis {axis} has {len} pixels; the recursive Gaussian needs at least 4")]
    AxisTooShortForRecursion { axis: usize, len: usize },

    /// `DerivativeImageFilter`'s `direction` was not a valid axis of the
    /// input image.
    #[error("direction {direction} is out of range for a {dimension}-D image")]
    InvalidDirection { direction: usize, dimension: usize },

    /// `RelabelComponentImageFilter` would need to assign more surviving
    /// objects than the output pixel type can represent
    /// (`NumericTraits<OutputPixelType>::max()`).
    #[error("relabel needs more than {max} object labels, exceeding the output pixel type's range")]
    TooManyObjects { max: u64 },

    /// A histogram-driven threshold calculator (Otsu, Triangle, ...) was
    /// asked for fewer than one histogram bin.
    #[error("number of histogram bins must be >= 1, got {0}")]
    InvalidHistogramBins(u32),

    /// Otsu's multi-threshold search needs strictly more histogram bins than
    /// requested thresholds (`itkOtsuMultipleThresholdsCalculator.hxx`'s
    /// `IncrementThresholds` indexes off `numberOfHistogramBins - 2 - ...`,
    /// which requires this to hold).
    #[error(
        "otsu threshold search needs number_of_histogram_bins ({bins}) > number_of_thresholds ({thresholds})"
    )]
    InvalidThresholdCount { bins: u32, thresholds: u32 },

    /// `RegionOfInterestImageFilter`'s requested region does not fit inside
    /// the input's `LargestPossibleRegion`.
    #[error(
        "region [index={index:?}, size={size:?}] does not fit inside input of size {input_size:?}"
    )]
    RegionOutOfBounds {
        index: Vec<usize>,
        size: Vec<usize>,
        input_size: Vec<usize>,
    },

    /// `CropImageFilter::VerifyInputInformation`: the requested crop bounds
    /// exceed the input's size along some axis.
    #[error("axis {axis} crop bounds (lower={lower}, upper={upper}) exceed input size {size}")]
    InvalidCropBounds {
        axis: usize,
        lower: usize,
        upper: usize,
        size: usize,
    },

    /// `ExtractImageFilter::SetExtractionRegion`: every size component was
    /// zero, leaving no output dimensions.
    #[error("extraction size collapses every axis; at least one axis must be non-zero")]
    ExtractCollapsedAllAxes,

    /// `ExtractImageFilter::GenerateOutputInformation`
    /// (`DIRECTIONCOLLAPSETOSUBMATRIX`): the submatrix of retained direction
    /// cosines is singular.
    #[error("collapsed direction submatrix is singular")]
    SingularCollapsedDirection,

    /// `PermuteAxesImageFilter::SetOrder`: `order` is not a rearrangement of
    /// `0..dim`.
    #[error("order {0:?} is not a permutation of 0..{1}")]
    InvalidPermutation(Vec<usize>, usize),

    /// A bitwise/logic filter (`And`/`Or`/`Xor`/`Not`) was given a
    /// floating-point image. ITK's `BitwiseOperators`/`NotOperator` concept
    /// checks only instantiate for integer pixel types (`itkAndImageFilter.h`
    /// et al.; SimpleITK's generated wrappers restrict these to
    /// `IntegerPixelIDTypeList`), which is a compile error in C++ and a
    /// runtime error here.
    #[error("bitwise/logic filters require an integer pixel type, got {0:?}")]
    RequiresIntegerPixelType(PixelId),

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for filter results.
pub type Result<T> = std::result::Result<T, FilterError>;
