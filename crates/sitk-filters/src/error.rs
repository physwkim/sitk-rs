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

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for filter results.
pub type Result<T> = std::result::Result<T, FilterError>;
