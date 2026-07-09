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

    /// `DiscreteGaussianImageFilter`'s per-axis `Variance` was negative.
    #[error("discrete Gaussian variances must be >= 0, got {0:?}")]
    InvalidVariance(Vec<f64>),

    /// `GaussianOperator::SetMaximumError`: every value must lie in the open
    /// interval `(0.0, 1.0)`.
    #[error("maximum_error values must be in the open interval (0.0, 1.0), got {0:?}")]
    InvalidMaximumError(Vec<f64>),

    /// `CurvatureFlowImageFilter`'s `TimeStep` violates the explicit-scheme
    /// stability bound derived for the finite-difference curvature update
    /// (see `curvature_flow`'s doc comment): the caller must pick a
    /// `time_step` in `[0, max_stable]`.
    #[error("time_step {time_step} is outside the stable range [0, {max_stable}]")]
    UnstableTimeStep { time_step: f64, max_stable: f64 },

    /// The recursive Gaussian was asked to filter an axis shorter than the
    /// four pixels its fourth-order recursion needs (matching ITK's
    /// `RecursiveSeparableImageFilter` requirement).
    #[error("axis {axis} has {len} pixels; the recursive Gaussian needs at least 4")]
    AxisTooShortForRecursion { axis: usize, len: usize },

    /// An axis-index parameter was not a valid axis of the input image:
    /// `DerivativeImageFilter`'s `direction`, or a projection filter's
    /// `ProjectionDimension` (`itkProjectionImageFilter.hxx`'s
    /// `GenerateOutputInformation`/`DynamicThreadedGenerateData`, both of
    /// which throw when `m_ProjectionDimension >= InputImageDimension`).
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

    /// `IsolatedConnectedImageFilter::GenerateData` requires both seed lists
    /// to be non-empty (`itkIsolatedConnectedImageFilter.hxx` errors out via
    /// `itkExceptionMacro` when either is empty, rather than silently
    /// producing an empty output as the flood-fill-only filters do).
    #[error("{which} must be non-empty")]
    EmptySeeds { which: &'static str },

    /// A histogram-driven threshold calculator's `.hxx` throws via
    /// `itkGenericExceptionMacro`/`itkExceptionStringMacro` on a numerical
    /// degeneracy or non-convergence it cannot recover from:
    /// `IntermodesThresholdCalculator`'s bimodal-smoothing iteration cap
    /// (`itkIntermodesThresholdCalculator.hxx`) or
    /// `KittlerIllingworthThresholdCalculator`'s division-by-near-zero
    /// guards (`itkKittlerIllingworthThresholdCalculator.hxx`).
    #[error("{calculator} threshold calculator failed: {reason}")]
    ThresholdCalculatorFailed {
        calculator: &'static str,
        reason: &'static str,
    },

    /// `MorphologicalWatershedImageFilter`'s `Level`, cast to the input pixel
    /// type, was negative. A negative level would make the `HMinimaImageFilter`
    /// marker (`input + level`) fall below its mask (`input`), which
    /// `itkReconstructionImageFilter.hxx` rejects outright ("Marker pixels must
    /// be <= mask pixels.").
    #[error("watershed level must be >= 0, got {0}")]
    InvalidWatershedLevel(f64),

    /// `FastMarchingImageFilter::GenerateData` throws "Normalization Factor
    /// is null or negative" when `m_NormalizationFactor < itk::Math::eps`
    /// (`f64::EPSILON`).
    #[error("normalization_factor must be >= f64::EPSILON, got {0}")]
    InvalidNormalizationFactor(f64),

    /// `FastMarchingImageFilter::UpdateValue` throws "Discriminant of
    /// quadratic equation is negative" rather than picking a root, when the
    /// upwind quadratic `bb^2 - aa*cc` comes out below zero.
    #[error("discriminant of the fast marching quadratic equation is negative")]
    NegativeDiscriminant,

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for filter results.
pub type Result<T> = std::result::Result<T, FilterError>;
