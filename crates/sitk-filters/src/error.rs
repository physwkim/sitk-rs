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

    /// `SLICImageFilter::SetSuperGridSize` was given a zero super-pixel size,
    /// which would make the grid initialisation divide by zero (ITK feeds it
    /// straight to `ShrinkImageFilter` as a shrink factor).
    #[error("super grid sizes must be >= 1, got {0:?}")]
    InvalidSuperGridSize(Vec<u32>),

    /// An expand factor was zero (must be a positive integer). Distinct from
    /// [`FilterError::InvalidShrinkFactor`] because it names the filter that
    /// actually rejected it.
    #[error("expand factors must be >= 1, got {0:?}")]
    InvalidExpandFactor(Vec<usize>),

    /// `itk::Functor::Clamp::SetBounds` throws when, after intersecting the
    /// caller's requested bounds with the output pixel type's own
    /// representable range, `lower > upper`.
    #[error("invalid clamp bounds: [{lower}; {upper}]")]
    InvalidClampBounds { lower: f64, upper: f64 },

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

    /// `BinaryThresholdImageFilter::BeforeThreadedGenerateData` throws "Lower
    /// threshold cannot be greater than upper threshold." `DoubleThresholdImageFilter`
    /// builds two independent `BinaryThresholdImageFilter` instances (narrow:
    /// Threshold2/Threshold3, wide: Threshold1/Threshold4), each subject to
    /// this same precondition on its own pair.
    #[error("lower threshold {lower} cannot be greater than upper threshold {upper}")]
    InvalidThresholdRange { lower: f64, upper: f64 },

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

    /// A filter whose SimpleITK yaml declares `pixel_types: RealPixelIDTypeList`
    /// (`GradientAnisotropicDiffusionImageFilter`,
    /// `CurvatureAnisotropicDiffusionImageFilter`) was given a non-floating-point
    /// image. SimpleITK never instantiates those wrappers for integer pixel
    /// types, which is a compile-time restriction in C++ and a runtime error
    /// here; ITK's `FiniteDifferenceImageFilter::GenerateData` separately warns
    /// "Output pixel type MUST be float or double to prevent computational
    /// errors".
    #[error("this filter requires a floating-point pixel type, got {0:?}")]
    RequiresRealPixelType(PixelId),

    /// `AnisotropicDiffusionImageFilter::InitializeIteration` gates the
    /// conductance recalibration on
    /// `GetElapsedIterations() % m_ConductanceScalingUpdateInterval == 0`, so a
    /// zero interval is a division by zero.
    #[error("conductance_scaling_update_interval must be >= 1, got 0")]
    ZeroConductanceScalingUpdateInterval,

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

    /// `ReconstructionByErosionImageFilter`/`ReconstructionByDilationImageFilter`,
    /// via `itkReconstructionImageFilter.hxx`'s per-pixel precondition check
    /// ("be sure that the pixels in the images follow the preconditions"):
    /// reconstruction by erosion requires the marker image to be pixelwise
    /// `>=` the mask image; by dilation, `<=`.
    #[error("reconstruction marker image must be pixelwise {relation} the mask image")]
    InvalidReconstructionMarker { relation: &'static str },

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

    /// `IsoContourDistanceImageFilter::ComputeValue` throws
    /// `itkGenericExceptionMacro("diff " << diff << " < NumericTraits<
    /// PixelRealType >::min()")` when the two endpoints of a level-set
    /// crossing are separated by less than the real type's smallest positive
    /// normal value (here `f64::MIN_POSITIVE`), which would make the
    /// `1 / diff` in the distance estimate blow up.
    #[error(
        "iso-contour level-set crossing has a degenerate value difference {0} (< f64::MIN_POSITIVE)"
    )]
    IsoContourDegenerateDifference(f64),

    /// `IsoContourDistanceImageFilter::ComputeValue` throws
    /// `itkExceptionStringMacro("Gradient norm is lower than pixel
    /// precision")` when the interpolated gradient at a level-set crossing has
    /// no magnitude, leaving the crossing's normal direction undefined.
    #[error("iso-contour gradient norm is lower than pixel precision")]
    IsoContourZeroGradient,

    /// A convolution filter was given a kernel of a different dimension than
    /// the image. ITK expresses this as a template parameter shared by both
    /// inputs (`itk::ConvolutionImageFilter<TInputImage, TKernelImage, ...>`,
    /// all of `ImageDimension`), so the mismatch cannot be built in C++.
    #[error("kernel dimension {kernel} does not match image dimension {image}")]
    KernelDimensionMismatch { image: usize, kernel: usize },

    /// A convolution filter was given a kernel with a zero-length axis, for
    /// which `GetKernelPadSize` would pad an all-zero operator into existence
    /// and `GetKernelRadius` would report radius 0.
    #[error("convolution kernel must have at least one pixel along every axis, got {0:?}")]
    EmptyKernel(Vec<usize>),

    /// `Normalize` was requested for a kernel whose pixels sum to zero.
    /// `NormalizeToConstantImageFilter` divides by `GetSum() / Constant`
    /// (itkNormalizeToConstantImageFilter.hxx:71-73), so ITK's `Div` functor
    /// would replace every coefficient with `NumericTraits<RealType>::max()`.
    #[error("normalize requires a convolution kernel whose pixels sum to a non-zero value")]
    ZeroKernelSum,

    /// `CheckerBoardImageFilter::GenerateData` computes `factors[d] =
    /// size[d] / checker_pattern[d]` (integer division) and later divides an
    /// index by `factors[d]`; a pattern count of `0`, or one exceeding the
    /// image size along that axis (making `factors[d]` truncate to `0`),
    /// would be an integer division by zero in the C++.
    #[error(
        "checker_pattern {pattern:?} must be >= 1 and <= the image size {size:?} along every axis"
    )]
    InvalidCheckerPattern { pattern: Vec<u32>, size: Vec<usize> },

    /// `TileImageFilter` was given zero input images to lay out.
    #[error("tile requires at least one input image")]
    EmptyImageList,

    /// `UnsharpMaskImageFilter::VerifyPreconditions` throws "Threshold must
    /// be non-negative!" when `Threshold < 0`.
    #[error("unsharp_mask threshold must be >= 0, got {0}")]
    InvalidUnsharpThreshold(f64),

    /// `BinaryThinningImageFilter::ComputeThinImage` hardcodes 2-D 8-neighbor
    /// offsets (`itkBinaryThinningImageFilter.hxx`'s `o2`..`o9`, each a
    /// 2-element `OffsetType`), so ITK only wraps/instantiates this filter
    /// for 2-D images (`itkBinaryThinningImageFilter.wrap`'s
    /// `itk_wrap_image_filter(..., 2, 2)`); a higher-dimensional
    /// instantiation would not even compile in C++.
    #[error("binary_thinning only supports 2-D images, got {0}-D")]
    UnsupportedThinningDimension(usize),

    /// `ShapeLabelMapFilter::PerimeterFromInterceptCount` has hand-tuned
    /// direction-weight overloads for 2-D and 3-D only
    /// (`itkShapeLabelMapFilter.hxx`'s `MapIntercept2Type` /
    /// `MapIntercept3Type` specializations), and SimpleITK only instantiates
    /// `LabelShapeStatisticsImageFilter` for those two dimensions.
    #[error("label_shape_statistics only supports 2-D and 3-D images, got {0}-D")]
    UnsupportedShapeDimension(usize),

    /// `DirectedHausdorffDistanceImageFilter::AfterThreadedGenerateData`
    /// throws `"pixelcount is equal to 0"` via `itkGenericExceptionMacro`
    /// when the first input image has no non-zero pixels (the "from" set
    /// is empty, so no minimum distance can be measured from it).
    #[error(
        "directed Hausdorff distance requires at least one non-zero pixel in the first input image"
    )]
    EmptyHausdorffForegroundSet,

    /// `BSplineScatteredDataPointSetToImageFilter::SetSplineOrder` and
    /// `BSplineControlPointImageFilter::SetSplineOrder` both throw "The spline
    /// order in each dimension must be greater than 0".
    #[error("spline_order must be >= 1, got 0")]
    InvalidSplineOrder,

    /// `BSplineScatteredDataPointSetToImageFilter::GenerateData` throws "The
    /// number of control points must be greater than the spline order" when
    /// `m_NumberOfControlPoints[i] < m_SplineOrder[i] + 1` — a lattice that
    /// short cannot support even one B-spline span.
    #[error(
        "axis {axis} has {control_points} control points, which must exceed the spline order {spline_order}"
    )]
    InvalidControlPointCount {
        axis: usize,
        control_points: usize,
        spline_order: usize,
    },

    /// A B-spline filter's parametric reparameterization
    /// (`r[i] = spans[i] / ((size[i] - 1) * spacing[i])`) divides by
    /// `size[i] - 1`, so every axis needs at least two pixels.
    #[error("b-spline fitting needs at least 2 pixels along every axis, got size {0:?}")]
    BSplineAxisTooShort(Vec<usize>),

    /// `BSplineScatteredDataPointSetToImageFilter::ThreadedGenerateDataForFitting`
    /// and `BSplineControlPointImageFilter::DynamicThreadedGenerateData` both
    /// throw "The ... point component ... is outside the corresponding
    /// parametric domain of [0, spans)" for a sample that does not lie on the
    /// parametric grid, after the `m_BSplineEpsilon` edge tolerance.
    #[error(
        "b-spline parametric coordinate {value} on axis {axis} is outside the domain [0, {spans})"
    )]
    BSplineParametricDomain { axis: usize, value: f64, spans: f64 },

    /// `N4BiasFieldCorrectionImageFilter::SharpenImage` divides by
    /// `m_NumberOfHistogramBins - 1` to build the histogram slope.
    #[error("n4 number_of_histogram_bins must be >= 2, got {0}")]
    N4InvalidHistogramBins(u32),

    /// `N4BiasFieldCorrectionImageFilter::GenerateData` reaches the end of the
    /// first fitting level with `m_LogBiasFieldControlPointLattice` still null
    /// — and then dereferences it — whenever the inner iteration loop never
    /// runs, i.e. `maximum_number_of_iterations` is empty or its first entry is
    /// zero, or `convergence_threshold` is not below `RealType`'s maximum.
    #[error("n4 completed its first fitting level without estimating a bias field")]
    N4NoBiasFieldEstimated,

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for filter results.
pub type Result<T> = std::result::Result<T, FilterError>;
