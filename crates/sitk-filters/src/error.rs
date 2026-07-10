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

    /// Two-input filter whose inputs may differ in size but not in
    /// dimensionality (`MaskedFFTNormalizedCorrelationImageFilter`, whose
    /// `ImageDimension` is one compile-time constant shared by both inputs).
    #[error("input images have different dimensions: {a} vs {b}")]
    ImageDimensionMismatch { a: usize, b: usize },

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

    /// `RankImageFilter`'s port ([`crate::rank::rank`]): the structuring
    /// element's on-cells were entirely cropped away at some pixel, leaving
    /// nothing to select a rank from. Upstream's `RankHistogram` would
    /// instead compute `m_Entries - 1` on an unsigned `SizeValueType` while
    /// `m_Entries == 0`, silently wrapping to a huge value rather than
    /// erroring — well-defined C++ unsigned overflow, but nonsensical
    /// output; this port rejects the condition explicitly instead of
    /// reproducing the wraparound. Only reachable through a caller-built
    /// `StructuringElement::from_mask` whose on-mask excludes the center
    /// offset, since `box_`/`cross`/`ball` always keep the center on (always
    /// in-bounds for every pixel). Tracked in the upstream-findings ledger,
    /// §4.32.
    #[error("rank filter's structuring element window is empty at some pixel")]
    EmptyRankNeighborhood,

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

    /// A bin-shrink factor was zero (must be a positive integer). Distinct
    /// from [`FilterError::InvalidShrinkFactor`] because it names the filter
    /// that actually rejected it -- `BinShrinkImageFilter`'s vector-form
    /// `SetShrinkFactors(ShrinkFactorsType)` (the one the `dim_vec`
    /// procedural setter uses) does not clamp to `>= 1` the way its scalar
    /// convenience setter does, so an unclamped `0` would otherwise reach
    /// `GenerateOutputInformation`'s division and cast the resulting
    /// infinity to an integer type, which is undefined behavior in the
    /// original C++.
    #[error("bin shrink factors must be >= 1, got {0:?}")]
    InvalidBinShrinkFactor(Vec<usize>),

    /// `BinShrinkImageFilter::GenerateOutputInformation` throws "InputImage
    /// is too small! An output pixel does not map to a whole input bin."
    /// when `floor(inputSize[axis]/factor) < 1` -- i.e. the shrink factor
    /// exceeds the image's size along that axis. Unlike
    /// [`FilterError::InvalidShrinkFactor`]'s filter (`ShrinkImageFilter`
    /// silently clamps the output size to a minimum of 1 pixel instead),
    /// `BinShrinkImageFilter` throws outright, so this port does too.
    #[error(
        "bin shrink factor {factor} on axis {axis} exceeds input size {size} (output size would be < 1)"
    )]
    BinShrinkFactorTooLarge {
        axis: usize,
        size: usize,
        factor: usize,
    },

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

    /// A filter whose SimpleITK yaml declares `pixel_types:
    /// IntegerPixelIDTypeList` was given a floating-point image. SimpleITK
    /// never instantiates those wrappers for `Float32`/`Float64`, which is a
    /// compile-time restriction in C++ and a runtime error here.
    ///
    /// For the bitwise/logic filters (`And`/`Or`/`Xor`/`Not`) the restriction
    /// also comes from ITK itself: the `BitwiseOperators`/`NotOperator`
    /// concept checks only instantiate for integer pixel types
    /// (`itkAndImageFilter.h` et al.). Other users of this variant --
    /// `AntiAliasBinaryImageFilter`, `LabelShapeStatisticsImageFilter`,
    /// `LabelOverlapMeasuresImageFilter` -- are restricted by the yaml alone.
    #[error("this filter requires an integer pixel type, got {0:?}")]
    RequiresIntegerPixelType(PixelId),

    /// A filter whose SimpleITK yaml restricts it to the unsigned integer
    /// pixel types (`LabelVotingImageFilter`,
    /// `MultiLabelSTAPLEImageFilter`), whose label values index a vote or
    /// confusion-matrix row directly.
    #[error("this filter requires an unsigned integer pixel type, got {0:?}")]
    RequiresUnsignedIntegerPixelType(PixelId),

    /// `UnaryMinusImageFilter.yaml`'s `pixel_types` is `SignedPixelIDTypeList`
    /// (plus a complex-pixel branch this crate does not support): ITK's
    /// `Functor::UnaryMinus` doc comment notes "Assumed that the output type
    /// is signed", and only instantiates for signed types in C++.
    #[error("this filter requires a signed pixel type, got {0:?}")]
    RequiresSignedPixelType(PixelId),

    /// `MultiLabelSTAPLEImageFilter::InitializePriorProbabilities` throws when
    /// the caller-supplied prior array is shorter than the number of labels.
    #[error("prior_probabilities needs at least {expected} entries (one per label), got {got}")]
    InvalidPriorProbabilities { got: usize, expected: usize },

    /// A filter whose SimpleITK yaml declares `pixel_types: RealPixelIDTypeList`
    /// (`GradientAnisotropicDiffusionImageFilter`,
    /// `CurvatureAnisotropicDiffusionImageFilter`) was given a non-floating-point
    /// image. SimpleITK never instantiates those wrappers for integer pixel
    /// types, which is a compile-time restriction in C++ and a runtime error
    /// here; ITK's `FiniteDifferenceImageFilter::GenerateData` separately warns
    /// "Output pixel type MUST be float or double to prevent computational
    /// errors".
    ///
    /// Also reused for `pixel_types: RealVectorPixelIDTypeList`
    /// (`VectorConnectedComponentImageFilter`): the same restriction, checked
    /// against the *component* type of a vector image rather than a scalar
    /// image's own type (`itkVectorConnectedComponentImageFilter.h`'s
    /// `InputValyeTypeIsFloatingCheck` concept check -- sic, upstream's own
    /// typo).
    #[error("this filter requires a floating-point pixel type, got {0:?}")]
    RequiresRealPixelType(PixelId),

    /// `BinaryPruningImageFilter.yaml`'s `pixel_types` is
    /// `typelist2::typelist<BasicPixelID<uint8_t>>` -- SimpleITK's generated
    /// `BinaryPruning()` wrapper only instantiates the filter for `UInt8`,
    /// even though the underlying ITK template also accepts other unsigned
    /// integer and real pixel types (`itkBinaryPruningImageFilter.wrap`'s
    /// `WRAP_ITK_USIGN_INT`/`WRAP_ITK_REAL` groups).
    #[error("binary_pruning only supports UInt8 images, got {0:?}")]
    RequiresUInt8PixelType(PixelId),

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

    /// `itk::watershed::SegmentTreeGenerator::CompileMergeList` throws
    /// ("An unexpected and fatal error has occurred.") when a segment in the
    /// table has an empty adjacency list, because it then dereferences
    /// `edge_list.front()`. That happens exactly when the initial
    /// segmentation found a single segment covering the whole image — a flat
    /// image, or one whose `Threshold` flooded away every minimum but one.
    #[error(
        "the watershed initial segmentation produced a segment with no adjacencies \
         (segment {label}); the image segments to a single region"
    )]
    WatershedSegmentWithoutEdges { label: u64 },

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

    /// `RegularizedHeavisideStepFunction::SetEpsilon` throws unless
    /// `epsilon > NumericTraits<double>::epsilon()`. Only the regularised
    /// variants validate it; `HeavisideStepFunction` ignores `epsilon`.
    #[error("epsilon must be > f64::EPSILON for a regularized Heaviside, got {0}")]
    InvalidHeavisideEpsilon(f64),

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

    /// A multi-input filter (`TileImageFilter`, `STAPLEImageFilter`,
    /// `LabelVotingImageFilter`, `MultiLabelSTAPLEImageFilter`) was given zero
    /// input images.
    #[error("this filter requires at least one input image")]
    EmptyImageList,

    /// `ImageToImageFilter::VerifyInputInformation` (`itkImageToImageFilter.hxx`)
    /// throws "Inputs do not occupy the same physical space!" when an input's
    /// origin, spacing or direction differs from the primary input's by more
    /// than `GlobalDefaultCoordinateTolerance`/`GlobalDefaultDirectionTolerance`
    /// (both `1e-6` by default). `JoinSeriesImageFilter` does not override this
    /// base check, so it still applies across every joined input.
    #[error("input image {index} does not occupy the same physical space as the first input")]
    PhysicalSpaceMismatch { index: usize },

    /// An input to a multi-input filter is smaller, along some axis, than the
    /// primary (first) input, whose extent determines the filter's requested
    /// region for every input. Upstream this surfaces as ITK's
    /// `InvalidRequestedRegionError`, thrown when the pipeline propagates the
    /// primary input's region onto an input too small to contain it -- e.g.
    /// `JoinSeriesImageFilter::GenerateInputRequestedRegion`, which copies the
    /// output region (sized from the first input) onto every input unchanged.
    #[error(
        "input image {index} has size {size:?}, smaller than the primary input's size {primary_size:?}"
    )]
    InputSmallerThanPrimary {
        index: usize,
        size: Vec<usize>,
        primary_size: Vec<usize>,
    },

    /// `MaskedAssignImageFilter.yaml`'s `filter_type` fixes the mask image's
    /// ITK template parameter to `itk::Image<std::uint8_t, ...>`, with no
    /// fallback casting path the way `MaskImageFilter.yaml`'s `MaskImage`
    /// input has (that filter's generated wrapper re-derives a `UInt8` mask
    /// via `NotEqual`, with a deprecation warning, when given another
    /// integer pixel type). This filter's generated wrapper has no such
    /// `custom_itk_cast`, so SimpleITK's default `CastImageToITK` -- a
    /// `dynamic_cast`, not a value cast -- throws outright on any other
    /// pixel type; this port reproduces that as a hard runtime error.
    #[error("mask image must be UInt8, got {0:?}")]
    RequiresUInt8MaskPixelType(PixelId),

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

    /// `BinaryPruningImageFilter::ComputePruneImage` hardcodes 2-D 8-neighbor
    /// offsets (`itkBinaryPruningImageFilter.hxx`'s `offset1`..`offset8`,
    /// each a 2-element `OffsetType`), and both ITK and SimpleITK only wrap
    /// this filter for 2-D images (`itkBinaryPruningImageFilter.wrap`'s
    /// `itk_wrap_image_filter(..., 2, 2)`).
    #[error("binary_pruning only supports 2-D images, got {0}-D")]
    UnsupportedPruningDimension(usize),

    /// `ShapeLabelMapFilter::PerimeterFromInterceptCount` has hand-tuned
    /// direction-weight overloads for 2-D and 3-D only
    /// (`itkShapeLabelMapFilter.hxx`'s `MapIntercept2Type` /
    /// `MapIntercept3Type` specializations), and SimpleITK only instantiates
    /// `LabelShapeStatisticsImageFilter` for those two dimensions.
    #[error("label_shape_statistics only supports 2-D and 3-D images, got {0}-D")]
    UnsupportedShapeDimension(usize),

    /// `ContourExtractor2DImageFilter` is `itkConceptMacro(DimensionShouldBe2,
    /// ...)`-constrained, and SimpleITK's `custom_register:
    /// factory.RegisterMemberFunctions<PixelIDTypeList, 2, 2>()` only
    /// instantiates it for 2-D images.
    #[error("contour_extractor_2d only supports 2-D images, got {0}-D")]
    UnsupportedContourExtractorDimension(usize),

    /// `ContourExtractor2DImageFilter::GenerateDataForLabels` searches for a
    /// label value that does not occur in the image, to use as the
    /// out-of-bounds constant, and throws
    /// `"Need at least one unused value in the space of labels"` when the
    /// image's labels exhaust the pixel type's whole value range.
    #[error(
        "contour_extractor_2d with label_contours needs at least one unused value in the {0:?} label space, but every representable value occurs in the image"
    )]
    ContourExtractorNoUnusedLabel(PixelId),

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

    /// `BSplineDecompositionImageFilter::SetPoles` throws "SplineOrder must be
    /// between 0 and 5. Requested spline order has not been implemented yet."
    /// Unlike [`FilterError::InvalidSplineOrder`], order 0 *is* valid here (it
    /// has no poles and the decomposition is the identity); only the upper
    /// bound is enforced, because ITK tabulates poles for orders 0 through 5
    /// only.
    #[error("spline order must be between 0 and 5, got {0}")]
    UnsupportedSplineOrder(u32),

    /// `LinearAnisotropicDiffusionLBRImageFilter::StencilFunctor::Stencil` is
    /// overloaded only on `Dispatch<2>` and `Dispatch<3>`, so the lattice-basis
    /// -reduction stencil exists in 2-D and 3-D only. In C++ any other
    /// dimension fails to compile; this port rejects it at run time.
    #[error("lattice-basis-reduction diffusion supports 2-D and 3-D images only, got {0}-D")]
    UnsupportedLbrDimension(usize),

    /// `LinearAnisotropicDiffusionLBRImageFilter::SetRatioToMaxStableTimeStep`
    /// throws "Ratio to max time step ... should be within ]0,1]".
    #[error("ratio to the maximum stable time step must lie in (0, 1], got {0}")]
    InvalidTimeStepRatio(f64),

    /// `LinearAnisotropicDiffusionLBRImageFilter::SetMaxNumberOfTimeSteps`
    /// throws "Max number of time steps must be positive". Reachable from
    /// SimpleITK because `MaxTimeStepsBetweenTensorUpdates` is a `uint8_t`
    /// that `AnisotropicDiffusionLBRImageFilter` stores unvalidated and only
    /// forwards at `Update` time.
    #[error("max time steps between tensor updates must be >= 1, got 0")]
    ZeroMaxTimeSteps,

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

    /// `SliceImageFilter::VerifyInputInformation` throws "Step size is zero"
    /// when any axis's `step` is `0` (`GenerateOutputInformation`'s size
    /// formula divides by `step`, and a stride of zero could never advance
    /// the source index).
    #[error("slice step must be non-zero along every axis, got {0:?}")]
    InvalidSliceStep(Vec<i32>),

    /// `GrayscaleConnectedOpeningImageFilter`/`GrayscaleConnectedClosingImageFilter`'s
    /// `Seed` must be a valid index into the image. ITK's own `GenerateData`
    /// dereferences `inputImage->GetPixel(m_Seed)` with no bounds check at
    /// all (an out-of-range `Seed` is undefined behavior in C++, silently
    /// reading whatever lies at the aliased offset); this port checks
    /// instead, since an out-of-range multi-index could otherwise alias a
    /// different in-bounds flat offset over this crate's linear pixel
    /// buffer rather than simply crash.
    #[error("seed index {seed:?} is out of bounds for an image of size {size:?}")]
    InvalidSeedIndex { seed: Vec<usize>, size: Vec<usize> },

    /// `FastMarchingUpwindGradientImageFilter::VerifyTargetReachedModeConditions`:
    /// "No target point set. Cannot set the target reached mode." Raised for
    /// every target-reached mode but `NoTargets`, i.e. whenever
    /// `number_of_targets` is non-zero and the target-point list is empty.
    /// `colliding_fronts` hits it through the `stop_on_targets` marches, whose
    /// targets are the *other* front's seeds.
    #[error("target reached mode requires at least one target point")]
    NoTargetPoints,

    /// `PatchBasedDenoisingImageFilter::Initialize` throws "Patch is larger
    /// than the entire image (in at least one dimension)" when the index
    /// `2 * PatchRadiusInVoxels` falls outside the largest possible region,
    /// i.e. when some axis is shorter than the patch diameter.
    #[error("patch diameter {diameter:?} does not fit in an image of size {size:?}")]
    PatchLargerThanImage {
        size: Vec<usize>,
        diameter: Vec<usize>,
    },

    /// `PatchBasedDenoisingImageFilter::EnforceConstraints` throws "Each image
    /// component must be nonconstant" — the kernel-bandwidth rescale factor is
    /// `100 / (max - min)`, which a constant image would divide by zero.
    #[error("patch-based denoising requires a nonconstant image; every pixel has the value {0}")]
    ConstantImage(f64),

    /// `PatchBasedDenoisingImageFilter::EnforceConstraints` throws when the
    /// POISSON or RICIAN noise model is selected and the image has a negative
    /// value.
    #[error("the POISSON and RICIAN noise models require a nonnegative image; the minimum is {0}")]
    NegativeIntensityForNoiseModel(f64),

    /// `PatchBasedDenoisingImageFilter::Initialize` throws "Gaussian kernel
    /// sigma ... must be larger than" `MinSigma`
    /// (`NumericTraits<double>::min() * 100`).
    #[error("kernel bandwidth sigma {0} must be larger than {1}")]
    KernelBandwidthSigmaTooSmall(f64, f64),

    /// `PatchBasedDenoisingImageFilter::InitializePatchWeightsSmoothDisc`
    /// throws "Center pixel's weight ... must be greater than 0.0" when the
    /// resampled smooth-disc mask has a nonpositive centre.
    #[error("the smooth-disc patch mask has a nonpositive center weight {0}")]
    PatchCenterWeightNotPositive(f64),

    /// `PatchBasedDenoisingImageFilter::ThreadedComputeSigmaUpdate`'s
    /// `itkAssertOrThrowMacro(probJointEntropy[ic] > 0.0, ...)`. Reached when
    /// `NumberOfSamplePatches` is zero, so the per-pixel probability is `0/0`.
    #[error("kernel bandwidth estimation sampled no patches; NumberOfSamplePatches must be >= 1")]
    NoPatchesSampled,

    /// SimpleITK's `PatchBasedDenoisingImageFilter` derives the sampler radius
    /// as `Math::Floor<unsigned int>(sqrt(SampleVariance) * 2.5)`, which is
    /// undefined for a negative variance.
    #[error("sample variance must be nonnegative, got {0}")]
    InvalidSampleVariance(f64),

    /// SimpleITK's `GetImageFromVectorImage` (`sitkImageConvert.hxx:38-42`)
    /// throws "Expected number of elements in vector image to be the same as
    /// the dimension!" -- every `ITKDisplacementField` filter reinterprets its
    /// `itk::VectorImage<T, N>` input as an `itk::Image<itk::Vector<T, N>, N>`,
    /// which is only sound when the run-time component count equals `N`.
    #[error(
        "a displacement field needs one component per dimension: got {components} components for a {dimension}-D image"
    )]
    DisplacementFieldComponentMismatch { components: usize, dimension: usize },

    /// `InverseDisplacementFieldImageFilter::PrepareKernelBaseSpline`
    /// (`itkInverseDisplacementFieldImageFilter.hxx:131-135`) computes the
    /// subsampled grid as `size[i] / m_SubsamplingFactor`, an integer division
    /// that is undefined behavior in C++ for a zero factor. SimpleITK exposes
    /// the setter without a guard, so this port rejects it here.
    #[error("subsampling factor must be >= 1, got {0}")]
    InvalidSubsamplingFactor(u32),

    /// `GaussianOperator::SetMaximumError` throws "Maximum Error Must be in the
    /// range [ 0.0 , 1.0 ]" when the value is not strictly inside `(0, 1)`
    /// (itkGaussianOperator.h:86-95 — the message says closed, the test is
    /// open).
    #[error("Gaussian maximum error must lie strictly inside (0, 1), got {0}")]
    GaussianMaximumErrorOutOfRange(f64),

    /// `MergeLabelMapFilter::MergeWithStrict` (`itkMergeLabelMapFilter.hxx:138-142`)
    /// throws when an input's label is already present in the output.
    #[error("label {label} from input {input} is already in use")]
    MergeLabelInUse { label: i64, input: usize },

    /// `MergeLabelMapFilter::MergeWithStrict` (`itkMergeLabelMapFilter.hxx:145-149`)
    /// throws when an input's label equals the output background value.
    #[error("label {label} from input {input} is output background value")]
    MergeLabelIsBackground { label: i64, input: usize },

    /// `MergeLabelMapFilter` was called with no inputs. `ProcessObject` refuses
    /// to run without input 0.
    #[error("this filter requires at least one input label map")]
    EmptyLabelMapList,

    /// `DICOMOrientImageFilter::ImageDimension` is `static_assert`ed to `3`
    /// (`itkDICOMOrientImageFilter.h:142`), and `DICOMOrientImageFilter.yaml`'s
    /// `custom_register` only instantiates the SimpleITK wrapper for 3-D
    /// images (`factory.RegisterMemberFunctions<PixelIDTypeList, 3>()`).
    #[error("dicom_orient only supports 3-D images, got {0}-D")]
    UnsupportedDicomOrientDimension(usize),

    /// `DICOMOrientImageFilter::VerifyPreconditions`
    /// (`itkDICOMOrientImageFilter.hxx:296-306`) throws
    /// "DesiredCoordinateOrientation is 'INVALID'." when the desired
    /// orientation string does not parse to one of the 48 valid 3-letter
    /// codes.
    #[error(
        "desired coordinate orientation '{0}' does not parse to a valid 3-letter orientation code"
    )]
    InvalidDesiredOrientation(String),

    /// `sitkSTLToITKDirection` (`sitkTemplateFunctions.h:187-207`), used by
    /// `GetOrientationFromDirectionCosines`: a non-empty direction vector
    /// must have exactly `3*3 == 9` elements.
    #[error("length of input ({0}) does not match matrix dimensions (3, 3)")]
    InvalidDirectionCosinesLength(usize),

    /// `HessianToObjectnessMeasureImageFilter::VerifyPreconditions`
    /// (`itkHessianToObjectnessMeasureImageFilter.hxx:210-217`) throws
    /// "ObjectDimension must be lower than ImageDimension." The composite
    /// `ObjectnessMeasureImageFilter` performs no check of its own; it
    /// forwards the setting and the inner filter throws at `Update` time.
    #[error(
        "object dimension {object_dimension} must be lower than image dimension {image_dimension}"
    )]
    InvalidObjectDimension {
        object_dimension: usize,
        image_dimension: usize,
    },

    /// [`crate::objectness::objectness_measure`] diagonalizes the Hessian with
    /// this crate's `linalg::symmetric_eigen`, which is written for matrices up
    /// to `3 x 3`. SimpleITK's default build instantiates `ObjectnessMeasure`
    /// for 2-D and 3-D images only, so no supported input reaches this error;
    /// a 1-D or 4-D image would.
    #[error("objectness_measure supports 2-D and 3-D images only, got {0}-D")]
    UnsupportedObjectnessDimension(usize),

    /// A core image error surfaced.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for filter results.
pub type Result<T> = std::result::Result<T, FilterError>;
