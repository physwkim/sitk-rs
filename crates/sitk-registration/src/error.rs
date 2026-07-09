//! Error type for registration.

use sitk_core::PixelId;

/// Errors returned by the registration crate.
#[derive(Debug, thiserror::Error)]
pub enum RegistrationError {
    /// Fixed and moving images differ in spatial dimension.
    #[error("dimension mismatch: fixed is {fixed}-D, moving is {moving}-D")]
    DimensionMismatch { fixed: usize, moving: usize },

    /// A transform's dimension does not match the images'.
    #[error("transform is {transform}-D but images are {image}-D")]
    TransformDimensionMismatch { transform: usize, image: usize },

    /// An image's direction cosine matrix is singular and cannot be inverted.
    #[error("moving image direction matrix is singular")]
    SingularDirection,

    /// Optimizer scales were supplied with the wrong length.
    #[error("optimizer scales length {got} != number of parameters {expected}")]
    ScalesLength { got: usize, expected: usize },

    /// No sampled fixed point mapped inside the moving image, so the metric is
    /// undefined. Usually means the initial transform is far off.
    #[error("no valid sample points: every fixed point mapped outside the moving image")]
    NoValidSamples,

    /// The multi-resolution shrink-factor and smoothing-sigma schedules have
    /// different lengths (they must have one entry per level).
    #[error(
        "multi-resolution schedule length mismatch: {shrink} shrink levels vs {sigma} sigma levels"
    )]
    PyramidScheduleLength { shrink: usize, sigma: usize },

    /// Wrapped core error (e.g. constructing the output image).
    #[error(transparent)]
    Core(#[from] sitk_core::Error),

    /// Wrapped filter error from per-level shrinking/smoothing.
    #[error(transparent)]
    Filter(#[from] sitk_filters::FilterError),

    /// Wrapped transform error from per-level resampling of the fixed image.
    #[error(transparent)]
    Transform(#[from] sitk_transform::TransformError),

    /// A pixel type that this path does not handle.
    #[error("unsupported pixel type {0:?}")]
    UnsupportedPixelType(PixelId),

    /// `CenteredTransformInitializer` in `Moments` mode cannot form a center of
    /// gravity because an image's total intensity mass is zero (matches ITK's
    /// `ImageMomentsCalculator`, which aborts to avoid dividing by zero).
    #[error("center of gravity is undefined: {which} image has zero total intensity mass")]
    ZeroTotalMass { which: &'static str },

    /// The Mattes MI metric was asked for too few histogram bins. The cubic
    /// B-spline Parzen window pads the histogram by two bins on each axis end, so
    /// at least `2·padding + 1 = 5` bins are required for a non-empty central
    /// range (matches `itk::MattesMutualInformationImageToImageMetricv4`'s
    /// `numberOfHistogramBins − 2·padding` bin-size divisor).
    #[error("Mattes MI needs at least 5 histogram bins, got {bins}")]
    TooFewHistogramBins { bins: usize },

    /// The Mattes MI metric cannot be built because an image is constant: its
    /// intensity range is zero, so the marginal entropy is zero and mutual
    /// information is undefined (matches ITK, which throws in this case).
    #[error("Mattes mutual information is undefined: {which} image has a constant intensity value")]
    ConstantIntensity { which: &'static str },

    /// `LandmarkBasedTransformInitializer`: the fixed and moving landmark
    /// containers have different lengths (matches ITK's
    /// `itkExceptionStringMacro("Different number of fixed and moving
    /// landmarks")`).
    #[error("landmark count mismatch: {fixed} fixed vs {moving} moving")]
    LandmarkCountMismatch { fixed: usize, moving: usize },

    /// `LandmarkBasedTransformInitializer`: fewer landmarks were supplied
    /// than the requested transform needs to be uniquely determined — a
    /// rigid transform needs at least `dimension` landmarks to compute a
    /// rotation, an affine transform needs at least `dimension + 1`. Unlike
    /// ITK, which silently falls back to an identity rotation when a rigid
    /// transform is under-supplied, this port rejects the input.
    #[error("insufficient landmarks: got {got}, need at least {required}")]
    InsufficientLandmarks { got: usize, required: usize },

    /// `LandmarkBasedTransformInitializer`: the landmark weight vector's
    /// length does not match the number of landmarks (matches ITK's
    /// `itkExceptionStringMacro("size mismatch between number of landmarks
    /// pairs and weights")`, checked only for the affine path, which is the
    /// only path that reads landmark weights).
    #[error("landmark weight length {got} != number of landmarks {expected}")]
    LandmarkWeightLength { got: usize, expected: usize },

    /// `LandmarkBasedTransformInitializer`: the affine least-squares normal
    /// equations matrix is singular (e.g. collinear or coplanar landmarks),
    /// so no unique affine transform fits the landmarks.
    #[error("degenerate landmark configuration: normal-equations matrix is singular")]
    DegenerateLandmarks,
}

/// Registration result alias.
pub type Result<T> = std::result::Result<T, RegistrationError>;
