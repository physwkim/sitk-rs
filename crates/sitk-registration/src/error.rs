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
}

/// Registration result alias.
pub type Result<T> = std::result::Result<T, RegistrationError>;
