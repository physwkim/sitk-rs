use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;
use sitk_core::PixelId;
use thiserror::Error;

/// Every way a GPU op can decline to produce a result.
///
/// Each variant is a *fallback* condition, never a fatal one: the fallback API
/// (`try_*`) maps all of them to `None` and the caller runs the CPU path.
#[derive(Debug, Error)]
pub enum CudaError {
    /// No CUDA driver could be loaded, or the driver reports no usable device.
    /// This is the state on a machine with no GPU, and it is not an error the
    /// user should ever see — it means "use the CPU".
    #[error("no usable CUDA device: {0}")]
    NoDevice(String),

    /// NVRTC could not compile a kernel.
    #[error("NVRTC compile failed: {0}")]
    Nvrtc(#[from] CompileError),

    /// The CUDA driver returned an error during allocation, transfer, module
    /// load, or launch.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    /// The op has no GPU kernel for this pixel type. The CPU path supports
    /// every scalar type; the GPU path currently supports `Float32`.
    #[error("no CUDA kernel for pixel type {0:?}")]
    UnsupportedPixelType(PixelId),

    /// The op has a kernel for this pixel type but not for this image's *shape*:
    /// the pyramid ops are 3-D, a shrink factor must be at least 1, and the
    /// recursive Gaussian's fourth-order recursion needs at least four voxels on
    /// every axis it smooths (the same requirement the CPU filter states).
    #[error("no CUDA kernel for this geometry: {0}")]
    UnsupportedGeometry(String),

    /// The op's input is degenerate on its own terms (for
    /// `rescale_intensity`: an empty image, or one whose min equals its max).
    /// The CPU path defines the error the user sees, so the GPU path declines
    /// and lets it.
    #[error("degenerate input; deferring to the CPU implementation")]
    DegenerateInput,

    /// A fixed mask gates samples by their **grid** index, so it means nothing
    /// against an explicit, host-selected point list — where the samples are already
    /// a subset in an arbitrary order and the same index refers to a different voxel.
    /// The two are refused together rather than silently gating the wrong samples.
    #[error(
        "a fixed mask is indexed by the fixed grid, so it cannot be combined with an \
         explicit fixed-point list"
    )]
    MaskedExplicitPoints,

    /// The image is a vector/complex image, or its buffer does not match its
    /// declared pixel type — `sitk_core` already names this precisely, so
    /// carry its message rather than reclassifying it.
    #[error("image is not a scalar image of the expected type: {0}")]
    NotScalar(#[from] sitk_core::Error),
}
