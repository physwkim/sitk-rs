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
    ///
    /// The rule is not "no mask with a selected sample set" but *a fixed mask requires
    /// a sample set that knows its grid index*: an index list
    /// ([`FixedPoints::Indices`](crate::FixedPoints::Indices)) carries that index and
    /// is masked correctly. A bare point list has thrown it away, and this is where it
    /// says so.
    #[error(
        "a fixed mask is indexed by the fixed grid, so it cannot be combined with an \
         explicit fixed-point list"
    )]
    MaskedExplicitPoints,

    /// A sample's index does not name a voxel of the fixed grid. Checked on the host
    /// before launch: the kernel would read outside the volume, and a clamped read
    /// would silently sample the wrong voxel.
    #[error("fixed sample index {index} is outside the fixed grid ({voxels} voxels)")]
    SampleIndexOutOfGrid { index: i64, voxels: usize },

    /// The two passes of the correlation metric disagreed about how many samples are
    /// valid.
    ///
    /// They cannot, by construction: both run the same sampler over the same sample
    /// set under the same point map, so the same samples survive. If they ever do
    /// differ, the sample means were divided by one population while the moments were
    /// accumulated over another — a metric that is silently wrong rather than loudly
    /// broken. Raised instead of trusting the invariant.
    #[error(
        "the correlation passes disagree on the valid-sample count: {sums} in the sums \
         pass, {moments} in the moments pass"
    )]
    PassCountMismatch { sums: usize, moments: usize },

    /// The point map handed to a resident metric has no stages, or more than the
    /// device replays ([`MAX_STAGES`](crate::MAX_STAGES)).
    ///
    /// The device replays the host's stages in the host's order — that is what makes
    /// the continuous index bit-identical — so it cannot silently drop, fold or
    /// truncate them. An empty list is refused too: a zero-stage replay is the
    /// identity map, which is a *plausible* wrong answer rather than an obvious one.
    #[error("the point map has {stages} stages; the device replays 1..={max}")]
    PointMapStageCount { stages: usize, max: usize },

    /// The image is a vector/complex image, or its buffer does not match its
    /// declared pixel type — `sitk_core` already names this precisely, so
    /// carry its message rather than reclassifying it.
    #[error("image is not a scalar image of the expected type: {0}")]
    NotScalar(#[from] sitk_core::Error),
}
