use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;
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

    /// The op's input is degenerate on its own terms: an empty sample set, a
    /// moving image that is not 3-D, or geometry the kernel cannot represent.
    /// The CPU path defines the error the user sees, so the GPU path declines
    /// and lets it.
    #[error("degenerate input; deferring to the CPU implementation")]
    DegenerateInput,
}
