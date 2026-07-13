//! Optional CUDA backend for sitk-rs.
//!
//! # Two API layers, one meaning each
//!
//! Every op here comes in two forms, and the split is deliberate — a single
//! function returning "error" for both "this GPU run failed" and "there is no
//! GPU" would make the caller guess which it got:
//!
//! - the **strict** form (`rescale_intensity_gpu`) returns
//!   [`Result<_, CudaError>`]: it says exactly what the GPU did, and is what
//!   tests and benchmarks call;
//! - the **fallback** form (`try_rescale_intensity`) returns [`Option`]:
//!   `None` means "the GPU produced no result, for *any* reason — no driver,
//!   no device, NVRTC failure, out of memory, unsupported pixel type" and the
//!   caller must run the CPU implementation. It never panics.
//!
//! `sitk-filters` calls only the fallback form, so no GPU condition can turn a
//! working CPU call into a failure.
//!
//! # Feature gate
//!
//! Everything below is behind the `cuda` feature, **default off**. With the
//! feature off this crate is an empty lib with no dependencies.

#![cfg(feature = "cuda")]

mod backend;
mod buffer;
mod error;
mod image;
mod mask;
mod ops;
mod pinned;

pub use backend::{Backend, backend};
pub use buffer::DeviceBuffer;
pub use error::CudaError;
pub use image::{DeviceImage, Geometry};
pub use mask::DeviceMask;
pub use ops::device::rescale_intensity;
pub use ops::gaussian::smooth_gaussian;
pub use ops::mean_squares::{DIM, FixedPoints, Moments, MovingGeometry, ResidentMetric};
pub use ops::pyramid::{recursive_gaussian, resample_linear, resample_nearest, shrink};
pub use ops::rescale_intensity::{
    rescale_intensity_gpu, rescale_intensity_gpu_into, rescale_intensity_resident,
    try_rescale_intensity,
};
pub use pinned::PinnedBuffer;

/// Wall-clock split of one GPU op, in milliseconds.
///
/// Measured with a stream synchronize between phases, so `kernel_ms` excludes
/// transfer time rather than hiding behind it. For a per-pixel op the transfer
/// terms dominate; reporting them separately is the point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTimings {
    /// Host-to-device copy of the input buffer.
    pub h2d_ms: f64,
    /// Preparing the output allocation so the D2H does not fault under the DMA
    /// (see [`sitk_core::alloc::resident_vec`]).
    ///
    /// This is a *host* cost, and it is broken out rather than folded into
    /// `d2h_ms` because folding it there is precisely the mistake that made a
    /// page-fault storm look like a slow PCIe link.
    pub alloc_ms: f64,
    /// All kernel launches (for `rescale_intensity`: the min/max reduction and
    /// the map), synchronized.
    pub kernel_ms: f64,
    /// Device-to-host copy of the output buffer into an already-resident
    /// destination — the DMA alone, at link speed.
    pub d2h_ms: f64,
}

impl GpuTimings {
    /// Sum of the four phases: the op's whole wall-clock cost.
    pub fn total_ms(&self) -> f64 {
        self.h2d_ms + self.alloc_ms + self.kernel_ms + self.d2h_ms
    }
}
