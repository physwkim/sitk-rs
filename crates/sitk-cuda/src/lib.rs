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
mod ops;

pub use backend::{Backend, backend};
pub use buffer::DeviceBuffer;
pub use error::CudaError;
pub use ops::rescale_intensity::{rescale_intensity_gpu, try_rescale_intensity};

/// Wall-clock split of one GPU op, in milliseconds.
///
/// Measured with a stream synchronize between phases, so `kernel_ms` excludes
/// transfer time rather than hiding behind it. For a per-pixel op the transfer
/// terms are expected to dominate; reporting them separately is the point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTimings {
    /// Host-to-device copy of the input buffer.
    pub h2d_ms: f64,
    /// All kernel launches (for `rescale_intensity`: the min/max reduction and
    /// the map), synchronized.
    pub kernel_ms: f64,
    /// Device-to-host copy of the output buffer, **including** the first-touch
    /// page faults on the freshly allocated output `Vec` the op must return.
    /// On this machine those faults are ~6× the copy itself for a 64 MiB
    /// output (~35 ms vs ~5 ms); see [`DeviceBuffer::copy_to_host`] for the
    /// pooled-destination path that avoids them.
    pub d2h_ms: f64,
}

impl GpuTimings {
    /// Sum of the three phases.
    pub fn total_ms(&self) -> f64 {
        self.h2d_ms + self.kernel_ms + self.d2h_ms
    }
}
