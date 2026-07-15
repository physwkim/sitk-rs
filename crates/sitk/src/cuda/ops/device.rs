//! Device-resident ops: `&DeviceImage -> DeviceImage`, nothing crossing the bus.
//!
//! The signature is the contract. An op here cannot perform a round trip, because
//! there is no host buffer in its type to bounce through — the volume it reads is
//! already on the device and the volume it returns stays there. A pipeline of `k`
//! such ops pays the crossing **once**, at [`DeviceImage::upload`] and
//! [`DeviceImage::to_host`], instead of `k` times.
//!
//! There is no fallback branch in this module. An op returns
//! [`Result<_, CudaError>`] and names its failure; the *caller* decides, once, at
//! the top of the pipeline, whether to run the host chain instead.

use crate::cuda::backend::backend;
use crate::cuda::error::CudaError;
use crate::cuda::image::DeviceImage;
use crate::cuda::ops::rescale_intensity::rescale_intensity_resident;

/// `rescale_intensity` on a resident volume: linearly map `[min, max]` onto
/// `[output_min, output_max]`, reading and writing device memory only.
///
/// The same two kernels the host-side [`crate::cuda::rescale_intensity_gpu`] runs — the
/// exact `min`/`max` reduction and the map — with the H2D, the D2H, and the host
/// allocation all absent. At 256³ that is 1.06 ms of kernel against 17.42 ms for
/// the CPU at 96 threads, where the one-shot form measured 30.42 ms because it
/// spent 16.98 ms of it on the bus.
///
/// Declines a degenerate volume ([`CudaError::DegenerateInput`]: empty, or min
/// equal to max) exactly as the host-side form does.
pub fn rescale_intensity(
    src: &DeviceImage,
    output_min: f64,
    output_max: f64,
) -> Result<DeviceImage, CudaError> {
    let backend = backend()?;
    let mut dst = src.like()?;
    rescale_intensity_resident(
        backend,
        src.buffer(),
        dst.buffer_mut(),
        output_min,
        output_max,
    )?;
    Ok(dst)
}
