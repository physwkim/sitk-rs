//! [`DeviceImage`] — an image that **lives on the device** across calls.
//!
//! # Why this type exists
//!
//! The port's GPU filters were `fn(&Image) -> Image`, which pays H2D + kernel +
//! D2H on *every* call. At 256³ an `f32` volume is 67 MB each way and the link
//! runs at ~13 GB/s, so a round trip is ~17 ms — against a `rescale_intensity`
//! kernel that measures **1.06 ms**. That API cannot win, and the "GPU is slower
//! than the CPU" verdict it produced was a statement about the bus, not about the
//! device.
//!
//! `DeviceImage` removes the round trip by removing the host from the middle of
//! the pipeline. Three properties, and they are the design:
//!
//! 1. **The bus crossing is a named call.** [`DeviceImage::upload`] and
//!    [`DeviceImage::to_host`] are the only two functions in this crate that move
//!    a volume across PCIe. No op does it behind the caller's back.
//! 2. **An op's signature cannot express a round trip.** A device op takes
//!    `&DeviceImage` and returns `DeviceImage`; there is no host buffer in the
//!    type for it to bounce through.
//! 3. **The CPU fallback lives at the pipeline boundary, not inside each op.** If
//!    [`upload`](DeviceImage::upload) fails — no driver, no device, out of memory,
//!    unsupported pixel type — the caller runs the host chain, once, for the whole
//!    pipeline. There is no per-call "did the GPU take it?" branch anywhere below
//!    this line, because a hidden per-op dispatch is exactly what made the bus cost
//!    invisible in the first place.
//!
//! # `f32` only, and it says so
//!
//! The device type is `Float32`. [`upload`](DeviceImage::upload) of any other
//! pixel type fails with [`CudaError::UnsupportedPixelType`], **naming the type** —
//! it does not silently widen it, and it does not silently decline to the CPU. A
//! caller with a `UInt16` CT casts on the host first (`sitk_filters::cast`), which
//! is one visible pass, rather than having a conversion happen inside a function
//! whose name does not mention it.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use sitk_core::{Image, PixelId};

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;

/// Threads per block for the widen kernel.
const BLOCK: u32 = 256;

const WIDEN: &str = r#"
extern "C" __global__ void widen_f32_f64(
    const float* __restrict__ x,
    double* __restrict__ y,
    const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (double)x[i];
}
"#;

/// Identity of a device buffer, so a metric can tell "same volume, next iteration"
/// from "new volume, upload it". Monotonic and process-wide.
fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// An image's geometry — everything about an [`Image`] except its voxels.
///
/// Tiny (a few dozen numbers), so it stays on the host: a device op needs the
/// spacing and direction to *compute* with, and the host needs them to rebuild an
/// [`Image`] on the way back down.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub size: Vec<usize>,
    pub spacing: Vec<f64>,
    pub origin: Vec<f64>,
    /// Row-major `dim × dim`.
    pub direction: Vec<f64>,
}

impl Geometry {
    /// The geometry of `img`, copied.
    pub fn of(img: &Image) -> Self {
        Self {
            size: img.size().to_vec(),
            spacing: img.spacing().to_vec(),
            origin: img.origin().to_vec(),
            direction: img.direction().to_vec(),
        }
    }

    /// Voxel count.
    pub fn len(&self) -> usize {
        self.size.iter().product()
    }

    /// `true` if the geometry describes no voxels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Spatial dimension.
    pub fn dimension(&self) -> usize {
        self.size.len()
    }
}

/// A `Float32` scalar image resident in device memory.
///
/// Constructed by [`upload`](Self::upload), consumed and produced by the device
/// ops, and brought back to the host exactly once, by [`to_host`](Self::to_host).
/// See the [module docs](self) for why those are the only two crossings.
pub struct DeviceImage {
    buf: DeviceBuffer<f32>,
    geom: Geometry,
    id: u64,
}

impl DeviceImage {
    /// Copy a host image to the device (**the H2D**, and one of only two bus
    /// crossings in this crate).
    ///
    /// `img` must be a `Float32` scalar image. Any other pixel type is refused
    /// with [`CudaError::UnsupportedPixelType`] carrying the type that was
    /// offered — the device path never widens a volume behind the caller's back,
    /// and never quietly decides to be a CPU path.
    pub fn upload(img: &Image) -> Result<Self, CudaError> {
        if img.pixel_id() != PixelId::Float32 {
            return Err(CudaError::UnsupportedPixelType(img.pixel_id()));
        }
        let host = img.scalar_slice::<f32>()?;
        if host.is_empty() {
            return Err(CudaError::DegenerateInput);
        }
        let backend = backend()?;
        Ok(Self {
            buf: DeviceBuffer::from_host(backend, host)?,
            geom: Geometry::of(img),
            id: next_id(),
        })
    }

    /// Copy the volume back to the host (**the D2H**, the other bus crossing) as a
    /// `Float32` image carrying this image's geometry.
    ///
    /// The destination comes from [`sitk_core::alloc::resident_vec`], so the DMA
    /// lands on mapped pages rather than faulting in every one of them under
    /// itself — the difference between 7 ms and 61 ms at 256³.
    pub fn to_host(&self) -> Result<Image, CudaError> {
        let backend = backend()?;
        let mut host = sitk_core::alloc::resident_vec::<f32>(self.buf.len());
        self.buf.copy_to_host(backend, &mut host)?;
        backend.synchronize()?;

        let mut img = Image::from_vec(&self.geom.size, host)?;
        img.set_spacing(&self.geom.spacing)?;
        img.set_origin(&self.geom.origin)?;
        img.set_direction(&self.geom.direction)?;
        Ok(img)
    }

    /// An uninitialized device image with the same geometry as `self` — the
    /// destination an op writes into. Allocates on the device only; nothing
    /// crosses the bus.
    pub(crate) fn like(&self) -> Result<Self, CudaError> {
        let backend = backend()?;
        Ok(Self {
            buf: DeviceBuffer::zeros(backend, self.buf.len())?,
            geom: self.geom.clone(),
            id: next_id(),
        })
    }

    /// This image's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geom
    }

    /// Voxel count.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// `true` if the image holds no voxels. Never true for an image that came
    /// from [`upload`](Self::upload), which refuses an empty one.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Identity of this device allocation: distinct for every image ever built,
    /// so a resident metric can tell "the same volume, next iteration" from "a new
    /// volume, upload it".
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The device buffer.
    pub(crate) fn buffer(&self) -> &DeviceBuffer<f32> {
        &self.buf
    }

    /// The device buffer, writable.
    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.buf
    }

    /// A `f64` copy of the voxels, **made on the device**: device-to-device, no
    /// bus traffic.
    ///
    /// The mean-squares kernel reduces in `double` (a metric summed over 16.7
    /// million samples in `float` would lose the low bits of the sum), so the
    /// resident metric needs the volume as `f64`. Widening on the device costs one
    /// pass over device memory — ~1 ms at 256³ — where widening on the host would
    /// cost a 134 MB H2D, which is the transfer this whole type exists to delete.
    ///
    /// `(double)x` is exact for every `f32`, so this changes no value: the metric
    /// sees the same numbers the host path uploaded.
    pub(crate) fn widen_f64(&self) -> Result<DeviceBuffer<f64>, CudaError> {
        let backend: &Backend = backend()?;
        let n = self.buf.len();
        let mut out = DeviceBuffer::<f64>::zeros(backend, n)?;

        let f = backend.function(WIDEN, "widen_f32_f64")?;
        let n_i64 = n as i64;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = backend.stream().launch_builder(&f);
        launch
            .arg(self.buf.device())
            .arg(out.device_mut())
            .arg(&n_i64);
        // SAFETY: the kernel's three parameters match the three arguments pushed
        // above in order and type (`const float*`, `double*`, `long long`); both
        // buffers hold `n` elements and every access is guarded by `i < n`.
        unsafe { launch.launch(cfg)? };
        backend.synchronize()?;
        Ok(out)
    }
}
