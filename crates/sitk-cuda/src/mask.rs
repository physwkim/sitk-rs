//! [`DeviceMask`] — a binary predicate over a grid, resident on the device.
//!
//! # Why this is not a field on `DeviceImage`
//!
//! A mask is not a property of an image; it is an argument to the *metric*. If a
//! `DeviceImage` carried an optional mask, every op in this crate would have to
//! answer a question it has no business answering: does [`rescale_intensity`] carry
//! the mask through? Does [`recursive_gaussian`] *blur* it? Does [`shrink`]
//! subsample it, and does [`resample_linear`] interpolate it into fractional values
//! that are no longer a predicate at all? Each answer is a policy, and a field whose
//! meaning depends on which op last touched the image is the dual-meaning cell that
//! produces a new edge case every review round.
//!
//! So a mask is its own type, consumed at the one place that has a use for it —
//! [`ResidentMetric`](crate::ResidentMetric) — and an unmasked run allocates nothing
//! and says so in its types (`Option<&DeviceMask>` is `None`).
//!
//! [`rescale_intensity`]: crate::rescale_intensity
//! [`recursive_gaussian`]: crate::recursive_gaussian
//! [`shrink`]: crate::shrink
//! [`resample_linear`]: crate::resample_linear

use cudarc::driver::{LaunchConfig, PushKernelArg};
use sitk_core::Image;

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;
use crate::image::{DeviceImage, Geometry};

/// Threads per block for the two elementwise mask kernels.
const BLOCK: u32 = 256;

/// The two operations a mask needs on the device, and nothing else.
///
/// `threshold_nonzero` is the host's `v != 0.0` (`FixedSamples::from_image_with`),
/// evaluated on the `f32` volume a device resample produced. `(double)x != 0.0` and
/// `x != 0.0f` decide the same way for every `f32` — including `-0.0` (not inside)
/// and `NaN` (inside, in both) — so widening first would be a longer way to the same
/// answer.
///
/// `mask_and` is `intersect_masks`: `x != 0 && y != 0`, elementwise.
const MASK_OPS: &str = r#"
extern "C" __global__ void threshold_nonzero(
    const float* __restrict__ x,
    unsigned char* __restrict__ y,
    const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (unsigned char)(x[i] != 0.0f);
}

extern "C" __global__ void mask_and(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    unsigned char* __restrict__ y,
    const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (unsigned char)(a[i] != 0 && b[i] != 0);
}
"#;

fn launch_config(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A binary mask on a grid, one byte per voxel: `1` = inside, `0` = dropped.
///
/// One byte, not one bit: a bit-packed mask would need a shift and an `and` per
/// sample in the metric's inner loop to save 15 MB at 256³, against a volume that is
/// already spending 201 MB. The byte load coalesces; the bit twiddle would not pay.
///
/// The convention is the host's: **any nonzero voxel is inside**
/// (`FixedSamples::from_image_with` — "a binary image on the same grid as `fixed`;
/// any nonzero value is inside"), so a mask that arrived as `f32` 0.0/1.0, as `UInt8`
/// 0/255, or as anything else means the same thing here as it does there.
pub struct DeviceMask {
    buf: DeviceBuffer<u8>,
    geom: Geometry,
}

impl DeviceMask {
    /// Upload a host mask image, thresholding **nonzero → inside** exactly as the
    /// host metric does.
    ///
    /// Errors with [`CudaError::UnsupportedPixelType`] for a pixel type that has no
    /// device path, and [`CudaError::DegenerateInput`] for an empty image.
    pub fn upload(mask: &Image) -> Result<Self, CudaError> {
        let backend = backend()?;
        let values = mask
            .to_f64_vec()
            .map_err(|_| CudaError::UnsupportedPixelType(mask.pixel_id()))?;
        if values.is_empty() {
            return Err(CudaError::DegenerateInput);
        }
        let bytes: Vec<u8> = values.iter().map(|&v| u8::from(v != 0.0)).collect();
        Ok(Self {
            buf: DeviceBuffer::from_host(backend, &bytes)?,
            geom: Geometry::of(mask),
        })
    }

    /// Threshold a **device** image into a mask, `nonzero → inside` — the same rule
    /// [`upload`](Self::upload) applies on the host, applied where the image already
    /// is.
    ///
    /// This is how a mask reaches a pyramid level: the host resamples its masks onto
    /// the level's grid with nearest-neighbour interpolation and re-reads them as
    /// predicates (`prepare_level`), and the device does the same with
    /// [`resample_nearest`](crate::resample_nearest) — whose output is a
    /// [`DeviceImage`] of 0.0/1.0 that this turns back into a predicate without
    /// touching the bus.
    pub fn from_device_image(img: &DeviceImage) -> Result<Self, CudaError> {
        let backend: &Backend = backend()?;
        let n = img.len();
        if n == 0 {
            return Err(CudaError::DegenerateInput);
        }
        let mut buf = DeviceBuffer::<u8>::zeros(backend, n)?;
        let n_i64 = n as i64;

        let f = backend.function(MASK_OPS, "threshold_nonzero")?;
        let mut launch = backend.stream().launch_builder(&f);
        launch
            .arg(img.buffer().device())
            .arg(buf.device_mut())
            .arg(&n_i64);
        // SAFETY: three parameters, three arguments, matching in order and type; both
        // buffers hold `n` elements and the kernel guards every access on `i < n`.
        unsafe { launch.launch(launch_config(n))? };
        backend.synchronize()?;

        Ok(Self {
            buf,
            geom: img.geometry().clone(),
        })
    }

    /// The elementwise **and** of two masks on the same grid — the device form of
    /// the host's `intersect_masks`, which folds the in-buffer predicate and the
    /// user's fixed mask into the one mask a level carries.
    ///
    /// Refuses two masks on different grids with [`CudaError::DegenerateInput`]:
    /// masks are indexed by flat grid index, so intersecting across grids would gate
    /// voxels that are not the same voxels.
    pub fn intersect(&self, other: &Self) -> Result<Self, CudaError> {
        if self.geom != other.geom {
            return Err(CudaError::DegenerateInput);
        }
        let backend: &Backend = backend()?;
        let n = self.buf.len();
        let mut buf = DeviceBuffer::<u8>::zeros(backend, n)?;
        let n_i64 = n as i64;

        let f = backend.function(MASK_OPS, "mask_and")?;
        let mut launch = backend.stream().launch_builder(&f);
        launch
            .arg(self.buf.device())
            .arg(other.buf.device())
            .arg(buf.device_mut())
            .arg(&n_i64);
        // SAFETY: four parameters, four arguments, matching in order and type; all
        // three buffers hold `n` elements (the geometries are equal, checked above)
        // and the kernel guards every access on `i < n`.
        unsafe { launch.launch(launch_config(n))? };
        backend.synchronize()?;

        Ok(Self {
            buf,
            geom: self.geom.clone(),
        })
    }

    /// Bring the mask back to the host as a `UInt8` image of 0/1 carrying this grid's
    /// geometry.
    ///
    /// This is a **test** road, not a pipeline one: the level's mask never needs to
    /// come down in a run. It exists so that the device level mask can be compared
    /// **byte for byte** against the host's, which is the only pin that catches a
    /// mask that is built but silently wrong — a wrong mask does not produce a wrong
    /// number, it produces a right number over the wrong sample set.
    pub fn to_host(&self) -> Result<Image, CudaError> {
        let backend = backend()?;
        let bytes = self.buf.to_host(backend)?;
        let mut img = Image::from_vec(&self.geom.size, bytes)?;
        img.set_spacing(&self.geom.spacing)?;
        img.set_origin(&self.geom.origin)?;
        img.set_direction(&self.geom.direction)?;
        Ok(img)
    }

    /// The grid this mask gates.
    pub fn geometry(&self) -> &Geometry {
        &self.geom
    }

    /// Voxels in the mask.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the mask has no voxels — never true for a mask built by
    /// [`upload`](Self::upload), which refuses an empty image.
    pub fn is_empty(&self) -> bool {
        self.buf.len() == 0
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer<u8> {
        &self.buf
    }
}
