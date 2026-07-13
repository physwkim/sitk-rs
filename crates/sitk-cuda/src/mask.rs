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

use sitk_core::Image;

use crate::backend::backend;
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;
use crate::image::Geometry;

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
