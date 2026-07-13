use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::backend::Backend;
use crate::error::CudaError;

/// A typed device allocation, owned and freed by RAII.
///
/// Thin on purpose: it exists so that allocation, H2D, and D2H all go through
/// one place that speaks [`CudaError`] and one stream, not so that it can
/// re-implement `CudaSlice`. Kernel launches borrow the inner slice through
/// [`DeviceBuffer::device`] / [`DeviceBuffer::device_mut`].
pub struct DeviceBuffer<T> {
    slice: CudaSlice<T>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Allocate on the device and copy `host` up (H2D).
    pub fn from_host(backend: &Backend, host: &[T]) -> Result<Self, CudaError> {
        Ok(Self {
            slice: backend.stream().clone_htod(host)?,
        })
    }

    /// Copy `src` into this already-allocated device buffer (H2D), which must be
    /// at least as long.
    ///
    /// The counterpart to [`copy_to_host`](Self::copy_to_host): no allocation, so
    /// a caller that pushes to the same buffer every iteration — the
    /// resident-volume registration loop pushing a transform each step — pays one
    /// `cuMemAlloc` for the run rather than one per iteration.
    pub fn copy_from_host(&mut self, backend: &Backend, src: &[T]) -> Result<(), CudaError> {
        Ok(backend.stream().memcpy_htod(src, &mut self.slice)?)
    }

    /// Copy the device buffer back down into a fresh `Vec` (D2H).
    ///
    /// **Slow, and kept only for callers with nowhere to put the result.** The
    /// `Vec` is freshly mapped, so the DMA faults in every one of its pages, and
    /// that fault cost — not the PCIe link — dominates: a 512 MiB D2H this way
    /// measures 481 ms (1.1 GB/s) against a link that runs at ~12–13 GB/s.
    ///
    /// Prefer, in order: [`copy_to_host`](Self::copy_to_host) into a destination
    /// reused across calls; or, when the result must be owned, a destination from
    /// [`sitk_core::alloc::resident_vec`], which is ~6× faster than this.
    pub fn to_host(&self, backend: &Backend) -> Result<Vec<T>, CudaError> {
        Ok(backend.stream().clone_dtoh(&self.slice)?)
    }

    /// Copy the device buffer into an existing host slice (D2H), which must be
    /// at least as long. A destination that is already resident — reused across
    /// calls, or from [`sitk_core::alloc::resident_vec`] — avoids the fault cost
    /// described on [`DeviceBuffer::to_host`] and runs at link speed.
    pub fn copy_to_host(&self, backend: &Backend, dst: &mut [T]) -> Result<(), CudaError> {
        Ok(backend.stream().memcpy_dtoh(&self.slice, dst)?)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// `true` if the allocation holds no elements.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Borrow the allocation as a kernel argument.
    pub fn device(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Borrow the allocation as a writable kernel argument.
    pub fn device_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceBuffer<T> {
    /// Allocate `len` zeroed elements on the device.
    pub fn zeros(backend: &Backend, len: usize) -> Result<Self, CudaError> {
        Ok(Self {
            slice: backend.stream().alloc_zeros::<T>(len)?,
        })
    }
}
