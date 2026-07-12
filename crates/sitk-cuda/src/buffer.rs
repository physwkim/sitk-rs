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

    /// Copy the device buffer back down into a fresh `Vec` (D2H).
    ///
    /// The `Vec` is freshly mapped, so the DMA write faults in every one of its
    /// pages: on this machine a 64 MiB D2H costs ~35 ms this way versus ~5 ms
    /// into an already-resident destination. The page-fault cost is the
    /// allocation's, not the PCIe link's. Callers that can reuse a destination
    /// across calls want [`DeviceBuffer::copy_to_host`].
    pub fn to_host(&self, backend: &Backend) -> Result<Vec<T>, CudaError> {
        Ok(backend.stream().clone_dtoh(&self.slice)?)
    }

    /// Copy the device buffer into an existing host slice (D2H), which must be
    /// the same length. Reusing a destination across calls avoids the
    /// first-touch page-fault cost described on [`DeviceBuffer::to_host`].
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
