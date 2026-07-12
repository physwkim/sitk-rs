//! Page-locked host memory.
//!
//! # When this is the right tool, and when it is not
//!
//! Pinned memory buys two things: the DMA skips the driver's internal staging
//! copy (~12–13 GB/s instead of ~9.5), and the pages are resident by
//! construction, so nothing faults. Measured, 512 MiB:
//!
//! ```text
//! D2H -> pinned, reused destination      40.73 ms   13.18 GB/s
//! D2H -> pre-touched Vec, reused         43.36 ms   12.38 GB/s
//! D2H -> pinned, then copy to fresh Vec 816.68 ms    0.66 GB/s   <-- the trap
//! ```
//!
//! The trap is the one that matters: pinning does **not** help an op that must
//! return an owned `Vec`, because the pinned-to-`Vec` copy faults every page of
//! the fresh allocation and undoes the win. Nor does it help a *one-shot* H2D
//! from a `Vec` the caller already owns — staging that `Vec` into a pinned buffer
//! costs a full host-to-host copy (measured 99 ms total vs 56 ms for copying the
//! unpinned `Vec` directly).
//!
//! So this type earns its keep exactly where the buffer is **reused across
//! calls**: the resident-volume registration loop, where the fixed and moving
//! volumes are uploaded once and the per-iteration traffic is a few hundred
//! bytes. For a one-shot op's output, use [`crate::host::resident_vec`] instead.
//!
//! # Cacheable, not write-combined
//!
//! cudarc's own `alloc_pinned` hardcodes `CU_MEMHOSTALLOC_WRITECOMBINED`, which
//! is right for a buffer the CPU only ever *writes* (an H2D source) and wrong for
//! one the CPU *reads* (a D2H destination): WC memory is uncached, so host reads
//! of it crawl. This allocates with flags = 0 — ordinary cacheable page-locked
//! memory — so a `PinnedBuffer` is safe to read back on the host.

use std::marker::PhantomData;

use cudarc::driver::{DeviceRepr, result};

use crate::error::CudaError;

/// Page-locked, cacheable host memory: a D2H destination or H2D source that
/// neither faults nor stages.
///
/// Freed on drop. `T` must be [`DeviceRepr`] and zero-initializable, which every
/// pixel/scalar type this crate transfers is.
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: the allocation is plain host memory owned exclusively by this handle;
// there is no thread affinity in `cuMemAllocHost`'s contract, and `&mut` access
// is required to mutate it, so the usual `Send`/`Sync` reasoning for a `Box<[T]>`
// applies.
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T: DeviceRepr + Default + Copy> PinnedBuffer<T> {
    /// Allocate `len` zeroed elements of page-locked host memory.
    pub fn zeros(len: usize) -> Result<Self, CudaError> {
        let bytes = len * std::mem::size_of::<T>();
        // SAFETY: `malloc_host` returns `bytes` of page-locked host memory, or an
        // error. Flags = 0 selects cacheable (not write-combined) memory — see
        // the module docs for why that is the deliberate choice here.
        let ptr = unsafe { result::malloc_host(bytes, 0) }? as *mut T;
        assert!(!ptr.is_null() && ptr.is_aligned());
        let mut buf = Self {
            ptr,
            len,
            _marker: PhantomData,
        };
        buf.as_mut_slice().fill(T::default());
        Ok(buf)
    }

    /// The buffer as a host slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` is a live, aligned allocation of `len` initialized `T`s
        // (filled at construction), and `&self` bars concurrent mutation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The buffer as a mutable host slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as `as_slice`, and `&mut self` grants exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `malloc_host` in the constructor and is freed
        // exactly once, here. A failure to free is not actionable during drop.
        let _ = unsafe { result::free_host(self.ptr as _) };
    }
}
