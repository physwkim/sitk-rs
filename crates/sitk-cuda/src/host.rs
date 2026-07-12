//! Host-side output allocation.
//!
//! # The defect this exists to close
//!
//! A GPU op must return an owned buffer, so it allocates a fresh `Vec` and lets
//! the D2H DMA write into it. Those pages are freshly mapped and not resident,
//! so the DMA faults in every one of them — and the fault, not the PCIe link, is
//! what dominates. Measured end-to-end through `rescale_intensity_gpu` at 512³
//! (512 MiB each way), median of 3 warm runs:
//!
//! ```text
//! D2H into a fresh Vec (what this crate used to do)   476 ms    1.1 GB/s
//! D2H into a resident destination                      41 ms   13.0 GB/s
//! ```
//!
//! The link runs at ~13 GB/s. The 1.1 GB/s figure was never PCIe; it was 131,072
//! minor page faults being billed to the bus.
//!
//! # What actually helps, measured in the op rather than in a microbenchmark
//!
//! Cost of making that fresh 512 MiB `Vec` resident:
//!
//! ```text
//! 1 thread,  madvise          357 ms
//! 16 threads, no madvise      185 ms
//! 16 threads, madvise         109 ms
//! ```
//!
//! Both terms are load-bearing and neither is dramatic. A microbenchmark will
//! flatter both: allocating and freeing the same buffer in a loop gets a *warm*
//! mapping back from the allocator and reports ~30 ms, which is not a cost any
//! real caller pays. The numbers above are from the op, holding a live input
//! volume, which is the state that matters.
//!
//! Faults do not parallelize well (the kernel must zero every anonymous page and
//! serializes on the mm lock), which is why 16 threads is the plateau and 96 is
//! worse than 16.
//!
//! # Why not pinned memory here
//!
//! Pinned host memory is the textbook answer and it is the *wrong* one for an op
//! that must hand back an owned `Vec`: the DMA into a pinned buffer is fast
//! (13.2 GB/s), but copying that buffer into the freshly allocated `Vec` the
//! caller gets faults every page anyway — measured 817 ms at 512 MiB, worse than
//! doing nothing at all. Pinned buffers pay off only when the destination is
//! *reused* across calls, which is what [`crate::PinnedBuffer`] is for.
//!
//! # What remains
//!
//! After this, the output allocation (109 ms) is the single largest term in a
//! 512³ `rescale_intensity` — larger than H2D, D2H, and the kernel combined. It
//! is inherent to returning a freshly allocated buffer: Linux must zero every
//! anonymous page it hands out. The only way to remove it is to stop allocating,
//! i.e. reuse an output buffer across calls, which a one-shot op has no way to do
//! and an iterative device-resident loop gets for free.

/// Threads used to fault in a large output buffer. Measured in the op at 512³:
/// 1 thread 357 ms, 16 threads 109 ms, and 96 threads is *worse* than 16 —
/// page-zeroing and the mm lock serialize, so more threads only add spawn cost.
const PREFAULT_THREADS: usize = 16;

/// Below this, the allocation is not worth preparing: a `Vec` this small is
/// either already in a resident heap arena or costs fewer faults than the
/// threads would cost to spawn.
const PREFAULT_MIN_BYTES: usize = 4 << 20;

#[cfg(target_os = "linux")]
mod huge {
    /// Ask the kernel to back `buf` with 2 MiB pages.
    ///
    /// `madvise` requires a page-aligned address, and a `Vec`'s pointer is not
    /// (glibc puts a header in front of it), so this advises the page-aligned
    /// *interior* — the head and tail fragments stay 4 KiB-backed, which costs a
    /// handful of faults out of tens of thousands.
    ///
    /// Worth 109 ms against 185 ms on this machine at 512 MiB (THP `defrag` =
    /// `madvise`, so the collapse happens synchronously under the fault).
    ///
    /// **Advisory, and deliberately infallible.** A kernel with transparent huge
    /// pages disabled, a mapping it declines to collapse, an `EINVAL` — all are
    /// ignored. The caller touches every page afterwards regardless, so the
    /// buffer ends up resident either way and the result is identical, only
    /// slower. Nothing here may turn a working call into a failing one to save
    /// 76 ms.
    pub fn advise(buf: &mut [impl Sized]) {
        let base = buf.as_mut_ptr() as usize;
        let end = base + std::mem::size_of_val(buf);
        let start = base.next_multiple_of(4096);
        let stop = end & !4095;
        if stop > start {
            // SAFETY: `[start, stop)` is a page-aligned subrange of a live
            // allocation this thread owns. `MADV_HUGEPAGE` only changes the
            // kernel's page-size policy for that range; it neither moves nor
            // writes the memory. The return value is intentionally discarded —
            // see the doc comment.
            unsafe { libc::madvise(start as *mut _, stop - start, libc::MADV_HUGEPAGE) };
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod huge {
    /// No transparent-huge-page equivalent is wired up off Linux; the prefault
    /// below still makes the buffer resident, which is the load-bearing half.
    pub fn advise(_buf: &mut [impl Sized]) {}
}

/// Write one byte per 4 KiB page, forcing the kernel to map it. `write_volatile`
/// because storing a value the page already holds is a no-op the optimizer would
/// otherwise delete — and deleting it would silently restore the defect.
fn touch<T>(buf: &mut [T]) {
    let stride = (4096 / std::mem::size_of::<T>()).max(1);
    let mut i = 0;
    while i < buf.len() {
        // SAFETY: `i < buf.len()`, and `T: Copy` via the caller's bound, so
        // reading the value back out and writing it unchanged is well-defined.
        unsafe {
            let p = buf.as_mut_ptr().add(i);
            std::ptr::write_volatile(p, std::ptr::read_volatile(p));
        }
        i += stride;
    }
}

/// A zeroed `Vec<T>` of `len` elements whose pages are already resident, ready to
/// be a D2H destination without faulting under the DMA.
///
/// For small buffers this is a plain `vec![]` — see [`PREFAULT_MIN_BYTES`].
pub fn resident_vec<T: Copy + Default + Send>(len: usize) -> Vec<T> {
    let mut v = vec![T::default(); len];
    if std::mem::size_of_val(v.as_slice()) < PREFAULT_MIN_BYTES {
        return v;
    }
    huge::advise(v.as_mut_slice());

    let chunk = len.div_ceil(PREFAULT_THREADS);
    std::thread::scope(|s| {
        for part in v.chunks_mut(chunk) {
            s.spawn(|| touch(part));
        }
    });
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_vec_is_zeroed_and_correctly_sized() {
        // Above the prefault threshold, so it takes the madvise + touch path.
        let v: Vec<f32> = resident_vec(2_000_000);
        assert_eq!(v.len(), 2_000_000);
        assert!(
            v.iter().all(|&x| x == 0.0),
            "prefault must not perturb values"
        );
    }

    #[test]
    fn small_allocations_skip_the_prefault_and_are_still_correct() {
        let v: Vec<f32> = resident_vec(16);
        assert_eq!(v, vec![0.0f32; 16]);
    }
}
