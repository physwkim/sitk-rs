//! Resident output buffers: huge-page-backed, and faulted **in parallel, once**.
//!
//! # The cost this module exists to delete
//!
//! Linux hands out anonymous memory lazily: `malloc` returns an address, and the
//! kernel does not attach a physical page until something writes to it. That
//! first write traps into the kernel, which zeroes a fresh 4 KiB page and maps
//! it. So a freshly allocated output volume is not free — it costs one page
//! fault per 4 KiB, and the bill lands on whoever writes it first. For a 256³
//! `Float64` volume that is 32 768 faults; the port used to pay ~438 000 of them
//! per `rescale_intensity` call, and the *arithmetic* was never the problem.
//!
//! Two things make that bill smaller:
//!
//! - **Huge pages.** [`MADV_HUGEPAGE`](libc::MADV_HUGEPAGE) asks the kernel to
//!   back the region with 2 MiB pages instead of 4 KiB ones, which is 512× fewer
//!   faults for the same bytes — and 512× fewer TLB entries to cover the volume,
//!   which the *reader* of the buffer feels too.
//! - **Faulting on every core instead of one.** The fault is a kernel-side page
//!   zeroing, and it parallelizes. [`resident_vec`] touches the whole buffer
//!   from the rayon pool before returning it, so the pages arrive on many cores
//!   at once rather than one at a time inside a hot loop.
//!
//! # The advice is advisory
//!
//! `madvise` is a *hint*. A kernel with transparent huge pages disabled, a
//! region the kernel declines to back, an unaligned or too-small buffer — all of
//! these make it fail, and every one of them is fine. Failure here is silently
//! ignored and the plain allocation is used: **a page fault is never worth
//! failing a working call over.** Nothing in this module can change a computed
//! value; it changes only where the pages come from.
//!
//! # Measured on this machine: the kernel declines the advice
//!
//! `/sys/kernel/mm/transparent_hugepage/enabled` reads `[madvise]` and
//! `madvise(MADV_HUGEPAGE)` returns 0 — yet `/proc/vmstat` shows
//! `thp_fault_alloc 0` **and** `thp_fault_fallback 0` after 39 days of uptime,
//! with 6 086 free order-9 blocks. The kernel is not failing to *find* a huge
//! page; it never asks for one. A 2 MiB-aligned fresh `mmap` carrying the same
//! advice gets none either, so this is not an alignment bug in this module.
//!
//! So on this box the huge-page half measures as exactly **zero**: 33 687 minor
//! faults for a 256³ `rescale_intensity` with the advice, 33 689 without. The
//! code stays because it is correct and free — one syscall per output volume —
//! and on a kernel that grants THP it removes 511 of every 512 faults. It cannot
//! change a computed value either way.
//!
//! # What it does not fix
//!
//! Making the fault cheaper is not the same as not paying it. A caller in a loop
//! still pays it once per call, because the buffer dies at the end of the call.
//! That half is closed by the `_into` forms — [`crate::core::map_pixels_into`] and
//! [`crate::core::NeighborhoodIterator::par_map_window_into`] — which write into a
//! destination the caller owns and can reuse.

use std::mem::size_of;

use rayon::prelude::*;

/// Buffers below this are left entirely alone: a `madvise` syscall and a pool
/// hand-off both cost more than faulting a handful of small pages, and one
/// transparent huge page is 2 MiB, so a smaller region cannot be backed by one
/// anyway.
const RESIDENT_THRESHOLD_BYTES: usize = 2 << 20;

/// Bytes per prefault task. Large enough that the pool hand-off disappears
/// against the page-zeroing the task triggers.
const PREFAULT_GRAIN_BYTES: usize = 1 << 21;

/// A zero-filled `Vec<T>` of length `len` whose pages are **already resident**:
/// huge-page-backed where the kernel allows it, and faulted in on the rayon pool
/// rather than one at a time by whoever writes first.
///
/// Bit-for-bit identical to `vec![T::default(); len]` — this changes only *when*
/// and *how* the pages are faulted, never a value. Small allocations fall
/// through to exactly that.
///
/// # When this is the right tool — and when it is not
///
/// The prefault is a full write pass. It pays only when the buffer's *real*
/// first writer is **serial**, because then the fault bill is serial too and
/// this moves it onto every core: a device-to-host `memcpy`, a decoder, an
/// `io::Read::read_exact`.
///
/// It does **not** pay when the buffer is filled by a parallel pass, because
/// that pass's own workers already fault their own slices concurrently — there
/// is no serial fault to move, and the prefault is then a second write pass for
/// nothing. Measured: routing [`crate::core::map_pixels`]'s output through here cost
/// `binary_dilate` ~4% and gained `rescale_intensity` nothing. That is why the
/// maps in [`crate::core::parallel`] take [`resident_capacity`] instead — the advice
/// without the prefault.
pub fn resident_vec<T: Default + Send>(len: usize) -> Vec<T> {
    let bytes = len * size_of::<T>();
    if bytes < RESIDENT_THRESHOLD_BYTES {
        return (0..len).map(|_| T::default()).collect();
    }

    let mut v: Vec<T> = resident_capacity(len);

    // Fault the whole thing, on every core. The write *is* the fault, so it has
    // to be the FIRST touch — `resize`/`vec![]` would fault the buffer serially
    // and leave nothing to parallelize. Hence the write goes into the spare
    // capacity, from the pool.
    let grain = (PREFAULT_GRAIN_BYTES / size_of::<T>().max(1)).max(1);
    v.spare_capacity_mut()[..len]
        .par_chunks_mut(grain)
        .for_each(|chunk| {
            for slot in chunk {
                slot.write(T::default());
            }
        });

    // SAFETY: `resident_capacity(len)` reserved at least `len` slots, and the
    // loop above wrote every one of the first `len` exactly once —
    // `par_chunks_mut` partitions the slice, so each slot is in exactly one
    // chunk. The elements are therefore initialized.
    unsafe { v.set_len(len) };
    v
}

/// An **empty** `Vec<T>` with capacity for `len` elements, whose pages have been
/// advised as huge-page candidates but **not touched**.
///
/// For the caller that is about to fill every slot in parallel: the fill's own
/// workers fault their own pages, concurrently, so there is no serial fault to
/// hoist and a prefault pass here would just write the buffer twice. What is
/// still worth doing is the *advice* — it is one syscall, it costs nothing, and
/// on a kernel that grants huge pages it removes 511 of every 512 faults the
/// fill would otherwise take, along with the TLB pressure the buffer's readers
/// pay later.
///
/// The returned `Vec` has length 0. Filling it means writing
/// [`Vec::spare_capacity_mut`] and then [`Vec::set_len`], which is `unsafe`; the
/// safe wrappers around that live in [`crate::core::parallel`].
pub(crate) fn resident_capacity<T>(len: usize) -> Vec<T> {
    // Reserve first, advise second: the advice needs an address, and
    // `with_capacity` has not touched a single page yet.
    let v: Vec<T> = Vec::with_capacity(len);
    advise_huge(v.as_ptr() as usize, len * size_of::<T>());
    v
}

/// Ask the kernel to back `[ptr, ptr + bytes)` with huge pages.
///
/// Advisory in the strict sense: the result is deliberately discarded. `madvise`
/// requires a page-aligned start, so this advises only the whole pages *inside*
/// the allocation; the ragged ends stay 4 KiB-backed, which is correct and
/// costs at most two faults.
#[cfg(target_os = "linux")]
fn advise_huge(ptr: usize, bytes: usize) {
    if bytes < RESIDENT_THRESHOLD_BYTES {
        return;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return;
    }
    let page = page as usize;
    let start = ptr.next_multiple_of(page);
    let end = (ptr + bytes) / page * page;
    if end <= start {
        return;
    }
    // SAFETY: `[start, end)` lies inside the allocation `ptr` owns, and both are
    // page-aligned. `madvise` with `MADV_HUGEPAGE` neither reads nor writes the
    // region — it only records a policy on the VMA — so it cannot observe the
    // uninitialized bytes there, and cannot invalidate the pointer.
    //
    // The return value is intentionally dropped: see the module docs. Every
    // failure mode (THP off, an unbacked VMA, an old kernel) leaves us with a
    // perfectly good ordinary allocation.
    let _ = unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
}

/// Non-Linux platforms have no `MADV_HUGEPAGE`; the allocation is plain, which
/// is the same fallback a Linux kernel that refuses the advice gives us.
#[cfg(not(target_os = "linux"))]
fn advise_huge(_ptr: usize, _bytes: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The only contract that matters: the buffer is indistinguishable from
    /// `vec![T::default(); len]`. Residency is a property of the pages, not of
    /// the values, and no test may be able to tell the difference.
    #[test]
    fn a_resident_vec_equals_the_plain_allocation() {
        for len in [0usize, 1, 1023, RESIDENT_THRESHOLD_BYTES, 1 << 20] {
            let got: Vec<f64> = resident_vec(len);
            assert_eq!(got.len(), len);
            assert_eq!(got, vec![0.0f64; len]);
        }
    }

    /// Both sides of the size threshold, and both the huge-page path and the
    /// plain one, must produce a zeroed buffer of the right length for a type
    /// whose default is not all-zero bits at the byte level either.
    #[test]
    fn every_element_type_lands_zeroed_on_the_parallel_path() {
        let n = RESIDENT_THRESHOLD_BYTES; // > threshold for u8 and for f32/f64
        assert!(resident_vec::<u8>(n).iter().all(|&x| x == 0));
        assert!(resident_vec::<f32>(n).iter().all(|&x| x == 0.0));
        assert!(resident_vec::<i64>(n).iter().all(|&x| x == 0));
    }
}
