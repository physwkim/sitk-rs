//! Deterministic CPU parallelism — the only parallel primitive in the port.
//!
//! # The determinism contract
//!
//! Every filter in this port must produce **bit-identical** output regardless of
//! how many threads run it. Floating-point addition is not associative, so a
//! parallel reduction that re-associates a sum (rayon's `fold`/`reduce` with
//! `f64::add`, say) yields a result that depends on how the work happened to be
//! split — and rayon splits by *steal order*, which depends on the thread count
//! and on scheduling luck. Such a reduction is non-deterministic by
//! construction, and this module makes it **unwritable**:
//!
//! - `rayon` is a dependency of `sitk-core` **only**. No other crate in the
//!   workspace lists it, so no filter can reach `par_iter()` at all. Every
//!   parallel loop in the port goes through this module.
//! - The parallel surface here is exactly three shapes: an **order-preserving
//!   map** ([`map_indexed`]), an **independent-line pass** ([`for_each_line_mut`]),
//!   and two **exactly-associative reductions** ([`min_max`], [`bin_counts`]).
//! - There is deliberately **no** `fold`/`reduce` taking a caller-supplied
//!   combine closure. A caller cannot hand this module a float `+` to
//!   re-associate, because no entry point accepts one. Adding a new reduction
//!   means adding a function *here*, next to this comment, with a proof that its
//!   operator is exactly associative.
//! - Chunk boundaries come from one private function, `grain`, whose signature
//!   is `usize` in and `usize` out: it *cannot* observe the thread count, because
//!   there is no argument through which it could. No entry point takes a chunk
//!   count, a grain size, or a thread count either, so no decomposition in the
//!   port can depend on `rayon::current_num_threads()` — by construction, not by
//!   convention.
//!
//! ## Why each shape is bit-stable
//!
//! - **[`map_indexed`]** — output element `i` is `f(i)`, written to slot `i`.
//!   No accumulator crosses elements, so there is nothing to re-associate; the
//!   result equals the sequential map for any decomposition.
//! - **[`for_each_line_mut`]** — the buffer is partitioned into 1-D lines along
//!   one axis; each line is read and written by exactly one task, and the
//!   sequential recursion *within* a line is untouched. Cross-line order is
//!   irrelevant because lines are disjoint. Bit-identical to the serial line
//!   loop for any decomposition.
//! - **[`min_max`]** — chunk-local folds combined left-to-right in chunk order.
//!   `min`/`max` *select an element of the input set*: they are associative and
//!   idempotent, so any bracketing of the fold yields the same `f64` bits. (The
//!   accumulator starts at `±INFINITY` and `f64::min`/`f64::max` return the
//!   non-NaN operand, so a NaN in the data can never enter the accumulator on
//!   either the serial or the chunked path.)
//! - **[`bin_counts`]** — chunk-local `u64` histograms combined left-to-right.
//!   Integer addition is exactly associative and commutative; the bin index of
//!   an element is a pure function of that element.
//!
//! Because all four are *exact*, the serial fast path taken below
//! `SERIAL_THRESHOLD` returns the same bits as the parallel path — the
//! threshold is a speed knob, never a correctness one.

use std::mem::MaybeUninit;

use rayon::prelude::*;

use crate::core::pixel::Scalar;

/// The coarsest grain a map pass is cut to — its per-task element count once the
/// input is big enough that [`grain`] stops subdividing. Also the target
/// per-task element count of the line pass ([`for_each_line_mut`], which decomposes
/// by whole blocks and has its own task-count guard). Only a scheduling knob:
/// every primitive here is exact, so changing it cannot change a result.
const GRAIN: usize = 4096;

/// Below this many elements, run serially — thread hand-off costs more than the
/// work. Bit-identical to the parallel path by the exactness argument above.
const SERIAL_THRESHOLD: usize = 1 << 14;

/// The coarsest grain a reduction is cut to. See [`grain`] for the finer grains
/// a short input gets, and why the chunk count stays a function of the length.
const REDUCE_CHUNK: usize = 1 << 16;

/// The widest pool [`grain`] will try to raise enough tasks for.
///
/// An **upper bound on** the worker count of any box this runs on — never a
/// *reading* of the running pool. That distinction is the determinism contract:
/// a grain that depended on the thread count would make the chunk decomposition,
/// and so every fold over it, a function of the schedule. This is a constant in
/// the source, so it is not.
///
/// A box wider than this is fed by whatever tasks the input length can raise
/// (bounded by `len / MIN_GRAIN`, below) — no rule keyed on `len` alone can do
/// better, because a short input simply does not contain more parallelism.
const TARGET_TASKS: usize = 256;

/// The floor on per-task work for the **elementwise and reduce** passes —
/// [`fill_zip`], [`for_each_mut`], [`min_max`], [`bin_counts`], and the line pass.
///
/// A floor exists to keep the work in one task well above the cost of dispatching
/// one (order 1 µs for a rayon job). That is a floor on **work**; it is spelled in
/// **elements**, and the conversion between them is the pass's cost per element. So
/// the floor is a property of the *cost class*, not of the module.
///
/// This class does one or two ops per element, well under a nanosecond each. At
/// 2048 elements a task carries ~2 µs, already the same order as its own dispatch;
/// below that the pass starts paying more to schedule than to compute. Measured, on
/// 64³/`tN`, 1024 against 2048, paired within each round over 10 clean rounds across
/// two independent windows: `otsu_threshold` **1.141×** (band [1.10, 1.18]) and
/// **1.145×** ([1.04, 1.31]) — a replicated regression, and the reason this class's
/// floor does not follow the indexed class's down to 1024.
///
/// (That measurement supersedes a single-shot 64³ grain sweep previously recorded
/// here, which read 1024 → 1.38 ms against 2048 → 1.57 ms and so pointed the other
/// way. It was one leg on an unpinned process shape; the number above is paired
/// within-round and replicated, and this port's noise floor — ~60% on process shape
/// — is larger than the gap that sweep claimed to see.)
const MIN_GRAIN: usize = 2048;

/// Elements per chunk for a pass over `len` elements, at most `ceiling`.
///
/// **A pure function of `len`.** It reads nothing else: no thread count, no pool
/// handle, no ambient state — the signature is `usize` in, `usize` out, and that
/// is the whole determinism argument. `par_chunks(grain(len, c))` therefore cuts
/// at the same multiples of the same integer on every box, at every thread count,
/// under every steal order. The in-order fold over those chunks is then a fold
/// over a fixed decomposition, which is what makes it bit-stable — see the module
/// docs.
///
/// Between the two clamps it targets [`TARGET_TASKS`] chunks, so a short input is
/// cut finely enough to fill a wide pool instead of handing it four chunks. Both
/// ends are load-bearing:
///
/// - Without the [`MIN_GRAIN`] floor a tiny input would raise thousands of
///   near-empty tasks and pay more in scheduling than it saves.
/// - Without the `ceiling` a large input would too. `otsu_threshold` at 256³
///   measures 38.0 ms at a 65 536 grain and 47.0 ms at 2048 — a 24% regression —
///   because a reduction's sequential combine is `O(chunks)`. The ceiling is why
///   this rule is a **no-op at and above `TARGET_TASKS * ceiling` elements**: for
///   any such `len`, `len.div_ceil(TARGET_TASKS) >= ceiling`, the clamp pins the
///   grain at `ceiling`, and the emitted chunk boundaries are the same integers
///   they were before this rule existed.
const fn grain(len: usize, ceiling: usize) -> usize {
    grain_floored(len, ceiling, MIN_GRAIN)
}

/// [`grain`] with the floor named by the caller, so no pass silently inherits a
/// floor derived for a cost class it is not in. Every grain helper below states
/// which class it is: the floor is a parameter of the rule, not a global.
const fn grain_floored(len: usize, ceiling: usize, floor: usize) -> usize {
    // `clamp` is not `const`; this is the same expression.
    let g = len.div_ceil(TARGET_TASKS);
    if g < floor {
        floor
    } else if g > ceiling {
        ceiling
    } else {
        g
    }
}

/// [`grain`] for the **elementwise** maps — [`map_slice`] and [`for_each_mut`],
/// a few nanoseconds per element. Cheap class: [`MIN_GRAIN`].
const fn map_grain(len: usize) -> usize {
    grain(len, GRAIN)
}

/// [`grain`] for the chunked reductions — [`min_max`] and [`bin_counts`].
const fn reduce_grain(len: usize) -> usize {
    grain(len, REDUCE_CHUNK)
}

/// [`grain`] for the **line pass** ([`for_each_line_mut`]), expressed in whole
/// blocks — the atom that path decomposes by.
///
/// A line lies *inside* one block, so a task takes a run of whole blocks and can
/// never split one. This converts the same per-task element target every other
/// pass here uses into that run length. Two consequences, both pinned as
/// integers below:
///
/// - **It is a no-op at and above `medium`.** `grain(len, GRAIN) == GRAIN` for
///   any `len >= TARGET_TASKS * GRAIN` (1 048 576 elements), so the runs it emits
///   are the same integers the former fixed `GRAIN` emitted. Both bench volumes
///   at or above `medium` are far past that bound —
///   `the_line_grain_emits_the_same_block_runs_as_the_fixed_grain_at_and_above_medium`.
/// - **It cannot lift the block-count cap.** When `block >= grain` the run length
///   is 1 whatever the grain, and the task count is exactly `outer`. At 64³ along
///   axis 1 that is 64 tasks on a box with 96 workers, and *no* grain rule closes
///   it — only a decomposition that splits a block. See
///   `the_line_pass_task_count_is_capped_by_the_block_count_not_by_the_grain`.
const fn line_blocks_per_task(len: usize, block: usize) -> usize {
    let g = grain(len, GRAIN);
    // `usize::max` is not `const`; this is the same expression.
    let runs = g.div_ceil(block);
    if runs < 1 { 1 } else { runs }
}

/// The line pass takes the whole-block path only if it can raise at least this
/// many tasks; otherwise it splits a block's lines column-wise instead.
///
/// **This is a path selector, not a floor on the task count**, and the difference
/// is load-bearing: raising it does not create one task. It only diverts more
/// shapes to the column path — and for an `inner == 1` shape (the axis-0 pass,
/// whose lines *are* the blocks) there is no column path to divert to, so the
/// pass falls into the **serial** branch instead. Raising this to the width of
/// this box's pool (96) would therefore send the entire 64³ axis-0 line pass down
/// one thread, which is the opposite of the intent.
/// `min_block_tasks_never_sends_a_bench_volume_line_pass_serial` fails if a future
/// edit tries it.
///
/// The pool is fed by [`line_blocks_per_task`], which is where the task count
/// actually comes from.
const MIN_BLOCK_TASKS: usize = 32;

/// Upper bound on the column-chunk tasks per block on the column-split path,
/// which materializes one row-view per (task, row).
const MAX_COLUMN_TASKS: usize = 512;

/// Runs `f` on a rayon pool of exactly `threads` threads.
///
/// The determinism contract's test seam. Every op in the port must return
/// bit-identical output under every `threads`, and
/// `sitk-filters/tests/bit_parity.rs` asserts exactly that for each of the
/// benchmark ops. It is also how a benchmark harness pins the `t1` (single
/// thread) configuration of `doc/bench-spec.md`.
///
/// # Panics
///
/// If the pool cannot be built (`threads == 0`, or the platform refuses the
/// threads).
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
        .install(f)
}

/// `out[i] = f(i)` for `i` in `0..len`, in parallel, collected in index order.
///
/// The bit-for-bit result of `(0..len).map(f).collect()`: `f` sees only `i`, and
/// element `i` lands in slot `i` whatever the decomposition.
///
/// The buffer comes from [`crate::core::alloc::resident_vec`], so its pages are
/// faulted in on the pool rather than one 4 KiB page at a time by the loop that
/// fills it. That is invisible in the result — see [`map_indexed_into`], which
/// is the loop body this runs.
pub fn map_indexed<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    map_indexed_init(len, || (), |(), i| f(i))
}

/// `dst[i] = f(i)` for every element of `dst`, in parallel — [`map_indexed`]
/// writing into a destination the **caller owns**.
///
/// This is the form that closes the page-fault bill rather than merely making it
/// cheaper: a caller that runs the same pass in a loop allocates `dst` once, and
/// pays for its pages once, instead of once per iteration. [`map_indexed`] is
/// this function plus an allocation, so there is one loop body, not two that can
/// drift.
///
/// Element `i` is written by exactly one task, from `i` alone, so the result is
/// bit-identical to the sequential `for i in 0..dst.len() { dst[i] = f(i) }` at
/// any thread count.
pub fn map_indexed_into<R, F>(dst: &mut [R], f: F)
where
    R: Send + Copy,
    F: Fn(usize) -> R + Sync + Send,
{
    map_indexed_init_into(dst, || (), |(), i| f(i));
}

/// `src.iter().map(f).collect()`, in parallel — the elementwise transform of a
/// whole buffer.
///
/// Prefer this over [`map_indexed`] whenever the input *is* a slice. Both are
/// bit-identical to the sequential map, but this one hands each task a
/// contiguous `&[T]` and walks it with a plain slice iterator, which the
/// optimizer vectorizes and which needs no bounds check — where `map_indexed`'s
/// `|i| src[i]` pays a bounds check per element and does not vectorize. On the
/// port's widest-used map (`Image::to_f64_vec`, a whole-image widening that
/// nearly every filter starts with) that difference is large enough to make an
/// op that is *not* parallelized measurably slower.
///
/// The iterator is kept *indexed* (`par_iter` + [`with_min_len`], never
/// `flat_map_iter`): an indexed `collect` writes each element straight into its
/// final slot in one preallocated buffer, whereas an unindexed one has every
/// task heap-allocate its own `Vec` and then copies the lot. `with_min_len`
/// bounds the split depth without changing the result — it is a scheduling
/// hint, not a decomposition the output depends on.
///
/// Order-preserving by construction: an indexed `collect` places element `i` at
/// index `i`, so the result is `f(&src[i])` for every `i` regardless of how the
/// range was split.
///
/// [`with_min_len`]: rayon::iter::IndexedParallelIterator::with_min_len
pub fn map_slice<T, R, F>(src: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    collect_filled(src.len(), |slots| fill_zip(src, slots, &f))
}

/// `dst[i] = f(&src[i])` for every element, in parallel — [`map_slice`] writing
/// into a destination the **caller owns**, so a caller in a loop pays for the
/// output pages once rather than once per call.
///
/// [`map_slice`] is this function plus an allocation: one loop body, not two.
///
/// Both slices are walked as contiguous chunks, so the pass keeps the
/// vectorizable, bounds-check-free inner loop [`map_slice`] documents. Element
/// `i` is written by exactly one task from `src[i]` alone.
///
/// # Panics
///
/// If `src.len() != dst.len()`. This is the caller passing the wrong buffer, not
/// a runtime condition to recover from — the image-level entry points
/// ([`crate::core::map_pixels_into`]) turn a size mismatch into a typed error before
/// it can reach here.
pub fn map_slice_into<T, R, F>(src: &[T], dst: &mut [R], f: F)
where
    T: Sync,
    R: Send + Copy,
    F: Fn(&T) -> R + Sync + Send,
{
    assert_eq!(
        src.len(),
        dst.len(),
        "map_slice_into: source and destination must have the same length"
    );
    fill_zip(src, as_uninit_mut(dst), &f);
}

/// [`map_indexed`] with a per-task scratch value: `init` runs once per worker
/// task, and `f(&mut scratch, i)` produces element `i`.
///
/// Same bit-for-bit guarantee as [`map_indexed`] — element `i` is `f`'s return
/// value for `i` and lands in slot `i` — provided `scratch` is only ever used as
/// working storage that `f` fully overwrites for each `i`, never as an
/// accumulator that carries state between elements. That restriction is what
/// keeps the result independent of how items are grouped into tasks, so it is
/// this function's contract.
///
/// It exists because the alternative — allocating per element — is what makes a
/// parallel sliding-window filter slower than its serial version: 16.7 M windows
/// each doing a heap allocation and a refcount bump on a shared cache line
/// serializes 96 cores on the allocator and the atomic.
pub fn map_indexed_init<R, S, I, F>(len: usize, init: I, f: F) -> Vec<R>
where
    R: Send,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    collect_filled(len, |slots| fill_indexed(slots, init, f))
}

/// [`map_indexed_init`] writing into a destination the **caller owns** — and the
/// single loop body every map in this module is built from ([`map_indexed`],
/// [`map_indexed_into`], [`map_indexed_init`] and, through
/// [`crate::core::NeighborhoodIterator::par_map_window_into`], the whole stencil
/// family).
///
/// Element `i` is written by exactly one task, from `i` and per-task scratch
/// that `f` fully overwrites per element (the [`map_indexed_init`] contract), so
/// the result is bit-identical to the sequential loop at any thread count.
pub fn map_indexed_init_into<R, S, I, F>(dst: &mut [R], init: I, f: F)
where
    R: Send + Copy,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    fill_indexed(as_uninit_mut(dst), init, f);
}

/// A run of consecutive indices whose cost per element is the same.
///
/// The unit of [`map_indexed_init_by_cost`]'s partition. `class` is an opaque tag:
/// this module never asks what it *means*, only which runs share it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostRun {
    /// How many consecutive indices this run covers.
    pub len: usize,
    /// Which cost class they belong to. Only equality is used.
    pub class: u8,
}

/// [`map_indexed_init`] over an index space the caller has partitioned into **cost
/// classes** — the fill for a pass whose cost per index is not uniform.
///
/// # The defect this closes
///
/// `fill_indexed` cuts `0..len` into equal-**count** chunks, which are equal-**cost**
/// chunks only when every index costs the same. A stencil pass is where that fails: a
/// pixel whose window overhangs the image takes the checked path, which materializes
/// the window through the boundary condition and measures ~**8×** the interior path
/// per pixel (~1.0 µs against ~130 ns, 5³ window). Those pixels are not scattered —
/// the layout concentrates them into whole rows and planes — so an equal-count chunker
/// hands *one task* the entire `z = 0` plane, and that task becomes the wall. Measured
/// at 64³/`t96` before this function existed: the longest chunk was **0.93–1.00 of the
/// whole pass**, and for `mean` the last chunk to finish was chunk 0, which started at
/// t≈0 and ran alone while the other 127 completed around it.
///
/// # The rule, and why it carries no cost constant
///
/// **Each class is split into its own `TARGET_TASKS` chunks**, so
///
/// ```text
/// c_max = max over classes of (W_class / TARGET_TASKS)  <=  W / TARGET_TASKS
/// ```
///
/// The ratio *between* the classes never enters: the classes are counted, not weighted,
/// so there is no cost model here to get wrong and none to fit to a benchmark. A caller
/// that mislabels a run makes the balance worse, never the answer. The `GRAIN` ceiling
/// still applies per class — dropping it measured **1.41× slower** on `binary_dilate` at
/// 256³ — while the *floor* does not, because a floor in elements cannot serve two costs
/// per element and `TARGET_TASKS` is what bounds the task count.
///
/// # Bit-for-bit
///
/// This changes **which task** computes an element, never **how**: chunks are still
/// contiguous index ranges, element `i` is still `f(&mut scratch, i)`, and it still lands
/// in slot `i`. It is the same argument that lets `grain` be tuned freely — no
/// element's value depends on the decomposition, so no decomposition can move a bit.
///
/// # Panics
///
/// If the runs do not sum to `len` — a partition that does not cover the index space
/// exactly once would leave slots uninitialized, which is a caller bug.
pub fn map_indexed_init_by_cost<R, S, I, F>(len: usize, runs: &[CostRun], init: I, f: F) -> Vec<R>
where
    R: Send,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    collect_filled(len, |slots| fill_by_cost(slots, runs, init, f))
}

/// [`map_indexed_init_by_cost`] writing into a destination the **caller owns**.
///
/// # Panics
///
/// If the runs do not sum to `dst.len()`.
pub fn map_indexed_init_into_by_cost<R, S, I, F>(dst: &mut [R], runs: &[CostRun], init: I, f: F)
where
    R: Send + Copy,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    fill_by_cost(as_uninit_mut(dst), runs, init, f);
}

// ---------------------------------------------------------------------------
// The two fill loops every map above is built from, and the allocation that
// turns a fill into a `Vec`. Each `_into` form and its allocating twin call the
// SAME loop, so they cannot drift.
// ---------------------------------------------------------------------------

/// `slots[i] = f(&mut scratch, i)` for every slot — the indexed fill.
///
/// Element `i` is written by exactly one task from `i` alone, so the result
/// equals the sequential loop bit-for-bit at any thread count. Every slot is
/// written exactly once: `par_chunks_mut` partitions the slice.
///
/// # Why the leaf is capped at one chunk
///
/// [`with_max_len(1)`] forces the job tree to split down to a single
/// [`map_grain`] chunk per task. Without it rayon splits *adaptively* — a job divides only
/// when another worker tries to steal it — so once every worker is busy, no
/// further splitting happens and whoever holds the largest un-split leaf runs it
/// to the end alone. Measured on `mean` (a 5³ window, 256³ voxels, 48 threads):
/// one worker held a single task of 262 144 voxels that ran 175 ms, *the entire
/// region*, while the other 47 finished their 80 ms and slept. The wall was that
/// one leaf. Capping the leaf took the same pass from 174 ms to 75 ms (11x to
/// 30x over `t1`) and lifted the busy-core count from 21 to 47 of 48.
///
/// This is a scheduling knob and nothing more: it changes how the index range is
/// cut, never which `i` an element is computed from, so it cannot move a bit —
/// the same argument that lets [`grain`] be tuned freely.
///
/// It is deliberately **not** applied to [`fill_zip`] or [`for_each_mut`]. Those
/// carry the *cheap* elementwise maps (a vectorized transform of a contiguous
/// slice, a few nanoseconds per element), where the tail is negligible and a job
/// dispatch per chunk is not: forcing the split there measurably *slowed*
/// `rescale_intensity` (12.1 ms to 14.3 ms at 48 threads). The two fills serve
/// two cost classes — indexed/stencil work that is expensive per element, and
/// elementwise work that is nearly free per element — and the split policy
/// follows from which one you are in, not from a tuned number.
///
/// # This fill is for indexed passes whose cost per index is *uniform*
///
/// It cuts equal-**count** chunks, which are equal-**cost** chunks only when every
/// index costs the same — a convolution's kernel, a metric's per-sample term, a
/// distance transform's per-voxel update. A **stencil** pass is the case where the
/// assumption fails: its cost per pixel splits ~8× between the interior and the
/// checked path, the layout concentrates the expensive pixels into whole rows and
/// planes, and an equal-count chunker then hands one task the entire `z = 0` plane.
/// Measured at 64³/`t96`: the longest chunk was **0.93–1.00 of the whole pass**. That
/// pass goes through [`map_indexed_init_by_cost`] now, which is handed the partition
/// by the only layer that knows it — see `NeighborhoodIterator::cost_runs`.
///
/// This fill once carried a second grain floor (`MIN_GRAIN_INDEXED = 1024`) whose only
/// service was to halve that straggler. It was a workaround for the wrong
/// decomposition and it is gone; the passes still on this fill are the uniform-cost
/// ones, which the floor was never derived for.
///
/// [`with_max_len(1)`]: rayon::iter::IndexedParallelIterator::with_max_len
fn fill_indexed<R, S, I, F>(slots: &mut [MaybeUninit<R>], init: I, f: F)
where
    R: Send,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    if slots.len() < SERIAL_THRESHOLD {
        let mut scratch = init();
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.write(f(&mut scratch, i));
        }
        return;
    }
    let g = map_grain(slots.len());
    slots
        .par_chunks_mut(g)
        .enumerate()
        .with_max_len(1)
        .for_each_init(init, |scratch, (c, chunk)| {
            let base = c * g;
            for (j, slot) in chunk.iter_mut().enumerate() {
                slot.write(f(scratch, base + j));
            }
        });
}

/// The chunk boundaries [`fill_by_cost`] cuts at, as **lengths in index order**.
///
/// A pure function of the partition — no thread count, no pool, no ambient state, the
/// same integers on every box at every thread count, exactly as [`grain`] is. It is
/// separated from the fill so it can be pinned as integers rather than inferred from a
/// wall clock.
///
/// Two properties, both pinned below:
///
/// - **No chunk spans two classes.** Cuts are taken *within* a run, never across one,
///   so every chunk is cost-homogeneous.
/// - **No chunk holds more than `W_class / TARGET_TASKS` of its class**, and none
///   exceeds the [`GRAIN`] ceiling. That is what bounds `c_max` without ever comparing
///   the classes to each other.
fn cost_chunks(runs: &[CostRun]) -> Vec<usize> {
    let mut total = [0usize; 256];
    for r in runs {
        total[r.class as usize] += r.len;
    }
    // The module's own grain rule, applied to each class's own total. The *ceiling* is
    // the same [`GRAIN`] every indexed pass has always had, and it is load-bearing in
    // the other direction: without it a class with a large total emits chunks far bigger
    // than the flat rule ever did — measured at 256³, `binary_dilate` **1.41× slower**,
    // because a 62 500-element chunk is 15× what the ceiling allows. The *floor* is
    // dropped (1), and only here.
    let grain_of = |class: u8| grain_floored(total[class as usize], GRAIN, 1);

    let mut chunks = Vec::new();
    for r in runs {
        let g = grain_of(r.class);
        let mut left = r.len;
        while left > 0 {
            let take = left.min(g);
            chunks.push(take);
            left -= take;
        }
    }
    chunks
}

/// `slots[i] = f(&mut scratch, i)` for every slot, cut at [`cost_chunks`] — the
/// cost-class fill behind [`map_indexed_init_by_cost`].
///
/// Every slot is written exactly once: the chunk lengths sum to `slots.len()` and the
/// slice is handed out by successive `split_at_mut`, so the chunks are disjoint by
/// construction and the borrow checker is the proof.
///
/// # Panics
///
/// If the runs do not sum to `slots.len()`.
fn fill_by_cost<R, S, I, F>(slots: &mut [MaybeUninit<R>], runs: &[CostRun], init: I, f: F)
where
    R: Send,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    let covered: usize = runs.iter().map(|r| r.len).sum();
    assert_eq!(
        covered,
        slots.len(),
        "fill_by_cost: the cost-class runs must cover the index space exactly once"
    );
    if slots.len() < SERIAL_THRESHOLD {
        let mut scratch = init();
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.write(f(&mut scratch, i));
        }
        return;
    }

    let mut rest = slots;
    let mut tasks = Vec::new();
    let mut base = 0usize;
    for len in cost_chunks(runs) {
        let (chunk, tail) = rest.split_at_mut(len);
        tasks.push((base, chunk));
        base += len;
        rest = tail;
    }
    debug_assert!(rest.is_empty());

    // `with_max_len(1)` for the same reason [`fill_indexed`] needs it — and the reason is
    // *not* that the chunks are unequal. A rayon job may hold a whole **run** of these
    // tasks and execute it sequentially: building the leaves here does not prevent that,
    // because rayon splits the task list adaptively, stops splitting once every worker is
    // busy, and whoever holds the longest run runs it alone. Measured at 256³ without the
    // cap: `binary_dilate` **3.5×** and `mean` **2.2×** *slower* than the flat chunker —
    // the straggler this function exists to remove, reintroduced one level up.
    tasks
        .into_par_iter()
        .with_max_len(1)
        .for_each_init(init, |scratch, (b, chunk)| {
            for (j, slot) in chunk.iter_mut().enumerate() {
                slot.write(f(scratch, b + j));
            }
        });
}

/// `slots[i] = f(&src[i])` for every slot — the zipped fill, which keeps both
/// sides contiguous so the inner loop stays vectorizable and bounds-check-free
/// (see [`map_slice`]).
fn fill_zip<T, R, F>(src: &[T], slots: &mut [MaybeUninit<R>], f: &F)
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    debug_assert_eq!(src.len(), slots.len());
    if src.len() < SERIAL_THRESHOLD {
        for (slot, x) in slots.iter_mut().zip(src) {
            slot.write(f(x));
        }
        return;
    }
    let g = map_grain(src.len());
    slots
        .par_chunks_mut(g)
        .zip(src.par_chunks(g))
        .for_each(|(dst_chunk, src_chunk)| {
            for (slot, x) in dst_chunk.iter_mut().zip(src_chunk) {
                slot.write(f(x));
            }
        });
}

/// Allocate `len` slots — advised as huge-page candidates, untouched — let
/// `fill` write every one of them, and hand back the `Vec`.
///
/// The buffer is deliberately **not** prefaulted: `fill` is a parallel pass, so
/// its own workers fault their own pages concurrently. There is no serial fault
/// to hoist, and a prefault would only write the buffer twice — measured, and
/// rejected, in [`crate::core::alloc::resident_vec`]'s docs.
fn collect_filled<R, G>(len: usize, fill: G) -> Vec<R>
where
    G: FnOnce(&mut [MaybeUninit<R>]),
{
    let mut v = crate::core::alloc::resident_capacity::<R>(len);
    fill(&mut v.spare_capacity_mut()[..len]);
    // SAFETY: `resident_capacity(len)` reserved at least `len` slots, and `fill`
    // — every implementation of which lives in this module — writes each of the
    // first `len` exactly once. The elements are therefore initialized.
    unsafe { v.set_len(len) };
    v
}

/// View an initialized `&mut [R]` as uninitialized slots, so the `_into` forms
/// can run the same fill loop the allocating ones do.
///
/// `R: Copy` is what makes this sound to *expose*, not just to write: a `Copy`
/// type has no destructor, so overwriting a live element without dropping it
/// leaks nothing. (The bound is why `_into` takes `R: Copy` while the allocating
/// forms do not.)
fn as_uninit_mut<R: Copy>(dst: &mut [R]) -> &mut [MaybeUninit<R>] {
    let len = dst.len();
    // SAFETY: `MaybeUninit<R>` is guaranteed to have the same size, alignment
    // and ABI as `R`. Every element of `dst` is initialized, and an initialized
    // value is a valid `MaybeUninit<R>`, so the cast exposes no uninit memory.
    // The borrow is exclusive and `len` is unchanged.
    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<R>>(), len) }
}

/// `f(i, &mut out[i])` for every element of `out`, in parallel — the in-place
/// counterpart of [`map_indexed`], for rewriting a buffer that already exists.
///
/// Element `i` is passed to exactly one call, which sees only `i` and that
/// element, so nothing is re-associated and the result equals the sequential
/// `out.iter_mut().enumerate().for_each(f)` bit-for-bit.
pub fn for_each_mut<T, F>(out: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync + Send,
{
    if out.len() < SERIAL_THRESHOLD {
        out.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let g = map_grain(out.len());
    out.par_chunks_mut(g).enumerate().for_each(|(c, chunk)| {
        let base = c * g;
        for (j, x) in chunk.iter_mut().enumerate() {
            f(base + j, x);
        }
    });
}

/// Rows per staging chunk in [`map_rows_fold_in_order`]. Bounds the staging
/// buffer, nothing else: `combine` sees `0..n` in order whatever this is, so it
/// cannot change a single bit of the result.
const FOLD_CHUNK_ROWS: usize = 1 << 16;

/// **Parallel per-element compute, sequential in-order combine** — a float
/// reduction that is bit-identical to the sequential loop it replaces.
///
/// # The problem this solves
///
/// A float sum cannot be parallelized by splitting it: `+` is not associative,
/// so any re-bracketing changes the bits, and this module refuses to offer such
/// a reduction ([see the module docs](self)). But an expensive per-element
/// *computation* whose result is merely *added* to an accumulator is a different
/// shape. Split it in two:
///
/// - `compute(scratch, i, row)` — the expensive part (a coordinate transform, an
///   interpolation, a Jacobian). It sees only element `i`, touches no
///   accumulator, and writes its contribution into `row`. **Runs in parallel.**
/// - `combine(i, row)` — the cheap part: the additions. **Runs on one thread, for
///   `i = 0, 1, 2, … n-1`, in order.**
///
/// So the accumulator sees exactly the sequence of values, in exactly the order,
/// that the original serial loop fed it. It is not a fold over per-chunk
/// partials — that *would* be a re-association. It is the *same fold*, over the
/// same elements, in the same order, with only the work that feeds it moved off
/// the critical path. The result is bit-identical to the sequential loop, at any
/// thread count, and identical to it — not merely reproducible.
///
/// `compute` returns `false` to mark element `i` invalid; `combine` is then not
/// called for it, exactly as a `continue` would have skipped it.
///
/// # Why a non-deterministic reduction stays unwritable
///
/// `combine` is `FnMut` and is never handed to rayon. A caller cannot get its
/// accumulator into a worker thread, so it cannot re-associate anything even by
/// accident. The parallel half (`compute`) is `Fn` and cannot own the
/// accumulator. The determinism is in the *shape* of the API.
///
/// # Cost
///
/// Amdahl, not accuracy: the combine stays serial, so the speedup is bounded by
/// how much of the per-element work sits in `compute`. That is the right trade
/// exactly when the compute dwarfs the additions — which is the case this exists
/// for.
pub fn map_rows_fold_in_order<S, I, C, F>(
    n: usize,
    width: usize,
    init: I,
    compute: C,
    mut combine: F,
) where
    S: Send,
    I: Fn() -> S + Sync + Send,
    C: Fn(&mut S, usize, &mut [f64]) -> bool + Sync + Send,
    F: FnMut(usize, &[f64]),
{
    assert!(width > 0, "a staged row needs at least one column");
    if n == 0 {
        return;
    }
    if n < SERIAL_THRESHOLD {
        let mut scratch = init();
        let mut row = vec![0.0; width];
        for i in 0..n {
            if compute(&mut scratch, i, &mut row) {
                combine(i, &row);
            }
        }
        return;
    }

    let chunk_rows = FOLD_CHUNK_ROWS.min(n);
    let mut rows = vec![0.0f64; chunk_rows * width];
    let mut valid = vec![false; chunk_rows];

    let mut start = 0usize;
    while start < n {
        let count = chunk_rows.min(n - start);

        // Parallel: every row is written by exactly one task, from its own index.
        rows[..count * width]
            .par_chunks_mut(width)
            .zip(valid[..count].par_iter_mut())
            .enumerate()
            .for_each_init(&init, |scratch, (r, (row, ok))| {
                *ok = compute(scratch, start + r, row);
            });

        // Sequential, in index order — the original loop's fold, untouched.
        for r in 0..count {
            if valid[r] {
                combine(start + r, &rows[r * width..(r + 1) * width]);
            }
        }
        start += count;
    }
}

/// The minimum and maximum of `vals` widened to `f64`, or `None` if empty.
///
/// Equals the sequential `lo = lo.min(v); hi = hi.max(v)` scan bit-for-bit: see
/// the module docs' associativity argument.
///
/// Generic over the *stored* type so a caller can scan an image's native buffer
/// directly. `Scalar::as_f64` is the same lossless widening `Image::to_f64_vec`
/// applies, so `min_max(img.scalar_slice::<f32>()?)` returns the identical bits
/// to `min_max(&img.to_f64_vec()?)` — without materializing the `f64` copy. (See
/// [`crate::core::fused`] for why that copy was the port's dominant cost.)
pub fn min_max<T: Scalar>(vals: &[T]) -> Option<(f64, f64)> {
    if vals.is_empty() {
        return None;
    }
    if vals.len() < SERIAL_THRESHOLD {
        return Some(fold_min_max(vals));
    }
    // Chunk boundaries depend on `vals.len()` alone (see `grain`). Partials are
    // collected in chunk order and combined left-to-right — the same bracketing on
    // every run and every thread count.
    let partials: Vec<(f64, f64)> = vals
        .par_chunks(reduce_grain(vals.len()))
        .map(fold_min_max)
        .collect::<Vec<_>>();
    Some(
        partials
            .into_iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (l, h)| {
                (lo.min(l), hi.max(h))
            }),
    )
}

fn fold_min_max<T: Scalar>(vals: &[T]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in vals {
        let v = v.as_f64();
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

/// Counts of `vals` per bin, where `bin_of(v)` yields the bin a value falls in,
/// or `None` for a value that is not counted at all.
///
/// `bin_of` is the caller's binning rule and must be a pure function of the one
/// value it is handed; an index `>= bins` panics.
///
/// Equals the sequential `frequency[bin_of(v)] += 1` loop bit-for-bit: `u64`
/// addition is exactly associative, so combining chunk-local histograms in chunk
/// order reproduces the serial counts exactly.
pub fn bin_counts<F>(vals: &[f64], bins: usize, bin_of: F) -> Vec<u64>
where
    F: Fn(f64) -> Option<usize> + Sync + Send,
{
    let fold = |chunk: &[f64]| {
        let mut counts = vec![0u64; bins];
        for &v in chunk {
            if let Some(bin) = bin_of(v) {
                counts[bin] += 1;
            }
        }
        counts
    };
    if vals.len() < SERIAL_THRESHOLD {
        return fold(vals);
    }
    // Chunk boundaries depend on `vals.len()` alone (see `grain`).
    let partials: Vec<Vec<u64>> = vals
        .par_chunks(reduce_grain(vals.len()))
        .map(fold)
        .collect();
    partials
        .into_iter()
        .fold(vec![0u64; bins], |mut acc, part| {
            for (a, p) in acc.iter_mut().zip(part) {
                *a += p;
            }
            acc
        })
}

/// One 1-D line through a buffer along the pass axis: `len` elements, `stride`
/// apart, starting at absolute buffer index [`Line::start`].
///
/// A line is handed to exactly one task, and lines never overlap, so
/// [`Line::get`]/[`Line::set`] can rewrite it in place — as the recursive
/// Gaussian does — without any cross-task interaction.
pub struct Line<'a, 'b, T> {
    kind: LineKind<'a, 'b, T>,
    start: usize,
    stride: usize,
    len: usize,
}

enum LineKind<'a, 'b, T> {
    /// The line lies inside one contiguous block, `stride` apart from `first`.
    Strided {
        block: &'a mut [T],
        first: usize,
        stride: usize,
    },
    /// The line's `k`-th element is `rows[k][col]` — used when the pass axis is
    /// the slowest one, whose lines cross every row of the buffer.
    Rows {
        rows: &'a mut [&'b mut [T]],
        col: usize,
    },
}

impl<T: Copy> Line<'_, '_, T> {
    /// Number of elements on the line (the pass axis's length).
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` only for a zero-length pass axis.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Absolute index, in the whole buffer, of the line's element `0`. With
    /// [`Line::stride`] this lets a caller read the *input* buffer at the same
    /// coordinates it is writing.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Absolute distance, in the whole buffer, between consecutive elements.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The line's `k`-th element.
    pub fn get(&self, k: usize) -> T {
        match &self.kind {
            LineKind::Strided {
                block,
                first,
                stride,
            } => block[first + k * stride],
            LineKind::Rows { rows, col } => rows[k][*col],
        }
    }

    /// Overwrites the line's `k`-th element.
    pub fn set(&mut self, k: usize, v: T) {
        match &mut self.kind {
            LineKind::Strided {
                block,
                first,
                stride,
            } => block[*first + k * *stride] = v,
            LineKind::Rows { rows, col } => rows[k][*col] = v,
        }
    }
}

/// Runs `f` on every 1-D line of `buf` along `axis`, in parallel.
///
/// `buf` is a dimension-0-fastest image buffer of shape `size`; a *line* is the
/// set of elements that differ only in their `axis` coordinate. Lines partition
/// the buffer, so `f` may rewrite its line in place. `init` builds one scratch
/// value per task, reused across that task's lines (the recursive Gaussian needs
/// three line-length scratch buffers and would otherwise allocate per line).
///
/// Bit-identical to the sequential line loop: no state crosses lines, and the
/// order within a line is `f`'s own.
pub fn for_each_line_mut<T, S, I, F>(buf: &mut [T], size: &[usize], axis: usize, init: I, f: F)
where
    T: Copy + Send + Sync,
    S: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, Line<'_, '_, T>) + Sync + Send,
{
    let n = size[axis];
    if n == 0 || buf.is_empty() {
        return;
    }
    // Elements between consecutive samples on the pass axis, and the contiguous
    // block that holds a whole set of lines: index = o * block + k * inner + i,
    // with o < outer, k < n (the pass axis), i < inner.
    let inner: usize = size[..axis].iter().product();
    let block = n * inner;
    let outer = buf.len() / block;

    let blocks_per_task = line_blocks_per_task(buf.len(), block);
    let block_tasks = outer.div_ceil(blocks_per_task);

    if buf.len() < SERIAL_THRESHOLD || (block_tasks < MIN_BLOCK_TASKS && inner == 1) {
        let mut scratch = init();
        for o in 0..outer {
            let base = o * block;
            for i in 0..inner {
                let line = Line {
                    kind: LineKind::Strided {
                        block: &mut buf[base..base + block],
                        first: i,
                        stride: inner,
                    },
                    start: base + i,
                    stride: inner,
                    len: n,
                };
                f(&mut scratch, line);
            }
        }
        return;
    }

    if block_tasks >= MIN_BLOCK_TASKS {
        // Enough blocks to keep the pool busy: hand each task a run of whole
        // contiguous blocks. Task boundaries follow from `size` alone.
        buf.par_chunks_mut(block * blocks_per_task)
            .enumerate()
            .for_each_init(&init, |scratch, (t, chunk)| {
                let first_block = t * blocks_per_task;
                for (b, blk) in chunk.chunks_mut(block).enumerate() {
                    let base = (first_block + b) * block;
                    for i in 0..inner {
                        let line = Line {
                            kind: LineKind::Strided {
                                block: blk,
                                first: i,
                                stride: inner,
                            },
                            start: base + i,
                            stride: inner,
                            len: n,
                        };
                        f(scratch, line);
                    }
                }
            });
        return;
    }

    // Too few blocks (the pass axis is the slowest one, so `outer` is small).
    // Split each block's lines by column instead: column `i` of every row is one
    // line, and a contiguous column range of every row is a disjoint set of
    // lines. `split_at_mut` proves the disjointness to the compiler — no `unsafe`.
    let col_grain = inner.div_ceil(MAX_COLUMN_TASKS).max(1);
    for o in 0..outer {
        let base = o * block;
        let blk = &mut buf[base..base + block];
        let ncols = inner.div_ceil(col_grain);
        let mut tasks: Vec<Vec<&mut [T]>> = (0..ncols).map(|_| Vec::with_capacity(n)).collect();
        for row in blk.chunks_mut(inner) {
            let mut rest = row;
            for task in tasks.iter_mut() {
                let take = rest.len().min(col_grain);
                let (head, tail) = rest.split_at_mut(take);
                task.push(head);
                rest = tail;
            }
        }
        tasks
            .into_par_iter()
            .enumerate()
            .for_each_init(&init, |scratch, (c, mut rows)| {
                let width = rows[0].len();
                for j in 0..width {
                    let i = c * col_grain + j;
                    let line = Line {
                        kind: LineKind::Rows {
                            rows: &mut rows,
                            col: j,
                        },
                        start: base + i,
                        stride: inner,
                        len: n,
                    };
                    f(scratch, line);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strides(size: &[usize]) -> Vec<usize> {
        let mut s = vec![1usize; size.len()];
        for d in 1..size.len() {
            s[d] = s[d - 1] * size[d - 1];
        }
        s
    }

    /// Every line pass must visit every element exactly once, with the absolute
    /// `start`/`stride` it advertises — on both the block path and the
    /// column-split path, and for every axis.
    #[test]
    fn line_pass_covers_every_element_exactly_once_on_every_axis() {
        for size in [
            vec![64usize, 40, 33],
            vec![37usize, 61, 45],
            vec![256usize, 3],
            vec![5usize, 5, 5],
        ] {
            let n: usize = size.iter().product();
            let strides = strides(&size);
            for axis in 0..size.len() {
                let mut buf: Vec<f64> = vec![0.0; n];
                for_each_line_mut(
                    &mut buf,
                    &size,
                    axis,
                    || (),
                    |(), mut line| {
                        assert_eq!(line.stride(), strides[axis]);
                        assert_eq!(line.len(), size[axis]);
                        for k in 0..line.len() {
                            // Stamp the absolute index this slot must own.
                            line.set(k, (line.start() + k * line.stride()) as f64);
                        }
                    },
                );
                let expected: Vec<f64> = (0..n).map(|i| i as f64).collect();
                assert_eq!(buf, expected, "size {size:?} axis {axis}");
            }
        }
    }

    /// An in-place line rewrite that reads what it writes: bit-identical to the
    /// same recurrence run serially, on every axis and both decomposition paths.
    #[test]
    fn in_place_line_recurrence_matches_the_serial_pass() {
        let size = vec![48usize, 39, 27];
        let n: usize = size.iter().product();
        let strides = strides(&size);
        let src: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 * 0.001).collect();

        for axis in 0..size.len() {
            // Serial reference: a prefix recurrence along the axis, in place.
            let mut want = src.clone();
            let stride = strides[axis];
            let ln = size[axis];
            let outer: usize = n / (ln * stride);
            for o in 0..outer {
                for i in 0..stride {
                    let start = o * ln * stride + i;
                    let mut acc = 0.0f64;
                    for k in 0..ln {
                        acc = 0.5 * acc + want[start + k * stride];
                        want[start + k * stride] = acc;
                    }
                }
            }

            let mut got = src.clone();
            for_each_line_mut(
                &mut got,
                &size,
                axis,
                || (),
                |(), mut line| {
                    let mut acc = 0.0f64;
                    for k in 0..line.len() {
                        acc = 0.5 * acc + line.get(k);
                        line.set(k, acc);
                    }
                },
            );
            assert_eq!(got, want, "axis {axis}");
        }
    }

    #[test]
    fn for_each_mut_rewrites_every_element_with_its_own_index() {
        let n = SERIAL_THRESHOLD * 4 + 9;
        let mut got: Vec<f64> = vec![0.0; n];
        for_each_mut(&mut got, |i, x| *x = (i as f64) * 0.25);
        let want: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn map_indexed_matches_the_serial_map_past_the_parallel_threshold() {
        let n = SERIAL_THRESHOLD * 4 + 7;
        let got = map_indexed(n, |i| (i as f64).sqrt());
        let want: Vec<f64> = (0..n).map(|i| (i as f64).sqrt()).collect();
        assert_eq!(got, want);
    }

    /// The bench-spec volumes, in elements.
    const SMALL: usize = 64 * 64 * 64; // 262_144
    const MEDIUM: usize = 256 * 256 * 256; // 16_777_216
    const LARGE: usize = 512 * 512 * 512; // 134_217_728

    /// The chunk starts `par_chunks(g)` emits for `len` elements.
    fn boundaries(len: usize, g: usize) -> Vec<usize> {
        (0..len.div_ceil(g)).map(|c| c * g).collect()
    }

    /// The whole determinism argument, as a test: the grain is a function of `len`
    /// and of nothing else. It cannot observe the thread count — there is no
    /// argument through which it could — so this pins the *values*, which is the
    /// part a future edit could get wrong.
    #[test]
    fn the_grain_is_a_pure_function_of_the_length() {
        for &len in &[0, 1, SERIAL_THRESHOLD, SMALL, MEDIUM, LARGE, LARGE * 3 + 7] {
            let (m, r) = (map_grain(len), reduce_grain(len));
            for _ in 0..4 {
                assert_eq!(map_grain(len), m, "map_grain({len}) is not deterministic");
                assert_eq!(
                    reduce_grain(len),
                    r,
                    "reduce_grain({len}) is not deterministic"
                );
            }
            assert!((MIN_GRAIN..=GRAIN).contains(&m), "map_grain({len}) = {m}");
            assert!(
                (MIN_GRAIN..=REDUCE_CHUNK).contains(&r),
                "reduce_grain({len}) = {r}"
            );
        }
    }

    /// The defect this rule closes: a 64³ volume handed a 96-worker pool four
    /// chunks. Both passes must now raise more tasks than the box has workers.
    #[test]
    fn a_small_volume_raises_more_tasks_than_a_wide_box_has_workers() {
        assert_eq!(reduce_grain(SMALL), 2048);
        assert_eq!(SMALL.div_ceil(reduce_grain(SMALL)), 128); // was 4, at a 65536 grain
        assert_eq!(map_grain(SMALL), 2048);
        assert_eq!(SMALL.div_ceil(map_grain(SMALL)), 128); // was 64, at a 4096 grain
    }

    /// The cost-class runs a cubic stencil pass emits, built the way
    /// `NeighborhoodIterator::cost_runs` builds them: a row is checked whenever a
    /// coordinate above dimension 0 is within `radius` of an edge.
    fn stencil_runs(side: usize, radius: usize) -> Vec<CostRun> {
        let mut runs: Vec<CostRun> = Vec::new();
        for row in 0..side * side {
            let (y, z) = (row % side, row / side);
            let checked = [y, z].iter().any(|&c| c < radius || c + radius >= side);
            let class = u8::from(checked);
            match runs.last_mut() {
                Some(run) if run.class == class => run.len += side,
                _ => runs.push(CostRun { len: side, class }),
            }
        }
        runs
    }

    /// Which run of the partition index `at` falls in.
    fn run_at(runs: &[CostRun], at: usize) -> (CostRun, usize) {
        let mut seen = 0usize;
        for r in runs {
            seen += r.len;
            if at < seen {
                return (*r, seen);
            }
        }
        panic!("index {at} is outside the partition");
    }

    /// **The straggler, as integers.** The equal-count chunker gives one task the whole
    /// `z = 0` plane; the cost-class chunker cannot, because it never cuts across a class
    /// and it splits each class by that class's own total.
    ///
    /// This is the defect `occupancy-result.md` measured — `c_max/wall` = 0.93–1.00, the
    /// makespan *being* one boundary chunk — restated as the property that caused it, so
    /// an edit that reintroduces it fails here rather than in a benchmark months later.
    #[test]
    fn no_chunk_holds_more_checked_pixels_than_the_checked_class_grain() {
        let (side, radius) = (64usize, 2usize);
        let runs = stencil_runs(side, radius);
        let checked: usize = runs.iter().filter(|r| r.class == 1).map(|r| r.len).sum();
        let class_grain = grain_floored(checked, GRAIN, 1);

        let chunks = cost_chunks(&runs);
        assert_eq!(chunks.iter().sum::<usize>(), side * side * side);

        let mut at = 0usize;
        let mut worst_checked = 0usize;
        for len in &chunks {
            let (run, run_end) = run_at(&runs, at);
            assert!(
                at + len <= run_end,
                "a chunk straddles two cost classes at index {at}"
            );
            if run.class == 1 {
                worst_checked = worst_checked.max(*len);
            }
            at += len;
        }
        assert!(
            worst_checked <= class_grain,
            "a chunk holds {worst_checked} checked pixels, above the class grain {class_grain}"
        );

        // NON-VACUITY: the old equal-count rule really does hand one task a chunk that is
        // *entirely* checked, or this pin guards against nothing. At 64³ the first 4096
        // indices are the z = 0 plane, and a 2048 grain puts 2048 of them — all checked —
        // in chunk 0, which is 16x the checked pixels the class rule allows in one chunk.
        let flat = map_grain(side * side * side);
        assert_eq!(flat, MIN_GRAIN);
        assert!(
            flat <= side * side,
            "chunk 0 of the flat rule must be all-checked, or this pin is vacuous"
        );
        assert!(
            flat > class_grain,
            "the flat rule must give a bigger checked chunk ({flat}) than the class rule \
             ({class_grain}), or the split changes nothing"
        );
    }

    /// Each class is split by its *own* total, which is what bounds `c_max` without ever
    /// comparing the two costs — the property that keeps a cost constant out of this
    /// module. The task count is bounded by the target plus one remainder per run.
    #[test]
    fn each_cost_class_is_split_into_at_most_the_task_target() {
        let runs = stencil_runs(64, 2);
        let chunks = cost_chunks(&runs);

        for class in [0u8, 1] {
            let runs_of_class = runs.iter().filter(|r| r.class == class).count();
            let mut at = 0usize;
            let mut count = 0usize;
            for len in &chunks {
                if run_at(&runs, at).0.class == class {
                    count += 1;
                }
                at += len;
            }
            assert!(
                count <= TARGET_TASKS + runs_of_class,
                "class {class}: {count} chunks, above the target {TARGET_TASKS} plus one \
                 remainder per run ({runs_of_class})"
            );
        }
    }

    /// A one-class partition is the module's own grain rule, ceiling and all — the
    /// cost-class chunker must not change a pass whose cost per index really is uniform,
    /// and it must not emit chunks the flat rule would have refused.
    ///
    /// The ceiling is the half of the rule this test exists for. Dropping it measured
    /// **1.41× slower** on `binary_dilate` at 256³: a 16.7 M class total asks for a
    /// 65 536-element chunk, 16× what [`GRAIN`] allows, and cost-homogeneous chunks that
    /// large are worse than heterogeneous small ones. Only the *floor* is dropped.
    #[test]
    fn a_single_class_partition_is_the_flat_grain_rule_without_the_floor() {
        for &len in &[SMALL, MEDIUM, LARGE] {
            let chunks = cost_chunks(&[CostRun { len, class: 0 }]);
            let g = grain_floored(len, GRAIN, 1);
            assert_eq!(chunks.iter().sum::<usize>(), len);
            assert_eq!(chunks.len(), len.div_ceil(g));
            assert!(
                chunks.iter().all(|&c| c <= GRAIN),
                "a chunk exceeds the ceiling"
            );
        }

        // At and above the length where the elementwise floor stops binding, the one-class
        // partition emits exactly the flat rule's chunks — so this change cannot move a
        // uniform-cost pass at medium or large.
        for &len in &[TARGET_TASKS * MIN_GRAIN, MEDIUM, LARGE] {
            assert_eq!(
                cost_chunks(&[CostRun { len, class: 0 }]).len(),
                len.div_ceil(map_grain(len)),
                "the one-class partition differs from the flat rule at len = {len}"
            );
        }

        // Non-vacuity: at 64³ the floor DOES bind the flat rule (2048) while the partition
        // asks for 1024, so the equality above is not a tautology.
        assert_eq!(map_grain(SMALL), MIN_GRAIN);
        assert_ne!(
            cost_chunks(&[CostRun {
                len: SMALL,
                class: 0
            }])
            .len(),
            SMALL.div_ceil(map_grain(SMALL))
        );
    }

    /// The three bench volumes, cubic, as `for_each_line_mut` sees them.
    fn cube(n: usize) -> Vec<usize> {
        vec![n, n, n]
    }

    /// The line pass's block geometry for one `(size, axis)`: `(block, outer)`.
    /// `index = o * block + k * inner + i`, exactly as `for_each_line_mut` derives
    /// it — a line lies inside one block, and a block holds `inner` of them.
    fn block_geometry(size: &[usize], axis: usize) -> (usize, usize) {
        let len: usize = size.iter().product();
        let inner: usize = size[..axis].iter().product();
        let block = size[axis] * inner;
        (block, len / block)
    }

    /// The **former** rule: a fixed `GRAIN` per task, whatever the input length.
    /// Frozen here as the baseline the new rule must reproduce above `medium`.
    fn fixed_grain_blocks_per_task(block: usize) -> usize {
        GRAIN.div_ceil(block).max(1)
    }

    /// **The line-pass rule is a no-op at and above `medium`** — the same integer
    /// proof that made the elementwise seam mergeable, applied to the pass that
    /// decomposes by whole blocks. `par_chunks_mut(block * blocks_per_task)` is
    /// what cuts the buffer, so its emitted chunk starts are the decomposition;
    /// they are compared **as integers**, on every axis, against the fixed-grain
    /// rule they replace.
    #[test]
    fn the_line_grain_emits_the_same_block_runs_as_the_fixed_grain_at_and_above_medium() {
        for n in [256usize, 512] {
            let size = cube(n);
            let len: usize = size.iter().product();
            for axis in 0..3 {
                let (block, _) = block_geometry(&size, axis);
                let new = line_blocks_per_task(len, block);
                let old = fixed_grain_blocks_per_task(block);
                assert_eq!(new, old, "{n}³ axis {axis}: the block run moved");
                assert_eq!(
                    boundaries(len, block * new),
                    boundaries(len, block * old),
                    "{n}³ axis {axis}: the emitted chunk starts moved"
                );
            }
        }

        // Non-vacuity. If the two rules agreed *everywhere*, the assertions above
        // would be comparing a rule with itself and would prove nothing about the
        // no-op. They must diverge at 64³ along axis 0 — the one shape this change
        // exists to move — or this test is asserting a tautology.
        let size = cube(64);
        let (block, _) = block_geometry(&size, 0);
        assert_ne!(
            line_blocks_per_task(SMALL, block),
            fixed_grain_blocks_per_task(block),
            "the new rule matches the old one even at 64³ axis 0 — nothing changed, \
             so the no-op proved above is vacuous"
        );
        assert_ne!(
            boundaries(SMALL, block * line_blocks_per_task(SMALL, block)),
            boundaries(SMALL, block * fixed_grain_blocks_per_task(block)),
        );
    }

    /// What the new grain is worth on the line pass, stated as the task counts it
    /// raises — and what it **cannot** reach.
    ///
    /// Axis 0 is the shape the grain governs: its blocks are smaller than the
    /// grain, so a task takes a run of them, and cutting the run from the length
    /// doubles the tasks (64 → 128).
    #[test]
    fn the_line_pass_raises_more_tasks_at_the_small_volume() {
        let size = cube(64);
        let (block, outer) = block_geometry(&size, 0);
        let was = outer.div_ceil(fixed_grain_blocks_per_task(block));
        let now = outer.div_ceil(line_blocks_per_task(SMALL, block));
        assert_eq!(was, 64);
        assert_eq!(now, 128);
    }

    /// **The residual this rule does not close, pinned so it is not forgotten.**
    ///
    /// A line lies inside one block, so a task takes *whole blocks* — which caps
    /// the block path at `outer` tasks no matter how fine the grain gets. At 64³
    /// along axis 1 a block is 4096 elements, already larger than any grain the
    /// rule can emit, so the run length is 1 and the pass raises exactly `outer` =
    /// 64 tasks — fewer than this box's 96 workers, before and after this change.
    ///
    /// Closing it needs a decomposition that splits a block by column, not a
    /// constant. This asserts the cap exists rather than papering over it.
    #[test]
    fn the_line_pass_task_count_is_capped_by_the_block_count_not_by_the_grain() {
        let size = cube(64);
        let (block, outer) = block_geometry(&size, 1);
        assert_eq!((block, outer), (4096, 64));

        // Every grain the rule can emit, and the finest it could ever emit: the
        // run length stays 1, so the task count stays `outer`.
        for g in [MIN_GRAIN, GRAIN, 1usize] {
            assert_eq!(g.div_ceil(block).max(1), 1, "grain {g} split a block");
        }
        assert_eq!(outer.div_ceil(line_blocks_per_task(SMALL, block)), 64);
        assert!(
            outer < 96,
            "the cap this test pins is gone — a 64³ axis-1 line pass now raises \
             {outer} tasks; re-derive the residual"
        );
    }

    /// `MIN_BLOCK_TASKS` selects a *path*; it is not a floor on the task count,
    /// and raising it to the pool's width does not feed the pool — it starves it.
    ///
    /// For an `inner == 1` shape (the axis-0 pass) there is no column path to fall
    /// back to, so `block_tasks < MIN_BLOCK_TASKS` means **serial**. The 64³ axis-0
    /// pass raises 128 tasks; a `MIN_BLOCK_TASKS` above that sends all 128 down one
    /// thread. This test is what fails if someone "raises the floor to match the 96
    /// workers", which is the natural-looking move and is a 96× regression.
    #[test]
    fn min_block_tasks_never_sends_a_bench_volume_line_pass_serial() {
        for n in [64usize, 256, 512] {
            let size = cube(n);
            let len: usize = size.iter().product();
            let (block, outer) = block_geometry(&size, 0);
            let inner: usize = size[..0].iter().product();
            assert_eq!(inner, 1, "axis 0's lines are its blocks");
            let block_tasks = outer.div_ceil(line_blocks_per_task(len, block));
            assert!(
                block_tasks >= MIN_BLOCK_TASKS,
                "{n}³ axis 0 raises {block_tasks} tasks but MIN_BLOCK_TASKS is \
                 {MIN_BLOCK_TASKS}: this shape has no column path, so it would run \
                 SERIAL. MIN_BLOCK_TASKS is a path selector, not a task floor — feed \
                 the pool through `line_blocks_per_task` instead."
            );
        }
    }

    /// **The rule is a no-op at and above `TARGET_TASKS * ceiling` elements** — not
    /// "a small change", none: the emitted chunk boundaries are the same integers
    /// the fixed grain emitted. Both bench volumes at or above `medium` are in that
    /// range for both passes, so their decomposition, and every checksum folded
    /// over it, is untouched by construction.
    #[test]
    fn the_rule_emits_the_same_boundaries_as_the_fixed_grain_at_and_above_medium() {
        for &len in &[MEDIUM, LARGE] {
            assert_eq!(
                reduce_grain(len),
                REDUCE_CHUNK,
                "reduce grain moved at {len}"
            );
            assert_eq!(map_grain(len), GRAIN, "map grain moved at {len}");
            assert_eq!(
                boundaries(len, reduce_grain(len)),
                boundaries(len, REDUCE_CHUNK)
            );
            assert_eq!(boundaries(len, map_grain(len)), boundaries(len, GRAIN));
        }
        // The exact crossover, both sides, for each ceiling. `div_ceil` rounds up,
        // so the grain is already pinned at the ceiling for every length above
        // `TARGET_TASKS * (ceiling - 1)` — that product, not `TARGET_TASKS *
        // ceiling - 1`, is the last length the rule still subdivides.
        assert_eq!(reduce_grain(TARGET_TASKS * REDUCE_CHUNK), REDUCE_CHUNK);
        assert_eq!(
            reduce_grain(TARGET_TASKS * (REDUCE_CHUNK - 1)),
            REDUCE_CHUNK - 1
        );
        assert_eq!(map_grain(TARGET_TASKS * GRAIN), GRAIN);
        assert_eq!(map_grain(TARGET_TASKS * (GRAIN - 1)), GRAIN - 1);
    }

    /// A finer grain re-brackets the fold, so this is the test that says the
    /// re-bracketing is invisible: the two reductions return the same bits at every
    /// grain the rule can emit, because `min`/`max` select an input element and
    /// `u64` addition is exact.
    #[test]
    fn both_reductions_are_bit_identical_at_every_grain_the_rule_can_emit() {
        let n = SMALL;
        let vals: Vec<f64> = (0..n)
            .map(|i| ((i * 2654435761usize) % 100_003) as f64 * 0.5 - 1000.0)
            .collect();
        let bins = 128usize;
        let bin_of = |v: f64| Some((((v + 1000.0) / 25.0) as usize).min(bins - 1));

        let want_mm = fold_min_max(&vals);
        let mut want_bc = vec![0u64; bins];
        for &v in &vals {
            want_bc[bin_of(v).unwrap()] += 1;
        }

        for g in [1024usize, 2048, 4096, 8192, 65536, n, n * 2] {
            let mm = vals
                .par_chunks(g)
                .map(fold_min_max)
                .collect::<Vec<_>>()
                .into_iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (l, h)| {
                    (lo.min(l), hi.max(h))
                });
            assert_eq!(
                mm.0.to_bits(),
                want_mm.0.to_bits(),
                "min moved at grain {g}"
            );
            assert_eq!(
                mm.1.to_bits(),
                want_mm.1.to_bits(),
                "max moved at grain {g}"
            );

            let bc = vals
                .par_chunks(g)
                .map(|chunk| {
                    let mut c = vec![0u64; bins];
                    for &v in chunk {
                        c[bin_of(v).unwrap()] += 1;
                    }
                    c
                })
                .collect::<Vec<_>>()
                .into_iter()
                .fold(vec![0u64; bins], |mut acc, part| {
                    for (a, p) in acc.iter_mut().zip(part) {
                        *a += p;
                    }
                    acc
                });
            assert_eq!(bc, want_bc, "the histogram moved at grain {g}");
        }
    }

    #[test]
    fn min_max_matches_the_serial_scan_bit_for_bit() {
        let n = SERIAL_THRESHOLD * 8 + 13;
        let vals: Vec<f64> = (0..n)
            .map(|i| ((i * 2654435761usize) % 100_003) as f64 * 0.5 - 1000.0)
            .collect();
        let want = fold_min_max(&vals);
        assert_eq!(min_max(&vals), Some(want));
        assert_eq!(min_max::<f64>(&[]), None);
    }

    /// A NaN in the data must not poison the accumulator on either path — the
    /// serial scan skips it, and so must the chunked one.
    #[test]
    fn min_max_ignores_nan_the_same_way_the_serial_scan_does() {
        let n = SERIAL_THRESHOLD * 4;
        let mut vals: Vec<f64> = (0..n).map(|i| (i % 977) as f64).collect();
        vals[0] = f64::NAN;
        vals[n / 2] = f64::NAN;
        vals[n - 1] = f64::NAN;
        let (lo, hi) = min_max(&vals).unwrap();
        assert_eq!((lo, hi), fold_min_max(&vals));
        assert_eq!((lo, hi), (0.0, 976.0));
    }

    #[test]
    fn bin_counts_matches_the_serial_histogram() {
        let bins = 128usize;
        let n = SERIAL_THRESHOLD * 8 + 5;
        let vals: Vec<f64> = (0..n).map(|i| ((i * 48271) % 1000) as f64).collect();
        let bin_of = |v: f64| Some(((v / 1000.0) * bins as f64) as usize);
        let got = bin_counts(&vals, bins, bin_of);
        let mut want = vec![0u64; bins];
        for &v in &vals {
            want[bin_of(v).unwrap()] += 1;
        }
        assert_eq!(got, want);
        assert_eq!(got.iter().sum::<u64>(), n as u64);
    }

    /// The `None` arm must drop a value from every bin, not fold it into one —
    /// the range-guarded histogram (`Histogram::from_bounds`) depends on it.
    #[test]
    fn bin_counts_skips_the_values_the_rule_rejects() {
        let bins = 4usize;
        let n = SERIAL_THRESHOLD * 4;
        let vals: Vec<f64> = (0..n).map(|i| (i % 8) as f64).collect();
        let got = bin_counts(&vals, bins, |v| (v < 4.0).then_some(v as usize));
        assert_eq!(got, vec![(n / 8) as u64; 4]);
        assert_eq!(got.iter().sum::<u64>(), (n / 2) as u64);
    }

    /// The reason the primitive exists: the accumulator must see the same
    /// additions in the same order as the loop it replaces, at *any* thread
    /// count. The values make re-association visible in the bits — the leading
    /// `1.0` swallows every subsequent `1e-17` by rounding, so a sum that
    /// brackets the tail separately (a chunked fold, a tree reduce) lands on a
    /// different `f64`. The sequential sum here is 1.0 exactly; any parallel
    /// re-association gets 1.0 + something.
    #[test]
    fn map_rows_fold_in_order_is_bit_identical_to_the_sequential_fold() {
        let sample = |i: usize| if i == 0 { 1.0f64 } else { 1e-17 };
        // 0 and 1 are the empty/degenerate paths, 999 takes the serial path,
        // and the last straddles two staging chunks with a partial tail.
        for n in [0usize, 1, 999, FOLD_CHUNK_ROWS * 2 + 137] {
            let mut want = 0.0f64;
            for i in 0..n {
                want += sample(i);
            }
            if n > 1 {
                assert_eq!(want, 1.0, "the tail must vanish into the leading 1.0");
            }

            for threads in [1usize, 2, 3, 7, 16, 64] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut got = 0.0f64;
                let mut order = Vec::with_capacity(n);
                pool.install(|| {
                    map_rows_fold_in_order(
                        n,
                        1,
                        || (),
                        |(), i, row| {
                            row[0] = sample(i);
                            true
                        },
                        |i, row| {
                            order.push(i);
                            got += row[0];
                        },
                    );
                });
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "n={n}, {threads} threads moved the bits"
                );
                assert_eq!(
                    order,
                    (0..n).collect::<Vec<_>>(),
                    "combine ran out of order"
                );
            }
        }
    }

    /// An invalid element must be skipped entirely, exactly as `continue` would
    /// have skipped it — not combined as a zero row.
    #[test]
    fn map_rows_fold_in_order_skips_invalid_rows_and_carries_every_column() {
        let n = SERIAL_THRESHOLD * 4 + 3;
        let width = 3;
        let ok = |i: usize| !i.is_multiple_of(3);

        let mut got = vec![0.0f64; width];
        let mut combined = 0usize;
        map_rows_fold_in_order(
            n,
            width,
            || (),
            |(), i, row| {
                if !ok(i) {
                    return false;
                }
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = (i * (c + 1)) as f64;
                }
                true
            },
            |_, row| {
                combined += 1;
                for (acc, &v) in got.iter_mut().zip(row) {
                    *acc += v;
                }
            },
        );

        let mut want = vec![0.0f64; width];
        let mut want_combined = 0usize;
        for i in (0..n).filter(|&i| ok(i)) {
            want_combined += 1;
            for (c, acc) in want.iter_mut().enumerate() {
                *acc += (i * (c + 1)) as f64;
            }
        }
        assert_eq!(combined, want_combined);
        assert_eq!(got, want);
    }
}
