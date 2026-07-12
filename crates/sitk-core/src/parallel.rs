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
//! - Chunk boundaries are private constants applied to the input length. No
//!   entry point takes a chunk count, a grain size, or a thread count, so no
//!   decomposition in the port can depend on `rayon::current_num_threads()`.
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
//! [`SERIAL_THRESHOLD`] returns the same bits as the parallel path — the
//! threshold is a speed knob, never a correctness one.

use rayon::prelude::*;

/// Target elements per worker task. Only a scheduling knob: every primitive
/// here is exact, so changing it cannot change a result.
const GRAIN: usize = 4096;

/// Below this many elements, run serially — thread hand-off costs more than the
/// work. Bit-identical to the parallel path by the exactness argument above.
const SERIAL_THRESHOLD: usize = 1 << 14;

/// Elements per reduction chunk. Fixed, so the chunk count is
/// `input.len().div_ceil(REDUCE_CHUNK)` — a pure function of the input length,
/// never of the thread count.
const REDUCE_CHUNK: usize = 1 << 16;

/// A line pass parallelizes over blocks only if it can raise at least this many
/// of them; otherwise it splits the block's lines column-wise instead.
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
pub fn map_indexed<R, F>(len: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if len < SERIAL_THRESHOLD {
        return (0..len).map(f).collect();
    }
    (0..len)
        .into_par_iter()
        .with_min_len(GRAIN)
        .map(f)
        .collect()
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
    if src.len() < SERIAL_THRESHOLD {
        return src.iter().map(f).collect();
    }
    src.par_iter().with_min_len(GRAIN).map(f).collect()
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
    if len < SERIAL_THRESHOLD {
        let mut scratch = init();
        return (0..len).map(|i| f(&mut scratch, i)).collect();
    }
    (0..len)
        .into_par_iter()
        .with_min_len(GRAIN)
        .map_init(init, |scratch, i| f(scratch, i))
        .collect()
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
    out.par_chunks_mut(GRAIN)
        .enumerate()
        .for_each(|(c, chunk)| {
            let base = c * GRAIN;
            for (j, x) in chunk.iter_mut().enumerate() {
                f(base + j, x);
            }
        });
}

/// The minimum and maximum of `vals`, or `None` if empty.
///
/// Equals the sequential `lo = lo.min(v); hi = hi.max(v)` scan bit-for-bit: see
/// the module docs' associativity argument.
pub fn min_max(vals: &[f64]) -> Option<(f64, f64)> {
    if vals.is_empty() {
        return None;
    }
    if vals.len() < SERIAL_THRESHOLD {
        return Some(fold_min_max(vals));
    }
    // Chunk boundaries depend on `vals.len()` alone. Partials are collected in
    // chunk order and combined left-to-right — the same bracketing on every run
    // and every thread count.
    let partials: Vec<(f64, f64)> = vals
        .par_chunks(REDUCE_CHUNK)
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

fn fold_min_max(vals: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in vals {
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
    let partials: Vec<Vec<u64>> = vals.par_chunks(REDUCE_CHUNK).map(fold).collect();
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

    let blocks_per_task = GRAIN.div_ceil(block).max(1);
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

    #[test]
    fn min_max_matches_the_serial_scan_bit_for_bit() {
        let n = SERIAL_THRESHOLD * 8 + 13;
        let vals: Vec<f64> = (0..n)
            .map(|i| ((i * 2654435761usize) % 100_003) as f64 * 0.5 - 1000.0)
            .collect();
        let want = fold_min_max(&vals);
        assert_eq!(min_max(&vals), Some(want));
        assert_eq!(min_max(&[]), None);
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
}
