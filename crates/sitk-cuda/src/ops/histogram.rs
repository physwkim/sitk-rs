//! A **deterministic** weighted histogram on the device — the reduction a device
//! Mattes mutual information would have to be built on, and the reason it is not built
//! yet.
//!
//! # Why this exists
//!
//! The natural GPU histogram is `atomicAdd(&hist[bin], w)`. Integer atomics are exact
//! and order-independent, so an *unweighted* histogram is deterministic for free.
//! Floating-point atomics are not: `atomicAdd` gives no defined summation order, and
//! double addition is not associative, so the same binary on the same input produces
//! **different bits on different runs**. Fed to `RegularStepGradientDescentOptimizer` —
//! which halves its step on a comparison — the same binary sends *itself* to two
//! different poses. A metric like that cannot be pinned: a test that cannot fail when
//! the code is wrong is not a test.
//!
//! So the reduction is built first, on its own, and it is held to the standard the
//! metric will need:
//!
//! - **run-to-run determinism**: same binary, same input, same bits, every run;
//! - **bit-identity with the host**: the device histogram is *exactly* what
//!   `for i in 0..n { hist[key[i]] += value[i] }` computes in `f64` on the CPU — not
//!   close to it.
//!
//! # How
//!
//! The order is what has to be pinned, so the order is made explicit: the entries are
//! **stably sorted by their bin** and each bin's segment is then summed **in ascending
//! entry index**. That sequence is a function of the input alone — not of the block
//! size, not of the grid size, not of which SM ran which block — and it is the same
//! sequence the naive host loop performs, which is why the two agree on the bits.
//!
//! The sort is a bin-keyed counting sort, deterministic at every step:
//!
//! 1. `count_tiles` — per-tile, per-bin counts, with **integer** `atomicAdd`. Exact and
//!    order-independent: the count does not care who got there first.
//! 2. `bin_totals` / `bin_offsets` / `tile_cursors` — where every (tile, bin) group's
//!    entries begin. All integer arithmetic, all sequential in a defined order.
//! 3. `scatter_stable` — one block per tile; the block walks its chunk in fixed
//!    sub-tiles, and inside a sub-tile each entry's rank is *counted* (how many
//!    lower-index entries in this sub-tile share my bin) rather than claimed from an
//!    atomic counter. A rank claimed from an atomic is whoever-arrives-first; a rank
//!    counted is the entry's index. That is the whole difference between this and a
//!    fast histogram.
//! 4. `segment_sums` — one thread per bin, summing its segment left to right.
//!
//! [`histogram_atomic`] is the fast, wrong one. It is kept, and it is *tested*, because
//! the pin on determinism is only worth something if the thing it forbids can be
//! demonstrated to happen.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;

/// The number of contiguous chunks the entry list is cut into for the scatter — one
/// block per tile.
///
/// It is a **constant, not a tuning knob**, and that is deliberate: the summation order
/// this reduction guarantees is "ascending entry index within a bin", which the tile
/// decomposition must not be able to change. It cannot: tiles are contiguous ascending
/// ranges of the entry list, cursors are laid out in ascending tile order, and each
/// tile's entries land in ascending index order within their bin — so the sorted array,
/// and therefore the sum, is the same for any tile count. `the_result_does_not_depend_on_
/// the_launch_configuration` pins exactly that by varying the block size.
const TILES: usize = 2048;

/// Entries per thread-block sweep inside a tile. Must equal the scatter kernel's block
/// size — the kernel indexes `s_key[threadIdx.x]`.
const SUB: usize = 256;

/// One thread per bin, for the scan and segment-sum kernels.
const BINS_PER_BLOCK: usize = 128;

const HISTOGRAM: &str = r#"
#define SUB 256

// Per-tile, per-bin counts. `atomicAdd` on an INT is exact and order-independent, so
// this is deterministic even though nothing about the arrival order is.
extern "C" __global__ void count_tiles(
    const unsigned int* __restrict__ keys,
    int* __restrict__ tilecount,
    const long long n,
    const long long chunk,
    const int nbins,
    const int ntiles)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        long long t = i / chunk;
        if (t >= ntiles) t = ntiles - 1;
        atomicAdd(&tilecount[t * nbins + (long long)keys[i]], 1);
    }
}

// How many entries each bin has, over the whole list.
extern "C" __global__ void bin_totals(
    const int* __restrict__ tilecount,
    long long* __restrict__ total,
    const int nbins,
    const int ntiles)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbins) return;
    long long s = 0;
    for (int t = 0; t < ntiles; ++t) s += (long long)tilecount[(long long)t * nbins + b];
    total[b] = s;
}

// Where each bin's segment begins in the sorted array. A single thread walks the bins in
// order: nbins is small (a joint Mattes histogram is thousands of bins, not millions),
// and a parallel scan here would buy nothing and cost a second thing to keep correct.
extern "C" __global__ void bin_offsets(
    const long long* __restrict__ total,
    long long* __restrict__ offset,
    const int nbins)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    long long acc = 0;
    for (int b = 0; b < nbins; ++b) {
        offset[b] = acc;
        acc += total[b];
    }
}

// Where each (tile, bin) group begins: the bin's offset, plus everything earlier tiles
// put in that bin. This is what makes the scatter stable *across* tiles without the
// tiles having to run in any particular order.
extern "C" __global__ void tile_cursors(
    const int* __restrict__ tilecount,
    const long long* __restrict__ offset,
    long long* __restrict__ cursor,
    const int nbins,
    const int ntiles)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbins) return;
    long long acc = offset[b];
    for (int t = 0; t < ntiles; ++t) {
        cursor[(long long)t * nbins + b] = acc;
        acc += (long long)tilecount[(long long)t * nbins + b];
    }
}

// The stable scatter. One block per tile; the block sweeps its chunk in SUB-sized
// sub-tiles, in ascending order.
//
// The rank of an entry within its (sub-tile, bin) group is COUNTED -- "how many entries
// before me in this sub-tile have my bin" -- not claimed from an atomic counter. That is
// the line between this kernel and a fast one: an atomically-claimed rank is a function
// of who arrived first, and the whole point here is that nothing is.
extern "C" __global__ void scatter_stable(
    const unsigned int* __restrict__ keys,
    const double* __restrict__ vals,
    long long* __restrict__ cursor,
    double* __restrict__ sorted,
    const long long n,
    const long long chunk,
    const int nbins)
{
    const long long t = blockIdx.x;
    const long long begin = t * chunk;
    long long end = begin + chunk;
    if (end > n) end = n;

    __shared__ unsigned int s_key[SUB];
    __shared__ int s_rank[SUB];

    const int j = threadIdx.x;
    for (long long base = begin; base < end; base += SUB) {
        const long long left = end - base;
        const int m = (int)(left < (long long)SUB ? left : (long long)SUB);

        if (j < m) s_key[j] = keys[base + j];
        __syncthreads();

        if (j < m) {
            int r = 0;
            for (int q = 0; q < j; ++q) {
                if (s_key[q] == s_key[j]) ++r;
            }
            s_rank[j] = r;
        }
        __syncthreads();

        if (j < m) {
            const long long b = (long long)s_key[j];
            sorted[cursor[t * nbins + b] + (long long)s_rank[j]] = vals[base + j];
        }
        __syncthreads();

        // Advance the cursor once per bin present in this sub-tile: the LAST entry of a
        // bin knows how many there were (its own rank + 1), and it is the only thread
        // that writes, so no atomic and no race.
        if (j < m) {
            const unsigned int b = s_key[j];
            bool last = true;
            for (int q = j + 1; q < m; ++q) {
                if (s_key[q] == b) { last = false; break; }
            }
            if (last) cursor[t * nbins + (long long)b] += (long long)s_rank[j] + 1;
        }
        __syncthreads();
    }
}

// One thread per bin, summing its segment left to right. The segment is in ascending
// entry index, so this is exactly the host's `for i in 0..n { h[key[i]] += v[i] }`.
extern "C" __global__ void segment_sums(
    const double* __restrict__ sorted,
    const long long* __restrict__ offset,
    const long long* __restrict__ total,
    double* __restrict__ out,
    const int nbins)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbins) return;
    const long long o = offset[b];
    const long long c = total[b];
    double acc = 0.0;
    for (long long k = 0; k < c; ++k) acc += sorted[o + k];
    out[b] = acc;
}

// The fast, non-deterministic one. Kept on purpose: it is what the pin forbids, and a
// pin that forbids something unobservable is decoration.
extern "C" __global__ void hist_atomic(
    const unsigned int* __restrict__ keys,
    const double* __restrict__ vals,
    double* __restrict__ out,
    const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&out[keys[i]], vals[i]);
}
"#;

fn cfg(n: usize, block: usize) -> LaunchConfig {
    let block = block as u32;
    LaunchConfig {
        grid_dim: (n.div_ceil(block as usize).max(1) as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Reject what the kernels cannot index: mismatched lengths, an empty list, no bins, or
/// a key that names no bin (which would scatter outside the histogram).
fn check(keys: &[u32], values: &[f64], nbins: usize) -> Result<(), CudaError> {
    if keys.len() != values.len() {
        return Err(CudaError::HistogramShape(format!(
            "{} keys and {} values",
            keys.len(),
            values.len()
        )));
    }
    if keys.is_empty() {
        return Err(CudaError::HistogramShape("no entries".into()));
    }
    if nbins == 0 {
        return Err(CudaError::HistogramShape("no bins".into()));
    }
    if let Some(&k) = keys.iter().find(|&&k| k as usize >= nbins) {
        return Err(CudaError::HistogramShape(format!(
            "key {k} names no bin (the histogram has {nbins})"
        )));
    }
    Ok(())
}

/// The weighted histogram `out[keys[i]] += values[i]`, summed **in ascending `i` within
/// each bin** — deterministic run to run, and bit-identical to the same loop in `f64` on
/// the host.
///
/// See the [module docs](self) for why that guarantee is the whole point and what it
/// costs. Errors with [`CudaError::HistogramShape`] on mismatched lengths, an empty
/// entry list, zero bins, or a key outside `0..nbins`.
pub fn histogram(keys: &[u32], values: &[f64], nbins: usize) -> Result<Vec<f64>, CudaError> {
    check(keys, values, nbins)?;
    histogram_with_block(keys, values, nbins, SUB)
}

/// [`histogram`], with the count kernel's block size chosen by the caller.
///
/// This exists for one test — the one that asserts the result does **not** depend on it.
/// A reduction whose value moves with the launch configuration is a reduction whose
/// value moves with the machine.
pub fn histogram_with_block(
    keys: &[u32],
    values: &[f64],
    nbins: usize,
    block: usize,
) -> Result<Vec<f64>, CudaError> {
    check(keys, values, nbins)?;
    let backend: &Backend = backend()?;
    let n = keys.len();
    let ntiles = TILES.min(n.div_ceil(SUB)).max(1);
    let chunk = n.div_ceil(ntiles);

    let d_keys = DeviceBuffer::from_host(backend, keys)?;
    let d_vals = DeviceBuffer::from_host(backend, values)?;
    let mut tilecount = DeviceBuffer::<i32>::zeros(backend, ntiles * nbins)?;
    let mut total = DeviceBuffer::<i64>::zeros(backend, nbins)?;
    let mut offset = DeviceBuffer::<i64>::zeros(backend, nbins)?;
    let mut cursor = DeviceBuffer::<i64>::zeros(backend, ntiles * nbins)?;
    let mut sorted = DeviceBuffer::<f64>::zeros(backend, n)?;
    let mut out = DeviceBuffer::<f64>::zeros(backend, nbins)?;

    let (n_i, chunk_i, nbins_i, ntiles_i) = (n as i64, chunk as i64, nbins as i32, ntiles as i32);

    let f = backend.function_exact(HISTOGRAM, "count_tiles")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(d_keys.device())
        .arg(tilecount.device_mut())
        .arg(&n_i)
        .arg(&chunk_i)
        .arg(&nbins_i)
        .arg(&ntiles_i);
    // SAFETY: six parameters, six arguments, matching in order and type. The grid-stride
    // loop is bounded by `i < n`; `t` is clamped to `ntiles - 1` and `keys[i] < nbins` was
    // checked on the host, so `t * nbins + key` stays inside `tilecount`.
    unsafe { launch.launch(cfg(n, block))? };

    let f = backend.function_exact(HISTOGRAM, "bin_totals")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(tilecount.device())
        .arg(total.device_mut())
        .arg(&nbins_i)
        .arg(&ntiles_i);
    // SAFETY: four parameters, four arguments. One thread per bin, guarded by `b < nbins`;
    // the inner loop indexes `tilecount` inside `ntiles * nbins`.
    unsafe { launch.launch(cfg(nbins, BINS_PER_BLOCK))? };

    let f = backend.function_exact(HISTOGRAM, "bin_offsets")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(total.device())
        .arg(offset.device_mut())
        .arg(&nbins_i);
    // SAFETY: three parameters, three arguments. A single thread walks `0..nbins`.
    unsafe { launch.launch(cfg(1, 1))? };

    let f = backend.function_exact(HISTOGRAM, "tile_cursors")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(tilecount.device())
        .arg(offset.device())
        .arg(cursor.device_mut())
        .arg(&nbins_i)
        .arg(&ntiles_i);
    // SAFETY: five parameters, five arguments. One thread per bin, guarded by `b < nbins`;
    // both matrices are `ntiles * nbins`.
    unsafe { launch.launch(cfg(nbins, BINS_PER_BLOCK))? };

    let f = backend.function_exact(HISTOGRAM, "scatter_stable")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(d_keys.device())
        .arg(d_vals.device())
        .arg(cursor.device_mut())
        .arg(sorted.device_mut())
        .arg(&n_i)
        .arg(&chunk_i)
        .arg(&nbins_i);
    // SAFETY: seven parameters, seven arguments. The block size is exactly `SUB`, which is
    // the shared arrays' length, and the kernel indexes them by `threadIdx.x < m <= SUB`.
    // Every write to `sorted` is at `cursor[t][b] + rank`, and the cursors partition
    // `0..n` by construction (they are the exclusive prefix sums of the same counts the
    // entries were counted into), so the writes are in range and non-overlapping.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (ntiles as u32, 1, 1),
            block_dim: (SUB as u32, 1, 1),
            shared_mem_bytes: 0,
        })?
    };

    let f = backend.function_exact(HISTOGRAM, "segment_sums")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(sorted.device())
        .arg(offset.device())
        .arg(total.device())
        .arg(out.device_mut())
        .arg(&nbins_i);
    // SAFETY: five parameters, five arguments. One thread per bin, guarded by `b < nbins`;
    // `offset[b] + total[b] <= n` because the offsets are the prefix sums of the totals.
    unsafe { launch.launch(cfg(nbins, BINS_PER_BLOCK))? };

    backend.synchronize()?;
    out.to_host(backend)
}

/// The same histogram, accumulated with `atomicAdd` on `double` — **not deterministic**,
/// and here to be shown so.
///
/// `atomicAdd` fixes no summation order, and `f64` addition is not associative, so two
/// runs of this on the same input can return different bits. Nothing may depend on it;
/// `histogram` is what a metric would use. See
/// `histogram_determinism.rs::the_atomic_histogram_is_not_deterministic_and_that_is_why_
/// this_module_exists`, which measures the spread.
pub fn histogram_atomic(keys: &[u32], values: &[f64], nbins: usize) -> Result<Vec<f64>, CudaError> {
    check(keys, values, nbins)?;
    let backend: &Backend = backend()?;
    let n = keys.len();

    let d_keys = DeviceBuffer::from_host(backend, keys)?;
    let d_vals = DeviceBuffer::from_host(backend, values)?;
    let mut out = DeviceBuffer::<f64>::zeros(backend, nbins)?;
    let n_i = n as i64;

    let f = backend.function_exact(HISTOGRAM, "hist_atomic")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(d_keys.device())
        .arg(d_vals.device())
        .arg(out.device_mut())
        .arg(&n_i);
    // SAFETY: four parameters, four arguments. `i < n` guards the reads and
    // `keys[i] < nbins` was checked on the host, so the atomic lands inside `out`.
    unsafe { launch.launch(cfg(n, SUB))? };
    backend.synchronize()?;
    out.to_host(backend)
}

/// The histogram the device is held to, on the host: the naive loop, in `f64`, in
/// ascending entry order.
///
/// This is the *definition*, not a test helper that happens to agree — `histogram` is
/// pinned bit-identical to it. Exported because the pin is the point.
pub fn histogram_host(keys: &[u32], values: &[f64], nbins: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nbins];
    for (&k, &v) in keys.iter().zip(values.iter()) {
        out[k as usize] += v;
    }
    out
}
