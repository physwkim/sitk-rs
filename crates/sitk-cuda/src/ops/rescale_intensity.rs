//! `rescale_intensity` — op 1 of `doc/bench-spec.md`.
//!
//! Pure per-pixel, so it proves the whole pipeline (H2D → reduce → map → D2H →
//! fallback → correctness) with no algorithmic risk.
//!
//! # Numerics
//!
//! The CPU implementation widens every pixel to `f64`, takes `min`/`max` in
//! `f64`, computes `scale = (out_max - out_min) / (max - min)`, evaluates
//! `(v - min) * scale + out_min` in `f64`, and narrows the result back to the
//! pixel type with a C-style cast. The kernels do exactly that: the reduction
//! runs in `double`, the map loads `float`, computes in `double`, and stores
//! `(float)` — CUDA's default float↔double conversions are round-to-nearest-
//! even, the same rounding Rust's `as` uses. The only intentional freedom is
//! reduction *order*: `min`/`max` are exact and order-independent, so a
//! tree reduction cannot drift from the CPU's sequential scan.

use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use sitk_core::{Image, PixelId};

use crate::GpuTimings;
use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;

/// Threads per block. Also the shared-memory extent of the reduction, so the
/// kernel source and the launch config must agree — both read this constant.
const BLOCK: u32 = 256;
/// Upper bound on reduction blocks; the reduction is a grid-stride loop, so
/// this caps the partial-result buffers rather than the input size.
const MAX_REDUCE_BLOCKS: u32 = 1024;

const KERNELS: &str = r#"
#define BLOCK 256

extern "C" __global__ void minmax_f32(
    const float* __restrict__ x,
    const long long n,
    double* __restrict__ out_min,
    double* __restrict__ out_max)
{
    __shared__ double smin[BLOCK];
    __shared__ double smax[BLOCK];

    const int tid = threadIdx.x;
    const long long stride = (long long)blockDim.x * gridDim.x;

    double lo = __longlong_as_double(0x7ff0000000000000LL);  // +inf
    double hi = __longlong_as_double(0xfff0000000000000LL);  // -inf
    for (long long i = (long long)blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const double v = (double)x[i];
        lo = fmin(lo, v);
        hi = fmax(hi, v);
    }
    smin[tid] = lo;
    smax[tid] = hi;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = fmin(smin[tid], smin[tid + s]);
            smax[tid] = fmax(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_min[blockIdx.x] = smin[0];
        out_max[blockIdx.x] = smax[0];
    }
}

extern "C" __global__ void rescale_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const long long n,
    const double lo,
    const double scale,
    const double out_min)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (float)(((double)x[i] - lo) * scale + out_min);
    }
}
"#;

/// GPU `rescale_intensity`, or the exact reason the GPU did not run it.
///
/// The strict form: every failure is named. Callers that must not fail want
/// [`try_rescale_intensity`], which falls back to the CPU instead.
///
/// Supports `Float32` scalar images. Returns [`CudaError::DegenerateInput`] on
/// an empty image or one whose min equals its max — the CPU path owns the
/// user-visible error for those, and this path declines rather than inventing
/// a second one.
///
/// `GpuTimings::kernel_ms` covers both kernel launches *and* the 8 KiB copy of
/// the reduction's per-block partials back to the host, which is where the
/// final min/max fold happens (an exact, order-independent fold over ≤ 1024
/// values). `alloc_ms` is the host-side cost of making the output buffer
/// resident, and `d2h_ms` is then the DMA alone — see
/// [`sitk_core::alloc`] for why those two must not be reported as one number.
pub fn rescale_intensity_gpu(
    img: &Image,
    output_min: f64,
    output_max: f64,
) -> Result<(Image, GpuTimings), CudaError> {
    let n = check_input(img)?;

    // Made resident *before* the D2H, so the DMA lands on mapped pages instead
    // of faulting in 131,072 of them. This is the whole of Task 0: the op used
    // to hand `clone_dtoh` a fresh `Vec` and bill the resulting fault storm to
    // the PCIe link.
    //
    // Even so, this allocation is the largest single term left in the op — 108 ms
    // at 512³, because Linux must still zero every one of those pages before it
    // hands them over. The only way to *not* pay it is to not allocate, which a
    // function returning a fresh `Image` cannot express. That is what
    // [`rescale_intensity_gpu_into`] is for.
    let t = Instant::now();
    let mut host_out = sitk_core::alloc::resident_vec::<f32>(n);
    let alloc_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut timings = run(img, output_min, output_max, &mut host_out)?;
    timings.alloc_ms = alloc_ms;

    let mut out = Image::from_vec(img.size(), host_out)?;
    out.copy_geometry_from(img);
    Ok((out, timings))
}

/// GPU `rescale_intensity` writing into a destination the caller already owns —
/// the zero-allocation form.
///
/// `dst` must be a `Float32` scalar image with the same voxel count as `img`; it
/// is overwritten in full, and its geometry is copied from `img`. Nothing on the
/// host is allocated, so `GpuTimings::alloc_ms` is 0.
///
/// This exists because the allocation, not the bus, is the cost. A caller that
/// loops — a pipeline stage, a per-slice sweep, an optimizer — can hand the same
/// destination back on every call and pay the page faults exactly once, for the
/// life of the buffer, instead of once per call. At 512³ that is 108 ms per call
/// that simply stops being spent.
///
/// The one-shot [`rescale_intensity_gpu`] / [`try_rescale_intensity`] pair is
/// unchanged: a caller with no destination to reuse keeps the API it has.
pub fn rescale_intensity_gpu_into(
    img: &Image,
    output_min: f64,
    output_max: f64,
    dst: &mut Image,
) -> Result<GpuTimings, CudaError> {
    let n = check_input(img)?;
    if dst.pixel_id() != PixelId::Float32 {
        return Err(CudaError::UnsupportedPixelType(dst.pixel_id()));
    }
    if dst.size() != img.size() {
        return Err(CudaError::DegenerateInput);
    }
    let size = img.size().to_vec();
    let host_out = dst.scalar_vec_mut::<f32>()?;
    if host_out.len() != n {
        return Err(CudaError::DegenerateInput);
    }
    debug_assert_eq!(size.iter().product::<usize>(), n);

    let timings = run(img, output_min, output_max, host_out)?;
    dst.copy_geometry_from(img);
    Ok(timings)
}

/// The input contract both forms share: `Float32`, scalar, non-empty. Returns the
/// voxel count.
fn check_input(img: &Image) -> Result<usize, CudaError> {
    if img.pixel_id() != PixelId::Float32 {
        return Err(CudaError::UnsupportedPixelType(img.pixel_id()));
    }
    let n = img.scalar_slice::<f32>()?.len();
    if n == 0 {
        return Err(CudaError::DegenerateInput);
    }
    Ok(n)
}

/// H2D → reduce → map → D2H into `host_out`, which the caller owns. The only
/// difference between the two public forms is where `host_out` came from, so it
/// is the only thing this does not decide.
///
/// `timings.alloc_ms` is left at 0: this function allocates nothing on the host.
fn run(
    img: &Image,
    output_min: f64,
    output_max: f64,
    host_out: &mut [f32],
) -> Result<GpuTimings, CudaError> {
    let host_in = img.scalar_slice::<f32>()?;
    let n = host_in.len();
    let backend = backend()?;
    let mut timings = GpuTimings::default();

    // ---- H2D -------------------------------------------------------------
    let t = Instant::now();
    let d_in = DeviceBuffer::from_host(backend, host_in)?;
    backend.synchronize()?;
    timings.h2d_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- kernel 1: min/max reduction -------------------------------------
    // Ahead of the map, so a degenerate image declines before it is mapped.
    let t = Instant::now();
    let (lo, hi) = device_min_max(backend, &d_in, n)?;
    if lo == hi {
        return Err(CudaError::DegenerateInput);
    }
    let scale = (output_max - output_min) / (hi - lo);
    timings.kernel_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- kernel 2: the map -----------------------------------------------
    let t = Instant::now();
    let mut d_out = DeviceBuffer::<f32>::zeros(backend, n)?;
    let f = backend.function(KERNELS, "rescale_f32")?;
    let n_i64 = n as i64;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(d_in.device())
        .arg(d_out.device_mut())
        .arg(&n_i64)
        .arg(&lo)
        .arg(&scale)
        .arg(&output_min);
    // SAFETY: the kernel's six parameters match the six arguments pushed above
    // in order and type (`const float*`, `float*`, `long long`, `double` × 3),
    // both buffers hold `n` `f32`s, and the kernel guards every access with
    // `i < n`.
    unsafe { launch.launch(cfg)? };
    backend.synchronize()?;
    timings.kernel_ms += t.elapsed().as_secs_f64() * 1e3;

    // ---- D2H -------------------------------------------------------------
    let t = Instant::now();
    d_out.copy_to_host(backend, host_out)?;
    backend.synchronize()?;
    timings.d2h_ms = t.elapsed().as_secs_f64() * 1e3;

    Ok(timings)
}

/// `rescale_intensity` with **both buffers already on the device**, and neither
/// leaving it: no H2D, no D2H, no host allocation. Returns the kernel wall-clock
/// in milliseconds (synchronized).
///
/// This is the op stripped of the bus. Every GPU-vs-CPU number the port has
/// published for a filter was measured through the one-shot `fn(&Image) -> Image`
/// forms above, which pay a round trip *per call* — 67 MB each way at 256³ for
/// `f32`. That measures the **API**, not the kernel, and the two answers are not
/// the same question: a pipeline that keeps the volume resident across several
/// ops pays the crossing once for the whole chain, not once per op.
///
/// `d_out` must be at least as long as `d_in`. Declines a degenerate image
/// exactly as the one-shot forms do, so the fallback contract is unchanged.
pub fn rescale_intensity_resident(
    backend: &Backend,
    d_in: &DeviceBuffer<f32>,
    d_out: &mut DeviceBuffer<f32>,
    output_min: f64,
    output_max: f64,
) -> Result<f64, CudaError> {
    let n = d_in.len();
    if n == 0 || d_out.len() < n {
        return Err(CudaError::DegenerateInput);
    }
    let t = Instant::now();

    let (lo, hi) = device_min_max(backend, d_in, n)?;
    if lo == hi {
        return Err(CudaError::DegenerateInput);
    }
    let scale = (output_max - output_min) / (hi - lo);

    let f = backend.function(KERNELS, "rescale_f32")?;
    let n_i64 = n as i64;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(d_in.device())
        .arg(d_out.device_mut())
        .arg(&n_i64)
        .arg(&lo)
        .arg(&scale)
        .arg(&output_min);
    // SAFETY: identical to the launch in `run` — six parameters matching six
    // arguments in order and type, both buffers hold at least `n` `f32`s, every
    // access guarded by `i < n`.
    unsafe { launch.launch(cfg)? };
    backend.synchronize()?;

    Ok(t.elapsed().as_secs_f64() * 1e3)
}

/// Exact min/max of the device buffer, folded on the host over the reduction's
/// per-block partials. `min`/`max` are exact and associative, so this equals
/// the CPU's sequential scan bit-for-bit.
fn device_min_max(
    backend: &Backend,
    d_in: &DeviceBuffer<f32>,
    n: usize,
) -> Result<(f64, f64), CudaError> {
    let blocks = (n.div_ceil(BLOCK as usize) as u32).min(MAX_REDUCE_BLOCKS);
    let mut d_min = DeviceBuffer::<f64>::zeros(backend, blocks as usize)?;
    let mut d_max = DeviceBuffer::<f64>::zeros(backend, blocks as usize)?;

    let f = backend.function(KERNELS, "minmax_f32")?;
    let n_i64 = n as i64;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(d_in.device())
        .arg(&n_i64)
        .arg(d_min.device_mut())
        .arg(d_max.device_mut());
    // SAFETY: the kernel's four parameters match the four arguments pushed
    // above in order and type (`const float*`, `long long`, `double*`,
    // `double*`); the input holds `n` `f32`s and the kernel reads it only
    // under `i < n`; both output buffers hold exactly `blocks` `f64`s and the
    // kernel writes index `blockIdx.x < blocks`. Shared memory is declared
    // statically in the kernel at `BLOCK` doubles, matching `block_dim`.
    unsafe { launch.launch(cfg)? };

    let mins = d_min.to_host(backend)?;
    let maxs = d_max.to_host(backend)?;
    backend.synchronize()?;

    let lo = mins.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = maxs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((lo, hi))
}

/// GPU `rescale_intensity` with a CPU fallback contract: `None` means the GPU
/// produced no result — no driver, no device, NVRTC failure, out of memory,
/// unsupported pixel type, degenerate input — and the caller must run the CPU
/// implementation. Never panics.
pub fn try_rescale_intensity(img: &Image, output_min: f64, output_max: f64) -> Option<Image> {
    rescale_intensity_gpu(img, output_min, output_max)
        .ok()
        .map(|(image, _timings)| image)
}
