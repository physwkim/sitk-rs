//! Device-resident Gaussian smoothing: `smooth_gaussian` with the volume never
//! leaving the device.
//!
//! # Numerics: this mirrors the CPU filter, operation for operation
//!
//! `sitk_filters::smooth_gaussian` builds a symmetric 1-D kernel per axis —
//! `σ_idx = sigma[d] / spacing[d]`, `w[k] = exp(−k²/(2σ_idx²))`, truncated at
//! `⌈4σ_idx⌉`, normalized to sum 1 — and convolves the axes in sequence, in `f64`,
//! with an edge-replicating (zero-flux) boundary, narrowing to the pixel type only
//! at the end.
//!
//! This does the same, in the same order:
//!
//! - the **weights are computed on the host**, by the same `f64` arithmetic, and
//!   uploaded (a few dozen doubles) — they are not recomputed in the kernel, so
//!   there is no chance of a differently-rounded weight;
//! - the intermediate field is `f64` on the device, not `f32`, so an axis pass
//!   reads what the previous pass wrote at full precision, as the CPU does;
//! - the taps are accumulated with `__dmul_rn` / `__dadd_rn`, which forbid the
//!   compiler from contracting `w·v + acc` into an FMA. An FMA would be *more*
//!   accurate and would therefore disagree with the CPU, which does not fuse;
//! - the boundary clamps the coordinate to `[0, size−1]`, the same replication;
//! - an axis with `sigma == 0` is skipped, as on the CPU.
//!
//! The result is bit-identical to the CPU filter's, asserted by a test rather than
//! assumed.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;
use crate::image::DeviceImage;

const BLOCK: u32 = 256;

const KERNELS: &str = r#"
// One separable axis of the Gaussian, f64 in and f64 out, edge-replicating.
//
// `__dmul_rn` / `__dadd_rn` are deliberate: they forbid contraction into an FMA.
// A fused multiply-add here would round differently from the CPU filter, which
// does not fuse -- and "more accurate but different" is still different.
extern "C" __global__ void gauss_axis(
    const double* __restrict__ x,
    double* __restrict__ y,
    const long long n,
    const long long stride,     // element stride of the axis being convolved
    const long long size_d,     // extent of that axis
    const double* __restrict__ w,
    const long long radius)
{
    const long long gstride = (long long)blockDim.x * gridDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < n; p += gstride) {
        const long long coord = (p / stride) % size_d;
        const long long base  = p - coord * stride;
        double acc = 0.0;
        for (long long ki = 0; ki < 2 * radius + 1; ++ki) {
            long long c = coord + ki - radius;
            if (c < 0) c = 0;
            if (c > size_d - 1) c = size_d - 1;
            acc = __dadd_rn(acc, __dmul_rn(w[ki], x[base + c * stride]));
        }
        y[p] = acc;
    }
}

extern "C" __global__ void narrow_f64_f32(
    const double* __restrict__ x,
    float* __restrict__ y,
    const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)x[i];
}
"#;

/// Gaussian-smooth a resident volume: separable FIR convolution, per-dimension
/// `sigma` in **physical units** (the same contract as
/// `sitk_filters::smooth_gaussian`), reading and writing device memory only.
///
/// Bit-identical to the CPU filter — see the [module docs](self) for why that is a
/// claim about the arithmetic rather than a hope.
///
/// Fails with [`CudaError::DegenerateInput`] if `sigma` has the wrong length or any
/// entry is negative. An axis with `sigma == 0` is left untouched.
pub fn smooth_gaussian(src: &DeviceImage, sigma: &[f64]) -> Result<DeviceImage, CudaError> {
    let geom = src.geometry().clone();
    let dim = geom.dimension();
    if sigma.len() != dim || sigma.iter().any(|&s| s < 0.0) {
        return Err(CudaError::DegenerateInput);
    }
    let backend: &Backend = backend()?;
    let n = src.len();

    // Ping-pong between two f64 fields, exactly as the CPU rebinds `buf` each axis.
    let mut cur = src.widen_f64()?;
    let mut next = DeviceBuffer::<f64>::zeros(backend, n)?;

    let strides = strides(&geom.size);
    for d in 0..dim {
        if sigma[d] <= 0.0 {
            continue;
        }
        let (weights, radius) = gaussian_kernel(sigma[d] / geom.spacing[d]);
        let d_w = DeviceBuffer::from_host(backend, &weights)?;

        let f = backend.function(KERNELS, "gauss_axis")?;
        let (n_i64, stride, size_d, radius_i64) = (
            n as i64,
            strides[d] as i64,
            geom.size[d] as i64,
            radius as i64,
        );
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK as usize).min(1024) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = backend.stream().launch_builder(&f);
        launch
            .arg(cur.device())
            .arg(next.device_mut())
            .arg(&n_i64)
            .arg(&stride)
            .arg(&size_d)
            .arg(d_w.device())
            .arg(&radius_i64);
        // SAFETY: the kernel's seven parameters match the seven arguments pushed
        // above in order and type (`const double*`, `double*`, four `long long`s
        // around a `const double*`); both fields hold `n` doubles and every access
        // is guarded by the grid-stride bound and by the clamp on the tap
        // coordinate; `w` holds `2·radius + 1` doubles, which is the tap count.
        unsafe { launch.launch(cfg)? };
        backend.synchronize()?;

        std::mem::swap(&mut cur, &mut next);
    }

    // Narrow back to the device image's f32, as the CPU filter narrows to the
    // input's pixel type at the end rather than between axes.
    let mut dst = src.like()?;
    let f = backend.function(KERNELS, "narrow_f64_f32")?;
    let n_i64 = n as i64;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(cur.device())
        .arg(dst.buffer_mut().device_mut())
        .arg(&n_i64);
    // SAFETY: three parameters, three arguments, matching in order and type; both
    // buffers hold `n` elements and the kernel guards on `i < n`.
    unsafe { launch.launch(cfg)? };
    backend.synchronize()?;

    Ok(dst)
}

/// The CPU filter's kernel, computed by the CPU filter's arithmetic: taps at
/// integer offsets, truncated at `⌈4σ⌉`, normalized to sum 1. Returns
/// `(weights, radius)` with `weights.len() == 2·radius + 1`.
fn gaussian_kernel(sigma_idx: f64) -> (Vec<f64>, usize) {
    let radius = (4.0 * sigma_idx).ceil().max(1.0) as usize;
    let denom = 2.0 * sigma_idx * sigma_idx;
    let mut kernel = vec![0.0f64; 2 * radius + 1];
    let mut sum = 0.0;
    for (ki, w) in kernel.iter_mut().enumerate() {
        let k = ki as f64 - radius as f64;
        *w = (-(k * k) / denom).exp();
        sum += *w;
    }
    for w in &mut kernel {
        *w /= sum;
    }
    (kernel, radius)
}

/// First-index-fastest strides.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}
