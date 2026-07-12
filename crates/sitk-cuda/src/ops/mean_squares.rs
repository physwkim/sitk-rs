//! Device-resident mean-squares metric: the moment reduction.
//!
//! # Why this exists, and why it is not a per-pixel filter
//!
//! `rescale_intensity` proved that a per-pixel op cannot win on a GPU: you ship
//! the whole volume across the bus to do one arithmetic operation to it. Image
//! registration inverts that. The fixed and moving volumes are **constant across
//! hundreds of optimizer iterations**, so they are uploaded *once*; each
//! iteration then pushes a handful of numbers up and pulls a handful down, while
//! the GPU does the compute-dense part — resample the moving image under the
//! current transform and reduce a similarity metric over millions of voxels.
//! The PCIe cost is paid once and amortized to nothing.
//!
//! # The moment formulation
//!
//! The naive kernel accumulates one gradient slot per transform parameter, which
//! is fine for a rigid transform (6) and impossible for a B-spline (thousands).
//! It also needs a different kernel per transform type.
//!
//! Neither is necessary. The metric derivative is
//!
//! ```text
//! ∂value/∂pₖ = (2/N) Σᵢ diffᵢ · Σ_d ∇M(T(xᵢ))_d · J(xᵢ)[d][k]
//! ```
//!
//! and for every *globally affine* transform (translation, rigid, Euler, versor,
//! similarity, affine) each Jacobian column is affine in the point:
//!
//! ```text
//! J(x)[d][k] = J(0)[d][k] + Σ_e x_e · C_e[d][k]
//! ```
//!
//! Substituting and exchanging the sums, the whole derivative — for any number of
//! parameters — factors through just **14 scalars** that do not mention the
//! parameters at all:
//!
//! ```text
//! sq        = Σᵢ diffᵢ²                        (1)
//! S0[d]     = Σᵢ diffᵢ · ∇Mᵢ[d]                (3)
//! S1[d][e]  = Σᵢ diffᵢ · ∇Mᵢ[d] · xᵢ[e]        (9)
//! count     = number of valid samples          (1)
//!
//! value   = sq / count
//! ∂/∂pₖ   = (2/count) · ( Σ_d J(0)[d][k]·S0[d] + Σ_d Σ_e C_e[d][k]·S1[d][e] )
//! ```
//!
//! This is an identity, not an approximation. So **one kernel serves the entire
//! global-affine family**, the device never learns what a transform *is*, and the
//! host does the contraction in `f64` using the transform's own existing
//! `jacobian_wrt_parameters` — no downcast, no type whitelist, no per-transform
//! kernel. The device is told only the point map `x ↦ A·x + b` (12 doubles).
//!
//! A transform whose point map or Jacobian is *not* affine in the point (B-spline,
//! displacement field) fails the caller's linearity probe and falls back to the
//! CPU. The fallback is a property of the mathematics, not of a type list.
//!
//! # Determinism
//!
//! A float sum reduction is not associative, so a GPU sum ordinarily depends on
//! block and warp scheduling. That is not tolerable here: the optimizer is a
//! feedback loop, so a metric that varies run to run makes the *registration
//! result* vary run to run. This is a correctness property, not a performance one.
//!
//! So: **no `atomicAdd` anywhere**. The grid is a fixed size, each block reduces
//! its slice through a fixed shared-memory tree, and the per-block partials are
//! folded on the host in block-index order. Every run performs exactly the same
//! additions in exactly the same order, and the result is bit-identical run to
//! run — asserted by a test, not assumed.
//!
//! It is *not* bit-identical to the CPU's sequential sum, and cannot be: no
//! parallel reduction can reproduce a left-to-right f64 accumulation. The
//! divergence is reduction-rounding only (~√N·ε).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;

/// Threads per block. The kernel's shared-memory tree is exactly this wide.
const BLOCK: u32 = 256;
/// Blocks. **Fixed**, not derived from the sample count: the reduction order —
/// and therefore the result — must not change with the input size. The kernel is
/// a grid-stride loop, so this covers any `n`.
const GRID: u32 = 512;
/// `sq` (1) + `S0[3]` (3) + `S1[3][3]` (9) + `count` (1).
const NSLOT: usize = 14;
/// This backend handles 3-D only. The moment algebra generalizes, but the kernel
/// is written for `dim = 3` and a 2-D caller falls back to the CPU.
pub const DIM: usize = 3;

const KERNEL: &str = r#"
#define BLOCK 256
#define NSLOT 14

// Mirrors sitk_transform::interpolator::is_inside exactly: a sample is valid iff
// every continuous-index component lies in [-0.5, size-0.5).
__device__ __forceinline__ bool is_inside(const double* c, const long long* size) {
    for (int d = 0; d < 3; ++d) {
        if (!(c[d] >= -0.5 && c[d] < (double)size[d] - 0.5)) return false;
    }
    return true;
}

extern "C" __global__ void ms_moments(
    const double* __restrict__ fvals,     // fixed sample values, n
    const double* __restrict__ fpts,      // fixed sample points, n * 3 (row-major)
    const long long n,
    const double* __restrict__ mbuf,      // moving image buffer
    const long long* __restrict__ msize,  // 3
    const long long* __restrict__ mstride,// 3
    const double* __restrict__ morigin,   // 3
    const double* __restrict__ mmat,      // 3x3 phys_to_index, row-major
    const unsigned char* __restrict__ mmask,
    const int has_mask,
    const double* __restrict__ ab,        // A (9, row-major) then b (3)
    double* __restrict__ partials)        // GRID * NSLOT
{
    double acc[NSLOT];
    for (int k = 0; k < NSLOT; ++k) acc[k] = 0.0;

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        const double x[3] = { fpts[s*3+0], fpts[s*3+1], fpts[s*3+2] };

        // p = A*x + b  --- the transform's point map, whatever transform it is.
        double p[3];
        for (int d = 0; d < 3; ++d) {
            p[d] = ab[9+d] + ab[d*3+0]*x[0] + ab[d*3+1]*x[1] + ab[d*3+2]*x[2];
        }

        // c = M * (p - origin): continuous index in the moving image.
        double c[3];
        for (int r = 0; r < 3; ++r) {
            double a = 0.0;
            for (int j = 0; j < 3; ++j) a += mmat[r*3+j] * (p[j] - morigin[j]);
            c[r] = a;
        }

        if (has_mask) {
            // MovingImage::mask_allows: round to nearest voxel, reject if outside.
            long long flat = 0; bool ok = true;
            for (int d = 0; d < 3; ++d) {
                const double r = round(c[d]);
                if (r < 0.0 || (long long)r >= msize[d]) { ok = false; break; }
                flat += (long long)r * mstride[d];
            }
            if (!ok || !mmask[flat]) continue;
        }
        if (!is_inside(c, msize)) continue;

        // Trilinear value + exact gradient of the interpolant, in the same corner
        // order and with the same clamping as linear_value_and_gradient.
        double base[3], frac[3];
        for (int d = 0; d < 3; ++d) {
            const double f = floor(c[d]);
            base[d] = f;
            frac[d] = c[d] - f;
        }
        double value = 0.0;
        double gi[3] = { 0.0, 0.0, 0.0 };
        for (int corner = 0; corner < 8; ++corner) {
            long long offset = 0;
            double weight = 1.0;
            for (int d = 0; d < 3; ++d) {
                const int bit = (corner >> d) & 1;
                weight *= bit ? frac[d] : (1.0 - frac[d]);
                long long idx = (long long)base[d] + bit;
                if (idx < 0) idx = 0;
                if (idx > msize[d] - 1) idx = msize[d] - 1;
                offset += idx * mstride[d];
            }
            const double b = mbuf[offset];
            value += weight * b;
            for (int j = 0; j < 3; ++j) {
                double w = 1.0;
                for (int d = 0; d < 3; ++d) {
                    if (d == j) continue;
                    const int bit = (corner >> d) & 1;
                    w *= bit ? frac[d] : (1.0 - frac[d]);
                }
                const double sign = ((corner >> j) & 1) ? 1.0 : -1.0;
                gi[j] += sign * w * b;
            }
        }

        // Index-space gradient -> physical-space: g[d] = sum_j gi[j] * M[j][d].
        double g[3];
        for (int d = 0; d < 3; ++d) {
            g[d] = gi[0]*mmat[0*3+d] + gi[1]*mmat[1*3+d] + gi[2]*mmat[2*3+d];
        }

        const double diff = value - fvals[s];
        acc[0] += diff * diff;
        for (int d = 0; d < 3; ++d) {
            const double dg = diff * g[d];
            acc[1 + d] += dg;
            for (int e = 0; e < 3; ++e) acc[4 + d*3 + e] += dg * x[e];
        }
        acc[13] += 1.0;
    }

    // Fixed shared-memory tree, one slot at a time. No atomics: the reduction
    // order is identical on every run, so the result is too.
    __shared__ double sh[BLOCK];
    const int tid = threadIdx.x;
    for (int k = 0; k < NSLOT; ++k) {
        sh[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        if (tid == 0) partials[blockIdx.x * NSLOT + k] = sh[0];
        __syncthreads();
    }
}
"#;

/// The 14 moments of one metric evaluation. Parameter-count- and
/// transform-independent; the host contracts these into value + derivative.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    /// `Σ diff²`.
    pub sq: f64,
    /// `Σ diff · ∇M[d]`.
    pub s0: [f64; DIM],
    /// `Σ diff · ∇M[d] · x[e]`, indexed `[d][e]`.
    pub s1: [[f64; DIM]; DIM],
    /// Samples that mapped inside the moving image.
    pub count: usize,
}

/// The moving image's geometry, as the kernel needs it.
pub struct MovingGeometry<'a> {
    pub buf: &'a [f64],
    pub size: &'a [usize],
    pub strides: &'a [usize],
    pub origin: &'a [f64],
    /// `diag(1/spacing) · D⁻¹`, row-major `3 × 3`.
    pub phys_to_index: &'a [f64],
    pub mask: Option<&'a [bool]>,
}

/// Fixed and moving volumes resident on the device, evaluable against any number
/// of transforms without re-uploading either.
///
/// Built once per pyramid level; [`evaluate`](Self::evaluate) is then called once
/// (or twice) per optimizer iteration and moves **96 bytes up** (the point map)
/// and **`GRID · 14 · 8` = 57 KiB down** (the per-block partials). Nothing else
/// crosses the bus, and nothing is reallocated per iteration — the partials
/// buffer and its host destination are owned here and reused.
pub struct ResidentMetric {
    n: usize,
    d_fvals: DeviceBuffer<f64>,
    d_fpts: DeviceBuffer<f64>,
    d_mbuf: DeviceBuffer<f64>,
    d_msize: DeviceBuffer<i64>,
    d_mstride: DeviceBuffer<i64>,
    d_morigin: DeviceBuffer<f64>,
    d_mmat: DeviceBuffer<f64>,
    d_mmask: DeviceBuffer<u8>,
    has_mask: i32,
    /// Reused across iterations: the per-iteration H2D writes into this rather
    /// than allocating (`copy_from_host`, not `from_host`).
    d_ab: DeviceBuffer<f64>,
    d_partials: DeviceBuffer<f64>,
    /// Reused host destination for the partials, so the per-iteration D2H never
    /// touches a fresh page. This is the first-touch defect, closed by
    /// construction: the buffer outlives the call.
    h_partials: Vec<f64>,
}

impl ResidentMetric {
    /// Upload the fixed samples and the moving volume. This is the *only* large
    /// transfer in a registration run.
    pub fn new(
        fixed_values: &[f64],
        fixed_points: &[f64],
        moving: &MovingGeometry<'_>,
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        let n = fixed_values.len();
        if n == 0 || moving.size.len() != DIM || fixed_points.len() != n * DIM {
            return Err(CudaError::DegenerateInput);
        }

        let as_i64 = |v: &[usize]| v.iter().map(|&x| x as i64).collect::<Vec<_>>();
        let (mask_bytes, has_mask) = match moving.mask {
            // A zero-length allocation is not a valid kernel pointer, so the
            // no-mask case still allocates one byte and gates on `has_mask`.
            None => (vec![0u8; 1], 0),
            Some(m) => (m.iter().map(|&b| u8::from(b)).collect(), 1),
        };

        Ok(Self {
            n,
            d_fvals: DeviceBuffer::from_host(backend, fixed_values)?,
            d_fpts: DeviceBuffer::from_host(backend, fixed_points)?,
            d_mbuf: DeviceBuffer::from_host(backend, moving.buf)?,
            d_msize: DeviceBuffer::from_host(backend, &as_i64(moving.size))?,
            d_mstride: DeviceBuffer::from_host(backend, &as_i64(moving.strides))?,
            d_morigin: DeviceBuffer::from_host(backend, moving.origin)?,
            d_mmat: DeviceBuffer::from_host(backend, moving.phys_to_index)?,
            d_mmask: DeviceBuffer::from_host(backend, &mask_bytes)?,
            has_mask,
            d_ab: DeviceBuffer::zeros(backend, 12)?,
            d_partials: DeviceBuffer::zeros(backend, GRID as usize * NSLOT)?,
            h_partials: vec![0.0; GRID as usize * NSLOT],
        })
    }

    /// Number of fixed samples resident on the device.
    pub fn sample_count(&self) -> usize {
        self.n
    }

    /// Evaluate the moments for the point map `x ↦ A·x + b`.
    ///
    /// `a` is row-major `3 × 3`, `b` is length 3. Deterministic: the same inputs
    /// give bit-identical moments on every call and every run.
    pub fn evaluate(&mut self, a: &[f64; 9], b: &[f64; 3]) -> Result<Moments, CudaError> {
        let backend: &Backend = backend()?;

        let mut ab = [0.0f64; 12];
        ab[..9].copy_from_slice(a);
        ab[9..].copy_from_slice(b);
        self.d_ab.copy_from_host(backend, &ab)?;

        let f = backend.function(KERNEL, "ms_moments")?;
        let n_i64 = self.n as i64;
        let cfg = LaunchConfig {
            grid_dim: (GRID, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = backend.stream().launch_builder(&f);
        launch
            .arg(self.d_fvals.device())
            .arg(self.d_fpts.device())
            .arg(&n_i64)
            .arg(self.d_mbuf.device())
            .arg(self.d_msize.device())
            .arg(self.d_mstride.device())
            .arg(self.d_morigin.device())
            .arg(self.d_mmat.device())
            .arg(self.d_mmask.device())
            .arg(&self.has_mask)
            .arg(self.d_ab.device())
            .arg(self.d_partials.device_mut());
        // SAFETY: the twelve arguments match the kernel's twelve parameters in
        // order and type. `d_fvals` holds `n` f64 and `d_fpts` holds `n*3`, both
        // read only under `s < n`; the geometry buffers hold exactly 3, 3, 3 and
        // 9 elements as the kernel indexes them; `d_mmask` is read only when
        // `has_mask != 0`, in which case it has one byte per moving voxel and the
        // kernel bounds-checks the index it builds; `d_ab` holds 12; `d_partials`
        // holds `GRID * NSLOT`, and the kernel writes `blockIdx.x * NSLOT + k`
        // for `blockIdx.x < GRID`, `k < NSLOT`. Shared memory is declared
        // statically at `BLOCK` doubles, matching `block_dim`.
        unsafe { launch.launch(cfg)? };

        self.d_partials
            .copy_to_host(backend, &mut self.h_partials)?;
        backend.synchronize()?;

        // Fold the per-block partials in block-index order. Fixed order, on the
        // host, in f64 — this is the step that makes the result reproducible.
        let mut m = Moments::default();
        let mut count = 0.0f64;
        for blk in 0..GRID as usize {
            let p = &self.h_partials[blk * NSLOT..(blk + 1) * NSLOT];
            m.sq += p[0];
            for d in 0..DIM {
                m.s0[d] += p[1 + d];
                for e in 0..DIM {
                    m.s1[d][e] += p[4 + d * DIM + e];
                }
            }
            count += p[13];
        }
        m.count = count as usize;
        Ok(m)
    }
}
