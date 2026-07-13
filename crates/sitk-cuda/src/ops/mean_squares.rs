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

use std::sync::OnceLock;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;
use crate::image::DeviceImage;

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

/// The kernel body, with the two volumes' element types left open — `FSCALAR` for
/// the fixed samples, `MSCALAR` for the moving image. Nothing else about the
/// instantiations differs.
///
/// The volumes are the only scalars: every load widens to `double` on the spot
/// (`(double)x` is exact for every `f32`) and every multiply, add and accumulator
/// below is `double` in every instantiation. That is what makes a narrowed volume
/// bit-identical to a wide one **when the voxels came from an `f32` image** —
/// which is exactly when it is narrowed.
///
/// The two are separate because they are not symmetric. The fixed value is **one**
/// load per sample; the moving image is **eight** (the trilinear corners). Each
/// `f32` load costs an `f32 → f64` conversion, which issues on the FP64 pipe that
/// Ada runs at 1:64 — so narrowing the moving image buys half the memory at eight
/// ninths of the conversion cost, while narrowing the fixed samples buys the other
/// half of the memory at one ninth of it. See [`Volumes`].
const KERNEL_BODY: &str = r#"
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
    const FSCALAR* __restrict__ fvals,    // fixed sample values, n
    const double* __restrict__ fpts,      // fixed sample points, n * 3 (row-major); unused if !has_pts
    const int has_pts,
    const long long* __restrict__ fsize,  // 3, fixed grid size (used if !has_pts)
    const double* __restrict__ forigin,   // 3
    const double* __restrict__ fmat,      // 3x3 index_to_physical, row-major
    const long long n,
    const MSCALAR* __restrict__ mbuf,     // moving image buffer
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
        // The sample's physical point. When the sample set is the whole fixed grid
        // in traversal order -- the common case -- it is a pure function of `s` and
        // the grid, so it is DERIVED here rather than uploaded: at 256^3 the points
        // array is 402 MB, which was 60% of the only large transfer in the run.
        // Same arithmetic, same order, as the host's `write_point_at`.
        double x[3];
        if (has_pts) {
            x[0] = fpts[s*3+0]; x[1] = fpts[s*3+1]; x[2] = fpts[s*3+2];
        } else {
            const double i = (double)(s % fsize[0]);
            const double j = (double)((s / fsize[0]) % fsize[1]);
            const double k = (double)(s / (fsize[0] * fsize[1]));
            for (int r = 0; r < 3; ++r) {
                double acc_r = forigin[r];
                acc_r += fmat[r*3+0] * i;
                acc_r += fmat[r*3+1] * j;
                acc_r += fmat[r*3+2] * k;
                x[r] = acc_r;
            }
        }

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
            const double b = (double)mbuf[offset];
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

        const double diff = value - (double)fvals[s];
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

/// The kernel source for a volume element type, compiled once per process (the
/// backend caches modules by source).
fn kernel_src(fixed: &str, moving: &str) -> &'static str {
    static WIDE: OnceLock<String> = OnceLock::new();
    static SPLIT: OnceLock<String> = OnceLock::new();
    static NARROW: OnceLock<String> = OnceLock::new();
    let cell = match (fixed, moving) {
        ("float", "float") => &NARROW,
        ("float", _) => &SPLIT,
        _ => &WIDE,
    };
    cell.get_or_init(|| format!("#define FSCALAR {fixed}\n#define MSCALAR {moving}\n{KERNEL_BODY}"))
        .as_str()
}

/// The fixed samples and the moving volume, in the precision the kernel reads them.
///
/// Not a tuning knob for the caller: the *host-producer* path
/// ([`ResidentMetric::new`]) accepts any pixel type, including `Float64` and
/// `Int64`, so it stays [`F64`](Volumes::F64) — narrowing there would drop bits the
/// caller handed us. The *device* path ([`ResidentMetric::from_device`]) reads `f32`
/// images, so narrowing is lossless and the only question is which volume to narrow.
///
/// It narrows the **fixed** samples and leaves the moving image wide, because the
/// two are not symmetric: one fixed load per sample against eight moving loads (the
/// trilinear corners), while the memory splits evenly between them. See
/// [`from_device`](ResidentMetric::from_device) for the measurement.
enum Volumes {
    F64 {
        fvals: DeviceBuffer<f64>,
        mbuf: DeviceBuffer<f64>,
    },
    /// Fixed narrow, moving wide.
    Split {
        fvals: DeviceBuffer<f32>,
        mbuf: DeviceBuffer<f64>,
    },
}

impl Volumes {
    fn lens(&self) -> (usize, usize) {
        match self {
            Volumes::F64 { fvals, mbuf } => (fvals.len(), mbuf.len()),
            Volumes::Split { fvals, mbuf } => (fvals.len(), mbuf.len()),
        }
    }

    /// Device bytes held by the two volumes.
    fn bytes(&self) -> usize {
        match self {
            Volumes::F64 { fvals, mbuf } => 8 * fvals.len() + 8 * mbuf.len(),
            Volumes::Split { fvals, mbuf } => 4 * fvals.len() + 8 * mbuf.len(),
        }
    }
}

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

/// The moving image's geometry, as the kernel needs it. The voxels themselves
/// arrive through the producer argument of [`ResidentMetric::new`], not here —
/// the host holds them in their native pixel type, so there is no `f64` slice to
/// borrow.
pub struct MovingGeometry<'a> {
    /// Voxel count; must equal the product of `size`.
    pub len: usize,
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
/// Where the fixed samples' physical points come from.
///
/// The points are `origin + idx_to_phys · index`, so when the sample set is the
/// whole fixed grid in traversal order they are a pure function of the sample
/// index and need not exist as a buffer at all. [`Grid`](Self::Grid) says so, and
/// the kernel derives each point from its own `s`. At 256³ that removes a 402 MB
/// upload — 60% of the run's only large transfer.
///
/// [`Explicit`](Self::Explicit) is for a sampled or masked set, whose points are
/// an arbitrary subset in an arbitrary order and must be uploaded.
pub enum FixedPoints<'a> {
    /// One point per sample, row-major `N × 3`.
    Explicit(&'a [f64]),
    /// Every voxel of `size`, in dim-0-fastest order. `idx_to_phys` is row-major
    /// `3 × 3`; the sample count must equal the product of `size`.
    Grid {
        size: &'a [usize],
        origin: &'a [f64],
        idx_to_phys: &'a [f64],
    },
}

pub struct ResidentMetric {
    n: usize,
    vols: Volumes,
    d_fpts: DeviceBuffer<f64>,
    has_pts: i32,
    d_fsize: DeviceBuffer<i64>,
    d_forigin: DeviceBuffer<f64>,
    d_fmat: DeviceBuffer<f64>,
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
    /// `fixed_values(start, out)` writes the fixed samples `start..start+out.len()`,
    /// widened to `f64`; `moving_values` does the same for the moving volume's
    /// voxels. Producers rather than slices: the host keeps both images in their
    /// **native** pixel type, so there is no `f64` volume to borrow — the widening
    /// is staged a chunk at a time straight into the upload (see
    /// [`DeviceBuffer::from_chunks`]).
    pub fn new(
        n: usize,
        fixed_values: impl FnMut(usize, &mut [f64]),
        fixed_points: FixedPoints<'_>,
        moving: &MovingGeometry<'_>,
        moving_values: impl FnMut(usize, &mut [f64]),
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        Self::build(
            n,
            Volumes::F64 {
                fvals: DeviceBuffer::from_chunks(backend, n, fixed_values)?,
                mbuf: DeviceBuffer::from_chunks(backend, moving.len, moving_values)?,
            },
            fixed_points,
            moving,
        )
    }

    /// Build the metric from volumes that are **already on the device** — the
    /// registration half of the residency pipeline.
    ///
    /// This is the transfer that residency deletes. The host-side [`new`](Self::new)
    /// uploads the fixed samples and the moving volume, and when those volumes came
    /// out of a device filter chain seconds earlier, it is re-uploading voxels that
    /// were *already on the device*: 113.7 ms at 256³, the largest single item in
    /// the chain. Here nothing crosses the bus.
    ///
    /// # Which volume is narrowed, and why only one
    ///
    /// The images are `f32`, so keeping either volume `f32` is lossless
    /// (`(double)x` is exact) — but each `f32` load costs a conversion on the FP64
    /// pipe, and the two volumes are read at very different rates: **one** fixed
    /// load per sample against **eight** moving loads (the trilinear corners), while
    /// the memory splits evenly. Measured at 256³ on an RTX 5000 Ada:
    ///
    /// ```text
    ///   256^3, mean of 3        per evaluation           volume bytes
    ///   f64 fixed / f64 moving     5.755 ms                 268 MB
    ///   f32 fixed / f32 moving     6.088 ms   (+5.8%)       134 MB
    ///   f32 fixed / f64 moving     5.747 ms   (-0.1%)       201 MB   <- this
    ///
    ///   128^3                      0.816 / 0.873 (+7.0%) / 0.826 (+1.2%) ms
    /// ```
    ///
    /// So the fixed samples are narrowed and the moving image is left wide: half the
    /// memory saving for a conversion cost that does not show above the noise at 256³
    /// and is ~1% at 128³. Narrowing *both* pays eight more conversions per sample for
    /// the other half of the memory, and costs 6–7%.
    ///
    /// The buffers are private device-to-device copies, so the caller may drop or
    /// reuse the images.
    ///
    /// `fixed_points` is normally [`FixedPoints::Grid`] — a device-resident image is
    /// a full grid with no sampling and no mask — and `moving` carries the moving
    /// image's geometry (its `mask` must be `None`; a device image has no mask).
    pub fn from_device(
        fixed: &DeviceImage,
        fixed_points: FixedPoints<'_>,
        moving: &DeviceImage,
        moving_geometry: &MovingGeometry<'_>,
    ) -> Result<Self, CudaError> {
        if moving.len() != moving_geometry.len {
            return Err(CudaError::DegenerateInput);
        }
        let backend = backend()?;
        Self::build(
            fixed.len(),
            Volumes::Split {
                // The fixed samples: narrow, one load per sample.
                fvals: DeviceBuffer::copy_of(backend, fixed.buffer().device())?,
                // The moving image: wide, eight loads per sample.
                mbuf: moving.widen_f64()?,
            },
            fixed_points,
            moving_geometry,
        )
    }

    /// The construction both constructors share: everything except where the
    /// volumes came from and what precision they are in, which is the only thing
    /// the two disagree about.
    fn build(
        n: usize,
        vols: Volumes,
        fixed_points: FixedPoints<'_>,
        moving: &MovingGeometry<'_>,
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        let (fvals_len, mbuf_len) = vols.lens();
        if n == 0
            || fvals_len != n
            || mbuf_len != moving.len
            || moving.size.len() != DIM
            || moving.size.iter().product::<usize>() != moving.len
        {
            return Err(CudaError::DegenerateInput);
        }

        let as_i64 = |v: &[usize]| v.iter().map(|&x| x as i64).collect::<Vec<_>>();

        // A zero-length allocation is not a valid kernel pointer, so the unused
        // side of the choice below still allocates a single dummy element and the
        // kernel gates on `has_pts`.
        let (pts, has_pts, fsize, forigin, fmat) = match fixed_points {
            FixedPoints::Explicit(p) => {
                if p.len() != n * DIM {
                    return Err(CudaError::DegenerateInput);
                }
                (p, 1, vec![1i64; DIM], vec![0.0; DIM], vec![0.0; DIM * DIM])
            }
            FixedPoints::Grid {
                size,
                origin,
                idx_to_phys,
            } => {
                if size.len() != DIM
                    || origin.len() != DIM
                    || idx_to_phys.len() != DIM * DIM
                    || size.iter().product::<usize>() != n
                {
                    return Err(CudaError::DegenerateInput);
                }
                (
                    &[0.0f64][..],
                    0,
                    as_i64(size),
                    origin.to_vec(),
                    idx_to_phys.to_vec(),
                )
            }
        };
        let (mask_bytes, has_mask) = match moving.mask {
            // A zero-length allocation is not a valid kernel pointer, so the
            // no-mask case still allocates one byte and gates on `has_mask`.
            None => (vec![0u8; 1], 0),
            Some(m) => (m.iter().map(|&b| u8::from(b)).collect(), 1),
        };

        Ok(Self {
            n,
            vols,
            d_fpts: DeviceBuffer::from_host(backend, pts)?,
            has_pts,
            d_fsize: DeviceBuffer::from_host(backend, &fsize)?,
            d_forigin: DeviceBuffer::from_host(backend, &forigin)?,
            d_fmat: DeviceBuffer::from_host(backend, &fmat)?,
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

    /// Device bytes held by the two volumes — what the precision choice above is
    /// spending.
    pub fn volume_bytes(&self) -> usize {
        self.vols.bytes()
    }

    /// Evaluate the moments for the point map `x ↦ A·x + b`.
    ///
    /// `a` is row-major `3 × 3`, `b` is length 3. Deterministic: the same inputs
    /// give bit-identical moments on every call and every run.
    pub fn evaluate(&mut self, a: &[f64; 9], b: &[f64; 3]) -> Result<Moments, CudaError> {
        let backend: &Backend = backend()?;

        // Field-by-field, so the volumes can be matched on while the partials
        // buffer is borrowed mutably for the same launch.
        let Self {
            n,
            vols,
            d_fpts,
            has_pts,
            d_fsize,
            d_forigin,
            d_fmat,
            d_msize,
            d_mstride,
            d_morigin,
            d_mmat,
            d_mmask,
            has_mask,
            d_ab,
            d_partials,
            h_partials,
        } = self;

        let mut ab = [0.0f64; 12];
        ab[..9].copy_from_slice(a);
        ab[9..].copy_from_slice(b);
        d_ab.copy_from_host(backend, &ab)?;

        let n_i64 = *n as i64;
        let cfg = LaunchConfig {
            grid_dim: (GRID, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // The two instantiations differ in the element type of `fvals`/`mbuf` and
        // in nothing else — same sixteen arguments, same order, same grid.
        macro_rules! launch_moments {
            ($fscalar:expr, $mscalar:expr, $fvals:expr, $mbuf:expr) => {{
                let f = backend.function(kernel_src($fscalar, $mscalar), "ms_moments")?;
                let mut launch = backend.stream().launch_builder(&f);
                launch
                    .arg($fvals.device())
                    .arg(d_fpts.device())
                    .arg(&*has_pts)
                    .arg(d_fsize.device())
                    .arg(d_forigin.device())
                    .arg(d_fmat.device())
                    .arg(&n_i64)
                    .arg($mbuf.device())
                    .arg(d_msize.device())
                    .arg(d_mstride.device())
                    .arg(d_morigin.device())
                    .arg(d_mmat.device())
                    .arg(d_mmask.device())
                    .arg(&*has_mask)
                    .arg(d_ab.device())
                    .arg(d_partials.device_mut());
                // SAFETY: the sixteen arguments match the kernel's sixteen
                // parameters in order and type — `fvals` is `FSCALAR` and `mbuf` is
                // `MSCALAR`, which are the element types of the buffers this arm
                // matched.
                // `fvals` holds `n` elements, read only under `s < n`. `d_fpts`
                // holds `n*3` and is read only when `has_pts != 0`; otherwise it is
                // a one-element dummy and the kernel reads `d_fsize`/`d_forigin`/
                // `d_fmat` (3, 3 and 9 elements) instead, whose product-of-size
                // equals `n` so the derived index stays in the grid. The moving
                // geometry buffers hold exactly 3, 3, 3 and 9 elements as the
                // kernel indexes them; `d_mmask` is read only when `has_mask != 0`,
                // in which case it has one byte per moving voxel and the kernel
                // bounds-checks the index it builds; `d_ab` holds 12; `d_partials`
                // holds `GRID * NSLOT`, and the kernel writes
                // `blockIdx.x * NSLOT + k` for `blockIdx.x < GRID`, `k < NSLOT`.
                // Shared memory is declared statically at `BLOCK` doubles, matching
                // `block_dim`.
                unsafe { launch.launch(cfg)? };
            }};
        }
        match vols {
            Volumes::F64 { fvals, mbuf } => launch_moments!("double", "double", fvals, mbuf),
            Volumes::Split { fvals, mbuf } => launch_moments!("float", "double", fvals, mbuf),
        }

        d_partials.copy_to_host(backend, h_partials)?;
        backend.synchronize()?;

        // Fold the per-block partials in block-index order. Fixed order, on the
        // host, in f64 — this is the step that makes the result reproducible.
        let mut m = Moments::default();
        let mut count = 0.0f64;
        for blk in 0..GRID as usize {
            let p = &h_partials[blk * NSLOT..(blk + 1) * NSLOT];
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
