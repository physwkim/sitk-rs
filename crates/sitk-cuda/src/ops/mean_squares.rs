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
use crate::mask::DeviceMask;

/// Threads per block. The kernel's shared-memory tree is exactly this wide.
const BLOCK: u32 = 256;
/// Blocks. **Fixed**, not derived from the sample count: the reduction order —
/// and therefore the result — must not change with the input size. The kernel is
/// a grid-stride loop, so this covers any `n`.
const GRID: u32 = 512;
/// `sq` (1) + `S0[3]` (3) + `S1[3][3]` (9) + `count` (1).
const NSLOT: usize = 14;
/// The sample-set forms, mirroring the kernel's `MODE_*` defines. See [`FixedPoints`].
const MODE_GRID: i32 = 0;
const MODE_POINTS: i32 = 1;
const MODE_INDICES: i32 = 2;

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

// acc + a*b, with the multiply and the add rounded SEPARATELY.
//
// The chain that produces the continuous index -- fixed point x, mapped point p,
// index c -- must be bit-identical to the host's, because `floor(c)` and
// `is_inside(c)` are *branches*, and the trilinear interpolant's gradient is
// discontinuous across them. A 1-ULP difference in c is harmless for the value
// (which is continuous in c) but takes the OTHER one-sided gradient at a sample
// whose index lands exactly on a voxel plane -- and a fixed grid that maps onto
// the moving grid is not exotic, it is what the identity transform does on
// commensurate geometry. Measured on such data before this was fixed: the value
// agreed to 1e-15 and the derivative was off by 7%.
//
// Rust does not fuse a multiply and an add; NVRTC does, by default. `__dmul_rn` /
// `__dadd_rn` are the IEEE round-to-nearest primitives, which the compiler may not
// contract -- so the guarantee holds regardless of the flags this is compiled with,
// and it is stated in the code rather than in a build option. Everything downstream
// of `c` (weights, value, gradient, the reduction) is a continuous function of it
// and stays contracted: that is where the arithmetic is, and a ULP there is the
// reduction-rounding the metric is already gated at.
__device__ __forceinline__ double fmadd_rn(double acc, double a, double b) {
    return __dadd_rn(acc, __dmul_rn(a, b));
}

// Where a sample comes from. The kernel reads exactly one of `fpts` / `fidx`, and
// in MODE_GRID neither: the sample IS the grid voxel of the same index.
#define MODE_GRID    0
#define MODE_POINTS  1
#define MODE_INDICES 2

extern "C" __global__ void ms_moments(
    const FSCALAR* __restrict__ fvals,    // fixed values: n in MODE_POINTS, the whole grid otherwise
    const double* __restrict__ fpts,      // fixed sample points, n * 3 (row-major); MODE_POINTS only
    const int mode,
    const long long* __restrict__ fidx,   // sample -> fixed grid index, n; MODE_INDICES only
    const long long* __restrict__ fsize,  // 3, fixed grid size (MODE_GRID and MODE_INDICES)
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
    const unsigned char* __restrict__ fmask, // fixed-grid mask, n; unused if !has_fmask
    const int has_fmask,
    const double* __restrict__ ab,        // A (9, row-major) then b (3)
    double* __restrict__ partials)        // GRID * NSLOT
{
    double acc[NSLOT];
    for (int k = 0; k < NSLOT; ++k) acc[k] = 0.0;

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        // The sample's voxel in the FIXED GRID. In MODE_GRID sample `s` *is* grid
        // voxel `s`, so `gv == s` and every expression below is what it has always
        // been, term for term. In MODE_INDICES the host chose the voxels and `gv` is
        // the one it chose. MODE_POINTS has no grid index at all -- the host gathered
        // the values and the points itself -- and `gv` degenerates to `s`, which is how
        // that path indexes its own gathered arrays.
        const long long gv = (mode == MODE_INDICES) ? fidx[s] : s;

        // The fixed mask drops the sample before any work is done on it, gated by the
        // sample's GRID voxel. That is what makes the mask meaningful in MODE_INDICES
        // as well as MODE_GRID -- and what makes it meaningless in MODE_POINTS, where
        // there is no grid index and the two are refused together at construction.
        //
        // This `continue` does not perturb the reduction. The tree is a function of
        // (BLOCK, GRID, n) and nothing else, and within a thread the surviving terms
        // keep their order -- skipping a term removes it, it does not reorder the
        // rest. `is_inside` and the moving mask below already skip samples this way,
        // so a data-dependent valid count is the status quo, not something the fixed
        // mask introduces.
        if (has_fmask && !fmask[gv]) continue;

        // The sample's physical point. Unless the host uploaded points, it is a pure
        // function of the sample's grid voxel `gv` and the grid, so it is DERIVED here
        // rather than uploaded: at 256^3 the points array is 402 MB, which was 60% of
        // the only large transfer in the run. A sampled set derives it from the SAME
        // expression -- so sample `s` of a sampled run reads the point that voxel
        // `gv` reads in a full run, bit for bit, rather than a separately computed one.
        // Same arithmetic, same order, as the host's `write_point_at`.
        double x[3];
        if (mode == MODE_POINTS) {
            x[0] = fpts[s*3+0]; x[1] = fpts[s*3+1]; x[2] = fpts[s*3+2];
        } else {
            const double i = (double)(gv % fsize[0]);
            const double j = (double)((gv / fsize[0]) % fsize[1]);
            const double k = (double)(gv / (fsize[0] * fsize[1]));
            // VirtualGrid::point: the origin is the accumulator's seed, then one
            // rounded multiply-add per axis.
            for (int r = 0; r < 3; ++r) {
                double acc_r = forigin[r];
                acc_r = fmadd_rn(acc_r, fmat[r*3+0], i);
                acc_r = fmadd_rn(acc_r, fmat[r*3+1], j);
                acc_r = fmadd_rn(acc_r, fmat[r*3+2], k);
                x[r] = acc_r;
            }
        }

        // p = A*x + b --- the transform's point map, whatever transform it is.
        // `TransformBase::transform_point` is `mat_vec(matrix, x)` and THEN the
        // offset, so the accumulator starts at zero and `b` lands last.
        double p[3];
        for (int d = 0; d < 3; ++d) {
            double acc_d = 0.0;
            acc_d = fmadd_rn(acc_d, ab[d*3+0], x[0]);
            acc_d = fmadd_rn(acc_d, ab[d*3+1], x[1]);
            acc_d = fmadd_rn(acc_d, ab[d*3+2], x[2]);
            p[d] = __dadd_rn(acc_d, ab[9+d]);
        }

        // c = M * (p - origin): continuous index in the moving image. The host
        // subtracts the origin first, then runs `mat_vec` from zero.
        double c[3];
        for (int r = 0; r < 3; ++r) {
            double a = 0.0;
            for (int j = 0; j < 3; ++j) a = fmadd_rn(a, mmat[r*3+j], __dsub_rn(p[j], morigin[j]));
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

        // The fixed value at the sample's grid voxel. Same `gv` as the point and the
        // mask: value, point and gate cannot disagree about which voxel this is.
        const double diff = value - (double)fvals[gv];
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
/// [`Explicit`](Self::Explicit) is for a set whose points are *not* voxel centers of
/// the fixed grid — it carries the points themselves and nothing else.
///
/// [`Indices`](Self::Indices) is for a **sampled** set, and it is the one a sampling
/// strategy uses. A sampled set is not an arbitrary point cloud: our sampler does not
/// jitter (`sitk_registration::metric::FixedSamples::from_image_with`, a documented
/// deviation from ITK), so every sample is a voxel center and the sample is fully
/// described by its flat grid index. Saying it that way costs 8 bytes per sample
/// instead of 24, and buys two things the point list cannot:
///
/// - the point is *derived* from the same closed form [`Grid`](Self::Grid) uses, so a
///   sampled run reads what a full run reads at that voxel, bit for bit, rather than a
///   separately computed approximation of it;
/// - the sample knows its grid index, so a **fixed mask** — which gates by grid index —
///   still means something. That is the whole difference between this and
///   [`Explicit`](Self::Explicit), and it is why the mask is refused with one and
///   allowed with the other.
pub enum FixedPoints<'a> {
    /// One point per sample, row-major `N × 3`. The values are the host's gathered
    /// per-sample values, indexed by sample.
    Explicit(&'a [f64]),
    /// Every voxel of `size`, in dim-0-fastest order. `idx_to_phys` is row-major
    /// `3 × 3`; the sample count must equal the product of `size`.
    Grid {
        size: &'a [usize],
        origin: &'a [f64],
        idx_to_phys: &'a [f64],
    },
    /// Sample `s` is the fixed grid's flat voxel `idx[s]`, in the grid described by
    /// `size` / `origin` / `idx_to_phys`. Duplicates are allowed — `Random` draws with
    /// replacement — and the values are the whole grid's, indexed by `idx[s]`.
    Indices {
        idx: &'a [i64],
        size: &'a [usize],
        origin: &'a [f64],
        idx_to_phys: &'a [f64],
    },
}

pub struct ResidentMetric {
    n: usize,
    vols: Volumes,
    d_fpts: DeviceBuffer<f64>,
    /// Which of `d_fpts` / `d_fidx` the kernel reads, if either: one of `MODE_*`.
    mode: i32,
    d_fidx: DeviceBuffer<i64>,
    d_fsize: DeviceBuffer<i64>,
    d_forigin: DeviceBuffer<f64>,
    d_fmat: DeviceBuffer<f64>,
    d_msize: DeviceBuffer<i64>,
    d_mstride: DeviceBuffer<i64>,
    d_morigin: DeviceBuffer<f64>,
    d_mmat: DeviceBuffer<f64>,
    d_mmask: DeviceBuffer<u8>,
    has_mask: i32,
    d_fmask: DeviceBuffer<u8>,
    has_fmask: i32,
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
            None,
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
        Self::from_device_masked(fixed, fixed_points, None, moving, moving_geometry)
    }

    /// [`from_device`](Self::from_device) with a **fixed mask**: a sample whose voxel
    /// is zero in `fixed_mask` is not a sample at all, exactly as a zero voxel of the
    /// host's fixed mask drops that sample from `FixedSamples`.
    ///
    /// The mask lives on the *fixed grid* and is indexed by the sample's grid index,
    /// so it is only meaningful for [`FixedPoints::Grid`]. Combining it with
    /// [`FixedPoints::Explicit`] is refused by name
    /// ([`CudaError::MaskedExplicitPoints`]) rather than checked in the kernel: with
    /// an explicit point list the host has already selected the samples, and a mask
    /// indexed by position in that list is a different object with the same name.
    ///
    /// The mask must cover the fixed grid exactly — same voxel count — or
    /// [`CudaError::DegenerateInput`].
    pub fn from_device_masked(
        fixed: &DeviceImage,
        fixed_points: FixedPoints<'_>,
        fixed_mask: Option<&DeviceMask>,
        moving: &DeviceImage,
        moving_geometry: &MovingGeometry<'_>,
    ) -> Result<Self, CudaError> {
        if moving.len() != moving_geometry.len {
            return Err(CudaError::DegenerateInput);
        }
        // The sample count is the sample set's, not the volume's: a sampled run holds
        // the whole fixed image on the device (the values are gathered *in the kernel*,
        // by grid index) while evaluating only the voxels the host selected.
        let n = match &fixed_points {
            FixedPoints::Indices { idx, .. } => idx.len(),
            _ => fixed.len(),
        };
        let backend = backend()?;
        Self::build(
            n,
            Volumes::Split {
                // The fixed samples: narrow, one load per sample.
                fvals: DeviceBuffer::copy_of(backend, fixed.buffer().device())?,
                // The moving image: wide, eight loads per sample.
                mbuf: moving.widen_f64()?,
            },
            fixed_points,
            fixed_mask,
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
        fixed_mask: Option<&DeviceMask>,
        moving: &MovingGeometry<'_>,
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        let (fvals_len, mbuf_len) = vols.lens();
        if n == 0
            || mbuf_len != moving.len
            || moving.size.len() != DIM
            || moving.size.iter().product::<usize>() != moving.len
        {
            return Err(CudaError::DegenerateInput);
        }

        let as_i64 = |v: &[usize]| v.iter().map(|&x| x as i64).collect::<Vec<_>>();
        let grid_ok = |size: &[usize], origin: &[f64], idx_to_phys: &[f64]| {
            size.len() == DIM && origin.len() == DIM && idx_to_phys.len() == DIM * DIM
        };

        // A zero-length allocation is not a valid kernel pointer, so each unused arm
        // below still allocates a single dummy element; the kernel reads `fpts` only in
        // `MODE_POINTS` and `fidx` only in `MODE_INDICES`.
        //
        // `fvals` is indexed by the *sample's grid voxel*, so it holds the whole fixed
        // grid — except in `MODE_POINTS`, where there is no grid and the host gathered
        // one value per sample.
        let (pts, idx, mode, fsize, forigin, fmat) = match fixed_points {
            FixedPoints::Explicit(p) => {
                if p.len() != n * DIM || fvals_len != n {
                    return Err(CudaError::DegenerateInput);
                }
                (
                    p,
                    &[0i64][..],
                    MODE_POINTS,
                    vec![1i64; DIM],
                    vec![0.0; DIM],
                    vec![0.0; DIM * DIM],
                )
            }
            FixedPoints::Grid {
                size,
                origin,
                idx_to_phys,
            } => {
                if !grid_ok(size, origin, idx_to_phys)
                    || size.iter().product::<usize>() != n
                    || fvals_len != n
                {
                    return Err(CudaError::DegenerateInput);
                }
                (
                    &[0.0f64][..],
                    &[0i64][..],
                    MODE_GRID,
                    as_i64(size),
                    origin.to_vec(),
                    idx_to_phys.to_vec(),
                )
            }
            FixedPoints::Indices {
                idx,
                size,
                origin,
                idx_to_phys,
            } => {
                let voxels = size.iter().product::<usize>();
                if !grid_ok(size, origin, idx_to_phys) || idx.len() != n || fvals_len != voxels {
                    return Err(CudaError::DegenerateInput);
                }
                // Every sample must name a voxel of the grid. Checked here, once, rather
                // than in the kernel: an out-of-range index would read outside the volume,
                // and clamping it would silently sample the wrong voxel.
                if let Some(&bad) = idx.iter().find(|&&i| i < 0 || i as usize >= voxels) {
                    return Err(CudaError::SampleIndexOutOfGrid { index: bad, voxels });
                }
                (
                    &[0.0f64][..],
                    idx,
                    MODE_INDICES,
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
            Some(m) => {
                // The kernel indexes this by the *moving* grid's flat index, so a mask
                // that is not that grid would gate the wrong voxels — or read past the
                // buffer. Refused, not clamped.
                if m.len() != moving.len {
                    return Err(CudaError::DegenerateInput);
                }
                (m.iter().map(|&b| u8::from(b)).collect(), 1)
            }
        };

        // **A fixed mask requires a sample set that knows its grid index.** Enforced
        // here, so the kernel never has to ask: it gates on `fmask[g]`, and `g` is a
        // grid voxel in `MODE_GRID` and `MODE_INDICES` alike.
        //
        // `MODE_POINTS` is the one that cannot satisfy it. A bare point list is a
        // host-selected subset in an arbitrary order that has *thrown the grid index
        // away*; a mask indexed into that list is a different object with the same
        // name, and would silently gate the wrong samples. Refused by name, not
        // clamped — and it stays refused now that `MODE_INDICES` exists, because an
        // index list is precisely the thing that kept the index.
        let (d_fmask, has_fmask) = match fixed_mask {
            None => (DeviceBuffer::zeros(backend, 1)?, 0),
            Some(m) => {
                if mode == MODE_POINTS {
                    return Err(CudaError::MaskedExplicitPoints);
                }
                // The mask is indexed by the fixed grid's *flat* index, so it must be
                // that grid — the same voxel count on a different shape indexes
                // different voxels, and would gate silently wrong ones.
                if as_i64(&m.geometry().size) != fsize {
                    return Err(CudaError::DegenerateInput);
                }
                (DeviceBuffer::copy_of(backend, m.buffer().device())?, 1)
            }
        };

        Ok(Self {
            n,
            vols,
            d_fpts: DeviceBuffer::from_host(backend, pts)?,
            mode,
            d_fidx: DeviceBuffer::from_host(backend, idx)?,
            d_fsize: DeviceBuffer::from_host(backend, &fsize)?,
            d_forigin: DeviceBuffer::from_host(backend, &forigin)?,
            d_fmat: DeviceBuffer::from_host(backend, &fmat)?,
            d_msize: DeviceBuffer::from_host(backend, &as_i64(moving.size))?,
            d_mstride: DeviceBuffer::from_host(backend, &as_i64(moving.strides))?,
            d_morigin: DeviceBuffer::from_host(backend, moving.origin)?,
            d_mmat: DeviceBuffer::from_host(backend, moving.phys_to_index)?,
            d_mmask: DeviceBuffer::from_host(backend, &mask_bytes)?,
            has_mask,
            d_fmask,
            has_fmask,
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
            mode,
            d_fidx,
            d_fsize,
            d_forigin,
            d_fmat,
            d_msize,
            d_mstride,
            d_morigin,
            d_mmat,
            d_mmask,
            has_mask,
            d_fmask,
            has_fmask,
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
        // in nothing else — same eighteen arguments, same order, same grid.
        macro_rules! launch_moments {
            ($fscalar:expr, $mscalar:expr, $fvals:expr, $mbuf:expr) => {{
                let f = backend.function(kernel_src($fscalar, $mscalar), "ms_moments")?;
                let mut launch = backend.stream().launch_builder(&f);
                launch
                    .arg($fvals.device())
                    .arg(d_fpts.device())
                    .arg(&*mode)
                    .arg(d_fidx.device())
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
                    .arg(d_fmask.device())
                    .arg(&*has_fmask)
                    .arg(d_ab.device())
                    .arg(d_partials.device_mut());
                // SAFETY: the nineteen arguments match the kernel's nineteen
                // parameters in order and type — `fvals` is `FSCALAR` and `mbuf` is
                // `MSCALAR`, which are the element types of the buffers this arm
                // matched.
                //
                // Every read the kernel makes of the fixed side is indexed by `gv`,
                // and `gv` is a voxel of the fixed grid in every mode: it is `s < n`
                // in `MODE_GRID` (where `build` checked the grid's product-of-size
                // equals `n`) and `MODE_POINTS` (where `fvals` holds `n` gathered
                // values), and it is `fidx[s]` in `MODE_INDICES`, where `build` has
                // checked *every* index against the grid's voxel count and `fvals`
                // holds that many. So `fvals[gv]` and `fmask[gv]` are in bounds in all
                // three, and the point derived from `gv` stays in the grid.
                //
                // `d_fpts` holds `n*3` and is read only in `MODE_POINTS`; `d_fidx`
                // holds `n` and is read only in `MODE_INDICES`; each is a one-element
                // dummy otherwise, and neither is a valid allocation at length zero,
                // which is why the dummy exists. `d_fsize`/`d_forigin`/`d_fmat` hold
                // 3, 3 and 9. The moving geometry buffers hold exactly 3, 3, 3 and 9
                // elements as the kernel indexes them; `d_mmask` is read only when
                // `has_mask != 0`, in which case it has one byte per moving voxel and
                // the kernel bounds-checks the index it builds. `d_fmask` is read only
                // when `has_fmask != 0`, in which case `build` has checked it covers
                // the fixed grid. `d_ab` holds 12; `d_partials` holds `GRID * NSLOT`,
                // and the kernel writes `blockIdx.x * NSLOT + k` for
                // `blockIdx.x < GRID`, `k < NSLOT`. Shared memory is declared
                // statically at `BLOCK` doubles, matching `block_dim`.
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
