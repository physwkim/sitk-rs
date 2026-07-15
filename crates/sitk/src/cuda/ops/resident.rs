//! What every device metric shares: the resident volumes, the sample set, the
//! masks, the point-map upload, the **sampler**, and the reduction.
//!
//! # Why this is one module and not a copy per metric
//!
//! A device metric is two things: a *sampler* — fixed voxel → physical point →
//! transform → continuous index → validity → trilinear value and gradient — and an
//! *accumulator*, which is the only part that knows which metric it is. The sampler
//! is where the host-parity contract lives, and it is unforgiving: the chain that
//! produces the continuous index must be bit-identical to the host's, because
//! `floor(c)` and `is_inside(c)` are **branches** and the interpolant's gradient is
//! discontinuous across them. The mean-squares kernel carries the scar — before its
//! multiply-adds were pinned to `__dmul_rn`/`__dadd_rn`, the value agreed to 1e-15
//! and the derivative was off by **7%**.
//!
//! A second metric with its own copy of that chain is a second chance to drift from
//! the host, and drift there is silent: the value stays right and the derivative
//! goes wrong. So there is exactly one copy — [`SAMPLER_SRC`] — and both the
//! mean-squares kernel and the two correlation kernels are built from it. Same for
//! the reduction: one shared-memory tree, one host-side fold in block order, so
//! *determinism* is a property of this module rather than a habit each metric has to
//! remember.
//!
//! What a metric supplies is its accumulator and its slot count. Nothing else.

use cudarc::driver::LaunchConfig;

use crate::cuda::backend::{Backend, backend};
use crate::cuda::buffer::DeviceBuffer;
use crate::cuda::error::CudaError;
use crate::cuda::image::DeviceImage;
use crate::cuda::mask::DeviceMask;

/// Threads per block. The kernel's shared-memory tree is exactly this wide.
pub(crate) const BLOCK: u32 = 256;
/// Blocks. **Fixed**, not derived from the sample count: the reduction order —
/// and therefore the result — must not change with the input size. The kernels are
/// grid-stride loops, so this covers any `n`.
pub(crate) const GRID: u32 = 512;

/// The sample-set forms, mirroring the sampler's `MODE_*` defines. See [`FixedPoints`].
pub(crate) const MODE_GRID: i32 = 0;
pub(crate) const MODE_POINTS: i32 = 1;
pub(crate) const MODE_INDICES: i32 = 2;

/// This backend handles 3-D only. The moment algebra generalizes, but the kernels
/// are written for `dim = 3` and a 2-D caller falls back to the CPU.
pub const DIM: usize = 3;

/// One stage of the transform's point map: `p ↦ mat_vec(matrix, p) + offset`, with
/// `matrix` row-major `3 × 3`.
///
/// The caller hands the device the transform's **stored** matrix and offset — the
/// fields its own `transform_point` evaluates — and one stage per map the host
/// applies, in the host's application order. Not a form probed out of the transform,
/// and not several maps folded into one: either would be algebraically equal to the
/// host and differ from it in the last bits, and the last bits of the continuous
/// index decide `floor`, `is_inside` and `round`, which are branches. See
/// [`SAMPLER_SRC`] for the argument in full.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointStage {
    /// Row-major `3 × 3`.
    pub matrix: [f64; DIM * DIM],
    pub offset: [f64; DIM],
}

/// The most stages the device replays in one point map.
///
/// A composed registration transform is two stages (the optimized transform and a
/// moving-initial), and a composite of a handful more; there is no legitimate map
/// with dozens. The cap keeps the stage buffer a fixed-size allocation and the
/// replay a bounded loop, and a longer map is refused by name
/// ([`CudaError::PointMapStageCount`]) rather than truncated.
pub const MAX_STAGES: usize = 8;

/// The sampler, the mode defines and the reduction — prepended to every metric
/// kernel. `FSCALAR`/`MSCALAR` are `#define`d by [`kernel_src`](Self) before this
/// is concatenated.
///
/// `take_sample<WANT_GRAD>` is the single place the host-parity chain exists. A
/// metric kernel calls it and accumulates; it does not re-derive a point, an index,
/// or a validity test. `WANT_GRAD` is a template parameter rather than a runtime
/// flag so the gradient arithmetic is *compiled out* of the pass that does not want
/// it (correlation's first pass) with no branch and no divergence.
pub(crate) const SAMPLER_SRC: &str = r#"
#define BLOCK 256

// Mirrors crate::transform::interpolator::is_inside exactly: a sample is valid iff
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
// and it is stated in the code rather than in a build option.
//
// The pin covers the interpolant's weights, value and gradient too, and THAT IS NOT
// DECORATION -- it is the second discrete consumer, found when Mattes arrived. The
// comment that used to sit here said the arithmetic downstream of `c` could stay
// contracted because "a ULP there is the reduction-rounding the metric is already
// gated at". That was true of every metric that existed at the time and is FALSE of
// Mattes: mean squares and correlation feed the interpolated value into `diff` and a
// sum -- continuous, no branch -- whereas Mattes feeds it into
//
//     index = (long long)(mv / bin_size - normalized_min)
//
// a TRUNCATION, i.e. a branch, exactly like `floor(c)` one step earlier. A 1-ULP `mv`
// at a sample whose term lands on a bin boundary picks the other Parzen bin, moves
// half a unit of mass into a neighbouring histogram cell, and no longer agrees with
// the host on the bits. So the interpolated value is now a bit-identity surface, and
// it is pinned here, in the ONE sampler, rather than in a Mattes-only copy of the
// chain -- a second copy is what this module exists to forbid.
__device__ __forceinline__ double fmadd_rn(double acc, double a, double b) {
    return __dadd_rn(acc, __dmul_rn(a, b));
}

// Where a sample comes from. The sampler reads exactly one of `fpts` / `fidx`, and
// in MODE_GRID neither: the sample IS the grid voxel of the same index.
#define MODE_GRID    0
#define MODE_POINTS  1
#define MODE_INDICES 2

// One valid sample, as every metric sees it.
struct Sample {
    double x[3];    // the fixed sample's physical point
    double fval;    // the fixed value at that sample's grid voxel
    double value;   // the moving image at T(x)
    double g[3];    // the moving gradient at T(x), physical space (WANT_GRAD only)
};

// The sampler. Returns false if the sample is not valid -- dropped by the fixed
// mask, by the moving mask, or by mapping outside the moving image -- in which case
// `out` is untouched and the caller accumulates nothing.
//
// This is the whole of the host-parity contract, in one place, for every metric.
template <bool WANT_GRAD>
__device__ __forceinline__ bool take_sample(
    const long long s,
    const FSCALAR* __restrict__ fvals,
    const double* __restrict__ fpts,
    const int mode,
    const long long* __restrict__ fidx,
    const long long* __restrict__ fsize,
    const double* __restrict__ forigin,
    const double* __restrict__ fmat,
    const MSCALAR* __restrict__ mbuf,
    const long long* __restrict__ msize,
    const long long* __restrict__ mstride,
    const double* __restrict__ morigin,
    const double* __restrict__ mmat,
    const unsigned char* __restrict__ mmask,
    const int has_mask,
    const unsigned char* __restrict__ fmask,
    const int has_fmask,
    const double* __restrict__ stages,
    const int nstage,
    Sample* out)
{
    // The sample's voxel in the FIXED GRID. In MODE_GRID sample `s` *is* grid
    // voxel `s`, so `gv == s`. In MODE_INDICES the host chose the voxels and `gv` is
    // the one it chose. MODE_POINTS has no grid index at all -- the host gathered
    // the values and the points itself -- and `gv` degenerates to `s`, which is how
    // that path indexes its own gathered arrays.
    const long long gv = (mode == MODE_INDICES) ? fidx[s] : s;

    // The fixed mask drops the sample before any work is done on it, gated by the
    // sample's GRID voxel. That is what makes the mask meaningful in MODE_INDICES
    // as well as MODE_GRID -- and what makes it meaningless in MODE_POINTS, where
    // there is no grid index and the two are refused together at construction.
    //
    // Skipping does not perturb the reduction. The tree is a function of
    // (BLOCK, GRID, n) and nothing else, and within a thread the surviving terms
    // keep their order -- dropping a term removes it, it does not reorder the rest.
    if (has_fmask && !fmask[gv]) return false;

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

    // The transform's point map: `nstage` stages of `p <- mat_vec(M, p) + t`, each
    // 12 doubles (M row-major, then t), replayed IN THE HOST'S ORDER.
    //
    // The stages are the transform's OWN stored matrix/offset pairs, not a form
    // probed out of it, and they are replayed rather than folded into one map.
    // Both of those are bit-identity requirements, not style:
    //
    //   - A probe (`b = T(0)`, `A[:,e] = T(e_e) - b`) recovers the matrix through a
    //     subtraction that cancels the offset back off -- algebraically exact, not
    //     bitwise. The stored fields have no such cancellation.
    //   - Folding two stages into one matrix product is algebraically equal and NOT
    //     bit equal: the host rounds ONCE PER STAGE (`Composed::transform_point`
    //     applies its maps in sequence), so the device must round once per stage
    //     too. A transform whose host evaluation is not this expression has no
    //     stages and is refused on the host, by name.
    //
    // Within a stage, `mat_vec` starts the accumulator at zero and the offset lands
    // last, which is exactly `crate::core::matrix::mat_vec` followed by the `+ offset`
    // of `MatrixOffsetTransformBase::transform_point`.
    double p[3] = { x[0], x[1], x[2] };
    for (int st = 0; st < nstage; ++st) {
        const double* ab = stages + st * 12;
        double q[3];
        for (int d = 0; d < 3; ++d) {
            double acc_d = 0.0;
            acc_d = fmadd_rn(acc_d, ab[d*3+0], p[0]);
            acc_d = fmadd_rn(acc_d, ab[d*3+1], p[1]);
            acc_d = fmadd_rn(acc_d, ab[d*3+2], p[2]);
            q[d] = __dadd_rn(acc_d, ab[9+d]);
        }
        p[0] = q[0]; p[1] = q[1]; p[2] = q[2];
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
        if (!ok || !mmask[flat]) return false;
    }
    if (!is_inside(c, msize)) return false;

    // Trilinear value + exact gradient of the interpolant, in the same corner
    // order, with the same clamping, and -- since Mattes -- with the same ROUNDING as
    // `linear_value_and_gradient`: every product and every sum below is rounded
    // separately, because the host's is (Rust does not fuse) and because the value
    // this produces is truncated into a Parzen bin. See `fmadd_rn`.
    double base[3], frac[3];
    for (int d = 0; d < 3; ++d) {
        const double f = floor(c[d]);
        base[d] = f;
        frac[d] = __dsub_rn(c[d], f);
    }
    double value = 0.0;
    double gi[3] = { 0.0, 0.0, 0.0 };
    for (int corner = 0; corner < 8; ++corner) {
        long long offset = 0;
        double weight = 1.0;
        for (int d = 0; d < 3; ++d) {
            const int bit = (corner >> d) & 1;
            weight = __dmul_rn(weight, bit ? frac[d] : __dsub_rn(1.0, frac[d]));
            long long idx = (long long)base[d] + bit;
            if (idx < 0) idx = 0;
            if (idx > msize[d] - 1) idx = msize[d] - 1;
            offset += idx * mstride[d];
        }
        const double b = (double)mbuf[offset];
        value = fmadd_rn(value, weight, b);
        if (WANT_GRAD) {
            for (int j = 0; j < 3; ++j) {
                double w = 1.0;
                for (int d = 0; d < 3; ++d) {
                    if (d == j) continue;
                    const int bit = (corner >> d) & 1;
                    w = __dmul_rn(w, bit ? frac[d] : __dsub_rn(1.0, frac[d]));
                }
                const double sign = ((corner >> j) & 1) ? 1.0 : -1.0;
                // The host writes `sign * w_without_j * b`, left to right: `sign` is
                // +/-1 so `sign*w` is exact, and the product with `b` is the rounding
                // that matters.
                gi[j] = fmadd_rn(gi[j], __dmul_rn(sign, w), b);
            }
        }
    }

    if (WANT_GRAD) {
        // Index-space gradient -> physical-space: g[d] = sum_j gi[j] * M[j][d]. The
        // host's `.map(..).sum()` seeds the accumulator at 0.0 and adds three
        // separately-rounded products, in j order.
        for (int d = 0; d < 3; ++d) {
            double acc_g = 0.0;
            acc_g = fmadd_rn(acc_g, gi[0], mmat[0*3+d]);
            acc_g = fmadd_rn(acc_g, gi[1], mmat[1*3+d]);
            acc_g = fmadd_rn(acc_g, gi[2], mmat[2*3+d]);
            out->g[d] = acc_g;
        }
    }
    out->x[0] = x[0]; out->x[1] = x[1]; out->x[2] = x[2];
    out->value = value;
    // The fixed value at the sample's grid voxel. Same `gv` as the point and the
    // gate: value, point and mask cannot disagree about which voxel this is.
    out->fval = (double)fvals[gv];
    return true;
}

// The reduction, shared by every metric: a fixed shared-memory tree, one slot at a
// time. No atomics -- the order is a function of (BLOCK, GRID, n) alone, so it is
// identical on every run and so is the result. Every thread must reach this.
__device__ __forceinline__ void emit_partials(
    const double* acc, const int nslot, double* __restrict__ partials)
{
    __shared__ double sh[BLOCK];
    const int tid = threadIdx.x;
    for (int k = 0; k < nslot; ++k) {
        sh[tid] = acc[k];
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        if (tid == 0) partials[blockIdx.x * nslot + k] = sh[0];
        __syncthreads();
    }
}
"#;

/// The fixed samples and the moving volume, in the precision the kernel reads them.
///
/// Not a tuning knob for the caller: the *host-producer* path accepts any pixel
/// type, including `Float64` and `Int64`, so it stays [`F64`](Volumes::F64) —
/// narrowing there would drop bits the caller handed us. The *device* path reads
/// `f32` images, so narrowing is lossless and the only question is which volume to
/// narrow.
///
/// It narrows the **fixed** samples and leaves the moving image wide, because the
/// two are not symmetric: one fixed load per sample against eight moving loads (the
/// trilinear corners), while the memory splits evenly between them.
pub(crate) enum Volumes {
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
    pub(crate) fn lens(&self) -> (usize, usize) {
        match self {
            Volumes::F64 { fvals, mbuf } => (fvals.len(), mbuf.len()),
            Volumes::Split { fvals, mbuf } => (fvals.len(), mbuf.len()),
        }
    }

    /// Device bytes held by the two volumes.
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Volumes::F64 { fvals, mbuf } => 8 * fvals.len() + 8 * mbuf.len(),
            Volumes::Split { fvals, mbuf } => 4 * fvals.len() + 8 * mbuf.len(),
        }
    }

    /// The `(FSCALAR, MSCALAR)` this arm compiles the kernel for.
    pub(crate) fn scalars(&self) -> (&'static str, &'static str) {
        match self {
            Volumes::F64 { .. } => ("double", "double"),
            Volumes::Split { .. } => ("float", "double"),
        }
    }

    /// Both volumes, already in the device buffers the kernel takes as arguments.
    /// The two arms differ in the element type of `fvals` and in nothing else, which
    /// is why every launch site matches on this rather than duplicating an arg list.
    pub(crate) fn from_device(
        fixed: &DeviceImage,
        moving: &DeviceImage,
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        Ok(Volumes::Split {
            // The fixed samples: narrow, one load per sample.
            fvals: DeviceBuffer::copy_of(backend, fixed.buffer().device())?,
            // The moving image: wide, eight loads per sample.
            mbuf: moving.widen_f64()?,
        })
    }
}

/// The moving image's geometry, as the kernel needs it. The voxels themselves
/// arrive through the producer argument of the host-side constructor, not here —
/// the host holds them in their native pixel type, so there is no `f64` slice to
/// borrow.
pub struct MovingGeometry<'a> {
    /// Voxel count; must equal the product of `size`.
    pub len: usize,
    pub size: &'a [usize],
    pub strides: &'a [usize],
    pub origin: &'a [f64],
    /// `inverse(Direction · diag(spacing))`, row-major `3 × 3` — the inverse of
    /// the whole composed matrix (ITK `itkImageBase.hxx:175`,
    /// `crate::core::coord::physical_to_index_matrix`), not the direction alone
    /// divided by spacing. The two agree for a diagonal geometry and diverge for
    /// an oblique direction.
    pub phys_to_index: &'a [f64],
    pub mask: Option<&'a [bool]>,
}

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
/// jitter (`crate::registration::metric::FixedSamples::from_image_with`, a documented
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

/// The volumes, the sample set, the masks and the point map — everything a metric
/// kernel reads that is not its own accumulator.
///
/// Built once per pyramid level and evaluated against any number of transforms
/// without re-uploading anything: [`upload_point_map`](Self::upload_point_map)
/// moves **96 bytes per point-map stage up** per iteration, and the metric's partials
/// come back down.
pub(crate) struct Resident {
    pub(crate) n: usize,
    pub(crate) vols: Volumes,
    pub(crate) d_fpts: DeviceBuffer<f64>,
    /// Which of `d_fpts` / `d_fidx` the kernel reads, if either: one of `MODE_*`.
    pub(crate) mode: i32,
    pub(crate) d_fidx: DeviceBuffer<i64>,
    pub(crate) d_fsize: DeviceBuffer<i64>,
    pub(crate) d_forigin: DeviceBuffer<f64>,
    pub(crate) d_fmat: DeviceBuffer<f64>,
    pub(crate) d_msize: DeviceBuffer<i64>,
    pub(crate) d_mstride: DeviceBuffer<i64>,
    pub(crate) d_morigin: DeviceBuffer<f64>,
    pub(crate) d_mmat: DeviceBuffer<f64>,
    pub(crate) d_mmask: DeviceBuffer<u8>,
    pub(crate) has_mask: i32,
    pub(crate) d_fmask: DeviceBuffer<u8>,
    pub(crate) has_fmask: i32,
    /// The point map's stages, `MAX_STAGES * 12` doubles, of which the first
    /// `nstage * 12` are live. Reused across iterations: the per-iteration H2D writes
    /// into this rather than allocating (`copy_from_host`, not `from_host`).
    pub(crate) d_stages: DeviceBuffer<f64>,
    /// How many of them the kernel replays. Set by [`Resident::upload_point_map`].
    pub(crate) nstage: i32,
}

impl Resident {
    /// Validate the sample set, the masks and the geometry, and upload everything
    /// that does not change from iteration to iteration.
    pub(crate) fn build(
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
        // here, so the kernel never has to ask: it gates on `fmask[gv]`, and `gv` is a
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
            d_stages: DeviceBuffer::zeros(backend, MAX_STAGES * 12)?,
            // No point map has been uploaded yet. A zero-stage replay is the identity,
            // which would be a *plausible* wrong answer, so `upload_point_map` refuses
            // an empty stage list and every metric calls it before every launch.
            nstage: 0,
        })
    }

    /// The sample count the metric evaluates — the sample *set's*, not the volume's:
    /// a sampled run holds the whole fixed image on the device while evaluating only
    /// the voxels the host selected.
    pub(crate) fn sample_count_of(fixed: &DeviceImage, points: &FixedPoints<'_>) -> usize {
        match points {
            FixedPoints::Indices { idx, .. } => idx.len(),
            _ => fixed.len(),
        }
    }

    /// Push this iteration's point map — `stages.len()` stages of 12 doubles — into
    /// the buffer the kernel reads. The only per-iteration H2D in a registration run
    /// (96 bytes per stage).
    ///
    /// Refuses an empty list, and a list longer than [`MAX_STAGES`], by name: the
    /// device replays what the host evaluates or it does not run at all.
    pub(crate) fn upload_point_map(
        &mut self,
        backend: &Backend,
        stages: &[PointStage],
    ) -> Result<(), CudaError> {
        if stages.is_empty() || stages.len() > MAX_STAGES {
            return Err(CudaError::PointMapStageCount {
                stages: stages.len(),
                max: MAX_STAGES,
            });
        }
        let mut flat = [0.0f64; MAX_STAGES * 12];
        for (st, chunk) in stages.iter().zip(flat.chunks_exact_mut(12)) {
            chunk[..9].copy_from_slice(&st.matrix);
            chunk[9..].copy_from_slice(&st.offset);
        }
        self.nstage = stages.len() as i32;
        // The whole buffer, not the live prefix: one fixed-size transfer, and the
        // dead tail is never read (`nstage` bounds the replay).
        self.d_stages.copy_from_host(backend, &flat)
    }

    /// The launch geometry every metric kernel uses. Fixed, so the reduction order is.
    pub(crate) fn launch_config() -> LaunchConfig {
        LaunchConfig {
            grid_dim: (GRID, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// The per-block partials of one reduction, and the host buffer they land in.
///
/// The **fold is the determinism**: the device writes `GRID` partials per slot and
/// the host adds them in **block-index order**, in `f64`. Every run performs the
/// same additions in the same order, so the result is bit-identical run to run —
/// which is a correctness property, not a performance one, because the optimizer is
/// a feedback loop and a metric that varies run to run makes the *registration
/// result* vary run to run.
///
/// It is *not* bit-identical to the host's sequential sum, and cannot be: no
/// parallel reduction reproduces a left-to-right `f64` accumulation. The divergence
/// is reduction-rounding only (~√N·ε).
pub(crate) struct Partials {
    nslot: usize,
    d: DeviceBuffer<f64>,
    /// Reused host destination, so the per-iteration D2H never touches a fresh page.
    /// The first-touch defect, closed by construction: the buffer outlives the call.
    h: Vec<f64>,
}

impl Partials {
    pub(crate) fn new(backend: &Backend, nslot: usize) -> Result<Self, CudaError> {
        Ok(Self {
            nslot,
            d: DeviceBuffer::zeros(backend, GRID as usize * nslot)?,
            h: vec![0.0; GRID as usize * nslot],
        })
    }

    pub(crate) fn device_mut(&mut self) -> &mut DeviceBuffer<f64> {
        &mut self.d
    }

    /// Copy the partials down and fold them in block order. Returns one `f64` per
    /// slot.
    pub(crate) fn fold(&mut self, backend: &Backend) -> Result<Vec<f64>, CudaError> {
        self.d.copy_to_host(backend, &mut self.h)?;
        backend.synchronize()?;

        let mut out = vec![0.0f64; self.nslot];
        for blk in 0..GRID as usize {
            let p = &self.h[blk * self.nslot..(blk + 1) * self.nslot];
            for (o, &v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        Ok(out)
    }
}
