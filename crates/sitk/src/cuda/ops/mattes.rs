//! Mattes mutual information on the device: the joint Parzen histogram, and the
//! derivative moments taken against it.
//!
//! # The fan-out, and why it does not cost the histogram its guarantee
//!
//! Mattes estimates the joint density with a Parzen window, and the two axes get
//! **different** windows — this is the first thing to get right, because it sets the
//! size of everything downstream. The *fixed* axis uses a zero-order (box) window:
//! one bin per sample. Only the *moving* axis uses the cubic B-spline, which is
//! supported on four bins. So a sample's contribution is a **1 × 4** row, not a 4 × 4
//! block: `MattesMutualInformationImageToImageMetricv4::ComputeSingleFixedImageParzenWindowIndex`
//! picks one fixed bin, and `ProcessPoint` spreads the moving intensity over
//! `[movingIndex − 1 .. movingIndex + 2]`. The host does exactly this
//! (`MattesMutualInformationMetric::build_histogram`), so the entry list is **4n**, not
//! 16n.
//!
//! That fan-out does **not** weaken [`crate::cuda::histogram`]'s determinism argument, and it
//! is worth being precise about why, because the argument is the whole reason this
//! metric can exist. The counting sort promises one thing: bin `b`'s entries are summed
//! **in ascending entry index**. It does not care where the entries came from or how
//! many an input produced. Flattening `(fixed_bin, moving_bin)` to `fixed·bins + moving`
//! is just a key, and emitting entry `4s + j` for sample `s`'s `j`-th Parzen tap makes
//! the entry order lexicographic in `(s, j)` — which is **exactly** the order the host's
//! two nested loops add in. So the device histogram is bit-identical to
//!
//! ```text
//! for s in samples { for j in 0..4 { joint[key(s, j)] += B3(arg(s, j)) } }
//! ```
//!
//! which is the host's loop, not a re-association of it. The fan-out widens `n` by 4×
//! and buys nothing else: the sort is `O(n)` in the entry count, so it costs 4× the
//! scatter traffic and 4× the `sorted` buffer, and the bin count goes from `bins` to
//! `bins² + 1`, which sizes the counting sort's `ntiles × nbins` matrices (at the
//! default 50 bins: 2 501 bins, 20 MB of counts and 40 MB of cursors, allocated once
//! per level and reused).
//!
//! # The dead bin, instead of a compaction
//!
//! A sample that maps outside the moving image, or is dropped by a mask, or whose
//! interpolated value falls outside the histogram's moving range, contributes nothing —
//! the host `continue`s. The device cannot `continue`: entry `4s + j` must exist for the
//! entry list to stay index-aligned with the samples, and compacting the list would need
//! a prefix sum, i.e. a second thing to keep deterministic.
//!
//! So a dropped sample emits its four entries into **bin `bins²`**, a dead bin that
//! exists only to receive them, with value `0.0`; the marginal's dead bin is `bins`. The
//! sums of the live bins are untouched by this — a dead entry is in a *different*
//! segment, and a segment's order is the ascending order of the entries *in it*. The
//! dead bin's own sum is discarded. This is the compaction, done by the sort, for free.
//!
//! # The derivative, and the one expression that is not bit-identical
//!
//! The value above is bit-identical to the host. The derivative is **not**, and the
//! reason is a single named expression: `ParametricTransform::jacobian_wrt_parameters`.
//!
//! ITK accumulates a second, *derivative* joint histogram of width `nparams` —
//! `jointPDFDerivatives[(f·bins + m)·nparams + k] += (∇M · J[·][k]) · B3′(arg)` — and the
//! host port does the same. To reproduce that array bit for bit the device would have to
//! evaluate the transform's own `jacobian_wrt_parameters` at every sample, which is the
//! one thing this backend deliberately does not know how to do: the device is told the
//! Jacobian only as the probed affine decomposition `J(x) = J(0) + Σ_e x_e·(J(e_e) − J(0))`,
//! whose `C_e` is recovered by a **cancelling subtraction** — algebraically exact, wrong
//! in the last bits, and exactly the shape of defect the point-map stage list exists to
//! reject.
//!
//! But it is rejected *there* because the point map feeds `floor`, `is_inside` and
//! `round` — branches. **Nothing discrete depends on the Jacobian.** Every discrete
//! decision in Mattes — sample validity, the moving-range reject, and both Parzen bin
//! indices — is a function of the continuous index `c` and the interpolated value `mv`,
//! and both of those are pinned bitwise (`c` by the stage replay, `mv` by the sampler's
//! rounded multiply-adds, which Mattes is the reason for: it is the first metric to
//! *truncate* the interpolated value rather than merely add it). So the derivative's
//! band cannot move a bin, and the value it is taken against is exact.
//!
//! Given that, the `bins² × nparams` array is not worth materializing. Substituting the
//! affine decomposition and exchanging the sums collapses it to **twelve** numbers that
//! do not mention the parameters at all — the same identity the mean-squares metric
//! runs on, applied to the pRatio-weighted sample:
//!
//! ```text
//! wᵢⱼ      = pRatio[bin(i,j)] · B3′(arg(i,j))       (pRatio already carries 1/(binSize·N))
//! A[d]     = Σᵢⱼ wᵢⱼ · ∇Mᵢ[d]                        (3)
//! B[d][e]  = Σᵢⱼ wᵢⱼ · ∇Mᵢ[d] · xᵢ[e]                (9)
//!
//! ∂value/∂pₖ = Σ_d J(0)[d][k]·A[d] + Σ_d Σ_e C_e[d][k]·B[d][e]
//! ```
//!
//! so the device cost is independent of the parameter count and the host does the
//! contraction in `f64` with the transform's own Jacobian. It needs the finished pRatio
//! table, so it is a **second pass** (20 KB up per iteration), exactly as the host's
//! own sparse-support path re-walks the samples once the histogram is known.
//!
//! # Determinism
//!
//! The histogram is the counting sort's, bit-identical to the host's loop and invariant
//! to the launch configuration. The twelve derivative moments go through the shared
//! reduction tree and the host's block-order fold — no atomics, fixed grid — so they are
//! bit-identical run to run. Neither half re-opens the hole
//! [`crate::cuda::histogram_atomic`] demonstrates.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::cuda::backend::{Backend, backend};
use crate::cuda::buffer::DeviceBuffer;
use crate::cuda::error::CudaError;
use crate::cuda::image::DeviceImage;
use crate::cuda::mask::DeviceMask;
use crate::cuda::ops::histogram::HistogramScratch;
use crate::cuda::ops::resident::{
    BLOCK, DIM, FixedPoints, GRID, MovingGeometry, Partials, PointStage, Resident, SAMPLER_SRC,
    Volumes,
};

/// Taps of the cubic B-spline Parzen window on the moving axis. The fixed axis has a
/// box window (one tap), so a sample is `1 × 4` entries — see the [module docs](self).
const TAPS: usize = 4;

/// `A[3]` + `B[3][3]`: the pRatio-weighted moments the derivative contracts through.
const NSLOT: usize = 12;

/// The joint-histogram geometry, derived by the **host** from the fixed and moving
/// intensity ranges and handed over.
///
/// Derived on the host, not here, and that is the point: the host metric owns the
/// formula (`(trueMax − trueMin)/(bins − 2·padding)`, `trueMin/binSize − padding`), and
/// a second derivation on this side would be a second chance to disagree with it in the
/// last bits — after which every bin index in the run is suspect. The device is told the
/// numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MattesBins {
    pub bins: usize,
    /// Bins reserved at each axis end so the cubic window never needs a boundary
    /// condition (ITK's `padding`, 2).
    pub padding: usize,
    pub fixed_bin_size: f64,
    pub moving_bin_size: f64,
    pub fixed_normalized_min: f64,
    pub moving_normalized_min: f64,
    /// The moving intensity range. A sample interpolating outside it is dropped, as on
    /// the host.
    pub moving_true_min: f64,
    pub moving_true_max: f64,
}

/// The unnormalized joint histogram and fixed marginal, and the valid-sample count —
/// bit-identical to `MattesMutualInformationMetric::build_histogram`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointHistogram {
    /// Row-major `[fixed_bin * bins + moving_bin]`.
    pub joint_pdf: Vec<f64>,
    pub fixed_marginal: Vec<f64>,
    pub valid: usize,
}

/// The twelve pRatio-weighted moments of one derivative pass. Parameter-count- and
/// transform-independent; the host contracts them with the transform's own Jacobian.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DerivativeMoments {
    /// `Σ w · ∇M[d]`.
    pub a: [f64; DIM],
    /// `Σ w · ∇M[d] · x[e]`, indexed `[d][e]`.
    pub b: [[f64; DIM]; DIM],
}

const KERNEL_BODY: &str = r#"
#define TAPS 4
#define NSLOT 12

// itk::BSplineKernelFunction<3>::Evaluate, in the host's expression order with every
// operation rounded separately. The host is Rust, which does not fuse; NVRTC does, and
// this value is a histogram weight whose sum is pinned bit-identical to the host's.
__device__ __forceinline__ double cubic_bspline(double u) {
    const double a = fabs(u);
    if (a < 1.0) {
        const double sq = __dmul_rn(a, a);
        // (4 - 6*sq + 3*sq*a) / 6
        const double t = __dadd_rn(__dsub_rn(4.0, __dmul_rn(6.0, sq)),
                                   __dmul_rn(__dmul_rn(3.0, sq), a));
        return __ddiv_rn(t, 6.0);
    }
    if (a < 2.0) {
        const double sq = __dmul_rn(a, a);
        // (8 - 12*a + 6*sq - sq*a) / 6
        double t = __dsub_rn(8.0, __dmul_rn(12.0, a));
        t = __dadd_rn(t, __dmul_rn(6.0, sq));
        t = __dsub_rn(t, __dmul_rn(sq, a));
        return __ddiv_rn(t, 6.0);
    }
    return 0.0;
}

// itk::BSplineDerivativeKernelFunction<3>::Evaluate -- written in terms of the SIGNED u,
// with distinct branches per sign, exactly as the host writes it.
__device__ __forceinline__ double cubic_bspline_derivative(double u) {
    if (u >= 0.0 && u < 1.0) {
        // -2*u + 1.5*u*u  ==  (-2*u) + ((1.5*u)*u)
        return __dadd_rn(__dmul_rn(-2.0, u), __dmul_rn(__dmul_rn(1.5, u), u));
    }
    if (u > -1.0 && u < 0.0) {
        return __dsub_rn(__dmul_rn(-2.0, u), __dmul_rn(__dmul_rn(1.5, u), u));
    }
    if (u >= 1.0 && u < 2.0) {
        // -2 + 2*u - 0.5*u*u
        return __dsub_rn(__dadd_rn(-2.0, __dmul_rn(2.0, u)),
                         __dmul_rn(__dmul_rn(0.5, u), u));
    }
    if (u > -2.0 && u <= -1.0) {
        return __dadd_rn(__dadd_rn(2.0, __dmul_rn(2.0, u)),
                         __dmul_rn(__dmul_rn(0.5, u), u));
    }
    return 0.0;
}

// The sample's fractional bin coordinate. The host: `value / bin_size - normalized_min`.
// Double division is correctly rounded on the device, and the subtraction is pinned.
__device__ __forceinline__ double parzen_term(double value, double bin_size, double nmin) {
    return __dsub_rn(__ddiv_rn(value, bin_size), nmin);
}

// The Parzen bin index: TRUNCATE the term, then clamp to the interior so all four taps
// stay inside the histogram. `ComputeSingleFixedImageParzenWindowIndex`, and the same
// clamp `ProcessPoint` applies to the moving index.
//
// This truncation is why the interpolated value had to be pinned to the bit. The term is
// a continuous function of `mv`, and `(long long)term` is not.
__device__ __forceinline__ int parzen_index(double term, int bins, int padding) {
    long long index = (long long)term;
    const long long lo = (long long)padding;
    const long long hi = (long long)bins - (long long)padding - 1;
    if (index < lo) index = lo;
    else if (index > hi) index = hi;
    return (int)index;
}

// One entry per (sample, tap): the joint key/value list the counting sort consumes, plus
// the fixed marginal's per-sample key/value list.
//
// A dropped sample writes its taps into the DEAD bin with weight 0. See the module docs:
// this is the compaction, and it is free.
extern "C" __global__ void mattes_entries(
    const FSCALAR* __restrict__ fvals,
    const double* __restrict__ fpts,
    const int mode,
    const long long* __restrict__ fidx,
    const long long* __restrict__ fsize,
    const double* __restrict__ forigin,
    const double* __restrict__ fmat,
    const long long n,
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
    const int bins,
    const int padding,
    const double fixed_bin_size,
    const double moving_bin_size,
    const double fixed_normalized_min,
    const double moving_normalized_min,
    const double moving_true_min,
    const double moving_true_max,
    unsigned int* __restrict__ jkeys,    // TAPS * n
    double* __restrict__ jvals,          // TAPS * n
    unsigned int* __restrict__ mkeys,    // n
    double* __restrict__ mvals)          // n
{
    const unsigned int dead_joint = (unsigned int)bins * (unsigned int)bins;
    const unsigned int dead_marginal = (unsigned int)bins;

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        Sample sm;
        // Value only: the histogram does not need the moving gradient, and the second
        // pass is the one that pays for it.
        bool ok = take_sample<false>(s, fvals, fpts, mode, fidx, fsize, forigin, fmat,
                                     mbuf, msize, mstride, morigin, mmat,
                                     mmask, has_mask, fmask, has_fmask,
                                     stages, nstage, &sm);
        // The host's guard, on the host's terms: a value outside the histogram's moving
        // range has no well-defined bin.
        if (ok && (sm.value < moving_true_min || sm.value > moving_true_max)) ok = false;

        if (!ok) {
            mkeys[s] = dead_marginal;
            mvals[s] = 0.0;
            for (int j = 0; j < TAPS; ++j) {
                jkeys[s * TAPS + j] = dead_joint;
                jvals[s * TAPS + j] = 0.0;
            }
            continue;
        }

        const double moving_term = parzen_term(sm.value, moving_bin_size, moving_normalized_min);
        const int mi = parzen_index(moving_term, bins, padding);
        const int fi = parzen_index(parzen_term(sm.fval, fixed_bin_size, fixed_normalized_min),
                                    bins, padding);

        mkeys[s] = (unsigned int)fi;
        mvals[s] = 1.0;

        const int start = mi - 1;
        for (int j = 0; j < TAPS; ++j) {
            const int m = start + j;
            const double arg = __dsub_rn((double)m, moving_term);
            jkeys[s * TAPS + j] = (unsigned int)(fi * bins + m);
            jvals[s * TAPS + j] = cubic_bspline(arg);
        }
    }
}

// The second pass: the twelve pRatio-weighted moments. `pratio` is the finished table,
// pRatio * n_factor, exactly as the host's sparse-support path stores it.
extern "C" __global__ void mattes_deriv(
    const FSCALAR* __restrict__ fvals,
    const double* __restrict__ fpts,
    const int mode,
    const long long* __restrict__ fidx,
    const long long* __restrict__ fsize,
    const double* __restrict__ forigin,
    const double* __restrict__ fmat,
    const long long n,
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
    const int bins,
    const int padding,
    const double fixed_bin_size,
    const double moving_bin_size,
    const double fixed_normalized_min,
    const double moving_normalized_min,
    const double moving_true_min,
    const double moving_true_max,
    const double* __restrict__ pratio,   // bins * bins
    double* __restrict__ partials)       // GRID * NSLOT
{
    double acc[NSLOT];
    for (int k = 0; k < NSLOT; ++k) acc[k] = 0.0;

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        Sample sm;
        if (!take_sample<true>(s, fvals, fpts, mode, fidx, fsize, forigin, fmat,
                               mbuf, msize, mstride, morigin, mmat,
                               mmask, has_mask, fmask, has_fmask,
                               stages, nstage, &sm)) continue;
        if (sm.value < moving_true_min || sm.value > moving_true_max) continue;

        const double moving_term = parzen_term(sm.value, moving_bin_size, moving_normalized_min);
        const int mi = parzen_index(moving_term, bins, padding);
        const int fi = parzen_index(parzen_term(sm.fval, fixed_bin_size, fixed_normalized_min),
                                    bins, padding);

        const int start = mi - 1;
        for (int j = 0; j < TAPS; ++j) {
            const int m = start + j;
            const double arg = (double)m - moving_term;
            const double w = cubic_bspline_derivative(arg) * pratio[(long long)fi * bins + m];
            for (int d = 0; d < 3; ++d) {
                const double wg = w * sm.g[d];
                acc[d] += wg;
                for (int e = 0; e < 3; ++e) acc[3 + d*3 + e] += wg * sm.x[e];
            }
        }
    }

    emit_partials(acc, NSLOT, partials);
}

// min / max / count over a set, reduced with a shared tree and folded on the host.
//
// min and max are SELECTIONS, not sums: they are exact and order-independent, so this
// agrees with the host's sequential `min_max` on the bits without needing the host's
// order. The count is a sum of 1.0s, exact below 2^53.
__device__ __forceinline__ void emit_range(double mn, double mx, double cnt,
                                           double* __restrict__ partials)
{
    __shared__ double sh[BLOCK];
    const int tid = threadIdx.x;

    sh[tid] = mn;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmin(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x * 3 + 0] = sh[0];
    __syncthreads();

    sh[tid] = mx;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmax(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x * 3 + 1] = sh[0];
    __syncthreads();

    sh[tid] = cnt;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x * 3 + 2] = sh[0];
    __syncthreads();
}

// The FIXED SAMPLES' intensity range -- the sample set's, not the grid's. A masked-out
// or unsampled voxel is not in `FixedSamples` on the host, so it is not in the range
// that sizes the histogram's fixed axis, and it must not be in this one either.
extern "C" __global__ void fixed_range(
    const FSCALAR* __restrict__ fvals,
    const int mode,
    const long long* __restrict__ fidx,
    const unsigned char* __restrict__ fmask,
    const int has_fmask,
    const long long n,
    double* __restrict__ partials)
{
    // NVRTC compiles without <math.h>, so `INFINITY` is not in scope: build it from bits.
    const double inf = __longlong_as_double(0x7FF0000000000000LL);
    double mn = inf, mx = -inf, cnt = 0.0;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        const long long gv = (mode == MODE_INDICES) ? fidx[s] : s;
        if (has_fmask && !fmask[gv]) continue;
        const double v = (double)fvals[gv];
        mn = fmin(mn, v);
        mx = fmax(mx, v);
        cnt += 1.0;
    }
    emit_range(mn, mx, cnt, partials);
}

// The MOVING VOLUME's intensity range -- over the voxels THE MOVING MASK ADMITS, which
// is what the host's `MovingImage::value_range` reduces over.
//
// The mask is not optional here and it is not a filter on the answer -- it decides the
// question. This range sizes the histogram's moving axis, so a masked-out voxel brighter
// than anything the mask admits would stretch every bin and move every sample's Parzen
// index. The host reduced over the whole buffer and so did this kernel, together, which
// is exactly the failure a host-vs-device bit-identity pin cannot see: both were wrong in
// the same direction. Fixed on both sides at once (see the port's ledger 2.162); ITK masks
// it at itkMattesMutualInformationImageToImageMetricv4.hxx:199-214.
//
// The mask is in the buffer's own traversal order, so this indexes it with `i` directly --
// no continuous index, no rounding. Same as `fixed_range` one kernel up.
extern "C" __global__ void moving_range(
    const MSCALAR* __restrict__ mbuf,
    const unsigned char* __restrict__ mmask,
    const int has_mmask,
    const long long len,
    double* __restrict__ partials)
{
    // NVRTC compiles without <math.h>, so `INFINITY` is not in scope: build it from bits.
    const double inf = __longlong_as_double(0x7FF0000000000000LL);
    double mn = inf, mx = -inf, cnt = 0.0;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < len; i += stride) {
        if (has_mmask && !mmask[i]) continue;
        const double v = (double)mbuf[i];
        mn = fmin(mn, v);
        mx = fmax(mx, v);
        cnt += 1.0;
    }
    emit_range(mn, mx, cnt, partials);
}
"#;

/// The kernel source for a volume element type, compiled once per process (the backend
/// caches modules by source).
fn kernel_src(fixed: &str, moving: &str) -> String {
    format!("#define FSCALAR {fixed}\n#define MSCALAR {moving}\n{SAMPLER_SRC}\n{KERNEL_BODY}")
}

/// Mattes mutual information over volumes resident on the device.
///
/// Built once per pyramid level, evaluated once (or twice) per optimizer iteration. It
/// owns the counting sort's working set and the entry lists, so the iteration loop
/// allocates nothing: what crosses the bus per iteration is the point map up (96 bytes a
/// stage), the pRatio table up (`bins²` doubles), and the histogram plus twelve moments
/// down.
///
/// See the [module docs](self) for the fan-out, the dead bin, and precisely which half
/// of this is bit-identical to the host.
pub struct ResidentMattes {
    res: Resident,
    bins: usize,
    /// `TAPS * n` joint keys and values — the entry list the counting sort consumes.
    jkeys: DeviceBuffer<u32>,
    jvals: DeviceBuffer<f64>,
    /// `n` marginal keys and values (one box-window tap per sample).
    mkeys: DeviceBuffer<u32>,
    mvals: DeviceBuffer<f64>,
    joint: HistogramScratch,
    marginal: HistogramScratch,
    /// The pRatio table the derivative pass reads, re-uploaded per iteration.
    pratio: DeviceBuffer<f64>,
    partials: Partials,
    /// The fixed sample set's and the moving volume's intensity ranges, reduced once at
    /// construction — the numbers the host sizes the histogram from.
    fixed_range: (f64, f64),
    moving_range: (f64, f64),
}

impl ResidentMattes {
    /// Build from two device-resident images and a bin count.
    ///
    /// The bin count sizes the counting sort (`bins² + 1` joint bins), so it is fixed
    /// here; the *derived* geometry — bin sizes and normalized minima — is the host's and
    /// arrives per call as a [`MattesBins`].
    pub fn from_device_masked(
        fixed: &DeviceImage,
        fixed_points: FixedPoints<'_>,
        fixed_mask: Option<&DeviceMask>,
        moving: &DeviceImage,
        moving_geometry: &MovingGeometry<'_>,
        bins: usize,
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        if moving.len() != moving_geometry.len {
            return Err(CudaError::DegenerateInput);
        }
        if bins < 5 {
            return Err(CudaError::HistogramShape(format!(
                "{bins} Parzen bins; the cubic window needs 2·padding + 1 = 5"
            )));
        }
        let n = Resident::sample_count_of(fixed, &fixed_points);
        let res = Resident::build(
            n,
            Volumes::from_device(fixed, moving)?,
            fixed_points,
            fixed_mask,
            moving_geometry,
        )?;

        let mut me = Self {
            res,
            bins,
            jkeys: DeviceBuffer::zeros(backend, TAPS * n)?,
            jvals: DeviceBuffer::zeros(backend, TAPS * n)?,
            mkeys: DeviceBuffer::zeros(backend, n)?,
            mvals: DeviceBuffer::zeros(backend, n)?,
            // `+ 1` for the dead bin each list sends its dropped samples to.
            joint: HistogramScratch::new(backend, TAPS * n, bins * bins + 1)?,
            marginal: HistogramScratch::new(backend, n, bins + 1)?,
            pratio: DeviceBuffer::zeros(backend, bins * bins)?,
            partials: Partials::new(backend, NSLOT)?,
            fixed_range: (0.0, 0.0),
            moving_range: (0.0, 0.0),
        };
        me.fixed_range = me.reduce_fixed_range(backend)?;
        me.moving_range = me.reduce_moving_range(backend)?;
        Ok(me)
    }

    /// Number of fixed samples.
    pub fn sample_count(&self) -> usize {
        self.res.n
    }

    /// Device bytes held by the two volumes.
    pub fn volume_bytes(&self) -> usize {
        self.res.vols.bytes()
    }

    /// `(min, max)` of the **fixed sample set** and of the **moving volume** — the two
    /// ranges the host sizes the joint histogram's axes from.
    ///
    /// Exact, and equal to the host's on the bits: min and max are selections, so the
    /// device's tree and the host's sequential scan cannot disagree about them.
    pub fn value_ranges(&self) -> ((f64, f64), (f64, f64)) {
        (self.fixed_range, self.moving_range)
    }

    fn launch_config() -> LaunchConfig {
        LaunchConfig {
            grid_dim: (GRID, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Fold `GRID` `(min, max, count)` partials. `(0, 0)` when the set is empty, which is
    /// what the host's `min_max().unwrap_or((0.0, 0.0))` returns.
    fn fold_range(partials: &[f64]) -> (f64, f64) {
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut cnt = 0.0f64;
        for blk in 0..GRID as usize {
            mn = mn.min(partials[blk * 3]);
            mx = mx.max(partials[blk * 3 + 1]);
            cnt += partials[blk * 3 + 2];
        }
        if cnt == 0.0 { (0.0, 0.0) } else { (mn, mx) }
    }

    fn reduce_fixed_range(&mut self, backend: &Backend) -> Result<(f64, f64), CudaError> {
        let n_i = self.res.n as i64;
        let mut out = DeviceBuffer::<f64>::zeros(backend, GRID as usize * 3)?;
        let (fscalar, mscalar) = self.res.vols.scalars();
        let f = backend.function(&kernel_src(fscalar, mscalar), "fixed_range")?;

        macro_rules! launch_range {
            ($fvals:expr) => {{
                let mut launch = backend.stream().launch_builder(&f);
                launch
                    .arg($fvals.device())
                    .arg(&self.res.mode)
                    .arg(self.res.d_fidx.device())
                    .arg(self.res.d_fmask.device())
                    .arg(&self.res.has_fmask)
                    .arg(&n_i)
                    .arg(out.device_mut());
                // SAFETY: seven parameters, seven arguments. `gv` is a fixed-grid voxel in
                // every mode (`Resident::build` checked the index list and the value count),
                // `fidx` is read only in `MODE_INDICES` and `fmask` only when `has_fmask`,
                // and `partials` holds `GRID * 3`, which `emit_range` writes exactly.
                unsafe { launch.launch(Self::launch_config())? };
            }};
        }
        match &self.res.vols {
            Volumes::F64 { fvals, .. } => launch_range!(fvals),
            Volumes::Split { fvals, .. } => launch_range!(fvals),
        }
        backend.synchronize()?;
        Ok(Self::fold_range(&out.to_host(backend)?))
    }

    fn reduce_moving_range(&mut self, backend: &Backend) -> Result<(f64, f64), CudaError> {
        let (_, mlen) = self.res.vols.lens();
        let len_i = mlen as i64;
        let mut out = DeviceBuffer::<f64>::zeros(backend, GRID as usize * 3)?;
        let (fscalar, mscalar) = self.res.vols.scalars();
        let f = backend.function(&kernel_src(fscalar, mscalar), "moving_range")?;

        let has_mmask = self.res.has_mask;

        macro_rules! launch_range {
            ($mbuf:expr) => {{
                let mut launch = backend.stream().launch_builder(&f);
                launch
                    .arg($mbuf.device())
                    .arg(self.res.d_mmask.device())
                    .arg(&has_mmask)
                    .arg(&len_i)
                    .arg(out.device_mut());
                // SAFETY: five parameters, five arguments. The grid-stride loop is bounded
                // by `i < len`, which is `mbuf`'s length and — when `has_mmask` — `d_mmask`'s
                // (the mask is the moving buffer's own traversal order); `partials` holds
                // `GRID * 3`.
                unsafe { launch.launch(Self::launch_config())? };
            }};
        }
        match &self.res.vols {
            Volumes::F64 { mbuf, .. } => launch_range!(mbuf),
            Volumes::Split { mbuf, .. } => launch_range!(mbuf),
        }
        backend.synchronize()?;
        Ok(Self::fold_range(&out.to_host(backend)?))
    }

    /// Reject a geometry that does not describe the histogram this was built for.
    fn check_bins(&self, geom: &MattesBins) -> Result<(), CudaError> {
        if geom.bins != self.bins {
            return Err(CudaError::HistogramShape(format!(
                "geometry has {} bins; this metric was built for {}",
                geom.bins, self.bins
            )));
        }
        if geom.bins < 2 * geom.padding + 1 {
            return Err(CudaError::HistogramShape(format!(
                "{} bins cannot hold a {}-bin pad at each end",
                geom.bins, geom.padding
            )));
        }
        Ok(())
    }

    /// The unnormalized joint histogram, the fixed marginal and the valid-sample count
    /// at the point map `stages` — **bit-identical** to the host's `build_histogram`.
    pub fn joint_histogram(
        &mut self,
        stages: &[PointStage],
        geom: &MattesBins,
    ) -> Result<JointHistogram, CudaError> {
        self.check_bins(geom)?;
        let backend: &Backend = backend()?;
        self.emit_entries(backend, stages, geom)?;

        let joint_full = self
            .joint
            .run(backend, &self.jkeys, &self.jvals, BLOCK as usize)?;
        let marginal_full = self
            .marginal
            .run(backend, &self.mkeys, &self.mvals, BLOCK as usize)?;

        let bins = self.bins;
        // Drop the dead bin off each. Its sum is zero by construction (every entry in it
        // has value 0.0) and nothing reads it — it exists so the entry list can stay
        // index-aligned with the samples.
        let joint_pdf = joint_full[..bins * bins].to_vec();
        let fixed_marginal = marginal_full[..bins].to_vec();

        // Every valid sample added exactly 1.0 to its fixed bin, so the marginal sums to
        // the valid count -- exactly, in `f64`, for any count below 2^53. This is the
        // host's `valid` counter, recovered rather than reduced a second time.
        let valid = fixed_marginal.iter().sum::<f64>() as usize;

        Ok(JointHistogram {
            joint_pdf,
            fixed_marginal,
            valid,
        })
    }

    /// The twelve pRatio-weighted derivative moments at the same point map.
    ///
    /// `pratio` is the finished `bins × bins` table — `pRatio · n_factor`, exactly as the
    /// host's sparse-support path builds it — so this must be called *after*
    /// [`joint_histogram`](Self::joint_histogram) and the host's histogram walk.
    pub fn derivative_moments(
        &mut self,
        stages: &[PointStage],
        geom: &MattesBins,
        pratio: &[f64],
    ) -> Result<DerivativeMoments, CudaError> {
        self.check_bins(geom)?;
        if pratio.len() != self.bins * self.bins {
            return Err(CudaError::HistogramShape(format!(
                "pRatio table has {} entries; the histogram has {}",
                pratio.len(),
                self.bins * self.bins
            )));
        }
        let backend: &Backend = backend()?;
        self.res.upload_point_map(backend, stages)?;
        self.pratio.copy_from_host(backend, pratio)?;

        let Resident {
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
            d_stages,
            nstage,
        } = &mut self.res;

        let n_i = *n as i64;
        let (bins_i, padding_i) = (self.bins as i32, geom.padding as i32);
        let (fscalar, mscalar) = vols.scalars();
        let f = backend.function(&kernel_src(fscalar, mscalar), "mattes_deriv")?;
        let d_pratio = &self.pratio;
        let d_partials = self.partials.device_mut();

        macro_rules! launch_deriv {
            ($fvals:expr, $mbuf:expr) => {{
                let mut launch = backend.stream().launch_builder(&f);
                launch
                    .arg($fvals.device())
                    .arg(d_fpts.device())
                    .arg(&*mode)
                    .arg(d_fidx.device())
                    .arg(d_fsize.device())
                    .arg(d_forigin.device())
                    .arg(d_fmat.device())
                    .arg(&n_i)
                    .arg($mbuf.device())
                    .arg(d_msize.device())
                    .arg(d_mstride.device())
                    .arg(d_morigin.device())
                    .arg(d_mmat.device())
                    .arg(d_mmask.device())
                    .arg(&*has_mask)
                    .arg(d_fmask.device())
                    .arg(&*has_fmask)
                    .arg(d_stages.device())
                    .arg(&*nstage)
                    .arg(&bins_i)
                    .arg(&padding_i)
                    .arg(&geom.fixed_bin_size)
                    .arg(&geom.moving_bin_size)
                    .arg(&geom.fixed_normalized_min)
                    .arg(&geom.moving_normalized_min)
                    .arg(&geom.moving_true_min)
                    .arg(&geom.moving_true_max)
                    .arg(d_pratio.device())
                    .arg(d_partials.device_mut());
                // SAFETY: twenty-nine arguments for twenty-nine parameters, in order and in
                // type — the sampler's nineteen (identical to `ms_moments`, and bounded by
                // the same `Resident::build` checks), the eight geometry scalars, the pRatio
                // table, and the partials. `pratio` is indexed at `fi * bins + m`, where
                // `parzen_index` clamps `fi` and `mi` into `[padding, bins - padding - 1]`
                // and `m` runs over `mi - 1 ..= mi + 2`, so with `padding >= 1` (checked in
                // `check_bins`) the index stays inside `bins * bins`. `partials` holds
                // `GRID * NSLOT`, which `emit_partials` writes exactly.
                unsafe { launch.launch(Self::launch_config())? };
            }};
        }
        match vols {
            Volumes::F64 { fvals, mbuf } => launch_deriv!(fvals, mbuf),
            Volumes::Split { fvals, mbuf } => launch_deriv!(fvals, mbuf),
        }

        let p = self.partials.fold(backend)?;
        let mut m = DerivativeMoments::default();
        for d in 0..DIM {
            m.a[d] = p[d];
            for e in 0..DIM {
                m.b[d][e] = p[3 + d * DIM + e];
            }
        }
        Ok(m)
    }

    /// Fill the joint and marginal entry lists at `stages`.
    fn emit_entries(
        &mut self,
        backend: &Backend,
        stages: &[PointStage],
        geom: &MattesBins,
    ) -> Result<(), CudaError> {
        self.res.upload_point_map(backend, stages)?;

        let Resident {
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
            d_stages,
            nstage,
        } = &mut self.res;

        let n_i = *n as i64;
        let (bins_i, padding_i) = (self.bins as i32, geom.padding as i32);
        let (fscalar, mscalar) = vols.scalars();
        let f = backend.function(&kernel_src(fscalar, mscalar), "mattes_entries")?;
        let (jkeys, jvals) = (&mut self.jkeys, &mut self.jvals);
        let (mkeys, mvals) = (&mut self.mkeys, &mut self.mvals);

        macro_rules! launch_entries {
            ($fvals:expr, $mbuf:expr) => {{
                let mut launch = backend.stream().launch_builder(&f);
                launch
                    .arg($fvals.device())
                    .arg(d_fpts.device())
                    .arg(&*mode)
                    .arg(d_fidx.device())
                    .arg(d_fsize.device())
                    .arg(d_forigin.device())
                    .arg(d_fmat.device())
                    .arg(&n_i)
                    .arg($mbuf.device())
                    .arg(d_msize.device())
                    .arg(d_mstride.device())
                    .arg(d_morigin.device())
                    .arg(d_mmat.device())
                    .arg(d_mmask.device())
                    .arg(&*has_mask)
                    .arg(d_fmask.device())
                    .arg(&*has_fmask)
                    .arg(d_stages.device())
                    .arg(&*nstage)
                    .arg(&bins_i)
                    .arg(&padding_i)
                    .arg(&geom.fixed_bin_size)
                    .arg(&geom.moving_bin_size)
                    .arg(&geom.fixed_normalized_min)
                    .arg(&geom.moving_normalized_min)
                    .arg(&geom.moving_true_min)
                    .arg(&geom.moving_true_max)
                    .arg(jkeys.device_mut())
                    .arg(jvals.device_mut())
                    .arg(mkeys.device_mut())
                    .arg(mvals.device_mut());
                // SAFETY: thirty-one arguments for thirty-one parameters, in order and in
                // type. The sampler's nineteen are `ms_moments`' and are bounded by the same
                // `Resident::build` checks. Every sample writes exactly `TAPS` joint entries
                // at `s * TAPS + j` and one marginal entry at `s`, and the buffers hold
                // `TAPS * n` and `n`. Every key it can emit names a bin that exists: a valid
                // sample's is `fi * bins + m` with `fi, mi` clamped into
                // `[padding, bins - padding - 1]` and `m` in `mi - 1 ..= mi + 2`, so
                // `0 <= key < bins * bins`; a dropped sample's is the dead bin `bins * bins`,
                // and the histogram was sized at `bins * bins + 1`.
                unsafe { launch.launch(Self::launch_config())? };
            }};
        }
        match vols {
            Volumes::F64 { fvals, mbuf } => launch_entries!(fvals, mbuf),
            Volumes::Split { fvals, mbuf } => launch_entries!(fvals, mbuf),
        }
        backend.synchronize()?;
        Ok(())
    }
}
