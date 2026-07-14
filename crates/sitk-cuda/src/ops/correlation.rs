//! Device-resident normalized cross-correlation (NCC): two passes, 3 + 28 moments.
//!
//! The metric is `value = −sfm²/(sff·smm)` over the mean-subtracted samples —
//! `itk::CorrelationImageToImageMetricv4`, and here specifically
//! `sitk_registration::correlation::CorrelationMetric`, which is the reference this
//! is measured against.
//!
//! # Why two passes, and not the one-pass kernel that exists
//!
//! Every mean-subtracted accumulator is **affine in the means**, so the
//! mean-subtraction can be deferred and the whole metric reduced in **one** pass of
//! 42 raw moments:
//!
//! ```text
//! sff    = Σ fᵢ²        − N·f̄²
//! fdm[d] = Σ fᵢ·∇Mᵢ[d]  − f̄·Σ ∇Mᵢ[d]
//! ```
//!
//! That form is algebraically identical to this one and numerically worse, in a way
//! that depends on the caller's data rather than on the algorithm: each line is a
//! difference of comparable magnitudes, so `sff`'s relative error is
//! `ε·(1 + f̄²/var(f))`. Measured (`sitk_registration::correlation`'s
//! `one_pass_moment_form` tests) on a CT-like volume at mean 1000, that
//! amplification is **1941**, and through a real reduction the one-pass form loses
//! **7.8e-11** on the value where this two-pass form loses **1.5e-14** — a factor of
//! 5.3e3. At mean 10000 the amplification is 187863. No fixed tolerance can be
//! written for a form whose error the caller's intensity range picks, so the cheap
//! form is refused and this one pays ~2× the memory traffic instead.
//!
//! # The moments
//!
//! **Pass 1** (`ncc_sums`, 3 slots) is the host's `means()`: `Σf`, `Σm`, `count`
//! over the valid samples. The host divides. No gradient is computed — `WANT_GRAD`
//! is a template parameter of the shared sampler, so the gradient arithmetic is
//! compiled *out* of this pass rather than branched around.
//!
//! **Pass 2** (`ncc_moments`, 28 slots) takes `f̄`/`m̄` as scalars, so `f1ᵢ`/`m1ᵢ` are
//! as local inside the kernel as mean-squares' `diffᵢ` is:
//!
//! ```text
//! count, sff = Σ f1², smm = Σ m1², sfm = Σ f1·m1              (4)
//! F0[d] = Σ f1·∇M[d],   F1[d][e] = Σ f1·∇M[d]·x[e]           (3 + 9)
//! M0[d] = Σ m1·∇M[d],   M1[d][e] = Σ m1·∇M[d]·x[e]           (3 + 9)
//! ```
//!
//! # The derivative does not break the reduction
//!
//! ```text
//! ∂value/∂pₖ = −2·sfm/(sff·smm) · ( fdmₖ − (sfm/smm)·mdmₖ )
//! ```
//!
//! `sfm`, `sff` and `smm` are global — but they enter only as **coefficients that do
//! not depend on the sample**, so they factor straight out of the sum and land in
//! the host-side contraction, exactly where mean-squares already does its own. The
//! device never sees them. And since each Jacobian column is affine in the point,
//! `fdmₖ = Σ_d J(0)[d][k]·F0[d] + Σ_d Σ_e C_e[d][k]·F1[d][e]` — so the derivative is
//! **parameter-count-free** and one kernel serves the whole global-affine family,
//! just as it does for mean squares.
//!
//! That family is not a coincidence with this metric: `CorrelationMetric` is
//! **global-transform-only** by construction (its `check_transform` refuses local
//! support by name, mirroring ITK's constructor). The transforms the moment
//! factorization cannot express are exactly the ones the metric already refuses.
//!
//! # What this holds, and what it does not
//!
//! `count` is **exact** against the host — an integer over the same predicate on the
//! same samples. The moments are the same mathematical sums the host forms, in a
//! different order, so value and derivative diverge by **reduction rounding alone**
//! (~√N·ε), which is the same story mean squares tells and for the same reason. It
//! is bit-identical run to run (fixed grid, fixed tree, host fold in block order,
//! no atomics) and it is *not* bit-identical to the host's serial left-to-right sum.
//!
//! Two things it does not hold, stated here rather than discovered in a pin:
//!
//! - The value passes through zero (`sfm` is a signed sum), so a *relative* band on
//!   it is meaningless near decorrelation. NCC is normalized to `[−1, 0]`, so its
//!   pins band it **absolutely**.
//! - The degenerate test `sff·smm <= ε` is a **threshold on a reduced quantity**, and
//!   two paths whose products differ by √N·ε can straddle it. The device does not own
//!   that branch — it returns raw moments and the *host* applies the same test with
//!   the same constant, one implementation — but on a degenerate (constant) volume
//!   the two can still land on opposite sides. See the boundary pin.

use std::sync::OnceLock;

use cudarc::driver::PushKernelArg;

use crate::backend::{Backend, backend};
use crate::error::CudaError;
use crate::image::DeviceImage;
use crate::mask::DeviceMask;
use crate::ops::resident::{DIM, FixedPoints, MovingGeometry, Partials, Resident, SAMPLER_SRC};

/// `Σf`, `Σm`, `count`.
const NSLOT_SUMS: usize = 3;
/// `count` (1) + `sff`/`smm`/`sfm` (3) + `F0`/`M0` (3+3) + `F1`/`M1` (9+9).
const NSLOT_MOMENTS: usize = 28;

/// Both accumulators. The sampler that feeds them and the reduction that drains
/// them are [`SAMPLER_SRC`], shared with mean squares — one copy of the
/// host-parity chain, for every metric.
const KERNEL_BODY: &str = r#"
#define NSLOT_SUMS 3
#define NSLOT_MOMENTS 28

// Pass 1 == CorrelationMetric::means(). The sums that give the sample means, over
// exactly the samples pass 2 will accumulate: the same sampler, the same predicate,
// so the two passes cannot disagree about which samples are valid. No gradient --
// take_sample<false> compiles it out.
extern "C" __global__ void ncc_sums(
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
    const double* __restrict__ ab,
    double* __restrict__ partials)
{
    double acc[NSLOT_SUMS];
    for (int k = 0; k < NSLOT_SUMS; ++k) acc[k] = 0.0;

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        Sample sm;
        if (!take_sample<false>(s, fvals, fpts, mode, fidx, fsize, forigin, fmat,
                                mbuf, msize, mstride, morigin, mmat,
                                mmask, has_mask, fmask, has_fmask, ab, &sm)) continue;
        acc[0] += sm.fval;
        acc[1] += sm.value;
        acc[2] += 1.0;
    }

    emit_partials(acc, NSLOT_SUMS, partials);
}

// Pass 2 == the GetValueAndDerivative threader. `favg`/`mavg` arrive as scalars, so
// f1/m1 are per-sample local quantities here -- the global coupling was resolved
// before this launch, and what remains global (sfm/sff/smm) enters only as
// coefficients of the finished sums, on the host.
extern "C" __global__ void ncc_moments(
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
    const double* __restrict__ ab,
    const double favg,
    const double mavg,
    double* __restrict__ partials)
{
    double acc[NSLOT_MOMENTS];
    for (int k = 0; k < NSLOT_MOMENTS; ++k) acc[k] = 0.0;

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        Sample sm;
        if (!take_sample<true>(s, fvals, fpts, mode, fidx, fsize, forigin, fmat,
                               mbuf, msize, mstride, morigin, mmat,
                               mmask, has_mask, fmask, has_fmask, ab, &sm)) continue;

        // The host forms exactly these two, from the same means.
        const double f1 = sm.fval - favg;
        const double m1 = sm.value - mavg;

        acc[0] += 1.0;
        acc[1] += f1 * f1;   // sff
        acc[2] += m1 * m1;   // smm
        acc[3] += f1 * m1;   // sfm

        for (int d = 0; d < 3; ++d) {
            const double fg = f1 * sm.g[d];
            const double mg = m1 * sm.g[d];
            acc[4 + d]  += fg;                                   // F0[d]
            acc[16 + d] += mg;                                   // M0[d]
            for (int e = 0; e < 3; ++e) {
                acc[7  + d*3 + e] += fg * sm.x[e];               // F1[d][e]
                acc[19 + d*3 + e] += mg * sm.x[e];               // M1[d][e]
            }
        }
    }

    emit_partials(acc, NSLOT_MOMENTS, partials);
}
"#;

/// The kernel source for a volume element type, compiled once per process (the
/// backend caches modules by source). Both kernels live in one module, so one NVRTC
/// compile serves both passes.
fn kernel_src(fixed: &str, moving: &str) -> &'static str {
    static WIDE: OnceLock<String> = OnceLock::new();
    static SPLIT: OnceLock<String> = OnceLock::new();
    let cell = match fixed {
        "float" => &SPLIT,
        _ => &WIDE,
    };
    cell.get_or_init(|| {
        format!("#define FSCALAR {fixed}\n#define MSCALAR {moving}\n{SAMPLER_SRC}\n{KERNEL_BODY}")
    })
    .as_str()
}

/// One NCC evaluation's moments — everything the host needs to form the value and
/// the derivative, and nothing that mentions the transform's parameters.
///
/// `count` is the valid-sample count, and it is the **same** in both passes by
/// construction (one sampler, one predicate); [`ResidentCorrelation::evaluate`]
/// checks that rather than assuming it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CorrelationMoments {
    /// Samples that survived the masks and mapped inside the moving image.
    pub count: usize,
    /// The sample means, from pass 1. The host divided these, exactly as
    /// `CorrelationMetric::means` does.
    pub mean_fixed: f64,
    pub mean_moving: f64,
    /// `Σ f1²`.
    pub sff: f64,
    /// `Σ m1²`.
    pub smm: f64,
    /// `Σ f1·m1`.
    pub sfm: f64,
    /// `Σ f1 · ∇M[d]`.
    pub f0: [f64; DIM],
    /// `Σ f1 · ∇M[d] · x[e]`, indexed `[d][e]`.
    pub f1: [[f64; DIM]; DIM],
    /// `Σ m1 · ∇M[d]`.
    pub m0: [f64; DIM],
    /// `Σ m1 · ∇M[d] · x[e]`, indexed `[d][e]`.
    pub m1: [[f64; DIM]; DIM],
}

/// Fixed and moving volumes resident on the device, evaluated as NCC against any
/// number of transforms without re-uploading either.
///
/// Two launches per evaluation and two D2H folds (`512·3·8` = 12 KiB and
/// `512·28·8` = 114 KiB); 96 bytes go up. The sample set, the masks and the volumes
/// are [`Resident`] — the same ones mean squares uses, so a fixed mask, a sampled
/// index list and an explicit point list mean here exactly what they mean there,
/// including [`CudaError::MaskedExplicitPoints`].
pub struct ResidentCorrelation {
    res: Resident,
    sums: Partials,
    moments: Partials,
}

impl ResidentCorrelation {
    /// Build from volumes already on the device, with an optional **fixed mask**.
    ///
    /// The mask is gated by the sample's *grid voxel*, so it requires a sample set
    /// that knows its grid index ([`FixedPoints::Grid`] or [`FixedPoints::Indices`]);
    /// with [`FixedPoints::Explicit`] it is refused by name
    /// ([`CudaError::MaskedExplicitPoints`]) — the same invariant, enforced in the
    /// same place, because it is the same [`Resident`].
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
        let backend = backend()?;
        let n = Resident::sample_count_of(fixed, &fixed_points);
        Ok(Self {
            res: Resident::build(
                n,
                crate::ops::resident::Volumes::from_device(fixed, moving)?,
                fixed_points,
                fixed_mask,
                moving_geometry,
            )?,
            sums: Partials::new(backend, NSLOT_SUMS)?,
            moments: Partials::new(backend, NSLOT_MOMENTS)?,
        })
    }

    /// Number of fixed samples resident on the device.
    pub fn sample_count(&self) -> usize {
        self.res.n
    }

    /// Device bytes held by the two volumes.
    pub fn volume_bytes(&self) -> usize {
        self.res.vols.bytes()
    }

    /// Evaluate both passes for the point map `x ↦ A·x + b`.
    ///
    /// `a` is row-major `3 × 3`, `b` is length 3. Deterministic: the same inputs give
    /// bit-identical moments on every call and every run.
    ///
    /// If no sample is valid the moments are all zero with `count == 0`, and the host
    /// takes the degenerate branch — the same one `CorrelationMetric::evaluate` takes
    /// when `means()` returns `None`.
    pub fn evaluate(
        &mut self,
        a: &[f64; 9],
        b: &[f64; 3],
    ) -> Result<CorrelationMoments, CudaError> {
        let backend: &Backend = backend()?;
        self.res.upload_point_map(backend, a, b)?;

        // Pass 1: the sums that give the means.
        let s = self.launch(backend, Pass::Sums)?;
        let count = s[2] as usize;
        if count == 0 {
            return Ok(CorrelationMoments::default());
        }
        let mean_fixed = s[0] / count as f64;
        let mean_moving = s[1] / count as f64;

        // Pass 2: the mean-subtracted moments, with the means as kernel scalars.
        let p = self.launch(backend, Pass::Moments(mean_fixed, mean_moving))?;

        // The two passes ran the same sampler over the same samples under the same
        // point map, so they must agree about how many were valid. If they ever do
        // not, the means were divided by one population and the moments accumulated
        // over another — a silently wrong metric. Checked, not assumed.
        let count2 = p[0] as usize;
        if count2 != count {
            return Err(CudaError::PassCountMismatch {
                sums: count,
                moments: count2,
            });
        }

        let mut m = CorrelationMoments {
            count,
            mean_fixed,
            mean_moving,
            sff: p[1],
            smm: p[2],
            sfm: p[3],
            ..Default::default()
        };
        for d in 0..DIM {
            m.f0[d] = p[4 + d];
            m.m0[d] = p[16 + d];
            for e in 0..DIM {
                m.f1[d][e] = p[7 + d * DIM + e];
                m.m1[d][e] = p[19 + d * DIM + e];
            }
        }
        Ok(m)
    }

    /// One launch and its fold. The two passes take the same eighteen resident
    /// arguments in the same order and differ only in the two scalars pass 2 adds —
    /// so the argument list exists **once**, and a change to the sampler's inputs
    /// cannot reach one pass and miss the other.
    fn launch(&mut self, backend: &Backend, pass: Pass) -> Result<Vec<f64>, CudaError> {
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
            d_ab,
        } = &mut self.res;

        let n_i64 = *n as i64;
        let cfg = Resident::launch_config();
        let (fscalar, mscalar) = vols.scalars();
        let src = kernel_src(fscalar, mscalar);

        let (name, partials) = match pass {
            Pass::Sums => ("ncc_sums", &mut self.sums),
            Pass::Moments(..) => ("ncc_moments", &mut self.moments),
        };
        let f = backend.function(src, name)?;
        let d_partials = partials.device_mut();

        macro_rules! launch_pass {
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
                    .arg(d_ab.device());
                // Pass 2's two extra scalars, and only pass 2's.
                let means;
                if let Pass::Moments(favg, mavg) = pass {
                    means = (favg, mavg);
                    launch.arg(&means.0).arg(&means.1);
                }
                launch.arg(d_partials.device_mut());
                // SAFETY: the arguments match the launched kernel's parameters in
                // order and type — the eighteen resident ones for `ncc_sums`, and the
                // same eighteen plus `favg`/`mavg` for `ncc_moments`, which is the only
                // way `Pass` can select the name. `fvals` is `FSCALAR` and `mbuf` is
                // `MSCALAR` by the same `Volumes::scalars` match that chose the source.
                //
                // Every fixed-side read is indexed by the sampler's `gv`, a voxel of
                // the fixed grid in all three modes — `Resident::build` checked the
                // grid's product-of-size against `n` (`MODE_GRID`), the gathered value
                // count (`MODE_POINTS`), and *every* index against the voxel count
                // (`MODE_INDICES`). `d_fpts` (`n*3`) is read only in `MODE_POINTS` and
                // `d_fidx` (`n`) only in `MODE_INDICES`; each is a one-element dummy
                // otherwise, since a zero-length allocation is not a valid pointer.
                // `d_fsize`/`d_forigin`/`d_fmat` hold 3, 3, 9, as do the moving
                // geometry buffers (3, 3, 3, 9). `d_mmask` is read only when
                // `has_mask != 0` and the sampler bounds-checks the index it builds;
                // `d_fmask` only when `has_fmask != 0`, where `build` checked it covers
                // the fixed grid. `d_ab` holds 12. `d_partials` holds `GRID * nslot` for
                // this pass's `nslot`, and `emit_partials` writes
                // `blockIdx.x * nslot + k` for `blockIdx.x < GRID`, `k < nslot`. Shared
                // memory is declared statically at `BLOCK` doubles, matching `block_dim`.
                unsafe { launch.launch(cfg)? };
            }};
        }
        match vols {
            crate::ops::resident::Volumes::F64 { fvals, mbuf } => launch_pass!(fvals, mbuf),
            crate::ops::resident::Volumes::Split { fvals, mbuf } => launch_pass!(fvals, mbuf),
        }

        partials.fold(backend)
    }
}

/// Which pass to launch, and — for the second — the means the first produced.
#[derive(Clone, Copy)]
enum Pass {
    Sums,
    Moments(f64, f64),
}
