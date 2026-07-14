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
//! # What this module is, and what it is not
//!
//! It is an **accumulator**: fourteen slots and the loop that fills them. The
//! sampler that feeds it — point derivation, transform, validity, trilinear value
//! and gradient — and the reduction that drains it both live in
//! [`resident`](crate::ops::resident), shared with the correlation metric, because
//! that chain is the host-parity contract and a second copy of it is a second
//! chance to drift from the host silently. See that module for why.
//!
//! # Determinism
//!
//! No `atomicAdd` anywhere: a fixed grid, a fixed shared-memory tree, and a host
//! fold in block-index order. Bit-identical run to run — asserted by a test, not
//! assumed. Not bit-identical to the CPU's sequential sum, and it cannot be; the
//! divergence is reduction-rounding only (~√N·ε).

use std::sync::OnceLock;

use cudarc::driver::PushKernelArg;

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;
use crate::image::DeviceImage;
use crate::mask::DeviceMask;
use crate::ops::resident::{Partials, Resident, SAMPLER_SRC, Volumes};

pub use crate::ops::resident::{DIM, FixedPoints, MovingGeometry};

/// `sq` (1) + `S0[3]` (3) + `S1[3][3]` (9) + `count` (1).
const NSLOT: usize = 14;

/// The accumulator. Everything it needs to *reach* a sample — and everything it
/// does with the fourteen numbers afterwards — comes from
/// [`SAMPLER_SRC`](crate::ops::resident::SAMPLER_SRC), which is prepended to this.
const KERNEL_BODY: &str = r#"
#define NSLOT 14

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
        Sample sm;
        if (!take_sample<true>(s, fvals, fpts, mode, fidx, fsize, forigin, fmat,
                               mbuf, msize, mstride, morigin, mmat,
                               mmask, has_mask, fmask, has_fmask, ab, &sm)) continue;

        const double diff = sm.value - sm.fval;
        acc[0] += diff * diff;
        for (int d = 0; d < 3; ++d) {
            const double dg = diff * sm.g[d];
            acc[1 + d] += dg;
            for (int e = 0; e < 3; ++e) acc[4 + d*3 + e] += dg * sm.x[e];
        }
        acc[13] += 1.0;
    }

    emit_partials(acc, NSLOT, partials);
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
    cell.get_or_init(|| {
        format!("#define FSCALAR {fixed}\n#define MSCALAR {moving}\n{SAMPLER_SRC}\n{KERNEL_BODY}")
    })
    .as_str()
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

/// Fixed and moving volumes resident on the device, evaluable against any number
/// of transforms without re-uploading either.
///
/// Built once per pyramid level; [`evaluate`](Self::evaluate) is then called once
/// (or twice) per optimizer iteration and moves **96 bytes up** (the point map)
/// and **`GRID · 14 · 8` = 57 KiB down** (the per-block partials). Nothing else
/// crosses the bus, and nothing is reallocated per iteration.
pub struct ResidentMetric {
    res: Resident,
    partials: Partials,
}

impl ResidentMetric {
    /// Upload the fixed samples and the moving volume. This is the *only* large
    /// transfer in a registration run.
    ///
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
        Self::assemble(
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
    /// so it requires a sample set that **knows** its grid index — [`FixedPoints::Grid`]
    /// or [`FixedPoints::Indices`]. Combining it with [`FixedPoints::Explicit`] is
    /// refused by name ([`CudaError::MaskedExplicitPoints`]).
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
        let n = Resident::sample_count_of(fixed, &fixed_points);
        Self::assemble(
            n,
            Volumes::from_device(fixed, moving)?,
            fixed_points,
            fixed_mask,
            moving_geometry,
        )
    }

    fn assemble(
        n: usize,
        vols: Volumes,
        fixed_points: FixedPoints<'_>,
        fixed_mask: Option<&DeviceMask>,
        moving: &MovingGeometry<'_>,
    ) -> Result<Self, CudaError> {
        let backend = backend()?;
        Ok(Self {
            res: Resident::build(n, vols, fixed_points, fixed_mask, moving)?,
            partials: Partials::new(backend, NSLOT)?,
        })
    }

    /// Number of fixed samples resident on the device.
    pub fn sample_count(&self) -> usize {
        self.res.n
    }

    /// Device bytes held by the two volumes — what the precision choice above is
    /// spending.
    pub fn volume_bytes(&self) -> usize {
        self.res.vols.bytes()
    }

    /// Evaluate the moments for the point map `x ↦ A·x + b`.
    ///
    /// `a` is row-major `3 × 3`, `b` is length 3. Deterministic: the same inputs
    /// give bit-identical moments on every call and every run.
    pub fn evaluate(&mut self, a: &[f64; 9], b: &[f64; 3]) -> Result<Moments, CudaError> {
        let backend: &Backend = backend()?;
        self.res.upload_point_map(backend, a, b)?;

        // Field-by-field, so the volumes can be matched on while the partials
        // buffer is borrowed mutably for the same launch.
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
        let f = backend.function(kernel_src(fscalar, mscalar), "ms_moments")?;
        let d_partials = self.partials.device_mut();

        // The two instantiations differ in the element type of `fvals`/`mbuf` and
        // in nothing else — same nineteen arguments, same order, same grid.
        macro_rules! launch_moments {
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
                    .arg(d_ab.device())
                    .arg(d_partials.device_mut());
                // SAFETY: the nineteen arguments match the kernel's nineteen
                // parameters in order and type — `fvals` is `FSCALAR` and `mbuf` is
                // `MSCALAR`, which are the element types of the buffers this arm
                // matched (`Volumes::scalars`, the same match).
                //
                // Every read the sampler makes of the fixed side is indexed by `gv`,
                // and `gv` is a voxel of the fixed grid in every mode: it is `s < n`
                // in `MODE_GRID` (where `Resident::build` checked the grid's
                // product-of-size equals `n`) and `MODE_POINTS` (where `fvals` holds
                // `n` gathered values), and it is `fidx[s]` in `MODE_INDICES`, where
                // `build` has checked *every* index against the grid's voxel count and
                // `fvals` holds that many. So `fvals[gv]` and `fmask[gv]` are in bounds
                // in all three, and the point derived from `gv` stays in the grid.
                //
                // `d_fpts` holds `n*3` and is read only in `MODE_POINTS`; `d_fidx`
                // holds `n` and is read only in `MODE_INDICES`; each is a one-element
                // dummy otherwise, and neither is a valid allocation at length zero,
                // which is why the dummy exists. `d_fsize`/`d_forigin`/`d_fmat` hold
                // 3, 3 and 9. The moving geometry buffers hold exactly 3, 3, 3 and 9
                // elements as the kernel indexes them; `d_mmask` is read only when
                // `has_mask != 0`, in which case it has one byte per moving voxel and
                // the sampler bounds-checks the index it builds. `d_fmask` is read only
                // when `has_fmask != 0`, in which case `build` has checked it covers
                // the fixed grid. `d_ab` holds 12; `d_partials` holds `GRID * NSLOT`,
                // and `emit_partials` writes `blockIdx.x * NSLOT + k` for
                // `blockIdx.x < GRID`, `k < NSLOT`. Shared memory is declared
                // statically at `BLOCK` doubles, matching `block_dim`.
                unsafe { launch.launch(cfg)? };
            }};
        }
        match vols {
            Volumes::F64 { fvals, mbuf } => launch_moments!(fvals, mbuf),
            Volumes::Split { fvals, mbuf } => launch_moments!(fvals, mbuf),
        }

        let p = self.partials.fold(backend)?;

        let mut m = Moments {
            sq: p[0],
            count: p[13] as usize,
            ..Default::default()
        };
        for d in 0..DIM {
            m.s0[d] = p[1 + d];
            for e in 0..DIM {
                m.s1[d][e] = p[4 + d * DIM + e];
            }
        }
        Ok(m)
    }
}
