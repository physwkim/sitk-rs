//! The three ops a multi-resolution pyramid level is built from, on the device:
//! [`recursive_gaussian`], [`shrink`], [`resample_linear`].
//!
//! # Why three, and why they stay three
//!
//! A registration pyramid level is *smooth, then place on a coarser grid*. The
//! host does exactly that (`ImageRegistrationMethod::prepare_level`): it smooths
//! the fixed image with the recursive (IIR) Gaussian, **shrinks** the smoothed
//! image to obtain the coarse grid's geometry, and then **resamples** the smoothed
//! image onto that grid with linear interpolation.
//!
//! `shrink` and `resample` are not the same operation and must not be collapsed
//! into one. `shrink` *subsamples*: it picks input voxel `j·f + offset`, with
//! `offset` the center-preserving shift **rounded to an integer**. `resample`
//! *interpolates*: it evaluates the input at the output voxel's physical point,
//! and that point sits on the **unrounded** shift. The two agree only when the
//! factor is 1. Substituting the shrunk *values* for the resampled ones would
//! introduce a sub-voxel translation bias of up to half a voxel — a bias a
//! registration would then dutifully optimize against.
//!
//! # Bit-identity, not tolerance
//!
//! These ops exist to put a level's images where `execute` puts them, so they are
//! held to the strongest contract this crate has: **the same bits**. Each kernel
//! is a transcription of the CPU filter's arithmetic — same expressions, same
//! order, `double` throughout, narrowed to `f32` exactly once at the end — and it
//! is compiled with [`Backend::function_exact`](crate::Backend::function_exact),
//! which turns multiply-add contraction off so the device rounds where the host
//! rounds. `pyramid_parity.rs` asserts voxel-for-voxel equality against
//! `sitk_filters::recursive_gaussian` / `::shrink` / `ResampleImageFilter`.
//!
//! One divergence is structural rather than arithmetic, and it is worth stating:
//! a [`DeviceImage`] holds `f32`, so these ops smooth and resample in `f64` and
//! store `f32`. `sitk_filters::recursive_gaussian` narrows back to its *input's*
//! pixel type — for a `UInt16` CT it re-quantizes every level to `UInt16`. The
//! device path therefore reproduces `execute` run on the **`Float32` casts** of
//! the images, which is precisely what
//! [`DeviceImage::upload`](crate::DeviceImage::upload) put on the device.
//!
//! # The coefficients
//!
//! The Deriche/Farnebäck coefficient math is [`sitk_core::deriche`] — the one
//! implementation, shared with `sitk-filters`' host recursion rather than copied.
//! `sitk-filters` depends on *this* crate, so the edge cannot run the other way;
//! `sitk-core` is below both. `to_array()` hands the kernel the flat `[f64; 20]`
//! it names `N0 = c[0] … BM4 = c[19]` — that array is the seam a Rust struct
//! cannot cross. `the_device_recursive_gaussian_is_bit_identical_to_the_host_filter`
//! still pins the device output against the real filter, so the two paths cannot
//! drift apart silently.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use sitk_core::deriche::{Coefficients, GaussianOrder};

use crate::backend::{Backend, backend};
use crate::buffer::DeviceBuffer;
use crate::error::CudaError;
use crate::image::{DeviceImage, Geometry};

/// Threads per block for every kernel here.
const BLOCK: u32 = 256;

/// The pyramid ops are 3-D, as the metric kernel is.
const DIM: usize = 3;

/// Grid for `n` independent items.
fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize).max(1) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// First-index-fastest strides.
fn strides(size: &[usize]) -> Vec<i64> {
    let mut s = vec![1i64; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1] as i64;
    }
    s
}

fn require_3d(geom: &Geometry) -> Result<(), CudaError> {
    if geom.dimension() != DIM {
        return Err(CudaError::UnsupportedGeometry(format!(
            "{}-D image; the device pyramid ops are {DIM}-D",
            geom.dimension()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// recursive Gaussian (IIR)
// ---------------------------------------------------------------------------

/// One line of one axis per thread, through the fourth-order causal +
/// anti-causal recursion — a transcription of `sitk-filters`'
/// `filter_line`/`filter_axis`, operation for operation.
///
/// `out` and `scratch` are the causal and anti-causal accumulators. They are in
/// global memory because a line can be 256 voxels long and neither fits in
/// registers; a thread reads back only values it wrote itself, so no
/// synchronization is needed between them.
const GAUSS_AXIS: &str = r#"
#define N0 c[0]
#define N1 c[1]
#define N2 c[2]
#define N3 c[3]
#define D1 c[4]
#define D2 c[5]
#define D3 c[6]
#define D4 c[7]
#define M1 c[8]
#define M2 c[9]
#define M3 c[10]
#define M4 c[11]
#define BN1 c[12]
#define BN2 c[13]
#define BN3 c[14]
#define BN4 c[15]
#define BM1 c[16]
#define BM2 c[17]
#define BM3 c[18]
#define BM4 c[19]

extern "C" __global__ void gauss_axis(
    const double* __restrict__ in,
    double* __restrict__ out,
    double* __restrict__ scratch,
    const double* __restrict__ c,
    const long long n_lines,
    const long long ln,
    const long long stride)
{
    const long long line = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= n_lines) return;

    // A line along the axis is {base + k*stride : k < ln}; the lines are indexed
    // so that consecutive threads walk the fastest remaining axis.
    const long long base = (line / stride) * stride * ln + (line % stride);

#define D(k) (in[base + (k) * stride])
#define O(k) (out[base + (k) * stride])
#define S(k) (scratch[base + (k) * stride])

    // ---- causal (forward) ----
    const double v1 = D(0);

    O(0) = v1 * N0 + v1 * N1 + v1 * N2 + v1 * N3;
    O(1) = D(1) * N0 + v1 * N1 + v1 * N2 + v1 * N3;
    O(2) = D(2) * N0 + D(1) * N1 + v1 * N2 + v1 * N3;
    O(3) = D(3) * N0 + D(2) * N1 + D(1) * N2 + v1 * N3;

    O(0) -= v1 * BN1 + v1 * BN2 + v1 * BN3 + v1 * BN4;
    O(1) -= O(0) * D1 + v1 * BN2 + v1 * BN3 + v1 * BN4;
    O(2) -= O(1) * D1 + O(0) * D2 + v1 * BN3 + v1 * BN4;
    O(3) -= O(2) * D1 + O(1) * D2 + O(0) * D3 + v1 * BN4;

    for (long long i = 4; i < ln; ++i) {
        O(i) = D(i) * N0 + D(i - 1) * N1 + D(i - 2) * N2 + D(i - 3) * N3;
        O(i) -= O(i - 1) * D1 + O(i - 2) * D2 + O(i - 3) * D3 + O(i - 4) * D4;
    }

    // ---- anti-causal (backward) ----
    const double v2 = D(ln - 1);

    S(ln - 1) = v2 * M1 + v2 * M2 + v2 * M3 + v2 * M4;
    S(ln - 2) = D(ln - 1) * M1 + v2 * M2 + v2 * M3 + v2 * M4;
    S(ln - 3) = D(ln - 2) * M1 + D(ln - 1) * M2 + v2 * M3 + v2 * M4;
    S(ln - 4) = D(ln - 3) * M1 + D(ln - 2) * M2 + D(ln - 1) * M3 + v2 * M4;

    S(ln - 1) -= v2 * BM1 + v2 * BM2 + v2 * BM3 + v2 * BM4;
    S(ln - 2) -= S(ln - 1) * D1 + v2 * BM2 + v2 * BM3 + v2 * BM4;
    S(ln - 3) -= S(ln - 2) * D1 + S(ln - 1) * D2 + v2 * BM3 + v2 * BM4;
    S(ln - 4) -= S(ln - 3) * D1 + S(ln - 2) * D2 + S(ln - 1) * D3 + v2 * BM4;

    for (long long i = ln - 4; i > 0; --i) {
        S(i - 1) = D(i) * M1 + D(i + 1) * M2 + D(i + 2) * M3 + D(i + 3) * M4;
        S(i - 1) -= S(i) * D1 + S(i + 1) * D2 + S(i + 2) * D3 + S(i + 3) * D4;
    }

    // ---- roll the anti-causal part into the output ----
    for (long long k = 0; k < ln; ++k) {
        O(k) += S(k);
    }
}
"#;

/// `y[i] = (float)x[i]` — the single narrowing at the end of the `f64` chain,
/// matching the host's `Scalar::from_f64` (`v as f32`).
const NARROW: &str = r#"
extern "C" __global__ void narrow_f64_f32(
    const double* __restrict__ x,
    float* __restrict__ y,
    const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)x[i];
}
"#;

/// Gaussian-smooth a resident volume with the recursive (IIR) filter — the device
/// form of [`sitk_filters::recursive_gaussian`], bit for bit.
///
/// `sigma` is per dimension, in **physical** units (so the recursion runs at
/// `sigma[d] / spacing[d]` index units, as on the host). An axis with `sigma == 0`
/// is untouched; all-zero `sigma` copies the volume, as the host filter does.
///
/// Errors with [`CudaError::UnsupportedGeometry`] on a non-3-D image, a `sigma` of
/// the wrong length, a negative `sigma`, or a smoothed axis shorter than the four
/// voxels the fourth-order recursion needs — the same refusal the CPU filter makes.
///
/// [`sitk_filters::recursive_gaussian`]: https://docs.rs/sitk-filters
pub fn recursive_gaussian(src: &DeviceImage, sigma: &[f64]) -> Result<DeviceImage, CudaError> {
    let geom = src.geometry().clone();
    require_3d(&geom)?;
    if sigma.len() != DIM {
        return Err(CudaError::UnsupportedGeometry(format!(
            "sigma has {} entries, image is {DIM}-D",
            sigma.len()
        )));
    }
    if sigma.iter().any(|&s| s < 0.0) {
        return Err(CudaError::UnsupportedGeometry(format!(
            "negative sigma: {sigma:?}"
        )));
    }

    let backend: &Backend = backend()?;
    let n = geom.len();
    let size = geom.size.clone();
    let st = strides(&size);

    // The whole chain runs in `f64` and narrows once, exactly as the host filter
    // does: `to_f64_vec` → per-axis recursion → `image_from_f64`.
    let mut work = src.widen_f64()?;
    let mut out = DeviceBuffer::<f64>::zeros(backend, n)?;
    let mut scratch = DeviceBuffer::<f64>::zeros(backend, n)?;

    for d in 0..DIM {
        if sigma[d] <= 0.0 {
            continue;
        }
        if size[d] < 4 {
            return Err(CudaError::UnsupportedGeometry(format!(
                "axis {d} has {} voxels; the recursion needs at least 4",
                size[d]
            )));
        }
        // `ZeroOrder` ignores the physical `sigma` and `normalize_across_scale`
        // (ITK never rescales the zero order), so those two arguments are inert
        // here; the kernel names the flat array `N0 = c[0] … BM4 = c[19]`.
        let c = Coefficients::new(
            GaussianOrder::ZeroOrder,
            sigma[d] / geom.spacing[d],
            sigma[d],
            false,
        );
        let coeff = DeviceBuffer::from_host(backend, &c.to_array())?;

        let ln = size[d] as i64;
        let stride = st[d];
        let n_lines = (n / size[d]) as i64;

        let f = backend.function_exact(GAUSS_AXIS, "gauss_axis")?;
        let mut launch = backend.stream().launch_builder(&f);
        launch
            .arg(work.device())
            .arg(out.device_mut())
            .arg(scratch.device_mut())
            .arg(coeff.device())
            .arg(&n_lines)
            .arg(&ln)
            .arg(&stride);
        // SAFETY: seven parameters, seven arguments, matching in order and type.
        // `work`/`out`/`scratch` all hold `n` elements and every access is
        // `base + k*stride` with `base` derived from a line index `< n_lines` and
        // `0 <= k < ln`, so it stays in `[0, n)`; `coeff` holds the 20 doubles the
        // kernel's macros index. The recursion's `ln >= 4` precondition is checked
        // above.
        unsafe { launch.launch(cfg(n_lines as usize))? };
        backend.synchronize()?;

        std::mem::swap(&mut work, &mut out);
    }

    // Nothing was smoothed (every sigma zero): `work` is still the widened copy,
    // and narrowing it reproduces the input exactly.
    narrow_into_image(backend, &work, geom)
}

/// `(float)` the `f64` working buffer into a fresh [`DeviceImage`] carrying `geom`.
fn narrow_into_image(
    backend: &Backend,
    work: &DeviceBuffer<f64>,
    geom: Geometry,
) -> Result<DeviceImage, CudaError> {
    let n = geom.len();
    let mut img = DeviceImage::with_geometry(geom)?;

    let f = backend.function_exact(NARROW, "narrow_f64_f32")?;
    let n_i64 = n as i64;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(work.device())
        .arg(img.buffer_mut().device_mut())
        .arg(&n_i64);
    // SAFETY: three parameters, three arguments, matching in order and type; both
    // buffers hold `n` elements and the kernel guards every access on `i < n`.
    unsafe { launch.launch(cfg(n))? };
    backend.synchronize()?;
    Ok(img)
}

// ---------------------------------------------------------------------------
// shrink
// ---------------------------------------------------------------------------

const SHRINK: &str = r#"
extern "C" __global__ void shrink3(
    const float* __restrict__ in,
    float* __restrict__ out,
    const long long* __restrict__ g,   // in_size[3], out_size[3], factor[3], offset[3], in_stride[3], out_stride[3]
    const long long n_out)
{
    const long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    long long in_flat = 0;
    for (int d = 0; d < 3; ++d) {
        const long long oi = (o / g[15 + d]) % g[3 + d];
        long long ii = oi * g[6 + d] + g[9 + d];
        const long long last = g[d] - 1;
        if (ii > last) ii = last;
        in_flat += ii * g[12 + d];
    }
    out[o] = in[in_flat];
}
"#;

/// The output geometry and sampling offset of a shrink — `ShrinkImageFilter`'s
/// `GenerateOutputInformation`, transcribed from `sitk_filters::shrink`.
///
/// `δ` is the center-preserving shift in index units; the **origin** moves by the
/// unrounded `δ` while the **sampling offset** is `δ` rounded. That gap — up to
/// half a voxel — is exactly why a shrink is not a resample.
fn shrink_geometry(geom: &Geometry, factors: &[usize]) -> Result<(Geometry, Vec<i64>), CudaError> {
    let dim = geom.dimension();
    if factors.len() != dim {
        return Err(CudaError::UnsupportedGeometry(format!(
            "{} shrink factors, image is {dim}-D",
            factors.len()
        )));
    }
    if factors.contains(&0) {
        return Err(CudaError::UnsupportedGeometry(format!(
            "shrink factor of zero: {factors:?}"
        )));
    }

    let mut out_size = vec![0usize; dim];
    let mut out_spacing = vec![0.0f64; dim];
    let mut delta = vec![0.0f64; dim];
    let mut offset = vec![0i64; dim];
    for d in 0..dim {
        let f = factors[d];
        out_size[d] = (geom.size[d] / f).max(1);
        out_spacing[d] = geom.spacing[d] * f as f64;
        delta[d] = (geom.size[d] as f64 - 1.0) / 2.0 - f as f64 * (out_size[d] as f64 - 1.0) / 2.0;
        offset[d] = (delta[d] + 0.5).floor().max(0.0) as i64;
    }

    let mut out_origin = geom.origin.clone();
    for (i, o) in out_origin.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &dj) in delta.iter().enumerate() {
            acc += geom.direction[i * dim + j] * geom.spacing[j] * dj;
        }
        *o += acc;
    }

    Ok((
        Geometry {
            size: out_size,
            spacing: out_spacing,
            origin: out_origin,
            direction: geom.direction.clone(),
        },
        offset,
    ))
}

/// Subsample a resident volume by an integer factor per axis — the device form of
/// `sitk_filters::shrink`, bit for bit (it is a gather: no arithmetic touches a
/// voxel value).
///
/// **This is not a resample.** It picks input voxel `j·f + offset`, where `offset`
/// is the center-preserving shift *rounded*; the output origin carries the
/// *unrounded* shift. See the [module docs](self).
///
/// Errors with [`CudaError::UnsupportedGeometry`] on a non-3-D image, the wrong
/// number of factors, or a factor of zero.
pub fn shrink(src: &DeviceImage, factors: &[usize]) -> Result<DeviceImage, CudaError> {
    let geom = src.geometry();
    require_3d(geom)?;
    let (out_geom, offset) = shrink_geometry(geom, factors)?;

    let backend: &Backend = backend()?;
    let n_out = out_geom.len();
    let in_st = strides(&geom.size);
    let out_st = strides(&out_geom.size);

    let mut packed: Vec<i64> = Vec::with_capacity(18);
    packed.extend(geom.size.iter().map(|&s| s as i64));
    packed.extend(out_geom.size.iter().map(|&s| s as i64));
    packed.extend(factors.iter().map(|&f| f as i64));
    packed.extend(offset.iter().copied());
    packed.extend(in_st.iter().copied());
    packed.extend(out_st.iter().copied());
    let g = DeviceBuffer::from_host(backend, &packed)?;

    let mut dst = DeviceImage::with_geometry(out_geom)?;
    let n_i64 = n_out as i64;
    let f = backend.function_exact(SHRINK, "shrink3")?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(src.buffer().device())
        .arg(dst.buffer_mut().device_mut())
        .arg(g.device())
        .arg(&n_i64);
    // SAFETY: four parameters, four arguments, matching in order and type. `g`
    // holds the 18 packed `i64`s the kernel indexes (six groups of three); the
    // gathered input index is clamped to `in_size[d] - 1` per axis, so `in_flat`
    // stays inside the source, and `o < n_out` bounds the store.
    unsafe { launch.launch(cfg(n_out))? };
    backend.synchronize()?;
    Ok(dst)
}

// ---------------------------------------------------------------------------
// resample (linear, identity transform)
// ---------------------------------------------------------------------------

/// `ResampleImageFilter` with `Interpolator::Linear` and the identity transform,
/// transcribed operation for operation: `mat_vec`'s left-to-right accumulation,
/// `is_inside`'s half-open `[-0.5, size-0.5)` test, and `linear_at`'s corner order,
/// cumulative weight product, clamped corner index and `weight != 0.0` guard.
///
/// `p` packs, in order: `out_origin[3]`, `out_index_to_phys[9]`, `in_origin[3]`,
/// `in_phys_to_index[9]`, `default_value` — 25 doubles.
const RESAMPLE: &str = r#"
// The continuous index of output voxel `o` in the input's index space, and whether
// it is inside the input buffer. Shared by both kernels below **on purpose**: the
// nearest-neighbour resample must land on the same continuous index as the linear
// one, and two transcriptions of this arithmetic would be two things to keep in
// step. `resample_linear3`'s bit-identity test against the CPU filter is what
// guards this function.
__device__ __forceinline__ bool cindex_of(
    const long long o,
    const double* __restrict__ p,
    const long long* __restrict__ g,
    double* cindex)
{
    double index_f[3];
    for (int d = 0; d < 3; ++d) {
        index_f[d] = (double)((o / g[9 + d]) % g[3 + d]);
    }

    // phys = out_origin + M_out * index
    double phys[3];
    for (int r = 0; r < 3; ++r) {
        double acc = 0.0;
        for (int c = 0; c < 3; ++c) acc += p[3 + r * 3 + c] * index_f[c];
        phys[r] = p[r] + acc;
    }

    // cindex = M_in * (phys - in_origin)   [the transform is the identity]
    double diff[3];
    for (int d = 0; d < 3; ++d) diff[d] = phys[d] - p[12 + d];
    for (int r = 0; r < 3; ++r) {
        double acc = 0.0;
        for (int c = 0; c < 3; ++c) acc += p[15 + r * 3 + c] * diff[c];
        cindex[r] = acc;
    }

    // is_inside: pixel-centred coverage [-0.5, size - 0.5) on every axis.
    bool inside = true;
    for (int d = 0; d < 3; ++d) {
        if (!(cindex[d] >= -0.5 && cindex[d] < (double)g[d] - 0.5)) inside = false;
    }
    return inside;
}

extern "C" __global__ void resample_linear3(
    const float* __restrict__ in,
    float* __restrict__ out,
    const double* __restrict__ p,
    const long long* __restrict__ g,   // in_size[3], out_size[3], in_stride[3], out_stride[3]
    const long long n_out)
{
    const long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    double cindex[3];
    const bool inside = cindex_of(o, p, g, cindex);

    double v = p[24];
    if (inside) {
        long long base[3];
        double frac[3];
        for (int d = 0; d < 3; ++d) {
            const double f = floor(cindex[d]);
            base[d] = (long long)f;
            frac[d] = cindex[d] - f;
        }
        double acc = 0.0;
        for (int corner = 0; corner < 8; ++corner) {
            double weight = 1.0;
            long long offset = 0;
            for (int d = 0; d < 3; ++d) {
                const int bit = (corner >> d) & 1;
                weight *= (bit == 1) ? frac[d] : (1.0 - frac[d]);
                long long idx = base[d] + (long long)bit;
                if (idx < 0) idx = 0;
                if (idx > g[d] - 1) idx = g[d] - 1;
                offset += idx * g[6 + d];
            }
            if (weight != 0.0) acc += weight * (double)in[offset];
        }
        v = acc;
    }
    out[o] = (float)v;
}

extern "C" __global__ void resample_nearest3(
    const float* __restrict__ in,
    float* __restrict__ out,
    const double* __restrict__ p,
    const long long* __restrict__ g,
    const long long n_out)
{
    const long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    double cindex[3];
    const bool inside = cindex_of(o, p, g, cindex);

    double v = p[24];
    if (inside) {
        // `nearest_at`: round HALF UP (ITK's RoundHalfIntegerUp), then clamp.
        //
        // It must be floor(c + 0.5) and not a library rounding call. `rint`/
        // `nearbyint` are round-half-to-EVEN, which disagrees at *every* positive
        // half-integer (rint(0.5) = 0, floor(0.5 + 0.5) = 1) -- and a tie is not
        // exotic here: a grid offset by exactly half a voxel puts every sample on
        // one. `round()` is half-away-from-zero, which agrees for c >= 0 and differs
        // only at c = -0.5, where the clamp below hides it. So only floor(c + 0.5)
        // reproduces the host on all three.
        //
        // The clamp is the CPU's, and with `inside` already true it can only bite at
        // c in [-0.5, 0), which maps to index 0.
        long long offset = 0;
        for (int d = 0; d < 3; ++d) {
            long long i = (long long)floor(cindex[d] + 0.5);
            if (i < 0) i = 0;
            if (i > g[d] - 1) i = g[d] - 1;
            offset += i * g[6 + d];
        }
        v = (double)in[offset];
    }
    out[o] = (float)v;
}
"#;

/// `index_to_physical` (`D · diag(spacing)`) and `physical_to_index`
/// (`diag(1/spacing) · D⁻¹`), row-major — the two affines `ResampleImageFilter`
/// precomputes. `None` if the direction matrix is singular.
fn affines(geom: &Geometry) -> Option<(Vec<f64>, Vec<f64>)> {
    let dim = geom.dimension();
    let mut fwd = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            fwd[r * dim + c] = geom.direction[r * dim + c] * geom.spacing[c];
        }
    }
    // The same `invert` `ResampleImageFilter` calls — `sitk_core::matrix` is a
    // dependency of this crate, so this one is shared rather than duplicated, and
    // the inverse is bit-identical by construction.
    let inv = sitk_core::matrix::invert(&geom.direction, dim)?;
    let mut back = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            back[r * dim + c] = inv[r * dim + c] / geom.spacing[r];
        }
    }
    Some((fwd, back))
}

/// Resample a resident volume onto `grid` with **linear** interpolation through
/// the identity transform, sampling outside the buffer as `default_value` — the
/// device form of `ResampleImageFilter::execute(input, identity)` with `grid` as
/// the reference image.
///
/// **This is not a shrink.** Every output voxel is evaluated at its own physical
/// point, which for a coarse grid falls *between* input voxels. See the
/// [module docs](self).
///
/// Only the identity transform is offered, because that is the mapping the fixed
/// image takes onto the virtual grid when no fixed-initial transform is
/// configured — the only configuration
/// [`execute_on_device`](https://docs.rs/sitk-registration) accepts. A resample
/// *through* a transform is the moving image's job, and the metric kernel already
/// does it per sample without materializing a volume.
///
/// Errors with [`CudaError::UnsupportedGeometry`] on a non-3-D image or a singular
/// direction matrix.
pub fn resample_linear(
    src: &DeviceImage,
    grid: &Geometry,
    default_value: f64,
) -> Result<DeviceImage, CudaError> {
    resample(src, grid, default_value, "resample_linear3")
}

/// Resample a resident volume onto `grid` with **nearest-neighbour** interpolation
/// through the identity transform — the device form of
/// `ResampleImageFilter::execute(input, identity)` with
/// `Interpolator::NearestNeighbor` and `grid` as the reference image.
///
/// This exists for **masks**. A mask is a binary predicate over physical space, and
/// carrying it to a coarse level means re-sampling it without blurring and
/// re-thresholding — which is why the host uses nearest-neighbour for exactly this
/// and linear for the image (`prepare_level`). The values are 0/1, so an arithmetic
/// error here is invisible in the *values*: the failure mode is entirely in the
/// index arithmetic, where a half-voxel tie broken the wrong way flips a shell of
/// boundary voxels, changes the metric's valid-sample count, and shows up as a
/// count mismatch rather than as a wrong number. Hence
/// `the_device_nearest_resample_is_bit_identical_to_the_host_filter`, which pins the
/// op before anything is wired to it.
///
/// Rounding is **half-up** (`floor(c + 0.5)`, ITK's `RoundHalfIntegerUp`), not
/// `round()` — the two disagree at exact half-integers below zero.
///
/// Errors with [`CudaError::UnsupportedGeometry`] on a non-3-D image or a singular
/// direction matrix.
pub fn resample_nearest(
    src: &DeviceImage,
    grid: &Geometry,
    default_value: f64,
) -> Result<DeviceImage, CudaError> {
    resample(src, grid, default_value, "resample_nearest3")
}

/// The body both resamples share: the geometry checks, the 25-double parameter
/// pack, the 12-`i64` size/stride pack, and the launch. Only the kernel name
/// differs, so the two interpolators cannot drift apart in the arithmetic that
/// decides *where* they sample — only in what they do once they are there.
fn resample(
    src: &DeviceImage,
    grid: &Geometry,
    default_value: f64,
    kernel: &str,
) -> Result<DeviceImage, CudaError> {
    let in_geom = src.geometry();
    require_3d(in_geom)?;
    require_3d(grid)?;

    let (out_fwd, _) = affines(grid).ok_or_else(|| {
        CudaError::UnsupportedGeometry("output direction matrix is singular".into())
    })?;
    let (_, in_back) = affines(in_geom).ok_or_else(|| {
        CudaError::UnsupportedGeometry("input direction matrix is singular".into())
    })?;

    let backend: &Backend = backend()?;
    let n_out = grid.len();

    let mut p: Vec<f64> = Vec::with_capacity(25);
    p.extend(grid.origin.iter().copied());
    p.extend(out_fwd.iter().copied());
    p.extend(in_geom.origin.iter().copied());
    p.extend(in_back.iter().copied());
    p.push(default_value);
    let pb = DeviceBuffer::from_host(backend, &p)?;

    let mut packed: Vec<i64> = Vec::with_capacity(12);
    packed.extend(in_geom.size.iter().map(|&s| s as i64));
    packed.extend(grid.size.iter().map(|&s| s as i64));
    packed.extend(strides(&in_geom.size));
    packed.extend(strides(&grid.size));
    let g = DeviceBuffer::from_host(backend, &packed)?;

    let mut dst = DeviceImage::with_geometry(grid.clone())?;
    let n_i64 = n_out as i64;
    let f = backend.function_exact(RESAMPLE, kernel)?;
    let mut launch = backend.stream().launch_builder(&f);
    launch
        .arg(src.buffer().device())
        .arg(dst.buffer_mut().device_mut())
        .arg(pb.device())
        .arg(g.device())
        .arg(&n_i64);
    // SAFETY: five parameters, five arguments, matching in order and type. `p`
    // holds the 25 doubles and `g` the 12 `i64`s the kernel indexes; every input
    // read is at a corner index clamped to `[0, in_size[d] - 1]` per axis, and the
    // store is guarded by `o < n_out`.
    unsafe { launch.launch(cfg(n_out))? };
    backend.synchronize()?;
    Ok(dst)
}
