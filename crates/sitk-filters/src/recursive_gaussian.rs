//! Bit-exact Gaussian smoothing and its 1st/2nd derivatives by the
//! Deriche/Farnebäck recursive IIR filter, porting
//! `itk::RecursiveGaussianImageFilter` (all three
//! `RecursiveGaussianImageFilterEnums::GaussianOrder` values — `ZeroOrder`,
//! `FirstOrder`, `SecondOrder`, see [`GaussianOrder`]) and the
//! forward+backward recursion of `itk::RecursiveSeparableImageFilter`.
//!
//! Unlike the FIR [`smooth_gaussian`](crate::smooth_gaussian) — which samples
//! a truncated Gaussian kernel — this approximates the continuous Gaussian
//! (or its derivative) by a fourth-order recursive filter whose cost is
//! independent of `sigma`, and it reproduces ITK's arithmetic
//! operation-for-operation.
//!
//! **The coefficients are not here.** They are
//! [`sitk_core::deriche::Coefficients`] — ITK's `SetUp` / `ComputeDCoefficients`
//! / `ComputeNCoefficients` / `ComputeRemainingCoefficients`, over the improved
//! Deriche set of Farnebäck & Westin (J. Math. Imaging Vis. 2006, Table 3). They
//! sit in `sitk-core` rather than in this module because `sitk-cuda`'s device
//! pyramid runs the same recursion on the GPU and needs the same twenty numbers,
//! and `sitk-filters` depends on `sitk-cuda` — so `sitk-core` is the only crate
//! both can reach. See that module's docs for the normalization and symmetry
//! rules each [`GaussianOrder`] follows.
//!
//! What *is* here is the recursion the coefficients drive
//! (`RecursiveSeparableImageFilter::FilterDataArray`, ported as the private
//! [`filter_line`]/[`filter_axis`]). It does not depend on the order at all —
//! only the coefficients do — so all three orders share this one recursion.
//!
//! `NormalizeAcrossScale` (Lindeberg scale-space normalization) multiplies
//! `FirstOrder`/`SecondOrder` output by an extra `sigma^order` so a feature's
//! peak derivative response does not depend on the scale it is measured at;
//! it is off by default, matching ITK, and produces a plain (unnormalized)
//! derivative when off.
//!
//! `sigma` is per dimension in **physical units** (matching ITK's
//! `SmoothingRecursiveGaussianImageFilter` default): along axis `d` the
//! Gaussian standard deviation is `sigma[d]`, so in index units it is
//! `sigmad = sigma[d] / spacing[d]`. Axes are filtered in sequence
//! (separable); an axis with `sigma == 0` is left untouched. The boundary
//! replicates the edge value (the border value "extends to infinity"),
//! matching ITK, so a constant image and the DC component are preserved
//! exactly under `ZeroOrder`.
//!
//! The recursion needs at least four pixels along each filtered axis (ITK's
//! `RecursiveSeparableImageFilter` requirement); a shorter filtered axis is
//! an [`AxisTooShortForRecursion`](crate::FilterError::AxisTooShortForRecursion)
//! error.
//!
//! Three public entry points, all taking `sigma` per dimension:
//! - [`recursive_gaussian`] applies [`GaussianOrder::ZeroOrder`] (smoothing)
//!   to every dimension. This is the pre-existing signature, kept exactly as
//!   it was (rather than adding a defaulted parameter, which Rust has no
//!   syntax for) because `sitk-registration`'s multi-resolution pyramid
//!   already calls it and must keep compiling unchanged.
//! - [`recursive_gaussian_with_order`] additionally takes a per-dimension
//!   `orders: &[GaussianOrder]` slice — so a caller differentiates along one
//!   axis while smoothing the others, matching how ITK composes per-axis
//!   `RecursiveGaussianImageFilter`s in `GradientRecursiveGaussianImageFilter`
//!   — and a `normalize_across_scale` flag. `recursive_gaussian` is a thin
//!   wrapper over it (`ZeroOrder` on every axis, `normalize_across_scale =
//!   false`).
//! - [`smoothing_recursive_gaussian`] is the SimpleITK-level composite
//!   (`SmoothingRecursiveGaussianImageFilter`): same `ZeroOrder` recursion as
//!   `recursive_gaussian`, but always narrowing to `float`
//!   (`RebindImageType<float>`) and accepting vector images (per-component).

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::deriche::Coefficients;
use sitk_core::{Image, PixelId, parallel};

pub use sitk_core::deriche::GaussianOrder;

/// Gaussian-smooth `img` with a per-dimension physical-space `sigma`, using
/// the recursive (IIR) filter that ports `itk::RecursiveGaussianImageFilter`
/// with [`GaussianOrder::ZeroOrder`] on every dimension. See
/// [`recursive_gaussian_with_order`] to take a derivative instead.
///
/// Errors if `sigma` has the wrong length, any value is negative, or a filtered
/// axis (`sigma > 0`) has fewer than four pixels.
pub fn recursive_gaussian(img: &Image, sigma: &[f64]) -> Result<Image> {
    let orders = vec![GaussianOrder::ZeroOrder; sigma.len()];
    recursive_gaussian_with_order(img, sigma, &orders, false)
}

/// Gaussian-smooth or -differentiate `img` with a per-dimension physical-space
/// `sigma` and a per-dimension [`GaussianOrder`], using the same recursive
/// (IIR) filter as [`recursive_gaussian`]. To take, say, the x-derivative of a
/// 2-D image while smoothing along y (matching how ITK's
/// `GradientRecursiveGaussianImageFilter` composes per-axis filters), pass
/// `orders = &[GaussianOrder::FirstOrder, GaussianOrder::ZeroOrder]`.
///
/// `normalize_across_scale` applies ITK's Lindeberg scale-space
/// normalization: `FirstOrder`/`SecondOrder` output is scaled by an extra
/// `sigma^order` so a feature's peak derivative response does not depend on
/// the scale it is measured at (`ZeroOrder` is unaffected either way). Off by
/// default in ITK; `false` gives a plain (unnormalized) derivative.
///
/// Errors if `sigma` or `orders` has the wrong length, any `sigma` value is
/// negative, or a filtered axis (`sigma > 0`) has fewer than four pixels.
pub fn recursive_gaussian_with_order(
    img: &Image,
    sigma: &[f64],
    orders: &[GaussianOrder],
    normalize_across_scale: bool,
) -> Result<Image> {
    let buf = recursive_gaussian_f64(img, sigma, orders, normalize_across_scale)?;
    image_from_f64(img.pixel_id(), img.size(), img, &buf)
}

/// The `f64` core of [`recursive_gaussian_with_order`], returning the raw
/// filtered values before narrowing to any pixel type.
///
/// [`recursive_gaussian_with_order`] narrows its result back to `img`'s own
/// pixel type, which is correct for that function's contract but wrong for a
/// composite that must narrow to a *different* type without an intermediate
/// quantization step — [`crate::recursive_gaussian::smoothing_recursive_gaussian`]'s
/// `RebindImageType<float>` output and
/// [`crate::gradient::gradient_recursive_gaussian`]'s per-axis derivative
/// composite both need the unrounded `f64` result of a component that may
/// itself be, say, `UInt8`. `pub(crate)` for exactly those two callers.
pub(crate) fn recursive_gaussian_f64(
    img: &Image,
    sigma: &[f64],
    orders: &[GaussianOrder],
    normalize_across_scale: bool,
) -> Result<Vec<f64>> {
    let dim = img.dimension();
    // Validated before `to_f64_vec`, so a bad `sigma`/`orders` length still
    // reports itself ahead of a non-scalar pixel type, exactly as it did when
    // this function held the axis loop itself.
    check_sigma_and_orders(dim, sigma, orders)?;

    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let mut buf = img.to_f64_vec()?;
    recursive_gaussian_f64_into(
        &mut buf,
        &size,
        &spacing,
        sigma,
        orders,
        normalize_across_scale,
    )?;
    Ok(buf)
}

/// [`recursive_gaussian_f64`] run on a buffer the **caller owns** — the axis
/// loop, and the only place it lives.
///
/// [`recursive_gaussian_f64`] is this function plus `to_f64_vec`: one recursion,
/// not two that can drift.
///
/// It exists because the recursion is separable and a caller composes it: a
/// gradient magnitude runs the whole `dim`-axis cascade once *per axis*, so
/// anything the recursion allocates internally, that caller allocates `dim`
/// times. Every one of those is a full volume — 134 MB at 256³ — and a fresh
/// volume is not merely a `malloc`: its pages are faulted in one at a time under
/// a kernel lock, which is why [`sitk_core::parallel::map_indexed`]'s allocating
/// form measures a parallel efficiency of 0.09 against 0.90 for the same map
/// writing into a buffer that already exists. Handing the recursion a `&mut
/// [f64]` is what makes those volumes unwritable rather than merely unwise: a
/// caller holding a slice has no way to allocate one.
///
/// `buf` is `size.iter().product()` elements, dimension-0-fastest, and is
/// filtered **in place**. `spacing` is the image's, in the same axis order; the
/// recursion reparametrizes as `sigma[d] / spacing[d]`.
///
/// Errors if `sigma` or `orders` has the wrong length, any `sigma` value is
/// negative, or a filtered axis (`sigma[d] > 0`) has fewer than four pixels.
pub(crate) fn recursive_gaussian_f64_into(
    buf: &mut [f64],
    size: &[usize],
    spacing: &[f64],
    sigma: &[f64],
    orders: &[GaussianOrder],
    normalize_across_scale: bool,
) -> Result<()> {
    let dim = size.len();
    check_sigma_and_orders(dim, sigma, orders)?;
    debug_assert_eq!(buf.len(), size.iter().product::<usize>());

    let strides = strides(size);
    for d in 0..dim {
        if sigma[d] <= 0.0 {
            continue;
        }
        if size[d] < 4 {
            return Err(FilterError::AxisTooShortForRecursion {
                axis: d,
                len: size[d],
            });
        }
        let coeff = Coefficients::new(
            orders[d],
            sigma[d] / spacing[d],
            sigma[d],
            normalize_across_scale,
        );
        filter_axis(buf, size, &strides, d, &coeff);
    }

    Ok(())
}

/// [`recursive_gaussian_f64_into`] reading its input from `src` instead of from
/// the buffer it writes — for the caller that runs the whole cascade **once per
/// axis** off one unchanging input.
///
/// That caller (a gradient magnitude, a Laplacian, a per-axis derivative) cannot
/// use the in-place form directly: the recursion destroys its input, so every
/// axis needs its own copy of `src`. Copying it is what this replaces, and the
/// copy was not free — `work.copy_from_slice(&src)` is `std`'s **single-threaded**
/// `memcpy`, and at 512³ it is also where a freshly-allocated `dst` gets its 1.07 GB
/// of pages faulted in, on one core. Measured, that copy cost 517 ms on the first
/// axis and 684 ms across the three, **42% of the whole filter**, against 80 ms for
/// the *parallel* widening that fills a buffer of exactly the same size.
///
/// So the copy is not made parallel here, it is **removed**: the first filtered
/// axis reads `src` and writes `dst`, and every later axis runs in place on `dst`,
/// exactly as before. `dst`'s pages are then first touched by a parallel line pass
/// rather than by a serial `memcpy`.
///
/// This cannot move a bit. [`filter_line`] runs on a line that is fully gathered
/// into a contiguous scratch buffer before it is called, and it writes the line
/// back only after it returns, so each output element is computed from exactly the
/// values it was computed from before, in the same order — the only thing that
/// changed is *which buffer those values were read out of*.
///
/// `src` and `dst` must be the same length; `dst`'s prior contents are ignored.
/// If no axis is filtered at all (every `sigma[d] <= 0`), the result is `src`
/// unchanged, so `dst` is filled from `src` — that is the one case where a copy is
/// still the correct answer, because there is no pass to fold it into.
pub(crate) fn recursive_gaussian_f64_from_into(
    src: &[f64],
    dst: &mut [f64],
    size: &[usize],
    spacing: &[f64],
    sigma: &[f64],
    orders: &[GaussianOrder],
    normalize_across_scale: bool,
) -> Result<()> {
    let dim = size.len();
    check_sigma_and_orders(dim, sigma, orders)?;
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(dst.len(), size.iter().product::<usize>());

    let strides = strides(size);
    let mut wrote_dst = false;
    for d in 0..dim {
        if sigma[d] <= 0.0 {
            continue;
        }
        if size[d] < 4 {
            return Err(FilterError::AxisTooShortForRecursion {
                axis: d,
                len: size[d],
            });
        }
        let coeff = Coefficients::new(
            orders[d],
            sigma[d] / spacing[d],
            sigma[d],
            normalize_across_scale,
        );
        if wrote_dst {
            filter_axis(dst, size, &strides, d, &coeff);
        } else {
            filter_axis_from(src, dst, size, &strides, d, &coeff);
            wrote_dst = true;
        }
    }

    if !wrote_dst {
        dst.copy_from_slice(src);
    }
    Ok(())
}

/// The `sigma`/`orders` shape checks both entry points above make, in the order
/// they have always been made in.
fn check_sigma_and_orders(dim: usize, sigma: &[f64], orders: &[GaussianOrder]) -> Result<()> {
    if sigma.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: sigma.len(),
        });
    }
    if orders.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: orders.len(),
        });
    }
    if sigma.iter().any(|&s| s < 0.0) {
        return Err(FilterError::InvalidSigma(sigma.to_vec()));
    }
    Ok(())
}

/// `SmoothingRecursiveGaussianImageFilter`
/// (`itkSmoothingRecursiveGaussianImageFilter.hxx`): Gaussian-smooth `img`
/// with a per-dimension physical-space `sigma`
/// (`SmoothingRecursiveGaussianImageFilter.yaml:9-14`, `dim_vec: true`,
/// default `[1.0, 1.0, 1.0]`), applying `recursive_gaussian_f64`'s
/// [`GaussianOrder::ZeroOrder`] recursion to every axis.
///
/// `normalize_across_scale` is ITK's `NormalizeAcrossScale`
/// (`SmoothingRecursiveGaussianImageFilter.yaml:27-29`, default `false`,
/// matching the ITK class default at `itkSmoothingRecursiveGaussianImageFilter.h:175`);
/// [`GaussianOrder::ZeroOrder`] ignores it either way (see
/// `Coefficients::new`'s `ZeroOrder` arm), so it has no observable effect on
/// this filter's output — reproduced faithfully rather than dropped, since a
/// caller may still pass it through generically.
///
/// **Cascade order.** ITK's own pipeline (`itkSmoothingRecursiveGaussianImageFilter.hxx:26-57`)
/// filters axis `D-1` first (`m_FirstSmoothingFilter`), then axes `0..D-2` in
/// order (`m_SmoothingFilters`) — chosen for cache/in-place performance, not
/// correctness. This port filters axes `0..D` in the crate's canonical order
/// (the same order [`recursive_gaussian`]/[`recursive_gaussian_with_order`]
/// already use). Every axis here is [`GaussianOrder::ZeroOrder`] smoothing —
/// a separable linear operation — so the two cascades are the *same*
/// composition of independent per-axis passes; they can only disagree in
/// floating-point summation order, at the ULP level, not in the axes each
/// pixel's value is averaged over.
///
/// **Output pixel type.** The yaml's `output_image_type` is
/// `InputImageType::RebindImageType<float>` (`SmoothingRecursiveGaussianImageFilter.yaml:7`) —
/// rebinding a `VectorImage<T, D>` gives `VectorImage<float, D>`
/// (`itkVectorImage.h:201`) — so this always outputs [`PixelId::Float32`] for
/// a scalar input, [`PixelId::VectorFloat32`] for a vector one, regardless of
/// the input's own pixel type. This is a distinct rule from the crate's
/// `real_pixel_id`/`NumericTraits<T>::RealType` family (ledger §5.6): that
/// family widens `Float64` input to `Float64` output but this port's members
/// keep `Float32`, a documented divergence from ITK's `double` `RealType`.
/// `RebindImageType<float>` instead narrows unconditionally — a `Float64`
/// input still produces `Float32` output here, exactly as the real SimpleITK
/// procedural function does — so there is no Float32↔Float64 flip to
/// document against that family; this filter (and [`crate::gradient::gradient`]/
/// [`crate::gradient::gradient_recursive_gaussian`], which share the same
/// yaml rule) are simply outside it.
///
/// **Vector images.** `pixel_types` is `BasicPixelIDTypeList` **+**
/// `VectorPixelIDTypeList` (`SmoothingRecursiveGaussianImageFilter.yaml:6`) —
/// unlike [`recursive_gaussian`]/[`recursive_gaussian_with_order`], which are
/// scalar-only (their `to_f64_vec`/`image_from_f64` scalar seam rejects a
/// vector image outright). The yaml's own doc says "For multi-component
/// images, the filter works on each component independently"; this port
/// reproduces that literally: [`sitk_core::Image::extract_component`] each
/// component, run the same `recursive_gaussian_f64` recursion on it, narrow
/// straight to `f32` (skipping any intermediate narrowing to the component's
/// own type — seeing `img.pixel_id()` narrowed away and then re-cast would
/// quantize a `UInt8` component to an 8-bit integer before it ever reaches
/// `float`, which is not what `RebindImageType<float>` means), and
/// [`sitk_core::Image::from_component_images`] them back together. A complex
/// image is rejected the same way [`crate::dicom_orient::dicom_orient`]'s
/// vector/scalar branch rejects one: `is_vector()` is `false` for
/// `ComplexFloat32`/`ComplexFloat64`, so it falls into the scalar branch,
/// whose `to_f64_vec` returns `Error::RequiresScalarPixelType`.
///
/// Errors if `sigma` has the wrong length, any value is negative, or a
/// filtered axis (`sigma[d] > 0`) has fewer than four pixels.
pub fn smoothing_recursive_gaussian(
    img: &Image,
    sigma: &[f64],
    normalize_across_scale: bool,
) -> Result<Image> {
    let dim = img.dimension();
    let zero_orders = vec![GaussianOrder::ZeroOrder; dim];

    if img.pixel_id().is_vector() {
        let n = img.number_of_components_per_pixel();
        let mut components = Vec::with_capacity(n);
        for c in 0..n {
            let component = img.extract_component(c)?;
            let buf =
                recursive_gaussian_f64(&component, sigma, &zero_orders, normalize_across_scale)?;
            components.push(image_from_f64(
                PixelId::Float32,
                component.size(),
                &component,
                &buf,
            )?);
        }
        let refs: Vec<&Image> = components.iter().collect();
        Ok(Image::from_component_images(&refs)?)
    } else {
        let buf = recursive_gaussian_f64(img, sigma, &zero_orders, normalize_across_scale)?;
        image_from_f64(PixelId::Float32, img.size(), img, &buf)
    }
}

/// Filter every line of `buf` along axis `d` in place, gathering each line into
/// a contiguous buffer, running the recursion, and scattering it back.
///
/// **Parallel over lines** ([`parallel::for_each_line_mut`]). A line along axis
/// `d` reads and writes only its own elements, so lines are independent and the
/// pass is bit-identical to the sequential line loop: [`filter_line`]'s
/// fourth-order recursion — the only place floats accumulate — runs in its own
/// unchanged sequential order *within* each line, and no value crosses lines.
/// The three scratch buffers are per-task, not per-line, so the recursion still
/// allocates once rather than once per line.
///
/// `strides` is unused by the decomposition (the primitive derives the line
/// stride from `size` and `d` itself); it stays in the signature because the
/// callers already carry it.
fn filter_axis(buf: &mut [f64], size: &[usize], strides: &[usize], d: usize, coeff: &Coefficients) {
    debug_assert_eq!(strides[d], size[..d].iter().product::<usize>());
    let ln = size[d];
    parallel::for_each_line_mut(
        buf,
        size,
        d,
        || (vec![0.0f64; ln], vec![0.0f64; ln], vec![0.0f64; ln]),
        |(line, outs, scratch), mut slot| {
            for (k, v) in line.iter_mut().enumerate() {
                *v = slot.get(k);
            }
            filter_line(line, coeff, outs, scratch);
            for (k, &v) in outs.iter().enumerate() {
                slot.set(k, v);
            }
        },
    );
}

/// [`filter_axis`] reading each line from `src` rather than from the buffer it
/// writes. `src` and `dst` have the same shape, so a line's `k`-th element lives
/// at `slot.start() + k * slot.stride()` in **either** buffer — which is what
/// `Line`'s `start`/`stride` are exposed for.
///
/// The gather, the recursion, and the write-back are the same three steps as
/// [`filter_axis`], in the same order; only the buffer the gather reads from
/// differs. The line is fully materialized into a contiguous scratch buffer
/// before [`filter_line`] sees it, so every output element is computed from the
/// same values in the same order as before: bit-identical, by construction.
fn filter_axis_from(
    src: &[f64],
    dst: &mut [f64],
    size: &[usize],
    strides: &[usize],
    d: usize,
    coeff: &Coefficients,
) {
    debug_assert_eq!(strides[d], size[..d].iter().product::<usize>());
    let ln = size[d];
    parallel::for_each_line_mut(
        dst,
        size,
        d,
        || (vec![0.0f64; ln], vec![0.0f64; ln], vec![0.0f64; ln]),
        |(line, outs, scratch), mut slot| {
            let (start, stride) = (slot.start(), slot.stride());
            for (k, v) in line.iter_mut().enumerate() {
                *v = src[start + k * stride];
            }
            filter_line(line, coeff, outs, scratch);
            for (k, &v) in outs.iter().enumerate() {
                slot.set(k, v);
            }
        },
    );
}

/// One line through the fourth-order causal + anti-causal recursion, porting
/// `RecursiveSeparableImageFilter::FilterDataArray` operation-for-operation.
/// This recursion is the same for every [`GaussianOrder`] — only the
/// [`Coefficients`] fed in differ. The border value is assumed to extend to
/// infinity on both ends. Requires `data.len() >= 4`; `outs` and `scratch`
/// are caller-provided scratch buffers of the same length as `data` (their
/// prior contents are overwritten).
fn filter_line(data: &[f64], c: &Coefficients, outs: &mut [f64], scratch: &mut [f64]) {
    let ln = data.len();

    // ---- Causal (forward) pass ----
    let out_v1 = data[0];

    outs[0] = out_v1 * c.n0 + out_v1 * c.n1 + out_v1 * c.n2 + out_v1 * c.n3;
    outs[1] = data[1] * c.n0 + out_v1 * c.n1 + out_v1 * c.n2 + out_v1 * c.n3;
    outs[2] = data[2] * c.n0 + data[1] * c.n1 + out_v1 * c.n2 + out_v1 * c.n3;
    outs[3] = data[3] * c.n0 + data[2] * c.n1 + data[1] * c.n2 + out_v1 * c.n3;

    // The border value is multiplied by the boundary coefficients m_BNi.
    outs[0] -= out_v1 * c.bn1 + out_v1 * c.bn2 + out_v1 * c.bn3 + out_v1 * c.bn4;
    outs[1] -= outs[0] * c.d1 + out_v1 * c.bn2 + out_v1 * c.bn3 + out_v1 * c.bn4;
    outs[2] -= outs[1] * c.d1 + outs[0] * c.d2 + out_v1 * c.bn3 + out_v1 * c.bn4;
    outs[3] -= outs[2] * c.d1 + outs[1] * c.d2 + outs[0] * c.d3 + out_v1 * c.bn4;

    for i in 4..ln {
        outs[i] = data[i] * c.n0 + data[i - 1] * c.n1 + data[i - 2] * c.n2 + data[i - 3] * c.n3;
        outs[i] -=
            outs[i - 1] * c.d1 + outs[i - 2] * c.d2 + outs[i - 3] * c.d3 + outs[i - 4] * c.d4;
    }

    // ---- Anti-causal (backward) pass into scratch ----
    let out_v2 = data[ln - 1];

    scratch[ln - 1] = out_v2 * c.m1 + out_v2 * c.m2 + out_v2 * c.m3 + out_v2 * c.m4;
    scratch[ln - 2] = data[ln - 1] * c.m1 + out_v2 * c.m2 + out_v2 * c.m3 + out_v2 * c.m4;
    scratch[ln - 3] = data[ln - 2] * c.m1 + data[ln - 1] * c.m2 + out_v2 * c.m3 + out_v2 * c.m4;
    scratch[ln - 4] =
        data[ln - 3] * c.m1 + data[ln - 2] * c.m2 + data[ln - 1] * c.m3 + out_v2 * c.m4;

    // The border value is multiplied by the boundary coefficients m_BMi.
    scratch[ln - 1] -= out_v2 * c.bm1 + out_v2 * c.bm2 + out_v2 * c.bm3 + out_v2 * c.bm4;
    scratch[ln - 2] -= scratch[ln - 1] * c.d1 + out_v2 * c.bm2 + out_v2 * c.bm3 + out_v2 * c.bm4;
    scratch[ln - 3] -=
        scratch[ln - 2] * c.d1 + scratch[ln - 1] * c.d2 + out_v2 * c.bm3 + out_v2 * c.bm4;
    scratch[ln - 4] -=
        scratch[ln - 3] * c.d1 + scratch[ln - 2] * c.d2 + scratch[ln - 1] * c.d3 + out_v2 * c.bm4;

    // ITK's loop: for (i = ln - 4; i > 0; i--) writes scratch[i - 1].
    let mut i = ln - 4;
    while i > 0 {
        scratch[i - 1] =
            data[i] * c.m1 + data[i + 1] * c.m2 + data[i + 2] * c.m3 + data[i + 3] * c.m4;
        scratch[i - 1] -= scratch[i] * c.d1
            + scratch[i + 1] * c.d2
            + scratch[i + 2] * c.d3
            + scratch[i + 3] * c.d4;
        i -= 1;
    }

    // Roll the anti-causal part into the output.
    for (o, &s) in outs.iter_mut().zip(scratch.iter()) {
        *o += s;
    }
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::{Image, PixelId};

    #[test]
    fn zero_sigma_is_identity() {
        let img = Image::from_vec(&[6, 5], (0..30).map(|v| v as f64).collect()).unwrap();
        let out = recursive_gaussian(&img, &[0.0, 0.0]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn constant_image_is_preserved() {
        // Unity DC gain (the alpha0 normalization) plus edge-replicating borders
        // keep a constant exactly.
        let img = Image::from_vec(&[10, 10], vec![5.0; 100]).unwrap();
        let out = recursive_gaussian(&img, &[2.0, 2.0]).unwrap();
        for v in out.to_f64_vec().unwrap() {
            assert!((v - 5.0).abs() < 1e-10, "constant not preserved: {v}");
        }
    }

    #[test]
    fn one_dimensional_constant_is_preserved_exactly() {
        // A single line stresses the boundary coefficients directly.
        let img = Image::from_vec(&[16, 1], vec![3.5; 16]).unwrap();
        let out = recursive_gaussian(&img, &[1.5, 0.0]).unwrap();
        for v in out.to_f64_vec().unwrap() {
            assert!((v - 3.5).abs() < 1e-10, "1-D constant not preserved: {v}");
        }
    }

    #[test]
    fn interior_impulse_conserves_mass_and_is_symmetric() {
        // A single impulse well away from the border keeps its total mass (unity
        // DC gain) and spreads symmetrically about the center on both axes,
        // confirming the separable pass is applied correctly on each axis. The
        // grid is sized so the heavier IIR tails are fully contained (at n=81,
        // sigma=2 the edge leakage is ~1e-10).
        let n = 81;
        let c = n / 2;
        let mut data = vec![0.0f64; n * n];
        data[c * n + c] = 100.0;
        let img = Image::from_vec(&[n, n], data).unwrap();
        let v = recursive_gaussian(&img, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec()
            .unwrap();

        let total: f64 = v.iter().sum();
        assert!((total - 100.0).abs() < 1e-6, "mass not conserved: {total}");

        let peak = v[c * n + c];
        assert!(peak < 100.0 && peak > 0.0, "peak not spread: {peak}");
        assert!(
            (v[c * n + (c - 1)] - v[c * n + (c + 1)]).abs() < 1e-9,
            "x asymmetric"
        );
        assert!(
            (v[(c - 1) * n + c] - v[(c + 1) * n + c]).abs() < 1e-9,
            "y asymmetric"
        );
    }

    #[test]
    fn impulse_response_width_matches_the_itk_recursive_gaussian() {
        // The zero-order recursive filter approximates a Gaussian, but its
        // effective width is a *fixed* fraction of the nominal sigma: the second
        // moment of the impulse response is 0.93800 * sigma^2 at every scale
        // (measured identical to 5 digits at sigma = 2, 4, 8), whereas the FIR
        // that samples the true kernel gives ~sigma^2. This ~6.2% narrowing is a
        // genuine property of the Farnebäck ZeroOrder coefficients as ITK uses
        // them, not truncation. Pinning the ratio guards the coefficient math
        // against a transposed constant.
        let n = 201;
        let center = n / 2;
        let mut data = vec![0.0f64; n];
        data[center] = 1.0;
        let img = Image::from_vec(&[n, 1], data).unwrap();

        let sigma = 4.0;
        let out = recursive_gaussian(&img, &[sigma, 0.0])
            .unwrap()
            .to_f64_vec()
            .unwrap();

        // On this grid (edges at 25*sigma) the tails are fully contained, so the
        // DC gain (unit mass) shows to near machine precision.
        let mass: f64 = out.iter().sum();
        assert!((mass - 1.0).abs() < 1e-7, "impulse mass: {mass}");

        let var: f64 = out
            .iter()
            .enumerate()
            .map(|(i, &w)| w * (i as f64 - center as f64).powi(2))
            .sum();
        let ratio = var / (sigma * sigma);
        assert!(
            (0.935..=0.941).contains(&ratio),
            "variance/sigma^2 ratio {ratio} outside the recursive filter's 0.938 band"
        );
    }

    #[test]
    fn approximates_the_fir_gaussian_on_a_smooth_blob() {
        // The recursive and FIR filters approximate the same continuous Gaussian,
        // so on a smooth signal well inside the borders they agree closely.
        let n = 48;
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f64 - 24.0, y as f64 - 24.0);
                data[y * n + x] = (-(dx * dx + dy * dy) / 60.0).exp();
            }
        }
        let img = Image::from_vec(&[n, n], data).unwrap();

        let rec = recursive_gaussian(&img, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let fir = crate::smooth_gaussian(&img, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec()
            .unwrap();

        // Compare the interior (away from the borders) where both are accurate.
        let mut max_abs = 0.0f64;
        for y in 8..n - 8 {
            for x in 8..n - 8 {
                max_abs = max_abs.max((rec[y * n + x] - fir[y * n + x]).abs());
            }
        }
        assert!(max_abs < 5e-3, "recursive vs FIR interior diff {max_abs}");
    }

    #[test]
    fn physical_sigma_accounts_for_spacing() {
        // With spacing 2, a physical sigma of 2 is only 1 voxel of blur, so an
        // impulse spreads less (higher retained peak) than with spacing 1.
        let n = 41;
        let mut data = vec![0.0f64; n * n];
        data[20 * n + 20] = 100.0;

        let mut fine = Image::from_vec(&[n, n], data.clone()).unwrap();
        fine.set_spacing(&[1.0, 1.0]).unwrap();
        let mut coarse = Image::from_vec(&[n, n], data).unwrap();
        coarse.set_spacing(&[2.0, 2.0]).unwrap();

        let peak_fine = recursive_gaussian(&fine, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec()
            .unwrap()[20 * n + 20];
        let peak_coarse = recursive_gaussian(&coarse, &[2.0, 2.0])
            .unwrap()
            .to_f64_vec()
            .unwrap()[20 * n + 20];
        assert!(
            peak_coarse > peak_fine,
            "coarser spacing should blur less: {peak_coarse} vs {peak_fine}"
        );
    }

    #[test]
    fn wrong_sigma_length_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian(&img, &[1.0]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn negative_sigma_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian(&img, &[-1.0, 1.0]),
            Err(FilterError::InvalidSigma(_))
        ));
    }

    #[test]
    fn short_filtered_axis_is_rejected() {
        // Fewer than four pixels along a filtered axis cannot feed the
        // fourth-order recursion (ITK throws the same requirement).
        let img = Image::new(&[3, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian(&img, &[1.0, 1.0]),
            Err(FilterError::AxisTooShortForRecursion { axis: 0, len: 3 })
        ));
    }

    #[test]
    fn short_axis_is_fine_when_not_filtered() {
        // A short axis is only a problem if it is actually filtered (sigma > 0).
        let img = Image::from_vec(&[3, 8], vec![2.0; 24]).unwrap();
        let out = recursive_gaussian(&img, &[0.0, 1.0]).unwrap();
        for v in out.to_f64_vec().unwrap() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    // ---- derivative orders -------------------------------------------------

    /// 1-D helper: run [`recursive_gaussian_with_order`] with a single order
    /// on a `[n, 1]` image, returning the filtered line.
    fn filter_1d(data: &[f64], sigma: f64, order: GaussianOrder) -> Vec<f64> {
        let img = Image::from_vec(&[data.len(), 1], data.to_vec()).unwrap();
        recursive_gaussian_with_order(
            &img,
            &[sigma, 0.0],
            &[order, GaussianOrder::ZeroOrder],
            false,
        )
        .unwrap()
        .to_f64_vec()
        .unwrap()
    }

    #[test]
    fn first_order_of_constant_is_near_zero() {
        // The derivative of a constant is 0; the recursion's border extension
        // ("the border value extends to infinity") makes this exact for an
        // interior/infinite constant, so only floating-point roundoff remains.
        let out = filter_1d(&vec![7.0; 32], 3.0, GaussianOrder::FirstOrder);
        for v in out {
            assert!(v.abs() < 1e-9, "first-order of constant not ~0: {v}");
        }
    }

    #[test]
    fn second_order_of_linear_ramp_is_near_zero() {
        // The second derivative of a line is 0. Gaussian smoothing preserves a
        // linear ramp exactly (away from the border), so SecondOrder should
        // read ~0 in the interior. Near the border the "extends to infinity"
        // assumption is violated by a ramp (its true continuation keeps
        // sloping, not staying flat), so the smoothed ramp deviates from the
        // ideal ramp by a boundary-leakage term that decays geometrically
        // (rate `exp(L/sigmad)` per index step, from the recursion's poles);
        // at margin=60 with sigma=3 (sigmad=3) that is `exp(-1.39/3)^60 ~
        // 1e-12`, far below the 1e-8 tolerance used here.
        let n = 200;
        let margin = 60;
        let ramp: Vec<f64> = (0..n).map(|i| 2.5 * i as f64 - 10.0).collect();
        let out = filter_1d(&ramp, 3.0, GaussianOrder::SecondOrder);
        for &v in &out[margin..n - margin] {
            assert!(v.abs() < 1e-8, "second-order of ramp not ~0: {v}");
        }
    }

    #[test]
    fn first_order_matches_analytic_derivative_of_smoothed_gaussian() {
        // f(x) = exp(-(x-c)^2 / (2*s0^2)), a Gaussian bump of width s0.
        // Convolving with a Gaussian of width `sigma` gives (analytically)
        // another Gaussian of width sqrt(s0^2 + sigma^2), scaled by the ratio
        // of the two Gaussians' normalization (s0 / sqrt(s0^2+sigma^2)) since
        // the input here is unnormalized (peak 1, not unit-area). We compare
        // FirstOrder's output against the analytic derivative of that
        // composite Gaussian.
        //
        // The recursive filter only approximates convolution with a true
        // Gaussian (the zero-order impulse response's variance is ~0.938 *
        // sigma^2, see `impulse_response_width_matches_the_itk_recursive_gaussian`),
        // so some relative deviation from the ideal-Gaussian analytic
        // reference is expected even far from any boundary. Empirically (this
        // test, at sigma = 2 and 5.5) the observed peak relative error is
        // ~0.07%-0.19%; 1% of the peak slope keeps a >5x margin over that
        // while catching a mis-derived coefficient (which mismatches by tens
        // of percent, not fractions of one).
        let n = 401;
        let center = 200.0;
        let s0 = 12.0;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let dx = i as f64 - center;
                (-dx * dx / (2.0 * s0 * s0)).exp()
            })
            .collect();

        for &sigma in &[2.0, 5.5] {
            let out = filter_1d(&data, sigma, GaussianOrder::FirstOrder);
            let s_eff2 = s0 * s0 + sigma * sigma;
            let s_eff = s_eff2.sqrt();
            let amp = s0 / s_eff;
            let peak_slope = amp / (s_eff * (-0.5f64).exp().sqrt()); // slope magnitude at the inflection

            // Sample away from the borders (>|8*sigma|) where boundary leakage
            // is negligible relative to the tolerance below.
            for i in (60..n - 60).step_by(5) {
                let x = i as f64 - center;
                let expected = -amp * x / s_eff2 * (-x * x / (2.0 * s_eff2)).exp();
                let got = out[i];
                assert!(
                    (got - expected).abs() < 0.01 * peak_slope,
                    "sigma={sigma} i={i}: got {got}, expected {expected} (peak_slope {peak_slope})"
                );
            }
        }
    }

    #[test]
    fn second_order_matches_analytic_second_derivative_of_smoothed_gaussian() {
        // Same composite-Gaussian setup as
        // `first_order_matches_analytic_derivative_of_smoothed_gaussian`, but
        // comparing SecondOrder against the analytic second derivative
        // d^2/dx^2 [ amp * exp(-x^2 / (2*s_eff^2)) ]
        //   = amp * (x^2 - s_eff^2) / s_eff^4 * exp(-x^2 / (2*s_eff^2))
        //
        // Empirically (this test, at sigma = 2 and 5.5) the observed peak
        // relative error is ~0.20%-0.40% (larger than FirstOrder's, since the
        // second derivative amplifies the same ~0.938-sigma^2 approximation
        // bias further); 1.5% keeps a >3.5x margin over that.
        let n = 401;
        let center = 200.0;
        let s0 = 12.0;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let dx = i as f64 - center;
                (-dx * dx / (2.0 * s0 * s0)).exp()
            })
            .collect();

        for &sigma in &[2.0, 5.5] {
            let out = filter_1d(&data, sigma, GaussianOrder::SecondOrder);
            let s_eff2 = s0 * s0 + sigma * sigma;
            let amp = s0 / s_eff2.sqrt();
            let peak_curvature = amp / (s_eff2 * s_eff2) * s_eff2; // = amp / s_eff2, curvature scale at x=0

            for i in (60..n - 60).step_by(5) {
                let x = i as f64 - center;
                let expected =
                    amp * (x * x - s_eff2) / (s_eff2 * s_eff2) * (-x * x / (2.0 * s_eff2)).exp();
                let got = out[i];
                assert!(
                    (got - expected).abs() < 0.015 * peak_curvature,
                    "sigma={sigma} i={i}: got {got}, expected {expected} (peak_curvature {peak_curvature})"
                );
            }
        }
    }

    #[test]
    fn first_order_of_ramp_matches_constant_slope() {
        // Gaussian-smoothing a linear ramp reproduces it exactly (in the
        // interior), so its first derivative is the ramp's constant slope.
        // Same boundary-leakage argument (and tolerance) as
        // `second_order_of_linear_ramp_is_near_zero`.
        let n = 200;
        let margin = 60;
        let slope = 2.5;
        let ramp: Vec<f64> = (0..n).map(|i| slope * i as f64 - 10.0).collect();
        let out = filter_1d(&ramp, 3.0, GaussianOrder::FirstOrder);
        for &v in &out[margin..n - margin] {
            assert!((v - slope).abs() < 1e-8, "first-order of ramp: {v}");
        }
    }

    #[test]
    fn two_dimensional_derivative_matches_along_each_axis() {
        // A separable 2-D Gaussian bump f(x,y) = exp(-((x-cx)^2+(y-cy)^2)/(2*s0^2)).
        // Differentiating along one axis while smoothing (ZeroOrder) along the
        // other reproduces the 1-D analytic derivative in that axis, since the
        // bump factors as g(x) * g(y).
        let n = 121;
        let c = 60.0;
        let s0 = 10.0;
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f64 - c, y as f64 - c);
                data[y * n + x] = (-(dx * dx + dy * dy) / (2.0 * s0 * s0)).exp();
            }
        }
        let img = Image::from_vec(&[n, n], data).unwrap();
        let sigma = 3.0;
        let s_eff2 = s0 * s0 + sigma * sigma;
        let amp = s0 / s_eff2.sqrt();
        let peak_slope = amp / s_eff2.sqrt() * (-0.5f64).exp().sqrt();

        // d/dx: FirstOrder on axis 0, ZeroOrder on axis 1.
        let dx_out = recursive_gaussian_with_order(
            &img,
            &[sigma, sigma],
            &[GaussianOrder::FirstOrder, GaussianOrder::ZeroOrder],
            false,
        )
        .unwrap()
        .to_f64_vec()
        .unwrap();
        // Empirically the observed peak relative error here is ~0.38%; 1.5%
        // (same margin as the 1-D SecondOrder case) covers the 2-D case's
        // extra separable-product rounding.
        for yi in (30..n - 30).step_by(10) {
            for xi in (30..n - 30).step_by(10) {
                let (x, y) = (xi as f64 - c, yi as f64 - c);
                let gy = (-y * y / (2.0 * s_eff2)).exp();
                let expected = -amp * x / s_eff2 * (-x * x / (2.0 * s_eff2)).exp() * amp * gy;
                let got = dx_out[yi * n + xi];
                assert!(
                    (got - expected).abs() < 0.015 * peak_slope * amp,
                    "d/dx at ({xi},{yi}): got {got}, expected {expected}"
                );
            }
        }

        // d/dy: ZeroOrder on axis 0, FirstOrder on axis 1.
        let dy_out = recursive_gaussian_with_order(
            &img,
            &[sigma, sigma],
            &[GaussianOrder::ZeroOrder, GaussianOrder::FirstOrder],
            false,
        )
        .unwrap()
        .to_f64_vec()
        .unwrap();
        for yi in (30..n - 30).step_by(10) {
            for xi in (30..n - 30).step_by(10) {
                let (x, y) = (xi as f64 - c, yi as f64 - c);
                let gx = (-x * x / (2.0 * s_eff2)).exp();
                let expected = -amp * y / s_eff2 * (-y * y / (2.0 * s_eff2)).exp() * amp * gx;
                let got = dy_out[yi * n + xi];
                assert!(
                    (got - expected).abs() < 0.015 * peak_slope * amp,
                    "d/dy at ({xi},{yi}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn wrong_orders_length_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            recursive_gaussian_with_order(&img, &[1.0, 1.0], &[GaussianOrder::ZeroOrder], false),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    // ---- smoothing_recursive_gaussian --------------------------------------

    #[test]
    fn smoothing_recursive_gaussian_scalar_matches_recursive_gaussian_narrowed_to_float32() {
        let n = 41;
        let mut data = vec![0.0f64; n * n];
        data[20 * n + 20] = 100.0;
        let img = Image::from_vec(&[n, n], data).unwrap();

        let out = smoothing_recursive_gaussian(&img, &[2.0, 2.0], false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);

        let reference = recursive_gaussian(&img, &[2.0, 2.0]).unwrap();
        let reference_f32 = crate::cast(&reference, PixelId::Float32).unwrap();
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            reference_f32.scalar_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn smoothing_recursive_gaussian_output_is_always_float32() {
        let img = Image::from_vec(&[9, 9], vec![5u8; 81]).unwrap();
        let out = smoothing_recursive_gaussian(&img, &[1.0, 1.0], false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    /// The out-of-place first pass must be **bit-identical** to copying the input
    /// and filtering in place — that equality is the entire licence for
    /// `recursive_gaussian_f64_from_into` to exist, so it is pinned rather than
    /// argued. `assert_eq!` on `f64` is exact here on purpose: a value that is
    /// merely close is a failure.
    ///
    /// Anisotropic spacing, a non-cubic volume, and every `GaussianOrder`,
    /// including the axis-skipping `sigma == 0` case (whose fallback is the one
    /// path that still copies) and the all-zero case (where nothing is filtered
    /// and the result must be the input).
    #[test]
    fn out_of_place_first_pass_is_bit_identical_to_copy_then_filter_in_place() {
        let size = [9usize, 7, 5];
        let spacing = [1.0f64, 0.75, 1.3];
        let n: usize = size.iter().product();

        // Deterministic, non-degenerate, and not symmetric about any axis.
        let src: Vec<f64> = (0..n)
            .map(|i| ((i * 37 % 101) as f64) * 0.5 - 11.0 + (i % 7) as f64)
            .collect();

        use GaussianOrder::{FirstOrder, SecondOrder, ZeroOrder};
        let cases: &[([GaussianOrder; 3], [f64; 3])] = &[
            ([FirstOrder, ZeroOrder, ZeroOrder], [1.5, 1.5, 1.5]),
            ([ZeroOrder, FirstOrder, ZeroOrder], [1.5, 1.5, 1.5]),
            ([ZeroOrder, ZeroOrder, SecondOrder], [2.0, 1.0, 0.8]),
            ([SecondOrder, FirstOrder, ZeroOrder], [0.9, 1.7, 2.3]),
            // sigma == 0 on the leading axis: the first *filtered* axis is axis 1,
            // so the out-of-place pass must land there, not on axis 0.
            ([ZeroOrder, FirstOrder, ZeroOrder], [0.0, 1.5, 1.5]),
            // Nothing filtered at all: the result is the input, unchanged.
            ([ZeroOrder, ZeroOrder, ZeroOrder], [0.0, 0.0, 0.0]),
        ];

        for (normalize, (orders, sigma)) in [false, true]
            .into_iter()
            .flat_map(|nz| cases.iter().map(move |c| (nz, c)))
        {
            // Reference: the in-place recursion, fed the copy it has always been fed.
            let mut reference = src.clone();
            recursive_gaussian_f64_into(&mut reference, &size, &spacing, sigma, orders, normalize)
                .unwrap();

            // Under test: the same cascade, with the first pass reading `src`.
            // `dst` starts full of a poison value, so a pass that fails to write
            // an element cannot pass by accidentally holding the right one.
            let mut dst = vec![f64::NAN; n];
            recursive_gaussian_f64_from_into(
                &src, &mut dst, &size, &spacing, sigma, orders, normalize,
            )
            .unwrap();

            assert_eq!(
                dst, reference,
                "out-of-place first pass diverged from copy-then-filter-in-place \
                 (orders={orders:?}, sigma={sigma:?}, normalize={normalize})"
            );
            // And `src` itself is untouched — the caller reuses it on the next axis.
            assert!(src.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn smoothing_recursive_gaussian_avoids_intermediate_quantization() {
        // A UInt8 edge (0 -> 255) smoothed with a wide Gaussian blurs to a
        // fractional value near the edge (e.g. ~90.7), which does NOT survive
        // an intermediate round to the nearest u8 and back — pinning that this
        // composite narrows straight from the f64 recursion to f32 rather than
        // quantizing to the component's own (UInt8) pixel type first.
        let n = 41;
        let mut data = vec![0u8; n];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if i < n / 2 { 0 } else { 255 };
        }
        let img = Image::from_vec(&[n, 1], data).unwrap();
        let out = smoothing_recursive_gaussian(&img, &[3.0, 0.0], false).unwrap();
        let vals = out.scalar_slice::<f32>().unwrap();

        // The exact-double reference, computed the same way but never narrowed
        // to anything but f32: this must match to full f32 precision.
        let buf = recursive_gaussian_f64(
            &img,
            &[3.0, 0.0],
            &[GaussianOrder::ZeroOrder, GaussianOrder::ZeroOrder],
            false,
        )
        .unwrap();
        for (got, &expected) in vals.iter().zip(buf.iter()) {
            assert_eq!(*got, expected as f32);
        }
        // And at least one value near the edge is non-integral once widened
        // back to f64 -- proof that no intermediate u8 rounding occurred.
        let near_edge = vals[n / 2] as f64;
        assert!(
            (near_edge - near_edge.round()).abs() > 1e-3,
            "value near the edge looks like it was rounded to an integer: {near_edge}"
        );
    }

    #[test]
    fn smoothing_recursive_gaussian_normalize_across_scale_is_inert_for_zero_order() {
        let n = 41;
        let mut data = vec![0.0f64; n * n];
        data[20 * n + 20] = 100.0;
        let img = Image::from_vec(&[n, n], data).unwrap();

        let off = smoothing_recursive_gaussian(&img, &[2.0, 2.0], false).unwrap();
        let on = smoothing_recursive_gaussian(&img, &[2.0, 2.0], true).unwrap();
        assert_eq!(
            off.scalar_slice::<f32>().unwrap(),
            on.scalar_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn smoothing_recursive_gaussian_vector_image_smooths_each_component_independently() {
        // Two components with different profiles: component 0 an impulse,
        // component 1 a ramp -- verifies each is filtered on its own, not
        // mixed, and that the vector composite matches running the scalar
        // composite on each extracted component.
        let n = 41;
        let mut data = vec![0.0f64; n * 2];
        for x in 0..n {
            data[x * 2] = if x == 20 { 100.0 } else { 0.0 };
            data[x * 2 + 1] = x as f64;
        }
        let img = Image::from_vec_vector(&[n, 1], 2, data).unwrap();

        let out = smoothing_recursive_gaussian(&img, &[3.0, 0.0], false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.number_of_components_per_pixel(), 2);

        for c in 0..2 {
            let scalar_component = img.extract_component(c).unwrap();
            let scalar_out = smoothing_recursive_gaussian(&scalar_component, &[3.0, 0.0], false)
                .unwrap()
                .scalar_slice::<f32>()
                .unwrap()
                .to_vec();
            let vector_component = out
                .extract_component(c)
                .unwrap()
                .scalar_slice::<f32>()
                .unwrap()
                .to_vec();
            assert_eq!(scalar_component.pixel_id(), PixelId::Float64);
            assert_eq!(scalar_out, vector_component);
        }
    }

    #[test]
    fn smoothing_recursive_gaussian_rejects_a_complex_image() {
        let img = Image::new(&[8, 8], PixelId::ComplexFloat32);
        assert!(matches!(
            smoothing_recursive_gaussian(&img, &[1.0, 1.0], false).unwrap_err(),
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(
                PixelId::ComplexFloat32
            ))
        ));
    }

    #[test]
    fn smoothing_recursive_gaussian_wrong_sigma_length_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            smoothing_recursive_gaussian(&img, &[1.0], false),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn smoothing_recursive_gaussian_negative_sigma_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            smoothing_recursive_gaussian(&img, &[-1.0, 1.0], false),
            Err(FilterError::InvalidSigma(_))
        ));
    }

    #[test]
    fn smoothing_recursive_gaussian_short_filtered_axis_is_rejected() {
        let img = Image::new(&[3, 8], PixelId::Float64);
        assert!(matches!(
            smoothing_recursive_gaussian(&img, &[1.0, 1.0], false),
            Err(FilterError::AxisTooShortForRecursion { axis: 0, len: 3 })
        ));
    }
}
