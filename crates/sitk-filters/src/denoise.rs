//! ITK's smoothing / denoising family, verified against
//! `Modules/Filtering/Smoothing/include/` (`itkMeanImageFilter.h`/`.hxx`,
//! `itkMedianImageFilter.h`/`.hxx`, `itkDiscreteGaussianImageFilter.h`/`.hxx`,
//! `itkBinomialBlurImageFilter.h`/`.hxx`), `Core/Common/include/`
//! (`itkGaussianOperator.h`/`.hxx`, the discrete Gaussian's Bessel-function
//! kernel), `Modules/Filtering/ImageFeature/include/itkBilateralImageFilter.h`/
//! `.hxx` (plus `Filtering/ImageSources/include/itkGaussianImageSource.hxx`
//! and `Core/Common/include/itkGaussianSpatialFunction.hxx` for the domain
//! kernel it builds), and `Modules/Filtering/CurvatureFlow/include/`
//! (`itkCurvatureFlowImageFilter.h`/`.hxx`, `itkCurvatureFlowFunction.h`/`.hxx`)
//! plus the shared solver in `Core/FiniteDifference/include/`
//! (`itkFiniteDifferenceImageFilter.hxx`, `itkDenseFiniteDifferenceImageFilter.hxx`).
//!
//! [`mean`] and [`median`] walk a [`NeighborhoodIterator`] with a per-axis
//! `radius` under [`ZeroFluxNeumannBoundaryCondition`] (the boundary every
//! filter in this module uses, matching each ITK class). [`median`] selects
//! via `select_nth_unstable_by` at index `len/2` — ITK's own
//! `std::nth_element` position — which on an even-length window is the
//! *upper* median, never an average of the two middle values; every window
//! here happens to be odd-length (`Π (2·radius[d]+1)`), but [`select_median`]
//! itself is exercised directly against an even-length slice in the tests to
//! prove that convention.
//!
//! [`discrete_gaussian`] convolves a Lindeberg discrete-Gaussian kernel
//! (`GaussianOperator::GenerateCoefficients`'s modified-Bessel-function
//! construction, transcribed operation-for-operation in
//! [`gaussian_operator_kernel`]) separably, one axis at a time, truncated by
//! `maximum_error`/`maximum_kernel_width`.
//!
//! [`binomial_blur`] reproduces `BinomialBlurImageFilter`'s imperative
//! forward+reverse index-walk (not a closed-form convolution) exactly:
//! per repetition, per axis, a forward pass averages each non-last pixel
//! with its `+1` neighbor (both read from the *pre-pass* buffer, since the
//! neighbor read is always strictly later in the walk than any write so
//! far), then a reverse pass averages each non-first pixel with its `-1`
//! neighbor (both read from the forward pass's *output*, for the same
//! reason in reverse). The composition reduces to the standard `[1,2,1]/4`
//! kernel at every interior tap, but the two ends are asymmetric:
//! `new[0] = (old[0]+old[1])/2` and `new[last] = (old[last-1]+3·old[last])/4`
//! — not a zero-flux-Neumann-equivalent reflection.
//!
//! [`bilateral`] builds a normalized ND domain Gaussian (radius auto-sized
//! from `domain_sigma` via ITK's `ceil(2.5·domain_sigma/spacing[d])`) and a
//! quantized range-Gaussian lookup table (`number_of_range_gaussian_samples`
//! buckets over `[0, 4·range_sigma)`, matching ITK's own table — not a
//! continuous `exp()` — since ITK's per-pixel weight really is bucketed by
//! `Math::Floor`). A neighbor only contributes when its range distance is
//! `< 4·range_sigma`; the domain Gaussian's `1/(σ√2π)`-style normalization
//! constants cancel in the final `val/normFactor` ratio (both are constant
//! across the window), so they are omitted rather than reconstructing
//! `GaussianImageSource`'s own internal normalization.
//!
//! [`curvature_flow`] is `DenseFiniteDifferenceImageFilter`'s explicit-Euler
//! solver (`CalculateChange` then `ApplyUpdate`, run from a frozen snapshot
//! each iteration — never in place) specialized to `CurvatureFlowFunction`'s
//! update, `Iₜ = κ|∇I|` discretized as
//! `(Σᵢ (Σⱼ≠ᵢ Iⱼⱼ)·Iᵢ² − 2·Σᵢ<ⱼ Iᵢ·Iⱼ·Iᵢⱼ) / Σᵢ Iᵢ²` (zero when `Σᵢ Iᵢ² < 1e-9`).
//! ITK's own `CurvatureFlowFunction`/`AnisotropicDiffusionFunction` compute
//! *no* stability bound at all — `ComputeGlobalTimeStep` just returns
//! whatever `time_step` the caller set, and the class docs only say the step
//! "should be" small. This module adds its own: linearizing the update
//! around an axis-aligned gradient (where every cross-derivative term
//! vanishes and the scheme becomes an ordinary explicit heat equation over
//! the `dim − 1` axes perpendicular to the gradient, each with unit
//! diffusivity in the `1/spacing[d]²`-scaled grid) gives the standard von
//! Neumann bound `time_step ≤ 1 / (2·Σᵢ scale[i]²)`, `scale[i] = 1/spacing[i]`
//! (or `1` when `use_image_spacing` is off) — enforced here as a caller
//! guard ([`FilterError::UnstableTimeStep`]), not an ITK-sourced constant.
//! Output is always [`PixelId::Float64`] (SimpleITK's
//! `NumericTraits<InputPixelType>::RealType`, which is `double` for every
//! pixel type this crate has *except* `long double`, which none of ours are).

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{
    Image, Neighborhood, NeighborhoodIterator, PixelId, Scalar, ZeroFluxNeumannBoundaryCondition,
    dispatch_scalar,
};

/// An `f64` copy of `img`'s pixels with `img`'s geometry, the working buffer
/// for every filter in this module that computes in `f64` (mirrors
/// `gradient.rs`'s helper of the same name).
fn scratch_f64(img: &Image) -> Result<Image> {
    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec())?;
    scratch.copy_geometry_from(img);
    Ok(scratch)
}

// ---- mean -------------------------------------------------------------

/// `MeanImageFilter`: the box average over a per-axis `radius` neighborhood,
/// accumulated in `f64` (ITK's `InputRealType`, which is `double` for every
/// pixel type here) and narrowed back to `img`'s own pixel type, under
/// [`ZeroFluxNeumannBoundaryCondition`].
///
/// Errors if `radius.len() != img.dimension()`.
pub fn mean(img: &Image, radius: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }

    let scratch = scratch_f64(img)?;
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, radius, ZeroFluxNeumannBoundaryCondition)?;
    let neighborhood_size = iter.len() as f64;

    let out: Vec<f64> = iter
        .map(|(_, nb)| nb.values().iter().sum::<f64>() / neighborhood_size)
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- median -------------------------------------------------------------

/// `std::nth_element(values.begin(), values.begin()+values.len()/2,
/// values.end())`'s selected element: the value that would sit at index
/// `len/2` in sorted order. On an even-length slice this is the *upper*
/// median (never an average of the two middle values) — `itkMedianImageFilter.hxx`'s
/// literal indexing, `neighborhoodSize / 2`.
fn select_median<T: Copy + PartialOrd>(values: &mut [T]) -> T {
    let mid = values.len() / 2;
    let (_, &mut median, _) = values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    median
}

fn median_typed<T: Scalar>(img: &Image, radius: &[usize]) -> Result<Image> {
    let iter = NeighborhoodIterator::<T, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)?;
    let mut out = Vec::with_capacity(img.number_of_pixels());
    for (_, nb) in iter {
        let mut values = nb.values().to_vec();
        out.push(select_median(&mut values));
    }
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `MedianImageFilter`: the [`select_median`] of a per-axis `radius`
/// neighborhood, selected directly in `img`'s own pixel type (never rounded
/// through `f64`), under [`ZeroFluxNeumannBoundaryCondition`].
///
/// Errors if `radius.len() != img.dimension()`.
pub fn median(img: &Image, radius: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }
    dispatch_scalar!(img.pixel_id(), median_typed, img, radius)
}

// ---- discrete_gaussian ----------------------------------------------------

/// `GaussianOperator::ModifiedBesselI0`: the modified Bessel function `I₀(y)`,
/// via Abramowitz & Stegun's rational-polynomial approximation.
fn modified_bessel_i0(y: f64) -> f64 {
    let d = y.abs();
    if d < 3.75 {
        let mut m = y / 3.75;
        m *= m;
        1.0 + m
            * (3.5156229
                + m * (3.0899424
                    + m * (1.2067492 + m * (0.2659732 + m * (0.360768e-1 + m * 0.45813e-2)))))
    } else {
        let m = 3.75 / d;
        (d.exp() / d.sqrt())
            * (0.39894228
                + m * (0.1328592e-1
                    + m * (0.225319e-2
                        + m * (-0.157565e-2
                            + m * (0.916281e-2
                                + m * (-0.2057706e-1
                                    + m * (0.2635537e-1
                                        + m * (-0.1647633e-1 + m * 0.392377e-2))))))))
    }
}

/// `GaussianOperator::ModifiedBesselI1`: the modified Bessel function `I₁(y)`.
fn modified_bessel_i1(y: f64) -> f64 {
    let d = y.abs();
    let accumulator = if d < 3.75 {
        let mut m = y / 3.75;
        m *= m;
        d * (0.5
            + m * (0.87890594
                + m * (0.51498869
                    + m * (0.15084934 + m * (0.2658733e-1 + m * (0.301532e-2 + m * 0.32411e-3))))))
    } else {
        let m = 3.75 / d;
        let mut acc = 0.2282967e-1 + m * (-0.2895312e-1 + m * (0.1787654e-1 - m * 0.420059e-2));
        acc = 0.39894228
            + m * (-0.3988024e-1
                + m * (-0.362018e-2 + m * (0.163801e-2 + m * (-0.1031555e-1 + m * acc))));
        acc * (d.exp() / d.sqrt())
    };
    if y < 0.0 { -accumulator } else { accumulator }
}

/// `GaussianOperator::ModifiedBesselI`: the modified Bessel function `Iₙ(y)`
/// for `n >= 2`, via Numerical Recipes' downward recurrence.
fn modified_bessel_i(n: i32, y: f64) -> f64 {
    debug_assert!(n >= 2, "modified_bessel_i is only valid for n >= 2");
    const ACCURACY: f64 = 40.0;
    if y == 0.0 {
        return 0.0;
    }

    let toy = 2.0 / y.abs();
    let mut qip = 0.0f64;
    let mut accumulator = 0.0f64;
    let mut qi = 1.0f64;

    let mut j = 2 * (n + (ACCURACY * n as f64).sqrt() as i32);
    while j > 0 {
        let qim = qip + j as f64 * toy * qi;
        qip = qi;
        qi = qim;
        if qi.abs() > 1.0e10 {
            accumulator *= 1.0e-10;
            qi *= 1.0e-10;
            qip *= 1.0e-10;
        }
        if j == n {
            accumulator = qip;
        }
        j -= 1;
    }

    accumulator *= modified_bessel_i0(y) / qi;

    if y < 0.0 && (n & 1) != 0 {
        -accumulator
    } else {
        accumulator
    }
}

/// `GaussianOperator::GenerateCoefficients`: the symmetric discrete-Gaussian
/// kernel for index-space `variance >= 0`, truncated once the two-sided area
/// under the tail coefficients reaches `1 - maximum_error` (or the kernel
/// hits `maximum_kernel_width` taps first), then normalized to sum to
/// exactly `1` regardless of which truncation fired. Returns a
/// `2·radius + 1`-length kernel, `radius = tail_count`; `variance == 0.0`
/// still yields the 3-tap identity kernel `[0.0, 1.0, 0.0]` (`radius == 1`),
/// never a literal `radius == 0`.
fn gaussian_operator_kernel(
    variance: f64,
    maximum_error: f64,
    maximum_kernel_width: u32,
) -> Vec<f64> {
    let et = (-variance).exp();

    let mut c = vec![et * modified_bessel_i0(variance)];
    let mut sum = c[0];
    c.push(et * modified_bessel_i1(variance));
    sum += c[1] * 2.0;

    let cap = 1.0 - maximum_error;
    let mut i: i32 = 2;
    while sum < cap {
        let v = et * modified_bessel_i(i, variance);
        c.push(v);
        sum += v * 2.0;
        if v <= 0.0 {
            break;
        }
        if c.len() as u32 > maximum_kernel_width {
            break;
        }
        i += 1;
    }
    for v in &mut c {
        *v /= sum;
    }

    let radius = c.len() - 1;
    let mut kernel = vec![0.0f64; 2 * radius + 1];
    for (k, &ck) in c.iter().enumerate() {
        kernel[radius + k] = ck;
        kernel[radius - k] = ck;
    }
    kernel
}

/// `DiscreteGaussianImageFilter`: separable convolution with
/// [`gaussian_operator_kernel`], one axis at a time, under
/// [`ZeroFluxNeumannBoundaryCondition`]. `variance` and `maximum_error` are
/// per axis (`ArrayType` in ITK); when `use_image_spacing`, each axis's
/// variance is converted from physical to index units via `variance[d] /
/// spacing[d]²` (`GetKernelVarianceArray`) before the kernel is built.
/// Output keeps `img`'s pixel type.
///
/// Errors if `variance.len()` or `maximum_error.len() != img.dimension()`,
/// any `variance` is negative, or any `maximum_error` is outside the open
/// interval `(0.0, 1.0)` (`GaussianOperator::SetMaximumError`'s own bound).
pub fn discrete_gaussian(
    img: &Image,
    variance: &[f64],
    maximum_error: &[f64],
    maximum_kernel_width: u32,
    use_image_spacing: bool,
) -> Result<Image> {
    let smoothed = discrete_gaussian_f64(
        img,
        variance,
        maximum_error,
        maximum_kernel_width,
        use_image_spacing,
    )?;
    image_from_f64(img.pixel_id(), img.size(), img, &smoothed.to_f64_vec())
}

/// [`discrete_gaussian`]'s validation and separable-convolution core, stopping
/// before the narrow-back-to-input-type step: returns the always-`f64`
/// intermediate image. `canny_edge_detection`'s smoothing stage
/// (`CannyEdgeDetectionImageFilter`'s `m_GaussianFilter`) consumes this
/// directly — its derivative stages need the full `f64` field, not a
/// round-trip through the input pixel type.
pub(crate) fn discrete_gaussian_f64(
    img: &Image,
    variance: &[f64],
    maximum_error: &[f64],
    maximum_kernel_width: u32,
    use_image_spacing: bool,
) -> Result<Image> {
    let dim = img.dimension();
    if variance.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: variance.len(),
        });
    }
    if maximum_error.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: maximum_error.len(),
        });
    }
    if variance.iter().any(|&v| v < 0.0) {
        return Err(FilterError::InvalidVariance(variance.to_vec()));
    }
    if maximum_error.iter().any(|&e| !(e > 0.0 && e < 1.0)) {
        return Err(FilterError::InvalidMaximumError(maximum_error.to_vec()));
    }

    let spacing = img.spacing().to_vec();
    let size = img.size().to_vec();
    let mut current = scratch_f64(img)?;

    for d in 0..dim {
        let adjusted_variance = if use_image_spacing {
            variance[d] / (spacing[d] * spacing[d])
        } else {
            variance[d]
        };
        let kernel =
            gaussian_operator_kernel(adjusted_variance, maximum_error[d], maximum_kernel_width);
        let half = kernel.len() / 2;
        let mut radius = vec![0usize; dim];
        radius[d] = half;

        let iter = NeighborhoodIterator::<f64, _>::new(
            &current,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )?;
        let out: Vec<f64> = iter
            .map(|(_, nb)| {
                let mut off = vec![0i64; dim];
                kernel
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| {
                        off[d] = k as i64 - half as i64;
                        c * nb.get(&off)
                    })
                    .sum()
            })
            .collect();

        current = Image::from_vec(&size, out)?;
        current.copy_geometry_from(img);
    }

    Ok(current)
}

// ---- binomial_blur ----------------------------------------------------

/// One repetition's forward+reverse pass along axis `d`
/// (`BinomialBlurImageFilter::GenerateData`'s per-dimension inner loop,
/// traced index-for-index — see the module doc comment). Both passes are
/// safe to compute non-sequentially from their respective input snapshots:
/// every read in the real imperative walk is always strictly *later* in
/// raster order than any write performed so far in that same pass.
fn blur_axis_forward_reverse(buf: &[f64], size: &[usize], strides: &[usize], d: usize) -> Vec<f64> {
    let stride = strides[d];
    let size_d = size[d];
    if size_d <= 1 {
        return buf.to_vec();
    }

    let mut fwd = buf.to_vec();
    for (p, &v) in buf.iter().enumerate() {
        let coord_d = (p / stride) % size_d;
        if coord_d < size_d - 1 {
            fwd[p] = (v + buf[p + stride]) / 2.0;
        }
    }

    let mut out = fwd.clone();
    for (p, &v) in fwd.iter().enumerate() {
        let coord_d = (p / stride) % size_d;
        if coord_d > 0 {
            out[p] = (v + fwd[p - stride]) / 2.0;
        }
    }
    out
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `BinomialBlurImageFilter`: `repetitions` rounds of a forward+reverse
/// averaging pass over every axis in turn (axis inner, repetition outer —
/// matching ITK's loop nesting, since each repetition's blur depends on the
/// previous one's full output). Has no spacing awareness in ITK (no
/// `GetSpacing` call anywhere in `GenerateData`) and none is added here.
/// Output keeps `img`'s pixel type.
pub fn binomial_blur(img: &Image, repetitions: u32) -> Result<Image> {
    let dim = img.dimension();
    let size = img.size().to_vec();
    let strides_ = strides(&size);
    let mut buf = img.to_f64_vec();

    for _ in 0..repetitions {
        for d in 0..dim {
            buf = blur_axis_forward_reverse(&buf, &size, &strides_, d);
        }
    }

    image_from_f64(img.pixel_id(), &size, img, &buf)
}

// ---- bilateral ----------------------------------------------------

/// Per-offset ND coordinates for a `radius`-sized window, dimension-0-fastest
/// (matches `NeighborhoodIterator::new`'s own internal offset table).
fn window_offsets(radius: &[usize]) -> Vec<Vec<i64>> {
    let dim = radius.len();
    let n: usize = radius.iter().map(|&r| 2 * r + 1).product();
    let mut offsets = Vec::with_capacity(n);
    let mut offset: Vec<i64> = radius.iter().map(|&r| -(r as i64)).collect();
    for _ in 0..n {
        offsets.push(offset.clone());
        for d in 0..dim {
            offset[d] += 1;
            if offset[d] > radius[d] as i64 {
                offset[d] = -(radius[d] as i64);
            } else {
                break;
            }
        }
    }
    offsets
}

/// `BilateralImageFilter`: domain (spatial) x range (intensity) weighted
/// average. `domain_sigma` is isotropic in physical space (ITK's
/// `SetDomainSigma(double)` convenience setter, the only form SimpleITK
/// wraps procedurally); the window radius is auto-sized per axis from it,
/// `ceil(2.5·domain_sigma / spacing[d])` (`m_DomainMu = 2.5`, ITK's
/// unexposed constant). `range_sigma` is in intensity units; a neighbor only
/// contributes when `|neighbor - center| < 4·range_sigma` (`m_RangeMu =
/// 4.0`), and its range weight is looked up from a `number_of_range_gaussian_samples`-bucket
/// table over `[0, 4·range_sigma)` rather than evaluated continuously — see
/// the module doc comment. Boundary is [`ZeroFluxNeumannBoundaryCondition`].
/// Output keeps `img`'s pixel type.
pub fn bilateral(
    img: &Image,
    domain_sigma: f64,
    range_sigma: f64,
    number_of_range_gaussian_samples: u32,
) -> Result<Image> {
    const DOMAIN_MU: f64 = 2.5;
    const RANGE_MU: f64 = 4.0;

    let dim = img.dimension();
    let spacing = img.spacing().to_vec();

    let radius: Vec<usize> = (0..dim)
        .map(|d| (DOMAIN_MU * domain_sigma / spacing[d]).ceil().max(0.0) as usize)
        .collect();

    let offsets = window_offsets(&radius);
    let mut domain_kernel: Vec<f64> = offsets
        .iter()
        .map(|off| {
            let exponent: f64 = off
                .iter()
                .zip(&spacing)
                .map(|(&o, &s)| {
                    let physical = o as f64 * s;
                    physical * physical
                })
                .sum();
            (-0.5 * exponent / (domain_sigma * domain_sigma)).exp()
        })
        .collect();
    let domain_norm: f64 = domain_kernel.iter().sum();
    for w in &mut domain_kernel {
        *w /= domain_norm;
    }

    // ITK indexes `Math::Floor<SizeValueType>(tableArg)` unchecked; clamping
    // here only guards against the same floating-point edge (`tableArg`
    // reaching `samples` when `range_distance` is a hair under the `<`
    // threshold) that would be an out-of-bounds read in the original.
    let samples = number_of_range_gaussian_samples.max(1) as usize;
    let dynamic_range_used = RANGE_MU * range_sigma;
    let range_variance = range_sigma * range_sigma;
    let range_gaussian_denom = range_sigma * (2.0 * std::f64::consts::PI).sqrt();
    let table_delta = dynamic_range_used / samples as f64;
    let table: Vec<f64> = (0..samples)
        .map(|i| {
            let v = i as f64 * table_delta;
            (-0.5 * v * v / range_variance).exp() / range_gaussian_denom
        })
        .collect();
    let distance_to_table_index = samples as f64 / dynamic_range_used;

    let scratch = scratch_f64(img)?;
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let out: Vec<f64> = iter
        .map(|(_, nb)| {
            let center = nb.center_value();
            let mut val = 0.0f64;
            let mut norm_factor = 0.0f64;
            for (off, &dk) in offsets.iter().zip(&domain_kernel) {
                let pixel = nb.get(off);
                let range_distance = (pixel - center).abs();
                if range_distance < dynamic_range_used {
                    let table_arg = range_distance * distance_to_table_index;
                    let idx = (table_arg.floor() as usize).min(samples - 1);
                    let product = dk * table[idx];
                    norm_factor += product;
                    val += pixel * product;
                }
            }
            val / norm_factor
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- curvature_flow ----------------------------------------------------

/// `CurvatureFlowFunction::ComputeUpdate`: the discretized `κ|∇I|` update at
/// one pixel — `(Σᵢ (Σⱼ≠ᵢ secderiv[j])·firstderiv[i]² − 2·Σᵢ<ⱼ
/// firstderiv[i]·firstderiv[j]·crossderiv[i][j]) / Σᵢ firstderiv[i]²`, zero
/// when the gradient magnitude squared is below `1e-9` (ITK's own
/// threshold). `scale[d]` is `ComputeNeighborhoodScales`'s per-axis factor,
/// `ScaleCoefficients[d] / radius[d]` — `ScaleCoefficients[d]` is `1/spacing[d]`
/// when using image spacing, else `1`, and `CurvatureFlowFunction`'s own radius
/// is always `1`, so for [`curvature_flow`] it is exactly `ScaleCoefficients[d]`.
/// `MinMaxCurvatureFlowFunction` widens the radius to its stencil radius `r`
/// and therefore passes `ScaleCoefficients[d] / r` — see
/// [`crate::min_max_curvature_flow`], which reuses this update unchanged.
///
/// `nb` may carry any radius `>= 1` per axis; only the `±1` offsets are read.
pub(crate) fn curvature_flow_update(nb: &Neighborhood<f64>, dim: usize, scale: &[f64]) -> f64 {
    let center = nb.center_value();
    let mut first = vec![0.0f64; dim];
    let mut second = vec![0.0f64; dim];
    let mut cross = vec![vec![0.0f64; dim]; dim];
    let mut magnitude_sqr = 0.0f64;
    let mut off = vec![0i64; dim];

    for i in 0..dim {
        off[i] = 1;
        let plus = nb.get(&off);
        off[i] = -1;
        let minus = nb.get(&off);
        off[i] = 0;

        first[i] = 0.5 * (plus - minus) * scale[i];
        second[i] = (plus - 2.0 * center + minus) * scale[i] * scale[i];
        magnitude_sqr += first[i] * first[i];

        for j in (i + 1)..dim {
            off[i] = -1;
            off[j] = -1;
            let mm = nb.get(&off);
            off[j] = 1;
            let mp = nb.get(&off);
            off[i] = 1;
            let pp = nb.get(&off);
            off[j] = -1;
            let pm = nb.get(&off);
            off[i] = 0;
            off[j] = 0;
            cross[i][j] = 0.25 * (mm - mp - pm + pp) * scale[i] * scale[j];
        }
    }

    if magnitude_sqr < 1e-9 {
        return 0.0;
    }

    let mut update = 0.0f64;
    for (i, &fi) in first.iter().enumerate() {
        let temp: f64 = (0..dim).filter(|&j| j != i).map(|j| second[j]).sum();
        update += temp * fi * fi;
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            update -= 2.0 * first[i] * first[j] * cross[i][j];
        }
    }
    update / magnitude_sqr
}

/// `CurvatureFlowImageFilter`: `number_of_iterations` rounds of explicit
/// Euler, `I ← I + time_step · κ|∇I|`, each computed from a frozen snapshot
/// of the *previous* iteration's output (`CalculateChange` then
/// `ApplyUpdate` — never updated in place within an iteration), under
/// [`ZeroFluxNeumannBoundaryCondition`]. `time_step` must lie in `[0,
/// max_stable]`, `max_stable = 1 / (2·Σᵢ scale[i]²)` — see the module doc
/// comment for the derivation (this bound is added by this crate; ITK's own
/// `CurvatureFlowFunction` enforces none). Output is always
/// [`PixelId::Float64`].
///
/// Errors if `time_step` is outside `[0, max_stable]`.
pub fn curvature_flow(
    img: &Image,
    number_of_iterations: u32,
    time_step: f64,
    use_image_spacing: bool,
) -> Result<Image> {
    let dim = img.dimension();
    let spacing = img.spacing().to_vec();
    let scale: Vec<f64> = (0..dim)
        .map(|d| {
            if use_image_spacing {
                1.0 / spacing[d]
            } else {
                1.0
            }
        })
        .collect();

    let max_stable = 1.0 / (2.0 * scale.iter().map(|s| s * s).sum::<f64>());
    if !(0.0..=max_stable).contains(&time_step) {
        return Err(FilterError::UnstableTimeStep {
            time_step,
            max_stable,
        });
    }

    let size = img.size().to_vec();
    let radius = vec![1usize; dim];
    let mut buf = img.to_f64_vec();

    for _ in 0..number_of_iterations {
        let mut snapshot = Image::from_vec(&size, buf.clone())?;
        snapshot.copy_geometry_from(img);
        let iter = NeighborhoodIterator::<f64, _>::new(
            &snapshot,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )?;
        for ((_, nb), v) in iter.zip(buf.iter_mut()) {
            *v += time_step * curvature_flow_update(&nb, dim, &scale);
        }
    }

    image_from_f64(PixelId::Float64, &size, img, &buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- mean ----

    #[test]
    fn mean_radius_zero_is_identity() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = mean(&img, &[0, 0]).unwrap();
        assert_eq!(out.to_f64_vec(), img.to_f64_vec());
    }

    #[test]
    fn mean_constant_image_is_fixed_point() {
        let img = Image::from_vec(&[6, 6], vec![7.0f64; 36]).unwrap();
        let out = mean(&img, &[1, 1]).unwrap();
        assert!(out.to_f64_vec().iter().all(|&v| (v - 7.0).abs() < 1e-12));
    }

    #[test]
    fn mean_single_impulse_spreads_to_exactly_the_kernel() {
        let n = 5;
        let mut data = vec![0.0f64; n * n];
        data[2 * n + 2] = 90.0;
        let img = Image::from_vec(&[n, n], data).unwrap();
        let out = mean(&img, &[1, 1]).unwrap();
        let vals = out.to_f64_vec();
        for y in 0..n {
            for x in 0..n {
                let expected = if x.abs_diff(2) <= 1 && y.abs_diff(2) <= 1 {
                    10.0
                } else {
                    0.0
                };
                assert!(
                    (vals[y * n + x] - expected).abs() < 1e-12,
                    "at ({x},{y}): got {}, expected {expected}",
                    vals[y * n + x]
                );
            }
        }
    }

    #[test]
    fn mean_per_axis_radius_blurs_only_the_nonzero_axis() {
        let n = 5;
        let mut data = vec![0.0f64; n * n];
        data[2 * n + 2] = 90.0;
        let img = Image::from_vec(&[n, n], data).unwrap();
        let out = mean(&img, &[1, 0]).unwrap();
        let vals = out.to_f64_vec();
        // radius=[1,0]: window is 3x1, size 3, spreads only along x.
        for x in 1..=3 {
            assert!((vals[2 * n + x] - 30.0).abs() < 1e-12);
        }
        assert!(vals[n + 2].abs() < 1e-12);
        assert!(vals[3 * n + 2].abs() < 1e-12);
    }

    #[test]
    fn mean_wrong_radius_length_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            mean(&img, &[1]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    // ---- median ----

    #[test]
    fn select_median_even_length_is_upper_median_not_average() {
        // nth_element at index len/2 = 2 on [1,2,3,4]: sorted position 2 is
        // the value 3, NOT the average of the two middle values (2.5).
        let mut v = [4, 1, 3, 2];
        assert_eq!(select_median(&mut v), 3);
    }

    #[test]
    fn median_radius_zero_is_identity() {
        let img = Image::from_vec(&[4, 3], (0u8..12).collect()).unwrap();
        let out = median(&img, &[0, 0]).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            img.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn median_constant_image_is_fixed_point() {
        let img = Image::from_vec(&[5, 5], vec![9u8; 25]).unwrap();
        let out = median(&img, &[1, 1]).unwrap();
        assert!(out.scalar_slice::<u8>().unwrap().iter().all(|&v| v == 9));
    }

    #[test]
    fn median_removes_a_lone_salt_and_pepper_pixel() {
        let img = Image::from_vec(&[7, 1], vec![5u8, 5, 5, 99, 5, 5, 5]).unwrap();
        let out = median(&img, &[1, 0]).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[5, 5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn median_leaves_a_step_edge_intact() {
        let img = Image::from_vec(&[6, 1], vec![0u8, 0, 0, 5, 5, 5]).unwrap();
        let out = median(&img, &[1, 0]).unwrap();
        // every 3-window at/around the step already has a 2-1 majority equal
        // to the input's own value there, so the whole step is a fixed point.
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            img.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn median_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        let out = median(&img, &[1, 1]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    // ---- discrete_gaussian ----

    #[test]
    fn gaussian_operator_kernel_zero_variance_is_identity() {
        let kernel = gaussian_operator_kernel(0.0, 0.01, 32);
        assert_eq!(kernel, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn gaussian_operator_kernel_is_symmetric_and_normalized() {
        let kernel = gaussian_operator_kernel(4.0, 0.01, 32);
        let radius = kernel.len() / 2;
        for k in 0..=radius {
            assert!((kernel[radius - k] - kernel[radius + k]).abs() < 1e-15);
        }
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "kernel sum {sum}");
    }

    #[test]
    fn discrete_gaussian_variance_zero_is_identity() {
        let img = Image::from_vec(&[6, 6], (0..36).map(|v| v as f64).collect()).unwrap();
        let out = discrete_gaussian(&img, &[0.0, 0.0], &[0.01, 0.01], 32, true).unwrap();
        let vals = out.to_f64_vec();
        for (a, b) in vals.iter().zip(img.to_f64_vec()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn discrete_gaussian_constant_image_is_preserved() {
        let img = Image::from_vec(&[12, 12], vec![3.0f64; 144]).unwrap();
        let out = discrete_gaussian(&img, &[4.0, 4.0], &[0.01, 0.01], 32, true).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 3.0).abs() < 1e-9, "constant not preserved: {v}");
        }
    }

    #[test]
    fn discrete_gaussian_constant_image_is_preserved_under_truncation() {
        // A tiny maximum_kernel_width forces early truncation; the kernel
        // still normalizes to sum 1 regardless of why it stopped growing.
        let img = Image::from_vec(&[12, 12], vec![5.0f64; 144]).unwrap();
        let out = discrete_gaussian(&img, &[10.0, 10.0], &[0.01, 0.01], 3, true).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 5.0).abs() < 1e-9, "constant not preserved: {v}");
        }
    }

    #[test]
    fn discrete_gaussian_anisotropic_spacing_changes_blur_amount() {
        let n = 25;
        let mut data = vec![0.0f64; n * n];
        data[12 * n + 12] = 100.0;
        let mut fine = Image::from_vec(&[n, n], data.clone()).unwrap();
        fine.set_spacing(&[1.0, 1.0]).unwrap();
        let mut coarse = Image::from_vec(&[n, n], data).unwrap();
        coarse.set_spacing(&[2.0, 2.0]).unwrap();

        let peak_fine = discrete_gaussian(&fine, &[4.0, 4.0], &[0.01, 0.01], 32, true)
            .unwrap()
            .to_f64_vec()[12 * n + 12];
        let peak_coarse = discrete_gaussian(&coarse, &[4.0, 4.0], &[0.01, 0.01], 32, true)
            .unwrap()
            .to_f64_vec()[12 * n + 12];
        assert!(
            peak_coarse > peak_fine,
            "coarser spacing should blur less: {peak_coarse} vs {peak_fine}"
        );
    }

    #[test]
    fn discrete_gaussian_use_image_spacing_false_ignores_spacing() {
        let n = 25;
        let mut data = vec![0.0f64; n * n];
        data[12 * n + 12] = 100.0;
        let mut img = Image::from_vec(&[n, n], data.clone()).unwrap();
        img.set_spacing(&[3.0, 3.0]).unwrap();
        let mut unit = Image::from_vec(&[n, n], data).unwrap();
        unit.set_spacing(&[1.0, 1.0]).unwrap();

        let peak_ignored = discrete_gaussian(&img, &[4.0, 4.0], &[0.01, 0.01], 32, false)
            .unwrap()
            .to_f64_vec()[12 * n + 12];
        let peak_unit = discrete_gaussian(&unit, &[4.0, 4.0], &[0.01, 0.01], 32, true)
            .unwrap()
            .to_f64_vec()[12 * n + 12];
        assert!((peak_ignored - peak_unit).abs() < 1e-9);
    }

    #[test]
    fn discrete_gaussian_negative_variance_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            discrete_gaussian(&img, &[-1.0, 1.0], &[0.01, 0.01], 32, true),
            Err(FilterError::InvalidVariance(_))
        ));
    }

    #[test]
    fn discrete_gaussian_maximum_error_out_of_range_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            discrete_gaussian(&img, &[1.0, 1.0], &[1.0, 0.01], 32, true),
            Err(FilterError::InvalidMaximumError(_))
        ));
        assert!(matches!(
            discrete_gaussian(&img, &[1.0, 1.0], &[0.0, 0.01], 32, true),
            Err(FilterError::InvalidMaximumError(_))
        ));
    }

    #[test]
    fn discrete_gaussian_wrong_length_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            discrete_gaussian(&img, &[1.0], &[0.01, 0.01], 32, true),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            discrete_gaussian(&img, &[1.0, 1.0], &[0.01], 32, true),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn discrete_gaussian_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[5, 5], vec![1u8; 25]).unwrap();
        let out = discrete_gaussian(&img, &[1.0, 1.0], &[0.01, 0.01], 32, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    // ---- binomial_blur ----

    #[test]
    fn binomial_blur_zero_repetitions_is_identity() {
        let img = Image::from_vec(&[5, 4], (0..20).map(|v| v as f64).collect()).unwrap();
        let out = binomial_blur(&img, 0).unwrap();
        assert_eq!(out.to_f64_vec(), img.to_f64_vec());
    }

    #[test]
    fn binomial_blur_constant_image_is_fixed_point() {
        let img = Image::from_vec(&[6, 6, 6], vec![4.0f64; 216]).unwrap();
        let out = binomial_blur(&img, 3).unwrap();
        assert!(out.to_f64_vec().iter().all(|&v| (v - 4.0).abs() < 1e-9));
    }

    #[test]
    fn binomial_blur_one_repetition_matches_hand_derived_1_2_1_and_boundary() {
        let img = Image::from_vec(&[6, 1], vec![0.0f64, 10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = binomial_blur(&img, 1).unwrap();
        let vals = out.to_f64_vec();
        // left boundary: (o[0]+o[1])/2; interior: (o[i-1]+2o[i]+o[i+1])/4;
        // right boundary: (o[last-1]+3*o[last])/4.
        let expected = [5.0, 10.0, 20.0, 30.0, 40.0, 47.5];
        for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-12, "at {i}: got {v}, expected {e}");
        }
    }

    #[test]
    fn binomial_blur_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[4, 4], vec![1u16; 16]).unwrap();
        let out = binomial_blur(&img, 2).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt16);
    }

    // ---- bilateral ----

    #[test]
    fn bilateral_radius_zero_is_identity() {
        // domain_sigma < 0 gives ceil(2.5*domain_sigma/spacing) <= 0, so the
        // window auto-sizes to a single tap (the center only); at that tap
        // the numerator (position - mean) is exactly 0 regardless of sigma's
        // sign, so this is well-defined (unlike domain_sigma == 0.0, which
        // would make ITK's own domain-Gaussian evaluate 0/0 at that tap).
        let img = Image::from_vec(&[5, 5], (0..25).map(|v| v as f64).collect()).unwrap();
        let out = bilateral(&img, -0.1, 50.0, 100).unwrap();
        let vals = out.to_f64_vec();
        for (a, b) in vals.iter().zip(img.to_f64_vec()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn bilateral_constant_image_is_fixed_point() {
        let img = Image::from_vec(&[9, 9], vec![42.0f64; 81]).unwrap();
        let out = bilateral(&img, 2.0, 30.0, 100).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 42.0).abs() < 1e-9, "constant not preserved: {v}");
        }
    }

    #[test]
    fn bilateral_preserves_a_sharp_step_better_than_domain_only_average() {
        // A small range_sigma should heavily suppress cross-edge blending.
        let n = 11;
        let mut data = vec![0.0f64; n];
        for (x, v) in data.iter_mut().enumerate() {
            *v = if x < n / 2 { 0.0 } else { 100.0 };
        }
        let img = Image::from_vec(&[n, 1], data).unwrap();
        let out = bilateral(&img, 3.0, 1.0, 100).unwrap();
        let vals = out.to_f64_vec();
        // just past the edge, output should stay near the local step value,
        // not be pulled toward the 50 a pure domain-Gaussian average gives.
        assert!(vals[n / 2] > 90.0, "edge not preserved: {}", vals[n / 2]);
        assert!(
            vals[n / 2 - 1] < 10.0,
            "edge not preserved: {}",
            vals[n / 2 - 1]
        );
    }

    #[test]
    fn bilateral_anisotropic_spacing_changes_effective_domain_radius() {
        let n = 25;
        let mut data = vec![0.0f64; n * n];
        data[12 * n + 12] = 100.0;
        let mut fine = Image::from_vec(&[n, n], data.clone()).unwrap();
        fine.set_spacing(&[1.0, 1.0]).unwrap();
        let mut coarse = Image::from_vec(&[n, n], data).unwrap();
        coarse.set_spacing(&[2.0, 2.0]).unwrap();

        let peak_fine = bilateral(&fine, 4.0, 1000.0, 100).unwrap().to_f64_vec()[12 * n + 12];
        let peak_coarse = bilateral(&coarse, 4.0, 1000.0, 100).unwrap().to_f64_vec()[12 * n + 12];
        assert!(
            peak_coarse > peak_fine,
            "coarser spacing should blur less: {peak_coarse} vs {peak_fine}"
        );
    }

    #[test]
    fn bilateral_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[5, 5], vec![10u8; 25]).unwrap();
        let out = bilateral(&img, 1.0, 20.0, 100).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    // ---- curvature_flow ----

    #[test]
    fn curvature_flow_zero_iterations_is_identity_cast_to_f64() {
        let img = Image::from_vec(&[5, 5], (0u8..25).collect()).unwrap();
        let out = curvature_flow(&img, 0, 0.05, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        let expected: Vec<f64> = img.to_f64_vec();
        assert_eq!(out.to_f64_vec(), expected);
    }

    #[test]
    fn curvature_flow_time_step_zero_is_identity_regardless_of_iterations() {
        let img = Image::from_vec(&[6, 6], (0u8..36).collect()).unwrap();
        let out = curvature_flow(&img, 10, 0.0, true).unwrap();
        let expected: Vec<f64> = img.to_f64_vec();
        assert_eq!(out.to_f64_vec(), expected);
    }

    #[test]
    fn curvature_flow_constant_image_is_fixed_point() {
        // Zero gradient everywhere => magnitude_sqr < 1e-9 => update == 0.
        let img = Image::from_vec(&[6, 6], vec![7.0f64; 36]).unwrap();
        let out = curvature_flow(&img, 5, 0.05, true).unwrap();
        assert!(out.to_f64_vec().iter().all(|&v| (v - 7.0).abs() < 1e-12));
    }

    #[test]
    fn curvature_flow_pattern_constant_along_one_axis_is_a_fixed_point() {
        // A straight level line (value depends on x only) has zero curvature
        // everywhere: secderiv along the constant axis is exactly 0, so
        // every update term vanishes regardless of the x-profile's own shape.
        let (w, h) = (7usize, 7usize);
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = ((x as f64) - 3.0).powi(2) + 5.0 * (x as f64).sin();
            }
        }
        let img = Image::from_vec(&[w, h], data.clone()).unwrap();
        let out = curvature_flow(&img, 4, 0.05, true).unwrap();
        for (a, b) in out.to_f64_vec().iter().zip(&data) {
            assert!((a - b).abs() < 1e-9, "expected fixed point: {a} vs {b}");
        }
    }

    #[test]
    fn curvature_flow_bilinear_matches_hand_derived_update() {
        // f(x,y) = x*y on a 3x3 grid: secderiv_x = secderiv_y = 0 everywhere,
        // crossderiv_xy = 1 (index units) at the center, so the whole update
        // reduces to the cross term: -2*scale_x*scale_y*(scale_x*scale_y) /
        // (scale_x^2+scale_y^2).
        let data = vec![0.0f64, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0];
        let mut img_spaced = Image::from_vec(&[3, 3], data.clone()).unwrap();
        img_spaced.set_spacing(&[2.0, 1.0]).unwrap();
        let img_unit = Image::from_vec(&[3, 3], data).unwrap();

        let time_step = 0.01;
        let out_spaced = curvature_flow(&img_spaced, 1, time_step, true).unwrap();
        let out_unit = curvature_flow(&img_unit, 1, time_step, false).unwrap();

        // spacing=[2,1] => scale=[0.5,1.0]: update = -2*0.25*1/(0.25+1) = -0.4
        let expected_spaced = 1.0 + time_step * -0.4;
        // unit scale=[1,1]: update = -2*1*1/(1+1) = -1.0
        let expected_unit = 1.0 - time_step;

        assert!((out_spaced.to_f64_vec()[4] - expected_spaced).abs() < 1e-9);
        assert!((out_unit.to_f64_vec()[4] - expected_unit).abs() < 1e-9);
    }

    #[test]
    fn curvature_flow_unstable_time_step_is_rejected() {
        let img = Image::from_vec(&[5, 5], vec![1.0f64; 25]).unwrap();
        // unit spacing, 2D: max_stable = 1/(2*(1+1)) = 0.25.
        let err = curvature_flow(&img, 1, 0.3, true).unwrap_err();
        match err {
            FilterError::UnstableTimeStep {
                time_step,
                max_stable,
            } => {
                assert_eq!(time_step, 0.3);
                assert!((max_stable - 0.25).abs() < 1e-12);
            }
            other => panic!("expected UnstableTimeStep, got {other:?}"),
        }
    }

    #[test]
    fn curvature_flow_negative_time_step_is_rejected() {
        let img = Image::from_vec(&[5, 5], vec![1.0f64; 25]).unwrap();
        assert!(matches!(
            curvature_flow(&img, 1, -0.01, true),
            Err(FilterError::UnstableTimeStep { .. })
        ));
    }

    #[test]
    fn curvature_flow_output_is_always_float64() {
        let img = Image::from_vec(&[5, 5], vec![1.0f32; 25]).unwrap();
        let out = curvature_flow(&img, 1, 0.05, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
    }
}
