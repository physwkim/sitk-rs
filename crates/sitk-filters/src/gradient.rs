//! Gradient / edge-detection filters, porting ITK's derivative-operator and
//! recursive-Gaussian gradient family: `itkGradientMagnitudeImageFilter.h`,
//! `itkDerivativeImageFilter.h` (+ `itkDerivativeOperator.h`),
//! `itkLaplacianImageFilter.h` (+ `itkLaplacianOperator.h`),
//! `itkSobelEdgeDetectionImageFilter.h` (+ `itkSobelOperator.h`),
//! `itkGradientMagnitudeRecursiveGaussianImageFilter.h`,
//! `itkLaplacianRecursiveGaussianImageFilter.h`, `itkGradientImageFilter.h`,
//! and `itkGradientRecursiveGaussianImageFilter.h`.
//!
//! The five direct (non-Gaussian) filters share one substrate: walk a
//! [`NeighborhoodIterator`] over an `f64` copy of the input under
//! [`ZeroFluxNeumannBoundaryCondition`] — the boundary condition all five use
//! in ITK — narrowing back to the output pixel type (`crate::image_from_f64`)
//! only once, at the end. [`gradient`] is the vector-output member of this
//! group: it assembles all `dim` central-difference components per pixel in
//! one neighborhood pass instead of returning a scalar.
//!
//! [`gradient_magnitude_recursive_gaussian`], [`laplacian_recursive_gaussian`]
//! and [`gradient_recursive_gaussian`] instead compose per-axis calls to
//! `recursive_gaussian_f64_from_into`, exactly as
//! ITK's `GradientMagnitudeRecursiveGaussianImageFilter`/
//! `LaplacianRecursiveGaussianImageFilter`/`GradientRecursiveGaussianImageFilter`
//! compose per-axis `RecursiveGaussianImageFilter`s (one
//! [`GaussianOrder::FirstOrder`] or [`GaussianOrder::SecondOrder`] axis,
//! [`GaussianOrder::ZeroOrder`] elsewhere) — then divide each axis's
//! contribution by `spacing[d]` (gradient) or `spacing[d]^2` (Laplacian)
//! *again*: the recursion's own `sigmad = sigma /
//! spacing[d]` reparametrization makes its derivative output index-space, and
//! these filters need it in physical space, matching ITK's `GenerateData`
//! (`a + Math::sqr(b / spacing[dim])` and `a + b * (1.0 / spacing2)`
//! respectively) and `itkGradientRecursiveGaussianImageFilter.hxx`'s
//! `it.Get() / spacing`.
//!
//! They take the `_into` form, on one working buffer reused across the axis
//! loop, because they are the callers that run the whole `dim`-axis cascade once
//! *per axis*: anything the recursion allocates internally, they allocate `dim`
//! times over, and each one is a full volume whose pages must be faulted in
//! under a kernel lock — work that does not parallelize.
//!
//! Output pixel type follows SimpleITK's yaml: [`gradient_magnitude`],
//! [`gradient_magnitude_recursive_gaussian`] and [`laplacian_recursive_gaussian`]
//! all declare `output_pixel_type: float` and so always produce
//! [`PixelId::Float32`]; [`derivative`], [`laplacian`] and
//! [`sobel_edge_detection`] declare `RealPixelIDTypeList` with no override and
//! so keep the input's pixel type; [`gradient`] and
//! [`gradient_recursive_gaussian`] fix their `output_image_type` to
//! `itk::VectorImage<float, D>` and so always produce
//! [`PixelId::VectorFloat32`], regardless of input pixel type.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::recursive_gaussian::{GaussianOrder, recursive_gaussian_f64_from_into};
use sitk_core::{
    Image, NeighborhoodIterator, PixelId, Scalar, ZeroFluxNeumannBoundaryCondition,
    dispatch_scalar, matrix, parallel,
};

/// An `f64` copy of `img`'s pixels with `img`'s geometry (spacing in
/// particular), used as the working buffer for every filter in this module.
fn scratch_f64(img: &Image) -> Result<Image> {
    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec()?)?;
    scratch.copy_geometry_from(img);
    Ok(scratch)
}

// ---- gradient_magnitude ----------------------------------------------------

/// `GradientMagnitudeImageFilter`: the Euclidean norm of the central-difference
/// gradient, `sqrt(sum_d ((f(x+e_d) - f(x-e_d)) / (2 * scale_d))^2)`, under
/// [`ZeroFluxNeumannBoundaryCondition`]. `use_image_spacing` (ITK's
/// `UseImageSpacing`, on by default) sets `scale_d = spacing[d]`; off,
/// `scale_d = 1`. Output is always [`PixelId::Float32`] (SimpleITK's
/// `output_pixel_type: float`).
pub fn gradient_magnitude(img: &Image, use_image_spacing: bool) -> Result<Image> {
    let out = gradient_magnitude_values(img, use_image_spacing)?;
    image_from_f64(PixelId::Float32, img.size(), img, &out)
}

/// The raw `f64` gradient-magnitude values, before [`gradient_magnitude`]
/// narrows them to `PixelId::Float32`.
///
/// `pub(crate)`: [`crate::watershed_classic::isolated_watershed`] needs them
/// at full `RealType` precision. ITK instantiates the gradient magnitude as
/// `GradientMagnitudeImageFilter<InputImageType, RealImageType>`, whose output
/// pixel type is `NumericTraits<InputPixelType>::RealType` — `double` for
/// every integer input — whereas SimpleITK's standalone
/// `GradientMagnitudeImageFilter.yaml` fixes the output at `float`. Going
/// through [`gradient_magnitude`] would quantize the watershed's height
/// function to `f32` for `u8`/`u16`/... inputs, which ITK does not do.
pub(crate) fn gradient_magnitude_values(img: &Image, use_image_spacing: bool) -> Result<Vec<f64>> {
    /// The stencil, over the image's **native** pixel type.
    ///
    /// Reading `T` and widening per access, instead of first materializing an
    /// `f64` copy of the whole volume, is lossless — so every value the
    /// arithmetic sees is the `f64` the copy would have held — and it halves the
    /// bytes the stencil streams. The window itself is borrowed, not copied; see
    /// [`sitk_core::WindowView`].
    fn compute<T: Scalar>(img: &Image, scales: &[f64]) -> Result<Vec<f64>> {
        let dim = img.dimension();
        let radius = vec![1usize; dim];
        let iter =
            NeighborhoodIterator::<T, _>::new(img, &radius, ZeroFluxNeumannBoundaryCondition)?;

        // Window strides, so a neighbor can be addressed by its linear slot
        // rather than by re-deriving an ND index per access. Every axis of this
        // window has extent 3, so the stride along axis `d` is `3^d` — exactly
        // what `Neighborhood::get` computed, once per call, in a loop.
        let center = iter.len() / 2;
        let mut window_stride = vec![0usize; dim];
        let mut stride = 1usize;
        for s in window_stride.iter_mut() {
            *s = stride;
            stride *= 3;
        }

        // Parallel over output voxels. The `acc += g * g` sum runs over the axes
        // of one voxel's own window, in axis order, exactly as before — nothing
        // accumulates across voxels, so the output bits are unchanged.
        Ok(iter.par_map_window(|_, w| {
            let mut acc = 0.0;
            for d in 0..dim {
                let plus = w.get_f64(center + window_stride[d]);
                let minus = w.get_f64(center - window_stride[d]);
                let g = 0.5 * (plus - minus) / scales[d];
                acc += g * g;
            }
            acc.sqrt()
        }))
    }

    let dim = img.dimension();
    let scales: Vec<f64> = (0..dim)
        .map(|d| {
            if use_image_spacing {
                img.spacing()[d]
            } else {
                1.0
            }
        })
        .collect();
    dispatch_scalar!(img.pixel_id(), compute, img, &scales)
}

// ---- derivative -------------------------------------------------------------

/// `DerivativeOperator::GenerateCoefficients` (itkDerivativeOperator.hxx),
/// ported operation-for-operation: the 1-D coefficients of the `order`-th
/// central-difference operator, indexed `[-radius, radius]`. `order == 0`
/// yields the identity, `[1.0]`.
///
/// `pub(crate)`: also reused by [`crate::canny`], which applies this same
/// `DerivativeOperator` (unflipped, unscaled — see that module's docs for why
/// the sign convention doesn't matter there) directly inside its fused
/// per-pixel neighborhood pass, rather than through this module's `derivative`
/// filter function.
pub(crate) fn derivative_operator_coefficients(order: u32) -> Vec<f64> {
    let w = (2 * order.div_ceil(2) + 1) as usize;
    let mut coeff = vec![0.0f64; w];
    coeff[w / 2] = 1.0;

    for _ in 0..order / 2 {
        let mut previous = coeff[1] - 2.0 * coeff[0];
        let mut j = 1;
        while j < w - 1 {
            let next = coeff[j - 1] + coeff[j + 1] - 2.0 * coeff[j];
            coeff[j - 1] = previous;
            previous = next;
            j += 1;
        }
        let next = coeff[j - 1] - 2.0 * coeff[j];
        coeff[j - 1] = previous;
        coeff[j] = next;
    }

    for _ in 0..order % 2 {
        let mut previous = 0.5 * coeff[1];
        let mut j = 1;
        while j < w - 1 {
            let next = -0.5 * coeff[j - 1] + 0.5 * coeff[j + 1];
            coeff[j - 1] = previous;
            previous = next;
            j += 1;
        }
        let next = -0.5 * coeff[j - 1];
        coeff[j - 1] = previous;
        coeff[j] = next;
    }

    coeff
}

/// `DerivativeImageFilter`: the `order`-th derivative along `direction`,
/// computed by convolving `derivative_operator_coefficients`'s output — reversed
/// (ITK's `FlipAxes`, so the sign is the standard central-difference sign,
/// e.g. `order=1` gives `(f(x+1)-f(x-1))/(2*scale)`) and, if
/// `use_image_spacing`, scaled once by `1/spacing[direction]` (ITK's
/// `ScaleCoefficients`: a single power regardless of `order`, so a 2nd
/// derivative is *not* divided by `spacing^2` — this literal ITK behavior is
/// reproduced as-is) — under [`ZeroFluxNeumannBoundaryCondition`]. Output
/// keeps `img`'s pixel type.
///
/// Errors if `direction >= img.dimension()`.
pub fn derivative(
    img: &Image,
    direction: usize,
    order: u32,
    use_image_spacing: bool,
) -> Result<Image> {
    let dim = img.dimension();
    if direction >= dim {
        return Err(FilterError::InvalidDirection {
            direction,
            dimension: dim,
        });
    }

    let mut coeff = derivative_operator_coefficients(order);
    coeff.reverse();
    if use_image_spacing {
        let scale = 1.0 / img.spacing()[direction];
        for c in &mut coeff {
            *c *= scale;
        }
    }
    let half = coeff.len() / 2;

    let scratch = scratch_f64(img)?;
    let mut radius = vec![0usize; dim];
    radius[direction] = half;
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let out: Vec<f64> = iter
        .map(|(_, nb)| {
            let mut off = vec![0i64; dim];
            coeff
                .iter()
                .enumerate()
                .map(|(k, &c)| {
                    off[direction] = k as i64 - half as i64;
                    c * nb.get(&off)
                })
                .sum()
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- laplacian --------------------------------------------------------------

/// `LaplacianImageFilter`/`LaplacianOperator`: the isotropic second
/// derivative, `sum_d (f(x+e_d) + f(x-e_d) - 2*f(x)) / scale_d^2`, under
/// [`ZeroFluxNeumannBoundaryCondition`]. `use_image_spacing` sets `scale_d =
/// spacing[d]`; off, `scale_d = 1`. Output keeps `img`'s pixel type.
pub fn laplacian(img: &Image, use_image_spacing: bool) -> Result<Image> {
    let dim = img.dimension();
    let scales_sq: Vec<f64> = (0..dim)
        .map(|d| {
            let s = if use_image_spacing {
                img.spacing()[d]
            } else {
                1.0
            };
            s * s
        })
        .collect();
    let scratch = scratch_f64(img)?;
    let radius = vec![1usize; dim];
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let out: Vec<f64> = iter
        .map(|(_, nb)| {
            let center = nb.center_value();
            let mut acc = 0.0;
            let mut off = vec![0i64; dim];
            for d in 0..dim {
                off[d] = 1;
                let plus = nb.get(&off);
                off[d] = -1;
                let minus = nb.get(&off);
                off[d] = 0;
                acc += (plus + minus - 2.0 * center) / scales_sq[d];
            }
            acc
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- sobel_edge_detection ---------------------------------------------------

/// All ND offsets in `{-1, 0, 1}^dim`; visiting order does not matter since
/// [`Neighborhood::get`](sitk_core::Neighborhood::get) addresses each by its
/// own ND offset rather than by position.
fn unit_box_offsets(dim: usize) -> Vec<Vec<i64>> {
    let mut offsets = vec![vec![]];
    for _ in 0..dim {
        let mut next = Vec::with_capacity(offsets.len() * 3);
        for prefix in &offsets {
            for delta in [-1i64, 0, 1] {
                let mut v = prefix.clone();
                v.push(delta);
                next.push(v);
            }
        }
        offsets = next;
    }
    offsets
}

/// The Sobel operator's weight at `offset` for a derivative along `direction`:
/// `derivative = [-1, 0, 1]` along `direction`, `smoothing = [1, 2, 1]` along
/// every other axis, matching `itkSobelOperator.hxx`'s `GenerateCoefficients`
/// (the non-legacy, N-D case: `K_a(x) = d[x_a] * Product_{i != a} s[x_i]`).
/// `use_legacy` selects ITK's hardcoded 3-D-only legacy stencil instead: a
/// non-separable 1/3/6 pair-weight over the two non-derivative axes
/// (`[1,3,1;3,6,3;1,3,1]`), verified directly against ITK's literal
/// `direction=0` coefficient array.
fn sobel_weight(offset: &[i64], direction: usize, use_legacy: bool) -> f64 {
    let d = offset[direction] as f64;
    if offset.len() == 3 && use_legacy {
        let others: Vec<i64> = (0..3)
            .filter(|&a| a != direction)
            .map(|a| offset[a])
            .collect();
        let pair = match (others[0] == 0, others[1] == 0) {
            (true, true) => 6.0,
            (false, false) => 1.0,
            _ => 3.0,
        };
        return d * pair;
    }
    (0..offset.len())
        .filter(|&a| a != direction)
        .fold(d, |acc, a| if offset[a] == 0 { acc * 2.0 } else { acc })
}

/// `SobelEdgeDetectionImageFilter`: the Euclidean norm of the per-axis Sobel
/// operator response, `sqrt(sum_d g_d^2)`, under
/// [`ZeroFluxNeumannBoundaryCondition`]. `use_legacy_operator_coefficients`
/// (ITK's `UseLegacyOperatorCoefficients`; SimpleITK's yaml default is
/// `false`, though ITK's own C++ class default is `true`) selects the
/// non-separable 3-D-only legacy stencil in place of the separable
/// `[-1,0,1] * [1,2,1]` kernel — it only changes anything for a 3-D image.
/// Output keeps `img`'s pixel type.
pub fn sobel_edge_detection(img: &Image, use_legacy_operator_coefficients: bool) -> Result<Image> {
    let dim = img.dimension();
    let scratch = scratch_f64(img)?;
    let radius = vec![1usize; dim];
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, &radius, ZeroFluxNeumannBoundaryCondition)?;
    let offsets = unit_box_offsets(dim);

    let out: Vec<f64> = iter
        .map(|(_, nb)| {
            let mut acc = 0.0;
            for direction in 0..dim {
                let g: f64 = offsets
                    .iter()
                    .map(|off| {
                        sobel_weight(off, direction, use_legacy_operator_coefficients) * nb.get(off)
                    })
                    .sum();
                acc += g * g;
            }
            acc.sqrt()
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

// ---- gradient_magnitude_recursive_gaussian / laplacian_recursive_gaussian --

/// `GradientMagnitudeRecursiveGaussianImageFilter`: the Euclidean norm of the
/// gradient of `img` convolved with a Gaussian of physical-space `sigma`
/// (isotropic — one value for every axis, matching ITK's single `Sigma`
/// parameter). Composes per-axis [`recursive_gaussian_f64_from_into`] calls —
/// [`GaussianOrder::FirstOrder`] on one axis, [`GaussianOrder::ZeroOrder`] on
/// the rest — dividing each axis's derivative by `spacing[d]` again to convert
/// it from the recursion's index space to physical space.
/// `normalize_across_scale` is ITK's `NormalizeAcrossScale` (off by default).
/// Output is always [`PixelId::Float32`].
///
/// Errors if `sigma < 0`, or an axis (every axis, since `sigma` is shared) has
/// fewer than four pixels.
pub fn gradient_magnitude_recursive_gaussian(
    img: &Image,
    sigma: f64,
    normalize_across_scale: bool,
) -> Result<Image> {
    let dim = img.dimension();
    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let sigma_array = vec![sigma; dim];

    // Three full volumes, allocated once each and reused across the axis loop:
    // the `f64` input, the working buffer the recursion runs in, and the
    // accumulator. This used to be fifteen — 2.0 GB of fresh pages per call at
    // 256³, against 134 MB for `smoothing_recursive_gaussian`, which runs the
    // same recursion and beats ITK. Twelve of them came from the axis loop: the
    // recursion allocated its own volume per axis, handed it back through an
    // `Image` (`recursive_gaussian_with_order` narrows `Float64` to `Float64` —
    // the identity), which was immediately unwrapped again by `to_f64_vec`, and
    // then `acc` was reallocated to hold the sum. A fresh volume must have its
    // pages faulted in under a kernel lock, which is work that does not
    // parallelize (efficiency 0.09), so those twelve were most of what the
    // filter did at high thread counts.
    let src = img.to_f64_vec()?;
    let mut work = vec![0.0f64; src.len()];
    let mut acc = vec![0.0f64; src.len()];

    for d in 0..dim {
        let mut orders = vec![GaussianOrder::ZeroOrder; dim];
        orders[d] = GaussianOrder::FirstOrder;
        // The recursion destroys its input, so each axis needs its own copy of
        // `src` — but it is the *recursion's first axis pass* that makes it, by
        // reading `src` and writing `work`, instead of a `memcpy` ahead of a pass
        // that then reads `work` back. The copy this replaces was `std`'s
        // single-threaded one, and at 512³ it was also where `work`'s 1.07 GB of
        // fresh pages got faulted in, on one core: 517 ms on the first axis, 684 ms
        // over the three, 42% of the filter. Folding it into the first pass deletes
        // that traffic rather than spreading it, and lets those pages be faulted by
        // a parallel line pass.
        recursive_gaussian_f64_from_into(
            &src,
            &mut work,
            &size,
            &spacing,
            &sigma_array,
            &orders,
            normalize_across_scale,
        )?;
        // The axis loop stays sequential, so `acc[i]` still accumulates its
        // `dim` terms in axis order — only the elementwise step within one axis
        // is spread across threads, and each element's arithmetic is untouched.
        // The tap stays a *division* by `spacing[d]`: multiplying by a
        // precomputed `1.0 / spacing[d]` is a different `f64` and would move the
        // checksum.
        parallel::for_each_mut(&mut acc, |i, a| {
            let g = work[i] / spacing[d];
            *a += g * g;
        });
    }
    parallel::for_each_mut(&mut acc, |_, a| *a = a.sqrt());

    image_from_f64(PixelId::Float32, img.size(), img, &acc)
}

/// `LaplacianRecursiveGaussianImageFilter`: the Laplacian-of-Gaussian of
/// `img`, `sum_d d2/dx_d^2 [G_sigma * img]`. Composes per-axis
/// [`recursive_gaussian_f64_from_into`] calls — [`GaussianOrder::SecondOrder`] on
/// one axis, [`GaussianOrder::ZeroOrder`] on the rest — dividing each axis's
/// second derivative by `spacing[d]^2` again to convert it from
/// the recursion's index space to physical space.
/// `normalize_across_scale` is ITK's `NormalizeAcrossScale` (off by default).
/// Output is always [`PixelId::Float32`].
///
/// Errors if `sigma < 0`, or an axis (every axis, since `sigma` is shared) has
/// fewer than four pixels.
pub fn laplacian_recursive_gaussian(
    img: &Image,
    sigma: f64,
    normalize_across_scale: bool,
) -> Result<Image> {
    let dim = img.dimension();
    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let sigma_array = vec![sigma; dim];

    // Same three-volume shape as `gradient_magnitude_recursive_gaussian`, and it
    // was the same defect: a per-axis recursion that allocated its own volume,
    // round-tripped it through an `Image` for nothing, and unwrapped it again.
    let src = img.to_f64_vec()?;
    let mut work = vec![0.0f64; src.len()];
    let mut acc = vec![0.0f64; src.len()];

    for d in 0..dim {
        let mut orders = vec![GaussianOrder::ZeroOrder; dim];
        orders[d] = GaussianOrder::SecondOrder;
        // The recursion's first axis pass makes the per-axis copy of `src` itself,
        // by reading `src` and writing `work` — see
        // `gradient_magnitude_recursive_gaussian`, which carried the same
        // single-threaded `memcpy`.
        recursive_gaussian_f64_from_into(
            &src,
            &mut work,
            &size,
            &spacing,
            &sigma_array,
            &orders,
            normalize_across_scale,
        )?;
        // `inv_spacing_sq` is precomputed and multiplied, as it always was here —
        // unlike the gradient magnitude above, which divides. Keeping each
        // filter's own arithmetic is what keeps each checksum.
        let inv_spacing_sq = 1.0 / (spacing[d] * spacing[d]);
        parallel::for_each_mut(&mut acc, |i, a| {
            *a += work[i] * inv_spacing_sq;
        });
    }

    image_from_f64(PixelId::Float32, img.size(), img, &acc)
}

// ---- gradient (plain, vector output) ---------------------------------------

/// `GradientImageFilter` (`itkGradientImageFilter.hxx`): the per-axis central
/// difference `(f(x+e_d) - f(x-e_d)) / (2 * scale_d)` at every pixel,
/// assembled into a `dim`-component covariant-vector image — one component
/// per axis — under [`ZeroFluxNeumannBoundaryCondition`]
/// (`itkGradientImageFilter.h:229-231`, the filter's default
/// `m_BoundaryCondition`).
///
/// `use_image_spacing` (ITK's `UseImageSpacing`, `GradientImageFilter.yaml`
/// default `true`, matching the ITK class default at
/// `itkGradientImageFilter.h:222`) sets `scale_d = spacing[d]`; off,
/// `scale_d = 1`. Each axis's weight is exactly
/// `derivative_operator_coefficients(1)`, reversed and (if
/// `use_image_spacing`) scaled by `1/spacing[d]` — the same per-axis
/// coefficients [`derivative`] uses, evaluated here for every axis at once
/// and assembled into one vector pixel instead of `dim` separate scalar
/// calls.
///
/// `use_image_direction` (ITK's `UseImageDirection`) rotates the assembled
/// gradient vector by the image's direction cosine matrix
/// (`itkImageBase.h:634-653`'s `TransformLocalVectorToPhysicalVector`:
/// `output = Direction * input`, row-major, no spacing) before it is written
/// out — **`GradientImageFilter.yaml`'s wrapped default is `false`**
/// (`GradientImageFilter.yaml:23-25`), even though the underlying ITK class
/// itself defaults this flag to `true` (`itkGradientImageFilter.h:226`,
/// `bool m_UseImageDirection{ true }`) — the same ITK-class-default-vs-yaml-
/// wrapped-default split already documented for [`sobel_edge_detection`]'s
/// `use_legacy_operator_coefficients`.
///
/// Output is always [`PixelId::VectorFloat32`] with `dim` components
/// (`GradientImageFilter.yaml:7`'s `output_image_type:
/// itk::VectorImage<float, InputImageType::ImageDimension>`; the yaml's
/// `filter_type` also fixes the two extra ITK template parameters
/// (`OperatorValueType`, `OutputValueType`) to `float`, so upstream computes
/// this filter's arithmetic at `float` precision — this port instead computes
/// in `f64` throughout, like every other filter in this module, and narrows
/// only at the final `f32` output; the two-tap central-difference weights are
/// exact halves in either precision, so this does not change any documented
/// value beyond ULP-level rounding).
///
/// Scalar input only (`pixel_types: BasicPixelIDTypeList`, no vector variant)
/// — a vector or complex image is rejected the same way every other
/// scalar-only filter in this module is, through the `scratch_f64`/
/// `to_f64_vec` scalar seam's [`sitk_core::Error::RequiresScalarPixelType`].
pub fn gradient(img: &Image, use_image_spacing: bool, use_image_direction: bool) -> Result<Image> {
    let dim = img.dimension();
    let spacing = img.spacing().to_vec();
    let direction = img.direction().to_vec();
    let scratch = scratch_f64(img)?;
    let radius = vec![1usize; dim];
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let data: Vec<f32> = iter
        .flat_map(|(_, nb)| {
            let mut off = vec![0i64; dim];
            let mut vector = vec![0.0f64; dim];
            for d in 0..dim {
                off[d] = 1;
                let plus = nb.get(&off);
                off[d] = -1;
                let minus = nb.get(&off);
                off[d] = 0;
                let mut g = 0.5 * (plus - minus);
                if use_image_spacing {
                    g /= spacing[d];
                }
                vector[d] = g;
            }
            if use_image_direction {
                vector = matrix::mat_vec(&direction, &vector, dim);
            }
            vector.into_iter().map(f32::from_f64)
        })
        .collect();

    let mut result = Image::from_vec_vector(img.size(), dim, data)?;
    result.copy_geometry_from(img);
    Ok(result)
}

// ---- gradient_recursive_gaussian --------------------------------------------

/// `GradientRecursiveGaussianImageFilter`
/// (`itkGradientRecursiveGaussianImageFilter.hxx`): the gradient of `img`
/// convolved with a Gaussian of physical-space `sigma` (isotropic — one value
/// for every axis, matching ITK's single `Sigma` parameter,
/// `GradientRecursiveGaussianImageFilter.yaml:9-11`, default `1.0`), assembled
/// into a covariant-vector image.
///
/// For each axis `d`, this composes per-axis [`recursive_gaussian_f64_from_into`] calls
/// exactly like [`gradient_magnitude_recursive_gaussian`] does —
/// [`GaussianOrder::FirstOrder`] on axis `d`, [`GaussianOrder::ZeroOrder`] on
/// the rest — then divides that axis's derivative by `spacing[d]` again to
/// convert it from the recursion's index space to physical space,
/// matching the hxx's explicit `it.Get() / spacing`
/// (`itkGradientRecursiveGaussianImageFilter.hxx:245-251`). Axis `d`'s result
/// becomes output component `d` (scalar input) or `nc * dim + d` for input
/// component `nc` (vector input, `itkGradientRecursiveGaussianImageFilter.hxx:239`'s
/// `m_ImageAdaptor->SelectNthElement(nc * ImageDimension + dim)`).
///
/// **Cascade order.** As with [`crate::smoothing_recursive_gaussian`], ITK's actual
/// pipeline processes the derivative axis first and the isotropic-sigma
/// smoothing axes after, in a specific per-`dim` order
/// (`itkGradientRecursiveGaussianImageFilter.hxx:205-256`); this port always
/// processes axes `0..D` in the crate's canonical order, matching how
/// [`gradient_magnitude_recursive_gaussian`]/[`laplacian_recursive_gaussian`]
/// already do it. Since `Sigma` is isotropic, every smoothing axis uses the
/// same sigma regardless of cascade order, so the two orders compute the same
/// separable composition and can only disagree at the ULP level.
///
/// **Vector images.** `pixel_types` is `BasicPixelIDTypeList` **+**
/// `VectorPixelIDTypeList` (`GradientRecursiveGaussianImageFilter.yaml:6`);
/// each input component is processed independently
/// (`itkGradientRecursiveGaussianImageFilter.hxx:185-191`'s `nComponents`
/// loop), via [`sitk_core::Image::extract_component`]. A complex image falls
/// through to the scalar branch and is rejected there by `to_f64_vec`, the
/// same as [`crate::smoothing_recursive_gaussian`].
///
/// `normalize_across_scale` is ITK's `NormalizeAcrossScale`
/// (`GradientRecursiveGaussianImageFilter.yaml:19-21`, default `false`,
/// matching the ITK class default at
/// `itkGradientRecursiveGaussianImageFilter.h:254`).
///
/// `use_image_direction` (ITK's `UseImageDirection`) rotates each assembled
/// `dim`-component gradient sub-vector — one per input component — by the
/// direction cosine matrix (`TransformOutputPixel`,
/// `itkGradientRecursiveGaussianImageFilter.hxx:271-283` /
/// `itkGradientRecursiveGaussianImageFilter.h:211-237`: `output = Direction *
/// input`, row-major, no spacing). **`GradientRecursiveGaussianImageFilter.yaml`'s
/// wrapped default is `false`** (`GradientRecursiveGaussianImageFilter.yaml:29-31`),
/// even though the ITK class itself defaults this flag to `true`
/// (`itkGradientRecursiveGaussianImageFilter.h:257`,
/// `bool m_UseImageDirection{ true }`) — the same split as [`gradient`]'s
/// `use_image_direction`.
///
/// Output is always [`PixelId::VectorFloat32`], with `dim * input_components`
/// components (`GradientRecursiveGaussianImageFilter.yaml:7`'s
/// `output_image_type: itk::VectorImage<float, ImageDimension>`, sized by
/// `itkGradientRecursiveGaussianImageFilter.hxx:298`'s
/// `SetNumberOfComponentsPerPixel(inputComponents * ImageDimension)`).
///
/// Errors if `sigma < 0`, or an axis (every axis, since `sigma` is shared) has
/// fewer than four pixels.
pub fn gradient_recursive_gaussian(
    img: &Image,
    sigma: f64,
    normalize_across_scale: bool,
    use_image_direction: bool,
) -> Result<Image> {
    let dim = img.dimension();
    let spacing = img.spacing().to_vec();
    let direction = img.direction().to_vec();
    let sigma_array = vec![sigma; dim];
    let input_components = img.number_of_components_per_pixel();

    let mut out = vec![0.0f64; img.number_of_pixels() * input_components * dim];

    let size = img.size().to_vec();
    // One working buffer for the whole `component x axis` loop, instead of a
    // fresh volume inside the recursion on every one of its `input_components *
    // dim` iterations. `src` is refilled per component, `work` per axis.
    let mut work = vec![0.0f64; img.number_of_pixels()];

    for nc in 0..input_components {
        let component = if img.pixel_id().is_vector() {
            img.extract_component(nc)?
        } else {
            img.clone()
        };
        let src = component.to_f64_vec()?;
        for d in 0..dim {
            let mut orders = vec![GaussianOrder::ZeroOrder; dim];
            orders[d] = GaussianOrder::FirstOrder;
            // Same as the two filters above: the recursion's first axis pass reads
            // `src` and writes `work`, so the per-axis copy is the pass, not a
            // single-threaded `memcpy` in front of it.
            recursive_gaussian_f64_from_into(
                &src,
                &mut work,
                &size,
                &spacing,
                &sigma_array,
                &orders,
                normalize_across_scale,
            )?;
            let inv_spacing = 1.0 / spacing[d];
            for (p, &v) in work.iter().enumerate() {
                out[p * input_components * dim + nc * dim + d] = v * inv_spacing;
            }
        }
    }

    if use_image_direction {
        for g in 0..(img.number_of_pixels() * input_components) {
            let base = g * dim;
            let rotated = matrix::mat_vec(&direction, &out[base..base + dim], dim);
            out[base..base + dim].copy_from_slice(&rotated);
        }
    }

    let data: Vec<f32> = out.into_iter().map(f32::from_f64).collect();
    let mut result = Image::from_vec_vector(img.size(), input_components * dim, data)?;
    result.copy_geometry_from(img);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_2d(w: usize, h: usize, slope: f64) -> Vec<f64> {
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = slope * x as f64;
            }
        }
        data
    }

    // ---- gradient_magnitude ----

    #[test]
    fn gradient_magnitude_constant_image_is_zero_2d() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let out = gradient_magnitude(&img, true).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_magnitude_constant_image_is_zero_3d() {
        let img = Image::from_vec(&[3, 3, 3], vec![7.0f64; 27]).unwrap();
        let out = gradient_magnitude(&img, true).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_magnitude_linear_ramp_matches_slope_over_spacing() {
        let (w, h) = (7usize, 7usize);
        let slope = 3.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = gradient_magnitude(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        // interior point: dI/dx = slope/spacing_x = 1.5, dI/dy = 0.
        let expected = slope / 2.0;
        assert!((vals[3 * w + 3] - expected).abs() < 1e-9);
    }

    #[test]
    fn gradient_magnitude_use_image_spacing_false_ignores_spacing() {
        let (w, h) = (7usize, 7usize);
        let slope = 3.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = gradient_magnitude(&img, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[w + 1] - slope).abs() < 1e-9);
    }

    #[test]
    fn gradient_magnitude_border_uses_zero_flux_neumann() {
        // 1-D-in-2-D column so the border behavior is easy to hand-derive:
        // x: 0,1,4,9,16 (squares); zero-flux clamps the neighbor past the edge.
        let w = 5;
        let img = Image::from_vec(&[w, 1], vec![0.0f64, 1.0, 4.0, 9.0, 16.0]).unwrap();
        let out = gradient_magnitude(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        // at x=0: neighbors clamp to (0, 1) -> (1-0)/2 = 0.5.
        assert!((vals[0] - 0.5).abs() < 1e-9);
        // at x=4 (last): neighbors clamp to (9, 16) -> (16-9)/2 = 3.5.
        assert!((vals[4] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn gradient_magnitude_output_is_always_float32() {
        let img = Image::from_vec(&[3, 3], vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let out = gradient_magnitude(&img, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    // ---- derivative ----

    #[test]
    fn derivative_constant_image_is_zero() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let out = derivative(&img, 0, 1, true).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn derivative_first_order_ramp_matches_slope_over_spacing() {
        let (w, h) = (7usize, 7usize);
        let slope = 4.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = derivative(&img, 0, 1, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[3 * w + 3] - slope / 2.0).abs() < 1e-9);
    }

    #[test]
    fn derivative_use_image_spacing_false() {
        let (w, h) = (7usize, 7usize);
        let slope = 4.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = derivative(&img, 0, 1, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[3 * w + 3] - slope).abs() < 1e-9);
    }

    #[test]
    fn derivative_second_order_ramp_is_zero_in_interior() {
        let (w, h) = (9usize, 3usize);
        let img = Image::from_vec(&[w, h], ramp_2d(w, h, 5.0)).unwrap();
        let out = derivative(&img, 0, 2, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!(vals[w + 4].abs() < 1e-9);
    }

    #[test]
    fn derivative_second_order_scales_by_single_spacing_power_bug_compatible() {
        // ITK's ScaleCoefficients divides by spacing exactly once regardless of
        // order, so a 2nd-derivative quadratic (I=x^2, d2I/dx2=2 exactly, in
        // index space) with spacing=2 yields 2 * (1/2) = 1.0, NOT 2/(2^2)=0.5.
        let (w, h) = (9usize, 3usize);
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x * x) as f64;
            }
        }
        let mut img = Image::from_vec(&[w, h], data).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = derivative(&img, 0, 2, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[w + 4] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn derivative_border_uses_zero_flux_neumann() {
        let w = 5;
        let img = Image::from_vec(&[w, 1], vec![0.0f64, 1.0, 4.0, 9.0, 16.0]).unwrap();
        let out = derivative(&img, 0, 1, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-9);
        assert!((vals[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn derivative_invalid_direction_is_rejected() {
        let img = Image::from_vec(&[3, 3], vec![1.0f64; 9]).unwrap();
        let err = derivative(&img, 5, 1, true).unwrap_err();
        assert_eq!(
            err,
            FilterError::InvalidDirection {
                direction: 5,
                dimension: 2
            }
        );
    }

    #[test]
    fn derivative_3d_matches_slope_over_spacing() {
        let (w, h, d) = (7usize, 3usize, 3usize);
        let slope = 2.0;
        let mut data = vec![0.0f64; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = slope * x as f64;
                }
            }
        }
        let mut img = Image::from_vec(&[w, h, d], data).unwrap();
        img.set_spacing(&[4.0, 1.0, 1.0]).unwrap();
        let out = derivative(&img, 0, 1, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        let idx = w * h + w + 3;
        assert!((vals[idx] - slope / 4.0).abs() < 1e-9);
    }

    // ---- laplacian ----

    #[test]
    fn laplacian_constant_image_is_zero() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let out = laplacian(&img, true).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn laplacian_quadratic_bowl_matches_curvature() {
        // I(x,y) = x^2 + y^2; discrete second difference is exactly 2 per axis
        // (index space), so Laplacian = 2 + 2 = 4 with unit spacing.
        let (w, h) = (7usize, 7usize);
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x * x + y * y) as f64;
            }
        }
        let img = Image::from_vec(&[w, h], data).unwrap();
        let out = laplacian(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[3 * w + 3] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn laplacian_anisotropic_spacing_divides_by_spacing_squared() {
        let (w, h) = (7usize, 7usize);
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x * x + y * y) as f64;
            }
        }
        let mut img = Image::from_vec(&[w, h], data).unwrap();
        img.set_spacing(&[2.0, 0.5]).unwrap();
        let out = laplacian(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        // 2/spacing_x^2 + 2/spacing_y^2 = 2/4 + 2/0.25 = 0.5 + 8.0 = 8.5.
        assert!((vals[3 * w + 3] - 8.5).abs() < 1e-9);
    }

    #[test]
    fn laplacian_use_image_spacing_false() {
        let (w, h) = (7usize, 7usize);
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x * x + y * y) as f64;
            }
        }
        let mut img = Image::from_vec(&[w, h], data).unwrap();
        img.set_spacing(&[2.0, 0.5]).unwrap();
        let out = laplacian(&img, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[3 * w + 3] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn laplacian_border_uses_zero_flux_neumann() {
        let w = 5;
        let img = Image::from_vec(&[w, 1], vec![0.0f64, 1.0, 4.0, 9.0, 16.0]).unwrap();
        let out = laplacian(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        // at x=0: neighbors clamp to (0,1); (0+1-0)/1 - wait compute directly:
        // plus=1 (x=1), minus=0 (clamped x=0), center=0 -> (1+0-0)=1... but
        // ITK direction weight also applies per-axis; here it's the sum over
        // the single axis: (plus+minus-2*center) = (1+0-0)=1.
        assert!((vals[0] - 1.0).abs() < 1e-9);
        // interior x=2: plus=9,minus=1,center=4 -> 9+1-8=2... but the discrete
        // 2nd difference of squares is exactly 2 in the true interior; x=2 is
        // interior here (neighbors x=1,3 both valid): 9+1-2*4=2.
        assert!((vals[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn laplacian_3d_matches_curvature() {
        let (w, h, d) = (5usize, 5usize, 5usize);
        let mut data = vec![0.0f64; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (x * x + y * y + z * z) as f64;
                }
            }
        }
        let img = Image::from_vec(&[w, h, d], data).unwrap();
        let out = laplacian(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        let idx = 2 * w * h + 2 * w + 2;
        assert!((vals[idx] - 6.0).abs() < 1e-9);
    }

    // ---- sobel_edge_detection ----

    #[test]
    fn sobel_constant_image_is_zero() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let out = sobel_edge_detection(&img, false).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn sobel_2d_ramp_matches_closed_form() {
        // I(x,y) = k*x. Sobel-x response = 8k (sum of derivative weights
        // -1,0,1 each multiplied by smoothing 1,2,1 gives net 8k for a
        // constant-slope ramp); Sobel-y response = 0.
        let (w, h) = (7usize, 7usize);
        let k = 2.0;
        let img = Image::from_vec(&[w, h], ramp_2d(w, h, k)).unwrap();
        let out = sobel_edge_detection(&img, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        assert!((vals[3 * w + 3] - 8.0 * k).abs() < 1e-9);
    }

    #[test]
    fn sobel_3d_non_legacy_matches_closed_form() {
        let (w, h, d) = (5usize, 5usize, 5usize);
        let k = 2.0;
        let mut data = vec![0.0f64; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = k * x as f64;
                }
            }
        }
        let img = Image::from_vec(&[w, h, d], data).unwrap();
        let out = sobel_edge_detection(&img, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        let idx = 2 * w * h + 2 * w + 2;
        // separable weight sum along x: derivative[-1,0,1] * smoothing_y[1,2,1]
        // * smoothing_z[1,2,1], net factor 4*4=16 per unit slope difference,
        // doubled by the +/-1 taps -> 32k.
        assert!((vals[idx] - 32.0 * k).abs() < 1e-9);
    }

    #[test]
    fn sobel_3d_legacy_matches_closed_form() {
        let (w, h, d) = (5usize, 5usize, 5usize);
        let k = 2.0;
        let mut data = vec![0.0f64; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = k * x as f64;
                }
            }
        }
        let img = Image::from_vec(&[w, h, d], data).unwrap();
        let out = sobel_edge_detection(&img, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        let idx = 2 * w * h + 2 * w + 2;
        assert!((vals[idx] - 44.0 * k).abs() < 1e-9);
    }

    #[test]
    fn sobel_border_uses_zero_flux_neumann() {
        let (w, h) = (3usize, 3usize);
        let mut data = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x + 10 * y) as f64;
            }
        }
        let img = Image::from_vec(&[w, h], data).unwrap();
        let out = sobel_edge_detection(&img, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        // top-left corner (0,0) under zero-flux clamp: the x-kernel
        // [-1,0,1;-2,0,2;-1,0,1] against clamped neighbors (0,0,1;0,0,1;10,10,11)
        // gives gx = 1+2-10+11 = 4; the y-kernel [-1,-2,-1;0,0,0;1,2,1] gives
        // gy = -1+10+20+11 = 40.
        let expected = (4.0f64 * 4.0 + 40.0 * 40.0).sqrt();
        assert!((vals[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn sobel_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        let out = sobel_edge_detection(&img, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        let img64 = Image::from_vec(&[3, 3], vec![1.0f64; 9]).unwrap();
        let out64 = sobel_edge_detection(&img64, false).unwrap();
        assert_eq!(out64.pixel_id(), PixelId::Float64);
    }

    // ---- gradient_magnitude_recursive_gaussian ----

    #[test]
    fn gmrg_constant_image_is_near_zero() {
        let img = Image::from_vec(&[41, 41], vec![7.0f64; 41 * 41]).unwrap();
        let out = gradient_magnitude_recursive_gaussian(&img, 2.0, false).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn gmrg_linear_ramp_interior_matches_slope_over_spacing() {
        let n = 161usize;
        let margin = 50usize;
        let slope = 4.0;
        let mut img = Image::from_vec(&[n, n], ramp_2d(n, n, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = gradient_magnitude_recursive_gaussian(&img, 3.0, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        let expected = slope / 2.0;
        for y in margin..(n - margin) {
            for x in margin..(n - margin) {
                let v = vals[y * n + x];
                assert!(
                    (v - expected).abs() < 1e-2,
                    "at ({x},{y}): got {v}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn gmrg_spacing_scaling_matches_exact_ratio() {
        // sigmad = sigma/spacing is what recursive_gaussian_with_order actually
        // uses; scaling spacing and sigma by the same factor keeps sigmad (and
        // so the index-space derivative buffer) bit-identical, making this
        // filter's own extra 1/spacing division produce an EXACT ratio.
        let n = 121usize;
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                data[y * n + x] = (x as f64 - 60.0).abs();
            }
        }
        let img1 = {
            let mut img = Image::from_vec(&[n, n], data.clone()).unwrap();
            img.set_spacing(&[1.0, 1.0]).unwrap();
            img
        };
        let img2 = {
            let mut img = Image::from_vec(&[n, n], data.clone()).unwrap();
            img.set_spacing(&[2.0, 2.0]).unwrap();
            img
        };
        let out1 = gradient_magnitude_recursive_gaussian(&img1, 3.0, false).unwrap();
        let out2 = gradient_magnitude_recursive_gaussian(&img2, 6.0, false).unwrap();
        let v1 = out1.to_f64_vec().unwrap();
        let v2 = out2.to_f64_vec().unwrap();
        for y in (10..n - 10).step_by(7) {
            for x in (10..n - 10).step_by(7) {
                let i = y * n + x;
                assert!(
                    (v1[i] - 2.0 * v2[i]).abs() < 1e-6,
                    "at ({x},{y}): v1={} v2={}",
                    v1[i],
                    v2[i]
                );
            }
        }
    }

    #[test]
    fn gmrg_output_is_always_float32() {
        let img = Image::from_vec(&[9, 9], vec![1u8; 81]).unwrap();
        let out = gradient_magnitude_recursive_gaussian(&img, 1.0, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn gmrg_3d_constant_image_is_near_zero() {
        let img = Image::from_vec(&[9, 9, 9], vec![3.0f64; 9 * 9 * 9]).unwrap();
        let out = gradient_magnitude_recursive_gaussian(&img, 1.0, false).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-5));
    }

    // ---- laplacian_recursive_gaussian ----

    #[test]
    fn lrg_constant_image_is_near_zero() {
        let img = Image::from_vec(&[41, 41], vec![7.0f64; 41 * 41]).unwrap();
        let out = laplacian_recursive_gaussian(&img, 2.0, false).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn lrg_linear_ramp_is_near_zero_in_interior() {
        let n = 161usize;
        let margin = 50usize;
        let mut img = Image::from_vec(&[n, n], ramp_2d(n, n, 2.5)).unwrap();
        img.set_spacing(&[1.5, 1.0]).unwrap();
        let out = laplacian_recursive_gaussian(&img, 3.0, false).unwrap();
        let vals = out.to_f64_vec().unwrap();
        for y in margin..(n - margin) {
            for x in margin..(n - margin) {
                let v = vals[y * n + x];
                assert!(v.abs() < 1e-2, "at ({x},{y}): got {v}, expected ~0");
            }
        }
    }

    #[test]
    fn lrg_spacing_scaling_matches_exact_ratio() {
        let n = 121usize;
        let mut data = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let dx = x as f64 - 60.0;
                data[y * n + x] = dx * dx;
            }
        }
        let img1 = {
            let mut img = Image::from_vec(&[n, n], data.clone()).unwrap();
            img.set_spacing(&[1.0, 1.0]).unwrap();
            img
        };
        let img2 = {
            let mut img = Image::from_vec(&[n, n], data.clone()).unwrap();
            img.set_spacing(&[2.0, 2.0]).unwrap();
            img
        };
        let out1 = laplacian_recursive_gaussian(&img1, 3.0, false).unwrap();
        let out2 = laplacian_recursive_gaussian(&img2, 6.0, false).unwrap();
        let v1 = out1.to_f64_vec().unwrap();
        let v2 = out2.to_f64_vec().unwrap();
        let mid = 60 * n + 60;
        assert!(
            (v1[mid] - 4.0 * v2[mid]).abs() < 1e-4,
            "v1={} v2={}",
            v1[mid],
            v2[mid]
        );
    }

    #[test]
    fn lrg_output_is_always_float32() {
        let img = Image::from_vec(&[9, 9], vec![1u8; 81]).unwrap();
        let out = laplacian_recursive_gaussian(&img, 1.0, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn lrg_3d_constant_image_is_near_zero() {
        let img = Image::from_vec(&[9, 9, 9], vec![3.0f64; 9 * 9 * 9]).unwrap();
        let out = laplacian_recursive_gaussian(&img, 1.0, false).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|&v| v.abs() < 1e-4));
    }

    // ---- gradient ----

    #[test]
    fn gradient_constant_image_is_zero() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let out = gradient(&img, true, false).unwrap();
        assert!(out.components_to_f64_vec().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_linear_ramp_matches_slope_over_spacing() {
        let (w, h) = (7usize, 7usize);
        let slope = 3.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = gradient(&img, true, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.number_of_components_per_pixel(), 2);
        let vals = out.components_to_f64_vec();
        // interior point (3,3): dI/dx = slope/spacing_x = 1.5, dI/dy = 0.
        let idx = (3 * w + 3) * 2;
        assert!((vals[idx] - slope / 2.0).abs() < 1e-6);
        assert!(vals[idx + 1].abs() < 1e-6);
    }

    #[test]
    fn gradient_use_image_spacing_false_ignores_spacing() {
        let (w, h) = (7usize, 7usize);
        let slope = 3.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = gradient(&img, false, false).unwrap();
        let vals = out.components_to_f64_vec();
        let idx = (3 * w + 3) * 2;
        assert!((vals[idx] - slope).abs() < 1e-6);
    }

    #[test]
    fn gradient_border_uses_zero_flux_neumann() {
        // 1-D-in-2-D row: x = 0,1,4,9,16 (squares); zero-flux clamps the
        // neighbor past the edge, matching gradient_magnitude's border test.
        let w = 5;
        let img = Image::from_vec(&[w, 1], vec![0.0f64, 1.0, 4.0, 9.0, 16.0]).unwrap();
        let out = gradient(&img, true, false).unwrap();
        let vals = out.components_to_f64_vec();
        // at x=0: neighbors clamp to (0, 1) -> (1-0)/2 = 0.5.
        assert!((vals[0] - 0.5).abs() < 1e-9);
        // at x=4 (last): neighbors clamp to (9, 16) -> (16-9)/2 = 3.5.
        assert!((vals[4 * 2] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn gradient_output_is_always_vector_float32() {
        let img = Image::from_vec(&[3, 3, 3], vec![1u8; 27]).unwrap();
        let out = gradient(&img, true, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.number_of_components_per_pixel(), 3);
    }

    #[test]
    fn gradient_yaml_default_use_image_direction_false_does_not_rotate() {
        // GradientImageFilter.yaml's wrapped default for UseImageDirection is
        // `false`, unlike the ITK class default of `true` -- pin that a
        // non-identity direction matrix has no effect at the yaml default.
        let (w, h) = (7usize, 7usize);
        let slope = 3.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let out = gradient(&img, true, false).unwrap();
        let vals = out.components_to_f64_vec();
        let idx = (3 * w + 3) * 2;
        assert!((vals[idx] - slope).abs() < 1e-6);
        assert!(vals[idx + 1].abs() < 1e-6);
    }

    #[test]
    fn gradient_use_image_direction_true_rotates_by_direction_matrix() {
        // Direction [0,-1,1,0] (row-major) is a 90-degree CCW rotation:
        // mat_vec gives output = (-v1, v0). At the interior point the
        // un-rotated gradient is (slope, 0), so the rotated result must be
        // (0, slope).
        let (w, h) = (7usize, 7usize);
        let slope = 3.0;
        let mut img = Image::from_vec(&[w, h], ramp_2d(w, h, slope)).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let out = gradient(&img, true, true).unwrap();
        let vals = out.components_to_f64_vec();
        let idx = (3 * w + 3) * 2;
        assert!(vals[idx].abs() < 1e-6);
        assert!((vals[idx + 1] - slope).abs() < 1e-6);
    }

    #[test]
    fn gradient_rejects_a_vector_image() {
        let img = Image::from_vec_vector(&[4, 4], 2, vec![1.0f32; 32]).unwrap();
        assert!(matches!(
            gradient(&img, true, false).unwrap_err(),
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(
                PixelId::VectorFloat32
            ))
        ));
    }

    #[test]
    fn gradient_rejects_a_complex_image() {
        let img = Image::new(&[8, 8], PixelId::ComplexFloat32);
        assert!(matches!(
            gradient(&img, true, false).unwrap_err(),
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(
                PixelId::ComplexFloat32
            ))
        ));
    }

    // ---- gradient_recursive_gaussian ----

    #[test]
    fn grg_scalar_interior_matches_slope_over_spacing_on_a_ramp() {
        // Away from the boundary, Gaussian-smoothing a linear ramp leaves it
        // unchanged, so the recursive-Gaussian gradient's interior value
        // matches the plain central-difference slope/spacing.
        let n = 61usize;
        let margin = 20usize;
        let slope = 2.5;
        let mut img = Image::from_vec(&[n, n], ramp_2d(n, n, slope)).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = gradient_recursive_gaussian(&img, 1.5, false, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.number_of_components_per_pixel(), 2);
        let vals = out.components_to_f64_vec();
        for y in margin..(n - margin) {
            for x in margin..(n - margin) {
                let idx = (y * n + x) * 2;
                assert!(
                    (vals[idx] - slope / 2.0).abs() < 1e-3,
                    "at ({x},{y}): dx {} expected {}",
                    vals[idx],
                    slope / 2.0
                );
                assert!(
                    vals[idx + 1].abs() < 1e-3,
                    "at ({x},{y}): dy {} expected 0",
                    vals[idx + 1]
                );
            }
        }
    }

    #[test]
    fn grg_output_is_always_vector_float32() {
        let img = Image::from_vec(&[16, 16], vec![1u8; 256]).unwrap();
        let out = gradient_recursive_gaussian(&img, 1.0, false, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.number_of_components_per_pixel(), 2);
    }

    #[test]
    fn grg_normalize_across_scale_scales_the_first_order_output_by_sigma() {
        // NormalizeAcrossScale multiplies FirstOrder output by an extra
        // sigma^1 = sigma (Lindeberg scale-space normalization); it is not
        // inert here the way it is for SmoothingRecursiveGaussian's
        // ZeroOrder-only recursion.
        let n = 61usize;
        let slope = 2.5;
        let sigma = 1.5;
        let img = Image::from_vec(&[n, n], ramp_2d(n, n, slope)).unwrap();
        let plain = gradient_recursive_gaussian(&img, sigma, false, false)
            .unwrap()
            .components_to_f64_vec();
        let normalized = gradient_recursive_gaussian(&img, sigma, true, false)
            .unwrap()
            .components_to_f64_vec();
        let idx = (30 * n + 30) * 2;
        assert!(
            (normalized[idx] - plain[idx] * sigma).abs() < 1e-3,
            "normalized {} expected {}",
            normalized[idx],
            plain[idx] * sigma
        );
    }

    #[test]
    fn grg_vector_image_differentiates_each_component_independently() {
        // Two components with different profiles on a proper 2-D image
        // (both axes need >= 4 pixels for the recursion): component 0 an
        // impulse, component 1 a ramp along x -- verifies each is filtered
        // on its own and that the vector composite matches running the
        // scalar composite on each extracted component, laid out as
        // nc*dim + d.
        let n = 21;
        let mut data = vec![0.0f64; n * n * 2];
        for y in 0..n {
            for x in 0..n {
                let p = y * n + x;
                data[p * 2] = if x == n / 2 && y == n / 2 { 100.0 } else { 0.0 };
                data[p * 2 + 1] = x as f64;
            }
        }
        let img = Image::from_vec_vector(&[n, n], 2, data).unwrap();

        let out = gradient_recursive_gaussian(&img, 1.5, false, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(out.number_of_components_per_pixel(), 4); // dim(2) * input_components(2)
        let vector_out = out.components_to_f64_vec();

        for c in 0..2 {
            let scalar_component = img.extract_component(c).unwrap();
            let scalar_out = gradient_recursive_gaussian(&scalar_component, 1.5, false, false)
                .unwrap()
                .components_to_f64_vec();
            for p in 0..(n * n) {
                for d in 0..2 {
                    let expected = scalar_out[p * 2 + d];
                    let got = vector_out[p * 4 + c * 2 + d];
                    assert!(
                        (got - expected).abs() < 1e-9,
                        "component {c} axis {d} pixel {p}: got {got}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn grg_yaml_default_use_image_direction_false_does_not_rotate() {
        let n = 61usize;
        let slope = 2.5;
        let mut img = Image::from_vec(&[n, n], ramp_2d(n, n, slope)).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let out = gradient_recursive_gaussian(&img, 1.5, false, false).unwrap();
        let vals = out.components_to_f64_vec();
        let idx = (30 * n + 30) * 2;
        assert!((vals[idx] - slope).abs() < 1e-3);
        assert!(vals[idx + 1].abs() < 1e-3);
    }

    #[test]
    fn grg_use_image_direction_true_rotates_by_direction_matrix() {
        let n = 61usize;
        let slope = 2.5;
        let mut img = Image::from_vec(&[n, n], ramp_2d(n, n, slope)).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let out = gradient_recursive_gaussian(&img, 1.5, false, true).unwrap();
        let vals = out.components_to_f64_vec();
        let idx = (30 * n + 30) * 2;
        // un-rotated gradient at the interior is (slope, 0); rotated by
        // [0,-1,1,0] (mat_vec: output = (-v1, v0)) gives (0, slope).
        assert!(vals[idx].abs() < 1e-3);
        assert!((vals[idx + 1] - slope).abs() < 1e-3);
    }

    #[test]
    fn grg_rejects_a_complex_image() {
        let img = Image::new(&[8, 8], PixelId::ComplexFloat32);
        assert!(matches!(
            gradient_recursive_gaussian(&img, 1.0, false, false).unwrap_err(),
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(
                PixelId::ComplexFloat32
            ))
        ));
    }

    #[test]
    fn grg_negative_sigma_is_rejected() {
        let img = Image::new(&[8, 8], PixelId::Float64);
        assert!(matches!(
            gradient_recursive_gaussian(&img, -1.0, false, false),
            Err(FilterError::InvalidSigma(_))
        ));
    }

    #[test]
    fn grg_short_axis_is_rejected() {
        let img = Image::new(&[2, 8], PixelId::Float64);
        assert!(matches!(
            gradient_recursive_gaussian(&img, 1.0, false, false),
            Err(FilterError::AxisTooShortForRecursion { axis: 0, len: 2 })
        ));
    }
}
