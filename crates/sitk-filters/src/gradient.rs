//! Gradient / edge-detection filters, porting ITK's derivative-operator and
//! recursive-Gaussian gradient family: `itkGradientMagnitudeImageFilter.h`,
//! `itkDerivativeImageFilter.h` (+ `itkDerivativeOperator.h`),
//! `itkLaplacianImageFilter.h` (+ `itkLaplacianOperator.h`),
//! `itkSobelEdgeDetectionImageFilter.h` (+ `itkSobelOperator.h`),
//! `itkGradientMagnitudeRecursiveGaussianImageFilter.h`, and
//! `itkLaplacianRecursiveGaussianImageFilter.h`.
//!
//! The four direct (non-Gaussian) filters share one substrate: walk a
//! [`NeighborhoodIterator`] over an `f64` copy of the input under
//! [`ZeroFluxNeumannBoundaryCondition`] — the boundary condition all four use
//! in ITK — narrowing back to the output pixel type (`crate::image_from_f64`)
//! only once, at the end.
//!
//! [`gradient_magnitude_recursive_gaussian`] and [`laplacian_recursive_gaussian`]
//! instead compose per-axis calls to [`recursive_gaussian_with_order`], exactly
//! as ITK's `GradientMagnitudeRecursiveGaussianImageFilter`/
//! `LaplacianRecursiveGaussianImageFilter` compose per-axis
//! `RecursiveGaussianImageFilter`s (one [`GaussianOrder::FirstOrder`] or
//! [`GaussianOrder::SecondOrder`] axis, [`GaussianOrder::ZeroOrder`] elsewhere)
//! — then divide each axis's contribution by `spacing[d]` (gradient) or
//! `spacing[d]^2` (Laplacian) *again*: `recursive_gaussian_with_order`'s own
//! `sigmad = sigma / spacing[d]` reparametrization makes its derivative output
//! index-space, and these two filters need it in physical space, matching
//! ITK's `GenerateData` (`a + Math::sqr(b / spacing[dim])` and
//! `a + b * (1.0 / spacing2)` respectively).
//!
//! Output pixel type follows SimpleITK's yaml: [`gradient_magnitude`],
//! [`gradient_magnitude_recursive_gaussian`] and [`laplacian_recursive_gaussian`]
//! all declare `output_pixel_type: float` and so always produce
//! [`PixelId::Float32`]; [`derivative`], [`laplacian`] and
//! [`sobel_edge_detection`] declare `RealPixelIDTypeList` with no override and
//! so keep the input's pixel type.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::recursive_gaussian::{GaussianOrder, recursive_gaussian_with_order};
use sitk_core::{Image, NeighborhoodIterator, PixelId, ZeroFluxNeumannBoundaryCondition};

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
    let scratch = scratch_f64(img)?;
    let radius = vec![1usize; dim];
    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, &radius, ZeroFluxNeumannBoundaryCondition)?;

    let out: Vec<f64> = iter
        .map(|(_, nb)| {
            let mut acc = 0.0;
            let mut off = vec![0i64; dim];
            for d in 0..dim {
                off[d] = 1;
                let plus = nb.get(&off);
                off[d] = -1;
                let minus = nb.get(&off);
                off[d] = 0;
                let g = 0.5 * (plus - minus) / scales[d];
                acc += g * g;
            }
            acc.sqrt()
        })
        .collect();

    image_from_f64(PixelId::Float32, img.size(), img, &out)
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
/// parameter). Composes per-axis [`recursive_gaussian_with_order`] calls —
/// [`GaussianOrder::FirstOrder`] on one axis, [`GaussianOrder::ZeroOrder`] on
/// the rest — dividing each axis's derivative by `spacing[d]` again to convert
/// it from `recursive_gaussian_with_order`'s index space to physical space.
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
    let spacing = img.spacing().to_vec();
    let scratch = scratch_f64(img)?;
    let sigma_array = vec![sigma; dim];

    let mut acc = vec![0.0f64; img.number_of_pixels()];
    for d in 0..dim {
        let mut orders = vec![GaussianOrder::ZeroOrder; dim];
        orders[d] = GaussianOrder::FirstOrder;
        let deriv =
            recursive_gaussian_with_order(&scratch, &sigma_array, &orders, normalize_across_scale)?;
        for (a, v) in acc.iter_mut().zip(deriv.to_f64_vec()?) {
            let g = v / spacing[d];
            *a += g * g;
        }
    }
    let out: Vec<f64> = acc.into_iter().map(f64::sqrt).collect();

    image_from_f64(PixelId::Float32, img.size(), img, &out)
}

/// `LaplacianRecursiveGaussianImageFilter`: the Laplacian-of-Gaussian of
/// `img`, `sum_d d2/dx_d^2 [G_sigma * img]`. Composes per-axis
/// [`recursive_gaussian_with_order`] calls — [`GaussianOrder::SecondOrder`] on
/// one axis, [`GaussianOrder::ZeroOrder`] on the rest — dividing each axis's
/// second derivative by `spacing[d]^2` again to convert it from
/// `recursive_gaussian_with_order`'s index space to physical space.
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
    let spacing = img.spacing().to_vec();
    let scratch = scratch_f64(img)?;
    let sigma_array = vec![sigma; dim];

    let mut acc = vec![0.0f64; img.number_of_pixels()];
    for d in 0..dim {
        let mut orders = vec![GaussianOrder::ZeroOrder; dim];
        orders[d] = GaussianOrder::SecondOrder;
        let deriv =
            recursive_gaussian_with_order(&scratch, &sigma_array, &orders, normalize_across_scale)?;
        let inv_spacing_sq = 1.0 / (spacing[d] * spacing[d]);
        for (a, v) in acc.iter_mut().zip(deriv.to_f64_vec()?) {
            *a += v * inv_spacing_sq;
        }
    }

    image_from_f64(PixelId::Float32, img.size(), img, &acc)
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
}
