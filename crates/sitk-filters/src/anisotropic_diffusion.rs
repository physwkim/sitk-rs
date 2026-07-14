//! ITK's Perona-Malik anisotropic diffusion family, ported from
//! `Modules/Filtering/AnisotropicSmoothing/include/`
//! (`itkAnisotropicDiffusionFunction.h`, `itkAnisotropicDiffusionImageFilter.hxx`,
//! `itkScalarAnisotropicDiffusionFunction.hxx`,
//! `itkGradientAnisotropicDiffusionImageFilter.h` →
//! `itkGradientNDAnisotropicDiffusionFunction.hxx`,
//! `itkCurvatureAnisotropicDiffusionImageFilter.h` →
//! `itkCurvatureNDAnisotropicDiffusionFunction.hxx`) plus the shared solver in
//! `Core/FiniteDifference/include/` (`itkFiniteDifferenceImageFilter.hxx`,
//! `itkDenseFiniteDifferenceImageFilter.hxx`).
//!
//! # The shared driver
//!
//! Both filters are `DenseFiniteDifferenceImageFilter` explicit-Euler solvers:
//! the input is copied to the output, then `number_of_iterations` rounds of
//! "compute the whole update buffer from a frozen snapshot, then
//! `output += time_step · update`" run (`GenerateData`'s `while (!Halt())`
//! loop, `CalculateChange` then `ApplyUpdate`). Zero iterations therefore
//! leaves the cast copy of the input untouched. Every neighborhood read uses
//! [`ZeroFluxNeumannBoundaryCondition`], the default boundary condition of
//! ITK's `ConstNeighborhoodIterator`, and the ND functions both set
//! `radius = 1` along every axis, so the stencil is the full `3^dim` window.
//!
//! `FiniteDifferenceImageFilter::InitializeFunctionCoefficients` sets the
//! per-axis scale coefficients to `1/spacing[d]` when `m_UseImageSpacing`
//! (`true` by default, and never overridden by SimpleITK) and to `1`
//! otherwise; every finite difference below is multiplied by them.
//!
//! # The conductance normalization `K`
//!
//! `AnisotropicDiffusionImageFilter::InitializeIteration` calls
//! `CalculateAverageGradientMagnitudeSquared(output)` whenever
//! `elapsed_iterations % conductance_scaling_update_interval == 0`, and each ND
//! function's `InitializeIteration` then forms
//!
//! ```text
//! K = average_gradient_magnitude_squared · conductance² · (−2)
//! ```
//!
//! so the conductance terms `exp(g²/K)` are ITK's spelling of
//! `exp(−g²/(2·conductance²·⟨|∇u|²⟩))`. `itkScalarAnisotropicDiffusionFunction.hxx`'s
//! estimator ([`average_gradient_magnitude_squared`]) is the mean over *every*
//! pixel of the current solution (interior and zero-flux boundary faces alike,
//! `counter` counting pixels, not pixel×axis) of
//! `Σᵢ (scale[i]·(u[p+eᵢ] − u[p−eᵢ]) / −2)²`. When `K == 0` — a constant image,
//! or `conductance == 0` — both `.hxx` files take the `if (m_K != 0.0)` false
//! branch and leave the conductances at `0.0`, which makes a constant image an
//! exact fixed point of both filters.
//!
//! # Time step
//!
//! `InitializeIteration` compares `time_step` against
//! `min(spacing) / 2^(dim+1)` (with `min(spacing)` replaced by `1` when
//! `use_image_spacing` is off) and, when it is larger, emits an
//! `itkWarningMacro` and *proceeds anyway* — the commented-out clamp right
//! above the warning shows ITK deliberately does not correct it. These
//! functions reproduce that: an unstable `time_step` is accepted and computed
//! with. [`stable_time_step_bound`] exposes the bound so callers can check it
//! themselves. This diverges from [`crate::curvature_flow`], which rejects an
//! unstable step with [`FilterError::UnstableTimeStep`]; there the bound is
//! this crate's own derivation, whereas here it is ITK's, and ITK only warns.
//!
//! # Pixel types
//!
//! Both SimpleITK yaml files declare `pixel_types: RealPixelIDTypeList`, so the
//! wrappers only instantiate for `float` and `double` and the output type
//! equals the input type. Anything else is [`FilterError::RequiresRealPixelType`]
//! here (a compile error in C++; `FiniteDifferenceImageFilter::GenerateData`
//! additionally warns "Output pixel type MUST be float or double"). All
//! arithmetic is done in `f64` and narrowed on store, so a `Float32` input
//! differs from ITK in rounding only, not in the update equation.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{
    Image, NeighborhoodIterator, PixelId, Stencil, WindowView, ZeroFluxNeumannBoundaryCondition,
    parallel,
};

/// `CurvatureNDAnisotropicDiffusionFunction::m_MIN_NORM`, the additive floor
/// under the square root that keeps the normalized first-order differences
/// finite where the gradient vanishes. (`GradientNDAnisotropicDiffusionFunction`
/// declares the same constant but never reads it.)
const MIN_NORM: f64 = 1.0e-10;

/// `AnisotropicDiffusionImageFilter::InitializeIteration`'s stability bound,
/// `min(spacing) / 2^(dim+1)` — with `min(spacing)` taken as `1` when
/// `use_image_spacing` is off, exactly as the `.hxx` does. ITK only warns when
/// `time_step` exceeds this; these filters likewise accept and compute.
pub fn stable_time_step_bound(img: &Image, use_image_spacing: bool) -> f64 {
    let min_spacing = if use_image_spacing {
        img.spacing().iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        1.0
    };
    min_spacing / 2.0f64.powi(img.dimension() as i32 + 1)
}

/// The per-axis scale coefficients of
/// `FiniteDifferenceImageFilter::InitializeFunctionCoefficients`.
fn scale_coefficients(img: &Image, use_image_spacing: bool) -> Vec<f64> {
    img.spacing()
        .iter()
        .map(|&s| if use_image_spacing { 1.0 / s } else { 1.0 })
        .collect()
}

/// `ScalarAnisotropicDiffusionFunction::CalculateAverageGradientMagnitudeSquared`:
/// the mean over every pixel of the current solution of the squared,
/// spacing-scaled centered gradient. Boundary faces read through
/// [`ZeroFluxNeumannBoundaryCondition`]; `counter` advances once per *pixel*,
/// while `accumulator` takes one squared term per pixel *and axis*.
fn average_gradient_magnitude_squared(snapshot: &Image, scale: &[f64]) -> Result<f64> {
    let dim = snapshot.dimension();
    let radius = vec![1usize; dim];
    let iter =
        NeighborhoodIterator::<f64, _>::new(snapshot, &radius, ZeroFluxNeumannBoundaryCondition)?;

    // This is the one place in the module that is a **reduction across pixels**,
    // not a map: `accumulator` sums one squared term per pixel *and axis* over the
    // whole volume, and `f64` addition is not associative, so a rayon `fold`/`sum`
    // would make the result depend on how the work happened to be split.
    //
    // It therefore goes through `map_rows_fold_in_order`, which is the only
    // reduction seam in this port that is bit-stable by construction: the per-pixel
    // terms are computed in parallel into that pixel's own row, and the `+=` runs
    // on one thread, over the rows in index order. `accumulator` sees the identical
    // addition sequence the serial loop fed it — pixel by pixel, axis by axis —
    // so the mean is bit-identical at any thread count, and `combine` is never
    // handed to rayon, so a later caller cannot re-associate it by accident.
    let size = snapshot.size();
    let mut accumulator = 0.0f64;
    let mut counter = 0usize;

    parallel::map_rows_fold_in_order(
        snapshot.number_of_pixels(),
        dim,
        || (iter.window_scratch(), vec![0usize; dim], vec![0i64; dim]),
        |(scratch, center, off), i, row| {
            // Unrank the linear index into an ND center, dimension 0 fastest — the
            // inverse of `Image::linear_index`, and the order the serial walk
            // visited pixels in.
            let mut rest = i;
            for (c, &extent) in center.iter_mut().zip(size) {
                *c = rest % extent;
                rest /= extent;
            }

            // The window is **borrowed**, not materialized. This reduction's
            // decomposition is fixed by `map_rows_fold_in_order` rather than by the
            // iterator's own walk, so `par_map_window` cannot serve it — but
            // `with_window_at` can, and `scratch` is touched only at the ~2% of
            // centers whose window overhangs the image.
            iter.with_window_at(center, scratch, |w| {
                for (a, (&s, cell)) in scale.iter().zip(row.iter_mut()).enumerate() {
                    off[a] = 1;
                    let plus = w.get_offset(off);
                    off[a] = -1;
                    let minus = w.get_offset(off);
                    off[a] = 0;
                    let val = (plus - minus) / -2.0 * s;
                    *cell = val * val;
                }
            });
            true
        },
        |_, row| {
            // One thread, rows in index order: `counter` advances once per pixel and
            // `accumulator` takes the pixel's `dim` squared terms in axis order —
            // exactly the sequence the serial loop produced.
            counter += 1;
            for &term in row {
                accumulator += term;
            }
        },
    );

    if counter == 0 {
        return Ok(0.0);
    }
    Ok(accumulator / counter as f64)
}

/// `GradientNDAnisotropicDiffusionFunction::ComputeUpdate`: the classic
/// Perona-Malik update, with the gradient magnitude entering each per-axis
/// conductance as the half-difference along that axis plus the averaged
/// cross-axis centered differences (`0.25·(dx[j] + dx_aug)²`), which is the
/// "more robust technique for gradient magnitude estimation" the class doc
/// refers to.
fn gradient_update<S: Stencil + ?Sized>(nb: &S, scale: &[f64], k: f64) -> f64 {
    let dim = scale.len();
    let center = nb.center();
    let mut off = vec![0i64; dim];

    // Centralized derivatives, one per dimension.
    let mut dx = vec![0.0f64; dim];
    for (i, d) in dx.iter_mut().enumerate() {
        off[i] = 1;
        let plus = nb.at(&off);
        off[i] = -1;
        let minus = nb.at(&off);
        off[i] = 0;
        *d = (plus - minus) / 2.0 * scale[i];
    }

    let mut delta = 0.0f64;
    for i in 0..dim {
        off[i] = 1;
        let plus_i = nb.at(&off);
        off[i] = -1;
        let minus_i = nb.at(&off);
        off[i] = 0;

        // "Half" directional derivatives.
        let mut dx_forward = (plus_i - center) * scale[i];
        let mut dx_backward = (center - minus_i) * scale[i];

        // The conductance varies per axis because the gradient magnitude
        // approximation is different along each axis.
        let mut accum = 0.0f64;
        let mut accum_d = 0.0f64;
        for j in 0..dim {
            if j == i {
                continue;
            }
            off[i] = 1;
            off[j] = 1;
            let aug_plus = nb.at(&off);
            off[j] = -1;
            let aug_minus = nb.at(&off);
            off[i] = -1;
            let dim_minus = nb.at(&off);
            off[j] = 1;
            let dim_plus = nb.at(&off);
            off[i] = 0;
            off[j] = 0;

            let dx_aug = (aug_plus - aug_minus) / 2.0 * scale[j];
            let dx_dim = (dim_plus - dim_minus) / 2.0 * scale[j];
            accum += 0.25 * (dx[j] + dx_aug).powi(2);
            accum_d += 0.25 * (dx[j] + dx_dim).powi(2);
        }

        let (cx, cxd) = if k != 0.0 {
            (
                ((dx_forward * dx_forward + accum) / k).exp(),
                ((dx_backward * dx_backward + accum_d) / k).exp(),
            )
        } else {
            (0.0, 0.0)
        };

        // Conductance modified first order derivatives, differenced into a
        // conductance modified second order derivative.
        dx_forward *= cx;
        dx_backward *= cxd;
        delta += dx_forward - dx_backward;
    }

    delta
}

/// `CurvatureNDAnisotropicDiffusionFunction::ComputeUpdate`: the modified
/// curvature diffusion equation. The conductance-modified first-order
/// differences are normalized by the local gradient magnitude before being
/// differenced into `speed`, and the result is multiplied by an upwind
/// `|∇u|` chosen by the sign of `speed`.
///
/// ITK's centered differences here come from a `NeighborhoodInnerProduct` with
/// a first-order `DerivativeOperator`, whose coefficients are `{0.5, 0, −0.5}`
/// — i.e. the *negated* centered difference. They are only ever consumed as
/// `(dx[j] + dx_aug)²`, with both terms produced the same way, so this port
/// uses the un-negated centered difference for both and the square is
/// identical.
fn curvature_update<S: Stencil + ?Sized>(nb: &S, scale: &[f64], k: f64) -> f64 {
    let dim = scale.len();
    let center = nb.center();
    let mut off = vec![0i64; dim];

    let mut dx_forward = vec![0.0f64; dim];
    let mut dx_backward = vec![0.0f64; dim];
    let mut dx = vec![0.0f64; dim];
    for i in 0..dim {
        off[i] = 1;
        let plus = nb.at(&off);
        off[i] = -1;
        let minus = nb.at(&off);
        off[i] = 0;

        dx_forward[i] = (plus - center) * scale[i];
        dx_backward[i] = (center - minus) * scale[i];
        dx[i] = (plus - minus) / 2.0 * scale[i];
    }

    let mut speed = 0.0f64;
    for i in 0..dim {
        // Gradient magnitude approximations.
        let mut grad_mag_sq = dx_forward[i] * dx_forward[i];
        let mut grad_mag_sq_d = dx_backward[i] * dx_backward[i];
        for j in 0..dim {
            if j == i {
                continue;
            }
            off[i] = 1;
            off[j] = 1;
            let aug_plus = nb.at(&off);
            off[j] = -1;
            let aug_minus = nb.at(&off);
            off[i] = -1;
            let dim_minus = nb.at(&off);
            off[j] = 1;
            let dim_plus = nb.at(&off);
            off[i] = 0;
            off[j] = 0;

            let dx_aug = (aug_plus - aug_minus) / 2.0 * scale[j];
            let dx_dim = (dim_plus - dim_minus) / 2.0 * scale[j];
            grad_mag_sq += 0.25 * (dx[j] + dx_aug).powi(2);
            grad_mag_sq_d += 0.25 * (dx[j] + dx_dim).powi(2);
        }

        let grad_mag = (MIN_NORM + grad_mag_sq).sqrt();
        let grad_mag_d = (MIN_NORM + grad_mag_sq_d).sqrt();

        let (cx, cxd) = if k != 0.0 {
            ((grad_mag_sq / k).exp(), (grad_mag_sq_d / k).exp())
        } else {
            (0.0, 0.0)
        };

        // First order normalized finite-difference conductance products,
        // differenced into a second order conductance-modified curvature.
        speed += (dx_forward[i] / grad_mag) * cx - (dx_backward[i] / grad_mag_d) * cxd;
    }

    // "Upwind" gradient magnitude term.
    let mut propagation_gradient = 0.0f64;
    for i in 0..dim {
        propagation_gradient += if speed > 0.0 {
            dx_backward[i].min(0.0).powi(2) + dx_forward[i].max(0.0).powi(2)
        } else {
            dx_backward[i].max(0.0).powi(2) + dx_forward[i].min(0.0).powi(2)
        };
    }

    propagation_gradient.sqrt() * speed
}

/// `AnisotropicDiffusionImageFilter`'s iteration loop, shared by both filters:
/// per iteration, refresh `K` when `elapsed % interval == 0`, compute the whole
/// update buffer from the frozen previous solution, then add `time_step ·
/// update` to it.
fn diffuse(
    img: &Image,
    time_step: f64,
    conductance_parameter: f64,
    conductance_scaling_update_interval: u32,
    number_of_iterations: u32,
    use_image_spacing: bool,
    update: impl for<'w> Fn(&WindowView<'w, f64>, &[f64], f64) -> f64 + Sync + Send,
) -> Result<Image> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    if conductance_scaling_update_interval == 0 {
        return Err(FilterError::ZeroConductanceScalingUpdateInterval);
    }

    let dim = img.dimension();
    let scale = scale_coefficients(img, use_image_spacing);
    let size = img.size().to_vec();
    let radius = vec![1usize; dim];
    let mut buf = img.to_f64_vec()?;
    let mut k = 0.0f64;

    for elapsed in 0..number_of_iterations {
        let mut snapshot = Image::from_vec(&size, buf.clone())?;
        snapshot.copy_geometry_from(img);

        if elapsed % conductance_scaling_update_interval == 0 {
            let average = average_gradient_magnitude_squared(&snapshot, &scale)?;
            k = average * conductance_parameter * conductance_parameter * -2.0;
        }

        let iter = NeighborhoodIterator::<f64, _>::new(
            &snapshot,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )?;
        // Parallel over voxels, reading the **borrowed** window. Each voxel's update
        // reads its own window in `snapshot` — a separate image, cloned from `buf`
        // before the sweep — so no voxel reads another's new value: the sweep is a
        // map, not a recurrence. The iteration loop stays sequential; each pass
        // reads the whole previous one.
        //
        // The window used to be materialized into per-task scratch here, because
        // `gradient_update`/`curvature_update` took a `&Neighborhood`. They now take
        // a `Stencil`, so the borrowed window satisfies them and nothing is copied.
        //
        // Unlike `min_max_curvature_flow`'s sweep, this one has no `continue` to
        // preserve: every voxel takes the `+=` unconditionally, so there is no skip
        // that a `+= 0.0` could silently turn a stored `-0.0` into `+0.0`.
        let deltas: Vec<f64> = iter.par_map_window(|_, w| time_step * update(&w, &scale, k));
        parallel::for_each_mut(&mut buf, |i, v| *v += deltas[i]);
    }

    image_from_f64(pixel_id, &size, img, &buf)
}

/// `GradientAnisotropicDiffusionImageFilter`: the classic Perona-Malik
/// gradient-magnitude-driven anisotropic diffusion, in N dimensions.
///
/// SimpleITK defaults: `time_step = 0.125`, `conductance_parameter = 3.0`,
/// `conductance_scaling_update_interval = 1`, `number_of_iterations = 5`.
/// `use_image_spacing` is not exposed by SimpleITK; ITK's default is `true`.
///
/// An unstable `time_step` (one above [`stable_time_step_bound`]) is accepted
/// and computed with, as ITK's `itkWarningMacro`-and-proceed does.
///
/// Errors if `img`'s pixel type is not `Float32`/`Float64`
/// (`RealPixelIDTypeList`), or if `conductance_scaling_update_interval` is `0`
/// (the `.hxx` would divide by zero).
pub fn gradient_anisotropic_diffusion(
    img: &Image,
    time_step: f64,
    conductance_parameter: f64,
    conductance_scaling_update_interval: u32,
    number_of_iterations: u32,
    use_image_spacing: bool,
) -> Result<Image> {
    diffuse(
        img,
        time_step,
        conductance_parameter,
        conductance_scaling_update_interval,
        number_of_iterations,
        use_image_spacing,
        |w, scale, k| gradient_update(w, scale, k),
    )
}

/// `CurvatureAnisotropicDiffusionImageFilter`: the modified curvature
/// diffusion equation (MCDE), in N dimensions.
///
/// SimpleITK defaults: `time_step = 0.0625`, `conductance_parameter = 3.0`,
/// `conductance_scaling_update_interval = 1`, `number_of_iterations = 5`.
/// `use_image_spacing` is not exposed by SimpleITK; ITK's default is `true`.
///
/// An unstable `time_step` (one above [`stable_time_step_bound`]) is accepted
/// and computed with, as ITK's `itkWarningMacro`-and-proceed does.
///
/// Errors if `img`'s pixel type is not `Float32`/`Float64`
/// (`RealPixelIDTypeList`), or if `conductance_scaling_update_interval` is `0`.
pub fn curvature_anisotropic_diffusion(
    img: &Image,
    time_step: f64,
    conductance_parameter: f64,
    conductance_scaling_update_interval: u32,
    number_of_iterations: u32,
    use_image_spacing: bool,
) -> Result<Image> {
    diffuse(
        img,
        time_step,
        conductance_parameter,
        conductance_scaling_update_interval,
        number_of_iterations,
        use_image_spacing,
        |w, scale, k| curvature_update(w, scale, k),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variance(vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
    }

    /// A 9x9 image of deterministic high-frequency noise around a mean of 50.
    ///
    /// Deliberately *not* a checkerboard: a checkerboard has a zero centered
    /// difference at every pixel, so `average_gradient_magnitude_squared` would
    /// be `0`, `K` would be `0`, and every filter here would be the identity.
    fn ripple() -> Image {
        let n = 9;
        let data: Vec<f64> = (0..n * n)
            .map(|idx| 42.0 + ((idx * 37) % 17) as f64)
            .collect();
        Image::from_vec(&[n, n], data).unwrap()
    }

    // ---- pixel type / parameter guards ----

    #[test]
    fn non_real_pixel_type_is_rejected() {
        let img = Image::from_vec(&[3, 3], vec![1i16; 9]).unwrap();
        assert_eq!(
            gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 5, true),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
        assert_eq!(
            curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 1, 5, true),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
    }

    #[test]
    fn float32_input_yields_float32_output() {
        let img = Image::from_vec(&[5, 5], vec![1.0f32; 25]).unwrap();
        let out = gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 2, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        let out = curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 1, 2, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn zero_conductance_scaling_update_interval_is_rejected() {
        let img = ripple();
        assert_eq!(
            gradient_anisotropic_diffusion(&img, 0.125, 3.0, 0, 5, true),
            Err(FilterError::ZeroConductanceScalingUpdateInterval)
        );
        assert_eq!(
            curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 0, 5, true),
            Err(FilterError::ZeroConductanceScalingUpdateInterval)
        );
    }

    #[test]
    fn unstable_time_step_is_accepted_not_rejected() {
        let img = ripple();
        // 2-D, unit spacing: the bound is 1/2^3 = 0.125.
        assert_eq!(stable_time_step_bound(&img, true), 0.125);
        assert!(gradient_anisotropic_diffusion(&img, 10.0, 3.0, 1, 1, true).is_ok());
        assert!(curvature_anisotropic_diffusion(&img, 10.0, 3.0, 1, 1, true).is_ok());
    }

    #[test]
    fn stable_time_step_bound_tracks_min_spacing_and_dimension() {
        let mut img = Image::from_vec(&[3, 3, 3], vec![0.0f64; 27]).unwrap();
        img.set_spacing(&[2.0, 0.5, 3.0]).unwrap();
        // min spacing 0.5, dim 3 -> 0.5 / 2^4.
        assert_eq!(stable_time_step_bound(&img, true), 0.5 / 16.0);
        // spacing off -> 1 / 2^4.
        assert_eq!(stable_time_step_bound(&img, false), 1.0 / 16.0);
    }

    // ---- fixed points and identity ----

    #[test]
    fn constant_image_is_a_fixed_point() {
        let img = Image::from_vec(&[6, 5], vec![7.0f64; 30]).unwrap();
        for out in [
            gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 5, true).unwrap(),
            curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 1, 5, true).unwrap(),
        ] {
            assert!(out.to_f64_vec().unwrap().iter().all(|&v| v == 7.0));
        }
    }

    #[test]
    fn zero_conductance_is_a_fixed_point() {
        // K = 0 forces both conductance terms to 0.0 in the `.hxx`'s
        // `if (m_K != 0.0)` guard.
        let img = ripple();
        let before = img.to_f64_vec().unwrap();
        for out in [
            gradient_anisotropic_diffusion(&img, 0.125, 0.0, 1, 3, true).unwrap(),
            curvature_anisotropic_diffusion(&img, 0.0625, 0.0, 1, 3, true).unwrap(),
        ] {
            assert_eq!(out.to_f64_vec().unwrap(), before);
        }
    }

    #[test]
    fn zero_iterations_is_identity() {
        let img = ripple();
        let before = img.to_f64_vec().unwrap();
        for out in [
            gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 0, true).unwrap(),
            curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 1, 0, true).unwrap(),
        ] {
            assert_eq!(out.to_f64_vec().unwrap(), before);
            assert_eq!(out.pixel_id(), PixelId::Float64);
        }
    }

    // ---- hand-derived single-iteration stencil ----

    /// A 5x5 image, zero but for `v = 1` at the center, unit spacing.
    ///
    /// The average gradient magnitude squared is `accumulator / counter` with
    /// `counter = 25` and only the four axis neighbors of the hot pixel
    /// contributing `(v/2)²` each: `avg = 4·(1/4)·v² / 25 = v²/25`. Hence
    /// `K = −2·c²·v²/25` and `v²/K = −25/(2c²)`.
    ///
    /// For `gradient_update`, the diagonal neighbors get exactly zero (their
    /// `dx_forward`/`dx_backward` both vanish along both axes, and the update
    /// is a sum of those two terms scaled by conductances), the center gets
    /// `−4·v·E` and each axis neighbor `+v·E`, where `E = exp(−25/(2c²))`.
    #[test]
    fn gradient_hot_pixel_spreads_to_the_plus_stencil_only() {
        let n = 5;
        let mut data = vec![0.0f64; n * n];
        data[2 * n + 2] = 1.0;
        let img = Image::from_vec(&[n, n], data).unwrap();

        let c = 3.0;
        let dt = 0.125;
        let out = gradient_anisotropic_diffusion(&img, dt, c, 1, 1, true).unwrap();
        let vals = out.to_f64_vec().unwrap();

        let e = (-25.0 / (2.0 * c * c)).exp();
        let expect_center = 1.0 + dt * (-4.0 * e);
        let expect_axis = dt * e;

        for y in 0..n {
            for x in 0..n {
                let expected = match (x.abs_diff(2), y.abs_diff(2)) {
                    (0, 0) => expect_center,
                    (1, 0) | (0, 1) => expect_axis,
                    _ => 0.0,
                };
                assert!(
                    (vals[y * n + x] - expected).abs() < 1e-12,
                    "({x},{y}): {} != {expected}",
                    vals[y * n + x]
                );
            }
        }
        // Gradient AD's update is a divergence, so it conserves total mass.
        assert!((vals.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// Same image under `curvature_update`. Writing `g = sqrt(MIN_NORM + v²)`
    /// and `E = exp(−25/(2c²))`: the center has `speed = −4vE/g` and upwind
    /// `|∇u| = 2v`, so its update is `−8v²E/g`; each axis neighbor has
    /// `speed = +vE/g` and upwind `|∇u| = v`, so its update is `+v²E/g`; the
    /// diagonals have `speed = 0` and a zero upwind term. Unlike gradient AD,
    /// MCDE does not conserve mass.
    #[test]
    fn curvature_hot_pixel_spreads_to_the_plus_stencil_only() {
        let n = 5;
        let mut data = vec![0.0f64; n * n];
        data[2 * n + 2] = 1.0;
        let img = Image::from_vec(&[n, n], data).unwrap();

        let c = 3.0;
        let dt = 0.0625;
        let out = curvature_anisotropic_diffusion(&img, dt, c, 1, 1, true).unwrap();
        let vals = out.to_f64_vec().unwrap();

        let e = (-25.0 / (2.0 * c * c)).exp();
        let g = (MIN_NORM + 1.0).sqrt();
        let expect_center = 1.0 + dt * (-8.0 * e / g);
        let expect_axis = dt * (e / g);

        for y in 0..n {
            for x in 0..n {
                let expected = match (x.abs_diff(2), y.abs_diff(2)) {
                    (0, 0) => expect_center,
                    (1, 0) | (0, 1) => expect_axis,
                    _ => 0.0,
                };
                assert!(
                    (vals[y * n + x] - expected).abs() < 1e-12,
                    "({x},{y}): {} != {expected}",
                    vals[y * n + x]
                );
            }
        }
    }

    // ---- edge preservation ----

    /// The defining invariant of Perona-Malik: the conductance
    /// `exp(−g²/(2c²⟨|∇u|²⟩))` is strictly decreasing in the local gradient, so
    /// within a single image (one shared `K`) a high-contrast step loses a far
    /// smaller *fraction* of its contrast than a low-contrast step does.
    ///
    /// Comparing a step against a linear ramp of equal total magnitude would not
    /// test this: a linear ramp has zero second difference, so it is nearly a
    /// fixed point of any diffusion regardless of conductance. The contrast
    /// ratio below isolates the conductance term itself.
    #[test]
    fn gradient_ad_preserves_a_strong_edge_far_better_than_a_weak_one() {
        // 20 columns, 3 rows, constant along y: a weak pulse of height 2 and a
        // strong pulse of height 100, each 4 columns wide and separated from
        // everything else by 4 flat columns.
        let (w, h) = (20usize, 3usize);
        let profile: Vec<f64> = (0..w)
            .map(|x| match x {
                4..=7 => 2.0,
                12..=15 => 100.0,
                _ => 0.0,
            })
            .collect();
        let data: Vec<f64> = (0..w * h).map(|idx| profile[idx % w]).collect();
        let img = Image::from_vec(&[w, h], data).unwrap();

        let out = gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 5, true).unwrap();
        let vals = out.to_f64_vec().unwrap();
        let row = |x: usize| vals[w + x]; // middle row

        let weak_kept = (row(4) - row(3)) / 2.0;
        let strong_kept = (row(12) - row(11)) / 100.0;

        // Both edges share one `K = −2·c²·⟨|∇u|²⟩ ≈ −9004` (the mean is
        // dominated by the four `±100` jumps: `4·(100/2)²·3 rows / 60 pixels ≈
        // 500`). The weak edge then sees `exp(−2²/9004) ≈ 1` — it diffuses at
        // essentially the full isotropic rate — while the strong edge sees
        // `exp(−100²/9004) ≈ 0.33`. Measured: the weak edge keeps 0.374 of its
        // contrast, the strong edge 0.670.
        assert!(
            strong_kept > weak_kept,
            "conductance must decrease with gradient: weak {weak_kept}, strong {strong_kept}"
        );
        assert!(
            weak_kept < 0.45,
            "weak edge kept {weak_kept}; expected substantial diffusion"
        );
        assert!(
            strong_kept > 0.6,
            "strong edge kept only {strong_kept} of its contrast"
        );
    }

    // ---- conductance sweep: isotropic limit ----

    /// As `conductance → ∞`, `K → −∞` and every conductance term tends to
    /// `exp(0) = 1`, leaving the plain explicit heat equation; as
    /// `conductance → 0`, `K → 0⁻` and every conductance term tends to `0`,
    /// leaving the identity. Variance reduction is monotone between the two.
    #[test]
    fn large_conductance_smooths_isotropically_small_conductance_barely_moves() {
        let img = ripple();
        let v0 = variance(&img.to_f64_vec().unwrap());

        let small = gradient_anisotropic_diffusion(&img, 0.125, 0.01, 1, 3, true).unwrap();
        let large = gradient_anisotropic_diffusion(&img, 0.125, 1.0e4, 1, 3, true).unwrap();

        let v_small = variance(&small.to_f64_vec().unwrap());
        let v_large = variance(&large.to_f64_vec().unwrap());

        assert!(
            (v_small - v0).abs() / v0 < 1e-6,
            "small conductance changed variance from {v0} to {v_small}"
        );
        // Measured: 24.51 -> 1.49, a 94% reduction.
        assert!(
            v_large < 0.1 * v0,
            "large conductance only reduced variance from {v0} to {v_large}"
        );
    }

    // ---- spacing ----

    #[test]
    fn use_image_spacing_off_ignores_spacing() {
        let mut coarse = ripple();
        coarse.set_spacing(&[2.0, 3.0]).unwrap();
        let unit = ripple();

        let a = gradient_anisotropic_diffusion(&coarse, 0.02, 3.0, 1, 3, false).unwrap();
        let b = gradient_anisotropic_diffusion(&unit, 0.02, 3.0, 1, 3, false).unwrap();
        assert_eq!(a.to_f64_vec().unwrap(), b.to_f64_vec().unwrap());

        // ... and with spacing on, the same pixel data diffuses differently.
        let c = gradient_anisotropic_diffusion(&coarse, 0.02, 3.0, 1, 3, true).unwrap();
        assert!(
            c.to_f64_vec()
                .unwrap()
                .iter()
                .zip(b.to_f64_vec().unwrap())
                .any(|(x, y)| (x - y).abs() > 1e-9)
        );
    }

    #[test]
    fn use_image_spacing_on_matches_unit_spacing_when_spacing_is_unit() {
        let img = ripple();
        let a = curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 1, 3, true).unwrap();
        let b = curvature_anisotropic_diffusion(&img, 0.0625, 3.0, 1, 3, false).unwrap();
        assert_eq!(a.to_f64_vec().unwrap(), b.to_f64_vec().unwrap());
    }

    // ---- conductance scaling update interval ----

    /// `elapsed % interval == 0` gates the `K` refresh, so an interval of `2`
    /// reuses iteration 0's `K` for iteration 1 and must differ from an
    /// interval of `1` once the solution has moved.
    #[test]
    fn conductance_scaling_update_interval_gates_the_k_refresh() {
        let img = ripple();
        let every = gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 3, true).unwrap();
        let every_other = gradient_anisotropic_diffusion(&img, 0.125, 3.0, 2, 3, true).unwrap();
        assert!(
            every
                .to_f64_vec()
                .unwrap()
                .iter()
                .zip(every_other.to_f64_vec().unwrap())
                .any(|(a, b)| (a - b).abs() > 1e-12)
        );

        // A single iteration only ever uses the `elapsed == 0` refresh, so
        // every interval agrees there.
        let one_a = gradient_anisotropic_diffusion(&img, 0.125, 3.0, 1, 1, true).unwrap();
        let one_b = gradient_anisotropic_diffusion(&img, 0.125, 3.0, 7, 1, true).unwrap();
        assert_eq!(one_a.to_f64_vec().unwrap(), one_b.to_f64_vec().unwrap());
    }

    // ---- average gradient magnitude squared ----

    /// `counter` counts pixels, not pixel×axis, and boundary faces are included
    /// under zero-flux Neumann. For the hot-pixel image that makes the mean
    /// `v²/25` rather than `v²/50` (per-axis) or `v²/9` (interior only).
    #[test]
    fn average_gradient_magnitude_squared_averages_over_pixels_not_axes() {
        let n = 5;
        let mut data = vec![0.0f64; n * n];
        data[2 * n + 2] = 1.0;
        let img = Image::from_vec(&[n, n], data).unwrap();
        let avg = average_gradient_magnitude_squared(&img, &[1.0, 1.0]).unwrap();
        assert!((avg - 1.0 / 25.0).abs() < 1e-15);
    }

    /// A linear ramp along x with slope `s` per pixel has centered gradient `s`
    /// everywhere in the interior; under zero-flux Neumann the two boundary
    /// columns see a half-difference of `s/2`. With unit scale this pins the
    /// estimator exactly.
    #[test]
    fn average_gradient_magnitude_squared_uses_zero_flux_neumann_at_the_boundary() {
        let (w, h) = (4usize, 2usize);
        let s = 3.0;
        let data: Vec<f64> = (0..w * h).map(|i| s * (i % w) as f64).collect();
        let img = Image::from_vec(&[w, h], data).unwrap();

        // Per pixel: x-term is s² in the two interior columns and (s/2)² in the
        // two boundary columns; the y-term is 0 everywhere (rows are equal,
        // and zero-flux makes both rows' y-neighbors identical).
        let expected = h as f64 * (2.0 * s * s + 2.0 * (s / 2.0).powi(2)) / (w * h) as f64;
        let avg = average_gradient_magnitude_squared(&img, &[1.0, 1.0]).unwrap();
        assert!((avg - expected).abs() < 1e-12, "{avg} != {expected}");

        // Scale coefficients enter squared: halving them quarters the mean.
        let scaled = average_gradient_magnitude_squared(&img, &[0.5, 0.5]).unwrap();
        assert!((scaled - expected / 4.0).abs() < 1e-12);
    }
}

/// Thread-count parity pins for the diffusion sweep **and** for the one true
/// cross-pixel reduction in this crate's stencil work.
///
/// # Two different things are pinned here, and only one of them is a map
///
/// The sweep is a map: each voxel's delta is a function of its own window in the
/// snapshot, so parallelizing it cannot reach the arithmetic. `-0.0` does not
/// apply to it — unlike [`crate::min_max_curvature_flow`]'s sweep, there is no
/// `continue` to preserve; every voxel takes the `+=` unconditionally, so no skip
/// could be silently turned into a `+= 0.0` that flips a stored `-0.0`.
///
/// `average_gradient_magnitude_squared` is **not** a map. It sums one squared term
/// per pixel *and* axis over the whole volume into a single `f64`, and `f64`
/// addition is not associative: a rayon `fold`/`sum` would make the answer depend
/// on how the work happened to be split, which is exactly the thing this port
/// forbids. It goes through `parallel::map_rows_fold_in_order` instead — terms
/// computed in parallel, the `+=` replayed on one thread in pixel order — and
/// [`the_reduction_is_order_sensitive_and_thread_stable`] is the pin that this
/// buys anything: it asserts both that the sum genuinely re-associates on this
/// input (so a rayon fold *would* have moved it) and that the shipped
/// reduction does not move.
#[cfg(test)]
mod stencil_thread_parity {
    use super::*;
    use crate::stencil_test_support::{
        PIXELS, THREADS, assert_bits_eq, volume, window_sum_order_is_observable,
    };
    // The serial references below deliberately keep the OWNED window: they are
    // copies of the loops that were deleted, and those loops materialized a
    // `Neighborhood` per voxel. That the same `gradient_update`/`curvature_update`
    // bodies can still be driven from an owned window — instantiated here as
    // `gradient_update::<Neighborhood<f64>>` — is what makes them a valid reference
    // for the borrowed-window code they are pinning.
    use sitk_core::{Neighborhood, parallel};

    const TIME_STEP: f64 = 0.0625;
    const CONDUCTANCE: f64 = 3.0;
    const ITERATIONS: u32 = 3;

    // ---- the serial references: the exact loops that were deleted -----------

    /// `average_gradient_magnitude_squared`, as the serial loop computed it:
    /// one `accumulator +=` per pixel per axis, in pixel-then-axis order.
    fn average_gradient_magnitude_squared_serial(snapshot: &Image, scale: &[f64]) -> f64 {
        let dim = snapshot.dimension();
        let radius = vec![1usize; dim];
        let iter = NeighborhoodIterator::<f64, _>::new(
            snapshot,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();

        let mut accumulator = 0.0f64;
        let mut counter = 0usize;
        let mut off = vec![0i64; dim];

        for (_, nb) in iter {
            counter += 1;
            for (i, &s) in scale.iter().enumerate() {
                off[i] = 1;
                let plus = nb.at(&off);
                off[i] = -1;
                let minus = nb.at(&off);
                off[i] = 0;
                let val = (plus - minus) / -2.0 * s;
                accumulator += val * val;
            }
        }
        if counter == 0 {
            return 0.0;
        }
        accumulator / counter as f64
    }

    /// The exact serial `diffuse` loop that was deleted.
    fn diffuse_serial(
        img: &Image,
        time_step: f64,
        conductance_parameter: f64,
        interval: u32,
        iterations: u32,
        use_image_spacing: bool,
        update: fn(&Neighborhood<f64>, &[f64], f64) -> f64,
    ) -> Vec<f64> {
        let dim = img.dimension();
        let scale = scale_coefficients(img, use_image_spacing);
        let size = img.size().to_vec();
        let radius = vec![1usize; dim];
        let mut buf = img.to_f64_vec().unwrap();
        let mut k = 0.0f64;

        for elapsed in 0..iterations {
            let mut snapshot = Image::from_vec(&size, buf.clone()).unwrap();
            snapshot.copy_geometry_from(img);

            if elapsed % interval == 0 {
                let average = average_gradient_magnitude_squared_serial(&snapshot, &scale);
                k = average * conductance_parameter * conductance_parameter * -2.0;
            }

            let iter = NeighborhoodIterator::<f64, _>::new(
                &snapshot,
                &radius,
                ZeroFluxNeumannBoundaryCondition,
            )
            .unwrap();
            for ((_, nb), v) in iter.zip(buf.iter_mut()) {
                *v += time_step * update(&nb, &scale, k);
            }
        }
        buf
    }

    fn narrowed_like(img: &Image, values: &[f64]) -> Vec<f64> {
        image_from_f64(img.pixel_id(), img.size(), img, values)
            .unwrap()
            .to_f64_vec()
            .unwrap()
    }

    // ---- non-vacuity --------------------------------------------------------

    #[test]
    fn the_window_sum_order_is_observable() {
        let img = volume(PixelId::Float64);
        assert!(
            window_sum_order_is_observable(&img, &[1, 1, 1]),
            "no voxel changed bits when its window sum was reversed — this volume cannot \
             observe a re-association, so the sweep pins below would pass even on an update \
             that summed its window in a different order"
        );
    }

    /// The pin that earns `map_rows_fold_in_order` its place.
    ///
    /// Two assertions, and the first is what makes the second mean something:
    ///
    /// 1. **The reduction really does re-associate on this input.** Summing the
    ///    same per-pixel terms in reverse order gives a *different* `f64` — so a
    ///    rayon `fold`/`sum`, whose split depends on thread count and steal order,
    ///    could have landed on a different answer here. Without this, the second
    ///    assertion would hold trivially and prove nothing about the seam.
    ///
    /// 2. **The shipped reduction does not move.** `average_gradient_magnitude_-
    ///    squared` is bit-identical to the serial `accumulator +=` loop at 1, 4, 48
    ///    and 96 threads. It can be: the terms are computed in parallel into each
    ///    pixel's own row, and the `+=` is replayed on one thread over the rows in
    ///    index order, so the accumulator sees the identical addition sequence the
    ///    serial loop fed it — a fixed decomposition that is a function of the input
    ///    length alone, never of how many threads showed up.
    #[test]
    fn the_reduction_is_order_sensitive_and_thread_stable() {
        let img = volume(PixelId::Float64);
        let scale = scale_coefficients(&img, true);
        let dim = img.dimension();

        // (1) the sum is genuinely non-associative on this data
        let radius = vec![1usize; dim];
        let iter =
            NeighborhoodIterator::<f64, _>::new(&img, &radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        let mut terms = Vec::with_capacity(img.number_of_pixels() * dim);
        let mut off = vec![0i64; dim];
        for (_, nb) in iter {
            for (i, &s) in scale.iter().enumerate() {
                off[i] = 1;
                let plus = nb.at(&off);
                off[i] = -1;
                let minus = nb.at(&off);
                off[i] = 0;
                let val = (plus - minus) / -2.0 * s;
                terms.push(val * val);
            }
        }
        let forward: f64 = terms.iter().fold(0.0, |a, b| a + b);
        let backward: f64 = terms.iter().rev().fold(0.0, |a, b| a + b);
        assert_ne!(
            forward.to_bits(),
            backward.to_bits(),
            "summing the reduction's terms forwards and backwards gave the identical f64, so \
             this input cannot observe a re-association: a rayon fold would have passed here \
             too, and the thread-stability assertion below would prove nothing about the \
             ordered seam"
        );

        // (2) the shipped reduction is bit-identical to the serial fold, at every
        //     thread count
        let expected = average_gradient_magnitude_squared_serial(&img, &scale);
        for threads in THREADS {
            let got = parallel::with_threads(threads, || {
                average_gradient_magnitude_squared(&img, &scale)
            })
            .unwrap();
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "average_gradient_magnitude_squared moved with {threads} threads: {got:?} vs \
                 serial {expected:?}"
            );
        }
    }

    // ---- the pins -----------------------------------------------------------

    #[test]
    fn gradient_anisotropic_diffusion_is_bit_identical_at_every_thread_count() {
        for pixel in PIXELS {
            let img = volume(pixel);
            for interval in [1u32, 2] {
                let expected = narrowed_like(
                    &img,
                    &diffuse_serial(
                        &img,
                        TIME_STEP,
                        CONDUCTANCE,
                        interval,
                        ITERATIONS,
                        true,
                        gradient_update::<Neighborhood<f64>>,
                    ),
                );
                for threads in THREADS {
                    let got = parallel::with_threads(threads, || {
                        gradient_anisotropic_diffusion(
                            &img,
                            TIME_STEP,
                            CONDUCTANCE,
                            interval,
                            ITERATIONS,
                            true,
                        )
                    })
                    .unwrap()
                    .to_f64_vec()
                    .unwrap();
                    assert_bits_eq(
                        &got,
                        &expected,
                        &format!(
                            "gradient_anisotropic_diffusion({pixel:?}, interval={interval}, \
                             {threads} threads)"
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn curvature_anisotropic_diffusion_is_bit_identical_at_every_thread_count() {
        for pixel in PIXELS {
            let img = volume(pixel);
            let expected = narrowed_like(
                &img,
                &diffuse_serial(
                    &img,
                    TIME_STEP,
                    CONDUCTANCE,
                    1,
                    ITERATIONS,
                    true,
                    curvature_update::<Neighborhood<f64>>,
                ),
            );
            for threads in THREADS {
                let got = parallel::with_threads(threads, || {
                    curvature_anisotropic_diffusion(
                        &img,
                        TIME_STEP,
                        CONDUCTANCE,
                        1,
                        ITERATIONS,
                        true,
                    )
                })
                .unwrap()
                .to_f64_vec()
                .unwrap();
                assert_bits_eq(
                    &got,
                    &expected,
                    &format!("curvature_anisotropic_diffusion({pixel:?}, {threads} threads)"),
                );
            }
        }
    }
}
