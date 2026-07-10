//! ITK's min/max curvature flow pair, verified against
//! `Modules/Filtering/CurvatureFlow/include/`
//! (`itkMinMaxCurvatureFlowFunction.h`/`.hxx`,
//! `itkMinMaxCurvatureFlowImageFilter.h`/`.hxx`,
//! `itkBinaryMinMaxCurvatureFlowFunction.h`/`.hxx`,
//! `itkBinaryMinMaxCurvatureFlowImageFilter.h`/`.hxx`) and their shared bases
//! `itkCurvatureFlowFunction.hxx` / `itkCurvatureFlowImageFilter.hxx` plus
//! `Core/FiniteDifference/include/itkFiniteDifferenceFunction.hxx` and
//! `itkFiniteDifferenceImageFilter.hxx`.
//!
//! Both filters are `CurvatureFlowImageFilter` — the same explicit-Euler
//! `DenseFiniteDifferenceImageFilter` loop and the same discretized `κ|∇I|`
//! update ([`crate::denoise::curvature_flow`], whose
//! `curvature_flow_update` is reused here verbatim rather than forked) — with
//! the update *gated to one sign* before it is applied:
//!
//! * `MinMaxCurvatureFlowFunction::ComputeUpdate`: `max(κ|∇I|, 0)` when the
//!   ball average is `< threshold`, else `min(κ|∇I|, 0)`.
//! * `BinaryMinMaxCurvatureFlowFunction::ComputeUpdate`: the *opposite* gate —
//!   `min(κ|∇I|, 0)` when the ball average is `< threshold`, else
//!   `max(κ|∇I|, 0)`.
//!
//! and with a different source for `threshold`: [`min_max_curvature_flow`]
//! recomputes it per pixel from the neighborhood (the average intensity
//! perpendicular to the local gradient, at distance `stencil_radius` from the
//! center), while [`binary_min_max_curvature_flow`] takes it as a constant
//! parameter. Both compare it against the same "ball average": the inner
//! product of the neighborhood with `InitializeStencilOperator`'s hypersphere
//! mask (`1` where `Σ_d (i_d − r)² ≤ r²`, `0` elsewhere, normalized to sum to
//! one — `stencil_operator`). When the ungated update is exactly `0` the
//! `.hxx` returns early and neither the threshold nor the ball average is
//! computed; this port keeps that short-circuit.
//!
//! Upstream bugs corrected here rather than reproduced:
//!
//! * **§1.7 — the 3-D `ComputeThreshold` polar angle.** The `.hxx` rescales the
//!   gradient to length `r` (which the 2-D overload needs, since it uses the
//!   rescaled components directly as lattice offsets) and *then* computes
//!   `theta = acos(gradient[2])`. `gradient[2]` is `r · n_z` at that point, not
//!   the direction cosine `n_z`, so for `r >= 2` the polar angle is wrong; it
//!   would leave `acos`'s domain entirely were it not for the adjacent clamp of
//!   `gradient[2]` into `[-1, 1]`, which instead collapses every gradient with
//!   `|n_z| > 1/r` onto the pole `theta == 0`. The four sample points are
//!   `r · ∂n̂/∂θ` and `r · (1/sinθ) · ∂n̂/∂φ` for the *unit* gradient
//!   `n̂ = (sinθ cosφ, sinθ sinφ, cosθ)`, so the intended angle is unambiguous:
//!   this port computes `theta = acos(gradient[2] / r)`. `phi` needs no fix —
//!   it is `atan(gradient[1] / gradient[0])`, and the length-`r` factor cancels
//!   in the ratio. One note on the singular-`phi` branch: the `.hxx` overrides
//!   `phi` to `π/2` when `Math::AlmostEquals(gradient[0], PixelType{})` (a
//!   ~4-ULP / `0.1·eps` window around zero), whereas this port tests
//!   `gradient[0] == 0.0` exactly; the two decide differently only for a
//!   denormal (subnormal-magnitude nonzero) `gradient[0]`, so this is a
//!   pre-existing, numerically-inconsequential divergence and the code stays.
//!
//! * **§1.8 — the derivative scaling.** `MinMaxCurvatureFlowFunction::
//!   SetStencilRadius` widens the *finite difference function's* radius to
//!   `stencil_radius` in every axis, because the ball mask and the threshold
//!   sampling need that much neighborhood. `FiniteDifferenceFunction::
//!   ComputeNeighborhoodScales` then returns `ScaleCoefficients[d] /
//!   m_Radius[d]`, i.e. it assumes the derivatives are sampled `m_Radius[d]`
//!   pixels out. They are not: the inherited `CurvatureFlowFunction::
//!   ComputeUpdate` samples only the immediate `±1` neighbors. The update is
//!   quadratic in the scales, so upstream's `κ|∇I|` comes out
//!   `stencil_radius²` times smaller than plain
//!   [`crate::denoise::curvature_flow`]'s on the same image — exact only at
//!   `stencil_radius == 1`. This port uses `ScaleCoefficients[d]` for the
//!   derivatives, so the ungated update equals plain curvature flow's at every
//!   radius, which is what "min/max curvature flow" means: the curvature flow
//!   update, gated to one sign. `ComputeThreshold` is unaffected — all three
//!   overloads already read the raw `ScaleCoefficients`, never the
//!   neighborhood scales.
//!
//! Faithfully reproduced upstream quirks, each of them observable:
//!
//! * **`SetStencilRadius` clamps to `>= 1`.** `m_StencilRadius = (value > 1) ?
//!   value : 1`, so a `stencil_radius` of `0` behaves exactly like `1` and the
//!   `if (m_StencilRadius == 0)` early-outs inside the 2-D/3-D
//!   `ComputeThreshold` specializations are unreachable dead code (not ported).
//!
//! * **`ComputeThreshold` is dimension-dispatched.** For `ImageDimension == 2`
//!   the `.hxx` rotates the gradient by ±90° and reads the two rounded lattice
//!   points at distance `r`. For `ImageDimension == 3` it builds four points at
//!   distance `r` from the spherical angles `(theta, phi)` of the gradient.
//!   For every other dimension the `DispatchBase` overload brute-force scans
//!   the whole neighborhood and averages the pixels whose offset has norm
//!   `>= r` and whose cosine with the gradient is `< 0.262` in absolute value.
//!
//! * **A zero gradient makes the threshold `0`, not the center pixel.** All
//!   three `ComputeThreshold` overloads return `PixelType{}` when the gradient
//!   magnitude is exactly zero. (Such a pixel also has a zero update, so the
//!   early return above means the value is never actually used.)
//!
//! * **`Math::Round<SizeValueType>` is `floor(x + 0.5)`**, applied to the
//!   lattice-point coordinates; every argument is provably in `[0, 2r]` so no
//!   unsigned wrap occurs.
//!
//! Deliberate divergences from the `.hxx`:
//!
//! * **`time_step` is range-checked.** ITK's `CurvatureFlowFunction::
//!   ComputeGlobalTimeStep` returns the caller's step unexamined. This crate's
//!   [`crate::denoise::curvature_flow`] already rejects steps outside
//!   `[0, 1 / (2·Σ_d ScaleCoefficients[d]²)]` with
//!   [`FilterError::UnstableTimeStep`]; the same bound is enforced here. With
//!   §1.8 fixed the derivative scales no longer depend on `stencil_radius`, so
//!   the bound is exactly plain curvature flow's — and the one-sided gate only
//!   ever damps the scheme, so a step accepted here is at least as stable.
//!
//! * **Everything is computed in `f64`.** ITK accumulates the threshold, the
//!   ball average and the stencil mask in `PixelType` (`float` for a
//!   `Float32` image).
//!
//! Both SimpleITK YAMLs declare `pixel_types: RealPixelIDTypeList`, so a
//! non-`Float32`/`Float64` input is [`FilterError::RequiresRealPixelType`] and
//! the output pixel type equals the input's. SimpleITK defaults are
//! `TimeStep = 0.05`, `NumberOfIterations = 5`, `StencilRadius = 2`, and for the
//! binary variant `Threshold = 0`.

use crate::denoise::curvature_flow_update;
use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{
    Image, Neighborhood, NeighborhoodIterator, PixelId, ZeroFluxNeumannBoundaryCondition,
};

/// `itk::Math::Round<SizeValueType>`: round to nearest, halves upward
/// (`floor(x + 0.5)`). Every call site feeds it a value in `[0, 2·r]`.
fn itk_round(x: f64) -> usize {
    (x + 0.5).floor() as usize
}

/// `MinMaxCurvatureFlowFunction::InitializeStencilOperator`: a `(2r+1)^dim`
/// mask, dimension-0-fastest, holding `1/n` at each of the `n` offsets inside
/// the closed hypersphere of radius `r` (`Σ_d (counter[d] − r)² ≤ r²`) and `0`
/// elsewhere. The center is always inside, so `n >= 1` and the `numPixelsInSphere
/// != 0` guard in the `.hxx` never fires — it is kept here anyway.
fn stencil_operator(dim: usize, radius: usize) -> Vec<f64> {
    let span = 2 * radius + 1;
    let mut op = vec![0.0f64; span.pow(dim as u32)];
    let sqr_radius = (radius * radius) as i64;
    let mut counter = vec![0usize; dim];
    let mut num_pixels_in_sphere = 0usize;

    for slot in op.iter_mut() {
        let length: i64 = counter
            .iter()
            .map(|&c| {
                let d = c as i64 - radius as i64;
                d * d
            })
            .sum();
        if length <= sqr_radius {
            *slot = 1.0;
            num_pixels_in_sphere += 1;
        }

        // dimension-0-fastest carry, matching the `.hxx`'s `counter` walk
        for c in counter.iter_mut() {
            *c += 1;
            if *c == span {
                *c = 0;
            } else {
                break;
            }
        }
    }

    if num_pixels_in_sphere != 0 {
        for v in op.iter_mut() {
            *v /= num_pixels_in_sphere as f64;
        }
    }
    op
}

/// The centered, spacing-scaled gradient the three `ComputeThreshold` overloads
/// share: `0.5·(I[+e_d] − I[−e_d])·ScaleCoefficients[d]`. Note this uses the raw
/// `ScaleCoefficients`, *not* `ComputeNeighborhoodScales`'s `/radius[d]` form.
fn threshold_gradient(nb: &Neighborhood<f64>, dim: usize, coeff: &[f64]) -> Vec<f64> {
    let mut off = vec![0i64; dim];
    (0..dim)
        .map(|d| {
            off[d] = 1;
            let plus = nb.get(&off);
            off[d] = -1;
            let minus = nb.get(&off);
            off[d] = 0;
            0.5 * (plus - minus) * coeff[d]
        })
        .collect()
}

/// `MinMaxCurvatureFlowFunction::ComputeThreshold(const DispatchBase &, ...)`:
/// the brute-force scan used for every `ImageDimension` other than 2 and 3.
/// Averages the neighborhood pixels whose integer offset from the center has
/// Euclidean norm `>= radius` and whose cosine against the gradient is
/// `< 0.262` in absolute value (i.e. within ~74.8° of perpendicular). Returns
/// `0` when no pixel qualifies, and `0` when the gradient is exactly zero.
fn compute_threshold_generic(
    nb: &Neighborhood<f64>,
    dim: usize,
    radius: usize,
    coeff: &[f64],
) -> f64 {
    let gradient = threshold_gradient(nb, dim, coeff);
    let mut grad_magnitude: f64 = gradient.iter().map(|g| g * g).sum();
    if grad_magnitude == 0.0 {
        return 0.0;
    }
    grad_magnitude = grad_magnitude.sqrt();

    let span = 2 * radius + 1;
    let mut counter = vec![0usize; dim];
    let mut threshold = 0.0f64;
    let mut num_pixels = 0usize;

    for &value in nb.values() {
        let mut dot_product = 0.0f64;
        let mut vector_magnitude = 0.0f64;
        for (d, &g) in gradient.iter().enumerate() {
            let diff = counter[d] as f64 - radius as f64;
            dot_product += diff * g;
            vector_magnitude += diff * diff;
        }
        let vector_magnitude = vector_magnitude.sqrt();
        if vector_magnitude != 0.0 {
            dot_product /= grad_magnitude * vector_magnitude;
        }
        if vector_magnitude >= radius as f64 && dot_product.abs() < 0.262 {
            threshold += value;
            num_pixels += 1;
        }

        for c in counter.iter_mut() {
            *c += 1;
            if *c == span {
                *c = 0;
            } else {
                break;
            }
        }
    }

    if num_pixels > 0 {
        threshold /= num_pixels as f64;
    }
    threshold
}

/// `MinMaxCurvatureFlowFunction::ComputeThreshold(const Dispatch<2> &, ...)`:
/// rescale the gradient to length `radius`, rotate it ±90°, round both
/// endpoints onto the lattice, average the two pixels. `0` on a zero gradient.
fn compute_threshold_2d(nb: &Neighborhood<f64>, radius: usize, coeff: &[f64]) -> f64 {
    let mut gradient = threshold_gradient(nb, 2, coeff);
    let mut grad_magnitude = gradient[0] * gradient[0] + gradient[1] * gradient[1];
    if grad_magnitude == 0.0 {
        return 0.0;
    }
    grad_magnitude = grad_magnitude.sqrt() / radius as f64;
    gradient[0] /= grad_magnitude;
    gradient[1] /= grad_magnitude;

    let r = radius as f64;
    let span = 2 * radius + 1;
    let values = nb.values();

    let first = values[itk_round(r - gradient[1]) + span * itk_round(r + gradient[0])];
    let second = values[itk_round(r + gradient[1]) + span * itk_round(r - gradient[0])];
    (first + second) * 0.5
}

/// `MinMaxCurvatureFlowFunction::ComputeThreshold(const Dispatch<3> &, ...)`:
/// four points on the circle of radius `radius` that the `.hxx` *intends* to be
/// perpendicular to the gradient, averaged.
///
/// **§1.7 fixed here.** `gradient` is rescaled to length `radius` (the 2-D
/// overload needs that length for its rotated endpoints), but the polar angle
/// is the angle of the *unit* gradient, so `theta` is `acos(gradient[2] /
/// radius)`, not upstream's `acos(gradient[2])`. The clamp into `[-1, 1]` is
/// kept as a pure rounding guard — the quotient is in range by construction.
fn compute_threshold_3d(nb: &Neighborhood<f64>, radius: usize, coeff: &[f64]) -> f64 {
    let mut gradient = threshold_gradient(nb, 3, coeff);
    let mut grad_magnitude: f64 = gradient.iter().map(|g| g * g).sum();
    if grad_magnitude == 0.0 {
        return 0.0;
    }
    grad_magnitude = grad_magnitude.sqrt() / radius as f64;
    for g in gradient.iter_mut() {
        *g /= grad_magnitude;
    }

    let theta = (gradient[2] / radius as f64).clamp(-1.0, 1.0).acos();
    // `Math::AlmostEquals(gradient[0], PixelType{})` is a 4-ULP window around
    // zero, which for a zero reference means exact zero (or a denormal).
    let phi = if gradient[0] == 0.0 {
        std::f64::consts::PI * 0.5
    } else {
        (gradient[1] / gradient[0]).atan()
    };

    let r = radius as f64;
    let (cos_theta, sin_theta) = (theta.cos(), theta.sin());
    let (cos_phi, sin_phi) = (phi.cos(), phi.sin());

    let r_sin_theta = r * sin_theta;
    let r_cos_theta_cos_phi = r * cos_theta * cos_phi;
    let r_cos_theta_sin_phi = r * cos_theta * sin_phi;
    let r_sin_phi = r * sin_phi;
    let r_cos_phi = r * cos_phi;

    let span = 2 * radius + 1;
    let values = nb.values();
    let at = |x: usize, y: usize, z: usize| values[x + span * y + span * span * z];

    // angle = 0, 90, 180, 270 around the (intended) gradient axis
    let p1 = at(
        itk_round(r + r_cos_theta_cos_phi),
        itk_round(r + r_cos_theta_sin_phi),
        itk_round(r - r_sin_theta),
    );
    let p2 = at(itk_round(r - r_sin_phi), itk_round(r + r_cos_phi), radius);
    let p3 = at(
        itk_round(r - r_cos_theta_cos_phi),
        itk_round(r - r_cos_theta_sin_phi),
        itk_round(r + r_sin_theta),
    );
    let p4 = at(itk_round(r + r_sin_phi), itk_round(r - r_cos_phi), radius);

    (p1 + p2 + p3 + p4) * 0.25
}

/// `MinMaxCurvatureFlowFunction::ComputeThreshold(Dispatch<ImageDimension>(), it)`.
fn compute_threshold(nb: &Neighborhood<f64>, dim: usize, radius: usize, coeff: &[f64]) -> f64 {
    match dim {
        2 => compute_threshold_2d(nb, radius, coeff),
        3 => compute_threshold_3d(nb, radius, coeff),
        _ => compute_threshold_generic(nb, dim, radius, coeff),
    }
}

/// Which way the ball average gates the curvature update — the sole difference
/// between `MinMaxCurvatureFlowFunction::ComputeUpdate` and
/// `BinaryMinMaxCurvatureFlowFunction::ComputeUpdate`.
#[derive(Debug, Clone, Copy)]
enum Gate {
    /// `avg < ComputeThreshold(it) ? max(update, 0) : min(update, 0)`.
    MinMax,
    /// `avg < m_Threshold ? min(update, 0) : max(update, 0)`.
    Binary(f64),
}

/// The shared `CurvatureFlowImageFilter` solver loop, with `Gate`'s clamp
/// applied to each pixel's update.
fn min_max_flow(
    img: &Image,
    number_of_iterations: u32,
    time_step: f64,
    stencil_radius: usize,
    use_image_spacing: bool,
    gate: Gate,
) -> Result<Image> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }

    let dim = img.dimension();
    // `SetStencilRadius`: `m_StencilRadius = (value > 1) ? value : 1`.
    let stencil_radius = stencil_radius.max(1);

    // `FiniteDifferenceImageFilter::InitializeIteration`'s ScaleCoefficients.
    // These are also the derivative scales: `curvature_flow_update` samples the
    // `±1` neighbors, so the finite-difference step is one pixel regardless of
    // `stencil_radius`. Upstream instead reaches for `ComputeNeighborhoodScales`,
    // whose `ScaleCoefficients[d] / m_Radius[d]` divides by `stencil_radius` —
    // that is §1.8, fixed here (see the module doc).
    let coeff: Vec<f64> = img
        .spacing()
        .iter()
        .map(|&s| if use_image_spacing { 1.0 / s } else { 1.0 })
        .collect();

    let max_stable = 1.0 / (2.0 * coeff.iter().map(|s| s * s).sum::<f64>());
    if !(0.0..=max_stable).contains(&time_step) {
        return Err(FilterError::UnstableTimeStep {
            time_step,
            max_stable,
        });
    }

    let operator = stencil_operator(dim, stencil_radius);
    let size = img.size().to_vec();
    let radius = vec![stencil_radius; dim];
    let mut buf = img.to_f64_vec()?;

    for _ in 0..number_of_iterations {
        let mut snapshot = Image::from_vec(&size, buf.clone())?;
        snapshot.copy_geometry_from(img);
        let iter = NeighborhoodIterator::<f64, _>::new(
            &snapshot,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )?;
        for ((_, nb), v) in iter.zip(buf.iter_mut()) {
            let update = curvature_flow_update(&nb, dim, &coeff);
            if update == 0.0 {
                continue;
            }
            let avg_value: f64 = nb
                .values()
                .iter()
                .zip(&operator)
                .map(|(pixel, weight)| pixel * weight)
                .sum();
            let gated = match gate {
                Gate::MinMax => {
                    let threshold = compute_threshold(&nb, dim, stencil_radius, &coeff);
                    if avg_value < threshold {
                        update.max(0.0)
                    } else {
                        update.min(0.0)
                    }
                }
                Gate::Binary(threshold) => {
                    if avg_value < threshold {
                        update.min(0.0)
                    } else {
                        update.max(0.0)
                    }
                }
            };
            *v += time_step * gated;
        }
    }

    image_from_f64(pixel_id, &size, img, &buf)
}

/// `MinMaxCurvatureFlowImageFilter`: curvature flow whose update is clamped to
/// `max(κ|∇I|, 0)` where the `stencil_radius`-ball average is below the local
/// perpendicular-direction average, and to `min(κ|∇I|, 0)` where it is not.
/// Structures larger than `stencil_radius` therefore hold still while smaller
/// ones are flattened.
///
/// SimpleITK defaults: `time_step = 0.05`, `number_of_iterations = 5`,
/// `stencil_radius = 2`. A `stencil_radius` of `0` is clamped up to `1`, as
/// `SetStencilRadius` does. `use_image_spacing` is not exposed by SimpleITK;
/// ITK's default is `true`.
///
/// Errors if `img`'s pixel type is not `Float32`/`Float64`
/// (`RealPixelIDTypeList`), or if `time_step` is outside
/// `[0, 1 / (2·Σ_d scale[d]²)]` — see the module doc for that bound, which this
/// crate adds and ITK does not.
pub fn min_max_curvature_flow(
    img: &Image,
    number_of_iterations: u32,
    time_step: f64,
    stencil_radius: usize,
    use_image_spacing: bool,
) -> Result<Image> {
    min_max_flow(
        img,
        number_of_iterations,
        time_step,
        stencil_radius,
        use_image_spacing,
        Gate::MinMax,
    )
}

/// `BinaryMinMaxCurvatureFlowImageFilter`: as [`min_max_curvature_flow`], but
/// the gate compares the `stencil_radius`-ball average against the caller's
/// `threshold` — the value that separates the two classes of an essentially
/// binary image — and the two branches are swapped: `min(κ|∇I|, 0)` below the
/// threshold, `max(κ|∇I|, 0)` at or above it.
///
/// SimpleITK defaults: `time_step = 0.05`, `number_of_iterations = 5`,
/// `stencil_radius = 2`, `threshold = 0.0`.
///
/// Errors under the same two conditions as [`min_max_curvature_flow`].
pub fn binary_min_max_curvature_flow(
    img: &Image,
    number_of_iterations: u32,
    time_step: f64,
    stencil_radius: usize,
    threshold: f64,
    use_image_spacing: bool,
) -> Result<Image> {
    min_max_flow(
        img,
        number_of_iterations,
        time_step,
        stencil_radius,
        use_image_spacing,
        Gate::Binary(threshold),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise::curvature_flow;

    const EPS: f64 = 1e-12;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1e-9, "pixel {i}: {a} != {e}");
        }
    }

    /// A 6x6 zero image with a 2x2 block of `1.0` at x,y in {2,3}.
    fn block_2x2() -> Image {
        let mut data = vec![0.0f64; 36];
        for y in 2..4 {
            for x in 2..4 {
                data[x + 6 * y] = 1.0;
            }
        }
        Image::from_vec(&[6, 6], data).unwrap()
    }

    fn neighborhood_at(img: &Image, radius: usize, center: &[usize]) -> Neighborhood<f64> {
        let r = vec![radius; img.dimension()];
        NeighborhoodIterator::<f64, _>::new(img, &r, ZeroFluxNeumannBoundaryCondition)
            .unwrap()
            .neighborhood_at(center)
    }

    // ---- stencil_operator (InitializeStencilOperator) ----

    #[test]
    fn stencil_operator_radius_2_in_2d_is_the_13_point_ball() {
        let op = stencil_operator(2, 2);
        assert_eq!(op.len(), 25);
        let inside: Vec<usize> = op
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, _)| i)
            .collect();
        // dx^2+dy^2 <= 4 over dx,dy in [-2,2]: 13 offsets.
        assert_eq!(inside.len(), 13);
        for &i in &inside {
            assert!((op[i] - 1.0 / 13.0).abs() < EPS);
        }
        // corners (±2,±2) and the (±2,±1)/(±1,±2) ring are outside.
        for corner in [0usize, 4, 20, 24] {
            assert_eq!(op[corner], 0.0);
        }
        assert!((op.iter().sum::<f64>() - 1.0).abs() < EPS);
    }

    #[test]
    fn stencil_operator_radius_1_is_the_cross_and_sums_to_one() {
        let op = stencil_operator(2, 1);
        assert_eq!(op.len(), 9);
        let expected = 1.0 / 5.0;
        // center + 4 axis neighbors; the 4 diagonals have d^2 = 2 > 1.
        for (i, &v) in op.iter().enumerate() {
            let want = if matches!(i, 1 | 3 | 4 | 5 | 7) {
                expected
            } else {
                0.0
            };
            assert!((v - want).abs() < EPS, "slot {i}");
        }
    }

    #[test]
    fn stencil_operator_center_is_always_inside_even_at_radius_1_in_3d() {
        let op = stencil_operator(3, 1);
        assert_eq!(op.len(), 27);
        assert!(op[13] > 0.0);
        // 1 center + 6 face neighbors
        assert_eq!(op.iter().filter(|&&v| v != 0.0).count(), 7);
    }

    // ---- ComputeThreshold ----

    #[test]
    fn threshold_2d_and_generic_agree_for_an_axis_aligned_gradient() {
        // I(x,y) = x^2 on a 5x5 grid. At the center (2,2) the gradient is
        // (4, 0), so both overloads must land exactly on the offsets (0, ±2),
        // whose value is I = 4. Widening the generic scan to also accept the
        // (±1, ±2) ring would give 14/3 instead.
        let data: Vec<f64> = (0..25).map(|i| ((i % 5) * (i % 5)) as f64).collect();
        let img = Image::from_vec(&[5, 5], data).unwrap();
        let nb = neighborhood_at(&img, 2, &[2, 2]);
        let coeff = [1.0, 1.0];

        assert!((compute_threshold_2d(&nb, 2, &coeff) - 4.0).abs() < 1e-9);
        assert!((compute_threshold_generic(&nb, 2, 2, &coeff) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn threshold_2d_rotates_the_gradient_by_ninety_degrees() {
        // I(x,y) = x + y: gradient (0.5, 0.5) -> rescaled to length 2 it is
        // (√2, √2), so the sampled offsets are round(2∓√2) = 1 and 3 in each
        // axis, i.e. (-1, +1) and (+1, -1) from the center. Both have I = 4.
        let data: Vec<f64> = (0..25).map(|i| (i % 5 + i / 5) as f64).collect();
        let img = Image::from_vec(&[5, 5], data).unwrap();
        let nb = neighborhood_at(&img, 2, &[2, 2]);
        assert!((compute_threshold_2d(&nb, 2, &[1.0, 1.0]) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn threshold_is_zero_when_the_gradient_vanishes() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let nb = neighborhood_at(&img, 2, &[2, 2]);
        // Not `7.0` (the center pixel) — the `.hxx` returns `PixelType{}`.
        assert_eq!(compute_threshold_2d(&nb, 2, &[1.0, 1.0]), 0.0);
        assert_eq!(compute_threshold_generic(&nb, 2, 2, &[1.0, 1.0]), 0.0);

        let img3 = Image::from_vec(&[5, 5, 5], vec![7.0f64; 125]).unwrap();
        let nb3 = neighborhood_at(&img3, 2, &[2, 2, 2]);
        assert_eq!(compute_threshold_3d(&nb3, 2, &[1.0, 1.0, 1.0]), 0.0);
    }

    /// §1.7 fix: `theta` is the polar angle of the **unit** gradient, so the
    /// four sample points really are perpendicular to it.
    ///
    /// `I(x,y,z) = x² + z` on a 5×5×5 grid, `coeff = 1`, `r = 2`. At `(2,2,2)`:
    ///   `g = (0.5·(9−1), 0.5·(4−4), 0.5·(6−4)) = (4, 0, 1)`, `|g| = √17`.
    /// The `.hxx` divides by `|g|/r`, leaving `g = (4,0,1)·2/√17 =
    /// (1.9402850, 0, 0.4850713)` — length exactly `r = 2`.
    ///
    /// Corrected: `n_z = 0.4850713 / 2 = 0.24253563`, so
    /// `theta = acos(0.24253563) = 1.3258177` rad, `cosθ = 0.24253563`,
    /// `sinθ = √(1 − 0.0588235) = 0.97014250`. `phi = atan(0/1.94) = 0`, so
    /// `cosφ = 1`, `sinφ = 0`. Check: `(sinθcosφ, sinθsinφ, cosθ) =
    /// (0.9701425, 0, 0.2425356) = (4,0,1)/√17` — the unit gradient. ✓
    ///
    /// With `r·sinθ = 1.9402850` and `r·cosθ·cosφ = 0.48507125`, and
    /// `Math::Round(x) = floor(x + 0.5)`:
    ///   - P1 `(round(2+0.48507), round(2+0), round(2−1.94029))` = `(2,2,0)` → `I = 4+0 = 4`
    ///   - P2 `(round(2−0), round(2+2), 2)` = `(2,4,2)` → `I = 4+2 = 6`
    ///   - P3 `(round(2−0.48507), round(2−0), round(2+1.94029))` = `(2,2,4)` → `I = 4+4 = 8`
    ///   - P4 `(round(2+0), round(2−2), 2)` = `(2,0,2)` → `I = 4+2 = 6`
    ///
    /// Average `(4+6+8+6)/4 = 6.0`.
    ///
    /// Upstream instead feeds `0.4850713` straight into `acos`, giving
    /// `theta = 1.0642064` rad; its points are `(3,2,0)=9`, `(2,4,2)=6`,
    /// `(1,2,4)=5`, `(2,0,2)=6` → `6.5`, asserted absent below.
    #[test]
    fn threshold_3d_polar_angle_uses_the_unit_gradient() {
        let mut data = vec![0.0f64; 125];
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    data[x + 5 * y + 25 * z] = (x * x + z) as f64;
                }
            }
        }
        let img = Image::from_vec(&[5, 5, 5], data).unwrap();
        let nb = neighborhood_at(&img, 2, &[2, 2, 2]);
        let t = compute_threshold_3d(&nb, 2, &[1.0, 1.0, 1.0]);
        assert!((t - 6.0).abs() < 1e-9, "got {t}");
        assert!((t - 6.5).abs() > 0.4, "upstream's unnormalized acos value");
    }

    /// §1.7 fix, pole case: a gradient along `+z` alone gives `n_z = 1` exactly,
    /// hence `theta = 0`, and the four points are the in-plane ring at `z = r`.
    /// (Upstream reached `theta == 0` here too, but only because its clamp of
    /// `gradient[2] = r` into `[-1, 1]` rescued `acos` from a NaN.)
    ///
    /// `I(x,y,z) = z² + (x−2)⁴` on 5×5×5, `r = 2`. At `(2,2,2)`:
    ///   `g_x = 0.5·((3−2)⁴ − (1−2)⁴) = 0.5·(1−1) = 0`
    ///   `g_y = 0`, `g_z = 0.5·(3² − 1²) = 4`.
    /// So `|g| = 4`, rescaled `g = (0, 0, 2)`, `n_z = 2/2 = 1`, `theta = 0`,
    /// `cosθ = 1`, `sinθ = 0`. `gradient[0] == 0` so `phi = π/2`: `sinφ = 1`,
    /// `cosφ = 0`. Then `r·sinθ = 0`, `r·cosθ·cosφ = 0`, `r·cosθ·sinφ = 2`,
    /// `r·sinφ = 2`, `r·cosφ = 0`, giving
    ///   P1 `(2,4,2)`, P2 `(0,2,2)`, P3 `(2,0,2)`, P4 `(4,2,2)`.
    /// Every point has `z = 2`, which is what pins `theta == 0`; the `(x−2)⁴`
    /// term makes the ring's four values distinguishable rather than uniform:
    /// `I(2,4,2) = 4`, `I(0,2,2) = 4+16 = 20`, `I(2,0,2) = 4`,
    /// `I(4,2,2) = 4+16 = 20`. Average `(4+20+4+20)/4 = 12.0`.
    #[test]
    fn threshold_3d_polar_gradient_samples_the_ring_in_the_center_plane() {
        let mut data = vec![0.0f64; 125];
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let dx = x as i64 - 2;
                    data[x + 5 * y + 25 * z] = (z * z) as f64 + (dx * dx * dx * dx) as f64;
                }
            }
        }
        let img = Image::from_vec(&[5, 5, 5], data).unwrap();
        let nb = neighborhood_at(&img, 2, &[2, 2, 2]);
        let t = compute_threshold_3d(&nb, 2, &[1.0, 1.0, 1.0]);
        assert!(t.is_finite());
        assert!((t - 12.0).abs() < 1e-9, "got {t}");
    }

    /// `r == 1` is the one radius at which upstream's `acos(gradient[2])` was
    /// already fed a true direction cosine, so the §1.7 fix must leave it
    /// unchanged. `I(x,y,z) = z` on 3×3×3: `g = (0, 0, 0.5)`, `|g| = 0.5`,
    /// rescaled to length `1` it is `(0,0,1)`; `n_z = 1/1 = 1`, `theta = 0`.
    /// The four points are the x/y ring at `z = 1`, all holding `I = 1`.
    #[test]
    fn threshold_3d_at_radius_one_is_unchanged_by_the_polar_angle_fix() {
        let mut data = vec![0.0f64; 27];
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    data[x + 3 * y + 9 * z] = z as f64;
                }
            }
        }
        let img = Image::from_vec(&[3, 3, 3], data).unwrap();
        let nb = neighborhood_at(&img, 1, &[1, 1, 1]);
        assert!((compute_threshold_3d(&nb, 1, &[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-9);
    }

    /// §1.7 fix, invariant form: for an arbitrary (non-axial, non-polar)
    /// gradient the four *unrounded* sample points must be perpendicular to the
    /// gradient and at distance `r` from the center. This is what upstream's
    /// unnormalized `theta` broke and what no single hand-computed average can
    /// pin on its own.
    ///
    /// Recompute the offsets exactly as `compute_threshold_3d` does, from the
    /// same gradient, and check `offset · g == 0` and `|offset| == r`.
    #[test]
    fn threshold_3d_sample_offsets_are_perpendicular_to_the_gradient() {
        let r = 3.0f64;
        // An arbitrary gradient with all three components distinct and nonzero.
        let raw = [2.0f64, -5.0, 3.0];
        let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        // The `.hxx` rescales to length r before the angles are taken.
        let g: Vec<f64> = raw.iter().map(|v| v * r / norm).collect();

        let theta = (g[2] / r).clamp(-1.0, 1.0).acos();
        let phi = (g[1] / g[0]).atan();
        let (ct, st) = (theta.cos(), theta.sin());
        let (cp, sp) = (phi.cos(), phi.sin());

        let offsets = [
            [r * ct * cp, r * ct * sp, -r * st],
            [-r * sp, r * cp, 0.0],
            [-r * ct * cp, -r * ct * sp, r * st],
            [r * sp, -r * cp, 0.0],
        ];
        for off in offsets {
            let dot = off[0] * raw[0] + off[1] * raw[1] + off[2] * raw[2];
            assert!(dot.abs() < 1e-9, "offset {off:?} not perpendicular: {dot}");
            let len = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
            assert!((len - r).abs() < 1e-9, "offset {off:?} has length {len}");
        }
    }

    // ---- pixel type gate (RealPixelIDTypeList) ----

    #[test]
    fn min_max_rejects_a_non_real_pixel_type() {
        let img = Image::from_vec(&[5, 5], vec![1i16; 25]).unwrap();
        assert!(matches!(
            min_max_curvature_flow(&img, 1, 0.05, 2, true),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        ));
    }

    #[test]
    fn binary_rejects_a_non_real_pixel_type() {
        let img = Image::from_vec(&[5, 5], vec![1u8; 25]).unwrap();
        assert!(matches!(
            binary_min_max_curvature_flow(&img, 1, 0.05, 2, 0.0, true),
            Err(FilterError::RequiresRealPixelType(PixelId::UInt8))
        ));
    }

    #[test]
    fn output_pixel_type_equals_input_pixel_type() {
        let img = Image::from_vec(&[5, 5], vec![1.0f32; 25]).unwrap();
        assert_eq!(
            min_max_curvature_flow(&img, 1, 0.05, 2, true)
                .unwrap()
                .pixel_id(),
            PixelId::Float32
        );
        assert_eq!(
            binary_min_max_curvature_flow(&img, 1, 0.05, 2, 0.0, true)
                .unwrap()
                .pixel_id(),
            PixelId::Float32
        );
    }

    // ---- time_step guard ----

    /// §1.8 fix: the derivative scales are the raw `ScaleCoefficients`, so the
    /// stability bound `1 / (2·Σ_d scale[d]²)` no longer depends on
    /// `stencil_radius`. 2-D, unit spacing: `Σ_d scale[d]² = 1 + 1 = 2`, so
    /// `max_stable = 1/4 = 0.25` at *every* radius — plain curvature flow's
    /// bound. (Upstream's radius-scaled scales gave `0.25·r²`, i.e. `1.0` at
    /// `r = 2`, so `time_step = 0.3` was accepted there and is rejected here.)
    #[test]
    fn unstable_time_step_is_rejected_with_the_plain_curvature_flow_bound() {
        let img = Image::from_vec(&[5, 5], vec![1.0f64; 25]).unwrap();
        for radius in [1usize, 2, 3] {
            match min_max_curvature_flow(&img, 1, 0.3, radius, true).unwrap_err() {
                FilterError::UnstableTimeStep {
                    time_step,
                    max_stable,
                } => {
                    assert_eq!(time_step, 0.3);
                    assert!((max_stable - 0.25).abs() < EPS, "radius {radius}");
                }
                other => panic!("expected UnstableTimeStep, got {other:?}"),
            }
        }
        // 0.25 itself is accepted (the bound is inclusive) at every radius.
        for radius in [1usize, 2, 3] {
            assert!(min_max_curvature_flow(&img, 1, 0.25, radius, true).is_ok());
        }
    }

    #[test]
    fn negative_time_step_is_rejected() {
        let img = Image::from_vec(&[5, 5], vec![1.0f64; 25]).unwrap();
        assert!(matches!(
            min_max_curvature_flow(&img, 1, -0.01, 2, true),
            Err(FilterError::UnstableTimeStep { .. })
        ));
        assert!(matches!(
            binary_min_max_curvature_flow(&img, 1, -0.01, 2, 0.0, true),
            Err(FilterError::UnstableTimeStep { .. })
        ));
    }

    // ---- fixed points forced by the curvature formula ----

    #[test]
    fn zero_iterations_is_the_identity() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        assert_close(
            &min_max_curvature_flow(&img, 0, 0.05, 2, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &img.to_f64_vec().unwrap(),
        );
        assert_close(
            &binary_min_max_curvature_flow(&img, 0, 0.05, 2, 3.0, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &img.to_f64_vec().unwrap(),
        );
    }

    #[test]
    fn constant_image_is_a_fixed_point() {
        // magnitudeSqr == 0 < 1e-9 everywhere -> update 0 -> gate never runs.
        let img = Image::from_vec(&[7, 7], vec![3.5f64; 49]).unwrap();
        assert_close(
            &min_max_curvature_flow(&img, 5, 0.05, 2, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &[3.5; 49],
        );
        assert_close(
            &binary_min_max_curvature_flow(&img, 5, 0.05, 2, 3.5, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &[3.5; 49],
        );
    }

    #[test]
    fn linear_ramp_is_a_fixed_point() {
        // Planar level sets: every secderiv and crossderiv is 0, so
        // `update == 0` and the flow cannot move regardless of the gate.
        let data: Vec<f64> = (0..49).map(|i| (i % 7) as f64).collect();
        let img = Image::from_vec(&[7, 7], data.clone()).unwrap();
        assert_close(
            &min_max_curvature_flow(&img, 5, 0.05, 2, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &data,
        );
        assert_close(
            &binary_min_max_curvature_flow(&img, 5, 0.05, 2, 3.0, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &data,
        );
    }

    #[test]
    fn single_hot_pixel_is_a_fixed_point_under_both_flows() {
        // Derived from `CurvatureFlowFunction::ComputeUpdate`, not assumed:
        // at the hot pixel every first derivative is 0 (magnitudeSqr < 1e-9);
        // at an axis neighbor, say (3,2), only firstderiv_x and secderiv_x are
        // nonzero, so `temp` for i=x is secderiv_y = 0 and the i=y term carries
        // firstderiv_y^2 = 0 — update 0. At a diagonal neighbor both first
        // derivatives vanish. So the whole image is stationary, and min/max
        // curvature flow cannot shrink an isolated single pixel any more than
        // plain curvature flow can.
        let mut data = vec![0.0f64; 25];
        data[12] = 1.0;
        let img = Image::from_vec(&[5, 5], data.clone()).unwrap();

        assert_close(
            &min_max_curvature_flow(&img, 5, 0.05, 2, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &data,
        );
        assert_close(
            &curvature_flow(&img, 5, 0.05, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &data,
        );
    }

    #[test]
    fn one_dimensional_image_is_a_fixed_point() {
        // In 1D `temp` sums over an empty set and there are no cross terms, so
        // `update` is identically 0 — the `DispatchBase` threshold is never
        // even reached.
        let data: Vec<f64> = vec![0.0, 5.0, 1.0, 9.0, 2.0, 0.0, 7.0];
        let img = Image::from_vec(&[7], data.clone()).unwrap();
        assert_close(
            &min_max_curvature_flow(&img, 3, 0.4, 2, true)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            &data,
        );
    }

    // ---- the gate ----

    /// §1.8 fix: the derivative scales are `ScaleCoefficients[d]` (here `1.0`,
    /// since `use_image_spacing == false`), *not* `ScaleCoefficients[d] / r`.
    ///
    /// Hand-derived at the block corner `(2,2)`, `scale = (1, 1)`. Reading
    /// `block_2x2`: `I(2,2) = I(3,2) = I(2,3) = I(3,3) = 1`, everything else 0.
    ///   `f_x = 0.5·(I(3,2) − I(1,2)) = 0.5·(1 − 0) = 0.5`, likewise `f_y = 0.5`
    ///   `s_x = I(3,2) − 2·I(2,2) + I(1,2) = 1 − 2 + 0 = −1`, likewise `s_y = −1`
    ///   `c_xy = 0.25·(I(1,1) − I(1,3) − I(3,1) + I(3,3)) = 0.25·(0−0−0+1) = 0.25`
    ///   `magnitudeSqr = 0.25 + 0.25 = 0.5`
    ///   `update = (s_y·f_x² + s_x·f_y² − 2·f_x·f_y·c_xy) / magnitudeSqr`
    ///          `= (−0.25 − 0.25 − 0.125) / 0.5 = −1.25`
    ///
    /// Upstream's radius-scaled `scale = 0.5` gives `f = (0.25, 0.25)`,
    /// `s = (−0.25, −0.25)`, `c = 0.0625`, `magnitudeSqr = 0.125`, hence
    /// `update = −0.0390625 / 0.125 = −0.3125` — smaller by exactly `r² = 4`.
    ///
    /// The gate is unchanged by the fix: the ball average is `4/13 > 0`, and
    /// the perpendicular threshold reads the two background pixels at `(1,3)`
    /// and `(3,1)` → `0`, so `avg >= threshold` selects `min(update, 0)` and
    /// the negative update survives. By symmetry all four block pixels move by
    /// the same amount; every other pixel has `update == 0` (at an axis
    /// neighbour like `(1,2)` only `f_x` and `s_x` are nonzero, so both terms
    /// vanish; at a diagonal neighbour both first derivatives vanish).
    #[test]
    fn min_max_shrinks_a_two_by_two_block_and_leaves_the_background_alone() {
        let img = block_2x2();
        let time_step = 0.1;
        let out = min_max_curvature_flow(&img, 1, time_step, 2, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        let mut expected = img.to_f64_vec().unwrap();
        for y in 2..4 {
            for x in 2..4 {
                // 1.0 + 0.1 * (-1.25) = 0.875
                expected[x + 6 * y] = 0.875;
            }
        }
        assert_close(&out, &expected);
    }

    /// §1.8 fix, stated as the invariant it restores: the ungated update *is*
    /// plain curvature flow's, at every `stencil_radius`. Forcing the gate open
    /// with an extreme `binary_min_max_curvature_flow` threshold must therefore
    /// reproduce `curvature_flow`'s one-sided image identically for `r = 1`,
    /// `2` and `3`. Upstream's radius-scaled derivatives made the `r = 2` and
    /// `r = 3` results `4×` and `9×` smaller than the `r = 1` one.
    ///
    /// `threshold = -1e30` is below every ball average, so `avg < threshold` is
    /// false everywhere and the max-gate applies uniformly.
    #[test]
    fn the_ungated_update_is_radius_independent_and_equals_plain_curvature_flow() {
        let data: Vec<f64> = (0..49).map(|i| ((i * 7 + i / 7 * 3) % 5) as f64).collect();
        let img = Image::from_vec(&[7, 7], data.clone()).unwrap();
        let time_step = 0.1;

        let plain = curvature_flow(&img, 1, time_step, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let expect_max: Vec<f64> = data
            .iter()
            .zip(&plain)
            .map(|(&i, &p)| i + (p - i).max(0.0))
            .collect();
        // The test only means something if the plain flow actually moved.
        assert!(data.iter().zip(&plain).any(|(&i, &p)| (p - i).abs() > 1e-6));

        for radius in [1usize, 2, 3] {
            let out = binary_min_max_curvature_flow(&img, 1, time_step, radius, -1e30, true)
                .unwrap()
                .to_f64_vec()
                .unwrap();
            assert_close(&out, &expect_max);
        }
    }

    #[test]
    fn binary_threshold_straddling_the_ball_average_flips_the_gate() {
        // The ball average at each of the four block pixels is 4/13 = 0.3077.
        // The update there is -1.25 (derived in
        // `min_max_shrinks_a_two_by_two_block_and_leaves_the_background_alone`).
        let img = block_2x2();
        let time_step = 0.1;

        // threshold above the ball average -> min-gate -> the negative update
        // survives, exactly as the min/max variant's local threshold produced.
        let below = binary_min_max_curvature_flow(&img, 1, time_step, 2, 0.5, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let mut expected = img.to_f64_vec().unwrap();
        for y in 2..4 {
            for x in 2..4 {
                // 1.0 + 0.1 * (-1.25) = 0.875
                expected[x + 6 * y] = 0.875;
            }
        }
        assert_close(&below, &expected);

        // threshold below the ball average -> max-gate -> the negative update
        // is clamped to 0 and the image is stationary.
        let above = binary_min_max_curvature_flow(&img, 1, time_step, 2, 0.2, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_close(&above, &img.to_f64_vec().unwrap());
    }

    #[test]
    fn binary_threshold_extremes_reduce_to_one_sided_curvature_flow() {
        // The ungated update is bit-for-bit `curvature_flow`'s (§1.8), so an
        // extreme threshold reduces this filter to one-sided curvature flow.
        // A threshold above every ball average forces the min-gate everywhere;
        // one below every ball average forces the max-gate. Radius-independence
        // of the update is pinned separately by
        // `the_ungated_update_is_radius_independent_and_equals_plain_curvature_flow`.
        let data: Vec<f64> = (0..49).map(|i| ((i * 7 + i / 7 * 3) % 5) as f64).collect();
        let img = Image::from_vec(&[7, 7], data.clone()).unwrap();
        let time_step = 0.1;

        let plain = curvature_flow(&img, 1, time_step, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        let min_side = binary_min_max_curvature_flow(&img, 1, time_step, 1, 1e30, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let max_side = binary_min_max_curvature_flow(&img, 1, time_step, 1, -1e30, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        let expect_min: Vec<f64> = data
            .iter()
            .zip(&plain)
            .map(|(&i, &p)| i + (p - i).min(0.0))
            .collect();
        let expect_max: Vec<f64> = data
            .iter()
            .zip(&plain)
            .map(|(&i, &p)| i + (p - i).max(0.0))
            .collect();

        assert_close(&min_side, &expect_min);
        assert_close(&max_side, &expect_max);
        // The test only means something if the plain flow actually moved.
        assert!(data.iter().zip(&plain).any(|(&i, &p)| (p - i).abs() > 1e-6));
    }

    // ---- stencil radius bounds ----

    /// `SetStencilRadius`'s `(value > 1) ? value : 1` clamp, plus a guard that
    /// the test is not vacuous — radius 2 really does give a different answer.
    ///
    /// With §1.8 fixed the *update* is radius-independent, so the radii can only
    /// differ through the **gate** (the ball average and, for the min/max
    /// variant, the perpendicular threshold). `block_2x2` no longer separates
    /// them, so this uses an image built to make the gate flip. 7×7, all zero
    /// except `I(3,3) = 0.5`, `I(4,3) = 1`, `I(3,1) = I(3,5) = 1`.
    ///
    /// Update at the center `(3,3)`, `scale = (1,1)`:
    ///   `f_x = 0.5·(I(4,3) − I(2,3)) = 0.5`, `f_y = 0.5·(I(3,4) − I(3,2)) = 0`
    ///   `s_x = 1 − 2·0.5 + 0 = 0`, `s_y = 0 − 2·0.5 + 0 = −1`
    ///   `c_xy = 0.25·(I(2,2) − I(2,4) − I(4,2) + I(4,4)) = 0`
    ///   `magnitudeSqr = 0.25`
    ///   `update = (s_y·f_x² + s_x·f_y² − 0) / 0.25 = −0.25 / 0.25 = −1`
    ///
    /// Gate at `r = 1`: ball = the 5-point cross,
    /// `avg = (0.5 + 0 + 1 + 0 + 0)/5 = 0.3`. The gradient `(0.5, 0)` rescaled
    /// to length 1 is `(1, 0)`, so the two perpendicular points are `(3,4)` and
    /// `(3,2)`, both 0 → `threshold = 0`. `avg >= threshold` → `min(−1, 0) = −1`.
    ///
    /// Gate at `r = 2`: ball = the 13-point disc; the members it covers with a
    /// nonzero value are `(3,3) = 0.5` at `(0,0)`, `(4,3) = 1` at `(1,0)`, and
    /// `(3,1) = (3,5) = 1` at `(0,∓2)` — so `avg = 3.5/13 = 0.2692`. The
    /// gradient rescaled to length 2 is `(2, 0)`, so the perpendicular points
    /// are `(3,5)` and `(3,1)`, both 1 → `threshold = 1`. Now
    /// `avg < threshold` → `max(−1, 0) = 0`.
    ///
    /// With `time_step = 0.2` and one iteration: `I(3,3)` becomes
    /// `0.5 + 0.2·(−1) = 0.3` at radius 1, and stays `0.5` at radius 2.
    #[test]
    fn stencil_radius_zero_is_clamped_to_one() {
        let at = |x: usize, y: usize| x + 7 * y;
        let mut data = vec![0.0f64; 49];
        data[at(3, 3)] = 0.5;
        data[at(4, 3)] = 1.0;
        data[at(3, 1)] = 1.0;
        data[at(3, 5)] = 1.0;
        let img = Image::from_vec(&[7, 7], data).unwrap();

        let zero = min_max_curvature_flow(&img, 1, 0.2, 0, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let one = min_max_curvature_flow(&img, 1, 0.2, 1, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_close(&zero, &one);

        let two = min_max_curvature_flow(&img, 1, 0.2, 2, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        // Hand-derived center pixel, the site of the gate flip.
        assert!((one[at(3, 3)] - 0.3).abs() < 1e-9, "{}", one[at(3, 3)]);
        assert!((two[at(3, 3)] - 0.5).abs() < 1e-9, "{}", two[at(3, 3)]);
    }

    #[test]
    fn stencil_radius_larger_than_the_image_still_runs_under_zero_flux() {
        // radius 4 on a 3x3 image: every neighborhood read past the edge is
        // clamped to the nearest voxel, as ITK's ZeroFluxNeumann does.
        let img = Image::from_vec(&[3, 3], vec![2.0f64; 9]).unwrap();
        let out = min_max_curvature_flow(&img, 3, 0.05, 4, true).unwrap();
        assert_close(&out.to_f64_vec().unwrap(), &[2.0; 9]);

        let data: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 0.0];
        let img = Image::from_vec(&[3, 3], data).unwrap();
        let out = min_max_curvature_flow(&img, 3, 0.05, 4, true).unwrap();
        assert!(out.to_f64_vec().unwrap().iter().all(|v| v.is_finite()));
    }

    // ---- spacing ----

    #[test]
    fn isotropic_spacing_scales_the_update_by_the_inverse_square() {
        // scale[d] = 1/spacing[d] (§1.8: no `/r` factor). Doubling an isotropic
        // spacing halves every scale, and the update is quadratic in them, so
        // the applied delta is quartered. The gate is unaffected: rescaling the gradient
        // uniformly changes neither its direction (so `ComputeThreshold` picks
        // the same lattice points) nor the ball average.
        let img_unit = block_2x2();
        let mut img_spaced = block_2x2();
        img_spaced.set_spacing(&[2.0, 2.0]).unwrap();

        let time_step = 0.1;
        let unit = min_max_curvature_flow(&img_unit, 1, time_step, 2, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let spaced = min_max_curvature_flow(&img_spaced, 1, time_step, 2, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();

        let base = img_unit.to_f64_vec().unwrap();
        for i in 0..base.len() {
            let delta_unit = unit[i] - base[i];
            let delta_spaced = spaced[i] - base[i];
            assert!(
                (delta_spaced - delta_unit / 4.0).abs() < 1e-12,
                "pixel {i}: {delta_spaced} != {}",
                delta_unit / 4.0
            );
        }
        assert!(base.iter().zip(&unit).any(|(b, u)| (u - b).abs() > 1e-6));
    }
}
