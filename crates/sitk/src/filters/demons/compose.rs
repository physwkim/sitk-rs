//! Vector-field composition: `WarpVectorImageFilter` over
//! `VectorLinearInterpolateNearestNeighborExtrapolateImageFunction`, and the
//! `ExponentialDisplacementFieldImageFilter` built on top of it.
//!
//! `DiffeomorphicDemonsRegistrationFilter` is the only caller. Its update rule
//! is `s <- s ∘ exp(u)`, which in this representation is
//!
//! ```text
//! e            = exp(u)
//! s_next[i]    = s(p_i + e[i]) + e[i]
//! ```
//!
//! where `p_i` is pixel `i`'s physical point and `s(·)` is the field
//! interpolated at a physical point.
//!
//! # The interpolator extrapolates rather than padding
//!
//! `VectorLinearInterpolateNearestNeighborExtrapolateImageFunction` overrides
//! every `IsInsideBuffer` to return `true`
//! (itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h:99-121),
//! so `WarpVectorImageFilter`'s `m_EdgePaddingValue` branch
//! (itkWarpVectorImageFilter.hxx:171-186) is dead: a point outside the buffer is
//! never padded with zero, it is nearest-neighbour extrapolated. The clamp
//! happens on the base index, with the fractional distance forced to `0`
//! (the .hxx's lines 45-68), so the result is the field evaluated at the
//! continuous index clamped into `[0, size - 1]` on every axis.

use super::field::Field;
use super::geometry::Geometry;

/// `VectorLinearInterpolateNearestNeighborExtrapolateImageFunction::EvaluateAtContinuousIndex`
/// (the .hxx's lines 34-122), writing into `output`.
///
/// The `2^dim` corners are visited in `counter` order with bit `d` selecting
/// axis `d`'s upper neighbour, a corner of zero overlap is not read at all
/// (so the clamped `base + 1` never leaves the buffer), and the loop exits as
/// soon as the accumulated overlap reaches exactly `1.0`.
fn interpolate(field: &Field, cindex: &[f64], output: &mut [f64]) {
    let dim = field.dimension();
    let mut base = vec![0i64; dim];
    let mut distance = vec![0.0f64; dim];

    for d in 0..dim {
        let end = field.size[d] as i64 - 1;
        let floor = cindex[d].floor() as i64;
        if floor >= 0 {
            if floor < end {
                base[d] = floor;
                distance[d] = cindex[d] - floor as f64;
            } else {
                base[d] = end;
            }
        }
    }

    output.fill(0.0);
    let mut total_overlap = 0.0f64;
    let mut neighbor = vec![0usize; dim];

    for counter in 0..(1usize << dim) {
        let mut overlap = 1.0f64;
        let mut upper = counter;
        for d in 0..dim {
            if upper & 1 == 1 {
                neighbor[d] = (base[d] + 1) as usize;
                overlap *= distance[d];
            } else {
                neighbor[d] = base[d] as usize;
                overlap *= 1.0 - distance[d];
            }
            upper >>= 1;
        }

        if overlap != 0.0 {
            let value = field.vector_at(field.offset(&neighbor));
            for (component, &v) in output.iter_mut().zip(value) {
                *component += overlap * v;
            }
            total_overlap += overlap;
        }

        if total_overlap == 1.0 {
            break;
        }
    }
}

/// `WarpVectorImageFilter::ThreadedGenerateData`
/// (itkWarpVectorImageFilter.hxx:150-189): `out[i] = input(p_i + displacement[i])`.
///
/// `input` and `displacement` share `geometry`, as every use in this family
/// does — the warper's output origin, spacing and direction are set from the
/// update buffer, which carries the output field's geometry.
pub(crate) fn warp(input: &Field, displacement: &Field, geometry: &Geometry) -> Field {
    let dim = input.dimension();
    let mut output = Field::zeros(&input.size);
    let mut index = vec![0usize; dim];

    for pixel in 0..input.number_of_pixels() {
        input.multi_index(pixel, &mut index);
        let mut point = geometry.index_to_physical_point(&index);
        for (coordinate, &offset) in point.iter_mut().zip(displacement.vector_at(pixel)) {
            *coordinate += offset;
        }
        let cindex = geometry.physical_point_to_continuous_index(&point);
        interpolate(
            input,
            &cindex,
            &mut output.data[pixel * dim..pixel * dim + dim],
        );
    }

    output
}

/// `ExponentialDisplacementFieldImageFilter::GenerateData`'s automatic count
/// (itkExponentialDisplacementFieldImageFilter.hxx:70-116), with
/// `m_ComputeInverse` at its default `false`.
///
/// The rationale is that the first-order approximation `exp(Φ/2^N) = Φ/2^N`
/// must be diffeomorphic, i.e. `max(norm(Φ))/2^N < 0.5 · minimum pixel spacing`.
///
/// # Two quirks
///
/// * A field of all zeros does not give `0` iterations. `maxnorm2 == 0` takes
///   the `NumericTraits<double>::min()` branch — `DBL_MIN`, a *positive*
///   denormal — which passes `numiterfloat >= 0.0` and truncates to `1`. Pinned
///   by `a_zero_field_still_runs_one_squaring_step`.
/// * The comment says "take the ceil", but the code truncates `numiterfloat +
///   1.0`. That is `ceil` only when `numiterfloat` is not an integer; at an
///   exact integer it is `numiterfloat + 1`. Pinned by
///   `an_exact_integer_iteration_count_is_rounded_up_anyway`.
fn automatic_number_of_iterations(input: &Field, geometry: &Geometry, maximum: u32) -> u32 {
    let mut maxnorm2 = 0.0f64;
    for pixel in 0..input.number_of_pixels() {
        let norm2: f64 = input.vector_at(pixel).iter().map(|v| v * v).sum();
        if norm2 > maxnorm2 {
            maxnorm2 = norm2;
        }
    }

    let mut minimum_spacing = geometry.spacing[0];
    for &spacing in &geometry.spacing[1..] {
        if spacing < minimum_spacing {
            minimum_spacing = spacing;
        }
    }
    maxnorm2 /= minimum_spacing * minimum_spacing;

    let numiterfloat = if maxnorm2 > 0.0 {
        2.0 + 0.5 * maxnorm2.ln() / std::f64::consts::LN_2
    } else {
        f64::MIN_POSITIVE
    };

    if numiterfloat >= 0.0 {
        // `static_cast<unsigned int>(numiterfloat + 1.0)`, then thresholded.
        // The cast is undefined in C++ once the value passes `UINT_MAX`; Rust's
        // saturating `as` gives `u32::MAX`, which the `min` immediately clamps.
        ((numiterfloat + 1.0) as u32).min(maximum)
    } else {
        0
    }
}

/// `ExponentialDisplacementFieldImageFilter::GenerateData`
/// (itkExponentialDisplacementFieldImageFilter.hxx:60-209) with
/// `m_ComputeInverse` at its default `false`: scaling and squaring.
///
/// `v <- input / 2^N`, then `N` times `v <- v + v ∘ (Id + v)`.
///
/// With `automatic` clear, `N` is `maximum_number_of_iterations` verbatim; at
/// `N == 0` the filter is the identity (its `m_Caster` branch, lines 122-134),
/// which is exactly the first-order update `exp(u) = u`.
///
/// # Divergence from C++
///
/// ITK scales by `1 << numiter` on `int`, which is undefined behaviour for
/// `numiter >= 31`. The automatic count can reach `515` on a field of huge
/// displacements, and `SetMaximumNumberOfIterations` accepts any `unsigned`.
/// This port computes `2^numiter` in `f64`, which agrees with `1 << numiter`
/// for every `numiter` where the shift is defined.
pub(crate) fn exponential(
    input: &Field,
    geometry: &Geometry,
    automatic: bool,
    maximum_number_of_iterations: u32,
) -> Field {
    let numiter = if automatic {
        automatic_number_of_iterations(input, geometry, maximum_number_of_iterations)
    } else {
        maximum_number_of_iterations
    };

    if numiter == 0 {
        return input.clone();
    }

    // `m_Divider`: the first-order approximation.
    let scale = 2.0f64.powi(numiter as i32);
    let mut output = Field {
        data: input.data.iter().map(|v| v / scale).collect(),
        size: input.size.clone(),
    };

    // `m_Warper` then `m_Adder`, `numiter` times: `out = out + out ∘ (Id + out)`.
    for _ in 0..numiter {
        let warped = warp(&output, &output, geometry);
        for (component, &w) in output.data.iter_mut().zip(&warped.data) {
            *component += w;
        }
    }

    output
}

#[cfg(test)]
mod tests;
