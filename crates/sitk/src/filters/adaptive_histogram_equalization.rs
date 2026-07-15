//! `AdaptiveHistogramEqualizationImageFilter`: power-law adaptive histogram
//! equalization / unsharp masking over a per-pixel sliding window.
//!
//! Verified against
//! `Modules/Filtering/ImageStatistics/include/itkAdaptiveHistogramEqualizationImageFilter.h(.hxx)`,
//! `itkAdaptiveEqualizationHistogram.h` (the `GetValue`/`CumulativeFunction`
//! math) and `itkMovingHistogramImageFilter.hxx` (the
//! `AddPixel`/`AddBoundary` boundary accounting), and SimpleITK's
//! `Code/BasicFilters/yaml/AdaptiveHistogramEqualizationImageFilter.yaml`
//! (`Alpha`/`Beta` default `0.3`, `Radius` default `[5, 5, ...]`).

use crate::core::Image;
use crate::filters::error::{FilterError, Result};
use crate::filters::image_from_f64;
use std::collections::HashMap;

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `itk::Math::sgn`: `-1`/`0`/`+1` via `(x != 0) ? ((x > 0) ? 1 : -1) : 0`.
fn sgn(x: f64) -> f64 {
    if x != 0.0 {
        if x > 0.0 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

/// `itk::Function::AdaptiveEqualizationHistogram::CumulativeFunction`:
/// `0.5·sgn(u-v)·|2(u-v)|^alpha − beta·0.5·sgn(u-v)·|2(u-v)| + beta·u`.
fn cumulative_function(u: f64, v: f64, alpha: f64, beta: f64) -> f64 {
    let s = sgn(u - v);
    let ad = (2.0 * (u - v)).abs();
    0.5 * s * ad.powf(alpha) - beta * 0.5 * s * ad + beta * u
}

/// `AdaptiveHistogramEqualizationImageFilter`: per-pixel power-law adaptive
/// histogram equalization over an axis-aligned `radius` window (`2*radius[d]
/// + 1` pixels wide per axis), blended from classical histogram
/// equalization (`alpha=0`) toward an unsharp mask (`alpha=1`) by `alpha`,
/// and from that toward the identity by `beta`
/// (`itkAdaptiveHistogramEqualizationImageFilter.hxx`'s `ConfigureHistogram`,
/// `itkAdaptiveEqualizationHistogram.h`'s `GetValue`).
///
/// The normalizing extrema (`Minimum`/`Maximum` in `GetValue`) are the
/// *global* min/max of `img` (`MinimumMaximumImageFilter` run once over the
/// whole image in `BeforeThreadedGenerateData`), shared by every pixel's
/// window rather than recomputed locally.
///
/// Boundary handling: a window position that extends past the image edge
/// does **not** clamp the out-of-bounds offset to the nearest in-bounds
/// pixel (unlike e.g. [`crate::core::ZeroFluxNeumannBoundaryCondition`], or
/// [`crate::filters::expand`]'s corner sampling); instead, out-of-bounds offsets are
/// excluded from the local frequency map entirely and merely counted
/// (`AddBoundary`/`RemoveBoundary` in `itkMovingHistogramImageFilter.hxx`),
/// which shrinks the normalizing denominator (`kernel_size -
/// boundary_count`) so the pixels that *are* present get proportionally more
/// weight — "the boundary condition ignores the part of the neighborhood
/// outside the image, and over-weights the valid part of the neighborhood"
/// per the upstream class's own doc comment.
///
/// Errors if `radius.len()` doesn't match `img`'s dimension.
///
/// Fixed here (upstream bug §1.9): on a perfectly constant image
/// `input_minimum == input_maximum`, so upstream's normalized coordinate
/// `u = (pixel - minimum) / iscale - 0.5` is `0.0 / 0.0 == NaN`, which
/// poisons every term of `CumulativeFunction` — including the `beta * u`
/// term, since `0.0 * NaN == NaN` — and the whole output image comes out
/// `NaN` for *any* `alpha`/`beta`. This port instead returns the constant
/// image unchanged, which is the value the rest of the formula already
/// carries: with every window value equal to the center pixel, `u - v == 0`,
/// so `CumulativeFunction` reduces to `beta * u`, the window sum to
/// `beta * u`, and the reconstruction to
/// `iscale * (beta * u + 0.5) + minimum = iscale * (0.5 - 0.5 * beta) +
/// minimum`, which is exactly `minimum` — i.e. the constant — at
/// `iscale == 0`.
///
/// At `alpha = 1, beta = 1`, `CumulativeFunction` collapses algebraically to
/// `u - v` scaled so the window sum reduces to exactly `u` again (see the
/// module tests), reproducing the class doc's own claim: "If beta = 1 (and
/// alpha = 1), then the output image matches the input image."
///
/// Deliberate divergence: upstream's `AdaptiveEqualizationHistogram` declares
/// `u`/`v` (the normalized-to-`[-0.5, 0.5]` coordinates) as `RealType =
/// float`, so each is computed in `double` and then narrowed to `float32`
/// before use in `CumulativeFunction` and the output reconstruction — a
/// sub-ULP precision loss on every pixel. This port keeps `u`/`v`/`sum` in
/// `f64` throughout, matching the crate's established "widen to `f64`, never
/// narrow until the final output quantization" idiom (see e.g. [`crate::filters::clamp`]).
/// This changes no qualitative behavior (the `alpha=1,beta=1` exact-identity
/// collapse survives the switch from `f32` to `f64` intermediates unchanged)
/// — it only makes this port's results *more* precise than upstream's, never
/// differently shaped.
pub fn adaptive_histogram_equalization(
    img: &Image,
    radius: &[usize],
    alpha: f64,
    beta: f64,
) -> Result<Image> {
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }

    let vals = img.to_f64_vec()?;
    let (mut input_minimum, mut input_maximum) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &vals {
        input_minimum = input_minimum.min(v);
        input_maximum = input_maximum.max(v);
    }
    let iscale = input_maximum - input_minimum;
    let size = img.size();

    // A constant image carries no gray-level range to equalize: `iscale == 0`
    // makes upstream's `u = (pixel - minimum) / iscale - 0.5` a `0/0` NaN.
    // Every pixel already equals `input_minimum`, and that is the value the
    // reconstruction converges to — return the input unchanged.
    if iscale == 0.0 {
        return image_from_f64(img.pixel_id(), size, img, &vals);
    }

    let strides = strides(size);
    let kernel_size: usize = radius.iter().map(|&r| 2 * r + 1).product();

    // The (2r+1)^dim box of signed per-axis offsets around the center.
    let mut offsets: Vec<Vec<i64>> = Vec::with_capacity(kernel_size);
    let mut cur: Vec<i64> = radius.iter().map(|&r| -(r as i64)).collect();
    for _ in 0..kernel_size {
        offsets.push(cur.clone());
        for d in 0..dim {
            cur[d] += 1;
            if cur[d] > radius[d] as i64 {
                cur[d] = -(radius[d] as i64);
            } else {
                break;
            }
        }
    }

    let mut out = vec![0.0f64; vals.len()];
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut center = vec![0i64; dim];
        for d in 0..dim {
            center[d] = ((flat / strides[d]) % size[d]) as i64;
        }

        // Frequency map keyed by the raw pixel value's bit pattern (mirrors
        // itk::Function::AdaptiveEqualizationHistogram's
        // unordered_map<TInputPixel, size_t>).
        let mut map: HashMap<u64, usize> = HashMap::new();
        let mut boundary_count = 0usize;
        for offset in &offsets {
            let mut inside = true;
            let mut nflat = 0usize;
            for d in 0..dim {
                let n = center[d] + offset[d];
                if n < 0 || n as usize >= size[d] {
                    inside = false;
                    break;
                }
                nflat += n as usize * strides[d];
            }
            if inside {
                *map.entry(vals[nflat].to_bits()).or_insert(0) += 1;
            } else {
                boundary_count += 1;
            }
        }

        let ikernel = (kernel_size - boundary_count) as f64;
        let pixel = vals[flat];
        let u = (pixel - input_minimum) / iscale - 0.5;
        let mut sum = 0.0;
        for (&bits, &count) in &map {
            let v = (f64::from_bits(bits) - input_minimum) / iscale - 0.5;
            sum += count as f64 * cumulative_function(u, v, alpha, beta) / ikernel;
        }
        *slot = iscale * (sum + 0.5) + input_minimum;
    }

    image_from_f64(img.pixel_id(), size, img, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_is_the_identity_regardless_of_alpha_beta() {
        // Hand-derivation, for a constant image with value c (= 7 here):
        //   input_minimum = input_maximum = c, so iscale = 0.
        // Every window value v equals the center pixel value c, so u == v and
        //   s  = sgn(u - v) = sgn(0) = 0
        //   ad = |2(u - v)|  = 0
        //   CumulativeFunction = 0.5*0*0^alpha - beta*0.5*0*0 + beta*u
        //                      = beta*u                for every alpha, beta.
        // The counts in the frequency map sum to ikernel, so
        //   sum = ikernel * beta*u / ikernel = beta*u,
        // and the reconstruction is
        //   iscale*(beta*u + 0.5) + minimum
        //     = beta*(iscale*u) + 0.5*iscale + minimum.
        // With iscale*u = (c - minimum) - 0.5*iscale = -0.5*iscale, this is
        //   iscale*(0.5 - 0.5*beta) + minimum,
        // which at iscale = 0 is exactly minimum = c, for every alpha/beta.
        // (Upstream instead evaluates u as 0/0 = NaN and floods the image.)
        let img = Image::from_vec(&[4, 4], vec![7.0f64; 16]).unwrap();
        for (alpha, beta) in [(0.3, 0.3), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0)] {
            let out = adaptive_histogram_equalization(&img, &[1, 1], alpha, beta).unwrap();
            assert_eq!(
                out.to_f64_vec().unwrap(),
                vec![7.0; 16],
                "alpha={alpha} beta={beta}"
            );
        }
    }

    #[test]
    fn constant_image_identity_holds_for_negative_and_integer_pixels() {
        // The `iscale == 0` guard reconstructs the constant itself, not the
        // `minimum` of some rescaled range, so a negative constant survives.
        let img = Image::from_vec(&[3, 3], vec![-4.5f64; 9]).unwrap();
        let out = adaptive_histogram_equalization(&img, &[2, 2], 0.3, 0.3).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![-4.5; 9]);

        let img = Image::from_vec(&[2, 2], vec![200u8; 4]).unwrap();
        let out = adaptive_histogram_equalization(&img, &[1, 1], 0.3, 0.3).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![200.0; 4]);
        assert_eq!(out.pixel_id(), crate::core::PixelId::UInt8);
    }

    #[test]
    fn a_nonconstant_image_is_not_short_circuited_by_the_constant_guard() {
        // One differing pixel makes iscale = 1 > 0, so the full moving-window
        // path runs and the output is not the input.
        let mut data = vec![3.0f64; 9];
        data[4] = 4.0;
        let img = Image::from_vec(&[3, 3], data.clone()).unwrap();
        let out = adaptive_histogram_equalization(&img, &[1, 1], 0.3, 0.3).unwrap();
        let got = out.to_f64_vec().unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        assert_ne!(got, data);
    }

    #[test]
    fn alpha_one_beta_one_is_exact_identity() {
        // Class doc: "If beta = 1 (and alpha = 1), then the output image
        // matches the input image."
        let data: Vec<f64> = (0..64).map(|v| ((v * 37) % 97) as f64).collect();
        let img = Image::from_vec(&[8, 8], data.clone()).unwrap();
        let out = adaptive_histogram_equalization(&img, &[5, 5], 1.0, 1.0).unwrap();
        let got = out.to_f64_vec().unwrap();
        for (g, e) in got.iter().zip(&data) {
            assert!((g - e).abs() < 1e-9, "{got:?} vs {data:?}");
        }
    }

    #[test]
    fn radius_covering_whole_image_stops_changing_output_further() {
        // Once radius >= size-1 along every axis, every pixel's window
        // already reaches every other pixel in the image; growing the
        // radius further only adds boundary-discounted offsets, which
        // cancel out of ikernel = kernel_size - boundary_count, so the
        // output must be identical for any larger radius.
        let data: Vec<f64> = (0..16).map(|v| ((v * 13) % 17) as f64).collect();
        let img = Image::from_vec(&[4, 4], data).unwrap();
        let covering = adaptive_histogram_equalization(&img, &[3, 3], 0.3, 0.3).unwrap();
        let larger = adaptive_histogram_equalization(&img, &[8, 8], 0.3, 0.3).unwrap();
        let a = covering.to_f64_vec().unwrap();
        let b = larger.to_f64_vec().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn wrong_radius_length_is_rejected() {
        let img = Image::new(&[4, 4], crate::core::PixelId::Float64);
        assert!(matches!(
            adaptive_histogram_equalization(&img, &[5], 0.3, 0.3),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn output_pixel_type_follows_input() {
        let img = Image::from_vec(
            &[4, 4],
            vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();
        let out = adaptive_histogram_equalization(&img, &[1, 1], 0.3, 0.3).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::UInt8);
    }
}
