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

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::Image;
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
/// Note the IEEE-754 quirk this body carries for `x == NaN`: `NaN != 0.0` is
/// `true` (NaN compares unequal to everything), and `NaN > 0.0` is `false`,
/// so the `else` branch fires and `sgn(NaN) == -1` — not the `0` a "sign of
/// an undefined value" intuition might expect. This matters for
/// [`adaptive_histogram_equalization`]'s constant-image behavior (see its
/// doc comment).
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
/// pixel (unlike e.g. [`sitk_core::ZeroFluxNeumannBoundaryCondition`], or
/// [`crate::expand`]'s corner sampling); instead, out-of-bounds offsets are
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
/// On a perfectly constant image, `input_minimum == input_maximum`, so every
/// pixel's normalized coordinate `u = (pixel - minimum) / (maximum -
/// minimum) - 0.5` is `0.0 / 0.0 == NaN` — and that `NaN` poisons every term
/// of `CumulativeFunction` from there, *including* the `beta * u` term
/// unconditionally (IEEE-754 `0.0 * NaN == NaN`, not `0.0`, so this doesn't
/// even vanish at `beta == 0`). The output is therefore `NaN` everywhere on
/// a constant image, for *any* `alpha`/`beta` — not the identity a "fixed
/// point" intuition might expect. Upstream's `.hxx` has no special case for
/// `maximum == minimum`, so this port doesn't add one either, matching this
/// crate's existing precedent at [`crate::normalize`] (also an unguarded
/// `0.0 / 0.0` on a constant image).
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
/// narrow until the final output quantization" idiom (see e.g. [`crate::clamp`]).
/// This changes no qualitative behavior (the constant-image `NaN` and the
/// `alpha=1,beta=1` exact-identity collapse both survive the switch from
/// `f32` to `f64` intermediates unchanged) — it only makes this port's
/// results *more* precise than upstream's, never differently shaped.
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
    fn constant_image_is_nan_everywhere_regardless_of_alpha_beta() {
        let img = Image::from_vec(&[4, 4], vec![7.0f64; 16]).unwrap();
        for (alpha, beta) in [(0.3, 0.3), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0)] {
            let out = adaptive_histogram_equalization(&img, &[1, 1], alpha, beta).unwrap();
            assert!(
                out.to_f64_vec().unwrap().iter().all(|v| v.is_nan()),
                "alpha={alpha} beta={beta} did not produce all-NaN output"
            );
        }
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
        let img = Image::new(&[4, 4], sitk_core::PixelId::Float64);
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
        assert_eq!(out.pixel_id(), sitk_core::PixelId::UInt8);
    }
}
