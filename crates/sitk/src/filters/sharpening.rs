//! Edge-enhancement filters that combine a smoothed/second-derivative
//! version of an image with the original: `itkUnsharpMaskImageFilter.h(.hxx)`
//! and `itkLaplacianSharpeningImageFilter.h(.hxx)` (both `ITKImageFeature`).

use crate::core::compensated::{CompensatedSum, compensated_sum};
use crate::core::{Image, PixelId};
use crate::filters::error::{FilterError, Result};
use crate::filters::image_from_f64;
use crate::filters::recursive_gaussian::recursive_gaussian;

/// An `f64` copy of `img`'s pixels with `img`'s geometry, used so a filter's
/// internal (smoothed/second-derivative) computation stays full-precision
/// instead of narrowing to `img`'s own pixel type midway through.
fn scratch_f64(img: &Image) -> Result<Image> {
    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec()?)?;
    scratch.copy_geometry_from(img);
    Ok(scratch)
}

// ---- unsharp_mask -----------------------------------------------------------

/// `NumericTraits<T>::NonpositiveMin()`: for an integer type this is the
/// type's minimum; for a float type it is the most negative finite value
/// (`-max()`), not `-infinity`.
fn nonpositive_min(target: PixelId) -> f64 {
    match target {
        PixelId::UInt8 | PixelId::VectorUInt8 => u8::MIN as f64,
        PixelId::Int8 | PixelId::VectorInt8 => i8::MIN as f64,
        PixelId::UInt16 | PixelId::VectorUInt16 => u16::MIN as f64,
        PixelId::Int16 | PixelId::VectorInt16 => i16::MIN as f64,
        PixelId::UInt32 | PixelId::VectorUInt32 => u32::MIN as f64,
        PixelId::Int32 | PixelId::VectorInt32 => i32::MIN as f64,
        PixelId::UInt64 | PixelId::VectorUInt64 => u64::MIN as f64,
        PixelId::Int64 | PixelId::VectorInt64 => i64::MIN as f64,
        PixelId::Float32 | PixelId::ComplexFloat32 | PixelId::VectorFloat32 => -(f32::MAX as f64),
        PixelId::Float64 | PixelId::ComplexFloat64 | PixelId::VectorFloat64 => -f64::MAX,
    }
}

/// `NumericTraits<T>::max()`.
fn numeric_max(target: PixelId) -> f64 {
    match target {
        PixelId::UInt8 | PixelId::VectorUInt8 => u8::MAX as f64,
        PixelId::Int8 | PixelId::VectorInt8 => i8::MAX as f64,
        PixelId::UInt16 | PixelId::VectorUInt16 => u16::MAX as f64,
        PixelId::Int16 | PixelId::VectorInt16 => i16::MAX as f64,
        PixelId::UInt32 | PixelId::VectorUInt32 => u32::MAX as f64,
        PixelId::Int32 | PixelId::VectorInt32 => i32::MAX as f64,
        PixelId::UInt64 | PixelId::VectorUInt64 => u64::MAX as f64,
        PixelId::Int64 | PixelId::VectorInt64 => i64::MAX as f64,
        PixelId::Float32 | PixelId::ComplexFloat32 | PixelId::VectorFloat32 => f32::MAX as f64,
        PixelId::Float64 | PixelId::ComplexFloat64 | PixelId::VectorFloat64 => f64::MAX,
    }
}

/// Replicates the documented (not literal C++, which is undefined)
/// `Clamp == false` narrowing for an integer output type: SimpleITK's
/// `UnsharpMaskImageFilter.yaml` doc says "casting to output pixel format is
/// done using C++ defaults, meaning that values are not clamped but rather
/// wrap around e.g. 260 -> 4 (unsigned char)". `v.trunc() as i128` mirrors
/// truncating a real value to an integer (safe: `i128` covers every value
/// this filter's arithmetic can realistically produce), and the second `as`
/// wraps into the narrower type with 2's-complement truncation, matching
/// the documented wraparound.
fn wrap_to_pixel_type(target: PixelId, v: f64) -> f64 {
    match target {
        PixelId::UInt8 | PixelId::VectorUInt8 => (v.trunc() as i128 as u8) as f64,
        PixelId::Int8 | PixelId::VectorInt8 => (v.trunc() as i128 as i8) as f64,
        PixelId::UInt16 | PixelId::VectorUInt16 => (v.trunc() as i128 as u16) as f64,
        PixelId::Int16 | PixelId::VectorInt16 => (v.trunc() as i128 as i16) as f64,
        PixelId::UInt32 | PixelId::VectorUInt32 => (v.trunc() as i128 as u32) as f64,
        PixelId::Int32 | PixelId::VectorInt32 => (v.trunc() as i128 as i32) as f64,
        PixelId::UInt64 | PixelId::VectorUInt64 => (v.trunc() as i128 as u64) as f64,
        PixelId::Int64 | PixelId::VectorInt64 => (v.trunc() as i128 as i64) as f64,
        PixelId::Float32
        | PixelId::ComplexFloat32
        | PixelId::VectorFloat32
        | PixelId::Float64
        | PixelId::ComplexFloat64
        | PixelId::VectorFloat64 => v,
    }
}

/// `UnsharpMaskImageFilter`: `sharpened = original + [|original - blurred| -
/// threshold] * amount` for pixels where `|original - blurred| > threshold`,
/// else `original` unchanged (`UnsharpMaskingFunctor::operator()`).
/// `blurred` is `original` smoothed by `SmoothingRecursiveGaussianImageFilter`
/// with per-axis physical-space `sigmas` ([`recursive_gaussian`]), computed
/// on a full-precision `f64` scratch copy so the input's pixel type is only
/// applied once, at the very end.
///
/// `threshold` must be `>= 0` (`VerifyPreconditions`). When `clamp` is true
/// the result is clamped to `[NonpositiveMin, max]` of `img`'s pixel type
/// before narrowing; when false and the pixel type is an integer, an
/// out-of-range result wraps (see `wrap_to_pixel_type`) rather than
/// saturating.
pub fn unsharp_mask(
    img: &Image,
    sigmas: &[f64],
    amount: f64,
    threshold: f64,
    clamp: bool,
) -> Result<Image> {
    let dim = img.dimension();
    if sigmas.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: sigmas.len(),
        });
    }
    if sigmas.iter().any(|&s| s < 0.0) {
        return Err(FilterError::InvalidSigma(sigmas.to_vec()));
    }
    if threshold < 0.0 {
        return Err(FilterError::InvalidUnsharpThreshold(threshold));
    }

    let scratch = scratch_f64(img)?;
    let blurred = recursive_gaussian(&scratch, sigmas)?.to_f64_vec()?;
    let original = img.to_f64_vec()?;
    let target = img.pixel_id();

    let out: Vec<f64> = original
        .iter()
        .zip(blurred.iter())
        .map(|(&v, &s)| {
            let diff = v - s;
            let result = if diff > threshold {
                v + (diff - threshold) * amount
            } else if -diff > threshold {
                v + (diff + threshold) * amount
            } else {
                v
            };

            if clamp {
                result.clamp(nonpositive_min(target), numeric_max(target))
            } else if target.is_floating_point() {
                result
            } else {
                wrap_to_pixel_type(target, result)
            }
        })
        .collect();

    image_from_f64(target, img.size(), img, &out)
}

// ---- laplacian_sharpening ---------------------------------------------------

/// `LaplacianSharpeningImageFilter`: combine the input with its Laplacian,
/// rescaled to the input's own intensity range, then re-centered on the
/// input's mean and clamped to `[input_min, input_max]`
/// (`itkLaplacianSharpeningImageFilter.hxx`'s `GenerateData`):
///
/// ```text
/// laplacian[i]      = LaplacianOperator-convolved input (ZeroFluxNeumannBoundaryCondition)
/// combined[i]       = input[i] - ((laplacian[i] - filtered_min) * (input_scale / filtered_scale) + input_min)
/// output[i]         = clamp(combined[i] - enhanced_mean + input_mean, input_min, input_max)
/// ```
///
/// where `input_scale = input_max - input_min`, `filtered_scale =
/// filtered_max - filtered_min` (the Laplacian's own range), and
/// `enhanced_mean` is the mean of `combined`. The Laplacian operator's
/// coefficients (`itkLaplacianOperator.h(.hxx)`, `hsq = (1/spacing[i])^2`
/// when `use_image_spacing`, else `1`, center `-Σ 2·hsq`) are algebraically
/// the crate's existing [`crate::filters::laplacian`], so this reuses it on a
/// full-precision scratch copy rather than re-deriving the operator.
///
/// Deviates from the literal `.hxx` in exactly one place: when
/// `filtered_scale == 0` (the Laplacian is perfectly flat — always true for
/// a constant input, since its Laplacian is 0 everywhere, and possible in
/// principle for other inputs too), the literal formula computes `0.0 *
/// (input_scale / 0.0)`, which is `NaN` under IEEE 754 for *any*
/// `input_scale` (`0 * finite` is `0`, but `0 * ±Infinity` is `NaN`, and
/// `filtered[i] - filtered_shift` is exactly `0` for every pixel whenever
/// `filtered_scale == 0`). This port treats the rescaled-Laplacian term as
/// contributing `0` in that case — "no Laplacian variation to rescale" — is
/// the only value consistent with a flat Laplacian, and is what makes a
/// constant image an exact fixed point end to end (see the module tests)
/// instead of propagating `NaN`.
///
/// `use_image_spacing` (default `true` upstream) is `LaplacianOperator`'s
/// `UseImageSpacing`. Upstream's `GenerateCoefficients` additionally throws
/// "Image spacing cannot be zero" unconditionally, even on the
/// `use_image_spacing == false` branch that never reads it — that check is
/// omitted here because it can never fire: `crate::core::Image::set_spacing`
/// already rejects non-positive spacing (`Error::NonPositiveSpacing`), and
/// `Image::new`/`from_vec` default every axis to spacing `1.0`, so no
/// `Image` value this crate can construct ever has zero spacing.
pub fn laplacian_sharpening(img: &Image, use_image_spacing: bool) -> Result<Image> {
    let original = img.to_f64_vec()?;
    let n = original.len();
    if n == 0 {
        return Err(FilterError::DegenerateRange);
    }

    let (mut input_min, mut input_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut input_sum = CompensatedSum::new();
    for &v in &original {
        input_min = input_min.min(v);
        input_max = input_max.max(v);
        input_sum.add(v);
    }
    let input_mean = input_sum.sum() / n as f64;
    let input_shift = input_min;
    let input_scale = input_max - input_min;

    let scratch = scratch_f64(img)?;
    let filtered = crate::filters::laplacian(&scratch, use_image_spacing)?.to_f64_vec()?;

    let (mut filtered_min, mut filtered_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &filtered {
        filtered_min = filtered_min.min(v);
        filtered_max = filtered_max.max(v);
    }
    let filtered_shift = filtered_min;
    let filtered_scale = filtered_max - filtered_min;

    let combined: Vec<f64> = original
        .iter()
        .zip(filtered.iter())
        .map(|(&orig, &f)| {
            let adjustment = if filtered_scale == 0.0 {
                0.0
            } else {
                (f - filtered_shift) * (input_scale / filtered_scale)
            };
            orig - (adjustment + input_shift)
        })
        .collect();

    let enhanced_mean = compensated_sum(combined.iter().copied()) / n as f64;

    let out: Vec<f64> = combined
        .iter()
        .map(|&c| {
            let shifted = c - enhanced_mean + input_mean;
            shifted.clamp(input_min, input_max)
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

#[cfg(test)]
mod compensated_laplacian_sharpening {
    //! `LaplacianSharpeningImageFilter` takes **both** its means from
    //! `StatisticsImageFilter::GetMean()` (`itkLaplacianSharpeningImageFilter.hxx:54-61`
    //! for the input, `:131-136` for the enhanced image), and that filter accumulates
    //! through a `CompensatedSummation` — so upstream compensates these two reductions
    //! *indirectly*, by delegating them. This port hand-rolled both with a naive `+=`.
    //!
    //! Ledger §2.172. It is the member of the §2.165 family that the sweep's anchor could
    //! not see: the anchor was the token `CompensatedSummation` in an ITK source file, and
    //! this filter's source does not contain it — it reaches compensation one filter away.
    use super::*;
    use crate::core::compensated::compensated_sum;

    const N: usize = 64;

    /// 4096 intensities near `1e6` whose fractional parts do not sum exactly: the running
    /// sum reaches ~4.1e9, where each term's low bits fall below an ulp, so the naive walk
    /// sheds them one rounding at a time and the compensated walk carries them. The two
    /// means come out **one ulp apart**.
    ///
    /// The narrow value range is deliberate and load-bearing. A fixture built the obvious
    /// way — one enormous pixel among small ones — separates the means by ~1.0, but the
    /// filter's final `clamp(.., input_min, input_max)` then saturates every pixel and the
    /// two pipelines produce *identical* output: the pin would have passed against nothing.
    /// The mean difference has to be small enough to survive the clamp and still move bits.
    fn fixture() -> Image {
        let v: Vec<f64> = (0..N * N)
            .map(|i| 1.0e6 + (i % 7) as f64 * 0.1 + 1.0e-3 * i as f64)
            .collect();
        Image::from_vec(&[N, N], v).unwrap()
    }

    /// The filter's own pipeline, with the two means computed either naively or
    /// compensated. Everything else is identical, so any difference in the output is the
    /// means and nothing else.
    fn replica(img: &Image, compensated: bool) -> Vec<f64> {
        let original = img.to_f64_vec().unwrap();
        let n = original.len();
        let mean = |xs: &[f64]| {
            if compensated {
                compensated_sum(xs.iter().copied()) / n as f64
            } else {
                xs.iter().sum::<f64>() / n as f64
            }
        };

        let (mut input_min, mut input_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &original {
            input_min = input_min.min(v);
            input_max = input_max.max(v);
        }
        let input_mean = mean(&original);
        let input_scale = input_max - input_min;

        let scratch = scratch_f64(img).unwrap();
        let filtered = crate::filters::laplacian(&scratch, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let (mut filtered_min, mut filtered_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &filtered {
            filtered_min = filtered_min.min(v);
            filtered_max = filtered_max.max(v);
        }
        let filtered_scale = filtered_max - filtered_min;

        let combined: Vec<f64> = original
            .iter()
            .zip(filtered.iter())
            .map(|(&orig, &f)| {
                let adjustment = if filtered_scale == 0.0 {
                    0.0
                } else {
                    (f - filtered_min) * (input_scale / filtered_scale)
                };
                orig - (adjustment + input_min)
            })
            .collect();
        let enhanced_mean = mean(&combined);

        combined
            .iter()
            .map(|&c| (c - enhanced_mean + input_mean).clamp(input_min, input_max))
            .collect()
    }

    /// **The fixture is real.** If the naive and compensated pipelines agreed, the pin
    /// below could not fail and would be proving nothing — the failure mode that caught
    /// two of this sweep's other pins.
    #[test]
    fn a_naive_mean_and_a_compensated_mean_are_different_numbers_here() {
        let img = fixture();
        let data = img.to_f64_vec().unwrap();
        let naive: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let compensated = compensated_sum(data.iter().copied()) / data.len() as f64;
        assert_ne!(
            naive.to_bits(),
            compensated.to_bits(),
            "the fixture must separate the two walks: {naive} vs {compensated}"
        );
        assert_ne!(
            replica(&img, false),
            replica(&img, true),
            "the two means must reach the output, or the pin below is vacuous"
        );
    }

    /// **The pin.** The filter's means must be the compensated ones. Revert either sum in
    /// `laplacian_sharpening` to `+=` / `.sum::<f64>()` and this fails.
    #[test]
    fn the_sharpening_means_are_compensated_and_a_naive_walk_is_not_good_enough() {
        let img = fixture();
        let out = laplacian_sharpening(&img, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let compensated = replica(&img, true);
        for (i, (&a, &b)) in out.iter().zip(compensated.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "pixel {i}: {a} vs {b}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- unsharp_mask -----------------------------------------------------

    #[test]
    fn amount_zero_is_identity() {
        // Every branch of the functor reduces to `result = v` when
        // `amount == 0`, regardless of the sign or magnitude of `diff`.
        let data: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0];
        let img = Image::from_vec(&[8], data.clone()).unwrap();
        let out = unsharp_mask(&img, &[1.0], 0.0, 0.0, false).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), data);
    }

    #[test]
    fn clamp_saturates_at_u8_extremes() {
        // Step edge: the recursive Gaussian pulls the pre-edge low pixel's
        // blurred value above it (diff < 0) and the post-edge high pixel's
        // blurred value below it (diff > 0); a huge amount blows either
        // past u8 range in a known direction, and `clamp` saturates it.
        let data: Vec<u8> = vec![0, 0, 0, 0, 200, 200, 200, 200];
        let img = Image::from_vec(&[8], data).unwrap();
        let out = unsharp_mask(&img, &[1.0], 1.0e6, 0.0, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out[3], 0.0, "pre-edge pixel should saturate low: {out:?}");
        assert_eq!(
            out[4], 255.0,
            "post-edge pixel should saturate high: {out:?}"
        );
    }

    #[test]
    fn no_clamp_wraps_instead_of_saturating() {
        let data: Vec<u8> = vec![0, 0, 0, 0, 200, 200, 200, 200];
        let img = Image::from_vec(&[8], data).unwrap();
        let clamped = unsharp_mask(&img, &[1.0], 1.0e6, 0.0, true)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let wrapped = unsharp_mask(&img, &[1.0], 1.0e6, 0.0, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        // Whatever the exact wrapped values are, they must not equal the
        // saturated 0/255 boundary that `clamp` produces for the same input
        // (a value that overflows u8 essentially never wraps back to
        // exactly 0 or 255).
        assert_ne!(clamped, wrapped);
    }

    #[test]
    fn wrap_to_pixel_type_matches_documented_example() {
        // SimpleITK's UnsharpMaskImageFilter.yaml doc, verbatim: "values are
        // not clamped but rather wrap around e.g. 260 -> 4 (unsigned char)".
        assert_eq!(wrap_to_pixel_type(PixelId::UInt8, 260.0), 4.0);
    }

    #[test]
    fn wrong_sigma_length_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            unsharp_mask(&img, &[1.0], 0.5, 0.0, false),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn negative_sigma_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            unsharp_mask(&img, &[-1.0, 1.0], 0.5, 0.0, false),
            Err(FilterError::InvalidSigma(_))
        ));
    }

    #[test]
    fn negative_threshold_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            unsharp_mask(&img, &[1.0, 1.0], 0.5, -0.1, false),
            Err(FilterError::InvalidUnsharpThreshold(t)) if t == -0.1
        ));
    }

    // ---- laplacian_sharpening ----------------------------------------------

    #[test]
    fn constant_image_is_a_fixed_point() {
        // The Laplacian of a constant image is 0 everywhere, so
        // `filtered_scale == 0` (and `input_scale == 0` too) — the
        // degenerate case the module doc's NaN-avoidance guard targets.
        let img = Image::from_vec(&[6, 6], vec![7.0f64; 36]).unwrap();
        let out = laplacian_sharpening(&img, true).unwrap();
        for v in out.to_f64_vec().unwrap() {
            assert!((v - 7.0).abs() < 1e-9, "expected 7.0, got {v}");
        }
    }

    #[test]
    fn output_stays_within_input_range() {
        // `std::clamp(shiftedValue, inputMinimum, inputMaximum)` is
        // unconditional, so no output pixel can fall outside the input's
        // own [min, max], no matter how the Laplacian rescaling behaves.
        let data: Vec<f64> = (0..64).map(|v| ((v * 37) % 97) as f64).collect();
        let img = Image::from_vec(&[8, 8], data).unwrap();
        let out = laplacian_sharpening(&img, true).unwrap();
        for v in out.to_f64_vec().unwrap() {
            assert!((0.0..=96.0).contains(&v), "{v} outside input range");
        }
    }

    #[test]
    fn use_image_spacing_false_ignores_anisotropic_spacing() {
        // With use_image_spacing=false every axis's derivative scaling is 1,
        // so anisotropic spacing must not affect the result at all.
        let data: Vec<f64> = (0..64).map(|v| ((v * 37) % 97) as f64).collect();
        let mut img = Image::from_vec(&[8, 8], data.clone()).unwrap();
        let isotropic = laplacian_sharpening(&img, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        img.set_spacing(&[1.0, 3.0]).unwrap();
        let anisotropic = laplacian_sharpening(&img, false)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(isotropic, anisotropic);
    }

    #[test]
    fn output_pixel_type_follows_input() {
        let img = Image::from_vec(&[4, 4], vec![5i16; 16]).unwrap();
        let out = laplacian_sharpening(&img, true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Int16);
    }
}
