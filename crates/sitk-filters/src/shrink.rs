//! `ShrinkImageFilter`: subsample an image by an integer factor per dimension.
//!
//! Bit-exact port of `itk::ShrinkImageFilter`. The output is a strict
//! subsampling (no interpolation): `out[j] = in[j·factor + offset]`, where the
//! per-axis `offset` and the output origin are chosen so the physical *center*
//! of the input and output images coincide. Output geometry:
//!
//! ```text
//! outSize[d]    = max(1, floor(inSize[d] / factor[d]))
//! outSpacing[d] = inSpacing[d] · factor[d]
//! δ[d]          = (inSize[d]-1)/2 − factor[d]·(outSize[d]-1)/2      (index units)
//! outOrigin     = inOrigin + Direction · diag(inSpacing) · δ
//! offset[d]     = round(δ[d])                                        (>= 0)
//! ```
//!
//! Direction is unchanged. This is the coarse-grid producer for a
//! multi-resolution registration pyramid (paired with [`smooth_gaussian`]).
//!
//! [`bin_shrink`], verified against
//! `Modules/Filtering/ImageGrid/include/itkBinShrinkImageFilter.h`/`.hxx`, is
//! a *different* filter with a similarly-named purpose: instead of picking
//! one sample per output pixel, it **averages** every input pixel in the
//! `Π factor[d]`-sized bin block, `AccumulatePixelType` (`double`) accumulated
//! then narrowed back to the input's own pixel type by `RoundIfInteger` --
//! `itk::Math::Round` (round-half-*up*, [`round_half_up`]) for an integer
//! pixel type, a plain (unrounded) narrowing cast for a floating-point one.
//! Its output geometry differs from [`shrink`]'s too:
//!
//! ```text
//! outSize[d]    = floor(inSize[d] / factor[d])   (errors if this is < 1)
//! outSpacing[d] = inSpacing[d] · factor[d]
//! outOrigin     = TransformContinuousIndexToPhysicalPoint(0.5·(factor[d]-1))
//! ```
//!
//! i.e. the output origin is the physical location of input continuous index
//! `0.5·(factor[d]-1)` (the centroid of the first bin block) rather than
//! [`shrink`]'s center-of-image-preserving `δ`. `GenerateOutputInformation`
//! throws when an axis's bin factor exceeds its input size (`outSize[d] < 1`,
//! [`FilterError::BinShrinkFactorTooLarge`]) instead of clamping to 1 pixel
//! the way [`shrink`] does.
//!
//! [`smooth_gaussian`]: crate::smooth_gaussian

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::Image;

/// Subsample `img` by `factors` (one positive integer per dimension).
///
/// Errors if `factors` has the wrong length or any factor is zero.
pub fn shrink(img: &Image, factors: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    if factors.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: factors.len(),
        });
    }
    if factors.contains(&0) {
        return Err(FilterError::InvalidShrinkFactor(factors.to_vec()));
    }

    let in_size = img.size();
    let in_spacing = img.spacing();
    let in_origin = img.origin();
    let direction = img.direction();

    // Output size, spacing, per-axis center-preserving shift δ, and sampling
    // offset (ITK ShrinkImageFilter::GenerateOutputInformation).
    let mut out_size = vec![0usize; dim];
    let mut out_spacing = vec![0.0f64; dim];
    let mut delta = vec![0.0f64; dim];
    let mut offset = vec![0usize; dim];
    for d in 0..dim {
        let f = factors[d];
        out_size[d] = (in_size[d] / f).max(1);
        out_spacing[d] = in_spacing[d] * f as f64;
        delta[d] = (in_size[d] as f64 - 1.0) / 2.0 - f as f64 * (out_size[d] as f64 - 1.0) / 2.0;
        // round-half-up, clamped to a valid non-negative sample offset.
        offset[d] = (delta[d] + 0.5).floor().max(0.0) as usize;
    }

    // Output origin: inOrigin + Direction · diag(inSpacing) · δ.
    let mut out_origin = in_origin.to_vec();
    for (i, o) in out_origin.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &dj) in delta.iter().enumerate() {
            acc += direction[i * dim + j] * in_spacing[j] * dj;
        }
        *o += acc;
    }

    let in_strides = strides(in_size);
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();

    let mut sources: Vec<Option<usize>> = vec![None; out_count];
    for (o, slot) in sources.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for d in 0..dim {
            let oi = (o / out_strides[d]) % out_size[d];
            let ii = (oi * factors[d] + offset[d]).min(in_size[d] - 1);
            in_flat += ii * in_strides[d];
        }
        *slot = Some(in_flat);
    }

    // `shrink` preserves the input direction (only spacing/origin change), so
    // gather inherits the input geometry and only spacing/origin are overridden.
    let mut out = img.gather(&out_size, &sources, 0.0)?;
    out.set_spacing(&out_spacing)?;
    out.set_origin(&out_origin)?;
    Ok(out)
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `itk::Math::RoundHalfIntegerUp` (`itk::Math::Round`'s synonym): rounds
/// half-integers *up* (toward `+inf`), e.g. `-1.5 -> -1`. Disagrees with
/// Rust's `f64::round` (round half away from zero) for negative
/// half-integers; see `threshold.rs`'s copy of this same helper for the
/// derivation.
fn round_half_up(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// Σ of the `Π factors[d]` input pixels in the bin block whose first corner
/// is `out_index[d] * factors[d]` along every axis
/// (`BinShrinkImageFilter::DynamicThreadedGenerateData`'s accumulation for
/// one output pixel, in `AccumulatePixelType = NumericTraits<InputPixelType>::RealType`,
/// `double` for every pixel type here).
fn bin_block_sum(
    vals: &[f64],
    in_strides: &[usize],
    factors: &[usize],
    out_index: &[usize],
) -> f64 {
    let dim = factors.len();
    let mut sum = 0.0f64;
    let mut off = vec![0usize; dim];
    loop {
        let flat: usize = (0..dim)
            .map(|d| (out_index[d] * factors[d] + off[d]) * in_strides[d])
            .sum();
        sum += vals[flat];

        let mut d = 0;
        loop {
            off[d] += 1;
            if off[d] < factors[d] {
                break;
            }
            off[d] = 0;
            d += 1;
            if d == dim {
                return sum;
            }
        }
    }
}

/// `BinShrinkImageFilter`: shrink `img` by an integer `factors` per
/// dimension, averaging each `Π factors[d]`-sized bin block rather than
/// [`shrink`]'s single-sample subsampling (see the module doc's geometry and
/// rounding comparison).
///
/// Errors if `factors` has the wrong length, any factor is zero, or a
/// factor exceeds the input size along its axis (`outputSize[d]` would be
/// `< 1`).
pub fn bin_shrink(img: &Image, factors: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    if factors.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: factors.len(),
        });
    }
    if factors.contains(&0) {
        return Err(FilterError::InvalidBinShrinkFactor(factors.to_vec()));
    }

    let in_size = img.size().to_vec();
    let in_spacing = img.spacing().to_vec();

    let mut out_size = vec![0usize; dim];
    let mut out_spacing = vec![0.0f64; dim];
    let mut origin_index = vec![0.0f64; dim];
    for d in 0..dim {
        out_size[d] = in_size[d] / factors[d];
        if out_size[d] < 1 {
            return Err(FilterError::BinShrinkFactorTooLarge {
                axis: d,
                size: in_size[d],
                factor: factors[d],
            });
        }
        out_spacing[d] = in_spacing[d] * factors[d] as f64;
        origin_index[d] = 0.5 * (factors[d] as f64 - 1.0);
    }
    let out_origin = img.continuous_index_to_physical_point(&origin_index);

    let in_strides = strides(&in_size);
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();
    let in_vals = img.to_f64_vec()?;
    let bin_count = factors.iter().product::<usize>() as f64;
    let round_integer_output = !img.pixel_id().is_floating_point();

    let out_vals: Vec<f64> = (0..out_count)
        .map(|o| {
            let out_index: Vec<usize> = (0..dim)
                .map(|d| (o / out_strides[d]) % out_size[d])
                .collect();
            let avg = bin_block_sum(&in_vals, &in_strides, factors, &out_index) / bin_count;
            if round_integer_output {
                round_half_up(avg)
            } else {
                avg
            }
        })
        .collect();

    let mut out = image_from_f64(img.pixel_id(), &out_size, img, &out_vals)?;
    out.set_spacing(&out_spacing)?;
    out.set_origin(&out_origin)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::Image;

    #[test]
    fn factor_one_is_identity() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = shrink(&img, &[1, 1]).unwrap();
        assert_eq!(out.size(), img.size());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn shrink_geometry_matches_itk_formulas() {
        // 40-pixel axis, factor 2 -> 20 pixels, spacing doubles.
        // δ = 39/2 − 2·19/2 = 19.5 − 19 = 0.5, offset = round(0.5) = 1.
        // outOrigin = 0 + 1·1·0.5 = 0.5 (unit spacing, identity direction).
        let mut img = Image::new(&[40, 40], sitk_core::PixelId::Float64);
        img.set_spacing(&[1.0, 1.0]).unwrap();
        img.set_origin(&[0.0, 0.0]).unwrap();
        let out = shrink(&img, &[2, 2]).unwrap();
        assert_eq!(out.size(), &[20, 20]);
        assert_eq!(out.spacing(), &[2.0, 2.0]);
        assert_eq!(out.origin(), &[0.5, 0.5]);
    }

    #[test]
    fn subsamples_with_center_offset() {
        // 1-D-ish (4x1) axis 0 factor 2: outSize=2, δ=(3)/2−2·1/2=1.5−1=0.5,
        // offset=round(0.5)=1 -> samples in[1], in[3].
        let img = Image::from_vec(&[4, 1], vec![10.0, 11.0, 12.0, 13.0]).unwrap();
        let out = shrink(&img, &[2, 1]).unwrap();
        assert_eq!(out.size(), &[2, 1]);
        assert_eq!(out.to_f64_vec().unwrap(), vec![11.0, 13.0]);
    }

    #[test]
    fn shrink_moves_u64_pixels_losslessly() {
        // Same subsampling as `subsamples_with_center_offset`, but with u64
        // values above 2^53 that an f64 round-trip would round. The non-vacuity
        // guard proves the samples are values the old seam would corrupt.
        let hi = (1u64 << 53) + 1;
        assert_ne!(hi, (hi as f64) as u64);
        let img = Image::from_vec(&[4, 1], vec![hi, hi + 1, hi + 2, hi + 3]).unwrap();
        let out = shrink(&img, &[2, 1]).unwrap();
        // offset 1: samples in[1], in[3].
        assert_eq!(out.scalar_slice::<u64>().unwrap(), &[hi + 1, hi + 3]);
    }

    #[test]
    fn odd_factor_stays_in_bounds() {
        // 40 axis, factor 3 -> outSize=13; δ=19.5−3·6=1.5, offset=2;
        // max in index = 12·3+2 = 38 < 40.
        let mut img = Image::new(&[40, 40], sitk_core::PixelId::Float64);
        img.set_spacing(&[1.0, 1.0]).unwrap();
        let out = shrink(&img, &[3, 3]).unwrap();
        assert_eq!(out.size(), &[13, 13]);
        assert_eq!(out.spacing(), &[3.0, 3.0]);
    }

    #[test]
    fn wrong_factor_length_is_rejected() {
        let img = Image::new(&[4, 4], sitk_core::PixelId::Float64);
        assert!(matches!(
            shrink(&img, &[2]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn zero_factor_is_rejected() {
        let img = Image::new(&[4, 4], sitk_core::PixelId::Float64);
        assert!(matches!(
            shrink(&img, &[0, 2]),
            Err(FilterError::InvalidShrinkFactor(_))
        ));
    }

    // ---- bin_shrink -----------------------------------------------------

    #[test]
    fn bin_shrink_factor_one_is_identity() {
        let mut img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        img.set_spacing(&[2.0, 3.0]).unwrap();
        img.set_origin(&[10.0, 20.0]).unwrap();
        let out = bin_shrink(&img, &[1, 1]).unwrap();
        assert_eq!(out.size(), img.size());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn bin_shrink_output_size_spacing_and_origin_are_hand_derived() {
        // size=[4,4], spacing=[2,3], origin=[10,20], factor=[2,2]:
        // outSize = floor(4/2) = 2 on each axis; outSpacing = [4,6];
        // originIndex = 0.5*(2-1) = 0.5 on each axis (continuous index in
        // input space), so outOrigin = origin + spacing*0.5 = [11, 21.5].
        let mut img = Image::from_vec(&[4, 4], (0..16).map(|v| v as f64).collect()).unwrap();
        img.set_spacing(&[2.0, 3.0]).unwrap();
        img.set_origin(&[10.0, 20.0]).unwrap();
        let out = bin_shrink(&img, &[2, 2]).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.spacing(), &[4.0, 6.0]);
        assert_eq!(out.origin(), &[11.0, 21.5]);
    }

    #[test]
    fn bin_shrink_averages_the_bin_block() {
        // x fastest, 4x4:
        //    0  1  2  3
        //    4  5  6  7
        //    8  9 10 11
        //   12 13 14 15
        // factor=[2,2] -> four 2x2 blocks, each averaged:
        //   {0,1,4,5}->2.5   {2,3,6,7}->4.5
        //   {8,9,12,13}->10.5 {10,11,14,15}->12.5
        let img = Image::from_vec(&[4, 4], (0..16).map(|v| v as f64).collect()).unwrap();
        let out = bin_shrink(&img, &[2, 2]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn bin_shrink_integer_output_rounds_half_up_not_truncated() {
        // Same grid and blocks as `bin_shrink_averages_the_bin_block`
        // (2.5, 4.5, 10.5, 12.5), but on a UInt8 image: RoundIfInteger's
        // `Math::Round` (round-half-up) gives 3, 5, 11, 13 -- not the 2, 4,
        // 10, 12 a plain truncating cast would produce.
        let img = Image::from_vec(&[4, 4], (0..16u8).collect()).unwrap();
        let out = bin_shrink(&img, &[2, 2]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![3.0, 5.0, 11.0, 13.0]);
    }

    #[test]
    fn bin_shrink_per_axis_factor_asymmetry() {
        // Same 4x4 grid, factor=[2,1]: only the x axis bins (pairs), y is
        // untouched, so every one of the 4 rows survives with 2 columns.
        //   y=0: {0,1}->0.5  {2,3}->2.5
        //   y=1: {4,5}->4.5  {6,7}->6.5
        //   y=2: {8,9}->8.5  {10,11}->10.5
        //   y=3: {12,13}->12.5 {14,15}->14.5
        let img = Image::from_vec(&[4, 4], (0..16).map(|v| v as f64).collect()).unwrap();
        let out = bin_shrink(&img, &[2, 1]).unwrap();
        assert_eq!(out.size(), &[2, 4]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
        );
    }

    #[test]
    fn bin_shrink_wrong_factor_length_is_rejected() {
        let img = Image::new(&[4, 4], sitk_core::PixelId::Float64);
        assert!(matches!(
            bin_shrink(&img, &[2]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn bin_shrink_zero_factor_is_rejected() {
        let img = Image::new(&[4, 4], sitk_core::PixelId::Float64);
        assert!(matches!(
            bin_shrink(&img, &[0, 2]),
            Err(FilterError::InvalidBinShrinkFactor(_))
        ));
    }

    #[test]
    fn bin_shrink_factor_exceeding_size_is_rejected() {
        let img = Image::new(&[4, 4], sitk_core::PixelId::Float64);
        assert!(matches!(
            bin_shrink(&img, &[5, 2]),
            Err(FilterError::BinShrinkFactorTooLarge {
                axis: 0,
                size: 4,
                factor: 5
            })
        ));
    }
}
