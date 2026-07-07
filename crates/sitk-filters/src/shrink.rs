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

    let in_vals = img.to_f64_vec();
    let in_strides = strides(in_size);
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();

    let mut out_vals = vec![0.0f64; out_count];
    for (o, slot) in out_vals.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for d in 0..dim {
            let oi = (o / out_strides[d]) % out_size[d];
            let ii = (oi * factors[d] + offset[d]).min(in_size[d] - 1);
            in_flat += ii * in_strides[d];
        }
        *slot = in_vals[in_flat];
    }

    let mut out = image_from_f64(img.pixel_id(), &out_size, img, &out_vals)?;
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
        assert_eq!(out.to_f64_vec(), img.to_f64_vec());
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
        assert_eq!(out.to_f64_vec(), vec![11.0, 13.0]);
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
}
