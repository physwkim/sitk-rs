//! `ExpandImageFilter`: upsample an image by an integer factor per axis,
//! interpolating the input at each output pixel's continuous input-space
//! location.
//!
//! Verified against `Modules/Filtering/ImageGrid/include/itkExpandImageFilter.h(.hxx)`
//! and SimpleITK's `Code/BasicFilters/yaml/ExpandImageFilter.yaml`
//! (`ExpandFactors` default `[1, 1, ...]`, `Interpolator` default
//! `sitkLinear`).
//!
//! The per-output-pixel interpolation reuses the transform crate's shared
//! sampling primitive (`crate::transform::resample::InterpolatedImage`) rather
//! than a filter-local linear/nearest implementation: after the single-crate
//! consolidation there is no dependency edge to avoid, and that primitive
//! already carries every kernel SimpleITK's `ExpandImageFilter.yaml` exposes
//! (`sitkLinear`, `sitkNearestNeighbor`, `sitkBSpline`, `sitkGaussian`, and the
//! windowed-sinc family), so `Interpolator` here is a re-export of
//! [`crate::transform::Interpolator`] and Expand gains the full set for free.

use crate::core::Image;
use crate::filters::error::{FilterError, Result};
use crate::filters::image_from_f64;
use crate::transform::Interpolator;
use crate::transform::resample::InterpolatedImage;

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `ExpandImageFilter`: upsamples `img` by the per-axis integer `factors`,
/// resampling with `interpolator` (default `Linear`, matching
/// `ExpandImageFilter.yaml`) at continuous input-space index
/// `(out_index+0.5)/factor - 0.5` (`itkExpandImageFilter.hxx`'s
/// `DynamicThreadedGenerateData`).
///
/// Output geometry (`GenerateOutputInformation`):
///
/// ```text
/// outSize[d]    = inSize[d] * factor[d]
/// outSpacing[d] = inSpacing[d] / factor[d]
/// fraction[d]   = (factor[d]-1) / factor[d]
/// shift[d]      = -(inSpacing[d]/2) * fraction[d]      (input-space units)
/// outOrigin     = inOrigin + Direction * shift
/// ```
///
/// The origin *does* shift — this is not the "origin stays put" one might
/// assume by analogy with [`crate::filters::shrink()`]. Expand resamples at continuous
/// index `(outIndex+0.5)/factor - 0.5`, i.e. a finer grid whose first sample
/// isn't centered on the input's first pixel unless `factor == 1`
/// (`fraction == 0`), so the physical origin must shift by half the input
/// spacing scaled by `(factor-1)/factor` to keep the two grids aligned in
/// physical space. Direction is unchanged.
///
/// Every sampled continuous index satisfies `cindex[d] ∈ [-0.5, size[d]-0.5)`
/// for `factor[d] >= 1`: the minimum sample `(0+0.5)/factor - 0.5` is `> -0.5`
/// (the `0.5/factor` term is strictly positive), and the maximum sample
/// `size - 0.5 - 0.5/factor` is `< size - 0.5`. That is exactly the half-open
/// range `InterpolatedImage::sample` treats as inside the buffer
/// (`is_inside`: `c >= -0.5 && c < size-0.5`), so the sample is always `Some`
/// and the `.expect()` below cannot fire — a broken bound would panic rather
/// than silently interpolate out-of-domain data. Within that range the
/// interpolator's own boundary clamp (linear's per-corner
/// `std::min`/`std::max` against `Start/EndIndex`, nearest's
/// `RoundHalfIntegerUp`-then-clamp) reproduces ITK's edge handling.
///
/// Errors if `factors.len()` doesn't match `img`'s dimension, or any factor
/// is `0`. Upstream's own array-typed setter
/// (`itk::ExpandImageFilter::SetExpandFactors(ExpandFactorsType)`, which is
/// what SimpleITK's per-axis wrapper actually calls) does **not** clamp a
/// zero factor up to 1 — only the unrelated scalar convenience setter
/// (`SetExpandFactors(unsigned int)`, reachable solely through the custom
/// `SetExpandFactor` uniform-factor helper) does that clamping — so a zero
/// factor set per-axis reaches `GenerateOutputInformation` and divides by
/// zero there. This port declines to reproduce that division-by-zero
/// footgun and rejects a zero factor as an error instead: a deliberate
/// divergence, analogous to [`crate::filters::shrink()`]'s `InvalidShrinkFactor`.
pub fn expand(img: &Image, factors: &[usize], interpolator: Interpolator) -> Result<Image> {
    let dim = img.dimension();
    if factors.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: factors.len(),
        });
    }
    if factors.contains(&0) {
        return Err(FilterError::InvalidExpandFactor(factors.to_vec()));
    }

    let in_size = img.size();
    let in_spacing = img.spacing();
    let in_origin = img.origin();
    let direction = img.direction();

    let mut out_size = vec![0usize; dim];
    let mut out_spacing = vec![0.0f64; dim];
    let mut shift = vec![0.0f64; dim];
    for d in 0..dim {
        let f = factors[d] as f64;
        out_size[d] = in_size[d] * factors[d];
        out_spacing[d] = in_spacing[d] / f;
        let fraction = (f - 1.0) / f;
        shift[d] = -(in_spacing[d] / 2.0) * fraction;
    }

    // outputOrigin = inputOrigin + Direction * shift.
    let mut out_origin = in_origin.to_vec();
    for (i, o) in out_origin.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &sj) in shift.iter().enumerate() {
            acc += direction[i * dim + j] * sj;
        }
        *o += acc;
    }

    let interp = InterpolatedImage::new(img, interpolator)?;
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();

    let mut out_vals = vec![0.0f64; out_count];
    for (o, slot) in out_vals.iter_mut().enumerate() {
        let mut cindex = vec![0.0f64; dim];
        for d in 0..dim {
            let oi = (o / out_strides[d]) % out_size[d];
            cindex[d] = (oi as f64 + 0.5) / factors[d] as f64 - 0.5;
        }
        *slot = interp
            .sample(&cindex)
            .expect("expand samples cindex ∈ [-0.5, size-0.5), always inside the buffer");
    }

    let mut out = image_from_f64(img.pixel_id(), &out_size, img, &out_vals)?;
    out.set_spacing(&out_spacing)?;
    out.set_origin(&out_origin)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_one_is_identity() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = expand(&img, &[1, 1], Interpolator::Linear).unwrap();
        assert_eq!(out.size(), img.size());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn factor_two_linear_values_hand_derived() {
        // 1-D, 2 pixels [10, 20], factor 2 -> 4 output samples.
        // cindex(out) = (out+0.5)/2 - 0.5:
        //   out=0: -0.25 -> base=-1 (clamps to 0), frac=0.75: both corners
        //          clamp to index 0 -> 0.25*10 + 0.75*10 = 10.
        //   out=1:  0.25 -> base=0, frac=0.25: 0.75*10 + 0.25*20 = 12.5.
        //   out=2:  0.75 -> base=0, frac=0.75: 0.25*10 + 0.75*20 = 17.5.
        //   out=3:  1.25 -> base=1, frac=0.25: upper corner (index 2) clamps
        //          to 1 -> 0.75*20 + 0.25*20 = 20.
        let img = Image::from_vec(&[2], vec![10.0, 20.0]).unwrap();
        let out = expand(&img, &[2], Interpolator::Linear).unwrap();
        assert_eq!(out.size(), &[4]);
        let got = out.to_f64_vec().unwrap();
        let expected = [10.0, 12.5, 17.5, 20.0];
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-12, "{got:?} vs {expected:?}");
        }
    }

    #[test]
    fn origin_and_spacing_pinned_to_itk_formula() {
        // 1-D, size 2, spacing 1, origin 0, factor 2:
        // outSpacing = 1/2 = 0.5; fraction = (2-1)/2 = 0.5;
        // shift = -(1/2)*0.5 = -0.25; outOrigin = 0 + (-0.25) = -0.25.
        let img = Image::from_vec(&[2], vec![10.0, 20.0]).unwrap();
        let out = expand(&img, &[2], Interpolator::Linear).unwrap();
        assert_eq!(out.spacing(), &[0.5]);
        assert_eq!(out.origin(), &[-0.25]);
    }

    #[test]
    fn anisotropic_2d_origin_and_spacing() {
        // factors [2, 3], spacing [2.0, 3.0], origin [1.0, -2.0], identity
        // direction: outSpacing = [1.0, 1.0]; fraction = [0.5, 2/3];
        // shift = [-(2.0/2)*0.5, -(3.0/2)*(2/3)] = [-0.5, -1.0];
        // outOrigin = [1.0-0.5, -2.0-1.0] = [0.5, -3.0].
        let mut img = Image::new(&[3, 2], crate::core::PixelId::Float64);
        img.set_spacing(&[2.0, 3.0]).unwrap();
        img.set_origin(&[1.0, -2.0]).unwrap();
        let out = expand(&img, &[2, 3], Interpolator::Linear).unwrap();
        assert_eq!(out.size(), &[6, 6]);
        assert_eq!(out.spacing(), &[1.0, 1.0]);
        assert_eq!(out.origin(), &[0.5, -3.0]);
    }

    #[test]
    fn nearest_neighbor_never_blends() {
        // Every output value must equal one of the input's exact values —
        // nearest neighbor never interpolates between them.
        let img = Image::from_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap();
        let out = expand(&img, &[4], Interpolator::NearestNeighbor).unwrap();
        for v in out.to_f64_vec().unwrap() {
            assert!([1.0, 2.0, 3.0].contains(&v), "unexpected blended value {v}");
        }
    }

    #[test]
    fn wrong_factor_length_is_rejected() {
        let img = Image::new(&[4, 4], crate::core::PixelId::Float64);
        assert!(matches!(
            expand(&img, &[2], Interpolator::Linear),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn zero_factor_is_rejected() {
        let img = Image::new(&[4, 4], crate::core::PixelId::Float64);
        assert!(matches!(
            expand(&img, &[0, 2], Interpolator::Linear),
            Err(FilterError::InvalidExpandFactor(_))
        ));
    }

    #[test]
    fn output_pixel_type_follows_input() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let out = expand(&img, &[2, 2], Interpolator::Linear).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::UInt8);
    }

    #[test]
    fn factor_one_bspline_reproduces_input_exactly() {
        // At factor 1 every sampled cindex is an integer index, and BSpline is
        // interpolating — it must reproduce the input samples exactly. This
        // also proves the shared interpolator's `is_inside` guard never trips
        // (an out-of-buffer sample would return `None` and panic in `expand`).
        let img = Image::from_vec(&[5], vec![1.0, 4.0, 9.0, 16.0, 25.0]).unwrap();
        let out = expand(&img, &[1], Interpolator::BSpline).unwrap();
        let got = out.to_f64_vec().unwrap();
        for (g, e) in got.iter().zip([1.0, 4.0, 9.0, 16.0, 25.0]) {
            assert!((g - e).abs() < 1e-9, "{got:?}");
        }
    }

    #[test]
    fn bspline_and_gaussian_diverge_from_linear_on_a_curved_signal() {
        // On a non-linear signal the higher-order (BSpline) and smoothing
        // (Gaussian) kernels must produce a different upsampling than the
        // piecewise-linear one — the whole point of exposing them on Expand.
        let img = Image::from_vec(&[5], vec![1.0, 4.0, 9.0, 16.0, 25.0]).unwrap();
        let linear = expand(&img, &[3], Interpolator::Linear)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let bspline = expand(&img, &[3], Interpolator::BSpline)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let gaussian = expand(&img, &[3], Interpolator::Gaussian)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_ne!(linear, bspline);
        assert_ne!(linear, gaussian);
    }

    #[test]
    fn windowed_sinc_stays_finite_across_the_whole_grid() {
        // A windowed-sinc kernel touches a radius-5 neighbourhood; run it over
        // a 2-D expansion to confirm every sample resolves (no `None`/panic)
        // and stays finite.
        let img = Image::from_vec(&[4, 4], (0..16).map(|v| v as f64).collect()).unwrap();
        let out = expand(&img, &[2, 3], Interpolator::HammingWindowedSinc).unwrap();
        assert_eq!(out.size(), &[8, 12]);
        for v in out.to_f64_vec().unwrap() {
            assert!(v.is_finite(), "non-finite sample {v}");
        }
    }
}
