//! `ClampImageFilter`: cast an image to another pixel type, clamping the
//! result to a `[lower_bound, upper_bound]` range.
//!
//! Verified against `Modules/Filtering/ImageIntensity/include/itkClampImageFilter.h(.hxx)`
//! (the `Functor::Clamp` functor) and SimpleITK's
//! `Code/BasicFilters/yaml/ClampImageFilter.yaml` (`LowerBound`/`UpperBound`
//! default to `±DBL_MAX`, i.e. "no additional restriction beyond the output
//! type's own range").

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::morphology::bounds_for;
use sitk_core::{Image, PixelId};

/// `ClampImageFilter`: casts `img` to `output_type`, clamping the result to
/// `[lower_bound, upper_bound]`.
///
/// Bounds handling mirrors SimpleITK's generated wrapper (not the raw
/// `itk::Functor::Clamp` constructor, which starts from the *default*
/// `NonpositiveMin()`/`max()` of the output type): the caller's `f64` bound
/// is intersected with the output type's own representable range,
/// `ClampImageFilter.yaml`'s `custom_itk_cast` —
///
/// ```text
/// lower = output_type::NonpositiveMin()
/// if lower < lower_bound { lower = cast<output_type>(lower_bound) }
/// upper = output_type::max()
/// if upper > upper_bound { upper = cast<output_type>(upper_bound) }
/// ```
///
/// — rather than the simpler `lower.max(...)`/`upper.min(...)`, so that the
/// default sentinel (`-DBL_MAX`/`DBL_MAX`) is never cast down into a narrow
/// output type (a `static_cast` of `±DBL_MAX` into e.g. `int32_t` is
/// undefined behaviour in C++); it only casts a bound that is actually
/// tighter than the type's own limit.
///
/// The per-pixel functor (`Functor::Clamp::operator()`) then compares the
/// input value against `[lower, upper]` in `f64` and, for an in-range pixel,
/// casts the *original* pixel value directly to `output_type` (not the
/// double-clamped copy — irrelevant for values already in range, but keeps
/// this a faithful `static_cast`, e.g. truncating a fractional float toward
/// zero rather than rounding).
///
/// Errors ([`FilterError::InvalidClampBounds`]) if, after that intersection,
/// `lower > upper` — matching `itk::Functor::Clamp::SetBounds`, which throws
/// in the same case (equal bounds are allowed: every pixel collapses to that
/// single value).
///
/// Deliberately not ported: `ClampImageFilter::GenerateData`'s in-place
/// fast path, which grafts the input straight to the output without
/// iterating when running in-place with bounds equal to the output type's
/// full range. That's a pure performance optimization with no effect on the
/// produced values (the per-pixel loop below already computes the identity
/// in that case), so skipping it changes no observable behavior.
pub fn clamp(
    img: &Image,
    output_type: PixelId,
    lower_bound: f64,
    upper_bound: f64,
) -> Result<Image> {
    let (out_max, out_min) = bounds_for(output_type);

    let lower = if out_min < lower_bound {
        crate::quantize_to_pixel_type(output_type, lower_bound)
    } else {
        out_min
    };
    let upper = if out_max > upper_bound {
        crate::quantize_to_pixel_type(output_type, upper_bound)
    } else {
        out_max
    };

    if lower > upper {
        return Err(FilterError::InvalidClampBounds { lower, upper });
    }

    let vals = img.to_f64_vec()?;
    let out: Vec<f64> = vals
        .iter()
        .map(|&v| {
            if v < lower {
                lower
            } else if v > upper {
                upper
            } else {
                crate::quantize_to_pixel_type(output_type, v)
            }
        })
        .collect();

    image_from_f64(output_type, img.size(), img, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_type_full_range_is_identity() {
        let a = Image::from_vec(&[4, 1], vec![0u8, 1, 128, 255]).unwrap();
        let out = clamp(&a, PixelId::UInt8, -f64::MAX, f64::MAX).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 1, 128, 255]);
    }

    #[test]
    fn bounds_intersect_with_output_type_range_when_looser() {
        // Requested bounds (-100000, 100000) are looser than Int16's own
        // [-32768, 32767], so the effective bounds fall back to the type's
        // own range: below/above those clamp, in between casts through.
        let a = Image::from_vec(&[4, 1], vec![-40000i32, -20000, 20000, 40000]).unwrap();
        let out = clamp(&a, PixelId::Int16, -100000.0, 100000.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Int16);
        assert_eq!(
            out.scalar_slice::<i16>().unwrap(),
            &[i16::MIN, -20000, 20000, i16::MAX]
        );
    }

    #[test]
    fn requested_bounds_win_when_tighter_than_output_type_range() {
        // UInt8's own range is [0, 255]; requested [50, 200] is tighter, so
        // it's the requested bounds that take effect.
        let a = Image::from_vec(&[5, 1], vec![0u8, 50, 100, 200, 255]).unwrap();
        let out = clamp(&a, PixelId::UInt8, 50.0, 200.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[50, 50, 100, 200, 200]);
    }

    #[test]
    fn lower_greater_than_upper_after_intersection_errors() {
        // Int32's own range is loose enough that both requested bounds win
        // the intersection, and 200 > 50 survives to the SetBounds check.
        let a = Image::from_vec(&[1, 1], vec![0i32]).unwrap();
        let err = clamp(&a, PixelId::Int32, 200.0, 50.0).unwrap_err();
        assert_eq!(
            err,
            FilterError::InvalidClampBounds {
                lower: 200.0,
                upper: 50.0
            }
        );
    }

    #[test]
    fn cross_type_cast_truncates_toward_zero_like_static_cast() {
        // In-range pass-through casts the original value, matching
        // static_cast<int32_t>(double) truncation (not rounding).
        let a = Image::from_vec(&[2, 1], vec![3.7f64, -3.7]).unwrap();
        let out = clamp(&a, PixelId::Int32, -f64::MAX, f64::MAX).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[3, -3]);
    }

    #[test]
    fn equal_bounds_collapse_every_pixel() {
        let a = Image::from_vec(&[3, 1], vec![-5.0f64, 0.0, 5.0]).unwrap();
        let out = clamp(&a, PixelId::Float64, 2.0, 2.0).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn preserves_geometry() {
        let mut a = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        a.set_spacing(&[0.5, 2.0]).unwrap();
        a.set_origin(&[3.0, -1.0]).unwrap();
        let out = clamp(&a, PixelId::UInt8, 0.0, 255.0).unwrap();
        assert_eq!(out.spacing(), a.spacing());
        assert_eq!(out.origin(), a.origin());
    }
}
