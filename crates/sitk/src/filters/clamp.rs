//! `ClampImageFilter`: cast an image to another pixel type, clamping the
//! result to a `[lower_bound, upper_bound]` range.
//!
//! Verified against `Modules/Filtering/ImageIntensity/include/itkClampImageFilter.h(.hxx)`
//! (the `Functor::Clamp` functor) and SimpleITK's
//! `Code/BasicFilters/yaml/ClampImageFilter.yaml` (`LowerBound`/`UpperBound`
//! default to `±DBL_MAX`, i.e. "no additional restriction beyond the output
//! type's own range").

use crate::core::{Image, PixelId, Scalar, dispatch_scalar};
use crate::filters::error::{FilterError, Result};
use crate::filters::image_from_f64;
use crate::filters::morphology::bounds_for;
use crate::filters::{FromWide, read_pixels_i128};

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
/// The per-pixel functor (`Functor::Clamp::operator()`,
/// `itkClampImageFilter.h`) is, verbatim:
///
/// ```text
/// const auto dA = static_cast<double>(A);
/// if (dA < m_LowerBound) return m_LowerBound;
/// if (dA > m_UpperBound) return m_UpperBound;
/// return static_cast<OutputType>(A);
/// ```
///
/// — so the **comparison is done in `double`** (`dA`, the input widened to
/// `f64`), and the in-range pixel returns `static_cast<OutputType>(A)` of the
/// *original* `A`, **not** of `dA`. This port keeps both halves exactly: the
/// comparison stays in `f64` (a `UInt64` above `2^53` compares as its rounded
/// `f64`, matching ITK's `dA` — including ITK's own rounding quirk at a bound),
/// while the in-range cast goes through the native integer value so it is a true
/// `static_cast<Out>(A)`.
///
/// The old port cast the *`f64` copy* in the in-range branch
/// (`static_cast<Out>(dA)`), collapsing a `UInt64`/`Int64` value above `2^53`
/// (e.g. `2^53 + 1 -> 2^53`). For **integer input to integer output** the value
/// path now runs through `i128` (`read_pixels_i128` / `build_clamp_from_i128`),
/// so `static_cast<Out>(A)` is exact. **Any float on either side** keeps the
/// `f64` path: for a float input it is already exact (`f32 -> f64` lossless), and
/// for an integer input clamped to `Float32` it preserves the existing
/// `native -> f64 -> f32` rounding (integer input to `Float64` is identical
/// either way).
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
        crate::filters::quantize_to_pixel_type(output_type, lower_bound)
    } else {
        out_min
    };
    let upper = if out_max > upper_bound {
        crate::filters::quantize_to_pixel_type(output_type, upper_bound)
    } else {
        out_max
    };

    if lower > upper {
        return Err(FilterError::InvalidClampBounds { lower, upper });
    }

    if img.pixel_id().is_integer_scalar() && output_type.is_integer_scalar() {
        // Integer -> integer: compare `(double)A` against the `f64` bounds
        // (ITK's `dA`), but cast the *native* pixel for the in-range branch.
        let wide = read_pixels_i128(img)?;
        dispatch_scalar!(
            output_type,
            build_clamp_from_i128,
            img.size(),
            img,
            &wide,
            lower,
            upper
        )
    } else {
        // Any float on either side: the `f64` path is exact for a float input
        // and preserves the existing rounding for an integer input to a float
        // output.
        let vals = img.to_f64_vec()?;
        let out: Vec<f64> = vals
            .iter()
            .map(|&v| {
                if v < lower {
                    lower
                } else if v > upper {
                    upper
                } else {
                    crate::filters::quantize_to_pixel_type(output_type, v)
                }
            })
            .collect();

        image_from_f64(output_type, img.size(), img, &out)
    }
}

/// The in-range native cast for integer-input clamp (see [`clamp`]): compares
/// `static_cast<double>(A)` against the `f64` bounds — ITK's `dA` — and returns
/// the native `static_cast<Out>(A)` for an in-range pixel, or the output-type
/// bound otherwise. The bounds narrow to `Out` exactly as the `f64` path's
/// `image_from_f64` would, so the clamped branch is unchanged.
fn build_clamp_from_i128<T: Scalar + FromWide>(
    size: &[usize],
    geom: &Image,
    wide: &[i128],
    lower: f64,
    upper: f64,
) -> Result<Image> {
    let lower_out = T::from_f64(lower);
    let upper_out = T::from_f64(upper);
    let out: Vec<T> = wide
        .iter()
        .map(|&a| {
            let da = a as f64; // static_cast<double>(A), ITK's dA
            if da < lower {
                lower_out
            } else if da > upper {
                upper_out
            } else {
                T::from_i128(a) // static_cast<Out>(A), native
            }
        })
        .collect();

    let mut out_img = Image::from_vec(size, out)?;
    out_img.copy_geometry_from(geom);
    Ok(out_img)
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
    fn integer_above_2_53_clamps_and_passes_through_losslessly() {
        // 2^53 + 1 is not f64-representable; the old `native -> f64 -> f64` path
        // collapsed it to 2^53. With full-range bounds every pixel is in range,
        // so a same-type clamp is the identity, bit-for-bit.
        let hard = (1u64 << 53) + 1;
        let a = Image::from_vec(&[3, 1], vec![0u64, hard, u64::MAX]).unwrap();
        let out = clamp(&a, PixelId::UInt64, -f64::MAX, f64::MAX).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt64);
        assert_eq!(out.scalar_slice::<u64>().unwrap(), &[0, hard, u64::MAX]);

        // A tight upper bound below the hard value still clamps it exactly to the
        // (integer) bound, and the in-range small value casts natively.
        let clamped = clamp(&a, PixelId::UInt64, 0.0, 1000.0).unwrap();
        assert_eq!(clamped.scalar_slice::<u64>().unwrap(), &[0, 1000, 1000]);
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
