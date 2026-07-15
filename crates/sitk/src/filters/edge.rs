//! `ZeroCrossingBasedEdgeDetectionImageFilter`
//! (`Modules/Filtering/ImageFeature/include/itkZeroCrossingBasedEdgeDetectionImageFilter.h(.hxx)`):
//! a three-stage mini-pipeline built from pieces already ported elsewhere in
//! this crate —
//! [`crate::filters::denoise::discrete_gaussian_f64`] (smoothing) into
//! [`crate::filters::gradient::laplacian`] (second derivative) into
//! [`crate::filters::canny::zero_crossing_values`] (sign-change marking) — composed
//! exactly as `GenerateData()` wires the three internal filters:
//!
//! - `gaussianFilter` only has `SetVariance`/`SetMaximumError` called on it, so
//!   every other `DiscreteGaussianImageFilter` setting stays at its own
//!   default: `MaximumKernelWidth = 32`, `UseImageSpacing = true`
//!   (`itkDiscreteGaussianImageFilter.h`'s constructor).
//! - `laplacianFilter` is never touched beyond `SetInput`, so
//!   `LaplacianImageFilter`'s own default applies too:
//!   `UseImageSpacing = true` (`itkLaplacianImageFilter.h`'s
//!   `m_UseImageSpacing(true)`).
//! - `zerocrossingFilter` gets `ForegroundValue`/`BackgroundValue` passed
//!   straight through.
//!
//! **Output pixel type.** The outer filter's `SameTypeCheck` /
//! `PixelTypeIsFloatingPointCheck` concepts
//! (`itkZeroCrossingBasedEdgeDetectionImageFilter.h`) pin
//! `TInputImage::PixelType == TOutputImage::PixelType`, restricted to
//! `float`/`double` (`ZeroCrossingBasedEdgeDetectionImageFilter.yaml`'s
//! `pixel_types: RealPixelIDTypeList`, with no `output_pixel_type` override —
//! so the SimpleITK-generated output type is the *input's own* type). This is
//! unlike the standalone [`crate::filters::canny::zero_crossing`], whose
//! `ZeroCrossingImageFilter.yaml` hardcodes `output_pixel_type: uint8_t`
//! regardless of input. So this port calls
//! [`crate::filters::canny::zero_crossing_values`] — the untyped `f64` core shared by
//! both `zero_crossing` and `canny_edge_detection` — directly, and narrows
//! back to `img`'s own pixel type rather than to `UInt8`.
//!
//! **A real upstream discrepancy, documented but not applied.** The raw ITK
//! constructor fills `m_MaximumError` with `0.01`
//! (`ZeroCrossingBasedEdgeDetectionImageFilter()`'s member-init list), while
//! `ZeroCrossingBasedEdgeDetectionImageFilter.yaml`'s SimpleITK-facing
//! procedural default is `0.1`. This port takes `variance`/`maximum_error` as
//! required arguments with no default of its own, so the discrepancy is noted
//! here only as documentation.

use crate::core::{Image, PixelId};
use crate::filters::canny::zero_crossing_values;
use crate::filters::denoise::discrete_gaussian_f64;
use crate::filters::error::{FilterError, Result};
use crate::filters::gradient::laplacian;
use crate::filters::image_from_f64;

/// `ZeroCrossingBasedEdgeDetectionImageFilter`: Gaussian-smooth `img`
/// (`variance`/`maximum_error` per axis, under
/// [`discrete_gaussian_f64`]'s `MaximumKernelWidth = 32`,
/// `use_image_spacing = true` — the embedded `DiscreteGaussianImageFilter`'s
/// own untouched defaults), take its Laplacian ([`laplacian`] with
/// `use_image_spacing = true`, `LaplacianImageFilter`'s own default), then
/// mark every zero-crossing of that field ([`zero_crossing_values`]) with
/// `foreground_value`/`background_value`. Output keeps `img`'s own pixel
/// type (see the module docs for why this differs from
/// [`crate::filters::canny::zero_crossing`]'s hardcoded `UInt8`).
///
/// Errors if `img`'s pixel type isn't `Float32`/`Float64`
/// ([`FilterError::RequiresRealPixelType`] — the outer filter's
/// `PixelTypeIsFloatingPointCheck` concept is a C++ compile-time constraint
/// with no runtime equivalent, enforced here instead), or if
/// `variance`/`maximum_error` fail [`discrete_gaussian_f64`]'s own validation
/// (wrong length, negative variance, `maximum_error` outside `(0.0, 1.0)`).
pub fn zero_crossing_based_edge_detection(
    img: &Image,
    variance: &[f64],
    maximum_error: &[f64],
    foreground_value: u8,
    background_value: u8,
) -> Result<Image> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }

    let smoothed = discrete_gaussian_f64(img, variance, maximum_error, 32, true)?;
    let laplacian_img = laplacian(&smoothed, true)?;
    let vals = zero_crossing_values(
        &laplacian_img,
        foreground_value as f64,
        background_value as f64,
    )?;
    image_from_f64(pixel_id, img.size(), img, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    /// A 1-D step edge, smoothed by a positive-variance Gaussian, has exactly
    /// one Laplacian sign-change, right at the step — so the marked edge is a
    /// single line, not a smear.
    ///
    /// The image is sized so the whole thing sits inside the truncated
    /// Gaussian kernel's support (length 8, step at the midpoint, `variance =
    /// 1.0`): with any wider flat background, the smoothed Laplacian is
    /// *exactly* `0.0` far from the step (the truncated kernel's convolution
    /// window there never includes an input pixel from across the step), and
    /// `zero_crossing_values`' `this_one == 0.0 && that != 0.0` clause
    /// correctly (and, per `itkZeroCrossingImageFilter.hxx`, faithfully)
    /// marks *that* boundary too — a real, expected property of a
    /// finite-support kernel applied to a large uniform region, not a defect,
    /// but it means a wide flat background produces extra marks unrelated to
    /// the step itself. Keeping the whole image inside the kernel's support
    /// avoids that and isolates the single true crossing.
    #[test]
    fn step_edge_marks_the_edge_line_once() {
        let img = Image::from_vec(
            &[8],
            vec![0.0f64, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
        )
        .unwrap();

        let out = zero_crossing_based_edge_detection(&img, &[1.0], &[0.01], 1, 0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);

        let vals = out.scalar_slice::<f64>().unwrap();
        assert_eq!(vals, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn output_pixel_type_matches_input_not_uint8() {
        let img = Image::from_vec(&[8], vec![0.0f32, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
        let out = zero_crossing_based_edge_detection(&img, &[1.0], &[0.01], 1, 0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn foreground_and_background_values_pass_through() {
        let mut data = vec![0.0f64; 20];
        for v in data.iter_mut().skip(10) {
            *v = 100.0;
        }
        let img = Image::from_vec(&[20], data).unwrap();

        let out = zero_crossing_based_edge_detection(&img, &[1.0], &[0.01], 9, 3).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        assert!(vals.iter().all(|&v| v == 9.0 || v == 3.0));
        assert!(vals.contains(&9.0));
    }

    #[test]
    fn rejects_non_floating_point_pixel_type() {
        let img = Image::from_vec(&[4], vec![1u8, 2, 3, 4]).unwrap();
        assert_eq!(
            zero_crossing_based_edge_detection(&img, &[1.0], &[0.01], 1, 0).unwrap_err(),
            FilterError::RequiresRealPixelType(PixelId::UInt8)
        );
    }

    #[test]
    fn rejects_wrong_length_variance() {
        let img = Image::from_vec(&[4, 4], vec![0.0f64; 16]).unwrap();
        assert!(matches!(
            zero_crossing_based_edge_detection(&img, &[1.0], &[0.01, 0.01], 1, 0),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_maximum_error() {
        let img = Image::from_vec(&[4], vec![0.0f64; 4]).unwrap();
        assert!(matches!(
            zero_crossing_based_edge_detection(&img, &[1.0], &[1.5], 1, 0),
            Err(FilterError::InvalidMaximumError(_))
        ));
    }
}
