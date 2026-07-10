//! Normalized cross-correlation of an image against a fixed template, by
//! direct spatial evaluation of a mean-centered, unit-normalized
//! `itk::NeighborhoodOperator` at every pixel.
//!
//! Ported from `itk::NormalizedCorrelationImageFilter`
//! (itkNormalizedCorrelationImageFilter.h/.hxx), a subclass of
//! `NeighborhoodOperatorImageFilter`, plus SimpleITK's own
//! `CreateOperatorFromImage` helper (`sitkImageToKernel.hxx`) that turns the
//! `TemplateImage` input into that operator. Parameter names follow
//! `NormalizedCorrelationImageFilter.yaml` (`Image`, `MaskImage`,
//! `TemplateImage`, the last with `no_size_check: true`).
//!
//! This is a *different* filter from [`crate::fft_correlation`]'s
//! `FFTNormalizedCorrelationImageFilter`/`MaskedFFTNormalizedCorrelationImageFilter`:
//! those compute correlation at every possible shift between two whole
//! images via FFT; this one correlates one fixed, small template against a
//! neighborhood centered at every output pixel, in the spatial domain.
//!
//! # Building the template operator
//!
//! `CreateOperatorFromImage` (sitkImageToKernel.hxx:37-85) requires an odd
//! extent along every axis (`ImageKernelOperator::GenerateCoefficients`,
//! itkImageKernelOperator.hxx:68-77), so it first `ConstantPadImageFilter`s a
//! single zero onto the **lower** side of every even axis
//! (sitkImageToKernel.hxx:55-65: `padSize[i] = 1 - size[i] % 2`), then calls
//! `CreateToRadius(radius)` with `radius[i] = paddedSize[i] / 2`
//! (sitkImageToKernel.hxx:76-82). Since an even `size` and its one-larger
//! padded extent floor-divide by two to the same quotient, `radius[i] =
//! size[i] / 2` computed on the *raw* template size already gives the right
//! answer — this port never materializes the padded image, and instead reads
//! [`padded_template_coefficients`] straight off the raw buffer, substituting
//! `0.0` for the padding cell. **This means an even-extent template's own
//! zero pad is a real data point in the mean/variance below**, not just
//! boundary filler for the image neighborhood — the `EvenKernel` yaml test
//! tag exists because this is a deliberate, tested upstream behavior, not a
//! degenerate accident.
//!
//! `NeighborhoodOperator::Fill` copies `GenerateCoefficients()`'s buffer into
//! the operator's storage via a `std::slice` over the *whole* neighborhood,
//! with no reordering (itkImageKernelOperator.hxx:86-104): both the image
//! buffer and `itk::Neighborhood` share the same dimension-0-fastest layout
//! (itkNeighborhood.hxx:41-67), so a raw template pixel at box-position `m`
//! lands at exactly that same relative offset from center. This crate's own
//! [`sitk_core::NeighborhoodIterator`] windows use that identical order
//! (`Neighborhood::values`'s doc), so [`padded_template_coefficients`]'s
//! output lines up element-for-element with a gathered image window with no
//! reindexing on either side — and, critically, **no flip**: unlike
//! [`crate::convolution::convolution`] (which reverses the kernel buffer
//! before padding, `itkConvolutionImageFilter.hxx:96-125`), this filter
//! reads the template buffer forwards. This is genuine cross-*correlation*,
//! not convolution.
//!
//! # The per-pixel formula
//!
//! `DynamicThreadedGenerateData` (itkNormalizedCorrelationImageFilter.hxx:89-242)
//! first normalizes the (possibly zero-padded) template to zero mean and unit
//! L2 norm, computed over all `num = Π(2·radius[d]+1)` of its coefficients:
//!
//! ```text
//! mean = Σt / num
//! var  = (Σt² − (Σt)²/num) / (num − 1)
//! k    = sqrt(var) · sqrt(num − 1)
//! t'[i] = (t[i] − mean) / k                    // Σ t'[i]² == 1
//! ```
//!
//! Then, for every output pixel whose `NeighborhoodIterator` window `v` (same
//! `num`-sized box, [`sitk_core::ZeroFluxNeumannBoundaryCondition`] at the
//! image border — `NeighborhoodOperatorImageFilter`'s
//! `DefaultBoundaryCondition`, itkNeighborhoodOperatorImageFilter.h:93, and
//! this filter's yaml exposes no way to override it):
//!
//! ```text
//! numerator   = Σ v[i]·t'[i]
//! denominator = sqrt(Σv[i]² − (Σv[i])²/num)
//! output      = numerator / denominator
//! ```
//!
//! A pixel outside `mask` (when one is given) is set to `0.0` instead
//! (itkNormalizedCorrelationImageFilter.hxx:230-234) — never computed at
//! all, so it costs nothing extra to check first.
//!
//! Neither division is guarded upstream: a constant-valued template makes
//! `k == 0.0` (so every `t'[i]` is `0.0 / 0.0 == NaN`, propagating to every
//! output pixel), and a locally-constant image neighborhood makes
//! `denominator == 0.0` — and, since `Σt'[i] == 0` exactly, `numerator` is
//! *also* exactly `0.0` there, so the quotient is `NaN` rather than `±∞`.
//! This port reproduces both by doing the same `f64` arithmetic ITK does in
//! `OutputPixelRealType` (`NumericTraits<OutputPixelType>::RealType`, itself
//! always `f32`/`f64` since [`crate::real_pixel_id`] is `OutputPixelType`)
//! with no extra guard.
//!
//! # Output pixel type
//!
//! `output_pixel_type: typename itk::NumericTraits<InputImageType::PixelType>::RealType`,
//! the same expression [`crate::intensity::normalize_to_constant`] resolves
//! through [`crate::real_pixel_id`] — including that helper's own tracked
//! `Float32 → Float32` divergence from ITK's true `RealType` rule (always
//! `double`); see its doc comment and upstream-findings ledger §5.6.
//!
//! # Mask
//!
//! `SetMaskImage` registers the mask as `ProcessObject` input 1
//! (itkNormalizedCorrelationImageFilter.hxx:30-36), and
//! `NormalizedCorrelationImageFilter` does not override
//! `ImageToImageFilter::VerifyInputInformation`, so the mask is still subject
//! to the base class's same-physical-space precondition
//! (itkImageToImageFilter.hxx:148-...) — reproduced here as
//! [`FilterError::PhysicalSpaceMismatch`] with `index: 1`, ITK's own input
//! number for the mask. The `TemplateImage` input never goes through
//! `SetInput`/`SetNthInput` at all (SimpleITK's generated wrapper converts it
//! to a `NeighborhoodOperator` and calls `SetTemplate` instead, entirely
//! outside the registered-input pipeline), so no such check applies to it —
//! only its dimension is required to match (`SameDimensionCheck`).
//!
//! **Divergence:** `NormalizedCorrelationImageFilter.yaml` declares
//! `MaskImage` as a required input (no `optional: true`, unlike e.g.
//! `ConnectedComponentImageFilter.yaml`'s own `MaskImage`), so SimpleITK's
//! generated procedural wrapper cannot be called without a mask at all, even
//! though the underlying `itk::NormalizedCorrelationImageFilter` genuinely
//! supports a null one (`if (!mask) { ... }`,
//! itkNormalizedCorrelationImageFilter.hxx:167-198) and its own class doc
//! calls masking optional. This port exposes `mask: Option<&Image>`, matching
//! this crate's established convention for every other optional-mask filter
//! (`n4_bias_field`, `fft_correlation`, `scalar_connected_component`,
//! `stochastic_fractal_dimension`) rather than SimpleITK's incidentally more
//! restrictive generated signature.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{Image, NeighborhoodIterator, ZeroFluxNeumannBoundaryCondition};

/// `ImageToImageFilter`'s default `GlobalDefaultCoordinateTolerance` and
/// `GlobalDefaultDirectionTolerance` (`itkImageToImageFilter.h`), neither of
/// which this filter chain overrides.
const COORDINATE_TOLERANCE: f64 = 1e-6;
const DIRECTION_TOLERANCE: f64 = 1e-6;

/// `ImageBase::IsCongruentImageGeometry` (`itkImageBase.hxx:391-406`): origin
/// and spacing compared with a tolerance scaled by `primary`'s first-axis
/// spacing, direction compared with a flat tolerance.
fn same_physical_space(primary: &Image, other: &Image) -> bool {
    let coord_tol = (COORDINATE_TOLERANCE * primary.spacing()[0]).abs();
    let origin_ok = primary
        .origin()
        .iter()
        .zip(other.origin())
        .all(|(a, b)| (a - b).abs() <= coord_tol);
    let spacing_ok = primary
        .spacing()
        .iter()
        .zip(other.spacing())
        .all(|(a, b)| (a - b).abs() <= coord_tol);
    let direction_ok = primary
        .direction()
        .iter()
        .zip(other.direction())
        .all(|(a, b)| (a - b).abs() <= DIRECTION_TOLERANCE);
    origin_ok && spacing_ok && direction_ok
}

/// Decompose linear offset `i` into a multi-index of `size`, first index
/// fastest (matching [`Image`]'s layout).
fn unravel(mut i: usize, size: &[usize], out: &mut [usize]) {
    for (o, &s) in out.iter_mut().zip(size) {
        *o = i % s;
        i /= s;
    }
}

/// Linear offset of a multi-index within `size`, first index fastest.
fn ravel(index: &[usize], size: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&i, &s) in index.iter().zip(size) {
        offset += i * stride;
        stride *= s;
    }
    offset
}

/// `CreateOperatorFromImage`'s coefficient buffer (see the module docs): the
/// raw template values, zero-padded on the low side of every even axis to
/// extent `2*radius[d]+1`, in the same dimension-0-fastest order
/// [`sitk_core::Neighborhood::values`] uses.
fn padded_template_coefficients(
    template_values: &[f64],
    template_size: &[usize],
    radius: &[usize],
) -> Vec<f64> {
    let dim = template_size.len();
    let window_size: Vec<usize> = radius.iter().map(|&r| 2 * r + 1).collect();
    // 1 where the template extent is even; `padSize[i] = 1 - size[i] % 2`.
    let pad: Vec<usize> = template_size.iter().map(|&s| 1 - s % 2).collect();

    let count: usize = window_size.iter().product();
    let mut coeffs = vec![0.0f64; count];
    let mut m = vec![0usize; dim];
    let mut template_index = vec![0usize; dim];
    for (i, slot) in coeffs.iter_mut().enumerate() {
        unravel(i, &window_size, &mut m);
        if m.iter().zip(&pad).any(|(&mi, &p)| mi < p) {
            continue; // the zero the lower pad introduced
        }
        for ((t, &mi), &p) in template_index.iter_mut().zip(&m).zip(&pad) {
            *t = mi - p;
        }
        *slot = template_values[ravel(&template_index, template_size)];
    }
    coeffs
}

/// Mean-center and unit-normalize the template operator's coefficients
/// (itkNormalizedCorrelationImageFilter.hxx:99-133): `k` is chosen so that
/// `Σ normalized[i]² == 1`.
fn normalize_template(coeffs: &[f64]) -> Vec<f64> {
    let num = coeffs.len() as f64;
    let sum: f64 = coeffs.iter().sum();
    let sum_of_squares: f64 = coeffs.iter().map(|v| v * v).sum();
    let mean = sum / num;
    let var = (sum_of_squares - sum * sum / num) / (num - 1.0);
    let std = var.sqrt();
    let k = std * (num - 1.0).sqrt();
    coeffs.iter().map(|&v| (v - mean) / k).collect()
}

/// `NormalizedCorrelationImageFilter`: correlate `image` against `template`
/// (mean-centered and unit-normalized first), optionally restricted to the
/// pixels where `mask` is non-zero.
///
/// The output has `image`'s geometry and
/// `NumericTraits<image's pixel type>::RealType` as its pixel type (see the
/// module docs for both that type and the [`FilterError`] this raises).
pub fn normalized_correlation(
    image: &Image,
    mask: Option<&Image>,
    template: &Image,
) -> Result<Image> {
    if template.dimension() != image.dimension() {
        return Err(FilterError::KernelDimensionMismatch {
            image: image.dimension(),
            kernel: template.dimension(),
        });
    }
    if template.size().contains(&0) {
        return Err(FilterError::EmptyKernel(template.size().to_vec()));
    }
    if let Some(mask) = mask {
        if mask.size() != image.size() {
            return Err(FilterError::SizeMismatch {
                a: image.size().to_vec(),
                b: mask.size().to_vec(),
            });
        }
        if !same_physical_space(image, mask) {
            return Err(FilterError::PhysicalSpaceMismatch { index: 1 });
        }
    }

    let template_size = template.size();
    let radius: Vec<usize> = template_size.iter().map(|&s| s / 2).collect();
    let template_values = template.to_f64_vec()?;
    let coeffs = padded_template_coefficients(&template_values, template_size, &radius);
    let normalized_template = normalize_template(&coeffs);
    let real_template_size = coeffs.len() as f64;

    let widened = Image::from_vec(image.size(), image.to_f64_vec()?)?;
    let mask_values: Option<Vec<f64>> = mask.map(|m| m.to_f64_vec()).transpose()?;

    let iter =
        NeighborhoodIterator::<f64, _>::new(&widened, &radius, ZeroFluxNeumannBoundaryCondition)?;
    let mut out = Vec::with_capacity(image.number_of_pixels());
    for (i, (_, nb)) in iter.enumerate() {
        if let Some(mask_values) = &mask_values {
            if mask_values[i] == 0.0 {
                out.push(0.0);
                continue;
            }
        }
        let values = nb.values();
        let numerator: f64 = values
            .iter()
            .zip(&normalized_template)
            .map(|(&v, &t)| v * t)
            .sum();
        let sum: f64 = values.iter().sum();
        let sum_of_squares: f64 = values.iter().map(|&v| v * v).sum();
        let denominator = (sum_of_squares - sum * sum / real_template_size).sqrt();
        out.push(numerator / denominator);
    }

    let output_id = crate::real_pixel_id(image.pixel_id());
    image_from_f64(output_id, image.size(), image, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    fn img_f32(size: &[usize], data: Vec<f32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    /// Odd-extent template, no mask: hand-computed at an interior pixel and
    /// at a `ZeroFluxNeumannBoundaryCondition`-clamped edge pixel (values
    /// derived independently in Python; see the task notes for the
    /// derivation of `mean`/`var`/`k`/`numerator`/`denominator`).
    #[test]
    fn odd_template_matches_hand_computed_values_interior_and_boundary() {
        let image = img_f32(&[7], vec![1.0, 2.0, 3.0, 10.0, 3.0, 2.0, 1.0]);
        let template = img_f32(&[3], vec![1.0, 0.0, 2.0]);
        let out = normalized_correlation(&image, None, &template).unwrap();
        let got = out.scalar_slice::<f32>().unwrap();
        assert!((got[3] as f64 - (-0.8660254037844384)).abs() < 1e-6);
        assert!((got[0] as f64 - 0.8660254037844384).abs() < 1e-6);
    }

    /// An even-extent template is zero-padded on the *low* side before its
    /// own mean/variance are computed, so the padding zero is a real data
    /// point in the normalization -- the module docs' central quirk, and the
    /// reason `NormalizedCorrelationImageFilter.yaml` has an explicit
    /// `EvenKernel` test tag upstream. `[2.0, 4.0]` pads to `[0.0, 2.0, 4.0]`
    /// (radius 1), not `[2.0, 4.0, 0.0]` or an unpadded 2-tap operator.
    #[test]
    fn even_template_is_zero_padded_on_the_low_side_before_normalizing() {
        let image = img_f32(&[5], vec![5.0, 1.0, 8.0, 2.0, 9.0]);
        let template = img_f32(&[2], vec![2.0, 4.0]);
        let out = normalized_correlation(&image, None, &template).unwrap();
        let got = out.scalar_slice::<f32>().unwrap();
        assert!((got[2] as f64 - 0.13206763594884358).abs() < 1e-6);
    }

    /// A pixel outside the mask is forced to exactly `0.0`, never computed;
    /// an unmasked pixel gets the same value it would without a mask.
    #[test]
    fn masked_out_pixels_are_forced_to_zero_others_are_unaffected() {
        let image = img_f32(&[5], vec![5.0, 1.0, 8.0, 2.0, 9.0]);
        let template = img_f32(&[3], vec![1.0, 0.0, 2.0]);
        let mask = img_f32(&[5], vec![1.0, 1.0, 0.0, 1.0, 1.0]);

        let unmasked = normalized_correlation(&image, None, &template).unwrap();
        let masked = normalized_correlation(&image, Some(&mask), &template).unwrap();

        assert_eq!(masked.scalar_slice::<f32>().unwrap()[2], 0.0);
        assert_eq!(
            masked.scalar_slice::<f32>().unwrap()[1],
            unmasked.scalar_slice::<f32>().unwrap()[1]
        );
    }

    /// A locally constant image neighborhood makes both `numerator` and
    /// `denominator` exactly `0.0` (the normalized template always sums to
    /// zero), so the quotient is `NaN`, not `0.0` or `±inf` -- neither ITK
    /// nor this port guards the division.
    #[test]
    fn a_flat_neighborhood_produces_nan_not_a_guarded_zero() {
        let image = img_f32(&[5], vec![4.0, 4.0, 4.0, 4.0, 4.0]);
        let template = img_f32(&[3], vec![1.0, 0.0, 2.0]);
        let out = normalized_correlation(&image, None, &template).unwrap();
        assert!(out.scalar_slice::<f32>().unwrap()[2].is_nan());
    }

    /// A constant-valued template makes `k == 0.0`, so every normalized
    /// coefficient is `0.0 / 0.0 == NaN`, propagating to every output pixel
    /// regardless of the image.
    #[test]
    fn a_constant_template_produces_nan_everywhere() {
        let image = img_f32(&[5], vec![5.0, 1.0, 8.0, 2.0, 9.0]);
        let template = img_f32(&[3], vec![3.0, 3.0, 3.0]);
        let out = normalized_correlation(&image, None, &template).unwrap();
        assert!(out.scalar_slice::<f32>().unwrap()[2].is_nan());
    }

    #[test]
    fn template_dimension_mismatch_is_rejected() {
        let image = img_f32(&[4, 4], vec![0.0; 16]);
        let template = img_f32(&[3], vec![1.0, 0.0, 2.0]);
        let err = normalized_correlation(&image, None, &template).unwrap_err();
        assert_eq!(
            err,
            FilterError::KernelDimensionMismatch {
                image: 2,
                kernel: 1
            }
        );
    }

    #[test]
    fn mask_size_mismatch_is_rejected() {
        let image = img_f32(&[5], vec![5.0, 1.0, 8.0, 2.0, 9.0]);
        let template = img_f32(&[3], vec![1.0, 0.0, 2.0]);
        let mask = img_f32(&[4], vec![1.0, 1.0, 1.0, 1.0]);
        let err = normalized_correlation(&image, Some(&mask), &template).unwrap_err();
        assert_eq!(
            err,
            FilterError::SizeMismatch {
                a: vec![5],
                b: vec![4],
            }
        );
    }

    #[test]
    fn empty_template_axis_is_rejected() {
        let image = img_f32(&[5], vec![5.0, 1.0, 8.0, 2.0, 9.0]);
        let template = Image::from_vec::<f32>(&[0], vec![]).unwrap();
        let err = normalized_correlation(&image, None, &template).unwrap_err();
        assert_eq!(err, FilterError::EmptyKernel(vec![0]));
    }

    #[test]
    fn non_scalar_image_is_rejected() {
        let image = Image::from_vec_vector(&[2, 1], 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let template = img_f32(&[3, 1], vec![1.0, 0.0, 2.0]);
        let err = normalized_correlation(&image, None, &template).unwrap_err();
        assert_eq!(
            err,
            sitk_core::Error::RequiresScalarPixelType(PixelId::VectorFloat32).into()
        );
    }

    /// Output pixel type follows [`crate::real_pixel_id`]: a `Float32` input
    /// stays `Float32` (this port's tracked divergence from ITK's true
    /// `RealType` rule -- see the module docs and ledger §5.6), an integer
    /// input promotes to `Float64`.
    #[test]
    fn output_pixel_type_follows_real_pixel_id() {
        let image = img_f32(&[5], vec![5.0, 1.0, 8.0, 2.0, 9.0]);
        let template = img_f32(&[3], vec![1.0, 0.0, 2.0]);
        let out = normalized_correlation(&image, None, &template).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);

        let int_image = Image::from_vec(&[5], vec![5i32, 1, 8, 2, 9]).unwrap();
        let out_int = normalized_correlation(&int_image, None, &template).unwrap();
        assert_eq!(out_int.pixel_id(), PixelId::Float64);
    }
}
