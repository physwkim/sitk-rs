//! The parts of `PDEDeformableRegistrationFilter` and
//! `FiniteDifferenceImageFilter` that every filter in the family shares:
//! input validation, the initial field, the halting rule, and the output's
//! geometry.

use sitk_core::{Image, PixelId};

use super::field::Field;
use crate::{FilterError, Result};

/// Take the first `dim` entries of a `dim_vec` parameter, as
/// `sitkSTLVectorToITK` does (sitkTemplateFunctions.h:97-112): it throws when
/// the vector is shorter than the image dimension and silently truncates when
/// it is longer.
pub(crate) fn per_axis(values: &[f64], dim: usize) -> Result<Vec<f64>> {
    if values.len() < dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: values.len(),
        });
    }
    Ok(values[..dim].to_vec())
}

/// SimpleITK casts both images to one `TImageType`, so they share a pixel type
/// and a dimension; `GaussianOperator::SetMaximumError` (itkGaussianOperator.h:86-95)
/// throws unless `maximum_error` lies strictly inside `(0, 1)`.
pub(crate) fn validate_image_pair(fixed: &Image, moving: &Image, maximum_error: f64) -> Result<()> {
    if fixed.pixel_id() != moving.pixel_id() {
        return Err(FilterError::TypeMismatch {
            a: fixed.pixel_id(),
            b: moving.pixel_id(),
        });
    }
    if fixed.dimension() != moving.dimension() {
        return Err(FilterError::ImageDimensionMismatch {
            a: fixed.dimension(),
            b: moving.dimension(),
        });
    }
    if !(maximum_error > 0.0 && maximum_error < 1.0) {
        return Err(FilterError::GaussianMaximumErrorOutOfRange(maximum_error));
    }
    Ok(())
}

/// `PDEDeformableRegistrationFilter::CopyInputToOutput`
/// (itkPDEDeformableRegistrationFilter.hxx:152-179): copy the initial field if
/// one is set, else fill with zeros.
pub(crate) fn initial_field(fixed: &Image, initial: Option<&Image>) -> Result<Field> {
    let dim = fixed.dimension();
    let Some(initial) = initial else {
        return Ok(Field::zeros(fixed.size()));
    };
    if initial.pixel_id() != PixelId::VectorFloat64 {
        return Err(FilterError::TypeMismatch {
            a: PixelId::VectorFloat64,
            b: initial.pixel_id(),
        });
    }
    if initial.number_of_components_per_pixel() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: initial.number_of_components_per_pixel(),
        });
    }
    if initial.size() != fixed.size() {
        return Err(FilterError::SizeMismatch {
            a: initial.size().to_vec(),
            b: fixed.size().to_vec(),
        });
    }
    Ok(Field {
        data: initial.component_slice::<f64>()?.to_vec(),
        size: fixed.size().to_vec(),
    })
}

/// `PDEDeformableRegistrationFilter::Halt` (which only adds the unreachable
/// `StopRegistration` flag) over `FiniteDifferenceImageFilter::Halt`
/// (itkFiniteDifferenceImageFilter.hxx:208-233).
///
/// The RMS test is a strict `>` on the *filter's* `RMSChange`, which each
/// filter's `ApplyUpdate` sets from its difference function after the update has
/// been applied.
pub(crate) fn halt(
    elapsed: u32,
    rms_change: f64,
    number_of_iterations: u32,
    maximum_rms_error: f64,
) -> bool {
    if elapsed >= number_of_iterations {
        return true;
    }
    if elapsed == 0 {
        return false;
    }
    maximum_rms_error > rms_change
}

/// Wrap the solved field as a `VectorFloat64` image.
///
/// `PDEDeformableRegistrationFilter::GenerateOutputInformation`
/// (itkPDEDeformableRegistrationFilter.hxx:182-206) copies the output's
/// information from the initial field when one is set, else from the fixed
/// image.
pub(crate) fn output_field(fixed: &Image, initial: Option<&Image>, field: Field) -> Result<Image> {
    let mut image = Image::from_vec_vector(fixed.size(), fixed.dimension(), field.data)?;
    image.copy_geometry_from(initial.unwrap_or(fixed));
    Ok(image)
}
