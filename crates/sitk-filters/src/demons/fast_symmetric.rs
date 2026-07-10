//! `FastSymmetricForcesDemonsRegistrationFilter`
//! (itkFastSymmetricForcesDemonsRegistrationFilter.h/.hxx), whose difference
//! function is `ESMDemonsRegistrationFunction` — see the `esm` module for the
//! PDE, its warped moving image, and the quirks that come with it.
//!
//! # How this differs from `DemonsRegistrationFilter`
//!
//! * **The displacement field is smoothed *after* the update is added**, at the
//!   end of `ApplyUpdate` (lines 215-219), rather than before the update is
//!   computed from it. `InitializeIteration` (lines 37-48) only hands the field
//!   to the difference function. So a one-iteration run from a zero field
//!   returns the *smoothed* update here, and the raw update there. Pinned by
//!   `smoothing_the_displacement_field_happens_after_the_update_is_added`.
//! * **`GetRMSChange()` is overridden** (line 99-106) to return the difference
//!   function's `m_RMSChange`, which the constructor sets to
//!   `NumericTraits<double>::max()`. `DemonsRegistrationFilter` does not
//!   override it, so it reports `FiniteDifferenceImageFilter::m_RMSChange{}`.
//!   The two agree after any iteration; they differ when none ran. Pinned by
//!   `a_zero_iteration_run_reports_the_functions_initial_rms_change`.
//! * The update is scaled by the time step only when `|dt - 1| > 1e-4`
//!   (lines 189-200). `ESMDemonsRegistrationFunction::ComputeGlobalTimeStep`
//!   returns `m_TimeStep`, constructed as `1.0` and never written, so
//!   `m_Multiplier` never runs.
//!
//! `AllocateUpdateBuffer` (lines 160-175) only re-declares the base's regions
//! and geometry; nothing observable follows from it.

use sitk_core::Image;

use super::DemonsResult;
use super::common::{halt, initial_field, output_field, per_axis, validate_image_pair};
use super::esm::{EsmFunction, EsmGradient};
use super::field::{Field, Smoothing, smooth_field};
use super::image_function::RealImage;
use crate::Result;

/// The yaml's parameter surface (`FastSymmetricForcesDemonsRegistrationFilter.yaml`).
/// [`Default`] carries the yaml defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct FastSymmetricForcesDemonsParams {
    /// Per-axis Gaussian standard deviations for smoothing the displacement
    /// field, in **pixel** units. Default `[1.0, 1.0, 1.0]`. Must hold at least
    /// one value per image dimension; extra values are ignored, as
    /// `sitkSTLVectorToITK` truncates.
    pub standard_deviations: Vec<f64>,
    /// Iteration cap. Default `10`.
    pub number_of_iterations: u32,
    /// Stop once an iteration's RMS change is strictly below this. Default
    /// `0.02`.
    pub maximum_rms_error: f64,
    /// Which image supplies the demons force's gradient. Default
    /// [`EsmGradient::Symmetric`].
    pub use_gradient_type: EsmGradient,
    /// Caps the update step: it scales the demons normalizer by its square.
    /// Default `0.5`. A value of `0.0` or less means "unrestricted" and drops
    /// the intensity term from the denominator entirely.
    pub maximum_update_step_length: f64,
    /// Gaussian-smooth the displacement field after each update is applied,
    /// giving an elastic regularisation. Default `true`.
    pub smooth_displacement_field: bool,
    /// Gaussian-smooth the update field before applying it, giving a viscous
    /// regularisation. Default `false`.
    pub smooth_update_field: bool,
    /// Per-axis standard deviations for smoothing the update field, in pixel
    /// units. Default `[1.0, 1.0, 1.0]`. Only read when
    /// [`FastSymmetricForcesDemonsParams::smooth_update_field`] is set.
    pub update_field_standard_deviations: Vec<f64>,
    /// Truncation limit on the Gaussian kernel's half-width. Default `30`.
    pub maximum_kernel_width: u32,
    /// Allowed area error of the discrete Gaussian approximation. Default `0.1`.
    /// `GaussianOperator::SetMaximumError` throws unless this lies strictly
    /// inside `(0, 1)`.
    pub maximum_error: f64,
    /// Below this absolute intensity difference a pixel's update is the zero
    /// vector — though it still counts toward the metric. Default `0.001`.
    pub intensity_difference_threshold: f64,
    /// Default `true`, and **inert** — see [`super`]'s notes on
    /// `m_ScaleCoefficients`.
    pub use_image_spacing: bool,
}

impl Default for FastSymmetricForcesDemonsParams {
    fn default() -> Self {
        FastSymmetricForcesDemonsParams {
            standard_deviations: vec![1.0; 3],
            number_of_iterations: 10,
            maximum_rms_error: 0.02,
            use_gradient_type: EsmGradient::Symmetric,
            maximum_update_step_length: 0.5,
            smooth_displacement_field: true,
            smooth_update_field: false,
            update_field_standard_deviations: vec![1.0; 3],
            maximum_kernel_width: 30,
            maximum_error: 0.1,
            intensity_difference_threshold: 0.001,
            use_image_spacing: true,
        }
    }
}

/// `FastSymmetricForcesDemonsRegistrationFilter`: register `moving` onto `fixed`
/// with the ESM demons force, returning the displacement field.
///
/// `fixed` and `moving` must be scalar images of the same pixel type and
/// dimension. `initial_displacement_field`, when given, must be a
/// `VectorFloat64` image with one component per dimension and the same size as
/// `fixed`; otherwise the field starts at zero.
///
/// Errors on a vector `fixed`/`moving`, on mismatched pixel types or
/// dimensions, on a `standard_deviations` or `update_field_standard_deviations`
/// shorter than the image dimension, on a `maximum_error` outside `(0, 1)`, and
/// on an initial field of the wrong pixel type, component count, or size.
pub fn fast_symmetric_forces_demons_registration(
    fixed: &Image,
    moving: &Image,
    initial_displacement_field: Option<&Image>,
    params: &FastSymmetricForcesDemonsParams,
) -> Result<DemonsResult> {
    validate_image_pair(fixed, moving, params.maximum_error)?;

    let dim = fixed.dimension();
    // `RealImage::new` runs `Image::to_f64_vec`, whose scalar guard rejects a
    // vector fixed or moving image.
    let fixed_real = RealImage::new(fixed)?;
    let moving_real = RealImage::new(moving)?;

    let displacement_smoothing = Smoothing {
        standard_deviations: per_axis(&params.standard_deviations, dim)?,
        maximum_error: params.maximum_error,
        maximum_kernel_width: params.maximum_kernel_width,
    };
    let update_smoothing = Smoothing {
        standard_deviations: per_axis(&params.update_field_standard_deviations, dim)?,
        maximum_error: params.maximum_error,
        maximum_kernel_width: params.maximum_kernel_width,
    };

    let mut field = initial_field(fixed, initial_displacement_field)?;
    let mut update = Field::zeros(fixed.size());
    let mut function = EsmFunction::new(
        &fixed_real,
        &moving_real,
        params.use_gradient_type,
        params.maximum_update_step_length,
        params.intensity_difference_threshold,
    );

    let pixels = field.number_of_pixels();
    let mut index = vec![0usize; dim];

    let mut elapsed_iterations = 0u32;
    // `FiniteDifferenceImageFilter::m_RMSChange{}`, which is what `Halt` reads.
    let mut halt_rms_change = 0.0f64;

    while !halt(
        elapsed_iterations,
        halt_rms_change,
        params.number_of_iterations,
        params.maximum_rms_error,
    ) {
        // `InitializeIteration` hands the current field to the function, which
        // warps the moving image through it. The field is *not* smoothed here.
        let warped = function.initialize_iteration(&field);

        // `DenseFiniteDifferenceImageFilter::CalculateChange`.
        for pixel in 0..pixels {
            field.multi_index(pixel, &mut index);
            function.compute_update(
                &warped,
                &index,
                field.vector_at(pixel),
                &mut update.data[pixel * dim..pixel * dim + dim],
            );
        }
        function.finish_iteration();

        // `FastSymmetricForcesDemonsRegistrationFilter::ApplyUpdate`.
        if params.smooth_update_field {
            smooth_field(&mut update, &update_smoothing);
        }
        // `m_Adder`: `output = output + update`. The `m_Multiplier` branch is
        // unreachable because `dt` is the constant `1.0`.
        for (component, change) in field.data.iter_mut().zip(&update.data) {
            *component += change;
        }
        halt_rms_change = function.rms_change;
        if params.smooth_displacement_field {
            smooth_field(&mut field, &displacement_smoothing);
        }

        elapsed_iterations += 1;
    }

    Ok(DemonsResult {
        displacement_field: output_field(fixed, initial_displacement_field, field)?,
        elapsed_iterations,
        // The overridden `GetRMSChange()`: the *function's* value, `f64::MAX`
        // until an iteration processes at least one pixel.
        rms_change: function.rms_change,
        metric: function.metric,
    })
}

#[cfg(test)]
mod tests;
