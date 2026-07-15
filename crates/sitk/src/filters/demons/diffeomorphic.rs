//! `DiffeomorphicDemonsRegistrationFilter`
//! (itkDiffeomorphicDemonsRegistrationFilter.h/.hxx), Vercauteren's variant.
//!
//! The PDE is `ESMDemonsRegistrationFunction`'s, the same one
//! [`super::fast_symmetric_forces_demons_registration`] uses — see the `esm`
//! module. What differs is how the update is applied. Where the other filters
//! add it,
//!
//! ```text
//! s <- s + u
//! ```
//!
//! this one *composes* it, so that each step is a diffeomorphism:
//!
//! ```text
//! s <- s ∘ exp(u),    s_next[i] = s(p_i + e[i]) + e[i],   e = exp(u)
//! ```
//!
//! `exp` is `ExponentialDisplacementFieldImageFilter`'s scaling and squaring;
//! the composition is a `WarpVectorImageFilter` followed by an
//! `AddImageFilter`. Both live in the `compose` module, with their own quirks
//! documented there.
//!
//! # How many squaring steps
//!
//! `ApplyUpdate` (lines 205-243) does *not* let the exponentiator choose. When
//! [`DiffeomorphicDemonsParams::maximum_update_step_length`] is positive it
//! imposes
//!
//! ```text
//! numiterfloat = 2 + log2(maximum_update_step_length)
//! numiter      = numiterfloat > 0 ? ceil(numiterfloat) : 0
//! ```
//!
//! — the count that makes `max(norm(Φ))/2^N <= 0.25 · pixel spacing`, per the
//! comment — and turns the automatic count off. At the yaml default of `0.5`
//! that is exactly one squaring step. At `0.25` or below it is *zero*, which
//! makes `exp(u) = u` and the update rule silently first-order. Pinned by
//! `a_short_enough_step_length_makes_the_exponential_the_identity`.
//!
//! Only when the step length is zero or negative does the exponentiator pick
//! its own count, capped at `2000`.
//!
//! # Upstream behaviour reproduced here
//!
//! * **The displacement field is smoothed after the composition**, at the end
//!   of `ApplyUpdate` (lines 275-279), as in
//!   `FastSymmetricForcesDemonsRegistrationFilter` and unlike
//!   `DemonsRegistrationFilter`.
//! * **`GetRMSChange()` is overridden** (lines 132-139) to return the
//!   difference function's `m_RMSChange`, `NumericTraits<double>::max()` until
//!   an iteration processes a pixel.
//! * The `m_Multiplier` branch (lines 190-199) runs only when `|dt - 1| >
//!   1e-4`; `ESMDemonsRegistrationFunction::ComputeGlobalTimeStep` returns the
//!   constant `1.0`, so it never does.
//! * The RMS change and the metric are the ones the *function* accumulated over
//!   the raw update `u`, not over the composed exponential `e`.

use crate::core::Image;

use super::DemonsResult;
use super::common::{halt, initial_field, output_field, per_axis, validate_verifying_image_pair};
use super::compose::{exponential, warp};
use super::esm::{EsmFunction, EsmGradient};
use super::field::{Field, Smoothing, smooth_field};
use super::geometry::Geometry;
use super::image_function::RealImage;
use crate::filters::Result;

/// The yaml's parameter surface (`DiffeomorphicDemonsRegistrationFilter.yaml`).
/// [`Default`] carries the yaml defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffeomorphicDemonsParams {
    /// Per-axis Gaussian standard deviations for smoothing the displacement
    /// field, in **pixel** units. Default `[1.0, 1.0, 1.0]`.
    pub standard_deviations: Vec<f64>,
    /// Iteration cap. Default `10`.
    pub number_of_iterations: u32,
    /// Stop once an iteration's RMS change is strictly below this. Default
    /// `0.02`.
    pub maximum_rms_error: f64,
    /// Which image supplies the demons force's gradient. Default
    /// [`EsmGradient::Symmetric`].
    pub use_gradient_type: EsmGradient,
    /// Skip the exponential and compose with `Id + u` directly, i.e. use the
    /// first-order approximation `exp(u) ≈ u`. Default `false`.
    pub use_first_order_exp: bool,
    /// Caps the update step. It does double duty: it scales the demons
    /// normalizer by its square, *and* it fixes the exponential's squaring
    /// count. Default `0.5`. A value of `0.0` or less means "unrestricted" —
    /// the denominator drops its intensity term and the exponential picks its
    /// own count.
    pub maximum_update_step_length: f64,
    /// Gaussian-smooth the displacement field after each update is composed,
    /// giving an elastic regularisation. Default `true`.
    pub smooth_displacement_field: bool,
    /// Gaussian-smooth the update field before composing it, giving a viscous
    /// regularisation. Default `false`.
    pub smooth_update_field: bool,
    /// Per-axis standard deviations for smoothing the update field, in pixel
    /// units. Default `[1.0, 1.0, 1.0]`.
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

impl Default for DiffeomorphicDemonsParams {
    fn default() -> Self {
        DiffeomorphicDemonsParams {
            standard_deviations: vec![1.0; 3],
            number_of_iterations: 10,
            maximum_rms_error: 0.02,
            use_gradient_type: EsmGradient::Symmetric,
            use_first_order_exp: false,
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

/// The squaring count `ApplyUpdate` imposes on the exponentiator
/// (itkDiffeomorphicDemonsRegistrationFilter.hxx:222-242), and whether the
/// exponentiator is left to choose its own.
///
/// Returns `(automatic, maximum_number_of_iterations)`.
fn squaring_steps(maximum_update_step_length: f64) -> (bool, u32) {
    if maximum_update_step_length > 0.0 {
        let numiterfloat = 2.0 + maximum_update_step_length.ln() / std::f64::consts::LN_2;
        let numiter = if numiterfloat > 0.0 {
            numiterfloat.ceil() as u32
        } else {
            0
        };
        (false, numiter)
    } else {
        // "just set a high value so that automatic number of step is not
        // thresholded".
        (true, 2000)
    }
}

/// `DiffeomorphicDemonsRegistrationFilter`: register `moving` onto `fixed` with
/// the ESM demons force composed through the exponential, returning the
/// displacement field.
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
pub fn diffeomorphic_demons_registration(
    fixed: &Image,
    moving: &Image,
    initial_displacement_field: Option<&Image>,
    params: &DiffeomorphicDemonsParams,
) -> Result<DemonsResult> {
    validate_verifying_image_pair(fixed, moving, params.maximum_error)?;

    let dim = fixed.dimension();
    let fixed_real = RealImage::new(fixed)?;
    let moving_real = RealImage::new(moving)?;

    // `GenerateOutputInformation` gives the output — and so the update buffer,
    // the warper and the exponentiator — the initial field's geometry when one
    // is set, else the fixed image's.
    let field_geometry = Geometry::new(initial_displacement_field.unwrap_or(fixed))?;

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

    let (automatic, maximum_squaring_steps) = squaring_steps(params.maximum_update_step_length);

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
        // `InitializeIteration` hands the field to the function, which warps
        // the moving image through it. The field is *not* smoothed here.
        let warped_moving = function.initialize_iteration(&field);

        // `DenseFiniteDifferenceImageFilter::CalculateChange`.
        for pixel in 0..pixels {
            field.multi_index(pixel, &mut index);
            function.compute_update(
                &warped_moving,
                &index,
                field.vector_at(pixel),
                &mut update.data[pixel * dim..pixel * dim + dim],
            );
        }
        function.finish_iteration();

        // `ApplyUpdate`.
        if params.smooth_update_field {
            smooth_field(&mut update, &update_smoothing);
        }
        let step = if params.use_first_order_exp {
            // `s <- s ∘ (Id + u)`: the warper takes the update buffer directly.
            update.clone()
        } else {
            exponential(&update, &field_geometry, automatic, maximum_squaring_steps)
        };
        // `m_Warper` then `m_Adder`: `s <- s ∘ (Id + step) + step`.
        field = warp(&field, &step, &field_geometry);
        for (component, &change) in field.data.iter_mut().zip(&step.data) {
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
