//! PDE deformable registration: Thirion's demons and its variants.
//!
//! Ported from ITK's `Modules/Registration/PDEDeformable`:
//!
//! | Layer | ITK source |
//! |---|---|
//! | the solver loop and halting | `itkFiniteDifferenceImageFilter.hxx` (`GenerateData`, `Halt`) |
//! | update application | `itkDenseFiniteDifferenceImageFilter.hxx` (`ThreadedCalculateChange`, `ThreadedApplyUpdate`) |
//! | field smoothing, zero initial field | `itkPDEDeformableRegistrationFilter.hxx` (`SmoothDisplacementField`, `SmoothUpdateField`, `CopyInputToOutput`) |
//! | the smoothing kernel | `itkGaussianOperator.hxx` (`GenerateCoefficients`) |
//! | per-iteration hooks | `itkDemonsRegistrationFilter.hxx` (`InitializeIteration`, `ApplyUpdate`) |
//! | the PDE | `itkDemonsRegistrationFunction.hxx` (`ComputeUpdate`, `InitializeIteration`, `ReleaseGlobalDataPointer`) |
//! | this module's public surface | `DemonsRegistrationFilter.yaml` |
//!
//! [`fast_symmetric_forces_demons_registration`] is the same solver over
//! `ESMDemonsRegistrationFunction`, and
//! [`symmetric_forces_demons_registration`] over
//! `SymmetricForcesDemonsRegistrationFunction`. Every filter in the family
//! reports the same three measurements, so they share [`DemonsResult`].
//!
//! The rest of this page describes [`demons_registration`] itself.
//!
//! # Inputs and output
//!
//! Two scalar images of the same pixel type — SimpleITK casts both to one
//! `TImageType` — plus an optional initial displacement field. The output is a
//! `VectorFloat64` displacement field with one component per dimension, mapping
//! the moving image onto the fixed one: the moving image is sampled at
//! `fixed_point + displacement`.
//!
//! # The equation
//!
//! At each pixel, with `f` the fixed value, `m` the moving value interpolated at
//! the displaced point, `∇` the gradient of the fixed image (or the moving one
//! under [`DemonsParams::use_moving_image_gradient`]), and
//! `K = mean(spacing_k²)` the fixed image's mean square spacing:
//!
//! ```text
//! speed       = f - m
//! denominator = speed² / K + |∇|²
//! update      = speed * ∇ / denominator
//! ```
//!
//! `K` exists to reconcile the units of the two denominator terms: `speed²` is
//! intensity², while `|∇|²` is intensity²/mm². The update is the zero vector
//! when the pixel maps outside the moving image's buffer, when
//! `|speed| < intensity_difference_threshold`, or when `denominator` falls
//! below the hard-coded `1e-9`
//! (`m_DenominatorThreshold`, itkDemonsRegistrationFunction.hxx:39).
//!
//! Each iteration then, in order (`FiniteDifferenceImageFilter::GenerateData`,
//! lines 71-90):
//!
//! 1. resets the metric accumulators and, if
//!    [`DemonsParams::smooth_displacement_field`], Gaussian-smooths the current
//!    field — `DemonsRegistrationFilter::InitializeIteration` smooths the field
//!    *before* the update is computed from it, not after it is applied;
//! 2. computes the update field, accumulating the metric and the RMS change;
//! 3. if [`DemonsParams::smooth_update_field`], Gaussian-smooths that update,
//!    then adds `update * dt` to the field. `dt` is the constant `1.0`
//!    (`DemonsRegistrationFunction::ComputeGlobalTimeStep`,
//!    itkDemonsRegistrationFunction.h:134-138).
//!
//! # Upstream behaviour reproduced here
//!
//! * **`use_image_spacing` is inert.** It reaches
//!   `FiniteDifferenceImageFilter::InitializeFunctionCoefficients`, which calls
//!   `SetScaleCoefficients` on the difference function — and no function in
//!   `Modules/Registration/PDEDeformable` ever reads `m_ScaleCoefficients` or
//!   calls `ComputeNeighborhoodScales`. The Demons PDE takes its spacing from
//!   the fixed image directly (the normalizer `K` and the gradient calculator).
//!   The flag is part of the yaml's surface, so it is part of this function's
//!   surface; it changes nothing. Pinned by
//!   `use_image_spacing_is_inert`.
//! * **The metric denominator counts pixels, not iterations.** `m_Metric` and
//!   `m_RMSChange` are `sum / m_NumberOfPixelsProcessed`, and a pixel is
//!   "processed" only if it maps inside the moving image's buffer — but it is
//!   counted *before* the intensity-difference and denominator thresholds are
//!   tested, so thresholded pixels contribute a zero change and still divide.
//! * **`metric` and `rms_change` are stale by one iteration** in the sense the
//!   yaml documents: they are the values accumulated while computing the update
//!   that was last applied, and the field has moved since.
//! * If no pixel maps inside the moving image, `m_Metric` and `m_RMSChange` keep
//!   their previous values — initially `f64::MAX`
//!   (itkDemonsRegistrationFunction.hxx:52-56). The filter's own `RMSChange`,
//!   however, starts at `0.0` (`FiniteDifferenceImageFilter`'s `m_RMSChange{}`),
//!   which is what a zero-iteration run reports.
//! * **Halting** (`FiniteDifferenceImageFilter::Halt`, lines 208-233) stops when
//!   `elapsed >= number_of_iterations`, or — after at least one iteration —
//!   when `maximum_rms_error > rms_change`. The RMS test is a strict `>` on the
//!   *filter's* `RMSChange`, which `DemonsRegistrationFilter::ApplyUpdate` sets
//!   from the function after each update.
//!
//! # Not ported
//!
//! `StopRegistration()` is a yaml "measurement" that asks the filter to halt
//! after the current iteration. It is only reachable from an ITK observer
//! attached mid-`Update()`; this port runs to completion synchronously, so
//! there is no point at which a caller could invoke it.
//!
//! # An upstream defect, unreachable from here
//!
//! `PDEDeformableRegistrationFilter::SetUpdateFieldStandardDeviations(double)`
//! writes `m_StandardDeviations`, not `m_UpdateFieldStandardDeviations`
//! (itkPDEDeformableRegistrationFilter.hxx:99-104) — a copy-paste of the
//! `SetStandardDeviations` overload three lines above. SimpleITK never calls it:
//! its `dim_vec` member is converted with `sitkSTLVectorToITK` and passed to the
//! `FixedArray` overload generated by `itkSetMacro`
//! (`ExecuteInternalSetITKFilterParameters.cxx.jinja:5`), so the scalar overload
//! is dead from SimpleITK's side. This port takes a per-axis vector and applies
//! it to the update field, as the yaml documents.

mod common;
mod compose;
mod diffeomorphic;
mod esm;
mod fast_symmetric;
mod field;
mod geometry;
mod image_function;
mod level_set_motion;
mod symmetric;

use crate::core::Image;

use crate::filters::Result;
use common::{halt, initial_field, output_field, per_axis, validate_image_pair};
use field::{Field, Smoothing, smooth_field};
use image_function::RealImage;

pub use diffeomorphic::{DiffeomorphicDemonsParams, diffeomorphic_demons_registration};
pub use esm::EsmGradient;
pub use fast_symmetric::{
    FastSymmetricForcesDemonsParams, fast_symmetric_forces_demons_registration,
};
pub use level_set_motion::{LevelSetMotionParams, level_set_motion_registration};
pub use symmetric::{SymmetricForcesDemonsParams, symmetric_forces_demons_registration};

/// The yaml's parameter surface. [`Default`] carries the yaml defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct DemonsParams {
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
    /// Take the demons force's gradient from the moving image (evaluated at the
    /// displaced point) rather than the fixed image (evaluated at the pixel
    /// index). Default `false`.
    pub use_moving_image_gradient: bool,
    /// Gaussian-smooth the displacement field each iteration, giving an elastic
    /// regularisation. Default `true`.
    pub smooth_displacement_field: bool,
    /// Gaussian-smooth the update field before applying it, giving a viscous
    /// regularisation. Default `false`.
    pub smooth_update_field: bool,
    /// Per-axis standard deviations for smoothing the update field, in pixel
    /// units. Default `[1.0, 1.0, 1.0]`. Only read when
    /// [`DemonsParams::smooth_update_field`] is set.
    pub update_field_standard_deviations: Vec<f64>,
    /// Truncation limit on the Gaussian kernel's half-width. Default `30`. The
    /// kept half-kernel reaches `maximum_kernel_width + 1` taps, because
    /// `GaussianOperator::GenerateCoefficients` tests the limit *after* pushing
    /// the offending coefficient; the kernel is normalised either way.
    pub maximum_kernel_width: u32,
    /// Allowed area error of the discrete Gaussian approximation, which sets
    /// the kernel width. Default `0.1`. `GaussianOperator::SetMaximumError`
    /// throws unless this lies strictly inside `(0, 1)`.
    pub maximum_error: f64,
    /// Below this absolute intensity difference a pixel is considered matched
    /// and its update is the zero vector. Default `0.001`.
    pub intensity_difference_threshold: f64,
    /// Default `true`, and **inert** — see the module docs.
    pub use_image_spacing: bool,
}

impl Default for DemonsParams {
    fn default() -> Self {
        DemonsParams {
            standard_deviations: vec![1.0; 3],
            number_of_iterations: 10,
            maximum_rms_error: 0.02,
            use_moving_image_gradient: false,
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

/// The displacement field together with the three measurements every filter in
/// the family reports.
#[derive(Clone, Debug, PartialEq)]
pub struct DemonsResult {
    /// `GetOutput()`: a `VectorFloat64` image with one component per dimension.
    pub displacement_field: Image,
    /// `GetElapsedIterations()`.
    pub elapsed_iterations: u32,
    /// `GetRMSChange()`. `0.0` when no iteration ran, mirroring
    /// `FiniteDifferenceImageFilter`'s `m_RMSChange{}`.
    pub rms_change: f64,
    /// `GetMetric()`: the mean square intensity difference over the pixels that
    /// mapped inside the moving image, during the last update computed.
    /// `f64::MAX` when no iteration ran or no pixel ever mapped inside,
    /// mirroring `m_Metric = NumericTraits<double>::max()`.
    pub metric: f64,
}

/// `DemonsRegistrationFunction`'s per-iteration state
/// (itkDemonsRegistrationFunction.hxx).
struct DemonsFunction<'a> {
    fixed: &'a RealImage,
    moving: &'a RealImage,
    use_moving_image_gradient: bool,
    intensity_difference_threshold: f64,
    /// `m_DenominatorThreshold`, hard-coded to `1e-9` in the constructor
    /// (itkDemonsRegistrationFunction.hxx:39) and exposed by no setter ITK's
    /// filter or SimpleITK's yaml surfaces.
    denominator_threshold: f64,
    /// `m_Normalizer`: the fixed image's mean square spacing.
    normalizer: f64,

    sum_of_squared_difference: f64,
    number_of_pixels_processed: u64,
    sum_of_squared_change: f64,
    metric: f64,
    rms_change: f64,
}

impl<'a> DemonsFunction<'a> {
    fn new(fixed: &'a RealImage, moving: &'a RealImage, params: &DemonsParams) -> Self {
        DemonsFunction {
            fixed,
            moving,
            use_moving_image_gradient: params.use_moving_image_gradient,
            intensity_difference_threshold: params.intensity_difference_threshold,
            denominator_threshold: 1e-9,
            normalizer: 1.0,
            sum_of_squared_difference: 0.0,
            number_of_pixels_processed: 0,
            sum_of_squared_change: 0.0,
            metric: f64::MAX,
            rms_change: f64::MAX,
        }
    }

    /// `DemonsRegistrationFunction::InitializeIteration`
    /// (itkDemonsRegistrationFunction.hxx:108-140).
    fn initialize_iteration(&mut self) {
        let spacing = self.fixed.spacing();
        self.normalizer = spacing.iter().map(|s| s * s).sum::<f64>() / spacing.len() as f64;
        self.sum_of_squared_difference = 0.0;
        self.number_of_pixels_processed = 0;
        self.sum_of_squared_change = 0.0;
    }

    /// `DemonsRegistrationFunction::ComputeUpdate`
    /// (itkDemonsRegistrationFunction.hxx:142-231). The neighbourhood radius is
    /// zero, so only the field's centre pixel is read.
    fn compute_update(&mut self, index: &[usize], displacement: &[f64], update: &mut [f64]) {
        let dim = self.fixed.dimension();
        let fixed_value = self.fixed.at(index);

        let mut mapped_point = self.fixed.index_to_physical_point(index);
        for j in 0..dim {
            mapped_point[j] += displacement[j];
        }

        if !self.moving.is_inside_buffer(&mapped_point) {
            update.fill(0.0);
            return;
        }
        let moving_value = self.moving.linear_interpolate(&mapped_point);

        let gradient = if self.use_moving_image_gradient {
            self.moving.central_difference_at_point(&mapped_point)
        } else {
            self.fixed.central_difference_at_index(index)
        };

        let gradient_squared_magnitude: f64 = gradient.iter().map(|g| g * g).sum();

        let speed_value = fixed_value - moving_value;
        let sqr_speed_value = speed_value * speed_value;

        // The metric is accumulated before the thresholds are tested, so a
        // thresholded pixel still counts toward the denominator.
        self.sum_of_squared_difference += sqr_speed_value;
        self.number_of_pixels_processed += 1;

        let denominator = sqr_speed_value / self.normalizer + gradient_squared_magnitude;

        if speed_value.abs() < self.intensity_difference_threshold
            || denominator < self.denominator_threshold
        {
            update.fill(0.0);
            return;
        }

        for j in 0..dim {
            update[j] = speed_value * gradient[j] / denominator;
            self.sum_of_squared_change += update[j] * update[j];
        }
    }

    /// `DemonsRegistrationFunction::ReleaseGlobalDataPointer`
    /// (itkDemonsRegistrationFunction.hxx:233-248), which is where the metric
    /// and the RMS change are actually formed — once per `CalculateChange`.
    /// Both are left at their previous values when no pixel was processed.
    fn finish_iteration(&mut self) {
        if self.number_of_pixels_processed != 0 {
            let n = self.number_of_pixels_processed as f64;
            self.metric = self.sum_of_squared_difference / n;
            self.rms_change = (self.sum_of_squared_change / n).sqrt();
        }
    }
}

/// `DemonsRegistrationFilter`: register `moving` onto `fixed` by Thirion's
/// demons, returning the displacement field.
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
pub fn demons_registration(
    fixed: &Image,
    moving: &Image,
    initial_displacement_field: Option<&Image>,
    params: &DemonsParams,
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

    let mut function = DemonsFunction::new(&fixed_real, &moving_real, params);
    let mut update = Field::zeros(fixed.size());
    let pixels = field.number_of_pixels();
    let mut index = vec![0usize; dim];

    let mut elapsed_iterations = 0u32;
    // `FiniteDifferenceImageFilter::m_RMSChange{}`.
    let mut rms_change = 0.0f64;

    // `FiniteDifferenceImageFilter::GenerateData`'s `while (!this->Halt())`.
    while !halt(
        elapsed_iterations,
        rms_change,
        params.number_of_iterations,
        params.maximum_rms_error,
    ) {
        // `PDEDeformableRegistrationFilter::InitializeIteration` →
        // `FiniteDifferenceImageFilter::InitializeIteration` → the function's.
        function.initialize_iteration();
        // `DemonsRegistrationFilter::InitializeIteration` smooths the field it
        // is about to differentiate.
        if params.smooth_displacement_field {
            smooth_field(&mut field, &displacement_smoothing);
        }

        // `DenseFiniteDifferenceImageFilter::CalculateChange`.
        for pixel in 0..pixels {
            field.multi_index(pixel, &mut index);
            function.compute_update(
                &index,
                field.vector_at(pixel),
                &mut update.data[pixel * dim..pixel * dim + dim],
            );
        }
        function.finish_iteration();
        // `DemonsRegistrationFunction::ComputeGlobalTimeStep` returns `m_TimeStep`,
        // constructed as `1.0` and never written again.
        let dt = 1.0;

        // `DemonsRegistrationFilter::ApplyUpdate`.
        if params.smooth_update_field {
            smooth_field(&mut update, &update_smoothing);
        }
        // `DenseFiniteDifferenceImageFilter::ThreadedApplyUpdate`.
        for (f, u) in field.data.iter_mut().zip(update.data.iter()) {
            *f += u * dt;
        }
        rms_change = function.rms_change;

        elapsed_iterations += 1;
    }

    Ok(DemonsResult {
        displacement_field: output_field(fixed, initial_displacement_field, field)?,
        elapsed_iterations,
        rms_change,
        metric: function.metric,
    })
}

#[cfg(test)]
mod tests;
