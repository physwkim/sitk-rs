//! `LevelSetMotionRegistrationFilter`
//! (itkLevelSetMotionRegistrationFilter.h/.hxx) and its
//! `LevelSetMotionRegistrationFunction`.
//!
//! This is not a demons force. The moving image is Gaussian-smoothed once per
//! iteration, its gradient is taken with a **minmod** limiter on one-sided
//! differences, and the update is a level-set motion term:
//!
//! ```text
//! g            = minmod(forward, backward)  on the smoothed moving image
//! update       = speed * g / (|g| + alpha)
//! ```
//!
//! `alpha` keeps the update bounded where the gradient vanishes: `|update| <=
//! |speed|` always, and `-> speed * sign(g)` as `alpha -> 0`.
//!
//! # The time step is not `1`
//!
//! Every other filter in this family has `ComputeGlobalTimeStep() == 1.0`. This
//! one returns `1 / max_L1`, where `max_L1` is the largest
//! `Σ_d |update[d]| / spacing[d]` over the image
//! (itkLevelSetMotionRegistrationFunction.hxx:352-364). The applied change is
//! `update * dt`, so **the largest displacement any pixel receives in one
//! iteration is exactly one pixel in L1** — that is the whole point of the
//! normalisation, and it makes the applied field independent of `alpha`
//! whenever a single gradient direction dominates. Pinned by
//! `the_largest_applied_displacement_is_exactly_one_pixel_in_l1`.
//!
//! When no pixel produces an update, `m_MaxL1Norm` keeps its initial
//! `NumericTraits<double>::NonpositiveMin()` (the .h's line 150), which is
//! negative, so `dt` falls back to `1.0`.
//!
//! # Upstream behaviour reproduced here
//!
//! * **Both smoothers are off by default.** The constructor calls
//!   `SmoothDisplacementFieldOff()` and `SmoothUpdateFieldOff()` (hxx:31-33),
//!   where `PDEDeformableRegistrationFilter`'s base defaults leave the
//!   displacement one on. Pinned by `neither_smoother_is_on_by_default`.
//! * **`use_image_spacing` is *not* inert here**, unlike in every other filter
//!   of this family. It picks the physical offset the one-sided differences
//!   step by, and the divisor of the L1 norm that sets the time step
//!   (hxx:239-244, 331-336). Pinned by `use_image_spacing_changes_the_result`.
//! * **`Halt()` is extended** (hxx:78-89) with "stop once the RMS change is
//!   exactly zero", which fires even when `maximum_rms_error` is `0.0` and the
//!   base rule's strict `>` never can. Pinned by
//!   `a_zero_rms_change_halts_even_at_a_zero_maximum_rms_error`.
//! * **`GetRMSChange()` is *not* overridden**, so a zero-iteration run reports
//!   the filter's `0.0` — as `DemonsRegistrationFilter` does, and unlike the
//!   three ESM/symmetric filters. `GetMetric()` *is* forwarded to the function,
//!   so it reports `f64::MAX`. Pinned by
//!   `a_zero_iteration_run_reports_the_filters_rms_change_and_the_functions_metric`.
//! * **The RMS change is measured before the time step scales the update**
//!   (hxx:325-334 accumulates `m_SumOfSquaredChange` from the raw `update`).
//! * **The metric counts a thresholded pixel.** `m_SumOfSquaredDifference` and
//!   `m_NumberOfPixelsProcessed` are accumulated at hxx:311-316, *before* the
//!   intensity-difference and gradient-magnitude guards return a zero update.
//!   A pixel whose mapped point leaves the moving image's buffer is skipped
//!   before that, and counts for nothing.
//! * **The one-sided differences step along a physical axis by an index axis's
//!   spacing** (`mPoint[j] += mSpacing[j]`, hxx:252): the same borrowed-axis
//!   quirk `CentralDifferenceImageFunction::Evaluate` has. Under a non-identity
//!   direction matrix the offset is along physical axis `j` but its length is
//!   index axis `j`'s spacing.
//! * A one-sided difference whose sample point leaves the buffer is `0.0`, and
//!   `minmod` then returns `0` for that axis because `forward * backward` is not
//!   strictly positive. The first and last slice along every axis therefore have
//!   a zero derivative along it. Pinned by
//!   `a_border_pixel_has_no_derivative_along_the_border_axis`.
//!
//! # The smoothed moving image
//!
//! `m_MovingImageSmoothingFilter` is a
//! `SmoothingRecursiveGaussianImageFilter<MovingImageType>`
//! (itkLevelSetMotionRegistrationFunction.h:116) with `NormalizeAcrossScale`
//! off, so upstream its **output pixel type is the moving image's** — smoothing
//! a `u8` moving image quantised the smoothed one back to `u8` before its
//! gradient was taken. **Fixed here (§2.49):** the recursive Gaussian is taken
//! in full `f64` precision and held in a `Float64` image, so the gradient sees
//! the true smoothed values rather than an integer-rounded copy. The smoother
//! still inherits `RecursiveSeparableImageFilter`'s short-axis requirement — an
//! axis of fewer than four pixels is [`FilterError::AxisTooShortForRecursion`],
//! as `RecursiveSeparableImageFilter` throws.
//!
//! ITK filters the last index axis first and the rest in order
//! (itkSmoothingRecursiveGaussianImageFilter.hxx:38-48), while
//! `recursive_gaussian` filters axis `0` first. Separable linear filters
//! commute, so the two agree up to floating-point association.

use crate::core::{Image, PixelId};

use super::DemonsResult;
use super::common::{halt, initial_field, output_field, per_axis, validate_verifying_image_pair};
use super::field::{Field, Smoothing, smooth_field};
use super::image_function::RealImage;
use crate::filters::Result;
use crate::filters::image_from_f64;
use crate::filters::recursive_gaussian::{GaussianOrder, recursive_gaussian_f64};

/// The yaml's parameter surface (`LevelSetMotionRegistrationFilter.yaml`).
/// [`Default`] carries the yaml defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelSetMotionParams {
    /// Standard deviation, in **physical units**, of the Gaussian that smooths
    /// the moving image before its gradient is taken. The same value on every
    /// axis. Default `1.0`.
    pub gradient_smoothing_standard_deviations: f64,
    /// Iteration cap. Default `10`.
    pub number_of_iterations: u32,
    /// Stop once an iteration's RMS change is strictly below this. Default
    /// `0.02`. An RMS change of exactly `0.0` stops the filter regardless.
    pub maximum_rms_error: f64,
    /// Per-axis Gaussian standard deviations for smoothing the displacement
    /// field, in **pixel** units. Default `[1.0, 1.0, 1.0]`. Only read when
    /// [`LevelSetMotionParams::smooth_displacement_field`] is set.
    pub standard_deviations: Vec<f64>,
    /// Gaussian-smooth the displacement field at the top of each iteration.
    /// Default **`false`** — the filter's constructor turns the base's default
    /// off, because "no regularization of the deformation field is performed in
    /// LevelSetMotionRegistration".
    pub smooth_displacement_field: bool,
    /// Gaussian-smooth the update field before applying it. Default `false`.
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
    /// Bounds the update where the gradient is small: the denominator is
    /// `|gradient| + alpha`. Default `0.1`.
    pub alpha: f64,
    /// Below this absolute intensity difference a pixel's update is the zero
    /// vector — though it still counts toward the metric. Default `0.001`.
    pub intensity_difference_threshold: f64,
    /// Below this gradient magnitude a pixel's update is the zero vector.
    /// Default `1e-9`.
    pub gradient_magnitude_threshold: f64,
    /// Default `true`. Unlike the rest of this family, it is **live**: it
    /// selects the physical step of the one-sided differences and the divisor
    /// of the time step's L1 norm.
    pub use_image_spacing: bool,
}

impl Default for LevelSetMotionParams {
    fn default() -> Self {
        LevelSetMotionParams {
            gradient_smoothing_standard_deviations: 1.0,
            number_of_iterations: 10,
            maximum_rms_error: 0.02,
            standard_deviations: vec![1.0; 3],
            smooth_displacement_field: false,
            smooth_update_field: false,
            update_field_standard_deviations: vec![1.0; 3],
            maximum_kernel_width: 30,
            maximum_error: 0.1,
            alpha: 0.1,
            intensity_difference_threshold: 0.001,
            gradient_magnitude_threshold: 1e-9,
            use_image_spacing: true,
        }
    }
}

/// `m(x, y) = sign(x) · min(|x|, |y|)` if `xy > 0`, else `0`
/// (itkLevelSetMotionRegistrationFunction.hxx:281-303).
fn minmod(forward: f64, backward: f64) -> f64 {
    if forward * backward > 0.0 {
        forward.abs().min(backward.abs()) * forward.signum()
    } else {
        0.0
    }
}

/// `LevelSetMotionRegistrationFunction`'s per-iteration state.
struct LevelSetMotionFunction<'a> {
    fixed: &'a RealImage,
    moving: &'a RealImage,
    /// The moving image after `SmoothingRecursiveGaussianImageFilter`, widened
    /// back to `f64` — so a quantising cast to an integer moving pixel type is
    /// already baked in.
    smooth_moving: &'a RealImage,
    /// `mSpacing`: the *moving* image's spacing, or all ones when
    /// `UseImageSpacing` is off.
    step: Vec<f64>,
    alpha: f64,
    gradient_magnitude_threshold: f64,
    intensity_difference_threshold: f64,

    sum_of_squared_difference: f64,
    number_of_pixels_processed: u64,
    sum_of_squared_change: f64,
    /// `GlobalDataStruct::m_MaxL1Norm`, initialised to
    /// `NumericTraits<double>::NonpositiveMin()`.
    max_l1_norm: f64,
    pub(crate) metric: f64,
    pub(crate) rms_change: f64,
}

impl<'a> LevelSetMotionFunction<'a> {
    fn new(
        fixed: &'a RealImage,
        moving: &'a RealImage,
        smooth_moving: &'a RealImage,
        params: &LevelSetMotionParams,
    ) -> Self {
        let step = if params.use_image_spacing {
            moving.spacing().to_vec()
        } else {
            vec![1.0; moving.dimension()]
        };
        LevelSetMotionFunction {
            fixed,
            moving,
            smooth_moving,
            step,
            alpha: params.alpha,
            gradient_magnitude_threshold: params.gradient_magnitude_threshold,
            intensity_difference_threshold: params.intensity_difference_threshold,
            sum_of_squared_difference: 0.0,
            number_of_pixels_processed: 0,
            sum_of_squared_change: 0.0,
            max_l1_norm: f64::MIN,
            metric: f64::MAX,
            rms_change: f64::MAX,
        }
    }

    /// `InitializeIteration` (hxx:172-194). The moving image is smoothed by the
    /// caller, once — ITK's pipeline caches it the same way.
    fn initialize_iteration(&mut self) {
        self.sum_of_squared_difference = 0.0;
        self.number_of_pixels_processed = 0;
        self.sum_of_squared_change = 0.0;
        self.max_l1_norm = f64::MIN;
    }

    /// The minmod gradient of the smoothed moving image at `point`
    /// (hxx:246-303).
    fn minmod_gradient(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.fixed.dimension();
        let central = self.smooth_moving.linear_interpolate(point);
        let mut sample = point.to_vec();
        let mut gradient = vec![0.0f64; dim];

        for (d, derivative) in gradient.iter_mut().enumerate() {
            let step = self.step[d];

            sample[d] = point[d] + step;
            let forward = if self.smooth_moving.is_inside_buffer(&sample) {
                (self.smooth_moving.linear_interpolate(&sample) - central) / step
            } else {
                0.0
            };

            sample[d] = point[d] - step;
            let backward = if self.smooth_moving.is_inside_buffer(&sample) {
                (central - self.smooth_moving.linear_interpolate(&sample)) / step
            } else {
                0.0
            };
            sample[d] = point[d];

            *derivative = minmod(forward, backward);
        }

        gradient
    }

    /// `LevelSetMotionRegistrationFunction::ComputeUpdate` (hxx:196-347).
    fn compute_update(&mut self, index: &[usize], displacement: &[f64], update: &mut [f64]) {
        update.fill(0.0);

        let fixed_value = self.fixed.at(index);
        let mut mapped_point = self.fixed.index_to_physical_point(index);
        for (coordinate, &offset) in mapped_point.iter_mut().zip(displacement) {
            *coordinate += offset;
        }

        // A pixel that maps outside is skipped before the metric sees it.
        if !self.moving.is_inside_buffer(&mapped_point) {
            return;
        }
        let moving_value = self.moving.linear_interpolate(&mapped_point);

        let gradient = self.minmod_gradient(&mapped_point);
        let gradient_magnitude = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        let speed_value = fixed_value - moving_value;
        self.sum_of_squared_difference += speed_value * speed_value;
        self.number_of_pixels_processed += 1;

        if speed_value.abs() < self.intensity_difference_threshold
            || gradient_magnitude < self.gradient_magnitude_threshold
        {
            return;
        }

        let mut l1_norm = 0.0f64;
        for ((component, &g), &step) in update.iter_mut().zip(&gradient).zip(&self.step) {
            *component = speed_value * g / (gradient_magnitude + self.alpha);
            self.sum_of_squared_change += *component * *component;
            l1_norm += component.abs() / step;
        }

        if l1_norm > self.max_l1_norm {
            self.max_l1_norm = l1_norm;
        }
    }

    /// `ComputeGlobalTimeStep` (hxx:349-364). `m_MaxL1Norm` is negative until a
    /// pixel produces an update, so a wholly-thresholded iteration gets `1.0`.
    fn global_time_step(&self) -> f64 {
        if self.max_l1_norm > 0.0 {
            1.0 / self.max_l1_norm
        } else {
            1.0
        }
    }

    /// `ReleaseGlobalDataPointer` (hxx:366-384).
    fn finish_iteration(&mut self) {
        if self.number_of_pixels_processed != 0 {
            let n = self.number_of_pixels_processed as f64;
            self.metric = self.sum_of_squared_difference / n;
            self.rms_change = (self.sum_of_squared_change / n).sqrt();
        }
    }
}

/// `LevelSetMotionRegistrationFilter::Halt` (hxx:76-89): the base rule, plus
/// "an RMS change of exactly zero stops the filter".
fn level_set_motion_halt(
    elapsed: u32,
    rms_change: f64,
    number_of_iterations: u32,
    maximum_rms_error: f64,
) -> bool {
    halt(elapsed, rms_change, number_of_iterations, maximum_rms_error)
        || (rms_change == 0.0 && elapsed != 0)
}

/// `LevelSetMotionRegistrationFilter`: register `moving` onto `fixed` by level
/// set motion, returning the displacement field.
///
/// `fixed` and `moving` must be scalar images of the same pixel type and
/// dimension. `initial_displacement_field`, when given, must be a
/// `VectorFloat64` image with one component per dimension and the same size as
/// `fixed`; otherwise the field starts at zero.
///
/// Errors on a vector `fixed`/`moving`, on mismatched pixel types or
/// dimensions, on a `standard_deviations` or `update_field_standard_deviations`
/// shorter than the image dimension, on a `maximum_error` outside `(0, 1)`, on
/// an initial field of the wrong pixel type, component count, or size, on a
/// negative `gradient_smoothing_standard_deviations`, and — through the
/// recursive Gaussian that smooths the moving image — on any moving-image axis
/// shorter than four pixels.
pub fn level_set_motion_registration(
    fixed: &Image,
    moving: &Image,
    initial_displacement_field: Option<&Image>,
    params: &LevelSetMotionParams,
) -> Result<DemonsResult> {
    validate_verifying_image_pair(fixed, moving, params.maximum_error)?;

    let dim = fixed.dimension();
    let fixed_real = RealImage::new(fixed)?;
    let moving_real = RealImage::new(moving)?;

    // `m_MovingImageSmoothingFilter`. ITK's pipeline recomputes it once, on the
    // first `InitializeIteration`, and caches it thereafter; neither the moving
    // image nor sigma changes between iterations, so hoisting it out of the loop
    // is the same computation.
    //
    // §2.49 fix: smooth in full `f64` precision, not the moving image's own
    // pixel type. ITK types the smoother as
    // `SmoothingRecursiveGaussianImageFilter<MovingImageType>`, whose output
    // pixel type is the moving image's, so an integer moving image had its
    // smoothed copy quantised back to that integer type *before* the gradient
    // was taken — silently corrupting the gradient the level-set motion rides.
    // Taking the recursive Gaussian in `f64` and holding the result in a
    // `Float64` image feeds the true smoothed values to the gradient.
    let smoothed_values = recursive_gaussian_f64(
        moving,
        &vec![params.gradient_smoothing_standard_deviations; dim],
        &vec![GaussianOrder::ZeroOrder; dim],
        false,
    )?;
    let smoothed = image_from_f64(PixelId::Float64, moving.size(), moving, &smoothed_values)?;
    let smooth_moving_real = RealImage::new(&smoothed)?;

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
    let mut function =
        LevelSetMotionFunction::new(&fixed_real, &moving_real, &smooth_moving_real, params);

    let pixels = field.number_of_pixels();
    let mut index = vec![0usize; dim];

    let mut elapsed_iterations = 0u32;
    // `FiniteDifferenceImageFilter::m_RMSChange{}`, which `Halt` reads and
    // which the un-overridden `GetRMSChange()` returns.
    let mut rms_change = 0.0f64;

    while !level_set_motion_halt(
        elapsed_iterations,
        rms_change,
        params.number_of_iterations,
        params.maximum_rms_error,
    ) {
        function.initialize_iteration();
        // `LevelSetMotionRegistrationFilter::InitializeIteration` smooths the
        // field it is about to differentiate.
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
        let dt = function.global_time_step();
        function.finish_iteration();

        // `ApplyUpdate`.
        if params.smooth_update_field {
            smooth_field(&mut update, &update_smoothing);
        }
        for (component, &change) in field.data.iter_mut().zip(&update.data) {
            *component += change * dt;
        }
        rms_change = function.rms_change;

        elapsed_iterations += 1;
    }

    Ok(DemonsResult {
        displacement_field: output_field(fixed, initial_displacement_field, field)?,
        elapsed_iterations,
        // `GetRMSChange()` is not overridden: this is the *filter's* value.
        rms_change,
        // `GetMetric()` is forwarded to the function.
        metric: function.metric,
    })
}

#[cfg(test)]
mod tests;
