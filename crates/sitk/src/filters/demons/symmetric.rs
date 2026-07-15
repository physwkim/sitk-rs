//! `SymmetricForcesDemonsRegistrationFilter`
//! (itkSymmetricForcesDemonsRegistrationFilter.h/.hxx) and its
//! `SymmetricForcesDemonsRegistrationFunction`.
//!
//! The force is `2·speed·(∇f + ∇(m∘φ)) / (speed²/K + |∇f + ∇(m∘φ)|²)` with
//! `K = mean(spacing²)` — the same shape as [`super::demons_registration`]'s,
//! doubled, and with the moving image's gradient added to the fixed image's.
//!
//! Unlike `FastSymmetricForcesDemonsRegistrationFilter`, no warped image is
//! materialised: `∇(m∘φ)` is a central difference of the moving image sampled at
//! the *neighbouring pixels' own mapped points*
//! (itkSymmetricForcesDemonsRegistrationFunction.hxx:144-193). The solver loop
//! is `DemonsRegistrationFilter`'s — the field is smoothed at the top of the
//! iteration, before the update is computed from it.
//!
//! # Upstream behaviour reproduced here
//!
//! * **A centre that maps outside the moving image is not skipped.** It takes
//!   `movingValue = 0` (line 195) and goes on to produce an update and a metric
//!   contribution, where `DemonsRegistrationFunction` and
//!   `ESMDemonsRegistrationFunction` both return early. Pinned by
//!   `a_centre_mapped_outside_takes_a_zero_moving_value_rather_than_being_skipped`.
//! * **The metric drops a two-pixel border on every axis** ("there are often
//!   artifacts which falsify the metric", lines 246-259), so an image with any
//!   axis shorter than five pixels has *no* metric pixels at all and reports
//!   `f64::MAX`. Pinned by `an_image_thinner_than_five_pixels_has_no_metric_pixels`.
//! * **The RMS change divides two different populations.**
//!   `m_SumOfSquaredChange` accumulates over *every* pixel (line 244), while
//!   `m_NumberOfPixelsProcessed` counts only the non-border ones, and
//!   `ReleaseGlobalDataPointer` divides the first by the second. Pinned by
//!   `the_rms_change_divides_the_whole_image_sum_by_the_interior_pixel_count`.
//! * **The metric is evaluated at the post-update point** `mappedCenterPoint +
//!   update` (lines 245, 261-267), not at the point the update was computed
//!   from. Pinned by `the_metric_samples_the_moving_image_after_the_update`.
//! * **A backward neighbour that maps outside is silently not subtracted**
//!   (lines 185-188), leaving `movingGradient[dim]` as half the forward sample
//!   rather than a one-sided difference. Pinned by
//!   `an_out_of_buffer_backward_neighbour_is_not_subtracted`.
//! * `GetRMSChange()` is overridden to return the difference function's
//!   `m_RMSChange`, `NumericTraits<double>::max()` until an iteration counts a
//!   pixel — as in `FastSymmetricForcesDemonsRegistrationFilter`, and unlike
//!   `DemonsRegistrationFilter`.
//!
//! # An upstream inconsistency, reproduced
//!
//! `m_FixedImageGradientCalculator` keeps `UseImageDirection` at its default
//! `true` (the constructor, lines 28-47, does not turn it off), so `∇f` comes
//! back rotated into physical space — while `∇(m∘φ)` is built by hand in index
//! space and divided by the fixed image's spacing. The two are added
//! component-wise (line 218). Under a non-identity direction matrix they are
//! vectors in different frames. `ESMDemonsRegistrationFunction` avoids this by
//! turning the calculator's direction off and rotating the sum once, at the end.

use crate::core::Image;

use super::DemonsResult;
use super::common::{halt, initial_field, output_field, per_axis, validate_verifying_image_pair};
use super::field::{Field, Smoothing, smooth_field};
use super::image_function::RealImage;
use crate::filters::Result;

/// The yaml's parameter surface (`SymmetricForcesDemonsRegistrationFilter.yaml`).
/// [`Default`] carries the yaml defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetricForcesDemonsParams {
    /// Per-axis Gaussian standard deviations for smoothing the displacement
    /// field, in **pixel** units. Default `[1.0, 1.0, 1.0]`.
    pub standard_deviations: Vec<f64>,
    /// Iteration cap. Default `10`.
    pub number_of_iterations: u32,
    /// Stop once an iteration's RMS change is strictly below this. Default
    /// `0.02`.
    pub maximum_rms_error: f64,
    /// Gaussian-smooth the displacement field at the top of each iteration,
    /// before the update is computed from it. Default `true`.
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
    /// Below this absolute intensity difference a pixel's update is the zero
    /// vector — though it still contributes to the metric. Default `0.001`.
    pub intensity_difference_threshold: f64,
    /// Default `true`, and **inert** — see [`super`]'s notes on
    /// `m_ScaleCoefficients`.
    pub use_image_spacing: bool,
}

impl Default for SymmetricForcesDemonsParams {
    fn default() -> Self {
        SymmetricForcesDemonsParams {
            standard_deviations: vec![1.0; 3],
            number_of_iterations: 10,
            maximum_rms_error: 0.02,
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

/// `SymmetricForcesDemonsRegistrationFunction`'s per-iteration state.
struct SymmetricForcesFunction<'a> {
    fixed: &'a RealImage,
    moving: &'a RealImage,
    intensity_difference_threshold: f64,
    /// `m_DenominatorThreshold`, hard-coded to `1e-9` (line 33).
    denominator_threshold: f64,
    /// `m_Normalizer`: the fixed image's mean square spacing.
    normalizer: f64,

    sum_of_squared_difference: f64,
    number_of_pixels_processed: u64,
    sum_of_squared_change: f64,
    metric: f64,
    rms_change: f64,
}

impl<'a> SymmetricForcesFunction<'a> {
    fn new(
        fixed: &'a RealImage,
        moving: &'a RealImage,
        intensity_difference_threshold: f64,
    ) -> Self {
        SymmetricForcesFunction {
            fixed,
            moving,
            intensity_difference_threshold,
            denominator_threshold: 1e-9,
            normalizer: 1.0,
            sum_of_squared_difference: 0.0,
            number_of_pixels_processed: 0,
            sum_of_squared_change: 0.0,
            metric: f64::MAX,
            rms_change: f64::MAX,
        }
    }

    /// `SymmetricForcesDemonsRegistrationFunction::InitializeIteration`
    /// (lines 92-122).
    fn initialize_iteration(&mut self) {
        let spacing = self.fixed.spacing();
        self.normalizer = spacing.iter().map(|s| s * s).sum::<f64>() / spacing.len() as f64;
        self.sum_of_squared_difference = 0.0;
        self.number_of_pixels_processed = 0;
        self.sum_of_squared_change = 0.0;
    }

    /// The point the pixel at `index` maps to: its own physical position plus
    /// the displacement the field carries there.
    fn mapped_point(&self, field: &Field, index: &[usize]) -> Vec<f64> {
        let mut point = self.fixed.index_to_physical_point(index);
        for (coordinate, &displacement) in
            point.iter_mut().zip(field.vector_at(field.offset(index)))
        {
            *coordinate += displacement;
        }
        point
    }

    /// `∇(m∘φ)` (lines 144-193): a central difference of the moving image
    /// sampled at the two axial neighbours' *own* mapped points, divided by the
    /// fixed image's spacing.
    ///
    /// A forward neighbour that maps outside the moving image contributes `0`; a
    /// backward one that maps outside is simply not subtracted, so the result is
    /// then half the forward sample rather than a one-sided difference.
    fn moving_gradient(&self, field: &Field, index: &[usize]) -> Vec<f64> {
        let dim = self.fixed.dimension();
        let mut gradient = vec![0.0f64; dim];
        let mut neighbor = index.to_vec();

        for (d, derivative) in gradient.iter_mut().enumerate() {
            // `index[dim] < FirstIndex[dim] + 1 || index[dim] > LastIndex[dim] - 2`.
            let last = self.fixed.size()[d] as i64 - 2;
            let here = index[d];
            if (here as i64) < 1 || here as i64 > last {
                continue;
            }

            neighbor[d] = here + 1;
            let forward = self.mapped_point(field, &neighbor);
            let mut value = if self.moving.is_inside_buffer(&forward) {
                self.moving.linear_interpolate(&forward)
            } else {
                0.0
            };

            neighbor[d] = here - 1;
            let backward = self.mapped_point(field, &neighbor);
            if self.moving.is_inside_buffer(&backward) {
                value -= self.moving.linear_interpolate(&backward);
            }
            neighbor[d] = here;

            *derivative = value * 0.5 / self.fixed.spacing()[d];
        }

        gradient
    }

    /// `SymmetricForcesDemonsRegistrationFunction::ComputeUpdate`
    /// (lines 124-271).
    fn compute_update(&mut self, field: &Field, index: &[usize], update: &mut [f64]) {
        let fixed_value = self.fixed.at(index);
        // `UseImageDirection` is left on, so this is the *physical* gradient.
        let fixed_gradient = self.fixed.central_difference_at_index(index);
        let moving_gradient = self.moving_gradient(field, index);

        let mapped_center = self.mapped_point(field, index);
        let moving_value = if self.moving.is_inside_buffer(&mapped_center) {
            self.moving.linear_interpolate(&mapped_center)
        } else {
            0.0
        };

        let combined: Vec<f64> = fixed_gradient
            .iter()
            .zip(&moving_gradient)
            .map(|(f, m)| f + m)
            .collect();
        let squared_magnitude: f64 = combined.iter().map(|g| g * g).sum();

        let speed_value = fixed_value - moving_value;
        let denominator = speed_value * speed_value / self.normalizer + squared_magnitude;

        if speed_value.abs() < self.intensity_difference_threshold
            || denominator < self.denominator_threshold
        {
            update.fill(0.0);
        } else {
            for (component, &gradient) in update.iter_mut().zip(&combined) {
                *component = 2.0 * speed_value * gradient / denominator;
            }
        }

        // The squared change is summed over every pixel, but the metric skips a
        // two-pixel border; `ReleaseGlobalDataPointer` divides the one by the
        // other's count.
        let mut new_point = mapped_center;
        let mut outside_region = false;
        for (j, &change) in update.iter().enumerate() {
            self.sum_of_squared_change += change * change;
            new_point[j] += change;
            if (index[j] as i64) < 2 || index[j] as i64 > self.fixed.size()[j] as i64 - 3 {
                outside_region = true;
            }
        }

        if !outside_region {
            let new_moving_value = if self.moving.is_inside_buffer(&new_point) {
                self.moving.linear_interpolate(&new_point)
            } else {
                0.0
            };
            let residual = fixed_value - new_moving_value;
            self.sum_of_squared_difference += residual * residual;
            self.number_of_pixels_processed += 1;
        }
    }

    /// `SymmetricForcesDemonsRegistrationFunction::ReleaseGlobalDataPointer`
    /// (lines 273-289).
    fn finish_iteration(&mut self) {
        if self.number_of_pixels_processed != 0 {
            let n = self.number_of_pixels_processed as f64;
            self.metric = self.sum_of_squared_difference / n;
            self.rms_change = (self.sum_of_squared_change / n).sqrt();
        }
    }
}

/// `SymmetricForcesDemonsRegistrationFilter`: register `moving` onto `fixed`
/// with the symmetric demons force, returning the displacement field.
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
pub fn symmetric_forces_demons_registration(
    fixed: &Image,
    moving: &Image,
    initial_displacement_field: Option<&Image>,
    params: &SymmetricForcesDemonsParams,
) -> Result<DemonsResult> {
    validate_verifying_image_pair(fixed, moving, params.maximum_error)?;

    let dim = fixed.dimension();
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
    let mut function = SymmetricForcesFunction::new(
        &fixed_real,
        &moving_real,
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
        function.initialize_iteration();
        // `SymmetricForcesDemonsRegistrationFilter::InitializeIteration`
        // (itkSymmetricForcesDemonsRegistrationFilter.hxx:34-55) smooths the
        // field it is about to differentiate.
        if params.smooth_displacement_field {
            smooth_field(&mut field, &displacement_smoothing);
        }

        // `DenseFiniteDifferenceImageFilter::CalculateChange`.
        for pixel in 0..pixels {
            field.multi_index(pixel, &mut index);
            function.compute_update(
                &field,
                &index,
                &mut update.data[pixel * dim..pixel * dim + dim],
            );
        }
        function.finish_iteration();

        // `SymmetricForcesDemonsRegistrationFilter::ApplyUpdate` (lines 113-135).
        if params.smooth_update_field {
            smooth_field(&mut update, &update_smoothing);
        }
        // `DenseFiniteDifferenceImageFilter::ThreadedApplyUpdate`, with the
        // constant `dt == 1.0` of `ComputeGlobalTimeStep`.
        for (component, change) in field.data.iter_mut().zip(&update.data) {
            *component += change;
        }
        halt_rms_change = function.rms_change;

        elapsed_iterations += 1;
    }

    Ok(DemonsResult {
        displacement_field: output_field(fixed, initial_displacement_field, field)?,
        elapsed_iterations,
        // The overridden `GetRMSChange()`: the function's value, `f64::MAX`
        // until an iteration counts at least one non-border pixel.
        rms_change: function.rms_change,
        metric: function.metric,
    })
}

#[cfg(test)]
mod tests;
