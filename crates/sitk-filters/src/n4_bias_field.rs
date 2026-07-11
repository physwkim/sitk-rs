//! `N4BiasFieldCorrectionImageFilter`: Tustison's N4 retrospective correction
//! of the smooth multiplicative bias field in MR images, ported from
//! `itkN4BiasFieldCorrectionImageFilter.h(.hxx)`.
//!
//! The algorithm works in log space. Each iteration:
//!
//! 1. **Sharpen** the current log-uncorrected image by deconvolving its
//!    intensity histogram with a Gaussian of the given full width at half
//!    maximum, using a Wiener filter, and remapping every voxel through the
//!    resulting expectation `E(u | v)` ([`sharpen_image`]).
//! 2. Take the **residual** `log-uncorrected − log-sharpened`, fit a B-spline
//!    scalar field to it at every included voxel (confidence-weighted), and
//!    **add** the fitted control-point lattice to the running estimate
//!    ([`update_bias_field_estimate`]).
//! 3. Measure **convergence** as the coefficient of variation of
//!    `exp(previous − current)` over the included voxels
//!    ([`calculate_convergence_measurement`]).
//!
//! Between fitting levels the control-point lattice is refined dyadically, so
//! later levels model finer structure. The corrected output is
//! `input / exp(log bias field)`.
//!
//! **Divergences from upstream, all deliberate:**
//!
//! - ITK pins `RealType = float`; this computes in `f64`, the workspace's
//!   `to_f64_vec` idiom. Every formula is transcribed literally otherwise.
//! - ITK's `SharpenImage` indexes its histogram with `itk::Math::floor` —
//!   a `static_cast<int>` of a `float` that is `0.0/0.0 = NaN` when the
//!   bin range is degenerate (see [`sharpen_image`]); a NaN→int conversion
//!   is undefined behaviour in C++ and the resulting index is used
//!   unchecked. Here the index is bounds-checked and the voxel skipped.
//! - Degeneracies ITK dereferences a null lattice on, or divides by zero on,
//!   are rejected up front: see [`FilterError::N4NoBiasFieldEstimated`] and
//!   [`FilterError::N4InvalidHistogramBins`].

mod bspline;

use crate::error::{FilterError, Result};
use crate::fft::{Complex, transform_1d_unnormalized};
use crate::{image_from_f64, quantize_to_pixel_type};
use bspline::{FitInput, Lattice};
use sitk_core::{Image, PixelId};

/// Parameters of [`n4_bias_field_correction`], defaulted to SimpleITK's
/// `N4BiasFieldCorrectionImageFilter.yaml`.
#[derive(Clone, Debug, PartialEq)]
pub struct N4BiasFieldCorrectionSettings {
    /// Stop a fitting level once the coefficient of variation of the ratio
    /// between successive bias-field estimates drops below this. Default
    /// `0.001`.
    pub convergence_threshold: f64,

    /// One iteration cap per fitting level; its length *is* the number of
    /// fitting levels (SimpleITK's `custom_itk_cast` calls
    /// `SetNumberOfFittingLevels(m_MaximumNumberOfIterations.size())`).
    /// Default `[50, 50, 50, 50]`.
    pub maximum_number_of_iterations: Vec<u32>,

    /// Width of the Gaussian the histogram is deconvolved with, in log
    /// intensity units. Default `0.15`.
    pub bias_field_full_width_at_half_maximum: f64,

    /// Noise estimate defining the Wiener filter; the `Z` of Sled 1998 is its
    /// square root. Default `0.01`.
    pub wiener_filter_noise: f64,

    /// Bins in the log-intensity histogram. Default `200`.
    pub number_of_histogram_bins: u32,

    /// Control points of the initial B-spline lattice, per axis. The mesh size
    /// along an axis is this minus the spline order. Entries beyond the image's
    /// dimension are ignored, matching SimpleITK's `dim_vec` member. Default
    /// `[4, 4, 4]`.
    pub number_of_control_points: Vec<u32>,

    /// Degree of the B-spline modelling the log bias field. Default `3`.
    pub spline_order: u32,

    /// When true, only mask voxels equal to `mask_label` are used; when false,
    /// every non-zero mask voxel is. Default `true`.
    pub use_mask_label: bool,

    /// The mask value selecting included voxels when `use_mask_label`.
    /// Default `1`.
    pub mask_label: u8,
}

impl Default for N4BiasFieldCorrectionSettings {
    fn default() -> Self {
        Self {
            convergence_threshold: 0.001,
            maximum_number_of_iterations: vec![50; 4],
            bias_field_full_width_at_half_maximum: 0.15,
            wiener_filter_noise: 0.01,
            number_of_histogram_bins: 200,
            number_of_control_points: vec![4; 3],
            spline_order: 3,
            use_mask_label: true,
            mask_label: 1,
        }
    }
}

/// The corrected image together with the log bias field that produced it.
#[derive(Clone, Debug)]
pub struct N4BiasFieldCorrectionResult {
    /// `input / exp(log_bias_field)`, in the input's pixel type.
    pub corrected: Image,

    /// The log bias field sampled on the input grid, in the input's pixel type.
    ///
    /// This is SimpleITK's `GetLogBiasFieldAsImage(referenceImage)` with the
    /// input as reference: the final control-point lattice reconstructed onto
    /// that grid. `corrected` is built from the same field, so
    /// `corrected == input / exp(log_bias_field)` voxelwise.
    pub log_bias_field: Image,
}

/// `N4BiasFieldCorrectionImageFilter`: estimate and divide out the smooth
/// multiplicative bias field of `image`, returning the corrected image.
///
/// `mask_image`, if given, restricts bias-field estimation to the voxels
/// selected by `settings.use_mask_label`/`settings.mask_label`. It is cast to
/// `uint8` first, as SimpleITK does (`MaskImageType` is
/// `itk::Image<uint8_t, Dimension>`).
///
/// `confidence_image`, if given, weights each voxel in the B-spline fit and
/// excludes voxels whose confidence is not strictly positive. SimpleITK does
/// not surface this input; ITK does, and the fit's `SetPointWeights` path
/// exists only for it.
///
/// The input must have a floating-point pixel type
/// (`pixel_types: RealPixelIDTypeList`). Voxels that are not strictly positive
/// never enter the log transform, so — as the upstream documentation warns —
/// inputs with negative or sub-unit values correct poorly.
pub fn n4_bias_field_correction(
    image: &Image,
    mask_image: Option<&Image>,
    confidence_image: Option<&Image>,
    settings: &N4BiasFieldCorrectionSettings,
) -> Result<Image> {
    Ok(
        n4_bias_field_correction_with_log_bias_field(
            image,
            mask_image,
            confidence_image,
            settings,
        )?
        .corrected,
    )
}

/// [`n4_bias_field_correction`], additionally returning the log bias field.
pub fn n4_bias_field_correction_with_log_bias_field(
    image: &Image,
    mask_image: Option<&Image>,
    confidence_image: Option<&Image>,
    settings: &N4BiasFieldCorrectionSettings,
) -> Result<N4BiasFieldCorrectionResult> {
    let filter = N4::new(image, mask_image, confidence_image, settings)?;
    filter.run()
}

/// The filter's per-run state: everything `GenerateData` keeps in members plus
/// the masked-voxel predicate its four loops share.
struct N4<'a> {
    image: &'a Image,
    settings: &'a N4BiasFieldCorrectionSettings,
    /// Voxelwise `mask ∧ confidence` predicate, evaluated once. ITK re-tests
    /// the same expression in `GenerateData`, `SharpenImage`,
    /// `UpdateBiasFieldEstimate` and `CalculateConvergenceMeasurement`.
    included: Vec<bool>,
    /// Per-voxel B-spline fitting weight: the confidence image's value, or
    /// `1.0` when no confidence image was supplied.
    confidence_weights: Vec<f64>,
    spline_order: usize,
    number_of_control_points: Vec<usize>,
    number_of_fitting_levels: usize,
}

impl<'a> N4<'a> {
    fn new(
        image: &'a Image,
        mask_image: Option<&Image>,
        confidence_image: Option<&Image>,
        settings: &'a N4BiasFieldCorrectionSettings,
    ) -> Result<Self> {
        if !matches!(image.pixel_id(), PixelId::Float32 | PixelId::Float64) {
            return Err(FilterError::RequiresRealPixelType(image.pixel_id()));
        }
        let dim = image.dimension();
        for other in [mask_image, confidence_image].into_iter().flatten() {
            if other.size() != image.size() {
                return Err(FilterError::SizeMismatch {
                    a: image.size().to_vec(),
                    b: other.size().to_vec(),
                });
            }
        }
        if settings.number_of_histogram_bins < 2 {
            return Err(FilterError::N4InvalidHistogramBins(
                settings.number_of_histogram_bins,
            ));
        }
        if settings.spline_order == 0 {
            return Err(FilterError::InvalidSplineOrder);
        }
        if settings.number_of_control_points.len() < dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: settings.number_of_control_points.len(),
            });
        }

        // `CastImageToITK<MaskImageType>`: the mask reaches ITK as `uint8`.
        let mask: Option<Vec<f64>> = mask_image
            .map(|m| -> Result<Vec<f64>> {
                Ok(m.to_f64_vec()?
                    .into_iter()
                    .map(|v| quantize_to_pixel_type(PixelId::UInt8, v))
                    .collect())
            })
            .transpose()?;
        let confidence: Option<Vec<f64>> = confidence_image.map(|c| c.to_f64_vec()).transpose()?;

        let label = f64::from(settings.mask_label);
        let included: Vec<bool> = (0..image.number_of_pixels())
            .map(|i| {
                let by_mask = match &mask {
                    None => true,
                    Some(m) if settings.use_mask_label => m[i] == label,
                    Some(m) => m[i] != 0.0,
                };
                let by_confidence = match &confidence {
                    None => true,
                    Some(c) => c[i] > 0.0,
                };
                by_mask && by_confidence
            })
            .collect();

        let confidence_weights = match &confidence {
            None => vec![1.0; image.number_of_pixels()],
            Some(c) => c.clone(),
        };

        Ok(Self {
            image,
            settings,
            included,
            confidence_weights,
            spline_order: settings.spline_order as usize,
            number_of_control_points: settings.number_of_control_points[..dim]
                .iter()
                .map(|&n| n as usize)
                .collect(),
            // `SetNumberOfFittingLevels(scalar)` fills every axis, so
            // `maximumNumberOfLevels` is that scalar.
            number_of_fitting_levels: settings.maximum_number_of_iterations.len(),
        })
    }

    /// `N4BiasFieldCorrectionImageFilter::GenerateData`.
    fn run(&self) -> Result<N4BiasFieldCorrectionResult> {
        let input = self.image.to_f64_vec()?;
        let size = self.image.size();
        let spacing = self.image.spacing();
        let dim = size.len();

        // Log-transform only the included, strictly positive voxels; every
        // other voxel keeps its raw value, exactly as upstream leaves it.
        let log_input: Vec<f64> = input
            .iter()
            .zip(&self.included)
            .map(|(&v, &keep)| if keep && v > 0.0 { v.ln() } else { v })
            .collect();
        let number_of_included_pixels = self.included.iter().filter(|&&b| b).count();

        let mut log_uncorrected = log_input.clone();
        let mut log_bias_field = vec![0.0; input.len()];
        let mut log_sharpened = vec![0.0; input.len()];
        let mut lattice: Option<Lattice> = None;

        let maximum_number_of_levels = self.number_of_fitting_levels;
        for current_level in 0..maximum_number_of_levels {
            let mut elapsed_iterations = 0u32;
            let mut convergence = f64::MAX;
            while elapsed_iterations < self.settings.maximum_number_of_iterations[current_level]
                && convergence > self.settings.convergence_threshold
            {
                elapsed_iterations += 1;

                self.sharpen_image(&log_uncorrected, &mut log_sharpened)?;
                let residual: Vec<f64> = log_uncorrected
                    .iter()
                    .zip(&log_sharpened)
                    .map(|(u, s)| u - s)
                    .collect();

                let new_log_bias_field = self.update_bias_field_estimate(
                    &residual,
                    number_of_included_pixels,
                    &mut lattice,
                )?;
                convergence =
                    self.calculate_convergence_measurement(&log_bias_field, &new_log_bias_field);
                log_bias_field = new_log_bias_field;

                for ((u, &li), &b) in log_uncorrected
                    .iter_mut()
                    .zip(&log_input)
                    .zip(&log_bias_field)
                {
                    *u = li - b;
                }
            }

            // Upstream doubles the lattice along axis `d` when
            // `m_NumberOfFittingLevels[d] + 1 >= m_CurrentLevel &&
            //  m_CurrentLevel != maximumNumberOfLevels - 1`. The first
            // conjunct is vacuously true — `m_NumberOfFittingLevels` is filled
            // with `maximumNumberOfLevels`, which already exceeds every
            // `m_CurrentLevel` — so only the last-level test bites, uniformly
            // across axes.
            let factor = if current_level + 1 == maximum_number_of_levels {
                1
            } else {
                2
            };
            let refined = {
                let current = lattice
                    .as_ref()
                    .ok_or(FilterError::N4NoBiasFieldEstimated)?;
                bspline::refine(current, self.spline_order, &vec![factor; dim])
            };
            lattice = Some(refined);
        }

        let lattice = lattice.ok_or(FilterError::N4NoBiasFieldEstimated)?;
        // SimpleITK's `GetLogBiasFieldAsImage(referenceImage)` rebuilds the
        // field from the final lattice rather than reusing `logBiasField`; with
        // the input as reference the two agree, since the last level's
        // refinement is the identity.
        let reconstructed = bspline::reconstruct(&lattice, self.spline_order, size, spacing)?;

        let corrected: Vec<f64> = input
            .iter()
            .zip(&log_bias_field)
            .map(|(&v, &b)| v / b.exp())
            .collect();

        Ok(N4BiasFieldCorrectionResult {
            corrected: image_from_f64(self.image.pixel_id(), size, self.image, &corrected)?,
            log_bias_field: image_from_f64(
                self.image.pixel_id(),
                size,
                self.image,
                &reconstructed,
            )?,
        })
    }

    /// `N4BiasFieldCorrectionImageFilter::SharpenImage`: deconvolve the
    /// log-intensity histogram with a Gaussian of width
    /// `bias_field_full_width_at_half_maximum` via a Wiener filter, then remap
    /// each included voxel through the expectation `E(u | v)`.
    ///
    /// The bin range is found with upstream's `else if` chain, which only
    /// tests a pixel against `binMinimum` when it did *not* set a new
    /// `binMaximum`. A traversal whose included voxels are strictly increasing
    /// therefore leaves `binMinimum` at `RealType::max()`, poisoning the
    /// histogram slope. That quirk is reproduced; the array indexing it
    /// corrupts is bounds-checked here instead of being undefined behaviour.
    fn sharpen_image(&self, unsharpened: &[f64], sharpened: &mut [f64]) -> Result<()> {
        let bins = self.settings.number_of_histogram_bins as usize;

        let mut bin_maximum = f64::MIN;
        let mut bin_minimum = f64::MAX;
        for (&pixel, _) in unsharpened.iter().zip(&self.included).filter(|&(_, &k)| k) {
            if pixel > bin_maximum {
                bin_maximum = pixel;
            } else if pixel < bin_minimum {
                bin_minimum = pixel;
            }
        }
        let histogram_slope = (bin_maximum - bin_minimum) / (bins - 1) as f64;

        // Triangular Parzen windowing of the included voxels into `bins` bins.
        let mut histogram = vec![0.0f64; bins];
        for (&pixel, _) in unsharpened.iter().zip(&self.included).filter(|&(_, &k)| k) {
            let cidx = (pixel - bin_minimum) / histogram_slope;
            let Some(idx) = finite_floor(cidx, bins) else {
                continue;
            };
            let offset = cidx - idx as f64;
            if offset == 0.0 {
                histogram[idx] += 1.0;
            } else if idx < bins - 1 {
                histogram[idx] += 1.0 - offset;
                histogram[idx + 1] += offset;
            }
        }

        let padded = padded_histogram_size(bins);
        let histogram_offset = (0.5 * (padded - bins) as f64) as usize;

        let mut vf = vec![Complex::new(0.0, 0.0); padded];
        for (n, &h) in histogram.iter().enumerate() {
            vf[n + histogram_offset] = Complex::new(h, 0.0);
        }
        transform_1d_unnormalized(&mut vf, false);

        // The Gaussian, sampled symmetrically about bin 0.
        let scaled_fwhm = self.settings.bias_field_full_width_at_half_maximum / histogram_slope;
        let exp_factor = 4.0 * 2.0f64.ln() / (scaled_fwhm * scaled_fwhm);
        let scale_factor = 2.0 * (2.0f64.ln() / std::f64::consts::PI).sqrt() / scaled_fwhm;

        let mut ff = vec![Complex::new(0.0, 0.0); padded];
        ff[0] = Complex::new(scale_factor, 0.0);
        let half = (0.5 * padded as f64) as usize;
        for n in 1..=half {
            let v = Complex::new(
                scale_factor * (-(n as f64) * (n as f64) * exp_factor).exp(),
                0.0,
            );
            ff[n] = v;
            ff[padded - n] = v;
        }
        if padded.is_multiple_of(2) {
            // A no-op restatement of the `n == half` term above
            // (`0.25 * padded^2 == half^2`), kept for parity with upstream.
            ff[half] = Complex::new(
                scale_factor * (0.25 * -(padded as f64) * (padded as f64) * exp_factor).exp(),
                0.0,
            );
        }
        transform_1d_unnormalized(&mut ff, false);

        // Wiener deconvolution: `Gf = conj(Ff) / (conj(Ff) * Ff + noise)`.
        let noise = self.settings.wiener_filter_noise;
        let mut u: Vec<Complex> = (0..padded)
            .map(|n| {
                let c = Complex::new(ff[n].re, -ff[n].im);
                let gf = c / (c * ff[n] + Complex::new(noise, 0.0));
                Complex::new(vf[n].re * gf.re, vf[n].im * gf.re)
            })
            .collect();
        transform_1d_unnormalized(&mut u, true);
        for x in u.iter_mut() {
            *x = Complex::new(x.re.max(0.0), 0.0);
        }

        // E(u | v) = (x * U ⊛ F) / (U ⊛ F), with the unnormalized round trip's
        // factor of `padded` cancelling between the two.
        let mut numerator: Vec<Complex> = (0..padded)
            .map(|n| {
                let x = bin_minimum + (n as f64 - histogram_offset as f64) * histogram_slope;
                Complex::new(x * u[n].re, 0.0)
            })
            .collect();
        let mut denominator = u;
        for buf in [&mut numerator, &mut denominator] {
            transform_1d_unnormalized(buf, false);
            for (n, x) in buf.iter_mut().enumerate() {
                *x = *x * ff[n];
            }
            transform_1d_unnormalized(buf, true);
        }

        let expectation: Vec<f64> = (0..bins)
            .map(|j| {
                let n = j + histogram_offset;
                if denominator[n].re != 0.0 {
                    numerator[n].re / denominator[n].re
                } else {
                    0.0
                }
            })
            .collect();

        sharpened.fill(0.0);
        for (i, &keep) in self.included.iter().enumerate() {
            if !keep {
                continue;
            }
            let cidx = (unsharpened[i] - bin_minimum) / histogram_slope;
            let Some(idx) = finite_floor(cidx, bins) else {
                continue;
            };
            sharpened[i] = if idx < bins - 1 {
                expectation[idx] + (expectation[idx + 1] - expectation[idx]) * (cidx - idx as f64)
            } else {
                expectation[bins - 1]
            };
        }
        Ok(())
    }

    /// `N4BiasFieldCorrectionImageFilter::UpdateBiasFieldEstimate`: fit a
    /// B-spline scalar field to `residual` over the included voxels, add its
    /// control-point lattice to the running estimate, and reconstruct the
    /// total field on the input grid.
    ///
    /// Upstream re-imports the residual with an identity direction cosine
    /// matrix — "the B-spline approximation algorithm works in parametric
    /// space and not physical space" — so the sample point of voxel `idx` is
    /// `origin + spacing * idx` with no rotation.
    fn update_bias_field_estimate(
        &self,
        residual: &[f64],
        number_of_included_pixels: usize,
        lattice: &mut Option<Lattice>,
    ) -> Result<Vec<f64>> {
        let size = self.image.size();
        let spacing = self.image.spacing();
        let origin = self.image.origin();
        let dim = size.len();

        let mut points = Vec::with_capacity(number_of_included_pixels * dim);
        let mut values = Vec::with_capacity(number_of_included_pixels);
        let mut weights = Vec::with_capacity(number_of_included_pixels);
        for (i, &keep) in self.included.iter().enumerate() {
            if !keep {
                continue;
            }
            let mut rem = i;
            for d in 0..dim {
                points.push(origin[d] + (rem % size[d]) as f64 * spacing[d]);
                rem /= size[d];
            }
            values.push(residual[i]);
            weights.push(self.confidence_weights[i]);
        }

        // Once a lattice exists, refinement has fixed its size; fit into a
        // lattice of the same shape so the two can be added.
        let number_of_control_points: Vec<usize> = match lattice {
            None => self.number_of_control_points.clone(),
            Some(l) => l.size.clone(),
        };

        let phi = bspline::fit(&FitInput {
            size,
            spacing,
            origin,
            spline_order: self.spline_order,
            number_of_control_points: &number_of_control_points,
            points: &points,
            values: &values,
            weights: &weights,
        })?;

        match lattice {
            None => *lattice = Some(phi),
            Some(l) => l.add_assign(&phi),
        }
        let lattice = lattice.as_ref().expect("just populated");
        bspline::reconstruct(lattice, self.spline_order, size, spacing)
    }

    /// `N4BiasFieldCorrectionImageFilter::CalculateConvergenceMeasurement`: the
    /// coefficient of variation of `exp(field1 − field2)` over the included
    /// voxels, accumulated with the same one-pass recurrence upstream uses.
    fn calculate_convergence_measurement(&self, field1: &[f64], field2: &[f64]) -> f64 {
        let mut mu = 0.0;
        let mut sigma = 0.0;
        let mut n = 0.0f64;
        for ((&a, &b), _) in field1
            .iter()
            .zip(field2)
            .zip(&self.included)
            .filter(|&(_, &keep)| keep)
        {
            let pixel = (a - b).exp();
            n += 1.0;
            if n > 1.0 {
                sigma += (pixel - mu) * (pixel - mu) * (n - 1.0) / n;
            }
            mu = mu * (1.0 - 1.0 / n) + pixel / n;
        }
        (sigma / (n - 1.0)).sqrt() / mu
    }
}

/// The power-of-two length `SharpenImage` zero-pads its `bins`-bin histogram
/// to:
///
/// ```text
/// exponent = ceil(log(bins) / log(2)) + 1
/// padded   = pow(2, exponent) + 0.5
/// ```
///
/// The `float` in `std::log(static_cast<RealType>(m_NumberOfHistogramBins))`
/// is load-bearing: `RealType` is `float`, so this is `logf`, and for a
/// power-of-two `bins` the rounded-up `logf` pushes the quotient just past the
/// integer — `logf(256) / log(2) == 8.0000000225`, whose `ceil` is `9`, not
/// `8`. Upstream therefore pads a power-of-two histogram to `4 * bins` and
/// everything else to `2 * next_power_of_two(bins)`. Computing the log in
/// `f64` would halve the padding at exactly those bin counts and change the
/// circular convolution's wraparound, so the `f32` log is reproduced here even
/// though the rest of this port widens to `f64`.
fn padded_histogram_size(bins: usize) -> usize {
    let exponent = (f64::from((bins as f32).ln()) / 2.0f64.ln()).ceil() + 1.0;
    (2.0f64.powf(exponent) + 0.5) as usize
}

/// `static_cast<unsigned int>(itk::Math::floor(cidx))` guarded against the
/// negative and non-finite `cidx` upstream's unsigned cast leaves undefined.
fn finite_floor(cidx: f64, bins: usize) -> Option<usize> {
    if !cidx.is_finite() || cidx < 0.0 {
        return None;
    }
    let idx = cidx.floor() as usize;
    (idx < bins).then_some(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: usize = 32;

    /// Multiplicative log-bias field: a smooth, strictly-B-spline-representable
    /// surface would let N4 recover it exactly; a low-order trigonometric bump
    /// is a fairer test of the fit.
    fn true_log_bias(x: usize, y: usize) -> f64 {
        let u = x as f64 / (SIZE - 1) as f64;
        let v = y as f64 / (SIZE - 1) as f64;
        0.18 * (1.6 * u - 0.4) + 0.14 * (v - 0.5) * (v - 0.5) - 0.08 * u * v
    }

    /// Two tissue classes plus a deterministic ripple, so the histogram is not
    /// a pair of deltas and the raster traversal is not monotonic.
    fn tissue(x: usize, y: usize) -> f64 {
        let cx = x as f64 - 15.5;
        let cy = y as f64 - 15.5;
        let base = if cx * cx + cy * cy < 81.0 {
            160.0
        } else {
            90.0
        };
        base * (1.0 + 0.01 * ((x * 7 + y * 13) % 5) as f64)
    }

    fn phantom() -> (Image, Vec<f64>) {
        let mut data = Vec::with_capacity(SIZE * SIZE);
        let mut field = Vec::with_capacity(SIZE * SIZE);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let b = true_log_bias(x, y);
                field.push(b);
                data.push((tissue(x, y) * b.exp()) as f32);
            }
        }
        (Image::from_vec(&[SIZE, SIZE], data).unwrap(), field)
    }

    fn settings() -> N4BiasFieldCorrectionSettings {
        N4BiasFieldCorrectionSettings {
            maximum_number_of_iterations: vec![20, 20],
            number_of_control_points: vec![4, 4],
            ..Default::default()
        }
    }

    /// Coefficient of variation of `values` restricted to one tissue class,
    /// which is what bias correction is supposed to shrink.
    fn coefficient_of_variation(values: &[f64], select: impl Fn(usize, usize) -> bool) -> f64 {
        let picked: Vec<f64> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| select(x, y))
            .map(|(x, y)| values[y * SIZE + x])
            .collect();
        let n = picked.len() as f64;
        let mean = picked.iter().sum::<f64>() / n;
        let var = picked.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
        var.sqrt() / mean
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for (x, y) in a.iter().zip(b) {
            cov += (x - ma) * (y - mb);
            va += (x - ma) * (x - ma);
            vb += (y - mb) * (y - mb);
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    fn in_blob(x: usize, y: usize) -> bool {
        let cx = x as f64 - 15.5;
        let cy = y as f64 - 15.5;
        cx * cx + cy * cy < 81.0
    }

    #[test]
    fn correction_shrinks_the_within_tissue_coefficient_of_variation() {
        let (image, _) = phantom();
        let before = image.to_f64_vec().unwrap();
        let after = n4_bias_field_correction(&image, None, None, &settings())
            .unwrap()
            .to_f64_vec()
            .unwrap();

        for (name, select) in [
            ("blob", &in_blob as &dyn Fn(usize, usize) -> bool),
            ("background", &|x, y| !in_blob(x, y)),
        ] {
            let cv_before = coefficient_of_variation(&before, select);
            let cv_after = coefficient_of_variation(&after, select);
            assert!(
                cv_after < 0.5 * cv_before,
                "{name}: coefficient of variation did not improve ({cv_before} -> {cv_after})"
            );
        }
    }

    #[test]
    fn the_recovered_log_field_tracks_the_injected_one() {
        let (image, truth) = phantom();
        let result =
            n4_bias_field_correction_with_log_bias_field(&image, None, None, &settings()).unwrap();
        let recovered = result.log_bias_field.to_f64_vec().unwrap();
        let r = pearson(&recovered, &truth);
        assert!(r > 0.99, "log-bias-field correlation was only {r}");
    }

    /// The two outputs are two views of one estimate: the corrected image is
    /// exactly `input / exp(log_bias_field)`.
    #[test]
    fn the_corrected_image_is_the_input_divided_by_the_exponentiated_field() {
        let (image, _) = phantom();
        let input = image.to_f64_vec().unwrap();
        let result =
            n4_bias_field_correction_with_log_bias_field(&image, None, None, &settings()).unwrap();
        let corrected = result.corrected.to_f64_vec().unwrap();
        let field = result.log_bias_field.to_f64_vec().unwrap();
        for i in 0..input.len() {
            let expected = (input[i] / field[i].exp()) as f32 as f64;
            assert!(
                (corrected[i] - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "voxel {i}: {} vs {expected}",
                corrected[i]
            );
        }
    }

    /// `use_mask_label` selects a single label; with it off, every non-zero
    /// mask voxel counts. Restricting the estimate to the blob must change the
    /// result, and the two mask semantics must disagree when the mask carries
    /// two labels.
    #[test]
    fn mask_label_semantics_select_different_voxel_sets() {
        let (image, _) = phantom();
        let mask: Vec<u8> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| if in_blob(x, y) { 1u8 } else { 2u8 }))
            .collect();
        let mask = Image::from_vec(&[SIZE, SIZE], mask).unwrap();

        let label_only = N4BiasFieldCorrectionSettings {
            use_mask_label: true,
            mask_label: 1,
            ..settings()
        };
        let all_nonzero = N4BiasFieldCorrectionSettings {
            use_mask_label: false,
            ..settings()
        };

        let a = n4_bias_field_correction(&image, Some(&mask), None, &label_only)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let b = n4_bias_field_correction(&image, Some(&mask), None, &all_nonzero)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert!(
            a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6),
            "the two mask semantics produced identical corrections"
        );

        // Label 2 selects the background, a different set again.
        let other_label = N4BiasFieldCorrectionSettings {
            mask_label: 2,
            ..label_only
        };
        let c = n4_bias_field_correction(&image, Some(&mask), None, &other_label)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert!(a.iter().zip(&c).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    /// A confidence image of zeros outside the blob excludes exactly the voxels
    /// a `{0, 1}` mask would, so the two must agree bit for bit.
    #[test]
    fn a_binary_confidence_image_matches_the_equivalent_mask() {
        let (image, _) = phantom();
        let binary: Vec<f32> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| if in_blob(x, y) { 1.0f32 } else { 0.0 }))
            .collect();
        let confidence = Image::from_vec(&[SIZE, SIZE], binary.clone()).unwrap();
        let mask =
            Image::from_vec(&[SIZE, SIZE], binary.iter().map(|&v| v as u8).collect()).unwrap();

        let by_confidence = n4_bias_field_correction(&image, None, Some(&confidence), &settings())
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let by_mask = n4_bias_field_correction(&image, Some(&mask), None, &settings())
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(by_confidence, by_mask);
    }

    /// A non-uniform confidence image reweights the B-spline fit, so it cannot
    /// agree with the unweighted run over the same voxel set.
    #[test]
    fn confidence_weights_change_the_fit() {
        let (image, _) = phantom();
        let graded: Vec<f32> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| 0.1 + 0.9 * (x + y) as f32 / (2 * SIZE) as f32))
            .collect();
        let confidence = Image::from_vec(&[SIZE, SIZE], graded).unwrap();

        let weighted = n4_bias_field_correction(&image, None, Some(&confidence), &settings())
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let plain = n4_bias_field_correction(&image, None, None, &settings())
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert!(
            weighted
                .iter()
                .zip(&plain)
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    /// Every fitting level after the first sees a refined lattice, so adding a
    /// level changes the estimate. A single level must still converge.
    #[test]
    fn additional_fitting_levels_refine_the_lattice() {
        let (image, truth) = phantom();
        let one = N4BiasFieldCorrectionSettings {
            maximum_number_of_iterations: vec![20],
            ..settings()
        };
        let three = N4BiasFieldCorrectionSettings {
            maximum_number_of_iterations: vec![20, 20, 20],
            ..settings()
        };
        let a = n4_bias_field_correction_with_log_bias_field(&image, None, None, &one).unwrap();
        let b = n4_bias_field_correction_with_log_bias_field(&image, None, None, &three).unwrap();
        let fa = a.log_bias_field.to_f64_vec().unwrap();
        let fb = b.log_bias_field.to_f64_vec().unwrap();
        assert!(fa.iter().zip(&fb).any(|(x, y)| (x - y).abs() > 1e-6));

        // The refined lattice resolves the injected field strictly better.
        let ra = pearson(&fa, &truth);
        let rb = pearson(&fb, &truth);
        assert!(ra > 0.98, "single-level correlation was only {ra}");
        assert!(
            rb > ra,
            "refinement did not improve correlation ({ra} -> {rb})"
        );
    }

    #[test]
    fn an_integer_input_is_rejected() {
        let image = Image::from_vec(&[4usize, 4], vec![1u8; 16]).unwrap();
        assert_eq!(
            n4_bias_field_correction(&image, None, None, &settings()).unwrap_err(),
            FilterError::RequiresRealPixelType(PixelId::UInt8)
        );
    }

    #[test]
    fn a_mask_of_the_wrong_size_is_rejected() {
        let (image, _) = phantom();
        let mask = Image::from_vec(&[4usize, 4], vec![1u8; 16]).unwrap();
        assert_eq!(
            n4_bias_field_correction(&image, Some(&mask), None, &settings()).unwrap_err(),
            FilterError::SizeMismatch {
                a: vec![SIZE, SIZE],
                b: vec![4, 4],
            }
        );
    }

    #[test]
    fn one_histogram_bin_is_rejected() {
        let (image, _) = phantom();
        let s = N4BiasFieldCorrectionSettings {
            number_of_histogram_bins: 1,
            ..settings()
        };
        assert_eq!(
            n4_bias_field_correction(&image, None, None, &s).unwrap_err(),
            FilterError::N4InvalidHistogramBins(1)
        );
    }

    #[test]
    fn spline_order_zero_is_rejected() {
        let (image, _) = phantom();
        let s = N4BiasFieldCorrectionSettings {
            spline_order: 0,
            ..settings()
        };
        assert_eq!(
            n4_bias_field_correction(&image, None, None, &s).unwrap_err(),
            FilterError::InvalidSplineOrder
        );
    }

    #[test]
    fn too_few_control_point_entries_are_rejected() {
        let (image, _) = phantom();
        let s = N4BiasFieldCorrectionSettings {
            number_of_control_points: vec![4],
            ..settings()
        };
        assert_eq!(
            n4_bias_field_correction(&image, None, None, &s).unwrap_err(),
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    /// An empty iteration schedule, or a zero first level, leaves the lattice
    /// unbuilt where upstream would dereference a null pointer.
    #[test]
    fn a_schedule_that_never_fits_is_rejected() {
        let (image, _) = phantom();
        for schedule in [vec![], vec![0], vec![0, 20]] {
            let s = N4BiasFieldCorrectionSettings {
                maximum_number_of_iterations: schedule.clone(),
                ..settings()
            };
            assert_eq!(
                n4_bias_field_correction(&image, None, None, &s).unwrap_err(),
                FilterError::N4NoBiasFieldEstimated,
                "schedule {schedule:?}"
            );
        }
    }

    /// `padded_histogram_size` must be a power of two strictly above `bins`,
    /// and must reproduce upstream's `logf` rounding: a power-of-two `bins`
    /// pads to `4 * bins`, not `2 * bins`.
    #[test]
    fn the_padded_histogram_size_reproduces_the_float_log_rounding() {
        assert_eq!(padded_histogram_size(200), 512);
        assert_eq!(padded_histogram_size(255), 512);
        assert_eq!(padded_histogram_size(257), 1024);
        // Powers of two: `ceil(logf(n) / log 2)` overshoots by one.
        assert_eq!(padded_histogram_size(2), 8);
        assert_eq!(padded_histogram_size(64), 256);
        assert_eq!(padded_histogram_size(128), 512);
        assert_eq!(padded_histogram_size(256), 1024);

        for bins in 2..=600usize {
            let padded = padded_histogram_size(bins);
            assert!(padded.is_power_of_two(), "bins={bins} padded={padded}");
            assert!(padded > bins, "bins={bins} padded={padded}");
        }
    }

    /// The SimpleITK defaults: four fitting levels of fifty iterations, cubic
    /// splines, a 4-control-point lattice, 200 histogram bins.
    #[test]
    fn defaults_match_the_simpleitk_yaml() {
        let d = N4BiasFieldCorrectionSettings::default();
        assert_eq!(d.convergence_threshold, 0.001);
        assert_eq!(d.maximum_number_of_iterations, vec![50, 50, 50, 50]);
        assert_eq!(d.bias_field_full_width_at_half_maximum, 0.15);
        assert_eq!(d.wiener_filter_noise, 0.01);
        assert_eq!(d.number_of_histogram_bins, 200);
        assert_eq!(d.number_of_control_points, vec![4, 4, 4]);
        assert_eq!(d.spline_order, 3);
        assert!(d.use_mask_label);
        assert_eq!(d.mask_label, 1);
    }

    /// `number_of_control_points` is SimpleITK's `dim_vec`: the 3-D default
    /// applies unchanged to a 2-D image, which reads only its first two
    /// entries.
    #[test]
    fn the_three_dimensional_control_point_default_applies_to_a_two_dimensional_image() {
        let (image, _) = phantom();
        let s = N4BiasFieldCorrectionSettings {
            maximum_number_of_iterations: vec![5],
            ..Default::default()
        };
        assert_eq!(s.number_of_control_points.len(), 3);
        assert!(n4_bias_field_correction(&image, None, None, &s).is_ok());
    }

    /// A 3-D run exercises the dimension-generic lattice and collapse code.
    #[test]
    fn a_three_dimensional_volume_is_corrected() {
        const N: usize = 12;
        let mut data = Vec::with_capacity(N * N * N);
        for z in 0..N {
            for y in 0..N {
                for x in 0..N {
                    let b = 0.3 * (x as f64 / N as f64) + 0.2 * (y as f64 / N as f64)
                        - 0.1 * (z as f64 / N as f64);
                    let base = 100.0 + 4.0 * ((x + 2 * y + 3 * z) % 3) as f64;
                    data.push((base * b.exp()) as f32);
                }
            }
        }
        let image = Image::from_vec(&[N, N, N], data).unwrap();
        let s = N4BiasFieldCorrectionSettings {
            maximum_number_of_iterations: vec![10, 10],
            ..Default::default()
        };
        let before = image.to_f64_vec().unwrap();
        let after = n4_bias_field_correction(&image, None, None, &s)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let cv = |v: &[f64]| {
            let n = v.len() as f64;
            let m = v.iter().sum::<f64>() / n;
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt() / m
        };
        assert!(
            cv(&after) < cv(&before),
            "{} !< {}",
            cv(&after),
            cv(&before)
        );
    }
}
