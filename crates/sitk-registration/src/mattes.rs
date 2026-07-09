//! Mattes mutual-information image-to-image metric
//! (`itk::MattesMutualInformationImageToImageMetricv4`).
//!
//! Mutual information measures the statistical dependence between the fixed
//! image `F` and the transformed moving image `M(T(x))` from their joint
//! intensity distribution, **without assuming a linear intensity relationship**.
//! That makes it the metric for *multi-modality* registration (e.g. CT↔MR, or
//! any pair related by an arbitrary invertible intensity map), where mean
//! squares — which wants `M ≈ F` — fails.
//!
//! ```text
//! MI = Σ_{f,m} p(f,m) · log( p(f,m) / ( p_F(f) · p_M(m) ) )
//! ```
//!
//! The joint density `p(f,m)` is estimated with **Parzen windowing** over a
//! `bins × bins` histogram (Mattes et al. 2003): each sample's fixed intensity
//! lands in one bin through a zero-order (box) window, and its moving intensity
//! is spread over four bins through a **cubic B-spline** window. The metric the
//! optimizer minimizes is `value = −MI`; its derivative with respect to the
//! transform parameters is the analytic Mattes/Thévenaz–Unser form
//!
//! ```text
//! ∂value/∂p_k = Σ_{f,m} ( ∂p(f,m)/∂p_k ) · log( p(f,m) / p_M(m) )
//! ```
//!
//! where `∂p(f,m)/∂p_k` comes from the cubic B-spline window's derivative times
//! `∇M(T(x)) · J_T(x)` — the moving image gradient projected through the
//! transform Jacobian, exactly as in mean squares.
//!
//! ## Parity notes vs ITK
//!
//! * **Full sampling.** Like the mean-squares metric here, this uses *every*
//!   fixed pixel (SimpleITK's default sampling strategy = None), so the fixed
//!   and moving intensity ranges that size the histogram are the whole-image
//!   ranges — matching ITK's dense, unmasked `Initialize()` path.
//! * **Gradient source.** `∇M` is the exact gradient of the *linear
//!   interpolant* (`MovingImage::value_and_physical_gradient`), so the metric
//!   derivative is the true gradient of the interpolated MI value (an
//!   optimizer's finite difference of the value reproduces it). This is the same
//!   deliberate deviation the mean-squares metric documents: ITK defaults to a
//!   separately-computed (Gaussian-smoothed or central-difference) gradient
//!   image that is not consistent with the interpolated value.
//! * **Global-support derivative path (covers BSpline).** The dense
//!   `jointPDFDerivatives` accumulation is ported. In ITK v4 this is the path
//!   taken by every transform whose `HasLocalSupport()` is false — which, per
//!   `itk::ObjectToObjectMetric::HasLocalSupport`, means every category *except*
//!   `DisplacementField`. A [`BSplineTransform`] reports
//!   `GetTransformCategory() == BSpline`, so it is **not** local-support and is
//!   handled here exactly as ITK handles it: the (sparse) transform Jacobian is
//!   folded densely over all parameters. This makes **deformable, multi-modality
//!   registration** (BSpline + mutual information) work; it is finite-difference
//!   verified against the value in the tests.
//! * **Displacement-field local-support branch.** ITK's genuine local-support
//!   path fires only for a [`DisplacementFieldTransform`]
//!   (`HasLocalSupport() == true`), where each pixel's displacement is governed
//!   by its own parameter block. It is ported: `evaluate` dispatches on
//!   [`ParametricTransform::has_local_support`] to a per-pixel accumulation that
//!   keeps only ITK's compact `m_LocalDerivativeByParzenBin` (`4 × params`),
//!   `m_JointPdfIndex1DArray` (`params`), and `m_PRatioArray` (`bins²`) instead
//!   of the dense `bins² × params` array — the same result, memory linear rather
//!   than quadratic in the parameters. A test asserts it reproduces the global
//!   path's derivative for the same field.
//!
//! [`BSplineTransform`]: sitk_transform::BSplineTransform
//! [`DisplacementFieldTransform`]: sitk_transform::DisplacementFieldTransform

use sitk_core::Image;
use sitk_transform::ParametricTransform;

use crate::error::{RegistrationError, Result};
use crate::metric::{FixedSamples, MetricValue, MovingImage};
use crate::scales::PhysicalShiftScales;

/// Bins of padding at each histogram-axis end, reserved so the cubic B-spline
/// Parzen window never needs a boundary condition. ITK's `padding`.
const PADDING: usize = 2;

/// The order-3 (cubic) B-spline kernel `B₃(u)`, the moving-image Parzen window.
/// Verbatim from `itk::BSplineKernelFunction<3>::Evaluate`.
fn cubic_bspline(u: f64) -> f64 {
    let a = u.abs();
    if a < 1.0 {
        let sq = a * a;
        (4.0 - 6.0 * sq + 3.0 * sq * a) / 6.0
    } else if a < 2.0 {
        let sq = a * a;
        (8.0 - 12.0 * a + 6.0 * sq - sq * a) / 6.0
    } else {
        0.0
    }
}

/// The derivative `B₃'(u)` of the cubic B-spline kernel. Verbatim from
/// `itk::BSplineDerivativeKernelFunction<3>::Evaluate` — note it is written in
/// terms of the signed `u`, not `|u|`, with distinct branches per sign.
fn cubic_bspline_derivative(u: f64) -> f64 {
    if (0.0..1.0).contains(&u) {
        -2.0 * u + 1.5 * u * u
    } else if u > -1.0 && u < 0.0 {
        -2.0 * u - 1.5 * u * u
    } else if (1.0..2.0).contains(&u) {
        -2.0 + 2.0 * u - 0.5 * u * u
    } else if u > -2.0 && u <= -1.0 {
        2.0 + 2.0 * u + 0.5 * u * u
    } else {
        0.0
    }
}

/// The Mattes mutual-information metric. Holds the precomputed fixed samples,
/// moving image, and the joint-histogram geometry (bin sizes and normalized
/// minima) derived once from the fixed/moving intensity ranges.
/// [`evaluate`](Self::evaluate) returns `value = −MI` plus its
/// parameter-derivative for a given transform.
pub struct MattesMutualInformationMetric {
    fixed: FixedSamples,
    moving: MovingImage,
    num_bins: usize,
    /// Moving intensity range, used to reject out-of-range interpolated values.
    moving_true_min: f64,
    moving_true_max: f64,
    /// Histogram bin sizes: `(trueMax − trueMin) / (bins − 2·padding)`.
    fixed_bin_size: f64,
    moving_bin_size: f64,
    /// Normalized minima: `trueMin / binSize − padding`. A pixel value `v` maps
    /// to the fractional bin coordinate `v / binSize − normalizedMin`.
    fixed_normalized_min: f64,
    moving_normalized_min: f64,
}

impl MattesMutualInformationMetric {
    /// Build the metric from a fixed and moving image and a histogram bin count
    /// (ITK/SimpleITK default 50). Fails if dimensions disagree, the moving
    /// direction matrix is singular, fewer than five bins are requested, or
    /// either image is constant (MI is then undefined).
    pub fn new(fixed: &Image, moving: &Image, number_of_histogram_bins: usize) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        if number_of_histogram_bins < 2 * PADDING + 1 {
            return Err(RegistrationError::TooFewHistogramBins {
                bins: number_of_histogram_bins,
            });
        }

        let fixed_samples = FixedSamples::from_image(fixed);
        let moving_image = MovingImage::from_image(moving)?;

        let (fixed_min, fixed_max) = fixed_samples.value_range();
        let (moving_min, moving_max) = moving_image.value_range();
        if fixed_max - fixed_min <= f64::EPSILON {
            return Err(RegistrationError::ConstantIntensity { which: "fixed" });
        }
        if moving_max - moving_min <= f64::EPSILON {
            return Err(RegistrationError::ConstantIntensity { which: "moving" });
        }

        // Bin size padded so the cubic Parzen window stays inside the histogram;
        // the minimum is shifted by `padding` bins to match.
        let denom = (number_of_histogram_bins - 2 * PADDING) as f64;
        let fixed_bin_size = (fixed_max - fixed_min) / denom;
        let moving_bin_size = (moving_max - moving_min) / denom;

        Ok(Self {
            fixed: fixed_samples,
            moving: moving_image,
            num_bins: number_of_histogram_bins,
            moving_true_min: moving_min,
            moving_true_max: moving_max,
            fixed_bin_size,
            moving_bin_size,
            fixed_normalized_min: fixed_min / fixed_bin_size - PADDING as f64,
            moving_normalized_min: moving_min / moving_bin_size - PADDING as f64,
        })
    }

    /// Build the metric from an already-configured [`FixedSamples`] and
    /// [`MovingImage`] — the seam for a custom sampling strategy, fixed/moving
    /// mask, or interpolator (see [`FixedSamples::from_image_with`] and
    /// [`MovingImage::from_image_with_interpolator`]). Fails if their spatial
    /// dimensions disagree, fewer than five bins are requested, or either
    /// sample set's intensity range is constant — the same checks [`new`](Self::new)
    /// performs.
    pub fn from_samples(
        fixed: FixedSamples,
        moving: MovingImage,
        number_of_histogram_bins: usize,
    ) -> Result<Self> {
        if fixed.dim != moving.dim() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dim,
                moving: moving.dim(),
            });
        }
        if number_of_histogram_bins < 2 * PADDING + 1 {
            return Err(RegistrationError::TooFewHistogramBins {
                bins: number_of_histogram_bins,
            });
        }

        let (fixed_min, fixed_max) = fixed.value_range();
        let (moving_min, moving_max) = moving.value_range();
        if fixed_max - fixed_min <= f64::EPSILON {
            return Err(RegistrationError::ConstantIntensity { which: "fixed" });
        }
        if moving_max - moving_min <= f64::EPSILON {
            return Err(RegistrationError::ConstantIntensity { which: "moving" });
        }

        let denom = (number_of_histogram_bins - 2 * PADDING) as f64;
        let fixed_bin_size = (fixed_max - fixed_min) / denom;
        let moving_bin_size = (moving_max - moving_min) / denom;

        Ok(Self {
            fixed,
            moving,
            num_bins: number_of_histogram_bins,
            moving_true_min: moving_min,
            moving_true_max: moving_max,
            fixed_bin_size,
            moving_bin_size,
            fixed_normalized_min: fixed_min / fixed_bin_size - PADDING as f64,
            moving_normalized_min: moving_min / moving_bin_size - PADDING as f64,
        })
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// Build a physical-shift scale/learning-rate estimator for `transform` over
    /// this metric's fixed sample points (shared with the mean-squares metric).
    pub fn physical_shift_scales(
        &self,
        transform: &dyn ParametricTransform,
    ) -> PhysicalShiftScales {
        self.fixed.physical_shift_scales(transform)
    }

    /// The Parzen-window bin index of a pixel `value` on the axis with bin size
    /// `bin_size` and normalized minimum `normalized_min`, clamped to the
    /// interior `[padding, bins − padding − 1]` so all four cubic-window taps
    /// stay in range. Mirrors ITK's `ComputeSingleFixedImageParzenWindowIndex`
    /// and the identical clamp applied to the moving index in `ProcessPoint`.
    fn parzen_window_index(&self, value: f64, bin_size: f64, normalized_min: f64) -> usize {
        let term = value / bin_size - normalized_min;
        // ITK static_cast<OffsetValueType> truncates toward zero; `term` is
        // always ≥ padding ≥ 0 by construction, so truncation == floor here.
        let mut index = term as isize;
        let lo = PADDING as isize;
        let hi = self.num_bins as isize - PADDING as isize - 1;
        if index < lo {
            index = lo;
        } else if index > hi {
            index = hi;
        }
        index as usize
    }

    /// Evaluate `value = −MI` and its parameter-derivative for `transform`.
    ///
    /// The value is identical for every transform; only how the derivative is
    /// accumulated differs by transform category, exactly as ITK keys on
    /// `HasLocalSupport()`. A global transform (translation, affine, versor,
    /// B-spline — everything except a displacement field) takes the dense
    /// `evaluate_global_support` path; a
    /// [local-support](ParametricTransform::has_local_support) displacement field
    /// takes the per-pixel `evaluate_local_support` path, which never
    /// materializes the `bins² × numberOfParameters` derivative array.
    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        if transform.has_local_support() {
            self.evaluate_local_support(transform)
        } else {
            self.evaluate_global_support(transform)
        }
    }

    /// Global-support derivative accumulation (ITK's `!HasLocalSupport` branch).
    ///
    /// Two passes over the fixed samples' contributions, exactly as ITK: the
    /// first accumulates the joint histogram, the fixed marginal, and the
    /// per-bin joint-PDF parameter derivatives; the second walks the histogram
    /// to form `−MI` and folds each bin's `pRatio` into the derivative.
    fn evaluate_global_support(&self, transform: &dyn ParametricTransform) -> MetricValue {
        let dim = self.fixed.dim;
        let bins = self.num_bins;
        let nparams = transform.number_of_parameters();
        let n = self.fixed.len();

        // Joint histogram, row-major [fixedBin * bins + movingBin].
        let mut joint_pdf = vec![0.0f64; bins * bins];
        // Fixed marginal (box window ⇒ one bin per sample).
        let mut fixed_marginal = vec![0.0f64; bins];
        // Joint-PDF derivatives, [(fixedBin * bins + movingBin) * nparams + k].
        let mut joint_pdf_derivatives = vec![0.0f64; bins * bins * nparams];
        let mut valid = 0usize;

        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];

            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match self.moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue, // maps outside the moving buffer
            };
            // Reject values outside the histogram's moving range (matches ITK;
            // a linear interpolant of in-range values only exceeds this by
            // round-off, but the guard keeps the bin index well-defined).
            if mv < self.moving_true_min || mv > self.moving_true_max {
                continue;
            }

            let moving_term = mv / self.moving_bin_size - self.moving_normalized_min;
            let moving_index =
                self.parzen_window_index(mv, self.moving_bin_size, self.moving_normalized_min);
            let fixed_index =
                self.parzen_window_index(fv, self.fixed_bin_size, self.fixed_normalized_min);

            // Fixed marginal: zero-order (box) window ⇒ increment one bin.
            fixed_marginal[fixed_index] += 1.0;

            // Cubic window covers the four bins [moving_index − 1 .. + 2].
            let jac = transform.jacobian_wrt_parameters(fp);
            let mut pdf_moving_index = moving_index - 1;
            let mut arg = pdf_moving_index as f64 - moving_term;
            for _ in 0..4 {
                let val = cubic_bspline(arg);
                joint_pdf[fixed_index * bins + pdf_moving_index] += val;

                let deriv_weight = cubic_bspline_derivative(arg);
                let base = (fixed_index * bins + pdf_moving_index) * nparams;
                for k in 0..nparams {
                    // inner = ∇M · (column k of the transform Jacobian).
                    let mut inner = 0.0;
                    for (d, &g) in grad_phys.iter().enumerate() {
                        inner += jac[d * nparams + k] * g;
                    }
                    joint_pdf_derivatives[base + k] += inner * deriv_weight;
                }

                arg += 1.0;
                pdf_moving_index += 1;
            }

            valid += 1;
        }

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }

        // Total histogram mass; each valid sample contributes ~1 (the cubic
        // window's four taps sum to 1), so this ≈ valid.
        let joint_sum: f64 = joint_pdf.iter().sum();
        if joint_sum < f64::EPSILON {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: valid,
            };
        }

        // Fold 1/(binSize·N) into every joint-PDF derivative: 1/binSize is the
        // chain-rule factor |∂arg/∂value| and 1/N normalizes with the
        // histogram-mass normalization applied to the PDF below.
        //
        // Sign vs ITK: ITK's `nFactor` is *negative* because its v4 optimizers
        // ADD the returned derivative (so metrics store the descent direction,
        // −∇value). This crate's optimizers SUBTRACT (`p −= lr·derivative`), so
        // every metric stores the true gradient +∇value — the same convention
        // flip the mean-squares metric makes by differencing `M − F` where ITK
        // differences `F − M`. Hence the positive sign here. The finite-
        // difference test pins this down: `derivative == d(value)/d(param)`.
        let n_factor = 1.0 / (self.moving_bin_size * valid as f64);
        for d in joint_pdf_derivatives.iter_mut() {
            *d *= n_factor;
        }

        // Normalize the joint PDF and the fixed marginal to sum to 1.
        let inv_sum = 1.0 / joint_sum;
        for p in joint_pdf.iter_mut() {
            *p *= inv_sum;
        }
        for p in fixed_marginal.iter_mut() {
            *p *= inv_sum;
        }

        // Moving marginal = column sums of the normalized joint PDF.
        let mut moving_marginal = vec![0.0f64; bins];
        for f in 0..bins {
            for (m, mm) in moving_marginal.iter_mut().enumerate() {
                *mm += joint_pdf[f * bins + m];
            }
        }

        let close_to_zero = f64::EPSILON;
        let mut sum = 0.0f64;
        let mut derivative = vec![0.0f64; nparams];
        for f in 0..bins {
            let fm = fixed_marginal[f];
            if fm <= close_to_zero {
                continue;
            }
            let log_fm = fm.ln();
            for m in 0..bins {
                let mm = moving_marginal[m];
                let jp = joint_pdf[f * bins + m];
                if mm > close_to_zero && jp > close_to_zero {
                    let p_ratio = (jp / mm).ln();
                    sum += jp * (p_ratio - log_fm);

                    let base = (f * bins + m) * nparams;
                    for (k, dk) in derivative.iter_mut().enumerate() {
                        *dk += joint_pdf_derivatives[base + k] * p_ratio;
                    }
                }
            }
        }

        // The optimizer minimizes, so the value is −MI (maximizing MI); the
        // derivative is +∇(−MI), the true gradient of that value (see the
        // `n_factor` sign note above).
        MetricValue {
            value: -sum,
            derivative,
            valid_points: valid,
        }
    }

    /// Local-support derivative accumulation (ITK's `HasLocalSupport` branch,
    /// fired only for a displacement field).
    ///
    /// A local-support transform governs each point by its own small parameter
    /// block ([`local_support_jacobian`]), so a sample's four cubic-window taps
    /// only ever touch *that* block. Instead of the dense `bins² ×
    /// numberOfParameters` derivative array, this keeps ITK's compact trio:
    /// `local_deriv_by_bin` (`4 × numberOfParameters`, the per-parzen-bin
    /// projected gradient — `m_LocalDerivativeByParzenBin`), `jointpdf_index_1d`
    /// (the 1-D joint-PDF index each parameter's sample lands in —
    /// `m_JointPdfIndex1DArray`), and `pratio` (`bins²`, the per-bin `pRatio`
    /// scaled by `n_factor` — `m_PRatioArray`). The result is *identical* to the
    /// global path — each parameter's contribution is exactly its owning
    /// sample's — but the memory is linear in the parameters, not quadratic in
    /// the histogram times the parameters.
    ///
    /// [`local_support_jacobian`]: ParametricTransform::local_support_jacobian
    fn evaluate_local_support(&self, transform: &dyn ParametricTransform) -> MetricValue {
        let dim = self.fixed.dim;
        let bins = self.num_bins;
        let nparams = transform.number_of_parameters();
        let num_local = transform.number_of_local_parameters();
        let n = self.fixed.len();

        let mut joint_pdf = vec![0.0f64; bins * bins];
        let mut fixed_marginal = vec![0.0f64; bins];
        // ITK m_LocalDerivativeByParzenBin[bin][param] and m_JointPdfIndex1DArray.
        let mut local_deriv_by_bin = vec![0.0f64; 4 * nparams];
        let mut jointpdf_index_1d = vec![0usize; nparams];
        let mut valid = 0usize;

        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];

            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match self.moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue, // maps outside the moving buffer
            };
            if mv < self.moving_true_min || mv > self.moving_true_max {
                continue;
            }
            // The parameter block + local Jacobian of the region owning this
            // sample; skip if the sample falls outside every region.
            let (offset, local_jac) = match transform.local_support_jacobian(fp) {
                Some(oj) => oj,
                None => continue,
            };

            let moving_term = mv / self.moving_bin_size - self.moving_normalized_min;
            let moving_index =
                self.parzen_window_index(mv, self.moving_bin_size, self.moving_normalized_min);
            let fixed_index =
                self.parzen_window_index(fv, self.fixed_bin_size, self.fixed_normalized_min);

            fixed_marginal[fixed_index] += 1.0;

            // inner[mu] = ∇M · (column mu of the LOCAL Jacobian). It does not
            // depend on the parzen bin, so compute it once per sample.
            let mut inner = vec![0.0f64; num_local];
            for (mu, im) in inner.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (j, &g) in grad_phys.iter().enumerate() {
                    acc += local_jac[j * num_local + mu] * g;
                }
                *im = acc;
            }

            // Every parameter of this block lands in the same joint-PDF row/first
            // tap: index (fixed_index, moving_index − 1).
            let moving_start = moving_index - 1;
            let jpi1d = fixed_index * bins + moving_start;
            for mu in 0..num_local {
                jointpdf_index_1d[offset + mu] = jpi1d;
            }

            // Cubic window covers the four bins [moving_index − 1 .. + 2].
            let mut pdf_moving_index = moving_start;
            let mut arg = pdf_moving_index as f64 - moving_term;
            for (b, chunk) in local_deriv_by_bin.chunks_exact_mut(nparams).enumerate() {
                debug_assert!(b < 4);
                joint_pdf[fixed_index * bins + pdf_moving_index] += cubic_bspline(arg);
                let deriv_weight = cubic_bspline_derivative(arg);
                for (mu, &im) in inner.iter().enumerate() {
                    chunk[offset + mu] += im * deriv_weight;
                }
                arg += 1.0;
                pdf_moving_index += 1;
            }

            valid += 1;
        }

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }
        let joint_sum: f64 = joint_pdf.iter().sum();
        if joint_sum < f64::EPSILON {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: valid,
            };
        }

        let n_factor = 1.0 / (self.moving_bin_size * valid as f64);
        let inv_sum = 1.0 / joint_sum;
        for p in joint_pdf.iter_mut() {
            *p *= inv_sum;
        }
        for p in fixed_marginal.iter_mut() {
            *p *= inv_sum;
        }

        let mut moving_marginal = vec![0.0f64; bins];
        for f in 0..bins {
            for (m, mm) in moving_marginal.iter_mut().enumerate() {
                *mm += joint_pdf[f * bins + m];
            }
        }

        // Value pass. Identical to the global path, but instead of folding the
        // dense derivatives it records pRatio·n_factor per bin (m_PRatioArray),
        // to be applied to the compact per-bin local derivatives below.
        let close_to_zero = f64::EPSILON;
        let mut sum = 0.0f64;
        let mut pratio = vec![0.0f64; bins * bins];
        for f in 0..bins {
            let fm = fixed_marginal[f];
            if fm <= close_to_zero {
                continue;
            }
            let log_fm = fm.ln();
            for m in 0..bins {
                let mm = moving_marginal[m];
                let jp = joint_pdf[f * bins + m];
                if mm > close_to_zero && jp > close_to_zero {
                    let p_ratio = (jp / mm).ln();
                    sum += jp * (p_ratio - log_fm);
                    pratio[f * bins + m] = p_ratio * n_factor;
                }
            }
        }

        // Apply the per-bin pRatio to each parameter's owning sample. The sign is
        // + to match this crate's global path (see the `n_factor` sign note in
        // `evaluate_global_support`); ITK subtracts under its opposite convention.
        let mut derivative = vec![0.0f64; nparams];
        for (i, di) in derivative.iter_mut().enumerate() {
            let base = jointpdf_index_1d[i];
            let mut acc = 0.0;
            for b in 0..4 {
                acc += local_deriv_by_bin[b * nparams + i] * pratio[base + b];
            }
            *di = acc;
        }

        MetricValue {
            value: -sum,
            derivative,
            valid_points: valid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::TranslationTransform;

    /// A 2-D Gaussian blob of amplitude `amp` and width `sigma`, centred at
    /// `(cx, cy)` in physical (== index, unit spacing) coordinates, on a small
    /// constant pedestal so the background is not a single degenerate bin.
    fn gaussian(w: usize, h: usize, cx: f64, cy: f64, sigma: f64, amp: f64) -> Image {
        let mut v = vec![0.0f64; w * h];
        let s2 = 2.0 * sigma * sigma;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                v[y * w + x] = amp * (-(dx * dx + dy * dy) / s2).exp();
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn cubic_bspline_partition_of_unity() {
        // The four integer-spaced taps of the cubic window sum to 1 for any
        // fractional offset — the property that makes each sample contribute
        // unit mass to the joint histogram.
        for i in 0..20 {
            let frac = i as f64 / 20.0;
            let sum: f64 = (-1..=2).map(|k| cubic_bspline(k as f64 - frac)).sum();
            assert!((sum - 1.0).abs() < 1e-12, "frac {frac}: sum {sum}");
        }
    }

    #[test]
    fn cubic_bspline_derivative_matches_finite_difference() {
        let h = 1e-6;
        for i in -40..40 {
            let u = i as f64 / 10.0;
            // Skip the knot points where the piecewise derivative is only
            // one-sided.
            if (u - u.round()).abs() < 1e-9 {
                continue;
            }
            let fd = (cubic_bspline(u + h) - cubic_bspline(u - h)) / (2.0 * h);
            let an = cubic_bspline_derivative(u);
            assert!((fd - an).abs() < 1e-4, "u {u}: fd {fd} vs analytic {an}");
        }
    }

    #[test]
    fn constant_image_is_rejected() {
        let flat = Image::from_vec(&[8, 8], vec![3.0; 64]).unwrap();
        let varied = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        assert!(matches!(
            MattesMutualInformationMetric::new(&flat, &varied, 50),
            Err(RegistrationError::ConstantIntensity { which: "fixed" })
        ));
        assert!(matches!(
            MattesMutualInformationMetric::new(&varied, &flat, 50),
            Err(RegistrationError::ConstantIntensity { which: "moving" })
        ));
    }

    #[test]
    fn too_few_bins_is_rejected() {
        let a = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        assert!(matches!(
            MattesMutualInformationMetric::new(&a, &a, 4),
            Err(RegistrationError::TooFewHistogramBins { bins: 4 })
        ));
        // Exactly 2·padding + 1 = 5 is accepted.
        assert!(MattesMutualInformationMetric::new(&a, &a, 5).is_ok());
    }

    #[test]
    fn mi_is_minimized_at_alignment() {
        // −MI is lowest (MI highest) when the images are aligned; a shift away
        // from alignment raises it. Identical images ⇒ perfectly aligned at the
        // identity translation.
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let img = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let metric = MattesMutualInformationMetric::new(&img, &img, 50).unwrap();

        let aligned = metric
            .evaluate(&TranslationTransform::new(vec![0.0, 0.0]))
            .value;
        let shifted = metric
            .evaluate(&TranslationTransform::new(vec![5.0, -4.0]))
            .value;
        assert!(
            aligned < shifted,
            "aligned {aligned} should be below shifted {shifted}"
        );
    }

    #[test]
    fn bspline_derivative_matches_finite_difference() {
        // The existing dense/global-support path produces a correct Mattes
        // derivative for a BSpline transform, which is !HasLocalSupport in ITK
        // (GetTransformCategory() == BSpline, and HasLocalSupport() is true only
        // for a displacement field) and so takes exactly this global path. The
        // (sparse) BSpline Jacobian is folded densely over all parameters;
        // compare the analytic MI derivative to a central finite difference.
        use sitk_transform::{BSplineTransform, ParametricTransform};

        let (w, h, sigma) = (32usize, 32usize, 6.0);
        let fixed = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
        // contrast-inverted moving (multi-modality flavour): 1 − blob.
        let s2 = 2.0 * sigma * sigma;
        let mut mv = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - 16.0, y as f64 - 16.0);
                mv[y * w + x] = 1.0 - (-(dx * dx + dy * dy) / s2).exp();
            }
        }
        let moving = Image::from_vec(&[w, h], mv).unwrap();
        let metric = MattesMutualInformationMetric::new(&fixed, &moving, 32).unwrap();

        let mut base = BSplineTransform::from_image_domain(&fixed, &[4, 4]).unwrap();
        let n = base.number_of_parameters();
        // small non-zero coefficient field so we test off the identity.
        let params: Vec<f64> = (0..n)
            .map(|i| ((i * 31 % 13) as f64 - 6.0) * 0.05)
            .collect();
        base.set_parameters(&params);
        let analytic = metric.evaluate(&base).derivative;

        let step = 1e-3;
        let mut checked = 0;
        for k in 0..n {
            // skip parameters with negligible analytic gradient (support gaps)
            if analytic[k].abs() < 1e-4 {
                continue;
            }
            let mut pp = params.clone();
            pp[k] += step;
            let mut pm = params.clone();
            pm[k] -= step;
            let mut tp = base.clone();
            tp.set_parameters(&pp);
            let mut tm = base.clone();
            tm.set_parameters(&pm);
            let fd = (metric.evaluate(&tp).value - metric.evaluate(&tm).value) / (2.0 * step);
            assert!(
                (fd - analytic[k]).abs() < 5e-3,
                "param {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
            checked += 1;
        }
        assert!(
            checked > 5,
            "expected several non-trivial params, got {checked}"
        );
    }

    #[test]
    fn derivative_matches_finite_difference() {
        // Fixed and moving are the same blob; evaluate at a generic translation
        // (off pixel and bin boundaries) and compare the analytic MI derivative
        // to a central finite difference of the value.
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let moving = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let metric = MattesMutualInformationMetric::new(&fixed, &moving, 50).unwrap();

        let p0 = [1.3f64, -0.7];
        let eval = |p: &[f64]| metric.evaluate(&TranslationTransform::new(p.to_vec()));
        let analytic = eval(&p0).derivative;

        let step = 1e-3;
        for k in 0..2 {
            let mut pp = p0;
            pp[k] += step;
            let mut pm = p0;
            pm[k] -= step;
            let fd = (eval(&pp).value - eval(&pm).value) / (2.0 * step);
            assert!(
                (fd - analytic[k]).abs() < 5e-3,
                "param {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    /// Fixed blob + contrast-inverted moving on a displacement field with a small
    /// non-zero deformation, used by both local-support tests below.
    fn displacement_field_case() -> (
        MattesMutualInformationMetric,
        sitk_transform::DisplacementFieldTransform,
    ) {
        use sitk_transform::{DisplacementFieldTransform, ParametricTransform};
        let (w, h, sigma) = (16usize, 16usize, 4.0);
        let fixed = gaussian(w, h, 8.0, 8.0, sigma, 1.0);
        // contrast-inverted moving (multi-modality flavour): 1 − blob.
        let s2 = 2.0 * sigma * sigma;
        let mut mv = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - 8.0, y as f64 - 8.0);
                mv[y * w + x] = 1.0 - (-(dx * dx + dy * dy) / s2).exp();
            }
        }
        let moving = Image::from_vec(&[w, h], mv).unwrap();
        let metric = MattesMutualInformationMetric::new(&fixed, &moving, 32).unwrap();

        let mut field = DisplacementFieldTransform::from_image_domain(&fixed).unwrap();
        let np = field.number_of_parameters();
        // small non-zero field so we test off the identity.
        let params: Vec<f64> = (0..np)
            .map(|i| ((i * 13 % 11) as f64 - 5.0) * 0.02)
            .collect();
        field.set_parameters(&params);
        (metric, field)
    }

    #[test]
    fn local_support_reproduces_the_global_support_derivative() {
        // The per-pixel local-support accumulation is an exact reorganization of
        // the dense global path: each displacement parameter's contribution is
        // precisely its owning pixel's sample. So for a displacement field the
        // two branches must agree in value AND derivative — the compact memory
        // trio costs nothing in accuracy.
        let (metric, field) = displacement_field_case();
        let local = metric.evaluate_local_support(&field);
        let global = metric.evaluate_global_support(&field);

        assert_eq!(local.valid_points, global.valid_points);
        assert!(
            (local.value - global.value).abs() < 1e-12,
            "value: local {} vs global {}",
            local.value,
            global.value
        );
        assert_eq!(local.derivative.len(), global.derivative.len());
        let max_diff = local
            .derivative
            .iter()
            .zip(&global.derivative)
            .map(|(l, g)| (l - g).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "max derivative diff {max_diff}");
    }

    #[test]
    fn displacement_field_derivative_matches_finite_difference() {
        // Independently confirm the local-support branch (selected by `evaluate`
        // for a displacement field) is the true gradient of the value, via a
        // central finite difference — not merely that it agrees with the global
        // path.
        use sitk_transform::ParametricTransform;
        let (metric, field) = displacement_field_case();
        let params = field.parameters();
        let np = params.len();
        let analytic = metric.evaluate(&field).derivative;

        let step = 1e-3;
        let mut checked = 0;
        for k in 0..np {
            if analytic[k].abs() < 1e-4 {
                continue;
            }
            let mut pp = params.clone();
            pp[k] += step;
            let mut pm = params.clone();
            pm[k] -= step;
            let mut tp = field.clone();
            tp.set_parameters(&pp);
            let mut tm = field.clone();
            tm.set_parameters(&pm);
            let fd = (metric.evaluate(&tp).value - metric.evaluate(&tm).value) / (2.0 * step);
            assert!(
                (fd - analytic[k]).abs() < 5e-3,
                "param {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
            checked += 1;
        }
        assert!(
            checked > 5,
            "expected several non-trivial params, got {checked}"
        );
    }
}
