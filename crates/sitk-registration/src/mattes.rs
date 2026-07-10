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
//! * **Global-support derivative path.** The dense `jointPDFDerivatives`
//!   accumulation is ported, taken by every transform whose Jacobian is
//!   already dense and small (translation, affine, similarity, Euler,
//!   versor) — ITK's `!HasLocalSupport` branch, i.e. every category *except*
//!   `DisplacementField` per `itk::ObjectToObjectMetric::HasLocalSupport`.
//! * **Sparse-support derivative path (covers BSpline and displacement
//!   fields).** A [`BSplineTransform`] reports `GetTransformCategory() ==
//!   BSpline` (`itk::BSplineBaseTransform::GetTransformCategory`,
//!   `itkBSplineBaseTransform.h`), so per ITK's `HasLocalSupport()` — which
//!   checks exactly `GetTransformCategory() == DisplacementField` — it is
//!   **not** local-support, and ITK's own metric threader
//!   (`ImageToImageMetricv4GetValueAndDerivativeThreaderBase::
//!   StorePointDerivativeResult`) folds its Jacobian densely over every
//!   parameter, the same as any other global transform. This crate produces
//!   the identical result — finite-difference verified, and cross-checked
//!   against the dense path to `1e-12` on a shared B-spline problem — through
//!   a different, purely internal computation:
//!   [`MattesMutualInformationMetric::evaluate`] dispatches to a private
//!   `evaluate_sparse_support` for any
//!   transform implementing
//!   [`ParametricTransform::sparse_jacobian_wrt_parameters`] — currently
//!   [`BSplineTransform`] and [`DisplacementFieldTransform`] — which
//!   accumulates the derivative by touching only each sample's affected
//!   parameters, never materializing the `bins² × numberOfParameters` array.
//!   This is a genuinely different algorithm from ITK's `HasLocalSupport`
//!   branch (which fires only for a displacement field, one contiguous
//!   parameter block per sample): a B-spline control point is shared by every
//!   sample whose support region overlaps it, unlike a displacement-field
//!   pixel touched by at most one sample, so the accumulation re-walks the
//!   samples in a second pass once the joint histogram is known, rather than
//!   caching one contributing sample per parameter — see
//!   `evaluate_sparse_support`'s own comments for why that
//!   is necessary, not just a rewrite. A test proves it still reproduces the
//!   (unchanged) dense path's derivative for a displacement field, exactly as
//!   the branch it replaces did.
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

        let fixed_samples = FixedSamples::from_image(fixed)?;
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
    /// accumulated differs. This probes
    /// [`sparse_jacobian_wrt_parameters`](ParametricTransform::sparse_jacobian_wrt_parameters)
    /// on the first fixed sample: if `transform` answers (currently
    /// [`BSplineTransform`] and [`DisplacementFieldTransform`]), every sample
    /// answers, and `evaluate_sparse_support` never materializes the `bins² ×
    /// numberOfParameters` derivative array; otherwise `evaluate_global_support`
    /// folds the dense Jacobian, exactly as ITK's `!HasLocalSupport` branch does.
    /// This is deliberately *not* keyed on
    /// [`has_local_support`](ParametricTransform::has_local_support) — that flag
    /// mirrors ITK's `HasLocalSupport()` (`DisplacementField` category only) and
    /// must keep its ITK-parity meaning for other consumers (e.g. derivative
    /// normalization); the sparse-Jacobian probe is a separate, purely internal
    /// capability signal.
    ///
    /// [`BSplineTransform`]: sitk_transform::BSplineTransform
    /// [`DisplacementFieldTransform`]: sitk_transform::DisplacementFieldTransform
    /// The unnormalized joint histogram, the unnormalized fixed marginal, and
    /// the valid-sample count, from a **value-only** walk of the fixed samples
    /// (no moving gradient, no transform Jacobian).
    ///
    /// This is the first pass of both [`value`](Self::value) and
    /// `evaluate_sparse_support`. `evaluate_global_support` cannot use it: it
    /// fuses the histogram and the `bins² × nparams` joint-PDF derivative into
    /// one walk, and splitting them would cost it a second pass.
    fn build_histogram(&self, transform: &dyn ParametricTransform) -> (Vec<f64>, Vec<f64>, usize) {
        let dim = self.fixed.dim;
        let bins = self.num_bins;
        let n = self.fixed.len();

        let mut joint_pdf = vec![0.0f64; bins * bins];
        let mut fixed_marginal = vec![0.0f64; bins];
        let mut valid = 0usize;

        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];

            let mp = transform.transform_point(fp);
            let mv = match self.moving.value_at(&mp) {
                Some(v) => v,
                None => continue, // maps outside the moving buffer
            };
            if mv < self.moving_true_min || mv > self.moving_true_max {
                continue;
            }

            let moving_term = mv / self.moving_bin_size - self.moving_normalized_min;
            let moving_index =
                self.parzen_window_index(mv, self.moving_bin_size, self.moving_normalized_min);
            let fixed_index =
                self.parzen_window_index(fv, self.fixed_bin_size, self.fixed_normalized_min);
            fixed_marginal[fixed_index] += 1.0;

            let pdf_moving_start = moving_index - 1;
            for pdf_moving_index in pdf_moving_start..pdf_moving_start + 4 {
                let arg = pdf_moving_index as f64 - moving_term;
                joint_pdf[fixed_index * bins + pdf_moving_index] += cubic_bspline(arg);
            }
            valid += 1;
        }

        (joint_pdf, fixed_marginal, valid)
    }

    /// The metric value `−MI` alone at `transform`, for a caller that does not
    /// need the derivative.
    ///
    /// One value-only pass over the samples, then the histogram walk. Neither
    /// the `bins² × nparams` joint-PDF derivative array of the global path nor
    /// the second sample walk of the sparse path is built, so this is the same
    /// for either transform category — there is nothing left to dispatch on.
    pub fn value(&self, transform: &dyn ParametricTransform) -> f64 {
        let bins = self.num_bins;
        let (mut joint_pdf, mut fixed_marginal, valid) = self.build_histogram(transform);
        if valid == 0 {
            return f64::MAX;
        }
        let joint_sum: f64 = joint_pdf.iter().sum();
        if joint_sum < f64::EPSILON {
            return f64::MAX;
        }

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

        let close_to_zero = f64::EPSILON;
        let mut sum = 0.0f64;
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
                    sum += jp * ((jp / mm).ln() - log_fm);
                }
            }
        }
        -sum
    }

    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        let dim = self.fixed.dim;
        let sparse_capable = match self.fixed.points.get(..dim) {
            Some(p0) => transform.sparse_jacobian_wrt_parameters(p0).is_some(),
            None => false,
        };
        if sparse_capable {
            self.evaluate_sparse_support(transform)
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
            let pdf_moving_start = moving_index - 1;
            for pdf_moving_index in pdf_moving_start..pdf_moving_start + 4 {
                let arg = pdf_moving_index as f64 - moving_term;
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

    /// Sparse-support derivative accumulation, taken when `transform`
    /// implements [`sparse_jacobian_wrt_parameters`] (currently
    /// [`BSplineTransform`] and [`DisplacementFieldTransform`]).
    ///
    /// This makes **two passes** over the fixed samples, unlike the
    /// single-pass `evaluate_global_support`:
    ///
    /// 1. Build the joint histogram and fixed marginal exactly as the dense
    ///    path does, using [`MovingImage::value_at`] — a value-only sample
    ///    that skips the gradient this pass does not need.
    /// 2. Once the histogram is final (so every bin's `pRatio` is known),
    ///    re-walk the samples: for each one, pull its sparse Jacobian entries
    ///    — `(parameter_index, column)` pairs, at most `(order+1)^dim` of them
    ///    for a B-spline, exactly `dim` for a displacement-field pixel — and
    ///    scatter-add `∂value/∂p_k` directly into `derivative[k]`.
    ///
    /// A second pass is *necessary*, not just a simpler rewrite of the old
    /// single-pass local-support branch this replaces. That branch exploited
    /// a displacement field's specific geometry: under a grid-aligned virtual
    /// domain, each pixel — and so each sample — owns a disjoint parameter
    /// block, so a parameter's bin location could be cached once per sample
    /// and applied after the histogram finished (`m_JointPdfIndex1DArray` /
    /// `m_LocalDerivativeByParzenBin`). A B-spline control point has no such
    /// owning sample: it is shared by every sample whose `(order+1)^dim`
    /// support region covers it, so a single per-parameter cache slot cannot
    /// hold contributions from multiple overlapping samples. Recomputing each
    /// sample's Jacobian and gradient in a second pass — reading, not
    /// caching, the finished `pRatio` table — handles overlap correctly by
    /// construction: every sample sharing a parameter adds its own
    /// contribution via `derivative[idx] +=`.
    ///
    /// Every sample counts toward the *value* and joint histogram regardless
    /// of whether it lands inside any parameter's support — a B-spline sample
    /// outside the valid region (see
    /// [`BSplineTransform::sparse_jacobian_wrt_parameters`]) still maps
    /// through the transform (as identity there) and still has a moving
    /// intensity; it simply contributes zero derivative, an empty entry list.
    /// Only a sample landing outside the *moving buffer* is skipped from the
    /// value entirely, exactly as in `evaluate_global_support`.
    ///
    /// [`sparse_jacobian_wrt_parameters`]: ParametricTransform::sparse_jacobian_wrt_parameters
    /// [`BSplineTransform`]: sitk_transform::BSplineTransform
    /// [`BSplineTransform::sparse_jacobian_wrt_parameters`]: sitk_transform::BSplineTransform
    /// [`DisplacementFieldTransform`]: sitk_transform::DisplacementFieldTransform
    fn evaluate_sparse_support(&self, transform: &dyn ParametricTransform) -> MetricValue {
        let dim = self.fixed.dim;
        let bins = self.num_bins;
        let nparams = transform.number_of_parameters();
        let n = self.fixed.len();

        // Pass 1: joint histogram + fixed marginal only.
        let (mut joint_pdf, mut fixed_marginal, valid) = self.build_histogram(transform);

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

        // Value pass, identical in form to `evaluate_global_support`, but
        // recording pRatio·n_factor per bin for pass 2 below instead of
        // folding it into a dense derivative array immediately.
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

        // Pass 2: re-walk the samples, scatter-adding each one's sparse
        // Jacobian contribution weighted by the now-finished per-bin pRatio.
        let mut derivative = vec![0.0f64; nparams];
        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];

            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match self.moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue,
            };
            if mv < self.moving_true_min || mv > self.moving_true_max {
                continue;
            }
            let entries = match transform.sparse_jacobian_wrt_parameters(fp) {
                Some(e) if !e.is_empty() => e,
                _ => continue, // outside every parameter's support: zero derivative
            };

            let moving_term = mv / self.moving_bin_size - self.moving_normalized_min;
            let moving_index =
                self.parzen_window_index(mv, self.moving_bin_size, self.moving_normalized_min);
            let fixed_index =
                self.parzen_window_index(fv, self.fixed_bin_size, self.fixed_normalized_min);
            let moving_start = moving_index - 1;

            for (idx, col) in &entries {
                let inner: f64 = col.iter().zip(grad_phys.iter()).map(|(&c, &g)| c * g).sum();

                let mut acc = 0.0;
                for pdf_moving_index in moving_start..moving_start + 4 {
                    let arg = pdf_moving_index as f64 - moving_term;
                    let deriv_weight = cubic_bspline_derivative(arg);
                    acc += deriv_weight * pratio[fixed_index * bins + pdf_moving_index];
                }
                derivative[*idx] += inner * acc;
            }
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
        // `evaluate` routes a BSpline transform through `evaluate_sparse_support`
        // (it implements `sparse_jacobian_wrt_parameters`), which must produce
        // the same MI derivative ITK's dense `!HasLocalSupport` path would —
        // BSpline is `GetTransformCategory() == BSpline`, not `DisplacementField`,
        // so `HasLocalSupport()` is still false for it in the ITK-parity sense;
        // only this crate's *internal* accumulation is sparse. Compare the
        // analytic MI derivative to a central finite difference.
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
    fn sparse_support_reproduces_the_global_support_derivative_for_a_displacement_field() {
        // The sparse-support accumulation must exactly reproduce the dense
        // global path for a displacement field: each pixel's parameter block is
        // touched by exactly that pixel's sample, so the two-pass recomputation
        // and the single-pass dense fold must agree in value AND derivative.
        let (metric, field) = displacement_field_case();
        let sparse = metric.evaluate_sparse_support(&field);
        let global = metric.evaluate_global_support(&field);

        assert_eq!(sparse.valid_points, global.valid_points);
        assert!(
            (sparse.value - global.value).abs() < 1e-12,
            "value: sparse {} vs global {}",
            sparse.value,
            global.value
        );
        assert_eq!(sparse.derivative.len(), global.derivative.len());
        let max_diff = sparse
            .derivative
            .iter()
            .zip(&global.derivative)
            .map(|(l, g)| (l - g).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "max derivative diff {max_diff}");
    }

    #[test]
    fn bspline_sparse_support_matches_the_dense_reference() {
        // Same property as above, for a BSpline transform: even though every
        // control point is shared by many overlapping samples (unlike a
        // displacement field's disjoint per-pixel blocks), the two-pass
        // sparse accumulation must still exactly reproduce the dense fold.
        use sitk_transform::{BSplineTransform, ParametricTransform};

        let (w, h, sigma) = (32usize, 32usize, 6.0);
        let fixed = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
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

        let mut t = BSplineTransform::from_image_domain(&fixed, &[4, 4]).unwrap();
        let n = t.number_of_parameters();
        let params: Vec<f64> = (0..n)
            .map(|i| ((i * 31 % 13) as f64 - 6.0) * 0.05)
            .collect();
        t.set_parameters(&params);

        let sparse = metric.evaluate_sparse_support(&t);
        let dense = metric.evaluate_global_support(&t);

        assert_eq!(sparse.valid_points, dense.valid_points);
        assert!(
            (sparse.value - dense.value).abs() < 1e-12,
            "value: sparse {} vs dense {}",
            sparse.value,
            dense.value
        );
        let max_diff = sparse
            .derivative
            .iter()
            .zip(&dense.derivative)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "max derivative diff {max_diff}");
    }

    #[test]
    fn bspline_sparse_support_counts_samples_outside_its_valid_region() {
        // A BSpline domain narrower than the fixed image leaves some fixed
        // samples outside the transform's valid region — there
        // `sparse_jacobian_wrt_parameters` returns `Some(empty)`, contributing
        // zero derivative, but the sample must still count toward the value and
        // joint histogram exactly as the dense path counts it (it still maps
        // through the transform, as identity there, and still has a moving
        // intensity). This is the case the old cache-based local-support branch
        // got wrong for a non-displacement-field transform: it treated "no
        // owning parameter block" as "drop the sample" instead of "drop only
        // its derivative contribution".
        use sitk_transform::{BSplineTransform, ParametricTransform, TransformBase};

        let (w, h, sigma) = (32usize, 32usize, 5.0);
        let fixed = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
        let moving = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
        let metric = MattesMutualInformationMetric::new(&fixed, &moving, 32).unwrap();

        // Domain covers only [8, 24) of the 32-pixel image.
        let mut t = BSplineTransform::new(
            2,
            &[8.0, 8.0],
            &[16.0, 16.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[4, 4],
        )
        .unwrap();
        let n = t.number_of_parameters();
        let params: Vec<f64> = (0..n)
            .map(|i| ((i * 31 % 13) as f64 - 6.0) * 0.02)
            .collect();
        t.set_parameters(&params);

        // Sanity-check the setup actually exercises the out-of-region path.
        assert_eq!(
            t.transform_point(&[0.0, 0.0]),
            vec![0.0, 0.0],
            "corner should be outside the domain (identity)"
        );
        assert_ne!(
            t.transform_point(&[16.0, 16.0]),
            vec![16.0, 16.0],
            "center should be inside the domain (deformed)"
        );

        let sparse = metric.evaluate_sparse_support(&t);
        let dense = metric.evaluate_global_support(&t);

        assert_eq!(
            sparse.valid_points, dense.valid_points,
            "sparse path must count every sample the dense path counts, \
             even outside the BSpline's valid region"
        );
        assert!(
            (sparse.value - dense.value).abs() < 1e-12,
            "value: sparse {} vs dense {}",
            sparse.value,
            dense.value
        );
        let max_diff = sparse
            .derivative
            .iter()
            .zip(&dense.derivative)
            .map(|(a, b)| (a - b).abs())
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

    #[test]
    fn value_agrees_with_evaluate_on_the_global_support_path() {
        let fixed = gaussian(20, 20, 10.0, 10.0, 4.0, 1.0);
        let moving = gaussian(20, 20, 11.5, 9.0, 4.0, 1.0);
        let metric = MattesMutualInformationMetric::new(&fixed, &moving, 32).unwrap();
        for t in [[0.0, 0.0], [1.3, -0.7], [-2.5, 2.5]] {
            let transform = TranslationTransform::new(t.to_vec());
            let full = metric.evaluate(&transform).value;
            let value_only = metric.value(&transform);
            assert!(
                (full - value_only).abs() <= 1e-12 * full.abs().max(1.0),
                "at {t:?}: evaluate {full} vs value {value_only}"
            );
        }
    }

    #[test]
    fn value_agrees_with_evaluate_on_the_sparse_support_path() {
        // `value` has no sparse/global dispatch — it never builds a derivative,
        // so there is nothing to dispatch on. It must still reproduce the value
        // `evaluate_sparse_support` computes for a displacement field.
        use sitk_transform::ParametricTransform;

        let (metric, mut field) = displacement_field_case();
        assert!(field.has_local_support());
        let mut params = field.parameters();
        for (i, p) in params.iter_mut().enumerate() {
            *p = if i % 2 == 0 { 0.4 } else { -0.3 };
        }
        field.set_parameters(&params);

        let full = metric.evaluate(&field).value;
        let value_only = metric.value(&field);
        assert!(
            (full - value_only).abs() <= 1e-12 * full.abs().max(1.0),
            "evaluate {full} vs value {value_only}"
        );
    }
}
