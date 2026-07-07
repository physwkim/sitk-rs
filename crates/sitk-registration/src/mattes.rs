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
//!   interpolant* ([`MovingImage::value_and_physical_gradient`]), so the metric
//!   derivative is the true gradient of the interpolated MI value (an
//!   optimizer's finite difference of the value reproduces it). This is the same
//!   deliberate deviation the mean-squares metric documents: ITK defaults to a
//!   separately-computed (Gaussian-smoothed or central-difference) gradient
//!   image that is not consistent with the interpolated value.
//! * **Global transforms only.** The dense global-support derivative path is
//!   ported; ITK's local-support (displacement-field / BSpline) branch is not,
//!   since this crate's transforms are all global. When a deformable transform
//!   is added, the `HasLocalSupport` accumulation joins here.

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
    /// Two passes over the fixed samples' contributions, exactly as ITK: the
    /// first accumulates the joint histogram, the fixed marginal, and the
    /// per-bin joint-PDF parameter derivatives; the second walks the histogram
    /// to form `−MI` and folds each bin's `pRatio` into the derivative.
    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
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
}
