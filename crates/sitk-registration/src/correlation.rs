//! Normalized cross-correlation image-to-image metric
//! (`itk::CorrelationImageToImageMetricv4`).
//!
//! This is a **same-modality** metric: it is invariant to an (unknown) affine
//! intensity map between the fixed and moving image — brightness/contrast
//! differences that mean squares is sensitive to but normalized cross
//! correlation (NCC) is not. ITK's class implements the *square* of NCC,
//! negated so smaller is better:
//!
//! ```text
//! f1ᵢ = fᵢ − f̄,   m1ᵢ = mᵢ(p) − m̄(p)          (f̄, m̄: sample means)
//! sff = Σ f1ᵢ²,   smm = Σ m1ᵢ²,   sfm = Σ f1ᵢ·m1ᵢ
//!
//! value(p) = − sfm² / (sff · smm)
//! ```
//!
//! over the fixed-image sample points that map, under transform `T`, inside
//! the moving image. `value` ranges over `[−1, 0]`: `−1` is perfect
//! correlation (up to sign), `0` is uncorrelated. Although the moving mean
//! `m̄(p)` depends on `p` through every sample, its contribution to
//! `d(value)/dp` vanishes exactly, because `Σ f1ᵢ = Σ m1ᵢ = 0` by construction
//! of a mean-subtracted quantity. So the derivative below is the metric's
//! *exact* total derivative, not an approximation that freezes the mean at the
//! value it had when the means were last recomputed:
//!
//! ```text
//! ∂value/∂pₖ = −2 · sfm/(sff·smm) · ( fdmₖ − sfm/smm · mdmₖ )
//!
//! fdmₖ = Σ f1ᵢ · (∇M(T(xᵢ)) · Jₖ(xᵢ)),   mdmₖ = Σ m1ᵢ · (∇M(T(xᵢ)) · Jₖ(xᵢ))
//! ```
//!
//! where `J` is the transform Jacobian
//! ([`ParametricTransform::jacobian_wrt_parameters`]).
//!
//! ## Two passes, exactly as ITK
//!
//! `f̄`/`m̄` must be known before any `f1`/`m1` term can be formed, so ITK
//! splits the computation across two threaders, and this module mirrors that
//! with two loops inside [`evaluate`](CorrelationMetric::evaluate): the first
//! (`CorrelationImageToImageMetricv4HelperThreader`) accumulates the sums that
//! give the sample means; the second
//! (`CorrelationImageToImageMetricv4GetValueAndDerivativeThreader`)
//! accumulates `sff`/`smm`/`sfm`/`fdm`/`mdm` and combines them into `value`
//! and the derivative above.
//!
//! ## Derivative-sign convention vs ITK
//!
//! ITK's header (`itkCorrelationImageToImageMetricv4.h`) documents that its
//! `GetValueAndDerivative` omits the minus sign the *true* calculus derivative
//! of `value` mathematically has, "to match the requirement of the metricv4
//! optimization framework": the threader's `AfterThreadedExecution` (the
//! `.hxx`) literally computes `+2·sfm/(sff·smm)·(fdm − sfm/smm·mdm)`, i.e.
//! **`−∂value/∂p`** — the steepest *descent* direction, because ITK's v4
//! optimizers *add* the returned derivative. This crate's optimizers
//! *subtract* (`p −= lr·derivative`), so every metric here stores
//! `+∂value/∂p`, the true gradient — the same convention flip
//! [`MattesMutualInformationMetric`](crate::mattes::MattesMutualInformationMetric)
//! documents for its `nFactor` and mean squares documents for differencing
//! `M − F` where ITK differences `F − M`. Concretely, this module's derivative
//! is the *negation* of the literal ITK code's
//! `fc · sfm/(sff·smm) · (fdm − sfm/smm·mdm)` (`fc = 2.0` there; effectively
//! `fc = −2.0` here). The finite-difference test below pins this down:
//! `derivative == d(value)/d(param)`.
//!
//! ## No local-support branch
//!
//! Unlike [`MattesMutualInformationMetric`](crate::mattes::MattesMutualInformationMetric),
//! ITK's `CorrelationImageToImageMetricv4` constructor unconditionally throws
//! when the moving transform has local support (a displacement field): its
//! class doc states *"This metric only works with the global transform. It
//! throws an exception if the transform has local support,"* and the
//! constructor literally checks `GetTransformCategory() ==
//! TransformCategoryEnum::DisplacementField` and throws. There is no
//! local-support threader for this metric in ITK to port, so
//! [`evaluate`](CorrelationMetric::evaluate) mirrors the constructor check
//! instead of silently computing a wrong per-pixel derivative: it panics if
//! [`transform.has_local_support()`](ParametricTransform::has_local_support)
//! is true.

use sitk_core::Image;
use sitk_transform::ParametricTransform;

use crate::error::{RegistrationError, Result};
use crate::metric::{FixedSamples, MetricValue, MovingImage};
use crate::scales::{ScalesEstimator, ScalesEstimatorKind};

/// The normalized cross-correlation metric. Holds the precomputed fixed
/// samples and moving image; [`evaluate`](Self::evaluate) returns
/// `value = −(normalized cross correlation)²` plus its parameter-derivative
/// for a given transform.
pub struct CorrelationMetric {
    fixed: FixedSamples,
    moving: MovingImage,
}

impl CorrelationMetric {
    /// Build the metric from a fixed and moving image. Fails if dimensions
    /// disagree or the moving direction matrix is singular.
    pub fn new(fixed: &Image, moving: &Image) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        Self::from_samples(
            FixedSamples::from_image(fixed)?,
            MovingImage::from_image(moving)?,
        )
    }

    /// Build the metric directly from a pre-built fixed sample set and moving
    /// image — the driver's entry point once it has applied whatever
    /// sampling strategy, interpolator, and mask configure `FixedSamples` and
    /// `MovingImage`. [`new`](Self::new) is the raw-`Image` convenience
    /// wrapper and delegates here.
    pub fn from_samples(fixed: FixedSamples, moving: MovingImage) -> Result<Self> {
        Ok(Self { fixed, moving })
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// Build a scale/learning-rate estimator of `kind` for `transform` over
    /// this metric's virtual domain (shared with the other metrics).
    pub fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        self.fixed.scales_estimator(transform, &self.moving, kind)
    }

    /// This metric's transform-category precondition: it is global-transform
    /// only. Mirrors `itkCorrelationImageToImageMetricv4.hxx:43-46`, whose
    /// constructor throws `"does not support displacement field transforms!!"`.
    ///
    /// The single place the precondition is stated. Call it before optimizing;
    /// [`evaluate`](Self::evaluate) asserts it.
    pub fn check_transform(&self, transform: &dyn ParametricTransform) -> Result<()> {
        if transform.has_local_support() {
            Err(RegistrationError::RequiresGlobalTransform {
                metric: "Correlation",
            })
        } else {
            Ok(())
        }
    }

    /// Evaluate `value = −sfm²/(sff·smm)` and its parameter-derivative for
    /// `transform`, over the fixed samples that map inside the moving image
    /// under `transform`. See the [module docs](self) for the exact formulas
    /// and the derivative-sign convention.
    ///
    /// # Panics
    ///
    /// Panics if [`check_transform`](Self::check_transform) rejects
    /// `transform` — that is, if it has
    /// [local support](ParametricTransform::has_local_support) (e.g. a
    /// [`DisplacementFieldTransform`](sitk_transform::DisplacementFieldTransform)).
    /// This metric is global-transform-only — see the
    /// [module docs](self#no-local-support-branch).
    /// The metric value alone at `transform`, for a caller that does not need
    /// the derivative. Runs the same two passes as
    /// [`evaluate`](Self::evaluate), but reads the moving image value-only and
    /// never forms a transform Jacobian, so it costs no `O(nsamples · nparams)`
    /// accumulation.
    ///
    /// # Panics
    ///
    /// As [`evaluate`](Self::evaluate): this metric is global-transform-only.
    pub fn value(&self, transform: &dyn ParametricTransform) -> f64 {
        assert!(
            self.check_transform(transform).is_ok(),
            "CorrelationMetric is global-transform-only; call check_transform first"
        );

        let (avg_fix, avg_mov) = match self.means(transform) {
            Some(m) => m,
            None => return f64::MAX,
        };

        let dim = self.fixed.dim;
        let mut fm = 0.0f64;
        let mut f2 = 0.0f64;
        let mut m2 = 0.0f64;
        let mut valid_points = 0usize;
        for s in 0..self.fixed.len() {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let mp = transform.transform_point(fp);
            let mv = match self.moving.value_at(&mp) {
                Some(v) => v,
                None => continue,
            };
            let f1 = self.fixed.values[s] - avg_fix;
            let m1 = mv - avg_mov;
            f2 += f1 * f1;
            m2 += m1 * m1;
            fm += f1 * m1;
            valid_points += 1;
        }

        if valid_points == 0 {
            return f64::MAX;
        }
        let m2f2 = m2 * f2;
        if m2f2 <= f64::EPSILON {
            return f64::MAX;
        }
        -fm * fm / m2f2
    }

    /// Pass 1 (`CorrelationImageToImageMetricv4HelperThreader`): the fixed and
    /// moving sample means over the valid point set. `None` when no sample maps
    /// inside the moving image.
    fn means(&self, transform: &dyn ParametricTransform) -> Option<(f64, f64)> {
        let dim = self.fixed.dim;
        let mut fix_sum = 0.0f64;
        let mut mov_sum = 0.0f64;
        let mut valid = 0usize;
        for s in 0..self.fixed.len() {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let mp = transform.transform_point(fp);
            let mv = match self.moving.value_at(&mp) {
                Some(v) => v,
                None => continue, // maps outside the moving buffer
            };
            fix_sum += self.fixed.values[s];
            mov_sum += mv;
            valid += 1;
        }
        if valid == 0 {
            return None;
        }
        Some((fix_sum / valid as f64, mov_sum / valid as f64))
    }

    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        assert!(
            self.check_transform(transform).is_ok(),
            "CorrelationMetric is global-transform-only; call check_transform first"
        );

        let dim = self.fixed.dim;
        let nparams = transform.number_of_parameters();
        let n = self.fixed.len();

        let (avg_fix, avg_mov) = match self.means(transform) {
            Some(m) => m,
            None => {
                return MetricValue {
                    value: f64::MAX,
                    derivative: vec![0.0; nparams],
                    valid_points: 0,
                };
            }
        };

        // Pass 2 (CorrelationImageToImageMetricv4GetValueAndDerivativeThreader):
        // sff/smm/sfm and the derivative accumulators fdm/mdm.
        let mut fm = 0.0f64;
        let mut f2 = 0.0f64;
        let mut m2 = 0.0f64;
        let mut fdm = vec![0.0f64; nparams];
        let mut mdm = vec![0.0f64; nparams];
        let mut valid_points = 0usize;

        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];
            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match self.moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue, // maps outside the moving buffer
            };

            let f1 = fv - avg_fix;
            let m1 = mv - avg_mov;
            f2 += f1 * f1;
            m2 += m1 * m1;
            fm += f1 * m1;

            let jac = transform.jacobian_wrt_parameters(fp);
            for (k, (fdmk, mdmk)) in fdm.iter_mut().zip(mdm.iter_mut()).enumerate() {
                // inner = ∇M · (column k of the transform Jacobian).
                let mut inner = 0.0;
                for (d, &g) in grad_phys.iter().enumerate() {
                    inner += g * jac[d * nparams + k];
                }
                *fdmk += f1 * inner;
                *mdmk += m1 * inner;
            }

            valid_points += 1;
        }

        if valid_points == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }

        let m2f2 = m2 * f2;
        if m2f2 <= f64::EPSILON {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points,
            };
        }

        let value = -fm * fm / m2f2;

        // Crate convention: +∂value/∂p (see the module docs' sign-convention
        // section). The literal ITK code computes
        // `+2·fm/(f2·m2)·(fdm − fm/m2·mdm)`; this is its negation.
        let mut derivative = vec![0.0f64; nparams];
        for (k, dk) in derivative.iter_mut().enumerate() {
            *dk = -2.0 * fm / (f2 * m2) * (fdm[k] - fm / m2 * mdm[k]);
        }

        MetricValue {
            value,
            derivative,
            valid_points,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::TranslationTransform;

    /// A 2-D Gaussian blob of amplitude `amp` and width `sigma`, centred at
    /// `(cx, cy)` in physical (== index, unit spacing) coordinates, on a small
    /// constant pedestal so the background is not perfectly flat.
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
    fn identical_images_at_identity_give_negative_one_with_zero_gradient() {
        let (w, h, sigma) = (32usize, 32usize, 5.0);
        let img = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
        let metric = CorrelationMetric::new(&img, &img).unwrap();

        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let r = metric.evaluate(&t);

        assert!((r.value - (-1.0)).abs() < 1e-9, "value {}", r.value);
        assert!(r.derivative[0].abs() < 1e-6, "d/dtx {}", r.derivative[0]);
        assert!(r.derivative[1].abs() < 1e-6, "d/dty {}", r.derivative[1]);
        assert_eq!(r.valid_points, w * h);
    }

    #[test]
    fn affine_intensity_rescale_of_moving_gives_the_same_value() {
        // The property mean squares lacks: NCC (squared) is invariant to an
        // affine intensity map a·I + b applied to one image, even away from
        // perfect alignment.
        let (w, h, sigma) = (32usize, 32usize, 5.0);
        let fixed = gaussian(w, h, 14.0, 16.0, sigma, 1.0);
        let moving = gaussian(w, h, 17.0, 15.0, sigma, 1.0);
        let rescaled: Vec<f64> = moving
            .to_f64_vec()
            .unwrap()
            .iter()
            .map(|v| 2.5 * v + 7.0)
            .collect();
        let rescaled_moving = Image::from_vec(&[w, h], rescaled).unwrap();

        let metric_orig = CorrelationMetric::new(&fixed, &moving).unwrap();
        let metric_rescaled = CorrelationMetric::new(&fixed, &rescaled_moving).unwrap();

        // A generic (non-aligning) offset, so this isn't just the trivial
        // "perfect correlation either way" case at identity.
        let t = TranslationTransform::new(vec![1.3, -0.7]);
        let orig = metric_orig.evaluate(&t).value;
        let rescaled_value = metric_rescaled.evaluate(&t).value;

        assert!(
            (orig - rescaled_value).abs() < 1e-9,
            "orig {orig} vs rescaled {rescaled_value}"
        );
    }

    #[test]
    fn derivative_matches_finite_difference() {
        // Fixed and moving are the same blob; evaluate at a generic
        // translation (off pixel and off any half-integer, so no sample sits
        // on the is_inside boundary and flips validity under ±h) and compare
        // the analytic derivative to a central finite difference of the
        // value.
        let (w, h, sigma) = (32usize, 32usize, 5.0);
        let fixed = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
        let moving = gaussian(w, h, 16.0, 16.0, sigma, 1.0);
        let metric = CorrelationMetric::new(&fixed, &moving).unwrap();

        let p0 = [1.3f64, -0.7];
        let eval = |p: &[f64]| metric.evaluate(&TranslationTransform::new(p.to_vec()));
        let analytic = eval(&p0).derivative;

        let step = 1e-4;
        for k in 0..2 {
            let mut pp = p0;
            pp[k] += step;
            let mut pm = p0;
            pm[k] -= step;
            let fd = (eval(&pp).value - eval(&pm).value) / (2.0 * step);
            assert!(
                (fd - analytic[k]).abs() < 1e-3,
                "param {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    #[test]
    fn gradient_descent_recovers_a_translated_blob() {
        use crate::optimizer::GradientDescentOptimizer;

        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        // Moving blob offset by (+3, -3) relative to fixed.
        let moving = gaussian(w, h, 23.0, 17.0, sigma, 1.0);
        let metric = CorrelationMetric::new(&fixed, &moving).unwrap();

        let initial = vec![0.0f64, 0.0];
        let t0 = TranslationTransform::new(initial.clone());
        let scales_est = metric.scales_estimator(&t0, ScalesEstimatorKind::default());
        let m0 = metric.evaluate(&t0);
        // Estimate the learning rate once from the initial gradient (ITK's
        // EstimateLearningRate::Once), then hold it fixed.
        let lr = scales_est.estimate_learning_rate(&m0.derivative);

        let opt = GradientDescentOptimizer::new(lr, 300);
        let result = opt.optimize(initial, |p| {
            let t = TranslationTransform::new(p.to_vec());
            let mv = metric.evaluate(&t);
            (mv.value, mv.derivative)
        });

        assert!(
            (result.parameters[0] - 3.0).abs() < 0.05,
            "{:?}",
            result.parameters
        );
        assert!(
            (result.parameters[1] + 3.0).abs() < 0.05,
            "{:?}",
            result.parameters
        );
    }

    #[test]
    #[should_panic(expected = "global-transform-only")]
    fn displacement_field_transform_panics() {
        use sitk_transform::DisplacementFieldTransform;

        let (w, h, sigma) = (16usize, 16usize, 4.0);
        let img = gaussian(w, h, 8.0, 8.0, sigma, 1.0);
        let metric = CorrelationMetric::new(&img, &img).unwrap();
        let field = DisplacementFieldTransform::from_image_domain(&img).unwrap();

        let _ = metric.evaluate(&field);
    }

    #[test]
    fn check_transform_rejects_a_displacement_field_before_evaluating() {
        use sitk_transform::{DisplacementFieldTransform, TranslationTransform};

        let (w, h, sigma) = (16usize, 16usize, 4.0);
        let img = gaussian(w, h, 8.0, 8.0, sigma, 1.0);
        let metric = CorrelationMetric::new(&img, &img).unwrap();

        let field = DisplacementFieldTransform::from_image_domain(&img).unwrap();
        let err = metric.check_transform(&field).unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::RequiresGlobalTransform {
                    metric: "Correlation"
                }
            ),
            "unexpected error {err:?}"
        );

        let global = TranslationTransform::new(vec![0.0, 0.0]);
        assert!(metric.check_transform(&global).is_ok());
    }

    #[test]
    fn value_agrees_with_evaluate() {
        let fixed = gaussian(20, 20, 10.0, 10.0, 4.0, 1.0);
        let moving = gaussian(20, 20, 11.5, 9.0, 4.0, 1.0);
        let metric = CorrelationMetric::new(&fixed, &moving).unwrap();
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
}
