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
//! ## Parallelism, and why the numbers do not move
//!
//! All three sample loops — [`means`](CorrelationMetric::means) (pass 1),
//! [`evaluate`](CorrelationMetric::evaluate) (pass 2) and
//! [`value`](CorrelationMetric::value) — run on every core and return **the same
//! bits they returned when they were serial**, at any thread count. They get that
//! by construction, from [`sitk_core::parallel::map_rows_fold_in_order`], the
//! same primitive [`crate::metric`]'s mean-squares backend uses: the expensive
//! per-sample work (transform, interpolation and gradient, Jacobian) never
//! touches an accumulator, so it runs in parallel; the accumulators are then fed
//! the per-sample contributions on a single thread, in sample order, executing
//! the identical sequence of additions the serial loop did.
//!
//! Nothing is re-associated. That is not a nicety here — float `+` is not
//! associative, so per-thread partials would re-round every sum, and the CPU
//! path's numbers are what the device pins in `tests/cuda_correlation.rs` are
//! compared against. A re-association would move those bands, and it would also
//! walk the optimizer to a different registration result, because the optimizer
//! is a feedback loop and one ulp of drift compounds.
//! `correlation_is_bit_identical_at_every_thread_count` pins it.
//!
//! There is no sparse branch to leave serial, unlike mean squares:
//! [`check_transform`](CorrelationMetric::check_transform) already refuses
//! local-support transforms, so a dense staged row is never the wrong shape.
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
use sitk_core::parallel;
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

        let mut fm = 0.0f64;
        let mut f2 = 0.0f64;
        let mut m2 = 0.0f64;
        let mut valid_points = 0usize;

        // Parallel, and bit-identical to the serial loop it replaces: each
        // sample's `f1²`/`m1²`/`f1·m1` is formed in parallel from that sample
        // alone, and the three accumulators are then fed those rows on one
        // thread in sample order — the identical sequence of `+=`. See the
        // module docs' parallelism section.
        let (fixed, moving) = (&self.fixed, &self.moving);
        parallel::map_rows_fold_in_order(
            fixed.len(),
            3,
            || fixed.scratch(),
            |scratch, s, row| {
                let fp = fixed.point(s, scratch);
                let mp = transform.transform_point(fp);
                let Some(mv) = moving.value_at(&mp) else {
                    return false;
                };
                let f1 = fixed.value(s) - avg_fix;
                let m1 = mv - avg_mov;
                row[0] = f1 * f1;
                row[1] = m1 * m1;
                row[2] = f1 * m1;
                true
            },
            |_, row| {
                f2 += row[0];
                m2 += row[1];
                fm += row[2];
                valid_points += 1;
            },
        );

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
        let mut fix_sum = 0.0f64;
        let mut mov_sum = 0.0f64;
        let mut valid = 0usize;

        // Parallel, bit-identical to the serial loop (see the module docs): the
        // transform and interpolation run per sample with no accumulator in
        // reach, and `fix_sum`/`mov_sum` are then fed sample by sample, in
        // order, on one thread.
        let (fixed, moving) = (&self.fixed, &self.moving);
        parallel::map_rows_fold_in_order(
            fixed.len(),
            2,
            || fixed.scratch(),
            |scratch, s, row| {
                let fp = fixed.point(s, scratch);
                let mp = transform.transform_point(fp);
                let Some(mv) = moving.value_at(&mp) else {
                    return false; // maps outside the moving buffer
                };
                row[0] = fixed.value(s);
                row[1] = mv;
                true
            },
            |_, row| {
                fix_sum += row[0];
                mov_sum += row[1];
                valid += 1;
            },
        );

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

        // Parallel, and bit-identical to the serial loop it replaces. Every
        // sample's expensive half — the transform, the interpolation and its
        // gradient, the Jacobian, and each parameter's `f1·inner` / `m1·inner`
        // product — is computed from sample `s` alone into that sample's own
        // row, touching no accumulator. The accumulators are then fed those rows
        // on one thread, for s = 0, 1, … n−1, executing the identical sequence
        // of `+=`. Nothing is re-associated, so the value and every derivative
        // component keep their exact bits at any thread count — which is what
        // the device pins in `tests/cuda_correlation.rs` compare against.
        //
        // Row layout, width `3 + 2·nparams`:
        //   [ f1·m1, f1², m1², fdm₀ … fdm_{k−1}, mdm₀ … mdm_{k−1} ]
        let (fixed, moving) = (&self.fixed, &self.moving);
        parallel::map_rows_fold_in_order(
            n,
            3 + 2 * nparams,
            || fixed.scratch(),
            |scratch, s, row| {
                let fp = fixed.point(s, scratch);
                let fv = fixed.value(s);
                let mp = transform.transform_point(fp);
                let Some((mv, grad_phys)) = moving.value_and_physical_gradient(&mp) else {
                    return false; // maps outside the moving buffer
                };

                let f1 = fv - avg_fix;
                let m1 = mv - avg_mov;
                row[0] = f1 * m1;
                row[1] = f1 * f1;
                row[2] = m1 * m1;

                let jac = transform.jacobian_wrt_parameters(fp);
                let (_, deriv_row) = row.split_at_mut(3);
                let (fdm_row, mdm_row) = deriv_row.split_at_mut(nparams);
                for (k, (fdmk, mdmk)) in fdm_row.iter_mut().zip(mdm_row.iter_mut()).enumerate() {
                    // inner = ∇M · (column k of the transform Jacobian).
                    let mut inner = 0.0;
                    for (d, &g) in grad_phys.iter().enumerate() {
                        inner += g * jac[d * nparams + k];
                    }
                    *fdmk = f1 * inner;
                    *mdmk = m1 * inner;
                }
                true
            },
            |_, row| {
                fm += row[0];
                f2 += row[1];
                m2 += row[2];
                for (k, (fdmk, mdmk)) in fdm.iter_mut().zip(mdm.iter_mut()).enumerate() {
                    *fdmk += row[3 + k];
                    *mdmk += row[3 + nparams + k];
                }
                valid_points += 1;
            },
        );

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
    use sitk_transform::{TransformBase, TranslationTransform};

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

    /// The serial reference: the exact loops this module ran before
    /// [`sitk_core::parallel::map_rows_fold_in_order`] replaced them — two passes,
    /// one accumulator sequence, no parallelism anywhere. Lives here so the
    /// bit-identity claim is checked against the *original* sequence of `+=` and
    /// not against another copy of the parallel code.
    fn evaluate_serial(
        metric: &CorrelationMetric,
        transform: &dyn ParametricTransform,
    ) -> MetricValue {
        let nparams = transform.number_of_parameters();
        let (fixed, moving) = (&metric.fixed, &metric.moving);

        let mut fix_sum = 0.0f64;
        let mut mov_sum = 0.0f64;
        let mut valid = 0usize;
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            let mp = transform.transform_point(fp);
            let Some(mv) = moving.value_at(&mp) else {
                continue;
            };
            fix_sum += fixed.value(s);
            mov_sum += mv;
            valid += 1;
        }
        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }
        let (avg_fix, avg_mov) = (fix_sum / valid as f64, mov_sum / valid as f64);

        let mut fm = 0.0f64;
        let mut f2 = 0.0f64;
        let mut m2 = 0.0f64;
        let mut fdm = vec![0.0f64; nparams];
        let mut mdm = vec![0.0f64; nparams];
        let mut valid_points = 0usize;
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            let fv = fixed.value(s);
            let mp = transform.transform_point(fp);
            let Some((mv, grad_phys)) = moving.value_and_physical_gradient(&mp) else {
                continue;
            };
            let f1 = fv - avg_fix;
            let m1 = mv - avg_mov;
            f2 += f1 * f1;
            m2 += m1 * m1;
            fm += f1 * m1;

            let jac = transform.jacobian_wrt_parameters(fp);
            for (k, (fdmk, mdmk)) in fdm.iter_mut().zip(mdm.iter_mut()).enumerate() {
                let mut inner = 0.0;
                for (d, &g) in grad_phys.iter().enumerate() {
                    inner += g * jac[d * nparams + k];
                }
                *fdmk += f1 * inner;
                *mdmk += m1 * inner;
            }
            valid_points += 1;
        }

        let m2f2 = m2 * f2;
        if valid_points == 0 || m2f2 <= f64::EPSILON {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points,
            };
        }
        let mut derivative = vec![0.0f64; nparams];
        for (k, dk) in derivative.iter_mut().enumerate() {
            *dk = -2.0 * fm / (f2 * m2) * (fdm[k] - fm / m2 * mdm[k]);
        }
        MetricValue {
            value: -fm * fm / m2f2,
            derivative,
            valid_points,
        }
    }

    /// The serial reference for the **value-only** pass. It needs its own, because
    /// [`CorrelationMetric::value`] reads the moving image through `value_at`
    /// while [`CorrelationMetric::evaluate`] reads it through
    /// `value_and_physical_gradient` — two different interpolator entry points,
    /// which the existing `value_agrees_with_evaluate` only pins to 1e-12, not to
    /// the bit. Pinning `value()` against `evaluate()`'s reference would therefore
    /// be testing the interpolator, not the fold.
    fn value_serial(metric: &CorrelationMetric, transform: &dyn ParametricTransform) -> f64 {
        let (fixed, moving) = (&metric.fixed, &metric.moving);

        let mut fix_sum = 0.0f64;
        let mut mov_sum = 0.0f64;
        let mut valid = 0usize;
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            let mp = transform.transform_point(fp);
            let Some(mv) = moving.value_at(&mp) else {
                continue;
            };
            fix_sum += fixed.value(s);
            mov_sum += mv;
            valid += 1;
        }
        if valid == 0 {
            return f64::MAX;
        }
        let (avg_fix, avg_mov) = (fix_sum / valid as f64, mov_sum / valid as f64);

        let mut fm = 0.0f64;
        let mut f2 = 0.0f64;
        let mut m2 = 0.0f64;
        let mut valid_points = 0usize;
        let mut scratch = fixed.scratch();
        for s in 0..fixed.len() {
            let fp = fixed.point(s, &mut scratch);
            let mp = transform.transform_point(fp);
            let Some(mv) = moving.value_at(&mp) else {
                continue;
            };
            let f1 = fixed.value(s) - avg_fix;
            let m1 = mv - avg_mov;
            f2 += f1 * f1;
            m2 += m1 * m1;
            fm += f1 * m1;
            valid_points += 1;
        }
        let m2f2 = m2 * f2;
        if valid_points == 0 || m2f2 <= f64::EPSILON {
            return f64::MAX;
        }
        -fm * fm / m2f2
    }

    /// **The pin.** Every number this metric produces must be bit-identical to the
    /// serial fold above, at every thread count — not close, identical. The device
    /// tests (`tests/cuda_correlation.rs`) band their GPU results against these
    /// exact CPU numbers, and the optimizer is a feedback loop in which one ulp
    /// compounds, so a re-associated sum is a broken metric, not a rounding
    /// detail.
    ///
    /// Non-vacuous by size: `map_rows_fold_in_order` runs the serial fast path
    /// below its own element threshold, so a small image would pin nothing. The
    /// 160×160 grid below is 25 600 samples, above that threshold — asserted, so
    /// the test fails loudly rather than silently degrading if the threshold moves.
    #[test]
    fn correlation_is_bit_identical_at_every_thread_count() {
        let (w, h) = (160usize, 160usize);
        let fixed = gaussian(w, h, 78.0, 82.0, 20.0, 1.0);
        let moving = gaussian(w, h, 83.0, 77.0, 20.0, 1.3);
        let metric = CorrelationMetric::new(&fixed, &moving).unwrap();
        assert!(
            metric.sample_count() > 1 << 14,
            "{} samples: at or below `parallel`'s serial threshold, so this test \
             would pin the serial path against itself",
            metric.sample_count()
        );

        // A generic transform: off-lattice, so interpolation is exercised, and
        // asymmetric, so a re-association of any one accumulator shows up.
        let t = TranslationTransform::new(vec![2.3, -1.7]);
        let want = evaluate_serial(&metric, &t);
        let want_value_only = value_serial(&metric, &t);

        // The pin has teeth only if the fold order is *observable* on this input:
        // if summing the same per-sample contributions in the reverse order gave
        // the same bits, then every assertion below would pass even against a
        // re-associating implementation, and this test would be decoration. Sum
        // one accumulator (`sff`) forwards and backwards and require that they
        // disagree — proving f64 addition is genuinely non-associative *here*, on
        // this data, and so that the equalities below are load-bearing.
        let (avg_fix, _) = metric.means(&t).unwrap();
        let mut scratch = metric.fixed.scratch();
        let mut contributions = Vec::with_capacity(metric.sample_count());
        for s in 0..metric.fixed.len() {
            let fp = metric.fixed.point(s, &mut scratch);
            let mp = t.transform_point(fp);
            if metric.moving.value_at(&mp).is_none() {
                continue;
            }
            let f1 = metric.fixed.value(s) - avg_fix;
            contributions.push(f1 * f1);
        }
        let forward = contributions.iter().fold(0.0f64, |a, &c| a + c);
        let backward = contributions.iter().rev().fold(0.0f64, |a, &c| a + c);
        assert_ne!(
            forward.to_bits(),
            backward.to_bits(),
            "summing {} contributions forwards and backwards gave identical bits, so this input \
             cannot detect a re-association and the assertions below prove nothing — pick data \
             with a wider exponent spread",
            contributions.len()
        );

        for threads in [1usize, 4, 48, 96] {
            let got = sitk_core::parallel::with_threads(threads, || metric.evaluate(&t));
            assert_eq!(
                got.value.to_bits(),
                want.value.to_bits(),
                "value moved at {threads} threads: {} ({:#018x}) vs serial {} ({:#018x})",
                got.value,
                got.value.to_bits(),
                want.value,
                want.value.to_bits()
            );
            assert_eq!(
                got.valid_points, want.valid_points,
                "valid_points moved at {threads} threads"
            );
            for (k, (g, w)) in got.derivative.iter().zip(&want.derivative).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "derivative[{k}] moved at {threads} threads: {g} ({:#018x}) vs serial {w} \
                     ({:#018x})",
                    g.to_bits(),
                    w.to_bits()
                );
            }

            let got_value_only = sitk_core::parallel::with_threads(threads, || metric.value(&t));
            assert_eq!(
                got_value_only.to_bits(),
                want_value_only.to_bits(),
                "value() moved at {threads} threads: {got_value_only} vs serial {want_value_only}"
            );
        }
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
