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
//! by construction, from [`crate::core::parallel::map_rows_fold_in_order`], the
//! same primitive [`crate::registration::metric`]'s mean-squares backend uses: the expensive
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
//! [`MattesMutualInformationMetric`](crate::registration::mattes::MattesMutualInformationMetric)
//! documents for its `nFactor` and mean squares documents for differencing
//! `M − F` where ITK differences `F − M`. Concretely, this module's derivative
//! is the *negation* of the literal ITK code's
//! `fc · sfm/(sff·smm) · (fdm − sfm/smm·mdm)` (`fc = 2.0` there; effectively
//! `fc = −2.0` here). The finite-difference test below pins this down:
//! `derivative == d(value)/d(param)`.
//!
//! ## No local-support branch
//!
//! Unlike [`MattesMutualInformationMetric`](crate::registration::mattes::MattesMutualInformationMetric),
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

use crate::core::Image;
use crate::core::parallel;
use crate::transform::ParametricTransform;

use crate::registration::error::{RegistrationError, Result};
use crate::registration::metric::{FixedSamples, MetricValue, MovingImage};
use crate::registration::scales::{ScalesEstimator, ScalesEstimatorKind};

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
    /// [`DisplacementFieldTransform`](crate::transform::DisplacementFieldTransform)).
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
    use crate::transform::{TransformBase, TranslationTransform};

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
    /// [`crate::core::parallel::map_rows_fold_in_order`] replaced them — two passes,
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
            let got = crate::core::parallel::with_threads(threads, || metric.evaluate(&t));
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

            let got_value_only = crate::core::parallel::with_threads(threads, || metric.value(&t));
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
        use crate::registration::optimizer::GradientDescentOptimizer;

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
        use crate::transform::DisplacementFieldTransform;

        let (w, h, sigma) = (16usize, 16usize, 4.0);
        let img = gaussian(w, h, 8.0, 8.0, sigma, 1.0);
        let metric = CorrelationMetric::new(&img, &img).unwrap();
        let field = DisplacementFieldTransform::from_image_domain(&img).unwrap();

        let _ = metric.evaluate(&field);
    }

    #[test]
    fn check_transform_rejects_a_displacement_field_before_evaluating() {
        use crate::transform::{DisplacementFieldTransform, TranslationTransform};

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

/// What the one-pass moment form would cost — the design the device does **not**
/// take, measured rather than asserted.
///
/// # The road not taken
///
/// A device NCC could be a **single** reduction. Every mean-subtracted
/// accumulator is affine in the means, so the mean-subtraction can be deferred
/// to the host:
///
/// ```text
/// sff       = Σ fᵢ²        − N·f̄²
/// sfm       = Σ fᵢ·mᵢ      − N·f̄·m̄
/// fdm[d]    = Σ fᵢ·∇Mᵢ[d]  − f̄·Σ ∇Mᵢ[d]
/// ```
///
/// One resample pass, 42 raw moments, no dependence on a global quantity inside
/// the kernel. It is **algebraically identical** to the two-pass form this module
/// computes — and numerically it is not, because each line above is a difference
/// of two comparable magnitudes. The relative error of `sff` computed that way is
/// `ε · Σfᵢ²/sff = ε · (1 + f̄²/var(f))`, and that amplification factor is a
/// property of **the caller's data**, not of the algorithm: a CT volume sitting at
/// mean 1000 with σ = 50 carries a factor of ~400.
///
/// The device therefore runs the host's **two** passes — pass 1 reduces the sums
/// that give the means, pass 2 reduces the mean-subtracted moments with the means
/// as kernel scalars — and pays ~2× the memory traffic for it. That is a
/// deliberate purchase, and these tests are its receipt:
///
/// * [`the_two_forms_are_the_same_algebra`] — on a centred volume the two agree to
///   the last bits, so the divergence below is *not* a coding error in either form.
/// * [`the_one_pass_form_loses_digits_to_the_dc_offset`] — on a CT-like volume the
///   one-pass form is orders worse, with the measured factor in the failure
///   message.
/// * [`what_the_one_pass_form_loses_is_a_property_of_the_data`] — the loss grows
///   with the pedestal, tracking `1 + f̄²/var(f)`. This is why no fixed tolerance
///   can be written for the cheap form.
///
/// If a later change collapses the two device passes into one, the second and
/// third tests fail. That is their entire purpose.
#[cfg(test)]
mod one_pass_moment_form {
    use super::*;
    use crate::transform::{TransformBase, TranslationTransform};

    /// Neumaier compensated summation — the **reference** both forms are measured
    /// against. `sff`/`smm` have all-non-negative terms, so their condition number
    /// is 1 and this is accurate to `ε`.
    fn csum(terms: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for &t in terms {
            let s = sum + t;
            c += if sum.abs() >= t.abs() {
                (sum - s) + t
            } else {
                (t - s) + sum
            };
            sum = s;
        }
        sum + c
    }

    /// Plain left-to-right `f64` summation — what a real reduction (the host's
    /// serial loop, or the device's block tree) actually does. Measuring both forms
    /// through *this* is what says how the cheap form would behave in the kernel;
    /// measuring them through [`csum`] isolates the *form* from the summation order
    /// by giving the cheap form a perfect one.
    fn psum(terms: &[f64]) -> f64 {
        terms.iter().fold(0.0f64, |a, &t| a + t)
    }

    /// The per-sample terms of one NCC evaluation: the fixed and moving values at
    /// the valid samples, and the moving gradient there. Both moment forms are
    /// built from exactly these, so nothing but the *arithmetic* differs.
    struct Terms {
        f: Vec<f64>,
        m: Vec<f64>,
        /// `g[k]` is the k-th component of ∇M, one entry per valid sample. Under a
        /// `TranslationTransform` the Jacobian is the identity, so `fdm[k] = Σ f1·g[k]`
        /// directly — the derivative accumulator without the Jacobian machinery.
        g: [Vec<f64>; 3],
    }

    fn terms(fixed: &Image, moving: &Image, shift: &[f64]) -> Terms {
        let samples = FixedSamples::from_image(fixed).unwrap();
        let mov = MovingImage::from_image(moving).unwrap();
        let t = TranslationTransform::new(shift.to_vec());

        let mut out = Terms {
            f: Vec::new(),
            m: Vec::new(),
            g: [Vec::new(), Vec::new(), Vec::new()],
        };
        let mut scratch = samples.scratch();
        for s in 0..samples.len() {
            let fp = samples.point(s, &mut scratch);
            let mp = t.transform_point(fp);
            let Some((mv, grad)) = mov.value_and_physical_gradient(&mp) else {
                continue;
            };
            out.f.push(samples.value(s));
            out.m.push(mv);
            for (gk, &g) in out.g.iter_mut().zip(&grad) {
                gk.push(g);
            }
        }
        assert!(out.f.len() > 1000, "too few valid samples to measure with");
        out
    }

    /// The quantities a device NCC reduction produces, whichever form produced
    /// them. `fdm` is the derivative accumulator `Σ f1ᵢ·∇Mᵢ[k]` (the Jacobian is
    /// the identity under a `TranslationTransform`, so this *is* the derivative's
    /// fixed half).
    struct Form {
        sff: f64,
        sfm: f64,
        value: f64,
        fdm: [f64; 3],
    }

    /// How a form's sums are accumulated. Both forms are run through **both**, so
    /// the summation order and the algebraic form are separated rather than
    /// confounded.
    type Sum = fn(&[f64]) -> f64;

    /// The **two-pass** form, the one the device takes: means first, then the
    /// mean-subtracted moments. Every term is formed exactly as
    /// [`CorrelationMetric::evaluate`] forms it.
    fn two_pass(t: &Terms, sum: Sum) -> Form {
        let n = t.f.len() as f64;
        let fbar = sum(&t.f) / n;
        let mbar = sum(&t.m) / n;

        let f1: Vec<f64> = t.f.iter().map(|f| f - fbar).collect();
        let m1: Vec<f64> = t.m.iter().map(|m| m - mbar).collect();

        let sff = sum(&f1.iter().map(|a| a * a).collect::<Vec<_>>());
        let smm = sum(&m1.iter().map(|a| a * a).collect::<Vec<_>>());
        let sfm = sum(&f1.iter().zip(&m1).map(|(a, b)| a * b).collect::<Vec<_>>());

        let mut fdm = [0.0f64; 3];
        for (k, fdmk) in fdm.iter_mut().enumerate() {
            *fdmk = sum(&f1
                .iter()
                .zip(&t.g[k])
                .map(|(a, g)| a * g)
                .collect::<Vec<_>>());
        }

        Form {
            sff,
            sfm,
            value: -sfm * sfm / (sff * smm),
            fdm,
        }
    }

    /// The **one-pass** form, the one the device refuses: raw moments, mean
    /// subtracted afterwards. Algebraically identical to [`two_pass`]; every line
    /// below is a difference of two comparable magnitudes, and that is the whole
    /// difference.
    fn one_pass(t: &Terms, sum: Sum) -> Form {
        let n = t.f.len() as f64;
        let fbar = sum(&t.f) / n;
        let mbar = sum(&t.m) / n;

        let sum_ff = sum(&t.f.iter().map(|a| a * a).collect::<Vec<_>>());
        let sum_mm = sum(&t.m.iter().map(|a| a * a).collect::<Vec<_>>());
        let sum_fm = sum(&t.f.iter().zip(&t.m).map(|(a, b)| a * b).collect::<Vec<_>>());

        let sff = sum_ff - n * fbar * fbar;
        let smm = sum_mm - n * mbar * mbar;
        let sfm = sum_fm - n * fbar * mbar;

        let mut fdm = [0.0f64; 3];
        for (k, fdmk) in fdm.iter_mut().enumerate() {
            let sum_fg = sum(&t
                .f
                .iter()
                .zip(&t.g[k])
                .map(|(a, g)| a * g)
                .collect::<Vec<_>>());
            let sum_g = sum(&t.g[k]);
            *fdmk = sum_fg - fbar * sum_g;
        }

        Form {
            sff,
            sfm,
            value: -sfm * sfm / (sff * smm),
            fdm,
        }
    }

    fn rel(a: f64, b: f64) -> f64 {
        if b == 0.0 {
            a.abs()
        } else {
            (a - b).abs() / b.abs()
        }
    }

    /// The amplification the algebra predicts for `sff`: `Σf²/sff = 1 + f̄²/var(f)`.
    /// This is the factor the one-pass form multiplies its summation error by.
    fn dc_amplification(t: &Terms) -> f64 {
        let n = t.f.len() as f64;
        let fbar = csum(&t.f) / n;
        let sum_ff = csum(&t.f.iter().map(|a| a * a).collect::<Vec<_>>());
        let sff = csum(
            &t.f.iter()
                .map(|f| (f - fbar) * (f - fbar))
                .collect::<Vec<_>>(),
        );
        sum_ff / sff
    }

    /// The same factor for the derivative accumulator: `|f̄·Σ∇M[k]| / |fdm[k]|`,
    /// worst `k`.
    ///
    /// I predicted, when I designed this, that `fdm`'s cancellation would be the
    /// *worst* of the three because its subtrahend `Σ∇M[k]` is a signed sum that can
    /// approach zero. That was wrong, and wrong in the direction that matters: a
    /// subtrahend near zero has nothing to cancel *against*, so the loss is small,
    /// not large. `fdm` loses only when `|f̄·Σ∇M|` is **large** next to `fdm` itself
    /// — and over a whole volume the gradients largely cancel, so it is not. The DC
    /// offset lands on `sff`/`smm`/`sfm`, and therefore on the value.
    /// [`the_dc_loss_lands_on_the_value_not_the_derivative`] pins that as measured.
    fn fdm_cancellation(t: &Terms) -> f64 {
        let n = t.f.len() as f64;
        let fbar = csum(&t.f) / n;
        let exact = two_pass(t, csum);
        (0..3)
            .map(|k| {
                let sum_g = csum(&t.g[k]);
                (fbar * sum_g).abs() / exact.fdm[k].abs()
            })
            .fold(0.0f64, f64::max)
    }

    /// A textured volume with mean ≈ `pedestal` and a spread of ≈ 50 intensity
    /// units — a CT-like signal-to-DC ratio when `pedestal` is 1000. Deterministic
    /// (SplitMix64), so the measured numbers below are reproducible.
    fn volume(n: usize, pedestal: f64, shift: f64) -> Image {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let c = n as f64 / 2.0;
        let mut v = vec![0.0f64; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let (x, y, z) = (i as f64 - c + shift, j as f64 - c, k as f64 - c);
                    // A smooth blob plus texture; σ over the volume lands near 50.
                    let blob = 90.0 * (-(x * x + y * y + z * z) / (2.0 * 90.0)).exp();
                    let ripple = 30.0 * (0.4 * x).sin() * (0.3 * y).cos();
                    let noise = 12.0 * (next() - 0.5);
                    v[(k * n + j) * n + i] = pedestal + blob + ripple + noise;
                }
            }
        }
        Image::from_vec(&[n, n, n], v).unwrap()
    }

    /// The two forms are the same algebra: on a **centred** volume (`f̄ ≈ 0`, so the
    /// cancellation has nothing to cancel) they agree to the last bits. Without
    /// this, the divergence measured by the next two tests could just be a bug in
    /// one of them.
    #[test]
    fn the_two_forms_are_the_same_algebra() {
        let t = terms(
            &volume(40, 0.0, 0.0),
            &volume(40, 0.0, 1.5),
            &[1.3, -0.7, 0.9],
        );
        let reference = two_pass(&t, csum);
        let cheap = one_pass(&t, csum);

        let amp = dc_amplification(&t);
        assert!(amp < 2.0, "volume is not centred: DC amplification {amp}");

        for (name, a, b) in [
            ("sff", cheap.sff, reference.sff),
            ("sfm", cheap.sfm, reference.sfm),
            ("value", cheap.value, reference.value),
            ("fdm[0]", cheap.fdm[0], reference.fdm[0]),
            ("fdm[1]", cheap.fdm[1], reference.fdm[1]),
            ("fdm[2]", cheap.fdm[2], reference.fdm[2]),
        ] {
            assert!(
                rel(a, b) <= 1e-13,
                "{name}: the two forms disagree at {:.3e} on a centred volume — \
                 that is a bug in one of them, not the cancellation this module measures",
                rel(a, b)
            );
        }
    }

    /// The one-pass form loses digits to the DC offset, and the loss is what the
    /// device would actually eat.
    ///
    /// Both forms are measured against the compensated two-pass reference, twice:
    ///
    /// * **through [`csum`]** — the cheap form's *best case*, a perfect summation
    ///   order. What survives is the cancellation in the final subtraction alone.
    /// * **through [`psum`]** — a real reduction. Here the DC amplification
    ///   multiplies the summation error too, which is the regime a kernel is in.
    ///
    /// The floors asserted below sit an order under the measured values, and the
    /// message prints what was measured, so no reader has to trust the constants.
    #[test]
    fn the_one_pass_form_loses_digits_to_the_dc_offset() {
        let t = terms(
            &volume(40, 1000.0, 0.0),
            &volume(40, 1000.0, 1.5),
            &[1.3, -0.7, 0.9],
        );
        let amp = dc_amplification(&t);
        assert!(
            amp > 100.0,
            "the CT-like volume did not reach the DC regime: amplification {amp}"
        );

        let reference = two_pass(&t, csum);

        // Best case: both forms perfectly summed. Only the algebra differs.
        let best = one_pass(&t, csum);
        let best_sff = rel(best.sff, reference.sff);
        let best_value = rel(best.value, reference.value);

        // Real case: both forms through a plain reduction, as a kernel sums.
        let real_two = two_pass(&t, psum);
        let real_one = one_pass(&t, psum);
        let kept_sff = rel(real_two.sff, reference.sff);
        let kept_value = rel(real_two.value, reference.value);
        let lost_sff = rel(real_one.sff, reference.sff);
        let lost_value = rel(real_one.value, reference.value);

        eprintln!(
            "N0 @ mean 1000 (DC amplification {amp:.1}, {} samples):\n\
             \x20   one-pass, perfectly summed : sff {best_sff:.3e}  value {best_value:.3e}\n\
             \x20   TWO-pass, plainly summed   : sff {kept_sff:.3e}  value {kept_value:.3e}   <- the form the device runs\n\
             \x20   one-pass, plainly summed   : sff {lost_sff:.3e}  value {lost_value:.3e}   <- the form the device refuses\n\
             \x20   cost of the cheap form     : sff {:.0}x  value {:.0}x",
            t.f.len(),
            lost_sff / kept_sff,
            lost_value / kept_value,
        );

        // Even perfectly summed, the cheap form cannot reach the reference: the
        // cancellation is in the subtraction, not the sum.
        assert!(
            best_sff > 1e-15 && best_value > 1e-14,
            "at DC amplification {amp:.1} the one-pass form did not diverge even in \
             its best case (sff {best_sff:.3e}, value {best_value:.3e}) — it is not \
             being exercised, so this pin is guarding nothing"
        );

        // And through a real reduction it is orders worse than the form we run.
        assert!(
            lost_sff > 30.0 * kept_sff,
            "the one-pass form cost only {:.1}x on sff (one-pass {lost_sff:.3e} vs \
             two-pass {kept_sff:.3e}) at DC amplification {amp:.1}; the pin was \
             written from a measured ~1e3x. Either the volume no longer has a DC \
             offset, or the form under test is no longer the one-pass form",
            lost_sff / kept_sff
        );
        assert!(
            lost_value > 30.0 * kept_value,
            "the one-pass form cost only {:.1}x on the value (one-pass {lost_value:.3e} \
             vs two-pass {kept_value:.3e}) at DC amplification {amp:.1}",
            lost_value / kept_value
        );
    }

    /// Where the DC loss lands — and where I predicted it would land, wrongly.
    ///
    /// Designing this wave I claimed `fdm`'s cancellation would be the worst of the
    /// three, because its subtrahend `f̄·Σ∇M[k]` is a signed sum that can approach
    /// zero. The measurement says otherwise, and the reasoning was backwards: a
    /// subtrahend near zero has nothing to cancel against. Over a whole volume the
    /// moving gradients largely cancel, so `|f̄·Σ∇M|` is *small* next to `fdm`, and
    /// `fdm` barely moves. The DC offset lands on `sff`/`smm`/`sfm` — and so on the
    /// value, which is exactly the quantity the optimizer reads.
    ///
    /// Pinned, so the refusal rests on where the loss *is* rather than on where I
    /// guessed it would be.
    #[test]
    fn the_dc_loss_lands_on_the_value_not_the_derivative() {
        let t = terms(
            &volume(40, 1000.0, 0.0),
            &volume(40, 1000.0, 1.5),
            &[1.3, -0.7, 0.9],
        );
        let reference = two_pass(&t, csum);
        let cheap = one_pass(&t, csum);

        let e_sff = rel(cheap.sff, reference.sff);
        let e_fdm = (0..3)
            .map(|k| rel(cheap.fdm[k], reference.fdm[k]))
            .fold(0.0f64, f64::max);
        let fdm_cancel = fdm_cancellation(&t);
        let sff_cancel = dc_amplification(&t);

        eprintln!(
            "N0 where the loss lands: sff cancellation {sff_cancel:.1} -> error {e_sff:.3e} | \
             fdm cancellation {fdm_cancel:.2} -> error {e_fdm:.3e}"
        );

        // The measured relation, not a threshold I invented: the error each
        // quantity loses tracks *its own* cancellation factor. `sff` cancels at
        // ~1.9e3 and loses ~2.4e-14; `fdm` cancels at ~1.1e1 and loses ~1.3e-15.
        // Two orders apart in cancellation, and the errors follow.
        assert!(
            fdm_cancel * 20.0 < sff_cancel,
            "fdm's cancellation ({fdm_cancel:.2}) is no longer far below sff's \
             ({sff_cancel:.1}) — the gradients stopped cancelling over the volume, \
             so 'the DC loss misses the derivative' is not true of this data and \
             the module doc above needs rewriting, not this constant"
        );
        assert!(
            e_fdm < e_sff,
            "fdm ({e_fdm:.3e}) lost more than sff ({e_sff:.3e}) — my original \
             prediction after all; if this fires, the module doc above is wrong \
             and the derivative needs the same scrutiny as the value"
        );
    }

    /// The loss is a property of the **data**, not of the algorithm: it grows with
    /// the pedestal, tracking `1 + f̄²/var(f)`. This is the whole argument for
    /// paying the second pass — no fixed tolerance can be written for a form whose
    /// error the caller's intensity range decides.
    #[test]
    fn what_the_one_pass_form_loses_is_a_property_of_the_data() {
        let mut previous: Option<(f64, f64)> = None;
        for pedestal in [0.0, 1_000.0, 10_000.0] {
            let t = terms(
                &volume(40, pedestal, 0.0),
                &volume(40, pedestal, 1.5),
                &[1.3, -0.7, 0.9],
            );
            let reference = two_pass(&t, csum);
            let cheap = one_pass(&t, psum);
            let kept = two_pass(&t, psum);

            let amp = dc_amplification(&t);
            let err = rel(cheap.sff, reference.sff);
            eprintln!(
                "N0 pedestal {pedestal:>8.0}: amplification {amp:>10.1}  \
                 one-pass sff error {err:.3e}  (two-pass {:.3e})",
                rel(kept.sff, reference.sff)
            );

            if let Some((prev_amp, prev_err)) = previous {
                assert!(
                    amp > prev_amp * 10.0,
                    "amplification did not grow with the pedestal: {prev_amp} -> {amp}"
                );
                assert!(
                    err > prev_err,
                    "the one-pass error did not grow with the DC offset \
                     ({prev_err:.3e} at amplification {prev_amp:.1} -> \
                     {err:.3e} at {amp:.1}); the cancellation this pin measures is gone, \
                     which means the form under test is no longer the one-pass form"
                );
            }
            previous = Some((amp, err));
        }
    }
}
