//! `itk::ImageRegistrationMethodv4` / SimpleITK `ImageRegistrationMethod`.
//!
//! Ties a similarity metric, an optimizer, and a parametric transform into one
//! optimization loop: starting from an initial transform, the optimizer searches
//! the transform's parameter space to minimize the metric between the fixed and
//! (transformed) moving image.
//!
//! Supports a **multi-resolution pyramid** (`itk::ImageRegistrationMethodv4`'s
//! shrink/smoothing schedule): at each level the fixed image is Gaussian-smoothed
//! and shrunk, the moving image is Gaussian-smoothed (not shrunk, since it is
//! resampled through the transform), and the transform optimized at the coarse
//! level initializes the next finer one. With no schedule set it runs a single
//! full-resolution level. The metric is mean squares, interpolation is linear,
//! and optimization is gradient descent — the smallest end-to-end registration.
//! More metrics/optimizers and sampling strategies come later.

use sitk_core::Image;
use sitk_filters::{shrink, smooth_gaussian};
use sitk_transform::{AffineTransform, Interpolator, ParametricTransform, ResampleImageFilter};

use crate::error::{RegistrationError, Result};
use crate::metric::{CpuBackend, MeanSquaresMetric, MetricBackend};
use crate::optimizer::{GradientDescentOptimizer, StopReason};

/// When the gradient-descent learning rate is estimated from physical shift,
/// mirroring SimpleITK's `estimateLearningRate` option
/// (`itk::GradientDescentOptimizerv4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimateLearningRate {
    /// Estimate once from the initial gradient, then hold it fixed — ITK's
    /// default (`DoEstimateLearningRateOnce`). Steps shrink with the gradient,
    /// so it refines to high precision.
    Once,
    /// Re-estimate every iteration from the current gradient
    /// (`DoEstimateLearningRateAtEachIteration`). Every step moves the samples
    /// by ~one voxel and does not shrink, so it converges only coarsely
    /// (≈ voxel precision) and stops via value-plateau convergence monitoring
    /// rather than a small step.
    EachIteration,
}

/// How optimizer parameter scales are chosen.
enum ScalesMode {
    /// All-ones (no balancing).
    Unit,
    /// Caller-supplied scales.
    Manual(Vec<f64>),
    /// Estimated from physical shift
    /// (`RegistrationParameterScalesFromPhysicalShift`).
    PhysicalShift,
}

/// How the learning rate is chosen.
enum LearningRateMode {
    /// Caller-supplied fixed rate.
    Manual,
    /// Estimated from physical shift, once or at each iteration.
    Estimate(EstimateLearningRate),
}

/// SimpleITK `ImageRegistrationMethod` defaults for convergence monitoring in
/// the estimate-at-each-iteration mode (window of the last 10 metric values,
/// stop when the trend flattens to `1e-6`).
const CONVERGENCE_WINDOW_SIZE: usize = 10;
const MINIMUM_CONVERGENCE_VALUE: f64 = 1e-6;

/// The optimized transform plus diagnostics from a registration run.
#[derive(Clone, Debug)]
pub struct RegistrationResult<T> {
    /// The transform at the optimizer's final iterate.
    pub transform: T,
    /// Metric value at that transform (lower is better for mean squares).
    pub metric_value: f64,
    /// Optimizer steps taken.
    pub iterations: usize,
    /// Why the optimizer stopped.
    pub stop_reason: StopReason,
    /// Fixed samples that mapped inside the moving image at the final transform.
    pub valid_points: usize,
}

/// The registration driver. Configure the optimizer (and optionally parameter
/// scales, a multi-resolution pyramid, and the metric compute backend), then
/// [`execute`](Self::execute) against a fixed/moving image pair and an initial
/// transform.
pub struct ImageRegistrationMethod {
    optimizer: GradientDescentOptimizer,
    scales_mode: ScalesMode,
    learning_rate_mode: LearningRateMode,
    backend: Box<dyn MetricBackend>,
    /// One shrink factor per resolution level (applied to every dimension),
    /// coarsest first. Empty means a single full-resolution level.
    shrink_factors_per_level: Vec<usize>,
    /// One Gaussian smoothing sigma per resolution level, coarsest first. Must
    /// match `shrink_factors_per_level` in length when either is set.
    smoothing_sigmas_per_level: Vec<f64>,
    /// Whether `smoothing_sigmas_per_level` are in physical units (ITK's
    /// default, `true`) or in voxels of the fixed image.
    smoothing_sigmas_in_physical_units: bool,
}

impl Default for ImageRegistrationMethod {
    fn default() -> Self {
        Self {
            optimizer: GradientDescentOptimizer::new(1.0, 100),
            scales_mode: ScalesMode::Unit,
            learning_rate_mode: LearningRateMode::Manual,
            backend: Box::new(CpuBackend),
            shrink_factors_per_level: Vec::new(),
            smoothing_sigmas_per_level: Vec::new(),
            smoothing_sigmas_in_physical_units: true,
        }
    }
}

impl ImageRegistrationMethod {
    /// A registration method with default settings (mean-squares metric, linear
    /// interpolation, CPU backend, gradient descent at learning rate 1 for 100
    /// iterations). The metric is mean squares — the only Phase-0 metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use gradient descent with a caller-supplied fixed learning rate and
    /// iteration cap.
    pub fn set_optimizer_as_gradient_descent(
        &mut self,
        learning_rate: f64,
        iterations: usize,
    ) -> &mut Self {
        self.optimizer = GradientDescentOptimizer::new(learning_rate, iterations);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use gradient descent whose learning rate is **estimated** from physical
    /// shift (no hand-tuned rate), mirroring ITK/SimpleITK's learning-rate
    /// estimation. `estimate` selects [`EstimateLearningRate::Once`] (ITK's
    /// default — refines to high precision) or
    /// [`EstimateLearningRate::EachIteration`] (converges coarsely, stopped by
    /// value-plateau monitoring). Pair with
    /// [`set_optimizer_scales_from_physical_shift`] for a fully automatic
    /// optimizer.
    ///
    /// [`set_optimizer_scales_from_physical_shift`]:
    /// Self::set_optimizer_scales_from_physical_shift
    pub fn set_optimizer_as_gradient_descent_estimated(
        &mut self,
        iterations: usize,
        estimate: EstimateLearningRate,
    ) -> &mut Self {
        // The stored rate is a placeholder; it is overwritten by the estimate.
        self.optimizer = GradientDescentOptimizer::new(1.0, iterations);
        if estimate == EstimateLearningRate::EachIteration {
            // A non-shrinking step schedule needs value-plateau monitoring to
            // stop; the once schedule stops via the min-step tolerance.
            self.optimizer
                .set_convergence(CONVERGENCE_WINDOW_SIZE, MINIMUM_CONVERGENCE_VALUE);
        }
        self.learning_rate_mode = LearningRateMode::Estimate(estimate);
        self
    }

    /// Set per-parameter optimizer scales manually. Length is validated against
    /// the transform's parameter count in [`execute`](Self::execute).
    pub fn set_optimizer_scales(&mut self, scales: Vec<f64>) -> &mut Self {
        self.scales_mode = ScalesMode::Manual(scales);
        self
    }

    /// Estimate optimizer scales automatically from physical shift
    /// (`itk::RegistrationParameterScalesFromPhysicalShift`), so matrix and
    /// translation parameters are balanced without hand-tuning.
    pub fn set_optimizer_scales_from_physical_shift(&mut self) -> &mut Self {
        self.scales_mode = ScalesMode::PhysicalShift;
        self
    }

    /// Set the minimum scaled-step length below which the optimizer stops early.
    pub fn set_min_step_tolerance(&mut self, tol: f64) -> &mut Self {
        self.optimizer.set_min_step_tolerance(tol);
        self
    }

    /// Replace the metric compute backend (the GPU seam). Defaults to
    /// [`CpuBackend`]; a CUDA/`wgpu` backend implements the same
    /// [`MetricBackend`] trait.
    pub fn set_metric_backend(&mut self, backend: Box<dyn MetricBackend>) -> &mut Self {
        self.backend = backend;
        self
    }

    /// Set the per-level shrink factors of the multi-resolution pyramid
    /// (`itk::ImageRegistrationMethodv4::SetShrinkFactorsPerLevel`), coarsest
    /// level first — e.g. `[4, 2, 1]`. Each factor is applied to every
    /// dimension. Must be paired with
    /// [`set_smoothing_sigmas_per_level`](Self::set_smoothing_sigmas_per_level)
    /// of the same length. An empty schedule (the default) runs a single
    /// full-resolution level.
    pub fn set_shrink_factors_per_level(&mut self, factors: Vec<usize>) -> &mut Self {
        self.shrink_factors_per_level = factors;
        self
    }

    /// Set the per-level Gaussian smoothing sigmas
    /// (`itk::ImageRegistrationMethodv4::SetSmoothingSigmasPerLevel`), coarsest
    /// level first — e.g. `[2.0, 1.0, 0.0]` (a `0` level is unsmoothed). By
    /// default these are in physical units; see
    /// [`set_smoothing_sigmas_are_specified_in_physical_units`].
    ///
    /// [`set_smoothing_sigmas_are_specified_in_physical_units`]:
    /// Self::set_smoothing_sigmas_are_specified_in_physical_units
    pub fn set_smoothing_sigmas_per_level(&mut self, sigmas: Vec<f64>) -> &mut Self {
        self.smoothing_sigmas_per_level = sigmas;
        self
    }

    /// Choose whether the per-level smoothing sigmas are in physical units
    /// (`true`, ITK's default) or in voxels of the fixed image (`false`, scaled
    /// by the fixed spacing before smoothing).
    pub fn set_smoothing_sigmas_are_specified_in_physical_units(
        &mut self,
        physical: bool,
    ) -> &mut Self {
        self.smoothing_sigmas_in_physical_units = physical;
        self
    }

    /// Register `moving` onto `fixed`, optimizing `initial` and returning it
    /// with run diagnostics from the finest (last) level.
    ///
    /// Runs the multi-resolution pyramid when a shrink/smoothing schedule is set
    /// (see [`set_shrink_factors_per_level`] and
    /// [`set_smoothing_sigmas_per_level`]); otherwise a single full-resolution
    /// level. Per level the fixed image is Gaussian-smoothed then shrunk and the
    /// moving image Gaussian-smoothed; the coarse-level transform initializes the
    /// next finer one. Transforms act in physical space, shared across levels, so
    /// the parameters carry over directly with no rescaling.
    ///
    /// Errors if the transform/image dimensions disagree, the shrink and
    /// smoothing schedules differ in length, the moving direction matrix is
    /// singular, scales are the wrong length, or no fixed sample maps inside the
    /// moving image at the final transform.
    ///
    /// [`set_shrink_factors_per_level`]: Self::set_shrink_factors_per_level
    /// [`set_smoothing_sigmas_per_level`]: Self::set_smoothing_sigmas_per_level
    pub fn execute<T: ParametricTransform>(
        &self,
        fixed: &Image,
        moving: &Image,
        initial: T,
    ) -> Result<RegistrationResult<T>> {
        if initial.dimension() != fixed.dimension() {
            return Err(RegistrationError::TransformDimensionMismatch {
                transform: initial.dimension(),
                image: fixed.dimension(),
            });
        }

        let dim = fixed.dimension();
        let schedule = self.level_schedule(fixed)?;

        let mut transform = initial;
        let mut diagnostics = None;
        for (level_factors, level_sigma) in &schedule {
            let sigma = self.physical_sigma(fixed, *level_sigma);
            let (fixed_level, moving_level) =
                self.prepare_level(fixed, moving, &sigma, level_factors, dim)?;
            let r = self.run_single_level(&fixed_level, &moving_level, transform)?;
            diagnostics = Some((r.metric_value, r.iterations, r.stop_reason, r.valid_points));
            transform = r.transform;
        }
        let (metric_value, iterations, stop_reason, valid_points) =
            diagnostics.expect("level schedule always has at least one level");

        Ok(RegistrationResult {
            transform,
            metric_value,
            iterations,
            stop_reason,
            valid_points,
        })
    }

    /// The per-level `(shrink_factors, sigma)` schedule, coarsest first. With no
    /// schedule configured this is one full-resolution level (factor 1, sigma 0).
    /// Errors if the shrink and smoothing schedules differ in length.
    fn level_schedule(&self, fixed: &Image) -> Result<Vec<(Vec<usize>, f64)>> {
        let dim = fixed.dimension();
        if self.shrink_factors_per_level.is_empty() && self.smoothing_sigmas_per_level.is_empty() {
            return Ok(vec![(vec![1; dim], 0.0)]);
        }
        if self.shrink_factors_per_level.len() != self.smoothing_sigmas_per_level.len() {
            return Err(RegistrationError::PyramidScheduleLength {
                shrink: self.shrink_factors_per_level.len(),
                sigma: self.smoothing_sigmas_per_level.len(),
            });
        }
        Ok(self
            .shrink_factors_per_level
            .iter()
            .zip(self.smoothing_sigmas_per_level.iter())
            .map(|(&f, &s)| (vec![f; dim], s))
            .collect())
    }

    /// Per-dimension physical smoothing sigma for a scalar level sigma. When the
    /// schedule is given in voxel units, scale by the fixed image's spacing
    /// (matching ITK, whose smoother always works in physical units).
    fn physical_sigma(&self, fixed: &Image, sigma: f64) -> Vec<f64> {
        if self.smoothing_sigmas_in_physical_units {
            vec![sigma; fixed.dimension()]
        } else {
            fixed.spacing().iter().map(|&sp| sigma * sp).collect()
        }
    }

    /// Build one resolution level's `(fixed, moving)` pair from the physical
    /// smoothing `sigma` and per-dimension shrink `factors`.
    ///
    /// The moving image is only smoothed (it is resampled through the transform,
    /// so it is not shrunk). The fixed image is smoothed and then placed on the
    /// coarse **virtual-domain** grid: ITK shrinks the virtual domain with
    /// `ShrinkImageFilter`, so we take that grid's geometry, but the fixed values
    /// on it are obtained by **resampling the smoothed fixed with linear
    /// interpolation** — matching ITK's metric, which interpolates the smoothed
    /// fixed at each virtual point. Reusing `ShrinkImageFilter`'s subsampled
    /// pixel values instead would introduce a sub-voxel translation bias, because
    /// that filter's output origin (from the real-valued center shift) and its
    /// sampling offset (that shift rounded to an integer) intentionally differ by
    /// up to half a voxel.
    fn prepare_level(
        &self,
        fixed: &Image,
        moving: &Image,
        sigma: &[f64],
        factors: &[usize],
        dim: usize,
    ) -> Result<(Image, Image)> {
        let smoothed_fixed = smooth_gaussian(fixed, sigma)?;
        let coarse_grid = shrink(&smoothed_fixed, factors)?;
        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&coarse_grid)
            .set_interpolator(Interpolator::Linear);
        let fixed_level = resampler.execute(&smoothed_fixed, &AffineTransform::identity(dim))?;
        let moving_level = smooth_gaussian(moving, sigma)?;
        Ok((fixed_level, moving_level))
    }

    /// Optimize `initial` against one already shrunk/smoothed fixed/moving pair
    /// — a single resolution level of [`execute`](Self::execute).
    fn run_single_level<T: ParametricTransform>(
        &self,
        fixed: &Image,
        moving: &Image,
        initial: T,
    ) -> Result<RegistrationResult<T>> {
        let metric = MeanSquaresMetric::new(fixed, moving)?;
        let nparams = initial.number_of_parameters();
        let mut transform = initial;
        let start = transform.parameters();
        let backend = self.backend.as_ref();

        // A physical-shift estimator is needed if scales or the learning rate
        // are estimated. Jacobians are parameter-independent for these
        // transforms, so building it once at the initial transform is exact.
        let needs_estimator = matches!(self.scales_mode, ScalesMode::PhysicalShift)
            || matches!(self.learning_rate_mode, LearningRateMode::Estimate(_));
        let estimator = needs_estimator.then(|| metric.physical_shift_scales(&transform));

        let scales: Vec<f64> = match &self.scales_mode {
            ScalesMode::Unit => vec![1.0; nparams],
            ScalesMode::Manual(s) => {
                if s.len() != nparams {
                    return Err(RegistrationError::ScalesLength {
                        got: s.len(),
                        expected: nparams,
                    });
                }
                s.clone()
            }
            ScalesMode::PhysicalShift => estimator.as_ref().unwrap().estimate_scales(),
        };

        let mut optimizer = self.optimizer.clone();
        optimizer.set_scales(scales.clone());

        let scaled = |grad: &[f64]| -> Vec<f64> {
            grad.iter()
                .zip(scales.iter())
                .map(|(&g, &s)| g / s)
                .collect()
        };

        let result = match self.learning_rate_mode {
            // Caller-supplied fixed rate.
            LearningRateMode::Manual => optimizer.optimize(start, |p| {
                transform.set_parameters(p);
                let m = metric.evaluate(&transform, backend);
                (m.value, m.derivative)
            }),
            // Rate estimated once from the initial gradient, then held fixed so
            // steps shrink with the gradient (ITK's default). Each step is also
            // capped at the estimator's one-voxel maximum shift: a level that
            // restarts from a near-converged transform has a ~0 initial gradient,
            // which makes the once-estimated rate enormous and the *next* step
            // (once the gradient grows again) explode. The cap makes "no step
            // exceeds one voxel" hold by construction. It is inactive whenever
            // the fixed rate already bounds the step — which is every step of a
            // monotonically converging run, since the once-rate is exactly the
            // per-step rate at the initial gradient and only grows as the
            // gradient shrinks — so single-resolution runs are unchanged.
            LearningRateMode::Estimate(EstimateLearningRate::Once) => {
                let est = estimator.as_ref().unwrap();
                transform.set_parameters(&start);
                let m0 = metric.evaluate(&transform, backend);
                let lr_once = est.estimate_learning_rate(&scaled(&m0.derivative));
                optimizer.optimize_with_lr(
                    start,
                    |p| {
                        transform.set_parameters(p);
                        let m = metric.evaluate(&transform, backend);
                        (m.value, m.derivative)
                    },
                    |grad| lr_once.min(est.estimate_learning_rate(&scaled(grad))),
                )
            }
            // Rate re-estimated each iteration from the current gradient; the
            // non-shrinking step schedule stops via convergence monitoring
            // (enabled on the optimizer by the estimated-mode setter).
            LearningRateMode::Estimate(EstimateLearningRate::EachIteration) => {
                let est = estimator.as_ref().unwrap();
                optimizer.optimize_with_lr(
                    start,
                    |p| {
                        transform.set_parameters(p);
                        let m = metric.evaluate(&transform, backend);
                        (m.value, m.derivative)
                    },
                    |grad| est.estimate_learning_rate(&scaled(grad)),
                )
            }
        };

        transform.set_parameters(&result.parameters);
        let final_metric = metric.evaluate(&transform, backend);
        if final_metric.valid_points == 0 {
            return Err(RegistrationError::NoValidSamples);
        }

        Ok(RegistrationResult {
            transform,
            metric_value: final_metric.value,
            iterations: result.iterations,
            stop_reason: result.stop_reason,
            valid_points: final_metric.valid_points,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::TranslationTransform;

    /// A 2-D Gaussian blob of amplitude `amp` and width `sigma`, centred at
    /// `(cx, cy)` in physical (== index, unit spacing) coordinates.
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
    fn recovers_a_translation_with_automatic_scales_and_learning_rate() {
        // Fully automatic: physical-shift scales + estimated-once learning rate,
        // no hand-tuned values (ITK's default optimizer configuration).
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(500, EstimateLearningRate::Once);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}",
            result.metric_value,
            result.iterations
        );
    }

    #[test]
    fn each_iteration_estimation_converges_coarsely_and_stops_on_plateau() {
        // Estimate-at-each-iteration holds every step at ~one voxel, so it
        // recovers the shift only to roughly voxel precision and is stopped by
        // value-plateau convergence monitoring — not by running out of
        // iterations. This documents the mode's coarse-but-automatic behavior.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let iterations = 2000;
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(
                iterations,
                EstimateLearningRate::EachIteration,
            );
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        // Stopped on the value plateau, well before the iteration cap.
        assert_eq!(result.stop_reason, StopReason::Converged);
        assert!(
            result.iterations < iterations,
            "did not stop early: {} iterations",
            result.iterations
        );
        // Coarse recovery: within ~one voxel of the true shift.
        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1.0 && (p[1] - ty).abs() < 1.0,
            "recovered {p:?}, expected [{tx}, {ty}] (± ~1 voxel), metric {}",
            result.metric_value
        );
        // The metric dropped far below the unregistered mismatch.
        let initial = MeanSquaresMetric::new(&fixed, &moving)
            .unwrap()
            .evaluate(&TranslationTransform::new(vec![0.0, 0.0]), &CpuBackend)
            .value;
        assert!(
            result.metric_value < 0.2 * initial,
            "final {} not far below initial {}",
            result.metric_value,
            initial
        );
    }

    #[test]
    fn recovers_an_affine_with_automatic_scales_and_learning_rate() {
        // The 6-parameter affine path with NO hand-tuned scales or learning
        // rate: physical-shift scales balance the matrix (≈1) and translation
        // (≈image extent) parameters, and the learning rate is estimated once.
        use sitk_transform::AffineTransform;
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(2000, EstimateLearningRate::Once);
        let init = AffineTransform::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![20.0, 20.0],
        );
        let result = reg.execute(&fixed, &moving, init).unwrap();

        let p = result.transform.parameters();
        let expected = [1.0, 0.0, 0.0, 1.0, tx, ty];
        for (k, (&got, &want)) in p.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "param {k}: got {got}, want {want} (full {p:?}, metric {})",
                result.metric_value
            );
        }
    }

    #[test]
    fn recovers_a_known_translation() {
        // Fixed blob centred at (20,20); moving blob shifted by (+3, −2).
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(100.0, 300);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}",
            result.metric_value,
            result.iterations
        );
        // The metric at the recovered transform is far below the initial mismatch.
        let initial = MeanSquaresMetric::new(&fixed, &moving)
            .unwrap()
            .evaluate(&TranslationTransform::new(vec![0.0, 0.0]), &CpuBackend)
            .value;
        assert!(
            result.metric_value < 0.05 * initial,
            "final {} not << initial {}",
            result.metric_value,
            initial
        );
    }

    #[test]
    fn recovers_a_translation_through_an_affine_transform() {
        // The moving image is a pure translation of the fixed, so the optimal
        // affine is (identity matrix, translation = shift). This exercises the
        // 6-parameter affine path — Jacobian and optimizer scales — end to end.
        // Matrix params (≈1) and translation params (≈image extent) need scales
        // to be optimized together, as ITK's ScalesEstimator provides.
        use sitk_transform::AffineTransform;
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(100.0, 1000)
            .set_optimizer_scales(vec![1600.0, 1600.0, 1600.0, 1600.0, 1.0, 1.0]);
        let init = AffineTransform::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![20.0, 20.0],
        );
        let result = reg.execute(&fixed, &moving, init).unwrap();

        let p = result.transform.parameters();
        // Matrix stays near identity; translation recovers the shift.
        let expected = [1.0, 0.0, 0.0, 1.0, tx, ty];
        for (k, (&got, &want)) in p.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "param {k}: got {got}, want {want} (full {p:?}, metric {})",
                result.metric_value
            );
        }
    }

    #[test]
    fn transform_dimension_mismatch_is_rejected() {
        let fixed = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let moving = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 1);
        let err = reg
            .execute(
                &fixed,
                &moving,
                TranslationTransform::new(vec![0.0, 0.0, 0.0]),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::TransformDimensionMismatch { .. }
        ));
    }

    #[test]
    fn wrong_length_scales_are_rejected() {
        let fixed = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let moving = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 1)
            .set_optimizer_scales(vec![1.0, 1.0, 1.0]); // 3 scales for a 2-param transform
        let err = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::ScalesLength {
                got: 3,
                expected: 2
            }
        ));
    }

    #[test]
    fn single_level_schedule_equals_no_schedule() {
        // A one-level pyramid with factor 1 and sigma 0 must reproduce the
        // default (no-schedule) single-resolution run exactly: shrinking by 1 is
        // an identity grid and smoothing by 0 is a no-op, so the resampled fixed
        // is bit-for-bit the original.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let run = |schedule: Option<(Vec<usize>, Vec<f64>)>| {
            let mut reg = ImageRegistrationMethod::new();
            reg.set_optimizer_scales_from_physical_shift()
                .set_optimizer_as_gradient_descent_estimated(300, EstimateLearningRate::Once);
            if let Some((sh, sg)) = schedule {
                reg.set_shrink_factors_per_level(sh)
                    .set_smoothing_sigmas_per_level(sg);
            }
            reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
                .unwrap()
                .transform
                .parameters()
        };

        let default = run(None);
        let one_level = run(Some((vec![1], vec![0.0])));
        assert_eq!(
            default, one_level,
            "one-level [1]/[0] pyramid diverged from the no-schedule run"
        );
    }

    #[test]
    fn recovers_translation_multiresolution() {
        // A 3-level pyramid (shrink [4,2,1], sigma [2,1,0]) recovers a known
        // translation to sub-voxel accuracy through the coarse-to-fine schedule.
        let (w, h, sigma, amp) = (48usize, 48usize, 6.0, 1.0);
        let (tx, ty) = (5.0f64, -3.0f64);
        let fixed = gaussian(w, h, 24.0, 24.0, sigma, amp);
        let moving = gaussian(w, h, 24.0 + tx, 24.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(200, EstimateLearningRate::Once)
            .set_shrink_factors_per_level(vec![4, 2, 1])
            .set_smoothing_sigmas_per_level(vec![2.0, 1.0, 0.0]);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        let e = ((p[0] - tx).powi(2) + (p[1] - ty).powi(2)).sqrt();
        assert!(
            e < 0.5,
            "multi-res recovered {p:?}, expected [{tx}, {ty}] (err {e}, metric {})",
            result.metric_value
        );
    }

    #[test]
    fn multiresolution_extends_capture_range() {
        // Sharp blobs (sigma 5) far apart (|offset| ≈ 21.6): at full resolution
        // they do not overlap, so the single-resolution metric gradient is ~0 and
        // the optimizer cannot move. Coarse smoothing makes the blobs overlap, so
        // the pyramid captures the alignment single resolution cannot.
        let (w, h, sigma, amp) = (64usize, 64usize, 5.0, 1.0);
        let (tx, ty) = (18.0f64, -12.0f64);
        let fixed = gaussian(w, h, 32.0, 32.0, sigma, amp);
        let moving = gaussian(w, h, 32.0 + tx, 32.0 + ty, sigma, amp);
        let err = |p: &[f64]| ((p[0] - tx).powi(2) + (p[1] - ty).powi(2)).sqrt();

        // Single resolution: stuck near the start (no overlap → no gradient).
        let mut single = ImageRegistrationMethod::new();
        single
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(400, EstimateLearningRate::Once);
        let single_err = err(&single
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap()
            .transform
            .parameters());
        assert!(
            single_err > 5.0,
            "single resolution unexpectedly captured this offset (err {single_err})"
        );

        // Multi resolution: captures the alignment to sub-voxel accuracy.
        let mut multi = ImageRegistrationMethod::new();
        multi
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(150, EstimateLearningRate::Once)
            .set_shrink_factors_per_level(vec![4, 2, 1])
            .set_smoothing_sigmas_per_level(vec![2.0, 1.0, 0.0]);
        let multi_err = err(&multi
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap()
            .transform
            .parameters());
        assert!(
            multi_err < 0.5,
            "multi resolution failed to capture the offset (err {multi_err})"
        );
    }

    #[test]
    fn pyramid_schedule_length_mismatch_is_rejected() {
        let fixed = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let moving = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 1)
            .set_shrink_factors_per_level(vec![4, 2, 1])
            .set_smoothing_sigmas_per_level(vec![2.0, 1.0]); // 3 shrink vs 2 sigma
        let err = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::PyramidScheduleLength {
                shrink: 3,
                sigma: 2
            }
        ));
    }
}
