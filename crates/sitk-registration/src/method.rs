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
//! full-resolution level. The metric is mean squares
//! (`itk::MeanSquaresImageToImageMetricv4`, the default) or Mattes mutual
//! information (`itk::MattesMutualInformationImageToImageMetricv4`, for
//! multi-modality), selected with
//! [`set_metric_as_mean_squares`](ImageRegistrationMethod::set_metric_as_mean_squares)
//! / [`set_metric_as_mattes_mutual_information`](ImageRegistrationMethod::set_metric_as_mattes_mutual_information);
//! interpolation is linear. The optimizer is either fixed-step gradient descent
//! (`itk::GradientDescentOptimizerv4`) or regular-step gradient descent
//! (`itk::RegularStepGradientDescentOptimizerv4`), the latter halving its step
//! on each overshoot so it converges cleanly at an already-registered pyramid
//! level and refines the finest one to high precision. More sampling strategies
//! come later.

use sitk_core::Image;
use sitk_filters::{recursive_gaussian, shrink};
use sitk_transform::{AffineTransform, Interpolator, ParametricTransform, ResampleImageFilter};

use crate::error::{RegistrationError, Result};
use crate::mattes::MattesMutualInformationMetric;
use crate::metric::{CpuBackend, MeanSquaresMetric, MetricBackend, MetricValue};
use crate::optimizer::{GradientDescentOptimizer, RegularStepGradientDescentOptimizer, StopReason};
use crate::scales::PhysicalShiftScales;

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

/// Which similarity metric the method optimizes. Selected via
/// [`ImageRegistrationMethod::set_metric_as_mean_squares`] /
/// [`ImageRegistrationMethod::set_metric_as_mattes_mutual_information`].
enum MetricKind {
    /// Mean squares (`itk::MeanSquaresImageToImageMetricv4`) — the default,
    /// suited to same-modality images with a linear intensity relationship.
    MeanSquares,
    /// Mattes mutual information
    /// (`itk::MattesMutualInformationImageToImageMetricv4`) with the given
    /// number of joint-histogram bins — suited to multi-modality registration.
    MattesMutualInformation { number_of_histogram_bins: usize },
}

/// A constructed metric for one resolution level, dispatching
/// [`evaluate`](Self::evaluate) and [`physical_shift_scales`] to the concrete
/// metric selected by [`MetricKind`]. Both expose the same `MetricValue`
/// interface, so the optimizer loop is metric-agnostic.
///
/// [`physical_shift_scales`]: Self::physical_shift_scales
enum ActiveMetric {
    MeanSquares(MeanSquaresMetric),
    Mattes(MattesMutualInformationMetric),
}

impl ActiveMetric {
    /// Value + parameter-derivative at `transform`. The mean-squares reduction
    /// runs through the (GPU-swappable) `backend`; Mattes MI is CPU-only and
    /// ignores it.
    fn evaluate(
        &self,
        transform: &dyn ParametricTransform,
        backend: &dyn MetricBackend,
    ) -> MetricValue {
        match self {
            ActiveMetric::MeanSquares(m) => m.evaluate(transform, backend),
            ActiveMetric::Mattes(m) => m.evaluate(transform),
        }
    }

    /// Physical-shift scale/learning-rate estimator over the fixed samples.
    fn physical_shift_scales(&self, transform: &dyn ParametricTransform) -> PhysicalShiftScales {
        match self {
            ActiveMetric::MeanSquares(m) => m.physical_shift_scales(transform),
            ActiveMetric::Mattes(m) => m.physical_shift_scales(transform),
        }
    }
}

/// Which optimizer the method drives. The learning-rate estimation
/// ([`LearningRateMode`]) and parameter scales ([`ScalesMode`]) apply to
/// whichever is selected.
enum OptimizerKind {
    /// Fixed-step gradient descent (`itk::GradientDescentOptimizerv4`).
    GradientDescent(GradientDescentOptimizer),
    /// Regular-step gradient descent (`itk::RegularStepGradientDescentOptimizerv4`):
    /// a fixed step length halved on each direction reversal, converging to
    /// `minimum_step_length` and stopping cleanly at an already-converged level.
    RegularStep(RegularStepGradientDescentOptimizer),
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
    optimizer: OptimizerKind,
    metric_kind: MetricKind,
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
            optimizer: OptimizerKind::GradientDescent(GradientDescentOptimizer::new(1.0, 100)),
            metric_kind: MetricKind::MeanSquares,
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
    /// iterations). Switch to Mattes mutual information for multi-modality with
    /// [`set_metric_as_mattes_mutual_information`](Self::set_metric_as_mattes_mutual_information).
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
        self.optimizer = OptimizerKind::GradientDescent(GradientDescentOptimizer::new(
            learning_rate,
            iterations,
        ));
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
        let mut optimizer = GradientDescentOptimizer::new(1.0, iterations);
        if estimate == EstimateLearningRate::EachIteration {
            // A non-shrinking step schedule needs value-plateau monitoring to
            // stop; the once schedule stops via the min-step tolerance.
            optimizer.set_convergence(CONVERGENCE_WINDOW_SIZE, MINIMUM_CONVERGENCE_VALUE);
        }
        self.optimizer = OptimizerKind::GradientDescent(optimizer);
        self.learning_rate_mode = LearningRateMode::Estimate(estimate);
        self
    }

    /// Use regular-step gradient descent
    /// (`itk::RegularStepGradientDescentOptimizerv4`) with a caller-supplied
    /// initial step length, minimum step length, and iteration cap. Each step
    /// moves a fixed length in the gradient's direction; the length is halved
    /// whenever the gradient reverses (an overshoot) and iteration stops when it
    /// falls below `minimum_step` (or the scaled gradient magnitude falls below
    /// `gradient_magnitude_tolerance` — a stationary point). Unlike a fixed-rate
    /// gradient descent it refines toward `minimum_step` precision without
    /// hand-timing the rate; `gradient_magnitude_tolerance` (ITK's default is
    /// `1e-4`) sets how flat the gradient must be to declare convergence, and is
    /// often the binding stop on smooth objectives — lower it for finer results.
    pub fn set_optimizer_as_regular_step_gradient_descent(
        &mut self,
        learning_rate: f64,
        minimum_step: f64,
        iterations: usize,
        gradient_magnitude_tolerance: f64,
    ) -> &mut Self {
        let mut optimizer =
            RegularStepGradientDescentOptimizer::new(learning_rate, minimum_step, iterations);
        optimizer.set_gradient_magnitude_tolerance(gradient_magnitude_tolerance);
        self.optimizer = OptimizerKind::RegularStep(optimizer);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use regular-step gradient descent whose initial step length is
    /// **estimated** from physical shift (bounded to ~one voxel, as ITK's
    /// learning-rate estimation does), then halved on overshoot toward
    /// `minimum_step`. This is the robust choice for a multi-resolution pyramid:
    /// its `gradient_magnitude_tolerance` stops a level that restarts already
    /// converged instead of taking a runaway fixed-rate step, and it still
    /// refines the finest level toward `minimum_step` precision. Because the
    /// estimated initial step is ~one voxel regardless of how small the restart
    /// gradient is, `gradient_magnitude_tolerance` (ITK's default `1e-4`) — not
    /// `minimum_step` — is usually what bounds the finest precision on smooth
    /// images; lower it to refine further. `estimate` selects once
    /// ([`EstimateLearningRate::Once`], the usual choice) or per-iteration
    /// re-estimation. Pair with [`set_optimizer_scales_from_physical_shift`].
    ///
    /// [`set_optimizer_scales_from_physical_shift`]:
    /// Self::set_optimizer_scales_from_physical_shift
    pub fn set_optimizer_as_regular_step_gradient_descent_estimated(
        &mut self,
        minimum_step: f64,
        iterations: usize,
        gradient_magnitude_tolerance: f64,
        estimate: EstimateLearningRate,
    ) -> &mut Self {
        // The stored step-length scale is a placeholder overwritten by the
        // estimate; RegularStep stops via its own step/gradient tolerances, so
        // no value-plateau monitoring is needed even for the each-iteration mode.
        let mut optimizer = RegularStepGradientDescentOptimizer::new(1.0, minimum_step, iterations);
        optimizer.set_gradient_magnitude_tolerance(gradient_magnitude_tolerance);
        self.optimizer = OptimizerKind::RegularStep(optimizer);
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
    /// For regular-step gradient descent this is its `minimum_step_length`.
    pub fn set_min_step_tolerance(&mut self, tol: f64) -> &mut Self {
        match &mut self.optimizer {
            OptimizerKind::GradientDescent(gd) => {
                gd.set_min_step_tolerance(tol);
            }
            OptimizerKind::RegularStep(rs) => {
                rs.set_minimum_step_length(tol);
            }
        }
        self
    }

    /// Use the **mean-squares** metric (`itk::MeanSquaresImageToImageMetricv4`),
    /// the default. Suited to same-modality images related by a roughly linear
    /// intensity map. SimpleITK `SetMetricAsMeanSquares`.
    pub fn set_metric_as_mean_squares(&mut self) -> &mut Self {
        self.metric_kind = MetricKind::MeanSquares;
        self
    }

    /// Use the **Mattes mutual-information** metric
    /// (`itk::MattesMutualInformationImageToImageMetricv4`) with
    /// `number_of_histogram_bins` joint-histogram bins (SimpleITK's default is
    /// 50). Suited to **multi-modality** registration — images related by an
    /// arbitrary invertible intensity map, where mean squares fails. Errors at
    /// [`execute`](Self::execute) time if fewer than five bins are requested or
    /// an image is constant. SimpleITK `SetMetricAsMattesMutualInformation`.
    pub fn set_metric_as_mattes_mutual_information(
        &mut self,
        number_of_histogram_bins: usize,
    ) -> &mut Self {
        self.metric_kind = MetricKind::MattesMutualInformation {
            number_of_histogram_bins,
        };
        self
    }

    /// Replace the metric compute backend (the GPU seam). Defaults to
    /// [`CpuBackend`]; a CUDA/`wgpu` backend implements the same
    /// [`MetricBackend`] trait.
    pub fn set_metric_backend(&mut self, backend: Box<dyn MetricBackend>) -> &mut Self {
        self.backend = backend;
        self
    }

    /// Construct the metric selected by [`MetricKind`] for one resolution
    /// level's fixed/moving pair.
    fn build_metric(&self, fixed: &Image, moving: &Image) -> Result<ActiveMetric> {
        match &self.metric_kind {
            MetricKind::MeanSquares => Ok(ActiveMetric::MeanSquares(MeanSquaresMetric::new(
                fixed, moving,
            )?)),
            MetricKind::MattesMutualInformation {
                number_of_histogram_bins,
            } => Ok(ActiveMetric::Mattes(MattesMutualInformationMetric::new(
                fixed,
                moving,
                *number_of_histogram_bins,
            )?)),
        }
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
    /// Smoothing uses the bit-exact recursive Gaussian
    /// ([`recursive_gaussian`](sitk_filters::recursive_gaussian()), the
    /// Deriche/Farnebäck IIR), matching ITK's
    /// `SmoothingRecursiveGaussianImageFilter`. Both images are smoothed at full
    /// resolution, so the recursive filter's ≥4-pixels-per-smoothed-axis
    /// requirement bites only on a pathologically small input (a level with
    /// `sigma == 0` is a no-op and imposes nothing).
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
        let smoothed_fixed = recursive_gaussian(fixed, sigma)?;
        let coarse_grid = shrink(&smoothed_fixed, factors)?;
        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&coarse_grid)
            .set_interpolator(Interpolator::Linear);
        let fixed_level = resampler.execute(&smoothed_fixed, &AffineTransform::identity(dim))?;
        let moving_level = recursive_gaussian(moving, sigma)?;
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
        let metric = self.build_metric(fixed, moving)?;
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

        let scaled = |grad: &[f64]| -> Vec<f64> {
            grad.iter()
                .zip(scales.iter())
                .map(|(&g, &s)| g / s)
                .collect()
        };
        // The objective the optimizer minimizes: set the transform's parameters
        // and evaluate the metric. Duplicated per branch because it borrows
        // `transform` mutably and each branch drives its own optimizer call.
        macro_rules! objective {
            () => {
                |p: &[f64]| {
                    transform.set_parameters(p);
                    let m = metric.evaluate(&transform, backend);
                    (m.value, m.derivative)
                }
            };
        }

        let result = match &self.optimizer {
            OptimizerKind::GradientDescent(gd) => {
                let mut optimizer = gd.clone();
                optimizer.set_scales(scales.clone());
                match self.learning_rate_mode {
                    // Caller-supplied fixed rate.
                    LearningRateMode::Manual => optimizer.optimize(start, objective!()),
                    // Rate estimated once from the initial gradient, then held
                    // fixed so steps shrink with the gradient (ITK's default).
                    // Each step is also capped at the estimator's one-voxel
                    // maximum shift: a level that restarts from a near-converged
                    // transform has a ~0 initial gradient, which makes the
                    // once-estimated rate enormous and the *next* step (once the
                    // gradient grows again) explode. The cap makes "no step
                    // exceeds one voxel" hold by construction. It is inactive
                    // whenever the fixed rate already bounds the step — which is
                    // every step of a monotonically converging run, since the
                    // once-rate is exactly the per-step rate at the initial
                    // gradient and only grows as the gradient shrinks — so
                    // single-resolution runs are unchanged. (Regular-step descent
                    // removes the need for this cap entirely.)
                    LearningRateMode::Estimate(EstimateLearningRate::Once) => {
                        let est = estimator.as_ref().unwrap();
                        transform.set_parameters(&start);
                        let m0 = metric.evaluate(&transform, backend);
                        let lr_once = est.estimate_learning_rate(&scaled(&m0.derivative));
                        optimizer.optimize_with_lr(start, objective!(), |grad| {
                            lr_once.min(est.estimate_learning_rate(&scaled(grad)))
                        })
                    }
                    // Rate re-estimated each iteration from the current gradient;
                    // the non-shrinking step schedule stops via convergence
                    // monitoring (enabled on the optimizer by the estimated-mode
                    // setter).
                    LearningRateMode::Estimate(EstimateLearningRate::EachIteration) => {
                        let est = estimator.as_ref().unwrap();
                        optimizer.optimize_with_lr(start, objective!(), |grad| {
                            est.estimate_learning_rate(&scaled(grad))
                        })
                    }
                }
            }
            OptimizerKind::RegularStep(rs) => {
                let mut optimizer = rs.clone();
                optimizer.set_scales(scales.clone());
                match self.learning_rate_mode {
                    // Caller-supplied fixed initial step length.
                    LearningRateMode::Manual => optimizer.optimize(start, objective!()),
                    // ITK's RegularStep sets `m_LearningRate =
                    // (maxStep/stepScale)·‖g₀‖` once, giving a first step of about
                    // one voxel; the fixed step length then halves on overshoot
                    // toward `minimum_step_length`. No step cap is needed: the
                    // gradient-magnitude tolerance stops a near-converged restart
                    // before any runaway step.
                    LearningRateMode::Estimate(EstimateLearningRate::Once) => {
                        let est = estimator.as_ref().unwrap();
                        transform.set_parameters(&start);
                        let m0 = metric.evaluate(&transform, backend);
                        let scaled0 = scaled(&m0.derivative);
                        let grad_mag_0 = scaled0.iter().map(|g| g * g).sum::<f64>().sqrt();
                        optimizer
                            .set_learning_rate(est.estimate_learning_rate(&scaled0) * grad_mag_0);
                        optimizer.optimize(start, objective!())
                    }
                    // Step-length scale re-estimated each iteration from the
                    // current gradient (`(maxStep/stepScale)·‖g‖`); relaxation and
                    // the step/gradient tolerances still govern stopping. The
                    // closure receives the already-scaled gradient.
                    LearningRateMode::Estimate(EstimateLearningRate::EachIteration) => {
                        let est = estimator.as_ref().unwrap();
                        optimizer.optimize_with_lr(start, objective!(), |scaled_grad| {
                            let gm = scaled_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                            est.estimate_learning_rate(scaled_grad) * gm
                        })
                    }
                }
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
    fn bspline_recovers_a_translation_as_a_deformation_field() {
        // A cubic B-spline free-form transform is over-parameterised for a
        // global translation, but a constant coefficient field represents one
        // exactly (the weights sum to 1), so the deformable registration must
        // recover it end-to-end: the transform maps the fixed blob centre onto
        // the moving one, and the metric drops far below its identity baseline.
        // This exercises the whole deformable path — B-spline weights, the
        // per-control-point Jacobian, physical-shift scales over ~100
        // parameters, and the optimiser over the full coefficient vector.
        use sitk_transform::{BSplineTransform, Transform};

        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (cx, cy) = (20.0f64, 20.0f64);
        let (tx, ty) = (2.0f64, -1.5f64);
        let fixed = gaussian(w, h, cx, cy, sigma, amp);
        // fixed(x) ≈ moving(T(x)) is minimised when T(c) = c + (tx, ty).
        let moving = gaussian(w, h, cx + tx, cy + ty, sigma, amp);

        let bspline = BSplineTransform::from_image_domain(&fixed, &[4, 4]).unwrap();
        // Identity-baseline mean-squares, for comparison with the final metric.
        let baseline = MeanSquaresMetric::new(&fixed, &moving)
            .unwrap()
            .evaluate(&bspline, &CpuBackend)
            .value;

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-4,
                200,
                1e-6,
                EstimateLearningRate::Once,
            );
        let result = reg.execute(&fixed, &moving, bspline).unwrap();

        let mapped = result.transform.transform_point(&[cx, cy]);
        assert!(
            (mapped[0] - (cx + tx)).abs() < 0.5 && (mapped[1] - (cy + ty)).abs() < 0.5,
            "blob centre mapped to {mapped:?}, expected {:?}; metric {} (baseline {baseline}), iters {}",
            [cx + tx, cy + ty],
            result.metric_value,
            result.iterations
        );
        assert!(
            result.metric_value < 0.1 * baseline,
            "metric {} not below 0.1×baseline {baseline}",
            result.metric_value
        );
    }

    #[test]
    fn mattes_mi_recovers_a_translation_under_contrast_inversion() {
        // The multi-modality case: the moving image is the fixed blob shifted
        // AND contrast-inverted (a dark blob on a bright field where the fixed
        // is a bright blob on a dark field). Mean squares wants M ≈ F and is
        // maximally confused here; Mattes mutual information sees the intensity
        // dependence regardless of the (inverting) intensity map and recovers
        // the shift.
        let (w, h, sigma, amp) = (48usize, 48usize, 6.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 24.0, 24.0, sigma, amp);
        // moving(p) = amp − blob(p; centre shifted by (tx, ty)): inverted contrast.
        let bright = gaussian(w, h, 24.0 + tx, 24.0 + ty, sigma, amp);
        let moving = Image::from_vec(
            &[w, h],
            bright.to_f64_vec().iter().map(|v| amp - v).collect(),
        )
        .unwrap();

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mattes_mutual_information(32)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                0.01,
                200,
                1e-6,
                EstimateLearningRate::Once,
            );
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 0.5 && (p[1] - ty).abs() < 0.5,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}",
            result.metric_value,
            result.iterations
        );
    }

    #[test]
    fn bspline_mattes_recovers_a_deformable_shift_under_contrast_inversion() {
        // Deformable AND multi-modality together: the moving image is the fixed
        // blob shifted AND contrast-inverted, and the transform is a free-form
        // cubic B-spline. Mean squares is defeated by the inversion; Mattes
        // mutual information drives the deformation, and because a constant
        // coefficient field represents the shift exactly, the recovered warp maps
        // the fixed blob centre onto the moving one while mutual information
        // improves over the identity. A BSpline is !HasLocalSupport in ITK, so
        // this runs through the metric's ordinary global-support derivative path.
        use sitk_transform::{BSplineTransform, Transform};

        let (w, h, sigma, amp) = (40usize, 40usize, 6.0, 1.0);
        let (cx, cy) = (20.0f64, 20.0f64);
        let (tx, ty) = (2.0f64, -1.5f64);
        let fixed = gaussian(w, h, cx, cy, sigma, amp);
        let bright = gaussian(w, h, cx + tx, cy + ty, sigma, amp);
        let moving = Image::from_vec(
            &[w, h],
            bright.to_f64_vec().iter().map(|v| amp - v).collect(),
        )
        .unwrap();

        // Baseline mutual information at the identity deformation.
        let identity = BSplineTransform::from_image_domain(&fixed, &[4, 4]).unwrap();
        let baseline = crate::mattes::MattesMutualInformationMetric::new(&fixed, &moving, 32)
            .unwrap()
            .evaluate(&identity)
            .value;

        let bspline = BSplineTransform::from_image_domain(&fixed, &[4, 4]).unwrap();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mattes_mutual_information(32)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                0.01,
                200,
                1e-6,
                EstimateLearningRate::Once,
            );
        let result = reg.execute(&fixed, &moving, bspline).unwrap();

        let mapped = result.transform.transform_point(&[cx, cy]);
        assert!(
            (mapped[0] - (cx + tx)).abs() < 0.6 && (mapped[1] - (cy + ty)).abs() < 0.6,
            "blob centre mapped to {mapped:?}, expected {:?}; metric {} (baseline {baseline}), iters {}",
            [cx + tx, cy + ty],
            result.metric_value,
            result.iterations
        );
        assert!(
            result.metric_value < baseline,
            "mutual information did not improve: metric {} vs baseline {baseline}",
            result.metric_value
        );
    }

    #[test]
    fn displacement_field_recovers_a_translation() {
        // A dense displacement field has one free vector per pixel — the most
        // flexible deformable transform. Registering a translated blob drives the
        // field to align the images (each pixel nulls its own intensity
        // residual), dropping the metric far below the identity baseline; on the
        // blob's steep flank, where the gradient carries signal, the recovered
        // displacement approaches the true translation.
        use sitk_transform::{DisplacementFieldTransform, Transform};

        let (w, h, sigma, amp) = (20usize, 20usize, 4.0, 1.0);
        let (cx, cy) = (10.0f64, 10.0f64);
        let (tx, ty) = (2.0f64, -1.0f64);
        let fixed = gaussian(w, h, cx, cy, sigma, amp);
        let moving = gaussian(w, h, cx + tx, cy + ty, sigma, amp);

        let identity = DisplacementFieldTransform::from_image_domain(&fixed).unwrap();
        let baseline = MeanSquaresMetric::new(&fixed, &moving)
            .unwrap()
            .evaluate(&identity, &CpuBackend)
            .value;

        let field = DisplacementFieldTransform::from_image_domain(&fixed).unwrap();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-4,
                200,
                1e-7,
                EstimateLearningRate::Once,
            );
        let result = reg.execute(&fixed, &moving, field).unwrap();

        // The field aligned the images: metric far below the identity baseline.
        assert!(
            result.metric_value < 0.15 * baseline,
            "metric {} not below 0.15×baseline {baseline}",
            result.metric_value
        );
        // On the blob's flank (one sigma right of centre, where the x-gradient
        // carries signal) the recovered x-displacement approaches the true shift.
        let flank = [cx + sigma, cy];
        let mapped = result.transform.transform_point(&flank);
        assert!(
            (mapped[0] - (flank[0] + tx)).abs() < 0.7,
            "flank mapped to {mapped:?}, expected x≈{}; metric {} (baseline {baseline})",
            flank[0] + tx,
            result.metric_value
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

    /// Sum of two Gaussian blobs at `c1`, `c2` — a rotationally asymmetric
    /// pattern, so a rigid rotation (not just a translation) is needed to align
    /// two such images.
    fn two_blobs(w: usize, h: usize, c1: (f64, f64), c2: (f64, f64), sigma: f64) -> Image {
        let s2 = 2.0 * sigma * sigma;
        let mut v = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let g = |c: (f64, f64)| {
                    let (dx, dy) = (x as f64 - c.0, y as f64 - c.1);
                    (-(dx * dx + dy * dy) / s2).exp()
                };
                v[y * w + x] = g(c1) + g(c2);
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn recovers_a_rigid_euler2d_rotation_and_translation() {
        // Ground truth: rotate the fixed features by +theta about the image
        // centre, then translate. The optimal Euler2D that aligns moving back
        // onto fixed is exactly that (theta, tx, ty), since moving has a feature
        // at p' = R(theta)(p − c) + c + t wherever fixed has one at p.
        use sitk_transform::Euler2DTransform;
        let (w, h) = (48usize, 48usize);
        let (cx, cy) = (24.0f64, 24.0f64);
        let sigma = 4.0;
        let (a, b) = ((34.0, 24.0), (24.0, 31.0)); // 10 px right, 7 px above centre

        let theta = 0.08f64; // ~4.6°
        let (tx, ty) = (1.0f64, -0.5f64);
        let rot = |p: (f64, f64)| {
            let (dx, dy) = (p.0 - cx, p.1 - cy);
            let (ct, st) = (theta.cos(), theta.sin());
            (cx + ct * dx - st * dy + tx, cy + st * dx + ct * dy + ty)
        };

        let fixed = two_blobs(w, h, a, b, sigma);
        let moving = two_blobs(w, h, rot(a), rot(b), sigma);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                300,
                1e-8,
                EstimateLearningRate::Once,
            );
        let init = Euler2DTransform::new(0.0, [0.0, 0.0], [cx, cy]);
        let result = reg.execute(&fixed, &moving, init).unwrap();

        let p = result.transform.parameters(); // [angle, tx, ty]
        assert!(
            (p[0] - theta).abs() < 1e-2,
            "angle: got {}, want {theta} (full {p:?}, metric {})",
            p[0],
            result.metric_value
        );
        assert!(
            (p[1] - tx).abs() < 5e-2 && (p[2] - ty).abs() < 5e-2,
            "translation: got ({}, {}), want ({tx}, {ty}) (metric {})",
            p[1],
            p[2],
            result.metric_value
        );

        // The metric at the recovered transform is far below the initial mismatch.
        let initial = MeanSquaresMetric::new(&fixed, &moving)
            .unwrap()
            .evaluate(
                &Euler2DTransform::new(0.0, [0.0, 0.0], [cx, cy]),
                &CpuBackend,
            )
            .value;
        assert!(
            result.metric_value < 0.05 * initial,
            "final {} not << initial {}",
            result.metric_value,
            initial
        );
    }

    #[test]
    fn recovers_a_similarity2d_scale_rotation_and_translation() {
        // Ground truth: scale + rotate the fixed features about the image centre,
        // then translate: p' = s·R(theta)(p − c) + c + t. The optimal Similarity2D
        // aligning moving back onto fixed is exactly (s, theta, tx, ty).
        use sitk_transform::Similarity2DTransform;
        let (w, h) = (48usize, 48usize);
        let (cx, cy) = (24.0f64, 24.0f64);
        let sigma = 4.0;
        let (a, b) = ((34.0, 24.0), (24.0, 31.0)); // 10 px right, 7 px above centre

        let scale = 1.1f64;
        let theta = 0.06f64; // ~3.4°
        let (tx, ty) = (0.8f64, -0.4f64);
        let map = |p: (f64, f64)| {
            let (dx, dy) = (p.0 - cx, p.1 - cy);
            let (ct, st) = (theta.cos(), theta.sin());
            (
                cx + scale * (ct * dx - st * dy) + tx,
                cy + scale * (st * dx + ct * dy) + ty,
            )
        };

        let fixed = two_blobs(w, h, a, b, sigma);
        // moving = fixed ∘ S⁻¹, so its blobs sit at the transformed centres AND
        // are widened by the isotropic scale (an s-times-wider Gaussian). Placing
        // same-width blobs would make the exact similarity a poor fit and the
        // recovered scale biased.
        let moving = two_blobs(w, h, map(a), map(b), sigma * scale);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                400,
                1e-9,
                EstimateLearningRate::Once,
            );
        let init = Similarity2DTransform::new(1.0, 0.0, [0.0, 0.0], [cx, cy]);
        let result = reg.execute(&fixed, &moving, init).unwrap();

        let p = result.transform.parameters(); // [scale, angle, tx, ty]
        assert!(
            (p[0] - scale).abs() < 2e-2,
            "scale: got {}, want {scale} (full {p:?}, metric {})",
            p[0],
            result.metric_value
        );
        assert!(
            (p[1] - theta).abs() < 1e-2,
            "angle: got {}, want {theta} (full {p:?}, metric {})",
            p[1],
            result.metric_value
        );
        assert!(
            (p[2] - tx).abs() < 5e-2 && (p[3] - ty).abs() < 5e-2,
            "translation: got ({}, {}), want ({tx}, {ty}) (metric {})",
            p[2],
            p[3],
            result.metric_value
        );
    }

    /// A `n³` volume with an isotropic Gaussian blob (width `sigma`) at each
    /// listed physical (== index) centre; values sum across blobs.
    fn blobs_3d(n: usize, centers: &[[f64; 3]], sigma: f64) -> Image {
        let s2 = 2.0 * sigma * sigma;
        let mut v = vec![0.0f64; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = [x as f64, y as f64, z as f64];
                    let mut acc = 0.0;
                    for c in centers {
                        let d2: f64 = (0..3).map(|k| (p[k] - c[k]).powi(2)).sum();
                        acc += (-d2 / s2).exp();
                    }
                    v[(z * n + y) * n + x] = acc;
                }
            }
        }
        Image::from_vec(&[n, n, n], v).unwrap()
    }

    #[test]
    fn recovers_a_rigid_euler3d_rotation_and_translation() {
        // Ground truth: rotate the fixed features about the volume centre by a
        // known Euler3D, then translate; the optimal Euler3D aligning moving back
        // onto fixed is exactly that transform. Three blobs on orthogonal axes at
        // distinct radii break all rotational symmetry, so every angle is
        // observable. Rotation preserves the isotropic blob width, so the moving
        // blobs keep sigma (no scale correction, unlike a similarity).
        use sitk_transform::{Euler3DTransform, Transform};
        let n = 20usize;
        let c = [10.0f64, 10.0, 10.0];
        let sigma = 2.0;
        let feats = [[15.0, 10.0, 10.0], [10.0, 14.0, 10.0], [10.0, 10.0, 13.0]];

        let gt = Euler3DTransform::new(0.06, -0.05, 0.08, [0.7, -0.5, 0.4], c);
        let moved: Vec<[f64; 3]> = feats
            .iter()
            .map(|f| {
                let m = gt.transform_point(f);
                [m[0], m[1], m[2]]
            })
            .collect();

        let fixed = blobs_3d(n, &feats, sigma);
        let moving = blobs_3d(n, &moved, sigma);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                150,
                1e-8,
                EstimateLearningRate::Once,
            );
        let init = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], c);
        let result = reg.execute(&fixed, &moving, init).unwrap();

        let p = result.transform.parameters(); // [ax, ay, az, tx, ty, tz]
        let want = gt.parameters();
        for k in 0..3 {
            assert!(
                (p[k] - want[k]).abs() < 2e-2,
                "angle {k}: got {}, want {} (full {p:?}, metric {})",
                p[k],
                want[k],
                result.metric_value
            );
        }
        for k in 3..6 {
            assert!(
                (p[k] - want[k]).abs() < 5e-2,
                "translation {k}: got {}, want {} (full {p:?}, metric {})",
                p[k],
                want[k],
                result.metric_value
            );
        }
    }

    #[test]
    fn recovers_a_rigid_versor3d_rotation_and_translation() {
        // Same three-blob volume as the Euler3D test, but the ground-truth
        // rotation is a versor; the optimal VersorRigid3D recovers its right part
        // and translation. Rotation preserves the isotropic blob width.
        use sitk_transform::{Transform, VersorRigid3DTransform};
        let n = 20usize;
        let c = [10.0f64, 10.0, 10.0];
        let sigma = 2.0;
        let feats = [[15.0, 10.0, 10.0], [10.0, 14.0, 10.0], [10.0, 10.0, 13.0]];

        let gt = VersorRigid3DTransform::new(0.05, -0.04, 0.06, [0.7, -0.5, 0.4], c);
        let moved: Vec<[f64; 3]> = feats
            .iter()
            .map(|f| {
                let m = gt.transform_point(f);
                [m[0], m[1], m[2]]
            })
            .collect();

        let fixed = blobs_3d(n, &feats, sigma);
        let moving = blobs_3d(n, &moved, sigma);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                150,
                1e-8,
                EstimateLearningRate::Once,
            );
        let init = VersorRigid3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], c);
        let result = reg.execute(&fixed, &moving, init).unwrap();

        let p = result.transform.parameters(); // [vx, vy, vz, tx, ty, tz]
        let want = gt.parameters();
        for k in 0..3 {
            assert!(
                (p[k] - want[k]).abs() < 2e-2,
                "versor {k}: got {}, want {} (full {p:?}, metric {})",
                p[k],
                want[k],
                result.metric_value
            );
        }
        for k in 3..6 {
            assert!(
                (p[k] - want[k]).abs() < 5e-2,
                "translation {k}: got {}, want {} (full {p:?}, metric {})",
                p[k],
                want[k],
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
    fn regular_step_recovers_a_translation() {
        // Regular-step gradient descent with an estimated initial step recovers
        // the shift and stops at a stationary point (not the iteration cap),
        // refining to high precision by halving the step on each overshoot.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                300,
                1e-8,
                EstimateLearningRate::Once,
            );
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-4 && (p[1] - ty).abs() < 1e-4,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}",
            result.metric_value,
            result.iterations
        );
        assert_ne!(
            result.stop_reason,
            StopReason::MaxIterations,
            "regular-step run hit the iteration cap instead of converging"
        );
    }

    #[test]
    fn regular_step_with_a_manual_learning_rate_recovers_a_translation() {
        // The manual (un-estimated) regular-step path: a fixed initial step
        // length of two voxels, halved on overshoot, still recovers the shift.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_regular_step_gradient_descent(2.0, 1e-6, 300, 1e-8);
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
    fn regular_step_multiresolution_reaches_higher_precision_than_gradient_descent() {
        // The regular-step optimizer closes the finest-level precision gap of
        // the estimate-once gradient descent on the pyramid. On a cleanly
        // registerable pair both converge, but the fixed-step-with-relaxation
        // schedule reaches far below the gradient-descent result at the same
        // iteration budget, and stops at a stationary point rather than the cap.
        let (w, h, sigma, amp) = (48usize, 48usize, 6.0, 1.0);
        let (tx, ty) = (5.0f64, -3.0f64);
        let fixed = gaussian(w, h, 24.0, 24.0, sigma, amp);
        let moving = gaussian(w, h, 24.0 + tx, 24.0 + ty, sigma, amp);
        let err = |p: &[f64]| ((p[0] - tx).powi(2) + (p[1] - ty).powi(2)).sqrt();

        let mut gd = ImageRegistrationMethod::new();
        gd.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(200, EstimateLearningRate::Once)
            .set_shrink_factors_per_level(vec![4, 2, 1])
            .set_smoothing_sigmas_per_level(vec![2.0, 1.0, 0.0]);
        let gd_err = err(&gd
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap()
            .transform
            .parameters());

        let mut rs = ImageRegistrationMethod::new();
        rs.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                200,
                1e-7,
                EstimateLearningRate::Once,
            )
            .set_shrink_factors_per_level(vec![4, 2, 1])
            .set_smoothing_sigmas_per_level(vec![2.0, 1.0, 0.0]);
        let result = rs
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();
        let rs_err = err(&result.transform.parameters());

        assert!(
            rs_err < 1e-3,
            "regular-step multi-res err {rs_err} not below 1e-3 (metric {})",
            result.metric_value
        );
        assert!(
            rs_err < gd_err,
            "regular-step err {rs_err} not below gradient-descent err {gd_err}"
        );
        assert_ne!(
            result.stop_reason,
            StopReason::MaxIterations,
            "regular-step finest level hit the iteration cap instead of converging"
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
