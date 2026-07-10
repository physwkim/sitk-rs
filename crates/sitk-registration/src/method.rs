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
//! interpolation is linear. The optimizer is fixed-step gradient descent
//! (`itk::GradientDescentOptimizerv4`), regular-step gradient descent
//! (`itk::RegularStepGradientDescentOptimizerv4`, halving its step on each
//! overshoot so it converges cleanly at an already-registered pyramid level and
//! refines the finest one to high precision), or bound-constrained limited-memory
//! BFGS (`itk::LBFGSBOptimizerv4`, via
//! [`set_optimizer_as_lbfgsb`](ImageRegistrationMethod::set_optimizer_as_lbfgsb) —
//! a line-search quasi-Newton method that ignores parameter scales and drives the
//! raw metric gradient, with optional per-parameter bounds). More sampling
//! strategies come later.
//!
//! # The virtual domain and the three transforms
//!
//! The metric does not compare the two images directly. It samples a **virtual
//! reference domain** — a grid, unrelated to either image, set by
//! [`set_virtual_domain`] / [`set_virtual_domain_from_image`] and defaulting to
//! the fixed image's own grid — and at each virtual point `x` it compares
//!
//! ```text
//! F( fixed_initial(x) )   against   M( moving_initial( optimized(x) ) )
//! ```
//!
//! Only `optimized` — the transform set by [`set_initial_transform`] — has its
//! parameters driven by the optimizer. The other two are fixed context:
//! `moving_initial` ([`set_moving_initial_transform`]) is the alignment an
//! earlier registration stage already found, and `fixed_initial`
//! ([`set_fixed_initial_transform`]) relocates the fixed image's sample points.
//! Because the two sit on opposite sides of the comparison, the same
//! displacement applied to each moves the optimum in opposite directions.
//!
//! [`metric_evaluate`] runs that comparison once, with no optimizer and no
//! pyramid, and returns the metric value —
//! `itk::simple::ImageRegistrationMethod::MetricEvaluate`.
//!
//! [`set_virtual_domain`]: ImageRegistrationMethod::set_virtual_domain
//! [`set_virtual_domain_from_image`]: ImageRegistrationMethod::set_virtual_domain_from_image
//! [`set_initial_transform`]: ImageRegistrationMethod::set_initial_transform
//! [`set_moving_initial_transform`]: ImageRegistrationMethod::set_moving_initial_transform
//! [`set_fixed_initial_transform`]: ImageRegistrationMethod::set_fixed_initial_transform
//! [`metric_evaluate`]: ImageRegistrationMethod::metric_evaluate

use sitk_core::Image;
use sitk_filters::{recursive_gaussian, shrink};
use sitk_transform::{
    AffineTransform, CompositeTransform, Interpolator, ParametricTransform, ResampleImageFilter,
    Transform, TransformBase, TranslationTransform,
};

use crate::ants_correlation::AntsNeighborhoodCorrelationMetric;
use crate::correlation::CorrelationMetric;
use crate::demons::DemonsMetric;
use crate::error::{RegistrationError, Result};
use crate::gradient_free::{
    AmoebaOptimizer, ExhaustiveOptimizer, OnePlusOneEvolutionaryOptimizer, PowellOptimizer,
};
use crate::joint_histogram::JointHistogramMutualInformationMetric;
use crate::lbfgs2::LBFGS2Optimizer;
use crate::lbfgsb::LBFGSBOptimizer;
use crate::mattes::MattesMutualInformationMetric;
use crate::metric::{
    CpuBackend, FixedSamples, MeanSquaresMetric, MetricBackend, MetricValue, MovingImage,
    SamplingStrategy,
};
use crate::optimizer::{
    ConjugateGradientLineSearchOptimizer, GradientDescentLineSearchOptimizer,
    GradientDescentOptimizer, Objective, RegularStepGradientDescentOptimizer, StopReason,
};
use crate::scales::{
    DEFAULT_CENTRAL_REGION_RADIUS, DEFAULT_SMALL_PARAMETER_VARIATION, ScalesEstimator,
    ScalesEstimatorKind,
};

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

/// ITK's `m_WeightsAreIdentity` tolerance
/// (`itkObjectToObjectOptimizerBase.cxx:157`).
const WEIGHTS_IDENTITY_TOLERANCE: f64 = 1e-4;

/// ITK's `m_WeightsAreIdentity` test (`itkObjectToObjectOptimizerBase.cxx:143-166`):
/// an unset (empty) weights array is identity, and so is one whose every entry
/// is within [`WEIGHTS_IDENTITY_TOLERANCE`] of `1.0`. An identity array is never
/// multiplied into the gradient — a weight of `1.00005` is discarded, not
/// applied (ledger §2.116).
fn weights_are_identity(weights: &[f64]) -> bool {
    weights
        .iter()
        .all(|w| (1.0 - w).abs() <= WEIGHTS_IDENTITY_TOLERANCE)
}

/// How optimizer parameter scales are chosen — SimpleITK's
/// `m_OptimizerScalesType` plus this crate's [`Unit`](Self::Unit) default.
enum ScalesMode {
    /// All-ones (no balancing).
    Unit,
    /// Caller-supplied scales (SimpleITK `Manual`, set by `SetOptimizerScales`).
    Manual(Vec<f64>),
    /// Estimated by one of ITK's `RegistrationParameterScales*` estimators.
    Estimated(ScalesEstimatorKind),
}

/// How the learning rate is chosen.
enum LearningRateMode {
    /// Caller-supplied fixed rate.
    Manual,
    /// Estimated from physical shift, once or at each iteration.
    Estimate(EstimateLearningRate),
}

/// Which similarity metric the method optimizes. Selected via one of the
/// `set_metric_as_*` methods on [`ImageRegistrationMethod`].
enum MetricKind {
    /// Mean squares (`itk::MeanSquaresImageToImageMetricv4`) — the default,
    /// suited to same-modality images with a linear intensity relationship.
    MeanSquares,
    /// Mattes mutual information
    /// (`itk::MattesMutualInformationImageToImageMetricv4`) with the given
    /// number of joint-histogram bins — suited to multi-modality registration.
    MattesMutualInformation { number_of_histogram_bins: usize },
    /// Global normalized cross-correlation
    /// (`itk::CorrelationImageToImageMetricv4`). Global-support transforms only.
    Correlation,
    /// ANTS local neighborhood cross-correlation
    /// (`itk::ANTSNeighborhoodCorrelationImageToImageMetricv4`) over a window of
    /// diameter `2·radius + 1`.
    AntsNeighborhoodCorrelation { radius: usize },
    /// Joint-histogram mutual information
    /// (`itk::JointHistogramMutualInformationImageToImageMetricv4`).
    JointHistogramMutualInformation {
        number_of_histogram_bins: usize,
        variance_for_joint_pdf_smoothing: f64,
    },
    /// Demons (`itk::DemonsImageToImageMetricv4`). Local-support (displacement
    /// field) transforms only.
    Demons { intensity_difference_threshold: f64 },
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
    Correlation(CorrelationMetric),
    AntsNeighborhoodCorrelation(AntsNeighborhoodCorrelationMetric),
    JointHistogram(JointHistogramMutualInformationMetric),
    Demons(DemonsMetric),
}

/// The transform the metric actually maps a virtual-domain point through to
/// reach the **moving** image: the optimized transform followed by the
/// moving-initial transform.
///
/// `itk::ImageRegistrationMethodv4::InitializeRegistrationAtEachLevel` builds
/// `m_CompositeTransform` by adding the moving-initial transform and *then* the
/// output (optimized) transform (`itkImageRegistrationMethodv4.hxx:349,360`),
/// hands it to the metric with `SetMovingTransform`
/// (`:524`), and calls `SetOnlyMostRecentTransformToOptimizeOn` (`:438`).
/// `itk::CompositeTransform::TransformPoint` applies its queue **in reverse add
/// order** (`itkCompositeTransform.hxx:60-71`), so the mapped moving point is
///
/// ```text
/// moving_point = moving_initial( optimized( virtual_point ) )
/// ```
///
/// and `SetOnlyMostRecentTransformToOptimizeOn` means only `optimized`'s
/// parameters are exposed to the optimizer. `SimpleITK`'s `MetricEvaluate`
/// assembles the identical queue by hand (`sitkImageRegistrationMethod.cxx:
/// 1057-1088`). The fixed image is *not* reached through this chain: it is
/// sampled at `fixed_initial(virtual_point)` instead, which this port applies
/// when it resamples the fixed image onto the virtual grid — see
/// [`ImageRegistrationMethod::prepare_level`].
///
/// `moving_initial == None` is upstream's identity case: both
/// `ExecuteInternal` and `EvaluateInternal` skip a transform whose class name is
/// `"IdentityTransform"` rather than composing it (ledger §3.33).
struct Composed<'a, T: ParametricTransform + ?Sized> {
    optimized: &'a mut T,
    moving_initial: Option<&'a Transform>,
}

impl<T: ParametricTransform + ?Sized> TransformBase for Composed<'_, T> {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        let p = self.optimized.transform_point(point);
        match self.moving_initial {
            Some(g) => g.transform_point(&p),
            None => p,
        }
    }

    fn dimension(&self) -> usize {
        self.optimized.dimension()
    }

    fn is_linear(&self) -> bool {
        self.optimized.is_linear() && self.moving_initial.is_none_or(|g| g.is_linear())
    }

    fn jacobian_wrt_position(&self, point: &[f64]) -> Vec<f64> {
        let jf = self.optimized.jacobian_wrt_position(point);
        match self.moving_initial {
            None => jf,
            Some(g) => {
                let jg = g.jacobian_wrt_position(&self.optimized.transform_point(point));
                mat_mul(&jg, &jf, self.dimension())
            }
        }
    }
}

impl<T: ParametricTransform + ?Sized> ParametricTransform for Composed<'_, T> {
    fn number_of_parameters(&self) -> usize {
        self.optimized.number_of_parameters()
    }

    fn parameters(&self) -> Vec<f64> {
        self.optimized.parameters()
    }

    fn set_parameters(&mut self, params: &[f64]) -> sitk_transform::Result<()> {
        self.optimized.set_parameters(params)
    }

    fn fixed_parameters(&self) -> Vec<f64> {
        self.optimized.fixed_parameters()
    }

    fn number_of_fixed_parameters(&self) -> usize {
        self.optimized.number_of_fixed_parameters()
    }

    fn set_fixed_parameters(&mut self, params: &[f64]) -> sitk_transform::Result<()> {
        self.optimized.set_fixed_parameters(params)
    }

    fn has_local_support(&self) -> bool {
        self.optimized.has_local_support()
    }

    fn number_of_local_parameters(&self) -> usize {
        self.optimized.number_of_local_parameters()
    }

    /// `∂ moving_initial(optimized(x)) / ∂p = J_g(optimized(x)) · J_f(x, p)`,
    /// the chain rule `itk::CompositeTransform::ComputeJacobianWithRespectToParameters`
    /// applies to every block but the last-applied one.
    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        let jf = self.optimized.jacobian_wrt_parameters(point);
        match self.moving_initial {
            None => jf,
            Some(g) => {
                let dim = self.dimension();
                let jg = g.jacobian_wrt_position(&self.optimized.transform_point(point));
                mat_mul_rect(&jg, &jf, dim, self.number_of_parameters())
            }
        }
    }

    /// The sparse Jacobian's columns are `∂T/∂p_k` — each a `dim`-vector — so
    /// the same left-multiplication by `J_g` that
    /// [`jacobian_wrt_parameters`](Self::jacobian_wrt_parameters) applies to the
    /// dense array applies column-wise here.
    fn sparse_jacobian_wrt_parameters(&self, point: &[f64]) -> Option<Vec<(usize, Vec<f64>)>> {
        let sparse = self.optimized.sparse_jacobian_wrt_parameters(point)?;
        let g = match self.moving_initial {
            None => return Some(sparse),
            Some(g) => g,
        };
        let dim = self.dimension();
        let jg = g.jacobian_wrt_position(&self.optimized.transform_point(point));
        Some(
            sparse
                .into_iter()
                .map(|(k, col)| {
                    let mapped = (0..dim)
                        .map(|r| (0..dim).map(|c| jg[r * dim + c] * col[c]).sum())
                        .collect();
                    (k, mapped)
                })
                .collect(),
        )
    }
}

/// A `Float64` image carrying `values` on `reference`'s grid.
fn with_geometry_of(reference: &Image, values: Vec<f64>) -> Result<Image> {
    let mut image = Image::from_vec(reference.size(), values)?;
    image.set_origin(reference.origin())?;
    image.set_spacing(reference.spacing())?;
    image.set_direction(reference.direction())?;
    Ok(image)
}

/// The pointwise AND of two binary masks on one grid, `None` when neither is
/// present. A voxel survives only if it is nonzero in every mask given.
fn intersect_masks(a: Option<Image>, b: Option<Image>) -> Result<Option<Image>> {
    let (a, b) = match (a, b) {
        (None, None) => return Ok(None),
        (Some(only), None) | (None, Some(only)) => return Ok(Some(only)),
        (Some(a), Some(b)) => (a, b),
    };
    let (av, bv) = (a.to_f64_vec()?, b.to_f64_vec()?);
    let both = av
        .iter()
        .zip(bv.iter())
        .map(|(&x, &y)| f64::from(x != 0.0 && y != 0.0))
        .collect();
    Ok(Some(with_geometry_of(&a, both)?))
}

/// `a · b` for two row-major `n × n` matrices.
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    mat_mul_rect(a, b, n, n)
}

/// `a · b` where `a` is row-major `n × n` and `b` is row-major `n × cols`.
fn mat_mul_rect(a: &[f64], b: &[f64], n: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * cols];
    for r in 0..n {
        for c in 0..cols {
            out[r * cols + c] = (0..n).map(|k| a[r * n + k] * b[k * cols + c]).sum();
        }
    }
    out
}

/// The registration objective handed to the line-search optimizers: set the
/// transform's parameters, then ask the metric.
///
/// This exists rather than a closure so that [`Objective::value`] routes to
/// [`ActiveMetric::value`]. The line searches probe the value up to
/// `maximum_line_search_iterations` times per iteration and read no gradient on
/// those probes; a closure has no value-only kernel, so each probe would pay
/// for a derivative it never reads.
struct MetricObjective<'a, T: ParametricTransform> {
    transform: Composed<'a, T>,
    metric: &'a ActiveMetric,
    backend: &'a dyn MetricBackend,
}

impl<T: ParametricTransform> Objective for MetricObjective<'_, T> {
    fn value_and_gradient(&mut self, p: &[f64]) -> (f64, Vec<f64>) {
        // `Objective` has no fallible surface, and the optimizer driving this
        // always probes at `self.transform`'s own parameter dimension.
        self.transform
            .set_parameters(p)
            .expect("optimizer probes at the transform's own parameter dimension");
        let m = self.metric.evaluate(&self.transform, self.backend);
        (m.value, m.derivative)
    }

    fn value(&mut self, p: &[f64]) -> f64 {
        self.transform
            .set_parameters(p)
            .expect("optimizer probes at the transform's own parameter dimension");
        self.metric.value(&self.transform, self.backend)
    }
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
            ActiveMetric::Correlation(m) => m.evaluate(transform),
            ActiveMetric::AntsNeighborhoodCorrelation(m) => m.evaluate(transform),
            ActiveMetric::JointHistogram(m) => m.evaluate(transform),
            ActiveMetric::Demons(m) => m.evaluate(transform),
        }
    }

    /// Reject a transform whose support category this metric cannot handle,
    /// *before* any `evaluate` call can hit that metric's own debug assertion.
    /// [`CorrelationMetric`] is global-support only and [`DemonsMetric`] is
    /// local-support only; the rest accept both.
    fn check_transform(&self, transform: &dyn ParametricTransform) -> Result<()> {
        match self {
            ActiveMetric::Correlation(m) => m.check_transform(transform),
            ActiveMetric::Demons(m) => m.check_transform(transform),
            ActiveMetric::MeanSquares(_)
            | ActiveMetric::Mattes(_)
            | ActiveMetric::AntsNeighborhoodCorrelation(_)
            | ActiveMetric::JointHistogram(_) => Ok(()),
        }
    }

    /// Value alone at `transform`, for the gradient-free optimizers and the
    /// line searches' golden-section probes.
    ///
    /// Every metric has a value-only kernel: none of these builds a
    /// parameter-derivative, and none reads the moving-image gradient except
    /// where a validity predicate needs it. The value each returns is the one
    /// [`evaluate`](Self::evaluate) would return, over the identical valid
    /// sample set — pinned per metric by a `value_agrees_with_evaluate` test.
    fn value(&self, transform: &dyn ParametricTransform, backend: &dyn MetricBackend) -> f64 {
        match self {
            ActiveMetric::MeanSquares(m) => m.value(transform, backend),
            ActiveMetric::Mattes(m) => m.value(transform),
            ActiveMetric::Correlation(m) => m.value(transform),
            ActiveMetric::AntsNeighborhoodCorrelation(m) => m.value(transform),
            ActiveMetric::JointHistogram(m) => m.value(transform),
            ActiveMetric::Demons(m) => m.value(transform),
        }
    }

    /// Scale/learning-rate estimator of `kind` over the virtual domain.
    fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        match self {
            ActiveMetric::MeanSquares(m) => m.scales_estimator(transform, kind),
            ActiveMetric::Mattes(m) => m.scales_estimator(transform, kind),
            ActiveMetric::Correlation(m) => m.scales_estimator(transform, kind),
            ActiveMetric::AntsNeighborhoodCorrelation(m) => m.scales_estimator(transform, kind),
            ActiveMetric::JointHistogram(m) => m.scales_estimator(transform, kind),
            ActiveMetric::Demons(m) => m.scales_estimator(transform, kind),
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
    /// Gradient descent with a per-iteration golden-section line search
    /// (`itk::GradientDescentLineSearchOptimizerv4`): each step's learning rate
    /// is chosen to most reduce the metric along the gradient.
    LineSearch(GradientDescentLineSearchOptimizer),
    /// Conjugate gradient descent with a per-iteration golden-section line search
    /// (`itk::ConjugateGradientLineSearchOptimizerv4`): the line search's
    /// conjugate-direction variant, descending ill-conditioned basins faster.
    ConjugateGradientLineSearch(ConjugateGradientLineSearchOptimizer),
    /// Bound-constrained limited-memory BFGS (`itk::LBFGSBOptimizerv4`). Unlike
    /// the gradient-descent optimizers it does **not** use parameter scales or a
    /// learning rate — it drives the raw metric gradient through a line search —
    /// and it optionally clamps every parameter to a scalar `[lower, upper]` box.
    Lbfgsb(LbfgsbConfig),
    /// Unconstrained limited-memory BFGS (`itk::LBFGS2Optimizerv4`). Like
    /// [`Lbfgsb`](Self::Lbfgsb) it ignores parameter scales and the learning
    /// rate, driving the raw metric gradient through its own line search.
    Lbfgs2(LBFGS2Optimizer),
    /// Nelder–Mead downhill simplex (`itk::AmoebaOptimizerv4`).
    Amoeba(AmoebaOptimizer),
    /// Powell's direction-set method (`itk::PowellOptimizerv4`).
    Powell(PowellOptimizer),
    /// (1+1) evolutionary strategy (`itk::OnePlusOneEvolutionaryOptimizerv4`).
    OnePlusOneEvolutionary(OnePlusOneEvolutionaryOptimizer),
    /// Brute-force scan of a regular parameter grid centered on the initial
    /// transform (`itk::ExhaustiveOptimizerv4`).
    Exhaustive(ExhaustiveOptimizer),
}

impl OptimizerKind {
    /// Whether this optimizer ignores parameter scales and the learning-rate
    /// estimator. Both L-BFGS variants force identity scales in ITK
    /// (`itk::LBFGSBOptimizerv4`, `itk::LBFGS2Optimizerv4`) and drive the raw
    /// metric gradient directly.
    fn ignores_scales(&self) -> bool {
        matches!(self, OptimizerKind::Lbfgsb(_) | OptimizerKind::Lbfgs2(_))
    }
}

/// Configuration for the L-BFGS-B optimizer, mirroring SimpleITK
/// `SetOptimizerAsLBFGSB`. The bounds are **scalar**, applied to every parameter;
/// a bound equal to its sentinel ([`f64::MIN`] lower, [`f64::MAX`] upper) means
/// "unbounded on that side" (SimpleITK's `DBL_MIN`/`DBL_MAX` defaults).
#[derive(Clone, Debug)]
struct LbfgsbConfig {
    gradient_convergence_tolerance: f64,
    number_of_iterations: usize,
    maximum_number_of_corrections: usize,
    maximum_number_of_function_evaluations: usize,
    cost_function_convergence_factor: f64,
    lower_bound: f64,
    upper_bound: f64,
}

impl LbfgsbConfig {
    /// Build the optimizer for a transform with `nparams` parameters, translating
    /// the scalar bounds into a per-parameter bound selection exactly as SimpleITK
    /// does: `lower != f64::MIN` activates a lower bound, `upper != f64::MAX` an
    /// upper bound, and the (lower, upper) activation maps to netlib's `nbd`
    /// (`0` unbounded, `1` lower, `2` both, `3` upper).
    fn build(&self, nparams: usize) -> LBFGSBOptimizer {
        let mut opt = LBFGSBOptimizer::new(self.number_of_iterations);
        opt.set_gradient_convergence_tolerance(self.gradient_convergence_tolerance)
            .set_max_corrections(self.maximum_number_of_corrections)
            .set_max_function_evaluations(self.maximum_number_of_function_evaluations)
            .set_cost_function_convergence_factor(self.cost_function_convergence_factor);

        let lower_active = self.lower_bound != f64::MIN;
        let upper_active = self.upper_bound != f64::MAX;
        let nbd: i32 = match (lower_active, upper_active) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
        if nbd != 0 {
            opt.set_bound_selection(vec![nbd; nparams])
                .set_lower_bound(vec![self.lower_bound; nparams])
                .set_upper_bound(vec![self.upper_bound; nparams]);
        }
        opt
    }
}

/// SimpleITK `ImageRegistrationMethod` defaults for convergence monitoring in
/// the estimate-at-each-iteration mode (window of the last 10 metric values,
/// stop when the trend flattens to `1e-6`).
const CONVERGENCE_WINDOW_SIZE: usize = 10;
const MINIMUM_CONVERGENCE_VALUE: f64 = 1e-6;

/// The geometry of the **virtual reference domain** the metric samples in —
/// SimpleITK `SetVirtualDomain(size, origin, spacing, direction)` /
/// `SetVirtualDomainFromImage(image)`, stored on the method and pushed onto the
/// metric by `SetupMetric` (`sitkImageRegistrationMethod.cxx:1122-1135`).
///
/// Every metric sample point is a voxel center of this grid; the fixed image is
/// read at `fixed_initial(point)` and the moving image at
/// `moving_initial(optimized(point))`. When it is unset, ITK falls back to the
/// fixed image's own grid (`itkImageRegistrationMethodv4.hxx:388-392`), which is
/// why a virtual domain equal to the fixed image's geometry is a no-op.
#[derive(Clone, Debug, PartialEq)]
struct VirtualDomain {
    size: Vec<usize>,
    origin: Vec<f64>,
    spacing: Vec<f64>,
    direction: Vec<f64>,
}

impl VirtualDomain {
    /// An all-zero image carrying this geometry, used purely as a grid: the
    /// per-level shrink and the fixed-image resampling both take their output
    /// geometry from it. ITK does the same, allocating `m_VirtualDomainImage`
    /// and never reading its pixels (`itkImageRegistrationMethodv4.hxx:394-397`).
    fn grid(&self) -> Result<Image> {
        let n = self.size.iter().product();
        let mut image = Image::from_vec(&self.size, vec![0.0f64; n])?;
        image.set_origin(&self.origin)?;
        image.set_spacing(&self.spacing)?;
        image.set_direction(&self.direction)?;
        Ok(image)
    }
}

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
    /// Per-local-parameter optimizer weights (SimpleITK `SetOptimizerWeights`).
    /// Empty means identity, as ITK's own empty `m_Weights` array does.
    optimizer_weights: Vec<f64>,
    learning_rate_mode: LearningRateMode,
    backend: Box<dyn MetricBackend>,
    /// Interpolator used to read the moving image at a mapped fixed point
    /// (SimpleITK `SetInterpolator`). Defaults to linear.
    interpolator: Interpolator,
    /// How the fixed (virtual-domain) samples are drawn (SimpleITK
    /// `SetMetricSamplingStrategy`). Defaults to every voxel.
    sampling_strategy: SamplingStrategy,
    /// Sampling percentage, one entry per resolution level. Empty means 1.0 at
    /// every level; a single entry applies to every level.
    sampling_percentage_per_level: Vec<f64>,
    /// Seed for [`SamplingStrategy::Random`].
    sampling_seed: u64,
    /// Binary mask on the fixed image's grid; zero voxels are never sampled.
    fixed_mask: Option<Image>,
    /// Binary mask on the moving image's grid; a fixed point that maps into a
    /// zero voxel counts as outside.
    moving_mask: Option<Image>,
    /// One shrink factor per resolution level (applied to every dimension),
    /// coarsest first. Empty means a single full-resolution level.
    shrink_factors_per_level: Vec<usize>,
    /// One Gaussian smoothing sigma per resolution level, coarsest first. Must
    /// match `shrink_factors_per_level` in length when either is set.
    smoothing_sigmas_per_level: Vec<f64>,
    /// Whether `smoothing_sigmas_per_level` are in physical units (ITK's
    /// default, `true`) or in voxels of the fixed image.
    smoothing_sigmas_in_physical_units: bool,
    /// The transform the optimizer drives, set by
    /// [`set_initial_transform`](ImageRegistrationMethod::set_initial_transform)
    /// and read by
    /// [`execute_with_initial_transform`](ImageRegistrationMethod::execute_with_initial_transform)
    /// and [`metric_evaluate`](ImageRegistrationMethod::metric_evaluate).
    /// `None` means upstream's default-constructed identity.
    initial_transform: Option<Transform>,
    /// SimpleITK `m_InitialTransformInPlace` (default `true`).
    initial_transform_in_place: bool,
    /// Applied *after* the optimized transform on the way to the moving image.
    /// `None` = identity. See [`Composed`].
    moving_initial_transform: Option<Transform>,
    /// Applied to a virtual-domain point on the way to the fixed image.
    /// `None` = identity.
    fixed_initial_transform: Option<Transform>,
    /// The virtual reference domain. `None` = the fixed image's own grid.
    virtual_domain: Option<VirtualDomain>,
}

impl Default for ImageRegistrationMethod {
    fn default() -> Self {
        Self {
            optimizer: OptimizerKind::GradientDescent(GradientDescentOptimizer::new(1.0, 100)),
            metric_kind: MetricKind::MeanSquares,
            scales_mode: ScalesMode::Unit,
            optimizer_weights: Vec::new(),
            learning_rate_mode: LearningRateMode::Manual,
            backend: Box::new(CpuBackend),
            interpolator: Interpolator::Linear,
            sampling_strategy: SamplingStrategy::None,
            sampling_percentage_per_level: Vec::new(),
            sampling_seed: 0,
            fixed_mask: None,
            moving_mask: None,
            shrink_factors_per_level: Vec::new(),
            smoothing_sigmas_per_level: Vec::new(),
            smoothing_sigmas_in_physical_units: true,
            initial_transform: None,
            initial_transform_in_place: true,
            moving_initial_transform: None,
            fixed_initial_transform: None,
            virtual_domain: None,
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

    /// Use gradient descent with a per-iteration golden-section line search
    /// (`itk::GradientDescentLineSearchOptimizerv4`, SimpleITK
    /// `SetOptimizerAsGradientDescentLineSearch`) at a caller-supplied base
    /// learning rate. Each iteration a golden-section search picks the rate in
    /// `[learning_rate·0, learning_rate·5]` that most reduces the metric along
    /// the gradient, then steps; the rate found seeds the next iteration's
    /// bracket. Value-plateau monitoring stops the run (SimpleITK always
    /// configures it for this optimizer), backed by the min-step tolerance as
    /// the gradient — and thus the step — shrinks toward the minimum.
    pub fn set_optimizer_as_gradient_descent_line_search(
        &mut self,
        learning_rate: f64,
        iterations: usize,
    ) -> &mut Self {
        let mut optimizer = GradientDescentLineSearchOptimizer::new(learning_rate, iterations);
        optimizer.set_convergence(CONVERGENCE_WINDOW_SIZE, MINIMUM_CONVERGENCE_VALUE);
        self.optimizer = OptimizerKind::LineSearch(optimizer);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Like [`set_optimizer_as_gradient_descent_line_search`](Self::set_optimizer_as_gradient_descent_line_search)
    /// but the base learning rate is **estimated** from physical shift. ITK's
    /// line search estimates the base rate only **once**, at the first iteration
    /// (`if m_CurrentIteration == 0`); the golden-section search then adapts the
    /// rate every iteration, so — unlike plain gradient descent — there is no
    /// per-iteration re-estimation mode, and an over-large initial estimate (as
    /// from a near-converged multi-resolution restart) is corrected by the first
    /// line search rather than needing a one-voxel step cap: an exploding step
    /// raises the metric, which the search rejects. Pair with
    /// [`set_optimizer_scales_from_physical_shift`](Self::set_optimizer_scales_from_physical_shift).
    pub fn set_optimizer_as_gradient_descent_line_search_estimated(
        &mut self,
        iterations: usize,
    ) -> &mut Self {
        // The stored base rate is a placeholder overwritten by the estimate.
        let mut optimizer = GradientDescentLineSearchOptimizer::new(1.0, iterations);
        optimizer.set_convergence(CONVERGENCE_WINDOW_SIZE, MINIMUM_CONVERGENCE_VALUE);
        self.optimizer = OptimizerKind::LineSearch(optimizer);
        self.learning_rate_mode = LearningRateMode::Estimate(EstimateLearningRate::Once);
        self
    }

    /// Use conjugate gradient descent with a per-iteration golden-section line
    /// search (`itk::ConjugateGradientLineSearchOptimizerv4`, SimpleITK
    /// `SetOptimizerAsConjugateGradientLineSearch`) at a caller-supplied base
    /// learning rate. Like
    /// [`set_optimizer_as_gradient_descent_line_search`](Self::set_optimizer_as_gradient_descent_line_search)
    /// but the search direction is the modified Polak–Ribière conjugate of the
    /// gradient, so successive steps do not undo one another and an elongated
    /// basin is descended in far fewer iterations. Value-plateau monitoring stops
    /// the run (SimpleITK always configures it for this optimizer).
    pub fn set_optimizer_as_conjugate_gradient_line_search(
        &mut self,
        learning_rate: f64,
        iterations: usize,
    ) -> &mut Self {
        let mut optimizer = ConjugateGradientLineSearchOptimizer::new(learning_rate, iterations);
        optimizer.set_convergence(CONVERGENCE_WINDOW_SIZE, MINIMUM_CONVERGENCE_VALUE);
        self.optimizer = OptimizerKind::ConjugateGradientLineSearch(optimizer);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Like [`set_optimizer_as_conjugate_gradient_line_search`](Self::set_optimizer_as_conjugate_gradient_line_search)
    /// but the base learning rate is **estimated** from physical shift. As with
    /// the plain line search, ITK estimates the base rate only once, at the first
    /// iteration; the golden-section search then adapts it every iteration, so
    /// there is no per-iteration re-estimation mode and no one-voxel step cap is
    /// needed. Pair with
    /// [`set_optimizer_scales_from_physical_shift`](Self::set_optimizer_scales_from_physical_shift).
    pub fn set_optimizer_as_conjugate_gradient_line_search_estimated(
        &mut self,
        iterations: usize,
    ) -> &mut Self {
        // The stored base rate is a placeholder overwritten by the estimate.
        let mut optimizer = ConjugateGradientLineSearchOptimizer::new(1.0, iterations);
        optimizer.set_convergence(CONVERGENCE_WINDOW_SIZE, MINIMUM_CONVERGENCE_VALUE);
        self.optimizer = OptimizerKind::ConjugateGradientLineSearch(optimizer);
        self.learning_rate_mode = LearningRateMode::Estimate(EstimateLearningRate::Once);
        self
    }

    /// Use the bound-constrained limited-memory BFGS optimizer
    /// (`itk::LBFGSBOptimizerv4`), SimpleITK `SetOptimizerAsLBFGSB`. It drives the
    /// raw metric gradient through a Moré–Thuente line search and, unlike the
    /// gradient-descent optimizers, **ignores parameter scales and the
    /// learning-rate estimator** (ITK forces LBFGSB's scales to identity), so any
    /// [`set_optimizer_scales_from_physical_shift`](Self::set_optimizer_scales_from_physical_shift)
    /// setting has no effect here.
    ///
    /// The bounds are **scalar**, applied to every parameter. Pass [`f64::MIN`]
    /// for `lower_bound` and [`f64::MAX`] for `upper_bound` to run unbounded on
    /// that side (SimpleITK's `DBL_MIN`/`DBL_MAX` defaults); any other value
    /// activates the bound. Stopping is governed by `gradient_convergence_tolerance`
    /// (projected-gradient infinity norm), `cost_function_convergence_factor`
    /// (relative function decrease), `number_of_iterations`, and
    /// `maximum_number_of_function_evaluations`; `maximum_number_of_corrections`
    /// is the limited-memory depth.
    #[allow(clippy::too_many_arguments)]
    pub fn set_optimizer_as_lbfgsb(
        &mut self,
        gradient_convergence_tolerance: f64,
        number_of_iterations: usize,
        maximum_number_of_corrections: usize,
        maximum_number_of_function_evaluations: usize,
        cost_function_convergence_factor: f64,
        lower_bound: f64,
        upper_bound: f64,
    ) -> &mut Self {
        self.optimizer = OptimizerKind::Lbfgsb(LbfgsbConfig {
            gradient_convergence_tolerance,
            number_of_iterations,
            maximum_number_of_corrections,
            maximum_number_of_function_evaluations,
            cost_function_convergence_factor,
            lower_bound,
            upper_bound,
        });
        // LBFGSB has no learning rate; drop any estimated-rate mode a previous
        // optimizer selection may have set so no estimator is built for it.
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use **unconstrained** limited-memory BFGS (`itk::LBFGS2Optimizerv4`),
    /// SimpleITK `SetOptimizerAsLBFGS2`. Like [`set_optimizer_as_lbfgsb`] it
    /// ignores parameter scales and the learning rate. `number_of_iterations ==
    /// 0` means "iterate until convergence" (SimpleITK's default).
    ///
    /// The remaining line-search knobs keep SimpleITK's defaults
    /// (`MoreThuente`, Wolfe coefficient `0.9`, gradient accuracy `0.9`,
    /// machine-precision tolerance `1e-16`); reach for
    /// [`LBFGS2Optimizer`] directly to change them.
    ///
    /// [`set_optimizer_as_lbfgsb`]: Self::set_optimizer_as_lbfgsb
    #[allow(clippy::too_many_arguments)]
    pub fn set_optimizer_as_lbfgs2(
        &mut self,
        solution_accuracy: f64,
        number_of_iterations: usize,
        hessian_approximate_accuracy: usize,
        delta_convergence_distance: usize,
        delta_convergence_tolerance: f64,
        line_search_maximum_evaluations: usize,
        line_search_minimum_step: f64,
        line_search_maximum_step: f64,
        line_search_accuracy: f64,
    ) -> &mut Self {
        let mut opt = LBFGS2Optimizer::new();
        opt.set_solution_accuracy(solution_accuracy)
            .set_number_of_iterations(number_of_iterations)
            .set_hessian_approximate_accuracy(hessian_approximate_accuracy)
            .set_delta_convergence_distance(delta_convergence_distance)
            .set_delta_convergence_tolerance(delta_convergence_tolerance)
            .set_line_search_maximum_evaluations(line_search_maximum_evaluations)
            .set_line_search_minimum_step(line_search_minimum_step)
            .set_line_search_maximum_step(line_search_maximum_step)
            .set_line_search_accuracy(line_search_accuracy);
        self.optimizer = OptimizerKind::Lbfgs2(opt);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use the **Nelder–Mead downhill simplex** (`itk::AmoebaOptimizerv4`),
    /// SimpleITK `SetOptimizerAsAmoeba`. Gradient-free: the metric derivative is
    /// never computed. `simplex_delta` is added to and subtracted from the
    /// initial parameters to build the starting simplex, and is divided by the
    /// optimizer scales per parameter.
    pub fn set_optimizer_as_amoeba(
        &mut self,
        simplex_delta: f64,
        number_of_iterations: usize,
        parameters_convergence_tolerance: f64,
        function_convergence_tolerance: f64,
        with_restarts: bool,
    ) -> &mut Self {
        let mut opt = AmoebaOptimizer::new(simplex_delta, number_of_iterations);
        opt.set_parameters_convergence_tolerance(parameters_convergence_tolerance)
            .set_function_convergence_tolerance(function_convergence_tolerance)
            .set_with_restarts(with_restarts);
        self.optimizer = OptimizerKind::Amoeba(opt);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use **Powell's direction-set method** (`itk::PowellOptimizerv4`),
    /// SimpleITK `SetOptimizerAsPowell`. Gradient-free: the metric derivative is
    /// never computed.
    pub fn set_optimizer_as_powell(
        &mut self,
        number_of_iterations: usize,
        maximum_line_iterations: usize,
        step_length: f64,
        step_tolerance: f64,
        value_tolerance: f64,
    ) -> &mut Self {
        let mut opt = PowellOptimizer::new();
        opt.set_number_of_iterations(number_of_iterations)
            .set_maximum_line_iterations(maximum_line_iterations)
            .set_step_length(step_length)
            .set_step_tolerance(step_tolerance)
            .set_value_tolerance(value_tolerance);
        self.optimizer = OptimizerKind::Powell(opt);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use the **(1+1) evolutionary strategy**
    /// (`itk::OnePlusOneEvolutionaryOptimizerv4`), SimpleITK
    /// `SetOptimizerAsOnePlusOneEvolutionary`. Gradient-free: the metric
    /// derivative is never computed.
    ///
    /// A non-positive `growth_factor` or `shrink_factor` selects ITK's
    /// `epsilon`-derived default (SimpleITK passes `-1.0` for both). Unlike
    /// SimpleITK, `seed` has no wall-clock default: it is required, so every run
    /// is reproducible.
    pub fn set_optimizer_as_one_plus_one_evolutionary(
        &mut self,
        number_of_iterations: usize,
        epsilon: f64,
        initial_radius: f64,
        growth_factor: f64,
        shrink_factor: f64,
        seed: i32,
    ) -> &mut Self {
        let mut opt = OnePlusOneEvolutionaryOptimizer::new(number_of_iterations, seed);
        opt.set_epsilon(epsilon).set_initial_radius(initial_radius);
        if growth_factor > 0.0 {
            opt.set_growth_factor(growth_factor);
        }
        if shrink_factor > 0.0 {
            opt.set_shrink_factor(shrink_factor);
        }
        self.optimizer = OptimizerKind::OnePlusOneEvolutionary(opt);
        self.learning_rate_mode = LearningRateMode::Manual;
        self
    }

    /// Use the **exhaustive grid scan** (`itk::ExhaustiveOptimizerv4`),
    /// SimpleITK `SetOptimizerAsExhaustive`. Gradient-free: the metric
    /// derivative is never computed.
    ///
    /// `number_of_steps` has one entry per transform parameter; parameter `k` is
    /// swept over `2·number_of_steps[k] + 1` grid points spaced
    /// `step_length / scales[k]` apart and centered on the initial value, so the
    /// scan costs `∏ (2·steps[k] + 1)` metric evaluations. The result is the
    /// grid point of least metric value — no local refinement follows, so this
    /// is normally an initializer for another optimizer, not a final stage.
    pub fn set_optimizer_as_exhaustive(
        &mut self,
        number_of_steps: Vec<usize>,
        step_length: f64,
    ) -> &mut Self {
        self.optimizer =
            OptimizerKind::Exhaustive(ExhaustiveOptimizer::new(number_of_steps, step_length));
        self.learning_rate_mode = LearningRateMode::Manual;
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
    ///
    /// Upstream's `SetOptimizerScalesFromPhysicalShift(centralRegionRadius = 5,
    /// smallParameterVariation = 0.01)` takes both estimator knobs as default
    /// arguments; this method uses those defaults (ledger §5.19).
    pub fn set_optimizer_scales_from_physical_shift(&mut self) -> &mut Self {
        self.scales_mode = ScalesMode::Estimated(ScalesEstimatorKind::PhysicalShift {
            central_region_radius: DEFAULT_CENTRAL_REGION_RADIUS,
            small_parameter_variation: DEFAULT_SMALL_PARAMETER_VARIATION,
        });
        self
    }

    /// Estimate optimizer scales from the mean squared transform-Jacobian
    /// column norm over the sampled virtual domain — SimpleITK
    /// `SetOptimizerScalesFromJacobian(centralRegionRadius = 5)` /
    /// `itk::RegistrationParameterScalesFromJacobian`.
    ///
    /// Unlike the two shift estimators, this one *averages* over the sample
    /// points instead of taking the worst one, so it is less sensitive to a
    /// single distant voxel and produces smaller scales for the matrix
    /// parameters of a rotation or affine transform.
    ///
    /// `central_region_radius` is carried faithfully but has no observable
    /// effect for any transform this crate registers — see [`crate::scales`]
    /// and ledger §2.115. Pass [`DEFAULT_CENTRAL_REGION_RADIUS`] for upstream's
    /// default.
    pub fn set_optimizer_scales_from_jacobian(
        &mut self,
        central_region_radius: usize,
    ) -> &mut Self {
        self.scales_mode = ScalesMode::Estimated(ScalesEstimatorKind::Jacobian {
            central_region_radius,
        });
        self
    }

    /// Estimate optimizer scales from the shift a small parameter variation
    /// produces in the **moving image's continuous-index** units — SimpleITK
    /// `SetOptimizerScalesFromIndexShift(centralRegionRadius = 5,
    /// smallParameterVariation = 0.01)` /
    /// `itk::RegistrationParameterScalesFromIndexShift`.
    ///
    /// This is [`set_optimizer_scales_from_physical_shift`] with the shift
    /// divided through by the moving image's spacing and direction, so an
    /// anisotropic moving image weights the parameters that move samples along
    /// its fine axis more heavily.
    ///
    /// `central_region_radius` is carried faithfully but has no observable
    /// effect for any transform this crate registers — see [`crate::scales`]
    /// and ledger §2.115. Pass [`DEFAULT_CENTRAL_REGION_RADIUS`] and
    /// [`DEFAULT_SMALL_PARAMETER_VARIATION`] for upstream's defaults.
    ///
    /// [`set_optimizer_scales_from_physical_shift`]: Self::set_optimizer_scales_from_physical_shift
    pub fn set_optimizer_scales_from_index_shift(
        &mut self,
        central_region_radius: usize,
        small_parameter_variation: f64,
    ) -> &mut Self {
        self.scales_mode = ScalesMode::Estimated(ScalesEstimatorKind::IndexShift {
            central_region_radius,
            small_parameter_variation,
        });
        self
    }

    /// Set the per-local-parameter optimizer **weights** — SimpleITK
    /// `SetOptimizerWeights` / `itk::ObjectToObjectOptimizerBase::SetWeights`.
    ///
    /// Weights multiply the gradient at the same point scales divide it
    /// (`gradient[j] *= weights[j % n] / scales[j % n]`,
    /// `itkGradientDescentOptimizerv4.hxx:205-239`), and are the documented way
    /// to hold a parameter constant: a zero weight freezes it.
    ///
    /// Three upstream behaviors this reproduces:
    ///
    /// - The length must equal the transform's
    ///   [`number_of_local_parameters`], **not** its parameter count. For a
    ///   displacement field that is just `dim`; the array is then tiled across
    ///   every pixel's parameter block. A mismatch raises
    ///   [`RegistrationError::OptimizerWeightsLength`] when the registration
    ///   runs, not here.
    /// - Weights within `1e-4` of `1.0` are treated as exactly identity and
    ///   never multiplied in (`m_WeightsAreIdentity`,
    ///   `itkObjectToObjectOptimizerBase.cxx:143-166`; ledger §2.116).
    /// - The length is validated for **every** optimizer, but only the
    ///   gradient-descent family applies the weights. The vnl-backed
    ///   optimizers — `LBFGSOptimizerv4`, `LBFGSBOptimizerv4`, and the
    ///   gradient-free `AmoebaOptimizerv4` — validate and then ignore them.
    ///   `LBFGS2Optimizerv4` derives from the gradient-descent template and
    ///   *does* apply them (ledger §2.117).
    ///
    /// An empty vector (the default) means identity.
    ///
    /// [`number_of_local_parameters`]: sitk_transform::ParametricTransform::number_of_local_parameters
    pub fn set_optimizer_weights(&mut self, weights: Vec<f64>) -> &mut Self {
        self.optimizer_weights = weights;
        self
    }

    /// The optimizer weights set by [`set_optimizer_weights`], empty when unset
    /// — SimpleITK `GetOptimizerWeights`.
    ///
    /// [`set_optimizer_weights`]: Self::set_optimizer_weights
    pub fn optimizer_weights(&self) -> &[f64] {
        &self.optimizer_weights
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
            OptimizerKind::LineSearch(ls) => {
                ls.set_min_step_tolerance(tol);
            }
            OptimizerKind::ConjugateGradientLineSearch(cg) => {
                cg.set_min_step_tolerance(tol);
            }
            // Neither L-BFGS variant has a scaled-step tolerance; they stop via
            // their projected-gradient and function-decrease criteria instead.
            OptimizerKind::Lbfgsb(_) | OptimizerKind::Lbfgs2(_) => {}
            // Amoeba and Powell have their own parameter-convergence tolerances
            // (`set_parameters_convergence_tolerance`, `set_step_tolerance`),
            // set through their own SimpleITK setters; (1+1) evolutionary stops
            // on its search radius (`epsilon`) and Exhaustive on its grid.
            OptimizerKind::Amoeba(_)
            | OptimizerKind::Powell(_)
            | OptimizerKind::OnePlusOneEvolutionary(_)
            | OptimizerKind::Exhaustive(_) => {}
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

    /// Use the **normalized cross-correlation** metric
    /// (`itk::CorrelationImageToImageMetricv4`), a global reduction over every
    /// sample. Suited to same-modality images related by an *affine* intensity
    /// map (mean squares assumes the identity map). SimpleITK
    /// `SetMetricAsCorrelation`.
    ///
    /// [`execute`](Self::execute) returns
    /// [`RegistrationError::RequiresGlobalTransform`] for a displacement-field
    /// transform, matching ITK, whose constructor throws for one.
    pub fn set_metric_as_correlation(&mut self) -> &mut Self {
        self.metric_kind = MetricKind::Correlation;
        self
    }

    /// Use the **ANTS neighborhood cross-correlation** metric
    /// (`itk::ANTSNeighborhoodCorrelationImageToImageMetricv4`): normalized
    /// cross-correlation computed inside a window of diameter `2·radius + 1`
    /// around each sample and summed. SimpleITK
    /// `SetMetricAsANTSNeighborhoodCorrelation` (default radius 5).
    ///
    /// Unlike [`set_metric_as_correlation`](Self::set_metric_as_correlation) it
    /// accepts both global and displacement-field transforms. Errors at
    /// [`execute`](Self::execute) if the window does not fit inside the fixed
    /// image along some axis.
    pub fn set_metric_as_ants_neighborhood_correlation(&mut self, radius: usize) -> &mut Self {
        self.metric_kind = MetricKind::AntsNeighborhoodCorrelation { radius };
        self
    }

    /// Use the **joint-histogram mutual-information** metric
    /// (`itk::JointHistogramMutualInformationImageToImageMetricv4`) with
    /// `number_of_histogram_bins` bins (SimpleITK's default is 20) and
    /// `variance_for_joint_pdf_smoothing` (default `1.5`) as the variance, in
    /// bins², of the discrete Gaussian smoothed over each histogram axis.
    /// SimpleITK `SetMetricAsJointHistogramMutualInformation`.
    ///
    /// A second multi-modality metric alongside
    /// [`set_metric_as_mattes_mutual_information`](Self::set_metric_as_mattes_mutual_information);
    /// it estimates the joint density by Gaussian-smoothing a hard-binned
    /// histogram rather than by a cubic B-spline Parzen window. Errors at
    /// [`execute`](Self::execute) if fewer than six bins are requested or an
    /// image is constant.
    pub fn set_metric_as_joint_histogram_mutual_information(
        &mut self,
        number_of_histogram_bins: usize,
        variance_for_joint_pdf_smoothing: f64,
    ) -> &mut Self {
        self.metric_kind = MetricKind::JointHistogramMutualInformation {
            number_of_histogram_bins,
            variance_for_joint_pdf_smoothing,
        };
        self
    }

    /// Use the **Demons** metric (`itk::DemonsImageToImageMetricv4`), the
    /// optical-flow force `(f − m)·∇f / (‖∇f‖² + (f − m)²/normalizer)` written
    /// straight into each pixel's own parameter block. SimpleITK
    /// `SetMetricAsDemons` (default `intensityDifferenceThreshold` `0.001`): a
    /// sample whose `|f − m|` falls below the threshold contributes no force.
    ///
    /// [`execute`](Self::execute) returns
    /// [`RegistrationError::RequiresLocalSupportTransform`] for anything but a
    /// [`DisplacementFieldTransform`](sitk_transform::DisplacementFieldTransform),
    /// matching `itk::DemonsImageToImageMetricv4::Initialize`.
    pub fn set_metric_as_demons(&mut self, intensity_difference_threshold: f64) -> &mut Self {
        self.metric_kind = MetricKind::Demons {
            intensity_difference_threshold,
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

    /// Set the interpolator used to read the moving image at a mapped fixed
    /// point (SimpleITK `SetInterpolator`). Defaults to
    /// [`Interpolator::Linear`].
    ///
    /// [`Interpolator::NearestNeighbor`] makes the metric piecewise constant, so
    /// its analytic derivative is zero almost everywhere — usable with the
    /// gradient-free optimizers, useless with the gradient-descent ones.
    /// [`Interpolator::BSpline`] (cubic) and [`Interpolator::Gaussian`] are
    /// smoother than linear and give a better-conditioned gradient at a higher
    /// per-sample cost.
    pub fn set_interpolator(&mut self, interpolator: Interpolator) -> &mut Self {
        self.interpolator = interpolator;
        self
    }

    /// Choose how the fixed image's virtual domain is sampled (SimpleITK
    /// `SetMetricSamplingStrategy`). Defaults to [`SamplingStrategy::None`] —
    /// every voxel. `Regular` and `Random` use the percentage set by
    /// [`set_metric_sampling_percentage`](Self::set_metric_sampling_percentage).
    pub fn set_metric_sampling_strategy(&mut self, strategy: SamplingStrategy) -> &mut Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set the fraction of virtual-domain voxels sampled at every level
    /// (SimpleITK `SetMetricSamplingPercentage`). The percentage is of the voxel
    /// count *after* the level's shrink factor is applied. `seed` seeds
    /// [`SamplingStrategy::Random`]; it is ignored by the other strategies.
    pub fn set_metric_sampling_percentage(&mut self, percentage: f64, seed: u64) -> &mut Self {
        self.sampling_percentage_per_level = vec![percentage];
        self.sampling_seed = seed;
        self
    }

    /// Set the sampling fraction per resolution level, coarsest first
    /// (SimpleITK `SetMetricSamplingPercentagePerLevel`). Must have one entry
    /// per level; a length mismatch is reported by [`execute`](Self::execute) as
    /// [`RegistrationError::SamplingPercentageLength`].
    pub fn set_metric_sampling_percentage_per_level(
        &mut self,
        percentage: Vec<f64>,
        seed: u64,
    ) -> &mut Self {
        self.sampling_percentage_per_level = percentage;
        self.sampling_seed = seed;
        self
    }

    /// Restrict fixed-image sampling to the nonzero voxels of `mask`, a binary
    /// image on the fixed image's grid (SimpleITK `SetMetricFixedMask`).
    ///
    /// SimpleITK takes a mask in the fixed image's *physical space* and wraps it
    /// in an `itk::ImageMaskSpatialObject`, evaluated at each virtual point. In
    /// a multi-resolution run this crate resamples the mask onto each level's
    /// coarse virtual grid with nearest-neighbor interpolation, which evaluates
    /// the same physical-space predicate at the same points.
    ///
    /// Errors at [`execute`](Self::execute) if `mask` does not share the fixed
    /// image's size.
    pub fn set_metric_fixed_mask(&mut self, mask: &Image) -> &mut Self {
        self.fixed_mask = Some(mask.clone());
        self
    }

    /// Treat a fixed point that maps into a zero voxel of `mask` as outside the
    /// moving image (SimpleITK `SetMetricMovingMask`). `mask` is a binary image
    /// on the moving image's grid.
    ///
    /// Errors at [`execute`](Self::execute) if `mask` does not share the moving
    /// image's size.
    pub fn set_metric_moving_mask(&mut self, mask: &Image) -> &mut Self {
        self.moving_mask = Some(mask.clone());
        self
    }

    /// Set the transform the optimizer drives — SimpleITK
    /// `SetInitialTransform(const Transform &)`
    /// (`sitkImageRegistrationMethod.cxx:115-122`), which stores a deep copy
    /// (`MakeUnique`) and turns the in-place flag **on**.
    ///
    /// [`execute_with_initial_transform`](Self::execute_with_initial_transform)
    /// then optimizes it and returns it as the same concrete transform kind, and
    /// [`initial_transform`](Self::initial_transform) reflects the optimum. Use
    /// [`set_initial_transform_in_place`](Self::set_initial_transform_in_place)
    /// with `false` to leave the stored transform at its starting value and
    /// receive the optimum as a fresh composite instead.
    ///
    /// Note that this **re-enables** the in-place flag: calling it after a
    /// `set_initial_transform_in_place(t, false)` silently restores in-place
    /// optimization, exactly as upstream's single-argument overload does
    /// (ledger §3.36).
    pub fn set_initial_transform(&mut self, transform: Transform) -> &mut Self {
        self.initial_transform = Some(transform);
        self.initial_transform_in_place = true;
        self
    }

    /// Set the transform the optimizer drives and choose whether the optimizer
    /// writes through to it — SimpleITK `SetInitialTransform(Transform &, bool
    /// inPlace)` (`sitkImageRegistrationMethod.cxx:136-155`).
    ///
    /// Upstream, `inPlace` decides whether `itk::ImageRegistrationMethodv4`
    /// *grafts* the initial transform as its output — so the optimizer mutates
    /// that very ITK object — or `Clone()`s it and optimizes the copy
    /// (`itkImageRegistrationMethodv4.hxx:733-758`). Two things follow, and both
    /// are what this port reproduces:
    ///
    /// - **`true`**: after `Execute`, the method's stored initial transform holds
    ///   the optimized parameters, and `Execute` returns *it* — same concrete
    ///   transform kind as was set.
    /// - **`false`**: the stored initial transform is untouched, and `Execute`
    ///   returns a [`CompositeTransform`] wrapping a copy of the optimized
    ///   transform (ledger §3.34).
    ///
    /// Upstream's `inPlace = true` *additionally* aliases the caller's own
    /// `Transform` object, because C++ `Transform`s share one refcounted ITK
    /// pointer; a Rust `Transform` is a value, so that aliasing has no
    /// counterpart here (ledger §4.64). Everything observable through the
    /// method — the stored transform and the returned one — matches upstream.
    pub fn set_initial_transform_in_place(
        &mut self,
        transform: Transform,
        in_place: bool,
    ) -> &mut Self {
        self.initial_transform = Some(transform);
        self.initial_transform_in_place = in_place;
        self
    }

    /// The stored initial transform, or `None` if none was set. After an
    /// in-place [`execute_with_initial_transform`](Self::execute_with_initial_transform)
    /// this holds the optimized parameters; after a not-in-place one it still
    /// holds the starting values.
    pub fn initial_transform(&self) -> Option<&Transform> {
        self.initial_transform.as_ref()
    }

    /// Whether the optimizer writes through to the stored initial transform
    /// (SimpleITK `GetInitialTransformInPlace`). Defaults to `true`.
    pub fn initial_transform_in_place(&self) -> bool {
        self.initial_transform_in_place
    }

    /// Set the transform applied to the moving image **after** the optimized one
    /// — SimpleITK `SetMovingInitialTransform`, ITK
    /// `ImageRegistrationMethodv4::SetMovingInitialTransform`.
    ///
    /// The metric samples the moving image at
    /// `moving_initial(optimized(virtual_point))` (see [`Composed`] for the
    /// source lines that fix this order), so a moving-initial transform is the
    /// alignment already achieved by an earlier stage: the optimizer starts from
    /// where it left off without folding that stage into its own parameters.
    ///
    /// It is *not* optimized — only the initial transform's parameters are.
    pub fn set_moving_initial_transform(&mut self, transform: Transform) -> &mut Self {
        self.moving_initial_transform = Some(transform);
        self
    }

    /// Set the transform applied to a virtual-domain point on the way to the
    /// **fixed** image — SimpleITK `SetFixedInitialTransform`, ITK
    /// `ImageRegistrationMethodv4::SetFixedInitialTransform`, which reaches the
    /// metric as `SetFixedTransform` (`itkImageRegistrationMethodv4.hxx:516`)
    /// and is applied by `ImageToImageMetricv4::TransformAndEvaluateFixedPoint`
    /// (`itkImageToImageMetricv4.h:831-847`).
    ///
    /// The fixed and moving initial transforms sit on **opposite sides** of the
    /// comparison — the metric compares `F(fixed_initial(x))` against
    /// `M(moving_initial(optimized(x)))` — so setting each to the same
    /// translation displaces the two images' sample points in the same
    /// direction, and the optimum of `optimized` moves the opposite way for one
    /// versus the other.
    pub fn set_fixed_initial_transform(&mut self, transform: Transform) -> &mut Self {
        self.fixed_initial_transform = Some(transform);
        self
    }

    /// Set the virtual reference domain the metric samples in — SimpleITK
    /// `SetVirtualDomain` (`sitkImageRegistrationMethod.cxx:157-184`).
    ///
    /// `size` fixes the dimension `d`; `origin` and `spacing` must have length
    /// `d` and `direction` length `d²` (row-major). Sample points are this
    /// grid's voxel centers, independent of both images' grids. Unset (the
    /// default), the domain is the fixed image's own grid.
    ///
    /// Errors with [`RegistrationError::VirtualDomainLength`] on a length
    /// mismatch, exactly where SimpleITK raises "Expected virtualOrigin to be of
    /// length N!".
    pub fn set_virtual_domain(
        &mut self,
        size: Vec<usize>,
        origin: Vec<f64>,
        spacing: Vec<f64>,
        direction: Vec<f64>,
    ) -> Result<&mut Self> {
        let dim = size.len();
        let check = |field, got, expected| {
            if got == expected {
                Ok(())
            } else {
                Err(RegistrationError::VirtualDomainLength {
                    field,
                    got,
                    expected,
                })
            }
        };
        check("origin", origin.len(), dim)?;
        check("spacing", spacing.len(), dim)?;
        check("direction", direction.len(), dim * dim)?;

        self.virtual_domain = Some(VirtualDomain {
            size,
            origin,
            spacing,
            direction,
        });
        Ok(self)
    }

    /// Take the virtual reference domain's geometry from `image` — SimpleITK
    /// `SetVirtualDomainFromImage` (`sitkImageRegistrationMethod.cxx:186-193`),
    /// which copies its size, origin, spacing and direction and ignores its
    /// pixels.
    pub fn set_virtual_domain_from_image(&mut self, image: &Image) -> &mut Self {
        self.virtual_domain = Some(VirtualDomain {
            size: image.size().to_vec(),
            origin: image.origin().to_vec(),
            spacing: image.spacing().to_vec(),
            direction: image.direction().to_vec(),
        });
        self
    }

    /// Validate every configured transform and the virtual domain against the
    /// images' dimension. Upstream reaches the same conclusions via failed
    /// `dynamic_cast`s ("Possible miss matching dimensions!",
    /// `sitkImageRegistrationMethod.cxx:784-808`) and `sitkSTLVectorToITK`.
    fn check_dimensions(&self, fixed: &Image, moving: &Image) -> Result<()> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        let dim = fixed.dimension();
        for t in [
            self.moving_initial_transform.as_ref(),
            self.fixed_initial_transform.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if t.dimension() != dim {
                return Err(RegistrationError::TransformDimensionMismatch {
                    transform: t.dimension(),
                    image: dim,
                });
            }
        }
        if let Some(v) = &self.virtual_domain
            && v.size.len() != dim
        {
            return Err(RegistrationError::VirtualDomainLength {
                field: "size",
                got: v.size.len(),
                expected: dim,
            });
        }
        Ok(())
    }

    /// The sampling percentage for level `level`. An empty schedule means 1.0
    /// (every candidate voxel); a single entry applies to every level.
    fn sampling_percentage(&self, level: usize) -> f64 {
        match self.sampling_percentage_per_level.len() {
            0 => 1.0,
            1 => self.sampling_percentage_per_level[0],
            _ => self.sampling_percentage_per_level[level],
        }
    }

    /// Construct the metric selected by [`MetricKind`] over one already-prepared
    /// fixed/moving pair, applying `strategy` at `percentage` (with the
    /// resampled fixed mask), the interpolator, and the moving mask.
    ///
    /// `fixed` here is the fixed image **already resampled onto the virtual
    /// domain** by [`prepare_level`](Self::prepare_level), so its grid is the
    /// sample grid. The sampling arguments are explicit rather than read from
    /// `self` because [`metric_evaluate`](Self::metric_evaluate) samples densely
    /// regardless of the configured strategy — upstream sets the strategy on the
    /// *registration*, not the metric, and `EvaluateInternal` never builds a
    /// registration (ledger §3.35).
    fn build_metric(
        &self,
        fixed: &Image,
        moving: &Image,
        fixed_mask: Option<&Image>,
        strategy: SamplingStrategy,
        percentage: f64,
    ) -> Result<ActiveMetric> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        let samples = FixedSamples::from_image_with(
            fixed,
            strategy,
            percentage,
            self.sampling_seed,
            fixed_mask,
        )?;
        if samples.is_empty() {
            return Err(RegistrationError::NoValidSamples);
        }
        let mut moving_image =
            MovingImage::from_image_with_interpolator(moving, self.interpolator)?;
        if let Some(mask) = &self.moving_mask {
            moving_image = moving_image.with_moving_mask(mask)?;
        }

        match &self.metric_kind {
            MetricKind::MeanSquares => Ok(ActiveMetric::MeanSquares(
                MeanSquaresMetric::from_samples(samples, moving_image)?,
            )),
            MetricKind::MattesMutualInformation {
                number_of_histogram_bins,
            } => Ok(ActiveMetric::Mattes(
                MattesMutualInformationMetric::from_samples(
                    samples,
                    moving_image,
                    *number_of_histogram_bins,
                )?,
            )),
            MetricKind::Correlation => Ok(ActiveMetric::Correlation(
                CorrelationMetric::from_samples(samples, moving_image)?,
            )),
            // ANTS and Demons read the level's fixed image directly — the former
            // to raster its neighborhood windows, the latter for its fixed
            // gradient and spacing normalizer — so both take `fixed` alongside
            // the (possibly sparse) sample set.
            MetricKind::AntsNeighborhoodCorrelation { radius } => {
                Ok(ActiveMetric::AntsNeighborhoodCorrelation(
                    AntsNeighborhoodCorrelationMetric::from_samples(
                        fixed,
                        samples,
                        moving_image,
                        *radius,
                    )?,
                ))
            }
            MetricKind::JointHistogramMutualInformation {
                number_of_histogram_bins,
                variance_for_joint_pdf_smoothing,
            } => Ok(ActiveMetric::JointHistogram(
                JointHistogramMutualInformationMetric::from_samples(
                    samples,
                    moving_image,
                    *number_of_histogram_bins,
                    *variance_for_joint_pdf_smoothing,
                )?,
            )),
            MetricKind::Demons {
                intensity_difference_threshold,
            } => Ok(ActiveMetric::Demons(DemonsMetric::from_samples(
                fixed,
                samples,
                moving_image,
                *intensity_difference_threshold,
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
    /// A configured [`set_moving_initial_transform`], [`set_fixed_initial_transform`]
    /// and [`set_virtual_domain`] all apply; the stored
    /// [`set_initial_transform`] does not — `initial` is the transform this call
    /// optimizes. Use [`execute_with_initial_transform`] for upstream's
    /// stored-transform form, which additionally honors the in-place flag.
    ///
    /// Errors if the transform/image dimensions disagree, the shrink and
    /// smoothing schedules differ in length, the moving direction matrix is
    /// singular, scales are the wrong length, or no fixed sample maps inside the
    /// moving image at the final transform.
    ///
    /// [`set_shrink_factors_per_level`]: Self::set_shrink_factors_per_level
    /// [`set_smoothing_sigmas_per_level`]: Self::set_smoothing_sigmas_per_level
    /// [`set_moving_initial_transform`]: Self::set_moving_initial_transform
    /// [`set_fixed_initial_transform`]: Self::set_fixed_initial_transform
    /// [`set_virtual_domain`]: Self::set_virtual_domain
    /// [`set_initial_transform`]: Self::set_initial_transform
    /// [`execute_with_initial_transform`]: Self::execute_with_initial_transform
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
        self.check_dimensions(fixed, moving)?;

        let dim = fixed.dimension();
        let schedule = self.level_schedule(fixed)?;

        if let Some(mask) = &self.fixed_mask
            && mask.size() != fixed.size()
        {
            return Err(RegistrationError::MaskSizeMismatch {
                which: "fixed",
                mask: mask.size().to_vec(),
                image: fixed.size().to_vec(),
            });
        }
        // One percentage for every level, or one shared by all levels, or none.
        if self.sampling_percentage_per_level.len() > 1
            && self.sampling_percentage_per_level.len() != schedule.len()
        {
            return Err(RegistrationError::SamplingPercentageLength {
                got: self.sampling_percentage_per_level.len(),
                expected: schedule.len(),
            });
        }

        let mut transform = initial;
        let mut diagnostics = None;
        for (level, (level_factors, level_sigma)) in schedule.iter().enumerate() {
            let sigma = self.physical_sigma(fixed, *level_sigma);
            let (fixed_level, moving_level, fixed_mask_level) =
                self.prepare_level(fixed, moving, &sigma, level_factors, dim)?;
            let r = self.run_single_level(
                &fixed_level,
                &moving_level,
                fixed_mask_level.as_ref(),
                level,
                transform,
            )?;
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

    /// Register `moving` onto `fixed` starting from the transform stored by
    /// [`set_initial_transform`](Self::set_initial_transform) — SimpleITK
    /// `Execute(fixed, moving)` (`sitkImageRegistrationMethod.cxx:763-990`).
    ///
    /// The in-place flag decides what comes back and what happens to the stored
    /// transform (`:961-989`):
    ///
    /// - **in-place** (the default): the stored transform is updated to the
    ///   optimum and returned, keeping its concrete kind.
    /// - **not in-place**: the stored transform keeps its starting values, and
    ///   the optimum is returned as a [`CompositeTransform`] holding a single
    ///   sub-transform. Upstream wraps it because `sitk::Transform` has no
    ///   constructor from an arbitrary ITK transform — its own source calls this
    ///   out as a TODO (ledger §3.34).
    ///
    /// Errors with [`RegistrationError::NoInitialTransform`] when no initial
    /// transform was set, and otherwise exactly as [`execute`](Self::execute).
    pub fn execute_with_initial_transform(
        &mut self,
        fixed: &Image,
        moving: &Image,
    ) -> Result<RegistrationResult<Transform>> {
        let initial = self
            .initial_transform
            .clone()
            .ok_or(RegistrationError::NoInitialTransform)?;
        let result = self.execute(fixed, moving, initial)?;

        if self.initial_transform_in_place {
            self.initial_transform = Some(result.transform.clone());
            return Ok(result);
        }

        let mut composite = CompositeTransform::new(result.transform.dimension());
        composite.add_transform(result.transform)?;
        Ok(RegistrationResult {
            transform: composite.into(),
            metric_value: result.metric_value,
            iterations: result.iterations,
            stop_reason: result.stop_reason,
            valid_points: result.valid_points,
        })
    }

    /// Evaluate the configured metric once, at the configured transforms, with
    /// no optimization — SimpleITK `MetricEvaluate(fixed, moving)`
    /// (`sitkImageRegistrationMethod.cxx:993-1093`).
    ///
    /// The metric compares `F(fixed_initial(x))` against
    /// `M(moving_initial(initial(x)))` over every voxel center `x` of the
    /// virtual domain, and returns its value — for mean squares, the mean of the
    /// squared differences over the samples that map inside the moving image.
    ///
    /// What upstream's `EvaluateInternal` skips, and so does this:
    ///
    /// - the **optimizer** and the **parameter-scales estimator**: neither is
    ///   constructed, so `set_optimizer_*` and `set_optimizer_scales_*` have no
    ///   effect here;
    /// - the **multi-resolution schedule**: the images are used at full
    ///   resolution, unsmoothed and unshrunk;
    /// - the **metric sampling strategy, percentage and seed**: upstream sets
    ///   those on `itk::ImageRegistrationMethodv4`, which `EvaluateInternal`
    ///   never builds, so the metric samples the virtual domain densely
    ///   (ledger §3.35).
    ///
    /// The interpolator, both image masks, the virtual domain, and all three
    /// transforms *do* apply — `SetupMetric` configures them.
    ///
    /// With no initial transform set this evaluates at the identity (ledger
    /// §4.65). Upstream's `MetricEvaluate` also rejects a fixed/moving pair of
    /// differing pixel types; this port works in `f64` throughout and does not
    /// (ledger §4.67).
    pub fn metric_evaluate(&self, fixed: &Image, moving: &Image) -> Result<f64> {
        self.check_dimensions(fixed, moving)?;
        let dim = fixed.dimension();

        let identity = Transform::from(TranslationTransform::new(vec![0.0; dim]));
        let mut initial = match &self.initial_transform {
            Some(t) => t.clone(),
            None => identity,
        };
        if initial.dimension() != dim {
            return Err(RegistrationError::TransformDimensionMismatch {
                transform: initial.dimension(),
                image: dim,
            });
        }

        // Full resolution, no shrink, no smoothing: `sigma = 0` makes
        // `recursive_gaussian` a no-op and a unit shrink factor keeps the grid.
        let (fixed_level, moving_level, fixed_mask_level) =
            self.prepare_level(fixed, moving, &vec![0.0; dim], &vec![1; dim], dim)?;
        let metric = self.build_metric(
            &fixed_level,
            &moving_level,
            fixed_mask_level.as_ref(),
            SamplingStrategy::None,
            1.0,
        )?;

        let composed = Composed {
            optimized: &mut initial,
            moving_initial: self.moving_initial_transform.as_ref(),
        };
        metric.check_transform(&composed)?;
        Ok(metric.value(&composed, self.backend.as_ref()))
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
    /// `ShrinkImageFilter` (`itkImageRegistrationMethodv4.hxx:444-452`), so we
    /// take that grid's geometry, but the fixed values on it are obtained by
    /// **resampling the smoothed fixed** — matching ITK's metric, which
    /// interpolates the smoothed fixed at each virtual point. Reusing
    /// `ShrinkImageFilter`'s subsampled pixel values instead would introduce a
    /// sub-voxel translation bias, because that filter's output origin (from the
    /// real-valued center shift) and its sampling offset (that shift rounded to
    /// an integer) intentionally differ by up to half a voxel.
    ///
    /// The base grid that is shrunk is the configured **virtual domain**, or the
    /// fixed image's own grid when none is set — ITK's fallback
    /// (`itkImageRegistrationMethodv4.hxx:388-392`). The resampling transform is
    /// the **fixed-initial transform**, so the level's fixed image holds
    /// `F(fixed_initial(x))` at each virtual point `x`, which is exactly what
    /// `ImageToImageMetricv4::TransformAndEvaluateFixedPoint` evaluates
    /// per-sample (`itkImageToImageMetricv4.h:831-847`). Both default to no-ops,
    /// so a method with neither configured resamples the smoothed fixed onto its
    /// own shrunk grid through the identity, as before.
    ///
    /// A virtual point whose mapped fixed point falls outside the fixed image's
    /// buffer is **not a sample**: ITK's `TransformAndEvaluateFixedPoint` returns
    /// `pointIsValid = false` for it. Resampling alone cannot express that (it
    /// would substitute the default pixel value), so when a virtual domain or a
    /// fixed-initial transform is configured this also resamples an all-ones
    /// image over the fixed grid to get the in-buffer predicate on the virtual
    /// grid, and folds it into the level's fixed mask. With neither configured
    /// every coarse voxel center lies inside the fixed buffer by construction,
    /// so the predicate is identically true and is skipped.
    ///
    /// A configured fixed mask is carried to the level the same way, with
    /// **nearest-neighbor** interpolation and no smoothing: the mask is a binary
    /// predicate over physical space, evaluated by ITK at the *mapped fixed*
    /// point (`m_FixedImageMask->IsInsideInWorldSpace(mappedFixedPoint)`), so it
    /// travels through the fixed-initial transform exactly as the fixed image
    /// does, rather than being blurred and re-thresholded.
    ///
    /// The fixed image is read with **linear** interpolation here, whereas
    /// upstream hands `m_Interpolator` to the metric as its fixed interpolator
    /// (`sitkImageRegistrationMethod.cxx:1137-1139`); [`set_interpolator`] still
    /// selects the moving-image interpolator, which is the one the optimizer's
    /// gradient flows through (ledger §4.66).
    ///
    /// [`set_interpolator`]: Self::set_interpolator
    fn prepare_level(
        &self,
        fixed: &Image,
        moving: &Image,
        sigma: &[f64],
        factors: &[usize],
        dim: usize,
    ) -> Result<(Image, Image, Option<Image>)> {
        let smoothed_fixed = recursive_gaussian(fixed, sigma)?;
        let coarse_grid = match &self.virtual_domain {
            Some(v) => shrink(&v.grid()?, factors)?,
            None => shrink(&smoothed_fixed, factors)?,
        };

        // Output point x ↦ input point fixed_initial(x): `ResampleImageFilter`
        // maps each output voxel's physical point through the transform and
        // samples the input there, which is the mapping the metric applies.
        let onto_virtual = |input: &Image, interpolator: Interpolator| -> Result<Image> {
            let mut resampler = ResampleImageFilter::new();
            resampler
                .set_reference_image(&coarse_grid)
                .set_interpolator(interpolator)
                .set_default_pixel_value(0.0);
            Ok(match &self.fixed_initial_transform {
                Some(t) => resampler.execute(input, t)?,
                None => resampler.execute(input, &AffineTransform::identity(dim))?,
            })
        };

        let fixed_level = onto_virtual(&smoothed_fixed, Interpolator::Linear)?;
        let moving_level = recursive_gaussian(moving, sigma)?;

        let inside_fixed =
            if self.virtual_domain.is_some() || self.fixed_initial_transform.is_some() {
                let ones = with_geometry_of(fixed, vec![1.0f64; fixed.size().iter().product()])?;
                Some(onto_virtual(&ones, Interpolator::NearestNeighbor)?)
            } else {
                None
            };
        let user_mask = match &self.fixed_mask {
            Some(mask) => Some(onto_virtual(mask, Interpolator::NearestNeighbor)?),
            None => None,
        };
        let fixed_mask_level = intersect_masks(inside_fixed, user_mask)?;

        Ok((fixed_level, moving_level, fixed_mask_level))
    }

    /// Optimize `initial` against one already shrunk/smoothed fixed/moving pair
    /// — a single resolution level of [`execute`](Self::execute).
    fn run_single_level<T: ParametricTransform>(
        &self,
        fixed: &Image,
        moving: &Image,
        fixed_mask: Option<&Image>,
        level: usize,
        initial: T,
    ) -> Result<RegistrationResult<T>> {
        let metric = self.build_metric(
            fixed,
            moving,
            fixed_mask,
            self.sampling_strategy,
            self.sampling_percentage(level),
        )?;
        let nparams = initial.number_of_parameters();
        let mut transform = initial;
        let start = transform.parameters();
        let backend = self.backend.as_ref();
        let moving_initial = self.moving_initial_transform.as_ref();

        // Every metric call goes through the moving-initial composition, so the
        // optimizer sees the same objective the metric evaluates: the moving
        // image is read at `moving_initial(transform(x))`. With no
        // moving-initial transform this is a pass-through to `transform`.
        macro_rules! composed {
            () => {
                Composed {
                    optimized: &mut transform,
                    moving_initial,
                }
            };
        }

        metric.check_transform(&composed!())?;

        // ITK validates the weights' length in
        // `ObjectToObjectOptimizerBase::StartOptimization`, which every v4
        // optimizer calls — including the ones that go on to ignore the weights
        // entirely (ledger §2.117). So this check precedes the `ignores_scales`
        // split, and the length is the transform's *local* parameter count.
        let num_local = transform.number_of_local_parameters();
        if !self.optimizer_weights.is_empty() && self.optimizer_weights.len() != num_local {
            return Err(RegistrationError::OptimizerWeightsLength {
                got: self.optimizer_weights.len(),
                expected: num_local,
            });
        }

        // Both L-BFGS variants ignore parameter scales and the learning-rate
        // estimator (ITK's LBFGSBOptimizerv4/LBFGS2Optimizerv4 force identity
        // scales), so neither is built for them — they drive the raw metric
        // gradient directly.
        let ignores_scales = self.optimizer.ignores_scales();

        // An estimator is needed if scales or the learning rate are estimated.
        // When only the learning rate is, ITK still runs an estimator — the one
        // the optimizer was given — so the kind follows `scales_mode` and falls
        // back to the physical-shift default. Jacobians are parameter-independent
        // for these transforms, so building it once at the initial transform is
        // exact.
        let estimator_kind = match &self.scales_mode {
            ScalesMode::Estimated(kind) => *kind,
            ScalesMode::Unit | ScalesMode::Manual(_) => ScalesEstimatorKind::default(),
        };
        let needs_estimator = !ignores_scales
            && (matches!(self.scales_mode, ScalesMode::Estimated(_))
                || matches!(self.learning_rate_mode, LearningRateMode::Estimate(_)));
        let estimator =
            needs_estimator.then(|| metric.scales_estimator(&composed!(), estimator_kind));

        let scales: Vec<f64> = if ignores_scales {
            vec![1.0; nparams]
        } else {
            match &self.scales_mode {
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
                ScalesMode::Estimated(_) => estimator.as_ref().unwrap().estimate_scales(),
            }
        };

        // ITK modifies the gradient in place by `weights[j % n] / scales[j % n]`
        // (`ModifyGradientByScalesOverSubRange`). This crate's optimizers instead
        // *divide* the gradient by a per-parameter array, so the driver hands
        // them the reciprocal of that factor, `scales[j] / weights[j % n]`.
        // A zero weight — upstream's documented way to hold a parameter constant
        // — becomes an infinite divisor, and `g / ∞ == 0` is exactly the frozen
        // parameter `g * 0 == 0` gives ITK.
        //
        // The optimizers that ignore scales ignore the weights with them: only
        // the gradient-descent family owns `ModifyGradientByScalesOverSubRange`
        // (ledger §2.117).
        let weights = &self.optimizer_weights;
        let scales: Vec<f64> = if ignores_scales || weights_are_identity(weights) {
            scales
        } else {
            scales
                .iter()
                .enumerate()
                .map(|(j, &s)| s / weights[j % weights.len()])
                .collect()
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
                    let mut c = composed!();
                    // The closure signature the optimizer requires has no
                    // fallible surface; it always probes at `nparams`.
                    c.set_parameters(p)
                        .expect("optimizer probes at the transform's own parameter dimension");
                    let m = metric.evaluate(&c, backend);
                    (m.value, m.derivative)
                }
            };
        }
        // The line searches take an `Objective` so their golden-section probes
        // reach `ActiveMetric::value` instead of the closure adapter's
        // discard-the-gradient fallback.
        macro_rules! line_search_objective {
            () => {
                MetricObjective {
                    transform: composed!(),
                    metric: &metric,
                    backend,
                }
            };
        }
        // The same objective for the gradient-free optimizers, which consume the
        // value alone.
        macro_rules! value_objective {
            () => {
                |p: &[f64]| {
                    let mut c = composed!();
                    c.set_parameters(p)
                        .expect("optimizer probes at the transform's own parameter dimension");
                    metric.value(&c, backend)
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
                        let mut c = composed!();
                        c.set_parameters(&start)?;
                        let m0 = metric.evaluate(&c, backend);
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
                        let mut c = composed!();
                        c.set_parameters(&start)?;
                        let m0 = metric.evaluate(&c, backend);
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
            OptimizerKind::LineSearch(ls) => {
                let mut optimizer = ls.clone();
                optimizer.set_scales(scales.clone());
                match self.learning_rate_mode {
                    // Caller-supplied fixed base rate; the golden-section search
                    // adapts it each iteration.
                    LearningRateMode::Manual => {
                        optimizer.optimize_objective(start, line_search_objective!())
                    }
                    // ITK's line search estimates the base rate once, at the
                    // first iteration; the golden-section search then adapts it
                    // every step. No per-iteration re-estimation and no one-voxel
                    // step cap are needed — an over-large estimate from a
                    // near-converged restart is corrected by the first line
                    // search, since a step that overshoots raises the metric and
                    // the search rejects it. (Both estimate modes collapse to
                    // this single once-estimation for this optimizer, so the
                    // setter fixes the mode to `Once`.)
                    LearningRateMode::Estimate(_) => {
                        let est = estimator.as_ref().unwrap();
                        let mut c = composed!();
                        c.set_parameters(&start)?;
                        let m0 = metric.evaluate(&c, backend);
                        let lr_once = est.estimate_learning_rate(&scaled(&m0.derivative));
                        optimizer.set_learning_rate(lr_once);
                        optimizer.optimize_objective(start, line_search_objective!())
                    }
                }
            }
            OptimizerKind::ConjugateGradientLineSearch(cg) => {
                let mut optimizer = cg.clone();
                optimizer.set_scales(scales.clone());
                match self.learning_rate_mode {
                    // Caller-supplied fixed base rate; the golden-section search
                    // adapts it each iteration along the conjugate direction.
                    LearningRateMode::Manual => {
                        optimizer.optimize_objective(start, line_search_objective!())
                    }
                    // Base rate estimated once from the initial gradient, exactly
                    // as the plain line search does — ITK's conjugate optimizer
                    // also calls EstimateLearningRate only at m_CurrentIteration
                    // == 0, and the golden-section search adapts it thereafter, so
                    // no per-iteration re-estimation or step cap is needed.
                    LearningRateMode::Estimate(_) => {
                        let est = estimator.as_ref().unwrap();
                        let mut c = composed!();
                        c.set_parameters(&start)?;
                        let m0 = metric.evaluate(&c, backend);
                        let lr_once = est.estimate_learning_rate(&scaled(&m0.derivative));
                        optimizer.set_learning_rate(lr_once);
                        optimizer.optimize_objective(start, line_search_objective!())
                    }
                }
            }
            OptimizerKind::Lbfgsb(cfg) => {
                // No scales, no learning-rate estimation: LBFGSB minimizes the raw
                // metric directly under its own bound/convergence configuration.
                let optimizer = cfg.build(nparams);
                optimizer.optimize(start, objective!())
            }
            OptimizerKind::Lbfgs2(l) => {
                // Unbounded L-BFGS; like LBFGSB it ignores scales and drives the
                // raw metric gradient through its own line search.
                l.optimize(start, objective!())
            }
            // The four gradient-free optimizers below take the *unscaled*
            // parameters and apply `scales` themselves — each ports ITK's
            // `SingleValuedVnlCostFunctionAdaptorv4` internal/external mapping
            // (`internal = external · scales`) — so the driver neither pre-scales
            // the start point nor post-scales the result.
            OptimizerKind::Amoeba(a) => {
                let mut optimizer = a.clone();
                optimizer.set_scales(scales.clone());
                optimizer.optimize(start, value_objective!())
            }
            OptimizerKind::Powell(p) => {
                let mut optimizer = p.clone();
                optimizer.set_scales(scales.clone());
                optimizer.optimize(start, value_objective!())
            }
            OptimizerKind::OnePlusOneEvolutionary(e) => {
                let mut optimizer = e.clone();
                optimizer.set_scales(scales.clone());
                optimizer.optimize(start, value_objective!())
            }
            OptimizerKind::Exhaustive(e) => {
                let mut optimizer = e.clone();
                optimizer.set_scales(scales.clone());
                optimizer.optimize(start, value_objective!())
            }
        };

        let final_metric = {
            let mut c = composed!();
            c.set_parameters(&result.parameters)?;
            metric.evaluate(&c, backend)
        };
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
mod initial_transform_tests {
    use super::*;
    use sitk_transform::{Euler2DTransform, ScaleTransform};

    /// A 2-D Gaussian blob of width `sigma` centred at the **physical** point
    /// `center`, sampled on a grid of `size` voxels at `spacing` from `origin`
    /// (identity direction). The same continuous function sampled on two
    /// different grids gives two images that agree in physical space, which is
    /// what lets a virtual domain unrelated to either grid still register them.
    fn blob(
        size: &[usize],
        spacing: &[f64],
        origin: &[f64],
        center: [f64; 2],
        sigma: f64,
    ) -> Image {
        let (w, h) = (size[0], size[1]);
        let s2 = 2.0 * sigma * sigma;
        let mut v = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let px = origin[0] + x as f64 * spacing[0];
                let py = origin[1] + y as f64 * spacing[1];
                let (dx, dy) = (px - center[0], py - center[1]);
                v[y * w + x] = (-(dx * dx + dy * dy) / s2).exp();
            }
        }
        let mut image = Image::from_vec(&[w, h], v).unwrap();
        image.set_spacing(spacing).unwrap();
        image.set_origin(origin).unwrap();
        image
    }

    /// The 3×3 ramp `f(x, y) = 3y + x`, unit spacing at the origin. Every
    /// one-voxel step in `+x` raises it by exactly 1, so a translated
    /// mean-squares evaluation over it is exact integer arithmetic.
    fn ramp3x3() -> Image {
        Image::from_vec(&[3, 3], (0..9).map(f64::from).collect::<Vec<f64>>()).unwrap()
    }

    fn translation(tx: f64, ty: f64) -> Transform {
        TranslationTransform::new(vec![tx, ty]).into()
    }

    /// A fully-recovered translation is the fixture the composition-order tests
    /// read: two identical blobs, so the optimum of `optimized` is whatever the
    /// initial transforms leave for it to undo.
    fn recover_translation(
        reg: &ImageRegistrationMethod,
        fixed: &Image,
        moving: &Image,
    ) -> Vec<f64> {
        reg.execute(fixed, moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap()
            .transform
            .parameters()
    }

    // ---- MetricEvaluate --------------------------------------------------

    /// Mean squares of the ramp against itself under a one-voxel translation.
    ///
    /// The metric samples the moving image at `T(x) = x + (1, 0)`. For the six
    /// virtual points with `x < 2` that lands one voxel right, where the ramp is
    /// exactly 1 greater; the three points at `x == 2` map to continuous index
    /// 3.0, outside a size-3 buffer, and are not samples. So the value is
    /// `(1/6)·Σ 1² = 1`, with no floating-point slack at all.
    #[test]
    fn metric_evaluate_of_a_one_voxel_translation_is_hand_computable() {
        let image = ramp3x3();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_initial_transform(translation(1.0, 0.0));

        assert_eq!(reg.metric_evaluate(&image, &image).unwrap(), 1.0);
    }

    /// Two voxels right: only the `x == 0` column stays inside, its three
    /// samples each differing by exactly 2, so the value is `(1/3)·Σ 2² = 4`.
    #[test]
    fn metric_evaluate_scales_with_the_squared_difference() {
        let image = ramp3x3();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_initial_transform(translation(2.0, 0.0));

        assert_eq!(reg.metric_evaluate(&image, &image).unwrap(), 4.0);
    }

    /// With no initial transform the metric evaluates at the identity, so an
    /// image against itself scores 0.
    #[test]
    fn metric_evaluate_defaults_to_the_identity_transform() {
        let image = ramp3x3();
        let reg = ImageRegistrationMethod::new();
        assert_eq!(reg.metric_evaluate(&image, &image).unwrap(), 0.0);
    }

    /// `EvaluateInternal` builds no `itk::ImageRegistrationMethodv4`, and the
    /// sampling strategy lives on that object — so `MetricEvaluate` samples the
    /// virtual domain densely no matter how the strategy is configured
    /// (ledger §3.35). Same for the multi-resolution schedule.
    #[test]
    fn metric_evaluate_ignores_the_sampling_strategy_and_the_pyramid() {
        let image = ramp3x3();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_initial_transform(translation(1.0, 0.0))
            .set_metric_sampling_strategy(SamplingStrategy::Random)
            .set_metric_sampling_percentage(0.1, 7)
            .set_shrink_factors_per_level(vec![4, 1])
            .set_smoothing_sigmas_per_level(vec![2.0, 0.0]);

        assert_eq!(reg.metric_evaluate(&image, &image).unwrap(), 1.0);
    }

    /// `MetricEvaluate` performs no optimization: it leaves the stored initial
    /// transform exactly as it was set, and takes `&self`.
    #[test]
    fn metric_evaluate_does_not_optimize_the_initial_transform() {
        let image = ramp3x3();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_initial_transform(translation(2.0, 0.0));

        reg.metric_evaluate(&image, &image).unwrap();
        assert_eq!(
            reg.initial_transform().unwrap().parameters(),
            vec![2.0, 0.0]
        );
    }

    // ---- composition order -----------------------------------------------

    /// The moving-initial transform is applied **after** the optimized one:
    /// `moving_initial(optimized(x))`, per `itk::CompositeTransform`'s
    /// reverse-queue evaluation. Pinned against the reversed order with a
    /// non-commuting pair — a scale and a translation.
    #[test]
    fn composed_applies_the_optimized_transform_before_the_moving_initial_one() {
        let moving_initial: Transform = ScaleTransform::new(vec![2.0, 2.0], vec![0.0, 0.0]).into();
        let mut optimized = TranslationTransform::new(vec![1.0, 0.0]);
        let composed = Composed {
            optimized: &mut optimized,
            moving_initial: Some(&moving_initial),
        };

        // scale(translate(x)) = 2·(x + (1,0)) = (2x₀ + 2, 2x₁)
        assert_eq!(composed.transform_point(&[3.0, 5.0]), vec![8.0, 10.0]);
        // The reversed order, translate(scale(x)) = (2x₀ + 1, 2x₁), would give
        // [7, 10] — this is the assertion that fixes the order.
    }

    /// Without a moving-initial transform the composition is the identity
    /// wrapper: every accessor is the optimized transform's own.
    #[test]
    fn composed_without_a_moving_initial_transform_is_a_pass_through() {
        let mut optimized = Euler2DTransform::new(0.3, [1.0, 2.0], [3.0, 4.0]);
        let expected_point = optimized.transform_point(&[5.0, 6.0]);
        let expected_jacobian = optimized.jacobian_wrt_parameters(&[5.0, 6.0]);
        let composed = Composed {
            optimized: &mut optimized,
            moving_initial: None,
        };

        assert_eq!(composed.transform_point(&[5.0, 6.0]), expected_point);
        assert_eq!(
            composed.jacobian_wrt_parameters(&[5.0, 6.0]),
            expected_jacobian
        );
    }

    /// The chain-ruled parameter Jacobian `J_g(f(x))·J_f(x)` is the true
    /// derivative of the composed map, so the gradient the optimizer descends is
    /// the gradient of the objective the metric evaluates.
    #[test]
    fn composed_parameter_jacobian_matches_a_finite_difference() {
        let moving_initial: Transform = Euler2DTransform::new(0.4, [2.0, -1.0], [1.0, 1.0]).into();
        let mut optimized = Euler2DTransform::new(0.2, [1.0, 3.0], [4.0, 2.0]);
        let point = [7.0, -3.0];

        let (dim, nparams) = (2usize, 3usize);
        let analytic = {
            let composed = Composed {
                optimized: &mut optimized,
                moving_initial: Some(&moving_initial),
            };
            composed.jacobian_wrt_parameters(&point)
        };

        let h = 1e-6;
        let base = optimized.parameters();
        for k in 0..nparams {
            let mut plus = base.clone();
            plus[k] += h;
            let mut minus = base.clone();
            minus[k] -= h;

            let at = |p: &[f64]| {
                let mut t = optimized.clone();
                t.set_parameters(p).unwrap();
                let composed = Composed {
                    optimized: &mut t,
                    moving_initial: Some(&moving_initial),
                };
                composed.transform_point(&point)
            };
            let (fp, fm) = (at(&plus), at(&minus));
            for i in 0..dim {
                let fd = (fp[i] - fm[i]) / (2.0 * h);
                assert!(
                    (analytic[i * nparams + k] - fd).abs() < 1e-5,
                    "d(out[{i}])/d(p[{k}]): analytic {} vs finite difference {fd}",
                    analytic[i * nparams + k]
                );
            }
        }
    }

    /// The fixed- and moving-initial transforms sit on opposite sides of the
    /// metric's comparison, so the same translation pushes the optimum the
    /// opposite way.
    ///
    /// With identical images the metric compares `F(fixed_initial(x))` against
    /// `M(moving_initial(optimized(x)))`:
    ///
    /// - a moving-initial `+d` is undone by `optimized = −d`, because the moving
    ///   sample point is `x + optimized + d` and must land back on `x`;
    /// - a fixed-initial `+d` moves the fixed sample point to `x + d`, which the
    ///   moving sample point `x + optimized` matches only at `optimized = +d`.
    #[test]
    fn moving_and_fixed_initial_transforms_displace_the_optimum_in_opposite_directions() {
        let size = [40usize, 40];
        let spacing = [1.0f64, 1.0];
        let origin = [0.0f64, 0.0];
        let image = blob(&size, &spacing, &origin, [20.0, 20.0], 7.0);
        let (dx, dy) = (3.0f64, -2.0);

        let configure = |reg: &mut ImageRegistrationMethod| {
            reg.set_metric_as_mean_squares()
                .set_optimizer_scales_from_physical_shift()
                .set_optimizer_as_regular_step_gradient_descent_estimated(
                    1e-6,
                    300,
                    1e-8,
                    EstimateLearningRate::Once,
                );
        };

        let mut moving_side = ImageRegistrationMethod::new();
        configure(&mut moving_side);
        moving_side.set_moving_initial_transform(translation(dx, dy));
        let p_moving = recover_translation(&moving_side, &image, &image);

        let mut fixed_side = ImageRegistrationMethod::new();
        configure(&mut fixed_side);
        fixed_side.set_fixed_initial_transform(translation(dx, dy));
        let p_fixed = recover_translation(&fixed_side, &image, &image);

        assert!(
            (p_moving[0] + dx).abs() < 1e-2 && (p_moving[1] + dy).abs() < 1e-2,
            "moving-initial optimum {p_moving:?}, expected [{}, {}]",
            -dx,
            -dy
        );
        assert!(
            (p_fixed[0] - dx).abs() < 1e-2 && (p_fixed[1] - dy).abs() < 1e-2,
            "fixed-initial optimum {p_fixed:?}, expected [{dx}, {dy}]"
        );
        // The statement of the test: opposite signs on both axes.
        assert!(p_moving[0] * p_fixed[0] < 0.0 && p_moving[1] * p_fixed[1] < 0.0);
    }

    /// A moving-initial transform absorbs the alignment an earlier stage already
    /// found: with it set to the true offset the optimizer starts at the optimum
    /// and stays there, and the metric there is ~0.
    #[test]
    fn a_moving_initial_transform_carries_a_previous_stage_alignment() {
        let size = [40usize, 40];
        let (spacing, origin) = ([1.0f64, 1.0], [0.0f64, 0.0]);
        let fixed = blob(&size, &spacing, &origin, [20.0, 20.0], 7.0);
        let moving = blob(&size, &spacing, &origin, [23.0, 18.0], 7.0);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_moving_initial_transform(translation(3.0, -2.0))
            .set_initial_transform(translation(0.0, 0.0));

        // F(x) vs M(x + (3,−2)) — the blobs coincide, so mean squares is ~0.
        let value = reg.metric_evaluate(&fixed, &moving).unwrap();
        assert!(value < 1e-12, "metric at the composed optimum was {value}");

        // Without it, the same evaluation sees the full 3-voxel misalignment.
        let mut plain = ImageRegistrationMethod::new();
        plain.set_metric_as_mean_squares();
        assert!(plain.metric_evaluate(&fixed, &moving).unwrap() > 1e-3);
    }

    // ---- in-place ---------------------------------------------------------

    fn in_place_fixture() -> (Image, Image) {
        let size = [40usize, 40];
        let (spacing, origin) = ([1.0f64, 1.0], [0.0f64, 0.0]);
        (
            blob(&size, &spacing, &origin, [20.0, 20.0], 7.0),
            blob(&size, &spacing, &origin, [23.0, 18.0], 7.0),
        )
    }

    fn tuned(reg: &mut ImageRegistrationMethod) {
        reg.set_metric_as_mean_squares()
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-6,
                300,
                1e-8,
                EstimateLearningRate::Once,
            );
    }

    /// In place: the stored transform is updated to the optimum and `Execute`
    /// hands back that same concrete transform kind.
    #[test]
    fn in_place_execute_updates_the_stored_transform_and_keeps_its_kind() {
        let (fixed, moving) = in_place_fixture();
        let mut reg = ImageRegistrationMethod::new();
        tuned(&mut reg);
        reg.set_initial_transform(translation(0.0, 0.0));
        assert!(reg.initial_transform_in_place());

        let result = reg.execute_with_initial_transform(&fixed, &moving).unwrap();

        assert!(matches!(result.transform, Transform::Translation(_)));
        let optimum = result.transform.parameters();
        assert!((optimum[0] - 3.0).abs() < 1e-2 && (optimum[1] + 2.0).abs() < 1e-2);
        assert_eq!(reg.initial_transform().unwrap().parameters(), optimum);
    }

    /// Not in place: the stored transform keeps its starting values, and the
    /// optimum comes back wrapped in a single-entry `CompositeTransform`
    /// (ledger §3.34).
    #[test]
    fn not_in_place_execute_leaves_the_stored_transform_and_returns_a_composite() {
        let (fixed, moving) = in_place_fixture();
        let mut reg = ImageRegistrationMethod::new();
        tuned(&mut reg);
        reg.set_initial_transform_in_place(translation(0.0, 0.0), false);
        assert!(!reg.initial_transform_in_place());

        let result = reg.execute_with_initial_transform(&fixed, &moving).unwrap();

        let sub = match &result.transform {
            Transform::Composite(c) => {
                assert_eq!(c.number_of_transforms(), 1);
                c.nth_transform(0).unwrap()
            }
            other => panic!("expected a composite, got {other:?}"),
        };
        assert!(matches!(sub, Transform::Translation(_)));
        let optimum = sub.parameters();
        assert!((optimum[0] - 3.0).abs() < 1e-2 && (optimum[1] + 2.0).abs() < 1e-2);

        // The stored transform never moved.
        assert_eq!(
            reg.initial_transform().unwrap().parameters(),
            vec![0.0, 0.0]
        );
    }

    /// The two modes reach the same optimum; they differ only in what they
    /// write back and what they hand out.
    #[test]
    fn in_place_and_not_in_place_agree_on_the_optimum() {
        let (fixed, moving) = in_place_fixture();

        let mut a = ImageRegistrationMethod::new();
        tuned(&mut a);
        a.set_initial_transform_in_place(translation(0.0, 0.0), true);
        let ra = a.execute_with_initial_transform(&fixed, &moving).unwrap();

        let mut b = ImageRegistrationMethod::new();
        tuned(&mut b);
        b.set_initial_transform_in_place(translation(0.0, 0.0), false);
        let rb = b.execute_with_initial_transform(&fixed, &moving).unwrap();

        assert_eq!(ra.transform.parameters(), rb.transform.parameters());
        assert_eq!(ra.metric_value, rb.metric_value);
    }

    /// The single-argument setter turns the in-place flag back on, exactly as
    /// `SetInitialTransform(const Transform &)` does after a
    /// `SetInitialTransform(t, false)`.
    #[test]
    fn the_single_argument_setter_restores_the_in_place_flag() {
        let mut reg = ImageRegistrationMethod::new();
        reg.set_initial_transform_in_place(translation(0.0, 0.0), false);
        assert!(!reg.initial_transform_in_place());
        reg.set_initial_transform(translation(0.0, 0.0));
        assert!(reg.initial_transform_in_place());
    }

    #[test]
    fn execute_without_an_initial_transform_errors() {
        let (fixed, moving) = in_place_fixture();
        let mut reg = ImageRegistrationMethod::new();
        assert!(matches!(
            reg.execute_with_initial_transform(&fixed, &moving),
            Err(RegistrationError::NoInitialTransform)
        ));
    }

    // ---- virtual domain ---------------------------------------------------

    /// The virtual domain is the grid the metric samples in, and it is unrelated
    /// to either image's grid. Here all three geometries differ — the fixed
    /// image is 40² at spacing 1 from the origin, the moving image 80² at
    /// spacing 0.5 from `(−5, −5)`, and the virtual domain 20² at spacing 1.5
    /// from `(8, 8)` — and the same physical translation is still recovered,
    /// because every transform acts in physical space.
    #[test]
    fn a_virtual_domain_unrelated_to_both_images_still_recovers_the_translation() {
        let (tx, ty) = (3.0f64, -2.0);
        let fixed = blob(&[40, 40], &[1.0, 1.0], &[0.0, 0.0], [20.0, 20.0], 7.0);
        let moving = blob(
            &[80, 80],
            &[0.5, 0.5],
            &[-5.0, -5.0],
            [20.0 + tx, 20.0 + ty],
            7.0,
        );

        let mut reg = ImageRegistrationMethod::new();
        tuned(&mut reg);
        reg.set_virtual_domain(
            vec![20, 20],
            vec![8.0, 8.0],
            vec![1.5, 1.5],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        let p = recover_translation(&reg, &fixed, &moving);
        assert!(
            (p[0] - tx).abs() < 5e-2 && (p[1] - ty).abs() < 5e-2,
            "recovered {p:?}, expected [{tx}, {ty}]"
        );
    }

    /// A virtual domain equal to the fixed image's geometry is ITK's own
    /// fallback, so setting it must change nothing.
    #[test]
    fn a_virtual_domain_equal_to_the_fixed_grid_is_a_no_op() {
        let (fixed, moving) = in_place_fixture();

        let mut plain = ImageRegistrationMethod::new();
        plain.set_metric_as_mean_squares();
        let a = plain.metric_evaluate(&fixed, &moving).unwrap();

        let mut with_domain = ImageRegistrationMethod::new();
        with_domain
            .set_metric_as_mean_squares()
            .set_virtual_domain_from_image(&fixed);
        let b = with_domain.metric_evaluate(&fixed, &moving).unwrap();

        assert_eq!(a, b);
    }

    /// `SetVirtualDomainFromImage` copies the image's four geometry fields and
    /// nothing else — a virtual domain taken from the *moving* image samples on
    /// the moving grid.
    #[test]
    fn set_virtual_domain_from_image_copies_the_geometry() {
        let moving = blob(&[80, 80], &[0.5, 0.5], &[-5.0, -5.0], [20.0, 20.0], 7.0);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_virtual_domain_from_image(&moving);

        let domain = reg.virtual_domain.as_ref().unwrap();
        assert_eq!(domain.size, moving.size());
        assert_eq!(domain.origin, moving.origin());
        assert_eq!(domain.spacing, moving.spacing());
        assert_eq!(domain.direction, moving.direction());
    }

    /// A virtual point whose mapped fixed point falls outside the fixed image is
    /// not a sample (ITK's `pointIsValid == false`), rather than a sample of the
    /// resampler's default pixel value.
    ///
    /// The moving image is one column *wider* than the fixed one, and the
    /// virtual domain covers that wider extent. The last virtual column maps to
    /// fixed index 3 — outside a size-3 buffer — while mapping to a perfectly
    /// valid moving voxel, so only the fixed side can drop it. The two images
    /// agree everywhere they overlap, so:
    ///
    /// - dropping the column (ITK's behavior, and this port's) gives 0;
    /// - zero-filling it from the resampler's default pixel value would give
    ///   `(1/12)·3·100² = 2500`.
    #[test]
    fn virtual_points_outside_the_fixed_image_are_dropped_not_zero_filled() {
        let fixed = ramp3x3();
        let moving = Image::from_vec(
            &[4, 3],
            (0..3)
                .flat_map(|y| (0..4).map(move |x| if x < 3 { f64::from(3 * y + x) } else { 100.0 }))
                .collect::<Vec<f64>>(),
        )
        .unwrap();

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_virtual_domain(
                vec![4, 3],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap();

        assert_eq!(reg.metric_evaluate(&fixed, &moving).unwrap(), 0.0);
    }

    /// The fixed-initial transform reaches the fixed image the same way, so a
    /// fixed-initial `+1` voxel shift reads the ramp one voxel right — the
    /// mirror of the moving-side evaluation, and the sample that maps outside is
    /// again dropped. `F(x + 1) − M(x) = 1` over the six survivors.
    #[test]
    fn a_fixed_initial_transform_shifts_the_fixed_sample_point() {
        let image = ramp3x3();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares()
            .set_fixed_initial_transform(translation(1.0, 0.0));

        assert_eq!(reg.metric_evaluate(&image, &image).unwrap(), 1.0);
    }

    #[test]
    fn set_virtual_domain_rejects_mismatched_vector_lengths() {
        let mut reg = ImageRegistrationMethod::new();
        assert!(matches!(
            reg.set_virtual_domain(vec![4, 4], vec![0.0], vec![1.0, 1.0], vec![1.0; 4]),
            Err(RegistrationError::VirtualDomainLength {
                field: "origin",
                got: 1,
                expected: 2
            })
        ));
        assert!(matches!(
            reg.set_virtual_domain(vec![4, 4], vec![0.0; 2], vec![1.0; 2], vec![1.0; 3]),
            Err(RegistrationError::VirtualDomainLength {
                field: "direction",
                got: 3,
                expected: 4
            })
        ));
    }

    #[test]
    fn a_three_dimensional_initial_transform_on_two_dimensional_images_errors() {
        let image = ramp3x3();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_moving_initial_transform(TranslationTransform::new(vec![0.0; 3]).into());
        assert!(matches!(
            reg.metric_evaluate(&image, &image),
            Err(RegistrationError::TransformDimensionMismatch {
                transform: 3,
                image: 2
            })
        ));
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
    fn lbfgsb_recovers_a_translation() {
        // Unbounded LBFGSB drives the raw metric gradient (no scales, no learning
        // rate) through its line search and recovers the translation end-to-end.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        // Tight pgtol/factr so it converges to high precision on the smooth basin.
        reg.set_optimizer_as_lbfgsb(1e-8, 500, 5, 2000, 1e3, f64::MIN, f64::MAX);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}, stop {:?}",
            result.metric_value,
            result.iterations,
            result.stop_reason
        );
    }

    #[test]
    fn line_search_recovers_a_translation_with_estimated_rate() {
        // Golden-section line search with the base learning rate estimated once
        // from physical shift: the per-step rate search recovers the translation
        // end-to-end with no hand-tuned rate.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_line_search_estimated(300);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}, stop {:?}",
            result.metric_value,
            result.iterations,
            result.stop_reason
        );
    }

    #[test]
    fn line_search_recovers_a_translation_with_manual_rate() {
        // With a caller-supplied base rate and physical-shift scales, the line
        // search still aligns: a base rate far larger than optimal is tamed by
        // the golden-section search each iteration rather than overshooting, as a
        // fixed-rate gradient descent at the same rate would.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_line_search(5.0, 300);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}, stop {:?}",
            result.metric_value,
            result.iterations,
            result.stop_reason
        );
    }

    #[test]
    fn conjugate_gradient_line_search_recovers_a_translation_with_estimated_rate() {
        // Conjugate-gradient line search with the base learning rate estimated
        // once from physical shift recovers the translation end-to-end with no
        // hand-tuned rate.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_conjugate_gradient_line_search_estimated(300);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}, stop {:?}",
            result.metric_value,
            result.iterations,
            result.stop_reason
        );
    }

    #[test]
    fn conjugate_gradient_line_search_recovers_a_translation_with_manual_rate() {
        // With a caller-supplied base rate and physical-shift scales, the
        // conjugate-gradient line search aligns the images; the golden-section
        // search tames a base rate larger than optimal each iteration.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_conjugate_gradient_line_search(5.0, 300);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}, stop {:?}",
            result.metric_value,
            result.iterations,
            result.stop_reason
        );
    }

    #[test]
    fn lbfgsb_respects_parameter_bounds() {
        // The true shift (3, −2) lies outside the box [−1.5, 1.5]²; the
        // box-constrained minimizer of the mean-squares objective (monotone within
        // the box, which sits entirely on the basin's near side) is the nearest
        // corner (1.5, −1.5). Verify the bounds bind and are respected.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_lbfgsb(1e-8, 500, 5, 2000, 1e5, -1.5, 1.5);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (-1.5 - 1e-9..=1.5 + 1e-9).contains(&p[0])
                && (-1.5 - 1e-9..=1.5 + 1e-9).contains(&p[1]),
            "recovered {p:?} outside the box [-1.5, 1.5]"
        );
        assert!(
            (p[0] - 1.5).abs() < 1e-2 && (p[1] + 1.5).abs() < 1e-2,
            "recovered {p:?}, expected the constrained corner [1.5, -1.5]"
        );
    }

    #[test]
    fn lbfgsb_lower_bound_only_binds_the_out_of_range_parameter() {
        // A scalar lower bound of 0 (upper unbounded → nbd = 1, lower only) applied
        // to both parameters. True shift (3, −2): p0 = 3 stays feasible and is
        // recovered, while p1's free optimum −2 is below 0, so it pins to the
        // lower bound. Exercises the lower-only bound-selection mapping end-to-end.
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_lbfgsb(1e-8, 500, 5, 2000, 1e3, 0.0, f64::MAX);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            p[0] >= -1e-9 && p[1] >= -1e-9,
            "recovered {p:?} below the lower bound 0"
        );
        // p1 is pinned exactly to the lower bound; p0 recovers near 3 (its
        // residual is looser than the unbounded case because the x-gradient is
        // weighted by the smaller y-overlap once p1 is held off the true −2).
        assert!(
            (p[0] - tx).abs() < 5e-2 && p[1].abs() < 1e-3,
            "recovered {p:?}, expected p0≈{tx} (feasible) and p1 pinned to 0"
        );
    }

    /// The registration used by every gradient-free / interpolator / sampling
    /// test below: a 40×40 Gaussian blob shifted by `(3, −2)`, recovered as a
    /// translation from the identity.
    fn shifted_blob_pair() -> (Image, Image, f64, f64) {
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(40, 40, 20.0, 20.0, 7.0, 1.0);
        let moving = gaussian(40, 40, 20.0 + tx, 20.0 + ty, 7.0, 1.0);
        (fixed, moving, tx, ty)
    }

    #[test]
    fn amoeba_recovers_a_translation() {
        // Gradient-free: the simplex never asks the metric for a derivative.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_amoeba(2.0, 400, 1e-10, 1e-12, false);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}",
            result.metric_value
        );
    }

    #[test]
    fn powell_recovers_a_translation() {
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_powell(100, 100, 1.0, 1e-8, 1e-10);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}",
            result.metric_value
        );
    }

    #[test]
    fn one_plus_one_evolutionary_recovers_a_translation() {
        // Seeded, so the stochastic search is reproducible run to run.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_one_plus_one_evolutionary(800, 1.5e-4, 1.01, -1.0, -1.0, 42);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-2 && (p[1] - ty).abs() < 1e-2,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}",
            result.metric_value
        );
    }

    #[test]
    fn exhaustive_lands_on_the_exact_grid_point() {
        // The true shift (3, −2) is a point of the ±4-step unit grid centred on
        // the identity, and the metric there is exactly zero, so the brute-force
        // scan must return it exactly — no local refinement, no tolerance.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_exhaustive(vec![4, 4], 1.0);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert_eq!((p[0], p[1]), (tx, ty), "recovered {p:?}");
    }

    #[test]
    fn lbfgs2_recovers_a_translation() {
        // Unconstrained L-BFGS on the raw metric gradient: no scales, no
        // learning rate. `number_of_iterations == 0` means "run to convergence".
        // `solution_accuracy` is tightened from SimpleITK's 1e-5 default: it is
        // a *gradient-norm* tolerance (‖g‖ ≤ eps·max(1, ‖x‖)), and the mean-
        // squares gradient of a σ=7 blob is small enough that 1e-5 stops about
        // 1e-3 short of the true shift.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_lbfgs2(1e-8, 0, 6, 0, 1e-5, 40, 1e-20, 1e20, 1e-4);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}",
            result.metric_value
        );
    }

    #[test]
    fn nearest_neighbor_interpolator_drives_the_exhaustive_scan() {
        // Nearest neighbor makes the metric piecewise constant, so its analytic
        // derivative is zero almost everywhere and gradient descent cannot move.
        // A grid scan is exactly the optimizer that does not care: on the
        // integer grid the nearest-neighbor read is the moving voxel itself, so
        // the true shift still scores an exact zero.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_interpolator(Interpolator::NearestNeighbor)
            .set_optimizer_as_exhaustive(vec![4, 4], 1.0);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert_eq!((p[0], p[1]), (tx, ty), "recovered {p:?}");
        assert!(
            result.metric_value < 1e-24,
            "metric {} at the exact shift",
            result.metric_value
        );
    }

    #[test]
    fn bspline_and_gaussian_interpolators_recover_a_translation() {
        // Both are smoother than linear; each must drive the same gradient
        // descent to the same shift.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        for interp in [Interpolator::BSpline, Interpolator::Gaussian] {
            let mut reg = ImageRegistrationMethod::new();
            reg.set_interpolator(interp)
                .set_optimizer_scales_from_physical_shift()
                .set_optimizer_as_gradient_descent_estimated(500, EstimateLearningRate::Once);
            let result = reg
                .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
                .unwrap();

            let p = result.transform.parameters();
            assert!(
                (p[0] - tx).abs() < 5e-3 && (p[1] - ty).abs() < 5e-3,
                "{interp:?} recovered {p:?}, expected [{tx}, {ty}]"
            );
        }
    }

    /// The number of fixed samples `reg` draws, counted by evaluating the metric
    /// once at the identity — where every sample maps inside the moving image,
    /// so `valid_points` is exactly the sample count. (At a nonzero shift the
    /// samples near the trailing border map out and are dropped, which is why
    /// the recovered run's `valid_points` is always smaller.)
    fn sample_count_at_identity(reg: &mut ImageRegistrationMethod, fixed: &Image) -> usize {
        reg.set_optimizer_as_gradient_descent(0.0, 1);
        reg.execute(fixed, fixed, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap()
            .valid_points
    }

    #[test]
    fn regular_sampling_uses_every_fourth_voxel_and_still_registers() {
        // 25% regular sampling strides by ceil(1/0.25) = 4 over the 1600 voxels,
        // so exactly 400 samples remain.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut counter = ImageRegistrationMethod::new();
        counter
            .set_metric_sampling_strategy(SamplingStrategy::Regular)
            .set_metric_sampling_percentage(0.25, 0);
        assert_eq!(sample_count_at_identity(&mut counter, &fixed), 400);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_sampling_strategy(SamplingStrategy::Regular)
            .set_metric_sampling_percentage(0.25, 0)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(500, EstimateLearningRate::Once);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-2 && (p[1] - ty).abs() < 1e-2,
            "recovered {p:?}, expected [{tx}, {ty}]"
        );
    }

    #[test]
    fn random_sampling_draws_the_requested_count_and_still_registers() {
        // 25% of 1600 = 400 draws (with replacement), seeded for reproducibility.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut counter = ImageRegistrationMethod::new();
        counter
            .set_metric_sampling_strategy(SamplingStrategy::Random)
            .set_metric_sampling_percentage(0.25, 7);
        assert_eq!(sample_count_at_identity(&mut counter, &fixed), 400);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_sampling_strategy(SamplingStrategy::Random)
            .set_metric_sampling_percentage(0.25, 7)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(500, EstimateLearningRate::Once);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 5e-2 && (p[1] - ty).abs() < 5e-2,
            "recovered {p:?}, expected [{tx}, {ty}]"
        );
    }

    /// A binary mask on a `w × h` grid that is 1 inside `[lo, hi)` on both axes.
    fn box_mask(w: usize, h: usize, lo: usize, hi: usize) -> Image {
        let mut v = vec![0.0f64; w * h];
        for (y, row) in v.chunks_mut(w).enumerate().take(hi).skip(lo) {
            for value in row.iter_mut().take(hi).skip(lo) {
                *value = 1.0;
            }
            let _ = y;
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn fixed_mask_restricts_sampling_to_its_nonzero_voxels() {
        // A 20×20 box mask leaves 400 of the 1600 fixed voxels sampled, and the
        // blob is entirely inside the box, so the shift is still recovered.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mask = box_mask(40, 40, 10, 30);
        let mut counter = ImageRegistrationMethod::new();
        counter.set_metric_fixed_mask(&mask);
        assert_eq!(sample_count_at_identity(&mut counter, &fixed), 400);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_fixed_mask(&mask)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_gradient_descent_estimated(500, EstimateLearningRate::Once);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-2 && (p[1] - ty).abs() < 1e-2,
            "recovered {p:?}, expected [{tx}, {ty}]"
        );
    }

    #[test]
    fn moving_mask_drops_the_samples_that_map_into_its_zero_voxels() {
        // Without a moving mask every one of the 1600 fixed samples maps inside.
        // Masking the moving image to a 20×20 box drops every sample that lands
        // outside it, so the valid-point count falls to the box's area.
        let (fixed, moving, _, _) = shifted_blob_pair();
        let mask = box_mask(40, 40, 10, 30);

        let mut unmasked = ImageRegistrationMethod::new();
        unmasked.set_optimizer_as_gradient_descent(0.0, 1);
        let base = unmasked
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();
        assert_eq!(base.valid_points, 1600);

        let mut masked = ImageRegistrationMethod::new();
        masked
            .set_metric_moving_mask(&mask)
            .set_optimizer_as_gradient_descent(0.0, 1);
        let result = masked
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();
        assert_eq!(result.valid_points, 400);
    }

    #[test]
    fn fixed_mask_size_mismatch_is_rejected() {
        let (fixed, moving, _, _) = shifted_blob_pair();
        let mask = box_mask(30, 30, 5, 25);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_fixed_mask(&mask);
        let err = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::MaskSizeMismatch { which: "fixed", .. }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn moving_mask_size_mismatch_is_rejected() {
        let (fixed, moving, _, _) = shifted_blob_pair();
        let mask = box_mask(30, 30, 5, 25);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_moving_mask(&mask);
        let err = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::MaskSizeMismatch {
                    which: "moving",
                    ..
                }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn per_level_sampling_percentage_length_must_match_the_level_count() {
        let (fixed, moving, _, _) = shifted_blob_pair();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_shrink_factors_per_level(vec![2, 1])
            .set_smoothing_sigmas_per_level(vec![1.0, 0.0])
            .set_metric_sampling_strategy(SamplingStrategy::Regular)
            .set_metric_sampling_percentage_per_level(vec![0.5, 0.25, 0.1], 0);
        let err = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::SamplingPercentageLength {
                    got: 3,
                    expected: 2
                }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn per_level_sampling_percentage_applies_to_the_shrunk_virtual_domain() {
        // Level 0 shrinks 40×40 by 2 to 20×20 (400 voxels) and samples 50% of
        // them at stride 2; level 1 is full resolution (1600 voxels) at 25%,
        // stride 4. The percentage applies to the *shrunk* virtual domain, so
        // the finest level's sample count is 400 — the same number the coarse
        // level would give at 100%, which is why the per-level list matters.
        let (fixed, moving, tx, ty) = shifted_blob_pair();
        let mut counter = ImageRegistrationMethod::new();
        counter
            .set_shrink_factors_per_level(vec![2, 1])
            .set_smoothing_sigmas_per_level(vec![1.0, 0.0])
            .set_metric_sampling_strategy(SamplingStrategy::Regular)
            .set_metric_sampling_percentage_per_level(vec![0.5, 0.25], 0);
        assert_eq!(sample_count_at_identity(&mut counter, &fixed), 400);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_shrink_factors_per_level(vec![2, 1])
            .set_smoothing_sigmas_per_level(vec![1.0, 0.0])
            .set_metric_sampling_strategy(SamplingStrategy::Regular)
            .set_metric_sampling_percentage_per_level(vec![0.5, 0.25], 0)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-4,
                300,
                1e-6,
                EstimateLearningRate::Once,
            );
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();

        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 5e-2 && (p[1] - ty).abs() < 5e-2,
            "recovered {p:?}, expected [{tx}, {ty}]"
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
        use sitk_transform::{BSplineTransform, TransformBase};

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
            bright
                .to_f64_vec()
                .unwrap()
                .iter()
                .map(|v| amp - v)
                .collect(),
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
        use sitk_transform::{BSplineTransform, TransformBase};

        let (w, h, sigma, amp) = (40usize, 40usize, 6.0, 1.0);
        let (cx, cy) = (20.0f64, 20.0f64);
        let (tx, ty) = (2.0f64, -1.5f64);
        let fixed = gaussian(w, h, cx, cy, sigma, amp);
        let bright = gaussian(w, h, cx + tx, cy + ty, sigma, amp);
        let moving = Image::from_vec(
            &[w, h],
            bright
                .to_f64_vec()
                .unwrap()
                .iter()
                .map(|v| amp - v)
                .collect(),
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
        use sitk_transform::{DisplacementFieldTransform, TransformBase};

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
        use sitk_transform::{Euler3DTransform, TransformBase};
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
        use sitk_transform::{TransformBase, VersorRigid3DTransform};
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

    // ---- metrics: Correlation / ANTS NCC / joint-histogram MI / Demons ----

    /// `moving(p) = a·fixed(p − t) + b` — an *affine* intensity map. Mean
    /// squares wants `M ≈ F` and is defeated by it; correlation is invariant to
    /// it by construction (it subtracts the means and divides by the standard
    /// deviations).
    fn affinely_remapped_shifted_blob() -> (Image, Image, f64, f64) {
        let (w, h) = (48usize, 48usize);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 24.0, 24.0, 6.0, 1.0);
        let shifted = gaussian(w, h, 24.0 + tx, 24.0 + ty, 6.0, 1.0);
        let moving = Image::from_vec(
            &[w, h],
            shifted
                .to_f64_vec()
                .unwrap()
                .iter()
                .map(|v| 2.1 * v + 0.4)
                .collect(),
        )
        .unwrap();
        (fixed, moving, tx, ty)
    }

    #[test]
    fn correlation_recovers_a_translation_under_an_affine_intensity_map() {
        let (fixed, moving, tx, ty) = affinely_remapped_shifted_blob();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_correlation()
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-3,
                200,
                1e-7,
                EstimateLearningRate::Once,
            );
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();
        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 0.1 && (p[1] - ty).abs() < 0.1,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}",
            result.metric_value,
            result.iterations
        );
    }

    #[test]
    fn correlation_rejects_a_displacement_field_transform() {
        // itk::CorrelationImageToImageMetricv4's constructor throws for a
        // displacement field; the driver must surface that as an error, not
        // trip the metric's own debug assertion.
        use sitk_transform::DisplacementFieldTransform;

        let fixed = gaussian(16, 16, 8.0, 8.0, 3.0, 1.0);
        let moving = gaussian(16, 16, 9.0, 8.0, 3.0, 1.0);
        let field = DisplacementFieldTransform::from_image_domain(&fixed).unwrap();

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_correlation()
            .set_optimizer_as_gradient_descent(1e-3, 1);
        let Err(err) = reg.execute(&fixed, &moving, field) else {
            panic!("a displacement field must be rejected by the correlation metric");
        };
        assert!(
            matches!(
                err,
                RegistrationError::RequiresGlobalTransform {
                    metric: "Correlation"
                }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn ants_neighborhood_correlation_recovers_a_translation() {
        // The local (windowed) counterpart of the global correlation metric: the
        // same affine intensity map, recovered from per-window normalization.
        let (fixed, moving, tx, ty) = affinely_remapped_shifted_blob();
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_ants_neighborhood_correlation(3)
            .set_optimizer_scales_from_physical_shift()
            .set_optimizer_as_regular_step_gradient_descent_estimated(
                1e-3,
                200,
                1e-7,
                EstimateLearningRate::Once,
            );
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();
        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 0.2 && (p[1] - ty).abs() < 0.2,
            "recovered {p:?}, expected [{tx}, {ty}], metric {}, iters {}",
            result.metric_value,
            result.iterations
        );
    }

    #[test]
    fn ants_neighborhood_radius_wider_than_the_fixed_image_is_rejected() {
        let fixed = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let moving = gaussian(8, 8, 5.0, 4.0, 2.0, 1.0);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_ants_neighborhood_correlation(5)
            .set_optimizer_as_gradient_descent(1e-3, 1);
        let Err(err) = reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
        else {
            panic!("a window of diameter 11 must not be accepted on an 8-wide image");
        };
        assert!(
            matches!(
                err,
                RegistrationError::NeighborhoodRadiusExceedsImage {
                    radius: 5,
                    window: 11,
                    size: 8,
                    axis: 0
                }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn joint_histogram_mi_recovers_a_translation_under_contrast_inversion() {
        // Same multi-modality setup as the Mattes test, driven by the other
        // mutual-information metric: a Gaussian-smoothed hard-binned histogram
        // instead of a cubic B-spline Parzen window.
        let (w, h, sigma, amp) = (48usize, 48usize, 6.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 24.0, 24.0, sigma, amp);
        let bright = gaussian(w, h, 24.0 + tx, 24.0 + ty, sigma, amp);
        let moving = Image::from_vec(
            &[w, h],
            bright
                .to_f64_vec()
                .unwrap()
                .iter()
                .map(|v| amp - v)
                .collect(),
        )
        .unwrap();

        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_joint_histogram_mutual_information(20, 1.5)
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
    fn joint_histogram_mi_smoothing_variance_reaches_the_metric() {
        // The variance is a real knob, not a stored-and-ignored field: a wider
        // joint-PDF smoothing blurs the density and changes the metric value at
        // the same (identity) transform.
        let (fixed, moving, _, _) = shifted_blob_pair();
        let value_at_identity = |variance: f64| {
            let mut reg = ImageRegistrationMethod::new();
            reg.set_metric_as_joint_histogram_mutual_information(20, variance)
                .set_optimizer_as_gradient_descent(0.0, 1);
            reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
                .unwrap()
                .metric_value
        };
        let tight = value_at_identity(0.5);
        let wide = value_at_identity(4.0);
        assert!(
            (tight - wide).abs() > 1e-6,
            "smoothing variance did not change the metric: {tight} vs {wide}"
        );
    }

    #[test]
    fn demons_recovers_a_translation_with_a_displacement_field() {
        use sitk_transform::{DisplacementFieldTransform, TransformBase};

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
        // Every pixel's force is divided by the metric-wide valid-point count
        // (see the `demons` module docs), so an individually Newton-sized step
        // needs a learning rate scaled up by roughly that count. Demons has no
        // line search and no step cap, so an over-large rate (or too many
        // iterations at a large one) oscillates instead of converging: on this
        // fixture `(5.0, 300)` reaches 0.09×baseline, `(20.0, 1000)` diverges
        // back above it.
        reg.set_metric_as_demons(0.001)
            .set_optimizer_as_gradient_descent(5.0, 300);
        let result = reg.execute(&fixed, &moving, field).unwrap();

        // Demons drives each pixel's own displacement; alignment shows up as a
        // mean-squares residual far below the identity baseline.
        let aligned = MeanSquaresMetric::new(&fixed, &moving)
            .unwrap()
            .evaluate(&result.transform, &CpuBackend)
            .value;
        assert!(
            aligned < 0.3 * baseline,
            "mean squares after Demons {aligned} not below 0.3×baseline {baseline}"
        );
        // On the blob's flank the recovered x-displacement approaches the shift.
        let flank = [cx + sigma, cy];
        let mapped = result.transform.transform_point(&flank);
        assert!(
            (mapped[0] - (flank[0] + tx)).abs() < 0.7,
            "flank mapped to {mapped:?}, expected x≈{}",
            flank[0] + tx
        );
    }

    #[test]
    fn demons_rejects_a_global_transform() {
        // itk::DemonsImageToImageMetricv4::Initialize throws unless the moving
        // transform is a displacement field.
        let fixed = gaussian(16, 16, 8.0, 8.0, 3.0, 1.0);
        let moving = gaussian(16, 16, 9.0, 8.0, 3.0, 1.0);
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_demons(0.001)
            .set_optimizer_as_gradient_descent(1e-3, 1);
        let Err(err) = reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
        else {
            panic!("a translation transform must be rejected by the Demons metric");
        };
        assert!(
            matches!(
                err,
                RegistrationError::RequiresLocalSupportTransform { metric: "Demons" }
            ),
            "unexpected error {err:?}"
        );
    }

    // ---- SetOptimizerScalesFrom{Jacobian,IndexShift} ---------------------

    /// A 5×5 fixed image and a moving image shifted by one voxel, with an
    /// affine transform centred at `(1,1)`. On this domain the estimators'
    /// scales are hand-derivable (see `scales.rs`):
    ///
    /// | parameter | PhysicalShift | Jacobian |
    /// |---|---|---|
    /// | the four matrix entries | `max|x−1|² = 9` | `mean (x−1)² = 5` |
    /// | the two translations | `1` | `1` |
    ///
    /// One gradient-descent iteration from the identity moves parameter `k` by
    /// `−lr·g[k]/scale[k]`, so the *ratio* of the two runs' first steps is
    /// `9/5` on the matrix entries and `1` on the translations — independent of
    /// the metric gradient `g`, which never has to be computed by hand.
    fn one_affine_step(scales: impl Fn(&mut ImageRegistrationMethod)) -> Vec<f64> {
        use sitk_transform::AffineTransform;
        let fixed = gaussian(5, 5, 2.0, 2.0, 1.5, 1.0);
        let moving = gaussian(5, 5, 3.0, 2.0, 1.5, 1.0);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 1);
        scales(&mut reg);
        let init =
            AffineTransform::new(2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]);
        let result = reg.execute(&fixed, &moving, init).unwrap();
        // Report the *step*: the identity affine starts at [1,0,0,1,0,0].
        let start = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        result
            .transform
            .parameters()
            .iter()
            .zip(start)
            .map(|(&p, s)| p - s)
            .collect()
    }

    #[test]
    fn jacobian_scales_step_nine_fifths_further_than_physical_shift_scales() {
        let phys = one_affine_step(|r| {
            r.set_optimizer_scales_from_physical_shift();
        });
        let jac = one_affine_step(|r| {
            r.set_optimizer_scales_from_jacobian(DEFAULT_CENTRAL_REGION_RADIUS);
        });

        // Matrix entries: scale 9 vs 5, so the Jacobian step is 9/5 = 1.8× larger.
        for k in 0..4 {
            assert!(
                phys[k].abs() > 1e-12,
                "step[{k}] = {} is too small to form a ratio",
                phys[k]
            );
            let ratio = jac[k] / phys[k];
            assert!(
                (ratio - 9.0 / 5.0).abs() < 1e-9,
                "matrix step ratio[{k}] = {ratio} != 9/5"
            );
        }
        // Translations: scale 1 under both estimators, so the steps coincide.
        for k in 4..6 {
            assert!(
                (jac[k] - phys[k]).abs() < 1e-12,
                "translation step[{k}]: jacobian {} != physical shift {}",
                jac[k],
                phys[k]
            );
        }
    }

    #[test]
    fn index_shift_scales_divide_the_step_by_the_squared_moving_spacing() {
        // Moving image spacing (2, 1): its physical-to-index matrix is
        // diag(½, 1), so a unit translation along x moves the moving-image
        // continuous index by only ½.
        //   PhysicalShift: scale = (δ·1/δ)² = 1   on both axes.
        //   IndexShift:    scale = (δ·½/δ)² = ¼   on x, 1 on y.
        // One gradient-descent step is −lr·g[k]/scale[k], so the index-shift
        // run steps exactly 1/0.25 = 4× further along x and identically along y.
        let step = |anisotropic_scales: bool| -> Vec<f64> {
            let fixed = gaussian(9, 9, 4.0, 4.0, 2.0, 1.0);
            let mut moving = gaussian(9, 9, 5.0, 4.0, 2.0, 1.0);
            moving.set_spacing(&[2.0, 1.0]).unwrap();

            let mut reg = ImageRegistrationMethod::new();
            reg.set_optimizer_as_gradient_descent(1.0, 1);
            if anisotropic_scales {
                reg.set_optimizer_scales_from_index_shift(
                    DEFAULT_CENTRAL_REGION_RADIUS,
                    DEFAULT_SMALL_PARAMETER_VARIATION,
                );
            } else {
                reg.set_optimizer_scales_from_physical_shift();
            }
            reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
                .unwrap()
                .transform
                .parameters()
        };

        let phys = step(false);
        let idx = step(true);
        assert!(phys[0].abs() > 1e-12, "x step {} too small", phys[0]);
        let ratio = idx[0] / phys[0];
        assert!((ratio - 4.0).abs() < 1e-9, "x step ratio {ratio} != 4");
        assert!(
            (idx[1] - phys[1]).abs() < 1e-12,
            "y step: index shift {} != physical shift {}",
            idx[1],
            phys[1]
        );
    }

    #[test]
    fn jacobian_scales_recover_a_translation_through_an_affine_transform() {
        use sitk_transform::AffineTransform;
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_jacobian(DEFAULT_CENTRAL_REGION_RADIUS)
            .set_optimizer_as_gradient_descent_estimated(2000, EstimateLearningRate::Once);
        let init = AffineTransform::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![20.0, 20.0],
        );
        let result = reg.execute(&fixed, &moving, init).unwrap();
        let p = result.transform.parameters();
        assert!(
            (p[4] - tx).abs() < 5e-2 && (p[5] - ty).abs() < 5e-2,
            "recovered translation [{}, {}], expected [{tx}, {ty}]",
            p[4],
            p[5]
        );
    }

    #[test]
    fn index_shift_scales_recover_a_translation() {
        let (w, h, sigma, amp) = (40usize, 40usize, 7.0, 1.0);
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, amp);
        let moving = gaussian(w, h, 20.0 + tx, 20.0 + ty, sigma, amp);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_scales_from_index_shift(
            DEFAULT_CENTRAL_REGION_RADIUS,
            DEFAULT_SMALL_PARAMETER_VARIATION,
        )
        .set_optimizer_as_gradient_descent_estimated(500, EstimateLearningRate::Once);
        let result = reg
            .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
            .unwrap();
        let p = result.transform.parameters();
        assert!(
            (p[0] - tx).abs() < 1e-3 && (p[1] - ty).abs() < 1e-3,
            "recovered {p:?}, expected [{tx}, {ty}]"
        );
    }

    #[test]
    fn the_central_region_radius_does_not_change_any_estimator_for_a_linear_transform() {
        // Upstream reaches CentralRegionSampling only for a local-support
        // transform, so the radius cannot move a linear transform's scales.
        // Pinned end-to-end through the setters (ledger §2.115).
        let jac0 = one_affine_step(|r| {
            r.set_optimizer_scales_from_jacobian(0);
        });
        let jac9 = one_affine_step(|r| {
            r.set_optimizer_scales_from_jacobian(9);
        });
        assert_eq!(jac0, jac9);

        let idx0 = one_affine_step(|r| {
            r.set_optimizer_scales_from_index_shift(0, DEFAULT_SMALL_PARAMETER_VARIATION);
        });
        let idx9 = one_affine_step(|r| {
            r.set_optimizer_scales_from_index_shift(9, DEFAULT_SMALL_PARAMETER_VARIATION);
        });
        assert_eq!(idx0, idx9);
    }

    #[test]
    fn a_smaller_small_parameter_variation_leaves_the_index_shift_scales_unchanged() {
        // scaleᵢ = (maxShiftᵢ/δ)² and maxShiftᵢ = δ·‖Jᵢ‖ exactly for the linear
        // transforms this crate registers, so δ cancels analytically. It is
        // carried because upstream's shift is a re-transform rather than a
        // Jacobian product, where δ would not cancel for a nonlinear transform.
        //
        // The cancellation is exact in ℝ but not in binary floating point:
        // squaring δ = 1e-6 and dividing by δ² rounds differently than at
        // δ = 0.01, so the two runs agree to a relative 1e-9, not bit-for-bit.
        let coarse = one_affine_step(|r| {
            r.set_optimizer_scales_from_index_shift(DEFAULT_CENTRAL_REGION_RADIUS, 0.01);
        });
        let fine = one_affine_step(|r| {
            r.set_optimizer_scales_from_index_shift(DEFAULT_CENTRAL_REGION_RADIUS, 1e-6);
        });
        for (k, (&c, &f)) in coarse.iter().zip(fine.iter()).enumerate() {
            assert!(
                (c - f).abs() <= 1e-9 * c.abs().max(f.abs()),
                "step[{k}]: δ=0.01 gives {c}, δ=1e-6 gives {f}"
            );
        }
    }

    // ---- Set/GetOptimizerWeights -----------------------------------------

    #[test]
    fn optimizer_weights_round_trip_and_default_to_empty() {
        let mut reg = ImageRegistrationMethod::new();
        assert!(reg.optimizer_weights().is_empty());
        reg.set_optimizer_weights(vec![1.0, 0.0]);
        assert_eq!(reg.optimizer_weights(), &[1.0, 0.0]);
    }

    #[test]
    fn weights_are_identity_within_one_ten_thousandth_inclusive() {
        // ITK: `if (Absolute(1 - w) > tolerance) { identity = false; }` with
        // tolerance 1e-4 — so a difference of exactly 1e-4 is still identity.
        assert!(weights_are_identity(&[]));
        assert!(weights_are_identity(&[1.0, 1.0]));
        assert!(weights_are_identity(&[1.0 + 1e-4, 1.0 - 1e-4]));
        assert!(!weights_are_identity(&[1.0, 1.0 + 1.01e-4]));
        assert!(!weights_are_identity(&[0.0, 1.0]));
    }

    #[test]
    fn a_weight_within_the_identity_tolerance_is_discarded_not_applied() {
        // 1.00005 is within 1e-4 of 1, so ITK never multiplies it in and the
        // run is bit-identical to an unweighted one. 1.001 is outside, so it is
        // applied and the trajectory changes.
        let run = |weights: Vec<f64>| -> Vec<f64> {
            let fixed = gaussian(40, 40, 20.0, 20.0, 7.0, 1.0);
            let moving = gaussian(40, 40, 23.0, 18.0, 7.0, 1.0);
            let mut reg = ImageRegistrationMethod::new();
            reg.set_optimizer_as_gradient_descent(100.0, 5)
                .set_optimizer_weights(weights);
            reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
                .unwrap()
                .transform
                .parameters()
        };

        let unweighted = run(Vec::new());
        assert_eq!(run(vec![1.00005, 1.00005]), unweighted);
        assert_ne!(run(vec![1.001, 1.001]), unweighted);
    }

    #[test]
    fn a_zero_weight_freezes_its_parameter() {
        // Upstream's documented use of weights: "may be used to easily mask out
        // a particular parameter during optimization to hold it constant"
        // (itkObjectToObjectOptimizerBase.h:97-103). The gradient is multiplied
        // by 0, so y never moves off its initial value while x still converges.
        let (tx, ty) = (3.0f64, -2.0f64);
        let fixed = gaussian(40, 40, 20.0, 20.0, 7.0, 1.0);
        let moving = gaussian(40, 40, 20.0 + tx, 20.0 + ty, 7.0, 1.0);

        let run = |weights: Vec<f64>| {
            let mut reg = ImageRegistrationMethod::new();
            reg.set_optimizer_as_gradient_descent(100.0, 300)
                .set_optimizer_weights(weights);
            reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
                .unwrap()
                .transform
                .parameters()
        };

        // Unweighted, both parameters converge to the true translation.
        let free = run(Vec::new());
        assert!(
            (free[0] - tx).abs() < 1e-3 && (free[1] - ty).abs() < 1e-3,
            "unweighted run recovered {free:?}, expected [{tx}, {ty}]"
        );

        // Weight 0 on y: it never leaves its initial value, exactly. x still
        // converges, though not to `tx` — it converges to the best x under the
        // constraint y = 0, which the y-mismatch pulls slightly off 3.
        let frozen = run(vec![1.0, 0.0]);
        assert_eq!(
            frozen[1], 0.0,
            "the zero-weighted parameter moved to {}",
            frozen[1]
        );
        assert!(
            (frozen[0] - tx).abs() < 5e-2,
            "the unfrozen parameter recovered {} not ≈{tx}",
            frozen[0]
        );
    }

    #[test]
    fn weights_of_the_wrong_length_are_rejected_with_itks_message() {
        use sitk_transform::AffineTransform;
        let fixed = gaussian(20, 20, 10.0, 10.0, 4.0, 1.0);
        let moving = gaussian(20, 20, 11.0, 10.0, 4.0, 1.0);

        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 1)
            .set_optimizer_weights(vec![1.0; 5]);
        let init = AffineTransform::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        );
        let err = reg.execute(&fixed, &moving, init).unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::OptimizerWeightsLength {
                    got: 5,
                    expected: 6
                }
            ),
            "unexpected error {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "Size of weights (5) must equal number of local parameters (6)."
        );
    }

    #[test]
    fn displacement_field_weights_are_sized_by_the_local_parameter_count() {
        // ITK validates against metric->GetNumberOfLocalParameters(), which for
        // a displacement field is just `dim` — not its (huge) parameter count.
        // The short array is then tiled across every pixel's parameter block.
        use sitk_transform::DisplacementFieldTransform;
        let fixed = gaussian(12, 12, 6.0, 6.0, 3.0, 1.0);
        let moving = gaussian(12, 12, 7.0, 6.0, 3.0, 1.0);
        let field = DisplacementFieldTransform::from_image_domain(&fixed).unwrap();
        let nparams = field.number_of_parameters();
        assert_eq!(nparams, 12 * 12 * 2);

        // Length `dim` is accepted, and the zero y-weight freezes the y
        // component of *every* pixel's displacement.
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 5)
            .set_optimizer_weights(vec![1.0, 0.0]);
        let result = reg
            .execute(
                &fixed,
                &moving,
                DisplacementFieldTransform::from_image_domain(&fixed).unwrap(),
            )
            .unwrap();
        let p = result.transform.parameters();
        for pixel in 0..nparams / 2 {
            assert_eq!(p[2 * pixel + 1], 0.0, "pixel {pixel} moved along y");
        }
        assert!(
            p.iter().step_by(2).any(|&v| v.abs() > 1e-6),
            "no pixel moved along x"
        );

        // Length `nparams` — the whole-transform count — is rejected.
        let mut reg = ImageRegistrationMethod::new();
        reg.set_optimizer_as_gradient_descent(1.0, 1)
            .set_optimizer_weights(vec![1.0; nparams]);
        let err = reg
            .execute(
                &fixed,
                &moving,
                DisplacementFieldTransform::from_image_domain(&fixed).unwrap(),
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::OptimizerWeightsLength {
                    got: 288,
                    expected: 2
                }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn lbfgsb_validates_the_weights_length_and_then_ignores_the_weights() {
        // ObjectToObjectOptimizerBase::StartOptimization validates m_Weights for
        // every v4 optimizer, but only the gradient-descent family owns
        // ModifyGradientByScalesOverSubRange. So L-BFGS-B rejects a mis-sized
        // array and silently discards a well-sized one (ledger §2.117).
        let fixed = gaussian(40, 40, 20.0, 20.0, 7.0, 1.0);
        let moving = gaussian(40, 40, 23.0, 18.0, 7.0, 1.0);
        let run = |weights: Vec<f64>| {
            let mut reg = ImageRegistrationMethod::new();
            reg.set_optimizer_as_lbfgsb(1e-5, 100, 5, 2000, 1e7, -1e3, 1e3)
                .set_optimizer_weights(weights);
            reg.execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
        };

        // A zero weight would freeze y under gradient descent; L-BFGS-B ignores
        // it and still recovers the full translation.
        let weighted = run(vec![1.0, 0.0]).unwrap().transform.parameters();
        let unweighted = run(Vec::new()).unwrap().transform.parameters();
        assert_eq!(weighted, unweighted);
        assert!((weighted[1] + 2.0).abs() < 1e-2, "y = {}", weighted[1]);

        // The length is still validated.
        let err = run(vec![1.0, 1.0, 1.0]).unwrap_err();
        assert!(
            matches!(
                err,
                RegistrationError::OptimizerWeightsLength {
                    got: 3,
                    expected: 2
                }
            ),
            "unexpected error {err:?}"
        );
    }
}
