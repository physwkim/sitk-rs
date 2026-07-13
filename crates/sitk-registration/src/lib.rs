//! Image registration for sitk-rs: metrics, optimizers, and the
//! [`ImageRegistrationMethod`] driver.
//!
//! The core is the smallest end-to-end registration that actually aligns two
//! images: the **mean-squares** metric ([`MeanSquaresMetric`]) sampled over the
//! full fixed grid, **linear** interpolation of the moving image, a
//! **gradient-descent** optimizer ([`GradientDescentOptimizer`] with a fixed or
//! estimated rate, or [`RegularStepGradientDescentOptimizer`], which halves its
//! step on each overshoot), and [`TranslationTransform`]/[`AffineTransform`] as
//! the moving parameters, mirroring `itk::ImageRegistrationMethodv4`. It runs at
//! a single resolution by default and over a **multi-resolution pyramid** when a
//! shrink/smoothing schedule is configured (see
//! [Multi-resolution](#multi-resolution) below).
//!
//! The metric is selectable: mean squares ([`MeanSquaresMetric`]) by default, or
//! **Mattes mutual information** ([`MattesMutualInformationMetric`],
//! `itk::MattesMutualInformationImageToImageMetricv4`) for **multi-modality**
//! registration — images related by an arbitrary invertible intensity map, where
//! mean squares fails — via
//! [`set_metric_as_mattes_mutual_information`](ImageRegistrationMethod::set_metric_as_mattes_mutual_information).
//!
//! Optimizer scales and the learning rate are **estimated automatically** from
//! physical shift ([`ScalesEstimator`], ITK's
//! `RegistrationParameterScalesFromPhysicalShift` +
//! `GradientDescentOptimizerv4` learning-rate estimation), so no hand-tuning is
//! required. The Jacobian and index-shift estimators are available too, via
//! [`ImageRegistrationMethod::set_optimizer_scales_from_jacobian`] and
//! [`ImageRegistrationMethod::set_optimizer_scales_from_index_shift`]:
//!
//! ```
//! use sitk_core::Image;
//! use sitk_registration::{EstimateLearningRate, ImageRegistrationMethod};
//! use sitk_transform::{ParametricTransform, TranslationTransform};
//!
//! # fn blob(cx: f64) -> Image {
//! #     let mut v = vec![0.0f64; 32 * 32];
//! #     for y in 0..32 { for x in 0..32 {
//! #         let (dx, dy) = (x as f64 - cx, y as f64 - 16.0);
//! #         v[y * 32 + x] = (-(dx*dx + dy*dy) / 50.0).exp();
//! #     }}
//! #     Image::from_vec(&[32, 32], v).unwrap()
//! # }
//! let fixed = blob(16.0);
//! let moving = blob(18.0); // shifted by +2 in x
//!
//! let mut reg = ImageRegistrationMethod::new();
//! reg.set_optimizer_scales_from_physical_shift()
//!     .set_optimizer_as_gradient_descent_estimated(300, EstimateLearningRate::Once);
//! let result = reg
//!     .execute(&fixed, &moving, TranslationTransform::new(vec![0.0, 0.0]))
//!     .unwrap();
//!
//! assert!((result.transform.parameters()[0] - 2.0).abs() < 1e-3);
//! ```
//!
//! ## Multi-resolution
//!
//! Configuring a shrink/smoothing schedule runs a coarse-to-fine pyramid
//! (`itk::ImageRegistrationMethodv4`'s per-level scheme):
//! [`set_shrink_factors_per_level`] and [`set_smoothing_sigmas_per_level`] take
//! one entry per level, coarsest first. Per level the fixed image is
//! Gaussian-smoothed and placed on the shrunk **virtual-domain** grid (the fixed
//! values are resampled onto that grid by linear interpolation, as ITK's metric
//! interpolates the smoothed fixed at each virtual point), the moving image is
//! Gaussian-smoothed, and the transform optimized at the coarse level
//! initializes the next finer one. Smoothing widens the metric's capture range,
//! so the pyramid aligns from initial offsets a single resolution level cannot.
//!
//! ```
//! use sitk_registration::{EstimateLearningRate, ImageRegistrationMethod};
//!
//! let mut reg = ImageRegistrationMethod::new();
//! reg.set_optimizer_scales_from_physical_shift()
//!     .set_optimizer_as_gradient_descent_estimated(150, EstimateLearningRate::Once)
//!     .set_shrink_factors_per_level(vec![4, 2, 1])
//!     .set_smoothing_sigmas_per_level(vec![2.0, 1.0, 0.0]);
//! ```
//!
//! For a pyramid, [`set_optimizer_as_regular_step_gradient_descent_estimated`]
//! is usually the better optimizer: a level that restarts from an
//! already-registered coarser transform has a near-zero gradient, which a fixed
//! estimate-once rate cannot descend precisely, whereas the regular step halves
//! its length on each overshoot and stops cleanly on its gradient-magnitude
//! tolerance — reaching far higher finest-level precision at the same iteration
//! budget.
//!
//! The pyramid smooths with the bit-exact recursive Gaussian
//! ([`filters::recursive_gaussian`], a Deriche/Farnebäck IIR porting
//! `itk::RecursiveGaussianImageFilter`), matching ITK's
//! `SmoothingRecursiveGaussianImageFilter`; the result-faithful separable FIR
//! ([`filters::smooth_gaussian`]) remains available behind the same seam. Both
//! images are smoothed at full resolution, so the recursive filter's
//! ≥4-pixels-per-smoothed-axis requirement bites only on a pathologically small
//! input (a `sigma == 0` level is a no-op). The shrink is a bit-exact
//! [`filters::shrink`] (`itk::ShrinkImageFilter`).
//!
//! [`set_shrink_factors_per_level`]:
//! ImageRegistrationMethod::set_shrink_factors_per_level
//! [`set_smoothing_sigmas_per_level`]:
//! ImageRegistrationMethod::set_smoothing_sigmas_per_level
//! [`set_optimizer_as_regular_step_gradient_descent_estimated`]:
//! ImageRegistrationMethod::set_optimizer_as_regular_step_gradient_descent_estimated
//! [`filters::smooth_gaussian`]: sitk_filters::smooth_gaussian
//! [`filters::recursive_gaussian`]: sitk_filters::recursive_gaussian()
//! [`filters::shrink`]: sitk_filters::shrink()
//!
//! ## GPU seam
//!
//! The metric's per-sample reduction is isolated behind [`MetricBackend`]; the
//! shipped [`CpuBackend`] runs on the host. A backend implementing the same trait
//! drops in via [`ImageRegistrationMethod::set_metric_backend`] — no change to
//! the metric or the registration loop.
//!
//! `CudaMetricBackend` (behind the `cuda` feature, **default off**, so it is
//! absent from these docs unless the feature is on) is one such backend: it keeps
//! the fixed samples and the moving volume resident on the device across the whole
//! optimizer run, so each iteration ships only the transform's twelve affine
//! coefficients up and a fixed-size moment vector back. It is a strict accelerator
//! — it falls back to [`CpuBackend`] on every condition that is not a GPU success
//! (no driver, no device, a non-affine transform), so it cannot turn a working
//! registration into a failing one.
//!
//! [`AffineTransform`]: sitk_transform::AffineTransform
//! [`TranslationTransform`]: sitk_transform::TranslationTransform
//! [`ScalesEstimator`]: crate::ScalesEstimator

pub mod ants_correlation;
pub mod bspline_initializer;
pub mod centered_versor;
pub mod convergence;
pub mod correlation;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod demons;
#[cfg(feature = "cuda")]
pub mod device;
mod eigen;
pub mod error;
pub mod gradient_free;
pub mod initializer;
pub mod joint_histogram;
pub mod landmark;
pub mod lbfgs2;
pub mod lbfgsb;
pub mod mattes;
pub mod method;
pub mod metric;
pub mod optimizer;
pub mod scales;

pub use ants_correlation::AntsNeighborhoodCorrelationMetric;
pub use bspline_initializer::BSplineTransformInitializer;
pub use centered_versor::CenteredVersorTransformInitializer;
pub use convergence::WindowConvergenceMonitor;
pub use correlation::CorrelationMetric;
#[cfg(feature = "cuda")]
pub use cuda::CudaMetricBackend;
pub use demons::DemonsMetric;
#[cfg(feature = "cuda")]
pub use device::{DeviceMeanSquaresMetric, DeviceMetricError};
pub use error::{RegistrationError, Result};
pub use gradient_free::{
    AmoebaOptimizer, ExhaustiveOptimizer, OnePlusOneEvolutionaryOptimizer, PowellOptimizer,
};
pub use initializer::{CenteredTransformInitializer, OperationMode};
pub use joint_histogram::JointHistogramMutualInformationMetric;
pub use landmark::LandmarkBasedTransformInitializer;
pub use lbfgs2::{LBFGS2Optimizer, LineSearchMethod};
pub use lbfgsb::LBFGSBOptimizer;
pub use mattes::MattesMutualInformationMetric;
pub use method::{EstimateLearningRate, ImageRegistrationMethod, RegistrationResult};
pub use metric::{CpuBackend, MeanSquaresMetric, MetricBackend, MetricValue, SamplingStrategy};
pub use optimizer::{
    ConjugateGradientLineSearchOptimizer, GradientDescentLineSearchOptimizer,
    GradientDescentOptimizer, Objective, OptimizerResult, RegularStepGradientDescentOptimizer,
    StopReason,
};
pub use scales::{
    DEFAULT_CENTRAL_REGION_RADIUS, DEFAULT_SMALL_PARAMETER_VARIATION, ScalesEstimator,
    ScalesEstimatorKind, VirtualGrid,
};
