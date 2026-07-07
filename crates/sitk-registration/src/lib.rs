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
//! Optimizer scales and the learning rate are **estimated automatically** from
//! physical shift ([`PhysicalShiftScales`], ITK's
//! `RegistrationParameterScalesFromPhysicalShift` +
//! `GradientDescentOptimizerv4` learning-rate estimation), so no hand-tuning is
//! required:
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
//! The Gaussian is a result-faithful separable FIR
//! ([`filters::smooth_gaussian`]); the bit-exact recursive Gaussian
//! ([`filters::recursive_gaussian`], a Deriche/Farnebäck IIR porting
//! `itk::RecursiveGaussianImageFilter`) shares its signature and can replace it
//! at this seam without touching callers, though the pyramid has not yet been
//! switched to it. The shrink is a bit-exact [`filters::shrink`]
//! (`itk::ShrinkImageFilter`).
//!
//! [`set_shrink_factors_per_level`]:
//! ImageRegistrationMethod::set_shrink_factors_per_level
//! [`set_smoothing_sigmas_per_level`]:
//! ImageRegistrationMethod::set_smoothing_sigmas_per_level
//! [`set_optimizer_as_regular_step_gradient_descent_estimated`]:
//! ImageRegistrationMethod::set_optimizer_as_regular_step_gradient_descent_estimated
//! [`filters::smooth_gaussian`]: sitk_filters::smooth_gaussian
//! [`filters::recursive_gaussian`]: sitk_filters::recursive_gaussian()
//! [`filters::shrink`]: sitk_filters::shrink
//!
//! ## GPU seam
//!
//! The metric's per-sample reduction is isolated behind [`MetricBackend`]; the
//! shipped [`CpuBackend`] runs on the host. A CUDA (`cudarc`) or portable
//! `wgpu`/Metal backend implements the same trait and drops in via
//! [`ImageRegistrationMethod::set_metric_backend`] — no change to the metric or
//! the registration loop. (This crate builds and is tested CPU-only; a GPU
//! backend requires GPU hardware, absent on the development machine.)
//!
//! [`AffineTransform`]: sitk_transform::AffineTransform
//! [`TranslationTransform`]: sitk_transform::TranslationTransform
//! [`PhysicalShiftScales`]: crate::PhysicalShiftScales

pub mod convergence;
pub mod error;
pub mod method;
pub mod metric;
pub mod optimizer;
pub mod scales;

pub use convergence::WindowConvergenceMonitor;
pub use error::{RegistrationError, Result};
pub use method::{EstimateLearningRate, ImageRegistrationMethod, RegistrationResult};
pub use metric::{CpuBackend, MeanSquaresMetric, MetricBackend, MetricValue};
pub use optimizer::{
    GradientDescentOptimizer, OptimizerResult, RegularStepGradientDescentOptimizer, StopReason,
};
pub use scales::PhysicalShiftScales;
