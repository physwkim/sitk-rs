//! Optional CUDA backend for sitk-rs.
//!
//! # Scope: the registration metric, and nothing else
//!
//! This crate implements **one** thing — the per-sample reduction behind
//! `sitk-registration`'s mean-squares metric ([`ResidentMetric`]). It is
//! deliberately not a GPU version of the filter API, and the filter API does
//! not dispatch to it.
//!
//! That boundary was drawn by measurement, not taste. `rescale_intensity` was
//! ported here first, precisely because a pure per-pixel op is the easiest kind
//! of kernel to get right — and it lost. Even with the host output allocation
//! driven to literally zero (a caller-owned destination, no page faults left to
//! pay), it measured **1.4× slower than the CPU at 256³ and 2.2× slower at
//! 512³**. A per-pixel op crosses PCIe twice to do one pass of arithmetic the
//! CPU does at memory bandwidth without moving a byte, and the ratio gets
//! *worse* with volume size, so there is no crossover to wait for. The kernel,
//! its `_into` form and the filter's dispatch branch were all removed;
//! `doc/bench-results.md` keeps the measurement that justifies their absence.
//!
//! The registration metric is the opposite shape, and that is why it stays: the
//! fixed samples and the moving volume are uploaded **once** and then evaluated
//! hundreds of times as the optimizer walks, so the transfer amortizes to
//! nothing and what is left is arithmetic — where the device wins by 67–201×
//! per iteration over 96 CPU cores.
//!
//! The rule this leaves behind: **a GPU op earns its place only if the data it
//! moves is reused.** One-shot per-pixel work belongs on the CPU.
//!
//! # No hard dependency, ever
//!
//! [`backend`] returns [`CudaError::NoDevice`] rather than panicking when there
//! is no driver, no device, or no NVRTC; `sitk-registration`'s CUDA backend
//! turns any such failure into a silent fall back to its CPU path. A machine
//! with no GPU runs the same code and gets the same answers.
//!
//! # Feature gate
//!
//! Everything below is behind the `cuda` feature, **default off**. With the
//! feature off this crate is an empty lib with no dependencies.

#![cfg(feature = "cuda")]

mod backend;
mod buffer;
mod error;
mod ops;

pub use backend::{Backend, backend};
pub use buffer::DeviceBuffer;
pub use error::CudaError;
pub use ops::mean_squares::{DIM, FixedPoints, Moments, MovingGeometry, ResidentMetric};
