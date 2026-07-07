//! Spatial transforms and resampling for sitk-rs.
//!
//! Phase 0 covers [`TranslationTransform`], [`AffineTransform`], and
//! [`ResampleImageFilter`] with nearest-neighbour and linear interpolation —
//! enough to close the read → transform → resample → write vertical slice. The
//! remaining transform classes and interpolators follow in later phases.

pub mod error;
pub mod interpolator;
pub mod resample;
pub mod transform;

pub use error::{Result, TransformError};
pub use resample::{Interpolator, ResampleImageFilter};
pub use transform::{AffineTransform, ParametricTransform, Transform, TranslationTransform};
