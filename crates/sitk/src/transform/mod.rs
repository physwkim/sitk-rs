//! Spatial transforms and resampling for sitk-rs.
//!
//! Phase 0 covers [`TranslationTransform`], [`AffineTransform`], and
//! [`crate::transform::ResampleImageFilter`] with nearest-neighbour and linear interpolation —
//! enough to close the read → transform → resample → write vertical slice. The
//! remaining transform classes and interpolators follow in later phases.

pub mod bspline;
pub mod composite;
pub mod displacement;
pub mod erased;
pub mod error;
pub mod interpolator;
pub mod matrix_offset;
pub mod parametric;
pub mod resample;
pub mod transform_geometry;
pub mod transform_to_displacement_field;
pub mod warp;

pub use bspline::BSplineTransform;
pub use composite::CompositeTransform;
pub use displacement::DisplacementFieldTransform;
pub use erased::{Transform, TransformKind};
pub use error::{Result, TransformError};
pub use parametric::{
    AffineTransform, CenteredTransform, ComposeScaleSkewVersor3DTransform, Euler2DTransform,
    Euler3DTransform, ParametricTransform, ScaleLogarithmicTransform, ScaleSkewVersor3DTransform,
    ScaleTransform, ScaleVersor3DTransform, Similarity2DTransform, Similarity3DTransform,
    TransformBase, TranslationTransform, VersorRigid3DTransform, VersorTransform,
};
pub use resample::{Interpolator, ResampleImageFilter};
pub use transform_geometry::transform_geometry;
pub use transform_to_displacement_field::TransformToDisplacementFieldFilter;
pub use warp::WarpImageFilter;
