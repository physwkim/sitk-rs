//! # sitk
//!
//! A pure-Rust port of [SimpleITK](https://simpleitk.org/). This umbrella crate
//! re-exports the workspace pieces under one namespace so users depend on a
//! single crate, mirroring SimpleITK's single-module surface.
//!
//! Phase 0 (this release) is a thin vertical slice: the [`Image`] core model,
//! MetaImage IO, a handful of pixel-wise / statistical [`filters`], and affine
//! [`transform`]-driven [resampling][transform::ResampleImageFilter]. It exists
//! to fix the architecture — runtime pixel dispatch, physical-space geometry,
//! and the read→filter→resample→write pipeline — that the remaining SimpleITK
//! surface will be built on.
//!
//! ```no_run
//! use sitk::{filters, io, transform::{AffineTransform, ResampleImageFilter}};
//!
//! let image = io::read_image("input.mha")?;
//! let smoothed = filters::rescale_intensity(&image, 0.0, 255.0)?;
//! let t = AffineTransform::identity(image.dimension());
//! let resampled = ResampleImageFilter::new()
//!     .set_reference_image(&smoothed)
//!     .execute(&smoothed, &t)?;
//! io::write_image(&resampled, "output.mha")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Core image model and pixel dispatch.
pub use sitk_core as core;
/// Image filters (procedural interface).
pub use sitk_filters as filters;
/// Image file IO.
pub use sitk_io as io;
/// Image registration (metrics, optimizers, `ImageRegistrationMethod`).
pub use sitk_registration as registration;
/// Spatial transforms and resampling.
pub use sitk_transform as transform;

// The most-used types promoted to the crate root, as SimpleITK exposes them
// directly (`sitk.Image`, `sitk.Cast`, ...).
pub use sitk_core::{Image, PixelBuffer, PixelId, Scalar};
