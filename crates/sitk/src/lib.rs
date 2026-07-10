//! # sitk
//!
//! A pure-Rust port of [SimpleITK](https://simpleitk.org/). This umbrella crate
//! re-exports the workspace pieces under one namespace so users depend on a
//! single crate, mirroring SimpleITK's single-module surface.
//!
//! The workspace now covers most of SimpleITK's `BasicFilters` surface (over
//! 250 filters — arithmetic, statistics, segmentation, registration-adjacent
//! smoothing, morphology, the [`LabelMap`] object model and its filter
//! family), [`core`] complex and vector pixel types, the 15 concrete
//! [`transform`] classes plus [`transform::ResampleImageFilter`] /
//! [`transform::WarpImageFilter`], and a coarse-to-fine [`registration`]
//! pipeline (`ImageRegistrationMethod`-equivalent). [`io`] currently supports
//! only uncompressed MetaImage (`.mha`/`.mhd`); the remaining ITK ImageIO
//! formats (NIfTI, DICOM, PNG, ...) and the `Transform` file formats
//! (`.tfm`/`.txt`/`.h5`) are not yet ported — see [`io`]'s and
//! [`transform`]'s module docs, and `doc/upstream-findings.md` §6, for the
//! exact remaining gaps. In particular there is **no `ReadTransform` /
//! `WriteTransform`** yet (SimpleITK's erased `Transform` value type is not
//! ported — `doc/upstream-findings.md` §5.10 — so there is nothing for a
//! transform reader to construct), and consequently **no top-level
//! `Resample`/`Warp` free functions** either, since SimpleITK's own
//! `sitk.Resample(image, transform, ...)` convenience overloads exist
//! primarily to take a transform loaded from disk; the class-based
//! [`transform::ResampleImageFilter`] / [`transform::WarpImageFilter`] cover
//! the same functionality for an in-memory transform.
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
// directly (`sitk.Image`, `sitk.AffineTransform`, `sitk.Cast`, ...).
pub use sitk_core::{
    Image, LabelMap, LabelObject, LabelObjectLine, MAX_DIM, PixelBuffer, PixelId, Scalar,
};
pub use sitk_transform::{
    AffineTransform, BSplineTransform, CenteredTransform, ComposeScaleSkewVersor3DTransform,
    CompositeTransform, DisplacementFieldTransform, Euler2DTransform, Euler3DTransform,
    ParametricTransform, ScaleLogarithmicTransform, ScaleSkewVersor3DTransform, ScaleTransform,
    ScaleVersor3DTransform, Similarity2DTransform, Similarity3DTransform, TransformBase,
    TranslationTransform, VersorRigid3DTransform, VersorTransform,
};
