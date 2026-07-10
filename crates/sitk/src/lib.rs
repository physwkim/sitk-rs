//! # sitk
//!
//! A pure-Rust port of [SimpleITK](https://simpleitk.org/). This umbrella crate
//! re-exports the workspace pieces under one namespace so users depend on a
//! single crate, mirroring SimpleITK's single-module surface.
//!
//! The workspace now covers most of SimpleITK's `BasicFilters` surface (over
//! 250 filters — arithmetic, statistics, segmentation, registration-adjacent
//! smoothing, morphology, the [`LabelMap`] object model and its filter
//! family), [`core`] complex and vector pixel types, the erased [`Transform`]
//! value type and its 16 concrete [`transform`] classes, plus
//! [`transform::ResampleImageFilter`] / [`transform::WarpImageFilter`], and a
//! coarse-to-fine [`registration`] pipeline (`ImageRegistrationMethod`-
//! equivalent). [`io`] reads and writes uncompressed MetaImage
//! (`.mha`/`.mhd`), raw-encoding NRRD (`.nrrd`/`.nhdr`), and uncompressed
//! NIfTI-1 (`.nii`, `.hdr`/`.img`); the remaining ITK ImageIO formats (DICOM,
//! PNG, ...) are not yet ported — see [`io`]'s and [`transform`]'s module
//! docs, and `doc/upstream-findings.md` §6, for the exact remaining gaps.
//! [`read_transform`] / [`write_transform`] round-trip the Insight legacy
//! transform format (`.tfm`/`.txt`; `.h5`/`.mat`/`.xfm` are not yet ported),
//! and the top-level [`resample`] free function mirrors SimpleITK's
//! `sitk.Resample(image, transform, ...)` single-image overload for a
//! [`Transform`] loaded from disk or built in memory; there is no top-level
//! `Warp` free function yet — [`transform::WarpImageFilter`] is class-based
//! only.
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
/// Read a transform from an Insight legacy transform file (`.tfm`/`.txt`) —
/// `itk::simple::ReadTransform`.
pub use sitk_io::read_transform;
/// Write a transform to an Insight legacy transform file (`.tfm`/`.txt`) —
/// `itk::simple::WriteTransform`.
pub use sitk_io::write_transform;
pub use sitk_transform::{
    AffineTransform, BSplineTransform, CenteredTransform, ComposeScaleSkewVersor3DTransform,
    CompositeTransform, DisplacementFieldTransform, Euler2DTransform, Euler3DTransform,
    ParametricTransform, ScaleLogarithmicTransform, ScaleSkewVersor3DTransform, ScaleTransform,
    ScaleVersor3DTransform, Similarity2DTransform, Similarity3DTransform, Transform, TransformBase,
    TransformKind, TranslationTransform, VersorRigid3DTransform, VersorTransform,
};

/// Resample `image` through `transform`, defaulting to `image`'s own output
/// grid — `itk::simple::Resample(image1, transform, interpolator,
/// defaultPixelValue, outputPixelType, useNearestNeighborExtrapolator)`'s
/// single-image overload (`sitkAdditionalProcedures.cxx:33-49`), with every
/// optional parameter left at its SimpleITK default (linear interpolation,
/// default pixel value `0`, output type matching the input, no
/// nearest-neighbor extrapolator). For a reference image, explicit output
/// geometry, or non-default settings, use [`transform::ResampleImageFilter`]
/// directly — it already accepts the erased [`Transform`] since `Transform`
/// implements [`transform::TransformBase`].
pub fn resample(image: &Image, transform: &Transform) -> transform::Result<Image> {
    transform::ResampleImageFilter::new().execute(image, transform)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives read (transform) → resample → write purely through `sitk::`
    /// paths: a translation is written and read back as an erased
    /// [`Transform`], [`resample`] applies it to an image, and the result is
    /// written and re-read through [`io`].
    #[test]
    fn transform_io_and_resample_round_trip_through_the_facade() {
        let img = Image::from_vec(&[4, 1], vec![0.0f32, 1.0, 2.0, 3.0]).unwrap();

        let mut transform_path = std::env::temp_dir();
        transform_path.push(format!("sitk_facade_test_{}_t.tfm", std::process::id()));
        let written: Transform = TranslationTransform::new(vec![1.0, 0.0]).into();
        write_transform(&written, &transform_path).unwrap();

        let read_back = read_transform(&transform_path).unwrap();
        std::fs::remove_file(&transform_path).ok();
        assert_eq!(read_back.kind(), TransformKind::Translation);

        let resampled = resample(&img, &read_back).unwrap();

        let mut image_path = std::env::temp_dir();
        image_path.push(format!("sitk_facade_test_{}_out.mha", std::process::id()));
        io::write_image(&resampled, &image_path).unwrap();
        let reread = io::read_image(&image_path).unwrap();
        std::fs::remove_file(&image_path).ok();

        assert_eq!(reread.scalar_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 0.0]);
    }
}
