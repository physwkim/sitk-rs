//! Filters that cross the scalar/vector pixel boundary.
//!
//! SimpleITK groups these under `ImageCompose` and `ImageIntensity`, but they
//! share the property that separates them from every other filter in this
//! crate: they are the only ones whose input or output is a vector image. Every
//! other filter is scalar-only, structurally — see the `vector_guard` tests in
//! [`crate`].
//!
//! Correspondingly none of them route through `to_f64_vec`/`image_from_f64`
//! (the scalar seam, which refuses vector images); they read the interleaved
//! buffer through [`Image::component_slice`] and the component primitives
//! [`Image::from_component_images`] / [`Image::extract_component`].

use sitk_core::Image;

use crate::Result;

/// `ComposeImageFilter` (`itkComposeImageFilter.hxx`): interleave several
/// same-typed scalar images into one vector image, one input per component.
///
/// The output's component count is `images.len()`
/// (itkComposeImageFilter.hxx:75, `SetNumberOfComponentsPerPixel`), its pixel
/// type is the inputs' vector variant, and its geometry is `images[0]`'s.
///
/// Errors when `images` is empty (`sitkMultiInputImageFilterTemplate.cxx.jinja`:
/// "Atleast one input is required"), when the inputs disagree on pixel type
/// (SimpleITK's `CheckImageMatchingPixelType`) or size, or when any input is
/// already a vector image (`pixel_types: BasicPixelIDTypeList`).
///
/// Note that a single-image `compose` is not a no-op: `compose(&[&float_img])`
/// yields a `VectorFloat32` image of one component, which SimpleITK keeps
/// distinct from `Float32` — the vector-ness is in the pixel id, not the count.
pub fn compose(images: &[&Image]) -> Result<Image> {
    Ok(Image::from_component_images(images)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::{Error, PixelId};

    #[test]
    fn compose_interleaves_components_and_sets_the_vector_pixel_id() {
        let r = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let g = Image::from_vec(&[2, 2], vec![10u8, 20, 30, 40]).unwrap();
        let b = Image::from_vec(&[2, 2], vec![100u8, 200, 250, 255]).unwrap();

        let rgb = compose(&[&r, &g, &b]).unwrap();
        assert_eq!(rgb.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(rgb.number_of_components_per_pixel(), 3);
        assert_eq!(rgb.size(), &[2, 2]);
        assert_eq!(
            rgb.component_slice::<u8>().unwrap(),
            &[1, 10, 100, 2, 20, 200, 3, 30, 250, 4, 40, 255]
        );
    }

    /// The output takes `images[0]`'s geometry.
    #[test]
    fn compose_copies_the_first_inputs_geometry() {
        let mut a = Image::new(&[2], PixelId::Float32);
        a.set_spacing(&[0.5]).unwrap();
        a.set_origin(&[3.0]).unwrap();
        let b = Image::new(&[2], PixelId::Float32);

        let v = compose(&[&a, &b]).unwrap();
        assert_eq!(v.spacing(), &[0.5]);
        assert_eq!(v.origin(), &[3.0]);
    }

    /// One input is not a no-op: `VectorFloat32` with one component is a
    /// different pixel id from `Float32`.
    #[test]
    fn compose_of_one_image_is_a_one_component_vector_image() {
        let a = Image::from_vec(&[2], vec![1.0f32, 2.0]).unwrap();
        let v = compose(&[&a]).unwrap();
        assert_eq!(v.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(v.number_of_components_per_pixel(), 1);
        assert_ne!(v, a);
    }

    #[test]
    fn compose_of_no_images_errors() {
        assert!(matches!(
            compose(&[]).unwrap_err(),
            crate::FilterError::Core(Error::EmptyComponentImageList)
        ));
    }

    #[test]
    fn compose_rejects_mismatched_pixel_types() {
        let a = Image::new(&[2], PixelId::Float32);
        let b = Image::new(&[2], PixelId::UInt8);
        assert!(matches!(
            compose(&[&a, &b]).unwrap_err(),
            crate::FilterError::Core(Error::PixelTypeMismatch { .. })
        ));
    }

    #[test]
    fn compose_rejects_mismatched_sizes() {
        let a = Image::new(&[2], PixelId::Float32);
        let b = Image::new(&[3], PixelId::Float32);
        assert!(matches!(
            compose(&[&a, &b]).unwrap_err(),
            crate::FilterError::Core(Error::GeometryMismatch { .. })
        ));
    }

    #[test]
    fn compose_rejects_a_vector_input() {
        let a = Image::new(&[2], PixelId::Float32);
        let v = Image::from_vec_vector(&[2], 2, vec![0.0f32; 4]).unwrap();
        assert!(matches!(
            compose(&[&a, &v]).unwrap_err(),
            crate::FilterError::Core(Error::RequiresScalarPixelType(PixelId::VectorFloat32))
        ));
    }
}
