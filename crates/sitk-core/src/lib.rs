//! Core data model for sitk-rs: the runtime-typed [`Image`], its pixel-type
//! dispatch, and the physical-space geometry that ITK/SimpleITK attach to every
//! image.
//!
//! This crate is deliberately algorithm-free — it holds pixels and geometry and
//! provides the [`dispatch_scalar!`] macro that lets the filter and transform
//! crates recover static typing over a runtime pixel type.

pub mod error;
pub mod image;
pub mod matrix;
pub mod pixel;

pub use error::{Error, Result};
pub use image::{Image, PixelBuffer};
pub use pixel::{PixelId, Scalar};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_image_has_default_geometry() {
        let img = Image::new(&[4, 3], PixelId::UInt8);
        assert_eq!(img.dimension(), 2);
        assert_eq!(img.size(), &[4, 3]);
        assert_eq!(img.number_of_pixels(), 12);
        assert_eq!(img.pixel_id(), PixelId::UInt8);
        assert_eq!(img.spacing(), &[1.0, 1.0]);
        assert_eq!(img.origin(), &[0.0, 0.0]);
        assert_eq!(img.direction(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn from_vec_roundtrips_typed_slice() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let img = Image::from_vec(&[3, 2], data.clone()).unwrap();
        assert_eq!(img.scalar_slice::<f32>().unwrap(), data.as_slice());
        // Wrong type is a typed error, not a panic.
        assert!(img.scalar_slice::<u8>().is_err());
    }

    #[test]
    fn from_vec_rejects_wrong_length() {
        let err = Image::from_vec(&[3, 2], vec![0u8; 5]).unwrap_err();
        assert_eq!(
            err,
            Error::BufferSizeMismatch {
                expected: 6,
                actual: 5
            }
        );
    }

    #[test]
    fn linear_index_is_first_axis_fastest() {
        let img = Image::new(&[4, 3], PixelId::UInt8);
        assert_eq!(img.linear_index(&[0, 0]), 0);
        assert_eq!(img.linear_index(&[1, 0]), 1);
        assert_eq!(img.linear_index(&[0, 1]), 4);
        assert_eq!(img.linear_index(&[3, 2]), 11);
    }

    #[test]
    fn physical_point_roundtrip_default_geometry() {
        let img = Image::new(&[10, 10], PixelId::UInt8);
        let idx = [3.0, 7.0];
        let p = img.continuous_index_to_physical_point(&idx);
        assert_eq!(p, vec![3.0, 7.0]);
        let back = img.physical_point_to_continuous_index(&p).unwrap();
        for (a, b) in back.iter().zip(idx.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn physical_point_roundtrip_nontrivial_geometry() {
        let mut img = Image::new(&[10, 10], PixelId::UInt8);
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-3.0, 10.0]).unwrap();
        // 90-degree rotation direction cosines.
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let idx = [4.0, 6.0];
        let p = img.continuous_index_to_physical_point(&idx);
        let back = img.physical_point_to_continuous_index(&p).unwrap();
        for (a, b) in back.iter().zip(idx.iter()) {
            assert!((a - b).abs() < 1e-10, "idx roundtrip failed: {back:?}");
        }
    }

    #[test]
    fn dispatch_scalar_selects_concrete_type() {
        fn pixel_size<T: Scalar>() -> usize {
            std::mem::size_of::<T>()
        }
        let img = Image::new(&[2, 2], PixelId::Float64);
        assert_eq!(dispatch_scalar!(img.pixel_id(), pixel_size), 8);
        let img = Image::new(&[2, 2], PixelId::Int16);
        assert_eq!(dispatch_scalar!(img.pixel_id(), pixel_size), 2);
    }

    #[test]
    fn set_spacing_rejects_non_positive() {
        let mut img = Image::new(&[2, 2], PixelId::UInt8);
        assert_eq!(img.set_spacing(&[1.0, 0.0]), Err(Error::NonPositiveSpacing));
        assert!(img.set_spacing(&[1.0]).is_err());
    }
}
