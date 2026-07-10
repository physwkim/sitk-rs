//! Core data model for sitk-rs: the runtime-typed [`Image`], its pixel-type
//! dispatch, and the physical-space geometry that ITK/SimpleITK attach to every
//! image.
//!
//! This crate is deliberately algorithm-free — it holds pixels and geometry and
//! provides the [`dispatch_scalar!`] macro that lets the filter and transform
//! crates recover static typing over a runtime pixel type.

pub mod boundary;
pub mod error;
pub mod image;
pub mod matrix;
pub mod neighborhood;
pub mod pixel;

pub use boundary::{
    BoundaryCondition, ConstantBoundaryCondition, MirrorBoundaryCondition,
    PeriodicBoundaryCondition, ZeroFluxNeumannBoundaryCondition,
};
pub use error::{Error, Result};
pub use image::{Image, PixelBuffer, ScalarView};
pub use neighborhood::{Neighborhood, NeighborhoodIterator};
pub use pixel::{Complex, PixelId, Real, Scalar};

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

    // ---- vector pixel types -----------------------------------------------

    /// Every `PixelId`, in discriminant order. A compile-time exhaustive match
    /// would be better still, but this at least fails loudly when a variant is
    /// added without extending the partition test below.
    const ALL_PIXEL_IDS: [PixelId; 22] = [
        PixelId::UInt8,
        PixelId::Int8,
        PixelId::UInt16,
        PixelId::Int16,
        PixelId::UInt32,
        PixelId::Int32,
        PixelId::UInt64,
        PixelId::Int64,
        PixelId::Float32,
        PixelId::Float64,
        PixelId::ComplexFloat32,
        PixelId::ComplexFloat64,
        PixelId::VectorUInt8,
        PixelId::VectorInt8,
        PixelId::VectorUInt16,
        PixelId::VectorInt16,
        PixelId::VectorUInt32,
        PixelId::VectorInt32,
        PixelId::VectorUInt64,
        PixelId::VectorInt64,
        PixelId::VectorFloat32,
        PixelId::VectorFloat64,
    ];

    #[test]
    fn pixel_id_discriminants_match_simpleitk() {
        // sitkPixelIDValues.h:103-131 — scalars 0..=9, complex 10..=11,
        // vectors 12..=21. Label ids 22..=25 are not modelled here.
        for (i, id) in ALL_PIXEL_IDS.iter().enumerate() {
            assert_eq!(*id as i8, i as i8, "{id:?}");
        }
        assert_eq!(PixelId::ComplexFloat32 as i8, 10);
        assert_eq!(PixelId::ComplexFloat64 as i8, 11);
        assert_eq!(PixelId::VectorUInt8 as i8, 12);
        assert_eq!(PixelId::VectorFloat64 as i8, 21);
    }

    #[test]
    fn pixel_id_predicates_partition_the_enum() {
        // `is_scalar`/`is_complex`/`is_vector` are mutually exclusive and total.
        // Every category test in this workspace is a whitelist over them, so a
        // new variant that satisfied none would be rejected everywhere rather
        // than admitted by some `else`.
        for id in ALL_PIXEL_IDS {
            let hits = [id.is_scalar(), id.is_complex(), id.is_vector()]
                .iter()
                .filter(|b| **b)
                .count();
            assert_eq!(hits, 1, "{id:?} belongs to {hits} categories");
        }
        assert_eq!(ALL_PIXEL_IDS.iter().filter(|i| i.is_scalar()).count(), 10);
        assert_eq!(ALL_PIXEL_IDS.iter().filter(|i| i.is_complex()).count(), 2);
        assert_eq!(ALL_PIXEL_IDS.iter().filter(|i| i.is_vector()).count(), 10);
    }

    #[test]
    fn complex_pixel_id_projections() {
        for (complex, component) in [
            (PixelId::ComplexFloat32, PixelId::Float32),
            (PixelId::ComplexFloat64, PixelId::Float64),
        ] {
            assert_eq!(complex.component_id(), component);
            // GetSizeOfPixelComponent: per component, not per pixel.
            assert_eq!(complex.size_in_bytes(), component.size_in_bytes());
            assert!(complex.is_floating_point());
            assert!(complex.is_signed());
            assert!(!complex.is_scalar());
            assert!(!complex.is_vector());
        }
        assert_eq!(<f32 as Real>::COMPLEX_ID, PixelId::ComplexFloat32);
        assert_eq!(<f64 as Real>::COMPLEX_ID, PixelId::ComplexFloat64);
    }

    #[test]
    fn pixel_id_component_and_vector_projections_are_total() {
        const SCALARS: [PixelId; 10] = [
            PixelId::UInt8,
            PixelId::Int8,
            PixelId::UInt16,
            PixelId::Int16,
            PixelId::UInt32,
            PixelId::Int32,
            PixelId::UInt64,
            PixelId::Int64,
            PixelId::Float32,
            PixelId::Float64,
        ];
        for id in SCALARS {
            assert!(id.is_scalar());
            assert!(!id.is_vector());
            assert!(id.vector_id().is_vector());
            assert_eq!(id.vector_id().component_id(), id);
            assert_eq!(id.component_id(), id);
            assert_eq!(id.vector_id().vector_id(), id.vector_id());
            // The scalar/vector projections agree on every derived property.
            assert_eq!(id.size_in_bytes(), id.vector_id().size_in_bytes());
            assert_eq!(id.is_signed(), id.vector_id().is_signed());
            assert_eq!(id.is_floating_point(), id.vector_id().is_floating_point());
        }
    }

    #[test]
    fn new_vector_image_defaults_to_dimension_components() {
        // SimpleITK's `Image(size, sitkVectorFloat32)` substitutes ImageDimension
        // for a component count of 0 (sitkImage.hxx:70-73).
        let img = Image::new(&[4, 3], PixelId::VectorFloat32);
        assert_eq!(img.number_of_components_per_pixel(), 2);
        assert_eq!(img.number_of_pixels(), 12);
        assert_eq!(img.buffer().len(), 24);

        let img = Image::new(&[4, 3, 2], PixelId::VectorFloat32);
        assert_eq!(img.number_of_components_per_pixel(), 3);
        assert_eq!(img.number_of_pixels(), 24);
        assert_eq!(img.buffer().len(), 72);
    }

    #[test]
    fn new_image_scalar_pixel_id_keeps_one_component() {
        let img = Image::new(&[4, 3], PixelId::Float32);
        assert_eq!(img.number_of_components_per_pixel(), 1);
        assert_eq!(img.buffer().len(), img.number_of_pixels());
    }

    #[test]
    fn new_vector_rejects_illegal_component_counts() {
        // A scalar pixel id admits exactly one component.
        assert_eq!(
            Image::new_vector(&[2, 2], PixelId::Float32, 3),
            Err(Error::InvalidComponentCount {
                pixel_id: PixelId::Float32,
                components_per_pixel: 3,
            })
        );
        assert!(Image::new_vector(&[2, 2], PixelId::Float32, 1).is_ok());
        // A vector pixel id admits any count >= 1.
        assert_eq!(
            Image::new_vector(&[2, 2], PixelId::VectorFloat32, 0),
            Err(Error::InvalidComponentCount {
                pixel_id: PixelId::VectorFloat32,
                components_per_pixel: 0,
            })
        );
        assert!(Image::new_vector(&[2, 2], PixelId::VectorFloat32, 1).is_ok());
        assert!(Image::new_vector(&[2, 2], PixelId::VectorFloat32, 7).is_ok());
    }

    #[test]
    fn one_component_vector_is_distinct_from_scalar() {
        // SimpleITK's sitkVectorFloat32 with one component names
        // itk::VectorImage<float>, not itk::Image<float>.
        let scalar = Image::from_vec(&[2, 2], vec![1.0f32; 4]).unwrap();
        let vector = Image::from_vec_vector(&[2, 2], 1, vec![1.0f32; 4]).unwrap();
        assert_eq!(scalar.pixel_id(), PixelId::Float32);
        assert_eq!(vector.pixel_id(), PixelId::VectorFloat32);
        assert_ne!(scalar.pixel_id(), vector.pixel_id());
        assert_eq!(scalar.number_of_components_per_pixel(), 1);
        assert_eq!(vector.number_of_components_per_pixel(), 1);
        assert_eq!(scalar.buffer(), vector.buffer());
        assert_ne!(scalar, vector);
        // The scalar guard fires on the one-component vector image all the same.
        assert!(scalar.scalar_slice::<f32>().is_ok());
        assert_eq!(
            vector.scalar_slice::<f32>(),
            Err(Error::RequiresScalarPixelType(PixelId::VectorFloat32))
        );
    }

    #[test]
    fn from_vec_vector_checks_interleaved_length() {
        assert!(Image::from_vec_vector(&[3, 2], 3, vec![0u8; 18]).is_ok());
        assert_eq!(
            Image::from_vec_vector(&[3, 2], 3, vec![0u8; 17]),
            Err(Error::BufferSizeMismatch {
                expected: 18,
                actual: 17,
            })
        );
        assert_eq!(
            Image::from_vec_vector(&[3, 2], 0, vec![0u8; 0]),
            Err(Error::InvalidComponentCount {
                pixel_id: PixelId::VectorUInt8,
                components_per_pixel: 0,
            })
        );
    }

    #[test]
    fn scalar_accessors_reject_vector_images() {
        let mut img = Image::from_vec_vector(&[2, 2], 3, vec![0.0f64; 12]).unwrap();
        let expected = || Error::RequiresScalarPixelType(PixelId::VectorFloat64);
        assert_eq!(img.scalar_slice::<f64>(), Err(expected()));
        assert_eq!(img.scalar_vec_mut::<f64>().err(), Some(expected()));
        // Component-aware accessors see the whole interleaved buffer.
        assert_eq!(img.component_slice::<f64>().unwrap().len(), 12);
        assert_eq!(img.components_to_f64_vec().len(), 12);
    }

    #[test]
    fn scalar_accessors_still_reject_the_wrong_scalar_type() {
        // The vector guard must not mask the pre-existing type check.
        let img = Image::from_vec(&[2, 2], vec![0.0f64; 4]).unwrap();
        assert_eq!(
            img.scalar_slice::<u8>(),
            Err(Error::PixelTypeMismatch {
                expected: PixelId::Float64,
                requested: PixelId::UInt8,
            })
        );
    }

    #[test]
    fn component_slice_rejects_the_wrong_component_type() {
        let img = Image::from_vec_vector(&[2, 2], 3, vec![0.0f64; 12]).unwrap();
        assert_eq!(
            img.component_slice::<f32>(),
            Err(Error::PixelTypeMismatch {
                expected: PixelId::Float64,
                requested: PixelId::Float32,
            })
        );
    }

    #[test]
    fn component_index_interleaves() {
        let img = Image::new(&[4, 3], PixelId::VectorUInt8);
        assert_eq!(img.number_of_components_per_pixel(), 2);
        assert_eq!(img.linear_index(&[1, 0]), 1);
        assert_eq!(img.component_index(&[0, 0], 0), 0);
        assert_eq!(img.component_index(&[0, 0], 1), 1);
        assert_eq!(img.component_index(&[1, 0], 0), 2);
        assert_eq!(img.component_index(&[3, 2], 1), 23);
        // A scalar image's component index degenerates to its linear index.
        let img = Image::new(&[4, 3], PixelId::UInt8);
        assert_eq!(img.component_index(&[3, 2], 0), img.linear_index(&[3, 2]));
    }

    #[test]
    fn get_and_set_vector_roundtrip() {
        let mut img = Image::from_vec_vector(&[2, 2], 3, (0..12).collect::<Vec<i16>>()).unwrap();
        assert_eq!(img.get_vector::<i16>(&[0, 0]).unwrap(), &[0, 1, 2]);
        assert_eq!(img.get_vector::<i16>(&[1, 1]).unwrap(), &[9, 10, 11]);

        img.set_vector::<i16>(&[1, 0], &[-1, -2, -3]).unwrap();
        assert_eq!(img.get_vector::<i16>(&[1, 0]).unwrap(), &[-1, -2, -3]);
        // Neighbouring pixels are untouched.
        assert_eq!(img.get_vector::<i16>(&[0, 0]).unwrap(), &[0, 1, 2]);
        assert_eq!(img.get_vector::<i16>(&[0, 1]).unwrap(), &[6, 7, 8]);

        // A scalar image is a one-component vector for these accessors.
        let scalar = Image::from_vec(&[2, 2], vec![5u8, 6, 7, 8]).unwrap();
        assert_eq!(scalar.get_vector::<u8>(&[1, 0]).unwrap(), &[6]);
    }

    #[test]
    fn set_vector_rejects_wrong_component_count() {
        let mut img = Image::from_vec_vector(&[2, 2], 3, vec![0i16; 12]).unwrap();
        assert_eq!(
            img.set_vector::<i16>(&[0, 0], &[1, 2]),
            Err(Error::InvalidComponentCount {
                pixel_id: PixelId::VectorInt16,
                components_per_pixel: 2,
            })
        );
        assert_eq!(
            img.set_vector::<i16>(&[0, 0], &[1, 2, 3, 4]),
            Err(Error::InvalidComponentCount {
                pixel_id: PixelId::VectorInt16,
                components_per_pixel: 4,
            })
        );
    }

    #[test]
    fn compose_and_extract_are_inverse() {
        let a = Image::from_vec(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Image::from_vec(&[2, 2], vec![10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let v = Image::from_component_images(&[&a, &b]).unwrap();

        assert_eq!(v.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(v.number_of_components_per_pixel(), 2);
        assert_eq!(
            v.component_slice::<f32>().unwrap(),
            &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
        );
        assert_eq!(v.extract_component(0).unwrap(), a);
        assert_eq!(v.extract_component(1).unwrap(), b);
    }

    #[test]
    fn compose_preserves_first_inputs_geometry() {
        let mut a = Image::from_vec(&[2, 2], vec![0u8; 4]).unwrap();
        a.set_spacing(&[0.5, 2.0]).unwrap();
        a.set_origin(&[-1.0, 3.0]).unwrap();
        a.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let b = Image::from_vec(&[2, 2], vec![0u8; 4]).unwrap();

        let v = Image::from_component_images(&[&a, &b]).unwrap();
        assert_eq!(v.spacing(), a.spacing());
        assert_eq!(v.origin(), a.origin());
        assert_eq!(v.direction(), a.direction());
        // ...and extraction gives it back.
        assert_eq!(v.extract_component(1).unwrap().spacing(), a.spacing());
    }

    #[test]
    fn compose_of_one_image_is_a_one_component_vector_image() {
        let a = Image::from_vec(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let v = Image::from_component_images(&[&a]).unwrap();
        assert_eq!(v.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(v.number_of_components_per_pixel(), 1);
        assert_eq!(v.extract_component(0).unwrap(), a);
    }

    #[test]
    fn compose_rejects_empty_mismatched_and_vector_inputs() {
        assert_eq!(
            Image::from_component_images(&[]),
            Err(Error::EmptyComponentImageList)
        );

        let a = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        let wrong_type = Image::from_vec(&[2, 2], vec![0.0f64; 4]).unwrap();
        assert_eq!(
            Image::from_component_images(&[&a, &wrong_type]),
            Err(Error::PixelTypeMismatch {
                expected: PixelId::Float32,
                requested: PixelId::Float64,
            })
        );

        let wrong_size = Image::from_vec(&[2, 3], vec![0.0f32; 6]).unwrap();
        assert_eq!(
            Image::from_component_images(&[&a, &wrong_size]),
            Err(Error::GeometryMismatch { dimension: 2 })
        );

        let already_vector = Image::from_vec_vector(&[2, 2], 2, vec![0.0f32; 8]).unwrap();
        assert_eq!(
            Image::from_component_images(&[&already_vector]),
            Err(Error::RequiresScalarPixelType(PixelId::VectorFloat32))
        );
    }

    #[test]
    fn extract_component_bounds_and_scalar_input() {
        let v = Image::from_vec_vector(&[2, 2], 3, vec![0u16; 12]).unwrap();
        assert!(v.extract_component(2).is_ok());
        assert_eq!(
            v.extract_component(3),
            Err(Error::ComponentIndexOutOfRange {
                index: 3,
                components_per_pixel: 3,
            })
        );

        let scalar = Image::from_vec(&[2, 2], vec![0u16; 4]).unwrap();
        assert_eq!(
            scalar.extract_component(0),
            Err(Error::RequiresVectorPixelType(PixelId::UInt16))
        );
    }

    #[test]
    fn extracted_component_has_the_component_pixel_type() {
        let v = Image::from_vec_vector(&[2, 1], 2, vec![1u8, 2, 3, 4]).unwrap();
        let c = v.extract_component(1).unwrap();
        assert_eq!(c.pixel_id(), PixelId::UInt8);
        assert_eq!(c.number_of_components_per_pixel(), 1);
        assert_eq!(c.scalar_slice::<u8>().unwrap(), &[2, 4]);
    }

    #[test]
    fn dispatch_scalar_on_a_vector_id_selects_the_component_type() {
        fn pixel_size<T: Scalar>() -> usize {
            std::mem::size_of::<T>()
        }
        let img = Image::new(&[2, 2], PixelId::VectorFloat64);
        assert_eq!(dispatch_scalar!(img.pixel_id(), pixel_size), 8);
        let img = Image::new(&[2, 2], PixelId::VectorInt16);
        assert_eq!(dispatch_scalar!(img.pixel_id(), pixel_size), 2);
    }

    // ---- complex pixel types ----------------------------------------------

    #[test]
    fn new_complex_image_has_stride_two_and_one_component_per_pixel() {
        let img = Image::new(&[4, 3], PixelId::ComplexFloat32);
        // sitkPimpleImageBase.hxx:202-209 — `1` for a basic pixel type.
        assert_eq!(img.number_of_components_per_pixel(), 1);
        assert_eq!(img.buffer_stride(), 2);
        assert_eq!(img.number_of_pixels(), 12);
        assert_eq!(img.buffer().len(), 24);
        assert_eq!(img.buffer().component_id(), PixelId::Float32);
        // ...unlike `Image::new` on a vector id, which takes `size.len()`.
        assert_eq!(
            Image::new(&[4, 3], PixelId::VectorFloat32).buffer_stride(),
            2
        );
        assert_eq!(
            Image::new(&[4, 3], PixelId::VectorFloat32).number_of_components_per_pixel(),
            2
        );
    }

    #[test]
    fn new_vector_rejects_a_complex_component_count_other_than_one() {
        // AllocateInternal's basic-pixel-type branch (sitkImage.hxx:63-67)
        // accepts only 1 (or the 0 that `Image::new` substitutes away).
        assert!(Image::new_vector(&[2, 2], PixelId::ComplexFloat64, 1).is_ok());
        for bad in [0usize, 2, 3] {
            assert_eq!(
                Image::new_vector(&[2, 2], PixelId::ComplexFloat64, bad),
                Err(Error::InvalidComponentCount {
                    pixel_id: PixelId::ComplexFloat64,
                    components_per_pixel: bad,
                })
            );
        }
    }

    #[test]
    fn from_vec_complex_interleaves_re_im() {
        let data = vec![
            Complex::new(1.0f32, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        let img = Image::from_vec_complex(&[2, 2], data).unwrap();
        assert_eq!(img.pixel_id(), PixelId::ComplexFloat32);
        // The exact layout `GetBufferAsFloat()` reinterpret-casts to.
        assert_eq!(
            img.complex_components::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(img.component_slice::<f32>().unwrap().len(), 8);
        assert_eq!(img.components_to_f64_vec().len(), 8);
    }

    #[test]
    fn from_vec_complex_length_is_counted_in_pixels() {
        assert_eq!(
            Image::from_vec_complex(&[2, 2], vec![Complex::new(0.0f64, 0.0); 3]),
            Err(Error::BufferSizeMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn get_and_set_complex_roundtrip() {
        let mut img = Image::new(&[3, 2], PixelId::ComplexFloat64);
        assert_eq!(
            img.get_complex::<f64>(&[1, 1]).unwrap(),
            Complex::new(0.0, 0.0)
        );

        img.set_complex::<f64>(&[1, 1], Complex::new(-1.5, 2.25))
            .unwrap();
        assert_eq!(
            img.get_complex::<f64>(&[1, 1]).unwrap(),
            Complex::new(-1.5, 2.25)
        );
        // Neighbouring pixels untouched, and the write landed at 2*linear_index.
        assert_eq!(
            img.get_complex::<f64>(&[0, 1]).unwrap(),
            Complex::new(0.0, 0.0)
        );
        assert_eq!(
            img.get_complex::<f64>(&[2, 1]).unwrap(),
            Complex::new(0.0, 0.0)
        );
        let flat = img.complex_components::<f64>().unwrap();
        assert_eq!(flat[2 * img.linear_index(&[1, 1])], -1.5);
        assert_eq!(flat[2 * img.linear_index(&[1, 1]) + 1], 2.25);
    }

    #[test]
    fn set_complex_preserves_negative_zero() {
        // -0.0 is a distinct bit pattern that atan2 and the sign of a real part
        // both observe; the buffer must not normalize it away.
        let mut img = Image::new(&[1], PixelId::ComplexFloat32);
        img.set_complex::<f32>(&[0], Complex::new(-0.0f32, -0.0))
            .unwrap();
        let v = img.get_complex::<f32>(&[0]).unwrap();
        assert!(v.re.is_sign_negative() && v.im.is_sign_negative());
        assert_eq!(v, Complex::new(0.0, 0.0)); // -0.0 == 0.0 by IEEE
    }

    #[test]
    fn complex_component_index_and_get_vector_use_the_stride() {
        let img = Image::from_vec_complex(&[2, 2], vec![Complex::new(1.0f32, 2.0); 4]).unwrap();
        assert_eq!(img.component_index(&[0, 0], 0), 0);
        assert_eq!(img.component_index(&[0, 0], 1), 1);
        assert_eq!(img.component_index(&[1, 0], 0), 2);
        assert_eq!(img.component_index(&[1, 1], 1), 7);
        // get_vector hands back the whole pixel: [re, im].
        assert_eq!(img.get_vector::<f32>(&[1, 1]).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn scalar_accessors_reject_complex_images() {
        // The whitelist guard: `!is_vector()` would have admitted these and
        // handed a 2N-long slice to a consumer that indexes it per pixel.
        let mut img = Image::new(&[2, 2], PixelId::ComplexFloat32);
        let expected = || Error::RequiresScalarPixelType(PixelId::ComplexFloat32);
        assert_eq!(img.scalar_slice::<f32>(), Err(expected()));
        assert_eq!(img.scalar_view::<f32>().err(), Some(expected()));
        assert_eq!(img.scalar_vec_mut::<f32>().err(), Some(expected()));
        assert_eq!(img.to_f64_vec(), Err(expected()));
    }

    #[test]
    fn complex_accessors_reject_scalar_and_vector_images() {
        let mut scalar = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            scalar.get_complex::<f32>(&[0, 0]),
            Err(Error::RequiresComplexPixelType(PixelId::Float32))
        );
        assert_eq!(
            scalar.set_complex::<f32>(&[0, 0], Complex::new(1.0, 1.0)),
            Err(Error::RequiresComplexPixelType(PixelId::Float32))
        );
        assert_eq!(
            scalar.complex_components::<f32>(),
            Err(Error::RequiresComplexPixelType(PixelId::Float32))
        );
        assert_eq!(
            scalar.complex_components_mut::<f32>().err(),
            Some(Error::RequiresComplexPixelType(PixelId::Float32))
        );

        let vector = Image::from_vec_vector(&[2, 2], 2, vec![0.0f32; 8]).unwrap();
        assert_eq!(
            vector.complex_components::<f32>(),
            Err(Error::RequiresComplexPixelType(PixelId::VectorFloat32))
        );
    }

    #[test]
    fn complex_accessors_reject_the_wrong_component_type() {
        let img = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            img.complex_components::<f64>(),
            Err(Error::PixelTypeMismatch {
                expected: PixelId::Float32,
                requested: PixelId::Float64,
            })
        );
    }

    #[test]
    fn compose_rejects_a_complex_input() {
        let c = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            Image::from_component_images(&[&c]),
            Err(Error::RequiresScalarPixelType(PixelId::ComplexFloat32))
        );
    }

    #[test]
    fn extract_component_rejects_a_complex_input() {
        // Complex is not a vector, so `VectorIndexSelectionCast` refuses it —
        // `complex_components` / `ComplexToReal` are the way to its halves.
        let c = Image::new(&[2, 2], PixelId::ComplexFloat64);
        assert_eq!(
            c.extract_component(0),
            Err(Error::RequiresVectorPixelType(PixelId::ComplexFloat64))
        );
    }

    #[test]
    fn dispatch_scalar_on_a_complex_id_selects_the_component_type() {
        fn pixel_size<T: Scalar>() -> usize {
            std::mem::size_of::<T>()
        }
        let img = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(dispatch_scalar!(img.pixel_id(), pixel_size), 4);
        let img = Image::new(&[2, 2], PixelId::ComplexFloat64);
        assert_eq!(dispatch_scalar!(img.pixel_id(), pixel_size), 8);
    }
}
