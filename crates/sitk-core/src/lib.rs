//! Core data model for sitk-rs: the runtime-typed [`Image`], its pixel-type
//! dispatch, and the physical-space geometry that ITK/SimpleITK attach to every
//! image.
//!
//! This crate is otherwise algorithm-free — it holds pixels and geometry and
//! provides the [`dispatch_scalar!`] macro that lets the filter and transform
//! crates recover static typing over a runtime pixel type. The one exception
//! is [`ops`]'s `std::ops` operator overloads (`img1 + img2`, ...): Rust's
//! orphan rule only lets the crate that defines [`Image`] implement a foreign
//! trait like `std::ops::Add` for it, so that arithmetic has to live here
//! rather than in `sitk-filters` — see the [`ops`] module docs.

pub mod boundary;
pub mod error;
pub mod image;
pub mod label_map;
pub mod matrix;
pub mod neighborhood;
pub mod ops;
pub mod pixel;

pub use boundary::{
    BoundaryCondition, ConstantBoundaryCondition, MirrorBoundaryCondition,
    PeriodicBoundaryCondition, ZeroFluxNeumannBoundaryCondition,
};
pub use error::{Error, Result};
pub use image::{Image, PixelBuffer, ScalarView};
pub use label_map::{LabelMap, LabelObject, LabelObjectLine, MAX_DIM};
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
            // `size_in_bytes` is the component size — as is this port's
            // `size_of_pixel_component`, unlike SimpleITK's, which doubles it
            // for a complex pixel; see §3.20 and
            // `size_of_pixel_component_reports_the_complex_component`.
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

    // ---- pixel index bounds: the `checked_pixel_start` seam -----------------

    #[test]
    fn pixel_accessors_reject_an_index_past_an_axis() {
        // [3, 0] on a 3x3 image has linear index 3 — a valid *buffer* offset
        // that names the pixel at [0, 1]. Every accessor must refuse it, as
        // `PimpleImage::GetIndex`'s `IsInside` test does.
        let mut img = Image::from_vec(&[3, 3], (0..9u8).collect::<Vec<u8>>()).unwrap();
        let expected = || Error::IndexOutOfBounds {
            index: vec![3, 0],
            size: vec![3, 3],
        };
        assert_eq!(img.get_pixel_as::<u8>(&[3, 0]), Err(expected()));
        assert_eq!(img.set_pixel_as::<u8>(&[3, 0], 1), Err(expected()));
        assert_eq!(img.get_vector::<u8>(&[3, 0]), Err(expected()));
        assert_eq!(img.set_vector::<u8>(&[3, 0], &[1]), Err(expected()));
        // The last in-bounds index still resolves.
        assert_eq!(img.get_pixel_as::<u8>(&[2, 2]).unwrap(), 8);
    }

    #[test]
    fn complex_accessors_bounds_check_the_index() {
        let mut img = Image::new(&[2, 2], PixelId::ComplexFloat32);
        let expected = || Error::IndexOutOfBounds {
            index: vec![0, 2],
            size: vec![2, 2],
        };
        assert_eq!(img.get_complex::<f32>(&[0, 2]), Err(expected()));
        assert_eq!(
            img.set_complex::<f32>(&[0, 2], Complex::new(1.0, 1.0)),
            Err(expected())
        );
    }

    #[test]
    fn pixel_accessors_need_at_least_dimension_index_elements() {
        let img = Image::from_vec(&[2, 2, 2], (0..8u8).collect::<Vec<u8>>()).unwrap();
        assert_eq!(
            img.get_pixel_as::<u8>(&[1, 1]),
            Err(Error::IndexDimensionMismatch {
                dimension: 3,
                actual: 2,
            })
        );
        // "additional elements will be ignored" — sitkImage.h:499-501.
        assert_eq!(img.get_pixel_as::<u8>(&[1, 1, 1, 9]).unwrap(), 7);
    }

    #[test]
    fn pixel_type_is_checked_before_the_index() {
        // `InternalGetPixelAs` selects on the pixel type `if constexpr` and only
        // then calls `GetIndex`, so a wrong `T` wins over an out-of-bounds index.
        let img = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            img.get_pixel_as::<u8>(&[99, 99]),
            Err(Error::PixelTypeMismatch {
                expected: PixelId::Float32,
                requested: PixelId::UInt8,
            })
        );
    }

    // ---- scalar get_pixel_as / set_pixel_as --------------------------------

    #[test]
    fn get_and_set_pixel_as_roundtrip() {
        let mut img = Image::from_vec(&[3, 2], (0..6i16).collect::<Vec<i16>>()).unwrap();
        assert_eq!(img.get_pixel_as::<i16>(&[0, 0]).unwrap(), 0);
        assert_eq!(img.get_pixel_as::<i16>(&[2, 1]).unwrap(), 5);

        img.set_pixel_as::<i16>(&[1, 1], -7).unwrap();
        assert_eq!(img.get_pixel_as::<i16>(&[1, 1]).unwrap(), -7);
        // Neighbours untouched.
        assert_eq!(img.get_pixel_as::<i16>(&[0, 1]).unwrap(), 3);
        assert_eq!(img.get_pixel_as::<i16>(&[2, 1]).unwrap(), 5);
    }

    #[test]
    fn get_pixel_as_demands_an_exact_pixel_type_and_never_converts() {
        // Upstream `GetPixelAsInt8` on a sitkFloat32 image throws rather than
        // rounding; every non-matching `T` is a type error, widening or not.
        let mut img = Image::from_vec(&[2, 2], vec![1.5f32; 4]).unwrap();
        for err in [
            img.get_pixel_as::<f64>(&[0, 0]).err(),
            img.get_pixel_as::<i8>(&[0, 0]).err(),
            img.get_pixel_as::<u32>(&[0, 0]).err(),
        ] {
            assert!(matches!(err, Some(Error::PixelTypeMismatch { .. })));
        }
        assert!(matches!(
            img.set_pixel_as::<f64>(&[0, 0], 1.0),
            Err(Error::PixelTypeMismatch { .. })
        ));
        assert_eq!(img.get_pixel_as::<f32>(&[0, 0]).unwrap(), 1.5);
    }

    #[test]
    fn scalar_pixel_accessors_reject_vector_and_complex_images() {
        // The whitelist guard: a new pixel category is rejected by default.
        let mut vector = Image::from_vec_vector(&[2, 2], 1, vec![0.0f32; 4]).unwrap();
        assert_eq!(
            vector.get_pixel_as::<f32>(&[0, 0]),
            Err(Error::RequiresScalarPixelType(PixelId::VectorFloat32))
        );
        assert_eq!(
            vector.set_pixel_as::<f32>(&[0, 0], 1.0),
            Err(Error::RequiresScalarPixelType(PixelId::VectorFloat32))
        );

        let mut complex = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            complex.get_pixel_as::<f32>(&[0, 0]),
            Err(Error::RequiresScalarPixelType(PixelId::ComplexFloat32))
        );
        assert_eq!(
            complex.set_pixel_as::<f32>(&[0, 0], 1.0),
            Err(Error::RequiresScalarPixelType(PixelId::ComplexFloat32))
        );
    }

    // ---- meta-data dictionary ---------------------------------------------

    #[test]
    fn meta_data_roundtrips_and_erases() {
        let mut img = Image::new(&[2, 2], PixelId::UInt8);
        assert!(img.meta_data_keys().is_empty());
        assert!(!img.has_meta_data_key("modality"));
        assert_eq!(img.meta_data("modality"), None);

        img.set_meta_data("modality", "MR");
        assert!(img.has_meta_data_key("modality"));
        assert_eq!(img.meta_data("modality"), Some("MR"));

        // Replaces, never duplicates.
        img.set_meta_data("modality", "CT");
        assert_eq!(img.meta_data("modality"), Some("CT"));
        assert_eq!(img.meta_data_keys(), vec!["modality"]);

        // `EraseMetaData` reports whether the key was there.
        assert!(img.erase_meta_data("modality"));
        assert!(!img.erase_meta_data("modality"));
        assert!(img.meta_data_keys().is_empty());
    }

    #[test]
    fn meta_data_keys_are_in_ascending_byte_order() {
        // `MetaDataDictionary` is a `std::map<std::string, _>`; `GetKeys` walks
        // it in key order, which for `char_traits::compare` is byte order.
        let mut img = Image::new(&[2, 2], PixelId::UInt8);
        for key in ["b", "Z", "a1", "a", "0", "aa"] {
            img.set_meta_data(key, "");
        }
        assert_eq!(img.meta_data_keys(), vec!["0", "Z", "a", "a1", "aa", "b"]);
    }

    #[test]
    fn copy_information_does_not_copy_the_meta_data_dictionary() {
        // sitkImage.h:386-395: "The meta-data dictionary is *not* copied."
        let mut src = Image::new(&[2, 2], PixelId::UInt8);
        src.set_meta_data("origin-note", "from src");
        src.set_origin(&[3.0, 4.0]).unwrap();

        let mut dst = Image::new(&[2, 2], PixelId::UInt8);
        dst.copy_geometry_from(&src);
        assert_eq!(dst.origin(), &[3.0, 4.0]);
        assert!(dst.meta_data_keys().is_empty());
    }

    #[test]
    fn extract_component_starts_with_an_empty_dictionary() {
        // `extract_component` builds a fresh image through `assemble` and does
        // not copy the dictionary. (The `to_vector_image`/`to_scalar_image`
        // converters do, since §3.21 — see `both_converters_carry_the_meta_data_dictionary`.)
        let mut img = Image::from_vec_vector(&[2, 2], 2, vec![0.0f32; 8]).unwrap();
        img.set_meta_data("k", "v");
        assert!(
            img.extract_component(0)
                .unwrap()
                .meta_data_keys()
                .is_empty()
        );
    }

    // ---- integer index <-> physical point ----------------------------------

    #[test]
    fn transform_index_to_physical_point_applies_spacing_origin_and_direction() {
        let mut img = Image::new(&[4, 4], PixelId::UInt8);
        img.set_spacing(&[2.0, 3.0]).unwrap();
        img.set_origin(&[10.0, 20.0]).unwrap();
        assert_eq!(img.transform_index_to_physical_point(&[2, 1]), [14.0, 23.0]);

        // A 90-degree rotation, row-major.
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        assert_eq!(img.transform_index_to_physical_point(&[2, 1]), [7.0, 24.0]);
    }

    #[test]
    fn transform_physical_point_to_index_rounds_half_integers_up() {
        // `Math::RoundHalfIntegerUp` is `floor(x + 0.5)`: 1.5 -> 2, -1.5 -> -1.
        let img = Image::new(&[4, 4], PixelId::UInt8);
        assert_eq!(
            img.transform_physical_point_to_index(&[1.5, -1.5]).unwrap(),
            [2, -1]
        );
        assert_eq!(
            img.transform_physical_point_to_index(&[2.5, -2.5]).unwrap(),
            [3, -2]
        );
        assert_eq!(
            img.transform_physical_point_to_index(&[-0.5, 0.49])
                .unwrap(),
            [0, 0]
        );
        assert_eq!(
            img.transform_physical_point_to_index(&[-0.51, 1.49])
                .unwrap(),
            [-1, 1]
        );
    }

    #[test]
    fn physical_point_to_index_is_the_rounded_continuous_index() {
        let mut img = Image::new(&[4, 4], PixelId::UInt8);
        img.set_spacing(&[2.0, 4.0]).unwrap();
        img.set_origin(&[1.0, -1.0]).unwrap();
        // (3 - 1)/2 = 1.0, (5 + 1)/4 = 1.5 -> rounds up to 2.
        let point = [3.0, 5.0];
        assert_eq!(
            img.physical_point_to_continuous_index(&point).unwrap(),
            [1.0, 1.5]
        );
        assert_eq!(
            img.transform_physical_point_to_index(&point).unwrap(),
            [1, 2]
        );
    }

    #[test]
    fn index_to_physical_point_roundtrips_through_the_index_transform() {
        let mut img = Image::new(&[8, 8, 8], PixelId::UInt8);
        img.set_spacing(&[0.5, 1.25, 2.0]).unwrap();
        img.set_origin(&[-3.0, 7.0, 0.25]).unwrap();
        img.set_direction(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .unwrap();
        let index = [5i64, 2, 7];
        let point = img.transform_index_to_physical_point(&index);
        assert_eq!(
            img.transform_physical_point_to_index(&point).unwrap(),
            index
        );
    }

    #[test]
    fn physical_point_to_index_errors_on_a_singular_direction() {
        let mut img = Image::new(&[4, 4], PixelId::UInt8);
        img.set_direction(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            img.transform_physical_point_to_index(&[0.0, 0.0]),
            Err(Error::SingularDirection)
        );
    }

    // ---- geometry comparators ----------------------------------------------

    fn geometry(size: &[usize], spacing: &[f64], origin: &[f64]) -> Image {
        let mut img = Image::new(size, PixelId::UInt8);
        img.set_spacing(spacing).unwrap();
        img.set_origin(origin).unwrap();
        img
    }

    #[test]
    fn congruent_geometry_tolerance_is_inclusive() {
        // coordinateTol = |coordinateTolerance * self.spacing[0]| = 0.5 * 2 = 1.0,
        // and `vnl::is_equal` accepts a difference of exactly the tolerance.
        let a = geometry(&[4, 4], &[2.0, 1.0], &[0.0, 0.0]);
        let on_boundary = geometry(&[4, 4], &[2.0, 1.0], &[1.0, 0.0]);
        let past_boundary = geometry(&[4, 4], &[2.0, 1.0], &[1.0 + f64::EPSILON, 0.0]);
        assert!(a.is_congruent_image_geometry(&on_boundary, 0.5, 1e-6));
        assert!(!a.is_congruent_image_geometry(&past_boundary, 0.5, 1e-6));
    }

    #[test]
    fn congruent_geometry_tolerance_is_asymmetric() {
        // itkImageBase.hxx:400-401 scales the coordinate tolerance by *this*
        // image's first-dimension spacing, so the relation is not symmetric.
        let coarse = geometry(&[4, 4], &[2.0, 1.0], &[0.0, 0.0]);
        let fine = geometry(&[4, 4], &[1.0, 1.0], &[0.0, 0.0]);
        // spacing differs by 1.0; coarse's tolerance is 0.5*2 = 1.0, fine's is 0.5.
        assert!(coarse.is_congruent_image_geometry(&fine, 0.5, 1.0));
        assert!(!fine.is_congruent_image_geometry(&coarse, 0.5, 1.0));
    }

    #[test]
    fn congruent_geometry_direction_tolerance_is_not_scaled_by_spacing() {
        let mut a = geometry(&[4, 4], &[10.0, 10.0], &[0.0, 0.0]);
        let mut b = geometry(&[4, 4], &[10.0, 10.0], &[0.0, 0.0]);
        a.set_direction(&[1.0, 0.0, 0.0, 1.0]).unwrap();
        b.set_direction(&[1.5, 0.0, 0.0, 1.0]).unwrap();
        assert!(a.is_congruent_image_geometry(&b, 1e-6, 0.5));
        assert!(!a.is_congruent_image_geometry(&b, 1e-6, 0.4));
    }

    #[test]
    fn congruent_geometry_rejects_nan_and_a_dimension_mismatch() {
        let a = geometry(&[4, 4], &[1.0, 1.0], &[0.0, 0.0]);
        let nan_origin = geometry(&[4, 4], &[1.0, 1.0], &[f64::NAN, 0.0]);
        // `!(|a-b| <= tol)` is true for NaN, so the images are not equal at any
        // tolerance.
        assert!(!a.is_congruent_image_geometry(&nan_origin, f64::INFINITY, f64::INFINITY));

        let three_d = geometry(&[4, 4, 4], &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]);
        assert!(!a.is_congruent_image_geometry(&three_d, 1e-6, 1e-6));
        assert!(!a.is_same_image_geometry_as(&three_d, 1e-6, 1e-6));
    }

    #[test]
    fn same_image_geometry_additionally_compares_the_region() {
        let a = geometry(&[4, 4], &[1.0, 1.0], &[0.0, 0.0]);
        let bigger = geometry(&[5, 4], &[1.0, 1.0], &[0.0, 0.0]);
        let tol = Image::DEFAULT_IMAGE_COORDINATE_TOLERANCE;
        let dir_tol = Image::DEFAULT_IMAGE_DIRECTION_TOLERANCE;
        assert!(a.is_congruent_image_geometry(&bigger, tol, dir_tol));
        assert!(!a.is_same_image_geometry_as(&bigger, tol, dir_tol));
        assert!(a.is_same_image_geometry_as(&a.clone(), tol, dir_tol));
    }

    #[test]
    fn default_geometry_tolerances_match_upstream() {
        assert_eq!(Image::DEFAULT_IMAGE_COORDINATE_TOLERANCE, 1e-6);
        assert_eq!(Image::DEFAULT_IMAGE_DIRECTION_TOLERANCE, 1e-6);
    }

    // ---- size_of_pixel_component / pixel_id_type_as_string / Display -------

    #[test]
    fn size_of_pixel_component_reports_the_complex_component() {
        // sitkImage.cxx:206-212 returns 2*sizeof(component) for the complex
        // pixel types, contradicting its own doc; sitkImageTests.cxx:1166 pins
        // that wrong value. §3.20: this port returns the documented one.
        assert_eq!(
            Image::new(&[2], PixelId::Float32).size_of_pixel_component(),
            4
        );
        assert_eq!(
            Image::new(&[2], PixelId::ComplexFloat32).size_of_pixel_component(),
            4
        );
        assert_eq!(
            Image::new(&[2], PixelId::ComplexFloat64).size_of_pixel_component(),
            8
        );
        // A vector image reports its component size, whatever the vector length.
        let v = Image::from_vec_vector(&[2], 5, vec![0.0f64; 10]).unwrap();
        assert_eq!(v.size_of_pixel_component(), 8);
        // ... and it now agrees with `PixelId::size_in_bytes` everywhere.
        assert_eq!(PixelId::ComplexFloat32.size_in_bytes(), 4);
        assert_eq!(
            Image::new(&[2], PixelId::ComplexFloat64).size_of_pixel_component(),
            PixelId::ComplexFloat64.size_in_bytes()
        );
    }

    #[test]
    fn pixel_id_type_as_string_matches_upstream_spelling() {
        assert_eq!(PixelId::UInt8.as_str(), "8-bit unsigned integer");
        assert_eq!(PixelId::Int64.as_str(), "64-bit signed integer");
        assert_eq!(PixelId::Float64.as_str(), "64-bit float");
        assert_eq!(PixelId::ComplexFloat32.as_str(), "complex of 32-bit float");
        assert_eq!(
            PixelId::VectorUInt16.as_str(),
            "vector of 16-bit unsigned integer"
        );
        assert_eq!(
            Image::new(&[2], PixelId::VectorFloat64).pixel_id_type_as_string(),
            "vector of 64-bit float"
        );
    }

    #[test]
    fn display_names_the_pixel_type_geometry_and_dictionary() {
        let mut img = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        img.set_meta_data("modality", "MR");
        let text = img.to_string();
        assert!(text.contains("Image (32-bit float)"), "{text}");
        assert!(text.contains("Dimension: 2"), "{text}");
        assert!(text.contains("Size: [2, 2]"), "{text}");
        assert!(text.contains("modality: MR"), "{text}");

        let empty = Image::new(&[2, 2], PixelId::UInt8).to_string();
        assert!(
            empty.contains("MetaDataDictionary:\n    (empty)"),
            "{empty}"
        );
    }

    // ---- to_vector_image / to_scalar_image ---------------------------------

    /// A 3-D ramp whose **first** axis is the trivial component axis
    /// [`Image::to_vector_image`] requires — unit spacing, zero origin — while
    /// the trailing axes keep non-unit spacing/origin so the drop is still
    /// observable.
    fn ramp_3d() -> Image {
        let mut img =
            Image::from_vec(&[2, 3, 4], (0..24).map(|i| i as f32).collect::<Vec<f32>>()).unwrap();
        img.set_spacing(&[1.0, 6.0, 7.0]).unwrap();
        img.set_origin(&[0.0, 2.0, 3.0]).unwrap();
        img
    }

    #[test]
    fn to_vector_image_folds_the_first_dimension_into_the_components() {
        let img = ramp_3d();
        let v = img.to_vector_image().unwrap();
        assert_eq!(v.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(v.number_of_components_per_pixel(), 2);
        assert_eq!(v.size(), &[3, 4]);
        assert_eq!(v.spacing(), &[6.0, 7.0]);
        assert_eq!(v.origin(), &[2.0, 3.0]);
        assert_eq!(v.direction(), &[1.0, 0.0, 0.0, 1.0]);
        // The buffer is untouched: the scalar layout already interleaves.
        assert_eq!(v.buffer(), img.buffer());
        assert_eq!(v.get_vector::<f32>(&[0, 0]).unwrap(), &[0.0, 1.0]);
        assert_eq!(v.get_vector::<f32>(&[2, 3]).unwrap(), &[22.0, 23.0]);
    }

    #[test]
    fn to_vector_image_takes_the_trailing_direction_submatrix() {
        let mut img = ramp_3d();
        img.set_direction(&[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
            .unwrap();
        let v = img.to_vector_image().unwrap();
        assert_eq!(v.direction(), &[0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn to_vector_image_rejects_a_non_identity_first_dimension_direction() {
        // sitkImage.hxx:134-145, an exact comparison against the identity.
        for direction in [
            [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // direction[0][0] != 1
            [1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  // direction[0][1] != 0
            [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0],  // direction[1][0] != 0
        ] {
            let mut img = ramp_3d();
            img.set_direction(&direction).unwrap();
            assert_eq!(
                img.to_vector_image(),
                Err(Error::NonIdentityFirstDimensionDirection)
            );
        }
    }

    #[test]
    fn to_vector_image_returns_a_vector_image_unchanged() {
        // `ToVectorInternal`'s `if constexpr (IsVector<...>) return *this` — and
        // it never reaches the direction check.
        let mut v = Image::from_vec_vector(&[2, 2], 3, vec![1.0f32; 12]).unwrap();
        v.set_direction(&[-1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(v.to_vector_image().unwrap(), v);
    }

    #[test]
    fn to_vector_image_rejects_two_dimensional_and_complex_images() {
        let flat = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            flat.to_vector_image(),
            Err(Error::CannotConvertToVectorImage {
                pixel_id: PixelId::Float32,
                dimension: 2,
            })
        );
        let complex = Image::new(&[2, 2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            complex.to_vector_image(),
            Err(Error::CannotConvertToVectorImage {
                pixel_id: PixelId::ComplexFloat32,
                dimension: 3,
            })
        );
    }

    #[test]
    fn to_scalar_image_unfolds_the_components_into_a_leading_dimension() {
        let mut v =
            Image::from_vec_vector(&[3, 4], 2, (0..24).map(|i| i as f32).collect()).unwrap();
        v.set_spacing(&[6.0, 7.0]).unwrap();
        v.set_origin(&[2.0, 3.0]).unwrap();
        v.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let s = v.to_scalar_image().unwrap();
        assert_eq!(s.pixel_id(), PixelId::Float32);
        assert_eq!(s.size(), &[2, 3, 4]);
        // The new axis gets unit spacing, zero origin, identity direction.
        assert_eq!(s.spacing(), &[1.0, 6.0, 7.0]);
        assert_eq!(s.origin(), &[0.0, 2.0, 3.0]);
        assert_eq!(
            s.direction(),
            &[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(s.buffer(), v.buffer());
    }

    #[test]
    fn to_scalar_image_returns_a_scalar_image_unchanged_and_rejects_complex() {
        let s = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(s.to_scalar_image().unwrap(), s);

        let complex = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            complex.to_scalar_image(),
            Err(Error::CannotConvertToScalarImage {
                pixel_id: PixelId::ComplexFloat32,
                dimension: 2,
            })
        );
    }

    #[test]
    fn vector_scalar_roundtrip_is_lossless_for_a_trivial_first_axis() {
        // §3.21: upstream's `ToVectorImage` silently dropped spacing[0]/origin[0]
        // and `ToScalarImage` refilled 1.0/0.0, making the round trip lossy. This
        // port requires the first axis to be trivial (so nothing is lost) and
        // carries the meta-data dictionary, so the round trip is the identity.
        let mut img = ramp_3d();
        img.set_meta_data("k", "v");
        let back = img.to_vector_image().unwrap().to_scalar_image().unwrap();
        assert_eq!(back.pixel_id(), img.pixel_id());
        assert_eq!(back.size(), img.size());
        assert_eq!(back.buffer(), img.buffer());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.meta_data("k"), Some("v"));
    }

    #[test]
    fn to_vector_image_rejects_a_non_trivial_first_dimension_geometry() {
        // §3.21: a non-unit spacing[0] or non-zero origin[0] would be silently
        // dropped by the component-axis collapse, so it is refused instead.
        let mut spaced = ramp_3d();
        spaced.set_spacing(&[5.0, 6.0, 7.0]).unwrap();
        assert_eq!(
            spaced.to_vector_image(),
            Err(Error::NonTrivialFirstDimensionGeometry)
        );

        let mut shifted = ramp_3d();
        shifted.set_origin(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            shifted.to_vector_image(),
            Err(Error::NonTrivialFirstDimensionGeometry)
        );
    }

    #[test]
    fn both_converters_carry_the_meta_data_dictionary() {
        // §3.21: unlike upstream, both directions copy the dictionary.
        let mut img = ramp_3d();
        img.set_meta_data("k", "v");
        assert_eq!(img.to_vector_image().unwrap().meta_data("k"), Some("v"));

        let mut v =
            Image::from_vec_vector(&[3, 4], 2, (0..24).map(|i| i as f32).collect()).unwrap();
        v.set_meta_data("j", "w");
        assert_eq!(v.to_scalar_image().unwrap().meta_data("j"), Some("w"));
    }
}
