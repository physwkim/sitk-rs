//! Filters that cross the scalar/vector pixel boundary.
//!
//! SimpleITK groups these under `ImageCompose` and `ImageIntensity`, but they
//! share the property that separates them from most filters in this crate: their
//! input or output is a vector image. The only other module of which that is
//! true is [`crate::filters::displacement_field`]; every remaining filter is scalar-only,
//! structurally — see the `vector_guard` tests in [`crate`].
//!
//! Correspondingly none of them route through `to_f64_vec`/`image_from_f64`
//! (the scalar seam, which refuses vector images); they read the interleaved
//! buffer through [`Image::component_slice`] and the component primitives
//! [`Image::from_component_images`] / [`Image::extract_component`].

use crate::core::{Error, Image, PixelId, Scalar, dispatch_scalar};

use crate::filters::Result;
use crate::filters::geometry::require_same_physical_space;

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
    // ComposeImageFilter indexes every input, so the inherited verifier walks
    // them all; each component past the first must share input 0's grid.
    if let Some((first, rest)) = images.split_first() {
        for (i, img) in rest.iter().enumerate() {
            require_same_physical_space(first, img, i + 1)?;
        }
    }
    Ok(Image::from_component_images(images)?)
}

/// `VectorIndexSelectionCastImageFilter`
/// (`itkVectorIndexSelectionCastImageFilter.h`): extract component `index` of a
/// vector image as a scalar image, cast to `output_pixel_type`.
///
/// `output_pixel_type` is SimpleITK's `OutputPixelType` member, whose default
/// `sitkUnknown` means "the input's own component type" — the yaml's
/// `custom_type2` reads `type2 = (m_OutputPixelType != sitkUnknown) ?
/// m_OutputPixelType : type1`, and its `output_pixel_type` is `typename
/// InputImageType2::InternalPixelType`. `None` is that default. Because the
/// output type is taken as `type2`'s *internal* pixel type, a vector
/// `output_pixel_type` selects its component type; `pixel_types2` explicitly
/// admits both lists (`typelist2::append<BasicPixelIDTypeList,
/// VectorPixelIDTypeList>`).
///
/// The functor is `static_cast<TOutput>(A[m_Index])`. This port's cast is
/// [`crate::filters::cast`]'s: saturating on float→int, where C++'s out-of-range
/// `static_cast` is undefined.
///
/// # Deviation: the bounds check
///
/// ITK's `BeforeThreadedGenerateData` rejects `index >= numberOfComponents`,
/// where `numberOfComponents` is the *larger* of the run-time component count
/// and `sizeof(PixelRealType) / sizeof(PixelScalarRealType)`. That second term
/// is a component count only for a fixed-length pixel such as `itk::Vector<T,
/// N>`. For the `VectorImage` this filter is instantiated on in SimpleITK the
/// pixel type is `VariableLengthVector<T>`, whose `RealType` is
/// `VariableLengthVector<double>` — a two-word struct holding a pointer and a
/// length — so the term is `sizeof(VariableLengthVector<double>) /
/// sizeof(double)`, unrelated to the vector's length, and it can only raise the
/// accepted bound. An `index` inside the widened gap passes the check and then
/// reads past the pixel's components in the functor.
///
/// This port checks the documented intent — `index >=
/// number_of_components_per_pixel()` — and returns
/// [`Error::ComponentIndexOutOfRange`].
pub fn vector_index_selection_cast(
    img: &Image,
    index: usize,
    output_pixel_type: Option<PixelId>,
) -> Result<Image> {
    let component = img.extract_component(index)?;
    match output_pixel_type {
        Some(target) => crate::filters::cast(&component, target.component_id()),
        None => Ok(component),
    }
}

/// `VectorMagnitudeImageFilter` (`itkVectorMagnitudeImageFilter.h`): the
/// Euclidean norm of every pixel's component vector, as a scalar image.
///
/// The output pixel type is the input's *component* type, not its real type:
/// the yaml's `filter_type` names `itk::Image<typename itk::NumericTraits<
/// typename InputImageType::PixelType>::ValueType, ...>`, and the `ValueType` of
/// a `VariableLengthVector<T>` is `T`. A `VectorUInt8` input therefore yields a
/// `UInt8` image.
///
/// The norm accumulates in `VariableLengthVector::RealValueType` =
/// `NumericTraits<T>::RealType` (itkVariableLengthVector.hxx:391-399,
/// `GetSquaredNorm`), which is `double` for **every** scalar component type —
/// `NumericTraits<float>::RealType` is `double` (itkNumericTraits.h:1349/1356)
/// — so this port sums squares in `f64` for all inputs, `f32` included.
///
/// The functor is `static_cast<TOutput>(A.GetNorm())`; that cast is undefined
/// in C++ when the norm exceeds an integer output's range, and saturates here.
/// Truncation toward zero is shared with C++ (`u8` norm `1.9` becomes `1`).
///
/// Errors with [`Error::RequiresVectorPixelType`] on a scalar image
/// (`pixel_types: VectorPixelIDTypeList`).
pub fn vector_magnitude(img: &Image) -> Result<Image> {
    if !img.pixel_id().is_vector() {
        return Err(Error::RequiresVectorPixelType(img.pixel_id()).into());
    }
    dispatch_scalar!(img.pixel_id(), vector_magnitude_typed, img)
}

fn vector_magnitude_typed<T: Scalar>(img: &Image) -> Result<Image> {
    let n = img.number_of_components_per_pixel();
    let components = img.component_slice::<T>()?;

    // `RealValueType` = `NumericTraits<T>::RealType` is `double` for every
    // scalar component type, `float` included (itkNumericTraits.h:1349/1356).
    let norm = |pixel: &[T]| -> f64 {
        pixel
            .iter()
            .map(|&c| c.as_f64().powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let out: Vec<T> = components
        .chunks_exact(n)
        .map(|pixel| T::from_f64(norm(pixel)))
        .collect();

    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `EdgePotentialImageFilter` (`itkEdgePotentialImageFilter.h`): `exp(-|g|)`
/// of a gradient (covariant vector) image, as a scalar image.
///
/// The functor is `static_cast<TOutput>(std::exp(-1.0 * A.GetNorm()))`
/// (itkEdgePotentialImageFilter.h:57). `A.GetNorm()` is
/// `VariableLengthVector::GetNorm` (itkVariableLengthVector.hxx:382-401), whose
/// accumulator is `RealValueType` = `NumericTraits<T>::RealType` — `double` for
/// **every** scalar component type (`NumericTraits<float>::RealType` is
/// `double`, itkNumericTraits.h:1349/1356) — and `std::exp` runs on `double`
/// too, so the whole computation is `f64` for all inputs. Same accumulator
/// rule as [`vector_magnitude`].
///
/// The output pixel type: the yaml's `output_pixel_type` is
/// `NumericTraits<NumericTraits<PixelType>::ValueType>::RealType` — the *real*
/// type of the component type, which is `double` for every input, so upstream
/// always outputs [`PixelId::Float64`]. **This port currently diverges**: it
/// outputs [`PixelId::Float32`] for a [`PixelId::VectorFloat32`] input (the
/// §2.39/§5.6 RealType misbelief; the value itself is computed in `f64` and
/// narrowed once). Flipping the output type is a breaking change tracked in
/// the upstream-findings ledger §5.6. There is no narrowing *range* concern
/// either way: every value of `exp(-|g|)` lies in `(0, 1]`.
///
/// Errors with [`Error::RequiresVectorPixelType`] on a scalar image
/// (`pixel_types: VectorPixelIDTypeList`). ITK's own constraint is
/// stronger — the functor calls `A.GetNorm()`, which does not compile for a
/// scalar pixel — so there is no upstream run-time check to mirror.
pub fn edge_potential(img: &Image) -> Result<Image> {
    if !img.pixel_id().is_vector() {
        return Err(Error::RequiresVectorPixelType(img.pixel_id()).into());
    }
    dispatch_scalar!(img.pixel_id(), edge_potential_typed, img)
}

fn edge_potential_typed<T: Scalar>(img: &Image) -> Result<Image> {
    let n = img.number_of_components_per_pixel();
    let components = img.component_slice::<T>()?;

    // The computation is `f64` throughout (`RealValueType` is `double` even
    // for a `float` component); only the output pixel type still splits —
    // Float32 for VectorFloat32 is a §5.6 divergence from the yaml's
    // always-double `output_pixel_type`.
    let mut result = if T::PIXEL_ID == PixelId::Float32 {
        let out: Vec<f32> = components
            .chunks_exact(n)
            .map(|pixel| {
                let sum: f64 = pixel.iter().map(|&c| c.as_f64().powi(2)).sum();
                (-sum.sqrt()).exp() as f32
            })
            .collect();
        Image::from_vec(img.size(), out)?
    } else {
        let out: Vec<f64> = components
            .chunks_exact(n)
            .map(|pixel| {
                let sum: f64 = pixel.iter().map(|&c| c.as_f64().powi(2)).sum();
                (-sum.sqrt()).exp()
            })
            .collect();
        Image::from_vec(img.size(), out)?
    };
    result.copy_geometry_from(img);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Error, PixelId};

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

    /// ComposeImageFilter inherits ImageToImageFilter::VerifyInputInformation, so
    /// the component images must share physical space (a mismatch is refused —
    /// pinned in tests/physical_space_precondition.rs). Given congruent inputs the
    /// output carries that geometry.
    #[test]
    fn compose_carries_the_shared_geometry() {
        let mut a = Image::new(&[2], PixelId::Float32);
        a.set_spacing(&[0.5]).unwrap();
        a.set_origin(&[3.0]).unwrap();
        let mut b = Image::new(&[2], PixelId::Float32);
        b.set_spacing(&[0.5]).unwrap();
        b.set_origin(&[3.0]).unwrap();

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
            crate::filters::FilterError::Core(Error::EmptyComponentImageList)
        ));
    }

    #[test]
    fn compose_rejects_mismatched_pixel_types() {
        let a = Image::new(&[2], PixelId::Float32);
        let b = Image::new(&[2], PixelId::UInt8);
        assert!(matches!(
            compose(&[&a, &b]).unwrap_err(),
            crate::filters::FilterError::Core(Error::PixelTypeMismatch { .. })
        ));
    }

    #[test]
    fn compose_rejects_mismatched_sizes() {
        let a = Image::new(&[2], PixelId::Float32);
        let b = Image::new(&[3], PixelId::Float32);
        assert!(matches!(
            compose(&[&a, &b]).unwrap_err(),
            crate::filters::FilterError::Core(Error::GeometryMismatch { .. })
        ));
    }

    #[test]
    fn compose_rejects_a_vector_input() {
        let a = Image::new(&[2], PixelId::Float32);
        let v = Image::from_vec_vector(&[2], 2, vec![0.0f32; 4]).unwrap();
        assert!(matches!(
            compose(&[&a, &v]).unwrap_err(),
            crate::filters::FilterError::Core(Error::RequiresScalarPixelType(
                PixelId::VectorFloat32
            ))
        ));
    }

    fn rgb() -> Image {
        // Pixel p has components (p, 10 + p, 100 + p).
        let mut data = Vec::new();
        for p in 0..4u8 {
            data.extend_from_slice(&[p, 10 + p, 100 + p]);
        }
        Image::from_vec_vector(&[2, 2], 3, data).unwrap()
    }

    #[test]
    fn index_selection_extracts_the_requested_component() {
        let blue = vector_index_selection_cast(&rgb(), 2, None).unwrap();
        assert_eq!(blue.pixel_id(), PixelId::UInt8);
        assert_eq!(blue.number_of_components_per_pixel(), 1);
        assert_eq!(blue.scalar_slice::<u8>().unwrap(), &[100, 101, 102, 103]);
    }

    /// `None` means the input's own component type, not the input's pixel id.
    #[test]
    fn index_selection_defaults_to_the_component_type() {
        let v = Image::from_vec_vector(&[2], 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let c = vector_index_selection_cast(&v, 0, None).unwrap();
        assert_eq!(c.pixel_id(), PixelId::Float32);
        assert_eq!(c.scalar_slice::<f32>().unwrap(), &[1.0, 3.0]);
    }

    #[test]
    fn index_selection_casts_to_the_requested_output_type() {
        let g = vector_index_selection_cast(&rgb(), 1, Some(PixelId::Float64)).unwrap();
        assert_eq!(g.pixel_id(), PixelId::Float64);
        assert_eq!(g.scalar_slice::<f64>().unwrap(), &[10.0, 11.0, 12.0, 13.0]);
    }

    /// A vector `output_pixel_type` names its component type, mirroring
    /// `InputImageType2::InternalPixelType`.
    #[test]
    fn index_selection_takes_a_vector_output_type_as_its_component_type() {
        let g = vector_index_selection_cast(&rgb(), 1, Some(PixelId::VectorFloat32)).unwrap();
        assert_eq!(g.pixel_id(), PixelId::Float32);
        assert_eq!(g.number_of_components_per_pixel(), 1);
    }

    #[test]
    fn index_selection_at_the_last_component_is_in_range() {
        assert!(vector_index_selection_cast(&rgb(), 2, None).is_ok());
    }

    #[test]
    fn index_selection_at_the_component_count_is_out_of_range() {
        assert!(matches!(
            vector_index_selection_cast(&rgb(), 3, None).unwrap_err(),
            crate::filters::FilterError::Core(Error::ComponentIndexOutOfRange {
                index: 3,
                components_per_pixel: 3
            })
        ));
    }

    /// Index 0 of a one-component vector image is its only component.
    #[test]
    fn index_selection_on_a_one_component_vector_image() {
        let v = Image::from_vec_vector(&[2], 1, vec![7.0f32, 8.0]).unwrap();
        assert_eq!(
            vector_index_selection_cast(&v, 0, None)
                .unwrap()
                .scalar_slice::<f32>()
                .unwrap(),
            &[7.0, 8.0]
        );
        assert!(vector_index_selection_cast(&v, 1, None).is_err());
    }

    #[test]
    fn index_selection_rejects_a_scalar_image() {
        let s = Image::new(&[2], PixelId::Float32);
        assert!(matches!(
            vector_index_selection_cast(&s, 0, None).unwrap_err(),
            crate::filters::FilterError::Core(Error::RequiresVectorPixelType(PixelId::Float32))
        ));
    }

    /// The cast saturates rather than being undefined, per this crate's policy.
    #[test]
    fn index_selection_cast_saturates_out_of_range_values() {
        let v = Image::from_vec_vector(&[2], 1, vec![-5.0f32, 400.0]).unwrap();
        let c = vector_index_selection_cast(&v, 0, Some(PixelId::UInt8)).unwrap();
        assert_eq!(c.scalar_slice::<u8>().unwrap(), &[0, 255]);
    }

    /// 3-4-5 triangle, so the norm is exact in every pixel type.
    #[test]
    fn magnitude_is_the_euclidean_norm_and_keeps_the_component_type() {
        let v = Image::from_vec_vector(&[2], 2, vec![3.0f32, 4.0, 6.0, 8.0]).unwrap();
        let m = vector_magnitude(&v).unwrap();
        assert_eq!(m.pixel_id(), PixelId::Float32);
        assert_eq!(m.number_of_components_per_pixel(), 1);
        assert_eq!(m.scalar_slice::<f32>().unwrap(), &[5.0, 10.0]);
    }

    /// The output type is `NumericTraits<PixelType>::ValueType` — the component
    /// type — not its RealType. An integer input keeps an integer output.
    #[test]
    fn magnitude_of_an_integer_vector_stays_integer_and_truncates() {
        let v = Image::from_vec_vector(&[2], 2, vec![3u8, 4, 1, 1]).unwrap();
        let m = vector_magnitude(&v).unwrap();
        assert_eq!(m.pixel_id(), PixelId::UInt8);
        // sqrt(2) == 1.414... truncates toward zero, as `static_cast<uint8_t>`.
        assert_eq!(m.scalar_slice::<u8>().unwrap(), &[5, 1]);
    }

    /// C++ leaves the out-of-range `static_cast` undefined; this port saturates.
    #[test]
    fn magnitude_saturates_an_out_of_range_integer_output() {
        let v = Image::from_vec_vector(&[1], 2, vec![200u8, 200]).unwrap();
        // sqrt(200^2 + 200^2) == 282.8, past u8::MAX.
        assert_eq!(
            vector_magnitude(&v).unwrap().scalar_slice::<u8>().unwrap(),
            &[255]
        );
    }

    /// A one-component vector image's norm is the absolute value of its only
    /// component.
    #[test]
    fn magnitude_of_a_one_component_vector_is_the_absolute_value() {
        let v = Image::from_vec_vector(&[2], 1, vec![-3.0f64, 4.0]).unwrap();
        assert_eq!(
            vector_magnitude(&v).unwrap().scalar_slice::<f64>().unwrap(),
            &[3.0, 4.0]
        );
    }

    #[test]
    fn magnitude_copies_geometry() {
        let mut v = Image::from_vec_vector(&[2], 2, vec![0.0f32; 4]).unwrap();
        v.set_spacing(&[2.5]).unwrap();
        v.set_origin(&[-1.0]).unwrap();
        let m = vector_magnitude(&v).unwrap();
        assert_eq!(m.spacing(), &[2.5]);
        assert_eq!(m.origin(), &[-1.0]);
    }

    #[test]
    fn magnitude_rejects_a_scalar_image() {
        let s = Image::new(&[2], PixelId::Float32);
        assert!(matches!(
            vector_magnitude(&s).unwrap_err(),
            crate::filters::FilterError::Core(Error::RequiresVectorPixelType(PixelId::Float32))
        ));
    }

    /// `exp(-|g|)` on a 3-4-5 pixel and a zero pixel: `exp(-5)` and `exp(0)`.
    /// The output of a `VectorFloat32` input is `Float32`.
    #[test]
    fn edge_potential_of_a_float32_field_is_exp_of_minus_the_norm() {
        let v = Image::from_vec_vector(&[2], 2, vec![3.0f32, 4.0, 0.0, 0.0]).unwrap();
        let p = edge_potential(&v).unwrap();
        assert_eq!(p.pixel_id(), PixelId::Float32);
        assert_eq!(p.number_of_components_per_pixel(), 1);
        let got = p.scalar_slice::<f32>().unwrap();
        assert_eq!(got[0], (-5.0f64).exp() as f32);
        assert_eq!(got[1], 1.0);
    }

    /// `NumericTraits<double>::RealType` is `double`.
    #[test]
    fn edge_potential_of_a_float64_field_stays_float64() {
        // (5, 12) has norm 13.
        let v = Image::from_vec_vector(&[1], 2, vec![5.0f64, 12.0]).unwrap();
        let p = edge_potential(&v).unwrap();
        assert_eq!(p.pixel_id(), PixelId::Float64);
        assert_eq!(p.scalar_slice::<f64>().unwrap(), &[(-13.0f64).exp()]);
    }

    /// `NumericTraits<unsigned char>::RealType` is `double`, so an integer
    /// component type still yields a `Float64` image — the result is never
    /// truncated to the component type the way `vector_magnitude`'s is.
    #[test]
    fn edge_potential_of_an_integer_field_is_float64() {
        let v = Image::from_vec_vector(&[2], 2, vec![3u8, 4, 0, 0]).unwrap();
        let p = edge_potential(&v).unwrap();
        assert_eq!(p.pixel_id(), PixelId::Float64);
        assert_eq!(p.scalar_slice::<f64>().unwrap(), &[(-5.0f64).exp(), 1.0]);
    }

    /// A one-component field's norm is the absolute value, so a negative
    /// component still produces a potential in `(0, 1]`.
    #[test]
    fn edge_potential_of_a_one_component_field_uses_the_absolute_value() {
        let v = Image::from_vec_vector(&[2], 1, vec![-3.0f64, 3.0]).unwrap();
        let p = edge_potential(&v).unwrap();
        assert_eq!(
            p.scalar_slice::<f64>().unwrap(),
            &[(-3.0f64).exp(), (-3.0f64).exp()]
        );
    }

    #[test]
    fn edge_potential_copies_geometry() {
        let mut v = Image::from_vec_vector(&[2], 2, vec![0.0f32; 4]).unwrap();
        v.set_spacing(&[2.5]).unwrap();
        v.set_origin(&[-1.0]).unwrap();
        let p = edge_potential(&v).unwrap();
        assert_eq!(p.spacing(), &[2.5]);
        assert_eq!(p.origin(), &[-1.0]);
    }

    #[test]
    fn edge_potential_rejects_a_scalar_image() {
        let s = Image::new(&[2], PixelId::Float32);
        assert!(matches!(
            edge_potential(&s).unwrap_err(),
            crate::filters::FilterError::Core(Error::RequiresVectorPixelType(PixelId::Float32))
        ));
    }

    /// `gradient` → `edge_potential` is the pipeline this filter exists for:
    /// a covariant-vector gradient image in, an edge-potential map out.
    #[test]
    fn edge_potential_of_a_composed_gradient_field() {
        // A pixel with gradient (0, 0) is flat -> potential 1; (0, 5) is an
        // edge -> potential exp(-5).
        let gx = Image::from_vec(&[2], vec![0.0f64, 0.0]).unwrap();
        let gy = Image::from_vec(&[2], vec![0.0f64, 5.0]).unwrap();
        let p = edge_potential(&compose(&[&gx, &gy]).unwrap()).unwrap();
        assert_eq!(p.scalar_slice::<f64>().unwrap(), &[1.0, (-5.0f64).exp()]);
    }

    /// `compose` then `vector_magnitude` is the pipeline the two filters exist
    /// to support: per-component scalar images to a magnitude image.
    #[test]
    fn compose_then_magnitude_round_trips_through_the_vector_image() {
        let x = Image::from_vec(&[2], vec![3.0f64, 5.0]).unwrap();
        let y = Image::from_vec(&[2], vec![4.0f64, 12.0]).unwrap();
        let m = vector_magnitude(&compose(&[&x, &y]).unwrap()).unwrap();
        assert_eq!(m.scalar_slice::<f64>().unwrap(), &[5.0, 13.0]);
    }
}
