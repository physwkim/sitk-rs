//! The runtime-typed N-dimensional [`Image`] and its type-erased [`PixelBuffer`].

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;

use crate::coord;
use crate::error::{Error, Result};
use crate::matrix;
use crate::pixel::{Complex, PixelId, Real, Scalar};

/// Type-erased *component* storage: one `Vec` variant per scalar component type.
///
/// Data is stored in ITK/SimpleITK order — the first index (x) varies fastest.
/// The buffer holds `number_of_pixels * buffer_stride` elements, **interleaved**:
/// the components of one pixel are adjacent. A scalar image has stride 1; a
/// vector image has stride `number_of_components_per_pixel`, exactly as
/// `itk::VectorImage` lays out its single contiguous `ImportImageContainer`
/// (itkVectorImage.h: the pixel components are stored contiguously in a buffer
/// of length `NumberOfPixels * VectorLength`); a complex image has stride 2,
/// `re, im, re, im, ...`, which is what SimpleITK's `GetBufferAsFloat()` on a
/// `sitkComplexFloat32` image reinterpret-casts to
/// (sitkPimpleImageBase.hxx:838-842).
///
/// A `PixelBuffer` therefore knows its *component* type, never which pixel
/// category the image that owns it belongs to; that distinction lives on
/// [`Image`].
#[derive(Clone, Debug, PartialEq)]
pub enum PixelBuffer {
    UInt8(Vec<u8>),
    Int8(Vec<i8>),
    UInt16(Vec<u16>),
    Int16(Vec<i16>),
    UInt32(Vec<u32>),
    Int32(Vec<i32>),
    UInt64(Vec<u64>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl PixelBuffer {
    /// A zero-filled buffer of `len` *components* of `id`'s component type.
    ///
    /// A vector or complex `id` selects the same variant as its component's
    /// scalar id; `len` is a component count, not a pixel count.
    pub fn zeroed(id: PixelId, len: usize) -> Self {
        match id {
            PixelId::UInt8 | PixelId::VectorUInt8 => PixelBuffer::UInt8(vec![0; len]),
            PixelId::Int8 | PixelId::VectorInt8 => PixelBuffer::Int8(vec![0; len]),
            PixelId::UInt16 | PixelId::VectorUInt16 => PixelBuffer::UInt16(vec![0; len]),
            PixelId::Int16 | PixelId::VectorInt16 => PixelBuffer::Int16(vec![0; len]),
            PixelId::UInt32 | PixelId::VectorUInt32 => PixelBuffer::UInt32(vec![0; len]),
            PixelId::Int32 | PixelId::VectorInt32 => PixelBuffer::Int32(vec![0; len]),
            PixelId::UInt64 | PixelId::VectorUInt64 => PixelBuffer::UInt64(vec![0; len]),
            PixelId::Int64 | PixelId::VectorInt64 => PixelBuffer::Int64(vec![0; len]),
            PixelId::Float32 | PixelId::ComplexFloat32 | PixelId::VectorFloat32 => {
                PixelBuffer::Float32(vec![0.0; len])
            }
            PixelId::Float64 | PixelId::ComplexFloat64 | PixelId::VectorFloat64 => {
                PixelBuffer::Float64(vec![0.0; len])
            }
        }
    }

    /// The runtime tag of this buffer's *components*. Always a scalar
    /// [`PixelId`]; see [`Image::pixel_id`] for the owning image's pixel type.
    pub fn component_id(&self) -> PixelId {
        match self {
            PixelBuffer::UInt8(_) => PixelId::UInt8,
            PixelBuffer::Int8(_) => PixelId::Int8,
            PixelBuffer::UInt16(_) => PixelId::UInt16,
            PixelBuffer::Int16(_) => PixelId::Int16,
            PixelBuffer::UInt32(_) => PixelId::UInt32,
            PixelBuffer::Int32(_) => PixelId::Int32,
            PixelBuffer::UInt64(_) => PixelId::UInt64,
            PixelBuffer::Int64(_) => PixelId::Int64,
            PixelBuffer::Float32(_) => PixelId::Float32,
            PixelBuffer::Float64(_) => PixelId::Float64,
        }
    }

    /// Number of *components* held — for the owning image this is
    /// `number_of_pixels * components_per_pixel`, which equals its pixel count
    /// only when the image is scalar.
    pub fn len(&self) -> usize {
        match self {
            PixelBuffer::UInt8(v) => v.len(),
            PixelBuffer::Int8(v) => v.len(),
            PixelBuffer::UInt16(v) => v.len(),
            PixelBuffer::Int16(v) => v.len(),
            PixelBuffer::UInt32(v) => v.len(),
            PixelBuffer::Int32(v) => v.len(),
            PixelBuffer::UInt64(v) => v.len(),
            PixelBuffer::Int64(v) => v.len(),
            PixelBuffer::Float32(v) => v.len(),
            PixelBuffer::Float64(v) => v.len(),
        }
    }

    /// `true` if the buffer holds no components.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the buffer as `&[T]`, or `None` if `T` is not the component type.
    ///
    /// Crate-private, and deliberately so: the returned slice holds every
    /// component of every pixel, and carries no evidence of whether the owning
    /// image is scalar. Only [`Image`]'s named accessors may hand one out —
    /// the `scalar_*` family after the vector guard, the `component_*` family
    /// under a name that says what the length means.
    pub(crate) fn as_slice<T: Scalar>(&self) -> Option<&[T]> {
        fn cast<T: 'static>(v: &dyn Any) -> Option<&[T]> {
            v.downcast_ref::<Vec<T>>().map(Vec::as_slice)
        }
        match self {
            PixelBuffer::UInt8(v) => cast(v),
            PixelBuffer::Int8(v) => cast(v),
            PixelBuffer::UInt16(v) => cast(v),
            PixelBuffer::Int16(v) => cast(v),
            PixelBuffer::UInt32(v) => cast(v),
            PixelBuffer::Int32(v) => cast(v),
            PixelBuffer::UInt64(v) => cast(v),
            PixelBuffer::Int64(v) => cast(v),
            PixelBuffer::Float32(v) => cast(v),
            PixelBuffer::Float64(v) => cast(v),
        }
    }

    /// The mutable counterpart of [`PixelBuffer::as_slice`]; crate-private for
    /// the same reason.
    pub(crate) fn as_mut_vec<T: Scalar>(&mut self) -> Option<&mut Vec<T>> {
        fn cast<T: 'static>(v: &mut dyn Any) -> Option<&mut Vec<T>> {
            v.downcast_mut::<Vec<T>>()
        }
        match self {
            PixelBuffer::UInt8(v) => cast(v),
            PixelBuffer::Int8(v) => cast(v),
            PixelBuffer::UInt16(v) => cast(v),
            PixelBuffer::Int16(v) => cast(v),
            PixelBuffer::UInt32(v) => cast(v),
            PixelBuffer::Int32(v) => cast(v),
            PixelBuffer::UInt64(v) => cast(v),
            PixelBuffer::Int64(v) => cast(v),
            PixelBuffer::Float32(v) => cast(v),
            PixelBuffer::Float64(v) => cast(v),
        }
    }

    /// Widen every stored component to `f64`, preserving interleaved order.
    ///
    /// Nearly every filter starts here, so the widening runs through
    /// [`crate::parallel::map_slice`]: a pure elementwise cast, bit-identical
    /// to the sequential map at any thread count.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        fn widen<T: Scalar>(v: &[T]) -> Vec<f64> {
            crate::parallel::map_slice(v, |&x| x.as_f64())
        }
        match self {
            PixelBuffer::UInt8(v) => widen(v),
            PixelBuffer::Int8(v) => widen(v),
            PixelBuffer::UInt16(v) => widen(v),
            PixelBuffer::Int16(v) => widen(v),
            PixelBuffer::UInt32(v) => widen(v),
            PixelBuffer::Int32(v) => widen(v),
            PixelBuffer::UInt64(v) => widen(v),
            PixelBuffer::Int64(v) => widen(v),
            PixelBuffer::Float32(v) => widen(v),
            PixelBuffer::Float64(v) => widen(v),
        }
    }
}

/// An N-dimensional image: a [`PixelBuffer`] plus the physical-space geometry
/// (size, spacing, origin, direction cosine matrix) that ITK/SimpleITK attach to
/// every image.
///
/// Geometry vectors are all indexed in axis order matching [`Image::size`]; the
/// direction matrix is stored row-major and is `dimension x dimension`.
///
/// # Scalar, complex, and vector images
///
/// Mirroring SimpleITK's `sitkImage`, one `Image` type carries `itk::Image<T>`,
/// `itk::Image<std::complex<T>>`, and `itk::VectorImage<T>`: [`Image::pixel_id`]
/// names which.
///
/// Two quantities are easily confused, and upstream keeps them apart:
///
/// - [`Image::buffer_stride`] — how many buffer components one pixel occupies.
///   This is the **stored** field. `1` for scalar, `2` for complex,
///   `number_of_components_per_pixel` for vector.
/// - [`Image::number_of_components_per_pixel`] — SimpleITK's
///   `GetNumberOfComponentsPerPixel()`, which returns the ITK vector length
///   only `if constexpr (IsVector<TImageType>::Value)` and otherwise `1`
///   (sitkPimpleImageBase.hxx:202-209). It is **derived**, and it reports `1`
///   for a complex image even though that image's buffer holds two components
///   per pixel.
///
/// They coincide for scalar and vector images, which is why one field once
/// served for both. Storing the stride and deriving the SimpleITK quantity
/// gives every path the same meaning for the stored value.
///
/// The following invariant holds by construction — every `Image` is built
/// through the private `assemble` seam, which rejects any other combination:
///
/// ```text
/// pixel_id.is_scalar()   =>  buffer_stride == 1
/// pixel_id.is_complex()  =>  buffer_stride == 2
/// pixel_id.is_vector()   =>  buffer_stride >= 1
/// buffer.component_id()  ==  pixel_id.component_id()
/// buffer.len()           ==  number_of_pixels * buffer_stride
/// ```
///
/// Consequently the scalar accessors ([`Image::scalar_slice`],
/// [`Image::scalar_vec_mut`]) can — and do — reject every non-scalar image with
/// [`Error::RequiresScalarPixelType`] rather than hand back an interleaved
/// buffer that a scalar consumer would misread. That guard is a *whitelist* on
/// [`PixelId::is_scalar`]: a pixel category added later is rejected by default.
///
/// # The meta-data dictionary
///
/// Every image carries a string-to-string dictionary, reached through
/// [`Image::meta_data_keys`], [`Image::has_meta_data_key`],
/// [`Image::meta_data`], [`Image::set_meta_data`] and
/// [`Image::erase_meta_data`] — exactly the five methods SimpleITK exposes
/// (sitkImage.h:401-432).
///
/// ITK's underlying `itk::MetaDataDictionary` is a
/// `std::map<std::string, MetaDataObjectBase::Pointer>`
/// (itkMetaDataDictionary.h:67), so an entry may hold *any* type. SimpleITK
/// flattens that on both sides: `SetMetaData` always encapsulates a
/// `std::string` (sitkImage.cxx:394-401), and `GetMetaData` returns a
/// dictionary string as-is, falling back to `mdd.Get(key)->Print(ss)` for every
/// other stored type (sitkImage.cxx:378-392). This port stores the flattened
/// form directly and does not model the typed `MetaDataObject` hierarchy.
///
/// The dictionary is a [`BTreeMap`], so [`Image::meta_data_keys`] yields keys in
/// ascending byte order. That is upstream's order too: `MetaDataDictionary::
/// GetKeys` walks the `std::map` in iteration order (itkMetaDataDictionary.cxx:
/// 100-112), and `std::map<std::string, _>` orders by `std::string`'s
/// `char_traits::compare`, i.e. `memcmp` over unsigned bytes — the same total
/// order Rust's `Ord for str` uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    buffer: PixelBuffer,
    pixel_id: PixelId,
    /// Buffer components per pixel. One meaning on every path — see the type
    /// docs. Never SimpleITK's `GetNumberOfComponentsPerPixel()`, which is
    /// [`Image::number_of_components_per_pixel`].
    buffer_stride: usize,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    metadata: BTreeMap<String, String>,
}

/// An [`Image`] borrow carrying static proof that the image is scalar (one
/// component per pixel) and that its pixel type is `T`.
///
/// The proof is discharged once, at [`Image::scalar_view`] — the only
/// constructor — and the fields are private, so a `ScalarView` cannot be
/// forged. Consumers that must read pixels from an infallible signature take a
/// `&ScalarView<'_, T>` instead of a `&Image`; they then cannot be handed a
/// vector image at all, so they need no runtime guard and have no panic path.
///
/// It also hoists the buffer lookup out of per-pixel loops.
#[derive(Debug)]
pub struct ScalarView<'a, T> {
    image: &'a Image,
    pixels: &'a [T],
}

// Derived `Clone`/`Copy` would demand `T: Clone`/`T: Copy`; the view only ever
// copies two shared borrows.
impl<T> Clone for ScalarView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ScalarView<'_, T> {}

impl<'a, T: Scalar> ScalarView<'a, T> {
    /// The image this view borrows.
    pub fn image(&self) -> &'a Image {
        self.image
    }

    /// The image's pixels, one element per pixel, dimension-0-fastest.
    pub fn pixels(&self) -> &'a [T] {
        self.pixels
    }

    /// Per-dimension size of the image.
    pub fn size(&self) -> &'a [usize] {
        self.image.size()
    }

    /// Number of image dimensions.
    pub fn dimension(&self) -> usize {
        self.image.dimension()
    }

    /// Reads the pixel at an in-bounds ND `index`.
    ///
    /// Panics if `index` is out of bounds, exactly as indexing a slice does.
    pub fn at(&self, index: &[usize]) -> T {
        self.pixels[self.image.linear_index(index)]
    }
}

impl Image {
    /// The single construction seam. Every `Image` in this workspace is built
    /// here, so the type's invariant (see the type docs) cannot be violated by
    /// any constructor, filter, or IO reader.
    ///
    /// `components_per_pixel` is SimpleITK's `numberOfComponents` constructor
    /// argument, i.e. the value [`Image::number_of_components_per_pixel`] will
    /// report — not the buffer stride, which this seam is the sole owner of.
    /// `AllocateInternal` (sitkImage.hxx:60-67, :95-100) accepts only `1` for a
    /// basic pixel type, complex included, and any count for a vector one.
    ///
    /// The assembled image's meta-data dictionary is always empty: a freshly
    /// allocated `itk::Image` has an empty dictionary. Callers that must carry a
    /// dictionary across a conversion copy it themselves after assembling —
    /// [`Image::to_vector_image`] and [`Image::to_scalar_image`] do, a
    /// deliberate divergence from upstream's `GetVectorImageFromScalarImage` /
    /// `GetScalarImageFromVectorImage` (sitkImageConvert.hxx:74-177), which build
    /// a new `itk::Image` and never copy the dictionary (ledger §3.21).
    fn assemble(
        buffer: PixelBuffer,
        pixel_id: PixelId,
        components_per_pixel: usize,
        size: Vec<usize>,
        spacing: Vec<f64>,
        origin: Vec<f64>,
        direction: Vec<f64>,
    ) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");

        let Some(buffer_stride) = pixel_id.buffer_stride_for(components_per_pixel) else {
            return Err(Error::InvalidComponentCount {
                pixel_id,
                components_per_pixel,
            });
        };
        if buffer.component_id() != pixel_id.component_id() {
            return Err(Error::PixelTypeMismatch {
                expected: pixel_id.component_id(),
                requested: buffer.component_id(),
            });
        }

        let number_of_pixels: usize = size.iter().product();
        let expected = number_of_pixels * buffer_stride;
        if buffer.len() != expected {
            return Err(Error::BufferSizeMismatch {
                expected,
                actual: buffer.len(),
            });
        }

        let dim = size.len();
        if spacing.len() != dim || origin.len() != dim || direction.len() != dim * dim {
            return Err(Error::GeometryMismatch { dimension: dim });
        }

        Ok(Image {
            buffer,
            pixel_id,
            buffer_stride,
            size,
            spacing,
            origin,
            direction,
            metadata: BTreeMap::new(),
        })
    }

    /// Default geometry for a `dim`-dimensional image: unit spacing, zero
    /// origin, identity direction.
    fn default_geometry(dim: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (vec![1.0; dim], vec![0.0; dim], matrix::identity(dim))
    }

    /// A new zero-filled image of the given `size` and pixel type, with default
    /// geometry (unit spacing, zero origin, identity direction).
    ///
    /// `size` is in SimpleITK order (`[x, y, z, ...]`) and must be non-empty.
    ///
    /// A vector `id` gets `size.len()` components per pixel, reproducing
    /// SimpleITK's `Image(size, valueEnum, numberOfComponents = 0)`, whose
    /// `AllocateInternal` (sitkImage.hxx:70-73) substitutes
    /// `TImageType::ImageDimension` for a component count of zero. Use
    /// [`Image::new_vector`] to choose the count. A scalar or complex `id` gets
    /// one component per pixel, as upstream's basic-pixel-type branch does.
    pub fn new(size: &[usize], id: PixelId) -> Self {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let components = if id.is_vector() { size.len() } else { 1 };
        Self::new_vector(size, id, components)
            .expect("`size.len() >= 1` components is legal for every pixel id")
    }

    /// A new zero-filled image with an explicit component count.
    ///
    /// A scalar or complex `id` accepts only `components_per_pixel == 1`; a
    /// vector `id` accepts any count `>= 1`. Mirrors SimpleITK's
    /// `Image(size, valueEnum, numberOfComponents)` and its `AllocateInternal`
    /// check (sitkImage.hxx:63-67), which throws "Specified number of
    /// components as N but did not specify pixelID as a vector type!".
    ///
    /// The allocated buffer is `Π size * buffer_stride` components long, so a
    /// complex `id` allocates two components per pixel while still reporting
    /// one from [`Image::number_of_components_per_pixel`].
    pub fn new_vector(size: &[usize], id: PixelId, components_per_pixel: usize) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let n: usize = size.iter().product();
        let stride = id.buffer_stride_for(components_per_pixel).unwrap_or(0);
        let (spacing, origin, direction) = Self::default_geometry(size.len());
        Self::assemble(
            PixelBuffer::zeroed(id, n * stride),
            id,
            components_per_pixel,
            size.to_vec(),
            spacing,
            origin,
            direction,
        )
    }

    /// Build a scalar image from a typed buffer laid out in first-index-fastest
    /// order.
    ///
    /// Errors if `data.len()` does not equal the product of `size`.
    pub fn from_vec<T: Scalar>(size: &[usize], data: Vec<T>) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let (spacing, origin, direction) = Self::default_geometry(size.len());
        Self::assemble(
            T::into_buffer(data),
            T::PIXEL_ID,
            1,
            size.to_vec(),
            spacing,
            origin,
            direction,
        )
    }

    /// Build a vector image from an **interleaved** typed buffer: the
    /// `components_per_pixel` components of each pixel are adjacent, and pixels
    /// run in first-index-fastest order.
    ///
    /// The pixel type is `T`'s vector variant, so `from_vec_vector::<f32>(size,
    /// 1, data)` yields a [`PixelId::VectorFloat32`] image with one component
    /// per pixel — distinct from the [`PixelId::Float32`] image
    /// [`Image::from_vec`] would build from the same data, exactly as
    /// SimpleITK's `sitkVectorFloat32` is distinct from `sitkFloat32`.
    ///
    /// Errors if `data.len()` does not equal `Π size * components_per_pixel`,
    /// or if `components_per_pixel` is zero.
    pub fn from_vec_vector<T: Scalar>(
        size: &[usize],
        components_per_pixel: usize,
        data: Vec<T>,
    ) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let (spacing, origin, direction) = Self::default_geometry(size.len());
        Self::assemble(
            T::into_buffer(data),
            T::PIXEL_ID.vector_id(),
            components_per_pixel,
            size.to_vec(),
            spacing,
            origin,
            direction,
        )
    }

    /// Build a complex image from a typed buffer of one [`Complex<T>`] per
    /// pixel, laid out in first-index-fastest order.
    ///
    /// The pixel type is `T`'s complex variant, so `from_vec_complex::<f32>`
    /// yields a [`PixelId::ComplexFloat32`] image, whose
    /// [`Image::number_of_components_per_pixel`] is `1` and whose buffer holds
    /// `2 * Π size` interleaved `f32`.
    ///
    /// Errors with [`Error::BufferSizeMismatch`] — counted in *pixels*, since
    /// `data` is one element per pixel — if `data.len()` does not equal the
    /// product of `size`.
    pub fn from_vec_complex<T: Real>(size: &[usize], data: Vec<Complex<T>>) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let number_of_pixels: usize = size.iter().product();
        if data.len() != number_of_pixels {
            return Err(Error::BufferSizeMismatch {
                expected: number_of_pixels,
                actual: data.len(),
            });
        }
        let mut interleaved = Vec::with_capacity(data.len() * 2);
        for c in data {
            interleaved.push(c.re);
            interleaved.push(c.im);
        }
        let (spacing, origin, direction) = Self::default_geometry(size.len());
        Self::assemble(
            T::into_buffer(interleaved),
            T::COMPLEX_ID,
            1,
            size.to_vec(),
            spacing,
            origin,
            direction,
        )
    }

    /// Assemble a scalar image from parts, validating that geometry lengths
    /// agree with the buffer size. Used by IO where all fields are read from a
    /// file.
    pub fn from_parts(
        buffer: PixelBuffer,
        size: Vec<usize>,
        spacing: Vec<f64>,
        origin: Vec<f64>,
        direction: Vec<f64>,
    ) -> Result<Self> {
        let pixel_id = buffer.component_id();
        Self::assemble(buffer, pixel_id, 1, size, spacing, origin, direction)
    }

    /// Assemble a vector image from parts: an interleaved component buffer
    /// plus an explicit `components_per_pixel`, validating that geometry
    /// lengths agree with the buffer size. The vector counterpart of
    /// [`Image::from_parts`], which always assumes one component per pixel;
    /// used by IO formats that carry the component count as a separate
    /// on-disk field (MetaImage's `ElementNumberOfChannels`).
    pub fn from_parts_vector(
        buffer: PixelBuffer,
        components_per_pixel: usize,
        size: Vec<usize>,
        spacing: Vec<f64>,
        origin: Vec<f64>,
        direction: Vec<f64>,
    ) -> Result<Self> {
        let pixel_id = buffer.component_id().vector_id();
        Self::assemble(
            buffer,
            pixel_id,
            components_per_pixel,
            size,
            spacing,
            origin,
            direction,
        )
    }

    /// Interleave `images` — one scalar image per component — into a vector
    /// image. This is `itk::ComposeImageFilter`'s primitive.
    ///
    /// Every input must be scalar, of the same pixel type and the same size
    /// ("All input images are expected to have the same template parameters and
    /// have the same size and origin" — `ComposeImageFilter.yaml`'s
    /// detaileddescription). The output takes its geometry from `images[0]` and
    /// its pixel type from that component type's vector variant.
    ///
    /// A complex input is rejected along with a vector one: its buffer holds
    /// two components per pixel, and `interleave` reads one. (ITK's
    /// `ComposeImageFilter` *does* compose two real images into a complex one —
    /// itkComposeImageFilter.hxx:132-138 — but that is the separate output-type
    /// specialization behind `RealAndImaginaryToComplex`, not this vector path.)
    ///
    /// Errors on an empty `images` list.
    pub fn from_component_images(images: &[&Image]) -> Result<Self> {
        let Some(first) = images.first() else {
            return Err(Error::EmptyComponentImageList);
        };
        for img in images {
            if !img.pixel_id.is_scalar() {
                return Err(Error::RequiresScalarPixelType(img.pixel_id));
            }
            if img.pixel_id != first.pixel_id {
                return Err(Error::PixelTypeMismatch {
                    expected: first.pixel_id,
                    requested: img.pixel_id,
                });
            }
            if img.size != first.size {
                return Err(Error::GeometryMismatch {
                    dimension: first.dimension(),
                });
            }
        }

        fn interleave<T: Scalar>(images: &[&Image]) -> Result<PixelBuffer> {
            let pixels = images[0].number_of_pixels();
            let slices: Vec<&[T]> = images
                .iter()
                .map(|img| img.scalar_slice::<T>())
                .collect::<Result<_>>()?;
            let mut out = Vec::with_capacity(pixels * slices.len());
            for p in 0..pixels {
                for s in &slices {
                    out.push(s[p]);
                }
            }
            Ok(T::into_buffer(out))
        }

        let buffer = crate::dispatch_scalar!(first.pixel_id, interleave, images)?;
        Self::assemble(
            buffer,
            first.pixel_id.vector_id(),
            images.len(),
            first.size.clone(),
            first.spacing.clone(),
            first.origin.clone(),
            first.direction.clone(),
        )
    }

    /// De-interleave component `index` of a vector image into a scalar image.
    /// This is `itk::VectorIndexSelectionCastImageFilter`'s primitive, before
    /// that filter's output cast.
    ///
    /// The output's pixel type is `pixel_id().component_id()` and it inherits
    /// this image's geometry. Errors with [`Error::RequiresVectorPixelType`] on
    /// a scalar image and [`Error::ComponentIndexOutOfRange`] on an `index >=`
    /// [`Image::number_of_components_per_pixel`].
    pub fn extract_component(&self, index: usize) -> Result<Image> {
        if !self.pixel_id.is_vector() {
            return Err(Error::RequiresVectorPixelType(self.pixel_id));
        }
        if index >= self.buffer_stride {
            return Err(Error::ComponentIndexOutOfRange {
                index,
                components_per_pixel: self.buffer_stride,
            });
        }

        fn take<T: Scalar>(img: &Image, index: usize) -> Result<PixelBuffer> {
            let all = img.component_slice::<T>()?;
            let stride = img.buffer_stride;
            Ok(T::into_buffer(
                all.iter().skip(index).step_by(stride).copied().collect(),
            ))
        }

        let buffer = crate::dispatch_scalar!(self.pixel_id, take, self, index)?;
        Self::assemble(
            buffer,
            self.pixel_id.component_id(),
            1,
            self.size.clone(),
            self.spacing.clone(),
            self.origin.clone(),
            self.direction.clone(),
        )
    }

    /// Gather output pixels from this image's **native** buffer.
    ///
    /// Output pixel `i` is a bit-exact copy of this image's pixel at linear
    /// source index `sources[i]`, or — when `sources[i]` is `None` — a constant
    /// fill of `constant` quantized to the pixel type (`T::from_f64`). The copy
    /// runs on the stored component buffer and never widens to `f64`, so it is
    /// exact for every pixel type, including the `UInt64`/`Int64` magnitudes
    /// above `2^53` that an `as f64` round-trip (`to_f64_vec` → `from_f64`)
    /// would silently round.
    ///
    /// This is the native primitive behind every pure pixel-movement filter —
    /// flip, permute, crop, extract, slice, shrink-subsample, cyclic shift. Each
    /// computes a per-output source index and calls this, exactly as ITK moves
    /// those pixels through `ImageAlgorithm::Copy` / a native `static_cast`
    /// rather than a widened accumulator.
    ///
    /// A vector or complex image copies whole pixels: all
    /// [`buffer_stride()`](Image::buffer_stride) interleaved components of the
    /// source pixel move together, and a `None` slot fills every component with
    /// the quantized `constant`.
    ///
    /// The output inherits this image's geometry when `out_size` has the same
    /// dimension as this image, and default geometry (unit spacing, zero origin,
    /// identity direction) otherwise. Either way the caller overrides
    /// spacing/origin/direction as its grid transform requires — the geometry a
    /// pixel-movement op needs is never the input's unchanged.
    ///
    /// Errors with [`Error::BufferSizeMismatch`] if `sources.len()` is not the
    /// product of `out_size`, and with [`Error::GatherSourceOutOfBounds`] if any
    /// `Some(idx)` is `>=` [`Image::number_of_pixels`].
    pub fn gather(
        &self,
        out_size: &[usize],
        sources: &[Option<usize>],
        constant: f64,
    ) -> Result<Image> {
        let out_count: usize = out_size.iter().product();
        if sources.len() != out_count {
            return Err(Error::BufferSizeMismatch {
                expected: out_count,
                actual: sources.len(),
            });
        }
        let n = self.number_of_pixels();
        if let Some(&idx) = sources.iter().flatten().find(|&&idx| idx >= n) {
            return Err(Error::GatherSourceOutOfBounds {
                index: idx,
                number_of_pixels: n,
            });
        }

        fn build<T: Scalar>(img: &Image, sources: &[Option<usize>], constant: f64) -> PixelBuffer {
            let stride = img.buffer_stride;
            let src = img
                .buffer
                .as_slice::<T>()
                .expect("dispatch_scalar selects the buffer's own component type");
            let fill = T::from_f64(constant);
            let mut out = Vec::with_capacity(sources.len() * stride);
            for &s in sources {
                match s {
                    Some(idx) => {
                        let start = idx * stride;
                        out.extend_from_slice(&src[start..start + stride]);
                    }
                    None => {
                        for _ in 0..stride {
                            out.push(fill);
                        }
                    }
                }
            }
            T::into_buffer(out)
        }

        let buffer = crate::dispatch_scalar!(self.pixel_id, build, self, sources, constant);
        let (spacing, origin, direction) = if out_size.len() == self.dimension() {
            (
                self.spacing.clone(),
                self.origin.clone(),
                self.direction.clone(),
            )
        } else {
            Self::default_geometry(out_size.len())
        };
        Self::assemble(
            buffer,
            self.pixel_id,
            self.number_of_components_per_pixel(),
            out_size.to_vec(),
            spacing,
            origin,
            direction,
        )
    }

    /// Number of spatial dimensions.
    pub fn dimension(&self) -> usize {
        self.size.len()
    }

    /// Size along each axis, in SimpleITK order.
    pub fn size(&self) -> &[usize] {
        &self.size
    }

    /// The runtime pixel-type tag. A `Vector*` variant for a multi-component
    /// image, even when it carries a single component per pixel.
    pub fn pixel_id(&self) -> PixelId {
        self.pixel_id
    }

    /// Components per pixel — SimpleITK's `GetNumberOfComponentsPerPixel()`.
    ///
    /// `1` for a scalar image, `1` for a **complex** image, and the vector
    /// length for a vector image. Derived, not stored: upstream returns the ITK
    /// vector length only `if constexpr (IsVector<TImageType>::Value)` and
    /// otherwise `1` (sitkPimpleImageBase.hxx:202-209), and `IsVector` is not
    /// specialized for `BasicPixelID<std::complex<T>>`.
    ///
    /// A complex image's buffer nonetheless holds two components per pixel;
    /// that count is [`Image::buffer_stride`].
    pub fn number_of_components_per_pixel(&self) -> usize {
        if self.pixel_id.is_vector() {
            self.buffer_stride
        } else {
            1
        }
    }

    /// Buffer components one pixel occupies: `1` scalar, `2` complex,
    /// [`Image::number_of_components_per_pixel`] vector.
    ///
    /// This is the multiplier relating [`Image::number_of_pixels`] to
    /// `buffer().len()`, and the stride of [`Image::component_slice`].
    pub fn buffer_stride(&self) -> usize {
        self.buffer_stride
    }

    /// Total number of pixels — the product of [`Image::size`], *not* the
    /// buffer length (which is this times [`Image::buffer_stride`]).
    pub fn number_of_pixels(&self) -> usize {
        self.size.iter().product()
    }

    /// Physical spacing between pixels along each axis.
    pub fn spacing(&self) -> &[f64] {
        &self.spacing
    }

    /// Physical coordinate of the first pixel.
    pub fn origin(&self) -> &[f64] {
        &self.origin
    }

    /// Row-major `dimension x dimension` direction cosine matrix.
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// Set the spacing; errors on wrong length or a non-positive component.
    pub fn set_spacing(&mut self, spacing: &[f64]) -> Result<()> {
        if spacing.len() != self.dimension() {
            return Err(Error::GeometryMismatch {
                dimension: self.dimension(),
            });
        }
        if spacing.iter().any(|&s| s <= 0.0 || s.is_nan()) {
            return Err(Error::NonPositiveSpacing);
        }
        self.spacing = spacing.to_vec();
        Ok(())
    }

    /// Set the origin; errors on wrong length.
    pub fn set_origin(&mut self, origin: &[f64]) -> Result<()> {
        if origin.len() != self.dimension() {
            return Err(Error::GeometryMismatch {
                dimension: self.dimension(),
            });
        }
        self.origin = origin.to_vec();
        Ok(())
    }

    /// Set the direction cosine matrix (row-major); errors on wrong length.
    pub fn set_direction(&mut self, direction: &[f64]) -> Result<()> {
        let dim = self.dimension();
        if direction.len() != dim * dim {
            return Err(Error::GeometryMismatch { dimension: dim });
        }
        self.direction = direction.to_vec();
        Ok(())
    }

    /// Copy spacing, origin, and direction from another image of equal dimension.
    /// Used by filters that preserve input geometry.
    ///
    /// This is SimpleITK's `CopyInformation` (sitkImage.h:386-395,
    /// sitkImage.cxx:349-357), which likewise sets only the origin, spacing and
    /// direction: "The meta-data dictionary is **not** copied."
    pub fn copy_geometry_from(&mut self, other: &Image) {
        debug_assert_eq!(self.dimension(), other.dimension());
        self.spacing = other.spacing.clone();
        self.origin = other.origin.clone();
        self.direction = other.direction.clone();
    }

    /// Borrow the type-erased component buffer (used by dispatch macros and IO).
    ///
    /// For a vector image this is the interleaved component storage; consult
    /// [`Image::number_of_components_per_pixel`] before interpreting it.
    pub fn buffer(&self) -> &PixelBuffer {
        &self.buffer
    }

    /// Borrow the type-erased component buffer mutably.
    ///
    /// Growing or shrinking the buffer would break the [`Image`] invariant that
    /// ties its length to `number_of_pixels * components_per_pixel`; only
    /// element-wise mutation is sound.
    pub fn buffer_mut(&mut self) -> &mut PixelBuffer {
        &mut self.buffer
    }

    /// The scalar guard: `Ok(())` when [`PixelId::is_scalar`], and
    /// [`Error::RequiresScalarPixelType`] otherwise.
    ///
    /// Every scalar-typed read of an `Image` goes through this, so no consumer
    /// can reach an interleaved buffer while believing it holds one value per
    /// pixel. The test is a **whitelist** on the scalar category, not a
    /// blacklist on the vector one: a complex image's buffer is `2N` long, and
    /// `!is_vector()` would have admitted it.
    fn require_scalar(&self) -> Result<()> {
        if !self.pixel_id.is_scalar() {
            return Err(Error::RequiresScalarPixelType(self.pixel_id));
        }
        Ok(())
    }

    /// Borrow a scalar image's buffer as a concrete `&[T]`, one element per
    /// pixel.
    ///
    /// Errors with [`Error::RequiresScalarPixelType`] on a vector or complex
    /// image and with [`Error::PixelTypeMismatch`] if `T` is not the image's
    /// pixel type.
    pub fn scalar_slice<T: Scalar>(&self) -> Result<&[T]> {
        self.require_scalar()?;
        self.buffer.as_slice::<T>().ok_or(Error::PixelTypeMismatch {
            expected: self.pixel_id,
            requested: T::PIXEL_ID,
        })
    }

    /// Borrow this image together with proof that it is scalar with pixel type
    /// `T`, as a [`ScalarView`].
    ///
    /// Errors exactly as [`Image::scalar_slice`]. This is the only way to build
    /// a `ScalarView`, which is why an API that takes one — such as
    /// [`BoundaryCondition::get_pixel`](crate::BoundaryCondition::get_pixel) —
    /// can read pixels infallibly without a runtime type or component check.
    pub fn scalar_view<T: Scalar>(&self) -> Result<ScalarView<'_, T>> {
        Ok(ScalarView {
            image: self,
            pixels: self.scalar_slice::<T>()?,
        })
    }

    /// Borrow a scalar image's backing `Vec<T>` mutably; errors on a vector
    /// image or on pixel-type mismatch.
    pub fn scalar_vec_mut<T: Scalar>(&mut self) -> Result<&mut Vec<T>> {
        self.require_scalar()?;
        let id = self.pixel_id;
        self.buffer
            .as_mut_vec::<T>()
            .ok_or(Error::PixelTypeMismatch {
                expected: id,
                requested: T::PIXEL_ID,
            })
    }

    /// Borrow the whole interleaved component buffer as `&[T]`, for every pixel
    /// category — `T` is the *component* type. SimpleITK's `GetBufferAsFloat()`
    /// and friends (sitkPimpleImageBase.hxx:826-848).
    ///
    /// Length is `number_of_pixels() * buffer_stride()`. This is the accessor
    /// vector filters use; scalar consumers want [`Image::scalar_slice`], which
    /// refuses non-scalar images, and complex consumers want
    /// [`Image::complex_components`], which refuses non-complex ones.
    pub fn component_slice<T: Scalar>(&self) -> Result<&[T]> {
        self.buffer.as_slice::<T>().ok_or(Error::PixelTypeMismatch {
            expected: self.pixel_id.component_id(),
            requested: T::PIXEL_ID,
        })
    }

    /// Borrow the whole interleaved component buffer mutably as `&mut Vec<T>`.
    /// The counterpart of [`Image::component_slice`].
    pub fn component_vec_mut<T: Scalar>(&mut self) -> Result<&mut Vec<T>> {
        let expected = self.pixel_id.component_id();
        self.buffer
            .as_mut_vec::<T>()
            .ok_or(Error::PixelTypeMismatch {
                expected,
                requested: T::PIXEL_ID,
            })
    }

    /// Copy a scalar image's buffer into an `f64` vector regardless of stored
    /// pixel type, one element per pixel. A typed accessor, not an algorithm —
    /// filters and resampling both widen to `f64` to compute uniformly.
    ///
    /// Errors with [`Error::RequiresScalarPixelType`] on a vector or complex
    /// image. Every caller of this function indexes the result by pixel, and a
    /// non-scalar image's buffer is `buffer_stride()` values per pixel;
    /// returning it here would silently misalign every one of them. Those
    /// callers want [`Image::components_to_f64_vec`].
    ///
    /// Together with [`Image::scalar_slice`] this is the whole scalar read
    /// surface of `Image`, so a filter cannot reach pixel data without passing
    /// the guard.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        self.require_scalar()?;
        Ok(self.buffer.to_f64_vec())
    }

    /// Copy the interleaved component buffer into an `f64` vector, for every
    /// pixel category. Length is `number_of_pixels() * buffer_stride()`.
    pub fn components_to_f64_vec(&self) -> Vec<f64> {
        self.buffer.to_f64_vec()
    }

    /// Linear buffer offset of a multi-index (first index fastest). Does not
    /// bounds-check against `size`.
    ///
    /// This is a *pixel* offset. The components of that pixel start at
    /// `linear_index(index) * buffer_stride()`; see [`Image::component_index`].
    pub fn linear_index(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.dimension());
        let mut offset = 0usize;
        let mut stride = 1usize;
        for (&idx, &sz) in index.iter().zip(self.size.iter()) {
            offset += idx * stride;
            stride *= sz;
        }
        offset
    }

    /// Offset into the interleaved component buffer of `component` of the pixel
    /// at `index`. Does not bounds-check either argument.
    pub fn component_index(&self, index: &[usize], component: usize) -> usize {
        self.linear_index(index) * self.buffer_stride + component
    }

    /// The single bounds-checking seam for the pixel accessors: the offset of
    /// the pixel at `index`'s first component in the interleaved buffer.
    ///
    /// Every `get_*`/`set_*` pixel accessor reaches its buffer through this
    /// function, so none of them can read or write a pixel other than the one
    /// `index` names. [`Image::linear_index`] and [`Image::component_index`]
    /// stay unchecked — they are the loop primitives filters use over indices
    /// they have already constrained, and both say so.
    ///
    /// The two rejections mirror upstream exactly:
    ///
    /// - `index.len() < dimension()` — `sitkSTLVectorToITK`
    ///   (sitkTemplateFunctions.h:100-105). A **longer** index is accepted and
    ///   its extra elements ignored, as `sitkImage.h:499-501` promises
    ///   ("additional elements will be ignored").
    /// - any `index[d] >= size[d]` — `PimpleImage::GetIndex`
    ///   (sitkPimpleImageBase.hxx:788-797), whose `IsInside` test against the
    ///   largest possible region throws "index out of bounds". SimpleITK's
    ///   indices are `uint32_t`, so its lower bound is met by the type; here
    ///   `usize` does the same.
    fn checked_pixel_start(&self, index: &[usize]) -> Result<usize> {
        let dim = self.dimension();
        if index.len() < dim {
            return Err(Error::IndexDimensionMismatch {
                dimension: dim,
                actual: index.len(),
            });
        }
        let mut offset = 0usize;
        let mut stride = 1usize;
        for d in 0..dim {
            if index[d] >= self.size[d] {
                return Err(Error::IndexOutOfBounds {
                    index: index[..dim].to_vec(),
                    size: self.size.clone(),
                });
            }
            offset += index[d] * stride;
            stride *= self.size[d];
        }
        Ok(offset * self.buffer_stride)
    }

    /// The components of the pixel at `index`, as a `&[T]` of length
    /// [`Image::buffer_stride`] — SimpleITK's `GetPixelAsVector*`.
    ///
    /// Works for scalar images too, where the slice has length 1, and for
    /// complex images, where it is `[re, im]`. The length is the *stride*, not
    /// [`Image::number_of_components_per_pixel`]; the two differ only for a
    /// complex image, and returning that image's single "component" would mean
    /// handing back half its pixel.
    ///
    /// Errors on component-type mismatch, on an `index` shorter than
    /// [`Image::dimension`], and on an out-of-bounds `index` — the private
    /// `checked_pixel_start` seam every pixel accessor shares.
    ///
    /// The guards run in upstream's order: `InternalGetPixelAs`
    /// (sitkPimpleImageBase.hxx:800-823) selects its branch on the pixel type
    /// first and only then calls `GetIndex`, so a wrong `T` is reported even
    /// when `index` is also out of bounds.
    ///
    /// # Divergence
    ///
    /// SimpleITK's `GetPixelAsVectorFloat32` throws on a complex image — its
    /// `InternalGetPixelAs` is gated on `IsVector<ImageType>::Value`
    /// (sitkPimpleImageBase.hxx:813). One uniform "give me this pixel's
    /// components" rule is preferred here over a fourth guard;
    /// [`Image::get_complex`] is the typed accessor.
    pub fn get_vector<T: Scalar>(&self, index: &[usize]) -> Result<&[T]> {
        let all = self.component_slice::<T>()?;
        let start = self.checked_pixel_start(index)?;
        Ok(&all[start..start + self.buffer_stride])
    }

    /// Overwrite the components of the pixel at `index` — SimpleITK's
    /// `SetPixelAsVector*`.
    ///
    /// Errors as [`Image::get_vector`] does, and with
    /// [`Error::InvalidComponentCount`] if `values.len()` is not
    /// [`Image::buffer_stride`]. Same divergence note.
    ///
    /// The length check comes last, as upstream's does: `InternalSetPixelAs`
    /// (sitkPimpleImageBase.hxx:867-878) fetches `GetPixel(GetIndex(idx))`
    /// before comparing `px.GetSize()` against `v.size()`.
    pub fn set_vector<T: Scalar>(&mut self, index: &[usize], values: &[T]) -> Result<()> {
        self.component_slice::<T>()?;
        let start = self.checked_pixel_start(index)?;
        let stride = self.buffer_stride;
        if values.len() != stride {
            return Err(Error::InvalidComponentCount {
                pixel_id: self.pixel_id,
                components_per_pixel: values.len(),
            });
        }
        let all = self.component_vec_mut::<T>()?;
        all[start..start + stride].copy_from_slice(values);
        Ok(())
    }

    /// The complex guard: `Ok(())` when [`PixelId::is_complex`], and
    /// [`Error::RequiresComplexPixelType`] otherwise. A whitelist, like
    /// `require_scalar`.
    fn require_complex(&self) -> Result<()> {
        if !self.pixel_id.is_complex() {
            return Err(Error::RequiresComplexPixelType(self.pixel_id));
        }
        Ok(())
    }

    /// The complex pixel at `index` — SimpleITK's
    /// `GetPixelAsComplexFloat32`/`64` (sitkImage.cxx:596-608).
    ///
    /// Errors with [`Error::RequiresComplexPixelType`] on a non-complex image,
    /// [`Error::PixelTypeMismatch`] if `T` is not the component type, and
    /// [`Error::IndexOutOfBounds`] / [`Error::IndexDimensionMismatch`] on a bad
    /// `index`. Guard order is upstream's: pixel type before index.
    pub fn get_complex<T: Real>(&self, index: &[usize]) -> Result<Complex<T>> {
        let all = self.complex_components::<T>()?;
        let start = self.checked_pixel_start(index)?;
        Ok(Complex::new(all[start], all[start + 1]))
    }

    /// Overwrite the complex pixel at `index` — SimpleITK's
    /// `SetPixelAsComplexFloat32`/`64`.
    ///
    /// Errors exactly as [`Image::get_complex`].
    pub fn set_complex<T: Real>(&mut self, index: &[usize], value: Complex<T>) -> Result<()> {
        self.complex_components::<T>()?;
        let start = self.checked_pixel_start(index)?;
        let all = self.complex_components_mut::<T>()?;
        all[start] = value.re;
        all[start + 1] = value.im;
        Ok(())
    }

    /// A complex image's interleaved `re, im, re, im, ...` buffer, of length
    /// `2 * number_of_pixels()`.
    ///
    /// The exact analogue of `GetBufferAsFloat()` on a `sitkComplexFloat32`
    /// image, which upstream produces by `reinterpret_cast`ing the
    /// `std::complex<float>` buffer (sitkPimpleImageBase.hxx:838-842):
    /// "Vector and Complex pixel types are both accessed via the appropriate
    /// component type method" (sitkImage.h:622-623).
    ///
    /// Errors with [`Error::RequiresComplexPixelType`] on a non-complex image —
    /// unlike [`Image::component_slice`], which serves every category and says
    /// so in its name.
    pub fn complex_components<T: Real>(&self) -> Result<&[T]> {
        self.require_complex()?;
        self.component_slice::<T>()
    }

    /// The mutable counterpart of [`Image::complex_components`]. Growing or
    /// shrinking the returned `Vec` would break the [`Image`] invariant that
    /// ties its length to `2 * number_of_pixels()`.
    pub fn complex_components_mut<T: Real>(&mut self) -> Result<&mut Vec<T>> {
        self.require_complex()?;
        self.component_vec_mut::<T>()
    }

    /// Map a continuous index to a physical point — ITK's
    /// `TransformContinuousIndexToPhysicalPoint` (itkImageBase.h:558-572).
    ///
    /// `p[r] = (Σ_c IndexToPhysicalPoint[r][c] · index[c]) + origin[r]`, with the
    /// origin added **last** (the continuous-method fold) and
    /// `IndexToPhysicalPoint = Direction · diag(spacing)` built once by
    /// [`coord::index_to_physical_matrix`](crate::coord). One implementation,
    /// shared with every consumer — see [`crate::coord`].
    pub fn continuous_index_to_physical_point(&self, index: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        debug_assert_eq!(index.len(), dim);
        let i2p = coord::index_to_physical_matrix(&self.direction, &self.spacing, dim);
        coord::continuous_index_to_physical_point(&i2p, &self.origin, index, dim)
    }

    /// Map a physical point to a continuous index — ITK's
    /// `TransformPhysicalPointToContinuousIndex` (itkImageBase.h:517-532):
    /// `cindex = PhysicalPointToIndex · (p − origin)`, where
    /// `PhysicalPointToIndex = inverse(Direction · diag(spacing))` — the inverse
    /// of the **composed** matrix, not the direction alone. This is what makes a
    /// diagonal geometry reciprocal-multiply (`(1/spacing)·d`) exactly as ITK
    /// does, rather than dividing (`d/spacing`); the two differ in the last bit
    /// and the difference flips a discrete index at the boundary.
    ///
    /// Errors if the composed matrix is singular.
    pub fn physical_point_to_continuous_index(&self, point: &[f64]) -> Result<Vec<f64>> {
        let dim = self.dimension();
        debug_assert_eq!(point.len(), dim);
        let p2i = coord::physical_to_index_matrix(&self.direction, &self.spacing, dim)
            .ok_or(Error::SingularDirection)?;
        Ok(coord::physical_point_to_continuous_index(
            &p2i,
            &self.origin,
            point,
            dim,
        ))
    }

    /// Map an integer index to a physical point — SimpleITK's
    /// `TransformIndexToPhysicalPoint` (sitkImage.h:291, sitkImage.cxx:420-425),
    /// ITK's integer method (itkImageBase.h:592-604).
    ///
    /// `p[r] = origin[r]; p[r] += IndexToPhysicalPoint[r][c] · index[c]` — the
    /// origin is the initial accumulator term (origin **first**), which is ITK's
    /// integer fold and differs from the continuous method
    /// ([`Image::continuous_index_to_physical_point`], origin-last) at large
    /// origins. Both share the one `IndexToPhysicalPoint` matrix.
    pub fn transform_index_to_physical_point(&self, index: &[i64]) -> Vec<f64> {
        let dim = self.dimension();
        debug_assert_eq!(index.len(), dim);
        let i2p = coord::index_to_physical_matrix(&self.direction, &self.spacing, dim);
        coord::index_to_physical_point(&i2p, &self.origin, index, dim)
    }

    /// Map a physical point to the integer index of the pixel containing it —
    /// SimpleITK's `TransformPhysicalPointToIndex` (sitkImage.h:295,
    /// sitkImage.cxx:412-417), ITK `itkImageBase.h:465-479`.
    ///
    /// [`Image::physical_point_to_continuous_index`] rounded with
    /// [`coord::round_half_integer_up`](crate::coord) =
    /// `Math::RoundHalfIntegerUp` (half toward +∞, `floor(x+0.5)`). ITK's
    /// `TransformPhysicalPointToIndex` and `TransformPhysicalPointToContinuousIndex`
    /// read the same `m_PhysicalPointToIndex` matrix, so rounding the continuous
    /// index reproduces it exactly.
    ///
    /// Errors with [`Error::SingularDirection`] if the composed matrix cannot be
    /// inverted. (ITK cannot reach that state here: `SetDirection` refuses a
    /// singular matrix, so its precomputed inverse always exists.)
    ///
    /// # Divergence
    ///
    /// ITK's cast to `IndexValueType` is C++-undefined when the rounded value
    /// leaves the integer's range — `RoundHalfIntegerUp`'s own doc warns the
    /// argument's magnitude must stay below `max()/2`. Rust's `as` saturates,
    /// which is defined; in-range values agree exactly.
    pub fn transform_physical_point_to_index(&self, point: &[f64]) -> Result<Vec<i64>> {
        let continuous = self.physical_point_to_continuous_index(point)?;
        Ok(continuous
            .into_iter()
            .map(coord::round_half_integer_up)
            .collect())
    }

    /// Whether the pixels of `self` and `other` at the same index occupy the
    /// same physical space — SimpleITK's `IsCongruentImageGeometry`
    /// (sitkImage.h:347, sitkImage.cxx:233-244), which delegates to
    /// `itk::ImageBase::IsCongruentImageGeometry` (itkImageBase.hxx:390-406).
    ///
    /// Origin, spacing and direction are compared element-wise with
    /// `vnl_vector::is_equal` / `vnl_matrix::is_equal` semantics
    /// (vnl_vector.hxx:793-805, vnl_matrix.hxx:1134-1148): a pair is equal iff
    /// `!(|a - b| <= tol)` is false, so the tolerance is **inclusive** and any
    /// `NaN` makes the images unequal. Sizes are not compared; use
    /// [`Image::is_same_image_geometry_as`] for that.
    ///
    /// Images of different [`Image::dimension`] are never congruent
    /// (sitkImage.cxx:236-239).
    ///
    /// # Upstream quirk: the tolerance is asymmetric
    ///
    /// The origin and spacing tolerance is scaled by `self`'s **first-dimension
    /// spacing only** — `coordinateTol = |coordinateTolerance * GetSpacing()[0]|`,
    /// with the inline comment "use first dimension spacing"
    /// (itkImageBase.hxx:400-401). So the relation is not symmetric when the two
    /// images differ in `spacing[0]`, and axes 1.. are judged against axis 0's
    /// scale. The direction tolerance is not scaled at all. Reproduced, and
    /// pinned by `congruent_geometry_tolerance_is_asymmetric`.
    pub fn is_congruent_image_geometry(
        &self,
        other: &Image,
        coordinate_tolerance: f64,
        direction_tolerance: f64,
    ) -> bool {
        fn all_within(a: &[f64], b: &[f64], tol: f64) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
        }

        if self.dimension() != other.dimension() {
            return false;
        }
        let coordinate_tol = (coordinate_tolerance * self.spacing[0]).abs();
        all_within(&self.origin, &other.origin, coordinate_tol)
            && all_within(&self.spacing, &other.spacing, coordinate_tol)
            && all_within(&self.direction, &other.direction, direction_tolerance)
    }

    /// Whether `self` and `other` have the same grid in physical space —
    /// SimpleITK's `IsSameImageGeometryAs` (sitkImage.h:357,
    /// sitkImage.cxx:246-257).
    ///
    /// [`Image::is_congruent_image_geometry`] plus equality of the largest
    /// possible region (itkImageBase.hxx:410-417). Every image in this port has
    /// a zero start index — SimpleITK's do too, since `Image::Allocate` builds
    /// the region from a size alone — so the region test reduces to
    /// [`Image::size`] equality.
    ///
    /// Upstream's C++ defaults for the two tolerances are
    /// [`Image::DEFAULT_IMAGE_COORDINATE_TOLERANCE`] and
    /// [`Image::DEFAULT_IMAGE_DIRECTION_TOLERANCE`]; Rust has no default
    /// arguments, so they are passed explicitly.
    pub fn is_same_image_geometry_as(
        &self,
        other: &Image,
        coordinate_tolerance: f64,
        direction_tolerance: f64,
    ) -> bool {
        self.is_congruent_image_geometry(other, coordinate_tolerance, direction_tolerance)
            && self.size == other.size
    }

    /// Bytes per pixel component — SimpleITK's `GetSizeOfPixelComponent()`
    /// (sitkImage.h:253, sitkImage.cxx:162-217).
    ///
    /// # Divergence from upstream: complex reports the component, not the pixel
    ///
    /// Upstream documents "Returns the `sizeof` the pixel component type", and
    /// for the scalar and vector pixel types it does. For the two complex pixel
    /// types its `switch` returns `2 * sizeof(float)` / `2 * sizeof(double)`
    /// (sitkImage.cxx:206-212) — the size of the whole `std::complex` pixel, not
    /// of its component — and `sitkImageTests.cxx:1166` pins that wrong value.
    /// This port returns the documented value: `4` for `ComplexFloat32`, `8` for
    /// `ComplexFloat64`. See §3.20 of `doc/upstream-findings.md`. It is exactly
    /// [`PixelId::size_in_bytes`].
    pub fn size_of_pixel_component(&self) -> usize {
        self.pixel_id.size_in_bytes()
    }

    /// The pixel type as a human-readable string — SimpleITK's
    /// `GetPixelIDTypeAsString()` (sitkImage.h:216, sitkImage.cxx:220-224),
    /// which forwards to `GetPixelIDValueAsString`. See [`PixelId::as_str`].
    pub fn pixel_id_type_as_string(&self) -> &'static str {
        self.pixel_id.as_str()
    }

    /// The keys of the meta-data dictionary, in ascending byte order —
    /// SimpleITK's `GetMetaDataKeys()` (sitkImage.h:401-408,
    /// sitkImage.cxx:361-367). See the [`Image`] type docs for why that order is
    /// upstream's.
    pub fn meta_data_keys(&self) -> Vec<&str> {
        self.metadata.keys().map(String::as_str).collect()
    }

    /// Whether `key` is in the meta-data dictionary — SimpleITK's
    /// `HasMetaDataKey` (sitkImage.h:412, sitkImage.cxx:369-375).
    pub fn has_meta_data_key(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }

    /// The value stored under `key`, or `None` — SimpleITK's `GetMetaData`
    /// (sitkImage.h:421, sitkImage.cxx:377-392).
    ///
    /// # Divergence
    ///
    /// Upstream throws for an absent key: `GetMetaData` falls through to
    /// `mdd.Get(key)->Print(ss)`, and `MetaDataDictionary::Get` raises
    /// "Requesting invalid key ..." (itkMetaDataDictionary.cxx:150-158).
    /// `None` carries the same information without an error type, and
    /// [`Image::has_meta_data_key`] is the explicit pre-check upstream callers
    /// use.
    pub fn meta_data(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Create or replace the entry under `key` — SimpleITK's `SetMetaData`
    /// (sitkImage.h:427, sitkImage.cxx:393-401), which always encapsulates a
    /// `std::string`.
    pub fn set_meta_data(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_owned(), value.to_owned());
    }

    /// Remove the entry under `key`, reporting whether it was there —
    /// SimpleITK's `EraseMetaData` (sitkImage.h:432, sitkImage.cxx:402-409),
    /// which returns `itk::MetaDataDictionary::Erase`'s bool
    /// (itkMetaDataDictionary.cxx:180-197).
    pub fn erase_meta_data(&mut self, key: &str) -> bool {
        self.metadata.remove(key).is_some()
    }

    /// Reinterpret the first dimension as this image's pixel components —
    /// SimpleITK's `ToVectorImage` (sitkImage.h:455,
    /// sitkImageExplicit.cxx:113-133, sitkImage.hxx:104-162).
    ///
    /// The buffer is unchanged: ITK's `GetVectorImageFromScalarImage`
    /// (sitkImageConvert.hxx:127-177) hands the very same `PixelContainer` to
    /// the new `itk::VectorImage`, because a scalar image with its first index
    /// varying fastest already stores each output pixel's components adjacently.
    /// The new image has `size[0]` components per pixel; its size, spacing and
    /// origin drop element 0, and its direction is the trailing
    /// `(d-1) x (d-1)` submatrix.
    ///
    /// A vector image is returned unchanged (`ToVectorInternal`'s
    /// `if constexpr (IsVector<TImageType>::Value) return *this`,
    /// sitkImage.hxx:107-110) — with no direction check.
    ///
    /// # Errors
    ///
    /// - [`Error::CannotConvertToVectorImage`] for a **complex** image and for a
    ///   scalar image of fewer than three dimensions. Upstream's member-function
    ///   factory registers `ScalarPixelIDTypeList ++ VectorPixelIDTypeList` only
    ///   over dimensions `3..=SITK_MAX_DIMENSION` (sitkImageExplicit.cxx:
    ///   119-124), and `ScalarPixelIDTypeList` is `BasicPixelIDTypeList`, the ten
    ///   scalars — complex is not in it (sitkPixelIDTypeLists.h:40-57). Missing
    ///   registrations throw "Unable to convert an image with pixel type ... to a
    ///   vector image!".
    /// - [`Error::NonIdentityFirstDimensionDirection`] when the direction
    ///   matrix's first row or column differs from the identity's
    ///   (sitkImage.hxx:134-145). The comparison is exact, as upstream's is.
    /// - [`Error::NonTrivialFirstDimensionGeometry`] when `spacing[0] != 1.0`
    ///   or `origin[0] != 0.0` — this port's fix for ledger §3.21, below.
    ///
    /// # Divergences
    ///
    /// Upstream's `inPlace` flag decides whether the *receiver* is also rebound
    /// to the converted image; the converted image is returned either way. An
    /// owned Rust `Image` has no such aliasing, so this method always takes
    /// `&self` and the caller writes `img = img.to_vector_image()?` for the
    /// in-place effect. Upstream's `SITK_MAX_DIMENSION` ceiling (a build option,
    /// default 5 — `CMake/sitkMaxDimensionOption.cmake:6`) has no analogue here.
    ///
    /// The first axis becomes the vector's component axis, which has no spacing
    /// or origin of its own. Upstream silently **drops** `spacing[0]`/`origin[0]`
    /// there (and the meta-data dictionary too), so a scalar→vector→scalar round
    /// trip loses them without warning (ledger §3.21). This port instead
    /// **requires the first axis to be trivial** — `spacing[0] == 1.0`,
    /// `origin[0] == 0.0`, identity direction row and column — exactly the
    /// synthetic component axis [`Image::to_scalar_image`] produces, so nothing
    /// meaningful is ever dropped and the round trip is lossless. The meta-data
    /// dictionary **is** carried across, unlike upstream.
    pub fn to_vector_image(&self) -> Result<Image> {
        if self.pixel_id.is_vector() {
            return Ok(self.clone());
        }
        let dim = self.dimension();
        if !self.pixel_id.is_scalar() || dim < 3 {
            return Err(Error::CannotConvertToVectorImage {
                pixel_id: self.pixel_id,
                dimension: dim,
            });
        }

        // direction[i][0] and direction[0][i] must be the identity's.
        for i in 1..dim {
            if self.direction[i * dim] != 0.0 || self.direction[i] != 0.0 {
                return Err(Error::NonIdentityFirstDimensionDirection);
            }
        }
        if self.direction[0] != 1.0 {
            return Err(Error::NonIdentityFirstDimensionDirection);
        }

        // The component axis has no geometry slot; refuse to drop a non-trivial
        // one, mirroring the direction guard above (ledger §3.21).
        if self.spacing[0] != 1.0 || self.origin[0] != 0.0 {
            return Err(Error::NonTrivialFirstDimensionGeometry);
        }

        let out_dim = dim - 1;
        let mut direction = vec![0.0; out_dim * out_dim];
        for i in 0..out_dim {
            for j in 0..out_dim {
                direction[i * out_dim + j] = self.direction[(i + 1) * dim + (j + 1)];
            }
        }
        let mut out = Self::assemble(
            self.buffer.clone(),
            self.pixel_id.vector_id(),
            self.size[0],
            self.size[1..].to_vec(),
            self.spacing[1..].to_vec(),
            self.origin[1..].to_vec(),
            direction,
        )?;
        out.metadata = self.metadata.clone();
        Ok(out)
    }

    /// Reinterpret this vector image's pixel components as a new leading
    /// dimension — SimpleITK's `ToScalarImage` (sitkImage.h:476,
    /// sitkImageExplicit.cxx:135-160, sitkImage.hxx:165-205), the inverse of
    /// [`Image::to_vector_image`].
    ///
    /// The buffer is again shared unchanged
    /// (`GetScalarImageFromVectorImage`, sitkImageConvert.hxx:74-125). The new
    /// leading axis has `number_of_components_per_pixel()` pixels; upstream sets
    /// its spacing to `1.0`, its origin to `0.0`, and the new row and column of
    /// the direction matrix to the identity's (sitkImageConvert.hxx:96-124).
    ///
    /// A scalar image is returned unchanged (`ToScalarInternal`'s
    /// `if constexpr (IsBasic<TImageType>::Value) return *this`,
    /// sitkImage.hxx:169-172).
    ///
    /// # Errors
    ///
    /// [`Error::CannotConvertToScalarImage`] for a **complex** image. `IsBasic`
    /// is true for `itk::Image<std::complex<T>, N>` (sitkPixelIDTokens.h:39-49),
    /// so that `return *this` branch would take it — but upstream's factory
    /// registers only `ScalarPixelIDTypeList ++ VectorPixelIDTypeList`
    /// (sitkImageExplicit.cxx:143-148), neither of which holds a complex pixel
    /// id, so `HasMemberFunction` fails and it throws first. The whitelist here
    /// reproduces the reachable behavior.
    ///
    /// # Divergences
    ///
    /// `inPlace` and `SITK_MAX_DIMENSION` as in [`Image::to_vector_image`].
    /// Upstream additionally throws for a vector image *at* `SITK_MAX_DIMENSION`
    /// (its factory covers vectors only up to `SITK_MAX_DIMENSION - 1`); with no
    /// dimension ceiling here, that rejection has nothing to key on.
    ///
    /// The synthetic leading axis gets `spacing = 1.0`, `origin = 0.0` and an
    /// identity direction row/column — the trivial component axis that
    /// [`Image::to_vector_image`] requires, so the pair round-trips losslessly.
    /// The meta-data dictionary is carried across, unlike upstream (ledger §3.21).
    pub fn to_scalar_image(&self) -> Result<Image> {
        if self.pixel_id.is_scalar() {
            return Ok(self.clone());
        }
        let dim = self.dimension();
        if !self.pixel_id.is_vector() {
            return Err(Error::CannotConvertToScalarImage {
                pixel_id: self.pixel_id,
                dimension: dim,
            });
        }

        let out_dim = dim + 1;
        let mut size = Vec::with_capacity(out_dim);
        size.push(self.number_of_components_per_pixel());
        size.extend_from_slice(&self.size);

        let mut spacing = Vec::with_capacity(out_dim);
        spacing.push(1.0);
        spacing.extend_from_slice(&self.spacing);

        let mut origin = Vec::with_capacity(out_dim);
        origin.push(0.0);
        origin.extend_from_slice(&self.origin);

        let mut direction = matrix::identity(out_dim);
        for i in 0..dim {
            for j in 0..dim {
                direction[(i + 1) * out_dim + (j + 1)] = self.direction[i * dim + j];
            }
        }
        let mut out = Self::assemble(
            self.buffer.clone(),
            self.pixel_id.component_id(),
            1,
            size,
            spacing,
            origin,
            direction,
        )?;
        out.metadata = self.metadata.clone();
        Ok(out)
    }

    /// The scalar pixel at `index` — SimpleITK's `GetPixelAsInt8` and the nine
    /// other `GetPixelAs*` scalar overloads (sitkImage.h:494-513,
    /// sitkPimpleImageBase.hxx:452-501).
    ///
    /// `T` must be the image's pixel type **exactly**: upstream's
    /// `InternalGetPixelAs<TReturn>` takes its `GetPixel` branch only under
    /// `IsBasic<ImageType>::Value && std::is_same<ValuePixelType, TReturn>`
    /// (sitkPimpleImageBase.hxx:805-811) and otherwise throws "The image is of
    /// type: ... but the GetPixel access method does not match the type!". There
    /// is no conversion — `GetPixelAsInt8` on a `sitkFloat32` image throws.
    ///
    /// # Errors
    ///
    /// - [`Error::RequiresScalarPixelType`] on a vector or complex image. The
    ///   guard is [`Image::scalar_slice`]'s whitelist on [`PixelId::is_scalar`],
    ///   so a pixel category added later is rejected by default. A complex image
    ///   is rejected here as upstream rejects it — `ValuePixelType` is
    ///   `std::complex<float>`, never `float` — even though `IsBasic` holds for
    ///   it; [`Image::get_complex`] is its accessor.
    /// - [`Error::PixelTypeMismatch`] when `T` is not the image's pixel type.
    /// - [`Error::IndexDimensionMismatch`] / [`Error::IndexOutOfBounds`] on a bad
    ///   `index`, checked after the pixel type, as upstream checks it.
    pub fn get_pixel_as<T: Scalar>(&self, index: &[usize]) -> Result<T> {
        let pixels = self.scalar_slice::<T>()?;
        let offset = self.checked_pixel_start(index)?;
        Ok(pixels[offset])
    }

    /// Overwrite the scalar pixel at `index` — SimpleITK's `SetPixelAsInt8` and
    /// the nine other scalar overloads (sitkImage.h:557-576,
    /// sitkPimpleImageBase.hxx:674-720).
    ///
    /// Errors exactly as [`Image::get_pixel_as`]; `InternalSetPixelAs`
    /// (sitkPimpleImageBase.hxx:855-865) applies the same `is_same` gate and
    /// throws "... does not match the type of SetPixel method called."
    pub fn set_pixel_as<T: Scalar>(&mut self, index: &[usize], value: T) -> Result<()> {
        self.scalar_slice::<T>()?;
        let offset = self.checked_pixel_start(index)?;
        self.scalar_vec_mut::<T>()?[offset] = value;
        Ok(())
    }
}

impl Image {
    /// SimpleITK's `Image::DefaultImageCoordinateTolerance` (sitkImage.h:694),
    /// the default `coordinateTolerance` of [`Image::is_same_image_geometry_as`].
    pub const DEFAULT_IMAGE_COORDINATE_TOLERANCE: f64 = 1e-6;

    /// SimpleITK's `Image::DefaultImageDirectionTolerance` (sitkImage.h:695),
    /// the default `directionTolerance` of [`Image::is_same_image_geometry_as`].
    pub const DEFAULT_IMAGE_DIRECTION_TOLERANCE: f64 = 1e-6;
}

/// A human-readable dump of the image's pixel type, geometry and meta-data —
/// the role SimpleITK's `ToString()` (sitkImage.h:435, sitkImage.cxx:226-231)
/// plays, reachable as `image.to_string()`.
///
/// # Divergence
///
/// Upstream's `ToString()` is `itk::Image::Print(os)`, i.e. the `PrintSelf`
/// chain `Object` -> `DataObject` -> `ImageBase` -> `Image`
/// (itkImageBase.hxx:501-531, itkImage.hxx:146-155, itkDataObject.cxx:258-283).
/// That text is not reproducible outside ITK's object system, and would not be
/// worth reproducing if it were: it embeds the RTTI type name, the reference
/// count, the pipeline and update `MTime`s, a `RealTimeStamp`, the observer
/// list, and the raw pointer addresses of the source `ProcessObject` and the
/// `PixelContainer`. The fields below are the subset that names the image
/// rather than the process that made it, ordered as `ImageBase::PrintSelf`
/// orders them.
impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Image ({})", self.pixel_id_type_as_string())?;
        writeln!(f, "  Dimension: {}", self.dimension())?;
        writeln!(
            f,
            "  NumberOfComponentsPerPixel: {}",
            self.number_of_components_per_pixel()
        )?;
        writeln!(f, "  LargestPossibleRegion:")?;
        writeln!(f, "    Index: {:?}", vec![0usize; self.dimension()])?;
        writeln!(f, "    Size: {:?}", self.size)?;
        writeln!(f, "  Spacing: {:?}", self.spacing)?;
        writeln!(f, "  Origin: {:?}", self.origin)?;
        writeln!(f, "  Direction:")?;
        for row in self.direction.chunks(self.dimension()) {
            writeln!(f, "    {row:?}")?;
        }
        writeln!(f, "  MetaDataDictionary:")?;
        if self.metadata.is_empty() {
            writeln!(f, "    (empty)")?;
        } else {
            for (key, value) in &self.metadata {
                writeln!(f, "    {key}: {value}")?;
            }
        }
        Ok(())
    }
}

/// Dispatch a generic function on an image's runtime pixel type, recovering the
/// static type of its *components*.
///
/// `$func` names a generic `fn f<T: Scalar>(..) -> R` in scope (a bare
/// identifier, so a turbofish can be appended); the same `R` is returned for
/// every arm. The first argument is the [`PixelId`] to switch on.
///
/// A vector or complex [`PixelId`] selects the same `T` as its component's
/// scalar id, so `$func` sees the type the buffer actually stores. That is not
/// a licence to read the buffer as if it were scalar: `$func` reaches the
/// pixels through [`Image::scalar_slice`], which rejects every non-scalar image
/// with [`crate::Error::RequiresScalarPixelType`], or through the explicitly
/// component-aware [`Image::component_slice`].
///
/// ```
/// use sitk_core::{Image, Scalar, dispatch_scalar};
///
/// fn count<T: Scalar>(img: &Image) -> usize { img.number_of_pixels() }
///
/// let img = Image::from_vec(&[2, 3], vec![0.0f64; 6]).unwrap();
/// let n = dispatch_scalar!(img.pixel_id(), count, &img);
/// assert_eq!(n, 6);
/// ```
#[macro_export]
macro_rules! dispatch_scalar {
    // The one type table, shared by every public form below. `$extra` is the
    // tail of the turbofish, so a caller whose `$func` carries further generic
    // parameters (an inferred closure type, say) can leave them to inference
    // instead of forcing a second copy of this table.
    (@table $id:expr, $func:ident, [$($extra:tt)*] $(, $arg:expr)* $(,)?) => {{
        match $id {
            $crate::PixelId::UInt8
            | $crate::PixelId::VectorUInt8 => $func::<u8 $($extra)*>($($arg),*),
            $crate::PixelId::Int8
            | $crate::PixelId::VectorInt8 => $func::<i8 $($extra)*>($($arg),*),
            $crate::PixelId::UInt16
            | $crate::PixelId::VectorUInt16 => $func::<u16 $($extra)*>($($arg),*),
            $crate::PixelId::Int16
            | $crate::PixelId::VectorInt16 => $func::<i16 $($extra)*>($($arg),*),
            $crate::PixelId::UInt32
            | $crate::PixelId::VectorUInt32 => $func::<u32 $($extra)*>($($arg),*),
            $crate::PixelId::Int32
            | $crate::PixelId::VectorInt32 => $func::<i32 $($extra)*>($($arg),*),
            $crate::PixelId::UInt64
            | $crate::PixelId::VectorUInt64 => $func::<u64 $($extra)*>($($arg),*),
            $crate::PixelId::Int64
            | $crate::PixelId::VectorInt64 => $func::<i64 $($extra)*>($($arg),*),
            $crate::PixelId::Float32
            | $crate::PixelId::ComplexFloat32
            | $crate::PixelId::VectorFloat32 => $func::<f32 $($extra)*>($($arg),*),
            $crate::PixelId::Float64
            | $crate::PixelId::ComplexFloat64
            | $crate::PixelId::VectorFloat64 => $func::<f64 $($extra)*>($($arg),*),
        }
    }};
    ($id:expr, $func:ident $(, $arg:expr)* $(,)?) => {
        $crate::dispatch_scalar!(@table $id, $func, [] $(, $arg)*)
    };
}

/// [`dispatch_scalar!`] for a `$func` that is generic over the scalar type **and**
/// over further parameters left to inference — typically a closure type, which
/// cannot be named and so cannot be written into a turbofish.
///
/// Switches on the same [`PixelId`] table. The dispatched scalar type is always
/// the **first** generic parameter of `$func`; `$infer` supplies the rest as
/// `_`, one per remaining parameter:
///
/// ```text
/// dispatch_scalar_infer!([, _]    id, f, a)  =>  f::<u8, _>(a)
/// dispatch_scalar_infer!([, _, _] id, f, a)  =>  f::<u8, _, _>(a)
/// ```
///
/// [`crate::fused::map_pixels`] nests two of these to monomorphize its pass over
/// the *(input, output)* type pair while the caller's `f64 -> f64` closure stays
/// a generic parameter — so the inner loop keeps no dynamic dispatch.
#[macro_export]
macro_rules! dispatch_scalar_infer {
    ([$($infer:tt)*] $id:expr, $func:ident $(, $arg:expr)* $(,)?) => {
        $crate::dispatch_scalar!(@table $id, $func, [$($infer)*] $(, $arg)*)
    };
}
