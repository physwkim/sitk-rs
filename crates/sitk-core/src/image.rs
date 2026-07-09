//! The runtime-typed N-dimensional [`Image`] and its type-erased [`PixelBuffer`].

use crate::error::{Error, Result};
use crate::matrix;
use crate::pixel::{PixelId, Scalar};

/// Type-erased *component* storage: one `Vec` variant per scalar component type.
///
/// Data is stored in ITK/SimpleITK order — the first index (x) varies fastest.
/// For a scalar image the buffer holds one element per pixel. For a vector
/// image it holds `number_of_pixels * components_per_pixel` elements,
/// **interleaved**: the components of one pixel are adjacent, exactly as
/// `itk::VectorImage` lays out its single contiguous `ImportImageContainer`
/// (itkVectorImage.h: the pixel components are stored contiguously in a buffer
/// of length `NumberOfPixels * VectorLength`).
///
/// A `PixelBuffer` therefore knows its *component* type, never whether the
/// image that owns it is scalar or vector; that distinction lives on [`Image`].
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
    /// A vector `id` selects the same variant as its component's scalar id;
    /// `len` is a component count, not a pixel count.
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
            PixelId::Float32 | PixelId::VectorFloat32 => PixelBuffer::Float32(vec![0.0; len]),
            PixelId::Float64 | PixelId::VectorFloat64 => PixelBuffer::Float64(vec![0.0; len]),
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

    /// Widen every stored component to `f64`, preserving interleaved order.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        fn widen<T: Scalar>(v: &[T]) -> Vec<f64> {
            v.iter().map(|&x| x.as_f64()).collect()
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
/// # Scalar and vector images
///
/// Mirroring SimpleITK's `sitkImage`, one `Image` type carries both
/// `itk::Image` and `itk::VectorImage`: [`Image::pixel_id`] names which, and
/// [`Image::number_of_components_per_pixel`] gives the vector length. The
/// following invariant holds by construction — every `Image` is built through
/// the private `assemble` seam, which rejects any other combination:
///
/// ```text
/// components_per_pixel >= 1
/// !pixel_id.is_vector()  =>  components_per_pixel == 1
/// buffer.component_id()  ==  pixel_id.component_id()
/// buffer.len()           ==  number_of_pixels * components_per_pixel
/// ```
///
/// Consequently the scalar accessors ([`Image::scalar_slice`],
/// [`Image::scalar_vec_mut`]) can — and do — reject a vector image with
/// [`Error::RequiresScalarPixelType`] rather than hand back an interleaved
/// buffer that a scalar consumer would misread.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    buffer: PixelBuffer,
    pixel_id: PixelId,
    components_per_pixel: usize,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
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

        let legal_components = if pixel_id.is_vector() {
            components_per_pixel >= 1
        } else {
            components_per_pixel == 1
        };
        if !legal_components {
            return Err(Error::InvalidComponentCount {
                pixel_id,
                components_per_pixel,
            });
        }
        if buffer.component_id() != pixel_id.component_id() {
            return Err(Error::PixelTypeMismatch {
                expected: pixel_id.component_id(),
                requested: buffer.component_id(),
            });
        }

        let number_of_pixels: usize = size.iter().product();
        let expected = number_of_pixels * components_per_pixel;
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
            components_per_pixel,
            size,
            spacing,
            origin,
            direction,
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
    /// [`Image::new_vector`] to choose the count.
    pub fn new(size: &[usize], id: PixelId) -> Self {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let components = if id.is_vector() { size.len() } else { 1 };
        Self::new_vector(size, id, components)
            .expect("`size.len() >= 1` components is legal for every pixel id")
    }

    /// A new zero-filled image with an explicit component count.
    ///
    /// A scalar `id` accepts only `components_per_pixel == 1`; a vector `id`
    /// accepts any count `>= 1`. Mirrors SimpleITK's
    /// `Image(size, valueEnum, numberOfComponents)` and its `AllocateInternal`
    /// check (sitkImage.hxx:63-67), which throws "Specified number of
    /// components as N but did not specify pixelID as a vector type!".
    pub fn new_vector(size: &[usize], id: PixelId, components_per_pixel: usize) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let n: usize = size.iter().product();
        let (spacing, origin, direction) = Self::default_geometry(size.len());
        Self::assemble(
            PixelBuffer::zeroed(id, n * components_per_pixel),
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

    /// Interleave `images` — one scalar image per component — into a vector
    /// image. This is `itk::ComposeImageFilter`'s primitive.
    ///
    /// Every input must be scalar, of the same pixel type and the same size
    /// ("All input images are expected to have the same template parameters and
    /// have the same size and origin" — `ComposeImageFilter.yaml`'s
    /// detaileddescription). The output takes its geometry from `images[0]` and
    /// its pixel type from that component type's vector variant.
    ///
    /// Errors on an empty `images` list.
    pub fn from_component_images(images: &[&Image]) -> Result<Self> {
        let Some(first) = images.first() else {
            return Err(Error::EmptyComponentImageList);
        };
        for img in images {
            if img.pixel_id.is_vector() {
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
        if index >= self.components_per_pixel {
            return Err(Error::ComponentIndexOutOfRange {
                index,
                components_per_pixel: self.components_per_pixel,
            });
        }

        fn take<T: Scalar>(img: &Image, index: usize) -> Result<PixelBuffer> {
            let all = img.component_slice::<T>()?;
            let stride = img.components_per_pixel;
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
    /// Always `1` for a scalar image, `>= 1` for a vector image.
    pub fn number_of_components_per_pixel(&self) -> usize {
        self.components_per_pixel
    }

    /// Total number of pixels — the product of [`Image::size`], *not* the
    /// buffer length (which is this times
    /// [`Image::number_of_components_per_pixel`]).
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

    /// The scalar guard: `Ok(())` for a scalar image, and
    /// [`Error::RequiresScalarPixelType`] for a vector one.
    ///
    /// Every scalar-typed read of an `Image` goes through this, so no consumer
    /// can reach an interleaved buffer while believing it holds one value per
    /// pixel.
    fn require_scalar(&self) -> Result<()> {
        if self.pixel_id.is_vector() {
            return Err(Error::RequiresScalarPixelType(self.pixel_id));
        }
        Ok(())
    }

    /// Borrow a scalar image's buffer as a concrete `&[T]`, one element per
    /// pixel.
    ///
    /// Errors with [`Error::RequiresScalarPixelType`] on a vector image and
    /// with [`Error::PixelTypeMismatch`] if `T` is not the image's pixel type.
    pub fn scalar_slice<T: Scalar>(&self) -> Result<&[T]> {
        self.require_scalar()?;
        T::buffer_ref(&self.buffer).ok_or(Error::PixelTypeMismatch {
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
        T::buffer_mut(&mut self.buffer).ok_or(Error::PixelTypeMismatch {
            expected: id,
            requested: T::PIXEL_ID,
        })
    }

    /// Borrow the whole interleaved component buffer as `&[T]`, for scalar and
    /// vector images alike — `T` is the *component* type.
    ///
    /// Length is `number_of_pixels() * number_of_components_per_pixel()`. This
    /// is the accessor vector filters use; scalar consumers want
    /// [`Image::scalar_slice`], which refuses vector images.
    pub fn component_slice<T: Scalar>(&self) -> Result<&[T]> {
        T::buffer_ref(&self.buffer).ok_or(Error::PixelTypeMismatch {
            expected: self.pixel_id.component_id(),
            requested: T::PIXEL_ID,
        })
    }

    /// Borrow the whole interleaved component buffer mutably as `&mut Vec<T>`.
    /// The counterpart of [`Image::component_slice`].
    pub fn component_vec_mut<T: Scalar>(&mut self) -> Result<&mut Vec<T>> {
        let expected = self.pixel_id.component_id();
        T::buffer_mut(&mut self.buffer).ok_or(Error::PixelTypeMismatch {
            expected,
            requested: T::PIXEL_ID,
        })
    }

    /// Copy a scalar image's buffer into an `f64` vector regardless of stored
    /// pixel type, one element per pixel. A typed accessor, not an algorithm —
    /// filters and resampling both widen to `f64` to compute uniformly.
    ///
    /// Errors with [`Error::RequiresScalarPixelType`] on a vector image. Every
    /// caller of this function indexes the result by pixel, and a vector image's
    /// buffer is `components_per_pixel` values per pixel; returning it here
    /// would silently misalign every one of them. Vector callers want
    /// [`Image::components_to_f64_vec`].
    ///
    /// Together with [`Image::scalar_slice`] this is the whole scalar read
    /// surface of `Image`, so a filter cannot reach pixel data without passing
    /// the guard.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        self.require_scalar()?;
        Ok(self.buffer.to_f64_vec())
    }

    /// Copy the interleaved component buffer into an `f64` vector, for scalar
    /// and vector images alike. Length is `number_of_pixels() *
    /// number_of_components_per_pixel()`.
    pub fn components_to_f64_vec(&self) -> Vec<f64> {
        self.buffer.to_f64_vec()
    }

    /// Linear buffer offset of a multi-index (first index fastest). Does not
    /// bounds-check against `size`.
    ///
    /// This is a *pixel* offset. For a vector image, the components of that
    /// pixel start at `linear_index(index) * number_of_components_per_pixel()`;
    /// see [`Image::component_index`].
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
        self.linear_index(index) * self.components_per_pixel + component
    }

    /// The components of the pixel at `index`, as a `&[T]` of length
    /// [`Image::number_of_components_per_pixel`] — SimpleITK's
    /// `GetPixelAsVector*`.
    ///
    /// Works for scalar images too, where the slice has length 1. Errors on
    /// component-type mismatch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds, like indexing a slice.
    pub fn get_vector<T: Scalar>(&self, index: &[usize]) -> Result<&[T]> {
        let start = self.component_index(index, 0);
        let components = self.components_per_pixel;
        let all = self.component_slice::<T>()?;
        Ok(&all[start..start + components])
    }

    /// Overwrite the components of the pixel at `index` — SimpleITK's
    /// `SetPixelAsVector*`.
    ///
    /// Errors on component-type mismatch, or if `values.len()` is not
    /// [`Image::number_of_components_per_pixel`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds, like indexing a slice.
    pub fn set_vector<T: Scalar>(&mut self, index: &[usize], values: &[T]) -> Result<()> {
        let components = self.components_per_pixel;
        if values.len() != components {
            return Err(Error::InvalidComponentCount {
                pixel_id: self.pixel_id,
                components_per_pixel: values.len(),
            });
        }
        let start = self.component_index(index, 0);
        let all = self.component_vec_mut::<T>()?;
        all[start..start + components].copy_from_slice(values);
        Ok(())
    }

    /// Map a continuous index to a physical point:
    /// `p = origin + Direction * (spacing ⊙ index)`.
    pub fn continuous_index_to_physical_point(&self, index: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        debug_assert_eq!(index.len(), dim);
        let scaled: Vec<f64> = (0..dim).map(|d| index[d] * self.spacing[d]).collect();
        let rotated = matrix::mat_vec(&self.direction, &scaled, dim);
        (0..dim).map(|d| self.origin[d] + rotated[d]).collect()
    }

    /// Map a physical point to a continuous index:
    /// `index = (Direction⁻¹ * (p - origin)) ⊘ spacing`.
    ///
    /// Errors if the direction matrix is singular.
    pub fn physical_point_to_continuous_index(&self, point: &[f64]) -> Result<Vec<f64>> {
        let dim = self.dimension();
        debug_assert_eq!(point.len(), dim);
        let inv = matrix::invert(&self.direction, dim).ok_or(Error::SingularDirection)?;
        let diff: Vec<f64> = (0..dim).map(|d| point[d] - self.origin[d]).collect();
        let unrotated = matrix::mat_vec(&inv, &diff, dim);
        Ok((0..dim).map(|d| unrotated[d] / self.spacing[d]).collect())
    }
}

/// Dispatch a generic function on an image's runtime pixel type, recovering the
/// static type of its *components*.
///
/// `$func` names a generic `fn f<T: Scalar>(..) -> R` in scope (a bare
/// identifier, so a turbofish can be appended); the same `R` is returned for
/// every arm. The first argument is the [`PixelId`] to switch on.
///
/// A vector [`PixelId`] selects the same `T` as its component's scalar id, so
/// `$func` sees the type the buffer actually stores. That is not a licence to
/// read the buffer as if it were scalar: `$func` reaches the pixels through
/// [`Image::scalar_slice`], which rejects a vector image with
/// [`crate::Error::RequiresScalarPixelType`], or through the explicitly
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
    ($id:expr, $func:ident $(, $arg:expr)* $(,)?) => {{
        match $id {
            $crate::PixelId::UInt8 | $crate::PixelId::VectorUInt8 => $func::<u8>($($arg),*),
            $crate::PixelId::Int8 | $crate::PixelId::VectorInt8 => $func::<i8>($($arg),*),
            $crate::PixelId::UInt16 | $crate::PixelId::VectorUInt16 => $func::<u16>($($arg),*),
            $crate::PixelId::Int16 | $crate::PixelId::VectorInt16 => $func::<i16>($($arg),*),
            $crate::PixelId::UInt32 | $crate::PixelId::VectorUInt32 => $func::<u32>($($arg),*),
            $crate::PixelId::Int32 | $crate::PixelId::VectorInt32 => $func::<i32>($($arg),*),
            $crate::PixelId::UInt64 | $crate::PixelId::VectorUInt64 => $func::<u64>($($arg),*),
            $crate::PixelId::Int64 | $crate::PixelId::VectorInt64 => $func::<i64>($($arg),*),
            $crate::PixelId::Float32 | $crate::PixelId::VectorFloat32 => $func::<f32>($($arg),*),
            $crate::PixelId::Float64 | $crate::PixelId::VectorFloat64 => $func::<f64>($($arg),*),
        }
    }};
}
