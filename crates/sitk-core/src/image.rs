//! The runtime-typed N-dimensional [`Image`] and its type-erased [`PixelBuffer`].

use crate::error::{Error, Result};
use crate::matrix;
use crate::pixel::{PixelId, Scalar};

/// Type-erased pixel storage: one `Vec` variant per scalar pixel type.
///
/// Data is stored in ITK/SimpleITK order — the first index (x) varies fastest —
/// as a single contiguous buffer of `number_of_pixels` elements.
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
    /// A zero-filled buffer of `len` pixels of the given type.
    pub fn zeroed(id: PixelId, len: usize) -> Self {
        match id {
            PixelId::UInt8 => PixelBuffer::UInt8(vec![0; len]),
            PixelId::Int8 => PixelBuffer::Int8(vec![0; len]),
            PixelId::UInt16 => PixelBuffer::UInt16(vec![0; len]),
            PixelId::Int16 => PixelBuffer::Int16(vec![0; len]),
            PixelId::UInt32 => PixelBuffer::UInt32(vec![0; len]),
            PixelId::Int32 => PixelBuffer::Int32(vec![0; len]),
            PixelId::UInt64 => PixelBuffer::UInt64(vec![0; len]),
            PixelId::Int64 => PixelBuffer::Int64(vec![0; len]),
            PixelId::Float32 => PixelBuffer::Float32(vec![0.0; len]),
            PixelId::Float64 => PixelBuffer::Float64(vec![0.0; len]),
        }
    }

    /// The runtime tag of this buffer.
    pub fn pixel_id(&self) -> PixelId {
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

    /// Number of pixels held.
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

    /// `true` if the buffer holds no pixels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An N-dimensional image: a [`PixelBuffer`] plus the physical-space geometry
/// (size, spacing, origin, direction cosine matrix) that ITK/SimpleITK attach to
/// every image.
///
/// Geometry vectors are all indexed in axis order matching [`Image::size`]; the
/// direction matrix is stored row-major and is `dimension x dimension`.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    buffer: PixelBuffer,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
}

impl Image {
    /// A new zero-filled image of the given `size` and pixel type, with default
    /// geometry (unit spacing, zero origin, identity direction).
    ///
    /// `size` is in SimpleITK order (`[x, y, z, ...]`) and must be non-empty.
    pub fn new(size: &[usize], id: PixelId) -> Self {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let n: usize = size.iter().product();
        let dim = size.len();
        Image {
            buffer: PixelBuffer::zeroed(id, n),
            size: size.to_vec(),
            spacing: vec![1.0; dim],
            origin: vec![0.0; dim],
            direction: matrix::identity(dim),
        }
    }

    /// Build an image from a typed buffer laid out in first-index-fastest order.
    ///
    /// Errors if `data.len()` does not equal the product of `size`.
    pub fn from_vec<T: Scalar>(size: &[usize], data: Vec<T>) -> Result<Self> {
        assert!(!size.is_empty(), "image dimension must be >= 1");
        let n: usize = size.iter().product();
        if data.len() != n {
            return Err(Error::BufferSizeMismatch {
                expected: n,
                actual: data.len(),
            });
        }
        let dim = size.len();
        Ok(Image {
            buffer: T::into_buffer(data),
            size: size.to_vec(),
            spacing: vec![1.0; dim],
            origin: vec![0.0; dim],
            direction: matrix::identity(dim),
        })
    }

    /// Assemble an image from parts, validating that geometry lengths agree with
    /// the buffer size. Used by IO where all fields are read from a file.
    pub fn from_parts(
        buffer: PixelBuffer,
        size: Vec<usize>,
        spacing: Vec<f64>,
        origin: Vec<f64>,
        direction: Vec<f64>,
    ) -> Result<Self> {
        let dim = size.len();
        let n: usize = size.iter().product();
        if buffer.len() != n {
            return Err(Error::BufferSizeMismatch {
                expected: n,
                actual: buffer.len(),
            });
        }
        if spacing.len() != dim || origin.len() != dim || direction.len() != dim * dim {
            return Err(Error::GeometryMismatch { dimension: dim });
        }
        Ok(Image {
            buffer,
            size,
            spacing,
            origin,
            direction,
        })
    }

    /// Number of spatial dimensions.
    pub fn dimension(&self) -> usize {
        self.size.len()
    }

    /// Size along each axis, in SimpleITK order.
    pub fn size(&self) -> &[usize] {
        &self.size
    }

    /// The runtime pixel-type tag.
    pub fn pixel_id(&self) -> PixelId {
        self.buffer.pixel_id()
    }

    /// Total number of pixels.
    pub fn number_of_pixels(&self) -> usize {
        self.buffer.len()
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

    /// Borrow the type-erased buffer (used by dispatch macros).
    pub fn buffer(&self) -> &PixelBuffer {
        &self.buffer
    }

    /// Borrow the type-erased buffer mutably.
    pub fn buffer_mut(&mut self) -> &mut PixelBuffer {
        &mut self.buffer
    }

    /// Borrow the buffer as a concrete `&[T]`; errors if `T` does not match the
    /// image's pixel type.
    pub fn scalar_slice<T: Scalar>(&self) -> Result<&[T]> {
        T::buffer_ref(&self.buffer).ok_or_else(|| Error::PixelTypeMismatch {
            expected: self.pixel_id(),
            requested: T::PIXEL_ID,
        })
    }

    /// Borrow the backing `Vec<T>` mutably; errors on pixel-type mismatch.
    pub fn scalar_vec_mut<T: Scalar>(&mut self) -> Result<&mut Vec<T>> {
        let id = self.pixel_id();
        T::buffer_mut(&mut self.buffer).ok_or(Error::PixelTypeMismatch {
            expected: id,
            requested: T::PIXEL_ID,
        })
    }

    /// Copy the buffer into an `f64` vector regardless of stored pixel type.
    /// A typed accessor, not an algorithm — filters and resampling both widen to
    /// `f64` to compute uniformly.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        fn collect<T: Scalar>(img: &Image) -> Vec<f64> {
            img.scalar_slice::<T>()
                .expect("dispatch guarantees T matches pixel_id")
                .iter()
                .map(|&x| x.as_f64())
                .collect()
        }
        crate::dispatch_scalar!(self.pixel_id(), collect, self)
    }

    /// Linear buffer offset of a multi-index (first index fastest). Does not
    /// bounds-check against `size`.
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

/// Dispatch a generic function on an image's runtime pixel type.
///
/// `$func` names a generic `fn f<T: Scalar>(..) -> R` in scope (a bare
/// identifier, so a turbofish can be appended); the same `R` is returned for
/// every arm. The first argument is the [`PixelId`] to switch on.
///
/// ```ignore
/// fn count<T: Scalar>(img: &Image) -> usize { img.number_of_pixels() }
/// let n = dispatch_scalar!(img.pixel_id(), count, &img);
/// ```
#[macro_export]
macro_rules! dispatch_scalar {
    ($id:expr, $func:ident $(, $arg:expr)* $(,)?) => {{
        match $id {
            $crate::PixelId::UInt8 => $func::<u8>($($arg),*),
            $crate::PixelId::Int8 => $func::<i8>($($arg),*),
            $crate::PixelId::UInt16 => $func::<u16>($($arg),*),
            $crate::PixelId::Int16 => $func::<i16>($($arg),*),
            $crate::PixelId::UInt32 => $func::<u32>($($arg),*),
            $crate::PixelId::Int32 => $func::<i32>($($arg),*),
            $crate::PixelId::UInt64 => $func::<u64>($($arg),*),
            $crate::PixelId::Int64 => $func::<i64>($($arg),*),
            $crate::PixelId::Float32 => $func::<f32>($($arg),*),
            $crate::PixelId::Float64 => $func::<f64>($($arg),*),
        }
    }};
}
