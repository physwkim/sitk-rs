//! Pixel type identity and the [`Scalar`] trait bridging Rust scalar types to
//! runtime-tagged pixel buffers.
//!
//! Mirrors SimpleITK's `PixelIDValueEnum`: a SimpleITK `Image` is not templated
//! on its pixel type at the API level; the type is carried at runtime and every
//! filter dispatches on it. We reproduce that with a runtime [`PixelId`] tag and
//! an enum-of-`Vec` buffer, and recover static typing inside filters through the
//! [`Scalar`] trait.

use crate::image::PixelBuffer;

/// Runtime pixel-type tag, mirroring SimpleITK's `PixelIDValueEnum`
/// (sitkPixelIDValues.h:100-134).
///
/// The ten scalar variants tag an `itk::Image<T, N>`; the ten `Vector*`
/// variants tag an `itk::VectorImage<T, N>`, whose pixels hold
/// [`Image::number_of_components_per_pixel`](crate::Image::number_of_components_per_pixel)
/// components of the same underlying scalar type. SimpleITK's
/// `VectorPixelIDTypeList` (sitkPixelIDTypeLists.h:125-141) instantiates a
/// vector variant for every one of the ten scalar types and for no other, so
/// this list is exactly the scalar list mirrored once.
///
/// A vector variant is a *distinct pixel type* from its component's scalar
/// variant even at one component per pixel: SimpleITK's `sitkVectorFloat32`
/// with `GetNumberOfComponentsPerPixel() == 1` is not `sitkFloat32`, because
/// the two name different ITK image templates.
///
/// # Deviation: discriminant values
///
/// SimpleITK derives each discriminant from the pixel type's position in
/// `AllPixelIDTypeList`, which interleaves the two `std::complex` pixel types
/// (`sitkComplexFloat32`, `sitkComplexFloat64`) between the scalars and the
/// vectors. This port has no complex pixel type, so the vector variants take
/// the ten values immediately after the scalars (10..=19) rather than 12..=21.
/// Nothing in this workspace reads a `PixelId` discriminant numerically, and
/// the values are not part of any serialized format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum PixelId {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    UInt64 = 6,
    Int64 = 7,
    Float32 = 8,
    Float64 = 9,
    VectorUInt8 = 10,
    VectorInt8 = 11,
    VectorUInt16 = 12,
    VectorInt16 = 13,
    VectorUInt32 = 14,
    VectorInt32 = 15,
    VectorUInt64 = 16,
    VectorInt64 = 17,
    VectorFloat32 = 18,
    VectorFloat64 = 19,
}

impl PixelId {
    /// Size in bytes of one *component* of this pixel type — SimpleITK's
    /// `Image::GetSizeOfPixelComponent()`. A vector pixel occupies
    /// `size_in_bytes() * number_of_components_per_pixel()` bytes.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            PixelId::UInt8 | PixelId::Int8 | PixelId::VectorUInt8 | PixelId::VectorInt8 => 1,
            PixelId::UInt16 | PixelId::Int16 | PixelId::VectorUInt16 | PixelId::VectorInt16 => 2,
            PixelId::UInt32
            | PixelId::Int32
            | PixelId::Float32
            | PixelId::VectorUInt32
            | PixelId::VectorInt32
            | PixelId::VectorFloat32 => 4,
            PixelId::UInt64
            | PixelId::Int64
            | PixelId::Float64
            | PixelId::VectorUInt64
            | PixelId::VectorInt64
            | PixelId::VectorFloat64 => 8,
        }
    }

    /// `true` for the ten multi-component (`itk::VectorImage`) pixel types.
    pub const fn is_vector(self) -> bool {
        matches!(
            self,
            PixelId::VectorUInt8
                | PixelId::VectorInt8
                | PixelId::VectorUInt16
                | PixelId::VectorInt16
                | PixelId::VectorUInt32
                | PixelId::VectorInt32
                | PixelId::VectorUInt64
                | PixelId::VectorInt64
                | PixelId::VectorFloat32
                | PixelId::VectorFloat64
        )
    }

    /// The scalar type of this pixel's components: the identity on a scalar
    /// pixel type, and `T` for `Vector<T>` (ITK's
    /// `VectorImage<T, N>::InternalPixelType`).
    ///
    /// Always one of the ten scalar variants.
    pub const fn component_id(self) -> PixelId {
        match self {
            PixelId::VectorUInt8 => PixelId::UInt8,
            PixelId::VectorInt8 => PixelId::Int8,
            PixelId::VectorUInt16 => PixelId::UInt16,
            PixelId::VectorInt16 => PixelId::Int16,
            PixelId::VectorUInt32 => PixelId::UInt32,
            PixelId::VectorInt32 => PixelId::Int32,
            PixelId::VectorUInt64 => PixelId::UInt64,
            PixelId::VectorInt64 => PixelId::Int64,
            PixelId::VectorFloat32 => PixelId::Float32,
            PixelId::VectorFloat64 => PixelId::Float64,
            scalar => scalar,
        }
    }

    /// The vector pixel type whose components are of this pixel's scalar type:
    /// the identity on a vector pixel type, and `Vector<T>` for `T`.
    ///
    /// Always one of the ten vector variants.
    pub const fn vector_id(self) -> PixelId {
        match self {
            PixelId::UInt8 => PixelId::VectorUInt8,
            PixelId::Int8 => PixelId::VectorInt8,
            PixelId::UInt16 => PixelId::VectorUInt16,
            PixelId::Int16 => PixelId::VectorInt16,
            PixelId::UInt32 => PixelId::VectorUInt32,
            PixelId::Int32 => PixelId::VectorInt32,
            PixelId::UInt64 => PixelId::VectorUInt64,
            PixelId::Int64 => PixelId::VectorInt64,
            PixelId::Float32 => PixelId::VectorFloat32,
            PixelId::Float64 => PixelId::VectorFloat64,
            vector => vector,
        }
    }

    /// `true` when this pixel's *components* are floating point.
    pub const fn is_floating_point(self) -> bool {
        matches!(self.component_id(), PixelId::Float32 | PixelId::Float64)
    }

    /// `true` for the ten scalar integer pixel types — SimpleITK's
    /// `IntegerPixelIDTypeList` (`sitkPixelIDTypeLists.h:159`), which is what a
    /// label image's pixel type must be drawn from.
    ///
    /// A vector id is *not* an integer scalar even when its components are:
    /// `dispatch_scalar!` would resolve it to that component type and quietly
    /// read an interleaved buffer as one value per pixel.
    pub const fn is_integer_scalar(self) -> bool {
        !self.is_vector() && !self.is_floating_point()
    }

    /// `true` when this pixel's *components* can represent a negative value:
    /// the signed integer types and the two floating-point types.
    pub const fn is_signed(self) -> bool {
        !matches!(
            self.component_id(),
            PixelId::UInt8 | PixelId::UInt16 | PixelId::UInt32 | PixelId::UInt64
        )
    }
}

/// A Rust scalar type that can back a pixel buffer.
///
/// Recovering a concrete `&[T]` from a type-erased [`PixelBuffer`] is
/// deliberately *not* on this trait — see [`PixelBuffer::as_slice`], which is
/// crate-private. A `&[T]` taken straight off the buffer says nothing about
/// whether the owning image is scalar, and reading it as one element per pixel
/// silently misreads a vector image. Outside this crate the only ways to a
/// `&[T]` are [`Image::scalar_slice`] / [`Image::scalar_vec_mut`] /
/// [`Image::scalar_view`] (guarded against vector images) and
/// [`Image::component_slice`] / [`Image::component_vec_mut`] (which name in
/// their signature that they return interleaved components).
///
/// [`PixelBuffer::as_slice`]: crate::PixelBuffer::as_slice
/// [`Image::scalar_slice`]: crate::Image::scalar_slice
/// [`Image::scalar_vec_mut`]: crate::Image::scalar_vec_mut
/// [`Image::scalar_view`]: crate::Image::scalar_view
/// [`Image::component_slice`]: crate::Image::component_slice
/// [`Image::component_vec_mut`]: crate::Image::component_vec_mut
pub trait Scalar: Copy + PartialOrd + 'static {
    /// The runtime tag for this Rust type. Always a scalar variant: `Scalar` is
    /// implemented only for the ten component types, and a vector image's
    /// buffer stores those components.
    const PIXEL_ID: PixelId;

    /// Widen to `f64` for computation (`self as f64`).
    fn as_f64(self) -> f64;

    /// Narrow from `f64` with C++ `static_cast`-style semantics (`v as Self`).
    ///
    /// Note: Rust `as` from float to integer saturates and maps `NaN` to 0,
    /// whereas C++ `static_cast` of an out-of-range float is undefined. In-range
    /// values agree; the saturation behaviour is the safer choice and is a known
    /// divergence to verify against ITK when exact parity is required.
    fn from_f64(v: f64) -> Self;

    /// Wrap a `Vec<Self>` into the matching [`PixelBuffer`] variant.
    ///
    /// The write direction stays public: building a buffer cannot misread one,
    /// and an [`Image`](crate::Image) can only be assembled from it through the
    /// invariant-checking constructors.
    fn into_buffer(v: Vec<Self>) -> PixelBuffer;
}

macro_rules! impl_scalar {
    ($( $ty:ty => $variant:ident ),+ $(,)?) => {
        $(
            impl Scalar for $ty {
                const PIXEL_ID: PixelId = PixelId::$variant;

                #[inline]
                fn as_f64(self) -> f64 { self as f64 }

                #[inline]
                fn from_f64(v: f64) -> Self { v as $ty }

                #[inline]
                fn into_buffer(v: Vec<Self>) -> PixelBuffer {
                    PixelBuffer::$variant(v)
                }
            }
        )+
    };
}

impl_scalar! {
    u8 => UInt8,
    i8 => Int8,
    u16 => UInt16,
    i16 => Int16,
    u32 => UInt32,
    i32 => Int32,
    u64 => UInt64,
    i64 => Int64,
    f32 => Float32,
    f64 => Float64,
}
