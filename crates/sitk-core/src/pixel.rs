//! Pixel type identity and the [`Scalar`] trait bridging Rust scalar types to
//! runtime-tagged pixel buffers.
//!
//! Mirrors SimpleITK's `PixelIDValueEnum`: a SimpleITK `Image` is not templated
//! on its pixel type at the API level; the type is carried at runtime and every
//! filter dispatches on it. We reproduce that with a runtime [`PixelId`] tag and
//! an enum-of-`Vec` buffer, and recover static typing inside filters through the
//! [`Scalar`] trait.

use crate::image::PixelBuffer;

/// Runtime pixel-type tag. Discriminants match the scalar subset of SimpleITK's
/// `PixelIDValueEnum` ordering so the values stay compatible as the port grows.
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
}

impl PixelId {
    /// Size in bytes of one pixel of this type.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            PixelId::UInt8 | PixelId::Int8 => 1,
            PixelId::UInt16 | PixelId::Int16 => 2,
            PixelId::UInt32 | PixelId::Int32 | PixelId::Float32 => 4,
            PixelId::UInt64 | PixelId::Int64 | PixelId::Float64 => 8,
        }
    }

    /// `true` for the two floating-point pixel types.
    pub const fn is_floating_point(self) -> bool {
        matches!(self, PixelId::Float32 | PixelId::Float64)
    }

    /// `true` for pixel types that can represent a negative value: the
    /// signed integer types and the two floating-point types.
    pub const fn is_signed(self) -> bool {
        !matches!(
            self,
            PixelId::UInt8 | PixelId::UInt16 | PixelId::UInt32 | PixelId::UInt64
        )
    }
}

/// A Rust scalar type that can back a pixel buffer.
///
/// The buffer-bridging methods let a filter recover a concrete `&[T]` from the
/// type-erased [`PixelBuffer`] without `unsafe`: each implementation matches
/// exactly the one enum variant that holds its own type.
pub trait Scalar: Copy + PartialOrd + 'static {
    /// The runtime tag for this Rust type.
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

    /// Borrow the buffer as `&[Self]`, or `None` if the tag does not match.
    fn buffer_ref(buf: &PixelBuffer) -> Option<&[Self]>;

    /// Borrow the backing `Vec<Self>` mutably, or `None` if the tag mismatches.
    fn buffer_mut(buf: &mut PixelBuffer) -> Option<&mut Vec<Self>>;

    /// Wrap a `Vec<Self>` into the matching [`PixelBuffer`] variant.
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
                fn buffer_ref(buf: &PixelBuffer) -> Option<&[Self]> {
                    match buf {
                        PixelBuffer::$variant(v) => Some(v.as_slice()),
                        _ => None,
                    }
                }

                #[inline]
                fn buffer_mut(buf: &mut PixelBuffer) -> Option<&mut Vec<Self>> {
                    match buf {
                        PixelBuffer::$variant(v) => Some(v),
                        _ => None,
                    }
                }

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
