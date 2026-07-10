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
/// (sitkPixelIDValues.h:100-134), discriminants included.
///
/// # The three categories
///
/// SimpleITK's `AllPixelIDTypeList` (sitkPixelIDTypeLists.h:180) concatenates
/// `BasicPixelIDTypeList ++ ComplexPixelIDTypeList ++ VectorPixelIDTypeList`,
/// and each pixel type's discriminant is its position in that list. This enum
/// reproduces those positions exactly:
///
/// - `0..=9` — the ten scalars, tagging `itk::Image<T, N>`.
/// - `10..=11` — the two `std::complex` pixel types, tagging
///   `itk::Image<std::complex<T>, N>`. Complex is a **basic** pixel type
///   upstream, not a vector one: `IsVector` is specialized only for
///   `VectorPixelID`/`itk::VectorImage` (sitkPixelIDTokens.h:53-69), so
///   `GetNumberOfComponentsPerPixel()` returns `1` for a complex image
///   (sitkPimpleImageBase.hxx:202-209) even though its buffer holds two
///   `T` per pixel.
/// - `12..=21` — the ten `Vector*` variants, tagging `itk::VectorImage<T, N>`,
///   whose pixels hold
///   [`Image::number_of_components_per_pixel`](crate::Image::number_of_components_per_pixel)
///   components of the same underlying scalar type. `VectorPixelIDTypeList`
///   (sitkPixelIDTypeLists.h:125-141) instantiates a vector variant for every
///   one of the ten scalar types and for no other, so this list is exactly the
///   scalar list mirrored once.
///
/// `22..=25` are `sitkLabelUInt8..sitkLabelUInt64` upstream
/// (sitkPixelIDValues.h:130-133) and are deliberately left unassigned here: an
/// `itk::LabelMap` derives from `ImageBase` and has no pixel container
/// (itkLabelMap.h:69), so it is not an [`Image`](crate::Image) in this port.
///
/// [`PixelId::is_scalar`], [`PixelId::is_complex`] and [`PixelId::is_vector`]
/// partition this enum exhaustively; `pixel_id_predicates_partition_the_enum`
/// pins that. Category tests must be written as whitelists over those three, so
/// that a future fourth category defaults to *rejected* rather than to
/// whichever branch happened to be the `else`.
///
/// A vector variant is a *distinct pixel type* from its component's scalar
/// variant even at one component per pixel: SimpleITK's `sitkVectorFloat32`
/// with `GetNumberOfComponentsPerPixel() == 1` is not `sitkFloat32`, because
/// the two name different ITK image templates.
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
    ComplexFloat32 = 10,
    ComplexFloat64 = 11,
    VectorUInt8 = 12,
    VectorInt8 = 13,
    VectorUInt16 = 14,
    VectorInt16 = 15,
    VectorUInt32 = 16,
    VectorInt32 = 17,
    VectorUInt64 = 18,
    VectorInt64 = 19,
    VectorFloat32 = 20,
    VectorFloat64 = 21,
    // 22..=25 are sitkLabelUInt8..sitkLabelUInt64 upstream; see the type docs.
}

impl PixelId {
    /// Size in bytes of one *component* of this pixel type: `4` for
    /// `ComplexFloat32`, whose buffer stores `f32`, and `1` for `VectorUInt8`.
    /// A pixel occupies `size_in_bytes()` times its image's
    /// [`Image::buffer_stride`](crate::Image::buffer_stride) bytes.
    ///
    /// SimpleITK's `Image::GetSizeOfPixelComponent()` returns
    /// `2 * sizeof(component)` for a complex pixel type, contradicting its own
    /// doc; this port fixed that (§3.20), so
    /// [`Image::size_of_pixel_component`](crate::Image::size_of_pixel_component)
    /// agrees with this method for every pixel type.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            PixelId::UInt8 | PixelId::Int8 | PixelId::VectorUInt8 | PixelId::VectorInt8 => 1,
            PixelId::UInt16 | PixelId::Int16 | PixelId::VectorUInt16 | PixelId::VectorInt16 => 2,
            PixelId::UInt32
            | PixelId::Int32
            | PixelId::Float32
            | PixelId::ComplexFloat32
            | PixelId::VectorUInt32
            | PixelId::VectorInt32
            | PixelId::VectorFloat32 => 4,
            PixelId::UInt64
            | PixelId::Int64
            | PixelId::Float64
            | PixelId::ComplexFloat64
            | PixelId::VectorUInt64
            | PixelId::VectorInt64
            | PixelId::VectorFloat64 => 8,
        }
    }

    /// The human-readable name SimpleITK's `GetPixelIDValueAsString`
    /// (sitkPixelIDValues.cxx:40-146) prints for this pixel type, byte for byte.
    ///
    /// The `sitkUnknown` ("Unknown pixel id") and `sitkLabel*` ("label of ...")
    /// arms have no counterpart here: an unknown pixel type is unrepresentable,
    /// and a [`LabelMap`](crate::LabelMap) is not an [`Image`](crate::Image) in
    /// this port (see the type docs).
    pub const fn as_str(self) -> &'static str {
        match self {
            PixelId::UInt8 => "8-bit unsigned integer",
            PixelId::Int8 => "8-bit signed integer",
            PixelId::UInt16 => "16-bit unsigned integer",
            PixelId::Int16 => "16-bit signed integer",
            PixelId::UInt32 => "32-bit unsigned integer",
            PixelId::Int32 => "32-bit signed integer",
            PixelId::UInt64 => "64-bit unsigned integer",
            PixelId::Int64 => "64-bit signed integer",
            PixelId::Float32 => "32-bit float",
            PixelId::Float64 => "64-bit float",
            PixelId::ComplexFloat32 => "complex of 32-bit float",
            PixelId::ComplexFloat64 => "complex of 64-bit float",
            PixelId::VectorUInt8 => "vector of 8-bit unsigned integer",
            PixelId::VectorInt8 => "vector of 8-bit signed integer",
            PixelId::VectorUInt16 => "vector of 16-bit unsigned integer",
            PixelId::VectorInt16 => "vector of 16-bit signed integer",
            PixelId::VectorUInt32 => "vector of 32-bit unsigned integer",
            PixelId::VectorInt32 => "vector of 32-bit signed integer",
            PixelId::VectorUInt64 => "vector of 64-bit unsigned integer",
            PixelId::VectorInt64 => "vector of 64-bit signed integer",
            PixelId::VectorFloat32 => "vector of 32-bit float",
            PixelId::VectorFloat64 => "vector of 64-bit float",
        }
    }

    /// `true` for the ten single-component (`itk::Image<T, N>`) pixel types.
    ///
    /// This is the positive form of the "not a vector, not complex" test. Every
    /// scalar-only guard in this workspace is written against it rather than
    /// against `!is_vector()`, so a pixel category added later cannot slip
    /// through a scalar accessor.
    pub const fn is_scalar(self) -> bool {
        matches!(
            self,
            PixelId::UInt8
                | PixelId::Int8
                | PixelId::UInt16
                | PixelId::Int16
                | PixelId::UInt32
                | PixelId::Int32
                | PixelId::UInt64
                | PixelId::Int64
                | PixelId::Float32
                | PixelId::Float64
        )
    }

    /// `true` for the two `std::complex` pixel types
    /// (`sitkComplexFloat32`, `sitkComplexFloat64`).
    pub const fn is_complex(self) -> bool {
        matches!(self, PixelId::ComplexFloat32 | PixelId::ComplexFloat64)
    }

    /// `true` for the ten multi-component (`itk::VectorImage`) pixel types.
    ///
    /// A complex pixel type is **not** a vector type — upstream's `IsVector`
    /// token is specialized only for `VectorPixelID` (sitkPixelIDTokens.h:53-69).
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

    /// How many buffer components one pixel of this type occupies, when the
    /// image reports `components_per_pixel` from
    /// [`Image::number_of_components_per_pixel`](crate::Image::number_of_components_per_pixel);
    /// `None` when that count is illegal for this pixel category.
    ///
    /// This is the three-way rule of [`Image`](crate::Image)'s invariant, held
    /// in one total `match` so that every category names its own stride rather
    /// than inheriting an `else` branch. `Image::assemble` is the only caller.
    /// A basic pixel type — complex included — admits exactly one component
    /// (`AllocateInternal`, sitkImage.hxx:60-67), a vector type any count `>= 1`.
    pub(crate) const fn buffer_stride_for(self, components_per_pixel: usize) -> Option<usize> {
        match self {
            PixelId::UInt8
            | PixelId::Int8
            | PixelId::UInt16
            | PixelId::Int16
            | PixelId::UInt32
            | PixelId::Int32
            | PixelId::UInt64
            | PixelId::Int64
            | PixelId::Float32
            | PixelId::Float64 => {
                if components_per_pixel == 1 {
                    Some(1)
                } else {
                    None
                }
            }
            PixelId::ComplexFloat32 | PixelId::ComplexFloat64 => {
                if components_per_pixel == 1 {
                    Some(2)
                } else {
                    None
                }
            }
            PixelId::VectorUInt8
            | PixelId::VectorInt8
            | PixelId::VectorUInt16
            | PixelId::VectorInt16
            | PixelId::VectorUInt32
            | PixelId::VectorInt32
            | PixelId::VectorUInt64
            | PixelId::VectorInt64
            | PixelId::VectorFloat32
            | PixelId::VectorFloat64 => {
                if components_per_pixel >= 1 {
                    Some(components_per_pixel)
                } else {
                    None
                }
            }
        }
    }

    /// The scalar type of this pixel's components: the identity on a scalar
    /// pixel type, `T` for `Vector<T>` (ITK's
    /// `VectorImage<T, N>::InternalPixelType`), and `T` for `std::complex<T>`
    /// (`NumericTraits<std::complex<T>>::ValueType`).
    ///
    /// Always one of the ten scalar variants. For a complex pixel type this is
    /// the element type of the interleaved buffer SimpleITK hands back from
    /// `GetBufferAsFloat()` (sitkPimpleImageBase.hxx:838-842).
    pub const fn component_id(self) -> PixelId {
        match self {
            PixelId::ComplexFloat32 => PixelId::Float32,
            PixelId::ComplexFloat64 => PixelId::Float64,
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

    /// The vector pixel type whose components are of this pixel's *component*
    /// type — the inverse of [`PixelId::component_id`] on the vector variants,
    /// and the identity on a vector pixel type.
    ///
    /// Always one of the ten vector variants. `ComplexFloat32` maps to
    /// `VectorFloat32` because `Float32` is its component type; upstream has no
    /// `sitkVectorComplexFloat32`, and no caller here vectorizes a complex
    /// pixel type.
    pub const fn vector_id(self) -> PixelId {
        match self {
            PixelId::ComplexFloat32 => PixelId::VectorFloat32,
            PixelId::ComplexFloat64 => PixelId::VectorFloat64,
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
    /// read an interleaved buffer as one value per pixel. Neither is a complex
    /// id, whose buffer holds two components per pixel.
    ///
    /// Written as a whitelist so that a pixel type added later is rejected by
    /// default rather than admitted by an omission in a negated test.
    pub const fn is_integer_scalar(self) -> bool {
        matches!(
            self,
            PixelId::UInt8
                | PixelId::Int8
                | PixelId::UInt16
                | PixelId::Int16
                | PixelId::UInt32
                | PixelId::Int32
                | PixelId::UInt64
                | PixelId::Int64
        )
    }

    /// `(NumericTraits<T>::NonpositiveMin(), NumericTraits<T>::max())` for the
    /// eight integer scalar types, `None` for every other pixel type.
    ///
    /// This is the constructor-side guard for [`LabelMap`](crate::LabelMap):
    /// taking the bounds *is* the proof that the pixel type can back a label
    /// image, so `LabelMap::push_label_object` can consult them without a
    /// runtime check of its own.
    ///
    /// `UInt64`'s upper bound is clamped to `i64::MAX`. A label is an `i64`
    /// throughout this port, so a `u64` label above `i64::MAX` is unrepresentable
    /// before it ever reaches the bound.
    pub const fn integer_scalar_bounds(self) -> Option<(i64, i64)> {
        match self {
            PixelId::UInt8 => Some((0, u8::MAX as i64)),
            PixelId::Int8 => Some((i8::MIN as i64, i8::MAX as i64)),
            PixelId::UInt16 => Some((0, u16::MAX as i64)),
            PixelId::Int16 => Some((i16::MIN as i64, i16::MAX as i64)),
            PixelId::UInt32 => Some((0, u32::MAX as i64)),
            PixelId::Int32 => Some((i32::MIN as i64, i32::MAX as i64)),
            PixelId::UInt64 => Some((0, i64::MAX)),
            PixelId::Int64 => Some((i64::MIN, i64::MAX)),
            PixelId::Float32
            | PixelId::Float64
            | PixelId::ComplexFloat32
            | PixelId::ComplexFloat64
            | PixelId::VectorUInt8
            | PixelId::VectorInt8
            | PixelId::VectorUInt16
            | PixelId::VectorInt16
            | PixelId::VectorUInt32
            | PixelId::VectorInt32
            | PixelId::VectorUInt64
            | PixelId::VectorInt64
            | PixelId::VectorFloat32
            | PixelId::VectorFloat64 => None,
        }
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

/// The two component types that admit a `std::complex` pixel: `f32` and `f64`.
///
/// This is SimpleITK's `ComplexPixelIDTypeList` (sitkPixelIDTypeLists.h:104)
/// read as a trait, and it doubles as `RealPixelIDTypeList`
/// (sitkPixelIDTypeLists.h:98) — the two lists range over the same component
/// types, which is why `ComplexToReal` maps `ComplexFloat32 -> Float32` and
/// `RealAndImaginaryToComplex` maps `Float32 -> ComplexFloat32`.
pub trait Real: Scalar {
    /// The complex pixel type whose components are `Self`.
    const COMPLEX_ID: PixelId;
}

impl Real for f32 {
    const COMPLEX_ID: PixelId = PixelId::ComplexFloat32;
}

impl Real for f64 {
    const COMPLEX_ID: PixelId = PixelId::ComplexFloat64;
}

/// One complex pixel, passed and returned by value.
///
/// SimpleITK's `GetPixelAsComplexFloat32` / `SetPixelAsComplexFloat32`
/// (sitkImage.h:536-538, sitkImage.cxx:596-608) likewise trade in
/// `std::complex<T>` values; the buffer itself stays interleaved `re, im, ...`
/// and is reached through
/// [`Image::complex_components`](crate::Image::complex_components), the exact
/// analogue of `GetBufferAsFloat()` on a complex image
/// (sitkPimpleImageBase.hxx:838-842).
///
/// `#[repr(C)]` records that the field order matches `std::complex<T>`'s
/// storage. No code in this workspace reinterprets a `&[T]` as a
/// `&[Complex<T>]`: doing so would need `unsafe`, and the workspace has none.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// Real part — `std::complex<T>::real()`.
    pub re: T,
    /// Imaginary part — `std::complex<T>::imag()`.
    pub im: T,
}

impl<T> Complex<T> {
    /// A complex value from its real and imaginary parts.
    pub const fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }
}
