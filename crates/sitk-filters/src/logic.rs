//! Logic and bitwise pixel filters for sitk-rs, verified against the ITK v6
//! source under `Modules/Filtering/ImageIntensity/include/`:
//! `itkBitwiseOpsFunctors.h`, `itkLogicOpsFunctors.h`, `itkAndImageFilter.h`,
//! `itkOrImageFilter.h`, `itkXorImageFilter.h`, `itkNotImageFilter.h`,
//! `itkMaskImageFilter.h`, `itkMaskNegatedImageFilter.h`,
//! `itkMaximumImageFilter.h`, `itkMinimumImageFilter.h`.
//!
//! [`and`]/[`or`]/[`xor`] are pixel-type-compute (`functor.rs`'s policy
//! (b)): ITK's `Functor::AND`/`OR`/`XOR` evaluate `&`/`|`/`^` directly in the
//! pixel type (`static_cast<TOutput>(A & B)`), which C++ only defines for
//! integer types -- `AndImageFilter`'s `BitwiseOperators` concept check
//! (mirrored by SimpleITK's `IntegerPixelIDTypeList` restriction on the
//! generated wrappers) rejects float instantiations at compile time. This
//! port checks [`PixelId::is_floating_point`] at runtime and returns
//! [`FilterError::RequiresIntegerPixelType`] in place of the C++ compile
//! error; [`BinaryFunctor`] impls for `f32`/`f64` exist only to satisfy the
//! dispatch engine's blanket bound and `unreachable!()` if ever reached.
//!
//! [`not`] (`itkLogicOpsFunctors.h`'s `Functor::NOT`, integer-only for the
//! same reason) maps to `0`/`1`, not a bitwise complement: `if (!A) return
//! ForegroundValue(1); return BackgroundValue(0);`. The only operation is an
//! exact equality-to-zero test, which `f64` promotion preserves exactly for
//! every integer pixel type this crate supports, so it's implemented on the
//! [`UnaryFunctor`] (f64-compute) engine instead of adding a new
//! single-image pixel-type-compute engine for one filter.
//!
//! [`mask`]/[`mask_negated`]/[`maximum`]/[`minimum`] are also
//! pixel-type-compute, but defined for every pixel type
//! (`itkMaximumImageFilter.h`'s `Maximum` functor compares directly with
//! `operator>`, no promotion), so they carry no integer gate. ITK lets the
//! mask image in `mask`/`mask_negated` use a different pixel type than the
//! main image (commonly `UInt8`); this port's [`BinaryFunctor`] engine
//! requires both operands to share one pixel type (the same constraint
//! `add`/`subtract`/... already have in `lib.rs`), so the mask image must be
//! cast to the main image's pixel type first.

use crate::functor::{self, BinaryFunctor, UnaryFunctor};
use crate::{FilterError, Result};
use sitk_core::{Image, Scalar};

fn require_integer_pixel_type(img: &Image) -> Result<()> {
    if img.pixel_id().is_floating_point() {
        return Err(FilterError::RequiresIntegerPixelType(img.pixel_id()));
    }
    Ok(())
}

// ---- bitwise AND / OR / XOR (integer pixel types only) --------------------

/// `AND` functor (`itkBitwiseOpsFunctors.h`): `a & b`.
struct AndOp;
/// `OR` functor (`itkBitwiseOpsFunctors.h`): `a | b`.
struct OrOp;
/// `XOR` functor (`itkBitwiseOpsFunctors.h`): `a ^ b`.
struct XorOp;

macro_rules! impl_bitwise_int {
    ($op:ty, $sym:tt, $($t:ty),+ $(,)?) => {$(
        impl BinaryFunctor<$t> for $op {
            fn apply(&self, a: $t, b: $t) -> $t { a $sym b }
        }
    )+};
}

macro_rules! impl_bitwise_float_unreachable {
    ($op:ty) => {
        impl BinaryFunctor<f32> for $op {
            fn apply(&self, _a: f32, _b: f32) -> f32 {
                unreachable!("gated to integer pixel types by require_integer_pixel_type")
            }
        }
        impl BinaryFunctor<f64> for $op {
            fn apply(&self, _a: f64, _b: f64) -> f64 {
                unreachable!("gated to integer pixel types by require_integer_pixel_type")
            }
        }
    };
}

impl_bitwise_int!(AndOp, &, u8, i8, u16, i16, u32, i32, u64, i64);
impl_bitwise_int!(OrOp, |, u8, i8, u16, i16, u32, i32, u64, i64);
impl_bitwise_int!(XorOp, ^, u8, i8, u16, i16, u32, i32, u64, i64);
impl_bitwise_float_unreachable!(AndOp);
impl_bitwise_float_unreachable!(OrOp);
impl_bitwise_float_unreachable!(XorOp);

/// `AndImageFilter`: pixel-wise `a & b`. Integer pixel types only (see the
/// module docs); errors with [`FilterError::RequiresIntegerPixelType`] on a
/// floating-point image.
pub fn and(a: &Image, b: &Image) -> Result<Image> {
    require_integer_pixel_type(a)?;
    functor::binary_apply(a, b, &AndOp)
}

/// In-place variant of [`and`]: reuses `a`'s buffer.
pub fn and_in_place(a: Image, b: &Image) -> Result<Image> {
    require_integer_pixel_type(&a)?;
    functor::binary_apply_in_place(a, b, &AndOp)
}

/// `OrImageFilter`: pixel-wise `a | b`. Integer pixel types only (see the
/// module docs); errors with [`FilterError::RequiresIntegerPixelType`] on a
/// floating-point image.
pub fn or(a: &Image, b: &Image) -> Result<Image> {
    require_integer_pixel_type(a)?;
    functor::binary_apply(a, b, &OrOp)
}

/// In-place variant of [`or`]: reuses `a`'s buffer.
pub fn or_in_place(a: Image, b: &Image) -> Result<Image> {
    require_integer_pixel_type(&a)?;
    functor::binary_apply_in_place(a, b, &OrOp)
}

/// `XorImageFilter`: pixel-wise `a ^ b`. Integer pixel types only (see the
/// module docs); errors with [`FilterError::RequiresIntegerPixelType`] on a
/// floating-point image.
pub fn xor(a: &Image, b: &Image) -> Result<Image> {
    require_integer_pixel_type(a)?;
    functor::binary_apply(a, b, &XorOp)
}

/// In-place variant of [`xor`]: reuses `a`'s buffer.
pub fn xor_in_place(a: Image, b: &Image) -> Result<Image> {
    require_integer_pixel_type(&a)?;
    functor::binary_apply_in_place(a, b, &XorOp)
}

// ---- NOT (integer pixel types only) ----------------------------------------

/// `NOT` functor (`itkLogicOpsFunctors.h`): `!A ? ForegroundValue(1) :
/// BackgroundValue(0)`, i.e. `0` maps to `1` and any nonzero value maps to
/// `0`. See the module docs for why this runs on the f64-compute engine.
struct NotOp;
impl UnaryFunctor for NotOp {
    fn apply(&self, x: f64) -> f64 {
        if x == 0.0 { 1.0 } else { 0.0 }
    }
}

/// `NotImageFilter`: pixel-wise logical NOT, mapping `0` to `1` and any
/// nonzero value to `0` (not a bitwise complement). Integer pixel types only
/// (see the module docs); errors with
/// [`FilterError::RequiresIntegerPixelType`] on a floating-point image.
pub fn not(img: &Image) -> Result<Image> {
    require_integer_pixel_type(img)?;
    functor::unary_apply(img, &NotOp)
}

/// In-place variant of [`not`]: reuses `img`'s buffer.
pub fn not_in_place(img: Image) -> Result<Image> {
    require_integer_pixel_type(&img)?;
    functor::unary_apply_in_place(img, &NotOp)
}

// ---- mask / mask_negated (all pixel types) ---------------------------------

/// `MaskInput` functor (`itkMaskImageFilter.h`): `b != masking_value ?
/// static_cast<TOutput>(a) : outside_value`.
struct MaskOp {
    outside_value: f64,
    masking_value: f64,
}
impl<T: Scalar> BinaryFunctor<T> for MaskOp {
    fn apply(&self, a: T, b: T) -> T {
        if b != T::from_f64(self.masking_value) {
            a
        } else {
            T::from_f64(self.outside_value)
        }
    }
}

/// `MaskImageFilter`: keep `img`'s pixel where the mask differs from
/// `masking_value` (ITK default `0`), else `outside_value` (ITK default
/// `0`). `mask_img` must share `img`'s pixel type (see the module docs).
pub fn mask(
    img: &Image,
    mask_img: &Image,
    outside_value: f64,
    masking_value: f64,
) -> Result<Image> {
    functor::binary_apply(
        img,
        mask_img,
        &MaskOp {
            outside_value,
            masking_value,
        },
    )
}

/// In-place variant of [`mask`]: reuses `img`'s buffer.
pub fn mask_in_place(
    img: Image,
    mask_img: &Image,
    outside_value: f64,
    masking_value: f64,
) -> Result<Image> {
    functor::binary_apply_in_place(
        img,
        mask_img,
        &MaskOp {
            outside_value,
            masking_value,
        },
    )
}

/// `MaskNegatedInput` functor (`itkMaskNegatedImageFilter.h`): `b !=
/// masking_value ? outside_value : static_cast<TOutput>(a)` -- the logical
/// complement of [`MaskOp`].
struct MaskNegatedOp {
    outside_value: f64,
    masking_value: f64,
}
impl<T: Scalar> BinaryFunctor<T> for MaskNegatedOp {
    fn apply(&self, a: T, b: T) -> T {
        if b != T::from_f64(self.masking_value) {
            T::from_f64(self.outside_value)
        } else {
            a
        }
    }
}

/// `MaskNegatedImageFilter`: keep `img`'s pixel where the mask equals
/// `masking_value` (ITK default `0`), else `outside_value` (ITK default
/// `0`) -- the logical complement of [`mask`]. `mask_img` must share `img`'s
/// pixel type (see the module docs).
pub fn mask_negated(
    img: &Image,
    mask_img: &Image,
    outside_value: f64,
    masking_value: f64,
) -> Result<Image> {
    functor::binary_apply(
        img,
        mask_img,
        &MaskNegatedOp {
            outside_value,
            masking_value,
        },
    )
}

/// In-place variant of [`mask_negated`]: reuses `img`'s buffer.
pub fn mask_negated_in_place(
    img: Image,
    mask_img: &Image,
    outside_value: f64,
    masking_value: f64,
) -> Result<Image> {
    functor::binary_apply_in_place(
        img,
        mask_img,
        &MaskNegatedOp {
            outside_value,
            masking_value,
        },
    )
}

// ---- maximum / minimum (all pixel types) -----------------------------------

/// `Maximum` functor (`itkMaximumImageFilter.h`): `a > b ? a : b`, compared
/// directly in the pixel type (no promotion).
struct MaxOp;
impl<T: Scalar> BinaryFunctor<T> for MaxOp {
    fn apply(&self, a: T, b: T) -> T {
        if a > b { a } else { b }
    }
}

/// `Minimum` functor (`itkMinimumImageFilter.h`): `a < b ? a : b`, compared
/// directly in the pixel type (no promotion).
struct MinOp;
impl<T: Scalar> BinaryFunctor<T> for MinOp {
    fn apply(&self, a: T, b: T) -> T {
        if a < b { a } else { b }
    }
}

functor::binary_functor! {
    /// `MaximumImageFilter`: pixel-wise `max(a, b)`.
    pub fn maximum, maximum_in_place = MaxOp;
}

functor::binary_functor! {
    /// `MinimumImageFilter`: pixel-wise `min(a, b)`.
    pub fn minimum, minimum_in_place = MinOp;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- and / or / xor: 0 / MAX operand boundaries ----

    #[test]
    fn and_zero_and_max_operands() {
        let zero = img_u8(&[2, 1], vec![0, 0]);
        let max = img_u8(&[2, 1], vec![u8::MAX, u8::MAX]);
        let mixed = img_u8(&[2, 1], vec![0b1010_1010, 0b0101_0101]);
        assert_eq!(
            and(&zero, &max).unwrap().scalar_slice::<u8>().unwrap(),
            &[0, 0]
        );
        assert_eq!(
            and(&max, &max).unwrap().scalar_slice::<u8>().unwrap(),
            &[u8::MAX, u8::MAX]
        );
        assert_eq!(
            and(&mixed, &max).unwrap().scalar_slice::<u8>().unwrap(),
            &[0b1010_1010, 0b0101_0101]
        );
    }

    #[test]
    fn or_zero_and_max_operands() {
        let zero = img_u8(&[2, 1], vec![0, 0]);
        let max = img_u8(&[2, 1], vec![u8::MAX, u8::MAX]);
        assert_eq!(
            or(&zero, &max).unwrap().scalar_slice::<u8>().unwrap(),
            &[u8::MAX, u8::MAX]
        );
        assert_eq!(
            or(&zero, &zero).unwrap().scalar_slice::<u8>().unwrap(),
            &[0, 0]
        );
    }

    #[test]
    fn xor_zero_and_max_operands() {
        let zero = img_u8(&[2, 1], vec![0, 0]);
        let max = img_u8(&[2, 1], vec![u8::MAX, u8::MAX]);
        assert_eq!(
            xor(&zero, &max).unwrap().scalar_slice::<u8>().unwrap(),
            &[u8::MAX, u8::MAX]
        );
        assert_eq!(
            xor(&max, &max).unwrap().scalar_slice::<u8>().unwrap(),
            &[0, 0]
        );
    }

    #[test]
    fn bitwise_ops_reject_float_pixel_type() {
        let a = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            and(&a, &b),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
        assert_eq!(
            or(&a, &b),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
        assert_eq!(
            xor(&a, &b),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
        assert_eq!(
            not(&a),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
    }

    #[test]
    fn and_in_place_matches_allocating() {
        let a = img_u8(&[2, 1], vec![0b1100, 0b1010]);
        let b = img_u8(&[2, 1], vec![0b1010, 0b1100]);
        let allocated = and(&a, &b).unwrap();
        let in_place = and_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- not: 0 vs nonzero ----

    #[test]
    fn not_zero_and_nonzero() {
        let a = img_u8(&[3, 1], vec![0, 1, 255]);
        assert_eq!(not(&a).unwrap().scalar_slice::<u8>().unwrap(), &[1, 0, 0]);
    }

    #[test]
    fn not_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![0, 1, 255]);
        let allocated = not(&a).unwrap();
        let in_place = not_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- mask: mask zero vs nonzero vs explicit masking_value ----

    #[test]
    fn mask_default_masking_value_zero_keeps_where_nonzero() {
        let img = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask_img = img_u8(&[3, 1], vec![0, 1, 0]);
        let out = mask(&img, &mask_img, 0.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 20, 0]);
    }

    #[test]
    fn mask_explicit_masking_value() {
        let img = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask_img = img_u8(&[3, 1], vec![5, 5, 9]);
        // masking_value = 5: mask == 5 gets outside_value; mask != 5 (9)
        // keeps a.
        let out = mask(&img, &mask_img, 99.0, 5.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[99, 99, 30]);
    }

    #[test]
    fn mask_negated_is_complement_of_mask() {
        let img = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask_img = img_u8(&[3, 1], vec![0, 1, 0]);
        let masked = mask(&img, &mask_img, 0.0, 0.0).unwrap();
        let negated = mask_negated(&img, &mask_img, 0.0, 0.0).unwrap();
        assert_eq!(masked.scalar_slice::<u8>().unwrap(), &[0, 20, 0]);
        assert_eq!(negated.scalar_slice::<u8>().unwrap(), &[10, 0, 30]);
    }

    #[test]
    fn mask_in_place_matches_allocating() {
        let img = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask_img = img_u8(&[3, 1], vec![0, 1, 0]);
        let allocated = mask(&img, &mask_img, 7.0, 0.0).unwrap();
        let in_place = mask_in_place(img, &mask_img, 7.0, 0.0).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- maximum / minimum ----

    #[test]
    fn maximum_and_minimum_basic() {
        let a = img_u8(&[3, 1], vec![1, 20, 30]);
        let b = img_u8(&[3, 1], vec![10, 2, 30]);
        assert_eq!(
            maximum(&a, &b).unwrap().scalar_slice::<u8>().unwrap(),
            &[10, 20, 30]
        );
        assert_eq!(
            minimum(&a, &b).unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 2, 30]
        );
    }

    #[test]
    fn maximum_minimum_work_on_float_pixel_types() {
        let a = Image::from_vec(&[2, 1], vec![1.5f32, -2.5]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![-1.0f32, 3.0]).unwrap();
        assert_eq!(
            maximum(&a, &b).unwrap().scalar_slice::<f32>().unwrap(),
            &[1.5, 3.0]
        );
        assert_eq!(
            minimum(&a, &b).unwrap().scalar_slice::<f32>().unwrap(),
            &[-1.0, -2.5]
        );
    }

    #[test]
    fn maximum_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![1, 20, 30]);
        let b = img_u8(&[3, 1], vec![10, 2, 30]);
        let allocated = maximum(&a, &b).unwrap();
        let in_place = maximum_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }
}
