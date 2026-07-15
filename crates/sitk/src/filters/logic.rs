//! Logic and bitwise pixel filters for sitk-rs, verified against the ITK v6
//! source under `Modules/Filtering/ImageIntensity/include/`:
//! `itkBitwiseOpsFunctors.h`, `itkLogicOpsFunctors.h`, `itkAndImageFilter.h`,
//! `itkOrImageFilter.h`, `itkXorImageFilter.h`, `itkNotImageFilter.h`,
//! `itkMaskImageFilter.h`, `itkMaskNegatedImageFilter.h`,
//! `itkMaximumImageFilter.h`, `itkMinimumImageFilter.h`, and under
//! `Modules/Filtering/LabelMap/include/`: `itkBinaryNotImageFilter.h`.
//!
//! [`greater_equal`]/[`less_equal`]/[`not_equal`] (`itkLogicOpsFunctors.h`'s
//! `LogicOpBase`-derived `GreaterEqual`/`LessEqual`/`NotEqual`: `A op B ?
//! foreground_value : background_value`) are the crate's
//! [`crate::filters::functor::ComparisonFunctor`] policy: pixel-type-compute like
//! `and`/`or`/`xor`, but the output pixel type is always `UInt8` regardless
//! of the (shared) input pixel type, so there is no integer-only gate (every
//! pixel type this crate supports has a total order) and no in-place variant
//! (matching `divide_real`'s precedent in `math.rs`). ITK's own header also
//! declares sibling `Greater`/`Less`/`Equal` functors
//! (`GreaterImageFilter.yaml`/`LessImageFilter.yaml`/`EqualImageFilter.yaml`),
//! implemented below as [`equal`]/[`greater`]/[`less`] on the same
//! `ComparisonFunctor` policy. `Equal`'s `Math::ExactlyEquals(A,
//! static_cast<TInput1>(B))` (itkLogicOpsFunctors.h:130) is a
//! compiler-warning-suppressing wrapper around `==`, not a different
//! comparison, and the `static_cast<TInput1>(B)` is a no-op here since both
//! operands share the crate's single generic `T`.
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
//! [`bitwise_not`] is the unary face of the same policy (Rust's `!` on an
//! integer is C++'s `~`, bitwise complement), built on
//! [`crate::filters::functor::UnaryPixelFunctor`] instead of [`BinaryFunctor`].
//!
//! [`not`] (`itkLogicOpsFunctors.h`'s `Functor::NOT`, integer-only for the
//! same reason) maps to `0`/`1`, not a bitwise complement: `if (!A) return
//! ForegroundValue(1); return BackgroundValue(0);`. The only operation is an
//! exact equality-to-zero test, which `f64` promotion preserves exactly for
//! every integer pixel type this crate supports, so it's implemented on the
//! [`UnaryFunctor`] (f64-compute) engine instead of adding a new
//! single-image pixel-type-compute engine for one filter.
//!
//! [`binary_not`] (`itkBinaryNotImageFilter.h`, `ITKLabelMap` module,
//! integer-only for the same reason as `and`/`or`/`xor`) is unrelated to
//! [`bitwise_not`] despite the name: it flips a single image's pixels
//! between `foreground_value` and `background_value` (`A ==
//! ForegroundValue ? BackgroundValue : ForegroundValue`), so it runs on the
//! pixel-type-compute engine like `mask`/`mask_negated` below rather than
//! `bitwise_not`'s bit-complement. ITK's own C++ default is
//! `ForegroundValue = NumericTraits<PixelType>::max()`, `BackgroundValue =
//! NumericTraits<PixelType>::NonpositiveMin()`
//! (`itkBinaryNotImageFilter.h`'s constructor); SimpleITK's yaml overrides
//! this with its own default of `ForegroundValue = 1.0`, `BackgroundValue =
//! 0.0` (`BinaryNotImageFilter.yaml`), set unconditionally by the generated
//! wrapper regardless of the ITK-level default. This port takes both values
//! as required parameters instead of hard-coding either default, matching
//! [`mask`]/[`mask_negated`]'s precedent. Separately,
//! `BinaryNotImageFilter.yaml`'s prose description ("output_pixel =
//! static_cast<PixelType>( input1_pixel != input2_pixel )") describes a
//! *two*-image `!=` comparison and does not match either the single-image
//! `operator()` in `itkBinaryNotImageFilter.h` or `number_of_inputs: 1` in
//! the same yaml file -- an upstream copy-paste error in the yaml's prose,
//! not a behavior this port reproduces.
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
//!
//! [`masked_assign`]/[`masked_assign_constant`] (`itkMaskedAssignImageFilter.h`/
//! `.hxx`, `MaskedAssignImageFilter.yaml`): pixel-wise `mask != 0 ? assign :
//! image`, a `TernaryGeneratorImageFilter` with no numeric computation at
//! all, just a per-pixel selection. Unlike `mask`/`mask_negated` immediately
//! above, the mask's pixel type here is not cast to the main image's type --
//! `MaskedAssignImageFilter.yaml`'s `filter_type` fixes the ITK mask template
//! parameter to `itk::Image<std::uint8_t, ...>` outright, with no fallback
//! casting path the way `MaskImageFilter.yaml` has (see
//! [`FilterError::RequiresUInt8MaskPixelType`]'s doc for the distinction).
//! `assign` (an `&Image`) must share `image`'s pixel type and size exactly,
//! since ITK's `AssignImageType` defaults to `OutputImageType`, which
//! defaults to `InputImageType`, with no cast path for a mismatch either.
//! [`masked_assign_constant`] is `SetAssignConstant`: a `double` narrowed to
//! `image`'s pixel type via `Scalar::from_f64`, used pixel-wise in place of
//! an `AssignImage`.

use crate::core::{Image, Scalar, dispatch_scalar};
use crate::filters::functor::{
    self, BinaryFunctor, ComparisonFunctor, UnaryFunctor, UnaryPixelFunctor,
};
use crate::filters::{FilterError, Result, require_same_shape};

pub(crate) fn require_integer_pixel_type(img: &Image) -> Result<()> {
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

// ---- BitwiseNot (integer pixel types only) ---------------------------------

/// `BitwiseNot` functor (`itkBitwiseOpsFunctors.h`): `~a`, evaluated
/// directly in the pixel type (Rust's `!` on an integer is C++'s unary `~`).
/// Integer pixel types only, for the same reason as `and`/`or`/`xor`.
struct BitwiseNotOp;

macro_rules! impl_bitwise_not_int {
    ($($t:ty),+ $(,)?) => {$(
        impl UnaryPixelFunctor<$t> for BitwiseNotOp {
            fn apply(&self, x: $t) -> $t { !x }
        }
    )+};
}

impl_bitwise_not_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl UnaryPixelFunctor<f32> for BitwiseNotOp {
    fn apply(&self, _x: f32) -> f32 {
        unreachable!("gated to integer pixel types by require_integer_pixel_type")
    }
}
impl UnaryPixelFunctor<f64> for BitwiseNotOp {
    fn apply(&self, _x: f64) -> f64 {
        unreachable!("gated to integer pixel types by require_integer_pixel_type")
    }
}

/// `BitwiseNotImageFilter`: pixel-wise `~a`. Integer pixel types only (see
/// the module docs); errors with [`FilterError::RequiresIntegerPixelType`]
/// on a floating-point image.
pub fn bitwise_not(img: &Image) -> Result<Image> {
    require_integer_pixel_type(img)?;
    functor::unary_pixel_apply(img, &BitwiseNotOp)
}

/// In-place variant of [`bitwise_not`]: reuses `img`'s buffer.
pub fn bitwise_not_in_place(img: Image) -> Result<Image> {
    require_integer_pixel_type(&img)?;
    functor::unary_pixel_apply_in_place(img, &BitwiseNotOp)
}

// ---- BinaryNot (integer pixel types only) ----------------------------------

/// `BinaryNot` functor (`itkBinaryNotImageFilter.h`): `a == foreground_value
/// ? background_value : foreground_value` -- flips between the two values;
/// see the module docs for the semantics, the ITK-vs-SimpleITK default
/// divergence, and why this is unrelated to [`bitwise_not`] despite the
/// name. Integer pixel types only, for the same reason as `and`/`or`/`xor`.
struct BinaryNotOp {
    foreground_value: f64,
    background_value: f64,
}
impl<T: Scalar> UnaryPixelFunctor<T> for BinaryNotOp {
    fn apply(&self, x: T) -> T {
        if x == T::from_f64(self.foreground_value) {
            T::from_f64(self.background_value)
        } else {
            T::from_f64(self.foreground_value)
        }
    }
}

/// `BinaryNotImageFilter`: pixel-wise flip between `foreground_value` and
/// `background_value` (see the module docs). Integer pixel types only;
/// errors with [`FilterError::RequiresIntegerPixelType`] on a
/// floating-point image.
pub fn binary_not(img: &Image, foreground_value: f64, background_value: f64) -> Result<Image> {
    require_integer_pixel_type(img)?;
    functor::unary_pixel_apply(
        img,
        &BinaryNotOp {
            foreground_value,
            background_value,
        },
    )
}

/// In-place variant of [`binary_not`]: reuses `img`'s buffer.
pub fn binary_not_in_place(
    img: Image,
    foreground_value: f64,
    background_value: f64,
) -> Result<Image> {
    require_integer_pixel_type(&img)?;
    functor::unary_pixel_apply_in_place(
        img,
        &BinaryNotOp {
            foreground_value,
            background_value,
        },
    )
}

// ---- comparisons (all pixel types, u8 output) ------------------------------

/// `GreaterEqual` functor (`itkLogicOpsFunctors.h`): `a >= b ?
/// foreground_value : background_value`, compared directly in the pixel
/// type (no promotion; see the module docs).
struct GreaterEqualOp {
    foreground_value: u8,
    background_value: u8,
}
impl<T: Scalar> ComparisonFunctor<T> for GreaterEqualOp {
    fn apply(&self, a: T, b: T) -> u8 {
        if a >= b {
            self.foreground_value
        } else {
            self.background_value
        }
    }
}

/// `LessEqual` functor (`itkLogicOpsFunctors.h`): `a <= b ? foreground_value
/// : background_value`, compared directly in the pixel type (no promotion;
/// see the module docs).
struct LessEqualOp {
    foreground_value: u8,
    background_value: u8,
}
impl<T: Scalar> ComparisonFunctor<T> for LessEqualOp {
    fn apply(&self, a: T, b: T) -> u8 {
        if a <= b {
            self.foreground_value
        } else {
            self.background_value
        }
    }
}

/// `NotEqual` functor (`itkLogicOpsFunctors.h`): `a != b ? foreground_value
/// : background_value`, compared directly in the pixel type (no promotion;
/// see the module docs).
struct NotEqualOp {
    foreground_value: u8,
    background_value: u8,
}
impl<T: Scalar> ComparisonFunctor<T> for NotEqualOp {
    fn apply(&self, a: T, b: T) -> u8 {
        if a != b {
            self.foreground_value
        } else {
            self.background_value
        }
    }
}

functor::comparison_functor! {
    /// `GreaterEqualImageFilter`: pixel-wise `a >= b ? foreground_value :
    /// background_value`. Output is always `UInt8`.
    pub fn greater_equal(foreground_value: u8, background_value: u8) = GreaterEqualOp { foreground_value, background_value };
}

functor::comparison_functor! {
    /// `LessEqualImageFilter`: pixel-wise `a <= b ? foreground_value :
    /// background_value`. Output is always `UInt8`.
    pub fn less_equal(foreground_value: u8, background_value: u8) = LessEqualOp { foreground_value, background_value };
}

functor::comparison_functor! {
    /// `NotEqualImageFilter`: pixel-wise `a != b ? foreground_value :
    /// background_value`. Output is always `UInt8`.
    pub fn not_equal(foreground_value: u8, background_value: u8) = NotEqualOp { foreground_value, background_value };
}

/// `Equal` functor (`itkLogicOpsFunctors.h`): `a == b ? foreground_value :
/// background_value`, compared directly in the pixel type (no promotion; see
/// the module docs).
struct EqualOp {
    foreground_value: u8,
    background_value: u8,
}
impl<T: Scalar> ComparisonFunctor<T> for EqualOp {
    fn apply(&self, a: T, b: T) -> u8 {
        if a == b {
            self.foreground_value
        } else {
            self.background_value
        }
    }
}

/// `Greater` functor (`itkLogicOpsFunctors.h`): `a > b ? foreground_value :
/// background_value`, compared directly in the pixel type (no promotion; see
/// the module docs).
struct GreaterOp {
    foreground_value: u8,
    background_value: u8,
}
impl<T: Scalar> ComparisonFunctor<T> for GreaterOp {
    fn apply(&self, a: T, b: T) -> u8 {
        if a > b {
            self.foreground_value
        } else {
            self.background_value
        }
    }
}

/// `Less` functor (`itkLogicOpsFunctors.h`): `a < b ? foreground_value :
/// background_value`, compared directly in the pixel type (no promotion; see
/// the module docs).
struct LessOp {
    foreground_value: u8,
    background_value: u8,
}
impl<T: Scalar> ComparisonFunctor<T> for LessOp {
    fn apply(&self, a: T, b: T) -> u8 {
        if a < b {
            self.foreground_value
        } else {
            self.background_value
        }
    }
}

functor::comparison_functor! {
    /// `EqualImageFilter`: pixel-wise `a == b ? foreground_value :
    /// background_value`. Output is always `UInt8`.
    pub fn equal(foreground_value: u8, background_value: u8) = EqualOp { foreground_value, background_value };
}

functor::comparison_functor! {
    /// `GreaterImageFilter`: pixel-wise `a > b ? foreground_value :
    /// background_value`. Output is always `UInt8`.
    pub fn greater(foreground_value: u8, background_value: u8) = GreaterOp { foreground_value, background_value };
}

functor::comparison_functor! {
    /// `LessImageFilter`: pixel-wise `a < b ? foreground_value :
    /// background_value`. Output is always `UInt8`.
    pub fn less(foreground_value: u8, background_value: u8) = LessOp { foreground_value, background_value };
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

// ---- masked_assign (image, UInt8 mask, assign image or constant) ----------

fn masked_assign_typed<T: Scalar>(image: &Image, mask: &[u8], assign: &Image) -> Result<Image> {
    let s = image.scalar_slice::<T>()?;
    let a = assign.scalar_slice::<T>()?;
    let out: Vec<T> = s
        .iter()
        .zip(mask)
        .zip(a)
        .map(|((&px, &m), &av)| if m != 0 { av } else { px })
        .collect();
    let mut out_img = Image::from_vec(image.size(), out)?;
    out_img.copy_geometry_from(image);
    Ok(out_img)
}

fn masked_assign_typed_in_place<T: Scalar>(
    mut image: Image,
    mask: &[u8],
    assign: &Image,
) -> Result<Image> {
    let a = assign.scalar_slice::<T>()?;
    let v = image.scalar_vec_mut::<T>()?;
    for ((x, &m), &av) in v.iter_mut().zip(mask).zip(a) {
        if m != 0 {
            *x = av;
        }
    }
    Ok(image)
}

/// `MaskedAssignImageFilter`: pixel-wise `mask != 0 ? assign : image` (see
/// the module docs). `mask` must be `UInt8`, share `image`'s size and sit on `image`'s
/// grid ([`crate::filters::mask_input::uint8_mask_voxels`]); `assign` must share `image`'s
/// pixel type and size.
pub fn masked_assign(image: &Image, mask: &Image, assign: &Image) -> Result<Image> {
    let mask_bytes = crate::filters::mask_input::uint8_mask_voxels(image, mask)?;
    require_same_shape(image, assign)?;
    dispatch_scalar!(
        image.pixel_id(),
        masked_assign_typed,
        image,
        mask_bytes,
        assign
    )
}

/// In-place variant of [`masked_assign`]: reuses `image`'s buffer.
pub fn masked_assign_in_place(image: Image, mask: &Image, assign: &Image) -> Result<Image> {
    let mask_bytes = crate::filters::mask_input::uint8_mask_voxels(&image, mask)?;
    require_same_shape(&image, assign)?;
    dispatch_scalar!(
        image.pixel_id(),
        masked_assign_typed_in_place,
        image,
        mask_bytes,
        assign
    )
}

fn masked_assign_constant_typed<T: Scalar>(
    image: &Image,
    mask: &[u8],
    assign_constant: f64,
) -> Result<Image> {
    let s = image.scalar_slice::<T>()?;
    let av = T::from_f64(assign_constant);
    let out: Vec<T> = s
        .iter()
        .zip(mask)
        .map(|(&px, &m)| if m != 0 { av } else { px })
        .collect();
    let mut out_img = Image::from_vec(image.size(), out)?;
    out_img.copy_geometry_from(image);
    Ok(out_img)
}

fn masked_assign_constant_typed_in_place<T: Scalar>(
    mut image: Image,
    mask: &[u8],
    assign_constant: f64,
) -> Result<Image> {
    let av = T::from_f64(assign_constant);
    let v = image.scalar_vec_mut::<T>()?;
    for (x, &m) in v.iter_mut().zip(mask) {
        if m != 0 {
            *x = av;
        }
    }
    Ok(image)
}

/// `MaskedAssignImageFilter` with a constant `assign` value in place of an
/// `AssignImage` (`SetAssignConstant`, `AssignConstant` in
/// `MaskedAssignImageFilter.yaml`): pixel-wise `mask != 0 ? assign_constant :
/// image`. `assign_constant` is narrowed to `image`'s pixel type via
/// `Scalar::from_f64` before use (see the module docs). `mask` must be `UInt8`,
/// share `image`'s size and sit on `image`'s grid.
pub fn masked_assign_constant(image: &Image, mask: &Image, assign_constant: f64) -> Result<Image> {
    let mask_bytes = crate::filters::mask_input::uint8_mask_voxels(image, mask)?;
    dispatch_scalar!(
        image.pixel_id(),
        masked_assign_constant_typed,
        image,
        mask_bytes,
        assign_constant
    )
}

/// In-place variant of [`masked_assign_constant`]: reuses `image`'s buffer.
pub fn masked_assign_constant_in_place(
    image: Image,
    mask: &Image,
    assign_constant: f64,
) -> Result<Image> {
    let mask_bytes = crate::filters::mask_input::uint8_mask_voxels(&image, mask)?;
    dispatch_scalar!(
        image.pixel_id(),
        masked_assign_constant_typed_in_place,
        image,
        mask_bytes,
        assign_constant
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

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

    // ---- bitwise_not: 0 / MAX / mixed-bits boundaries ----

    #[test]
    fn bitwise_not_zero_max_and_mixed() {
        let a = img_u8(&[3, 1], vec![0, u8::MAX, 0b1010_1010]);
        assert_eq!(
            bitwise_not(&a).unwrap().scalar_slice::<u8>().unwrap(),
            &[u8::MAX, 0, 0b0101_0101]
        );
    }

    #[test]
    fn bitwise_not_rejects_float_pixel_type() {
        let a = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        assert_eq!(
            bitwise_not(&a),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
    }

    #[test]
    fn bitwise_not_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![0, u8::MAX, 0b1010_1010]);
        let allocated = bitwise_not(&a).unwrap();
        let in_place = bitwise_not_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- binary_not: equals-foreground vs not, default-like vs custom values ----

    #[test]
    fn binary_not_flips_between_foreground_and_background() {
        // SimpleITK's own yaml default is foreground=1.0, background=0.0.
        let a = img_u8(&[3, 1], vec![0, 1, 5]);
        assert_eq!(
            binary_not(&a, 1.0, 0.0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[1, 0, 1]
        );
    }

    #[test]
    fn binary_not_custom_foreground_and_background() {
        let a = img_u8(&[3, 1], vec![100, 7, 3]);
        assert_eq!(
            binary_not(&a, 100.0, 7.0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[7, 100, 100]
        );
    }

    #[test]
    fn binary_not_rejects_float_pixel_type() {
        let a = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        assert_eq!(
            binary_not(&a, 1.0, 0.0),
            Err(FilterError::RequiresIntegerPixelType(a.pixel_id()))
        );
    }

    #[test]
    fn binary_not_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![0, 1, 5]);
        let allocated = binary_not(&a, 1.0, 0.0).unwrap();
        let in_place = binary_not_in_place(a, 1.0, 0.0).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- greater_equal / less_equal / not_equal: >, ==, < boundaries ----

    #[test]
    fn greater_equal_above_equal_and_below_boundaries() {
        let a = Image::from_vec(&[3, 1], vec![5i32, 3, 3]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![3i32, 3, 5]).unwrap();
        assert_eq!(
            greater_equal(&a, &b, 1, 0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[1, 1, 0]
        );
    }

    #[test]
    fn less_equal_above_equal_and_below_boundaries() {
        let a = Image::from_vec(&[3, 1], vec![5i32, 3, 3]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![3i32, 3, 5]).unwrap();
        assert_eq!(
            less_equal(&a, &b, 1, 0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[0, 1, 1]
        );
    }

    #[test]
    fn not_equal_equal_and_unequal_boundaries() {
        let a = Image::from_vec(&[3, 1], vec![1i32, 2, 2]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![1i32, 3, 2]).unwrap();
        assert_eq!(
            not_equal(&a, &b, 1, 0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[0, 1, 0]
        );
    }

    #[test]
    fn comparisons_use_custom_foreground_and_background_values() {
        let a = Image::from_vec(&[2, 1], vec![5i32, 3]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![3i32, 5]).unwrap();
        assert_eq!(
            greater_equal(&a, &b, 100, 7)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[100, 7]
        );
    }

    #[test]
    fn comparisons_output_is_always_uint8_regardless_of_input_type() {
        let a = Image::from_vec(&[2, 1], vec![5.0f32, 3.0]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![3.0f32, 5.0]).unwrap();
        let out = greater_equal(&a, &b, 1, 0).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::UInt8);
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 0]);
    }

    #[test]
    fn comparisons_reject_mismatched_pixel_types() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            greater_equal(&a, &b, 1, 0),
            Err(FilterError::TypeMismatch {
                a: a.pixel_id(),
                b: b.pixel_id()
            })
        );
    }

    // ---- equal / greater / less: ==, >, < boundaries, defaults, guards ----

    #[test]
    fn equal_matching_and_differing_pixels() {
        let a = Image::from_vec(&[3, 1], vec![1i32, 2, 2]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![1i32, 3, 2]).unwrap();
        assert_eq!(
            equal(&a, &b, 1, 0).unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 0, 1]
        );
    }

    #[test]
    fn greater_above_equal_and_below_boundaries() {
        let a = Image::from_vec(&[3, 1], vec![5i32, 3, 3]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![3i32, 3, 5]).unwrap();
        assert_eq!(
            greater(&a, &b, 1, 0).unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 0, 0]
        );
    }

    #[test]
    fn less_above_equal_and_below_boundaries() {
        let a = Image::from_vec(&[3, 1], vec![5i32, 3, 3]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![3i32, 3, 5]).unwrap();
        assert_eq!(
            less(&a, &b, 1, 0).unwrap().scalar_slice::<u8>().unwrap(),
            &[0, 0, 1]
        );
    }

    #[test]
    fn equal_greater_less_use_yaml_default_foreground_1_background_0() {
        // EqualImageFilter.yaml / GreaterImageFilter.yaml / LessImageFilter.yaml:
        // ForegroundValue default 1u, BackgroundValue default 0u.
        let a = Image::from_vec(&[2, 1], vec![5i32, 3]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![3i32, 5]).unwrap();
        assert_eq!(
            greater(&a, &b, 1, 0).unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 0]
        );
        assert_eq!(
            less(&a, &b, 1, 0).unwrap().scalar_slice::<u8>().unwrap(),
            &[0, 1]
        );
    }

    #[test]
    fn equal_greater_less_output_is_always_uint8_regardless_of_input_type() {
        let a = Image::from_vec(&[2, 1], vec![5.0f32, 3.0]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![3.0f32, 5.0]).unwrap();
        for out in [
            greater(&a, &b, 1, 0).unwrap(),
            less(&a, &b, 1, 0).unwrap(),
            equal(&a, &b, 1, 0).unwrap(),
        ] {
            assert_eq!(out.pixel_id(), crate::core::PixelId::UInt8);
        }
    }

    #[test]
    fn equal_greater_less_reject_a_complex_pixel_type() {
        let a = Image::new(&[2, 1], PixelId::ComplexFloat32);
        let b = Image::new(&[2, 1], PixelId::ComplexFloat32);
        let expected = || {
            Err(FilterError::Core(
                crate::core::Error::RequiresScalarPixelType(PixelId::ComplexFloat32),
            ))
        };
        assert_eq!(equal(&a, &b, 1, 0), expected());
        assert_eq!(greater(&a, &b, 1, 0), expected());
        assert_eq!(less(&a, &b, 1, 0), expected());
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

    // ---- masked_assign: mask zero vs nonzero boundaries and error paths ----

    #[test]
    fn masked_assign_selects_by_mask() {
        let image = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask = img_u8(&[3, 1], vec![0, 1, 0]);
        let assign = img_u8(&[3, 1], vec![100, 200, 250]);
        assert_eq!(
            masked_assign(&image, &mask, &assign)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[10, 200, 30]
        );
    }

    #[test]
    fn masked_assign_in_place_matches_allocating() {
        let image = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask = img_u8(&[3, 1], vec![0, 1, 0]);
        let assign = img_u8(&[3, 1], vec![100, 200, 250]);
        let allocated = masked_assign(&image, &mask, &assign).unwrap();
        let in_place = masked_assign_in_place(image, &mask, &assign).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn masked_assign_rejects_non_uint8_mask() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let mask = Image::from_vec(&[2, 1], vec![0i32, 1]).unwrap();
        let assign = img_u8(&[2, 1], vec![100, 200]);
        assert_eq!(
            masked_assign(&image, &mask, &assign),
            Err(FilterError::RequiresUInt8MaskPixelType(mask.pixel_id()))
        );
    }

    #[test]
    fn masked_assign_rejects_mismatched_mask_size() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let mask = img_u8(&[3, 1], vec![0, 1, 0]);
        let assign = img_u8(&[2, 1], vec![100, 200]);
        assert_eq!(
            masked_assign(&image, &mask, &assign),
            Err(FilterError::SizeMismatch {
                a: image.size().to_vec(),
                b: mask.size().to_vec(),
            })
        );
    }

    /// `MaskedAssignImageFilter` is a `TernaryGeneratorImageFilter`, so the mask is
    /// `SetInput2` — a pipeline input, whose physical space
    /// `ImageToImageFilter::VerifyInputInformation` compares against the image's. The
    /// aligned mask is accepted first, or the refusal below proves nothing.
    #[test]
    fn masked_assign_rejects_a_mask_on_a_different_grid() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let assign = img_u8(&[2, 1], vec![100, 200]);

        let aligned = img_u8(&[2, 1], vec![0, 1]);
        masked_assign(&image, &aligned, &assign)
            .expect("an aligned mask must be accepted, or the refusal below proves nothing");

        let mut shifted = img_u8(&[2, 1], vec![0, 1]);
        shifted.set_origin(&[5.0, 0.0]).unwrap();
        assert_eq!(
            masked_assign(&image, &shifted, &assign),
            Err(FilterError::PhysicalSpaceMismatch { index: 1 })
        );
        assert_eq!(
            masked_assign_constant(&image, &shifted, 7.0),
            Err(FilterError::PhysicalSpaceMismatch { index: 1 })
        );
    }

    #[test]
    fn masked_assign_rejects_mismatched_assign_pixel_type() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let mask = img_u8(&[2, 1], vec![0, 1]);
        let assign = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            masked_assign(&image, &mask, &assign),
            Err(FilterError::TypeMismatch {
                a: image.pixel_id(),
                b: assign.pixel_id(),
            })
        );
    }

    #[test]
    fn masked_assign_rejects_mismatched_assign_size() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let mask = img_u8(&[2, 1], vec![0, 1]);
        let assign = img_u8(&[3, 1], vec![100, 200, 250]);
        assert_eq!(
            masked_assign(&image, &mask, &assign),
            Err(FilterError::SizeMismatch {
                a: image.size().to_vec(),
                b: assign.size().to_vec(),
            })
        );
    }

    // ---- masked_assign_constant ----

    #[test]
    fn masked_assign_constant_selects_by_mask() {
        let image = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask = img_u8(&[3, 1], vec![0, 1, 1]);
        assert_eq!(
            masked_assign_constant(&image, &mask, 99.0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[10, 99, 99]
        );
    }

    #[test]
    fn masked_assign_constant_saturates_out_of_range_value() {
        // 300.0 does not fit in u8; Scalar::from_f64 saturates to 255,
        // matching ToPixelType's narrowing in MaskedAssignImageFilter.yaml.
        let image = img_u8(&[1, 1], vec![10]);
        let mask = img_u8(&[1, 1], vec![1]);
        assert_eq!(
            masked_assign_constant(&image, &mask, 300.0)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[255]
        );
    }

    #[test]
    fn masked_assign_constant_in_place_matches_allocating() {
        let image = img_u8(&[3, 1], vec![10, 20, 30]);
        let mask = img_u8(&[3, 1], vec![0, 1, 1]);
        let allocated = masked_assign_constant(&image, &mask, 99.0).unwrap();
        let in_place = masked_assign_constant_in_place(image, &mask, 99.0).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn masked_assign_constant_rejects_non_uint8_mask() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let mask = Image::from_vec(&[2, 1], vec![0i32, 1]).unwrap();
        assert_eq!(
            masked_assign_constant(&image, &mask, 0.0),
            Err(FilterError::RequiresUInt8MaskPixelType(mask.pixel_id()))
        );
    }

    #[test]
    fn masked_assign_constant_rejects_mismatched_mask_size() {
        let image = img_u8(&[2, 1], vec![10, 20]);
        let mask = img_u8(&[3, 1], vec![0, 1, 0]);
        assert_eq!(
            masked_assign_constant(&image, &mask, 0.0),
            Err(FilterError::SizeMismatch {
                a: image.size().to_vec(),
                b: mask.size().to_vec(),
            })
        );
    }
}
