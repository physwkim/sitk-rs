//! `std::ops` operator overloads for [`Image`], porting
//! `Code/BasicFilters/include/sitkImageOperators.h:44-269` — the 44
//! non-assignment operators (`+ - * / % - ~ & | ^`; the compound-assignment
//! operators at `sitkImageOperators.h:272-367` are a separate, unported
//! surface: SimpleITK's `+=` etc. mutate through `Image::ProxyForInPlaceOperation`,
//! which has no counterpart here).
//!
//! # Why this lives in `sitk-core`, not `sitk-filters`
//!
//! The task that produced this module was to "delegate to the existing
//! sitk-filters functions." That is not possible here, for two independent,
//! hard reasons:
//!
//! 1. **Orphan rule.** `impl std::ops::Add for Image` can only be written in
//!    the crate that defines `Image` — `sitk-core`. Neither `sitk-filters`
//!    nor the `sitk` facade crate own `Image` (re-exporting it via `pub use`
//!    does not confer ownership for coherence purposes), so the impl cannot
//!    legally live anywhere else.
//! 2. **Acyclic crate graph.** `sitk-filters` already depends on `sitk-core`.
//!    If these impls lived in `sitk-core` but called `sitk_filters::add` etc.,
//!    `sitk-core` would have to depend on `sitk-filters`, which depends back
//!    on `sitk-core` — Cargo rejects the cycle.
//!
//! So this module is a **self-contained reimplementation** of the same
//! verified policy `sitk-filters`'s `add`/`subtract`/`multiply`/`divide`/
//! `modulus`/`unary_minus`/`and`/`or`/`xor` use (pixel-type-compute,
//! `crates/sitk-filters/src/lib.rs`'s `AddOp`/`SubOp`/`MulOp`/`DivOp`/`ModOp`/
//! `UnaryMinusOp` and `crates/sitk-filters/src/logic.rs`'s `AndOp`/`OrOp`/
//! `XorOp`/`BitwiseNotOp`), built directly against `sitk-core`'s own
//! `Scalar`/`dispatch_scalar!` primitives instead of calling those functions.
//! A future migration of the shared functor engine down into `sitk-core`
//! (so both crates delegate to one copy) is a larger, separately-scoped
//! structural change — not done here.
//!
//! # Operator ⊕ scalar constant: verified against the actual codegen, not
//! against `sitk-filters`'s existing `*_constant` functions
//!
//! `sitk-filters::add_constant`/`subtract_constant`/`multiply_constant`
//! promote to `f64` and narrow back with a saturating cast, and their module
//! doc claims this "matches SimpleITK's double-constant functor". Checking
//! the actual codegen template that produces `Add(image, double)` etc.
//! (`SimpleITK/Code/BasicFilters/templates/sitkBinaryFunctorFilterTemplate.cxx.jinja:140-147`)
//! shows that claim is wrong: the `double` constant is converted to the
//! **image's pixel type first** —
//! `ToPixelType(constant, c); filter->SetConstant2(c);` — where `ToPixelType`
//! (`SimpleITK/Code/BasicFilters/src/sitkToPixelType.hxx:31-36`) is a plain
//! `static_cast<TPixel>(inPixel)`. The already-pixel-typed constant is then
//! run through the *same* wrapping pixel-type functor
//! (`Add2`/`Sub2`/`Mult`/`Div`/`Modulus`/`AND`/`OR`/`XOR`) as the image ⊕
//! image case — cast-to-pixel-type-first, then wrap, not
//! accumulate-in-`f64`-then-saturate. This holds uniformly for every
//! `BinaryFunctorFilter`-templated filter regardless of `constant_type`
//! (`double` for Add/Subtract/Multiply/Divide, `uint32_t` for Modulus, `int`
//! for And/Or/Xor — all four yamls set `template_code_filename:
//! BinaryFunctorFilter`).
//!
//! The operators below implement this verified policy directly: the constant
//! is cast to the pixel type via [`Scalar::from_f64`] (this crate's
//! saturating stand-in for C++ `static_cast`'s undefined behavior on
//! out-of-range floats, the same convention `sitk-filters` itself already
//! uses everywhere else a pixel-typed cast is needed), and the *same*
//! wrapping op then runs between that value and each pixel — uniformly with
//! the image ⊕ image case. This is **more** correct than
//! `sitk-filters::add_constant` et al., not less, and it means `img + 3.7`
//! (this module) and `add_constant(&img, 3.7)` (`sitk-filters`) can now
//! disagree on a fractional or out-of-range constant. `add_constant`/
//! `subtract_constant`/`multiply_constant` are not changed here — that is a
//! separate, potentially-breaking fix to already-shipped, tested behavior;
//! tracked as an open decision point (`doc/upstream-findings.md` §5.14,
//! divergence recorded at §4.43).
//!
//! `modulus`/`and`/`or`/`xor` have no `sitk-filters` `*_constant` counterpart
//! at all (grep-verified absent), so there is no existing behavior to
//! disagree with for those four.
//!
//! # Scalar pixel types only
//!
//! Like every `sitk-filters` filter built on the pixel-type-compute functor
//! seam, these operators read pixels through [`Image::scalar_slice`] /
//! [`Image::scalar_vec_mut`], which reject a vector or complex image with
//! [`Error::RequiresScalarPixelType`] even though upstream's `Add`/`Subtract`/
//! `Multiply` accept a broader pixel type list
//! (`AddImageFilter.yaml`'s `pixel_types: NonLabelPixelIDTypeList` includes
//! vector and complex). This is the same restriction `sitk_filters::add` /
//! `subtract` / `multiply` already have today, not a new gap introduced by
//! this module.
//!
//! # Panics, not `Result`
//!
//! `sitkImageOperators.h`'s operators are `inline` wrappers with no error
//! handling of their own — a shape/type mismatch or a disallowed pixel type
//! (e.g. `%`/`&`/`|`/`^`/unary `-` on the wrong kind of image) throws a C++
//! exception from inside `Execute`. A Rust operator trait's `fn add(self,
//! rhs) -> Image` has no `Result` in its signature to report that with, so
//! every one of these impls **panics** in the situations where SimpleITK
//! throws — matching the parity behavior (a construct that cannot proceed
//! stops the program either way) rather than returning a nonsensical
//! `Image`. Callers who want a `Result` should use the named functions in
//! [`sitk_filters`](https://docs.rs/sitk-filters) instead of the operator
//! syntax: `sitk_filters::add`/`subtract`/`multiply`/`divide`/`modulus`/
//! `unary_minus`/`and`/`or`/`xor`/`bitwise_not`.

use crate::dispatch_scalar;
use crate::image::Image;
use crate::pixel::Scalar;

// ---- panicking preconditions (SimpleITK throws; operators have no `Result`) ----

fn require_same_shape(a: &Image, b: &Image) {
    assert!(
        a.pixel_id() == b.pixel_id(),
        "sitk-core::ops: pixel type mismatch ({:?} vs {:?}); SimpleITK's operators throw here \
         -- cast with `sitk_filters::cast` first, or use a named `sitk_filters` function for a \
         `Result` instead of a panic",
        a.pixel_id(),
        b.pixel_id()
    );
    assert!(
        a.size() == b.size(),
        "sitk-core::ops: size mismatch ({:?} vs {:?}); SimpleITK's operators throw here",
        a.size(),
        b.size()
    );
}

/// `Modulus`/`And`/`Or`/`Xor`/`BitwiseNot`'s integer-only gate, matching
/// `sitk_filters::logic::require_integer_pixel_type`'s exact semantics
/// (`is_floating_point()`, not `is_integer_scalar()`, so a vector or complex
/// integer pixel id passes this check and is rejected downstream by
/// [`Image::scalar_slice`] instead, with a more specific error).
fn require_integer(img: &Image) {
    assert!(
        !img.pixel_id().is_floating_point(),
        "sitk-core::ops: requires an integer pixel type, got {:?}; SimpleITK's Modulus/And/Or/Xor/\
         BitwiseNot operators throw here -- use `sitk_filters::{{modulus,and,or,xor,bitwise_not}}` \
         for a `Result` instead of a panic",
        img.pixel_id()
    );
}

/// Unary `-`'s signed-only gate, matching
/// `sitk_filters::require_signed_pixel_type`'s exact semantics.
fn require_signed(img: &Image) {
    assert!(
        img.pixel_id().is_signed(),
        "sitk-core::ops: unary `-` requires a signed pixel type, got {:?}; SimpleITK's \
         UnaryMinus operator throws here -- use `sitk_filters::unary_minus` for a `Result` \
         instead of a panic",
        img.pixel_id()
    );
}

fn no_gate(_img: &Image) {}

fn scalar_slice_or_panic<T: Scalar>(img: &Image) -> &[T] {
    match img.scalar_slice::<T>() {
        Ok(s) => s,
        Err(e) => panic!("sitk-core::ops: {e}"),
    }
}

fn scalar_vec_mut_or_panic<T: Scalar>(img: &mut Image) -> &mut Vec<T> {
    match img.scalar_vec_mut::<T>() {
        Ok(v) => v,
        Err(e) => panic!("sitk-core::ops: {e}"),
    }
}

// ---- the pixel-type-compute dispatch engine --------------------------------
//
// A minimal, self-contained copy of `sitk-filters/src/functor.rs`'s
// `BinaryFunctor`/`UnaryPixelFunctor` pixel-type-compute engine (see the
// module docs for why `sitk-core` cannot depend on that crate instead).
// There is no f64-compute engine here: every operator this module implements
// -- including every constant case -- is pixel-type-compute (see the module
// docs' `ToPixelType` finding), so unlike `functor.rs` there is only one
// policy to support.

trait BinOp<T: Scalar> {
    fn apply(&self, a: T, b: T) -> T;
}

trait AllScalarsBinOp:
    BinOp<u8>
    + BinOp<i8>
    + BinOp<u16>
    + BinOp<i16>
    + BinOp<u32>
    + BinOp<i32>
    + BinOp<u64>
    + BinOp<i64>
    + BinOp<f32>
    + BinOp<f64>
{
}

impl<F> AllScalarsBinOp for F where
    F: BinOp<u8>
        + BinOp<i8>
        + BinOp<u16>
        + BinOp<i16>
        + BinOp<u32>
        + BinOp<i32>
        + BinOp<u64>
        + BinOp<i64>
        + BinOp<f32>
        + BinOp<f64>
{
}

trait UnOp<T: Scalar> {
    fn apply(&self, x: T) -> T;
}

trait AllScalarsUnOp:
    UnOp<u8>
    + UnOp<i8>
    + UnOp<u16>
    + UnOp<i16>
    + UnOp<u32>
    + UnOp<i32>
    + UnOp<u64>
    + UnOp<i64>
    + UnOp<f32>
    + UnOp<f64>
{
}

impl<F> AllScalarsUnOp for F where
    F: UnOp<u8>
        + UnOp<i8>
        + UnOp<u16>
        + UnOp<i16>
        + UnOp<u32>
        + UnOp<i32>
        + UnOp<u64>
        + UnOp<i64>
        + UnOp<f32>
        + UnOp<f64>
{
}

fn binary_apply_typed<T: Scalar>(a: &Image, b: &Image, op: &dyn BinOp<T>) -> Image {
    let sa = scalar_slice_or_panic::<T>(a);
    let sb = scalar_slice_or_panic::<T>(b);
    let out: Vec<T> = sa.iter().zip(sb).map(|(&x, &y)| op.apply(x, y)).collect();
    let mut img = Image::from_vec(a.size(), out).expect("same length as a's buffer");
    img.copy_geometry_from(a);
    img
}

fn binary_apply_typed_in_place<T: Scalar>(mut a: Image, b: &Image, op: &dyn BinOp<T>) -> Image {
    let sb: Vec<T> = scalar_slice_or_panic::<T>(b).to_vec();
    let va = scalar_vec_mut_or_panic::<T>(&mut a);
    for (x, y) in va.iter_mut().zip(sb) {
        *x = op.apply(*x, y);
    }
    a
}

fn binary_apply_scalar_typed<T: Scalar>(
    img: &Image,
    c: f64,
    op: &dyn BinOp<T>,
    reversed: bool,
) -> Image {
    let s = scalar_slice_or_panic::<T>(img);
    let cv = T::from_f64(c);
    let out: Vec<T> = if reversed {
        s.iter().map(|&x| op.apply(cv, x)).collect()
    } else {
        s.iter().map(|&x| op.apply(x, cv)).collect()
    };
    let mut out_img = Image::from_vec(img.size(), out).expect("same length as img's buffer");
    out_img.copy_geometry_from(img);
    out_img
}

fn binary_apply_scalar_typed_in_place<T: Scalar>(
    mut img: Image,
    c: f64,
    op: &dyn BinOp<T>,
) -> Image {
    let cv = T::from_f64(c);
    let v = scalar_vec_mut_or_panic::<T>(&mut img);
    for x in v.iter_mut() {
        *x = op.apply(*x, cv);
    }
    img
}

fn unary_apply_typed<T: Scalar>(img: &Image, op: &dyn UnOp<T>) -> Image {
    let s = scalar_slice_or_panic::<T>(img);
    let out: Vec<T> = s.iter().map(|&x| op.apply(x)).collect();
    let mut out_img = Image::from_vec(img.size(), out).expect("same length as img's buffer");
    out_img.copy_geometry_from(img);
    out_img
}

fn unary_apply_typed_in_place<T: Scalar>(mut img: Image, op: &dyn UnOp<T>) -> Image {
    let v = scalar_vec_mut_or_panic::<T>(&mut img);
    for x in v.iter_mut() {
        *x = op.apply(*x);
    }
    img
}

fn binary_apply<F: AllScalarsBinOp>(a: &Image, b: &Image, op: &F) -> Image {
    require_same_shape(a, b);
    dispatch_scalar!(a.pixel_id(), binary_apply_typed, a, b, op)
}

fn binary_apply_in_place<F: AllScalarsBinOp>(a: Image, b: &Image, op: &F) -> Image {
    require_same_shape(&a, b);
    dispatch_scalar!(a.pixel_id(), binary_apply_typed_in_place, a, b, op)
}

fn binary_apply_scalar<F: AllScalarsBinOp>(img: &Image, c: f64, op: &F, reversed: bool) -> Image {
    dispatch_scalar!(
        img.pixel_id(),
        binary_apply_scalar_typed,
        img,
        c,
        op,
        reversed
    )
}

fn binary_apply_scalar_in_place<F: AllScalarsBinOp>(img: Image, c: f64, op: &F) -> Image {
    dispatch_scalar!(
        img.pixel_id(),
        binary_apply_scalar_typed_in_place,
        img,
        c,
        op
    )
}

fn unary_apply<F: AllScalarsUnOp>(img: &Image, op: &F) -> Image {
    dispatch_scalar!(img.pixel_id(), unary_apply_typed, img, op)
}

fn unary_apply_in_place<F: AllScalarsUnOp>(img: Image, op: &F) -> Image {
    dispatch_scalar!(img.pixel_id(), unary_apply_typed_in_place, img, op)
}

// ---- concrete ops -----------------------------------------------------------
//
// Matches `crates/sitk-filters/src/lib.rs`'s `AddOp`/`SubOp`/`MulOp`/`DivOp`/
// `ModOp`/`UnaryMinusOp` and `crates/sitk-filters/src/logic.rs`'s `AndOp`/
// `OrOp`/`XorOp`/`BitwiseNotOp` value-for-value: `wrapping_*` on integers
// (ITK's `static_cast<TOutput>(A op B)` 2's-complement wraparound, defined
// here instead of the C++ overflow that is undefined behavior), plain IEEE
// arithmetic on floats, `Div`/`Modulus` returning the type's max value on a
// zero divisor (`NumericTraits<TOutput>::max()`).

struct AddOp;
struct SubOp;
struct MulOp;
struct DivOp;
struct ModOp;
struct AndOp;
struct OrOp;
struct XorOp;
struct NegOp;
struct BitwiseNotOp;

macro_rules! impl_arith_int {
    ($($t:ty),+ $(,)?) => {$(
        impl BinOp<$t> for AddOp { fn apply(&self, a: $t, b: $t) -> $t { a.wrapping_add(b) } }
        impl BinOp<$t> for SubOp { fn apply(&self, a: $t, b: $t) -> $t { a.wrapping_sub(b) } }
        impl BinOp<$t> for MulOp { fn apply(&self, a: $t, b: $t) -> $t { a.wrapping_mul(b) } }
        impl BinOp<$t> for DivOp {
            fn apply(&self, a: $t, b: $t) -> $t {
                if b == 0 { <$t>::MAX } else { a.wrapping_div(b) }
            }
        }
        impl BinOp<$t> for ModOp {
            fn apply(&self, a: $t, b: $t) -> $t {
                if b == 0 { <$t>::MAX } else { a.wrapping_rem(b) }
            }
        }
        impl BinOp<$t> for AndOp { fn apply(&self, a: $t, b: $t) -> $t { a & b } }
        impl BinOp<$t> for OrOp { fn apply(&self, a: $t, b: $t) -> $t { a | b } }
        impl BinOp<$t> for XorOp { fn apply(&self, a: $t, b: $t) -> $t { a ^ b } }
    )+};
}

macro_rules! impl_arith_float {
    ($($t:ty),+ $(,)?) => {$(
        impl BinOp<$t> for AddOp { fn apply(&self, a: $t, b: $t) -> $t { a + b } }
        impl BinOp<$t> for SubOp { fn apply(&self, a: $t, b: $t) -> $t { a - b } }
        impl BinOp<$t> for MulOp { fn apply(&self, a: $t, b: $t) -> $t { a * b } }
        impl BinOp<$t> for DivOp {
            fn apply(&self, a: $t, b: $t) -> $t { if b == 0.0 { <$t>::MAX } else { a / b } }
        }
        impl BinOp<$t> for ModOp {
            fn apply(&self, _a: $t, _b: $t) -> $t {
                unreachable!("gated to integer pixel types by require_integer")
            }
        }
        impl BinOp<$t> for AndOp {
            fn apply(&self, _a: $t, _b: $t) -> $t {
                unreachable!("gated to integer pixel types by require_integer")
            }
        }
        impl BinOp<$t> for OrOp {
            fn apply(&self, _a: $t, _b: $t) -> $t {
                unreachable!("gated to integer pixel types by require_integer")
            }
        }
        impl BinOp<$t> for XorOp {
            fn apply(&self, _a: $t, _b: $t) -> $t {
                unreachable!("gated to integer pixel types by require_integer")
            }
        }
    )+};
}

impl_arith_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl_arith_float!(f32, f64);

macro_rules! impl_neg_signed {
    ($($t:ty),+ $(,)?) => {$(
        impl UnOp<$t> for NegOp { fn apply(&self, x: $t) -> $t { x.wrapping_neg() } }
    )+};
}

macro_rules! impl_neg_float {
    ($($t:ty),+ $(,)?) => {$(
        impl UnOp<$t> for NegOp { fn apply(&self, x: $t) -> $t { -x } }
    )+};
}

macro_rules! impl_neg_unsigned_unreachable {
    ($($t:ty),+ $(,)?) => {$(
        impl UnOp<$t> for NegOp {
            fn apply(&self, _x: $t) -> $t { unreachable!("gated to signed pixel types by require_signed") }
        }
    )+};
}

impl_neg_signed!(i8, i16, i32, i64);
impl_neg_float!(f32, f64);
impl_neg_unsigned_unreachable!(u8, u16, u32, u64);

macro_rules! impl_bitwise_not_int {
    ($($t:ty),+ $(,)?) => {$(
        impl UnOp<$t> for BitwiseNotOp { fn apply(&self, x: $t) -> $t { !x } }
    )+};
}

macro_rules! impl_bitwise_not_float_unreachable {
    ($($t:ty),+ $(,)?) => {$(
        impl UnOp<$t> for BitwiseNotOp {
            fn apply(&self, _x: $t) -> $t { unreachable!("gated to integer pixel types by require_integer") }
        }
    )+};
}

impl_bitwise_not_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl_bitwise_not_float_unreachable!(f32, f64);

// ---- the 44 trait impls -----------------------------------------------------
//
// One `impl_binary_ops!`/`impl_unary_ops!` invocation per
// `sitkImageOperators.h` operator family; each expands to exactly the 5 (or,
// for unary, 2) overloads that header declares for that family, in the same
// shapes: image⊕image (by-ref, alloc / by-value lhs, in-place), image⊕const
// (by-ref, alloc / by-value lhs, in-place), and const⊕image (by-ref only --
// upstream has no rvalue-image overload for the reversed order either).

macro_rules! impl_binary_ops {
    ($trait:ident, $method:ident, $op:expr, $const_ty:ty, $gate:expr) => {
        impl std::ops::$trait<&Image> for &Image {
            type Output = Image;
            fn $method(self, rhs: &Image) -> Image {
                ($gate)(self);
                binary_apply(self, rhs, &$op)
            }
        }
        impl std::ops::$trait<&Image> for Image {
            type Output = Image;
            fn $method(self, rhs: &Image) -> Image {
                ($gate)(&self);
                binary_apply_in_place(self, rhs, &$op)
            }
        }
        impl std::ops::$trait<$const_ty> for &Image {
            type Output = Image;
            fn $method(self, c: $const_ty) -> Image {
                ($gate)(self);
                binary_apply_scalar(self, c as f64, &$op, false)
            }
        }
        impl std::ops::$trait<$const_ty> for Image {
            type Output = Image;
            fn $method(self, c: $const_ty) -> Image {
                ($gate)(&self);
                binary_apply_scalar_in_place(self, c as f64, &$op)
            }
        }
        impl std::ops::$trait<&Image> for $const_ty {
            type Output = Image;
            fn $method(self, img: &Image) -> Image {
                ($gate)(img);
                binary_apply_scalar(img, self as f64, &$op, true)
            }
        }
    };
}

macro_rules! impl_unary_ops {
    ($trait:ident, $method:ident, $op:expr, $gate:expr) => {
        impl std::ops::$trait for &Image {
            type Output = Image;
            fn $method(self) -> Image {
                ($gate)(self);
                unary_apply(self, &$op)
            }
        }
        impl std::ops::$trait for Image {
            type Output = Image;
            fn $method(self) -> Image {
                ($gate)(&self);
                unary_apply_in_place(self, &$op)
            }
        }
    };
}

impl_binary_ops!(Add, add, AddOp, f64, no_gate);
impl_binary_ops!(Sub, sub, SubOp, f64, no_gate);
impl_binary_ops!(Mul, mul, MulOp, f64, no_gate);
impl_binary_ops!(Div, div, DivOp, f64, no_gate);
impl_binary_ops!(Rem, rem, ModOp, u32, require_integer);
impl_binary_ops!(BitAnd, bitand, AndOp, i32, require_integer);
impl_binary_ops!(BitOr, bitor, OrOp, i32, require_integer);
impl_binary_ops!(BitXor, bitxor, XorOp, i32, require_integer);

impl_unary_ops!(Neg, neg, NegOp, require_signed);
impl_unary_ops!(Not, not, BitwiseNotOp, require_integer);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelId;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn img_i8(size: &[usize], data: Vec<i8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn add_image_image_wraps() {
        let a = img_u8(&[1, 1], vec![250]);
        let b = img_u8(&[1, 1], vec![10]);
        let out = &a + &b;
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[4]);
    }

    #[test]
    fn add_owned_lhs_reuses_buffer_and_matches_allocating() {
        let a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        let b = img_u8(&[2, 2], vec![10, 20, 30, 40]);
        let allocated = &a + &b;
        let in_place = a + &b;
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn add_image_constant_casts_constant_to_pixel_type_first() {
        // ToPixelType(3.7, u8) truncates to 3 (static_cast, not rounding),
        // then 250u8.wrapping_add(3) == 253 -- not the f64-then-saturate
        // policy `sitk_filters::add_constant` uses (see the module docs).
        let a = img_u8(&[1, 1], vec![250]);
        let out = &a + 3.7;
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[253]);
    }

    #[test]
    fn subtract_constant_from_image_and_reversed_order_differ() {
        let a = img_i8(&[1, 1], vec![10]);
        let img_minus_scalar = &a - 3.0;
        let scalar_minus_img = 3.0 - &a;
        assert_eq!(img_minus_scalar.scalar_slice::<i8>().unwrap(), &[7]);
        assert_eq!(scalar_minus_img.scalar_slice::<i8>().unwrap(), &[-7]);
    }

    #[test]
    fn divide_by_zero_constant_yields_type_max() {
        let a = img_u8(&[1, 1], vec![5]);
        let out = &a / 0.0;
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[u8::MAX]);
    }

    #[test]
    fn divide_image_image_by_zero_yields_type_max() {
        let a = img_u8(&[1, 1], vec![5]);
        let b = img_u8(&[1, 1], vec![0]);
        let out = &a / &b;
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[u8::MAX]);
    }

    #[test]
    fn modulus_image_and_constant() {
        let a = Image::from_vec(&[2, 1], vec![10u32, 7]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![3u32, 4]).unwrap();
        let out = &a % &b;
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 3]);

        let out_c = &a % 3u32;
        assert_eq!(out_c.scalar_slice::<u32>().unwrap(), &[1, 1]);

        let out_rev = 10u32 % &a;
        assert_eq!(out_rev.scalar_slice::<u32>().unwrap(), &[0, 3]);
    }

    #[test]
    fn modulus_by_zero_yields_type_max() {
        let a = Image::from_vec(&[1, 1], vec![5u32]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![0u32]).unwrap();
        let out = &a % &b;
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[u32::MAX]);
    }

    #[test]
    fn bitwise_and_or_xor_image_and_constant() {
        let a = Image::from_vec(&[1, 1], vec![0b1100u32]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![0b1010u32]).unwrap();
        assert_eq!((&a & &b).scalar_slice::<u32>().unwrap(), &[0b1000]);
        assert_eq!((&a | &b).scalar_slice::<u32>().unwrap(), &[0b1110]);
        assert_eq!((&a ^ &b).scalar_slice::<u32>().unwrap(), &[0b0110]);
        assert_eq!((&a & 0b1010i32).scalar_slice::<u32>().unwrap(), &[0b1000]);
        assert_eq!((0b1010i32 & &a).scalar_slice::<u32>().unwrap(), &[0b1000]);
    }

    #[test]
    fn unary_minus_wraps_on_signed_min() {
        let a = img_i8(&[1, 1], vec![i8::MIN]);
        let out = -&a;
        assert_eq!(out.scalar_slice::<i8>().unwrap(), &[i8::MIN]);
    }

    #[test]
    fn unary_minus_owned_matches_allocating() {
        let a = img_i8(&[1, 1], vec![5]);
        let allocated = -&a.clone();
        let in_place = -a;
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn bitwise_not_complements_bits() {
        let a = img_u8(&[1, 1], vec![0b0000_1111]);
        let out = !&a;
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0b1111_0000]);
    }

    #[test]
    #[should_panic(expected = "pixel type mismatch")]
    fn add_image_image_pixel_type_mismatch_panics() {
        let a = img_u8(&[1, 1], vec![1]);
        let b = Image::from_vec(&[1, 1], vec![1i16]).unwrap();
        let _ = &a + &b;
    }

    #[test]
    #[should_panic(expected = "size mismatch")]
    fn add_image_image_size_mismatch_panics() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = img_u8(&[3, 1], vec![1, 2, 3]);
        let _ = &a + &b;
    }

    #[test]
    #[should_panic(expected = "requires an integer pixel type")]
    fn modulus_on_float_panics() {
        let a = Image::from_vec(&[1, 1], vec![1.5f64]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![2.0f64]).unwrap();
        let _ = &a % &b;
    }

    #[test]
    #[should_panic(expected = "requires a signed pixel type")]
    fn unary_minus_on_unsigned_panics() {
        let a = img_u8(&[1, 1], vec![1]);
        let _ = -&a;
    }

    #[test]
    #[should_panic(expected = "requires a scalar pixel type")]
    fn add_rejects_vector_image_through_scalar_slice() {
        let a = Image::from_vec_vector::<f32>(&[1, 1], 3, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(a.pixel_id(), PixelId::VectorFloat32);
        let b = Image::from_vec_vector::<f32>(&[1, 1], 3, vec![4.0, 5.0, 6.0]).unwrap();
        let _ = &a + &b;
    }
}
