//! The pixel-functor seam: the generic engine behind ITK's
//! `UnaryGeneratorImageFilter` / `UnaryFunctorImageFilter` and
//! `BinaryFunctorImageFilter` (see the reference headers under
//! `Modules/{Core/Common,Filtering/ImageFilterBase}/include/`).
//!
//! ITK's functor filters are templated on a `TFunction` type whose
//! `operator()` runs per pixel; two distinct output-type policies show up
//! among ITK's actual functors, and the difference is observable in
//! out-of-range results:
//!
//! - **f64-compute** ([`UnaryFunctor`]): promote to `double`, apply the
//!   operation, `static_cast` back down. This is how ITK's math functors
//!   work, e.g. `itkAcosImageFilter.h`'s `Functor::Acos`:
//!   `static_cast<TOutput>(std::acos(static_cast<double>(A)))`. This crate
//!   narrows via [`Scalar::from_f64`], a saturating cast, in place of C++
//!   `static_cast`'s undefined behavior on out-of-range floats.
//! - **pixel-type-compute** ([`BinaryFunctor`], [`UnaryPixelFunctor`]):
//!   operate directly in the pixel type, no promotion. This is how ITK's
//!   bitwise/logic functors work, e.g. `itkBitwiseOpsFunctors.h`'s
//!   `Functor::AND`: `static_cast<TOutput>(A & B)` with `A & B` evaluated in
//!   the pixel type; and how ITK's arithmetic functors work
//!   (`itkArithmeticOpsFunctors.h`'s `Add2`, `Sub2`, `Mult`, `Div`,
//!   `Modulus`, `UnaryMinus`), which is why this crate's `add` / `subtract` /
//!   `multiply` / `divide` / `modulus` / `unary_minus` wrap on integer
//!   overflow instead of promoting to a wider type. [`UnaryPixelFunctor`] is
//!   the one-image face of this same policy (`UnaryMinus`,
//!   `itkBitwiseOpsFunctors.h`'s `BitwiseNot`), as [`UnaryFunctor`] is the
//!   one-image face of f64-compute.
//!
//! [`unary_functor!`] and [`binary_functor!`] each wire a functor value to a
//! pair of public functions: an allocating one taking `&Image`, and an
//! in-place one that consumes an owned `Image` and reuses its buffer,
//! mirroring the fact that both `UnaryFunctorImageFilter` and
//! `BinaryFunctorImageFilter` derive from `InPlaceImageFilter` in ITK.
//! [`UnaryPixelFunctor`] has no such macro: every current consumer
//! (`unary_minus`, `bitwise_not`, `binary_not`) needs a pixel-type
//! precondition check before dispatch, so they call [`unary_pixel_apply`] /
//! [`unary_pixel_apply_in_place`] directly instead, the same way
//! `and`/`or`/`xor`/`not` bypass `binary_functor!`/`unary_functor!` for the
//! same reason.

use crate::Result;
use crate::require_same_shape;
use sitk_core::{Image, Scalar, dispatch_scalar};

/// A per-pixel operation on one image, computed in `f64` and narrowed back to
/// the pixel type via [`Scalar::from_f64`] — ITK's
/// `static_cast<TOutput>(f(static_cast<double>(A)))` policy, used by the math
/// functors (`std::acos`, `std::sqrt`, `std::log`, ...; see
/// `itkAcosImageFilter.h`). A functor is a value (often zero-sized) with an
/// `apply` method, mirroring an ITK functor object's `operator()`.
pub trait UnaryFunctor {
    /// Evaluate the functor at `x`, in double precision.
    fn apply(&self, x: f64) -> f64;
}

/// A per-pixel operation on two images, computed directly in the pixel type
/// `T` — ITK's `static_cast<TOutput>(A op B)` policy where `op` itself runs
/// in `T`, used by the bitwise/logic functors (`AND`, `OR`, `XOR`; see
/// `itkBitwiseOpsFunctors.h`) and the arithmetic functors (`Add2`, `Sub2`,
/// `Mult`, `Div`; see `itkArithmeticOpsFunctors.h`).
pub trait BinaryFunctor<T: Scalar> {
    /// Evaluate the functor at `(a, b)`, in the pixel type.
    fn apply(&self, a: T, b: T) -> T;
}

/// Implements [`BinaryFunctor`] for every pixel scalar type this crate
/// supports. [`binary_apply`] and [`binary_apply_in_place`] dispatch across
/// all ten [`sitk_core::PixelId`] variants at runtime from a single generic
/// function, so the functor type they accept must be usable at each one;
/// this bound expresses that requirement once instead of repeating all ten
/// at every call site.
pub(crate) trait AllScalarsBinaryFunctor:
    BinaryFunctor<u8>
    + BinaryFunctor<i8>
    + BinaryFunctor<u16>
    + BinaryFunctor<i16>
    + BinaryFunctor<u32>
    + BinaryFunctor<i32>
    + BinaryFunctor<u64>
    + BinaryFunctor<i64>
    + BinaryFunctor<f32>
    + BinaryFunctor<f64>
{
}

impl<F> AllScalarsBinaryFunctor for F where
    F: BinaryFunctor<u8>
        + BinaryFunctor<i8>
        + BinaryFunctor<u16>
        + BinaryFunctor<i16>
        + BinaryFunctor<u32>
        + BinaryFunctor<i32>
        + BinaryFunctor<u64>
        + BinaryFunctor<i64>
        + BinaryFunctor<f32>
        + BinaryFunctor<f64>
{
}

/// A per-pixel operation on one image, computed directly in the pixel type
/// `T` — ITK's `static_cast<TOutput>(op A)` policy where `op` runs in `T`
/// with no `double` promotion, used by the unary members of
/// `itkArithmeticOpsFunctors.h` (`UnaryMinus`) and `itkBitwiseOpsFunctors.h`
/// (`BitwiseNot`). The single-image counterpart of [`BinaryFunctor`].
pub trait UnaryPixelFunctor<T: Scalar> {
    /// Evaluate the functor at `x`, in the pixel type.
    fn apply(&self, x: T) -> T;
}

/// Implements [`UnaryPixelFunctor`] for every pixel scalar type this crate
/// supports; see [`AllScalarsBinaryFunctor`] for why this blanket bound
/// exists (the same `dispatch_scalar!` requirement applies here).
pub(crate) trait AllScalarsUnaryPixelFunctor:
    UnaryPixelFunctor<u8>
    + UnaryPixelFunctor<i8>
    + UnaryPixelFunctor<u16>
    + UnaryPixelFunctor<i16>
    + UnaryPixelFunctor<u32>
    + UnaryPixelFunctor<i32>
    + UnaryPixelFunctor<u64>
    + UnaryPixelFunctor<i64>
    + UnaryPixelFunctor<f32>
    + UnaryPixelFunctor<f64>
{
}

impl<F> AllScalarsUnaryPixelFunctor for F where
    F: UnaryPixelFunctor<u8>
        + UnaryPixelFunctor<i8>
        + UnaryPixelFunctor<u16>
        + UnaryPixelFunctor<i16>
        + UnaryPixelFunctor<u32>
        + UnaryPixelFunctor<i32>
        + UnaryPixelFunctor<u64>
        + UnaryPixelFunctor<i64>
        + UnaryPixelFunctor<f32>
        + UnaryPixelFunctor<f64>
{
}

// ---- unary (pixel-type-compute policy) engine --------------------------

fn unary_pixel_apply_typed<T: Scalar>(img: &Image, f: &dyn UnaryPixelFunctor<T>) -> Result<Image> {
    let s = img.scalar_slice::<T>().expect("dispatch guarantees type");
    let out: Vec<T> = s.iter().map(|&x| f.apply(x)).collect();
    let mut out_img = Image::from_vec(img.size(), out)?;
    out_img.copy_geometry_from(img);
    Ok(out_img)
}

fn unary_pixel_apply_typed_in_place<T: Scalar>(
    mut img: Image,
    f: &dyn UnaryPixelFunctor<T>,
) -> Result<Image> {
    let v = img.scalar_vec_mut::<T>().expect("dispatch guarantees type");
    for x in v.iter_mut() {
        *x = f.apply(*x);
    }
    Ok(img)
}

/// Apply a [`UnaryPixelFunctor`] over `img`, allocating a new [`Image`] of
/// the same pixel type.
pub(crate) fn unary_pixel_apply<F: AllScalarsUnaryPixelFunctor>(
    img: &Image,
    f: &F,
) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), unary_pixel_apply_typed, img, f)
}

/// Apply a [`UnaryPixelFunctor`] over `img` in place, reusing its buffer
/// instead of allocating a new [`Image`].
pub(crate) fn unary_pixel_apply_in_place<F: AllScalarsUnaryPixelFunctor>(
    img: Image,
    f: &F,
) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), unary_pixel_apply_typed_in_place, img, f)
}

// ---- unary (f64-compute policy) engine -------------------------------

/// Apply a [`UnaryFunctor`] over `img`, allocating a new [`Image`] of the
/// same pixel type.
pub(crate) fn unary_apply<F: UnaryFunctor>(img: &Image, f: &F) -> Result<Image> {
    let vals: Vec<f64> = img.to_f64_vec().iter().map(|&v| f.apply(v)).collect();
    crate::image_from_f64(img.pixel_id(), img.size(), img, &vals)
}

// `dispatch_scalar!` requires the dispatched function to be generic over a
// single `T: Scalar`; the functor is threaded through as a trait object
// (`UnaryFunctor` is object-safe, so this costs nothing but a vtable call).
fn unary_apply_typed_in_place<T: Scalar>(img: &mut Image, f: &dyn UnaryFunctor) -> Result<()> {
    let v = img.scalar_vec_mut::<T>()?;
    for x in v.iter_mut() {
        *x = T::from_f64(f.apply(x.as_f64()));
    }
    Ok(())
}

/// Apply a [`UnaryFunctor`] over `img` in place, reusing its buffer instead
/// of allocating a new [`Image`].
pub(crate) fn unary_apply_in_place<F: UnaryFunctor>(mut img: Image, f: &F) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), unary_apply_typed_in_place, &mut img, f)?;
    Ok(img)
}

// ---- binary (pixel-type-compute policy) engine ------------------------

// As above: `dispatch_scalar!` needs a single `T: Scalar` generic, so the
// functor crosses as `&dyn BinaryFunctor<T>` rather than a second type
// parameter.
fn binary_apply_typed<T: Scalar>(a: &Image, b: &Image, f: &dyn BinaryFunctor<T>) -> Result<Image> {
    let sa = a.scalar_slice::<T>().expect("dispatch guarantees type");
    let sb = b.scalar_slice::<T>().expect("dispatch guarantees type");
    let out: Vec<T> = sa.iter().zip(sb).map(|(&x, &y)| f.apply(x, y)).collect();
    let mut img = Image::from_vec(a.size(), out)?;
    img.copy_geometry_from(a);
    Ok(img)
}

fn binary_apply_typed_in_place<T: Scalar>(
    mut a: Image,
    b: &Image,
    f: &dyn BinaryFunctor<T>,
) -> Result<Image> {
    let sb = b.scalar_slice::<T>().expect("dispatch guarantees type");
    let va = a.scalar_vec_mut::<T>().expect("dispatch guarantees type");
    for (x, &y) in va.iter_mut().zip(sb) {
        *x = f.apply(*x, y);
    }
    Ok(a)
}

/// Apply a [`BinaryFunctor`] pixel-wise over `a` and `b`, allocating a new
/// [`Image`]. Errors if `a` and `b` differ in pixel type or size.
pub(crate) fn binary_apply<F: AllScalarsBinaryFunctor>(
    a: &Image,
    b: &Image,
    f: &F,
) -> Result<Image> {
    require_same_shape(a, b)?;
    dispatch_scalar!(a.pixel_id(), binary_apply_typed, a, b, f)
}

/// Apply a [`BinaryFunctor`] pixel-wise over `a` and `b` in place, reusing
/// `a`'s buffer instead of allocating a new [`Image`]. Errors if `a` and `b`
/// differ in pixel type or size.
pub(crate) fn binary_apply_in_place<F: AllScalarsBinaryFunctor>(
    a: Image,
    b: &Image,
    f: &F,
) -> Result<Image> {
    require_same_shape(&a, b)?;
    dispatch_scalar!(a.pixel_id(), binary_apply_typed_in_place, a, b, f)
}

// ---- macros -------------------------------------------------------------

/// Emit a `pub fn $name(img: &Image, ...) -> Result<Image>` (allocating) and
/// a `pub fn $name_in_place(img: Image, ...) -> Result<Image>` (reuses
/// `img`'s buffer), both backed by a [`UnaryFunctor`] value — ITK's
/// f64-compute policy (see the module docs and `itkAcosImageFilter.h`).
///
/// `$functor` is evaluated once per generated function and may reference the
/// trailing parameters (e.g. a runtime constant), letting one functor value
/// carry per-call state the way an ITK functor object does.
macro_rules! unary_functor {
    (
        $(#[$doc:meta])*
        pub fn $name:ident, $name_in_place:ident ( $($p:ident : $pt:ty),* $(,)? ) = $functor:expr;
    ) => {
        $(#[$doc])*
        pub fn $name(img: &Image $(, $p: $pt)*) -> Result<Image> {
            $crate::functor::unary_apply(img, &($functor))
        }

        #[doc = concat!(
            "In-place variant of [`", stringify!($name),
            "`]: reuses `img`'s buffer instead of allocating a new [`Image`]."
        )]
        pub fn $name_in_place(img: Image $(, $p: $pt)*) -> Result<Image> {
            $crate::functor::unary_apply_in_place(img, &($functor))
        }
    };
}

/// Emit a `pub fn $name(a: &Image, b: &Image) -> Result<Image>` (allocating)
/// and a `pub fn $name_in_place(a: Image, b: &Image) -> Result<Image>`
/// (reuses `a`'s buffer), both backed by a [`BinaryFunctor`] value — ITK's
/// pixel-type-compute policy (see the module docs and
/// `itkBitwiseOpsFunctors.h`).
macro_rules! binary_functor {
    (
        $(#[$doc:meta])*
        pub fn $name:ident, $name_in_place:ident = $functor:expr;
    ) => {
        $(#[$doc])*
        pub fn $name(a: &Image, b: &Image) -> Result<Image> {
            $crate::functor::binary_apply(a, b, &($functor))
        }

        #[doc = concat!(
            "In-place variant of [`", stringify!($name),
            "`]: reuses `a`'s buffer instead of allocating a new [`Image`]."
        )]
        pub fn $name_in_place(a: Image, b: &Image) -> Result<Image> {
            $crate::functor::binary_apply_in_place(a, b, &($functor))
        }
    };
}

pub(crate) use binary_functor;
pub(crate) use unary_functor;

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    /// A trivial `f64`-compute functor: doubles the value. Stands in for a
    /// math filter like `AcosImageFilter` to pin down policy (a) in
    /// isolation from any real filter's semantics.
    struct DoubleF64;
    impl UnaryFunctor for DoubleF64 {
        fn apply(&self, x: f64) -> f64 {
            x * 2.0
        }
    }

    /// A trivial pixel-type-compute functor: wrapping add. Stands in for a
    /// bitwise/arithmetic filter like `AndImageFilter` to pin down policy
    /// (b) in isolation from any real filter's semantics.
    struct WrappingAddU8;
    impl BinaryFunctor<u8> for WrappingAddU8 {
        fn apply(&self, a: u8, b: u8) -> u8 {
            a.wrapping_add(b)
        }
    }
    // AllScalarsBinaryFunctor needs an impl for every scalar type; only u8
    // is exercised below, so the rest just delegate to plain `+`.
    macro_rules! impl_passthrough {
        ($($t:ty),+) => {$(
            impl BinaryFunctor<$t> for WrappingAddU8 {
                fn apply(&self, a: $t, b: $t) -> $t { a + b }
            }
        )+};
    }
    impl_passthrough!(i8, u16, i16, u32, i32, u64, i64, f32, f64);

    /// A trivial unary pixel-type-compute functor: wrapping negation. Stands
    /// in for `UnaryMinus`/`BitwiseNot` to pin down the unary face of policy
    /// (b) in isolation from any real filter's semantics.
    struct WrappingNegU8;
    impl UnaryPixelFunctor<u8> for WrappingNegU8 {
        fn apply(&self, x: u8) -> u8 {
            x.wrapping_neg()
        }
    }
    macro_rules! impl_unary_passthrough {
        ($($t:ty),+) => {$(
            impl UnaryPixelFunctor<$t> for WrappingNegU8 {
                fn apply(&self, x: $t) -> $t { x }
            }
        )+};
    }
    impl_unary_passthrough!(i8, u16, i16, u32, i32, u64, i64, f32, f64);

    #[test]
    fn unary_pixel_policy_wraps_on_overflow() {
        // policy (b), unary face: 1u8.wrapping_neg() == 255, matching C++'s
        // static_cast<TOutput>(-A) 2's-complement wrap (no f64 promotion).
        let a = img_u8(&[1, 1], vec![1]);
        let out = unary_pixel_apply(&a, &WrappingNegU8).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[255]);
    }

    #[test]
    fn unary_pixel_in_place_matches_allocating() {
        let a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        let allocated = unary_pixel_apply(&a, &WrappingNegU8).unwrap();
        let in_place = unary_pixel_apply_in_place(a, &WrappingNegU8).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn unary_f64_policy_saturates_on_overflow() {
        // policy (a): compute in f64 (150.0 * 2.0 = 300.0), then saturate
        // via Scalar::from_f64 -- 300 does not fit in u8, so it clamps to 255.
        let a = img_u8(&[1, 1], vec![150]);
        let out = unary_apply(&a, &DoubleF64).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[255]);
    }

    #[test]
    fn binary_pixel_policy_wraps_on_overflow() {
        // policy (b): compute in u8 itself (250u8.wrapping_add(10) == 4),
        // matching C++'s static_cast<TOutput>(A op B) 2's-complement wrap.
        let a = img_u8(&[1, 1], vec![250]);
        let b = img_u8(&[1, 1], vec![10]);
        let out = binary_apply(&a, &b, &WrappingAddU8).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[4]);
    }

    #[test]
    fn unary_in_place_matches_allocating() {
        let a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        let allocated = unary_apply(&a, &DoubleF64).unwrap();
        let in_place = unary_apply_in_place(a, &DoubleF64).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn binary_in_place_matches_allocating() {
        let a = img_u8(&[2, 2], vec![1, 2, 3, 4]);
        let b = img_u8(&[2, 2], vec![250, 10, 0, 5]);
        let allocated = binary_apply(&a, &b, &WrappingAddU8).unwrap();
        let in_place = binary_apply_in_place(a, &b, &WrappingAddU8).unwrap();
        assert_eq!(allocated, in_place);
    }
}
