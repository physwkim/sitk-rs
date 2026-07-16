//! Unary math and multi-image f64-compute filters for sitk-rs, verified
//! against the ITK v6 source under
//! `Modules/Filtering/ImageIntensity/include/` (`itkAbsImageFilter.h`,
//! `itkAcosImageFilter.h`, `itkAsinImageFilter.h`, `itkAtanImageFilter.h`,
//! `itkCosImageFilter.h`, `itkSinImageFilter.h`, `itkTanImageFilter.h`,
//! `itkExpImageFilter.h`, `itkExpNegativeImageFilter.h`, `itkLogImageFilter.h`,
//! `itkLog10ImageFilter.h`, `itkSqrtImageFilter.h`, `itkSquareImageFilter.h`,
//! `itkBoundedReciprocalImageFilter.h`, `itkAtan2ImageFilter.h`,
//! `itkBinaryMagnitudeImageFilter.h`, `itkArithmeticOpsFunctors.h`'s
//! `DivFloor`/`DivReal`) and
//! `Modules/Filtering/ImageCompare/include/itkSquaredDifferenceImageFilter.h`
//! / `itkAbsoluteValueDifferenceImageFilter.h`.
//!
//! The single-image filters are f64-compute (`functor.rs`'s policy (a)):
//! each ITK functor promotes its input to `double`, applies a `<cmath>`
//! function, and `static_cast`s back down (`itkAcosImageFilter.h`'s `Acos`:
//! `static_cast<TOutput>(std::acos(static_cast<double>(A)))`), matching this
//! crate's [`UnaryFunctor`] / `functor::unary_functor!` seam exactly.
//! Out-of-domain inputs (`sqrt` of a negative, `log` of `0`) produce
//! `NaN`/`-inf` in `f64`, which [`crate::core::Scalar::from_f64`] then narrows:
//! integer outputs saturate (`NaN` maps to `0`, `-inf` maps to the type's
//! minimum), float outputs keep the `NaN`/`-inf` exactly.
//!
//! Everything from [`squared_difference`] on is also f64-compute but over
//! *two* images (e.g. `itkSquaredDifferenceImageFilter.h`'s
//! `SquaredDifference2` functor: `diff = double(A) - double(B);
//! static_cast<TOutput>(diff * diff)`), which is outside both `functor.rs`
//! traits ([`UnaryFunctor`] takes one image; [`crate::filters::BinaryFunctor`] computes in
//! the pixel type rather than `f64`). Doing the subtraction in the pixel
//! type would be wrong for unsigned inputs (`5u8 - 10u8` wraps instead of
//! producing `-5`), so these are implemented directly against
//! `to_f64_vec`/`image_from_f64`, the same way `rescale_intensity` and
//! `statistics` in `lib.rs` handle multi-image or reduction f64 math that
//! falls outside the functor seam.
//!
//! [`nary_add`]/[`nary_maximum`] (`N` images,
//! `Modules/Filtering/ImageIntensity/include/itkNaryFunctorImageFilter.h`'s
//! `TFunction::operator()(const std::vector<TInput>&)`, specialised by
//! `itkNaryAddImageFilter.h`'s `Add1` and `itkNaryMaximumImageFilter.h`'s
//! `Maximum1`) and [`ternary_add`]/[`ternary_magnitude`]/
//! [`ternary_magnitude_squared`] (fixed 3 images, `itkArithmeticOpsFunctors.h`'s
//! `Add3`, `itkTernaryMagnitudeImageFilter.h`'s `Modulus3`,
//! `itkTernaryMagnitudeSquaredImageFilter.h`'s `ModulusSquare3`) all reduce
//! uniformly in `f64` here, narrowing once at the end -- consistent with
//! this whole module's f64-compute policy and with the crate's established
//! precedent for multi-image/multi-pixel reductions elsewhere (e.g.
//! `projection::maximum_projection`/`minimum_projection`'s
//! `fold(f64::NEG_INFINITY, f64::max)` over pixel lines). This is a
//! deliberate, and not uniform, divergence from the raw C++, which computes
//! each of these five functors differently:
//! - `Add1` (nary) accumulates in `NumericTraits<TInput>::AccumulateType`, a
//!   *wider* integer type than `TInput`, narrowing only once at the end (the
//!   header's own doc: "No numeric overflow checking is performed").
//! - `Maximum1` (nary) folds natively in `TOutput` with no promotion at all.
//! - `Add3` (ternary) computes natively in the shared input pixel type with
//!   no promotion either (`static_cast<TOutput>(A + B + C)`), so unlike
//!   `Add1`, an intermediate `A + B` can itself overflow/wrap for a narrow
//!   integer type before `+ C` is even applied.
//! - `Modulus3` (ternary) already computes in `double`
//!   (`static_cast<TOutput>(std::sqrt(static_cast<double>(A*A + B*B +
//!   C*C)))`), matching this port exactly.
//! - `ModulusSquare3` (ternary) computes natively in the shared input pixel
//!   type (`static_cast<TOutput>(A*A + B*B + C*C)`, no `double` at all),
//!   unlike its sibling `Modulus3` immediately above it in the same header
//!   family.
//!
//! Folding every one of these in `f64` instead can misorder/round results
//! that depend on exact integer overflow or on `u64`/`i64` magnitudes beyond
//! `f64`'s 53-bit exact range; see each function's own doc for the precise
//! divergence.

use crate::core::{Image, PixelId, Scalar, dispatch_scalar};
use crate::filters::functor::{self, UnaryFunctor};
use crate::filters::geometry::require_same_physical_space;
use crate::filters::{FilterError, Result, image_from_f64, require_same_shape};

// ---- round (RealPixelIDTypeList only) ----------------------------------
//
// `RoundImageFilter.yaml`'s `pixel_types: RealPixelIDTypeList` restricts the
// input (and, with no `output_pixel_type` override, the output) to
// `Float32`/`Float64` -- unlike every other filter in this module, so
// `round`/`round_in_place` are hand-written instead of going through
// `functor::unary_functor!`, gating first (mirrors `logic::binary_not`'s
// gate-then-delegate shape).
//
// `itkRoundImageFilter.h`'s functor calls `itk::Math::Round<TOutput,
// TInput>(A)`, a synonym for `RoundHalfIntegerUp<TOutput, TInput>(A)`
// (itkMath.h:191-204). `itkTemplateFloatingToIntegerMacro` (itkMath.h:130-146)
// dispatches purely on `sizeof(TReturn)` -- `<= 4` routes through
// `Detail::RoundHalfIntegerUp_32` (which always returns `int32_t`), `<= 8`
// through `Detail::RoundHalfIntegerUp_64` (`int64_t`) -- with **no check that
// `TReturn` is actually an integer type**, despite the doc comment two lines
// above claiming "TReturn must be an integer type" (itkMath.h:171). Since
// `RoundImageFilter`'s `TOutput` is `Float32`/`Float64` (`sizeof` 4/8), a
// `Float32` output takes the `int32_t`-intermediate path and a `Float64`
// output the `int64_t`-intermediate path, each ending in
// `static_cast<TOutput>(int32_or_64_result)` (itkMath.h:136/140).
// `RoundHalfIntegerUp_base<TReturn, TInput>` (itkMathDetail.h:108-116) is
// `x += 0.5; r = static_cast<TReturn>(x); nonnegative(x) ? r : (x == TInput(r)
// ? r : r - 1)`, which is algebraically `floor(x + 0.5)` computed in `TInput`
// arithmetic and then narrowed to `TReturn` -- exactly what [`Round::apply`]
// computes directly in `f64`, *for every input whose rounded value fits in
// the intermediate integer type*. When it doesn't -- `|round(x)|` exceeding
// `i32::MAX` for a `Float32` input, or `i64::MAX` for a `Float64` input -- the
// `static_cast<int32_t>(x)`/`static_cast<int64_t>(x)` step is itself an
// out-of-range float-to-int cast, undefined behavior in C++ (platforms
// commonly return the "integer indefinite" sentinel, e.g. `INT32_MIN`, via
// `cvttss2si`/`cvttsd2si`). This port's `f64`-compute engine has no such
// intermediate: [`Round::apply`] always returns the mathematically correct
// rounded value, never that sentinel. Tracked as a deliberate divergence
// (upstream UB, defined here) in the upstream-findings ledger, §4.35.
struct Round;
impl UnaryFunctor for Round {
    fn apply(&self, x: f64) -> f64 {
        (x + 0.5).floor()
    }
}

fn require_real_pixel_type(img: &Image) -> Result<()> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    Ok(())
}

/// `RoundImageFilter`: pixel-wise rounding to the nearest integer, ties
/// rounding away from zero on the positive side (`RoundHalfIntegerUp`: `1.5 ->
/// 2`, `-1.5 -> -1`, `2.5 -> 3`). Output keeps `img`'s own `Float32`/`Float64`
/// pixel type (`RoundImageFilter.yaml` has no `output_pixel_type` override).
/// Errors with [`FilterError::RequiresRealPixelType`] on any other pixel type
/// (`pixel_types: RealPixelIDTypeList`). See the module docs above for the
/// two-stage-cast quirk this simplifies away.
pub fn round(img: &Image) -> Result<Image> {
    require_real_pixel_type(img)?;
    functor::unary_apply(img, &Round)
}

/// In-place variant of [`round`]: reuses `img`'s buffer instead of allocating
/// a new [`Image`].
pub fn round_in_place(img: Image) -> Result<Image> {
    require_real_pixel_type(&img)?;
    functor::unary_apply_in_place(img, &Round)
}

// ---- abs --------------------------------------------------------------

/// `Abs` functor (`itkAbsImageFilter.h`): `static_cast<TOutput>(itk::Math::Absolute(A))`.
struct Abs;
impl UnaryFunctor for Abs {
    fn apply(&self, x: f64) -> f64 {
        x.abs()
    }
}

functor::unary_functor! {
    /// `AbsImageFilter`: pixel-wise absolute value, output type follows input.
    pub fn abs, abs_in_place() = Abs;
}

// ---- trigonometric ------------------------------------------------------

/// `Acos` functor (`itkAcosImageFilter.h`): `std::acos`.
struct Acos;
impl UnaryFunctor for Acos {
    fn apply(&self, x: f64) -> f64 {
        x.acos()
    }
}

functor::unary_functor! {
    /// `AcosImageFilter`: pixel-wise inverse cosine (`std::acos`); `NaN` outside `[-1, 1]`.
    pub fn acos, acos_in_place() = Acos;
}

/// `Asin` functor (`itkAsinImageFilter.h`): `std::asin`.
struct Asin;
impl UnaryFunctor for Asin {
    fn apply(&self, x: f64) -> f64 {
        x.asin()
    }
}

functor::unary_functor! {
    /// `AsinImageFilter`: pixel-wise inverse sine (`std::asin`); `NaN` outside `[-1, 1]`.
    pub fn asin, asin_in_place() = Asin;
}

/// `Atan` functor (`itkAtanImageFilter.h`): `std::atan`.
struct Atan;
impl UnaryFunctor for Atan {
    fn apply(&self, x: f64) -> f64 {
        x.atan()
    }
}

functor::unary_functor! {
    /// `AtanImageFilter`: pixel-wise one-argument inverse tangent (`std::atan`).
    pub fn atan, atan_in_place() = Atan;
}

/// `Cos` functor (`itkCosImageFilter.h`): `std::cos`.
struct Cos;
impl UnaryFunctor for Cos {
    fn apply(&self, x: f64) -> f64 {
        x.cos()
    }
}

functor::unary_functor! {
    /// `CosImageFilter`: pixel-wise cosine (`std::cos`).
    pub fn cos, cos_in_place() = Cos;
}

/// `Sin` functor (`itkSinImageFilter.h`): `std::sin`.
struct Sin;
impl UnaryFunctor for Sin {
    fn apply(&self, x: f64) -> f64 {
        x.sin()
    }
}

functor::unary_functor! {
    /// `SinImageFilter`: pixel-wise sine (`std::sin`).
    pub fn sin, sin_in_place() = Sin;
}

/// `Tan` functor (`itkTanImageFilter.h`): `std::tan`.
struct Tan;
impl UnaryFunctor for Tan {
    fn apply(&self, x: f64) -> f64 {
        x.tan()
    }
}

functor::unary_functor! {
    /// `TanImageFilter`: pixel-wise tangent (`std::tan`).
    pub fn tan, tan_in_place() = Tan;
}

// ---- exponential / logarithmic ------------------------------------------

/// `Exp` functor (`itkExpImageFilter.h`): `std::exp`.
struct Exp;
impl UnaryFunctor for Exp {
    fn apply(&self, x: f64) -> f64 {
        x.exp()
    }
}

functor::unary_functor! {
    /// `ExpImageFilter`: pixel-wise `e^x` (`std::exp`).
    pub fn exp, exp_in_place() = Exp;
}

/// `ExpNegative` functor (`itkExpNegativeImageFilter.h`): `std::exp(-K * x)`.
struct ExpNegative;
impl UnaryFunctor for ExpNegative {
    fn apply(&self, x: f64) -> f64 {
        (-x).exp()
    }
}

functor::unary_functor! {
    /// `ExpNegativeImageFilter`: pixel-wise `exp(-K * x)`, with `K` fixed at
    /// ITK's default of `1.0`. `itkExpNegativeImageFilter.h`'s `ExpNegative`
    /// functor takes a runtime `Factor`, but SimpleITK's generated wrapper
    /// (`ExpNegativeImageFilter.yaml`, `members: []`) never exposes it, so
    /// this port doesn't either.
    pub fn exp_negative, exp_negative_in_place() = ExpNegative;
}

/// `Log` functor (`itkLogImageFilter.h`): `std::log` (natural log).
struct Log;
impl UnaryFunctor for Log {
    fn apply(&self, x: f64) -> f64 {
        x.ln()
    }
}

functor::unary_functor! {
    /// `LogImageFilter`: pixel-wise natural log (`std::log`); `-inf` at `0`, `NaN` below `0`.
    pub fn log, log_in_place() = Log;
}

/// `Log10` functor (`itkLog10ImageFilter.h`): `std::log10`.
struct Log10;
impl UnaryFunctor for Log10 {
    fn apply(&self, x: f64) -> f64 {
        x.log10()
    }
}

functor::unary_functor! {
    /// `Log10ImageFilter`: pixel-wise base-10 log (`std::log10`); `-inf` at `0`, `NaN` below `0`.
    pub fn log10, log10_in_place() = Log10;
}

// ---- power / reciprocal ---------------------------------------------------

/// `Sqrt` functor (`itkSqrtImageFilter.h`): `std::sqrt`.
struct Sqrt;
impl UnaryFunctor for Sqrt {
    fn apply(&self, x: f64) -> f64 {
        x.sqrt()
    }
}

functor::unary_functor! {
    /// `SqrtImageFilter`: pixel-wise square root (`std::sqrt`); `NaN` below `0`.
    pub fn sqrt, sqrt_in_place() = Sqrt;
}

/// `Square` functor (`itkSquareImageFilter.h`): `ra = RealType(A); ra * ra`.
struct Square;
impl UnaryFunctor for Square {
    fn apply(&self, x: f64) -> f64 {
        x * x
    }
}

functor::unary_functor! {
    /// `SquareImageFilter`: pixel-wise `x^2`.
    pub fn square, square_in_place() = Square;
}

/// `BoundedReciprocal` functor (`itkBoundedReciprocalImageFilter.h`): `1.0 / (1.0 + x)`.
struct BoundedReciprocal;
impl UnaryFunctor for BoundedReciprocal {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + x)
    }
}

functor::unary_functor! {
    /// `BoundedReciprocalImageFilter`: pixel-wise `1 / (1 + x)`.
    pub fn bounded_reciprocal, bounded_reciprocal_in_place() = BoundedReciprocal;
}

// ---- two-image f64-compute shared helpers ------------------------------
//
// Shared by every two-image filter below (squared_difference,
// absolute_value_difference, atan2, binary_magnitude, divide_floor): cast
// both operands to `f64`, combine, narrow once. `divide_real` needs a
// different output pixel type (its `NumericTraits<T>::RealType` promotion),
// so it takes `output_id` explicitly instead of defaulting to `a`'s.

fn two_image_f64_with_output(
    a: &Image,
    b: &Image,
    output_id: PixelId,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Image> {
    require_same_shape(a, b)?;
    require_same_physical_space(a, b, 1)?;
    let va = a.to_f64_vec()?;
    let vb = b.to_f64_vec()?;
    let out: Vec<f64> = va.iter().zip(&vb).map(|(&x, &y)| f(x, y)).collect();
    image_from_f64(output_id, a.size(), a, &out)
}

fn two_image_f64(a: &Image, b: &Image, f: impl Fn(f64, f64) -> f64) -> Result<Image> {
    two_image_f64_with_output(a, b, a.pixel_id(), f)
}

fn two_image_f64_typed_in_place<T: Scalar>(
    img: &mut Image,
    other: &[f64],
    f: &dyn Fn(f64, f64) -> f64,
) -> Result<()> {
    let v = img.scalar_vec_mut::<T>()?;
    for (x, &y) in v.iter_mut().zip(other) {
        *x = T::from_f64(f(x.as_f64(), y));
    }
    Ok(())
}

fn two_image_f64_in_place(mut a: Image, b: &Image, f: &dyn Fn(f64, f64) -> f64) -> Result<Image> {
    require_same_shape(&a, b)?;
    require_same_physical_space(&a, b, 1)?;
    let vb = b.to_f64_vec()?;
    dispatch_scalar!(a.pixel_id(), two_image_f64_typed_in_place, &mut a, &vb, f)?;
    Ok(a)
}

// ---- squared difference (two images, f64-compute) --------------------------

/// `SquaredDifferenceImageFilter`: pixel-wise `(a - b)^2`, computed in `f64`
/// (`itkSquaredDifferenceImageFilter.h`'s `SquaredDifference2` functor). See
/// the module docs for why this bypasses the `BinaryFunctor` seam.
pub fn squared_difference(a: &Image, b: &Image) -> Result<Image> {
    two_image_f64(a, b, |x, y| {
        let diff = x - y;
        diff * diff
    })
}

/// In-place variant of [`squared_difference`]: reuses `a`'s buffer.
pub fn squared_difference_in_place(a: Image, b: &Image) -> Result<Image> {
    two_image_f64_in_place(a, b, &|x, y| {
        let diff = x - y;
        diff * diff
    })
}

// ---- absolute value difference (two images, f64-compute) ------------------

/// `AbsoluteValueDifferenceImageFilter`
/// (`itkAbsoluteValueDifferenceImageFilter.h`'s `AbsoluteValueDifference2`
/// functor): pixel-wise `|a - b|`, computed in `f64`
/// (`dA = double(A); dB = double(B); diff = dA - dB; abs = diff > 0 ? diff :
/// -diff`).
pub fn absolute_value_difference(a: &Image, b: &Image) -> Result<Image> {
    two_image_f64(a, b, |x, y| (x - y).abs())
}

/// In-place variant of [`absolute_value_difference`]: reuses `a`'s buffer.
pub fn absolute_value_difference_in_place(a: Image, b: &Image) -> Result<Image> {
    two_image_f64_in_place(a, b, &|x, y| (x - y).abs())
}

// ---- atan2 (two images, f64-compute) ---------------------------------------

/// `Atan2ImageFilter` (`itkAtan2ImageFilter.h`'s `Atan2` functor): pixel-wise
/// two-argument inverse tangent, `std::atan2(double(a), double(b))` — `a` is
/// the first input (`SetInput1`, the `y` argument), `b` the second
/// (`SetInput2`, the `x` argument).
pub fn atan2(a: &Image, b: &Image) -> Result<Image> {
    two_image_f64(a, b, |x, y| x.atan2(y))
}

/// In-place variant of [`atan2`]: reuses `a`'s buffer.
pub fn atan2_in_place(a: Image, b: &Image) -> Result<Image> {
    two_image_f64_in_place(a, b, &|x, y| x.atan2(y))
}

// ---- binary magnitude (two images, f64-compute) ----------------------------

/// `BinaryMagnitudeImageFilter` (`itkBinaryMagnitudeImageFilter.h`'s
/// `Modulus2` functor): pixel-wise `sqrt(a^2 + b^2)`, computed in `f64`
/// (`dA = double(A); dB = double(B); sqrt(dA*dA + dB*dB)`).
pub fn binary_magnitude(a: &Image, b: &Image) -> Result<Image> {
    two_image_f64(a, b, |x, y| (x * x + y * y).sqrt())
}

/// In-place variant of [`binary_magnitude`]: reuses `a`'s buffer.
pub fn binary_magnitude_in_place(a: Image, b: &Image) -> Result<Image> {
    two_image_f64_in_place(a, b, &|x, y| (x * x + y * y).sqrt())
}

// ---- divide floor / divide real (two images, f64-compute) -----------------

/// `DivideFloorImageFilter` (`itkArithmeticOpsFunctors.h`'s `DivFloor`
/// functor): pixel-wise `floor(double(a) / double(b))`, output pixel type is
/// `a`'s own type (SimpleITK's `filter_type` pins the C++ output image type
/// to `InputImageType`, not a promoted `OutputImageType`).
///
/// ITK special-cases an infinite `floor` result for integral output types
/// (`b == 0` with `a != 0`) to `NumericTraits<TOutput>::max()`/
/// `NonpositiveMin()` rather than a UB `static_cast<TOutput>(inf)`; this
/// crate's [`Scalar::from_f64`] narrows `f64::INFINITY`/`NEG_INFINITY` to
/// exactly those same type-max/type-min values via Rust's saturating `as`
/// cast, so no separate branch is needed here — the general f64-compute
/// narrowing already reproduces ITK's special case. The genuine `0 / 0`
/// case (`a == 0 && b == 0`) is `NaN` after `floor`; ITK's `isinf(NaN)` check
/// is false there, so it falls through to a UB `static_cast<TOutput>(NaN)`
/// (unspecified in C++), while this crate's saturating cast defines it as
/// `0` for integer outputs (matching this crate's established
/// `NaN`-narrows-to-`0` policy, e.g. `math::sqrt`'s negative-input tests) and
/// preserves `NaN` exactly for float outputs.
pub fn divide_floor(a: &Image, b: &Image) -> Result<Image> {
    two_image_f64(a, b, |x, y| (x / y).floor())
}

/// In-place variant of [`divide_floor`]: reuses `a`'s buffer.
pub fn divide_floor_in_place(a: Image, b: &Image) -> Result<Image> {
    two_image_f64_in_place(a, b, &|x, y| (x / y).floor())
}

/// `PowImageFilter` (`itkPowImageFilter.h`'s `Pow` functor): pixel-wise
/// `pow(RealType1(a), RealType2(b))`, output pixel type is `a`'s own type
/// (the functor's default template parameters: `TInput2 = TInput1`,
/// `TOutput = TInput1`, and the yaml has no `output_image_type` override).
///
/// Unlike every other filter in this module, this is an *exact* match to the
/// C++, not a precision simplification: `NumericTraits<T>::RealType` is
/// `double` for every basic pixel type, including `float`
/// (`itkNumericTraits.h`'s `NumericTraits<float>::RealType = double`), so ITK
/// itself always calls `std::pow(double, double)` here regardless of `a`/`b`'s
/// concrete input types -- exactly what `two_image_f64` does by promoting
/// both operands to `f64`.
pub fn pow(a: &Image, b: &Image) -> Result<Image> {
    two_image_f64(a, b, |x, y| x.powf(y))
}

/// In-place variant of [`pow`]: reuses `a`'s buffer.
pub fn pow_in_place(a: Image, b: &Image) -> Result<Image> {
    two_image_f64_in_place(a, b, &|x, y| x.powf(y))
}

/// Output pixel-type mapping used by [`divide_real`]: stays `Float32` for a
/// `Float32` input, promotes everything else to `Float64`. **Diverges from
/// ITK**: the yaml's `output_pixel_type` is `NumericTraits<T>::RealType`,
/// which is `double` for every scalar type *including* `float`
/// (itkNumericTraits.h:1349/1356) — upstream always outputs `Float64`.
/// Flipping `Float32 → Float64` is a breaking change tracked in the
/// upstream-findings ledger §5.6. Mirrors `intensity::real_type` /
/// `projection::real_type` / `fft_correlation::real_type` /
/// `lib.rs::real_pixel_id` (same family).
fn real_type(id: PixelId) -> PixelId {
    match id {
        PixelId::Float32 => PixelId::Float32,
        _ => PixelId::Float64,
    }
}

/// `DivideRealImageFilter` (`itkArithmeticOpsFunctors.h`'s `DivReal`
/// functor): pixel-wise `RealType(a) / RealType(b)`. The output pixel type
/// is `a`'s `NumericTraits<T>::RealType` (see `real_type`), matching
/// `DivideRealImageFilter.yaml`'s `output_pixel_type` override -- unlike
/// [`divide_floor`], the output is always real, so there is no
/// integer-narrowing special case: `b == 0` naturally yields `+inf`/`-inf`/
/// `NaN` under IEEE 754, preserved exactly since the output pixel type is
/// always floating-point. The division itself runs in `f64`, which **matches**
/// `DivReal`'s `RealType(A) / RealType(B)` exactly — `RealType` is `double`
/// for every scalar input type, `Float32` included. The only divergence is
/// the output pixel type for `Float32` inputs (see `real_type`, §5.6).
///
/// No in-place variant: like [`intensity::normalize`](crate::filters::intensity::normalize),
/// the output pixel type does not generally match the input's, so there is
/// no buffer to reuse.
pub fn divide_real(a: &Image, b: &Image) -> Result<Image> {
    let output_id = real_type(a.pixel_id());
    two_image_f64_with_output(a, b, output_id, |x, y| x / y)
}

// ---- nary reductions (N images, f64-compute) -------------------------------

/// Every nary filter below needs at least one input image; when there is
/// more than one, every image must share the first's pixel type and size
/// (`itkNaryFunctorImageFilter.h` is one ITK template instantiation over a
/// single pixel type, so ITK enforces this at compile time; here it's a
/// runtime check). Mirrors `label_fusion::require_inputs`, duplicated here
/// since that helper is private to its own module.
fn require_inputs(images: &[&Image]) -> Result<()> {
    let Some((first, rest)) = images.split_first() else {
        return Err(FilterError::EmptyImageList);
    };
    for (i, img) in rest.iter().enumerate() {
        require_same_shape(first, img)?;
        require_same_physical_space(first, img, i + 1)?;
    }
    Ok(())
}

/// Shared by [`nary_add`]/[`nary_maximum`]: fold every input image's pixel
/// values through `f`, seeded with `init`. Output pixel type is the first
/// input's.
fn nary_reduce_f64(images: &[&Image], init: f64, f: impl Fn(f64, f64) -> f64) -> Result<Image> {
    require_inputs(images)?;
    let first = images[0];
    let mut out = vec![init; first.size().iter().product()];
    for img in images {
        for (o, x) in out.iter_mut().zip(img.to_f64_vec()?) {
            *o = f(*o, x);
        }
    }
    image_from_f64(first.pixel_id(), first.size(), first, &out)
}

/// `NaryAddImageFilter` (`itkNaryAddImageFilter.h`'s `Add1` functor):
/// pixel-wise sum of every input image. See the module docs for how this
/// diverges from `Add1`'s wide-accumulator-then-narrow-once C++.
pub fn nary_add(images: &[&Image]) -> Result<Image> {
    nary_reduce_f64(images, 0.0, |acc, x| acc + x)
}

/// `NaryMaximumImageFilter` (`itkNaryMaximumImageFilter.h`'s `Maximum1`
/// functor): pixel-wise maximum across every input image. See the module
/// docs for how this diverges from `Maximum1`'s native-`TOutput` fold.
pub fn nary_maximum(images: &[&Image]) -> Result<Image> {
    nary_reduce_f64(images, f64::NEG_INFINITY, f64::max)
}

// ---- ternary ops (three images, f64-compute) -------------------------------

fn three_image_f64(
    a: &Image,
    b: &Image,
    c: &Image,
    f: impl Fn(f64, f64, f64) -> f64,
) -> Result<Image> {
    require_same_shape(a, b)?;
    require_same_shape(a, c)?;
    require_same_physical_space(a, b, 1)?;
    require_same_physical_space(a, c, 2)?;
    let va = a.to_f64_vec()?;
    let vb = b.to_f64_vec()?;
    let vc = c.to_f64_vec()?;
    let out: Vec<f64> = va
        .iter()
        .zip(&vb)
        .zip(&vc)
        .map(|((&x, &y), &z)| f(x, y, z))
        .collect();
    image_from_f64(a.pixel_id(), a.size(), a, &out)
}

fn three_image_f64_typed_in_place<T: Scalar>(
    img: &mut Image,
    other_b: &[f64],
    other_c: &[f64],
    f: &dyn Fn(f64, f64, f64) -> f64,
) -> Result<()> {
    let v = img.scalar_vec_mut::<T>()?;
    for ((x, &y), &z) in v.iter_mut().zip(other_b).zip(other_c) {
        *x = T::from_f64(f(x.as_f64(), y, z));
    }
    Ok(())
}

fn three_image_f64_in_place(
    mut a: Image,
    b: &Image,
    c: &Image,
    f: &dyn Fn(f64, f64, f64) -> f64,
) -> Result<Image> {
    require_same_shape(&a, b)?;
    require_same_shape(&a, c)?;
    require_same_physical_space(&a, b, 1)?;
    require_same_physical_space(&a, c, 2)?;
    let vb = b.to_f64_vec()?;
    let vc = c.to_f64_vec()?;
    dispatch_scalar!(
        a.pixel_id(),
        three_image_f64_typed_in_place,
        &mut a,
        &vb,
        &vc,
        f
    )?;
    Ok(a)
}

/// `TernaryAddImageFilter` (`itkArithmeticOpsFunctors.h`'s `Add3` functor):
/// pixel-wise `a + b + c`. See the module docs for how this diverges from
/// `Add3`'s native-pixel-type C++ (and from its own `nary_add` sibling,
/// which upstream uses a wide accumulator that `Add3` does not).
pub fn ternary_add(a: &Image, b: &Image, c: &Image) -> Result<Image> {
    three_image_f64(a, b, c, |x, y, z| x + y + z)
}

/// In-place variant of [`ternary_add`]: reuses `a`'s buffer.
pub fn ternary_add_in_place(a: Image, b: &Image, c: &Image) -> Result<Image> {
    three_image_f64_in_place(a, b, c, &|x, y, z| x + y + z)
}

/// `TernaryMagnitudeImageFilter` (`itkTernaryMagnitudeImageFilter.h`'s
/// `Modulus3` functor): pixel-wise `sqrt(a^2 + b^2 + c^2)`. ITK's raw
/// `Modulus3` already computes this in `double`
/// (`static_cast<TOutput>(std::sqrt(static_cast<double>(A*A + B*B +
/// C*C)))`), so this port matches it exactly -- no divergence to document.
pub fn ternary_magnitude(a: &Image, b: &Image, c: &Image) -> Result<Image> {
    three_image_f64(a, b, c, |x, y, z| (x * x + y * y + z * z).sqrt())
}

/// In-place variant of [`ternary_magnitude`]: reuses `a`'s buffer.
pub fn ternary_magnitude_in_place(a: Image, b: &Image, c: &Image) -> Result<Image> {
    three_image_f64_in_place(a, b, c, &|x, y, z| (x * x + y * y + z * z).sqrt())
}

/// `TernaryMagnitudeSquaredImageFilter`
/// (`itkTernaryMagnitudeSquaredImageFilter.h`'s `ModulusSquare3` functor):
/// pixel-wise `a^2 + b^2 + c^2`. See the module docs for how this diverges
/// from `ModulusSquare3`'s native-pixel-type C++ (and from its own
/// `ternary_magnitude` sibling, which upstream *does* compute in `double`
/// despite living in the same functor header family).
pub fn ternary_magnitude_squared(a: &Image, b: &Image, c: &Image) -> Result<Image> {
    three_image_f64(a, b, c, |x, y, z| x * x + y * y + z * z)
}

/// In-place variant of [`ternary_magnitude_squared`]: reuses `a`'s buffer.
pub fn ternary_magnitude_squared_in_place(a: Image, b: &Image, c: &Image) -> Result<Image> {
    three_image_f64_in_place(a, b, c, &|x, y, z| x * x + y * y + z * z)
}

// ---- round tests --------------------------------------------------------

#[cfg(test)]
mod round_tests {
    use super::*;

    #[test]
    fn round_half_up_ties_and_ordinary_values() {
        // RoundHalfIntegerUp: ties round up, not to even and not away from zero.
        let a = Image::from_vec(&[6, 1], vec![1.5f64, -1.5, 2.5, -2.5, 0.5, -0.5]).unwrap();
        assert_eq!(
            round(&a).unwrap().scalar_slice::<f64>().unwrap(),
            &[2.0, -1.0, 3.0, -2.0, 1.0, 0.0]
        );
    }

    #[test]
    fn round_ordinary_non_halfway_values() {
        let a = Image::from_vec(&[4, 1], vec![1.1f64, 1.9, -1.1, -1.9]).unwrap();
        assert_eq!(
            round(&a).unwrap().scalar_slice::<f64>().unwrap(),
            &[1.0, 2.0, -1.0, -2.0]
        );
    }

    #[test]
    fn round_in_place_matches_allocating() {
        let a = Image::from_vec(&[3, 1], vec![1.5f64, -1.5, 0.4]).unwrap();
        let allocated = round(&a).unwrap();
        let in_place = round_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn round_output_pixel_type_matches_input() {
        let f32_img = Image::from_vec(&[1, 1], vec![1.5f32]).unwrap();
        let out = round(&f32_img).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[2.0f32]);

        let f64_img = Image::from_vec(&[1, 1], vec![1.5f64]).unwrap();
        let out = round(&f64_img).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
    }

    #[test]
    fn round_rejects_non_real_pixel_type() {
        let a = Image::from_vec(&[2, 1], vec![1u8, 2]).unwrap();
        assert_eq!(
            round(&a),
            Err(FilterError::RequiresRealPixelType(PixelId::UInt8))
        );
    }

    #[test]
    fn round_rejects_a_complex_pixel_type() {
        let a = Image::new(&[2, 1], PixelId::ComplexFloat32);
        assert_eq!(
            round(&a),
            Err(FilterError::RequiresRealPixelType(PixelId::ComplexFloat32))
        );
    }

    #[test]
    fn round_large_magnitude_is_defined_unlike_upstreams_int32_intermediate() {
        // 3e9 exceeds i32::MAX (~2.147e9): ITK's Round<float, float> routes
        // through an int32_t intermediate here (see the module docs), which
        // is undefined behavior in C++ for this magnitude. This port's
        // f64-compute engine has no such intermediate and always returns the
        // mathematically correct (here: already-integral, no-op) result.
        let a = Image::from_vec(&[1, 1], vec![3.0e9f32]).unwrap();
        assert_eq!(
            round(&a).unwrap().scalar_slice::<f32>().unwrap(),
            &[3.0e9f32]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- abs ----

    #[test]
    fn abs_negative_values() {
        let a = Image::from_vec(&[3, 1], vec![-3i16, 0, 7]).unwrap();
        assert_eq!(abs(&a).unwrap().scalar_slice::<i16>().unwrap(), &[3, 0, 7]);
    }

    #[test]
    fn abs_in_place_matches_allocating() {
        let a = Image::from_vec(&[3, 1], vec![-3i16, 0, 7]).unwrap();
        let allocated = abs(&a).unwrap();
        let in_place = abs_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- sqrt of a negative: NaN under Scalar::from_f64 ----

    #[test]
    fn sqrt_negative_saturates_to_zero_on_integer_output() {
        // f64::sqrt(-1.0) is NaN; Scalar::from_f64 for an integer type maps
        // NaN to 0 (Rust's `as` semantics), so this differs from C++
        // static_cast's undefined behavior by design (see pixel.rs docs).
        let a = img_u8(&[2, 1], vec![4, 0]);
        assert_eq!(sqrt(&a).unwrap().scalar_slice::<u8>().unwrap(), &[2, 0]);

        let neg = Image::from_vec(&[1, 1], vec![-1.0f32]).unwrap();
        let cast_to_u8 = crate::filters::cast(&neg, crate::core::PixelId::UInt8).unwrap();
        assert_eq!(
            sqrt(&cast_to_u8).unwrap().scalar_slice::<u8>().unwrap(),
            &[0]
        );
    }

    #[test]
    fn sqrt_negative_preserves_nan_on_float_output() {
        let a = Image::from_vec(&[1, 1], vec![-1.0f64]).unwrap();
        let out = sqrt(&a).unwrap();
        assert!(out.scalar_slice::<f64>().unwrap()[0].is_nan());
    }

    // ---- log(0) ----

    #[test]
    fn log_of_zero_is_negative_infinity_on_float_output() {
        let a = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        let out = log(&a).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap()[0], f64::NEG_INFINITY);
    }

    #[test]
    fn log_of_zero_saturates_to_type_min_on_integer_output() {
        // f64::ln(0.0) is -inf; Scalar::from_f64 for i16 maps -inf to
        // i16::MIN (Rust's saturating `as` semantics).
        let a = img_u8(&[1, 1], vec![0]);
        let cast_to_i16 = crate::filters::cast(&a, crate::core::PixelId::Int16).unwrap();
        assert_eq!(
            log(&cast_to_i16).unwrap().scalar_slice::<i16>().unwrap(),
            &[i16::MIN]
        );
    }

    // ---- bounded_reciprocal at 0 ----

    #[test]
    fn bounded_reciprocal_at_zero_is_one() {
        let a = Image::from_vec(&[2, 1], vec![0.0f64, 1.0]).unwrap();
        assert_eq!(
            bounded_reciprocal(&a)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap(),
            &[1.0, 0.5]
        );
    }

    // ---- square overflow on u8 ----

    #[test]
    fn square_overflow_saturates_on_u8() {
        // 250.0^2 = 62500.0, which does not fit in u8; Scalar::from_f64
        // saturates to 255 (contrast lib.rs's `multiply`, whose
        // pixel-type-compute policy wraps instead).
        let a = img_u8(&[2, 1], vec![250, 10]);
        assert_eq!(
            square(&a).unwrap().scalar_slice::<u8>().unwrap(),
            &[255, 100]
        );
    }

    #[test]
    fn square_in_place_matches_allocating() {
        let a = img_u8(&[2, 1], vec![250, 10]);
        let allocated = square(&a).unwrap();
        let in_place = square_in_place(a).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- other unary math: basic sanity ----

    #[test]
    fn trig_and_exp_basic_values() {
        let zero = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        assert_eq!(
            acos(&zero).unwrap().scalar_slice::<f64>().unwrap()[0],
            0.0f64.acos()
        );
        assert_eq!(asin(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(atan(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(cos(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 1.0);
        assert_eq!(sin(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(tan(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 0.0);
        assert_eq!(exp(&zero).unwrap().scalar_slice::<f64>().unwrap()[0], 1.0);
        assert_eq!(
            exp_negative(&zero).unwrap().scalar_slice::<f64>().unwrap()[0],
            1.0
        );
        assert_eq!(
            log10(&zero).unwrap().scalar_slice::<f64>().unwrap()[0],
            f64::NEG_INFINITY
        );
    }

    // ---- squared_difference ----

    #[test]
    fn squared_difference_basic() {
        let a = img_u8(&[3, 1], vec![5, 0, 250]);
        let b = img_u8(&[3, 1], vec![10, 0, 240]);
        // (5-10)^2=25, (0-0)^2=0, (250-240)^2=100 -- all fit u8, no wrap
        // (unlike a pixel-type-compute `5u8 - 10u8` which would wrap first).
        assert_eq!(
            squared_difference(&a, &b)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[25, 0, 100]
        );
    }

    #[test]
    fn squared_difference_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![5, 0, 250]);
        let b = img_u8(&[3, 1], vec![10, 0, 240]);
        let allocated = squared_difference(&a, &b).unwrap();
        let in_place = squared_difference_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn squared_difference_mismatched_inputs_error() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        assert!(matches!(
            squared_difference(&a, &b),
            Err(crate::filters::FilterError::TypeMismatch { .. })
        ));
    }

    // ---- absolute_value_difference ----

    #[test]
    fn absolute_value_difference_basic() {
        let a = img_u8(&[3, 1], vec![5, 10, 250]);
        let b = img_u8(&[3, 1], vec![10, 5, 10]);
        assert_eq!(
            absolute_value_difference(&a, &b)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[5, 5, 240]
        );
    }

    #[test]
    fn absolute_value_difference_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![5, 10, 250]);
        let b = img_u8(&[3, 1], vec![10, 5, 10]);
        let allocated = absolute_value_difference(&a, &b).unwrap();
        let in_place = absolute_value_difference_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- atan2 ----

    #[test]
    fn atan2_basic_values_and_argument_order() {
        // Atan2(A, B) = std::atan2(double(A), double(B)): A is the first
        // input (the `y` argument), B the second (the `x` argument), so
        // atan2(1, 0) != atan2(0, 1).
        let a = Image::from_vec(&[2, 1], vec![1.0f64, 0.0]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![0.0f64, 1.0]).unwrap();
        let out = atan2(&a, &b).unwrap();
        assert_eq!(
            out.scalar_slice::<f64>().unwrap(),
            &[std::f64::consts::FRAC_PI_2, 0.0]
        );
    }

    #[test]
    fn atan2_in_place_matches_allocating() {
        let a = Image::from_vec(&[2, 1], vec![1.0f64, -1.0]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![1.0f64, 0.0]).unwrap();
        let allocated = atan2(&a, &b).unwrap();
        let in_place = atan2_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- binary_magnitude ----

    #[test]
    fn binary_magnitude_3_4_5_triangle() {
        let a = img_u8(&[1, 1], vec![3]);
        let b = img_u8(&[1, 1], vec![4]);
        assert_eq!(
            binary_magnitude(&a, &b)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[5]
        );
    }

    #[test]
    fn binary_magnitude_in_place_matches_allocating() {
        let a = img_u8(&[1, 1], vec![3]);
        let b = img_u8(&[1, 1], vec![4]);
        let allocated = binary_magnitude(&a, &b).unwrap();
        let in_place = binary_magnitude_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- divide_floor ----

    #[test]
    fn divide_floor_basic_and_negative() {
        let a = Image::from_vec(&[2, 1], vec![7i32, -7]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![2i32, 2]).unwrap();
        // floor(7/2) = 3; floor(-7/2) = floor(-3.5) = -4 (not truncation).
        assert_eq!(
            divide_floor(&a, &b).unwrap().scalar_slice::<i32>().unwrap(),
            &[3, -4]
        );
    }

    #[test]
    fn divide_floor_by_zero_saturates_to_type_max_or_min_on_integer_output() {
        let a = Image::from_vec(&[2, 1], vec![5i32, -5]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![0i32, 0]).unwrap();
        // floor(5.0/0.0) = floor(+inf) = +inf -> i32::MAX (NumericTraits::max);
        // floor(-5.0/0.0) = floor(-inf) = -inf -> i32::MIN (NonpositiveMin).
        assert_eq!(
            divide_floor(&a, &b).unwrap().scalar_slice::<i32>().unwrap(),
            &[i32::MAX, i32::MIN]
        );
    }

    #[test]
    fn divide_floor_zero_by_zero_is_zero_on_integer_output_nan_on_float_output() {
        let a = img_u8(&[1, 1], vec![0]);
        let b = img_u8(&[1, 1], vec![0]);
        // floor(0.0/0.0) = floor(NaN) = NaN; ITK's isinf(NaN) check is false,
        // so it falls through to a UB static_cast<TOutput>(NaN) in C++. This
        // crate's Scalar::from_f64 defines NaN -> 0 for integer outputs.
        assert_eq!(
            divide_floor(&a, &b).unwrap().scalar_slice::<u8>().unwrap(),
            &[0]
        );

        let af = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        let bf = Image::from_vec(&[1, 1], vec![0.0f64]).unwrap();
        assert!(
            divide_floor(&af, &bf)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap()[0]
                .is_nan()
        );
    }

    #[test]
    fn divide_floor_in_place_matches_allocating() {
        let a = Image::from_vec(&[2, 1], vec![7i32, -7]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![2i32, 2]).unwrap();
        let allocated = divide_floor(&a, &b).unwrap();
        let in_place = divide_floor_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- pow ----

    #[test]
    fn pow_basic_integer_output() {
        let a = Image::from_vec(&[3, 1], vec![2i32, 3, 5]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![10i32, 2, 0]).unwrap();
        // pow(2,10)=1024; pow(3,2)=9; pow(5,0)=1.
        assert_eq!(
            pow(&a, &b).unwrap().scalar_slice::<i32>().unwrap(),
            &[1024, 9, 1]
        );
    }

    #[test]
    fn pow_keeps_a_own_pixel_type_as_output() {
        let a = Image::from_vec(&[1, 1], vec![2.0f32]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![0.5f32]).unwrap();
        let out = pow(&a, &b).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::Float32);
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[std::f32::consts::SQRT_2]
        );
    }

    #[test]
    fn pow_overflow_saturates_on_integer_output() {
        // 2^10 = 1024, does not fit in u8; Scalar::from_f64 saturates to 255.
        let a = img_u8(&[1, 1], vec![2]);
        let b = img_u8(&[1, 1], vec![10]);
        assert_eq!(pow(&a, &b).unwrap().scalar_slice::<u8>().unwrap(), &[255]);
    }

    #[test]
    fn pow_negative_base_fractional_exponent_is_nan_on_float_output() {
        // std::pow(-1.0, 0.5) is NaN (no real result); f64::powf matches.
        let a = Image::from_vec(&[1, 1], vec![-1.0f64]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![0.5f64]).unwrap();
        assert!(pow(&a, &b).unwrap().scalar_slice::<f64>().unwrap()[0].is_nan());
    }

    #[test]
    fn pow_zero_to_the_zero_is_one() {
        // std::pow(0.0, 0.0) == 1.0 under IEC 60559 (and f64::powf matches);
        // not a special case ITK guards against, just a real edge worth
        // pinning since it looks like it should be 0.
        let a = img_u8(&[1, 1], vec![0]);
        let b = img_u8(&[1, 1], vec![0]);
        assert_eq!(pow(&a, &b).unwrap().scalar_slice::<u8>().unwrap(), &[1]);
    }

    #[test]
    fn pow_in_place_matches_allocating() {
        let a = Image::from_vec(&[3, 1], vec![2i32, 3, 5]).unwrap();
        let b = Image::from_vec(&[3, 1], vec![10i32, 2, 0]).unwrap();
        let allocated = pow(&a, &b).unwrap();
        let in_place = pow_in_place(a, &b).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- divide_real ----

    #[test]
    fn divide_real_promotes_integer_input_to_float64() {
        let a = img_u8(&[1, 1], vec![1]);
        let b = img_u8(&[1, 1], vec![4]);
        let out = divide_real(&a, &b).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::Float64);
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[0.25]);
    }

    #[test]
    fn divide_real_keeps_float32_output_as_float32() {
        let a = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![4.0f32]).unwrap();
        let out = divide_real(&a, &b).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::Float32);
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[0.25]);
    }

    #[test]
    fn divide_real_by_zero_is_infinite_not_saturated() {
        // Unlike divide_floor, DivReal's output is always real, so b == 0
        // just yields IEEE 754 +-inf / NaN, never a NumericTraits max/min
        // substitution.
        let a = img_u8(&[3, 1], vec![5, 0, 0]);
        let b = img_u8(&[3, 1], vec![0, 0, 5]);
        let out = divide_real(&a, &b).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        assert_eq!(vals[0], f64::INFINITY);
        assert!(vals[1].is_nan());
        assert_eq!(vals[2], 0.0);
    }

    // ---- nary_add ----

    #[test]
    fn nary_add_empty_list_is_error() {
        assert_eq!(
            nary_add(&[]),
            Err(crate::filters::FilterError::EmptyImageList)
        );
    }

    #[test]
    fn nary_add_single_image_is_identity() {
        let a = img_u8(&[3, 1], vec![1, 2, 3]);
        assert_eq!(
            nary_add(&[&a]).unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 2, 3]
        );
    }

    #[test]
    fn nary_add_sums_across_all_images() {
        let a = img_u8(&[3, 1], vec![1, 2, 3]);
        let b = img_u8(&[3, 1], vec![10, 20, 30]);
        let c = img_u8(&[3, 1], vec![100, 100, 100]);
        assert_eq!(
            nary_add(&[&a, &b, &c])
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[111, 122, 133]
        );
    }

    #[test]
    fn nary_add_mismatched_pixel_type_is_error() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        assert!(matches!(
            nary_add(&[&a, &b]),
            Err(crate::filters::FilterError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn nary_add_mismatched_size_is_error() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = img_u8(&[3, 1], vec![1, 2, 3]);
        assert!(matches!(
            nary_add(&[&a, &b]),
            Err(crate::filters::FilterError::SizeMismatch { .. })
        ));
    }

    // ---- nary_maximum ----

    #[test]
    fn nary_maximum_empty_list_is_error() {
        assert_eq!(
            nary_maximum(&[]),
            Err(crate::filters::FilterError::EmptyImageList)
        );
    }

    #[test]
    fn nary_maximum_single_image_is_identity() {
        let a = img_u8(&[3, 1], vec![1, 2, 3]);
        assert_eq!(
            nary_maximum(&[&a]).unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 2, 3]
        );
    }

    #[test]
    fn nary_maximum_picks_pixelwise_max_across_all_images() {
        let a = img_u8(&[3, 1], vec![5, 200, 0]);
        let b = img_u8(&[3, 1], vec![10, 100, 255]);
        let c = img_u8(&[3, 1], vec![7, 150, 128]);
        assert_eq!(
            nary_maximum(&[&a, &b, &c])
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[10, 200, 255]
        );
    }

    // ---- ternary_add ----

    #[test]
    fn ternary_add_basic() {
        let a = img_u8(&[3, 1], vec![1, 2, 3]);
        let b = img_u8(&[3, 1], vec![10, 20, 30]);
        let c = img_u8(&[3, 1], vec![100, 100, 100]);
        assert_eq!(
            ternary_add(&a, &b, &c)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[111, 122, 133]
        );
    }

    #[test]
    fn ternary_add_in_place_matches_allocating() {
        let a = img_u8(&[3, 1], vec![1, 2, 3]);
        let b = img_u8(&[3, 1], vec![10, 20, 30]);
        let c = img_u8(&[3, 1], vec![100, 100, 100]);
        let allocated = ternary_add(&a, &b, &c).unwrap();
        let in_place = ternary_add_in_place(a, &b, &c).unwrap();
        assert_eq!(allocated, in_place);
    }

    #[test]
    fn ternary_add_mismatched_second_input_is_error() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let c = img_u8(&[2, 1], vec![1, 2]);
        assert!(matches!(
            ternary_add(&a, &b, &c),
            Err(crate::filters::FilterError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn ternary_add_mismatched_third_input_is_error() {
        let a = img_u8(&[2, 1], vec![1, 2]);
        let b = img_u8(&[2, 1], vec![1, 2]);
        let c = img_u8(&[3, 1], vec![1, 2, 3]);
        assert!(matches!(
            ternary_add(&a, &b, &c),
            Err(crate::filters::FilterError::SizeMismatch { .. })
        ));
    }

    // ---- ternary_magnitude ----

    #[test]
    fn ternary_magnitude_2_3_6_is_7() {
        let a = img_u8(&[1, 1], vec![2]);
        let b = img_u8(&[1, 1], vec![3]);
        let c = img_u8(&[1, 1], vec![6]);
        // sqrt(2^2 + 3^2 + 6^2) = sqrt(49) = 7, an exact integer result.
        assert_eq!(
            ternary_magnitude(&a, &b, &c)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[7]
        );
    }

    #[test]
    fn ternary_magnitude_in_place_matches_allocating() {
        let a = img_u8(&[1, 1], vec![2]);
        let b = img_u8(&[1, 1], vec![3]);
        let c = img_u8(&[1, 1], vec![6]);
        let allocated = ternary_magnitude(&a, &b, &c).unwrap();
        let in_place = ternary_magnitude_in_place(a, &b, &c).unwrap();
        assert_eq!(allocated, in_place);
    }

    // ---- ternary_magnitude_squared ----

    #[test]
    fn ternary_magnitude_squared_2_3_6_is_49() {
        let a = img_u8(&[1, 1], vec![2]);
        let b = img_u8(&[1, 1], vec![3]);
        let c = img_u8(&[1, 1], vec![6]);
        // 2^2 + 3^2 + 6^2 = 4 + 9 + 36 = 49, fits in u8 with no saturation.
        assert_eq!(
            ternary_magnitude_squared(&a, &b, &c)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[49]
        );
    }

    #[test]
    fn ternary_magnitude_squared_saturates_instead_of_wrapping_on_u8() {
        // 200^2 + 200^2 + 200^2 = 120000, which does not fit in u8.
        // Scalar::from_f64 saturates to 255; the raw C++ ModulusSquare3
        // computes natively in u8 and would instead wrap the intermediate
        // squared terms (200u8 * 200u8 overflows before the sum is taken).
        let a = img_u8(&[1, 1], vec![200]);
        let b = img_u8(&[1, 1], vec![200]);
        let c = img_u8(&[1, 1], vec![200]);
        assert_eq!(
            ternary_magnitude_squared(&a, &b, &c)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[255]
        );
    }

    #[test]
    fn ternary_magnitude_squared_in_place_matches_allocating() {
        let a = img_u8(&[1, 1], vec![2]);
        let b = img_u8(&[1, 1], vec![3]);
        let c = img_u8(&[1, 1], vec![6]);
        let allocated = ternary_magnitude_squared(&a, &b, &c).unwrap();
        let in_place = ternary_magnitude_squared_in_place(a, &b, &c).unwrap();
        assert_eq!(allocated, in_place);
    }
}
