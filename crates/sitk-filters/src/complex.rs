//! Filters that cross the complex/real pixel boundary.
//!
//! SimpleITK groups these under `ImageIntensity` and `ImageCompose`. They share
//! the property that separates them from the rest of this crate: their input or
//! output pixel type is `std::complex<float>` or `std::complex<double>` — a
//! *basic* pixel type upstream, not a vector one, whose buffer nonetheless
//! holds two components per pixel.
//!
//! Correspondingly none of them route through `to_f64_vec`/`image_from_f64`
//! (the scalar seam, which refuses complex images): they read the interleaved
//! `re, im, ...` buffer through [`Image::complex_components`] and build their
//! complex output through [`Image::from_vec_complex`].
//!
//! # Precision
//!
//! Every functor here computes in the *component* type, never in `f64`: ITK's
//! `ComplexToPhase<std::complex<float>, float>` evaluates `std::atan2` in
//! `float`, and `MagnitudeAndPhaseToComplex` expands `std::polar` in `float`.
//! Reproducing the component-type arithmetic keeps this port's output
//! bit-for-bit with ITK's, so this module is a deliberate exception to the
//! crate-wide "compute in `f64`" divergence (ledger §4.1).
//!
//! # Upstream notes
//!
//! - **`ComplexToModulus` used `std::sqrt(re² + im²)`, not `std::hypot` — fixed
//!   here (§2.58).** `itkComplexToModulusImageFilter.h:49` is
//!   `(TOutput)(std::sqrt(A.real() * A.real() + A.imag() * A.imag()))`. On a
//!   `ComplexFloat32` image the intermediate squares overflow to `inf` above
//!   `sqrt(FLT_MAX) ≈ 1.845e19`, lose precision through subnormals below
//!   `sqrt(FLT_MIN) ≈ 1.084e-19`, and flush to `0` below
//!   `sqrt(FLT_TRUE_MIN) ≈ 3.74e-23` — silently returning `inf`/`0` for a
//!   modulus that is a perfectly ordinary finite `f32`. This port computes the
//!   modulus with `hypot`, which suffers none of those three failures and
//!   agrees with `sqrt(re² + im²)` **to component-type rounding** wherever the
//!   latter is finite and nonzero — the two can still differ by ~1 ULP even in
//!   the normal range, because `hypot` avoids forming `re² + im²` (and its
//!   intermediate rounding / over- / underflow), so its result is the
//!   better-rounded of the two. The port's modulus being thereby *more*
//!   accurate than upstream's `std::sqrt` form is a deliberate
//!   correctness-over-mirroring choice, not an incidental drift. Pinned by
//!   `complex_to_modulus_is_finite_on_f32_where_the_naive_norm_overflows` and
//!   `complex_to_modulus_is_exact_on_f32_where_the_naive_norm_underflows`.
//!
//! - **`ComplexToPhaseImageFilter.yaml`'s `briefdescription` is wrong**: it
//!   reads "Computes pixel-wise the modulus of a complex image", copy-pasted
//!   from `ComplexToModulusImageFilter.yaml`. The filter computes
//!   `std::atan2(A.imag(), A.real())`
//!   (`itkComplexToPhaseImageFilter.h:50`), which is what this port does.
//!
//! - **`RealAndImaginaryToComplexImageFilter` is `itk::ComposeImageFilter`**
//!   under an alias (its yaml's `itk_name`), taking the complex specialization
//!   at `itkComposeImageFilter.hxx:132-138` — a `constexpr` branch that exists
//!   because `std::complex` "provides no `operator[]`"
//!   (`itkComposeImageFilter.h:104-105`), unlike every vector pixel type.
//!
//! - **`std::polar(rho, theta)` is undefined in C++ for a negative or `NaN`
//!   `rho`** (\[complex.value.ops\]/6), yet
//!   `itkMagnitudeAndPhaseToComplexImageFilter.h:72` calls it on an unchecked
//!   pixel value. libstdc++ evaluates `complex(rho * cos(theta),
//!   rho * sin(theta))` regardless. Diverge-for-C++-UB: this port *defines* the
//!   operation as exactly that expansion. Pinned by
//!   `magnitude_and_phase_to_complex_accepts_a_negative_magnitude`.
//!
//! # Input checks on the two-input filters
//!
//! Both check pixel type then size, through the crate's `require_same_shape`.
//! Upstream additionally rejects inputs that do not occupy the same physical
//! space (`ImageToImageFilter::VerifyInputInformation` compares origin,
//! spacing, and direction within a tolerance); this crate's two-input filters
//! uniformly do not, and that pre-existing divergence is not changed here.

use sitk_core::{Complex, Image, PixelId, Real};

use crate::error::FilterError;
use crate::{Result, require_same_shape};

/// The component-type arithmetic the four complex functors need, held on one
/// trait so the `f32` and `f64` instantiations cannot silently diverge.
///
/// Each method is the exact C++ expression its ITK functor evaluates, in the
/// component type; `Self` is `TOutput`.
trait ComplexMath: Real {
    /// The modulus `|re + im·i|`. **Fixed here (§2.58):** computed with
    /// `hypot(re, im)`, not upstream's `std::sqrt(re*re + im*im)`
    /// (`itkComplexToModulusImageFilter.h:49`), which overflows to `inf` and
    /// flushes to `0` on `ComplexFloat32` where `hypot` does not. See the module
    /// docs.
    fn modulus(re: Self, im: Self) -> Self;

    /// `std::atan2(A.imag(), A.real())` — `itkComplexToPhaseImageFilter.h:50`.
    fn phase(re: Self, im: Self) -> Self;

    /// `std::polar(rho, theta)` as libstdc++ expands it:
    /// `complex(rho * cos(theta), rho * sin(theta))` —
    /// `itkMagnitudeAndPhaseToComplexImageFilter.h:72`. See the module docs on
    /// the negative-`rho` undefined behavior this makes defined.
    fn polar(rho: Self, theta: Self) -> Complex<Self>;
}

macro_rules! impl_complex_math {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ComplexMath for $ty {
                #[inline]
                fn modulus(re: Self, im: Self) -> Self {
                    re.hypot(im)
                }

                #[inline]
                fn phase(re: Self, im: Self) -> Self {
                    im.atan2(re)
                }

                #[inline]
                fn polar(rho: Self, theta: Self) -> Complex<Self> {
                    Complex::new(rho * theta.cos(), rho * theta.sin())
                }
            }
        )+
    };
}

impl_complex_math!(f32, f64);

/// Which real-valued projection of a complex pixel a [`complex_unary`] call
/// takes. One enum rather than four dispatch sites, since all four functors
/// share the input guard, the output pixel type, and the de-interleave loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    /// `A.real()` — `itkComplexToRealImageFilter.h:49`.
    Real,
    /// `A.imag()` — `itkComplexToImaginaryImageFilter.h:50`.
    Imaginary,
    /// `itkComplexToModulusImageFilter.h:49`.
    Modulus,
    /// `itkComplexToPhaseImageFilter.h:50`.
    Phase,
}

fn project<T: ComplexMath>(img: &Image, part: Part) -> Result<Image> {
    let all = img.complex_components::<T>()?;
    let out: Vec<T> = all
        .chunks_exact(2)
        .map(|c| match part {
            Part::Real => c[0],
            Part::Imaginary => c[1],
            Part::Modulus => T::modulus(c[0], c[1]),
            Part::Phase => T::phase(c[0], c[1]),
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// The single dispatch seam for the four `ComplexTo*` filters.
///
/// `pixel_types: ComplexPixelIDTypeList` in each yaml, so the wrapper is
/// instantiated for `sitkComplexFloat32`/`sitkComplexFloat64` and no other
/// pixel type. `output_pixel_type: typename InputImageType::PixelType::value_type`
/// makes the output `Float32` / `Float64` respectively — never an integer type,
/// so no rounding rule applies.
fn complex_unary(img: &Image, part: Part) -> Result<Image> {
    match img.pixel_id() {
        PixelId::ComplexFloat32 => project::<f32>(img, part),
        PixelId::ComplexFloat64 => project::<f64>(img, part),
        other => Err(sitk_core::Error::RequiresComplexPixelType(other).into()),
    }
}

/// `ComplexToRealImageFilter` (`itkComplexToRealImageFilter.h:49`): the real
/// part of every pixel, as a `Float32`/`Float64` image.
///
/// Negative zero survives: `-0.0 + 3.0i` has real part `-0.0`, not `0.0`.
pub fn complex_to_real(img: &Image) -> Result<Image> {
    complex_unary(img, Part::Real)
}

/// `ComplexToImaginaryImageFilter` (`itkComplexToImaginaryImageFilter.h:50`):
/// the imaginary part of every pixel, as a `Float32`/`Float64` image.
pub fn complex_to_imaginary(img: &Image) -> Result<Image> {
    complex_unary(img, Part::Imaginary)
}

/// `ComplexToModulusImageFilter` (`itkComplexToModulusImageFilter.h:49`): the
/// modulus `|re + im·i|` per pixel, computed in the component type.
///
/// **Fixed here (§2.58):** computed with `hypot`, not upstream's
/// `sqrt(re*re + im*im)`, which overflows to `inf` and flushes to `0` on
/// `ComplexFloat32` for finite moduli — see the module docs.
pub fn complex_to_modulus(img: &Image) -> Result<Image> {
    complex_unary(img, Part::Modulus)
}

/// `ComplexToPhaseImageFilter` (`itkComplexToPhaseImageFilter.h:50`):
/// `atan2(im, re)` per pixel, in `(-π, π]`, computed in the component type.
///
/// (The yaml's `briefdescription` says "modulus"; it is a copy-paste error —
/// see the module docs.)
pub fn complex_to_phase(img: &Image) -> Result<Image> {
    complex_unary(img, Part::Phase)
}

/// Which complex-valued combination of two real pixels a [`complex_binary`]
/// call forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compose {
    /// `OutputPixelType{ (ValueType)in0, (ValueType)in1 }` —
    /// `itkComposeImageFilter.hxx:132-138`.
    RealAndImaginary,
    /// `std::complex<TOutput>(std::polar(A, B))` —
    /// `itkMagnitudeAndPhaseToComplexImageFilter.h:72`.
    MagnitudeAndPhase,
}

fn combine<T: ComplexMath>(a: &Image, b: &Image, how: Compose) -> Result<Image> {
    let x = a.scalar_slice::<T>()?;
    let y = b.scalar_slice::<T>()?;
    let data: Vec<Complex<T>> = x
        .iter()
        .zip(y.iter())
        .map(|(&p, &q)| match how {
            Compose::RealAndImaginary => Complex::new(p, q),
            Compose::MagnitudeAndPhase => T::polar(p, q),
        })
        .collect();
    let mut result = Image::from_vec_complex(a.size(), data)?;
    result.copy_geometry_from(a);
    Ok(result)
}

/// The single dispatch seam for the two `*ToComplex` filters.
///
/// `pixel_types: RealPixelIDTypeList` (sitkPixelIDTypeLists.h:98) — `Float32`
/// and `Float64` and nothing else, so the pixel-type check is a whitelist and a
/// complex or vector input falls out as [`FilterError::RequiresRealPixelType`].
///
/// The check order mirrors SimpleITK's generated two-input wrapper: it selects
/// the member function from `image1`'s pixel id (an unsupported type throws
/// first), then casts `image2` to that same ITK type (a mismatch throws next),
/// and only then does ITK compare regions.
fn complex_binary(a: &Image, b: &Image, how: Compose) -> Result<Image> {
    match a.pixel_id() {
        PixelId::Float32 => {
            require_same_shape(a, b)?;
            combine::<f32>(a, b, how)
        }
        PixelId::Float64 => {
            require_same_shape(a, b)?;
            combine::<f64>(a, b, how)
        }
        other => Err(FilterError::RequiresRealPixelType(other)),
    }
}

/// `RealAndImaginaryToComplexImageFilter` — `itk::ComposeImageFilter`'s complex
/// specialization (`itkComposeImageFilter.hxx:132-138`): pixel `i` of the
/// output is `re[i] + im[i]·j`.
///
/// Both inputs must be `Float32` or `Float64`
/// (`pixel_types: RealPixelIDTypeList`), of the same pixel type and size; the
/// output is that type's complex variant, with `re`'s geometry.
///
/// Errors with [`FilterError::RequiresRealPixelType`] on any other pixel type,
/// [`FilterError::TypeMismatch`] when the two disagree, and
/// [`FilterError::SizeMismatch`] where ITK throws "All Inputs must have the
/// same dimensions." (`itkComposeImageFilter.hxx:99-102`).
pub fn real_and_imaginary_to_complex(re: &Image, im: &Image) -> Result<Image> {
    complex_binary(re, im, Compose::RealAndImaginary)
}

/// `MagnitudeAndPhaseToComplexImageFilter`
/// (`itkMagnitudeAndPhaseToComplexImageFilter.h:72`): pixel `i` of the output is
/// `polar(magnitude[i], phase[i])`, i.e.
/// `magnitude[i]·cos(phase[i]) + magnitude[i]·sin(phase[i])·j`.
///
/// Same input rules and errors as [`real_and_imaginary_to_complex`]. A negative
/// or `NaN` magnitude is undefined in C++ and defined here — see the module
/// docs.
pub fn magnitude_and_phase_to_complex(magnitude: &Image, phase: &Image) -> Result<Image> {
    complex_binary(magnitude, phase, Compose::MagnitudeAndPhase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilterError;
    use sitk_core::Complex;

    fn cimg<T: Real>(size: &[usize], data: Vec<Complex<T>>) -> Image {
        Image::from_vec_complex(size, data).unwrap()
    }

    #[test]
    fn complex_to_real_takes_the_first_component() {
        let img = cimg(
            &[2, 1],
            vec![Complex::new(1.5f64, -2.5), Complex::new(7.0, 3.0)],
        );
        let out = complex_to_real(&img).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        assert_eq!(out.size(), &[2, 1]);
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[1.5, 7.0]);
    }

    #[test]
    fn complex_to_imaginary_takes_the_second_component() {
        let img = cimg(
            &[2, 1],
            vec![Complex::new(1.5f32, -2.5), Complex::new(7.0, 3.0)],
        );
        let out = complex_to_imaginary(&img).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[-2.5, 3.0]);
    }

    #[test]
    fn complex_to_real_and_imaginary_preserve_negative_zero() {
        let img = cimg(&[1], vec![Complex::new(-0.0f64, -0.0)]);
        let re = complex_to_real(&img).unwrap();
        let im = complex_to_imaginary(&img).unwrap();
        assert!(re.scalar_slice::<f64>().unwrap()[0].is_sign_negative());
        assert!(im.scalar_slice::<f64>().unwrap()[0].is_sign_negative());
    }

    #[test]
    fn complex_to_modulus_is_the_pythagorean_norm() {
        let img = cimg(
            &[3, 1],
            vec![
                Complex::new(3.0f64, 4.0),
                Complex::new(-3.0, -4.0),
                Complex::new(0.0, 0.0),
            ],
        );
        let out = complex_to_modulus(&img).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[5.0, 5.0, 0.0]);
    }

    #[test]
    fn complex_to_modulus_is_finite_on_f32_where_the_naive_norm_overflows() {
        // 2e19² = 4e38 > f32::MAX ≈ 3.403e38, so the upstream `re * re` is
        // already `inf` and its `sqrt` stays `inf`. `hypot` returns the true
        // modulus 2e19·√2 ≈ 2.828e19, which fits comfortably in f32. §2.58 fix.
        let img = cimg(&[1], vec![Complex::new(2e19f32, 2e19)]);
        let out = complex_to_modulus(&img).unwrap();
        let got = out.scalar_slice::<f32>().unwrap()[0];
        assert!(got.is_finite());
        assert_eq!(got, 2e19f32.hypot(2e19f32));
        // Sanity: the upstream `sqrt(re*re + im*im)` would have been `inf` here.
        assert_eq!(
            (2e19f32 * 2e19f32 + 2e19f32 * 2e19f32).sqrt(),
            f32::INFINITY
        );
    }

    #[test]
    fn complex_to_modulus_is_exact_on_f32_where_the_naive_norm_underflows() {
        // 1e-30² = 1e-60, far below f32's smallest subnormal (≈1.4e-45), so the
        // upstream `re * re` flushes to +0 and its modulus is 0. `hypot` returns
        // 1e-30 exactly (a normal f32). §2.58 fix.
        let img = cimg(&[1], vec![Complex::new(1e-30f32, 0.0)]);
        let out = complex_to_modulus(&img).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap()[0], 1e-30f32);
        // Sanity: the upstream `sqrt(re*re + im*im)` would have been 0 here.
        assert_eq!((1e-30f32 * 1e-30f32).sqrt(), 0.0);
    }

    #[test]
    fn complex_to_modulus_in_f64_is_the_full_precision_modulus() {
        // The same magnitudes as the f32 test, in f64: `hypot` gives the true
        // modulus 2e19·√2 in either component type (with the fix the f32 result
        // now agrees to f32 precision instead of overflowing to `inf`).
        let img = cimg(&[1], vec![Complex::new(2e19f64, 2e19)]);
        let out = complex_to_modulus(&img).unwrap();
        let got = out.scalar_slice::<f64>().unwrap()[0];
        assert!((got - 2e19 * std::f64::consts::SQRT_2).abs() < 1e5);
    }

    #[test]
    fn complex_to_phase_covers_the_four_quadrants() {
        use std::f64::consts::PI;
        let img = cimg(
            &[4, 1],
            vec![
                Complex::new(1.0f64, 1.0), //  Q1
                Complex::new(-1.0, 1.0),   //  Q2
                Complex::new(-1.0, -1.0),  //  Q3
                Complex::new(1.0, -1.0),   //  Q4
            ],
        );
        let out = complex_to_phase(&img).unwrap();
        assert_eq!(
            out.scalar_slice::<f64>().unwrap(),
            &[PI / 4.0, 3.0 * PI / 4.0, -3.0 * PI / 4.0, -PI / 4.0]
        );
    }

    #[test]
    fn complex_to_phase_follows_ieee_atan2_on_the_signed_zeros() {
        // atan2(±0, +0) = ±0; atan2(±0, -0) = ±π. Only reachable because the
        // buffer stores -0.0 and 0.0 as distinct values.
        use std::f64::consts::PI;
        let img = cimg(
            &[5, 1],
            vec![
                Complex::new(0.0f64, 0.0),
                Complex::new(-0.0, 0.0),
                Complex::new(-0.0, -0.0),
                Complex::new(0.0, -0.0),
                Complex::new(-1.0, -0.0),
            ],
        );
        let out = complex_to_phase(&img).unwrap();
        let got = out.scalar_slice::<f64>().unwrap();
        assert_eq!(got[0], 0.0);
        assert!(got[0].is_sign_positive());
        assert_eq!(got[1], PI);
        assert_eq!(got[2], -PI);
        assert_eq!(got[3], 0.0);
        assert!(got[3].is_sign_negative());
        assert_eq!(got[4], -PI);
    }

    #[test]
    fn complex_unary_filters_copy_geometry() {
        let mut img = cimg(&[2, 2], vec![Complex::new(1.0f32, 1.0); 4]);
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        for out in [
            complex_to_real(&img).unwrap(),
            complex_to_imaginary(&img).unwrap(),
            complex_to_modulus(&img).unwrap(),
            complex_to_phase(&img).unwrap(),
        ] {
            assert_eq!(out.spacing(), img.spacing());
            assert_eq!(out.origin(), img.origin());
            assert_eq!(out.direction(), img.direction());
        }
    }

    #[test]
    fn real_and_imaginary_to_complex_pairs_the_two_inputs() {
        let re = Image::from_vec(&[2, 1], vec![1.0f64, 2.0]).unwrap();
        let im = Image::from_vec(&[2, 1], vec![3.0f64, 4.0]).unwrap();
        let out = real_and_imaginary_to_complex(&re, &im).unwrap();
        assert_eq!(out.pixel_id(), PixelId::ComplexFloat64);
        assert_eq!(out.number_of_components_per_pixel(), 1);
        assert_eq!(
            out.complex_components::<f64>().unwrap(),
            &[1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn real_and_imaginary_to_complex_round_trips_through_the_projections() {
        let re = Image::from_vec(&[3, 1], vec![1.5f32, -0.0, 7.25]).unwrap();
        let im = Image::from_vec(&[3, 1], vec![-2.5f32, 3.0, 0.0]).unwrap();
        let c = real_and_imaginary_to_complex(&re, &im).unwrap();
        assert_eq!(c.pixel_id(), PixelId::ComplexFloat32);
        assert_eq!(complex_to_real(&c).unwrap(), re);
        assert_eq!(complex_to_imaginary(&c).unwrap(), im);
    }

    #[test]
    fn real_and_imaginary_to_complex_takes_the_first_inputs_geometry() {
        let mut re = Image::from_vec(&[2, 1], vec![1.0f64, 2.0]).unwrap();
        re.set_spacing(&[0.5, 2.0]).unwrap();
        re.set_origin(&[-1.0, 3.0]).unwrap();
        let im = Image::from_vec(&[2, 1], vec![3.0f64, 4.0]).unwrap();
        let out = real_and_imaginary_to_complex(&re, &im).unwrap();
        assert_eq!(out.spacing(), re.spacing());
        assert_eq!(out.origin(), re.origin());
    }

    #[test]
    fn magnitude_and_phase_to_complex_is_polars_expansion() {
        use std::f64::consts::PI;
        let mag = Image::from_vec(&[3, 1], vec![2.0f64, 2.0, 0.0]).unwrap();
        let phase = Image::from_vec(&[3, 1], vec![0.0f64, PI / 2.0, 1.0]).unwrap();
        let out = magnitude_and_phase_to_complex(&mag, &phase).unwrap();
        assert_eq!(out.pixel_id(), PixelId::ComplexFloat64);

        // theta = 0: exactly (rho, 0).
        assert_eq!(
            out.get_complex::<f64>(&[0, 0]).unwrap(),
            Complex::new(2.0, 0.0)
        );
        // theta = pi/2: `rho * cos(theta)` is NOT special-cased to zero —
        // cos(PI/2) is 6.123233995736766e-17 in f64.
        let q = out.get_complex::<f64>(&[1, 0]).unwrap();
        assert_eq!(q.re, 2.0 * (PI / 2.0).cos());
        assert_eq!(q.re, 1.2246467991473532e-16);
        assert_eq!(q.im, 2.0);
        // rho = 0 zeroes both parts whatever the phase.
        assert_eq!(
            out.get_complex::<f64>(&[2, 0]).unwrap(),
            Complex::new(0.0, 0.0)
        );
    }

    #[test]
    fn magnitude_and_phase_to_complex_accepts_a_negative_magnitude() {
        // std::polar's precondition is rho >= 0 ([complex.value.ops]/6); ITK
        // does not check, and libstdc++ just multiplies. Defined here as that
        // same product, so a negative magnitude reflects through the origin.
        let mag = Image::from_vec(&[1], vec![-1.0f64]).unwrap();
        let phase = Image::from_vec(&[1], vec![0.0f64]).unwrap();
        let out = magnitude_and_phase_to_complex(&mag, &phase).unwrap();
        assert_eq!(
            out.get_complex::<f64>(&[0]).unwrap(),
            Complex::new(-1.0, 0.0)
        );
    }

    #[test]
    fn magnitude_and_phase_to_complex_computes_in_f32_for_f32_inputs() {
        // polar(1, pi/2) in f32: cos(pi/2f32) = -4.371139e-8, not the f64 value.
        let theta = std::f32::consts::PI / 2.0;
        let mag = Image::from_vec(&[1], vec![1.0f32]).unwrap();
        let phase = Image::from_vec(&[1], vec![theta]).unwrap();
        let out = magnitude_and_phase_to_complex(&mag, &phase).unwrap();
        assert_eq!(out.pixel_id(), PixelId::ComplexFloat32);
        let v = out.get_complex::<f32>(&[0]).unwrap();
        assert_eq!(v.re, theta.cos());
        assert_eq!(v.re, -4.371139e-8);
        assert_eq!(v.im, 1.0);
    }

    #[test]
    fn magnitude_and_phase_then_modulus_and_phase_round_trips() {
        use std::f64::consts::PI;
        let mag = Image::from_vec(&[2, 1], vec![2.0f64, 5.0]).unwrap();
        let phase = Image::from_vec(&[2, 1], vec![PI / 6.0, -3.0 * PI / 4.0]).unwrap();
        let c = magnitude_and_phase_to_complex(&mag, &phase).unwrap();

        let m = complex_to_modulus(&c).unwrap();
        let p = complex_to_phase(&c).unwrap();
        for (got, want) in m.scalar_slice::<f64>().unwrap().iter().zip([2.0, 5.0]) {
            assert!((got - want).abs() < 1e-15, "{got} vs {want}");
        }
        for (got, want) in p
            .scalar_slice::<f64>()
            .unwrap()
            .iter()
            .zip([PI / 6.0, -3.0 * PI / 4.0])
        {
            assert!((got - want).abs() < 1e-15, "{got} vs {want}");
        }
    }

    #[test]
    fn complex_binary_filters_reject_non_real_first_inputs() {
        let f32_img = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        for bad in [
            Image::from_vec(&[2, 1], vec![1u8, 2]).unwrap(),
            Image::new(&[2, 1], PixelId::ComplexFloat32),
            Image::from_vec_vector(&[2, 1], 2, vec![1.0f32; 4]).unwrap(),
        ] {
            let id = bad.pixel_id();
            assert_eq!(
                real_and_imaginary_to_complex(&bad, &f32_img),
                Err(FilterError::RequiresRealPixelType(id))
            );
            assert_eq!(
                magnitude_and_phase_to_complex(&bad, &f32_img),
                Err(FilterError::RequiresRealPixelType(id))
            );
        }
    }

    #[test]
    fn complex_binary_filters_reject_mismatched_types_and_sizes() {
        let a = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let wrong_type = Image::from_vec(&[2, 1], vec![1.0f64, 2.0]).unwrap();
        assert_eq!(
            real_and_imaginary_to_complex(&a, &wrong_type),
            Err(FilterError::TypeMismatch {
                a: PixelId::Float32,
                b: PixelId::Float64,
            })
        );
        let wrong_size = Image::from_vec(&[3, 1], vec![1.0f32; 3]).unwrap();
        assert_eq!(
            magnitude_and_phase_to_complex(&a, &wrong_size),
            Err(FilterError::SizeMismatch {
                a: vec![2, 1],
                b: vec![3, 1],
            })
        );
        // A complex *second* input is a type mismatch, not a real-type error:
        // upstream reaches it through CastImageToITK, not the wrapper factory.
        let complex_b = Image::new(&[2, 1], PixelId::ComplexFloat32);
        assert_eq!(
            real_and_imaginary_to_complex(&a, &complex_b),
            Err(FilterError::TypeMismatch {
                a: PixelId::Float32,
                b: PixelId::ComplexFloat32,
            })
        );
    }

    #[test]
    fn complex_unary_filters_reject_non_complex_inputs() {
        let scalar = Image::from_vec(&[2, 1], vec![1.0f32, 2.0]).unwrap();
        let vector = Image::from_vec_vector(&[2, 1], 2, vec![1.0f32; 4]).unwrap();
        for img in [&scalar, &vector] {
            for f in [
                complex_to_real,
                complex_to_imaginary,
                complex_to_modulus,
                complex_to_phase,
            ] {
                assert!(matches!(
                    f(img),
                    Err(FilterError::Core(
                        sitk_core::Error::RequiresComplexPixelType(_)
                    ))
                ));
            }
        }
    }
}
