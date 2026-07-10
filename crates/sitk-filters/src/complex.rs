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
//! `ComplexToModulus<std::complex<float>, float>` evaluates
//! `A.real() * A.real() + A.imag() * A.imag()` in `float` and calls the `float`
//! overload of `std::sqrt`. Reproducing that is what makes the overflow and
//! underflow cases below observable, so this module is a deliberate exception
//! to the crate-wide "compute in `f64`" divergence (ledger §4.1).
//!
//! # Upstream notes
//!
//! - **`ComplexToModulus` does not use `std::hypot`.**
//!   `itkComplexToModulusImageFilter.h:49` is
//!   `(TOutput)(std::sqrt(A.real() * A.real() + A.imag() * A.imag()))`. On a
//!   `ComplexFloat32` image the squares overflow to `inf` above
//!   `sqrt(FLT_MAX) ≈ 1.845e19`, lose precision through subnormals below
//!   `sqrt(FLT_MIN) ≈ 1.084e-19`, and flush to `0` below
//!   `sqrt(FLT_TRUE_MIN) ≈ 3.74e-23`. `hypot` has none of those three failures.
//!   Reproduced verbatim; pinned by
//!   `complex_to_modulus_overflows_on_f32_where_hypot_would_not` and
//!   `complex_to_modulus_underflows_on_f32_where_hypot_would_not`.
//!
//! - **`ComplexToPhaseImageFilter.yaml`'s `briefdescription` is wrong**: it
//!   reads "Computes pixel-wise the modulus of a complex image", copy-pasted
//!   from `ComplexToModulusImageFilter.yaml`. The filter computes
//!   `std::atan2(A.imag(), A.real())`
//!   (`itkComplexToPhaseImageFilter.h:50`), which is what this port does.

use sitk_core::{Image, PixelId, Real};

use crate::Result;

/// The component-type arithmetic the four complex functors need, held on one
/// trait so the `f32` and `f64` instantiations cannot silently diverge.
///
/// Each method is the exact C++ expression its ITK functor evaluates, in the
/// component type; `Self` is `TOutput`.
trait ComplexMath: Real {
    /// `std::sqrt(A.real() * A.real() + A.imag() * A.imag())` —
    /// `itkComplexToModulusImageFilter.h:49`. Not `hypot`; see the module docs.
    fn modulus(re: Self, im: Self) -> Self;

    /// `std::atan2(A.imag(), A.real())` — `itkComplexToPhaseImageFilter.h:50`.
    fn phase(re: Self, im: Self) -> Self;
}

macro_rules! impl_complex_math {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ComplexMath for $ty {
                #[inline]
                fn modulus(re: Self, im: Self) -> Self {
                    (re * re + im * im).sqrt()
                }

                #[inline]
                fn phase(re: Self, im: Self) -> Self {
                    im.atan2(re)
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

/// `ComplexToModulusImageFilter` (`itkComplexToModulusImageFilter.h:49`):
/// `sqrt(re² + im²)` per pixel, computed in the component type.
///
/// Reproduces upstream's overflow and underflow on `ComplexFloat32` — see the
/// module docs; this is `sqrt(re*re + im*im)`, not `hypot(re, im)`.
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
    fn complex_to_modulus_overflows_on_f32_where_hypot_would_not() {
        // 2e19² = 4e38 > f32::MAX ≈ 3.403e38, so `re * re` is already `inf`.
        // `hypot(2e19, 2e19)` would give ≈ 2.828e19, which fits comfortably.
        let img = cimg(&[1], vec![Complex::new(2e19f32, 2e19)]);
        let out = complex_to_modulus(&img).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap()[0], f32::INFINITY);
        assert!(2e19f32.hypot(2e19f32).is_finite());
    }

    #[test]
    fn complex_to_modulus_underflows_on_f32_where_hypot_would_not() {
        // 1e-30² = 1e-60, far below f32's smallest subnormal (≈1.4e-45), so
        // `re * re` flushes to +0 and the modulus is 0 instead of 1e-30.
        let img = cimg(&[1], vec![Complex::new(1e-30f32, 0.0)]);
        let out = complex_to_modulus(&img).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap()[0], 0.0);
        assert_eq!(1e-30f32.hypot(0.0), 1e-30);
    }

    #[test]
    fn complex_to_modulus_in_f64_does_not_overflow_at_the_f32_bound() {
        // The same magnitudes as the f32 overflow test, in f64: the functor
        // computes in the component type, so widening the pixel type changes
        // the answer.
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
