//! `NoiseImageFilter`
//! (`Modules/Filtering/ImageFilterBase/include/itkNoiseImageFilter.h(.hxx)`):
//! the local (sample) standard deviation over a per-axis `radius` window,
//! under [`ZeroFluxNeumannBoundaryCondition`].
//!
//! `DynamicThreadedGenerateData` accumulates `sum` and `sumOfSquares` over
//! the neighborhood in a single pass, *in that order* (`sum += value;
//! sumOfSquares += value * value;` inside one loop over `bit.GetPixel(i)`),
//! then computes
//!
//! ```text
//! var = (sumOfSquares - sum * sum / num) / (num - 1.0)
//! ```
//!
//! — the shifted-data (naive two-pass-in-one) variance formula, `num - 1.0`
//! divisor (sample/Bessel-corrected variance, *not* the population `num`
//! divisor), narrowed via `std::sqrt`. This port reproduces the same
//! accumulation order and division, including its cancellation behavior: on
//! a perfectly constant window, `sumOfSquares` and `sum * sum / num` are
//! mathematically (and, since every term is representable exactly for the
//! small integer test fixtures this crate exercises, also in floating point)
//! equal, so `var` is exactly `0.0`, not a small nonzero residual.
//!
//! **Output pixel type.** `NoiseImageFilter.yaml` sets neither
//! `output_pixel_type` nor `output_image_type`, and the SimpleITK code
//! generator's `ExecuteInternalTypedefs.cxx.jinja` falls back to
//! `OutputImageType = InputImageType` whenever both are absent — verified
//! directly against that template (`ExpandTemplateGenerator/templates/`),
//! not assumed. So the output keeps `img`'s own pixel type: an integer input
//! narrows its computed standard deviation into that integer type exactly as
//! `static_cast<OutputPixelType>(std::sqrt(var))` would, truncating any
//! fractional noise estimate.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{Image, NeighborhoodIterator, ZeroFluxNeumannBoundaryCondition};

/// `NoiseImageFilter`: the local sample standard deviation of `img` over a
/// per-axis `radius` neighborhood (see the module docs for the exact
/// accumulation order and divisor). Output keeps `img`'s own pixel type.
///
/// Errors if `radius.len() != img.dimension()`.
pub fn noise(img: &Image, radius: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }

    let mut scratch = Image::from_vec(img.size(), img.to_f64_vec()?)?;
    scratch.copy_geometry_from(img);

    let iter =
        NeighborhoodIterator::<f64, _>::new(&scratch, radius, ZeroFluxNeumannBoundaryCondition)?;
    let num = iter.len() as f64;

    let out: Vec<f64> = iter
        .map(|(_, nb)| {
            let mut sum = 0.0f64;
            let mut sum_of_squares = 0.0f64;
            for &value in nb.values() {
                sum += value;
                sum_of_squares += value * value;
            }
            let var = (sum_of_squares - (sum * sum / num)) / (num - 1.0);
            var.sqrt()
        })
        .collect();

    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    #[test]
    fn constant_image_has_zero_noise() {
        let img = Image::from_vec(&[5, 5], vec![7.0f64; 25]).unwrap();
        let out = noise(&img, &[1, 1]).unwrap();
        assert!(out.scalar_slice::<f64>().unwrap().iter().all(|&v| v == 0.0));
    }

    /// Hand-computed sample standard deviation of a known 3-pixel window
    /// (`radius = 1` along a single axis, interior pixel): values `[2, 4,
    /// 6]`, `sum = 12`, `sum_of_squares = 4+16+36 = 56`,
    /// `var = (56 - 144/3) / 2 = (56 - 48) / 2 = 4`, `sqrt(4) = 2`.
    #[test]
    fn matches_hand_computed_std_for_a_known_window() {
        let img = Image::from_vec(&[5], vec![0.0f64, 2.0, 4.0, 6.0, 0.0]).unwrap();
        let out = noise(&img, &[1]).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap()[2], 2.0);
    }

    /// Pins the `num - 1.0` (sample, Bessel-corrected) divisor rather than
    /// `num`: with the population divisor the same window would give
    /// `var = 8/3`, `sqrt(8/3) ≈ 1.633`, not `2.0`.
    #[test]
    fn divisor_is_n_minus_one_not_n() {
        let img = Image::from_vec(&[5], vec![0.0f64, 2.0, 4.0, 6.0, 0.0]).unwrap();
        let out = noise(&img, &[1]).unwrap();
        let population_divisor_value: f64 = (8.0f64 / 3.0).sqrt();
        assert_ne!(
            out.scalar_slice::<f64>().unwrap()[2],
            population_divisor_value
        );
        assert_eq!(out.scalar_slice::<f64>().unwrap()[2], 2.0);
    }

    #[test]
    fn output_pixel_type_matches_input() {
        let img = Image::from_vec(&[3, 3], vec![1u8; 9]).unwrap();
        let out = noise(&img, &[1, 1]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    #[test]
    fn rejects_wrong_length_radius() {
        let img = Image::from_vec(&[4, 4], vec![0.0f64; 16]).unwrap();
        assert_eq!(
            noise(&img, &[1]).unwrap_err(),
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }
}
