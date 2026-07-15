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

use crate::core::{
    Image, NeighborhoodIterator, Scalar, ZeroFluxNeumannBoundaryCondition, dispatch_scalar,
};
use crate::filters::error::{FilterError, Result};
use crate::filters::image_from_f64;

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

    let out = dispatch_scalar!(img.pixel_id(), noise_pass, img, radius)?;
    image_from_f64(img.pixel_id(), img.size(), img, &out)
}

/// The stencil, as a parallel map over output voxels.
///
/// Reads `T` and widens per access rather than materializing an `f64` copy of
/// the whole volume first. That is the same `Scalar::as_f64` the copy's
/// `to_f64_vec` applied, so every value the accumulation sees is the one the copy
/// held — and the copy's scalar-pixel rejection survives its deletion, because
/// `NeighborhoodIterator::new` takes a `scalar_view::<T>()` and returns the same
/// [`crate::core::Error::RequiresScalarPixelType`].
///
/// `sum` and `sum_of_squares` accumulate over **one voxel's own window**, in
/// window order (`WindowView::rows` concatenated is exactly the order
/// `Neighborhood::values` held, so the two accumulators see the identical
/// sequence ITK's single `bit.GetPixel(i)` loop fed them), and the interleaving
/// `sum += value; sum_of_squares += value * value;` is preserved. No accumulator
/// crosses voxels, so no thread count can reach the arithmetic: the result is
/// bit-identical to the serial walk by construction.
fn noise_pass<T: Scalar>(img: &Image, radius: &[usize]) -> Result<Vec<f64>> {
    let iter = NeighborhoodIterator::<T, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)?;
    let num = iter.len() as f64;

    Ok(iter.par_map_window(|_, w| {
        let mut sum = 0.0f64;
        let mut sum_of_squares = 0.0f64;
        for run in w.rows() {
            for &v in run {
                let value = v.as_f64();
                sum += value;
                sum_of_squares += value * value;
            }
        }
        let var = (sum_of_squares - (sum * sum / num)) / (num - 1.0);
        var.sqrt()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

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

/// Thread-count parity for [`noise`], which was a serial `iter().map().collect()`
/// over a `Neighborhood<f64>` copied out per voxel, fed by a full `f64` copy of
/// the input volume. Both are gone; the pass is now a
/// [`NeighborhoodIterator::par_map_window`] over a borrowed
/// [`crate::core::WindowView`] of the input's own pixels.
///
/// **No `-0.0` exposure.** That trap is specific to converting a first
/// accumulate into a store — `0.0 + x` and `x` differ only at `x == -0.0`. This
/// filter converts nothing: `sum` and `sum_of_squares` still start at `0.0` and
/// still accumulate, and `var.sqrt()` is unchanged. There is no store to
/// substitute.
#[cfg(test)]
mod thread_parity {
    use super::*;
    use crate::core::{PixelId, parallel};

    /// The `f64` copy of the whole volume and the serial neighborhood walk that
    /// [`noise`] used to run. The reference the parallel pass is pinned against.
    fn noise_serial(img: &Image, radius: &[usize]) -> Vec<f64> {
        let mut scratch = Image::from_vec(img.size(), img.to_f64_vec().unwrap()).unwrap();
        scratch.copy_geometry_from(img);

        let iter =
            NeighborhoodIterator::<f64, _>::new(&scratch, radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        let num = iter.len() as f64;

        iter.map(|(_, nb)| {
            let mut sum = 0.0f64;
            let mut sum_of_squares = 0.0f64;
            for &value in nb.values() {
                sum += value;
                sum_of_squares += value * value;
            }
            let var = (sum_of_squares - (sum * sum / num)) / (num - 1.0);
            var.sqrt()
        })
        .collect()
    }

    /// A 32³ volume — 32 768 voxels, over `parallel`'s 16 384 serial threshold, so
    /// the window pass really runs on rayon instead of falling back to the serial
    /// fast path and pinning nothing.
    ///
    /// Both pixel types are pinned, for different reasons. `Float64` carries full
    /// 53-bit mantissas, so the window's `sum` genuinely rounds and the order of
    /// its terms is observable — that is what gives the pin teeth against a
    /// re-association. `Float32` exercises the widening-per-access path that
    /// replaced the deleted volume copy.
    fn volume(pixel: PixelId) -> Image {
        let n = 32usize;
        let value = |i: usize, j: usize, k: usize| {
            let (x, y, z) = (i as f64, j as f64, k as f64);
            (0.7 * x).sin() * 40.0
                + (0.3 * y).cos() * 25.0
                + (x * y * 0.01 + z * 0.9).sin() * 13.0
                + ((i * 37 + j * 11 + k * 7) % 29) as f64
        };
        let mut data = vec![0.0f64; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    data[(k * n + j) * n + i] = value(i, j, k);
                }
            }
        }
        match pixel {
            PixelId::Float64 => Image::from_vec(&[n, n, n], data).unwrap(),
            PixelId::Float32 => {
                let d: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Image::from_vec(&[n, n, n], d).unwrap()
            }
            other => panic!("volume() does not build {other:?}"),
        }
    }

    const PIXELS: [PixelId; 2] = [PixelId::Float64, PixelId::Float32];

    /// Narrow a reference's raw `f64` values through the same exit [`noise`] takes
    /// — it keeps the input's pixel type, so on an `f32` input its output is
    /// `f32`-rounded, and comparing against an un-rounded reference would fail on
    /// the rounding rather than on anything the parallelization did.
    fn narrowed_like(img: &Image, values: &[f64]) -> Vec<f64> {
        image_from_f64(img.pixel_id(), img.size(), img, values)
            .unwrap()
            .to_f64_vec()
            .unwrap()
    }

    /// The pin would assert nothing if this filter's window accumulation were
    /// order-insensitive: a sum that cannot round is unchanged by any
    /// re-association, so "the bits match" would hold no matter what the code did.
    ///
    /// On the `Float64` volume the order must be observable — reversing a voxel's
    /// window must move `sum`'s bits somewhere. That is the teeth.
    ///
    /// On `Float32` it need not be, and the pin does not claim it is: the values
    /// carry 24-bit mantissas, so short window sums can be exact. The `Float32`
    /// volume is pinned for the widening path, not for the fold order. This test
    /// records which volume carries which claim, so neither pin can later be read
    /// as proving the other's.
    #[test]
    fn the_within_window_sum_order_is_observable_on_float64() {
        let img = volume(PixelId::Float64);
        let radius = [1usize, 1, 1];
        let mut scratch = Image::from_vec(img.size(), img.to_f64_vec().unwrap()).unwrap();
        scratch.copy_geometry_from(&img);
        let iter = NeighborhoodIterator::<f64, _>::new(
            &scratch,
            &radius,
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap();

        let mut moved = 0usize;
        for (_, nb) in iter {
            let forward = nb.values().iter().fold(0.0f64, |a, &v| a + v);
            let backward = nb.values().iter().rev().fold(0.0f64, |a, &v| a + v);
            if forward.to_bits() != backward.to_bits() {
                moved += 1;
            }
        }
        assert!(
            moved > 0,
            "no voxel's window sum changed bits when its values were reversed, so \
             this volume cannot observe a re-association and the pin below would \
             pass even if the window accumulation were reordered"
        );
    }

    /// The other axis of vacuity: the filter must actually depend on the input. A
    /// flat volume gives every voxel zero noise and makes every comparison
    /// trivially true.
    #[test]
    fn the_reference_output_is_not_degenerate() {
        for pixel in PIXELS {
            let img = volume(pixel);
            let values = noise_serial(&img, &[1, 1, 1]);
            let nonzero = values.iter().filter(|v| **v != 0.0).count();
            assert!(
                nonzero > values.len() / 2,
                "{pixel:?}: only {nonzero}/{} voxels have non-zero noise — the test \
                 volume is too flat to pin anything",
                values.len()
            );
        }
    }

    /// `noise` is bit-identical to the deleted serial loop at every thread count,
    /// on both pixel types, at an isotropic and an anisotropic radius (the latter
    /// gives the window a different shape per axis, so a wrong stride shows up).
    #[test]
    fn noise_is_bit_identical_at_every_thread_count() {
        for pixel in PIXELS {
            let img = volume(pixel);
            assert!(
                img.number_of_pixels() > 1 << 14,
                "volume must exceed the serial threshold, or the parallel path never runs"
            );

            for radius in [[1usize, 1, 1], [2, 1, 3]] {
                let expected = narrowed_like(&img, &noise_serial(&img, &radius));
                for threads in [1usize, 4, 48, 96] {
                    let got = parallel::with_threads(threads, || noise(&img, &radius).unwrap());
                    let got = got.to_f64_vec().unwrap();
                    assert_eq!(got.len(), expected.len());
                    for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "noise({pixel:?}, radius={radius:?}) moved at voxel {i} with \
                             {threads} threads: {a:?} vs serial {b:?}"
                        );
                    }
                }
            }
        }
    }
}
