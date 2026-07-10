//! Normalized cross correlation of two images computed with FFTs, with
//! optional masks — D. Padfield's masked NCC.
//!
//! Ported from `itk::MaskedFFTNormalizedCorrelationImageFilter`
//! (itkMaskedFFTNormalizedCorrelationImageFilter.hxx) and its trivial subclass
//! `itk::FFTNormalizedCorrelationImageFilter`
//! (itkFFTNormalizedCorrelationImageFilter.hxx), whose `GenerateData` is
//! nothing but `Superclass::GenerateData()`: the unmasked filter *is* the
//! masked one with both masks left unset, which `PreProcessMask` then fills
//! with images of ones. Parameter names and defaults follow SimpleITK's
//! `FFTNormalizedCorrelationImageFilter.yaml` /
//! `MaskedFFTNormalizedCorrelationImageFilter.yaml`
//! (`RequiredNumberOfOverlappingPixels = 0`,
//! `RequiredFractionOfOverlappingPixels = 0.0`).
//!
//! # The algorithm
//!
//! With `f` the fixed image, `g` the moving image, `Mf`/`Mg` their 0/1 masks
//! and `⋆` cross-correlation, ITK assembles (hxx:180-238)
//!
//! ```text
//! N   = Mf ⋆ Mg                       (overlapping pixel count per shift)
//! num = f ⋆ g       - (f ⋆ Mg)(Mf ⋆ g) / N
//! Df  = f² ⋆ Mg     - (f ⋆ Mg)²       / N
//! Dg  = Mf ⋆ g²     - (Mf ⋆ g)²       / N
//! NCC = num / sqrt(max(Df,0) · max(Dg,0))
//! ```
//!
//! Every `⋆` is one multiplication of Fourier transforms: with the moving
//! image and mask reflected through every axis (`RotateImage`, hxx:294-313)
//! correlation becomes convolution, so the whole thing costs 6 forward and 6
//! inverse transforms of `fixedImage`, `fixedMask`, `fixedImage²` and the
//! three rotated moving counterparts.
//!
//! `N` is rounded to the nearest integer and clamped to be non-negative;
//! `Df`, `Dg` are clamped to be non-negative before the square root — all
//! three are exact integers or non-negative reals in exact arithmetic, and
//! the clamps only undo FFT round-off (hxx:183-184, 214, 230).
//!
//! # Output geometry
//!
//! The output is `size(fixed) + size(moving) - 1` pixels per axis
//! (`GenerateOutputInformation`, hxx:682-691): output index `k` holds the NCC
//! for the moving image displaced by `s = k - (size(moving) - 1)`, so `s`
//! sweeps `-(size(moving) - 1) ..= size(fixed) - 1` and a self-correlation
//! peaks at the exact centre `k = size - 1`.
//!
//! Its origin is `fixed`'s continuous index `-(size(moving) - 1) / 2` mapped
//! to physical space, which centres the moving image on each correlation
//! score (hxx:699-706). Spacing and direction come from `fixed`.
//!
//! # Zeroing unreliable scores
//!
//! `PostProcessCorrelation` (hxx:79-101) zeroes a score whose denominator is
//! below `m_PrecisionTolerance` (below which the quotient is round-off), or
//! that comes from no overlapping pixels at all, or from fewer than
//! `requiredNumberOfOverlappingPixels`; the rest is clamped to `[-1, 1]`.
//! The tolerance is *derived*, not a parameter (`CalculatePrecisionTolerance`,
//! hxx:570-598):
//!
//! ```text
//! tolerance = 1000 · 2^-p · 2^floor(log2(max denominator))
//! ```
//!
//! with `p = 23` when the output pixel type is `float` and `p = 52` when it is
//! `double` — i.e. a thousand ulps at the denominator's own magnitude. See
//! [`fft_normalized_correlation`] for how the required overlap is derived from
//! the two parameters.
//!
//! # Transform lengths
//!
//! This filter does **not** use `itkFFTPadImageFilter`, and its valid lengths
//! are not the FFT backend's `SizeGreatestPrimeFactor`. It carries its own
//! search: [`find_closest_valid_dimension`] (hxx:549-564) walks up from
//! `size(fixed) + size(moving) - 1` until `FactorizeNumber` (hxx:528-545)
//! divides the length away by 2s, 3s and 5s alone — "These are the only
//! factors that are valid for the FFT calculation", says a comment written for
//! a backend that is no longer the default. `FFTConvolutionImageFilter` on the
//! same PocketFFT backend accepts factors up to 11 (ledger §2.110). Ported as
//! written, so this crate's correlation transform lengths match upstream's
//! exactly.
//!
//! # Deliberate divergences from ITK
//!
//! - **Precision.** ITK computes the transforms and every pixel-wise stage in
//!   `RealPixelType` — `float` for a `float` input. This port computes in
//!   `f64` throughout and narrows once, when the output image is built. The
//!   derived tolerance still keys off the ITK real type, so a `float` input
//!   gets ITK's `2^-23` tolerance.
//! - **Squares.** `f²` is `MultiplyImageFilter<InputImageType, RealImageType>`
//!   (hxx:201-202), so ITK squares in the *input* pixel type and only then
//!   casts to real: `int32` inputs whose squares exceed `INT32_MAX` overflow.
//!   This port squares in `f64`.

use sitk_core::{Image, PixelId};

use crate::convolution::{ravel, unravel};
use crate::error::{FilterError, Result};
use crate::fft::{self, Complex};
use crate::image_from_f64;

/// Output pixel-type mapping for both filters: `Float32` for a `Float32`
/// input, `Float64` for everything else. **Diverges from ITK**: the yaml
/// declares `NumericTraits<T>::RealType`, which is `double` for every scalar
/// type *including* `float` (itkNumericTraits.h:1349/1356), so upstream
/// always outputs `Float64`. Breaking to fix; tracked in the
/// upstream-findings ledger §5.6 (same family as
/// `math::real_type`/`lib.rs::real_pixel_id`).
fn real_type(id: PixelId) -> PixelId {
    match id {
        PixelId::Float32 => PixelId::Float32,
        _ => PixelId::Float64,
    }
}

// ---- pixel-wise stages ----------------------------------------------------

/// `itk::Functor::Div` (itkArithmeticOpsFunctors.h:130-151): a zero divisor
/// yields `NumericTraits<Output>::max()` rather than an exception or an
/// infinity. Matches this crate's [`crate::DivOp`] for float pixel types.
fn element_quotient(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| if y == 0.0 { f64::MAX } else { x / y })
        .collect()
}

/// `ThresholdImageFilter::ThresholdBelow(0)` with outside value 0
/// (`ElementPositive`, hxx:490-506): a pixel passes through only when
/// `0 <= v <= NumericTraits::max()`, so negatives — and NaN, which fails the
/// comparison — become zero.
fn element_positive(values: &mut [f64]) {
    for v in values.iter_mut() {
        if *v < 0.0 || v.is_nan() {
            *v = 0.0;
        }
    }
}

/// `RoundImageFilter` (`ElementRound`, hxx:508-521), whose functor is
/// `itk::Math::Round == RoundHalfIntegerUp` (itkMath.h:193-205): halfway cases
/// go towards +∞, so this is `floor(v + 0.5)`, not `f64::round`.
fn element_round(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = (*v + 0.5).floor();
    }
}

// ---- transform lengths ----------------------------------------------------

/// `FactorizeNumber` (hxx:528-545): divide `n` by 2, then 3, then 5, as often
/// as each divides, and return what is left. `1` means the length is valid.
///
/// The `for (offset = 1; offset <= 3; ++offset)` loop with `ifac += offset` is
/// upstream's way of walking `ifac` through `2, 3, 5`.
///
/// **Divergence.** `FactorizeNumber(0)` never terminates upstream: `0 % 2 == 0`
/// and `0 / 2 == 0`, so the inner `for (; n % ifac == 0;)` spins forever. A
/// zero-extent input axis reaches it through `combinedImageSize = 0`. This port
/// returns `0` (an unfactorable length) instead of hanging — ledger §4.74.
fn factorize_number(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut n = n;
    let mut ifac = 2usize;
    for offset in 1..=3usize {
        while n % ifac == 0 {
            n /= ifac;
        }
        ifac += offset;
    }
    n
}

/// `FindClosestValidDimension` (hxx:549-564): the smallest length `>= n` whose
/// only prime factors are 2, 3 and 5.
///
/// Not `itkFFTPadImageFilter`'s search, and not the FFT backend's greatest
/// prime factor — see the module docs.
fn find_closest_valid_dimension(n: usize) -> usize {
    let mut candidate = n;
    while factorize_number(candidate) != 1 {
        candidate += 1;
    }
    candidate
}

// ---- FFT stages -----------------------------------------------------------

/// `CalculateForwardFFT` (hxx:375-403): zero-pad `values` on the upper side of
/// every axis out to `fft_size`, then transform.
fn forward_fft(values: &[f64], size: &[usize], fft_size: &[usize]) -> Vec<Complex> {
    let total: usize = fft_size.iter().product();
    let mut buf = vec![Complex::default(); total];
    let mut index = vec![0usize; size.len()];
    for (i, &v) in values.iter().enumerate() {
        unravel(i, size, &mut index);
        buf[ravel(&index, fft_size)] = Complex::new(v, 0.0);
    }
    fft::transform_nd(&mut buf, fft_size, false);
    buf
}

/// `CalculateInverseFFT` (hxx:408-437): invert, take the real part (the
/// imaginary part is round-off, the inputs having been real), and extract the
/// `[0, combined_size)` region the transform length may have overshot.
fn inverse_fft_cropped(
    mut spectrum: Vec<Complex>,
    fft_size: &[usize],
    combined_size: &[usize],
) -> Vec<f64> {
    fft::transform_nd(&mut spectrum, fft_size, true);
    let total: usize = combined_size.iter().product();
    let mut index = vec![0usize; combined_size.len()];
    (0..total)
        .map(|i| {
            unravel(i, combined_size, &mut index);
            spectrum[ravel(&index, fft_size)].re
        })
        .collect()
}

/// `ElementProduct` over two spectra (hxx:442-454).
fn spectrum_product(a: &[Complex], b: &[Complex]) -> Vec<Complex> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

/// One correlation term: `ifft(fft(a) * fft(b))`, cropped.
fn correlate(
    a: &[Complex],
    b: &[Complex],
    fft_size: &[usize],
    combined_size: &[usize],
) -> Vec<f64> {
    inverse_fft_cropped(spectrum_product(a, b), fft_size, combined_size)
}

// ---- pre-processing -------------------------------------------------------

/// `PreProcessMask` (hxx:317-350): an unset mask becomes an image of ones; a
/// set one is binarized by `BinaryThresholdImageFilter` with upper threshold
/// 0, inside value 0 and outside value 1 — so every pixel `<= 0` is masked
/// out and every strictly positive pixel is kept.
fn pre_process_mask(image: &Image, mask: Option<&Image>) -> Result<Vec<f64>> {
    Ok(match mask {
        None => vec![1.0; image.number_of_pixels()],
        Some(mask) => mask
            .to_f64_vec()?
            .into_iter()
            .map(|v| if v <= 0.0 { 0.0 } else { 1.0 })
            .collect(),
    })
}

/// `PreProcessImage` (hxx:354-370): zero the image wherever the (already
/// binarized) mask is zero, by multiplying the two. The equations above are
/// only correct on masked-out-to-zero images.
fn pre_process_image(image: &Image, mask: &[f64]) -> Result<Vec<f64>> {
    Ok(image
        .to_f64_vec()?
        .into_iter()
        .zip(mask)
        .map(|(v, &m)| v * m)
        .collect())
}

/// `RotateImage` (hxx:294-313): `FlipImageFilter` with every axis flipped maps
/// index `i` to `size - 1 - i` along each axis, which for a first-index-fastest
/// buffer is exactly a reversal.
fn rotate(values: &mut [f64]) {
    values.reverse();
}

/// `CalculatePrecisionTolerance` (hxx:570-598) on the denominator image.
///
/// `max <= 0` cannot happen for a genuine maximum of a `sqrt` image other than
/// `max == 0`, where `log(0) == -inf` makes `pow(2, floor(-inf))` — and so the
/// tolerance — zero, exactly as in C++.
fn precision_tolerance(max_denominator: f64, real: PixelId) -> f64 {
    let mantissa_bits = if real == PixelId::Float32 {
        -23.0
    } else {
        -52.0
    };
    1000.0 * 2.0f64.powf(mantissa_bits) * 2.0f64.powf(max_denominator.log2().floor())
}

// ---- input validation -----------------------------------------------------

/// SimpleITK casts every input through `CastImageToITK<InputImageType>` (both
/// yamls' `custom_itk_cast`), which requires one pixel type across the images
/// and their masks; `VerifyInputInformation` (hxx:602-633) additionally
/// requires each mask to match its own image's size.
fn check_inputs(
    fixed: &Image,
    moving: &Image,
    fixed_mask: Option<&Image>,
    moving_mask: Option<&Image>,
) -> Result<()> {
    if fixed.dimension() != moving.dimension() {
        return Err(FilterError::ImageDimensionMismatch {
            a: fixed.dimension(),
            b: moving.dimension(),
        });
    }
    if fixed.pixel_id() != moving.pixel_id() {
        return Err(FilterError::TypeMismatch {
            a: fixed.pixel_id(),
            b: moving.pixel_id(),
        });
    }
    for (image, mask) in [(fixed, fixed_mask), (moving, moving_mask)] {
        let Some(mask) = mask else { continue };
        if image.pixel_id() != mask.pixel_id() {
            return Err(FilterError::TypeMismatch {
                a: image.pixel_id(),
                b: mask.pixel_id(),
            });
        }
        if image.size() != mask.size() {
            return Err(FilterError::SizeMismatch {
                a: image.size().to_vec(),
                b: mask.size().to_vec(),
            });
        }
    }
    Ok(())
}

// ---- the filter -----------------------------------------------------------

/// `MaskedFFTNormalizedCorrelationImageFilter::GenerateData` (hxx:111-289).
fn generate_data(
    fixed: &Image,
    moving: &Image,
    fixed_mask: Option<&Image>,
    moving_mask: Option<&Image>,
    required_number_of_overlapping_pixels: u64,
    required_fraction_of_overlapping_pixels: f64,
) -> Result<Image> {
    check_inputs(fixed, moving, fixed_mask, moving_mask)?;
    let dim = fixed.dimension();

    let fixed_mask = pre_process_mask(fixed, fixed_mask)?;
    let mut moving_mask = pre_process_mask(moving, moving_mask)?;
    let fixed_image = pre_process_image(fixed, &fixed_mask)?;
    let mut moving_image = pre_process_image(moving, &moving_mask)?;
    rotate(&mut moving_image);
    rotate(&mut moving_mask);

    // The size of the correlation of the two images, and the transform length
    // upstream rounds it up to (hxx:155-162).
    let combined_size: Vec<usize> = (0..dim)
        .map(|d| fixed.size()[d] + moving.size()[d] - 1)
        .collect();
    let fft_size: Vec<usize> = combined_size
        .iter()
        .map(|&n| find_closest_valid_dimension(n))
        .collect();

    let fixed_fft = forward_fft(&fixed_image, fixed.size(), &fft_size);
    let fixed_mask_fft = forward_fft(&fixed_mask, fixed.size(), &fft_size);
    let moving_fft = forward_fft(&moving_image, moving.size(), &fft_size);
    let moving_mask_fft = forward_fft(&moving_mask, moving.size(), &fft_size);

    // How many voxels overlap at each shift. Exact integers up to round-off.
    let mut overlap = correlate(&fixed_mask_fft, &moving_mask_fft, &fft_size, &combined_size);
    element_round(&mut overlap);
    element_positive(&mut overlap);

    let fixed_cumulative_sum = correlate(&fixed_fft, &moving_mask_fft, &fft_size, &combined_size);
    let moving_cumulative_sum = correlate(&fixed_mask_fft, &moving_fft, &fft_size, &combined_size);
    let cross = correlate(&fixed_fft, &moving_fft, &fft_size, &combined_size);
    let products: Vec<f64> = fixed_cumulative_sum
        .iter()
        .zip(&moving_cumulative_sum)
        .map(|(&a, &b)| a * b)
        .collect();
    let numerator: Vec<f64> = cross
        .iter()
        .zip(element_quotient(&products, &overlap))
        .map(|(&c, q)| c - q)
        .collect();

    let fixed_squared: Vec<f64> = fixed_image.iter().map(|&v| v * v).collect();
    let fixed_squared_fft = forward_fft(&fixed_squared, fixed.size(), &fft_size);
    let fixed_squares: Vec<f64> = fixed_cumulative_sum.iter().map(|&v| v * v).collect();
    let mut fixed_denom: Vec<f64> = correlate(
        &fixed_squared_fft,
        &moving_mask_fft,
        &fft_size,
        &combined_size,
    )
    .iter()
    .zip(element_quotient(&fixed_squares, &overlap))
    .map(|(&c, q)| c - q)
    .collect();
    element_positive(&mut fixed_denom);

    let moving_squared: Vec<f64> = moving_image.iter().map(|&v| v * v).collect();
    let moving_squared_fft = forward_fft(&moving_squared, moving.size(), &fft_size);
    let moving_squares: Vec<f64> = moving_cumulative_sum.iter().map(|&v| v * v).collect();
    let mut moving_denom: Vec<f64> = correlate(
        &fixed_mask_fft,
        &moving_squared_fft,
        &fft_size,
        &combined_size,
    )
    .iter()
    .zip(element_quotient(&moving_squares, &overlap))
    .map(|(&c, q)| c - q)
    .collect();
    element_positive(&mut moving_denom);

    let denominator: Vec<f64> = fixed_denom
        .iter()
        .zip(&moving_denom)
        .map(|(&a, &b)| (a * b).sqrt())
        .collect();

    let max_denominator = denominator.iter().copied().fold(f64::MIN, f64::max);
    let tolerance = precision_tolerance(max_denominator, real_type(fixed.pixel_id()));

    let ncc = element_quotient(&numerator, &denominator);

    // A required overlap larger than any shift can achieve would zero the whole
    // map, so ITK first clamps it down to the observed maximum (hxx:248-256),
    // then takes whichever of the two parameters demands more pixels
    // (hxx:262-264). Both truncate towards zero on the way to an integer count.
    let maximum_number_of_overlapping_pixels =
        overlap.iter().copied().fold(f64::MIN, f64::max) as u64;
    let required_number_of_overlapping_pixels =
        required_number_of_overlapping_pixels.min(maximum_number_of_overlapping_pixels);
    let required = ((required_fraction_of_overlapping_pixels
        * maximum_number_of_overlapping_pixels as f64) as u64)
        .max(required_number_of_overlapping_pixels);

    // `Functor::PostProcessCorrelation` (hxx:79-101).
    let required = required as f64;
    let values: Vec<f64> = (0..denominator.len())
        .map(|i| {
            if denominator[i] < tolerance || overlap[i] == 0.0 || overlap[i] < required {
                0.0
            } else {
                // The functor's `< -1` / `> 1` chain, which — like `clamp` —
                // passes a NaN quotient straight through.
                ncc[i].clamp(-1.0, 1.0)
            }
        })
        .collect();

    let mut output = image_from_f64(real_type(fixed.pixel_id()), &combined_size, fixed, &values)?;

    // `GenerateOutputInformation` (hxx:699-706). ITK narrows the extent to
    // `float` before halving it; the cast is exact for any axis shorter than
    // 2^24 pixels.
    let offset: Vec<f64> = (0..dim)
        .map(|d| -f64::from((moving.size()[d] - 1) as f32) / 2.0)
        .collect();
    let origin = fixed.continuous_index_to_physical_point(&offset);
    output.set_origin(&origin)?;
    Ok(output)
}

/// `MaskedFFTNormalizedCorrelationImageFilter`: the masked normalized cross
/// correlation of `fixed` and `moving`, over every shift of one against the
/// other.
///
/// The two images need not agree in size, but each mask must match its own
/// image; all four must share one pixel type. A `None` mask means "no
/// masking", which ITK implements as an image of ones; a mask that *is* given
/// is binarized, with every pixel `<= 0` masked out (so negatives exclude, and
/// any positive value includes, regardless of magnitude).
///
/// The output is `size(fixed) + size(moving) - 1` pixels per axis, of the
/// input's `NumericTraits<T>::RealType` pixel type — see the module docs for
/// the index-to-shift correspondence and for how the output origin centres
/// each score.
///
/// `required_number_of_overlapping_pixels` and
/// `required_fraction_of_overlapping_pixels` (both defaulting to `0` in
/// SimpleITK, i.e. "no zeroing") each name a minimum overlap below which a
/// score is zeroed as statistically unreliable. The *effective* minimum is
/// `max(required_number_of_overlapping_pixels, trunc(fraction · maximum
/// observed overlap))`, after the former has itself been clamped down to that
/// same maximum — so an absurdly large count degrades to "full overlap only"
/// rather than to an all-zero map.
pub fn masked_fft_normalized_correlation(
    fixed: &Image,
    moving: &Image,
    fixed_mask: Option<&Image>,
    moving_mask: Option<&Image>,
    required_number_of_overlapping_pixels: u64,
    required_fraction_of_overlapping_pixels: f64,
) -> Result<Image> {
    generate_data(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        required_number_of_overlapping_pixels,
        required_fraction_of_overlapping_pixels,
    )
}

/// `FFTNormalizedCorrelationImageFilter`: the normalized cross correlation of
/// `fixed` and `moving`, over every shift of one against the other.
///
/// Exactly [`masked_fft_normalized_correlation`] with both masks unset — ITK's
/// subclass overrides nothing but `GenerateData`, which forwards straight to
/// the masked base. There is no computational overhead to that: the transforms
/// of the images of ones are needed either way, to count the overlap.
pub fn fft_normalized_correlation(
    fixed: &Image,
    moving: &Image,
    required_number_of_overlapping_pixels: u64,
    required_fraction_of_overlapping_pixels: f64,
) -> Result<Image> {
    generate_data(
        fixed,
        moving,
        None,
        None,
        required_number_of_overlapping_pixels,
        required_fraction_of_overlapping_pixels,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- transform lengths (ledger §2.110, §4.74) ---------------------------

    /// `FactorizeNumber` divides out 2s, 3s and 5s and nothing else, so a
    /// length is valid exactly when it is 5-smooth.
    #[test]
    fn find_closest_valid_dimension_accepts_only_two_three_and_five() {
        fn five_smooth(mut n: usize) -> bool {
            for f in [2usize, 3, 5] {
                while n % f == 0 {
                    n /= f;
                }
            }
            n == 1
        }
        for n in 1..=200usize {
            let m = find_closest_valid_dimension(n);
            assert!(m >= n, "{n} -> {m} shrank");
            assert!(five_smooth(m), "{n} -> {m} is not 5-smooth");
            for candidate in n..m {
                assert!(!five_smooth(candidate), "{n} overshot past {candidate}");
            }
        }
        // 7 is a fast PocketFFT radix, so `FFTConvolutionImageFilter` leaves 49
        // alone while this filter walks it up to 50 = 2 * 5^2.
        assert_eq!(find_closest_valid_dimension(49), 50);
        assert_eq!(crate::fft::padded_length(49), 49);
        // The two rules agree wherever the length is already 5-smooth.
        for n in [1usize, 8, 12, 100, 128] {
            assert_eq!(find_closest_valid_dimension(n), n);
            assert_eq!(crate::fft::padded_length(n), n);
        }
    }

    /// `FactorizeNumber(0)` spins forever upstream; here it reports "no
    /// factorization" and the search moves on. Ledger §4.74.
    #[test]
    fn factorize_number_of_zero_terminates() {
        assert_eq!(factorize_number(0), 0);
        assert_eq!(find_closest_valid_dimension(0), 1);
    }

    fn values(image: &Image) -> Vec<f64> {
        image.scalar_slice::<f64>().unwrap().to_vec()
    }

    /// Deterministic non-degenerate filler.
    fn noise(size: &[usize], seed: u64) -> Image {
        let n: usize = size.iter().product();
        let mut state = seed;
        let data = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % 1000) as f64 / 100.0 - 5.0
            })
            .collect();
        img(size, data)
    }

    /// The multi-index of the largest value, first index fastest.
    fn argmax(image: &Image) -> Vec<usize> {
        let vals = values(image);
        let (best, _) = vals
            .iter()
            .enumerate()
            .fold(
                (0usize, f64::MIN),
                |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                },
            );
        let mut index = vec![0usize; image.dimension()];
        unravel(best, image.size(), &mut index);
        index
    }

    fn at(image: &Image, index: &[usize]) -> f64 {
        values(image)[image.linear_index(index)]
    }

    /// `moving[x] = fixed[x + offset]`, a `size`-sized window of `fixed`.
    fn window(fixed: &Image, offset: &[usize], size: &[usize]) -> Image {
        let n: usize = size.iter().product();
        let mut index = vec![0usize; size.len()];
        let vals = values(fixed);
        let data = (0..n)
            .map(|i| {
                unravel(i, size, &mut index);
                let source: Vec<usize> = index.iter().zip(offset).map(|(&a, &b)| a + b).collect();
                vals[fixed.linear_index(&source)]
            })
            .collect();
        img(size, data)
    }

    // ---- geometry ---------------------------------------------------------

    #[test]
    fn output_size_is_the_combined_size() {
        let fixed = noise(&[6, 4], 1);
        let moving = noise(&[3, 5], 2);
        let out = fft_normalized_correlation(&fixed, &moving, 0, 0.0).unwrap();
        assert_eq!(out.size(), &[8, 8]);
    }

    #[test]
    fn output_pixel_type_is_the_inputs_real_type() {
        let f32_image = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        let out = fft_normalized_correlation(&f32_image, &f32_image, 0, 0.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);

        let u8_image = Image::from_vec(&[3, 3], vec![1u8; 9]).unwrap();
        let out = fft_normalized_correlation(&u8_image, &u8_image, 0, 0.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
    }

    #[test]
    fn output_origin_centres_the_moving_image_on_each_score() {
        let mut fixed = noise(&[6, 4], 3);
        fixed.set_spacing(&[2.0, 0.5]).unwrap();
        fixed.set_origin(&[10.0, -3.0]).unwrap();
        let moving = noise(&[3, 5], 4);

        let out = fft_normalized_correlation(&fixed, &moving, 0, 0.0).unwrap();
        // Continuous index (-(3-1)/2, -(5-1)/2) = (-1, -2) of the fixed image.
        assert_eq!(out.origin(), &[10.0 - 2.0, -3.0 - 1.0]);
        assert_eq!(out.spacing(), fixed.spacing());
        assert_eq!(out.direction(), fixed.direction());
    }

    // ---- the correlation itself -------------------------------------------

    #[test]
    fn autocorrelation_peaks_at_the_centre_shift_with_value_one() {
        // Symmetric under a flip through both axes, so the peak cannot be
        // mistaken for an artefact of the rotation.
        #[rustfmt::skip]
        let fixed = img(&[5, 5], vec![
            1.0, 2.0, 3.0, 2.0, 1.0,
            2.0, 4.0, 6.0, 4.0, 2.0,
            3.0, 6.0, 9.0, 6.0, 3.0,
            2.0, 4.0, 6.0, 4.0, 2.0,
            1.0, 2.0, 3.0, 2.0, 1.0,
        ]);
        let out = fft_normalized_correlation(&fixed, &fixed, 25, 0.0).unwrap();
        assert_eq!(out.size(), &[9, 9]);
        // Full overlap happens only at the zero shift, k = size - 1 = 4.
        assert_eq!(argmax(&out), vec![4, 4]);
        assert!(
            (at(&out, &[4, 4]) - 1.0).abs() < 1e-9,
            "{}",
            at(&out, &[4, 4])
        );
    }

    #[test]
    fn the_peak_sits_at_the_shift_between_the_two_images() {
        let fixed = noise(&[8, 8], 5);
        let offset = [2usize, 1];
        let moving = window(&fixed, &offset, &[3, 3]);

        // Require the template to overlap in full, so only the 6x6 shifts that
        // place it wholly inside the fixed image can win.
        let out = fft_normalized_correlation(&fixed, &moving, 9, 0.0).unwrap();
        assert_eq!(out.size(), &[10, 10]);

        // k = offset + size(moving) - 1.
        let peak = vec![offset[0] + 2, offset[1] + 2];
        assert_eq!(argmax(&out), peak);
        assert!((at(&out, &peak) - 1.0).abs() < 1e-9, "{}", at(&out, &peak));
    }

    #[test]
    fn a_flipped_template_does_not_win_the_correlation() {
        // Guards the rotation's direction: `RotateImage` reflects the moving
        // image so that the FFT product is a correlation, not a convolution.
        // Correlating with a flipped window would peak at the mirrored shift.
        let fixed = noise(&[8, 8], 6);
        let moving = window(&fixed, &[1, 3], &[3, 3]);
        let mut flipped_values = values(&moving);
        flipped_values.reverse();
        let flipped = img(&[3, 3], flipped_values);

        let straight = fft_normalized_correlation(&fixed, &moving, 9, 0.0).unwrap();
        let mirrored = fft_normalized_correlation(&fixed, &flipped, 9, 0.0).unwrap();
        assert_eq!(argmax(&straight), vec![3, 5]);
        assert_ne!(argmax(&mirrored), vec![3, 5]);
    }

    #[test]
    fn every_score_stays_inside_the_correlation_bounds() {
        let fixed = noise(&[7, 5], 7);
        let moving = noise(&[4, 6], 8);
        let out = fft_normalized_correlation(&fixed, &moving, 0, 0.0).unwrap();
        for (i, &v) in values(&out).iter().enumerate() {
            assert!((-1.0..=1.0).contains(&v), "index {i}: {v}");
        }
    }

    #[test]
    fn one_dimensional_correlation_matches_the_direct_masked_ncc() {
        // Independent evaluation of Padfield's equations at one shift, in the
        // spatial domain, on the overlap of the two supports.
        let fixed = img(&[5], vec![1.0, 4.0, -2.0, 3.0, 0.5]);
        let moving = img(&[3], vec![2.0, -1.0, 1.5]);
        let out = fft_normalized_correlation(&fixed, &moving, 3, 0.0).unwrap();

        for shift in 0..=2usize {
            let f: Vec<f64> = values(&fixed)[shift..shift + 3].to_vec();
            let g = values(&moving);
            let n = 3.0;
            let sum_f: f64 = f.iter().sum();
            let sum_g: f64 = g.iter().sum();
            let numerator: f64 =
                f.iter().zip(&g).map(|(a, b)| a * b).sum::<f64>() - sum_f * sum_g / n;
            let var_f: f64 = f.iter().map(|a| a * a).sum::<f64>() - sum_f * sum_f / n;
            let var_g: f64 = g.iter().map(|b| b * b).sum::<f64>() - sum_g * sum_g / n;
            let want = numerator / (var_f * var_g).sqrt();
            let got = at(&out, &[shift + 2]);
            assert!((got - want).abs() < 1e-9, "shift {shift}: {got} vs {want}");
        }
    }

    // ---- masks ------------------------------------------------------------

    #[test]
    fn all_ones_masks_reproduce_the_unmasked_filter() {
        let fixed = noise(&[6, 5], 9);
        let moving = noise(&[3, 4], 10);
        let fixed_mask = img(&[6, 5], vec![1.0; 30]);
        let moving_mask = img(&[3, 4], vec![1.0; 12]);

        let plain = fft_normalized_correlation(&fixed, &moving, 0, 0.0).unwrap();
        let masked = masked_fft_normalized_correlation(
            &fixed,
            &moving,
            Some(&fixed_mask),
            Some(&moving_mask),
            0,
            0.0,
        )
        .unwrap();
        assert_eq!(values(&masked), values(&plain));
    }

    #[test]
    fn mask_values_are_binarized_by_their_sign() {
        // Non-positive masks out, any positive value masks in.
        let fixed = noise(&[5, 4], 11);
        let moving = noise(&[3, 3], 12);
        let raw = img(
            &[5, 4],
            (0..20)
                .map(|i| match i % 4 {
                    0 => -3.0,
                    1 => 0.0,
                    2 => 7.0,
                    _ => 0.25,
                })
                .collect(),
        );
        let binary = img(
            &[5, 4],
            (0..20).map(|i| if i % 4 < 2 { 0.0 } else { 1.0 }).collect(),
        );

        let from_raw =
            masked_fft_normalized_correlation(&fixed, &moving, Some(&raw), None, 0, 0.0).unwrap();
        let from_binary =
            masked_fft_normalized_correlation(&fixed, &moving, Some(&binary), None, 0, 0.0)
                .unwrap();
        assert_eq!(values(&from_raw), values(&from_binary));
    }

    #[test]
    fn a_mask_recovers_the_shift_that_corruption_hides() {
        let fixed = noise(&[10, 10], 13);
        let offset = [3usize, 4];
        let clean = window(&fixed, &offset, &[5, 5]);
        let peak = vec![offset[0] + 4, offset[1] + 4];

        // Corrupt the template's last column with a large constant.
        let mut corrupted_values = values(&clean);
        for row in 0..5 {
            corrupted_values[4 + row * 5] = 500.0;
        }
        let corrupted = img(&[5, 5], corrupted_values);
        // A moving mask that excludes exactly that column.
        let moving_mask = img(
            &[5, 5],
            (0..25)
                .map(|i| if i % 5 == 4 { 0.0 } else { 1.0 })
                .collect(),
        );

        let unmasked = fft_normalized_correlation(&fixed, &corrupted, 25, 0.0).unwrap();
        assert_ne!(
            argmax(&unmasked),
            peak,
            "the corruption must actually hide the shift"
        );

        let masked = masked_fft_normalized_correlation(
            &fixed,
            &corrupted,
            None,
            Some(&moving_mask),
            20,
            0.0,
        )
        .unwrap();
        assert_eq!(argmax(&masked), peak);
        assert!(
            (at(&masked, &peak) - 1.0).abs() < 1e-9,
            "{}",
            at(&masked, &peak)
        );
    }

    // ---- required overlap -------------------------------------------------

    /// The 8x8 / 3x3 pin used by the overlap-zeroing boundary cases. Overlap at
    /// output index `k` is `Π min(k[d] + 1, 3, 10 - k[d])`, so `k = (0,0)` sees
    /// 1 pixel, `k = (1,1)` sees 4, and `k = (2,2)` sees the full 9.
    fn overlap_fixture() -> (Image, Image) {
        (noise(&[8, 8], 14), noise(&[3, 3], 15))
    }

    #[test]
    fn zero_required_overlap_leaves_partial_overlap_scores_alive() {
        let (fixed, moving) = overlap_fixture();
        let out = fft_normalized_correlation(&fixed, &moving, 0, 0.0).unwrap();
        // 1 overlapping pixel: both variances vanish, so the precision
        // tolerance zeroes it even without a required count.
        assert_eq!(at(&out, &[0, 0]), 0.0);
        // 4 overlapping pixels: a real score survives.
        assert_ne!(at(&out, &[1, 1]), 0.0);
        assert_ne!(at(&out, &[2, 2]), 0.0);
    }

    #[test]
    fn required_number_of_overlapping_pixels_zeroes_the_low_overlap_border() {
        let (fixed, moving) = overlap_fixture();
        let out = fft_normalized_correlation(&fixed, &moving, 9, 0.0).unwrap();
        // 4 < 9 overlapping pixels: now zeroed.
        assert_eq!(at(&out, &[1, 1]), 0.0);
        // Exactly 9: the comparison is `overlap < required`, so this survives.
        assert_ne!(at(&out, &[2, 2]), 0.0);
    }

    #[test]
    fn required_number_of_overlapping_pixels_is_clamped_to_the_observed_maximum() {
        let (fixed, moving) = overlap_fixture();
        // The maximum overlap is 9; a larger request must degrade to 9 rather
        // than zero the entire map.
        let clamped = fft_normalized_correlation(&fixed, &moving, 1_000_000, 0.0).unwrap();
        let exact = fft_normalized_correlation(&fixed, &moving, 9, 0.0).unwrap();
        assert_eq!(values(&clamped), values(&exact));
        assert_ne!(at(&clamped, &[2, 2]), 0.0);
    }

    #[test]
    fn required_fraction_of_overlapping_pixels_scales_the_observed_maximum() {
        let (fixed, moving) = overlap_fixture();
        // 1.0 * 9 == 9 required pixels.
        let by_fraction = fft_normalized_correlation(&fixed, &moving, 0, 1.0).unwrap();
        let by_count = fft_normalized_correlation(&fixed, &moving, 9, 0.0).unwrap();
        assert_eq!(values(&by_fraction), values(&by_count));

        // trunc(0.5 * 9) == 4, and the `<` comparison keeps the 4-pixel corner.
        let half = fft_normalized_correlation(&fixed, &moving, 0, 0.5).unwrap();
        assert_ne!(at(&half, &[1, 1]), 0.0);
        // trunc(0.55 * 9) == 4 too — the truncation, not a round, decides.
        let just_over = fft_normalized_correlation(&fixed, &moving, 0, 0.55).unwrap();
        assert_eq!(values(&just_over), values(&half));
    }

    #[test]
    fn the_larger_of_the_two_required_overlaps_wins() {
        let (fixed, moving) = overlap_fixture();
        // count 9 beats trunc(0.5 * 9) == 4.
        let both = fft_normalized_correlation(&fixed, &moving, 9, 0.5).unwrap();
        let count_only = fft_normalized_correlation(&fixed, &moving, 9, 0.0).unwrap();
        assert_eq!(values(&both), values(&count_only));
        assert_eq!(at(&both, &[1, 1]), 0.0);
    }

    // ---- error paths ------------------------------------------------------

    #[test]
    fn images_of_different_dimension_are_rejected() {
        let fixed = noise(&[4, 4], 16);
        let moving = noise(&[4], 17);
        assert_eq!(
            fft_normalized_correlation(&fixed, &moving, 0, 0.0),
            Err(FilterError::ImageDimensionMismatch { a: 2, b: 1 })
        );
    }

    #[test]
    fn images_of_different_pixel_type_are_rejected() {
        let fixed = noise(&[4, 4], 18);
        let moving = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        assert_eq!(
            fft_normalized_correlation(&fixed, &moving, 0, 0.0),
            Err(FilterError::TypeMismatch {
                a: PixelId::Float64,
                b: PixelId::Float32,
            })
        );
    }

    #[test]
    fn a_mask_that_does_not_match_its_own_image_is_rejected() {
        let fixed = noise(&[4, 4], 19);
        let moving = noise(&[3, 3], 20);
        let bad = img(&[3, 3], vec![1.0; 9]);
        assert_eq!(
            masked_fft_normalized_correlation(&fixed, &moving, Some(&bad), None, 0, 0.0),
            Err(FilterError::SizeMismatch {
                a: vec![4, 4],
                b: vec![3, 3],
            })
        );
        // The moving mask is checked against the moving image, not the fixed.
        let bad = img(&[4, 4], vec![1.0; 16]);
        assert_eq!(
            masked_fft_normalized_correlation(&fixed, &moving, None, Some(&bad), 0, 0.0),
            Err(FilterError::SizeMismatch {
                a: vec![3, 3],
                b: vec![4, 4],
            })
        );
    }

    #[test]
    fn a_mask_of_a_different_pixel_type_is_rejected() {
        let fixed = noise(&[3, 3], 21);
        let mask = Image::from_vec(&[3, 3], vec![1u8; 9]).unwrap();
        assert_eq!(
            masked_fft_normalized_correlation(&fixed, &fixed, Some(&mask), None, 0, 0.0),
            Err(FilterError::TypeMismatch {
                a: PixelId::Float64,
                b: PixelId::UInt8,
            })
        );
    }

    // ---- the derived pieces ----------------------------------------------

    #[test]
    fn round_takes_halfway_cases_upwards() {
        let mut v = vec![-1.5, -0.5, 0.5, 1.5, 2.5];
        element_round(&mut v);
        assert_eq!(v, vec![-1.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn positive_clamps_negatives_and_nan() {
        let mut v = vec![-1.0, 0.0, 2.0, f64::NAN];
        element_positive(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn quotient_by_zero_yields_the_pixel_types_maximum() {
        assert_eq!(element_quotient(&[3.0], &[0.0]), vec![f64::MAX]);
    }

    #[test]
    fn precision_tolerance_is_a_thousand_ulps_at_the_maxima_magnitude() {
        // 1000 * 2^-52 * 2^floor(log2(12)) = 1000 * 2^-52 * 8.
        let want = 1000.0 * 2.0f64.powi(-52) * 8.0;
        assert_eq!(precision_tolerance(12.0, PixelId::Float64), want);
        assert_eq!(
            precision_tolerance(12.0, PixelId::Float32),
            1000.0 * 2.0f64.powi(-23) * 8.0
        );
        // log(0) == -inf, so `pow(2, floor(-inf))` collapses the tolerance.
        assert_eq!(precision_tolerance(0.0, PixelId::Float64), 0.0);
    }
}
