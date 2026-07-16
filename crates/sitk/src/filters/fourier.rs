//! The five public FFT filters of SimpleITK's `FFT` group that transform or
//! resize an image: [`forward_fft`], [`inverse_fft`],
//! [`real_to_half_hermitian_forward_fft`], [`half_hermitian_to_real_inverse_fft`]
//! and [`fft_pad`].
//!
//! (`FFTShiftImageFilter` is the sixth, and lives in [`mod@crate::filters::fft_shift`].)
//!
//! # Which ITK implementation this ports
//!
//! `itk::ForwardFFTImageFilter` and friends are abstract bases; the object
//! factory picks a backend. This checkout registers **PocketFFT** as the
//! always-available default
//! (`Modules/Filtering/FFT/src/itkPocketFFTImageFilterInitFactory.cxx`), so
//! these four transforms are ports of the `itkPocketFFT*FFTImageFilter.hxx`
//! `GenerateData` bodies on top of `crate::filters::fft`, which reimplements the
//! pocketfft kernels this workspace takes no dependency on.
//!
//! None of them pads. A `ForwardFFTImageFilter` on a 97-pixel axis transforms
//! 97 points (`ForwardFFTImageFilter.yaml` has no pad member); the caller who
//! wants a fast length reaches for [`fft_pad`] first, which is exactly how
//! `FFTConvolutionImageFilter` composes them.
//!
//! # Pixel types
//!
//! Straight from the yamls' `pixel_types` and each ITK filter's default
//! `TOutputImage`:
//!
//! | filter | input | output |
//! |---|---|---|
//! | [`forward_fft`], [`real_to_half_hermitian_forward_fft`] | `RealPixelIDTypeList`: `Float32`, `Float64` | `ComplexFloat32`, `ComplexFloat64` |
//! | [`inverse_fft`], [`half_hermitian_to_real_inverse_fft`] | `ComplexPixelIDTypeList`: `ComplexFloat32`, `ComplexFloat64` | `Float32`, `Float64` |
//! | [`fft_pad`] | `BasicPixelIDTypeList`: the ten scalar types | same as input |
//!
//! `itk::ForwardFFTImageFilter<TInputImage>` defaults `TOutputImage` to
//! `Image<std::complex<typename TInputImage::PixelType>, Dim>`
//! (`itkForwardFFTImageFilter.h`), so a `Float32` image transforms to a
//! `ComplexFloat32` one and never to `ComplexFloat64`.
//!
//! # Half-Hermitian layout
//!
//! `RealToHalfHermitianForwardFFTImageFilter` returns only the `x < N₀/2 + 1`
//! slice of the full spectrum (`itkRealToHalfHermitianForwardFFTImageFilter.hxx:58`);
//! all other axes keep their full extent. The discarded half is recoverable
//! only if the *parity* of `N₀` is known, since both `N₀ = 2M − 2` and
//! `N₀ = 2M − 1` produce `M` columns. ITK carries that parity on the forward
//! filter as `ActualXDimensionIsOdd`, set from the input in
//! `GenerateOutputInformation` (`:70`), and the inverse filter reads its *own*
//! copy of the flag to size its output
//! (`itkHalfHermitianToRealInverseFFTImageFilter.hxx:65-70`).
//!
//! SimpleITK exposes the flag on the inverse filter only
//! (`HalfHermitianToRealInverseFFTImageFilter.yaml`; the forward yaml's
//! `members` is empty), so a SimpleITK caller must remember the parity itself
//! and the `false` default silently round-trips a 13-wide input to 12 wide.
//! This port does not reproduce that: [`real_to_half_hermitian_forward_fft`]
//! returns the flag ITK's forward filter already computes, and
//! [`half_hermitian_to_real_inverse_fft`] requires it — pass the one through to
//! the other and the round trip cannot lose the width. Ledger §3.39.
//!
//! # Precision
//!
//! The transforms run in `f64` and round once on the way out — the crate-wide
//! divergence of ledger §4.1. Upstream instantiates pocketfft on the component
//! type, so a `ComplexFloat32` round trip accumulates `float` round-off there
//! and `f64` round-off here.
//!
//! Ledger: §2.109, §2.111, §3.38, §3.39, §4.1.

use crate::core::{Complex as PixelComplex, Image, PixelId, Real};

use crate::filters::Result;
use crate::filters::convolution::ConvolutionBoundaryCondition;
use crate::filters::error::FilterError;
use crate::filters::fft::{self, Complex, LineKernel};
use crate::filters::geometry;

/// `FFTPadImageFilter::DefaultSizeGreatestPrimeFactor()`
/// (`FFTPadImageFilter.yaml`'s `custom_methods`): the largest prime factor the
/// FFT backend has a fast kernel for.
///
/// `11`, from PocketFFT (`itkPocketFFTForwardFFTImageFilter.hxx:72-76`). The
/// yaml's own documentation still says "5 for VNL" — ledger §3.38.
pub const DEFAULT_SIZE_GREATEST_PRIME_FACTOR: usize = fft::DEFAULT_SIZE_GREATEST_PRIME_FACTOR;

// ---- shared plumbing -------------------------------------------------------

/// Read a real scalar image into the transform's `f64` complex buffer.
fn embed_real<T: Real>(img: &Image) -> Result<Vec<Complex>> {
    Ok(img
        .scalar_slice::<T>()?
        .iter()
        .map(|&v| Complex::new(v.as_f64(), 0.0))
        .collect())
}

/// Read a complex image's interleaved `re, im, …` buffer into the transform's
/// `f64` complex buffer.
fn embed_complex<T: Real>(img: &Image) -> Result<Vec<Complex>> {
    Ok(img
        .complex_components::<T>()?
        .chunks_exact(2)
        .map(|c| Complex::new(c[0].as_f64(), c[1].as_f64()))
        .collect())
}

/// Round an `f64` complex buffer back to a `Complex{Float32,Float64}` image of
/// extent `size`, keeping `img`'s origin, spacing and direction.
///
/// Upstream's FFT filters change only the largest possible region; spacing
/// "has no meaning in the result of an FFT" and is propagated unchanged
/// (`itkRealToHalfHermitianForwardFFTImageFilter.hxx:44-46`).
fn complex_image_like<T: Real>(img: &Image, size: &[usize], buf: &[Complex]) -> Result<Image> {
    let data: Vec<PixelComplex<T>> = buf
        .iter()
        .map(|c| PixelComplex::new(T::from_f64(c.re), T::from_f64(c.im)))
        .collect();
    let mut out = Image::from_vec_complex(size, data)?;
    out.copy_geometry_from(img);
    Ok(out)
}

/// The same, for a real-valued output.
fn real_image_like<T: Real>(img: &Image, size: &[usize], values: &[f64]) -> Result<Image> {
    let data: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
    let mut out = Image::from_vec(size, data)?;
    out.copy_geometry_from(img);
    Ok(out)
}

/// A real input pixel type that is not `Float32`/`Float64` is out of the
/// `RealPixelIDTypeList` the forward yamls declare.
fn dispatch_real<F>(img: &Image, f: F) -> Result<Image>
where
    F: FnOnce(&Image, PixelId) -> Result<Image>,
{
    match img.pixel_id() {
        id @ (PixelId::Float32 | PixelId::Float64) => f(img, id),
        other => Err(FilterError::RequiresRealPixelType(other)),
    }
}

/// The `ComplexPixelIDTypeList` half of the same check.
fn dispatch_complex<F>(img: &Image, f: F) -> Result<Image>
where
    F: FnOnce(&Image, PixelId) -> Result<Image>,
{
    match img.pixel_id() {
        id @ (PixelId::ComplexFloat32 | PixelId::ComplexFloat64) => f(img, id),
        other => Err(crate::core::Error::RequiresComplexPixelType(other).into()),
    }
}

// ---- ForwardFFTImageFilter -------------------------------------------------

fn forward_fft_typed<T: Real>(img: &Image) -> Result<Image> {
    let size = img.size().to_vec();
    let mut buf = embed_real::<T>(img)?;
    fft::transform_nd(
        &mut buf,
        &size,
        false,
        LineKernel::for_output(T::COMPLEX_ID),
    );
    complex_image_like::<T>(img, &size, &buf)
}

/// `ForwardFFTImageFilter` (`itkPocketFFTForwardFFTImageFilter.hxx:53-67`):
/// the full complex Fourier transform of a real image, same size as the input.
///
/// The unnormalized `exp(-2πi·k·x/N)` kernel — pocketfft's `c2c(FORWARD)` with
/// `fct = 1`. The output has Hermitian symmetry, which is what
/// [`real_to_half_hermitian_forward_fft`] exploits.
///
/// No padding happens: an axis of prime length transforms at that length,
/// through `crate::filters::fft`'s Bluestein path. Use [`fft_pad`] first for a fast
/// length.
///
/// `Float32` in gives `ComplexFloat32` out, `Float64` gives `ComplexFloat64`;
/// any other pixel type is [`FilterError::RequiresRealPixelType`].
pub fn forward_fft(img: &Image) -> Result<Image> {
    dispatch_real(img, |img, id| match id {
        PixelId::Float32 => forward_fft_typed::<f32>(img),
        _ => forward_fft_typed::<f64>(img),
    })
}

// ---- InverseFFTImageFilter -------------------------------------------------

fn inverse_fft_typed<T: Real>(img: &Image) -> Result<Image> {
    let size = img.size().to_vec();
    let total: usize = size.iter().product();
    let mut buf = embed_complex::<T>(img)?;
    fft::transform_nd(&mut buf, &size, true, LineKernel::for_output(T::PIXEL_ID));
    debug_assert_eq!(buf.len(), total);
    let values: Vec<f64> = buf.iter().map(|c| c.re).collect();
    real_image_like::<T>(img, &size, &values)
}

/// `InverseFFTImageFilter` (`itkPocketFFTInverseFFTImageFilter.hxx:31-73`):
/// the full complex inverse transform, of which only the real part is kept.
///
/// `c2c(BACKWARD)` scaled by `1 / totalSize`, then `work[i].real()`. As the
/// yaml puts it: "If the input does not have Hermitian symmetry, the imaginary
/// component is discarded" — no check, no warning.
///
/// `ComplexFloat32` in gives `Float32` out; `ComplexFloat64` gives `Float64`.
pub fn inverse_fft(img: &Image) -> Result<Image> {
    dispatch_complex(img, |img, id| match id {
        PixelId::ComplexFloat32 => inverse_fft_typed::<f32>(img),
        _ => inverse_fft_typed::<f64>(img),
    })
}

// ---- RealToHalfHermitianForwardFFTImageFilter ------------------------------

/// `outputSize[0] = inputSize[0] / 2 + 1`, every other axis unchanged
/// (`itkRealToHalfHermitianForwardFFTImageFilter.hxx:58-65`).
fn half_hermitian_size(full: &[usize]) -> Vec<usize> {
    let mut half = full.to_vec();
    half[0] = full[0] / 2 + 1;
    half
}

fn real_to_half_hermitian_typed<T: Real>(img: &Image) -> Result<Image> {
    let full = img.size().to_vec();
    let half = half_hermitian_size(&full);

    // pocketfft's `r2c` over all axes is an `r2c` along x followed by a `c2c`
    // over the rest (pocketfft_hdronly.h:3683-3691); on a real input that is
    // the full complex transform with the redundant x columns dropped.
    let mut buf = embed_real::<T>(img)?;
    fft::transform_nd(
        &mut buf,
        &full,
        false,
        LineKernel::for_output(T::COMPLEX_ID),
    );

    let out: Vec<Complex> = crop_leading_axis(&buf, &full, half[0]);
    complex_image_like::<T>(img, &half, &out)
}

/// Keep the first `keep` entries of the (contiguous) leading axis of every line.
fn crop_leading_axis(data: &[Complex], full: &[usize], keep: usize) -> Vec<Complex> {
    data.chunks_exact(full[0])
        .flat_map(|line| line[..keep].iter().copied())
        .collect()
}

/// `RealToHalfHermitianForwardFFTImageFilter`
/// (`itkPocketFFTRealToHalfHermitianForwardFFTImageFilter.hxx:29-63`): the
/// forward transform with the redundant half of the x axis dropped, so the
/// output is `N₀/2 + 1` wide.
///
/// The dropped columns are the conjugate reflection of the kept ones, so
/// nothing is lost *except* the parity of `N₀`. That parity is returned
/// alongside the spectrum — it is ITK's `GetActualXDimensionIsOdd()`
/// (`itkRealToHalfHermitianForwardFFTImageFilter.hxx:70`) — and is what
/// [`half_hermitian_to_real_inverse_fft`] must be given to invert the spectrum
/// back to the original width. See the module docs and ledger §3.39.
///
/// `Float32` in gives `ComplexFloat32` out, `Float64` gives `ComplexFloat64`.
pub fn real_to_half_hermitian_forward_fft(img: &Image) -> Result<(Image, bool)> {
    let out = dispatch_real(img, |img, id| match id {
        PixelId::Float32 => real_to_half_hermitian_typed::<f32>(img),
        _ => real_to_half_hermitian_typed::<f64>(img),
    })?;
    Ok((out, !img.size()[0].is_multiple_of(2)))
}

// ---- HalfHermitianToRealInverseFFTImageFilter ------------------------------

/// `outputSize[0] = (inputSize[0] - 1) * 2`, plus one when the flag is set
/// (`itkHalfHermitianToRealInverseFFTImageFilter.hxx:65-70`).
fn half_hermitian_output_size(half: &[usize], actual_x_dimension_is_odd: bool) -> Vec<usize> {
    let mut full = half.to_vec();
    full[0] = (half[0] - 1) * 2 + usize::from(actual_x_dimension_is_odd);
    full
}

/// Expand one half-Hermitian x-line of `half` values into the full `len`-long
/// spectrum, exactly as pocketfft's `general_c2r` fills its scratch buffer
/// (`pocketfft_hdronly.h:3566-3585`).
///
/// Two elements lose their imaginary part on the way in: the DC term, and — for
/// an even `len` — the Nyquist term, whose imaginary parts pocketfft simply
/// never reads (`tdata[0] = in[…].r`, `tdata[len-1] = in[…].r`). A truly
/// Hermitian line has zero there; a hand-built input need not, and upstream
/// silently ignores whatever it finds. Ledger §2.111.
fn expand_hermitian_line(half: &[Complex], len: usize, out: &mut [Complex]) {
    out[0] = Complex::new(half[0].re, 0.0);
    let mirror_end = len.div_ceil(2);
    for k in 1..mirror_end {
        out[k] = half[k];
        out[len - k] = half[k].conj();
    }
    if len.is_multiple_of(2) {
        out[len / 2] = Complex::new(half[len / 2].re, 0.0);
    }
}

fn half_hermitian_to_real_typed<T: Real>(img: &Image, odd: bool) -> Result<Image> {
    let half = img.size().to_vec();
    let full = half_hermitian_output_size(&half, odd);
    let total: usize = full.iter().product();
    if total == 0 {
        // `half[0] == 1` with an even parity means `(1 - 1) * 2 == 0` columns:
        // ITK allocates an empty region and pocketfft returns immediately
        // (`util::prod(shape_out) == 0`). Same here, before the line loop can
        // index into a zero-length buffer.
        return real_image_like::<T>(img, &full, &[]);
    }

    // `general_c2r` transforms every axis but x first, on the *half*-sized
    // array (pocketfft_hdronly.h:3717-3731), and only then expands x.
    let mut buf = embed_complex::<T>(img)?;
    let other_axes: Vec<usize> = (1..half.len()).collect();
    let line_kernel = LineKernel::for_output(T::PIXEL_ID);
    fft::transform_axes(&mut buf, &half, &other_axes, false, line_kernel);

    let scale = 1.0 / total as f64;
    let mut line = vec![Complex::default(); full[0]];
    let mut values = Vec::with_capacity(total);
    for chunk in buf.chunks_exact(half[0]) {
        expand_hermitian_line(chunk, full[0], &mut line);
        fft::transform_axes(&mut line, &[full[0]], &[0], false, line_kernel);
        values.extend(line.iter().map(|c| c.re * scale));
    }

    real_image_like::<T>(img, &full, &values)
}

/// `HalfHermitianToRealInverseFFTImageFilter`
/// (`itkPocketFFTHalfHermitianToRealInverseFFTImageFilter.hxx:27-67`): invert a
/// half-Hermitian spectrum back to a real image.
///
/// `actual_x_dimension_is_odd` is SimpleITK's `SetActualXDimensionIsOdd`, and
/// it alone decides whether the `M`-wide input came from a `2M − 2` or a
/// `2M − 1` wide real image. Unlike upstream it has no `false` default here:
/// pass the flag [`real_to_half_hermitian_forward_fft`] returned. Get it wrong
/// and you get a differently sized, differently valued image — no error.
///
/// `ComplexFloat32` in gives `Float32` out; `ComplexFloat64` gives `Float64`.
///
/// The imaginary parts of the DC column, and of the Nyquist column when the
/// output width is even, are ignored rather than checked; see
/// `expand_hermitian_line`.
pub fn half_hermitian_to_real_inverse_fft(
    img: &Image,
    actual_x_dimension_is_odd: bool,
) -> Result<Image> {
    dispatch_complex(img, |img, id| match id {
        PixelId::ComplexFloat32 => {
            half_hermitian_to_real_typed::<f32>(img, actual_x_dimension_is_odd)
        }
        _ => half_hermitian_to_real_typed::<f64>(img, actual_x_dimension_is_odd),
    })
}

// ---- FFTPadImageFilter -----------------------------------------------------

/// `FFTPadImageFilter` (`itkFFTPadImageFilter.hxx:41-73`): grow every axis to
/// the next size whose greatest prime factor is at most
/// `size_greatest_prime_factor`, filling the new pixels through
/// `boundary_condition`.
///
/// The new pixels are split `padSize / 2` low and the rest high, so an odd
/// `padSize` puts the extra pixel on the *upper* side. The output origin shifts
/// by the low pad, as it does for every `PadImageFilter`.
///
/// `size_greatest_prime_factor` has three regimes, one of which contradicts the
/// filter's documentation (ledger §2.109):
///
/// - `>= 2` — the next-size search. The default, [`DEFAULT_SIZE_GREATEST_PRIME_FACTOR`],
///   is `11`.
/// - `== 1` — grow each axis to an *even* size. The doc-comment says `1` and
///   below "disable the extra padding"; the code disagrees.
/// - `== 0` — no padding.
///
/// SimpleITK's parameter is a signed `int` while ITK's member is an unsigned
/// `SizeValueType`, so a negative value there wraps to a huge one and disables
/// padding; `usize::MAX` does the same here.
///
/// Accepts the ten scalar pixel types (`BasicPixelIDTypeList`), and returns the
/// same type.
pub fn fft_pad(
    img: &Image,
    boundary_condition: ConvolutionBoundaryCondition,
    size_greatest_prime_factor: usize,
) -> Result<Image> {
    let dim = img.dimension();
    let mut lower = Vec::with_capacity(dim);
    let mut upper = Vec::with_capacity(dim);
    for &n in img.size() {
        let pad = fft::fft_pad_size(n, size_greatest_prime_factor);
        lower.push(pad / 2);
        upper.push(pad - pad / 2);
    }

    match boundary_condition {
        ConvolutionBoundaryCondition::ZeroPad => geometry::constant_pad(img, &lower, &upper, 0.0),
        ConvolutionBoundaryCondition::ZeroFluxNeumannPad => {
            geometry::zero_flux_neumann_pad(img, &lower, &upper)
        }
        ConvolutionBoundaryCondition::PeriodicPad => geometry::wrap_pad(img, &lower, &upper),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(size: &[usize]) -> Image {
        let total: usize = size.iter().product();
        let data: Vec<f64> = (0..total).map(|i| (i as f64 * 0.37).sin() + 0.1).collect();
        Image::from_vec(size, data).unwrap()
    }

    fn real_values(img: &Image) -> Vec<f64> {
        img.scalar_slice::<f64>().unwrap().to_vec()
    }

    fn complex_values(img: &Image) -> Vec<(f64, f64)> {
        img.complex_components::<f64>()
            .unwrap()
            .chunks_exact(2)
            .map(|c| (c[0], c[1]))
            .collect()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "{what}: [{i}] {x} vs {y}");
        }
    }

    // ---- pixel types ------------------------------------------------------

    #[test]
    fn forward_fft_maps_float32_to_complex_float32() {
        let img = Image::from_vec(&[4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            forward_fft(&img).unwrap().pixel_id(),
            PixelId::ComplexFloat32
        );
    }

    #[test]
    fn forward_fft_maps_float64_to_complex_float64() {
        let img = ramp(&[4]);
        assert_eq!(
            forward_fft(&img).unwrap().pixel_id(),
            PixelId::ComplexFloat64
        );
    }

    #[test]
    fn inverse_fft_maps_complex_float32_to_float32() {
        let img = Image::from_vec(&[4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let spectrum = forward_fft(&img).unwrap();
        assert_eq!(inverse_fft(&spectrum).unwrap().pixel_id(), PixelId::Float32);
    }

    #[test]
    fn inverse_fft_maps_complex_float64_to_float64() {
        let spectrum = forward_fft(&ramp(&[4])).unwrap();
        assert_eq!(inverse_fft(&spectrum).unwrap().pixel_id(), PixelId::Float64);
    }

    #[test]
    fn half_hermitian_filters_keep_the_same_pixel_type_mapping() {
        let img = Image::from_vec(&[4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (half, _) = real_to_half_hermitian_forward_fft(&img).unwrap();
        assert_eq!(half.pixel_id(), PixelId::ComplexFloat32);
        assert_eq!(
            half_hermitian_to_real_inverse_fft(&half, false)
                .unwrap()
                .pixel_id(),
            PixelId::Float32
        );
    }

    #[test]
    fn forward_fft_rejects_a_non_real_pixel_type() {
        let img = Image::new(&[4], PixelId::UInt8);
        assert!(matches!(
            forward_fft(&img),
            Err(FilterError::RequiresRealPixelType(PixelId::UInt8))
        ));
        assert!(real_to_half_hermitian_forward_fft(&img).is_err());
    }

    #[test]
    fn inverse_fft_rejects_a_non_complex_pixel_type() {
        let img = ramp(&[4]);
        assert!(inverse_fft(&img).is_err());
        assert!(half_hermitian_to_real_inverse_fft(&img, false).is_err());
    }

    // ---- analytic spectra -------------------------------------------------

    /// A delta at the origin has an all-ones spectrum, at any length.
    #[test]
    fn forward_fft_of_a_delta_is_all_ones() {
        for n in [12usize, 97, 100] {
            let mut data = vec![0.0f64; n];
            data[0] = 1.0;
            let img = Image::from_vec(&[n], data).unwrap();
            for (k, (re, im)) in complex_values(&forward_fft(&img).unwrap())
                .into_iter()
                .enumerate()
            {
                assert!((re - 1.0).abs() < 1e-12 && im.abs() < 1e-12, "n={n} k={k}");
            }
        }
    }

    /// A constant image concentrates all its energy in bin 0, with weight `N`.
    #[test]
    fn forward_fft_of_a_constant_is_a_delta_of_weight_n() {
        for n in [12usize, 97, 100] {
            let img = Image::from_vec(&[n], vec![3.0f64; n]).unwrap();
            let spectrum = complex_values(&forward_fft(&img).unwrap());
            assert!((spectrum[0].0 - 3.0 * n as f64).abs() < 1e-9, "n={n}");
            for (k, (re, im)) in spectrum.iter().enumerate().skip(1) {
                assert!(re.hypot(*im) < 1e-9, "n={n} k={k}");
            }
        }
    }

    /// `cos(2π k₀ x / N)` splits into `N/2` at bins `k₀` and `N − k₀`.
    #[test]
    fn forward_fft_of_a_single_tone_is_two_conjugate_bins() {
        let n = 100usize;
        let k0 = 7usize;
        let data: Vec<f64> = (0..n)
            .map(|x| (2.0 * std::f64::consts::PI * (k0 * x) as f64 / n as f64).cos())
            .collect();
        let spectrum = complex_values(&forward_fft(&Image::from_vec(&[n], data).unwrap()).unwrap());
        for (k, (re, im)) in spectrum.iter().enumerate() {
            let want = if k == k0 || k == n - k0 {
                n as f64 / 2.0
            } else {
                0.0
            };
            assert!((re - want).abs() < 1e-8 && im.abs() < 1e-8, "k={k}");
        }
    }

    /// Parseval on the filter surface: `Σ x² = (1/N)·Σ |X|²`.
    #[test]
    fn forward_fft_satisfies_parseval() {
        for size in [vec![97usize], vec![12, 5], vec![13, 3, 2]] {
            let img = ramp(&size);
            let total: usize = size.iter().product();
            let energy: f64 = real_values(&img).iter().map(|v| v * v).sum();
            let spectral: f64 = complex_values(&forward_fft(&img).unwrap())
                .iter()
                .map(|(re, im)| re * re + im * im)
                .sum::<f64>()
                / total as f64;
            assert!(
                (energy - spectral).abs() < 1e-9 * energy,
                "size={size:?}: {energy} vs {spectral}"
            );
        }
    }

    // ---- round trips ------------------------------------------------------

    #[test]
    fn forward_then_inverse_is_the_identity_in_one_two_and_three_dimensions() {
        for size in [
            vec![12usize],
            vec![97],
            vec![100],
            vec![12, 97],
            vec![100, 12],
            vec![12, 5, 3],
            vec![13, 3, 2],
        ] {
            let img = ramp(&size);
            let round_trip = inverse_fft(&forward_fft(&img).unwrap()).unwrap();
            assert_eq!(round_trip.size(), img.size());
            assert_close(&real_values(&round_trip), &real_values(&img), 1e-12, "f64");
        }
    }

    #[test]
    fn half_hermitian_round_trip_is_the_identity_at_even_and_odd_x() {
        for size in [
            vec![12usize],
            vec![97],
            vec![100],
            vec![13, 4],
            vec![12, 5],
            vec![97, 3, 2],
        ] {
            let img = ramp(&size);
            let (half, odd) = real_to_half_hermitian_forward_fft(&img).unwrap();
            assert_eq!(odd, size[0] % 2 != 0, "size={size:?}");
            assert_eq!(half.size()[0], size[0] / 2 + 1, "size={size:?}");
            assert_eq!(&half.size()[1..], &size[1..], "size={size:?}");

            let round_trip = half_hermitian_to_real_inverse_fft(&half, odd).unwrap();
            assert_eq!(round_trip.size(), img.size(), "size={size:?}");
            assert_close(
                &real_values(&round_trip),
                &real_values(&img),
                1e-12,
                &format!("{size:?}"),
            );
        }
    }

    /// The half spectrum is the first `N₀/2 + 1` columns of the full one.
    #[test]
    fn half_hermitian_is_the_truncated_full_spectrum() {
        for size in [vec![12usize, 5], vec![13, 4], vec![97]] {
            let img = ramp(&size);
            let full = complex_values(&forward_fft(&img).unwrap());
            let (half_img, _) = real_to_half_hermitian_forward_fft(&img).unwrap();
            let half = complex_values(&half_img);
            let keep = half_img.size()[0];
            let lines = full.len() / size[0];
            for line in 0..lines {
                for k in 0..keep {
                    let (fr, fi) = full[line * size[0] + k];
                    let (hr, hi) = half[line * keep + k];
                    assert!((fr - hr).abs() < 1e-12 && (fi - hi).abs() < 1e-12);
                }
            }
        }
    }

    /// Both inverses agree when fed the same (Hermitian) spectrum.
    #[test]
    fn the_two_inverses_agree_on_a_hermitian_spectrum() {
        for size in [vec![12usize], vec![13], vec![100, 3]] {
            let img = ramp(&size);
            let from_full = inverse_fft(&forward_fft(&img).unwrap()).unwrap();
            let (half, odd) = real_to_half_hermitian_forward_fft(&img).unwrap();
            let from_half = half_hermitian_to_real_inverse_fft(&half, odd).unwrap();
            assert_close(
                &real_values(&from_full),
                &real_values(&from_half),
                1e-12,
                &format!("{size:?}"),
            );
        }
    }

    /// The parity flag alone decides the width — a 7-column spectrum inverts to
    /// a 12-wide or a 13-wide image, and neither is an error — so the forward
    /// transform hands back the flag that restores the input width. §3.39.
    #[test]
    fn the_forward_transform_returns_the_parity_that_restores_the_width() {
        // 13 columns in, ⌊13/2⌋+1 = 7 out, and the flag says "odd".
        let odd_img = ramp(&[13]);
        let (half, odd) = real_to_half_hermitian_forward_fft(&odd_img).unwrap();
        assert_eq!(half.size(), &[7]);
        assert!(odd);
        assert_eq!(
            half_hermitian_to_real_inverse_fft(&half, odd)
                .unwrap()
                .size(),
            &[13]
        );
        // Upstream's `false` default silently narrows that same spectrum to 12.
        assert_eq!(
            half_hermitian_to_real_inverse_fft(&half, false)
                .unwrap()
                .size(),
            &[12]
        );

        // 12 columns in, ⌊12/2⌋+1 = 7 out — the same width — and the flag
        // distinguishes the two cases.
        let even_img = ramp(&[12]);
        let (half, odd) = real_to_half_hermitian_forward_fft(&even_img).unwrap();
        assert_eq!(half.size(), &[7]);
        assert!(!odd);
        assert_eq!(
            half_hermitian_to_real_inverse_fft(&half, odd)
                .unwrap()
                .size(),
            &[12]
        );
    }

    /// pocketfft's `general_c2r` reads `in[0].r` and, for an even width,
    /// `in[len/2].r`: the imaginary parts of the DC and Nyquist columns never
    /// reach the transform. Ledger §2.111.
    #[test]
    fn dc_and_nyquist_imaginary_parts_are_ignored() {
        let img = ramp(&[8]);
        let (half, _) = real_to_half_hermitian_forward_fft(&img).unwrap();
        let clean = real_values(&half_hermitian_to_real_inverse_fft(&half, false).unwrap());

        let mut poisoned = half.clone();
        {
            let comps = poisoned.complex_components_mut::<f64>().unwrap();
            comps[1] = 17.0; // im(DC)
            comps[2 * 4 + 1] = -3.5; // im(Nyquist), the 5th of 5 columns
        }
        let same = real_values(&half_hermitian_to_real_inverse_fft(&poisoned, false).unwrap());
        assert_close(&clean, &same, 1e-12, "poisoned DC/Nyquist");
    }

    /// An odd output width has no Nyquist column, so only the DC imaginary part
    /// is dropped.
    #[test]
    fn an_odd_width_drops_only_the_dc_imaginary_part() {
        let img = ramp(&[7]);
        let (half, _) = real_to_half_hermitian_forward_fft(&img).unwrap();
        let clean = real_values(&half_hermitian_to_real_inverse_fft(&half, true).unwrap());

        let mut poisoned = half.clone();
        poisoned.complex_components_mut::<f64>().unwrap()[1] = 9.0;
        let same = real_values(&half_hermitian_to_real_inverse_fft(&poisoned, true).unwrap());
        assert_close(&clean, &same, 1e-12, "poisoned DC");

        // The last column, index 3, is *not* Nyquist here: it does matter.
        let mut poisoned = half.clone();
        poisoned.complex_components_mut::<f64>().unwrap()[2 * 3 + 1] += 1.0;
        let moved = real_values(&half_hermitian_to_real_inverse_fft(&poisoned, true).unwrap());
        assert!(
            clean.iter().zip(&moved).any(|(a, b)| (a - b).abs() > 1e-6),
            "the last column of an odd-width half spectrum must be live"
        );
    }

    /// "If the input does not have Hermitian symmetry, the imaginary component
    /// is discarded" (`InverseFFTImageFilter.yaml`).
    #[test]
    fn inverse_fft_discards_the_imaginary_part_without_complaint() {
        let data = vec![PixelComplex::new(1.0f64, 5.0), PixelComplex::new(2.0, -3.0)];
        let img = Image::from_vec_complex(&[2], data).unwrap();
        // X = [1+5i, 2-3i] -> x[0] = (X0 + X1)/2 = 1.5 + 1i, x[1] = (X0 - X1)/2 = -0.5 + 4i.
        assert_close(
            &real_values(&inverse_fft(&img).unwrap()),
            &[1.5, -0.5],
            1e-12,
            "real part",
        );
    }

    // ---- geometry ---------------------------------------------------------

    #[test]
    fn transforms_keep_the_input_geometry() {
        let mut img = ramp(&[6, 4]);
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        let spectrum = forward_fft(&img).unwrap();
        assert_eq!(spectrum.spacing(), img.spacing());
        assert_eq!(spectrum.origin(), img.origin());
        assert_eq!(spectrum.direction(), img.direction());

        let (half, _) = real_to_half_hermitian_forward_fft(&img).unwrap();
        assert_eq!(half.spacing(), img.spacing());
        assert_eq!(half.origin(), img.origin());
    }

    // ---- fft_pad ----------------------------------------------------------

    #[test]
    fn fft_pad_grows_each_axis_to_an_eleven_smooth_size() {
        for (n, want) in [(13usize, 14usize), (100, 100), (97, 98), (23, 24), (1, 1)] {
            let img = ramp(&[n]);
            let padded = fft_pad(
                &img,
                ConvolutionBoundaryCondition::ZeroFluxNeumannPad,
                DEFAULT_SIZE_GREATEST_PRIME_FACTOR,
            )
            .unwrap();
            assert_eq!(padded.size(), &[want], "n={n}");
        }
    }

    #[test]
    fn fft_pad_puts_the_odd_pixel_on_the_upper_side() {
        // 13 -> 14: padSize = 1, lower = 0, upper = 1.
        let img = Image::from_vec(&[13], (0..13).map(|i| i as f64).collect()).unwrap();
        let padded = fft_pad(&img, ConvolutionBoundaryCondition::ZeroPad, 11).unwrap();
        let v = real_values(&padded);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[12], 12.0);
        assert_eq!(v[13], 0.0);
        assert_eq!(padded.origin(), img.origin());
    }

    #[test]
    fn fft_pad_splits_an_even_pad_and_shifts_the_origin() {
        // 23 -> 24: padSize = 1 -> lower 0. Use 17 -> 18: padSize = 1 too.
        // 19 -> 20: padSize = 1. Take 31 -> 32: padSize = 1. Need padSize >= 2:
        // 29 -> 30 is 1; 53 -> 54 is 1; 47 -> 48 is 1. 43 -> 44: 1. 41 -> 42: 1.
        // 89 -> 90: 1. 87 = 3*29 -> 88: padSize 1. 46 = 2*23 -> 48: padSize 2.
        let img = Image::from_vec(&[46], (0..46).map(|i| i as f64).collect()).unwrap();
        let padded = fft_pad(&img, ConvolutionBoundaryCondition::ZeroPad, 11).unwrap();
        assert_eq!(padded.size(), &[48]);
        let v = real_values(&padded);
        assert_eq!(v[0], 0.0, "the low pad pixel");
        assert_eq!(v[1], 0.0, "input pixel 0");
        assert_eq!(v[46], 45.0, "input pixel 45");
        assert_eq!(v[47], 0.0, "the high pad pixel");
        assert_eq!(padded.origin(), &[-1.0]);
    }

    #[test]
    fn fft_pad_boundary_conditions_fill_differently() {
        let img = Image::from_vec(&[46], (0..46).map(|i| i as f64).collect()).unwrap();
        let zero = real_values(&fft_pad(&img, ConvolutionBoundaryCondition::ZeroPad, 11).unwrap());
        let neumann = real_values(
            &fft_pad(&img, ConvolutionBoundaryCondition::ZeroFluxNeumannPad, 11).unwrap(),
        );
        let periodic =
            real_values(&fft_pad(&img, ConvolutionBoundaryCondition::PeriodicPad, 11).unwrap());
        assert_eq!((zero[0], zero[47]), (0.0, 0.0));
        assert_eq!((neumann[0], neumann[47]), (0.0, 45.0));
        assert_eq!((periodic[0], periodic[47]), (45.0, 0.0));
    }

    /// `SizeGreatestPrimeFactor == 1` pads to an even size, contradicting the
    /// filter's own documentation. Ledger §2.109.
    #[test]
    fn fft_pad_with_a_prime_factor_of_one_makes_the_size_even() {
        assert_eq!(
            fft_pad(&ramp(&[13, 8]), ConvolutionBoundaryCondition::ZeroPad, 1)
                .unwrap()
                .size(),
            &[14, 8]
        );
    }

    #[test]
    fn fft_pad_with_a_prime_factor_of_zero_does_nothing() {
        assert_eq!(
            fft_pad(&ramp(&[13, 8]), ConvolutionBoundaryCondition::ZeroPad, 0)
                .unwrap()
                .size(),
            &[13, 8]
        );
    }

    /// `SizeGreatestPrimeFactor == 2`, the VNL setting `FFTPadImageFilter.yaml`'s
    /// second test pins.
    #[test]
    fn fft_pad_with_a_prime_factor_of_two_reaches_a_power_of_two() {
        assert_eq!(
            fft_pad(&ramp(&[13, 100]), ConvolutionBoundaryCondition::ZeroPad, 2)
                .unwrap()
                .size(),
            &[16, 128]
        );
    }

    #[test]
    fn fft_pad_keeps_the_pixel_type() {
        let img = Image::from_vec(&[13], vec![1u8; 13]).unwrap();
        let padded = fft_pad(&img, ConvolutionBoundaryCondition::ZeroFluxNeumannPad, 11).unwrap();
        assert_eq!(padded.pixel_id(), PixelId::UInt8);
        assert_eq!(padded.size(), &[14]);
    }

    /// The composition the convolution filters use: pad to a fast size,
    /// transform, invert, and the interior is unchanged.
    #[test]
    fn fft_pad_then_round_trip_recovers_the_padded_image() {
        let img = ramp(&[13, 5]);
        let padded = fft_pad(&img, ConvolutionBoundaryCondition::ZeroFluxNeumannPad, 11).unwrap();
        assert_eq!(padded.size(), &[14, 5]);
        let back = inverse_fft(&forward_fft(&padded).unwrap()).unwrap();
        assert_close(
            &real_values(&back),
            &real_values(&padded),
            1e-12,
            "round trip",
        );
    }
}
