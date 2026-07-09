//! Deconvolution: recover an image from its convolution with a known kernel.
//!
//! Six filters, all sharing `itk::FFTConvolutionImageFilter`'s padding and
//! transform plumbing (`itkFFTConvolutionImageFilter.hxx`, ported in
//! [`crate::convolution`]):
//!
//! | filter | ITK header | parameter |
//! |---|---|---|
//! | [`inverse_deconvolution`] | itkInverseDeconvolutionImageFilter | `KernelZeroMagnitudeThreshold` |
//! | [`wiener_deconvolution`] | itkWienerDeconvolutionImageFilter | `NoiseVariance` |
//! | [`tikhonov_deconvolution`] | itkTikhonovDeconvolutionImageFilter | `RegularizationConstant` |
//! | [`landweber_deconvolution`] | itkLandweberDeconvolutionImageFilter | `Alpha`, `NumberOfIterations` |
//! | [`projected_landweber_deconvolution`] | itkProjectedLandweberDeconvolutionImageFilter | ditto |
//! | [`richardson_lucy_deconvolution`] | itkRichardsonLucyDeconvolutionImageFilter | `NumberOfIterations` |
//!
//! Every one also takes `Normalize`, `BoundaryCondition` and `OutputRegionMode`
//! with the same defaults [`crate::convolution::fft_convolution`] uses, as each
//! filter's SimpleITK yaml declares.
//!
//! # The shared pipeline
//!
//! `PrepareInputs` pads the input out to a power-of-two extent through the
//! boundary condition ([`crate::convolution::pad_input`]) and builds the
//! transfer function `H` — the kernel, upper-zero-padded and cyclically shifted
//! so its origin sits at index 0, transformed
//! ([`crate::convolution::kernel_spectrum`]). The three *spectral* filters then
//! evaluate one binary functor over `(I[k], H[k])`, invert the transform and
//! crop ([`crate::convolution::crop_output`]). The three *iterative* filters
//! keep a real-valued estimate in that padded domain, refine it once per
//! iteration against the cached `H`, and crop at the end
//! (itkIterativeDeconvolutionImageFilter.hxx:42-134).
//!
//! ITK's transforms are half-Hermitian (real input, half spectrum); this port's
//! [`crate::fft`] is a full complex DFT. Every functor here maps a
//! conjugate-symmetric spectrum to a conjugate-symmetric spectrum, so the
//! inverse transform's imaginary part is zero up to round-off and taking `.re`
//! is exact.
//!
//! # Two ITK quirks worth knowing
//!
//! **`KernelZeroMagnitudeThreshold` is compared against different quantities.**
//! [`inverse_deconvolution`] rejects a frequency when `|H| < ε`, but
//! [`tikhonov_deconvolution`] and [`wiener_deconvolution`] reject it when their
//! *denominator* — which carries `|H|²`, not `|H|` — falls below the same `ε`
//! (itkInverseDeconvolutionImageFilter.h:145-151 vs
//! itkTikhonovDeconvolutionImageFilter.h:142-148 and
//! itkWienerDeconvolutionImageFilter.h:173-178). So "Tikhonov with
//! `regularization_constant == 0` is the inverse filter" and "Wiener with
//! `noise_variance == 0` is the inverse filter" hold only where `|H| >= √ε`,
//! i.e. `|H| >= 1e-2` at the default `ε = 1e-4`. Between `1e-4` and `1e-2` the
//! inverse filter divides while the other two zero the frequency out. SimpleITK
//! exposes `KernelZeroMagnitudeThreshold` on `InverseDeconvolutionImageFilter`
//! only; Wiener and Tikhonov inherit the base class's `1.0e-4` and cannot
//! change it, so [`KERNEL_ZERO_MAGNITUDE_THRESHOLD`] is a constant here.
//!
//! **The iterative filters ignore `OutputRegionMode`.**
//! `IterativeDeconvolutionImageFilter::GenerateData` overwrites the output's
//! requested, buffered and largest-possible regions with the *input's* before
//! `PadInput` and `CropOutput` ever read them
//! (itkIterativeDeconvolutionImageFilter.hxx:110-116), discarding the `VALID`
//! region `ConvolutionImageFilterBase::GenerateOutputInformation` had
//! installed. The parameter is on their SimpleITK yamls all the same, so it is
//! on these functions too — and, faithfully, it does nothing.

use sitk_core::Image;

use crate::convolution::{
    ConvolutionBoundaryCondition, OutputRegionMode, PaddedInput, as_f64_image, crop_output,
    kernel_radius, kernel_spectrum, output_region, pad_input, prepare_kernel,
};
use crate::error::Result;
use crate::fft::{self, Complex};
use crate::image_from_f64;

/// `InverseDeconvolutionImageFilter`'s `m_KernelZeroMagnitudeThreshold` default
/// (itkInverseDeconvolutionImageFilter.hxx:30), which `WienerDeconvolutionImageFilter`
/// and `TikhonovDeconvolutionImageFilter` inherit and never override — neither
/// ITK nor SimpleITK gives their callers a way to set it.
pub const KERNEL_ZERO_MAGNITUDE_THRESHOLD: f64 = 1.0e-4;

/// `Functor::DivideOrZeroOut::m_Threshold`, `1e-5 * NumericTraits<double>::OneValue()`
/// (itkArithmeticOpsFunctors.h:164-167): Richardson-Lucy's guard against
/// dividing the blurred input by a vanishing re-blurred estimate.
const DIVIDE_OR_ZERO_OUT_THRESHOLD: f64 = 1e-5;

// ---- shared setup ---------------------------------------------------------

/// Everything the six filters agree on before any transform runs: the
/// (optionally normalized) kernel, its radius, and the output region.
struct Plan {
    kernel_values: Vec<f64>,
    kernel_size: Vec<usize>,
    radius: Vec<usize>,
    out_index: Vec<usize>,
    out_size: Vec<usize>,
}

/// Validates the kernel exactly as [`crate::convolution::fft_convolution`] does
/// — same [`crate::error::FilterError`] variants for a dimension mismatch, a
/// zero-length kernel axis, or a `Normalize`d kernel summing to zero.
fn plan(
    image: &Image,
    kernel: &Image,
    normalize: bool,
    output_region_mode: OutputRegionMode,
) -> Result<Plan> {
    let kernel_values = prepare_kernel(image, kernel, normalize)?;
    let kernel_size = kernel.size().to_vec();
    let radius = kernel_radius(&kernel_size);
    let (out_index, out_size) = output_region(image.size(), &kernel_size, output_region_mode);
    Ok(Plan {
        kernel_values,
        kernel_size,
        radius,
        out_index,
        out_size,
    })
}

/// `FFTConvolutionImageFilter::PrepareInputs` (itkFFTConvolutionImageFilter.hxx:116-126).
struct Prepared {
    padded: PaddedInput,
    /// `H`: the discrete Fourier transform of the shifted, padded kernel.
    transfer: Vec<Complex>,
}

impl Plan {
    fn prepare(
        &self,
        image: &Image,
        boundary_condition: ConvolutionBoundaryCondition,
    ) -> Result<Prepared> {
        let widened = as_f64_image(image)?;
        let padded = pad_input(
            &widened,
            &self.radius,
            &self.out_index,
            &self.out_size,
            boundary_condition,
        )?;
        let transfer = kernel_spectrum(
            &self.kernel_values,
            &self.kernel_size,
            &self.radius,
            &padded.size,
        );
        Ok(Prepared { padded, transfer })
    }

    /// `CropOutput` followed by the narrowing back to the input's pixel type.
    fn finish(&self, image: &Image, padded: &PaddedInput, estimate: &[f64]) -> Result<Image> {
        let values = crop_output(
            estimate,
            &padded.size,
            &padded.lower,
            &self.radius,
            &self.out_size,
        );
        image_from_f64(image.pixel_id(), &self.out_size, image, &values)
    }
}

/// An empty output region (`VALID` with a kernel longer than the image) short-
/// circuits before any padding: there is nothing to transform.
fn empty_output(image: &Image, out_size: &[usize]) -> Result<Image> {
    image_from_f64(image.pixel_id(), out_size, image, &[])
}

fn forward(real: &[f64], size: &[usize]) -> Vec<Complex> {
    let mut spectrum: Vec<Complex> = real.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft::transform_nd(&mut spectrum, size, false);
    spectrum
}

/// The inverse transform, of a spectrum this module's functors have kept
/// conjugate-symmetric; ITK's `HalfHermitianToRealInverseFFTImageFilter`
/// produces the same real image by construction.
fn inverse_real(spectrum: &[Complex], size: &[usize]) -> Vec<f64> {
    let mut buf = spectrum.to_vec();
    fft::transform_nd(&mut buf, size, true);
    buf.into_iter().map(|x| x.re).collect()
}

// ---- spectral filters -----------------------------------------------------

/// The three one-shot filters differ only in the binary functor a
/// `BinaryGeneratorImageFilter` evaluates over `(I[k], H[k])`.
fn spectral_deconvolution(
    image: &Image,
    kernel: &Image,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
    functor: impl Fn(Complex, Complex) -> Complex,
) -> Result<Image> {
    let plan = plan(image, kernel, normalize, output_region_mode)?;
    if plan.out_size.contains(&0) {
        return empty_output(image, &plan.out_size);
    }
    let Prepared { padded, transfer } = plan.prepare(image, boundary_condition)?;

    let mut spectrum = forward(&padded.values, &padded.size);
    for (x, &h) in spectrum.iter_mut().zip(&transfer) {
        *x = functor(*x, h);
    }
    let estimate = inverse_real(&spectrum, &padded.size);

    plan.finish(image, &padded, &estimate)
}

/// `InverseDeconvolutionImageFilter`: the direct linear inverse filter,
/// `F(ω) = G(ω) / H(ω)` wherever `|H(ω)| >= ε`, and `0` elsewhere.
///
/// `kernel_zero_magnitude_threshold` is `ε`; SimpleITK's default is
/// [`KERNEL_ZERO_MAGNITUDE_THRESHOLD`]. It bounds the amplification: a
/// frequency the kernel attenuates to `|H|` is amplified by `1 / |H|`, so the
/// threshold trades noise blow-up for the frequencies it discards outright.
///
/// Ported from itkInverseDeconvolutionImageFilter.hxx and its functor
/// (itkInverseDeconvolutionImageFilter.h:142-152).
pub fn inverse_deconvolution(
    image: &Image,
    kernel: &Image,
    kernel_zero_magnitude_threshold: f64,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    spectral_deconvolution(
        image,
        kernel,
        normalize,
        boundary_condition,
        output_region_mode,
        |i, h| {
            if h.abs() >= kernel_zero_magnitude_threshold {
                i / h
            } else {
                Complex::default()
            }
        },
    )
}

/// `WienerDeconvolutionImageFilter`: the inverse filter damped by the
/// signal-to-noise ratio, `W(ω) = H*(ω) / (|H(ω)|² + P_n / (P_f - P_n))`.
///
/// `noise_variance` is `P_n`, the (constant) noise power spectral density;
/// `P_f = |G(ω)|²` is the blurred input's power spectral density, from which
/// ITK subtracts the noise to estimate the signal's. SimpleITK's default is
/// `0.0`, which — where `|H| >= 1e-2`, see the module docs — reduces to
/// [`inverse_deconvolution`].
///
/// Ported from itkWienerDeconvolutionImageFilter.hxx and its functor
/// (itkWienerDeconvolutionImageFilter.h:163-181).
///
/// # Divergence
///
/// ITK computes `P_n / (P_f - P_n)` in `std::complex<double>` even though both
/// operands are real. At the exact `P_f == P_n` frequencies (`|G(ω)|²` equal to
/// the noise variance to the last bit) libstdc++'s complex division yields
/// `(inf, nan)`, and the functor propagates a NaN into the output. This port
/// does that division in `f64` — the imaginary parts are identically zero — so
/// the denominator becomes `±inf`, clears the magnitude threshold, and the
/// frequency is scaled to zero. Elsewhere the two agree bit for bit.
pub fn wiener_deconvolution(
    image: &Image,
    kernel: &Image,
    noise_variance: f64,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    spectral_deconvolution(
        image,
        kernel,
        normalize,
        boundary_condition,
        output_region_mode,
        |i, h| {
            let pn = noise_variance;
            // The power spectral density of the estimate: that of the blurred
            // input, less the noise's.
            let pf = i.norm();
            let denominator = h.norm() + pn / (pf - pn);
            if denominator.abs() >= KERNEL_ZERO_MAGNITUDE_THRESHOLD {
                i * h.conj().scale(1.0 / denominator)
            } else {
                Complex::default()
            }
        },
    )
}

/// `TikhonovDeconvolutionImageFilter`: the inverse filter with a regularization
/// term in the denominator, `H*(ω) / (|H(ω)|² + μ)`.
///
/// `regularization_constant` is `μ >= 0`. Larger values suppress noise at the
/// cost of approximation error. SimpleITK's default is `0.0`, which — where
/// `|H| >= 1e-2`, see the module docs — reduces to [`inverse_deconvolution`].
///
/// Ported from itkTikhonovDeconvolutionImageFilter.hxx and its functor
/// (itkTikhonovDeconvolutionImageFilter.h:139-151).
pub fn tikhonov_deconvolution(
    image: &Image,
    kernel: &Image,
    regularization_constant: f64,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    spectral_deconvolution(
        image,
        kernel,
        normalize,
        boundary_condition,
        output_region_mode,
        |i, h| {
            // Real, and non-negative for the non-negative `μ` the filter
            // documents, so ITK compares it to the threshold unsigned.
            let denominator = h.norm() + regularization_constant;
            if denominator >= KERNEL_ZERO_MAGNITUDE_THRESHOLD {
                i * h.conj().scale(1.0 / denominator)
            } else {
                Complex::default()
            }
        },
    )
}

// ---- iterative filters ----------------------------------------------------

/// `IterativeDeconvolutionImageFilter::GenerateData`
/// (itkIterativeDeconvolutionImageFilter.hxx:102-134): seed the estimate with
/// the padded input (`Initialize`, ibid. 44-67), run `iteration` exactly
/// `number_of_iterations` times over the padded real estimate, then crop
/// (`Finish`, ibid. 69-79).
///
/// `number_of_iterations == 0` therefore returns the input unchanged, cropped
/// out of its own padding.
fn iterative_deconvolution(
    image: &Image,
    kernel: &Image,
    number_of_iterations: u32,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    mut iteration: impl FnMut(&Prepared, Vec<f64>) -> Vec<f64>,
) -> Result<Image> {
    // `OutputRegionMode` cannot reach here: `GenerateData` resets the output
    // region to the input's before `PadInput` reads it (see the module docs).
    let plan = plan(image, kernel, normalize, OutputRegionMode::Same)?;
    if plan.out_size.contains(&0) {
        return empty_output(image, &plan.out_size);
    }
    let prepared = plan.prepare(image, boundary_condition)?;

    let mut estimate = prepared.padded.values.clone();
    for _ in 0..number_of_iterations {
        estimate = iteration(&prepared, estimate);
    }

    plan.finish(image, &prepared.padded, &estimate)
}

/// One Landweber step, `Functor::LandweberMethod`
/// (itkLandweberDeconvolutionImageFilter.h:47-52):
/// `x̂ₖ₊₁ = α H* G + (1 - α |H|²) x̂ₖ`, all in the Fourier domain.
fn landweber_step(
    prepared: &Prepared,
    input_spectrum: &[Complex],
    alpha: f64,
    estimate: Vec<f64>,
) -> Vec<f64> {
    let size = &prepared.padded.size;
    let mut spectrum = forward(&estimate, size);
    for ((x, &h), &g) in spectrum
        .iter_mut()
        .zip(&prepared.transfer)
        .zip(input_spectrum)
    {
        *x = h.conj().scale(alpha) * g + x.scale(1.0 - alpha * h.norm());
    }
    inverse_real(&spectrum, size)
}

/// `ThresholdImageFilter::ThresholdBelow(0)` as
/// `ProjectedIterativeDeconvolutionImageFilter` configures it
/// (itkProjectedIterativeDeconvolutionImageFilter.hxx:39-42): a pixel survives
/// only if `0 <= v <= DBL_MAX`, otherwise it becomes the outside value, `0`
/// (itkThresholdImageFilter.hxx:115-124). Both `+inf` and NaN fail that test.
fn project_to_nonnegative(estimate: &mut [f64]) {
    for v in estimate.iter_mut() {
        if !(*v >= 0.0 && *v <= f64::MAX) {
            *v = 0.0;
        }
    }
}

/// `LandweberDeconvolutionImageFilter`: gradient descent on `‖f ⊗ h - g‖²`
/// with a fixed step, converging to the least-squares estimate of the unblurred
/// image.
///
/// `alpha` is the relaxation factor (SimpleITK default `0.1`);
/// `number_of_iterations` defaults to `1`. Intermediate estimates may go
/// negative — see [`projected_landweber_deconvolution`] for the constrained
/// variant.
///
/// `output_region_mode` is accepted for parity with the SimpleITK yaml and
/// ignored, as ITK ignores it; the output always covers the whole input.
///
/// Ported from itkLandweberDeconvolutionImageFilter.hxx.
pub fn landweber_deconvolution(
    image: &Image,
    kernel: &Image,
    alpha: f64,
    number_of_iterations: u32,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    let _ = output_region_mode;

    // `Initialize` transforms the padded input once and reuses it every
    // iteration (itkLandweberDeconvolutionImageFilter.hxx:48).
    let mut input_spectrum = Vec::new();
    iterative_deconvolution(
        image,
        kernel,
        number_of_iterations,
        normalize,
        boundary_condition,
        |prepared, estimate| {
            if input_spectrum.is_empty() {
                input_spectrum = forward(&prepared.padded.values, &prepared.padded.size);
            }
            landweber_step(prepared, &input_spectrum, alpha, estimate)
        },
    )
}

/// `ProjectedLandweberDeconvolutionImageFilter`: [`landweber_deconvolution`]
/// with every intermediate estimate's negative pixels projected to zero, for
/// signals known to be non-negative (photon counts, say).
///
/// The projection runs *after each iteration*, so it also shapes the estimate
/// the next iteration starts from — running one iteration then clamping is not
/// the same thing.
///
/// `output_region_mode` is accepted for parity with the SimpleITK yaml and
/// ignored, as ITK ignores it; the output always covers the whole input.
///
/// Ported from itkProjectedLandweberDeconvolutionImageFilter.hxx via its
/// `ProjectedIterativeDeconvolutionImageFilter` mix-in.
pub fn projected_landweber_deconvolution(
    image: &Image,
    kernel: &Image,
    alpha: f64,
    number_of_iterations: u32,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    let _ = output_region_mode;

    let mut input_spectrum = Vec::new();
    iterative_deconvolution(
        image,
        kernel,
        number_of_iterations,
        normalize,
        boundary_condition,
        |prepared, estimate| {
            if input_spectrum.is_empty() {
                input_spectrum = forward(&prepared.padded.values, &prepared.padded.size);
            }
            let mut next = landweber_step(prepared, &input_spectrum, alpha, estimate);
            project_to_nonnegative(&mut next);
            next
        },
    )
}

/// `RichardsonLucyDeconvolutionImageFilter`: the multiplicative
/// expectation-maximization update for Poisson noise,
/// `x̂ₖ₊₁ = x̂ₖ · (h ⋆ (g / (x̂ₖ ⊗ h)))`, where `⋆` correlates with the flipped
/// kernel (a multiplication by `H*` in the Fourier domain).
///
/// `number_of_iterations` defaults to `1`. The update is multiplicative, so a
/// non-negative input and a non-negative kernel keep every estimate
/// non-negative (up to transform round-off) and never grow its support. With a
/// `Normalize`d kernel the total intensity of the padded estimate is conserved.
///
/// The division `g / (x̂ₖ ⊗ h)` is ITK's `DivideOrZeroOutImageFilter`: a
/// denominator below `1e-5` yields `0` rather than a quotient
/// (itkArithmeticOpsFunctors.h:181-189). Note that this is a *signed*
/// comparison, so a negative re-blurred estimate also zeroes the quotient.
///
/// `output_region_mode` is accepted for parity with the SimpleITK yaml and
/// ignored, as ITK ignores it; the output always covers the whole input.
///
/// Ported from itkRichardsonLucyDeconvolutionImageFilter.hxx.
pub fn richardson_lucy_deconvolution(
    image: &Image,
    kernel: &Image,
    number_of_iterations: u32,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    let _ = output_region_mode;

    iterative_deconvolution(
        image,
        kernel,
        number_of_iterations,
        normalize,
        boundary_condition,
        |prepared, estimate| {
            let size = &prepared.padded.size;

            // `x̂ₖ ⊗ h`, the current estimate re-blurred.
            let mut spectrum = forward(&estimate, size);
            for (x, &h) in spectrum.iter_mut().zip(&prepared.transfer) {
                *x = *x * h;
            }
            let blurred = inverse_real(&spectrum, size);

            // `g / (x̂ₖ ⊗ h)`, or zero where the denominator has collapsed.
            let quotient: Vec<f64> = prepared
                .padded
                .values
                .iter()
                .zip(&blurred)
                .map(|(&numerator, &denominator)| {
                    if denominator < DIVIDE_OR_ZERO_OUT_THRESHOLD {
                        0.0
                    } else {
                        numerator / denominator
                    }
                })
                .collect();

            // `h ⋆ quotient`, the correlation with the flipped kernel.
            let mut spectrum = forward(&quotient, size);
            for (x, &h) in spectrum.iter_mut().zip(&prepared.transfer) {
                *x = *x * h.conj();
            }
            let correction = inverse_real(&spectrum, size);

            estimate
                .iter()
                .zip(&correction)
                .map(|(&x, &c)| x * c)
                .collect()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolution::fft_convolution;
    use crate::error::FilterError;

    use ConvolutionBoundaryCondition::{PeriodicPad, ZeroFluxNeumannPad, ZeroPad};
    use OutputRegionMode::{Same, Valid};

    /// Slack for quantities the exact arithmetic makes zero but the transforms
    /// only make small: a sum that should cancel, a pixel that should sit on
    /// the non-negativity boundary.
    const ROUND_OFF: f64 = 1e-12;

    fn img(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn values(image: &Image) -> Vec<f64> {
        image.scalar_slice::<f64>().unwrap().to_vec()
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "index {i}: {got:?} vs {want:?}");
        }
    }

    /// A kernel whose transfer function never comes near zero: `[1, 3, 1] / 5`
    /// has `H[k] = (3 + 2 cos ω) / 5 ∈ [0.2, 1]`, so no frequency is thresholded
    /// away at the default `ε` and no filter here has to discard information.
    fn smooth_kernel() -> Image {
        img(&[3], vec![1.0, 3.0, 1.0])
    }

    /// An 8-pixel signal whose support keeps a `radius == 1` kernel's reach off
    /// the border. Convolving it under [`ZeroPad`] therefore produces the exact
    /// linear convolution, and the padded arrays the deconvolution filters build
    /// are the circular convolution of a padded original with the kernel — so a
    /// well-conditioned kernel inverts it to the last few bits.
    fn compact_signal() -> Image {
        img(&[8], vec![0.0, 0.0, 4.0, 9.0, 2.0, 7.0, 0.0, 0.0])
    }

    /// `f ⊗ h` under the same padding the deconvolution filters will undo.
    fn blur(signal: &Image, kernel: &Image) -> Image {
        fft_convolution(signal, kernel, true, ZeroPad, Same).unwrap()
    }

    /// A kernel of a single pixel `c`: its transfer function is exactly `c` at
    /// every frequency, which pins the magnitude thresholds without any FFT
    /// arithmetic getting in the way.
    fn delta_kernel(c: f64) -> Image {
        img(&[1], vec![c])
    }

    // ---- recovering a known convolution ------------------------------------

    #[test]
    fn inverse_deconvolution_recovers_a_known_convolution() {
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = blur(&original, &kernel);
        assert!(
            values(&blurred) != values(&original),
            "the fixture must actually be blurred"
        );

        let recovered = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&recovered), &values(&original), 1e-9);
    }

    #[test]
    fn tikhonov_and_wiener_recover_a_known_convolution_at_their_defaults() {
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = blur(&original, &kernel);

        let tikhonov = tikhonov_deconvolution(&blurred, &kernel, 0.0, true, ZeroPad, Same).unwrap();
        assert_close(&values(&tikhonov), &values(&original), 1e-9);

        let wiener = wiener_deconvolution(&blurred, &kernel, 0.0, true, ZeroPad, Same).unwrap();
        assert_close(&values(&wiener), &values(&original), 1e-9);
    }

    #[test]
    fn landweber_and_richardson_lucy_approach_the_original() {
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = blur(&original, &kernel);

        let landweber =
            landweber_deconvolution(&blurred, &kernel, 1.0, 400, true, ZeroPad, Same).unwrap();
        assert_close(&values(&landweber), &values(&original), 1e-6);

        let rl =
            richardson_lucy_deconvolution(&blurred, &kernel, 400, true, ZeroPad, Same).unwrap();
        assert_close(&values(&rl), &values(&original), 1e-9);
    }

    // ---- inverse: amplification and the magnitude threshold -----------------

    #[test]
    fn inverse_deconvolution_amplifies_by_one_over_the_kernel_magnitude() {
        // `H[k] == 1e-3` at every frequency, comfortably above the threshold,
        // so every pixel is divided by it.
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let out = inverse_deconvolution(
            &image,
            &delta_kernel(1e-3),
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&out), &[1000.0, 2000.0, 3000.0, 4000.0], 1e-6);
    }

    #[test]
    fn inverse_deconvolution_zeroes_frequencies_below_the_threshold() {
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);

        // `|H| == 1e-5 < 1e-4`: every frequency is discarded, output is zero.
        let zeroed = inverse_deconvolution(
            &image,
            &delta_kernel(1e-5),
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_eq!(values(&zeroed), vec![0.0; 4]);

        // Drop the threshold and the same kernel divides instead.
        let divided =
            inverse_deconvolution(&image, &delta_kernel(1e-5), 0.0, false, ZeroPad, Same).unwrap();
        assert_close(&values(&divided), &[1e5, 2e5, 3e5, 4e5], 1e-3);
    }

    #[test]
    fn inverse_deconvolution_threshold_is_inclusive() {
        // `|H| >= ε` keeps the frequency; `|H| == ε` exactly is therefore kept.
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let kept = inverse_deconvolution(
            &image,
            &delta_kernel(KERNEL_ZERO_MAGNITUDE_THRESHOLD),
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&kept), &[1e4, 2e4, 3e4, 4e4], 1e-4);
    }

    // ---- Wiener / Tikhonov reduce to the inverse filter ---------------------

    #[test]
    fn wiener_reduces_to_the_inverse_filter_as_the_noise_variance_vanishes() {
        let blurred = blur(&compact_signal(), &smooth_kernel());
        let kernel = smooth_kernel();

        let inverse = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Same,
        )
        .unwrap();
        let wiener = wiener_deconvolution(&blurred, &kernel, 0.0, true, ZeroPad, Same).unwrap();
        assert_close(&values(&wiener), &values(&inverse), 1e-9);
    }

    #[test]
    fn tikhonov_reduces_to_the_inverse_filter_as_the_regularization_vanishes() {
        let blurred = blur(&compact_signal(), &smooth_kernel());
        let kernel = smooth_kernel();

        let inverse = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Same,
        )
        .unwrap();
        let tikhonov = tikhonov_deconvolution(&blurred, &kernel, 0.0, true, ZeroPad, Same).unwrap();
        assert_close(&values(&tikhonov), &values(&inverse), 1e-9);
    }

    #[test]
    fn wiener_and_tikhonov_threshold_the_squared_magnitude_where_inverse_thresholds_the_magnitude()
    {
        // `|H| == 1e-3`: above `ε == 1e-4`, so the inverse filter divides. But
        // `|H|² == 1e-6` is below `ε`, so both regularized filters — at the
        // parameter values that supposedly make them the inverse filter —
        // discard every frequency instead. This is the ITK quirk, not an
        // approximation: see the module docs.
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = delta_kernel(1e-3);

        let inverse = inverse_deconvolution(
            &image,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&inverse), &[1000.0, 2000.0, 3000.0, 4000.0], 1e-6);

        let tikhonov = tikhonov_deconvolution(&image, &kernel, 0.0, false, ZeroPad, Same).unwrap();
        assert_eq!(values(&tikhonov), vec![0.0; 4]);

        let wiener = wiener_deconvolution(&image, &kernel, 0.0, false, ZeroPad, Same).unwrap();
        assert_eq!(values(&wiener), vec![0.0; 4]);
    }

    #[test]
    fn tikhonov_regularization_rescues_a_frequency_the_squared_magnitude_lost() {
        // The same `|H|² == 1e-6` denominator, lifted over `ε` by `μ`.
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = delta_kernel(1e-3);
        let mu = 1e-3;

        let out = tikhonov_deconvolution(&image, &kernel, mu, false, ZeroPad, Same).unwrap();
        // `I * conj(H) / (|H|² + μ)` with a real `H`.
        let gain = 1e-3 / (1e-6 + mu);
        assert_close(
            &values(&out),
            &[gain, 2.0 * gain, 3.0 * gain, 4.0 * gain],
            1e-9,
        );
    }

    // ---- Wiener: the noise-variance arithmetic ------------------------------

    #[test]
    fn wiener_denominator_matches_the_hand_computed_functor() {
        // One pixel, one-pixel unit kernel: the transform is the identity, so
        // `I == 2`, `H == 1`, `Pf == 4`, and for `Pn == 1` the functor is
        // `I / (|H|² + Pn/(Pf - Pn)) = 2 / (1 + 1/3) = 1.5`.
        let image = img(&[1], vec![2.0]);
        let out = wiener_deconvolution(&image, &delta_kernel(1.0), 1.0, false, ZeroPad, Same);
        assert_close(&values(&out.unwrap()), &[1.5], 1e-12);
    }

    #[test]
    fn wiener_zeroes_a_frequency_whose_power_equals_the_noise_variance() {
        // `Pf == Pn` makes `Pn / (Pf - Pn)` infinite; the denominator clears the
        // magnitude threshold and scales the frequency to nothing. ITK's complex
        // division yields NaN here instead — see this function's docs.
        let image = img(&[1], vec![2.0]);
        let out = wiener_deconvolution(&image, &delta_kernel(1.0), 4.0, false, ZeroPad, Same);
        assert_eq!(values(&out.unwrap()), vec![0.0]);
    }

    #[test]
    fn wiener_damps_more_as_the_noise_variance_grows() {
        let blurred = blur(&compact_signal(), &smooth_kernel());
        let kernel = smooth_kernel();

        let energy = |variance: f64| {
            let out = wiener_deconvolution(&blurred, &kernel, variance, true, ZeroPad, Same);
            values(&out.unwrap()).iter().map(|v| v * v).sum::<f64>()
        };
        let (none, some, lots) = (energy(0.0), energy(1.0), energy(10.0));
        assert!(none > some, "{none} should exceed {some}");
        assert!(some > lots, "{some} should exceed {lots}");
    }

    // ---- Landweber ----------------------------------------------------------

    #[test]
    fn landweber_converges_toward_the_inverse_solution() {
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = blur(&original, &kernel);

        let error = |iterations: u32| {
            let out =
                landweber_deconvolution(&blurred, &kernel, 1.0, iterations, true, ZeroPad, Same)
                    .unwrap();
            values(&out)
                .iter()
                .zip(values(&original))
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f64, f64::max)
        };

        // The slowest mode decays as `(1 - α |H|²)ⁿ`, and `smooth_kernel`'s
        // smallest `|H|²` is `0.04`, so `α == 1` contracts it by `0.96` per
        // iteration — a few hundred iterations to reach the round-off floor.
        let (e1, e5, e20, e100, e400) = (error(1), error(5), error(20), error(100), error(400));
        assert!(e1 > e5, "{e1} should exceed {e5}");
        assert!(e5 > e20, "{e5} should exceed {e20}");
        assert!(e20 > e100, "{e20} should exceed {e100}");
        assert!(e100 > e400, "{e100} should exceed {e400}");
        assert!(e400 < 1e-6, "converged error {e400} is too large");
    }

    #[test]
    fn a_smaller_alpha_converges_more_slowly() {
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = blur(&original, &kernel);

        let error = |alpha: f64| {
            let out =
                landweber_deconvolution(&blurred, &kernel, alpha, 20, true, ZeroPad, Same).unwrap();
            values(&out)
                .iter()
                .zip(values(&original))
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f64, f64::max)
        };
        let (slow, fast) = (error(0.1), error(1.0));
        assert!(slow > fast, "alpha=0.1 error {slow} should exceed {fast}");
    }

    #[test]
    fn landweber_with_alpha_one_and_a_unit_kernel_reproduces_the_input() {
        // `H == 1` makes the step `x̂ₖ₊₁ = G`, so the estimate is the input from
        // the first iteration onwards.
        let image = img(&[4], vec![1.0, -2.0, 3.0, -4.0]);
        for iterations in [1u32, 2, 7] {
            let out = landweber_deconvolution(
                &image,
                &delta_kernel(1.0),
                1.0,
                iterations,
                false,
                ZeroPad,
                Same,
            )
            .unwrap();
            assert_close(&values(&out), &[1.0, -2.0, 3.0, -4.0], 1e-9);
        }
    }

    // ---- projected Landweber ------------------------------------------------

    #[test]
    fn the_projected_variant_clamps_negative_pixels_each_iteration() {
        // With `H == 1` and `alpha == 1` the plain filter reproduces the signed
        // input exactly (see above), so the only difference the projection can
        // make is the clamp itself.
        let image = img(&[4], vec![1.0, -2.0, 3.0, -4.0]);
        let kernel = delta_kernel(1.0);

        let plain = landweber_deconvolution(&image, &kernel, 1.0, 3, false, ZeroPad, Same).unwrap();
        assert_close(&values(&plain), &[1.0, -2.0, 3.0, -4.0], 1e-9);

        let projected =
            projected_landweber_deconvolution(&image, &kernel, 1.0, 3, false, ZeroPad, Same)
                .unwrap();
        assert_close(&values(&projected), &[1.0, 0.0, 3.0, 0.0], 1e-9);
    }

    #[test]
    fn the_projected_variant_is_non_negative_where_the_plain_one_undershoots() {
        // Landweber's additive step drives the border of this non-negative
        // fixture below zero within two iterations. The projection is the last
        // thing each iteration does, so the estimate it hands to `Finish` — and
        // therefore every output pixel — is non-negative exactly, not just up
        // to round-off.
        let kernel = smooth_kernel();
        let blurred = blur(&compact_signal(), &kernel);

        let plain =
            landweber_deconvolution(&blurred, &kernel, 0.1, 2, true, ZeroPad, Same).unwrap();
        let undershoot = values(&plain).iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            undershoot < -1e-6,
            "the fixture must actually undershoot, got {undershoot}"
        );

        let projected =
            projected_landweber_deconvolution(&blurred, &kernel, 0.1, 2, true, ZeroPad, Same)
                .unwrap();
        for (i, &v) in values(&projected).iter().enumerate() {
            assert!(v >= 0.0, "pixel {i} went to {v}");
        }
    }

    #[test]
    fn projection_feeds_the_next_iteration_rather_than_only_the_output() {
        // Clamping once at the end would leave `x̂₂` built from a negative `x̂₁`;
        // clamping every iteration does not. With `alpha == 0.5` and `H == 1`,
        // `x̂ₖ₊₁ = 0.5 G + 0.5 x̂ₖ`, so a negative pixel of `G` recovers toward
        // `G` under the plain filter but is pinned at `0.5 * G < 0 → 0` under
        // the projected one at every step.
        let image = img(&[2], vec![-4.0, 8.0]);
        let kernel = delta_kernel(1.0);

        let plain = landweber_deconvolution(&image, &kernel, 0.5, 2, false, ZeroPad, Same).unwrap();
        assert_close(&values(&plain), &[-4.0, 8.0], 1e-9);

        // x̂₁ = 0.5*(-4) + 0.5*(-4) = -4 → 0; x̂₂ = 0.5*(-4) + 0.5*0 = -2 → 0.
        let projected =
            projected_landweber_deconvolution(&image, &kernel, 0.5, 2, false, ZeroPad, Same)
                .unwrap();
        assert_close(&values(&projected), &[0.0, 8.0], 1e-9);
    }

    // ---- Richardson-Lucy ----------------------------------------------------

    #[test]
    fn richardson_lucy_with_a_unit_kernel_is_the_identity() {
        // `H == 1` makes the re-blurred estimate the estimate itself, so the
        // quotient is `g / x̂ₖ` and the multiplicative update lands back on `g`.
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        for iterations in [1u32, 2, 5] {
            let out = richardson_lucy_deconvolution(
                &image,
                &delta_kernel(1.0),
                iterations,
                false,
                ZeroPad,
                Same,
            )
            .unwrap();
            assert_close(&values(&out), &[1.0, 2.0, 3.0, 4.0], 1e-9);
        }
    }

    #[test]
    fn richardson_lucy_zeroes_a_pixel_whose_denominator_falls_below_1e_minus_5() {
        // With a unit kernel the denominator *is* the pixel, so this pins
        // `DivideOrZeroOut`'s `1e-5` threshold directly: `9e-6` collapses to
        // zero, `2e-5` divides and survives.
        let image = img(&[4], vec![1.0, 9e-6, 2e-5, 4.0]);
        let out =
            richardson_lucy_deconvolution(&image, &delta_kernel(1.0), 1, false, ZeroPad, Same)
                .unwrap();
        assert_close(&values(&out), &[1.0, 0.0, 2e-5, 4.0], 1e-9);
    }

    #[test]
    fn richardson_lucy_preserves_non_negativity() {
        // `blur` leaves round-off-sized negatives outside the signal's support;
        // the multiplicative update must not amplify them into real ones.
        let blurred = blur(&compact_signal(), &smooth_kernel());
        let kernel = smooth_kernel();
        for (i, &v) in values(&blurred).iter().enumerate() {
            assert!(v >= -ROUND_OFF, "fixture pixel {i} is {v}");
        }

        for iterations in [1u32, 3, 10, 40] {
            let out =
                richardson_lucy_deconvolution(&blurred, &kernel, iterations, true, ZeroPad, Same)
                    .unwrap();
            for (i, &v) in values(&out).iter().enumerate() {
                assert!(
                    v >= -ROUND_OFF,
                    "iteration {iterations}, pixel {i} went to {v}"
                );
            }
        }
    }

    #[test]
    fn richardson_lucy_conserves_total_intensity_with_a_normalized_kernel() {
        // `Σⱼ h[i-j] == 1` makes the correlation's row sums unity, so
        // `Σ x̂ₖ₊₁ = Σ x̂ₖ · (hᵀ r) = Σ r · (h x̂ₖ) = Σ g` over the padded domain.
        // The multiplicative update never grows the support, and the fixture's
        // support sits strictly inside the crop, so the cropped sums agree too.
        let blurred = blur(&compact_signal(), &smooth_kernel());
        let kernel = smooth_kernel();
        let total: f64 = values(&blurred).iter().sum();

        for iterations in [1u32, 2, 5, 25] {
            let out =
                richardson_lucy_deconvolution(&blurred, &kernel, iterations, true, ZeroPad, Same)
                    .unwrap();
            let got: f64 = values(&out).iter().sum();
            assert!(
                (got - total).abs() < 1e-6,
                "iteration {iterations}: total {got} drifted from {total}"
            );
        }
    }

    // ---- iteration-count semantics ------------------------------------------

    #[test]
    fn zero_iterations_returns_the_input_unchanged() {
        // `GenerateData`'s loop never runs, so `Finish` crops the padded input
        // straight back out (itkIterativeDeconvolutionImageFilter.hxx:122-133).
        let image = compact_signal();
        let kernel = smooth_kernel();

        for boundary in [ZeroPad, ZeroFluxNeumannPad, PeriodicPad] {
            let landweber =
                landweber_deconvolution(&image, &kernel, 0.1, 0, true, boundary, Same).unwrap();
            assert_close(&values(&landweber), &values(&image), 1e-12);

            let projected =
                projected_landweber_deconvolution(&image, &kernel, 0.1, 0, true, boundary, Same)
                    .unwrap();
            assert_close(&values(&projected), &values(&image), 1e-12);

            let rl =
                richardson_lucy_deconvolution(&image, &kernel, 0, true, boundary, Same).unwrap();
            assert_close(&values(&rl), &values(&image), 1e-12);
        }
    }

    #[test]
    fn zero_iterations_does_not_project_negative_pixels() {
        // The projection lives in `Iteration()`, not `Initialize()`.
        let image = img(&[4], vec![1.0, -2.0, 3.0, -4.0]);
        let out = projected_landweber_deconvolution(
            &image,
            &delta_kernel(1.0),
            0.1,
            0,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&out), &[1.0, -2.0, 3.0, -4.0], 1e-12);
    }

    // ---- output region ------------------------------------------------------

    #[test]
    fn valid_mode_crops_the_spectral_filters() {
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = blur(&original, &kernel);

        let out = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Valid,
        )
        .unwrap();
        assert_eq!(out.size(), &[6]);
        assert_close(&values(&out), &values(&original)[1..7], 1e-9);
    }

    #[test]
    fn valid_mode_leaves_an_empty_region_for_an_oversized_kernel() {
        let image = img(&[3], vec![1.0, 2.0, 3.0]);
        let kernel = img(&[5], vec![1.0; 5]);
        let out = inverse_deconvolution(
            &image,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Valid,
        )
        .unwrap();
        assert_eq!(out.size(), &[0]);
        assert_eq!(out.number_of_pixels(), 0);
    }

    #[test]
    fn the_iterative_filters_ignore_the_output_region_mode() {
        // `IterativeDeconvolutionImageFilter::GenerateData` overwrites the
        // output region with the input's before anything reads it, so `VALID`
        // is indistinguishable from `SAME` — see the module docs.
        let image = compact_signal();
        let kernel = smooth_kernel();

        for mode in [Same, Valid] {
            let landweber =
                landweber_deconvolution(&image, &kernel, 0.1, 3, true, ZeroPad, mode).unwrap();
            assert_eq!(landweber.size(), image.size());

            let projected =
                projected_landweber_deconvolution(&image, &kernel, 0.1, 3, true, ZeroPad, mode)
                    .unwrap();
            assert_eq!(projected.size(), image.size());

            let rl =
                richardson_lucy_deconvolution(&image, &kernel, 3, true, ZeroPad, mode).unwrap();
            assert_eq!(rl.size(), image.size());
        }

        let same = landweber_deconvolution(&image, &kernel, 0.1, 3, true, ZeroPad, Same).unwrap();
        let valid = landweber_deconvolution(&image, &kernel, 0.1, 3, true, ZeroPad, Valid).unwrap();
        assert_eq!(values(&valid), values(&same));
    }

    // ---- boundary condition and normalize ----------------------------------

    #[test]
    fn the_boundary_condition_reaches_the_padded_input() {
        let image = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = smooth_kernel();
        let out = |bc| {
            values(
                &inverse_deconvolution(
                    &image,
                    &kernel,
                    KERNEL_ZERO_MAGNITUDE_THRESHOLD,
                    true,
                    bc,
                    Same,
                )
                .unwrap(),
            )
        };
        let (zero, flux, wrap) = (out(ZeroPad), out(ZeroFluxNeumannPad), out(PeriodicPad));
        assert_ne!(zero, flux);
        assert_ne!(zero, wrap);
        assert_ne!(flux, wrap);
    }

    #[test]
    fn normalize_divides_the_kernel_by_its_own_sum() {
        // The un-normalized `[1, 3, 1]` blurs *and* scales by 5; the inverse
        // filter told to normalize therefore recovers `5 * original`.
        let original = compact_signal();
        let kernel = smooth_kernel();
        let blurred = fft_convolution(&original, &kernel, false, ZeroPad, Same).unwrap();

        let recovered = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Same,
        )
        .unwrap();
        let want: Vec<f64> = values(&original).iter().map(|v| v * 5.0).collect();
        assert_close(&values(&recovered), &want, 1e-9);

        let exact = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&exact), &values(&original), 1e-9);
    }

    // ---- pixel type and geometry -------------------------------------------

    #[test]
    fn output_keeps_the_input_pixel_type_and_geometry() {
        let mut image = Image::from_vec(&[3, 2], vec![10u8, 20, 30, 40, 50, 60]).unwrap();
        image.set_spacing(&[0.5, 2.0]).unwrap();
        image.set_origin(&[7.0, -1.0]).unwrap();
        let kernel = img(&[1, 1], vec![1.0]);

        let out = inverse_deconvolution(
            &image,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            false,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_eq!(out.pixel_id(), sitk_core::PixelId::UInt8);
        assert_eq!(out.spacing(), image.spacing());
        assert_eq!(out.origin(), image.origin());
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[10, 20, 30, 40, 50, 60]);
    }

    // ---- error paths --------------------------------------------------------

    #[test]
    fn a_kernel_of_the_wrong_dimension_is_rejected_by_every_filter() {
        let image = img(&[3, 3], vec![1.0; 9]);
        let kernel = img(&[3], vec![1.0; 3]);
        let want = Err(FilterError::KernelDimensionMismatch {
            image: 2,
            kernel: 1,
        });

        assert_eq!(
            inverse_deconvolution(&image, &kernel, 1e-4, false, ZeroPad, Same),
            want
        );
        assert_eq!(
            wiener_deconvolution(&image, &kernel, 0.0, false, ZeroPad, Same),
            want
        );
        assert_eq!(
            tikhonov_deconvolution(&image, &kernel, 0.0, false, ZeroPad, Same),
            want
        );
        assert_eq!(
            landweber_deconvolution(&image, &kernel, 0.1, 1, false, ZeroPad, Same),
            want
        );
        assert_eq!(
            projected_landweber_deconvolution(&image, &kernel, 0.1, 1, false, ZeroPad, Same),
            want
        );
        assert_eq!(
            richardson_lucy_deconvolution(&image, &kernel, 1, false, ZeroPad, Same),
            want
        );
    }

    #[test]
    fn a_kernel_with_a_zero_length_axis_is_rejected() {
        let image = img(&[3], vec![1.0, 2.0, 3.0]);
        let kernel = img(&[0], vec![]);
        assert_eq!(
            inverse_deconvolution(&image, &kernel, 1e-4, false, ZeroPad, Same),
            Err(FilterError::EmptyKernel(vec![0]))
        );
        assert_eq!(
            richardson_lucy_deconvolution(&image, &kernel, 1, false, ZeroPad, Same),
            Err(FilterError::EmptyKernel(vec![0]))
        );
    }

    #[test]
    fn normalize_rejects_a_kernel_summing_to_zero() {
        let image = img(&[3], vec![1.0, 2.0, 3.0]);
        let kernel = img(&[3], vec![1.0, 0.0, -1.0]);
        assert_eq!(
            inverse_deconvolution(&image, &kernel, 1e-4, true, ZeroPad, Same),
            Err(FilterError::ZeroKernelSum)
        );
        assert_eq!(
            landweber_deconvolution(&image, &kernel, 0.1, 1, true, ZeroPad, Same),
            Err(FilterError::ZeroKernelSum)
        );
        // Without `Normalize` the same kernel is a perfectly good operator.
        assert!(inverse_deconvolution(&image, &kernel, 1e-4, false, ZeroPad, Same).is_ok());
    }

    // ---- multi-dimensional --------------------------------------------------

    #[test]
    fn deconvolution_round_trips_in_2d() {
        let original = img(
            &[6, 6],
            (0..36)
                .map(|i| {
                    let (x, y) = (i % 6, i / 6);
                    if (2..4).contains(&x) && (2..4).contains(&y) {
                        5.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        let kernel = img(&[3, 3], vec![1.0, 3.0, 1.0, 3.0, 9.0, 3.0, 1.0, 3.0, 1.0]);
        let blurred = blur(&original, &kernel);

        let recovered = inverse_deconvolution(
            &blurred,
            &kernel,
            KERNEL_ZERO_MAGNITUDE_THRESHOLD,
            true,
            ZeroPad,
            Same,
        )
        .unwrap();
        assert_close(&values(&recovered), &values(&original), 1e-8);
    }
}
