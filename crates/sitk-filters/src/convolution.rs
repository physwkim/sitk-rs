//! Convolution of an image with an arbitrary image kernel, by direct spatial
//! evaluation ([`convolution`]) or by multiplication in the Fourier domain
//! ([`fft_convolution`]).
//!
//! Ported from `itk::ConvolutionImageFilter` (itkConvolutionImageFilter.hxx),
//! `itk::FFTConvolutionImageFilter` (itkFFTConvolutionImageFilter.hxx) and
//! their shared base `itk::ConvolutionImageFilterBase`
//! (itkConvolutionImageFilterBase.hxx). Parameter names and defaults follow
//! SimpleITK's `ConvolutionImageFilter.yaml` / `FFTConvolutionImageFilter.yaml`
//! (`normalize` off, boundary condition zero-flux-Neumann, output region
//! `SAME`).
//!
//! # Kernel geometry
//!
//! Both filters *convolve*: the kernel is reflected before the inner product.
//! `ConvolutionImageFilter::ComputeConvolution`
//! (itkConvolutionImageFilter.hxx:96-125) builds its
//! `itk::NeighborhoodOperator` by
//!
//! 1. running the kernel through a `FlipImageFilter` with every axis flipped,
//!    which reverses the pixel buffer (itkFlipImageFilter.hxx:146-175);
//! 2. `ConstantPadImageFilter`-ing a single zero onto the **lower** side of
//!    every axis whose kernel extent is even (`GetKernelPadSize`,
//!    itkConvolutionImageFilter.hxx:208-224), making every extent odd;
//! 3. `ImageKernelOperator::CreateToRadius(radius)` with
//!    `radius[d] = kernelSize[d] / 2` (`GetKernelRadius`, ibid. 226-240), whose
//!    coefficients are the padded buffer verbatim
//!    (itkImageKernelOperator.hxx:52-81).
//!
//! `NeighborhoodInnerProduct` then accumulates `Σ_i op[i] * image[x + off(i)]`
//! with `off(i)` running from `-radius` in first-index-fastest order
//! (itkNeighborhoodInnerProduct.hxx:33-62, itkNeighborhood.hxx:41-67).
//! Unfolding the flip and the lower pad, that is
//!
//! ```text
//! out[x] = Σ_j kernel[j] * image[x + c - j],   c[d] = kernelSize[d] / 2
//! ```
//!
//! for **both** parities: the kernel's origin sits at index `size / 2`, so a
//! kernel that is `1` at `c + k` and zero elsewhere shifts the image by `+k`.
//! For an even extent the taps span offsets `-(c - 1) ..= c`, one short on the
//! low side — which is exactly why `GetValidRegion` compensates with
//! `validIndex -= 1; validSize += 1` (itkConvolutionImageFilterBase.hxx:79-84).
//!
//! # Accumulation
//!
//! `normalize` divides the kernel by its own sum
//! (`NormalizeToConstantImageFilter` with `Constant == 1`,
//! itkNormalizeToConstantImageFilter.hxx:53-76), summed in
//! `StatisticsImageFilter`'s `RealType` — `f64` here. The inner product
//! likewise accumulates in `NumericTraits<OutputPixelType>::RealType`
//! (itkNeighborhoodOperatorImageFilter.h:77), so this port computes in `f64`
//! throughout and narrows once, at the end, to the input's pixel type.
//!
//! # Geometry
//!
//! ITK expresses `VALID` as a shifted `LargestPossibleRegion` index while
//! leaving the origin alone, and SimpleITK surfaces that image with its origin
//! unchanged. This port does the same: the output keeps the input's spacing,
//! origin and direction in both region modes.

use sitk_core::{
    BoundaryCondition, ConstantBoundaryCondition, Image, NeighborhoodIterator,
    PeriodicBoundaryCondition, ScalarView, ZeroFluxNeumannBoundaryCondition,
};

use crate::error::{FilterError, Result};
use crate::fft::{self, Complex};
use crate::image_from_f64;

/// Out-of-bounds pixel rule applied while the kernel overhangs the image edge.
///
/// The three variants SimpleITK exposes on both convolution filters
/// (`ConvolutionImageFilter.yaml`), each backed by the matching
/// `sitk_core::boundary` implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvolutionBoundaryCondition {
    /// `itk::ConstantBoundaryCondition` with a zero constant.
    ZeroPad,
    /// `itk::ZeroFluxNeumannBoundaryCondition` — clamp to the nearest edge
    /// pixel. SimpleITK's default for both filters.
    #[default]
    ZeroFluxNeumannPad,
    /// `itk::PeriodicBoundaryCondition` — wrap around the image extent.
    PeriodicPad,
}

/// Which part of the convolution to return
/// (`ConvolutionImageFilterBaseEnums::ConvolutionImageFilterOutputRegion`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputRegionMode {
    /// The full input extent; the boundary condition fills the overhang.
    /// SimpleITK's default.
    #[default]
    Same,
    /// Only the pixels whose kernel window lies wholly inside the input, so no
    /// boundary condition is ever consulted (`GetValidRegion`,
    /// itkConvolutionImageFilterBase.hxx:48-90). Empty along any axis shorter
    /// than the kernel.
    Valid,
}

// ---- shared index helpers -------------------------------------------------

/// Decompose linear offset `i` into a multi-index of `size`, first index
/// fastest (matching [`Image`]'s layout).
pub(crate) fn unravel(mut i: usize, size: &[usize], out: &mut [usize]) {
    for (o, &s) in out.iter_mut().zip(size) {
        *o = i % s;
        i /= s;
    }
}

/// Linear offset of a multi-index within `size`, first index fastest.
pub(crate) fn ravel(index: &[usize], size: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&i, &s) in index.iter().zip(size) {
        offset += i * stride;
        stride *= s;
    }
    offset
}

/// The input, widened to `f64` so the boundary conditions and the accumulation
/// share one pixel type (ITK casts to `InternalImageType` for the same reason,
/// itkFFTConvolutionImageFilter.hxx:284-291).
pub(crate) fn as_f64_image(img: &Image) -> Result<Image> {
    Ok(Image::from_vec(img.size(), img.to_f64_vec()?)?)
}

/// Kernel values as `f64`, normalized to unit sum when asked.
///
/// `NormalizeToConstantImageFilter` divides by `GetSum() / Constant` with
/// `Constant == 1` (itkNormalizeToConstantImageFilter.hxx:71-76); the sum comes
/// from `StatisticsImageFilter` in `RealType`, i.e. `f64`.
pub(crate) fn prepare_kernel(image: &Image, kernel: &Image, normalize: bool) -> Result<Vec<f64>> {
    if kernel.dimension() != image.dimension() {
        return Err(FilterError::KernelDimensionMismatch {
            image: image.dimension(),
            kernel: kernel.dimension(),
        });
    }
    if kernel.size().contains(&0) {
        return Err(FilterError::EmptyKernel(kernel.size().to_vec()));
    }

    let mut values = kernel.to_f64_vec()?;
    if normalize {
        let sum: f64 = values.iter().sum();
        if sum == 0.0 {
            return Err(FilterError::ZeroKernelSum);
        }
        for v in &mut values {
            *v /= sum;
        }
    }
    Ok(values)
}

/// Per-axis `kernelSize[d] / 2` — ITK's kernel radius *and* its kernel origin
/// (`GetKernelRadius`, itkConvolutionImageFilter.hxx:226-240;
/// `FFTConvolutionImageFilter::GetKernelRadius`, itkFFTConvolutionImageFilter.hxx:490-502).
pub(crate) fn kernel_radius(kernel_size: &[usize]) -> Vec<usize> {
    kernel_size.iter().map(|&s| s / 2).collect()
}

/// The `SAME`-mode region: the whole input.
fn same_region(image_size: &[usize]) -> (Vec<usize>, Vec<usize>) {
    (vec![0; image_size.len()], image_size.to_vec())
}

/// `ConvolutionImageFilterBase::GetValidRegion`
/// (itkConvolutionImageFilterBase.hxx:48-90), verbatim: shrink by the radius on
/// both sides, then give an even-extent axis one pixel back on the low side to
/// account for the zero the kernel padding put there.
fn valid_region(image_size: &[usize], kernel_size: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let dim = image_size.len();
    let mut index = vec![0usize; dim];
    let mut size = vec![0usize; dim];
    for (((i, s), &n), &k) in index
        .iter_mut()
        .zip(size.iter_mut())
        .zip(image_size)
        .zip(kernel_size)
    {
        let radius = k / 2;
        if n < 2 * radius {
            *i = 0;
            *s = 0;
        } else {
            *i = radius;
            *s = n - 2 * radius;
            if k % 2 == 0 {
                *i -= 1;
                *s += 1;
            }
        }
    }
    (index, size)
}

pub(crate) fn output_region(
    image_size: &[usize],
    kernel_size: &[usize],
    mode: OutputRegionMode,
) -> (Vec<usize>, Vec<usize>) {
    match mode {
        OutputRegionMode::Same => same_region(image_size),
        OutputRegionMode::Valid => valid_region(image_size, kernel_size),
    }
}

/// Read `size` pixels of `img` starting at the (possibly negative) `index`,
/// routing every sample — in-bounds or not — through the boundary condition.
/// All three conditions read straight through for an in-bounds index.
fn sample_region<B: BoundaryCondition<f64>>(
    img: &ScalarView<'_, f64>,
    index: &[i64],
    size: &[usize],
    boundary: &B,
) -> Vec<f64> {
    let dim = size.len();
    let count: usize = size.iter().product();
    let mut out = Vec::with_capacity(count);
    let mut offset = vec![0usize; dim];
    let mut nd = vec![0i64; dim];
    for i in 0..count {
        unravel(i, size, &mut offset);
        for ((n, &base), &o) in nd.iter_mut().zip(index).zip(&offset) {
            *n = base + o as i64;
        }
        out.push(boundary.get_pixel(&nd, img));
    }
    out
}

// ---- direct spatial convolution -------------------------------------------

/// The `ImageKernelOperator`'s coefficients: the kernel buffer reversed, then
/// zero-padded on the low side of every even axis, so that the operator's
/// extent is `2 * radius + 1` everywhere
/// (itkConvolutionImageFilter.hxx:96-125 + itkImageKernelOperator.hxx:52-81).
fn kernel_operator(kernel_values: &[f64], kernel_size: &[usize], radius: &[usize]) -> Vec<f64> {
    let dim = kernel_size.len();
    let op_size: Vec<usize> = radius.iter().map(|&r| 2 * r + 1).collect();
    // 1 where the kernel extent is even; `GetKernelPadSize`.
    let pad: Vec<usize> = kernel_size.iter().map(|&s| 1 - s % 2).collect();

    let count: usize = op_size.iter().product();
    let mut op = vec![0.0f64; count];
    let mut m = vec![0usize; dim];
    let mut kernel_index = vec![0usize; dim];
    for (i, slot) in op.iter_mut().enumerate() {
        unravel(i, &op_size, &mut m);
        if m.iter().zip(&pad).any(|(&mi, &p)| mi < p) {
            continue; // the zero the lower pad introduced
        }
        for (((k, &mi), &p), &s) in kernel_index.iter_mut().zip(&m).zip(&pad).zip(kernel_size) {
            *k = s - 1 - (mi - p); // flip
        }
        *slot = kernel_values[ravel(&kernel_index, kernel_size)];
    }
    op
}

/// Evaluate `Σ_i op[i] * image[x + off(i)]` over the output region, exactly as
/// `NeighborhoodOperatorImageFilter` does (itkNeighborhoodOperatorImageFilter.hxx:82-108).
///
/// `Neighborhood::values` and `op` share ITK's first-index-fastest neighbor
/// ordering, so the dot product needs no reindexing.
fn convolve_spatial<B: BoundaryCondition<f64>>(
    img: &Image,
    op: &[f64],
    radius: &[usize],
    out_index: &[usize],
    out_size: &[usize],
    boundary: B,
) -> Result<Vec<f64>> {
    let dim = img.dimension();
    let iter = NeighborhoodIterator::<f64, _>::new(img, radius, boundary)?;

    let count: usize = out_size.iter().product();
    let mut values = Vec::with_capacity(count);
    let mut offset = vec![0usize; dim];
    let mut center = vec![0usize; dim];
    for i in 0..count {
        unravel(i, out_size, &mut offset);
        for ((c, &base), &o) in center.iter_mut().zip(out_index).zip(&offset) {
            *c = base + o;
        }
        let window = iter.neighborhood_at(&center);
        values.push(
            window
                .values()
                .iter()
                .zip(op)
                .map(|(&v, &c)| c * v)
                .sum::<f64>(),
        );
    }
    Ok(values)
}

/// `ConvolutionImageFilter`: convolve `image` with `kernel` by direct
/// evaluation of the inner product at every output pixel.
///
/// The output has `image`'s pixel type and geometry. In [`OutputRegionMode::Valid`]
/// its extent is `imageSize - kernelSize + 1` along each axis (zero where the
/// kernel is longer than the image); in [`OutputRegionMode::Same`] it matches
/// `image`.
///
/// Ported from itkConvolutionImageFilter.hxx; see the module docs for the
/// kernel-origin and even-extent conventions.
pub fn convolution(
    image: &Image,
    kernel: &Image,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    let kernel_values = prepare_kernel(image, kernel, normalize)?;
    let kernel_size = kernel.size();
    let radius = kernel_radius(kernel_size);
    let (out_index, out_size) = output_region(image.size(), kernel_size, output_region_mode);

    if out_size.contains(&0) {
        return image_from_f64(image.pixel_id(), &out_size, image, &[]);
    }

    let op = kernel_operator(&kernel_values, kernel_size, &radius);
    let widened = as_f64_image(image)?;
    let values = match boundary_condition {
        ConvolutionBoundaryCondition::ZeroPad => convolve_spatial(
            &widened,
            &op,
            &radius,
            &out_index,
            &out_size,
            ConstantBoundaryCondition::new(0.0f64),
        ),
        ConvolutionBoundaryCondition::ZeroFluxNeumannPad => convolve_spatial(
            &widened,
            &op,
            &radius,
            &out_index,
            &out_size,
            ZeroFluxNeumannBoundaryCondition,
        ),
        ConvolutionBoundaryCondition::PeriodicPad => convolve_spatial(
            &widened,
            &op,
            &radius,
            &out_index,
            &out_size,
            PeriodicBoundaryCondition,
        ),
    }?;

    image_from_f64(image.pixel_id(), &out_size, image, &values)
}

// ---- FFT convolution ------------------------------------------------------

/// What `FFTConvolutionImageFilter::PadInput` leaves behind
/// (itkFFTConvolutionImageFilter.hxx:143-296): the boundary-extended array the
/// forward transform consumes, plus the geometry [`crop_output`] needs to find
/// the requested region inside it.
pub(crate) struct PaddedInput {
    /// The padded pixels, first-index-fastest, `size.iter().product()` of them.
    pub(crate) values: Vec<f64>,
    /// The padded extent: every component a length `itkFFTPadImageFilter`
    /// accepts at `SizeGreatestPrimeFactor == 11` (`crate::fft::padded_length`).
    pub(crate) size: Vec<usize>,
    /// How many of the FFT-pad pixels `FFTPadImageFilter` put on the low side
    /// of each axis (`m_FFTPadSize[d] / 2`).
    pub(crate) lower: Vec<usize>,
}

fn pad_input_with<B: BoundaryCondition<f64>>(
    img: &Image,
    radius: &[usize],
    out_index: &[usize],
    out_size: &[usize],
    boundary: &B,
) -> Result<PaddedInput> {
    let roi_index: Vec<i64> = out_index
        .iter()
        .zip(radius)
        .map(|(&i, &r)| i as i64 - r as i64)
        .collect();
    let roi_size: Vec<usize> = out_size
        .iter()
        .zip(radius)
        .map(|(&s, &r)| s + 2 * r)
        .collect();
    let roi = Image::from_vec(
        &roi_size,
        sample_region(&img.scalar_view::<f64>()?, &roi_index, &roi_size, boundary),
    )?;

    let size: Vec<usize> = roi_size.iter().map(|&s| fft::padded_length(s)).collect();
    // `FFTPadImageFilter` puts `padSize / 2` of the new pixels on the low side.
    let lower: Vec<usize> = size
        .iter()
        .zip(&roi_size)
        .map(|(&p, &s)| (p - s) / 2)
        .collect();
    let pad_index: Vec<i64> = lower.iter().map(|&l| -(l as i64)).collect();
    let values = sample_region(&roi.scalar_view::<f64>()?, &pad_index, &size, boundary);

    Ok(PaddedInput {
        values,
        size,
        lower,
    })
}

/// `FFTConvolutionImageFilter::PadInput` (itkFFTConvolutionImageFilter.hxx:143-296).
///
/// For a whole-image request it reduces to: take the output region grown by the
/// kernel radius from the boundary-extended input (the `RegionOfInterest` at
/// `outputIndex - radius`, size `outputSize + 2*radius`, ibid. 214-221), then
/// hand *that* to `FFTPadImageFilter`, which grows each axis to the next size
/// whose greatest prime factor is at most `m_SizeGreatestPrimeFactor` (11, from
/// the PocketFFT backend — itkFFTConvolutionImageFilter.hxx:39, :260-263), with
/// `padSize / 2` pixels on the low side, and draws the new pixels from the
/// boundary condition applied to the region of interest
/// (itkFFTPadImageFilter.hxx:52-72).
///
/// `img` must already be an `f64` image ([`as_f64_image`]), matching ITK's cast
/// to `InternalImageType`.
pub(crate) fn pad_input(
    img: &Image,
    radius: &[usize],
    out_index: &[usize],
    out_size: &[usize],
    boundary_condition: ConvolutionBoundaryCondition,
) -> Result<PaddedInput> {
    match boundary_condition {
        ConvolutionBoundaryCondition::ZeroPad => pad_input_with(
            img,
            radius,
            out_index,
            out_size,
            &ConstantBoundaryCondition::new(0.0f64),
        ),
        ConvolutionBoundaryCondition::ZeroFluxNeumannPad => pad_input_with(
            img,
            radius,
            out_index,
            out_size,
            &ZeroFluxNeumannBoundaryCondition,
        ),
        ConvolutionBoundaryCondition::PeriodicPad => {
            pad_input_with(img, radius, out_index, out_size, &PeriodicBoundaryCondition)
        }
    }
}

/// `FFTConvolutionImageFilter::PrepareKernel`'s transfer function
/// (itkFFTConvolutionImageFilter.hxx:322-422): zero-pad the kernel on the
/// *upper* side up to `padded_size`, cyclically shift it by `-(kernelSize / 2)`
/// (itkCyclicShiftImageFilter.hxx:65-84) so the kernel origin lands on the
/// padded array's index 0, and transform.
///
/// `kernel_values` has already been through [`prepare_kernel`], so `Normalize`
/// is folded in.
pub(crate) fn kernel_spectrum(
    kernel_values: &[f64],
    kernel_size: &[usize],
    radius: &[usize],
    padded_size: &[usize],
) -> Vec<Complex> {
    let dim = padded_size.len();
    let total: usize = padded_size.iter().product();

    // `shifted[j] = padded[(j + radius) mod paddedSize]`.
    let mut spectrum = vec![Complex::default(); total];
    let mut m = vec![0usize; dim];
    let mut source = vec![0usize; dim];
    for (j, slot) in spectrum.iter_mut().enumerate() {
        unravel(j, padded_size, &mut m);
        for (((s, &mi), &r), &p) in source.iter_mut().zip(&m).zip(radius).zip(padded_size) {
            *s = (mi + r) % p;
        }
        // Outside the kernel's own extent the upper zero pad supplies a zero.
        if source.iter().zip(kernel_size).all(|(&s, &k)| s < k) {
            *slot = Complex::new(kernel_values[ravel(&source, kernel_size)], 0.0);
        }
    }
    fft::transform_nd(&mut spectrum, padded_size, false);
    spectrum
}

/// `FFTConvolutionImageFilter::CropOutput` (itkFFTConvolutionImageFilter.hxx:445-488):
/// extract `out_size` pixels at `paddedIndex + fftPadSize / 2 + kernelRadius`.
///
/// Every tap the circular convolution reads for those pixels lands inside the
/// region of interest, so the FFT padding never contributes.
pub(crate) fn crop_output(
    padded: &[f64],
    padded_size: &[usize],
    lower: &[usize],
    radius: &[usize],
    out_size: &[usize],
) -> Vec<f64> {
    let dim = padded_size.len();
    let count: usize = out_size.iter().product();
    let mut values = Vec::with_capacity(count);
    let mut m = vec![0usize; dim];
    let mut index = vec![0usize; dim];
    for i in 0..count {
        unravel(i, out_size, &mut m);
        for (((x, &l), &r), &o) in index.iter_mut().zip(lower).zip(radius).zip(&m) {
            *x = l + r + o;
        }
        values.push(padded[ravel(&index, padded_size)]);
    }
    values
}

/// The Fourier half of `FFTConvolutionImageFilter::GenerateData`
/// (itkFFTConvolutionImageFilter.hxx:80-112): transform the padded input and
/// the kernel, multiply, invert, crop.
///
/// Agrees with [`convolve_spatial`] up to round-off.
fn convolve_fft(
    img: &Image,
    kernel_values: &[f64],
    kernel_size: &[usize],
    radius: &[usize],
    out_index: &[usize],
    out_size: &[usize],
    boundary_condition: ConvolutionBoundaryCondition,
) -> Result<Vec<f64>> {
    let padded = pad_input(img, radius, out_index, out_size, boundary_condition)?;
    let transfer = kernel_spectrum(kernel_values, kernel_size, radius, &padded.size);

    let mut spectrum: Vec<Complex> = padded
        .values
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .collect();
    fft::transform_nd(&mut spectrum, &padded.size, false);
    for (x, &k) in spectrum.iter_mut().zip(&transfer) {
        *x = *x * k;
    }
    fft::transform_nd(&mut spectrum, &padded.size, true);

    let real: Vec<f64> = spectrum.iter().map(|x| x.re).collect();
    Ok(crop_output(
        &real,
        &padded.size,
        &padded.lower,
        radius,
        out_size,
    ))
}

/// `FFTConvolutionImageFilter`: the same convolution as [`convolution`],
/// computed as a pointwise product of discrete Fourier transforms.
///
/// Agrees with [`convolution`] to within FFT round-off for every kernel parity,
/// dimension, boundary condition and output-region mode.
///
/// Ported from itkFFTConvolutionImageFilter.hxx. The transforms are this
/// crate's own mixed-radix DFT (`crate::fft`), and the padding is
/// `itkFFTPadImageFilter`'s search at the `SizeGreatestPrimeFactor == 11` that
/// filter's constructor seeds from the PocketFFT backend (`.hxx:39`).
pub fn fft_convolution(
    image: &Image,
    kernel: &Image,
    normalize: bool,
    boundary_condition: ConvolutionBoundaryCondition,
    output_region_mode: OutputRegionMode,
) -> Result<Image> {
    let kernel_values = prepare_kernel(image, kernel, normalize)?;
    let kernel_size = kernel.size();
    let radius = kernel_radius(kernel_size);
    let (out_index, out_size) = output_region(image.size(), kernel_size, output_region_mode);

    if out_size.contains(&0) {
        return image_from_f64(image.pixel_id(), &out_size, image, &[]);
    }

    let widened = as_f64_image(image)?;
    let values = convolve_fft(
        &widened,
        &kernel_values,
        kernel_size,
        &radius,
        &out_index,
        &out_size,
        boundary_condition,
    )?;

    image_from_f64(image.pixel_id(), &out_size, image, &values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    use ConvolutionBoundaryCondition::{PeriodicPad, ZeroFluxNeumannPad, ZeroPad};
    use OutputRegionMode::{Same, Valid};

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

    /// Deterministic filler so the cross-check exercises non-degenerate data.
    fn ramp(size: &[usize], seed: u64) -> Image {
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

    // ---- kernel-origin convention -----------------------------------------

    #[test]
    fn identity_kernel_reproduces_the_input() {
        let image = img(&[4, 3], (0..12).map(|i| i as f64).collect());
        let kernel = img(&[1, 1], vec![1.0]);
        for &bc in &[ZeroPad, ZeroFluxNeumannPad, PeriodicPad] {
            let direct = convolution(&image, &kernel, false, bc, Same).unwrap();
            assert_eq!(values(&direct), values(&image));
            let fourier = fft_convolution(&image, &kernel, false, bc, Same).unwrap();
            assert_close(&values(&fourier), &values(&image), 1e-10);
        }
    }

    #[test]
    fn kernel_origin_sits_at_size_over_two_for_an_odd_kernel() {
        // out[x] = Σ_j k[j] * in[x + c - j] with c = 3/2 = 1. A single 1 at
        // j = c + 1 = 2 therefore delays the image by one pixel.
        let image = img(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let delayed = convolution(
            &image,
            &img(&[3], vec![0.0, 0.0, 1.0]),
            false,
            ZeroPad,
            Same,
        );
        assert_eq!(values(&delayed.unwrap()), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        // A 1 at j = c - 1 = 0 advances it by one.
        let advanced = convolution(
            &image,
            &img(&[3], vec![1.0, 0.0, 0.0]),
            false,
            ZeroPad,
            Same,
        );
        assert_eq!(values(&advanced.unwrap()), vec![2.0, 3.0, 4.0, 5.0, 0.0]);
    }

    #[test]
    fn kernel_origin_sits_at_size_over_two_for_an_even_kernel() {
        // c = 4/2 = 2, so the taps span offsets -1..=2 (one short on the low
        // side: that is the zero the lower kernel pad introduces).
        let image = img(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let delayed = convolution(
            &image,
            &img(&[4], vec![0.0, 0.0, 0.0, 1.0]),
            false,
            ZeroPad,
            Same,
        );
        assert_eq!(values(&delayed.unwrap()), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let advanced = convolution(
            &image,
            &img(&[4], vec![1.0, 0.0, 0.0, 0.0]),
            false,
            ZeroPad,
            Same,
        );
        assert_eq!(values(&advanced.unwrap()), vec![3.0, 4.0, 5.0, 0.0, 0.0]);
    }

    // ---- even-kernel index math -------------------------------------------

    #[test]
    fn even_kernel_same_mode_matches_hand_derived_taps() {
        // in = [1..5], k = [1,2,3,4], c = 2, zero pad.
        // out[x] = 1*in[x+2] + 2*in[x+1] + 3*in[x] + 4*in[x-1]
        //        = [10, 20, 30, 34, 31]
        let image = img(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let kernel = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let out = convolution(&image, &kernel, false, ZeroPad, Same).unwrap();
        assert_eq!(values(&out), vec![10.0, 20.0, 30.0, 34.0, 31.0]);
    }

    #[test]
    fn even_kernel_valid_mode_keeps_only_the_boundary_free_taps() {
        // Of the five SAME pixels only x = 1 and x = 2 read in[x-1..=x+2]
        // entirely inside [0, 5).
        let image = img(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let kernel = img(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let out = convolution(&image, &kernel, false, ZeroPad, Valid).unwrap();
        assert_eq!(out.size(), &[2]);
        assert_eq!(values(&out), vec![20.0, 30.0]);
        // The boundary condition cannot reach a VALID pixel.
        let flux = convolution(&image, &kernel, false, ZeroFluxNeumannPad, Valid).unwrap();
        assert_eq!(values(&flux), vec![20.0, 30.0]);
    }

    // ---- VALID region arithmetic ------------------------------------------

    #[test]
    fn valid_region_size_is_image_minus_kernel_plus_one_for_both_parities() {
        for n in 1..=8usize {
            for k in 1..=8usize {
                let (index, size) = valid_region(&[n], &[k]);
                assert_eq!(
                    size[0],
                    n.saturating_sub(k - 1),
                    "valid size for image {n}, kernel {k}"
                );
                if size[0] > 0 {
                    assert_eq!(
                        index[0],
                        (k - 1) / 2,
                        "valid index for image {n}, kernel {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_longer_than_the_image_gives_a_valid_empty_but_full_same_region() {
        let image = img(&[3], vec![1.0, 2.0, 3.0]);
        let kernel = img(&[5], vec![1.0; 5]);

        for f in [convolution, fft_convolution] {
            let empty = f(&image, &kernel, false, ZeroFluxNeumannPad, Valid).unwrap();
            assert_eq!(empty.size(), &[0]);
            assert_eq!(empty.number_of_pixels(), 0);

            // SAME still evaluates: the boundary condition covers the overhang.
            let same = f(&image, &kernel, false, ZeroPad, Same).unwrap();
            assert_eq!(same.size(), &[3]);
            assert_close(&values(&same), &[6.0, 6.0, 6.0], 1e-10);
        }
    }

    // ---- normalize --------------------------------------------------------

    #[test]
    fn normalize_divides_the_kernel_by_its_own_sum() {
        let image = img(&[5], vec![3.0, 6.0, 9.0, 12.0, 15.0]);
        let kernel = img(&[3], vec![1.0, 1.0, 1.0]);
        let off = convolution(&image, &kernel, false, ZeroFluxNeumannPad, Same).unwrap();
        assert_eq!(values(&off), vec![12.0, 18.0, 27.0, 36.0, 42.0]);
        let on = convolution(&image, &kernel, true, ZeroFluxNeumannPad, Same).unwrap();
        assert_close(&values(&on), &[4.0, 6.0, 9.0, 12.0, 14.0], 1e-12);
    }

    #[test]
    fn normalize_rejects_a_kernel_summing_to_zero() {
        let image = img(&[3], vec![1.0, 2.0, 3.0]);
        let kernel = img(&[3], vec![1.0, 0.0, -1.0]);
        assert_eq!(
            convolution(&image, &kernel, true, ZeroPad, Same),
            Err(FilterError::ZeroKernelSum)
        );
        assert_eq!(
            fft_convolution(&image, &kernel, true, ZeroPad, Same),
            Err(FilterError::ZeroKernelSum)
        );
        // Without normalize the same kernel is a perfectly good derivative.
        assert!(convolution(&image, &kernel, false, ZeroPad, Same).is_ok());
    }

    // ---- boundary conditions ----------------------------------------------

    #[test]
    fn boundary_condition_changes_the_border_pixels_and_nothing_else() {
        let image = img(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let kernel = img(&[3], vec![1.0, 1.0, 1.0]);
        let zero = values(&convolution(&image, &kernel, false, ZeroPad, Same).unwrap());
        let flux = values(&convolution(&image, &kernel, false, ZeroFluxNeumannPad, Same).unwrap());
        let wrap = values(&convolution(&image, &kernel, false, PeriodicPad, Same).unwrap());

        assert_eq!(zero, vec![3.0, 6.0, 9.0, 12.0, 9.0]);
        assert_eq!(flux, vec![4.0, 6.0, 9.0, 12.0, 14.0]);
        assert_eq!(wrap, vec![8.0, 6.0, 9.0, 12.0, 10.0]);
        assert_eq!(zero[1..4], flux[1..4]);
        assert_eq!(zero[1..4], wrap[1..4]);
    }

    // ---- input validation --------------------------------------------------

    #[test]
    fn kernel_dimension_must_match_the_image() {
        let image = img(&[3, 3], vec![0.0; 9]);
        let kernel = img(&[3], vec![1.0; 3]);
        assert_eq!(
            convolution(&image, &kernel, false, ZeroPad, Same),
            Err(FilterError::KernelDimensionMismatch {
                image: 2,
                kernel: 1
            })
        );
    }

    #[test]
    fn kernel_with_a_zero_length_axis_is_rejected() {
        let image = img(&[3], vec![1.0, 2.0, 3.0]);
        let kernel = img(&[0], vec![]);
        assert_eq!(
            convolution(&image, &kernel, false, ZeroPad, Same),
            Err(FilterError::EmptyKernel(vec![0]))
        );
    }

    // ---- pixel type and geometry ------------------------------------------

    #[test]
    fn output_keeps_the_input_pixel_type_and_geometry() {
        let mut image = Image::from_vec(&[3, 2], vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        image.set_spacing(&[0.5, 2.0]).unwrap();
        image.set_origin(&[7.0, -1.0]).unwrap();
        let kernel = img(&[1, 1], vec![1.0]);

        for mode in [Same, Valid] {
            let out = convolution(&image, &kernel, false, ZeroPad, mode).unwrap();
            assert_eq!(out.pixel_id(), PixelId::UInt8);
            assert_eq!(out.spacing(), image.spacing());
            assert_eq!(out.origin(), image.origin());
            assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4, 5, 6]);
        }
    }

    // ---- spatial vs FFT ----------------------------------------------------

    fn cross_check(image_size: &[usize], kernel_size: &[usize], normalize: bool) {
        let image = ramp(image_size, 0x5eed_1234);
        let mut kernel = ramp(kernel_size, 0xabcd_ef01);
        if normalize {
            // `ramp` straddles zero and can sum to nearly nothing; bias it so
            // the division by the kernel sum stays well-conditioned.
            let biased: Vec<f64> = values(&kernel).iter().map(|v| v + 10.0).collect();
            kernel = img(kernel_size, biased);
        }

        for bc in [ZeroPad, ZeroFluxNeumannPad, PeriodicPad] {
            for mode in [Same, Valid] {
                let direct = convolution(&image, &kernel, normalize, bc, mode).unwrap();
                let fourier = fft_convolution(&image, &kernel, normalize, bc, mode).unwrap();
                assert_eq!(
                    direct.size(),
                    fourier.size(),
                    "size for image {image_size:?} kernel {kernel_size:?} {bc:?} {mode:?}"
                );
                assert_close(&values(&fourier), &values(&direct), 1e-9);
            }
        }
    }

    #[test]
    fn spatial_and_fft_agree_in_1d() {
        // The FFT path transforms `7 + 2 * (kernel / 2)` points: 9, 11, 7, 12 —
        // all 11-smooth, so `FFTPadImageFilter` adds nothing. (The radix-2
        // backend this module used to sit on transformed 16, 16, 8, 16 instead;
        // the results are the same, which is the point of ledger §4.2.)
        cross_check(&[7], &[3], false);
        cross_check(&[7], &[4], false);
        cross_check(&[7], &[1], false);
        cross_check(&[8], &[5], false);
    }

    #[test]
    fn spatial_and_fft_agree_in_2d() {
        cross_check(&[5, 6], &[3, 3], false);
        cross_check(&[5, 6], &[4, 4], false);
        cross_check(&[5, 6], &[3, 4], false);
        cross_check(&[5, 6], &[1, 3], false);
    }

    #[test]
    fn spatial_and_fft_agree_in_3d() {
        cross_check(&[4, 5, 3], &[3, 3, 3], false);
        cross_check(&[4, 5, 3], &[2, 2, 2], false);
    }

    #[test]
    fn spatial_and_fft_agree_with_normalize_on() {
        cross_check(&[7], &[4], true);
        cross_check(&[5, 6], &[3, 4], true);
    }

    #[test]
    fn spatial_and_fft_agree_when_the_kernel_exceeds_the_image() {
        cross_check(&[3], &[7], false);
        cross_check(&[3, 3], &[5, 4], false);
    }
}
