//! Crate-private complex DFT, the transform behind [`crate::fourier`],
//! [`crate::convolution::fft_convolution`], [`crate::deconvolution`],
//! [`crate::fft_correlation`] and `N4BiasFieldCorrection`'s `SharpenImage`.
//!
//! # Which backend this ports
//!
//! ITK hands its transforms to a pluggable FFT backend selected by the object
//! factory. In this checkout the always-compiled, always-registered default is
//! **PocketFFT** (`Modules/Filtering/FFT/src/itkPocketFFTImageFilterInitFactory.cxx:48`
//! registers `PocketFFTForwardFFTImageFilter`), whose
//! `GetSizeGreatestPrimeFactor()` returns `11` with the comment "All sizes are
//! supported; sizes factoring into 2,3,5,7,11 use fast kernels"
//! (`itkPocketFFTForwardFFTImageFilter.hxx:72-76`). The VNL backend, whose
//! base-class `GetSizeGreatestPrimeFactor() == 2` once justified a radix-2-only
//! transform here, is a deprecated wrapper.
//!
//! So this module provides what PocketFFT provides:
//!
//! - direct radix kernels for the factors `2, 3, 5, 7, 11` ([`FAST_RADICES`]),
//!   composed by Cooley-Tukey into any length whose prime factors all lie in
//!   that set, and
//! - **Bluestein's algorithm** for every other length, so an arbitrary length
//!   (a large prime, say `97`) transforms exactly.
//!
//! Both are exact DFTs; they differ only in operation count and round-off.
//!
//! # Two kernels
//!
//! The scalar butterflies above are one of the two 1-D kernels here. The other
//! is `rustfft` (`MIT OR Apache-2.0`) — the same class of implementation as
//! pocketfft, a planner over hand-written SIMD butterflies with Bluestein for
//! the lengths that do not factor — with `realfft` (`MIT`) on top of it for
//! real-valued input, which is what lets [`transform_r2c`]/[`transform_c2r`]
//! spend a *half-length* complex transform on axis 0 instead of a full one.
//! rustfft is about 3x the speed of the scalar kernel and a few ulps less exact,
//! because it computes the roots of unity that pocketfft hardcodes.
//!
//! Which one a pass runs is [`LineKernel`], chosen per call site and never by
//! default; that type documents what the choice costs and why the real-input
//! pair can afford the fast kernel while the complex-spectrum API cannot.
//!
//! # Padding
//!
//! [`padded_length`] is `itkFFTPadImageFilter`'s next-size search
//! (`itkFFTPadImageFilter.hxx:55-63`) at the seeded
//! [`DEFAULT_SIZE_GREATEST_PRIME_FACTOR`], which is what
//! `FFTConvolutionImageFilter` passes down
//! (`itkFFTConvolutionImageFilter.hxx:39`, `:260-263`). Note that
//! `MaskedFFTNormalizedCorrelationImageFilter` does **not** use
//! `itkFFTPadImageFilter` — it has its own 2/3/5-only search; see
//! [`crate::fft_correlation`] and ledger §2.110.
//!
//! # Precision
//!
//! Every transform runs in `f64` regardless of the caller's pixel type, the
//! crate-wide divergence of ledger §4.1. Upstream instantiates PocketFFT on the
//! pixel's component type, so a `ComplexFloat32` image is transformed in
//! `float` there and in `f64` here, then rounded once on the way out.
//!
//! Ledger: §2.109, §2.110, §4.2.

use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use sitk_core::parallel;

/// A complex number. Crate-private: the public complex-pixel surface is
/// [`sitk_core::Complex`], and no value of this type escapes `sitk-filters`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Complex {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl Complex {
    pub(crate) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `std::conj`.
    pub(crate) const fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// `std::norm`: the *squared* magnitude, as the deconvolution functors use
    /// it (`itkTikhonovDeconvolutionImageFilter.h:142`).
    pub(crate) fn norm(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// `itk::Math::Absolute` of a complex value, i.e. `std::abs` — the
    /// magnitude (`itkInverseDeconvolutionImageFilter.h:145`).
    pub(crate) fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Multiplication by a real scalar, the shape every deconvolution functor
    /// mixes a `double` parameter into a complex spectrum with.
    pub(crate) fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    /// `exp(i * theta)`. Only the tests reach for an unreduced angle; the
    /// transform itself goes through [`twiddle`] and [`radix_root`].
    #[cfg(test)]
    fn unit(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }

    /// `rustfft`'s complex type. The same two `f64` in the same order, so the
    /// conversion is a move and cannot perturb a value.
    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re, self.im)
    }

    const fn from_c64(c: Complex64) -> Self {
        Self::new(c.re, c.im)
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Complex;
    fn div(self, rhs: Complex) -> Complex {
        let d = rhs.re * rhs.re + rhs.im * rhs.im;
        Complex::new(
            (self.re * rhs.re + self.im * rhs.im) / d,
            (self.im * rhs.re - self.re * rhs.im) / d,
        )
    }
}

// ---- ITK's size arithmetic -------------------------------------------------

/// `itk::Math::IsPrime` (itkMath.h:767-793).
fn is_prime(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut x = 5usize;
    while x <= n / x {
        if n.is_multiple_of(x) || n.is_multiple_of(x + 2) {
            return false;
        }
        x += 6;
    }
    true
}

/// `itk::Math::GreatestPrimeFactor` (itkMath.h:798-815).
///
/// Note the two degenerate inputs the loop never enters: `greatest_prime_factor(0)`
/// and `greatest_prime_factor(1)` both return `2`, so a length-1 axis is
/// already acceptable to every `size_greatest_prime_factor >= 2`.
pub(crate) fn greatest_prime_factor(n: usize) -> usize {
    let mut n = n;
    let mut v = 2usize;
    while v <= n {
        if n.is_multiple_of(v) && is_prime(v) {
            n /= v;
        } else {
            v += 1;
        }
    }
    v
}

/// `FFTPadImageFilter`'s default `SizeGreatestPrimeFactor`.
///
/// `itkFFTPadImageFilter.hxx:34-35` seeds it from
/// `ForwardFFTImageFilter<Image<float, Dim>>::New()->GetSizeGreatestPrimeFactor()`,
/// which the object factory resolves to
/// `PocketFFTForwardFFTImageFilter::GetSizeGreatestPrimeFactor() == 11`
/// (`itkPocketFFTForwardFFTImageFilter.hxx:72-76`). SimpleITK surfaces the same
/// number through `FFTPadImageFilter::DefaultSizeGreatestPrimeFactor()`
/// (`FFTPadImageFilter.yaml`'s `custom_methods`), whose *documentation* still
/// says "5 for VNL" — ledger §3.38.
pub(crate) const DEFAULT_SIZE_GREATEST_PRIME_FACTOR: usize = 11;

/// The number of pixels `itkFFTPadImageFilter` adds to an axis of `size`
/// pixels, `GenerateOutputInformation` verbatim (itkFFTPadImageFilter.hxx:55-68).
///
/// Three regimes, and the middle one contradicts the filter's own
/// documentation (ledger §2.109):
///
/// - `gpf > 1`: the smallest `pad` for which `GreatestPrimeFactor(size + pad) <= gpf`.
/// - `gpf == 1`: `pad = size % 2` — the axis is grown to an *even* size, not
///   left alone as the doc-comment ("A greatest prime factor of 1 or less -
///   typically 0 - disable the extra padding") claims.
/// - `gpf == 0`: no padding.
///
/// Upstream's member is an unsigned `SizeValueType` while SimpleITK's parameter
/// is a signed `int`, so a negative value there becomes a huge unsigned one and
/// lands in the `gpf > 1` arm with a `while` condition that is never true —
/// i.e. no padding. `usize::MAX` reproduces that here.
pub(crate) fn fft_pad_size(size: usize, gpf: usize) -> usize {
    let mut pad = 0usize;
    if gpf > 1 {
        while greatest_prime_factor(size + pad) > gpf {
            pad += 1;
        }
    } else if gpf == 1 {
        pad += (size + pad) % 2;
    }
    pad
}

/// The length `itkFFTPadImageFilter` grows an axis of `n` pixels to at the
/// default [`DEFAULT_SIZE_GREATEST_PRIME_FACTOR`]: the smallest `m >= n` whose
/// greatest prime factor is at most 11.
///
/// This is the transform length `FFTConvolutionImageFilter` and the
/// deconvolution filters derived from it use.
pub(crate) fn padded_length(n: usize) -> usize {
    n + fft_pad_size(n, DEFAULT_SIZE_GREATEST_PRIME_FACTOR)
}

// ---- the transform ---------------------------------------------------------

/// The factors PocketFFT has direct kernels for
/// (`itkPocketFFTForwardFFTImageFilter.hxx:74`). Ordered ascending: the
/// Cooley-Tukey recursion peels the smallest available factor first.
const FAST_RADICES: [usize; 5] = [2, 3, 5, 7, 11];

/// The most values a single radix butterfly holds, `FAST_RADICES.last()`.
const MAX_RADIX: usize = 11;

/// Whether `n` factors entirely into [`FAST_RADICES`], i.e. whether the
/// Cooley-Tukey path can reach length 1.
fn is_fast_length(n: usize) -> bool {
    let mut n = n;
    for r in FAST_RADICES {
        while n.is_multiple_of(r) {
            n /= r;
        }
    }
    n == 1
}

/// `(cos(2πj/r), sin(2πj/r))` for `j = 1 ..= (r-1)/2`, transcribed from the
/// `constexpr T0 tw{j}r / tw{j}i` literals of pocketfft's `pass3b`, `pass5b`,
/// `pass7` and `pass11` (`pocketfft_hdronly.h:1077`, `:1190`, `:1261`, `:1441`).
///
/// These are *not* `(2.0 * PI * j as f64 / r as f64).cos()`. `cos(2π/3)` is
/// exactly `-0.5`, but the double nearest `2π/3` has a cosine of
/// `-0.4999999999999998` — two ulps out, and enough to turn an exact
/// delta-kernel deconvolution round trip into `9.999999999999998`. pocketfft
/// hardcodes the correctly-rounded values, and so does this port.
///
/// Upstream writes each one as a `long double` literal and lets the compiler
/// round it to `T0`; below are the same values as the shortest `f64` literals
/// that round-trip to the same bits. The upstream digits, for the record:
///
/// ```text
/// pass3b   tw1 = -0.5                                  0.8660254037844386467637231707529362
/// pass5b   tw1 =  0.3090169943749474241022934171828191 0.9510565162951535721164393333793821
///          tw2 = -0.8090169943749474241022934171828191 0.5877852522924731291687059546390728
/// pass7    tw1 =  0.6234898018587335305250048840042398 0.7818314824680298087084445266740578
///          tw2 = -0.2225209339563144042889025644967948 0.9749279121818236070181316829939312
///          tw3 = -0.9009688679024191262361023195074451 0.4338837391175581204757683328483590
/// pass11   tw1 =  0.8412535328311811688618116489193677 0.5406408174555975821076359543186917
///          tw2 =  0.4154150130018864255292741492296232 0.9096319953545183714117153830790285
///          tw3 = -0.1423148382732851404437926686163697 0.9898214418809327323760920377767188
///          tw4 = -0.6548607339452850640569250724662936 0.7557495743542582837740358439723444
///          tw5 = -0.9594929736144973898903680570663277 0.2817325568414296977114179153466169
/// ```
fn radix_twiddles(radix: usize) -> &'static [(f64, f64)] {
    match radix {
        3 => &[(-0.5, 0.866_025_403_784_438_6)],
        5 => &[
            (0.309_016_994_374_947_45, 0.951_056_516_295_153_5),
            (-0.809_016_994_374_947_5, 0.587_785_252_292_473_1),
        ],
        7 => &[
            (0.623_489_801_858_733_5, 0.781_831_482_468_029_8),
            (-0.222_520_933_956_314_4, 0.974_927_912_181_823_6),
            (-0.900_968_867_902_419_1, 0.433_883_739_117_558_1),
        ],
        11 => &[
            (0.841_253_532_831_181_2, 0.540_640_817_455_597_6),
            (0.415_415_013_001_886_44, 0.909_631_995_354_518_3),
            (-0.142_314_838_273_285_14, 0.989_821_441_880_932_7),
            (-0.654_860_733_945_285_1, 0.755_749_574_354_258_3),
            (-0.959_492_973_614_497_4, 0.281_732_556_841_429_67),
        ],
        _ => unreachable!("no radix kernel for {radix}"),
    }
}

/// `exp(sign · 2πi · t / radix)`, the radix-point kernel's own root of unity.
///
/// The `j > radix/2` half is the conjugate of the `radix − j` entry, which is
/// how pocketfft's butterflies use the same `tw{j}` pair for both.
fn radix_root(radix: usize, t: usize, sign: f64) -> Complex {
    if t == 0 {
        return Complex::new(1.0, 0.0);
    }
    let table = radix_twiddles(radix);
    if t <= radix / 2 {
        let (re, im) = table[t - 1];
        Complex::new(re, sign * im)
    } else {
        let (re, im) = table[radix - t - 1];
        Complex::new(re, -sign * im)
    }
}

/// `exp(sign · 2πi · i / n)` — `sincos_2pibyn<T>::calc`
/// (`pocketfft_hdronly.h:336-368`) verbatim.
///
/// The octant reduction `x = 8i` keeps the argument of `cos`/`sin` inside
/// `[0, π/4]`, where the library routines are most accurate: a direct
/// `(2.0 * PI * i as f64 / n as f64).cos()` loses bits to the inexact `2π/n`
/// long before the transform's own error shows up.
///
/// Upstream computes `ang = 0.25 * π / n` in `long double` and rounds once;
/// this port has no `long double`, so `ang` carries the `f64` rounding of `π`.
fn twiddle(i: usize, n: usize, sign: f64) -> Complex {
    let ang = 0.25 * PI / n as f64;
    let mut x = (i % n) * 8;
    let at = |k: usize| (k as f64) * ang;

    let (re, im) = if x < 4 * n {
        if x < 2 * n {
            if x < n {
                (at(x).cos(), at(x).sin())
            } else {
                (at(2 * n - x).sin(), at(2 * n - x).cos())
            }
        } else {
            x -= 2 * n;
            if x < n {
                (-at(x).sin(), at(x).cos())
            } else {
                (-at(2 * n - x).cos(), at(2 * n - x).sin())
            }
        }
    } else {
        x = 8 * n - x;
        if x < 2 * n {
            if x < n {
                (at(x).cos(), -at(x).sin())
            } else {
                (at(2 * n - x).sin(), -at(2 * n - x).cos())
            }
        } else {
            x -= 2 * n;
            if x < n {
                (-at(x).sin(), -at(x).cos())
            } else {
                (-at(2 * n - x).cos(), -at(2 * n - x).sin())
            }
        }
    };
    Complex::new(re, sign * im)
}

/// Decimation-in-time Cooley-Tukey over the [`FAST_RADICES`].
///
/// Reads `n` values from `inp` at `stride` spacing and writes the length-`n`
/// DFT into `out` (which must be exactly `n` long, and must not alias `inp`).
/// `sign` is `-1.0` for the `exp(-2πi jk/n)` kernel and `+1.0` for `exp(+…)`.
///
/// Requires [`is_fast_length`]`(n)`; every sub-length `n / r` inherits that
/// property, so the recursion bottoms out at `n == 1`.
fn fft_fast(inp: &[Complex], stride: usize, out: &mut [Complex], n: usize, sign: f64) {
    if n == 1 {
        out[0] = inp[0];
        return;
    }

    let radix = FAST_RADICES
        .into_iter()
        .find(|r| n.is_multiple_of(*r))
        .expect("fft_fast called on a length with a prime factor above 11");
    let m = n / radix;

    // Sub-transform p gathers the input samples congruent to p mod radix.
    for (p, chunk) in out.chunks_exact_mut(m).enumerate() {
        fft_fast(&inp[p * stride..], stride * radix, chunk, m, sign);
    }

    // The radix-point butterfly: for each k, combine the radix sub-transforms
    // at offset k. `out[q * m + k]` for q in 0..radix reads and writes exactly
    // the same radix slots, so the butterfly is in-place per k.
    let mut temp = [Complex::default(); MAX_RADIX];
    // The radix-point kernel's own roots of unity. Radix 2 needs none: its
    // butterfly is the sum and the difference.
    let roots: [Complex; MAX_RADIX] = if radix == 2 {
        [Complex::default(); MAX_RADIX]
    } else {
        std::array::from_fn(|t| radix_root(radix, t % radix, sign))
    };

    for k in 0..m {
        for (p, slot) in temp[..radix].iter_mut().enumerate() {
            // `p * k` never exceeds `(radix - 1) * (m - 1) < n`, and `twiddle`
            // reduces it again; `p == 0` and `k == 0` both give an exact `1`.
            *slot = out[p * m + k] * twiddle(p * k, n, sign);
        }
        if radix == 2 {
            let (u, v) = (temp[0], temp[1]);
            out[k] = u + v;
            out[m + k] = u - v;
        } else {
            for q in 0..radix {
                let mut acc = temp[0];
                for p in 1..radix {
                    acc = acc + temp[p] * roots[(p * q) % radix];
                }
                out[q * m + k] = acc;
            }
        }
    }
}

/// Bluestein's chirp-z algorithm: the DFT of *any* length, as a cyclic
/// convolution of a power-of-two length that [`fft_fast`] handles.
///
/// With `w = exp(sign·2πi/n)` and `c[t] = exp(sign·πi·t²/n)`, the identity
/// `jk = (j² + k² − (k−j)²) / 2` turns `X[k] = Σ_j x[j] w^{jk}` into
/// `X[k] = c[k] · Σ_j (x[j] c[j]) · conj(c[k−j])`, a linear convolution. It is
/// evaluated as a cyclic convolution of length `m >= 2n − 1` (the shortest
/// power of two), which is long enough that no wrapped term reaches `k < n`.
fn fft_bluestein(buf: &mut [Complex], sign: f64) {
    let n = buf.len();
    let m = (2 * n - 1).next_power_of_two();

    // `c[j] = exp(sign·πi·j²/n) = exp(sign·2πi·(j² mod 2n)/(2n))`. The phase has
    // period `2n` in `j²`, so the reduction is exact and keeps `j²` from ever
    // being formed as an `f64` — pocketfft's `bluestein` does the same, with the
    // same `sincos_2pibyn(2n)` table this [`twiddle`] evaluates pointwise.
    let two_n = 2 * n;
    let chirp: Vec<Complex> = (0..n)
        .map(|j| {
            let jj = ((j as u128 * j as u128) % two_n as u128) as usize;
            twiddle(jj, two_n, sign)
        })
        .collect();

    let mut a = vec![Complex::default(); m];
    for (slot, (&x, &c)) in a.iter_mut().zip(buf.iter().zip(&chirp)) {
        *slot = x * c;
    }

    // `conj(c)` is even in its index, so the negative lags fold onto `m - t`.
    let mut b = vec![Complex::default(); m];
    b[0] = chirp[0].conj();
    for t in 1..n {
        b[t] = chirp[t].conj();
        b[m - t] = chirp[t].conj();
    }

    fft_cyclic_convolve(&mut a, &mut b);

    let inv = 1.0 / m as f64;
    for (x, (&y, &c)) in buf.iter_mut().zip(a.iter().zip(&chirp)) {
        *x = y.scale(inv) * c;
    }
}

/// Cyclic convolution of two equal power-of-two-length buffers, result in `a`,
/// unnormalized by `a.len()` (the caller folds that factor in).
fn fft_cyclic_convolve(a: &mut [Complex], b: &mut [Complex]) {
    transform_1d_unscaled(a, false);
    transform_1d_unscaled(b, false);
    for (x, &y) in a.iter_mut().zip(b.iter()) {
        *x = *x * y;
    }
    transform_1d_unscaled(a, true);
}

/// `itk::PocketFFTCommon::Transform1D(data, len, forward, /*scale=*/1)`: an
/// unnormalized transform whose kernel is `exp(-2πi nk/N)` when `forward` and
/// `exp(+2πi nk/N)` otherwise, matching pocketfft's `c2c` convention.
///
/// `N4BiasFieldCorrectionImageFilter::SharpenImage` needs both signs and
/// neither normalization: it runs `Transform1D(..., false, 1)` where a
/// textbook forward DFT would sit and `Transform1D(..., true, 1)` where the
/// inverse would, so a round trip comes back scaled by `N`. That scale cancels
/// in the `E(u|v)` numerator/denominator ratio, which is the only thing
/// `SharpenImage` reads out.
pub(crate) fn transform_1d_unnormalized(buf: &mut [Complex], forward: bool) {
    transform_1d_unscaled(buf, !forward);
}

/// The unscaled kernel, with `positive_exponent` selecting the twiddle sign.
///
/// Dispatches on the length: [`fft_fast`] when every prime factor is at most
/// 11, [`fft_bluestein`] otherwise. Both compute the same DFT.
fn transform_1d_unscaled(buf: &mut [Complex], positive_exponent: bool) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    let sign = if positive_exponent { 1.0 } else { -1.0 };
    if is_fast_length(n) {
        let inp = buf.to_vec();
        fft_fast(&inp, 1, buf, n, sign);
    } else {
        fft_bluestein(buf, sign);
    }
}

/// Which 1-D kernel a separable pass runs. The two agree to the last few ulps
/// and disagree below that, so the choice is a bit-level commitment and is made
/// per call site, never by default.
///
/// # Why there are two
///
/// [`Planned`](LineKernel::Planned) is `rustfft`, and it is the faster kernel by
/// roughly 3x: SIMD butterflies against this port's scalar ones. But rustfft
/// derives its roots of unity from `cos`/`sin` of `-2π·j/n`
/// (`rustfft::twiddles::compute_twiddle`), where pocketfft — and therefore
/// [`Exact`](LineKernel::Exact), see [`radix_twiddles`] — hardcodes the
/// correctly-rounded literals. `cos(2π/3)` is the standing example: rustfft gets
/// `-0.4999999999999998`, two ulps from the exact `-0.5`.
///
/// Two ulps of twiddle is nothing next to a transform's own round-off, and for
/// [`transform_r2c`]/[`transform_c2r`] — whose only caller is
/// `fft_convolution`, an approximation of a spatial convolution to begin with —
/// it is invisible. It is *not* invisible to a filter that round-trips through
/// the spectrum and casts back to an integer pixel type: with exact twiddles the
/// delta-kernel round trip of `itkInverseDeconvolutionImageFilter` returns
/// `20.0` and truncates to `20`, and with rustfft's it returns
/// `19.99999999999999289` and truncates to `19`. That is a visible, wrong pixel,
/// and it is what `deconvolution::tests::output_keeps_the_input_pixel_type_and_
/// geometry` pins.
///
/// The truncation is not this port's choice to revisit: ITK narrows a float
/// result into an integer pixel with a plain `static_cast` on exactly this path
/// (`itkExtractImageFilter.hxx:270` → `itkImageAlgorithm.hxx:46`), and rounding
/// is opt-in there (a separate `RoundImageFilter`). Ledger §2.155. So the last
/// bits of a spectrum *are* load-bearing when the output is integral, and that
/// is what confines the fast kernel to the real-input pair.
///
/// So: the complex-spectrum API that filters round-trip through keeps the exact
/// kernel, and the real-input pair — the one that carries the volume-sized cost
/// — takes the fast one. Speed where the bits do not reach the output, exactness
/// where they do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LineKernel {
    /// This port's own butterflies on pocketfft's hardcoded twiddles.
    Exact,
    /// `rustfft`'s planner: SIMD butterflies on computed twiddles.
    Planned,
}

/// `rustfft`'s plan for a length-`n` complex transform, `positive_exponent`
/// selecting the sign.
///
/// rustfft calls `exp(-2πi jk/n)` *forward* and `exp(+2πi jk/n)` *inverse*, and
/// normalizes neither — the convention this module already had. The plan is
/// `Send + Sync` and transforms behind `&self`, so one plan serves every line of
/// an axis on every worker.
fn plan_c2c(planner: &mut FftPlanner<f64>, n: usize, positive_exponent: bool) -> Arc<dyn Fft<f64>> {
    if positive_exponent {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    }
}

/// Unscaled separable transform of `data` along `axes` only, laid out
/// first-index-fastest with extent `size`, on the [`LineKernel::Exact`] kernel.
///
/// `forward` selects the `exp(-2πi …)` kernel. This is pocketfft's `c2c` with
/// `fct = 1` over a subset of axes — the shape `general_c2r` uses when it
/// transforms every axis but the halved one
/// (`pocketfft_hdronly.h:3728`).
pub(crate) fn transform_axes(data: &mut [Complex], size: &[usize], axes: &[usize], forward: bool) {
    transform_axes_with(data, size, axes, forward, LineKernel::Exact);
}

/// [`transform_axes`], with the 1-D kernel named. See [`LineKernel`] for what
/// the choice costs.
fn transform_axes_with(
    data: &mut [Complex],
    size: &[usize],
    axes: &[usize],
    forward: bool,
    kernel: LineKernel,
) {
    let total: usize = size.iter().product();
    debug_assert_eq!(data.len(), total);
    if total == 0 {
        return;
    }

    // Parallel over lines ([`parallel::for_each_line_mut`]). Each 1-D transform
    // reads and writes only its own line, so lines are independent; the
    // butterflies *within* a line — the only place complex arithmetic
    // accumulates — run in an order fixed by the length alone, whichever kernel
    // it is. Output is bit-identical to the sequential pass at any worker count.
    // The axes stay sequential (each consumes the previous axis's result), and
    // the line and scratch buffers are per-task rather than shared.
    let mut planner = FftPlanner::<f64>::new();
    for &axis in axes {
        let len = size[axis];
        if len < 2 {
            continue;
        }
        match kernel {
            LineKernel::Exact => parallel::for_each_line_mut(
                data,
                size,
                axis,
                || vec![Complex::default(); len],
                |line, mut slot| {
                    for (t, v) in line.iter_mut().enumerate() {
                        *v = slot.get(t);
                    }
                    transform_1d_unscaled(line, !forward);
                    for (t, &v) in line.iter().enumerate() {
                        slot.set(t, v);
                    }
                },
            ),
            // The plan holds this length's twiddle tables: built once per axis,
            // shared across every line of it, because planning it per line would
            // cost more than the transform.
            LineKernel::Planned => {
                let fft = plan_c2c(&mut planner, len, !forward);
                let scratch_len = fft.get_inplace_scratch_len();
                parallel::for_each_line_mut(
                    data,
                    size,
                    axis,
                    || {
                        (
                            vec![Complex64::default(); len],
                            vec![Complex64::default(); scratch_len],
                        )
                    },
                    |(line, scratch), mut slot| {
                        for (t, v) in line.iter_mut().enumerate() {
                            *v = slot.get(t).to_c64();
                        }
                        fft.process_with_scratch(line, scratch);
                        for (t, &v) in line.iter().enumerate() {
                            slot.set(t, Complex::from_c64(v));
                        }
                    },
                );
            }
        }
    }
}

/// Separable N-dimensional transform of `data`, laid out first-index-fastest
/// with extent `size`.
///
/// The inverse pass divides by the total pixel count, as ITK's inverse FFT
/// filters do (`itkPocketFFTInverseFFTImageFilter.hxx:57`).
pub(crate) fn transform_nd(data: &mut [Complex], size: &[usize], inverse: bool) {
    let total: usize = size.iter().product();
    debug_assert_eq!(data.len(), total);
    if total == 0 {
        return;
    }
    let axes: Vec<usize> = (0..size.len()).collect();
    transform_axes(data, size, &axes, !inverse);
    if inverse {
        let scale = 1.0 / total as f64;
        parallel::for_each_mut(data, |_, x| *x = x.scale(scale));
    }
}

/// Extent of the halved axis in a half-spectrum: `n / 2 + 1` bins.
///
/// Both parities in one expression, deliberately. For even `n` the last bin is
/// Nyquist (`n / 2`); for odd `n` there is no Nyquist bin and the last is
/// `(n - 1) / 2`. Nothing downstream special-cases the two — see
/// [`transform_c2r`], whose mirror `n - x` lands inside `0 .. n / 2 + 1` for
/// every `x` above the half either way.
pub(crate) fn half_extent(n: usize) -> usize {
    n / 2 + 1
}

/// The extent of the half-spectrum of an image of extent `size`: axis 0 halved,
/// every other axis full.
pub(crate) fn half_size(size: &[usize]) -> Vec<usize> {
    let mut half = size.to_vec();
    if let Some(n0) = half.first_mut() {
        *n0 = half_extent(*n0);
    }
    half
}

/// Forward transform of **real** data, keeping only the half of the spectrum that
/// is not redundant: extent `n0 / 2 + 1` along axis 0, full along the rest.
///
/// A real signal's spectrum is conjugate-symmetric, so the discarded half is
/// determined by the half that is kept — computing, storing and multiplying it is
/// work for a result already in hand. This is what ITK's R2C does, and it halves
/// both the flops and the memory traffic of every axis after the first.
///
/// Pairs with [`transform_c2r`]. The product of two half-spectra is the
/// half-spectrum of the product, because the multiply is elementwise: the halving
/// costs the caller nothing but the extent it indexes with.
pub(crate) fn transform_r2c(real: &[f64], size: &[usize]) -> Vec<Complex> {
    let total: usize = size.iter().product();
    debug_assert_eq!(real.len(), total);
    let hsize = half_size(size);
    let htotal: usize = hsize.iter().product();
    let mut half = vec![Complex::default(); htotal];
    if total == 0 {
        return half;
    }

    let n0 = size[0];
    let h = hsize[0];

    // Axis 0, on the real input: a real-to-complex transform, which *produces*
    // the `h = n0 / 2 + 1` bins rather than computing all `n0` of them and
    // discarding half. Lines along axis 0 are contiguous in both buffers, so a
    // line of `half` starting at `start` mirrors the line of `real` starting at
    // `start / h * n0` — see `Line::start`.
    //
    // A length-1 axis has no transform to take: the single bin is the sample,
    // which is also what the complex kernel's `n < 2` early return gave.
    if n0 < 2 {
        parallel::for_each_mut(&mut half, |i, v| *v = Complex::new(real[i], 0.0));
    } else {
        let r2c = RealFftPlanner::<f64>::new().plan_fft_forward(n0);
        let scratch_len = r2c.get_scratch_len();
        parallel::for_each_line_mut(
            &mut half,
            &hsize,
            0,
            || {
                (
                    vec![0.0f64; n0],
                    vec![Complex64::default(); h],
                    vec![Complex64::default(); scratch_len],
                )
            },
            |(line, spectrum, scratch), mut slot| {
                let base = slot.start() / h * n0;
                line.copy_from_slice(&real[base..base + n0]);
                r2c.process_with_scratch(line, spectrum, scratch)
                    .expect("r2c: the buffers are the planner's own lengths");
                for (t, &v) in spectrum.iter().enumerate() {
                    slot.set(t, Complex::from_c64(v));
                }
            },
        );
    }

    // The remaining axes, on the halved array: the same transform the full
    // spectrum would get, over half as many lines.
    let axes: Vec<usize> = (1..size.len()).collect();
    transform_axes_with(&mut half, &hsize, &axes, true, LineKernel::Planned);
    half
}

/// Inverse of [`transform_r2c`]: the real signal whose half-spectrum is `half`.
///
/// Equals the real part of the full inverse transform of the conjugate-symmetric
/// extension of `half`, scaled by `1 / total` exactly as [`transform_nd`] scales
/// its inverse.
///
/// The axes must be inverted in this order — every full axis first, the halved
/// one last — because only the halved axis is stored in half, and it is the last
/// pass that turns complex into real. That reordering is also why this is not
/// bit-identical to inverting a full spectrum: the same additions happen in a
/// different order.
///
/// # The symmetry this reads, and when it holds
///
/// Once the full axes are inverted, a line along axis 0 is the spectrum of a
/// *real* signal in `x` alone — for each spatial `(y, z, …)` separately — so its
/// missing bins are `G[n0 - x] = conj(G[x])` **on that same line**. That is
/// exactly the hypothesis a C2R transform is entitled to make, and it is why the
/// full axes must be inverted *first*: the mirror in the other axes belongs to
/// the untransformed spectrum, and inverting those axes is what consumes it.
/// Handing C2R a half-spectrum whose other axes are still in the frequency domain
/// would reconstruct the wrong bins.
///
/// Both parities are the planner's business — it dispatches on `n0`'s — and an
/// even `n0`'s Nyquist bin is the last bin of the half either way. The one thing
/// the caller owes it is a DC bin (and Nyquist, when `n0` is even) with no
/// imaginary part; see the note at the call.
pub(crate) fn transform_c2r(half: &mut [Complex], size: &[usize]) -> Vec<f64> {
    let total: usize = size.iter().product();
    let hsize = half_size(size);
    let htotal: usize = hsize.iter().product();
    debug_assert_eq!(half.len(), htotal);
    let mut real = vec![0.0f64; total];
    if total == 0 {
        return real;
    }

    let n0 = size[0];
    let h = hsize[0];

    // Every axis but the halved one: half as many lines as the full spectrum has.
    let axes: Vec<usize> = (1..size.len()).collect();
    transform_axes_with(half, &hsize, &axes, false, LineKernel::Planned);

    let scale = 1.0 / total as f64;
    let half = &*half;

    // Axis 0: a complex-to-real transform of the line's own half. Lines are
    // contiguous along axis 0 in both buffers, so the line of `real` at `start`
    // is the line of `half` at `start / n0 * h`.
    if n0 < 2 {
        parallel::for_each_mut(&mut real, |i, v| *v = half[i].re * scale);
        return real;
    }
    let c2r = RealFftPlanner::<f64>::new().plan_fft_inverse(n0);
    let scratch_len = c2r.get_scratch_len();
    parallel::for_each_line_mut(
        &mut real,
        size,
        0,
        || {
            (
                vec![Complex64::default(); h],
                vec![0.0f64; n0],
                vec![Complex64::default(); scratch_len],
            )
        },
        |(spectrum, line, scratch), mut slot| {
            let base = slot.start() / n0 * h;
            for (t, v) in spectrum.iter_mut().enumerate() {
                *v = half[base + t].to_c64();
            }
            // The DC bin — and the Nyquist bin, when `n0` is even — are their own
            // conjugate, so they are real in exact arithmetic. Round-off leaves an
            // imaginary residue there, and C2R refuses a spectrum that carries one.
            // Zero it: `realfft` clears both itself before transforming and reports
            // the clearing as an `Err` afterwards
            // (`ComplexToRealEven::process_with_scratch`), so doing it here moves no
            // output bit and turns a spurious `Err` into the `Ok` the lengths already
            // guarantee.
            spectrum[0].im = 0.0;
            if n0.is_multiple_of(2) {
                spectrum[h - 1].im = 0.0;
            }
            c2r.process_with_scratch(spectrum, line, scratch)
                .expect("c2r: the buffers are the planner's own lengths, DC and Nyquist are real");
            for (t, &v) in line.iter().enumerate() {
                slot.set(t, v * scale);
            }
        },
    );
    real
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The normalized 1-D pair: `inverse` conjugates the kernel and scales by
    /// `1 / len`. `transform_nd` is this, applied axis by axis.
    fn transform_1d(buf: &mut [Complex], inverse: bool) {
        transform_1d_unscaled(buf, inverse);
        if inverse {
            let scale = 1.0 / buf.len() as f64;
            for x in buf.iter_mut() {
                *x = x.scale(scale);
            }
        }
    }

    /// The naive `O(n²)` DFT, the definition every fast path must reproduce.
    fn naive_dft(input: &[Complex], sign: f64) -> Vec<Complex> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex::default();
                for (t, x) in input.iter().enumerate() {
                    acc = acc + *x * Complex::unit(sign * 2.0 * PI * (k * t) as f64 / n as f64);
                }
                acc
            })
            .collect()
    }

    fn sample(n: usize) -> Vec<Complex> {
        (0..n)
            .map(|i| Complex::new((i as f64 * 0.7).sin() + 0.25, (i as f64 * 0.3).cos() - 0.4))
            .collect()
    }

    #[test]
    fn is_prime_matches_itk_math() {
        let expected = [2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        for n in 0..=50usize {
            assert_eq!(is_prime(n), expected.contains(&n), "is_prime({n})");
        }
    }

    /// `GreatestPrimeFactor`'s two degenerate inputs: the `v <= n` loop never
    /// runs, so both answer `2`.
    #[test]
    fn greatest_prime_factor_of_zero_and_one_is_two() {
        assert_eq!(greatest_prime_factor(0), 2);
        assert_eq!(greatest_prime_factor(1), 2);
    }

    #[test]
    fn greatest_prime_factor_of_small_numbers() {
        for (n, want) in [
            (2usize, 2usize),
            (3, 3),
            (4, 2),
            (12, 3),
            (13, 13),
            (14, 7),
            (97, 97),
            (100, 5),
            (128, 2),
            (1024, 2),
        ] {
            assert_eq!(greatest_prime_factor(n), want, "gpf({n})");
        }
    }

    /// `padded_length` is minimal and valid, checked against the transcribed
    /// `Math::GreatestPrimeFactor` for every axis size a small image can have.
    #[test]
    fn padded_length_matches_itk_fft_pad_search() {
        for n in 0..=200usize {
            let m = padded_length(n);
            assert!(m >= n, "padded_length({n}) = {m} shrank the axis");
            assert!(
                greatest_prime_factor(m) <= DEFAULT_SIZE_GREATEST_PRIME_FACTOR,
                "padded_length({n}) = {m} has a prime factor above 11"
            );
            for candidate in n..m {
                assert!(
                    greatest_prime_factor(candidate) > DEFAULT_SIZE_GREATEST_PRIME_FACTOR,
                    "padded_length({n}) overshot: {candidate} was already valid"
                );
            }
        }
    }

    /// The sizes the ledger row and the task pin: an 11-smooth length is left
    /// alone, `13` walks up to `14 = 2·7`, and a prime above 11 never survives.
    #[test]
    fn padded_length_pins_the_documented_sizes() {
        for (n, want) in [
            (0usize, 0usize),
            (1, 1),
            (13, 14),
            (17, 18),
            (23, 24),
            (97, 98),
            (100, 100),
            (127, 128),
            (129, 132),
        ] {
            assert_eq!(padded_length(n), want, "padded_length({n})");
        }
    }

    /// `SizeGreatestPrimeFactor == 1` grows the axis to an even size — the
    /// behaviour the filter's own doc-comment denies (ledger §2.109).
    #[test]
    fn size_greatest_prime_factor_of_one_pads_to_an_even_size() {
        for n in 0..8usize {
            assert_eq!(fft_pad_size(n, 1), n % 2, "fft_pad_size({n}, 1)");
        }
    }

    /// `0` — and, through the `int` -> `SizeValueType` conversion, any negative
    /// SimpleITK argument — disables padding entirely.
    #[test]
    fn size_greatest_prime_factor_of_zero_disables_padding() {
        for n in 0..40usize {
            assert_eq!(fft_pad_size(n, 0), 0);
            assert_eq!(fft_pad_size(n, usize::MAX), 0);
        }
    }

    /// `SizeGreatestPrimeFactor == 2` is the power-of-two rule this module used
    /// to hardcode; it must still fall out of the general search.
    #[test]
    fn size_greatest_prime_factor_of_two_is_the_next_power_of_two() {
        for n in 1..=256usize {
            assert_eq!(n + fft_pad_size(n, 2), n.next_power_of_two(), "n = {n}");
        }
    }

    // ---- twiddle accuracy -------------------------------------------------

    /// Every hardcoded radix root is a root of unity to within an ulp, and the
    /// ones with exact `f64` representations are exact.
    #[test]
    fn radix_roots_are_roots_of_unity() {
        for radix in [3usize, 5, 7, 11] {
            for t in 0..radix {
                let root = radix_root(radix, t, -1.0);
                let want_re = (2.0 * PI * t as f64 / radix as f64).cos();
                let want_im = -(2.0 * PI * t as f64 / radix as f64).sin();
                assert!(
                    (root.re - want_re).abs() < 1e-15 && (root.im - want_im).abs() < 1e-15,
                    "radix {radix}, t {t}"
                );
                assert!(
                    (root.abs() - 1.0).abs() <= f64::EPSILON,
                    "radix {radix}, t {t}: |root| = {}",
                    root.abs()
                );
            }
        }
        // `cos(2π/3) == -0.5` exactly; the naive `(2π/3).cos()` is two ulps out.
        assert_eq!(radix_root(3, 1, -1.0).re, -0.5);
        assert_eq!(radix_root(3, 2, -1.0).re, -0.5);
        assert!(
            (2.0 * PI / 3.0).cos() != -0.5,
            "the naive value must differ"
        );
    }

    /// A radix root raised to its own radix returns to `1`.
    #[test]
    fn radix_roots_close_the_cycle() {
        for radix in [3usize, 5, 7, 11] {
            for t in 1..radix {
                let root = radix_root(radix, t, -1.0);
                let mut acc = Complex::new(1.0, 0.0);
                for _ in 0..radix {
                    acc = acc * root;
                }
                assert!((acc - Complex::new(1.0, 0.0)).abs() < 1e-14, "{radix}^{t}");
            }
        }
    }

    /// `twiddle` reproduces `exp(sign·2πi·i/n)` for every `i`, exactly at the
    /// quadrant boundaries and to an ulp elsewhere.
    #[test]
    fn twiddle_matches_the_unreduced_exponential() {
        for n in [3usize, 4, 8, 12, 97, 100, 194] {
            for i in 0..n {
                for sign in [-1.0f64, 1.0] {
                    let got = twiddle(i, n, sign);
                    let want = Complex::unit(sign * 2.0 * PI * i as f64 / n as f64);
                    assert!(
                        (got.re - want.re).abs() < 1e-14 && (got.im - want.im).abs() < 1e-14,
                        "n {n}, i {i}, sign {sign}: {got:?} vs {want:?}"
                    );
                }
            }
        }
        assert_eq!(twiddle(0, 7, -1.0), Complex::new(1.0, 0.0));
        // Quadrant boundaries: n = 4 gives i, -1, -i with no rounding at all.
        assert_eq!(twiddle(1, 4, 1.0).im, 1.0);
        assert_eq!(twiddle(2, 4, 1.0).re, -1.0);
        assert_eq!(twiddle(3, 4, 1.0).im, -1.0);
        // `i` is reduced mod `n`, so a full turn is the identity.
        assert_eq!(twiddle(97, 97, -1.0), Complex::new(1.0, 0.0));
    }

    #[test]
    fn fast_lengths_are_exactly_the_eleven_smooth_ones() {
        assert!(is_fast_length(1));
        for n in [2usize, 3, 4, 5, 7, 11, 12, 100, 128, 1024, 1155] {
            assert!(is_fast_length(n), "{n} should be fast");
        }
        for n in [13usize, 17, 26, 97, 169] {
            assert!(!is_fast_length(n), "{n} should need Bluestein");
        }
    }

    // ---- analytically known transforms ------------------------------------

    /// `x[j] = δ[j]` transforms to the all-ones spectrum, at every length and
    /// on both the fast and the Bluestein path.
    #[test]
    fn forward_of_a_delta_is_all_ones() {
        for n in [1usize, 2, 8, 12, 97, 100] {
            let mut buf = vec![Complex::default(); n];
            buf[0] = Complex::new(1.0, 0.0);
            transform_1d(&mut buf, false);
            for (k, x) in buf.iter().enumerate() {
                assert!(
                    (x.re - 1.0).abs() < 1e-12 && x.im.abs() < 1e-12,
                    "n = {n}, k = {k}: {x:?}"
                );
            }
        }
    }

    /// A constant `c` transforms to `n·c` at bin 0 and zero everywhere else.
    #[test]
    fn forward_of_a_constant_is_a_delta_of_weight_n() {
        for n in [1usize, 12, 97, 100] {
            let mut buf = vec![Complex::new(2.5, 0.0); n];
            transform_1d(&mut buf, false);
            assert!((buf[0].re - 2.5 * n as f64).abs() < 1e-10, "n = {n}");
            assert!(buf[0].im.abs() < 1e-10, "n = {n}");
            for (k, x) in buf.iter().enumerate().skip(1) {
                assert!(x.abs() < 1e-9, "n = {n}, k = {k}: {x:?}");
            }
        }
    }

    /// A single tone `exp(2πi·k₀·j/n)` transforms to `n` at bin `k₀` alone,
    /// under this module's `exp(-2πi…)` forward kernel.
    #[test]
    fn forward_of_a_single_tone_is_a_delta_at_its_bin() {
        for n in [12usize, 97, 100] {
            for k0 in [0usize, 1, 3, n / 2, n - 1] {
                let mut buf: Vec<Complex> = (0..n)
                    .map(|j| Complex::unit(2.0 * PI * (k0 * j % n) as f64 / n as f64))
                    .collect();
                transform_1d(&mut buf, false);
                for (k, x) in buf.iter().enumerate() {
                    let want = if k == k0 { n as f64 } else { 0.0 };
                    assert!(
                        (x.re - want).abs() < 1e-8 && x.im.abs() < 1e-8,
                        "n = {n}, k0 = {k0}, k = {k}: {x:?}"
                    );
                }
            }
        }
    }

    /// Parseval: `Σ|x[j]|² = (1/n)·Σ|X[k]|²`, at 11-smooth and Bluestein lengths.
    #[test]
    fn forward_satisfies_parseval() {
        for n in [2usize, 12, 13, 97, 100, 101] {
            let input = sample(n);
            let energy: f64 = input.iter().map(|x| x.norm()).sum();
            let mut buf = input;
            transform_1d(&mut buf, false);
            let spectral: f64 = buf.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
            assert!(
                (energy - spectral).abs() <= 1e-9 * energy.max(1.0),
                "n = {n}: {energy} vs {spectral}"
            );
        }
    }

    /// Both paths agree with the definition, for both signs.
    #[test]
    fn forward_1d_agrees_with_the_naive_dft() {
        for n in [2usize, 3, 5, 7, 11, 12, 13, 17, 97, 100, 121] {
            let input = sample(n);
            for (positive, sign) in [(false, -1.0), (true, 1.0)] {
                let want = naive_dft(&input, sign);
                let mut got = input.clone();
                transform_1d_unscaled(&mut got, positive);
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g.re - w.re).abs() < 1e-9 && (g.im - w.im).abs() < 1e-9,
                        "n = {n}, sign = {sign}, k = {k}: {g:?} vs {w:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn round_trip_1d_restores_input() {
        for n in [1usize, 2, 12, 13, 97, 100, 128] {
            let original = sample(n);
            let mut buf = original.clone();
            transform_1d(&mut buf, false);
            transform_1d(&mut buf, true);
            for (k, (a, b)) in buf.iter().zip(&original).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                    "n = {n}, k = {k}"
                );
            }
        }
    }

    #[test]
    fn round_trip_nd_restores_input() {
        for size in [vec![4usize, 8, 2], vec![12, 5], vec![13, 3], vec![97]] {
            let total: usize = size.iter().product();
            let original = sample(total);
            let mut buf = original.clone();
            transform_nd(&mut buf, &size, false);
            transform_nd(&mut buf, &size, true);
            for (i, (a, b)) in buf.iter().zip(&original).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10,
                    "size = {size:?}, i = {i}"
                );
            }
        }
    }

    /// A 2-D transform is the composition of two 1-D ones, so the separable
    /// implementation must match a transform of the rows followed by the
    /// columns done by hand.
    #[test]
    fn transform_nd_is_separable() {
        let size = [3usize, 5];
        let mut got = sample(15);
        let want = {
            let mut data = got.clone();
            // Rows (axis 0, stride 1).
            for c in 0..size[1] {
                let mut line: Vec<Complex> = (0..size[0]).map(|r| data[c * size[0] + r]).collect();
                line = naive_dft(&line, -1.0);
                for (r, &v) in line.iter().enumerate() {
                    data[c * size[0] + r] = v;
                }
            }
            // Columns (axis 1, stride 3).
            for r in 0..size[0] {
                let mut line: Vec<Complex> = (0..size[1]).map(|c| data[c * size[0] + r]).collect();
                line = naive_dft(&line, -1.0);
                for (c, &v) in line.iter().enumerate() {
                    data[c * size[0] + r] = v;
                }
            }
            data
        };
        transform_nd(&mut got, &size, false);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10,
                "i = {i}"
            );
        }
    }

    /// `transform_axes` leaves the axes it is not given untouched: transforming
    /// axis 1 alone, then axis 0 alone, equals transforming both.
    #[test]
    fn transform_axes_composes_one_axis_at_a_time() {
        let size = [7usize, 4];
        let original = sample(28);
        let mut both = original.clone();
        transform_axes(&mut both, &size, &[0, 1], true);
        let mut split = original;
        transform_axes(&mut split, &size, &[1], true);
        transform_axes(&mut split, &size, &[0], true);
        for (a, b) in split.iter().zip(&both) {
            assert!((a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn length_one_axis_is_the_identity() {
        let mut buf = vec![Complex::new(3.0, -1.0)];
        transform_1d(&mut buf, false);
        assert_eq!(buf[0], Complex::new(3.0, -1.0));
        transform_1d(&mut buf, true);
        assert_eq!(buf[0], Complex::new(3.0, -1.0));
    }

    /// The unnormalized pair `SharpenImage` uses: a round trip scales by `n`,
    /// at a Bluestein length as much as at a fast one.
    #[test]
    fn unnormalized_round_trip_scales_by_the_length() {
        for n in [16usize, 13] {
            let original = sample(n);
            let mut buf = original.clone();
            transform_1d_unnormalized(&mut buf, false);
            transform_1d_unnormalized(&mut buf, true);
            for (a, b) in buf.iter().zip(&original) {
                let want = b.scale(n as f64);
                assert!((a.re - want.re).abs() < 1e-9 && (a.im - want.im).abs() < 1e-9);
            }
        }
    }
}
