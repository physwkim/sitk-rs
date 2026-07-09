//! Crate-private complex DFT backing [`crate::convolution::fft_convolution`].
//!
//! ITK hands its transforms to a pluggable FFT backend
//! (`itkVnlForwardFFTImageFilter` → `vnl_fft_base`, in
//! `Modules/ThirdParty/VNL`). This workspace takes no new dependency, so the
//! transform lives here, and no complex type escapes this module.
//!
//! **Why radix-2 is enough.** `FFTConvolutionImageFilter`'s constructor seeds
//! `m_SizeGreatestPrimeFactor` from
//! `FFTFilterType::New()->GetSizeGreatestPrimeFactor()`
//! (itkFFTConvolutionImageFilter.hxx:37-40), and with the vnl backend that
//! resolves to the base-class value `2`
//! (itkRealToHalfHermitianForwardFFTImageFilter.hxx:98-104 —
//! `itkVnlForwardFFTImageFilter` does not override it).
//! `itkFFTPadImageFilter.hxx:52-72` then grows each axis by the smallest
//! `padSize` for which `Math::GreatestPrimeFactor(size + padSize) <= 2`, i.e.
//! to a power of two. So [`padded_length`] and this transform's supported
//! lengths coincide exactly: every length the filter can ask for is a power of
//! two, and a mixed-radix / Bluestein fallback would be unreachable code.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

/// A complex number. Deliberately crate-private: the public convolution API is
/// real-valued in both directions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Complex {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl Complex {
    pub(crate) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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

/// The length `itkFFTPadImageFilter` would pad an axis of `n` pixels up to,
/// for `SizeGreatestPrimeFactor == 2`: the smallest `m >= n` whose greatest
/// prime factor is at most 2 (itkFFTPadImageFilter.hxx:55-63).
///
/// `Math::GreatestPrimeFactor(1)` returns 2 (itkMath.h:798-814, the `v <= n`
/// loop never runs), so `n == 1` needs no padding — which is what
/// `usize::next_power_of_two` yields.
pub(crate) fn padded_length(n: usize) -> usize {
    n.next_power_of_two()
}

/// In-place radix-2 Cooley-Tukey transform of a power-of-two-length buffer.
///
/// `inverse` conjugates the twiddle factors and scales by `1 / len`, so
/// `transform_1d(inverse=true)` inverts `transform_1d(inverse=false)`.
fn transform_1d(buf: &mut [Complex], inverse: bool) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two(), "radix-2 needs a power-of-two length");
    if n < 2 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let angle = sign * 2.0 * PI / len as f64;
        let step = Complex::new(angle.cos(), angle.sin());
        let half = len / 2;
        for block in (0..n).step_by(len) {
            // Restarted from exactly 1 in every block, so the recurrence's
            // drift is bounded by `half` multiplies rather than by `n`.
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[block + k];
                let v = buf[block + k + half] * w;
                buf[block + k] = u + v;
                buf[block + k + half] = u - v;
                w = w * step;
            }
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for x in buf.iter_mut() {
            x.re *= scale;
            x.im *= scale;
        }
    }
}

/// Separable N-dimensional transform of `data`, laid out first-index-fastest
/// with extent `size` (every component a power of two).
///
/// Each axis is scaled by `1 / size[d]` on the inverse pass, so the round trip
/// divides by the total pixel count, as ITK's inverse FFT filters do.
pub(crate) fn transform_nd(data: &mut [Complex], size: &[usize], inverse: bool) {
    let total: usize = size.iter().product();
    debug_assert_eq!(data.len(), total);
    if total == 0 {
        return;
    }

    let mut stride = 1usize;
    for &len in size {
        if len > 1 {
            let outer = total / (stride * len);
            let mut line = vec![Complex::default(); len];
            for o in 0..outer {
                for k in 0..stride {
                    let start = o * stride * len + k;
                    for (t, slot) in line.iter_mut().enumerate() {
                        *slot = data[start + t * stride];
                    }
                    transform_1d(&mut line, inverse);
                    for (t, &v) in line.iter().enumerate() {
                        data[start + t * stride] = v;
                    }
                }
            }
        }
        stride *= len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Math::GreatestPrimeFactor(n) <= 2`, transcribed from itkMath.h:798-814.
    fn greatest_prime_factor_at_most_two(n: usize) -> bool {
        let mut n = n;
        let mut v = 2usize;
        while v <= n {
            if n % v == 0 && (2..v).all(|d| v % d != 0) {
                n /= v;
            } else {
                v += 1;
            }
        }
        v <= 2
    }

    #[test]
    fn padded_length_matches_itk_fft_pad_search() {
        for n in 1..=64usize {
            let m = padded_length(n);
            assert!(m >= n, "padded_length({n}) = {m} shrank the axis");
            assert!(
                greatest_prime_factor_at_most_two(m),
                "padded_length({n}) = {m} has a prime factor > 2"
            );
            // Minimality: nothing between n and m is acceptable to ITK's loop.
            for candidate in n..m {
                assert!(
                    !greatest_prime_factor_at_most_two(candidate),
                    "padded_length({n}) overshot: {candidate} was already valid"
                );
            }
        }
    }

    #[test]
    fn forward_of_a_delta_is_all_ones() {
        let mut buf = vec![Complex::default(); 8];
        buf[0] = Complex::new(1.0, 0.0);
        transform_1d(&mut buf, false);
        for x in &buf {
            assert!((x.re - 1.0).abs() < 1e-12 && x.im.abs() < 1e-12);
        }
    }

    #[test]
    fn round_trip_1d_restores_input() {
        let original: Vec<Complex> = (0..16)
            .map(|i| Complex::new(i as f64 * 0.5 - 3.0, (i % 3) as f64))
            .collect();
        let mut buf = original.clone();
        transform_1d(&mut buf, false);
        transform_1d(&mut buf, true);
        for (a, b) in buf.iter().zip(&original) {
            assert!((a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12);
        }
    }

    #[test]
    fn round_trip_nd_restores_input() {
        let size = [4usize, 8, 2];
        let total: usize = size.iter().product();
        let original: Vec<Complex> = (0..total)
            .map(|i| Complex::new((i * 7 % 13) as f64, (i % 5) as f64))
            .collect();
        let mut buf = original.clone();
        transform_nd(&mut buf, &size, false);
        transform_nd(&mut buf, &size, true);
        for (a, b) in buf.iter().zip(&original) {
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

    #[test]
    fn forward_1d_agrees_with_the_naive_dft() {
        let n = 8usize;
        let input: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut fast = input.clone();
        transform_1d(&mut fast, false);
        for (k, got) in fast.iter().enumerate() {
            let mut want = Complex::default();
            for (t, x) in input.iter().enumerate() {
                let angle = -2.0 * PI * (k * t) as f64 / n as f64;
                want = want + *x * Complex::new(angle.cos(), angle.sin());
            }
            assert!((got.re - want.re).abs() < 1e-10 && (got.im - want.im).abs() < 1e-10);
        }
    }
}
