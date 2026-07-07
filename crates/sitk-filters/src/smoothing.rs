//! Gaussian smoothing by separable FIR convolution.
//!
//! `sigma` is given per dimension in **physical units** (matching ITK's
//! `SmoothingRecursiveGaussianImageFilter` default,
//! `SmoothingSigmasAreSpecifiedInPhysicalUnits = true`): along axis `d` the
//! Gaussian has standard deviation `sigma[d]`, so in index units it is
//! `σ_idx = sigma[d] / spacing[d]`. The 1-D kernel is sampled at integer index
//! offsets, `w[k] = exp(-k² / (2·σ_idx²))`, truncated at `⌈4·σ_idx⌉` and
//! normalized to sum 1; the boundary replicates the edge value (zero-flux),
//! matching ITK's "border value extends to infinity" convention. Axes are
//! filtered in sequence (separable). An axis with `sigma == 0` is left
//! untouched.
//!
//! This is a *result-faithful* Gaussian: it approximates the same continuous
//! Gaussian ITK's recursive filter does, but is not bit-identical to
//! `RecursiveGaussianImageFilter` (a Deriche/Farnebäck IIR). It is isolated
//! behind this seam so that the bit-exact recursive port
//! ([`recursive_gaussian`](crate::recursive_gaussian())), which shares this
//! signature, can replace it without touching callers.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::Image;

/// Gaussian-smooth `img` with a per-dimension physical-space `sigma`.
///
/// Errors if `sigma` has the wrong length or any value is negative.
pub fn smooth_gaussian(img: &Image, sigma: &[f64]) -> Result<Image> {
    let dim = img.dimension();
    if sigma.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: sigma.len(),
        });
    }
    if sigma.iter().any(|&s| s < 0.0) {
        return Err(FilterError::InvalidSigma(sigma.to_vec()));
    }

    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let strides = strides(&size);
    let mut buf = img.to_f64_vec();

    for d in 0..dim {
        if sigma[d] <= 0.0 {
            continue;
        }
        let sigma_idx = sigma[d] / spacing[d];
        let (kernel, radius) = gaussian_kernel(sigma_idx);
        buf = convolve_axis(&buf, &size, &strides, d, &kernel, radius);
    }

    image_from_f64(img.pixel_id(), &size, img, &buf)
}

/// Symmetric discrete Gaussian kernel for an index-space `sigma_idx > 0`,
/// truncated at `⌈4·sigma_idx⌉` and normalized. Returns `(kernel, radius)` where
/// `kernel` has length `2·radius + 1` and tap `ki` is offset `ki − radius`.
fn gaussian_kernel(sigma_idx: f64) -> (Vec<f64>, usize) {
    let radius = (4.0 * sigma_idx).ceil().max(1.0) as usize;
    let denom = 2.0 * sigma_idx * sigma_idx;
    let mut kernel = vec![0.0f64; 2 * radius + 1];
    let mut sum = 0.0;
    for (ki, w) in kernel.iter_mut().enumerate() {
        let k = ki as f64 - radius as f64;
        *w = (-(k * k) / denom).exp();
        sum += *w;
    }
    for w in &mut kernel {
        *w /= sum;
    }
    (kernel, radius)
}

/// Convolve `buf` along axis `d` with `kernel` (edge-replicating boundary).
fn convolve_axis(
    buf: &[f64],
    size: &[usize],
    strides: &[usize],
    d: usize,
    kernel: &[f64],
    radius: usize,
) -> Vec<f64> {
    let stride = strides[d];
    let size_d = size[d] as isize;
    let mut out = vec![0.0f64; buf.len()];
    for (p, slot) in out.iter_mut().enumerate() {
        let coord_d = (p / stride) % size[d];
        let line_base = p - coord_d * stride;
        let mut acc = 0.0;
        for (ki, &w) in kernel.iter().enumerate() {
            let c =
                (coord_d as isize + ki as isize - radius as isize).clamp(0, size_d - 1) as usize;
            acc += w * buf[line_base + c * stride];
        }
        *slot = acc;
    }
    out
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::{Image, PixelId};

    #[test]
    fn zero_sigma_is_identity() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = smooth_gaussian(&img, &[0.0, 0.0]).unwrap();
        assert_eq!(out.to_f64_vec(), img.to_f64_vec());
    }

    #[test]
    fn constant_image_is_preserved() {
        // A normalized kernel + edge-replicating boundary preserves a constant.
        let img = Image::from_vec(&[8, 8], vec![5.0; 64]).unwrap();
        let out = smooth_gaussian(&img, &[2.0, 2.0]).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 5.0).abs() < 1e-12, "constant not preserved: {v}");
        }
    }

    #[test]
    fn smoothing_conserves_total_mass_of_an_interior_impulse() {
        // A single interior impulse away from the border keeps its total mass
        // (kernel sums to 1) and spreads symmetrically.
        let n = 21;
        let mut data = vec![0.0f64; n * n];
        data[10 * n + 10] = 100.0;
        let img = Image::from_vec(&[n, n], data).unwrap();
        let out = smooth_gaussian(&img, &[1.5, 1.5]).unwrap();
        let v = out.to_f64_vec();
        let total: f64 = v.iter().sum();
        assert!((total - 100.0).abs() < 1e-6, "mass not conserved: {total}");
        // Peak stays at the center and is reduced (spread out).
        let peak = v[10 * n + 10];
        assert!(peak < 100.0 && peak > 0.0);
        // Symmetric about the center along both axes.
        assert!((v[10 * n + 9] - v[10 * n + 11]).abs() < 1e-12);
        assert!((v[9 * n + 10] - v[11 * n + 10]).abs() < 1e-12);
    }

    #[test]
    fn physical_sigma_accounts_for_spacing() {
        // With spacing 2, a physical sigma of 2 is only 1 voxel of blur, so an
        // impulse spreads less than with spacing 1.
        let n = 21;
        let mut data = vec![0.0f64; n * n];
        data[10 * n + 10] = 100.0;

        let mut fine = Image::from_vec(&[n, n], data.clone()).unwrap();
        fine.set_spacing(&[1.0, 1.0]).unwrap();
        let mut coarse = Image::from_vec(&[n, n], data).unwrap();
        coarse.set_spacing(&[2.0, 2.0]).unwrap();

        let peak_fine = smooth_gaussian(&fine, &[2.0, 2.0]).unwrap().to_f64_vec()[10 * n + 10];
        let peak_coarse = smooth_gaussian(&coarse, &[2.0, 2.0]).unwrap().to_f64_vec()[10 * n + 10];
        assert!(
            peak_coarse > peak_fine,
            "coarser spacing should blur less: {peak_coarse} vs {peak_fine}"
        );
    }

    #[test]
    fn wrong_sigma_length_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            smooth_gaussian(&img, &[1.0]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn negative_sigma_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            smooth_gaussian(&img, &[-1.0, 1.0]),
            Err(FilterError::InvalidSigma(_))
        ));
    }
}
