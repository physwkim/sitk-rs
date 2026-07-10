//! `itk::FFTShiftImageFilter`
//! (`Modules/Filtering/FFT/include/itkFFTShiftImageFilter.hxx`): cyclically
//! shift an image so the zero-frequency corner of a Fourier transform lands
//! at the image center.
//!
//! Ported through the superclass that actually performs the wraparound copy,
//! `itk::CyclicShiftImageFilter`
//! (`Modules/Filtering/ImageGrid/include/itkCyclicShiftImageFilter.hxx`):
//! `FFTShiftImageFilter::GenerateData` computes `shift[i] = size[i] / 2`
//! (integer division), negated when `inverse` is set, then
//! `CyclicShiftImageFilter::DynamicThreadedGenerateData` reads each output
//! pixel from `input[(index[i] - outIdx[i] - shift[i]) mod size[i]]`, wrapped
//! into `[0, size[i])`. This crate's images always start at index 0
//! (there is no `LargestPossibleRegion` index offset concept here), so the
//! `outIdx` term is always zero and drops out of the port.
//!
//! `FFTShiftImageFilter.yaml` declares `pixel_types: NonLabelPixelIDTypeList`
//! (SimpleITK's full pixel-type list minus label maps, which for the C++
//! library includes complex pixel types); this crate's [`PixelId`] has no
//! complex variant at all, so [`fft_shift`] naturally covers every pixel type
//! it can represent -- the shift itself is pixel-type-agnostic index
//! permutation, identical to how [`crate::geometry::wrap_pad`] and
//! [`crate::grid_utility`] handle their own pure index math.
//!
//! For an *even*-sized axis, `size[i] / 2` and `-(size[i] / 2)` are congruent
//! mod `size[i]`, so `inverse` has no effect there. For an *odd*-sized axis
//! they differ by one slot, so a forward shift followed by an inverse shift
//! (or vice versa) is required to round-trip that axis -- exactly the class
//! doc's note: "applying this filter twice will not produce the same image as
//! the original one without using SetInverse(true) on one (and only one) of
//! the two filters."

use crate::error::Result;
use crate::image_from_f64;
use sitk_core::Image;

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `FFTShiftImageFilter`: cyclically shift every axis by `size[i] / 2`
/// (`inverse` negates the shift), moving the zero-frequency corner to the
/// image center. See the module docs for the odd-size/`inverse` interaction.
pub fn fft_shift(img: &Image, inverse: bool) -> Result<Image> {
    let size = img.size();
    let dim = size.len();
    let strides = strides(size);
    let vals = img.to_f64_vec()?;

    let shift: Vec<i64> = size
        .iter()
        .map(|&s| {
            let base = (s / 2) as i64;
            if inverse { -base } else { base }
        })
        .collect();

    let out: Vec<f64> = (0..vals.len())
        .map(|flat| {
            let mut src_flat = 0usize;
            for d in 0..dim {
                let idx = (flat / strides[d]) % size[d];
                let len = size[d] as i64;
                let mut shifted = (idx as i64 - shift[d]) % len;
                if shifted < 0 {
                    shifted += len;
                }
                src_flat += shifted as usize * strides[d];
            }
            vals[src_flat]
        })
        .collect();

    image_from_f64(img.pixel_id(), size, img, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Even-sized 1-D axis (`N=4`): the classic half-swap
    /// `[0,1,2,3] -> [2,3,0,1]`, and `inverse` gives the identical result
    /// (the class doc: `Inverse` "has no effect if none of the size of the
    /// input image is even").
    #[test]
    fn even_size_forward_and_inverse_agree() {
        let img = Image::from_vec(&[4, 1], vec![0i32, 1, 2, 3]).unwrap();
        let forward = fft_shift(&img, false).unwrap();
        let inverse = fft_shift(&img, true).unwrap();
        assert_eq!(forward.scalar_slice::<i32>().unwrap(), &[2, 3, 0, 1]);
        assert_eq!(inverse.scalar_slice::<i32>().unwrap(), &[2, 3, 0, 1]);
    }

    /// Odd-sized 1-D axis (`N=5`, shift `5/2=2`): forward and inverse differ
    /// (`shift=2` vs `shift=-2 ≡ 3 mod 5`), hand-derived by tracing
    /// `(idx - shift) mod 5` for each output index.
    #[test]
    fn odd_size_forward_and_inverse_disagree() {
        let img = Image::from_vec(&[5, 1], vec![0i32, 1, 2, 3, 4]).unwrap();
        let forward = fft_shift(&img, false).unwrap();
        let inverse = fft_shift(&img, true).unwrap();
        assert_eq!(forward.scalar_slice::<i32>().unwrap(), &[3, 4, 0, 1, 2]);
        assert_eq!(inverse.scalar_slice::<i32>().unwrap(), &[2, 3, 4, 0, 1]);
    }

    /// Odd-sized axis round-trip: a forward shift followed by an inverse
    /// shift restores the original image (the class doc's documented way to
    /// round-trip an odd dimension).
    #[test]
    fn odd_size_forward_then_inverse_round_trips() {
        let img = Image::from_vec(&[5, 1], vec![0i32, 1, 2, 3, 4]).unwrap();
        let shifted = fft_shift(&img, false).unwrap();
        let restored = fft_shift(&shifted, true).unwrap();
        assert_eq!(
            restored.scalar_slice::<i32>().unwrap(),
            img.scalar_slice::<i32>().unwrap()
        );
    }

    /// A 2-D image shifts each axis independently: one even axis (no
    /// `inverse` effect) and one odd axis (`inverse`-sensitive), combined.
    #[test]
    fn two_d_shifts_each_axis_independently() {
        // 4 (even, x) x 3 (odd, y): shift = (2, 1).
        #[rustfmt::skip]
        let img = Image::from_vec(&[4, 3], vec![
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
        ]).unwrap();
        let out = fft_shift(&img, false).unwrap();
        // x-shift 2 on each row, y-shift 1 (row r reads from row (r-1) mod 3).
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[
            10, 11,  8,  9,
             2,  3,  0,  1,
             6,  7,  4,  5,
        ]);
    }

    /// Output geometry (spacing/origin/direction) is copied from the input
    /// unchanged -- this filter only permutes pixel values.
    #[test]
    fn geometry_is_unchanged() {
        let mut img = Image::from_vec(&[4, 1], vec![0i32, 1, 2, 3]).unwrap();
        img.set_spacing(&[2.0, 1.0]).unwrap();
        img.set_origin(&[5.0, -1.0]).unwrap();
        let out = fft_shift(&img, false).unwrap();
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }
}
