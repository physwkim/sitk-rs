//! `itk::FFTShiftImageFilter`
//! (`Modules/Filtering/FFT/include/itkFFTShiftImageFilter.hxx`): cyclically
//! shift an image so the zero-frequency corner of a Fourier transform lands
//! at the image center.
//!
//! Ported through the superclass that actually performs the wraparound copy,
//! `itk::CyclicShiftImageFilter`
//! (`Modules/Filtering/ImageGrid/include/itkCyclicShiftImageFilter.h(.hxx)`),
//! which this module also exposes directly as [`cyclic_shift`]:
//! `FFTShiftImageFilter::GenerateData` computes `shift[i] = size[i] / 2`
//! (integer division), negated when `inverse` is set, then
//! `CyclicShiftImageFilter::DynamicThreadedGenerateData`
//! (`itkCyclicShiftImageFilter.hxx:70-78`) reads each output pixel from
//! `input[(index[i] - outIdx[i] - shift[i]) mod size[i]]`, wrapped into
//! `[0, size[i])`. This crate's images always start at index 0 (there is no
//! `LargestPossibleRegion` index offset concept here), so the `outIdx` term
//! is always zero and drops out of the port. Both functions share that same
//! wraparound core, `cyclic_sources`, differing only in how `shift` is
//! produced.
//!
//! `FFTShiftImageFilter.yaml` declares `pixel_types: NonLabelPixelIDTypeList`
//! (SimpleITK's full pixel-type list minus label maps, which for the C++
//! library includes complex pixel types). The shift is pure pixel-type-agnostic
//! index permutation, so [`fft_shift`] moves each pixel through the native
//! [`crate::core::Image::gather`] primitive and covers every pixel type this
//! crate represents, complex and vector included — shifting a complex spectrum
//! is the filter's canonical use. (An earlier port routed the copy through
//! `to_f64_vec`, which both rounded `UInt64`/`Int64` magnitudes above 2^53 and
//! rejected complex/vector images outright; the native gather closes both.)
//!
//! For an *even*-sized axis, `size[i] / 2` and `-(size[i] / 2)` are congruent
//! mod `size[i]`, so `inverse` has no effect there. For an *odd*-sized axis
//! they differ by one slot, so a forward shift followed by an inverse shift
//! (or vice versa) is required to round-trip that axis -- exactly the class
//! doc's note: "applying this filter twice will not produce the same image as
//! the original one without using SetInverse(true) on one (and only one) of
//! the two filters."
//!
//! `CyclicShiftImageFilter.yaml` declares `pixel_types: NonLabelPixelIDTypeList`
//! too, and `Shift: type int, dim_vec: true, default [0, 0, 0]` (a per-axis
//! shift, positive or negative, defaulting to no shift at all).

use crate::core::Image;
use crate::filters::error::{FilterError, Result};

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

fn require_dim(len: usize, dim: usize) -> Result<()> {
    if len != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: len,
        });
    }
    Ok(())
}

/// The wraparound core shared by [`fft_shift`] and [`cyclic_shift`], as a
/// per-output-pixel source index: `output[index] = input[(index[d] - shift[d])
/// mod size[d]]` for every axis `d` (`itkCyclicShiftImageFilter.hxx:70-78`,
/// with the always-zero `outIdx` term dropped, see the module docs). The result
/// feeds [`Image::gather`], which moves each pixel natively.
fn cyclic_sources(size: &[usize], strides: &[usize], shift: &[i64]) -> Vec<Option<usize>> {
    let dim = size.len();
    let count: usize = size.iter().product();
    (0..count)
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
            Some(src_flat)
        })
        .collect()
}

/// `FFTShiftImageFilter`: cyclically shift every axis by `size[i] / 2`
/// (`inverse` negates the shift), moving the zero-frequency corner to the
/// image center. See the module docs for the odd-size/`inverse` interaction.
pub fn fft_shift(img: &Image, inverse: bool) -> Result<Image> {
    let size = img.size();
    let strides = strides(size);

    let shift: Vec<i64> = size
        .iter()
        .map(|&s| {
            let base = (s / 2) as i64;
            if inverse { -base } else { base }
        })
        .collect();

    let sources = cyclic_sources(size, &strides, &shift);
    Ok(img.gather(size, &sources, 0.0)?)
}

/// `CyclicShiftImageFilter` (`itkCyclicShiftImageFilter.h(.hxx)`): shift every
/// axis cyclically by the caller-given `shift[d]` (positive or negative;
/// `CyclicShiftImageFilter.yaml`'s `Shift` member defaults to all zero, one
/// entry per image axis). A pixel that moves across a boundary wraps around
/// it, matching the class doc's example: a 40x40 image shifted by `[13, 47]`
/// puts input pixel `[0, 0]` at output index `[13, 7]`
/// (`itkCyclicShiftImageFilter.h:36-38`).
pub fn cyclic_shift(img: &Image, shift: &[i64]) -> Result<Image> {
    let size = img.size();
    require_dim(shift.len(), size.len())?;
    let strides = strides(size);

    let sources = cyclic_sources(size, &strides, shift);
    Ok(img.gather(size, &sources, 0.0)?)
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

    // ---- CyclicShift ----

    /// Hand-traced positive shift: `output[idx] = input[(idx - 2) mod 5]`.
    /// `Image::from_vec(&[5, 1], ...)` is a 2-D image, so `shift` needs one
    /// entry per axis; the trailing size-1 axis's shift is inert.
    #[test]
    fn cyclic_shift_positive_shift_wraps_forward() {
        let img = Image::from_vec(&[5, 1], vec![0i32, 1, 2, 3, 4]).unwrap();
        let out = cyclic_shift(&img, &[2, 0]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[3, 4, 0, 1, 2]);
    }

    /// Hand-traced negative shift: `output[idx] = input[(idx + 2) mod 5]`.
    #[test]
    fn cyclic_shift_negative_shift_wraps_backward() {
        let img = Image::from_vec(&[5, 1], vec![0i32, 1, 2, 3, 4]).unwrap();
        let out = cyclic_shift(&img, &[-2, 0]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[2, 3, 4, 0, 1]);
    }

    /// `CyclicShiftImageFilter` is `FFTShiftImageFilter`'s superclass with a
    /// caller-chosen shift instead of a fixed `size/2`: `fft_shift(img, true)`
    /// on a size-5 axis uses `shift = -(5/2) = -2`, so this must agree with
    /// `cyclic_shift(img, [-2, 0])` pixel-for-pixel.
    #[test]
    fn cyclic_shift_agrees_with_fft_shift_inverse_at_the_same_effective_shift() {
        let img = Image::from_vec(&[5, 1], vec![0i32, 1, 2, 3, 4]).unwrap();
        let via_cyclic_shift = cyclic_shift(&img, &[-2, 0]).unwrap();
        let via_fft_shift = fft_shift(&img, true).unwrap();
        assert_eq!(
            via_cyclic_shift.scalar_slice::<i32>().unwrap(),
            via_fft_shift.scalar_slice::<i32>().unwrap()
        );
    }

    /// `CyclicShiftImageFilter.yaml`'s default `Shift` is all-zero: the
    /// identity permutation.
    #[test]
    fn cyclic_shift_yaml_default_shift_is_identity() {
        let img = Image::from_vec(&[4, 1], vec![10i32, 20, 30, 40]).unwrap();
        let out = cyclic_shift(&img, &[0, 0]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[10, 20, 30, 40]);
    }

    /// Each axis shifts independently, with a mix of positive and negative
    /// per-axis shifts (the class doc: "Negative Shifts are supported").
    #[test]
    fn cyclic_shift_shifts_each_axis_independently_with_mixed_signs() {
        #[rustfmt::skip]
        let img = Image::from_vec(&[4, 3], vec![
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
        ]).unwrap();
        // x-shift +1, y-shift -1: output[x,y] = input[(x-1) mod 4, (y+1) mod 3].
        let out = cyclic_shift(&img, &[1, -1]).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[
             7,  4,  5,  6,
            11,  8,  9, 10,
             3,  0,  1,  2,
        ]);
    }

    /// The class doc's own worked example: on a 40x40 image shifted by
    /// `[13, 47]`, input pixel `[0, 0]` lands at output index `[13, 7]`
    /// (`itkCyclicShiftImageFilter.h:36-38`).
    #[test]
    fn cyclic_shift_matches_the_class_docs_worked_example() {
        let size = [40usize, 40usize];
        let mut data = vec![0i32; size[0] * size[1]];
        data[0] = 99; // input[0, 0]
        let img = Image::from_vec(&size, data).unwrap();
        let out = cyclic_shift(&img, &[13, 47]).unwrap();
        let got = out.scalar_slice::<i32>().unwrap();
        let flat = 13 + 7 * size[0]; // output index [13, 7]
        assert_eq!(got[flat], 99);
        assert_eq!(got.iter().filter(|&&v| v == 99).count(), 1);
    }

    #[test]
    fn cyclic_shift_rejects_a_shift_vector_of_the_wrong_dimension() {
        let img = Image::from_vec(&[4, 3], vec![0i32; 12]).unwrap();
        let err = cyclic_shift(&img, &[1]).unwrap_err();
        assert_eq!(
            err,
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn cyclic_shift_moves_u64_pixels_losslessly() {
        // 2^53 + 1 is the smallest u64 an f64 cannot hold; the old to_f64_vec
        // seam rounded it. Non-vacuity guard proves the values would corrupt.
        let hi = (1u64 << 53) + 1;
        assert_ne!(hi, (hi as f64) as u64);
        let img = Image::from_vec(&[4, 1], vec![hi, hi + 1, hi + 2, hi + 3]).unwrap();
        let out = cyclic_shift(&img, &[1, 0]).unwrap();
        // out[idx] = in[(idx - 1) mod 4].
        assert_eq!(
            out.scalar_slice::<u64>().unwrap(),
            &[hi + 3, hi, hi + 1, hi + 2]
        );
    }

    #[test]
    fn cyclic_shift_now_moves_complex_pixels_natively() {
        // FFTShift/CyclicShift are registered for NonLabelPixelIDTypeList
        // (complex included) upstream, and shifting a complex spectrum is the
        // filter's canonical use. Routing through native Image::gather lifts the
        // scalar-only limitation the old to_f64_vec seam imposed — the shift is
        // pure pixel-type-agnostic index permutation.
        use crate::core::Complex;
        let img = Image::from_vec_complex(
            &[4, 1],
            vec![
                Complex::new(1.0f32, 10.0),
                Complex::new(2.0, 20.0),
                Complex::new(3.0, 30.0),
                Complex::new(4.0, 40.0),
            ],
        )
        .unwrap();
        let out = cyclic_shift(&img, &[1, 0]).unwrap();
        assert_eq!(out.pixel_id(), crate::core::PixelId::ComplexFloat32);
        // out[idx] = in[(idx - 1) mod 4], whole complex pixels moved together.
        assert_eq!(
            out.get_complex::<f32>(&[0, 0]).unwrap(),
            Complex::new(4.0, 40.0)
        );
        assert_eq!(
            out.get_complex::<f32>(&[1, 0]).unwrap(),
            Complex::new(1.0, 10.0)
        );
        assert_eq!(
            out.get_complex::<f32>(&[3, 0]).unwrap(),
            Complex::new(3.0, 30.0)
        );
    }
}
