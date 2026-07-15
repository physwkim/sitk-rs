//! `SliceImageFilter` (ITK `Modules/Filtering/ImageGrid/include/itkSliceImageFilter.hxx`):
//! Python-slice-like `[start, stop)` indexing with a step, generalized to
//! N-D and to negative steps (which reverse that axis).
//!
//! [`sitk_core::Image`] is always zero-indexed (see `crate::geometry`'s
//! module docs), so every `inputIndex[i]` the `.hxx` reads from the input's
//! `LargestPossibleRegion` is `0` throughout this port.
//!
//! ## Two independent clamped "start" values
//!
//! The `.hxx` clamps `Start` against *two different* ranges for two
//! different purposes, and this port keeps them distinct rather than
//! collapsing them into one shared clamp:
//!
//! - `GenerateOutputInformation` clamps `Start`/`Stop` into
//!   `[0 - (step<0), size - (step<0)]` — one wider than the valid pixel
//!   index range whenever `step < 0` — to decide the output size and the
//!   output's origin (`inputStartIndex`, the *first sampled* input index,
//!   transformed through the input's own spacing/direction/origin).
//! - `DynamicThreadedGenerateData` separately clamps `Start` into
//!   `[0, size - 1]` (always the valid pixel range, regardless of `step`'s
//!   sign) to compute each output pixel's source index,
//!   `destIndex * step + start`.
//!
//! These two clamped values can only disagree when `Start` clamps to one of
//! `GenerateOutputInformation`'s *outer* bounds (`Start >= size` for
//! `step > 0`, or `Start <= -1` for `step < 0`) — and in both of those cases
//! the branch that would otherwise use the diverging value is unreachable:
//! `Stop`'s own clamp shares that same outer bound as its ceiling/floor, so
//! `stop > start` (or `stop < start`) can never hold, forcing `outputSize =
//! 0`. So the two clamps are ported faithfully as written (down to the
//! empty-region's origin, which the `.hxx` still computes even though no
//! pixel is ever read through it), even though no reachable nonempty output
//! can actually observe them differing.
//!
//! ## Step
//!
//! `Step == 0` is rejected up front ([`FilterError::InvalidSliceStep`]),
//! matching `VerifyInputInformation`'s `itkExceptionMacro`. A negative step
//! reverses that axis: `outputSize[i]` still comes from the same formula
//! (`(stop - start - sgn(step)) / step + 1`, C++ truncating — i.e. Rust's
//! default — integer division), and the output direction's column `i` is
//! the input's negated (`flipMatrix[i][i] = sgn0(step[i])`,
//! `outputDirection = inputDirection * flipMatrix`, which scales columns).

use crate::error::{FilterError, Result};
use sitk_core::Image;

fn require_dim(len: usize, dim: usize) -> Result<()> {
    if len != dim {
        Err(FilterError::DimensionLength {
            expected: dim,
            got: len,
        })
    } else {
        Ok(())
    }
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `SliceImageFilter`: extract `img[start[d]:stop[d]:step[d]]` per axis,
/// Python-slice-like (`[start, stop)`, clamped to the image bounds; a
/// negative `step` reverses that axis). See the module docs for the exact
/// clamping/size/geometry formulas ported from `GenerateOutputInformation`
/// and `DynamicThreadedGenerateData`.
///
/// Errors if `start`/`stop`/`step` don't have one entry per image dimension,
/// or if any `step[d] == 0` ([`FilterError::InvalidSliceStep`]).
pub fn slice(img: &Image, start: &[i32], stop: &[i32], step: &[i32]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(start.len(), dim)?;
    require_dim(stop.len(), dim)?;
    require_dim(step.len(), dim)?;
    if step.contains(&0) {
        return Err(FilterError::InvalidSliceStep(step.to_vec()));
    }

    let in_size = img.size();
    let in_spacing = img.spacing();
    let in_direction = img.direction();

    let mut out_size = vec![0usize; dim];
    let mut out_spacing = vec![0.0f64; dim];
    let mut out_direction = in_direction.to_vec();
    // `GenerateOutputInformation`'s clamped start: feeds the output origin
    // (and, jointly with the clamped stop, the output size).
    let mut go_start = vec![0i64; dim];
    // `DynamicThreadedGenerateData`'s separately clamped start: feeds the
    // per-pixel source index (see the module docs on why these two can
    // diverge only where the output is provably empty).
    let mut dt_start = vec![0i64; dim];

    for d in 0..dim {
        let size_i = in_size[d] as i64;
        let step_i = step[d] as i64;

        out_spacing[d] = in_spacing[d] * step_i.unsigned_abs() as f64;
        if step_i < 0 {
            for row in 0..dim {
                out_direction[row * dim + d] *= -1.0;
            }
        }

        let neg = i64::from(step_i < 0);
        let lo = -neg;
        let hi = size_i - neg;
        let start_go = (start[d] as i64).clamp(lo, hi);
        let stop_go = (stop[d] as i64).clamp(lo, hi);
        go_start[d] = start_go;

        out_size[d] = if (step_i > 0 && stop_go > start_go) || (step_i < 0 && stop_go < start_go) {
            let sgn = if step_i > 0 { 1 } else { -1 };
            ((stop_go - start_go - sgn) / step_i + 1) as usize
        } else {
            0
        };

        dt_start[d] = if size_i > 0 {
            (start[d] as i64).clamp(0, size_i - 1)
        } else {
            0
        };
    }

    let go_start_f: Vec<f64> = go_start.iter().map(|&v| v as f64).collect();
    let out_origin = img.continuous_index_to_physical_point(&go_start_f);

    let in_strides = strides(in_size);
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();
    let mut sources: Vec<Option<usize>> = vec![None; out_count];
    for (o, slot) in sources.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for d in 0..dim {
            let oi = (o / out_strides[d]) % out_size[d];
            let src_idx = oi as i64 * step[d] as i64 + dt_start[d];
            in_flat += src_idx as usize * in_strides[d];
        }
        *slot = Some(in_flat);
    }

    let mut out = img.gather(&out_size, &sources, 0.0)?;
    out.set_spacing(&out_spacing)?;
    out.set_origin(&out_origin)?;
    out.set_direction(&out_direction)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- defaults are the identity slice ----

    #[test]
    fn defaults_are_the_identity_slice() {
        let src = img(&[4, 3], (0..12).map(|v| v as f64).collect());
        let out = slice(&src, &[0, 0], &[i32::MAX, i32::MAX], &[1, 1]).unwrap();
        assert_eq!(out.size(), src.size());
        assert_eq!(out.to_f64_vec().unwrap(), src.to_f64_vec().unwrap());
        assert_eq!(out.origin(), src.origin());
        assert_eq!(out.spacing(), src.spacing());
        assert_eq!(out.direction(), src.direction());
    }

    // ---- step == 0 errors ----

    #[test]
    fn zero_step_errors() {
        let src = img(&[4, 1], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(
            slice(&src, &[0, 0], &[4, 1], &[0, 1]),
            Err(FilterError::InvalidSliceStep(vec![0, 1]))
        );
    }

    // ---- start/stop clamping edges ----

    #[test]
    fn start_beyond_size_yields_empty_output() {
        let src = img(&[5, 1], (0..5).map(|v| v as f64).collect());
        let out = slice(&src, &[100, 0], &[200, i32::MAX], &[1, 1]).unwrap();
        assert_eq!(out.size(), &[0, 1]);
        assert_eq!(out.number_of_pixels(), 0);
    }

    #[test]
    fn stop_before_start_yields_empty_output() {
        // Doc: "If the stopping index is already beyond the starting index
        // then an image of size zero will be returned."
        let src = img(&[5, 1], (0..5).map(|v| v as f64).collect());
        let out = slice(&src, &[3, 0], &[1, i32::MAX], &[1, 1]).unwrap();
        assert_eq!(out.size(), &[0, 1]);
    }

    #[test]
    fn stop_clamped_to_size_includes_the_last_pixel() {
        let src = img(&[5, 1], (0..5).map(|v| v as f64).collect());
        let out = slice(&src, &[2, 0], &[i32::MAX, i32::MAX], &[1, 1]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![2.0, 3.0, 4.0]);
    }

    // ---- Python-slice-like sub-ranges, hand-derived ----

    #[test]
    fn positive_step_takes_every_other_element_python_slice_style() {
        // Python: list(range(10))[1:8:2] == [1, 3, 5, 7]
        let src = img(&[10, 1], (0..10).map(|v| v as f64).collect());
        let out = slice(&src, &[1, 0], &[8, i32::MAX], &[2, 1]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![1.0, 3.0, 5.0, 7.0]);
    }

    // ---- negative step reverses ----

    #[test]
    fn negative_step_reverses_the_whole_axis() {
        // Python: list(range(5))[4::-1] == [4, 3, 2, 1, 0]
        let src = img(&[5, 1], (0..5).map(|v| v as f64).collect());
        let out = slice(&src, &[4, 0], &[-1, i32::MAX], &[-1, 1]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn negative_step_sub_range_matches_python_slice_indexing() {
        // Python: list(range(10))[7:2:-2] == [7, 5, 3]
        let src = img(&[10, 1], (0..10).map(|v| v as f64).collect());
        let out = slice(&src, &[7, 0], &[2, i32::MAX], &[-2, 1]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![7.0, 5.0, 3.0]);
    }

    #[test]
    fn slice_moves_u64_pixels_losslessly() {
        // 2^53 + 1 is the smallest u64 an f64 cannot represent; the old seam
        // rounded it. Non-vacuity guard proves the value would be corrupted.
        let hi = (1u64 << 53) + 1;
        assert_ne!(hi, (hi as f64) as u64);
        let src = Image::from_vec(&[5, 1], (0..5).map(|v| hi + v).collect()).unwrap();
        // Forward sub-range [1:4:1] and a full negative-step reversal.
        let fwd = slice(&src, &[1, 0], &[4, i32::MAX], &[1, 1]).unwrap();
        assert_eq!(
            fwd.scalar_slice::<u64>().unwrap(),
            &[hi + 1, hi + 2, hi + 3]
        );
        let rev = slice(&src, &[4, 0], &[-1, i32::MAX], &[-1, 1]).unwrap();
        assert_eq!(
            rev.scalar_slice::<u64>().unwrap(),
            &[hi + 4, hi + 3, hi + 2, hi + 1, hi]
        );
    }

    // ---- geometry: origin moves to the first sampled pixel ----

    #[test]
    fn origin_moves_to_the_first_sampled_pixel_forward() {
        let mut src = img(&[6, 1], (0..6).map(|v| v as f64).collect());
        src.set_spacing(&[2.0, 1.0]).unwrap();
        src.set_origin(&[10.0, 0.0]).unwrap();
        let out = slice(&src, &[2, 0], &[i32::MAX, i32::MAX], &[1, 1]).unwrap();
        let expected_origin = src.continuous_index_to_physical_point(&[2.0, 0.0]);
        for (a, b) in out.origin().iter().zip(&expected_origin) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn origin_moves_to_the_first_sampled_pixel_reversed_and_direction_flips() {
        let mut src = img(&[6, 1], (0..6).map(|v| v as f64).collect());
        src.set_spacing(&[2.0, 1.0]).unwrap();
        src.set_origin(&[10.0, 0.0]).unwrap();
        let out = slice(&src, &[5, 0], &[-1, i32::MAX], &[-1, 1]).unwrap();
        let expected_origin = src.continuous_index_to_physical_point(&[5.0, 0.0]);
        for (a, b) in out.origin().iter().zip(&expected_origin) {
            assert!((a - b).abs() < 1e-12);
        }
        // Direction column 0 negated (reversed axis), column 1 untouched.
        let d = src.direction();
        assert_eq!(out.direction(), &[-d[0], d[1], -d[2], d[3]]);
        // Spacing magnitude is preserved (scaled by |step| == 1), sign lives
        // in direction, not spacing (ITK spacing is always non-negative).
        assert_eq!(out.spacing(), src.spacing());
    }

    #[test]
    fn step_magnitude_scales_output_spacing() {
        let mut src = img(&[9, 1], (0..9).map(|v| v as f64).collect());
        src.set_spacing(&[1.5, 1.0]).unwrap();
        let out = slice(&src, &[0, 0], &[i32::MAX, i32::MAX], &[3, 1]).unwrap();
        assert_eq!(out.spacing()[0], 4.5);
        assert_eq!(out.to_f64_vec().unwrap(), vec![0.0, 3.0, 6.0]);
    }

    // ---- 2-D: only the sliced axis is affected ----

    #[test]
    fn two_d_slice_only_affects_the_targeted_axis() {
        #[rustfmt::skip]
        let src = img(&[3, 3], vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]);
        let out = slice(&src, &[0, 1], &[i32::MAX, i32::MAX], &[1, 1]).unwrap();
        assert_eq!(out.size(), &[3, 2]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn dimension_mismatch_errors() {
        let src = img(&[4, 4], vec![0.0; 16]);
        assert!(matches!(
            slice(&src, &[0], &[1, 1], &[1, 1]),
            Err(FilterError::DimensionLength { .. })
        ));
    }
}
