//! `FastApproximateRankImageFilter`: a separable approximation of a rank
//! (order-statistic) filter, verified against
//! `Modules/Nonunit/Review/include/itkFastApproximateRankImageFilter.h`,
//! `itkMiniPipelineSeparableImageFilter.h`/`.hxx`, and
//! `Modules/Filtering/MathematicalMorphology/include/itkRankImageFilter.h`/`.hxx`
//! plus `itkRankHistogram.h` (the order statistic each axis pass computes).
//!
//! ITK's own doc comment: "Medians aren't separable, but if you want a large
//! robust smoother to be relatively quick then it is worthwhile pretending
//! that they are." [`fast_approximate_rank`] runs a 1-D rank filter along
//! each axis in ascending order, axis 0's output feeding axis 1's input and
//! so on (`MiniPipelineSeparableImageFilter::GenerateData`), each pass using
//! a window of up to `2*radius[d]+1` samples along that axis alone. This is
//! *not* the same value as `RankImageFilter` run directly over the full ND
//! box neighborhood — that is the filter's documented, intentional
//! approximation, not a divergence to fix.
//!
//! Each 1-D window is **cropped at the image boundary** rather than
//! zero-flux-replicated: `RankImageFilter`'s class doc says "the
//! neighborhood is cropped at the boundary, and is therefore smaller", and
//! `RankHistogram::AddBoundary`/`RemoveBoundary` (its `MovingHistogramImageFilter`
//! boundary hooks) are no-ops. This matches this crate's
//! [`box_mean`](crate::box_mean)/[`box_sigma`](crate::box_sigma) boundary
//! rule, not [`mean`](crate::mean)/[`median`](crate::median)'s
//! `ZeroFluxNeumannBoundaryCondition`.
//!
//! The order statistic selected from a window of `n` values (kept with
//! duplicates, conceptually sorted ascending) is the 0-indexed position
//! `k = floor(rank * (n - 1))` — from `itkRankHistogram.h`'s
//! `GetValue`/`GetValueBruteForce`: `target = (SizeValueType)(m_Rank *
//! (m_Entries - 1)) + 1` is a *1-indexed* cumulative-count target, so
//! `k = target - 1`. `m_Rank` is stored as `float`
//! (`itkSetClampMacro(Rank, float, 0.0, 1.0)`: a silent clamp, never an
//! error), and in ITK the multiply itself happens in `float` precision (the
//! integer `m_Entries - 1` operand converts to `float`, not `double`, under
//! C++'s usual arithmetic conversions since `m_Rank` is `float`) — reproduced
//! here as an `f32` multiply rather than promoting to `f64`. `rank = 1.0`
//! therefore always picks the true maximum (`k = n - 1`) and `rank = 0.0` the
//! true minimum (`k = 0`); `rank = 0.5` on an **even**-length boundary window
//! picks the *lower* of the two middle values (`k = floor(0.5*(n-1))` rounds
//! down), which is a different tie convention than
//! [`median`](crate::median)'s `select_nth_unstable_by(len/2, ..)` (the
//! *upper* middle value on an even-length window) — both are transcribed
//! verbatim from their respective ITK sources, so the two filters
//! deliberately disagree on this boundary case.
//!
//! **Upstream quirk, reproduced as-is:** `FastApproximateRankImageFilter::SetRank`
//! only forwards the rank to the first `ImageDimension - 1` of its
//! `ImageDimension` per-axis filters:
//!
//! ```cpp
//! for (unsigned int i = 0; i < TInputImage::ImageDimension - 1; ++i)
//!   this->m_Filters[i]->SetRank(m_Rank);
//! ```
//!
//! The last axis's filter (index `ImageDimension - 1`) is never reached by
//! this loop, so it keeps `RankImageFilter`'s own built-in default rank of
//! `0.5` forever — **the last axis is always median-filtered, no matter what
//! `rank` the caller requests.** For a 1-D image the loop body never
//! executes at all (`0 < ImageDimension - 1 == 0` is false), so the *only*
//! axis is always the median regardless of `rank`. [`fast_approximate_rank`]
//! reproduces this exactly: `rank` is applied to every axis except the last,
//! and `0.5` is applied to the last axis, unconditionally.

use crate::error::{FilterError, Result};
use sitk_core::{Image, Scalar, dispatch_scalar};

/// The value that would sit at 0-indexed position `k` in ascending sorted
/// order (`std::nth_element`-equivalent selection; never a full sort).
fn select_rank<T: Copy + PartialOrd>(values: &mut [T], k: usize) -> T {
    let (_, &mut v, _) = values.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    v
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// One separable pass: a 1-D rank filter along `axis`, window
/// `[coord-radius, coord+radius]` clipped to `[0, size[axis]-1]` (never
/// replicated), order statistic `k = floor(rank * (n - 1))`.
fn rank_pass<T: Copy + PartialOrd>(
    buf: &[T],
    size: &[usize],
    strides: &[usize],
    axis: usize,
    radius: usize,
    rank: f32,
) -> Vec<T> {
    let stride = strides[axis];
    let size_axis = size[axis];
    let mut window: Vec<T> = Vec::with_capacity(2 * radius + 1);
    (0..buf.len())
        .map(|p| {
            let coord = (p / stride) % size_axis;
            let line_base = p - coord * stride;
            let lo = coord.saturating_sub(radius);
            let hi = (coord + radius).min(size_axis - 1);
            window.clear();
            window.extend((lo..=hi).map(|c| buf[line_base + c * stride]));
            let k = (rank * (window.len() - 1) as f32) as usize;
            select_rank(&mut window, k)
        })
        .collect()
}

fn fast_approximate_rank_typed<T: Scalar>(
    img: &Image,
    radius: &[usize],
    rank: f32,
) -> Result<Image> {
    let dim = img.dimension();
    let size = img.size().to_vec();
    let strides_ = strides(&size);
    let mut buf: Vec<T> = img
        .scalar_slice::<T>()
        .expect("dispatch guarantees T matches pixel_id")
        .to_vec();

    for (axis, &r) in radius.iter().enumerate() {
        let axis_rank = if axis == dim - 1 { 0.5 } else { rank };
        buf = rank_pass(&buf, &size, &strides_, axis, r, axis_rank);
    }

    let mut result = Image::from_vec(&size, buf)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `FastApproximateRankImageFilter`: a separable rank filter. `rank` is
/// clamped to `[0, 1]` (`0.5` = median, `0.0` = minimum, `1.0` = maximum) and
/// applied as a 1-D pass per axis in ascending axis order — **except the
/// last axis, which is always median-filtered regardless of `rank`** (see
/// the module doc's "upstream quirk" section).
///
/// Errors if `radius.len() != img.dimension()`.
pub fn fast_approximate_rank(img: &Image, radius: &[usize], rank: f64) -> Result<Image> {
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }
    let rank = (rank as f32).clamp(0.0, 1.0);
    dispatch_scalar!(
        img.pixel_id(),
        fast_approximate_rank_typed,
        img,
        radius,
        rank
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    #[test]
    fn fast_approximate_rank_radius_zero_is_identity_every_axis() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = fast_approximate_rank(&img, &[0, 0], 1.0).unwrap();
        assert_eq!(out.to_f64_vec(), img.to_f64_vec());
    }

    #[test]
    fn fast_approximate_rank_odd_window_median_matches_plain_median() {
        // 1-D, interior pixel: n = 3, k = floor(0.5*2) = 1 -> the true
        // middle value, same as an ordinary median.
        let img = Image::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        // dim == 1, so the only axis is always forced to rank 0.5 regardless
        // of the 1.0 passed here -- this also exercises that quirk.
        let out = fast_approximate_rank(&img, &[1], 1.0).unwrap().to_f64_vec();
        assert_eq!(out[2], 30.0);
    }

    #[test]
    fn fast_approximate_rank_even_boundary_window_picks_lower_value_not_upper() {
        // 1-D, left edge pixel: window crops to n = 2 values [10, 20].
        // k = floor(0.5*(2-1)) = 0 -> the LOWER of the pair (10), not the
        // upper one `median()`'s len/2 convention would pick.
        let img = Image::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[1], 1.0).unwrap().to_f64_vec();
        assert_eq!(out[0], 10.0);
        // Right edge is symmetric: window = [40, 50], k = 0 -> 40.
        assert_eq!(out[4], 40.0);
    }

    #[test]
    fn fast_approximate_rank_last_axis_ignores_rank_2d() {
        // 3x3 grid, x fastest:
        //   1 2 3
        //   4 5 6
        //   7 8 9
        // radius = [1, 1], rank = 1.0 (max). Axis 0 (x) is not the last axis
        // so it uses rank=1.0 (max); axis 1 (y, the last axis) is forced to
        // 0.5 regardless of the 1.0 requested.
        let img =
            Image::from_vec(&[3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let out = fast_approximate_rank(&img, &[1, 1], 1.0)
            .unwrap()
            .to_f64_vec();
        // If axis 1 also used rank=1.0 (max), row y=2 would stay [7,8,9]-derived
        // maxima and every column's top would dominate. Instead the forced
        // median on axis 1 pulls rows 1 and 2 to the same values.
        assert_eq!(out, vec![2.0, 3.0, 3.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0]);
    }

    #[test]
    fn fast_approximate_rank_rank_one_is_the_true_maximum() {
        // dim=2 with size[1]=1 isolates axis 0 (user rank) from axis 1 (the
        // last axis, forced 0.5 but trivial on a length-1 axis). radius=2
        // gives each position its own clipped window:
        //   x=0: [10,20,30]      max=30
        //   x=1: [10,20,30,40]   max=40
        //   x=2: [10,20,30,40,50] max=50
        //   x=3: [20,30,40,50]   max=50
        //   x=4: [30,40,50]      max=50
        let img = Image::from_vec(&[5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[2, 0], 1.0)
            .unwrap()
            .to_f64_vec();
        assert_eq!(out, vec![30.0, 40.0, 50.0, 50.0, 50.0]);
    }

    #[test]
    fn fast_approximate_rank_rank_zero_is_the_true_minimum() {
        // Same clipped windows as the rank=1.0 case, minimum instead of
        // maximum: x=0..2 all include the global minimum 10; x=3 and x=4's
        // windows have already clipped it away.
        let img = Image::from_vec(&[5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[2, 0], 0.0)
            .unwrap()
            .to_f64_vec();
        assert_eq!(out, vec![10.0, 10.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn fast_approximate_rank_rank_above_one_is_clamped_not_out_of_bounds() {
        // itkSetClampMacro(Rank, float, 0.0, 1.0): silently clamps, never
        // errors. x=2's window has n=5 (the full row), so an unclamped
        // rank=2.0 would compute k = floor(2.0*4) = 8, out of bounds for a
        // 5-element window -- this must not panic, and must match the
        // clamped rank=1.0 result from the test above.
        let img = Image::from_vec(&[5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[2, 0], 2.0)
            .unwrap()
            .to_f64_vec();
        assert_eq!(out, vec![30.0, 40.0, 50.0, 50.0, 50.0]);
    }

    #[test]
    fn fast_approximate_rank_rank_below_zero_is_clamped_to_the_minimum() {
        let img = Image::from_vec(&[5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[2, 0], -1.0)
            .unwrap()
            .to_f64_vec();
        assert_eq!(out, vec![10.0, 10.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn fast_approximate_rank_output_pixel_type_follows_input() {
        let img = Image::from_vec(&[3, 1], vec![1u8, 2, 3]).unwrap();
        let out = fast_approximate_rank(&img, &[1, 0], 1.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    #[test]
    fn fast_approximate_rank_wrong_radius_length_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float64);
        assert!(matches!(
            fast_approximate_rank(&img, &[1], 0.5),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }
}
