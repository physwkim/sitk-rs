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
//! **Fixed upstream bug:** `FastApproximateRankImageFilter::SetRank` only
//! forwarded the rank to the first `ImageDimension - 1` of its
//! `ImageDimension` per-axis filters:
//!
//! ```cpp
//! for (unsigned int i = 0; i < TInputImage::ImageDimension - 1; ++i)
//!   this->m_Filters[i]->SetRank(m_Rank);
//! ```
//!
//! so the last axis's filter (index `ImageDimension - 1`) was never reached
//! by this loop and kept `RankImageFilter`'s own built-in default rank of
//! `0.5` forever — the last axis was always median-filtered, no matter what
//! `rank` the caller requested (and for a 1-D image the loop body never
//! executed at all, so the *only* axis was always the median). Fixed by
//! upstream PR InsightSoftwareConsortium/ITK#6580 (2026-07-10), which widens
//! the loop bound to `ImageDimension`: every axis's filter now receives the
//! same `rank`. [`fast_approximate_rank`] applies `rank` uniformly to every
//! axis, matching the fix.
//!
//! [`rank`] is `RankImageFilter` itself, ported directly rather than through
//! the separable approximation: every output pixel is the order statistic of
//! the *full* ND `kernel`-on neighborhood in one shot, using the same
//! `k = floor(rank * (n - 1))` formula and cropped (not replicated) boundary
//! rule derived above, plus [`crate::morphology::StructuringElement`] for the
//! `Box`/`Cross`/`Ball`/custom-mask kernel shapes
//! (`RankImageFilter.yaml`'s `Radius`/`KernelType` members). ITK's own
//! `MovingHistogramImageFilter` machinery (`AddBoundary`/`RemoveBoundary`
//! no-ops, an incremental sliding histogram) has no equivalent in this
//! crate's [`sitk_core::NeighborhoodIterator`] (every existing
//! [`sitk_core::BoundaryCondition`] substitutes *some* value for an
//! out-of-bounds offset, never excludes it), so [`rank`] hand-rolls a direct
//! per-offset bounds check instead of reusing that iterator — a
//! from-scratch, non-incremental gather per pixel rather than Huang's
//! sliding-histogram optimization, matching this port's usual
//! correctness-over-performance stance for filters whose only obstacle to
//! reuse is a missing iterator boundary mode.

use crate::error::{FilterError, Result};
use crate::morphology::StructuringElement;
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
    let size = img.size().to_vec();
    let strides_ = strides(&size);
    let mut buf: Vec<T> = img.scalar_slice::<T>()?.to_vec();

    for (axis, &r) in radius.iter().enumerate() {
        buf = rank_pass(&buf, &size, &strides_, axis, r, rank);
    }

    let mut result = Image::from_vec(&size, buf)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `FastApproximateRankImageFilter`: a separable rank filter. `rank` is
/// clamped to `[0, 1]` (`0.5` = median, `0.0` = minimum, `1.0` = maximum) and
/// applied as a 1-D pass per axis, in ascending axis order, uniformly across
/// every axis (see the module doc's "Fixed upstream bug" section).
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
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn fast_approximate_rank_odd_window_median_matches_plain_median() {
        // 1-D, interior pixel: n = 3, k = floor(0.5*2) = 1 -> the true
        // middle value, same as an ordinary median. rank = 0.5 is passed
        // explicitly (see `fast_approximate_rank_1d_honors_the_requested_rank`
        // for proof that a 1-D image's only axis actually receives whatever
        // rank the caller passes, now that the upstream last-axis bug is
        // fixed).
        let img = Image::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[1], 0.5)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out[2], 30.0);
    }

    #[test]
    fn fast_approximate_rank_even_boundary_window_picks_lower_value_not_upper() {
        // 1-D, left edge pixel, median rank: window crops to n = 2 values
        // [10, 20]. k = floor(0.5*(2-1)) = 0 -> the LOWER of the pair (10),
        // not the upper one `median()`'s len/2 convention would pick.
        let img = Image::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[1], 0.5)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out[0], 10.0);
        // Right edge is symmetric: window = [40, 50], k = 0 -> 40.
        assert_eq!(out[4], 40.0);
    }

    /// Fixed upstream bug (module doc, ITK#6580): a 1-D image's only axis is
    /// `ImageDimension - 1 == 0`, so upstream's `i < ImageDimension - 1`
    /// loop bound never ran at all and the axis kept `RankImageFilter`'s
    /// built-in default rank of `0.5` regardless of what the caller passed.
    /// `rank = 1.0` (max) proves the caller's own rank now reaches the
    /// axis: at the interior pixel the window is `[20, 30, 40]`, and the
    /// true maximum (40) is not the median (30) the old forced-0.5 default
    /// would have produced.
    #[test]
    fn fast_approximate_rank_1d_honors_the_requested_rank() {
        let img = Image::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[1], 1.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out[2], 40.0);
    }

    /// Fixed upstream bug (module doc, ITK#6580): `rank` now applies
    /// uniformly to every axis, including the last one, rather than the
    /// last axis being forced to the median regardless of `rank`.
    ///
    /// 3x3 grid, x fastest:
    ///   1 2 3
    ///   4 5 6
    ///   7 8 9
    /// radius = [1, 1], rank = 1.0 (max) on both axes.
    ///
    /// Axis 0 (x) pass, per-row cropped max:
    ///   row y=0 [1,2,3] -> [2,3,3]
    ///   row y=1 [4,5,6] -> [5,6,6]
    ///   row y=2 [7,8,9] -> [8,9,9]
    /// Axis 1 (y) pass, per-column cropped max, on the axis-0 result:
    ///   col x=0 [2,5,8] -> [5,8,8]
    ///   col x=1 [3,6,9] -> [6,9,9]
    ///   col x=2 [3,6,9] -> [6,9,9]
    #[test]
    fn fast_approximate_rank_applies_rank_uniformly_to_every_axis_2d() {
        let img =
            Image::from_vec(&[3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let out = fast_approximate_rank(&img, &[1, 1], 1.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out, vec![5.0, 6.0, 6.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0]);
    }

    #[test]
    fn fast_approximate_rank_rank_one_is_the_true_maximum() {
        // dim=2 with size[1]=1 isolates axis 0's behavior: axis 1 also
        // receives `rank` now, but is trivial on a length-1 axis (window
        // size 1 always selects that single value regardless of rank).
        // radius=2 gives each position its own clipped window:
        //   x=0: [10,20,30]      max=30
        //   x=1: [10,20,30,40]   max=40
        //   x=2: [10,20,30,40,50] max=50
        //   x=3: [20,30,40,50]   max=50
        //   x=4: [30,40,50]      max=50
        let img = Image::from_vec(&[5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[2, 0], 1.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
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
            .to_f64_vec()
            .unwrap();
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
            .to_f64_vec()
            .unwrap();
        assert_eq!(out, vec![30.0, 40.0, 50.0, 50.0, 50.0]);
    }

    #[test]
    fn fast_approximate_rank_rank_below_zero_is_clamped_to_the_minimum() {
        let img = Image::from_vec(&[5, 1], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let out = fast_approximate_rank(&img, &[2, 0], -1.0)
            .unwrap()
            .to_f64_vec()
            .unwrap();
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

// ---- RankImageFilter (exact, ND) -------------------------------------------

/// Per-offset ND coordinates for a `radius`-sized window, dimension-0-fastest
/// -- the same enumeration [`crate::morphology`]'s own (private)
/// `window_offsets` builds, and the order [`StructuringElement`]'s `on()`
/// mask lines up with; duplicated locally per this crate's existing
/// convention of re-deriving this small enumeration in each module that
/// needs it (see `object_morphology.rs`/`denoise.rs`'s own local copies),
/// rather than exporting it across a module boundary.
fn window_offsets(radius: &[usize]) -> Vec<Vec<i64>> {
    let dim = radius.len();
    let n: usize = radius.iter().map(|&r| 2 * r + 1).product();
    let mut offsets = Vec::with_capacity(n);
    let mut offset: Vec<i64> = radius.iter().map(|&r| -(r as i64)).collect();
    for _ in 0..n {
        offsets.push(offset.clone());
        for d in 0..dim {
            offset[d] += 1;
            if offset[d] > radius[d] as i64 {
                offset[d] = -(radius[d] as i64);
            } else {
                break;
            }
        }
    }
    offsets
}

fn rank_typed<T: Scalar>(img: &Image, kernel: &StructuringElement, rank: f32) -> Result<Image> {
    let size = img.size().to_vec();
    let dim = size.len();
    let strides_ = strides(&size);
    let buf: Vec<T> = img.scalar_slice::<T>()?.to_vec();

    let offsets = window_offsets(kernel.radius());
    let on_offsets: Vec<&Vec<i64>> = offsets
        .iter()
        .zip(kernel.on())
        .filter_map(|(off, &on)| on.then_some(off))
        .collect();

    let result: Vec<T> = (0..buf.len())
        .map(|flat| {
            let mut window: Vec<T> = Vec::with_capacity(on_offsets.len());
            'offset: for off in &on_offsets {
                let mut src_flat = 0usize;
                for d in 0..dim {
                    let coord = (flat / strides_[d]) % size[d];
                    let c = coord as i64 + off[d];
                    if c < 0 || c >= size[d] as i64 {
                        continue 'offset;
                    }
                    src_flat += c as usize * strides_[d];
                }
                window.push(buf[src_flat]);
            }
            if window.is_empty() {
                return Err(FilterError::EmptyRankNeighborhood);
            }
            let k = (rank * (window.len() - 1) as f32) as usize;
            Ok(select_rank(&mut window, k))
        })
        .collect::<Result<Vec<T>>>()?;

    let mut out = Image::from_vec(&size, result)?;
    out.copy_geometry_from(img);
    Ok(out)
}

/// `RankImageFilter` (`itkRankImageFilter.h(.hxx)`, order statistic from
/// `itkRankHistogram.h`): the exact ND rank filter -- unlike
/// [`fast_approximate_rank`], every output pixel is the `rank`-th order
/// statistic of the *full* `kernel`-on neighborhood at once (no per-axis
/// separable approximation, and no last-axis-forced-to-median quirk). The
/// neighborhood is cropped at the boundary, never replicated
/// (`RankHistogram::AddBoundary`/`RemoveBoundary` are no-ops, matching
/// `itkMovingHistogramImageFilter.hxx`'s boundary handling and this module's
/// [`fast_approximate_rank`]).
///
/// `rank` is clamped to `[0, 1]` and narrowed to `f32` before the order
/// statistic multiply (`itkSetClampMacro(Rank, float, 0.0, 1.0)`;
/// `RankImageFilter.yaml`'s `Rank` member is `double`-typed at the SimpleITK
/// wrapper boundary but narrows to the C++ class's `float m_Rank` — see the
/// module doc's derivation of `k = floor(rank * (n - 1))` in `f32`
/// precision, same as [`fast_approximate_rank`]).
///
/// `kernel`'s radius must match `img`'s dimension (mirrors
/// [`sitk_core::NeighborhoodIterator::new`]'s own `RadiusMismatch` for the
/// same condition, since this filter hand-rolls its cropped-boundary
/// neighborhood gather rather than going through that iterator — see the
/// module docs on why: none of this crate's boundary conditions model
/// "exclude out-of-bounds" the way `RankHistogram` needs).
///
/// Errors with [`FilterError::EmptyRankNeighborhood`] if `kernel`'s on-cells
/// are entirely cropped away at some pixel (see that variant's doc for when
/// this is actually reachable).
pub fn rank(img: &Image, kernel: &StructuringElement, rank: f64) -> Result<Image> {
    let dim = img.dimension();
    if kernel.radius().len() != dim {
        return Err(sitk_core::Error::RadiusMismatch { dimension: dim }.into());
    }
    let rank = (rank as f32).clamp(0.0, 1.0);
    dispatch_scalar!(img.pixel_id(), rank_typed, img, kernel, rank)
}

#[cfg(test)]
mod rank_tests {
    use super::*;
    use sitk_core::PixelId;

    #[test]
    fn rank_median_matches_hand_computed_interior_value() {
        // 3x3, box radius 1: interior pixel's full 9-neighborhood sorted is
        // [1..9], k = floor(0.5*8) = 4 -> the true median, 5. 0.5 is also
        // `RankImageFilter.yaml`'s default `Rank`, so this pins that default
        // at the same time.
        #[rustfmt::skip]
        let img = Image::from_vec(&[3, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let kernel = StructuringElement::box_(&[1, 1]);
        let out = rank(&img, &kernel, 0.5).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap()[4], 5.0);
    }

    #[test]
    fn rank_boundary_neighborhood_is_cropped_not_replicated() {
        // Top-left corner of a 3x3 box-radius-1 kernel only overlaps the
        // 2x2 block {1,2,4,5}: n=4, k = floor(rank*3).
        #[rustfmt::skip]
        let img = Image::from_vec(&[3, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let kernel = StructuringElement::box_(&[1, 1]);
        let out = rank(&img, &kernel, 0.0).unwrap();
        // rank=0.0 -> the minimum of {1,2,4,5} = 1, not the minimum of the
        // full image (which would also be 1 here, so also check rank=1.0).
        assert_eq!(out.scalar_slice::<f64>().unwrap()[0], 1.0);
        let out_max = rank(&img, &kernel, 1.0).unwrap();
        // max of the cropped corner window {1,2,4,5} = 5, NOT the image
        // maximum (9), proving the window really is cropped to 4 elements.
        assert_eq!(out_max.scalar_slice::<f64>().unwrap()[0], 5.0);
    }

    #[test]
    fn rank_zero_is_the_true_minimum_and_one_is_the_true_maximum() {
        #[rustfmt::skip]
        let img = Image::from_vec(&[3, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let kernel = StructuringElement::box_(&[1, 1]);
        let min_out = rank(&img, &kernel, 0.0).unwrap();
        let max_out = rank(&img, &kernel, 1.0).unwrap();
        assert_eq!(min_out.scalar_slice::<f64>().unwrap()[4], 1.0); // center's full 3x3 window
        assert_eq!(max_out.scalar_slice::<f64>().unwrap()[4], 9.0);
    }

    #[test]
    fn rank_cross_kernel_only_uses_the_plus_shaped_neighborhood() {
        // Cross radius 1 at the center excludes the four corners: window =
        // {2,4,5,6,8} (n=5), k = floor(0.5*4) = 2 -> sorted[2] = 5.
        #[rustfmt::skip]
        let img = Image::from_vec(&[3, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let kernel = StructuringElement::cross(&[1, 1]);
        let out = rank(&img, &kernel, 0.5).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap()[4], 5.0);
    }

    #[test]
    fn rank_output_pixel_type_follows_input() {
        #[rustfmt::skip]
        let img = Image::from_vec(&[3, 3], vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let kernel = StructuringElement::box_(&[1, 1]);
        let out = rank(&img, &kernel, 0.5).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    #[test]
    fn rank_rejects_a_kernel_of_the_wrong_dimension() {
        let img = Image::from_vec(&[3, 3], vec![0.0; 9]).unwrap();
        let kernel = StructuringElement::box_(&[1]);
        let err = rank(&img, &kernel, 0.5).unwrap_err();
        assert_eq!(
            err,
            FilterError::Core(sitk_core::Error::RadiusMismatch { dimension: 2 })
        );
    }

    #[test]
    fn rank_rejects_a_structuring_element_with_no_on_cells_reachable() {
        // A custom mask that excludes the center: at the single-pixel
        // image's only position, the sole offset (the center) is off, so no
        // in-bounds on-cell exists anywhere.
        let img = Image::from_vec(&[1, 1], vec![5.0]).unwrap();
        let kernel = StructuringElement::from_mask(&[0, 0], vec![false]).unwrap();
        let err = rank(&img, &kernel, 0.5).unwrap_err();
        assert_eq!(err, FilterError::EmptyRankNeighborhood);
    }

    #[test]
    fn rank_rejects_non_scalar_pixel_type() {
        let img = Image::new(&[3, 3], PixelId::ComplexFloat32);
        let kernel = StructuringElement::box_(&[1, 1]);
        let err = rank(&img, &kernel, 0.5).unwrap_err();
        assert_eq!(
            err,
            FilterError::Core(sitk_core::Error::RequiresScalarPixelType(
                PixelId::ComplexFloat32
            ))
        );
    }
}
