//! Projection filters: accumulate an image along one axis, collapsing that
//! axis to size 1.
//!
//! Ported from `itk::ProjectionImageFilter` (base geometry/iteration) plus one
//! accumulator functor per filter:
//! - `Modules/Filtering/ImageStatistics/include/itkProjectionImageFilter.h` /
//!   `.hxx`
//! - `itkMeanProjectionImageFilter.h` ([`mean_projection`])
//! - `itkMedianProjectionImageFilter.h` ([`median_projection`])
//! - `itkMaximumProjectionImageFilter.h` ([`maximum_projection`])
//! - `itkMinimumProjectionImageFilter.h` ([`minimum_projection`])
//! - `itkSumProjectionImageFilter.h` ([`sum_projection`])
//! - `itkStandardDeviationProjectionImageFilter.h`
//!   ([`standard_deviation_projection`])
//! - `itkBinaryProjectionImageFilter.h` ([`binary_projection`])
//!
//! Output pixel type per filter comes from SimpleITK's
//! `Code/BasicFilters/yaml/*ProjectionImageFilter.yaml`, not the raw ITK
//! default: `Mean`/`Sum`/`StandardDeviation` declare `output_pixel_type:
//! NumericTraits<InputPixelType>::RealType` (see [`real_type`]); the rest
//! (`Median`/`Maximum`/`Minimum`/`Binary`) declare none, so output type
//! matches input. `BinaryProjectionImageFilter.yaml` also overrides ITK's own
//! `ForegroundValue`/`BackgroundValue` defaults (`NumericTraits::max()` /
//! `NonpositiveMin()`) with fixed `1.0`/`0.0`, and casts both to the *input*
//! pixel type (`pixeltype: Input` on both members) even though ITK's C++
//! signature types `BackgroundValue` as the output pixel type — moot here
//! since output type equals input type for this filter.
//!
//! ## Collapsed-axis geometry, verbatim including a real upstream quirk
//!
//! Every filter here shares `ProjectionImageFilter::GenerateOutputInformation`,
//! and since SimpleITK's generated wrappers for this whole family use
//! `template_code_filename: ImageFilter` (output dimension always equals
//! input dimension), only the `InputImageDimension == OutputImageDimension`
//! branch is ever reachable — that's the only branch ported here. For that
//! branch: a retained axis keeps its size/spacing/origin; the projected axis
//! collapses to size 1, with `outSpacing[axis] = inSpacing[axis] *
//! inputSize[axis]` (one output pixel spans the whole projected extent) and
//! direction copied unchanged (same-dimension in/out is a straight
//! `outDirection[i][j] = inDirection[i][j]` copy).
//!
//! The collapsed axis's origin shift is copied *verbatim* from the .hxx:
//! `outOrigin[i] = inOrigin[i] + (i - 1) * inSpacing[i] / 2`, where `i` is the
//! loop variable, which in that branch equals `m_ProjectionDimension` itself
//! — **not** `inputSize[i]`. A physically-sensible shift centering the
//! collapsed extent would need `(inputSize[axis] - 1)`, not `(axis - 1)`; this
//! reads like a bug (using the axis index instead of the axis's pixel count),
//! but it is what current ITK (checked against `v6.0b02-5846-ge46eb723a5`,
//! `Modules/Filtering/ImageStatistics/include/itkProjectionImageFilter.hxx`)
//! actually computes, unchanged back through this checkout's full history for
//! this file (only whitespace/style reformatting touched the line). Per this
//! crate's porting rule (match upstream exactly, cite it, don't silently
//! "fix" it), it is reproduced here bit-for-bit rather than corrected.
//!
//! ITK's `i` is `unsigned int`, so for `axis == 0` — SimpleITK's own default
//! `ProjectionDimension` (`default: 0u` in every yaml above) — `i - 1`
//! silently wraps to `UINT_MAX` (a huge origin shift), rather than picking the
//! "no projection dimension set" (last-axis) default ITK's own C++
//! constructor uses; SimpleITK's generated code always calls
//! `SetProjectionDimension` unconditionally, so this wraparound is live for
//! every default-constructed call in SimpleITK. This port reproduces the
//! 32-bit wraparound with `(axis as u32).wrapping_sub(1)`.

use crate::error::{FilterError, Result};
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar};

/// `NumericTraits<T>::RealType` pixel-type mapping used by
/// [`mean_projection`], [`sum_projection`], and
/// [`standard_deviation_projection`]: stays `Float32` for a `Float32` input,
/// promotes everything else (every integer type, and `Float64` itself) to
/// `Float64`.
fn real_type(id: PixelId) -> PixelId {
    match id {
        PixelId::Float32 => PixelId::Float32,
        _ => PixelId::Float64,
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

/// Reduce every line along `axis` through `reduce`, producing the collapsed
/// (`size[axis] = 1`) output buffer. Mirrors `DynamicThreadedGenerateData`'s
/// `ImageLinearConstIteratorWithIndex` sweep: one call to the accumulator's
/// `Initialize()` + per-pixel `operator()` + `GetValue()` per line, here
/// folded into a single `reduce(&line)` call.
fn project_lines(
    vals: &[f64],
    size: &[usize],
    axis: usize,
    reduce: &impl Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let dim = size.len();
    let in_strides = strides(size);
    let n = size[axis];

    let mut out_size = size.to_vec();
    out_size[axis] = 1;
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();

    let mut out_vals = vec![0.0f64; out_count];
    let mut line = Vec::with_capacity(n);
    for (o, slot) in out_vals.iter_mut().enumerate() {
        line.clear();
        let mut base = 0usize;
        for d in 0..dim {
            if d == axis {
                continue;
            }
            let oi = (o / out_strides[d]) % out_size[d];
            base += oi * in_strides[d];
        }
        for k in 0..n {
            line.push(vals[base + k * in_strides[axis]]);
        }
        *slot = reduce(&line);
    }
    out_vals
}

/// Shared driver for every projection filter: validate `axis`, reduce every
/// line along it, and rebuild the collapsed-axis geometry (see the module
/// doc for the exact, verbatim-from-ITK origin/spacing formula).
///
/// Errors with [`FilterError::InvalidDirection`] if `axis` is not a valid
/// axis of `img` (`itkProjectionImageFilter.hxx`'s own
/// `m_ProjectionDimension >= TInputImage::ImageDimension` check).
fn project(
    img: &Image,
    axis: usize,
    output_id: PixelId,
    reduce: impl Fn(&[f64]) -> f64,
) -> Result<Image> {
    let dim = img.dimension();
    if axis >= dim {
        return Err(FilterError::InvalidDirection {
            direction: axis,
            dimension: dim,
        });
    }

    let in_size = img.size();
    let in_spacing = img.spacing();
    let in_origin = img.origin();

    let in_vals = img.to_f64_vec()?;
    let out_vals = project_lines(&in_vals, in_size, axis, &reduce);

    let mut out_size = in_size.to_vec();
    out_size[axis] = 1;
    let mut out = crate::image_from_f64(output_id, &out_size, img, &out_vals)?;

    let mut out_spacing = in_spacing.to_vec();
    out_spacing[axis] = in_spacing[axis] * in_size[axis] as f64;
    out.set_spacing(&out_spacing)?;

    let mut out_origin = in_origin.to_vec();
    let shift = (axis as u32).wrapping_sub(1) as f64 * in_spacing[axis] / 2.0;
    out_origin[axis] = in_origin[axis] + shift;
    out.set_origin(&out_origin)?;

    Ok(out)
}

/// `MeanProjectionImageFilter`: `sum(line) / line.len()`. Output pixel type is
/// `NumericTraits<InputPixelType>::RealType` (see [`real_type`]).
///
/// On an empty line (a projected axis of size 0) this is `0.0 / 0 == NaN`,
/// matching `MeanAccumulator::GetValue`'s unconditional floating-point
/// division — no special case needed, IEEE division already agrees with it.
pub fn mean_projection(img: &Image, projection_dimension: usize) -> Result<Image> {
    let output_id = real_type(img.pixel_id());
    project(img, projection_dimension, output_id, |line| {
        line.iter().sum::<f64>() / line.len() as f64
    })
}

/// `SumProjectionImageFilter`: `sum(line)`. Output pixel type is
/// `NumericTraits<InputPixelType>::RealType` (see [`real_type`]).
pub fn sum_projection(img: &Image, projection_dimension: usize) -> Result<Image> {
    let output_id = real_type(img.pixel_id());
    project(img, projection_dimension, output_id, |line| {
        line.iter().sum::<f64>()
    })
}

/// `StandardDeviationProjectionImageFilter`: sample standard deviation of
/// `line` (divisor `line.len() - 1`, matching
/// `StandardDeviationAccumulator::GetValue`'s `std::sqrt(squaredSum /
/// (m_Size - 1))`). Output pixel type is
/// `NumericTraits<InputPixelType>::RealType` (see [`real_type`]).
///
/// `line.len() <= 1` returns `0.0` exactly (`GetValue`'s explicit
/// divide-by-zero guard), rather than `0.0 / 0.0`.
pub fn standard_deviation_projection(img: &Image, projection_dimension: usize) -> Result<Image> {
    let output_id = real_type(img.pixel_id());
    project(img, projection_dimension, output_id, |line| {
        let n = line.len();
        if n <= 1 {
            return 0.0;
        }
        let mean = line.iter().sum::<f64>() / n as f64;
        let squared_sum: f64 = line.iter().map(|&v| (v - mean) * (v - mean)).sum();
        (squared_sum / (n as f64 - 1.0)).sqrt()
    })
}

/// `MedianProjectionImageFilter`: the `line.len() / 2`-th order statistic
/// (0-indexed) of `line`, matching `MedianAccumulator::GetValue`'s
/// `m_Values.begin() + m_Values.size() / 2` passed to `std::nth_element`.
///
/// For an even-length line this is the **upper** of the two middle values
/// (e.g. `[1, 2, 3, 4]` → index `4 / 2 == 2` → `3`), not their average — ITK
/// never averages here. Output pixel type matches the input (no
/// `output_pixel_type` override in `MedianProjectionImageFilter.yaml`).
pub fn median_projection(img: &Image, projection_dimension: usize) -> Result<Image> {
    project(img, projection_dimension, img.pixel_id(), |line| {
        let mut sorted = line.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("image pixel values are finite"));
        sorted[sorted.len() / 2]
    })
}

/// `MaximumProjectionImageFilter`: `max(line)`. Output pixel type matches the
/// input (no `output_pixel_type` override in
/// `MaximumProjectionImageFilter.yaml`).
pub fn maximum_projection(img: &Image, projection_dimension: usize) -> Result<Image> {
    project(img, projection_dimension, img.pixel_id(), |line| {
        line.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    })
}

/// `MinimumProjectionImageFilter`: `min(line)`. Output pixel type matches the
/// input (no `output_pixel_type` override in
/// `MinimumProjectionImageFilter.yaml`).
pub fn minimum_projection(img: &Image, projection_dimension: usize) -> Result<Image> {
    project(img, projection_dimension, img.pixel_id(), |line| {
        line.iter().copied().fold(f64::INFINITY, f64::min)
    })
}

/// `BinaryProjectionImageFilter`: `foreground_value` if any pixel along the
/// line equals `foreground_value`, else `background_value`
/// (`BinaryAccumulator::operator()`/`GetValue`). Output pixel type matches the
/// input (no `output_pixel_type` override in
/// `BinaryProjectionImageFilter.yaml`).
///
/// `foreground_value`/`background_value` are narrowed through the input pixel
/// type and back (`static_cast<InputPixelType>` in SimpleITK's generated
/// wrapper, both members are `pixeltype: Input`) before comparison, so a
/// fractional or out-of-range value truncates/saturates exactly as it would
/// going through the native pixel type, rather than comparing raw `f64`s.
/// SimpleITK's own defaults (not ITK's raw `NumericTraits::max()` /
/// `NonpositiveMin()`) are `foreground_value = 1.0`, `background_value =
/// 0.0` (`BinaryProjectionImageFilter.yaml`); pass those explicitly to match.
pub fn binary_projection(
    img: &Image,
    projection_dimension: usize,
    foreground_value: f64,
    background_value: f64,
) -> Result<Image> {
    fn round_trip<T: Scalar>(v: f64) -> f64 {
        T::from_f64(v).as_f64()
    }
    let foreground = dispatch_scalar!(img.pixel_id(), round_trip, foreground_value);
    let background = dispatch_scalar!(img.pixel_id(), round_trip, background_value);
    project(img, projection_dimension, img.pixel_id(), |line| {
        if line.contains(&foreground) {
            foreground
        } else {
            background
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    /// 2x3 image (asymmetric), first-index-fastest (x fastest):
    /// ```text
    /// y=0: x=0,1 -> 0,1
    /// y=1: x=0,1 -> 2,3
    /// y=2: x=0,1 -> 4,5
    /// ```
    fn asymmetric_2d() -> Image {
        Image::from_vec(&[2, 3], vec![0i32, 1, 2, 3, 4, 5]).unwrap()
    }

    // ---- hand-computed values, axis 0 (x, size 2) ----

    #[test]
    fn axis0_hand_computed_reductions() {
        let img = asymmetric_2d();
        // lines: [0,1], [2,3], [4,5]
        assert_eq!(
            mean_projection(&img, 0)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap(),
            &[0.5, 2.5, 4.5]
        );
        assert_eq!(
            sum_projection(&img, 0)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap(),
            &[1.0, 5.0, 9.0]
        );
        assert_eq!(
            maximum_projection(&img, 0)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[1, 3, 5]
        );
        assert_eq!(
            minimum_projection(&img, 0)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[0, 2, 4]
        );
        // n=2 per line: median picks the upper (larger) of the two, per the
        // nth_element(size/2) rule, not the average.
        assert_eq!(
            median_projection(&img, 0)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[1, 3, 5]
        );
    }

    // ---- hand-computed values, axis 1 (y, size 3) ----

    #[test]
    fn axis1_hand_computed_reductions() {
        let img = asymmetric_2d();
        // lines: [0,2,4], [1,3,5]
        assert_eq!(
            mean_projection(&img, 1)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap(),
            &[2.0, 3.0]
        );
        assert_eq!(
            sum_projection(&img, 1)
                .unwrap()
                .scalar_slice::<f64>()
                .unwrap(),
            &[6.0, 9.0]
        );
        assert_eq!(
            maximum_projection(&img, 1)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[4, 5]
        );
        assert_eq!(
            minimum_projection(&img, 1)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[0, 1]
        );
        // n=3 per line: median picks the exact middle (index 3/2 == 1).
        assert_eq!(
            median_projection(&img, 1)
                .unwrap()
                .scalar_slice::<i32>()
                .unwrap(),
            &[2, 3]
        );
    }

    // ---- output pixel types ----

    #[test]
    fn mean_sum_stddev_promote_integer_input_to_float64() {
        let img = asymmetric_2d(); // Int32
        assert_eq!(
            mean_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Float64
        );
        assert_eq!(
            sum_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Float64
        );
        assert_eq!(
            standard_deviation_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Float64
        );
    }

    #[test]
    fn mean_sum_stddev_keep_float32_input_as_float32() {
        let img = Image::from_vec(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            mean_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Float32
        );
        assert_eq!(
            sum_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Float32
        );
        assert_eq!(
            standard_deviation_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Float32
        );
    }

    #[test]
    fn median_maximum_minimum_binary_keep_input_pixel_type() {
        let img = asymmetric_2d(); // Int32
        assert_eq!(
            median_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Int32
        );
        assert_eq!(
            maximum_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Int32
        );
        assert_eq!(
            minimum_projection(&img, 0).unwrap().pixel_id(),
            PixelId::Int32
        );
        assert_eq!(
            binary_projection(&img, 0, 1.0, 0.0).unwrap().pixel_id(),
            PixelId::Int32
        );
    }

    // ---- standard deviation: divisor and degenerate size ----

    #[test]
    fn standard_deviation_uses_n_minus_1_divisor() {
        // Matches this crate's `statistics()` test: sample variance of
        // [2,4,4,6] is 8/3 (divisor n-1=3, not n=4).
        let img = Image::from_vec(&[4, 1], vec![2.0f64, 4.0, 4.0, 6.0]).unwrap();
        let out = standard_deviation_projection(&img, 0).unwrap();
        let v = out.scalar_slice::<f64>().unwrap()[0];
        assert!((v - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn standard_deviation_of_constant_line_is_zero() {
        let img = Image::from_vec(&[4, 1], vec![5.0f64, 5.0, 5.0, 5.0]).unwrap();
        let out = standard_deviation_projection(&img, 0).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[0.0]);
    }

    #[test]
    fn standard_deviation_of_size_one_line_is_zero_not_nan() {
        // Axis 1 already has size 1: every line has exactly one element.
        let img = Image::from_vec(&[3, 1], vec![1.0f64, 2.0, 3.0]).unwrap();
        let out = standard_deviation_projection(&img, 1).unwrap();
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[0.0, 0.0, 0.0]);
    }

    // ---- size-1 axis projection is identity-shaped ----

    #[test]
    fn projecting_an_already_size_one_axis_is_identity_for_values_and_shape() {
        let img = Image::from_vec(&[3, 1, 2], (0..6i32).collect()).unwrap();
        for f in [mean_projection, sum_projection] {
            let out = f(&img, 1).unwrap();
            assert_eq!(out.size(), img.size());
            assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
        }
        let out = median_projection(&img, 1).unwrap();
        assert_eq!(out.size(), img.size());
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
        let out = maximum_projection(&img, 1).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
        let out = minimum_projection(&img, 1).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    // ---- binary_projection ----

    #[test]
    fn binary_projection_mixed_foreground_background() {
        // 2x2: line x=0 is [200, 0] (has foreground), line x=1 is [0, 0] (no
        // foreground).
        let img = Image::from_vec(&[2, 2], vec![200u8, 0, 0, 0]).unwrap();
        let out = binary_projection(&img, 1, 200.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[200, 0]);
    }

    #[test]
    fn binary_projection_simpleitk_defaults_1_and_0() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 0, 0, 0]).unwrap();
        let out = binary_projection(&img, 1, 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 0]);
    }

    // ---- projection_dimension validation ----

    #[test]
    fn out_of_range_projection_dimension_errors() {
        let img = asymmetric_2d();
        assert_eq!(
            mean_projection(&img, 2),
            Err(FilterError::InvalidDirection {
                direction: 2,
                dimension: 2
            })
        );
    }

    // ---- collapsed-axis geometry ----

    #[test]
    fn geometry_spacing_and_size_of_collapsed_axis() {
        let mut img = asymmetric_2d();
        img.set_spacing(&[2.0, 3.0]).unwrap();
        let out = mean_projection(&img, 0).unwrap();
        assert_eq!(out.size(), &[1, 3]);
        // outSpacing[axis] = inSpacing[axis] * inputSize[axis] = 2.0 * 2.
        assert_eq!(out.spacing(), &[4.0, 3.0]);
        // retained axis 1 keeps its own size/spacing.
        assert_eq!(out.spacing()[1], img.spacing()[1]);
    }

    #[test]
    fn geometry_origin_shift_matches_axis_index_not_axis_size() {
        // itkProjectionImageFilter.hxx: outOrigin[i] = inOrigin[i] +
        // (i - 1) * inSpacing[i] / 2, where `i` is the *axis index*, not the
        // axis's pixel count. Use a 3-D image so axis 1 (shift (1-1)=0) and
        // axis 2 (shift (2-1)=1) are both directly reachable without hitting
        // the axis-0 wraparound (covered separately below).
        let mut img = Image::new(&[2, 2, 2], PixelId::Float64);
        img.set_spacing(&[1.0, 5.0, 7.0]).unwrap();
        img.set_origin(&[10.0, 20.0, 30.0]).unwrap();

        let out1 = mean_projection(&img, 1).unwrap();
        // (1 - 1) * 5.0 / 2 == 0.0
        assert_eq!(out1.origin()[1], 20.0);

        let out2 = mean_projection(&img, 2).unwrap();
        // (2 - 1) * 7.0 / 2 == 3.5
        assert_eq!(out2.origin()[2], 33.5);

        // Retained axes' origin is untouched on both.
        assert_eq!(out1.origin()[0], 10.0);
        assert_eq!(out2.origin()[0], 10.0);
    }

    #[test]
    fn geometry_origin_shift_wraps_for_axis_zero_matching_itk_unsigned_underflow() {
        // SimpleITK's own default ProjectionDimension is 0u, which is exactly
        // where ITK's `unsigned int i - 1` underflows to `UINT_MAX` instead
        // of the "sensible" shift. This port reproduces that wraparound
        // rather than silently avoiding it.
        let mut img = Image::new(&[2, 2], PixelId::Float64);
        img.set_spacing(&[2.0, 1.0]).unwrap();
        img.set_origin(&[0.0, 0.0]).unwrap();
        let out = mean_projection(&img, 0).unwrap();
        let expected_shift = (0u32.wrapping_sub(1)) as f64 * 2.0 / 2.0;
        assert_eq!(out.origin()[0], expected_shift);
        assert!(expected_shift > 1.0e9); // sanity: this really is huge, not 0.
    }

    #[test]
    fn geometry_direction_is_copied_unchanged() {
        let mut img = Image::new(&[2, 2], PixelId::Float64);
        let theta = std::f64::consts::FRAC_PI_6;
        img.set_direction(&[theta.cos(), -theta.sin(), theta.sin(), theta.cos()])
            .unwrap();
        let out = mean_projection(&img, 1).unwrap();
        assert_eq!(out.direction(), img.direction());
    }
}
