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
//! - `itkBinaryThresholdProjectionImageFilter.h`
//!   ([`binary_threshold_projection`])
//!
//! Output pixel type per filter comes from SimpleITK's
//! `Code/BasicFilters/yaml/*ProjectionImageFilter.yaml`, not the raw ITK
//! default: `Mean`/`Sum`/`StandardDeviation` declare `output_pixel_type:
//! NumericTraits<InputPixelType>::RealType` (see `real_type`); the rest
//! (`Median`/`Maximum`/`Minimum`/`Binary`) declare none, so output type
//! matches input. `BinaryProjectionImageFilter.yaml` also overrides ITK's own
//! `ForegroundValue`/`BackgroundValue` defaults (`NumericTraits::max()` /
//! `NonpositiveMin()`) with fixed `1.0`/`0.0`, and casts both to the *input*
//! pixel type (`pixeltype: Input` on both members) even though ITK's C++
//! signature types `BackgroundValue` as the output pixel type — moot here
//! since output type equals input type for this filter.
//! `BinaryThresholdProjectionImageFilter.yaml` fixes `output_pixel_type:
//! uint8_t` unconditionally (not input-following, unlike its
//! `Median`/`Maximum`/`Minimum`/`Binary` siblings), and types
//! `ForegroundValue`/`BackgroundValue` as plain `uint8_t` (`pixeltype:
//! Output`, no `double` round-trip) — this crate's convention for other
//! fixed-output-type `pixeltype: Output` parameters (e.g.
//! [`crate::filters::binary_threshold`]'s `inside`/`outside`).
//!
//! ## Collapsed-axis geometry (upstream origin bug fixed here — §1.1)
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
//! The collapsed axis's origin is **corrected** here relative to released ITK.
//! Upstream computes `outOrigin[i] = inOrigin[i] + (i - 1) * inSpacing[i] / 2`
//! (`Modules/Filtering/ImageStatistics/include/itkProjectionImageFilter.hxx:88`
//! at `v6.0b02-5846-ge46eb723a5`), where `i` is the loop variable — the axis
//! *index*, not the axis's pixel count. Since `i` is `unsigned int`, `axis == 0`
//! (SimpleITK's `default: 0u` in every yaml above, always applied because the
//! generated code calls `SetProjectionDimension` unconditionally) wraps `i - 1`
//! to `UINT_MAX`, shifting the origin by roughly `2^32 · spacing / 2`. For
//! `axis >= 1` the shift is small but still wrong. Reported as item B1 of ITK
//! issue #6575; fixed in this port to match the upstream fix branch
//! `bug-projection-collapsed-axis-origin`
//! (`BUG: Center collapsed-axis origin in projection and accumulate filters`).
//!
//! The corrected rule: the single output pixel (index `0` along `axis`) sits at
//! the center of the input's physical extent along `axis`. In continuous index
//! space that center is `(inputSize[axis] - 1) / 2` — an ITK image's start
//! index is folded in upstream as `inputIndex[axis] + (inputSize[axis] - 1) / 2`,
//! but a SimpleITK/`sitk_core` image always starts at index `0`, so the term
//! vanishes here. The physical shift is that continuous-index offset times the
//! spacing, carried through the direction matrix's `axis` **column**, so it has
//! a component in *every* coordinate:
//!
//! ```text
//! centerOffset  = (inputSize[axis] - 1) / 2 * inSpacing[axis]
//! outOrigin[d] += inDirection[d][axis] * centerOffset      for every d
//! ```
//!
//! For an identity direction this reduces to
//! `outOrigin[axis] = inOrigin[axis] + (inputSize[axis] - 1) * inSpacing[axis] / 2`
//! with the other components untouched.

use crate::core::{Image, PixelId, Scalar, dispatch_scalar};
use crate::filters::error::{FilterError, Result};

/// Output pixel-type mapping used by [`mean_projection`], [`sum_projection`],
/// and [`standard_deviation_projection`]: stays `Float32` for a `Float32`
/// input, promotes everything else to `Float64`. **Diverges from ITK**:
/// `NumericTraits<T>::RealType` is `double` for every scalar type *including*
/// `float` (itkNumericTraits.h:1349/1356), so upstream always outputs
/// `Float64`. Breaking to fix; tracked in the upstream-findings ledger §5.6
/// (same family as `math::real_type`/`lib.rs::real_pixel_id`).
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
/// line along it, and rebuild the collapsed-axis geometry (see the module doc
/// for the origin/spacing formula, including the §1.1 origin-centering fix).
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
    let mut out = crate::filters::image_from_f64(output_id, &out_size, img, &out_vals)?;

    let mut out_spacing = in_spacing.to_vec();
    out_spacing[axis] = in_spacing[axis] * in_size[axis] as f64;
    out.set_spacing(&out_spacing)?;

    // Center the collapsed axis on the input's physical extent: the single
    // output pixel (index 0 along `axis`) must land on the input's center
    // along that axis. The shift follows the direction matrix's `axis` column,
    // so it has a component in every coordinate. See the module doc.
    let in_direction = img.direction();
    let center_index = (in_size[axis] as f64 - 1.0) / 2.0;
    let center_offset = center_index * in_spacing[axis];
    let mut out_origin = in_origin.to_vec();
    for (d, o) in out_origin.iter_mut().enumerate() {
        *o += in_direction[d * dim + axis] * center_offset;
    }
    out.set_origin(&out_origin)?;

    Ok(out)
}

/// `MeanProjectionImageFilter`: `sum(line) / line.len()`. Output pixel type is
/// `NumericTraits<InputPixelType>::RealType` (see `real_type`).
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
/// `NumericTraits<InputPixelType>::RealType` (see `real_type`).
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
/// `NumericTraits<InputPixelType>::RealType` (see `real_type`).
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

/// `BinaryThresholdProjectionImageFilter`: `foreground_value` if any pixel
/// along the line is `>= threshold_value`, else `background_value`
/// (`Function::BinaryThresholdAccumulator::operator()`/`GetValue`: `if
/// (input >= m_ThresholdValue) m_IsForeground = true;`). Output pixel type is
/// fixed `UInt8` (see the module docs), unlike every sibling projection
/// filter above.
///
/// `threshold_value` is narrowed through the input pixel type and back
/// (`static_cast<InputPixelType>` in SimpleITK's generated wrapper,
/// `ThresholdValue` is `pixeltype: Input`) before comparison, via
/// `crate::filters::quantize_to_pixel_type` — the same narrowing
/// [`binary_projection`]'s `foreground_value`/`background_value` already use.
pub fn binary_threshold_projection(
    img: &Image,
    projection_dimension: usize,
    threshold_value: f64,
    foreground_value: u8,
    background_value: u8,
) -> Result<Image> {
    let threshold = crate::filters::quantize_to_pixel_type(img.pixel_id(), threshold_value);
    project(img, projection_dimension, PixelId::UInt8, |line| {
        if line.iter().any(|&v| v >= threshold) {
            foreground_value as f64
        } else {
            background_value as f64
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

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

    /// §1.1 fix: the collapsed axis's origin is centered on the *input's*
    /// physical extent along that axis — driven by `inputSize[axis]`, not by
    /// the axis index that upstream's `(i - 1)` accidentally uses.
    ///
    /// Sizes 2/3/4 are deliberately all different from their axis indices, so
    /// the correct value and the upstream `(i - 1) * spacing / 2` value differ
    /// on every axis. Image: size `[2, 3, 4]`, spacing `[1, 5, 7]`, origin
    /// `[10, 20, 30]`, identity direction. `shift = (size-1)/2 * spacing`:
    ///
    /// - axis 0: `(2-1)/2 * 1 = 0.5` → `origin[0] = 10 + 0.5 = 10.5`
    ///   (upstream: `(0-1)` wraps → `+2^31`; see the axis-zero test below)
    /// - axis 1: `(3-1)/2 * 5 = 5.0` → `origin[1] = 20 + 5.0 = 25.0`
    ///   (upstream would give `(1-1)*5/2 = 0` → `20.0`)
    /// - axis 2: `(4-1)/2 * 7 = 10.5` → `origin[2] = 30 + 10.5 = 40.5`
    ///   (upstream would give `(2-1)*7/2 = 3.5` → `33.5`)
    ///
    /// Cross-check axis 2 independently: the input's four pixel centers along
    /// z are at `30, 37, 44, 51`; their midpoint is `(30 + 51) / 2 = 40.5`.
    #[test]
    fn geometry_origin_centers_the_collapsed_axis_on_the_input_extent() {
        let mut img = Image::new(&[2, 3, 4], PixelId::Float64);
        img.set_spacing(&[1.0, 5.0, 7.0]).unwrap();
        img.set_origin(&[10.0, 20.0, 30.0]).unwrap();

        let out0 = mean_projection(&img, 0).unwrap();
        assert_eq!(out0.origin(), &[10.5, 20.0, 30.0]);

        let out1 = mean_projection(&img, 1).unwrap();
        assert_eq!(out1.origin(), &[10.0, 25.0, 30.0]);

        let out2 = mean_projection(&img, 2).unwrap();
        assert_eq!(out2.origin(), &[10.0, 20.0, 40.5]);
    }

    /// §1.1 fix, axis 0 specifically: SimpleITK's default `ProjectionDimension`
    /// is `0u`, which is exactly where upstream's `unsigned int i - 1`
    /// underflows to `UINT_MAX` and shifts the origin by `~2^31 * spacing`.
    /// The corrected shift is the ordinary centering one.
    ///
    /// Image: size `[2, 2]`, spacing `[2, 1]`, origin `[0, 0]`. The two pixel
    /// centers along x sit at `0.0` and `2.0`, so the extent's center is
    /// `1.0`; equivalently `(2-1)/2 * 2.0 = 1.0`. Upstream would produce
    /// `(0u - 1) as f64 * 2.0 / 2.0 = 4294967295.0`.
    #[test]
    fn geometry_origin_for_axis_zero_is_centered_not_unsigned_wrapped() {
        let mut img = Image::new(&[2, 2], PixelId::Float64);
        img.set_spacing(&[2.0, 1.0]).unwrap();
        img.set_origin(&[0.0, 0.0]).unwrap();
        let out = mean_projection(&img, 0).unwrap();
        assert_eq!(out.origin(), &[1.0, 0.0]);

        // The value upstream computes, asserted absent rather than merely
        // "small": (0u32 - 1) as f64 * 2.0 / 2.0.
        let upstream_wrapped = (0u32.wrapping_sub(1)) as f64 * 2.0 / 2.0;
        assert_eq!(upstream_wrapped, 4_294_967_295.0);
        assert_ne!(out.origin()[0], upstream_wrapped);
    }

    /// §1.1 fix, direction-aware half: the centering shift is a *physical*
    /// vector `direction[:, axis] * centerOffset`, so with a non-identity
    /// direction it has components in coordinates other than `axis`. Upstream
    /// only ever touched `outOrigin[axis]`.
    ///
    /// 2-D image, size `[2, 3]`, spacing `[1, 2]`, origin `[0, 0]`, direction
    /// the exact 90° rotation `[[0, -1], [1, 0]]` (row-major `[0,-1,1,0]`).
    /// Project axis 1: `centerOffset = (3-1)/2 * 2.0 = 2.0`. Column 1 of the
    /// direction matrix is `(direction[0][1], direction[1][1]) = (-1, 0)`, so
    /// `outOrigin = (0, 0) + (-1, 0) * 2.0 = (-2, 0)`.
    ///
    /// Cross-check by physical points: input index `(0, j)` maps to
    /// `origin + D * S * (0, j) = (-2j, 0)`, i.e. `(0,0)`, `(-2,0)`, `(-4,0)`
    /// for `j = 0,1,2` — midpoint `(-2, 0)`, matching. All values exact in
    /// binary floating point (`sin`/`cos` never enter).
    #[test]
    fn geometry_origin_centering_follows_the_direction_matrix_column() {
        let mut img = Image::new(&[2, 3], PixelId::Float64);
        img.set_spacing(&[1.0, 2.0]).unwrap();
        img.set_origin(&[0.0, 0.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let out = mean_projection(&img, 1).unwrap();
        assert_eq!(out.origin(), &[-2.0, 0.0]);
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

    // ---- binary_threshold_projection ----

    /// Threshold boundary is inclusive (`>=`): axis-0 lines of
    /// `asymmetric_2d` are `[0,1]`, `[2,3]`, `[4,5]`; at `threshold=3.0` the
    /// first line's max (1) falls short, the second line's max (3) exactly
    /// equals the threshold and still counts, the third (5) clears it.
    #[test]
    fn threshold_boundary_is_inclusive_at_the_exact_pixel_value() {
        let img = Image::from_vec(&[2, 3], vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let out = binary_threshold_projection(&img, 0, 3.0, 9, 2).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[2, 9, 9]);
    }

    /// The same lines just above the pixel value 3 (`threshold=3.5`):
    /// the second line's max (3) no longer qualifies, demonstrating `>=`
    /// rather than `>` right at the boundary (contrast with the `3.0` case
    /// above, where 3 exactly meeting the threshold still counted).
    #[test]
    fn threshold_boundary_excludes_a_value_just_below_it() {
        let img = Image::from_vec(&[2, 3], vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let out = binary_threshold_projection(&img, 0, 3.5, 9, 2).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[2, 2, 9]);
    }

    /// `threshold_value` is narrowed through the input pixel type before
    /// comparison: on an `Int32` image, `3.9` truncates to `3` (`as i32`),
    /// so it behaves exactly like the `3.0` boundary test above rather than
    /// excluding the pixel value 3.
    #[test]
    fn threshold_value_quantizes_through_input_pixel_type() {
        let img = asymmetric_2d(); // Int32: lines [0,1], [2,3], [4,5]
        let out = binary_threshold_projection(&img, 0, 3.9, 9, 2).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[2, 9, 9]);
    }

    /// Output pixel type is fixed `UInt8`, unlike every sibling projection
    /// filter (which keeps or promotes the input type).
    #[test]
    fn output_pixel_type_is_fixed_uint8() {
        let img = asymmetric_2d(); // Int32
        let out = binary_threshold_projection(&img, 0, 0.0, 1, 0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    /// Inherits `project`'s shared, §1.1-corrected origin-centering — same
    /// image and same hand-derived values as
    /// `geometry_origin_centers_the_collapsed_axis_on_the_input_extent`
    /// above, just through `binary_threshold_projection` instead of
    /// `mean_projection`, confirming every filter in the family shares the fix.
    #[test]
    fn origin_centering_is_shared_by_every_projection_filter() {
        let mut img = Image::new(&[2, 3, 4], PixelId::Float64);
        img.set_spacing(&[1.0, 5.0, 7.0]).unwrap();
        img.set_origin(&[10.0, 20.0, 30.0]).unwrap();

        // (2-1)/2 * 1 = 0.5
        let out0 = binary_threshold_projection(&img, 0, 0.0, 1, 0).unwrap();
        assert_eq!(out0.origin(), &[10.5, 20.0, 30.0]);

        // (3-1)/2 * 5 = 5.0
        let out1 = binary_threshold_projection(&img, 1, 0.0, 1, 0).unwrap();
        assert_eq!(out1.origin(), &[10.0, 25.0, 30.0]);

        // (4-1)/2 * 7 = 10.5
        let out2 = binary_threshold_projection(&img, 2, 0.0, 1, 0).unwrap();
        assert_eq!(out2.origin(), &[10.0, 20.0, 40.5]);
    }
}
