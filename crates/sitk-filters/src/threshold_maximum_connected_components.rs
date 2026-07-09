//! `ThresholdMaximumConnectedComponentsImageFilter`: bisection search for the
//! lower threshold that maximizes the number of connected components larger
//! than `MinimumObjectSizeInPixels`, reusing this crate's own
//! [`crate::binary_threshold`], [`crate::connected_component`], and
//! [`crate::relabel_component`] exactly as upstream chains
//! `BinaryThresholdImageFilter` -> `ConnectedComponentImageFilter` ->
//! `RelabelComponentImageFilter` internally.
//!
//! Verified against:
//!
//! - `Modules/Segmentation/ConnectedComponents/include/itkThresholdMaximumConnectedComponentsImageFilter.h(.hxx)`
//! - `itkRelabelComponentImageFilter.h(.hxx)` (`GetNumberOfObjects` semantics)
//! - `itkConnectedComponentImageFilter.h` (default connectivity)
//! - SimpleITK's `Code/BasicFilters/yaml/ThresholdMaximumConnectedComponentsImageFilter.yaml`
//!
//! ## The bisection runs in native `PixelType` units, not histogram bins
//!
//! `lowerBound`/`upperBound`/`midpoint*` are all `PixelType` (the *input*
//! image's own pixel type) holding raw intensity values, not bin indices --
//! unlike this crate's `Histogram`-based threshold calculators
//! (`crate::threshold`). The search interval starts at the image's actual
//! `[min, max]` (clamped above by `UpperBoundary`) and the loop terminates
//! once the interval narrows to `<= 2` raw intensity units.
//!
//! ## Preserved upstream quirk: `midpoint` is not the true center
//!
//! `GenerateData` computes the very first split point as
//!
//! ```text
//! midpoint = (upperBound - lowerBound) / 2
//! ```
//!
//! which is **half the span**, not `(lowerBound + upperBound) / 2` (the
//! actual midpoint of `[lowerBound, upperBound]`). This only coincides with
//! the true center when `lowerBound == 0`. Every subsequent `midpointL`/
//! `midpointR` recomputation *does* use the correct quarter-point formula
//! relative to whatever `midpoint` currently is
//! (`lowerBound + (midpoint - lowerBound) / 2` and
//! `upperBound - (upperBound - midpoint) / 2`), so this bug affects only the
//! search's starting point, not its self-correcting per-iteration algebra --
//! but for **integer pixel types with a nonzero `lowerBound`**, that first
//! `(midpoint - lowerBound)` can go negative (`midpoint < lowerBound`), and
//! for **unsigned pixel types** that negative subtraction wraps modulo
//! `2^bits` exactly as C++ unsigned-integer arithmetic is defined to
//! (`itk::Math`/plain `PixelType` ops, no saturation). [`bisect`] reproduces
//! this bit-for-bit via Rust's `wrapping_sub`/`wrapping_add` rather than
//! "fixing" the formula. For **floating-point pixel types**, this port uses
//! plain (non-wrapping) arithmetic, matching C++ float semantics -- no
//! modular wraparound is possible there, but the *biased starting point* bug
//! still applies, and floating images with an intensity span `<= 2.0` (e.g.
//! normalized to `[0, 1]` or `[-1, 1]`) never enter the loop at all, so
//! `threshold_value` is exactly that biased `span / 2` value.
//!
//! ## `UpperBoundary` is fixed for the whole search; `upperBound` is not
//!
//! `m_ThresholdFilter->SetUpperThreshold(m_UpperBoundary)` is set once,
//! before the loop, from the (type-saturating-clamped) `UpperBoundary`
//! parameter -- every `ComputeConnectedComponents()` call across the entire
//! bisection, and the final output, all binarize with that same fixed upper
//! bound. `upperBound` (the bisection search variable, initialized to
//! `min(image_max, UpperBoundary)`) only bounds where the *lower* threshold
//! candidate is searched; it is never itself used as a binarization bound.
//!
//! ## `NumberOfObjects` is 0 unless the loop runs at least once
//!
//! `m_NumberOfObjects` is only assigned inside the `while` loop body. A
//! search whose initial `(upperBound - lowerBound) <= 2` never enters the
//! loop, leaving [`ThresholdMaximumConnectedComponentsResult::number_of_objects`]
//! at its default `0` even though the final thresholded output can still
//! contain segmented objects.
//!
//! ## `GetNumberOfObjects` after relabeling
//!
//! `RelabelComponentImageFilter::m_NumberOfObjects` (`itkRelabelComponentImageFilter.hxx`)
//! starts as the total labeled-object count and is decremented by however
//! many objects fall below `MinimumObjectSize` -- i.e. it is the count of
//! objects that *survive* the minimum-size filter. Because
//! [`crate::relabel_component`] assigns surviving objects consecutive labels
//! `1..=N`, that count is exactly the maximum pixel value in its output, which
//! [`count_components`] uses instead of threading a separate count out of
//! `crate::label`.
//!
//! ## `outside_value != 0` breaks the internal search
//!
//! `m_ThresholdFilter`'s `InsideValue`/`OutsideValue` are set once, before
//! the loop, to whatever the caller configured (default `1`/`0`) -- so
//! every `ComputeConnectedComponents()` call during the search itself
//! binarizes with those same values, not a canonical `1`/`0`. But
//! [`crate::connected_component`] (like upstream's
//! `ConnectedComponentImageFilter`) always treats exactly pixel value `0`
//! as background and everything else as foreground. If `outside_value` is
//! left at its default `0`, this is harmless regardless of what
//! `inside_value` is (the search only needs foreground/background
//! separation, not a specific inside value). But a caller-chosen
//! **nonzero** `outside_value` makes *every* pixel nonzero after
//! binarization -- the whole image becomes one connected component for
//! every threshold candidate, so every comparison in [`bisect`] ties, the
//! loop always takes the "keep searching left" branch, and the search
//! degenerates toward `image_min` regardless of the image's actual
//! content. This is reproduced here exactly as upstream would behave, not
//! guarded against.
//!
//! ## Internal connectivity is always face-connected
//!
//! `ConnectedComponentImageFilter`'s `FullyConnected` defaults to `false`
//! and `ThresholdMaximumConnectedComponentsImageFilter` never overrides it;
//! this is not exposed by the yaml either. [`count_components`] always calls
//! [`crate::connected_component`] with `fully_connected = false`.
//!
//! ## `m_LowerBoundary` is dead
//!
//! The `m_LowerBoundary` member is assigned once in the constructor and
//! never read again anywhere in `GenerateData` or `ComputeConnectedComponents`
//! -- it has no observable effect on the algorithm and is not ported.
//!
//! ## Parameter defaults
//!
//! From `ThresholdMaximumConnectedComponentsImageFilter.yaml`:
//! `MinimumObjectSizeInPixels = 0` (no minimum), `UpperBoundary =
//! f64::MAX` (saturating-cast to the pixel type, matching the yaml's
//! `custom_itk_cast` clamp to `NumericTraits<PixelType>::max()`),
//! `InsideValue = 1`, `OutsideValue = 0`. Output pixel type is always
//! `UInt8` (`output_pixel_type: uint8_t`), matching
//! [`crate::binary_threshold`].

use crate::error::{FilterError, Result};
use crate::{binary_threshold, connected_component, relabel_component};
use sitk_core::{Image, Scalar, dispatch_scalar};

/// Result of [`threshold_maximum_connected_components`], mirroring
/// `GetThresholdValue()` / `GetNumberOfObjects()`.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdMaximumConnectedComponentsResult {
    pub image: Image,
    pub threshold_value: f64,
    pub number_of_objects: u64,
}

/// Per-pixel-type bisection arithmetic reproducing C++ `PixelType` overflow
/// semantics -- see the module doc's "Preserved upstream quirk" section.
trait Bisect: Scalar {
    fn wsub(self, rhs: Self) -> Self;
    fn wadd(self, rhs: Self) -> Self;
    fn half(self) -> Self;
}

macro_rules! impl_bisect_int {
    ($($t:ty),+ $(,)?) => {$(
        impl Bisect for $t {
            fn wsub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
            fn wadd(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
            fn half(self) -> Self { self / 2 }
        }
    )+};
}
impl_bisect_int!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! impl_bisect_float {
    ($($t:ty),+ $(,)?) => {$(
        impl Bisect for $t {
            fn wsub(self, rhs: Self) -> Self { self - rhs }
            fn wadd(self, rhs: Self) -> Self { self + rhs }
            fn half(self) -> Self { self / (2 as $t) }
        }
    )+};
}
impl_bisect_float!(f32, f64);

/// `ThresholdMaximumConnectedComponentsImageFilter::GenerateData`'s bisection
/// loop, generic over the native `PixelType` arithmetic -- see the module
/// doc. `count_at` mirrors `ComputeConnectedComponents()`.
fn bisect<T: Bisect>(
    lower0: T,
    upper0: T,
    mut count_at: impl FnMut(T) -> Result<u64>,
) -> Result<(T, u64)> {
    let mut lower_bound = lower0;
    let mut upper_bound = upper0;
    let mut midpoint = upper_bound.wsub(lower_bound).half();
    let mut midpoint_l = lower_bound.wadd(midpoint.wsub(lower_bound).half());
    let mut midpoint_r = upper_bound.wsub(upper_bound.wsub(midpoint).half());
    let mut number_of_objects = 0u64;

    let two = T::from_f64(2.0);
    while upper_bound.wsub(lower_bound) > two {
        let right_count = count_at(midpoint_r)?;
        let left_count = count_at(midpoint_l)?;

        if right_count > left_count {
            lower_bound = midpoint;
            midpoint = midpoint_r;
            number_of_objects = right_count;
        } else {
            upper_bound = midpoint;
            midpoint = midpoint_l;
            number_of_objects = left_count;
        }

        midpoint_l = lower_bound.wadd(midpoint.wsub(lower_bound).half());
        midpoint_r = upper_bound.wsub(upper_bound.wsub(midpoint).half());
    }

    Ok((midpoint, number_of_objects))
}

/// `ComputeConnectedComponents()`: binarize at `[lower, upper]`, label
/// connected components (always face-connected), relabel dropping objects
/// smaller than `minimum_object_size`, and return the surviving object
/// count -- see the module doc's "`GetNumberOfObjects` after relabeling".
fn count_components(
    img: &Image,
    lower: f64,
    upper: f64,
    inside_value: u8,
    outside_value: u8,
    minimum_object_size: u64,
) -> Result<u64> {
    let binarized = binary_threshold(img, lower, upper, inside_value, outside_value)?;
    let cc = connected_component(&binarized, false)?;
    let relabeled = relabel_component(&cc, minimum_object_size)?;
    let count = relabeled
        .to_f64_vec()
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);
    Ok(count.round() as u64)
}

fn threshold_max_cc_typed<T: Scalar + Bisect>(
    img: &Image,
    minimum_object_size: u32,
    upper_boundary: f64,
    inside_value: u8,
    outside_value: u8,
) -> Result<ThresholdMaximumConnectedComponentsResult> {
    let pixels = img.scalar_slice::<T>()?;
    if pixels.is_empty() {
        return Err(FilterError::DegenerateRange);
    }

    let mut image_min = pixels[0];
    let mut image_max = pixels[0];
    for &v in &pixels[1..] {
        if v < image_min {
            image_min = v;
        }
        if v > image_max {
            image_max = v;
        }
    }

    // SimpleITK's custom_itk_cast: `min(UpperBoundary, PixelType::max())`.
    // Rust's saturating `as` cast already clamps to T's representable
    // range, reproducing that `min` with no extra code.
    let upper_boundary_native = T::from_f64(upper_boundary);
    let upper_bound0 = if image_max < upper_boundary_native {
        image_max
    } else {
        upper_boundary_native
    };
    let minimum_object_size = minimum_object_size as u64;

    let (threshold_value, number_of_objects) = bisect(image_min, upper_bound0, |t: T| {
        count_components(
            img,
            t.as_f64(),
            upper_boundary_native.as_f64(),
            inside_value,
            outside_value,
            minimum_object_size,
        )
    })?;

    let out = binary_threshold(
        img,
        threshold_value.as_f64(),
        upper_boundary_native.as_f64(),
        inside_value,
        outside_value,
    )?;

    Ok(ThresholdMaximumConnectedComponentsResult {
        image: out,
        threshold_value: threshold_value.as_f64(),
        number_of_objects,
    })
}

/// `ThresholdMaximumConnectedComponentsImageFilter`: find the lower
/// threshold (searched over `[image_min, min(image_max, upper_boundary)]`)
/// that maximizes the number of connected components with at least
/// `minimum_object_size` pixels, then binarize the image at
/// `[threshold, upper_boundary]`. See the module doc for every quirk
/// reproduced here.
pub fn threshold_maximum_connected_components(
    img: &Image,
    minimum_object_size: u32,
    upper_boundary: f64,
    inside_value: u8,
    outside_value: u8,
) -> Result<ThresholdMaximumConnectedComponentsResult> {
    dispatch_scalar!(
        img.pixel_id(),
        threshold_max_cc_typed,
        img,
        minimum_object_size,
        upper_boundary,
        inside_value,
        outside_value
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    fn img_2d(w: usize, h: usize, data: Vec<u8>) -> Image {
        Image::from_vec(&[w, h], data).unwrap()
    }

    /// Two 3-column, 3-row blobs (intensity 200) directly adjacent to a
    /// 1-column bridge (intensity 100) with no background gap, flanked by
    /// background (0) columns. Thresholding at lower <= 100 keeps the
    /// bridge as foreground, merging everything (cols 1-7) into a single
    /// component; lower > 100 excludes the bridge (100 < lower), splitting
    /// into two 3x3 components (cols 1-3 and cols 5-7). Hand-derived
    /// component-count curve: 1 for lower in [0, 100], 2 for lower in
    /// (100, 200].
    ///
    /// Grid (9 wide x 3 tall), row-major values, x-fastest storage:
    /// ```text
    /// 0 200 200 200 100 200 200 200 0
    /// 0 200 200 200 100 200 200 200 0
    /// 0 200 200 200 100 200 200 200 0
    /// ```
    fn two_blobs_with_bridge() -> Image {
        let w = 9;
        let h = 3;
        let row = [0u8, 200, 200, 200, 100, 200, 200, 200, 0];
        let mut data = Vec::with_capacity(w * h);
        for _ in 0..h {
            data.extend_from_slice(&row);
        }
        img_2d(w, h, data)
    }

    /// [`two_blobs_with_bridge`] (9x3) plus two background rows and one
    /// isolated single-pixel speck (intensity 200) far from everything
    /// else -- same `image_min`/`image_max` (0/200), so the bisection path
    /// is identical to [`two_blobs_with_bridge`]'s own (every sampled
    /// midpoint keeps the speck foreground, adding a constant +1 to every
    /// count on both sides of every comparison, which changes no
    /// left-vs-right outcome). This isolates `minimum_object_size`'s effect
    /// to exactly the reported `number_of_objects`, not the search path.
    ///
    /// Grid (9 wide x 5 tall):
    /// ```text
    /// 0 200 200 200 100 200 200 200 0
    /// 0 200 200 200 100 200 200 200 0
    /// 0 200 200 200 100 200 200 200 0
    /// 0   0   0   0   0   0   0   0 0
    /// 0   0   0   0 200   0   0   0 0
    /// ```
    fn two_blobs_with_bridge_and_a_speck() -> Image {
        let w = 9;
        let blob_row = [0u8, 200, 200, 200, 100, 200, 200, 200, 0];
        let blank_row = [0u8; 9];
        let speck_row = [0u8, 0, 0, 0, 200, 0, 0, 0, 0];
        let mut data = Vec::with_capacity(w * 5);
        for _ in 0..3 {
            data.extend_from_slice(&blob_row);
        }
        data.extend_from_slice(&blank_row);
        data.extend_from_slice(&speck_row);
        img_2d(w, 5, data)
    }

    #[test]
    fn merges_below_bridge_and_fragments_above_known_optimum() {
        // image_min = 0, image_max = 200, span = 200 > 2, so the loop runs
        // and the biased-midpoint quirk (module doc) is inert here (lower
        // bound 0). Hand-traced bisection path (u8 arithmetic, no
        // wraparound since lower_bound starts at 0):
        //   iter1: L=count_at(50)=1 (merged), R=count_at(150)=2 (split) ->
        //     R>L: lower=100, mid=150, #obj=2
        //   iter2..6: L and R both land in (100, 200], tied at 2 -> always
        //     "else": upper shrinks toward 100 (mid: 125, 112, 106, 103, 101)
        //   iter7: L=count_at(100)=1 (100 >= 100, bridge included -> merged),
        //     R=count_at(102)=2 -> R>L: lower=101, mid=102, #obj=2
        //   upper(103) - lower(101) = 2, loop condition `> 2` is false ->
        //     terminate with threshold_value = 102, number_of_objects = 2.
        let img = two_blobs_with_bridge();
        let result = threshold_maximum_connected_components(&img, 0, f64::MAX, 1, 0).unwrap();
        assert_eq!(result.number_of_objects, 2);
        assert_eq!(result.threshold_value, 102.0);
        assert_eq!(result.image.pixel_id(), PixelId::UInt8);
        let out = result.image.scalar_slice::<u8>().unwrap();
        // Row: background, blob1 (cols 1-3), bridge-now-background (100 <
        // 102), blob2 (cols 5-7), background.
        assert_eq!(&out[0..9], &[0, 1, 1, 1, 0, 1, 1, 1, 0]);
        // Same pattern repeats for all 3 rows.
        assert_eq!(&out[9..18], &out[0..9]);
        assert_eq!(&out[18..27], &out[0..9]);
    }

    #[test]
    fn minimum_object_size_filters_small_components() {
        // On `two_blobs_with_bridge_and_a_speck`, the bisection path is
        // identical to `two_blobs_with_bridge`'s own (see that fixture's
        // doc) because the isolated speck is foreground at every sampled
        // midpoint in that trace, adding a constant +1 to both the left and
        // right count at every comparison -- changing no branch decision.
        // So both variants below converge to the same threshold_value
        // (102) and thus the same output pixels; only number_of_objects
        // differs by whether the size-1 speck survives relabeling.
        let img = two_blobs_with_bridge_and_a_speck();

        let unfiltered = threshold_maximum_connected_components(&img, 0, f64::MAX, 1, 0).unwrap();
        assert_eq!(unfiltered.threshold_value, 102.0);
        assert_eq!(unfiltered.number_of_objects, 3, "blob1 + blob2 + speck");

        let filtered = threshold_maximum_connected_components(&img, 2, f64::MAX, 1, 0).unwrap();
        assert_eq!(filtered.threshold_value, 102.0);
        assert_eq!(
            filtered.number_of_objects, 2,
            "the size-1 speck is dropped by minimum_object_size=2, blob1 + blob2 remain"
        );

        // MinimumObjectSizeInPixels affects only the search/count
        // bookkeeping -- the final output is a plain binary_threshold at
        // the converged threshold_value, so the speck pixel still shows as
        // foreground in *both* variants' output image.
        let unfiltered_out = unfiltered.image.scalar_slice::<u8>().unwrap();
        let filtered_out = filtered.image.scalar_slice::<u8>().unwrap();
        assert_eq!(unfiltered_out, filtered_out);
        let speck_index = 4 * 9 + 4; // row 4, col 4
        assert_eq!(unfiltered_out[speck_index], 1);
    }

    #[test]
    fn upper_boundary_excludes_pixels_above_it() {
        // Three intensity levels: 0 (background), 100 (mid blob), 250 (hot
        // spot). Capping upper_boundary at 150 excludes the 250 pixels from
        // ever being counted as foreground, regardless of the search --
        // BinaryThresholdImageFilter's upper bound is fixed at
        // upper_boundary for the whole run (module doc).
        let w = 6;
        let h = 1;
        let data = vec![0u8, 100, 100, 100, 250, 250];
        let img = img_2d(w, h, data);

        let result = threshold_maximum_connected_components(&img, 0, 150.0, 1, 0).unwrap();
        let out = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(
            &out[4..6],
            &[0, 0],
            "pixels above upper_boundary are excluded"
        );
    }

    #[test]
    fn inside_and_outside_values_plumb_through_when_outside_stays_zero() {
        // Same trace as `merges_below_bridge_and_fragments_above_known_optimum`
        // (outside_value=0 keeps `connected_component`'s background
        // detection working, module doc's "outside_value != 0" caveat does
        // not apply) -- only inside_value changes, from 1 to 7.
        let img = two_blobs_with_bridge();
        let result = threshold_maximum_connected_components(&img, 0, f64::MAX, 7, 0).unwrap();
        assert_eq!(result.threshold_value, 102.0);
        assert_eq!(result.number_of_objects, 2);
        let out = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(&out[0..9], &[0, 7, 7, 7, 0, 7, 7, 7, 0]);
    }

    #[test]
    fn nonzero_outside_value_breaks_the_search_and_degenerates_to_image_min() {
        // module doc's "outside_value != 0 breaks the internal search":
        // with outside_value=3 (nonzero), every binarized pixel is
        // foreground (`connected_component` only treats exactly 0 as
        // background), so the whole 9x3 grid is always a single component
        // and every bisection comparison ties. The `else` branch (ties go
        // there, since `right_count > left_count` is false on a tie) then
        // runs every iteration, shrinking upper_bound toward lower_bound
        // (0) every time -- hand-traced over 7 iterations (u8 arithmetic)
        // to threshold_value=0, number_of_objects=1.
        let img = two_blobs_with_bridge();
        let result = threshold_maximum_connected_components(&img, 0, f64::MAX, 7, 3).unwrap();
        assert_eq!(result.threshold_value, 0.0);
        assert_eq!(result.number_of_objects, 1);
        let out = result.image.scalar_slice::<u8>().unwrap();
        // threshold_value=0 and upper_boundary saturates to u8::MAX=255,
        // so every pixel (0, 100, or 200) falls in [0, 255] -- all inside.
        assert!(out.iter().all(|&v| v == 7));
    }

    #[test]
    fn constant_image_reports_zero_objects_and_never_enters_the_loop() {
        // image_min == image_max, so (upper_bound - lower_bound) == 0 <= 2:
        // the bisection loop never runs, m_NumberOfObjects stays at its
        // default 0 (module doc), even though the resulting output is a
        // single all-foreground "object".
        let img = img_2d(3, 3, vec![9u8; 9]);
        let result = threshold_maximum_connected_components(&img, 0, f64::MAX, 1, 0).unwrap();
        assert_eq!(result.number_of_objects, 0);
    }

    #[test]
    fn negative_constant_image_pins_the_biased_midpoint_quirk() {
        // A constant image at a negative value under a signed pixel type:
        // image_min == image_max == -50, so the loop never runs (span 0).
        // threshold_value = midpoint = (upperBound - lowerBound) / 2 =
        // (-50 - -50) / 2 = 0 -- NOT the true center -50 (module doc's
        // "Preserved upstream quirk"). Because threshold_value (0) exceeds
        // the actual constant pixel value (-50), every pixel fails the
        // `>= threshold_value` test and the output is entirely background,
        // despite the input being a single uniform intensity.
        let img = Image::from_vec(&[3, 1], vec![-50i16, -50, -50]).unwrap();
        let result = threshold_maximum_connected_components(&img, 0, f64::MAX, 1, 0).unwrap();
        assert_eq!(result.threshold_value, 0.0);
        let out = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(out, &[0, 0, 0]);
    }

    #[test]
    fn empty_image_errors() {
        let img = Image::from_vec(&[0, 0], Vec::<u8>::new()).unwrap();
        assert!(matches!(
            threshold_maximum_connected_components(&img, 0, f64::MAX, 1, 0),
            Err(FilterError::DegenerateRange)
        ));
    }
}
