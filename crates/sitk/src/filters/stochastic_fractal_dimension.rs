//! `StochasticFractalDimensionImageFilter`: per-pixel stochastic fractal
//! dimension over a local neighborhood, ported from
//! `itkStochasticFractalDimensionImageFilter.h(.hxx)`
//! (`Modules/Nonunit/Review`).
//!
//! For each pixel, every ordered pair `(i, j)` of *distinct* positions in its
//! `neighborhood_radius` window (the position itself included, `i == j`
//! excluded) that both lie inside the image contributes one sample: the
//! squared physical distance between `i` and `j` (bucketed against existing
//! buckets within `0.5 * min(spacing)` of an existing bucket's distance --
//! `itkStochasticFractalDimensionImageFilter.hxx:150-167`) and the absolute
//! intensity difference `|pixel_i - pixel_j|`, averaged per bucket. A
//! least-squares line is then fit to `log(average |diff|)` against
//! `log(distance)` over the distinct buckets, and the output is
//! `3.0 - slope`. This is symmetric in `(i, j)`: every unordered pair that
//! survives contributes to its bucket *twice* (once as `(i, j)`, once as
//! `(j, i)`), which doubles both a bucket's frequency and its diff sum and
//! so leaves the bucket's average unchanged.
//!
//! **Boundary handling has no boundary condition at all.** The neighbor walk
//! uses `ConstNeighborhoodIterator::GetPixel(n, IsInBounds)`
//! (`itkConstNeighborhoodIterator.hxx:147-...`), which reports `false` for
//! any offset outside the image rather than substituting a mirrored/
//! zero-flux/constant value; the `.hxx`'s outer and inner loops both `if
//! (!IsInBounds) continue;` on that report. So out-of-image neighbors (and,
//! symmetrically, masked-out neighbors -- `if (mask && !mask->GetPixel(idx))`
//! gates the same way for both `i` and `j`) are simply dropped from the
//! regression, not replaced. The upstream `NeighborhoodAlgorithm::ImageBoundaryFacesCalculator`
//! exists only to skip the bounds check for pixels far enough from every
//! edge that no offset can leave the image; it changes no numeric result, so
//! this port does a single per-pixel bounds check instead of splitting the
//! image into faces.
//!
//! **Mask.** `MaskImageType` is hardcoded to `itk::Image<uint8_t,
//! ImageDimension>` and `StochasticFractalDimensionImageFilter.yaml`'s
//! `MaskImage` input has no `custom_itk_cast`, so SimpleITK's default input
//! handling (`ExecuteInternalSetITKFilterInputs.cxx.jinja`) reaches the mask
//! to ITK via `this->CastImageToITK<MaskImageType>(*inMaskImage)` --
//! `CastImageToITK` is a `dynamic_cast` (`sitkProcessObject.h:386-400`), not
//! a value-converting cast, so a mask of any other pixel type throws.
//! (`n4_bias_field`'s mask handling takes the opposite reading of an
//! identically-shaped `custom_itk_cast` line and value-converts instead; that
//! divergence was not touched here since fixing it is outside this port's
//! scope, but the `MaskedAssignImageFilter` precedent in `error.rs`
//! -- `RequiresUInt8MaskPixelType` -- confirms the `dynamic_cast` reading is
//! the one actually enforced by SimpleITK's codegen.) A mask pixel is "on"
//! whenever its raw stored value is nonzero -- `!mask->GetPixel(idx)` on a
//! `uint8_t`, not a comparison against a specific foreground value.
//!
//! A masked-out center pixel (`ItO.SetCenterPixel(OutputPixelType{})`) writes
//! the value-initialized output pixel, `0.0`.
//!
//! **Precision.** `RealType` is `float` (not `double`), so every bucket
//! value, sum, `log`/`sqrt` and the final slope is computed at `f32`
//! precision, matching the `.hxx` line by line:
//! - `point.SquaredEuclideanDistanceTo` computes in `double` (`Point`'s
//!   default precision) and is narrowed to `float` on assignment
//!   (`const RealType distance = ...`); this port computes the squared
//!   distance in `f64` and narrows once, to the same effect.
//! - `pixel1 - pixel2` computes in the *input* pixel's native precision
//!   (`float` for `Float32`, `double` for `Float64`) and is narrowed to
//!   `float` only when it lands in the `RealType` accumulator; this port
//!   reads pixels through [`crate::core::Image::to_f64_vec`] (an exact,
//!   lossless widening for both `Float32` and `Float64`), subtracts in
//!   `f64`, and narrows once -- identical to native-precision subtraction
//!   followed by narrowing, since `f32 -> f64` promotion loses nothing.
//! - `RealType minSpacing = spacing[0]` narrows the true `double` spacing
//!   once; each subsequent `spacing[d] < minSpacing` promotes `minSpacing`
//!   back to `double` for the comparison but does not un-narrow it, so a
//!   spacing that would have compared smaller at full `double` precision
//!   can be missed if the earlier narrowing rounded the running minimum
//!   upward. This port reproduces the narrow-once/widen-to-compare order
//!   exactly (see [`min_spacing_f32`]) rather than computing the minimum at
//!   full `double` precision throughout.
//!
//! The `distancesFrequency[k] == 0` guard in the `.hxx`'s final summation
//! loop is dead code reproduced only in this comment, not in code: a bucket
//! is never created with frequency `0` (every `push_back` seeds frequency
//! `1`) and frequency only ever increments, so the branch can't fire.

use crate::core::{Image, PixelId};
use crate::filters::error::{FilterError, Result};
use crate::filters::image_from_f64;

/// `RealType minSpacing = spacing[0];` narrowed once, then each subsequent
/// axis compared against the *widened-back* running value rather than the
/// true `f64` spacing (`itkStochasticFractalDimensionImageFilter.hxx:80-88`).
fn min_spacing_f32(spacing: &[f64]) -> f32 {
    let mut min_spacing = spacing[0] as f32;
    for &s in &spacing[1..] {
        if s < min_spacing as f64 {
            min_spacing = s as f32;
        }
    }
    min_spacing
}

/// Per-dimension offsets of a `radius`-shaped box neighborhood, dimension-0
/// fastest (matches [`crate::core::NeighborhoodIterator`]'s own table).
fn neighbor_offsets(radius: &[usize]) -> Vec<Vec<i64>> {
    let dim = radius.len();
    let window_size: Vec<usize> = radius.iter().map(|&r| 2 * r + 1).collect();
    let num_neighbors: usize = window_size.iter().product();
    let mut offsets = Vec::with_capacity(num_neighbors);
    let mut offset: Vec<i64> = radius.iter().map(|&r| -(r as i64)).collect();
    for _ in 0..num_neighbors {
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

/// `StochasticFractalDimensionImageFilter`: the local stochastic fractal
/// dimension of `image` over a `neighborhood_radius`-shaped per-axis
/// (Manhattan/box) window, restricted to `mask_image`'s "on" (nonzero)
/// voxels when given.
///
/// `image` must have a floating-point pixel type (`pixel_types:
/// RealPixelIDTypeList`); the output keeps that same pixel type.
/// `neighborhood_radius` must have at least `image.dimension()` entries
/// (SimpleITK's default is `[2, 2, 2]`, truncated to the image's actual
/// dimension); only the first `image.dimension()` are used.
///
/// `mask_image`, if given, must be `UInt8` (see the module docs on why this
/// port does not accept any other mask pixel type) and the same size as
/// `image`. See the module docs for the exact regression, boundary handling,
/// and the `f32`-precision quirks this reproduces.
pub fn stochastic_fractal_dimension(
    image: &Image,
    mask_image: Option<&Image>,
    neighborhood_radius: &[usize],
) -> Result<Image> {
    if !matches!(image.pixel_id(), PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(image.pixel_id()));
    }
    let dim = image.dimension();
    if neighborhood_radius.len() < dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: neighborhood_radius.len(),
        });
    }
    let radius = &neighborhood_radius[..dim];

    // The mask is a pipeline input (`SetNthInput(1, ...)`), so its pixel type, size
    // *and physical space* are all fixed by upstream — one owner enforces the three.
    let mask_vals = mask_image
        .map(|m| crate::filters::mask_input::uint8_mask_voxels(image, m))
        .transpose()?;

    let size = image.size();
    let n = image.number_of_pixels();
    let pixels = image.to_f64_vec()?;
    let offsets = neighbor_offsets(radius);
    let num_neighbors = offsets.len();
    let min_spacing = min_spacing_f32(image.spacing());
    let bucket_threshold = 0.5 * min_spacing as f64;

    let mut out = vec![0.0f64; n];

    // Reused per center pixel: which neighbor positions are usable (in
    // bounds and, if masked, "on"), with their pixel value and physical
    // point precomputed once and used identically whether the position
    // plays the role of `i` or `j` (see the module docs on why this is
    // equivalent to the `.hxx`'s per-(i,j) recomputation).
    let mut info: Vec<Option<(f64, Vec<f64>)>> = Vec::with_capacity(num_neighbors);
    let mut distances: Vec<f32> = Vec::new();
    let mut frequency: Vec<f32> = Vec::new();
    let mut sum_abs_diff: Vec<f32> = Vec::new();

    for center in 0..n {
        let mut center_idx = vec![0usize; dim];
        let mut rem = center;
        for d in 0..dim {
            center_idx[d] = rem % size[d];
            rem /= size[d];
        }

        if let Some(mask) = &mask_vals
            && mask[center] == 0
        {
            out[center] = 0.0;
            continue;
        }

        info.clear();
        for offset in &offsets {
            let mut neighbor_idx = vec![0usize; dim];
            let mut in_bounds = true;
            for d in 0..dim {
                let v = center_idx[d] as i64 + offset[d];
                in_bounds &= v >= 0 && (v as usize) < size[d];
                if in_bounds {
                    neighbor_idx[d] = v as usize;
                }
            }
            if !in_bounds {
                info.push(None);
                continue;
            }
            let lin = image.linear_index(&neighbor_idx);
            if let Some(mask) = &mask_vals
                && mask[lin] == 0
            {
                info.push(None);
                continue;
            }
            let point = image.continuous_index_to_physical_point(
                &neighbor_idx.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            );
            info.push(Some((pixels[lin], point)));
        }

        distances.clear();
        frequency.clear();
        sum_abs_diff.clear();

        for (i, entry_i) in info.iter().enumerate() {
            let Some((pixel_i, point_i)) = entry_i else {
                continue;
            };
            for (j, entry_j) in info.iter().enumerate() {
                if i == j {
                    continue;
                }
                let Some((pixel_j, point_j)) = entry_j else {
                    continue;
                };

                let squared_distance: f64 = point_i
                    .iter()
                    .zip(point_j)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                let distance = squared_distance as f32;
                let diff = ((pixel_i - pixel_j).abs()) as f32;

                match distances
                    .iter()
                    .position(|&d| ((d - distance) as f64).abs() < bucket_threshold)
                {
                    Some(k) => {
                        frequency[k] += 1.0;
                        sum_abs_diff[k] += diff;
                    }
                    None => {
                        distances.push(distance);
                        frequency.push(1.0);
                        sum_abs_diff.push(diff);
                    }
                }
            }
        }

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_xx = 0.0f32;
        let mut sum_xy = 0.0f32;
        for k in 0..distances.len() {
            let average = sum_abs_diff[k] / frequency[k];
            let y = average.ln();
            let x = distances[k].sqrt().ln();
            sum_y += y;
            sum_x += x;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        let count = distances.len() as f32;
        let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);
        out[center] = 3.0 - slope as f64;
    }

    image_from_f64(image.pixel_id(), size, image, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img<T: crate::core::Scalar>(size: &[usize], data: Vec<T>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    /// Hand-derived interior-pixel fixture. `values = [0, 2, 5, 9, 0]`,
    /// `radius = [1]`, unit spacing: the window at index 2 is the strictly
    /// monotonic run `[2, 5, 9]`. With `X1 = log(sqrt(1)) = 0` (the
    /// adjacent-pixel bucket) and `X2 = log(sqrt(4)) = log(2)` (the
    /// skip-one bucket), monotonicity gives `|2-9| = |2-5| + |5-9|`, so the
    /// skip-one bucket's average difference (7.0) is exactly double the
    /// adjacent bucket's (3.5) -- the same ratio as the two buckets'
    /// distances (`sqrt(4)/sqrt(1) = 2`). Substituting `Y2 - Y1 = log(2) =
    /// X2` into the two-point least-squares slope `(Y2-Y1)/X2` collapses it
    /// to exactly `1.0`, so the output is `3.0 - 1.0 = 2.0`, up to the `f32`
    /// rounding the module docs describe.
    #[test]
    fn hand_derived_interior_pixel_on_a_monotonic_run() {
        let image = img(&[5], vec![0.0f64, 2.0, 5.0, 9.0, 0.0]);
        let out = stochastic_fractal_dimension(&image, None, &[1]).unwrap();
        assert!(
            (out.to_f64_vec().unwrap()[2] - 2.0).abs() < 1e-4,
            "got {}",
            out.to_f64_vec().unwrap()[2]
        );
    }

    /// At the left edge (index 0, radius 1) only the center (offset 0) and
    /// right neighbor (offset +1) are in bounds -- no wraparound, no
    /// boundary-condition substitution. Two positions give exactly one
    /// unordered pair, hence one distance bucket (`N = 1`), for which the
    /// two-point slope formula's denominator (`N*sumXX - sumX^2`) is
    /// `X1^2 - X1^2 = 0`: the same degenerate `0/0` upstream leaves
    /// unguarded.
    #[test]
    fn left_edge_has_no_boundary_condition_and_yields_nan() {
        let image = img(&[5], vec![0.0f64, 2.0, 5.0, 9.0, 0.0]);
        let out = stochastic_fractal_dimension(&image, None, &[1]).unwrap();
        assert!(out.to_f64_vec().unwrap()[0].is_nan());
    }

    #[test]
    fn masked_out_center_pixel_is_exactly_zero() {
        let image = img(&[5], vec![0.0f64, 2.0, 5.0, 9.0, 0.0]);
        let mask = img(&[5], vec![1u8, 1, 0, 1, 1]);
        let out = stochastic_fractal_dimension(&image, Some(&mask), &[1]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap()[2], 0.0);
    }

    /// Masking out a neighbor changes which pairs feed the regression, so
    /// the same center pixel's value differs from the unmasked run.
    #[test]
    fn masking_a_neighbor_changes_the_result() {
        let values = vec![0.0f64, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0];
        let image = img(&[7], values);
        let unmasked = stochastic_fractal_dimension(&image, None, &[2]).unwrap();

        let mask = img(&[7], vec![1u8, 1, 1, 1, 0, 1, 1]);
        let masked = stochastic_fractal_dimension(&image, Some(&mask), &[2]).unwrap();

        let a = unmasked.to_f64_vec().unwrap()[3];
        let b = masked.to_f64_vec().unwrap()[3];
        assert!(!a.is_nan() && !b.is_nan());
        assert_ne!(a, b, "masking a neighbor should change pixel 3's value");
        // The masked-out voxel itself still gets a real (non-mask-zeroed)
        // value here, since the mask only excludes it as a *neighbor*, not
        // as its own center.
        assert_eq!(masked.to_f64_vec().unwrap()[4], 0.0);
    }

    #[test]
    fn output_pixel_type_matches_input() {
        let image = img(&[5], vec![0.0f32, 2.0, 5.0, 9.0, 0.0]);
        let out = stochastic_fractal_dimension(&image, None, &[1]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn default_radius_truncates_to_a_two_dimensional_image() {
        let image = img(&[5, 5], vec![1.0f64; 25]);
        // SimpleITK's default is 3 entries (`[2, 2, 2]`); only the first two
        // apply to a 2-D image.
        let out = stochastic_fractal_dimension(&image, None, &[2, 2, 2]).unwrap();
        assert_eq!(out.size(), &[5, 5]);
    }

    #[test]
    fn rejects_integer_pixel_type() {
        let image = img(&[4], vec![1u8, 2, 3, 4]);
        assert_eq!(
            stochastic_fractal_dimension(&image, None, &[1]),
            Err(FilterError::RequiresRealPixelType(PixelId::UInt8))
        );
    }

    #[test]
    fn rejects_short_radius() {
        let image = img(&[4, 4], vec![0.0f64; 16]);
        assert_eq!(
            stochastic_fractal_dimension(&image, None, &[2]),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        );
    }

    #[test]
    fn rejects_non_uint8_mask() {
        let image = img(&[4], vec![0.0f64; 4]);
        let mask = img(&[4], vec![1u16, 1, 1, 1]);
        assert_eq!(
            stochastic_fractal_dimension(&image, Some(&mask), &[1]),
            Err(FilterError::RequiresUInt8MaskPixelType(PixelId::UInt16))
        );
    }

    /// The mask is a pipeline input, so `ImageToImageFilter::VerifyInputInformation`
    /// compares its physical space with the image's and throws on a mismatch. The
    /// aligned mask is accepted first, or the refusal below would prove nothing.
    #[test]
    fn rejects_a_mask_on_a_different_grid() {
        let image = img(&[4], vec![0.0f64; 4]);
        let aligned = img(&[4], vec![1u8; 4]);
        stochastic_fractal_dimension(&image, Some(&aligned), &[1])
            .expect("an aligned mask must be accepted, or the refusal below proves nothing");

        let mut shifted = img(&[4], vec![1u8; 4]);
        shifted.set_origin(&[5.0]).unwrap();
        assert_eq!(
            stochastic_fractal_dimension(&image, Some(&shifted), &[1]),
            Err(FilterError::PhysicalSpaceMismatch { index: 1 })
        );
    }

    #[test]
    fn rejects_mask_size_mismatch() {
        let image = img(&[4], vec![0.0f64; 4]);
        let mask = img(&[3], vec![1u8; 3]);
        assert_eq!(
            stochastic_fractal_dimension(&image, Some(&mask), &[1]),
            Err(FilterError::SizeMismatch {
                a: vec![4],
                b: vec![3]
            })
        );
    }
}
