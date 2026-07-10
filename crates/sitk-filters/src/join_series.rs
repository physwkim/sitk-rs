//! `JoinSeriesImageFilter`: join `N` same-dimension images into one image
//! with an extra trailing axis, ported from `itkJoinSeriesImageFilter.h(.hxx)`
//! (`Modules/Filtering/ImageCompose`).
//!
//! `GenerateOutputInformation` takes the first `N` (input-dimension) axes'
//! size, spacing, origin and the leading `N x N` direction-cosine block
//! straight from the *first* input; the new axis gets the caller's `origin`/
//! `spacing` and an identity row/column is appended to the direction matrix
//! (`itkJoinSeriesImageFilter.hxx:104-162`). "Note that all the inputs should
//! have the same information" in the class doc is not just a suggestion:
//! `JoinSeriesImageFilter::VerifyInputInformation` only *adds* a
//! number-of-components check on top of `Superclass::VerifyInputInformation`
//! (`itkJoinSeriesImageFilter.hxx:37-74`) -- it does not override or skip the
//! base class's physical-space congruency check
//! (`itkImageToImageFilter.hxx:148-223`), which throws "Inputs do not occupy
//! the same physical space!" whenever an input's origin, spacing or direction
//! differs from the first input's by more than `1e-6` (the coordinate
//! tolerance is `1e-6 * primary.spacing()[0]`, the direction tolerance is a
//! flat `1e-6`). This crate has no vector/multi-component pixel type, so the
//! component check is always vacuously satisfied and isn't ported as code.
//!
//! **Mismatched input sizes are not checked directly.** `GenerateOutputInformation`
//! reads the output's size only from the first input; every other input is
//! read through `GenerateInputRequestedRegion`, which copies the *output*
//! region (sized from the first input, truncated to the input dimension by
//! `CallCopyOutputRegionToInputRegion`) onto *every* indexed input unchanged
//! (`itkJoinSeriesImageFilter.hxx:180-218`). So:
//! - an input **smaller** than the first along any axis gets a requested
//!   region that doesn't fit inside its own `LargestPossibleRegion`, which
//!   ITK's pipeline rejects with `InvalidRequestedRegionError` before
//!   `GenerateData` ever runs;
//! - an input **larger** than the first is *not* rejected: only its
//!   `[0, primary_size)` corner sub-region is copied (`DynamicThreadedGenerateData`'s
//!   `ImageAlgorithm::Copy(GetInput(idx), output, inputRegion, outputRegion)`,
//!   where `inputRegion` is the same first-input-sized region for every
//!   input), and the rest of that larger input is silently ignored.
//!
//! This port reproduces both halves of that asymmetry: [`FilterError::InputSmallerThanPrimary`]
//! for the first case, and a plain corner-crop (no error) for the second.
//!
//! Empty input list: `VerifyInputInformation` reads `this->GetInput()` (the
//! primary/first indexed input) and throws "Input not set as expected!" if
//! it is null, which is the C++ equivalent of an empty input list; this
//! ports as [`FilterError::EmptyImageList`], matching every other
//! multi-input filter in this crate.
//!
//! Single input: none of the cross-input checks above ever run (there is
//! nothing to compare against), so a single image is joined as-is with a
//! new trailing axis of size 1.

use crate::error::{FilterError, Result};
use sitk_core::{Image, Scalar, dispatch_scalar};

/// `ImageToImageFilter`'s default `GlobalDefaultCoordinateTolerance` and
/// `GlobalDefaultDirectionTolerance` (`itkImageToImageFilter.h`), neither of
/// which `JoinSeriesImageFilter` overrides.
const COORDINATE_TOLERANCE: f64 = 1e-6;
const DIRECTION_TOLERANCE: f64 = 1e-6;

/// First-axis-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// `ImageBase::IsCongruentImageGeometry` (`itkImageBase.hxx:391-406`): origin
/// and spacing compared with a tolerance scaled by `primary`'s first-axis
/// spacing, direction compared with a flat tolerance.
fn same_physical_space(primary: &Image, other: &Image) -> bool {
    let coord_tol = (COORDINATE_TOLERANCE * primary.spacing()[0]).abs();
    let origin_ok = primary
        .origin()
        .iter()
        .zip(other.origin())
        .all(|(a, b)| (a - b).abs() <= coord_tol);
    let spacing_ok = primary
        .spacing()
        .iter()
        .zip(other.spacing())
        .all(|(a, b)| (a - b).abs() <= coord_tol);
    let direction_ok = primary
        .direction()
        .iter()
        .zip(other.direction())
        .all(|(a, b)| (a - b).abs() <= DIRECTION_TOLERANCE);
    origin_ok && spacing_ok && direction_ok
}

fn build_image<T: Scalar>(size: &[usize], vals: &[f64]) -> Result<Image> {
    let data: Vec<T> = vals.iter().map(|&v| T::from_f64(v)).collect();
    Ok(Image::from_vec(size, data)?)
}

/// `JoinSeriesImageFilter`: stack `images` (all sharing a pixel type and
/// dimension) along a new trailing axis, whose physical spacing and origin
/// are `spacing`/`origin`. SimpleITK defaults: `origin = 0.0`,
/// `spacing = 1.0`.
///
/// See the module docs for the exact geometry-inheritance rule, the
/// same-physical-space precondition, and the corner-crop/too-small asymmetry
/// for inputs whose size differs from the first.
pub fn join_series(images: &[&Image], origin: f64, spacing: f64) -> Result<Image> {
    let Some((primary, rest)) = images.split_first() else {
        return Err(FilterError::EmptyImageList);
    };
    let dim = primary.dimension();
    let pixel_id = primary.pixel_id();

    for (offset, other) in rest.iter().enumerate() {
        let index = offset + 1;
        if other.pixel_id() != pixel_id {
            return Err(FilterError::TypeMismatch {
                a: pixel_id,
                b: other.pixel_id(),
            });
        }
        if other.dimension() != dim {
            return Err(FilterError::ImageDimensionMismatch {
                a: dim,
                b: other.dimension(),
            });
        }
        if !same_physical_space(primary, other) {
            return Err(FilterError::PhysicalSpaceMismatch { index });
        }
    }

    let primary_size = primary.size().to_vec();
    for (offset, other) in rest.iter().enumerate() {
        let index = offset + 1;
        let other_size = other.size();
        if (0..dim).any(|d| other_size[d] < primary_size[d]) {
            return Err(FilterError::InputSmallerThanPrimary {
                index,
                size: other_size.to_vec(),
                primary_size: primary_size.clone(),
            });
        }
    }

    let out_dim = dim + 1;

    let mut out_size = primary_size.clone();
    out_size.push(images.len());

    let mut out_spacing = primary.spacing().to_vec();
    out_spacing.push(spacing);
    let mut out_origin = primary.origin().to_vec();
    out_origin.push(origin);

    // Embed the primary's `dim x dim` direction block into the output's
    // `out_dim x out_dim` matrix, with an identity row/column appended for
    // the new axis (`itkJoinSeriesImageFilter.hxx:139-161`).
    let in_dir = primary.direction();
    let mut out_direction = vec![0.0f64; out_dim * out_dim];
    for i in 0..out_dim {
        for j in 0..out_dim {
            out_direction[i * out_dim + j] = if i < dim && j < dim {
                in_dir[i * dim + j]
            } else if i == j {
                1.0
            } else {
                0.0
            };
        }
    }

    // Copy each image's `[0, primary_size)` corner sub-region into its slab
    // along the new axis. The new axis is the output's slowest-varying, so
    // slab `k` occupies the flat range `[k * slab_len, (k + 1) * slab_len)`.
    let slab_len: usize = primary_size.iter().product();
    let mut out_vals = vec![0.0f64; slab_len * images.len()];
    for (slab_index, img) in images.iter().enumerate() {
        let img_vals = img.to_f64_vec()?;
        let img_strides = strides(img.size());
        let out_offset = slab_index * slab_len;
        for flat in 0..slab_len {
            let mut src_flat = 0usize;
            let mut rem = flat;
            for d in 0..dim {
                let coord = rem % primary_size[d];
                rem /= primary_size[d];
                src_flat += coord * img_strides[d];
            }
            out_vals[out_offset + flat] = img_vals[src_flat];
        }
    }

    let mut out_image = dispatch_scalar!(pixel_id, build_image, &out_size, &out_vals)?;
    out_image.set_spacing(&out_spacing)?;
    out_image.set_origin(&out_origin)?;
    out_image.set_direction(&out_direction)?;
    Ok(out_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    fn img<T: Scalar>(size: &[usize], data: Vec<T>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn joins_two_2d_images_into_a_3d_stack() {
        let a = img(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = img(&[2, 2], vec![5.0f32, 6.0, 7.0, 8.0]);
        let out = join_series(&[&a, &b], 0.0, 1.0).unwrap();

        assert_eq!(out.dimension(), 3);
        assert_eq!(out.size(), &[2, 2, 2]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(out.pixel_id(), PixelId::Float32);
    }

    #[test]
    fn default_origin_and_spacing_extend_the_new_axis() {
        let a = img(&[2, 1], vec![0.0f64, 0.0]);
        let b = img(&[2, 1], vec![0.0f64, 0.0]);
        let out = join_series(&[&a, &b], 0.0, 1.0).unwrap();
        assert_eq!(out.spacing(), &[1.0, 1.0, 1.0]);
        assert_eq!(out.origin(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn custom_origin_and_spacing_apply_only_to_the_new_axis() {
        let mut a = img(&[2, 1], vec![0.0f64, 0.0]);
        a.set_spacing(&[2.0, 3.0]).unwrap();
        a.set_origin(&[10.0, 20.0]).unwrap();
        let mut b = img(&[2, 1], vec![0.0f64, 0.0]);
        b.set_spacing(&[2.0, 3.0]).unwrap();
        b.set_origin(&[10.0, 20.0]).unwrap();

        let out = join_series(&[&a, &b], 1.234, 0.0123).unwrap();
        assert_eq!(out.spacing(), &[2.0, 3.0, 0.0123]);
        assert_eq!(out.origin(), &[10.0, 20.0, 1.234]);
    }

    /// The output's leading `dim x dim` direction block is the primary's
    /// direction verbatim; the new axis gets an identity row/column.
    #[test]
    fn direction_embeds_primary_block_with_identity_new_axis() {
        let mut a = img(&[2, 2], vec![0.0f64; 4]);
        // 90-degree rotation.
        a.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let mut b = img(&[2, 2], vec![0.0f64; 4]);
        b.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let out = join_series(&[&a, &b], 0.0, 1.0).unwrap();
        assert_eq!(
            out.direction(),
            &[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn single_input_adds_a_size_one_new_axis() {
        let a = img(&[3, 2], vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = join_series(&[&a], 0.0, 1.0).unwrap();
        assert_eq!(out.size(), &[3, 2, 1]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn one_dimensional_inputs_join_into_a_2d_output() {
        let a = img(&[3], vec![1.0f64, 2.0, 3.0]);
        let b = img(&[3], vec![4.0f64, 5.0, 6.0]);
        let out = join_series(&[&a, &b], 0.0, 1.0).unwrap();
        assert_eq!(out.size(), &[3, 2]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn rejects_empty_input_list() {
        assert_eq!(join_series(&[], 0.0, 1.0), Err(FilterError::EmptyImageList));
    }

    #[test]
    fn rejects_pixel_type_mismatch() {
        let a = img(&[2], vec![1.0f32, 2.0]);
        let b = img(&[2], vec![1.0f64, 2.0]);
        assert_eq!(
            join_series(&[&a, &b], 0.0, 1.0),
            Err(FilterError::TypeMismatch {
                a: PixelId::Float32,
                b: PixelId::Float64
            })
        );
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let a = img(&[2, 2], vec![0.0f64; 4]);
        let b = img(&[2], vec![0.0f64; 2]);
        assert_eq!(
            join_series(&[&a, &b], 0.0, 1.0),
            Err(FilterError::ImageDimensionMismatch { a: 2, b: 1 })
        );
    }

    #[test]
    fn rejects_incongruent_spacing() {
        let a = img(&[2, 2], vec![0.0f64; 4]);
        let mut b = img(&[2, 2], vec![0.0f64; 4]);
        b.set_spacing(&[1.0, 2.0]).unwrap();
        assert_eq!(
            join_series(&[&a, &b], 0.0, 1.0),
            Err(FilterError::PhysicalSpaceMismatch { index: 1 })
        );
    }

    #[test]
    fn rejects_incongruent_origin() {
        let a = img(&[2, 2], vec![0.0f64; 4]);
        let mut b = img(&[2, 2], vec![0.0f64; 4]);
        b.set_origin(&[0.0, 5.0]).unwrap();
        assert_eq!(
            join_series(&[&a, &b], 0.0, 1.0),
            Err(FilterError::PhysicalSpaceMismatch { index: 1 })
        );
    }

    #[test]
    fn rejects_incongruent_direction() {
        let a = img(&[2, 2], vec![0.0f64; 4]);
        let mut b = img(&[2, 2], vec![0.0f64; 4]);
        b.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        assert_eq!(
            join_series(&[&a, &b], 0.0, 1.0),
            Err(FilterError::PhysicalSpaceMismatch { index: 1 })
        );
    }

    /// A later input smaller than the primary along any axis cannot supply
    /// the requested corner region -- ITK's `InvalidRequestedRegionError`.
    #[test]
    fn rejects_input_smaller_than_primary() {
        let a = img(&[3, 3], vec![0.0f64; 9]);
        let b = img(&[2, 3], vec![0.0f64; 6]);
        assert_eq!(
            join_series(&[&a, &b], 0.0, 1.0),
            Err(FilterError::InputSmallerThanPrimary {
                index: 1,
                size: vec![2, 3],
                primary_size: vec![3, 3],
            })
        );
    }

    /// A later input *larger* than the primary is not an error: only its
    /// `[0, primary_size)` corner is copied, and the rest is dropped
    /// silently, exactly as `itkJoinSeriesImageFilter.hxx` does.
    #[test]
    fn larger_input_is_corner_cropped_not_rejected() {
        let a = img(&[2, 2], vec![1.0f64, 2.0, 3.0, 4.0]);
        // 3x3, row-major-ish values 10..90 in this crate's axis-0-fastest
        // layout: column-major-looking when printed, but linear_index([x,y])
        // = x + 3*y.
        let b = img(
            &[3, 3],
            vec![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        );
        let out = join_series(&[&a, &b], 0.0, 1.0).unwrap();
        assert_eq!(out.size(), &[2, 2, 2]);
        // Slab 1 is b's top-left 2x2 corner: indices (0,0)=10, (1,0)=20,
        // (0,1)=40, (1,1)=50.
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0, 50.0]
        );
    }
}
