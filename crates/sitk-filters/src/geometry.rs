//! Image-grid / geometry filters: every filter here changes the index grid, so
//! each recomputes origin / spacing / direction exactly as its ITK
//! `GenerateOutputInformation()` does, then applies SimpleITK's universal
//! `FixNonZeroIndex` step (sitkImageFilter.h) that folds a non-zero region
//! start index into an origin shift, since [`Image`] is always zero-indexed.
//! `FixNonZeroIndex` reduces to `Image::continuous_index_to_physical_point`.
//!
//! Ported from:
//! - `itkCropImageFilter.h` / `.hxx` ([`crop`])
//! - `itkRegionOfInterestImageFilter.h` / `.hxx` ([`region_of_interest`])
//! - `itkExtractImageFilter.h` / `.hxx` (Core/Common; [`extract`]), submatrix
//!   direction-collapse strategy only (`DIRECTIONCOLLAPSETOSUBMATRIX`, the
//!   forced default of `CropImageFilter`)
//! - `itkPadImageFilterBase.h` / `.hxx`, `itkPadImageFilter.h` / `.hxx`,
//!   `itkConstantPadImageFilter.h`, `itkMirrorPadImageFilter.h`,
//!   `itkWrapPadImageFilter.h` ([`constant_pad`], [`mirror_pad`], [`wrap_pad`])
//! - `itkFlipImageFilter.h` / `.hxx` ([`flip`])
//! - `itkPermuteAxesImageFilter.h` / `.hxx` ([`permute_axes`])

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use sitk_core::{
    BoundaryCondition, ConstantBoundaryCondition, Image, MirrorBoundaryCondition,
    PeriodicBoundaryCondition, PixelId, Scalar, ScalarView, dispatch_scalar, matrix,
};

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

// ---- crop / region_of_interest --------------------------------------------
//
// Neither collapses dimensions, so both reduce to: copy the `[offset,
// offset+out_size)` block, and shift the origin to
// `img.continuous_index_to_physical_point(offset)` (RegionOfInterestImageFilter
// ::GenerateOutputInformation calls this directly; CropImageFilter delegates to
// ExtractImageFilter, whose non-collapsing path is the same computation, see
// `extract`'s module doc for the collapsing path).

fn extract_block(img: &Image, offset: &[usize], out_size: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    let in_size = img.size();
    let in_strides = strides(in_size);
    let out_strides = strides(out_size);
    let out_count: usize = out_size.iter().product();

    let in_vals = img.to_f64_vec()?;
    let mut out_vals = vec![0.0f64; out_count];
    for (o, slot) in out_vals.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for d in 0..dim {
            let oi = (o / out_strides[d]) % out_size[d];
            in_flat += (oi + offset[d]) * in_strides[d];
        }
        *slot = in_vals[in_flat];
    }

    let offset_f: Vec<f64> = offset.iter().map(|&o| o as f64).collect();
    let out_origin = img.continuous_index_to_physical_point(&offset_f);

    let mut out = image_from_f64(img.pixel_id(), out_size, img, &out_vals)?;
    out.set_origin(&out_origin)?;
    Ok(out)
}

/// `CropImageFilter`: remove `lower[d]` pixels from the start and `upper[d]`
/// from the end of each axis.
///
/// Errors if `lower[d] + upper[d]` exceeds `img.size()[d]` for any axis
/// (`CropImageFilter::VerifyInputInformation`); equal to the size is allowed
/// and yields a zero-length axis.
pub fn crop(img: &Image, lower: &[usize], upper: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(lower.len(), dim)?;
    require_dim(upper.len(), dim)?;

    let size = img.size();
    let mut out_size = vec![0usize; dim];
    for d in 0..dim {
        if lower[d] + upper[d] > size[d] {
            return Err(FilterError::InvalidCropBounds {
                axis: d,
                lower: lower[d],
                upper: upper[d],
                size: size[d],
            });
        }
        out_size[d] = size[d] - lower[d] - upper[d];
    }
    extract_block(img, lower, &out_size)
}

/// `RegionOfInterestImageFilter`: extract the block `[index, index + size)`.
///
/// Errors if `index[d] + size[d]` exceeds `img.size()[d]` for any axis.
pub fn region_of_interest(img: &Image, index: &[usize], size: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(index.len(), dim)?;
    require_dim(size.len(), dim)?;

    let input_size = img.size();
    for d in 0..dim {
        if index[d] + size[d] > input_size[d] {
            return Err(FilterError::RegionOutOfBounds {
                index: index.to_vec(),
                size: size.to_vec(),
                input_size: input_size.to_vec(),
            });
        }
    }
    extract_block(img, index, size)
}

// ---- extract ----------------------------------------------------------

fn build_bare_from_f64<T: Scalar>(size: &[usize], vals: &[f64]) -> Result<Image> {
    let out: Vec<T> = vals.iter().map(|&v| T::from_f64(v)).collect();
    Ok(Image::from_vec(size, out)?)
}

/// Build an image of `target` pixel type directly from `f64` values with no
/// geometry, for [`extract`]'s dimension-changing case (`Image::copy_geometry_from`
/// requires equal dimension, which does not hold when axes collapse).
fn bare_image_from_f64(target: PixelId, size: &[usize], vals: &[f64]) -> Result<Image> {
    dispatch_scalar!(target, build_bare_from_f64, size, vals)
}

/// `ExtractImageFilter` (itkExtractImageFilter.h, Core/Common): extract the
/// block `[index, index + size)`, collapsing any axis where `size[d] == 0`
/// into a fixed slice at `index[d]` (matching ITK's
/// `nonzeroSizeCount`-driven output dimension). Direction-collapse strategy is
/// always Submatrix, `CropImageFilter`'s forced default
/// (`SetDirectionCollapseToSubmatrix`).
///
/// A collapsed axis's own offset does not shift the output origin: ITK's
/// `GenerateOutputInformation` builds the output origin only from the
/// *retained* axes' spacing/direction submatrix and index, so a non-diagonal
/// direction cosine coupling a collapsed axis to a retained one is dropped —
/// this is ITK's actual (if surprising) behavior, reproduced here bit-for-bit.
///
/// Errors if a retained axis's `[index, index + size)` exceeds the input size,
/// if a collapsed axis's `index` is out of bounds, if every axis collapses
/// (`ExtractCollapsedAllAxes`), or if the collapsed direction submatrix is
/// singular (`SingularCollapsedDirection`, ITK's `vnl_determinant == 0` check).
pub fn extract(img: &Image, size: &[usize], index: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(size.len(), dim)?;
    require_dim(index.len(), dim)?;

    let in_size = img.size();
    for d in 0..dim {
        let in_bounds = if size[d] == 0 {
            index[d] < in_size[d]
        } else {
            index[d] + size[d] <= in_size[d]
        };
        if !in_bounds {
            return Err(FilterError::RegionOutOfBounds {
                index: index.to_vec(),
                size: size.to_vec(),
                input_size: in_size.to_vec(),
            });
        }
    }

    let retained: Vec<usize> = (0..dim).filter(|&d| size[d] != 0).collect();
    if retained.is_empty() {
        return Err(FilterError::ExtractCollapsedAllAxes);
    }
    let out_dim = retained.len();

    let in_spacing = img.spacing();
    let in_origin = img.origin();
    let in_direction = img.direction();

    let mut out_size = vec![0usize; out_dim];
    let mut out_spacing = vec![0.0f64; out_dim];
    let mut retained_origin = vec![0.0f64; out_dim];
    let mut out_direction = vec![0.0f64; out_dim * out_dim];
    for (a, &d) in retained.iter().enumerate() {
        out_size[a] = size[d];
        out_spacing[a] = in_spacing[d];
        retained_origin[a] = in_origin[d];
        for (b, &e) in retained.iter().enumerate() {
            out_direction[a * out_dim + b] = in_direction[d * dim + e];
        }
    }

    if out_dim != dim && matrix::invert(&out_direction, out_dim).is_none() {
        return Err(FilterError::SingularCollapsedDirection);
    }

    // FixNonZeroIndex, restricted to the retained axes' own submatrix
    // geometry (see the doc comment above on the collapsed-axis quirk).
    let retained_index: Vec<f64> = retained.iter().map(|&d| index[d] as f64).collect();
    let scaled: Vec<f64> = (0..out_dim)
        .map(|a| retained_index[a] * out_spacing[a])
        .collect();
    let rotated = matrix::mat_vec(&out_direction, &scaled, out_dim);
    let out_origin: Vec<f64> = (0..out_dim)
        .map(|a| retained_origin[a] + rotated[a])
        .collect();

    let in_strides = strides(in_size);
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();
    let in_vals = img.to_f64_vec()?;
    let mut out_vals = vec![0.0f64; out_count];
    for (o, slot) in out_vals.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for d in 0..dim {
            if size[d] == 0 {
                in_flat += index[d] * in_strides[d];
            }
        }
        for (a, &d) in retained.iter().enumerate() {
            let oi = (o / out_strides[a]) % out_size[a];
            in_flat += (oi + index[d]) * in_strides[d];
        }
        *slot = in_vals[in_flat];
    }

    let mut out = bare_image_from_f64(img.pixel_id(), &out_size, &out_vals)?;
    out.set_spacing(&out_spacing)?;
    out.set_origin(&out_origin)?;
    out.set_direction(&out_direction)?;
    Ok(out)
}

// ---- constant_pad / mirror_pad / wrap_pad ----------------------------------
//
// `PadImageFilter::GenerateOutputInformation`: outSize[d] = inSize[d] +
// lower[d] + upper[d], outStartIndex[d] = -lower[d]; spacing/direction are
// unchanged, and FixNonZeroIndex folds the (always non-zero-once-any lower[d]
// > 0) start index into an origin shift, using the *input*'s own geometry
// (spacing/direction unchanged at that point): out_origin =
// img.continuous_index_to_physical_point(-lower).
//
// `PadImageFilterBase::DynamicThreadedGenerateData` fills every output pixel
// through the filter's `ImageBoundaryCondition`, evaluated at the input-space
// index `outputIndex - lower` (an interior pixel is a boundary condition
// evaluated at an in-bounds index, which every impl reads through as-is).

fn pad_geometry(img: &Image, lower: &[usize], upper: &[usize]) -> (Vec<usize>, Vec<f64>) {
    let dim = img.dimension();
    let in_size = img.size();
    let out_size: Vec<usize> = (0..dim).map(|d| in_size[d] + lower[d] + upper[d]).collect();
    let neg_lower: Vec<f64> = lower.iter().map(|&l| -(l as f64)).collect();
    let out_origin = img.continuous_index_to_physical_point(&neg_lower);
    (out_size, out_origin)
}

fn pad_fill<T: Scalar, B: BoundaryCondition<T>>(
    img: &ScalarView<'_, T>,
    lower: &[usize],
    out_size: &[usize],
    boundary: &B,
) -> Vec<T> {
    let dim = out_size.len();
    let out_strides = strides(out_size);
    let out_count: usize = out_size.iter().product();
    let mut out = Vec::with_capacity(out_count);
    for o in 0..out_count {
        let mut ext_index = vec![0i64; dim];
        for d in 0..dim {
            let oi = (o / out_strides[d]) % out_size[d];
            ext_index[d] = oi as i64 - lower[d] as i64;
        }
        out.push(boundary.get_pixel(&ext_index, img));
    }
    out
}

fn constant_pad_typed<T: Scalar>(
    img: &Image,
    lower: &[usize],
    out_size: &[usize],
    constant: f64,
) -> Result<Image> {
    let bc = ConstantBoundaryCondition::new(T::from_f64(constant));
    let vals = pad_fill(&img.scalar_view::<T>()?, lower, out_size, &bc);
    Ok(Image::from_vec(out_size, vals)?)
}

fn mirror_pad_typed<T: Scalar>(img: &Image, lower: &[usize], out_size: &[usize]) -> Result<Image> {
    let vals: Vec<T> = pad_fill(
        &img.scalar_view::<T>()?,
        lower,
        out_size,
        &MirrorBoundaryCondition,
    );
    Ok(Image::from_vec(out_size, vals)?)
}

fn wrap_pad_typed<T: Scalar>(img: &Image, lower: &[usize], out_size: &[usize]) -> Result<Image> {
    let vals: Vec<T> = pad_fill(
        &img.scalar_view::<T>()?,
        lower,
        out_size,
        &PeriodicBoundaryCondition,
    );
    Ok(Image::from_vec(out_size, vals)?)
}

/// `ConstantPadImageFilter`: grow the image by `lower`/`upper` pixels per
/// axis, filling new pixels with `constant`.
pub fn constant_pad(img: &Image, lower: &[usize], upper: &[usize], constant: f64) -> Result<Image> {
    let dim = img.dimension();
    require_dim(lower.len(), dim)?;
    require_dim(upper.len(), dim)?;
    let (out_size, out_origin) = pad_geometry(img, lower, upper);
    let mut out = dispatch_scalar!(
        img.pixel_id(),
        constant_pad_typed,
        img,
        lower,
        &out_size,
        constant
    )?;
    out.set_spacing(img.spacing())?;
    out.set_origin(&out_origin)?;
    out.set_direction(img.direction())?;
    Ok(out)
}

/// `MirrorPadImageFilter`: grow the image by `lower`/`upper` pixels per axis,
/// filling new pixels by mirror-reflecting the input at its edges
/// ([`sitk_core::MirrorBoundaryCondition`]).
pub fn mirror_pad(img: &Image, lower: &[usize], upper: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(lower.len(), dim)?;
    require_dim(upper.len(), dim)?;
    let (out_size, out_origin) = pad_geometry(img, lower, upper);
    let mut out = dispatch_scalar!(img.pixel_id(), mirror_pad_typed, img, lower, &out_size)?;
    out.set_spacing(img.spacing())?;
    out.set_origin(&out_origin)?;
    out.set_direction(img.direction())?;
    Ok(out)
}

/// `WrapPadImageFilter`: grow the image by `lower`/`upper` pixels per axis,
/// filling new pixels by periodically wrapping the input
/// ([`sitk_core::PeriodicBoundaryCondition`]).
pub fn wrap_pad(img: &Image, lower: &[usize], upper: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(lower.len(), dim)?;
    require_dim(upper.len(), dim)?;
    let (out_size, out_origin) = pad_geometry(img, lower, upper);
    let mut out = dispatch_scalar!(img.pixel_id(), wrap_pad_typed, img, lower, &out_size)?;
    out.set_spacing(img.spacing())?;
    out.set_origin(&out_origin)?;
    out.set_direction(img.direction())?;
    Ok(out)
}

// ---- flip -------------------------------------------------------------

/// `FlipImageFilter`: reverse pixel order along each axis where `axes[d]` is
/// `true`.
///
/// `flip_about_origin` selects which of ITK's two origin/direction
/// conventions applies (`itkFlipImageFilter.hxx::GenerateOutputInformation`):
/// - `false`: the output covers the same physical extent read in reverse —
///   direction column `d` is negated for each flipped axis, origin becomes the
///   physical point of the last voxel along that axis.
/// - `true`: direction is unchanged, and the (still-computed) origin is
///   negated component-wise for each flipped axis, mirroring the image through
///   the physical-space origin.
pub fn flip(img: &Image, axes: &[bool], flip_about_origin: bool) -> Result<Image> {
    let dim = img.dimension();
    require_dim(axes.len(), dim)?;

    let size = img.size();
    let direction = img.direction();

    let new_index: Vec<f64> = (0..dim)
        .map(|d| if axes[d] { (size[d] - 1) as f64 } else { 0.0 })
        .collect();
    let mut out_origin = img.continuous_index_to_physical_point(&new_index);
    if flip_about_origin {
        for (d, o) in out_origin.iter_mut().enumerate() {
            if axes[d] {
                *o = -*o;
            }
        }
    }

    let mut out_direction = direction.to_vec();
    if !flip_about_origin {
        for d in 0..dim {
            if axes[d] {
                for row in 0..dim {
                    out_direction[row * dim + d] *= -1.0;
                }
            }
        }
    }

    let in_strides = strides(size);
    let in_vals = img.to_f64_vec()?;
    let mut out_vals = vec![0.0f64; in_vals.len()];
    for (o, slot) in out_vals.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for d in 0..dim {
            let oi = (o / in_strides[d]) % size[d];
            let ii = if axes[d] { size[d] - 1 - oi } else { oi };
            in_flat += ii * in_strides[d];
        }
        *slot = in_vals[in_flat];
    }

    let mut out = image_from_f64(img.pixel_id(), size, img, &out_vals)?;
    out.set_origin(&out_origin)?;
    out.set_direction(&out_direction)?;
    Ok(out)
}

// ---- permute_axes -----------------------------------------------------

/// `PermuteAxesImageFilter`: reorder axes so output axis `j` is input axis
/// `order[j]`. Origin is unchanged (`GenerateOutputInformation` copies
/// `inputOrigin[j]` to `outputOrigin[j]` directly, without permuting it).
///
/// Errors if `order` is not a permutation of `0..img.dimension()`
/// (`PermuteAxesImageFilter::SetOrder`).
pub fn permute_axes(img: &Image, order: &[usize]) -> Result<Image> {
    let dim = img.dimension();
    require_dim(order.len(), dim)?;

    let mut seen = vec![false; dim];
    for &o in order {
        if o >= dim || seen[o] {
            return Err(FilterError::InvalidPermutation(order.to_vec(), dim));
        }
        seen[o] = true;
    }
    let mut inverse_order = vec![0usize; dim];
    for (j, &o) in order.iter().enumerate() {
        inverse_order[o] = j;
    }

    let in_size = img.size();
    let in_spacing = img.spacing();
    let in_direction = img.direction();

    let out_size: Vec<usize> = order.iter().map(|&o| in_size[o]).collect();
    let out_spacing: Vec<f64> = order.iter().map(|&o| in_spacing[o]).collect();
    let mut out_direction = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for (j, &o) in order.iter().enumerate() {
            out_direction[i * dim + j] = in_direction[i * dim + o];
        }
    }

    let in_strides = strides(in_size);
    let out_strides = strides(&out_size);
    let out_count: usize = out_size.iter().product();
    let in_vals = img.to_f64_vec()?;
    let mut out_vals = vec![0.0f64; out_count];
    for (o_flat, slot) in out_vals.iter_mut().enumerate() {
        let mut in_flat = 0usize;
        for j in 0..dim {
            let k = inverse_order[j];
            let idx_k = (o_flat / out_strides[k]) % out_size[k];
            in_flat += idx_k * in_strides[j];
        }
        *slot = in_vals[in_flat];
    }

    let mut out = image_from_f64(img.pixel_id(), &out_size, img, &out_vals)?;
    out.set_spacing(&out_spacing)?;
    out.set_direction(&out_direction)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    /// An oblique (non-identity) 2-D rotation direction, used by every test
    /// that must prove a formula actually consults the direction matrix
    /// rather than assuming identity.
    fn rotated_2d(size: &[usize], spacing: &[f64], origin: &[f64], data: Vec<f64>) -> Image {
        let mut img = Image::from_vec(size, data).unwrap();
        img.set_spacing(spacing).unwrap();
        img.set_origin(origin).unwrap();
        let theta = std::f64::consts::FRAC_PI_6;
        img.set_direction(&[theta.cos(), -theta.sin(), theta.sin(), theta.cos()])
            .unwrap();
        img
    }

    // ---- crop ----

    #[test]
    fn crop_removes_pixels_and_shifts_origin_through_direction() {
        let img = rotated_2d(
            &[4, 4],
            &[2.0, 3.0],
            &[10.0, -5.0],
            (0..16).map(|v| v as f64).collect(),
        );
        let out = crop(&img, &[1, 1], &[1, 1]).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.direction(), img.direction());
        let expected_origin = img.continuous_index_to_physical_point(&[1.0, 1.0]);
        for (a, b) in out.origin().iter().zip(expected_origin.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // Row-major, first-index-fastest: interior 2x2 block of a 4x4 grid
        // starting at (1,1) is values [5,6,9,10].
        assert_eq!(out.to_f64_vec().unwrap(), vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn crop_zero_width_axis_is_allowed() {
        // lower + upper == size on axis 0: ITK's VerifyInputInformation only
        // throws on strictly-insufficient size, not equality.
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = crop(&img, &[4, 0], &[0, 0]).unwrap();
        assert_eq!(out.size(), &[0, 3]);
        assert_eq!(out.number_of_pixels(), 0);
    }

    #[test]
    fn crop_bounds_exceeding_size_errors() {
        let img = Image::new(&[4, 4], PixelId::UInt8);
        assert_eq!(
            crop(&img, &[3, 0], &[2, 0]),
            Err(FilterError::InvalidCropBounds {
                axis: 0,
                lower: 3,
                upper: 2,
                size: 4
            })
        );
    }

    #[test]
    fn crop_physical_point_of_surviving_voxel_is_invariant() {
        // A voxel that survives the crop must sit at the same physical point
        // before and after, even under anisotropic spacing and an oblique
        // direction.
        let img = rotated_2d(
            &[6, 5],
            &[1.5, 0.75],
            &[3.0, -2.0],
            (0..30).map(|v| v as f64).collect(),
        );
        let out = crop(&img, &[2, 1], &[1, 1]).unwrap();
        let survivor_in_input = [3.0, 2.0]; // index (2,1) + (1,1) inside the retained block
        let survivor_in_output = [1.0, 1.0];
        let p_in = img.continuous_index_to_physical_point(&survivor_in_input);
        let p_out = out.continuous_index_to_physical_point(&survivor_in_output);
        for (a, b) in p_in.iter().zip(p_out.iter()) {
            assert!((a - b).abs() < 1e-10, "{p_in:?} vs {p_out:?}");
        }
    }

    // ---- region_of_interest ----

    #[test]
    fn region_of_interest_extracts_block() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = region_of_interest(&img, &[1, 0], &[2, 2]).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        // Block starting at (1,0), 2x2: values [1,2,5,6].
        assert_eq!(out.to_f64_vec().unwrap(), vec![1.0, 2.0, 5.0, 6.0]);
        let expected_origin = img.continuous_index_to_physical_point(&[1.0, 0.0]);
        assert_eq!(out.origin(), expected_origin.as_slice());
    }

    #[test]
    fn region_of_interest_zero_size_axis_is_allowed() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|v| v as f64).collect()).unwrap();
        let out = region_of_interest(&img, &[2, 1], &[0, 2]).unwrap();
        assert_eq!(out.size(), &[0, 2]);
    }

    #[test]
    fn region_of_interest_out_of_bounds_errors() {
        let img = Image::new(&[4, 3], PixelId::UInt8);
        assert!(matches!(
            region_of_interest(&img, &[3, 0], &[2, 1]),
            Err(FilterError::RegionOutOfBounds { .. })
        ));
    }

    // ---- extract ----

    #[test]
    fn extract_same_dimension_matches_region_of_interest() {
        let img = rotated_2d(
            &[5, 4],
            &[1.0, 2.0],
            &[0.0, 0.0],
            (0..20).map(|v| v as f64).collect(),
        );
        let a = extract(&img, &[2, 2], &[1, 1]).unwrap();
        let b = region_of_interest(&img, &[1, 1], &[2, 2]).unwrap();
        assert_eq!(a.size(), b.size());
        assert_eq!(a.origin(), b.origin());
        assert_eq!(a.direction(), b.direction());
        assert_eq!(a.to_f64_vec().unwrap(), b.to_f64_vec().unwrap());
    }

    #[test]
    fn extract_collapses_zero_size_axis_and_drops_dimension() {
        // 3x3x3 volume, value(x,y,z) = x + 10y + 100z; extract the z=1 slice.
        let mut data = vec![0.0f64; 27];
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    data[z * 9 + y * 3 + x] = (x + 10 * y + 100 * z) as f64;
                }
            }
        }
        let mut img = Image::from_vec(&[3, 3, 3], data).unwrap();
        img.set_spacing(&[1.0, 1.0, 2.0]).unwrap();
        img.set_origin(&[5.0, 5.0, 5.0]).unwrap();
        let out = extract(&img, &[3, 3, 0], &[0, 0, 1]).unwrap();
        assert_eq!(out.dimension(), 2);
        assert_eq!(out.size(), &[3, 3]);
        // Identity direction with no cross terms: x,y origin unaffected by
        // the collapsed z offset (ITK drops that contribution entirely).
        assert_eq!(out.origin(), &[5.0, 5.0]);
        assert_eq!(out.spacing(), &[1.0, 1.0]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![
                100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 120.0, 121.0, 122.0
            ]
        );
    }

    #[test]
    fn extract_collapsing_all_axes_errors() {
        let img = Image::new(&[4, 4], PixelId::UInt8);
        assert_eq!(
            extract(&img, &[0, 0], &[1, 1]),
            Err(FilterError::ExtractCollapsedAllAxes)
        );
    }

    #[test]
    fn extract_collapsed_axis_out_of_bounds_errors() {
        let img = Image::new(&[3, 3, 3], PixelId::UInt8);
        assert!(matches!(
            extract(&img, &[3, 3, 0], &[0, 0, 5]),
            Err(FilterError::RegionOutOfBounds { .. })
        ));
    }

    #[test]
    fn extract_singular_collapsed_direction_errors() {
        // 3-D direction whose 2x2 submatrix over the retained (x,y) axes is
        // singular (both rows equal), even though the full 3x3 is not
        // degenerate in a way `Image::set_direction` rejects.
        let mut img = Image::new(&[2, 2, 2], PixelId::Float64);
        img.set_direction(&[
            1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ])
        .unwrap();
        assert_eq!(
            extract(&img, &[2, 2, 0], &[0, 0, 0]),
            Err(FilterError::SingularCollapsedDirection)
        );
    }

    // ---- constant_pad / mirror_pad / wrap_pad ----

    #[test]
    fn constant_pad_grows_and_shifts_origin_through_direction() {
        let img = rotated_2d(&[2, 2], &[2.0, 3.0], &[1.0, 1.0], vec![1.0, 2.0, 3.0, 4.0]);
        let out = constant_pad(&img, &[1, 0], &[0, 1], 9.0).unwrap();
        assert_eq!(out.size(), &[3, 3]);
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.direction(), img.direction());
        let expected_origin = img.continuous_index_to_physical_point(&[-1.0, 0.0]);
        for (a, b) in out.origin().iter().zip(expected_origin.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // Original 2x2 sits at output offset (1,0); new pixels are 9.0.
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![9.0, 1.0, 2.0, 9.0, 3.0, 4.0, 9.0, 9.0, 9.0]
        );
    }

    #[test]
    fn pad_of_size_zero_is_identity() {
        let img = Image::from_vec(&[3, 2], (0..6).map(|v| v as f64).collect()).unwrap();
        let out = constant_pad(&img, &[0, 0], &[0, 0], 42.0).unwrap();
        assert_eq!(out.size(), img.size());
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn mirror_pad_reflects_edges() {
        let img = Image::from_vec(&[4, 1], vec![10.0f64, 11.0, 12.0, 13.0]).unwrap();
        let out = mirror_pad(&img, &[2, 0], &[2, 0]).unwrap();
        assert_eq!(out.size(), &[8, 1]);
        // index -1,-2 mirror to 0,1; index 4,5 mirror to 3,2.
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![11.0, 10.0, 10.0, 11.0, 12.0, 13.0, 13.0, 12.0]
        );
    }

    #[test]
    fn wrap_pad_wraps_edges() {
        let img = Image::from_vec(&[4, 1], vec![10.0f64, 11.0, 12.0, 13.0]).unwrap();
        let out = wrap_pad(&img, &[2, 0], &[2, 0]).unwrap();
        assert_eq!(out.size(), &[8, 1]);
        // index -1,-2 wrap to 3,2; index 4,5 wrap to 0,1.
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![12.0, 13.0, 10.0, 11.0, 12.0, 13.0, 10.0, 11.0]
        );
    }

    // ---- flip ----

    #[test]
    fn flip_not_about_origin_negates_direction_column_and_shifts_origin() {
        let img = rotated_2d(
            &[3, 2],
            &[1.0, 2.0],
            &[5.0, -1.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        );
        let out = flip(&img, &[true, false], false).unwrap();
        assert_eq!(out.size(), img.size());
        let expected_origin = img.continuous_index_to_physical_point(&[2.0, 0.0]);
        for (a, b) in out.origin().iter().zip(expected_origin.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // direction column 0 negated, column 1 unchanged.
        let d = img.direction();
        assert_eq!(out.direction(), &[-d[0], d[1], -d[2], d[3]]);
        // x-axis reversed per row.
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]
        );
    }

    #[test]
    fn flip_about_origin_keeps_direction_and_negates_origin_component() {
        let img = rotated_2d(
            &[3, 2],
            &[1.0, 2.0],
            &[5.0, -1.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        );
        let out = flip(&img, &[true, false], true).unwrap();
        assert_eq!(out.direction(), img.direction());
        let mut expected_origin = img.continuous_index_to_physical_point(&[2.0, 0.0]);
        expected_origin[0] = -expected_origin[0];
        for (a, b) in out.origin().iter().zip(expected_origin.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // Pixel order still reverses along the flipped axis regardless of
        // the origin convention.
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]
        );
    }

    // ---- permute_axes ----

    #[test]
    fn permute_axes_swap_2d_reorders_spacing_direction_and_pixels_but_not_origin() {
        let img = rotated_2d(
            &[3, 2],
            &[1.0, 2.0],
            &[7.0, -3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        );
        let out = permute_axes(&img, &[1, 0]).unwrap();
        assert_eq!(out.size(), &[2, 3]);
        assert_eq!(out.spacing(), &[2.0, 1.0]);
        assert_eq!(out.origin(), img.origin());
        let d = img.direction();
        assert_eq!(out.direction(), &[d[1], d[0], d[3], d[2]]);
        // input(x,y) = x + 3y (row-major); output(y,x) = input(x,y).
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn permute_axes_applied_twice_with_self_inverse_order_is_identity() {
        let img = rotated_2d(
            &[4, 3],
            &[1.5, 0.5],
            &[2.0, 2.0],
            (0..12).map(|v| v as f64).collect(),
        );
        let once = permute_axes(&img, &[1, 0]).unwrap();
        let twice = permute_axes(&once, &[1, 0]).unwrap();
        assert_eq!(twice.size(), img.size());
        assert_eq!(twice.spacing(), img.spacing());
        assert_eq!(twice.origin(), img.origin());
        assert_eq!(twice.direction(), img.direction());
        assert_eq!(twice.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
    }

    #[test]
    fn permute_axes_rejects_non_permutation() {
        let img = Image::new(&[2, 2], PixelId::UInt8);
        assert_eq!(
            permute_axes(&img, &[0, 0]),
            Err(FilterError::InvalidPermutation(vec![0, 0], 2))
        );
        assert_eq!(
            permute_axes(&img, &[0, 2]),
            Err(FilterError::InvalidPermutation(vec![0, 2], 2))
        );
    }
}
