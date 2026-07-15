//! Grid-arrangement filters that combine or place whole images by index
//! math rather than by pixel value: `itkCheckerBoardImageFilter.h(.hxx)`
//! (`Filtering/ImageCompare`), `itkPasteImageFilter.h(.hxx)` and
//! `itkTileImageFilter.h(.hxx)` (both `Filtering/ImageGrid`). All three are
//! scoped here to same-dimension, same-pixel-type inputs (SimpleITK's own
//! `Paste`/`Tile` procedural wrappers likewise operate within one image
//! dimension; `Tile`'s dimension-raising 1-D-stack-of-(N-1)-D-images case is
//! out of scope).

use crate::core::{Image, Scalar, dispatch_scalar};
use crate::filters::error::{FilterError, Result};
use crate::filters::geometry::require_same_physical_space;
use crate::filters::require_same_shape;

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
        Err(FilterError::DimensionLength {
            expected: dim,
            got: len,
        })
    } else {
        Ok(())
    }
}

// ---- checker_board ----------------------------------------------------------

/// `CheckerBoardImageFilter`: alternate between `image1` and `image2` in a
/// checkerboard pattern (`itkCheckerBoardImageFilter.hxx`'s
/// `DynamicThreadedGenerateData`): with `factors[d] = size[d] /
/// checker_pattern[d]` (integer division), a pixel at index `idx` takes
/// `image2` when `sum_d (idx[d] / factors[d])` is odd, else `image1`.
/// `image1` and `image2` must have the same size and pixel type.
pub fn checker_board(image1: &Image, image2: &Image, checker_pattern: &[u32]) -> Result<Image> {
    require_same_shape(image1, image2)?;
    require_same_physical_space(image1, image2, 1)?;
    let dim = image1.dimension();
    require_dim(checker_pattern.len(), dim)?;

    let size = image1.size();
    let mut factors = vec![0usize; dim];
    for d in 0..dim {
        let cp = checker_pattern[d] as usize;
        if cp == 0 || cp > size[d] {
            return Err(FilterError::InvalidCheckerPattern {
                pattern: checker_pattern.to_vec(),
                size: size.to_vec(),
            });
        }
        factors[d] = size[d] / cp;
    }

    let strides = strides(size);
    dispatch_scalar!(
        image1.pixel_id(),
        checker_native,
        image1,
        image2,
        size,
        &strides,
        &factors
    )
}

/// Select each output pixel from `image1` or `image2`'s **native** buffer —
/// no `to_f64_vec` widening — so `UInt64`/`Int64` values above 2^53 keep their
/// exact bits. `scalar_slice` rejects a non-scalar input, preserving the
/// scalar-only contract the old seam enforced.
fn checker_native<T: Scalar>(
    image1: &Image,
    image2: &Image,
    size: &[usize],
    strides: &[usize],
    factors: &[usize],
) -> Result<Image> {
    let dim = size.len();
    let vals1 = image1.scalar_slice::<T>()?;
    let vals2 = image2.scalar_slice::<T>()?;
    let out: Vec<T> = (0..image1.number_of_pixels())
        .map(|flat| {
            let mut sum = 0usize;
            for d in 0..dim {
                let idx_d = (flat / strides[d]) % size[d];
                sum += idx_d / factors[d];
            }
            if sum % 2 == 1 {
                vals2[flat]
            } else {
                vals1[flat]
            }
        })
        .collect();
    let mut out_image = Image::from_vec(size, out)?;
    out_image.copy_geometry_from(image1);
    Ok(out_image)
}

// ---- paste --------------------------------------------------------------

/// `PasteImageFilter`: paste the `[source_index, source_index + source_size)`
/// block of `source` into `destination` at `destination_index`, producing an
/// image with `destination`'s size and geometry.
///
/// The destination-side region is silently cropped against `destination`'s
/// bounds if it would overrun them — `itkPasteImageFilter.hxx`'s
/// `DynamicThreadedGenerateData` crops the paste region against the output's
/// `LargestPossibleRegion` via `ImageRegion::Crop` rather than erroring (the
/// class doc: "If the output requested region does not include the
/// SourceRegion ... the output will just be a copy of the input."); a region
/// with zero overlap leaves `destination` unchanged. The source-side region,
/// by contrast, must fit inside `source` — a real out-of-bounds
/// `SourceRegion` fails when ITK's pipeline requests it from the source
/// image — so this ports as an error, not a crop.
pub fn paste(
    destination: &Image,
    source: &Image,
    source_index: &[usize],
    source_size: &[usize],
    destination_index: &[usize],
) -> Result<Image> {
    if destination.pixel_id() != source.pixel_id() {
        return Err(FilterError::TypeMismatch {
            a: destination.pixel_id(),
            b: source.pixel_id(),
        });
    }
    let dim = destination.dimension();
    require_dim(source.dimension(), dim)?;
    require_dim(source_index.len(), dim)?;
    require_dim(source_size.len(), dim)?;
    require_dim(destination_index.len(), dim)?;

    let source_extent = source.size();
    for d in 0..dim {
        if source_index[d] + source_size[d] > source_extent[d] {
            return Err(FilterError::RegionOutOfBounds {
                index: source_index.to_vec(),
                size: source_size.to_vec(),
                input_size: source_extent.to_vec(),
            });
        }
    }

    let dest_size = destination.size();
    let mut clipped_size = vec![0usize; dim];
    for d in 0..dim {
        clipped_size[d] = if destination_index[d] >= dest_size[d] {
            0
        } else {
            source_size[d].min(dest_size[d] - destination_index[d])
        };
    }

    dispatch_scalar!(
        destination.pixel_id(),
        paste_native,
        destination,
        source,
        &clipped_size,
        source_index,
        destination_index
    )
}

/// Overlay `source`'s clipped block onto a **native** copy of `destination`'s
/// buffer — no `to_f64_vec` widening — so `UInt64`/`Int64` values above 2^53
/// keep their exact bits on both the pass-through and the overlaid pixels.
/// `scalar_slice` rejects a non-scalar input, preserving the scalar-only
/// contract the old seam enforced.
fn paste_native<T: Scalar>(
    destination: &Image,
    source: &Image,
    clipped_size: &[usize],
    source_index: &[usize],
    destination_index: &[usize],
) -> Result<Image> {
    let dim = destination.dimension();
    let dest_size = destination.size();
    let mut out: Vec<T> = destination.scalar_slice::<T>()?.to_vec();
    if clipped_size.iter().all(|&s| s > 0) {
        let dest_strides = strides(dest_size);
        let src_strides = strides(source.size());
        let clipped_strides = strides(clipped_size);
        let src = source.scalar_slice::<T>()?;
        let count: usize = clipped_size.iter().product();
        for o in 0..count {
            let mut dest_flat = 0usize;
            let mut src_flat = 0usize;
            for d in 0..dim {
                let oi = (o / clipped_strides[d]) % clipped_size[d];
                dest_flat += (oi + destination_index[d]) * dest_strides[d];
                src_flat += (oi + source_index[d]) * src_strides[d];
            }
            out[dest_flat] = src[src_flat];
        }
    }
    let mut out_image = Image::from_vec(dest_size, out)?;
    out_image.copy_geometry_from(destination);
    Ok(out_image)
}

// ---- tile --------------------------------------------------------------

/// `TileImageFilter`: lay `images` out on a `layout`-shaped grid, one image
/// per cell in row-major (first-axis-fastest) order, ported from
/// `itkTileImageFilter.hxx`'s `GenerateOutputInformation` (offset/size
/// resolution) and `DynamicThreadedGenerateData` (per-cell copy). All images
/// must share `images[0]`'s dimension and pixel type; output geometry
/// (spacing, origin, direction) is copied from `images[0]`.
///
/// A `0` in `layout`'s *last* axis is resolved automatically, matching
/// upstream's `((images.len() - 1) / product_of_other_axes) + 1` (the only
/// axis upstream allows to default): `layout = [2, 0]` for 5 images lays
/// them out 2-wide, 3 rows tall. Every other `layout` entry must already be
/// a positive tile count set by the caller.
///
/// Grid cells beyond `images.len()` (when `layout`'s product exceeds the
/// input count) and any gap left by a smaller image in a cell whose row/
/// column is sized to a larger sibling are filled with
/// `default_pixel_value`.
pub fn tile(images: &[&Image], layout: &[usize], default_pixel_value: f64) -> Result<Image> {
    if images.is_empty() {
        return Err(FilterError::EmptyImageList);
    }
    let dim = images[0].dimension();
    require_dim(layout.len(), dim)?;
    let pixel_id = images[0].pixel_id();
    for img in &images[1..] {
        if img.pixel_id() != pixel_id {
            return Err(FilterError::TypeMismatch {
                a: pixel_id,
                b: img.pixel_id(),
            });
        }
        require_dim(img.dimension(), dim)?;
    }

    let mut layout = layout.to_vec();
    let last = dim - 1;
    if layout[last] == 0 {
        let used: usize = layout[..last].iter().product();
        layout[last] = (images.len() - 1).checked_div(used).map_or(1, |q| q + 1);
    }

    let total_tiles: usize = layout.iter().product();
    let tile_strides = strides(&layout);

    // `sizes[d][k]`: the max size along axis `d` among tiles at grid
    // position `k` along `d` (seeded at 1, matching upstream's
    // `tileImageSize.Fill(1)` for a wholly-empty row/column).
    let mut sizes: Vec<Vec<usize>> = layout.iter().map(|&l| vec![1usize; l]).collect();
    for (t, img) in images.iter().enumerate().take(total_tiles) {
        let img_size = img.size();
        for (d, sizes_d) in sizes.iter_mut().enumerate() {
            let coord_d = (t / tile_strides[d]) % layout[d];
            if img_size[d] > sizes_d[coord_d] {
                sizes_d[coord_d] = img_size[d];
            }
        }
    }

    let mut offsets: Vec<Vec<usize>> = sizes.iter().map(|s| vec![0usize; s.len()]).collect();
    let mut output_size = vec![0usize; dim];
    for d in 0..dim {
        for k in 1..layout[d] {
            offsets[d][k] = offsets[d][k - 1] + sizes[d][k - 1];
        }
        output_size[d] = offsets[d][layout[d] - 1] + sizes[d][layout[d] - 1];
    }

    dispatch_scalar!(
        pixel_id,
        tile_native,
        images,
        &layout,
        &tile_strides,
        &offsets,
        &output_size,
        total_tiles,
        default_pixel_value
    )
}

/// Place each tile's **native** buffer into the output at its grid offset,
/// gaps filled with the quantized `default_pixel_value` — no `to_f64_vec`
/// widening, so `UInt64`/`Int64` values above 2^53 keep their exact bits.
/// `scalar_slice` rejects a non-scalar input, preserving the scalar-only
/// contract the old seam enforced.
#[allow(clippy::too_many_arguments)]
fn tile_native<T: Scalar>(
    images: &[&Image],
    layout: &[usize],
    tile_strides: &[usize],
    offsets: &[Vec<usize>],
    output_size: &[usize],
    total_tiles: usize,
    default_pixel_value: f64,
) -> Result<Image> {
    let dim = output_size.len();
    let out_count: usize = output_size.iter().product();
    let mut out: Vec<T> = vec![T::from_f64(default_pixel_value); out_count];
    let out_strides = strides(output_size);

    for (t, img) in images.iter().enumerate().take(total_tiles) {
        let img_size = img.size();
        let img_strides = strides(img_size);
        let src = img.scalar_slice::<T>()?;
        let mut cell_index = vec![0usize; dim];
        for (d, cell_index_d) in cell_index.iter_mut().enumerate() {
            let coord_d = (t / tile_strides[d]) % layout[d];
            *cell_index_d = offsets[d][coord_d];
        }
        for (o, &val) in src.iter().enumerate() {
            let mut out_flat = 0usize;
            for d in 0..dim {
                let oi = (o / img_strides[d]) % img_size[d];
                out_flat += (oi + cell_index[d]) * out_strides[d];
            }
            out[out_flat] = val;
        }
    }

    let mut out_image = Image::from_vec(output_size, out)?;
    out_image.copy_geometry_from(images[0]);
    Ok(out_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    fn img(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- checker_board ----------------------------------------------------

    #[test]
    fn checker_board_even_division_pattern_1_1() {
        // 4x4, pattern (1,1): factors = (4,4); sum = idx0/4 + idx1/4, which
        // is 0 everywhere in a 4x4 grid (every index < 4), so every pixel
        // stays image1 (even sum).
        let a = img(&[4, 4], vec![1.0; 16]);
        let b = img(&[4, 4], vec![2.0; 16]);
        let out = checker_board(&a, &b, &[1, 1]).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![1.0; 16]);
    }

    #[test]
    fn checker_board_pattern_2_2_alternates_quadrants() {
        // 4x4, pattern (2,2): factors = (2,2). Quadrant (col/2, row/2) sums:
        // (0,0)->0 image1, (1,0)->1 image2, (0,1)->1 image2, (1,1)->2 image1.
        let a = img(&[4, 4], vec![1.0; 16]);
        let b = img(&[4, 4], vec![2.0; 16]);
        let out = checker_board(&a, &b, &[2, 2])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let strides = strides(&[4, 4]);
        for row in 0..4 {
            for col in 0..4 {
                let flat = col * strides[0] + row * strides[1];
                let expected = if (col / 2 + row / 2) % 2 == 1 {
                    2.0
                } else {
                    1.0
                };
                assert_eq!(out[flat], expected, "at ({col},{row})");
            }
        }
    }

    #[test]
    fn checker_board_odd_division_boundary() {
        // size 5, pattern 2: factors[0] = 5/2 = 2 (integer division). Cell
        // boundaries at idx/2: 0,1->0  2,3->1  4->2. Sums: 0,1 even(image1);
        // 2,3 odd(image2); 4 even(image1).
        let a = img(&[5, 1], vec![1.0; 5]);
        let b = img(&[5, 1], vec![2.0; 5]);
        let out = checker_board(&a, &b, &[2, 1])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn checker_board_pattern_exceeding_size_errors() {
        let a = img(&[3, 3], vec![1.0; 9]);
        let b = img(&[3, 3], vec![2.0; 9]);
        assert!(matches!(
            checker_board(&a, &b, &[4, 1]),
            Err(FilterError::InvalidCheckerPattern { .. })
        ));
    }

    #[test]
    fn checker_board_size_mismatch_errors() {
        let a = img(&[3, 3], vec![1.0; 9]);
        let b = img(&[2, 2], vec![2.0; 4]);
        assert!(matches!(
            checker_board(&a, &b, &[1, 1]),
            Err(FilterError::SizeMismatch { .. })
        ));
    }

    // ---- paste ----------------------------------------------------------

    #[test]
    fn paste_fully_inside() {
        let dest = img(&[4, 4], vec![0.0; 16]);
        let src = img(&[2, 2], vec![9.0; 4]);
        let out = paste(&dest, &src, &[0, 0], &[2, 2], &[1, 1])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let strides = strides(&[4, 4]);
        for row in 0..4 {
            for col in 0..4 {
                let flat = col * strides[0] + row * strides[1];
                let inside = (1..=2).contains(&col) && (1..=2).contains(&row);
                assert_eq!(out[flat], if inside { 9.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn paste_touching_edge_exactly() {
        // Destination region exactly spans the last two columns/rows: no crop.
        let dest = img(&[4, 4], vec![0.0; 16]);
        let src = img(&[2, 2], vec![9.0; 4]);
        let out = paste(&dest, &src, &[0, 0], &[2, 2], &[2, 2])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out.iter().filter(|&&v| v == 9.0).count(), 4);
    }

    #[test]
    fn paste_overrunning_destination_crops_silently() {
        // destination_index + source_size overruns the 4x4 destination by
        // one row/col; the overrun is dropped, not an error.
        let dest = img(&[4, 4], vec![0.0; 16]);
        let src = img(&[2, 2], vec![9.0; 4]);
        let out = paste(&dest, &src, &[0, 0], &[2, 2], &[3, 3])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        // Only index (3,3) receives the paste; (4,3)/(3,4)/(4,4) are outside.
        let strides = strides(&[4, 4]);
        assert_eq!(out[3 * strides[0] + 3 * strides[1]], 9.0);
        assert_eq!(out.iter().filter(|&&v| v == 9.0).count(), 1);
    }

    #[test]
    fn paste_zero_overlap_is_unmodified_copy() {
        let dest = img(&[4, 4], vec![0.0; 16]);
        let src = img(&[2, 2], vec![9.0; 4]);
        let out = paste(&dest, &src, &[0, 0], &[2, 2], &[4, 4])
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(out, dest.to_f64_vec().unwrap());
    }

    #[test]
    fn paste_source_region_out_of_bounds_errors() {
        let dest = img(&[4, 4], vec![0.0; 16]);
        let src = img(&[2, 2], vec![9.0; 4]);
        assert!(matches!(
            paste(&dest, &src, &[1, 1], &[2, 2], &[0, 0]),
            Err(FilterError::RegionOutOfBounds { .. })
        ));
    }

    #[test]
    fn paste_dimension_mismatch_errors() {
        let dest = img(&[4, 4], vec![0.0; 16]);
        let src = img(&[2, 2], vec![9.0; 4]);
        assert!(matches!(
            paste(&dest, &src, &[0], &[2], &[0]),
            Err(FilterError::DimensionLength { .. })
        ));
    }

    #[test]
    fn paste_pixel_type_mismatch_errors() {
        let dest = Image::from_vec(&[4, 4], vec![0u8; 16]).unwrap();
        let src = Image::from_vec(&[2, 2], vec![9.0f32; 4]).unwrap();
        assert!(matches!(
            paste(&dest, &src, &[0, 0], &[2, 2], &[0, 0]),
            Err(FilterError::TypeMismatch { .. })
        ));
    }

    // ---- tile ----------------------------------------------------------

    #[test]
    fn tile_1xn_lays_out_in_a_row() {
        let a = img(&[2, 1], vec![1.0, 1.0]);
        let b = img(&[2, 1], vec![2.0, 2.0]);
        let c = img(&[2, 1], vec![3.0, 3.0]);
        let out = tile(&[&a, &b, &c], &[3, 1], -1.0).unwrap();
        assert_eq!(out.size(), &[6, 1]);
        assert_eq!(
            out.to_f64_vec().unwrap(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
    }

    #[test]
    fn tile_2x2_with_differing_sizes_fills_gaps() {
        // Row 0: a (2x1), b (1x1) -> row height 1. Row 1: c (1x2), d (2x2)
        // (each 2 wide is irrelevant here since column widths take the max
        // per column: col0 = max(2,1)=2, col1 = max(1,2)=2).
        let a = img(&[2, 1], vec![1.0, 1.0]);
        let b = img(&[1, 1], vec![2.0]);
        let c = img(&[1, 2], vec![3.0, 3.0]);
        let d = img(&[2, 2], vec![4.0, 4.0, 4.0, 4.0]);
        let out = tile(&[&a, &b, &c, &d], &[2, 2], 0.0).unwrap();
        // col0 width = max(a.w=2, c.w=1) = 2; col1 width = max(b.w=1, d.w=2) = 2.
        // row0 height = max(a.h=1, b.h=1) = 1; row1 height = max(c.h=2, d.h=2) = 2.
        assert_eq!(out.size(), &[4, 3]);
        let s = strides(&[4, 3]);
        let at = |x: usize, y: usize, v: &[f64]| v[x * s[0] + y * s[1]];
        let v = out.to_f64_vec().unwrap();
        // Row 0 (y=0): a fills x=0..2, b fills x=2 only (its cell is 2 wide,
        // so x=3 is a gap filled with default 0.0).
        assert_eq!(at(0, 0, &v), 1.0);
        assert_eq!(at(1, 0, &v), 1.0);
        assert_eq!(at(2, 0, &v), 2.0);
        assert_eq!(at(3, 0, &v), 0.0);
        // Row 1..3 (y=1,2): c fills x=0 only (1 wide), d fills x=2..4.
        for y in 1..3 {
            assert_eq!(at(0, y, &v), 3.0);
            assert_eq!(at(1, y, &v), 0.0);
            assert_eq!(at(2, y, &v), 4.0);
            assert_eq!(at(3, y, &v), 4.0);
        }
    }

    #[test]
    fn tile_zero_in_last_axis_is_resolved_automatically() {
        let a = img(&[1, 1], vec![1.0]);
        let b = img(&[1, 1], vec![2.0]);
        let c = img(&[1, 1], vec![3.0]);
        // layout [2, 0] with 3 images: last axis = (3-1)/2 + 1 = 2 rows.
        let out = tile(&[&a, &b, &c], &[2, 0], 9.0).unwrap();
        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.to_f64_vec().unwrap(), vec![1.0, 2.0, 3.0, 9.0]);
    }

    #[test]
    fn tile_empty_image_list_errors() {
        assert!(matches!(
            tile(&[], &[1, 1], 0.0),
            Err(FilterError::EmptyImageList)
        ));
    }

    #[test]
    fn tile_pixel_type_mismatch_errors() {
        let a = Image::from_vec(&[1, 1], vec![1u8]).unwrap();
        let b = Image::from_vec(&[1, 1], vec![2.0f32]).unwrap();
        assert!(matches!(
            tile(&[&a, &b], &[2, 1], 0.0),
            Err(FilterError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn tile_output_pixel_type_follows_first_input() {
        let a = Image::from_vec(&[1, 1], vec![7u8]).unwrap();
        let out = tile(&[&a], &[1, 1], 0.0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    // ---- 64-bit-integer losslessness (native path, no f64 seam) ----------

    // Two distinct `UInt64` values above 2^53 whose exact bits cannot survive
    // an `f64` round-trip: the old `to_f64_vec` seam would have collapsed each
    // onto an even neighbour, so any test that recovers them exactly proves the
    // pixel path never widened through `f64`.
    const HI_A: u64 = (1 << 53) + 1;
    const HI_B: u64 = (1 << 53) + 3;

    fn assert_non_f64(v: u64) {
        // Non-vacuity guard: if this value *did* round-trip cleanly through
        // f64, recovering it exactly below would prove nothing.
        assert_ne!(v, (v as f64) as u64, "{v} survives an f64 round-trip");
    }

    #[test]
    fn checker_board_selects_u64_pixels_losslessly() {
        assert_non_f64(HI_A);
        assert_non_f64(HI_B);
        let a = Image::from_vec(&[4, 4], vec![HI_A; 16]).unwrap();
        let b = Image::from_vec(&[4, 4], vec![HI_B; 16]).unwrap();
        let out = checker_board(&a, &b, &[2, 2]).unwrap();
        let got = out.scalar_slice::<u64>().unwrap();
        let s = strides(&[4, 4]);
        for row in 0..4 {
            for col in 0..4 {
                let flat = col * s[0] + row * s[1];
                let want = if (col / 2 + row / 2) % 2 == 1 {
                    HI_B
                } else {
                    HI_A
                };
                assert_eq!(got[flat], want, "at ({col},{row})");
            }
        }
    }

    #[test]
    fn paste_moves_u64_pixels_losslessly() {
        assert_non_f64(HI_A);
        assert_non_f64(HI_B);
        let dest = Image::from_vec(&[4, 4], vec![HI_A; 16]).unwrap();
        let src = Image::from_vec(&[2, 2], vec![HI_B; 4]).unwrap();
        let out = paste(&dest, &src, &[0, 0], &[2, 2], &[1, 1]).unwrap();
        let got = out.scalar_slice::<u64>().unwrap();
        let s = strides(&[4, 4]);
        for row in 0..4 {
            for col in 0..4 {
                let flat = col * s[0] + row * s[1];
                let inside = (1..=2).contains(&col) && (1..=2).contains(&row);
                assert_eq!(got[flat], if inside { HI_B } else { HI_A });
            }
        }
    }

    #[test]
    fn tile_moves_u64_pixels_losslessly() {
        assert_non_f64(HI_A);
        assert_non_f64(HI_B);
        let a = Image::from_vec(&[2, 1], vec![HI_A, HI_A]).unwrap();
        let b = Image::from_vec(&[2, 1], vec![HI_B, HI_B]).unwrap();
        // `default_pixel_value` 0.0 fills no cell here (both cells are full),
        // so every output pixel is an exact copy of an input pixel.
        let out = tile(&[&a, &b], &[2, 1], 0.0).unwrap();
        assert_eq!(out.size(), &[4, 1]);
        assert_eq!(
            out.scalar_slice::<u64>().unwrap(),
            &[HI_A, HI_A, HI_B, HI_B]
        );
    }
}
