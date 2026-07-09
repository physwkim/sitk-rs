//! ITK's isolated-object morphology: dilation restricted to the
//! object/background boundary, evaluated only at object pixels that touch a
//! differently-valued neighbor -- not a plain full-image binary dilate.
//!
//! Verified against ITK's `Modules/Filtering/{MathematicalMorphology,
//! BinaryMathematicalMorphology}/include/`: `itkObjectMorphologyImageFilter.h`
//! / `.hxx` (the shared base class) and
//! `itkDilateObjectMorphologyImageFilter.h` / `.hxx`.
//!
//! ## The base algorithm is boundary-only, not a full kernel sweep
//!
//! `ObjectMorphologyImageFilter::DynamicThreadedGenerateData`
//! (itkObjectMorphologyImageFilter.hxx:116-163) walks every pixel `p` and
//! only calls the subclass's `Evaluate` (which paints `p`'s *own*
//! kernel-radius neighborhood in the output image) when both hold:
//!
//! 1. `input[p] == ObjectValue` (line 151, `Math::ExactlyEquals`), and
//! 2. `IsObjectPixelOnBoundary` says `p` touches a non-`ObjectValue`
//!    neighbor (line 153).
//!
//! `IsObjectPixelOnBoundary` (itkObjectMorphologyImageFilter.hxx:166-198)
//! always checks the **fixed 3-per-axis (`3^dim`) box** around `p` -- *never*
//! the caller's actual (possibly larger) kernel radius. With
//! `UseBoundaryCondition` at its class default of `false` (see below), a
//! neighbor that falls outside the image is simply skipped, not substituted
//! with any boundary value (the `isInside` guard on line 190). This port
//! reproduces exactly that: a fixed radius-1 box check, independent of the
//! kernel radius passed to [`dilate_object_morphology`], with out-of-image
//! neighbors ignored rather than substituted.
//!
//! `Evaluate` itself (`itkDilateObjectMorphologyImageFilter.hxx:31-48`) then
//! paints every kernel-on offset around `p` -- using the *caller's* kernel
//! radius, which may be larger than the radius-1 box used to detect the
//! boundary -- via `NeighborhoodIterator::SetPixel(n, v, status)`, "a special
//! SetPixel method which quietly ignores out-of-bounds attempts"
//! (`itkNeighborhoodIterator.h:277-280`): a kernel offset that lands outside
//! the image is silently dropped, not clamped or wrapped.
//!
//! ## `BeforeThreadedGenerateData` reduces to a plain copy
//!
//! `itkObjectMorphologyImageFilter.hxx:85-113` fills the whole output with
//! `1` if `ObjectValue == 0`, else `0`, then overwrites every output pixel
//! that doesn't equal `ObjectValue` with the corresponding input pixel. The
//! fill sentinel is chosen so it always differs from `ObjectValue` (`1 != 0`
//! when `ObjectValue == 0`; `0 != ObjectValue` whenever `ObjectValue != 0`),
//! so the "if output != ObjectValue" guard is true for every pixel on this
//! very first pass -- the whole dance is exactly equivalent to `output :=
//! input.clone()`, which is what this port does directly rather than
//! replaying the redundant fill.
//!
//! ## `UseBoundaryCondition` is unreachable from SimpleITK -- the filter's
//! own boundary condition object is dead configuration
//!
//! `DilateObjectMorphologyImageFilter`'s constructor
//! (`itkDilateObjectMorphologyImageFilter.hxx:25-29`) sets a
//! `ConstantBoundaryCondition` to `NumericTraits<PixelType>::NonpositiveMin()`
//! and calls `OverrideBoundaryCondition`. But `IsObjectPixelOnBoundary` only
//! ever *reads* that overridden condition (`iNIter.GetPixel(i)`, no
//! `isInside` check) when `m_UseBoundaryCondition == true`
//! (`itkObjectMorphologyImageFilter.h:162-172`: "Defaults to false ... if
//! false ... does not consider that outside extent"); the constructor never
//! calls `SetUseBoundaryCondition(true)`, and
//! `DilateObjectMorphologyImageFilter.yaml` exposes no member for it. So,
//! reached only through SimpleITK, this carefully-chosen sentinel boundary
//! condition is set but **never consulted** -- the filter always takes the
//! `else` (`isInside`-gated) branch, i.e. always behaves as if
//! `UseBoundaryCondition == false`. This port implements only that reachable
//! behavior.
//!
//! ## Dilation never diverges from a plain binary dilate
//!
//! The base class's own doc comment
//! (`itkObjectMorphologyImageFilter.h:36-40`) warns that the full
//! `itk*Binary*MorphologicalImageFilters` "preserve background pixels based
//! on values of neighboring background pixels -- potentially important
//! during erosion" -- calling out erosion specifically. For dilation, this
//! port never observed a difference from [`crate::morphology::binary_dilate`]
//! on the same kernel: for any pixel `y` a full/naive dilate would paint,
//! the object pixel nearest to `y` along the object always itself qualifies
//! as a boundary pixel (it must border a non-object pixel somewhere between
//! it and `y`, or be `y`'s own object source) and its kernel-radius reach
//! covers `y` at least as well as any interior pixel's would -- a boundary
//! pixel's reach always dominates. See
//! `dilate_matches_plain_binary_dilate_on_an_isolated_point` below.

use crate::error::{FilterError, Result};
use crate::morphology::StructuringElement;
use sitk_core::{Image, Scalar, dispatch_scalar};

/// Per-offset ND coordinates for a `radius`-sized window, dimension-0-fastest
/// -- the same enumeration [`crate::morphology`]'s own (private)
/// `window_offsets` builds, and the order [`StructuringElement`]'s `on()`
/// mask lines up with; duplicated locally per this crate's existing
/// convention of re-deriving this small enumeration in each module that
/// needs it, rather than exporting it across a module boundary (see
/// `crate::morphology::window_offsets`'s own doc comment).
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

/// `coord + offset`'s linear index, or `None` if any axis falls outside
/// `[0, size[d])` -- the "quietly ignore out-of-bounds" rule both the
/// boundary check and `Evaluate`'s painting use (see module docs).
fn linear_index_if_inside(
    coord: &[i64],
    offset: &[i64],
    size: &[usize],
    strides: &[i64],
) -> Option<usize> {
    let mut q = 0i64;
    for d in 0..coord.len() {
        let c = coord[d] + offset[d];
        if c < 0 || c >= size[d] as i64 {
            return None;
        }
        q += c * strides[d];
    }
    Some(q as usize)
}

/// Shared core for [`dilate_object_morphology`]
/// (`ObjectMorphologyImageFilter::DynamicThreadedGenerateData` +
/// `IsObjectPixelOnBoundary` + the subclass's `Evaluate` -- see module docs).
/// `paint_value` is the value `Evaluate` stamps into every kernel-on offset
/// around a boundary object pixel -- `ObjectValue` for dilate.
fn object_morphology_typed<T: Scalar>(
    img: &Image,
    kernel: &StructuringElement,
    object_value: f64,
    paint_value: f64,
) -> Result<Image> {
    let dim = img.dimension();
    if kernel.radius().len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: kernel.radius().len(),
        });
    }
    let object_value = T::from_f64(object_value);
    let paint_value = T::from_f64(paint_value);

    let input = img.scalar_slice::<T>()?;
    let size = img.size();

    let mut strides = vec![1i64; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1] as i64;
    }

    // `BeforeThreadedGenerateData` (itkObjectMorphologyImageFilter.hxx:85-113)
    // always ends up copying every input pixel to output -- see module docs.
    let mut output = input.to_vec();

    // Boundary detection always uses the fixed radius-1 box, independent of
    // the kernel's own radius (see module docs).
    let boundary_offsets = window_offsets(&vec![1usize; dim]);
    let kernel_offsets = window_offsets(kernel.radius());

    let mut coord = vec![0i64; dim];
    for (p, &v) in input.iter().enumerate() {
        if v != object_value {
            continue;
        }

        let mut rem = p;
        for (d, c) in coord.iter_mut().enumerate() {
            *c = (rem % size[d]) as i64;
            rem /= size[d];
        }

        let on_boundary = boundary_offsets.iter().any(|off| {
            linear_index_if_inside(&coord, off, size, &strides)
                .is_some_and(|q| input[q] != object_value)
        });
        if !on_boundary {
            continue;
        }

        for (k_off, &on) in kernel_offsets.iter().zip(kernel.on()) {
            if !on {
                continue;
            }
            if let Some(q) = linear_index_if_inside(&coord, k_off, size, &strides) {
                output[q] = paint_value;
            }
        }
    }

    let mut result = Image::from_vec(size, output)?;
    result.copy_geometry_from(img);
    Ok(result)
}

fn dilate_object_morphology_typed<T: Scalar>(
    img: &Image,
    kernel: &StructuringElement,
    object_value: f64,
) -> Result<Image> {
    object_morphology_typed::<T>(img, kernel, object_value, object_value)
}

/// `DilateObjectMorphologyImageFilter`: dilates the region equal to
/// `object_value`, evaluated only at the object/background boundary (see
/// module docs -- this is *not* the same algorithm as
/// [`crate::morphology::binary_dilate`], though the two were never observed
/// to actually disagree). Defaults per `DilateObjectMorphologyImageFilter.yaml`:
/// `object_value = 1`, `kernel` = a `sitkBall` of radius `1` per axis
/// ([`StructuringElement::ball`]).
pub fn dilate_object_morphology(
    img: &Image,
    kernel: &StructuringElement,
    object_value: f64,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        dilate_object_morphology_typed,
        img,
        kernel,
        object_value
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::binary_dilate;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn dilate_single_pixel_ball_radius1_grows_a_3x3_block() {
        let mut data = vec![0u8; 25];
        data[2 + 5 * 2] = 1; // (x=2, y=2), dead center of a 5x5 image
        let f = img_u8(&[5, 5], data);
        let kernel = StructuringElement::ball(&[1, 1]); // yaml default KernelType
        let out = dilate_object_morphology(&f, &kernel, 1.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]);
    }

    #[test]
    fn dilate_matches_plain_binary_dilate_on_an_isolated_point() {
        let mut data = vec![0u8; 25];
        data[2 + 5 * 2] = 1;
        let f = img_u8(&[5, 5], data);
        let kernel = StructuringElement::ball(&[1, 1]);
        let object_out = dilate_object_morphology(&f, &kernel, 1.0).unwrap();
        let binary_out = binary_dilate(&f, &kernel, 1.0, 0.0, false).unwrap();
        assert_eq!(
            object_out.scalar_slice::<u8>().unwrap(),
            binary_out.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn dilate_identity_when_no_object_pixels_present() {
        let f = img_u8(&[5], vec![0, 0, 0, 0, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let out = dilate_object_morphology(&f, &kernel, 1.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0, 0, 0]);
    }

    /// A solid block that touches every image edge has no in-image neighbor
    /// that ever differs from `ObjectValue`, and `UseBoundaryCondition`
    /// defaults to (and, via SimpleITK, is stuck at) `false` -- so no pixel
    /// is ever flagged boundary and the filter is a no-op, unlike a
    /// hypothetical "treat the edge as background" boundary rule (see module
    /// docs).
    #[test]
    fn dilate_identity_for_a_solid_object_touching_every_image_edge() {
        let f = img_u8(&[3, 3], vec![1u8; 9]);
        let kernel = StructuringElement::box_(&[1, 1]);
        let out = dilate_object_morphology(&f, &kernel, 1.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1u8; 9]);
    }

    /// `Evaluate`'s kernel-radius-0 window is just the object pixel's own
    /// position, so it always re-paints an already-`object_value` pixel back
    /// to itself: dilation at kernel radius 0 is always the identity.
    #[test]
    fn dilate_kernel_radius0_is_always_identity() {
        let f = img_u8(&[3], vec![1, 1, 0]);
        let kernel = StructuringElement::box_(&[0]);
        let out = dilate_object_morphology(&f, &kernel, 1.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 0]);
    }

    /// Pins the `BeforeThreadedGenerateData` derivation in the module docs:
    /// `object_value = 0` still dilates correctly, with no visible artifact
    /// from upstream's `FillBuffer(1)`-then-restore dance.
    #[test]
    fn dilate_object_value_zero_still_dilates_correctly() {
        let f = img_u8(&[4], vec![0, 0, 5, 5]);
        let kernel = StructuringElement::box_(&[1]);
        let out = dilate_object_morphology(&f, &kernel, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0, 5]);
    }

    #[test]
    fn dilate_rejects_a_kernel_radius_of_the_wrong_dimension() {
        let f = img_u8(&[3, 3], vec![0u8; 9]);
        let kernel = StructuringElement::box_(&[1]); // 1-D radius, 2-D image
        assert_eq!(
            dilate_object_morphology(&f, &kernel, 1.0).unwrap_err(),
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }
}
