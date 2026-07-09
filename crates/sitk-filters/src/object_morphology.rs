//! ITK's isolated-object morphology: dilation/erosion restricted to the
//! object/background boundary, evaluated only at object pixels that touch a
//! differently-valued neighbor -- not a plain full-image binary dilate/erode.
//!
//! Verified against ITK's `Modules/Filtering/{MathematicalMorphology,
//! BinaryMathematicalMorphology}/include/`: `itkObjectMorphologyImageFilter.h`
//! / `.hxx` (the shared base class), `itkDilateObjectMorphologyImageFilter.h`
//! / `.hxx`, `itkErodeObjectMorphologyImageFilter.h` / `.hxx`.
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
//! `Evaluate` itself (`itkDilateObjectMorphologyImageFilter.hxx:31-48` /
//! `itkErodeObjectMorphologyImageFilter.hxx:31-48`) then paints every
//! kernel-on offset around `p` -- using the *caller's* kernel radius, which
//! may be larger than the radius-1 box used to detect the boundary -- via
//! `NeighborhoodIterator::SetPixel(n, v, status)`, "a special SetPixel method
//! which quietly ignores out-of-bounds attempts"
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
//! (`ErodeObjectMorphologyImageFilter.hxx:25-29`: `NumericTraits<PixelType>
//! ::max()`) and calls `OverrideBoundaryCondition`. But
//! `IsObjectPixelOnBoundary` only ever *reads* that overridden condition
//! (`iNIter.GetPixel(i)`, no `isInside` check) when `m_UseBoundaryCondition
//! == true` (`itkObjectMorphologyImageFilter.h:162-172`: "Defaults to false
//! ... if false ... does not consider that outside extent"); neither
//! constructor ever calls `SetUseBoundaryCondition(true)`, and neither
//! `DilateObjectMorphologyImageFilter.yaml` nor
//! `ErodeObjectMorphologyImageFilter.yaml` exposes a member for it. So,
//! reached only through SimpleITK, these carefully-chosen sentinel boundary
//! conditions are set but **never consulted** -- the filter always takes the
//! `else` (`isInside`-gated) branch, i.e. always behaves as if
//! `UseBoundaryCondition == false`. This port implements only that reachable
//! behavior.
//!
//! ## Dilation matches a plain binary dilate; erosion does not
//!
//! The base class's own doc comment
//! (`itkObjectMorphologyImageFilter.h:36-40`) warns: "this filter operates
//! significantly faster than itkBinaryMorphologicalImageFilters; however
//! itk*Binary*MorphologicalImageFilters preserve background pixels based on
//! values of neighboring background pixels -- potentially important during
//! erosion." Concretely:
//!
//! - **Dilate never diverges** from [`crate::morphology::binary_dilate`] on
//!   the same kernel: for any pixel `y` a full/naive dilate would paint, the
//!   object pixel nearest to `y` along the object always itself qualifies as
//!   a boundary pixel (it must border a non-object pixel somewhere between
//!   it and `y`, or be `y`'s own object source) and its kernel-radius reach
//!   covers `y` at least as well as any interior pixel's would -- a boundary
//!   pixel's reach always dominates. See
//!   `dilate_matches_plain_binary_dilate_on_an_isolated_point` below.
//! - **Erode routinely over-erodes** relative to
//!   [`crate::morphology::binary_erode`]: a boundary pixel's *own*
//!   kernel-radius neighborhood is stamped to `background_value` wholesale,
//!   including interior (non-boundary) object pixels that a true
//!   structuring-element erosion would have kept (their *own* radius-1
//!   neighbors are all still object). A solid rectangular block is the clean
//!   case: with a radius-1 kernel, only the exact geometric center of a 5x5
//!   block survives here, versus the correct 3x3 surviving core a real
//!   erosion keeps -- see
//!   `erode_solid_block_only_the_exact_center_survives_unlike_plain_binary_erode`
//!   below, which pins this directly against [`crate::morphology::binary_erode`].
//!
//! ## `ErodeObjectMorphologyImageFilter`'s extra `background_value`
//!
//! `ErodeObjectMorphologyImageFilter.h:79-99` adds `SetErodeValue`/
//! `GetErodeValue` (pure aliases for `ObjectValue`, "Added for API
//! consistency with itkBinaryErode") and a genuinely new `BackgroundValue`
//! member (default `0`, per `ErodeObjectMorphologyImageFilter.yaml`): the
//! value `Evaluate` (`itkErodeObjectMorphologyImageFilter.hxx:31-48`) paints
//! into every kernel-on position around a boundary object pixel, in place of
//! `DilateObjectMorphologyImageFilter::Evaluate`'s `ObjectValue`. Setting it
//! equal to `object_value` makes "erosion" paint the object value onto
//! background neighbors instead of erasing anything -- functionally
//! indistinguishable from dilation, a direct consequence of the shared
//! formula below, not a special case (see
//! `erode_object_value_equals_background_value_behaves_like_dilate`).

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

/// Shared core for [`dilate_object_morphology`]/[`erode_object_morphology`]
/// (`ObjectMorphologyImageFilter::DynamicThreadedGenerateData` +
/// `IsObjectPixelOnBoundary` + the subclass's `Evaluate` -- see module docs).
/// `paint_value` is the value `Evaluate` stamps into every kernel-on offset
/// around a boundary object pixel -- `ObjectValue` for dilate,
/// `BackgroundValue` for erode -- the only thing that differs between the
/// two subclasses' `Evaluate`.
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

fn erode_object_morphology_typed<T: Scalar>(
    img: &Image,
    kernel: &StructuringElement,
    object_value: f64,
    background_value: f64,
) -> Result<Image> {
    object_morphology_typed::<T>(img, kernel, object_value, background_value)
}

/// `ErodeObjectMorphologyImageFilter`: erodes the region equal to
/// `object_value`, painting `background_value` into every kernel-on offset
/// around each boundary object pixel (see module docs for exactly how, and
/// why, this diverges from [`crate::morphology::binary_erode`]). Defaults
/// per `ErodeObjectMorphologyImageFilter.yaml`: `object_value = 1`,
/// `background_value = 0`, `kernel` = a `sitkBall` of radius `1` per axis.
pub fn erode_object_morphology(
    img: &Image,
    kernel: &StructuringElement,
    object_value: f64,
    background_value: f64,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        erode_object_morphology_typed,
        img,
        kernel,
        object_value,
        background_value
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::{binary_dilate, binary_erode};

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

    // ---- erode_object_morphology ----

    /// The headline divergence from the module docs: a solid 5x5 block of
    /// `object_value`, margined by a 1-pixel background border (so the image
    /// edge itself never enters into it -- see
    /// `erode_identity_for_a_solid_object_touching_every_image_edge` for that
    /// separate effect). Every block pixel at Chebyshev distance 2 from the
    /// block's center is a boundary pixel (it borders the background
    /// margin); its radius-1 kernel then stamps `background_value` onto
    /// everything within Chebyshev distance 1 of *itself*, which reaches
    /// every block pixel at distance 1 from the center too (nearest boundary
    /// pixel is exactly `2 - 1 = 1` away). Only the exact center, at distance
    /// 2 from every boundary pixel, survives. A true structuring-element
    /// erosion (`binary_erode`) instead keeps the whole distance-<=1 core (a
    /// pixel survives iff *its own* neighbors are all still object, which
    /// holds for every distance-1 pixel here) -- the two disagree on all 8
    /// of those distance-1 pixels.
    #[test]
    fn erode_solid_block_only_the_exact_center_survives_unlike_plain_binary_erode() {
        #[rustfmt::skip]
        let data = vec![
            0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        let f = img_u8(&[7, 7], data);
        let kernel = StructuringElement::ball(&[1, 1]); // yaml default KernelType

        let object_out = erode_object_morphology(&f, &kernel, 1.0, 0.0).unwrap();
        #[rustfmt::skip]
        assert_eq!(object_out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]);

        let binary_out = binary_erode(&f, &kernel, 1.0, 0.0, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(binary_out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]);

        assert_ne!(
            object_out.scalar_slice::<u8>().unwrap(),
            binary_out.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn erode_identity_when_no_object_pixels_present() {
        let f = img_u8(&[3], vec![0, 0, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let out = erode_object_morphology(&f, &kernel, 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    /// Dual of `dilate_identity_for_a_solid_object_touching_every_image_edge`:
    /// the same "ignore out-of-image neighbors, don't substitute" default
    /// means a solid block touching the edge is never eroded either, even
    /// though a naive "the far side of the image edge is background" rule
    /// would erode every border pixel.
    #[test]
    fn erode_identity_for_a_solid_object_touching_every_image_edge() {
        let f = img_u8(&[3, 3], vec![1u8; 9]);
        let kernel = StructuringElement::box_(&[1, 1]);
        let out = erode_object_morphology(&f, &kernel, 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1u8; 9]);
    }

    /// The boundary check is *always* the fixed radius-1 box (see module
    /// docs), decoupled from the kernel's own radius: even at kernel radius
    /// 0, a boundary object pixel (idx1, adjacent to background idx2) is
    /// still detected and erased. `binary_erode` at radius 0 has no such
    /// decoupling -- its kernel *is* the boundary check, so "survives iff
    /// itself is object" is trivially true for every object pixel, making it
    /// the identity. The two rules coincide at every other radius tested in
    /// this module, but not here.
    #[test]
    fn erode_kernel_radius0_diverges_from_binary_erodes_radius0_identity() {
        let f = img_u8(&[3], vec![1, 1, 0]);
        let kernel = StructuringElement::box_(&[0]);

        let object_out = erode_object_morphology(&f, &kernel, 1.0, 0.0).unwrap();
        assert_eq!(object_out.scalar_slice::<u8>().unwrap(), &[1, 0, 0]);

        let binary_out = binary_erode(&f, &kernel, 1.0, 0.0, true).unwrap();
        assert_eq!(binary_out.scalar_slice::<u8>().unwrap(), &[1, 1, 0]);
    }

    #[test]
    fn erode_custom_background_value_is_used_for_erased_pixels() {
        let f = img_u8(&[3], vec![9, 9, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let out = erode_object_morphology(&f, &kernel, 9.0, 7.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[7, 7, 7]);
    }

    /// Pins the module docs' `background_value == object_value` finding: the
    /// shared core's `paint_value` becomes indistinguishable from
    /// `object_value`, so "erosion" reproduces dilation exactly.
    #[test]
    fn erode_object_value_equals_background_value_behaves_like_dilate() {
        let f = img_u8(&[5], vec![1, 1, 1, 0, 0]);
        let kernel = StructuringElement::box_(&[1]);
        let eroded = erode_object_morphology(&f, &kernel, 1.0, 1.0).unwrap();
        let dilated = dilate_object_morphology(&f, &kernel, 1.0).unwrap();
        assert_eq!(
            eroded.scalar_slice::<u8>().unwrap(),
            dilated.scalar_slice::<u8>().unwrap()
        );
        assert_eq!(eroded.scalar_slice::<u8>().unwrap(), &[1, 1, 1, 1, 0]);
    }

    #[test]
    fn erode_rejects_a_kernel_radius_of_the_wrong_dimension() {
        let f = img_u8(&[3, 3], vec![0u8; 9]);
        let kernel = StructuringElement::box_(&[1]); // 1-D radius, 2-D image
        assert_eq!(
            erode_object_morphology(&f, &kernel, 1.0, 0.0).unwrap_err(),
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }
}
