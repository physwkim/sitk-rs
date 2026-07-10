//! `VectorConnectedComponentImageFilter`: label pixels whose vectors chain
//! together within a dot-product-based distance threshold.
//!
//! Port of `Modules/Segmentation/ConnectedComponents/include/`:
//! `itkVectorConnectedComponentImageFilter.h` is a thin instantiation of
//! `itkConnectedComponentFunctorImageFilter.hxx` -- the exact same base class
//! [`crate::scalar_connected_component::scalar_connected_component`] ports --
//! with `Functor::SimilarVectorsFunctor` as its join predicate instead of
//! `Functor::SimilarPixelsFunctor`. Everything in that sibling module's docs
//! about the sweep (one raster pass over each pixel's "previous" neighbors,
//! adopting/unioning labels, `EquivalencyTable` collapsed into a single
//! union-find) and the label-numbering convention applies here unchanged;
//! this module only replaces the join predicate and drops the `MaskImage`
//! parameter, which `VectorConnectedComponentImageFilter.yaml` does not
//! expose (only `DistanceThreshold` and `FullyConnected`).
//!
//! **This is not [`crate::label::connected_component`]**: every pixel gets a
//! label, including pixels that never match any neighbor -- there is no
//! background value, "0" is a perfectly ordinary label here.
//!
//! ## The join predicate
//!
//! `Functor::SimilarVectorsFunctor::operator()` (`itkVectorConnectedComponentImageFilter.h`):
//! two vectors `a`, `b` join when
//!
//! `static_cast<ValueType>(1.0 - |dot(a, b)|) <= threshold`
//!
//! where `dot(a, b) = sum_i a[i]*b[i]`, accumulated in
//! `NumericTraits<ValueType>::RealType` -- `f32` only for an `f32` component
//! type, `f64` otherwise, the same precision split
//! [`crate::vector::vector_magnitude`]'s `GetSquaredNorm`-style accumulation
//! already uses. `threshold` is `DistanceThreshold` cast to the component
//! type (`static_cast<InputValueType>`, matching
//! `VectorConnectedComponentImageFilter.yaml`'s `pixeltype: Input` cast).
//!
//! The functor's own docstring: "Assumes vectors are normalized" -- it does
//! **not** normalize `a`/`b` itself, so this port doesn't either. Aligned
//! (`dot = 1`) and anti-aligned / 180-degrees-out-of-phase (`dot = -1`)
//! vectors both give distance `0` (always join, at any non-negative
//! threshold) -- "vectors that are 180 degrees out of phase are similar" per
//! the class docs, because only `|dot|` is used. Orthogonal vectors
//! (`dot = 0`) give distance `1`, joining only when `threshold >= 1` --
//! notably, `VectorConfidenceConnectedImageFilter.yaml`'s
//! `DistanceThreshold` default of `1.0` sits exactly on that boundary, so
//! two regions of orthogonal (but otherwise unrelated) vectors merge under
//! the *default* threshold; see the
//! `orthogonal_regions_merge_under_the_default_threshold` test. A **zero**
//! vector is not special-cased anywhere in the functor (there is no
//! normalization to divide by zero in the first place): it gives `dot = 0`
//! against anything, so it joins a neighbor exactly like an orthogonal
//! vector would, never producing `NaN` or an error.
//!
//! `VectorConnectedComponentImageFilter.yaml`'s `pixel_types` is
//! `RealVectorPixelIDTypeList`, and `itkConceptMacro(InputValyeTypeIsFloatingCheck, ...)`
//! (sic -- upstream's own concept-check name has a typo, `Valye` for
//! `Value`; itkVectorConnectedComponentImageFilter.h:145) enforces it at
//! compile time in C++. This port checks
//! [`sitk_core::PixelId::is_floating_point`] on the component type at
//! runtime and returns [`FilterError::RequiresRealPixelType`].

use crate::error::{FilterError, Result};
use crate::reconstruction::{Half, NeighborWalker};
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar};

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while cur != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// `Functor::SimilarVectorsFunctor::operator()`; see the module docs.
fn similar<T: Scalar>(a: &[T], b: &[T], threshold: T) -> bool {
    let dot: f64 = if T::PIXEL_ID == PixelId::Float32 {
        let sum: f32 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x.as_f64() as f32) * (y.as_f64() as f32))
            .sum();
        f64::from(sum)
    } else {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x.as_f64() * y.as_f64())
            .sum()
    };
    T::from_f64(1.0 - dot.abs()).as_f64() <= threshold.as_f64()
}

fn vector_connected_component_typed<T: Scalar>(
    image: &Image,
    distance_threshold: f64,
    fully_connected: bool,
) -> Result<Image> {
    let size = image.size();
    let total: usize = size.iter().product();
    let components = image.number_of_components_per_pixel();
    let comp = image.component_slice::<T>()?;
    let threshold = T::from_f64(distance_threshold);

    let mut parent: Vec<usize> = (0..total).collect();
    let mut walker = NeighborWalker::new(size, fully_connected, Half::Previous);
    for pos in 0..total {
        let value = &comp[pos * components..pos * components + components];
        for &neigh in walker.at(pos, size) {
            let neighbor_value = &comp[neigh * components..neigh * components + components];
            if similar::<T>(value, neighbor_value, threshold) {
                union(&mut parent, pos, neigh);
            }
        }
    }

    let mut root_to_output: Vec<Option<u32>> = vec![None; total];
    let mut next_label = 1u32;
    let mut out = vec![0u32; total];
    for (pos, slot) in out.iter_mut().enumerate() {
        let root = find(&mut parent, pos);
        let label = *root_to_output[root].get_or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        *slot = label;
    }

    let mut out_image = Image::from_vec(size, out)?;
    out_image.copy_geometry_from(image);
    Ok(out_image)
}

/// `VectorConnectedComponentImageFilter`: labels pixels whose vectors chain
/// together within `distance_threshold` under
/// [`Functor::SimilarVectorsFunctor`](self)'s join predicate (see the module
/// docs) -- transitively, and with no background exclusion.
///
/// Errors with [`sitk_core::Error::RequiresVectorPixelType`] on a scalar
/// image, and [`FilterError::RequiresRealPixelType`] on a vector image whose
/// component type is not `Float32`/`Float64`.
pub fn vector_connected_component(
    image: &Image,
    distance_threshold: f64,
    fully_connected: bool,
) -> Result<Image> {
    if !image.pixel_id().is_vector() {
        return Err(sitk_core::Error::RequiresVectorPixelType(image.pixel_id()).into());
    }
    if !image.pixel_id().component_id().is_floating_point() {
        return Err(FilterError::RequiresRealPixelType(image.pixel_id()));
    }
    dispatch_scalar!(
        image.pixel_id(),
        vector_connected_component_typed,
        image,
        distance_threshold,
        fully_connected
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_img(size: &[usize], components: usize, data: Vec<f64>) -> Image {
        Image::from_vec_vector(size, components, data).unwrap()
    }

    /// Required hand-derived case: two blocks of orthogonal vectors
    /// (`(1,0)` vs `(0,1)`, `dot = 0`, distance `1`) stay separate under a
    /// threshold below `1`.
    #[test]
    fn two_orthogonal_vector_regions_stay_separate_below_the_boundary_threshold() {
        #[rustfmt::skip]
        let image = vec_img(&[4], 2, vec![
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
        ]);
        let out = vector_connected_component(&image, 0.5, false).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt32);
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 2, 2]);
    }

    /// At the yaml default threshold (`1.0`) the same orthogonal regions
    /// merge instead: `distance == 1 <= 1.0` is true (inclusive boundary).
    #[test]
    fn orthogonal_regions_merge_under_the_default_threshold() {
        #[rustfmt::skip]
        let image = vec_img(&[4], 2, vec![
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
        ]);
        let out = vector_connected_component(&image, 1.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 1, 1]);
    }

    /// "Vectors that are 180 degrees out of phase are similar": `dot = -1`
    /// gives distance `0`, joining even at threshold `0`.
    #[test]
    fn anti_parallel_vectors_join_at_zero_threshold() {
        let image = vec_img(&[2], 2, vec![1.0, 0.0, -1.0, 0.0]);
        let out = vector_connected_component(&image, 0.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1]);
    }

    /// The zero vector is not special-cased: it gives `dot = 0` against
    /// anything, exactly like an orthogonal vector, and never produces a
    /// `NaN` or an error.
    #[test]
    fn zero_vector_behaves_like_an_orthogonal_vector() {
        let image = vec_img(&[2], 2, vec![0.0, 0.0, 1.0, 0.0]);
        let below = vector_connected_component(&image, 0.5, false).unwrap();
        assert_eq!(below.scalar_slice::<u32>().unwrap(), &[1, 2]);
        let at_boundary = vector_connected_component(&image, 1.0, false).unwrap();
        assert_eq!(at_boundary.scalar_slice::<u32>().unwrap(), &[1, 1]);
    }

    /// Two single-pixel components that touch only diagonally: face
    /// connectivity keeps them separate, full connectivity merges them --
    /// mirrors `scalar_connected_component`'s own connectivity test.
    #[test]
    fn fully_connected_merges_diagonally_touching_vectors() {
        #[rustfmt::skip]
        let image = vec_img(&[3, 3], 2, vec![
            1.0,0.0,   0.0,1.0,  0.0,1.0,
            0.0,1.0,   1.0,0.0,  0.0,1.0,
            0.0,1.0,   0.0,1.0,  0.0,1.0,
        ]);
        let face = vector_connected_component(&image, 0.5, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u32>().unwrap(), &[
            1, 2, 2,
            2, 3, 2,
            2, 2, 2,
        ]);
        let full = vector_connected_component(&image, 0.5, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.scalar_slice::<u32>().unwrap(), &[
            1, 2, 2,
            2, 1, 2,
            2, 2, 2,
        ]);
    }

    /// Labels are contiguous `1..=N` in ascending raster order of first
    /// appearance, matching `scalar_connected_component`'s documented
    /// convention (ITK's own `EquivalencyTable`-based numbering is
    /// explicitly documented as arbitrary and possibly gapped --
    /// `itkConnectedComponentFunctorImageFilter.h`: "The final object labels
    /// are in no particular order (and some object labels may not be
    /// used)").
    #[test]
    fn labels_are_contiguous_by_raster_order_of_first_appearance() {
        #[rustfmt::skip]
        let image = vec_img(&[6], 2, vec![
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 0.0,
        ]);
        let out = vector_connected_component(&image, 0.5, false).unwrap();
        // Three separate blocks (the third is not adjacent to the first, even
        // though it shares the same vector), numbered 1, 2, 3 in raster order.
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn rejects_a_scalar_image() {
        let image = Image::from_vec(&[2], vec![1.0f64, 2.0]).unwrap();
        assert!(matches!(
            vector_connected_component(&image, 1.0, false).unwrap_err(),
            FilterError::Core(sitk_core::Error::RequiresVectorPixelType(PixelId::Float64))
        ));
    }

    #[test]
    fn rejects_an_integer_component_vector_image() {
        let image = Image::from_vec_vector(&[2], 2, vec![1u8, 0, 0, 1]).unwrap();
        assert_eq!(
            vector_connected_component(&image, 1.0, false).unwrap_err(),
            FilterError::RequiresRealPixelType(PixelId::VectorUInt8)
        );
    }
}
