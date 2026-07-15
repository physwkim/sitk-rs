//! `VectorConnectedComponentImageFilter`: label pixels whose vectors chain
//! together within a dot-product-based distance threshold.
//!
//! Port of `Modules/Segmentation/ConnectedComponents/include/`:
//! `itkVectorConnectedComponentImageFilter.h` is a thin instantiation of
//! `itkConnectedComponentFunctorImageFilter.hxx` -- the exact same base class
//! [`crate::filters::scalar_connected_component::scalar_connected_component`] ports --
//! with `Functor::SimilarVectorsFunctor` as its join predicate instead of
//! `Functor::SimilarPixelsFunctor`. Everything in that sibling module's docs
//! about the sweep (one raster pass over each pixel's "previous" neighbors,
//! adopting/unioning labels, `EquivalencyTable` collapsed into a single
//! union-find) and the label-numbering convention applies here unchanged;
//! this module only replaces the join predicate and drops the `MaskImage`
//! parameter, which `VectorConnectedComponentImageFilter.yaml` does not
//! expose (only `DistanceThreshold` and `FullyConnected`).
//!
//! **This is not [`crate::filters::label::connected_component`]**: every pixel gets a
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
//! `NumericTraits<ValueType>::RealType` -- `double` for **every** scalar
//! component type (`NumericTraits<float>::RealType` is `double`,
//! itkNumericTraits.h:1349/1356), the same accumulator rule
//! [`crate::filters::vector::vector_magnitude`]'s `GetSquaredNorm`-style accumulation
//! uses. `threshold` is `DistanceThreshold` cast to the component
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
//! `orthogonal_regions_merge_under_the_default_threshold` test.
//!
//! A **zero** vector is not special-cased *upstream*: `1 - |dot(0, b)| = 1`
//! joins any neighbor at the default `DistanceThreshold = 1`, exactly like an
//! orthogonal vector would — so a flat (zero) region silently bridges or
//! absorbs adjacent directioned regions on realistic input (a gradient- or
//! displacement-direction field is zero wherever the field is flat). **This
//! port fixes that (§2.40):** a zero vector has no direction, so it is
//! dissimilar to a directioned neighbor. The fix guards *only* the mixed
//! zero/non-zero case; zero-vs-zero is left to the ordinary predicate, whose
//! distance `1 - |dot(0, 0)| = 1` merges the two exactly when `threshold >= 1`
//! — as at the yaml default `1.0`, where an all-zero image stays one component
//! — but not below it: at `threshold = 0.5` even identical zero vectors
//! separate, like orthogonal directioned vectors would. It is that
//! `distance == 1 <= threshold` test, not the two vectors' identity, that
//! merges them; only the mixed zero/non-zero case changes behavior versus
//! upstream. The unnormalized dot product and the antiparallel-similarity are
//! documented, deliberate upstream behavior and are kept.
//!
//! `VectorConnectedComponentImageFilter.yaml`'s `pixel_types` is
//! `RealVectorPixelIDTypeList`, and `itkConceptMacro(InputValyeTypeIsFloatingCheck, ...)`
//! (sic -- upstream's own concept-check name has a typo, `Valye` for
//! `Value`; itkVectorConnectedComponentImageFilter.h:145) enforces it at
//! compile time in C++. This port checks
//! [`crate::core::PixelId::is_floating_point`] on the component type at
//! runtime and returns [`FilterError::RequiresRealPixelType`].

use crate::core::{Image, Scalar, dispatch_scalar};
use crate::filters::error::{FilterError, Result};
use crate::filters::reconstruction::{Half, NeighborWalker};

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
    // Fix (§2.40): a zero vector has no direction, so it is not directionally
    // similar to a *directioned* neighbor. Upstream leaves it unguarded, so
    // `1 - |dot(0, b)| = 1` joins any neighbor at the wrapped default
    // `DistanceThreshold = 1` — a flat (zero) region then silently bridges or
    // absorbs adjacent directioned regions on realistic input (a gradient- or
    // displacement-direction field is zero wherever the field is flat). Zero-
    // vs-zero is NOT special-cased to merge: the guard rejects only the mixed
    // case, so two zero vectors flow through the ordinary predicate (distance
    // `1 - |dot(0,0)| = 1`) and merge only when `threshold >= 1` (the yaml
    // default `1.0` -> an all-zero image is one component; at `0.5` the zeros
    // separate). Identity is not what merges them, `distance == 1 <= threshold`
    // is. Only the mixed zero/non-zero case changes behavior versus upstream.
    let a_zero = a.iter().all(|&x| x.as_f64() == 0.0);
    let b_zero = b.iter().all(|&x| x.as_f64() == 0.0);
    if a_zero != b_zero {
        return false;
    }
    // The dot product accumulates in `NumericTraits<ValueType>::RealType`,
    // which is `double` even for a `float` component type.
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x.as_f64() * y.as_f64())
        .sum();
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
/// Errors with [`crate::core::Error::RequiresVectorPixelType`] on a scalar
/// image, and [`FilterError::RequiresRealPixelType`] on a vector image whose
/// component type is not `Float32`/`Float64`.
pub fn vector_connected_component(
    image: &Image,
    distance_threshold: f64,
    fully_connected: bool,
) -> Result<Image> {
    if !image.pixel_id().is_vector() {
        return Err(crate::core::Error::RequiresVectorPixelType(image.pixel_id()).into());
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
    use crate::core::PixelId;

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

    /// Fix (§2.40): a zero vector is dissimilar to a directioned neighbor, so
    /// it no longer joins one even at the default threshold `1.0` — where
    /// upstream would merge them (`1 - |dot| = 1 <= 1`). The two pixels stay
    /// separate at both thresholds.
    #[test]
    fn a_zero_vector_does_not_join_a_directioned_neighbor() {
        let image = vec_img(&[2], 2, vec![0.0, 0.0, 1.0, 0.0]);
        let below = vector_connected_component(&image, 0.5, false).unwrap();
        assert_eq!(below.scalar_slice::<u32>().unwrap(), &[1, 2]);
        let at_boundary = vector_connected_component(&image, 1.0, false).unwrap();
        assert_eq!(at_boundary.scalar_slice::<u32>().unwrap(), &[1, 2]);
    }

    /// Two zero vectors are still similar (they are identical), so an all-zero
    /// image collapses to a single component — the fix guards only the mixed
    /// zero/non-zero case, not zero-against-zero.
    ///
    /// This threshold=`1.0` case pins nothing about the §2.40 guard on its own:
    /// zero-vs-zero skips the `a_zero != b_zero` early-return, then takes the
    /// normal dot-product path (`1 - |dot(0,0)| = 1 <= 1.0`), which merges with
    /// *or* without the guard. See
    /// `zero_vectors_stay_separate_below_the_boundary_threshold` for the case
    /// that actually pins the guard's zero-vs-zero semantics.
    #[test]
    fn zero_vectors_still_merge_with_each_other() {
        let image = vec_img(&[3], 2, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let out = vector_connected_component(&image, 1.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 1]);
    }

    /// Pins the §2.40 guard's *zero-vs-zero* semantics: the fix does **not**
    /// special-case identical zero vectors to always-merge — it only early-
    /// returns `false` for the mixed zero/non-zero case (`a_zero != b_zero`).
    /// Zero-vs-zero still flows through the ordinary dot-product predicate,
    /// giving distance `1 - |dot(0,0)| = 1`, so below the boundary threshold
    /// `1.0` even two identical zero vectors are dissimilar and stay separate,
    /// exactly like two orthogonal directioned vectors would. At threshold
    /// `0.5` the three zero pixels therefore form three components, not one —
    /// a naive "identical zeros always merge" special case would wrongly yield
    /// `[1, 1, 1]` here.
    #[test]
    fn zero_vectors_stay_separate_below_the_boundary_threshold() {
        let image = vec_img(&[3], 2, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let out = vector_connected_component(&image, 0.5, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 2, 3]);
    }

    /// A zero vector wedged between two identical directioned regions no longer
    /// bridges them: upstream's unguarded functor would chain
    /// `(1,0) ~ 0 ~ (1,0)` into one component at the default threshold; here the
    /// zero forms its own component and the two directioned runs stay apart.
    #[test]
    fn a_zero_vector_does_not_bridge_two_directioned_regions() {
        let image = vec_img(&[3], 2, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let out = vector_connected_component(&image, 1.0, false).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 2, 3]);
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
            FilterError::Core(crate::core::Error::RequiresVectorPixelType(
                PixelId::Float64
            ))
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
