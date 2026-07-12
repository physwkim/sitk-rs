//! N-dimensional neighborhood iteration over an [`Image`].
//!
//! Mirrors ITK's `itk::Neighborhood` (a fixed-size, self-describing window of
//! pixel values, itkNeighborhood.h) and `itk::ConstNeighborhoodIterator` (the
//! walk that produces one such window per pixel, with an interior fast path
//! that skips boundary checks entirely, itkConstNeighborhoodIterator.h).

use std::sync::Arc;

use crate::boundary::BoundaryCondition;
use crate::error::{Error, Result};
use crate::image::{Image, ScalarView};
use crate::parallel;
use crate::pixel::Scalar;

/// A snapshot of pixel values in an N-dimensional neighborhood window.
///
/// `values` is ordered dimension-0-fastest, matching ITK's neighborhood
/// offset table (itkNeighborhood.hxx:41-67, `ComputeNeighborhoodOffsetTable`)
/// and [`Image`]'s own pixel layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighborhood<T> {
    radius: Arc<[usize]>,
    size: Arc<[usize]>,
    values: Vec<T>,
}

impl<T: Copy> Neighborhood<T> {
    /// Per-dimension radius (itkNeighborhood.h:127-132, `GetRadius`).
    pub fn radius(&self) -> &[usize] {
        &self.radius
    }

    /// Per-dimension side length, `2 * radius + 1` (itkNeighborhood.h:150-155, `GetSize`).
    pub fn size(&self) -> &[usize] {
        &self.size
    }

    /// Neighbor values in dimension-0-fastest order.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Number of pixels in the window.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if the window holds no pixels (only possible for a zero-length image axis).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The value at the center of the window (itkNeighborhood.h:216-221, `GetCenterValue`).
    pub fn center_value(&self) -> T {
        self.values[self.values.len() / 2]
    }

    /// The value at ND `offset` from the center; `offset[d]` must be within
    /// `[-radius[d], radius[d]]` (itkNeighborhood.h:279-285 `GetOffset`,
    /// itkNeighborhood.hxx:102-113 `GetNeighborhoodIndex` inverted).
    pub fn get(&self, offset: &[i64]) -> T {
        let mut idx = self.values.len() as i64 / 2;
        let mut stride = 1i64;
        for (d, &o) in offset.iter().enumerate() {
            idx += o * stride;
            stride *= self.size[d] as i64;
        }
        self.values[idx as usize]
    }
}

/// Walks a local N-dimensional neighborhood of pixels across an [`Image`],
/// yielding one `(center index, `[`Neighborhood`]`)` pair per pixel in
/// dimension-0-fastest order.
///
/// Mirrors `itk::ConstNeighborhoodIterator<TImage, TBoundaryCondition>`
/// (itkConstNeighborhoodIterator.h). Pixels are read through a single
/// [`ScalarView`] taken at construction, never converted through `f64`; that
/// view is also the proof the boundary condition needs to read infallibly.
#[derive(Debug)]
pub struct NeighborhoodIterator<'a, T: Scalar, B: BoundaryCondition<T>> {
    view: ScalarView<'a, T>,
    radius: Arc<[usize]>,
    window_size: Arc<[usize]>,
    // Per-neighbor ND offset from the center, dimension-0-fastest.
    neighbor_offsets: Vec<Vec<i64>>,
    // Per-neighbor linear delta from the center's linear index; valid only
    // when the whole window lies inside the image (the fast path).
    neighbor_deltas: Vec<i64>,
    boundary: B,
    cursor: Vec<usize>,
    exhausted: bool,
}

impl<'a, T: Scalar, B: BoundaryCondition<T>> NeighborhoodIterator<'a, T, B> {
    /// Builds an iterator over `image` with the given per-dimension `radius`
    /// (ITK's `SizeType radius`) and `boundary` condition.
    ///
    /// Errors if `radius.len()` does not match `image.dimension()`, or if
    /// `T` is not `image`'s pixel type.
    pub fn new(image: &'a Image, radius: &[usize], boundary: B) -> Result<Self> {
        let dim = image.dimension();
        if radius.len() != dim {
            return Err(Error::RadiusMismatch { dimension: dim });
        }
        let view = image.scalar_view::<T>()?;

        let window_size: Vec<usize> = radius.iter().map(|&r| 2 * r + 1).collect();
        let num_neighbors: usize = window_size.iter().product();

        // Image strides, dimension 0 fastest (matches `Image::linear_index`).
        let mut strides = vec![0i64; dim];
        let mut accum = 1i64;
        for (d, stride) in strides.iter_mut().enumerate() {
            *stride = accum;
            accum *= image.size()[d] as i64;
        }

        // Per-neighbor ND offset and linear-delta tables, built once
        // (itkNeighborhood.hxx:41-67, `ComputeNeighborhoodOffsetTable`).
        let mut neighbor_offsets = Vec::with_capacity(num_neighbors);
        let mut neighbor_deltas = Vec::with_capacity(num_neighbors);
        let mut offset: Vec<i64> = radius.iter().map(|&r| -(r as i64)).collect();
        for _ in 0..num_neighbors {
            let delta: i64 = offset.iter().zip(&strides).map(|(&o, &s)| o * s).sum();
            neighbor_offsets.push(offset.clone());
            neighbor_deltas.push(delta);
            for d in 0..dim {
                offset[d] += 1;
                if offset[d] > radius[d] as i64 {
                    offset[d] = -(radius[d] as i64);
                } else {
                    break;
                }
            }
        }

        let exhausted = image.number_of_pixels() == 0;

        Ok(Self {
            view,
            radius: radius.to_vec().into(),
            window_size: window_size.into(),
            neighbor_offsets,
            neighbor_deltas,
            boundary,
            cursor: vec![0; dim],
            exhausted,
        })
    }

    /// Per-dimension neighborhood radius.
    pub fn radius(&self) -> &[usize] {
        &self.radius
    }

    /// Per-dimension window side length (`2 * radius + 1`).
    pub fn window_size(&self) -> &[usize] {
        &self.window_size
    }

    /// Number of pixels per yielded window.
    pub fn len(&self) -> usize {
        self.neighbor_offsets.len()
    }

    /// `true` if this iterator's window never contains any pixels (only
    /// possible for a zero-length image axis).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` if the whole window at `center` lies inside the image, so no
    /// boundary condition can ever be invoked for it
    /// (itkConstNeighborhoodIterator.hxx:22-46, `InBounds`).
    pub fn is_interior(&self, center: &[usize]) -> bool {
        center
            .iter()
            .enumerate()
            .all(|(d, &c)| c >= self.radius[d] && c + self.radius[d] < self.view.image().size()[d])
    }

    /// Fetches the window at `center` via direct offset arithmetic, with no
    /// per-neighbor bounds check. Only valid when `is_interior(center)`;
    /// exposed separately (rather than folded into [`Self::neighborhood_at`])
    /// so tests can prove it agrees with [`Self::neighborhood_at_checked`]
    /// everywhere it's valid.
    pub fn neighborhood_at_fast(&self, center: &[usize]) -> Neighborhood<T> {
        let mut values = Vec::with_capacity(self.len());
        self.push_values_fast(center, &mut values);
        self.wrap(values)
    }

    /// Fetches the window at `center`, checking each neighbor individually
    /// and falling back to the boundary condition for any that spill off the
    /// image (itkConstNeighborhoodIterator.h:194-209, `GetPixel`).
    pub fn neighborhood_at_checked(&self, center: &[usize]) -> Neighborhood<T> {
        let mut values = Vec::with_capacity(self.len());
        self.push_values_checked(center, &mut values);
        self.wrap(values)
    }

    fn wrap(&self, values: Vec<T>) -> Neighborhood<T> {
        Neighborhood {
            radius: Arc::clone(&self.radius),
            size: Arc::clone(&self.window_size),
            values,
        }
    }

    /// Appends the window at `center` by direct offset arithmetic. `out` must be
    /// empty. Only valid when `is_interior(center)`.
    fn push_values_fast(&self, center: &[usize], out: &mut Vec<T>) {
        debug_assert!(
            self.is_interior(center),
            "the fast path requires an interior center"
        );
        let center_linear = self.view.image().linear_index(center) as i64;
        let pixels = self.view.pixels();
        out.extend(
            self.neighbor_deltas
                .iter()
                .map(|&delta| pixels[(center_linear + delta) as usize]),
        );
    }

    /// Appends the window at `center`, consulting the boundary condition for any
    /// neighbor that spills off the image. `out` must be empty.
    fn push_values_checked(&self, center: &[usize], out: &mut Vec<T>) {
        let dim = self.view.image().dimension();
        let size = self.view.image().size();
        let mut nd = vec![0i64; dim];
        let mut idx = vec![0usize; dim];
        for offset in &self.neighbor_offsets {
            let mut inside = true;
            for d in 0..dim {
                let v = center[d] as i64 + offset[d];
                nd[d] = v;
                inside &= v >= 0 && (v as usize) < size[d];
            }
            let value = if inside {
                for (i, &v) in idx.iter_mut().zip(nd.iter()) {
                    *i = v as usize;
                }
                self.view.pixels()[self.view.image().linear_index(&idx)]
            } else {
                self.boundary.get_pixel(&nd, &self.view)
            };
            out.push(value);
        }
    }

    /// An empty window carrying this iterator's radius and window size, to be
    /// refilled at many centers by [`Self::refill`].
    ///
    /// The window's value buffer and its two `Arc`s are the per-pixel cost a
    /// sliding-window filter cannot afford to pay 16.7 M times over: reusing one
    /// buffer per worker task turns a per-voxel heap allocation and a pair of
    /// atomic refcount bumps on a shared cache line into one of each per task.
    pub fn window_buffer(&self) -> Neighborhood<T> {
        self.wrap(Vec::with_capacity(self.len()))
    }

    /// Refills `window` — which must come from [`Self::window_buffer`] on this
    /// same iterator — with the values at `center`, reusing its buffer.
    ///
    /// Leaves `window` exactly as [`Self::neighborhood_at`] would have built it.
    pub fn refill(&self, center: &[usize], window: &mut Neighborhood<T>) {
        window.values.clear();
        if self.is_interior(center) {
            self.push_values_fast(center, &mut window.values);
        } else {
            self.push_values_checked(center, &mut window.values);
        }
    }

    /// Fetches the window at `center`, using the interior fast path when the
    /// whole window fits inside the image and the boundary-checked path
    /// otherwise.
    pub fn neighborhood_at(&self, center: &[usize]) -> Neighborhood<T> {
        if self.is_interior(center) {
            self.neighborhood_at_fast(center)
        } else {
            self.neighborhood_at_checked(center)
        }
    }

    /// Applies `f` to every pixel's `(center, window)` in parallel, collecting
    /// the results in the same dimension-0-fastest order this type's [`Iterator`]
    /// walks — so `it.par_map(f)` and `it.map(|(c, nb)| f(&c, &nb)).collect()`
    /// agree element for element.
    ///
    /// This is the sliding-window seam for the whole port: the window at pixel
    /// `i` is a pure function of `i` and the input image, and result `i` lands in
    /// slot `i`, so no accumulator crosses pixels and nothing is re-associated.
    /// Whatever `f` computes *within* one window (a kernel dot product, a
    /// median selection, a structuring-element test) runs in `f`'s own sequential
    /// order, untouched. The output is therefore bit-identical to the sequential
    /// walk for any thread count — see [`crate::parallel`].
    ///
    /// The window and the center index are per-task scratch, refilled in place
    /// for each pixel ([`Self::window_buffer`], [`Self::refill`]) — `f` sees the
    /// same window contents it would have seen, but the walk does not allocate
    /// per pixel.
    pub fn par_map<R, F>(&self, f: F) -> Vec<R>
    where
        T: Send + Sync,
        B: Sync,
        R: Send,
        F: Fn(&[usize], &Neighborhood<T>) -> R + Sync + Send,
    {
        self.par_map_init(|| (), |(), center, window| f(center, window))
    }

    /// [`Self::par_map`] with a per-task scratch value of the caller's own, for a
    /// window function that needs working storage — a median's mutable copy of
    /// the window, say — and would otherwise allocate it per pixel.
    ///
    /// Same bit-for-bit guarantee, and the same contract as
    /// [`parallel::map_indexed_init`]: `scratch` is working storage that `f`
    /// fully overwrites per pixel, never an accumulator carried between pixels.
    pub fn par_map_init<R, S, I, F>(&self, init: I, f: F) -> Vec<R>
    where
        T: Send + Sync,
        B: Sync,
        R: Send,
        S: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, &[usize], &Neighborhood<T>) -> R + Sync + Send,
    {
        let size = self.view.image().size();
        let dim = size.len();
        parallel::map_indexed_init(
            self.view.image().number_of_pixels(),
            || (init(), vec![0usize; dim], self.window_buffer()),
            |(scratch, center, window), i| {
                // Unrank the linear index into an ND center, dimension 0 fastest
                // — the inverse of `Image::linear_index`, and the same order
                // `next` advances the cursor in.
                let mut rest = i;
                for (c, &s) in center.iter_mut().zip(size) {
                    *c = rest % s;
                    rest /= s;
                }
                self.refill(center, window);
                f(scratch, center, window)
            },
        )
    }
}

impl<'a, T: Scalar, B: BoundaryCondition<T>> Iterator for NeighborhoodIterator<'a, T, B> {
    type Item = (Vec<usize>, Neighborhood<T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let center = self.cursor.clone();
        let neighborhood = self.neighborhood_at(&center);

        // Advance the cursor, dimension 0 fastest, carrying into higher
        // dimensions on wrap (matches Image's storage order).
        let size = self.view.image().size();
        let mut carry = true;
        for (c, &s) in self.cursor.iter_mut().zip(size.iter()) {
            *c += 1;
            if *c < s {
                carry = false;
                break;
            }
            *c = 0;
        }
        if carry {
            self.exhausted = true;
        }

        Some((center, neighborhood))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{
        ConstantBoundaryCondition, PeriodicBoundaryCondition, ZeroFluxNeumannBoundaryCondition,
    };

    fn all_indices(size: &[usize]) -> Vec<Vec<usize>> {
        let mut result = vec![vec![]];
        for &s in size {
            let mut next = Vec::with_capacity(result.len() * s);
            for idx in &result {
                for v in 0..s {
                    let mut idx = idx.clone();
                    idx.push(v);
                    next.push(idx);
                }
            }
            result = next;
        }
        result
    }

    #[test]
    fn radius_length_mismatch_is_an_error() {
        let img = Image::from_vec(&[4, 3], vec![0u8; 12]).unwrap();
        let err = NeighborhoodIterator::<u8, _>::new(&img, &[1], ZeroFluxNeumannBoundaryCondition)
            .unwrap_err();
        assert_eq!(err, Error::RadiusMismatch { dimension: 2 });
    }

    #[test]
    fn walk_order_and_radius_zero_matches_source_pixels() {
        // With radius 0 every window is exactly the center pixel: the walk
        // itself must proceed dimension-0-fastest, matching Image's layout.
        let img = Image::from_vec(&[3, 2], vec![0u32, 1, 2, 3, 4, 5]).unwrap();
        let iter =
            NeighborhoodIterator::new(&img, &[0, 0], ZeroFluxNeumannBoundaryCondition).unwrap();
        let collected: Vec<(Vec<usize>, u32)> =
            iter.map(|(idx, nb)| (idx, nb.center_value())).collect();
        assert_eq!(
            collected,
            vec![
                (vec![0, 0], 0),
                (vec![1, 0], 1),
                (vec![2, 0], 2),
                (vec![0, 1], 3),
                (vec![1, 1], 4),
                (vec![2, 1], 5),
            ]
        );
    }

    #[test]
    fn fast_and_checked_paths_agree_for_every_interior_voxel_1d() {
        let size = [9usize];
        let n: usize = size.iter().product();
        let values: Vec<i32> = (0..n as i32).collect();
        let img = Image::from_vec(&size, values).unwrap();
        let iter =
            NeighborhoodIterator::new(&img, &[2], ConstantBoundaryCondition::new(-1i32)).unwrap();
        for idx in all_indices(&size) {
            if iter.is_interior(&idx) {
                assert_eq!(
                    iter.neighborhood_at_fast(&idx),
                    iter.neighborhood_at_checked(&idx),
                    "mismatch at {idx:?}"
                );
            }
        }
    }

    #[test]
    fn fast_and_checked_paths_agree_for_every_interior_voxel_2d() {
        let size = [6usize, 5];
        let n: usize = size.iter().product();
        let values: Vec<i32> = (0..n as i32).collect();
        let img = Image::from_vec(&size, values).unwrap();
        let iter =
            NeighborhoodIterator::<i32, _>::new(&img, &[1, 2], ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        for idx in all_indices(&size) {
            if iter.is_interior(&idx) {
                assert_eq!(
                    iter.neighborhood_at_fast(&idx),
                    iter.neighborhood_at_checked(&idx),
                    "mismatch at {idx:?}"
                );
            }
        }
    }

    #[test]
    fn fast_and_checked_paths_agree_for_every_interior_voxel_3d() {
        let size = [5usize, 4, 3];
        let n: usize = size.iter().product();
        let values: Vec<i32> = (0..n as i32).collect();
        let img = Image::from_vec(&size, values).unwrap();
        let iter = NeighborhoodIterator::<i32, _>::new(&img, &[1, 1, 1], PeriodicBoundaryCondition)
            .unwrap();
        for idx in all_indices(&size) {
            if iter.is_interior(&idx) {
                assert_eq!(
                    iter.neighborhood_at_fast(&idx),
                    iter.neighborhood_at_checked(&idx),
                    "mismatch at {idx:?}"
                );
            }
        }
    }

    #[test]
    fn zero_flux_neumann_window_1d_corners() {
        let img = Image::from_vec(&[5], vec![0u32, 1, 2, 3, 4]).unwrap();
        let iter =
            NeighborhoodIterator::<u32, _>::new(&img, &[1], ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        assert_eq!(iter.neighborhood_at(&[0]).values(), &[0, 0, 1]);
        assert_eq!(iter.neighborhood_at(&[4]).values(), &[3, 4, 4]);
    }

    #[test]
    fn constant_window_1d_corners() {
        let img = Image::from_vec(&[5], vec![0u32, 1, 2, 3, 4]).unwrap();
        let iter =
            NeighborhoodIterator::new(&img, &[1], ConstantBoundaryCondition::new(99u32)).unwrap();
        assert_eq!(iter.neighborhood_at(&[0]).values(), &[99, 0, 1]);
        assert_eq!(iter.neighborhood_at(&[4]).values(), &[3, 4, 99]);
    }

    #[test]
    fn periodic_window_1d_corners() {
        let img = Image::from_vec(&[5], vec![0u32, 1, 2, 3, 4]).unwrap();
        let iter =
            NeighborhoodIterator::<u32, _>::new(&img, &[1], PeriodicBoundaryCondition).unwrap();
        assert_eq!(iter.neighborhood_at(&[0]).values(), &[4, 0, 1]);
        assert_eq!(iter.neighborhood_at(&[4]).values(), &[3, 4, 0]);
    }

    fn image_2d() -> Image {
        let (w, h) = (4usize, 3usize);
        let mut data = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x + 10 * y) as u32;
            }
        }
        Image::from_vec(&[w, h], data).unwrap()
    }

    #[test]
    fn zero_flux_neumann_window_2d_corner_and_edge() {
        let img = image_2d();
        let iter =
            NeighborhoodIterator::<u32, _>::new(&img, &[1, 1], ZeroFluxNeumannBoundaryCondition)
                .unwrap();

        let corner = iter.neighborhood_at(&[0, 0]);
        assert_eq!(corner.get(&[-1, -1]), 0);
        assert_eq!(corner.get(&[1, 1]), 11);
        assert_eq!(corner.get(&[0, 0]), 0);

        // (1, 0): x is interior, y is on the top edge.
        let edge = iter.neighborhood_at(&[1, 0]);
        assert_eq!(edge.get(&[0, -1]), 1); // y clamps to 0 -> (1,0)=1
        assert_eq!(edge.get(&[0, 1]), 11); // in-bounds -> (1,1)=11
        assert_eq!(edge.get(&[-1, -1]), 0); // x=0,y clamps to 0 -> (0,0)=0
    }

    #[test]
    fn constant_window_2d_corner_and_edge() {
        let img = image_2d();
        let iter = NeighborhoodIterator::new(&img, &[1, 1], ConstantBoundaryCondition::new(99u32))
            .unwrap();

        let corner = iter.neighborhood_at(&[0, 0]);
        assert_eq!(corner.get(&[-1, -1]), 99);
        assert_eq!(corner.get(&[1, 1]), 11);

        let edge = iter.neighborhood_at(&[1, 0]);
        assert_eq!(edge.get(&[0, -1]), 99); // y out of bounds entirely
        assert_eq!(edge.get(&[0, 1]), 11);
        assert_eq!(edge.get(&[-1, -1]), 99);
    }

    #[test]
    fn periodic_window_2d_corner_and_edge() {
        let img = image_2d();
        let iter =
            NeighborhoodIterator::<u32, _>::new(&img, &[1, 1], PeriodicBoundaryCondition).unwrap();

        let corner = iter.neighborhood_at(&[0, 0]);
        assert_eq!(corner.get(&[-1, -1]), 23); // wraps to (3,2)
        assert_eq!(corner.get(&[-1, 0]), 3); // wraps x only -> (3,0)
        assert_eq!(corner.get(&[0, -1]), 20); // wraps y only -> (0,2)

        let edge = iter.neighborhood_at(&[1, 0]);
        assert_eq!(edge.get(&[0, -1]), 21); // y wraps to 2 -> (1,2)=21
        assert_eq!(edge.get(&[-1, -1]), 20); // x=0,y wraps to 2 -> (0,2)=20
    }

    fn image_3d() -> Image {
        let (w, h, d) = (3usize, 3usize, 3usize);
        let mut data = vec![0u32; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (x + 10 * y + 100 * z) as u32;
                }
            }
        }
        Image::from_vec(&[w, h, d], data).unwrap()
    }

    #[test]
    fn zero_flux_neumann_window_3d_corner_and_edge() {
        let img = image_3d();
        let iter =
            NeighborhoodIterator::<u32, _>::new(&img, &[1, 1, 1], ZeroFluxNeumannBoundaryCondition)
                .unwrap();

        let corner = iter.neighborhood_at(&[0, 0, 0]);
        assert_eq!(corner.get(&[-1, -1, -1]), 0);
        assert_eq!(corner.get(&[1, 1, 1]), 111);
        assert_eq!(corner.get(&[1, -1, -1]), 1);

        // (1, 0, 0): x is interior, y and z are on their low edge.
        let edge = iter.neighborhood_at(&[1, 0, 0]);
        assert_eq!(edge.get(&[0, -1, -1]), 1); // y,z clamp to 0 -> (1,0,0)=1
        assert_eq!(edge.get(&[-1, -1, -1]), 0); // x=0,y,z clamp to 0 -> (0,0,0)=0
        assert_eq!(edge.get(&[1, 1, 1]), 112); // in-bounds -> (2,1,1)=112
    }

    #[test]
    fn constant_window_3d_corner_and_edge() {
        let img = image_3d();
        let iter =
            NeighborhoodIterator::new(&img, &[1, 1, 1], ConstantBoundaryCondition::new(99u32))
                .unwrap();

        let corner = iter.neighborhood_at(&[0, 0, 0]);
        assert_eq!(corner.get(&[-1, -1, -1]), 99);
        assert_eq!(corner.get(&[1, -1, -1]), 99); // x in-bounds, y/z not.
        assert_eq!(corner.get(&[1, 1, 1]), 111);

        let edge = iter.neighborhood_at(&[1, 0, 0]);
        assert_eq!(edge.get(&[0, -1, -1]), 99);
        assert_eq!(edge.get(&[-1, 0, 0]), 0); // (0,0,0) fully in-bounds.
    }

    #[test]
    fn periodic_window_3d_corner_and_edge() {
        let img = image_3d();
        let iter = NeighborhoodIterator::<u32, _>::new(&img, &[1, 1, 1], PeriodicBoundaryCondition)
            .unwrap();

        let corner = iter.neighborhood_at(&[0, 0, 0]);
        assert_eq!(corner.get(&[-1, -1, -1]), 222); // wraps every axis -> (2,2,2)
        assert_eq!(corner.get(&[1, -1, -1]), 221); // x in-bounds, y,z wrap -> (1,2,2)

        let edge = iter.neighborhood_at(&[1, 0, 0]);
        assert_eq!(edge.get(&[0, -1, -1]), 221); // y,z wrap -> (1,2,2)
        assert_eq!(edge.get(&[-1, -1, -1]), 220); // x=0,y,z wrap -> (0,2,2)
    }
}
