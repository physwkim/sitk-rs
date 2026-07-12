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

/// A **zero-copy** view of the window at one center.
///
/// # Why this exists
///
/// [`Neighborhood`] *materializes* its window: it copies every neighbor's value
/// into a `Vec<T>`. Paid once per pixel, that copy — not the filter's kernel —
/// is what a sliding-window filter actually spends its time on. Measured at
/// 256³: walking every 3×3×3 window with a **no-op kernel** costs 679 ms of
/// `binary_dilate`'s 868 ms total (78%), and 469 ms of a 9-tap separable pass's
/// 556 ms (84%). The window copy *is* the op.
///
/// A `WindowView` copies nothing on the interior. It reads each neighbor
/// straight out of the image buffer through the linear-delta table
/// [`NeighborhoodIterator`] already builds at construction.
///
/// # One uniform accessor, no branch per access
///
/// Both paths are the same shape — a base slice, a base offset, and a delta
/// table — so [`Self::get`] has no `if` in it and the interior path stays a
/// straight indexed load:
///
/// - **interior**: `values` is the whole image buffer, `base` is the center's
///   linear index, `deltas` is the neighbor-delta table. Nothing is copied.
/// - **boundary**: the window is materialized into per-task scratch exactly as
///   the checked path always did (the boundary condition must be consulted, so
///   there is nothing to borrow), and then `values` is that scratch, `base` is
///   0, and `deltas` is the identity `0..len`.
///
/// Only pixels whose window overhangs the image take the second path — about
/// 2.3% of a 256³ volume at radius 1.
///
/// # Bit-parity
///
/// `get(j)` returns the value that `Neighborhood::values()[j]` held: the delta
/// table and the boundary fallback are the same ones the materializing path
/// used, so the values and their order are unchanged. A kernel reading this view
/// computes on the identical `f64` bits, in the identical sequence.
#[derive(Debug, Clone, Copy)]
pub struct WindowView<'a, T> {
    values: &'a [T],
    base: usize,
    deltas: &'a [i64],
    /// The window's extent along axis 0 — the length of each contiguous run.
    /// See [`Self::rows`].
    row_len: usize,
}

impl<'a, T: Scalar> WindowView<'a, T> {
    /// Number of pixels in the window.
    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    /// `true` if the window holds no pixels.
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// The `j`-th neighbor, in the same dimension-0-fastest order as
    /// [`Neighborhood::values`].
    #[inline]
    pub fn get(&self, j: usize) -> T {
        self.values[(self.base as i64 + self.deltas[j]) as usize]
    }

    /// The `j`-th neighbor, widened to `f64`.
    ///
    /// The widening a stencil would otherwise have paid for by materializing an
    /// entire `f64` copy of the image up front. `f32 -> f64` is lossless, so the
    /// value is the same one the `f64` copy held — see the module docs.
    #[inline]
    pub fn get_f64(&self, j: usize) -> f64 {
        self.get(j).as_f64()
    }

    /// The value at the center of the window.
    pub fn center(&self) -> T {
        self.get(self.len() / 2)
    }

    /// The window's **contiguous runs** along axis 0, in window order.
    ///
    /// Axis 0 has stride 1 in the image, so a window is not a scattered set of
    /// pixels — it is `len / row_len` runs of `row_len` adjacent ones. Handing a
    /// kernel each run as a `&[T]` lets its inner loop be a plain slice walk the
    /// optimizer can vectorize, instead of one indirect load per neighbor.
    ///
    /// This is what a materializing window bought by copying: contiguity. It
    /// turns out to be available without the copy.
    ///
    /// Concatenating the runs is exactly [`Self::iter`] — the same values in the
    /// same dimension-0-fastest order — so a kernel that sums over `rows()`
    /// accumulates in the identical sequence, and its result is bit-identical.
    pub fn rows(&self) -> impl Iterator<Item = &'a [T]> + 'a {
        let (values, base, row_len) = (self.values, self.base, self.row_len);
        self.deltas.chunks_exact(row_len).map(move |run| {
            // Within a run the deltas step by 1 (axis 0's stride), so the run is
            // `values[start .. start + row_len]` — on the borrowed interior path
            // and on the materialized boundary path alike.
            let start = (base as i64 + run[0]) as usize;
            &values[start..start + row_len]
        })
    }

    /// The neighbors in window order.
    ///
    /// One indirect load per neighbor. A kernel that reads the *whole* window
    /// should prefer [`Self::rows`], whose inner loop is a contiguous slice walk;
    /// this flat form stays the direct delta walk, because expressing it as
    /// `rows().flatten()` measurably pessimizes the kernels that read only a few
    /// of a window's values (a median's selection, a gradient's six taps).
    pub fn iter(&self) -> impl Iterator<Item = T> + 'a {
        let (values, base) = (self.values, self.base);
        self.deltas
            .iter()
            .map(move |&d| values[(base as i64 + d) as usize])
    }

    /// The neighbors in window order, widened to `f64`.
    ///
    /// This is the tap sequence a separable pass wants: for a window that is 1-D
    /// along the pass axis, `kernel.iter().zip(w.iter_f64())` *is* the tap
    /// product, in kernel order, with no per-tap index arithmetic.
    pub fn iter_f64(&self) -> impl Iterator<Item = f64> + 'a {
        self.iter().map(Scalar::as_f64)
    }
}

/// The ND center of a linear pixel index, walked incrementally.
///
/// Unranking `i` into an ND index costs one integer division *per dimension* —
/// and a division is tens of cycles, so at 16.7 M voxels × 3 axes it is a real
/// line on the bill. But a parallel task walks a **contiguous run** of `i`, and
/// the successor of an ND index is a carry-propagating increment: adds and
/// compares, no division. So the common step is O(1) cheap, and the full unrank
/// runs only when a task starts (or a caller jumps).
///
/// This is a pure memo — `seek(i)` returns the same center for the same `i`
/// whichever path it took — so it cannot affect determinism.
#[derive(Debug)]
struct Cursor {
    center: Vec<usize>,
    /// The `i` that `center` currently describes; `None` before the first seek.
    at: Option<usize>,
}

impl Cursor {
    fn new(size: &[usize]) -> Self {
        Self {
            center: vec![0; size.len()],
            at: None,
        }
    }

    /// The ND center of linear index `i`, dimension 0 fastest — the inverse of
    /// [`Image::linear_index`], and the same order the [`Iterator`] impl below
    /// advances in.
    fn seek(&mut self, i: usize, size: &[usize]) -> &[usize] {
        match self.at {
            // The hot path: the next pixel in this task's contiguous run.
            Some(prev) if i == prev + 1 => {
                for (c, &s) in self.center.iter_mut().zip(size) {
                    *c += 1;
                    if *c < s {
                        break;
                    }
                    *c = 0;
                }
            }
            _ => {
                let mut rest = i;
                for (c, &s) in self.center.iter_mut().zip(size) {
                    *c = rest % s;
                    rest /= s;
                }
            }
        }
        self.at = Some(i);
        &self.center
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
    // `0..num_neighbors`, the delta table a `WindowView` over a materialized
    // boundary window uses so that its accessor is the same one the interior
    // path uses. Built once here rather than per boundary pixel.
    identity_deltas: Vec<i64>,
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
            identity_deltas: (0..num_neighbors as i64).collect(),
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

    /// The [`WindowView`] at the pixel whose linear index is `linear` and whose
    /// ND center is `center` — borrowing the image directly when the window is
    /// interior, falling back to `scratch` when it overhangs.
    ///
    /// `linear` must be `center`'s linear index. The walk below already has it
    /// (it *is* the output slot), so passing it in avoids re-deriving it per
    /// voxel through `Image::linear_index`.
    ///
    /// `scratch` is per-task working storage; its previous contents are
    /// discarded. On the interior path it is not touched at all — that is the
    /// point.
    fn window_view<'s>(
        &'s self,
        center: &[usize],
        linear: usize,
        scratch: &'s mut Vec<T>,
    ) -> WindowView<'s, T> {
        debug_assert_eq!(linear, self.view.image().linear_index(center));
        let row_len = self.window_size[0];
        if self.is_interior(center) {
            WindowView {
                values: self.view.pixels(),
                base: linear,
                deltas: &self.neighbor_deltas,
                row_len,
            }
        } else {
            scratch.clear();
            self.push_values_checked(center, scratch);
            WindowView {
                values: scratch,
                base: 0,
                deltas: &self.identity_deltas,
                row_len,
            }
        }
    }

    /// Applies `f` to every pixel's `(center, `[`WindowView`]`)` in parallel,
    /// collecting the results in dimension-0-fastest order — the **zero-copy**
    /// counterpart of [`Self::par_map`].
    ///
    /// This is the sliding-window seam every stencil filter in the port should
    /// use. It has the identical bit-for-bit guarantee as [`Self::par_map`] (the
    /// window at pixel `i` is a pure function of `i` and the input image; result
    /// `i` lands in slot `i`; whatever `f` computes *within* a window runs in
    /// `f`'s own sequential order), but it does not copy the window into a
    /// `Vec<T>` per pixel — which is where the measured 78–84% of a
    /// sliding-window filter's runtime went. See [`WindowView`].
    pub fn par_map_window<R, F>(&self, f: F) -> Vec<R>
    where
        T: Send + Sync,
        B: Sync,
        R: Send,
        F: Fn(&[usize], WindowView<'_, T>) -> R + Sync + Send,
    {
        self.par_map_window_init(|| (), |(), center, window| f(center, window))
    }

    /// [`Self::par_map_window`] with a per-task scratch value of the caller's
    /// own, for a window function that needs working storage — a median's
    /// mutable copy of the window, say — and would otherwise allocate it per
    /// pixel.
    ///
    /// Same bit-for-bit guarantee, and the same contract as
    /// [`parallel::map_indexed_init`]: `scratch` is working storage that `f`
    /// fully overwrites per pixel, never an accumulator carried between pixels.
    pub fn par_map_window_init<R, S, I, F>(&self, init: I, f: F) -> Vec<R>
    where
        T: Send + Sync,
        B: Sync,
        R: Send,
        S: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, &[usize], WindowView<'_, T>) -> R + Sync + Send,
    {
        let size = self.view.image().size();
        parallel::map_indexed_init(
            self.view.image().number_of_pixels(),
            // Both scratches are allocated once per task, not per pixel — and on
            // the interior path the boundary buffer is never even written.
            || (init(), Cursor::new(size), Vec::with_capacity(self.len())),
            |(scratch, cursor, boundary), i| {
                let center = cursor.seek(i, size);
                let window = self.window_view(center, i, boundary);
                f(scratch, center, window)
            },
        )
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

    /// The whole contract of [`WindowView`]: it must deliver exactly the values
    /// [`Neighborhood`] materialized, in the same order, at *every* pixel — the
    /// interior ones it borrows and the boundary ones it copies alike. If this
    /// holds, no kernel can tell the two apart, and every pinned checksum in the
    /// port survives the switch.
    fn assert_window_view_matches_materialized<T, B>(img: &Image, radius: &[usize], boundary: B)
    where
        T: Scalar + std::fmt::Debug + PartialEq + Send + Sync,
        B: BoundaryCondition<T> + Sync,
    {
        let iter = NeighborhoodIterator::<T, B>::new(img, radius, boundary).unwrap();

        // What the materializing walk yields, pixel by pixel.
        let expected: Vec<Vec<T>> = iter
            .par_map(|_, nb| nb.values().to_vec())
            .into_iter()
            .collect();
        // What the zero-copy walk yields.
        let actual: Vec<Vec<T>> = iter.par_map_window(|_, w| w.iter().collect());

        assert_eq!(actual, expected, "window values diverged");

        // `get(j)` and `center()` must agree with the same source of truth.
        let centers: Vec<T> = iter.par_map_window(|_, w| w.center());
        let expected_centers: Vec<T> = iter.par_map(|_, nb| nb.center_value());
        assert_eq!(centers, expected_centers, "center value diverged");

        let indexed: Vec<Vec<T>> =
            iter.par_map_window(|_, w| (0..w.len()).map(|j| w.get(j)).collect());
        assert_eq!(indexed, expected, "get(j) diverged from iter()");
    }

    /// `Cursor`'s incremental step and its full unrank must agree at every
    /// index — if they ever diverged, a task's first pixel and its second would
    /// disagree about where they are, and the output would depend on where rayon
    /// happened to split.
    #[test]
    fn cursor_increment_agrees_with_a_full_unrank_at_every_index() {
        let size = [4usize, 3, 5];
        let n: usize = size.iter().product();

        let mut walking = Cursor::new(&size);
        for i in 0..n {
            // A fresh cursor always takes the unrank path for its first seek.
            let mut fresh = Cursor::new(&size);
            let expected = fresh.seek(i, &size).to_vec();
            assert_eq!(walking.seek(i, &size), &expected[..], "diverged at {i}");
        }

        // And a jump backwards (a task starting mid-volume) must re-unrank.
        let mut jumping = Cursor::new(&size);
        jumping.seek(n - 1, &size);
        let mut fresh = Cursor::new(&size);
        assert_eq!(jumping.seek(7, &size), fresh.seek(7, &size));
    }

    #[test]
    fn window_view_matches_materialized_3d_zero_flux() {
        assert_window_view_matches_materialized::<i32, _>(
            &Image::from_vec(&[5, 4, 3], (0..60).collect()).unwrap(),
            &[1, 1, 1],
            ZeroFluxNeumannBoundaryCondition,
        );
    }

    #[test]
    fn window_view_matches_materialized_3d_constant() {
        assert_window_view_matches_materialized::<i32, _>(
            &Image::from_vec(&[5, 4, 3], (0..60).collect()).unwrap(),
            &[1, 2, 1],
            ConstantBoundaryCondition::new(-7i32),
        );
    }

    #[test]
    fn window_view_matches_materialized_3d_periodic() {
        assert_window_view_matches_materialized::<i32, _>(
            &Image::from_vec(&[5, 4, 3], (0..60).collect()).unwrap(),
            &[2, 1, 1],
            PeriodicBoundaryCondition,
        );
    }

    /// A radius wider than the image: *every* pixel takes the boundary path, so
    /// this pins the materialized fallback rather than the borrowed fast path.
    #[test]
    fn window_view_matches_materialized_when_no_pixel_is_interior() {
        assert_window_view_matches_materialized::<u8, _>(
            &Image::from_vec(&[3, 3], (0..9u8).collect()).unwrap(),
            &[4, 4],
            ZeroFluxNeumannBoundaryCondition,
        );
    }

    /// The separable case the port's Gaussian passes run: a window that is 1-D
    /// along one axis. `iter_f64()` must be the tap sequence in kernel order, so
    /// that `kernel.zip(w.iter_f64())` replaces the per-tap `Neighborhood::get`
    /// index recompute *without* changing a single value or their order.
    #[test]
    fn window_view_taps_match_the_per_tap_neighborhood_get_on_a_separable_axis() {
        let img = Image::from_vec(&[7usize, 5, 4], (0..140).map(f64::from).collect()).unwrap();
        let half = 2usize;
        let kernel: Vec<f64> = vec![0.1, 0.2, 0.4, 0.2, 0.1];

        for axis in 0..3 {
            let mut radius = vec![0usize; 3];
            radius[axis] = half;
            let iter = NeighborhoodIterator::<f64, _>::new(
                &img,
                &radius,
                ZeroFluxNeumannBoundaryCondition,
            )
            .unwrap();

            // The old inner loop: re-derive an ND index for every tap.
            let by_get: Vec<f64> = iter.par_map_init(
                || vec![0i64; 3],
                |off, _, nb| {
                    kernel
                        .iter()
                        .enumerate()
                        .map(|(k, &c)| {
                            off[axis] = k as i64 - half as i64;
                            c * nb.get(off)
                        })
                        .sum()
                },
            );
            // The new one: the window IS the tap array, in order.
            let by_zip: Vec<f64> = iter.par_map_window(|_, w| {
                kernel
                    .iter()
                    .zip(w.iter_f64())
                    .map(|(&c, v)| c * v)
                    .sum::<f64>()
            });
            assert_eq!(by_zip, by_get, "tap sum diverged on axis {axis}");
        }
    }

    /// The widening a stencil used to pay for by materializing an `f64` copy of
    /// the whole image: reading the native type and widening per access must
    /// give the identical `f64` bits.
    #[test]
    fn window_view_get_f64_equals_widening_the_image_first() {
        let native =
            Image::from_vec(&[6usize, 5], (0..30).map(|i| i as f32 * 0.3).collect()).unwrap();
        let widened = Image::from_vec(native.size(), native.to_f64_vec().unwrap()).unwrap();

        let from_native =
            NeighborhoodIterator::<f32, _>::new(&native, &[1, 1], ZeroFluxNeumannBoundaryCondition)
                .unwrap()
                .par_map_window(|_, w| w.iter_f64().sum::<f64>());

        let from_widened = NeighborhoodIterator::<f64, _>::new(
            &widened,
            &[1, 1],
            ZeroFluxNeumannBoundaryCondition,
        )
        .unwrap()
        .par_map_window(|_, w| w.iter_f64().sum::<f64>());

        assert_eq!(from_native, from_widened);
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
