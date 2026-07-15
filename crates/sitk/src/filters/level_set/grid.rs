//! Index arithmetic shared by the level-set function terms and the
//! sparse-field solver.
//!
//! ITK reaches neighbors through `itk::NeighborhoodIterator`, whose
//! out-of-buffer reads go through a `ZeroFluxNeumannBoundaryCondition` (the
//! index is clamped into the image). [`Grid::clamped_index`] is that boundary
//! condition; [`Grid::in_bounds_index`] is the `bool` out-parameter of
//! `NeighborhoodIterator::SetPixel(i, v, bounds_status)`, which declines to
//! write outside the image.

/// A raster grid: sizes per axis plus the matching linear-index strides
/// (`stride[0] == 1`, first index fastest — the same layout as
/// `crate::core::Image::linear_index`).
pub(super) struct Grid {
    size: Vec<usize>,
    strides: Vec<usize>,
}

impl Grid {
    pub(super) fn new(size: &[usize]) -> Self {
        let mut strides = Vec::with_capacity(size.len());
        let mut stride = 1usize;
        for &s in size {
            strides.push(stride);
            stride *= s;
        }
        Grid {
            size: size.to_vec(),
            strides,
        }
    }

    pub(super) fn dim(&self) -> usize {
        self.size.len()
    }

    pub(super) fn size(&self) -> &[usize] {
        &self.size
    }

    pub(super) fn number_of_pixels(&self) -> usize {
        self.size.iter().product()
    }

    /// The multi-index of a linear buffer offset.
    pub(super) fn coord(&self, linear: usize) -> Vec<i64> {
        (0..self.dim())
            .map(|d| ((linear / self.strides[d]) % self.size[d]) as i64)
            .collect()
    }

    /// `ZeroFluxNeumannBoundaryCondition`: clamp each component into
    /// `[0, size[d] - 1]` and return the linear offset.
    pub(super) fn clamped_index(&self, coord: &[i64]) -> usize {
        let mut linear = 0usize;
        for ((&c, &size), &stride) in coord.iter().zip(&self.size).zip(&self.strides) {
            linear += c.clamp(0, size as i64 - 1) as usize * stride;
        }
        linear
    }

    /// The linear offset of `coord`, or `None` when any component falls
    /// outside the image.
    pub(super) fn in_bounds_index(&self, coord: &[i64]) -> Option<usize> {
        let mut linear = 0usize;
        for ((&c, &size), &stride) in coord.iter().zip(&self.size).zip(&self.strides) {
            if c < 0 || c >= size as i64 {
                return None;
            }
            linear += c as usize * stride;
        }
        Some(linear)
    }
}

/// `SparseFieldCityBlockNeighborList`'s `2 * dim` axis-aligned offsets, as
/// `(axis, delta)` pairs, in ITK's construction order
/// (itkSparseFieldLevelSetImageFilter.hxx:50-62): every negative-direction
/// offset from the *last* axis down to the first, then every
/// positive-direction offset from the first axis up.
///
/// The order is load-bearing: `UpdateActiveLayerValues` and
/// `ProcessStatusList` short-circuit on the first neighbor that matches a
/// status, so a different traversal order can pick a different neighbor.
pub(super) fn city_block_neighbors(dim: usize) -> Vec<(usize, i64)> {
    let mut neighbors = Vec::with_capacity(2 * dim);
    for i in 0..dim {
        neighbors.push((dim - 1 - i, -1));
    }
    for d in 0..dim {
        neighbors.push((d, 1));
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_round_trips_through_the_linear_index() {
        let grid = Grid::new(&[4, 3, 2]);
        for linear in 0..grid.number_of_pixels() {
            let coord = grid.coord(linear);
            assert_eq!(grid.in_bounds_index(&coord), Some(linear));
        }
    }

    #[test]
    fn clamped_index_applies_zero_flux_neumann_on_each_face() {
        let grid = Grid::new(&[4, 3]);
        assert_eq!(grid.clamped_index(&[-1, 1]), grid.clamped_index(&[0, 1]));
        assert_eq!(grid.clamped_index(&[4, 1]), grid.clamped_index(&[3, 1]));
        assert_eq!(grid.clamped_index(&[2, -5]), grid.clamped_index(&[2, 0]));
        assert_eq!(grid.clamped_index(&[2, 9]), grid.clamped_index(&[2, 2]));
    }

    #[test]
    fn in_bounds_index_rejects_every_out_of_range_component() {
        let grid = Grid::new(&[4, 3]);
        assert_eq!(grid.in_bounds_index(&[0, 0]), Some(0));
        assert_eq!(grid.in_bounds_index(&[3, 2]), Some(11));
        assert_eq!(grid.in_bounds_index(&[-1, 0]), None);
        assert_eq!(grid.in_bounds_index(&[4, 0]), None);
        assert_eq!(grid.in_bounds_index(&[0, -1]), None);
        assert_eq!(grid.in_bounds_index(&[0, 3]), None);
    }

    #[test]
    fn city_block_neighbors_follow_itks_construction_order() {
        assert_eq!(
            city_block_neighbors(3),
            vec![(2, -1), (1, -1), (0, -1), (0, 1), (1, 1), (2, 1)]
        );
    }
}
