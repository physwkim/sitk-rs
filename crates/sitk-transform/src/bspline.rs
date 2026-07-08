//! Cubic B-spline free-form deformation transform (`itk::BSplineTransform`).
//!
//! A [`BSplineTransform`] warps space by a smooth deformation field defined on a
//! regular grid of **control points**. Each control point `j` carries a
//! `dimension`-vector coefficient `δⱼ`; the displacement at an arbitrary point
//! `x` is the cubic-B-spline interpolation of the surrounding coefficients, and
//! the mapped point is `x + displacement(x)`. Coefficients of zero give the
//! identity. This is the transform for **deformable / non-rigid registration**,
//! where global rigid/affine/similarity transforms cannot capture local warping.
//!
//! ```text
//! displacement(x) = Σ_j  δⱼ · Π_d B₃( index_d(x) − j_d )
//! T(x) = x + displacement(x)
//! ```
//!
//! where `index(x)` is `x` in continuous-index coordinates of the control-point
//! grid, `B₃` is the cubic (order-3) B-spline basis, and the product runs over
//! the `(order+1)^dimension = 4^dimension` control points whose support covers
//! `x`. Outside the grid's valid region the displacement is zero (`T(x) = x`).
//!
//! # Grid geometry (matches ITK)
//!
//! The control-point grid is derived from a **transform domain** — an origin,
//! per-axis physical dimensions, a direction matrix, and a **mesh size** (the
//! number of B-spline polynomial patches per axis) — exactly as
//! `itk::BSplineTransform::SetTransformDomain*` / `SetTransformDomainMeshSize`:
//!
//! ```text
//! gridSize[i]    = meshSize[i] + splineOrder          (control points per axis)
//! gridSpacing[i] = physicalDimensions[i] / meshSize[i]
//! gridOrigin     = domainOrigin + D · (−½·gridSpacing·(splineOrder−1))
//! gridDirection  = D
//! ```
//!
//! The `−½·(splineOrder−1)` shift pads the grid with a border of control points
//! (one on each side for the cubic order) so the B-spline support of every point
//! in the domain lies inside the grid.
//!
//! # Parameters
//!
//! The parameter vector is the control-point coefficients: for each spatial
//! dimension `d`, the grid of `d`-th displacement components flattened in image
//! raster order (first axis fastest), the `dimension` grids concatenated —
//! `params[d · numberPerDimension + flatGridIndex]`. This matches ITK's flat
//! parameter layout (`SpaceDimension` coefficient images concatenated).
//!
//! # Jacobian
//!
//! `∂T_d / ∂(coefficient for dimension d' at control point j)` is the B-spline
//! weight of control point `j` when `d == d'`, and zero otherwise. So the
//! Jacobian is sparse: only the `4^dimension` in-support control points (times
//! the matching output dimension) are non-zero. This implementation returns it
//! through the dense [`ParametricTransform::jacobian_wrt_parameters`] contract (a
//! `dimension × numberOfParameters` row-major matrix that is mostly zero), so the
//! transform drops into the existing metric/optimizer unchanged. A sparse-Jacobian
//! fast path (and ITK's `HasLocalSupport` metric branch) is a later optimization.

use sitk_core::{Image, matrix};

use crate::error::{Result, TransformError};
use crate::interpolator::physical_to_index_matrix;
use crate::transform::{ParametricTransform, Transform};

/// The B-spline order. Fixed at 3 (cubic), ITK's default and the only order this
/// port implements; the Parzen/interpolation kernels elsewhere are cubic too.
const SPLINE_ORDER: usize = 3;

/// The ⅛-voxel amount by which `itk::BSplineTransformInitializer` expands the
/// image bounding box on every side, so the resulting transform domain strictly
/// contains every voxel (`itk`'s `BSplineTransformDomainEpsilon = 1 / (1 << 3)`).
const BSPLINE_TRANSFORM_DOMAIN_EPSILON: f64 = 1.0 / 8.0;

/// The cubic (order-3) B-spline basis `B₃(u)`. Same basis as
/// `itk::BSplineKernelFunction<3>`; here it is the interpolation kernel of the
/// deformation field rather than a Parzen window.
fn cubic_bspline(u: f64) -> f64 {
    let a = u.abs();
    if a < 1.0 {
        let sq = a * a;
        (4.0 - 6.0 * sq + 3.0 * sq * a) / 6.0
    } else if a < 2.0 {
        let sq = a * a;
        (8.0 - 12.0 * a + 6.0 * sq - sq * a) / 6.0
    } else {
        0.0
    }
}

/// Compute a B-spline transform domain — `(origin, per-axis physical dimensions,
/// row-major direction)` — from `image`'s geometry, porting
/// `itk::BSplineTransformInitializer::InitializeTransform`.
///
/// The image's `2^dim` corners (each axis expanded outward by
/// [`BSPLINE_TRANSFORM_DOMAIN_EPSILON`]) are mapped to physical space. The corner
/// nearest the bounding-box minimum becomes the domain origin. Each physical axis
/// is then matched — greedily, each edge used once — to the adjacent origin-corner
/// edge whose direction is most aligned with it (smallest angle); that edge's
/// length is the axis's physical dimension and its unit vector is the direction
/// column. This recovers the domain of an arbitrarily oriented (rotated) image.
fn bspline_initializer_domain(image: &Image) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dim = image.dimension();
    let size = image.size();
    let eps = BSPLINE_TRANSFORM_DOMAIN_EPSILON;

    // The 2^dim corners in physical space. Bit `i` of corner `d` selects the low
    // (continuous index −0.5−ε) or high (index size−0.5+ε) extreme along axis `i`.
    let n_corners = 1usize << dim;
    let corners: Vec<Vec<f64>> = (0..n_corners)
        .map(|d| {
            let index: Vec<f64> = (0..dim)
                .map(|i| {
                    let lo = -0.5 - eps;
                    if (d >> i) & 1 == 1 {
                        lo + size[i] as f64 + 2.0 * eps
                    } else {
                        lo
                    }
                })
                .collect();
            image.continuous_index_to_physical_point(&index)
        })
        .collect();

    // Bounding-box minimum (component-wise), then the corner closest to it.
    let mut bbox_min = corners[0].clone();
    for c in &corners[1..] {
        for i in 0..dim {
            bbox_min[i] = bbox_min[i].min(c[i]);
        }
    }
    let mut origin_id = 0usize;
    let mut min_distance = f64::INFINITY;
    for (d, c) in corners.iter().enumerate() {
        let dist: f64 = (0..dim).map(|i| (c[i] - bbox_min[i]).powi(2)).sum();
        if dist < min_distance {
            min_distance = dist;
            origin_id = d;
        }
    }
    let origin = corners[origin_id].clone();

    // Edge vector from the origin corner to the corner one bit away along axis `i`.
    let edge = |opposite: usize| -> Vec<f64> {
        (0..dim).map(|k| corners[opposite][k] - origin[k]).collect()
    };

    // Match each physical axis to the most-aligned unused adjacent edge, then read
    // off that axis's physical dimension (edge length) and direction column.
    let mut physical_dimensions = vec![0.0; dim];
    let mut direction = vec![0.0; dim * dim]; // row-major, column `d` per axis
    let mut min_corner_id = vec![usize::MAX; dim];
    for d in 0..dim {
        let mut min_angle = f64::INFINITY;
        for i in 0..dim {
            let opposite = (1usize << i) ^ origin_id;
            let v = edge(opposite);
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Angle to physical axis `d` = acos(v_d / ‖v‖) (e_d is the unit axis).
            let theta = (v[d] / norm).clamp(-1.0, 1.0).acos();
            if theta < min_angle && !min_corner_id[..d].contains(&opposite) {
                min_angle = theta;
                min_corner_id[d] = opposite;
            }
        }
        let v = edge(min_corner_id[d]);
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        physical_dimensions[d] = norm;
        for i in 0..dim {
            direction[i * dim + d] = v[i] / norm;
        }
    }

    (origin, physical_dimensions, direction)
}

/// A cubic B-spline free-form deformation transform. See the [module
/// docs](self).
#[derive(Clone, Debug)]
pub struct BSplineTransform {
    dim: usize,
    /// Control points per axis (`meshSize + splineOrder`).
    grid_size: Vec<usize>,
    /// Physical origin of control point `(0,…,0)`.
    grid_origin: Vec<f64>,
    /// Physical spacing between adjacent control points, per axis.
    grid_spacing: Vec<f64>,
    /// `diag(1/gridSpacing) · gridDirection⁻¹`, row-major `dim × dim`: maps a
    /// physical displacement from `grid_origin` to a continuous grid index.
    phys_to_index: Vec<f64>,
    /// Raster strides of the control-point grid (first axis fastest).
    grid_stride: Vec<usize>,
    /// Number of control points (`Π grid_size`) = parameters per dimension.
    num_per_dim: usize,
    /// Coefficients: `dim · num_per_dim`, layout `[dim0 grid][dim1 grid]…`.
    coefficients: Vec<f64>,
}

impl BSplineTransform {
    /// Build a cubic B-spline transform over a transform domain, mirroring
    /// `itk::BSplineTransform` configured via `SetTransformDomainOrigin` /
    /// `SetTransformDomainPhysicalDimensions` / `SetTransformDomainDirection` /
    /// `SetTransformDomainMeshSize`. All coefficients start at zero (identity).
    ///
    /// `domain_direction` is row-major `dim × dim`. Fails if any argument's
    /// length is inconsistent with `dim`, a mesh size is zero, or the direction
    /// matrix is singular.
    pub fn new(
        dim: usize,
        domain_origin: &[f64],
        domain_physical_dimensions: &[f64],
        domain_direction: &[f64],
        mesh_size: &[usize],
    ) -> Result<Self> {
        if domain_origin.len() != dim
            || domain_physical_dimensions.len() != dim
            || domain_direction.len() != dim * dim
            || mesh_size.len() != dim
            || mesh_size.contains(&0)
        {
            return Err(TransformError::InvalidBSplineDomain);
        }

        let grid_size: Vec<usize> = mesh_size.iter().map(|&m| m + SPLINE_ORDER).collect();
        let grid_spacing: Vec<f64> = (0..dim)
            .map(|i| domain_physical_dimensions[i] / mesh_size[i] as f64)
            .collect();

        // gridOrigin = domainOrigin + D · (−½·gridSpacing·(splineOrder−1)).
        let shift: Vec<f64> = (0..dim)
            .map(|i| -0.5 * grid_spacing[i] * (SPLINE_ORDER as f64 - 1.0))
            .collect();
        let rotated = matrix::mat_vec(domain_direction, &shift, dim);
        let grid_origin: Vec<f64> = (0..dim).map(|i| domain_origin[i] + rotated[i]).collect();

        let phys_to_index = physical_to_index_matrix(domain_direction, &grid_spacing, dim)
            .ok_or(TransformError::SingularDirection)?;

        // Raster strides, first axis fastest.
        let mut grid_stride = vec![1usize; dim];
        for i in 1..dim {
            grid_stride[i] = grid_stride[i - 1] * grid_size[i - 1];
        }
        let num_per_dim: usize = grid_size.iter().product();

        Ok(Self {
            dim,
            grid_size,
            grid_origin,
            grid_spacing,
            phys_to_index,
            grid_stride,
            num_per_dim,
            coefficients: vec![0.0; dim * num_per_dim],
        })
    }

    /// Build a cubic B-spline transform whose domain covers `image` — origin,
    /// direction, and physical dimensions `size·spacing` taken from the image —
    /// with the given per-axis `mesh_size`. This full-extent domain places every
    /// voxel centre (`index 0..size`) inside the valid region.
    ///
    /// This is a convenience domain, **not** a port of
    /// `itk::BSplineTransformInitializer` (whose corner/bounding-box domain adds
    /// a ⅛-voxel epsilon and derives the direction from image corners); use
    /// [`from_image_initializer`](Self::from_image_initializer) for the faithful
    /// initializer.
    pub fn from_image_domain(image: &Image, mesh_size: &[usize]) -> Result<Self> {
        let dim = image.dimension();
        let physical_dimensions: Vec<f64> = (0..dim)
            .map(|i| image.size()[i] as f64 * image.spacing()[i])
            .collect();
        Self::new(
            dim,
            image.origin(),
            &physical_dimensions,
            image.direction(),
            mesh_size,
        )
    }

    /// Build a cubic B-spline transform whose transform domain is initialized from
    /// `image`'s geometry, porting `itk::BSplineTransformInitializer` (SimpleITK
    /// `BSplineTransformInitializerFilter`) with the given per-axis `mesh_size`.
    ///
    /// Unlike [`from_image_domain`](Self::from_image_domain), which takes the
    /// domain as exactly `size·spacing` at the image origin, this maps the image's
    /// `2^dim` corners — each expanded outward by a ⅛-voxel epsilon — into physical
    /// space, places the origin at the corner nearest the bounding-box minimum, and
    /// derives the physical dimensions and direction from the origin corner's edges
    /// (so it handles an arbitrarily rotated direction matrix). The epsilon margin
    /// makes the domain strictly contain every voxel — even each voxel's own
    /// corners — matching ITK's initializer. All coefficients start at zero
    /// (identity).
    ///
    /// Fails if `mesh_size`'s length disagrees with the image dimension, a mesh
    /// size is zero, or the image direction matrix is singular.
    pub fn from_image_initializer(image: &Image, mesh_size: &[usize]) -> Result<Self> {
        let (origin, physical_dimensions, direction) = bspline_initializer_domain(image);
        Self::new(
            image.dimension(),
            &origin,
            &physical_dimensions,
            &direction,
            mesh_size,
        )
    }

    /// Control points per axis (`meshSize + splineOrder`).
    pub fn grid_size(&self) -> &[usize] {
        &self.grid_size
    }

    /// Physical spacing between adjacent control points, per axis.
    pub fn grid_spacing(&self) -> &[f64] {
        &self.grid_spacing
    }

    /// Physical origin of control point `(0,…,0)`.
    pub fn grid_origin(&self) -> &[f64] {
        &self.grid_origin
    }

    /// Number of control points (`Π grid_size`) = parameters per dimension.
    pub fn number_of_parameters_per_dimension(&self) -> usize {
        self.num_per_dim
    }

    /// Continuous grid index of physical point `p`:
    /// `phys_to_index · (p − grid_origin)`.
    fn continuous_index(&self, p: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        (0..dim)
            .map(|r| {
                (0..dim)
                    .map(|c| self.phys_to_index[r * dim + c] * (p[c] - self.grid_origin[c]))
                    .sum()
            })
            .collect()
    }

    /// Whether a continuous grid index lies in the valid region — the interior
    /// where the full cubic support fits inside the grid — snapping the far
    /// boundary inward as ITK's `InsideValidRegion` does. `index` is mutated by
    /// the snap. For the cubic order the valid interval per axis is
    /// `[1, gridSize − 2)`.
    fn inside_valid_region(&self, index: &mut [f64]) -> bool {
        let min_limit = 0.5 * (SPLINE_ORDER as f64 - 1.0);
        for (idx, &gsize) in index.iter_mut().zip(self.grid_size.iter()) {
            let max_limit = gsize as f64 - 0.5 * (SPLINE_ORDER as f64 - 1.0) - 1.0;
            // Epsilon approximation of ITK's ULP-exact boundary snap: a point
            // landing essentially on the far limit is nudged just inside so its
            // support region still fits, rather than being rejected.
            let ulp = 4.0 * f64::EPSILON * max_limit.abs().max(1.0);
            if (*idx - max_limit).abs() <= ulp {
                *idx = max_limit - 6.0 * f64::EPSILON * max_limit.abs().max(1.0);
            } else if *idx >= max_limit || *idx < min_limit {
                return false;
            }
        }
        true
    }

    /// The `(order+1)^dim` cubic-B-spline interpolation weights (support order:
    /// first axis fastest) and the per-axis support start index, for a continuous
    /// grid index inside the valid region. Mirrors
    /// `itk::BSplineInterpolationWeightFunction::Evaluate`.
    fn evaluate_weights(&self, index: &[f64]) -> (Vec<f64>, Vec<isize>) {
        let dim = self.dim;
        let taps = SPLINE_ORDER + 1;

        let mut start = vec![0isize; dim];
        // weights_1d[j][k] = B₃ of the k-th tap along axis j.
        let mut weights_1d = vec![[0.0f64; SPLINE_ORDER + 1]; dim];
        for ((&idx, s), w1) in index
            .iter()
            .zip(start.iter_mut())
            .zip(weights_1d.iter_mut())
        {
            *s = (idx + 0.5 - SPLINE_ORDER as f64 / 2.0).floor() as isize;
            let mut x = idx - *s as f64;
            for w in w1.iter_mut() {
                *w = cubic_bspline(x);
                x -= 1.0;
            }
        }

        let num_weights = taps.pow(dim as u32);
        let mut weights = vec![0.0f64; num_weights];
        for (k, w) in weights.iter_mut().enumerate() {
            let mut prod = 1.0;
            let mut rem = k;
            for wj in weights_1d.iter() {
                prod *= wj[rem % taps];
                rem /= taps;
            }
            *w = prod;
        }
        (weights, start)
    }
}

impl Transform for BSplineTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut index = self.continuous_index(point);
        if !self.inside_valid_region(&mut index) {
            return point.to_vec(); // outside grid ⇒ zero displacement
        }

        let (weights, start) = self.evaluate_weights(&index);
        let taps = SPLINE_ORDER + 1;
        let mut displacement = vec![0.0f64; dim];
        for (k, &w) in weights.iter().enumerate() {
            // Flatten support point k to its control-point grid raster index.
            let mut rem = k;
            let mut flat = 0usize;
            for (&s, &stride) in start.iter().zip(self.grid_stride.iter()) {
                let off = rem % taps;
                rem /= taps;
                flat += (s as usize + off) * stride;
            }
            for (d, disp) in displacement.iter_mut().enumerate() {
                *disp += w * self.coefficients[d * self.num_per_dim + flat];
            }
        }

        (0..dim).map(|d| point[d] + displacement[d]).collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

impl ParametricTransform for BSplineTransform {
    fn number_of_parameters(&self) -> usize {
        self.dim * self.num_per_dim
    }

    fn parameters(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(
            params.len(),
            self.coefficients.len(),
            "B-spline parameter vector length mismatch"
        );
        self.coefficients.copy_from_slice(params);
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let nparams = self.number_of_parameters();
        let mut jac = vec![0.0f64; dim * nparams];

        let mut index = self.continuous_index(point);
        if !self.inside_valid_region(&mut index) {
            return jac; // outside grid ⇒ zero Jacobian
        }

        let (weights, start) = self.evaluate_weights(&index);
        let taps = SPLINE_ORDER + 1;
        for (k, &w) in weights.iter().enumerate() {
            let mut rem = k;
            let mut flat = 0usize;
            for (&s, &stride) in start.iter().zip(self.grid_stride.iter()) {
                let off = rem % taps;
                rem /= taps;
                flat += (s as usize + off) * stride;
            }
            // ∂T_d/∂(coeff d at control point flat) = weight; other outputs 0.
            for d in 0..dim {
                jac[d * nparams + d * self.num_per_dim + flat] = w;
            }
        }
        jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit-spacing, identity-direction 2-D image of the given size.
    fn image(w: usize, h: usize) -> Image {
        Image::from_vec(&[w, h], vec![0.0; w * h]).unwrap()
    }

    /// A 2-D image with the given size, spacing, origin, and row-major direction.
    fn image_geom(
        size: [usize; 2],
        spacing: [f64; 2],
        origin: [f64; 2],
        direction: [f64; 4],
    ) -> Image {
        let mut img = Image::from_vec(&size, vec![0.0; size[0] * size[1]]).unwrap();
        img.set_spacing(&spacing).unwrap();
        img.set_origin(&origin).unwrap();
        img.set_direction(&direction).unwrap();
        img
    }

    #[test]
    fn cubic_weights_are_a_partition_of_unity() {
        // The 16 tensor weights sum to 1 for any interior continuous index.
        let t = BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[4, 4])
            .unwrap();
        for &idx in &[[2.3, 3.7], [1.0, 4.9], [3.5, 2.1]] {
            let (weights, _) = t.evaluate_weights(&idx);
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "idx {idx:?}: sum {sum}");
        }
    }

    #[test]
    fn grid_geometry_matches_itk() {
        // meshSize 4 over physical dimensions 8 (unit-origin, identity dir):
        // gridSize = 4+3 = 7, gridSpacing = 8/4 = 2, gridOrigin = 0 − spacing = −2.
        let t = BSplineTransform::new(2, &[0.0, 0.0], &[8.0, 8.0], &matrix::identity(2), &[4, 4])
            .unwrap();
        assert_eq!(t.grid_size(), &[7, 7]);
        assert_eq!(t.grid_spacing(), &[2.0, 2.0]);
        assert_eq!(t.grid_origin(), &[-2.0, -2.0]);
        assert_eq!(t.number_of_parameters(), 2 * 7 * 7);
    }

    #[test]
    fn zero_coefficients_are_identity() {
        let t = BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[5, 5])
            .unwrap();
        for p in &[[5.0, 5.0], [2.3, 7.1], [0.0, 0.0], [9.9, 0.1]] {
            let out = t.transform_point(p);
            assert!(
                (out[0] - p[0]).abs() < 1e-12 && (out[1] - p[1]).abs() < 1e-12,
                "identity failed at {p:?}: got {out:?}"
            );
        }
    }

    #[test]
    fn constant_coefficient_field_is_a_uniform_translation() {
        // A constant coefficient field of value c in dimension d gives a uniform
        // displacement c in that dimension for every interior point, because the
        // B-spline weights sum to 1.
        let mut t =
            BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[5, 5])
                .unwrap();
        let per = t.number_of_parameters_per_dimension();
        let (cx, cy) = (1.5, -0.75);
        let mut params = vec![0.0; t.number_of_parameters()];
        params[..per].fill(cx); // dimension-0 coefficients
        params[per..2 * per].fill(cy); // dimension-1 coefficients
        t.set_parameters(&params);

        for p in &[[5.0, 5.0], [3.2, 6.8], [1.1, 2.2]] {
            let out = t.transform_point(p);
            assert!(
                (out[0] - (p[0] + cx)).abs() < 1e-9 && (out[1] - (p[1] + cy)).abs() < 1e-9,
                "at {p:?}: got {out:?}, expected {:?}",
                [p[0] + cx, p[1] + cy]
            );
        }
    }

    #[test]
    fn points_outside_the_valid_region_are_unmapped() {
        // A far-outside point gets zero displacement even with non-zero
        // coefficients.
        let mut t =
            BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[5, 5])
                .unwrap();
        let params = vec![3.0; t.number_of_parameters()];
        t.set_parameters(&params);
        let far = [-50.0, -50.0];
        assert_eq!(t.transform_point(&far), far.to_vec());
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        // Random-ish coefficients; compare the analytic Jacobian to a central
        // finite difference of transform_point over every parameter, at an
        // interior point.
        let mut t =
            BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[4, 4])
                .unwrap();
        let n = t.number_of_parameters();
        let params: Vec<f64> = (0..n).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.1).collect();
        t.set_parameters(&params);

        let point = [4.3, 5.7];
        let jac = t.jacobian_wrt_parameters(&point);

        let h = 1e-5;
        for k in 0..n {
            let mut pp = params.clone();
            pp[k] += h;
            let mut pm = params.clone();
            pm[k] -= h;
            let mut tp = t.clone();
            tp.set_parameters(&pp);
            let mut tm = t.clone();
            tm.set_parameters(&pm);
            let op = tp.transform_point(&point);
            let om = tm.transform_point(&point);
            for d in 0..2 {
                let fd = (op[d] - om[d]) / (2.0 * h);
                assert!(
                    (fd - jac[d * n + k]).abs() < 1e-6,
                    "param {k}, out {d}: fd {fd} vs analytic {}",
                    jac[d * n + k]
                );
            }
        }
    }

    #[test]
    fn from_image_domain_places_all_voxel_centres_inside() {
        // With physicalDimensions = size·spacing, every voxel centre index
        // 0..size maps inside the valid region, so a constant coefficient field
        // translates every voxel.
        let img = image(16, 16);
        let mut t = BSplineTransform::from_image_domain(&img, &[4, 4]).unwrap();
        let per = t.number_of_parameters_per_dimension();
        let mut params = vec![0.0; t.number_of_parameters()];
        params[..per].fill(0.5);
        t.set_parameters(&params);
        // Corner voxel centres (0,0) and (15,15) both displace by +0.5 in x.
        for p in &[[0.0, 0.0], [15.0, 15.0], [0.0, 15.0]] {
            let out = t.transform_point(p);
            assert!(
                (out[0] - (p[0] + 0.5)).abs() < 1e-9,
                "voxel {p:?} not displaced: {out:?}"
            );
        }
    }

    #[test]
    fn from_image_initializer_matches_itk_axis_aligned() {
        // itk::BSplineTransformInitializer on an axis-aligned image places the
        // origin at the ε-expanded min corner (index −0.5−⅛ = −0.625) and takes
        // physicalDimensions = (size + 2ε)·spacing = (size + 0.25)·spacing.
        //   origin        = [5 − 0.625·2,  −4 − 0.625·3] = [3.75, −5.875]
        //   physicalDims   = [(10+0.25)·2, (8+0.25)·3]   = [20.5, 24.75]
        //   gridSpacing    = physicalDims / mesh          = [20.5/5, 24.75/4]
        //   gridOrigin     = origin − gridSpacing         = [−0.35, −12.0625]
        let img = image_geom([10, 8], [2.0, 3.0], [5.0, -4.0], [1.0, 0.0, 0.0, 1.0]);
        let t = BSplineTransform::from_image_initializer(&img, &[5, 4]).unwrap();

        assert_eq!(t.grid_size(), &[8, 7]);
        for (g, e) in t.grid_spacing().iter().zip([4.1, 6.1875]) {
            assert!((g - e).abs() < 1e-12, "grid_spacing {g} vs {e}");
        }
        for (g, e) in t.grid_origin().iter().zip([-0.35, -12.0625]) {
            assert!((g - e).abs() < 1e-12, "grid_origin {g} vs {e}");
        }
    }

    #[test]
    fn from_image_initializer_covers_voxel_corners() {
        // The ⅛-voxel epsilon makes the domain strictly contain every voxel — even
        // each voxel's own corners at index ±0.5. The plain `from_image_domain`
        // convenience (domain = size·spacing, no epsilon) leaves the low corner
        // outside its valid region. Identity-geometry image ⇒ index == physical.
        let img = image(16, 16);
        let fill = |t: &mut BSplineTransform| {
            let per = t.number_of_parameters_per_dimension();
            let mut params = vec![0.0; t.number_of_parameters()];
            params[..per].fill(0.5);
            t.set_parameters(&params);
        };

        let mut init = BSplineTransform::from_image_initializer(&img, &[4, 4]).unwrap();
        fill(&mut init);
        for p in &[[-0.5, -0.5], [15.5, 15.5], [15.5, -0.5], [-0.5, 15.5]] {
            let out = init.transform_point(p);
            assert!(
                (out[0] - (p[0] + 0.5)).abs() < 1e-9,
                "initializer left voxel corner {p:?} uncovered: {out:?}"
            );
        }

        // The low voxel corner is outside the plain size·spacing domain.
        let mut plain = BSplineTransform::from_image_domain(&img, &[4, 4]).unwrap();
        fill(&mut plain);
        let low = [-0.5, -0.5];
        assert_eq!(
            plain.transform_point(&low),
            low.to_vec(),
            "from_image_domain unexpectedly covered the voxel corner"
        );
    }

    #[test]
    fn from_image_initializer_handles_rotated_image() {
        // 90° direction D = [[0,−1],[1,0]] maps index-axis-0 → physical +y and
        // index-axis-1 → physical −x. The bbox-min-nearest corner is corner id 2
        // (index [−0.625, size1−0.375]) at physical [−5.625, −0.625]; the greedy
        // axis match assigns physical axis 0 ← image axis 1 (length 6+0.25) and
        // physical axis 1 ← image axis 0 (length 10+0.25), and the reconstructed
        // edges are axis-aligned so the domain direction comes out identity.
        //   physicalDims = [6.25, 10.25], gridSpacing = /5 = [1.25, 2.05]
        //   gridOrigin   = [−5.625, −0.625] − gridSpacing = [−6.875, −2.675]
        let img = image_geom([10, 6], [1.0, 1.0], [0.0, 0.0], [0.0, -1.0, 1.0, 0.0]);
        let t = BSplineTransform::from_image_initializer(&img, &[5, 5]).unwrap();

        assert_eq!(t.grid_size(), &[8, 8]);
        for (g, e) in t.grid_spacing().iter().zip([1.25, 2.05]) {
            assert!((g - e).abs() < 1e-12, "grid_spacing {g} vs {e}");
        }
        for (g, e) in t.grid_origin().iter().zip([-6.875, -2.675]) {
            assert!((g - e).abs() < 1e-12, "grid_origin {g} vs {e}");
        }
    }

    #[test]
    fn from_image_initializer_covers_voxels_under_general_rotation() {
        // Under a non-axis-aligned (30°) direction the initializer domain still
        // contains every voxel; a constant coefficient field displaces each voxel
        // centre uniformly (weights sum to 1 inside the valid region).
        let (c, s) = (30.0_f64.to_radians().cos(), 30.0_f64.to_radians().sin());
        let img = image_geom([12, 9], [1.5, 0.8], [2.0, -3.0], [c, -s, s, c]);
        let mut t = BSplineTransform::from_image_initializer(&img, &[4, 4]).unwrap();
        let per = t.number_of_parameters_per_dimension();
        let (cx, cy) = (0.3, -0.4);
        let mut params = vec![0.0; t.number_of_parameters()];
        params[..per].fill(cx);
        params[per..2 * per].fill(cy);
        t.set_parameters(&params);

        for &idx in &[[0.0, 0.0], [11.0, 8.0], [0.0, 8.0], [11.0, 0.0], [5.0, 4.0]] {
            let p = img.continuous_index_to_physical_point(&idx);
            let out = t.transform_point(&p);
            assert!(
                (out[0] - (p[0] + cx)).abs() < 1e-9 && (out[1] - (p[1] + cy)).abs() < 1e-9,
                "voxel index {idx:?} (phys {p:?}) not covered: {out:?}"
            );
        }
    }
}
