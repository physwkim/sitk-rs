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
//! the matching output dimension) are non-zero. The dense
//! [`ParametricTransform::jacobian_wrt_parameters`] contract (a `dimension ×
//! numberOfParameters` row-major matrix that is mostly zero) is still
//! available, so the transform drops into any metric/optimizer unchanged; it
//! is implemented in terms of the sparse form below.
//!
//! [`ParametricTransform::sparse_jacobian_wrt_parameters`] returns exactly
//! those non-zero entries — `(order+1)^dimension · dimension` `(parameter
//! index, column)` pairs — without ever allocating the dense array. This is
//! **not** ITK's `HasLocalSupport` metric branch: per
//! `itk::BSplineBaseTransform::GetTransformCategory`
//! (`itkBSplineBaseTransform.h`), a B-spline transform's category is
//! `TransformCategoryEnum::BSpline`, not `DisplacementField`, so
//! `itk::ObjectToObjectMetric::HasLocalSupport()` — which checks exactly
//! `GetTransformCategory() == DisplacementField` — is `false` for it, and
//! ITK's own metric threader (`ImageToImageMetricv4GetValueAndDerivativeThreaderBase
//! ::StorePointDerivativeResult`) accumulates a B-spline's derivative through
//! the *dense*, global-support branch. This crate's sparse accessor is a
//! separate, purely internal performance device — see
//! [`ParametricTransform::sparse_jacobian_wrt_parameters`] for how it stays
//! decoupled from `HasLocalSupport` parity.

use sitk_core::{Image, matrix};

use crate::error::{Result, TransformError};
use crate::interpolator::physical_to_index_matrix;
use crate::transform::{ParametricTransform, TransformBase};

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
#[derive(Clone, Debug, PartialEq)]
pub struct BSplineTransform {
    dim: usize,
    /// Control points per axis (`meshSize + splineOrder`).
    grid_size: Vec<usize>,
    /// Physical origin of control point `(0,…,0)`.
    grid_origin: Vec<f64>,
    /// Physical spacing between adjacent control points, per axis.
    grid_spacing: Vec<f64>,
    /// Orientation of the control-point grid, row-major `dim × dim` — the same
    /// matrix as the transform-domain direction, and the trailing block of the
    /// transform's fixed parameters.
    grid_direction: Vec<f64>,
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

        Self::from_grid_geometry(
            dim,
            &grid_size,
            &grid_origin,
            &grid_spacing,
            domain_direction,
        )
    }

    /// Build directly from the **control-point grid** geometry, with all
    /// coefficients zero. This is what `itk::BSplineTransform`'s fixed
    /// parameters encode, and what
    /// `SetCoefficientImageInformationFromFixedParameters` reconstructs
    /// (`itkBSplineTransform.hxx:62`); [`new`](Self::new) derives this geometry
    /// from a transform domain and then calls through here.
    ///
    /// Fails if any argument's length is inconsistent with `dim`, a grid size is
    /// zero, or `grid_direction` is singular.
    fn from_grid_geometry(
        dim: usize,
        grid_size: &[usize],
        grid_origin: &[f64],
        grid_spacing: &[f64],
        grid_direction: &[f64],
    ) -> Result<Self> {
        if grid_size.len() != dim
            || grid_origin.len() != dim
            || grid_spacing.len() != dim
            || grid_direction.len() != dim * dim
            || grid_size.contains(&0)
        {
            return Err(TransformError::InvalidBSplineDomain);
        }

        let phys_to_index = physical_to_index_matrix(grid_direction, grid_spacing, dim)
            .ok_or(TransformError::SingularDirection)?;

        // Raster strides, first axis fastest.
        let mut grid_stride = vec![1usize; dim];
        for i in 1..dim {
            grid_stride[i] = grid_stride[i - 1] * grid_size[i - 1];
        }
        let num_per_dim: usize = grid_size.iter().product();

        Ok(Self {
            dim,
            grid_size: grid_size.to_vec(),
            grid_origin: grid_origin.to_vec(),
            grid_spacing: grid_spacing.to_vec(),
            grid_direction: grid_direction.to_vec(),
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

    /// Orientation of the control-point grid, row-major `dim × dim`.
    pub fn grid_direction(&self) -> &[f64] {
        &self.grid_direction
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

impl TransformBase for BSplineTransform {
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

    /// `BSplineBaseTransform::GetTransformCategory()` returns `BSpline`, not
    /// `Linear`.
    fn is_linear(&self) -> bool {
        false
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

    /// `[gridSize, gridOrigin, gridSpacing, gridDirection]`, each block `dim`
    /// long except the row-major `dim × dim` direction — `dim · (dim + 3)`
    /// values in all, exactly as
    /// `itk::BSplineBaseTransform::SetFixedParametersFromTransformDomainInformation`
    /// lays them out (`itkBSplineBaseTransform.hxx:97-112`).
    fn fixed_parameters(&self) -> Vec<f64> {
        let dim = self.dim;
        let mut fp = Vec::with_capacity(dim * (dim + 3));
        fp.extend(self.grid_size.iter().map(|&s| s as f64));
        fp.extend_from_slice(&self.grid_origin);
        fp.extend_from_slice(&self.grid_spacing);
        fp.extend_from_slice(&self.grid_direction);
        fp
    }

    fn number_of_fixed_parameters(&self) -> usize {
        self.dim * (self.dim + 3)
    }

    /// Rebuilds the control-point grid from `params` and zeroes every
    /// coefficient, mirroring `SetCoefficientImageInformationFromFixedParameters`
    /// (which re-allocates the coefficient images). Grid sizes are truncated
    /// toward zero out of the `f64` block, as ITK's
    /// `static_cast<SizeValueType>(m_FixedParameters[i])` does.
    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        let dim = self.dim;
        let expected = dim * (dim + 3);
        if params.len() != expected {
            return Err(TransformError::InvalidFixedParameters {
                got: params.len(),
                expected: format!("{expected} (grid size, origin, spacing, direction)"),
            });
        }
        let grid_size: Vec<usize> = params[..dim]
            .iter()
            .map(|&s| if s >= 1.0 { s as usize } else { 0 })
            .collect();
        *self = Self::from_grid_geometry(
            dim,
            &grid_size,
            &params[dim..2 * dim],
            &params[2 * dim..3 * dim],
            &params[3 * dim..],
        )?;
        Ok(())
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let nparams = self.number_of_parameters();
        let mut jac = vec![0.0f64; dim * nparams];
        if let Some(entries) = self.sparse_jacobian_wrt_parameters(point) {
            for (idx, col) in entries {
                for (d, &w) in col.iter().enumerate() {
                    jac[d * nparams + idx] = w;
                }
            }
        }
        jac
    }

    fn sparse_jacobian_wrt_parameters(&self, point: &[f64]) -> Option<Vec<(usize, Vec<f64>)>> {
        let dim = self.dim;

        let mut index = self.continuous_index(point);
        if !self.inside_valid_region(&mut index) {
            return Some(Vec::new()); // outside grid ⇒ zero Jacobian, no affected params
        }

        let (weights, start) = self.evaluate_weights(&index);
        let taps = SPLINE_ORDER + 1;
        let mut entries = Vec::with_capacity(weights.len() * dim);
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
                let mut col = vec![0.0f64; dim];
                col[d] = w;
                entries.push((d * self.num_per_dim + flat, col));
            }
        }
        Some(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_parameters_are_the_control_point_grid_geometry() {
        let t = BSplineTransform::new(2, &[1.0, 2.0], &[4.0, 8.0], &[1.0, 0.0, 0.0, 1.0], &[2, 4])
            .unwrap();
        // gridSize = meshSize + 3; gridSpacing = physDim / meshSize;
        // gridOrigin = domainOrigin - gridSpacing (the order-3 border shift).
        assert_eq!(t.grid_size(), [5, 7]);
        assert_eq!(t.grid_spacing(), [2.0, 2.0]);
        assert_eq!(t.grid_origin(), [-1.0, 0.0]);
        assert_eq!(t.grid_direction(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            t.fixed_parameters(),
            vec![5.0, 7.0, -1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(t.number_of_fixed_parameters(), 2 * (2 + 3));
    }

    #[test]
    fn set_fixed_parameters_rebuilds_the_grid_and_zeroes_the_coefficients() {
        let mut t =
            BSplineTransform::new(2, &[0.0, 0.0], &[4.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[1, 1])
                .unwrap();
        t.set_parameters(&vec![1.0; t.number_of_parameters()]);

        let target =
            BSplineTransform::new(2, &[1.0, 2.0], &[4.0, 8.0], &[1.0, 0.0, 0.0, 1.0], &[2, 4])
                .unwrap();
        t.set_fixed_parameters(&target.fixed_parameters()).unwrap();

        // Geometry adopted, coefficients re-allocated to zero (ITK re-allocates
        // the coefficient images in SetCoefficientImageInformationFromFixedParameters).
        assert_eq!(t, target);
        assert!(t.parameters().iter().all(|&c| c == 0.0));

        assert!(t.set_fixed_parameters(&[1.0, 2.0]).is_err());
    }

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

    /// The reference grid used by the single-coefficient tests: domain
    /// `[0, 10]²`, mesh 5 ⇒ `gridSpacing = 2`, `gridSize = 8`, `gridOrigin = −2`.
    /// Continuous grid index of physical `p` is `(p + 2) / 2`, so grid index
    /// `g` sits at physical `2g − 2`, and the valid region `[1, 6)` in index
    /// space is physical `[0, 10)`.
    fn mesh5_grid() -> BSplineTransform {
        BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[5, 5]).unwrap()
    }

    /// Set exactly one control-point coefficient: grid point `(gx, gy)` of
    /// output dimension `d`.
    fn one_coefficient(t: &mut BSplineTransform, d: usize, gx: usize, gy: usize, value: f64) {
        let per = t.number_of_parameters_per_dimension();
        let mut params = vec![0.0; t.number_of_parameters()];
        params[d * per + gy * t.grid_size()[0] + gx] = value;
        t.set_parameters(&params);
    }

    #[test]
    fn a_single_coefficient_displaces_by_the_hand_derived_weight() {
        // Coefficient 1.0 at control point (2, 2), dimension 0. At an integer
        // grid index g the support is [g−1, g+2] with 1-D weights
        // B₃(1), B₃(0), B₃(−1), B₃(−2) = 1/6, 2/3, 1/6, 0; at a half index the
        // taps are 1/48, 23/48, 23/48, 1/48. Displacement is the tensor product
        // of the two axes' weights on control point (2, 2).
        let mut t = mesh5_grid();
        one_coefficient(&mut t, 0, 2, 2, 1.0);

        // (physical point, expected x-displacement)
        let cases: [([f64; 2], f64); 4] = [
            // grid index (2, 2): control point 2 sits at support offset 1 ⇒ 2/3 per axis.
            ([2.0, 2.0], (2.0 / 3.0) * (2.0 / 3.0)),
            // grid index (3, 3): support [2, 5], control point 2 at offset 0 ⇒ 1/6 per axis.
            ([4.0, 4.0], (1.0 / 6.0) * (1.0 / 6.0)),
            // grid index (2.5, 2): x taps 1/48, 23/48, 23/48, 1/48 over [1, 4] ⇒
            // control point 2 gets 23/48; y weight is 2/3.
            ([3.0, 2.0], (23.0 / 48.0) * (2.0 / 3.0)),
            // grid index (1, 1): support [0, 3], control point 2 at offset 2 ⇒ 1/6 per axis.
            ([0.0, 0.0], (1.0 / 6.0) * (1.0 / 6.0)),
        ];
        for (p, want_dx) in cases {
            let out = t.transform_point(&p);
            assert!(
                (out[0] - (p[0] + want_dx)).abs() < 1e-12,
                "{p:?}: dx {} vs {want_dx}",
                out[0] - p[0]
            );
            assert!(
                (out[1] - p[1]).abs() < 1e-12,
                "{p:?}: dimension-0 coefficient leaked into y ({})",
                out[1] - p[1]
            );
        }
    }

    #[test]
    fn a_single_coefficient_leaves_points_outside_its_support_untouched() {
        // The cubic support of control point (2, 2) is grid index (0, 4)² —
        // open at 4 since B₃(±2) = 0. Points whose support region excludes
        // control point 2 on either axis get exactly zero displacement, even
        // though they are well inside the valid region.
        let mut t = mesh5_grid();
        one_coefficient(&mut t, 0, 2, 2, 5.0);
        // grid index (4, 4) → physical (6, 6); grid index (2, 4) → (2, 6);
        // grid index (4, 2) → (6, 2); grid index (5, 5) → (8, 8).
        for p in &[[6.0, 6.0], [2.0, 6.0], [6.0, 2.0], [8.0, 8.0]] {
            let out = t.transform_point(p);
            assert_eq!(out, p.to_vec(), "point {p:?} outside the support moved");
        }
    }

    #[test]
    fn a_dimension_one_coefficient_displaces_only_y() {
        // Parameter ordering is all-X then all-Y: the same flat grid offset in
        // the second half of the vector must drive the y displacement and
        // nothing else, with the identical weight.
        let mut t = mesh5_grid();
        one_coefficient(&mut t, 1, 2, 2, 1.0);
        let out = t.transform_point(&[2.0, 2.0]);
        let want_dy = (2.0 / 3.0) * (2.0 / 3.0);
        assert!((out[0] - 2.0).abs() < 1e-12, "x moved: {}", out[0]);
        assert!(
            (out[1] - (2.0 + want_dy)).abs() < 1e-12,
            "dy {} vs {want_dy}",
            out[1] - 2.0
        );
    }

    #[test]
    fn parameters_round_trip_in_itk_ordering() {
        // `parameters()` must return exactly what `set_parameters` was given —
        // the `dim` coefficient grids concatenated, each in first-axis-fastest
        // raster order (`itk::BSplineBaseTransform::WrapAsImages`). A transposed
        // or interleaved layout would survive a length check but not this one.
        let mut t = mesh5_grid();
        let n = t.number_of_parameters();
        assert_eq!(n, 2 * 8 * 8);
        assert_eq!(t.number_of_parameters_per_dimension(), 64);
        assert!(t.parameters().iter().all(|&v| v == 0.0));

        let params: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 3.0).collect();
        t.set_parameters(&params);
        assert_eq!(t.parameters(), params);
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
    fn sparse_jacobian_reproduces_the_dense_jacobians_nonzero_entries() {
        // At several off-lattice points — including ones near the domain
        // border where the support region is clipped by `inside_valid_region`
        // — the sparse accessor's (index, column) entries must reproduce
        // exactly the non-zero entries of the dense Jacobian, and nothing
        // else.
        let t = BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[4, 4])
            .unwrap();
        let n = t.number_of_parameters();
        // grid_spacing = 2.5, grid_origin = −2.5 ⇒ continuous index (p+2.5)/2.5;
        // the valid interval [1, 5) per axis corresponds to physical p ∈ [0, 10) —
        // exactly the transform domain. Points near p = 0 and p = 10 sit close
        // to that clipped border but remain (just) inside it.
        for point in &[[4.3, 5.7], [0.05, 0.05], [9.95, 9.95], [0.05, 9.95]] {
            let dense = t.jacobian_wrt_parameters(point);
            let entries = t.sparse_jacobian_wrt_parameters(point).unwrap();

            // Every sparse entry matches the dense matrix at that column.
            for (idx, col) in &entries {
                for d in 0..2 {
                    assert_eq!(
                        dense[d * n + idx],
                        col[d],
                        "point {point:?}, param {idx}, dim {d}"
                    );
                }
            }
            // Every non-zero dense entry is covered by some sparse entry.
            let sparse_cols: std::collections::HashSet<usize> =
                entries.iter().map(|(idx, _)| *idx).collect();
            for (k, &v) in dense.iter().enumerate() {
                if v != 0.0 {
                    let col = k % n;
                    assert!(
                        sparse_cols.contains(&col),
                        "point {point:?}: dense non-zero at param {col} missing from sparse entries"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_jacobian_affects_exactly_order_plus_one_pow_dim_times_dim_params() {
        // An interior point (well away from the border) affects exactly
        // (order+1)^dim control points, each contributing one entry per
        // output dimension.
        let t = BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[4, 4])
            .unwrap();
        let entries = t.sparse_jacobian_wrt_parameters(&[5.0, 5.0]).unwrap();
        assert_eq!(entries.len(), 4usize.pow(2) * 2);

        let t3 = BSplineTransform::new(
            3,
            &[0.0, 0.0, 0.0],
            &[10.0, 10.0, 10.0],
            &matrix::identity(3),
            &[4, 4, 4],
        )
        .unwrap();
        let entries3 = t3.sparse_jacobian_wrt_parameters(&[5.0, 5.0, 5.0]).unwrap();
        assert_eq!(entries3.len(), 4usize.pow(3) * 3);
    }

    #[test]
    fn sparse_jacobian_is_empty_outside_the_valid_region() {
        // Far outside the grid the dense Jacobian is all-zero; the sparse
        // accessor reports that as `Some(empty)`, not `None` — the transform
        // does support a sparse representation, this point just contributes
        // nothing to it.
        let t = BSplineTransform::new(2, &[0.0, 0.0], &[10.0, 10.0], &matrix::identity(2), &[5, 5])
            .unwrap();
        assert_eq!(
            t.sparse_jacobian_wrt_parameters(&[-50.0, -50.0]),
            Some(Vec::new())
        );
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
