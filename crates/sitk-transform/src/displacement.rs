//! Dense displacement-field transform (`itk::DisplacementFieldTransform`).
//!
//! A [`DisplacementFieldTransform`] warps space by a **dense** vector field
//! sampled on a regular grid: every grid pixel carries its own
//! `dimension`-vector displacement, and the displacement at an arbitrary point
//! is the multilinear interpolation of the surrounding pixels. The mapped point
//! is `x + displacement(x)`; a zero field is the identity. Unlike the B-spline
//! transform — whose displacement is a smooth combination of a few control
//! points — a displacement field has one free vector *per pixel*, the most
//! flexible deformable transform.
//!
//! ```text
//! displacement(x) = Σ_p  wₚ(x) · dₚ          (wₚ = multilinear weights)
//! T(x) = x + displacement(x)
//! ```
//!
//! Outside the field's buffer (ITK's pixel-centred `[-0.5, size − 0.5)` region)
//! the displacement is zero, so `T(x) = x`.
//!
//! # Parameters
//!
//! The parameter vector *is* the field, laid out exactly as ITK's
//! `ImageVectorOptimizerParametersHelper` exposes the vector-image buffer:
//! **pixel-major, component-fastest** — `params[pixel · dimension + component]`,
//! where `pixel` is the grid raster index (first axis fastest). Identity is all
//! zeros. This is the layout ITK's local-support metric accumulation addresses
//! through `ComputeParameterOffsetFromVirtualIndex = pixel · dimension`.
//!
//! # Jacobian
//!
//! `∂T_c / ∂(displacement component c′ at pixel p)` is the interpolation weight
//! `wₚ(x)` when `c == c′` and zero otherwise, so the Jacobian is sparse: only
//! the `2^dimension` interpolation neighbours (times the matching output
//! component) are non-zero. At a grid point this reduces to the identity block
//! ITK returns from `ComputeJacobianWithRespectToParameters`. This
//! implementation returns it through the dense
//! [`ParametricTransform::jacobian_wrt_parameters`] contract (a
//! `dimension × numberOfParameters` matrix that is mostly zero), so the
//! transform drops into the existing metric/optimizer unchanged.
//!
//! Because a displacement field has `dimension × numberOfPixels` parameters,
//! that dense contract is only practical for small fields. For the genuine
//! local-support path — one that never materializes the dense Jacobian — the
//! transform also reports [`has_local_support`] and exposes, per point, the
//! `dimension × dimension` identity [`local_support_jacobian`] together with the
//! pixel's parameter-block offset; a metric with a local-support branch (Mattes
//! mutual information) uses these to accumulate the derivative per pixel.
//!
//! [`has_local_support`]: ParametricTransform::has_local_support
//! [`local_support_jacobian`]: ParametricTransform::local_support_jacobian

use sitk_core::Image;

use crate::error::{Result, TransformError};
use crate::interpolator::{is_inside, physical_to_index_matrix, strides};
use crate::transform::{ParametricTransform, Transform};

/// A dense displacement-field transform. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct DisplacementFieldTransform {
    dim: usize,
    /// Field grid size, per axis.
    size: Vec<usize>,
    /// Physical origin of pixel `(0,…,0)`.
    origin: Vec<f64>,
    /// `diag(1/spacing) · direction⁻¹`, row-major `dim × dim`: maps a physical
    /// displacement from `origin` to a continuous field index.
    phys_to_index: Vec<f64>,
    /// Raster strides of the field grid (first axis fastest).
    strides: Vec<usize>,
    /// Number of grid pixels (`Π size`).
    num_pixels: usize,
    /// The field, interleaved `[pixel · dim + component]`; the parameter vector.
    field: Vec<f64>,
}

impl DisplacementFieldTransform {
    /// Build a displacement-field transform on the given grid geometry, mirroring
    /// `itk::DisplacementFieldTransform` with the displacement field set to zero
    /// (identity). `direction` is row-major `dim × dim`. Fails if any argument's
    /// length is inconsistent with `dim`, the grid is empty, or the direction
    /// matrix is singular.
    pub fn new(
        dim: usize,
        size: &[usize],
        origin: &[f64],
        spacing: &[f64],
        direction: &[f64],
    ) -> Result<Self> {
        if size.len() != dim
            || origin.len() != dim
            || spacing.len() != dim
            || direction.len() != dim * dim
            || size.contains(&0)
        {
            return Err(TransformError::InvalidDisplacementFieldDomain);
        }
        let phys_to_index = physical_to_index_matrix(direction, spacing, dim)
            .ok_or(TransformError::SingularDirection)?;
        let num_pixels: usize = size.iter().product();
        Ok(Self {
            dim,
            size: size.to_vec(),
            origin: origin.to_vec(),
            phys_to_index,
            strides: strides(size),
            num_pixels,
            field: vec![0.0; num_pixels * dim],
        })
    }

    /// Build a displacement-field transform whose grid matches `image` (the ITK
    /// convention that the field shares the virtual/fixed domain), with a zero
    /// field.
    pub fn from_image_domain(image: &Image) -> Result<Self> {
        Self::new(
            image.dimension(),
            image.size(),
            image.origin(),
            image.spacing(),
            image.direction(),
        )
    }

    /// Field grid size, per axis.
    pub fn size(&self) -> &[usize] {
        &self.size
    }

    /// Number of grid pixels (`Π size`).
    pub fn number_of_pixels(&self) -> usize {
        self.num_pixels
    }

    /// Continuous field index of physical point `p`:
    /// `phys_to_index · (p − origin)`.
    fn continuous_index(&self, p: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        (0..dim)
            .map(|r| {
                (0..dim)
                    .map(|c| self.phys_to_index[r * dim + c] * (p[c] - self.origin[c]))
                    .sum()
            })
            .collect()
    }

    /// The `2^dim` multilinear interpolation neighbours as `(pixel raster offset,
    /// weight)` pairs, with border indices clamped into `[0, size − 1]` exactly as
    /// [`crate::interpolator::linear_at`]. Coincident (clamped) corners appear
    /// more than once, so consumers must **accumulate** their weights.
    fn corner_weights(&self, cindex: &[f64]) -> Vec<(usize, f64)> {
        let dim = self.dim;
        let mut base = vec![0isize; dim];
        let mut frac = vec![0.0f64; dim];
        for d in 0..dim {
            let f = cindex[d].floor();
            base[d] = f as isize;
            frac[d] = cindex[d] - f;
        }
        let mut out = Vec::with_capacity(1 << dim);
        for corner in 0..(1usize << dim) {
            let mut weight = 1.0;
            let mut offset = 0usize;
            for d in 0..dim {
                let bit = (corner >> d) & 1;
                weight *= if bit == 1 { frac[d] } else { 1.0 - frac[d] };
                let idx = (base[d] + bit as isize).clamp(0, self.size[d] as isize - 1) as usize;
                offset += idx * self.strides[d];
            }
            out.push((offset, weight));
        }
        out
    }
}

impl Transform for DisplacementFieldTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let cindex = self.continuous_index(point);
        if !is_inside(&cindex, &self.size) {
            return point.to_vec(); // outside the field ⇒ zero displacement
        }
        let mut out = point.to_vec();
        for (off, w) in self.corner_weights(&cindex) {
            for (c, o) in out.iter_mut().enumerate() {
                *o += w * self.field[off * dim + c];
            }
        }
        out
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

impl ParametricTransform for DisplacementFieldTransform {
    fn number_of_parameters(&self) -> usize {
        self.num_pixels * self.dim
    }

    fn parameters(&self) -> Vec<f64> {
        self.field.clone()
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(
            params.len(),
            self.field.len(),
            "displacement-field parameter vector length mismatch"
        );
        self.field.copy_from_slice(params);
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let nparams = self.number_of_parameters();
        let mut jac = vec![0.0f64; dim * nparams];
        let cindex = self.continuous_index(point);
        if !is_inside(&cindex, &self.size) {
            return jac; // outside the field ⇒ zero Jacobian
        }
        // ∂T_c/∂(component c at pixel off) = interpolation weight; accumulate
        // because border clamping can map several corners to the same pixel.
        for (off, w) in self.corner_weights(&cindex) {
            for c in 0..dim {
                jac[c * nparams + off * dim + c] += w;
            }
        }
        jac
    }

    fn has_local_support(&self) -> bool {
        true
    }

    fn number_of_local_parameters(&self) -> usize {
        self.dim
    }

    fn local_support_jacobian(&self, point: &[f64]) -> Option<(usize, Vec<f64>)> {
        let dim = self.dim;
        let cindex = self.continuous_index(point);
        if !is_inside(&cindex, &self.size) {
            return None; // outside the field ⇒ no local region governs this point
        }
        // The field shares the virtual/fixed domain, so a metric sample lands on
        // a grid point: round to that pixel and return its parameter-block offset
        // together with the identity local Jacobian ITK's
        // `ComputeJacobianWithRespectToParameters` yields there (frac = 0 ⇒ unit
        // weight at the pixel's own displacement).
        let mut pixel = 0usize;
        for (d, &ci) in cindex.iter().enumerate() {
            let idx = (ci.round() as isize).clamp(0, self.size[d] as isize - 1) as usize;
            pixel += idx * self.strides[d];
        }
        let mut local_jac = vec![0.0f64; dim * dim];
        for c in 0..dim {
            local_jac[c * dim + c] = 1.0;
        }
        Some((pixel * dim, local_jac))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_dir(dim: usize) -> Vec<f64> {
        let mut d = vec![0.0; dim * dim];
        for i in 0..dim {
            d[i * dim + i] = 1.0;
        }
        d
    }

    /// A 2-D field, unit spacing, identity direction, origin 0.
    fn field(w: usize, h: usize) -> DisplacementFieldTransform {
        DisplacementFieldTransform::new(2, &[w, h], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
            .unwrap()
    }

    #[test]
    fn zero_field_is_identity() {
        let t = field(6, 6);
        for p in &[[2.0, 3.0], [0.0, 0.0], [4.5, 1.2], [5.0, 5.0]] {
            assert_eq!(t.transform_point(p), p.to_vec());
        }
    }

    #[test]
    fn constant_field_is_a_uniform_translation() {
        // Every pixel displaced by (dx, dy): interpolation of a constant field
        // is that constant everywhere inside, at grid and off-grid points alike.
        let mut t = field(6, 6);
        let (dx, dy) = (1.5, -0.75);
        let mut params = vec![0.0; t.number_of_parameters()];
        for p in 0..t.number_of_pixels() {
            params[p * 2] = dx;
            params[p * 2 + 1] = dy;
        }
        t.set_parameters(&params);
        for p in &[[2.0, 3.0], [3.4, 1.6], [0.0, 5.0]] {
            let out = t.transform_point(p);
            assert!(
                (out[0] - (p[0] + dx)).abs() < 1e-12 && (out[1] - (p[1] + dy)).abs() < 1e-12,
                "at {p:?}: got {out:?}"
            );
        }
    }

    #[test]
    fn off_grid_displacement_is_linearly_interpolated() {
        // Two adjacent pixels along x carry different x-displacements; the
        // midpoint between them gets the average.
        let mut t = field(4, 4);
        let mut params = vec![0.0; t.number_of_parameters()];
        // pixel (1,1) raster index = 1 + 1*4 = 5; pixel (2,1) = 2 + 1*4 = 6.
        params[5 * 2] = 2.0; // dx at (1,1)
        params[6 * 2] = 4.0; // dx at (2,1)
        t.set_parameters(&params);
        // Point at continuous index (1.5, 1) is the x-midpoint of the two pixels.
        let out = t.transform_point(&[1.5, 1.0]);
        assert!((out[0] - (1.5 + 3.0)).abs() < 1e-12, "got {out:?}");
        assert!((out[1] - 1.0).abs() < 1e-12, "got {out:?}");
    }

    #[test]
    fn points_outside_the_field_are_unmapped() {
        let mut t = field(6, 6);
        let params = vec![3.0; t.number_of_parameters()];
        t.set_parameters(&params);
        let far = [-50.0, 100.0];
        assert_eq!(t.transform_point(&far), far.to_vec());
    }

    #[test]
    fn parameters_roundtrip() {
        let mut t = field(3, 3);
        let n = t.number_of_parameters();
        assert_eq!(n, 3 * 3 * 2);
        let params: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        t.set_parameters(&params);
        assert_eq!(t.parameters(), params);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let mut t = field(4, 4);
        let n = t.number_of_parameters();
        let params: Vec<f64> = (0..n).map(|i| ((i * 17 % 7) as f64 - 3.0) * 0.1).collect();
        t.set_parameters(&params);

        let point = [1.7, 2.3];
        let jac = t.jacobian_wrt_parameters(&point);
        let h = 1e-6;
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
    fn from_image_domain_matches_image_grid() {
        let img = Image::from_vec(&[5, 7], vec![0.0; 35]).unwrap();
        let t = DisplacementFieldTransform::from_image_domain(&img).unwrap();
        assert_eq!(t.size(), &[5, 7]);
        assert_eq!(t.number_of_pixels(), 35);
        assert_eq!(t.number_of_parameters(), 35 * 2);
    }

    #[test]
    fn local_support_is_advertised_with_dimension_local_parameters() {
        let t = field(4, 5);
        assert!(t.has_local_support());
        assert_eq!(t.number_of_local_parameters(), 2);
    }

    #[test]
    fn local_support_jacobian_is_the_owning_pixel_and_identity() {
        let t = field(4, 5);
        // Grid point (2,3): raster pixel = 2 + 3*4 = 14, block offset = 14*2 = 28.
        let (offset, jac) = t.local_support_jacobian(&[2.0, 3.0]).unwrap();
        assert_eq!(offset, 14 * 2);
        assert_eq!(jac, vec![1.0, 0.0, 0.0, 1.0]);
        // An off-grid point rounds to its nearest pixel: (1.6, 2.4) → (2,2),
        // raster 2 + 2*4 = 10, offset 20.
        let (offset, _) = t.local_support_jacobian(&[1.6, 2.4]).unwrap();
        assert_eq!(offset, 10 * 2);
        // Outside the field ⇒ no owning region.
        assert!(t.local_support_jacobian(&[-50.0, 100.0]).is_none());
    }

    #[test]
    fn local_support_jacobian_offset_matches_the_dense_jacobian_block() {
        // At a grid point the dense Jacobian is non-zero only in the owning
        // pixel's block, and that block is the identity the local path returns.
        let t = field(4, 4);
        let n = t.number_of_parameters();
        let point = [1.0, 2.0]; // raster pixel 1 + 2*4 = 9, offset 18
        let dense = t.jacobian_wrt_parameters(&point);
        let (offset, local) = t.local_support_jacobian(&point).unwrap();
        assert_eq!(offset, 18);
        for c in 0..2 {
            for mu in 0..2 {
                assert_eq!(dense[c * n + offset + mu], local[c * 2 + mu]);
            }
        }
        // Every non-zero dense entry lies inside that block.
        for (k, &v) in dense.iter().enumerate() {
            if v != 0.0 {
                let col = k % n;
                assert!((offset..offset + 2).contains(&col), "stray non-zero at {k}");
            }
        }
    }

    #[test]
    fn singular_direction_is_rejected() {
        let singular = vec![1.0, 1.0, 1.0, 1.0]; // rank-deficient
        assert!(matches!(
            DisplacementFieldTransform::new(2, &[4, 4], &[0.0, 0.0], &[1.0, 1.0], &singular),
            Err(TransformError::SingularDirection)
        ));
    }
}
