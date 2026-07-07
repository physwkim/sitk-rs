//! Spatial transforms.
//!
//! A [`Transform`] maps a point in one physical space to another. In resampling
//! it maps a point in the **output** image's physical space to the **input**
//! image's physical space (ITK's backward mapping convention).

use sitk_core::matrix;

/// A spatial coordinate transform.
pub trait Transform {
    /// Map a physical point to its transformed physical point.
    fn transform_point(&self, point: &[f64]) -> Vec<f64>;
    /// Spatial dimension the transform operates on.
    fn dimension(&self) -> usize;
}

/// A transform whose action is controlled by a flat parameter vector, and which
/// exposes the Jacobian of the mapped point with respect to those parameters.
/// This is the interface registration optimizes over, mirroring ITK's
/// `Transform::GetJacobianWithRespectToParameters`.
pub trait ParametricTransform: Transform {
    /// Number of free parameters.
    fn number_of_parameters(&self) -> usize;

    /// Current parameter vector (length [`number_of_parameters`]).
    ///
    /// [`number_of_parameters`]: ParametricTransform::number_of_parameters
    fn parameters(&self) -> Vec<f64>;

    /// Replace the parameter vector. `params.len()` must equal
    /// [`number_of_parameters`].
    ///
    /// [`number_of_parameters`]: ParametricTransform::number_of_parameters
    fn set_parameters(&mut self, params: &[f64]);

    /// Jacobian `∂(transform_point(point))ᵢ / ∂paramₖ`, row-major
    /// `dimension × number_of_parameters`, evaluated at `point`.
    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64>;
}

/// A pure translation: `y = x + t`. Mirrors `itk::TranslationTransform`.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationTransform {
    translation: Vec<f64>,
}

impl TranslationTransform {
    /// A translation by `translation` (length = dimension).
    pub fn new(translation: Vec<f64>) -> Self {
        assert!(!translation.is_empty(), "dimension must be >= 1");
        Self { translation }
    }

    /// The translation vector.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }
}

impl Transform for TranslationTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), self.translation.len());
        point
            .iter()
            .zip(self.translation.iter())
            .map(|(&p, &t)| p + t)
            .collect()
    }

    fn dimension(&self) -> usize {
        self.translation.len()
    }
}

impl ParametricTransform for TranslationTransform {
    fn number_of_parameters(&self) -> usize {
        self.translation.len()
    }

    fn parameters(&self) -> Vec<f64> {
        self.translation.clone()
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), self.translation.len(), "parameter length");
        self.translation.copy_from_slice(params);
    }

    fn jacobian_wrt_parameters(&self, _point: &[f64]) -> Vec<f64> {
        // ∂(x + t)ᵢ / ∂tₖ = δᵢₖ — the identity.
        let dim = self.translation.len();
        let mut j = vec![0.0; dim * dim];
        for i in 0..dim {
            j[i * dim + i] = 1.0;
        }
        j
    }
}

/// An affine transform `y = A·(x − center) + translation + center`, mirroring
/// `itk::MatrixOffsetTransformBase` / `itk::AffineTransform`.
///
/// The `center` of rotation is fixed; `matrix` and `translation` are the
/// optimizable parameters (matrix row-major first, then translation, matching
/// ITK's parameter ordering). The equivalent `offset` in `y = A·x + offset`,
/// with `offset = center + translation − A·center`, is cached and refreshed
/// whenever the parameters change.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineTransform {
    dim: usize,
    /// Row-major `dim x dim`.
    matrix: Vec<f64>,
    /// Length `dim`.
    translation: Vec<f64>,
    /// Length `dim`, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached `center + translation − A·center`.
    offset: Vec<f64>,
}

impl AffineTransform {
    /// Build from a row-major `dim x dim` `matrix`, a `translation`, and a
    /// `center` of rotation. Panics on inconsistent lengths.
    pub fn new(dim: usize, matrix: Vec<f64>, translation: Vec<f64>, center: Vec<f64>) -> Self {
        assert_eq!(matrix.len(), dim * dim, "matrix must be dim*dim");
        assert_eq!(translation.len(), dim, "translation must be length dim");
        assert_eq!(center.len(), dim, "center must be length dim");
        let offset = Self::compute_offset(dim, &matrix, &translation, &center);
        Self {
            dim,
            matrix,
            translation,
            center,
            offset,
        }
    }

    /// The identity affine transform of the given dimension.
    pub fn identity(dim: usize) -> Self {
        Self {
            dim,
            matrix: matrix::identity(dim),
            translation: vec![0.0; dim],
            center: vec![0.0; dim],
            offset: vec![0.0; dim],
        }
    }

    /// `offset = center + translation − A·center`.
    fn compute_offset(dim: usize, matrix: &[f64], translation: &[f64], center: &[f64]) -> Vec<f64> {
        let a_center = matrix::mat_vec(matrix, center, dim);
        (0..dim)
            .map(|d| center[d] + translation[d] - a_center[d])
            .collect()
    }

    /// Row-major `dim x dim` matrix.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// The translation part (`itk::MatrixOffsetTransformBase::GetTranslation`).
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Translation offset actually applied (`y = A·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }
}

impl Transform for AffineTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), self.dim);
        let ax = matrix::mat_vec(&self.matrix, point, self.dim);
        (0..self.dim).map(|d| ax[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

impl ParametricTransform for AffineTransform {
    fn number_of_parameters(&self) -> usize {
        self.dim * self.dim + self.dim
    }

    fn parameters(&self) -> Vec<f64> {
        let mut p = self.matrix.clone();
        p.extend_from_slice(&self.translation);
        p
    }

    fn set_parameters(&mut self, params: &[f64]) {
        let n = self.dim * self.dim;
        assert_eq!(params.len(), n + self.dim, "parameter length");
        self.matrix.copy_from_slice(&params[..n]);
        self.translation.copy_from_slice(&params[n..]);
        self.offset = Self::compute_offset(self.dim, &self.matrix, &self.translation, &self.center);
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // For y = A·(x − c) + t + c (center c fixed):
        //   ∂yᵢ / ∂A_rc = δᵢᵣ · (x_c − c_c),   ∂yᵢ / ∂t_k = δᵢₖ.
        let dim = self.dim;
        let nparams = self.number_of_parameters();
        let mut j = vec![0.0; dim * nparams];
        for i in 0..dim {
            for c in 0..dim {
                j[i * nparams + (i * dim + c)] = point[c] - self.center[c];
            }
            j[i * nparams + (dim * dim + i)] = 1.0;
        }
        j
    }
}

/// A rigid 2-D transform `y = R(θ)·(x − center) + center + translation`,
/// mirroring `itk::Euler2DTransform` / `itk::Rigid2DTransform`.
///
/// Parameters are `[angle, tx, ty]` (`angle` in radians); the `center` of
/// rotation is fixed (not a parameter). The rotation matrix
/// `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]` and the equivalent
/// `offset = translation + center − R·center` (in `y = R·x + offset`) are cached
/// and refreshed whenever the parameters change.
#[derive(Clone, Debug, PartialEq)]
pub struct Euler2DTransform {
    angle: f64,
    /// Length 2.
    translation: Vec<f64>,
    /// Length 2, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 2×2 `R(θ)`.
    matrix: Vec<f64>,
    /// Cached `translation + center − R·center`.
    offset: Vec<f64>,
}

impl Euler2DTransform {
    /// A rigid transform of `angle` radians about `center`, then `translation`.
    pub fn new(angle: f64, translation: [f64; 2], center: [f64; 2]) -> Self {
        let mut t = Self {
            angle,
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 4],
            offset: vec![0.0; 2],
        };
        t.recompute();
        t
    }

    /// The identity rigid transform (zero angle/translation, center at origin).
    pub fn identity() -> Self {
        Self::new(0.0, [0.0, 0.0], [0.0, 0.0])
    }

    /// Rotation angle in radians.
    pub fn angle(&self) -> f64 {
        self.angle
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 2×2 rotation matrix `R(θ)`.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = R·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Rebuild the cached matrix and offset from `angle`, `translation`, `center`.
    fn recompute(&mut self) {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        self.matrix = vec![c, -s, s, c];
        let m_center = matrix::mat_vec(&self.matrix, &self.center, 2);
        self.offset = (0..2)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
    }
}

impl Transform for Euler2DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 2);
        let mx = matrix::mat_vec(&self.matrix, point, 2);
        (0..2).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        2
    }
}

impl ParametricTransform for Euler2DTransform {
    fn number_of_parameters(&self) -> usize {
        3
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.angle, self.translation[0], self.translation[1]]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 3, "parameter length");
        self.angle = params[0];
        self.translation[0] = params[1];
        self.translation[1] = params[2];
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // y = R(θ)·(x − c) + c + t, parameters [θ, tx, ty]:
        //   ∂y/∂θ = R'(θ)·(x − c),  R'(θ) = [[−sin, −cos], [cos, −sin]]
        //   ∂y/∂t = I.
        let (ca, sa) = (self.angle.cos(), self.angle.sin());
        let (dx, dy) = (point[0] - self.center[0], point[1] - self.center[1]);
        // Row-major 2×3: [ ∂y0/∂θ, ∂y0/∂tx, ∂y0/∂ty ; ∂y1/∂θ, ... ].
        vec![-sa * dx - ca * dy, 1.0, 0.0, ca * dx - sa * dy, 0.0, 1.0]
    }
}

/// A similarity 2-D transform `y = s·R(θ)·(x − center) + center + translation`,
/// mirroring `itk::Similarity2DTransform` — a rigid rotation plus an isotropic
/// `scale`.
///
/// Parameters are `[scale, angle, tx, ty]` (`angle` in radians); the `center` is
/// fixed. The matrix `M = s·R(θ)` and the equivalent
/// `offset = translation + center − M·center` (in `y = M·x + offset`) are cached
/// and refreshed whenever the parameters change.
#[derive(Clone, Debug, PartialEq)]
pub struct Similarity2DTransform {
    scale: f64,
    angle: f64,
    /// Length 2.
    translation: Vec<f64>,
    /// Length 2, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 2×2 `s·R(θ)`.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl Similarity2DTransform {
    /// A similarity transform: rotate `angle` radians about `center`, scale by
    /// `scale`, then `translation`.
    pub fn new(scale: f64, angle: f64, translation: [f64; 2], center: [f64; 2]) -> Self {
        let mut t = Self {
            scale,
            angle,
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 4],
            offset: vec![0.0; 2],
        };
        t.recompute();
        t
    }

    /// The identity similarity transform (scale 1, zero angle/translation, center
    /// at origin).
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, [0.0, 0.0], [0.0, 0.0])
    }

    /// Isotropic scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Rotation angle in radians.
    pub fn angle(&self) -> f64 {
        self.angle
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 2×2 matrix `s·R(θ)`.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Rebuild the cached matrix and offset from `scale`, `angle`, `translation`,
    /// `center`.
    fn recompute(&mut self) {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let (mc, ms) = (c * self.scale, s * self.scale);
        self.matrix = vec![mc, -ms, ms, mc];
        let m_center = matrix::mat_vec(&self.matrix, &self.center, 2);
        self.offset = (0..2)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
    }
}

impl Transform for Similarity2DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 2);
        let mx = matrix::mat_vec(&self.matrix, point, 2);
        (0..2).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        2
    }
}

impl ParametricTransform for Similarity2DTransform {
    fn number_of_parameters(&self) -> usize {
        4
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.scale,
            self.angle,
            self.translation[0],
            self.translation[1],
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 4, "parameter length");
        self.scale = params[0];
        self.angle = params[1];
        self.translation[0] = params[2];
        self.translation[1] = params[3];
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // y = s·R(θ)·(x − c) + c + t, parameters [s, θ, tx, ty]:
        //   ∂y/∂s = R(θ)·(x − c)                  (unscaled rotation)
        //   ∂y/∂θ = s·R'(θ)·(x − c)
        //   ∂y/∂t = I.
        let (ca, sa) = (self.angle.cos(), self.angle.sin());
        let (dx, dy) = (point[0] - self.center[0], point[1] - self.center[1]);
        // Row-major 2×4: columns [s, θ, tx, ty].
        vec![
            ca * dx - sa * dy,
            (-sa * dx - ca * dy) * self.scale,
            1.0,
            0.0,
            sa * dx + ca * dy,
            (ca * dx - sa * dy) * self.scale,
            0.0,
            1.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_transforms_point() {
        let t = TranslationTransform::new(vec![2.0, -3.0]);
        assert_eq!(t.transform_point(&[10.0, 10.0]), vec![12.0, 7.0]);
    }

    #[test]
    fn affine_identity_is_noop() {
        let a = AffineTransform::identity(2);
        assert_eq!(a.transform_point(&[3.0, 4.0]), vec![3.0, 4.0]);
    }

    #[test]
    fn affine_pure_translation_matches_translation_transform() {
        let a = AffineTransform::new(2, matrix::identity(2), vec![5.0, -2.0], vec![0.0, 0.0]);
        assert_eq!(a.transform_point(&[1.0, 1.0]), vec![6.0, -1.0]);
    }

    #[test]
    fn affine_rotation_about_center_fixes_center() {
        // 90-degree rotation about center (5,5): the center maps to itself.
        let a = AffineTransform::new(2, vec![0.0, -1.0, 1.0, 0.0], vec![0.0, 0.0], vec![5.0, 5.0]);
        let c = a.transform_point(&[5.0, 5.0]);
        assert!((c[0] - 5.0).abs() < 1e-12 && (c[1] - 5.0).abs() < 1e-12);
        // (6,5) rotates to (5,6) about (5,5).
        let p = a.transform_point(&[6.0, 5.0]);
        assert!(
            (p[0] - 5.0).abs() < 1e-12 && (p[1] - 6.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn translation_parameters_roundtrip_and_jacobian_is_identity() {
        let mut t = TranslationTransform::new(vec![0.0, 0.0]);
        assert_eq!(t.number_of_parameters(), 2);
        t.set_parameters(&[3.0, -4.0]);
        assert_eq!(t.parameters(), vec![3.0, -4.0]);
        assert_eq!(t.transform_point(&[1.0, 1.0]), vec![4.0, -3.0]);
        // Jacobian is the 2x2 identity regardless of the point.
        assert_eq!(
            t.jacobian_wrt_parameters(&[7.0, 9.0]),
            vec![1.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn affine_parameters_are_matrix_then_translation() {
        let a = AffineTransform::new(2, vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], vec![0.0, 0.0]);
        assert_eq!(a.number_of_parameters(), 6);
        assert_eq!(a.parameters(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn affine_set_parameters_updates_offset() {
        let mut a = AffineTransform::identity(2);
        // Set matrix to identity, translation to (5,-2), center at origin.
        a.set_parameters(&[1.0, 0.0, 0.0, 1.0, 5.0, -2.0]);
        assert_eq!(a.transform_point(&[1.0, 1.0]), vec![6.0, -1.0]);
    }

    #[test]
    fn affine_jacobian_matches_analytic_form() {
        // Center (5,5); at point (7,3): matrix-col entries are (x−c), translation 1.
        let a = AffineTransform::new(2, matrix::identity(2), vec![0.0, 0.0], vec![5.0, 5.0]);
        let j = a.jacobian_wrt_parameters(&[7.0, 3.0]);
        // Row 0: [x0−c0, x1−c1, 0, 0, 1, 0] = [2, -2, 0, 0, 1, 0]
        // Row 1: [0, 0, x0−c0, x1−c1, 0, 1] = [0, 0, 2, -2, 0, 1]
        assert_eq!(
            j,
            vec![2.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 1.0]
        );
    }

    #[test]
    fn affine_jacobian_is_finite_difference_consistent() {
        // Numerically verify ∂y/∂p against the analytic Jacobian.
        let base = vec![0.9, 0.1, -0.2, 1.1, 0.3, -0.4];
        let center = vec![2.0, -1.0];
        let point = [4.0, 5.0];
        let mut a = AffineTransform::new(2, base[..4].to_vec(), base[4..].to_vec(), center.clone());
        let jac = a.jacobian_wrt_parameters(&point);
        let nparams = a.number_of_parameters();
        let h = 1e-6;
        for k in 0..nparams {
            let mut pp = base.clone();
            pp[k] += h;
            a.set_parameters(&pp);
            let yp = a.transform_point(&point);
            let mut pm = base.clone();
            pm[k] -= h;
            a.set_parameters(&pm);
            let ym = a.transform_point(&point);
            for i in 0..2 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * nparams + k]).abs() < 1e-6,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * nparams + k]
                );
            }
        }
    }

    #[test]
    fn euler2d_identity_is_noop() {
        let e = Euler2DTransform::identity();
        assert_eq!(e.number_of_parameters(), 3);
        assert_eq!(e.transform_point(&[3.0, 4.0]), vec![3.0, 4.0]);
    }

    #[test]
    fn euler2d_rotation_about_center_fixes_center() {
        use std::f64::consts::FRAC_PI_2;
        // 90° CCW about center (5,5): the center maps to itself,
        let e = Euler2DTransform::new(FRAC_PI_2, [0.0, 0.0], [5.0, 5.0]);
        let c = e.transform_point(&[5.0, 5.0]);
        assert!(
            (c[0] - 5.0).abs() < 1e-12 && (c[1] - 5.0).abs() < 1e-12,
            "{c:?}"
        );
        // and (6,5) rotates to (5,6): R=[[0,-1],[1,0]], (x−c)=(1,0) ⇒ (0,1) ⇒ +c.
        let p = e.transform_point(&[6.0, 5.0]);
        assert!(
            (p[0] - 5.0).abs() < 1e-12 && (p[1] - 6.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn euler2d_pure_translation_when_angle_is_zero() {
        let e = Euler2DTransform::new(0.0, [5.0, -2.0], [3.0, 7.0]);
        // Zero angle ⇒ R = I ⇒ y = x + t regardless of center.
        assert_eq!(e.transform_point(&[1.0, 1.0]), vec![6.0, -1.0]);
    }

    #[test]
    fn euler2d_parameters_are_angle_then_translation() {
        let mut e = Euler2DTransform::new(0.1, [0.0, 0.0], [2.0, -1.0]);
        e.set_parameters(&[0.5, 3.0, -4.0]);
        assert_eq!(e.parameters(), vec![0.5, 3.0, -4.0]);
        assert_eq!(e.angle(), 0.5);
        assert_eq!(e.translation(), &[3.0, -4.0]);
    }

    #[test]
    fn euler2d_jacobian_matches_itk_analytic_form() {
        // At angle 0, center (5,5), point (7,3): (x−c) = (2,−2).
        //   ∂y/∂θ = R'(0)·(x−c) = [[0,−1],[1,0]]·(2,−2) = (2, 2)
        //   ∂y/∂t = I
        let e = Euler2DTransform::new(0.0, [0.0, 0.0], [5.0, 5.0]);
        let j = e.jacobian_wrt_parameters(&[7.0, 3.0]);
        // Row 0: [∂y0/∂θ, 1, 0]; Row 1: [∂y1/∂θ, 0, 1].
        assert_eq!(j, vec![2.0, 1.0, 0.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn euler2d_jacobian_is_finite_difference_consistent() {
        let base = [0.3, 0.5, -0.7];
        let center = [2.0, -1.0];
        let point = [4.0, 5.0];
        let mut e = Euler2DTransform::new(base[0], [base[1], base[2]], center);
        let jac = e.jacobian_wrt_parameters(&point);
        let nparams = e.number_of_parameters();
        let h = 1e-6;
        for k in 0..nparams {
            let mut pp = base;
            pp[k] += h;
            e.set_parameters(&pp);
            let yp = e.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            e.set_parameters(&pm);
            let ym = e.transform_point(&point);
            for i in 0..2 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * nparams + k]).abs() < 1e-6,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * nparams + k]
                );
            }
        }
    }

    #[test]
    fn similarity2d_identity_is_noop() {
        let s = Similarity2DTransform::identity();
        assert_eq!(s.number_of_parameters(), 4);
        assert_eq!(s.parameters(), vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.transform_point(&[3.0, 4.0]), vec![3.0, 4.0]);
    }

    #[test]
    fn similarity2d_scales_about_center() {
        // Scale 2 about center (5,5), no rotation/translation: (6,5) ⇒ (7,5).
        let s = Similarity2DTransform::new(2.0, 0.0, [0.0, 0.0], [5.0, 5.0]);
        let c = s.transform_point(&[5.0, 5.0]);
        assert!(
            (c[0] - 5.0).abs() < 1e-12 && (c[1] - 5.0).abs() < 1e-12,
            "{c:?}"
        );
        let p = s.transform_point(&[6.0, 5.0]);
        assert!(
            (p[0] - 7.0).abs() < 1e-12 && (p[1] - 5.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn similarity2d_scaled_rotation_about_center() {
        use std::f64::consts::FRAC_PI_2;
        // Scale 2 + 90° about (5,5): (x−c)=(1,0) ⇒ R gives (0,1) ⇒ ×2 ⇒ (0,2) ⇒ +c.
        let s = Similarity2DTransform::new(2.0, FRAC_PI_2, [0.0, 0.0], [5.0, 5.0]);
        let p = s.transform_point(&[6.0, 5.0]);
        assert!(
            (p[0] - 5.0).abs() < 1e-12 && (p[1] - 7.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn similarity2d_parameters_are_scale_angle_translation() {
        let mut s = Similarity2DTransform::new(1.0, 0.0, [0.0, 0.0], [2.0, -1.0]);
        s.set_parameters(&[1.5, 0.5, 3.0, -4.0]);
        assert_eq!(s.parameters(), vec![1.5, 0.5, 3.0, -4.0]);
        assert_eq!(s.scale(), 1.5);
        assert_eq!(s.angle(), 0.5);
        assert_eq!(s.translation(), &[3.0, -4.0]);
    }

    #[test]
    fn similarity2d_jacobian_matches_itk_analytic_form() {
        // Scale 2, angle 0, center (5,5), point (7,3): (x−c)=(2,−2).
        //   ∂y/∂s = R(0)·(x−c) = (2, −2)
        //   ∂y/∂θ = s·R'(0)·(x−c) = 2·(2, 2) = (4, 4)
        //   ∂y/∂t = I
        let s = Similarity2DTransform::new(2.0, 0.0, [0.0, 0.0], [5.0, 5.0]);
        let j = s.jacobian_wrt_parameters(&[7.0, 3.0]);
        // Row 0: [∂y0/∂s, ∂y0/∂θ, 1, 0]; Row 1: [∂y1/∂s, ∂y1/∂θ, 0, 1].
        assert_eq!(j, vec![2.0, 4.0, 1.0, 0.0, -2.0, 4.0, 0.0, 1.0]);
    }

    #[test]
    fn similarity2d_jacobian_is_finite_difference_consistent() {
        let base = [1.3, 0.4, 0.5, -0.7];
        let center = [2.0, -1.0];
        let point = [4.0, 5.0];
        let mut s = Similarity2DTransform::new(base[0], base[1], [base[2], base[3]], center);
        let jac = s.jacobian_wrt_parameters(&point);
        let nparams = s.number_of_parameters();
        let h = 1e-6;
        for k in 0..nparams {
            let mut pp = base;
            pp[k] += h;
            s.set_parameters(&pp);
            let yp = s.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            s.set_parameters(&pm);
            let ym = s.transform_point(&point);
            for i in 0..2 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * nparams + k]).abs() < 1e-6,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * nparams + k]
                );
            }
        }
    }
}
