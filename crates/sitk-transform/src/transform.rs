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

    /// Whether the transform has *local support*: each point of space is
    /// governed by its own small block of parameters rather than by all of them
    /// (mirrors `itk::Transform::GetTransformCategory() == DisplacementField`,
    /// which is exactly what `ObjectToObjectMetric::HasLocalSupport()` keys on).
    /// A dense displacement field is the archetype; every global transform
    /// (translation, affine, versor, B-spline) returns `false`.
    ///
    /// Metrics use this to select a per-region derivative accumulation that
    /// avoids materializing the full (bins² × `number_of_parameters`) derivative
    /// array. The default is `false`.
    fn has_local_support(&self) -> bool {
        false
    }

    /// Number of parameters governing each local region — ITK's
    /// `GetNumberOfLocalParameters`. For a global transform this is just
    /// [`number_of_parameters`]; for a displacement field it is the point
    /// [`dimension`] (one displacement vector per pixel).
    ///
    /// [`number_of_parameters`]: ParametricTransform::number_of_parameters
    /// [`dimension`]: Transform::dimension
    fn number_of_local_parameters(&self) -> usize {
        self.number_of_parameters()
    }

    /// For a [local-support] transform, the `(offset, local_jacobian)` of the
    /// region containing `point`: `offset` is the start index of that region's
    /// parameter block in the flat parameter vector, and `local_jacobian` is the
    /// row-major `dimension × number_of_local_parameters` Jacobian of the mapped
    /// point with respect to *only* that block. Returns `None` when `point` lies
    /// outside the region the transform can influence, or for a global transform
    /// (the default).
    ///
    /// This is the crate's analogue of pairing ITK's
    /// `ComputeParameterOffsetFromVirtualIndex` with the local
    /// `ComputeJacobianWithRespectToParameters`; it lets a metric read one
    /// region's contribution without ever building the dense Jacobian.
    ///
    /// [local-support]: ParametricTransform::has_local_support
    fn local_support_jacobian(&self, _point: &[f64]) -> Option<(usize, Vec<f64>)> {
        None
    }
}

/// A transform with a fixed center of rotation and a translation that can be set
/// independently of the parameter vector, mirroring
/// `itk::MatrixOffsetTransformBase::SetCenter` / `SetTranslation`. This is the
/// interface `CenteredTransformInitializer` configures; a pure
/// [`TranslationTransform`] has no center and so does not implement it.
pub trait CenteredTransform: Transform {
    /// Set the fixed center of rotation (length = [`dimension`]). The matrix and
    /// translation are unchanged; the applied offset is recomputed.
    ///
    /// [`dimension`]: Transform::dimension
    fn set_center(&mut self, center: &[f64]);

    /// Set the translation (length = [`dimension`]). The matrix and center are
    /// unchanged; the applied offset is recomputed.
    ///
    /// [`dimension`]: Transform::dimension
    fn set_translation(&mut self, translation: &[f64]);
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

impl CenteredTransform for AffineTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), self.dim, "center length");
        self.center.copy_from_slice(center);
        self.offset = Self::compute_offset(self.dim, &self.matrix, &self.translation, &self.center);
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), self.dim, "translation length");
        self.translation.copy_from_slice(translation);
        self.offset = Self::compute_offset(self.dim, &self.matrix, &self.translation, &self.center);
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

impl CenteredTransform for Euler2DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 2, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 2, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
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

impl CenteredTransform for Similarity2DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 2, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 2, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
    }
}

/// A rigid 3-D transform parameterized by Euler angles:
/// `y = R·(x − center) + center + translation`, mirroring `itk::Euler3DTransform`.
///
/// Parameters are `[angleX, angleY, angleZ, tx, ty, tz]` (angles in radians); the
/// `center` is fixed. The rotation composes the per-axis rotations `Rx`, `Ry`,
/// `Rz`; the order is `Rz·Rx·Ry` by default (`compute_zyx = false`, ITK's default
/// and VTK order) and `Rz·Ry·Rx` when [`set_compute_zyx(true)`]. The matrix and
/// the equivalent `offset` in `y = M·x + offset`
/// (`offset = translation + center − M·center`) are cached.
///
/// [`set_compute_zyx(true)`]: Euler3DTransform::set_compute_zyx
#[derive(Clone, Debug, PartialEq)]
pub struct Euler3DTransform {
    angle_x: f64,
    angle_y: f64,
    angle_z: f64,
    compute_zyx: bool,
    /// Length 3.
    translation: Vec<f64>,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 rotation.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl Euler3DTransform {
    /// A rigid transform rotating by the Euler angles `(angle_x, angle_y,
    /// angle_z)` (radians) about `center`, then `translation`. Uses ITK's default
    /// composition order `Rz·Rx·Ry` (`compute_zyx = false`).
    pub fn new(
        angle_x: f64,
        angle_y: f64,
        angle_z: f64,
        translation: [f64; 3],
        center: [f64; 3],
    ) -> Self {
        let mut t = Self {
            angle_x,
            angle_y,
            angle_z,
            compute_zyx: false,
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 9],
            offset: vec![0.0; 3],
        };
        t.recompute();
        t
    }

    /// The identity transform (zero angles/translation, center at origin).
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    }

    /// Rotation angle about the X axis (radians).
    pub fn angle_x(&self) -> f64 {
        self.angle_x
    }

    /// Rotation angle about the Y axis (radians).
    pub fn angle_y(&self) -> f64 {
        self.angle_y
    }

    /// Rotation angle about the Z axis (radians).
    pub fn angle_z(&self) -> f64 {
        self.angle_z
    }

    /// Whether the composition order is `Rz·Ry·Rx` (`true`) or `Rz·Rx·Ry`
    /// (`false`, the default).
    pub fn compute_zyx(&self) -> bool {
        self.compute_zyx
    }

    /// Select the rotation composition order: `Rz·Ry·Rx` when `flag`, else
    /// `Rz·Rx·Ry`. Recomputes the matrix and offset.
    pub fn set_compute_zyx(&mut self, flag: bool) {
        self.compute_zyx = flag;
        self.recompute();
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 3×3 rotation matrix.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Rebuild the cached matrix (per-axis rotations composed in the configured
    /// order) and offset.
    fn recompute(&mut self) {
        let (cx, sx) = (self.angle_x.cos(), self.angle_x.sin());
        let (cy, sy) = (self.angle_y.cos(), self.angle_y.sin());
        let (cz, sz) = (self.angle_z.cos(), self.angle_z.sin());

        #[rustfmt::skip]
        let rx = [1.0, 0.0, 0.0,  0.0, cx, -sx,  0.0, sx, cx];
        #[rustfmt::skip]
        let ry = [cy, 0.0, sy,  0.0, 1.0, 0.0,  -sy, 0.0, cy];
        #[rustfmt::skip]
        let rz = [cz, -sz, 0.0,  sz, cz, 0.0,  0.0, 0.0, 1.0];

        // ITK ComputeMatrix: Rz·Rx·Ry by default, Rz·Ry·Rx when compute_zyx.
        self.matrix = if self.compute_zyx {
            matrix::matmul(&matrix::matmul(&rz, &ry, 3), &rx, 3)
        } else {
            matrix::matmul(&matrix::matmul(&rz, &rx, 3), &ry, 3)
        };

        let m_center = matrix::mat_vec(&self.matrix, &self.center, 3);
        self.offset = (0..3)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
    }
}

impl Transform for Euler3DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for Euler3DTransform {
    fn number_of_parameters(&self) -> usize {
        6
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.angle_x,
            self.angle_y,
            self.angle_z,
            self.translation[0],
            self.translation[1],
            self.translation[2],
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 6, "parameter length");
        self.angle_x = params[0];
        self.angle_y = params[1];
        self.angle_z = params[2];
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // Analytic ∂y/∂angle from itk::Euler3DTransform, plus an identity block
        // for the translation. Row-major 3×6.
        let (cx, sx) = (self.angle_x.cos(), self.angle_x.sin());
        let (cy, sy) = (self.angle_y.cos(), self.angle_y.sin());
        let (cz, sz) = (self.angle_z.cos(), self.angle_z.sin());
        let (px, py, pz) = (
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        );

        let mut j = vec![0.0f64; 18];
        if self.compute_zyx {
            j[0] = (cz * sy * cx + sz * sx) * py + (-cz * sy * sx + sz * cx) * pz;
            j[6] = (sz * sy * cx - cz * sx) * py + (-sz * sy * sx - cz * cx) * pz;
            j[12] = (cy * cx) * py + (-cy * sx) * pz;

            j[1] = (-cz * sy) * px + (cz * cy * sx) * py + (cz * cy * cx) * pz;
            j[7] = (-sz * sy) * px + (sz * cy * sx) * py + (sz * cy * cx) * pz;
            j[13] = (-cy) * px + (-sy * sx) * py + (-sy * cx) * pz;

            j[2] =
                (-sz * cy) * px + (-sz * sy * sx - cz * cx) * py + (-sz * sy * cx + cz * sx) * pz;
            j[8] = (cz * cy) * px + (cz * sy * sx - sz * cx) * py + (cz * sy * cx + sz * sx) * pz;
            j[14] = 0.0;
        } else {
            j[0] = (-sz * cx * sy) * px + (sz * sx) * py + (sz * cx * cy) * pz;
            j[6] = (cz * cx * sy) * px + (-cz * sx) * py + (-cz * cx * cy) * pz;
            j[12] = (sx * sy) * px + (cx) * py + (-sx * cy) * pz;

            j[1] = (-cz * sy - sz * sx * cy) * px + (cz * cy - sz * sx * sy) * pz;
            j[7] = (-sz * sy + cz * sx * cy) * px + (sz * cy + cz * sx * sy) * pz;
            j[13] = (-cx * cy) * px + (-cx * sy) * pz;

            j[2] =
                (-sz * cy - cz * sx * sy) * px + (-cz * cx) * py + (-sz * sy + cz * sx * cy) * pz;
            j[8] = (cz * cy - sz * sx * sy) * px + (-sz * cx) * py + (cz * sy + sz * sx * cy) * pz;
            j[14] = 0.0;
        }
        // Translation identity block: ∂yᵢ/∂t_i = 1 at columns 3, 4, 5.
        j[3] = 1.0;
        j[10] = 1.0;
        j[17] = 1.0;
        j
    }
}

impl CenteredTransform for Euler3DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 3, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 3, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
    }
}

/// A rigid 3-D transform parameterized by a versor (unit-quaternion) rotation:
/// `y = R·(x − center) + center + translation`, mirroring
/// `itk::VersorRigid3DTransform`.
///
/// Parameters are `[vx, vy, vz, tx, ty, tz]`, where `(vx, vy, vz)` is the
/// versor's **right part** — the rotation axis scaled by `sin(θ/2)`. The scalar
/// part is the dependent `w = √(1 − vx² − vy² − vz²)`, so the rotation is encoded
/// by three numbers with no gimbal lock. As in ITK, a right part with norm `≥ 1`
/// is scaled just under 1 before use (`Versor::Set` requires `‖v‖ ≤ 1`); the
/// `center` is fixed and the matrix/offset are cached.
///
/// The rotation Jacobian is divided by `w` and so is singular at `θ = π`
/// (`w = 0`) — a property of ITK's analytic form, not this port; registration
/// stays well away from it.
#[derive(Clone, Debug, PartialEq)]
pub struct VersorRigid3DTransform {
    /// Normalized versor right part.
    vx: f64,
    vy: f64,
    vz: f64,
    /// Normalized versor scalar part `√(1 − vx² − vy² − vz²)`.
    vw: f64,
    /// Length 3.
    translation: Vec<f64>,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 rotation.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl VersorRigid3DTransform {
    /// A rigid transform whose rotation has versor right part `(vx, vy, vz)`
    /// (axis·sin(θ/2)), about `center`, then `translation`. A right part with
    /// norm `≥ 1` is scaled to just under 1, matching ITK's `SetParameters`.
    pub fn new(vx: f64, vy: f64, vz: f64, translation: [f64; 3], center: [f64; 3]) -> Self {
        let mut t = Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            vw: 1.0,
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 9],
            offset: vec![0.0; 3],
        };
        t.set_versor(vx, vy, vz);
        t.recompute();
        t
    }

    /// The identity transform (versor `(0,0,0; w=1)`, zero translation, center at
    /// origin).
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    }

    /// Versor right-part X (axis·sin(θ/2)).
    pub fn versor_x(&self) -> f64 {
        self.vx
    }

    /// Versor right-part Y.
    pub fn versor_y(&self) -> f64 {
        self.vy
    }

    /// Versor right-part Z.
    pub fn versor_z(&self) -> f64 {
        self.vz
    }

    /// Versor scalar part `w = √(1 − vx² − vy² − vz²)`.
    pub fn versor_w(&self) -> f64 {
        self.vw
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 3×3 rotation matrix.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Set the normalized versor from a right part, mirroring
    /// `itk::VersorRigid3DTransform::SetParameters` + `Versor::Set(axis)`: scale a
    /// right part of norm `≥ 1 − ε` to just under 1, then `w = √(1 − ‖v‖²)`.
    fn set_versor(&mut self, vx: f64, vy: f64, vz: f64) {
        const EPS: f64 = 1e-10;
        let norm = (vx * vx + vy * vy + vz * vz).sqrt();
        let (ax, ay, az) = if norm >= 1.0 - EPS {
            let d = norm + EPS * norm;
            (vx / d, vy / d, vz / d)
        } else {
            (vx, vy, vz)
        };
        self.vx = ax;
        self.vy = ay;
        self.vz = az;
        self.vw = (1.0 - (ax * ax + ay * ay + az * az)).max(0.0).sqrt();
    }

    /// Rebuild the cached rotation matrix (`itk::Versor::GetMatrix`) and offset.
    fn recompute(&mut self) {
        let (x, y, z, w) = (self.vx, self.vy, self.vz, self.vw);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, xw) = (x * y, x * z, x * w);
        let (yz, yw, zw) = (y * z, y * w, z * w);
        #[rustfmt::skip]
        let m = vec![
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw),       2.0 * (xz + yw),
            2.0 * (xy + zw),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw),
            2.0 * (xz - yw),       2.0 * (yz + xw),       1.0 - 2.0 * (xx + yy),
        ];
        let m_center = matrix::mat_vec(&m, &self.center, 3);
        self.offset = (0..3)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
        self.matrix = m;
    }
}

impl Transform for VersorRigid3DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for VersorRigid3DTransform {
    fn number_of_parameters(&self) -> usize {
        6
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.vx,
            self.vy,
            self.vz,
            self.translation[0],
            self.translation[1],
            self.translation[2],
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 6, "parameter length");
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // Analytic ∂y/∂versor from itk::VersorRigid3DTransform (divided by vw),
        // plus the translation identity block. Row-major 3×6.
        let (vx, vy, vz, vw) = (self.vx, self.vy, self.vz, self.vw);
        let (px, py, pz) = (
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        );
        let (vxx, vyy, vzz, vww) = (vx * vx, vy * vy, vz * vz, vw * vw);
        let (vxy, vxz, vxw) = (vx * vy, vx * vz, vx * vw);
        let (vyz, vyw, vzw) = (vy * vz, vy * vw, vz * vw);

        let mut j = vec![0.0f64; 18];
        j[0] = 2.0 * ((vyw + vxz) * py + (vzw - vxy) * pz) / vw;
        j[6] = 2.0 * ((vyw - vxz) * px - 2.0 * vxw * py + (vxx - vww) * pz) / vw;
        j[12] = 2.0 * ((vzw + vxy) * px + (vww - vxx) * py - 2.0 * vxw * pz) / vw;

        j[1] = 2.0 * (-2.0 * vyw * px + (vxw + vyz) * py + (vww - vyy) * pz) / vw;
        j[7] = 2.0 * ((vxw - vyz) * px + (vzw + vxy) * pz) / vw;
        j[13] = 2.0 * ((vyy - vww) * px + (vzw - vxy) * py - 2.0 * vyw * pz) / vw;

        j[2] = 2.0 * (-2.0 * vzw * px + (vzz - vww) * py + (vxw - vyz) * pz) / vw;
        j[8] = 2.0 * ((vww - vzz) * px - 2.0 * vzw * py + (vyw + vxz) * pz) / vw;
        j[14] = 2.0 * ((vxw + vyz) * px + (vyw - vxz) * py) / vw;

        // Translation identity block: columns 3, 4, 5.
        j[3] = 1.0;
        j[10] = 1.0;
        j[17] = 1.0;
        j
    }
}

impl CenteredTransform for VersorRigid3DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 3, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 3, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
    }
}

/// A similarity 3-D transform `y = s·R(versor)·(x − center) + center + translation`,
/// mirroring `itk::Similarity3DTransform` — a versor rotation plus an isotropic
/// `scale`, the 3-D analog of [`Similarity2DTransform`].
///
/// Parameters are `[vx, vy, vz, tx, ty, tz, scale]` (ITK's 3-D order, scale last —
/// unlike `Similarity2DTransform`, whose ITK order puts scale first). `(vx, vy, vz)`
/// is the versor right part (axis·sin(θ/2)) with the same norm-clamping as
/// [`VersorRigid3DTransform`]; the `center` is fixed. The matrix `M = s·R` and the
/// equivalent `offset = translation + center − M·center` (in `y = M·x + offset`)
/// are cached and refreshed whenever the parameters change.
///
/// The versor-rotation Jacobian columns are divided by the versor scalar part `w`
/// (and scaled by `s`), so — as in ITK — they are singular at `θ = π` (`w = 0`);
/// registration stays away from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Similarity3DTransform {
    /// Normalized versor right part.
    vx: f64,
    vy: f64,
    vz: f64,
    /// Normalized versor scalar part `√(1 − vx² − vy² − vz²)`.
    vw: f64,
    /// Isotropic scale factor.
    scale: f64,
    /// Length 3.
    translation: Vec<f64>,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 `s·R`.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl Similarity3DTransform {
    /// A similarity transform: rotate by versor right part `(vx, vy, vz)`
    /// (axis·sin(θ/2)) about `center`, scale by `scale`, then `translation`. A
    /// right part with norm `≥ 1` is scaled to just under 1, matching ITK.
    pub fn new(
        scale: f64,
        vx: f64,
        vy: f64,
        vz: f64,
        translation: [f64; 3],
        center: [f64; 3],
    ) -> Self {
        let mut t = Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            vw: 1.0,
            scale,
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 9],
            offset: vec![0.0; 3],
        };
        t.set_versor(vx, vy, vz);
        t.recompute();
        t
    }

    /// The identity transform (scale 1, versor `(0,0,0; w=1)`, zero translation,
    /// center at origin).
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    }

    /// Versor right-part X (axis·sin(θ/2)).
    pub fn versor_x(&self) -> f64 {
        self.vx
    }

    /// Versor right-part Y.
    pub fn versor_y(&self) -> f64 {
        self.vy
    }

    /// Versor right-part Z.
    pub fn versor_z(&self) -> f64 {
        self.vz
    }

    /// Versor scalar part `w = √(1 − vx² − vy² − vz²)`.
    pub fn versor_w(&self) -> f64 {
        self.vw
    }

    /// Isotropic scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 3×3 matrix `s·R`.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Set the normalized versor from a right part, mirroring
    /// `itk::Similarity3DTransform::SetParameters` (same clamp as
    /// [`VersorRigid3DTransform`]): scale a right part of norm `≥ 1 − ε` to just
    /// under 1, then `w = √(1 − ‖v‖²)`.
    fn set_versor(&mut self, vx: f64, vy: f64, vz: f64) {
        const EPS: f64 = 1e-10;
        let norm = (vx * vx + vy * vy + vz * vz).sqrt();
        let (ax, ay, az) = if norm >= 1.0 - EPS {
            let d = norm + EPS * norm;
            (vx / d, vy / d, vz / d)
        } else {
            (vx, vy, vz)
        };
        self.vx = ax;
        self.vy = ay;
        self.vz = az;
        self.vw = (1.0 - (ax * ax + ay * ay + az * az)).max(0.0).sqrt();
    }

    /// Rebuild the cached matrix `M = s·R(versor)` (`itk::Versor::GetMatrix`
    /// scaled, as in `ComputeMatrix`) and the offset.
    fn recompute(&mut self) {
        let (x, y, z, w) = (self.vx, self.vy, self.vz, self.vw);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, xw) = (x * y, x * z, x * w);
        let (yz, yw, zw) = (y * z, y * w, z * w);
        let s = self.scale;
        #[rustfmt::skip]
        let m = vec![
            s * (1.0 - 2.0 * (yy + zz)), s * 2.0 * (xy - zw),         s * 2.0 * (xz + yw),
            s * 2.0 * (xy + zw),         s * (1.0 - 2.0 * (xx + zz)), s * 2.0 * (yz - xw),
            s * 2.0 * (xz - yw),         s * 2.0 * (yz + xw),         s * (1.0 - 2.0 * (xx + yy)),
        ];
        let m_center = matrix::mat_vec(&m, &self.center, 3);
        self.offset = (0..3)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
        self.matrix = m;
    }
}

impl Transform for Similarity3DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for Similarity3DTransform {
    fn number_of_parameters(&self) -> usize {
        7
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.vx,
            self.vy,
            self.vz,
            self.translation[0],
            self.translation[1],
            self.translation[2],
            self.scale,
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 7, "parameter length");
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale = params[6];
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // itk::Similarity3DTransform::ComputeJacobianWithRespectToParameters:
        //   cols 0..2 (versor) = the VersorRigid3D rotation Jacobian (÷w) × scale,
        //   cols 3..5 (translation) = identity,
        //   col 6 (scale) = (M·(p − center)) / scale = R·(p − center).
        let (vx, vy, vz, vw) = (self.vx, self.vy, self.vz, self.vw);
        let s = self.scale;
        let (px, py, pz) = (
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        );
        let (vxx, vyy, vzz, vww) = (vx * vx, vy * vy, vz * vz, vw * vw);
        let (vxy, vxz, vxw) = (vx * vy, vx * vz, vx * vw);
        let (vyz, vyw, vzw) = (vy * vz, vy * vw, vz * vw);

        // Row-major 3×7.
        let mut j = vec![0.0f64; 21];
        j[0] = s * 2.0 * ((vyw + vxz) * py + (vzw - vxy) * pz) / vw;
        j[7] = s * 2.0 * ((vyw - vxz) * px - 2.0 * vxw * py + (vxx - vww) * pz) / vw;
        j[14] = s * 2.0 * ((vzw + vxy) * px + (vww - vxx) * py - 2.0 * vxw * pz) / vw;

        j[1] = s * 2.0 * (-2.0 * vyw * px + (vxw + vyz) * py + (vww - vyy) * pz) / vw;
        j[8] = s * 2.0 * ((vxw - vyz) * px + (vzw + vxy) * pz) / vw;
        j[15] = s * 2.0 * ((vyy - vww) * px + (vzw - vxy) * py - 2.0 * vyw * pz) / vw;

        j[2] = s * 2.0 * (-2.0 * vzw * px + (vzz - vww) * py + (vxw - vyz) * pz) / vw;
        j[9] = s * 2.0 * ((vww - vzz) * px - 2.0 * vzw * py + (vyw + vxz) * pz) / vw;
        j[16] = s * 2.0 * ((vxw + vyz) * px + (vyw - vxz) * py) / vw;

        // Translation identity block: columns 3, 4, 5.
        j[3] = 1.0;
        j[11] = 1.0;
        j[19] = 1.0;

        // Scale column 6: (M·pp) / scale.
        let mpp = matrix::mat_vec(&self.matrix, &[px, py, pz], 3);
        j[6] = mpp[0] / s;
        j[13] = mpp[1] / s;
        j[20] = mpp[2] / s;
        j
    }
}

impl CenteredTransform for Similarity3DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 3, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 3, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
    }
}

/// A 3-D transform with a versor rotation, an **anisotropic** per-axis scale, and
/// translation (`itk::ScaleVersor3DTransform`) — 9 DOF, richer than
/// [`Similarity3DTransform`]'s single isotropic scale.
///
/// Parameters are `[vx, vy, vz, tx, ty, tz, sx, sy, sz]`: versor right part (3),
/// translation (3), per-axis scale (3). `(vx, vy, vz)` uses the same norm-clamping
/// as [`VersorRigid3DTransform`]; the `center` is fixed.
///
/// # Matrix (ITK's additive form, **not** `R·diag(scale)`)
///
/// ITK builds the matrix as the rotation with `(scaleᵢ − 1)` **added** to each
/// diagonal entry — `ComputeMatrix` calls the versor superclass then does
/// `M[i][i] += scaleᵢ − 1` — so
///
/// ```text
/// M = R(versor) + diag(sx − 1, sy − 1, sz − 1)
/// ```
///
/// This equals `diag(scale)` only when `R = I`; for a non-identity rotation it is
/// an additive, not multiplicative, scale (a quirk inherited from
/// `ScaleSkewVersor3DTransform`). The offset is `translation + center − M·center`.
///
/// # Jacobian
///
/// Because the scale enters only the (constant-w.r.t.-versor) diagonal, the versor
/// columns are exactly [`VersorRigid3DTransform`]'s (divided by `w`, **no** scale
/// factor — unlike `Similarity3DTransform`), the translation columns are the
/// identity, and the scale column `k` is diagonal: `∂yₖ/∂sₖ = (p − center)ₖ`,
/// off-diagonal zero. The versor columns share ITK's `θ = π` (`w = 0`) singularity.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleVersor3DTransform {
    /// Normalized versor right part.
    vx: f64,
    vy: f64,
    vz: f64,
    /// Normalized versor scalar part `√(1 − vx² − vy² − vz²)`.
    vw: f64,
    /// Per-axis scale, length 3.
    scale: Vec<f64>,
    /// Length 3.
    translation: Vec<f64>,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 `R + diag(scale − 1)`.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl ScaleVersor3DTransform {
    /// A transform: rotate by versor right part `(vx, vy, vz)` (axis·sin(θ/2))
    /// about `center`, apply the additive per-axis `scale`, then `translation`. A
    /// right part with norm `≥ 1` is scaled to just under 1, matching ITK.
    pub fn new(
        scale: [f64; 3],
        vx: f64,
        vy: f64,
        vz: f64,
        translation: [f64; 3],
        center: [f64; 3],
    ) -> Self {
        let mut t = Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            vw: 1.0,
            scale: scale.to_vec(),
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 9],
            offset: vec![0.0; 3],
        };
        t.set_versor(vx, vy, vz);
        t.recompute();
        t
    }

    /// The identity transform (scale `(1,1,1)`, versor `(0,0,0; w=1)`, zero
    /// translation, center at origin).
    pub fn identity() -> Self {
        Self::new(
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
            0.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
    }

    /// Versor right-part X (axis·sin(θ/2)).
    pub fn versor_x(&self) -> f64 {
        self.vx
    }

    /// Versor right-part Y.
    pub fn versor_y(&self) -> f64 {
        self.vy
    }

    /// Versor right-part Z.
    pub fn versor_z(&self) -> f64 {
        self.vz
    }

    /// Versor scalar part `w = √(1 − vx² − vy² − vz²)`.
    pub fn versor_w(&self) -> f64 {
        self.vw
    }

    /// The per-axis scale factors.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 3×3 matrix `R + diag(scale − 1)`.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Set the normalized versor from a right part, mirroring
    /// `itk::ScaleVersor3DTransform::SetParameters` (same clamp as
    /// [`VersorRigid3DTransform`]).
    fn set_versor(&mut self, vx: f64, vy: f64, vz: f64) {
        const EPS: f64 = 1e-10;
        let norm = (vx * vx + vy * vy + vz * vz).sqrt();
        let (ax, ay, az) = if norm >= 1.0 - EPS {
            let d = norm + EPS * norm;
            (vx / d, vy / d, vz / d)
        } else {
            (vx, vy, vz)
        };
        self.vx = ax;
        self.vy = ay;
        self.vz = az;
        self.vw = (1.0 - (ax * ax + ay * ay + az * az)).max(0.0).sqrt();
    }

    /// Rebuild the cached matrix `M = R(versor) + diag(scale − 1)` (ITK's
    /// `ComputeMatrix`: versor superclass rotation, then `M[i][i] += scaleᵢ − 1`)
    /// and the offset.
    fn recompute(&mut self) {
        let (x, y, z, w) = (self.vx, self.vy, self.vz, self.vw);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, xw) = (x * y, x * z, x * w);
        let (yz, yw, zw) = (y * z, y * w, z * w);
        #[rustfmt::skip]
        let mut m = vec![
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw),       2.0 * (xz + yw),
            2.0 * (xy + zw),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw),
            2.0 * (xz - yw),       2.0 * (yz + xw),       1.0 - 2.0 * (xx + yy),
        ];
        // Additive per-axis scale on the diagonal (not R·diag(scale)).
        m[0] += self.scale[0] - 1.0;
        m[4] += self.scale[1] - 1.0;
        m[8] += self.scale[2] - 1.0;
        let m_center = matrix::mat_vec(&m, &self.center, 3);
        self.offset = (0..3)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
        self.matrix = m;
    }
}

impl Transform for ScaleVersor3DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for ScaleVersor3DTransform {
    fn number_of_parameters(&self) -> usize {
        9
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.vx,
            self.vy,
            self.vz,
            self.translation[0],
            self.translation[1],
            self.translation[2],
            self.scale[0],
            self.scale[1],
            self.scale[2],
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 9, "parameter length");
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale[0] = params[6];
        self.scale[1] = params[7];
        self.scale[2] = params[8];
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // itk::ScaleVersor3DTransform::ComputeJacobianWithRespectToParameters:
        //   cols 0..2 (versor) = the VersorRigid3D rotation Jacobian (÷w), no
        //     scale factor (scale enters only the constant-w.r.t.-versor diagonal);
        //   cols 3..5 (translation) = identity;
        //   cols 6..8 (scale) = diagonal (p − center): ∂yₖ/∂sₖ = ppₖ.
        let (vx, vy, vz, vw) = (self.vx, self.vy, self.vz, self.vw);
        let (px, py, pz) = (
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        );
        let (vxx, vyy, vzz, vww) = (vx * vx, vy * vy, vz * vz, vw * vw);
        let (vxy, vxz, vxw) = (vx * vy, vx * vz, vx * vw);
        let (vyz, vyw, vzw) = (vy * vz, vy * vw, vz * vw);

        // Row-major 3×9.
        let mut j = vec![0.0f64; 27];
        j[0] = 2.0 * ((vyw + vxz) * py + (vzw - vxy) * pz) / vw;
        j[9] = 2.0 * ((vyw - vxz) * px - 2.0 * vxw * py + (vxx - vww) * pz) / vw;
        j[18] = 2.0 * ((vzw + vxy) * px + (vww - vxx) * py - 2.0 * vxw * pz) / vw;

        j[1] = 2.0 * (-2.0 * vyw * px + (vxw + vyz) * py + (vww - vyy) * pz) / vw;
        j[10] = 2.0 * ((vxw - vyz) * px + (vzw + vxy) * pz) / vw;
        j[19] = 2.0 * ((vyy - vww) * px + (vzw - vxy) * py - 2.0 * vyw * pz) / vw;

        j[2] = 2.0 * (-2.0 * vzw * px + (vzz - vww) * py + (vxw - vyz) * pz) / vw;
        j[11] = 2.0 * ((vww - vzz) * px - 2.0 * vzw * py + (vyw + vxz) * pz) / vw;
        j[20] = 2.0 * ((vxw + vyz) * px + (vyw - vxz) * py) / vw;

        // Translation identity block: columns 3, 4, 5.
        j[3] = 1.0;
        j[13] = 1.0;
        j[23] = 1.0;

        // Scale columns 6, 7, 8: diagonal (p − center).
        j[6] = px;
        j[16] = py;
        j[26] = pz;
        j
    }
}

impl CenteredTransform for ScaleVersor3DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 3, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 3, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
    }
}

/// A 3-D transform with a versor rotation, per-axis scale, **6-component skew**,
/// and translation (`itk::ScaleSkewVersor3DTransform`) — 15 parameters, extending
/// [`ScaleVersor3DTransform`] with the off-diagonal (shear) matrix entries.
///
/// Parameters are `[vx, vy, vz, tx, ty, tz, sx, sy, sz, k0, k1, k2, k3, k4, k5]`:
/// versor right part (3), translation (3), per-axis scale (3), skew (6). `(vx, vy,
/// vz)` uses the same norm-clamping as [`VersorRigid3DTransform`]; `center` fixed.
///
/// # Matrix (ITK's additive form)
///
/// As in [`ScaleVersor3DTransform`], scale and skew are **added** onto the versor
/// rotation — `ComputeMatrix` calls the versor superclass then adds `scaleᵢ − 1` to
/// each diagonal and the six skews to the off-diagonals in the order
/// `{xy, xz, yx, yz, zx, zy}`:
///
/// ```text
/// M = R(versor) + diag(sx−1, sy−1, sz−1) + [ 0  k0 k1 ; k2  0 k3 ; k4 k5  0 ]
/// ```
///
/// The offset is `translation + center − M·center`.
///
/// # Jacobian
///
/// Since scale and skew enter only the (constant-w.r.t.-versor) matrix entries, the
/// versor columns are exactly [`VersorRigid3DTransform`]'s (÷`w`, no scale factor),
/// the translation columns are the identity, the scale column `k` is diagonal
/// (`∂yₖ/∂sₖ = (p − center)ₖ`), and each skew column is the single `(p − center)`
/// component multiplying that off-diagonal entry. The versor columns share ITK's
/// `θ = π` (`w = 0`) singularity.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleSkewVersor3DTransform {
    /// Normalized versor right part.
    vx: f64,
    vy: f64,
    vz: f64,
    /// Normalized versor scalar part `√(1 − vx² − vy² − vz²)`.
    vw: f64,
    /// Per-axis scale, length 3.
    scale: Vec<f64>,
    /// Skew `{xy, xz, yx, yz, zx, zy}`, length 6.
    skew: Vec<f64>,
    /// Length 3.
    translation: Vec<f64>,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 `R + diag(scale − 1) + skew`.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl ScaleSkewVersor3DTransform {
    /// A transform: rotate by versor right part `(vx, vy, vz)` about `center`, add
    /// the per-axis `scale` and 6-component `skew` onto the rotation, then
    /// `translation`. A right part with norm `≥ 1` is scaled to just under 1.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scale: [f64; 3],
        skew: [f64; 6],
        vx: f64,
        vy: f64,
        vz: f64,
        translation: [f64; 3],
        center: [f64; 3],
    ) -> Self {
        let mut t = Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            vw: 1.0,
            scale: scale.to_vec(),
            skew: skew.to_vec(),
            translation: translation.to_vec(),
            center: center.to_vec(),
            matrix: vec![0.0; 9],
            offset: vec![0.0; 3],
        };
        t.set_versor(vx, vy, vz);
        t.recompute();
        t
    }

    /// The identity transform (scale `(1,1,1)`, zero skew, versor `(0,0,0; w=1)`,
    /// zero translation, center at origin).
    pub fn identity() -> Self {
        Self::new(
            [1.0, 1.0, 1.0],
            [0.0; 6],
            0.0,
            0.0,
            0.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
    }

    /// Versor right-part X (axis·sin(θ/2)).
    pub fn versor_x(&self) -> f64 {
        self.vx
    }

    /// Versor right-part Y.
    pub fn versor_y(&self) -> f64 {
        self.vy
    }

    /// Versor right-part Z.
    pub fn versor_z(&self) -> f64 {
        self.vz
    }

    /// Versor scalar part `w = √(1 − vx² − vy² − vz²)`.
    pub fn versor_w(&self) -> f64 {
        self.vw
    }

    /// The per-axis scale factors.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// The skew components `{xy, xz, yx, yz, zx, zy}`.
    pub fn skew(&self) -> &[f64] {
        &self.skew
    }

    /// The translation part.
    pub fn translation(&self) -> &[f64] {
        &self.translation
    }

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 3×3 matrix `R + diag(scale − 1) + skew`.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Set the normalized versor from a right part, mirroring
    /// `itk::ScaleSkewVersor3DTransform::SetParameters` (same clamp as
    /// [`VersorRigid3DTransform`]).
    fn set_versor(&mut self, vx: f64, vy: f64, vz: f64) {
        const EPS: f64 = 1e-10;
        let norm = (vx * vx + vy * vy + vz * vz).sqrt();
        let (ax, ay, az) = if norm >= 1.0 - EPS {
            let d = norm + EPS * norm;
            (vx / d, vy / d, vz / d)
        } else {
            (vx, vy, vz)
        };
        self.vx = ax;
        self.vy = ay;
        self.vz = az;
        self.vw = (1.0 - (ax * ax + ay * ay + az * az)).max(0.0).sqrt();
    }

    /// Rebuild the cached matrix `M = R + diag(scale − 1) + skew` (ITK's
    /// `ComputeMatrix`: versor superclass rotation, then add scale to the diagonal
    /// and the six skews to the off-diagonals in order `{xy, xz, yx, yz, zx, zy}`)
    /// and the offset.
    fn recompute(&mut self) {
        let (x, y, z, w) = (self.vx, self.vy, self.vz, self.vw);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, xw) = (x * y, x * z, x * w);
        let (yz, yw, zw) = (y * z, y * w, z * w);
        #[rustfmt::skip]
        let mut m = vec![
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw),       2.0 * (xz + yw),
            2.0 * (xy + zw),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw),
            2.0 * (xz - yw),       2.0 * (yz + xw),       1.0 - 2.0 * (xx + yy),
        ];
        // Additive scale on the diagonal.
        m[0] += self.scale[0] - 1.0;
        m[4] += self.scale[1] - 1.0;
        m[8] += self.scale[2] - 1.0;
        // Additive skew on the off-diagonals: {xy, xz, yx, yz, zx, zy}.
        m[1] += self.skew[0]; // M[0][1]
        m[2] += self.skew[1]; // M[0][2]
        m[3] += self.skew[2]; // M[1][0]
        m[5] += self.skew[3]; // M[1][2]
        m[6] += self.skew[4]; // M[2][0]
        m[7] += self.skew[5]; // M[2][1]
        let m_center = matrix::mat_vec(&m, &self.center, 3);
        self.offset = (0..3)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
        self.matrix = m;
    }
}

impl Transform for ScaleSkewVersor3DTransform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for ScaleSkewVersor3DTransform {
    fn number_of_parameters(&self) -> usize {
        15
    }

    fn parameters(&self) -> Vec<f64> {
        vec![
            self.vx,
            self.vy,
            self.vz,
            self.translation[0],
            self.translation[1],
            self.translation[2],
            self.scale[0],
            self.scale[1],
            self.scale[2],
            self.skew[0],
            self.skew[1],
            self.skew[2],
            self.skew[3],
            self.skew[4],
            self.skew[5],
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 15, "parameter length");
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale[0] = params[6];
        self.scale[1] = params[7];
        self.scale[2] = params[8];
        self.skew.copy_from_slice(&params[9..15]);
        self.recompute();
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // itk::ScaleSkewVersor3DTransform::ComputeJacobianWithRespectToParameters:
        //   cols 0..2 (versor) = the VersorRigid3D rotation Jacobian (÷w), no scale;
        //   cols 3..5 (translation) = identity;
        //   cols 6..8 (scale) = diagonal (p − center);
        //   cols 9..14 (skew) = the (p − center) component of the off-diagonal entry
        //     each skew fills: {xy, xz, yx, yz, zx, zy}.
        let (vx, vy, vz, vw) = (self.vx, self.vy, self.vz, self.vw);
        let (px, py, pz) = (
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        );
        let (vxx, vyy, vzz, vww) = (vx * vx, vy * vy, vz * vz, vw * vw);
        let (vxy, vxz, vxw) = (vx * vy, vx * vz, vx * vw);
        let (vyz, vyw, vzw) = (vy * vz, vy * vw, vz * vw);

        // Row-major 3×15.
        let mut j = vec![0.0f64; 45];
        j[0] = 2.0 * ((vyw + vxz) * py + (vzw - vxy) * pz) / vw;
        j[15] = 2.0 * ((vyw - vxz) * px - 2.0 * vxw * py + (vxx - vww) * pz) / vw;
        j[30] = 2.0 * ((vzw + vxy) * px + (vww - vxx) * py - 2.0 * vxw * pz) / vw;

        j[1] = 2.0 * (-2.0 * vyw * px + (vxw + vyz) * py + (vww - vyy) * pz) / vw;
        j[16] = 2.0 * ((vxw - vyz) * px + (vzw + vxy) * pz) / vw;
        j[31] = 2.0 * ((vyy - vww) * px + (vzw - vxy) * py - 2.0 * vyw * pz) / vw;

        j[2] = 2.0 * (-2.0 * vzw * px + (vzz - vww) * py + (vxw - vyz) * pz) / vw;
        j[17] = 2.0 * ((vww - vzz) * px - 2.0 * vzw * py + (vyw + vxz) * pz) / vw;
        j[32] = 2.0 * ((vxw + vyz) * px + (vyw - vxz) * py) / vw;

        // Translation identity block: columns 3, 4, 5.
        j[3] = 1.0;
        j[19] = 1.0;
        j[35] = 1.0;

        // Scale columns 6, 7, 8: diagonal (p − center).
        j[6] = px;
        j[22] = py;
        j[38] = pz;

        // Skew columns 9..14: {xy, xz, yx, yz, zx, zy}.
        j[9] = py; // ∂y0/∂k0 (M[0][1])
        j[10] = pz; // ∂y0/∂k1 (M[0][2])
        j[26] = px; // ∂y1/∂k2 (M[1][0])
        j[27] = pz; // ∂y1/∂k3 (M[1][2])
        j[43] = px; // ∂y2/∂k4 (M[2][0])
        j[44] = py; // ∂y2/∂k5 (M[2][1])
        j
    }
}

impl CenteredTransform for ScaleSkewVersor3DTransform {
    fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 3, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    fn set_translation(&mut self, translation: &[f64]) {
        assert_eq!(translation.len(), 3, "translation length");
        self.translation.copy_from_slice(translation);
        self.recompute();
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

    #[test]
    fn centered_transform_set_center_recomputes_offset_keeping_matrix() {
        // Euler2D: after set_center(c), the center maps to itself + translation,
        // and the matrix (rotation) is untouched.
        use std::f64::consts::FRAC_PI_2;
        let mut e = Euler2DTransform::new(FRAC_PI_2, [1.0, 2.0], [0.0, 0.0]);
        let matrix_before = e.matrix().to_vec();
        e.set_center(&[5.0, 5.0]);
        assert_eq!(e.center(), &[5.0, 5.0]);
        assert_eq!(e.matrix(), &matrix_before[..]);
        // y(center) = R·0 + center + translation = center + translation.
        let y = e.transform_point(&[5.0, 5.0]);
        assert!(
            (y[0] - 6.0).abs() < 1e-12 && (y[1] - 7.0).abs() < 1e-12,
            "{y:?}"
        );
    }

    #[test]
    fn centered_transform_set_translation_recomputes_offset() {
        // Similarity2D: set_translation shifts the mapped center by exactly Δt.
        let mut s = Similarity2DTransform::new(2.0, 0.0, [0.0, 0.0], [3.0, 4.0]);
        let before = s.transform_point(&[3.0, 4.0]); // = center (no translation yet)
        s.set_translation(&[10.0, -5.0]);
        assert_eq!(s.translation(), &[10.0, -5.0]);
        let after = s.transform_point(&[3.0, 4.0]);
        assert!(
            (after[0] - before[0] - 10.0).abs() < 1e-12
                && (after[1] - before[1] + 5.0).abs() < 1e-12,
            "before {before:?} after {after:?}"
        );
    }

    #[test]
    fn centered_transform_via_trait_object() {
        // The three MatrixOffset transforms are usable behind &mut dyn.
        let mut affine = AffineTransform::identity(2);
        let t: &mut dyn CenteredTransform = &mut affine;
        t.set_center(&[2.0, 2.0]);
        t.set_translation(&[1.0, 1.0]);
        // Identity matrix: y = x − c + c + t = x + t.
        assert_eq!(affine.transform_point(&[4.0, 4.0]), vec![5.0, 5.0]);
    }

    #[test]
    fn euler3d_identity_is_noop() {
        let e = Euler3DTransform::identity();
        assert_eq!(e.number_of_parameters(), 6);
        assert_eq!(e.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn euler3d_single_axis_rotations_match_basic_matrices() {
        use std::f64::consts::FRAC_PI_2;
        // 90° about Z: (1,0,0) → (0,1,0), z unchanged.
        let ez = Euler3DTransform::new(0.0, 0.0, FRAC_PI_2, [0.0; 3], [0.0; 3]);
        let p = ez.transform_point(&[1.0, 0.0, 7.0]);
        assert!(
            (p[0]).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12 && (p[2] - 7.0).abs() < 1e-12,
            "{p:?}"
        );
        // 90° about X: (0,1,0) → (0,0,1), x unchanged.
        let ex = Euler3DTransform::new(FRAC_PI_2, 0.0, 0.0, [0.0; 3], [0.0; 3]);
        let q = ex.transform_point(&[9.0, 1.0, 0.0]);
        assert!(
            (q[0] - 9.0).abs() < 1e-12 && (q[1]).abs() < 1e-12 && (q[2] - 1.0).abs() < 1e-12,
            "{q:?}"
        );
        // 90° about Y: (0,0,1) → (1,0,0), y unchanged.
        let ey = Euler3DTransform::new(0.0, FRAC_PI_2, 0.0, [0.0; 3], [0.0; 3]);
        let r = ey.transform_point(&[0.0, 4.0, 1.0]);
        assert!(
            (r[0] - 1.0).abs() < 1e-12 && (r[1] - 4.0).abs() < 1e-12 && (r[2]).abs() < 1e-12,
            "{r:?}"
        );
    }

    #[test]
    fn euler3d_rotation_about_center_fixes_center() {
        let c = [5.0, -2.0, 3.0];
        let e = Euler3DTransform::new(0.3, -0.5, 0.7, [0.0; 3], c);
        let y = e.transform_point(&c);
        for d in 0..3 {
            assert!((y[d] - c[d]).abs() < 1e-12, "center moved: {y:?}");
        }
    }

    #[test]
    fn euler3d_default_and_zyx_orders_differ_and_are_orthonormal() {
        // With multiple nonzero angles the two composition orders give different
        // matrices; each is still a rotation (Mᵀ·M = I).
        let mut e = Euler3DTransform::new(0.4, -0.6, 0.8, [0.0; 3], [0.0; 3]);
        let default = e.matrix().to_vec();
        e.set_compute_zyx(true);
        let zyx = e.matrix().to_vec();
        assert!(default.iter().zip(&zyx).any(|(a, b)| (a - b).abs() > 1e-6));
        for m in [&default, &zyx] {
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f64 = (0..3).map(|k| m[i * 3 + k] * m[j * 3 + k]).sum();
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expect).abs() < 1e-12, "not orthonormal");
                }
            }
        }
    }

    #[test]
    fn euler3d_jacobian_at_identity_is_so3_generators() {
        // At zero angles, centre 0, point (2,3,5): columns are the standard
        // so(3) generators applied to p, then the translation identity block.
        let e = Euler3DTransform::identity();
        let j = e.jacobian_wrt_parameters(&[2.0, 3.0, 5.0]);
        // col0=(0,-pz,py), col1=(pz,0,-px), col2=(-py,px,0).
        assert_eq!(
            j,
            vec![
                0.0, 5.0, -3.0, 1.0, 0.0, 0.0, //
                -5.0, 0.0, 2.0, 0.0, 1.0, 0.0, //
                3.0, -2.0, 0.0, 0.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn euler3d_jacobian_is_finite_difference_consistent_both_orders() {
        for zyx in [false, true] {
            let base = [0.3, -0.5, 0.7, 1.0, -2.0, 0.5];
            let center = [2.0, -1.0, 4.0];
            let point = [4.0, 5.0, -3.0];
            let mut e = Euler3DTransform::new(
                base[0],
                base[1],
                base[2],
                [base[3], base[4], base[5]],
                center,
            );
            e.set_compute_zyx(zyx);
            let jac = e.jacobian_wrt_parameters(&point);
            let n = e.number_of_parameters();
            let h = 1e-6;
            for k in 0..n {
                let mut pp = base;
                pp[k] += h;
                e.set_parameters(&pp);
                let yp = e.transform_point(&point);
                let mut pm = base;
                pm[k] -= h;
                e.set_parameters(&pm);
                let ym = e.transform_point(&point);
                for i in 0..3 {
                    let fd = (yp[i] - ym[i]) / (2.0 * h);
                    assert!(
                        (fd - jac[i * n + k]).abs() < 1e-6,
                        "zyx={zyx} param {k} dim {i}: fd {fd} vs analytic {}",
                        jac[i * n + k]
                    );
                }
            }
        }
    }

    #[test]
    fn euler3d_parameters_roundtrip() {
        let mut e = Euler3DTransform::identity();
        e.set_parameters(&[0.1, 0.2, 0.3, 4.0, 5.0, 6.0]);
        assert_eq!(e.parameters(), vec![0.1, 0.2, 0.3, 4.0, 5.0, 6.0]);
        assert_eq!(e.angle_x(), 0.1);
        assert_eq!(e.angle_y(), 0.2);
        assert_eq!(e.angle_z(), 0.3);
        assert_eq!(e.translation(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn versor3d_identity_is_noop() {
        let v = VersorRigid3DTransform::identity();
        assert_eq!(v.number_of_parameters(), 6);
        assert_eq!(v.versor_w(), 1.0);
        assert_eq!(v.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn versor3d_ninety_degrees_about_z_matches_rz() {
        use std::f64::consts::FRAC_PI_4;
        // Right part (0,0,sin(θ/2)) with θ=90° ⇒ Rz(90°): (1,0,0) → (0,1,0).
        let v = VersorRigid3DTransform::new(0.0, 0.0, FRAC_PI_4.sin(), [0.0; 3], [0.0; 3]);
        let p = v.transform_point(&[1.0, 0.0, 7.0]);
        assert!(
            (p[0]).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12 && (p[2] - 7.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn versor3d_matrix_is_orthonormal_and_fixes_center() {
        let c = [5.0, -2.0, 3.0];
        let v = VersorRigid3DTransform::new(0.1, -0.2, 0.15, [0.0; 3], c);
        let m = v.matrix();
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[i * 3 + k] * m[j * 3 + k]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-12, "not orthonormal");
            }
        }
        let y = v.transform_point(&c);
        for d in 0..3 {
            assert!((y[d] - c[d]).abs() < 1e-12, "center moved: {y:?}");
        }
    }

    #[test]
    fn versor3d_right_part_norm_above_one_is_scaled_below_one() {
        // A right part with norm > 1 is scaled to just under 1 (so w stays real).
        let v = VersorRigid3DTransform::new(0.8, 0.8, 0.8, [0.0; 3], [0.0; 3]);
        let n2 = v.versor_x().powi(2) + v.versor_y().powi(2) + v.versor_z().powi(2);
        assert!(n2 < 1.0, "norm² = {n2}");
        assert!(v.versor_w() >= 0.0 && v.versor_w().is_finite());
    }

    #[test]
    fn versor3d_jacobian_at_identity_is_twice_so3_generators() {
        // At the identity versor, columns are 2× the so(3) generators applied to
        // (p − centre): col0=(0,-2pz,2py), col1=(2pz,0,-2px), col2=(-2py,2px,0).
        let v = VersorRigid3DTransform::identity();
        let j = v.jacobian_wrt_parameters(&[2.0, 3.0, 5.0]);
        assert_eq!(
            j,
            vec![
                0.0, 10.0, -6.0, 1.0, 0.0, 0.0, //
                -10.0, 0.0, 4.0, 0.0, 1.0, 0.0, //
                6.0, -4.0, 0.0, 0.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn versor3d_jacobian_is_finite_difference_consistent() {
        // Small right part keeps ‖v‖ well below 1 (no renormalization), so the
        // finite difference exercises the analytic w = √(1−‖v‖²) dependence.
        let base = [0.12, -0.08, 0.1, 1.0, -2.0, 0.5];
        let center = [2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let mut v = VersorRigid3DTransform::new(
            base[0],
            base[1],
            base[2],
            [base[3], base[4], base[5]],
            center,
        );
        let jac = v.jacobian_wrt_parameters(&point);
        let n = v.number_of_parameters();
        let h = 1e-7;
        for k in 0..n {
            let mut pp = base;
            pp[k] += h;
            v.set_parameters(&pp);
            let yp = v.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            v.set_parameters(&pm);
            let ym = v.transform_point(&point);
            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-5,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }

    #[test]
    fn versor3d_parameters_roundtrip_for_small_right_part() {
        let mut v = VersorRigid3DTransform::identity();
        v.set_parameters(&[0.1, -0.2, 0.15, 4.0, 5.0, 6.0]);
        let p = v.parameters();
        // Small right part is stored unchanged (no renormalization).
        assert!(
            (p[0] - 0.1).abs() < 1e-12 && (p[1] + 0.2).abs() < 1e-12 && (p[2] - 0.15).abs() < 1e-12
        );
        assert_eq!(v.translation(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn similarity3d_identity_is_noop() {
        let t = Similarity3DTransform::identity();
        assert_eq!(t.number_of_parameters(), 7);
        assert_eq!(t.scale(), 1.0);
        assert_eq!(t.versor_w(), 1.0);
        assert_eq!(t.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn similarity3d_scales_about_center() {
        // No rotation (versor 0), scale 2 about centre c: p ↦ c + 2·(p − c).
        let c = [1.0, -2.0, 3.0];
        let t = Similarity3DTransform::new(2.0, 0.0, 0.0, 0.0, [0.0; 3], c);
        let p = [4.0, 1.0, -1.0];
        let y = t.transform_point(&p);
        for d in 0..3 {
            let expect = c[d] + 2.0 * (p[d] - c[d]);
            assert!((y[d] - expect).abs() < 1e-12, "dim {d}: {y:?}");
        }
        // The centre maps to itself when there is no translation.
        let yc = t.transform_point(&c);
        for d in 0..3 {
            assert!((yc[d] - c[d]).abs() < 1e-12, "centre moved: {yc:?}");
        }
    }

    #[test]
    fn similarity3d_matrix_is_scaled_rotation() {
        use std::f64::consts::FRAC_PI_4;
        // Right part (0,0,sin(45°)) ⇒ Rz(90°); scale 2 ⇒ M = 2·Rz(90°).
        // det(M) = scale³ and M/scale is orthonormal.
        let s = 2.0;
        let t = Similarity3DTransform::new(s, 0.0, 0.0, FRAC_PI_4.sin(), [0.0; 3], [0.0; 3]);
        let m = t.matrix();
        // Rz(90°) = [[0,-1,0],[1,0,0],[0,0,1]] ⇒ M = [[0,-2,0],[2,0,0],[0,0,2]].
        #[rustfmt::skip]
        let expect = [0.0, -2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        for (a, b) in m.iter().zip(expect) {
            assert!((a - b).abs() < 1e-12, "matrix {m:?}");
        }
        // (1,0,7) ↦ 2·Rz(90°)·(1,0,7) = (0,2,14).
        let p = t.transform_point(&[1.0, 0.0, 7.0]);
        assert!(
            (p[0]).abs() < 1e-12 && (p[1] - 2.0).abs() < 1e-12 && (p[2] - 14.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn similarity3d_parameters_roundtrip() {
        let mut t = Similarity3DTransform::identity();
        t.set_parameters(&[0.1, -0.2, 0.15, 4.0, 5.0, 6.0, 1.3]);
        let p = t.parameters();
        // Small right part is stored unchanged; scale is parameter 6 (last).
        assert!(
            (p[0] - 0.1).abs() < 1e-12 && (p[1] + 0.2).abs() < 1e-12 && (p[2] - 0.15).abs() < 1e-12
        );
        assert_eq!(&p[3..6], &[4.0, 5.0, 6.0]);
        assert!((p[6] - 1.3).abs() < 1e-12);
    }

    #[test]
    fn similarity3d_jacobian_is_finite_difference_consistent() {
        // Small right part keeps ‖v‖ below 1 (no renormalization); a non-unit scale
        // exercises both the scaled versor columns and the scale column.
        let base = [0.12, -0.08, 0.1, 1.0, -2.0, 0.5, 1.3];
        let center = [2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let mut t = Similarity3DTransform::new(
            base[6],
            base[0],
            base[1],
            base[2],
            [base[3], base[4], base[5]],
            center,
        );
        let jac = t.jacobian_wrt_parameters(&point);
        let n = t.number_of_parameters();
        let h = 1e-7;
        for k in 0..n {
            let mut pp = base;
            pp[k] += h;
            t.set_parameters(&pp);
            let yp = t.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            t.set_parameters(&pm);
            let ym = t.transform_point(&point);
            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-5,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }

    #[test]
    fn scale_versor3d_identity_is_noop() {
        let t = ScaleVersor3DTransform::identity();
        assert_eq!(t.number_of_parameters(), 9);
        assert_eq!(t.scale(), &[1.0, 1.0, 1.0]);
        assert_eq!(t.versor_w(), 1.0);
        assert_eq!(t.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn scale_versor3d_anisotropic_scale_no_rotation() {
        // versor 0 ⇒ R = I ⇒ M = diag(scale). With centre c and no translation,
        // y = diag(scale)·(p − c) + c.
        let c = [1.0, -1.0, 2.0];
        let t = ScaleVersor3DTransform::new([2.0, 3.0, 0.5], 0.0, 0.0, 0.0, [0.0; 3], c);
        let y = t.transform_point(&[3.0, 1.0, 4.0]);
        // [2·(3−1)+1, 3·(1+1)−1, 0.5·(4−2)+2] = [5, 5, 3].
        assert!(
            (y[0] - 5.0).abs() < 1e-12 && (y[1] - 5.0).abs() < 1e-12 && (y[2] - 3.0).abs() < 1e-12,
            "{y:?}"
        );
    }

    #[test]
    fn scale_versor3d_matrix_is_additive_not_multiplicative() {
        use std::f64::consts::FRAC_PI_4;
        // Rz(90°) with anisotropic scale [2,3,4]. ITK's additive form gives
        // M = R + diag(scale − 1) = [1,−1,0; 1,2,0; 0,0,4], which differs from the
        // multiplicative R·diag(scale) = [0,−3,0; 2,0,0; 0,0,4].
        let t = ScaleVersor3DTransform::new(
            [2.0, 3.0, 4.0],
            0.0,
            0.0,
            FRAC_PI_4.sin(),
            [0.0; 3],
            [0.0; 3],
        );
        #[rustfmt::skip]
        let additive = [1.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 4.0];
        #[rustfmt::skip]
        let multiplicative = [0.0, -3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0];
        for (a, b) in t.matrix().iter().zip(additive) {
            assert!((a - b).abs() < 1e-12, "matrix {:?}", t.matrix());
        }
        // Confirm the two forms are genuinely different (guards the parity note).
        assert_ne!(additive, multiplicative);
    }

    #[test]
    fn scale_versor3d_parameters_roundtrip() {
        let mut t = ScaleVersor3DTransform::identity();
        t.set_parameters(&[0.1, -0.2, 0.15, 4.0, 5.0, 6.0, 1.2, 0.8, 1.5]);
        let p = t.parameters();
        assert!(
            (p[0] - 0.1).abs() < 1e-12 && (p[1] + 0.2).abs() < 1e-12 && (p[2] - 0.15).abs() < 1e-12
        );
        assert_eq!(&p[3..6], &[4.0, 5.0, 6.0]);
        assert_eq!(&p[6..9], &[1.2, 0.8, 1.5]);
    }

    #[test]
    fn scale_versor3d_jacobian_is_finite_difference_consistent() {
        // Small right part keeps ‖v‖ below 1; anisotropic non-unit scale exercises
        // the versor columns (no scale factor) and the diagonal scale columns.
        let base = [0.12, -0.08, 0.1, 1.0, -2.0, 0.5, 1.3, 0.8, 1.5];
        let center = [2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let mut t = ScaleVersor3DTransform::new(
            [base[6], base[7], base[8]],
            base[0],
            base[1],
            base[2],
            [base[3], base[4], base[5]],
            center,
        );
        let jac = t.jacobian_wrt_parameters(&point);
        let n = t.number_of_parameters();
        let h = 1e-7;
        for k in 0..n {
            let mut pp = base;
            pp[k] += h;
            t.set_parameters(&pp);
            let yp = t.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            t.set_parameters(&pm);
            let ym = t.transform_point(&point);
            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-5,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }

    #[test]
    fn scale_skew_versor3d_identity_is_noop() {
        let t = ScaleSkewVersor3DTransform::identity();
        assert_eq!(t.number_of_parameters(), 15);
        assert_eq!(t.scale(), &[1.0, 1.0, 1.0]);
        assert_eq!(t.skew(), &[0.0; 6]);
        assert_eq!(t.versor_w(), 1.0);
        assert_eq!(t.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn scale_skew_versor3d_matrix_adds_scale_and_skew() {
        // No rotation (R = I): M = diag(scale) + skew off-diagonals in the order
        // {xy, xz, yx, yz, zx, zy} → M[0][1],M[0][2],M[1][0],M[1][2],M[2][0],M[2][1].
        let t = ScaleSkewVersor3DTransform::new(
            [2.0, 3.0, 4.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            [0.0; 3],
        );
        #[rustfmt::skip]
        let expect = [
            2.0, 0.1, 0.2,
            0.3, 3.0, 0.4,
            0.5, 0.6, 4.0,
        ];
        for (a, b) in t.matrix().iter().zip(expect) {
            assert!((a - b).abs() < 1e-12, "matrix {:?}", t.matrix());
        }
    }

    #[test]
    fn scale_skew_versor3d_reduces_to_scale_versor_when_skew_zero() {
        // With zero skew, ScaleSkewVersor3D must match ScaleVersor3D for the same
        // versor/scale/translation/centre — the additive skew block is the only
        // structural difference between the two transforms.
        let (vx, vy, vz) = (0.1, -0.2, 0.15);
        let scale = [1.3, 0.8, 1.5];
        let tr = [4.0, -1.0, 2.0];
        let c = [2.0, -3.0, 1.0];
        let a = ScaleSkewVersor3DTransform::new(scale, [0.0; 6], vx, vy, vz, tr, c);
        let b = ScaleVersor3DTransform::new(scale, vx, vy, vz, tr, c);
        for (ma, mb) in a.matrix().iter().zip(b.matrix()) {
            assert!((ma - mb).abs() < 1e-12, "matrix mismatch");
        }
        let p = [7.0, -3.0, 6.0];
        for (ya, yb) in a.transform_point(&p).iter().zip(b.transform_point(&p)) {
            assert!((ya - yb).abs() < 1e-12, "point mismatch");
        }
    }

    #[test]
    fn scale_skew_versor3d_parameters_roundtrip() {
        let mut t = ScaleSkewVersor3DTransform::identity();
        let params = [
            0.1, -0.2, 0.15, 4.0, 5.0, 6.0, 1.2, 0.8, 1.5, 0.05, -0.1, 0.15, -0.2, 0.1, -0.05,
        ];
        t.set_parameters(&params);
        let p = t.parameters();
        assert!(
            (p[0] - 0.1).abs() < 1e-12 && (p[1] + 0.2).abs() < 1e-12 && (p[2] - 0.15).abs() < 1e-12
        );
        assert_eq!(&p[3..6], &[4.0, 5.0, 6.0]);
        assert_eq!(&p[6..9], &[1.2, 0.8, 1.5]);
        assert_eq!(&p[9..15], &[0.05, -0.1, 0.15, -0.2, 0.1, -0.05]);
    }

    #[test]
    fn scale_skew_versor3d_jacobian_is_finite_difference_consistent() {
        // Small right part keeps ‖v‖ below 1; non-unit scale and non-zero skew
        // exercise the versor, translation, diagonal-scale, and skew columns.
        let base = [
            0.12, -0.08, 0.1, 1.0, -2.0, 0.5, 1.3, 0.8, 1.5, 0.05, -0.1, 0.15, -0.2, 0.1, -0.05,
        ];
        let center = [2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let mut t = ScaleSkewVersor3DTransform::new(
            [base[6], base[7], base[8]],
            [base[9], base[10], base[11], base[12], base[13], base[14]],
            base[0],
            base[1],
            base[2],
            [base[3], base[4], base[5]],
            center,
        );
        let jac = t.jacobian_wrt_parameters(&point);
        let n = t.number_of_parameters();
        let h = 1e-7;
        for k in 0..n {
            let mut pp = base;
            pp[k] += h;
            t.set_parameters(&pp);
            let yp = t.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            t.set_parameters(&pm);
            let ym = t.transform_point(&point);
            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-5,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }
}
