//! Spatial transforms.
//!
//! A [`TransformBase`] maps a point in one physical space to another. In resampling
//! it maps a point in the **output** image's physical space to the **input**
//! image's physical space (ITK's backward mapping convention).

use crate::core::matrix;
use crate::transform::error::{Result, TransformError};
use crate::transform::matrix_offset::MatrixOffsetMap;

/// A spatial coordinate transform.
pub trait TransformBase {
    /// Map a physical point to its transformed physical point.
    fn transform_point(&self, point: &[f64]) -> Vec<f64>;
    /// Spatial dimension the transform operates on.
    fn dimension(&self) -> usize;

    /// The **ordered stages** this transform's [`transform_point`] evaluates, each one
    /// exactly `mat_vec(matrix, p) + offset`, applied in the order returned — or `None`
    /// when there is no such decomposition that is exact **on the bits**.
    ///
    /// > if `point_map_stages()` returns `Some(stages)`, then for every finite `p`,
    /// > `transform_point(p)` **is** `stages.fold(p, |q, s| mat_vec(s.matrix, q) +
    /// > s.offset)` — the same operations, in the same order, on the same operands.
    ///
    /// # Why a list and not one matrix
    ///
    /// A transform that composes (`CompositeTransform`, or a registration's optimized
    /// transform followed by a moving-initial one) evaluates its stages **sequentially**,
    /// each rounding on its own. Multiplying the stage matrices together is algebraically
    /// the same map and is *not* the same arithmetic — it rounds once where the transform
    /// rounds twice. So the stages are handed over as stages, and a backend that wants
    /// the bits reproduces the sequence rather than folding it.
    ///
    /// # Who needs the bits
    ///
    /// The CUDA metric. Its sampler makes three **discrete** decisions per sample —
    /// `floor(c)` (which cell), `is_inside(c)` (whether the sample exists at all), and
    /// `round(c)` (which moving-mask voxel) — and one ulp in the mapped point flips any
    /// of them. Reconstructing the map by *probing* it (`b = T(0)`,
    /// `A[:,e] = T(e_e) − T(0)`) is ~1e-14 away from the transform's own arithmetic:
    /// fine for a value gated at 1e-9, and fatal for a predicate (ledger §2.158).
    ///
    /// # The default is `None`, and that is the safe direction
    ///
    /// A transform that does not override this is refused by any backend that needs bit
    /// equality — it falls back to the host rather than being approximated. See
    /// [`crate::transform::matrix_offset`] for the contract in full and for the variants that are
    /// mathematically linear and still refused (`ScaleTransform` evaluates
    /// `(p − c)·s + c`, which is a different rounding from `M·p + b`).
    ///
    /// [`transform_point`]: TransformBase::transform_point
    fn point_map_stages(&self) -> Option<Vec<MatrixOffsetMap>> {
        None
    }

    /// Whether `transform_point` is `x ↦ M·x + b` for some constant matrix
    /// `M` and offset `b` (independent of `x`), mirroring
    /// `itk::Transform::IsLinear()` / `GetTransformCategory() ==
    /// TransformCategoryEnum::Linear`. Every matrix-offset, translation, and
    /// scale transform in this crate is unconditionally linear, matching
    /// ITK's own `MatrixOffsetTransformBase`/`TranslationTransform`
    /// overrides (both hardcode `true` rather than deriving it from
    /// `GetTransformCategory()`), so the default here is `true`;
    /// [`BSplineTransform`] and [`DisplacementFieldTransform`] override it
    /// to `false` (`GetTransformCategory()` returns `BSpline`/
    /// `DisplacementField` there), and [`CompositeTransform`] overrides it
    /// to the conjunction of its sub-transforms' own `is_linear()`
    /// (`itk::MultiTransform::IsLinear()`).
    ///
    /// [`crate::transform::transform_geometry()`]'s linearity precondition is the only
    /// current caller.
    ///
    /// [`BSplineTransform`]: crate::transform::BSplineTransform
    /// [`DisplacementFieldTransform`]: crate::transform::DisplacementFieldTransform
    /// [`CompositeTransform`]: crate::transform::CompositeTransform
    fn is_linear(&self) -> bool {
        true
    }

    /// Jacobian `∂(transform_point(point))ᵢ / ∂pointⱼ`, row-major
    /// `dimension × dimension` — ITK's
    /// `TransformBase::ComputeJacobianWithRespectToPosition`. This is what
    /// [`CompositeTransform`] chain-rules through when it assembles the
    /// parameter Jacobian of a stack.
    ///
    /// The default is a central finite difference of [`transform_point`], exact
    /// only to `O(h²)`. Every transform whose spatial derivative is known in
    /// closed form overrides it: a matrix-offset transform returns its matrix,
    /// a translation the identity, a scale its diagonal. The default stands for
    /// [`BSplineTransform`] and [`DisplacementFieldTransform`], whose spatial
    /// derivative is a B-spline / field derivative this crate does not yet
    /// expose.
    ///
    /// [`transform_point`]: TransformBase::transform_point
    /// [`CompositeTransform`]: crate::transform::CompositeTransform
    /// [`BSplineTransform`]: crate::transform::BSplineTransform
    /// [`DisplacementFieldTransform`]: crate::transform::DisplacementFieldTransform
    fn jacobian_wrt_position(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        let mut jac = vec![0.0; dim * dim];
        for (c, &pc) in point.iter().enumerate().take(dim) {
            // Scale the step to the coordinate so a far-from-origin point does
            // not lose the perturbation to f64 cancellation.
            let h = 1e-6 * pc.abs().max(1.0);
            let mut plus = point.to_vec();
            let mut minus = point.to_vec();
            plus[c] += h;
            minus[c] -= h;
            let f_plus = self.transform_point(&plus);
            let f_minus = self.transform_point(&minus);
            for r in 0..dim {
                jac[r * dim + c] = (f_plus[r] - f_minus[r]) / (2.0 * h);
            }
        }
        jac
    }
}

/// The row-major `n × n` matrix with `v` on its diagonal.
fn diagonal(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut m = vec![0.0; n * n];
    for (d, &vd) in v.iter().enumerate() {
        m[d * n + d] = vd;
    }
    m
}

/// Reject a fixed-parameter array whose length is not `expected`, naming what
/// the array should have held. ITK's `SetFixedParameters` overrides throw here.
pub(crate) fn check_fixed_len(params: &[f64], expected: usize, what: &str) -> Result<()> {
    if params.len() == expected {
        Ok(())
    } else {
        Err(TransformError::InvalidFixedParameters {
            got: params.len(),
            expected: format!("{expected} ({what})"),
        })
    }
}

/// Reject a parameter array whose length is not `expected`. ITK's
/// `SetParameters` overrides disagree on strictness (`MatrixOffsetTransformBase`
/// and `TranslationTransform` throw only when the vector is *shorter*,
/// `VersorTransform` checks nothing, `BSplineTransform` demands exact
/// equality); this port requires exact equality everywhere (ledger §4.47).
pub(crate) fn check_len(params: &[f64], expected: usize) -> Result<()> {
    if params.len() == expected {
        Ok(())
    } else {
        Err(TransformError::InvalidParameters {
            got: params.len(),
            expected,
        })
    }
}

/// Fixed parameters of a matrix-offset transform: the center of rotation, and
/// nothing else (`itk::MatrixOffsetTransformBase::Get/SetFixedParameters`).
/// Expands inside `impl ParametricTransform for T` for any `T` with a `center`
/// field and a `recompute()` that refreshes the cached matrix/offset.
macro_rules! center_fixed_parameters {
    ($dim:expr) => {
        fn fixed_parameters(&self) -> Vec<f64> {
            self.center.clone()
        }

        fn number_of_fixed_parameters(&self) -> usize {
            $dim
        }

        fn set_fixed_parameters(&mut self, params: &[f64]) -> $crate::transform::error::Result<()> {
            $crate::transform::parametric::check_fixed_len(params, $dim, "the center of rotation")?;
            self.center.copy_from_slice(params);
            self.recompute();
            Ok(())
        }
    };
}

/// `dT/dx` of a matrix-offset transform `T(x) = M·x + offset` is exactly `M`
/// (`itk::MatrixOffsetTransformBase::ComputeJacobianWithRespectToPosition`).
macro_rules! matrix_jacobian_wrt_position {
    () => {
        fn jacobian_wrt_position(&self, _point: &[f64]) -> Vec<f64> {
            self.matrix.clone()
        }
    };
}

/// [`TransformBase::point_map_stages`] for the matrix-offset family: **one** stage, and
/// it is the very matrix and offset `transform_point` multiplies and adds
/// (`mat_vec(&self.matrix, point) + self.offset`). The accessor and the evaluator read
/// the same field, so they cannot disagree — that is what makes the bitwise claim
/// structural rather than a numerical coincidence.
macro_rules! matrix_point_map_stages {
    () => {
        fn point_map_stages(&self) -> Option<Vec<MatrixOffsetMap>> {
            Some(vec![MatrixOffsetMap {
                matrix: self.matrix.clone(),
                offset: self.offset.clone(),
            }])
        }
    };
}

/// A transform whose action is controlled by a flat parameter vector, and which
/// exposes the Jacobian of the mapped point with respect to those parameters.
/// This is the interface registration optimizes over, mirroring ITK's
/// `Transform::GetJacobianWithRespectToParameters`.
///
/// `Sync` is required: a metric evaluates the transform at every fixed sample,
/// and those evaluations run in parallel over a shared `&dyn
/// ParametricTransform`. Every transform here is plain data with no interior
/// mutability, so the bound is free — it exists to keep it that way.
pub trait ParametricTransform: TransformBase + Sync {
    /// Number of free parameters.
    fn number_of_parameters(&self) -> usize;

    /// Current parameter vector (length [`number_of_parameters`]).
    ///
    /// [`number_of_parameters`]: ParametricTransform::number_of_parameters
    fn parameters(&self) -> Vec<f64>;

    /// Replace the parameter vector. Errors with
    /// [`TransformError::InvalidParameters`] if `params.len()` is not
    /// [`number_of_parameters`], mirroring ITK's `SetParameters` overrides,
    /// which throw on a length mismatch.
    ///
    /// [`number_of_parameters`]: ParametricTransform::number_of_parameters
    fn set_parameters(&mut self, params: &[f64]) -> Result<()>;

    /// The *fixed* (non-optimizable) parameters — ITK's
    /// `Transform::GetFixedParameters`. These are the values the Insight legacy
    /// transform file records on its `FixedParameters:` line: the center of
    /// rotation for every matrix-offset transform, nothing at all for a pure
    /// translation, and the grid geometry (size, origin, spacing, direction)
    /// for a B-spline or displacement-field transform.
    fn fixed_parameters(&self) -> Vec<f64>;

    /// Replace the fixed parameters — ITK's `Transform::SetFixedParameters`.
    /// Errors with [`TransformError::InvalidFixedParameters`] wherever ITK
    /// throws.
    ///
    /// For [`BSplineTransform`] and [`DisplacementFieldTransform`] this
    /// *re-allocates* the coefficient / displacement grid and zeroes it, exactly
    /// as ITK's `SetCoefficientImageInformationFromFixedParameters` /
    /// `DisplacementFieldTransform::SetFixedParameters` do — which is why the
    /// Insight-legacy reader has to apply the fixed parameters before the
    /// parameters (`itkTxtTransformIO.cxx:186-217`).
    ///
    /// [`BSplineTransform`]: crate::transform::BSplineTransform
    /// [`DisplacementFieldTransform`]: crate::transform::DisplacementFieldTransform
    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()>;

    /// Number of fixed parameters (`itk::Transform::GetNumberOfFixedParameters`).
    fn number_of_fixed_parameters(&self) -> usize {
        self.fixed_parameters().len()
    }

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
    /// [`dimension`]: TransformBase::dimension
    fn number_of_local_parameters(&self) -> usize {
        self.number_of_parameters()
    }

    /// A sparse representation of [`jacobian_wrt_parameters`] at `point`: the
    /// list of `(parameter_index, column)` entries whose column — length
    /// [`dimension`], `∂Tᵢ/∂param[parameter_index]` — may be non-zero. Every
    /// *other* entry of the dense Jacobian at `point` is exactly zero, so a
    /// metric can accumulate the derivative by touching only these entries
    /// instead of allocating the full `dimension × number_of_parameters`
    /// array.
    ///
    /// A transform that implements this **always** returns `Some`, even when
    /// `point` contributes nothing (an empty `Vec` — e.g. outside a
    /// B-spline's valid region or a displacement field's buffer, where the
    /// dense Jacobian is all-zero too, not absent). `None` means this
    /// transform has no sparse representation at all; a metric reads that
    /// *once per transform*, not per point, as the signal to fall back to
    /// the dense [`jacobian_wrt_parameters`] contract for every sample. The
    /// default returns `None` — every transform whose Jacobian is already
    /// dense and small (translation, affine, similarity, Euler, versor)
    /// keeps it. [`BSplineTransform`] and [`DisplacementFieldTransform`]
    /// override it: both have a Jacobian that is structurally almost
    /// entirely zero at any point, just with a different affected-parameter
    /// *shape* — a scattered set of control points for a B-spline, one
    /// contiguous pixel block for a displacement field — which is exactly
    /// what this entry list abstracts over.
    ///
    /// This is independent of [`has_local_support`]: that flag mirrors
    /// ITK's `Transform::GetTransformCategory() == DisplacementField` /
    /// `ObjectToObjectMetric::HasLocalSupport()` exactly, and stays `false`
    /// for a B-spline transform (`BSplineBaseTransform::GetTransformCategory`
    /// returns `TransformCategoryEnum::BSpline`, not `DisplacementField` —
    /// see the [`BSplineTransform`] module docs). This accessor is a
    /// separate, crate-internal performance signal: it lets a metric skip
    /// the dense Jacobian without touching `has_local_support`'s
    /// ITK-faithful semantics, which stay available for whatever else keys
    /// on them (e.g. a future displacement-field scales path).
    ///
    /// [`jacobian_wrt_parameters`]: ParametricTransform::jacobian_wrt_parameters
    /// [`dimension`]: TransformBase::dimension
    /// [`has_local_support`]: ParametricTransform::has_local_support
    /// [`BSplineTransform`]: crate::transform::BSplineTransform
    /// [`DisplacementFieldTransform`]: crate::transform::DisplacementFieldTransform
    fn sparse_jacobian_wrt_parameters(&self, _point: &[f64]) -> Option<Vec<(usize, Vec<f64>)>> {
        None
    }
}

/// A transform with a fixed center of rotation and a translation that can be set
/// independently of the parameter vector, mirroring
/// `itk::MatrixOffsetTransformBase::SetCenter` / `SetTranslation`. This is the
/// interface `CenteredTransformInitializer` configures; a pure
/// [`TranslationTransform`] has no center and so does not implement it.
pub trait CenteredTransform: TransformBase {
    /// Set the fixed center of rotation (length = [`dimension`]). The matrix and
    /// translation are unchanged; the applied offset is recomputed.
    ///
    /// [`dimension`]: TransformBase::dimension
    fn set_center(&mut self, center: &[f64]);

    /// Set the translation (length = [`dimension`]). The matrix and center are
    /// unchanged; the applied offset is recomputed.
    ///
    /// [`dimension`]: TransformBase::dimension
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

impl TransformBase for TranslationTransform {
    /// One stage, with a **synthesized** identity matrix: this transform has no `matrix`
    /// field — it evaluates `p[d] + t[d]` — so the bitwise claim here is an IEEE-754
    /// argument rather than a shared field. `mat_vec(I, p)[d]` is
    /// `0.0 + 1.0·p_d + 0.0·p_e + 0.0·p_f`, and adding `±0.0` to a finite value is exact,
    /// so it is `p[d]` on the bit. Pinned, not trusted:
    /// `matrix_offset::tests::translation_is_bitwise_the_identity_matrix_form`.
    fn point_map_stages(&self) -> Option<Vec<MatrixOffsetMap>> {
        let dim = self.translation.len();
        let mut matrix = vec![0.0; dim * dim];
        for d in 0..dim {
            matrix[d * dim + d] = 1.0;
        }
        Some(vec![MatrixOffsetMap {
            matrix,
            offset: self.translation.clone(),
        }])
    }

    /// `T(x) = x + t`, so `dT/dx` is the identity.
    fn jacobian_wrt_position(&self, _point: &[f64]) -> Vec<f64> {
        let dim = self.translation.len();
        let mut jac = vec![0.0; dim * dim];
        for d in 0..dim {
            jac[d * dim + d] = 1.0;
        }
        jac
    }

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, self.translation.len())?;
        self.translation.copy_from_slice(params);
        Ok(())
    }

    /// A pure translation has **no** fixed parameters: `itk::TranslationTransform`
    /// never resizes the `itk::Transform` base's `m_FixedParameters`, which is
    /// default-constructed empty.
    fn fixed_parameters(&self) -> Vec<f64> {
        Vec::new()
    }

    fn number_of_fixed_parameters(&self) -> usize {
        0
    }

    /// ITK's base `Transform::SetFixedParameters` stores *whatever* it is given
    /// (`itkTransform.h`: `m_FixedParameters = fixedParameters;`), so a
    /// hand-written file with a non-empty `FixedParameters:` line silently
    /// attaches dead values to an `itk::TranslationTransform` and echoes them
    /// back on the next write. This port has nowhere to keep them and rejects a
    /// non-empty array instead (ledger §4.46).
    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_fixed_len(params, 0, "a translation has no fixed parameters")
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

impl TransformBase for AffineTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        let n = self.dim * self.dim;
        check_len(params, n + self.dim)?;
        self.matrix.copy_from_slice(&params[..n]);
        self.translation.copy_from_slice(&params[n..]);
        self.offset = Self::compute_offset(self.dim, &self.matrix, &self.translation, &self.center);
        Ok(())
    }

    /// The center of rotation
    /// (`itk::MatrixOffsetTransformBase::GetFixedParameters`).
    fn fixed_parameters(&self) -> Vec<f64> {
        self.center.clone()
    }

    fn number_of_fixed_parameters(&self) -> usize {
        self.dim
    }

    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_fixed_len(params, self.dim, "the center of rotation")?;
        self.center.copy_from_slice(params);
        self.offset = Self::compute_offset(self.dim, &self.matrix, &self.translation, &self.center);
        Ok(())
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

impl TransformBase for Euler2DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 3)?;
        self.angle = params[0];
        self.translation[0] = params[1];
        self.translation[1] = params[2];
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(2);

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

impl TransformBase for Similarity2DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 4)?;
        self.scale = params[0];
        self.angle = params[1];
        self.translation[0] = params[2];
        self.translation[1] = params[3];
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(2);

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

impl TransformBase for Euler3DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 6)?;
        self.angle_x = params[0];
        self.angle_y = params[1];
        self.angle_z = params[2];
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.recompute();
        Ok(())
    }

    /// `[cx, cy, cz, computeZYX]` — `itk::Euler3DTransform` appends its
    /// `m_ComputeZYX` flag to the center as a fourth fixed parameter
    /// (`itkEuler3DTransform.hxx:120-128`), so the rotation-composition order
    /// survives a write/read round trip.
    fn fixed_parameters(&self) -> Vec<f64> {
        let mut fp = self.center.clone();
        fp.push(if self.compute_zyx { 1.0 } else { 0.0 });
        fp
    }

    fn number_of_fixed_parameters(&self) -> usize {
        4
    }

    /// Accepts 3 or 4 values: `itk::Euler3DTransform::SetFixedParameters` reads
    /// the fourth only when the array has exactly 4 entries, "for backwards
    /// compatibility: the m_ComputeZYX flag was not serialized so it may or may
    /// not be included as part of the fixed parameters"
    /// (`itkEuler3DTransform.hxx:131-154`). A 3-entry array therefore leaves
    /// `compute_zyx` at its current value rather than resetting it.
    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != 3 && params.len() != 4 {
            return Err(TransformError::InvalidFixedParameters {
                got: params.len(),
                expected: "3 (the center of rotation) or 4 (center and computeZYX)".to_string(),
            });
        }
        self.center.copy_from_slice(&params[..3]);
        if params.len() == 4 {
            self.compute_zyx = params[3] != 0.0;
        }
        self.recompute();
        Ok(())
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

    /// Set the rotation from a row-major 3×3 matrix, converting it to the
    /// equivalent versor via the branch method of
    /// `itk::Versor<T>::Set(const MatrixType&)`
    /// (`itkVersor.hxx:301-381`, Shepperd 1978) — the same conversion
    /// `itk::VersorRigid3DTransform` performs internally whenever `SetMatrix`
    /// is called, via `MatrixOffsetTransformBase::SetMatrix`
    /// (`itkMatrixOffsetTransformBase.h:232-236`) invoking the virtual
    /// `VersorTransform::ComputeMatrixParameters`
    /// (`itkVersorTransform.hxx:130-135`): `m_Versor.Set(this->GetMatrix())`.
    ///
    /// Errors with [`TransformError::NotARotationMatrix`] if `matrix` is not
    /// orthonormal (`m·mᵀ ≈ I`) or is a reflection (`det < 0`), to within
    /// `itk::Versor<double>`'s own tolerance (`Epsilon() = 1e-10`,
    /// `itkVersor.h:305-309`), mirroring the `itkGenericExceptionMacro` guard
    /// in `Versor::Set` (`itkVersor.hxx:323-338`).
    ///
    /// Unlike ITK — which stores `matrix` verbatim in `m_Matrix` and only
    /// *derives* the versor (parameters) from it, so a later `GetMatrix()`
    /// echoes the caller's input exactly — this port has no separate cached
    /// matrix distinct from the versor: [`matrix`](Self::matrix) is always
    /// re-derived from the stored versor (as every other mutator on this type
    /// already does), so it can differ from the `matrix` argument by a few
    /// ULPs of floating-point rounding (ledger §4.45).
    pub fn set_matrix(&mut self, matrix: &[f64]) -> Result<()> {
        let (x, y, z) = versor_right_part_from_matrix(matrix)?;
        self.set_versor(x, y, z);
        self.recompute();
        Ok(())
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

impl TransformBase for VersorRigid3DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 6)?;
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(3);

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

impl TransformBase for Similarity3DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 7)?;
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale = params[6];
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(3);

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

impl TransformBase for ScaleVersor3DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 9)?;
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale[0] = params[6];
        self.scale[1] = params[7];
        self.scale[2] = params[8];
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(3);

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

impl TransformBase for ScaleSkewVersor3DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

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

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 15)?;
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale[0] = params[6];
        self.scale[1] = params[7];
        self.scale[2] = params[8];
        self.skew.copy_from_slice(&params[9..15]);
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(3);

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

/// A 3-D transform composing a versor rotation, per-axis scale, and an upper-
/// triangular skew **multiplicatively** (`itk::ComposeScaleSkewVersor3DTransform`)
/// — 12 parameters. This is the multiplicative sibling of the *additive*
/// [`ScaleSkewVersor3DTransform`].
///
/// Parameters are `[v0, v1, v2, tx, ty, tz, s0, s1, s2, k0, k1, k2]`: versor right
/// part (3), translation (3), per-axis scale (3), and **3** skew components (only
/// the upper triangle). `(v0, v1, v2)` uses the same norm-clamping as
/// [`VersorRigid3DTransform`]; `center` fixed.
///
/// # Matrix (multiplicative composition)
///
/// Unlike the additive variants, `ComputeMatrix` **multiplies** the rotation by a
/// scale-then-skew factor:
///
/// ```text
/// K = [ 1  k0 k1 ; 0  1  k2 ; 0  0  1 ]   (unit upper-triangular skew)
/// M = R(versor) · diag(s0, s1, s2) · K
/// ```
///
/// The offset is `translation + center − M·center`.
///
/// # Jacobian
///
/// The analytic Jacobian is ITK's `sympy`-derived expansion of `∂(R·S·K·(p−c))`.
/// **It treats the versor scalar `w` as independent of `(v0, v1, v2)`** (no `1/w`
/// chain-rule term, unlike [`VersorRigid3DTransform`]), so it is exact only in the
/// limit and carries an `O(‖v‖²)` error away from the identity rotation — matching
/// ITK, whose own test validates it only near the identity with a 10% relative
/// tolerance. It is preserved verbatim for parity.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposeScaleSkewVersor3DTransform {
    /// Normalized versor right part.
    vx: f64,
    vy: f64,
    vz: f64,
    /// Normalized versor scalar part `√(1 − vx² − vy² − vz²)`.
    vw: f64,
    /// Per-axis scale, length 3.
    scale: Vec<f64>,
    /// Upper-triangular skew `{k0=(0,1), k1=(0,2), k2=(1,2)}`, length 3.
    skew: Vec<f64>,
    /// Length 3.
    translation: Vec<f64>,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 `R · diag(scale) · K`.
    matrix: Vec<f64>,
    /// Cached `translation + center − M·center`.
    offset: Vec<f64>,
}

impl ComposeScaleSkewVersor3DTransform {
    /// A transform composing: versor rotation `(vx, vy, vz)` about `center`, then
    /// the multiplicative `diag(scale)·K` (upper-triangular skew), then
    /// `translation`. A right part with norm `≥ 1` is scaled to just under 1.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scale: [f64; 3],
        skew: [f64; 3],
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
            [0.0, 0.0, 0.0],
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

    /// The skew components `{k0=(0,1), k1=(0,2), k2=(1,2)}`.
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

    /// Row-major 3×3 matrix `R · diag(scale) · K`.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Translation offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Set the normalized versor from a right part, mirroring
    /// `itk::ComposeScaleSkewVersor3DTransform::SetParameters` (same clamp as
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

    /// Rebuild the cached matrix `M = R · diag(scale) · K` (ITK's `ComputeMatrix`:
    /// versor superclass rotation, times `Q = diag(scale)·K` with `K` unit upper-
    /// triangular) and the offset.
    fn recompute(&mut self) {
        let (x, y, z, w) = (self.vx, self.vy, self.vz, self.vw);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, xw) = (x * y, x * z, x * w);
        let (yz, yw, zw) = (y * z, y * w, z * w);
        #[rustfmt::skip]
        let r = vec![
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw),       2.0 * (xz + yw),
            2.0 * (xy + zw),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw),
            2.0 * (xz - yw),       2.0 * (yz + xw),       1.0 - 2.0 * (xx + yy),
        ];
        let (s0, s1, s2) = (self.scale[0], self.scale[1], self.scale[2]);
        let (k0, k1, k2) = (self.skew[0], self.skew[1], self.skew[2]);
        // Q = diag(scale) · [[1,k0,k1],[0,1,k2],[0,0,1]].
        #[rustfmt::skip]
        let q = vec![
            s0,  s0 * k0, s0 * k1,
            0.0, s1,      s1 * k2,
            0.0, 0.0,     s2,
        ];
        let m = matrix::matmul(&r, &q, 3);
        let m_center = matrix::mat_vec(&m, &self.center, 3);
        self.offset = (0..3)
            .map(|i| self.translation[i] + self.center[i] - m_center[i])
            .collect();
        self.matrix = m;
    }
}

impl TransformBase for ComposeScaleSkewVersor3DTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for ComposeScaleSkewVersor3DTransform {
    fn number_of_parameters(&self) -> usize {
        12
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
        ]
    }

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 12)?;
        self.set_versor(params[0], params[1], params[2]);
        self.translation[0] = params[3];
        self.translation[1] = params[4];
        self.translation[2] = params[5];
        self.scale[0] = params[6];
        self.scale[1] = params[7];
        self.scale[2] = params[8];
        self.skew[0] = params[9];
        self.skew[1] = params[10];
        self.skew[2] = params[11];
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(3);

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // itk::ComposeScaleSkewVersor3DTransform::ComputeJacobianWithRespectToParameters:
        // the sympy-derived expansion of ∂(R·S·K·(p−c)) with the versor scalar w
        // treated as INDEPENDENT of (v0,v1,v2) — preserved verbatim (see the struct
        // docs for the resulting O(‖v‖²) deviation from a true finite difference).
        let (v0, v1, v2, w) = (self.vx, self.vy, self.vz, self.vw);
        let (s0, s1, s2) = (self.scale[0], self.scale[1], self.scale[2]);
        let (k0, k1, k2) = (self.skew[0], self.skew[1], self.skew[2]);
        let x0 = point[0] - self.center[0];
        let x1 = point[1] - self.center[1];
        let x2 = point[2] - self.center[2];

        let (v0v0, v0v1, v0v2, v0w) = (v0 * v0, v0 * v1, v0 * v2, v0 * w);
        let (v1v1, v1v2, v1w) = (v1 * v1, v1 * v2, v1 * w);
        let (v2v2, v2w) = (v2 * v2, v2 * w);

        // Row-major 3×12.
        let mut j = vec![0.0f64; 36];

        // Versor columns 0, 1, 2.
        j[0] = 2.0 * s1 * v1 * x1 + x2 * (2.0 * k2 * s1 * v1 + 2.0 * s2 * v2);
        j[12] = 2.0 * s0 * v1 * x0 + x1 * (2.0 * k0 * s0 * v1 - 4.0 * s1 * v0)
            - x2 * (-2.0 * k1 * s0 * v1 + 4.0 * k2 * s1 * v0 + 2.0 * s2 * w);
        j[24] = 2.0 * s0 * v2 * x0
            + 2.0 * x1 * (k0 * s0 * v2 + s1 * w)
            + x2 * (2.0 * k1 * s0 * v2 + 2.0 * k2 * s1 * w - 4.0 * s2 * v0);

        j[1] = -4.0 * s0 * v1 * x0 - x1 * (4.0 * k0 * s0 * v1 - 2.0 * s1 * v0)
            + x2 * (-4.0 * k1 * s0 * v1 + 2.0 * k2 * s1 * v0 + 2.0 * s2 * w);
        j[13] = 2.0 * k0 * s0 * v0 * x1 + 2.0 * s0 * v0 * x0
            - x2 * (-2.0 * k1 * s0 * v0 - 2.0 * s2 * v2);
        j[25] = -2.0 * s0 * w * x0
            + 2.0 * x1 * (-k0 * s0 * w + s1 * v2)
            + x2 * (-2.0 * k1 * s0 * w + 2.0 * k2 * s1 * v2 - 4.0 * s2 * v1);

        j[2] = -4.0 * s0 * v2 * x0 - x1 * (4.0 * k0 * s0 * v2 + 2.0 * s1 * w)
            + x2 * (-4.0 * k1 * s0 * v2 - 2.0 * k2 * s1 * w + 2.0 * s2 * v0);
        j[14] = 2.0 * s0 * w * x0 + x1 * (2.0 * k0 * s0 * w - 4.0 * s1 * v2)
            - x2 * (-2.0 * k1 * s0 * w + 4.0 * k2 * s1 * v2 - 2.0 * s2 * v1);
        j[26] = 2.0 * s0 * v0 * x0
            + 2.0 * x1 * (k0 * s0 * v0 + s1 * v1)
            + x2 * (2.0 * k1 * s0 * v0 + 2.0 * k2 * s1 * v1);

        // Translation identity block: columns 3, 4, 5.
        j[3] = 1.0;
        j[16] = 1.0;
        j[29] = 1.0;

        // Scale columns 6, 7, 8.
        j[6] = -k0 * x1 * (2.0 * v1v1 + 2.0 * v2v2 - 1.0)
            - k1 * x2 * (2.0 * v1v1 + 2.0 * v2v2 - 1.0)
            - x0 * (2.0 * v1v1 + 2.0 * v2v2 - 1.0);
        j[18] =
            2.0 * k0 * x1 * (v0v1 + v2w) + 2.0 * k1 * x2 * (v0v1 + v2w) + 2.0 * x0 * (v0v1 + v2w);
        j[30] =
            2.0 * k0 * x1 * (v0v2 - v1w) + 2.0 * k1 * x2 * (v0v2 - v1w) + 2.0 * x0 * (v0v2 - v1w);

        j[7] = 2.0 * k2 * x2 * (v0v1 - v2w) - x1 * (-2.0 * v0v1 + 2.0 * v2w);
        j[19] = -k2 * x2 * (2.0 * v0v0 + 2.0 * v2v2 - 1.0) + x1 * (-2.0 * v0v0 - 2.0 * v2v2 + 1.0);
        j[31] = 2.0 * k2 * x2 * (v0w + v1v2) + 2.0 * x1 * (v0w + v1v2);

        j[8] = x2 * (2.0 * v0v2 + 2.0 * v1w);
        j[20] = -x2 * (2.0 * v0w - 2.0 * v1v2);
        j[32] = x2 * (-2.0 * v0v0 - 2.0 * v1v1 + 1.0);

        // Skew columns 9, 10, 11.
        j[9] = -s0 * x1 * (2.0 * v1v1 + 2.0 * v2v2 - 1.0);
        j[21] = 2.0 * s0 * x1 * (v0v1 + v2w);
        j[33] = 2.0 * s0 * x1 * (v0v2 - v1w);

        j[10] = -s0 * x2 * (2.0 * v1v1 + 2.0 * v2v2 - 1.0);
        j[22] = 2.0 * s0 * x2 * (v0v1 + v2w);
        j[34] = 2.0 * s0 * x2 * (v0v2 - v1w);

        j[11] = 2.0 * s1 * x2 * (v0v1 - v2w);
        j[23] = -s1 * x2 * (2.0 * v0v0 + 2.0 * v2v2 - 1.0);
        j[35] = 2.0 * s1 * x2 * (v0w + v1v2);
        j
    }
}

impl CenteredTransform for ComposeScaleSkewVersor3DTransform {
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

/// A pure 3-D rotation about a fixed center, with **no translation
/// parameter**: `y = R(versor)·(x − center) + center`, mirroring
/// `itk::VersorTransform`.
///
/// Parameters are `[vx, vy, vz]` only — the versor's right part (axis·sin(θ/2)),
/// with the same norm-clamping as [`VersorRigid3DTransform`]. Unlike
/// `VersorRigid3DTransform`, there is no translation concept at all: ITK's own
/// class docs carry a standing TODO — "Need to make sure that the translation
/// parameters in the base class cannot be set to non-zero values"
/// (`itkVersorTransform.h:40-41`) — so this port omits translation entirely
/// rather than expose a base-class setter ITK itself flags as something that
/// should never be used. Only the fixed `center` is settable, via the inherent
/// [`VersorTransform::set_center`] (not [`CenteredTransform`], which would
/// require a meaningful `set_translation`).
#[derive(Clone, Debug, PartialEq)]
pub struct VersorTransform {
    /// Normalized versor right part.
    vx: f64,
    vy: f64,
    vz: f64,
    /// Normalized versor scalar part `√(1 − vx² − vy² − vz²)`.
    vw: f64,
    /// Length 3, fixed (not a parameter).
    center: Vec<f64>,
    /// Cached row-major 3×3 rotation.
    matrix: Vec<f64>,
    /// Cached `center − M·center`.
    offset: Vec<f64>,
}

impl VersorTransform {
    /// A pure rotation whose versor right part is `(vx, vy, vz)` (axis·sin(θ/2)),
    /// about `center`. A right part with norm `≥ 1` is scaled to just under 1,
    /// matching ITK's `SetParameters`.
    pub fn new(vx: f64, vy: f64, vz: f64, center: [f64; 3]) -> Self {
        let mut t = Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            vw: 1.0,
            center: center.to_vec(),
            matrix: vec![0.0; 9],
            offset: vec![0.0; 3],
        };
        t.set_versor(vx, vy, vz);
        t.recompute();
        t
    }

    /// The identity transform (versor `(0,0,0; w=1)`, center at the origin).
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
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

    /// The fixed center of rotation.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major 3×3 rotation matrix.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Offset actually applied (`y = M·x + offset`).
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Set the fixed center of rotation (mirrors `itk::VersorTransform`'s
    /// inherited `SetCenter`).
    pub fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), 3, "center length");
        self.center.copy_from_slice(center);
        self.recompute();
    }

    /// Set the normalized versor from a right part, mirroring
    /// `itk::VersorTransform::SetParameters` + `Versor::Set(axis)`: scale a
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
        self.offset = (0..3).map(|i| self.center[i] - m_center[i]).collect();
        self.matrix = m;
    }
}

impl TransformBase for VersorTransform {
    matrix_jacobian_wrt_position!();
    matrix_point_map_stages!();

    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), 3);
        let mx = matrix::mat_vec(&self.matrix, point, 3);
        (0..3).map(|d| mx[d] + self.offset[d]).collect()
    }

    fn dimension(&self) -> usize {
        3
    }
}

impl ParametricTransform for VersorTransform {
    fn number_of_parameters(&self) -> usize {
        3
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.vx, self.vy, self.vz]
    }

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, 3)?;
        self.set_versor(params[0], params[1], params[2]);
        self.recompute();
        Ok(())
    }

    center_fixed_parameters!(3);

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // Analytic ∂y/∂versor from itk::VersorTransform (divided by vw).
        // Row-major 3×3 — no translation columns since translation is not a
        // parameter (see struct docs).
        let (vx, vy, vz, vw) = (self.vx, self.vy, self.vz, self.vw);
        let (px, py, pz) = (
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        );
        let (vxx, vyy, vzz, vww) = (vx * vx, vy * vy, vz * vz, vw * vw);
        let (vxy, vxz, vxw) = (vx * vy, vx * vz, vx * vw);
        let (vyz, vyw, vzw) = (vy * vz, vy * vw, vz * vw);

        let mut j = vec![0.0f64; 9];
        j[0] = 2.0 * ((vyw + vxz) * py + (vzw - vxy) * pz) / vw;
        j[3] = 2.0 * ((vyw - vxz) * px - 2.0 * vxw * py + (vxx - vww) * pz) / vw;
        j[6] = 2.0 * ((vzw + vxy) * px + (vww - vxx) * py - 2.0 * vxw * pz) / vw;

        j[1] = 2.0 * (-2.0 * vyw * px + (vxw + vyz) * py + (vww - vyy) * pz) / vw;
        j[4] = 2.0 * ((vxw - vyz) * px + (vzw + vxy) * pz) / vw;
        j[7] = 2.0 * ((vyy - vww) * px + (vzw - vxy) * py - 2.0 * vyw * pz) / vw;

        j[2] = 2.0 * (-2.0 * vzw * px + (vzz - vww) * py + (vxw - vyz) * pz) / vw;
        j[5] = 2.0 * ((vww - vzz) * px - 2.0 * vzw * py + (vyw + vxz) * pz) / vw;
        j[8] = 2.0 * ((vxw + vyz) * px + (vyw - vxz) * py) / vw;
        j
    }
}

/// An anisotropic per-axis scale about a fixed center:
/// `y = (x − center) ⊙ scale + center`, mirroring `itk::ScaleTransform`.
///
/// Parameters are `[scale_0, ..., scale_{dim−1}]`. There is **no translation
/// term**: ITK's own `TransformPoint` override
/// (`itkScaleTransform.hxx:105-117`) computes `(x − center) ⊙ scale + center`
/// directly and never references the inherited `MatrixOffsetTransformBase`
/// translation/offset, so a translation set through that inherited (but
/// otherwise-unused) base-class setter has no effect on this transform. This
/// port therefore exposes only the inherent [`ScaleTransform::set_center`],
/// not [`CenteredTransform`] (whose `set_translation` would be a silent
/// no-op).
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleTransform {
    dim: usize,
    /// Length `dim`.
    scale: Vec<f64>,
    /// Length `dim`, fixed (not a parameter).
    center: Vec<f64>,
}

impl ScaleTransform {
    /// An anisotropic scale by `scale` about `center`. Panics if the lengths
    /// disagree or are empty.
    pub fn new(scale: Vec<f64>, center: Vec<f64>) -> Self {
        assert_eq!(
            scale.len(),
            center.len(),
            "scale and center must have the same length"
        );
        assert!(!scale.is_empty(), "dimension must be >= 1");
        Self {
            dim: scale.len(),
            scale,
            center,
        }
    }

    /// The identity scale transform of the given dimension (all scales 1,
    /// center at the origin).
    pub fn identity(dim: usize) -> Self {
        Self {
            dim,
            scale: vec![1.0; dim],
            center: vec![0.0; dim],
        }
    }

    /// Per-axis scale factors.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// The fixed center scaling originates from.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Set the fixed center of scaling (mirrors `itk::ScaleTransform`'s
    /// inherited `SetCenter`; see the struct docs for why there is no paired
    /// `set_translation`).
    pub fn set_center(&mut self, center: &[f64]) {
        assert_eq!(center.len(), self.dim, "center length");
        self.center.copy_from_slice(center);
    }
}

impl TransformBase for ScaleTransform {
    /// `T(x)ᵢ = (xᵢ − cᵢ)·sᵢ + cᵢ`, so `dT/dx = diag(s)`
    /// (`itk::ScaleTransform::ComputeJacobianWithRespectToPosition`).
    fn jacobian_wrt_position(&self, _point: &[f64]) -> Vec<f64> {
        diagonal(&self.scale)
    }

    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        debug_assert_eq!(point.len(), self.dim);
        (0..self.dim)
            .map(|d| (point[d] - self.center[d]) * self.scale[d] + self.center[d])
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

impl ParametricTransform for ScaleTransform {
    fn number_of_parameters(&self) -> usize {
        self.dim
    }

    fn parameters(&self) -> Vec<f64> {
        self.scale.clone()
    }

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_len(params, self.dim)?;
        self.scale.copy_from_slice(params);
        Ok(())
    }

    /// The center of scaling — `itk::ScaleTransform` derives from
    /// `MatrixOffsetTransformBase`, whose fixed parameters are the center.
    fn fixed_parameters(&self) -> Vec<f64> {
        self.center.clone()
    }

    fn number_of_fixed_parameters(&self) -> usize {
        self.dim
    }

    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        check_fixed_len(params, self.dim, "the center of scaling")?;
        self.center.copy_from_slice(params);
        Ok(())
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // ∂yᵢ/∂scale_k = δᵢₖ · (xᵢ − centerᵢ) — itk::ScaleTransform
        // ComputeJacobianWithRespectToParameters.
        let mut j = vec![0.0; self.dim * self.dim];
        for d in 0..self.dim {
            j[d * self.dim + d] = point[d] - self.center[d];
        }
        j
    }
}

/// A [`ScaleTransform`] whose optimizable parameters are `log(scale)` rather
/// than `scale` itself, mirroring `itk::ScaleLogarithmicTransform`. Taking the
/// log linearizes the parameterization: an additive optimizer step in
/// log-space is a multiplicative step in scale-space, so growing and shrinking
/// by the same step size become symmetric.
///
/// `transform_point`, `dimension`, and the fixed `center` behave exactly as
/// [`ScaleTransform`] (this port also omits `CenteredTransform` for the same
/// reason — see the `ScaleTransform` docs); only `parameters()` /
/// `set_parameters()` / `jacobian_wrt_parameters()` differ.
///
/// **Deviation from `itkScaleLogarithmicTransform.hxx`:** ITK's
/// `ComputeJacobianWithRespectToParameters` writes `scale[d] * p[d]`, omitting
/// the `− center[d]` term that its own `ScaleTransform` superclass Jacobian
/// applies three lines away in the sibling file. By the chain rule
/// (`s = exp(u)` ⟹ `∂y/∂u = s·∂y/∂s`) the correct entry is
/// `scale[d] * (p[d] − center[d])`, which is what this port computes: using
/// ITK's literal formula would disagree with this port's own `transform_point`
/// at any non-zero center, which the mandatory off-center finite-difference
/// test below would catch. The bug only manifests when `center != 0`, and
/// `itkScaleLogarithmicTransformTest` exercised no Jacobian at all, which
/// explains why it went unnoticed upstream. Fixed upstream in
/// <https://github.com/InsightSoftwareConsortium/ITK/pull/6569>.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleLogarithmicTransform {
    inner: ScaleTransform,
}

impl ScaleLogarithmicTransform {
    /// A logarithmic-scale transform with (linear, not log) scale `scale`
    /// about `center` — matching the crate's convention of constructing from
    /// the transform's direct effect, not its parameter encoding.
    pub fn new(scale: Vec<f64>, center: Vec<f64>) -> Self {
        Self {
            inner: ScaleTransform::new(scale, center),
        }
    }

    /// The identity transform of the given dimension (all scales 1, center at
    /// the origin).
    pub fn identity(dim: usize) -> Self {
        Self {
            inner: ScaleTransform::identity(dim),
        }
    }

    /// Per-axis scale factors (linear, not log).
    pub fn scale(&self) -> &[f64] {
        self.inner.scale()
    }

    /// The fixed center scaling originates from.
    pub fn center(&self) -> &[f64] {
        self.inner.center()
    }

    /// Set the fixed center of scaling.
    pub fn set_center(&mut self, center: &[f64]) {
        self.inner.set_center(center);
    }
}

impl TransformBase for ScaleLogarithmicTransform {
    /// Identical to [`ScaleTransform`]: the logarithmic parameterization changes
    /// the parameter Jacobian, not the spatial one.
    fn jacobian_wrt_position(&self, point: &[f64]) -> Vec<f64> {
        self.inner.jacobian_wrt_position(point)
    }

    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        self.inner.transform_point(point)
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

impl ParametricTransform for ScaleLogarithmicTransform {
    fn number_of_parameters(&self) -> usize {
        self.inner.number_of_parameters()
    }

    fn parameters(&self) -> Vec<f64> {
        self.inner.scale().iter().map(|s| s.ln()).collect()
    }

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        let scale: Vec<f64> = params.iter().map(|p| p.exp()).collect();
        self.inner.set_parameters(&scale)
    }

    /// Delegates to [`ScaleTransform`]: the center of scaling.
    fn fixed_parameters(&self) -> Vec<f64> {
        self.inner.fixed_parameters()
    }

    fn number_of_fixed_parameters(&self) -> usize {
        self.inner.number_of_fixed_parameters()
    }

    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        self.inner.set_fixed_parameters(params)
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        // ∂yᵢ/∂log(scale_k) = δᵢₖ · scale_i · (pᵢ − centerᵢ) — see the struct
        // docs for the ITK center-omission this corrects.
        let dim = self.inner.dimension();
        let scale = self.inner.scale();
        let center = self.inner.center();
        let mut j = vec![0.0; dim * dim];
        for d in 0..dim {
            j[d * dim + d] = scale[d] * (point[d] - center[d]);
        }
        j
    }
}

// ---------------------------------------------------------------------------
// Inverses
//
// Every matrix-offset transform inverts through the same two steps as
// `itk::MatrixOffsetTransformBase::GetInverse` (`itkMatrixOffsetTransformBase.hxx:425-444`):
//
// 1. the inverse's center is this one's, its matrix is `M⁻¹`, and its offset is
//    `−M⁻¹·offset` (returning `false` when `M` is singular);
// 2. `ComputeTranslation` recovers the inverse's translation from that offset,
//    and `ComputeMatrixParameters` — the per-class virtual — recovers the
//    class's own parameters (angle, versor, scale, skew) from `M⁻¹`.
//
// The parameter-recovery step is what decides whether a class *has* an inverse:
// `ScaleVersor3DTransform` and `ScaleSkewVersor3DTransform` both raise
// `itkExceptionStringMacro("Setting the matrix of a ... transform is not
// supported at this time.")` from `ComputeMatrixParameters`
// (`itkScaleVersor3DTransform.hxx:201-204`, `itkScaleSkewVersor3DTransform.hxx:233-236`),
// so `GetInverse` on those throws upstream and returns
// [`TransformError::NoInverse`] here.
// ---------------------------------------------------------------------------

/// The linear map of the inverse of `y = M·x + offset`, i.e. `(M⁻¹, −M⁻¹·offset)`.
/// `None` when `M` is singular — the `m_Singular` early return of
/// `itk::MatrixOffsetTransformBase::GetInverse`.
fn invert_matrix_offset(
    matrix: &[f64],
    offset: &[f64],
    dim: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let inv = matrix::invert(matrix, dim)?;
    let inv_offset = matrix::mat_vec(&inv, offset, dim)
        .into_iter()
        .map(|v| -v)
        .collect();
    Some((inv, inv_offset))
}

/// `itk::MatrixOffsetTransformBase::ComputeTranslation`:
/// `translation = offset − (center − M·center)`.
fn translation_from_offset(matrix: &[f64], offset: &[f64], center: &[f64], dim: usize) -> Vec<f64> {
    let m_center = matrix::mat_vec(matrix, center, dim);
    (0..dim)
        .map(|d| offset[d] - center[d] + m_center[d])
        .collect()
}

/// Determinant of a row-major 3×3 matrix.
fn determinant3(m: &[f64]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// The right part `(x, y, z)` of the versor equivalent to the row-major 3×3
/// rotation `matrix`, via the branch method of `itk::Versor<T>::Set(const
/// MatrixType&)` (`itkVersor.hxx:301-381`, Shepperd 1978), normalized and
/// canonicalized to a non-negative scalar part.
///
/// Errors with [`TransformError::NotARotationMatrix`] when `matrix` is not
/// orthonormal or is a reflection, to within `itk::Versor<double>::Epsilon() =
/// 1e-10` — the `itkGenericExceptionMacro` guard in `Versor::Set`.
fn versor_right_part_from_matrix(matrix: &[f64]) -> Result<(f64, f64, f64)> {
    assert_eq!(matrix.len(), 9, "matrix must be row-major 3x3");
    const EPS: f64 = 1e-10;

    let m = |r: usize, c: usize| matrix[r * 3 + c];
    // I = m·mᵀ; check orthonormality and that it isn't a reflection.
    let dot_row = |a: usize, b: usize| (0..3).map(|k| m(a, k) * m(b, k)).sum::<f64>();
    let orthonormal = dot_row(0, 1).abs() <= EPS
        && dot_row(0, 2).abs() <= EPS
        && dot_row(1, 2).abs() <= EPS
        && (dot_row(0, 0) - 1.0).abs() <= EPS
        && (dot_row(1, 1) - 1.0).abs() <= EPS
        && (dot_row(2, 2) - 1.0).abs() <= EPS;
    if !orthonormal || determinant3(matrix) < 0.0 {
        return Err(TransformError::NotARotationMatrix);
    }

    let (mut x, mut y, mut z, mut w);
    let trace = m(0, 0) + m(1, 1) + m(2, 2) + 1.0;
    if trace > EPS {
        let s = 0.5 / trace.sqrt();
        w = 0.25 / s;
        x = (m(2, 1) - m(1, 2)) * s;
        y = (m(0, 2) - m(2, 0)) * s;
        z = (m(1, 0) - m(0, 1)) * s;
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let s = 2.0 * (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt();
        x = 0.25 * s;
        y = (m(0, 1) + m(1, 0)) / s;
        z = (m(0, 2) + m(2, 0)) / s;
        w = (m(1, 2) - m(2, 1)) / s;
    } else if m(1, 1) > m(2, 2) {
        let s = 2.0 * (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt();
        x = (m(0, 1) + m(1, 0)) / s;
        y = 0.25 * s;
        z = (m(1, 2) + m(2, 1)) / s;
        w = (m(0, 2) - m(2, 0)) / s;
    } else {
        let s = 2.0 * (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt();
        x = (m(0, 2) + m(2, 0)) / s;
        y = (m(1, 2) + m(2, 1)) / s;
        z = 0.25 * s;
        w = (m(0, 1) - m(1, 0)) / s;
    }

    // Normalize (`Versor::Normalize`), then canonicalize to a non-negative
    // scalar part by negating all four components together — a versor and its
    // negation represent the same rotation only if negated as a whole (the
    // double cover of SO(3)), and every type here reconstructs
    // `w = +√(1 − x² − y² − z²)` from the stored right part.
    let norm = (x * x + y * y + z * z + w * w).sqrt();
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;
    if w < 0.0 {
        x = -x;
        y = -y;
        z = -z;
    }
    Ok((x, y, z))
}

/// The rotation angle of a row-major 2×2 rotation matrix, porting
/// `itk::Rigid2DTransform::ComputeMatrixParameters`
/// (`itkRigid2DTransform.hxx:88-106`): `acos(r₀₀)`, negated when `r₁₀ < 0`.
///
/// ITK first replaces the matrix by `U·Vᵀ` of its SVD (the closest rotation);
/// the callers here always pass an exact rotation's inverse, for which the two
/// agree to a few ULPs. The `acos` argument is clamped into `[-1, 1]` so a
/// rounding excursion past `±1` yields `0`/`π` rather than `NaN`.
fn rotation_angle_2d(matrix: &[f64]) -> f64 {
    let angle = matrix[0].clamp(-1.0, 1.0).acos();
    if matrix[2] < 0.0 { -angle } else { angle }
}

/// The Euler angles `(x, y, z)` of a row-major 3×3 rotation matrix, porting
/// `itk::Euler3DTransform::ComputeMatrixParameters`
/// (`itkEuler3DTransform.hxx:179-226`) for both composition orders. The `asin`
/// arguments are clamped into `[-1, 1]` (ITK does not clamp).
fn euler_angles_from_matrix(matrix: &[f64], compute_zyx: bool) -> (f64, f64, f64) {
    let m = |r: usize, c: usize| matrix[r * 3 + c];
    if compute_zyx {
        let angle_y = -m(2, 0).clamp(-1.0, 1.0).asin();
        let c = angle_y.cos();
        if c.abs() > 0.00005 {
            let angle_x = (m(2, 1) / c).atan2(m(2, 2) / c);
            let angle_z = (m(1, 0) / c).atan2(m(0, 0) / c);
            (angle_x, angle_y, angle_z)
        } else {
            (0.0, angle_y, (-m(0, 1)).atan2(m(1, 1)))
        }
    } else {
        let angle_x = m(2, 1).clamp(-1.0, 1.0).asin();
        let a = angle_x.cos();
        if a.abs() > 0.00005 {
            let angle_y = (-m(2, 0) / a).atan2(m(2, 2) / a);
            let angle_z = (-m(0, 1) / a).atan2(m(1, 1) / a);
            (angle_x, angle_y, angle_z)
        } else {
            (angle_x, m(1, 0).atan2(m(0, 0)), 0.0)
        }
    }
}

impl TranslationTransform {
    /// The inverse translation, `−t` (`itk::TranslationTransform::GetInverse`).
    /// Always exists.
    pub fn inverse(&self) -> Self {
        Self::new(self.translation.iter().map(|t| -t).collect())
    }
}

impl AffineTransform {
    /// The inverse affine transform about the same center. Errors with
    /// [`TransformError::NoInverse`] when the matrix is singular — where
    /// `itk::MatrixOffsetTransformBase::GetInverse` returns `false`.
    pub fn inverse(&self) -> Result<Self> {
        let dim = self.dim;
        let (matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, dim)
            .ok_or(TransformError::NoInverse("the affine matrix is singular"))?;
        let translation = translation_from_offset(&matrix, &offset, &self.center, dim);
        Ok(Self::new(dim, matrix, translation, self.center.clone()))
    }
}

impl Euler2DTransform {
    /// The inverse rigid transform about the same center. Errors with
    /// [`TransformError::NoInverse`] only if the cached rotation matrix is
    /// singular, which a rotation never is.
    pub fn inverse(&self) -> Result<Self> {
        let (matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, 2)
            .ok_or(TransformError::NoInverse("the rotation matrix is singular"))?;
        let t = translation_from_offset(&matrix, &offset, &self.center, 2);
        Ok(Self::new(
            rotation_angle_2d(&matrix),
            [t[0], t[1]],
            [self.center[0], self.center[1]],
        ))
    }
}

impl Similarity2DTransform {
    /// The inverse similarity transform about the same center: reciprocal scale,
    /// negated angle. Errors with [`TransformError::NoInverse`] when the scale is
    /// zero — `itk::Similarity2DTransform::ComputeMatrixParameters` throws "Bad
    /// Rotation Matrix. Scale cannot be zero." there
    /// (`itkSimilarity2DTransform.hxx:144-155`).
    pub fn inverse(&self) -> Result<Self> {
        let (matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, 2).ok_or(
            TransformError::NoInverse("the similarity matrix is singular"),
        )?;
        // ITK: m_Scale = sqrt(sqr(m[0][0]) + sqr(m[0][1])); angle from m/scale.
        let scale = (matrix[0] * matrix[0] + matrix[1] * matrix[1]).sqrt();
        if scale < f64::MIN_POSITIVE {
            return Err(TransformError::NoInverse("the similarity scale is zero"));
        }
        let mut angle = (matrix[0] / scale).clamp(-1.0, 1.0).acos();
        if matrix[2] < 0.0 {
            angle = -angle;
        }
        let t = translation_from_offset(&matrix, &offset, &self.center, 2);
        Ok(Self::new(
            scale,
            angle,
            [t[0], t[1]],
            [self.center[0], self.center[1]],
        ))
    }
}

impl Euler3DTransform {
    /// The inverse rigid transform about the same center, carrying this
    /// transform's `compute_zyx` flag over — ITK's `GetInverse` copies it with
    /// `inverse->SetFixedParameters(this->GetFixedParameters())` before it
    /// extracts the angles, and the extraction itself branches on it.
    pub fn inverse(&self) -> Result<Self> {
        let (matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, 3)
            .ok_or(TransformError::NoInverse("the rotation matrix is singular"))?;
        let t = translation_from_offset(&matrix, &offset, &self.center, 3);
        let (angle_x, angle_y, angle_z) = euler_angles_from_matrix(&matrix, self.compute_zyx);
        let mut inverse = Self::new(
            angle_x,
            angle_y,
            angle_z,
            [t[0], t[1], t[2]],
            [self.center[0], self.center[1], self.center[2]],
        );
        inverse.set_compute_zyx(self.compute_zyx);
        Ok(inverse)
    }
}

impl VersorTransform {
    /// The inverse rotation about the same center. The inverse of `R·(x − c) + c`
    /// is `Rᵀ·(x − c) + c`, whose `MatrixOffsetTransformBase` translation is
    /// exactly zero — so it is again a pure versor transform.
    pub fn inverse(&self) -> Result<Self> {
        let matrix = matrix::invert(&self.matrix, 3)
            .ok_or(TransformError::NoInverse("the rotation matrix is singular"))?;
        let (x, y, z) = versor_right_part_from_matrix(&matrix)?;
        Ok(Self::new(
            x,
            y,
            z,
            [self.center[0], self.center[1], self.center[2]],
        ))
    }
}

impl VersorRigid3DTransform {
    /// The inverse rigid transform about the same center.
    pub fn inverse(&self) -> Result<Self> {
        let (matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, 3)
            .ok_or(TransformError::NoInverse("the rotation matrix is singular"))?;
        let t = translation_from_offset(&matrix, &offset, &self.center, 3);
        let (x, y, z) = versor_right_part_from_matrix(&matrix)?;
        Ok(Self::new(
            x,
            y,
            z,
            [t[0], t[1], t[2]],
            [self.center[0], self.center[1], self.center[2]],
        ))
    }
}

impl Similarity3DTransform {
    /// The inverse similarity transform about the same center, porting
    /// `itk::Similarity3DTransform::ComputeMatrixParameters`
    /// (`itkSimilarity3DTransform.hxx:288-300`): the scale is the cube root of
    /// the inverse matrix's determinant, and the versor comes from the matrix
    /// divided by it.
    pub fn inverse(&self) -> Result<Self> {
        let (matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, 3).ok_or(
            TransformError::NoInverse("the similarity matrix is singular"),
        )?;
        let scale = determinant3(&matrix).cbrt();
        if scale == 0.0 || !scale.is_finite() {
            return Err(TransformError::NoInverse("the similarity scale is zero"));
        }
        let rotation: Vec<f64> = matrix.iter().map(|v| v / scale).collect();
        let (x, y, z) = versor_right_part_from_matrix(&rotation)?;
        let t = translation_from_offset(&matrix, &offset, &self.center, 3);
        Ok(Self::new(
            scale,
            x,
            y,
            z,
            [t[0], t[1], t[2]],
            [self.center[0], self.center[1], self.center[2]],
        ))
    }
}

impl ComposeScaleSkewVersor3DTransform {
    /// The inverse transform about the same center, porting
    /// `itk::ComposeScaleSkewVersor3DTransform::ComputeMatrixParameters`
    /// (`itkComposeScaleSkewVersor3DTransform.hxx:240-292`): a Gram–Schmidt
    /// QR-style factorization of the inverse matrix's columns recovers the
    /// scales, the three skews and the rotation, with the first scale negated if
    /// the residual rotation is a reflection.
    pub fn inverse(&self) -> Result<Self> {
        let (inverse_matrix, offset) = invert_matrix_offset(&self.matrix, &self.offset, 3)
            .ok_or(TransformError::NoInverse("the matrix is singular"))?;
        // ITK's ComputeMatrixParameters factorizes a *copy*; the translation is
        // computed from the untouched inverse matrix beforehand.
        let t = translation_from_offset(&inverse_matrix, &offset, &self.center, 3);
        let mut m = inverse_matrix;
        let col_norm =
            |m: &[f64], c: usize| (m[c] * m[c] + m[3 + c] * m[3 + c] + m[6 + c] * m[6 + c]).sqrt();
        let scale_col = |m: &mut [f64], c: usize, s: f64| {
            m[c] /= s;
            m[3 + c] /= s;
            m[6 + c] /= s;
        };

        let mut scale = [0.0f64; 3];
        let mut skew = [0.0f64; 3];

        scale[0] = col_norm(&m, 0);
        if scale[0] == 0.0 {
            return Err(TransformError::NoInverse("a scale factor is zero"));
        }
        scale_col(&mut m, 0, scale[0]);

        let ortho = m[0] * m[1] + m[3] * m[4] + m[6] * m[7];
        m[1] -= ortho * m[0];
        m[4] -= ortho * m[3];
        m[7] -= ortho * m[6];
        scale[1] = col_norm(&m, 1);
        if scale[1] == 0.0 {
            return Err(TransformError::NoInverse("a scale factor is zero"));
        }
        scale_col(&mut m, 1, scale[1]);
        skew[0] = ortho / scale[0];

        let ortho0 = m[0] * m[2] + m[3] * m[5] + m[6] * m[8];
        let ortho1 = m[1] * m[2] + m[4] * m[5] + m[7] * m[8];
        m[2] -= ortho0 * m[0] + ortho1 * m[1];
        m[5] -= ortho0 * m[3] + ortho1 * m[4];
        m[8] -= ortho0 * m[6] + ortho1 * m[7];
        scale[2] = col_norm(&m, 2);
        if scale[2] == 0.0 {
            return Err(TransformError::NoInverse("a scale factor is zero"));
        }
        scale_col(&mut m, 2, scale[2]);
        skew[1] = ortho0 / scale[0];
        skew[2] = ortho1 / scale[1];

        if determinant3(&m) < 0.0 {
            scale[0] = -scale[0];
            m[0] = -m[0];
            m[3] = -m[3];
            m[6] = -m[6];
        }

        let (x, y, z) = versor_right_part_from_matrix(&m)?;
        Ok(Self::new(
            scale,
            skew,
            x,
            y,
            z,
            [t[0], t[1], t[2]],
            [self.center[0], self.center[1], self.center[2]],
        ))
    }
}

impl ScaleTransform {
    /// The reciprocal scale about the same center
    /// (`itk::ScaleTransform::GetInverse`, `itkScaleTransform.hxx:167-181`).
    ///
    /// ITK does not guard against a zero scale factor — `1.0 / m_Scale[i]` is
    /// then an infinity, which the returned transform carries. This port
    /// reproduces that rather than erroring.
    pub fn inverse(&self) -> Self {
        Self::new(
            self.scale.iter().map(|s| 1.0 / s).collect(),
            self.center.clone(),
        )
    }
}

impl ScaleLogarithmicTransform {
    /// The reciprocal scale about the same center — see [`ScaleTransform::inverse`],
    /// which this delegates to (`itk::ScaleLogarithmicTransform` derives from
    /// `itk::ScaleTransform` and does not override `GetInverse`).
    pub fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every supported inverse must undo its transform on a scatter of points.
    fn assert_undoes<T: TransformBase, U: TransformBase>(t: &T, inv: &U, dim: usize) {
        let points: &[&[f64]] = &[
            &[0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            &[-4.5, 0.25, 7.0],
            &[100.0, -100.0, 0.5],
        ];
        for p in points {
            let p = &p[..dim];
            let back = inv.transform_point(&t.transform_point(p));
            for d in 0..dim {
                assert!(
                    (back[d] - p[d]).abs() < 1e-9,
                    "round trip failed at {p:?}: got {back:?}"
                );
            }
        }
    }

    #[test]
    fn translation_inverse_undoes_the_translation() {
        let t = TranslationTransform::new(vec![2.0, -3.0]);
        assert_eq!(t.inverse().translation(), [-2.0, 3.0]);
        assert_undoes(&t, &t.inverse(), 2);
    }

    #[test]
    fn affine_inverse_undoes_the_affine() {
        let a = AffineTransform::new(2, vec![2.0, 1.0, 0.0, 3.0], vec![5.0, -1.0], vec![1.0, 2.0]);
        let inv = a.inverse().unwrap();
        assert_eq!(inv.center(), [1.0, 2.0]);
        assert_undoes(&a, &inv, 2);
    }

    #[test]
    fn affine_inverse_errors_on_a_singular_matrix() {
        let a = AffineTransform::new(2, vec![1.0, 2.0, 2.0, 4.0], vec![0.0, 0.0], vec![0.0, 0.0]);
        assert!(matches!(a.inverse(), Err(TransformError::NoInverse(_))));
    }

    #[test]
    fn euler2d_inverse_negates_the_angle_about_the_same_center() {
        let e = Euler2DTransform::new(0.7, [3.0, -2.0], [1.0, 5.0]);
        let inv = e.inverse().unwrap();
        assert!((inv.angle() + 0.7).abs() < 1e-12, "{}", inv.angle());
        assert_eq!(inv.center(), [1.0, 5.0]);
        assert_undoes(&e, &inv, 2);
    }

    #[test]
    fn similarity2d_inverse_reciprocates_the_scale() {
        let s = Similarity2DTransform::new(2.5, 0.3, [1.0, 2.0], [-1.0, 4.0]);
        let inv = s.inverse().unwrap();
        assert!((inv.scale() - 0.4).abs() < 1e-12, "{}", inv.scale());
        assert!((inv.angle() + 0.3).abs() < 1e-12, "{}", inv.angle());
        assert_undoes(&s, &inv, 2);
    }

    #[test]
    fn euler3d_inverse_undoes_the_rotation_in_both_composition_orders() {
        for zyx in [false, true] {
            let mut e = Euler3DTransform::new(0.3, -0.4, 0.5, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
            e.set_compute_zyx(zyx);
            let inv = e.inverse().unwrap();
            assert_eq!(inv.compute_zyx(), zyx);
            assert_eq!(inv.center(), [4.0, 5.0, 6.0]);
            assert_undoes(&e, &inv, 3);
        }
    }

    #[test]
    fn euler3d_inverse_handles_the_gimbal_lock_branch() {
        // angle_x = pi/2 drives cos(angle_x) to zero in the non-ZYX branch.
        let e = Euler3DTransform::new(
            std::f64::consts::FRAC_PI_2,
            0.0,
            0.4,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        );
        assert_undoes(&e, &e.inverse().unwrap(), 3);
    }

    #[test]
    fn versor_inverse_is_a_pure_rotation_about_the_same_center() {
        let v = VersorTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0]);
        let inv = v.inverse().unwrap();
        assert_eq!(inv.center(), [1.0, 2.0, 3.0]);
        assert_undoes(&v, &inv, 3);
    }

    #[test]
    fn versor_rigid3d_inverse_undoes_the_transform() {
        let v = VersorRigid3DTransform::new(0.1, -0.2, 0.3, [4.0, 5.0, 6.0], [1.0, 1.0, 1.0]);
        assert_undoes(&v, &v.inverse().unwrap(), 3);
    }

    #[test]
    fn similarity3d_inverse_reciprocates_the_scale() {
        let s = Similarity3DTransform::new(2.0, 0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]);
        let inv = s.inverse().unwrap();
        assert!((inv.scale() - 0.5).abs() < 1e-12, "{}", inv.scale());
        assert_undoes(&s, &inv, 3);
    }

    #[test]
    fn compose_scale_skew_versor3d_inverse_undoes_the_transform() {
        let c = ComposeScaleSkewVersor3DTransform::new(
            [2.0, 1.5, 0.5],
            [0.1, 0.2, -0.3],
            0.1,
            0.2,
            0.3,
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 1.0],
        );
        assert_undoes(&c, &c.inverse().unwrap(), 3);
    }

    #[test]
    fn scale_inverse_reciprocates_each_factor() {
        let s = ScaleTransform::new(vec![2.0, 4.0], vec![1.0, 1.0]);
        let inv = s.inverse();
        assert_eq!(inv.scale(), [0.5, 0.25]);
        assert_eq!(inv.center(), [1.0, 1.0]);
        assert_undoes(&s, &inv, 2);

        // ITK does not guard a zero scale; the reciprocal is an infinity.
        let zero = ScaleTransform::new(vec![0.0, 1.0], vec![0.0, 0.0]);
        assert_eq!(zero.inverse().scale()[0], f64::INFINITY);
    }

    #[test]
    fn scale_logarithmic_inverse_reciprocates_each_factor() {
        let s = ScaleLogarithmicTransform::new(vec![2.0, 4.0], vec![1.0, 1.0]);
        let inv = s.inverse();
        assert_eq!(inv.scale(), [0.5, 0.25]);
        assert_undoes(&s, &inv, 2);
    }

    #[test]
    fn translation_transforms_point() {
        let t = TranslationTransform::new(vec![2.0, -3.0]);
        assert_eq!(t.transform_point(&[10.0, 10.0]), vec![12.0, 7.0]);
    }

    #[test]
    fn translation_set_parameters_rejects_wrong_length() {
        let mut t = TranslationTransform::new(vec![0.0, 0.0]);
        assert!(matches!(
            t.set_parameters(&[1.0]),
            Err(TransformError::InvalidParameters {
                got: 1,
                expected: 2
            })
        ));
    }

    #[test]
    fn translation_has_no_fixed_parameters() {
        let mut t = TranslationTransform::new(vec![2.0, -3.0]);
        assert_eq!(t.fixed_parameters(), Vec::<f64>::new());
        assert_eq!(t.number_of_fixed_parameters(), 0);
        assert!(t.set_fixed_parameters(&[]).is_ok());
        assert!(matches!(
            t.set_fixed_parameters(&[1.0]),
            Err(TransformError::InvalidFixedParameters { got: 1, .. })
        ));
    }

    #[test]
    fn matrix_offset_fixed_parameters_are_the_center() {
        let mut a = AffineTransform::identity(3);
        assert_eq!(a.fixed_parameters(), vec![0.0; 3]);
        a.set_fixed_parameters(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(a.center(), [1.0, 2.0, 3.0]);
        assert_eq!(a.fixed_parameters(), vec![1.0, 2.0, 3.0]);

        let mut s = Similarity2DTransform::identity();
        s.set_fixed_parameters(&[4.0, 5.0]).unwrap();
        assert_eq!(s.center(), [4.0, 5.0]);

        // A wrong-length array is where ITK's SetFixedParameters throws.
        assert!(a.set_fixed_parameters(&[1.0, 2.0]).is_err());
        assert!(a.set_fixed_parameters(&[1.0, 2.0, 3.0, 4.0]).is_err());
    }

    #[test]
    fn setting_the_center_via_fixed_parameters_refreshes_the_offset() {
        // A 90° rotation whose center moves to (1, 0): that point must stay put.
        let mut e = Euler2DTransform::new(std::f64::consts::FRAC_PI_2, [0.0, 0.0], [0.0, 0.0]);
        e.set_fixed_parameters(&[1.0, 0.0]).unwrap();
        let mapped = e.transform_point(&[1.0, 0.0]);
        assert!((mapped[0] - 1.0).abs() < 1e-12, "{mapped:?}");
        assert!((mapped[1] - 0.0).abs() < 1e-12, "{mapped:?}");
    }

    #[test]
    fn euler3d_fixed_parameters_carry_compute_zyx() {
        let mut e = Euler3DTransform::new(0.1, 0.2, 0.3, [0.0; 3], [0.0; 3]);
        assert_eq!(e.fixed_parameters(), vec![0.0, 0.0, 0.0, 0.0]);
        e.set_compute_zyx(true);
        assert_eq!(e.fixed_parameters(), vec![0.0, 0.0, 0.0, 1.0]);

        // A 4-entry array restores both the center and the flag.
        let mut f = Euler3DTransform::identity();
        f.set_fixed_parameters(&[1.0, 2.0, 3.0, 1.0]).unwrap();
        assert_eq!(f.center(), [1.0, 2.0, 3.0]);
        assert!(f.compute_zyx());

        // A 3-entry array is accepted (ITK's backwards-compatibility branch) and
        // leaves the flag alone.
        f.set_fixed_parameters(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(f.center(), [4.0, 5.0, 6.0]);
        assert!(f.compute_zyx());

        assert!(f.set_fixed_parameters(&[1.0, 2.0]).is_err());
        assert!(f.set_fixed_parameters(&[1.0, 2.0, 3.0, 4.0, 5.0]).is_err());
    }

    #[test]
    fn scale_fixed_parameters_are_the_center() {
        let mut s = ScaleTransform::new(vec![2.0, 3.0], vec![0.0, 0.0]);
        s.set_fixed_parameters(&[1.0, 1.0]).unwrap();
        assert_eq!(s.fixed_parameters(), vec![1.0, 1.0]);
        assert_eq!(s.transform_point(&[1.0, 1.0]), vec![1.0, 1.0]);

        let mut l = ScaleLogarithmicTransform::new(vec![2.0, 3.0], vec![0.0, 0.0]);
        l.set_fixed_parameters(&[1.0, 1.0]).unwrap();
        assert_eq!(l.fixed_parameters(), vec![1.0, 1.0]);
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
        t.set_parameters(&[3.0, -4.0]).unwrap();
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
        a.set_parameters(&[1.0, 0.0, 0.0, 1.0, 5.0, -2.0]).unwrap();
        assert_eq!(a.transform_point(&[1.0, 1.0]), vec![6.0, -1.0]);
    }

    #[test]
    fn affine_set_parameters_rejects_wrong_length() {
        let mut a = AffineTransform::identity(2);
        assert!(matches!(
            a.set_parameters(&[1.0, 0.0, 0.0, 1.0, 5.0]),
            Err(TransformError::InvalidParameters {
                got: 5,
                expected: 6
            })
        ));
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
            a.set_parameters(&pp).unwrap();
            let yp = a.transform_point(&point);
            let mut pm = base.clone();
            pm[k] -= h;
            a.set_parameters(&pm).unwrap();
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
        e.set_parameters(&[0.5, 3.0, -4.0]).unwrap();
        assert_eq!(e.parameters(), vec![0.5, 3.0, -4.0]);
        assert_eq!(e.angle(), 0.5);
        assert_eq!(e.translation(), &[3.0, -4.0]);
    }

    #[test]
    fn euler2d_set_parameters_rejects_wrong_length() {
        let mut e = Euler2DTransform::identity();
        assert!(matches!(
            e.set_parameters(&[0.5, 3.0]),
            Err(TransformError::InvalidParameters {
                got: 2,
                expected: 3
            })
        ));
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
            e.set_parameters(&pp).unwrap();
            let yp = e.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            e.set_parameters(&pm).unwrap();
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
        s.set_parameters(&[1.5, 0.5, 3.0, -4.0]).unwrap();
        assert_eq!(s.parameters(), vec![1.5, 0.5, 3.0, -4.0]);
        assert_eq!(s.scale(), 1.5);
        assert_eq!(s.angle(), 0.5);
        assert_eq!(s.translation(), &[3.0, -4.0]);
    }

    #[test]
    fn similarity2d_set_parameters_rejects_wrong_length() {
        let mut s = Similarity2DTransform::identity();
        assert!(matches!(
            s.set_parameters(&[1.5, 0.5, 3.0]),
            Err(TransformError::InvalidParameters {
                got: 3,
                expected: 4
            })
        ));
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
            s.set_parameters(&pp).unwrap();
            let yp = s.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            s.set_parameters(&pm).unwrap();
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
                e.set_parameters(&pp).unwrap();
                let yp = e.transform_point(&point);
                let mut pm = base;
                pm[k] -= h;
                e.set_parameters(&pm).unwrap();
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
        e.set_parameters(&[0.1, 0.2, 0.3, 4.0, 5.0, 6.0]).unwrap();
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
            v.set_parameters(&pp).unwrap();
            let yp = v.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            v.set_parameters(&pm).unwrap();
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
        v.set_parameters(&[0.1, -0.2, 0.15, 4.0, 5.0, 6.0]).unwrap();
        let p = v.parameters();
        // Small right part is stored unchanged (no renormalization).
        assert!(
            (p[0] - 0.1).abs() < 1e-12 && (p[1] + 0.2).abs() < 1e-12 && (p[2] - 0.15).abs() < 1e-12
        );
        assert_eq!(v.translation(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn set_matrix_recovers_known_versor_for_z_rotation() {
        use std::f64::consts::FRAC_PI_4;
        // Rz(90 deg), row-major: (1,0,0) -> (0,1,0).
        let rz90 = vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut v = VersorRigid3DTransform::identity();
        v.set_matrix(&rz90).unwrap();

        let half = FRAC_PI_4; // half of 90 degrees
        assert!(v.versor_x().abs() < 1e-12);
        assert!(v.versor_y().abs() < 1e-12);
        assert!((v.versor_z() - half.sin()).abs() < 1e-12);
        assert!((v.versor_w() - half.cos()).abs() < 1e-12);

        let p = v.transform_point(&[1.0, 0.0, 7.0]);
        assert!(
            p[0].abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12 && (p[2] - 7.0).abs() < 1e-12,
            "{p:?}"
        );
    }

    #[test]
    fn set_matrix_keeps_center_fixed_and_updates_offset() {
        let center = [1.0, 2.0, 3.0];
        let mut v = VersorRigid3DTransform::new(0.0, 0.0, 0.0, [5.0, -1.0, 0.0], center);
        let rz90 = vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        v.set_matrix(&rz90).unwrap();

        assert_eq!(v.center(), &center);
        assert_eq!(v.translation(), &[5.0, -1.0, 0.0]);
        // y(center) = R*(center-center) + center + translation = center + translation.
        let y = v.transform_point(&center);
        assert!(
            (y[0] - 6.0).abs() < 1e-12 && (y[1] - 1.0).abs() < 1e-12 && (y[2] - 3.0).abs() < 1e-12,
            "{y:?}"
        );
    }

    #[test]
    fn set_matrix_rejects_non_orthonormal_matrix() {
        let scaled = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let mut v = VersorRigid3DTransform::identity();
        let err = v.set_matrix(&scaled).unwrap_err();
        assert!(matches!(err, TransformError::NotARotationMatrix));
    }

    #[test]
    fn set_matrix_rejects_reflection() {
        // det = -1: a valid orthonormal matrix, but a reflection, not a rotation.
        let reflect_z = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let mut v = VersorRigid3DTransform::identity();
        let err = v.set_matrix(&reflect_z).unwrap_err();
        assert!(matches!(err, TransformError::NotARotationMatrix));
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
        t.set_parameters(&[0.1, -0.2, 0.15, 4.0, 5.0, 6.0, 1.3])
            .unwrap();
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
            t.set_parameters(&pp).unwrap();
            let yp = t.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            t.set_parameters(&pm).unwrap();
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
        t.set_parameters(&[0.1, -0.2, 0.15, 4.0, 5.0, 6.0, 1.2, 0.8, 1.5])
            .unwrap();
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
            t.set_parameters(&pp).unwrap();
            let yp = t.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            t.set_parameters(&pm).unwrap();
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
        t.set_parameters(&params).unwrap();
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
            t.set_parameters(&pp).unwrap();
            let yp = t.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            t.set_parameters(&pm).unwrap();
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
    fn compose_scale_skew_versor3d_identity_is_noop() {
        let t = ComposeScaleSkewVersor3DTransform::identity();
        assert_eq!(t.number_of_parameters(), 12);
        assert_eq!(t.scale(), &[1.0, 1.0, 1.0]);
        assert_eq!(t.skew(), &[0.0, 0.0, 0.0]);
        assert_eq!(t.versor_w(), 1.0);
        assert_eq!(t.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn compose_scale_skew_versor3d_matrix_is_multiplicative() {
        // No rotation (R = I): M = diag(scale)·K with K = [[1,k0,k1],[0,1,k2],[0,0,1]].
        // diag(2,3,4)·K = [[2, 2·0.5, 2·0.25],[0, 3, 3·0.1],[0, 0, 4]].
        let t = ComposeScaleSkewVersor3DTransform::new(
            [2.0, 3.0, 4.0],
            [0.5, 0.25, 0.1],
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            [0.0; 3],
        );
        #[rustfmt::skip]
        let expect = [
            2.0, 1.0, 0.5,
            0.0, 3.0, 0.3,
            0.0, 0.0, 4.0,
        ];
        for (a, b) in t.matrix().iter().zip(expect) {
            assert!((a - b).abs() < 1e-12, "matrix {:?}", t.matrix());
        }
    }

    #[test]
    fn compose_scale_skew_versor3d_matrix_differs_from_additive_scale_versor() {
        use std::f64::consts::FRAC_PI_4;
        // Rz(90°), scale [2,3,4], zero skew. The multiplicative compose gives
        // M = R·diag(scale) = [0,-3,0; 2,0,0; 0,0,4], whereas the additive
        // ScaleVersor3D gives R + diag(scale-1) = [1,-1,0; 1,2,0; 0,0,4].
        let sz = FRAC_PI_4.sin();
        let compose = ComposeScaleSkewVersor3DTransform::new(
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            sz,
            [0.0; 3],
            [0.0; 3],
        );
        #[rustfmt::skip]
        let expect = [
            0.0, -3.0, 0.0,
            2.0,  0.0, 0.0,
            0.0,  0.0, 4.0,
        ];
        for (a, b) in compose.matrix().iter().zip(expect) {
            assert!(
                (a - b).abs() < 1e-12,
                "compose matrix {:?}",
                compose.matrix()
            );
        }
        // The additive sibling produces a genuinely different matrix.
        let additive =
            ScaleVersor3DTransform::new([2.0, 3.0, 4.0], 0.0, 0.0, sz, [0.0; 3], [0.0; 3]);
        let differs = compose
            .matrix()
            .iter()
            .zip(additive.matrix())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(differs, "compose and additive matrices unexpectedly equal");
    }

    #[test]
    fn compose_scale_skew_versor3d_parameters_roundtrip() {
        let mut t = ComposeScaleSkewVersor3DTransform::identity();
        let params = [
            0.1, -0.2, 0.15, 4.0, 5.0, 6.0, 1.2, 0.8, 1.5, 0.05, -0.1, 0.15,
        ];
        t.set_parameters(&params).unwrap();
        let p = t.parameters();
        assert!(
            (p[0] - 0.1).abs() < 1e-12 && (p[1] + 0.2).abs() < 1e-12 && (p[2] - 0.15).abs() < 1e-12
        );
        assert_eq!(&p[3..6], &[4.0, 5.0, 6.0]);
        assert_eq!(&p[6..9], &[1.2, 0.8, 1.5]);
        assert_eq!(&p[9..12], &[0.05, -0.1, 0.15]);
    }

    #[test]
    fn compose_scale_skew_versor3d_jacobian_matches_finite_difference_near_identity() {
        // Mirrors ITK's own Jacobian test: from the identity, bump one parameter to
        // 0.1, then finite-difference every parameter. The sympy Jacobian treats w
        // as independent, so it is only valid near identity and to a loose relative
        // tolerance (ITK uses 10%).
        let point = [10.0, 20.0, -10.0];
        let n = 12;
        let identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let h = 1e-6;
        for mc in 0..n {
            let mut params = identity;
            params[mc] = 0.1;
            let mut t = ComposeScaleSkewVersor3DTransform::identity();
            t.set_parameters(&params).unwrap();
            let jac = t.jacobian_wrt_parameters(&point);
            for i in 0..n {
                let mut p1 = params;
                p1[i] += h;
                t.set_parameters(&p1).unwrap();
                let y1 = t.transform_point(&point);
                let mut p2 = params;
                p2[i] -= h;
                t.set_parameters(&p2).unwrap();
                let y2 = t.transform_point(&point);
                for d in 0..3 {
                    let fd = (y1[d] - y2[d]) / (2.0 * h);
                    let analytic = jac[d * n + i];
                    assert!(
                        (fd - analytic).abs() <= 0.1 * fd.abs() + 1e-6,
                        "mc {mc} param {i} dim {d}: fd {fd} vs analytic {analytic}"
                    );
                }
            }
        }
    }

    #[test]
    fn versor_identity_is_noop() {
        let v = VersorTransform::identity();
        assert_eq!(v.number_of_parameters(), 3);
        assert_eq!(v.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn versor_90_degree_rotation_about_center() {
        use std::f64::consts::FRAC_PI_4;
        // Right part (0,0,sin(45°)) ⇒ Rz(90°) about a non-zero center.
        let c = [1.0, 2.0, 3.0];
        let v = VersorTransform::new(0.0, 0.0, FRAC_PI_4.sin(), c);
        // The center maps to itself.
        let yc = v.transform_point(&c);
        for d in 0..3 {
            assert!((yc[d] - c[d]).abs() < 1e-12, "center moved: {yc:?}");
        }
        // (c + (1,0,0)) rotates to c + (0,1,0) about c.
        let p = [c[0] + 1.0, c[1], c[2]];
        let y = v.transform_point(&p);
        assert!(
            (y[0] - c[0]).abs() < 1e-12
                && (y[1] - (c[1] + 1.0)).abs() < 1e-12
                && (y[2] - c[2]).abs() < 1e-12,
            "{y:?}"
        );
    }

    #[test]
    fn versor_jacobian_is_finite_difference_consistent() {
        // Small right part keeps ‖v‖ well below 1 (no renormalization), so the
        // finite difference exercises the analytic w = √(1−‖v‖²) dependence.
        let base = [0.12, -0.08, 0.1];
        let center = [2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let mut v = VersorTransform::new(base[0], base[1], base[2], center);
        let jac = v.jacobian_wrt_parameters(&point);
        let n = v.number_of_parameters();
        let h = 1e-7;
        for k in 0..n {
            let mut pp = base;
            pp[k] += h;
            v.set_parameters(&pp).unwrap();
            let yp = v.transform_point(&point);
            let mut pm = base;
            pm[k] -= h;
            v.set_parameters(&pm).unwrap();
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
    fn versor_set_parameters_rejects_wrong_length() {
        let mut v = VersorTransform::identity();
        assert!(matches!(
            v.set_parameters(&[0.1, -0.2]),
            Err(TransformError::InvalidParameters {
                got: 2,
                expected: 3
            })
        ));
    }

    #[test]
    fn scale_identity_is_noop() {
        let s = ScaleTransform::identity(3);
        assert_eq!(s.number_of_parameters(), 3);
        assert_eq!(s.transform_point(&[3.0, -4.0, 5.0]), vec![3.0, -4.0, 5.0]);
    }

    #[test]
    fn scale_about_non_zero_center() {
        let center = vec![1.0, -2.0, 3.0];
        let scale = vec![2.0, 0.5, -3.0];
        let s = ScaleTransform::new(scale.clone(), center.clone());
        let p = [4.0, 1.0, -1.0];
        let y = s.transform_point(&p);
        for d in 0..3 {
            let expect = center[d] + scale[d] * (p[d] - center[d]);
            assert!((y[d] - expect).abs() < 1e-12, "dim {d}: {y:?}");
        }
        // The center maps to itself.
        let yc = s.transform_point(&center);
        for d in 0..3 {
            assert!((yc[d] - center[d]).abs() < 1e-12, "center moved: {yc:?}");
        }
    }

    #[test]
    fn scale_jacobian_is_finite_difference_consistent() {
        let base = vec![1.7, -0.6, 2.3];
        let center = vec![2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let mut s = ScaleTransform::new(base.clone(), center);
        let jac = s.jacobian_wrt_parameters(&point);
        let n = s.number_of_parameters();
        let h = 1e-6;
        for k in 0..n {
            let mut pp = base.clone();
            pp[k] += h;
            s.set_parameters(&pp).unwrap();
            let yp = s.transform_point(&point);
            let mut pm = base.clone();
            pm[k] -= h;
            s.set_parameters(&pm).unwrap();
            let ym = s.transform_point(&point);
            for i in 0..n {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-6,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }

    #[test]
    fn scale_set_parameters_rejects_wrong_length() {
        let mut s = ScaleTransform::identity(3);
        assert!(matches!(
            s.set_parameters(&[2.0, 0.5]),
            Err(TransformError::InvalidParameters {
                got: 2,
                expected: 3
            })
        ));
    }

    /// [`ScaleLogarithmicTransform::set_parameters`] maps the given vector
    /// through `exp` and forwards it unchanged in length, so a wrong-length
    /// vector surfaces the same error the inner [`ScaleTransform`] would raise
    /// directly — the delegate propagates rather than re-checking.
    #[test]
    fn scale_logarithmic_set_parameters_propagates_the_inner_error() {
        let mut t = ScaleLogarithmicTransform::identity(3);
        assert!(matches!(
            t.set_parameters(&[0.3, -0.5]),
            Err(TransformError::InvalidParameters {
                got: 2,
                expected: 3
            })
        ));
    }

    #[test]
    fn scale_logarithmic_equals_scale_at_exp_params() {
        let center = vec![1.0, -2.0, 3.0];
        let log_scale: Vec<f64> = vec![0.3, -0.5, 0.1];
        let scale: Vec<f64> = log_scale.iter().map(|v| v.exp()).collect();
        let mut log_t = ScaleLogarithmicTransform::identity(3);
        log_t.set_center(&center);
        log_t.set_parameters(&log_scale).unwrap();
        let lin_t = ScaleTransform::new(scale, center);
        let p = [10.0, -3.0, 7.0];
        let y_log = log_t.transform_point(&p);
        let y_lin = lin_t.transform_point(&p);
        for d in 0..3 {
            assert!(
                (y_log[d] - y_lin[d]).abs() < 1e-12,
                "dim {d}: log {y_log:?} vs lin {y_lin:?}"
            );
        }
        // parameters() round-trips through log/exp.
        let p_out = log_t.parameters();
        for d in 0..3 {
            assert!((p_out[d] - log_scale[d]).abs() < 1e-12);
        }
    }

    #[test]
    fn scale_logarithmic_jacobian_is_finite_difference_consistent() {
        // Off-center point: exercises the − center[d] chain-rule term this port
        // corrects relative to ITK's literal `scale[d] * p[d]`.
        let base: Vec<f64> = vec![0.3, -0.5, 0.9];
        let center = vec![2.0, -1.0, 4.0];
        let point = [4.0, 5.0, -3.0];
        let scale: Vec<f64> = base.iter().map(|v| v.exp()).collect();
        let mut t = ScaleLogarithmicTransform::new(scale, center);
        t.set_parameters(&base).unwrap();
        let jac = t.jacobian_wrt_parameters(&point);
        let n = t.number_of_parameters();
        let h = 1e-6;
        for k in 0..n {
            let mut pp = base.clone();
            pp[k] += h;
            t.set_parameters(&pp).unwrap();
            let yp = t.transform_point(&point);
            let mut pm = base.clone();
            pm[k] -= h;
            t.set_parameters(&pm).unwrap();
            let ym = t.transform_point(&point);
            for i in 0..n {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * n + k]).abs() < 1e-5,
                    "param {k} dim {i}: fd {fd} vs analytic {}",
                    jac[i * n + k]
                );
            }
        }
    }

    /// Central finite difference of `transform_point` — the same formula the
    /// `TransformBase::jacobian_wrt_position` default uses, recomputed here so the
    /// analytic overrides are checked against the thing they replace rather
    /// than against themselves.
    fn fd_position_jacobian(t: &dyn TransformBase, point: &[f64]) -> Vec<f64> {
        let dim = t.dimension();
        let mut jac = vec![0.0; dim * dim];
        for (c, &pc) in point.iter().enumerate().take(dim) {
            let h = 1e-6 * pc.abs().max(1.0);
            let mut plus = point.to_vec();
            let mut minus = point.to_vec();
            plus[c] += h;
            minus[c] -= h;
            let f_plus = t.transform_point(&plus);
            let f_minus = t.transform_point(&minus);
            for r in 0..dim {
                jac[r * dim + c] = (f_plus[r] - f_minus[r]) / (2.0 * h);
            }
        }
        jac
    }

    fn assert_position_jacobian_matches_fd(t: &dyn TransformBase, point: &[f64], label: &str) {
        let analytic = t.jacobian_wrt_position(point);
        let fd = fd_position_jacobian(t, point);
        assert_eq!(analytic.len(), fd.len(), "{label}: jacobian length");
        for (k, (&a, &f)) in analytic.iter().zip(fd.iter()).enumerate() {
            assert!(
                (a - f).abs() < 1e-5,
                "{label}: entry {k}: analytic {a} vs finite difference {f}"
            );
        }
    }

    /// Every transform that overrides `jacobian_wrt_position` analytically must
    /// agree with the finite-difference default it replaces. Evaluated at an
    /// off-center, off-lattice point so a dropped center term or a transposed
    /// matrix cannot pass.
    #[test]
    fn analytic_position_jacobians_match_finite_difference() {
        let p2 = [1.7f64, -0.6];
        let p3 = [1.7f64, -0.6, 2.3];
        let c2 = [0.4f64, 0.9];
        let c3 = [0.4f64, 0.9, -1.1];

        assert_position_jacobian_matches_fd(
            &TranslationTransform::new(vec![0.3, -0.8]),
            &p2,
            "translation",
        );
        assert_position_jacobian_matches_fd(
            &AffineTransform::new(2, vec![1.1, 0.2, -0.3, 0.9], vec![0.5, -0.4], c2.to_vec()),
            &p2,
            "affine",
        );
        assert_position_jacobian_matches_fd(
            &Euler2DTransform::new(0.37, [0.5, -0.4], c2),
            &p2,
            "euler2d",
        );
        assert_position_jacobian_matches_fd(
            &Similarity2DTransform::new(1.3, 0.37, [0.5, -0.4], c2),
            &p2,
            "similarity2d",
        );
        assert_position_jacobian_matches_fd(
            &Euler3DTransform::new(0.2, -0.35, 0.44, [0.5, -0.4, 0.7], c3),
            &p3,
            "euler3d",
        );
        assert_position_jacobian_matches_fd(
            &VersorRigid3DTransform::new(0.1, -0.2, 0.15, [0.5, -0.4, 0.7], c3),
            &p3,
            "versorrigid3d",
        );
        assert_position_jacobian_matches_fd(
            &Similarity3DTransform::new(1.2, 0.1, -0.2, 0.15, [0.5, -0.4, 0.7], c3),
            &p3,
            "similarity3d",
        );
        assert_position_jacobian_matches_fd(
            &VersorTransform::new(0.1, -0.2, 0.15, c3),
            &p3,
            "versor",
        );
        // The three additive-matrix versors (M = R + diag(scale − 1) [+ skew])
        // and the multiplicative compose variant: `matrix()` is still exactly
        // dT/dx because transform_point is M·x + offset for all of them.
        assert_position_jacobian_matches_fd(
            &ScaleVersor3DTransform::new([1.2, 0.8, 1.05], 0.1, -0.2, 0.15, [0.5, -0.4, 0.7], c3),
            &p3,
            "scaleversor3d",
        );
        assert_position_jacobian_matches_fd(
            &ScaleSkewVersor3DTransform::new(
                [1.2, 0.8, 1.05],
                [0.03, -0.02, 0.04, 0.01, -0.05, 0.02],
                0.1,
                -0.2,
                0.15,
                [0.5, -0.4, 0.7],
                c3,
            ),
            &p3,
            "scaleskewversor3d",
        );
        assert_position_jacobian_matches_fd(
            &ComposeScaleSkewVersor3DTransform::new(
                [1.2, 0.8, 1.05],
                [0.03, -0.02, 0.04],
                0.1,
                -0.2,
                0.15,
                [0.5, -0.4, 0.7],
                c3,
            ),
            &p3,
            "composescaleskewversor3d",
        );
        assert_position_jacobian_matches_fd(
            &ScaleTransform::new(vec![1.3, 0.7, 2.1], c3.to_vec()),
            &p3,
            "scale",
        );
        assert_position_jacobian_matches_fd(
            &ScaleLogarithmicTransform::new(vec![1.3, 0.7, 2.1], c3.to_vec()),
            &p3,
            "scale_logarithmic",
        );
    }
}
