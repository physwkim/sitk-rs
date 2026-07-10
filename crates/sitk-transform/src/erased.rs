//! The erased [`Transform`] value type — `itk::simple::Transform`
//! (`sitkTransform.h:86`).
//!
//! SimpleITK's `Transform` is one concrete C++ class wrapping an
//! `itk::TransformBase::Pointer` plus a runtime `TransformEnum`. It is the type
//! every by-value transform API speaks: `ReadTransform`, `WriteTransform`,
//! `CompositeTransform::AddTransform`, `ImageRegistrationMethod::
//! SetInitialTransform`, `Transform::GetInverse`.
//!
//! This port models that runtime dispatch as a `#[non_exhaustive]` enum over the
//! crate's sixteen concrete transform types (ledger §5.10, option (a)) — the same
//! shape `PixelId` uses for runtime pixel dispatch. `From` converts every
//! concrete type into it, and it implements [`TransformBase`] and
//! [`ParametricTransform`] itself, so an erased transform drops straight into
//! [`ResampleImageFilter`](crate::ResampleImageFilter),
//! [`transform_geometry`](crate::transform_geometry) and
//! [`CompositeTransform`].
//!
//! # Divergences from `itk::simple::Transform`
//!
//! - Upstream's `Transform::GetName()` returns the constant `"Transform"`; each
//!   *derived* SimpleITK class (`AffineTransform`, `Euler2DTransform`, …)
//!   overrides it with its own name. There is no derived class here, so
//!   [`Transform::class_name`] returns the wrapped concrete transform's name
//!   directly — the useful half of that pair (ledger §4.48).
//! - Upstream's `ToString()` embeds the ITK object's `PrintSelf` dump. This
//!   port's [`Display`](std::fmt::Display) prints the transform's file-format
//!   identity instead — class name, parameters, fixed parameters (ledger §4.48).
//! - Upstream copies lazily (`MakeUnique`); this enum is a plain value and
//!   copies eagerly through `Clone`.

use std::fmt;

use crate::bspline::BSplineTransform;
use crate::composite::CompositeTransform;
use crate::displacement::DisplacementFieldTransform;
use crate::error::{Result, TransformError};
use crate::transform::{
    AffineTransform, ComposeScaleSkewVersor3DTransform, Euler2DTransform, Euler3DTransform,
    ParametricTransform, ScaleLogarithmicTransform, ScaleSkewVersor3DTransform, ScaleTransform,
    ScaleVersor3DTransform, Similarity2DTransform, Similarity3DTransform, TransformBase,
    TranslationTransform, VersorRigid3DTransform, VersorTransform,
};

/// Which concrete transform an erased [`Transform`] holds — `sitkTransformEnum`
/// (`sitkTransform.h:44-65`), returned by `Transform::GetTransformEnum()`.
///
/// As upstream, one variant covers both dimensionalities of a family:
/// [`Euler`](TransformKind::Euler) is `sitkEuler` for `Euler2DTransform` *and*
/// `Euler3DTransform`, and likewise for [`Similarity`](TransformKind::Similarity).
///
/// Upstream's `sitkIdentity`, `sitkQuaternionRigid` and `sitkUnknownTransform`
/// have no counterpart here: this crate has no matching transform class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransformKind {
    /// `sitkTranslation`
    Translation,
    /// `sitkScale`
    Scale,
    /// `sitkScaleLogarithmic`
    ScaleLogarithmic,
    /// `sitkEuler` — both `Euler2DTransform` and `Euler3DTransform`.
    Euler,
    /// `sitkSimilarity` — both `Similarity2DTransform` and `Similarity3DTransform`.
    Similarity,
    /// `sitkVersor`
    Versor,
    /// `sitkVersorRigid`
    VersorRigid,
    /// `sitkScaleSkewVersor`
    ScaleSkewVersor,
    /// `sitkComposeScaleSkewVersor`
    ComposeScaleSkewVersor,
    /// `sitkScaleVersor`
    ScaleVersor,
    /// `sitkAffine`
    Affine,
    /// `sitkComposite`
    Composite,
    /// `sitkDisplacementField`
    DisplacementField,
    /// `sitkBSplineTransform`
    BSpline,
}

/// A transform of any concrete type — `itk::simple::Transform`. See the
/// [module docs](self).
///
/// Build one with `Transform::from(concrete)` / `.into()`, or read one from a
/// file with `sitk_io::read_transform`.
///
/// ```
/// use sitk_transform::{Transform, TransformBase, TranslationTransform};
///
/// let t: Transform = TranslationTransform::new(vec![1.0, 2.0]).into();
/// assert_eq!(t.dimension(), 2);
/// assert_eq!(t.transform_point(&[0.0, 0.0]), vec![1.0, 2.0]);
/// assert_eq!(t.itk_transform_type_name(), "TranslationTransform_double_2_2");
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Transform {
    /// `itk::TranslationTransform`
    Translation(TranslationTransform),
    /// `itk::ScaleTransform`
    Scale(ScaleTransform),
    /// `itk::ScaleLogarithmicTransform`
    ScaleLogarithmic(ScaleLogarithmicTransform),
    /// `itk::Euler2DTransform`
    Euler2D(Euler2DTransform),
    /// `itk::Euler3DTransform`
    Euler3D(Euler3DTransform),
    /// `itk::Similarity2DTransform`
    Similarity2D(Similarity2DTransform),
    /// `itk::Similarity3DTransform`
    Similarity3D(Similarity3DTransform),
    /// `itk::VersorTransform`
    Versor(VersorTransform),
    /// `itk::VersorRigid3DTransform`
    VersorRigid3D(VersorRigid3DTransform),
    /// `itk::ScaleSkewVersor3DTransform`
    ScaleSkewVersor3D(ScaleSkewVersor3DTransform),
    /// `itk::ComposeScaleSkewVersor3DTransform`
    ComposeScaleSkewVersor3D(ComposeScaleSkewVersor3DTransform),
    /// `itk::ScaleVersor3DTransform`
    ScaleVersor3D(ScaleVersor3DTransform),
    /// `itk::AffineTransform`
    Affine(AffineTransform),
    /// `itk::CompositeTransform`
    Composite(CompositeTransform),
    /// `itk::DisplacementFieldTransform`
    DisplacementField(DisplacementFieldTransform),
    /// `itk::BSplineTransform`
    BSpline(BSplineTransform),
}

/// Run `$body` against whichever concrete transform `$self` holds, bound to `$t`.
macro_rules! dispatch {
    ($self:expr, $t:ident => $body:expr) => {
        match $self {
            Transform::Translation($t) => $body,
            Transform::Scale($t) => $body,
            Transform::ScaleLogarithmic($t) => $body,
            Transform::Euler2D($t) => $body,
            Transform::Euler3D($t) => $body,
            Transform::Similarity2D($t) => $body,
            Transform::Similarity3D($t) => $body,
            Transform::Versor($t) => $body,
            Transform::VersorRigid3D($t) => $body,
            Transform::ScaleSkewVersor3D($t) => $body,
            Transform::ComposeScaleSkewVersor3D($t) => $body,
            Transform::ScaleVersor3D($t) => $body,
            Transform::Affine($t) => $body,
            Transform::Composite($t) => $body,
            Transform::DisplacementField($t) => $body,
            Transform::BSpline($t) => $body,
        }
    };
}

/// `impl From<Concrete> for Transform` for each variant.
macro_rules! impl_from {
    ($($variant:ident($concrete:ty)),+ $(,)?) => {
        $(
            impl From<$concrete> for Transform {
                fn from(t: $concrete) -> Self {
                    Transform::$variant(t)
                }
            }
        )+
    };
}

impl_from!(
    Translation(TranslationTransform),
    Scale(ScaleTransform),
    ScaleLogarithmic(ScaleLogarithmicTransform),
    Euler2D(Euler2DTransform),
    Euler3D(Euler3DTransform),
    Similarity2D(Similarity2DTransform),
    Similarity3D(Similarity3DTransform),
    Versor(VersorTransform),
    VersorRigid3D(VersorRigid3DTransform),
    ScaleSkewVersor3D(ScaleSkewVersor3DTransform),
    ComposeScaleSkewVersor3D(ComposeScaleSkewVersor3DTransform),
    ScaleVersor3D(ScaleVersor3DTransform),
    Affine(AffineTransform),
    Composite(CompositeTransform),
    DisplacementField(DisplacementFieldTransform),
    BSpline(BSplineTransform),
);

impl Transform {
    /// The runtime kind tag — `Transform::GetTransformEnum()`.
    pub fn kind(&self) -> TransformKind {
        match self {
            Transform::Translation(_) => TransformKind::Translation,
            Transform::Scale(_) => TransformKind::Scale,
            Transform::ScaleLogarithmic(_) => TransformKind::ScaleLogarithmic,
            Transform::Euler2D(_) | Transform::Euler3D(_) => TransformKind::Euler,
            Transform::Similarity2D(_) | Transform::Similarity3D(_) => TransformKind::Similarity,
            Transform::Versor(_) => TransformKind::Versor,
            Transform::VersorRigid3D(_) => TransformKind::VersorRigid,
            Transform::ScaleSkewVersor3D(_) => TransformKind::ScaleSkewVersor,
            Transform::ComposeScaleSkewVersor3D(_) => TransformKind::ComposeScaleSkewVersor,
            Transform::ScaleVersor3D(_) => TransformKind::ScaleVersor,
            Transform::Affine(_) => TransformKind::Affine,
            Transform::Composite(_) => TransformKind::Composite,
            Transform::DisplacementField(_) => TransformKind::DisplacementField,
            Transform::BSpline(_) => TransformKind::BSpline,
        }
    }

    /// The wrapped transform's class name, without the ITK type/dimension
    /// suffix — e.g. `"Euler3DTransform"`. This is what the *derived* SimpleITK
    /// classes return from `GetName()`; the erased `itk::simple::Transform::
    /// GetName()` returns the constant `"Transform"` (ledger §4.48).
    pub fn class_name(&self) -> &'static str {
        match self {
            Transform::Translation(_) => "TranslationTransform",
            Transform::Scale(_) => "ScaleTransform",
            Transform::ScaleLogarithmic(_) => "ScaleLogarithmicTransform",
            Transform::Euler2D(_) => "Euler2DTransform",
            Transform::Euler3D(_) => "Euler3DTransform",
            Transform::Similarity2D(_) => "Similarity2DTransform",
            Transform::Similarity3D(_) => "Similarity3DTransform",
            Transform::Versor(_) => "VersorTransform",
            Transform::VersorRigid3D(_) => "VersorRigid3DTransform",
            Transform::ScaleSkewVersor3D(_) => "ScaleSkewVersor3DTransform",
            Transform::ComposeScaleSkewVersor3D(_) => "ComposeScaleSkewVersor3DTransform",
            Transform::ScaleVersor3D(_) => "ScaleVersor3DTransform",
            Transform::Affine(_) => "AffineTransform",
            Transform::Composite(_) => "CompositeTransform",
            Transform::DisplacementField(_) => "DisplacementFieldTransform",
            Transform::BSpline(_) => "BSplineTransform",
        }
    }

    /// The fully-qualified ITK type name — `itk::Transform::GetTransformTypeAsString()`
    /// (`itkTransform.hxx:36-44`): `"<ClassName>_double_<inDim>_<outDim>"`, e.g.
    /// `"AffineTransform_double_3_3"`. This is the value on a transform file's
    /// `Transform:` line, and the key the ITK transform factory instantiates from.
    ///
    /// `itk::BSplineTransform` appends `_<splineOrder>` when the order is not 3
    /// (`itkBSplineTransform.hxx:65-76`); this crate only implements the cubic
    /// order, so the suffix never appears.
    pub fn itk_transform_type_name(&self) -> String {
        let dim = self.dimension();
        format!("{}_double_{dim}_{dim}", self.class_name())
    }

    /// A new transform of the same type set to this one's inverse —
    /// `itk::simple::Transform::GetInverse()` (`sitkTransform.cxx:542-552`),
    /// which throws when the inverse does not exist.
    ///
    /// Errors with [`TransformError::NoInverse`] when the transform's linear map
    /// is singular, and for the four types that have no inverse at all:
    ///
    /// - `ScaleVersor3DTransform` and `ScaleSkewVersor3DTransform`, whose
    ///   `ComputeMatrixParameters` raises "Setting the matrix of a ... transform
    ///   is not supported at this time." upstream;
    /// - `BSplineTransform`, which defines no `GetInverse`;
    /// - `DisplacementFieldTransform`, whose `GetInverse` copies a separately-set
    ///   *inverse displacement field* that this port does not model (ledger §4.49).
    pub fn inverse(&self) -> Result<Transform> {
        Ok(match self {
            Transform::Translation(t) => t.inverse().into(),
            Transform::Scale(t) => t.inverse().into(),
            Transform::ScaleLogarithmic(t) => t.inverse().into(),
            Transform::Euler2D(t) => t.inverse()?.into(),
            Transform::Euler3D(t) => t.inverse()?.into(),
            Transform::Similarity2D(t) => t.inverse()?.into(),
            Transform::Similarity3D(t) => t.inverse()?.into(),
            Transform::Versor(t) => t.inverse()?.into(),
            Transform::VersorRigid3D(t) => t.inverse()?.into(),
            Transform::Affine(t) => t.inverse()?.into(),
            Transform::ComposeScaleSkewVersor3D(t) => t.inverse()?.into(),
            Transform::Composite(t) => t.inverse()?.into(),
            Transform::ScaleVersor3D(_) => {
                return Err(TransformError::NoInverse(
                    "itk::ScaleVersor3DTransform::ComputeMatrixParameters is unimplemented upstream",
                ));
            }
            Transform::ScaleSkewVersor3D(_) => {
                return Err(TransformError::NoInverse(
                    "itk::ScaleSkewVersor3DTransform::ComputeMatrixParameters is unimplemented upstream",
                ));
            }
            Transform::BSpline(_) => {
                return Err(TransformError::NoInverse(
                    "itk::BSplineTransform defines no inverse",
                ));
            }
            Transform::DisplacementField(_) => {
                return Err(TransformError::NoInverse(
                    "a displacement-field inverse needs an inverse field, which this port does not model",
                ));
            }
        })
    }
}

impl TransformBase for Transform {
    fn transform_point(&self, point: &[f64]) -> Vec<f64> {
        dispatch!(self, t => t.transform_point(point))
    }

    fn dimension(&self) -> usize {
        dispatch!(self, t => t.dimension())
    }

    fn is_linear(&self) -> bool {
        dispatch!(self, t => t.is_linear())
    }

    fn jacobian_wrt_position(&self, point: &[f64]) -> Vec<f64> {
        dispatch!(self, t => t.jacobian_wrt_position(point))
    }
}

impl ParametricTransform for Transform {
    fn number_of_parameters(&self) -> usize {
        dispatch!(self, t => t.number_of_parameters())
    }

    fn parameters(&self) -> Vec<f64> {
        dispatch!(self, t => t.parameters())
    }

    fn set_parameters(&mut self, params: &[f64]) {
        dispatch!(self, t => t.set_parameters(params))
    }

    fn fixed_parameters(&self) -> Vec<f64> {
        dispatch!(self, t => t.fixed_parameters())
    }

    fn number_of_fixed_parameters(&self) -> usize {
        dispatch!(self, t => t.number_of_fixed_parameters())
    }

    fn set_fixed_parameters(&mut self, params: &[f64]) -> Result<()> {
        dispatch!(self, t => t.set_fixed_parameters(params))
    }

    fn jacobian_wrt_parameters(&self, point: &[f64]) -> Vec<f64> {
        dispatch!(self, t => t.jacobian_wrt_parameters(point))
    }

    fn has_local_support(&self) -> bool {
        dispatch!(self, t => t.has_local_support())
    }

    fn number_of_local_parameters(&self) -> usize {
        dispatch!(self, t => t.number_of_local_parameters())
    }

    fn sparse_jacobian_wrt_parameters(&self, point: &[f64]) -> Option<Vec<(usize, Vec<f64>)>> {
        dispatch!(self, t => t.sparse_jacobian_wrt_parameters(point))
    }
}

/// `itk::simple::Transform::ToString()` (`sitkTransform.cxx:555-564`) prefixes
/// `"itk::simple::Transform\n"` and then dumps the ITK object's `PrintSelf`.
/// This port keeps the prefix and prints the transform's file-format identity in
/// place of `PrintSelf` — class name, parameters, fixed parameters
/// (ledger §4.48). A composite lists its sub-transforms instead of parameters,
/// exactly as the Insight legacy writer does.
impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "itk::simple::Transform")?;
        write_body(self, f, 0)
    }
}

fn write_body(t: &Transform, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
    let pad = "  ".repeat(depth + 1);
    writeln!(f, "{pad}Transform: {}", t.itk_transform_type_name())?;
    if let Transform::Composite(c) = t {
        for sub in c.transforms() {
            write_body(sub, f, depth + 1)?;
        }
        return Ok(());
    }
    writeln!(f, "{pad}Parameters: {}", join(&t.parameters()))?;
    writeln!(f, "{pad}FixedParameters: {}", join(&t.fixed_parameters()))
}

fn join(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_transform() -> Vec<Transform> {
        vec![
            TranslationTransform::new(vec![1.0, 2.0, 3.0]).into(),
            ScaleTransform::new(vec![2.0, 3.0, 4.0], vec![1.0, 1.0, 1.0]).into(),
            ScaleLogarithmicTransform::new(vec![2.0, 3.0, 4.0], vec![1.0, 1.0, 1.0]).into(),
            Euler2DTransform::new(0.3, [1.0, 2.0], [3.0, 4.0]).into(),
            Euler3DTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).into(),
            Similarity2DTransform::new(2.0, 0.3, [1.0, 2.0], [3.0, 4.0]).into(),
            Similarity3DTransform::new(2.0, 0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).into(),
            VersorTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0]).into(),
            VersorRigid3DTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).into(),
            ScaleSkewVersor3DTransform::new(
                [1.0, 2.0, 3.0],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            )
            .into(),
            ComposeScaleSkewVersor3DTransform::new(
                [1.0, 2.0, 3.0],
                [0.1, 0.2, 0.3],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            )
            .into(),
            ScaleVersor3DTransform::new(
                [1.0, 2.0, 3.0],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            )
            .into(),
            AffineTransform::new(2, vec![2.0, 1.0, 0.0, 3.0], vec![5.0, -1.0], vec![1.0, 2.0])
                .into(),
            CompositeTransform::new(3).into(),
            DisplacementFieldTransform::new(
                2,
                &[2, 3],
                &[0.0, 0.0],
                &[1.0, 1.0],
                &[1.0, 0.0, 0.0, 1.0],
            )
            .unwrap()
            .into(),
            BSplineTransform::new(2, &[0.0, 0.0], &[4.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[2, 2])
                .unwrap()
                .into(),
        ]
    }

    #[test]
    fn every_concrete_transform_has_a_variant_and_a_from_impl() {
        assert_eq!(every_transform().len(), 16);
    }

    #[test]
    fn itk_type_names_carry_the_class_and_dimensions() {
        let names: Vec<String> = every_transform()
            .iter()
            .map(|t| t.itk_transform_type_name())
            .collect();
        assert_eq!(
            names,
            vec![
                "TranslationTransform_double_3_3",
                "ScaleTransform_double_3_3",
                "ScaleLogarithmicTransform_double_3_3",
                "Euler2DTransform_double_2_2",
                "Euler3DTransform_double_3_3",
                "Similarity2DTransform_double_2_2",
                "Similarity3DTransform_double_3_3",
                "VersorTransform_double_3_3",
                "VersorRigid3DTransform_double_3_3",
                "ScaleSkewVersor3DTransform_double_3_3",
                "ComposeScaleSkewVersor3DTransform_double_3_3",
                "ScaleVersor3DTransform_double_3_3",
                "AffineTransform_double_2_2",
                "CompositeTransform_double_3_3",
                "DisplacementFieldTransform_double_2_2",
                "BSplineTransform_double_2_2",
            ]
        );
    }

    #[test]
    fn kinds_merge_the_2d_and_3d_members_of_a_family() {
        let euler2d: Transform = Euler2DTransform::identity().into();
        let euler3d: Transform = Euler3DTransform::identity().into();
        assert_eq!(euler2d.kind(), TransformKind::Euler);
        assert_eq!(euler3d.kind(), TransformKind::Euler);

        let sim2d: Transform = Similarity2DTransform::identity().into();
        let sim3d: Transform = Similarity3DTransform::identity().into();
        assert_eq!(sim2d.kind(), TransformKind::Similarity);
        assert_eq!(sim3d.kind(), TransformKind::Similarity);
    }

    #[test]
    fn the_erased_surface_delegates_to_the_concrete_transform() {
        let concrete = Euler2DTransform::new(0.0, [1.0, 2.0], [3.0, 4.0]);
        let erased: Transform = concrete.clone().into();

        assert_eq!(erased.dimension(), concrete.dimension());
        assert_eq!(erased.is_linear(), concrete.is_linear());
        assert_eq!(erased.parameters(), concrete.parameters());
        assert_eq!(erased.fixed_parameters(), concrete.fixed_parameters());
        assert_eq!(erased.number_of_parameters(), 3);
        assert_eq!(erased.number_of_fixed_parameters(), 2);
        assert_eq!(
            erased.transform_point(&[0.0, 0.0]),
            concrete.transform_point(&[0.0, 0.0])
        );
    }

    #[test]
    fn set_parameters_and_set_fixed_parameters_reach_the_concrete_transform() {
        let mut t: Transform = Euler2DTransform::identity().into();
        t.set_parameters(&[0.5, 1.0, 2.0]);
        t.set_fixed_parameters(&[3.0, 4.0]).unwrap();
        match &t {
            Transform::Euler2D(e) => {
                assert_eq!(e.angle(), 0.5);
                assert_eq!(e.translation(), [1.0, 2.0]);
                assert_eq!(e.center(), [3.0, 4.0]);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn non_linear_transforms_report_it_through_the_erased_surface() {
        let bspline: Transform =
            BSplineTransform::new(2, &[0.0, 0.0], &[4.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[2, 2])
                .unwrap()
                .into();
        assert!(!bspline.is_linear());

        let affine: Transform = AffineTransform::identity(2).into();
        assert!(affine.is_linear());
    }

    #[test]
    fn inverse_is_defined_exactly_where_upstream_defines_it() {
        for t in every_transform() {
            let expected_some = !matches!(
                t,
                Transform::ScaleVersor3D(_)
                    | Transform::ScaleSkewVersor3D(_)
                    | Transform::BSpline(_)
                    | Transform::DisplacementField(_)
            );
            assert_eq!(
                t.inverse().is_ok(),
                expected_some,
                "{} inverse()",
                t.class_name()
            );
        }
    }

    #[test]
    fn inverse_of_an_erased_transform_keeps_its_type_and_undoes_it() {
        let t: Transform =
            Similarity3DTransform::new(2.0, 0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).into();
        let inv = t.inverse().unwrap();
        assert_eq!(inv.kind(), TransformKind::Similarity);
        let p = [7.0, -8.0, 9.0];
        let back = inv.transform_point(&t.transform_point(&p));
        for d in 0..3 {
            assert!((back[d] - p[d]).abs() < 1e-9, "{back:?}");
        }
    }

    #[test]
    fn display_prints_the_file_format_identity() {
        let t: Transform = TranslationTransform::new(vec![1.0, -2.5]).into();
        assert_eq!(
            t.to_string(),
            "itk::simple::Transform\n  \
             Transform: TranslationTransform_double_2_2\n  \
             Parameters: 1 -2.5\n  \
             FixedParameters: \n"
        );
    }

    #[test]
    fn display_of_a_composite_lists_its_sub_transforms() {
        let mut c = CompositeTransform::new(2);
        c.add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        let t: Transform = c.into();
        assert_eq!(
            t.to_string(),
            "itk::simple::Transform\n  \
             Transform: CompositeTransform_double_2_2\n    \
             Transform: TranslationTransform_double_2_2\n    \
             Parameters: 1 2\n    \
             FixedParameters: \n"
        );
    }
}
