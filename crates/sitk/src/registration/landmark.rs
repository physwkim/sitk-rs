//! Initialize a rigid, similarity, or affine transform from pairs of
//! corresponding landmarks, mirroring `itk::LandmarkBasedTransformInitializer`.
//!
//! Given equal-length fixed and moving landmark lists (point `i` in one list
//! is taken to correspond to point `i` in the other), each `initialize_*`
//! method computes the transform that maps the fixed landmarks onto the
//! moving landmarks in a least-squares sense:
//!
//! - [`initialize_versor_rigid_3d`](LandmarkBasedTransformInitializer::initialize_versor_rigid_3d)
//!   — a 3-D rigid rotation + translation via the closed-form quaternion
//!   method of Horn (1987), *"Closed-form solution of absolute orientation
//!   using unit quaternions"*, JOSA A 4:629-642: the landmark cross-covariance
//!   matrix is packed into a symmetric 4×4 matrix whose eigenvector for the
//!   largest eigenvalue is the optimal rotation quaternion.
//! - [`initialize_euler_2d`](LandmarkBasedTransformInitializer::initialize_euler_2d)
//!   — a 2-D rigid rotation + translation via its own closed-form angle
//!   solution (`atan2` of the cross and dot products of the centered landmark
//!   vectors).
//! - [`initialize_affine`](LandmarkBasedTransformInitializer::initialize_affine)
//!   — a general affine transform via the least-squares normal equations
//!   (`Qa = C`), following the algorithm in ITK's `AffineTransform` branch
//!   (attributed to Eun Young Kim, itself based on H. Späth's method).
//!
//! Per-landmark weights ([`with_landmark_weight`](LandmarkBasedTransformInitializer::with_landmark_weight))
//! are read only by [`initialize_affine`] — matching ITK, whose rigid/Euler
//! branches never reference the weight vector at all.
//!
//! Unlike ITK, which silently falls back to an identity rotation when a rigid
//! transform is asked to initialize from fewer landmarks than its dimension,
//! this port rejects the input with [`RegistrationError::InsufficientLandmarks`].
//!
//! [`initialize_affine`]: LandmarkBasedTransformInitializer::initialize_affine
//!
//! ```
//! use sitk::registration::LandmarkBasedTransformInitializer;
//! use sitk::transform::{Euler2DTransform, TransformBase};
//!
//! // Rotate 30 degrees about the origin, then translate by (5, 0).
//! let angle = 30.0f64.to_radians();
//! let known = Euler2DTransform::new(angle, [5.0, 0.0], [0.0, 0.0]);
//! let fixed = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 3.0]];
//! let moving: Vec<Vec<f64>> = fixed.iter().map(|p| known.transform_point(p)).collect();
//!
//! let recovered = LandmarkBasedTransformInitializer::new(fixed, moving)
//!     .initialize_euler_2d()
//!     .unwrap();
//! assert!((recovered.angle() - angle).abs() < 1e-9);
//! ```

use crate::core::matrix;
use crate::registration::eigen::jacobi_eigen_symmetric;
use crate::registration::error::{RegistrationError, Result};
use crate::transform::{AffineTransform, Euler2DTransform, VersorRigid3DTransform};
use std::f64::consts::PI;

/// Computes a rigid, similarity, or affine transform from corresponding
/// landmark pairs. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct LandmarkBasedTransformInitializer {
    fixed_landmarks: Vec<Vec<f64>>,
    moving_landmarks: Vec<Vec<f64>>,
    landmark_weight: Option<Vec<f64>>,
}

impl LandmarkBasedTransformInitializer {
    /// A new initializer over `fixed_landmarks` and `moving_landmarks`, taken
    /// pairwise (point `i` in `fixed_landmarks` corresponds to point `i` in
    /// `moving_landmarks`).
    pub fn new(fixed_landmarks: Vec<Vec<f64>>, moving_landmarks: Vec<Vec<f64>>) -> Self {
        Self {
            fixed_landmarks,
            moving_landmarks,
            landmark_weight: None,
        }
    }

    /// Set a per-landmark weight, one entry per landmark pair. Read only by
    /// [`initialize_affine`](Self::initialize_affine); ITK's rigid/Euler
    /// branches never read landmark weights, so neither do these.
    pub fn with_landmark_weight(mut self, landmark_weight: Vec<f64>) -> Self {
        self.landmark_weight = Some(landmark_weight);
        self
    }

    /// Fixed and moving landmark counts must match.
    fn validate_counts(&self) -> Result<()> {
        if self.fixed_landmarks.len() != self.moving_landmarks.len() {
            return Err(RegistrationError::LandmarkCountMismatch {
                fixed: self.fixed_landmarks.len(),
                moving: self.moving_landmarks.len(),
            });
        }
        Ok(())
    }

    /// Compute the 3-D rigid rotation and translation that best maps the
    /// fixed landmarks onto the moving landmarks, mirroring ITK's
    /// `InternalInitializeTransform(VersorRigid3DTransformType*)`.
    ///
    /// Requires at least 3 landmark pairs (matches ITK's own
    /// `fixedLandmarks.size() >= ImageDimension` check that gates whether a
    /// rotation can be computed at all); with fewer, ITK falls back silently
    /// to an identity rotation, but this port returns
    /// [`RegistrationError::InsufficientLandmarks`] instead.
    pub fn initialize_versor_rigid_3d(&self) -> Result<VersorRigid3DTransform> {
        self.validate_counts()?;
        let n = self.fixed_landmarks.len();
        if n < 3 {
            return Err(RegistrationError::InsufficientLandmarks {
                got: n,
                required: 3,
            });
        }
        for p in self
            .fixed_landmarks
            .iter()
            .chain(self.moving_landmarks.iter())
        {
            debug_assert_eq!(p.len(), 3, "VersorRigid3D landmarks must be 3-D points");
        }

        let fixed_centroid = centroid(&self.fixed_landmarks, 3);
        let moving_centroid = centroid(&self.moving_landmarks, 3);

        // Cross-covariance M[i][j] = sum_k (fixed_k[i] - fixedCentroid[i]) *
        // (moving_k[j] - movingCentroid[j]); generally not symmetric.
        let mut m = [[0.0f64; 3]; 3];
        for (f, mv) in self.fixed_landmarks.iter().zip(&self.moving_landmarks) {
            let fc = [
                f[0] - fixed_centroid[0],
                f[1] - fixed_centroid[1],
                f[2] - fixed_centroid[2],
            ];
            let mc = [
                mv[0] - moving_centroid[0],
                mv[1] - moving_centroid[1],
                mv[2] - moving_centroid[2],
            ];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += fc[i] * mc[j];
                }
            }
        }

        // Pack M into the symmetric 4x4 quaternion profile matrix N, exactly
        // following ITK's `CreateMatrix`.
        let mut n4 = [[0.0f64; 4]; 4];
        n4[0][0] = m[0][0] + m[1][1] + m[2][2];
        n4[1][1] = m[0][0] - m[1][1] - m[2][2];
        n4[2][2] = -m[0][0] + m[1][1] - m[2][2];
        n4[3][3] = -m[0][0] - m[1][1] + m[2][2];
        n4[0][1] = m[1][2] - m[2][1];
        n4[1][0] = n4[0][1];
        n4[0][2] = m[2][0] - m[0][2];
        n4[2][0] = n4[0][2];
        n4[0][3] = m[0][1] - m[1][0];
        n4[3][0] = n4[0][3];
        n4[1][2] = m[0][1] + m[1][0];
        n4[2][1] = n4[1][2];
        n4[1][3] = m[2][0] + m[0][2];
        n4[3][1] = n4[1][3];
        n4[2][3] = m[1][2] + m[2][1];
        n4[3][2] = n4[2][3];

        let flat: Vec<f64> = n4.iter().flatten().copied().collect();
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&flat, 4);

        // The eigenvector for the largest eigenvalue is the optimal rotation
        // quaternion [w, x, y, z] (Horn 1987); `eigenvectors` stores it as
        // column `max_idx`.
        let (max_idx, _) = eigenvalues
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .expect("eigenvalues is non-empty");
        let mut q = [
            eigenvectors[max_idx],
            eigenvectors[4 + max_idx],
            eigenvectors[8 + max_idx],
            eigenvectors[12 + max_idx],
        ];
        // An eigenvector's sign is arbitrary; normalize so the scalar part
        // (q[0] = w) is non-negative, since `VersorRigid3DTransform` always
        // reconstructs w = sqrt(1 - vx^2 - vy^2 - vz^2) >= 0, and only a
        // consistent overall sign keeps (vx, vy, vz) paired with the correct
        // rotation matrix.
        if q[0] < 0.0 {
            for c in &mut q {
                *c = -*c;
            }
        }

        let translation = [
            moving_centroid[0] - fixed_centroid[0],
            moving_centroid[1] - fixed_centroid[1],
            moving_centroid[2] - fixed_centroid[2],
        ];
        let center = [fixed_centroid[0], fixed_centroid[1], fixed_centroid[2]];
        Ok(VersorRigid3DTransform::new(
            q[1],
            q[2],
            q[3],
            translation,
            center,
        ))
    }

    /// Compute the 2-D rigid rotation and translation that best maps the
    /// fixed landmarks onto the moving landmarks, mirroring ITK's
    /// `InternalInitializeTransform(Rigid2DTransformType*)`.
    ///
    /// Requires at least 2 landmark pairs (matches ITK's own
    /// `fixedLandmarks.size() >= 2` check that gates whether a rotation can
    /// be computed at all); with fewer, ITK falls back silently to an
    /// identity rotation, but this port returns
    /// [`RegistrationError::InsufficientLandmarks`] instead.
    ///
    /// Ported bug-for-bug from ITK: when the summed dot product `s_dot` of
    /// centered fixed/moving vectors falls below a hardcoded `0.00005`, ITK
    /// gives up on `atan2` and hardcodes the angle to `-π/2` — but
    /// `s_dot = cos(θ)·Σ|centered|²`, so this branch also fires for any
    /// *genuine* rotation near ±90°, not just truly degenerate landmark data,
    /// silently returning the wrong angle in that case.
    pub fn initialize_euler_2d(&self) -> Result<Euler2DTransform> {
        self.validate_counts()?;
        let n = self.fixed_landmarks.len();
        if n < 2 {
            return Err(RegistrationError::InsufficientLandmarks {
                got: n,
                required: 2,
            });
        }
        for p in self
            .fixed_landmarks
            .iter()
            .chain(self.moving_landmarks.iter())
        {
            debug_assert_eq!(p.len(), 2, "Euler2D landmarks must be 2-D points");
        }

        let fixed_centroid = centroid(&self.fixed_landmarks, 2);
        let moving_centroid = centroid(&self.moving_landmarks, 2);

        // The rotation angle is given by the cross and dot products of the
        // centered fixed/moving landmark vectors (least-squares optimal
        // angle for a pure rotation).
        let mut s_dot = 0.0f64;
        let mut s_cross = 0.0f64;
        for (f, mv) in self.fixed_landmarks.iter().zip(&self.moving_landmarks) {
            let fc = [f[0] - fixed_centroid[0], f[1] - fixed_centroid[1]];
            let mc = [mv[0] - moving_centroid[0], mv[1] - moving_centroid[1]];
            s_dot += mc[0] * fc[0] + mc[1] * fc[1];
            s_cross += mc[1] * fc[0] - mc[0] * fc[1];
        }

        let rotation_angle = if s_dot.abs() > 0.00005 {
            s_cross.atan2(s_dot)
        } else {
            -0.5 * PI
        };

        let translation = [
            moving_centroid[0] - fixed_centroid[0],
            moving_centroid[1] - fixed_centroid[1],
        ];
        let center = [fixed_centroid[0], fixed_centroid[1]];
        Ok(Euler2DTransform::new(rotation_angle, translation, center))
    }

    /// Compute the affine transform that best maps the fixed landmarks onto
    /// the moving landmarks in a weighted least-squares sense, mirroring
    /// ITK's `InternalInitializeTransform(AffineTransformType*)`.
    ///
    /// The landmark point dimension is taken from the first fixed landmark.
    /// Requires at least `dimension + 1` landmark pairs (an affine transform
    /// has `dimension^2 + dimension` degrees of freedom, and each landmark
    /// pair contributes `dimension` equations) — matches ITK's own check,
    /// which throws under this same condition.
    ///
    /// ITK solves the normal equations `Q*a = C` via QR decomposition; this
    /// port instead inverts `Q` directly
    /// ([`crate::core::matrix::invert`]), which gives the same unique solution
    /// whenever `Q` is non-singular but is less numerically robust very close
    /// to singular. A `Q` that direct inversion finds singular — e.g. from
    /// collinear or coplanar landmarks — is reported as
    /// [`RegistrationError::DegenerateLandmarks`].
    pub fn initialize_affine(&self) -> Result<AffineTransform> {
        self.validate_counts()?;
        if self.fixed_landmarks.is_empty() {
            return Err(RegistrationError::InsufficientLandmarks {
                got: 0,
                required: 1,
            });
        }
        let dim = self.fixed_landmarks[0].len();
        let n = self.fixed_landmarks.len();
        if n < dim + 1 {
            return Err(RegistrationError::InsufficientLandmarks {
                got: n,
                required: dim + 1,
            });
        }
        for p in self
            .fixed_landmarks
            .iter()
            .chain(self.moving_landmarks.iter())
        {
            debug_assert_eq!(
                p.len(),
                dim,
                "all Affine landmarks must share one dimension"
            );
        }

        let weight = match &self.landmark_weight {
            Some(w) => {
                if w.len() != n {
                    return Err(RegistrationError::LandmarkWeightLength {
                        got: w.len(),
                        expected: n,
                    });
                }
                w.clone()
            }
            None => vec![1.0; n],
        };
        // Normalize weights by their Frobenius norm (as a diagonal matrix),
        // matching ITK's `vnlWeight = vnlWeight / vnlWeight.fro_norm()`.
        let norm = weight.iter().map(|w| w * w).sum::<f64>().sqrt();
        let weight_norm: Vec<f64> = weight.iter().map(|w| w / norm).collect();

        // q_i = weight_i * [fixed_i; 1]  (length dim+1)
        // p_i = weight_i * moving_i      (length dim)
        // Q = sum_i q_i * q_i^T          ((dim+1) x (dim+1))
        // C = sum_i q_i * p_i^T          ((dim+1) x dim)
        let d1 = dim + 1;
        let mut q_mat = vec![0.0f64; d1 * d1];
        let mut c_mat = vec![0.0f64; d1 * dim];
        for ((fixed_pt, moving_pt), &wi) in self
            .fixed_landmarks
            .iter()
            .zip(self.moving_landmarks.iter())
            .zip(weight_norm.iter())
        {
            let mut qi = vec![0.0f64; d1];
            for (d, qd) in qi.iter_mut().take(dim).enumerate() {
                *qd = fixed_pt[d] * wi;
            }
            qi[dim] = wi;
            let mut pi = vec![0.0f64; dim];
            for (d, pd) in pi.iter_mut().enumerate() {
                *pd = moving_pt[d] * wi;
            }

            for r in 0..d1 {
                for c in 0..d1 {
                    q_mat[r * d1 + c] += qi[r] * qi[c];
                }
                for c in 0..dim {
                    c_mat[r * dim + c] += qi[r] * pi[c];
                }
            }
        }

        let q_inv = matrix::invert(&q_mat, d1).ok_or(RegistrationError::DegenerateLandmarks)?;

        // Solve Q*X = C, i.e. X = Q^-1 * C, shape (dim+1) x dim.
        let mut x = vec![0.0f64; d1 * dim];
        for r in 0..d1 {
            for c in 0..dim {
                let mut acc = 0.0;
                for k in 0..d1 {
                    acc += q_inv[r * d1 + k] * c_mat[k * dim + c];
                }
                x[r * dim + c] = acc;
            }
        }

        // Affine = X^T: rows 0..dim of X (transposed) give the dim x dim
        // matrix; row `dim` of X gives the translation.
        let mut m_a = vec![0.0f64; dim * dim];
        for t in 0..dim {
            for c in 0..dim {
                m_a[t * dim + c] = x[c * dim + t];
            }
        }
        let m_t: Vec<f64> = (0..dim).map(|t| x[dim * dim + t]).collect();

        Ok(AffineTransform::new(dim, m_a, m_t, vec![0.0; dim]))
    }
}

/// Mean of `points` (each of length `dim`) along each dimension.
fn centroid(points: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; dim];
    for p in points {
        for d in 0..dim {
            c[d] += p[d];
        }
    }
    let n = points.len() as f64;
    for v in &mut c {
        *v /= n;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::TransformBase;

    #[test]
    fn recovers_known_versor_rigid_3d() {
        // A 40-degree rotation about the slanted axis (1,1,0)/sqrt(2), plus
        // translation, applied to a non-planar point cloud.
        let axis = [
            1.0f64 / std::f64::consts::SQRT_2,
            1.0 / std::f64::consts::SQRT_2,
            0.0,
        ];
        let half = (40.0f64.to_radians()) / 2.0;
        let s = half.sin();
        let known = VersorRigid3DTransform::new(
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            [2.0, -3.0, 5.0],
            [0.0, 0.0, 0.0],
        );

        let fixed = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let moving: Vec<Vec<f64>> = fixed.iter().map(|p| known.transform_point(p)).collect();

        let recovered = LandmarkBasedTransformInitializer::new(fixed, moving)
            .initialize_versor_rigid_3d()
            .unwrap();

        // Rotation is intrinsic to the map (center-independent), so the
        // versor components must match directly.
        assert!((recovered.versor_x() - known.versor_x()).abs() < 1e-9);
        assert!((recovered.versor_y() - known.versor_y()).abs() < 1e-9);
        assert!((recovered.versor_z() - known.versor_z()).abs() < 1e-9);

        // Full map equivalence on a point outside the landmark set.
        let probe = [3.0, -1.0, 2.0];
        let expected = known.transform_point(&probe);
        let actual = recovered.transform_point(&probe);
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(
                (e - a).abs() < 1e-9,
                "expected {expected:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn recovers_known_euler_2d() {
        let known = Euler2DTransform::new(25.0f64.to_radians(), [4.0, -2.0], [0.0, 0.0]);
        let fixed = vec![
            vec![0.0, 0.0],
            vec![3.0, 0.0],
            vec![0.0, 2.0],
            vec![-1.0, 1.5],
        ];
        let moving: Vec<Vec<f64>> = fixed.iter().map(|p| known.transform_point(p)).collect();

        let recovered = LandmarkBasedTransformInitializer::new(fixed, moving)
            .initialize_euler_2d()
            .unwrap();

        assert!((recovered.angle() - known.angle()).abs() < 1e-9);

        let probe = [5.0, -5.0];
        let expected = known.transform_point(&probe);
        let actual = recovered.transform_point(&probe);
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(
                (e - a).abs() < 1e-9,
                "expected {expected:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn recovers_known_affine_shear_scale_translation() {
        // A 2-D affine with anisotropic scale and shear, plus translation.
        let known =
            AffineTransform::new(2, vec![1.2, 0.3, 0.1, 0.9], vec![3.0, -2.0], vec![0.0, 0.0]);
        let fixed = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 3.0],
        ];
        let moving: Vec<Vec<f64>> = fixed.iter().map(|p| known.transform_point(p)).collect();

        let recovered = LandmarkBasedTransformInitializer::new(fixed, moving)
            .initialize_affine()
            .unwrap();

        for (e, a) in known.matrix().iter().zip(recovered.matrix().iter()) {
            assert!(
                (e - a).abs() < 1e-9,
                "{:?} vs {:?}",
                known.matrix(),
                recovered.matrix()
            );
        }
        for (e, a) in known.offset().iter().zip(recovered.offset().iter()) {
            assert!(
                (e - a).abs() < 1e-9,
                "{:?} vs {:?}",
                known.offset(),
                recovered.offset()
            );
        }
    }

    #[test]
    fn weighted_affine_suppresses_outlier_landmark() {
        // Three landmarks exactly determine a 2-D affine (6 unknowns, 6
        // equations); a 4th "outlier" landmark is inconsistent with that
        // transform but given a tiny weight, so it should barely perturb the
        // recovered transform.
        let known =
            AffineTransform::new(2, vec![1.1, 0.0, 0.0, 0.9], vec![1.0, 2.0], vec![0.0, 0.0]);
        let good_fixed = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut fixed = good_fixed.clone();
        let mut moving: Vec<Vec<f64>> = good_fixed
            .iter()
            .map(|p| known.transform_point(p))
            .collect();

        // Outlier: fixed point (5,5) mapped to a wildly inconsistent moving
        // point (100, -100).
        fixed.push(vec![5.0, 5.0]);
        moving.push(vec![100.0, -100.0]);

        let recovered = LandmarkBasedTransformInitializer::new(fixed, moving)
            .with_landmark_weight(vec![1.0, 1.0, 1.0, 1e-8])
            .initialize_affine()
            .unwrap();

        for (e, a) in known.matrix().iter().zip(recovered.matrix().iter()) {
            assert!(
                (e - a).abs() < 1e-4,
                "{:?} vs {:?}",
                known.matrix(),
                recovered.matrix()
            );
        }
        for (e, a) in known.offset().iter().zip(recovered.offset().iter()) {
            assert!(
                (e - a).abs() < 1e-4,
                "{:?} vs {:?}",
                known.offset(),
                recovered.offset()
            );
        }
    }

    #[test]
    fn landmark_count_mismatch_is_rejected() {
        let fixed = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let moving = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let err = LandmarkBasedTransformInitializer::new(fixed, moving)
            .initialize_versor_rigid_3d()
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::LandmarkCountMismatch {
                fixed: 3,
                moving: 2
            }
        ));
    }

    #[test]
    fn too_few_landmarks_for_rigid_3d_is_rejected() {
        let fixed = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let moving = fixed.clone();
        let err = LandmarkBasedTransformInitializer::new(fixed, moving)
            .initialize_versor_rigid_3d()
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::InsufficientLandmarks {
                got: 2,
                required: 3
            }
        ));
    }

    #[test]
    fn affine_weight_length_mismatch_is_rejected() {
        let fixed = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let moving = fixed.clone();
        let err = LandmarkBasedTransformInitializer::new(fixed, moving)
            .with_landmark_weight(vec![1.0, 1.0])
            .initialize_affine()
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::LandmarkWeightLength {
                got: 2,
                expected: 3
            }
        ));
    }

    #[test]
    fn affine_collinear_landmarks_is_rejected() {
        // All landmarks lie on the x-axis: the normal-equations matrix is
        // singular (rank-deficient in y).
        let fixed = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
        ];
        let moving = fixed.clone();
        let err = LandmarkBasedTransformInitializer::new(fixed, moving)
            .initialize_affine()
            .unwrap_err();
        assert!(matches!(err, RegistrationError::DegenerateLandmarks));
    }
}
