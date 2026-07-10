//! Initialize a [`VersorRigid3DTransform`]'s center, translation, and
//! (optionally) rotation from paired 3-D images' intensity moments,
//! mirroring `itk::CenteredVersorTransformInitializer`
//! (`itkCenteredVersorTransformInitializer.h/.hxx`, SimpleITK's
//! `CenteredVersorTransformInitializerFilter`,
//! `sitkCenteredVersorTransformInitializerFilter.h/.cxx`).
//!
//! [`initialize`](CenteredVersorTransformInitializer::initialize) always runs
//! in [`OperationMode::Moments`](crate::OperationMode::Moments) — it calls
//! [`CenteredTransformInitializer`] directly for the center/translation step
//! — mirroring the ITK subclass's constructor, which forces
//! `Superclass::MomentsOn()` unconditionally
//! (`itkCenteredVersorTransformInitializer.hxx:24-30`).
//!
//! # ComputeRotation
//!
//! When `compute_rotation` is set (constructor argument here;
//! `SetComputeRotation`/`ComputeRotationOn`/`ComputeRotationOff` in ITK,
//! default **`false`** in both the ITK header
//! (`itkCenteredVersorTransformInitializer.h:102`) and the SimpleITK filter
//! (`sitkCenteredVersorTransformInitializerFilter.h:129`)),
//! `InitializeTransform` additionally computes each image's **principal
//! axes** — the eigenvectors of its second central-moment (weighted
//! covariance) matrix, via `itk::ImageMomentsCalculator::GetPrincipalAxes`
//! (`itkImageMomentsCalculator.hxx:148-171`) — and sets
//! `rotationMatrix = movingPrincipalAxis · fixedPrincipalAxis⁻¹`
//! (`itkCenteredVersorTransformInitializer.hxx:39-50`) as the transform's
//! matrix.
//!
//! `GetPrincipalAxes` sorts eigenvalues **ascending**, sign-canonicalizes
//! each eigenvector so its largest-magnitude component is positive, then
//! flips the *last* row's sign if needed to force `det(Pa) = +1` (a proper
//! rotation) — this port's [`principal_axes`] follows the same three steps,
//! reusing the shared [`crate::eigen::jacobi_eigen_symmetric`] solver.
//!
//! **Quirk reproduced, not a guaranteed exact recovery:** because each
//! image's eigenvector signs are canonicalized independently (based only on
//! that image's own components, with no cross-image consistency check
//! beyond the single last-row parity flip), the resulting rotation is a
//! coarse initial alignment guess, not a closed-form solution — even for two
//! images with non-degenerate, pairwise-distinct principal moments, the
//! computed rotation is not guaranteed to be the true relative rotation
//! between them (an individual axis can come out 180°-flipped whenever the
//! two images' independently-chosen eigenvector signs happen to disagree).
//! This matches ITK's own documented framing of this filter as an
//! *initializer* for iterative registration, not an exact aligner (ledger
//! §2.76).
//!
//! # Degenerate inputs
//!
//! A total intensity mass of zero is rejected by the underlying
//! [`CenteredTransformInitializer`] moments-mode step
//! ([`RegistrationError::ZeroTotalMass`](crate::RegistrationError::ZeroTotalMass)),
//! matching `itk::ImageMomentsCalculator::Compute`'s own zero-mass guard
//! (`itkImageMomentsCalculator.hxx:120-124`) — before `ComputeRotation` ever
//! runs. A single-voxel (or otherwise perfectly symmetric) image has a zero
//! second central-moment matrix; its "principal axes" then reduce to
//! whatever orthonormal basis the eigensolver returns for a triply-degenerate
//! zero eigenvalue (the identity, in practice, since
//! [`crate::eigen::jacobi_eigen_symmetric`] never rotates a matrix that is
//! already diagonal) — not an error, matching ITK, which likewise performs
//! no additional validation before eigendecomposing `m_Cm`.
//!
//! ```
//! use sitk_core::Image;
//! use sitk_registration::CenteredVersorTransformInitializer;
//! use sitk_transform::VersorRigid3DTransform;
//!
//! // A 3-D image with a single unit-mass voxel at index (x, y, z).
//! fn point_mass(size: &[usize], x: usize, y: usize, z: usize) -> Image {
//!     let mut v = vec![0.0f64; size[0] * size[1] * size[2]];
//!     v[(z * size[1] + y) * size[0] + x] = 1.0;
//!     Image::from_vec(size, v).unwrap()
//! }
//!
//! // Fixed mass at (2,1,1); moving mass at (5,4,1): translation aligns the
//! // two centers of gravity, exactly as CenteredTransformInitializer.
//! let fixed = point_mass(&[10, 10, 4], 2, 1, 1);
//! let moving = point_mass(&[10, 10, 4], 5, 4, 1);
//!
//! let mut transform = VersorRigid3DTransform::identity();
//! CenteredVersorTransformInitializer::new(false)
//!     .initialize(&fixed, &moving, &mut transform)
//!     .unwrap();
//! assert_eq!(transform.center(), &[2.0, 1.0, 1.0]);
//! assert_eq!(transform.translation(), &[3.0, 3.0, 0.0]);
//! ```

use crate::eigen::jacobi_eigen_symmetric;
use crate::error::Result;
use crate::initializer::{CenteredTransformInitializer, OperationMode};
use sitk_core::Image;
use sitk_transform::VersorRigid3DTransform;

/// Initializes a [`VersorRigid3DTransform`]'s center, translation, and
/// (optionally) rotation from paired fixed/moving images. See the
/// [module docs](self).
#[derive(Clone, Copy, Debug)]
pub struct CenteredVersorTransformInitializer {
    compute_rotation: bool,
}

impl CenteredVersorTransformInitializer {
    /// A new initializer. `compute_rotation` mirrors ITK's `ComputeRotation`
    /// flag (default `false` upstream): when set, `initialize` additionally
    /// aligns the images' principal axes. See the [module docs](self).
    pub fn new(compute_rotation: bool) -> Self {
        Self { compute_rotation }
    }

    /// Initialize `transform`'s center, translation, and (if
    /// `compute_rotation`) rotation from `fixed` and `moving`. See the
    /// [module docs](self).
    ///
    /// Errors if `fixed`/`moving` differ in dimension, either is not 3-D (the
    /// dimension of [`VersorRigid3DTransform`]), or either has zero total
    /// intensity mass — all surfaced by the underlying
    /// [`CenteredTransformInitializer`] call.
    pub fn initialize(
        &self,
        fixed: &Image,
        moving: &Image,
        transform: &mut VersorRigid3DTransform,
    ) -> Result<()> {
        CenteredTransformInitializer::new(OperationMode::Moments)
            .initialize(fixed, moving, transform)?;

        if self.compute_rotation {
            let fixed_cg = transform.center().to_vec();
            let moving_cg: Vec<f64> = fixed_cg
                .iter()
                .zip(transform.translation())
                .map(|(c, t)| c + t)
                .collect();

            let fixed_pa = principal_axes(&second_central_moments(fixed, &fixed_cg)?);
            let moving_pa = principal_axes(&second_central_moments(moving, &moving_cg)?);

            transform.set_matrix(&rotation_from_principal_axes(&moving_pa, &fixed_pa))?;
        }

        Ok(())
    }
}

/// Second central moments of `img` about its (already known) physical-space
/// center of gravity `cg`: `Cm[i][j] = Σ(value·physᵢ·physⱼ)/M0 − cgᵢ·cgⱼ`,
/// mirroring `itk::ImageMomentsCalculator::Compute`'s `m_Cm`
/// (`itkImageMomentsCalculator.hxx:105-113,133-134,143-144`). Row-major
/// `dim x dim`.
///
/// Iterates every pixel in the same raster order as
/// [`initializer::center_of_gravity`](crate::initializer), which already
/// computed `cg` via an identical pass — done as a second, independent pass
/// here since central moments require `cg` up front.
fn second_central_moments(img: &Image, cg: &[f64]) -> Result<Vec<f64>> {
    let dim = img.dimension();
    let size = img.size();
    let values = img.to_f64_vec()?;

    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1];
    }

    let mut m0 = 0.0f64;
    let mut cm = vec![0.0f64; dim * dim];
    let mut index = vec![0.0f64; dim];
    for (p, &v) in values.iter().enumerate() {
        for d in 0..dim {
            index[d] = ((p / strides[d]) % size[d]) as f64;
        }
        let phys = img.continuous_index_to_physical_point(&index);
        m0 += v;
        for i in 0..dim {
            for j in 0..dim {
                cm[i * dim + j] += v * phys[i] * phys[j];
            }
        }
    }

    // m0 != 0 here: the moments-mode CenteredTransformInitializer call in
    // `initialize` already summed the same total mass and would have
    // returned RegistrationError::ZeroTotalMass first.
    for c in cm.iter_mut() {
        *c /= m0;
    }
    for i in 0..dim {
        for j in 0..dim {
            cm[i * dim + j] -= cg[i] * cg[j];
        }
    }
    Ok(cm)
}

/// Row-major `3x3` principal axes of a symmetric second central-moment
/// matrix `cm`: ascending-eigenvalue-sorted unit eigenvectors as **rows**,
/// each sign-canonicalized so its largest-magnitude component is positive,
/// with the last row's sign flipped if needed so the whole matrix is a
/// proper rotation (`det = +1`) — mirrors
/// `itk::ImageMomentsCalculator::Compute`'s `m_Pa`
/// (`itkImageMomentsCalculator.hxx:148-171`), built there from
/// `itk::SymmetricEigenDecomposition` (ascending eigenvalues, columns
/// sign-canonicalized) plus a `RealEigenDecomposition`-based determinant
/// correction.
///
/// This port computes that correction's determinant directly by cofactor
/// expansion instead of via a second (complex-valued) eigendecomposition —
/// exact-arithmetic identical, since the product of a matrix's eigenvalues
/// always equals its determinant (complex-conjugate pairs cancel their
/// imaginary parts). The same simplification is already used in
/// `label_shape.rs` (ledger §4.5).
fn principal_axes(cm: &[f64]) -> Vec<f64> {
    let n = 3;
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(cm, n);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let mut pa = vec![0.0f64; n * n];
    for (row, &col) in order.iter().enumerate() {
        let mut v: Vec<f64> = (0..n).map(|r| eigenvectors[r * n + col]).collect();
        let (max_i, _) = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        if v[max_i] < 0.0 {
            for x in &mut v {
                *x = -*x;
            }
        }
        pa[row * n..row * n + n].copy_from_slice(&v);
    }

    if determinant3(&pa) < 0.0 {
        for c in pa.iter_mut().skip((n - 1) * n) {
            *c = -*c;
        }
    }
    pa
}

/// `movingPrincipalAxis · fixedPrincipalAxis⁻¹`
/// (`itkCenteredVersorTransformInitializer.hxx:47`). Since a principal-axes
/// matrix is orthogonal by construction (rows are the pairwise-orthogonal
/// unit eigenvectors of a symmetric matrix), `fixedPrincipalAxis⁻¹ =
/// fixedPrincipalAxisᵀ`, so entry `[i][j]` is the dot product of `moving`'s
/// row `i` and `fixed`'s row `j` — computed directly here rather than via a
/// separate transpose and matrix product.
fn rotation_from_principal_axes(moving_pa: &[f64], fixed_pa: &[f64]) -> Vec<f64> {
    let n = 3;
    let mut r = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            r[i * n + j] = (0..n)
                .map(|k| moving_pa[i * n + k] * fixed_pa[j * n + k])
                .sum();
        }
    }
    r
}

/// Determinant of a row-major 3×3 matrix via direct cofactor expansion.
fn determinant3(m: &[f64]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::matrix;

    /// A `w×h×d` image whose only nonzero pixel is a unit mass at index
    /// `(x, y, z)`.
    fn point_mass(w: usize, h: usize, d: usize, x: usize, y: usize, z: usize) -> Image {
        let mut v = vec![0.0f64; w * h * d];
        v[(z * h + y) * w + x] = 1.0;
        Image::from_vec(&[w, h, d], v).unwrap()
    }

    /// An image with unit masses at `points` (each `(x, y, z)`), for building
    /// a distribution with a chosen, non-degenerate covariance shape.
    fn masses(w: usize, h: usize, d: usize, points: &[(usize, usize, usize)]) -> Image {
        let mut v = vec![0.0f64; w * h * d];
        for &(x, y, z) in points {
            v[(z * h + y) * w + x] = 1.0;
        }
        Image::from_vec(&[w, h, d], v).unwrap()
    }

    #[test]
    fn without_rotation_matches_centered_transform_initializer() {
        let fixed = point_mass(20, 20, 6, 6, 8, 2);
        let moving = point_mass(20, 20, 6, 13, 4, 3);

        let mut transform = VersorRigid3DTransform::identity();
        CenteredVersorTransformInitializer::new(false)
            .initialize(&fixed, &moving, &mut transform)
            .unwrap();

        assert_eq!(transform.center(), &[6.0, 8.0, 2.0]);
        assert_eq!(transform.translation(), &[7.0, -4.0, 1.0]);
        // No rotation requested: matrix stays identity.
        assert_eq!(
            transform.matrix(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn compute_rotation_on_vs_off_differ() {
        // Distinct, non-degenerate extents along each axis (25, 9, 1) for
        // fixed; moving's masses are fixed's rotated 90 degrees about z
        // (x,y,z) -> (-y,x,z), which stays on an integer grid.
        let fixed = masses(
            21,
            21,
            5,
            &[
                (15, 10, 2),
                (5, 10, 2), // +-5 along x: variance ~25
                (10, 13, 2),
                (10, 7, 2), // +-3 along y: variance ~9
                (10, 10, 3),
                (10, 10, 1), // +-1 along z: variance ~1
            ],
        );
        let moving = masses(
            21,
            21,
            5,
            &[
                (10, 15, 2),
                (10, 5, 2),
                (7, 10, 2),
                (13, 10, 2),
                (10, 10, 3),
                (10, 10, 1),
            ],
        );

        let mut without_rotation = VersorRigid3DTransform::identity();
        CenteredVersorTransformInitializer::new(false)
            .initialize(&fixed, &moving, &mut without_rotation)
            .unwrap();
        assert_eq!(
            without_rotation.matrix(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );

        let mut with_rotation = VersorRigid3DTransform::identity();
        CenteredVersorTransformInitializer::new(true)
            .initialize(&fixed, &moving, &mut with_rotation)
            .unwrap();

        // Center/translation are identical either way (rotation doesn't
        // affect the moments-mode center/translation step).
        assert_eq!(with_rotation.center(), without_rotation.center());
        assert_eq!(with_rotation.translation(), without_rotation.translation());
        // But the matrix now differs: a genuine rotation was computed.
        assert_ne!(with_rotation.matrix(), without_rotation.matrix());

        // The recovered matrix is a proper rotation (orthonormal, det = +1),
        // regardless of which specific axis-sign convention the eigensolver
        // picked (see module docs: this is a coarse alignment, not
        // necessarily the exact relative rotation).
        let m = with_rotation.matrix();
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[i * 3 + k] * m[j * 3 + k]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-9, "matrix not orthonormal");
            }
        }
        assert!(
            (determinant3(m) - 1.0).abs() < 1e-9,
            "not a proper rotation"
        );
    }

    #[test]
    fn compute_rotation_pinned_value_for_known_principal_axes() {
        // Fixed: masses at (+-3,0,0),(0,+-2,0),(0,0,+-1) about center (10,10,10)
        // -> covariance diag(9,4,1) in (x,y,z). Moving: the same pattern with
        // x and y swapped -> covariance diag(4,9,1).
        let fixed = masses(
            21,
            21,
            21,
            &[
                (13, 10, 10),
                (7, 10, 10),
                (10, 12, 10),
                (10, 8, 10),
                (10, 10, 11),
                (10, 10, 9),
            ],
        );
        let moving = masses(
            21,
            21,
            21,
            &[
                (12, 10, 10),
                (8, 10, 10),
                (10, 13, 10),
                (10, 7, 10),
                (10, 10, 11),
                (10, 10, 9),
            ],
        );

        let mut transform = VersorRigid3DTransform::identity();
        CenteredVersorTransformInitializer::new(true)
            .initialize(&fixed, &moving, &mut transform)
            .unwrap();

        // Hand-computed from the exact algorithm (ascending eigenvalue sort,
        // largest-magnitude-positive sign canonicalization, last-row parity
        // correction, R[i][j] = moving_row_i . fixed_row_j): see module docs
        // for why this is a valid proper rotation but not necessarily "the"
        // intuitive 90-degree swap.
        let expected = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let got = transform.matrix();
        for (e, g) in expected.iter().zip(got.iter()) {
            assert!((e - g).abs() < 1e-9, "expected {expected:?}, got {got:?}");
        }
    }

    #[test]
    fn zero_mass_image_is_rejected_before_rotation_runs() {
        let fixed = Image::from_vec(&[4, 4, 4], vec![0.0; 64]).unwrap();
        let moving = point_mass(4, 4, 4, 1, 1, 1);
        let mut transform = VersorRigid3DTransform::identity();
        let err = CenteredVersorTransformInitializer::new(true)
            .initialize(&fixed, &moving, &mut transform)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::RegistrationError::ZeroTotalMass { which: "fixed" }
        ));
    }

    #[test]
    fn single_voxel_image_rotation_is_identity_not_an_error() {
        // A single-mass image has a zero second central-moment matrix
        // (perfectly degenerate): ITK performs no extra validation before
        // eigendecomposing it, and neither does this port.
        let fixed = point_mass(6, 6, 6, 3, 3, 3);
        let moving = point_mass(6, 6, 6, 4, 2, 3);
        let mut transform = VersorRigid3DTransform::identity();
        CenteredVersorTransformInitializer::new(true)
            .initialize(&fixed, &moving, &mut transform)
            .unwrap();
        assert_eq!(
            transform.matrix(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let fixed = Image::from_vec(&[4, 4, 4], vec![1.0; 64]).unwrap();
        let moving = Image::from_vec(&[4, 4], vec![1.0; 16]).unwrap();
        let mut transform = VersorRigid3DTransform::identity();
        let err = CenteredVersorTransformInitializer::new(false)
            .initialize(&fixed, &moving, &mut transform)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::RegistrationError::DimensionMismatch {
                fixed: 3,
                moving: 2
            }
        ));
    }

    #[test]
    fn non_3d_image_is_rejected() {
        let fixed = Image::from_vec(&[4, 4], vec![1.0; 16]).unwrap();
        let moving = Image::from_vec(&[4, 4], vec![1.0; 16]).unwrap();
        let mut transform = VersorRigid3DTransform::identity();
        let err = CenteredVersorTransformInitializer::new(false)
            .initialize(&fixed, &moving, &mut transform)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::RegistrationError::TransformDimensionMismatch {
                transform: 3,
                image: 2
            }
        ));
    }

    #[test]
    fn principal_axes_is_orthonormal_proper_rotation() {
        let cm = vec![2.0, 0.3, -0.1, 0.3, 3.0, 0.2, -0.1, 0.2, 5.0];
        let pa = principal_axes(&cm);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| pa[i * 3 + k] * pa[j * 3 + k]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-9, "not orthonormal");
            }
        }
        assert!((determinant3(&pa) - 1.0).abs() < 1e-9);

        // Each row must be an eigenvector of cm.
        let (eigenvalues, _) = jacobi_eigen_symmetric(&cm, 3);
        let mut sorted_eigenvalues = eigenvalues.clone();
        sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for row in 0..3 {
            let v: Vec<f64> = pa[row * 3..row * 3 + 3].to_vec();
            let cv = matrix::mat_vec(&cm, &v, 3);
            for d in 0..3 {
                assert!(
                    (cv[d] - sorted_eigenvalues[row] * v[d]).abs() < 1e-7,
                    "row {row} is not an eigenvector: {cv:?} vs {v:?} * {}",
                    sorted_eigenvalues[row]
                );
            }
        }
    }
}
