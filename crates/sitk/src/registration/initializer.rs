//! Initialize a centered transform's center and translation from the fixed and
//! moving images, mirroring `itk::CenteredTransformInitializer` (SimpleITK's
//! `CenteredTransformInitializerFilter`).
//!
//! Both modes set the transform's fixed **center** to the fixed image's center
//! and its **translation** so that the fixed center maps onto the moving center
//! (`translation = movingCenter − fixedCenter`); they differ only in how each
//! image's center is found:
//!
//! - [`OperationMode::Geometry`] — the physical point of the continuous index
//!   `(size − 1) / 2` (the image's geometric center).
//! - [`OperationMode::Moments`] — the intensity-weighted mean physical point
//!   (the center of gravity), matching `itk::ImageMomentsCalculator`.
//!
//! With no rotation set, the initialized transform maps the fixed center exactly
//! onto the moving center — the standard cold-start for a subsequent
//! registration. SimpleITK defaults this filter to `Moments`.
//!
//! ```
//! use sitk::core::Image;
//! use sitk::registration::{CenteredTransformInitializer, OperationMode};
//! use sitk::transform::Euler2DTransform;
//!
//! // Fixed 10×10 with origin (0,0); moving 10×10 shifted to origin (4,-2).
//! let fixed = Image::from_vec(&[10, 10], vec![1.0; 100]).unwrap();
//! let mut moving = Image::from_vec(&[10, 10], vec![1.0; 100]).unwrap();
//! moving.set_origin(&[4.0, -2.0]).unwrap();
//!
//! let mut tx = Euler2DTransform::identity();
//! CenteredTransformInitializer::new(OperationMode::Geometry)
//!     .initialize(&fixed, &moving, &mut tx)
//!     .unwrap();
//!
//! // Geometric centers are both index (4.5, 4.5); the moving one is at +(4,-2).
//! assert_eq!(tx.center(), &[4.5, 4.5]);
//! assert_eq!(tx.translation(), &[4.0, -2.0]);
//! ```

use crate::core::Image;
use crate::registration::error::{RegistrationError, Result};
use crate::transform::CenteredTransform;

/// How [`CenteredTransformInitializer`] locates each image's center.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationMode {
    /// Geometric center: the physical point of the continuous index
    /// `(size − 1) / 2`.
    Geometry,
    /// Center of gravity: the intensity-weighted mean physical point.
    Moments,
}

/// Sets a [`CenteredTransform`]'s center and translation to align the fixed and
/// moving images' centers. See the [module docs](self).
#[derive(Clone, Copy, Debug)]
pub struct CenteredTransformInitializer {
    mode: OperationMode,
}

impl CenteredTransformInitializer {
    /// A new initializer using `mode` (SimpleITK defaults to
    /// [`OperationMode::Moments`]).
    pub fn new(mode: OperationMode) -> Self {
        Self { mode }
    }

    /// Compute the fixed and moving centers under the configured mode and set
    /// `transform`'s center to the fixed center and its translation to
    /// `movingCenter − fixedCenter`. The transform's matrix (rotation/scale) is
    /// left untouched.
    ///
    /// Errors if the images differ in dimension, the transform's dimension does
    /// not match the images', or (in [`OperationMode::Moments`]) an image has
    /// zero total intensity mass.
    pub fn initialize<T: CenteredTransform + ?Sized>(
        &self,
        fixed: &Image,
        moving: &Image,
        transform: &mut T,
    ) -> Result<()> {
        let dim = fixed.dimension();
        if moving.dimension() != dim {
            return Err(RegistrationError::DimensionMismatch {
                fixed: dim,
                moving: moving.dimension(),
            });
        }
        if transform.dimension() != dim {
            return Err(RegistrationError::TransformDimensionMismatch {
                transform: transform.dimension(),
                image: dim,
            });
        }

        let fixed_center = self.image_center(fixed, "fixed")?;
        let moving_center = self.image_center(moving, "moving")?;
        let translation: Vec<f64> = (0..dim)
            .map(|i| moving_center[i] - fixed_center[i])
            .collect();

        transform.set_center(&fixed_center);
        transform.set_translation(&translation);
        Ok(())
    }

    /// The image's center in physical space under the configured mode.
    fn image_center(&self, img: &Image, which: &'static str) -> Result<Vec<f64>> {
        match self.mode {
            OperationMode::Geometry => Ok(geometric_center(img)),
            OperationMode::Moments => center_of_gravity(img, which),
        }
    }
}

/// Physical point of the continuous index `(size − 1) / 2` — the image's
/// geometric center (ITK's `CenteredTransformInitializer` geometry branch).
fn geometric_center(img: &Image) -> Vec<f64> {
    let center_index: Vec<f64> = img.size().iter().map(|&s| (s as f64 - 1.0) / 2.0).collect();
    img.continuous_index_to_physical_point(&center_index)
}

/// Intensity-weighted mean physical point `Σ phys(idx)·I(idx) / Σ I(idx)`,
/// matching `itk::ImageMomentsCalculator::GetCenterOfGravity`. Iterates every
/// pixel in raster order (first index fastest) using the raw pixel value as the
/// weight. Errors if the total mass is zero.
fn center_of_gravity(img: &Image, which: &'static str) -> Result<Vec<f64>> {
    let dim = img.dimension();
    let size = img.size();
    let values = img.to_f64_vec()?;

    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1];
    }

    let mut m0 = 0.0f64;
    let mut cg = vec![0.0f64; dim];
    let mut index = vec![0.0f64; dim];
    for (p, &v) in values.iter().enumerate() {
        for d in 0..dim {
            index[d] = ((p / strides[d]) % size[d]) as f64;
        }
        let phys = img.continuous_index_to_physical_point(&index);
        m0 += v;
        for d in 0..dim {
            cg[d] += phys[d] * v;
        }
    }

    if m0 == 0.0 {
        return Err(RegistrationError::ZeroTotalMass { which });
    }
    for c in &mut cg {
        *c /= m0;
    }
    Ok(cg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{AffineTransform, Euler2DTransform, TransformBase};

    /// A `w×h` image whose only nonzero pixel is a unit mass at index `(x, y)`.
    fn point_mass(w: usize, h: usize, x: usize, y: usize) -> Image {
        let mut v = vec![0.0f64; w * h];
        v[y * w + x] = 1.0;
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn geometry_center_is_physical_center_and_translation_aligns_centers() {
        // Fixed origin (0,0), moving origin (3,-1); both 8×6 unit spacing.
        let fixed = Image::from_vec(&[8, 6], vec![1.0; 48]).unwrap();
        let mut moving = Image::from_vec(&[8, 6], vec![1.0; 48]).unwrap();
        moving.set_origin(&[3.0, -1.0]).unwrap();

        let mut tx = Euler2DTransform::identity();
        CenteredTransformInitializer::new(OperationMode::Geometry)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap();

        // Geometric center index = ((8-1)/2, (6-1)/2) = (3.5, 2.5).
        assert_eq!(tx.center(), &[3.5, 2.5]);
        // translation = movingCenter − fixedCenter = origin difference (3,-1).
        assert_eq!(tx.translation(), &[3.0, -1.0]);
        // With no rotation the fixed center maps onto the moving center.
        let mapped = tx.transform_point(&[3.5, 2.5]);
        assert!(
            (mapped[0] - 6.5).abs() < 1e-12 && (mapped[1] - 1.5).abs() < 1e-12,
            "{mapped:?}"
        );
    }

    #[test]
    fn geometry_accounts_for_spacing() {
        // Spacing 2 ⇒ geometric center physical = index (3.5,2.5) × 2.
        let mut fixed = Image::from_vec(&[8, 6], vec![1.0; 48]).unwrap();
        fixed.set_spacing(&[2.0, 2.0]).unwrap();
        let mut moving = Image::from_vec(&[8, 6], vec![1.0; 48]).unwrap();
        moving.set_spacing(&[2.0, 2.0]).unwrap();

        let mut tx = AffineTransform::identity(2);
        CenteredTransformInitializer::new(OperationMode::Geometry)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap();
        assert_eq!(tx.center(), &[7.0, 5.0]);
        assert_eq!(tx.translation(), &[0.0, 0.0]);
    }

    #[test]
    fn moments_center_is_center_of_gravity() {
        // Fixed mass at index (2,1), moving mass at index (5,4): CoG = the mass
        // location; translation = (5,4) − (2,1) = (3,3).
        let fixed = point_mass(10, 10, 2, 1);
        let moving = point_mass(10, 10, 5, 4);

        let mut tx = Euler2DTransform::identity();
        CenteredTransformInitializer::new(OperationMode::Moments)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap();
        assert_eq!(tx.center(), &[2.0, 1.0]);
        assert_eq!(tx.translation(), &[3.0, 3.0]);
    }

    #[test]
    fn moments_weights_by_intensity() {
        // Two masses in the fixed image: value 1 at index 0 and value 3 at index
        // 4 (row 0) ⇒ CoG_x = (0·1 + 4·3)/(1+3) = 3, CoG_y = 0.
        let mut v = vec![0.0f64; 10 * 10];
        v[0] = 1.0;
        v[4] = 3.0;
        let fixed = Image::from_vec(&[10, 10], v).unwrap();
        let moving = point_mass(10, 10, 3, 0);

        let mut tx = AffineTransform::identity(2);
        CenteredTransformInitializer::new(OperationMode::Moments)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap();
        assert_eq!(tx.center(), &[3.0, 0.0]);
        assert_eq!(tx.translation(), &[0.0, 0.0]);
    }

    #[test]
    fn moments_zero_mass_is_rejected() {
        let fixed = Image::from_vec(&[4, 4], vec![0.0; 16]).unwrap();
        let moving = point_mass(4, 4, 1, 1);
        let mut tx = Euler2DTransform::identity();
        let err = CenteredTransformInitializer::new(OperationMode::Moments)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::ZeroTotalMass { which: "fixed" }
        ));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let fixed = Image::from_vec(&[4, 4], vec![1.0; 16]).unwrap();
        let moving = Image::from_vec(&[4, 4, 4], vec![1.0; 64]).unwrap();
        let mut tx = Euler2DTransform::identity();
        let err = CenteredTransformInitializer::new(OperationMode::Geometry)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::DimensionMismatch {
                fixed: 2,
                moving: 3
            }
        ));
    }

    #[test]
    fn preserves_rotation_and_maps_center_onto_center() {
        // A transform that already carries a rotation keeps it; only center and
        // translation are set, so the fixed center still maps onto the moving one.
        use std::f64::consts::FRAC_PI_4;
        let fixed = point_mass(20, 20, 6, 8);
        let moving = point_mass(20, 20, 13, 4);

        let mut tx = Euler2DTransform::new(FRAC_PI_4, [0.0, 0.0], [0.0, 0.0]);
        let matrix_before = tx.matrix().to_vec();
        CenteredTransformInitializer::new(OperationMode::Moments)
            .initialize(&fixed, &moving, &mut tx)
            .unwrap();

        assert_eq!(tx.matrix(), &matrix_before[..]); // rotation untouched
        assert_eq!(tx.angle(), FRAC_PI_4);
        // y(fixedCenter) = R·0 + fixedCenter + (movingCenter − fixedCenter).
        let mapped = tx.transform_point(&[6.0, 8.0]);
        assert!(
            (mapped[0] - 13.0).abs() < 1e-12 && (mapped[1] - 4.0).abs() < 1e-12,
            "{mapped:?}"
        );
    }
}
