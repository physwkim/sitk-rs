//! Initialize a cubic B-spline transform's domain (mesh size, origin,
//! physical dimensions, direction) from a reference image, mirroring
//! `itk::simple::BSplineTransformInitializerFilter`
//! (`sitkBSplineTransformInitializerFilter.h:48-155` /
//! `sitkBSplineTransformInitializerFilter.cxx:88-194`) over
//! `itk::BSplineTransformInitializer`.
//!
//! The domain math itself — corner enumeration, ⅛-voxel epsilon expansion,
//! bounding-box-nearest origin, greedy per-axis direction matching — is
//! already ported at
//! [`BSplineTransform::from_image_initializer`](sitk_transform::BSplineTransform::from_image_initializer);
//! this filter is a thin SimpleITK-shaped wrapper around it that supplies the
//! mesh-size default and validates the `order` parameter.
//!
//! # Mesh size
//!
//! Defaults to `1` per image dimension
//! ([`with_mesh_size`](BSplineTransformInitializer::with_mesh_size) overrides
//! it). SimpleITK's C++ default is a **hardcoded 3-long**
//! `std::vector<uint32_t>(3, 1u)` (`sitkBSplineTransformInitializerFilter.h:130`),
//! truncated to the image's actual dimension by `sitkSTLVectorToITK`
//! (ledger §3.3) — a C++ workaround for not knowing the image's dimension at
//! the filter's construction time. Since every entry is `1u` regardless, this
//! is behaviorally identical to "1 per dimension" for the 2-D/3-D images
//! SimpleITK's own filter supports; this port's [`execute`](BSplineTransformInitializer::execute)
//! learns the dimension from the image argument directly, with no such
//! workaround needed. A caller-supplied mesh size whose length does not match
//! the image's dimension is an error here (ledger §3.3/§4.9 convention: exact
//! length, not SimpleITK's truncate-if-longer).
//!
//! # Order
//!
//! SimpleITK's filter supports spline order 0-3 (`SetOrder`,
//! `sitkBSplineTransformInitializerFilter.h:86-95`; dispatch table
//! `sitkBSplineTransformInitializerFilter.cxx:123-136`, rejecting anything
//! else). This port implements only the cubic kernel
//! ([`BSplineTransform`](sitk_transform::BSplineTransform) hardcodes its
//! spline order to 3), so [`execute`](BSplineTransformInitializer::execute)
//! errors for any `order != 3`.
//!
//! # Dimension
//!
//! SimpleITK's member-function factory registers this filter only for 2-D
//! and 3-D images (`sitkBSplineTransformInitializerFilter.cxx:53-56`); this
//! port's domain math and `BSplineTransform` are both dimension-generic, so
//! `execute` has no such cap.
//!
//! ```
//! use sitk_core::Image;
//! use sitk_registration::BSplineTransformInitializer;
//!
//! let image = Image::from_vec(&[10, 8], vec![1.0; 80]).unwrap();
//! let transform = BSplineTransformInitializer::new().execute(&image).unwrap();
//! assert_eq!(transform.grid_size(), &[4, 4]); // mesh size 1 + cubic order 3
//! ```

use crate::error::{RegistrationError, Result};
use sitk_core::Image;
use sitk_transform::BSplineTransform;

/// Builds a cubic-order [`BSplineTransform`] whose domain is initialized from
/// a reference image. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct BSplineTransformInitializer {
    mesh_size: Option<Vec<usize>>,
    order: usize,
}

impl Default for BSplineTransformInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl BSplineTransformInitializer {
    /// A new initializer with the default mesh size (`1` per dimension,
    /// applied once the image's dimension is known at
    /// [`execute`](Self::execute) time) and cubic order.
    pub fn new() -> Self {
        Self {
            mesh_size: None,
            order: 3,
        }
    }

    /// Override the per-axis mesh size (number of B-spline polynomial
    /// patches per axis). Must match the image's dimension when
    /// [`execute`](Self::execute) is called, else
    /// [`RegistrationError::MeshSizeLength`].
    pub fn with_mesh_size(mut self, mesh_size: Vec<usize>) -> Self {
        self.mesh_size = Some(mesh_size);
        self
    }

    /// Override the B-spline order. Only `3` (cubic) is supported by this
    /// port; any other value is rejected by [`execute`](Self::execute) with
    /// [`RegistrationError::UnsupportedBSplineOrder`].
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Build the transform. See the [module docs](self).
    pub fn execute(&self, image: &Image) -> Result<BSplineTransform> {
        if self.order != 3 {
            return Err(RegistrationError::UnsupportedBSplineOrder { order: self.order });
        }

        let dim = image.dimension();
        let mesh_size = match &self.mesh_size {
            Some(m) => {
                if m.len() != dim {
                    return Err(RegistrationError::MeshSizeLength {
                        got: m.len(),
                        expected: dim,
                    });
                }
                m.clone()
            }
            None => vec![1usize; dim],
        };

        Ok(BSplineTransform::from_image_initializer(image, &mesh_size)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mesh_size_is_one_per_dimension_2d() {
        let image = Image::from_vec(&[10, 8], vec![1.0; 80]).unwrap();
        let transform = BSplineTransformInitializer::new().execute(&image).unwrap();
        // grid_size = mesh_size + spline_order = 1 + 3 = 4 per axis.
        assert_eq!(transform.grid_size(), &[4, 4]);
    }

    #[test]
    fn default_mesh_size_is_one_per_dimension_3d() {
        let image = Image::from_vec(&[6, 6, 6], vec![1.0; 216]).unwrap();
        let transform = BSplineTransformInitializer::new().execute(&image).unwrap();
        assert_eq!(transform.grid_size(), &[4, 4, 4]);
    }

    #[test]
    fn custom_mesh_size_is_used() {
        let image = Image::from_vec(&[10, 8], vec![1.0; 80]).unwrap();
        let transform = BSplineTransformInitializer::new()
            .with_mesh_size(vec![2, 5])
            .execute(&image)
            .unwrap();
        assert_eq!(transform.grid_size(), &[5, 8]); // mesh + 3
    }

    #[test]
    fn mesh_size_length_mismatch_is_rejected() {
        let image = Image::from_vec(&[10, 8], vec![1.0; 80]).unwrap();
        let err = BSplineTransformInitializer::new()
            .with_mesh_size(vec![1, 1, 1])
            .execute(&image)
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::MeshSizeLength {
                got: 3,
                expected: 2
            }
        ));
    }

    #[test]
    fn unsupported_order_is_rejected() {
        let image = Image::from_vec(&[10, 8], vec![1.0; 80]).unwrap();
        let err = BSplineTransformInitializer::new()
            .with_order(2)
            .execute(&image)
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::UnsupportedBSplineOrder { order: 2 }
        ));
    }

    #[test]
    fn matches_direct_bspline_transform_call() {
        // The wrapper must delegate exactly to `from_image_initializer`, not
        // reimplement any of the domain math.
        let image = Image::from_vec(&[7, 9], vec![1.0; 63]).unwrap();
        let via_wrapper = BSplineTransformInitializer::new()
            .with_mesh_size(vec![3, 4])
            .execute(&image)
            .unwrap();
        let direct = BSplineTransform::from_image_initializer(&image, &[3, 4]).unwrap();
        assert_eq!(via_wrapper.grid_origin(), direct.grid_origin());
        assert_eq!(via_wrapper.grid_spacing(), direct.grid_spacing());
        assert_eq!(via_wrapper.grid_size(), direct.grid_size());
    }

    #[test]
    fn zero_mesh_size_surfaces_transform_error() {
        let image = Image::from_vec(&[10, 8], vec![1.0; 80]).unwrap();
        let err = BSplineTransformInitializer::new()
            .with_mesh_size(vec![0, 1])
            .execute(&image)
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::Transform(sitk_transform::TransformError::InvalidBSplineDomain)
        ));
    }
}
