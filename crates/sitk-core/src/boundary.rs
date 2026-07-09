//! Out-of-bounds pixel value rules for [`crate::neighborhood::NeighborhoodIterator`].
//!
//! Mirrors ITK's `itk::ImageBoundaryCondition` hierarchy
//! (itkImageBoundaryCondition.h): each implementation supplies the pixel
//! value ITK would return for an index that has walked off the edge of the
//! image while sliding a neighborhood window across it.

use crate::image::Image;
use crate::pixel::Scalar;

/// Supplies a pixel value for a (possibly) out-of-bounds ND `index` into
/// `image`.
///
/// `index` is signed and may be negative or `>= image.size()[d]` along any
/// axis `d`; each implementation decides how to remap it back onto the image
/// (itkImageBoundaryCondition.h:153-154, `GetPixel`).
pub trait BoundaryCondition<T: Scalar> {
    fn get_pixel(&self, index: &[i64], image: &Image) -> T;
}

/// Reads the pixel at an already-valid ND `index`.
fn pixel_at<T: Scalar>(image: &Image, index: &[usize]) -> T {
    let pixels: &[T] = image.scalar_slice::<T>().expect(
        "NeighborhoodIterator only constructs a BoundaryCondition<T> for an image whose pixel type is T",
    );
    pixels[image.linear_index(index)]
}

/// ITK's default boundary condition: clamps the out-of-bounds index to the
/// nearest in-bounds voxel along each axis independently.
///
/// itkZeroFluxNeumannBoundaryCondition.hxx:154-183 (`GetPixel`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroFluxNeumannBoundaryCondition;

impl<T: Scalar> BoundaryCondition<T> for ZeroFluxNeumannBoundaryCondition {
    fn get_pixel(&self, index: &[i64], image: &Image) -> T {
        let clamped: Vec<usize> = index
            .iter()
            .zip(image.size())
            .map(|(&i, &size)| i.clamp(0, size as i64 - 1) as usize)
            .collect();
        pixel_at(image, &clamped)
    }
}

/// Returns a fixed constant for any out-of-bounds index; an in-bounds index
/// still reads through to the image.
///
/// itkConstantBoundaryCondition.hxx:79-89 (`GetPixel`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantBoundaryCondition<T> {
    constant: T,
}

impl<T> ConstantBoundaryCondition<T> {
    /// A boundary condition that returns `constant` for every out-of-bounds
    /// index.
    pub fn new(constant: T) -> Self {
        Self { constant }
    }

    /// The constant value returned for out-of-bounds indices.
    pub fn constant(&self) -> T
    where
        T: Copy,
    {
        self.constant
    }
}

impl<T: Scalar> BoundaryCondition<T> for ConstantBoundaryCondition<T> {
    fn get_pixel(&self, index: &[i64], image: &Image) -> T {
        let inside = index
            .iter()
            .zip(image.size())
            .all(|(&i, &size)| i >= 0 && (i as usize) < size);
        if !inside {
            return self.constant;
        }
        let idx: Vec<usize> = index.iter().map(|&i| i as usize).collect();
        pixel_at(image, &idx)
    }
}

/// Wraps out-of-bounds indices around the image extent.
///
/// itkPeriodicBoundaryCondition.hxx:179-201 (`GetPixel`).
#[derive(Debug, Clone, Copy, Default)]
pub struct PeriodicBoundaryCondition;

impl<T: Scalar> BoundaryCondition<T> for PeriodicBoundaryCondition {
    fn get_pixel(&self, index: &[i64], image: &Image) -> T {
        let wrapped: Vec<usize> = index
            .iter()
            .zip(image.size())
            .map(|(&i, &size)| i.rem_euclid(size as i64) as usize)
            .collect();
        pixel_at(image, &wrapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    // 1-D: values equal their index, size 5, so a pinned value doubles as
    // the source index it was clamped/wrapped/defaulted from.
    fn image_1d() -> Image {
        Image::from_vec(&[5], vec![0u32, 1, 2, 3, 4]).unwrap()
    }

    #[test]
    fn zero_flux_neumann_clamps_1d_left_and_right_corners() {
        let img = image_1d();
        let bc = ZeroFluxNeumannBoundaryCondition;
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[-1], &img), 0);
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[5], &img), 4);
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[-100], &img), 0);
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[100], &img), 4);
    }

    #[test]
    fn constant_returns_constant_only_out_of_bounds_1d() {
        let img = image_1d();
        let bc = ConstantBoundaryCondition::new(99u32);
        assert_eq!(bc.get_pixel(&[-1], &img), 99);
        assert_eq!(bc.get_pixel(&[5], &img), 99);
        assert_eq!(bc.get_pixel(&[0], &img), 0);
        assert_eq!(bc.get_pixel(&[4], &img), 4);
    }

    #[test]
    fn constant_default_is_zero() {
        let bc: ConstantBoundaryCondition<i32> = Default::default();
        assert_eq!(bc.constant(), 0);
    }

    #[test]
    fn periodic_wraps_1d_left_and_right_corners() {
        let img = image_1d();
        let bc = PeriodicBoundaryCondition;
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[-1], &img), 4);
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[5], &img), 0);
        // Multi-wrap: -11 is 4 mod 5 (itkPeriodicBoundaryCondition.hxx:190-195).
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[-11], &img), 4);
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[11], &img), 1);
    }

    // 2-D: value(x, y) = x + 10*y.
    fn image_2d() -> Image {
        let (w, h) = (4usize, 3usize);
        let mut data = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = (x + 10 * y) as u32;
            }
        }
        Image::from_vec(&[w, h], data).unwrap()
    }

    #[test]
    fn zero_flux_neumann_clamps_2d_corner_and_edge() {
        let img = image_2d();
        let bc = ZeroFluxNeumannBoundaryCondition;
        // Corner (-1, -1) clamps to (0, 0) = 0.
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1], &img), 0);
        // Opposite corner (4, 3) clamps to (3, 2) = 23.
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[4, 3], &img), 23);
        // Top edge (1, -1) clamps y only -> (1, 0) = 1.
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1], &img), 1);
    }

    #[test]
    fn constant_2d_out_of_bounds_if_any_axis_spills() {
        let img = image_2d();
        let bc = ConstantBoundaryCondition::new(99u32);
        assert_eq!(bc.get_pixel(&[-1, -1], &img), 99);
        assert_eq!(bc.get_pixel(&[1, -1], &img), 99); // x in-bounds, y not.
        assert_eq!(bc.get_pixel(&[1, 0], &img), 1); // both in-bounds.
    }

    #[test]
    fn periodic_2d_wraps_each_axis_independently() {
        let img = image_2d();
        let bc = PeriodicBoundaryCondition;
        // (-1, -1) wraps to (3, 2) = 23.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1], &img),
            23
        );
        // (1, -1) wraps y only -> (1, 2) = 21.
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1], &img), 21);
        // (4, 0) wraps x only -> (0, 0) = 0.
        assert_eq!(BoundaryCondition::<u32>::get_pixel(&bc, &[4, 0], &img), 0);
    }

    // 3-D: value(x, y, z) = x + 10*y + 100*z.
    fn image_3d() -> Image {
        let (w, h, d) = (3usize, 3usize, 3usize);
        let mut data = vec![0u32; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (x + 10 * y + 100 * z) as u32;
                }
            }
        }
        Image::from_vec(&[w, h, d], data).unwrap()
    }

    #[test]
    fn zero_flux_neumann_clamps_3d_corner_and_edge() {
        let img = image_3d();
        let bc = ZeroFluxNeumannBoundaryCondition;
        // Corner (-1,-1,-1) clamps to (0,0,0) = 0.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1, -1], &img),
            0
        );
        // Opposite corner (3,3,3) clamps to (2,2,2) = 222.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[3, 3, 3], &img),
            222
        );
        // Edge (1,-1,-1): x in-bounds, y and z clamp -> (1,0,0) = 1.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1, -1], &img),
            1
        );
    }

    #[test]
    fn constant_3d_out_of_bounds_if_any_axis_spills() {
        let img = image_3d();
        let bc = ConstantBoundaryCondition::new(99u32);
        assert_eq!(bc.get_pixel(&[-1, -1, -1], &img), 99);
        assert_eq!(bc.get_pixel(&[1, -1, -1], &img), 99); // only x in-bounds.
        assert_eq!(bc.get_pixel(&[1, 1, -1], &img), 99); // x, y in-bounds, z not.
        assert_eq!(bc.get_pixel(&[1, 1, 1], &img), 111); // all in-bounds.
    }

    #[test]
    fn periodic_3d_wraps_each_axis_independently() {
        let img = image_3d();
        let bc = PeriodicBoundaryCondition;
        // (-1,-1,-1) wraps every axis -> (2,2,2) = 222.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1, -1], &img),
            222
        );
        // (1,-1,-1): x in-bounds, y and z wrap -> (1,2,2) = 221.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1, -1], &img),
            221
        );
    }
}
