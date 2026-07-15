//! Out-of-bounds pixel value rules for [`crate::core::neighborhood::NeighborhoodIterator`].
//!
//! Mirrors ITK's `itk::ImageBoundaryCondition` hierarchy
//! (itkImageBoundaryCondition.h): each implementation supplies the pixel
//! value ITK would return for an index that has walked off the edge of the
//! image while sliding a neighborhood window across it.

use crate::core::image::ScalarView;
use crate::core::pixel::Scalar;

/// Supplies a pixel value for a (possibly) out-of-bounds ND `index` into
/// `image`.
///
/// `index` is signed and may be negative or `>= image.size()[d]` along any
/// axis `d`; each implementation decides how to remap it back onto the image
/// (itkImageBoundaryCondition.h:153-154, `GetPixel`).
///
/// `image` is a [`ScalarView`], not an [`Image`](crate::core::Image): the read is
/// infallible only because the view already proves the image is scalar and has
/// pixel type `T`. Callers discharge that proof once, with
/// [`Image::scalar_view`](crate::core::Image::scalar_view), outside the pixel loop.
pub trait BoundaryCondition<T: Scalar> {
    fn get_pixel(&self, index: &[i64], image: &ScalarView<'_, T>) -> T;
}

/// The pixel `index` lands on when each axis is remapped by `map` — the one
/// primitive every condition below is written in terms of.
///
/// A boundary condition is a per-axis rule, and the pixel it names sits at a
/// single linear index. Accumulating that index directly — dimension-0-fastest,
/// exactly as [`Image::linear_index`](crate::core::Image::linear_index) does — means
/// no implementation ever materializes the remapped ND index, and materializing
/// it is the only reason any of them would allocate.
///
/// That matters because `get_pixel` runs once per out-of-bounds *neighbor*, not
/// once per filter: a 256³ `mean` at radius 2 calls it tens of millions of
/// times. A `Vec` in each call is what [`crate::core::parallel`]'s module docs warn
/// about — it serializes every core on the allocator, and measurably did: the
/// window walk ran 13.8 of 48 cores, against 43 for the same kernel with no
/// window.
///
/// `map(i, size)` must return an in-bounds coordinate for that axis; the caller
/// of this function, not this function, is what makes that true.
pub(crate) fn remapped<T: Scalar>(
    index: &[i64],
    image: &ScalarView<'_, T>,
    map: impl Fn(i64, usize) -> usize,
) -> T {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&i, &size) in index.iter().zip(image.size()) {
        offset += map(i, size) * stride;
        stride *= size;
    }
    image.pixels()[offset]
}

/// ITK's default boundary condition: clamps the out-of-bounds index to the
/// nearest in-bounds voxel along each axis independently.
///
/// itkZeroFluxNeumannBoundaryCondition.hxx:154-183 (`GetPixel`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroFluxNeumannBoundaryCondition;

impl<T: Scalar> BoundaryCondition<T> for ZeroFluxNeumannBoundaryCondition {
    fn get_pixel(&self, index: &[i64], image: &ScalarView<'_, T>) -> T {
        remapped(index, image, |i, size| i.clamp(0, size as i64 - 1) as usize)
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
    fn get_pixel(&self, index: &[i64], image: &ScalarView<'_, T>) -> T {
        let inside = index
            .iter()
            .zip(image.size())
            .all(|(&i, &size)| i >= 0 && (i as usize) < size);
        if !inside {
            return self.constant;
        }
        // `inside` proves every axis is already in bounds, so the identity map
        // is the in-bounds coordinate.
        remapped(index, image, |i, _| i as usize)
    }
}

/// Wraps out-of-bounds indices around the image extent.
///
/// itkPeriodicBoundaryCondition.hxx:179-201 (`GetPixel`).
#[derive(Debug, Clone, Copy, Default)]
pub struct PeriodicBoundaryCondition;

impl<T: Scalar> BoundaryCondition<T> for PeriodicBoundaryCondition {
    fn get_pixel(&self, index: &[i64], image: &ScalarView<'_, T>) -> T {
        remapped(index, image, |i, size| i.rem_euclid(size as i64) as usize)
    }
}

/// Reflects out-of-bounds indices back into the image, repeating the edge
/// pixel: period `2 * size`, tiling `[0, size)` direct then `[size, 2*size)`
/// reversed (so index `-1` reads the same pixel as index `0`, index `-2` the
/// same as index `1`, etc).
///
/// `MirrorPadImageFilter` has no standalone `itk::ImageBoundaryCondition`
/// class to port from; it implements this index mapping directly in
/// `RegionIsOdd`/`ConvertOutputIndexToInputIndex`
/// (itkMirrorPadImageFilter.hxx). The closed form here is verified against
/// that filter's own ground-truth test (`itkMirrorPadImageTest.cxx`'s
/// `VerifyPixel`, sizes 8 and 12: index `-1` maps to `0`, index `size` maps
/// to `size - 1`, etc).
#[derive(Debug, Clone, Copy, Default)]
pub struct MirrorBoundaryCondition;

impl<T: Scalar> BoundaryCondition<T> for MirrorBoundaryCondition {
    fn get_pixel(&self, index: &[i64], image: &ScalarView<'_, T>) -> T {
        remapped(index, image, |i, size| {
            let period = 2 * size as i64;
            let m = i.rem_euclid(period);
            (if m < size as i64 { m } else { period - 1 - m }) as usize
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image::Image;

    /// Discharges the scalar-and-type proof `get_pixel` now requires.
    fn view<T: Scalar>(img: &Image) -> ScalarView<'_, T> {
        img.scalar_view::<T>().unwrap()
    }

    // 1-D: values equal their index, size 5, so a pinned value doubles as
    // the source index it was clamped/wrapped/defaulted from.
    fn image_1d() -> Image {
        Image::from_vec(&[5], vec![0u32, 1, 2, 3, 4]).unwrap()
    }

    #[test]
    fn zero_flux_neumann_clamps_1d_left_and_right_corners() {
        let img = image_1d();
        let bc = ZeroFluxNeumannBoundaryCondition;
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1], &view(&img)),
            0
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[5], &view(&img)),
            4
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-100], &view(&img)),
            0
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[100], &view(&img)),
            4
        );
    }

    #[test]
    fn constant_returns_constant_only_out_of_bounds_1d() {
        let img = image_1d();
        let bc = ConstantBoundaryCondition::new(99u32);
        assert_eq!(bc.get_pixel(&[-1], &view(&img)), 99);
        assert_eq!(bc.get_pixel(&[5], &view(&img)), 99);
        assert_eq!(bc.get_pixel(&[0], &view(&img)), 0);
        assert_eq!(bc.get_pixel(&[4], &view(&img)), 4);
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
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1], &view(&img)),
            4
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[5], &view(&img)),
            0
        );
        // Multi-wrap: -11 is 4 mod 5 (itkPeriodicBoundaryCondition.hxx:190-195).
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-11], &view(&img)),
            4
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[11], &view(&img)),
            1
        );
    }

    #[test]
    fn mirror_reflects_1d_left_and_right_corners() {
        let img = image_1d();
        let bc = MirrorBoundaryCondition;
        // Edge pixel repeats: index -1 and index 0 both read pixel 0.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1], &view(&img)),
            0
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-2], &view(&img)),
            1
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-5], &view(&img)),
            4
        );
        // Symmetric on the right: index `size` repeats the last pixel.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[5], &view(&img)),
            4
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[6], &view(&img)),
            3
        );
        // Second period (itkMirrorPadImageTest.cxx ground truth: index -9 on
        // an 8-pixel axis reads pixel 7, i.e. a direct, unflipped copy one
        // full period back). Mirrored here at size 5: period = 10.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-6], &view(&img)),
            4
        );
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-10], &view(&img)),
            0
        );
    }

    #[test]
    fn mirror_single_pixel_axis_always_reads_zero() {
        let img = Image::from_vec(&[1], vec![7u32]).unwrap();
        let bc = MirrorBoundaryCondition;
        for i in [-3, -2, -1, 0, 1, 2, 3] {
            assert_eq!(
                BoundaryCondition::<u32>::get_pixel(&bc, &[i], &view(&img)),
                7
            );
        }
    }

    #[test]
    fn mirror_matches_itk_ground_truth_size_8() {
        // itkMirrorPadImageTest.cxx's `VerifyPixel`: an 8-pixel axis, index
        // `row` maps to `rowVal` via reflect-with-edge-repeat, period 16.
        let img = Image::from_vec(&[8], (0..8).collect::<Vec<u32>>()).unwrap();
        let bc = MirrorBoundaryCondition;
        let cases: [(i64, u32); 10] = [
            (-1, 0),
            (-2, 1),
            (-8, 7),
            (-9, 7),
            (-10, 6),
            (-16, 0),
            (0, 0),
            (7, 7),
            (8, 7),
            (15, 0),
        ];
        for (index, expected) in cases {
            assert_eq!(
                BoundaryCondition::<u32>::get_pixel(&bc, &[index], &view(&img)),
                expected,
                "index {index}"
            );
        }
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
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1], &view(&img)),
            0
        );
        // Opposite corner (4, 3) clamps to (3, 2) = 23.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[4, 3], &view(&img)),
            23
        );
        // Top edge (1, -1) clamps y only -> (1, 0) = 1.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1], &view(&img)),
            1
        );
    }

    #[test]
    fn constant_2d_out_of_bounds_if_any_axis_spills() {
        let img = image_2d();
        let bc = ConstantBoundaryCondition::new(99u32);
        assert_eq!(bc.get_pixel(&[-1, -1], &view(&img)), 99);
        assert_eq!(bc.get_pixel(&[1, -1], &view(&img)), 99); // x in-bounds, y not.
        assert_eq!(bc.get_pixel(&[1, 0], &view(&img)), 1); // both in-bounds.
    }

    #[test]
    fn periodic_2d_wraps_each_axis_independently() {
        let img = image_2d();
        let bc = PeriodicBoundaryCondition;
        // (-1, -1) wraps to (3, 2) = 23.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1], &view(&img)),
            23
        );
        // (1, -1) wraps y only -> (1, 2) = 21.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1], &view(&img)),
            21
        );
        // (4, 0) wraps x only -> (0, 0) = 0.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[4, 0], &view(&img)),
            0
        );
    }

    #[test]
    fn mirror_2d_reflects_each_axis_independently() {
        let img = image_2d();
        let bc = MirrorBoundaryCondition;
        // (-1, -1) repeats the (0, 0) corner = 0.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1], &view(&img)),
            0
        );
        // (-1, 0) repeats x only -> (0, 0) = 0.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, 0], &view(&img)),
            0
        );
        // (1, -1) repeats y only -> (1, 0) = 1.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1], &view(&img)),
            1
        );
        // (4, 0) (one past the right edge, w=4) repeats x -> (3, 0) = 3.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[4, 0], &view(&img)),
            3
        );
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
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1, -1], &view(&img)),
            0
        );
        // Opposite corner (3,3,3) clamps to (2,2,2) = 222.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[3, 3, 3], &view(&img)),
            222
        );
        // Edge (1,-1,-1): x in-bounds, y and z clamp -> (1,0,0) = 1.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1, -1], &view(&img)),
            1
        );
    }

    #[test]
    fn constant_3d_out_of_bounds_if_any_axis_spills() {
        let img = image_3d();
        let bc = ConstantBoundaryCondition::new(99u32);
        assert_eq!(bc.get_pixel(&[-1, -1, -1], &view(&img)), 99);
        assert_eq!(bc.get_pixel(&[1, -1, -1], &view(&img)), 99); // only x in-bounds.
        assert_eq!(bc.get_pixel(&[1, 1, -1], &view(&img)), 99); // x, y in-bounds, z not.
        assert_eq!(bc.get_pixel(&[1, 1, 1], &view(&img)), 111); // all in-bounds.
    }

    #[test]
    fn periodic_3d_wraps_each_axis_independently() {
        let img = image_3d();
        let bc = PeriodicBoundaryCondition;
        // (-1,-1,-1) wraps every axis -> (2,2,2) = 222.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1, -1], &view(&img)),
            222
        );
        // (1,-1,-1): x in-bounds, y and z wrap -> (1,2,2) = 221.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1, -1], &view(&img)),
            221
        );
    }

    #[test]
    fn mirror_3d_reflects_each_axis_independently() {
        let img = image_3d();
        let bc = MirrorBoundaryCondition;
        // (-1,-1,-1) repeats the (0,0,0) corner = 0.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[-1, -1, -1], &view(&img)),
            0
        );
        // (1,-1,-1): x in-bounds, y and z repeat -> (1,0,0) = 1.
        assert_eq!(
            BoundaryCondition::<u32>::get_pixel(&bc, &[1, -1, -1], &view(&img)),
            1
        );
    }
}
