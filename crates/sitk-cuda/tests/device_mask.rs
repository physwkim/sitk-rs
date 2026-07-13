//! [`DeviceMask`] and the invariant that governs it: **`has_fmask ⟹ !has_pts`**.
//!
//! A fixed mask gates a sample by its *grid* index. An explicit fixed-point list is a
//! host-selected subset in an arbitrary order, where the same index refers to a
//! different voxel — so a mask indexed into it would gate silently wrong samples. The
//! combination is refused by name at construction ([`CudaError::MaskedExplicitPoints`]),
//! not clamped and not checked in the kernel, and this file is the gate on that.
//!
//! Also here: a mask whose grid is not the fixed grid is refused, and the mask's
//! nonzero-is-inside convention (the host's) holds for whatever pixel type it arrived in.
#![cfg(feature = "cuda")]

use sitk_core::Image;
use sitk_cuda::{
    CudaError, DeviceImage, DeviceMask, FixedPoints, MovingGeometry, ResidentMetric, backend,
};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

fn volume(n: usize) -> Image {
    let v: Vec<f32> = (0..n * n * n).map(|i| (i % 97) as f32).collect();
    Image::from_vec(&[n, n, n], v).unwrap()
}

/// A mask on the same grid keeping every third voxel: it drops two thirds of the samples,
/// in a pattern no stride bug reproduces by accident.
fn checker(n: usize) -> Image {
    let v: Vec<f32> = (0..n * n * n)
        .map(|i| ((i % 3 == 0) as u8) as f32)
        .collect();
    Image::from_vec(&[n, n, n], v).unwrap()
}

const SIZE: [usize; 3] = [16, 16, 16];
const STRIDES: [usize; 3] = [1, 16, 256];
const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];
const EYE: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn moving_geometry() -> MovingGeometry<'static> {
    MovingGeometry {
        len: 16 * 16 * 16,
        size: &SIZE,
        strides: &STRIDES,
        origin: &ORIGIN,
        phys_to_index: &EYE,
        mask: None,
    }
}

/// The invariant, asserted at construction: a fixed mask plus an explicit point list is
/// [`CudaError::MaskedExplicitPoints`] — refused by name, with no kernel launched.
#[test]
fn a_fixed_mask_with_an_explicit_point_list_is_refused_by_name() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let (fixed, moving) = (volume(n), volume(n));
    let mask = DeviceMask::upload(&checker(n)).unwrap();
    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();

    // One physical point per fixed sample — a legal explicit list on its own.
    let pts: Vec<f64> = (0..n * n * n)
        .flat_map(|s| {
            let (i, j, k) = (s % n, (s / n) % n, s / (n * n));
            [i as f64, j as f64, k as f64]
        })
        .collect();

    // Without the mask the same list builds fine, so the refusal below is about the
    // *combination* and not about the list.
    assert!(
        ResidentMetric::from_device(&d_f, FixedPoints::Explicit(&pts), &d_m, &moving_geometry())
            .is_ok(),
        "the explicit point list is itself valid"
    );

    match ResidentMetric::from_device_masked(
        &d_f,
        FixedPoints::Explicit(&pts),
        Some(&mask),
        &d_m,
        &moving_geometry(),
    ) {
        Err(CudaError::MaskedExplicitPoints) => {}
        Err(e) => panic!("refused, but by the wrong name: {e}"),
        Ok(_) => panic!("a fixed mask was combined with an explicit point list"),
    }
}

/// The mask is indexed by the fixed grid's flat index, so it must *be* that grid. A mask
/// with the right voxel count on the wrong shape indexes different voxels, and is refused.
#[test]
fn a_mask_on_a_different_grid_shape_is_refused() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let d_f = DeviceImage::upload(&volume(n)).unwrap();
    let d_m = DeviceImage::upload(&volume(n)).unwrap();

    // 8 × 32 × 16 = 4096 = 16³: same count, different grid.
    let reshaped = Image::from_vec(&[8, 32, 16], vec![1.0f32; n * n * n]).unwrap();
    let mask = DeviceMask::upload(&reshaped).unwrap();
    assert_eq!(
        mask.len(),
        n * n * n,
        "the count matches; only the shape differs"
    );

    match ResidentMetric::from_device_masked(
        &d_f,
        FixedPoints::Grid {
            size: &SIZE,
            origin: &ORIGIN,
            idx_to_phys: &EYE,
        },
        Some(&mask),
        &d_m,
        &moving_geometry(),
    ) {
        Err(CudaError::DegenerateInput) => {}
        Err(e) => panic!("refused, but by the wrong name: {e}"),
        Ok(_) => panic!("a mask on another grid shape built a metric"),
    }
}

/// An all-ones mask is the identity: it must give **bit-identical** moments to no mask at
/// all. This is what pins "the mask does not reorder the reduction" — same terms, same
/// order, same bits, with the masked branch compiled in and taken on every sample.
#[test]
fn an_all_ones_mask_is_bit_identical_to_no_mask() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let d_f = DeviceImage::upload(&volume(n)).unwrap();
    let d_m = DeviceImage::upload(&volume(n)).unwrap();
    let ones =
        DeviceMask::upload(&Image::from_vec(&[n, n, n], vec![1.0f32; n * n * n]).unwrap()).unwrap();
    let grid = || FixedPoints::Grid {
        size: &SIZE,
        origin: &ORIGIN,
        idx_to_phys: &EYE,
    };
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = [0.5, -0.25, 0.75];

    let unmasked = ResidentMetric::from_device(&d_f, grid(), &d_m, &moving_geometry())
        .unwrap()
        .evaluate(&a, &b)
        .unwrap();
    let masked =
        ResidentMetric::from_device_masked(&d_f, grid(), Some(&ones), &d_m, &moving_geometry())
            .unwrap()
            .evaluate(&a, &b)
            .unwrap();

    assert_eq!(masked.count, unmasked.count);
    assert_eq!(
        masked.sq.to_bits(),
        unmasked.sq.to_bits(),
        "an all-ones mask changed the reduction's result"
    );
}

/// The host's convention — **any nonzero voxel is inside** — holds for a mask that
/// arrived as `UInt8` 0/255 as much as for one that arrived as `f32` 0.0/1.0: the two
/// gate the same voxels and produce the same moments, bit for bit.
#[test]
fn any_nonzero_voxel_is_inside_whatever_the_pixel_type() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let d_f = DeviceImage::upload(&volume(n)).unwrap();
    let d_m = DeviceImage::upload(&volume(n)).unwrap();

    let f32_mask = checker(n);
    let u8_mask = Image::from_vec(
        &[n, n, n],
        f32_mask
            .scalar_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&v| if v != 0.0 { 255u8 } else { 0u8 })
            .collect::<Vec<u8>>(),
    )
    .unwrap();

    let grid = || FixedPoints::Grid {
        size: &SIZE,
        origin: &ORIGIN,
        idx_to_phys: &EYE,
    };
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = [0.5, -0.25, 0.75];
    let moments = |img: &Image| {
        let mask = DeviceMask::upload(img).unwrap();
        ResidentMetric::from_device_masked(&d_f, grid(), Some(&mask), &d_m, &moving_geometry())
            .unwrap()
            .evaluate(&a, &b)
            .unwrap()
    };

    let from_f32 = moments(&f32_mask);
    let from_u8 = moments(&u8_mask);

    assert!(
        from_f32.count > 0 && from_f32.count < n * n * n,
        "the checker mask must drop some samples and keep some; kept {}",
        from_f32.count
    );
    assert_eq!(from_u8.count, from_f32.count);
    assert_eq!(from_u8.sq.to_bits(), from_f32.sq.to_bits());
}
