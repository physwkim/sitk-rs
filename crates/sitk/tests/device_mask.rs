//! [`DeviceMask`] and the invariant that governs it: **a fixed mask requires a sample
//! set that knows its grid index.**
//!
//! A fixed mask gates a sample by its *grid* index, so the question is only ever whether
//! the sample set still knows what that index is. An explicit fixed-point *list* does not
//! — it is a host-selected subset in an arbitrary order that kept the points and threw
//! the indices away, and a mask indexed into it would gate silently wrong samples. That
//! combination is refused by name at construction ([`CudaError::MaskedExplicitPoints`]),
//! not clamped and not checked in the kernel, and this file is the gate on that.
//!
//! An *index* list ([`FixedPoints::Indices`]) is the sample set that kept the index, so
//! the mask means exactly what it means on the full grid — `fmask[idx[s]]` — and the two
//! compose. The earlier form of this invariant (`has_fmask ⟹ !has_pts`) forbade that
//! combination too, which was right when a selected sample set could only be a point
//! list and is wrong now that it can be an index list.
//!
//! Also here: a mask whose grid is not the fixed grid is refused, and the mask's
//! nonzero-is-inside convention (the host's) holds for whatever pixel type it arrived in.
#![cfg(feature = "cuda")]

use sitk::core::Image;
use sitk::cuda::{
    CudaError, DeviceImage, DeviceMask, FixedPoints, Geometry, MovingGeometry, PointStage,
    ResidentMetric, backend,
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
    let map = [PointStage {
        matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        offset: [0.5, -0.25, 0.75],
    }];

    let unmasked = ResidentMetric::from_device(&d_f, grid(), &d_m, &moving_geometry())
        .unwrap()
        .evaluate(&map)
        .unwrap();
    let masked =
        ResidentMetric::from_device_masked(&d_f, grid(), Some(&ones), &d_m, &moving_geometry())
            .unwrap()
            .evaluate(&map)
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
    let map = [PointStage {
        matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        offset: [0.5, -0.25, 0.75],
    }];
    let moments = |img: &Image| {
        let mask = DeviceMask::upload(img).unwrap();
        ResidentMetric::from_device_masked(&d_f, grid(), Some(&mask), &d_m, &moving_geometry())
            .unwrap()
            .evaluate(&map)
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

// ---------------------------------------------------------------------------
// The primitives a pyramid level needs to build its mask on the device:
// `DeviceImage::filled` (the ones predicate, without a per-level upload),
// `DeviceMask::from_device_image` (re-read a resampled mask as a predicate),
// `DeviceMask::intersect` (the device form of the host's `intersect_masks`), and
// `DeviceMask::to_host` (which exists so the level mask can be pinned byte-wise).
// ---------------------------------------------------------------------------

fn mask_bytes(m: &DeviceMask) -> Vec<u8> {
    m.to_host().unwrap().scalar_slice::<u8>().unwrap().to_vec()
}

/// `filled` puts a constant on an arbitrary grid without a host volume: the voxels
/// come back as the constant and the geometry is the one that was asked for.
#[test]
fn filled_builds_a_constant_volume_on_the_device() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let geom = Geometry {
        size: vec![5, 7, 3],
        spacing: vec![0.8, 0.9, 1.1],
        origin: vec![-3.0, 2.0, 1.0],
        direction: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };
    let img = DeviceImage::filled(geom.clone(), 1.0).unwrap();
    assert_eq!(img.geometry(), &geom);

    let host = img.to_host().unwrap();
    assert_eq!(host.size(), geom.size.as_slice());
    assert_eq!(host.spacing(), geom.spacing.as_slice());
    assert_eq!(host.origin(), geom.origin.as_slice());
    assert_eq!(host.direction(), geom.direction.as_slice());
    assert!(
        host.scalar_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 1.0),
        "a voxel of the ones volume is not one"
    );
}

/// Thresholding on the device is the same predicate as thresholding on the host:
/// `DeviceMask::from_device_image(upload(m))` and `DeviceMask::upload(m)` must agree
/// **byte for byte**, and the mask must actually contain zeros — a threshold that
/// mapped everything to 1 would agree with itself and gate nothing.
#[test]
fn thresholding_on_the_device_is_the_same_predicate_as_on_the_host() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    // Negative, zero, tiny and large values, so `!= 0.0` is exercised on both sides
    // of zero rather than on a 0/1 image where any rule agrees.
    let v: Vec<f32> = (0..n * n * n)
        .map(|i| match i % 5 {
            0 => 0.0,
            1 => -1.0,
            2 => 1e-30,
            3 => 255.0,
            _ => -0.0,
        })
        .collect();
    let img = Image::from_vec(&[n, n, n], v).unwrap();

    let host_side = DeviceMask::upload(&img).unwrap();
    let device_side = DeviceMask::from_device_image(&DeviceImage::upload(&img).unwrap()).unwrap();

    let (a, b) = (mask_bytes(&host_side), mask_bytes(&device_side));
    let zeros = a.iter().filter(|&&x| x == 0).count();
    assert!(
        zeros > 0 && zeros < a.len(),
        "the mask gates nothing: {zeros} zeros of {}",
        a.len()
    );
    assert_eq!(a, b, "the device threshold disagrees with the host's");
    // `-0.0 != 0.0` is false: a signed zero is outside, on both paths.
    assert!(
        a.iter().enumerate().all(|(i, &x)| (i % 5 != 4) || x == 0),
        "a -0.0 voxel was let in"
    );
}

/// `intersect` is `intersect_masks`: `x != 0 && y != 0`, elementwise — and it refuses
/// two masks on different grids rather than gating voxels that are not the same voxels.
#[test]
fn intersect_is_the_elementwise_and_and_refuses_a_grid_mismatch() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let a_img = Image::from_vec(
        &[n, n, n],
        (0..n * n * n)
            .map(|i| f32::from(u8::from(i % 2 == 0)))
            .collect::<Vec<f32>>(),
    )
    .unwrap();
    let b_img = Image::from_vec(
        &[n, n, n],
        (0..n * n * n)
            .map(|i| f32::from(u8::from(i % 3 == 0)))
            .collect::<Vec<f32>>(),
    )
    .unwrap();
    let a = DeviceMask::upload(&a_img).unwrap();
    let b = DeviceMask::upload(&b_img).unwrap();

    let got = mask_bytes(&a.intersect(&b).unwrap());
    let want: Vec<u8> = (0..n * n * n)
        .map(|i| u8::from(i % 2 == 0 && i % 3 == 0))
        .collect();
    assert_eq!(got, want);
    assert!(got.contains(&1), "the intersection is empty");

    // Disjoint masks: every voxel is dropped, and nothing is silently kept.
    let odd = DeviceMask::upload(
        &Image::from_vec(
            &[n, n, n],
            (0..n * n * n)
                .map(|i| f32::from(u8::from(i % 2 == 1)))
                .collect::<Vec<f32>>(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        mask_bytes(&a.intersect(&odd).unwrap())
            .iter()
            .all(|&x| x == 0)
    );

    // A different grid is refused, not intersected by flat index.
    let other =
        DeviceMask::upload(&Image::from_vec(&[8, 32, 16], vec![1.0f32; n * n * n]).unwrap())
            .unwrap();
    match a.intersect(&other) {
        Err(CudaError::DegenerateInput) => {}
        Err(e) => panic!("refused, but by the wrong name: {e}"),
        Ok(_) => panic!("two masks on different grids were intersected"),
    }
}

/// The other half of the invariant: an **index** list keeps the grid index, so a fixed
/// mask composes with it and gates exactly the voxels it gates on the full grid.
///
/// The list here names voxels the mask keeps *and* voxels it drops, so a device that
/// ignored the mask, or that indexed it by position-in-the-list instead of by grid voxel,
/// lands on a different count. Checked against the count computed on the host from the
/// same two objects — not against the device's own answer in another configuration.
#[test]
fn a_fixed_mask_composes_with_an_index_list_and_gates_by_grid_index() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let (fixed, moving) = (volume(n), volume(n));
    let keep = checker(n); // voxel v survives iff v % 3 == 0
    let mask = DeviceMask::upload(&keep).unwrap();
    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();

    // Every 5th voxel: a list that straddles the mask, since 5 and 3 are coprime.
    let voxels = n * n * n;
    let idx: Vec<i64> = (0..voxels).step_by(5).map(|v| v as i64).collect();
    let expected = idx.iter().filter(|&&v| v % 3 == 0).count();
    assert!(
        expected > 0 && expected < idx.len(),
        "the mask must drop some of the list and keep some of it, or this proves nothing"
    );

    let points = FixedPoints::Indices {
        idx: &idx,
        size: &SIZE,
        origin: &ORIGIN,
        idx_to_phys: &EYE,
    };
    let m = ResidentMetric::from_device_masked(&d_f, points, Some(&mask), &d_m, &moving_geometry())
        .expect("a fixed mask and an index list must compose")
        .evaluate(&[PointStage {
            matrix: EYE,
            offset: [0.0, 0.0, 0.0],
        }])
        .unwrap();

    // The identity map keeps every sample inside the moving image, so the only thing
    // that can reduce the count is the mask.
    assert_eq!(
        m.count,
        expected,
        "the mask gated {} of {} listed samples; by grid index it should gate {expected}",
        m.count,
        idx.len()
    );
}
