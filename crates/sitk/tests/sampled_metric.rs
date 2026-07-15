//! [`FixedPoints::Indices`]: a sample set that is a list of fixed-grid voxels.
//!
//! This is the form a sampling strategy uses, and the reason it is an *index* list and
//! not a *point* list is what this file pins. An index says which voxel; the kernel then
//! derives that voxel's physical point from the same closed form the full-grid path uses
//! and reads that voxel's value from the resident image. So a sampled run does not
//! approximate a full run at the voxels it kept — it *is* the full run at those voxels,
//! bit for bit. `the_identity_index_list_is_the_grid_bit_for_bit` is that claim, stated
//! in the strongest form available: hand the kernel `[0, 1, ..., n-1]` and it must return
//! the full-grid moments with every bit equal.
//!
//! The other property an index carries is that it can be *masked* — a fixed mask gates by
//! grid index, and an index list is precisely the sample set that kept its grid index.
//! That half lives in `device_mask.rs`, with the invariant it belongs to.
#![cfg(feature = "cuda")]

use sitk::core::Image;
use sitk::cuda::{
    CudaError, DeviceImage, FixedPoints, Moments, MovingGeometry, PointStage, ResidentMetric,
    backend,
};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

const N: usize = 16;
const VOXELS: usize = N * N * N;
const SIZE: [usize; 3] = [N, N, N];
const STRIDES: [usize; 3] = [1, N, N * N];
const ORIGIN: [f64; 3] = [-2.0, 1.0, 0.5];
/// Not the identity: an anisotropic, sheared index-to-physical map, so a sample whose
/// point were derived from the wrong voxel lands somewhere the moments notice.
const IDX_TO_PHYS: [f64; 9] = [1.1, 0.2, 0.0, 0.0, 0.9, 0.1, 0.05, 0.0, 1.3];
const PHYS_TO_INDEX: [f64; 9] = [0.9, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.8];

/// A volume with no symmetry a wrong index could hide behind.
fn volume(seed: u64) -> Image {
    let v: Vec<f32> = (0..VOXELS)
        .map(|i| {
            let x = (i as u64).wrapping_mul(2654435761).wrapping_add(seed);
            ((x >> 11) % 1000) as f32 / 7.0
        })
        .collect();
    Image::from_vec(&SIZE, v).unwrap()
}

fn moving_geometry() -> MovingGeometry<'static> {
    MovingGeometry {
        len: VOXELS,
        size: &SIZE,
        strides: &STRIDES,
        origin: &ORIGIN,
        phys_to_index: &PHYS_TO_INDEX,
        mask: None,
    }
}

fn grid_points() -> FixedPoints<'static> {
    FixedPoints::Grid {
        size: &SIZE,
        origin: &ORIGIN,
        idx_to_phys: &IDX_TO_PHYS,
    }
}

fn index_points(idx: &[i64]) -> FixedPoints<'_> {
    FixedPoints::Indices {
        idx,
        size: &SIZE,
        origin: &ORIGIN,
        idx_to_phys: &IDX_TO_PHYS,
    }
}

/// A point map that actually moves the samples — a rotation-ish `A` with an offset, so
/// the moments are not a degenerate sum of zeros.
const A: [f64; 9] = [0.98, -0.15, 0.03, 0.14, 0.97, -0.06, -0.02, 0.05, 0.99];
const B: [f64; 3] = [0.7, -0.4, 0.25];

/// The point map as the metric now takes it: one stage of `mat_vec(matrix, p) + offset`,
/// which is what a single matrix-offset transform hands over.
const MAP: [PointStage; 1] = [PointStage {
    matrix: A,
    offset: B,
}];

fn moments(fixed_points: FixedPoints<'_>) -> Moments {
    let (f, m) = (volume(1), volume(9));
    let (d_f, d_m) = (
        DeviceImage::upload(&f).unwrap(),
        DeviceImage::upload(&m).unwrap(),
    );
    ResidentMetric::from_device(&d_f, fixed_points, &d_m, &moving_geometry())
        .unwrap()
        .evaluate(&MAP)
        .unwrap()
}

fn assert_same_moments(a: &Moments, b: &Moments, what: &str) {
    assert_eq!(a.count, b.count, "{what}: count");
    assert_eq!(
        a.sq.to_bits(),
        b.sq.to_bits(),
        "{what}: sq {} vs {}",
        a.sq,
        b.sq
    );
    for d in 0..3 {
        assert_eq!(
            a.s0[d].to_bits(),
            b.s0[d].to_bits(),
            "{what}: s0[{d}] {} vs {}",
            a.s0[d],
            b.s0[d]
        );
        for e in 0..3 {
            assert_eq!(
                a.s1[d][e].to_bits(),
                b.s1[d][e].to_bits(),
                "{what}: s1[{d}][{e}] {} vs {}",
                a.s1[d][e],
                b.s1[d][e]
            );
        }
    }
}

/// **The pin.** Sample `s` is voxel `s` — the identity selection — so the index list
/// names exactly the voxels the grid path walks, in exactly the order it walks them.
/// Every sample therefore lands on the same thread, derives its point from the same
/// expression, reads the same value, and is summed into the same slot of the same
/// reduction tree. The moments must be **bit-identical**, not close: nothing about the
/// arithmetic has changed, only how the kernel was told which voxel to visit.
///
/// This is what makes an index list a *sampling* mechanism rather than a second
/// implementation of the metric. If it held only to a tolerance, a sampled run would be
/// a different computation that happens to agree, and every downstream count-equality
/// pin would be resting on that tolerance.
#[test]
fn the_identity_index_list_is_the_grid_bit_for_bit() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let idx: Vec<i64> = (0..VOXELS as i64).collect();
    let grid = moments(grid_points());
    let indexed = moments(index_points(&idx));

    assert!(
        grid.count > VOXELS / 2,
        "the point map drops too many samples ({} of {VOXELS}) for this to prove much",
        grid.count
    );
    assert_same_moments(&grid, &indexed, "identity index list vs grid");
}

/// The pin above is live: perturb a single index and the moments move. Without this,
/// a kernel that ignored `fidx` entirely — always deriving from `s` — would pass
/// `the_identity_index_list_is_the_grid_bit_for_bit` and every sampled run would
/// silently evaluate the full grid.
#[test]
fn one_wrong_index_moves_the_moments() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let mut idx: Vec<i64> = (0..VOXELS as i64).collect();
    let grid = moments(grid_points());
    idx[VOXELS / 3] = ((VOXELS / 3) + 1) as i64; // one voxel visited twice, one never

    let perturbed = moments(index_points(&idx));
    assert_ne!(
        grid.sq.to_bits(),
        perturbed.sq.to_bits(),
        "changing one index changed nothing: the kernel is not reading the index list"
    );
}

/// A sampled set evaluates the voxels it names and no others. Compared against the same
/// voxels reached the other way — the full grid gated by a mask that keeps exactly them
/// — the **count** must be exactly equal (it is the same set) while the sums agree only
/// to reduction-rounding: the two runs assign those voxels to different threads, so they
/// sum them in a different order. That is the honest pin here, and the bit-identity above
/// is what tells us the difference is the order and nothing else.
#[test]
fn a_sampled_index_list_evaluates_the_voxels_it_names() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let keep = |v: usize| v % 7 == 3;
    let idx: Vec<i64> = (0..VOXELS).filter(|&v| keep(v)).map(|v| v as i64).collect();
    assert!(idx.len() > 100, "too few samples to prove anything");

    let sampled = moments(index_points(&idx));

    // The same voxels, reached through the mask instead of through the list.
    let mask_img = Image::from_vec(
        &SIZE,
        (0..VOXELS)
            .map(|v| f32::from(u8::from(keep(v))))
            .collect::<Vec<f32>>(),
    )
    .unwrap();
    let mask = sitk::cuda::DeviceMask::upload(&mask_img).unwrap();
    let (f, m) = (volume(1), volume(9));
    let (d_f, d_m) = (
        DeviceImage::upload(&f).unwrap(),
        DeviceImage::upload(&m).unwrap(),
    );
    let masked = ResidentMetric::from_device_masked(
        &d_f,
        grid_points(),
        Some(&mask),
        &d_m,
        &moving_geometry(),
    )
    .unwrap()
    .evaluate(&MAP)
    .unwrap();

    assert_eq!(
        sampled.count, masked.count,
        "the index list and the mask disagree about which voxels are in the set"
    );
    let rel = (sampled.sq - masked.sq).abs() / masked.sq.abs();
    assert!(
        rel <= 1e-12,
        "same voxels, different sums: sq {} (list) vs {} (mask), rel {rel:e} — more than \
         the reduction order can explain",
        sampled.sq,
        masked.sq
    );
}

/// `Random` draws **with replacement**, so a voxel can appear twice in the list and must
/// then be counted twice — the host counts it twice, and a device that silently deduped
/// would compute a different metric from the same sample set.
#[test]
fn a_repeated_index_is_a_repeated_sample() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let once = moments(index_points(&[1234i64]));
    let twice = moments(index_points(&[1234i64, 1234]));

    assert_eq!(once.count, 1, "the single sample did not land inside");
    assert_eq!(twice.count, 2, "a repeated index was deduplicated");
    // Two identical doubles sum exactly, so this is an equality and not a band.
    assert_eq!(
        twice.sq.to_bits(),
        (2.0 * once.sq).to_bits(),
        "the repeated sample did not contribute twice"
    );
}

/// An index that does not name a voxel of the fixed grid is refused **by name**, on the
/// host, before any launch. The kernel would otherwise read outside the volume — and
/// clamping it would be worse than reading garbage, because it would silently sample the
/// wrong voxel and produce a plausible number.
#[test]
fn an_index_outside_the_grid_is_refused_by_name() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let (f, m) = (volume(1), volume(9));
    let (d_f, d_m) = (
        DeviceImage::upload(&f).unwrap(),
        DeviceImage::upload(&m).unwrap(),
    );

    for bad in [VOXELS as i64, VOXELS as i64 + 1000, -1] {
        let idx = [0i64, 5, bad, 7];
        match ResidentMetric::from_device(&d_f, index_points(&idx), &d_m, &moving_geometry()) {
            Err(CudaError::SampleIndexOutOfGrid { index, voxels }) => {
                assert_eq!(index, bad);
                assert_eq!(voxels, VOXELS);
            }
            Err(e) => panic!("index {bad} refused, but by the wrong name: {e}"),
            Ok(_) => panic!("index {bad} was accepted into a {VOXELS}-voxel grid"),
        }
    }

    // The last voxel is in the grid, and is accepted — so the check above is a bound,
    // not an off-by-one that rejects the top of the range.
    assert!(
        ResidentMetric::from_device(
            &d_f,
            index_points(&[VOXELS as i64 - 1]),
            &d_m,
            &moving_geometry()
        )
        .is_ok(),
        "the last voxel of the grid was refused"
    );
}
