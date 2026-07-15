//! The correlation metric's two passes: 3 moments, then 28.
//!
//! What is pinned here is the **reduction**, not the metric — the value and the
//! derivative are the host's business and are pinned against the host in
//! `sitk-registration`. Here the questions are narrower and sharper:
//!
//! * Are the 31 slots the sums they claim to be? (`the_moments_are_the_sums_they_claim`,
//!   against a reference recomputed in this file, so a slot wired to the wrong index or
//!   a term dropped from the loop cannot pass.)
//! * Is the reduction deterministic? (`the_moments_are_bit_identical_run_to_run`.)
//! * Does an index list still *mean* the grid it names, now that a second metric reads
//!   it? (`the_identity_index_list_is_the_grid_bit_for_bit`, the same claim
//!   `sampled_metric.rs` makes for mean squares — restated because the sample set is
//!   shared code and a shared invariant that is only tested through one caller is tested
//!   by luck.)
//! * Do the two passes see the same samples? (`both_passes_count_the_same_samples` — if
//!   they ever did not, the means would be divided by one population and the moments
//!   accumulated over another.)
//!
//! Each has a falsifier next to it: move one index, flip one mask byte, and the pin must
//! break.
#![cfg(feature = "cuda")]

use sitk::core::Image;
use sitk::cuda::{
    CorrelationMoments, CudaError, DeviceImage, DeviceMask, FixedPoints, MovingGeometry,
    PointStage, ResidentCorrelation, backend,
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

/// A point map that moves the samples without throwing most of them out.
const A: [f64; 9] = [0.98, -0.15, 0.03, 0.14, 0.97, -0.06, -0.02, 0.05, 0.99];
const B: [f64; 3] = [0.7, -0.4, 0.25];

/// One stage of `mat_vec(matrix, p) + offset` — the point map as the metric takes it.
fn one_stage(a: &[f64; 9], b: &[f64; 3]) -> [PointStage; 1] {
    [PointStage {
        matrix: *a,
        offset: *b,
    }]
}

/// A volume with a **pedestal**, on purpose: NCC subtracts the mean, so a zero-mean
/// volume would let a kernel that forgot to subtract it pass anyway.
fn volume(seed: u64) -> Image {
    let v: Vec<f32> = (0..VOXELS)
        .map(|i| {
            let x = (i as u64).wrapping_mul(2654435761).wrapping_add(seed);
            500.0 + ((x >> 11) % 1000) as f32 / 7.0
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

fn moments_with(
    fixed_points: FixedPoints<'_>,
    mask: Option<&DeviceMask>,
    a: &[f64; 9],
    b: &[f64; 3],
) -> CorrelationMoments {
    let (f, m) = (volume(1), volume(9));
    let (d_f, d_m) = (
        DeviceImage::upload(&f).unwrap(),
        DeviceImage::upload(&m).unwrap(),
    );
    ResidentCorrelation::from_device_masked(&d_f, fixed_points, mask, &d_m, &moving_geometry())
        .unwrap()
        .evaluate(&one_stage(a, b))
        .unwrap()
}

fn moments(fixed_points: FixedPoints<'_>) -> CorrelationMoments {
    moments_with(fixed_points, None, &A, &B)
}

fn assert_same(a: &CorrelationMoments, b: &CorrelationMoments, what: &str) {
    assert_eq!(a.count, b.count, "{what}: count");
    for (name, x, y) in [
        ("mean_fixed", a.mean_fixed, b.mean_fixed),
        ("mean_moving", a.mean_moving, b.mean_moving),
        ("sff", a.sff, b.sff),
        ("smm", a.smm, b.smm),
        ("sfm", a.sfm, b.sfm),
    ] {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: {name} {x} vs {y}");
    }
    for d in 0..3 {
        assert_eq!(
            a.f0[d].to_bits(),
            b.f0[d].to_bits(),
            "{what}: f0[{d}] {} vs {}",
            a.f0[d],
            b.f0[d]
        );
        assert_eq!(
            a.m0[d].to_bits(),
            b.m0[d].to_bits(),
            "{what}: m0[{d}] {} vs {}",
            a.m0[d],
            b.m0[d]
        );
        for e in 0..3 {
            assert_eq!(
                a.f1[d][e].to_bits(),
                b.f1[d][e].to_bits(),
                "{what}: f1[{d}][{e}] {} vs {}",
                a.f1[d][e],
                b.f1[d][e]
            );
            assert_eq!(
                a.m1[d][e].to_bits(),
                b.m1[d][e].to_bits(),
                "{what}: m1[{d}][{e}] {} vs {}",
                a.m1[d][e],
                b.m1[d][e]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// A reference implementation of the sampler, on the host, in this file.
//
// Deliberately *not* shared with the kernel: a reference that shares code with the
// thing it checks proves only that the code is self-consistent. This one is written
// from the formulas, and it is what says the 28 slots are the sums the module doc
// says they are — that `f1[d][e]` really is `Σ f1·∇M[d]·x[e]` and not, say,
// `Σ f1·∇M[e]·x[d]`, which a bit-identity or a determinism pin would never catch.
// ---------------------------------------------------------------------------

fn point_of(voxel: usize) -> [f64; 3] {
    let (i, j, k) = (
        (voxel % N) as f64,
        ((voxel / N) % N) as f64,
        (voxel / (N * N)) as f64,
    );
    let mut x = [0.0f64; 3];
    for (r, xr) in x.iter_mut().enumerate() {
        *xr = ORIGIN[r]
            + IDX_TO_PHYS[r * 3] * i
            + IDX_TO_PHYS[r * 3 + 1] * j
            + IDX_TO_PHYS[r * 3 + 2] * k;
    }
    x
}

/// Trilinear value and physical gradient, or `None` outside — `linear_value_and_gradient`
/// plus `is_inside`, written out.
fn sample_moving(vals: &[f64], p: &[f64; 3]) -> Option<(f64, [f64; 3])> {
    let mut c = [0.0f64; 3];
    for (r, cr) in c.iter_mut().enumerate() {
        *cr = (0..3)
            .map(|j| PHYS_TO_INDEX[r * 3 + j] * (p[j] - ORIGIN[j]))
            .sum();
    }
    for (d, &cd) in c.iter().enumerate() {
        if !(cd >= -0.5 && cd < SIZE[d] as f64 - 0.5) {
            return None;
        }
    }

    let mut base = [0.0f64; 3];
    let mut frac = [0.0f64; 3];
    for d in 0..3 {
        base[d] = c[d].floor();
        frac[d] = c[d] - base[d];
    }
    let mut value = 0.0f64;
    let mut gi = [0.0f64; 3];
    for corner in 0..8usize {
        let mut offset = 0usize;
        let mut weight = 1.0f64;
        for (d, (&fr, &ba)) in frac.iter().zip(&base).enumerate() {
            let bit = (corner >> d) & 1;
            weight *= if bit == 1 { fr } else { 1.0 - fr };
            let idx = (ba as isize + bit as isize).clamp(0, SIZE[d] as isize - 1) as usize;
            offset += idx * STRIDES[d];
        }
        let b = vals[offset];
        value += weight * b;
        for (j, gij) in gi.iter_mut().enumerate() {
            let mut w = 1.0f64;
            for (d, &fr) in frac.iter().enumerate() {
                if d == j {
                    continue;
                }
                let bit = (corner >> d) & 1;
                w *= if bit == 1 { fr } else { 1.0 - fr };
            }
            let sign = if (corner >> j) & 1 == 1 { 1.0 } else { -1.0 };
            *gij += sign * w * b;
        }
    }
    let mut g = [0.0f64; 3];
    for (d, gd) in g.iter_mut().enumerate() {
        *gd = (0..3).map(|j| gi[j] * PHYS_TO_INDEX[j * 3 + d]).sum();
    }
    Some((value, g))
}

/// The whole metric's moments, recomputed here from the formulas.
fn reference(voxels: &[usize]) -> CorrelationMoments {
    let f = volume(1).to_f64_vec().unwrap();
    let m = volume(9).to_f64_vec().unwrap();

    // Pass 1.
    let (mut sum_f, mut sum_m, mut count) = (0.0f64, 0.0f64, 0usize);
    let mut valid = Vec::new();
    for &v in voxels {
        let x = point_of(v);
        let mut p = [0.0f64; 3];
        for (d, pd) in p.iter_mut().enumerate() {
            *pd = A[d * 3] * x[0] + A[d * 3 + 1] * x[1] + A[d * 3 + 2] * x[2] + B[d];
        }
        let Some((mv, g)) = sample_moving(&m, &p) else {
            continue;
        };
        sum_f += f[v];
        sum_m += mv;
        count += 1;
        valid.push((x, f[v], mv, g));
    }
    assert!(count > 100, "reference has too few valid samples: {count}");

    let mean_fixed = sum_f / count as f64;
    let mean_moving = sum_m / count as f64;

    // Pass 2.
    let mut r = CorrelationMoments {
        count,
        mean_fixed,
        mean_moving,
        ..Default::default()
    };
    for (x, fv, mv, g) in valid {
        let f1 = fv - mean_fixed;
        let m1 = mv - mean_moving;
        r.sff += f1 * f1;
        r.smm += m1 * m1;
        r.sfm += f1 * m1;
        for (d, &gd) in g.iter().enumerate() {
            r.f0[d] += f1 * gd;
            r.m0[d] += m1 * gd;
            for (e, &xe) in x.iter().enumerate() {
                r.f1[d][e] += f1 * gd * xe;
                r.m1[d][e] += m1 * gd * xe;
            }
        }
    }
    r
}

/// `|a − b| / max(|b|, floor)` — the moments have wildly different magnitudes, so a
/// bare absolute tolerance would be meaningless for some and vacuous for others.
fn rel(a: f64, b: f64, floor: f64) -> f64 {
    (a - b).abs() / b.abs().max(floor)
}

/// **The slots are the sums they claim to be.** Every one of the 28 moments, plus the
/// two means and the count, against a reference written from the formulas rather than
/// shared with the kernel.
///
/// The count is **exact** — an integer over the same predicate. The sums are not, and
/// cannot be: the device folds a 512-block tree where the reference sums left to right,
/// and no parallel reduction reproduces that. The tolerance is reduction rounding at
/// this sample count, `1e-12` relative, and it is ~2 orders above what was measured
/// (printed below) rather than a number chosen to make the test pass.
#[test]
fn the_moments_are_the_sums_they_claim() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let all: Vec<usize> = (0..VOXELS).collect();
    let want = reference(&all);
    let got = moments(grid_points());

    assert_eq!(got.count, want.count, "count must be exact");

    let scale = want.sff.max(want.smm).max(1.0);
    let mut worst: (f64, &'static str) = (0.0, "none");
    let mut check = |name: &'static str, a: f64, b: f64, floor: f64| {
        let e = rel(a, b, floor);
        if e > worst.0 {
            worst = (e, name);
        }
        assert!(
            e <= 1e-12,
            "{name}: device {a} vs reference {b}, rel {e:.3e}"
        );
    };

    check("mean_fixed", got.mean_fixed, want.mean_fixed, 1.0);
    check("mean_moving", got.mean_moving, want.mean_moving, 1.0);
    check("sff", got.sff, want.sff, scale);
    check("smm", got.smm, want.smm, scale);
    check("sfm", got.sfm, want.sfm, scale);
    for d in 0..3 {
        let gscale = want
            .f0
            .iter()
            .chain(&want.m0)
            .fold(0.0f64, |m, v| m.max(v.abs()));
        check("f0", got.f0[d], want.f0[d], gscale);
        check("m0", got.m0[d], want.m0[d], gscale);
        for e in 0..3 {
            let m1scale = want
                .f1
                .iter()
                .flatten()
                .chain(want.m1.iter().flatten())
                .fold(0.0f64, |m, v| m.max(v.abs()));
            check("f1", got.f1[d][e], want.f1[d][e], m1scale);
            check("m1", got.m1[d][e], want.m1[d][e], m1scale);
        }
    }
    eprintln!(
        "N1 device vs host-reference moments: worst {:.3e} on {} ({} samples)",
        worst.0, worst.1, got.count
    );
}

/// The reduction is deterministic. Ten evaluations, every bit equal — the property the
/// whole no-atomics design exists for, since the optimizer is a feedback loop and a
/// metric that varies run to run makes the registration *result* vary run to run.
#[test]
fn the_moments_are_bit_identical_run_to_run() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let first = moments(grid_points());
    for i in 1..10 {
        let again = moments(grid_points());
        assert_same(&first, &again, &format!("evaluation {i}"));
    }
}

/// The identity index list is the grid, **bit for bit** — the same claim
/// `sampled_metric.rs` makes for mean squares, restated for this metric because the
/// sample set is shared code and an invariant tested through one caller only is tested
/// by luck.
#[test]
fn the_identity_index_list_is_the_grid_bit_for_bit() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let idx: Vec<i64> = (0..VOXELS as i64).collect();
    let grid = moments(grid_points());
    let indexed = moments(index_points(&idx));

    assert!(
        grid.count > VOXELS / 2 && grid.sff > 0.0,
        "the setup evaluates nothing: count {} sff {}",
        grid.count,
        grid.sff
    );
    assert_same(&grid, &indexed, "identity index list vs grid");
}

/// The falsifier for the pin above: move **one** index and the moments must move. Without
/// this, a kernel that ignored `fidx` entirely would pass every test in this file.
#[test]
fn one_wrong_index_moves_the_moments() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let idx: Vec<i64> = (0..VOXELS as i64).collect();
    let mut wrong = idx.clone();
    wrong[VOXELS / 3] = ((VOXELS / 3) + 1) as i64;

    let right = moments(index_points(&idx));
    let moved = moments(index_points(&wrong));
    assert_ne!(
        right.sfm.to_bits(),
        moved.sfm.to_bits(),
        "one wrong index left sfm unchanged: the index list is not being read"
    );
}

/// Both passes count the same samples. They must: one sampler, one predicate, one point
/// map. If they ever disagreed, the means would be divided by one population while the
/// moments were accumulated over another — and the metric would be quietly wrong rather
/// than loudly broken, which is why `evaluate` raises
/// [`CudaError::PassCountMismatch`] instead of trusting it.
///
/// The transform here throws roughly half the samples out, so the count is a real
/// intersection rather than "everything, trivially".
#[test]
fn both_passes_count_the_same_samples() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    // A large offset: much of the fixed grid now maps outside the moving image.
    let b = [9.0, -7.5, 6.0];
    let got = moments_with(grid_points(), None, &A, &b);

    let all: Vec<usize> = (0..VOXELS).collect();
    let want = reference(&all);
    assert!(
        got.count > 0 && got.count < want.count,
        "the offset did not partially exit the moving image: {} of {}",
        got.count,
        want.count
    );
    // `evaluate` would have returned PassCountMismatch; reaching here means the two
    // passes agreed, and the count is the one the host predicate produces.
    let moving_vals = volume(9).to_f64_vec().unwrap();
    let mut expected = 0usize;
    for &v in &all {
        let x = point_of(v);
        let mut p = [0.0f64; 3];
        for (d, pd) in p.iter_mut().enumerate() {
            *pd = A[d * 3] * x[0] + A[d * 3 + 1] * x[1] + A[d * 3 + 2] * x[2] + b[d];
        }
        if sample_moving(&moving_vals, &p).is_some() {
            expected += 1;
        }
    }
    assert_eq!(got.count, expected, "valid count must be exact");
}

/// A fixed mask gates by **grid voxel**, in this metric as in the other one — the
/// invariant lives in `Resident`, and this is the correlation metric asking it the same
/// question. Masking every voxel of one z-slab must drop exactly that slab's samples,
/// and the surviving moments must be the moments of an index list naming the survivors.
#[test]
fn a_fixed_mask_gates_this_metric_by_grid_index_too() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    // Keep the first half of the z axis.
    let keep: Vec<bool> = (0..VOXELS).map(|v| v / (N * N) < N / 2).collect();
    let mask_img = Image::from_vec(
        &SIZE,
        keep.iter().map(|&b| u8::from(b)).collect::<Vec<u8>>(),
    )
    .unwrap();
    let mask = DeviceMask::upload(&mask_img).unwrap();

    let masked = moments_with(grid_points(), Some(&mask), &A, &B);

    // The same voxels, named explicitly instead of masked.
    let idx: Vec<i64> = (0..VOXELS as i64).filter(|&v| keep[v as usize]).collect();
    let listed = moments_with(index_points(&idx), None, &A, &B);

    assert!(masked.count > 0, "the mask dropped everything");
    assert_eq!(
        masked.count, listed.count,
        "the mask and the equivalent index list disagree on the sample count"
    );
    // Both walk the same voxels, but not in the same thread order (the mask path skips
    // inside a full grid stride; the list path is dense), so the reduction differs and
    // this is a tolerance, not a bit-identity.
    let e = rel(masked.sfm, listed.sfm, listed.sff.max(1.0));
    assert!(
        e <= 1e-12,
        "masked sfm {} vs listed sfm {}, rel {e:.3e}",
        masked.sfm,
        listed.sfm
    );
}

/// The falsifier for the mask: flip **one** byte and the count must move by exactly one.
#[test]
fn one_flipped_mask_byte_drops_exactly_one_sample() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let all_ones: Vec<u8> = vec![1; VOXELS];
    let mut one_off = all_ones.clone();
    // A voxel near the middle of the grid, which the transform keeps inside.
    one_off[VOXELS / 2 + N / 2] = 0;

    let full = DeviceMask::upload(&Image::from_vec(&SIZE, all_ones).unwrap()).unwrap();
    let holed = DeviceMask::upload(&Image::from_vec(&SIZE, one_off).unwrap()).unwrap();

    let a = moments_with(grid_points(), Some(&full), &A, &B);
    let b = moments_with(grid_points(), Some(&holed), &A, &B);
    assert_eq!(
        a.count,
        b.count + 1,
        "flipping one mask byte changed the count by {} rather than 1",
        a.count as i64 - b.count as i64
    );
}

/// A fixed mask with an explicit point list stays refused **by name** for this metric
/// too. The invariant is `Resident`'s, not mean-squares', and deleting it here would be
/// a silent regression of a rule the other metric still enforces.
#[test]
fn a_fixed_mask_with_an_explicit_point_list_is_refused_by_name() {
    if no_device() {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let pts: Vec<f64> = (0..VOXELS).flat_map(point_of).collect();
    let mask_img = Image::from_vec(&SIZE, vec![1u8; VOXELS]).unwrap();
    let mask = DeviceMask::upload(&mask_img).unwrap();

    let (f, m) = (volume(1), volume(9));
    let (d_f, d_m) = (
        DeviceImage::upload(&f).unwrap(),
        DeviceImage::upload(&m).unwrap(),
    );
    let err = ResidentCorrelation::from_device_masked(
        &d_f,
        FixedPoints::Explicit(&pts),
        Some(&mask),
        &d_m,
        &moving_geometry(),
    )
    .err()
    .expect("a fixed mask with an explicit point list must be refused");
    assert!(
        matches!(err, CudaError::MaskedExplicitPoints),
        "refused, but not by name: {err}"
    );
}
