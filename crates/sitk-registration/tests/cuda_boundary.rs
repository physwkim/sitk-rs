//! The two **discrete** decisions the device sampler makes about a sample's existence —
//! `is_inside` (the in-buffer predicate) and `round(c)` (the moving-mask lookup) — driven
//! at geometries constructed so a sample lands exactly on each boundary.
//!
//! Both are asserted *exactly* by every other pin in this crate (`valid_points` is an
//! integer and is compared with `assert_eq!`). Neither has ever been exercised at a
//! straddling geometry. These tests construct one and measure.
//!
//! Only compiled with the `cuda` feature.
#![cfg(feature = "cuda")]

mod support;

use sitk_core::Image;
use sitk_cuda::DeviceImage;
use sitk_registration::metric::{FixedSamples, MovingImage};
use sitk_registration::{CpuBackend, DeviceMeanSquaresMetric, MeanSquaresMetric};
use sitk_transform::{Euler3DTransform, ParametricTransform, TransformBase, TranslationTransform};
use support::{
    in_buffer_straddles, mask_bits, moving_mask_straddles, no_device, on_buffer_boundary,
    on_round_tie,
};

/// Textured volume, unit spacing, origin at zero — so a continuous moving index *is* a
/// physical coordinate and a boundary can be constructed by inspection rather than by
/// solving a geometry.
fn volume(n: usize, shift: [f64; 3]) -> Image {
    let c = n as f64 / 2.0;
    let mut v = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (
                    i as f64 - c + shift[0],
                    j as f64 - c + shift[1],
                    k as f64 - c + shift[2],
                );
                let d2 = x * x + y * y + z * z;
                let s = 120.0 * (-d2 / (2.0 * (n as f64 / 5.0).powi(2))).exp()
                    + 10.0 * (x / 5.0).sin() * (y / 7.0).cos() * (z / 11.0).sin();
                v.push(s as f32);
            }
        }
    }
    Image::from_vec(&[n, n, n], v).unwrap()
}

fn counts(fixed: &Image, moving: &Image, t: &dyn ParametricTransform) -> (usize, usize) {
    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(fixed).unwrap(),
        MovingImage::from_image(moving).unwrap(),
    )
    .unwrap();
    let device = DeviceMeanSquaresMetric::from_device(
        &DeviceImage::upload(fixed).unwrap(),
        &DeviceImage::upload(moving).unwrap(),
    )
    .unwrap();
    (
        host.evaluate(t, &CpuBackend).valid_points,
        device.evaluate(t).unwrap().valid_points,
    )
}

fn masked_counts(
    fixed: &Image,
    moving: &Image,
    mask: &Image,
    t: &dyn ParametricTransform,
) -> (usize, usize) {
    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(fixed).unwrap(),
        MovingImage::from_image(moving)
            .unwrap()
            .with_moving_mask(mask)
            .unwrap(),
    )
    .unwrap();
    let bits = mask_bits(mask);
    let device = DeviceMeanSquaresMetric::from_device_masked(
        &DeviceImage::upload(fixed).unwrap(),
        &DeviceImage::upload(moving).unwrap(),
        None,
        Some(&bits),
    )
    .unwrap();
    (
        host.evaluate(t, &CpuBackend).valid_points,
        device.evaluate(t).unwrap().valid_points,
    )
}

/// The pose family for the sweeps: a rotation **about z** by 0.3 rad, about a centre
/// inside the fixed block, translated by `tx` in x.
///
/// Why a rotation and not a translation: the device is handed `A` probed as
/// `A[d][e] = T(e_e)[d] − T(0)[d]`. For a `TranslationTransform` the matrix is the
/// identity and that probe is *exact* (`(1 + t) − t == 1` for any t of a sane
/// magnitude, `(0 + t) − t == 0` always), so the two paths compute the same `c` bit for
/// bit and no ulp exists to straddle with. A rotation puts `cos γ ≈ 0.955` next to an
/// offset of ~20 in that subtraction, which loses about four bits — and *that* is the
/// gap the boundary tests need. The x row is the one that loses them, so the boundary
/// under test is on x.
fn rotated(tx: f64) -> Euler3DTransform {
    Euler3DTransform::new(0.0, 0.0, 0.3, [tx, 0.0, 0.0], [8.0, 8.0, 8.0])
}

/// The `tx` at which the *last* sample to leave crosses the x boundary at `target`,
/// found from the host's own map: `c_x` is affine in `tx` with unit slope, so
/// `tx* = target − max_s c_x(s)|_{tx=0}`.
fn critical_tx(fixed: &Image, moving: &Image, target: f64) -> f64 {
    let t0 = rotated(0.0);
    let mut worst = f64::NEG_INFINITY;
    for k in 0..fixed.size()[2] {
        for j in 0..fixed.size()[1] {
            for i in 0..fixed.size()[0] {
                let p = t0.transform_point(&[i as f64, j as f64, k as f64]);
                let c = p[0] - moving.origin()[0];
                worst = worst.max(c);
            }
        }
    }
    target - worst
}

/// Sweep `tx` in ulp-sized steps across `t0` and report, at each step, what the two
/// paths count.
fn sweep(t0: f64, span: i32) -> Vec<f64> {
    let ulp = (t0.abs() + 1.0) * f64::EPSILON;
    (-span..=span).map(|k| t0 + k as f64 * ulp).collect()
}

// ---------------------------------------------------------------------------------
// 1. The in-buffer predicate: `c_d ∈ [−0.5, size_d − 0.5)`.
// ---------------------------------------------------------------------------------

/// **A whole face of samples placed exactly on `c = size − 0.5`, under a transform whose
/// probed affine form is exact.** Measured, not predicted.
///
/// A pure translation: `T(x) = x + t`, matrix the identity. The device's probe recovers
/// that identity *exactly* — `(1 + t) − t == 1`, `(0 + t) − t == 0` — so the two paths
/// compute the same `c` bit for bit, and the whole 16×16 face that lands on
/// `c_x == 31.5` is dropped by both under the half-open rule. The point of the test is
/// that the face is really there (`on_buffer_boundary` is asserted non-empty and of the
/// exact expected size) and that the *closed* end at `c_x == −0.5` is really kept.
///
/// This is the case where the arithmetic coincides, and it coincides for a reason that
/// can be stated rather than for luck. The next test is the case where it does not.
#[test]
fn a_face_on_the_open_boundary_is_dropped_by_both_paths() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let moving = volume(32, [3.0, -2.0, 1.5]);
    let fixed = volume(16, [0.0; 3]);

    // The open end: fixed voxel i = 15 maps to c_x = 15 + 16.5 = 31.5 == size − 0.5.
    let open = TranslationTransform::new(vec![16.5, 0.0, 0.0]);
    let face = on_buffer_boundary(&fixed, &moving, &open);
    assert_eq!(
        face.len(),
        16 * 16,
        "the construction is supposed to put a whole 16x16 face exactly on c_x = 31.5"
    );
    assert!(face.iter().all(|(s, d)| *d == 0 && s.index[0] == 15));
    assert!(
        face.iter().all(|(s, _)| s.host_c[0] == 31.5),
        "exactly on the boundary, not near it"
    );
    // And the two paths agree on that face to the last bit — the probed identity is exact.
    assert!(
        face.iter()
            .all(|(s, _)| s.host_c[0].to_bits() == s.dev_c[0].to_bits()),
        "a translation's probed affine form is exact, so there is no ulp here to straddle"
    );

    let straddles = in_buffer_straddles(&fixed, &moving, &open);
    let (h, d) = counts(&fixed, &moving, &open);
    println!(
        "open end  (c_x = 31.5 on 256 samples): host {h}, device {d}, \
         predicate straddles {}",
        straddles.len()
    );
    assert!(straddles.is_empty());
    assert_eq!(h, d, "host and device disagree about the valid set");
    assert_eq!(
        h,
        15 * 16 * 16,
        "the face on c_x = size − 0.5 must be OUT: the rule is half-open"
    );

    // The closed end: fixed voxel i = 0 maps to c_x = −0.5, which is IN.
    let closed = TranslationTransform::new(vec![-0.5, 0.0, 0.0]);
    let face = on_buffer_boundary(&fixed, &moving, &closed);
    assert_eq!(face.len(), 16 * 16);
    assert!(face.iter().all(|(s, _)| s.host_c[0] == -0.5));
    let (h, d) = counts(&fixed, &moving, &closed);
    println!("closed end (c_x = -0.5 on 256 samples): host {h}, device {d}");
    assert_eq!(h, d);
    assert_eq!(
        h,
        16 * 16 * 16,
        "the face on c_x = −0.5 must be IN: the rule is half-OPEN, closed at the bottom"
    );
}

/// **The in-buffer predicate under a rotation whose offset is large: the two paths agree
/// at every swept pose. Measured.**
///
/// This pin used to assert the opposite, and the flip is the evidence that the point map
/// was fixed at the root. What it measured before: the device was handed an affine form
/// *probed* out of the transform, so its `A[0][0]` was recovered as
/// `(cos γ + offset) − offset` with an offset of ~20 — about four bits lost. The two
/// paths' `c_x` then differed by ~1.07e-14, and a sample within that gap of `c_x = 31.5`
/// was *inside the buffer on one path and outside on the other*: **3 of the 17 swept poses
/// disagreed about `valid_points`, by 16 samples each** (the whole `(15, 0, k)` line — a
/// z-rotation makes `c_x` a function of `(i, j)` alone, so the line flips as one).
///
/// The device is now handed the transform's own **stages** and replays them, so its `c` is
/// the host's bit for bit, and the disagreement is gone *by construction* rather than by
/// luck of the pose. The sweep is kept — it still walks `tx` in ulp steps across the value
/// at which the outermost samples cross `c_x = size − 0.5`, so the predicate is still
/// exercised right at its discontinuity — and every pose must now agree:
///
/// * the probe finds **no** sample whose `is_inside` differs between the paths, and
/// * the two metrics count the **same** `valid_points`.
///
/// `valid_points` is therefore exact without qualification, which is what every other pin
/// in this crate has always asserted with `assert_eq!`. They passed because their poses
/// were not straddling ones; now there are none.
#[test]
fn the_in_buffer_predicate_agrees_across_the_boundary() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let moving = volume(32, [3.0, -2.0, 1.5]);
    let fixed = volume(16, [0.0; 3]);
    let target = 32.0 - 0.5;

    let t0 = critical_tx(&fixed, &moving, target);
    println!("critical tx = {t0:.17e} (outermost sample lands on c_x = {target})");

    let mut host_counts = Vec::new();
    for tx in sweep(t0, 8) {
        let t = rotated(tx);
        let straddles = in_buffer_straddles(&fixed, &moving, &t);
        let (h, d) = counts(&fixed, &moving, &t);
        host_counts.push(h);
        println!(
            "tx {tx:.17e}  host {h:>5}  device {d:>5}  delta {:>3}  probe straddles {}",
            d as i64 - h as i64,
            straddles.len(),
        );
        assert!(
            straddles.is_empty(),
            "{} samples land on opposite sides of the in-buffer predicate at tx {tx:e}. The \
             two paths are computing different continuous indices again --- e.g. {:?}",
            straddles.len(),
            straddles.first(),
        );
        assert_eq!(
            h, d,
            "host and device disagree about valid_points at tx {tx:e}, right at the \
             buffer boundary"
        );
    }

    // The construction has to have worked: the sweep must actually walk samples across
    // the boundary, or agreement is vacuous.
    assert!(
        host_counts.first() != host_counts.last(),
        "the sweep did not move a sample across the buffer boundary ({host_counts:?}), \
         so it has not exercised the predicate at all"
    );
    println!(
        "all {} swept poses agree on valid_points, including the ones straddling the boundary",
        2 * 8 + 1
    );
}

// ---------------------------------------------------------------------------------
// 2. The moving-mask lookup: `round(c_d)`.
// ---------------------------------------------------------------------------------

/// A moving mask whose boundary lands on a **half-integer**: voxels with `i ≤ 20` are in,
/// `i ≥ 21` are out, so the round-to-nearest tie at `c_x = 20.5` is exactly the wall.
fn half_space_mask(n: usize, last_in: usize) -> Image {
    let v: Vec<f32> = (0..n * n * n)
        .map(|s| if s % n <= last_in { 1.0 } else { 0.0 })
        .collect();
    Image::from_vec(&[n, n, n], v).unwrap()
}

/// **The moving-mask lookup across the round-to-nearest tie: the two paths agree at every
/// swept pose. Measured.**
///
/// `round(c_x)` decides which mask voxel gates the sample, and the mask's wall is on the
/// tie itself (`c_x = 20.5`). Both paths round half away from zero (Rust's `f64::round`,
/// CUDA's `round`), so the rounding *rule* was never the problem — the two paths did not
/// have the same `c_x`. What this pin measured under the probed affine form:
///
/// * **4 of the 17 swept poses disagreed about `valid_points`**, by 16 samples each
/// * the host had `c_x = 20.5` exactly → voxel **21** → outside the mask → dropped; the
///   device had `c_x = 20.49999999999999289` → voxel **20** → inside the mask → kept
///
/// and the direction was the *opposite* of the in-buffer sweep's, which was the point: not
/// a bias in one predicate, but the same 1e-14 gap in the point map read by two different
/// discrete rules.
///
/// The stage replay gives the device the host's `c` bit for bit, so both rules now read
/// the same number and both sweeps agree. The tie is still exercised at every pose — the
/// host's `c_x` still lands exactly on 20.5 at the critical pose — and now the device's
/// does too.
#[test]
fn the_moving_mask_lookup_agrees_across_the_tie() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let moving = volume(32, [3.0, -2.0, 1.5]);
    let fixed = volume(16, [0.0; 3]);
    let mask = half_space_mask(32, 20);
    let target = 20.5;

    let t0 = critical_tx(&fixed, &moving, target);
    println!("critical tx = {t0:.17e} (outermost sample lands on c_x = {target})");

    let mut host_counts = Vec::new();
    let mut on_the_tie = 0usize;
    for tx in sweep(t0, 8) {
        let t = rotated(tx);
        let straddles = moving_mask_straddles(&fixed, &moving, &mask, &t);
        let (h, d) = masked_counts(&fixed, &moving, &mask, &t);
        host_counts.push(h);
        println!(
            "tx {tx:.17e}  host {h:>5}  device {d:>5}  delta {:>3}  probe straddles {}",
            d as i64 - h as i64,
            straddles.len(),
        );
        assert!(
            straddles.is_empty(),
            "{} samples are gated by different mask voxels on the two paths at tx {tx:e} \
             --- e.g. {:?}",
            straddles.len(),
            straddles.first(),
        );
        assert_eq!(
            h, d,
            "host and device disagree about valid_points at tx {tx:e}, right on the mask's \
             round-to-nearest tie"
        );
        // Agreement is only worth something if the tie was actually hit: a pose that
        // never puts a sample exactly on 20.5 agrees trivially.
        on_the_tie += usize::from(
            on_round_tie(&fixed, &moving, &t)
                .iter()
                .any(|(s, d)| *d == 0 && s.host_c[0] == target),
        );
    }

    assert!(
        host_counts.first() != host_counts.last(),
        "the sweep did not move a sample across the mask's tie ({host_counts:?}), so it \
         has not exercised the lookup at all"
    );
    assert!(
        on_the_tie > 0,
        "no swept pose put a sample exactly on the tie, so the lookup was never asked the \
         question this pin exists to ask"
    );
    println!(
        "all {} swept poses agree on valid_points; {on_the_tie} of them put a sample exactly \
         on the tie",
        2 * 8 + 1
    );
}
