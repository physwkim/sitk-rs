//! The **discrete** decisions the device sampler makes about a sample's existence,
//! driven at geometries constructed so a sample lands exactly on the deciding boundary.
//!
//! They are asserted *exactly* by every other pin in this crate (`valid_points` is an
//! integer and is compared with `assert_eq!`) and have never been exercised at a
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
use support::{in_buffer_straddles, no_device, on_buffer_boundary};

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

/// **The in-buffer predicate under a transform whose probed affine form is NOT exact:
/// the two paths count different numbers of valid samples. Measured.**
///
/// A rotation about z: the device's `A[0][0]` is recovered as `(cos γ + offset) − offset`
/// with an offset of ~20, which loses about four bits. So the two paths' `c_x` differ by
/// ~1e-14, and a sample within that gap of `c_x = 31.5` is *inside the buffer on one path
/// and outside on the other*.
///
/// The sweep walks `tx` in ulp steps across the value at which the outermost samples
/// cross. Measured, at 16³ fixed / 32³ moving:
///
/// * **3 of the 17 swept poses disagree about `valid_points`**
/// * at each, the device counts **16 fewer** valid samples than the host — the whole
///   `(15, 0, k)` line, `k = 0..16`: a z-rotation makes `c_x` a function of `(i, j)`
///   alone, so every sample on that line shares one `c_x` and all 16 flip together
/// * `|c_host − c_dev|` at those samples: **1.07e-14**
///
/// So the device's honest contract is **not** "`valid_points` is exact". It is:
///
/// > `valid_points` is exact **except** at poses where a sample lands within ~1e-14 of
/// > the moving buffer's boundary, where the two paths differ by the samples on that
/// > plane.
///
/// Every other pin in this crate asserts `valid_points` with `assert_eq!` and does not
/// state that qualification. They pass because their poses are not straddling ones — not
/// because the equality holds in general.
///
/// This is **not** the §2.158 cell-boundary straddle, where both answers are valid
/// one-sided limits of a genuinely discontinuous derivative and neither path is wrong.
/// `is_inside` is not discontinuous mathematics — it is an exact predicate on an exact
/// number, and the two paths disagree because they are evaluating **two different maps**
/// that happen to agree to 1e-14. One of them (the host's) is the reference.
///
/// If this test ever reports zero disagreements, the device's point map has become
/// bit-identical to the host's and the whole family is closed — rewrite the pin, do not
/// delete it.
#[test]
fn the_in_buffer_predicate_is_measured_across_the_boundary() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let moving = volume(32, [3.0, -2.0, 1.5]);
    let fixed = volume(16, [0.0; 3]);
    let target = 32.0 - 0.5;

    let t0 = critical_tx(&fixed, &moving, target);
    println!("critical tx = {t0:.17e} (outermost sample lands on c_x = {target})");

    let mut disagreements = 0usize;
    let mut host_counts = Vec::new();
    for tx in sweep(t0, 8) {
        let t = rotated(tx);
        let straddles = in_buffer_straddles(&fixed, &moving, &t);
        let (h, d) = counts(&fixed, &moving, &t);
        host_counts.push(h);
        let gap = straddles.first().map(|s| s.gap()).unwrap_or(0.0);
        println!(
            "tx {tx:.17e}  host {h:>5}  device {d:>5}  delta {:>3}  probe straddles {} \
             (|c_host − c_dev| = {gap:.3e})",
            d as i64 - h as i64,
            straddles.len(),
        );
        // The probe and the metrics must tell the same story: the probe's count of
        // predicate disagreements *is* the count difference, sample for sample. This is
        // what licenses the probe to be used as a precondition elsewhere.
        assert_eq!(
            d as i64 - h as i64,
            straddles
                .iter()
                .map(|s| if s.dev_c[0] < target { 1i64 } else { -1 })
                .sum::<i64>(),
            "the probe and the two metrics disagree about which samples are in --- one \
             of them is not reproducing the path it claims to"
        );
        if h == d {
            continue;
        }
        disagreements += 1;

        // What a disagreement looks like, measured: the device drops a whole line of
        // samples the host keeps.
        assert_eq!(
            h as i64 - d as i64,
            16,
            "the device is supposed to be the one that drops them (its c_x lands on the \
             far side of size − 0.5), and it is supposed to drop the whole line"
        );
        assert_eq!(straddles.len(), 16);
        assert!(
            straddles
                .iter()
                .all(|s| s.index[0] == 15 && s.index[1] == 0),
            "the straddling samples are the (15, 0, k) line --- a z-rotation makes c_x a \
             function of (i, j) alone, so the line flips as one"
        );
        assert!(
            straddles.iter().all(|s| s.host_c[0] < 31.5),
            "the host has them inside"
        );
        assert!(
            straddles.iter().all(|s| s.dev_c[0] >= 31.5),
            "and the device has them outside"
        );
        assert!(
            (1e-16..1e-12).contains(&gap),
            "|c_host − c_dev| = {gap:e} at the straddle. Bounded above because a larger \
             gap is a broken map, not a probed one; bounded BELOW because a gap of zero \
             means the two paths now compute the same point and this pin's whole story \
             has changed"
        );
    }

    // The construction has to have worked: the sweep must actually walk samples across
    // the boundary, or it measured nothing.
    assert!(
        host_counts.first() != host_counts.last(),
        "the sweep did not move a sample across the buffer boundary ({host_counts:?}), \
         so it has not exercised the predicate at all"
    );
    println!(
        "{disagreements} of {} swept poses disagree about valid_points, by 16 samples each",
        2 * 8 + 1
    );
    assert!(
        disagreements > 0,
        "no swept pose disagreed. Either the device's point map is now bit-identical to \
         the host's (good --- rewrite this pin) or the sweep no longer brackets the \
         crossing (bad --- the pin has gone vacuous)"
    );
}
