//! **The invariant the device path rests on: the continuous index is bit-identical.**
//!
//! Every discrete decision the sampler makes is a branch on the continuous index `c` —
//! `floor(c)` (which cell, hence which one-sided limit of a discontinuous ∇M),
//! `is_inside(c)` (whether the sample exists), `round(c)` (which mask voxel gates it).
//! A map that agrees with the host to 1e-14 does not agree about those; it agrees about
//! everything *except* those. So the contract is not a tolerance:
//!
//! > For every transform the device metric accepts, at every sample, the `c` the device
//! > computes is **bit for bit** the `c` the host computes.
//!
//! It holds because the device is handed the transform's own **stages** — the stored
//! `matrix`/`offset` pairs its `transform_point` evaluates, one per map it applies, in the
//! order it applies them — and replays them, one rounded `mat_vec` plus one rounded add per
//! stage. Same operations, same order, same bits. `cuda_boundary.rs` and
//! `cuda_mean_squares.rs` pin the *consequences* on the device (the valid-sample count and
//! the derivative at samples sitting exactly on a wall); this file pins the invariant
//! itself, over a large random pose set rather than at the poses where it happens to show.
//!
//! It also pins what is **refused**, and refused *by name*: a transform whose
//! `transform_point` evaluates some other expression cannot state a stage list, the device
//! will not substitute one that merely approximates it, and the metric says so.

#![cfg(feature = "cuda")]

mod support;

use sitk_core::Image;
use sitk_cuda::DeviceImage;
use sitk_registration::device::DeviceMetricError;
use sitk_registration::metric::{FixedSamples, MovingImage};
use sitk_registration::{DeviceMeanSquaresMetric, MeanSquaresMetric};
use sitk_transform::matrix_offset::replay_stages;
use sitk_transform::{
    AffineTransform, CompositeTransform, Euler3DTransform, ParametricTransform, ScaleTransform,
    Similarity3DTransform, TransformBase, TranslationTransform, VersorRigid3DTransform,
};
use support::{index_bit_mismatches, no_device};

/// A deterministic pose generator — a plain LCG, so the sweep is reproducible and a
/// failure is re-runnable. Values are irrational-ish and O(10): large enough that the
/// offsets in a centred transform lose bits the way a real pose does.
struct Poses(u64);

impl Poses {
    fn next_f64(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
        lo + u * (hi - lo)
    }
}

fn volume(n: usize, shift: [f64; 3]) -> Image {
    let mut v = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (
                    i as f64 + shift[0],
                    j as f64 + shift[1],
                    k as f64 + shift[2],
                );
                v.push(((x / 3.0).sin() * (y / 4.0).cos() + (z / 5.0).sin()) as f32);
            }
        }
    }
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[1.0, 1.0, 1.0]).unwrap();
    img
}

/// **The invariant, over 240 random poses of every accepted transform family.**
///
/// `to_bits()` equality on the continuous index, at *every* sample of *every* pose — not a
/// tolerance, and not only at the straddle poses. The straddle poses are where the
/// invariant *shows*; this is where it lives.
///
/// The families are the ones the device accepts: translation, Euler, versor-rigid,
/// similarity, affine, and a composite of three of them — which is the interesting one,
/// because a composite is where a fold would hide. Folding its stage matrices into one
/// product is algebraically the same map and rounds once where the host rounds three
/// times; the replay rounds where the host rounds.
#[test]
fn the_continuous_index_is_bit_identical_over_a_random_pose_set() {
    let fixed = volume(10, [0.0; 3]);
    let moving = volume(20, [3.0, -2.0, 1.5]);
    let mut rng = Poses(0x5EED_1234_ABCD_0001);

    let mut poses = 0usize;
    for _ in 0..40 {
        let c = [
            rng.next_f64(-20.0, 20.0),
            rng.next_f64(-20.0, 20.0),
            rng.next_f64(-20.0, 20.0),
        ];
        let t = [
            rng.next_f64(-15.0, 15.0),
            rng.next_f64(-15.0, 15.0),
            rng.next_f64(-15.0, 15.0),
        ];
        let (a, b, g) = (
            rng.next_f64(-0.6, 0.6),
            rng.next_f64(-0.6, 0.6),
            rng.next_f64(-0.6, 0.6),
        );
        let s = rng.next_f64(0.7, 1.4);
        // A versor's three parameters are the vector part; keep it well inside the unit
        // ball so the scalar part is real.
        let versor = VersorRigid3DTransform::new(
            rng.next_f64(-0.4, 0.4),
            rng.next_f64(-0.4, 0.4),
            rng.next_f64(-0.4, 0.4),
            t,
            c,
        );
        let euler = Euler3DTransform::new(a, b, g, t, c);
        let translation = TranslationTransform::new(t.to_vec());
        let similarity = Similarity3DTransform::new(
            s,
            rng.next_f64(-0.3, 0.3),
            rng.next_f64(-0.3, 0.3),
            rng.next_f64(-0.3, 0.3),
            t,
            c,
        );
        let affine = AffineTransform::new(
            3,
            vec![
                1.0 + rng.next_f64(-0.2, 0.2),
                rng.next_f64(-0.2, 0.2),
                rng.next_f64(-0.2, 0.2),
                rng.next_f64(-0.2, 0.2),
                1.0 + rng.next_f64(-0.2, 0.2),
                rng.next_f64(-0.2, 0.2),
                rng.next_f64(-0.2, 0.2),
                rng.next_f64(-0.2, 0.2),
                1.0 + rng.next_f64(-0.2, 0.2),
            ],
            t.to_vec(),
            c.to_vec(),
        );

        // The composite: three stages, replayed in application order. `CompositeTransform`
        // applies its queue in reverse add order, and `point_map_stages` flattens it the
        // same way — a fold here would round once instead of three times.
        let mut composite = CompositeTransform::new(3);
        composite.add_transform(euler.clone().into()).unwrap();
        composite
            .add_transform(TranslationTransform::new(vec![-t[1], t[2], -t[0]]).into())
            .unwrap();
        composite.add_transform(affine.clone().into()).unwrap();
        assert_eq!(
            composite.point_map_stages().map(|s| s.len()),
            Some(3),
            "the composite must hand over its three stages, not one folded matrix"
        );

        let family: Vec<(&str, &dyn ParametricTransform)> = vec![
            ("translation", &translation),
            ("euler", &euler),
            ("versor", &versor),
            ("similarity", &similarity),
            ("affine", &affine),
            ("composite", &composite),
        ];
        for (name, t) in family {
            let bad = index_bit_mismatches(&fixed, &moving, t);
            assert!(
                bad.is_empty(),
                "{name}: {} sample(s) of {} have a continuous index that differs between \
                 the host and the device chains -- e.g. host {:?} bits {:?}, device {:?} \
                 bits {:?}. The invariant is bit identity, so one differing bit is the \
                 whole failure",
                bad.len(),
                10 * 10 * 10,
                bad[0].host_c,
                bad[0].host_c.map(f64::to_bits),
                bad[0].dev_c,
                bad[0].dev_c.map(f64::to_bits),
            );
            poses += 1;
        }
    }
    println!("{poses} poses x 1000 samples: every continuous index bit-identical");
}

/// **The lost set, named and measured: a scale transform is refused, and here is exactly
/// what accepting it would have cost.**
///
/// `ScaleTransform` (and `ScaleLogarithmicTransform`, which delegates to it) evaluates the
/// centred `(p − c)·s + c`. That is `M·p + b` in exact arithmetic and it is **not** that on
/// the bits — so it has no stage list, and it is the *only* family the stage list loses
/// relative to the affine probe it replaced (a B-spline and a displacement field failed the
/// probe too, and are not a loss).
///
/// This test is the evidence for both halves of that claim, on the host, with no device:
///
/// 1. the probed affine form **does** reproduce `transform_point` to a tolerance — so the
///    old code accepted this transform, and the loss is real rather than hypothetical; and
/// 2. it does **not** reproduce it on the bits — which is why it is refused rather than
///    approximated.
#[test]
fn a_scale_transform_is_the_lost_set_and_here_is_what_it_would_have_cost() {
    let t = ScaleTransform::new(vec![1.3, 0.8, 1.1], vec![21.5, -13.25, 8.75]);
    assert!(
        t.point_map_stages().is_none(),
        "a centred scale has no bitwise matrix-offset form and must not claim one"
    );

    // The probe the device used to run: b = T(0), A[:,e] = T(e_e) - b.
    let b = t.transform_point(&[0.0, 0.0, 0.0]);
    let mut a = [0.0f64; 9];
    for e in 0..3 {
        let mut basis = [0.0f64; 3];
        basis[e] = 1.0;
        let te = t.transform_point(&basis);
        for d in 0..3 {
            a[d * 3 + e] = te[d] - b[d];
        }
    }

    let probe = [1.7, -3.1, 2.3];
    let truth = t.transform_point(&probe);
    let mut probed = [0.0f64; 3];
    let mut differing_bits = 0usize;
    for d in 0..3 {
        probed[d] = b[d] + (0..3).map(|e| a[d * 3 + e] * probe[e]).sum::<f64>();
        let err = (probed[d] - truth[d]).abs() / (1.0 + truth[d].abs());
        // (1) the old probe accepted it: 1e-9 was its tolerance.
        assert!(
            err <= 1e-9,
            "axis {d}: the probed form misses by {err:e}, so this transform was NOT accepted \
             by the old code and is not a loss at all"
        );
        if probed[d].to_bits() != truth[d].to_bits() {
            differing_bits += 1;
        }
    }
    println!("scale: host {truth:?}");
    println!("scale: probed affine form {probed:?}");
    println!(
        "scale: bits differ on {differing_bits} of 3 axes; host {:?} vs probed {:?}",
        truth.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        probed.map(f64::to_bits),
    );
    // (2) and it was accepted while being bitwise wrong -- which is the defect.
    assert!(
        differing_bits > 0,
        "the probed form of a centred scale is bit-identical to its own transform_point at \
         this probe point, so nothing is lost by refusing it and nothing was wrong with \
         accepting it. If this ever holds at EVERY point, give ScaleTransform a stage list"
    );
}

/// **The refusal is typed, and it is the right type.** A scale transform reaches the
/// device metric and is turned away by name — not by a silent CPU fallback, and not
/// misreported as a non-affine transform (its Jacobian *is* affine in the point; it is the
/// point map that has no bitwise form, and the two refusals are different refusals).
#[test]
fn the_device_metric_refuses_a_scale_transform_by_name() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let fixed = volume(16, [0.0; 3]);
    let moving = volume(16, [3.0, -2.0, 1.5]);
    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();
    let metric = DeviceMeanSquaresMetric::from_device(&d_fixed, &d_moving).unwrap();

    let scale = ScaleTransform::new(vec![1.1, 0.9, 1.05], vec![8.0, 8.0, 8.0]);
    let err = metric.evaluate(&scale).unwrap_err();
    println!("scale on the device metric: {err}");
    assert!(
        matches!(err, DeviceMetricError::NoBitwisePointMap),
        "a scale transform must be refused as NoBitwisePointMap, not as {err:?}"
    );

    // And the transform the device DOES take still runs, so the refusal is about the scale
    // and not about the metric being broken.
    let euler = Euler3DTransform::new(0.05, -0.03, 0.02, [1.5, -1.0, 0.5], [8.0, 8.0, 8.0]);
    assert!(metric.evaluate(&euler).is_ok());
}

/// The CPU still evaluates what the device refuses — the refusal costs the caller speed,
/// never an answer. `MeanSquaresMetric` on the host takes the scale transform the device
/// turned away, and the stage replay is not involved.
#[test]
fn the_cpu_still_evaluates_what_the_device_refuses() {
    let fixed = volume(16, [0.0; 3]);
    let moving = volume(16, [3.0, -2.0, 1.5]);
    let m = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&fixed).unwrap(),
        MovingImage::from_image(&moving).unwrap(),
    )
    .unwrap();
    let scale = ScaleTransform::new(vec![1.1, 0.9, 1.05], vec![8.0, 8.0, 8.0]);
    let v = m.evaluate(&scale, &sitk_registration::CpuBackend);
    assert!(v.valid_points > 0, "the scale maps nothing inside");
    assert!(v.value.is_finite());
}

/// The stage list IS `transform_point`, on the bits, for a composite — the property the
/// kernel's replay depends on, checked here against the transform the device is actually
/// handed rather than only inside `sitk-transform`'s own unit tests.
#[test]
fn replaying_a_composites_stages_reproduces_its_own_transform_point() {
    let euler = Euler3DTransform::new(0.2, -0.1, 0.05, [11.0, -7.0, 3.0], [12.0, 9.0, -4.0]);
    let affine = AffineTransform::new(
        3,
        vec![1.1, 0.05, -0.02, 0.03, 0.95, 0.01, -0.04, 0.02, 1.2],
        vec![-6.0, 2.0, 9.0],
        vec![3.0, -8.0, 5.0],
    );
    let mut composite = CompositeTransform::new(3);
    composite.add_transform(euler.into()).unwrap();
    composite.add_transform(affine.into()).unwrap();

    let stages = composite
        .point_map_stages()
        .expect("both halves have stages");
    assert_eq!(stages.len(), 2);

    for p in [
        [1.7, -3.1, 2.3],
        [-137.0, 91.5, 204.25],
        [0.0, 0.0, 0.0],
        [-0.0, 1e-300, 1e300],
    ] {
        let want = composite.transform_point(&p);
        let got = replay_stages(&stages, &p, 3);
        for d in 0..3 {
            assert_eq!(
                got[d].to_bits(),
                want[d].to_bits(),
                "axis {d} at {p:?}: replay {:e} != transform_point {:e}",
                got[d],
                want[d]
            );
        }
    }
}
