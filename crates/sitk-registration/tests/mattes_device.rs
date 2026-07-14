//! The device Mattes metric's pins.
//!
//! The claim, split in two because the two halves are held to two different standards:
//!
//! 1. **The value is the host's, bit for bit.** Not close to it. The joint histogram is
//!    the counting sort's, which is pinned bit-identical to the host's accumulation loop,
//!    and it is fed to the host metric's *own* tail. So there is nothing left for the two
//!    to disagree about except the per-sample chain — and that is the point of the pin.
//! 2. **The derivative is banded at 1e-9 relative**, because the device is told the
//!    Jacobian as a *probed* affine decomposition (`C_e = J(e_e) − J(0)`, a cancelling
//!    subtraction) rather than as the transform's own `jacobian_wrt_parameters`. That is
//!    the one expression that blocks bit-identity, and it is named.
//!
//! Both are worthless without the anti-vacuity, so
//! [`the_interpolated_value_reaches_a_discrete_decision_so_the_sampler_pin_is_load_bearing`]
//! **measures** the thing the bit-identity pin protects against: how many samples sit
//! within one ulp of a Parzen bin boundary, i.e. how many would land in a different
//! histogram cell if the device's trilinear arithmetic were off by a single bit. If that
//! count were zero, the value pin would be ceremony. It is not zero.
#![cfg(feature = "cuda")]

use sitk_core::Image;
use sitk_cuda::{CudaError, DeviceImage, backend};
use sitk_registration::{
    DeviceMattesMetric, DeviceMetricError, MattesMutualInformationMetric, RegistrationError,
};
use sitk_transform::{
    AffineTransform, BSplineTransform, Euler3DTransform, ParametricTransform, ScaleTransform,
    TranslationTransform,
};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

const BINS: usize = 50;
/// Bins of padding at each histogram-axis end — ITK's `padding`, and the host metric's.
const PADDING: usize = 2;

/// A structured 3-D volume on a non-trivial geometry, so the point map and the
/// physical-to-index map are both doing work.
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
                    + 10.0 * (x / 7.0).sin() * (y / 9.0).cos() * (z / 11.0).sin();
                v.push(s as f32);
            }
        }
    }
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[0.8, 0.9, 1.1]).unwrap();
    img.set_origin(&[-3.0, 2.0, 1.0]).unwrap();
    img
}

/// A **contrast-inverted** moving volume: the multi-modality case Mattes exists for, and
/// the one where a mean-squares metric is useless. `120 − f`, so the joint histogram is
/// an anti-diagonal ridge rather than a diagonal one.
fn inverted(n: usize, shift: [f64; 3]) -> Image {
    let base = volume(n, shift);
    let v: Vec<f32> = base
        .to_f64_vec()
        .unwrap()
        .iter()
        .map(|&x| (120.0 - x) as f32)
        .collect();
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[0.8, 0.9, 1.1]).unwrap();
    img.set_origin(&[-3.0, 2.0, 1.0]).unwrap();
    img
}

/// A **commensurate** pair: the moving grid is the fixed grid, unit spacing, same origin.
/// Under a small integer-ish translation a great many samples land exactly on voxel
/// centres, so the interpolated value is exactly a stored voxel — which is precisely
/// where a Parzen term lands on a bin boundary and one ulp changes the bin. Not exotic:
/// it is what the identity transform does on any real image pair from the same scanner.
fn commensurate(n: usize) -> Image {
    let mut v = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                // Integer-valued intensities on a small range: many samples share a bin
                // edge, which is the population the straddle lives in.
                v.push(((i * 7 + j * 13 + k * 3) % 64) as f32);
            }
        }
    }
    Image::from_vec(&[n, n, n], v).unwrap()
}

/// A non-trivial affine: a matrix that is not the identity and an off-centre rotation
/// centre, so the point map is two stages and the Jacobian depends on the point.
fn affine() -> AffineTransform {
    AffineTransform::new(
        3,
        vec![1.02, 0.03, -0.01, -0.02, 0.99, 0.04, 0.01, -0.03, 1.01],
        vec![0.6, -1.1, 0.4],
        vec![1.0, -2.0, 0.5],
    )
}

fn host(fixed: &Image, moving: &Image) -> MattesMutualInformationMetric {
    MattesMutualInformationMetric::new(fixed, moving, BINS).unwrap()
}

fn device(fixed: &Image, moving: &Image) -> (DeviceImage, DeviceImage, DeviceMattesMetric) {
    let d_f = DeviceImage::upload(fixed).unwrap();
    let d_m = DeviceImage::upload(moving).unwrap();
    let metric = DeviceMattesMetric::from_device(&d_f, &d_m, BINS).unwrap();
    (d_f, d_m, metric)
}

/// **The interpolated moving value reaches a discrete decision.** This is the measurement
/// that makes every bit-identity claim below falsifiable.
///
/// The Parzen bin is `(long long)(mv / binSize − normalizedMin)` — a *truncation*. So a
/// sample whose term sits within one ulp of an integer will land in a different bin if
/// `mv` moves by a single bit. Before Mattes, the device sampler deliberately left the
/// trilinear arithmetic FMA-contracted, on the argument that "everything downstream of
/// the continuous index is continuous in it" — true for mean squares and correlation,
/// which only ever *add* the value, and false here.
///
/// This counts the population: how many samples would change histogram cell under a
/// 1-ulp perturbation of `mv`. It asserts the count is **nonzero**, because a pin that
/// cannot fail when the code is wrong is not a pin.
#[test]
fn the_interpolated_value_reaches_a_discrete_decision_so_the_sampler_pin_is_load_bearing() {
    let n = 32;
    let fixed = commensurate(n);

    // The host metric's own geometry, re-derived here from the images' ranges — this is a
    // measurement, not a pin, so restating the formula is honest.
    let vals = fixed.to_f64_vec().unwrap();
    let (lo, hi) = vals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let denom = (BINS - 2 * PADDING) as f64;
    let bin_size = (hi - lo) / denom;
    let normalized_min = lo / bin_size - PADDING as f64;

    // The identity transform: every fixed voxel maps exactly onto a moving voxel, so the
    // trilinear interpolant returns a stored value exactly and the term lands on the grid
    // of bin edges.
    let mut straddles = 0usize;
    let mut samples = 0usize;
    for &mv in &vals {
        samples += 1;
        let term = mv / bin_size - normalized_min;
        let bin = term as i64;
        // One ulp up and one ulp down on the interpolated value.
        let up = (f64::from_bits(mv.to_bits() + 1) / bin_size - normalized_min) as i64;
        let down =
            (f64::from_bits(mv.to_bits().wrapping_sub(1)) / bin_size - normalized_min) as i64;
        if up != bin || down != bin {
            straddles += 1;
        }
    }

    assert!(
        straddles > 0,
        "no sample sits within one ulp of a Parzen bin boundary on this data, so the \
         sampler's bit-identity pin would be untestable — the fixture is wrong, not the code"
    );
    eprintln!(
        "straddle population: {straddles} of {samples} samples change Parzen bin under a \
         1-ulp move of the interpolated value"
    );
}

/// **The device value is the host value, on the bits.**
///
/// Not "agrees to 1e-15". `to_bits()` equality, at several transforms, including the
/// identity on commensurate geometry — the case where the straddles measured above are
/// dense.
#[test]
fn the_device_value_is_the_host_value_bit_for_bit() {
    if no_device() {
        return;
    }
    for (fixed, moving) in [
        (volume(32, [0.0; 3]), inverted(32, [1.5, -0.5, 0.75])),
        (commensurate(32), commensurate(32)),
    ] {
        let h = host(&fixed, &moving);
        let (_f, _m, d) = device(&fixed, &moving);

        for t in [
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.7, -1.3, 2.1],
            vec![-3.4, 2.2, -0.9],
        ] {
            let transform = TranslationTransform::new(t.clone());
            let want = h.value(&transform);
            let got = d.value(&transform).unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "translation {t:?}: device {got:?} vs host {want:?} (diff {})",
                got - want
            );
        }
    }
}

/// The value is bit-identical for a **composed, multi-stage** point map too — the device
/// replays the stages rather than folding them, and an affine has a matrix the identity
/// does not exercise.
#[test]
fn the_device_value_is_the_host_value_for_an_affine_and_a_rigid_transform() {
    if no_device() {
        return;
    }
    let fixed = volume(32, [0.0; 3]);
    let moving = inverted(32, [1.5, -0.5, 0.75]);
    let h = host(&fixed, &moving);
    let (_f, _m, d) = device(&fixed, &moving);

    let affine = affine();
    let want = h.value(&affine);
    let got = d.value(&affine).unwrap();
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "affine: device {got} vs host {want}"
    );

    let euler = Euler3DTransform::new(0.05, -0.03, 0.08, [1.2, -0.7, 0.4], [0.0, 0.0, 0.0]);
    let want = h.value(&euler);
    let got = d.value(&euler).unwrap();
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "euler: device {got} vs host {want}"
    );
}

/// The device counts exactly the samples the host counts. A straddle at the
/// `is_inside` boundary would move this and nothing else, so it is worth its own pin.
#[test]
fn the_valid_point_count_is_the_hosts_exactly() {
    if no_device() {
        return;
    }
    let fixed = volume(32, [0.0; 3]);
    let moving = inverted(32, [1.5, -0.5, 0.75]);
    let h = host(&fixed, &moving);
    let (_f, _m, d) = device(&fixed, &moving);

    for t in [vec![0.0, 0.0, 0.0], vec![2.5, -3.5, 1.5]] {
        let transform = TranslationTransform::new(t.clone());
        assert_eq!(
            d.evaluate(&transform).unwrap().valid_points,
            h.evaluate(&transform).valid_points,
            "translation {t:?}"
        );
    }
}

/// **Deterministic**: the same binary, the same input, the same bits — value *and*
/// derivative — on every call.
///
/// This is the property the atomic histogram cannot have
/// (`histogram_determinism.rs::the_atomic_histogram_is_not_deterministic_and_that_is_why_this_module_exists`),
/// and the reason the metric waited for the counting sort.
#[test]
fn the_device_metric_is_bit_identical_from_call_to_call() {
    if no_device() {
        return;
    }
    let fixed = volume(32, [0.0; 3]);
    let moving = inverted(32, [1.5, -0.5, 0.75]);
    let (_f, _m, d) = device(&fixed, &moving);
    let transform = TranslationTransform::new(vec![0.7, -1.3, 2.1]);

    let first = d.evaluate(&transform).unwrap();
    for run in 1..8 {
        let again = d.evaluate(&transform).unwrap();
        assert_eq!(
            again.value.to_bits(),
            first.value.to_bits(),
            "run {run}: the value moved"
        );
        for (k, (a, b)) in again.derivative.iter().zip(&first.derivative).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "run {run}, parameter {k}: the derivative moved"
            );
        }
    }
}

/// **The derivative is banded at 1e-9 relative**, and the band is defended in
/// `DeviceMattesMetric`'s docs: nothing discrete depends on the Jacobian, a structural
/// defect is `O(1)`, and a single misplaced sample would be ~4e-6 — three orders of
/// magnitude above this.
#[test]
fn the_device_derivative_is_the_hosts_to_within_the_probed_jacobian_band() {
    if no_device() {
        return;
    }
    let fixed = volume(32, [0.0; 3]);
    let moving = inverted(32, [1.5, -0.5, 0.75]);
    let h = host(&fixed, &moving);
    let (_f, _m, d) = device(&fixed, &moving);

    let affine = affine();

    for transform in [
        &TranslationTransform::new(vec![0.7, -1.3, 2.1]) as &dyn ParametricTransform,
        &affine,
    ] {
        let want = h.evaluate(transform);
        let got = d.evaluate(transform).unwrap();

        // The value half is still held to the bits, even inside the derivative call.
        assert_eq!(
            got.value.to_bits(),
            want.value.to_bits(),
            "the value must not soften because the derivative was asked for"
        );

        let scale = want
            .derivative
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(1e-12);
        for (k, (g, w)) in got.derivative.iter().zip(&want.derivative).enumerate() {
            let rel = (g - w).abs() / scale;
            assert!(
                rel < 1e-9,
                "parameter {k}: device {g} vs host {w} — relative {rel:.3e} exceeds the band"
            );
        }
    }
}

/// A transform whose `transform_point` is not `mat_vec(matrix, p) + offset` on its own
/// stored fields is refused **by name**, for the value as well as the derivative — a
/// centred `ScaleTransform` computes `(p − c)·s + c`, which is that map in exact
/// arithmetic and not in the last bits.
#[test]
fn a_scale_transform_is_refused_by_name_not_approximated() {
    if no_device() {
        return;
    }
    let fixed = volume(16, [0.0; 3]);
    let moving = inverted(16, [1.0, 0.0, 0.0]);
    let (_f, _m, d) = device(&fixed, &moving);

    let scale = ScaleTransform::new(vec![1.05, 0.98, 1.01], vec![0.5, -0.25, 0.75]);

    assert!(
        matches!(d.value(&scale), Err(DeviceMetricError::NoBitwisePointMap)),
        "the value path must refuse a scale transform by name"
    );
    assert!(
        matches!(
            d.evaluate(&scale),
            Err(DeviceMetricError::NoBitwisePointMap)
        ),
        "the derivative path must refuse it by the same name"
    );
}

/// A B-spline is refused **by name**, on both paths — and the name is `NoBitwisePointMap`,
/// not `NonAffineTransform`, which is worth pinning because it is not what the code's own
/// comment used to claim.
///
/// Measured, not assumed: `affine_form` probes the Jacobian at two fixed points, and both
/// lie outside a B-spline's support, where its Jacobian is **identically zero**. A zero
/// Jacobian is trivially affine in the point, so the probe *passes* — vacuously. The
/// transform is caught one line later, by the point-map test, which it fails for real
/// (`BSplineTransform` reports no stages). So nothing is approximated and nothing leaks to
/// the wrong kernel; but the refusal that fires is the point map's, and the comment in
/// `cuda.rs::affine_form` that said a B-spline "fails both tests" was wrong. It now says
/// what this test measures.
#[test]
fn a_bspline_transform_is_refused_by_name_and_the_name_is_the_point_maps() {
    if no_device() {
        return;
    }
    let fixed = volume(16, [0.0; 3]);
    let moving = inverted(16, [1.0, 0.0, 0.0]);
    let (_f, _m, d) = device(&fixed, &moving);

    let t = BSplineTransform::from_image_domain(&fixed, &[3, 3, 3]).unwrap();
    assert!(matches!(
        d.evaluate(&t),
        Err(DeviceMetricError::NoBitwisePointMap)
    ));
    assert!(matches!(
        d.value(&t),
        Err(DeviceMetricError::NoBitwisePointMap)
    ));
}

/// The device metric refuses what the host metric refuses, through the host metric's own
/// code: it derives the histogram geometry by calling `MattesGeometry::new`, so a constant
/// image or too few bins raises the host's error, not a device-flavoured re-statement of
/// it.
#[test]
fn the_host_metrics_refusals_are_the_device_metrics_refusals() {
    if no_device() {
        return;
    }
    let fixed = volume(16, [0.0; 3]);
    let moving = inverted(16, [1.0, 0.0, 0.0]);
    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();

    assert!(matches!(
        DeviceMattesMetric::from_device(&d_f, &d_m, 4),
        Err(DeviceMetricError::Cuda(CudaError::HistogramShape(_)))
            | Err(DeviceMetricError::MattesGeometry(
                RegistrationError::TooFewHistogramBins { .. }
            ))
    ));

    let flat = Image::from_vec(&[16, 16, 16], vec![3.0f32; 16 * 16 * 16]).unwrap();
    let d_flat = DeviceImage::upload(&flat).unwrap();
    assert!(matches!(
        DeviceMattesMetric::from_device(&d_flat, &d_m, BINS),
        Err(DeviceMetricError::MattesGeometry(
            RegistrationError::ConstantIntensity { which: "fixed" }
        ))
    ));
    assert!(matches!(
        DeviceMattesMetric::from_device(&d_f, &d_flat, BINS),
        Err(DeviceMetricError::MattesGeometry(
            RegistrationError::ConstantIntensity { which: "moving" }
        ))
    ));
}

/// End to end: `execute_on_device` now takes the Mattes metric, and registers a volume
/// against its **contrast-inverted** counterpart — the multi-modality case mean squares
/// cannot do at all, which is the whole reason this metric exists.
///
/// The claim is not "it converges". It is that the **device run lands where the host run
/// lands**: same descent, same answer. The metric value is bit-identical at every
/// iteration, and the derivative agrees to the probed-Jacobian band, so a well-scaled
/// descent has nothing left to diverge on.
#[test]
fn execute_on_device_registers_a_contrast_inverted_pair_with_mattes() {
    if no_device() {
        return;
    }
    use sitk_registration::{EstimateLearningRate, ImageRegistrationMethod};

    let n = 32;
    let fixed = volume(n, [0.0; 3]);
    let moving = inverted(n, [-2.0, 1.0, 0.0]);

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mattes_mutual_information(BINS)
        .set_optimizer_scales_from_physical_shift()
        .set_optimizer_as_regular_step_gradient_descent_estimated(
            0.01,
            200,
            1e-6,
            EstimateLearningRate::Once,
        );

    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();

    let on_device = reg
        .execute_on_device(&d_f, &d_m, TranslationTransform::new(vec![0.0, 0.0, 0.0]))
        .expect("the device path must take the Mattes metric");
    let on_host = reg
        .execute(
            &fixed,
            &moving,
            TranslationTransform::new(vec![0.0, 0.0, 0.0]),
        )
        .expect("the host path is the reference");

    let dp = on_device.transform.parameters();
    let hp = on_host.transform.parameters();

    // It actually registered: the answer is not the identity it started from.
    let moved = dp.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(moved > 0.5, "the device run did not move: {dp:?}");

    // And it registered to the *same place* the host did.
    for (k, (&d, &h)) in dp.iter().zip(hp.iter()).enumerate() {
        assert!(
            (d - h).abs() < 1e-3,
            "parameter {k}: device {d} vs host {h} (device {dp:?}, host {hp:?})"
        );
    }
    assert_eq!(
        on_device.iterations, on_host.iterations,
        "the two paths took different numbers of steps: device {dp:?}, host {hp:?}"
    );
}
