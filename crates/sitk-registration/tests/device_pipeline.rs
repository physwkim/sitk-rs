//! The device-resident pipeline: `upload → rescale → metric`, with nothing
//! crossing the bus in the middle.
//!
//! What is under test is the *residency contract*, not the kernels — those are
//! already covered by `cuda_mean_squares.rs` and `sitk-cuda`'s own tests. Here:
//! a device op produces what the host filter produces; a metric built from device
//! images produces what the host metric produces; the crossing is refused, by name,
//! for a pixel type the device does not have; and the answer is bit-identical run
//! to run.
//!
//! Only compiled with the `cuda` feature.
#![cfg(feature = "cuda")]

use sitk_core::Image;
use sitk_cuda::{CudaError, DeviceImage};
use sitk_registration::metric::{FixedSamples, MovingImage};
use sitk_registration::{
    CpuBackend, DeviceMeanSquaresMetric, DeviceMetricError, DeviceRegistrationError,
    ImageRegistrationMethod, MeanSquaresMetric,
};
use sitk_transform::{BSplineTransform, Euler3DTransform, ParametricTransform};

const OUT_MIN: f64 = 0.0;
const OUT_MAX: f64 = 255.0;

/// Smooth, textured `f32` volume — the same shape the CUDA metric tests use, so a
/// failure here is about residency rather than about the data.
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

fn probe_transform(n: usize) -> Euler3DTransform {
    let c = n as f64 / 2.0;
    Euler3DTransform::new(0.06, -0.04, 0.03, [2.5, -1.5, 0.75], [c, c, c])
}

fn no_device() -> bool {
    matches!(sitk_cuda::backend(), Err(CudaError::NoDevice(_)))
}

// The crossing's pixel-type contract — every scalar type casts on the device,
// bit-identically to `sitk_filters::cast`, and a type with no device path is
// refused by name without touching the driver — lives in `tests/upload_cast.rs`.

/// `upload` → `to_host` is the identity on voxels *and* on geometry: a volume that
/// makes the round trip must come back as the image it was.
#[test]
fn upload_then_to_host_round_trips_voxels_and_geometry() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(32, [0.0; 3]);
    let back = DeviceImage::upload(&img).unwrap().to_host().unwrap();

    assert_eq!(back.size(), img.size());
    assert_eq!(back.spacing(), img.spacing());
    assert_eq!(back.origin(), img.origin());
    assert_eq!(back.direction(), img.direction());
    assert_eq!(
        back.scalar_slice::<f32>().unwrap(),
        img.scalar_slice::<f32>().unwrap(),
        "the round trip changed a voxel"
    );
}

/// The resident op computes what the host filter computes.
///
/// Tolerance **1e-6 relative on the f32 result**, and the measured error is
/// printed. The two paths perform the same arithmetic — widen to `f64`, exact
/// `min`/`max`, `(v − lo)·scale + out_min`, narrow to `f32` round-to-nearest-even —
/// so they are expected to agree exactly; the band is there so that a *rounding*
/// difference in the reduction could never be mistaken for a modelling one, and the
/// count of bit-exact voxels is asserted separately below.
#[test]
fn a_resident_rescale_matches_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(64, [0.0; 3]);
    let cpu = sitk_filters::rescale_intensity(&img, OUT_MIN, OUT_MAX).unwrap();

    let d_in = DeviceImage::upload(&img).unwrap();
    let d_out = sitk_cuda::rescale_intensity(&d_in, OUT_MIN, OUT_MAX).unwrap();
    let gpu = d_out.to_host().unwrap();

    let (a, b) = (
        cpu.scalar_slice::<f32>().unwrap(),
        gpu.scalar_slice::<f32>().unwrap(),
    );
    assert_eq!(a.len(), b.len());

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut exact = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.to_bits() == y.to_bits() {
            exact += 1;
        }
        let (x, y) = (x as f64, y as f64);
        max_abs = max_abs.max((x - y).abs());
        max_rel = max_rel.max((x - y).abs() / (1.0 + x.abs()));
    }
    println!(
        "resident rescale vs host filter: max_abs {max_abs:e}, max_rel {max_rel:e}, \
         bit-exact {exact}/{}",
        a.len()
    );
    assert!(max_rel <= 1e-6, "max_rel {max_rel:e} exceeds 1e-6");
    assert_eq!(
        exact,
        a.len(),
        "the two paths do the same arithmetic and must agree bit for bit"
    );
}

/// A metric built from **device images** produces the host metric's answer.
///
/// Tolerance **1e-9 relative**, the same band the uploading GPU backend is held to
/// and for the same reason: the CPU sums N per-sample terms left to right and no
/// parallel reduction reproduces that order, so the divergence is reduction
/// rounding (~√N·ε). Anything larger is a modelling difference and fails here.
#[test]
fn a_metric_built_from_device_images_matches_the_cpu_metric() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let t = probe_transform(n);

    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&fixed).unwrap(),
        MovingImage::from_image(&moving).unwrap(),
    )
    .unwrap();
    let cpu = host.evaluate(&t, &CpuBackend);

    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();
    let device = DeviceMeanSquaresMetric::from_device(&d_fixed, &d_moving).unwrap();
    let gpu = device.evaluate(&t).unwrap();

    assert_eq!(device.sample_count(), n * n * n);
    assert_eq!(
        gpu.valid_points, cpu.valid_points,
        "the device metric walked a different valid-sample set"
    );
    assert!(cpu.valid_points > 0);

    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_err = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .fold(0.0f64, f64::max);
    println!("value rel err {v_err:e} | derivative rel err {d_err:e}");
    assert!(v_err <= 1e-9, "value rel err {v_err:e} exceeds 1e-9");
    assert!(d_err <= 1e-9, "derivative rel err {d_err:e} exceeds 1e-9");
    assert!(
        cpu.derivative.iter().any(|d| d.abs() > 1e-6),
        "the CPU derivative is ~zero here, so the comparison proves nothing"
    );
}

/// The whole chain, with the host absent from the middle: upload both volumes once,
/// rescale both **on the device**, and register against the buffers the filters
/// produced. Must equal the host chain (rescale on the CPU, metric on the CPU) to
/// the same reduction-rounding band.
#[test]
fn the_resident_chain_agrees_with_the_host_chain() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let t = probe_transform(n);

    // Host chain.
    let f = sitk_filters::rescale_intensity(&fixed, OUT_MIN, OUT_MAX).unwrap();
    let m = sitk_filters::rescale_intensity(&moving, OUT_MIN, OUT_MAX).unwrap();
    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&f).unwrap(),
        MovingImage::from_image(&m).unwrap(),
    )
    .unwrap();
    let cpu = host.evaluate(&t, &CpuBackend);

    // Device chain: two uploads, two kernels, no download.
    let d_f = sitk_cuda::rescale_intensity(&DeviceImage::upload(&fixed).unwrap(), OUT_MIN, OUT_MAX)
        .unwrap();
    let d_m =
        sitk_cuda::rescale_intensity(&DeviceImage::upload(&moving).unwrap(), OUT_MIN, OUT_MAX)
            .unwrap();
    let device = DeviceMeanSquaresMetric::from_device(&d_f, &d_m).unwrap();
    let gpu = device.evaluate(&t).unwrap();

    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_err = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .fold(0.0f64, f64::max);
    println!("chain: value rel err {v_err:e} | derivative rel err {d_err:e}");
    assert_eq!(gpu.valid_points, cpu.valid_points);
    assert!(v_err <= 1e-9, "value rel err {v_err:e} exceeds 1e-9");
    assert!(d_err <= 1e-9, "derivative rel err {d_err:e} exceeds 1e-9");
}

/// Run-to-run **bit-identity** of the device metric, asserted exactly — the same
/// correctness property the uploading backend is held to. An optimizer is a
/// feedback loop; a metric that moves in its last ulp between runs moves the
/// registration result.
#[test]
fn the_device_metric_is_bit_identical_run_to_run() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let t = probe_transform(n);

    let bits = |v: &sitk_registration::MetricValue| {
        (
            v.value.to_bits(),
            v.derivative.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
            v.valid_points,
        )
    };

    // A fresh upload and a fresh metric each round, so nothing can be memoized.
    let first = {
        let d_f = DeviceImage::upload(&fixed).unwrap();
        let d_m = DeviceImage::upload(&moving).unwrap();
        bits(
            &DeviceMeanSquaresMetric::from_device(&d_f, &d_m)
                .unwrap()
                .evaluate(&t)
                .unwrap(),
        )
    };
    for run in 1..6 {
        let d_f = DeviceImage::upload(&fixed).unwrap();
        let d_m = DeviceImage::upload(&moving).unwrap();
        let metric = DeviceMeanSquaresMetric::from_device(&d_f, &d_m).unwrap();
        assert_eq!(
            bits(&metric.evaluate(&t).unwrap()),
            first,
            "run {run}: the device metric moved"
        );
        // And twice within one resident metric.
        assert_eq!(bits(&metric.evaluate(&t).unwrap()), first);
    }
    println!("6 fresh uploads + repeated evaluations: all bit-identical");
}

/// The refusal is **named**, not silent. A B-spline's point map and Jacobian are not
/// affine in the point, so the moment identity the kernel evaluates does not hold —
/// and this metric says so rather than quietly running something else. There is no
/// per-call CPU fallback here on purpose; the caller owns that decision.
#[test]
fn a_non_affine_transform_is_refused_by_name() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 32;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();
    let metric = DeviceMeanSquaresMetric::from_device(&d_f, &d_m).unwrap();

    let mut t = BSplineTransform::new(
        3,
        &[0.0, 0.0, 0.0],
        &[n as f64, n as f64, n as f64],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[4, 4, 4],
    )
    .unwrap();
    let np = t.number_of_parameters();
    let coeffs: Vec<f64> = (0..np)
        .map(|k| 3.0 * ((k as f64) * 0.7).sin() + 1.5 * ((k as f64) * 0.13).cos())
        .collect();
    t.set_parameters(&coeffs).unwrap();

    match metric.evaluate(&t) {
        Err(DeviceMetricError::NonAffineTransform) => {
            println!("refused: {}", DeviceMetricError::NonAffineTransform);
        }
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("the device metric has no kernel for a B-spline and must say so"),
    }

    // The affine transform it *does* have a kernel for still works on the same metric.
    assert!(metric.evaluate(&probe_transform(n)).is_ok());
}

/// The device Gaussian is **bit-identical** to `sitk_filters::smooth_gaussian`.
///
/// Not a tolerance: the two paths perform the same operations in the same order —
/// the same host-computed `f64` weights, `f64` intermediates between axes, the same
/// clamped boundary, and no FMA contraction on the device (`__dmul_rn` /
/// `__dadd_rn`). Anything that breaks that equality is a real divergence, not
/// rounding, and this test is where it should be caught.
#[test]
fn the_device_gaussian_is_bit_identical_to_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48, [0.0; 3]);
    let sigma = [1.5, 2.0, 1.0];

    let cpu = sitk_filters::smooth_gaussian(&img, &sigma).unwrap();
    let gpu = sitk_cuda::smooth_gaussian(&DeviceImage::upload(&img).unwrap(), &sigma)
        .unwrap()
        .to_host()
        .unwrap();

    let (a, b) = (
        cpu.scalar_slice::<f32>().unwrap(),
        gpu.scalar_slice::<f32>().unwrap(),
    );
    let differing = a
        .iter()
        .zip(b.iter())
        .filter(|&(&x, &y)| x.to_bits() != y.to_bits())
        .count();
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0f64, f64::max);
    println!(
        "device gaussian vs host filter: {differing}/{} voxels differ, max_abs {max_abs:e}",
        a.len()
    );
    assert_eq!(
        differing, 0,
        "the device Gaussian diverged from the CPU filter"
    );
}

/// A zero `sigma` axis is untouched on the device too, as on the CPU.
#[test]
fn the_device_gaussian_leaves_a_zero_sigma_axis_alone() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(32, [0.0; 3]);
    let cpu = sitk_filters::smooth_gaussian(&img, &[0.0, 0.0, 0.0]).unwrap();
    let gpu = sitk_cuda::smooth_gaussian(&DeviceImage::upload(&img).unwrap(), &[0.0, 0.0, 0.0])
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(
        cpu.scalar_slice::<f32>().unwrap(),
        gpu.scalar_slice::<f32>().unwrap()
    );
    assert_eq!(
        gpu.scalar_slice::<f32>().unwrap(),
        img.scalar_slice::<f32>().unwrap(),
        "zero sigma must be the identity"
    );
}

/// The pipeline the user actually described, with nothing crossing the bus in the
/// middle: `upload → rescale → smooth → register`, against the same chain on the
/// host. Same reduction-rounding band as every other GPU-vs-CPU metric comparison.
#[test]
fn the_full_resident_pipeline_agrees_with_the_host_pipeline() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let sigma = [1.0, 1.0, 1.0];
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let t = probe_transform(n);

    let host_chain = |img: &Image| {
        let r = sitk_filters::rescale_intensity(img, OUT_MIN, OUT_MAX).unwrap();
        sitk_filters::smooth_gaussian(&r, &sigma).unwrap()
    };
    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&host_chain(&fixed)).unwrap(),
        MovingImage::from_image(&host_chain(&moving)).unwrap(),
    )
    .unwrap();
    let cpu = host.evaluate(&t, &CpuBackend);

    let device_chain = |img: &Image| {
        let d = DeviceImage::upload(img).unwrap();
        let r = sitk_cuda::rescale_intensity(&d, OUT_MIN, OUT_MAX).unwrap();
        sitk_cuda::smooth_gaussian(&r, &sigma).unwrap()
    };
    let d_f = device_chain(&fixed);
    let d_m = device_chain(&moving);
    let device = DeviceMeanSquaresMetric::from_device(&d_f, &d_m).unwrap();
    let gpu = device.evaluate(&t).unwrap();

    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_err = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .fold(0.0f64, f64::max);
    println!("full pipeline: value rel err {v_err:e} | derivative rel err {d_err:e}");
    assert_eq!(gpu.valid_points, cpu.valid_points);
    assert!(v_err <= 1e-9, "value rel err {v_err:e} exceeds 1e-9");
    assert!(d_err <= 1e-9, "derivative rel err {d_err:e} exceeds 1e-9");
    assert!(cpu.derivative.iter().any(|d| d.abs() > 1e-6));
}

/// The optimizer drives the device metric through the **public API**, and lands
/// where the host `execute` lands — *at this size, from this start*.
///
/// `execute_on_device` runs the same optimizer, the same scales estimator and the
/// same convergence test as `execute` — it is literally the same driver — so
/// agreement here says the device metric steers the feedback loop the same way, not
/// merely that one evaluation agrees.
///
/// **Do not read this as a promise that the endpoints always match.** They do not.
/// A 1e-12 difference in the derivative can flip a step-halving decision, and at
/// 256³ with unit scales the two runs converge to different local minima 7.5e-3
/// apart — see `execute_on_device`'s docs. What *is* guaranteed is pinned by
/// `the_device_metric_is_the_same_objective_as_the_host_metric` below, and what a
/// well-conditioned run does is pinned by
/// `a_well_conditioned_run_lands_in_the_same_place_on_both_paths`.
#[test]
fn execute_on_device_lands_where_execute_lands() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let c = n as f64 / 2.0;
    let initial = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mean_squares();
    reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-4, 25, 1e-8);

    let host = reg.execute(&fixed, &moving, initial()).unwrap();
    let device = reg
        .execute_on_device(
            &DeviceImage::upload(&fixed).unwrap(),
            &DeviceImage::upload(&moving).unwrap(),
            initial(),
        )
        .unwrap();

    println!(
        "host   : {} iters, metric {:.9}, params {:?}",
        host.iterations,
        host.metric_value,
        host.transform.parameters()
    );
    println!(
        "device : {} iters, metric {:.9}, params {:?}",
        device.iterations,
        device.metric_value,
        device.transform.parameters()
    );
    assert_eq!(
        device.iterations, host.iterations,
        "the two runs took different paths through the optimizer"
    );
    assert_eq!(device.valid_points, host.valid_points);
    for (k, (&d, &h)) in device
        .transform
        .parameters()
        .iter()
        .zip(host.transform.parameters().iter())
        .enumerate()
    {
        let rel = (d - h).abs() / (1.0 + h.abs());
        assert!(
            rel <= 1e-6,
            "param {k}: device {d:e} vs host {h:e} (rel {rel:e}) — the device metric steered \
             the optimizer somewhere else"
        );
    }
}

/// Every condition the device path cannot take is refused **at the boundary, by
/// name**, before the first iteration — so the caller can run the host method
/// instead. This is the fallback, and it is the only one: nothing inside the
/// optimizer loop asks where the metric lives.
#[test]
fn the_device_entry_point_refuses_at_the_boundary_by_name() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 32;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();
    let c = n as f64 / 2.0;
    let euler = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

    let run = |reg: &ImageRegistrationMethod| reg.execute_on_device(&d_f, &d_m, euler());

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mattes_mutual_information(32);
    assert!(matches!(
        run(&reg),
        Err(DeviceRegistrationError::UnsupportedMetric)
    ));

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mean_squares();
    reg.set_interpolator(sitk_transform::Interpolator::BSpline);
    assert!(matches!(
        run(&reg),
        Err(DeviceRegistrationError::UnsupportedInterpolator(_))
    ));

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_sampling_strategy(sitk_registration::metric::SamplingStrategy::Random);
    reg.set_metric_sampling_percentage(0.2, 7);
    assert!(matches!(
        run(&reg),
        Err(DeviceRegistrationError::UnsupportedSampling)
    ));

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_fixed_mask(&fixed);
    assert!(matches!(
        run(&reg),
        Err(DeviceRegistrationError::UnsupportedMask)
    ));

    // And a transform the moment identity does not cover: refused before the run,
    // carrying the metric's own named error rather than a generic failure.
    let reg = ImageRegistrationMethod::new();
    let mut bs = BSplineTransform::new(
        3,
        &[0.0, 0.0, 0.0],
        &[n as f64, n as f64, n as f64],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[4, 4, 4],
    )
    .unwrap();
    let np = bs.number_of_parameters();
    bs.set_parameters(
        &(0..np)
            .map(|k| ((k as f64) * 0.7).sin())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(matches!(
        reg.execute_on_device(&d_f, &d_m, bs),
        Err(DeviceRegistrationError::Metric(
            DeviceMetricError::NonAffineTransform
        ))
    ));
    println!("five refusals, each by name, all before the first iteration");
}

/// **The guarantee**, pinned: the device metric is the *same objective* as the host
/// metric — not "the same optimizer endpoint", which is a different and weaker claim
/// (see `execute_on_device`'s docs).
///
/// At any transform, the value and the derivative must agree to reduction-rounding
/// (~1e-12 measured; the band is 1e-9) and the valid-sample count must agree
/// *exactly*. This is the property every other guarantee rests on, and it is the one
/// a future kernel change must not break — a derivative that drifts to 1e-6 would
/// still pass a trajectory test on a benign case while quietly steering real
/// registrations somewhere else.
#[test]
fn the_device_metric_is_the_same_objective_as_the_host_metric() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));

    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&fixed).unwrap(),
        MovingImage::from_image(&moving).unwrap(),
    )
    .unwrap();
    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();
    let device = DeviceMeanSquaresMetric::from_device(&d_fixed, &d_moving).unwrap();

    let c = n as f64 / 2.0;
    // Spread across the space the optimizer actually walks: the identity, a pure
    // translation, a pure rotation, a mixed pose, and a pose far enough out that a
    // large part of the fixed grid maps outside the moving image.
    let transforms = [
        Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]),
        Euler3DTransform::new(0.0, 0.0, 0.0, [3.0, -2.0, 1.5], [c, c, c]),
        Euler3DTransform::new(0.11, -0.07, 0.05, [0.0; 3], [c, c, c]),
        Euler3DTransform::new(0.06, -0.04, 0.03, [2.5, -1.5, 0.75], [c, c, c]),
        Euler3DTransform::new(-0.25, 0.18, -0.12, [9.0, -7.0, 5.0], [c, c, c]),
    ];

    let mut worst_v = 0.0f64;
    let mut worst_d = 0.0f64;
    for (k, t) in transforms.iter().enumerate() {
        let cpu = host.evaluate(t, &CpuBackend);
        let gpu = device.evaluate(t).unwrap();

        assert_eq!(
            gpu.valid_points, cpu.valid_points,
            "transform {k}: the two metrics disagree about which samples are inside \
             ({} vs {})",
            gpu.valid_points, cpu.valid_points
        );
        assert!(cpu.valid_points > 0, "transform {k}: nothing mapped inside");
        assert!(
            cpu.derivative.iter().any(|d| d.abs() > 1e-6),
            "transform {k}: the CPU derivative is ~zero, so the comparison proves nothing"
        );

        let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
        let v = rel(gpu.value, cpu.value);
        let d = gpu
            .derivative
            .iter()
            .zip(cpu.derivative.iter())
            .map(|(&g, &c)| rel(g, c))
            .fold(0.0f64, f64::max);
        println!("transform {k}: value rel {v:e}, derivative rel {d:e}");
        worst_v = worst_v.max(v);
        worst_d = worst_d.max(d);
    }
    println!(
        "worst over {} transforms: value {worst_v:e}, derivative {worst_d:e}",
        transforms.len()
    );
    assert!(worst_v <= 1e-9, "value rel err {worst_v:e} exceeds 1e-9");
    assert!(
        worst_d <= 1e-9,
        "derivative rel err {worst_d:e} exceeds 1e-9"
    );
}

/// A **well-conditioned** run lands in the same place on both paths.
///
/// With unit scales, a 1-radian rotation step is the same size as a 1-mm translation
/// step; the descent is chaotic, and host and device converge to different local
/// minima (measured 7.5e-3 apart at 256³ — for the host too: it is the conditioning,
/// not the device). Scale the parameters by their physical effect, as a caller
/// registering a rotation should, and the amplification is gone: same iteration
/// count, same valid points, parameters at the rounding floor.
#[test]
fn a_well_conditioned_run_lands_in_the_same_place_on_both_paths() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let c = n as f64 / 2.0;
    let initial = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]);

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mean_squares();
    reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-6, 100, 1e-8);
    reg.set_optimizer_scales_from_physical_shift();

    let host = reg.execute(&fixed, &moving, initial()).unwrap();
    let device = reg
        .execute_on_device(
            &DeviceImage::upload(&fixed).unwrap(),
            &DeviceImage::upload(&moving).unwrap(),
            initial(),
        )
        .unwrap();

    println!(
        "host   : {} iters, {} valid, metric {:.12}",
        host.iterations, host.valid_points, host.metric_value
    );
    println!(
        "device : {} iters, {} valid, metric {:.12}",
        device.iterations, device.valid_points, device.metric_value
    );
    assert_eq!(device.iterations, host.iterations, "different walk lengths");
    assert_eq!(device.valid_points, host.valid_points);

    let worst = device
        .transform
        .parameters()
        .iter()
        .zip(host.transform.parameters().iter())
        .map(|(&d, &h)| (d - h).abs() / (1.0 + h.abs()))
        .fold(0.0f64, f64::max);
    println!("worst parameter disagreement: {worst:e}");
    assert!(
        worst <= 1e-9,
        "a well-conditioned run must land in the same place; worst {worst:e}"
    );
}

/// The pyramid, which the device path used to refuse outright: a three-level
/// schedule, shrunk and smoothed, run on both paths from the same start.
///
/// Each level's fixed image is built on the device by the same three ops
/// `prepare_level` uses, and each of the three is bit-identical to its CPU filter
/// (`pyramid_parity.rs`). This test pins the *composition*: the levels are built in
/// the same order, the transform is carried from level to level the same way, and
/// the run therefore lands in the same place — to the same tolerance the metric is
/// gated at.
///
/// Scales come from the physical-shift estimator for the reason `execute_on_device`
/// documents (and §2.157 records): with unit scales the descent is chaotic on
/// *both* paths and a 1e-12 metric difference is amplified into a different local
/// minimum. That is a property of the parameter space, not of the device, and this
/// test is about the pyramid.
#[test]
fn a_pyramid_run_lands_where_the_host_pyramid_lands() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let c = n as f64 / 2.0;
    let initial = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]);

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mean_squares();
    reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-6, 300, 1e-8);
    reg.set_optimizer_scales_from_physical_shift();
    reg.set_shrink_factors_per_level(vec![4, 2, 1]);
    reg.set_smoothing_sigmas_per_level(vec![2.0, 1.0, 0.0]);

    let host = reg.execute(&fixed, &moving, initial()).unwrap();
    let device = reg
        .execute_on_device(
            &DeviceImage::upload(&fixed).unwrap(),
            &DeviceImage::upload(&moving).unwrap(),
            initial(),
        )
        .expect("the device path must take a pyramid now");

    println!(
        "host   : {} iters, {} valid, metric {:.12}",
        host.iterations, host.valid_points, host.metric_value
    );
    println!(
        "device : {} iters, {} valid, metric {:.12}",
        device.iterations, device.valid_points, device.metric_value
    );

    // The finest level is full resolution, so the valid-sample count is the whole
    // grid on both paths — and if the coarse levels had landed anywhere different,
    // the finest level would have started from a different transform.
    assert_eq!(
        device.valid_points, host.valid_points,
        "the finest level sampled a different number of points"
    );
    assert_eq!(
        device.iterations, host.iterations,
        "the finest level took a different number of steps"
    );

    let worst = device
        .transform
        .parameters()
        .iter()
        .zip(host.transform.parameters().iter())
        .map(|(&d, &h)| (d - h).abs() / (1.0 + h.abs()))
        .fold(0.0f64, f64::max);
    println!("worst parameter disagreement across the pyramid: {worst:e}");
    assert!(
        worst <= 1e-9,
        "a three-level pyramid must land in the same place on both paths; worst {worst:e}"
    );

    // And it actually moved: a pyramid that silently did nothing would also agree.
    let shifted = device.transform.parameters()[3..6]
        .iter()
        .any(|&t| t.abs() > 0.5);
    assert!(
        shifted,
        "the registration did not move; the test proves nothing"
    );
}

/// A sample whose continuous index lands **exactly on a voxel plane** of the moving
/// image, which is where the trilinear interpolant's gradient is discontinuous.
///
/// This is not an exotic input: it is what the identity transform does whenever the
/// two grids are commensurate — a fixed image and a moving image on the same
/// spacing, offset by a whole number of voxels, which is the *starting point* of a
/// great many registrations. Here the moving image's origin is shifted by exactly
/// three voxels along x (2.1 = 3 × 0.7) and by a fraction of one along y and z, so
/// every sample sits on a knot in x and nowhere near one in y or z.
///
/// The value is continuous across the knot and always agreed. The **derivative** is
/// not: whichever side of the plane `floor()` lands on decides which one-sided
/// difference the gradient is. The device must therefore compute the continuous
/// index with the host's exact roundings — the kernel does the multiplies and adds
/// of the point chain with `__dmul_rn`/`__dadd_rn`, in the host's order, so NVRTC
/// cannot fuse them. Before that fix this case was off by **34%** in `d/d(angle_y)`
/// and **8%** in `d/d(translation_x)`, with the value still agreeing to 1e-15.
#[test]
fn the_device_metric_agrees_with_the_host_on_a_sample_that_lands_on_a_voxel_plane() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let spacing = [0.7, 1.1, 1.3];
    let origin = [-12.0, 5.5, 3.25];

    let mut fixed = volume(n, [0.0; 3]);
    fixed.set_spacing(&spacing).unwrap();
    fixed.set_origin(&origin).unwrap();

    let mut moving = volume(n, [3.0, -2.0, 1.5]);
    moving.set_spacing(&spacing).unwrap();
    // + exactly 3 voxels in x; a fraction of a voxel in y and z.
    moving
        .set_origin(&[origin[0] + 2.1, origin[1] - 1.4, origin[2] + 0.8])
        .unwrap();

    let c = n as f64 / 2.0;
    let identity = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]);

    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&fixed).unwrap(),
        MovingImage::from_image(&moving).unwrap(),
    )
    .unwrap();
    let h = host.evaluate(&identity, &CpuBackend);

    let device = DeviceMeanSquaresMetric::from_device(
        &DeviceImage::upload(&fixed).unwrap(),
        &DeviceImage::upload(&moving).unwrap(),
    )
    .unwrap();
    let d = device.evaluate(&identity).unwrap();

    assert_eq!(d.valid_points, h.valid_points);
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    let v = rel(d.value, h.value);
    let g = d
        .derivative
        .iter()
        .zip(h.derivative.iter())
        .map(|(&x, &y)| rel(x, y))
        .fold(0.0f64, f64::max);
    println!("on a voxel plane: value rel {v:e}, derivative rel {g:e}");
    assert!(v <= 1e-9, "value rel err {v:e} exceeds 1e-9");
    assert!(
        g <= 1e-9,
        "derivative rel err {g:e} exceeds 1e-9 — the device took the other \
         one-sided gradient at the knot"
    );
    assert!(
        h.derivative.iter().any(|x| x.abs() > 1e-6),
        "the derivative is ~zero here, so the comparison proves nothing"
    );
}
