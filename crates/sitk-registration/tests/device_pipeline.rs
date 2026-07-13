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
use sitk_cuda::{CudaError, DeviceImage, DeviceMask};
use sitk_registration::metric::{FixedSamples, MovingImage, SamplingStrategy};
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

/// A **half-volume mask**: an ellipsoid centred off-axis, so it drops samples both at
/// the grid's border and in its interior and is not reproducible by any stride mistake.
fn ellipsoid_mask(n: usize) -> Image {
    let c = n as f64 / 2.0;
    let v: Vec<f32> = (0..n * n * n)
        .map(|s| {
            let (i, j, k) = (s % n, (s / n) % n, s / (n * n));
            let (x, y, z) = (
                (i as f64 - c + 4.0) / (0.42 * n as f64),
                (j as f64 - c - 3.0) / (0.34 * n as f64),
                (k as f64 - c + 2.0) / (0.38 * n as f64),
            );
            f32::from(u8::from(x * x + y * y + z * z <= 1.0))
        })
        .collect();
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[0.8, 0.9, 1.1]).unwrap();
    img.set_origin(&[-3.0, 2.0, 1.0]).unwrap();
    img
}

/// A **masked** device metric against the **masked** host metric.
///
/// The host drops a masked-out voxel from `FixedSamples` — it never becomes a sample.
/// The device keeps the full grid and skips the voxel inside the kernel's grid-stride
/// loop. Those are the same set of terms, so the two must agree on the **valid-point
/// count exactly** (this is what would catch an off-by-one in the flat index, or a mask
/// read with the moving image's strides), and on value and derivative to the same
/// **1e-9 relative** reduction-rounding band the unmasked path lives in — nothing here
/// changes the arithmetic of a surviving term.
///
/// Bit-identity to the host is *not* claimed and never was: the host sums the surviving
/// terms left to right and the device reduces them in a fixed tree.
#[test]
fn a_masked_device_metric_matches_the_masked_host_metric() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let mask = ellipsoid_mask(n);
    let t = probe_transform(n);

    let kept = mask
        .scalar_slice::<f32>()
        .unwrap()
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    assert!(
        kept > 0 && kept < n * n * n,
        "the mask must drop some voxels and keep some; kept {kept} of {}",
        n * n * n
    );

    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image_with(&fixed, SamplingStrategy::None, 1.0, 0, Some(&mask)).unwrap(),
        MovingImage::from_image(&moving).unwrap(),
    )
    .unwrap();
    let cpu = host.evaluate(&t, &CpuBackend);

    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();
    let d_mask = DeviceMask::upload(&mask).unwrap();
    let device =
        DeviceMeanSquaresMetric::from_device_masked(&d_fixed, &d_moving, Some(&d_mask), None)
            .unwrap();
    let gpu = device.evaluate(&t).unwrap();

    // The mask drops samples inside the walk; it does not shrink the grid.
    assert_eq!(device.sample_count(), n * n * n);
    assert_eq!(
        gpu.valid_points, cpu.valid_points,
        "the masked device metric walked a different valid-sample set than the host"
    );
    assert!(
        cpu.valid_points > 0 && cpu.valid_points <= kept,
        "valid points {} against {kept} masked-in voxels",
        cpu.valid_points
    );

    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_err = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .fold(0.0f64, f64::max);
    println!(
        "masked: valid_points {} of {kept} masked-in | value rel err {v_err:e} | \
         derivative rel err {d_err:e}",
        cpu.valid_points
    );
    assert!(v_err <= 1e-9, "value rel err {v_err:e} exceeds 1e-9");
    assert!(d_err <= 1e-9, "derivative rel err {d_err:e} exceeds 1e-9");
    assert!(
        cpu.derivative.iter().any(|d| d.abs() > 1e-6),
        "the CPU derivative is ~zero here, so the comparison proves nothing"
    );
}

/// The mask has to *do* something: the masked metric must differ from the unmasked one
/// on the same volumes. Without this, a mask silently ignored by the kernel would pass
/// every agreement test above by matching a host metric that was also ignoring it.
#[test]
fn a_mask_changes_the_answer() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let t = probe_transform(n);
    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();
    let d_mask = DeviceMask::upload(&ellipsoid_mask(n)).unwrap();

    let unmasked = DeviceMeanSquaresMetric::from_device(&d_fixed, &d_moving)
        .unwrap()
        .evaluate(&t)
        .unwrap();
    let masked =
        DeviceMeanSquaresMetric::from_device_masked(&d_fixed, &d_moving, Some(&d_mask), None)
            .unwrap();
    let first = masked.evaluate(&t).unwrap();

    assert!(
        first.valid_points < unmasked.valid_points,
        "the mask dropped no samples: {} vs {}",
        first.valid_points,
        unmasked.valid_points
    );
    assert!(
        (first.value - unmasked.value).abs() > 1e-6,
        "the mask did not change the value"
    );

    // ...and it is still deterministic: the same metric, evaluated again, is bit-identical.
    let again = masked.evaluate(&t).unwrap();
    assert_eq!(again.value.to_bits(), first.value.to_bits());
    assert_eq!(again.valid_points, first.valid_points);
    for (a, b) in again.derivative.iter().zip(first.derivative.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "a masked derivative moved");
    }
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

    // A fixed-initial transform: the last refusal, and the only mask-shaped one left.
    // The fixed image and the in-buffer predicate would have to be resampled *through*
    // it, and both device resamples go through the identity only.
    let mut reg = ImageRegistrationMethod::new();
    reg.set_fixed_initial_transform(sitk_transform::Transform::Translation(
        sitk_transform::TranslationTransform::new(vec![1.0, 0.0, 0.0]),
    ));
    assert!(matches!(
        run(&reg),
        Err(DeviceRegistrationError::UnsupportedFixedInitialTransform)
    ));

    // ...and a fixed mask is *not* refused any more — it is carried onto every level.
    //
    // This configuration (the fixed image used as its own mask, with the default
    // optimizer) then diverges and ends in `NoValidSamples` — on **both** paths,
    // identically, which is the agreement that matters. What is asserted here is only
    // that the device no longer *declines* it. That it also lands where `execute` lands
    // is pinned by `the_configurations_the_boundary_now_accepts_land_where_execute_lands`.
    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_fixed_mask(&fixed);
    assert!(
        !matches!(
            run(&reg),
            Err(DeviceRegistrationError::UnsupportedMetric
                | DeviceRegistrationError::UnsupportedInterpolator(_)
                | DeviceRegistrationError::UnsupportedSampling
                | DeviceRegistrationError::UnsupportedFixedInitialTransform)
        ),
        "a fixed mask was refused at the boundary"
    );
    assert!(
        reg.execute(&fixed, &moving, euler()).is_err(),
        "the host takes this configuration; then the device declining to run it would          be a real divergence rather than a shared outcome"
    );

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
    println!("four refusals, each by name, all before the first iteration");
}

/// The boundary refuses a sampling **strategy**, not a sampling **percentage**.
///
/// `SamplingStrategy::None` samples every voxel and ignores the percentage — the
/// `None` arm of `FixedSamples::from_image_with` never reads it — so a percentage
/// set under the default strategy changes nothing on the host. The boundary used to
/// refuse it anyway, sending the caller to the CPU for a run this path computes
/// voxel for voxel. Nothing caught that, so this does: the run must be *taken*, and
/// it must land where the host lands.
#[test]
fn a_sampling_percentage_under_the_default_strategy_does_not_lose_the_device_path() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 32;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let c = n as f64 / 2.0;
    let initial = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]);

    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mean_squares();
    reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-5, 50, 1e-8);
    reg.set_optimizer_scales_from_physical_shift();
    // The strategy stays at its default (`None`); only the percentage is set.
    reg.set_metric_sampling_percentage(0.25, 7);

    let host = reg.execute(&fixed, &moving, initial()).unwrap();
    let device = reg
        .execute_on_device(
            &DeviceImage::upload(&fixed).unwrap(),
            &DeviceImage::upload(&moving).unwrap(),
            initial(),
        )
        .expect("a percentage the host itself ignores must not cost the device path");

    // Every voxel is a sample on both paths — the percentage changed nothing, which
    // is the whole point.
    assert_eq!(
        host.valid_points, device.valid_points,
        "the percentage was honored by one path and ignored by the other"
    );
    assert_eq!(host.iterations, device.iterations);

    let worst = device
        .transform
        .parameters()
        .iter()
        .zip(host.transform.parameters().iter())
        .map(|(&d, &h)| (d - h).abs() / (1.0 + h.abs()))
        .fold(0.0f64, f64::max);
    println!("worst parameter disagreement: {worst:e}");
    assert!(worst <= 1e-9, "worst {worst:e}");
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

    // Per-level diagnostics: both entry points fill them from the same place
    // (`drive`), so a pyramid that agreed only at the finest level — a coarse
    // level that burned its cap on one path and converged on the other — is
    // caught here rather than hidden by the last level's agreement.
    assert_eq!(
        device.levels.len(),
        3,
        "three scheduled levels, three records"
    );
    assert_eq!(host.levels.len(), device.levels.len());
    for (h, d) in host.levels.iter().zip(device.levels.iter()) {
        println!(
            "level {}: shrink {:?} sigma {} | host {} iters, {} valid, metric {:.12} \
             | device {} iters, {} valid, metric {:.12}",
            h.level,
            h.shrink_factors,
            h.smoothing_sigma,
            h.iterations,
            h.valid_points,
            h.metric_value,
            d.iterations,
            d.valid_points,
            d.metric_value
        );
        assert_eq!(d.level, h.level);
        assert_eq!(d.shrink_factors, h.shrink_factors);
        assert_eq!(d.smoothing_sigma, h.smoothing_sigma);
        assert_eq!(
            d.iterations, h.iterations,
            "level {} took a different number of steps",
            h.level
        );
        assert_eq!(
            d.valid_points, h.valid_points,
            "level {} sampled a different number of points",
            h.level
        );
        assert_eq!(
            d.stop_reason, h.stop_reason,
            "level {} stopped for a different reason",
            h.level
        );
        let rel = (d.metric_value - h.metric_value).abs() / (1.0 + h.metric_value.abs());
        assert!(
            rel <= 1e-9,
            "level {} converged to a different metric value: rel {rel:e}",
            h.level
        );
    }

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

/// A **moving** mask on the device metric agrees with the host's, exactly.
///
/// The kernel has had a moving mask since it was written (`mmask`/`has_mask`, with
/// `MovingImage::mask_allows`'s round-to-nearest test); the device metric simply never
/// passed one. So this is plumbing, and what it must not do is plumb a *different*
/// predicate: `valid_points` is asserted **equal**, not close, because a moving mask
/// decides which mapped points count, and a rounding rule that disagreed with the
/// host's would shift a shell of points in or out while leaving the value plausible.
///
/// Anti-vacuity: the mask must drop points relative to the unmasked run. A `None` that
/// slipped through the plumbing would otherwise pass every comparison here.
#[test]
fn a_moving_mask_on_the_device_metric_matches_the_host() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let mask = ellipsoid_mask(n);
    let t = probe_transform(n);

    let host = MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&fixed).unwrap(),
        MovingImage::from_image(&moving)
            .unwrap()
            .with_moving_mask(&mask)
            .unwrap(),
    )
    .unwrap();
    let cpu = host.evaluate(&t, &CpuBackend);

    let bits: Vec<bool> = mask
        .scalar_slice::<f32>()
        .unwrap()
        .iter()
        .map(|&v| v != 0.0)
        .collect();
    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();
    let masked =
        DeviceMeanSquaresMetric::from_device_masked(&d_fixed, &d_moving, None, Some(&bits))
            .unwrap();
    let gpu = masked.evaluate(&t).unwrap();

    let unmasked = DeviceMeanSquaresMetric::from_device(&d_fixed, &d_moving)
        .unwrap()
        .evaluate(&t)
        .unwrap();
    assert!(
        gpu.valid_points < unmasked.valid_points,
        "the moving mask dropped no points ({} vs {} unmasked)",
        gpu.valid_points,
        unmasked.valid_points
    );

    assert_eq!(
        gpu.valid_points, cpu.valid_points,
        "the moving-masked device metric walked a different sample set than the host"
    );
    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_err = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .fold(0.0f64, f64::max);
    println!(
        "moving mask: valid_points {} (both) | value rel err {v_err:e} | deriv rel err {d_err:e}",
        cpu.valid_points
    );
    assert!(v_err <= 1e-9, "value rel err {v_err:e} exceeds 1e-9");
    assert!(d_err <= 1e-9, "derivative rel err {d_err:e} exceeds 1e-9");
}

/// A moving mask that is not the moving image's grid is refused, not indexed into.
/// The kernel reads it by the moving grid's flat index; a shorter one would read past
/// the buffer, and an equal-length one on another shape would gate the wrong voxels.
#[test]
fn a_moving_mask_of_the_wrong_size_is_refused() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 32;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [1.0, 0.0, 0.0]));
    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();

    let short = vec![true; n * n * n - 1];
    match DeviceMeanSquaresMetric::from_device_masked(&d_fixed, &d_moving, None, Some(&short)) {
        Err(DeviceMetricError::Cuda(CudaError::DegenerateInput)) => {}
        Err(e) => panic!("refused, but by the wrong name: {e}"),
        Ok(_) => panic!("a moving mask shorter than the moving image built a metric"),
    }
}

/// C5: the configurations the boundary now **accepts** run on the device and land
/// where `execute` lands — a fixed mask, a moving mask, and a virtual domain.
///
/// Per-level `valid_points` is compared **exactly**. That is the assertion that would
/// catch a level whose mask was built on the wrong grid, or dropped: the run would
/// still converge and the parameters would still be close (the objective is smooth),
/// but it would be sampling a different set of voxels than `execute` samples, and the
/// count says so immediately. Parameters are compared at 1e-6 relative, the band the
/// unmasked end-to-end test already uses — the two paths are the same optimizer over a
/// metric that agrees to ~1e-14, not a bit-identical one.
#[test]
fn the_configurations_the_boundary_now_accepts_land_where_execute_lands() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = (volume(n, [0.0; 3]), volume(n, [3.0, -2.0, 1.5]));
    let mask = ellipsoid_mask(n);
    let d_f = DeviceImage::upload(&fixed).unwrap();
    let d_m = DeviceImage::upload(&moving).unwrap();
    let c = n as f64 / 2.0;
    let initial = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

    let configure = |reg: &mut ImageRegistrationMethod| {
        reg.set_metric_as_mean_squares();
        reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-4, 25, 1e-8);
    };

    // A virtual grid that overhangs the fixed image on every axis, so the in-buffer
    // predicate actually drops points rather than being all ones.
    let mut with_domain = ImageRegistrationMethod::new();
    configure(&mut with_domain);
    with_domain
        .set_virtual_domain(
            vec![40, 36, 44],
            vec![-14.0, -8.0, -12.0],
            vec![1.4, 1.5, 1.6],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

    let mut with_fixed_mask = ImageRegistrationMethod::new();
    configure(&mut with_fixed_mask);
    with_fixed_mask.set_metric_fixed_mask(&mask);

    let mut with_moving_mask = ImageRegistrationMethod::new();
    configure(&mut with_moving_mask);
    with_moving_mask.set_metric_moving_mask(&mask);

    let mut with_both = ImageRegistrationMethod::new();
    configure(&mut with_both);
    with_both.set_metric_fixed_mask(&mask);
    with_both
        .set_virtual_domain(
            vec![40, 36, 44],
            vec![-14.0, -8.0, -12.0],
            vec![1.4, 1.5, 1.6],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

    let unmasked_points = {
        let mut reg = ImageRegistrationMethod::new();
        configure(&mut reg);
        reg.execute(&fixed, &moving, initial())
            .unwrap()
            .valid_points
    };

    for (name, reg) in [
        ("fixed mask", &with_fixed_mask),
        ("moving mask", &with_moving_mask),
        ("virtual domain", &with_domain),
        ("fixed mask + virtual domain", &with_both),
    ] {
        let host = reg.execute(&fixed, &moving, initial()).unwrap();
        let device = reg
            .execute_on_device(&d_f, &d_m, initial())
            .unwrap_or_else(|e| panic!("{name}: the device path refused a run it now takes: {e}"));

        // Anti-vacuity: whatever the configuration does, it must not be a no-op. A
        // device that silently ignored the mask or the domain would agree with a host
        // that did too, and this is what makes that impossible.
        assert!(
            host.valid_points < unmasked_points,
            "{name}: the configuration drops no samples ({} vs {unmasked_points} plain); \
             a path that ignored it would pass this test",
            host.valid_points
        );

        println!(
            "{name}: host {} iters / {} valid | device {} iters / {} valid",
            host.iterations, host.valid_points, device.iterations, device.valid_points
        );
        assert_eq!(
            device.levels.len(),
            host.levels.len(),
            "{name}: different level counts"
        );
        for (h, d) in host.levels.iter().zip(device.levels.iter()) {
            assert_eq!(
                d.valid_points, h.valid_points,
                "{name}, level {}: the device walked a different sample set than the host",
                h.level
            );
        }
        assert_eq!(
            device.valid_points, host.valid_points,
            "{name}: final valid-point counts differ"
        );
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
                "{name}, param {k}: device {d:e} vs host {h:e} (rel {rel:e})"
            );
        }
    }
}
