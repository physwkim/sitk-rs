//! A **real registration**, both ways, through the public API:
//! [`ImageRegistrationMethod::execute`] against a host image pair, and
//! [`ImageRegistrationMethod::execute_on_device`] against a device-resident pair.
//!
//! This is not an `evaluate` loop. The optimizer runs, decides its own iteration
//! count, and returns a transform; the two runs are asked to land on the same
//! answer, and the wall clock is reported for the registration stage on its own
//! and for the whole chain (`load -> cast -> rescale -> smooth -> register`).
//!
//! ```text
//! cargo run --release --features cuda -p sitk-registration --example execute_bench -- 256 96
//! ```
//!
//! Arguments: `size threads iterations min_step [scales] [shrink] [sigmas]`. The last
//! two turn the run into a **multi-resolution pyramid**, which is how a registration is
//! really driven; without them the run is single-level:
//!
//! ```text
//! ... --example execute_bench -- 256 96 200 1e-4 scales 4,2,1 2,1,0
//! ```
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs the GPU: rebuild with --features cuda");
}

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use sitk_core::{Image, PixelId};
#[cfg(feature = "cuda")]
use sitk_cuda::{
    DeviceImage, rescale_intensity as device_rescale, smooth_gaussian as device_smooth,
};
#[cfg(feature = "cuda")]
use sitk_registration::metric::{FixedSamples, MovingImage};
#[cfg(feature = "cuda")]
use sitk_registration::{
    CpuBackend, DeviceMeanSquaresMetric, ImageRegistrationMethod, MeanSquaresMetric,
};
#[cfg(feature = "cuda")]
use sitk_transform::{Euler3DTransform, ParametricTransform};

#[cfg(feature = "cuda")]
const OUT_MIN: f64 = 0.0;
#[cfg(feature = "cuda")]
const OUT_MAX: f64 = 255.0;
#[cfg(feature = "cuda")]
const SIGMA: [f64; 3] = [1.0, 1.0, 1.0];

/// A `UInt16` volume, as a CT arrives from disk.
#[cfg(feature = "cuda")]
fn volume(n: usize, shift: f64) -> Image {
    let c = n as f64 / 2.0;
    let mut v = vec![0u16; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64 - shift, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                let s = 2000.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp()
                    + 200.0 * (0.4 * r).sin()
                    + 400.0;
                v[(z * n + y) * n + x] = s.clamp(0.0, 65535.0) as u16;
            }
        }
    }
    Image::from_vec(&[n, n, n], v).expect("volume")
}

#[cfg(feature = "cuda")]
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[cfg(feature = "cuda")]
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(256, |s| s.parse().expect("size"));
    let threads: usize = std::env::args()
        .nth(2)
        .map_or(96, |s| s.parse().expect("threads"));
    let iters: usize = std::env::args()
        .nth(3)
        .map_or(20, |s| s.parse().expect("iterations"));
    let min_step: f64 = std::env::args()
        .nth(4)
        .map_or(1e-4, |s| s.parse().expect("min step"));

    // A pyramid is only run if BOTH lists are given, and they must be the same length:
    // the schedule is one (shrink, sigma) pair per level.
    let list = |i: usize| -> Vec<String> {
        std::env::args()
            .nth(i)
            .map(|s| s.split(',').map(str::to_owned).collect())
            .unwrap_or_default()
    };
    let shrink: Vec<usize> = list(6).iter().map(|s| s.parse().expect("shrink")).collect();
    let sigmas: Vec<f64> = list(7).iter().map(|s| s.parse().expect("sigma")).collect();

    let fixed = volume(n, 0.0);
    let moving = volume(n, 3.0);
    let c = n as f64 / 2.0;
    let initial = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]);

    sitk_core::parallel::with_threads(threads, || {
        // The method is not `Send` (it owns a `Box<dyn MetricBackend>`), so it is
        // built inside the pool's scope rather than moved into it.
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares();
        reg.set_optimizer_as_regular_step_gradient_descent(1.0, min_step, iters, 1e-8);
        // Optional 5th arg: condition the optimizer's step in physical units, which
        // is what a caller registering a rotation *should* do. Without it a
        // 1-radian rotation step is the same size as a 1-mm translation step.
        if std::env::args().nth(5).as_deref() == Some("scales") {
            reg.set_optimizer_scales_from_physical_shift();
            println!("  (optimizer scales from physical shift)");
        }
        if !shrink.is_empty() {
            assert_eq!(
                shrink.len(),
                sigmas.len(),
                "the schedule is one (shrink, sigma) pair per level"
            );
            reg.set_shrink_factors_per_level(shrink.clone());
            reg.set_smoothing_sigmas_per_level(sigmas.clone());
        }

        // Burn NVRTC (once per process, size-independent) on a tiny volume — and on
        // the NATIVE type, so the device cast's kernel is compiled too and its
        // one-time compile does not land inside the timed upload below.
        let burn = 8 * shrink.iter().copied().max().unwrap_or(2);
        match DeviceImage::upload(&volume(burn, 0.0)) {
            Ok(d) => {
                let r = device_rescale(&d, OUT_MIN, OUT_MAX).unwrap();
                let s = device_smooth(&r, &SIGMA).unwrap();
                let _ = reg.execute_on_device(&s, &s, initial()).unwrap();
            }
            Err(e) => {
                println!("SKIPPED: no CUDA device: {e}");
                return;
            }
        }

        let schedule = if shrink.is_empty() {
            "single level".to_owned()
        } else {
            format!("pyramid shrink {shrink:?} sigma {sigmas:?}")
        };
        println!(
            "== {n}^3 UInt16 in, {threads} CPU threads, sigma {SIGMA:?}, real execute(), \
             {schedule}, max {iters} iterations per level, min step {min_step:e}\n"
        );

        // ---- host: cast -> rescale -> smooth -> execute -----------------------
        let mut host_pre = [0.0f64; 3];
        let host_chain = |img: &Image, stage: &mut [f64; 3]| {
            let t = Instant::now();
            let c = sitk_filters::cast(img, PixelId::Float32).unwrap();
            stage[0] += ms(t);
            let t = Instant::now();
            let r = sitk_filters::rescale_intensity(&c, OUT_MIN, OUT_MAX).unwrap();
            stage[1] += ms(t);
            let t = Instant::now();
            let s = sitk_filters::smooth_gaussian(&r, &SIGMA).unwrap();
            stage[2] += ms(t);
            s
        };
        let whole = Instant::now();
        let hf = host_chain(&fixed, &mut host_pre);
        let hm = host_chain(&moving, &mut host_pre);
        let t = Instant::now();
        let host = reg.execute(&hf, &hm, initial()).unwrap();
        let host_reg = ms(t);
        let host_total = ms(whole);

        // ---- resident: cast (host) -> upload -> rescale -> smooth -> execute ---
        let mut dev_pre = [0.0f64; 4];
        // No host cast: `upload` takes the UInt16 volume in its native type and
        // casts on the device, so the f32 volume never exists on the host.
        let device_chain = |img: &Image, stage: &mut [f64; 4]| {
            let t = Instant::now();
            let d = DeviceImage::upload(img).unwrap();
            stage[1] += ms(t);
            let t = Instant::now();
            let r = device_rescale(&d, OUT_MIN, OUT_MAX).unwrap();
            stage[2] += ms(t);
            let t = Instant::now();
            let s = device_smooth(&r, &SIGMA).unwrap();
            stage[3] += ms(t);
            s
        };
        let whole = Instant::now();
        let df = device_chain(&fixed, &mut dev_pre);
        let dm = device_chain(&moving, &mut dev_pre);
        let t = Instant::now();
        let device = reg.execute_on_device(&df, &dm, initial()).unwrap();
        let device_reg = ms(t);
        let device_total = ms(whole);

        println!(
            "host      cast {:7.1}  rescale {:7.1}  smooth {:8.1}  \
             execute {host_reg:9.1}   total {host_total:9.1} ms",
            host_pre[0], host_pre[1], host_pre[2]
        );
        println!(
            "resident  cast {:7.1}  upload+cast {:7.1}  rescale {:6.1}  smooth {:6.1}  \
             execute_on_device {device_reg:7.1}   total {device_total:9.1} ms",
            dev_pre[0], dev_pre[1], dev_pre[2], dev_pre[3]
        );
        println!(
            "\n  registration stage: {:.0}x     whole chain: {:.0}x",
            host_reg / device_reg,
            host_total / device_total
        );
        println!(
            "\n  host   : {} iters, {} valid, metric {:.9}, stop {:?}, params {:?}",
            host.iterations,
            host.valid_points,
            host.metric_value,
            host.stop_reason,
            host.transform.parameters()
        );
        println!(
            "  device : {} iters, {} valid, metric {:.9}, stop {:?}, params {:?}",
            device.iterations,
            device.valid_points,
            device.metric_value,
            device.stop_reason,
            device.transform.parameters()
        );
        let worst = device
            .transform
            .parameters()
            .iter()
            .zip(host.transform.parameters().iter())
            .map(|(&d, &h)| (d - h).abs() / (1.0 + h.abs()))
            .fold(0.0f64, f64::max);
        println!("  worst parameter disagreement: {worst:e} (relative)");

        // If the two optimizers land apart, the question is whether the *metric*
        // disagrees or whether the two trajectories diverged. Evaluate each metric
        // at BOTH endpoints: agreement here means the metric is sound and the
        // optimizer amplified rounding, not that the device computed a different
        // objective.
        let hmetric = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&hf).unwrap(),
            MovingImage::from_image(&hm).unwrap(),
        )
        .unwrap();
        let dmetric = DeviceMeanSquaresMetric::from_device(&df, &dm).unwrap();
        for (name, t) in [
            ("initial", &initial()),
            ("host-final", &host.transform),
            ("dev-final", &device.transform),
        ] {
            let h = hmetric.evaluate(t, &CpuBackend);
            let d = dmetric.evaluate(t).unwrap();
            let dv = (h.value - d.value).abs() / (1.0 + h.value.abs());
            let dd = h
                .derivative
                .iter()
                .zip(d.derivative.iter())
                .map(|(&a, &b)| (a - b).abs() / (1.0 + a.abs()))
                .fold(0.0f64, f64::max);
            println!(
                "  at {name:10}: value {:.12} rel {dv:e} | worst derivative rel {dd:e} | \
                 valid {} vs {}",
                h.value, h.valid_points, d.valid_points
            );
        }
    });
}
