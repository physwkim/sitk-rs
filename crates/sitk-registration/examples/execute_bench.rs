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
//! Arguments: `size threads iterations min_step [scales] [shrink] [sigmas] [sampling]
//! [metric]`. `shrink` and `sigmas` turn the run into a **multi-resolution pyramid**, which
//! is how a registration is really driven; without them the run is single-level:
//!
//! ```text
//! ... --example execute_bench -- 256 96 200 1e-4 scales 4,2,1 2,1,0
//! ```
//!
//! An 8th argument adds a **sampling strategy** (`random:0.01`, `regular:0.1`), applied to
//! both paths: the host draws the samples, the device is handed that draw.
//!
//! ```text
//! ... --example execute_bench -- 256 96 20 1e-4 - - - random:0.01
//! ```
//!
//! # Reading the parameter disagreement
//!
//! At 256³/20 iterations the two runs land ~1.7e-4 apart (relative), sampled or not. That
//! is **not** a metric difference and not a preprocessing difference — the cross-evaluation
//! printed below settles it, and is there for exactly this reason. Evaluated at the *same*
//! transform (both endpoints), the two metrics agree to ~5e-15 on the value and ~1e-12 on
//! the derivative, with identical valid-point counts.
//!
//! What separates the runs is the optimizer: `RegularStepGradientDescentOptimizer` halves
//! its step on overshoot, a *discontinuous* decision, so once a ~1e-12 derivative
//! difference flips one overshoot test the two take different steps and converge to two
//! different, both-valid poses. Same mechanism the registration tests pin around
//! (`a_fixed_initial_transform_converges_where_execute_converges`). Read the
//! cross-evaluation, not the parameter delta, to decide whether the device metric is
//! sound.
//!
//! # The correlation speed-up printed here is NOT a device number
//!
//! A 9th argument selects the metric (`meansquares`, the default, or `correlation`). Both
//! have a device kernel. But the *host* sides are not comparable to each other:
//! `CpuBackend::mean_squares` folds its samples in parallel across the thread pool, and
//! `CorrelationMetric::evaluate` is a plain serial loop with no rayon in it at all. So the
//! host correlation run is slow for a reason that has nothing to do with the GPU, and the
//! host/device ratio it produces credits the device with the host's missing threading.
//!
//! Measured, 128³, 20 iterations, 96 threads, physical-shift scales:
//!
//! ```text
//!                  host execute   device execute   ratio
//!   mean squares       3367.8 ms         24.7 ms    136x
//!   correlation       30091.1 ms         44.6 ms    674x
//! ```
//!
//! The host correlation run is 8.9× the host mean-squares run; the *device* correlation
//! evaluation is 1.88× the device mean-squares one (1.891 ms vs 1.005 ms per iteration,
//! printed at the end of every run). 1.88× is the number that belongs to this design — the
//! second pass NCC needs for its sample means, plus the wider partials buffer. The rest of
//! the 674× is the host metric's serial loop, and quoting it as a GPU speed-up would be
//! reporting a missing `par_iter` as a kernel.
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
use sitk_registration::metric::{FixedSamples, MovingImage, SamplingStrategy};
#[cfg(feature = "cuda")]
use sitk_registration::{
    CorrelationMetric, CpuBackend, DeviceCorrelationMetric, DeviceMeanSquaresMetric,
    ImageRegistrationMethod, MeanSquaresMetric, MetricValue,
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
    // Optional 9th arg: which metric. Both have a device kernel; correlation costs a
    // second device pass per evaluation (its moments need the sample means first), and
    // what that costs is measured at the end of this run rather than asserted.
    let correlation = match std::env::args().nth(9).as_deref() {
        None | Some("-") | Some("") | Some("meansquares") => false,
        Some("correlation") => true,
        Some(other) => panic!("metric is `meansquares` or `correlation`, not `{other}`"),
    };

    // A pyramid is only run if BOTH lists are given, and they must be the same length:
    // the schedule is one (shrink, sigma) pair per level.
    // `-` is "not given", so a later argument can be reached without a schedule.
    let list = |i: usize| -> Vec<String> {
        std::env::args()
            .nth(i)
            .filter(|s| s != "-" && !s.is_empty())
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
        if correlation {
            reg.set_metric_as_correlation();
        } else {
            reg.set_metric_as_mean_squares();
        }
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

        // Optional 8th arg: a **sampling strategy**, `random:0.01` or `regular:0.1`.
        // Both paths get it — the host draws the samples and the device is handed that
        // draw (`metric::draw_samples`), so this changes how many voxels each iteration
        // walks and nothing else about what is compared.
        if let Some(spec) = std::env::args()
            .nth(8)
            .filter(|s| s != "-" && !s.is_empty())
        {
            let (kind, pct) = spec.split_once(':').expect("sampling is kind:percentage");
            let strategy = match kind {
                "random" => SamplingStrategy::Random,
                "regular" => SamplingStrategy::Regular,
                other => panic!("sampling kind is `random` or `regular`, not `{other}`"),
            };
            let pct: f64 = pct.parse().expect("sampling percentage");
            reg.set_metric_sampling_strategy(strategy);
            reg.set_metric_sampling_percentage(pct, 7);
            println!(
                "  (sampling: {strategy:?} {:.1}% of the fixed grid — {} of {} voxels, and the \
                 index list the device is handed is {} KiB)",
                pct * 100.0,
                (n * n * n) as f64 * pct,
                n * n * n,
                (n * n * n) as f64 * pct * 8.0 / 1024.0,
            );
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
             {schedule}, metric {}, max {iters} iterations per level, min step {min_step:e}\n",
            if correlation {
                "correlation"
            } else {
                "mean squares"
            }
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
        if correlation {
            println!(
                "  ^ NOT a device number: CpuBackend::mean_squares folds its samples in \
                 parallel, CorrelationMetric::evaluate is a serial loop with no rayon in it. \
                 The host side of this ratio is single-threaded. What the second device pass \
                 costs is measured below, against the mean-squares kernel on the same volumes."
            );
        }
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
        let host_ms = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&hf).unwrap(),
            MovingImage::from_image(&hm).unwrap(),
        )
        .unwrap();
        let host_ncc = CorrelationMetric::from_samples(
            FixedSamples::from_image(&hf).unwrap(),
            MovingImage::from_image(&hm).unwrap(),
        )
        .unwrap();
        let dev_ms = DeviceMeanSquaresMetric::from_device(&df, &dm).unwrap();
        let dev_ncc = DeviceCorrelationMetric::from_device(&df, &dm).unwrap();

        let on_host = |t: &dyn ParametricTransform| -> MetricValue {
            if correlation {
                host_ncc.evaluate(t)
            } else {
                host_ms.evaluate(t, &CpuBackend)
            }
        };
        let on_device = |t: &dyn ParametricTransform| -> MetricValue {
            if correlation {
                dev_ncc.evaluate(t).unwrap()
            } else {
                dev_ms.evaluate(t).unwrap()
            }
        };

        for (name, t) in [
            ("initial", &initial() as &dyn ParametricTransform),
            ("host-final", &host.transform),
            ("dev-final", &device.transform),
        ] {
            let h = on_host(t);
            let d = on_device(t);
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

        // What the second pass costs, measured rather than claimed. Correlation needs the
        // sample means before any mean-subtracted moment can be formed, so it launches the
        // sampling kernel twice per evaluation: once for (sum f, sum m, count), once for the
        // 28 moments. Mean squares launches once, for 14. Both walk the same samples with
        // the same sampler, so the ratio below is the price of the extra pass and of the
        // wider partials buffer --- and nothing else.
        let probe = initial();
        let bench = |f: &mut dyn FnMut()| -> f64 {
            for _ in 0..3 {
                f();
            }
            let t = Instant::now();
            for _ in 0..20 {
                f();
            }
            ms(t) / 20.0
        };
        let ms_iter = bench(&mut || {
            std::hint::black_box(dev_ms.evaluate(&probe).unwrap());
        });
        let ncc_iter = bench(&mut || {
            std::hint::black_box(dev_ncc.evaluate(&probe).unwrap());
        });
        println!(
            "\n  device metric cost/iteration: mean squares {ms_iter:.3} ms (1 pass, 14 slots) | \
             correlation {ncc_iter:.3} ms (2 passes, 3 + 28 slots) | ratio {:.2}x",
            ncc_iter / ms_iter
        );
    });
}
