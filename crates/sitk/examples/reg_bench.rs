//! Registration cost breakdown: **setup** (FixedSamples) vs **iteration**
//! (metric evaluation), at a given image size, thread count, and backend.
//!
//! Rigid Euler3D, mean squares, full sampling, regular-step gradient descent,
//! 20 iterations by default.
//!
//! ```text
//! cargo run --release --example reg_bench -- <size> <threads> [iters] [cpu|gpu]
//! cargo run --release --features cuda --example reg_bench -- 256 96 20 gpu
//! ```
//!
//! The backend argument exists so that CPU `t1`, CPU `tN` and the GPU can be
//! measured **back to back in one machine state**. A GPU-vs-CPU ratio taken from
//! two sessions on this box is worthless — the same function has measured 184 ms
//! and 863 ms depending on how much of the page cache was in use.
//!
//! `gpu` requires the `cuda` feature; without it the argument is rejected rather
//! than silently falling back, so a run can never mislabel a CPU number as a GPU
//! one.
//!
//! Prints one JSON line so a driver script can collect the runs.

use std::time::Instant;

use sitk::core::Image;
use sitk::registration::{
    CpuBackend, ImageRegistrationMethod, MeanSquaresMetric, MetricBackend, SamplingStrategy,
};
use sitk::transform::{Euler3DTransform, ParametricTransform};

/// A smooth, non-symmetric intensity field. `shift` displaces it in physical
/// space, so the moving image is the fixed image under a known translation
/// without needing a resample filter in the timing path.
fn volume(n: usize, shift: f64) -> Image {
    let mut v = vec![0.0f64; n * n * n];
    let c = n as f64 / 2.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64 - shift, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                // A blob plus a fine ripple: gives the metric a gradient
                // everywhere, not just on one shell.
                v[(z * n + y) * n + x] =
                    200.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp() + 20.0 * (0.4 * r).sin();
            }
        }
    }
    Image::from_vec(&[n, n, n], v).expect("volume")
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// A fresh backend of the requested kind. One per use: the CUDA backend owns
/// device-resident buffers, and handing the same one to both the hand-rolled loop
/// and `execute()` would let the first warm the second.
fn make_backend(kind: &str) -> Box<dyn MetricBackend> {
    match kind {
        "cpu" => Box::new(CpuBackend),
        #[cfg(feature = "cuda")]
        "gpu" => Box::new(sitk::registration::CudaMetricBackend::new()),
        #[cfg(not(feature = "cuda"))]
        "gpu" => panic!("backend `gpu` requires --features cuda"),
        other => panic!("unknown backend `{other}` (expected `cpu` or `gpu`)"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().expect("size").parse().expect("size");
    let threads: usize = args.next().expect("threads").parse().expect("threads");
    let iters: usize = args.next().map_or(20, |s| s.parse().expect("iters"));
    let kind = args.next().unwrap_or_else(|| "cpu".to_string());

    // The GPU compiles its kernel with NVRTC on first use: ~200 ms, once per
    // process, independent of volume size. Burn it on a tiny volume so it is not
    // smeared into the first real evaluation and misreported as transfer cost.
    let mut nvrtc_ms = 0.0;
    if kind == "gpu" {
        let tiny = volume(16, 1.0);
        let t = Instant::now();
        let m = MeanSquaresMetric::new(&tiny, &tiny).expect("warmup metric");
        let tf = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [8.0; 3]);
        std::hint::black_box(m.evaluate(&tf, make_backend(&kind).as_ref()));
        nvrtc_ms = ms(t);
    }

    let fixed = volume(size, 0.0);
    let moving = volume(size, 3.0);
    let c = size as f64 / 2.0;
    let start = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

    sitk::core::parallel::with_threads(threads, || {
        // Setup: build FixedSamples + the moving image's sample arrays. This is
        // host work and both backends pay it.
        let t = Instant::now();
        let metric = MeanSquaresMetric::new(&fixed, &moving).expect("metric");
        let setup_ms = ms(t);

        let backend = make_backend(&kind);
        let tf = start();

        // The first evaluation is where the GPU uploads the two volumes. It is a
        // once-per-level cost, not a per-iteration one, so it is reported apart
        // from `iter_ms`. On the CPU it is just a warm-up.
        let t = Instant::now();
        let warm = metric.evaluate(&tf, backend.as_ref());
        let first_ms = ms(t);

        // Iteration: one value+derivative evaluation is what a regular-step GD
        // iteration spends its time on.
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), backend.as_ref()));
        }
        let iter_total_ms = ms(t);

        // The value-only reduction, for the Amdahl diagnostic: it stages one
        // column instead of `1 + nparams`, and skips the Jacobian. Comparing its
        // scaling with the full evaluation's separates a serial-combine wall
        // (which would cap both) from a parallel-side wall in the derivative.
        std::hint::black_box(metric.value(&tf, backend.as_ref()));
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(metric.value(std::hint::black_box(&tf), backend.as_ref()));
        }
        let value_ms = ms(t) / iters as f64;

        // Whole run, through the public registration method — includes setup,
        // the optimizer's own evaluations, and its bookkeeping.
        let mut m = ImageRegistrationMethod::new();
        m.set_metric_as_mean_squares()
            .set_metric_sampling_strategy(SamplingStrategy::None)
            .set_optimizer_as_regular_step_gradient_descent(1.0, 1e-4, iters, 1e-8)
            .set_optimizer_scales_from_physical_shift()
            .set_metric_backend(make_backend(&kind));
        let t = Instant::now();
        let out = m.execute(&fixed, &moving, start()).expect("registration");
        let run_ms = ms(t);

        println!(
            "{{\"size\":{size},\"threads\":{threads},\"iters\":{iters},\"backend\":\"{kind}\",\
             \"samples\":{},\"valid\":{},\
             \"setup_ms\":{setup_ms:.1},\"first_eval_ms\":{first_ms:.2},\"nvrtc_ms\":{nvrtc_ms:.1},\
             \"iter_ms\":{:.3},\"value_ms\":{value_ms:.3},\
             \"run_ms\":{run_ms:.1},\"metric0\":{:.6e},\"final\":{:?}}}",
            metric.sample_count(),
            warm.valid_points,
            iter_total_ms / iters as f64,
            warm.value,
            out.transform.parameters(),
        );
    });
}
