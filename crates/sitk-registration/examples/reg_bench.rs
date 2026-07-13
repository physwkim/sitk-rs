//! Registration cost breakdown: **setup** (FixedSamples) vs **iteration**
//! (metric evaluation), at a given image size and thread count.
//!
//! Matches the GPU panel's configuration: rigid Euler3D, mean squares, full
//! sampling, regular-step gradient descent, 20 iterations.
//!
//! ```text
//! cargo run --release --example reg_bench -- <size> <threads> [iters]
//! ```
//!
//! Prints one JSON line so a driver script can collect the runs.

use std::time::Instant;

use sitk_core::Image;
use sitk_registration::{CpuBackend, ImageRegistrationMethod, MeanSquaresMetric, SamplingStrategy};
use sitk_transform::{Euler3DTransform, ParametricTransform};

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

fn main() {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().expect("size").parse().expect("size");
    let threads: usize = args.next().expect("threads").parse().expect("threads");
    let iters: usize = args.next().map_or(20, |s| s.parse().expect("iters"));

    let fixed = volume(size, 0.0);
    let moving = volume(size, 3.0);
    let c = size as f64 / 2.0;
    let start = || Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

    sitk_core::parallel::with_threads(threads, || {
        // Setup: build FixedSamples + the moving image's device-side arrays.
        let t = Instant::now();
        let metric = MeanSquaresMetric::new(&fixed, &moving).expect("metric");
        let setup_ms = ms(t);

        // Iteration: one value+derivative evaluation is what a regular-step GD
        // iteration spends its time on.
        let tf = start();
        let warm = metric.evaluate(&tf, &CpuBackend);
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), &CpuBackend));
        }
        let iter_total_ms = ms(t);

        // The value-only reduction, for the Amdahl diagnostic: it stages one
        // column instead of `1 + nparams`, and skips the Jacobian. Comparing its
        // scaling with the full evaluation's separates a serial-combine wall
        // (which would cap both) from a parallel-side wall in the derivative.
        std::hint::black_box(metric.value(&tf, &CpuBackend));
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(metric.value(std::hint::black_box(&tf), &CpuBackend));
        }
        let value_ms = ms(t) / iters as f64;

        // Whole run, through the public registration method — includes setup,
        // the optimizer's own evaluations, and its bookkeeping.
        let mut m = ImageRegistrationMethod::new();
        m.set_metric_as_mean_squares()
            .set_metric_sampling_strategy(SamplingStrategy::None)
            .set_optimizer_as_regular_step_gradient_descent(1.0, 1e-4, iters, 1e-8)
            .set_optimizer_scales_from_physical_shift();
        let t = Instant::now();
        let out = m.execute(&fixed, &moving, start()).expect("registration");
        let run_ms = ms(t);

        println!(
            "{{\"size\":{size},\"threads\":{threads},\"iters\":{iters},\
             \"samples\":{},\"valid\":{},\
             \"setup_ms\":{setup_ms:.1},\"iter_ms\":{:.2},\"value_ms\":{value_ms:.2},\
             \"run_ms\":{run_ms:.1},\"metric0\":{:.6e},\"final\":{:?}}}",
            metric.sample_count(),
            warm.valid_points,
            iter_total_ms / iters as f64,
            warm.value,
            out.transform.parameters(),
        );
    });
}
