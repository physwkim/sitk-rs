//! The metric's scaling curve: one value+derivative evaluation, at a given image
//! size and worker count. Setup is excluded — this is what an optimizer iteration
//! actually spends its time on.
//!
//! ```text
//! cargo run --release --example metric_curve -- <size> <threads> [iters]
//! ```

use std::time::Instant;

use sitk::core::Image;
use sitk::registration::{CpuBackend, MeanSquaresMetric};
use sitk::transform::Euler3DTransform;

/// The same field `reg_bench` registers: a blob plus a fine ripple, so the metric
/// has a gradient everywhere rather than on one shell.
fn volume(n: usize, shift: f64) -> Image {
    let mut v = vec![0.0f64; n * n * n];
    let c = n as f64 / 2.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64 - shift, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                v[(z * n + y) * n + x] =
                    200.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp() + 20.0 * (0.4 * r).sin();
            }
        }
    }
    Image::from_vec(&[n, n, n], v).expect("volume")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().expect("size").parse().expect("size");
    let threads: usize = args.next().expect("threads").parse().expect("threads");
    let iters: usize = args.next().map_or(3, |s| s.parse().expect("iters"));

    let fixed = volume(size, 0.0);
    let moving = volume(size, 3.0);
    let c = size as f64 / 2.0;
    let tf = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

    sitk::core::parallel::with_threads(threads, || {
        let metric = MeanSquaresMetric::new(&fixed, &moving).expect("metric");
        let warm = metric.evaluate(&tf, &CpuBackend);

        let mut best = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), &CpuBackend));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{{\"size\":{size},\"threads\":{threads},\"samples\":{},\"eval_ms\":{best:.2},\
             \"metric\":{:.9e}}}",
            metric.sample_count(),
            warm.value,
        );
    });
}
