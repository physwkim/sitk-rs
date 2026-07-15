//! Per-evaluation cost and device bytes of the resident metric, in isolation.
//!
//! The chain benchmarks measure a pipeline; this measures only the thing the volume
//! precision changes — the metric kernel — so the three layouts (f64/f64, f32/f32,
//! f32 fixed + f64 moving) can be compared without upload, filter or optimizer noise
//! in the number. Which layout is built is a compile-time choice inside
//! `sitk-cuda`; run this once per build.
//!
//! ```text
//! cargo run --release --features cuda -p sitk-registration --example metric_layout -- 256 96
//! ```
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs the GPU: rebuild with --features cuda");
}

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use sitk::core::Image;
#[cfg(feature = "cuda")]
use sitk::cuda::DeviceImage;
#[cfg(feature = "cuda")]
use sitk::registration::DeviceMeanSquaresMetric;
#[cfg(feature = "cuda")]
use sitk::transform::Euler3DTransform;

#[cfg(feature = "cuda")]
const ITERS: usize = 50;

#[cfg(feature = "cuda")]
fn volume(n: usize, shift: f64) -> Image {
    let c = n as f64 / 2.0;
    let mut v = Vec::with_capacity(n * n * n);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64 - shift, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                let s = 2000.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp()
                    + 200.0 * (0.4 * r).sin()
                    + 400.0;
                v.push(s as f32);
            }
        }
    }
    Image::from_vec(&[n, n, n], v).unwrap()
}

#[cfg(feature = "cuda")]
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(256, |s| s.parse().expect("size"));
    let threads: usize = std::env::args()
        .nth(2)
        .map_or(96, |s| s.parse().expect("threads"));

    sitk::core::parallel::with_threads(threads, || {
        let c = n as f64 / 2.0;
        let t = Euler3DTransform::new(0.01, -0.008, 0.006, [1.5, -0.8, 0.4], [c, c, c]);

        let small = volume(16, 0.0);
        match DeviceImage::upload(&small) {
            Ok(d) => {
                // Burn NVRTC for whichever kernel this build instantiates.
                let m = DeviceMeanSquaresMetric::from_device(&d, &d).unwrap();
                let _ = m.evaluate(&t).unwrap();
            }
            Err(e) => {
                println!("SKIPPED: no CUDA device: {e}");
                return;
            }
        }

        let df = DeviceImage::upload(&volume(n, 0.0)).unwrap();
        let dm = DeviceImage::upload(&volume(n, 3.0)).unwrap();

        let build = Instant::now();
        let metric = DeviceMeanSquaresMetric::from_device(&df, &dm).unwrap();
        let build_ms = build.elapsed().as_secs_f64() * 1e3;

        // Warm, then time.
        let first = metric.evaluate(&t).unwrap();
        let run = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(metric.evaluate(std::hint::black_box(&t)).unwrap());
        }
        let per = run.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

        println!(
            "{n}^3  build {build_ms:6.2} ms   per evaluation {per:6.3} ms   \
             volume bytes {:>4} MB   value {:.15}",
            metric.volume_bytes() / 1_000_000,
            first.value
        );
    });
}
