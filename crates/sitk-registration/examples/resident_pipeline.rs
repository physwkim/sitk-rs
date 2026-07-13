//! The chain, end to end, three ways — the measurement the residency API was built
//! to answer.
//!
//! ```text
//! host      : rescale(fixed) + rescale(moving) -> metric setup -> 20 iterations   (CPU, N threads)
//! gpu today : one-shot GPU rescale x2 (a bus round trip each) -> the metric re-uploads
//!             the same voxels -> 20 iterations on the device
//! resident  : upload x2 -> device rescale x2 -> metric reads those same device
//!             buffers -> 20 iterations. Nothing crosses the bus after the upload.
//! ```
//!
//! ```text
//! cargo run --release --features cuda -p sitk-registration --example resident_pipeline -- 256 96
//! ```
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs the GPU: rebuild with --features cuda");
}

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use sitk_core::Image;
#[cfg(feature = "cuda")]
use sitk_cuda::{DeviceImage, rescale_intensity as device_rescale, rescale_intensity_gpu};
#[cfg(feature = "cuda")]
use sitk_registration::metric::{FixedSamples, MovingImage};
#[cfg(feature = "cuda")]
use sitk_registration::{
    CpuBackend, CudaMetricBackend, DeviceMeanSquaresMetric, MeanSquaresMetric,
};
#[cfg(feature = "cuda")]
use sitk_transform::Euler3DTransform;

#[cfg(feature = "cuda")]
const OUT_MIN: f64 = 0.0;
#[cfg(feature = "cuda")]
const OUT_MAX: f64 = 255.0;
#[cfg(feature = "cuda")]
const ITERS: usize = 20;

#[cfg(feature = "cuda")]
fn volume(n: usize, shift: f64) -> Image {
    let c = n as f64 / 2.0;
    let mut v = vec![0.0f32; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64 - shift, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                v[(z * n + y) * n + x] = (200.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp()
                    + 20.0 * (0.4 * r).sin()) as f32;
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
fn transform() -> Euler3DTransform {
    Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [0.0; 3])
}

#[cfg(feature = "cuda")]
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(256, |s| s.parse().expect("size"));
    let threads: usize = std::env::args()
        .nth(2)
        .map_or(96, |s| s.parse().expect("threads"));
    let mb = (n * n * n * 4) as f64 / 1e6;

    let fixed = volume(n, 0.0);
    let moving = volume(n, 3.0);
    let tf = transform();

    sitk_core::parallel::with_threads(threads, || {
        // Burn NVRTC (~160 ms, once per process, size-independent) so it is not
        // smeared into the first measurement.
        if let Ok(d) = DeviceImage::upload(&volume(16, 0.0)) {
            let _ = device_rescale(&d, OUT_MIN, OUT_MAX);
        } else {
            println!("SKIPPED: no CUDA device");
            return;
        }

        println!("== {n}^3 f32 ({mb:.0} MB/volume), {threads} CPU threads, {ITERS} iterations\n");

        // ---- host ------------------------------------------------------------
        let t = Instant::now();
        let f = sitk_filters::rescale_intensity(&fixed, OUT_MIN, OUT_MAX).unwrap();
        let m = sitk_filters::rescale_intensity(&moving, OUT_MIN, OUT_MAX).unwrap();
        let cpu_filters = ms(t);
        let t = Instant::now();
        let metric = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&f).unwrap(),
            MovingImage::from_image(&m).unwrap(),
        )
        .unwrap();
        let cpu_setup = ms(t);
        let t = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), &CpuBackend));
        }
        let cpu_iters = ms(t);
        let cpu_total = cpu_filters + cpu_setup + cpu_iters;

        // ---- gpu, today's one-shot API ---------------------------------------
        let t = Instant::now();
        let (f_g, _) = rescale_intensity_gpu(&fixed, OUT_MIN, OUT_MAX).unwrap();
        let (m_g, _) = rescale_intensity_gpu(&moving, OUT_MIN, OUT_MAX).unwrap();
        let g_filters = ms(t);
        let t = Instant::now();
        let metric_g = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&f_g).unwrap(),
            MovingImage::from_image(&m_g).unwrap(),
        )
        .unwrap();
        let g_setup = ms(t);
        let cuda_be = CudaMetricBackend::new();
        // The metric's own H2D happens on its first evaluation; bill it separately.
        let t = Instant::now();
        std::hint::black_box(metric_g.evaluate(&tf, &cuda_be));
        let g_upload = ms(t);
        let t = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(metric_g.evaluate(std::hint::black_box(&tf), &cuda_be));
        }
        let g_iters = ms(t);
        let g_total = g_filters + g_setup + g_upload + g_iters;

        // ---- resident --------------------------------------------------------
        let t = Instant::now();
        let d_fixed = DeviceImage::upload(&fixed).unwrap();
        let d_moving = DeviceImage::upload(&moving).unwrap();
        let r_upload = ms(t);
        let t = Instant::now();
        let d_f = device_rescale(&d_fixed, OUT_MIN, OUT_MAX).unwrap();
        let d_m = device_rescale(&d_moving, OUT_MIN, OUT_MAX).unwrap();
        let r_filters = ms(t);
        let t = Instant::now();
        let mut r_metric = DeviceMeanSquaresMetric::from_device(&d_f, &d_m).unwrap();
        let r_setup = ms(t);
        let t = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(r_metric.evaluate(std::hint::black_box(&tf)).unwrap());
        }
        let r_iters = ms(t);
        let r_total = r_upload + r_filters + r_setup + r_iters;

        // A registration returns a transform, not an image, so the resident chain
        // ends with no D2H at all. Measured anyway, for a caller that does want the
        // filtered volume back.
        let t = Instant::now();
        let _back = d_f.to_host().unwrap();
        let r_d2h = ms(t);

        println!(
            "host      filters {cpu_filters:8.1}  setup {cpu_setup:7.1}  \
             iters {cpu_iters:9.1}                       total {cpu_total:9.1} ms"
        );
        println!(
            "gpu today filters {g_filters:8.1}  setup {g_setup:7.1}  \
             iters {g_iters:9.1}  metric H2D {g_upload:7.1}  total {g_total:9.1} ms"
        );
        println!(
            "resident  filters {r_filters:8.1}  setup {r_setup:7.1}  \
             iters {r_iters:9.1}  upload     {r_upload:7.1}  total {r_total:9.1} ms"
        );
        println!("\n  resident vs host      {:.0}x", cpu_total / r_total);
        println!("  resident vs gpu today {:.1}x", g_total / r_total);
        println!("  optional D2H of a filtered volume (a registration needs none): {r_d2h:.1} ms");

        // Same answer, or the speed is worthless.
        let host_v = metric.value(&tf, &CpuBackend);
        let res_v = r_metric.value(&tf).unwrap();
        println!(
            "\n  metric value: host {host_v:.12} | resident {res_v:.12} | rel err {:e}",
            (host_v - res_v).abs() / (1.0 + host_v.abs())
        );
    });
}
