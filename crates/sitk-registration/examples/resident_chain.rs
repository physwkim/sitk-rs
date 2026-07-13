//! Does a **device-resident pipeline** beat the CPU, where a one-shot GPU op
//! does not?
//!
//! Every filter GPU number this port has published was measured through
//! `fn(&Image) -> Image`, which pays H2D + kernel + D2H on *every call*. At 256³
//! an `f32` volume is 67 MB each way, so a round trip is ~10 ms against a kernel
//! of ~1 ms. That verdict is about the **bus**, not the device: it says a
//! one-shot API loses. It says nothing about a chain that uploads once, runs
//! several kernels on the resident buffer, and hands the same buffer to the
//! registration metric with no upload at all.
//!
//! This measures both halves of that claim, at one size, in one machine state:
//!
//! 1. **Kernel alone**, input and output already resident, against CPU `tN`.
//! 2. **The chain**: `rescale → registration setup → 20 iterations`, run three
//!    ways — CPU throughout; GPU as the API exists *today* (one-shot rescale, so
//!    the volume crosses the bus twice, then the metric uploads it again); and
//!    GPU *resident*, where the volume is uploaded once and never comes back.
//!
//! ```text
//! cargo run --release --features cuda -p sitk-registration --example resident_chain -- 256
//! ```
#![cfg(feature = "cuda")]

use std::time::Instant;

use sitk_core::Image;
use sitk_cuda::{DeviceBuffer, backend, rescale_intensity_gpu, rescale_intensity_resident};
use sitk_registration::{CpuBackend, CudaMetricBackend, MeanSquaresMetric, MetricBackend};
use sitk_transform::Euler3DTransform;

const OUT_MIN: f64 = 0.0;
const OUT_MAX: f64 = 255.0;
const ITERS: usize = 20;

/// `f32` volume, smooth and non-symmetric; `shift` displaces it in physical
/// space so the moving image is the fixed image under a known translation.
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

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One value+derivative evaluation per iteration, as a regular-step gradient
/// descent step would.
fn iterate(metric: &MeanSquaresMetric, be: &dyn MetricBackend) -> f64 {
    let c = 0.0;
    let tf = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]);
    let t = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), be));
    }
    ms(t)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(256, |s| s.parse().expect("size"));
    let threads: usize = std::env::args()
        .nth(2)
        .map_or(96, |s| s.parse().expect("threads"));
    let voxels = n * n * n;
    let mb = (voxels * 4) as f64 / 1e6;

    let be = match backend() {
        Ok(b) => b,
        Err(e) => {
            println!("SKIPPED: no CUDA device: {e}");
            return;
        }
    };

    let fixed_raw = volume(n, 0.0);
    let moving_raw = volume(n, 3.0);

    sitk_core::parallel::with_threads(threads, || {
        // Burn NVRTC (~180 ms, once per process, size-independent) so it is not
        // smeared into the first measurement.
        let tiny = volume(16, 0.0);
        let _ = rescale_intensity_gpu(&tiny, OUT_MIN, OUT_MAX).expect("warmup");

        println!("== {n}^3 f32 ({voxels} voxels, {mb:.0} MB), {threads} CPU threads\n");

        // ---- 1. the op, with and without the bus ---------------------------
        let cpu_op = median(
            (0..5)
                .map(|_| {
                    let t = Instant::now();
                    std::hint::black_box(
                        sitk_filters::rescale_intensity(&fixed_raw, OUT_MIN, OUT_MAX).unwrap(),
                    );
                    ms(t)
                })
                .collect(),
        );

        let one_shot = median(
            (0..5)
                .map(|_| {
                    rescale_intensity_gpu(&fixed_raw, OUT_MIN, OUT_MAX)
                        .unwrap()
                        .1
                })
                .map(|t| t.total_ms())
                .collect(),
        );

        let host_in = fixed_raw.scalar_slice::<f32>().unwrap();
        let t = Instant::now();
        let d_in = DeviceBuffer::from_host(be, host_in).unwrap();
        be.synchronize().unwrap();
        let h2d = ms(t);
        let mut d_out = DeviceBuffer::<f32>::zeros(be, voxels).unwrap();
        let kernel = median(
            (0..5)
                .map(|_| {
                    rescale_intensity_resident(be, &d_in, &mut d_out, OUT_MIN, OUT_MAX).unwrap()
                })
                .collect(),
        );
        // Resident, and reused: a fresh `vec![]` is lazily mapped, so the DMA
        // would fault every page under itself and the copy would measure the
        // fault storm rather than the link (60 ms instead of 6 at 256³).
        let mut host_back = sitk_core::alloc::resident_vec::<f32>(voxels);
        let d2h = median(
            (0..5)
                .map(|_| {
                    let t = Instant::now();
                    d_out.copy_to_host(be, &mut host_back).unwrap();
                    be.synchronize().unwrap();
                    ms(t)
                })
                .collect(),
        );

        println!("rescale_intensity, one call:");
        println!("  CPU tN                        {cpu_op:8.2} ms");
        println!("  GPU one-shot (H2D+k+D2H)      {one_shot:8.2} ms   <- what we published");
        println!("  GPU kernel alone, resident    {kernel:8.2} ms   <- the device");
        println!("  H2D {h2d:.2} ms | D2H {d2h:.2} ms  ({mb:.0} MB each way)");
        println!(
            "  kernel-only speedup vs CPU tN: {:.1}x   |  bus round trip: {:.2} ms\n",
            cpu_op / kernel,
            h2d + d2h
        );

        // ---- 2. the chain ---------------------------------------------------
        // CPU: rescale both volumes, build the metric, 20 iterations.
        let t = Instant::now();
        let f = sitk_filters::rescale_intensity(&fixed_raw, OUT_MIN, OUT_MAX).unwrap();
        let m = sitk_filters::rescale_intensity(&moving_raw, OUT_MIN, OUT_MAX).unwrap();
        let cpu_filters = ms(t);
        let t = Instant::now();
        let metric = MeanSquaresMetric::new(&f, &m).expect("metric");
        let cpu_setup = ms(t);
        let cpu_iters = iterate(&metric, &CpuBackend);
        let cpu_total = cpu_filters + cpu_setup + cpu_iters;

        // GPU as the API is today: each filter call is a round trip, and then the
        // metric uploads the very volumes that were just on the device.
        let t = Instant::now();
        let (f_g, _) = rescale_intensity_gpu(&fixed_raw, OUT_MIN, OUT_MAX).unwrap();
        let (m_g, _) = rescale_intensity_gpu(&moving_raw, OUT_MIN, OUT_MAX).unwrap();
        let gpu_filters = ms(t);
        let t = Instant::now();
        let metric_g = MeanSquaresMetric::new(&f_g, &m_g).expect("metric");
        let gpu_setup = ms(t);
        let cuda_be = CudaMetricBackend::new();
        // The metric's own H2D happens on the first evaluation; report it apart.
        let tf0 = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [0.0; 3]);
        let t = Instant::now();
        std::hint::black_box(metric_g.evaluate(&tf0, &cuda_be));
        let gpu_upload = ms(t);
        let gpu_iters = iterate(&metric_g, &cuda_be);
        let gpu_total = gpu_filters + gpu_setup + gpu_upload + gpu_iters;

        println!("chain: rescale(fixed) + rescale(moving) -> metric setup -> {ITERS} iterations");
        println!(
            "  CPU tN            filters {cpu_filters:8.1}  setup {cpu_setup:7.1}  \
             iters {cpu_iters:8.1}                    total {cpu_total:9.1} ms"
        );
        println!(
            "  GPU (today's API) filters {gpu_filters:8.1}  setup {gpu_setup:7.1}  \
             iters {gpu_iters:8.1}  metric H2D {gpu_upload:7.1}  total {gpu_total:9.1} ms"
        );
        println!(
            "\n  What residency would delete from the GPU column: the two D2Hs the filters\n  \
             pay ({:.1} ms), the metric's re-upload of the same voxels ({gpu_upload:.1} ms incl.\n  \
             its first evaluation), and the host f64 widening inside setup ({cpu_setup:.1} ms).",
            2.0 * d2h
        );
        println!(
            "  Resident chain floor (measured parts, summed): H2D {h2d:.1} + kernels {:.1} + \
             iters {gpu_iters:.1} = {:.1} ms",
            2.0 * kernel,
            h2d + 2.0 * kernel + gpu_iters
        );
    });
}
