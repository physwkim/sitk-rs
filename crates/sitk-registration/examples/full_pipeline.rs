//! The pipeline the user described, measured whole:
//!
//! ```text
//! load (UInt16, as a CT is) -> cast to f32 -> rescale -> smooth (gaussian) -> register
//! ```
//!
//! Two ways. The host chain runs every stage on the CPU. The resident chain casts
//! on the host (the device type is `f32`, and the crossing refuses any other pixel
//! type by name rather than converting behind the caller's back), uploads **once**,
//! and then rescales, smooths, and registers against buffers that never leave the
//! device — no D2H at all, because a registration returns a transform, not an image.
//!
//! ```text
//! cargo run --release --features cuda -p sitk-registration --example full_pipeline -- 256 96
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
use sitk_registration::{CpuBackend, DeviceMeanSquaresMetric, MeanSquaresMetric};
#[cfg(feature = "cuda")]
use sitk_transform::Euler3DTransform;

#[cfg(feature = "cuda")]
const OUT_MIN: f64 = 0.0;
#[cfg(feature = "cuda")]
const OUT_MAX: f64 = 255.0;
#[cfg(feature = "cuda")]
const ITERS: usize = 20;
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

    let fixed = volume(n, 0.0);
    let moving = volume(n, 3.0);
    let tf = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [0.0; 3]);

    sitk_core::parallel::with_threads(threads, || {
        // Burn NVRTC (once per process, size-independent) for every kernel used below
        // — including the device cast, which is why this warms up on the NATIVE type.
        match DeviceImage::upload(&volume(16, 0.0)) {
            Ok(d) => {
                let r = device_rescale(&d, OUT_MIN, OUT_MAX).unwrap();
                let _ = device_smooth(&r, &SIGMA).unwrap();
            }
            Err(e) => {
                println!("SKIPPED: no CUDA device: {e}");
                return;
            }
        }

        println!(
            "== {n}^3 UInt16 in, {threads} CPU threads, sigma {SIGMA:?}, {ITERS} iterations\n"
        );

        // ---- host: cast -> rescale -> smooth -> register ----------------------
        let mut host_stage = [0.0f64; 3];
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
        let hf = host_chain(&fixed, &mut host_stage);
        let hm = host_chain(&moving, &mut host_stage);
        let t = Instant::now();
        let metric = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&hf).unwrap(),
            MovingImage::from_image(&hm).unwrap(),
        )
        .unwrap();
        let host_setup = ms(t);
        let t = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), &CpuBackend));
        }
        let host_iters = ms(t);
        let host_total = ms(whole);

        // ---- resident: cast (host) -> upload -> rescale -> smooth -> register --
        let mut dev_stage = [0.0f64; 4];
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
        let df = device_chain(&fixed, &mut dev_stage);
        let dm = device_chain(&moving, &mut dev_stage);
        let t = Instant::now();
        let dev_metric = DeviceMeanSquaresMetric::from_device(&df, &dm).unwrap();
        let dev_setup = ms(t);
        let t = Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(dev_metric.evaluate(std::hint::black_box(&tf)).unwrap());
        }
        let dev_iters = ms(t);
        let dev_total = ms(whole);

        println!(
            "host      cast {:7.1}  rescale {:7.1}  smooth {:8.1}  setup {host_setup:7.1}  \
             iters {host_iters:9.1}   total {host_total:9.1} ms",
            host_stage[0], host_stage[1], host_stage[2]
        );
        println!(
            "resident  cast {:7.1}  upload+cast {:7.1}  rescale {:6.1}  smooth {:6.1}  \
             setup {dev_setup:6.1}  iters {dev_iters:7.1}   total {dev_total:9.1} ms",
            dev_stage[0], dev_stage[1], dev_stage[2], dev_stage[3]
        );
        println!(
            "\n  resident vs host: {:.0}x   (both volumes, both filters, the metric, {ITERS} iterations)",
            host_total / dev_total
        );

        let hv = metric.value(&tf, &CpuBackend);
        let dv = dev_metric.value(&tf).unwrap();
        println!(
            "  metric value: host {hv:.12} | resident {dv:.12} | rel err {:e}",
            (hv - dv).abs() / (1.0 + hv.abs())
        );
    });
}
