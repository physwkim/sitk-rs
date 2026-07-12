//! GPU-vs-CPU gate for op 1 of `doc/bench-spec.md`, plus the fallback contract.
//!
//! Only compiled with the `cuda` feature; with the feature off this file
//! contributes no tests.
#![cfg(feature = "cuda")]

use sitk_core::{Image, PixelId};
use sitk_filters::{FilterError, rescale_intensity, rescale_intensity_cpu};

/// `doc/bench-spec.md` § Inputs — xorshift64*, mapped to `[0, 1000)`. Both the
/// Rust and the C++ harness implement this; it must not drift.
fn synth(seed: u64, size: [usize; 3]) -> Vec<f32> {
    let n = size[0] * size[1] * size[2];
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        out.push(((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) % 1000) as f32);
    }
    out
}

fn medium_volume() -> Image {
    let size = [256usize, 256, 256];
    let mut img = Image::from_vec(&size, synth(0x1234_5678_9abc_def0, size)).unwrap();
    img.set_spacing(&[1.0, 1.0, 1.0]).unwrap();
    img
}

/// The bench spec's correctness gate for the GPU: not bit-for-bit, but
/// `max_abs_err` / `max_rel_err` against the CPU result.
#[test]
fn gpu_matches_cpu_on_the_medium_volume() {
    let img = medium_volume();
    // The first call pays NVRTC compilation and module load inside its kernel
    // phase; reporting that as "the kernel time" would be a lie, so it is
    // reported separately as the cold number and the steady-state split is the
    // median of the runs after it.
    let (gpu, cold) = match sitk_cuda::rescale_intensity_gpu(&img, 0.0, 255.0) {
        Ok(v) => v,
        // A machine with the feature compiled in but no usable device is a
        // supported configuration — it is exactly what the CPU fallback is
        // for, and it is what `CUDA_VISIBLE_DEVICES=""` produces. There is no
        // GPU result to check, so say so loudly rather than fail or pretend.
        Err(e @ sitk_cuda::CudaError::NoDevice(_)) => {
            println!("SKIPPED: no CUDA device on this machine ({e}); the fallback tests still ran");
            return;
        }
        Err(e) => panic!("GPU is present but the op failed: {e}"),
    };
    let mut warm: Vec<sitk_cuda::GpuTimings> = (0..5)
        .map(|_| {
            sitk_cuda::rescale_intensity_gpu(&img, 0.0, 255.0)
                .unwrap()
                .1
        })
        .collect();
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let h2d = median(warm.iter().map(|t| t.h2d_ms).collect());
    let kernel = median(warm.iter().map(|t| t.kernel_ms).collect());
    let d2h = median(warm.iter().map(|t| t.d2h_ms).collect());
    warm.clear();

    // How much of `d2h_ms` is the fresh output allocation rather than the PCIe
    // link: the same copy into a destination that is already resident.
    let pooled_d2h = {
        let backend = sitk_cuda::backend().unwrap();
        let d = sitk_cuda::DeviceBuffer::from_host(backend, img.scalar_slice::<f32>().unwrap())
            .unwrap();
        let mut dst = vec![0f32; d.len()];
        let mut samples = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            d.copy_to_host(backend, &mut dst).unwrap();
            backend.synchronize().unwrap();
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        median(samples)
    };

    let t = std::time::Instant::now();
    let cpu = rescale_intensity_cpu(&img, 0.0, 255.0).unwrap();
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

    let g = gpu.scalar_slice::<f32>().unwrap();
    let c = cpu.scalar_slice::<f32>().unwrap();
    assert_eq!(g.len(), c.len());

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut differing = 0usize;
    for (&gv, &cv) in g.iter().zip(c.iter()) {
        let (gv, cv) = (gv as f64, cv as f64);
        let abs = (gv - cv).abs();
        if abs != 0.0 {
            differing += 1;
        }
        max_abs = max_abs.max(abs);
        if cv != 0.0 {
            max_rel = max_rel.max(abs / cv.abs());
        }
    }
    println!("voxels             = {}", g.len());
    println!("differing          = {differing}");
    println!("max_abs_err        = {max_abs:e}");
    println!("max_rel_err        = {max_rel:e}");
    println!("cold h2d_ms        = {:.3}", cold.h2d_ms);
    println!(
        "cold kernel_ms     = {:.3}  (includes NVRTC compile + module load)",
        cold.kernel_ms
    );
    println!("cold d2h_ms        = {:.3}", cold.d2h_ms);
    println!("cold total_ms      = {:.3}", cold.total_ms());
    println!("warm h2d_ms        = {h2d:.3}  (median of 5)");
    println!("warm kernel_ms     = {kernel:.3}  (median of 5)");
    println!("warm d2h_ms        = {d2h:.3}  (median of 5)");
    println!("warm total_ms      = {:.3}", h2d + kernel + d2h);
    println!(
        "warm d2h_ms pooled = {pooled_d2h:.3}  (resident dst: d2h_ms minus first-touch page faults)"
    );
    println!("cpu 1-thread ms    = {cpu_ms:.3}  (f64 widen + scan + map + narrow)");

    // Tolerance. Output range is [0, 255], so one f32 ULP near the top of the
    // range is ~1.5e-5 absolute and ~1.2e-7 relative. Both kernels compute in
    // `double` and narrow once at the store, exactly as the CPU path does, so
    // the expected error is zero and anything above a single f32 ULP means a
    // real numeric divergence — not rounding. The bounds below are one ULP,
    // not a loose "close enough" band.
    assert!(
        max_abs <= 1.53e-5,
        "max_abs_err {max_abs:e} exceeds one f32 ULP at 255"
    );
    assert!(
        max_rel <= 1.20e-7,
        "max_rel_err {max_rel:e} exceeds one f32 ULP relative"
    );
}

/// Task 0: the published claim was that `rescale_intensity`'s GPU time is
/// "essentially all PCIe", so per-pixel GPU offload can never pay. This walks
/// the bench-spec sizes and reports the phase split, including the CPU
/// single-thread reference, so the claim is decided by measurement.
///
/// Not an assertion of a timing bound — timings are not a test contract, and a
/// loaded machine would make one flaky. It asserts *correctness* (the output is
/// still bit-identical to the CPU's at every size) and prints the numbers.
#[test]
fn transfer_split_across_the_bench_spec_sizes() {
    for (label, n) in [("small", 64usize), ("medium", 256), ("large", 512)] {
        let size = [n, n, n];
        let img = match Image::from_vec(&size, synth(0x1234_5678_9abc_def0, size)) {
            Ok(i) => i,
            Err(e) => panic!("building the {label} volume failed: {e}"),
        };

        let (gpu, cold) = match sitk_cuda::rescale_intensity_gpu(&img, 0.0, 255.0) {
            Ok(v) => v,
            Err(e @ sitk_cuda::CudaError::NoDevice(_)) => {
                println!("SKIPPED: no CUDA device ({e})");
                return;
            }
            Err(e) => panic!("GPU present but {label} failed: {e}"),
        };
        let warm: Vec<sitk_cuda::GpuTimings> = (0..3)
            .map(|_| {
                sitk_cuda::rescale_intensity_gpu(&img, 0.0, 255.0)
                    .expect("the device answered once already")
                    .1
            })
            .collect();
        let med = |f: fn(&sitk_cuda::GpuTimings) -> f64| {
            let mut v: Vec<f64> = warm.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (h2d, alloc, kernel, d2h) = (
            med(|t| t.h2d_ms),
            med(|t| t.alloc_ms),
            med(|t| t.kernel_ms),
            med(|t| t.d2h_ms),
        );
        let total = h2d + alloc + kernel + d2h;

        let t = std::time::Instant::now();
        let cpu = rescale_intensity_cpu(&img, 0.0, 255.0).unwrap();
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

        let bytes = (n * n * n * 4) as f64;
        let gbs = |ms: f64| bytes / 1e9 / (ms / 1e3);

        println!(
            "--- {label} {n}^3 ({} MiB each way) ---",
            (bytes / 1048576.0) as usize
        );
        println!("  h2d    {h2d:8.2} ms  ({:5.2} GB/s)", gbs(h2d));
        println!("  alloc  {alloc:8.2} ms  (host: make the output resident)");
        println!("  kernel {kernel:8.2} ms");
        println!(
            "  d2h    {d2h:8.2} ms  ({:5.2} GB/s, DMA into a resident dst)",
            gbs(d2h)
        );
        println!(
            "  TOTAL  {total:8.2} ms   (cold, incl. NVRTC: {:.2} ms)",
            cold.total_ms()
        );
        println!("  cpu t1 {cpu_ms:8.2} ms");

        // The point of Task 0 was to make the op faster without moving a single
        // bit of the result. Bit-identity, at every size.
        assert_eq!(
            gpu.scalar_slice::<f32>().unwrap(),
            cpu.scalar_slice::<f32>().unwrap(),
            "{label}: GPU output must stay bit-identical to the CPU"
        );
    }
}

/// The fallback contract: an unsupported pixel type must not fail, it must run
/// on the CPU. `UInt8` has no GPU kernel.
#[test]
fn unsupported_pixel_type_falls_back_to_the_cpu() {
    let mut img = Image::new(&[4, 4], PixelId::UInt8);
    for (i, v) in img.scalar_vec_mut::<u8>().unwrap().iter_mut().enumerate() {
        *v = i as u8;
    }
    let out = rescale_intensity(&img, 0.0, 255.0).unwrap();
    let cpu = rescale_intensity_cpu(&img, 0.0, 255.0).unwrap();
    assert_eq!(out.pixel_id(), PixelId::UInt8);
    assert_eq!(
        out.scalar_slice::<u8>().unwrap(),
        cpu.scalar_slice::<u8>().unwrap()
    );
}

/// A degenerate image must raise the CPU path's error, not a GPU one: the GPU
/// declines and `rescale_intensity` falls through.
#[test]
fn degenerate_range_still_reports_the_cpu_error() {
    let img = Image::from_vec(&[4, 4], vec![7.0f32; 16]).unwrap();
    let err = rescale_intensity(&img, 0.0, 255.0).unwrap_err();
    assert!(matches!(err, FilterError::DegenerateRange), "got {err:?}");
}

/// The public op routes through the GPU when it is available, and its result
/// is the CPU result. This is the plumbing test: it exercises exactly what a
/// caller of `sitk_filters::rescale_intensity` gets with the feature on.
#[test]
fn public_op_agrees_with_the_cpu_on_a_small_volume() {
    let size = [64usize, 64, 64];
    let img = Image::from_vec(&size, synth(0xdead_beef_cafe_f00d, size)).unwrap();
    let out = rescale_intensity(&img, 0.0, 255.0).unwrap();
    let cpu = rescale_intensity_cpu(&img, 0.0, 255.0).unwrap();
    assert_eq!(
        out.scalar_slice::<f32>().unwrap(),
        cpu.scalar_slice::<f32>().unwrap()
    );
}
