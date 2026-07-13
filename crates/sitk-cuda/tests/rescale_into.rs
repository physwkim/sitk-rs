//! The zero-allocation `rescale_intensity_gpu_into` form.
//!
//! The claim under test is not "it is faster" but "it allocates nothing and
//! changes nothing": the output must be bit-identical to the allocating form, and
//! `alloc_ms` must be exactly zero because there is no host allocation to time.
#![cfg(feature = "cuda")]

use sitk_core::{Image, PixelId};
use sitk_cuda::{CudaError, backend, rescale_intensity_gpu, rescale_intensity_gpu_into};

/// The bench-spec's synthetic volume: xorshift64*, deterministic.
fn synth(n: usize) -> Image {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut v = Vec::with_capacity(n * n * n);
    for _ in 0..n * n * n {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let x = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        v.push(((x >> 40) as f32) / 16777.216);
    }
    Image::from_vec(&[n, n, n], v).unwrap()
}

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

#[test]
fn into_matches_the_allocating_form_bit_for_bit_and_allocates_nothing() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let img = synth(n);

    let (owned, t_owned) = rescale_intensity_gpu(&img, 0.0, 255.0).unwrap();

    // The destination a looping caller would keep alive across calls.
    let mut dst = Image::from_vec(&[n, n, n], vec![0.0f32; n * n * n]).unwrap();
    let t_into = rescale_intensity_gpu_into(&img, 0.0, 255.0, &mut dst).unwrap();

    assert_eq!(
        owned.scalar_slice::<f32>().unwrap(),
        dst.scalar_slice::<f32>().unwrap(),
        "the two forms must produce the same bytes"
    );
    assert_eq!(dst.size(), owned.size());
    assert_eq!(dst.origin(), owned.origin());
    assert_eq!(dst.spacing(), owned.spacing());

    assert_eq!(
        t_into.alloc_ms, 0.0,
        "the _into form allocates nothing on the host, so there is nothing to time"
    );
    println!(
        "owned: alloc {:.2} ms, total {:.2} ms | into: alloc {:.2} ms, total {:.2} ms",
        t_owned.alloc_ms,
        t_owned.total_ms(),
        t_into.alloc_ms,
        t_into.total_ms()
    );

    // Reusing the same destination must keep producing the same bytes, not drift.
    let first = dst.scalar_slice::<f32>().unwrap().to_vec();
    for _ in 0..3 {
        rescale_intensity_gpu_into(&img, 0.0, 255.0, &mut dst).unwrap();
        assert_eq!(dst.scalar_slice::<f32>().unwrap(), first.as_slice());
    }
}

/// What the reused destination is worth, at the bench-spec sizes.
///
/// `#[ignore]`d: a measurement, not an assertion. Run with
/// `cargo test -p sitk-cuda --features cuda --release -- --ignored --nocapture`.
#[test]
#[ignore = "measurement, not an assertion"]
fn what_the_reused_destination_saves() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    for &n in &[64usize, 256, 512] {
        let img = synth(n);
        // Warm up: the first call of the process compiles the kernels.
        let _ = rescale_intensity_gpu(&img, 0.0, 255.0).unwrap();

        let owned: Vec<_> = (0..3)
            .map(|_| rescale_intensity_gpu(&img, 0.0, 255.0).unwrap().1)
            .collect();
        let owned_total = median(owned.iter().map(|t| t.total_ms()).collect());
        let owned_alloc = median(owned.iter().map(|t| t.alloc_ms).collect());

        let mut dst = Image::from_vec(&[n, n, n], vec![0.0f32; n * n * n]).unwrap();
        let into: Vec<_> = (0..3)
            .map(|_| rescale_intensity_gpu_into(&img, 0.0, 255.0, &mut dst).unwrap())
            .collect();
        let into_total = median(into.iter().map(|t| t.total_ms()).collect());

        println!(
            "{n}^3: one-shot {owned_total:7.1} ms (of which alloc {owned_alloc:6.1}) | \
             reused dst {into_total:7.1} ms | saved {:6.1} ms/call ({:.2}x)",
            owned_total - into_total,
            owned_total / into_total,
        );
    }
}

#[test]
fn into_rejects_a_destination_that_does_not_match() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = synth(16);

    let mut wrong_size = Image::from_vec(&[8, 8, 8], vec![0.0f32; 512]).unwrap();
    assert!(matches!(
        rescale_intensity_gpu_into(&img, 0.0, 255.0, &mut wrong_size),
        Err(CudaError::DegenerateInput)
    ));

    let mut wrong_type = Image::from_vec(&[16, 16, 16], vec![0.0f64; 4096]).unwrap();
    assert!(matches!(
        rescale_intensity_gpu_into(&img, 0.0, 255.0, &mut wrong_type),
        Err(CudaError::UnsupportedPixelType(PixelId::Float64))
    ));
}
