//! The determinism gate for CPU parallelism (`sitk_core::parallel`).
//!
//! Every op below is hashed on a volume large enough to cross
//! `parallel`'s serial threshold — so these outputs are produced by the
//! *parallel* code paths, not the serial fallbacks that the small-image unit
//! tests elsewhere exercise.
//!
//! Two things are pinned, and an op must satisfy both:
//!
//! 1. **Bit-parity with the scalar port.** The expected checksums were harvested
//!    from the pre-parallel implementation of this crate, so a change of a single
//!    output bit — the thing a re-associated float reduction would cause — fails
//!    here. This is `doc/bench-spec.md`'s "Correctness gate", the Rust half.
//! 2. **Independence from the thread count.** Each op runs on rayon pools of 1,
//!    3, 8 and 32 threads, and every run must return the identical bit pattern.
//!    A decomposition that varied with the thread count (or with steal order)
//!    would show up as a mismatch between pools.
//!
//! The input is `doc/bench-spec.md`'s `synth` generator, so these hashes are
//! comparable with the benchmark harness's `output_checksum` field.

use sitk_core::{Image, PixelId, parallel};
use sitk_filters as f;

/// `doc/bench-spec.md`'s input generator: xorshift64*, mapped to `[0, 1000)`.
fn synth(seed: u64, voxels: usize) -> Vec<f32> {
    let mut state = seed;
    (0..voxels)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545F4914F6CDD1D) >> 33) % 1000) as f32
        })
        .collect()
}

/// FNV-1a 64 over the buffer's little-endian bytes (`doc/bench-spec.md`).
fn checksum(img: &Image) -> u64 {
    let bytes: Vec<u8> = match img.pixel_id() {
        PixelId::Float32 => img
            .scalar_slice::<f32>()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        PixelId::Float64 => img
            .scalar_slice::<f64>()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        PixelId::UInt8 => img.scalar_slice::<u8>().unwrap().to_vec(),
        PixelId::UInt32 => img
            .scalar_slice::<u32>()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        other => panic!("unhashed pixel type {other:?}"),
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 64³ = 262 144 voxels: past `parallel`'s serial threshold, and enough blocks
/// that the line passes take their *parallel* decomposition on every axis —
/// including the slowest axis, whose lines cross every row and so take the
/// column-split path.
const SIZE: [usize; 3] = [64, 64, 64];

fn float_volume() -> Image {
    let voxels = SIZE.iter().product();
    Image::from_vec(&SIZE, synth(0x2545_F491_4F6C_DD1D, voxels)).unwrap()
}

/// The binary/label input, thresholded at `>= 500.0` per `doc/bench-spec.md`.
fn binary_volume() -> Image {
    let voxels = SIZE.iter().product();
    let data: Vec<u8> = synth(0x2545_F491_4F6C_DD1D, voxels)
        .iter()
        .map(|&v| u8::from(v >= 500.0))
        .collect();
    Image::from_vec(&SIZE, data).unwrap()
}

/// Runs `op` on pools of 1, 3, 8 and 32 threads, asserts every run agrees
/// bit-for-bit, and asserts that value is `expected` — the checksum the
/// pre-parallel scalar implementation produced.
fn assert_bit_parity(name: &str, expected: u64, op: impl Fn() -> Image + Sync + Send + Copy) {
    let mut seen: Option<u64> = None;
    for threads in [1usize, 3, 8, 32] {
        let got = parallel::with_threads(threads, || checksum(&op()));
        if let Some(prev) = seen {
            assert_eq!(
                prev, got,
                "{name}: output changed with the thread count ({threads} threads gave \
                 {got:#018x}, an earlier pool gave {prev:#018x}) — the decomposition is \
                 not deterministic"
            );
        }
        seen = Some(got);
    }
    assert_eq!(
        seen.unwrap(),
        expected,
        "{name}: output differs from the pre-parallel scalar implementation \
         (got {:#018x}, want {expected:#018x}) — parallelizing this op changed its bits",
        seen.unwrap()
    );
}

#[test]
fn op01_rescale_intensity() {
    let img = float_volume();
    assert_bit_parity("rescale_intensity", 0x2ffb_0025_58b1_047d, || {
        f::rescale_intensity(&img, 0.0, 255.0).unwrap()
    });
}

#[test]
fn op02_smoothing_recursive_gaussian() {
    let img = float_volume();
    assert_bit_parity(
        "smoothing_recursive_gaussian",
        0x9e50_67c7_3b94_ee1e,
        || f::smoothing_recursive_gaussian(&img, &[2.0, 2.0, 2.0], false).unwrap(),
    );
}

#[test]
fn op03_discrete_gaussian() {
    let img = float_volume();
    assert_bit_parity("discrete_gaussian", 0xada6_7ce0_8b25_a022, || {
        f::discrete_gaussian(&img, &[4.0; 3], &[0.01; 3], 32, true).unwrap()
    });
}

#[test]
fn op04_median() {
    let img = float_volume();
    assert_bit_parity("median", 0xfd49_9f63_28c1_783c, || {
        f::median(&img, &[2, 2, 2]).unwrap()
    });
}

#[test]
fn op05_mean() {
    let img = float_volume();
    assert_bit_parity("mean", 0xfa72_d0d2_4ab3_2fc6, || {
        f::mean(&img, &[2, 2, 2]).unwrap()
    });
}

#[test]
fn op06_gradient_magnitude() {
    let img = float_volume();
    assert_bit_parity("gradient_magnitude", 0xabd4_eb37_32fd_a7cd, || {
        f::gradient_magnitude(&img, true).unwrap()
    });
}

#[test]
fn op07_gradient_magnitude_recursive_gaussian() {
    let img = float_volume();
    assert_bit_parity(
        "gradient_magnitude_recursive_gaussian",
        0x8b3f_9ad0_e06a_b3a8,
        || f::gradient_magnitude_recursive_gaussian(&img, 2.0, false).unwrap(),
    );
}

#[test]
fn op08_binary_dilate() {
    let img = binary_volume();
    let kernel = f::StructuringElement::ball(&[3, 3, 3]);
    assert_bit_parity("binary_dilate", 0x6d8c_8ca2_40f6_2325, || {
        f::binary_dilate(&img, &kernel, 1.0, 0.0, false).unwrap()
    });
}

#[test]
fn op09_signed_maurer_distance_map() {
    let img = binary_volume();
    assert_bit_parity("signed_maurer_distance_map", 0xee6b_c90c_6273_e5ad, || {
        f::signed_maurer_distance_map(&img, false, false, true, 0.0).unwrap()
    });
}

#[test]
fn op10_connected_component() {
    let img = binary_volume();
    assert_bit_parity("connected_component", 0x6fcb_a1e1_3b24_854f, || {
        f::connected_component(&img, false).unwrap()
    });
}

#[test]
fn op11_otsu_threshold() {
    let img = float_volume();
    assert_bit_parity("otsu_threshold", 0xdb21_aa5b_e685_6e85, || {
        f::otsu_threshold(&img, 128, false, 1, 0).unwrap().0
    });
}

/// The computed threshold is an `f64` the histogram search produces; pin its
/// bits too, not just the binarized image's.
#[test]
fn op11_otsu_threshold_value_is_thread_count_independent() {
    let img = float_volume();
    let mut seen: Option<u64> = None;
    for threads in [1usize, 3, 8, 32] {
        let t = parallel::with_threads(threads, || f::otsu_threshold(&img, 128, false, 1, 0))
            .unwrap()
            .1;
        if let Some(prev) = seen {
            assert_eq!(
                prev,
                t.to_bits(),
                "otsu threshold varied with the thread count"
            );
        }
        seen = Some(t.to_bits());
    }
}

/// The pin below survived a *sanctioned* algorithm change, and that is worth
/// recording rather than leaving for the next reader to rediscover.
///
/// The FFT was rewritten twice under this checksum: first to a half-Hermitian
/// R2C/C2R pair, then to the `rustfft`/`realfft` kernels. The second moves the
/// `f64` spectrum by a few ulps (rustfft computes the roots of unity that
/// pocketfft hardcodes — see `fft::LineKernel`), so the pin was released for
/// op12, and op12 only, to be re-pinned at whatever the new kernel produced.
/// It did not need re-pinning: this op's output is `Float32`, and a relative
/// perturbation of ~1e-14 in `f64` is ~7 orders of magnitude below the gap
/// between neighbouring `f32`, so every pixel rounds to the bits it had.
///
/// The checksum is therefore still the one harvested from the pre-parallel
/// scalar implementation, and it still means what the module doc says it means.
/// What it does *not* do is gate the FFT's accuracy — it cannot see a change
/// this far down. `fft_matches_spatial_convolution` does that, and it is the
/// stronger of the two: it gates against being *wrong*, not against changing.
#[test]
fn op12_fft_convolution() {
    let img = float_volume();
    // A 7³ normalized box kernel, per `doc/bench-spec.md`.
    let kernel = Image::from_vec(&[7usize, 7, 7], vec![1.0f32 / 343.0; 343]).unwrap();
    assert_bit_parity("fft_convolution", 0x4eb7_1775_6bb2_0520, || {
        f::fft_convolution(
            &img,
            &kernel,
            true,
            f::ConvolutionBoundaryCondition::ZeroFluxNeumannPad,
            f::OutputRegionMode::Same,
        )
        .unwrap()
    });
}

/// op12 against the definition it is an implementation of.
///
/// `fft_convolution` and `convolution` take the same image, the same kernel and
/// the same boundary condition, and differ only in *how* they compute the sum —
/// one multiplies spectra, the other adds 343 products per voxel. So the FFT is
/// wrong exactly insofar as it disagrees with the direct sum, and no checksum
/// can tell us that: a checksum pins the bits an implementation happens to
/// produce, and both of these implementations were rewritten under it.
///
/// In `f64`, where nothing is lost to the pixel type on the way out, the two
/// agree to `1.8e-12` absolute on values of order `1e3` — about `2e-15`
/// relative, which is the round-off of the transform itself and not something a
/// better implementation would improve on. The gate is set at `1e-10` absolute,
/// ~55x that, so it fires on a real defect and not on a kernel swap; the run
/// prints the figure it actually reached.
///
/// The extent is chosen so that the *odd* path is the one under test. A 27³
/// volume with a 7³ kernel gives a padded extent of `padded_length(27 + 6) = 33`
/// (`33 = 3·11`, so it needs no padding at all), which is odd — the half-spectrum
/// has no Nyquist bin, and C2R must reconstruct the top bin from the conjugate
/// symmetry rather than read it. Every other FFT test here lands on an even
/// extent, so without this one the odd parity would go ungated. 27³ also clears
/// `parallel`'s serial threshold, so this is the parallel path.
#[test]
fn fft_matches_spatial_convolution() {
    const TOL: f64 = 1e-10;
    let n = 27usize;
    let img = Image::from_vec(
        &[n, n, n],
        synth(0x5EED, n * n * n)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    let kernel = Image::from_vec(&[7usize, 7, 7], vec![1.0f64 / 343.0; 343]).unwrap();

    let spectral = f::fft_convolution(
        &img,
        &kernel,
        true,
        f::ConvolutionBoundaryCondition::ZeroFluxNeumannPad,
        f::OutputRegionMode::Same,
    )
    .unwrap();
    let spatial = f::convolution(
        &img,
        &kernel,
        true,
        f::ConvolutionBoundaryCondition::ZeroFluxNeumannPad,
        f::OutputRegionMode::Same,
    )
    .unwrap();

    assert_eq!(spectral.size(), spatial.size());
    let (a, b) = (
        spectral.scalar_slice::<f64>().unwrap(),
        spatial.scalar_slice::<f64>().unwrap(),
    );
    let (worst, at) = a
        .iter()
        .zip(b)
        .enumerate()
        .map(|(i, (&x, &y))| ((x - y).abs(), i))
        .fold((0.0f64, 0usize), |acc, d| if d.0 > acc.0 { d } else { acc });
    println!("max |fft - spatial| = {worst:.3e} at voxel {at}");
    assert!(
        worst <= TOL,
        "fft_convolution disagrees with the direct spatial sum by {worst:.3e} at voxel {at} \
         (spectral {}, spatial {}), tolerance {TOL:.0e}",
        a[at],
        b[at],
    );
}
