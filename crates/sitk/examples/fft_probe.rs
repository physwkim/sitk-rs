//! Ledger §4.116: `fft_convolution` at the benchmark's reference configuration —
//! `doc/bench-spec.md`'s `medium` 256³ `base_f32` volume and 7³ box kernel, which
//! pads to 264³. Prints `t1` and `tN`, plus a checksum of the output so a
//! before/after pair can be compared for value drift as well as for time.
//!
//! ```text
//! cargo run --release -p sitk-filters --example fft_probe -- <threads>
//! ```

use std::time::Instant;

use sitk::core::Image;
use sitk::filters::{ConvolutionBoundaryCondition, OutputRegionMode, fft_convolution};

/// `bench_ops::synth`, verbatim: the seed and generator the recorded numbers used.
fn synth(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) % 1000) as f32
        })
        .collect()
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .expect("threads")
        .parse()
        .expect("threads");
    let dim = 256usize;
    let img = Image::from_vec(&[dim, dim, dim], synth(0x5EED, dim * dim * dim)).expect("input");
    let kernel = Image::from_vec(&[7, 7, 7], vec![1.0f32; 343]).expect("7^3 box kernel");

    sitk::core::parallel::with_threads(threads, || {
        let t = Instant::now();
        let out = fft_convolution(
            &img,
            &kernel,
            true,
            ConvolutionBoundaryCondition::default(),
            OutputRegionMode::default(),
        )
        .expect("fft_convolution");
        let ms = t.elapsed().as_secs_f64() * 1e3;

        // A value fingerprint, so a timing run also reports whether the numbers
        // moved. Summed in index order, on one thread: the same bits every run.
        let vals = out.scalar_slice::<f32>().expect("f32 output");
        let mut sum = 0.0f64;
        let mut peak = 0.0f64;
        for &v in vals {
            sum += f64::from(v);
            peak = peak.max(f64::from(v).abs());
        }
        println!("{{\"threads\":{threads},\"ms\":{ms:.1},\"sum\":{sum:.6e},\"max\":{peak:.6e}}}");
    });
}
