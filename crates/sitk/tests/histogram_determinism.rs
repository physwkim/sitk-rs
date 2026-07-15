//! The device histogram's pins — the reduction a device Mattes would be built on.
//!
//! The claim is not "accurate". It is:
//!
//! 1. **deterministic** — same binary, same input, same bits, every run;
//! 2. **launch-configuration independent** — the value does not move with the block size;
//! 3. **bit-identical to the host** — exactly `for i in 0..n { h[k[i]] += v[i] }` in `f64`.
//!
//! Every one of those is worthless without the anti-vacuity that goes with it, so each is
//! paired with a measurement showing the property is *not* free on this input: the same
//! entries summed in a different order give different bits, and the `atomicAdd` histogram
//! — the one everybody writes — actually does return different bits from run to run.
#![cfg(feature = "cuda")]

use sitk::cuda::{CudaError, backend};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

/// An entry list with the two properties that make a float sum order-sensitive: heavy
/// contention (many entries per bin) and a wide dynamic range (large and tiny values in
/// the same bin, so adding a small value to a large partial sum loses low bits).
///
/// This is not adversarial input for its own sake — it is what a Mattes joint histogram
/// looks like: millions of samples, a few thousand bins, Parzen weights spanning orders
/// of magnitude.
fn entries(n: usize, nbins: usize) -> (Vec<u32>, Vec<f64>) {
    let mut keys = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    // A cheap deterministic PRNG — the input must be the same on every run, or the
    // determinism pin below would be testing the generator.
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..n {
        keys.push((next() % nbins as u64) as u32);
        // Values spanning ~13 orders of magnitude, both signs.
        let e = (next() % 27) as i32 - 13;
        let m = (next() % 1_000_000) as f64 / 1e6;
        let sign = if next() % 2 == 0 { 1.0 } else { -1.0 };
        vals.push(sign * m * 10f64.powi(e));
    }
    (keys, vals)
}

/// The host histogram summed in **reverse** entry order — the same numbers, a different
/// order, and therefore (for this input) different bits.
///
/// This is the anti-vacuity for everything below. If summing in a different order gave
/// the same bits, then the order would not matter, `atomicAdd` would be fine, and this
/// whole module would be ceremony.
fn host_reversed(keys: &[u32], vals: &[f64], nbins: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nbins];
    for (&k, &v) in keys.iter().zip(vals.iter()).rev() {
        out[k as usize] += v;
    }
    out
}

fn bits_differ(a: &[f64], b: &[f64]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// **The order matters on this input** — so every claim below is falsifiable.
///
/// Summing the same entries in reverse gives different bits in a large fraction of the
/// bins. A reduction that got the order wrong would therefore be *visible*, which is what
/// makes the determinism and bit-identity pins mean something.
#[test]
fn the_summation_order_is_visible_on_this_input() {
    let nbins = 2500;
    let (keys, vals) = entries(1 << 20, nbins);
    let forward = sitk::cuda::histogram_host(&keys, &vals, nbins);
    let reverse = host_reversed(&keys, &vals, nbins);

    let differ = bits_differ(&forward, &reverse);
    assert!(
        differ > nbins / 10,
        "forward and reverse summation of the same {} entries agree on {}/{nbins} bins; \
         this input cannot see a wrong summation order, so it cannot pin one",
        keys.len(),
        nbins - differ
    );
    println!("{differ}/{nbins} bins differ between forward and reverse host summation");
}

/// **Pin 1: the device histogram is bit-identical to the host's naive loop.**
///
/// Not "agrees to 1e-15". The device sorts the entries by bin, stably, and sums each
/// bin's segment left to right — which *is* the host's `h[k[i]] += v[i]` in ascending `i`,
/// operation for operation. So the bits are equal, and the previous test says a different
/// order would not have been.
#[test]
fn the_device_histogram_is_the_host_histogram_bit_for_bit() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    for (n, nbins) in [
        (1usize, 1usize),
        (255, 7),
        (4096, 2500),
        (1 << 20, 2500),
        (1 << 22, 64),
    ] {
        let (keys, vals) = entries(n, nbins);
        let host = sitk::cuda::histogram_host(&keys, &vals, nbins);
        let device = sitk::cuda::histogram(&keys, &vals, nbins).expect("device histogram");

        let differ = bits_differ(&host, &device);
        let first = host
            .iter()
            .zip(device.iter())
            .enumerate()
            .find(|(_, (h, d))| h.to_bits() != d.to_bits());
        assert_eq!(
            differ, 0,
            "n={n} nbins={nbins}: {differ}/{nbins} bins differ; first {first:?}"
        );
        println!("n={n} nbins={nbins}: all {nbins} bins bit-identical to the host");
    }
}

/// **Pin 2: the same input gives the same bits, run after run.**
///
/// Eight runs of the same binary on the same entries. Every bin, every run, the same bits.
/// This is the property `atomicAdd` cannot offer and the next test measures it failing to.
#[test]
fn the_device_histogram_is_deterministic_run_to_run() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let nbins = 2500;
    let (keys, vals) = entries(1 << 21, nbins);

    let first = sitk::cuda::histogram(&keys, &vals, nbins).expect("device histogram");
    for run in 1..8 {
        let again = sitk::cuda::histogram(&keys, &vals, nbins).expect("device histogram");
        let differ = bits_differ(&first, &again);
        assert_eq!(
            differ, 0,
            "run {run} differs from run 0 in {differ}/{nbins} bins — the reduction is not \
             deterministic and nothing built on it can be pinned"
        );
    }
    println!(
        "8 runs, {} entries, {nbins} bins: identical bits every run",
        keys.len()
    );
}

/// **Pin 3: the value does not move with the launch configuration.**
///
/// The block size changes how the entries are counted — how many threads, how many blocks,
/// which entry lands on which SM — and changes nothing about the answer. A reduction whose
/// value moved with the block size would be a reduction whose value moves with the machine,
/// and the pin on the host would be a pin on this box.
#[test]
fn the_result_does_not_depend_on_the_launch_configuration() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let nbins = 2500;
    let (keys, vals) = entries(1 << 20, nbins);

    let reference = sitk::cuda::histogram(&keys, &vals, nbins).expect("device histogram");
    for block in [32usize, 64, 128, 256, 512, 1024] {
        let got =
            sitk::cuda::histogram_with_block(&keys, &vals, nbins, block).expect("device histogram");
        let differ = bits_differ(&reference, &got);
        assert_eq!(
            differ, 0,
            "block {block}: {differ}/{nbins} bins differ from the default configuration"
        );
    }
    println!("block sizes 32..1024: identical bits");
}

/// **The measurement that justifies all of the above: `atomicAdd` really is not
/// deterministic.**
///
/// The fast histogram everybody writes, run eight times on the same input, on this
/// hardware. If it agreed with itself every run, the whole module would be solving a
/// problem this box does not have, and I would say so rather than keep the pin.
///
/// It does not agree. The spread is reported, in bins and in relative magnitude, and the
/// test asserts the disagreement exists — the *only* test here that would fail if
/// `atomicAdd` were secretly ordered.
#[test]
fn the_atomic_histogram_is_not_deterministic_and_that_is_why_this_module_exists() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let nbins = 2500;
    let (keys, vals) = entries(1 << 21, nbins);

    let first = sitk::cuda::histogram_atomic(&keys, &vals, nbins).expect("atomic histogram");
    let mut runs_differing = 0usize;
    let mut worst_bins = 0usize;
    let mut worst_rel = 0.0f64;
    for _ in 1..8 {
        let again = sitk::cuda::histogram_atomic(&keys, &vals, nbins).expect("atomic histogram");
        let differ = bits_differ(&first, &again);
        if differ > 0 {
            runs_differing += 1;
            worst_bins = worst_bins.max(differ);
            for (a, b) in first.iter().zip(again.iter()) {
                if a.to_bits() != b.to_bits() && a.abs() > 0.0 {
                    worst_rel = worst_rel.max((a - b).abs() / a.abs());
                }
            }
        }
    }
    println!(
        "atomicAdd histogram, {} entries, {nbins} bins: {runs_differing}/7 re-runs differed \
         from the first; worst run differed in {worst_bins}/{nbins} bins, worst relative \
         difference {worst_rel:e}",
        keys.len()
    );
    assert!(
        runs_differing > 0,
        "the atomicAdd histogram returned identical bits on all 8 runs. Then the \
         non-determinism this module exists to avoid is not observable here, and the \
         deterministic reduction should be justified again — or dropped."
    );

    // ...and it is not merely non-deterministic, it disagrees with the host too, which the
    // deterministic one does not.
    let host = sitk::cuda::histogram_host(&keys, &vals, nbins);
    println!(
        "atomicAdd vs host: {}/{nbins} bins differ on the bits",
        bits_differ(&host, &first)
    );
}

/// An entry list that is not a histogram is refused by name, not clamped into one.
///
/// A key outside the bin range is the dangerous one: clamping it would put a sample in the
/// wrong bin and return a perfectly plausible histogram.
#[test]
fn a_malformed_entry_list_is_refused() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    for (what, keys, vals, nbins) in [
        ("mismatched lengths", vec![0u32, 1], vec![1.0f64], 4usize),
        ("no entries", vec![], vec![], 4),
        ("no bins", vec![0u32], vec![1.0f64], 0),
        (
            "a key outside the bins",
            vec![0u32, 4],
            vec![1.0f64, 1.0],
            4,
        ),
    ] {
        match sitk::cuda::histogram(&keys, &vals, nbins) {
            Err(CudaError::HistogramShape(why)) => println!("{what}: refused — {why}"),
            other => panic!("{what} was accepted: {:?}", other.map(|_| ())),
        }
    }
}
