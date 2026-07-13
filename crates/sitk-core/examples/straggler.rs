//! Is a fork-join region slow because every worker is slower, or because a few
//! **stragglers** hold up the join?
//!
//! `rayon::broadcast` runs exactly one closure per worker thread — no splitting,
//! no stealing, no imbalance in the work by construction. Each worker does an
//! equal, fixed slice of the same pure-compute kernel and times *itself*. Then:
//!
//! - `max ≈ mean`, and `mean ≈ the single-thread time / 1` → every worker ran at
//!   full speed and the region is as fast as the silicon allows;
//! - `max ≫ mean` → the makespan is set by stragglers. The slices are equal, so a
//!   straggler is not imbalance in the work: it is a worker that lost its CPU —
//!   to another process on the box, or to its own SMT sibling.
//! - `mean` itself inflating with the thread count → a shared resource (memory,
//!   clock) throttling all of them.
//!
//! The three have different fixes, so guessing between them is not an option.
//!
//! ```text
//! cargo run --release -p sitk-core --example straggler -- <threads>
//! ```

use std::time::Instant;

/// The same per-sample kernel the metric's compute is shaped like: transcendental
/// -heavy, reading only its own index, touching no shared memory at all.
fn kernel(lo: usize, hi: usize) -> f64 {
    const N: usize = 64;
    let mut acc = 0.0f64;
    for s in lo..hi {
        let x = (s % N) as f64;
        let y = ((s / N) % N) as f64;
        let z = (s / (N * N)) as f64;
        for k in 0..7 {
            let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
            acc += a.sin() * a.cos() + a.sqrt();
        }
    }
    acc
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .expect("threads")
        .parse()
        .expect("threads");
    let n = 64 * 64 * 64;
    let per = n / threads;

    // The uncontended reference: one worker's slice, run alone on this thread.
    let t = Instant::now();
    std::hint::black_box(kernel(0, per));
    let alone_ms = t.elapsed().as_secs_f64() * 1e3;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");

    // The fixed cost of the region itself: a broadcast that does NO work. Whatever
    // this costs, every fork-join region on this machine pays it before a single
    // useful instruction runs — and the metric opens one per chunk, per
    // evaluation, per optimizer iteration.
    let mut empty = f64::MAX;
    for _ in 0..500 {
        let t = Instant::now();
        pool.broadcast(|_| ());
        empty = empty.min(t.elapsed().as_secs_f64() * 1e3);
    }

    // Best of 50 regions: the one least disturbed by whatever else the machine is
    // doing. Its straggler ratio is a lower bound on the real one.
    let mut best = (f64::MAX, 0.0, 0.0, 0.0);
    for _ in 0..50 {
        let wall = Instant::now();
        let times: Vec<f64> = pool.broadcast(|ctx| {
            let t = Instant::now();
            let w = ctx.index();
            std::hint::black_box(kernel(w * per, (w + 1) * per));
            t.elapsed().as_secs_f64() * 1e3
        });
        let wall = wall.elapsed().as_secs_f64() * 1e3;
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let max = times.iter().cloned().fold(0.0f64, f64::max);
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        if wall < best.0 {
            best = (wall, min, mean, max);
        }
    }
    let (wall, min, mean, max) = best;
    println!(
        "{{\"threads\":{threads},\"empty_region_ms\":{empty:.3},\
         \"slice_alone_ms\":{alone_ms:.3},\"wall_ms\":{wall:.3},\
         \"worker_min_ms\":{min:.3},\"worker_mean_ms\":{mean:.3},\"worker_max_ms\":{max:.3},\
         \"max_over_mean\":{:.2},\"mean_over_alone\":{:.2},\"wall_over_max\":{:.2}}}",
        max / mean,
        mean / alone_ms,
        wall / max,
    );
}
