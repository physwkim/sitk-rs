//! Why does the mean-squares metric stop scaling at ~24 cores?
//!
//! The experiment holds the **work** constant and varies only the **memory
//! footprint**. The fixed image is always 64³, so every configuration performs
//! exactly 262 144 samples of identical arithmetic — same transform, same
//! trilinear gather, same Jacobian. Only the moving volume changes size, with
//! its spacing scaled so the same physical box is covered:
//!
//! | moving | `MovingImage::buf` | vs L3 (60 MiB / socket) |
//! |--------|--------------------|-------------------------|
//! | 64³    | 2 MiB              | resident                |
//! | 128³   | 16 MiB             | resident                |
//! | 256³   | 134 MiB            | **does not fit**        |
//!
//! A gather that fits in L3 is not fed by DRAM. So if the L3-resident case
//! keeps scaling past 24 threads and the 134 MiB case flattens there, the
//! ceiling is memory bandwidth and not a defect in the decomposition. If *both*
//! flatten at 24, bandwidth is exonerated and the cause is in the code.
//!
//! `arith` is the control: the same fold, the same row staging, the same thread
//! count — with the gather replaced by arithmetic on the sample's own
//! coordinates. It touches no shared volume at all, so it measures what rayon
//! and the in-order combine can do on this machine with memory removed.
//!
//! ```text
//! cargo run --release --example scale_probe -- <moving_size|arith> <threads>
//! ```

use std::time::Instant;

use sitk::core::Image;
use sitk::registration::{CpuBackend, MeanSquaresMetric};
use sitk::transform::Euler3DTransform;

const FIXED: usize = 64;

/// A smooth, non-symmetric field, sampled on an `n³` grid covering the same
/// physical box whatever `n` is.
fn volume(n: usize, shift: f64) -> Image {
    let spacing = FIXED as f64 / n as f64;
    let mut v = vec![0.0f64; n * n * n];
    let c = FIXED as f64 / 2.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (
                    x as f64 * spacing - shift,
                    y as f64 * spacing,
                    z as f64 * spacing,
                );
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                v[(z * n + y) * n + x] = 200.0 * (-(r * r) / (0.18 * FIXED as f64).powi(2)).exp()
                    + 20.0 * (0.4 * r).sin();
            }
        }
    }
    let mut img = Image::from_vec(&[n, n, n], v).expect("volume");
    img.set_spacing(&[spacing, spacing, spacing])
        .expect("spacing");
    img
}

/// The control: the metric's fold shape with the volume gather removed.
fn arith(threads: usize) {
    let n = FIXED * FIXED * FIXED;
    let iters = 20;
    let t = Instant::now();
    let mut sum = 0.0f64;
    for _ in 0..iters {
        sitk::core::parallel::map_rows_fold_in_order(
            n,
            7,
            || (),
            |(), s, row| {
                // Roughly the per-sample flop count of the real path (transform,
                // trilinear weights, 6-parameter Jacobian) — but reading only
                // `s`, so no thread shares a cache line with any other.
                let x = (s % FIXED) as f64;
                let y = ((s / FIXED) % FIXED) as f64;
                let z = (s / (FIXED * FIXED)) as f64;
                let mut acc = 0.0;
                for (k, slot) in row.iter_mut().enumerate() {
                    let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
                    acc += a.sin() * a.cos() + a.sqrt();
                    *slot = acc;
                }
                true
            },
            |_, row| {
                for &r in row {
                    sum += r;
                }
            },
        );
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "{{\"case\":\"arith\",\"threads\":{threads},\"samples\":{n},\"ms\":{ms:.2},\
         \"checksum\":{sum:.6e}}}"
    );
}

/// The same fold, with the parallel compute and the serial in-order combine
/// **timed separately** — the one measurement that says whether the ceiling is
/// Amdahl on the combine or a wall on the parallel side.
fn phases(threads: usize) {
    let n = FIXED * FIXED * FIXED;
    let iters = 20;
    let mut sum = 0.0f64;
    let (mut par_ms, mut ser_ms) = (0.0f64, 0.0f64);

    for _ in 0..iters {
        // Phase A — the parallel compute, exactly the `arith` kernel, staging a
        // 7-wide row per sample. Same primitive the fold uses underneath.
        let t = Instant::now();
        let rows: Vec<[f64; 7]> = sitk::core::parallel::map_indexed(n, |s| {
            let x = (s % FIXED) as f64;
            let y = ((s / FIXED) % FIXED) as f64;
            let z = (s / (FIXED * FIXED)) as f64;
            let mut row = [0.0f64; 7];
            let mut acc = 0.0;
            for (k, slot) in row.iter_mut().enumerate() {
                let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
                acc += a.sin() * a.cos() + a.sqrt();
                *slot = acc;
            }
            row
        });
        par_ms += t.elapsed().as_secs_f64() * 1e3;

        // Phase B — the serial in-order combine: the additions, plus the read of
        // the staging buffer they walk.
        let t = Instant::now();
        for row in &rows {
            for &v in row {
                sum += v;
            }
        }
        ser_ms += t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&rows);
    }
    let (par_ms, ser_ms) = (par_ms / iters as f64, ser_ms / iters as f64);
    println!(
        "{{\"case\":\"phases\",\"threads\":{threads},\"par_ms\":{par_ms:.2},\
         \"serial_combine_ms\":{ser_ms:.2},\"serial_pct\":{:.1},\"checksum\":{sum:.6e}}}",
        100.0 * ser_ms / (par_ms + ser_ms),
    );
}

/// The same two phases, but the staging buffer is **allocated once and reused**
/// instead of once per iteration.
///
/// A fresh 14.7 MiB `Vec` per iteration is not free: glibc returns a block that
/// large to the kernel on `free`, so every iteration re-`mmap`s it and every
/// worker re-faults its own pages, and page faults take the process-wide
/// `mmap_lock`. That contention grows with the thread count, so it can masquerade
/// as a scaling wall in the compute. Reusing the buffer removes it entirely; the
/// difference between this and `phases` is the allocation's share of the ceiling.
fn phases_reuse(threads: usize, mult: usize) {
    let n = FIXED * FIXED * FIXED * mult;
    let iters = 20;
    let mut rows = vec![[0.0f64; 7]; n];
    let mut sum = 0.0f64;
    let (mut par_ms, mut ser_ms) = (0.0f64, 0.0f64);

    for _ in 0..iters {
        let t = Instant::now();
        sitk::core::parallel::map_indexed_into(&mut rows, |s| {
            let x = (s % FIXED) as f64;
            let y = ((s / FIXED) % FIXED) as f64;
            let z = (s / (FIXED * FIXED)) as f64;
            let mut row = [0.0f64; 7];
            let mut acc = 0.0;
            for (k, slot) in row.iter_mut().enumerate() {
                let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
                acc += a.sin() * a.cos() + a.sqrt();
                *slot = acc;
            }
            row
        });
        par_ms += t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        for row in &rows {
            for &v in row {
                sum += v;
            }
        }
        ser_ms += t.elapsed().as_secs_f64() * 1e3;
    }
    let (par_ms, ser_ms) = (par_ms / iters as f64, ser_ms / iters as f64);
    println!(
        "{{\"case\":\"phases_reuse\",\"n\":{n},\"threads\":{threads},\"par_ms\":{par_ms:.2},\
         \"serial_combine_ms\":{ser_ms:.2},\"serial_pct\":{:.1},\"checksum\":{sum:.6e}}}",
        100.0 * ser_ms / (par_ms + ser_ms),
    );
}

/// The hardware's own answer, with rayon removed entirely.
///
/// One thread, the same per-sample kernel, into a private buffer. Run N copies
/// of this as N **separate processes** and the only thing they share is the
/// silicon: no pool, no barrier, no staging buffer, no false sharing, nothing to
/// blame but the machine. If a lone copy takes t and 48 concurrent copies each
/// take 2t, then 48 cores deliver 24 cores of throughput and no decomposition
/// can do better.
fn solo() {
    let n = FIXED * FIXED * FIXED;
    let iters = 20;
    let mut rows = vec![[0.0f64; 7]; n];
    let t = Instant::now();
    for _ in 0..iters {
        for (s, row) in rows.iter_mut().enumerate() {
            let x = (s % FIXED) as f64;
            let y = ((s / FIXED) % FIXED) as f64;
            let z = (s / (FIXED * FIXED)) as f64;
            let mut acc = 0.0;
            for (k, slot) in row.iter_mut().enumerate() {
                let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
                acc += a.sin() * a.cos() + a.sqrt();
                *slot = acc;
            }
        }
        std::hint::black_box(&rows);
    }
    println!("{:.2}", t.elapsed().as_secs_f64() * 1e3 / iters as f64);
}

/// The parallel region alone, back to back, with **no serial phase between the
/// regions** — so the pool's workers never go to sleep.
///
/// `phases_reuse` runs a ~2 ms parallel region, then a ~2 ms serial combine
/// during which all 48 workers have nothing to do and park. If waking them again
/// costs a large slice of the next region, that shows up here as a much better
/// speedup than the same region gets in `phases_reuse`. Same kernel, same buffer,
/// same primitive: the only variable is whether the pool stayed hot.
fn hot(threads: usize) {
    let n = FIXED * FIXED * FIXED;
    let iters = 200;
    let mut rows = vec![[0.0f64; 7]; n];
    let t = Instant::now();
    for _ in 0..iters {
        sitk::core::parallel::map_indexed_into(&mut rows, |s| {
            let x = (s % FIXED) as f64;
            let y = ((s / FIXED) % FIXED) as f64;
            let z = (s / (FIXED * FIXED)) as f64;
            let mut row = [0.0f64; 7];
            let mut acc = 0.0;
            for (k, slot) in row.iter_mut().enumerate() {
                let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
                acc += a.sin() * a.cos() + a.sqrt();
                *slot = acc;
            }
            row
        });
        std::hint::black_box(&rows);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("{{\"case\":\"hot\",\"threads\":{threads},\"par_ms\":{ms:.2}}}");
}

/// `hot`, plus the **two heap allocations per sample** the real metric's compute
/// does: `MovingImage::value_and_physical_gradient` returns a `Vec<f64>` gradient
/// and `ParametricTransform::jacobian_wrt_parameters` returns a `Vec<f64>`
/// Jacobian, both freshly allocated for every one of the 262 144 samples.
///
/// Same kernel, same pool, same buffer as `hot` — the only added variable is
/// ~500 000 malloc/free pairs per region, spread over the workers. If `hot`
/// scales and this does not, the allocator is the parallel phase's ceiling.
fn hot_alloc(threads: usize) {
    let n = FIXED * FIXED * FIXED;
    let iters = 200;
    let mut rows = vec![[0.0f64; 7]; n];
    let t = Instant::now();
    for _ in 0..iters {
        sitk::core::parallel::map_indexed_into(&mut rows, |s| {
            let x = (s % FIXED) as f64;
            let y = ((s / FIXED) % FIXED) as f64;
            let z = (s / (FIXED * FIXED)) as f64;
            // The two per-sample Vecs: a dim-long gradient and a dim × nparams
            // Jacobian, exactly as the metric's compute allocates them.
            let grad: Vec<f64> = vec![x, y, z];
            let jac: Vec<f64> = vec![0.0; 3 * 6];
            let mut row = [0.0f64; 7];
            let mut acc = 0.0;
            for (k, slot) in row.iter_mut().enumerate() {
                let a = grad[0] * 1.000_1 + grad[1] * 0.999_7 + grad[2] * 1.000_3 + k as f64;
                acc += a.sin() * a.cos() + a.sqrt() + jac[k];
                *slot = acc;
            }
            std::hint::black_box((&grad, &jac));
            row
        });
        std::hint::black_box(&rows);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("{{\"case\":\"hot_alloc\",\"threads\":{threads},\"par_ms\":{ms:.2}}}");
}

/// Is the fork-join region slow because every worker is slower, or because a few
/// **stragglers** hold up the join?
///
/// One task per worker, each doing an equal, fixed slice of the same kernel and
/// timing *itself*. Then:
///
/// - every worker slow, `max ≈ mean` → a shared resource (bandwidth, clock) is
///   throttling all of them, and the region is doing the best it can;
/// - `max ≫ mean` → the region's makespan is set by stragglers. Since the slices
///   are equal by construction, a straggler is not imbalance in the *work*: it is
///   a worker that lost its CPU (to another process, or to its SMT sibling).
///
/// The two have opposite fixes, so guessing between them is not an option.
fn straggler(threads: usize) {
    let per = FIXED * FIXED * FIXED / threads;
    let iters = 50;
    let mut worst = Vec::new();

    for _ in 0..iters {
        let wall = Instant::now();
        let times: Vec<f64> = sitk::core::parallel::map_indexed(threads, |w| {
            let t = Instant::now();
            let mut acc = 0.0f64;
            for j in 0..per {
                let s = w * per + j;
                let x = (s % FIXED) as f64;
                let y = ((s / FIXED) % FIXED) as f64;
                let z = (s / (FIXED * FIXED)) as f64;
                for k in 0..7 {
                    let a = x * 1.000_1 + y * 0.999_7 + z * 1.000_3 + k as f64;
                    acc += a.sin() * a.cos() + a.sqrt();
                }
            }
            std::hint::black_box(acc);
            t.elapsed().as_secs_f64() * 1e3
        });
        let wall = wall.elapsed().as_secs_f64() * 1e3;
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let max = times.iter().cloned().fold(0.0f64, f64::max);
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        worst.push((wall, min, mean, max));
    }
    // Report the *best* region of the run: the one least disturbed by whatever
    // else the machine was doing. Its straggler ratio is a lower bound.
    worst.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let (wall, min, mean, max) = worst[0];
    println!(
        "{{\"case\":\"straggler\",\"threads\":{threads},\"wall_ms\":{wall:.2},\
         \"worker_min_ms\":{min:.2},\"worker_mean_ms\":{mean:.2},\"worker_max_ms\":{max:.2},\
         \"max_over_mean\":{:.2}}}",
        max / mean
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let case = args.next().expect("moving size or 'arith'");
    if case == "solo" {
        solo();
        return;
    }
    let threads: usize = args.next().expect("threads").parse().expect("threads");

    sitk::core::parallel::with_threads(threads, || {
        if case == "arith" {
            arith(threads);
            return;
        }
        if case == "phases" {
            phases(threads);
            return;
        }
        if case == "hot_alloc" {
            hot_alloc(threads);
            return;
        }
        if case == "straggler" {
            straggler(threads);
            return;
        }
        if case == "hot" {
            hot(threads);
            return;
        }
        if case == "phases_reuse" {
            let mult = std::env::args()
                .nth(3)
                .map_or(1, |s| s.parse().expect("mult"));
            phases_reuse(threads, mult);
            return;
        }
        let m: usize = case.parse().expect("moving size");

        let fixed = volume(FIXED, 0.0);
        let moving = volume(m, 3.0);
        let bytes = m * m * m * 8;
        let metric = MeanSquaresMetric::new(&fixed, &moving).expect("metric");
        let c = FIXED as f64 / 2.0;
        let tf = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);

        let iters = 20;
        let warm = metric.evaluate(&tf, &CpuBackend);
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(metric.evaluate(std::hint::black_box(&tf), &CpuBackend));
        }
        let eval_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        std::hint::black_box(metric.value(&tf, &CpuBackend));
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(metric.value(std::hint::black_box(&tf), &CpuBackend));
        }
        let value_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        println!(
            "{{\"case\":\"{m}\",\"threads\":{threads},\"moving_mib\":{:.1},\
             \"samples\":{},\"valid\":{},\"eval_ms\":{eval_ms:.2},\"value_ms\":{value_ms:.2},\
             \"metric\":{:.6e}}}",
            bytes as f64 / (1 << 20) as f64,
            metric.sample_count(),
            warm.valid_points,
            warm.value,
        );
    });
}
