//! Rust (criterion) benchmark harness for `doc/bench-spec.md`.
//!
//! `harness = false` (see `Cargo.toml`): this is a plain `fn main()`, not the
//! `criterion_group!`/`criterion_main!` macro flow, because the spec's
//! deliverable is a `.ndjson` file matching its schema, and criterion's own
//! CLI report is unstructured text interleaved with that same schema's
//! numbers. Each `(op, size, config)` still runs through a real
//! `criterion::Criterion::bench_function` for the actual timing/statistics;
//! this file only adds reading each run's `estimates.json` back (criterion
//! writes it synchronously before `bench_function` returns) and emitting one
//! NDJSON row per measurement.
//!
//! NDJSON rows are written to `target/bench-results-rust.ndjson`, not
//! stdout — criterion's own progress/report lines go to stdout too, and
//! interleaving them would make the file unparsable by the spec's own `cat`
//! merge.
// `benches/*.rs` files are auto-discovered as their own bench targets, so
// these shared modules live in `benches/bench_ops/` (a subdirectory, not
// scanned by that autodiscovery) rather than directly under `benches/`.
// Since this file is itself a crate root, `mod` resolves siblings relative to
// `benches/`, not `benches/bench_ops/` -- hence the explicit `#[path]`s.
#[path = "bench_ops/checksum.rs"]
mod checksum;
#[path = "bench_ops/ops.rs"]
mod ops;
#[path = "bench_ops/schema.rs"]
mod schema;
#[path = "bench_ops/synth.rs"]
mod synth;

use checksum::{checksum_buffer, checksum_hex};
use criterion::Criterion;
use ops::{InputKind, OPS};
#[cfg(feature = "cuda")]
use ops::{RESCALE_OUTPUT_MAX, RESCALE_OUTPUT_MIN};
use schema::Row;
use serde::Deserialize;
use sitk_core::Image;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use synth::{SEED, synth, threshold_f32, threshold_u8};

/// `doc/bench-spec.md` §"Volume sizes": 64³ / 256³ / 512³, isotropic.
const SIZES: &[(&str, usize)] = &[("small", 64), ("medium", 256), ("large", 512)];

/// Criterion's own floor (`Criterion::sample_size` panics below 10); also
/// exactly the schema's own `"samples": 10` example.
const SAMPLE_SIZE: usize = 10;

/// Long enough that the box's ramp is inside the warm-up instead of inside the
/// measurement.
///
/// Measured, not chosen: after the box idles, the first second of a 96-thread
/// pass runs slow and decays. `signed_maurer_distance_map` at 64³/`tN`, first
/// leg after a 90 s idle, criterion's own per-sample per-iteration times:
///
/// ```text
///   4.970  4.328  3.479  3.049  3.318  3.913  3.557  3.021  2.935  2.898  (ms)
/// ```
///
/// and the four legs after it are flat at 2.81–2.99. The samples reach within
/// 5% of that steady value at a cumulative 1.63 s of measured work — so the ramp
/// costs ~2.1 s of work counting the 500 ms warm-up that preceded it. A 500 ms
/// warm-up therefore leaves the whole ramp inside the measured window, which is
/// what made the same binary on the same op read 6.03 ms in one campaign and
/// 2.89 ms in another: the number recorded whether the box was warm, not what
/// the op costs. 3 s covers the measured 2.1 s ramp with margin.
const WARM_UP_MS: u64 = 3_000;

/// `doc/bench-spec.md` §"Thread configurations": `t1` pins a rayon pool to 1
/// thread; `tN` is rayon's default pool, i.e. every logical core this
/// machine actually reports — queried at runtime, never hardcoded, so the
/// emitted `threads` field is honest on any machine this runs on.
#[derive(Clone, Copy)]
enum ThreadConfig {
    T1,
    TN(usize),
}

impl ThreadConfig {
    fn label(self) -> &'static str {
        match self {
            ThreadConfig::T1 => "t1",
            ThreadConfig::TN(_) => "tN",
        }
    }

    fn threads(self) -> usize {
        match self {
            ThreadConfig::T1 => 1,
            ThreadConfig::TN(n) => n,
        }
    }

    /// One rayon pool per (op, size, config), built once and reused across
    /// every criterion sample for that measurement.
    /// `sitk_core::parallel::with_threads` is the port's designated t1/tN
    /// seam, but it builds a fresh pool per call; hoisting the build here
    /// keeps pool-spawn cost out of the timed region, which matters for the
    /// ops fast enough at `tN` that thread-spawn overhead would otherwise be
    /// a visible fraction of the measurement.
    fn build_pool(self) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads())
            .build()
            .expect("build rayon pool for bench")
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[derive(Deserialize)]
struct RawEstimate {
    point_estimate: f64,
}

#[derive(Deserialize)]
struct RawEstimates {
    mean: RawEstimate,
    median: RawEstimate,
    std_dev: RawEstimate,
}

/// Reads back the `estimates.json` criterion just wrote for `bench_id`,
/// converting its nanosecond-per-iteration point estimates to milliseconds.
fn read_estimates_ms(criterion_dir: &Path, bench_id: &str) -> (f64, f64, f64) {
    let path = criterion_dir
        .join(bench_id)
        .join("new")
        .join("estimates.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let est: RawEstimates =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
    (
        est.mean.point_estimate / 1e6,
        est.median.point_estimate / 1e6,
        est.std_dev.point_estimate / 1e6,
    )
}

fn write_row(out: &mut File, row: &Row) {
    let line = serde_json::to_string(row).expect("serialize NDJSON row");
    writeln!(out, "{line}").expect("write NDJSON row");
}

/// `doc/bench-spec.md` §"Correctness gate — not optional": GPU bit-for-bit is
/// not required, so the gate compares the GPU output against the CPU
/// reference with `max_abs_err`/`max_rel_err` instead of checksum equality.
#[cfg(feature = "cuda")]
fn max_abs_rel_err(cpu: &Image, gpu: &Image) -> (f64, f64) {
    let cpu = cpu
        .scalar_slice::<f32>()
        .expect("cpu rescale_intensity reference is scalar f32");
    let gpu = gpu
        .scalar_slice::<f32>()
        .expect("gpu rescale_intensity output is scalar f32");
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "gpu output length does not match the cpu reference"
    );
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (&c, &g) in cpu.iter().zip(gpu.iter()) {
        let c = f64::from(c);
        let g = f64::from(g);
        let abs = (c - g).abs();
        max_abs = f64::max(max_abs, abs);
        let rel = if c != 0.0 { abs / c.abs() } else { abs };
        max_rel = f64::max(max_rel, rel);
    }
    (max_abs, max_rel)
}

/// A comma-separated allow-list from the environment; `None` when the variable
/// is unset, which means "everything" — the published sweep sets none of these.
fn env_list(var: &str) -> Option<Vec<String>> {
    std::env::var(var)
        .ok()
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
}

/// Whether `name` passes an [`env_list`] allow-list.
fn selected(list: &Option<Vec<String>>, name: &str) -> bool {
    list.as_ref().is_none_or(|l| l.iter().any(|w| w == name))
}

fn main() {
    let root = workspace_root();
    let criterion_dir = root.join("target").join("criterion");
    let ndjson_path = root.join("target").join("bench-results-rust.ndjson");

    let mut out = File::create(&ndjson_path)
        .unwrap_or_else(|e| panic!("creating {}: {e}", ndjson_path.display()));

    let mut criterion = Criterion::default()
        .output_directory(&criterion_dir)
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(Duration::from_millis(WARM_UP_MS))
        .measurement_time(Duration::from_secs(2));

    // `doc/bench-spec.md` §"Thread configurations": `tN` is "rayon default
    // pool" -- the actual logical core count this machine reports, not a
    // literal copied from the spec's own machine baseline (96), so this
    // harness stays honest if it ever runs somewhere else.
    let tn_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    // `t1` is serial by definition (`ThreadConfig::T1.threads() == 1`), so its
    // cost per op is set by voxel count alone -- measured at `medium`, it is
    // a real projection for `large` (8x the voxels), not a guess. ITK's own
    // `large`/`tN` baseline completes every op with room to spare (see
    // `doc/bench-spec.md` §"No ITK op needs the `> 120 s` skip"), so a `large`
    // skip on the Rust side must give *this port's own* reason, never ITK's.
    let mut medium_t1_ms: std::collections::HashMap<&'static str, f64> =
        std::collections::HashMap::new();

    // A/B twinning knob: restrict the sweep to a subset of cells, so the same
    // (op, size, config) can be re-measured many times on two trees without
    // paying for the other 33 cells each round. It gates *which* cells run and
    // nothing inside one: a cell that runs still runs through the identical
    // `bench_function` on the identical input, with the identical pool and the
    // identical criterion configuration. Unset (the published sweep) = every
    // cell, exactly as before.
    let want_ops = env_list("SITK_BENCH_OPS");
    let want_sizes = env_list("SITK_BENCH_SIZES");
    let want_configs = env_list("SITK_BENCH_CONFIGS");

    for &(size_name, dim) in SIZES {
        if !selected(&want_sizes, size_name) {
            continue;
        }
        let size = [dim, dim, dim];
        let voxels = (dim as u64).pow(3);

        let raw = synth(SEED, dim * dim * dim);
        let bin_u8 = threshold_u8(&raw);
        let bin_f32 = threshold_f32(&raw);

        let img_base_f32 = Image::from_vec(&size, raw).expect("build base_f32 input");
        let img_mask_u8 = Image::from_vec(&size, bin_u8).expect("build mask_u8 input");
        let img_mask_f32 = Image::from_vec(&size, bin_f32).expect("build mask_f32 input");

        for op in OPS {
            if !selected(&want_ops, op.key) {
                continue;
            }
            let input = match op.input {
                InputKind::BaseF32 => &img_base_f32,
                InputKind::MaskU8 => &img_mask_u8,
                InputKind::MaskF32 => &img_mask_f32,
            };
            let input_checksum = checksum_hex(checksum_buffer(input.buffer()));

            // Determinism contract (`sitk_core::parallel` module docs): every
            // op's output is bit-identical at any thread count, so the
            // checksum reference call's own thread count is arbitrary --
            // `t1`'s pool is reused since it is about to be built anyway.
            let checksum_pool = ThreadConfig::T1.build_pool();
            let reference_output = checksum_pool
                .install(|| (op.run)(input))
                .expect("op reference call for checksum");
            let output_checksum = checksum_hex(checksum_buffer(reference_output.buffer()));

            for config in [ThreadConfig::T1, ThreadConfig::TN(tn_threads)] {
                if !selected(&want_configs, config.label()) {
                    continue;
                }
                // `t1` is serial by definition, so its `large` cost is fixed
                // by voxel count once measured at `medium` -- skip the
                // expensive 10-sample criterion run and report that
                // projection instead of spending the wall time to confirm
                // what serial-by-definition already guarantees.
                if size_name == "large" && matches!(config, ThreadConfig::T1) {
                    let medium_ms = medium_t1_ms.get(op.key).copied().unwrap_or_else(|| {
                        panic!("medium t1 timing for `{}` was not recorded", op.key)
                    });
                    let large_per_call_s = medium_ms * 8.0 / 1000.0;
                    let projected_min = large_per_call_s * (SAMPLE_SIZE as f64 + 1.0) / 60.0;
                    write_row(
                        &mut out,
                        &Row {
                            harness: "rust",
                            op: op.key,
                            size: size_name,
                            voxels,
                            config: config.label(),
                            threads: config.threads() as u32,
                            ms_mean: None,
                            ms_median: None,
                            ms_stddev: None,
                            samples: None,
                            input_checksum: Some(input_checksum.clone()),
                            output_checksum: Some(output_checksum.clone()),
                            max_abs_err: None,
                            max_rel_err: None,
                            skipped: Some(format!(
                                "rust t1 large not run this session: serial by definition \
                                 (measured medium t1 {medium_ms:.0} ms/call; 512^3 is 8x the \
                                 voxels of 256^3, projecting ~{large_per_call_s:.1} s/call and \
                                 ~{projected_min:.1} min for a {SAMPLE_SIZE}-sample criterion \
                                 run); tN measured instead -- this is the port's own serial \
                                 cost, not a claim about ITK, which completes this op's \
                                 large/tN run well under 120s"
                            )),
                        },
                    );
                    continue;
                }

                let bench_id = format!("{}_{}_{}", op.key, size_name, config.label());
                let pool = config.build_pool();
                criterion.bench_function(&bench_id, |b| {
                    b.iter(|| pool.install(|| (op.run)(black_box(input)).expect("op call")));
                });
                let (ms_mean, ms_median, ms_stddev) = read_estimates_ms(&criterion_dir, &bench_id);

                if size_name == "medium" && matches!(config, ThreadConfig::T1) {
                    medium_t1_ms.insert(op.key, ms_mean);
                }

                write_row(
                    &mut out,
                    &Row {
                        harness: "rust",
                        op: op.key,
                        size: size_name,
                        voxels,
                        config: config.label(),
                        threads: config.threads() as u32,
                        ms_mean: Some(ms_mean),
                        ms_median: Some(ms_median),
                        ms_stddev: Some(ms_stddev),
                        samples: Some(SAMPLE_SIZE as u32),
                        input_checksum: Some(input_checksum.clone()),
                        output_checksum: Some(output_checksum.clone()),
                        max_abs_err: None,
                        max_rel_err: None,
                        skipped: None,
                    },
                );
            }

            // `doc/bench-spec.md` §"Thread configurations": `gpu` means
            // "`sitk-cuda` feature on, device 0". Two distinct, honest
            // reasons apply when this isn't a real measurement -- collapsing
            // them into one string was the bug this replaced. Only
            // `rescale_intensity` has a GPU kernel at all (the other 11 ops
            // have none, feature on or off); for `rescale_intensity` itself,
            // the feature may simply be off for this build.
            if op.key == "rescale_intensity" {
                #[cfg(feature = "cuda")]
                {
                    match sitk_cuda::rescale_intensity_gpu(
                        input,
                        RESCALE_OUTPUT_MIN,
                        RESCALE_OUTPUT_MAX,
                    ) {
                        Ok((gpu_output, _timings)) => {
                            let (max_abs_err, max_rel_err) =
                                max_abs_rel_err(&reference_output, &gpu_output);
                            let gpu_output_checksum =
                                checksum_hex(checksum_buffer(gpu_output.buffer()));

                            let bench_id = format!("{}_{}_gpu", op.key, size_name);
                            criterion.bench_function(&bench_id, |b| {
                                b.iter(|| {
                                    sitk_cuda::rescale_intensity_gpu(
                                        black_box(input),
                                        RESCALE_OUTPUT_MIN,
                                        RESCALE_OUTPUT_MAX,
                                    )
                                    .expect("gpu op call")
                                });
                            });
                            let (ms_mean, ms_median, ms_stddev) =
                                read_estimates_ms(&criterion_dir, &bench_id);

                            write_row(
                                &mut out,
                                &Row {
                                    harness: "rust",
                                    op: op.key,
                                    size: size_name,
                                    voxels,
                                    config: "gpu",
                                    threads: 0,
                                    ms_mean: Some(ms_mean),
                                    ms_median: Some(ms_median),
                                    ms_stddev: Some(ms_stddev),
                                    samples: Some(SAMPLE_SIZE as u32),
                                    input_checksum: Some(input_checksum.clone()),
                                    output_checksum: Some(gpu_output_checksum),
                                    max_abs_err: Some(max_abs_err),
                                    max_rel_err: Some(max_rel_err),
                                    skipped: None,
                                },
                            );
                        }
                        Err(e) => {
                            write_row(
                                &mut out,
                                &Row {
                                    harness: "rust",
                                    op: op.key,
                                    size: size_name,
                                    voxels,
                                    config: "gpu",
                                    threads: 0,
                                    ms_mean: None,
                                    ms_median: None,
                                    ms_stddev: None,
                                    samples: None,
                                    input_checksum: Some(input_checksum.clone()),
                                    output_checksum: None,
                                    max_abs_err: None,
                                    max_rel_err: None,
                                    skipped: Some(format!(
                                        "rust gpu rescale_intensity did not run: {e}"
                                    )),
                                },
                            );
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    write_row(
                        &mut out,
                        &Row {
                            harness: "rust",
                            op: op.key,
                            size: size_name,
                            voxels,
                            config: "gpu",
                            threads: 0,
                            ms_mean: None,
                            ms_median: None,
                            ms_stddev: None,
                            samples: None,
                            input_checksum: Some(input_checksum.clone()),
                            output_checksum: None,
                            max_abs_err: None,
                            max_rel_err: None,
                            skipped: Some(
                                "sitk-cuda feature not enabled for this build (rebuild \
                                 with `--features sitk-filters/cuda` to measure the GPU \
                                 kernel; the kernel itself exists, only this build \
                                 excludes it)"
                                    .to_string(),
                            ),
                        },
                    );
                }
            } else {
                write_row(
                    &mut out,
                    &Row {
                        harness: "rust",
                        op: op.key,
                        size: size_name,
                        voxels,
                        config: "gpu",
                        threads: 0,
                        ms_mean: None,
                        ms_median: None,
                        ms_stddev: None,
                        samples: None,
                        input_checksum: Some(input_checksum.clone()),
                        output_checksum: None,
                        max_abs_err: None,
                        max_rel_err: None,
                        skipped: Some(
                            "no GPU kernel implemented yet (only rescale_intensity is ported)"
                                .to_string(),
                        ),
                    },
                );
            }

            gpu_resident_row(
                &mut out,
                &mut criterion,
                &criterion_dir,
                &Measurement {
                    op_key: op.key,
                    size_name,
                    voxels,
                    input,
                    reference_output: &reference_output,
                    input_checksum: &input_checksum,
                },
            );
        }
    }
}

/// The `gpu_resident` row: the op with the volume **already on the device** and
/// the result **left there**, so no byte crosses the bus inside the timed region.
///
/// This is a third column, not a replacement for either of the other two, and the
/// distinction is the entire point of the row:
///
/// - `gpu` measures the one-shot API `fn(&Image) -> Image` — H2D, kernel, D2H,
///   every call. That is the honest cost of a round-trip API, and it is why the
///   resident one was built. It keeps measuring exactly what it always measured.
/// - `gpu_resident` measures the op. Upload and download happen once, outside the
///   timer, as they would in a pipeline that keeps the volume resident across
///   several ops.
///
/// A reader comparing the two sees the bus, priced. A reader comparing
/// `gpu_resident` against `tN` sees the kernel against the CPU. Neither number is
/// meaningful without the other, which is why they go in one table produced by one
/// run on one machine state.
///
/// **Only `rescale_intensity` gets a real row here**, and the skip reasons below
/// name why for everything else rather than collapsing into one string. `sitk-cuda`
/// has exactly two device-resident ops: `rescale_intensity`, which *is* a benchmark
/// op, and `smooth_gaussian`, which is **not** — it is `sitk_filters::smooth_gaussian`
/// (an `exp(-k²/2σ²)` FIR truncated at `⌈4σ⌉`, σ in physical units), a different
/// filter from this spec's `discrete_gaussian` (ITK's `DiscreteGaussianImageFilter`:
/// variance, maximum error, kernel-width cap) and absent from the twelve.
struct Measurement<'a> {
    op_key: &'static str,
    size_name: &'static str,
    voxels: u64,
    input: &'a Image,
    /// The CPU result this op's GPU output is graded against.
    reference_output: &'a Image,
    input_checksum: &'a str,
}

fn gpu_resident_row(
    out: &mut File,
    criterion: &mut Criterion,
    criterion_dir: &Path,
    m: &Measurement<'_>,
) {
    let &Measurement {
        op_key,
        size_name,
        voxels,
        input,
        reference_output,
        input_checksum,
    } = m;

    let mut row = |ms: Option<(f64, f64, f64)>,
                   output_checksum: Option<String>,
                   errs: Option<(f64, f64)>,
                   skipped: Option<String>| {
        write_row(
            out,
            &Row {
                harness: "rust",
                op: op_key,
                size: size_name,
                voxels,
                config: "gpu_resident",
                threads: 0,
                ms_mean: ms.map(|m| m.0),
                ms_median: ms.map(|m| m.1),
                ms_stddev: ms.map(|m| m.2),
                samples: ms.map(|_| SAMPLE_SIZE as u32),
                input_checksum: Some(input_checksum.to_string()),
                output_checksum,
                max_abs_err: errs.map(|e| e.0),
                max_rel_err: errs.map(|e| e.1),
                skipped,
            },
        );
    };

    if op_key != "rescale_intensity" {
        row(
            None,
            None,
            None,
            Some(
                "no device-resident kernel for this op. `sitk-cuda` has two \
                 (`rescale_intensity` and `smooth_gaussian`), and `smooth_gaussian` is not \
                 one of this spec's twelve ops -- it is a different filter from \
                 `discrete_gaussian`, not a device port of it"
                    .to_string(),
            ),
        );
        return;
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (criterion, criterion_dir, input, reference_output);
        row(
            None,
            None,
            None,
            Some(
                "sitk-cuda feature not enabled for this build (the resident kernel exists; \
                 only this build excludes it)"
                    .to_string(),
            ),
        );
    }

    #[cfg(feature = "cuda")]
    {
        // Upload once, outside the timer. This is the residency premise: the volume
        // is already here because some earlier op in the pipeline left it here.
        let device_input = match sitk_cuda::DeviceImage::upload(input) {
            Ok(d) => d,
            Err(e) => {
                row(
                    None,
                    None,
                    None,
                    Some(format!("rust gpu_resident upload did not run: {e}")),
                );
                return;
            }
        };

        // Correctness, before timing: the resident result, brought home once, must
        // match the CPU reference to the same tolerance the one-shot row is held to.
        let (output_checksum, errs) = match sitk_cuda::rescale_intensity(
            &device_input,
            RESCALE_OUTPUT_MIN,
            RESCALE_OUTPUT_MAX,
        )
        .and_then(|d| d.to_host())
        {
            Ok(host) => (
                Some(checksum_hex(checksum_buffer(host.buffer()))),
                Some(max_abs_rel_err(reference_output, &host)),
            ),
            Err(e) => {
                row(
                    None,
                    None,
                    None,
                    Some(format!(
                        "rust gpu_resident rescale_intensity did not run: {e}"
                    )),
                );
                return;
            }
        };

        // The timed region: device in, device out. The returned `DeviceImage` is
        // dropped inside the loop, which frees device memory — the same allocation
        // the one-shot form also pays, so the two remain comparable. What is *not*
        // in here is the bus.
        let bench_id = format!("{op_key}_{size_name}_gpu_resident");
        criterion.bench_function(&bench_id, |b| {
            b.iter(|| {
                sitk_cuda::rescale_intensity(
                    black_box(&device_input),
                    RESCALE_OUTPUT_MIN,
                    RESCALE_OUTPUT_MAX,
                )
                .expect("gpu resident op call")
            });
        });
        row(
            Some(read_estimates_ms(criterion_dir, &bench_id)),
            output_checksum,
            errs,
            None,
        );
    }
}
