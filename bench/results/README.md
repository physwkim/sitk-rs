# Frozen sweep results

Twelve ops x three sizes x two rounds, both harnesses, per `doc/bench-spec.md`.
Taken on the post-allocator-fix, post-GMRG-fix, post-`with_max_len(1)` tree
(the last of those is `363407a`).

## Files

| file | rows | status |
|---|---|---|
| `rust-r1.ndjson` | 144 | valid at `tN`; see the `t1` caveat below |
| `rust-r2.ndjson` | 144 | valid |
| `rust-r3-postfix.ndjson` | 144 | valid. A **third** Rust round on `fd2b372`, after the three `gradient.rs` fixes (out-of-place first pass, fused narrowing, store-don't-accumulate). Run 05:48–06:03, foreign load p50 1.0 / p90 1.4 cores. **Supersedes r1/r2 for `gradient_magnitude` and `gmrg` only** — every other op is unchanged code, and its 60 output checksums were compared against r2's with **none moved**, so the two sets are consistent and either may be quoted for the other ten ops. No C++ round was re-run against it: the ITK numbers it is compared to are `cpp-r2`'s, which is correct — ITK's code did not change. |
| `cpp-tN-r1.ndjson` | 36 | valid |
| `cpp-t1-r1.DISCARDED.ndjson` | 36 | **DISCARDED — do not use. See below.** |
| `cpp-r2.ndjson` | 72 | valid (`t1` + `tN`) |
| `rust-cuda.ndjson` | 144 | valid — the **only** source of `gpu` / `gpu_resident` rows |
| `load-trace-foreign.txt` | — | foreign load, 5 s samples: `total`, `bench`, `foreign` |
| `load-trace-foreign-gpu.txt` | — | same, for the cuda leg |
| `load-trace-total.txt` | — | total busy cores, 5 s samples (first sampler; see below) |

`rust-r{1,2}.ndjson` are default-features builds, so every `gpu` row in them
reads `skipped: the cuda feature is off in this build`. The GPU columns come
from `rust-cuda.ndjson` — the same tree, rebuilt with `--features
sitk-filters/cuda` and re-run on the same quiet box. Its 60 CPU rows carry the
same output checksums as `rust-r2.ndjson`, zero moved, so the cuda build does
not perturb the CPU path and the GPU rows are on the same footing as the rest.

The valid pairs, and the only ones `compare.py` should be given:

    python3 bench/compare.py bench/results/rust-r1.ndjson bench/results/cpp-tN-r1.ndjson
    python3 bench/compare.py bench/results/rust-r2.ndjson bench/results/cpp-r2.ndjson

`cpp-t1-r1.DISCARDED.ndjson` is kept in the tree deliberately. It is the
evidence for the discard, not a spare copy of the data; the `.DISCARDED`
in the name is there so that a `bench/results/*.ndjson` glob cannot feed
it back into a comparison by accident.

## Why the C++ `t1` round-1 leg was discarded

All 36 rows (12 ops x 3 sizes), not a subset.

That leg ran 01:12:33-01:47:27. `load-trace-foreign.txt` records foreign
load bursts inside that window: 01:12-01:18 peaking at 45.9 cores, and
01:36-01:38 peaking at **93.8 foreign cores** on a 96-core box. The C++
`t1` config is single-threaded and has nowhere to hide from that.

The corruption is visible in the data itself, not only in the trace. At
medium, comparing the discarded round-1 leg against the clean round-2 leg:

| op | r1 median | r1 stddev | r2 median | r2 stddev |
|---|---|---|---|---|
| `mean` | 2325.3 | **502.5** | 1661.6 | **0.5** |
| `gradient_magnitude` | 475.9 | 2.1 | 314.3 | 1.3 |
| `otsu_threshold` | 1111.6 | — | 780.3 | — |

Every one is *slower* in round 1, which is the direction contention pushes.
The round-2 C++ `t1` leg ran 02:15:41-02:49:10, a foreign-clean window, and
its per-sample stddev collapses to well under 1% on the same ops.

## Two caveats that are NOT load, and must not be read as load

1. **Rust round-1's first ~5 minutes are ungated.** The foreign-load sampler
   started at 01:02:40; that leg began at ~00:57:47. There is no foreign-load
   data for the gap. No `t1` number from `rust-r1.ndjson` was used in the
   reported table; the `tN` legs of both rounds ran in verified-clean windows
   and agree with each other.

2. **Rust `t1` for `mean` and `gradient_magnitude_recursive_gaussian` is
   intrinsically noisy** — sample stddev 306/351 ms and 484/506 ms — in *both*
   rounds, including the clean one. That is the measurement, not contention.
   Their `t1` ratios are soft; quote the `tN` numbers, whose stddev is 0.6-1.4 ms.

## Why there are two load traces

`load-trace-total.txt` records *total* busy cores, so during a `tN` leg it
reads 90+ cores from the benchmark itself and cannot distinguish the
benchmark from a foreign co-runner. It is kept only because it covers the
first five minutes that the second sampler missed.

`load-trace-foreign.txt` subtracts the benchmark processes' own CPU time
(`/proc/<pid>/stat` utime+stime for `bench_ops` / `sitk_bench_cpp`) from the
`/proc/stat` total, leaving foreign load. That is the trace the gate is built
on. A sample reading `total=77.0 bench=76.7 foreign=0.3` is a quiet box under
a full-machine benchmark; the total-only trace would have called it a burst.

`loadavg` is not used anywhere and must not be: it reads 18-21 on this box
with nothing running.

### One artefact in the foreign traces, so nobody re-derives it

A sample reading `foreign=2345.6` appears once, in `load-trace-foreign-gpu.txt`,
at the instant the benchmark process exits. It is an accounting bug in the
sampler, not load: when the process vanishes, its `/proc/<pid>/stat` disappears,
the cumulative-CPU delta for `bench` goes negative, and `total - bench` blows up.
A value above 96 on a 96-core box is impossible, which is how it is recognisable.
Every other sample in that leg is under 4 cores, p50 0.8.

## `rust-r4-grain.ndjson` — 2026-07-14, merged main (post grain seam)

The full twelve-op × three-size sweep on merged `main`, run on a box whose foreign
load had gone (load average 32 → under 3; traced beside the sweep, and the only
peaks are the sweep's own 96-thread ops). **It supersedes `rust-r3-postfix.ndjson`
and rounds 1–2 for every row**, and it is the file behind §3's rewritten 64³ table.

What it changed: the 64³ cells, and only those — `otsu_threshold` 6.62× ITK → 0.68×,
`gradient_magnitude` 2.91× → 1.05×, `mean` 4.52× → 2.82×.

What it did **not** change, despite appearances: three cells (`gradient_magnitude`
medium and large, `discrete_gaussian` large) read as regressions against r3 and are
**noise** — see `twin-r4/` for the four-leg ABBA campaign that settled it, and §3's
"noise floor at large" for what that costs this document's claims.

## `twin-r4/` — the ABBA campaign that exonerated the `Stencil` refactor

Five raw legs (four full published-path sweeps plus a 12-round subset), kept in a
subdirectory **on purpose**: a `bench/results/*.ndjson` glob must not be able to feed
a twin leg into a published comparison. See its `PROVENANCE.md`.
