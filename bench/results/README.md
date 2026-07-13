# Frozen sweep results

Twelve ops x three sizes x two rounds, both harnesses, per `doc/bench-spec.md`.
Taken on the post-allocator-fix, post-GMRG-fix, post-`with_max_len(1)` tree
(the last of those is `363407a`).

## Files

| file | rows | status |
|---|---|---|
| `rust-r1.ndjson` | 144 | valid at `tN`; see the `t1` caveat below |
| `rust-r2.ndjson` | 144 | valid |
| `cpp-tN-r1.ndjson` | 36 | valid |
| `cpp-t1-r1.DISCARDED.ndjson` | 36 | **DISCARDED — do not use. See below.** |
| `cpp-r2.ndjson` | 72 | valid (`t1` + `tN`) |
| `load-trace-foreign.txt` | — | foreign load, 5 s samples: `total`, `bench`, `foreign` |
| `load-trace-total.txt` | — | total busy cores, 5 s samples (first sampler; see below) |

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
