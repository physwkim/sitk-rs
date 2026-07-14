# It is contention — and contention is not the lever. The lever is occupancy.

Graded against `rate-prediction.md`, committed (`df7e325`) before any leg of this
round. Box exclusively mine; every kept leg `foreign=none`, atomic rounds, load trace
in `rate/*/legs.log`.

## The `t1` leg — the number the port never had

64³, `t1`, paired within round, 4 rounds:

| op | `t1` @ 2048 | `t1` @ 1024 | 1024/2048 | ns/voxel |
|---|---|---|---|---|
| `mean` | 75.53 ms | 75.21 ms | **0.995** [0.99, 1.01] | 288 |
| `median` | 221.00 ms | 221.56 ms | **1.002** [1.00, 1.00] | 843 |
| `binary_dilate` | 137.82 ms | 138.91 ms | **1.005** [1.00, 1.03] | 526 |
| `gradient_magnitude` | 9.27 ms | 9.33 ms | **1.007** [1.00, 1.02] | 35 |

**P18 — HELD.** The grains agree at `t1`, to within 0.5–0.7%. A single worker's address
stream does not care how the range was cut, exactly as predicted. **Locality is dead:
the parallel win is caused, not inherited.**

**P19 — FAILED, and my baseline was wrong by 2.5× in the direction that flattered me.**
I predicted `W(64³) ≈ 15–25 ms` for `mean`, reasoning that a 1 MB input is
cache-resident and so cheaper per voxel than the 256³ leg's 117 ns/voxel. It is
**75.5 ms — 288 ns/voxel**, *2.5× more expensive per voxel*, not less. The arithmetic I
failed to do: the non-interior fraction is `1 − (60/64)³ = 17.6%` at 64³ against
`1 − (252/256)³ = 4.6%` at 256³, and boundary pixels take the slow window path. Small
volumes are *boundary-heavy*, and that swamps the cache advantage. The extrapolation
was begging the question and the leg was the only cure.

With real `W`, the true efficiencies at 96 workers: `mean` **5.2×** coarse → **16.8×**
fine; `gradient_magnitude` **2.3×** → **5.9×**. Not "somewhat under-parallel."

**P22 — FAILED.** I predicted the coarse grain would beat serial by only ~1.4×; it is
5.2×. The prediction rode on P19's wrong `W`.

## The thread sweep

Fine/coarse ratio (>1 = the fine grain wins), 3 rounds each, `t1` baseline from above:

| op | P=8 | 16 | 32 | 48 | **64** | 96 |
|---|---|---|---|---|---|---|
| `mean` | 1.16 | 1.24 | 2.66 | 1.55 | **1.56** | 3.16 |
| `median` | 1.03 | 1.07 | 1.22 | 1.63 | **2.37** | 2.33 |
| `gradient_magnitude` | 2.02 | 1.97 | 3.08 | 2.84 | **3.00** | 2.49 |

**P21 — HELD, and it is the held-out cell.** At `t64`, 128 tasks is a *perfect two-wave
fit*: zero quantisation idle, so **every packing model predicts no win at all**. The
fine grain wins **1.56× / 2.37× / 3.00×**. Packing is now dead by measurement as well as
by the R2 bound.

**P20 — MIXED, and I will not claim it.** `median` grows cleanly with `P` (1.03 → 2.33),
which is the contention signature I predicted. But `gradient_magnitude` is **already at
2.02× on 8 workers**, where 96-way sharing does not exist. A ratio that is large at `t8`
is not explained by contention among many workers, so the prediction as written is not
confirmed.

## What is contended, and how I know — and why it is not the lever

A probe inside `fill_indexed` itself, on a **warm pool** (the bench harness builds its
pool once and reuses it; see the correction below), recording per-worker nanoseconds
inside the chunk body and comparing against the pass wall. At `t96`:

| grain | chunks | wall | **total CPU (busy)** | **occupancy** = busy/wall | chunk cost vs its own `t1` |
|---|---|---|---|---|---|
| 2048 | 128 | 7–10 ms | **144–192 ms** | **14–28** of 96 | **1.96×** |
| 1024 | 256 | 4.7–7.6 ms | **142–180 ms** | **20–38** of 96 | **2.02×** |

Two findings, and they must not be conflated:

**1. The memory system does contend, it costs ~2×, and it is grain-independent.** Every
chunk costs almost exactly twice its serial cost when 96 workers run — **1.96× coarse,
2.02× fine** — and the *total CPU burned is the same at both grains* (~150–180 ms
against 75 ms of serial work). So contention is a **flat 2× tax on this pass**, paid
identically either way. **It cannot be the 3.3×**, and any fix aimed at it (blocking,
tiling, NUMA placement) would leave the grain result exactly where it is.

**2. What the grain changes is occupancy.** `wall = busy / occupancy`, and occupancy is
the only term that moves: 14–28 → 20–38 (`mean`), 8–15 → 20–32 (`gm`). **The pass never
holds more than a quarter to a third of the pool**, and a coarse chunk leaves it
emptier. That — not scheduling, not participation, not bandwidth — is what the fine
grain buys, and it is why the win survives at `t64` where packing predicts nothing.

This also dissolves the `gm`-at-`t8` anomaly that broke P20: occupancy is low even on 8
workers (4–6 of 8), so the grain pays there too. Occupancy, not `P`-scaled sharing, is
the common term.

## What remains unexplained, stated as such

**Why does a warm 96-worker pool hold only 20–38 workers inside a 256-leaf pass?** The
leaves are independent, `with_max_len(1)` forces the split to single chunks, and every
worker demonstrably touches the pass (probe v1: 88–96 participate). Occupancy is now a
*measured quantity* rather than an inference, and it is the right metric — but its
ceiling is a rayon scheduling question I have not answered, and I am not going to name a
cause I have not measured. The next probe is the one that timestamps steal/execute
transitions per worker, which turns "the workers are idle" into "the workers are idle
*here*."

## A correction I owe, because I nearly reported the artifact

The first version of this probe called `parallel::with_threads(96, …)` per iteration —
and `with_threads` **builds and tears down a fresh 96-thread pool on every call**
(`parallel.rs:266`). It was timing pool construction, and it reported occupancy ~10 and
walls that did not match the bench cell (10 ms against the harness's 4.5 ms). That
mismatch is what caught it. The numbers above are from a pool built once and reused,
which is what `bench_ops` does. **The cold-pool numbers are discarded, not reported.**
