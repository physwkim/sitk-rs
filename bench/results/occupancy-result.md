# The pass is one chunk long: the makespan is a boundary straggler

Graded against `occupancy-prediction.md`, committed (`f04109c`) before the probe existed.
Box exclusively mine; every kept round `foreign=none` (`occupancy/legs.log`).

## Method

A temporary timeline probe inside **`fill_indexed` itself** — the real function, the real
ops — recording `(worker, chunk, start_ns, end_ns)` per chunk on one monotonic clock.
Clocksource is `tsc`, so the two `Instant::now()` calls per chunk cost ~40 ns against a
chunk of 0.2–7 ms: a measurement, not a cost. One pool of 96 workers **built once**
(`with_threads` builds a fresh pool per call — that artifact cost a round already). The
grain is forced from an env var, so **both grains come out of one binary**, and the
grains **alternate per rep**, so neither owns the warm state. 64³, 16 reps × 3 rounds.
The probe is reverted; it is not in the tree.

The first run of this probe was in the **dev profile** — walls of 40–190 ms, 10–50× the
bench cell. Those numbers are discarded, not reported.

## The answer

**The pool holds 20–38 workers because for most of the pass there is nothing to hold.**
At the merged grain the pass is not a 256-way parallel pass at all — it is a short
parallel burst followed by **one chunk running alone**:

| op | g | wall | **c_max** | **c_max / wall** | c_max/c_mean | which chunk |
|---|---|---|---|---|---|---|
| `mean` | 2048 | 12.07 ms | 11.81 ms | **1.00** | 4.3 | **chunk 0**, starts at t=0.01·wall |
| `mean` | 1024 | 8.95 ms | 8.36 ms | **0.93** | 5.7 | chunk 0 |
| `median` | 2048 | 15.74 ms | 15.43 ms | **0.98** | 2.5 | chunk 2 |
| `gradient_magnitude` | 2048 | 3.71 ms | 2.94 ms | **0.79** | 5.0 | chunk 1 |

For coarse `mean`, the last chunk to *finish* is the first chunk to *start*: it begins at
t ≈ 0 and is still running when all 127 others are done. **The wall is that one chunk.**

**And the straggler has a name.** The chunk cost is not random — it is the **z = 0
boundary plane**. Chunk 0 covers flat indices `0..g`, which at 64³ is entirely `z = 0`,
where every voxel takes the slow boundary window path instead of the interior one:

| op | edge chunk | interior chunk | **edge / interior** |
|---|---|---|---|
| `mean` | 9 495 µs | 2 677 µs | **3.7×** |
| `gradient_magnitude` | 2 470 µs | 522 µs | **4.7×** |
| `median` | 11 141 µs | 5 591 µs | **2.0×** |

Backing that out against the `t1` leg (75.5 ms at 64³, 17.6% non-interior) gives a serial
boundary voxel at ≈1.0 µs against ≈130 ns interior — an **8× per-voxel path split**, which
is exactly what `window_at` does per pixel. The flat-index decomposition then concentrates
all of it into the first and last chunks. **Occupancy is low because chunk cost is
heterogeneous by a factor of 8 and the decomposition does not know it.**

## Where the idleness is, in one line each

Three phases, measured, not inferred (fraction of wall with fewer than 10 chunks running,
split at the concurrency peak):

| op | g | **ramp** (<10 running, pre-peak) | **tail** (<10 running, post-peak) | peak concurrency | **unstarted work in the tail** |
|---|---|---|---|---|---|
| `mean` | 2048 | 0.04 | **0.52** | 96 | **0** |
| `mean` | 1024 | 0.06 | **0.23** | 95 | **0** |
| `median` | 2048 | 0.03 | 0.28 | 96 | **0** |
| `gradient_magnitude` | 2048 | 0.13 | **0.49** | 95 | **0** |

The pool **does** fill — peak concurrency is 95–96 of 96, every op, every grain. It then
**collapses**, and in the last 10% of the wall about **2 chunks** are running. And the
last column is the one that settles it: **there is never a single unstarted chunk in the
tail, in any cell.** No worker ever idled while work existed. The idleness is not
starvation, not steal contention, not the join tree. It is 94 workers waiting on one long
boundary chunk.

## Grades

**P23 — FAILED.** I predicted arrival is the bulk of the idleness, with the 90th-percentile
worker arriving past 40% of the wall. Every worker arrives inside the **first ~0.9 ms**;
at the merged grain the 90th percentile is **0.11** of the wall for `mean` and **0.06** for
`median`. The ramp is real but *fixed and small* — and being fixed, it becomes the binding
floor only for the shortest pass (`gradient_magnitude` at grain 128 spends **46%** of its
wall ramping). H4 is a floor, not the mechanism.

**P24 — HELD.** A worker that has arrived stays busy: median inter-chunk gap **2–7 µs**
against a chunk of 0.2–7 ms — **0.1–0.4% of busy**. **H5, split-tree starvation, is dead**,
and dead twice over: the tail never contains unstarted work.

**P25 — FAILED, and in the direction that matters.** I predicted the tail is ≈one chunk
duration and "far too small to be the mechanism". The tail *is* one chunk — but one
**straggler** chunk of 2.5–5.7× the mean cost, and it is **the whole makespan**
(`c_max/wall` = 0.93–1.00). I had the size right and the significance exactly backwards.

**P26 — FAILED.** I predicted plentiful unstarted work while workers idled. In the tail
there is **zero** unstarted work in all 15 cells. Work availability was never the
constraint; work *shape* was.

**H7 (steal contention) — refuted, as the win demanded.** The falsifier was that more tasks
must make gaps worse. Going from 128 to 2048 chunks, the median inter-chunk gap **falls**
(`mean`: 6.9 µs → 2.1 µs). Contention on the deque is not what a coarse grain buys.

## The model, and the cell it was not built on

**P27, written before the sweep** (it is in the turn transcript, not fitted afterwards):
`wall(g) ≈ max(W/P, c_max(g), ramp)` with `c_max ∝ g`. So halving the grain must keep
cutting the wall until it flattens on the `W/P` floor. The sweep — 5 grains × 3 ops, and
grains 512/256/128 are **cells no model in this port was built on**:

| op | W/P floor | g=2048 | 1024 | 512 | 256 | 128 | occupancy 2048 → 128 |
|---|---|---|---|---|---|---|---|
| `mean` | 3.9 ms | 12.07 | 8.95 | 7.20 | 6.00 | **5.73** | **29.5 → 66.7** |
| `median` | 8.2 ms | 15.74 | 13.77 | 11.95 | 10.13 | **9.80** | **50.0 → 80.1** |
| `gradient_magnitude` | 0.7 ms | 3.71 | 2.72 | 2.19 | 1.91 | **1.86** | 19.9 → 36.2 |

**P27 HELD.** Every op falls monotonically and flattens just above its own `W/P` floor,
and `c_max/wall` falls from 1.00 to 0.29 as it does. **`busy` is invariant** across the
whole sweep (`mean` 358–391 ms, ±5%) — the work does not change, only the shape of the
tail. `gradient_magnitude` flattens *above* its floor because at 1.86 ms the fixed ~0.9 ms
ramp is now half the wall: the two floors meet, which is what a floor should do.

## What I cannot claim, stated as such

**The probe reproduces the mechanism but not the magnitude.** Its own coarse/fine ratio is
**1.30–1.36×**, where the bench cell says **3.16×**. The coarse walls agree with the bench
(12.1 ms probe vs 14.1 ms bench for `mean`); the **fine** wall does not (8.95 vs 4.47). So
the straggler explains *why occupancy is 20–38* — the question I was asked — and it
explains the *direction* of the grain win, but I have not shown it accounts for the whole
3.3×, and I am not going to say it does. The gap between the probe's operating point and
the bench harness's is unresolved and belongs in UNFIXED.

## What this makes the next fix, and it is not a grain

The grain is a **workaround for a heterogeneous decomposition**: it helps only because
halving `g` halves the straggler. The structural defect is that `fill_indexed` chunks a
flat index range while the *cost per index* varies 8× between the boundary shell and the
interior — so the chunker's one job, equal-cost pieces, is exactly what it fails at.

The structural fix is to make the decomposition cost-homogeneous: **split the boundary
shell from the interior** at the stencil layer (`par_map_window` knows the radius; the
shell is 17.6% of a 64³ volume and 4.6% of a 256³ one), parallelise each over its own
uniform cost class, and the makespan stops being hostage to which chunk happens to land on
`z = 0`. It is bit-neutral by construction — a map's output does not depend on how the
index range is cut. Predicted effect, before anyone runs it: `c_max` falls from `8·W/n` to
`≈W/n`, so `wall → W/P + ramp` and occupancy goes to 80+ *at any grain* — which would make
`MIN_GRAIN_INDEXED` stop mattering, and that is the tell that it was treating a symptom.
I have not built it and I am not claiming the number; it is the next round's work.
