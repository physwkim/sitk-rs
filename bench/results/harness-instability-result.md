# The harness measured the box's warm-up and called it the op

Graded against `harness-instability-prediction.md`, committed (`c1e7c92`) before any
leg. Box exclusively mine except where noted; every kept leg carries its
`/proc/stat` busy-core trace and foreign-process list, and contaminated legs are
dropped rather than averaged in.

## The mechanism, in one sentence

**After the box has been idle, the first ~2 s of a 96-thread pass runs up to 70%
slow and decays — and criterion's warm-up was 500 ms, so the ramp was inside the
*measurement* window, not the warm-up.** The harness was reporting how warm the box
was, and the harness's own cell order (`for size { for op { t1 then tN } }`) put a
*seconds-long single-threaded `t1` leg* immediately before every `tN` leg, which is
exactly what cools 95 of the 96 cores. Every `tN` number in this repository was
taken on a box that had just been idled by the harness itself.

The evidence, criterion's own per-sample per-iteration times,
`signed_maurer_distance_map` 64³/`tN`, first leg after a 90 s idle, then the four
legs after it:

```
leg 0   4.970  4.328  3.479  3.049  3.318  3.913  3.557  3.021  2.935  2.898   mean 3.287
leg 1   2.787  2.845  2.926  2.894  3.656  2.977  3.065  2.890  2.937  2.822   mean 2.986
leg 4   2.993  2.858  2.792  2.739  3.208  3.205  2.857  2.707  2.597  2.593   mean 2.805
```

The samples reach within 5% of the steady value at a cumulative **1.63 s of measured
work**, so the ramp costs ~2.1 s counting the 500 ms warm-up before it. With
`warm_up_time` raised to 3 s the same cold leg is **flat** — `2.627 2.682 2.888
2.720 2.890 2.778 2.669 2.738 2.764 2.742`, mean 2.755, within 4% of the warmest leg
— and the ramp is gone from the measured window.

## Grades

**P33 — FALSIFIED, and that is what redirected the round.** I predicted 20 launches
of the same binary on the same op would reproduce the 2× spread and come back
bimodal. On a quiet box, 16 launches came back at **2.86–3.40 ms, spread 1.19×,
unimodal**, 14 of 16 inside ±5%. **There is no per-process lottery.** The variance
was never a property of the process; it is a property of what ran *before* it.

**P34 — FAILED both ways, which is the useful outcome.** I predicted slow legs would
show ≥10× the minor faults of fast legs and a clock within 20%. Faults moved by
**2×**, not 10× — and the fault count is a *consequence* of the slow mode, not its
cause: pinning glibc's allocator (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_`
at 1 GiB) cut faults per iteration 391 → 328 and **did not remove the slow mode**
(paired A/B over 16 launches: the slow leg was still `pair0 A`, the first launch
after the idle). H10 is dead.

The clock half is dead too, and more decisively: the 5.66 ms leg ran at **2889 MHz**
and a 2.93 ms leg at **2946 MHz** — a 2× move in time across a **2% move in clock**.
`schedutil` and the 800–4100 MHz range are a red herring. **H9 is dead.**

**H8 (NUMA first-touch) — dead by P33.** A placement lottery has to be per-process,
and per-process variance is 1.19×.

**H11 (op-set) — ALIVE, and it is the second mechanism.** I wrote in the prediction
that the 6.03/2.89 pair were "both single-op processes"; **that was wrong, and I
checked it instead of trusting it.** The 6.03 ms leg was `campaign-SOLO-dist` r3
*first leg after a quiet gate*; the flat ~2.9 ms legs were later legs of the same
round. The old campaign logs say it plainly once you read them in run order —
`DIST4` r1: **6.13, 6.06, 4.17, 3.14**, then r2 and r3 flat at ~2.9.

## The second mechanism: the op-set flips short cells by 2×

Not everything is the ramp. In a **twelve-op** process, several short `tN` cells at
64³ flip between two modes a factor of two apart, and the mode changes from round to
round *in both arms of a paired A/B*:

| op, 64³ `tN` | modes seen in a 12-op process | measured **solo**, 6 launches | solo spread |
|---|---|---|---|
| `rescale_intensity` | 0.40 / 0.89 | **0.412–0.431** | 1.05× |
| `discrete_gaussian` | 1.96 / 3.87 | **1.90–2.13** | 1.12× |
| `smoothing_recursive_gaussian` | 2.11 / 4.14 | **1.97–2.19** | 1.11× |
| `gmrg` | 7.06 / 14.30 | — | — |

**Solo, the bimodality is gone and every cell lands on the fast mode.** So the slow
mode is inherited from the ops that ran earlier in the same process. I have *not*
identified what carries it — the candidates I can rule out are the ones above (clock,
NUMA, allocator threshold, per-process placement); heap layout is ruled out too,
because the mode flips between rounds with the process shape held fixed, and a heap
layout would be deterministic. **This is named, bounded, and unexplained, and it is
in UNFIXED.** The protocol does not need it explained — it needs it *avoided*, which
one op per process does.

## What the old harness was reporting, cell by cell

Same source, two binaries differing only in `warm_up_time` (500 ms vs 3 s), full 64³
sweep, three rounds, order alternating. `old/new > 1.00` means **the frozen harness
reported the op slower than it is**:

| op, 64³ | `t1` old/new | `tN` old/new |
|---|---|---|
| `gradient_magnitude_recursive_gaussian` | 1.41 | **2.02** |
| `signed_maurer_distance_map` | 1.02 | **1.87** |
| `discrete_gaussian` | 1.23 | **1.86** |
| `gradient_magnitude` | 1.40 | **1.47** |
| `connected_component` | **1.45** | 0.99 |
| `mean` | **1.43** | 1.26 |
| `binary_dilate` | 1.00 | 1.21 |
| `median` | 1.00 | 1.17 |
| `fft_convolution` | 1.02 | 1.11 |
| `otsu_threshold`, `rescale_intensity`, `smoothing_recursive_gaussian` | ≈1.00 | 0.95–1.03 |

Output checksums are identical across both arms at every cell: this changes what the
clock sees, never what the code computes.

At **256³** the same paired test gives 0.89–0.99 — **no resolvable warm-up defect**,
because a 60–120 ms iteration is not what a 2 s ramp is long compared to. The defect
is a small-cell defect. I did **not** measure it at 512³ (see UNFIXED).

## The protocol

Three parts. Each one is load-bearing and each was measured, not assumed:

1. **Warm-up 3 s** (`WARM_UP_MS`, `bench_ops.rs`). Derived from the ramp: 2.1 s
   measured, 3 s taken. Not fitted to a benchmark — fitted to the ramp's own decay.
2. **One op per process** (`SITK_BENCH_OPS=<op>` per launch). Kills the op-set
   bimodality above.
3. **Quiet gate** on `/proc/stat` busy cores (`< 3.0`) and a foreign-process list;
   a contaminated leg is dropped, not averaged. `/proc/loadavg` reads 18–21 on an
   idle box here and gates nothing.

Published number = **median of ≥6 launches**, with the spread printed beside it.
`bench/run_protocol.py` is the protocol; it refuses to certify a cell whose spread
exceeds the noise floor.

### The proof: the same binary, measured twice

Two independent campaigns per op, same binary, nothing else changed:

| op, 64³ `tN` | campaign 1 median | campaign 2 median | agreement |
|---|---|---|---|
| `signed_maurer_distance_map` | 2.651 (n=10) | 2.637 (n=10) | **1.005×** |
| `mean` | 1.746 (n=8) | 1.725 (n=8) | **1.012×** |
| `rescale_intensity` | 0.425 (n=6) | 0.427 (n=6) | **1.005×** |
| `smoothing_recursive_gaussian` | 2.016 (n=6) | 1.954 (n=6) | **1.032×** |
| `discrete_gaussian` | 2.097 (n=6) | 1.995 (n=6) | **1.051×** |

### The noise floor, measured

Worst within-campaign spread: **1.13× at 64³** (n≥6), **1.08× at 256³** (n=6),
**1.15× at 512³** (n=4). Worst cross-campaign median disagreement: **5.1%**.

> **A ratio below 1.15× is not a result.** Anything this document or
> `doc/bench-results.md` claims below that number is noise wearing a decimal point.

## What the protocol still cannot certify

`gradient_magnitude` at 64³/`tN` fails its own noise floor (spread 1.22× over 3
launches, wall ~0.5–0.7 ms). That is not a harness defect — it is Round 7's UNFIXED
item arriving as a measurement problem: the op's wall is at the pool wake-up floor
(~0.9 ms, measured in Round 6), so the number *is* the ramp. The driver flags it and
refuses to certify it, which is the correct behaviour and not a workaround.

## UNFIXED

- **The op-set's 2× mode is avoided, not explained.** One op per process makes it go
  away; I cannot say what carries it between ops inside a process. Clock, NUMA
  placement, allocator threshold and per-process placement are all excluded above.
- **512³ was never re-measured under the paired old/new test.** The warm-up defect is
  a small-cell defect at 64³ and absent at 256³; I am *assuming* nothing about 512³
  and have marked its rows unverified in the audit rather than claim they are safe.
- **The C++ side was not re-measured.** `bench/cpp/main.cxx:466` warms up with a
  **single call** — at 64³ that is a few ms of work against a ~2 s ramp, so ITK's 64³
  rows carry the same defect and probably harder. Both columns of every 64³ ratio in
  `doc/bench-results.md` are therefore contaminated, and I did not quantify the ITK
  half. Re-running it needs the C++ harness rebuilt.
- **`gradient_magnitude` 64³ has no certifiable number.** See above.
