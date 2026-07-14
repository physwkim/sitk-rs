# Benchmark results: sitk-rs vs ITK 6.0 (C++)

Measured under the contract in [`bench-spec.md`](bench-spec.md). Raw rows for
**two independent rounds** are frozen in [`../bench/results/`](../bench/results/),
with the load traces and a `README` naming every row that was discarded and why:

```
python3 bench/compare.py bench/results/rust-r1.ndjson bench/results/cpp-tN-r1.ndjson
python3 bench/compare.py bench/results/rust-r2.ndjson bench/results/cpp-r2.ndjson
```

Both exit 0: every op received a byte-identical input in both harnesses at all
three sizes. That equality is the precondition of the comparison, and it holds.
Sixty Rust rows carry output checksums and **zero moved between rounds**.

## Machine

- 96 logical cores — **48 physical**, 2 sockets × 24, SMT on. This matters: a
  "96×" ceiling does not exist. The all-core clock is 2.9 GHz against 3.6 GHz
  for a lone core, so the honest ceiling against a single-threaded baseline is
  **~38×**, before any software cost.
- 4× NVIDIA RTX 5000 Ada (32 GiB, cc 8.9), CUDA 13.0. Device 0 only.
- ITK 6.0, release, default Pool threader (not TBB), no FFTW.
- rustc 1.97.0, release, criterion.

`t1` = one thread. `tN` = all 96. `gpu` = the one-shot API (`fn(&Image) -> Image`,
so H2D + kernel + D2H are all inside the timed region). `gpu_resident` = the same
kernel with the image already on the device and the output staying there — the bus
is outside the timed region because in a real pipeline it is outside the call.

All figures are `ms_median`. **Ratios below are `rust / itk`, so > 1.00 means the
port is slower.**

## 0. How much of this can you trust?

Read this before quoting any number.

> ### RETRACTION, 2026-07-15: the harness was timing the box's warm-up
>
> **The entire 64³ table below is withdrawn. No 64³ rust/ITK ratio in this
> repository is quotable.** Two defects in the measurement, not in the code:
>
> 1. **The ramp was inside the measurement window.** criterion's warm-up was
>    500 ms; this box's ramp after idling is **~2.1 s of work**. Worse, the
>    harness's cell order is `for size { for op { t1 then tN } }`, so every `tN`
>    leg was immediately preceded by a seconds-long **single-threaded** `t1` leg —
>    which is exactly what cools 95 of 96 cores. **The harness cooled the box and
>    then timed the recovery.** Measured inflation on identical code, paired, three
>    rounds: `gmrg` **2.02×**, `signed_maurer` 1.87×, `discrete_gaussian` 1.86×,
>    `gradient_magnitude` 1.47×.
> 2. **The op-set inside a process flips short 64³ cells between two modes 2×
>    apart**, and the mode changes run to run. Solo, the same cells are stable to
>    1.05–1.12×. This mechanism is **avoided** by the new protocol (one op per
>    process), **not explained** — clock, NUMA, allocator and heap layout are all
>    excluded by measurement.
>
> **The ITK column is not a control**: `bench/cpp/main.cxx:466` warms up with a
> *single call*, so ITK's 64³ rows carry the same defect, harder. **Both columns of
> every 64³ ratio are contaminated.**
>
> **Noise floor under the protocol: 1.13× at 64³, 1.08× at 256³, 1.15× at 512³. A
> ratio inside the floor is not a tie — it is unresolved.** That strikes
> `smoothing_recursive_gaussian` medium (1.02×), `gradient_magnitude` large
> (1.02×) and `gmrg` large (1.01×) on its own, independent of the ramp.
>
> **The rule this document used to state — "a number is comparable only within the
> same sweep shape" — is too weak.** Numbers from the same sweep shape are not
> comparable either, because the mode flips between runs of an identical binary.
> One op per process is the only shape that reproduces. The protocol is
> `bench/run_protocol.py`: one op per process, a `/proc/stat` quiet gate, a median
> of ≥6 launches, and it **refuses to certify** a cell whose spread exceeds the
> floor. It already declines `gradient_magnitude` at 64³.
>
> Mechanism, grades and proof: `bench/results/harness-instability-result.md`.
> Row-by-row verdicts: `bench/results/harness-audit-of-bench-results.md`.
>
> **What survives.** The medium and large wins that sit *far* outside the floor —
> `binary_dilate` 0.02–0.03×, `connected_component` 0.25–0.27×, `median` 0.38–0.39×,
> `rescale_intensity` 0.41–0.42×, `signed_maurer` 0.30–0.46×, and the two real losses
> (`smoothing_recursive_gaussian` 1.75× large, `otsu_threshold` 1.37× large). At 256³
> the paired old/new-harness test found no resolvable defect (0.89–0.99×), which is
> the direct evidence for the medium rows. **512³ was never tested that way**, and a
> `large` cell's ~2 s window is the same order as the ramp, so the defect could be
> *larger* there — the 512³ rows survive on margin, not on a control.
>
> **What does not.** The whole 64³ table and every claim built on it; the three rows
> inside the floor; every `t1` figure in this document (the `t1` leg *is* the cooling
> mechanism, and none has been retaken); and the host *denominators* of the GPU
> speedups in §1 and §2, which inherit the same defect one level down.

- **`/proc/loadavg` is unusable as a gate on this box.** It reads 18–21 with four
  runnable tasks and no benchmark alive. Every number here is gated on real busy
  cores from `/proc/stat`, sampled every 5 s, with the benchmark's *own* load
  subtracted so a `tN` leg is not mistaken for foreign contention. Hours were lost
  to the naive gate before this was understood; the traces are in `bench/results/`.
- **The `tN` numbers are the ones to quote — and as of the retraction above, the `t1`
  numbers are not quotable at all.** Both rounds agree tightly at `tN` (`mean` stddev
  0.6 and 1.4 ms). Two `t1` rows — `mean` and `gmrg` — carry 15–21%
  sample stddev in *both* rounds, including the provably clean one, so their `t1`
  ratios are soft and are marked as such below. That is intrinsic to those
  measurements, not contention.
- **36 rows were discarded, not silently dropped.** All of C++ `t1` round 1: the
  foreign-load trace shows a burst peaking at **93.8 foreign cores** inside that
  leg's window, and a single-threaded benchmark cannot hide from that. The proof it
  was corrupted: `mean` medium has sample stddev **502.5 ms** in round 1 against
  **0.5 ms** in round 2. The condemned rows are committed as
  `cpp-t1-r1.DISCARDED.ndjson` alongside the trace that condemns them, so the
  discard is auditable.
- **Three of the port's wins are against a degraded ITK.** `binary_dilate` and
  `connected_component` beat an ITK that is *slower at 96 threads than at 1* (§5);
  `fft_convolution` runs ITK's VNL backend at `double` because this build has no
  FFTW. Those ratios are real but they are not a fair ITK.
- `median` at `large` has no `itk t1` row: ITK's single-threaded median did not
  finish inside the harness budget. It is absent, not zero.

## 1. The retraction: I measured the bus and concluded about the GPU

An earlier version of this document claimed the GPU cannot win a per-pixel op —
that PCIe was a hard floor and offload was not worth it. **That was wrong, and the
row that refutes it is a row I had already collected.**

`rescale_intensity` — of the twelve benchmark ops, the only one with a device
kernel — on the same tree and the same quiet box as every CPU row above:

| size | CPU tN ˢ | GPU one-shot | **GPU resident** | ITK tN ˢ |
|---|---|---|---|---|
| medium (256³) | 16.9 | 35.8 | **1.06** | 39.7 |
| large (512³) | 108.1 | 205.5 | **4.5** | 266.1 |

ˢ — the two CPU columns come from the twelve-op harness and inherit its defects (§0).
At 256³ the paired old/new-harness test found no resolvable inflation (0.89–0.99×);
at 512³ it was **never run**. So the CPU and ITK denominators here are soft, and both
in the same direction — a slow denominator inflates the GPU's win.

Both facts live in the same row:

- **The one-shot API really does lose to the CPU** — 35.8 against 16.9 at medium.
  This is the one comparison in the section that does **not** depend on a soft
  denominator in a way that could reverse it: the one-shot loses by 2.1×, and the
  bus cost that causes it (~17 ms of transfer at 256³) is arithmetic, not a timing.
- **The device really does win**, by **16×** over the port's own CPU and **37×**
  over ITK. At large: **24×** over CPU `tN`, **59×** over ITK. These four multiples
  are ratios against a soft denominator; the *sign* is not in doubt (1.06 ms against
  16.9 ms is far outside any plausible harness inflation), the *multiple* is.

The only difference between those two GPU columns is whether the bus is inside the
timed region. The gap between them — **34.8 ms at medium, 201.0 ms at large** — is
the cost of the API shape, and **it is larger than the kernel it wraps.** At 256³ an
`f32` volume is 67 MB, so a round trip is ~17 ms of transfer to do ~1 ms of work. I
measured the round trip, and published a verdict about the hardware.

The resident output is **bit-exact** against the CPU reference — `max_abs_err = 0.0`
at every size, not merely inside a tolerance.

**What this means for the API.** A dispatch that hides a round trip inside
`fn(&Image) -> Image` can never win, no matter how fast the kernel is. That is why
`sitk-cuda` exposes `DeviceImage` instead: `upload` and `to_host` are the only two
functions that cross PCIe, and an op's signature (`&DeviceImage -> DeviceImage`)
*cannot express* a round trip. The bus crossing is a thing the caller schedules,
not a thing a filter does behind their back.

Of the twelve benchmark ops, only `rescale_intensity` has a device kernel — the
device op set is larger than that (§6), but none of the rest is a bench op.
`discrete_gaussian` carries a
`skipped` field rather than a number: `sitk-cuda` has no device port of it, and
`smooth_gaussian` — which does exist on the device — is a **different filter**
(physical-units σ, truncated at ⌈4σ⌉) and not a port of ITK's
`DiscreteGaussianImageFilter` (variance, maximum error, kernel-width cap). Its
number is not printed under op03's name.

## 2. The device pipeline, end to end

The per-op table above understates the case, because in a real pipeline the bus is
crossed *once*, not once per filter. Measured at 256³, `UInt16` input, 20 iterations
(`load → cast → rescale → smooth → register`):

Four runs of the identical command, because the first two disagreed by 40% on the
host column and a total built on a coin flip is not a measurement. Utilization gate
from `/proc/stat` (~3 busy cores of 96) — not `loadavg`, which reads 18–21 on this
box with nothing running and **cannot be used as a gate**.

| row | host (CPU tN), 4 runs | device-resident, 4 runs |
|---|---|---|
| cast | 84.0 / 66.3 / 68.1 / 71.4 | on device (upload+cast 28.3–70.6) |
| rescale | 25.9 / 28.6 / 20.7 / 29.4 | 3.0 / 1.9 / 4.7 / 1.9 |
| smooth (both volumes) | **2325 / 1538 / 2103 / 1539** | **14.1 / 9.6 / 16.3 / 9.5** |
| registration setup | 22.2 / 21.8 / 21.8 / 21.0 | 7.5 / 7.5 / 7.4 / 7.5 |
| 20 iterations | 18545 / 13280 / 14755 / 13763 | 163.7 / 160.2 / 157.9 / 152.9 |
| **total** | **21,037 / 14,953 / 16,989 / 15,443** | **258.9 / 240.0 / 247.4 / 200.1** |
| **speedup** | | **81× / 62× / 69× / 77×** |

The metric value is identical in all four runs — host `89.934407782061`, resident
`89.934407781794`, relative error **2.94e-12**.

**Quote this as ~70×, not as a point estimate.** The device column is stable; the
*host* column is not — its `20 iterations` row varies by 40% and its `smooth` row
flips bimodally between ~1,540 and ~2,300 ms. The 62–81× spread is host noise, and
§0's warning about this document's own reproducibility is what produced it.

**That bimodality now has a name, and it is the harness's** (§0 retraction, item 2).
A multi-op process flips short host cells between two modes about 2× apart, and the
mode changes run to run; solo, the same cells are stable to 1.05–1.12×. This
pipeline runs five host stages in one process, so it is exactly the shape that
triggers it. **The consequence for this table is one-directional and it is worth
stating plainly: the host column is the *denominator* of every speedup here, so a
host mode that is 2× too slow inflates the speedup by 2×.** The 62–81× is therefore
an upper-bounded quantity, not a centred one — the honest reading is **"the device is
faster by something between the low tens and ~80×, most likely near the bottom of
that range"**, and it will stay that way until the host column is retaken one-stage-
per-process. The *device* column and the metric agreement (2.94e-12) are unaffected;
neither depends on the host timing.

**The metric-kernel fix cost 15% per iteration, not the 22% this document
previously extrapolated** (138.0 → 152.9–163.7). The kernel had been forming the
continuous index with FMA contraction and with the transform offset seeded into the
accumulator (the host adds it last); a 1-ULP index difference flips `floor()` for a
sample lying exactly on a voxel plane, where the trilinear gradient is
discontinuous, so the kernel took the *opposite one-sided derivative* —
`d/d(angle_y)` off by 34% while the value agreed to 1e-15, which is why every
value-only check passed it. `registration setup` rose 6.8 → 7.5 (+10%, and the most
reproducible number in the table). Both are the price of a derivative that is not
34% wrong.

### 2.1 A real `execute()`, re-measured on the fixed kernel

Not an evaluate loop: `ImageRegistrationMethod::execute()` against
`execute_on_device()`, same input, same convergence criteria, both paths free to
pick their own iteration count. Two runs each, gated on **real utilization from
`/proc/stat`** (1.92 busy cores of 96) — `/proc/loadavg` reads 18–21 on this box
with four runnable tasks and no cargo alive, and is not usable as a gate.

| | host `execute` | `execute_on_device` | registration stage | iterations, both paths |
|---|---|---|---|---|
| single level, 256³ | 23,219 / 18,513 ms | 209 / 210 ms | **111× / 88×** | 24, `StepTooSmall` |
| pyramid `[4,2,1]`, 256³ | 28,224 / 25,819 ms | 291 / 297 ms | **97× / 87×** | 22, `StepTooSmall` |
| pyramid `[4,2,1]`, 128³ | 2,672 / 2,695 ms | 42.1 / 42.3 ms | **63× / 64×** | 20, `StepTooSmall` |

Every run: identical iteration count, identical valid-point count, identical stop
reason on both paths, worst parameter disagreement **3.0e-14 to 2.3e-13**.

Three things a reader should take from this table, two of which contradict what
this document previously implied:

- **The published 107–119× was measured on the broken kernel. It is 88–111×.** The
  device now costs 8.4 ms per iteration against 6.9 ms before — **+22%** by this
  benchmark's arithmetic. §2's `20 iterations` row, where the count is *pinned* and
  so the trajectory cannot move it, puts the same cost at **+15%**. The pinned-count
  number is the cleaner measurement; the honest range is 15–22%, and it is what the
  correct derivative costs.
- **The pyramid is not what costs you the speedup — 87–97× with, 88–111× without.**
  The device pays 83 ms to build the extra levels (209 → 292); the host pays 5–7
  *seconds* and gets nothing back on this input, because a 3-voxel misalignment
  converges fine without a pyramid. A pyramid buys **capture range, not speed**, and
  this synthetic input needs no capture range. Read the host's pyramid row as a fact
  about the input, not as evidence that pyramids are a pessimization.
- **The volume is what costs you: 63× at 128³ against 87–97× at 256³.** The GPU's
  fixed costs stop being amortized. A reader running a 128³ registration should
  expect ~60×, not ~100×.

The Gaussian is where the CPU bleeds: 2,250.7 ms on 96 threads against 9.7 ms on
the device. The device Gaussian is **bit-identical** to the CPU filter, not close
to it — `f64` weights and intermediates, and `__dmul_rn`/`__dadd_rn` to forbid FMA
contraction, because an FMA would be *more* accurate and therefore *different*.

## 3. Results — medium (256³), the reference size

`tN` columns carry both rounds. Sorted by ratio: the port wins at the top.
**Read the ᵘ and ˢ marks before quoting any cell — and the whole `t1` pair of
columns is soft** (see below).

| op | rust tN | itk tN | **rust/itk (tN)** | rust t1 ˢ | itk t1 ˢ |
|---|---|---|---|---|---|
| binary_dilate | 65.6 / 67.1 | 2484 / 2480 | **0.03×** | 2390 | 1708 |
| connected_component | 1126 / 1128 | 4352 / 4483 | **0.25×** | 1190 | 679 |
| signed_maurer_distance_map | 87.9 / 88.7 | 244 / 295 | **0.30×** | 1601 | 3479 |
| median | 203 / 206 | 544 / 540 | **0.38×** | 8550 | 19497 |
| rescale_intensity | 16.0 / 16.9 | 38.0 / 39.7 | **0.42×** | 90.1 | 69.8 |
| otsu_threshold ˢ | 46.5 / 36.7 | 56.8 / 64.1 | 0.57× | 989 | 780 |
| discrete_gaussian ˢ | 114.0 / 131.8 | 162 / 174 | 0.76× | 1329 | 824 |
| **mean** | **62.5 / 62.7** | 80.8 / 78.7 | **0.80×** | 1967 ᵗ | 1662 |
| fft_convolution | 471 / 501 | 587 / 574 | 0.87× | 2148 | 1228 |
| ~~smoothing_recursive_gaussian~~ ᵘ | 64.6 / 67.2 | 52.3 / 66.0 | ~~1.02×~~ | 1005 | 818 |
| **gmrg** | **116.9** ᶠ | 207 / 247 | **0.47×** | 2319 ᵗ | 2426 |
| **gradient_magnitude** | **22.5** ᶠ | 36.2 / 35.1 | **0.64×** | 511 | 314 |

ᵘ — **struck: unresolved, not a tie.** The measured noise floor at 256³ is **1.08×**
and this ratio is 1.02×, i.e. inside it. Worse, the two ITK legs alone span 52.3 and
66.0 ms — 1.26× apart — so the denominator does not agree with itself. This row says
nothing about which implementation is faster, and it was previously read as "parity".

ˢ — **soft: the two legs disagree by more than the floor.** `otsu_threshold` 46.5 vs
36.7 (1.27×) and `discrete_gaussian` 114.0 vs 131.8 (1.16×), against a 1.08× floor.
Their *direction* (both well under 1.0×) survives; their magnitude does not. Quote
"the port wins here", not the number.

ˢ (on the `t1` columns) — **every `t1` figure in this document is soft.** The `t1`
leg is what the harness ran immediately before each `tN` leg, single-threaded and
seconds long, and it is the mechanism that cooled the box (see the §0 retraction).
The `t1` legs are themselves ramp-contaminated, and no `t1` number has been retaken
under the protocol. They are kept because the `tN`/`t1` *scaling* story is still
directionally useful; no `t1` cell is quotable as a measurement.

ᵗ — this `t1` carries 15–21% sample stddev in both rounds, including the provably
clean one. Soft on two counts now; quote the `tN` figure.

ᶠ — a **third** round, run 05:48–06:03 on `fd2b372` after the three `gradient.rs`
fixes in §4.1, foreign load p50 1.0 / p90 1.4 cores. One column, not two, because
it is one run; its 60 output checksums were compared against round 2's and **none
moved**. It supersedes rounds 1–2 for these two rows *only* — every other row in
this table is still rounds 1/2, which is why they still carry two columns.

### large (512³) — `rust t1` is not measured (serial by definition; the harness projects it)

| op | rust tN | itk tN | **rust/itk** |
|---|---|---|---|
| binary_dilate | 286 / 279 | 14812 / 15071 | **0.02×** |
| connected_component | 8418 / 8623 | 30180 / 31409 | **0.27×** |
| median | 1283 / 1298 | 3486 / 3328 | **0.39×** |
| rescale_intensity | 105 / 108 | 261 / 266 | **0.41×** |
| signed_maurer_distance_map | 613 / 581 | 1310 / 1257 | **0.46×** |
| mean | 354 / 354 | 488 / 484 | 0.73× |
| fft_convolution | 2700 / 2711 | 3327 / 3201 | 0.85× |
| discrete_gaussian | 626 / 549 | 600 / 605 | 0.91× |
| otsu_threshold | 260 / 260 | 189 / 190 | 1.37× |
| ~~gradient_magnitude~~ ᵘ | **100.8** ᶠ | 97.9 / 98.8 | ~~1.02×~~ |
| ~~gmrg~~ ᵘ | **770.8** ᶠ | 772 / 761 | ~~1.01×~~ |
| **smoothing_recursive_gaussian** | 386 / 380 | 212 / 218 | **1.75×** |

ᵘ — **struck. These were published as "ties at large"; they are not ties, they are
unresolved.** The measured noise floor at 512³ is **1.15×**, and 1.02× / 1.01× are
well inside it. A ratio inside the floor carries no information about which side is
faster — calling it parity claims a result the measurement cannot support. What the
rows *did* establish stands: both were **2.00×** and **2.93×** before the
`gradient.rs` fixes of §4.1, and those are outside the floor by a wide margin, so
the fixes closed a real gap. Where they landed is not known.

**A caveat that applies to this whole table, not just the struck rows.** The `large`
cells get a ~2 s criterion window and the box's ramp is ~2.1 s of work, so the
ramp-in-window defect (§0 retraction) could be *larger* here than at 64³, not
smaller. The paired old-harness/new-harness test was run at 64³ (up to 2.02×
inflation) and at 256³ (0.89–0.99×, no resolvable defect); **it was never run at
512³.** No 512³ row is asserted to be free of the defect — that is an untested
claim, and it is not being made. The ratios below the floor (0.02×–0.46×) are large
enough that no plausible ramp inflation reverses them; the ones near 1.0× are not.

### small (64³) — WITHDRAWN 2026-07-15

**The table that stood here is retracted in full. See the retraction in §0.**

It reported twelve `rust/itk` ratios, every one of them a `tN` cell measured in a
**twelve-op process** under a **500 ms** warm-up — both harness defects at once, at
the size where both are largest. The rust column was inflated by up to **2.02×**;
the ITK column, whose C++ harness warms up with a single call, carries the same
defect unquantified. **Both halves of every ratio are contaminated, so the ratios
are not merely imprecise — they are unfounded.**

Three named claims died with it, and they are named here rather than deleted,
because each was quoted elsewhere in this document and in the README:

- **"`mean` still loses at 64³, by 2.82×."** Not reproducible. That rust number
  (14.4 ms) was taken *before* the cost-class split **and** with the ramp inside the
  window. Under the protocol, on merged main, `mean` 64³/`tN` is **1.73–1.75 ms**
  across two independent campaigns. The ITK half has not been re-measured, so **no
  replacement ratio is claimed** — only that the old one cannot stand.
- **"`otsu_threshold` crossed to a win, 0.68×"** and **"`gradient_magnitude` is now
  a tie, 1.05×."** Both sit inside or beside the 1.13× floor *and* were taken with
  the ramp in the window. `gradient_magnitude` at 64³ **cannot be certified at all**
  under the protocol: its wall is at the pool wake-up floor and it fails its own
  spread test (1.22× > 1.13×). The protocol refuses to print it.
- The **"was"** column (6.62×, 4.52×, 2.91×) came from the same defective harness.
  The *direction* of the grain-seam win is independently corroborated by the ABBA
  controls (§"noise floor"), whose control cells sit far outside the floor. The
  *magnitudes* are not corroborated by anything.

What the port actually did to its 64³ performance is not in doubt — the cost-class
split (§5.3) is proved bit-neutral and its within-protocol, paired, same-binary
comparisons stand. What is in doubt is **every number that compared it to ITK.**
Retaking this table needs both halves: `bench/run_protocol.py` for the rust column,
and a C++ harness whose warm-up covers the ramp for ITK's. Neither is done.

### The noise floor at `large`, measured — and what this table may therefore claim

Publishing `rust-r4-grain.ndjson` surfaced three cells that appeared to have
*regressed* against the r3 table: `gradient_magnitude` medium (22.5 → 28.8 ms),
`gradient_magnitude` large (100.8 → 109.8), `discrete_gaussian` large (0.91× →
1.05×). A four-leg **ABBA twin** (full published path, ~15 min per leg, r3's tree
against merged main, ordered post/pre/pre/post so a monotonic drift cancels rather
than loading onto one tree) settled it:

| cell (tN) | pre legs | post legs | median post/pre |
|---|---|---|---|
| `gradient_magnitude` medium | 22.77, 25.74 | 24.42, 19.61 | **0.91×** |
| `gradient_magnitude` large | 98.15, 104.79 | 97.99, 100.04 | **0.98×** |
| `discrete_gaussian` large | 580.1, 638.4 | 655.7, 587.5 | **1.02×** |
| `gradient_magnitude` small *(control)* | 7.30, 7.22 | **2.44, 2.53** | **0.34×** |
| `discrete_gaussian` small *(control)* | 6.84, 6.40 | **3.04, 2.78** | **0.44×** |

Every suspect cell's range **overlaps**; the r4 tree measures `gradient_magnitude`
large at 98.0/100.0 ms — *below* r3's published 100.8 and nowhere near the 109.8
attributed to it. **A regression the regressing code cannot reproduce is not a
regression.** The control cells' ranges do *not* overlap, so the 64³ wins above are
real and reproduce. The refactor suspected of causing it was also innocent by
inspection: `gradient_magnitude_pass` was **already** on the borrowed-window path
at `fd2b372`, and `gaussian_axis_pass` is byte-identical between the two trees —
the hypothesis (a materialized copy amortized at 512³) described a copy the *older*
tree had already deleted.

What that costs this document is a claim, and it is the honest price:

- **Process shape moves a cell by ~60%.** The identical binary measures
  `gradient_magnitude` large at **155 ms** in a two-op process and **98 ms** in the
  twelve-op sweep. A number is comparable only *within the same sweep shape*.
- **Within-leg sample stddev at large is 9–25 ms** (`gradient_magnitude`) and
  **46–94 ms** (`discrete_gaussian`) — the latter is larger than the entire gap it
  was invoked to explain.
- **Within-tree drift across one campaign:** `gradient_magnitude` large on one tree
  alone walked 163.9 → 144.0 ms in a single session.

So: **a `large` ratio in this document is worth about ±15%, and a difference
smaller than that is not a result.** The 1.02× and 1.01× "ties" at large were
already written as ties rather than wins, which survives; but no `large` row here
should be read as a two-significant-figure fact, and a future change claiming a
sub-15% win at `large` on this box is claiming something this harness cannot
currently see. Quantifying `discrete_gaussian`'s 8–15% per-sample variance is
**open and not chased**.

## 4. Where the port lost to ITK, and what the cause turned out to be

Four ops in the **separable / stencil** family used to lose. Two of them no longer
do — and the numbers below are what a reader should check the story against.

| op, medium/tN | before | **after** |
|---|---|---|
| `mean` | 4.39× | **0.80×** — now *beats* ITK |
| `gradient_magnitude` | 4.25× | 1.51× → **0.64×** — now beats ITK (§4.1) |
| `gmrg` | 1.99× | 1.22× → **0.47×** — now beats ITK (§4.1) |
| `discrete_gaussian` | 1.25× | **0.76×** — now beats ITK |

The shape said what it was: the port's `t1` was competitive-to-better on several of
these and it was the `tN` column that fell behind. A **scaling** gap, not a
constant-factor one.

**Three of the four were the allocator.** Not the kernel, not the decomposition,
not the barrier count, not NUMA, not bandwidth — each eliminated by measurement: a
pure-compute region scales 43.1× on this box; a streaming map at 16 flops/element
scales 33.4×; one socket with all memory local scales *identically* to two; 125
loads from a single L1-hot address hit the same ceiling as the real 25-stream
window. The threads were not stalled, they were **blocked** — at t48 the window
walk ran 13.8 of 48 cores while the identical kernel through `map_indexed` ran
43.0. Idle cores mean a lock, and the lock was glibc's.

`mean` was making **30,910,860 heap allocations per call**.
`smoothing_recursive_gaussian` — the op in the same family that *beats* ITK —
makes 14,926, because it never constructs a `NeighborhoodIterator`. That single
difference was the whole gap. Two sites, both on the neighborhood **boundary**
path: `push_values_checked` built two ND buffers per boundary voxel, and every
`BoundaryCondition::get_pixel` impl `collect()`ed a `Vec` per out-of-bounds
neighbor.

| allocations per call, 256³ | before | after |
|---|---|---|
| `mean` r=2 | 30,910,860 | **13,212** |
| `median` r=2 | 30,913,804 | **16,191** |
| `discrete_gaussian` | 9,854,298 | **35,994** |
| `gradient_magnitude` | 4,318,099 | **12,049** |

Fixed structurally rather than by pooling: `boundary::remapped` folds each
condition's per-axis rule straight into a linear index, so no implementation ever
has an ND index to materialize, and `push_values_checked` now takes `nd: &mut [i64]`
— **a slice cannot grow, so the function has no way to allocate**. All 16
`bit_parity` checksums are unmoved: same pixels, same fold order, same bits.

### The fourth op was a *different* defect — and was not folded into the first

`gradient_magnitude_recursive_gaussian` made only 44,429 allocations and never
touched the boundary path, so it was kept out of the allocator finding rather than
attributed to it on the strength of being on the same list. Its cause turned out to
be the *bytes*, not the count: **15 full-volume `f64` buffers per call, 2.01 GB at
256³.** `smoothing_recursive_gaussian` — which runs the *same* IIR and beats ITK —
makes **one**, 134 MB. Fifteen times the memory traffic for three times the compute.
Fifteen huge `Vec`s is fifteen `malloc`s, which is why the allocation *count* said
innocent while the bytes said guilty.

Same structural anchor as the first fix: `recursive_gaussian_f64_into(buf: &mut [f64], …)`
holds the axis loop, so **a caller holding a `&mut [f64]` has no way to allocate a
volume** — the copy is unwritable, not merely unwise. Five sites in the family were
fixed, not one: `gmrg` (15 → 3 buffers), `laplacian_recursive_gaussian` (11 → 3),
`gradient_recursive_gaussian`, `coherence_enhancing_diffusion`, and `level_set`.
op07's checksum did not move.

### The "second ceiling" on `mean` did not exist. Retracted.

An earlier version of this section reported that `mean` scaled perfectly to 16
threads and then stopped dead at ~20 busy cores, and filed it as an open defect. **On
a box that could be *proven* quiet, `mean` scales t1 1966.6 → tN 62.7 ms = 31.4×,
with no plateau and no wall.** The shape was an artifact of a `t1` baseline measured
under foreign load. There *was* a real defect underneath — rayon left one un-split
leaf task running the tail of the region alone, fixed by `with_max_len(1)` on
`fill_indexed` — but that is a different defect from the one that was filed, and it
is not being credited as its resolution.

## 4.1 Three more fixes, and the two ops that used to lose no longer do

The two rows above that still lost were **not** one defect either, and neither was
what the allocator story predicted.

**`gmrg` large, 2.93×.** The 15→3 buffer fix had swapped a **parallel** `f32→f64`
widening for a **serial** `copy_from_slice`. `vec![0.0; n]` is a lazily-zeroed
mmap, so the page-fault bill lands on whichever phase writes the buffer first —
and that was the serial copy: **517 ms** on axis 0 at 512³, against **79.7 ms**
for the parallel widening of a buffer of exactly the same size. Same bytes, same
fresh pages, 6.5× apart, and the only difference is whether the first touch is
parallel. That is why the fix won at 256³ and lost at 512³. **The copy was deleted,
not parallelized** — the first filtered axis now reads `src` and writes `dst`
out-of-place; parallelizing would have spread 6.4 GB of traffic instead of removing
it.

**`gradient_magnitude`, every size.** It materialized a full `f64` volume (1.07 GB
at 512³) that existed only to be read back and narrowed to `f32` by the very next
pass. ITK never materializes it: it writes the output pixel once. The window pass
now emits the output type directly; `gradient_magnitude_values` is the *same
kernel* instantiated at `f64` for `watershed_classic`, so the two cannot drift.

**`gmrg`'s accumulator.** A `+=` into a fresh `alloc_zeroed` buffer costs **2.00
page faults per page** — the read faults in the shared zero page copy-on-write, and
the write then takes a second write-protect fault on the same page. A pure store
costs **1.00** (376.9 ms → 39.3 ms on 1.07 GB). The first axis now stores.

That last one is legal **only** because `gmrg`'s term is `g*g`, which can never be
`-0.0`, and `(+0.0) + x == x` bitwise for every value a square can take. The
Laplacian's term is a *second* derivative that **can** be `-0.0`, and
`(+0.0) + (-0.0) == +0.0` — so converting its first accumulate to a store would
emit `-0.0` where the add emitted `+0.0` and move a checksum. It keeps its zeroed
buffer, and the `-0.0` divergence itself is pinned by a test rather than asserted
in a comment. A blanket "same fix, three sites" sweep would have moved a checksum
here.

Measured, ITK at its best on both ops (neither is touched by the two ITK threading
regressions in §5, so nothing here borrows credit from a degraded baseline):

| op | size | ITK | rust before | rust after | after/ITK | was |
|---|---|---|---|---|---|---|
| `gradient_magnitude` | medium | 35.1 | 53.0 | **22.5** | **0.64×** | 1.51× |
| `gradient_magnitude` | large | 98.8 | 197.6 | **100.8** | 1.02× | 2.00× |
| `gmrg` | medium | 247.1 | 301.1 | **116.9** | **0.47×** | 1.22× |
| `gmrg` | large | 761.4 | 2231.5 | **770.8** | 1.01× | 2.93× |

Both beat ITK at medium. **The `large` pair was published as "ties"; both are struck
— 1.02× and 1.01× are inside the 1.15× floor and settle nothing.** What survives is
the `was` column: 2.00× and 2.93× are far outside the floor, so the §4.1 fixes closed
a real gap; where they landed is not measured.

### What still loses, stated plainly — **rewritten 2026-07-15, most of it withdrawn**

*(This list has now been wrong twice, in opposite directions, and both times the
error was a number the harness handed it. Written before §5.2, it attributed
`gradient_magnitude`'s 64³ loss to "fixed per-call overhead"; that was the grain.
Rewritten after §5.2, it then quoted the 64³ table — which §0 retracts in full. Six
of its eight entries were 64³ ratios and **every one of them is gone**. What is left
is the honest remainder.)*

**Still a loss, outside the noise floor, and measured at a size the ramp defect was
tested at:**

- **`smoothing_recursive_gaussian`: 1.75× at large.** Well outside the 1.15× floor.
  Its time is in the line pass, whose own task floor is still under-raised (§6). This
  is the port's largest remaining loss and the only one I would act on today.
- **`otsu_threshold`: 1.37× at large.** Outside the 1.15× floor. A real loss.

**Withdrawn — every 64³ entry.** `mean` 2.82×, `gradient_magnitude` 1.05×,
`otsu_threshold` 0.68×, `gmrg` 1.19×, `fft_convolution` 1.41×: all six were `tN`
cells from the retracted table, measured with the box's ramp inside the window and
against an ITK column with the same defect unquantified. **This document currently
does not know whether the port wins or loses against ITK at 64³, on any op.** The
`mean` entry in particular was the headline of §6 and of the README; under the
protocol `mean` 64³/`tN` measures 1.73–1.75 ms against the 14.4 ms that ratio was
built on, and the ITK half has not been retaken at all.

**Unchanged, because it never depended on a ratio:**

- Not benchmarked and not measured, but sitting next to the code just fixed:
  `derivative`, `laplacian` and `sobel_edge_detection` still run a **serial**
  `iter().map().collect()` over the old copying neighborhood path, plus a full
  `f64` scratch copy of the input. None of the three is in the benchmarked twelve,
  so no number is claimed for them — they are named because they are un-parallelized
  stencils, not because they are known to be slow.

`fft_convolution` **is closed**: it was 6.5× slower than ITK at `t1` before
rustfft/realfft landed and the real-input half-Hermitian path replaced three full
complex transforms of real data. It is now 0.87× at medium and 0.85× at large —
against an ITK with no FFTW, which is why it is not claimed as more.

## 5. Two ITK multithreading regressions, found by this benchmark

ITK gets *slower* with threads on two ops. These are upstream defects, recorded as
ledger §7.1 and §7.2 — they are performance defects, not correctness ones, which is
why they are not in §1 of the ledger.

- **`connected_component`**: `itk t1` 679 ms → `itk tN` **4352 ms** at medium. Six
  times slower on 96 threads than on one. At large: 5475 → 30,180.
- **`binary_dilate`**: `itk t1` 1708 ms → `itk tN` **2484 ms** at medium.

The port's win on these two (`0.03×`, `0.25×`) is therefore **against ITK's threaded
time, which is worse than its own single-threaded time**. Against ITK's *own best*
number, `binary_dilate`'s 37× win becomes **26×** and `connected_component`'s 4×
win becomes a **1.7× loss**. Stated here so the headline ratio is not read as more
than it is.

## 5.1 A fix that was measured, worked, and was withdrawn anyway

Worth recording because the *refutation* is the finding, and because a reader who
sees "64³ is per-call overhead" in §4.1 above deserves to know it was tested.

The hypothesis was that at 64³ the 96-thread pool is too wide for the work: the
same stencil pass cost 4.28 ms on 96 workers and 2.28 ms on 8. A rule was written —
`workers = clamp(len / 32768, 1, ambient)`, at one place all five parallel entry
points cross — and it was verified to be **live on the published path**: a counter
in the dispatcher recorded that the narrow pool ran on *every* call
(`NARROWED=915/1132/228`, zero inline, zero nested, zero refused).

It still had to be withdrawn. Twins through the harness's real path (fresh
96-thread pool per measurement, `pool.install` per iteration, criterion's own
statistics, five alternating rounds each):

| op, 64³ tN | pre-fix | post-fix | |
|---|---|---|---|
| `mean` | 21.74 ms | **11.77** | 1.85× — distributions do not overlap |
| `otsu_threshold` | 15.16 ms | **13.49** | 1.12× — do not overlap |
| `gradient_magnitude` | 6.07 ms | 6.65 | **0.91× — a 9% tax** |

**The rule keys on the wrong quantity.** All three ops have 262,144 elements, so
element count cannot tell them apart — but `mean` is a 125-tap window, `otsu` a
histogram, and `gradient_magnitude` a 6-tap stencil. The narrow pool wins where the
per-element work is heavy and *loses* where the kernel is light. The constant
`32768` had been derived from the gradient-magnitude stencil on the **direct** call
path, which is not the path the published table measures; on the published path
that crossover does not reproduce for the op it was derived from. Re-fitting the
constant until `gradient_magnitude` stopped complaining would have been
curve-fitting to the benchmark, so the branch was dropped rather than tuned.

What survives is the measurement, not the code:

- `mean` at 64³ really does lose **1.85×** to running on 96 workers instead of 8,
  reproducibly, on the published path. That is a live optimization awaiting a rule
  keyed on *work per element* rather than element count.
- `otsu_threshold` likewise, 1.12×.
- **`gradient_magnitude` at 64³ is not pool-width-bound.** Four candidates were
  priced and all four are now refuted: pool wake-up, a fixed allocation, window
  setup, and a `t1`/`tN` crossover. (This bullet used to end *"its 2.81× against ITK
  is unexplained"*. That ratio is retracted with the 64³ table — but the refutations
  are **paired, same-harness, port-against-port** comparisons, so the ramp defect
  cancels in them and all four candidates stay dead. What is withdrawn is the *size*
  of the gap it was failing to explain, not the fact that these four do not explain
  it.)

## 5.2 The empty board had one square left on it, and it was the grain

The refutation above ended with `gradient_magnitude` at 64³ *unexplained* — a 2.81×
loss with four dead candidates. The cause was none of them, and it was not specific
to that op: **the chunk grain was a constant.** `map_grain`/`reduce_grain` handed
rayon a fixed grain (`GRAIN = 4096`, and 65536 for the reduce), so a 64³ volume —
262,144 elements — could raise **four tasks** on a 96-worker pool no matter what the
kernel did. Every op with heavy per-element work was serialized into a quarter of
the machine, and the reason `otsu_threshold` was the worst hit is that its
`bin_index` (`histogram.rs:162`) is a **binary search with `partial_cmp().unwrap()`
per probe** — about 188 ns per element, which put **12.3 ms of otsu's 13.7 ms** in
`bin_counts`. Heavy per-element work through a four-task ceiling is the worst case
this defect has.

The seam is one function, `usize` in and `usize` out, so it *cannot* observe the
thread count — it bounds the pool by shape rather than querying it:

```rust
grain(len, ceiling) = clamp(len.div_ceil(TARGET_TASKS), MIN_GRAIN, ceiling)
```

with `TARGET_TASKS = 256` (dominating any plausible pool width) and `MIN_GRAIN =
2048` **derived** — the largest power of two under 262,144/96 = 2,730, which raises
128 tasks where 4,096 would raise only 64. Applied at all five chunked sites
(`min_max`, `bin_counts`, `fill_indexed`, `fill_zip`, `for_each_mut`).

**Why this one was mergeable where §5.1's was not:** it was proved a no-op at and
above 256³ rather than asserted to be. The emitted chunk boundaries are compared
**as integers** against the old fixed grain in a unit test, and a binding probe over
all twelve ops at all three sizes shows the rule binds at 64³ for all twelve and for
**no** op at 256³ or 512³. Medium and large therefore run a byte-identical
decomposition — which is also why the three cells that looked like large-size
regressions in the r4 sweep could not have been caused by it, and were not (§3, the
noise floor).

**The result — restated 2026-07-15, because the ratios it was written with are
retracted.** It originally read *"`otsu_threshold` 64³ 6.62× ITK → 0.68×, and
`gradient_magnitude` 64³ 2.91× → 1.05×"*. Both halves of both ratios came from the
defective harness (§0), so **no rust/ITK ratio is claimed for this fix.** What the
fix did is still established, by a measurement that never involved ITK: the ABBA twin
of §3 puts `gradient_magnitude` 64³ at **0.34×** and `discrete_gaussian` 64³ at
**0.44×** of the pre-seam tree — same harness on both legs, so the defect cancels,
and the ranges do not overlap. **The seam made the port's own 64³ path 2–3× faster.
Whether that is faster than ITK is now an open question, not a settled one.** Every
checksum unmoved; `bit_parity` 18/18 at 1, 4, 48 and 96 threads.

The ceiling is load-bearing in the direction assumed but not previously measured: at
256³ a flat 2,048 grain *regresses* `otsu_threshold` to 47.0 ms against 38.0 at
65,536, because the sequential combine is O(chunks). The `clamp`'s upper bound is
not tidiness.

## 6. What is still open

- **Retake the 64³ table.** This is now the top open item, ahead of every
  optimization below it, because until it is done the port has **no 64³ number it can
  quote**. Both halves are owed: the rust column under `bench/run_protocol.py` (one
  op per process, quiet gate, median of ≥6 launches, refuses cells whose spread
  exceeds the 1.13× floor), and the ITK column under a C++ harness whose warm-up
  covers the box's ~2.1 s ramp — `bench/cpp/main.cxx:466` currently warms up with a
  **single call**, so ITK's 64³ column carries the same defect, harder.
- **`mean` at 64³ — the *cause* is still open; the *ratio* is retracted.** This bullet
  used to read "the last 64³ loss, 2.82× ITK". That number is gone with the table, and
  under the protocol `mean` 64³/`tN` is 1.73–1.75 ms against the 14.4 ms it was built
  on — so whether `mean` loses to ITK at 64³ **is not currently known**. What is still
  measured, and never involved ITK, is the pool-width effect: the same 125-tap window
  costs **1.85× more** on 96 workers than on 8, paired on the published path. That is
  **window locality**, not task count, and it is a live optimization. No rule has been
  written for it, because the taps are non-monotonic (otsu's ~3 taps *win* on a narrow
  pool, gm's 6 taps *lose*, mean's 125 *win*) and `gradient_magnitude`'s optimal pool
  width **flips with the entry shape** — 8 workers on the direct path, 96 on the
  harness path. No rule keyed on the op's own structure can be right in both.
- **`for_each_line_mut`'s `MIN_BLOCK_TASKS = 32` floor** sits below this box's 96
  workers — the same arithmetic as §5.2's defect, but it decomposes by whole blocks
  rather than element count, so it needs a different rule. `smoothing_recursive_gaussian`
  and `gmrg` spend their time there and moved **0%** under the grain seam. Not fixed.
- **`smoothing_recursive_gaussian` at large (1.75×)**, not investigated.
- **The §2 pipeline table's `setup` / `20 iterations` rows** are re-measured; the
  `cast` / `rescale` / `smooth` rows still carry their original numbers.
- **Device coverage.** Cast (all 10 scalar types), `rescale_intensity`,
  `smooth_gaussian`, `recursive_gaussian`, `shrink`, `resample_linear`,
  `resample_nearest`, a constant fill, two mask kernels behind `DeviceMask`, and a
  mean-squares metric with fixed and moving masks, and a resample that carries a
  **point map** — so a fixed-initial transform works for the nine matrix-offset
  transform classes, bit-identically to `ResampleImageFilter`. Masks, the
  nearest-neighbour resample, the virtual domain and the fixed-initial transform all
  landed; the device level mask is byte-equal to the host's. Still missing: device
  Mattes/correlation/ANTS and sampling strategies. Still refused **by name**:
  `Scale` and `ScaleLogarithmic` (they evaluate `(p−c)·s + c`, a different rounding
  from `M·p + b` — refused, not approximated), `Composite`, `BSpline`,
  `DisplacementField`.
- **Multi-GPU.** Device 0 only. Four are present.
- **`connected_component` at large** is the port's own worst absolute number
  (8.6 s). It beats ITK only because ITK's threaded path is broken.
