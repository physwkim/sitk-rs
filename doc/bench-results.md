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

- **`/proc/loadavg` is unusable as a gate on this box.** It reads 18–21 with four
  runnable tasks and no benchmark alive. Every number here is gated on real busy
  cores from `/proc/stat`, sampled every 5 s, with the benchmark's *own* load
  subtracted so a `tN` leg is not mistaken for foreign contention. Hours were lost
  to the naive gate before this was understood; the traces are in `bench/results/`.
- **The `tN` numbers are the ones to quote.** Both rounds agree tightly there
  (`mean` stddev 0.6 and 1.4 ms). Two `t1` rows — `mean` and `gmrg` — carry 15–21%
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

`rescale_intensity` — the only op with a device kernel — on the same tree and the
same quiet box as every CPU row above:

| size | CPU tN | GPU one-shot | **GPU resident** | ITK tN |
|---|---|---|---|---|
| medium (256³) | 16.9 | 35.8 | **1.06** | 39.7 |
| large (512³) | 108.1 | 205.5 | **4.5** | 266.1 |

Both facts live in the same row:

- **The one-shot API really does lose to the CPU** — 35.8 against 16.9 at medium.
- **The device really does win**, by **16×** over the port's own CPU and **37×**
  over ITK. At large: **24×** over CPU `tN`, **59×** over ITK.

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

Only `rescale_intensity` has a device kernel today. `discrete_gaussian` carries a
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

| op | rust tN | itk tN | **rust/itk (tN)** | rust t1 | itk t1 |
|---|---|---|---|---|---|
| binary_dilate | 65.6 / 67.1 | 2484 / 2480 | **0.03×** | 2390 | 1708 |
| connected_component | 1126 / 1128 | 4352 / 4483 | **0.25×** | 1190 | 679 |
| signed_maurer_distance_map | 87.9 / 88.7 | 244 / 295 | **0.30×** | 1601 | 3479 |
| median | 203 / 206 | 544 / 540 | **0.38×** | 8550 | 19497 |
| rescale_intensity | 16.0 / 16.9 | 38.0 / 39.7 | **0.42×** | 90.1 | 69.8 |
| otsu_threshold | 46.5 / 36.7 | 56.8 / 64.1 | 0.57× | 989 | 780 |
| discrete_gaussian | 114.0 / 131.8 | 162 / 174 | 0.76× | 1329 | 824 |
| **mean** | **62.5 / 62.7** | 80.8 / 78.7 | **0.80×** | 1967 ᵗ | 1662 |
| fft_convolution | 471 / 501 | 587 / 574 | 0.87× | 2148 | 1228 |
| smoothing_recursive_gaussian | 64.6 / 67.2 | 52.3 / 66.0 | 1.02× | 1005 | 818 |
| **gmrg** | 288 / 301 | 207 / 247 | **1.22×** | 2319 ᵗ | 2426 |
| **gradient_magnitude** | 51.7 / 53.0 | 36.2 / 35.1 | **1.51×** | 511 | 314 |

ᵗ — this `t1` carries 15–21% sample stddev in both rounds, including the provably
clean one. Soft; quote the `tN` figure.

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
| **smoothing_recursive_gaussian** | 386 / 380 | 212 / 218 | **1.75×** |
| **gradient_magnitude** | 201 / 198 | 97.9 / 98.8 | **2.00×** |
| **gmrg** | 2181 / 2232 | 772 / 761 | **2.93×** |

### small (64³) — where per-call overhead shows, and the medium table hides it

Two rows are far worse here than at any other size, and neither is visible in the
headline: **`otsu_threshold` 6.62×** (15.4 ms against ITK's 2.3) and **`mean`
4.52×** (23.2 against 5.1). `mean` *beats* ITK at medium and large and loses by
4.5× at small — a fixed per-call cost that larger volumes amortize away. If your
volumes are small, read this row, not the headline.

## 4. Where the port lost to ITK, and what the cause turned out to be

Four ops in the **separable / stencil** family used to lose. Two of them no longer
do — and the numbers below are what a reader should check the story against.

| op, medium/tN | before | **after** |
|---|---|---|
| `mean` | 4.39× | **0.80×** — now *beats* ITK |
| `gradient_magnitude` | 4.25× | **1.51×** — still loses |
| `gmrg` | 1.99× | **1.22×** — still loses |
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

### What still loses, stated plainly

- **`gradient_magnitude`: 1.51× at medium, 2.00× at large.** The allocator fix took
  it from 4.25×; it is still slower than ITK at every size, and worse as the volume
  grows.
- **`gmrg`: 1.22× at medium, 2.93× at large.** The buffer fix answered medium and
  **did not answer large at all** — 2231 ms against ITK's 761.
- **`smoothing_recursive_gaussian`: 1.75× at large** (1.02× at medium).
- **`otsu_threshold`: 1.37× at large, 6.62× at small.**
- **`mean`: 4.52× at small**, which its medium crossover hides completely.

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

## 6. What is still open

- **`gradient_magnitude` (2.00× at large) and `gmrg` (2.93× at large).** The two
  fixes in §4 did not close these; `gmrg`'s large case in particular is untouched by
  the buffer fix that answered its medium case.
- **`otsu_threshold` at small (6.62×) and `mean` at small (4.52×)** — per-call
  overhead that the reference size amortizes away.
- **The §2 pipeline table's `setup` / `20 iterations` rows** are re-measured; the
  `cast` / `rescale` / `smooth` rows still carry their original numbers.
- **Device coverage.** Cast (all 10 scalar types), `rescale_intensity`,
  `smooth_gaussian`, `recursive_gaussian`, `shrink`, `resample_linear`, and a
  mean-squares metric. Still missing: device Mattes/correlation/ANTS, masks,
  sampling strategies, and a nearest-neighbour resample — the last of which is what
  a device mask needs, and a device mask is what closes two of the boundary's
  refusals.
- **Multi-GPU.** Device 0 only. Four are present.
- **`connected_component` at large** is the port's own worst absolute number
  (8.6 s). It beats ITK only because ITK's threaded path is broken.
