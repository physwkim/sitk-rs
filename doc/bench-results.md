# Benchmark results: sitk-rs vs ITK 6.0 (C++)

Measured under the contract in [`bench-spec.md`](bench-spec.md). Raw rows for
**two independent runs** are frozen in [`../bench/results/`](../bench/results/),
together with [`sweep-load-trace.txt`](../bench/results/sweep-load-trace.txt),
which stamps `/proc/loadavg` at every stage boundary:

```
python3 bench/compare.py bench/results/rust.ndjson      bench/results/cpp.ndjson       # run 1
python3 bench/compare.py bench/results/rust-run2.ndjson bench/results/cpp-run2.ndjson  # run 2
```

Both exit 0: every op received a byte-identical input in both harnesses at all
three sizes. That equality is the precondition of the comparison, and it holds.

All four files were written in **one machine state**, in one sitting: C++ `t1`,
C++ `tN`, Rust, twice through. No number here is reused from an earlier session.

## Machine

- 96 logical cores — **48 physical**, 2 sockets × 24, SMT on. This matters: a
  "96×" ceiling does not exist. The all-core clock is 2.9 GHz against 3.6 GHz
  for a lone core, so the honest ceiling against a single-threaded baseline is
  **~38×**, before any software cost.
- 4× NVIDIA RTX 5000 Ada (32 GiB, cc 8.9), CUDA 13.0. Device 0 only.
- ITK 6.0, release, default Pool threader (not TBB), no FFTW.
- rustc 1.97.0, release, criterion. `--features sitk-filters/cuda`.

`t1` = one thread. `tN` = all 96. `gpu` = the one-shot API (`fn(&Image) -> Image`,
so H2D + kernel + D2H are all inside the timed region). `gpu_resident` = the same
kernel with the image already on the device and the output staying there — the bus
is outside the timed region because in a real pipeline it is outside the call.

All figures are `ms_median`. **Ratios below are `rust / itk`, so > 1.00 means the
port is slower.**

## 0. How much of this can you trust?

Read this before quoting any number.

- **Ratios are good to about ±20–40%. A difference under ~1.5× is noise.** Two
  independent runs of the identical binaries disagree by that much.
- **The instability is asymmetric.** The port's own timings reproduce inside 5%
  across the two runs (`binary_dilate` 516.1 / 494.4, `otsu` 48.5 / 47.6,
  `discrete_gaussian` 192.8 / 184.2). ITK's move more (`rescale_intensity`
  `itk t1` 47.4 → 72.4, +53%; `binary_dilate` `itk t1` 1721.6 → 2464.0, +43%).
  The likely cause is structural: the C++ harness runs as 72 short process
  invocations while the Rust harness is one long process.
- **The box is shared.** `sweep-load-trace.txt` records what it was: the gate
  opened at load 1.33, and the `tN` stages then drove it to 20–66 *themselves*.
  A `tN` row is a 96-thread pool on a 48-core machine; its wall is set by its
  slowest worker, so `tN` rows are the ones that suffer under any foreign load.
  `t1` and `gpu_resident` are comparatively immune.
- `median` at `large` has no `itk t1` row: ITK's single-threaded median did not
  finish inside the harness budget. It is absent, not zero.

## 1. The retraction: I measured the bus and concluded about the GPU

An earlier version of this document claimed the GPU cannot win a per-pixel op —
that PCIe was a hard floor and offload was not worth it. **That was wrong, and the
row that refutes it is a row I had already collected.**

`rescale_intensity`, one machine state, both runs:

| size | CPU t1 | CPU tN | GPU one-shot | **GPU resident** | ITK tN |
|---|---|---|---|---|---|
| medium (256³) | 92.0 / 89.4 | 17.4 / 16.4 | 36.1 / 30.5 | **1.04 / 1.06** | 38.3 / 38.2 |
| large (512³) | — | 77.1 / 102.8 | 206.6 / 192.0 | **4.51 / 4.52** | 257.4 / 261.1 |

Both facts live in the same row:

- **The one-shot API really does lose**, by ~2×, at medium (36.1 vs 17.4).
- **The device really does win**, by **16×** (1.04 vs 17.4) — and by **37×**
  against ITK. At large it is **17–23×** over CPU `tN` and **57×** over ITK.

The only difference between those two GPU columns is whether the bus is inside
the timed region. At 256³ an `f32` volume is 67 MB, so a round trip is ~17 ms of
transfer to do ~1 ms of work. I measured the round trip, and published a verdict
about the hardware.

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

| | host (CPU tN) | device-resident |
|---|---|---|
| cast | 80.6 | 42.0 → 22.9 (now on device) |
| rescale | 24.1 | 6.1 |
| smooth (both volumes) | **2250.7** | **9.7** |
| registration setup | 22.4 | 6.8 |
| 20 iterations | 15469.3 | 138.0 |
| **total** | **17,880 ms** | **240.5 ms** — **74×** |

A real `ImageRegistrationMethod::execute()` at 256³ — not an evaluate loop —
is **16.2–17.7 s on the host against 148.6–151.0 ms on the device, 107–119×**.

> **Both of those numbers are single-level, and both predate a metric-kernel fix.**
> The device kernel was forming the continuous index with FMA contraction and with
> the transform offset seeded into the accumulator (the host adds it last). A 1-ULP
> index difference flips `floor()` for a sample lying exactly on a voxel plane,
> where the trilinear gradient is discontinuous — so the kernel took the *opposite
> one-sided derivative*: `d/d(angle_y)` off by 34% while the value agreed to 1e-15.
> Fixed at source; 3.2e-14 after. **The timings above have not been re-taken on the
> fixed kernel**, and no pyramid run has been timed at all.
>
> **The blast radius is one kernel, `mean_squares.cu`.** Of the six rows in the
> table, only `registration setup` and `20 iterations` sit downstream of it; `cast`,
> `rescale`, and `smooth` are untouched code and still describe what runs today, as
> does every row of §1 and §3.
>
> **But the 107–119× row can move for a reason that is not kernel speed.** It is a
> real `execute()`, so the optimizer picks its own iteration count; a corrected
> derivative changes the descent trajectory, and the wall clock can move in either
> direction with the per-iteration cost unchanged. The `20 iterations` row is immune
> (its count is pinned); that one is not. `execute_on_device` also now accepts a
> pyramid (bit-identical level images, identical iteration counts on both paths).
>
> These rows are left in place and marked, not deleted and not quietly re-labelled,
> until the re-measurement lands.

The Gaussian is where the CPU bleeds: 2,250.7 ms on 96 threads against 9.7 ms on
the device. The device Gaussian is **bit-identical** to the CPU filter, not close
to it — `f64` weights and intermediates, and `__dmul_rn`/`__dadd_rn` to forbid FMA
contraction, because an FMA would be *more* accurate and therefore *different*.

## 3. Results — medium (256³), the reference size

| op | rust t1 | rust tN | itk t1 | itk tN | rust/itk (tN) |
|---|---|---|---|---|---|
| rescale_intensity | 92.0 / 89.4 | 17.4 / 16.4 | 47.4 / 72.4 | 38.3 / 38.2 | **0.45×** |
| binary_dilate | 3048 / 2995 | 516 / 494 | 1722 / 2464 | 2528 / 2532 | **0.20×** |
| connected_component | 1144 / 1150 | 1085 / 1137 | 686 / 682 | 4837 / 4579 | **0.22×** |
| signed_maurer_distance_map | 2116 / 2390 | 113 / 93 | 3512 / 5134 | 290 / 237 | **0.39×** |
| otsu_threshold | 985 / 987 | 48.5 / 47.6 | 781 / 784 | 54.3 / 58.8 | 0.89× |
| median | 11312 / 9092 | 541 / 482 | 21170 / 22845 | 559 / 542 | 0.97× |
| fft_convolution | 2473 / 2709 | 457 / 482 | 1245 / 1274 | 431 / 518 | 1.06× |
| smoothing_recursive_gaussian | 996 / 1159 | 68.8 / 66.4 | 825 / 823 | 63.9 / 64.3 | 1.08× |
| discrete_gaussian | 1560 / 1556 | 193 / 184 | 828 / 843 | 154 / 160 | 1.25× |
| gradient_magnitude_recursive_gaussian | 2639 / 2680 | 379 / 355 | 2440 / 2979 | 202 / 216 | 1.88× |
| gradient_magnitude | 632 / 636 | 121 / 116 | 325 / 321 | 28.1 / 32.7 | **4.30×** |
| mean | 2459 / 2456 | 356 / 326 | 1685 / 1667 | 81.0 / 78.4 | **4.39×** |

### large (512³)

| op | rust tN | itk tN | rust/itk |
|---|---|---|---|
| binary_dilate | 2104 / 2258 | 16538 / 15435 | **0.13×** |
| connected_component | 9476 / 8437 | 32549 / 32263 | **0.29×** |
| rescale_intensity | 77.1 / 102.8 | 257 / 261 | **0.30×** |
| signed_maurer_distance_map | 593 / 655 | 1492 / 1516 | **0.40×** |
| median | 2511 / 2157 | 3543 / 3623 | 0.71× |
| fft_convolution | 3210 / 2366 | 3701 / 3252 | 0.87× |
| otsu_threshold | 267 / 255 | 198 / 193 | 1.35× |
| discrete_gaussian | 1039 / 1006 | 561 / 671 | 1.85× |
| smoothing_recursive_gaussian | 404 / 403 | 203 / 225 | 1.99× |
| mean | 1105 / 1070 | 471 / 495 | **2.35×** |
| gradient_magnitude_recursive_gaussian | 2091 / 1940 | 784 / 814 | **2.67×** |
| gradient_magnitude | 340 / 380 | 106 / 99.2 | **3.22×** |

## 4. Where the port loses to ITK — and what the cause turned out to be

Four ops, all in the **separable / stencil** family:

- `mean` — 4.4× at medium, 2.4× at large
- `gradient_magnitude` — 4.3× / 3.2×
- `gradient_magnitude_recursive_gaussian` — 1.9× / 2.7×
- `discrete_gaussian` — 1.25× / 1.85×

The shape said what it was: the port's `t1` is competitive-to-better on several of
these (`gradient_magnitude_recursive_gaussian` t1 2639 vs ITK 2440) and it is the
`tN` column that falls behind. A **scaling** gap, not a constant-factor one.

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

`mean` at t48: **338.6 → 185.1 ms (1.83×)**; parallel efficiency at t16 0.66 → 0.98.

**The table above still shows the pre-fix numbers.** It was measured before the fix
landed, and a clean sweep on a quiet box is owed before it can be updated. Two
things remain open even after it:

- **A second limiter on `mean`.** It now scales near-perfectly to t16 (efficiency
  0.98) and then stops dead — 196 ms at t16, 190 at t24, 185 at t48, with only ~20
  busy cores. Chunk granularity is ruled out (sweeping `GRAIN` 4096/1024/256 moves
  it under 7%). The allocator was the dominant cause, not the only one.
- **`gradient_magnitude_recursive_gaussian` is a different defect.** It makes 44,429
  allocations (0.00/voxel) and never touches the boundary path. Its 1.88× loss is
  most likely the per-axis full-volume `to_f64_vec()` copies. Uninvestigated — and
  not folded into the finding above just because it was on the same list.

`fft_convolution` **is closed**: it was 6.5× slower than ITK at `t1` before
rustfft/realfft landed and the real-input half-Hermitian path replaced three full
complex transforms of real data. It is now 1.06× at medium and 0.87× at large.

## 5. Two ITK multithreading regressions, found by this benchmark

ITK gets *slower* with threads on two ops. These are upstream defects, recorded as
ledger §7.1 and §7.2 — they are performance defects, not correctness ones, which is
why they are not in §1 of the ledger.

- **`connected_component`**: `itk t1` 686 ms → `itk tN` **4837 ms** at medium. Seven
  times slower on 96 threads than on one. At large: 5487 → 32,263.
- **`binary_dilate`**: `itk t1` 1722 ms → `itk tN` **2528 ms** at medium.

The port's win on these two ops (`0.20×`, `0.22×`) is therefore **against ITK's
threaded time, which is worse than its own single-threaded time**. Measured against
ITK's *own best* number, `binary_dilate`'s 4.9× win shrinks to 3.3×, and
`connected_component`'s 4.5× win becomes a **1.6× loss**. Stated here so the
headline ratio is not read as more than it is.

## 6. What is still open

- **The CPU scaling ceiling** (§4). 11–15× on 48 physical cores, cause unknown.
  Ruled out by measurement: bandwidth, false sharing, the allocator, NUMA
  placement, and the cost of bit-exactness. This is the largest unexplained number
  in the project.
- **The device timings are owed a re-measurement.** See the box in §2: the metric
  kernel changed under them, and the pyramid path they said was refused now exists.
- **Device coverage.** Cast (all 10 scalar types), `rescale_intensity`,
  `smooth_gaussian`, `recursive_gaussian`, `shrink`, `resample_linear`, and a
  mean-squares metric. Still missing: device Mattes/correlation/ANTS, masks, and
  sampling strategies.
- **Multi-GPU.** Device 0 only. Four are present.
- **`connected_component` at large** is the port's own worst absolute number
  (9.5 s). It beats ITK only because ITK's threaded path is broken.
