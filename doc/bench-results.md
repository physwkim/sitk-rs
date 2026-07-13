# Benchmark results: sitk-rs vs ITK 6.0 (C++)

Measured under the contract in [`bench-spec.md`](bench-spec.md). Raw rows for
**two independent runs** are frozen in [`../bench/results/`](../bench/results/):

```
python3 bench/compare.py bench/results/rust.ndjson      bench/results/cpp.ndjson       # run 1
python3 bench/compare.py bench/results/rust-run2.ndjson bench/results/cpp-run2.ndjson  # run 2
```

Both exit 0: every op received a byte-identical input in both harnesses at all
three sizes. That equality is the precondition of the comparison, and it holds.

Two runs are published, not one, because **this box does not reproduce its own
ITK numbers well enough to quote a ratio to two decimals** — see
[§0](#0-how-much-of-this-can-you-trust) before reading any number below.

## Machine

- 96 logical cores; 4× NVIDIA RTX 5000 Ada (32 GiB, cc 8.9), CUDA 13.0.
- ITK 6.0, release, default Pool threader (not TBB), no FFTW.
- rustc 1.97.0, release, criterion. `--features sitk-filters/cuda`.

`t1` = one thread, `tN` = all 96, `gpu` = the CUDA kernel. `ratio = rust / cpp`,
so **> 1.00× means the port is slower than ITK**. Both runs measure `t1`, `tN`
and `gpu` back-to-back per `(op, size)`, so drift hits all three configs of an
op equally.

## 0. How much of this can you trust?

Run 1 and run 2 are the same code on the same box. Their `rust/cpp` ratios
disagree by **3% to 42%** (medium, `t1`):

| op | ratio run 1 | ratio run 2 | disagreement |
|---|---|---|---|
| gradient_magnitude | 1.49× | 2.12× | **42%** |
| rescale_intensity | 1.36× | 1.86× | **37%** |
| signed_maurer_distance_map | 0.50× | 0.66× | 32% |
| gradient_magnitude_recursive_gaussian | 1.63× | 1.25× | 23% |
| mean | 2.39× | 1.96× | 18% |
| binary_dilate | 1.65× | 1.87× | 14% |
| otsu_threshold | 1.22× | 1.19× | 3% |

The instability is **not symmetric, and that is the useful part**. The port's own
timings reproduce well — medium `t1`: `binary_dilate` 3208.8 / 3204.0,
`otsu_threshold` 964.5 / 935.4, `discrete_gaussian` 1689.6 / 1645.2, all inside
5%, *even though run 2 was measured with 13 GB free and run 1 with 194 GB*. What
moves is **ITK**: `gradient_magnitude` `cpp t1` 472.0 → 318.9 (−32%),
`rescale_intensity` 70.6 → 48.7 (−31%), `binary_dilate` 1948.2 → 1710.5 (−12%).

Likely cause: the C++ harness runs as 72 short process invocations while the
Rust harness runs one long process, so ITK pays cold-start effects the port does
not. It is not established, and it is not the port's numbers that are in doubt.

**Rules for reading this file:**

- A ratio is good to roughly **±20–40%**. Quote it as a range or a direction.
- A difference under ~1.5× is **inside the noise**. Do not call it a win or a loss.
- The GPU-vs-CPU verdict in §2 *is* outside the noise: it reproduces across both
  runs. So does the direction of every large `t1` gap.
- The earlier version of this file compared `t1` and `tN` columns measured in
  *different sessions*, on a box where the identical function once measured
  184 ms and 863 ms. Those comparisons were void. This file does not repeat that:
  every number below is from one of the two runs above, both self-contained.

## 1. Results (run 1; run 2 in `*-run2.ndjson`)

Criterion median, milliseconds.

### medium (256³) — the reference size

| op | rust t1 | cpp t1 | ratio | rust tN | cpp tN | ratio | gpu |
|---|---|---|---|---|---|---|---|
| binary_dilate | 3208.8 | 1948.2 | 1.65× | 533.4 | 1711.1 | 0.31× | — |
| connected_component | 1125.7 | 882.1 | 1.28× | 1068.1 | 4480.9 | 0.24× | — |
| discrete_gaussian | 1689.6 | 1201.2 | 1.41× | 195.3 | 135.3 | 1.44× | — |
| fft_convolution | 12456.2 | 1883.3 | **6.61×** | 544.8 | 553.0 | 0.99× | — |
| gradient_magnitude | 704.8 | 472.0 | 1.49× | 118.0 | 31.3 | 3.77× | — |
| grad_mag_recursive_gaussian | 3993.5 | 2452.7 | 1.63× | 485.2 | 219.0 | 2.22× | — |
| mean | 3984.1 | 1666.4 | 2.39× | 368.8 | 67.5 | 5.47× | — |
| median | 11141.8 | 19897.4 | **0.56×** | 512.1 | 453.7 | 1.13× | — |
| otsu_threshold | 964.5 | 787.6 | 1.22× | 47.4 | 51.8 | 0.92× | — |
| rescale_intensity | 95.8 | 70.6 | 1.36× | **16.7** | 37.6 | **0.44×** | 60.1 |
| signed_maurer_distance_map | 1757.9 | 3504.3 | **0.50×** | 94.9 | 261.4 | 0.36× | — |
| smoothing_recursive_gaussian | 1008.3 | 1250.5 | 0.81× | 66.5 | 71.8 | 0.93× | — |

### large (512³)

The port's `t1` is not measured at 512³: it is serial by definition, costs up to
14 min/op under criterion's sample count, and measures the port against itself.
ITK's `t1` *is* measured for all 12 (only its `median` exceeds the spec's 120 s cap).

| op | cpp t1 | rust tN | cpp tN | ratio | gpu |
|---|---|---|---|---|---|
| binary_dilate | 15677.3 | 2237.5 | 16012.2 | 0.14× | — |
| connected_component | 5674.5 | 7739.7 | 32402.5 | 0.24× | — |
| discrete_gaussian | 6346.5 | 1101.0 | 458.5 | 2.40× | — |
| fft_convolution | 11951.3 | 3813.6 | 3496.7 | 1.09× | — |
| gradient_magnitude | 2490.2 | 479.7 | 106.6 | 4.50× | — |
| grad_mag_recursive_gaussian | 23147.3 | 2309.5 | 1007.4 | 2.29× | — |
| mean | 14033.6 | 1518.5 | 575.8 | 2.64× | — |
| median | (> 120 s) | 2070.1 | 3325.5 | 0.62× | — |
| otsu_threshold | 6343.2 | 258.7 | 204.7 | 1.26× | — |
| rescale_intensity | 416.5 | **113.9** | 271.5 | **0.42×** | 379.7 |
| signed_maurer_distance_map | 36166.5 | 638.7 | 1723.5 | 0.37× | — |
| smoothing_recursive_gaussian | 7568.0 | 427.5 | 203.4 | 2.10× | — |

`small` (64³) is in the NDJSON; at that size everything is dominated by fixed
overheads and no conclusion rests on it.

## 2. The CPU beats the GPU on per-pixel ops — and the earlier "tie" was a slow CPU

`rescale_intensity` is the one op with a CUDA kernel. Its output is bit-identical
to the CPU at every size (`max_abs_err = max_rel_err = 0.0`).

| size | gpu (r1 / r2) | cpu tN (r1 / r2) | gpu / cpu |
|---|---|---|---|
| medium | 60.1 / 59.1 | 16.7 / 16.9 | **3.59× / 3.50×** |
| large | 379.7 / 368.4 | 113.9 / 82.1 | **3.33× / 4.49×** |

**The GPU loses to the 96-core CPU by 3.3–4.5×, reproducibly across both runs.**

This is the same *verdict* an earlier version of this file reached, but the
reasoning it gave was wrong and the earlier numbers were an artifact. The old
file said the GPU "tied" the CPU at medium (72.8 vs 72.7 ms) and lost at large
because **PCIe was the floor**. Both halves were false:

- It was never PCIe. The link does **13.0 GB/s**; the op ran its D2H at
  **1.1 GB/s** because it copied into a freshly allocated `Vec` and the DMA
  faulted in all 131,072 of its pages. Fixing the destination took 512³ from
  538.9 ms to 213.8 ms. See §3.
- The "tie" existed only because **the CPU path was slow**. Once
  `map_pixels` removed the CPU's own materialization defect, CPU `tN` went
  72.7 ms → 16.7 ms and the tie became a 3.6× GPU loss.

The lesson is not "the GPU is bad." It is that **a GPU-vs-CPU number is
meaningless until the CPU baseline is actually using the machine.** A slow
baseline makes an offload look competitive, and a plausible hardware story
(PCIe!) is available to explain the result you already believe.

**The same trap is open one level up:** the GPU registration backend reports an
18.3× whole-run speedup at 256³ — measured against a registration metric that is
**single-threaded** (`rg` finds zero rayon/parallel uses in
`crates/sitk-registration/src`). That number is not yet trustworthy for the same
reason this one was not. Parallelizing the CPU metric and re-measuring is in
progress; the metric's reduction is a **float sum**, which is not associative, so
it cannot simply be handed to rayon without changing the metric's bits — and
because the optimizer is a feedback loop, changed metric bits change the
registration *result*.

## 3. The dominant defect, on both sides of the bus: first-touch page faults

Three symptoms, one cause. Linux zeroes every anonymous page before handing it
over, so a freshly allocated buffer is not free — it costs a page fault per 4 KB
on first touch, and the cost lands on whoever writes it first.

| where | symptom | fix / status |
|---|---|---|
| CPU filters | `rescale_intensity_cpu` materialized ~335 MB of intermediate `f64` per call: **438,829 minor faults**, 210 ms at 256³ | **fixed** — `map_pixels` fuses widen→compute→narrow; 98,403 faults, 86 ms |
| GPU D2H | copied into a fresh `Vec`; DMA faulted 131,072 pages, D2H at 1.1 GB/s | **fixed** — resident destination; 13.0 GB/s, 476 → 41 ms at 512³ |
| GPU host alloc | the returned output `Vec` — **108 ms at 512³, larger than H2D + D2H + kernel combined** | **UNFIXED** — needs a *reused* destination, which `fn(&Image) -> Image` cannot express |
| registration setup | `FixedSamples` build = ~400 MB of fresh pages; **98.9% of the GPU run at 256³** | **UNFIXED** — same family |

Note the shape of the last two: they are not fixable by making the code faster,
only by letting a buffer **outlive the call**. The one-shot API is the structural
cause. `map_pixels` and `par_map_window` are now the *only* two places the port
allocates an output volume, so a `map_pixels_into(&mut Image)` variant would
close it at two call sites rather than 598.

Two things measurement contradicted, recorded so they are not retried:
**pinning the H2D source is a net loss** for a one-shot op (99 vs 56 ms at 512³ —
staging into pinned memory costs a full host-to-host copy), and **pinning the D2H
destination is worse than nothing** (817 ms — the pinned→fresh-`Vec` copy faults
every page anyway). Pinning pays only for a **reused** buffer.

## 4. Where the port still loses to ITK, and why

After `map_pixels` / `WindowView` (which deleted the buffer-materialization
constant), the remaining `t1` gaps are real algorithmic or codegen gaps, not
allocation noise:

- **`fft_convolution`, 6.6–7.3×** — the single largest gap, and the cause is
  known and *not* allocation: the port takes three full **complex** 264³ DFTs of
  real-valued input (89% of the op's runtime, measured), where ITK uses a
  half-Hermitian **R2C** transform — half the arithmetic, half the memory.
  Recorded as ledger §4.116. Closing it means implementing R2C/C2R; scoped as its
  own task, deliberately not attempted.
- **`mean` (2.0–2.4×), `gradient_magnitude` (1.5–2.1×)** — the stencil ops that
  gained the least from `WindowView`. Next candidates, but note both are inside
  or near the noise band at `t1`; the `tN` gap (5.5× and 3.8×) is the real one.
- **`median` (0.5×), `signed_maurer_distance_map` (0.5×)** — the port wins, on
  algorithm rather than constant factor.

## 5. Two ITK multithreading regressions, found by this benchmark

ITK is *slower* at 96 threads than at 1 on two ops — defects in ITK, not the port
(ledger §7):

| op (medium) | cpp t1 | cpp tN | ITK speedup |
|---|---|---|---|
| binary_dilate | 1948.2 | 1711.1 | 1.14× (barely any) |
| connected_component | 882.1 | 4480.9 | **0.20× — 5× worse with 96 cores** |

The port's large `tN` wins on these two must be read with that in mind: against
ITK's *own best* time, not its 96-thread time, `connected_component` at 1068 ms
is **slower** than ITK's 882 ms `t1`, not 4× faster as the `tN` ratio suggests.
