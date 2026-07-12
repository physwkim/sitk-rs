# Benchmark results: sitk-rs vs ITK 6.0 (C++)

> **STALE — re-measurement in progress (2026-07-13).** The tables below were
> measured *before* `sitk_core::fused::map_pixels` / `WindowView` landed, and
> before `sitk-cuda`'s D2H page-fault fix. Five of the twelve `t1` numbers are
> known to be wrong now (`rescale_intensity` 3.40× faster, `gradient_magnitude`
> 1.79×, `discrete_gaussian` 1.71×, `binary_dilate` 1.66×, `mean` 1.20×), and
> the `rescale_intensity` GPU row is wrong at every size. The ITK columns are
> unaffected — ITK did not change.
>
> Beyond that, §2c below explains why the `t1` and `tN` columns should not be
> compared to each other *at all* right now: they were measured in different
> sessions on a box whose first-touch allocation cost swings ~4.7× with memory
> pressure. Do not quote a ratio from this file until it is regenerated.

Measured under the contract in [`bench-spec.md`](bench-spec.md). Raw rows are
frozen in [`../bench/results/`](../bench/results/); regenerate the tables with

```
python3 bench/compare.py bench/results/rust.ndjson bench/results/cpp.ndjson
```

`compare.py` exits non-zero and voids any op whose `input_checksum` differs
between the two harnesses. **It exits 0 on this data**: all 12 ops received a
byte-identical input in both harnesses at all three sizes, which is what makes
the numbers comparable at all.

## Machine

- 96 logical cores; 4× NVIDIA RTX 5000 Ada (32 GiB, cc 8.9), CUDA 13.0.
- ITK 6.0, release build, default threader (Pool, not TBB), no FFTW.
- rustc 1.97.0, release profile, criterion.

`t1` = one thread, `tN` = all 96, `gpu` = the CUDA kernel. `ratio = rust / cpp`,
so **> 1.00× means the port is slower than ITK**.

## Results

Times are the criterion median, in milliseconds.

### small (64³)

| op | rust t1 | cpp t1 | ratio | rust tN | cpp tN | ratio | gpu |
|---|---|---|---|---|---|---|---|
| binary_dilate | 275.2 | 36.9 | 7.46× | 57.6 | 37.8 | 1.52× | — |
| connected_component | 9.9 | 13.8 | 0.72× | 11.3 | 138.4 | 0.08× | — |
| discrete_gaussian | 73.8 | 26.7 | 2.77× | 22.5 | 7.5 | 3.01× | — |
| fft_convolution | 235.6 | 24.5 | 9.63× | 25.2 | 27.2 | 0.92× | — |
| gradient_magnitude | 24.2 | 8.1 | 2.99× | 14.1 | 2.6 | 5.41× | — |
| grad_mag_recursive_gaussian | 42.4 | 42.2 | 1.00× | 19.4 | 13.1 | 1.47× | — |
| mean | 129.1 | 38.4 | 3.36× | 45.1 | 5.4 | 8.36× | — |
| median | 271.8 | 433.1 | 0.63× | 50.1 | 32.2 | 1.56× | — |
| otsu_threshold | 14.0 | 17.5 | 0.80× | 15.2 | 2.8 | 5.44× | — |
| rescale_intensity | 0.8 | 0.7 | 1.10× | 0.6 | 1.0 | 0.60× | 0.5 |
| signed_maurer_distance_map | 31.9 | 57.5 | 0.55× | 5.8 | 33.6 | 0.17× | — |
| smoothing_recursive_gaussian | 11.9 | 13.5 | 0.89× | 4.5 | 2.4 | 1.93× | — |

### medium (256³)

| op | rust t1 | cpp t1 | ratio | rust tN | cpp tN | ratio | gpu |
|---|---|---|---|---|---|---|---|
| binary_dilate | 5781.1 | 1702.9 | 3.39× | 544.6 | 2549.7 | 0.21× | — |
| connected_component | 997.2 | 684.2 | 1.46× | 959.8 | 4564.5 | 0.21× | — |
| discrete_gaussian | 2906.5 | 832.5 | 3.49× | 243.8 | 149.9 | 1.63× | — |
| fft_convolution | 14519.7 | 1218.9 | 11.91× | 528.5 | 570.9 | 0.93× | — |
| gradient_magnitude | 1190.1 | 444.8 | 2.68× | 141.5 | 37.1 | 3.82× | — |
| grad_mag_recursive_gaussian | 2955.5 | 2941.0 | 1.00× | 361.6 | 219.5 | 1.65× | — |
| mean | 3004.0 | 2323.1 | 1.29× | 406.2 | 81.7 | 4.97× | — |
| median | 9263.7 | 19644.0 | 0.47× | 457.0 | 552.2 | 0.83× | — |
| otsu_threshold | 967.5 | 780.9 | 1.24× | 47.4 | 56.4 | 0.84× | — |
| rescale_intensity | 250.2 | 71.3 | 3.51× | 72.7 | 39.4 | 1.84× | 72.8 |
| signed_maurer_distance_map | 2406.0 | 3553.2 | 0.68× | 94.0 | 232.6 | 0.40× | — |
| smoothing_recursive_gaussian | 1265.8 | 1154.9 | 1.10× | 65.8 | 69.4 | 0.95× | — |

### large (512³)

The port's `t1` column is not measured at this size. A serial 512³ pass costs
14 min for the slowest op under criterion's sample count, and `t1` measures the
port against itself, not against ITK — so the budget went to `tN`, which is the
column the comparison turns on. ITK's `t1` is measured for all 12 (only its
`median` exceeds the spec's 120 s cap).

| op | cpp t1 | rust tN | cpp tN | ratio | gpu |
|---|---|---|---|---|---|
| binary_dilate | 15033.8 | 3305.5 | 17726.5 | 0.19× | — |
| connected_component | 5483.5 | 7429.7 | 32008.6 | 0.23× | — |
| discrete_gaussian | 6059.7 | 1432.3 | 465.5 | 3.08× | — |
| fft_convolution | 9666.0 | 3843.3 | 3123.5 | 1.23× | — |
| gradient_magnitude | 2401.6 | 540.4 | 87.9 | 6.14× | — |
| grad_mag_recursive_gaussian | 22426.2 | 1787.2 | 788.3 | 2.27× | — |
| mean | 14062.1 | 1600.9 | 506.0 | 3.16× | — |
| median | (> 120 s) | 2188.7 | 3600.0 | 0.61× | — |
| otsu_threshold | 6261.6 | 315.5 | 189.8 | 1.66× | — |
| rescale_intensity | 588.4 | 243.0 | 261.7 | 0.93× | 538.9 |
| signed_maurer_distance_map | 33618.2 | 536.8 | 2319.8 | 0.23× | — |
| smoothing_recursive_gaussian | 7501.6 | 284.2 | 217.0 | 1.31× | — |

## What the numbers say

### 1. The port's weakness is the single-thread constant factor, not parallelism

At `t1`, medium, the port is 2.7–11.9× slower than ITK on
`fft_convolution`, `discrete_gaussian`, `binary_dilate`, `gradient_magnitude`,
`mean` and `rescale_intensity`. rayon hides this behind 96 cores — but it only
hides it. The ops the port still *loses* at `tN` are exactly the ops with the
worst `t1` constant: `gradient_magnitude` (3.82× at tN, 2.68× at t1), `mean`
(4.97× / 1.29×), `discrete_gaussian` (1.63× / 3.49×).

So the next optimization target is the scalar inner loop — SIMD, memory access
patterns, removing needless `f64` widening — **not** more parallelism. Adding
threads to a 3× slower kernel buys a 3× slower kernel on more cores.

Where the port already wins at `t1` it wins on algorithm, not on constant
factor: `median` (0.47×), `signed_maurer_distance_map` (0.68×),
`connected_component` (0.72× small).

### 2. The GPU was never PCIe-bound — it was page-faulting. (Conclusion retracted 2026-07-13)

**This section previously read "GPU offload of per-pixel ops is not worth it —
PCIe dominates," and concluded that a per-pixel op can never win on the GPU
because the bus is the floor. That conclusion was wrong and is retracted.** It
is preserved here as a worked example of how a plausible scaling argument can
launder an allocation cost into a hardware limit.

The reasoning was: GPU time scales exactly linearly with voxel count
(72.8 ms at 256³ → 538.9 ms at 512³, i.e. 8× the data for 7.4× the time) while
the kernel is only ~1.3 ms, therefore the time is the PCIe round-trip,
therefore it is irreducible. Every step of that is true except the last, and
the arithmetic that breaks it is one division: 512³ moves 1.074 GB round-trip
in 538.9 ms, which is **2.0 GB/s** — about a *tenth* of what this machine's
link actually does.

Measured on this machine: **the link does 13.0 GB/s. The op was running its D2H
at 1.1 GB/s.** The gap was not the bus. `rescale_intensity_gpu` copied the
result into a **freshly allocated `Vec`**, so the DMA faulted in all 131,072 of
that buffer's pages as it wrote them, and the kernel's page-zeroing cost was
being billed to the transfer. The crate's own `buffer.rs` had already recorded
the effect ("the page-fault cost is the allocation's, not the PCIe link's") and
the op was calling the slow path anyway.

Copying into a resident destination instead: **512³ D2H 476 ms → 41 ms
(1.1 → 13.0 GB/s), and the whole op 538.9 ms → 213.8 ms**, with the output
still bit-identical to the CPU at every size (`max_abs_err = max_rel_err = 0.0`,
0 of 16,777,216 voxels differ at 256³).

Phase split after the fix (median of 3 warm runs, ms):

| | 64³ | 256³ | 512³ |
|---|---|---|---|
| h2d | 0.17 | 6.93 | 56.80 |
| **alloc** (host output) | 0.07 | 23.31 | **108.12** |
| kernel | 0.10 | 1.33 | 6.09 |
| d2h | 0.21 | 5.45 | 41.40 |
| **total** | **0.54** | **37.02** | **213.8** |

**The real wall is now the output allocation** — at 512³ it is 108 ms, larger
than H2D, D2H and the kernel *combined*. It is inherent to returning a freshly
owned buffer: Linux must zero every anonymous page before handing it over. Only
a **reused** destination removes it, which a one-shot `fn(&Image) -> Image` API
cannot express.

Two things measurement contradicted, recorded so they are not retried:

- **Pinning the H2D source is a net loss** for a one-shot op (99 ms vs 56 ms at
  512³): staging the caller's `Vec` into pinned memory costs a full host-to-host
  copy, more than the faster DMA saves.
- **Pinning the D2H destination is worse than doing nothing** (817 ms): the
  pinned→fresh-`Vec` copy faults every page anyway.

Pinning pays only for a **reused** buffer. That is the resident-volume shape
(registration's inner loop), not the one-shot filter shape.

This does *not* reinstate "expand the GPU to all 12 ops". It says the reason to
be selective is arithmetic intensity, not a bus floor — and it says the highest
-value GPU work is wherever a buffer can stay resident across many passes.

### 2b. The same defect, on the other side of the bus

The CPU path is sick with the identical disease. `rescale_intensity_cpu`
materializes ~335 MB of intermediate `f64` buffers per call (widen → min/max →
map → narrow), and the cost is not `malloc` — `vec![0f64; n]` on its own
measures 0.0 ms, because the pages are lazy. The cost is the kernel **zeroing
every page on first touch**: 438,829 minor faults for the staged shape versus
98,403 for a fused one, 210 ms versus 86 ms at 256³ (ITK: 71 ms).

So the port's dominant performance defect, on **both** sides of the bus, is
**first-touch page faults on freshly-allocated intermediate buffers**. Two
symptoms, one cause.

### 2c. Measurement stability — read before trusting any number above

The `t1`/`tN` columns in the tables above were measured across **different
sessions**, and that is not currently safe on this machine. The identical CPU
function measured **184 ms in one session and 863 ms in another — a 4.7× drift
with no code change**, because the box runs with ~10 GB free against ~473 GB of
page cache, so a large anonymous allocation must reclaim page cache rather than
take free pages, and the first-touch cost above swings with it. (Long-running
agent processes on the box contribute to that pressure.)

Consequence: **a GPU-vs-CPU verdict requires both paths measured in one session,
on one branch, back to back.** The 213.8 ms GPU figure is *not* compared against
the 243.0 ms CPU `tN` above, because those two numbers come from different
machine states. Do not restate that comparison until both are re-run together.

### 3. Two ITK multithreading regressions, found by this benchmark

ITK is *slower* at 96 threads than at 1 on two ops. These are defects in ITK,
not in the port, and the port's large `tN` win on them should be read with that
in mind:

| op (medium) | cpp t1 | cpp tN | ITK speedup |
|---|---|---|---|
| binary_dilate | 1702.9 | 2549.7 | **0.67×** |
| connected_component | 684.2 | 4564.5 | **0.15×** |

`connected_component` gets 6.7× *worse* when ITK is given 96 cores.
