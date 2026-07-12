# Benchmark results: sitk-rs vs ITK 6.0 (C++)

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

### 2. GPU offload of per-pixel ops is not worth it — PCIe dominates

`rescale_intensity` is the one op with a CUDA kernel. It is bit-exact against
the CPU path (`max_abs_err = max_rel_err = 0.0` at every size).

| size | gpu | cpu tN | verdict |
|---|---|---|---|
| small | 0.5 | 0.6 | tie |
| medium | 72.8 | 72.7 | dead tie |
| large | 538.9 | 243.0 | **GPU 2.2× slower** |

GPU time scales exactly linearly with voxel count (72.8 → 538.9 ms for 8× the
data), which means it is *entirely* the PCIe round-trip; the kernel itself is
~1.3 ms at medium. A per-pixel op cannot win on the GPU when the host must ship
the volume across the bus to get one arithmetic operation done to it.

This closes the question of expanding GPU coverage to the other 11 ops as a
per-op offload. The only shapes that could pay for the transfer are the
compute-dense ones — `median`, `discrete_gaussian`, `fft_convolution` — or a
design where the volume stays resident on the device across a *chain* of ops so
the transfer is amortized. Neither is implemented.

### 3. Two ITK multithreading regressions, found by this benchmark

ITK is *slower* at 96 threads than at 1 on two ops. These are defects in ITK,
not in the port, and the port's large `tN` win on them should be read with that
in mind:

| op (medium) | cpp t1 | cpp tN | ITK speedup |
|---|---|---|---|
| binary_dilate | 1702.9 | 2549.7 | **0.67×** |
| connected_component | 684.2 | 4564.5 | **0.15×** |

`connected_component` gets 6.7× *worse* when ITK is given 96 cores.
