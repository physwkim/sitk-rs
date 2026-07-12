# Benchmark contract — sitk-rs vs ITK C++

This document is the **contract** every benchmark implementation follows.
The Rust (criterion) harness, the C++ (ITK) harness, and any GPU harness
MUST generate identical inputs, run identical parameters, and emit the
same JSON schema, or the comparison is meaningless.

Do not change an op's parameters in one harness without changing the
other. If a parameter must change, change it here first.

## Machine baseline (recorded for every run)

- CPU: 96 logical cores
- GPU: 4× NVIDIA RTX 5000 Ada (32 GiB, compute capability 8.9)
- ITK reference build: `/home/stevek/work/ITK-worktrees/verify-build`
  (ITK 6.0, static libs)

## Thread configurations

Every op is measured in **three** configurations, because ITK's filters
are multithreaded by default and a 1-vs-96 comparison would be a lie:

| config | Rust | ITK C++ |
|---|---|---|
| `t1` | rayon pool pinned to 1 thread | `MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1)` |
| `tN` | rayon default pool (96) | ITK default (96) |
| `gpu` | `sitk-cuda` feature on, device 0 | (not applicable — recorded as `null`) |

`t1` is the honest scalar-vs-scalar number and the one that proves the
port's *algorithmic* parity. `tN` is the number users actually feel.

## Inputs — deterministic, identical across harnesses

Generated, never read from disk. Both harnesses implement the same
generator and MUST agree bit-for-bit (a `checksum` field in the output
pins this — see schema).

```
fn synth(seed: u64, size: [usize; 3]) -> Vec<f32>
    // xorshift64*, seeded per-volume, mapped to [0, 1000)
    // state = seed; for each voxel:
    //   state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    //   value = ((state.wrapping_mul(0x2545F4914F6CDD1D) >> 33) % 1000) as f32
```

Binary/label inputs derive from the same volume by thresholding at
`>= 500.0` (so they are also identical across harnesses).

Volume sizes (3-D, isotropic spacing `[1.0, 1.0, 1.0]`, identity
direction, origin `[0,0,0]`):

- `small`  = 64³   (262 144 voxels) — catches per-call overhead
- `medium` = 256³  (16.7 M voxels)  — the headline number
- `large`  = 512³  (134 M voxels)   — only for ops that are O(n) or
  O(n log n); skipped for ops whose ITK implementation would take
  > 120 s (record as `skipped: "too slow"`, do not silently omit)

Pixel type is `Float32` unless the op is inherently integral
(`connected_component`, `binary_erode`/`binary_dilate` take `UInt8`).

## The 12 ops

Each row: the Rust entry point, the ITK filter it ports, and the exact
parameters. Parameters were chosen to be non-degenerate (a radius-0 or
sigma-0 op measures nothing).

| # | op key | Rust entry point | ITK filter | parameters |
|---|---|---|---|---|
| 1 | `rescale_intensity` | `sitk_filters::rescale_intensity` | `RescaleIntensityImageFilter` | `output_min=0.0`, `output_max=255.0` |
| 2 | `smoothing_recursive_gaussian` | `sitk_filters::smoothing_recursive_gaussian` | `SmoothingRecursiveGaussianImageFilter` | `sigma=[2.0,2.0,2.0]`, `normalize_across_scale=false` |
| 3 | `discrete_gaussian` | `sitk_filters::discrete_gaussian` | `DiscreteGaussianImageFilter` | `variance=[4.0;3]`, `max_kernel_width=32`, `max_error=0.01`, `use_image_spacing=true` |
| 4 | `median` | `sitk_filters::median` | `MedianImageFilter` | `radius=[2,2,2]` (5³ window) |
| 5 | `mean` | `sitk_filters::mean` | `MeanImageFilter` | `radius=[2,2,2]` |
| 6 | `gradient_magnitude` | `sitk_filters::gradient_magnitude` | `GradientMagnitudeImageFilter` | `use_image_spacing=true` |
| 7 | `gradient_magnitude_recursive_gaussian` | `sitk_filters::gradient_magnitude_recursive_gaussian` | `GradientMagnitudeRecursiveGaussianImageFilter` | `sigma=2.0`, `normalize_across_scale=false` |
| 8 | `binary_dilate` | `sitk_filters::binary_dilate` | `BinaryDilateImageFilter` | `radius=[3,3,3]`, ball kernel, `foreground=1`, `background=0` |
| 9 | `signed_maurer_distance_map` | `sitk_filters::signed_maurer_distance_map` | `SignedMaurerDistanceMapImageFilter` | `inside_positive=false`, `squared_distance=false`, `use_image_spacing=true` |
| 10 | `connected_component` | `sitk_filters::connected_component` | `ConnectedComponentImageFilter` | `fully_connected=false` |
| 11 | `otsu_threshold` | `sitk_filters::otsu_threshold` | `OtsuThresholdImageFilter` | `bins=128`, `inside=1`, `outside=0` |
| 12 | `fft_convolution` | `sitk_filters::fft_convolution` | `FFTConvolutionImageFilter` | kernel = 7³ normalized box; `normalize=true` |

Ops 1, 6 are pure per-pixel / stencil (the easiest parallel + GPU wins).
Ops 2, 3, 7 are separable (parallel over lines).
Ops 4, 5, 8 are sliding-window (parallel over output voxels).
Ops 9, 10 are sequential-ish (the interesting hard cases).
Op 11 is a histogram reduction; op 12 is FFT-bound.

## Correctness gate — not optional

A benchmark that computes the wrong answer fast is worthless. Every
harness run also verifies, at `medium` size, on **one** warmup call:

- Rust: the result equals the current `main` scalar implementation's
  result **bit-for-bit** (the deterministic-reduction requirement).
  Any op that cannot meet this states so explicitly in its row of the
  results table — it does not silently pass.
- GPU: bit-for-bit is not required; report `max_abs_err` and
  `max_rel_err` against the CPU f64 result, per op.

## Output schema (both harnesses emit this)

One JSON object per (op, size, config) measurement, newline-delimited
(`.ndjson`), so results merge with `cat`:

```json
{
  "harness": "rust" | "cpp",
  "op": "median",
  "size": "medium",
  "voxels": 16777216,
  "config": "t1" | "tN" | "gpu",
  "threads": 96,
  "ms_mean": 812.4,
  "ms_median": 809.1,
  "ms_stddev": 6.2,
  "samples": 10,
  "input_checksum": "0x9f3c…",
  "output_checksum": "0x1a2b…",
  "skipped": null
}
```

- `input_checksum` — FNV-1a 64 over the input buffer's little-endian
  bytes. **Both harnesses must produce the same value** for the same
  (seed, size); a mismatch means the generators diverged and the whole
  comparison is void.
- `output_checksum` — same hash over the output buffer. Rust-vs-C++
  equality here is *not* required (ITK and the port have documented
  divergences in the ledger), but a per-harness change across commits
  signals a regression.
- `skipped` — non-null reason string instead of timings when an op/size
  is not run. Never omit the row.

## Reporting

The comparison table is generated from the merged `.ndjson`, one row per
(op, size), columns: `rust t1`, `cpp t1`, `ratio t1`, `rust tN`,
`cpp tN`, `ratio tN`, `rust gpu`, `speedup gpu vs rust tN`.

`ratio > 1.0` means the port is **slower** than ITK. State it plainly;
do not round a 2.3× regression into "comparable".
