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

Every op is measured in **four** configurations, because ITK's filters
are multithreaded by default and a 1-vs-96 comparison would be a lie —
and because a GPU number that includes the bus and one that does not are
different facts, both true, and neither one usable without the other:

| config | Rust | ITK C++ |
|---|---|---|
| `t1` | rayon pool pinned to 1 thread | `MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1)` |
| `tN` | rayon default pool (96) | ITK default (96) |
| `gpu` | `sitk-cuda` feature on, device 0; the **one-shot** API `fn(&Image) -> Image` — H2D, kernel, D2H, every call | (not applicable — recorded as `null`) |
| `gpu_resident` | same device, the **device-resident** API `fn(&DeviceImage) -> DeviceImage` — the volume is already on the device and the result stays there; upload and download happen once, outside the timed region | (not applicable — recorded as `null`) |

`t1` is the honest scalar-vs-scalar number and the one that proves the
port's *algorithmic* parity. `tN` is the number users actually feel.

`gpu` and `gpu_resident` measure the same kernel and differ only in what
surrounds it, which is the point: `gpu` prices a round-trip API, and
`gpu_resident` prices the op. A pipeline of `k` ops pays the crossing once
if it stays resident and `k` times if it does not, so **the difference
between these two columns is the cost of the API shape, stated in
milliseconds** rather than argued. Neither number may be quoted without the
other.

`gpu_resident` is Rust-only and, today, `rescale_intensity`-only. `sitk-cuda`
has exactly two device-resident ops; the other is `smooth_gaussian`, which is
**not** one of the twelve — it is a different filter from `discrete_gaussian`
(a truncated `exp(-k²/2σ²)` FIR against ITK's `DiscreteGaussianImageFilter`),
not a device port of it. Every other (op, size) still emits a `gpu_resident`
row saying exactly that, per the never-omit-the-row rule below.

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

**Seed is fixed at `42`** for every volume, every size, every harness.
(Amended after the first C++ run: the original spec defined the
generator but left the seed unstated, which would have voided the
`input_checksum` equality it depends on.)

Binary/label inputs derive from the same volume by thresholding at
`>= 500.0` (so they are also identical across harnesses).

### The three input variants, and which op takes which

An op's `input_checksum` is only comparable across harnesses if both
feed it the *same variant*. Ops do not all take the base volume:

| variant | content | pixel type | hashed as |
|---|---|---|---|
| `base_f32` | raw generator output, `[0, 1000)` | Float32 | f32 LE bytes |
| `mask_u8` | `base >= 500.0 ? 1 : 0` | UInt8 | u8 bytes |
| `mask_f32` | `base >= 500.0 ? 1.0 : 0.0` | Float32 | f32 LE bytes |

- `base_f32` — ops 1, 2, 3, 4, 5, 6, 7, 11, 12
- `mask_u8` — ops 8 (`binary_dilate`), 10 (`connected_component`)
- `mask_f32` — op 9 (`signed_maurer_distance_map`). A signed distance
  map is only meaningful on binary content, but the port's filter takes
  a Float32 image, so the input is binary *content* in a Float32 *type*,
  with `background_value = 0.0`. This resolves the original spec's
  contradiction (it said "Float32 unless inherently integral" and then
  listed only ops 8/10 as integral, leaving op 9 undefined).

Reference checksums, seed 42 (produced by the C++ harness, verified
against an independent Python implementation with exact u64 arithmetic;
first five voxels are `59, 641, 384, 121, 923`). **Every harness must
reproduce these. A mismatch voids the comparison — report it, do not
work around it.**

| size | `base_f32` | `mask_u8` | `mask_f32` |
|---|---|---|---|
| small 64³ | `0xa60a081f21af857e` | `0x5fb1f4b900bd027a` | `0x13c82d199e0a5b88` |
| medium 256³ | `0xb04930cda0bbce53` | `0x4d2b8759782954c6` | `0x4bb5460e5493a3e8` |
| large 512³ | `0xfbf1951b8b4b69aa` | `0x25cdf3c6351a03ae` | `0x4b1fb2c019bb1d68` |

FNV-1a 64: offset `0xcbf29ce484222325`, prime `0x100000001b3`, hashed
one byte at a time over the little-endian bytes of the buffer **as
handed to the filter**.

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
  "config": "t1" | "tN" | "gpu" | "gpu_resident",
  "threads": 96,
  "ms_mean": 812.4,
  "ms_median": 809.1,
  "ms_stddev": 6.2,
  "samples": 10,
  "input_checksum": "0x9f3c…",
  "output_checksum": "0x1a2b…",
  "max_abs_err": null,
  "max_rel_err": null,
  "skipped": null
}
```

- `max_abs_err` / `max_rel_err` — the GPU correctness gate's two numbers,
  measured against the CPU f64 result. Non-null only on `config: "gpu"`
  and `config: "gpu_resident"` rows; `null` on CPU rows. (Added after the
  first GPU run: the correctness-gate section above mandated these numbers
  but this schema block did not carry fields for them.) A `gpu_resident`
  row is graded the same way and to the same tolerance as its `gpu` row:
  the resident result is brought home **once**, outside the timed region,
  purely to be checked. A residency measurement that skipped the check
  would be measuring an unverified kernel.

- `input_checksum` — FNV-1a 64 over the input buffer's little-endian
  bytes. **Both harnesses must produce the same value** for the same
  (seed, size); a mismatch means the generators diverged and the whole
  comparison is void.
- `output_checksum` — same hash over the output buffer. Rust-vs-C++
  equality here is *not* required (ITK and the port have documented
  divergences in the ledger), but a per-harness change across commits
  signals a regression.
- `skipped` — non-null reason string instead of timings when an op/size
  is not run. Never omit the row. **The reason must name the actual
  cause.** A reason that names the wrong cause is worse than no row: a
  reader who sees "feature not present" concludes the backend is
  missing, when in fact the *kernel* for that op was never written. The
  two are different facts and must not be collapsed. Concretely, a GPU
  row is skipped for one of two distinct reasons — "no GPU kernel
  implemented yet for this op" (true regardless of build flags) or
  "the `cuda` feature is off in this build" (the kernel exists; this
  build excludes it) — and the row says which.

- A CPU-labeled row (`t1`/`tN`) must call the CPU implementation
  **explicitly**, never the feature-sensitive public dispatcher. In a
  `cuda`-enabled build the dispatcher would route to the GPU, and the
  harness would silently report GPU timings in a CPU column.

## Reporting

The comparison table is generated from the merged `.ndjson`, one row per
(op, size), columns: `rust t1`, `cpp t1`, `ratio t1`, `rust tN`,
`cpp tN`, `ratio tN`, `rust gpu`, `speedup gpu vs rust tN`.

`ratio > 1.0` means the port is **slower** than ITK. State it plainly;
do not round a 2.3× regression into "comparable".

## Known properties of the ITK baseline (measured, not assumed)

These are facts about *this ITK build*, established by the first C++ run.
They must be carried into the comparison table, because a ratio against
a degraded baseline is a misleading ratio:

- **Threader is `Pool`**, not TBB (`Module_ITKTBB=OFF`).
- **`binary_dilate` and `connected_component` are SLOWER in ITK at 96
  threads than at 1** (0.67× and 0.15× respectively — i.e. a 1.5× and
  6.7× multithreading *regression* in ITK itself). A port that beats
  ITK's `tN` on those two ops has beaten a regression, not a baseline.
  Quote both `t1` and `tN` for them and say so.
- **No FFTW** (`ITK_USE_FFTWF/D=OFF`), so `fft_convolution` runs ITK's
  VNL backend at `double` precision. That is ITK-as-built, not
  ITK-at-its-best; note it rather than claiming a clean FFT win.
- Every C++ sample constructs a fresh filter so `GenerateData()` cannot
  be served from cache; per-sample times are flat, confirming this.
- **No ITK op needs the `> 120 s` skip.** The `large` (512³) `tN` run
  completes all 12 ops; the slowest are `connected_component` (35.0 s)
  and `binary_dilate` (14.9 s). So a `skipped` row on the Rust side can
  never cite ITK's runtime as the reason — if the port skips a
  (op, size, config), the reason string must state the port's own
  reason honestly (e.g. serial `t1` cost), not borrow this rule.

### Measured ITK `large` (512³) `tN` baseline, median of 5

| op | ms |
|---|---|
| rescale_intensity | 270.8 |
| smoothing_recursive_gaussian | 214.2 |
| discrete_gaussian | 479.4 |
| median | 4061.5 |
| mean | 484.4 |
| gradient_magnitude | 96.5 |
| gradient_magnitude_recursive_gaussian | 766.7 |
| binary_dilate | 14864.1 |
| signed_maurer_distance_map | 1609.7 |
| connected_component | 35000.6 |
| otsu_threshold | 157.4 |
| fft_convolution | 2094.5 |
