# sitk-rs

A **pure-Rust port of [SimpleITK](https://simpleitk.org/)** — no ITK/C++ linkage.

> **Status: broad and deep, not complete.** The core model, ten image
> formats, ~90 filter modules, seventeen transform types, and a registration
> framework (six metrics, twelve optimizers, multi-resolution pyramid) are
> implemented and tested — **3,410 tests** on the CPU, **3,517** with the CUDA
> feature on. Every algorithm is checked against the ITK v6 source, and every
> upstream defect found along the way is recorded in
> [`doc/upstream-findings.md`](doc/upstream-findings.md).

## Why a rewrite, not a binding

SimpleITK is a thin facade: its ~298 filters are code-generated wrappers that
instantiate templated `itk::*ImageFilter` classes, and its `Image` wraps
`itk::Image`. The real numerical algorithms live in **ITK** (~1.5–2M LOC of
templated C++). A *pure-Rust* port therefore means porting the ITK algorithms
SimpleITK exposes — the facade itself is small. This repo ports the facade and
fills in the algorithms behind it, referencing ITK for behavioural parity.

Two things fall out of doing it in Rust rather than binding to it:

- **The C++ undefined behaviour has to go somewhere.** Signed/unsigned
  wraparound, `NaN → int` casts, reads of unallocated regions — Rust will not
  reproduce them, so each one forced a decision. They are all written down.
- **You can read the parity.** Where the port reproduces an ITK quirk it is
  pinned by a test; where it diverges, the module doc says why.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `sitk-core` | Runtime-typed `Image`, pixel dispatch (`dispatch_scalar!`), physical-space geometry, the parallel/fused primitives (`map_pixels`, `map_rows_fold_in_order`) |
| `sitk-io` | MetaImage, NIfTI, NRRD, DICOM, PNG, JPEG, TIFF, VTK, GIPL, HDF5; series reader/writer; transform IO (HDF5, MATLAB, `.tfm`) |
| `sitk-filters` | ~90 modules: morphology, level sets, distance maps, FFT/deconvolution, N4 bias field, watershed, denoising, label maps, thresholding, statistics, … |
| `sitk-transform` | 17 transform types (translation, affine, Euler, versor, similarity, scale-skew, B-spline, displacement field, composite), interpolation, `ResampleImageFilter` |
| `sitk-registration` | `ImageRegistrationMethod`: 6 metrics, 12 optimizers, physical-shift scales, multi-resolution pyramid, transform initializers, device-resident metric |
| `sitk-cuda` | Device-resident images and ops (`DeviceImage`, `upload`/`to_host`) — optional, behind the `cuda` feature |
| `sitk` | Umbrella crate re-exporting the above under one namespace |

## Architecture

- **Runtime pixel type.** A SimpleITK `Image` is not templated on its pixel type
  at the API level; the type is carried at runtime and every filter dispatches on
  it. We mirror that with a `PixelId` tag + an enum-of-`Vec` buffer, recovering
  static typing inside filters through the `Scalar` trait and `dispatch_scalar!`.
- **Physical space.** Every image carries spacing, origin, and a direction cosine
  matrix; index↔physical mapping follows ITK (`p = origin + D·(spacing⊙index)`).
- **Parallelism that cannot go nondeterministic.** Filters parallelize over rows
  via `map_rows_fold_in_order`: the per-sample work runs on rayon, and the
  combine is a **sequential in-order fold** — the identical sequence of additions
  the serial loop performed, not a re-association. The `combine` closure is never
  handed to rayon, so a thread-count-dependent reduction is *unwritable* against
  the API rather than merely discouraged. `bit_parity.rs` pins 16 ops' output
  checksums to catch it if it ever became writable.
- **The bus is a thing the caller schedules.** `sitk-cuda` exposes `DeviceImage`,
  and `upload` / `to_host` are the only two functions in the crate that cross
  PCIe. An op's signature (`&DeviceImage -> DeviceImage`) *cannot express* a
  round trip, so no filter can hide one behind your back. See
  [GPU](#gpu-device-residency-cuda) below for why that shape is the whole game.

## Registration

`sitk-registration` ports ITK's v4 registration framework
(`itk::ImageRegistrationMethodv4` / SimpleITK `ImageRegistrationMethod`).

- **Metrics:** mean squares, correlation, ANTS neighborhood correlation, Mattes
  mutual information, joint-histogram mutual information, Demons — each with
  value **and** analytic parameter-derivative.
- **Optimizers:** gradient descent (plain, line-search, conjugate-gradient
  line-search), regular-step gradient descent, L-BFGS2, L-BFGS-B, Amoeba,
  Powell, one-plus-one evolutionary, exhaustive — with the `_estimated` variants
  taking their scales and learning rate from physical shift
  (`itk::RegistrationParameterScalesFromPhysicalShift`), so no hand-tuning.
- **Multi-resolution pyramid:** per-level shrink + smoothing
  (`set_shrink_factors_per_level`, `set_smoothing_sigmas_per_level`). The fixed
  image is Gaussian-smoothed and *interpolated* onto the shrunk virtual-domain
  grid — reusing the shrunk pixel values would inject `ShrinkImageFilter`'s
  deliberate ≤½-voxel origin skew as a translation bias.
- **Initializers:** centered transform, centered versor, landmark-based, B-spline.
- **Sampling:** full grid, random, regular.

Smoothing in the pyramid uses the bit-exact recursive Gaussian
(`recursive_gaussian` — the Farnebäck 4th-order Deriche IIR that ports
`itk::RecursiveGaussianImageFilter`'s zero-order smoothing), matching ITK's
`SmoothingRecursiveGaussianImageFilter`. The separable truncated-FIR Gaussian
(`smooth_gaussian`) stays available behind the same seam.

**A warning that ITK does not print.** ITK initializes the optimizer scales to
all ones when you set neither scales nor an estimator
(`itkObjectToObjectOptimizerBase.cxx:103-107`), and the estimators are opt-in. On
a rotation-bearing transform, unit scales make descent *chaotic* — a ~500×
amplification per step, so two mathematically identical paths converge to
different local minima. Call `set_optimizer_scales_from_physical_shift()`. This
is reproduced ITK behaviour, not a port defect; recorded as ledger §2.157.

## GPU: device residency (CUDA)

Behind the optional `cuda` feature. **Not a seam waiting for an implementation —
it ships, it is tested, and it is measured.**

The API is built around one fact: an op that takes an `&Image` and returns an
`Image` has to cross PCIe twice, and at 256³ that is ~17 ms of bus to do ~1 ms of
work. Such an API can never win, however fast the kernel is. So there isn't one.
`DeviceImage` stays on the device; `upload` and `to_host` are the only crossings;
CPU fallback lives at the pipeline boundary, by name, never per-call.

Measured at 256³, `rescale_intensity`, one machine state (2× Xeon, 48 physical
cores; RTX 5000 Ada):

| | CPU 1 thread | CPU 96 threads | GPU one-shot | **GPU resident** | ITK 96 threads |
|---|---|---|---|---|---|
| ms | 92.0 | 17.4 | 36.1 | **1.04** | 38.3 |

The one-shot round trip really does lose to the CPU, by 2×. The resident kernel
really does win, by **16×** over the port's CPU and **37×** over ITK — and its
output is **bit-exact** against the CPU reference (`max_abs_err = 0.0`), not
merely inside a tolerance. Both facts live in the same row.

A real `ImageRegistrationMethod::execute()` — not an evaluate loop, both paths free
to pick their own iteration count, and every run agreeing on iterations, valid
points, stop reason, and parameters to 3e-14:

| | host | device | speedup |
|---|---|---|---|
| single level, 256³ | 18.5–23.2 s | 209–210 ms | **88–111×** |
| pyramid `[4,2,1]`, 256³ | 25.8–28.2 s | 291–297 ms | **87–97×** |
| pyramid `[4,2,1]`, 128³ | 2.7 s | 42 ms | **63×** |

**The pyramid is not what costs you — the volume is.** The device pays 83 ms to
build the extra levels; the ratio barely moves. But at 128³ the GPU's fixed costs
stop being amortized, so a reader running a small registration should expect ~60×,
not ~100×. The device Gaussian is
*bit-identical* to the CPU filter (`f64` weights and intermediates,
`__dmul_rn`/`__dadd_rn` to forbid FMA contraction — an FMA would be *more*
accurate and therefore *different*).

Device ops: cast (every one of the 10 scalar pixel types uploads; the 12
complex/vector ones are refused **by name**, enforced by an exhaustive match, so
a new `PixelId` is a compile error rather than a silent refusal),
`rescale_intensity`, `smooth_gaussian`, `recursive_gaussian`, `shrink`,
`resample_linear`, `resample_nearest`, a constant fill (`DeviceImage::filled`,
which is what keeps the all-ones in-buffer predicate off the bus — 67 MB per
level not uploaded at 256³), two mask kernels behind `DeviceMask`
(`threshold_nonzero`, `mask_and`), and a mean-squares metric.
`execute_on_device` drives a full
multi-resolution pyramid: every level image is **bit-identical** host vs. device,
and a `[4,2,1]`/`[2,1,0]` schedule takes the same 154 iterations to the same
236,479 valid points on both paths, with a worst parameter disagreement of
6.1e-13.

**The caveats, stated plainly:**

- **Those numbers are 15–22% worse than the ones this file used to publish, and the
  regression is a bug fix.** The kernel's continuous index was being formed with FMA
  contraction and with the transform offset seeded into the accumulator, where the
  host adds it last; a 1-ULP difference flips `floor()` for a sample sitting exactly
  on a voxel plane, and the trilinear gradient is discontinuous there, so the kernel
  took the *opposite one-sided derivative* — `d/d(angle_y)` off by **34%** while the
  *value* agreed to 1e-15, which is why every value-only check passed it. Fixed at
  source (`__dmul_rn`/`__dadd_rn`/`__dsub_rn` in the host's exact order, 3.2e-14
  after), and the per-iteration cost rose from 6.9 ms to 8.4 ms. The old 107–119×
  was the number for a derivative that was wrong.
- A `Float64` registration narrows to the device's `f32` payload — and that is a
  **measured** limit, not an asserted one. Narrowing costs **6.3e-10 of the pose**:
  four orders of magnitude below the band the optimizer itself imposes (below), and
  the device contributes none of it — the device's cast is bitwise the host's
  `as f32`. An `f64` payload would close nothing a caller can observe, so it is not
  built.
- The device metric is mean-squares, linear interpolation, **with fixed and moving
  masks, a virtual domain, and random/regular sampling**. The device level mask is
  built in the host's own order (NN-resample the in-buffer predicate, NN-resample
  the user mask, intersect) and is **byte-equal** to the host's, so the two paths
  walk exactly the same valid points — fixed mask 59,647 on both, moving mask
  59,617 on both, virtual domain 31,124 on both, both together 12,489 on both, 25
  iterations either way.
- **That equality holds because the device computes the host's continuous index,
  bit for bit — not because it computes one close to it.** The device replays the
  transform's own point-map **stages** (each stage's stored matrix and offset,
  applied in the host's application order, rounding once per stage — folding two
  stages into `G·F` is algebraically identical and is pinned to *disagree on the
  bits*, so the stage list is not ceremony). Pinned by `to_bits()` equality at
  every sample of 240 random poses across six transform families, a three-stage
  composite among them. This matters because the metric has three **discrete**
  consumers of that index — `floor` picks the cell, `is_inside` decides validity,
  `round` picks the mask voxel — and a discrete output has no tolerance to spend.
  An earlier design *probed* the affine instead (`b = T(0)`, `A[:,e] = T(e_e) − b`),
  landing ~1e-14 away, and it was measurably wrong: with a face of samples on the
  buffer boundary under a z-rotation, **3 of 17** ulp-swept poses disagreed about
  `valid_points` by 16 samples, and with a moving-mask wall on a half-integer,
  **4 of 17** disagreed by 16 in the *opposite* direction. Both sweeps now
  disagree at **0 of 17** while still crossing the boundary (`cuda_boundary.rs`).
  The cost is named rather than hidden: **`ScaleTransform` and
  `ScaleLogarithmicTransform` evaluate `(p − c)·s + c`, have no bitwise
  matrix/offset form, and are now refused from the device metric by name**
  (`NoBitwisePointMap`) instead of being approximated — the refusal test proves
  both halves, that the probed form did reproduce `transform_point` to 1e-9 (so
  the old code accepted it) and did not reproduce it on the bits (so accepting it
  was wrong).
- **On a sampled run the device does not draw.** `FixedSamples::from_image_with`
  stays the single owner of *which voxels*, and the device is handed its flat-index
  list (8 bytes per sample, and the kernel derives the point from the same closed
  form the full-grid path uses). Sameness is not a property two implementations
  agree on and a test hopes to catch — there is one implementation, so it is the
  same list, and the pin asserts exactly that, element for element.
- **No device Mattes yet — it was a refusal, and the reason for it has now been
  removed.** Mattes needs a joint histogram, and the natural GPU form is `atomicAdd`,
  whose summation order is undefined — not merely different from the host's, but
  *different on each run of the same binary*. That is measured on this box, not
  quoted: **7 of 7 re-runs** of the atomic form over 2²¹ entries into 2,500 bins
  returned different bits, the worst differing in **2,148 of 2,500 bins** at 1.15e-12
  relative — exactly the magnitude that flips the optimizer's overshoot test, so the
  same binary would send **itself** to two different poses. A pin that cannot fail
  when the code is wrong is not a pin, so the metric was refused rather than shipped.
  The reduction it was waiting for now exists (`sitk_cuda::histogram`): a **bin-keyed
  counting sort** — a sub-tile entry's rank is *counted*, not claimed from an atomic
  counter, which is the whole difference — then a **left-to-right segment sum**. That
  order *is* the host's naive loop, so it is pinned **bit-identical** to
  `for i in 0..n { h[k[i]] += v[i] }` rather than banded, and it is invariant to the
  launch configuration (block sizes 32…1024, identical bits) and to the run (8 runs,
  identical bits). Its counter-pin asserts that the atomic form *does* disagree with
  itself, so the module can never quietly become pointless. **The metric is not built
  on it yet, and the caveat stands until it is.** **ANTS** is last.
- A **fixed-initial transform** works for the nine matrix-offset transform classes
  (`Affine`, `Euler3D`, the versor/similarity family, `Translation`) **and for a
  `Composite` of them**: the device resample replays the transform's own point-map
  stages, in the transform's own order, and is bit-identical to `ResampleImageFilter`
  through every one of them — pinned on a grid where every continuous index is a
  half-voxel tie, so `floor(c + 0.5)` is a tie at every sample. A composite is
  *replayed*, not folded: `M₂·M₁` is the same map in exact arithmetic and rounds once
  where the transform rounds twice, and on that tie grid the fold differs from the host
  at 1,767 of 32,768 voxels (linear) and 158 of 32,768 (nearest). **`Scale` and
  `ScaleLogarithmic` are refused, not approximated** — they evaluate `(p−c)·s + c`,
  which is a *different rounding* from `M·p + b`, so probing a matrix out of them would
  be wrong in the fifteenth digit, and the in-buffer predicate is a 0/1 field rounded
  with `floor(c + 0.5)`, where one ulp is a whole voxel. `BSpline`,
  `DisplacementField`, and any `Composite` containing one of these four, are refused
  too, by name, and fall to the host.
- **With a fixed-initial transform, the exactly-equal valid-point count above does
  not survive a converged run** — host 152,383 against device 152,385 at 25
  iterations — and the cause is not the transform. The device reduces residuals in a
  different order than the host (~1e-13 relative, present with or without the
  transform); a fixed-initial transform puts a hard zero shell at the resampled
  border; and `RegularStepGradientDescentOptimizer` **halves its step on overshoot**,
  which is a discontinuous branch. One flipped overshoot test sends the two runs to
  two different — both valid — poses, where two border samples of 152,383 fall on
  opposite sides of the moving buffer. The counts are exactly equal when both paths
  are evaluated at the *same parameters*, which is where equality is a real property
  rather than the optimizer's luck; the converged run is pinned to the same iteration
  count, the same stop reason, and the same pose to 1e-3 (worst measured 2.4e-5).
  The attribution is **measured, not argued**, and the point map is not part of it:
  swept against iteration count, *without* the transform the reduction difference
  stays at 1e-13 through 25 iterations and the counts **never** move, while *with* it
  the parameter gap jumps 6.09e-10 → 1.89e-5 between iterations 5 and 10 — that is
  one flipped overshoot test — and the counts diverge from there. The zero shell is
  the enabler, the reduction order is the input, and the optimizer's branch is the
  amplifier.
- Device 0 only. Four GPUs are present; multi-GPU is untouched.
- ITK itself has no CUDA path (its only GPU registration is an OpenCL Demons
  filter), so this is new acceleration, not a port.

The CPU path is unaffected: the test suite passes with the feature **off**
(3,410), with it **on** (3,517), and with it on but `CUDA_VISIBLE_DEVICES=""`
(3,517) — a machine with no GPU is a supported configuration, not a crash.

## ITK parity — and what we found in ITK

Every algorithm is written against the ITK v6 source
(`v6.0b02-5846-ge46eb723a5`, checked out at `~/work/ITK`; SimpleITK's yamls at
`~/work/SimpleITK`). Neither is vendored here.

Porting a numerical library line-by-line turns out to be an excellent way to find
bugs in it. [`doc/upstream-findings.md`](doc/upstream-findings.md) is the index:

| Section | Rows | What it holds |
|---|---|---|
| §1 | 74 | **ITK bugs** — wrong results, NaN, or C++ UB on live code paths |
| §2 | 157 | ITK inconsistencies and quirks — **reproduced** and pinned by tests |
| §3 | 56 | SimpleITK wrapping issues |
| §4 | 116 | **Deliberate divergences of this port** — each with its reason |
| §5 | — | Open decision points (parity vs. correctness, awaiting a call) |
| §7 | 2 | ITK *performance* defects — ops that get slower with more threads |

The §1 bugs and the upstream-relevant §2 rows were re-verified against that
checkout and filed upstream on 2026-07-10 as
[ITK issue #6575](https://github.com/InsightSoftwareConsortium/ITK/issues/6575).
Thirty-nine of them are **fixed in this port** rather than reproduced.

The policy, per entry:

- **reproduced** — upstream behaviour is reproduced bit-for-bit, quirk included,
  and pinned by a test. Parity wins over correctness.
- **diverged** — this port intentionally differs; the module doc states the
  divergence and the reason (memory safety, defined behaviour where C++ has UB,
  or `f64` precision).
- **open** — a decision is pending.

The authoritative text for any entry is the module doc it names. The ledger is
the index, not the source of truth.

## Benchmarks

[`doc/bench-results.md`](doc/bench-results.md) — twelve ops against ITK 6.0 C++
at three sizes, single-threaded and all-cores, two independent runs, with the raw
NDJSON frozen in [`bench/results/`](bench/results/) and the contract in
[`doc/bench-spec.md`](doc/bench-spec.md). `bench/compare.py` proves both
harnesses received a byte-identical input before any timing is compared.

At 256³, all cores, `rust/itk` (below 1.00 means the port is faster): the port wins
on `binary_dilate` **0.03×**, `connected_component` 0.25×, `signed_maurer_distance_map`
0.30×, `median` 0.38×, `rescale_intensity` 0.42×, `gmrg` 0.47×, `otsu` 0.57×,
`gradient_magnitude` 0.64×, `discrete_gaussian` 0.76×, `mean` 0.80×,
`fft_convolution` 0.87×; and **loses on `smoothing_recursive_gaussian` (1.02×)**.
At 512³ `gradient_magnitude` and `gmrg` are ties (1.02×, 1.01×) — ties, not wins.

**Where the port loses now is small volumes**, not large ones: at 64³,
`gradient_magnitude` is 2.91×, `otsu` 6.62× and `mean` 4.52× — fixed per-call cost
that the reference size amortizes away.

The interesting one is `mean`. It used to lose by **4.39×**, and the cause was not
the kernel, the decomposition, bandwidth, or NUMA — each eliminated by measurement.
It was **glibc's allocator**: `mean` made **30,910,860 heap allocations per call** on
the neighborhood boundary path, so at 48 threads the window walk ran 13.8 busy cores.
The threads were not stalled, they were *blocked*. Fixed structurally —
`push_values_checked` now takes a `&mut [i64]`, and **a slice cannot grow, so the
function has no way to allocate** — and `mean` went from 4.39× slower than ITK to
**0.80×**, faster, with all 16 `bit_parity` checksums unmoved.

The second-most interesting is what the *fix* to `gmrg` then cost. Removing twelve
full-volume allocations had replaced a **parallel** `f32→f64` widening with a
**serial** `memcpy` — and because `vec![0.0; n]` is a lazily-zeroed mmap, the
page-fault bill lands on whichever phase touches the buffer first. That phase was
now serial: **517 ms** at 512³, against **79.7 ms** for the parallel widening of a
buffer of exactly the same size. The fix won at 256³ and *lost* at 512³. The copy
is now deleted rather than parallelized, and a related finding fell out of it — a
`+=` into a fresh zeroed buffer costs **two page faults per page** (the read faults
in the shared zero page, the write takes a second write-protect fault) where a
plain store costs one.

Read §0 of that document before quoting any number from it; it says how much of
each one you can trust.

## Roadmap

1. **Small-volume overhead** — now the port's only real loss. `gradient_magnitude`
   is 2.91× at 64³, `otsu` 6.62×, `mean` 4.52×: fixed per-call cost that the
   reference size amortizes away and the headline hides. None of the three
   `gradient.rs` fixes moved `gradient_magnitude`'s small case by more than its own
   noise, because it is overhead-bound, not bandwidth-bound.
2. **`smoothing_recursive_gaussian` at 512³ (1.75×)**, not investigated. And three
   un-parallelized stencils next to the ones just fixed — `derivative`, `laplacian`,
   `sobel_edge_detection` still run a serial `iter().map().collect()`. They are not
   in the benchmarked twelve, so no number is claimed for them.
3. **Device coverage.** A **Correlation** metric — its moments are the same shape
   the mean-squares kernel already reduces deterministically, so it inherits the
   exactness story unchanged. Mattes needs a deterministic histogram built first
   (above); ANTS is a new kernel shape and is last. Then multi-GPU — four GPUs are
   present and device 0 is the only one used.
4. **Filter breadth.** SimpleITK's `Code/BasicFilters/yaml/*.yaml` definitions
   are intended to be consumed directly to generate the remaining wrappers;
   the algorithm bodies are what get written in Rust.
5. **Close §5.** The open parity-vs-correctness decisions in the ledger.

## Build

```sh
cargo build --workspace
cargo nextest run --workspace                        # 3,410 tests
cargo nextest run --workspace --features sitk-filters/cuda   # 3,517, needs CUDA 13
```

License: Apache-2.0.
