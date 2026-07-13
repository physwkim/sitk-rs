# sitk-rs

A **pure-Rust port of [SimpleITK](https://simpleitk.org/)** — no ITK/C++ linkage.

> **Status: broad and deep, not complete.** The core model, ten image
> formats, ~90 filter modules, seventeen transform types, and a registration
> framework (six metrics, twelve optimizers, multi-resolution pyramid) are
> implemented and tested — **3,329 tests** on the CPU, **3,357** with the CUDA
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

End to end, `load → cast → rescale → smooth → register` at 256³ is
**17,880 ms on 96 CPU threads against 240.5 ms device-resident, 74×**. A real
`ImageRegistrationMethod::execute()` at 256³ is **16.2–17.7 s host against
148.6–151.0 ms on the device — 107–119×**. The device Gaussian is
*bit-identical* to the CPU filter (`f64` weights and intermediates,
`__dmul_rn`/`__dadd_rn` to forbid FMA contraction — an FMA would be *more*
accurate and therefore *different*).

Device ops: cast (every one of the 10 scalar pixel types uploads; the 12
complex/vector ones are refused **by name**, enforced by an exhaustive match, so
a new `PixelId` is a compile error rather than a silent refusal),
`rescale_intensity`, `smooth_gaussian`, `recursive_gaussian`, `shrink`,
`resample_linear`, and a mean-squares metric. `execute_on_device` drives a full
multi-resolution pyramid: every level image is **bit-identical** host vs. device,
and a `[4,2,1]`/`[2,1,0]` schedule takes the same 154 iterations to the same
236,479 valid points on both paths, with a worst parameter disagreement of
6.1e-13.

**The caveats, stated plainly:**

- **The 74× and 107–119× numbers above are single-level, and they predate a
  metric-kernel fix.** The kernel's continuous index was being formed with FMA
  contraction and with the transform offset seeded into the accumulator, where
  the host adds it last; a 1-ULP difference flips `floor()` for a sample sitting
  exactly on a voxel plane, and the trilinear gradient is discontinuous there, so
  the kernel took the *opposite one-sided derivative* — `d/d(angle_y)` off by
  **34%** while the *value* agreed to 1e-15, which is why every value-only check
  passed it. Fixed at source (`__dmul_rn`/`__dadd_rn`/`__dsub_rn` in the host's
  exact order, 3.2e-14 after). The timings have **not** been re-taken on the
  fixed kernel; a pyramid run has not been timed at all.
- The device metric is mean-squares, full grid, linear interpolation. No device
  Mattes/correlation/ANTS, no masks, no sampling strategies.
- Device 0 only. Four GPUs are present; multi-GPU is untouched.
- ITK itself has no CUDA path (its only GPU registration is an OpenCL Demons
  filter), so this is new acceleration, not a port.

The CPU path is unaffected: the test suite passes with the feature **off**
(3,329), with it **on** (3,357), and with it on but `CUDA_VISIBLE_DEVICES=""`
(3,357) — a machine with no GPU is a supported configuration, not a crash.

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

The headline: the port wins on `binary_dilate` (0.20×), `connected_component`
(0.22×), `rescale_intensity` (0.45×), `signed_maurer_distance_map` (0.39×), is
level on `otsu`/`median`/`fft_convolution`, and **loses on the separable/stencil
family** — `mean` 4.4×, `gradient_magnitude` 4.3×. That loss was a *scaling* gap,
not a constant-factor one, and its cause turned out to be **glibc's allocator**:
`mean` was making **30,910,860 heap allocations per call** on the neighborhood
boundary path, so at 48 threads the window walk ran 13.8 busy cores — blocked, not
stalled. Fixed structurally (`push_values_checked` now takes a `&mut [i64]`, and a
slice cannot grow, so the function *has no way* to allocate); `mean` at t48 went
338.6 → 185.1 ms with all 16 `bit_parity` checksums unmoved. **The published table
still shows the pre-fix numbers** — a clean sweep is owed, and the doc says so
rather than quietly re-labelling.

Read §0 of that document before quoting any number from it; it says how much of
each one you can trust.

## Roadmap

1. **Finish the CPU scaling work.** The allocator was the dominant cause and is
   fixed, but two things remain: `mean` now scales perfectly to 16 threads and
   then stops dead (196 ms at t16, 185 at t48, ~20 busy cores — cause unknown,
   chunk granularity ruled out), and `gradient_magnitude_recursive_gaussian`'s
   1.88× loss is a *different* defect (44,429 allocations, never touches the
   boundary path; most likely its per-axis full-volume `to_f64_vec()` copies).
2. **Re-measure.** The published tables predate both the allocator fix and the
   device metric-kernel fix.
3. **Device coverage.** Mattes/correlation/ANTS on the device; masks and sampling
   strategies. Then multi-GPU.
4. **Filter breadth.** SimpleITK's `Code/BasicFilters/yaml/*.yaml` definitions
   are intended to be consumed directly to generate the remaining wrappers;
   the algorithm bodies are what get written in Rust.
5. **Close §5.** The open parity-vs-correctness decisions in the ledger.

## Build

```sh
cargo build --workspace
cargo nextest run --workspace                        # 3,329 tests
cargo nextest run --workspace --features sitk-filters/cuda   # 3,357, needs CUDA 13
```

License: Apache-2.0.
