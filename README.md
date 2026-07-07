# sitk-rs

A **pure-Rust port of [SimpleITK](https://simpleitk.org/)** — no ITK/C++ linkage.

> **Status: early, registration-focused.** The core model, MetaImage IO, a
> handful of filters, affine resampling, and a working **registration**
> (mean-squares + gradient descent over a multi-resolution pyramid, recovering
> known translations/affines) are implemented and tested end to end. This is a
> foundation to build on, **not** a usable SimpleITK replacement yet — full
> filter coverage is deliberately deferred in favour of deepening registration.

## Why a rewrite, not a binding

SimpleITK is a thin facade: its ~298 filters are code-generated wrappers that
instantiate templated `itk::*ImageFilter` classes, and its `Image` wraps
`itk::Image`. The real numerical algorithms live in **ITK** (~1.5–2M LOC of
templated C++). A *pure-Rust* port therefore means porting the ITK algorithms
SimpleITK exposes — the facade itself is small. This repo ports the facade first
and fills in algorithms behind it, referencing ITK for behavioural parity.

The prior `sitk-registration-sys` crate is an autocxx binding (2D Elastix
registration only, ~6 exposed symbols) and is **not** the basis for this port.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `sitk-core` | Runtime-typed `Image`, pixel dispatch (`dispatch_scalar!`), physical-space geometry (spacing/origin/direction) |
| `sitk-io` | Image file IO — MetaImage (`.mha`/`.mhd`) reader/writer |
| `sitk-filters` | Pixel-wise / statistical filters, separable FIR Gaussian smoothing, `ShrinkImageFilter` (procedural API) |
| `sitk-transform` | `TranslationTransform`, `AffineTransform`, interpolation, `ResampleImageFilter` |
| `sitk-registration` | `ImageRegistrationMethod`: mean-squares metric, gradient-descent optimizer, automatic physical-shift scales/learning-rate, GPU-backend seam |
| `sitk` | Umbrella crate re-exporting the above under one namespace |

## Architecture

- **Runtime pixel type.** A SimpleITK `Image` is not templated on its pixel type
  at the API level; the type is carried at runtime and every filter dispatches on
  it. We mirror that with a `PixelId` tag + an enum-of-`Vec` buffer, recovering
  static typing inside filters through the `Scalar` trait and `dispatch_scalar!`.
- **Physical space.** Every image carries spacing, origin, and a direction cosine
  matrix; index↔physical mapping follows ITK (`p = origin + D·(spacing⊙index)`).
- **Codegen-ready.** SimpleITK's `Code/BasicFilters/yaml/*.yaml` filter
  definitions are intended to be consumed directly to generate filter wrappers in
  Phase 2 — the algorithm bodies are what get written in Rust.

## Phase 0 scope (done)

- `Image`: N-D, 10 scalar pixel types, geometry, typed dispatch, `f64` views.
- MetaImage IO: round-trips every scalar type, arbitrary dimension, full geometry
  (little/big-endian read, `.mha` local + `.mhd`+`.raw`). No compression/vector.
- Filters: `cast`, `add`/`subtract`/`multiply`/`divide` (image and constant),
  `abs`, `binary_threshold`, `rescale_intensity`, `statistics`, `minimum_maximum`.
- Transforms: `TranslationTransform`, `AffineTransform` (both parametric, with
  parameter Jacobians); `ResampleImageFilter` with nearest-neighbour and linear
  interpolation.
- End-to-end acceptance test: read → cast/scale/bias → resample(affine) →
  threshold → write → read-back.

## Registration (focused vertical slice)

`sitk-registration` ports the smallest end-to-end slice of ITK's v4 registration
framework (`itk::ImageRegistrationMethodv4` / SimpleITK `ImageRegistrationMethod`),
chosen as the near-term focus over broad filter coverage:

- **Metric:** mean squares (`itk::MeanSquaresImageToImageMetricv4`), sampled over
  the full fixed grid, with value **and** analytic parameter-derivative.
- **Optimizer:** gradient descent (`itk::GradientDescentOptimizerv4`) with
  per-parameter scales, early stop on a tiny step, and value-plateau convergence
  monitoring (`itk::WindowConvergenceMonitoringFunction`).
- **Automatic scales + learning rate:** both are estimated from physical shift
  (`itk::RegistrationParameterScalesFromPhysicalShift` + the optimizer's
  learning-rate estimation), so no hand-tuning is required — call
  `set_optimizer_scales_from_physical_shift()` and
  `set_optimizer_as_gradient_descent_estimated(iterations, estimate)`. Both ITK
  estimation modes are supported: `Once` (estimate from the initial gradient
  then hold fixed — ITK's default, refines to high precision) and
  `EachIteration` (re-estimate every step; the ~1-voxel non-shrinking step is
  stopped by the convergence monitor and recovers coarsely, ≈ voxel precision).
- **Interpolation:** linear, with the exact gradient of the interpolant used for
  the metric derivative.
- **Transforms optimized:** `TranslationTransform` and `AffineTransform`.
- **Multi-resolution pyramid:** an optional coarse-to-fine shrink/smoothing
  schedule (`itk::ImageRegistrationMethodv4`'s per-level scheme). Per level the
  fixed image is Gaussian-smoothed and placed on the shrunk **virtual-domain**
  grid (`itk::ShrinkImageFilter` geometry; the fixed values are resampled onto it
  by linear interpolation, exactly as ITK's metric interpolates the smoothed
  fixed at each virtual point — reusing the shrunk pixel values would inject the
  filter's deliberate ≤½-voxel origin/offset skew as a translation bias), the
  moving image is Gaussian-smoothed, and the coarse-level transform initializes
  the next finer one. Configure with `set_shrink_factors_per_level([4,2,1])` +
  `set_smoothing_sigmas_per_level([2,1,0])`.

Verified end to end with **automatic** scales/learning-rate (no hand-tuned
values): at a single level, `Once` recovers a known translation to ~1e-3 and a
translation-through-affine (6 parameters) to <1e-2; `EachIteration` recovers the
translation to ≈ voxel precision (~0.5 voxel) and stops on the value plateau
(13 iterations); manual scales/learning-rate paths remain available and recover
the translation to ~1e-8. A 3-level pyramid recovers a translation to sub-voxel
accuracy and, crucially, **captures offsets a single level cannot** — e.g. two
σ=5 blobs ~21 voxels apart (no overlap → zero single-level gradient → stuck) are
aligned to ~0.04 voxel by the pyramid.

The estimate-`Once` learning rate is additionally **capped per step** at the
estimator's one-voxel maximum shift. This is a no-op for a converging run (the
fixed rate already bounds every step) but prevents a pyramid level that restarts
from a near-converged transform — where the once-estimated rate is derived from a
~0 gradient and is therefore enormous — from taking an exploding step. Because
each level restarts near the previous optimum, the pyramid's finest level settles
to sub-voxel rather than 1e-8 precision on these synthetic blobs; a following
single-resolution `Once` pass refines further when needed.

Not yet: bit-exact recursive Gaussian (the FIR smoother is result-faithful, not
byte-identical to `itk::RecursiveGaussianImageFilter`); the other metrics (Mattes
MI, ANTS CC, correlation, Demons) and optimizers (LBFGS, Amoeba, Powell, …);
sampling strategies.

### GPU acceleration (CPU now, CUDA-ready seam)

The metric's per-sample reduction is isolated behind a `MetricBackend` trait.
Only the host `CpuBackend` ships and is tested — **this repo is developed on
Apple Silicon, which has no NVIDIA GPU or CUDA toolkit, so CUDA code cannot be
compiled or verified here.** A CUDA (`cudarc`) or portable `wgpu`/Metal backend
implements the same trait and drops in via
`ImageRegistrationMethod::set_metric_backend` with no change to the metric or the
registration loop. (ITK itself has no CUDA path; its only GPU registration is an
OpenCL Demons filter — so a CUDA metric backend is new acceleration, not a port.)

## ITK parity — verified against ITK v6 source

The Phase-0 numerics were cross-checked against the ITK source
(`/Users/stevek/codes/ITK`, v6.0b02). Reference files:

| Behaviour | sitk-rs | ITK reference |
|---|---|---|
| index↔physical mapping | `sitk-core/image.rs` | `itkImageBase.h` |
| image⊕image arithmetic | `sitk-filters/lib.rs` (`Arith`) | `itkArithmeticOpsFunctors.h` |
| nearest-neighbour rounding | `(c+0.5).floor()` | `Math::RoundHalfIntegerUp` (`itkImageBase.h`) |
| inside-buffer test `-0.5 ≤ c < size-0.5` | `resample.rs::is_inside` | `itkImageFunction.hxx` |
| linear boundary neighbour clamp | `.clamp(0, size-1)` | `itkLinearInterpolateImageFunction.hxx` (`std::min/max`) |
| sample variance `(ΣΣ − Σ²/n)/(n−1)` | `statistics` | `itkStatisticsImageFilter.hxx` |
| physical-shift scales/learning-rate (`δ=0.01`, `scale=(shift/δ)²`, `lr=maxStep/stepScale`) | `sitk-registration/scales.rs` | `itkRegistrationParameterScalesFromShiftBase.hxx`, `itkGradientDescentOptimizerv4.hxx` |
| value-plateau convergence (order-1/2-CP linear fit slope of the windowed energy) | `sitk-registration/convergence.rs` | `itkWindowConvergenceMonitoringFunction.hxx` |
| shrink geometry (`outSpacing=inSpacing·f`, `outSize=⌊inSize/f⌋`, center-preserving origin, integer sampling offset) | `sitk-filters/shrink.rs` | `itkShrinkImageFilter.hxx` |
| multi-resolution per-level scheme (shrink virtual domain; interpolate smoothed fixed at virtual points; smooth moving; carry transform) | `sitk-registration/method.rs` | `itkImageRegistrationMethodv4.hxx` |

**Confirmed matching:** NN rounding, inside-buffer boundary, linear boundary
clamp, variance divisor, and (as of this pass) image⊕image integer wraparound
and divide-by-zero → `NumericTraits<T>::max()`.

Remaining, deliberately-scoped deviations (documented in code):

- **image⊕constant arithmetic** accumulates in `f64` (SimpleITK's `double`
  constant) and narrows with a **saturating** cast; the final out-of-range
  float→int cast is undefined in C++, so saturation is a defined choice, not a
  bug. `cast`/`rescale`/`threshold` likewise narrow via saturating `f64→int`.
- **Gaussian smoothing** is a separable **truncated-FIR** Gaussian
  (`kernel[k]=exp(-(k·spacing)²/2σ²)`, normalized, 4σ radius, physical-unit σ,
  edge-replicating boundary) — result-faithful to the same continuous Gaussian
  but **not** byte-identical to `itk::RecursiveGaussianImageFilter`'s IIR
  recursion. It sits behind the `smooth_gaussian` boundary so a bit-exact
  recursive port can replace it without touching callers.
- **estimate-`Once` learning rate** is capped per step at the estimator's
  one-voxel maximum shift; this is inactive for any converging run (so
  single-resolution results are unchanged) and exists to stop a pyramid level
  that restarts from a near-converged transform from diverging on the enormous
  rate implied by its ~0 initial gradient — a robustness bound ITK's plain
  `GradientDescentOptimizerv4` does not impose.

## Roadmap

The near-term focus is **registration**, deepened incrementally, rather than
broad filter coverage:

1. **Registration depth (current focus):** multi-resolution pyramids
   (shrink/smooth per level) — **done**, with a bit-exact
   `RecursiveGaussianImageFilter` to follow behind the smoothing seam; more
   metrics (Mattes MI, correlation, ANTS CC) and optimizers
   (RegularStepGradientDescent, LBFGS); rigid/similarity/BSpline transforms; a
   CUDA/`wgpu` `MetricBackend`.
2. **Core infra:** neighborhood iterators, regions, the functor framework
   (unblocks the `BinaryFunctor`/`UnaryFunctor` filter families).
3. **Filter breadth:** yaml→Rust codegen; port the ~247 `ImageFilter`-shaped
   filters, validated against SimpleITK's per-filter md5 baselines.

Building the algorithm-faithful phases uses the ITK C++ source as a reference.
It is available locally at `/Users/stevek/codes/ITK` (v6.0b02); it is not
vendored into this repo (SimpleITK's SuperBuild fetches it from git). The
SimpleITK facade reference is at `/Users/stevek/codes/SimpleITK`.

## Build

```sh
cargo build --workspace
cargo nextest run --workspace     # or: cargo test --workspace
```

License: Apache-2.0.
