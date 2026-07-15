# Direction-2 type-narrowing — independent fresh-eyes verification

Adversarial re-check of the sister panel's "port keeps a NARROWER float where
ITK/SimpleITK keep double" candidates. Every claim re-derived firsthand from
ITK (`/home/stevek/work/ITK`) and SimpleITK (`/home/stevek/work/SimpleITK`);
the sister's self-acknowledged "wrong in my favor" bias means none of its
conclusions were taken on faith. Report-only, no code changed.

Verdict legend: **REAL-BUG** (port genuinely narrower than the ITK/SimpleITK
type AND the lost precision is observable in output beyond ULP) / **§4-POLICY**
(intentional NumericTraits deviation) / **NOT-REAL** (port matches; narrowing
is faithful to the output pixel type).

---

## 1. `crates/sitk-filters/src/bspline_decomposition.rs` — **REAL-BUG**

**Verdict: REAL-BUG.** For a `Float32` image the port rounds the IIR recursion
to `f32` after *every* step; ITK runs the entire recursion in `double` and
narrows to `float` only once per axis at write-out. Measured divergence up to
**24 f32 ULPs**.

### ITK side — the coefficient/scratch type is `double` for a `Float32` image
- `itkBSplineDecompositionImageFilter.h:85`
  `using CoeffType = typename itk::NumericTraits<typename TOutputImage::PixelType>::RealType;`
- `itkBSplineDecompositionImageFilter.h:133`
  `using CoefficientsVectorType = std::vector<CoeffType>;` — `m_Scratch` (`.h:175`)
  is stored in `CoeffType`.
- `itkNumericTraits.h:1349` `class NumericTraits<float>` → `:1356 using RealType = double;`
  So for a `Float32` output image `CoeffType == double`.
- The recursion operates entirely on `m_Scratch[n]` (= `double`):
  `.hxx:85` gain, `.hxx:96` causal recursion, `.hxx:104` anticausal recursion,
  `.hxx:183/199` the `CoeffType sum` in `SetInitialCausalCoefficient`.
  The **only** `float` narrowing is `.hxx:281`
  `Iter.Set(static_cast<OutputPixelType>(m_Scratch[j]))` — once, at write-out per
  axis pass (read back at `.hxx:297`).

### Port side — narrows to `f32` after every arithmetic step
- `bspline_decomposition.rs:105-110` `narrower(Float32) = |v| v as f32 as f64`.
- Applied inside the recursion at every step: `:137,:141,:145` (causal init),
  `:153` (anticausal init), `:177` (gain), `:182` (causal recursion),
  `:185,:187` (anticausal recursion).
- The docstring `:58-65` asserts "`RealType` … is `float` for a `Float32`
  image" — **factually wrong**: `NumericTraits<float>::RealType` is `double`
  (itkNumericTraits.h:1356). The test `float32_output_keeps_float32_and_rounds_per_step`
  (`:553-571`) pins exactly the wrong behavior (per-step `f32` rounding, and that
  it differs from the `f64` line).

### Measured separating input
Re-implemented both schemes (cubic, pole √3−2, tol 1e-10) on a `Float32` line
`s[i] = f32(sin(0.7·i)·100)`; ITK path = double recursion + single write-out
narrow, port path = per-step `f32`:

| N   | differing pixels | max f32-ULP diff | max abs diff |
|-----|------------------|------------------|--------------|
| 20  | 10 / 20          | 24               | 1.53e-5      |
| 50  | 30 / 50          | 24               | 1.53e-5      |
| 128 | 81 / 128         | 24               | 1.53e-5      |

24 f32 ULPs on ~half the pixels — far beyond ULP. The output is `Float32` in
both, so this difference lands directly in the stored output pixel.

### Conflict with the earlier slice-C note resolved
The slice-C "B-spline decomposition keeps MORE precision than ITK" was the
**transform-domain** coefficient computation (a different code path). This is
`BSplineDecompositionImageFilter` (the filter), where the port keeps **LESS**
precision. Both statements hold simultaneously; they are different code.

---

## 2a. `crates/sitk-filters/src/fast_marching.rs` — **REAL-BUG** (already ledgered §5.6)

**Verdict: REAL-BUG, but only for a `Float32` speed input; already documented as
a known divergence (§5.6).** SimpleITK's arrival-time output type is `double`
for every scalar input; the port keeps `Float32 → Float32`.

### SimpleITK side
- `FastMarchingImageFilter.yaml:5`
  `output_pixel_type: typename itk::NumericTraits<typename InputImageType::PixelType>::RealType`
  → `double` for a `Float32` speed input (itkNumericTraits.h:1356).
- `sitkPixelIDTypeLists.h:50` — `BasicPixelIDTypeList` includes `float`, so a
  `Float32` speed image is a reachable input.

### Port side
- `fast_marching.rs:145-147` `output_pixel_id(speed) = real_pixel_id(speed)`.
- `lib.rs:447-451` `real_pixel_id(Float32) = Float32`, everything else `Float64`.
- The march itself narrows to `f32` throughout when the output is `Float32`
  (`fast_marching.rs:427-433`, `:687`, `large_value :155-160`).

### Which inputs diverge
- **`Float32` speed input:** port output `Float32`, SimpleITK output `Float64`
  — whole output type differs and all arrival times are computed/stored in `f32`
  vs `f64`. Separating input: 3×3 unit-speed `Float32` image, seed (1,1); the
  corner is `DIAG = 1.7071067811865476`. Port stores `1.7071068_f32`
  (its own test pins this, `:869`), SimpleITK stores the `f64` value — abs diff
  ~1.9e-8 ≈ millions of `f64` ULPs. Observable.
- **Integer / `uint8` / `Float64` speed input:** port output `Float64`, SimpleITK
  `NumericTraits<T>::RealType = double` = `Float64` — **matches**.

The port already flags this at `:139-144` and tracks it as §5.6. Sister's claim
("SimpleITK output is double → port narrower") is TRUE for the `Float32`-input
case it applies to.

## 2b. `crates/sitk-filters/src/colliding_fronts.rs` — **NOT-REAL**

**Verdict: NOT-REAL.** SimpleITK's `CollidingFronts` output type is a literal
`float`, and ITK's internal marches are `float` too — the port matches on both
the output type and the internal precision. The sister's implicit claim that
`CollidingFronts` output is `double` is wrong.

### SimpleITK / ITK side
- `CollidingFrontsImageFilter.yaml:6` `output_pixel_type: float` (literal, **not**
  `RealType`). So the SimpleITK output is `float` regardless of input type.
- `itkCollidingFrontsImageFilter.h:93` `using LevelSetImageType = TOutputImage;`
  and `:100-101`
  `FastMarchingUpwindGradientImageFilterType = itk::FastMarchingUpwindGradientImageFilter<LevelSetImageType, SpeedImageType>`.
  With `TOutputImage`'s pixel type = `float`, both internal marches — their
  arrival times, gradients (`CovariantVector<float>`) and the dot product — run
  in `float`.

### Port side
- `colliding_fronts.rs:160-161` sets `narrow_to_f32: true` on both marches,
  `:289` output `PixelId::Float32`, and `:268-275` accumulates the gradient dot
  product in `f32`. Matches ITK internal-and-output `float` exactly.

Unlike `FastMarchingImageFilter` (whose type is `RealType` = `double`),
`CollidingFrontsImageFilter` hard-codes `float` upstream, so keeping `Float32`
is faithful, not narrower.

---

## Tested
- ITK BSpline `CoeffType`/scratch type for a `Float32` image is `double`
  (h:85/133/175, hxx recursion, NumericTraits<float>::RealType=double): PASS —
  port narrows per-step to `f32` instead → REAL-BUG.
- Measured BSpline port-vs-ITK output divergence on `Float32` lines
  (N=20/50/128): PASS — up to 24 f32 ULPs, ~half the pixels.
- SimpleITK FastMarching `output_pixel_type = RealType = double`, port keeps
  `Float32→Float32` for `Float32` input: PASS — REAL-BUG (ledgered §5.6);
  matches for non-`Float32` inputs.
- SimpleITK/ITK CollidingFronts output and internal marches are `float`; port
  matches: PASS — NOT-REAL.

## Failed
- (none)

## UNFIXED
- `bspline_decomposition.rs` per-step `f32` rounding (up to 24 f32 ULPs off ITK)
  — REAL-BUG confirmed, no code changed (verification round only). Root cause:
  `narrower` applied inside the recursion instead of only at write-out; docstring
  `:58-65` and test `:553-571` pin the wrong behavior. Fix owner: sitk-filters
  panel.
- `fast_marching.rs` `Float32`-input output narrowed to `Float32` vs SimpleITK
  `Float64` — REAL-BUG, already tracked as §5.6; no code changed. Fix owner:
  sitk-filters panel / ledger decision.

## Fixed
- (none — verification round, report-only)
