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

---

# Tail verify

Closes the type-narrowing sweep. The 6 headline sites (§4.124-4.129) are landed
keep-f64 per user choice. Two tasks, both re-derived firsthand from ITK
(`/home/stevek/work/ITK`) and SimpleITK; the sister panel's one integer-width
verdict (Task B) was **not** taken on faith. Report-only, no code changed.

Discriminator applied: does port `f64`/`i128` vs ITK `float`/32-bit feed a
**discrete** decision (threshold / branch / stop / round-to-int / overflow)?
YES → §4-POLICY (keep, document) or REAL-BUG (narrower → less correct); continuous
ULP-only → NOT-REAL.

## TASK A — transform / registration / displacement

### A1. `crates/sitk-filters/src/displacement_field/invert.rs` — **§4-POLICY (port WIDER, keep-f64)**

**Verdict: §4-POLICY.** ITK's `InvertDisplacementFieldImageFilter` computes the
whole iteration — composed field, scaled-norm image, `MaxErrorNorm`,
`MeanErrorNorm`, the tolerances, and the convergence stop — in `RealType`, which
for a `VectorFloat32` field is **`float`**. The port computes all of it in `f64`,
so the port is **wider** (more precise), and the extra precision feeds the
**discrete** convergence stop. Not a direction-2 bug (port is not narrower);
keep-f64 per the same policy as A9/§4.124-4.129, and extend the module doc.

- **ITK type is `float` for a `VectorFloat32` field.**
  `itkInvertDisplacementFieldImageFilter.h:131-132` `PixelType = VectorType = OutputFieldType::PixelType`,
  `:144` `using RealType = typename VectorType::ComponentType;` → `float`.
  `:145` `RealImageType = Image<RealType>` (the scaled-norm image is `float`).
  `:235-245` `m_MaxErrorToleranceThreshold`, `m_MeanErrorToleranceThreshold`,
  `m_MaxErrorNorm`, `m_MeanErrorNorm`, `m_Epsilon` are all `RealType` (`float`).
- **ITK computes and stops in that `float`.**
  `.hxx:114-116` stop = `m_MaxErrorNorm > m_MaxErrorToleranceThreshold && m_MeanErrorNorm > m_MeanErrorToleranceThreshold`
  (float comparisons). `.hxx:214-236` `localMean/localMax/scaledNorm` are `RealType`;
  `scaledNorm += sqr(displacement[d] * inverseSpacing[d])` accumulates in `float`
  (`inverseSpacing` is `VectorType`, `.hxx:213`). `.hxx:145` `m_MeanErrorNorm /= (RealType)numberOfPixels`.
  The composed field `m_ComposedField` is the `float` `DisplacementFieldType`.
- **Port computes everything in `f64`.** `invert.rs:173-174` `composed`/`scaled_norm`
  are `Vec<f64>`; `:200-220` norms in `f64`; `:176-182` stop compares `f64` norms;
  `:82-84` the module doc already flags only the `f64::MAX`-vs-`f32::MAX` sentinel.
- **Discrete? YES.** The stop iteration count depends on the norms, and the count
  determines the output field. **Separating input:** a `VectorFloat32` field whose
  `MeanErrorNorm` at some iteration sits near the `0.001` default tolerance — ITK
  accumulates `scaledNorm` over *all* pixels in a single `float` running sum
  (`.hxx:230,241`), so for a large field (e.g. 256³·3) the `float` accumulation
  error near a ~0.001 mean exceeds the gap to the threshold, and ITK stops one
  iteration earlier/later than the port's `f64` sum → a different inverse field.
  Also reachable at `MaxErrorNorm` near the `0.1` default with a residual a few
  `float` ULPs either side of `0.1`.

### A2. Other transform / registration candidates in the transform-map — **all NOT-REAL (confirmed)**

The census (`type-narrowing-transform-map.md`) found the ITK **v4** family
`double` end-to-end; re-confirmed firsthand there, none narrower. Listed so none
is left silent:

- **§C C1-C6** (`interpolator.rs`, `matrix_offset.rs`, `resample.rs`) — NOT-REAL.
  ITK `TCoordRep = TInterpolatorPrecisionType = double`; port `f64`. `is_inside`
  (C1), nearest/B-spline support index (C2/C3), point-map (C5), output saturating
  cast (C6) all `double` vs `f64`. Firsthand ITK cites in §C.
- **§B B1-B7** (`optimizer.rs`, `convergence.rs`, `lbfgs2/lbfgsb`, `scales`) —
  NOT-REAL. Value/gradient/window and the `cv <= min_cv` stop (B5) all `f64` both
  sides; ITK `TInternalComputationValueType = double`.
- **§A A1-A8** (`mattes`, `joint_histogram`, `correlation`, `ants_correlation`,
  `demons`) — NOT-REAL. ITK v4 metrics `double` (`itkMattes…v4.h:88/156/159`,
  `itkANTS…v4.h:86`); port `f64`.
- **§A A9** (mean-squares diff) — the *only* direction-1 (port wider) site; ITK
  subtracts in the image pixel type (`float`), port in `f64`. Resolved keep-f64
  (§4.124-4.129). Not direction-2.

Zero port-narrower (direction-2) sites in transform/registration.

## TASK B — `crates/sitk-filters/src/region_growing.rs` IsolatedConnected — **port i128 is MORE correct (§8), not moot**

**Verdict: port `i128` is MORE correct than ITK; a genuine (narrow) ITK
integer-overflow quirk that the port fixes. Not a bug in the port, not moot.**
The sister's "overflow on a realistic large region" framing is imprecise on the
*mechanism* — the accumulator is **not** summed over the region — but the
conclusion (ITK can overflow, port's i128 is safer) is correct.

- **What the ITK accumulator actually is.**
  `itkIsolatedConnectedImageFilter.hxx:146`
  `using AccumulateType = typename NumericTraits<InputImagePixelType>::AccumulateType;`
  used only for the **bisection threshold**: `.hxx:177-178` `lower/upper =
  (AccumulateType)m_Lower/m_Upper`, `.hxx:230,295` `guess = (upper + lower) / 2`.
  It is bounded by the pixel-value thresholds `[m_Lower, m_Upper]`, **not** a
  region sum — region size is irrelevant to overflow. (`seedIntensitySum`,
  `.hxx:211-217,276-282`, is `InputRealType` over a handful of *seed* points.)
- **`AccumulateType` widths (`itkNumericTraits.h`).**
  `unsigned char→unsigned short` (610), `short→int` (713), `unsigned short→unsigned int` (816),
  `int→long` (918) — all **widened**, cannot overflow. But
  **`unsigned int→unsigned int`** (1021, **not** widened), `long→long` (1143),
  `unsigned long→unsigned long` (1246), `long long→long long` (1668) — **not
  widened**, so `(upper + lower)` can exceed the type max before the `/2`.
- **Measured uint32 overflow.** Default `m_Upper = UINT_MAX` (`.h:47-48`). With the
  separating threshold in the upper half of the range, once the bisection raises
  `lower` past ~2.15e9: `upper=4294967295, lower=3.0e9` → ITK `unsigned int`
  `(upper+lower)` wraps to `2999999999`, `/2 = 1499999999`; the true midpoint is
  `3647483647`. ITK's guess (1.5e9) is **below** `lower`, so
  `while (lower + tol < guess)` (`.hxx:190`) is false and the search **terminates
  early with the wrong `m_IsolatedValue`**. Port `i128` gives `3647483647` and
  keeps bisecting. Small-value inputs are bit-identical between the two.
- **Discrete? YES** — `guess` is cast to the pixel type and fed to
  `ThresholdBetween` (`.hxx:195,260`), a flood-fill inclusion decision, so the
  overflow changes the segmentation and the reported `isolated_value`.
- **Separating input:** a `sitkUInt32` image, `seeds1` in a region of intensity
  ~4.0e9 and `seeds2` in an adjacent region, true separating threshold ≈ 3.5e9,
  defaults `lower=0`, `upper=UINT_MAX`, `find_upper_threshold`. ITK overflows once
  `lower` climbs past ~2.15e9 and returns a wrong isolated value / prematurely
  terminated bisection; the port converges to ≈3.5e9. `unsigned long`/`long long`
  overflow only at astronomically large magnitudes; `int32→long` and the small
  types are safe on LP64.
- **Port doc is accurate** (`region_growing.rs:87-96`): it names the exact quirk
  (`uint32_t` AccumulateType is un-widened `unsigned int`, "could theoretically
  overflow near UINT_MAX") and chooses `i128` deliberately — consistent with the
  fix-reproduced-ITK-bugs policy (correctness over bit-mirroring).

## Tested
- ITK `InvertDisplacementField` `RealType = VectorType::ComponentType = float` for
  a `VectorFloat32` field; norms/tolerances/stop all `float` (h:144/235-245,
  hxx:114-116/214-246): PASS — port `f64` is WIDER, feeds discrete stop → §4-POLICY.
- Transform/registration §A/§B/§C direction-2 candidates: PASS — all NOT-REAL
  (ITK v4 `double`, port `f64`); A9 is the lone direction-1, resolved keep-f64.
- ITK IsolatedConnected accumulator = bisection midpoint in
  `NumericTraits<pixel>::AccumulateType`, not a region sum (hxx:146,230,295): PASS.
- `AccumulateType` non-widened for `unsigned int`/`long`/`unsigned long`/`long long`
  (itkNumericTraits.h:1021/1143/1246/1668): PASS.
- Measured uint32 bisection: ITK wraps to guess 1.5e9 < lower and stops early;
  port i128 gives true 3.65e9: PASS — port MORE correct (§8), not moot, reachable
  via default `upper=UINT_MAX`. Small-value parity bit-identical: PASS.

## Failed
- (none)

## UNFIXED
- `displacement_field/invert.rs` — port computes the inversion in `f64` where ITK
  uses `float` (`RealType=ComponentType`) for a `VectorFloat32` field; feeds the
  discrete convergence stop. §4-POLICY keep-f64 (per §4.124-4.129); module doc
  currently notes only the sentinel, not the full-iteration float-vs-f64
  divergence. No code changed (verification round). Owner: sitk-filters panel /
  doc.
- `region_growing.rs` IsolatedConnected — port `i128` intentionally diverges from
  ITK's overflow-prone `AccumulateType` for wide integer types; MORE correct (§8),
  no fix owed. Recorded for the ledger, not a defect.

## Fixed
- (none — verification round, report-only)
