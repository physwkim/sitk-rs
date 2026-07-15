# Type-narrowing census — sitk-core + sitk-filters

Report-only sweep (build the both-ends map; verify+fix later, like the
boundary/convergence/coordinate sweeps). One row per site where the port's
intermediate compute type differs from ITK's for an intermediate value.
References read firsthand: ITK `/home/stevek/work/ITK`, SimpleITK
`/home/stevek/work/SimpleITK`.

## Candidate tally (verify+fix later)

Amplifying discrete-decision candidates, most-load-bearing first:

1. **N4 §4.122** (`n4_bias_field.rs`) — f64 vs hardcoded-float convergence CoV
   flips the stop iteration. Already documented; reachable for every input.
2. **Histogram-threshold family** (`threshold.rs`/`histogram.rs`; Otsu, Huang,
   Triangle, Intermodes, IsoData, Kittler, Li, MaxEntropy, Moments, Renyi,
   Shanbhag, Yen) — f64 bin edges vs ITK's hardcoded-float `Histogram::Initialize`
   flips the selected threshold bin (reclassifies every pixel on one side).
3. **`bspline_decomposition.rs`** — **direction-2 real bug**: port rounds the IIR
   recursion to f32 every step for Float32 images on the false premise
   `NumericTraits<Float32>::RealType==float`; ITK keeps full double.
4. **Canny / zero-crossing edge** (`canny.rs`, `edge.rs`) — f64 vs float image
   buffers flip zero-crossings and hysteresis seeds that *propagate* into edge
   chains. Float32-reachable.
5. **`min_max_curvature_flow.rs`** — f64 vs float min/max gate flips the update
   sign, amplified over iterations. Float32-reachable.
6. **Sparse-field segmentation level sets** (`level_set/function.rs`,
   `level_set/sparse_field.rs`) — **UNDOCUMENTED**: the whole PDE evolution runs
   in f64 where ITK's `ScalarValueType`/`ValueType` is float, feeding RMS
   convergence stop (`sparse_field.rs:227`, the direct §4.122 analog), narrow-band
   membership (`:494/:512`), and promote-ordering (`:507/:521`). Float32-gated.
7. **`chan_vese.rs` / `reinitialize_level_set.rs`** — f64 vs float `phi < 0` sign,
   iterated. Float32-gated (chan_vese documented, reinitialize weak).
8. **`vector_connected_component.rs`** — ITK rounds each dot-product term to float
   before summing; port keeps f64 → flips the union-find join predicate.
9. **Distance maps** (`distance.rs`: IsoContour narrowband seed, FastChamfer
   relaxation) — f64 vs float stored/re-read → flips which front wins
   (`ApproximateSignedDistanceMap`, Float32 output).
10. **`sharpening.rs`** unsharp-mask 3-way branch — f64 vs float `diff` flips
    sharpen-up/down/unchanged. Float32-reachable.
11. **Fast-marching** (`fast_marching.rs`, `fast_marching_upwind_gradient.rs`) —
    **direction-2**: the port narrows arrival times to **f32** (keeping Float32
    output per §5.6) where SimpleITK forces double; flips heap order / stopping /
    target break.
12. **`colliding_fronts.rs`** — **direction-2**: the port accumulates the gradient
    dot product in **f32** where ITK's `CovariantVector::operator*` accumulates in
    **double**; flips the negative-region membership. Port comment's "matches ITK
    float" is wrong.
13. **N4 histogram binning** (`n4_bias_field.rs:395-396`) — new §4.122 neighbor:
    f64 vs float `floor` of the per-pixel bin index cascades through every
    iteration.
14. **64-bit-integer `to_f64_vec` seam** (`change_label.rs`, `clamp.rs`, geometry/
    grid/join copy-faces) — **direction-2**: port narrower for `u64`/`i64` pixels
    `> 2^53`; gated on SimpleITK instantiating `UInt64`/`Int64`.
15. **`region_growing.rs` IsolatedConnected** — port i128 bisection vs ITK's
    32-bit `AccumulateType` (UInt32) → port more correct, exposes ITK overflow.
16. **`watershed_classic.rs`** — f64 vs `ScalarType`=PixelType gates; **gated** to
    standalone `watershed()` on non-double input (the SimpleITK-reachable
    `isolated_watershed` runs on a double gradient and matches).
17. **`displacement_field/invert.rs`** — f64 vs float convergence stop, VectorFloat32
    field only.

Direction-2 (port strictly **narrower**, real precision loss): `bspline_decomposition.rs`
(#3), fast-marching (#11), `colliding_fronts.rs` (#12), the u64/i64 `to_f64_vec`
seam (#14). Every other candidate is direction-1 (port more precise, extra
precision flips a discrete decision). Deliberate-and-correct f32 narrowings that
faithfully match ITK's genuine float — **not bugs**: `rank.rs`, `slic.rs`,
`objectness.rs`, `stochastic_fractal_dimension.rs`, `label.rs` histogram bins,
`label_set_morphology.rs` (the §2.57(d) fix). NOTE the trap: `colliding_fronts.rs`
and `fast_marching.rs` *look* like this deliberate-f32-match class but are
**bugs** — they narrowed below ITK's actual `double` (`AccumulateType`/`RealType`),
having been written against the false `NumericTraits<float>` premise.

## The load-bearing correction: `NumericTraits<float>::RealType == double`

Verified firsthand at `itkNumericTraits.h:1356` (and again independently by two
census panels): **ITK's `RealType` is `double` for `float` input**, not `float`.
`ScalarRealType`, `AccumulateType`, and every `NumericTraits<Float32>::RealType`
derivation are `double`. Consequence for the whole census:

- A port site that computes in `f64` where ITK computes in
  `NumericTraits<PixelType>::RealType` **matches ITK exactly** — it is *not* a
  precision superset. The recurring "§4.1: f64 where ITK pins float" premise is
  **false wherever ITK's type is `RealType`**. Those rows are faithful matches,
  not deliberate divergences.
- ITK genuinely uses `float` only where it uses the **raw image buffer type**
  (`OutputImagePixelType` / `PixelType` / an explicitly `Image<float>` internal
  stage) or a **hardcoded `float`** — not `RealType`. Those are the real
  direction-(1) sites, and the ones that feed a discrete decision are the
  candidates below.

## The discriminator, refined

For each site where the types differ, the question is not "does the precision
difference feed *a* discrete decision" but "does it feed an **amplifying**
discrete decision":

- **CANDIDATE** — the flip changes a decision whose consequence exceeds the
  ±1 LSB local narrowing: a **threshold pick** (reclassifies many pixels), a
  **convergence/iteration count** (changes every output pixel, cf. N4 §4.122),
  an **argmin/argmax label**, a **propagating** decision (hysteresis edge
  growth, iterated flow-sign), or a **map-key / index identity** (u64 pixel
  aliasing). Give the separating input.
- **§4-keep** — the only discrete consequence is the inherent final
  round-to-output-pixel cast of the very value being computed (±1 LSB, no
  amplification). This is the documented §4.1 superset, and it is faithful by
  design. A LARGE divergence magnitude (e.g. float catastrophic cancellation in
  a one-pass variance) is still §4-keep if it only shifts a reported continuous
  value.

Direction-(2) — port **narrower** than ITK — is a real bug regardless of the
discrete/continuous split, because the port loses precision ITK keeps.

---

## CANDIDATE rows (precision difference feeds an amplifying discrete decision)

### A. Histogram-threshold family — f64 bin edges vs ITK's hardcoded-`float` `Initialize`

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| `histogram.rs:275-298` (`over_range`), `:316-359` (`from_bounds`), `:228-298` (`from_values`/`from_fixed_range`) — bin edges `lower + j*(upper-lower)/bins` in **f64** | `itkHistogram.hxx:216-235` `Initialize`: `float interval = (float(upper)-float(lower))/float(size)`, `SetBinMin(j, (Meas)(lower + float(j)*interval))`; reached by `itkImageToHistogramFilter.hxx:178,237`; bin assignment `GetIndex` `.hxx:243-296` (binary search on the float boundaries) | f64 bin edges + f64 `bin_index` compare | **float** interval and boundaries (hardcoded, independent of `RealType`), widened to the histogram `MeasurementType` after the float arithmetic | Discrete: the selected threshold is a bin value; a differently-binned near-boundary sample can move the argmax/argmin bin → adjacent threshold pick, reclassifying every pixel on one side | **CANDIDATE** | Input whose `(max−min)/bins` is not float-exact (e.g. 8-bit data spanning `[0,255]` with 100 bins → interval `2.55`: `float(2.55)=2.549999952…` vs `f64` `2.55`), containing a pixel value sitting between the float-rounded and f64-rounded decisive bin boundary. Port bins it into class A, ITK into class B → the between-class-variance argmax (Otsu) / entropy-extremum (Huang, MaxEntropy, …) lands one bin away. Construct a concrete pin in the verify phase. |

Applies to the whole family that bins through `ImageToHistogramFilter` /
`Histogram::Initialize`: **Otsu, Huang, Triangle, Intermodes, IsoData,
KittlerIllingworth, Li, MaximumEntropy, Moments, RenyiEntropy, Shanbhag, Yen**
(`threshold.rs`, `histogram.rs`). The internal criterion accumulations
themselves are **not** the divergence — every calculator computes its
entropy/variance in `double` (`itkOtsuMultipleThresholdsCalculator`
`MeanType=VarianceType=NumericTraits<MeasurementType>::RealType=double`; Huang/
Triangle/… cast frequency and measurement to explicit `double`), matching the
port's f64. The sole divergence is the **f64-vs-float bin geometry**, which the
port already documents at `histogram.rs:313-315` as a deliberate deviation but
**without noting that it feeds the discrete threshold pick**. Reclassifying that
documented deviation as a candidate is this census's contribution here.

RETRACTED sub-claim (recorded so it is not re-derived): I initially flagged
Otsu's between-class variance as accumulating in `float` for float input. That
was wrong — it rests on `NumericTraits<float>::RealType`, which is `double`. The
variance is `double` on both sides. Only the bin geometry diverges.

### B. `bspline_decomposition.rs` — port NARROWER than ITK (direction 2, real bug)

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| `bspline_decomposition.rs:104-108` (`narrower(Float32) = \|v\| v as f32 as f64`, applied after every recursion step at `:137,141,145,153,176-177,182,187`); premise stated in module doc `:58-59` | `itkBSplineDecompositionImageFilter.h:85` `CoeffType = NumericTraits<OutputPixelType>::RealType` = **double** for Float32; recursion `.hxx:85-104,183-207` all in `CoeffType`; the **only** narrowing is `static_cast<OutputPixelType>` at write-out `.hxx:281` | **f32** rounding after **every** causal/anticausal IIR step (Float32 image) | **double** throughout, narrowed to float once at write-out | Continuous coefficient output, but the per-step f32 rounding compounds through the IIR recursion — far beyond ULP — and downstream feeds interpolation (and the final integer-output cast when resampled) | **CANDIDATE (direction 2)** | Any multi-sample Float32 line (e.g. the 19-sample `f19` in the port's own `accelerated_and_full_initializations_agree`). The port's module doc `:58-65` and test `float32_output_keeps_float32_and_rounds_per_step` (`:552-571`) are built on the **false** premise `NumericTraits<Float32>::RealType == float`; it is `double`, so ITK does not round per step. |

Contrast with adaptive-histogram (§4-keep row below): the *identical-looking*
doc premise "RealType is float for Float32" is **false** here (bspline derives
`CoeffType` from `NumericTraits` → double, so the port is a narrowing bug) but
**true** for adaptive-histogram (its functor hardcodes `using RealType = float`,
so the port's f64 is a genuine superset).

### C. Canny / zero-crossing edge family — f64 vs ITK's `OutputImagePixelType`=float image buffer

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| `canny.rs:205` (`grad_mag=sqrt`, f64 kept), `:207` (`if deriv_pos <= 0.0`), `:140` (2nd-dir-deriv field), `:253-262` (zero-crossing sign + `\|a\|<\|b\|` tie), `:354` (`edge_strength[seed] <= upper`), `:376` (`> lower`) | Entire pipeline in `OutputImageType`=TOutputImage=**float** for a Float32 image: `itkCannyEdgeDetectionImageFilter.h:97,214-215` (Gaussian `<Input,Output>`, Multiply `Output×Output→Output`); thresholds `m_UpperThreshold`/`m_LowerThreshold` are `OutputImagePixelType` (`.h:252-253`, with ITK's own comment `// should be float here?`); NMS/hysteresis `.hxx:285-300,350-360,435-448`; zero-crossing `itkZeroCrossingImageFilter.hxx:137-152` (`InputImagePixelType`=float) | f64 throughout | **float** image buffers + float thresholds | Discrete + **propagating**: zero-crossing sign picks the marked pixel; hysteresis `> m_UpperThreshold` seeds an edge that `FollowEdge` grows through `> m_LowerThreshold` — a single flipped seed grows or kills a whole edge chain | **CANDIDATE** | Float32 input; a pixel whose `gradMag·zeroCross` sits within one float-ULP of `m_UpperThreshold` (or a near-symmetric zero-crossing where `\|a\|` vs `\|b\|` ties): float rounds it below/above the threshold opposite to f64, adding or removing an edge chain. Spot-verified firsthand that ITK's buffers and thresholds are `OutputImagePixelType`. |
| `edge.rs:79-86` (`ZeroCrossingBasedEdgeDetection`: gaussian→laplacian→zero_crossing, all f64) | `itkZeroCrossingBasedEdgeDetectionImageFilter.hxx:47-49` + `itkZeroCrossingImageFilter.hxx:137-152` (float image buffer) | f64 | float | Final zero-crossing sign / `\|·\|` tie on the smoothed Laplacian (binary edge output) | **CANDIDATE** | Float32 near-symmetric smoothed-Laplacian value flips which side is marked. Same mechanism as canny, spot-verified via the shared `ZeroCrossing` type. |

### D. `min_max_curvature_flow.rs` — f64 vs ITK's `PixelType`=float neighborhood/gate

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| `min_max_curvature_flow.rs:449` (MinMax gate `avg_value < threshold`), `:457` (Binary gate), fed by `:441-445,179` (`avg_value` ball sum; weights `/= n as f64`) | `itkMinMaxCurvatureFlowFunction.hxx:395` / `itkBinaryMinMaxCurvatureFlowFunction.hxx:44`; weights `static_cast<PixelType>(1/n)` `.hxx:112`; inner-product returns **float** `itkNeighborhoodInnerProduct.hxx:48` | f64 | **float** (`PixelType` neighborhood arithmetic) | Discrete + **iterated**: the `avg < threshold` gate selects `max(update,0)` vs `min(update,0)`; a flipped sign compounds over flow iterations | **CANDIDATE** | Float32 smooth-gradient patch where `avg_value` and `threshold` agree to ~7 significant figures but straddle in f64 (e.g. `1/13` weight as float vs f64, ball sum narrowed to float) → opposite-sign update, amplified iteration over iteration. |
| `min_max_curvature_flow.rs:240,215,233-238` (`\|dot_product\| < 0.262` generic-D neighbor scan) | `itkMinMaxCurvatureFlowFunction.hxx:183` (`PixelType` throughout) | f64 | float | Discrete: set membership — which neighbors enter the perpendicular average | **CANDIDATE** | 4-D image, neighbor cosine ≈ 0.262: f64 `0.26199` (in) vs float `0.26200` (out) → different averaging set → gate flip. |
| `min_max_curvature_flow.rs:277-279` (2-D `itk_round(r±grad)`), `:292-342` (3-D `theta/phi` 4-ring `itk_round`) | `itkMinMaxCurvatureFlowFunction.hxx:254-264,284-373` (pixel diff + threshold in `float`) | f64 pixel-diff | float pixel-diff | Discrete: rescaled gradient on a `round(x+0.5)` half-boundary picks a different lattice pixel to sample | **CANDIDATE (narrow)** | Rescaled gradient landing on a half-integer lattice boundary picks the adjacent stencil pixel. |
| `min_max_curvature_flow.rs:437-438,468` (`update`; `if update==0.0`; `+= time_step*g`) | `itkCurvatureFlowFunction.hxx:135` (`static_cast<PixelType>(update)`); `itkMinMaxCurvatureFlowFunction.hxx:383` | f64 (un-narrowed, carried across iterations) | float on return each iteration | Discrete: early-skip `update==0.0`; and the port carries `buf` in f64 across iterations vs ITK's per-iteration float re-quantization, feeding every later gate | **CANDIDATE (marginal)** | True-double update nonzero but rounds to `0.0f32` (≲2⁻¹⁵⁰): ITK skips, port applies. Cross-iteration f64-vs-float buffer feeds D-row gates. |

### E. 64-bit integer pixels through the `to_f64_vec`/`from_f64` seam — port NARROWER (direction 2)

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| Root primitive `sitk-core/pixel.rs:432` (`as_f64 = self as f64`), `:435` (`v as T`); `image.rs:156-158,886-888` (`to_f64_vec`/`from_f64`) | n/a (ITK stays in native `PixelType` for these ops) | **f64** (rounds `\|value\| > 2^53`) | native `u64`/`i64` (exact) | Feeds every discrete op below | **Root seam** | Any `UInt64`/`Int64` pixel with magnitude `> 2^53`. Exact for u8..u32/i8..i32/f32/f64. |
| `change_label.rs:45,49,52,55` | `itkChangeLabelImageFilter.h:95` (`m_ChangeMap.find(A)`), `:139` (`std::map<InputPixelType,…>`) | f64 key | native `InputPixelType` | Discrete: label map-key identity | **CANDIDATE** | u64 pixel `9007199254740993`, map key `9007199254740992`: ITK leaves the pixel unchanged; port conflates both to the same f64 bits and remaps. |
| `clamp.rs:78` (input `to_f64_vec`), `:87` (in-range re-cast) | `itkClampImageFilter.h:105` (`static_cast<OutputType>(A)`, native `A`; clamp compare `static_cast<double>(A)` `:93`) | f64 (pre-rounded input) | native input pixel | Discrete: saturating cast to integer output | **CANDIDATE** | UInt64 input `9007199254740993`, full-range clamp to UInt64: ITK emits `…993`, port emits `…992`. (The clamp comparison itself matches — ITK also compares in double.) |
| `geometry.rs:113/127,276/292,507/519,565/577` (crop/ROI/extract/flip/permute), `grid_utility.rs:64-65/81,140/145/159,238/254` (checker/paste/tile), `join_series.rs:149,154` | ITK `RegionOfInterest`/`Extract`/`Flip`/`PermuteAxes`/`CheckerBoard`/`Paste`/`Tile`/`JoinSeries` `.hxx` via `ImageAlgorithm::Copy` (native pixel copy) — **ITK .hxx line UNVERIFIED** | f64 round-trip | native pixel copy | round-to-representable (`from_f64` int cast) | **CANDIDATE — copy-face** | u64/i64 pixel `> 2^53`. Pure data-movement corruption, same seam. Reachability + exact ITK lines to confirm in verify phase. |

Reachability caveat for every u64/i64 row: depends on whether SimpleITK
instantiates `sitkUInt64`/`sitkInt64` in that filter's pixel-ID list. Triage
before fixing.

---

## §4-keep rows (more precise, continuous output — faithful/deliberate)

| Port site | ITK site | Port type | ITK type | Note |
|---|---|---|---|---|
| `adaptive_histogram_equalization.rs:37-40,184-185` (`cumulative_function`, `u`/`v` in f64) | `itkAdaptiveEqualizationHistogram.h:42` (`using RealType = float` **hardcoded**), `:85,88,145-152` (`u`,`v`,`CumulativeFunction`,`std::pow` in float; `sum`/`iscale`/`ikernel` in double `:76,83,89`) | f64 | **float** for u/v/CumulativeFunction (genuine — hardcoded, not `RealType`) | Continuous per-pixel reconstruction; only the final output cast is discrete (±1 LSB, non-amplifying). Port's f64 is a legitimate superset. The port doc `:90-97` already declares this intentional. |
| `recursive_gaussian.rs:136,184-201` (buffer f64 across all axes, narrowed once at `:379/:389`) | `itkSmoothingRecursiveGaussianImageFilter.h:78-88` (`InternalRealType=float`); inter-pass cast `itkRecursiveSeparableImageFilter.hxx:277` | f64 held across all axes | **float** image re-quantized between every axis pass | Continuous output, never thresholded. NOT pure ULP: ITK re-rounds to float after each axis (2 extra roundings for 3-D); port rounds once. Port doc `:316-325` understates this as a "summation order" difference — it is an inter-axis re-quantization difference. |
| `gradient.rs:162-184` (grad-mag), `:295-304` (derivative), `:336-354` (laplacian), `:434-479` (sobel), `:747-762` (gradient), `:518-580` (grad-mag-recursive-gaussian), `:597-654` (LoG-recursive-gaussian), `:830-891` (gradient-recursive-gaussian) | `itkGradientMagnitudeImageFilter.hxx:104,165-171`, `itkDerivativeImageFilter.hxx:86`, `itkLaplacianImageFilter.hxx:53,97`, `itkSobelEdgeDetectionImageFilter.hxx:46,99-137`, `itkGradientImageFilter.hxx:99,167`, `itkGradientMagnitudeRecursiveGaussianImageFilter.h:76` (`InternalRealType=float` even for double input), `itkLaplacianRecursiveGaussianImageFilter`, `itkGradientRecursiveGaussianImageFilter.h:79` | f64 | **float** (Float32 input; the recursive-gaussian ones are `InternalRealType=float` for all inputs) | Continuous magnitude/vector/LoG outputs. ULP shift; standalone gradient's Laplacian into `edge.rs` is covered by the edge candidate row. |
| `sharpening.rs:118-131` (`unsharp_mask`: blurred `s`, `diff=v-s`, branch, `v+(diff∓threshold)*amount`) | `itkUnsharpMaskImageFilter.h:107-108,209-217` (`TInternalPrecision`; SimpleITK `UnsharpMaskImageFilter.yaml:7` sets it to `InputImageType::PixelType`=float for Float32) | f64 | float (Float32) | **The 3-way branch `if diff > threshold` (`:127-130`) is itself a CANDIDATE** (Float32, threshold=0, locally-symmetric neighborhood: float `diff==0.0f` → ITK unchanged, but f64 residual `+ε` → port sharpens); the continuous `v+(diff∓threshold)*amount` arithmetic once the branch is chosen is §4-keep. Listed here for the arithmetic; the branch is a candidate. |
| `anisotropic_diffusion.rs:161,203-239,290-346` (avg-grad reduction; gradient/curvature updates; `if speed>0.0`) | `itkScalarAnisotropicDiffusionFunction.hxx:80-110`, `itkGradientNDAnisotropicDiffusionFunction.hxx:87-115`, `itkCurvatureNDAnisotropicDiffusionFunction.hxx:84-148` | f64 | float pixel-diffs; `K`/`speed` accumulate in double | Branches (`m_K != 0.0` exact-zero can't flip; `speed > 0` computed in **double on both sides**) do not diverge; only continuous updates shift by ULP. |
| `expand.rs:162-164` (`out_spacing = in_spacing / factor as f64`), `shrink.rs:76` (`out_size = in_size / f`) | `itkExpandImageFilter.hxx:224` (`/float(factor)`), `itkShrinkImageFilter.hxx:262-263` (`floor(double/double)`) | f64 / usize | float divisor / floor of double | `float(factor)==f64(factor)` exactly for factor ≤ 2²⁴; output pixel count identical for all image sizes < 2⁵³. Metadata/count only, provably identical discrete result. |
| `math.rs:114-116` (`Round`) | `itkRoundImageFilter.h` → `RoundHalfIntegerUp<TOut,TIn>` computes in `TInput` | f64 | f32 for Float32 input | Port WIDER (ledger §4.35); half-integer f32 boundary rounds identically or better. |
| `sitk-core/label_map.rs:267` (`Size()` sums `l.length as u64`) | `itkLabelObject.hxx:217` (`int size = 0`) | u64 | **int (32-bit signed)** | Port WIDER/more correct (direction 1); ITK's `int` overflows negative for a label object exceeding 2³¹ pixels. |

### StatisticsImageFilter and the reduction infrastructure — FAITHFUL (both `double`)

The explicitly-named target resolves to a **match**, not a divergence:

- `itkStatisticsImageFilter`: `RealType = NumericTraits<PixelType>::RealType` =
  **double** for every scalar input (float included); sum/sumOfSquares in
  `CompensatedSummation<double>` (`.h:162-163`, `.hxx:103-104,117-122`). The
  port's compensated-f64 accumulator (`sitk-core/compensated.rs`,
  `sitk-filters/lib.rs:824-885`) matches bit-for-bit. Even the naive one-pass
  variance `(sumOfSquares − sum²/n)/(n−1)` is `double` on both sides — its
  magnitude of loss (catastrophic cancellation) can be large, but the outputs
  are reported continuous values with no discrete amplification, so this is
  faithful, not a candidate.
- ITK combines per-thread compensated sums with a plain `m_ThreadSum += sum`
  (`.hxx:130`), so ITK's own multithreaded statistics are **order-dependent**;
  the port's `map_rows_fold_in_order` fixed-order fold is both reproducible and
  matches the serial result — strictly better, still faithful.
- `min`/`max` are stored in **`PixelType`** in ITK (`.hxx:80,84`;
  `itkMinimumMaximumImageFilter.h:139-140`), where the port widens through
  `as_f64` (`parallel.rs:853-861`). For `u64`/`i64` pixels `> 2^53` the reported
  min/max is f64-rounded; **latent** because monotonic widening keeps the
  *selection* correct and SimpleITK's Statistics measurements are themselves
  `double` (so cast-of-min == min-of-casts). Shares the pixel.rs:432 seam.
- `parallel.rs` `map_rows_fold_in_order` (caller-owned accumulator, serial index
  order), `bin_counts` (exact u64), `compensated.rs` (`CompensatedSum{f64}` =
  `itk::CompensatedSummation<double>`), `ops.rs` (float ops in the pixel type,
  no f64 intermediate), `fused.rs`, `boundary.rs` (integer index math),
  `matrix.rs`/`coord.rs` (f64 = vnl double): **no type-narrowing divergence**.

### FFT / convolution / correlation / spline — mostly FAITHFUL (`double`)

- `fourier.rs` standalone FFT filters: f64 where PocketFFT runs on the image
  component type (float for Float32) — **§4-keep** (known seed §4.1), continuous.
- `convolution.rs`/`deconvolution.rs`: ITK `TInternalPrecision = double` by
  default (`itkFFTConvolutionImageFilter.h:56,95-101`,
  `itkInverseDeconvolutionImageFilter.h:58`) → port f64 **matches**; §4.1 does
  not apply here.
- `normalized_correlation.rs`/`convolution.rs` accumulators:
  `OutputPixelRealType`/`ComputingPixelType = NumericTraits<Out>::RealType` =
  **double** (`itkNormalizedCorrelationImageFilter.hxx:97`,
  `itkNeighborhoodOperatorImageFilter.h:77`) → port f64 **matches**.
- `complex.rs`: **§4.24 exception intact** — computes in the component type
  (f32 for ComplexFloat32) matching ITK's `std::atan2`/`std::polar` in float.
- `fft_correlation.rs:288-295` `CalculatePrecisionTolerance`: for a Float32
  input the port picks `p=23` (2⁻²³) where SimpleITK's always-Float64 output
  makes ITK pick `p=52`; the `denominator < tolerance` **score-zeroing**
  (`:496`) then differs. Flagged as a downstream consequence of the already
  tracked §5.6 `real_pixel_id` Float32-output divergence, not a fresh narrowing.
- Doc inaccuracy (not a behavioral divergence): `fft_correlation.rs:90-93`
  claims ITK squares in the input pixel type with int32 overflow; the current
  `itkSquareImageFilter` `Functor::Square` promotes to `RealType` before
  squaring, so no such overflow exists.

### Misc filters — verified CLEAN (port matches ITK)

`kmeans.rs` (f64=double), `slic.rs` (clusters f64=`ClusterComponentType`/double,
distance f32=`TDistancePixel`/float — §4.1 exception **correctly** narrower to
match ITK's float argmin), `patch_based_denoising.rs` (f64=`RealValueType`,
weights f32=`Image<float>`), `stochastic_fractal_dimension.rs` (f32=`RealType`
hardcoded float — §4 exception correct), `noise.rs`/`random.rs` (MT19937 u32
bit-exact, draws f64=double), `intensity.rs` (ShiftScale/Normalize/Sigmoid/
Windowing f64=`RealType`/double), `rescale_intensity` (`lib.rs:791`; f64 =
`NumericTraits<TInput>::RealType`=double — the scale/shift factoring order
differs but both in double, a fold-order concern outside this type census),
`denoise.rs`, `noise_estimate.rs`, `objectness.rs` (reproduces ITK's float
Hessian via `narrow_f32`), `coherence_enhancing_diffusion.rs` (ITK `TScalar=
double` for all inputs), `sources.rs`, `slice.rs`, `deriche.rs`,
`neighborhood.rs`, `smoothing.rs`, `label_fusion.rs`, `label_map*.rs`,
`label_to_rgb.rs`, `scalar_to_rgb_colormap.rs`: **zero type-narrowing divergence**.

### Out-of-census behavioral divergences flagged during verification (not type-narrowing)

- **SLIC empty-cluster handling** (`slic.rs:358-361`): divides accumulator by
  count unconditionally → an empty cluster becomes NaN and stays empty forever;
  current ITK (`itkSLICImageFilter.hxx:647-654`) guards `if (count>0) … else keep
  previous centroid`. Feeds discrete label assignment. The port doc `:94-98`
  describes an older NaN-producing ITK that no longer matches the local
  reference. Value/behavior divergence, not a compute-type one.
- **`fft_shift.rs`** routes all pixel types through `to_f64_vec`/`from_f64` — the
  same u64/i64 seam for a value-copy filter that ITK's `CyclicShiftImageFilter`
  does natively.
- **grid_source / itkGridImageSource** and **itkSLIC** algorithm versions on
  disk differ from what a couple of port module docs cite (`§1.6` wrap bug,
  NaN empty-cluster) — algorithm-version drift, not a type difference.

---

### F. Distance maps / vector-CC / isolated-connected / watershed / label-set morphology

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| `distance.rs:697-702` (`val_new0.abs() < out[center].abs()` narrowband seed pick) | `itkIsoContourDistanceImageFilter.hxx:363-372` (computes in **double** but stores/re-reads via `SetNext(static_cast<float>(valNew0))`) | f64 | double compute **stored/re-read as float** output pixel | Discrete: the `< existing` min pick choosing the narrowband seed | **CANDIDATE** (Float32 out; feeds `ApproximateSignedDistanceMap`) | Pixel reachable from two iso-crossings: ITK compares the 2nd candidate against the float-rounded first write, the port against the full-f64 first write; when rounding flips `<`, a different seed distance survives. |
| `distance.rs:836-863` (chamfer sweep: `center_value >= maximum_distance`; `val < vals[q]`) | `itkFastChamferDistanceImageFilter.hxx:124-166`; types `.h:108,117,160,163` (`WeightsType=FixedArray<float>`, `float m_MaximumDistance`, float PixelType) | f64 (all sweep arithmetic) | float (weights, running distance, saturation bound, relaxation compare) | Discrete: saturation skip `>= maximum_distance` and Gauss–Seidel relaxation min/max | **CANDIDATE** (`ApproximateSignedDistanceMap`, float output) | Accumulated chamfer distance float vs f64: where the float relaxation lands exactly on the incumbent neighbor, the port's f64 value is strictly less/greater and overwrites differently, changing which front wins. |
| `vector_connected_component.rs:122-126` (`dot = Σ x.as_f64()*y.as_f64()`, per-term product in f64) | `itkVectorConnectedComponentImageFilter.h:79-84` (`dotProduct += a[i]*b[i]` per-term in `ValueType`=float; `static_cast<ValueType>(1-\|dot\|) <= threshold`) | f64 per-term product | **float** per-term product, double accumulate | Discrete: union-find join predicate `1-\|dot\| <= threshold` | **CANDIDATE** (Float32-component vectors) | ITK rounds each `a[i]*b[i]` to float before summing; port keeps f64. Float32 unit-ish 2-vectors with `distance_threshold` within ~1e-7 of the `1-\|dot\|` boundary flip join↔split. The port comment `:120-121` notes only the accumulator is double and **misses** the per-term float rounding. |
| `region_growing.rs:698` (`(a+b)/2` bisection midpoint in **i128**; add `:682`, sub `:695`) | `itkIsolatedConnectedImageFilter.hxx:230,295` (`guess=(upper+lower)/2` in `AccumulateType`) | i128 | `NumericTraits<T>::AccumulateType` — **32-bit `unsigned int` for UInt32** (`itkNumericTraits.h:1028`, confirmed firsthand); 64-bit for Int64/UInt64 | Discrete: midpoint drives `guess → ThresholdBetween(guess)` flood → converged `isolated_value`/mask | **CANDIDATE (direction: port WIDER/more correct; UInt32 & 64-bit-magnitude inputs)** | UInt32 image, `lower=0`, `upper=4_000_000_000`, `find_upper_threshold=true`. Once `lo≈2e9,hi=4e9`: ITK `(2e9+4e9) mod 2³² ≈1.7e9`, `/2≈8.5e8`; port true `3e9`. Divergent guesses → different segmentation. Int64/UInt64 diverge the same way near ~9e18. Port exposes an ITK integer-overflow quirk. |
| `watershed_classic.rs:1054` (`threshold = flood_level*maximum_depth`) → `:1077` (`if saliency < threshold`); depth `max-min` `:928` | `itkWatershedSegmentTreeGenerator.hxx:148` (`static_cast<ScalarType>(m_FloodLevel*GetMaximumDepth())`), compare `:177` | f64 | `ScalarType = InputImageType::PixelType` (`itkWatershedImageFilter.h:176`) — integer for int input, float for Float32 | Discrete: strict `<` decides which merges enter the segment tree | **CANDIDATE — gated (standalone `watershed()` on non-double input only)** | UInt8 `[0,2,2,1,5]`, level=0.25, depth=5: ITK `static_cast<uchar>(1.25)=1`, saliency 1 → `1<1` false → empty tree; port `1<1.25` true → 1-entry tree. Also Float32 near-tie. **Gate:** the SimpleITK-reachable `isolated_watershed` runs on a Float64 gradient where `ScalarType==double` and the port matches; bites only via a standalone `watershed()` on non-double input. |
| `watershed_classic.rs:458` (`e.height - segment.min > maximum_saliency`, edge prune); `:902/910` (`threshold_value=threshold*(max-min)+min`; `v < threshold_value`, float branch keeps un-rounded) | `itkWatershedSegmenter` `PruneEdgeLists`/`Threshold` (`static_cast<InputPixelType>` of the level) over `ScalarType` | f64 | float (`ScalarType`, Float32 input) | Discrete: first-pruned-edge position / flooring test `v < threshold_value` | **CANDIDATE — gated (Float32 input, standalone watershed)** | f32-rounded `height-min` / threshold crosses the limit at a different edge/voxel than the exact f64 value. Same non-double gate as the row above. |
| `label_set_morphology.rs:178-180` (`same_run`: `a == b` in f64; run detection `:358`, dup `:516/524`) | `itkLabelSetUtils.h:297,305,516,524` (`RealType val=labBuf[idx]; val != labBuf[idxend]`); `RealType=FloatType=float` (`itkLabelSetMorphBaseImageFilter.h:67` + `itkNumericTraits.h:619`) | f64 | float | Discrete: run-boundary detection decides erode merge vs split | **CANDIDATE — but INTENTIONAL/documented §2.57(d) fix** (port fixes an ITK narrowing bug) | Int32 labels `16777216` & `16777217` adjacent: both collapse to `16777216.0f` in ITK → merged into one run; port keeps them distinct in f64 → they separate. Not an open bug — the port is deliberately the correct side. |

§4-keep in this cluster: `watershed_classic.rs:1096/1107,1215/1219` (non-strict
`<=` saliency gates — order-equivalent for integer saliency, `s <= trunc(x) ⟺
s <= x`; inert), `attribute_morphology.rs:156,162,198` (exact widening,
sort/`==` preserved), `reconstruction.rs:219-253` (exact widening; h-minima/
maxima `v±height` f64 == ITK ShiftScale RealType=double), Danielsson norm
compare (double both sides). Deliberate-match `f32` sites (direction-preserving,
NOT bugs): `rank.rs` order-stat index (matches ITK `float m_Rank`,
`itkRankHistogram.h:140,252,306`), `label.rs` histogram bins (match ITK float),
`label_set_morphology` parabola, `watershed_classic` `AlmostEquals`.

Zero differing sites (verified): `morphology.rs` (ball membership
`Σ(x/(r+0.5))² ≤ 1` is double in ITK too), `binary_morphology.rs` (u32 counts),
`object_morphology.rs`, `morphology_reconstruction.rs`, `geodesic_morphology.rs`,
`scalar_connected_component.rs` (`\|a-b\|` double→pixel-type cast reproduced),
`regional_extrema.rs`, `watershed.rs`, `toboggan.rs`, `label.rs`
(LabelStatistics RealType=double), `contour.rs`, `contour_extractor_2d.rs`
(InputRealType=double, interpolation fraction double both sides).

## SimpleITK reachability gating (firsthand from `Code/BasicFilters/yaml`)

Whether a Float32-input candidate is reachable depends on the SimpleITK
filter's pixel-ID list and any `output_pixel_type` override. Verified firsthand:

- **Reachable at Float32** (`RealPixelIDTypeList`, output = input float, no double
  override): `CannyEdgeDetectionImageFilter`, `ZeroCrossingBasedEdgeDetection`,
  `UnsharpMaskImageFilter`, `MinMaxCurvatureFlowImageFilter`,
  `ScalarChanAndVeseDenseLevelSetImageFilter`, `BSplineDecompositionImageFilter`.
  → the C/D/§C canny-edge, unsharp, min-max-curvature-flow, chan-vese, and the
  bspline direction-2 bug are all genuinely reachable.
- **Output forced to `RealType`=double** (`output_pixel_type:
  NumericTraits<InputPixelType>::RealType`): `CurvatureFlowImageFilter` (plain) —
  update image is `Image<double>`, port f64 **matches** → faithful. But
  `FastMarchingImageFilter` / `FastMarchingUpwindGradientImageFilter` are a
  **trap**: SimpleITK's `output_pixel_type` is also `RealType`=double, yet the
  **port** keeps `Float32→Float32` (its own §5.6 `real_pixel_id` divergence) and
  narrows arrival times to **f32** — so the port is NARROWER than ITK here, a
  **candidate**, not faithful. Verify the *port's* output type, not just
  SimpleITK's. (MinMaxCurvatureFlow has no override at all → float → candidate.)
- **N4** (`RealPixelIDTypeList`): `RealType` is **hardcoded float** internally,
  independent of the input, so §4.122 is reachable for **every** input type
  (Float64 input included), not just Float32.
- **Faithful in the level-set/registration cluster** (discrete gates computed in
  double): `AntiAliasBinaryImageFilter` (`output_pixel_type: RealType` = double),
  `DemonsRegistrationFilter` / `FastSymmetricForcesDemonsRegistrationFilter`
  (update, normalizer, and both threshold gates in `CoordinateType=double` —
  `itkDemonsRegistrationFunction.h:98`; the vector field is float-valued *output*,
  not a discrete-decision input). → not candidates.
- **Candidates in the level-set cluster**: `FastMarchingImageFilter` /
  `FastMarchingUpwindGradientImageFilter` (port keeps Float32 output and narrows
  arrival times to **f32** where SimpleITK is double — port narrower, §5.6),
  `CollidingFrontsImageFilter` (port accumulates the gradient dot product in
  **f32** where ITK's `CovariantVector::operator*` accumulates in **double** —
  port narrower), `ReinitializeLevelSetImageFilter` (`RealPixelIDTypeList`,
  output=input → float level set at Float32; weak — the port already narrows most
  of it), `ScalarChanAndVeseDenseLevelSet` (Float32), and the sparse-field
  segmentation level sets (undocumented f64 evolution).
  `DisplacementFieldJacobianDeterminant` (`TRealType` default **float**) produces
  a continuous determinant → §4-keep unless a downstream consumer thresholds it.

### G. Level-set / fast-marching / demons / N4 / chan-vese / displacement-field

Every candidate here is the **N4 §4.122 family**: the port runs the solver in
f64/f32 while ITK/SimpleITK instantiate the Float32 case with a **float** compute
type (or, via `RealType`, a **double** one the port narrowed below). Cross-
verified by the dedicated agent and this panel; two fork conflicts were resolved
firsthand (see below). Float64 input → ITK double → port matches (no row).

| Port site | ITK site | Port type | ITK type | Feeds | Class | Separating input |
|---|---|---|---|---|---|---|
| `level_set/sparse_field.rs:485,501,517,527` (rms accum) + `:535-538` (`(acc/counter as f64).sqrt()`) → `:227` halt `maximum_rms_error > rms_change` | `itkSparseFieldLevelSetImageFilter.hxx:344,395,427` (accum in `ValueType`), `:443` `sqrt(double(acc/ValueType(counter)))`; `itkFiniteDifferenceImageFilter.hxx:225` Halt | f64 accum + f64 division | **float** accum + float division (only final sqrt→double) | Discrete + amplifying: RMS convergence stop / iteration count | **CANDIDATE — UNDOCUMENTED, exact level-set analog of §4.122** | Float32 seg-level-set (GeodesicActiveContour/ShapeDetection/ThresholdSegmentation) whose float-accumulated RMS lands one side of `maximum_rms_error`, f64 the other → off-by-one iteration → different final level set |
| `level_set/sparse_field.rs:494` (`new_value >= upper_active_threshold`), `:512` (`new_value < lower_active_threshold`) | `itkSparseFieldLevelSetImageFilter.hxx:324,375`; thresholds `ValueType` (`.h:269`) | f64 | float (`ValueType`) | Discrete: narrow-band layer promote/demote membership | **CANDIDATE** | `constant_gradient_value=1.0` ⇒ threshold ±0.5; `new_value ≈ 0.5 ± 2⁻²⁴` promotes in f64 but not float → different narrow-band topology |
| `level_set/sparse_field.rs:507,521` (`new.abs() < old.abs()` promote tie) | `itkSparseFieldLevelSetImageFilter.hxx` UpdateActiveLayerValues | f64 | float | Discrete: which neighbor is pulled into the active layer (abs-ordering tie) | **CANDIDATE** | `abs()` tie between two candidate neighbors breaks differently under f64 vs float |
| `level_set/function.rs:105,232` (compute_update), `:283` (mean_curvature), `:132-144` (dx/grad_mag_sqr/dxy), `:375` (dt) | `itkLevelSetFunction.h:86` (`ScalarValueType = PixelType`), `:104-117` GlobalData members `ScalarValueType` | f64 | float (`ScalarValueType`) | Update magnitude → feeds sparse_field membership + RMS above | **CANDIDATE — root of the cluster, UNDOCUMENTED** | Whole PDE (curvature/propagation/advection/grad-mag) in f64 vs float. Unlike chan_vese/N4 (documented) and fast-marching (narrowed), this file neither narrows nor documents the deviation |
| `chan_vese.rs:773` (`if v < 0.0` label sign), `:820` (`lower <= v && v <= 0.0` mask), ComputeUpdate arith `:626` | `itkRegionBasedLevelSetFunction.h:71` / `itkScalarChanAndVeseLevelSetFunction.h:94` (`ScalarValueType = PixelType`; c_in/c_out accumulators are double) | f64 | float | Discrete: Heaviside `x>=0` branch, per-iter reinit sign, mask→Maurer distance→RMS (`:742`), final output sign | **CANDIDATE (documented, `chan_vese.rs:165-186`)** | Float32 pixel with `φ ≈ 0 ± 2⁻²⁴` classifies inside vs outside differently; the port doc's "agree to f32 round-off" claim breaks at the `φ≈0` sign branch |
| `fast_marching.rs:145-147,157,226,427-433` (**f32 arm** + `narrow`), storage `:451,688`, staleness `:461`, `value > stopping_value` break `:467`, membership `solution < large` `:686` | `FastMarchingImageFilter.yaml:5` `output_pixel_type = RealType` = **double** for Float32; `itkFastMarchingImageFilter.hxx:396-448` solves in **double**, stores/heaps in PixelType | **f32** (Float32 speed) | **double** (RealType=double for Float32) | Discrete: heap ordering, stopping-value break, target/membership test | **CANDIDATE — port NARROWER (direction 2; tied to §5.6 `real_pixel_id`)** | Float32 speed image with irrational diagonal arrival times: per-step f32-vs-double rounding flips a stopping/target break or a two-front min. u8/f64 input → both double, no divergence |
| `fast_marching_upwind_gradient.rs:237,260,263` (f32 narrow + stopping) + target-reached drop `fast_marching.rs:593-595` | `FastMarchingUpwindGradientImageFilter.yaml:5` `output_pixel_type = RealType` = double | **f32** (Float32 speed) | **double** (RealType) | Discrete: same march tests + target-reached mid-march stop | **CANDIDATE — port NARROWER (direction 2; §5.6)** | A narrowed arrival time that becomes `target_value` moves the mid-march stop onto a different pixel |
| `colliding_fronts.rs:273` (`fold(0.0f32, …a[i] as f32 * b[i] as f32)`) | `itkCollidingFrontsImageFilter.hxx:77` `MultiplyImageFilter` over `itkCovariantVector.hxx:104-109` `operator*` (accumulates in `NumericTraits<T>::AccumulateType` = **double** for float, `itkNumericTraits.h:1356`) | **f32** accumulator | **double** accumulate → float store | Discrete: negative-region membership `v <= negative_epsilon` (`:194→-1e-6`), flood seeding (`:202,220`), output sign (`:209`) | **CANDIDATE — port NARROWER (direction 2)** | Near-antiparallel gradients: `a·c+b·d` straddles −1e-6; f32-sum → −0.9e-6 (excluded) vs ITK's double-sum → −1.1e-6 (included). The port comment claiming ITK accumulates in float is **wrong** (firsthand-verified: `AccumulateType`=double). |
| `n4_bias_field.rs:279` (`v.ln()`), `:390` (histogram_slope), `:395-396` (`cidx`, `finite_floor`) | `itkN4BiasFieldCorrectionImageFilter.hxx:141` (`logf`), `:303` (`histogramSlope` float), `:318-319` (`cidx`/`Math::floor` float), `:461-462` (2nd pass) | f64 | float (`RealType`) | Discrete: histogram bin index (`floor`) → sharpened histogram → bias field, cascades every iteration | **CANDIDATE — new N4 neighbor of §4.122** | Log-intensity where `(pixel−binMin)/slope` = 4.0 in float but 3.9999998 in f64 → bin 3 vs 4. Port narrows the bin-*count* log (`:612`) but not the per-pixel binning log/slope/cidx |
| `n4_bias_field.rs:573` → `:293` (`convergence > convergence_threshold`) | `itkN4…hxx:206`; `RealType=float` (`.h:114`) | f64 | float | Discrete: convergence stop | **KNOWN SEED §4.122** (not counted new) | Already documented (port doc `:25`) |
| `n4_bias_field/bspline.rs:438` (`omega != 0.0`, f64 omega accum) | `itkN4…` fit `Math::NotAlmostEquals(omega,0)` on float omega | f64 accum + exact `!=0` | float accum + 4-ULP window | Discrete: divide-or-skip `phi = delta/omega` | **CANDIDATE (documented, argued-benign — `bspline.rs:438`)** | Type + gate divergence; port argues the non-finite-quotient guard catches subnormals |
| `displacement_field/invert.rs:176-177,219-220` (max/mean error norms f64) → `:181-182` stop | `itkInvertDisplacementFieldImageFilter.h:144` (`RealType = VectorType::ComponentType`); `.hxx:115-116` stop | f64 | **float for VectorFloat32** field, double for VectorFloat64 | Discrete: convergence tolerance stop | **CANDIDATE (VectorFloat32 field only)** | `RealVectorPixelIDTypeList` ⇒ float+double instantiations; a `max_error_norm` rounding to exactly the tolerance in float but not f64 stops one iteration off (`invert.rs:84` shows the port knows the field can be VectorFloat32) |

**Faithful / §4-keep in this cluster — verified firsthand:**

- `demons/*` (common, compose, diffeomorphic, esm, fast_symmetric, field,
  image_function, symmetric, geometry): field is `Vector<double>` (SimpleITK),
  update/normalizer/metric/RMS/thresholds all `double`; `numiterfloat`→uint round
  and floor interpolation indices in the same double ITK uses. `esm.rs` narrows
  the moving value to f32 only for Float32 pixels (matches `WarpImageFilter`).
  `demons/level_set_motion.rs:385-392` holds the smoothed moving image in f64
  where ITK quantizes to the moving pixel type before differencing — the **§2.49
  documented deliberate deviation** (direction-1, port more precise), feeding the
  minmod sign test `:170` and `gradient_magnitude < 1e-9` `:295`. `compose.rs`
  `2^numiter` in f64 is §4.20.
- `region_growing.rs`: accumulators f64 = ITK `InputRealType`/`AccumulateType`=
  double for every scalar; ConfidenceConnected mean/variance and IsolatedConnected
  bisection (`i128`, wider = inert §4-keep) match. (The IsolatedConnected UInt32
  overflow row lives under cluster F above.)
- `level_set/anti_alias.rs`: `AntiAliasBinaryImageFilter` output = `RealType` =
  **double** for integer input → port f64 matches (documented `anti_alias.rs:64-66`).
- `displacement_field/iterative_inverse.rs` (ITK explicit `double` + `double
  m_StopValue`), `displacement_field/inverse.rs` (ITK `KernelTransform<double>`;
  the eigendecomposition-vs-SVD is the documented §4.19), `displacement_field.rs`
  (plumbing): no narrowing.
- `displacement_field/jacobian_determinant.rs`: port f64 where ITK `TRealType`
  defaults to **float** — continuous determinant output, no in-filter discrete
  consumer → §4-keep (direction-1 superset).
- `reinitialize_level_set.rs:266` (`center/(center-neighbor)*spacing` in f64) is a
  **weak** candidate: the port already narrows center/neighbor (`:240,262`) and
  the stored node (`:271`) to f32, but does the interpolation division itself in
  f64 vs ITK's float (`itkLevelSetNeighborhoodExtractor.hxx:234`) → a within-1-ULP
  residual can flip a two-axis min tie or a narrow-band-edge seed (Float32 only).
- Zero differing sites: `fast_marching_base.rs` (double internal solve, narrows
  every store to match float output), `level_set/grid.rs` (integer index
  arithmetic), `level_set/mod.rs` (orchestration; sets up the f64 buffers that are
  the architectural cause of the sparse-field cluster but has no own compute site).

The **largest undocumented finding** is the sparse-field segmentation level sets
(`function.rs` + `sparse_field.rs`, driven by `mod.rs`'s f64 buffers): the whole
evolution runs in f64 where ITK's `ScalarValueType`/`ValueType` is float for a
Float32 image, feeding three discrete decisions — narrow-band membership
(`:494/:512`), RMS convergence stop (`:227`, the direct §4.122 analog), and
promote-neighbor abs-ordering (`:507/:521`) — none narrowed, none documented.

# Verification round — firm verdicts with separating inputs

Report-only. Each candidate: one-line verdict, both-side type evidence
(read firsthand in `/home/stevek/work/ITK`), a concrete separating input.
Direction-2 candidates (bspline_decomposition, colliding_fronts,
fast_marching) are the sister panel's — not re-verified here.

Verdict key: **§4-POLICY** = port computes in f64 where ITK uses `float`;
the port is *more precise*, but a discrete decision diverges from ITK's
exact float arithmetic (the N4 §4.122 precedent — user decides keep-precise
vs match-ITK-float). **REAL-BUG** = port is *narrower* than ITK and simply
wrong. **NOT-REAL** = faithful / continuous-only, no discrete divergence.

## Candidate 1 — HISTOGRAM-THRESHOLD family — §4-POLICY

Port `Histogram` bin edges are f64 (`histogram.rs:316-359`, doc `:313-315`
states the deviation explicitly). ITK `itkHistogram::Initialize`
(`itkHistogram.hxx:216-235`) hardcodes `float interval =
(float(upper)-float(lower))/float(size)` and `SetBinMin(j, lower +
float(j)*interval)` — float regardless of `MeasurementType`. Bin membership
(`GetIndex`, `:243-296`) is a discrete decision, and the selected threshold
(Otsu/Huang/… argmax over bins) is discrete and propagates to the binary
segmentation.

Separating input, single-pixel bin flip (computed, `hist_edge.rs`):
range `[0,255]`, 100 bins. ITK float edge[1] = `2.54999995231628418`,
port f64 edge[1] = `2.55`. A `Float32` pixel set to `2.54999995231628418`
(a representable value, = ITK's edge exactly) lands in ITK **bin 1**
(`v >= edge`) but port **bin 0** (`v < 2.55`). Gap `4.768e-8`; 64 of 99
interior edges have such a pixel for this range/bin-count.

Separating input, flipped *selected* threshold (computed, `otsu2.rs`):
a constructed 92-pixel `Float32` image (clusters at 38.0/127.0/216.0 plus
3 pixels on each ITK float edge in bins 60–71) makes Otsu's between-class
variance a near-tie between two peaks. ITK's float histogram selects
**bin 49, threshold 126.22**; the port's f64 histogram selects **bin 14,
threshold 36.98**. **10 of 92 pixels** land in different binary classes.

Reachability: all 12 threshold calculators (`threshold.rs`) + Otsu +
every `ImageToHistogram`-based filter. Bites only when the interval is
**not** float-exact — bins = 100/200/250, or an arbitrary float data range.
**Zero** divergence for float-exact intervals: `[0,255]`/128 (SimpleITK's
default OtsuThreshold), `[0,200]`/128, `[0,4095]`/256 all give
`fi == fp` (verified). For `Float32` input a pixel must sit exactly on a
float edge (constructible, or quantized/rescaled data); for `Float64`
input the edge gap is densely populated → generic divergence.

## Candidate 2 — N4 histogram binning (`n4_bias_field.rs:395-396`) — §4-POLICY

Port computes `bin_minimum`, `bin_maximum`, `histogram_slope`, and
`cidx = (pixel - bin_minimum)/histogram_slope` in f64, then
`finite_floor(cidx)` (discrete). ITK
(`itkN4BiasFieldCorrectionImageFilter.hxx:272,303,318-319`) computes all of
these in `RealType`, and N4 hardcodes `using RealType = float`
(`itkN4BiasFieldCorrectionImageFilter.h:114`, verified). So ITK's
`histogramSlope`, `cidx`, and `itk::Math::floor(cidx)` are float.

Separating input (computed, `n4.rs`): log-intensity range `[0,7]`, 200 bins
(default). ITK float slope = `0.03517587855458260`, port f64 slope =
`0.03517587939698492`. Log-voxel `p = 0.035175879` (f32) → ITK
`cidx = 1.000000000` (float) → floor **1**; port `cidx = 0.999999976`
(f64) → floor **0**. The voxel is Parzen-binned into bin 1 (ITK) vs bin 0
(port) → different histogram → different sharpened intensity map →
different bias field, and the divergence iterates. Five such voxels found
in the first sweep (`p = 0.0703…, 0.1055…, 0.1407…, 0.1759…`, each floor
off-by-one). N4 is `Float32`-reachable in SimpleITK; default 200 bins →
non-float-exact slope → generic.

## Candidate 3 — level-set / curvature — §4-POLICY (one site UNDOCUMENTED)

The whole sparse-field / chan-vese / min-max-curvature evolution runs in
f64 in the port, where ITK instantiates on the output pixel type. ITK
`SparseFieldLevelSetImageFilter::ValueType = OutputImageType::ValueType`
(`itkSparseFieldLevelSetImageFilter.h:269`, verified) — **float** for the
`Float32` level-set output SimpleITK produces. So `new_value`,
`rms_change_accumulator` (`itkSparseFieldLevelSetImageFilter.hxx:303,344`),
and the layer thresholds are float. chan_vese's port doc (`chan_vese.rs:167-173`)
already states "the Heaviside image, the update buffer, the level set and
ComputeUpdate's entire arithmetic are float; this port computes in f64
throughout." min_max_curvature_flow's gate compares ball-average vs
threshold in f64 where ITK uses `PixelType`=float (`min_max_curvature_flow.rs:49-54`).

Discrete decisions that can flip at the float-vs-f64 boundary:
- **RMS convergence stop** (`sparse_field.rs:227`, `maximum_rms_error >
  rms_change`) — **UNDOCUMENTED**.
- **Narrow-band membership** (`sparse_field.rs:494/512`, `new_value >=
  upper_active_threshold` / `< lower_active_threshold`).
- **chan_vese `phi < 0` sign** — pixel classified inside vs outside →
  different output label *and* different `c_in`/`c_out` region means.
- **min/max curvature gate** — ball-average `< threshold` picks
  `max(κ|∇I|,0)` vs `min(κ|∇I|,0)` → opposite smoothing sign.

Separating input, RMS-halt primitive (computed, `rms.rs`): a 400-pixel
active layer (`center = 1.0 + i*0.013`, `Δ = 0.0069 + (i%7)*1e-6`, all
`Float32`). Float accumulation → `rms_change = 0.006903004367`; f64
accumulation → `0.006903007927`. With `maximum_rms_error = 0.006903006`,
ITK's float RMS is *below* the bound → **halt fires**; the port's f64 RMS
is *above* it → **halt does not fire**. ITK stops, the port takes another
iteration → different iteration count → different final level set. Any
`maximum_rms_error` in `(0.006903004367, 0.006903007927]` flips the halt.

Full end-to-end ITK-vs-port segmentation divergence is **not executed** here
(would need a built ITK level-set run); the primitive-level flip above proves
the discrete decision is reachable, and the type divergence is confirmed
firsthand.

## Verification summary

| Candidate | Verdict | Discrete outcome that diverges | Separating input |
|---|---|---|---|
| Histogram-threshold (12 calc + Otsu) | §4-POLICY | selected threshold / bin membership | Otsu bin 49↔14, 10/92 px reclassified |
| N4 histogram binning | §4-POLICY | Parzen floor bin | log-voxel 0.035175879 → floor 1↔0 |
| Sparse-field RMS stop | §4-POLICY (undoc) | iteration count (halt) | max_rms 0.006903006 → halt ITK / run port |
| Sparse-field layer membership | §4-POLICY | promote/demote at ±g/2 | boundary value on `±constant_gradient/2` |
| chan_vese `phi<0` | §4-POLICY (doc) | inside/outside label + c_in/c_out | pixel with `phi ≈ 0` |
| min_max_curvature gate | §4-POLICY (doc) | min vs max smoothing sign | ball-average ≈ threshold |

All six are direction-1 (port f64, more precise; ITK float) — none is a
narrower-than-ITK REAL-BUG. Every one is the N4 §4.122 shape: keep the
more-precise f64 (document each as §4.x) or reproduce ITK's float
arithmetic to hold bit-for-bit. The sparse-field RMS stop is the only one
not yet documented in-tree.
