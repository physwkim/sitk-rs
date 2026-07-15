# Type-narrowing precision audit — conclusion

**Result: two real bugs found and fixed; three false candidates refuted; one
upstream-overflow the port is more correct than (§8); thirteen deliberate
keep-f64 divergences documented (§4.124–4.136).** This sweep asked, for every
site where the port's compute type could differ from ITK/SimpleITK's, whether
the precision difference feeds a *discrete* decision (a threshold, a stop, a
branch, a round-to-int, a saturating cast, an overflow) — the only place a
ULP or a wraparound can flip an observable output. It was built from both ends
independently (ITK source vs port code) and merged; candidates were then
converted to firm verdicts with computed separating inputs, with the
port-narrower ("direction-2") candidates cross-checked by an independent panel.
This document is the conclusion; the evidence is the census maps in
`bench/results/` and the commits cited below.

## The discriminator

Most "port keeps f64 where ITK pins float" sites are **faithful, not
supersets**, because `NumericTraits<float>::RealType == double`
(`itkNumericTraits.h:1356`): ITK's own `RealType`/`AccumulateType` computations
run in `double` even for a `float` image. ITK genuinely uses `float` only where
it (a) hardcodes `RealType = float` (N4, sparse-field level sets), (b) computes
in a raw buffer / `PixelType` (`OutputImagePixelType`, `FixedImagePixelType`),
or (c) hardcodes `float` arithmetic irrespective of the template type (the
histogram bin interval). Those are the only places a real candidate can live.

## What was wrong — two bugs

### 1. The 64-bit-integer `to_f64_vec`/`from_f64` seam (port narrower than ITK)

**Every pure pixel-movement and value-transform filter routed its pixels through
`Image::to_f64_vec()` → operate → `image_from_f64()`, and `to_f64_vec`/`as_f64`
rounds a `UInt64`/`Int64` magnitude above `2^53` — the largest integer `f64`
represents exactly.** So `2^53 + 1 = 9007199254740993` collapsed to `2^53`
wherever these filters touched it, while ITK does the identical operations
natively and losslessly (`std::map<TInput, TOutput>` for label remap, native
`static_cast` for clamp/cast, `ImageAlgorithm::Copy` for reindex). This was the
sole *direction-2* (port-narrower) find of the sweep, confirmed firsthand by
both panels with the `2^53 + 1` separating input.

The structural fix removes the f64 seam from the whole family rather than
patching each site:

- **`sitk-core::Image::gather`** — a native reindex primitive: output pixel `i`
  is a bit-exact copy of the stored native pixel at linear source index
  `sources[i]` (or a quantized constant fill), dispatched on `pixel_id` over the
  component buffer, never widened to `f64`; whole-pixel copies for
  vector/complex. This is what ITK gets from `ImageAlgorithm::Copy` /
  `static_cast`. Every reindex/pad/copy filter — `flip`, `permute_axes`,
  `crop`, `region_of_interest`, `extract`, the four pads, `slice`, `shrink`,
  `fft_shift`/`cyclic_shift`, `dicom_orient`, `join_series`,
  `checker_board`/`paste`/`tile` — now routes through it. Commits `33d00b2`
  (primitive) and `629e3c9`/`ab5e7fd`/`621c141`/`729f2ae`/`37f94f8`.
- **Value transforms** route natively without a new primitive: `change_label`
  builds a native `HashMap<T, T>` and passes un-remapped pixels through
  bit-exact; `cast` widens integer→integer through `i128` and narrows via native
  `static_cast`; `clamp` keeps ITK's `f64` comparison (its documented `dA`
  semantics) but casts the *original native pixel* on the in-range branch, so an
  in-range `UInt64` above `2^53` is preserved. Commits `52be530` (change_label),
  `ac9ba37` (cast), `3841bb7` (clamp).

Every routed op has a `2^53 + 1` (or `−(2^53 + 1)`) boundary test with an
`assert_ne!` non-vacuity guard proving the value differs from its f64
round-trip. The port now matches ITK — this is a parity fix, not a divergence,
so it earns no §4/§8 row.

### 2. BSplineDecompositionImageFilter narrowed its IIR recursion per step

**`BSplineDecompositionImageFilter` narrowed its IIR recursion per step.** ITK's
scratch line is `std::vector<CoeffType>` with `CoeffType =
NumericTraits<OutputPixel>::RealType`, i.e. `double` even for a `Float32` image
(`itkBSplineDecompositionImageFilter.h:85,133,175`). The gain multiply and the
whole causal/anticausal recursion run in `double`; the *only* narrowing to the
output type is the `static_cast<OutputPixelType>` at write-out
(`CopyScratchToCoefficients`, `.hxx:281`), and because each axis re-reads the
previous axis's rounded output (`.hxx:297`) that rounding happens **once per
axis**, not per step. The port instead rounded to `f32` after every step
(gain, both initializations, both recursions) on the false premise that
`RealType` is `float` for a `Float32` image — measured up to **24 f32 ULPs**
(max abs ~1.5e-5) off ITK on half the pixels of a `Float32` line. Fixed: the
line now runs entirely in `f64` and `coefficients_along_axis` narrows once as it
writes each line back, reproducing ITK's per-axis write-out rounding. The
mis-pinning test was replaced by a narrow-once regression guard. Commit
`52b6a95`.

## What was not wrong — three refuted candidates

Besides CollidingFronts (below), two tail candidates were downgraded on
firsthand evidence: the zero-crossing **`|a| < |b|` tie-break** is NOT-REAL
(`Absolute` is exact; a 9-million-pair sweep flipped 0 — only the Laplacian
*sign* diverges, §4.131), and **`watershed_classic`** is NOT-REAL/gated (the
classic `itkWatershedImageFilter` is not SimpleITK-exposed, and the reachable
`IsolatedWatershed` feeds a `double` GradientMagnitude — `RealPixelType =
double` — so `ScalarType == double` and the port matches).

**`CollidingFrontsImageFilter` is NOT narrower.** An initial census pass flagged
it (and `FastMarching`) as port-narrower. Independent firsthand verification
refuted CollidingFronts: SimpleITK's `output_pixel_type` is a literal `float`,
not `RealType`, and ITK templates the internal marches on `LevelSetImageType =
TOutputImage = float`, so the arrival times, gradients and dot product are all
`float` — the port's `Float32` output with an `f32`-accumulated dot product
matches ITK on both output type and internal precision. Recorded here so the
next census does not re-flag it.

`FastMarching` *is* narrower (SimpleITK output = `RealType` = `double` for a
`Float32` speed input, port keeps `Float32`), but that is the already-ledgered
**§5.6 `RealType = double` output-type family** — a release-level decision
(flip all to `double`, breaking, vs keep documented), not a new bug. Left as
tracked.

## More correct than ITK — one §8 divergence

**`IsolatedConnectedImageFilter`'s bisection-threshold midpoint overflows a
non-widened `AccumulateType` in ITK; this port computes it in `i128`.** ITK
forms `guess = (upper + lower) / 2` in `NumericTraits<PixelType>::AccumulateType`
(`itkIsolatedConnectedImageFilter.hxx:146,230,295`) — this is the *bisection
midpoint*, not a region sum, so region size is irrelevant. `AccumulateType` is
non-widened for `unsigned int`, `long`, `unsigned long`, `long long`
(`itkNumericTraits.h:1021,1027`), so with the default `upper = UINT_MAX` a
`UInt32` image overflows: `upper = 4.29e9, lower = 3.0e9` wraps `guess` to
~`1.5e9`, below `lower`, and the search exits early with the wrong isolated
value. The port's `i128` midpoint gives the true `3.65e9` and keeps converging;
small values stay bit-identical. Both panels confirmed the mechanism firsthand.
This is a genuine ITK integer-overflow quirk whose result is not a function of
the inputs, so the port is deliberately more correct — ledger **§8.4**.

## Deliberate divergences — thirteen keep-f64 decisions (§4.124–4.136)

Thirteen sites keep the port's `f64` where ITK/SimpleITK uses `float` and
thereby shift a discrete outcome only at a boundary. The user chose **keep f64**
(uniform crate precision, strictly more accurate; SimpleITK defaults are
float-exact so default usage does not diverge) — the same call made for N4
(§4.122). Each is now a ledger row with a computed separating input:

| § | Site | Discrete effect | ITK float source |
|---|------|-----------------|------------------|
| 4.124 | histogram-threshold family (12 filters) | different selected threshold → different segmentation (10/92 px at 100 bins) | `itkHistogram.hxx:224` float interval |
| 4.125 | N4 sharpening-histogram bin index | log-voxel floors to a different bin | `itkN4…h:114` `RealType=float` |
| 4.126 | sparse-field level-set RMS stop | ±1 iteration near `maximum_rms_error` | `itkSparseFieldLevelSetImageFilter.hxx:303,344,443` `ValueType=float` |
| 4.127 | chan_vese `phi<0` region test | opposite inside/outside label at `phi≈0` | `float` update chain |
| 4.128 | min/max-curvature-flow gate | opposite branch on a float-width near-zero window | `PixelType=float`, `itkMath.h:339-341` |
| 4.129 | mean-squares subtract | ±1 iteration at the convergence stop | `…MeanSquares…Threader.hxx:44` `FixedImagePixelType` |
| 4.130 | canny hysteresis seed (`value > upper_threshold`) | seeds an edge chain in one and not the other | `itkCannyEdgeDetectionImageFilter.h:186-192` float edge buffer |
| 4.131 | zero-crossing Laplacian sign | pixel marked on the opposite side of a crossing | `float` Laplacian buffer (tie-break is exact, not divergent) |
| 4.132 | unsharp `\|orig − blur\| > threshold` branch | pixel unchanged vs sharpened | `UnsharpMask.yaml:7` `TInternalPrecision=PixelType`, `.h:209-219` |
| 4.133 | IsoContour narrowband seed | float-chain running minimum vs f64 | `itkIsoContourDistanceImageFilter.hxx:363-372` `static_cast<float>` |
| 4.134 | FastChamfer relaxation weights | float running distance vs f64 | `itkFastChamferDistanceImageFilter.h:108` `FixedArray<float>` |
| 4.135 | VectorCC `1 − \|dot\| <= threshold` | two components joined vs split | `itkVectorConnectedComponentImageFilter.h:79-84` float per-term dot |
| 4.136 | InvertDisplacementField convergence stop | stops a different iteration → different inverse field | `itkInvertDisplacementFieldImageFilter.h:144` `RealType=ComponentType=float` |

Several of these were undocumented or misdocumented and were corrected in-tree:
`histogram.rs` had wrongly claimed ITK bins in `double` (asserting a parity
that does not exist — the bin *interval* is `float` regardless of
`MeasurementType`); the sparse-field RMS stop had no divergence note at all;
the IsoContour/FastChamfer float weights and the chan_vese/min-max-curvature
float chains are noted at their module docs.

## Verification

- 64-bit seam: every routed reindex, pad, copy, relabel, clamp and cast op has
  a `2^53 + 1` (or `−(2^53 + 1)`) boundary test with an `assert_ne!`
  non-vacuity guard proving the value differs from its f64 round-trip;
  `Image::gather` itself is pinned on identity, reversed, vector, and
  constant-fill paths for both `UInt64` and `Int64`.
- BSpline: the fixed filter's `Float32` output equals the `f64` line narrowed
  once, and differs from the per-step-narrowed line (regression guard).
- Every ITK/SimpleITK citation in §4.124–4.136 and §8.4 verified firsthand (the
  histogram `float interval` and both filter paths that reach it, the
  mean-squares `FixedImagePixelType` subtract, the sparse-field `ValueType`
  RMS accumulation, the BSpline `CoeffType`, the canny/unsharp/isocontour/
  chamfer/vectorCC/invert float sources with computed separating inputs, and
  the IsolatedConnected `AccumulateType` overflow measured on a `UInt32`
  bisection), cross-checked by an independent panel for the direction-2 and §8
  verdicts.
- Full-workspace gate green after the 64-bit seam fix, the bspline fix, and the
  doc work: `cargo fmt --all --check` clean, `cargo clippy --workspace
  --all-targets -- -D warnings` clean, `cargo nextest run --workspace` **3545
  passed / 1 skipped** (up from 3521 — the +24 are the new losslessness pins),
  doctests clean.

## Evidence

- `bench/results/type-narrowing-core-map.md` — sitk-core + sitk-filters census,
  headline + tail verify verdicts (rayon-core).
- `bench/results/type-narrowing-transform-map.md`,
  `bench/results/type-narrowing-direction2-verify.md` — transform + registration
  + device census, the direction-2 cross-check, and the §8 IsolatedConnected
  mechanism confirmation (cuda-backend).
- Commits: `52b6a95` (bspline fix); `d6029f7` (§4.124–4.129 + doc corrections);
  the 64-bit seam fix — `33d00b2` (`Image::gather`), `629e3c9`/`ab5e7fd`/
  `621c141`/`729f2ae`/`37f94f8` (reindex/pad/copy ops), `52be530`/`ac9ba37`/
  `3841bb7` (change_label/cast/clamp); this doc's §4.130–4.136 + §8.4 ledger
  rows.
