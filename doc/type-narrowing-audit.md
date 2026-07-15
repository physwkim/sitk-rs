# Type-narrowing precision audit — conclusion

**Result: one real bug found and fixed; one false candidate refuted; six
deliberate keep-f64 divergences documented.** This sweep asked, for every site
where the port's compute type could differ from ITK/SimpleITK's, whether the
precision difference feeds a *discrete* decision (a threshold, a stop, a
branch, a round-to-int, a saturating cast) — the only place a ULP difference
can flip an observable output. It was built from both ends independently (ITK
source vs port code) and merged; candidates were then converted to firm
verdicts with computed separating inputs, with the port-narrower ("direction-2")
candidates cross-checked by an independent panel. This document is the
conclusion; the evidence is the census maps in `bench/results/` and the commits
cited below.

## The discriminator

Most "port keeps f64 where ITK pins float" sites are **faithful, not
supersets**, because `NumericTraits<float>::RealType == double`
(`itkNumericTraits.h:1356`): ITK's own `RealType`/`AccumulateType` computations
run in `double` even for a `float` image. ITK genuinely uses `float` only where
it (a) hardcodes `RealType = float` (N4, sparse-field level sets), (b) computes
in a raw buffer / `PixelType` (`OutputImagePixelType`, `FixedImagePixelType`),
or (c) hardcodes `float` arithmetic irrespective of the template type (the
histogram bin interval). Those are the only places a real candidate can live.

## What was wrong — one bug

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

## What was not wrong — one refuted candidate

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

## Deliberate divergences — six keep-f64 decisions (§4.124–4.129)

Six sites keep the port's `f64` where ITK/SimpleITK uses `float` and thereby
shift a discrete outcome only at a boundary. The user chose **keep f64**
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

Two of these were undocumented or misdocumented and were corrected in-tree:
`histogram.rs` had wrongly claimed ITK bins in `double` (asserting a parity
that does not exist — the bin *interval* is `float` regardless of
`MeasurementType`); the sparse-field RMS stop had no divergence note at all.

## Verification

- BSpline: the fixed filter's `Float32` output equals the `f64` line narrowed
  once, and differs from the per-step-narrowed line (regression guard);
  `sitk-filters` `cargo nextest run` 2309 passed / 1 skipped, clippy clean.
- Every ITK/SimpleITK citation in §4.124–4.129 verified firsthand (the
  histogram `float interval` and both filter paths that reach it, the
  mean-squares `FixedImagePixelType` subtract, the sparse-field `ValueType`
  RMS accumulation, the BSpline `CoeffType`).
- Full-workspace gate green after the bspline fix and doc corrections:
  `cargo nextest run --workspace` 3521 passed / 1 skipped.

## Evidence

- `bench/results/type-narrowing-core-map.md` — sitk-core + sitk-filters census
  and verify verdicts (rayon-core).
- `bench/results/type-narrowing-transform-map.md`,
  `bench/results/type-narrowing-direction2-verify.md` — transform + registration
  + device census and the direction-2 cross-check (cuda-backend).
- Commits `52b6a95` (bspline fix), `d6029f7` (§4.124–4.129 + doc corrections).
