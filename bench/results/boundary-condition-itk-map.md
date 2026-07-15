# ITK BoundaryCondition map — the source half

**Scope.** For every ITK filter the sitk-rs port implements that reads a
neighborhood, what does a pixel *outside* the image return, and where is that
decided. ITK source read at `/home/stevek/work/ITK` (current `main`; the
`NeighborhoodOperatorImageFilter` default here is the modern one — see the
headline correction). This is the ITK-side table only; the port-code half is a
sister panel's, and the two get merged to find mismatches.

Every row is `.h`/`.hxx` cited. The load-bearing base-class anchors
(§ "Anchors") were re-read firsthand, not taken from the fan-out.

---

## Headline correction — the premise is inverted in this ITK

The brief stated *"ConstantBoundaryCondition (zero) — the default for
`NeighborhoodOperatorImageFilter` and the convolution/correlation family."*
**Measured at source, that is false in this checkout.** The neighborhood /
convolution / correlation family defaults to **ZeroFluxNeumann**, and the one
family that actually defaults to **Constant** is **mathematical morphology** —
the opposite assignment.

- `itkNeighborhoodOperatorImageFilter.h:93` — `using DefaultBoundaryCondition = ZeroFluxNeumannBoundaryCondition<InputImageType>;`
  ctor wires it: `.h:155` `m_BoundsCondition = static_cast<...>(&m_DefaultBoundaryCondition);`
- `itkConvolutionImageFilterBase.h:96` — `using DefaultBoundaryConditionType = ZeroFluxNeumannBoundaryCondition<TInputImage>;`
  ctor: `itkConvolutionImageFilterBase.hxx:26` `: m_BoundaryCondition(&m_DefaultBoundaryCondition)`
- `itkConstNeighborhoodIterator.h:52` — template default `TBoundaryCondition = ZeroFluxNeumannBoundaryCondition<TImage>`; active ptr `.h:624`.
- The only `ConstantBoundaryCondition` *default* among neighborhood filters:
  `itkMorphologyImageFilter.h:101` `using DefaultBoundaryConditionType = ConstantBoundaryCondition<InputImageType>;` filled with `PixelType{}`=0 (`itkMorphologyImageFilter.hxx:32`).
  (Elsewhere `ConstantBoundaryCondition` is a *default* only in the pure padder
  `itkConstantPadImageFilter.h:113`.)

So the divergence risk is not "port assumed Neumann, ITK is Constant." It is
the reverse plus one island: **ZeroFluxNeumann almost everywhere; Constant
(operation-specific value) in morphology; and a set of filters that use no
`BoundaryCondition` object at all** (recursive IIR, box integral-image, FFT
padding, distance transforms). A port that picked one global rule is wrong for
whichever of these families it did not special-case.

---

## Table 1 — ZeroFluxNeumann (nearest in-bounds edge value; border first-difference = 0)

`ZeroFluxNeumannBoundaryCondition` returns the nearest in-image edge pixel
(`itkZeroFluxNeumannBoundaryCondition.h:24-28`). This is the effective rule for
the large majority of the port's neighborhood filters.

| Filter | Where the ZeroFluxNeumann is set | Citation |
|---|---|---|
| DiscreteGaussianImageFilter | owns member BC, defaulted in ctor; user-settable via `SetInputBoundaryCondition`/`SetRealBoundaryCondition` | `itkDiscreteGaussianImageFilter.h:109,346`; applied `.hxx:230,257,270,290` |
| DiscreteGaussianDerivativeImageFilter | implicit — inner `NeighborhoodOperatorImageFilter` default; no member BC, no setter | `itkNeighborhoodOperatorImageFilter.h:93,155`; `itkDiscreteGaussianDerivativeImageFilter.hxx:179-213` |
| GradientImageFilter | owned member, overridable via `OverrideBoundaryCondition()` | `itkGradientImageFilter.h:229-230`; applied `.hxx:160` |
| GradientMagnitudeImageFilter | local stack `nbc`, overrides iterator | `itkGradientMagnitudeImageFilter.hxx:99,160` |
| DerivativeImageFilter | local `nbc` pushed onto internal `NeighborhoodOperatorImageFilter` | `itkDerivativeImageFilter.hxx:82,117` |
| LaplacianImageFilter | local `nbc` onto internal `NeighborhoodOperatorImageFilter` (LaplacianOperator) | `itkLaplacianImageFilter.hxx:94,122` |
| SobelEdgeDetectionImageFilter | local `nbc` onto each internal operator filter | `itkSobelEdgeDetectionImageFilter.hxx:100,117` |
| ZeroCrossingImageFilter | local `nbc` on neighborhood iterator | `itkZeroCrossingImageFilter.hxx:122,124` |
| ZeroCrossingBasedEdgeDetectionImageFilter | none of its own — inherits sub-filters (DiscreteGaussian→Laplacian→ZeroCrossing) | `itkZeroCrossingBasedEdgeDetectionImageFilter.hxx:47-49` |
| CannyEdgeDetectionImageFilter | `DefaultBoundaryConditionType` typedef + local `nbc` in compute passes | `.h:111,116`; `.hxx:98,121,376,417` |
| LaplacianSharpeningImageFilter | local `nbc` onto internal Laplacian operator filter | `itkLaplacianSharpeningImageFilter.hxx:96,99` |
| MeanImageFilter | ZeroFluxNeumann **pixel-access policy** on `ShapedImageNeighborhoodRange` boundary faces (interior uses buffered no-boundary policy) | `itkMeanImageFilter.hxx:66` (`ZeroFluxNeumannImageNeighborhoodPixelAccessPolicy`), interior `.hxx:55` |
| MedianImageFilter | boundary faces use `ShapedImageNeighborhoodRange` default policy (= ZeroFluxNeumann) | `itkMedianImageFilter.hxx:92-93`; default policy `itkShapedImageNeighborhoodRange.h:91` |
| BilateralImageFilter | local BC + `OverrideBoundaryCondition` in GenerateData (no member, not in ctor) | `itkBilateralImageFilter.hxx:256,261` |
| NoiseImageFilter | local BC + `OverrideBoundaryCondition` | `itkNoiseImageFilter.hxx:42,64` |
| BinaryMedianImageFilter | local `nbc` (not user-settable) | `itkBinaryMedianImageFilter.hxx:92,113` |
| VotingBinaryImageFilter | local `nbc` | `itkVotingBinaryImageFilter.hxx:93,112` |
| VotingBinaryHoleFillingImageFilter | local `nbc` | `itkVotingBinaryHoleFillingImageFilter.hxx:79,106` |
| VotingBinaryIterativeHoleFillingImageFilter | inherits — loops an internal HoleFilling filter | `itkVotingBinaryIterativeHoleFillingImageFilter.hxx:52` |
| ConvolutionImageFilter | base ZeroFluxNeumann default forwarded to internal `NeighborhoodOperatorImageFilter`; user-settable (`SetBoundaryCondition`) | `itkConvolutionImageFilterBase.h:96,100`; forward `itkConvolutionImageFilter.hxx:134` |
| NormalizedCorrelationImageFilter | inherits `NeighborhoodOperatorImageFilter` ZeroFluxNeumann default; forwards to iterator | `itkNormalizedCorrelationImageFilter.hxx:162` |
| AnisotropicDiffusion (Gradient/Curvature) | inherited FD default (see Anchors §FD) | `itkFiniteDifferenceFunction.h:93,100` |
| CurvatureFlowImageFilter | inherited FD default | `itkCurvatureFlowImageFilter.h:92`; `itkFiniteDifferenceFunction.h:93,100` |
| MinMax / BinaryMinMaxCurvatureFlow | inherited FD default (subclasses of CurvatureFlow) | `.h:79` / `.h:78` → base |
| FastChamferDistanceImageFilter | ZeroFluxNeumann via **default template arg** of `NeighborhoodIterator` (no explicit override) | `itkFastChamferDistanceImageFilter.hxx:102`; default `itkConstNeighborhoodIterator.h:52` |
| IsoContourDistanceImageFilter | ZeroFluxNeumann via default template arg of the neighborhood iterators | `itkIsoContourDistanceImageFilter.h:186-187`; `.hxx:226-227,269-270` |
| PatchBasedDenoisingImageFilter | explicit `ZeroFluxNeumannBoundaryCondition` typedef on patch iterator; disabled (`NeedToUseBoundaryConditionOff`) for proven-in-bounds patches | `itkPatchBasedDenoisingBaseImageFilter.h:187,191`; `itkPatchBasedDenoisingImageFilter.hxx:1646,2191,2245` |

Note on the "override or not" question for the `NeighborhoodOperatorImageFilter`
users (Derivative, Laplacian, Sobel, LaplacianSharpening): **all of them
explicitly override with their own ZeroFluxNeumann anyway**, so the result is
ZeroFluxNeumann under two independent guarantees — it holds even against an
older ITK whose base default was Constant.

---

## Table 2 — ConstantBoundaryCondition, value operation-specific (the morphology island)

The one family that departs from ZeroFluxNeumann. The constant is **not zero**
for the grayscale ops — it is the identity element of the op (min for dilate,
max for erode) so the border cannot change the result; and for binary ops the
fill flips between foreground and background by operation.

| Filter | Boundary MECHANISM | Where set | Citation |
|---|---|---|---|
| MorphologyImageFilter (base) | `ConstantBoundaryCondition(PixelType{}=0)` | member init + ctor | `itkMorphologyImageFilter.h:101,160`; `.hxx:32,64` |
| BasicDilateImageFilter | Constant `NumericTraits::NonpositiveMin()` (overrides base) | ctor | `itkBasicDilateImageFilter.hxx:27-28` |
| BasicErodeImageFilter | Constant `NumericTraits::max()` (overrides base) | ctor | `itkBasicErodeImageFilter.hxx:27-28` |
| GrayscaleDilateImageFilter | Constant `NonpositiveMin()`; sets it on all delegate subfilters | ctor→`SetBoundary` | `itkGrayscaleDilateImageFilter.hxx:36,101` |
| GrayscaleErodeImageFilter | Constant `max()`; sets it on all delegate subfilters | ctor→`SetBoundary` | `itkGrayscaleErodeImageFilter.hxx:36,101` |
| DilateObjectMorphologyImageFilter | Constant `NonpositiveMin()` — **only** consulted when `UseBoundaryCondition==true` (default false → in-bounds check, Table 3) | ctor | `itkDilateObjectMorphologyImageFilter.hxx:27-28` |
| ErodeObjectMorphologyImageFilter | Constant `max()` — same `UseBoundaryCondition` gate | ctor | `itkErodeObjectMorphologyImageFilter.hxx:27-28` |

---

## Table 3 — NOT a BoundaryCondition object (naming the token is a category error)

These hard-code edge behavior without an `ImageBoundaryCondition`. They are the
rows a `BoundaryCondition`-token scan (port or ITK side) cannot see; the merge
must match them by mechanism, not by the token.

| Filter | Actual mechanism | Citation |
|---|---|---|
| **Grayscale morphology (binary ops):** BinaryDilateImageFilter | bool `m_BoundaryToForeground = false` (out-of-image = **background**) + explicit `RegionType::IsInside()` tests; internal ConstantBoundaryCondition only on scratch tag image | `itkBinaryDilateImageFilter.hxx:36,420,442`; scratch `.hxx:178-179` |
| BinaryErodeImageFilter | bool `m_BoundaryToForeground = true` (out-of-image = **foreground**) + `IsInside()` tests — the *opposite* convention to dilate | `itkBinaryErodeImageFilter.hxx:36,414,436`; scratch `.hxx:168-169` |
| BinaryMorphologyImageFilter (base) | declares `m_BoundaryToForeground` default false, `m_ForegroundValue=max`, `m_BackgroundValue=NonpositiveMin` | `itkBinaryMorphologyImageFilter.h:225,229,232` |
| GrayscaleMorphologicalClosingImageFilter | `ConstantPadImageFilter` pads with `NonpositiveMin()`, then crops (SafeBorder) — not a per-access BC | `itkGrayscaleMorphologicalClosingImageFilter.hxx:39,146` |
| GrayscaleMorphologicalOpeningImageFilter | `ConstantPadImageFilter` pads with `max()`, then crops (SafeBorder) | `itkGrayscaleMorphologicalOpeningImageFilter.hxx:39,144` |
| ObjectMorphologyImageFilter (+ Dilate/Erode) | default `m_UseBoundaryCondition=false` → explicit `GetPixel(i,isInside)` in-bounds check, out-of-bounds neighbors **ignored** | `itkObjectMorphologyImageFilter.h:212`; `.hxx:189-190` |
| MovingHistogramImageFilter (base) | `inputRegion.IsInside(idx)` → in-bounds `AddPixel`, else histogram `AddBoundary()` hook | `itkMovingHistogramImageFilter.hxx:58-64,200-218` |
| MovingHistogramMorphologyImageFilter (base) | constant fill `m_Boundary` (default 0) fed to histogram (`m_Map[m_Boundary]++`); the Grayscale wrappers reset it to `NonpositiveMin`/`max` | `itkMovingHistogramMorphologyImageFilter.hxx:28,36`; `itkMorphologyHistogram.h:37-40` |
| RankImageFilter / MaskedRankImageFilter | `RankHistogram::AddBoundary()`/`RemoveBoundary()` are **empty no-ops** → out-of-bounds excluded, neighborhood **shrinks** at edges (not a fill, not a reflect) | `itkRankHistogram.h:238-243`; base `itkMovingHistogramImageFilter.hxx:58-64` |
| BinomialBlurImageFilter | hand-rolled forward/reverse averaging that **skips** the border pixel (edge pixel keeps its own value ≈ edge-replicate), plus region clamp | `itkBinomialBlurImageFilter.hxx:152,191,75-82` |
| BoxMeanImageFilter | integral-image (summed-area); at edges the window is **clipped to the region** and the sum divided by the in-bounds pixel count (mean of the *truncated* window) — numerically ≠ a ZeroFluxNeumann mean | `itkBoxUtilities.h:323-361`; `itkBoxMeanImageFilter.hxx:60-69` |
| BoxSigmaImageFilter | same integral-image window-clip mechanism (sum + sum-of-squares) | `itkBoxUtilities.h:500-542`; `itkBoxSigmaImageFilter.hxx:61-71` |
| **Recursive Gaussians (IIR):** SmoothingRecursiveGaussianImageFilter, RecursiveGaussianImageFilter, RecursiveSeparableImageFilter (base), GradientRecursiveGaussianImageFilter, GradientMagnitudeRecursiveGaussianImageFilter, LaplacianRecursiveGaussianImageFilter, UnsharpMaskImageFilter | forward+backward IIR recursion; border = the recursion's initial-condition seeding — the border sample is *assumed constant to ±∞* and primed by boundary coefficients `m_BN1..4`/`m_BM1..4` ("simulate edge extension"). No BC object, no clamp, no mirror. | `itkRecursiveSeparableImageFilter.hxx:70-88,106-122`; coeffs `itkRecursiveGaussianImageFilter.hxx:285-299`; e.g. `GradientMagnitudeRecursiveGaussian` `.h:80,83`, `LaplacianRecursiveGaussian` `.h:74,77`, `UnsharpMask` `.h:107-108` |
| StructureTensorImageFilter (AnisotropicDiffusionLBR) | built on `RecursiveGaussianImageFilter`/`GradientRecursiveGaussianImageFilter` — IIR border, no BC | `itkStructureTensorImageFilter.hxx:43,75,124` |
| CoherenceEnhancingDiffusionImageFilter | NOT FiniteDifference; diffusion in `LinearAnisotropicDiffusionLBRImageFilter` hand-codes Neumann via `region.IsInside()` + a dedicated `OutsideBufferIndex()` slot; structure tensor is recursive-Gaussian | `itkLinearAnisotropicDiffusionLBRImageFilter.hxx:146-153`; hierarchy `.h:71`, `itkAnisotropicDiffusionLBRImageFilter.h:54` |
| **Distance transforms:** SignedMaurerDistanceMapImageFilter | separable exact EDT (parabola lower-envelope / Voronoi); never reads outside the image — "border" is only the fg/bg partition via `m_BackgroundValue` | `itkSignedMaurerDistanceMapImageFilter.hxx:33,303-357,397` |
| DanielssonDistanceMapImageFilter / SignedDanielsson | two-pass raster propagation with `ReflectiveImageRegionConstIterator`; neighbor moves guarded by `it.IsReflected(dim)` — reflective scan, not a padded read | `itkDanielssonDistanceMapImageFilter.hxx:300,324,368` |
| ApproximateSignedDistanceMapImageFilter | no mechanism of its own — pipeline of IsoContour (ZeroFluxNeumann, Table 1) → FastChamfer (ZeroFluxNeumann, Table 1) | `itkApproximateSignedDistanceMapImageFilter.hxx:31-32,72,82` |
| **FFT-padded correlation:** FFTConvolutionImageFilter | boundary by **padding** (`FFTPadImageFilter`) filled from the base BC = ZeroFluxNeumann; comment: pad "taken from the boundary condition to avoid introducing extra information vs spatial convolution" | `itkFFTConvolutionImageFilter.hxx:196,258-259,264`; `itkFFTPadImageFilter.h:98`, `.hxx:37` |
| MaskedFFTNormalizedCorrelationImageFilter | **true zero-pad** — `ConstantPadImageFilter` with `SetConstant(0)`. This is a real divergence from `FFTConvolution`'s ZeroFluxNeumann pad; a shared padding routine would be wrong for one of them | `itkMaskedFFTNormalizedCorrelationImageFilter.hxx:379,386` |

---

## Anchors (re-read firsthand, they carry the whole map)

- **ZeroFluxNeumann semantics:** nearest in-bounds edge value; upwind first
  derivative on the boundary is zero. `itkZeroFluxNeumannBoundaryCondition.h:24-28`.
- **NeighborhoodOperator default:** `itkNeighborhoodOperatorImageFilter.h:93`
  (`DefaultBoundaryCondition = ZeroFluxNeumann`), ctor `.h:155`, member `.h:191`.
  Only `OverrideBoundaryCondition`/`GetBoundaryCondition` — no `SetBoundaryCondition`.
- **ConstNeighborhoodIterator default:** `itkConstNeighborhoodIterator.h:52`
  (template default ZeroFluxNeumann), active ptr `.h:624`.
- **Convolution base default:** `itkConvolutionImageFilterBase.h:96`, ctor `.hxx:26`;
  public `SetBoundaryCondition` at `.h:100`.
- **Finite-difference family default (§FD):** NOT an image-filter
  `SetBoundaryCondition` call — baked into the FD *function's* iterator type:
  `itkFiniteDifferenceFunction.h:93` (`DefaultBoundaryConditionType = ZeroFluxNeumann`)
  and `:100` (`NeighborhoodType = ConstNeighborhoodIterator<TImageType, DefaultBoundaryConditionType>`);
  constructed at `itkDenseFiniteDifferenceImageFilter.hxx:241,254`. This one typedef
  governs the whole diffusion / curvature-flow family. (Cross-checked: the anisotropic
  functions also re-assert it locally, e.g. `itkScalarAnisotropicDiffusionFunction.hxx:90,96`.)
- **Morphology default:** `itkMorphologyImageFilter.h:101` — the sole
  `ConstantBoundaryCondition` default among neighborhood filters.

---

## Bounding the anchor — where "BoundaryCondition" the token fails to see

`BoundaryCondition` is a token; a filter that hard-codes its edge behavior
without the token is invisible to a text scan for it. Upstream, those hide in
exactly the Table-3 rows, and they fall into distinct shapes the merge must
match by mechanism:

1. **A boolean + in-bounds test** instead of a BC object — Binary
   Dilate/Erode (`m_BoundaryToForeground` + `IsInside`), ObjectMorphology
   (`m_UseBoundaryCondition` default false + `GetPixel(i,isInside)`).
2. **A histogram hook** — MovingHistogram `AddBoundary()`; a constant for
   morphology (`m_Map[m_Boundary]++`), an **empty no-op** for Rank (silent
   neighborhood shrink — the most easily-missed row, because nothing names a
   boundary at all).
3. **A padding sub-filter** — `ConstantPadImageFilter` (grayscale
   open/close with `±extreme`; masked-FFT correlation with literal 0),
   `FFTPadImageFilter` (ZeroFluxNeumann pad). The fill value lives on the
   padder, not the neighborhood reader.
4. **A recursion seeding** — the IIR Gaussians' `m_BN*/m_BM*` boundary
   coefficients: the word "boundary" appears, but no `ImageBoundaryCondition`
   type does.
5. **An algorithmic border** — distance transforms (fg/bg partition;
   reflective iterator). No edge *fill* exists to name.

For the port merge, the token-searchable rows are Tables 1–2; Table 3 must be
matched filter-by-filter against the port's actual edge handling, because
neither side's `BoundaryCondition`-token scan will surface them.
