# DRAFT — ITK issue

Target repo: `InsightSoftwareConsortium/ITK`
Labels: `type:Bug`
Status: FILED 2026-07-10 as https://github.com/InsightSoftwareConsortium/ITK/issues/6575
(no label — `--label type:Bug` was rejected for a non-maintainer account;
maintainers apply it during triage).

Suggested title:

> BUG: Findings collected while porting ~60 filters — reported together in one issue

---

Following up on #6569: while porting SimpleITK's filter surface to another language, I kept a ledger of everything in the ITK sources that looked wrong or surprising. It has grown to 30 suspected bugs and 24 quirks — too many to file individually without flooding the tracker, so I am reporting them together here for triage. Happy to split any subset into standalone issues if that is more useful; I am not planning PRs for all of these, so please treat this as raw material.

<details>
<summary>How these were found, and what was checked before filing</summary>

Each filter was reimplemented from the `.hxx` sources and the two implementations compared on synthetic inputs; every divergence was traced back to the ITK code path that caused it. Before filing, every item was re-checked against `main` at `e46eb723a5`: the cited lines re-read in context, declared types and constructor defaults verified, and each claimed consequence re-derived from the quoted code. Items whose original analysis did not survive that re-check were corrected or dropped — some of my own earlier conclusions turned out to be wrong, so I would not be surprised if a few of the remaining ones are too. Corrections welcome.

One item is flagged inline as source-analysis only (B28: non-termination identified from the loop structure, not run to a live hang).

</details>

## Suspected bugs — wrong results, NaN, or UB in live code paths

| # | Site | One-line symptom |
|---|---|---|
| B1 | `ProjectionImageFilter` | Collapsed-axis origin shift uses the axis *index*, `unsigned`; axis 0 wraps to a ~2³¹·spacing shift |
| B2 | `N4BiasFieldCorrectionImageFilter::SharpenImage` | Degenerate bin range → `0.0/0.0 = NaN` → NaN→int cast (UB) → unchecked histogram index |
| B3 | `SLICImageFilter` | Empty cluster → `0/0` NaN centroid → NaN→integer index cast (UB) |
| B4 | `WienerDeconvolutionImageFilter` | `Pf == Pn` → complex division by zero → NaN output (guard passes on inf) |
| B5 | `CannySegmentationLevelSetFunction` | Propagation weight 0 → advection generated from a never-allocated speed image |
| B6 | `GridImageSource` | Kernel loop walks only the first line per axis; origin cancels out of the pattern |
| B7 | `MinMaxCurvatureFlowFunction` (3-D) | `acos` taken on a gradient rescaled to length *r*, not 1 — wrong polar angle for radius ≥ 2 |
| B8 | `MinMaxCurvatureFlowFunction` | Neighborhood scales divide by the stencil radius but only ±1 neighbors are sampled — update r² too small |
| B9 | `AdaptiveHistogramEqualization` | Constant input → `iscale == 0` → NaN everywhere |
| B10 | `MultiLabelSTAPLEImageFilter` | Confusion-matrix increment walks into the next row on any voting tie |
| B11 | `STAPLEImageFilter` | All-background inputs → `p_denom == 0` → NaN, unguarded |
| B12 | `LabelOverlapMeasuresImageFilter` | `false_positive_error` guard tests the wrong quantity — NaN reachable |
| B13 | `ThresholdMaximumConnectedComponentsImageFilter` | Initial bisection midpoint is `(upper−lower)/2`; unsigned wrap when it lands below `lower` |
| B14 | `ScalarImageKmeansImageFilter` | `UseNonContiguousLabels` interval underflows to `0xFFFFFFFF` for > 255 classes |
| B15 | `LabelVotingImageFilter` | Undecided label `max_label+1` wraps to 0 for a UInt8 image using all 256 labels |
| B16 | `MaskedFFTNormalizedCorrelationImageFilter` | Pixel squaring runs in the input integer type — int32 overflow before the cast to real |
| B17 | `IterativeDeconvolutionImageFilter` | Output regions clobbered before `PadInput`/`CropOutput` read them — `OutputRegionMode` silently ignored |
| B18 | `FastApproximateRankImageFilter::SetRank` | Rank forwarded to only the first Dim−1 per-axis filters — last axis always median |
| B19 | `AttributeMorphologyBaseImageFilter` | `stable_sort` end iterator off by one — last raster pixel never sorted |
| B20 | `BoxSigmaImageFilter` | Radius 0 → `0.0/0.0` → NaN for every pixel |
| B21 | `LevelSetNeighborhoodExtractor` | Strict sign tests: a pixel exactly on the contour neutralizes its neighborhood; grid-aligned contour starves the outward march |
| B22 | `StructureTensorImageFilter` (AnisotropicDiffusionLBR) | Tensor smoothing `K_ρ` applied along axis 0 only (single-direction filter, `SetDirection` never called) |
| B23 | `CoherenceEnhancingDiffusionImageFilter` | Constant input → ∞ rescale → NaN tensors → negative step count via UB cast → zero iterations by accident |
| B24 | `FastMarchingUpwindGradientImageFilter` | Duplicate target entries make `AllTargets` termination unreachable |
| B25 | `Transform::ApplyToImageMetadata` | Null `GetInverseTransform()` (singular or non-linear transform) dereferenced without a check |
| B26 | `PatchBasedDenoisingImageFilter` (POISSON) | `static_cast<PixelValueType>(0.99999)` is 0 for integer types — step size collapses to 1e-5 always |
| B27 | `PatchBasedDenoisingImageFilter` | Update write-back casts an unclamped negative float to an unsigned pixel type — UB |
| B28 | `UniformRandomSpatialNeighborSubsampler` | `Search` loops forever when the search box is exactly the query point (kernel-bandwidth path) |
| B29 | `GaussianRandomSpatialNeighborSubsampler` | Negative Gaussian variate cast to `unsigned int` — UB; rejection loop relies on the wrap |
| B30 | `RegionBasedLevelSetFunction` | Reinitialization smoothing term silently degenerates to a bare Laplacian when `CurvatureWeight == 0` |

<details>
<summary><b>B1 — ProjectionImageFilter: collapsed-axis origin uses the axis index, unsigned; axis 0 wraps</b></summary>

`itkProjectionImageFilter.hxx`, `GenerateOutputInformation`:

```cpp
for (unsigned int i = 0; i < InputImageDimension; ++i)   // L74
  ...
  outSpacing[i] = inSpacing[i] * inputSize[i];           // L87
  outOrigin[i] = inOrigin[i] + (i - 1) * inSpacing[i] / 2;  // L88
```

Two independent problems on the collapsed axis (`i == m_ProjectionDimension`):

1. The shift multiplies by the **axis index** `i`, not the axis's pixel count. Centering the collapsed extent would need `(inputSize[i] - 1) * inSpacing[i] / 2`. For any axis ≥ 1 the output origin is therefore shifted by a small, plausible-looking, wrong amount — silently wrong geometry.
2. `i` is `unsigned int`, so for `m_ProjectionDimension == 0` the expression `(i - 1)` wraps to `0xFFFFFFFF` and the origin is shifted by ≈ 2³¹ · spacing. The only guard (L42-46) rejects `m_ProjectionDimension >= ImageDimension`; axis 0 is accepted.

All seven projection filters (Mean/Maximum/Minimum/Sum/Median/StandardDeviation/BinaryProjection) inherit this. The expression is unchanged since the filter was moved out of Review (`2642282b44`); every later touch was formatting.

Downstream note: SimpleITK's generated wrappers default `ProjectionDimension` to `0` and always call the setter, so every default-parameter SimpleITK projection call takes the wraparound branch.

</details>

<details>
<summary><b>B2 — N4BiasFieldCorrection::SharpenImage: NaN histogram index on a degenerate bin range (UB + unchecked indexing)</b></summary>

`itkN4BiasFieldCorrectionImageFilter.hxx:310-311`:

```cpp
const RealType     cidx = (static_cast<RealType>(pixel) - binMinimum) / histogramSlope;
const unsigned int idx = itk::Math::floor(cidx);
```

When every included voxel has the same value, `binMaximum == binMinimum`, so `histogramSlope == 0` and `cidx = 0.0 / 0.0 = NaN`. `itk::Math::floor` is `static_cast<int>(std::floor(x))` (`itkMath.h:1070-1072`) — the float→int conversion of NaN is undefined behavior ([conv.fpint]), and the resulting `idx` then indexes the histogram array with no bounds check. The same structure repeats at L453-454.

A separate quirk feeds this: the min/max scan uses an `else if` chain, so a traversal whose included voxels are strictly increasing never assigns `binMinimum` at all (it stays `NumericTraits<RealType>::max()`), which also poisons `histogramSlope`.

</details>

<details>
<summary><b>B3 — SLICImageFilter: empty cluster → NaN centroid → NaN→integer cast (UB)</b></summary>

A cluster that ends an iteration with zero members divides its accumulator by zero (`vnl_vector /= 0`), leaving NaN in the centroid's spatial components. Those components are then converted with `Math::RoundHalfIntegerUp<IndexValueType>(cluster[numberOfComponents + d])` at `itkSLICImageFilter.hxx:208`, `:354`, and `:453` — a float→integer conversion whose behavior on non-finite input is undefined (`itkMath.h:181`: "The behavior of overflow is undefined"), and the resulting index is used to read the image.

</details>

<details>
<summary><b>B4 — WienerDeconvolution: Pf == Pn → complex division by zero → NaN output</b></summary>

`itkWienerDeconvolutionImageFilter.h:166-175` (functor; `TPixel` = `std::complex<double>`):

```cpp
TPixel Pn = m_NoisePowerSpectralDensityConstant;
TPixel Pf = std::norm(I);
TPixel denominator = std::norm(H) + (Pn / (Pf - Pn));
if (itk::Math::Absolute(denominator) >= m_KernelZeroMagnitudeThreshold) { ... }
```

`Pn / (Pf - Pn)` is a genuine complex/complex division with no guard for `Pf == Pn`. With libstdc++/libgcc's `__divdc3`, `(c, 0)/(0, 0)` yields `(inf, nan)`; `Math::Absolute` of that is `inf`, which **passes** the threshold guard, so `value = I * (conj(H) / denominator)` computes NaN that propagates to the output. (The exact `(inf, nan)` result is implementation-specific to the libgcc division algorithm; the unguarded division-by-zero is not.)

`Pf == Pn` is reachable for any frequency whose input power equals the configured noise PSD constant.

</details>

<details>
<summary><b>B5 — CannySegmentationLevelSetFunction: advection image computed from a never-allocated speed image</b></summary>

`itkSegmentationLevelSetImageFilter.hxx:81-92` generates the speed image only when the propagation weight is non-zero, and the advection image independently:

```cpp
if (Math::NotExactlyEquals(...GetPropagationWeight(), 0)) { this->GenerateSpeedImage(); }
if (Math::NotExactlyEquals(...GetAdvectionWeight(), 0))   { this->GenerateAdvectionImage(); }
```

`GenerateSpeedImage()` is the only caller of `AllocateSpeedImage()`. But Canny's `CalculateAdvectionImage` (`itkCannySegmentationLevelSetFunction.hxx:41-68`) calls `CalculateDistanceImage()`, which sizes its pipeline from the speed image:

```cpp
m_Distance->GetOutput()->SetRequestedRegion(this->GetSpeedImage()->GetRequestedRegion());  // L94
```

With propagation weight 0 and advection weight ≠ 0, the speed image is a default-constructed `ImageType::New()` with an empty region. The distance/multiply chain then runs over that empty region, and `ImageAlgorithm::Copy(multiply->GetOutput(), advectionImage, advectionImage->GetRequestedRegion(), ...)` (L64-67) copies the advection image's full region from a source that never computed it — reading outside the source's buffered region.

</details>

<details>
<summary><b>B6 — GridImageSource: per-axis kernel walks only the first line; origin cancels out</b></summary>

`itkGridImageSource.hxx`, `BeforeThreadedGenerateData`:

```cpp
It.SetDirection(i);
for (It.GoToBegin(); !It.IsAtEndOfLine(); ++It)   // L67 — never advances to the next line
```

The per-axis kernel is filled from the **first line** of each axis only, so only the Direction matrix's diagonal ever feeds the pattern — any off-diagonal direction is ignored. And:

```cpp
const RealType num = point[i] - static_cast<RealType>(j - 2) * this->m_GridSpacing[i] -
                     output->GetOrigin()[i] - this->m_GridOffset[i];   // L76-77
```

`point[i]` already contains `+origin[i]` (it comes from `TransformIndexToPhysicalPoint`), so subtracting `GetOrigin()[i]` cancels it — the image origin never affects the grid pattern.

</details>

<details>
<summary><b>B7 — MinMaxCurvatureFlowFunction (3-D): acos on a gradient of length r, not 1</b></summary>

`itkMinMaxCurvatureFlowFunction.hxx`, `ComputeThreshold(Dispatch<3>, ...)`:

```cpp
gradMagnitude = std::sqrt(gradMagnitude) / static_cast<PixelType>(m_StencilRadius);  // L311
for (double & j : gradient) { j /= gradMagnitude; }   // L313-316 → vector length = r
...
if (gradient[2] > 1.0)  { gradient[2] = 1.0; }        // L318-325
if (gradient[2] < -1.0) { gradient[2] = -1.0; }
double theta = std::acos(gradient[2]);                 // L326
```

The gradient is rescaled to length `m_StencilRadius` (default 2), so `gradient[2] = r·cos θ` and `acos` returns the wrong polar angle for any radius ≥ 2 (e.g. true cos θ = 0.4, r = 2 → acos(0.8) = 36.9° instead of 66.4°). The adjacent clamp is the only thing keeping `acos` inside its domain. The 2-D path has the same rescale but never calls `acos`, so only 3-D is affected.

</details>

<details>
<summary><b>B8 — MinMaxCurvatureFlowFunction: update scales as 1/r² because scales widen but sampling doesn't</b></summary>

`SetStencilRadius` sets the finite-difference radius to `m_StencilRadius` on all axes, and `FiniteDifferenceFunction::ComputeNeighborhoodScales` divides by that radius (`itkFiniteDifferenceFunction.hxx:87`: `neighborhoodScales[i] = m_ScaleCoefficients[i] / m_Radius[i]`). But the update actually computed is `CurvatureFlowFunction::ComputeUpdate`, which samples only ±1-stride neighbors. First derivatives therefore scale as 1/r, second/cross derivatives as 1/r²; the assembled curvature update comes out **r² times smaller** than plain curvature flow. Exact only at the (non-default) radius 1 — the default radius is 2, i.e. a 4× understated update.

</details>

<details>
<summary><b>B9 — AdaptiveHistogramEqualization: constant input → NaN everywhere</b></summary>

`itkAdaptiveEqualizationHistogram.h`, `GetValue`:

```cpp
const double iscale = static_cast<double>(m_Maximum) - m_Minimum;   // L76 — no guard
const RealType u = (static_cast<double>(pixel) - m_Minimum) / iscale - 0.5;  // L80
...
return (TOutputPixel)(iscale * (sum + 0.5) + m_Minimum);            // L90
```

`m_Minimum`/`m_Maximum` are the actual image min/max (`itkAdaptiveHistogramEqualizationImageFilter.hxx:51-52`). A constant image gives `iscale == 0`, `u = 0/0 = NaN`, and NaN output at every pixel.

</details>

<details>
<summary><b>B10 — MultiLabelSTAPLE: confusion-matrix increment walks into the next row on any voting tie</b></summary>

The confusion matrix is allocated `(TotalLabelCount + 1) × TotalLabelCount` (`itkMultiLabelSTAPLEImageFilter.hxx:107-108`) and incremented with

```cpp
++(this->m_ConfusionMatrixArray[k][in.Get()][out.Get()]);   // L146
```

`out.Get()` comes from an internal `LabelVotingImageFilter`, whose undecided label defaults to `maxLabel + 1 == TotalLabelCount` (`itkLabelVotingImageFilter.hxx:76`) — one past the last valid **column**. `vnl_matrix::operator[]` is unchecked and rows are contiguous, so `CM[in][TotalLabelCount]` aliases `CM[in + 1][0]`: the increment lands in the next input-label row's column 0. Since `in + 1 <= TotalLabelCount` is still a valid row, this corrupts the statistics silently instead of crashing.

Trigger: any pixel where the seeding vote ties — e.g. two raters assigning two different labels to the same pixel.

</details>

<details>
<summary><b>B11 — STAPLEImageFilter: all-background inputs → NaN, unguarded</b></summary>

`itkSTAPLEImageFilter.hxx:161-162`: `p[i] = p_num / p_denom;` (and the `q` twin). `p_denom` accumulates the weight image W, which is seeded from the per-pixel average of the segmentations (L80-116). If every input pixel is background, the foreground test at L94 never fires, W stays all-zero, and `p[i] = 0.0/0.0 = NaN` with no guard.

</details>

<details>
<summary><b>B12 — LabelOverlapMeasures: false-positive-error guard tests the wrong quantity; degenerate values are DBL_MAX, not inf</b></summary>

`itkLabelOverlapMeasuresImageFilter.hxx:389-398`:

```cpp
if (Math::ExactlyEquals(mapIt->second.m_Source, 0.0)) { value = NumericTraits<RealType>::max(); }
else {
  auto nComplementIntersection = nVox - mapIt->second.m_Union;   // TN
  value = m_SourceComplement / (m_SourceComplement + nComplementIntersection);
}
```

The denominator is FP + TN = `m_SourceComplement + (nVox - m_Union)`, but the guard tests `m_Source`. A non-background label covering the entire image (source == target everywhere) has `m_SourceComplement = 0` and `m_Union = nVox`, so the denominator is 0 while `m_Source = nVox` sails past the guard → `0.0/0.0 = NaN`.

Secondary observation: all eleven degenerate-denominator guards in the file return `NumericTraits<RealType>::max()` (≈ 1.8e308, finite) rather than infinity, which callers may not expect from a ratio measure.

</details>

<details>
<summary><b>B13 — ThresholdMaximumConnectedComponents: initial bisection midpoint is half the span; unsigned wrap</b></summary>

`itkThresholdMaximumConnectedComponentsImageFilter.hxx:107-109`:

```cpp
PixelType midpoint = (upperBound - lowerBound) / 2;                    // half the SPAN
PixelType midpointL = (lowerBound + (midpoint - lowerBound) / 2);
PixelType midpointR = (upperBound - (upperBound - midpoint) / 2);
```

L107 is the span/2, not the midpoint (`lower + (upper - lower)/2`); it is only correct when `lowerBound == 0`. The in-loop updates (L147-148) use the correct form, so only the initial seed is defective. For unsigned pixel types with `midpoint < lowerBound` (e.g. `unsigned int` with lower = 3·10⁹, upper = 4·10⁹ → midpoint = 5·10⁸), L108's `midpoint - lowerBound` wraps mod 2³², corrupting the initial bisection bounds. (For `unsigned char/short`, integer promotion computes the subtraction in signed `int` and the corruption surfaces at the narrowing store instead.)

</details>

<details>
<summary><b>B14 — ScalarImageKmeans: UseNonContiguousLabels interval underflows for > 255 classes</b></summary>

`itkScalarImageKmeansImageFilter.hxx:112-116`:

```cpp
unsigned int labelInterval = 1;
if (m_UseNonContiguousLabels)
  labelInterval = (NumericTraits<OutputPixelType>::max() / numberOfClasses) - 1;
```

For a `uint8` output, `255 / numberOfClasses == 0` once `numberOfClasses > 255`, and `0 - 1` in unsigned arithmetic gives `labelInterval == 0xFFFFFFFF`. Subsequent `label += labelInterval` (L124) and the exclusion-region label `labelInterval * numberOfClasses` (L189) produce wrapped garbage. Nothing caps the class count — `VerifyPreconditions` (L44-47) only rejects an empty mean list.

</details>

<details>
<summary><b>B15 — LabelVoting: undecided label wraps to 0 for a UInt8 image using all 256 labels</b></summary>

`itkLabelVotingImageFilter.hxx:70-77`:

```cpp
if (this->m_TotalLabelCount > itk::NumericTraits<OutputPixelType>::max())
  itkWarningMacro("No new label for undecided pixels, using zero.");
this->m_LabelForUndecidedPixels = static_cast<OutputPixelType>(this->m_TotalLabelCount);
```

With all 256 labels of a UInt8 image in use, `TotalLabelCount == 256` and the cast produces **0** — undecided pixels become label 0, indistinguishable from genuine label-0 votes. Only a warning is emitted; there is no error path.

</details>

<details>
<summary><b>B16 — MaskedFFTNormalizedCorrelation: pixel squaring in the input integer type</b></summary>

`itkMaskedFFTNormalizedCorrelationImageFilter.hxx:202` (and `:218` for the moving image) squares via `ElementProduct<InputImageType, RealImageType>`, which instantiates `MultiplyImageFilter<InputImageType, InputImageType, RealImageType>`. The `Mult` functor (`itkArithmeticOpsFunctors.h:117-121`) computes `A * B` **in the input type** and only then casts to the real output type — so `int32` inputs whose squares exceed `INT32_MAX` (|pixel| ≥ 46341) overflow (signed overflow: UB) before the cast to double.

</details>

<details>
<summary><b>B17 — IterativeDeconvolution: output regions clobbered before PadInput/CropOutput read them</b></summary>

`itkIterativeDeconvolutionImageFilter.hxx:113-116` (in `GenerateData`):

```cpp
outputPtr->SetRequestedRegion(inputPtr->GetRequestedRegion());
outputPtr->SetBufferedRegion(inputPtr->GetBufferedRegion());
outputPtr->SetLargestPossibleRegion(inputPtr->GetLargestPossibleRegion());
```

`OutputRegionMode` is applied in `ConvolutionImageFilterBase::GenerateOutputInformation` (`itkConvolutionImageFilterBase.hxx:39-45`), which runs before `GenerateData` — and is then overwritten here. `PadInput` and `CropOutput` read the output's requested region **after** the clobber (`itkFFTConvolutionImageFilter.hxx:157`, `:468`). Net effect: `OutputRegionMode` (SAME vs VALID) is accepted and silently ignored by every iterative deconvolution filter (Landweber, ProjectedLandweber, RichardsonLucy).

</details>

<details>
<summary><b>B18 — FastApproximateRank::SetRank: last axis always median</b></summary>

`itkFastApproximateRankImageFilter.h:83-95`:

```cpp
void SetRank(float rank) {
  if (m_Rank != rank) {
    m_Rank = rank;
    for (unsigned int i = 0; i < TInputImage::ImageDimension - 1; ++i)
      this->m_Filters[i]->SetRank(m_Rank);
```

The separable mini-pipeline holds `ImageDimension` per-axis filters (`itkMiniPipelineSeparableImageFilter.h:109`), but the loop stops at `ImageDimension - 1` — the **last axis keeps the default rank 0.5** (`itkRankImageFilter.hxx:44`), i.e. it is median-filtered regardless of the caller's rank. For 1-D images the loop body never runs at all.

</details>

<details>
<summary><b>B19 — AttributeMorphology (AreaOpening/AreaClosing): stable_sort end iterator off by one</b></summary>

`itkAttributeMorphologyBaseImageFilter.hxx:123`:

```cpp
std::stable_sort(&(m_SortPixels[0]), &(m_SortPixels[buffsize - 1]), m_CompareOffset);
```

`std::stable_sort` takes a half-open range, so index `buffsize - 1` is excluded: the last raster pixel keeps its initialization value and is always processed **last** in the main loop regardless of its intensity. That breaks the value-ordering invariant the union-find relies on (`FindRoot` returns `x` for both the `INACTIVE = -1` "never visited" sentinel and the `ACTIVE = -2` root sentinel, `.h:211-223`), so a neighbor that appears already-processed by value can in fact be unvisited, its uninitialized `m_AuxData` (−1) enters the area accounting, and an undersized component at that flat index can survive a `Lambda` it should fail. The fix is `&m_SortPixels[0] + buffsize`.

</details>

<details>
<summary><b>B20 — BoxSigma: radius 0 → NaN for every pixel</b></summary>

`itkBoxUtilities.h`, `BoxSigmaCalculatorFunction`, L476 (body) and L542 (border):

```cpp
oIt.Set(static_cast<OutputPixelType>(std::sqrt((squareSum - sum * sum / pixelscount) / (pixelscount - 1))));
```

With radius 0, `pixelscount == 1`, the numerator is exactly 0, and the result is `sqrt(0.0 / 0.0)` = NaN at every pixel. No guard anywhere for `pixelscount <= 1`. (`BoxMean` divides by `pixelscount` and is unaffected.)

</details>

<details>
<summary><b>B21 — LevelSetNeighborhoodExtractor: exact-zero pixels neutralize their neighborhood; grid-aligned contour starves the march</b></summary>

`itkLevelSetNeighborhoodExtractor.hxx`, `CalculateDistance`:

```cpp
if (centerValue == 0.0) { ... m_LastPointIsInside = true; return 0.0; }   // L200-206
const bool inside = (centerValue <= 0.0);                                  // L208
...
if ((neighValue > 0 && inside) || (neighValue < 0 && !inside))            // L233 — strict
```

A pixel sitting exactly on the contour early-returns without seeding any neighbor, and a neighbor whose value is exactly 0 satisfies neither strict crossing test. For a grid-aligned contour (a full layer of exact zeros between negative interior and positive exterior), no outside trial points are ever seeded, the outward fast-march starves, and (through `ReinitializeLevelSetImageFilter`) the entire outside keeps its far-field initialization value.

</details>

<details>
<summary><b>B22 — StructureTensorImageFilter (AnisotropicDiffusionLBR): K_ρ smoothing applied along axis 0 only</b></summary>

`Modules/Filtering/AnisotropicDiffusionLBR/include/itkStructureTensorImageFilter.hxx:125-128`:

```cpp
using GaussianFilterType = RecursiveGaussianImageFilter<TensorImageType>;
typename GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
gaussianFilter->SetInput(m_IntermediateResult);
gaussianFilter->SetSigma(m_FeatureScale);
```

`RecursiveGaussianImageFilter` is the **single-direction** primitive; `SetDirection` is never called anywhere in the file, and `m_Direction` defaults to 0 (`itkRecursiveSeparableImageFilter.h:253`). The structure-tensor definition calls for isotropic smoothing of the outer-product field (K_ρ), but this smooths along axis 0 and no other — the computed tensor is anisotropic in a coordinate-dependent way, which changes every output of `CoherenceEnhancingDiffusionImageFilter`. (An x-ramp and a y-ramp input produce structurally different tensor errors.) The intended filter is presumably `SmoothingRecursiveGaussianImageFilter` or a per-axis loop.

</details>

<details>
<summary><b>B23 — CoherenceEnhancingDiffusion: constant input survives only by accident (inf → NaN → negative step count via UB cast)</b></summary>

Chain, all in `Modules/Filtering/AnisotropicDiffusionLBR`:

1. `itkStructureTensorImageFilter.hxx:154`: `m_PostRescaling = 1. / maximumCalculator->GetMaximum();` — constant input → zero gradients → zero tensors → max trace 0 → `1./0. = +inf` (rescale enabled by default: `m_Adimensionize{ true }`).
2. The scale functor multiplies the zero tensors by +inf → `0 · inf = NaN`.
3. `LinearAnisotropicDiffusionLBRImageFilter::MaxStableTimeStep` (`.hxx:362-370`) runs `MinimumMaximumImageCalculator` over the NaN coefficients; the max is seeded at `NonpositiveMin()` and updated only under `value > m_Maximum` (`itkMinimumMaximumImageCalculator.hxx:89,95`), which NaN never wins → the "maximum" is −DBL_MAX.
4. `.hxx:413-414`: `int n = ceil(m_DiffusionTime / delta);` with `delta` a tiny negative — the double→int conversion of a huge negative value is undefined behavior and in practice yields a negative `n`.
5. The Euler loop `for (auto k = 0; k < n; ++k)` runs zero times and `GraftOutput(m_PreviousImage)` passes the input through unchanged.

The output happens to be reasonable (identity) but only through a UB cast on a NaN-poisoned pipeline.

</details>

<details>
<summary><b>B24 — FastMarchingUpwindGradient: duplicate target entries make AllTargets termination unreachable</b></summary>

`itkFastMarchingUpwindGradientImageFilter.hxx`, `UpdateNeighbors`: each accepted index inserts into `m_ReachedTargetPoints` at most once (the target-matching loop `break`s on first match, L193-202), so its size tops out at the number of **distinct** target indices. The AllTargets termination test is

```cpp
if (m_ReachedTargetPoints->Size() == m_TargetPoints->Size())   // L204
```

which counts duplicates on the right side — with duplicate index entries in the target container the equality can never hold and the filter never enters target-reached mode. (`SomeTargets` compares against the user-supplied `m_NumberOfTargets` (L184) and only misbehaves when that count exceeds the distinct-target count.)

</details>

<details>
<summary><b>B25 — Transform::ApplyToImageMetadata: null GetInverseTransform() dereferenced</b></summary>

`itkTransform.hxx:458-497` (reached e.g. via `TransformGeometryImageFilter`):

```cpp
// non-linear transform: itkWarningMacro only, falls through (L462-466)
const typename Self::Pointer inverse = this->GetInverseTransform();   // L468
...
origin = inverse->TransformPoint(origin);                             // L472 — no null check
```

`GetInverseTransform()` returns null for a singular linear transform (`MatrixOffsetTransformBase::GetInverseMatrix` catches the inversion failure and sets `m_Singular`; `InvertTransform` then returns `nullptr`, `itkTransform.h:590`) and for any transform whose `GetInverse` is unimplemented — including the non-linear case the function itself warns about two lines earlier before proceeding into the same dereference. Both paths are a null-pointer dereference; a typed exception seems intended.

</details>

<details>
<summary><b>B26 — PatchBasedDenoising (POISSON): float clamp constant truncates to 0 for integer pixel types</b></summary>

`itkPatchBasedDenoisingImageFilter.hxx:2049-2051`:

```cpp
const RealValueType gradientFidelity = (inVal - outVal) / (outVal + 0.00001);
// Prevent large unstable updates when out[pc] less than 1
const RealValueType stepSizeFidelity = std::min(outVal, static_cast<PixelValueType>(0.99999)) + 0.00001;
```

For every integer `PixelValueType`, `static_cast<PixelValueType>(0.99999)` is **0**, so the `std::min` always returns 0 and `stepSizeFidelity` collapses to `0.00001` regardless of the pixel value — the POISSON fidelity step is effectively disabled for integer images. The comment shows the intent was to cap a fractional value just below 1, i.e. the clamp should happen in `RealValueType`.

</details>

<details>
<summary><b>B27 — PatchBasedDenoising: unclamped negative update cast to an unsigned pixel type (UB)</b></summary>

`itkPatchBasedDenoisingImageFilter.hxx:2078`: `updateIt.Set(static_cast<PixelType>(result));` where `result` is a floating `RealType`. The GAUSSIAN fidelity branch (L2013-2019) and the pure-smoothing path (L1993-1994) can drive `result` negative and neither clamps (only RICIAN and POISSON clamp, L2038/L2056). For an unsigned `PixelType` the float→unsigned conversion of a negative value is undefined behavior ([conv.fpint]).

</details>

<details>
<summary><b>B28 — UniformRandomSpatialNeighborSubsampler: Search never terminates when the search box is exactly the query point</b></summary>

*Identified by source analysis; not run to a live hang.*

`itkUniformRandomSpatialNeighborSubsampler.hxx:151-168` (the `!m_CanSelectQuery` branch):

```cpp
while (pointsFound < numberOfPoints) {
  ...
  index[dim] = this->GetIntegerVariate(searchStartIndex[dim], searchEndIndex[dim], queryIndex[dim]);
  if (index != queryIndex) { ...; ++pointsFound; }
}
```

The only exit is `pointsFound` reaching `numberOfPoints`, and it increments only when the drawn index differs from the query index. When the search region is the single query point, every draw returns the query index and the loop spins forever (`numberOfPoints` is 1 there, and nothing reduces it).

Reachability: `PatchBasedDenoisingImageFilter`'s kernel-bandwidth pass builds a per-axis search span of `max(nIndex, size-radius-1) - min(nIndex, radius) + 1` (`itkPatchBasedDenoisingImageFilter.hxx:1571-1579`); for an image whose every axis is exactly `2·radius + 1`, an interior query yields a single-point region, and that pass calls `CanSelectQueryOff()` then `Search()` (L1582-1584). It is gated on `KernelBandwidthEstimation`, which defaults to off. Note `GaussianRandomSpatialNeighborSubsampler` inherits this `Search` unchanged (it overrides only `GetIntegerVariate`), so it is equally affected.

</details>

<details>
<summary><b>B29 — GaussianRandomSpatialNeighborSubsampler: negative variate cast to unsigned (UB)</b></summary>

`itkGaussianRandomSpatialNeighborSubsampler.hxx:53-57`:

```cpp
do {
  const RealType randVar = this->m_RandomNumberGenerator->GetNormalVariate(mean, m_Variance);
  randInt = static_cast<RandomIntType>(std::floor(randVar));
} while ((randInt < lowerBound) || (randInt > upperBound));
```

`RandomIntType` is `unsigned int` and a normal variate is unbounded below, so `std::floor(randVar)` is negative with positive probability and the float→unsigned conversion is undefined behavior. The rejection loop's correctness currently depends on the (undefined) wrapped value falling outside `[lowerBound, upperBound]`.

</details>

<details>
<summary><b>B30 — RegionBasedLevelSetFunction: reinitialization smoothing degenerates to a bare Laplacian</b></summary>

`itkRegionBasedLevelSetFunction.hxx`, `ComputeUpdate`:

```cpp
ScalarValueType curvature{};                                             // L243 — zero-init
if ((dh != 0.) && (this->m_CurvatureWeight != ScalarValueType{})) {      // L256
  curvature = this->ComputeCurvature(...);                                // L258
}
...
if (this->m_ReinitializationSmoothingWeight != ScalarValueType{}) {      // L266 — separate if
  laplacian_term = this->ComputeLaplacian(gd) - curvature;                // L268
}
```

The reinitialization smoothing term is supposed to be `Δφ − div(∇φ/|∇φ|)`, but `curvature` is only computed under the *curvature-weight* condition. With `CurvatureWeight == 0` (or `dh == 0` at a given pixel), the term silently degenerates to a bare `Δφ` — defined behavior, wrong formula, and it couples two conceptually independent parameters.

</details>

## Reproduced quirks and inconsistencies

These were verified the same way but may be judged intended, historical, or not worth changing — listed for the record, one line each, citations available on request (I have exact file:line and analysis for every row).

<details>
<summary><b>24 verified quirks (table)</b></summary>

| # | Site | Behavior |
|---|---|---|
| Q1 | Deconvolution family | The same `KernelZeroMagnitudeThreshold` ε is compared against \|H\| by Inverse but against the \|H\|²-carrying denominator by Tikhonov/Wiener — "regularization → 0 equals inverse" fails in the band ε ≤ \|H\| < √ε. |
| Q2 | `DiscreteGaussianDerivativeImageFilter` | Never calls `FlipAxes()`, unlike `DerivativeImageFilter`/`GradientImageFilter` (which do) — odd-order derivatives carry the opposite sign vs. its siblings. Also the inter-axis intermediate is `Image<OutputPixelType>`, so integer outputs truncate per axis pass (the local comments say "real to real"). |
| Q3 | `ThresholdSegmentationLevelSetImageFilter` | The class doc (`.h:63-65`) states positive = inside — the opposite of the convention its own base class documents (`itkSegmentationLevelSetImageFilter.h:82-84`: negative inside). |
| Q4 | Segmentation level-set functions | `CurvatureSpeed` is overridden to `PropagationSpeed` only by GeodesicActiveContour and ShapeDetection; Threshold/Laplacian/Canny keep the base-class constant 1 — an undocumented asymmetry across siblings. |
| Q5 | `BinaryContourImageFilter` | Non-foreground pixels keep their original values verbatim (`.hxx:164-179`) — they are not normalized to `BackgroundValue`, so several distinct non-foreground input values all survive into the output. |
| Q6 | `SimpleContourExtractorImageFilter` | `InputForegroundValue == InputBackgroundValue` marks every foreground pixel as contour (the neighborhood scan includes the center pixel, `.hxx:84`). |
| Q7 | `BinaryReconstructionByErosionImageFilter` | Built on the complement via `BinaryNot`; `BackgroundValue` is consumed only as a self-consistent internal sentinel — the parameter has no effect on the output. |
| Q8 | `AnisotropicDiffusionImageFilter` | An unstable time step warns and proceeds; the clamp is present but commented out (`.hxx:68`). |
| Q9 | `ThresholdMaximumConnectedComponentsImageFilter` | `m_LowerBoundary` is written once in the constructor and read only by `PrintSelf` — dead configuration with no setter. |
| Q10 | `MinMaxCurvatureFlowFunction` | `SetStencilRadius` clamps to ≥ 1 (`.hxx:43`), making the `radius == 0` early-outs (`.hxx:217-220`, `276-279`) unreachable dead code. |
| Q11 | `ScalarImageKmeansImageFilter` | Overrides the estimator's defaults in `GenerateData` (`SetMaximumIteration(200)`, threshold 0.0, `.hxx:91-92`), invisibly to users; and `KdTreeBasedKmeansEstimator`'s loop tests the limit pre-increment after the refinement pass (`.hxx:327-347`) — `MaximumIteration = n` runs n+1 passes while `GetCurrentIteration()` reports n. |
| Q12 | `SLICImageFilter` | Enforce-connectivity relabels an undersized component to the previously-encountered label, whatever it is (`.hxx:544-546`); there is no final relabel/compaction pass; per-pixel distances accumulate in `float`. |
| Q13 | `BoxMean`/`BoxSigmaImageFilter` | The window crops at the image border and divides by the cropped count (`itkBoxUtilities.h:322-324`, `499-501`) — border statistics use fewer samples. A `ZeroFluxNeumannBoundaryCondition` is declared (L215/389) but never used, suggesting replication was once intended. |
| Q14 | `BinShrinkImageFilter` | Throws when a shrink factor exceeds the axis size where `ShrinkImageFilter` silently clamps to 1 (`.hxx:289-296` vs `itkShrinkImageFilter.hxx:265-267`); integer outputs round half-up while float outputs truncate (`RoundIfInteger`, `.h:135-149`). |
| Q15 | `AntiAliasBinaryImageFilter` | A constant input (even all-maximum) shifts to 0 under the `max − (max−min)/2` iso-value and the strict `> 0` sign test (`.hxx:109`; base `.hxx:605`) — the output is uniformly negative, i.e. "all inside". |
| Q16 | `ReinitializeLevelSetImageFilter` | Narrow-band mode pre-fills the whole output with `±NumericTraits<PixelType>::max()` and only overwrites band points (`.hxx:206-225`) — far-field pixels keep ±max rather than a distance-like magnitude. |
| Q17 | `FastMarchingUpwindGradientImageFilter` | `m_TargetValue` is overwritten at every accepted point in NoTargets mode; in Some/AllTargets modes the reached test is sticky, so every later accepted point — target or not — keeps advancing it; OneTarget reports the last reached target, not the first (`.hxx:154-227`). |
| Q18 | `CollidingFrontsImageFilter` | `ApplyConnectivity`'s flood fill is seeded on `SeedPoints1` alone (`.hxx:114-122`), so the forced −1e-6 at every `SeedPoints2` seed (`.hxx:92-97`) is dropped from the output unless that seed is face-connected to front 1's region. |
| Q19 | `itk::watershed::Segmenter` | Edge lists are sorted with a height-only `operator<` (`itkWatershedSegmentTable.h:73-82`); a NaN height compares "equivalent" to everything — a strict-weak-ordering violation (formal UB; benign merge-equal in libstdc++). |
| Q20 | `itkMath.h FloatAlmostEqual` | The absolute-difference test (default `0.1 · epsilon`) returns equal **before** the signbit check (`itkMath.h:334-352`) — `0.0 == -0.0`, and near-zero pairs millions of ULPs apart compare equal, partially defeating the documented ULP semantics. |
| Q21 | `MultiphaseDenseFiniteDifferenceImageFilter` | Computes the CFL time step then discards it: `timeStep = 0.08; // FIXME !!! After all this work, assign a constant !!! Why ??` (`.hxx:153-154`). |
| Q22 | Chan-Vese `UseImageSpacing` | Builds `m_ScaleCoefficients` that nothing in the Chan-Vese chain reads; the PDE divides by `m_InvSpacing` from the *feature* image unconditionally while the re-distancing Maurer map inherits the *level-set* image's spacing — the flag's only live effect is `maurer->SetUseImageSpacing` (`.hxx:227`). Two images with different spacings drive the PDE and the re-distancing with different metrics. |
| Q23 | `ScalarChanAndVeseDenseLevelSet` RMS/labeling | `m_ReinitializeCounter` defaults to 1, so the reinit branch runs every iteration, zeroes the RMS accumulator, and refills it with the re-distancing residual (`.hxx:214-248`) — `MaximumRMSError` halting compares against the wrong quantity. `CopyInputToOutput` labels `phi < 0` strictly (`.hxx:69`), so exactly-zero boundary pixels are excluded from the label. |
| Q24 | `ObjectMorphologyImageFilter` | `BeforeThreadedGenerateData`'s fill-then-conditional-copy always reduces to `output := input` (the fill sentinel is constructed to differ from `ObjectValue`, `.hxx:87-113`); `UseBoundaryCondition` is dead configuration (installed but never enabled, `.h:145` even warns "Don't forget to set UseBoundaryCondition to true!"); boundary detection is a hard-coded 3^dim box decoupled from the kernel radius (`.hxx:130-198`) — dilation is unaffected, erosion diverges substantially from true erosion (the header caveats this). |

Also: `LinearAnisotropicDiffusionLBRImageFilter`'s Selling decomposition bails out after 200 superbase flips by printing `"Warning: Selling's algorithm not stabilized."` to `std::cerr` and continuing with a possibly non-obtuse superbase — which can yield negative stencil weights. The 2-D branch guards obtuseness with a debug-only `assert`; the 3-D branch has no check at all (`.hxx:185-263`).

</details>

<details>
<summary>AI assistance</summary>

- Tool: Claude Code (claude-opus-4-8)
- Role: found the defects while cross-checking an independent reimplementation of these filters against this source; performed the pre-filing re-verification pass (re-reading every cited line on `main` at `e46eb723a5`, checking declared types and constructor defaults, re-deriving each consequence)
- Contribution: the inventory above
- Testing: source-level verification against `e46eb723a5`; behavioral claims cross-checked against the independent reimplementation's pinned tests. B28 is source analysis only (not run to a live hang) and is labeled as such inline.
- Items whose original analysis did not survive re-verification were corrected or dropped before filing.

</details>
