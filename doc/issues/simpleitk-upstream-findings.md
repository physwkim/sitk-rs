# SimpleITK upstream findings — issue draft

Status: FILED 2026-07-10 as https://github.com/SimpleITK/SimpleITK/issues/2625
(label "Bug Report" accepted; bare `#6575`/`#6569` refs in the intro and
summary tables were expanded to `InsightSoftwareConsortium/ITK#...` in the
filed body so they do not cross-link to SimpleITK's own issues)
Target repo: SimpleITK/SimpleITK
Verified against: SimpleITK master @ `3e193179`, ITK master @ `e46eb723a5`
Cross-references: ITK issue #6575 (filed 2026-07-10, items B1/B17/Q3)
Duplicate search 2026-07-10: no existing issues found for any item
(searched: OutputRegionMode, ProjectionDimension, InputNarrowBandwidth,
GenerateGradientImage, GradientImage, CoherenceEnhancingDiffusion,
PatchBasedDenoising, ThresholdSegmentationLevelSet, sitkSTLVectorToITK,
FastMarchingUpwindGradient, ReinitializeLevelSet, Landweber, EdgeWeight,
NoiseModel, and phrase variants). Related but distinct: #1801
(FastMarchingUpwindGradient argument regression in 2.2, closed).

---

TITLE: Findings from a source-level read of the yaml filter definitions: inert parameters and doc errors

---

Following up on ITK PR #6569 and ITK issue #6575: while porting the
SimpleITK filter surface to Rust I have been reading the yaml filter
definitions and the code generator alongside the ITK sources, and
collected some findings on the wrapping layer itself. There are enough
of them that individual issues seemed inefficient, so I have collected
them into one report — happy to split any item into its own issue if
that is more convenient.

Everything below is from reading the source (yaml + jinja templates +
ITK headers/impls); none of it was confirmed at runtime. File/line
references are against SimpleITK master `3e193179` and ITK master
`e46eb723a5`. Apologies in advance if some of these are known or
intentional.

**Summary**

*Parameters that cannot take effect through the wrapped API:*

| # | Site | Finding |
|---|---|---|
| S1 | `{Landweber,ProjectedLandweber,RichardsonLucy}Deconvolution` yamls | `OutputRegionMode` is exposed and set, but the ITK iterative base discards it (ITK #6575 B17) — dead parameter |
| S2 | all 7 `*Projection` yamls | `ProjectionDimension` defaults to `0u` and is always set — differs from ITK's own default (last axis) and makes the axis-0 origin wraparound (ITK #6575 B1) the default behavior |
| S3 | `ReinitializeLevelSetImageFilter.yaml` | `InputNarrowBandwidth` is exposed and set, but its only read site is gated on a node container settable solely via the unexposed `SetInputNarrowBand(NodeContainer*)` — inert |
| S4 | `FastMarchingUpwindGradientImageFilter.yaml` | `GenerateGradientImage` is never enabled, so the `GradientImage` measurement wraps an ITK image whose buffer was never allocated |
| S5 | `FastMarchingUpwindGradientImageFilter.yaml` | the `NumberOfTargets` → target-reached-mode mapping can never select `AllTargets` |

*Documentation errors in the yaml descriptions:*

| # | Site | Finding |
|---|---|---|
| S6 | `CoherenceEnhancingDiffusionImageFilter.yaml` | the CED and EED formula labels are swapped (the current ITK header has them right) |
| S7 | `PatchBasedDenoisingImageFilter.yaml` | doc strings say `NoiseModel` "Defaults to GAUSSIAN" and `KernelBandwidthEstimation` "Defaults to true" — the yaml's own `default:` fields (and ITK's constructor) are `NOMODEL` and `false` |
| S8 | `ThresholdSegmentationLevelSetImageFilter.yaml` | output sign convention stated as positive-inside; reproduces the ITK derived-class header text that contradicts its own base class (ITK #6575 Q3) |

*Smaller observations:*

| # | Site | Finding |
|---|---|---|
| S9 | `sitkSTLVectorToITK` (`sitkTemplateFunctions.h`) | too-short per-axis vectors throw, too-long ones are truncated (documented at the helper) — no length check at the public filter API, so e.g. a 4-element `Size` on a 3-D `RegionOfInterest` call is quietly cut to 3 |
| S10 | `ThresholdSegmentationLevelSetImageFilter.yaml` | `EdgeWeight` and the three smoothing parameters are not wrapped; with ITK's constructor default `EdgeWeight = 0` the entire smoothing branch of `CalculateSpeedImage` is unreachable through SimpleITK |

Details for each item below.

<details>
<summary><b>S1 — Iterative deconvolution wrappers: OutputRegionMode is a dead parameter</b></summary>

`LandweberDeconvolutionImageFilter.yaml`,
`ProjectedLandweberDeconvolutionImageFilter.yaml:47-52`, and
`RichardsonLucyDeconvolutionImageFilter.yaml:38-43` all expose:

```yaml
- name: OutputRegionMode
  enum:
  - SAME
  - VALID
  default: itk::simple::LandweberDeconvolutionImageFilter::SAME
  itk_type: typename FilterType::OutputRegionModeEnum
```

and the generator
(`ExpandTemplateGenerator/templates/ExecuteInternalSetITKFilterParameters.cxx.jinja:9-10`)
emits an unconditional
`filter->SetOutputRegionMode(...)` on every Execute.

On the ITK side, `IterativeDeconvolutionImageFilter::GenerateData`
overwrites the output regions before `PadInput`/`CropOutput` read them
(`itkIterativeDeconvolutionImageFilter.hxx:113-116`), so the value is
accepted and ignored for all three iterative filters — reported as
InsightSoftwareConsortium/ITK#6575 item B17. For the one-shot
`Inverse`/`Tikhonov`/`Wiener` wrappers the same yaml member works fine
(consumed in `itkConvolutionImageFilterBase.hxx:39`).

The root cause is ITK's; reporting here because the SimpleITK API
surface advertises a parameter that has no effect, and depending on how
ITK resolves it the yamls may want to drop or annotate the member.
</details>

<details>
<summary><b>S2 — Projection wrappers: ProjectionDimension defaults to 0, unlike ITK, which makes the axis-0 origin wraparound the default</b></summary>

All seven projection yamls (`Mean` :9-11, `Maximum` :12-14,
`Minimum` :12-14, `Sum` :12-14, `Median` :12-14,
`StandardDeviation` :12-14, `Binary` :11-13) carry:

```yaml
- name: ProjectionDimension
  type: unsigned int
  default: 0u
```

and the generated Execute always calls
`filter->SetProjectionDimension(this->m_ProjectionDimension)`.

ITK's own constructor default is the **last** axis
(`itkProjectionImageFilter.hxx:30`:
`m_ProjectionDimension(InputImageDimension - 1)`). So the wrapper
default silently differs from the ITK default, and — until
InsightSoftwareConsortium/ITK#6575 item B1 is resolved — axis 0 is
exactly the value where `GenerateOutputInformation` computes the
collapsed-axis origin with an unsigned `(i - 1)` wraparound
(`itkProjectionImageFilter.hxx:88`), shifting the output origin by
≈ 2³¹ · spacing. Every default-parameter SimpleITK projection call
takes that branch; a plain-ITK user who never touches the setter never
does.
</details>

<details>
<summary><b>S3 — ReinitializeLevelSet: InputNarrowBandwidth is inert</b></summary>

`ReinitializeLevelSetImageFilter.yaml` exposes `LevelSetValue`,
`NarrowBanding`, `InputNarrowBandwidth` (double, default 12.0), and
`OutputNarrowBandwidth`, and the generated code does call the real
setter `filter->SetInputNarrowBandwidth(...)`
(`itkReinitializeLevelSetImageFilter.h:100`).

But `m_InputNarrowBandwidth` is read at exactly one site,
`itkReinitializeLevelSetImageFilter.hxx:239`, inside

```cpp
if (m_NarrowBanding && m_InputNarrowBand)   // .hxx:236
```

and `m_InputNarrowBand` (a `NodeContainer*`) can only be set through
the `SetInputNarrowBand(NodeContainer*)` overload
(`itkReinitializeLevelSetImageFilter.h:122`, `.hxx:39-43`), which
SimpleITK does not wrap. Through SimpleITK it is permanently null, the
gate is never satisfied even with `NarrowBanding = true`, and
`InputNarrowBandwidth` can never affect the output.
(`OutputNarrowBandwidth` is fine — read unconditionally at `.hxx:252`.)
</details>

<details>
<summary><b>S4 — FastMarchingUpwindGradient: the GradientImage measurement wraps a never-allocated image</b></summary>

`FastMarchingUpwindGradientImageFilter.yaml` exposes the measurement

```yaml
- name: GradientImage
  ...
  custom_itk_cast: this->CastITKToImage(filter->GetGradientImage().GetPointer());
```

but nothing in the yaml ever calls `SetGenerateGradientImage(true)`,
and ITK's default is false
(`itkFastMarchingUpwindGradientImageFilter.h:313`
`bool m_GenerateGradientImage{};` — the constructor does not set it).

The gradient buffer is allocated only under that flag
(`itkFastMarchingUpwindGradientImageFilter.hxx:86-93`) and populated
only at `.hxx:141-143`; the constructor merely `New()`s the image
object (`.hxx:33`). So the SimpleITK `GetGradientImage()` measurement
returns a wrapper around a default-constructed ITK image with no
buffer and empty regions — not useful, and not obviously an error to
the caller.

(Related but distinct from the closed #1801, which was about
target-point argument passing in 2.2.)
</details>

<details>
<summary><b>S5 — FastMarchingUpwindGradient: AllTargets is unreachable</b></summary>

The `TargetPoints` member's `custom_itk_cast`
(`FastMarchingUpwindGradientImageFilter.yaml:67-69`):

```cpp
if (this->m_NumberOfTargets==0) {filter->SetTargetReachedModeToNoTargets();}
else if (this->m_NumberOfTargets==1) {filter->SetTargetReachedModeToOneTarget();}
else {filter->SetTargetReachedModeToSomeTargets(std::min<size_t>(this->m_TargetPoints.size(), this->m_NumberOfTargets));}
```

maps 0 → `NoTargets`, 1 → `OneTarget`, n > 1 →
`SomeTargets(min(n, TargetPoints.size()))`. ITK's fourth mode
`AllTargets` (`itkFastMarchingUpwindGradientImageFilter.h:45`, setter
at `.h:221-223`) is never selected, so it cannot be requested through
SimpleITK. (`SomeTargets(TargetPoints.size())` is close but not
identical if the target list contains duplicates, since
`AllTargets`/`SomeTargets` count reached points differently in
`GenerateData`.)
</details>

<details>
<summary><b>S6 — CoherenceEnhancingDiffusion yaml: CED and EED formula labels are swapped</b></summary>

`CoherenceEnhancingDiffusionImageFilter.yaml` `detaileddescription`:

- lines 89-95, heading **"Coherence Enhancing Diffusion:"** carries
  `λ_i := g(μ_i − μ_min)` with `g(s) = 1 − (1−α)·exp(−(λ/s)^m)`
  (limits `g(0)=1, g(∞)=α`);
- lines 98-104, heading **"Edge enhancing diffusion:"** carries
  `λ_i := g(μ_max − μ_i)` with `g(s) = α + (1−α)·exp(−(λ/s)^m)`
  (limits `g(0)=α, g(∞)=1`).

The current ITK header
(`itkCoherenceEnhancingDiffusionImageFilter.h:50-58`) attaches those
two formulas the other way around — EED gets `μ_i − μ_min` /
`1−(1−α)exp`, CED gets `μ_max − μ_i` / `α+(1−α)exp` — and the header
matches the code. The formula bodies are identical between yaml and
header; only the mode labels are exchanged. The yaml text appears to
be generated from an older revision of the ITK doc.
</details>

<details>
<summary><b>S7 — PatchBasedDenoising yaml: doc strings contradict the yaml's own defaults</b></summary>

`PatchBasedDenoisingImageFilter.yaml`:

- `NoiseModel` (lines 77-85):
  `default: itk::simple::PatchBasedDenoisingImageFilter::NOMODEL`, but
  the brief/detailed doc strings (lines 82, 85) say "Set/Get the noise
  model type. Defaults to GAUSSIAN."
- `KernelBandwidthEstimation` (lines 118-126): `default: 'false'`, but
  the doc strings (lines 123, 126) say "… Defaults to true."

ITK's constructor defaults agree with the yaml `default:` fields, not
the doc strings (`itkPatchBasedDenoisingBaseImageFilter.h:415`
`m_KernelBandwidthEstimation{ false }`, `:424`
`m_NoiseModel{ NoiseModelEnum::NOMODEL }`), and the current ITK header
doc says "Defaults to NOMODEL." The yaml doc strings look like a stale
copy of an older ITK doc revision.

This one is worth a doc fix on its own: a user reading the generated
docs would believe denoising runs with a Gaussian fidelity term and
bandwidth estimation by default, when by default neither is active.
</details>

<details>
<summary><b>S8 — ThresholdSegmentationLevelSet yaml: inverted output sign convention (inherited from the ITK derived header)</b></summary>

`ThresholdSegmentationLevelSetImageFilter.yaml:125-127`
(`detaileddescription`):

> Positive values in the output image are inside the segmented region
> and negative values in the image are outside of the inside region.

The ITK base class states the opposite convention
(`itkSegmentationLevelSetImageFilter.h`, OUTPUTS paragraph): "By ITK
convention, NEGATIVE values are pixels INSIDE the segmented region and
POSITIVE values are pixels OUTSIDE."

To be fair to the yaml: it faithfully reproduces the ITK
*derived-class* header (`itkThresholdSegmentationLevelSetImageFilter.h:63-64`),
which contradicts its own base class — reported as
InsightSoftwareConsortium/ITK#6575 item Q3. Once ITK settles which
statement is right, the yaml doc should follow.
</details>

<details>
<summary><b>S9 — sitkSTLVectorToITK: too-short throws, too-long truncates; no length check at the public API</b></summary>

`Code/Common/include/sitkTemplateFunctions.h:95-112`:

```cpp
if (in.size() < itkVectorType::Dimension)
{
  sitkExceptionMacro(<< "Unable to convert vector to ITK type\n" ...);
}
itkVectorType out;
for (unsigned int i = 0; i < itkVectorType::Dimension; ++i)
  out[i] = in[i];
```

The asymmetry is documented in the helper's comment (lines 91-93:
"If there are more elements … they are truncated. If less, then an
exception is generated."), so this may be intentional. Reporting it
because nothing at the public filter API re-checks the length, so the
truncation is reachable silently from every `dim_vec` parameter. E.g.
`RegionOfInterestImageFilter`: `SetSize` is a plain store
(`MemberGetSetDeclarations.h.jinja:31-32`), and Execute feeds it
straight into `sitkSTLVectorToITK`
(`RegionOfInterestImageFilter.yaml:28-30`). On a 3-D image,
`SetSize({a,b,c,d})` quietly becomes `{a,b,c}` while `SetSize({a,b})`
throws. If the truncation is intentional (it is load-bearing for
`FastMarching`-style trial points, where element `[dim]` is the seed
value), a note in the generated per-filter docs might save users some
confusion.
</details>

<details>
<summary><b>S10 — ThresholdSegmentationLevelSet: the smoothing/edge branch is unreachable through SimpleITK</b></summary>

`ThresholdSegmentationLevelSetImageFilter.yaml` wraps `LowerThreshold`,
`UpperThreshold`, `MaximumRMSError`, `PropagationScaling`,
`CurvatureScaling`, `NumberOfIterations`, `ReverseExpansionDirection` —
but not `EdgeWeight`, `SmoothingIterations`, `SmoothingTimeStep`, or
`SmoothingConductance`
(`itkThresholdSegmentationLevelSetFunction.h:124/139/154/169`).

ITK's function constructor sets `EdgeWeight = 0`
(`itkThresholdSegmentationLevelSetFunction.h:191`
`this->SetEdgeWeight(0.0);`), and both `CalculateSpeedImage` gates are
`m_EdgeWeight != 0.0` (`itkThresholdSegmentationLevelSetFunction.hxx:38`
— the anisotropic-diffusion + Laplacian setup — and `.hxx:71-79` — the
edge term in the speed image). With the default pinned at 0 and no
setter wrapped, the entire smoothing/edge feature of this filter is
unreachable through SimpleITK. Possibly intentional scope reduction;
listing it for completeness since the other three parameters it gates
are also unwrapped.
</details>

---

<details>
<summary>AI assistance disclosure</summary>

These findings come from a SimpleITK→Rust porting effort in which an AI
assistant (Claude) was used extensively for source reading and
cross-verification. Every item above was independently re-verified
against SimpleITK master `3e193179` and ITK master `e46eb723a5` before
filing, with file/line references quoted from the actual sources. No
runtime reproduction was performed; all claims are source-level.
</details>
