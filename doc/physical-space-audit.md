# The physical-space precondition: every multi-image entry point in `sitk-filters`, classified

**What this is.** ITK verifies that *every* image input of a filter occupies the same
physical space — `ImageToImageFilter::VerifyInputInformation`
(`itkImageToImageFilter.hxx:148-223`) walks the input list and throws *"Inputs do not
occupy the same physical space!"* unless origin, spacing and direction agree within
`m_CoordinateTolerance`/`m_DirectionTolerance`. This port enforced that at six mask sites
(§2.176) and nowhere else. Closing the gap means routing the filters that *inherit* the
check — and **not** routing the ones that upstream deliberately exempts. A wrongly-added
check throws on input a user is entitled to pass; a missing check silently computes over
mismatched grids. Both are silent failures, so neither the population nor the
classification may be guessed.

**How each row was decided.** The class's own header/`.hxx` was read. `VerifyInputInformation`
is virtual: a filter is exempt iff **it or an ancestor** overrides it without calling the
superclass. ITK has exactly 21 overriders outside the `ImageToImageFilter`/`ImageSink`/
`ProcessObject` bases; each override body was read, not inferred from the class name.
Nothing here is inferred from a sibling filter — `HistogramMatchingImageFilter` sits among
filters that verify and is itself exempt.

## The four categories

| Category | Rule | Port must |
|---|---|---|
| **VERIFIES** | Inherits `ImageToImageFilter::VerifyInputInformation` (or `ImageSink`'s, which performs the same origin/spacing/direction comparison, `itkImageSink.hxx:168-225`) unmodified. | **Check** — refuse mismatched grids. |
| **VERIFIES-AND-ADDS** | Overrides, calls `Superclass::VerifyInputInformation()` first, then adds its own checks. | **Check**, plus whatever it adds. |
| **EXEMPT** | Overrides with an empty body (or, in `TileImageFilter`'s case, an override that deliberately does *not* call the superclass). The comment usually says so outright. | **Accept** mismatched grids. Adding a check here manufactures a divergence. |
| **NOT AN INPUT** | The class inherits the verifier, but the *second image is not a pipeline input at all* — it is copied in through a non-input setter, so the verifier never sees it and no comparison happens. | **Accept** — same observable behaviour as EXEMPT, for a different reason. |

The fourth category is not a variant of EXEMPT: the class *does* verify, and if a second
image were ever wired in as an input it would be compared. The exemption is an accident of
plumbing, and it is invisible from the class's `VerifyInputInformation` — only from the
setter.

## EXEMPT — upstream explicitly does not compare grids (11 entry points)

Adding a physical-space check to any of these is a **new divergence**, not a fix.

| Port entry point | Upstream class | Citation | The override |
|---|---|---|---|
| `histogram_matching` | `HistogramMatchingImageFilter` | `itkHistogramMatchingImageFilter.h:202-209` | `VerifyInputInformation() const override {}`, with the comment *"this filter does not expect the input images to occupy the same physical space"*. The reference image is input 1 and its grid is never compared — the filter only reads its histogram. |
| `convolution` | `ConvolutionImageFilter` | `itkConvolutionImageFilterBase.h:161-162` | Empty override on the base. The kernel is input 1; a kernel is a stencil, not a co-registered image. |
| `fft_convolution` | `FFTConvolutionImageFilter` | same base | |
| `inverse_deconvolution` | `InverseDeconvolutionImageFilter` | same base (via `FFTConvolutionImageFilter`) | |
| `wiener_deconvolution` | `WienerDeconvolutionImageFilter` | same base | |
| `tikhonov_deconvolution` | `TikhonovDeconvolutionImageFilter` | same base | |
| `landweber_deconvolution` | `LandweberDeconvolutionImageFilter` | same base (via `IterativeDeconvolutionImageFilter`) | |
| `projected_landweber_deconvolution` | `ProjectedLandweberDeconvolutionImageFilter` | same base | |
| `richardson_lucy_deconvolution` | `RichardsonLucyDeconvolutionImageFilter` | same base | |
| `paste` | `PasteImageFilter` | `itkPasteImageFilter.h:176-177` | Empty override. Pasting a patch by index has no physical-space meaning. |
| `demons_registration` | `DemonsRegistrationFilter` | `itkDemonsRegistrationFilter.h:148-149` | Empty override — **and its four siblings do not have it** (see the split below). |

## VERIFIES-AND-ADDS (2 entry points)

| Port entry point | Upstream class | Citation | What it adds |
|---|---|---|---|
| `masked_fft_normalized_correlation` | `MaskedFFTNormalizedCorrelationImageFilter` | `itkMaskedFFTNormalizedCorrelationImageFilter.hxx:602-640` | Calls `Superclass::VerifyInputInformation()` (*"The superclass method checks origin, spacing, and direction. We need a few additional checks."*) then requires each mask to match its image's **size**. |
| `fft_normalized_correlation` | `FFTNormalizedCorrelationImageFilter` | derives from the above | Inherits the same override; with no masks set, the added size checks are skipped and the superclass check remains. |

## NOT AN INPUT — the verifier exists and never sees the second image (2 sites)

| Port entry point | Upstream class | Why the verifier is blind |
|---|---|---|
| `scalar_chan_and_vese_dense_level_set` | `ScalarChanAndVeseDenseLevelSetImageFilter` | `SetFeatureImage(f)` calls `this->SetInput(f)` — the **feature** image is input 0 (`itkMultiphaseFiniteDifferenceImageFilter.h:100-102`). The **initial level set** goes in through `SetLevelSet(0, ls)`, which *deep-copies* it into `m_LevelSet[0]` (`itkMultiphaseDenseFiniteDifferenceImageFilter.h:285-296`) and never registers it as an input. SimpleITK's yaml wires exactly that: `SetFunctionCount(1); SetLevelSet(0, initial); ... SetInput(feature)`. One indexed input ⇒ nothing to compare. |
| `normalized_correlation`'s **template** | `NormalizedCorrelationImageFilter` | The template is converted to an `ImageKernelOperator` by SimpleITK's `CreateOperatorFromImage` and handed to `SetTemplate` — an operator, not an input. The **mask**, by contrast, *is* input 1 (`itkNormalizedCorrelationImageFilter.hxx:30-34`, `SetNthInput(1, mask)`) and is verified. The port already gets this right: it checks the mask's grid and not the template's. |

## VERIFIES — must check (67 entry points)

Grouped by upstream base; every one of them inherits the check unmodified.

**`BinaryGeneratorImageFilter` / `TernaryGeneratorImageFilter` functors (31: 19 named + 12 generated by `binary_functor!`)** — `add`,
`add_in_place`, `subtract`, `subtract_in_place`, `multiply`, `multiply_in_place`, `divide`,
`divide_in_place`, `minimum`, `minimum_in_place`, `maximum`, `maximum_in_place` (the twelve
generated by `binary_functor!`), `modulus`, `and`, `or`, `xor`, `mask`, `mask_negated`,
`squared_difference`, `absolute_value_difference`, `atan2`, `binary_magnitude`,
`divide_floor`, `divide_real`, `pow`, `ternary_add`(+`_in_place`), `ternary_magnitude`
(+`_in_place`), `ternary_magnitude_squared`(+`_in_place`). `divide_floor`/`divide_real` are
SimpleITK-only filter types (`DivideFloorImageFilter.yaml:11`,
`DivideRealImageFilter.yaml:12`) built on `itk::BinaryFunctorImageFilter` — same rule.

**Complex compose (2)** — `real_and_imaginary_to_complex`, `magnitude_and_phase_to_complex`.

**Morphological reconstruction and geodesic morphology (6)** — `reconstruction_by_dilation`,
`reconstruction_by_erosion` (`ReconstructionImageFilter`, marker and mask both
`itkSetInputMacro`), `binary_reconstruction_by_dilation`, `binary_reconstruction_by_erosion`,
`grayscale_geodesic_dilate`, `grayscale_geodesic_erode`
(`itkGrayscaleGeodesicDilateImageFilter.hxx:47,62`: `SetNthInput(0/1, …)`).

**Labels and statistics (6)** — `connected_component` *(already checks)*,
`scalar_connected_component` *(already checks)*, `label_statistics`,
`label_intensity_statistics` (`SetNthInput(1, feature)`), `label_overlay`,
`morphological_watershed_from_markers` (`SetNthInput(1, marker)`).
`label_statistics` and `label_overlap_measures` descend from `ImageSink`, whose
`VerifyInputInformation` performs the same physical-space comparison
(`itkImageSink.hxx:168-225`) — a *different implementation*, not an exemption.

**Overlap measures (4)** — `label_overlap_measures`, `directed_hausdorff_distance`,
`hausdorff_distance`, `similarity_index`.

**Level sets (5)** — `geodesic_active_contour_level_set`, `shape_detection_level_set`,
`threshold_segmentation_level_set`, `laplacian_segmentation_level_set`,
`canny_segmentation_level_set`. `SegmentationLevelSetImageFilter::SetFeatureImage` registers
the feature image as a **named** `ProcessObject` input
(`itkSegmentationLevelSetImageFilter.h:204-213`), and named inputs are walked by the same
`InputDataObjectConstIterator` the verifier uses — so unlike Chan–Vese, these *are* compared.

**Masks and extra inputs (6)** — `masked_assign`, `masked_assign_in_place`,
`masked_assign_constant`, `stochastic_fractal_dimension` *(all already check)*,
`n4_bias_field_correction` / `n4_bias_field_correction_with_log_bias_field` — whose **mask**
already checks but whose **confidence image** (`itkSetInputMacro(ConfidenceImage, …)`,
`itkN4BiasFieldCorrectionImageFilter.h:198`) does not.

**Grids (1)** — `checker_board` (`SetNthInput(1, image2)`,
`itkCheckerBoardImageFilter.hxx:46-50`).

**Displacement fields (1)** — `invert_displacement_field`
(`itkSetInputMacro(InverseFieldInitialEstimate, …)`,
`itkInvertDisplacementFieldImageFilter.h:179`).

**PDE demons, minus `demons_registration` (4)** — `symmetric_forces_demons_registration`,
`fast_symmetric_forces_demons_registration`, `diffeomorphic_demons_registration`,
`level_set_motion_registration`. See below.

## The demons split — an upstream inconsistency, not a mistake in this table

`PDEDeformableRegistrationFilter` takes its fixed image, moving image and initial
displacement field through `itkSetInputMacro` (`itkPDEDeformableRegistrationFilter.h:119`,
`:125`, `:131`), so all three are pipeline inputs and the inherited verifier compares them.
`DemonsRegistrationFilter` overrides it away (`itkDemonsRegistrationFilter.h:148-149`).
**None of its four siblings do** — `SymmetricForcesDemonsRegistrationFilter`,
`FastSymmetricForcesDemonsRegistrationFilter`, `DiffeomorphicDemonsRegistrationFilter` and
`LevelSetMotionRegistrationFilter` all derive from `PDEDeformableRegistrationFilter`
directly, not from `DemonsRegistrationFilter`, and none carries the override. So in ITK:
`Demons` accepts a fixed and moving image on different grids, and the other four throw.
That is not a rule this port gets to rationalize; it is reproduced as found.

## Counts

| | Entry points |
|---|---|
| VERIFIES (must check) | 67 |
| VERIFIES-AND-ADDS (must check, plus size) | 2 |
| EXEMPT (must **not** check) | 11 |
| NOT AN INPUT (must **not** check) | 1 (`scalar_chan_and_vese_dense_level_set`; `normalized_correlation`'s *template* is a second site inside a filter that otherwise verifies) |
| **Total** | **81** |

Of the **69** that must check, **7** already do — `connected_component`,
`scalar_connected_component`, `masked_assign`, `masked_assign_in_place`,
`masked_assign_constant`, `stochastic_fractal_dimension`, `normalized_correlation` (its
mask) — and `n4_bias_field_correction` checks its *mask* but not its *confidence image*.
**62 entry points plus N4's confidence image are to be routed.**
