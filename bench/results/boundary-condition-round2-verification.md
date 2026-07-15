# BoundaryCondition round 2 — candidate verification (both-side)

Three cross-reference candidates, each verified against **port code** (my tree)
and **ITK source** (`/home/stevek/work/ITK`). Verdict per candidate: REAL /
NOT-REAL / NEEDS-DECISION, with a both-side citation, the exact border value
each side produces, and an input shape that would separate them (or a proof
they cannot). **Verification only — no code changed.**

**Result: all three are NOT-REAL (the two maps agree, and both agree with ITK).**
The one failure mode C3 was set up to catch — "the map is internally consistent
and wrong because everything, including masked, zero-flux-pads" — does **not**
occur: the port already special-cased masked-FFT-correlation into its own
module that zero-pads, so masked matches ITK precisely *because* it is not in
the `ZeroFluxNeumannPad` convolution family.

---

## C1 — `masked_fft_normalized_correlation`: zero-pad on both sides → NOT-REAL

**Hypothesis:** the port might route masked-FFT-correlation through the shared
`ZeroFluxNeumannPad` convolution path (diverging from ITK's zero-pad).

**ITK — zero-pad.** `CalculateForwardFFT` pads the image to the FFT size with a
literal-0 `ConstantPadImageFilter` on the upper bound:
- `itkMaskedFFTNormalizedCorrelationImageFilter.hxx:379` — `const typename LocalInputImageType::PixelType constantPixel = 0;`
- `.hxx:383-386` — `ConstantPadImageFilter … padder->SetConstant(constantPixel); padder->SetPadUpperBound(upperPad);`
- Masked-out interior is also zeroed first: `PreProcessImage` multiplies image by binarized mask (`.hxx:354-370`).

**Port — zero-pad, in its own module (not the convolution family).** The
docstring is explicit: *"This filter does **not** use `itkFFTPadImageFilter`"*
(`fft_correlation.rs:72`). The FFT pad is a zero-initialized buffer with values
scattered to their in-bounds positions; the upper-side pad stays 0:
- `fft_correlation.rs:200` — `let mut buf = vec![Complex::default(); total];` (0+0i everywhere)
- `.rs:202-204` — only in-bounds `values` written; pad region untouched → 0.
- Mask zeroing mirrors ITK: `pre_process_image` multiplies image by binarized mask (`.rs:267-273`).

**Exact border value.** Both sides: every padded (out-of-original-extent) cell =
`0.0` exactly. The port map's §3 `ConvolutionBoundaryCondition` (default
`ZeroFluxNeumannPad`, edge-replicate) is **not** on this path — masked-FFT-
correlation is absent from that table on purpose.

**Separating input:** none exists. For any fixed/moving pair padded to
`fft_size`, both fill the upper pad with exact `0.0`; the mask path zeroes the
same interior cells. They cannot differ on the pad rule.

**Verdict: NOT-REAL — MATCH.** (This is the sharp one the brief flagged, and it
resolves *in favor of* the port: it already did the ITK-correct thing by keeping
masked-FFT zero-pad separate from the ZeroFluxNeumann convolution family.)

---

## C2 — `adaptive_histogram_equalization`: OOB dropped + denominator shrunk on both sides → NOT-REAL

**Hypothesis:** ITK might clamp/replicate OOB window samples (port drops them),
diverging on every boundary voxel.

**ITK — drops OOB, shrinks the denominator.** Derives from
`MovingHistogramImageFilter` (`itkAdaptiveHistogramEqualizationImageFilter.h:71`),
whose per-offset loop routes out-of-region indices to `AddBoundary()`, **not** a
clamped `AddPixel`:
- `itkMovingHistogramImageFilter.hxx:58-64` — `if (inputRegion.IsInside(idx)) histogram.AddPixel(GetPixel(idx)); else histogram.AddBoundary();`
- `itkAdaptiveEqualizationHistogram.h:94-97` — `AddBoundary() { ++m_BoundaryCount; }` (the sample is discarded; only a counter moves)
- `.h:84` — denominator is the shrunk kernel: `const double ikernel = m_KernelSize - m_BoundaryCount;`
- `.h:85` — `sum += itMap->second * CumulativeFunction(u, v) / ikernel;`

**Port — identical drop + shrink.**
- `adaptive_histogram_equalization.rs:167-170` — OOB test `if n < 0 || n as usize >= size[d] { inside = false; break; }`
- `.rs:175-179` — `if inside { *map… += 1 } else { boundary_count += 1 }` (OOB dropped, not clamped)
- `.rs:181` — `let ikernel = (kernel_size - boundary_count) as f64;`
- `.rs:185` — `sum += count as f64 * cumulative_function(u, v, alpha, beta) / ikernel;`

**Exact border value.** Corner pixel, radius 1, 2D (kernel 9): 5 of 9 offsets
OOB on both sides → both drop those 5 and divide the surviving 4 contributions
by `9 - 5 = 4`. Same map, same denominator, same reconstruction.

**Separating input:** none for the boundary axis — the drop rule and denominator
are identical.

**Verdict: NOT-REAL — MATCH on the boundary mechanism.**

> Separate, already-documented deviation (NOT the boundary candidate): the port
> keeps `u`/`v`/`sum` in `f64`, while ITK narrows them to `float32`
> (`itkAdaptiveEqualizationHistogram.h:42` `using RealType = float;`). This is a
> deliberate sub-ULP precision divergence the port flags in its own docstring
> (`adaptive_histogram_equalization.rs`, "Deliberate divergence"). It is a
> *precision* choice orthogonal to boundary handling; surfaced here only so the
> merge does not later mistake it for a boundary miss. If the panel wants
> bit-parity on this filter, that f64-vs-f32 choice is the row to revisit — but
> it is not C2.

---

## C3 — `fft_convolution` + 6 deconvolutions ZeroFluxNeumann-pad on both sides → NOT-REAL (control confirmed)

**Hypothesis (control):** confirm the port's `ZeroFluxNeumannPad` default reaches
`fft_convolution` and the six deconvolutions the same way ITK does, and that
none should have been zero-pad.

**ITK — all derive from `FFTConvolutionImageFilter` → ZeroFluxNeumann pad.**
- `itkInverseDeconvolutionImageFilter.h:60` — `: public FFTConvolutionImageFilter<…>`
- `itkWienerDeconvolutionImageFilter.h:81`, `itkTikhonovDeconvolutionImageFilter.h:55` — `: public InverseDeconvolutionImageFilter<…>` (→ FFTConvolution)
- `itkIterativeDeconvolutionImageFilter.h:56` — `: public FFTConvolutionImageFilter<…>`
- `itkLandweberDeconvolutionImageFilter.h:99`, `itkRichardsonLucyDeconvolutionImageFilter.h:63` — `: public IterativeDeconvolutionImageFilter<…>`
- `itkProjectedLandweberDeconvolutionImageFilter.h:57` — `: public ProjectedIterativeDeconvolutionImageFilter<…>` (→ Iterative → FFTConvolution)
- The pad is taken from the base boundary condition = ZeroFluxNeumann:
  `itkFFTConvolutionImageFilter.hxx:264` `fftPadder->SetBoundaryCondition(this->GetBoundaryCondition());`;
  base default `itkConvolutionImageFilterBase.h:96` + `FFTPad` default `itkFFTPadImageFilter.h:98`.

**Port — one per-call `ConvolutionBoundaryCondition`, default `ZeroFluxNeumannPad`, reaching all of them.**
- `convolution.rs:82-83` — `#[default] ZeroFluxNeumannPad`
- `convolution.rs:449` — `ZeroFluxNeumannPad => pad_input_with(…)` (edge-replicate pad)
- `convolution.rs:619` — `pub fn fft_convolution(… boundary: ConvolutionBoundaryCondition …)`
- `deconvolution.rs:144-149` — every deconvolution pads via the shared `pad_input(…, boundary_condition)`; the six entry points each take `boundary_condition: ConvolutionBoundaryCondition` (`inverse` :246, `wiener` :291, `tikhonov` :329, `landweber` :444, `projected_landweber` :489, `richardson_lucy` via `iterative_deconvolution` :372).

**Exact border value.** Both sides: a padded cell takes the nearest in-bounds
edge value (ZeroFluxNeumann replicate), not 0. E.g. 1D `[a,b,c]`, radius-1
kernel: the cell left of index 0 = `a` on both sides.

**None should be zero-pad.** The only zero-pad in the FFT family is
masked-FFT-correlation (C1), which the port keeps in a separate module. So the
map is internally consistent **and** correct: masked is the lone zero-pad
exception, everything else ZeroFluxNeumann-pads on both sides.

**Verdict: NOT-REAL — MATCH (control holds; C1 is genuinely the exception, and
it too matches).**

---

## Summary

| Candidate | Port | ITK | Verdict |
|---|---|---|---|
| C1 masked-FFT-correlation | zero-pad (`fft_correlation.rs:200-204`) | zero-pad (`…MaskedFFT….hxx:379-386`) | **NOT-REAL** (match) |
| C2 adaptive-hist-eq | drop OOB, ÷(kernel−boundary) (`…equalization.rs:167-185`) | drop OOB, ÷(kernel−boundary) (`…EqualizationHistogram.h:84,94-97`) | **NOT-REAL** (match; f64-vs-f32 precision is a separate, documented deviation) |
| C3 fft_convolution + 6 deconv | ZeroFluxNeumannPad default (`convolution.rs:82-83`, `deconvolution.rs:144-149`) | ZeroFluxNeumann pad via FFTConvolution base (`itkIterativeDeconvolution….h:56` etc.) | **NOT-REAL** (control confirmed) |

No REAL divergence among the three candidates. The one item worth the panel's
attention is not a boundary bug at all: the deliberate f64-vs-f32 intermediate
precision in adaptive histogram equalization (noted under C2), which is a
documented deviation, not a cross-reference find.
