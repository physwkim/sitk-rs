# Boundary-condition parity audit — conclusion

**Result: no divergence.** Every filter in this port that reads a neighbor
outside the image substitutes the same value ITK does, verified filter-by-filter
against ITK source at `/home/stevek/work/ITK`. This document is the conclusion;
the evidence is four files in `bench/results/`, cited below.

The audit was run because a wrong boundary rule is the same *shape* of defect the
physical-space and CompensatedSummation sweeps mined — silent in the interior,
wrong on every edge voxel, invisible to a test that only checks an interior
patch. It was built from both ends independently (ITK source vs port code) and
merged, because either map alone is a proxy for the other.

## The ITK truth, in three shapes

The premise most people carry is wrong for this ITK checkout, and it is worth
stating first: **`ZeroFluxNeumannBoundaryCondition` (nearest in-bounds edge
value — edge replication, so the first difference across the border is zero) is
the default almost everywhere**, including `NeighborhoodOperatorImageFilter`,
the convolution/correlation family, and the finite-difference diffusion family.
`ConstantBoundaryCondition` is *not* the neighborhood default; it is an island in
mathematical morphology. Concretely:

1. **ZeroFluxNeumann (edge replication)** — gradient/Laplacian/Sobel/Derivative,
   DiscreteGaussian, Canny, Mean, Median, Convolution, NormalizedCorrelation, and
   the whole anisotropic-diffusion / curvature-flow family.
2. **Constant, value operation-specific** — the morphology island: grayscale
   dilate fills with `NonpositiveMin()`, erode with `max()` (the op's identity, so
   the border cannot change the result); binary dilate fills background, erode
   foreground.
3. **No `BoundaryCondition` object at all** — recursive IIR Gaussians (edge
   extension via boundary coefficients), box integral-image mean/sigma (window
   clipped to region), FFT padding sub-filters, and distance transforms
   (algorithmic border). A `BoundaryCondition`-token scan cannot see these; they
   were matched by mechanism.

Detail: `bench/results/boundary-condition-itk-map.md` (ITK side, every row
`.h`/`.hxx` cited) and `bench/results/boundary-port-map.md` (port side, every row
`file:line`, the substitution rule read from code not inferred).

## The seven candidate divergences, all refuted

The cross-reference produced seven hypotheses. Each was verified against both
sides — port `file:line` and ITK `.hxx:line`, with the exact border value each
produces. All seven are **NOT-REAL** (faithful port). Evidence:
`bench/results/boundary-verification-r2.md` and
`bench/results/boundary-condition-round2-verification.md`.

- **masked-FFT normalized correlation** — the sharpest candidate. ITK zero-pads
  (`ConstantPadImageFilter`, `SetConstant(0)`); the port has its own module that
  zero-initializes and scatters only in-bounds values, so it too zero-pads, and
  is correctly *absent* from the `ZeroFluxNeumannPad` convolution family. Match.
- **fft_convolution + the six deconvolutions** — the control for the above. All
  ZeroFluxNeumann-pad on both sides; masked-FFT is the lone zero-pad exception.
  The map is internally consistent *and* correct.
- **adaptive histogram equalization** — both drop out-of-image offsets and shrink
  the denominator (`MovingHistogram` `AddBoundary`); a corner pixel divides the
  surviving 4 samples by `9−5=4` on both sides. Match on the boundary axis.
- **coherence-enhancing diffusion** — both drop the tap (ITK's LBR uses an
  `OutsideBufferIndex()` skip sentinel that is never dereferenced, not the
  replicating `ZeroFluxNeumannBoundaryCondition` class). This resolves the
  same-name concern: CED's real upstream genuinely drops the tap.
- **mean** — both clamp/replicate and divide by the *full* window count; an edge
  voxel `[a,b,…]` gives `(2a+b)/3`, not `BoxMean`'s `(a+b)/2`. SimpleITK `Mean` →
  `MeanImageFilter`, confirmed. The port explicitly does not emulate `BoxMean`.
- **Gaussian interpolator** — both truncate the kernel to the buffer and
  renormalize by the surviving weight sum; gradient formulas algebraically
  identical.
- **binary morphology polarity** — ITK defaults dilate→background,
  erode→foreground. The port has *no* shared default: `boundary_to_foreground` is
  a required parameter, and every internal caller threads the ITK-correct per-op
  value. The shared-default bug shape is structurally absent.

## Watch-items (not boundary bugs — surfaced so they are not later misread)

1. **SimpleITK facade default for binary morphology.** The `sitk` facade does not
   yet expose `BinaryDilate`/`BinaryErode`. When it does, it must pass
   `boundary_to_foreground = false` for dilate and `true` for erode — the ITK
   per-operation defaults. The `sitk-filters` layer is already correct; this is a
   future-binding note so the default is not lost at the facade.
2. **Adaptive histogram equalization keeps `f64` where ITK narrows to `float32`**
   (`itkAdaptiveEqualizationHistogram.h` `using RealType = float`). This is a
   deliberate, already-documented precision deviation, orthogonal to boundary
   handling. If bit-parity on this filter is ever wanted, that is the row to
   revisit — it is not a boundary miss.
