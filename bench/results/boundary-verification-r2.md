# Boundary parity — Round 2 verification (4 candidates)

Each candidate checked against **both** sides: port code (with `file:line`) and
ITK source at `/home/stevek/work/ITK`. Verdict + both-side citation + the exact
border value each side produces + a differ-or-cannot-differ proof. Report only;
no code changed.

**Result: all four verify as faithful ports (no divergence).**

| # | candidate | verdict |
|---|---|---|
| R1 | coherence-enhancing diffusion drop-the-tap | **NOT-REAL (MATCH)** |
| R2 | `mean` clamp-vs-clip | **NOT-REAL (MATCH)** |
| R3 | Gaussian interpolator truncate+renormalize | **NOT-REAL (MATCH)** |
| R4 | binary morphology default polarity | **NOT-REAL (MATCH)** |

---

## R1 — `coherence_enhancing_diffusion`: drop-the-tap on both sides — **MATCH**

**Port** — `coherence_enhancing_diffusion.rs`:
- OOB neighbor slot initialized to `-1`: `neighbors = vec![-1i64; npix*2*half]` (:610); set to a real linear index only when `inside` (:646-654).
- Diagonal row-sum gates on `if y >= 0` — OOB tap contributes to *neither* off-diagonal *nor* diagonal (:665-668).
- Update gates on `if y >= 0` (:691-695); functor `out[p]*delta + prev[p]*(1 - delta*diag[p])` (:700).

**ITK** — `itkLinearAnisotropicDiffusionLBRImageFilter.hxx`:
- OOB slot set to `OutsideBufferIndex()` (:153), which returns
  `NumericTraits<InternalSizeT>::max()` (:307-310) — a sentinel, **not a held
  value**.
- Diagonal loop gates `if (yIndex != OutsideBufferIndex())` — OOB skipped in
  both off-diagonal and diagonal (:347-353).
- Update loop gates identically (:487-493); functor
  `output*delta + input*(1 - delta*diag)` (:455-457).

**Border value each side (boundary pixel `p`, OOB neighbor present):**
both compute
`next[p] = δ·Σ_{in-bounds y} c·prev[y] + (1 − δ·Σ_{in-bounds} c)·prev[p]`.
The missing neighbor carries **zero flux**, and the diagonal drops by exactly
its coefficient so each row of `A − diag` still sums to zero. Identical bits.

**Proof they cannot differ:** ITK's `OutsideBufferIndex()` is a skip sentinel
consumed only by `!=` guards; it is never dereferenced as a buffer offset, so no
value is ever read from it. The port's `-1` is the same skip sentinel under the
same two guards. Both are true zero-flux (tap dropped), not edge replication.

*Resolves the §7 flag:* the port faithfully mirrors LBR, whose upstream
genuinely drops the tap and does **not** use ITK's `ZeroFluxNeumannBoundaryCondition`
class (which replicates). The same-name-different-bits worry does not bite here
because CED's actual upstream is the drop-the-tap LBR filter.

---

## R2 — `mean`: clamp, not clip; both divide by full window count — **MATCH**

**Port** — `denoise.rs` `pub fn mean` (:120):
- `NeighborhoodIterator::<T,_>::new(img, radius, ZeroFluxNeumannBoundaryCondition)` (:130).
- `neighborhood_size = iter.len()` — the **full** window count (:131).
- Sum over all window slots, `T::from_f64(acc / neighborhood_size)` (:151).

**ITK** — `itkMeanImageFilter.hxx`:
- Interior via `BufferedImageNeighborhoodPixelAccessPolicy` (all in-bounds), boundary
  faces via `ZeroFluxNeumannImageNeighborhoodPixelAccessPolicy` (:55-67) — edge
  **replication**.
- Divisor is `neighborhoodSize = neighborhoodOffsets.size()` — the full window
  count, in **both** subregions (:82, :100 `sum / neighborhoodSize`).

SimpleITK `Mean` → `itk::MeanImageFilter` (the port's own doc names
`MeanImageFilter`, :114). The distinct clip filter is `BoxMeanImageFilter`, which
the port explicitly documents it does **not** emulate for `mean`/`median`
(:224-232, "not `ZeroFluxNeumannBoundaryCondition`").

**Border value each side** (radius-1, 1-D, left edge of `[a, b, c, …]`):
window = {clamp(−1)=a, a, b}; mean = **(2a + b) / 3** on both sides.
`BoxMeanImageFilter` would give (a + b)/2 — that is the filter the port
correctly does *not* map `Mean` to.

**Proof they cannot differ:** both replicate the edge pixel into the sum and
divide by the fixed full window count; there is no in-bounds-count divisor on
either side.

---

## R3 — Gaussian interpolator: both truncate + renormalize by surviving weight — **MATCH**

**Port** — `interpolator.rs` `gaussian_value_and_gradient` (:401):
- Region truncated to the buffer: `begin = (…-cutoff).floor().max(0.0)`,
  `end = (…+cutoff).ceil().min(size[d])` (:419-420).
- `sum_me += v*w`, `sum_m += w` over only the truncated region (:457-458).
- `value = sum_me / sum_m` (:480) — **renormalize by surviving weight**.

**ITK** — `itkGaussianInterpolateImageFunction.hxx`:
- `ComputeInterpolationRegion`: `begin = max(region.index, floor(cindex+0.5-cutoff))`,
  `end = min(region.index+size, ceil(cindex+0.5+cutoff))` (:82-88) — same
  truncation to the buffer.
- `sum_me += V*w`, `sum_m += w` over that region (:149-150).
- `rc = sum_me / sum_m` (:160) — same renormalization.

Gradient formulas are algebraically identical: port
`(value·dsum_m − dsum_me)/sum_m · 1/(√2σ)` (:483) equals ITK
`(dsum_me − rc·dsum_m)/sum_m · 1/(−√2σ)` (:166-167).

**Border value each side:** at a sample whose cutoff overhangs the edge, both
sum only the in-buffer erf-weighted taps and divide by that truncated weight
sum — no edge tap is replicated, extended, or mirrored. Identical.

**Proof they cannot differ:** ITK does **not** clamp or extend; it truncates the
region and renormalizes by `sum_m`, exactly as the port does. (Out of scope for
the border rule: the port hard-codes `GAUSSIAN_SIGMA`/`GAUSSIAN_ALPHA`; those set
kernel width, not the edge rule — if they were to differ from SimpleITK's preset
it would shift *all* values, not just boundary ones, so it is a separate
question from this candidate.)

---

## R4 — binary morphology default polarity: per-op defaults reproduced — **NOT-REAL**

**ITK** — per-operation constructor defaults:
- `itkBinaryDilateImageFilter.hxx:36`: `m_BoundaryToForeground = false` (OOB = background).
- `itkBinaryErodeImageFilter.hxx:36`: `m_BoundaryToForeground = true` (OOB = foreground).

**Port** — `morphology.rs`:
- `binary_erode_typed`/`binary_dilate_typed` map the bool to the
  `ConstantBoundaryCondition` sentinel: `if boundary_to_foreground { foreground }
  else { background }` (:532-536, :575-579).
- There is **no single shared default** — `boundary_to_foreground` is a required
  explicit parameter on the public `binary_erode` (:614-628) and `binary_dilate`
  (:636-650). The bug shape the candidate describes (one shared default wrong for
  one op) is structurally absent.
- Every internal caller threads the ITK-correct per-op value:
  - `binary_morphological_opening`: erode `true`, dilate `false` (:665-666).
  - `binary_morphological_closing`: doc + minipipeline keep erode `true`, dilate
    `false` (:702-703).
  - bit-parity test and bench pass dilate `false` (bit_parity.rs:209,
    bench_ops/ops.rs:142).
- The `sitk` SimpleITK-facade crate does not yet expose BinaryErode/BinaryDilate
  (no call sites found), so no facade-level default exists to be wrong today.

**Border value each side** (radius-1 edge voxel):
- dilate, `false` → OOB = background: an edge pixel is painted foreground only
  if a real in-bounds neighbor is foreground (no spurious inward halo).
- erode, `true` → OOB = foreground: an edge pixel survives if all in-bounds
  neighbors are foreground (no spurious edge stripping).
The port produces exactly these; ITK produces exactly these.

**Differ-input that proves the port avoids the bug:** a solid foreground block
touching the image edge. If dilate had wrongly used `true`, every out-of-frame
neighbor would read foreground and the whole border would be painted foreground
(a one-pixel halo the true filter never adds); if erode had wrongly used `false`,
the entire edge layer would be stripped. The port passes `false`/`true`
respectively and matches ITK on both.

**Watch-item (not a divergence):** when the `sitk` facade eventually wires
`BinaryDilate`/`BinaryErode`, it must supply default `false` for dilate and
`true` for erode. This is a future-binding requirement, flagged so it is not lost
— the current `sitk-filters` layer is correct.

---

## Summary

The four sharpest cross-referenced leads — including the §7 same-name-different-bits
flag (R1) — all verify as **MATCH / faithful port**. No REAL divergence and no
NEEDS-DECISION among them. The one forward-looking note is R4's facade watch-item:
the per-operation `boundary_to_foreground` default must be supplied when the
SimpleITK-facing binding is written (dilate `false`, erode `true`).
