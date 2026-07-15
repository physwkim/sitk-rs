# Coordinate/index rounding parity audit — conclusion

**Result: real divergences found and fixed.** Unlike the boundary-condition and
convergence-criteria sweeps (which found the port already faithful), this sweep
mined genuine bugs in the conversion between physical points and voxel indices —
including a confirmed discrete-index flip — and closed them with one structural
fix plus a device-parity fix. This document is the conclusion; the evidence is
two maps in `bench/results/` and the commits cited below.

The audit was run because a wrong physical↔index conversion is the same *shape*
of defect the series targets — silent on interior voxels, wrong at the
half-integer boundary where a last-bit difference flips which voxel a point lands
in. It was built from both ends independently (ITK source vs port code) and
merged; the fix was authored on the host and then extended to the CUDA device
kernel that the host fix exposed.

## What was wrong

The rounding *function* was already faithful everywhere (`(x+0.5).floor()` =
ITK's `RoundHalfIntegerUp`). The bugs were in two families.

**Family 1 — fragmented coordinate arithmetic.** The port had **four** separate
physical↔index reimplementations that inverted *only the direction matrix* and
then divided by spacing as a separate step, in three different term-associations.
ITK instead caches two composed matrices — `IndexToPhysicalPoint = Direction ·
diag(spacing)` and its inverse (`itkImageBase.hxx:165-175`) — and does one
matrix-multiply per conversion. The divergence, verified numerically:

- **C1, the discrete-index flip.** For `point = 1.4999999999999998`, `spacing =
  3`, identity direction: the port computed `point / spacing =
  0.49999999999999994` → index **1**, where ITK reciprocal-multiplies by the
  composed inverse to get `0.4999999999999999` → index **0**. A 1-ULP difference
  that straddles the 0.5 rounding boundary and lands the point in the wrong
  voxel.
- **C2/C3, D1/D2, L3, R5/R6** — origin-fold and term-association bit divergences
  (`D·(s·i)` vs `(D·s)·i` vs `(D·i)·s`; origin-first integer fold vs origin-last
  continuous fold) across the four implementations, firing on oblique directions
  and large origins.

**Family 2 — wrong rounding function at four sites.** `metric.rs` (`mask_allows`),
`displacement.rs` (`sparse_jacobian`), `ants_correlation.rs` (neighborhood
raster), and `expand.rs` (`ExpandImageFilter`) used Rust's `.round()` (half *away*
from zero) where ITK uses `RoundHalfIntegerUp` (half *up*). At a continuous index
of exactly `−0.5`: ITK → voxel 0 (keep), port → −1 (drop).

## The fix

**Structural, per the user's decision.** A single coordinate-conversion primitive
`sitk-core::coord` composes `Direction · diag(spacing)`, inverts the *composed*
matrix (not just the direction), and does one matrix-multiply plus ITK's origin
fold — matching `itkImageBase.hxx` by construction. All four `sitk-core`
conversions and every consumer (label centroid, demons geometry, resample) route
through it; the four fragmented implementations are deleted. The four Family-2
sites now call `coord::round_half_integer_up`. Commits `f754477` (Family 1) and
`27b7016` (Family 2).

**Device parity.** The host fix was deliberately *not* reverted to match the
device; instead the exposed mismatch was fixed at source. The CUDA pyramid kernel
folded origin-last and inverted the direction only; it now folds origin-first
(`pyramid.rs cindex_of`) and inverts the composed matrix through `sitk-core::coord`
(`pyramid.rs affines()`), with a stale field-doc corrected in `resident.rs`.
Commit `20c2776`.

## Verification

- Separating-input regression tests, per boundary not per scenario: the C1 flip
  (`1.4999999999999998`/spacing 3/identity → index 0), `mask_allows −0.5` → keep
  voxel 0, oblique-direction and large-origin folds, and — for the device's
  composed-vs-direction-only inverse — a new oblique host↔device test whose
  non-vacuity is measured (4369 nearest-voxel tie flips between the two inverses).
- Gates green on `main` in both feature states: `cargo nextest run --workspace`
  3521 passed; `--features sitk-registration/cuda` 3635 passed (one intermittent
  SIGABRT under full-parallel device contention was characterized as the
  pre-existing unattributed flake — it passed 3/3 in isolation and 3635/3635 on
  re-run, and is not a coordinate regression).

## Residual, documented (§4.123)

For an **oblique** direction the composed-matrix inverse is computed with the
port's Gauss-Jordan `matrix::invert`, where ITK uses an SVD pseudo-inverse
(`itk::Matrix::GetInverse`, `itkMatrix.h:330-336`). For any diagonal or
axis-aligned geometry both yield exactly `diag(1/spacing)`, so physical↔index is
bit-identical — this is what closed the C1 flip. For a genuinely oblique direction
the two inverse algorithms can differ by ≤ a few ULP; the port matches ITK's
association and origin fold exactly, but not necessarily the last bits of the
inverse entries. No upstream defect — recorded as ledger §4.123 so a future
oblique-geometry diff against ITK is not misread as a bug.

## Evidence

- `bench/results/coord-rounding-port-map.md` — core primitive + filter/geometry
  consumers, both sides cited (rayon-core).
- `bench/results/coord-rounding-transform-map.md` — transform + registration +
  device sites, both sides cited (cuda-backend).
- Commits `f754477` (structural primitive + routing), `27b7016` (rounding
  function), `20c2776` (device parity).
