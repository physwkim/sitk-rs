# Coordinate / index rounding — the half-voxel-shift parity map

Report-only sweep (like the boundary and convergence sweeps): this maps every
port site that converts between **physical point**, **continuous index**, and
**discrete index**, cites the port and ITK line on each end, names the exact
rounding function and its tie-break, and records whether the direction matrix is
applied. Candidates are flagged with a *separating input* — the concrete
coordinate where port and ITK produce different bits. No fixes here.

Reference read firsthand: ITK at `/home/stevek/work/ITK`, SimpleITK at
`/home/stevek/work/SimpleITK`.

---

## 0. The two rounding facts (both ends confirmed)

**ITK's `Math::Round` = `Math::RoundHalfIntegerUp` — half toward +∞.**
`itkMath.h:200-205` (`Round` → `RoundHalfIntegerUp`), doc at `itkMath.h:176-178`:
`RoundHalfIntegerUp(1.5)==2`, `(2.5)==3`, `(-1.5)==-1`. The compiled 64-bit path
is `RoundHalfIntegerToEven_64(2*x + 0.5) >> 1` (`itkMathDetail.h:382-384`); the
generic base is `floor(x + 0.5)` (`itkMathDetail.h:108-116`). ITK
`TransformPhysicalPointToIndex` (`itkImageBase.h:476`) and the nearest-neighbor
interpolator both round through this.

**The port uses `(x + 0.5).floor()`** at every discrete-rounding site
(`sitk-core/src/image.rs:1143`, `sitk-transform/src/interpolator.rs:49`). This is
`RoundHalfIntegerUp`, **not** Rust `f64::round()` (which is half *away from zero*
and would diverge at negative half-integers). A dense ULP scan around every
half-integer in [-50, 50] plus generic fractionals found **no separating input**
between `(x+0.5).floor()` and ITK's compiled `RoundHalfIntegerToEven(2x+0.5)>>1`
form. **The rounding function is faithful.** Every divergence below is in the
*pre-rounding continuous-index arithmetic*, not the rounding step.

The one place the port reaches for `f64::round()` is
`sitk-transform/src/displacement.rs:325` — the sister panel's registration
Jacobian, noted at the boundary in §5.

---

## 1. Core primitives — `sitk-core/src/image.rs`

ITK precomputes two matrices once per geometry
(`itkImageBase.hxx:165-175`): `m_IndexToPhysicalPoint = Direction · diag(spacing)`
and `m_PhysicalPointToIndex = inverse(m_IndexToPhysicalPoint)` (a single vnl
inverse of the **composed** matrix), then applies them directly. **The port
caches no composed matrix** (`Image` holds only `spacing`, `origin`, `direction`
— `image.rs:252-254`); it re-derives the transform every call as *separate*
spacing and direction steps, inverting the **direction only**. That structural
difference is the root of C1–C3.

| # | Port method | Port file:line | ITK method | ITK file:line | Rounding | Tie-break | Direction matrix | Candidate |
|---|---|---|---|---|---|---|---|---|
| C1 | `physical_point_to_continuous_index` | image.rs:1092-1099 | `TransformPhysicalPointToContinuousIndex` | itkImageBase.h:517-532 | none (continuous) | — | inverse of **direction only**, then `/spacing` | **YES** (diagonal+) |
| C1' | `transform_physical_point_to_index` | image.rs:1139-1145 | `TransformPhysicalPointToIndex` | itkImageBase.h:465-479 | `(x+0.5).floor()` = RoundHalfIntegerUp | +∞ | via C1 | **YES** (inherits C1) |
| C2 | `continuous_index_to_physical_point` | image.rs:1080-1086 | `TransformContinuousIndexToPhysicalPoint` | itkImageBase.h:558-572 | none (continuous) | — | `Direction·(spacing⊙index)` | **YES** (oblique) |
| C3 | `transform_index_to_physical_point` | image.rs:1110-1114 | `TransformIndexToPhysicalPoint` | itkImageBase.h:592-604 | none (exact widen) | — | via C2 | **YES** (oblique / large origin) |

### C1 — reciprocal-multiply vs true division (fires even for identity/diagonal)

Port: `physical_point_to_continuous_index` inverts the direction with
Gauss-Jordan (`matrix.rs:44`), applies it, then divides element-wise by spacing:
`index[d] = unrotated[d] / spacing[d]` (image.rs:1098).
ITK: `m_PhysicalPointToIndex = inverse(Direction·diag(spacing))`; for a diagonal
geometry this is `diag(1/spacing)`, and the matrix-vector product multiplies:
`cvector[d] = (1/spacing[d]) · diff[d]`.

`(1/s)·d` and `d/s` are **not** bit-identical in IEEE-754. **Confirmed** at
`spacing=3, point=7, origin=0, identity`:
ITK `(1/3)·7 = 2.333333333333333`, port `7/3 = 2.3333333333333335` — different
bits. **This flips a discrete index** (C1'): at
`point = 1.4999999999999998, spacing=3, origin=0, identity`,
ITK continuous index `0.4999999999999999` → `floor(0.9999…9)=0`; port
`0.49999999999999994` → `floor(1.0)=1`. **ITK returns index 0, the port returns
index 1.** For oblique directions there is a second, independent divergence: the
port inverts the *direction only* by Gauss-Jordan and divides by spacing after,
whereas ITK inverts the *composed* matrix by vnl (SVD/LU) — different algorithm,
different bits.

### C2 — term reassociation `D·(s·i)` vs ITK `(D·s)·i` (oblique only)

ITK's precomposed entry is `m_IndexToPhysicalPoint[r][c] = Direction[r][c]·spacing[c]`
(one exact product, `itkImageBase.hxx:174` with diagonal `scale`), so its term is
`(D·s)·i`. The port scales first (`scaled[c]=index[c]·spacing[c]`, image.rs:1083)
then rotates, giving `D·(s·i)`. Reassociated. **Confirmed** at `D=0.6, s=3, i=7`:
ITK `(0.6·3)·7 = 12.599999999999998`, port `0.6·(3·7) = 12.6` — different bits.
For a diagonal direction `D∈{0,±1}` the products are exact and this collapses to
no divergence.

### C3 — inherits C2's term reassociation, plus an origin fold-order difference

`transform_index_to_physical_point` widens the integer index and routes through
`continuous_index_to_physical_point`, so it carries C2 (oblique term
reassociation). It **additionally** diverges in the origin fold order. ITK's
integer method `TransformIndexToPhysicalPoint` seeds the accumulator with origin
and adds terms after — `((origin + t0) + t1)` (`itkImageBase.h:598-602`) —
whereas the port (and ITK's *continuous* method, `itkImageBase.h:565-570`) sum
the terms first and add origin last — `((t0 + t1) + origin)`. **Confirmed** at
`origin=1e16, terms=1.0,1.0`: origin-first `1e16`, origin-last
`1.0000000000000002e16` — different bits. So the port's
`transform_index_to_physical_point` matches ITK's **continuous** method but not
its **integer** method; SimpleITK's `TransformIndexToPhysicalPoint` calls the
integer method. Fires only for origins whose magnitude dwarfs the spacing·index
terms (~1e16 with unit terms).

**Existing port coverage** (image.rs / lib.rs tests): rounding direction is
pinned (`transform_physical_point_to_index_rounds_half_integers_up`,
lib.rs:926-946, covers `1.5→2, -1.5→-1, 2.5→3, -2.5→-2`), and identity-geometry
round-trips are pinned (lib.rs:950-980). **None of the existing tests use a
non-power-of-2 spacing, an oblique direction, a large-magnitude origin, or the
C1' index-flip point — the pins are all in the region where port and ITK agree.**

---

## 2. Blast radius of the core primitives

Every filter that sets an output origin from a shifted index goes through C2
`continuous_index_to_physical_point` and is therefore bit-exact to ITK **for the
common axis-aligned modest-origin case**, diverging only under an oblique
direction or a huge origin:

- geometry (crop / ROI / pad / mirror-pad origins): `sitk-filters/src/geometry.rs:96,241,289,457`
- shrink / slice output origin: `shrink.rs:206`, `slice.rs:146`
- label-map-mask crop origin: `label_map_mask.rs:388`
- image sources (Gaussian/Gabor/grid physical-point generators): `sources.rs:178,266,526,601`
- fft correlation origin: `fft_correlation.rs:498`
- registration center-of-geometry: `sitk-registration/src/initializer.rs:119,143`, `centered_versor.rs:173` *(sister panel's crate; noted, not owned)*
- b-spline grid physical points: `sitk-transform/src/bspline.rs:138` *(sister panel)*

`physical_point_to_continuous_index` (C1) callers: `demons/compose.rs:105`
(via the demons `Geometry`, §3).

---

## 3. Demons `Geometry` — a second, independent re-implementation (`sitk-filters/src/demons/geometry.rs`)

The demons filters do **not** call the core primitives; they carry a private
`Geometry` that re-derives the same transforms, caching only `inverse_direction`
(geometry.rs:35-46). It reproduces C1 and adds a *third* association for the
forward map.

| # | Port method | Port file:line | Mirrors ITK | Rounding | Direction matrix | Candidate |
|---|---|---|---|---|---|---|
| D1 | `physical_point_to_continuous_index` | geometry.rs:72-91 | `TransformPhysicalPointToContinuousIndex` | none | inverse of direction only, then `/spacing` (line 88) | **YES** (same as C1, diagonal+) |
| D2 | `index_to_physical_point` | geometry.rs:54-69 | `TransformIndexToPhysicalPoint` | none | `d · idx · spacing` left-to-right = `(D·i)·s` (line 64) | **YES** (oblique) |
| D3 | `is_inside_buffer` | geometry.rs:103-108 | `ImageFunction::IsInsideBuffer` | none | `[-0.5, end+0.5)` half-open | NO (bounds faithful; fed by D1) |

### D2 — a *third* association, diverging from both ITK and the core

Line 64 evaluates `d * idx as f64 * spacing` left-to-right as `(D·i)·s`.
**Confirmed** at `D=0.6, i=7, s=3`: ITK `(D·s)·i = 12.599999999999998`, core
`D·(s·i) = 12.6`, demons `(D·i)·s = 12.600000000000001` — **three different
values**. So for oblique geometries the demons forward map disagrees with *both*
ITK and the port's own core primitive. (Diagonal direction: all three exact.)

### D3 — containment bounds are faithful

`cindex[d] >= -0.5 && cindex[d] < end + 0.5` with `end = size-1`
(geometry.rs:106-107) exactly matches `ImageFunction`'s half-open
`[start-0.5, end+0.5)` (`itkImageFunction.hxx:63-64`): lower inclusive, upper
exclusive, negation-of-positive so `NaN` reports outside. The only exposure is
that the continuous index it tests came from D1's diverging arithmetic, so a
point within ~1 ULP of a boundary can classify differently.

---

## 4. Interpolators + resample (`sitk-transform/src/interpolator.rs`, `resample.rs`)

These physically live in `sitk-transform` (the sister panel's crate) but are the
filter/geometry consumers this sweep was asked to cover, so they are mapped here.

| # | Site | Port file:line | ITK file:line | Port rounding | ITK rounding | Match? |
|---|---|---|---|---|---|---|
| R1 | nearest: c-index → nearest index | interpolator.rs:49 | itkImageFunction.h:202-206 → itkIndex.h `CopyWithRound` → Math::Round | `(cindex+0.5).floor()` (+∞) | RoundHalfIntegerUp (+∞) | **exact** |
| R2 | linear: base index | interpolator.rs:71-73 | itkLinearInterpolateImageFunction.hxx:44 | `.floor()` (−∞) | `Math::Floor` (−∞) | **exact** |
| R3 | containment (c-index in buffer) | interpolator.rs:29-34 | itkImageFunction.hxx:63-64 | `c>=-0.5 && c<size-0.5` | `[start-0.5, end+0.5)` | **exact** |
| R4 | linear neighbour clamp | interpolator.rs:83 | itkLinearInterpolateImageFunction.hxx:79,86 | `.clamp(0,size-1)` | `min/max` to `[0,size-1]` | **exact** |
| R5 | resample: out index → phys | resample.rs:255,268 | itkResampleImageFilter.hxx:392 (`TransformIndexToPhysicalPoint`) | `(D·s)·i`, origin **last** | `(D·s)·i`, origin **first** | term exact; **origin fold differs** |
| R6 | resample: in phys → c-index | resample.rs:257,271 | itkResampleImageFilter.hxx:398 (`TransformPhysicalPointToContinuousIndex`) | `D⁻¹/s` (Gauss-Jordan inverse, per-row `/s`) | `inverse(D·diag s)` (vnl) | diagonal **exact**; **oblique differs** |
| R7 | resample: containment gate | resample.rs:273 | itkResampleImageFilter.hxx:402 | via R3 | `IsInsideBuffer` | **exact** |

Key contrast with the core: resample precomposes `index_to_physical_matrix`
(`interpolator.rs:716-724`, `m[r][c]=direction[r][c]*spacing[c]`) exactly as ITK
does, so R5's *term* is `(D·s)·i` — matching ITK, **unlike** C2. And R6's
`physical_to_index_matrix` (`interpolator.rs:728-741`) stores `1/spacing` in the
matrix (`inv[r][c]/spacing[r]`) then multiplies, so for a diagonal geometry it
computes `(1/s)·d` — matching ITK, **unlike** C1's `d/s`. So **resample is
bit-exact to ITK for axis-aligned images while the core primitives are not** —
a genuine intra-port asymmetry worth a fix decision.

Residual resample candidates:
- **R5 origin fold**: `affine_apply` (interpolator.rs:744-747) adds origin after
  the mat-vec sum — `(t0+t1)+origin` — while ITK's `TransformIndexToPhysicalPoint`
  seeds with origin — `((origin+t0)+t1)`. Confirmed divergent at large origins
  (same mechanism as C3). Separating input: `origin≈1e16` with unit terms.
- **R6 oblique**: `D⁻¹/s` (direction-only Gauss-Jordan inverse, divide after) vs
  ITK's single vnl inverse of the composed matrix. Diverges for oblique
  directions; diagonal is exact.

---

## 5. Label geometry / moments / centroid — a *fourth* re-implementation (`sitk-filters/src/label_shape.rs`)

Label shape/statistics do **not** call the core primitives either; they carry
`continuous_index_to_physical` (label_shape.rs:275-291) and `index_to_physical`
(label_shape.rs:294-306), shared by `label_intensity.rs` center-of-gravity
(:381). This one is the closest to ITK: line 286 is
`acc += direction[i*dim+j] * spacing[j] * idx[j]`, i.e. term `(D·s)·i` — **ITK's
exact term form** — and line 284 seeds `acc = origin[i]` — **origin first**.

| # | Port method | Port file:line | ITK method | ITK file:line | Term | Origin fold | Candidate |
|---|---|---|---|---|---|---|---|
| L1 | `index_to_physical` (2nd moments) | label_shape.rs:294-306 | `TransformIndexToPhysicalPoint` | itkShapeLabelMapFilter.hxx:233,252 | `(D·s)·i` | origin **first** | **NO — bit-exact to ITK** |
| L2 | center-of-gravity (index→phys) | label_intensity.rs:381 | `TransformIndexToPhysicalPoint` | itkStatisticsLabelMapFilter.hxx:138 | `(D·s)·i` | origin **first** | **NO — bit-exact to ITK** |
| L3 | shape centroid (continuous→phys) | label_shape.rs:554 → :275-291 | `TransformContinuousIndexToPhysicalPoint` | itkShapeLabelMapFilter.hxx:298 | `(D·s)·i` | origin **first** | **YES** (large origin) |

**L3** is the subtle one agent-missed: the centroid is a *fractional* continuous
index, and ITK converts it with the **continuous** method, which adds origin
**last** (`itkImageBase.h:565-570`). The port converts it origin-**first**
(label_shape.rs:284). Term matches ITK, but the origin fold differs — the same
large-origin mechanism as C3/R5. Separating input: `origin≈1e16` with unit-scale
centroid terms. (For the second-moment sites L1/L2, ITK uses the integer method,
which *is* origin-first, so those are exact.)

## 6. Families with no rounding divergence (both ends verified)

- **Distance maps** (Danielsson `distance.rs:420-521`, Signed-Maurer
  `distance.rs:300-331`, Approximate `distance.rs:918`): all math is in **integer
  index / offset space**, spacing applied as `integer_offset × spacing` per axis;
  no direction, no `Transform*`, no rounding. ITK
  (`itkDanielssonDistanceMapImageFilter.hxx`,
  `itkSignedMaurerDistanceMapImageFilter.hxx`) likewise — grep returns only the
  `GetSpacing()` cache line. ASDM max distance truncates via `sqrt(Σsize²) as u64`,
  matching ITK's `static_cast<SizeValueType>`. **No divergence.**
- **Morphology ball structuring element** (`morphology.rs:257-275`): inclusion
  test `Σ_d (o[d]/(r[d]+0.5))² <= 1.0`, inclusive `<=`. ITK
  (`itkFlatStructuringElement.hxx:978-988` +
  `itkEllipsoidInteriorExteriorSpatialFunction.hxx:47-50`) samples at
  `contIndex = index+0.5`, center `radius+0.5`, normalization `0.5·(2r+1)=r+0.5`;
  the `+0.5`s cancel to the same `(o/(r+0.5))² <= 1` with inclusive `<=`. **Exact
  match** (the radius-1-ball = full 3×3 cube consequence that Danielsson dilation
  relies on holds identically).
- **Region / buffer containment**: discrete `i >= 0 && i < size`
  (`boundary.rs:101`, `image.rs:949`) matches ITK `ImageRegion::IsInside(Index)`
  (half-open). Continuous containment `[-0.5, size-0.5)` with upper-**exclusive**
  `<` (interpolator.rs:29-34, demons/geometry.rs:106-107) matches
  `ImageFunction::IsInsideBuffer` (`itkImageFunction.hxx:63-64`). **Not a
  divergence, but a latent trap:** ITK's *other* continuous test,
  `ImageRegion::IsInside(ContinuousIndex)` (`itkImageRegion.h:295`), uses
  **inclusive `<=`** on both bounds — closed `[-0.5, size-0.5]`. The port has no
  equivalent of that method; any future port code needing `ImageRegion`
  continuous semantics must use `<=` (a point at exactly `size-0.5` is inside for
  `ImageRegion` but outside for `ImageFunction`).

## 7. The fragmentation — one transform, four implementations

The port has **four** independent index→physical implementations, in three term
associations and two origin folds. None is wrong for axis-aligned modest-origin
geometry; they diverge — from ITK and *from each other* — only under oblique
directions or large origins.

| Impl | File | Term | Origin fold | Matches ITK when |
|---|---|---|---|---|
| core `continuous_index_to_physical_point` | image.rs:1080 | `D·(s·i)` | last | diagonal direction (any origin) |
| demons `index_to_physical_point` | demons/geometry.rs:54 | `(D·i)·s` | last | diagonal direction (any origin) |
| label_shape `continuous_index_to_physical` | label_shape.rs:275 | `(D·s)·i` | **first** | integer-index method exactly; centroid diverges at large origin |
| resample `index_to_physical_matrix`+`affine_apply` | interpolator.rs:716,744 | `(D·s)·i` | last | continuous method exactly; output-index-map diverges at large origin |

ITK itself uses `(D·s)·i` throughout (precomposed matrix) with **two** origin
folds: integer method origin-first (`itkImageBase.h:598`), continuous method
origin-last (`itkImageBase.h:565`). Only the port's `(D·s)·i` impls (label_shape,
resample) reproduce ITK's term; the core and demons reassociate. A single shared
primitive that (a) precomposes `Direction·diag(spacing)` and its composed inverse
and (b) matches ITK's per-method origin fold would collapse all of C1–C3, D1–D2,
L3, R5–R6 at once — the structural fix for the eventual fix round.

---

## 8. Boundary — sister panel (noted, not mapped)

- `sitk-transform/src/displacement.rs:325` — `(ci.round() as isize).clamp(...)`
  in `DisplacementFieldTransform` Jacobian sampling. **This is the one
  `f64::round()` (half away from zero) in the transform crate** and diverges from
  ITK's `RoundHalfIntegerUp` (half toward +∞) at negative exact halves:
  `ci=-1.5` → Rust `-2`, ITK `-1`; `ci=-0.5` → Rust `-1`, ITK `0`. The surrounding
  code asserts the sample lands on a grid point (`frac=0`) so the tie is not
  normally exercised, but the convention is wrong. **Sister panel's to verify.**
- Registration / b-spline call sites of the core primitives (§2) inherit C1–C3
  through the shared `sitk-core` methods. **Sister panel owns the fix call.**

---
