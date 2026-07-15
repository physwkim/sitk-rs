# Coordinate / index rounding parity map — transform + registration + device

Scope: `sitk-transform`, `sitk-registration`, and the `sitk-cuda` device
coordinate path. One row per site that converts between physical point,
continuous index, and discrete index, or applies a transform's point map.
Report-only round (verify + fix later, same as the boundary and convergence
sweeps). ITK read firsthand at `/home/stevek/work/ITK`; SimpleITK at
`/home/stevek/work/SimpleITK`.

## The sharp trap, stated once

ITK's physical→index and nearest-neighbour rounding is
`Math::RoundHalfIntegerUp` = `floor(x + 0.5)`, ties toward **+∞**
(`itkMath.h:176-178`, `itkMathDetail.h:110-116`,
`itkImageBase.h:476`): `1.5→2`, `2.5→3`, **`-1.5→-1`, `-0.5→0`**.

Rust `f64::round` (and CUDA C `round()`) is half **away from zero**:
`1.5→2`, `2.5→3`, but **`-1.5→-2`, `-0.5→-1`**.

They agree on every non-tie and on every non-negative tie; they disagree
only on **negative half-integers**. So the entire trap reduces to: *is a
continuous-index component ever exactly `k + 0.5` for negative `k`, at a
site that feeds a discrete decision?* The only such value reachable inside
the shared inside-buffer window `[-0.5, size-0.5)` is exactly `-0.5`.

The core `sitk-core` `Image` physical↔index primitives are the sister
panel's; this map cites which primitive each call site relies on but does
not re-map it. The relevant one:

| core primitive | file:line | rounding |
|---|---|---|
| `continuous_index_to_physical_point` | `sitk-core/src/image.rs:1080` | none (continuous) |
| `physical_point_to_continuous_index` | `sitk-core/src/image.rs:1092` | none (continuous) |
| `transform_index_to_physical_point` | `sitk-core/src/image.rs:1110` | none (widen i64→f64) |
| `transform_physical_point_to_index` | `sitk-core/src/image.rs:1139-1143` | **`(x+0.5).floor()` = RoundHalfIntegerUp** ✓ |

The core primitive is correct (half-up). The findings below are all sites
that **re-implement** rounding instead of routing through it.

---

## 1. sitk-transform — transform point maps

### 1a. Matrix-offset family (Affine / Euler2D+3D / Versor / VersorRigid3D / Similarity2D+3D / ScaleVersor3D / ScaleSkewVersor3D / ComposeScaleSkewVersor3D / Translation)

| item | port | ITK | rounding | matrix applied? |
|---|---|---|---|---|
| `TransformPoint` | `transform.rs:471-475` (and `:774,:982,:1287,:1531,:1812,:2115,:2433,:2715`) | `itkMatrixOffsetTransformBase.hxx:129` `m_Matrix*point + m_Offset` | none — pure `mat_vec(M,p)+b` | **yes, not folded**; centre pre-folded into offset (`ComputeOffset`, `:615-631`) |
| `Translation::TransformPoint` | `transform.rs:329-336` `p[d]+t[d]` | `itkTranslationTransform.hxx:124` `point + m_Offset` | none | synthesized identity matrix (IEEE-exact, pinned by `translation_is_bitwise_the_identity_matrix_form`) |

These are the point maps the device replays. `point_map_stages` returns
exactly one `MatrixOffsetMap` (`matrix_offset.rs:134`), and
`replay_stages` (`matrix_offset.rs:102`) reproduces `transform_point` **on
the bits** (`the_matrix_offset_family_is_bitwise_its_stored_matrix_and_offset`,
`the_stages_replay_transform_point_on_the_bits`). No coordinate rounding
happens in the map itself — the rounding is downstream, in whichever
discrete consumer reads the continuous index.

### 1b. Centred scale family (refused, not approximated)

| item | port | ITK | why refused |
|---|---|---|---|
| `ScaleTransform` / `ScaleLogarithmicTransform` | `transform.rs:2848-2853,:2971-2973` `(p−c)·s+c` | `itkScaleTransform.hxx:114` `(point[i]−center[i])*m_Scale[i]+center[i]` | `(p−c)s+c` rounds per point; folded `M·p+b` (`M=diag(s)`, `b=c−sc`) rounds `b` once — differs in the last bits (`folding_a_scale_transform_into_a_matrix_changes_the_bits`). `point_map_stages` → `None`, device refuses **by name**. |

### 1c. Composite

| item | port | ITK | rounding |
|---|---|---|---|
| `TransformPoint` | `composite.rs:144-149` iterate `.rev()` | `itkCompositeTransform.hxx:66-68` `rbegin()..rend()` | none; **reverse queue order** (last-added applies first) — matches ITK exactly |
| `point_map_stages` | `composite.rs:160-166` concat in `.rev()` order | — | stages handed over **in order, not folded**; a composite has no single bitwise matrix (`a_composite_hands_over_stages_and_refuses_to_be_folded`) |

### 1d. BSpline — physical→continuous grid index → floor

| item | port | ITK | rounding | classification |
|---|---|---|---|---|
| physical→continuous grid index | `bspline.rs:488-497` `phys_to_index·(p−grid_origin)` | `itkBSplineTransform.hxx:520-522` `TransformPhysicalPointToContinuousIndex` | none (continuous) | matches |
| support-region start | `bspline.rs:537` `(idx + 0.5 − SPLINE_ORDER/2).floor()` | `itkBSplineInterpolationWeightFunction.hxx:68` `Math::Floor(index + 0.5 − SplineOrder/2.0)` | **floor** (toward −∞) | **matches ITK exactly** ✓ |
| `inside_valid_region` far-edge snap | `bspline.rs:504-518` (`:508-513`) | ITK's ULP-exact `InsideValidRegion` snap | epsilon (`4·ε·max`) approximation | **CANDIDATE** — the code itself flags "Epsilon approximation of ITK's ULP-exact boundary snap." A point whose continuous grid index lands within `~4 ulp` of `gridSize − ½(order−1) − 1` is snapped inside by an ε-nudge rather than ITK's exact snap; classification `differs by ≤ a few ulp` at the far valid-region boundary. Separating input: a physical point mapping to a grid index within a few ulp of the per-axis far limit. |

### 1e. DisplacementField — physical→continuous field index

| item | port | ITK | rounding | classification |
|---|---|---|---|---|
| physical→continuous field index | `displacement.rs:161-170` | `itkDisplacementFieldTransform.hxx:87-91` `TransformPhysicalPointToContinuousIndex` | none (continuous) | matches |
| `is_inside` gate | `displacement.rs:205` → `interpolator::is_inside` | interpolator `IsInsideBuffer` | `[-0.5, size-0.5)` half-open | matches |
| linear base (`corner_weights`) | `displacement.rs:181` `cindex[d].floor()` | `itkVectorLinearInterpolateImageFunction.hxx:47` `Math::Floor(index)` | **floor** | **matches ITK** ✓ |
| **`sparse_jacobian_wrt_parameters` owning-pixel index** | **`displacement.rs:325` `ci.round()`** | ITK's local-support metric uses the integer **virtual index** directly (`ComputeParameterOffsetFromVirtualIndex`), no continuous-index rounding | **Rust `.round()` = half-away-from-zero** | **CANDIDATE** — differs from RoundHalfIntegerUp at negative half-integers. Reachable only if a metric evaluates the sparse Jacobian at a non-grid point whose continuous field index has a component exactly `-0.5`; the normal local-support path samples on grid points (`frac = 0`, no tie), so low reachability. Separating input: field continuous index component `= −0.5`. |

### 1f. Resample (`ResampleImageFilter`)

| item | port | rounding |
|---|---|---|
| output index → physical | `resample.rs:267-268` `affine_apply(D·diag(spacing), index, origin)` | none (continuous) |
| transform | `resample.rs:269` `transform.transform_point` | per §1a–1e |
| input physical → continuous index | `resample.rs:271` `mat_vec(in_phys_to_index, diff)` | none (continuous) |
| sample | `resample.rs:273` `sampler.sample(cindex)` | rounding is **inside the interpolator only** (§1g) |

End-to-end continuous; the only discrete decision is the interpolator's.

### 1g. Interpolator kernels (`interpolator.rs`, shared by resample + metric)

| kernel | port | ITK | rounding | classification |
|---|---|---|---|---|
| `is_inside` | `:29-34` `c ∈ [-0.5, size-0.5)` | `itkImageFunction` IsInsideBuffer | half-open | matches |
| `nearest_at` | `:49` `(cindex[d]+0.5).floor()` then clamp | `itkNearestNeighborInterpolateImageFunction.h:88` → `ConvertContinuousIndexToNearestIndex` → `Index::CopyWithRound` → `Math::Round` = RoundHalfIntegerUp | **`(c+0.5).floor()` = RoundHalfIntegerUp** | **matches ITK** ✓ |
| `linear_at` / `linear_value_and_gradient` | `:71,:117` `cindex[d].floor()` | `itkLinearInterpolateImageFunction.hxx:44` `Math::Floor` | floor | matches ✓ |
| `bspline_value_and_gradient` base | `:299` `cindex[d].floor() − 1` | `itkBSplineInterpolateImageFunction` | floor | matches |
| `gaussian_value_and_gradient` region | `:419-420` `(c+0.5−cutoff).floor()` / `(c+0.5+cutoff).ceil()` | `itkGaussianInterpolateImageFunction` | floor/ceil | matches |
| `windowed_sinc` base | `:658` `cindex[d].floor()` | `itkWindowedSincInterpolateImageFunction` | floor | matches |

---

## 2. sitk-registration — metric point sampling

ITK metricv4 order (`itkImageToImageMetricv4.hxx`): virtual point →
`TransformPoint` → **mask check on the ROUNDED index** (`:356`
`IsInsideInWorldSpace`) → interpolator `IsInsideBuffer` on the continuous
index (`:364`) → `Evaluate` (`:369`). The mask uses
`itkImageMaskSpatialObject.hxx:40` `TransformPhysicalPointToIndex` →
`RoundHalfIntegerUp` (**half-up**, `itkImageBase.h:476`). The interpolator's
inside test uses the un-rounded continuous index.

| item | port | ITK | rounding | classification |
|---|---|---|---|---|
| virtual index → physical fixed point | `metric.rs:1093,:1133,:1197` `t.transform_point(fp)` where `fp` from `VirtualGrid::point` | `itkImageToImageMetricv4.hxx:298` `LocalTransformPoint` | none (continuous) | matches |
| fixed sample → moving physical point | `metric.rs:1093` etc `transform.transform_point(fp)` | `:348` `TransformPoint` | none | matches |
| moving physical → continuous index | `metric.rs:795-807` `M·(p−origin)` | interpolator `ConvertPointToContinuousIndex` | none (continuous) | matches |
| moving inside-buffer | `metric.rs` via `interpolator::is_inside` | `:364` `IsInsideBuffer` (continuous) | half-open `[-0.5,size-0.5)` | matches |
| moving interpolation | `metric.rs:972,:1005` linear=floor / nearest=RoundHalfIntegerUp | `:369` `Evaluate` | per §1g | matches |
| **moving mask lookup** | **`metric.rs:819` `cd.round()`** then reject if `<0` or `≥size` | `:356` → `TransformPhysicalPointToIndex` = **RoundHalfIntegerUp** | **Rust `.round()` = half-away-from-zero** | **CANDIDATE (primary)** — see below |
| fixed mask | `metric.rs:494-540` `FixedSamples::from_image_with`, gated by **integer grid raster index** | `:305` mask at `mappedFixedPoint` (rounded) | none (exact grid index) | matches when fixed==virtual domain (no fixed-initial transform); if a fixed-initial transform reshapes the fixed point, the port gates by grid index while ITK rounds the mapped point — a *modeling* difference with no rounding, noted not filed |

### CANDIDATE (primary): moving-mask rounding is half-away, ITK is half-up

- Port host `MovingImage::mask_allows` (`metric.rs:819`) rounds the moving
  continuous index with Rust `cd.round()` (half **away** from zero) to pick
  the mask voxel. ITK's `ImageMaskSpatialObject::IsInsideInObjectSpace`
  (`itkImageMaskSpatialObject.hxx:40`) rounds with
  `TransformPhysicalPointToIndex` = `RoundHalfIntegerUp` (half **up**).
- The in-code comment claims "matching ITK's `ImageMaskSpatialObject`
  point-in-mask test," but the tie-break direction is wrong for negative
  half-integers.
- **Separating input:** a moving continuous-index component exactly `−0.5`
  with a moving mask present and voxel 0 admitted. There `is_inside`
  passes (`−0.5 ≥ −0.5`), ITK rounds to `0` and reads `mask[0]` (keep),
  the port rounds to `−1` and rejects the sample (drop). Reachable on
  commensurate geometry where a fixed sample maps exactly onto a moving
  half-voxel boundary. All other negative half-integers (`−1.5`, …) fall
  outside the inside-buffer window and are dropped regardless, so `−0.5`
  is the sole separating value.
- **Structural fix (later round):** route the mask lookup through the core
  `transform_physical_point_to_index` (which already does `(x+0.5).floor()`)
  or replace `cd.round()` with `(cd + 0.5).floor()`, so the metric mask, the
  interpolator's nearest tap, and the core primitive share one half-up rule.

### Pyramid / shrink used inside registration

The device pyramid is §3; the host shrink is `sitk_filters::shrink` (sister
filters panel). ITK conventions confirmed for cross-reference:
`itkShrinkImageFilter.hxx` — output size `std::floor(inputSize/factor)`
(`:263`), spacing exact multiply (`:259`), start index `std::ceil` (`:273`),
index map pure integer `outputIndex*factor + offset` (`:155`);
`itkMultiResolutionPyramidImageFilter.hxx:342,:349` same floor(size)/ceil(index).

---

## 3. Device path (sitk-cuda) — host-vs-device rounding parity

Every device metric (`ops/mattes.rs`, `ops/mean_squares.rs`,
`ops/correlation.rs`) samples through the **single** `take_sample` kernel
(`ops/resident.rs:153`), so the coordinate→index chain exists in exactly one
place. The point map arrives as **stages** via `point_stages`
(`cuda.rs:248`), which extracts the transform's stored `matrix`/`offset`
pairs and **re-checks the bitwise claim on the host** (`cuda.rs:268-281`,
`to_bits()` equality at two probes) before upload. Transforms with no stage
list are refused **by name** with a typed error
(`DeviceRegistrationError::UnsupportedDeviceTransform`, `device.rs:78-95`;
`UnsupportedFixedInitialTransform`, `device.rs:168-173`;
`method.rs:2991-2996`).

| device site | port | mirrors host | rounding | parity |
|---|---|---|---|---|
| fixed virtual point | `resident.rs:209-214` `fmadd_rn` accumulate | `VirtualGrid::point` | none (continuous) | **bit-identical** — `--fmad=false` / `__dmul_rn`+`__dadd_rn` so no fused MAC |
| stage replay | `resident.rs:238-249` per-stage `mat_vec+offset` | `replay_stages` / `MatrixOffsetTransformBase` | none; **rounds once per stage, not folded** | bit-identical |
| moving continuous index | `resident.rs:254-258` `M·(p−origin)` | `MovingImage::continuous_index` | none | bit-identical |
| **moving mask lookup** | **`resident.rs:264` `round(c[d])`** (CUDA C half-away) | **`metric.rs:819` `cd.round()`** (Rust half-away) | half-away-from-zero | **host↔device bit-identical** — but both inherit the §2 CANDIDATE vs ITK |
| linear base | `resident.rs:279` `floor(c[d])` | `linear_at` `:71` | floor | bit-identical |
| corner clamp | `resident.rs:291-293` clamp `[0,size-1]` | `linear_at` `:83` | — | bit-identical |
| `is_inside` | `resident.rs:90-95` `[-0.5,size-0.5)` | `interpolator::is_inside` | half-open | bit-identical |
| **resample nearest** | **`pyramid.rs:625` `floor(cindex[d]+0.5)`** | `nearest_at` `:49` `(c+0.5).floor()` | **RoundHalfIntegerUp (half-up)** | bit-identical; the kernel comment (`:611-625`) explicitly rejects `rint`/`round` and pins `floor(c+0.5)` |
| resample linear | `pyramid.rs:572` `floor(cindex[d])` | `linear_at` | floor | bit-identical |
| resample cindex chain | `pyramid.rs:495-543` (no `fmadd_rn`, but `--fmad=false`) | `resample.rs:267-271` | none | bit-identical |
| shrink out_size | `pyramid.rs:386` `size/f` (int div = floor) | `sitk_filters::shrink` | floor | mirrors host; matches ITK `floor` |
| shrink spacing | `pyramid.rs:387` `spacing*f` | — | exact multiply | mirrors host; matches ITK |
| shrink sampling offset | `pyramid.rs:389` `(delta+0.5).floor().max(0)` | `sitk_filters::shrink` | **RoundHalfIntegerUp** | mirrors host |

### Note on the deliberate rounding asymmetry (not a bug)

The device uses **two different** nearest-integer rules, each matching its
host counterpart exactly:

- **nearest-neighbour resample** → `floor(c+0.5)` (half-up), because host
  `nearest_at` is half-up (RoundHalfIntegerUp — correct vs ITK).
- **metric mask lookup** → `round()` (half-away), because host `mask_allows`
  is half-away.

The device is a faithful mirror of the host in **both** cases — the
host↔device parity this panel owns is intact everywhere. The only open
divergence is host↔ITK at the mask (§2), which the device correctly
inherits rather than introduces.

---

## Tested / Failed / UNFIXED / Fixed

**Tested** (report-only sweep — sites read and classified against ITK read firsthand):
- Core `transform_physical_point_to_index` half-up — matches ITK: pass
- Matrix-offset family point maps (10 variants) not folded, bitwise stages — matches ITK `itkMatrixOffsetTransformBase.hxx:129`: pass
- Centred scale family refused by name — matches (no bitwise fold): pass
- Composite reverse-order, stages not folded — matches `itkCompositeTransform.hxx:66-68`: pass
- BSpline support-start floor `(idx+0.5−order/2).floor()` — matches `itkBSplineInterpolationWeightFunction.hxx:68`: pass
- Displacement-field linear base floor — matches `itkVectorLinearInterpolateImageFunction.hxx:47`: pass
- Interpolator nearest `(c+0.5).floor()` half-up — matches `itkNearestNeighborInterpolateImageFunction.h:88`: pass
- Interpolator linear/bspline/gaussian/sinc floor — matches: pass
- Metric moving interpolation + inside-buffer on continuous index — matches metricv4 `:364-369`: pass
- Device single-sampler stage replay + floor + is_inside vs host — bit-identical: pass
- Device resample nearest `floor(c+0.5)` vs host `nearest_at` — bit-identical: pass
- Device mask `round()` vs host `mask_allows` `.round()` — bit-identical (host↔device): pass

**Failed:** none (report-only; no assertion run this round).

**UNFIXED** (candidate divergences filed for the verify+fix round):
- `metric.rs:819` `MovingImage::mask_allows` uses `cd.round()` (half-away);
  ITK `itkImageMaskSpatialObject.hxx:40` uses RoundHalfIntegerUp (half-up).
  Diverges at moving continuous index component `= −0.5` with a mask. Device
  `resident.rs:264` faithfully mirrors the host, so the divergence is
  host↔ITK, present on both backends.
- `displacement.rs:325` `sparse_jacobian_wrt_parameters` uses `ci.round()`
  (half-away); ITK addresses the local-support Jacobian by integer virtual
  index (no rounding). Diverges at field continuous index component `= −0.5`;
  low reachability (normal path samples on grid points).
- `bspline.rs:508-513` `inside_valid_region` far-edge snap is an ε-nudge
  approximation of ITK's ULP-exact snap; differs by ≤ a few ulp at the
  per-axis far valid-region boundary (flagged in-code).

**Fixed:** none (report-only round; no code changed).
