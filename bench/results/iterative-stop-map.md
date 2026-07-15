# Iterative stopping-criteria parity map — PDE / level-set / N4 / diffusion / fast-marching

Domain owned by this panel: level sets, N4, anisotropic/curvature diffusion,
fast marching, region growing, morphological reconstruction, reinitialize.
Optimizer/registration (demons, LBFGS, Mattes, image-registration `method`)
belongs to a sister panel and is **not** audited here.

Each row: port `file:line` of the termination test, ITK `.hxx:line` of the same,
the exact predicate both sides, and a divergence input or a cannot-differ proof.
Report only — no code changed, no cargo run this round.

**Verdict legend:** MATCH = bit/iteration-identical stop; NEEDS-DECISION = a
real but deliberate/precision divergence for the merge to rule on; candidate =
enumerated, not fully both-side-verified this round.

---

## Findings at a glance

- **REAL (1):** ConfidenceConnected single-pixel region (`n2 == 1`). Port keeps
  the 1-pixel mask; ITK produces an empty image via a `0/0 = NaN` variance that
  its `AlmostEquals` stop does not catch. Reachable input:
  `initial_neighborhood_radius = 0`, isolated single seed, `iterations >= 1`.
  See §4. — for the merge to rule on (the port's behavior is arguably better).
- **NEEDS-DECISION (1):** N4 convergence value/threshold is `float` in ITK,
  `f64` in the port (deliberate). Within float epsilon of `0.001` the stop can
  move by one iteration. See §1.
- **Sharp lead (resolved MATCH):** Chan-Vese binds to ITK's *multiphase* `Halt`
  (`>=` RMS, no `==0` guard), **not** the shared single-phase `halt` (`>` RMS,
  `==0` guard). Reusing the shared one would diverge at `maximum_rms_error ==
  rms_change`. The port picked correctly. See §5.
- **Intentional divergences (2, documented, fast marching):** §1.24 distinct- vs
  raw-target counting on duplicate targets; heap tie-order under
  `FastMarchingBase` topology modes (upstream order non-portable). See §2.
- **Everything else MATCH:** the shared level-set `Halt` + RMS metric (§0),
  N4's two loops modulo the precision note (§1), all fast-marching default paths
  (§2), the whole anisotropic/curvature-flow family as fixed-count (§3),
  reconstruction/geodesic/reinitialize (§4), anti-alias/colliding-fronts (§5).
- **candidate (1, out of listed domain):** kmeans `0..=MAXIMUM_ITERATION`
  inclusive-range off-by-one smell. Clustering, not PDE — see §6.

---

## 0. The crux primitive — `FiniteDifferenceImageFilter::Halt` (shared by the whole level-set + diffusion family) — **MATCH**

Every SimpleITK segmentation level set (geodesic active contour, shape
detection, threshold segmentation, Laplacian segmentation) and the
anti-alias filter routes its stop through **one** port function,
`SparseFieldSolver::halt` (`level_set/sparse_field.rs:220-227`), which mirrors
ITK's base-class `Halt()`.

**ITK** `itkFiniteDifferenceImageFilter.hxx:208-234`:
```cpp
if (GetElapsedIterations() >= m_NumberOfIterations) return true;   // >= cap
if (GetElapsedIterations() == 0)                    return false;  // never RMS-test pass 0
else if (GetMaximumRMSError() > m_RMSChange)        return true;   // strict >
else                                                return false;
```

**Port** `level_set/sparse_field.rs:221-227`:
```rust
if self.elapsed_iterations >= number_of_iterations { return true; }   // >= cap
if self.elapsed_iterations == 0                     { return false; } // never RMS-test pass 0
maximum_rms_error > self.rms_change                                   // strict >
```

Term-by-term identical: `>=` on the iteration cap, the `elapsed == 0`
short-circuit (RMS never tested on pass 0), and the **strict `>`** on
`MaximumRMSError > RMSChange` (i.e. stop when `rms_change < maximum_rms_error`).
`number_of_iterations == 0` halts before pass 0 on both sides.
**Cannot differ on the stop iteration** — the predicate is the same three tests
in the same order.

### 0a. The RMS metric the predicate compares against — **MATCH**

`rms_change` is computed in `update_active_layer_values`
(`level_set/sparse_field.rs:482-539`) vs ITK `UpdateActiveLayerValues`
(`itkSparseFieldLevelSetImageFilter.hxx:274-445`):

- Accumulate `(new_value − center)²` at all three branches (up-demote,
  down-demote, stay): port `:501,:517,:527` ↔ ITK `:344,:395,:427`.
- **Opposite-moving-neighbor survivor path skips both the accumulate and the
  count**: port `continue`s before `counter += 1` (`:497-500,:513-516,:531`) ↔
  ITK `++layerIt; ++updateIt; continue;` before the common `++counter`
  (`:335-341,:379-393,:433`). Survivors enter neither numerator nor denominator
  on either side.
- Final: `counter == 0 → 0`, else `sqrt(accumulator / counter)`:
  port `:535-539` ↔ ITK `:437-443`.

Denominator accounting is bit-identical. Non-vacuity: the survivor `continue`
is the subtle part — a naive port that counted survivors would inflate the
denominator and shrink `rms_change`, changing exactly when `maximum_rms_error >
rms_change` fires; the port excludes them, matching ITK.

The per-filter update functions (`level_set/function.rs`, `level_set/grid.rs`)
carry **no** hidden temporal convergence loop — only spatial stencil loops
(`for d in 0..dim`). The sole stop is §0's `halt`. (Grep-confirmed: no
`while`/`loop`/`break`/`converge` in `function.rs`/`grid.rs`.)

### Public-API level-set stops (all inherit §0)

| Port entry (`level_set/mod.rs`) | default `maximum_rms_error` | default `number_of_iterations` | stop |
|---|---|---|---|
| `geodesic_active_contour` | 0.01 | 1000 | §0 halt |
| `shape_detection` | 0.02 | 1000 | §0 halt |
| `threshold_segmentation` | 0.02 | 1000 | §0 halt |
| `laplacian_segmentation` | 0.02 | 1000 | §0 halt |
| `anti_alias` (`level_set/anti_alias.rs`) | 0.07 | 1000 | §0 halt |

All five call `SparseFieldSolver::run(maximum_rms_error, number_of_iterations)`
(`mod.rs:544`, `anti_alias.rs:117`) → §0 `halt`. Chan-Vese does **not** (dense,
separate — see §5).

---

## 1. N4 bias field — two nested loops — **MATCH** (one NEEDS-DECISION: float vs f64 at the threshold)

Port `n4_bias_field.rs`, ITK
`itkN4BiasFieldCorrectionImageFilter.hxx` (BiasCorrection module).

**Outer loop** `n4_bias_field.rs:286-291` ↔ ITK `.hxx:184-187`:
```rust
while elapsed_iterations < maximum_number_of_iterations[current_level]
   && convergence > convergence_threshold        // strict > ; stop when <= threshold
```
```cpp
while (m_ElapsedIterations++ < m_MaximumNumberOfIterations[m_CurrentLevel] &&
       m_CurrentConvergenceMeasurement > m_ConvergenceThreshold)
```
- Iteration cap: strict `<` both (ITK's post-increment bump is never observed →
  at most `cap` bodies both sides).
- Convergence early-stop: `convergence > threshold` (continue-while) both →
  stop when `convergence <= threshold`, same inequality sense.
- Initial convergence = `f64::MAX` ↔ `NumericTraits<RealType>::max()`
  (`:287` ↔ `.hxx:185`): first iteration always runs.
- Convergence tested **after** the field update, consulted next loop guard, both.

**Convergence measurement** (coefficient of variation) `n4_bias_field.rs:569-587`
↔ ITK `.hxx:647-677`: Welford one-pass, `pixel = exp(previous − current)`,
`sigma += (pixel − mu)²·(N−1)/N` guarded by `N > 1`, then
`mu = mu·(1 − 1/N) + pixel/N`, final `sqrt(sigma/(N−1)) / mu` (= σ/μ, **not**
μ/σ). Byte-for-byte the same recurrence; same included-voxel predicate
(mask ∧ `confidence > 0`). Caller order `(previous, current)` matches
(`:305-306` ↔ `.hxx:205`).

- Mesh doubling per level present and matched (`:325-336` ↔ `.hxx:228-237`).
- Defaults threaded to **SimpleITK** (`[50,50,50,50]`, threshold `0.001`),
  not ITK's bare single-level `[50]`; level count = array length by construction
  (`:259`), so ITK's `Size()==levels` guard is unrepresentable.

**NEEDS-DECISION (precision, not logic):** ITK's convergence value and threshold
are `float` (`RealType`); the port uses `f64` (deliberate, module doc `:24-26`).
Within float epsilon of `0.001` the two can land on opposite sides of the
`convergence <= threshold` test and shift the stop by **one iteration**. The
predicate structure is identical; only operand precision differs. Flagged for
the merge — this is the one place N4's iteration count can diverge.

---

## 2. Fast marching — trial-heap termination — **MATCH** (+ 2 documented intentional divergences)

Port `fast_marching.rs`, `fast_marching_base.rs`,
`fast_marching_upwind_gradient.rs`.

| # | Criterion | Port | ITK | Verdict |
|---|---|---|---|---|
| 2a | Old-filter stop | `node.value > stopping_value` break (`fast_marching.rs:467`) | `currentValue > m_StoppingValue` break (`itkFastMarchingImageFilter.hxx:261`); default `max()/2` (`.hxx:36,48`) | MATCH (strict `>`, truncate-not-clear, pinned `:796-818`) |
| 2b | New-framework stop | `current_value >= threshold` break, pre-accept (`fast_marching_base.rs:630`) | `m_CurrentValue >= m_Threshold` before CheckTopology (`itkFastMarchingThresholdStoppingCriterion.h:63`, `itkFastMarchingBase.hxx:150`) | MATCH (`>=` inclusive — genuinely different from 2a; each side faithful) |
| 2c | Upwind targets | `==` count, `+offset`, lower-only strict `<` (`fast_marching_upwind_gradient` path, `fast_marching.rs:585-596`) | `==` count, `+m_TargetOffset`, `<` (`itkFastMarchingUpwindGradientImageFilter.hxx:184,204,213-214`) | MATCH without duplicate targets |

**Intentional divergence 1 (upstream-findings §1.24):** port counts *distinct
in-bounds* targets (`fast_marching.rs:334-345`), ITK counts raw container size
including duplicates/OOB. Divergence input: targets `[x=3, x=3]`,
`number_of_targets=2` — ITK never reaches `==2` and marches the whole image; port
stops at the one distinct target. Deliberate bug-fix, pinned
(`fast_marching_upwind_gradient.rs:539-560`, `fast_marching.rs:947-977`).

**Intentional divergence 2 (heap tie order):** port imposes a deterministic
total order — ascending flat index on equal arrival time
(`fast_marching.rs:120-122`, `fast_marching_base.rs:355`); ITK's
`std::priority_queue<…, std::greater>` compares value alone, tie order
unspecified by C++ and differing libstdc++/libc++. **Cannot-differ** for the old
filter and upwind-gradient (the arrival field is tie-order-independent —
admitting an equal-valued alive neighbor leaves the solution unchanged). **Can
differ** only under `FastMarchingImageFilterBase` topology modes (`Strict` /
`NoHandles`), where which equal-valued node goes alive first decides a junction.
No bit-target exists upstream (non-portable); flagged, not fixable.

---

## 3. Anisotropic / curvature-flow diffusion — fixed count — **MATCH**

All fixed/deterministic iteration count; no early stop added or dropped.

| Filter | Port loop | ITK stop | Verdict |
|---|---|---|---|
| Gradient / Curvature Anisotropic Diffusion | `for elapsed in 0..number_of_iterations` (`anisotropic_diffusion.rs:377`) | inherits base FDIF `Halt()`; **default `MaximumRMSError == 0`** (`itkFiniteDifferenceImageFilter.h:362`) → RMS branch dead | MATCH |
| Plain CurvatureFlow | `for _ in 0..number_of_iterations` (`denoise.rs:1330`) | `Halt()` overridden to pure fixed-count (`itkCurvatureFlowImageFilter.h:156-164`) | MATCH |
| Min/Max + Binary Min/Max Curvature Flow | `for _ in 0..number_of_iterations` (`min_max_curvature_flow.rs:409`) | inherits CurvatureFlow fixed-count `Halt()` | MATCH |
| Coherence-Enhancing Diffusion (LBR) | `while remaining > 0.0` (`coherence_enhancing_diffusion.rs:808`), inner `n = ceil(time/delta)` capped (`:729-741`) | `while (remainingTime > 0)` (`itkLinearAnisotropicDiffusionLBRImageFilter.hxx:67`), inner `n = ceil(...)` (`.hxx:412-445`) | MATCH (deterministic time loop, no tolerance) |

**Load-bearing proof (anisotropic diffusion):** base FDIF `Halt()` *is* reachable
but `MaximumRMSError == 0` and `RMSChange` is an RMS ≥ 0, so `0 > RMSChange` is
never true → loop runs exactly `NumberOfIterations`. A workspace grep of the
AnisotropicSmoothing / CurvatureFlow / AnisotropicDiffusionLBR modules found no
`SetMaximumRMSError` override. No input exists where the count differs.

**Non-convergence loops correctly not miscounted:** the min/max ball threshold
search (`min_max_curvature_flow.rs:146,208,264,291`) is per-pixel *spatial*, not
temporal; the LBR Selling basis reduction `for _ in 0..SELLING_MAX_ITER` (200,
`:318,361`) matches ITK's `constexpr int maxIter = 200`
(`itkLinearAnisotropicDiffusionLBRImageFilter.hxx:186,241`) — a bounded lattice
reduction over stencil geometry, not a stop condition.

---

## 4. Region growing / morphological reconstruction / reinitialize — **MATCH except one REAL**

Port `region_growing.rs`, `reconstruction.rs`, `geodesic_morphology.rs`,
`reinitialize_level_set.rs`.

| Filter | Port stop | ITK stop | Verdict |
|---|---|---|---|
| ConfidenceConnected loop bound / counting | `for _ in 0..number_of_iterations` (`region_growing.rs:578`), initial flood outside loop (`:573`) | `loop < m_NumberOfIterations` from 0 (`itkConfidenceConnectedImageFilter.hxx:277`), initial flood outside loop (`:267-273`) | MATCH — initial seg is uncounted iteration 0; `N=0` → 1 flood only |
| ConfidenceConnected `n2 == 1` degenerate | `variance = if n2 > 1.0 {…} else {0.0}` → break (`region_growing.rs:595-602`) | `numberOfSamples==1` → `0/0 = NaN`; `AlmostEquals(NaN,0)` false → no break → NaN bounds → empty re-flood (`.hxx:303-322`) | **REAL** (see below) |
| Reconstruction (dilate/erode) | FIFO drain `while let Some(f) = fifo.pop_front()` (`reconstruction.rs:297`); single raster/anti-raster | FIFO drain `while (!IndexFifo.empty())` (`itkReconstructionImageFilter.hxx:286`); single raster/anti-raster | MATCH — same Vincent raster/anti-raster/FIFO hybrid, not raster-iterate-to-stability |
| GrayscaleGeodesicDilate/Erode | slice-equality `next == current` → return (`geodesic_morphology.rs:212,233`) | `done` flag, "no pixel changed in a full pass" (`itkGrayscaleGeodesicDilateImageFilter.hxx:205-216`) | MATCH — port may run 1 extra value-preserving pass; output identical |
| ReinitializeLevelSet | FM-based reinit, no RMS loop; FM stop `node.value > stopping_value` (`fast_marching.rs:467`); narrow-band `stopping_value = bandwidth/2 + 2` (`reinitialize_level_set.rs:142-143`) | FM-based reinit; `currentValue > m_StoppingValue` (`itkFastMarchingImageFilter.hxx:261`); `(m_OutputNarrowBandwidth/2)+2` | MATCH |

### REAL — ConfidenceConnected single-pixel region (`n2 == 1`)

The port treats `n2 <= 1` variance as `0.0` and **breaks cleanly, keeping the
one-pixel mask** (`region_growing.rs:595-602`). ITK passes the `numberOfSamples
== 0` guard (`.hxx:303`), then computes `(sumOfSquares − sum²/1)/(1 − 1) = 0/0 =
NaN` (`.hxx:309`); `Math::AlmostEquals(NaN, 0.0)` is `false` (`.hxx:312`) so ITK
does **not** break — it sets `lower = upper = NaN` (`.hxx:321-322`), the clamps
leave NaN (`NaN > x`/`NaN < x` both false), re-floods on `[NaN, NaN]` (empty),
and only breaks on the *next* pass via `numberOfSamples == 0`.

**Divergence input:** `initial_neighborhood_radius = 0`, a single in-bounds seed
whose face-neighbors all differ from the seed value, `number_of_iterations >= 1`.
Initial variance is `0.0` (radius-0 single seed) → initial flood = the 1 seed
pixel → first re-estimation has `n2 == 1`.
- **Port output:** the single seed pixel set to `replace_value`.
- **ITK output:** empty (all-zero) image.

The port's result is arguably the more sensible one, but it **is** a parity
divergence in the stopping path. The module doc's "num ≥ 1, so guarded" note
(`region_growing.rs:207-213`) covers `n2 == 0` but not this `n2 == 1` case. For
the merge to rule on. (Separately, the variance tolerance is a fixed `1e-12`
(`:484`) vs ITK's ULP `Math::AlmostEquals` — a noise-absorption nuance, not a
stopping divergence except at pathological near-zero variance.)

---

## 5. Chan-Vese / colliding fronts / anti-alias plug-in — **MATCH** (sharp lead: Chan-Vese picks the *distinct* multiphase halt)

| Filter | Port stop | ITK stop | Verdict |
|---|---|---|---|
| Chan-Vese (dense) | its OWN inline halt `while elapsed < N && maximum_rms_error < rms_change` (`chan_vese.rs:742`), `rms_change` seeded `f64::MAX` (`:735`) | `MultiphaseFiniteDifferenceImageFilter::Halt`: `elapsed >= N \|\| MaximumRMSError >= RMSChange` (`itkMultiphaseFiniteDifferenceImageFilter.hxx:256-257`), `m_RMSChange = max()` (`:87`) | MATCH |
| AntiAlias | shared §0 halt via `run()` (`anti_alias.rs:117` → `sparse_field.rs:220-227`) | single-phase `FiniteDifferenceImageFilter::Halt` (`itkFiniteDifferenceImageFilter.hxx:217-228`) | MATCH |
| CollidingFronts | no own convergence; two FM passes, FM stop `value > stopping_value` (`fast_marching.rs:467`) | `currentValue > m_StoppingValue` (`itkFastMarchingImageFilter.hxx:261`) | MATCH |

**Sharp lead — Chan-Vese uses a DIFFERENT halt from §0, and matches ITK's
multiphase form, which is subtly distinct from the single-phase one:**
- RMS comparator: multiphase `MaximumRMSError >= RMSChange` (**`>=`**) vs
  single-phase strict `>`.
- First-pass guard: single-phase has `if elapsed == 0 return false`; multiphase
  has **none**, relying on `m_RMSChange = max()` to keep the RMS clause false on
  pass 0.

Negating ITK multiphase `Halt` gives `(elapsed < N) && (maxRMS < rms)` —
exactly `chan_vese.rs:742`, with `<` (the negation of `>=`), no `==0` branch,
`rms` seeded to `f64::MAX`. **Non-vacuity / distinguishing input:** at
`maximum_rms_error == rms_change` with `elapsed > 0`, the multiphase halt fires
(`>=` true) and the port stops (`maxRMS < rms` false) — **agree**; had the port
wrongly reused the shared single-phase halt (`maximum_rms_error > rms_change`,
strict `>`), it would **continue** — an observable divergence at the `==`
boundary. The port avoided it. This is this round's analogue of the §7
naming-collision flag: the same `Halt` name, two different predicates, and the
port bound each filter to the correct one.

**RMS denominators differ by design (both correct):**
- Chan-Vese: `sqrt(Σ_wholeDomain (φ−d)² / phi.len())` (`chan_vese.rs:828-834`) ↔
  ITK `den` = Σ level-set pixel counts (`.hxx:184-185,233,240,248`). Whole
  domain.
- AntiAlias (shared sparse solver): `sqrt(Σ_activeLayer / counter)` — narrow
  band, survivors excluded (§0a).

---

## 6. Bounding — where a stop could still hide

**Population enumerated** (grep `number_of_iterations|maximum.*iterations` +
`while|break|converge` across `sitk-filters/src`): the five listed families
above are covered. Adjacent iterative filters that carry a stop but sit
**outside** the PDE/level-set/N4/diffusion domain — enumerated, each already
citing its ITK stop in the port module doc, **not fully both-side-verified this
round** (candidate for a later merge pass or the sister panel):

- `deconvolution.rs` — Landweber / Richardson-Lucy: fixed
  `for _ in 0..number_of_iterations` (`:383`); `== 0` returns input. Fixed count.
- `label_fusion.rs` — STAPLE EM: breaks on first iteration with change below a
  "7-digit" tol; `maximum_iterations.max(1)` per ITK `itkSetClampMacro`
  (`:262,301`); unbounded when `maximum_number_of_iterations = None`
  (`!m_HasMaximumNumberOfIterations`). Label domain.
- `displacement_field/iterative_inverse.rs` — `for _ in 0..number_of_iterations`
  (`:180`) with `if smallest_error < stop_value break` (`:209`), strict `<`,
  default `StopValue = 0.0` never fires (error is a norm ≥ 0) — same dead-early-
  stop shape as anisotropic diffusion. Transform domain.
- `patch_based_denoising.rs` — iterative; not read this round. Diffusion-adjacent.
- `slic.rs` — fixed `for _ in 0..maximum_number_of_iterations` (5 default,
  `:325`), no early stop. Superpixel/clustering domain.
- `kmeans.rs` — `for _pass in 0..=MAXIMUM_ITERATION` **inclusive** (`:198`) with
  `if change <= 0.0 break` (`:220`). The inclusive `0..=` is an off-by-one smell
  vs ITK's `while(true) { if CurrentIteration >= Max break; if converged break }`
  (module doc `:29-30` claims parity, centroid-change threshold 0.0). **candidate,
  needs both-side predicate.** Clustering domain, not PDE.
- `demons/*` — deformable registration: **sister panel**, not audited here.

**Where a stop hides, checked:** (a) level-set update functions — no hidden
temporal loop (§0). (b) min/max curvature ball search — spatial, not temporal
(§3). (c) LBR Selling reduction — bounded lattice reduction, not convergence
(§3). Remaining unread inner loops in `patch_based_denoising` are the one place
in the listed-adjacent set not yet opened.

---

## §4 verification addendum — ConfidenceConnected `n2 == 1` is **DOWNSTREAM-OF-§8.3**, not a new REAL

**Verdict (one line): DOWNSTREAM-OF-§8.3.** The single-pixel-region divergence is
the observable manifestation of the already-decided §8.3 variance substitution
(commit `2aad445`, ledger §1.77 upstream + §8.3 divergence). `n2 == 1` is merely
the mechanism by which `variance = 0` arises; there is **no separate predicate**.
Not a new finding — the merge records it as §8.3, and the §4 "REAL —
ConfidenceConnected single-pixel" row above is superseded by this addendum.

**Both-side evidence.** The 1-pixel path flows through the same
`n − 1 == 0 ⇒ substitute 0` branch §8.3 covers:

- Port `region_growing.rs:595-599`: `variance = if n2 > 1.0 { (sq2 − s2²/n2)/(n2 −
  1.0) } else { 0.0 }`. At `n2 == 1` → `variance = 0.0` (the §8.3 site — its
  in-code comment at ~`:600` reads *"ITK gets NaN, this port gets 0 — which then
  trips the almost-zero break below, so a single-voxel region terminates the
  iteration instead of casting NaN to the pixel type. Ledger §8"*).
- Port `region_growing.rs:600-602`: `if variance.abs() < VARIANCE_ALMOST_ZERO {
  break; }` → `0.0 < 1e-12` → break, keeping the 1-pixel mask. This break is a
  faithful port of ITK's own `if (Math::AlmostEquals(m_Variance, 0.0)) break`
  (`itkConfidenceConnectedImageFilter.hxx:312`) — **the break predicate itself
  matches**; only its *operand* (`variance`) differs, and that operand is
  §8.3's substituted `0` vs ITK's `0/0 = NaN`.
- ITK `itkConfidenceConnectedImageFilter.hxx:209,312,321-322`: `numberOfSamples
  == 1` → `(sumOfSquares − sum²/1)/(1 − 1) = 0/0 = NaN`; `AlmostEquals(NaN, 0)`
  false → no break → `lower = upper = NaN` → empty re-flood → empties output.

**Separating input** (identical to §8.3's pin): `initial_neighborhood_radius = 0`,
one in-bounds seed whose face-neighbors all differ, `number_of_iterations >= 1`.
Port keeps the 1 seed pixel; ITK empties. This is exactly what §8.3's pinned test
`confidence_connected_single_seed_at_radius_zero_substitutes_zero_for_itks_nan_variance`
asserts. **No predicate independent of the variance path produces this outcome** —
searched the loop body (`region_growing.rs:578-613`): the only stop is the
`variance.abs() < VARIANCE_ALMOST_ZERO` break, whose divergence is entirely
determined by §8.3's substituted operand. Therefore DOWNSTREAM-OF-§8.3.

**Correction to this map:** §4's REAL row for the single-pixel case is withdrawn;
it is the same divergence as §8.3, already committed and pinned. The
iterative-stop map contributes a cross-reference to §8.3, not a new REAL. (Net:
this round found **zero** new REAL divergences — see `iterative-stop-r3.md`.)
