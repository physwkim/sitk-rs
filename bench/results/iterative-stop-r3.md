# Iterative stopping-criteria — Round 3 loose-end verification

Report only, no code changed, no gates owed. Four items (V1–V4), each with
both-side predicate and a REAL / NOT-REAL / NEEDS-DECISION / disambiguation
verdict.

---

## V1 — ConfidenceConnected `n2 == 1`: **NOT a new REAL — it is the downstream of §8.3** (already decided, commit `2aad445`)

**Disambiguation verdict: the single-pixel divergence flagged in Round 2 flows
through the exact variance-substitution site the session already fixed in
commit `2aad445` (ledger §1.77 upstream, §8.3 divergence). It is not a separate
code path and needs no new code — only a ledger cross-reference.**

Trace (`region_growing.rs`):
- Initial variance for a radius-0 single seed: `total_num = 1`, so
  `if total_num > 1.0 { … } else { 0.0 }` (`:535-539`) → `variance = 0.0`. Bounds
  collapse to `[seedval, seedval]`; initial flood = the 1 seed pixel.
- First loop pass (`for _ in 0..number_of_iterations`, `:578`): `n2 == 1`, so
  `variance = if n2 > 1.0 { … } else { 0.0 }` (`:595-599`) → **`variance = 0.0`**
  — the same `n − 1 == 0 ⇒ substitute 0` site.
- `if variance.abs() < VARIANCE_ALMOST_ZERO { break; }` (`:600-602`) → `0.0 < 1e-12`
  → break, keeping the 1-pixel mask.

ITK (`itkConfidenceConnectedImageFilter.hxx`): `numberOfSamples == 1` divides by
`n − 1 == 0` → `0/0 = NaN` (`:209`); `AlmostEquals(NaN, 0)` is false (`:312`) so no
break; NaN bounds → empty re-flood → next pass breaks on `numberOfSamples == 0`
→ empty output.

The port keeps the pixel **because** it substituted `variance = 0` (§8.3), which
then trips the almost-zero break. ITK's NaN path is exactly what §8.3 documents
ITK doing. This is confirmed verbatim by the commit that introduced §8.3 — the
loop-site comment it added (`region_growing.rs` ~`:600`) reads:

> ITK gets NaN, this port gets 0 — which then trips the almost-zero break below,
> so a single-voxel region terminates the iteration instead of casting NaN to
> the pixel type. Ledger §8.

and it is pinned by
`confidence_connected_single_seed_at_radius_zero_substitutes_zero_for_itks_nan_variance`.

**Correction to my Round-2 map:** §4's "REAL — ConfidenceConnected single-pixel"
row double-counted this decided divergence as new. It should be reclassified as
"downstream of §8.3, already pinned" — a ledger cross-reference from the
iterative-stop map to §8.3, not a new finding.

---

## V2 — kmeans `0..=MAXIMUM_ITERATION`: **NOT-REAL (MATCH)** — the inclusive range is a deliberate match, not an off-by-one

**Port** `kmeans.rs`: `const MAXIMUM_ITERATION: u32 = 200` (`:164`);
`for _pass in 0..=MAXIMUM_ITERATION` (`:198`) = 201 passes; each pass does
assignment + centroid update, then `let change = Σ|means − new_means|;
means = new_means; if change <= 0.0 { break; }` (`:214-222`).

**ITK** `itkKdTreeBasedKmeansEstimator.hxx:317-347`:
```cpp
m_CurrentIteration = 0;
while (true) {
  // Lloyd pass: Filter(...) + UpdateCentroids
  if (m_CurrentIteration >= m_MaximumIteration) break;   // POST-pass cap check
  m_CentroidPositionChanges = GetSumOfSquaredPositionChanges(prev, cur);
  if (m_CentroidPositionChanges <= m_CentroidPositionChangesThreshold) break;
  ++m_CurrentIteration;
}
```
The cap check is **after** the Lloyd pass and `m_CurrentIteration` increments at
the bottom, so convergence-never yields **`m_MaximumIteration + 1` = 201** Lloyd
passes. Driver `itkScalarImageKmeansImageFilter.hxx:91-92`:
`SetMaximumIteration(200)`, `SetCentroidPositionChangesThreshold(0.0)`.

- Pass count: port 201 (`0..=200`) = ITK 201 (`M + 1`). MATCH — the port's
  inclusive range deliberately reproduces ITK's post-pass cap.
- Convergence: both `change <= 0.0` against an exact-0.0 threshold. ITK's metric
  is Σ squared position changes, the port's is Σ absolute changes; both are
  non-negative and are zero iff every centroid is unchanged, so they trip on the
  same pass. MATCH.
- Output: identical centroid sequence → identical final means → identical
  labeling. Even a hypothetical extra pass cannot move a fixed point.

**Cannot change output** — no input exists where the labeling differs. NOT-REAL.

---

## V3 — `patch_based_denoising` three loops: **all MATCH** (fully ported, scalar path, no stub)

| Loop | Port | ITK | Verdict |
|---|---|---|---|
| Outer denoising iteration | `for elapsed in 0..number_of_iterations.max(1)` (`patch_based_denoising.rs:1214,1101`); kernel gate `elapsed % update_frequency == 0` (`:1219`) | `while (!Halt())`, `Halt`: `elapsed >= m_NumberOfIterations` (`itkPatchBasedDenoisingBaseImageFilter.hxx:91,232`); clamp min 1 (`.h:271`); gate `elapsed % m_KernelBandwidthUpdateFrequency == 0` (`:97`) | MATCH (fixed count, same `>=`, same clamp, same gate/order) |
| σ Newton-Raphson | `for _ in 0..20`, `if update.abs() < *kernel_sigma * 0.01 { break; }` (`:984,987`); consts `MAX_SIGMA_UPDATE_ITERATIONS=20` (`:249`), `SIGMA_UPDATE_CONVERGENCE_TOLERANCE=0.01` (`:247`) | `for i<MaxSigmaUpdateIterations(20)`, `if |sigmaUpdate[ic]| < m_KernelBandwidthSigma[ic]*m_SigmaUpdateConvergenceTolerance(0.01)` (`itkPatchBasedDenoisingImageFilter.hxx:1406,1428`; consts `.h:190,560`) | MATCH (cap 20, tol 0.01, strict `<`, post-update σ operand, same rescale in/out) |
| Patch sampling | `while out.len() < box_points.min(number_of_results_requested)` (`:705,707`), with replacement | `while (pointsFound < min(m_NumberOfResultsRequested, box))` (`itkUniformRandomSpatialNeighborSubsampler.hxx:123,141`), with replacement | MATCH (fixed draw count, not a stop) |

Fully implemented for scalar pixels (no `todo!`/`unimplemented!`); the
RGB/Vector/tensor Riemannian paths are absent by design and touch none of the
three loops. Only non-code note: a doc-wording inaccuracy at
`patch_based_denoising.rs:118-123` ("upstream never terminates" — ITK's `:116`
guard returns empty, so runtime behavior still agrees). **No divergence
constructible for scalar images.**

---

## V4 — four adjacent iterative stops

### A. `deconvolution` (Landweber + RichardsonLucy) — **MATCH**
Port fixed `for _ in 0..number_of_iterations` (`deconvolution.rs:383`), no
early-stop ↔ ITK base `for (m_Iteration = 0; m_Iteration <
m_NumberOfIterations; ++m_Iteration)` with the only break on the external
`m_StopIteration` abort flag (`itkIterativeDeconvolutionImageFilter.hxx:117-126`),
default `false` (`.h:155`) and **not exposed by SimpleITK**. No
`m_ConvergenceTolerance`/residual test exists in this ITK version; neither
Landweber nor RichardsonLucy overrides `GenerateData`. Numerically identical for
every SimpleITK-reachable config. (Unmodeled-but-harmless: the port omits the
unreachable `m_StopIteration` break.)

### B. `label_fusion` STAPLE + MultiLabelSTAPLE — **MATCH** (independently re-verified by me)
STAPLE convergence port `label_fusion.rs:347-360` ↔ ITK
`itkSTAPLEImageFilter.hxx:216-250`: converged iff `iter != 0` and **all** raters'
`(pᵢ − last_pᵢ)² <= 1e-14` **and** `(qᵢ − last_qᵢ)² <= 1e-14`. ITK writes the
strict-`>` complement (`flag=false` on first `² > 1e-14`); the port writes the
`<=` complement directly — same boundary, `1e-14` counts as converged both
sides. Same `min_rms_error = 1.0e-14` (ITK `:51` ↔ port `:155`), same `iter != 0`
guard, same unconditional `last_p/q` copy, same `for(; iter <
m_MaximumIterations; ++iter)` strict-`<` loop (`:128` ↔ port `while iteration <
maximum_iterations` `:302`), same E→M→check order. Clamp: `max(1)` ↔
`itkSetClampMacro(MaximumIterations, 1, max)` (`.h:205`), default `u32::MAX`.
MultiLabelSTAPLE: unbounded when `None`, else `maximum_update <
termination_update_threshold` strict `<` over max matrix-entry update
(`label_fusion.rs:555,618` ↔ `itkMultiLabelSTAPLEImageFilter.hxx:266,378-380`).
**NOT-REAL** — this was the flagged highest-risk item; it is a faithful port, not
a re-derivation.

### C. `slic` — **MATCH** (NEEDS-DECISION resolved via SimpleITK yaml)
No early stop either side: port `for _ in 0..maximum_number_of_iterations`
(`slic.rs:325`), residual computed but unused; ITK `for (loopCnt = 0; loopCnt <
m_MaximumNumberOfIterations; ++loopCnt)` (`itkSLICImageFilter.hxx:599`),
`m_AverageResidual` computed (`:659`) but **no** early termination (a dead `//
while error <= threshold` comment at `:662`). MATCH on the loop.

Default count: port hardcodes `5` for all dimensions (`slic.rs:135`); ITK-**native**
default is `(ImageDimension > 2) ? 5 : 10` (`itkSLICImageFilter.hxx:45`) — 10 for
2D. **Resolved:** SimpleITK's `SLICImageFilter.yaml:29` sets
`MaximumNumberOfIterations` default `5u` **unconditionally**, overriding
ITK-native's 2D=10. Since sitk-rs targets SimpleITK parity, the port's
unconditional 5 is **faithful — MATCH-to-SimpleITK**. (A caller who set 10 for a
2D image to match ITK-native would be diverging from SimpleITK, not the port.)

### D. `displacement_field/iterative_inverse` — **MATCH**
Port `for _ in 0..number_of_iterations` (`iterative_inverse.rs:180`), `if
smallest_error < stop_value { break; }` (`:209`), defaults
`number_of_iterations = 5`, `stop_value = 0.0` (`:99-100`) ↔ ITK `for (i = 0; i <
m_NumberOfIterations; ++i)` (`itkIterativeInverseDisplacementFieldImageFilter.hxx:137`),
`if (smallestError < m_StopValue) break;` (`:201`), defaults `5` and `0.0`
(`.h:123,125`). Strict `<` on the residual norm; a norm is never `< 0.0` so the
default never fires — same dead-early-stop shape as anisotropic diffusion.
(Port's per-axis step §2.34 and per-pixel `smallest_error` reset §1.32 concern
the search, not the stop.)

---

## Round-3 net result

- **V1:** DOWNSTREAM-OF-§8.3 — the Round-2 "REAL" is withdrawn (already committed
  `2aad445` / pinned). **Zero new REAL this round.**
- **V2:** NOT-REAL — kmeans `0..=` deliberately matches ITK's `M+1` passes.
- **V3:** MATCH — all three patch_based_denoising loops, scalar path.
- **V4:** A/B/D MATCH; C MATCH-to-SimpleITK (SLIC default 5 confirmed via
  `SLICImageFilter.yaml`).
- **Open NEEDS-DECISION carried from the map (unchanged):** N4's `float`-vs-`f64`
  convergence operand near the `0.001` threshold (§1) — deliberate, for the merge.
