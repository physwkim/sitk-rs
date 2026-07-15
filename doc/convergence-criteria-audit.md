# Iterative stopping-criteria parity audit — conclusion

**Result: zero new defects.** Every iterative stop condition in this port — the
predicate that decides *when a loop terminates* — was mapped against ITK from
both sides (port `file:line` + ITK `.hxx:line`, exact operator, tolerance sense,
and cap), across the whole iterative surface. One deliberate divergence was
LIVE (changed real output) and has been **fixed to match ITK**; one is a
documented precision choice this port keeps; everything else is a faithful
match or resolves to an already-decided ledger row. This document is the
conclusion; the evidence is three maps in `bench/results/`, cited below.

The audit was run because a wrong stop condition is the same *shape* of defect
the physical-space and boundary-condition sweeps mined — an off-by-one in a
comparison operator (`<` vs `<=`), a relative-vs-absolute tolerance, or a
test-before-vs-after-update — invisible to a test that only checks a converged
result, but wrong on the iteration count and the final value. It was built from
both ends independently (ITK source vs port code) and merged.

## Domains covered

Two panels split the iterative surface with no overlap and no gap:

1. **PDE / segmentation / region-growing** (`iterative-stop-map.md`,
   `iterative-stop-r3.md`) — the shared FiniteDifference `Halt` and sparse-field
   RMS metric, N4, fast marching (old + new frameworks), anisotropic / curvature
   / CED diffusion, ConfidenceConnected / reconstruction / geodesic morphology /
   reinitialize, Chan-Vese multiphase halt and colliding fronts; plus the
   round-3 loose ends: kmeans, patch_based_denoising, STAPLE / MultiLabelSTAPLE,
   deconvolution, iterative_inverse, SLIC.
2. **Optimizer / registration / demons** (`stopping-criteria-optimizer-map.md`)
   — RegularStep / GradientDescent / line-search / conjugate-gradient / LBFGS2 /
   LBFGS-B / Amoeba / Powell / 1+1 evolutionary optimizers, the golden-section
   line search, the WindowConvergenceMonitor, the ImageRegistrationMethodv4 level
   advance, and the demons FiniteDifference halt shared by all five flavors.

## The two divergences that were real decisions

Everything else matched. Two rows required a call, both surfaced to the user:

1. **Finding A — the gradient-descent family's extra min-step stop (LIVE) —
   FIXED to match ITK.** The port's `GradientDescentOptimizer`, line-search, and
   conjugate-gradient optimizers each carried a `min_step_tolerance = 1e-8` stop
   that `itk::GradientDescentOptimizerv4` does not have; the fixed-rate and
   estimate-once paths also omitted ITK's value-plateau monitor and leaned on
   min-step entirely. Measured LIVE on the bowl `(x−3)²+(y+2)²` at lr 0.1: the
   port stopped `StepTooSmall @ 83` where ITK's monitor stops `Converged @ 37` —
   a different iteration *and* a different final transform, from the identical
   `SetOptimizerAsGradientDescent` call. The user chose ITK parity. The fix
   (commit on `main`, `fix(registration): match ITK — drop the gradient-descent
   family's min-step stop`) removes min-step from the three optimizers, installs
   the value-plateau monitor on every gradient-descent construction that
   previously leaned on it, and keeps the `total_energy == 0` guard as the now
   load-bearing more-correct behavior (ITK divides by `total_energy` → NaN →
   never stops). `RegularStepGradientDescentOptimizer` keeps its own
   `minimum_step_length` — ITK's RegularStep carries it, so the port matches; it
   was correctly excluded. Verification: `set_optimizer_as_gradient_descent`
   bowl@lr0.1 now pins `Converged @ 37`; workspace gates green (3511 tests).
2. **N4 convergence in `f64` vs ITK's `float` — KEPT, documented §4.122.** N4's
   per-level convergence test is byte-identical to ITK in predicate structure,
   recurrence, and `> threshold` sense; only the operand precision differs (port
   `f64`, ITK `RealType = float`). Near the `0.001` threshold the two can run one
   more or one fewer iteration at a level. Not an ITK defect — `float32`
   convergence is deterministic and legitimate — so this is a §4 deliberate
   divergence (the discrete consequence of §4.1's crate-wide `f64` choice), not
   a §8 upstream-defect row. The user chose to keep `f64`.

## The one candidate that looked REAL and was not

The PDE panel's round-2 map flagged **ConfidenceConnected at `n2 == 1`**
(single-pixel region: the port keeps the 1-pixel mask, ITK empties the image) as
a new REAL. Round-3 verification **withdrew it**: the divergence flows through the
exact `n − 1 == 0 ⇒ variance = 0`-substituted-for-ITK's-`0/0`-NaN path that
commit `2aad445` already fixed and pinned (ledger §1.77 upstream + §8.3). `n2 == 1`
is only the mechanism that produces `variance = 0`; the break predicate itself
matches ITK, and only its operand differs — §8.3's substituted `0` vs ITK's
`NaN`. It is an already-decided cross-reference, not a new finding.

## Everything else, verified MATCH or NOT-REAL

- **kmeans** — the inclusive `0..=MAXIMUM_ITERATION` is a deliberate match to
  ITK's post-pass cap check (both run 201 Lloyd passes); identical centroids.
- **patch_based_denoising** — all three loops (outer fixed-count, σ
  Newton-Raphson, patch sampling) match; fully ported scalar path.
- **STAPLE / MultiLabelSTAPLE** — bit-identical EM convergence (`iter != 0` +
  all `Δ² <= 1e-14`), independently re-verified against `itkSTAPLEImageFilter.hxx`.
- **deconvolution** — fixed-count; ITK's only break is the SimpleITK-unreachable
  `m_StopIteration`.
- **iterative_inverse** — strict `<` residual vs `stop_value = 0.0` default.
- **SLIC** — the unconditional default `5` is faithful to SimpleITK's
  `SLICImageFilter.yaml` (which overrides ITK-native's `(dim>2)?5:10`).
- **fast marching** — old `>` / new-framework `>=` are a genuine ITK old-vs-new
  difference, each side faithful; two documented intentional divergences
  (distinct-target counting §1.24, non-portable heap tie-order).
- **anisotropic / curvature / CED diffusion** — fixed iteration count, proven
  (base FDIF `MaximumRMSError == 0` makes the RMS break unreachable).
- **Chan-Vese** — binds to ITK's *multiphase* halt (`>=` RMS, no `==0` guard),
  not the single-phase halt; the port picked the correct one of the two.
- **LBFGS-B** — the load-bearing case: the port checks the iteration cap before
  pgtol/factr, reproducing ITK's non-obvious "max-iterations wins the tie".
- **RegularStep / LBFGS2 / Amoeba / Powell / 1+1 / golden-section / demons** —
  operators, tolerance senses, caps, test order, and defaults all match.

## Evidence

- `bench/results/iterative-stop-map.md` — PDE/segmentation map, both sides cited
  per family, with the §4 verification addendum withdrawing the ConfidenceConnected row.
- `bench/results/iterative-stop-r3.md` — round-3 verification of kmeans,
  patch_based_denoising, STAPLE, deconvolution, iterative_inverse, SLIC.
- `bench/results/stopping-criteria-optimizer-map.md` — optimizer/registration
  map, with the Finding-A reachability addendum measuring it LIVE.
