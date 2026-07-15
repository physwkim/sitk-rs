# Iterative stopping-criteria map — optimizer / registration / demons domain

Matched port↔ITK map of every iterative stop condition in this panel's domain.
Port cites `file:line` in my worktree; ITK cites `.hxx`/`.cxx`/`.c`:line under
`/home/stevek/work/ITK` (a lightly-modified checkout — line numbers are this
tree's). Each row gives the predicate written out from **both** sides, the
operator/direction compared verbatim, and either a separating input or a proof
none exists. **Report only — no code changed.**

**Bottom line.** Eleven stop conditions are verified byte-for-byte matches,
*including* the two load-bearing orderings (L-BFGS-B's maxiter-wins-ties, and
libLBFGS's converged-before-cap). Three items are not clean matches:

- **A — REAL (deliberate), and now measured LIVE — the gradient-descent family's
  min-step stop.** The port's `GradientDescentOptimizer` /
  `GradientDescentLineSearchOptimizer` / `ConjugateGradientLineSearchOptimizer`
  each carry a `min_step_tolerance` stop that ITK's `GradientDescentOptimizerv4`
  base *does not have*, and the fixed-rate path omits the value-plateau monitor
  ITK/SimpleITK configures. Different criterion → different stopping iteration.
  The **reachability addendum** (end of file) measures it: min-step is the
  *first-triggered* stop on well-conditioned smooth objectives across all three
  optimizers (StepTooSmall @ 83 vs ITK's monitor @ 37 on the fixed-rate bowl), so
  A changes iteration counts and final transforms on real runs — LIVE, not latent.
- **B — NEEDS-DECISION (stop-reason only) — the cap-vs-convergence reorder.**
  Iteration count and final parameters are bit-identical to ITK; only the
  reported `StopReason` enum can differ, and only when a criterion is first
  satisfied at exactly the iteration equal to the cap.
- **C — NEEDS-DECISION (reachability-gated) — the all-zero-energy guard** in the
  convergence monitor. A port-added guard ITK lacks; practically masked by B's
  min-step in the default wiring.

---

## Verified matches

### 1. RegularStepGradientDescentOptimizerv4 — `optimizer.rs` ↔ `itkRegularStepGradientDescentOptimizerv4.hxx`
| Stop | Port | ITK | Verdict |
|---|---|---|---|
| gradient-magnitude stationary | `if gradient_magnitude < self.gradient_magnitude_tolerance` (`optimizer.rs:414`) | `if (gradientMagnitude < m_GradientMagnitudeTolerance)` (`.hxx:115`) | **MATCH** (`<`) |
| direction-reversal relaxation | `if scalar_product < 0.0 { relaxation *= relaxation_factor }` (`:431`) | `if (scalarProduct < 0) … m_CurrentLearningRateRelaxation *= m_RelaxationFactor` (`.hxx:136,138`) | **MATCH** (`< 0`) |
| minimum step length | `if step_length < self.minimum_step_length` (`:436`) | `if (stepLength < m_MinimumStepLength)` (`.hxx:143`) | **MATCH** (`<`) |
| compensated reductions | `compensated_sum(...)` for magnitude (`:411`) and scalar product (`:425`) | `CompensatedSummationType` for both (`.hxx:107-113`, `:126-133`) | **MATCH** |
| defaults | relaxation `0.5`, grad-mag-tol `1e-4` (`:304`) | same | **MATCH** |

Convergence monitoring is **off** for this optimizer on both sides (ITK sets
`m_UseConvergenceMonitoring = false` in `StartOptimization`, `.hxx:41`; the port
never wires `set_convergence` into `RegularStepGradientDescentOptimizer`).

### 2. WindowConvergenceMonitoringFunction — `convergence.rs` ↔ `itkWindowConvergenceMonitoringFunction.hxx`
| Element | Port | ITK | Verdict |
|---|---|---|---|
| window-not-full → "don't stop" | `if len < window_size { return None }` (`convergence.rs:68`) | `if (…< m_WindowSize) return NumericTraits<>::max();` (`.hxx:65,67`) | **MATCH** |
| accumulate total energy | `total_energy += value.abs()` (`:61`) | `m_TotalEnergy += itk::Math::Absolute(value)` (`.hxx:48`) | **MATCH** |
| slide window | `push_back; if len > window_size pop_front` (`:57-60`) | `push_back; if (…> m_WindowSize) pop_front` (`.hxx:43-46`) | **MATCH** (`>`) |
| fit + negated slope | closed-form `c0 − c1` (order-1, 2 ctrl pts) (`:90-93`) | `-gradient[0][0]` of order-1/2-ctrl-pt BSpline at parametric 1.0 (`.hxx:88-91,117-120`) | **MATCH** (negated) |
| default window | SimpleITK `10` (`method.rs:669`) | `m_WindowSize(10)` (`.hxx:32`) | **MATCH** |

The optimizer-side comparison is `if m_ConvergenceValue <= m_MinimumConvergenceValue`
(`itkGradientDescentOptimizerv4.hxx:126`, `<=`), matched by the port's
`if cv <= min_cv` (`optimizer.rs:225`). *Exception: the all-zero-energy guard —
see finding C.*

### 3. GradientDescentLineSearch / ConjugateGradient golden section — `optimizer.rs` ↔ `itkGradientDescentLineSearchOptimizerv4.hxx` / `itkConjugateGradientLineSearchOptimizerv4.hxx`
| Stop | Port | ITK | Verdict |
|---|---|---|---|
| recursion-depth cap | `if *line_search_iterations > max_line_search_iterations` (`optimizer.rs:497`) | `if (m_LineSearchIterations > m_MaximumLineSearchIterations)` (`.hxx:106`) | **MATCH** (`>` strict) |
| bracket collapse | `if (c-a).abs() < epsilon*(b.abs()+x.abs())` (`:507`) | `if (Math::Absolute(c-a) < m_Epsilon*(Absolute(b)+Absolute(x)))` (`.hxx:121`) | **MATCH** (`<`) |
| defaults | ε `0.01`, maxLS `20`, lower `0`, upper `5` (`:624-625`) | `m_Epsilon(0.01)`, `(20)`, `(0)`, `(5.0)` (`.hxx:29-34`) | **MATCH** |
| CG γ denom guard | `if gamma_denom > f64::EPSILON` (`:1022`) | `if (gammaDenom > NumericTraits<>::epsilon())` (`.hxx:65`) | **MATCH** |
| CG γ restart | `if !(0.0..=5.0).contains(&gamma) { gamma = 0.0 }` (`:1028`) | `if (gamma < 0 || gamma > 5) gamma = 0;` (`.hxx:71`) | **MATCH** (keeps `[0,5]`) |

*(Port docstring at `:840` says "(0, 5]"; the **code** keeps `[0,5]` inclusive of
0, matching ITK. Doc nit, not a code divergence.)*

### 4. LBFGS2Optimizerv4 / libLBFGS — `lbfgs2.rs` ↔ `Modules/ThirdParty/libLBFGS/src/itklbfgs/lib/lbfgs.c`
| Stop | Port | ITK/libLBFGS | Verdict |
|---|---|---|---|
| already-minimized (pre-loop) | `if gnorm0/xnorm0 <= solution_accuracy` with `.max(1.0)` (`lbfgs2.rs:836-838`) | `if (xnorm<1.0) xnorm=1.0; if (gnorm/xnorm <= epsilon)` → `LBFGS_ALREADY_MINIMIZED` (`lbfgs.c:451-455`) | **MATCH** (`<=`) |
| gradient convergence | `if gnorm/xnorm <= solution_accuracy` (`:893`) | same (`lbfgs.c:508`) | **MATCH** (`<=`) |
| delta / past | `if iterations>=past { rate=(pf[iterations%past]-fx)/fx; if rate.abs()<delta }` (`:899-902`) | `if (past<=k){ rate=(pf[k%past]-fx)/fx; if (fabs(rate)<delta) }` (`lbfgs.c:521-526`) | **MATCH** (`<`) |
| iteration cap | `if number_of_iterations!=0 && iterations>=number_of_iterations` (`:909`) | `if (max_iterations!=0 && max_iterations < k+1)` (`lbfgs.c:536`) | **MATCH** (equivalent) |
| line-search failure revert | copy `xp→x, gp→g`, `LineSearchFailed` (`:881-886`) | `veccpy(x,xp); veccpy(g,gp); ret=ls` (`lbfgs.c:481-483`) | **MATCH** |
| **order** | gradient → delta → cap (`:893,899,909`) | gradient → delta → cap; converged wins over cap (`lbfgs.c:508,519,536`) | **MATCH** |
| defaults | m=6, ε=1e-5, past=0, δ=1e-5, maxLS=40, min=1e-20, max=1e20, ftol=1e-4, wolfe=0.9, gtol=0.9, xtol=1e-16 (`:601-615`) | `_defparam` (`lbfgs.c:113-118`) | **MATCH** (all) |

### 5. LBFGSBOptimizerv4 / netlib — `lbfgsb.rs` ↔ `.../v3p/netlib/opt/lbfgsb.c` + `itkLBFGSOptimizerBasev4.hxx` + `vnl_lbfgsb.cxx`
The load-bearing item: the **order** of the three optimality tests, because it
decides the exact stop iteration and the tie-break.
| Stop | Port | ITK (netlib + wrapper) | Verdict |
|---|---|---|---|
| max-func-evals | `Book::record` returns `num_evaluations > max` (`lbfgsb.rs:1222`) | `num_evaluations_ > get_max_function_evals()` (`vnl_lbfgsb.cxx:219`) | **MATCH** (`>`) |
| iteration cap **(first, wins ties)** | `if iter >= max_iterations` **before** pgtol/factr (`lbfgsb.rs:1516`) | `num_iterations_ >= m_NumberOfIterations` on `NEW_X`, one `setulb` call **before** L777 (`itkLBFGSOptimizerBasev4.hxx:34`) | **MATCH** (`>=`, maxiter wins) |
| projected-gradient | `if sbgnrm <= pgtol` (`:1520`; init-point `:1308`) | `if (sbgnrm <= *pgtol)` L777 (`lbfgsb.c:1090`; init `:890`) | **MATCH** (`<=`) |
| relative-reduction | `if fold-f <= tol*ddum`, `ddum=max(|fold|,|f|,1)`, `tol=factr*epsmch` (`:1524-1525,1254`) | `if (fold-*f <= tol*ddum)`, `ddum=max(|fold|,|f|,1)`, `tol=factr*epsmch` (`lbfgsb.c:1096-1099,763`) | **MATCH** (`<=`) |
| defaults | factr `1e7`, pgtol `1e-5`, maxFE `2000`, corrections `5` (`:1611-1615`) | `1e7` / `1e-5` / `2000` / `5` (`itkLBFGSOptimizerBasev4.h:161-163`, `itkLBFGSBOptimizerv4.h:183`) | **MATCH** |

The port's placement of the iteration cap *ahead of* pgtol/factr (`:1516` before
`:1520/:1525`) reproduces ITK's non-obvious behavior that when an iterate
simultaneously reaches the cap and satisfies pgtol, the run stops as
max-iterations, not convergence. Separating input to confirm the tie-break holds:
a run whose projected gradient first drops to `pgtol` on exactly the iteration
that equals `max_iterations` — both report max-iterations.

### 6. AmoebaOptimizerv4 / vnl_amoeba — `gradient_free.rs` ↔ `core/vnl/algo/vnl_amoeba.cxx`
| Stop | Port | ITK/vnl | Verdict |
|---|---|---|---|
| convergence | `if simplex_diameter < x_tol && sorted_fdiameter < f_tol` (`gradient_free.rs:132`) | `if (simplex_diameter(...) < X_tolerance && sorted_simplex_fdiameter(...) < F_tolerance)` (`vnl_amoeba.cxx:291`) | **MATCH** (both `<`) |
| evaluation cap | `while cnt < max_evaluations` (`:131`) | `while (cnt < maxiter)` (`vnl_amoeba.cxx:289`) | **MATCH** (`<`, an **eval** count) |
| helpers | consecutive-pair `simplex_diameter`, worst−best `sorted_fdiameter`, per-component `maxabsdiff` (`:79-103`) | same (`vnl_amoeba.cxx:109-139`) | **MATCH** |
| defaults | X-tol `1e-8`, F-tol `1e-4` (`:230-231`); cap = `number_of_iterations` (`:303`) | `1e-8`/`1e-4` (`itkAmoebaOptimizerv4.cxx:26-27`); vnl `maxiter=500` from `m_NumberOfIterations` (`.cxx:32`) | **MATCH** |

### 7. PowellOptimizerv4 — `gradient_free.rs` ↔ `itkPowellOptimizerv4.hxx`
| Stop | Port | ITK | Verdict |
|---|---|---|---|
| outer-loop off-by-one | `for iter in 0..=self.number_of_iterations` (`gradient_free.rs:632`) | `for (…; m_CurrentIteration <= m_MaximumIteration; …)` (`.hxx:414`) | **MATCH** (`<=`, runs N+1) |
| value convergence | `if 2.0*(fp-fx).abs() <= value_tolerance*(fp.abs()+fx.abs())` (`:665`) | `if (2.0*Absolute(fp-fx) <= m_ValueTolerance*(Absolute(fp)+Absolute(fx)))` (`.hxx:447`) | **MATCH** (`<=`) |
| line-iteration cap | `for _ in 0..maximum_line_iteration` (`:437`) | `for (…; m_CurrentLineIteration < m_MaximumLineIteration; …)` (`.hxx:252`) | **MATCH** (`<`) |
| line step-tol | `if (x-mid).abs() <= (tol2-0.5*(b-a)) || 0.5*(b-a) < step_tolerance` (`:442-444`) | same, `<=` then `<` (`.hxx:259`) | **MATCH** |
| defaults | SimpleITK stepLen `1`, stepTol `1e-6`, valueTol `1e-6`, maxLineIter `100`, iters `100` (`:543-548`) | v4 raw defaults are **0**; SimpleITK `SetOptimizerAsPowell` overrides to `1e-6` — port takes SimpleITK's | **MATCH** (see note) |

Note: ITK's `PowellOptimizerv4` constructor leaves `m_StepTolerance`/`m_ValueTolerance`
= 0; SimpleITK's convenience method sets `1e-6`. The port mirrors SimpleITK's
values, which is the correct reference. Not a divergence — flagged so the merge
doesn't misread the raw-ITK-vs-SimpleITK default gap.

### 8. OnePlusOneEvolutionaryOptimizerv4 — `gradient_free.rs` ↔ `itkOnePlusOneEvolutionaryOptimizerv4.hxx`
| Stop | Port | ITK | Verdict |
|---|---|---|---|
| search-radius collapse | `if frobenius_norm <= self.epsilon` (`gradient_free.rs:1366`) | `if (m_FrobeniusNorm <= m_Epsilon)` (`.hxx:225`) | **MATCH** (`<=`) |
| placement | after parent/child select, **before** rank-1 update; no update on the converging iteration (`:1357-1378`) | same: select → `fro_norm()` → test → return before update (`.hxx:202-256`) | **MATCH** |
| iteration cap | `for _ in 0..number_of_iterations` (`:1347`) | `for (…; m_CurrentIteration < m_MaximumIteration; …)` (`.hxx:143`) | **MATCH** (`<`) |
| defaults | ε `1.5e-4`, radius `1.01`, grow `1.05`, shrink `grow^-0.25` (`:1261,1327-1330`) | same (`.hxx:29-32`) | **MATCH** |

### 9. ExhaustiveOptimizerv4 — noted, not iterative-convergent
No convergence test on either side; walks the full `∏(2·steps+1)` grid and always
reports max-iterations (`gradient_free.rs:1499`). **Skipped** per the brief.

### 10. ImageRegistrationMethodv4 multi-resolution level advance — `method.rs` ↔ `itkImageRegistrationMethodv4.hxx`
| Element | Port | ITK | Verdict |
|---|---|---|---|
| level loop | `for (level, …) in schedule.iter().enumerate()` — plain, runs every level (`method.rs:2116`) | `for (m_CurrentLevel=0; m_CurrentLevel < m_NumberOfLevels; ++)` (`.hxx:796`) | **MATCH** (`<`, counted) |
| metric-based level early stop | none | none (grep for `ConvergenceMonitoring` in `GenerateData` → 0 hits) | **MATCH** (no early stop) |

Level advance is purely the counter on both sides; each level runs its optimizer
to that optimizer's own stop (findings 1–8), then the loop advances. No
whole-run convergence test spans levels.

### 11. Demons / FiniteDifference halt — `demons/common.rs` + `demons/mod.rs` ↔ `itkFiniteDifferenceImageFilter.hxx`
Shared by every demons flavor (`demons_registration`, fast-symmetric, symmetric,
diffeomorphic, level-set-motion — all route through `common::halt`).
| Stop | Port | ITK `Halt()` | Verdict |
|---|---|---|---|
| iteration cap | `if elapsed >= number_of_iterations { true }` (`common.rs:114`) | `if (GetElapsedIterations() >= m_NumberOfIterations)` (`.hxx:217`) | **MATCH** (`>=`) |
| iteration-0 guard | `if elapsed == 0 { false }` (`common.rs:117`) | `if (GetElapsedIterations() == 0) return false;` (`.hxx:221`) | **MATCH** |
| RMS early stop | `maximum_rms_error > rms_change` (`common.rs:120`) | `else if (GetMaximumRMSError() > m_RMSChange)` (`.hxx:225`) | **MATCH** (`>` — stop when rms_change strictly below threshold) |
| increment timing | `elapsed_iterations += 1` after apply (`mod.rs:416`) | `++m_ElapsedIterations` after `ApplyUpdate` (`.hxx:80`) | **MATCH** |

Demons has no metric-based early stop; `GetMetric()` is inspection-only on both
sides (`itkDemonsRegistrationFilter.hxx:67-76`). `StopRegistration()` is
unreachable in the synchronous port (documented, `demons/mod.rs:94-99`).

---

## Divergences / candidates

### A — REAL (deliberate): the gradient-descent family carries a min-step stop ITK lacks
**Port.** `GradientDescentOptimizer` (`optimizer.rs:250`), `GradientDescentLineSearchOptimizer`
(`:805`), and `ConjugateGradientLineSearchOptimizer` (`:1071`) each end an
iteration with `if step_sq.sqrt() < self.min_step_tolerance { StepTooSmall }`,
default `min_step_tolerance = 1e-8` (`:127,605-628,859-882`). For the **fixed-rate**
entry point `set_optimizer_as_gradient_descent` the port also does **not** enable
the value-plateau monitor (`method.rs:905-916` — no `set_convergence`), so min-step
is the *only* early stop besides the cap.

**ITK.** `GradientDescentOptimizerv4` (and its line-search subclasses, which
inherit its loop) has **no minimum-step-length stop** — its only early stop is the
`WindowConvergenceMonitoringFunction` (`itkGradientDescentOptimizerv4.hxx:82,126`).
A minimum step length exists *only* in `RegularStepGradientDescentOptimizerv4`
(`m_MinimumStepLength`, `.hxx:143`), which the port's `RegularStep` matches exactly
(finding 1). SimpleITK's `SetOptimizerAsGradientDescent(learningRate,
numberOfIterations, convergenceMinimumValue=1e-6, convergenceWindowSize=10, …)`
configures the convergence monitor on that base optimizer.

**Why it diverges.** ITK's fixed-rate gradient descent stops when the windowed
metric-value slope `≤ 1e-6`; the port's stops when the scaled step norm `< 1e-8`.
These are different criteria over different quantities, so they stop at different
iterations in general.

**Separating input.** Any smooth objective under a fixed learning rate: as the
gradient shrinks, the step-norm crosses `1e-8` and the metric-slope crosses `1e-6`
at different iterations. E.g. `f(p)=½p²`, lr `0.1`: the port stops when
`|0.1·p| < 1e-8` (`|p| < 1e-7`); ITK stops when the windowed slope of the last 10
`½p²` values drops to `1e-6` — a different iterate.

**To confirm at merge (both-side residual).** SimpleITK
`sitkImageRegistrationMethod.cxx` `SetOptimizerAsGradientDescent` →
`optimizer->SetConvergenceWindowSize(...)` / `SetMinimumConvergenceValue(...)`
wiring, to pin that ITK's fixed-rate path is convergence-monitored and never
min-step-stopped. The optimizer-class facts above are already confirmed.

### B — NEEDS-DECISION (stop-reason only): cap-vs-criterion reorder
**Port** tests its early criterion **before** the iteration cap:
- base GD: convergence `cv <= min_cv` (`optimizer.rs:222-230`) **then** `taken >= number_of_iterations` (`:232`).
- RegularStep: grad-mag (`:414`) and min-step (`:436`) **then** `taken >= number_of_iterations` (`:441`).
- line-search/CG: convergence (`:750,1001`) **then** cap (`:760,1011`).

**ITK** tests the iteration cap at the **top** of the loop, before evaluating the
metric or entering `AdvanceOneStep`: `if (m_CurrentIteration >= m_NumberOfIterations)`
(`itkGradientDescentOptimizerv4.hxx:82`), *then* the metric eval (`:99`), *then*
convergence (`:126`), *then* the step (`:156`). RegularStep's grad-mag/min-step
live inside `AdvanceOneStep` (`.hxx:115,143`), reached only after the cap break.

**Effect.** Before the cap the two check identical criteria at identical iterates,
so they stop at the **same iteration** with the **same parameters**. They differ
*only* at the coincident boundary — when a criterion is first satisfied at exactly
the iteration equal to the cap: ITK breaks on the cap (reports max-iterations)
before ever testing that criterion, while the port tests the criterion first and
reports Converged / GradientConverged / StepTooSmall. **Iteration count and final
parameters are bit-identical**; only the reported `StopReason` enum differs.

**Verdict.** NOT-REAL for "stops at a different iteration / point". NEEDS-DECISION
only if `StopReason`/`StopCondition` enum parity at the simultaneous boundary is
in scope. Separating input for the reason enum: a fixed run whose gradient magnitude
first drops below tolerance at exactly iteration `N = number_of_iterations` — port
reports GradientConverged, ITK reports max-iterations, both at `N` steps.

### C — NEEDS-DECISION (reachability-gated): all-zero-energy guard in the convergence monitor
**Port** short-circuits `if self.total_energy == 0.0 { return Some(0.0) }`
(`convergence.rs:72-74`), so an all-zero window reports convergence value `0`
(`≤ min_cv` → Converged).

**ITK** has no such guard: `GetConvergenceValue` normalizes by `m_TotalEnergy`
(`itkWindowConvergenceMonitoringFunction.hxx:102`, `m_EnergyValues[n] / m_TotalEnergy`).
With `m_TotalEnergy == 0` this is `0/0 = NaN`, the fitted slope is NaN, and the
optimizer's `NaN <= m_MinimumConvergenceValue` (`itkGradientDescentOptimizerv4.hxx:126`)
is **false** — ITK does **not** stop; it runs to another criterion.

**Reachability.** Requires every metric value ever added to be exactly `0.0`
(identical-image mean-squares) **and** the window to fill (≥10 values) before any
other stop fires. In the default wiring finding B's min-step (`1e-8`) fires at
iteration 1 (zero gradient → zero step) long before the window fills, masking this
path. It is reachable only with `min_step_tolerance` set to `0` on a perfectly-matched
mean-squares run. Flagged for completeness; practically unreachable in the shipped
configuration.

---

## Anchor bounding — where a stop condition hides from a loop-shape scan
A naive `rg` over `while` / `for … in 0..max_iter` / `break` / `converged` / `< min_step`
would miss these:
- **LBFGS2 line-search failure** is a `?`-style early return, not a loop token:
  `if self.line_search(...).is_err() { … LineSearchFailed; break }` (`lbfgs2.rs:877-887`).
- **LBFGS-B max-func-evals** is folded into a helper returning `bool`:
  `Book::record` (`lbfgsb.rs:1216-1223`); the driver's `break 'outer` sites read it.
- **LBFGS-B restart-on-singular** (`continue 'outer` at `:1364,1405,1510`) is
  *non-terminating* control flow that a break-scan could misread as a stop.
- **Golden-section recursion cap** is a bare depth `return`, no loop token
  (`optimizer.rs:497`).
- **Amoeba's cap** counts **evaluations** (`cnt < max_evaluations`), not iterations
  — a `number_of_iterations` anchor scans past it (`gradient_free.rs:131`).
- **Demons halt** is one indirection from its loop: `while !halt(...)`
  (`demons/mod.rs:377`) calling `common::halt` (`common.rs:108`).
- **Powell** has no `min_step` token; its stop is a value-tolerance inequality
  inside the outer `for` (`gradient_free.rs:665`), and the line search's collapse
  is a compound `||` (`:442-444`).

---

## Finding A — reachability addendum: **LIVE** (measured, not reasoned)

**Verdict: LIVE.** Under the shipped SimpleITK default wiring, the min-step stop is
the *first-satisfied* predicate on ordinary smooth objectives — not only for the
fixed-rate path (where the port installs no monitor at all) but even for the
line-search and conjugate-gradient paths where the value-plateau monitor *is*
present. So finding A changes iteration counts and final iterates on real runs; it
is not a theoretical-only divergence. It is dominated only by an ill-conditioned
objective whose golden-section step stays large while the metric value plateaus.

Measured by driving the **real port optimizers** through their public API at the
exact wiring `method.rs` installs (`CONVERGENCE_WINDOW_SIZE = 10`,
`MINIMUM_CONVERGENCE_VALUE = 1e-6`, default `min_step_tolerance = 1e-8`), cap
`1_000_000` so the cap never masks. Harness: throwaway binary path-depending on
`sitk-registration` (no crate code touched).

### Setter → optimizer → convergence wiring (the user-facing surface)

| SimpleITK-style setter (`method.rs`) | port optimizer | monitor `set_convergence(10,1e-6)`? | min-step 1e-8? | ITK counterpart's early stop |
|---|---|---|---|---|
| `set_optimizer_as_gradient_descent` (`:905-916`) | `GradientDescentOptimizer` | **NO** | yes | monitor only (`GradientDescentOptimizerv4`) |
| `set_optimizer_as_gradient_descent_estimated`, `EachIteration` (`:936-940`) | `GradientDescentOptimizer` | yes | yes | monitor only |
| `set_optimizer_as_gradient_descent_estimated`, `Once` (`:934-943`) | `GradientDescentOptimizer` | **NO** | yes | monitor only |
| `set_optimizer_as_gradient_descent_line_search[_estimated]` (`:1014-1044`) | `GradientDescentLineSearchOptimizer` | yes | yes | monitor only (inherits base loop) |
| `set_optimizer_as_conjugate_gradient_line_search[_estimated]` (`:1057-1086`) | `ConjugateGradientLineSearchOptimizer` | yes | yes | monitor only (inherits base loop) |
| `set_optimizer_as_regular_step_gradient_descent[_estimated]` (`:957-1003`) | `RegularStepGradientDescentOptimizer` | n/a | **matches ITK** | RegularStep's own min-step+grad-mag — **not finding A** |

`RegularStep` is excluded from A: ITK's `RegularStepGradientDescentOptimizerv4`
carries the same min-step, so the port matches it (verified match #1). A is the
three optimizers whose ITK base (`GradientDescentOptimizerv4`) has *no* min-step.

### Measured evidence

**(1) Fixed-rate `set_optimizer_as_gradient_descent` — LIVE by construction, quantified.**
The port installs no monitor, so min-step is the only non-cap stop. Same setter, same
defaults, same run, port-min-step vs the ITK-faithful monitor stand-in:

| objective | lr | port (`min-step 1e-8`, no monitor) | ITK stand-in (monitor `10,1e-6`) |
|---|---|---|---|
| bowl `(x−3)²+(y+2)²` | 0.1 | **StepTooSmall @ 83**, val 1.06e-15 | **Converged @ 37**, val 8.76e-7 |
| shallow `1e-3·bowl` | 0.5 | StepTooSmall @ 12791 | Converged @ 1592 |

83 vs 37 iterations, and different final iterates (1.06e-15 vs 8.76e-7). This is the
separating input made concrete: same `SetOptimizerAsGradientDescent(0.1, N)` call,
two different stop iterations and two different transforms. (The ITK stand-in is the
port's own `WindowConvergenceMonitor`, which verified match #2 pins byte-for-byte to
`itkWindowConvergenceMonitoringFunction`; it is a faithful proxy for ITK's stop
*mechanism* here. A fixed lr≥0.5 on the un-damped bowl overshoots and the monitor's
negated-slope fit reads a falling windowed value while diverging — an ITK-faithful
artifact of that monitor, orthogonal to A; the well-damped lr=0.1 bowl is the clean
comparison.)

**(2) Line-search path — monitor ON + min-step 1e-8, exactly as `method.rs` wires it.**
Min-step still fires first on well-conditioned objectives:

| objective | lr | stop_reason | iters |
|---|---|---|---|
| bowl | 0.1 / 1.0 | **StepTooSmall** | 5 / 5 |
| shallow | 0.1 / 1.0 | **StepTooSmall** | 9 / 7 |
| valley `(x−3)²+50(y+2)²` | 0.1 / 1.0 | Converged (monitor) | 109 / 77 |

**(3) Conjugate-gradient path — monitor ON + min-step 1e-8.** Same pattern:

| objective | lr | stop_reason | iters |
|---|---|---|---|
| bowl | 0.1 / 1.0 | **StepTooSmall** | 3 / 3 |
| shallow | 0.1 / 1.0 | **StepTooSmall** | 7 / 5 |
| valley | 0.1 / 1.0 | Converged (monitor) | 30 / 41 |

### Domination boundary (when A *is* latent)

Min-step is dominated only when the golden-section line search keeps taking a large
step while the metric value plateaus — i.e. an ill-conditioned objective (the
`condition ~50` valley), where the monitor's windowed-slope test trips before the
step-norm reaches `1e-8`. On well-conditioned smooth objectives — the common
registration case near the optimum, where the gradient (and thus the step) shrinks
smoothly to zero — the step-norm reaches `1e-8` while the windowed slope is still
above `1e-6`, so **min-step wins**. It is therefore not "always dominated"; the
document-and-move-on framing does not hold.

### Consequence for the merge decision

A is LIVE: on any well-conditioned smooth registration driven by
`set_optimizer_as_gradient_descent`, `…_line_search`, or `…_conjugate_gradient_line_search`
at defaults, the port stops on `StepTooSmall` at a **different iteration and a
different final transform** than ITK's monitor-only stop would produce. The fixed-rate
path is the strongest case — the port omits ITK's monitor entirely (83 vs 37 above).
This is a real output divergence for the merge to rule on, not a latent code smell.
