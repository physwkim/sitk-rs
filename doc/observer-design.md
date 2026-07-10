# Observer / introspection surface — design

Status: **proposal**, pending the §5.20 sign-off in
[`upstream-findings.md`](upstream-findings.md). No production code exists yet.

Scope: what SimpleITK spells `sitk::Command`, `sitk::EventEnum`,
`ProcessObject::AddCommand/Abort/GetProgress`, and
`ImageRegistrationMethod`'s "active measurements"
(`GetOptimizerIteration`, `GetMetricValue`, `StopRegistration`, …).

Reference checkouts: ITK `/home/stevek/work/ITK`
(`v6.0b02-5846-ge46eb723a5`), SimpleITK `/home/stevek/work/SimpleITK`.
Every line number below was read out of those trees.

---

## 1. What upstream actually is

### 1.1 The event set

`sitkEvent.h:31-64` — a plain C enum, values deliberately non-contiguous
(`sitkUserEvent = 7` is declared *after* `sitkMultiResolutionIterationEvent = 9`
but numbered before it):

| enum | value | doc comment (`sitkEvent.h`) |
|---|---|---|
| `sitkAnyEvent` | 0 | "Occurs for all event types." (`:33-35`) |
| `sitkAbortEvent` | 1 | "after the process has been aborted, but before exiting the Execute method" (`:36-38`) |
| `sitkDeleteEvent` | 2 | "when the underlying itk::ProcessObject is deleted" (`:39-41`) |
| `sitkEndEvent` | 3 | "at then end of normal processing" (`:42-44`) |
| `sitkIterationEvent` | 4 | "with some algorithms that run for a fixed or undetermined number of iterations" (`:45-47`) |
| `sitkProgressEvent` | 5 | "when the progress changes in most process objects" (`:48-50`) |
| `sitkStartEvent` | 6 | "when then itk::ProcessObject is starting" (`:51-53`) |
| `sitkUserEvent` | 7 | "Other events may fall into this enumeration." (`:61-63`) |
| `sitkMultiResolutionIterationEvent` | 9 | "when some filters change processing to a different scale … a sub-event of the more general IterationEvent. The general iteration event will also be invoked." (`:54-60`) |

Each maps 1:1 onto a file-static `itk::…Event` object
(`sitkProcessObject.cxx:39-47`) and is looked up by
`ProcessObject::GetITKEventObject` (`sitkProcessObject.cxx:524-550`), which
throws on an unknown enumerator.

### 1.2 `Command` and `FunctionCommand`

`sitkCommand.h:44-78`. `Command` is an abstract-ish base with one virtual
`Execute()` **taking no arguments** (`:60-61`) — the callback learns nothing
about which event fired or which object fired it. It carries a name
(`:53-57`) and derives from `ObjectOwnedBase` (`sitkObjectOwnedBase.h:47-118`),
which is a multimap of `itk::Object*` → on-delete callback.

`FunctionCommand` (`sitkFunctionCommand.h:33-75`) wraps a `std::function<void()>`,
settable from a free function, a member function, a `void*`-clientData
C function, or a lambda.

The ITK side is `SimpleAdaptorCommand` (`sitkProcessObject.cxx:53-97`): an
`itk::Command` holding a **raw** `sitk::Command*`, whose
`Execute(Object*, const EventObject&)` discards both arguments and calls
`m_That->Execute()`.

### 1.3 Attachment, lifetime, and the dangling-command rule

`ProcessObject::AddCommand(event, cmd)` (`sitkProcessObject.cxx:374-393`):

1. push `{event, &cmd}` onto `m_Commands`;
2. `cmd.AddProcessObject(this)` — registers a reverse link so that when the
   *command* dies it tells the process object (`sitkCommand.cxx:44-48` →
   `ObjectOwnedBase::AddObjectCallback`, which fires `o->onCommandDelete(this)`
   from the command's destructor);
3. if a process is already active, immediately attach an ITK observer;
   otherwise mark the tag as `ULONG_MAX` ("not registered").

The `std::function` overload (`sitkProcessObject.cxx:395-405`) heap-allocates a
`FunctionCommand`, adds it, then sets `OwnedByObjectsOn()` and leaks the
`unique_ptr` — the command now dies when the last process object drops it.

`onCommandDelete` (`sitkProcessObject.cxx:586-607`) removes the entry and
detaches the ITK observer. So a stack-allocated `Command` that goes out of scope
while attached is *handled*; that is the whole point of `ObjectOwnedBase`.

What is **not** handled — the two documented UB clauses:

- "Deleting a command this object has during a command call-back will produce
  undefined behavior." (`sitkProcessObject.h:265-266`)
- `RemoveAllCommands`: "Calling when this object is invoking anther command will
  produce undefined behavior." (`sitkProcessObject.h:284-288`)

i.e. **re-entrant mutation of the observer list is UB.** Reading is fine.

`PreUpdate` (`sitkProcessObject.cxx:472-505`) sets `m_ActiveProcess`, hangs a
delete-observer on it, then registers every stored command. `OnActiveProcessDelete`
(`:564-583`) caches `GetProgress()` into `m_ProgressMeasurement`, resets every
ITK tag, and nulls `m_ActiveProcess`.

There is **no `ProtectedCommand`** anywhere in this SimpleITK checkout
(`rg -il protectedcommand` over both trees: no hits). The lifetime mechanism is
`ObjectOwnedBase` + `onCommandDelete`, described above.

### 1.4 Re-entrancy: what a callback may call

`sitkProcessObject.h:258-262`, verbatim:

> Unless specified otherwise, it's safe to get any value during execution.
> "Measurements" will have valid values only after the Execute method has
> returned. "Active Measurements" will have valid values during events, and
> access the underlying ITK object.

Concretely, on `ProcessObject`:

- `GetProgress()` (`sitkProcessObject.cxx:451-459`) — active: reads
  `m_ActiveProcess->GetProgress()` when running, else the cached value.
- `Abort()` (`:462-469`) — active: `m_ActiveProcess->AbortGenerateDataOn()`;
  **no-op when nothing is running** (`sitkProcessObject.h:322`).
- `HasCommand`, `GetName`, `ToString` — pure reads, safe.
- `AddCommand` / `RemoveAllCommands` from inside a callback — UB (§1.3).

On `ImageRegistrationMethod`, the active measurements are nine `std::function`
slots (`sitkImageRegistrationMethod.h:804-815`) bound to the live ITK optimizer
in `CreateOptimizer` and cleared in `OnActiveProcessDelete`
(`sitkImageRegistrationMethod.cxx:1205-1225`). Each getter is
"call the slot if bound, else return the cached/zero fallback"
(`:560-658`). Two consequences a port must reproduce or consciously drop:

- **Outside `Execute` the getters silently return stale or zero values**, they
  do not throw. `GetOptimizerLearningRate()` returns `0.0`
  (`:593-601`), `GetOptimizerPosition()` returns an empty vector (`:582-590`).
- **Not every optimizer binds every slot.** Only the four gradient-descent-family
  branches bind learning-rate and convergence-value
  (`sitkImageRegistrationMethod_CreateOptimizer.cxx:118-119, 148-149, 182-183,
  216-217`), plus `OnePlusOneEvolutionary` binding convergence-value alone to
  `GetFrobeniusNorm()` (`:424-426`). LBFGSB (`:277`), LBFGS2 (`:307`) and
  Amoeba (`:361`) explicitly set `m_pfOptimizerStopRegistration = nullptr`, and
  `StopRegistration()` returns `false` for them
  (`sitkImageRegistrationMethod.cxx:560-568`;
  `sitkImageRegistrationMethod.h:740-747`). `Exhaustive` maps stop to
  `StopWalking()` (`:334`).

### 1.5 Where the events are actually fired (ITK)

- **Start / End**: `itk::ProcessObject::UpdateOutputData` fires `StartEvent`
  before `GenerateData()` (`itkProcessObject.cxx:1673`) and `EndEvent` after
  (`:1711`).
- **Progress**: `UpdateProgress(float)` clamps to `[0,1]`, stores, and invokes
  `ProgressEvent` (`itkProcessObject.cxx:1135-1142`). `IncrementProgress` adds
  atomically but **only invokes the event on the update thread**
  (`:1145-1165`) — worker threads increment silently. Granularity comes from
  `ProgressReporter`, which fires once every `numPixels / numUpdates` pixels,
  `numUpdates` defaulting to 100 (`itkProgressReporter.h:30-36, 95-107`;
  `itkProgressReporter.cxx:41`). **The contract is a monotone float in `[0,1]`,
  nothing more**: no promise of how many events, of ever observing `1.0` mid-run,
  or of ordering against `IterationEvent`.
- **Iteration** (finite-difference filters): `FiniteDifferenceImageFilter`'s
  `GenerateData` loop fires `IterationEvent` once per iteration
  (`itkFiniteDifferenceImageFilter.hxx:71-86`), and its `Halt()` sets progress
  to `elapsed / m_NumberOfIterations` (`:210-214`). This is the one filter
  family where progress is per-iteration rather than per-pixel.
- **Iteration** (optimizers): `GradientDescentOptimizerv4::ResumeOptimization`
  fires `IterationEvent` **before** `AdvanceOneStep()`, so an observer reads the
  metric value and position *at the iterate where the value was evaluated*
  (`itkGradientDescentOptimizerv4.hxx:142-144`; ITK commit `cf929f2ca7`,
  on `origin/main`, "BUG: Fire IterationEvent before AdvanceOneStep in v4
  gradient optimizers"). Immediately after the event it re-checks `m_Stop` and
  breaks *before stepping* (`:146-152`) — that is what makes
  `StopRegistration()` from inside a callback land on the current iterate.
  Not every v4 optimizer fires it: the eight that do are `ExhaustiveOptimizerv4`,
  `MultiStartOptimizerv4`, `MultiGradientOptimizerv4`, `LBFGSOptimizerBasev4`,
  `LBFGS2Optimizerv4`, `GradientDescentOptimizerv4`, `PowellOptimizerv4`,
  `OnePlusOneEvolutionaryOptimizerv4`. **`AmoebaOptimizerv4` never fires
  `IterationEvent`.**
- **MultiResolutionIteration**: `ImageRegistrationMethodv4::
  InitializeRegistrationAtEachLevel` fires it (`itkImageRegistrationMethodv4.hxx:318`),
  and that function is called once per level from `GenerateData`
  (`:796-803`). So: once per level, before that level's optimization.
- **Abort**: `sitk::ProcessObject::Abort()` only sets `AbortGenerateDataOn`.
  Two distinct downstream behaviors exist in ITK:
  1. *Throwing*: `ProgressReporter::CheckAbortGenerateData` /
     `TotalProgressReporter::CheckAbortGenerateData` throw `ProcessAborted`
     (`itkProgressReporter.h:76-91`, `itkTotalProgressReporter.h:61-75`).
     `UpdateOutputData` catches it, fires `AbortEvent`, resets the pipeline, and
     **rethrows** (`itkProcessObject.cxx:1686-1692`). Only the multi-threaders
     call these (`itkTBBMultiThreader.cxx`, `itkSingleMultiThreader.cxx`).
  2. *Cooperative*: 19 filters poll `GetAbortGenerateData()` and break their own
     loop (`itkSTAPLEImageFilter.hxx`, `itkFastMarchingImageFilter.hxx`,
     `itkConfidenceConnectedImageFilter.hxx`, …). `UpdateOutputData` then sees
     `m_AbortGenerateData` set, pushes progress to `1.0`
     (`itkProcessObject.cxx:1704-1708`), and fires `EndEvent` normally — **no
     `AbortEvent`, no exception**.

  `AbortEvent` therefore fires on path (1) only. SimpleITK catches
  `ProcessAborted` nowhere (`rg -n "ProcessAborted" Code/`: zero hits), so
  `sitkProcessObject.h:315-317`'s claim — "The expected behavior is that not
  exception should be throw out of this processes Execute method" — is **false
  for path (1)**: the `ProcessAborted` (an `itk::ExceptionObject` subclass,
  `itkExceptionObject.h:223-228`) escapes `Execute`. Candidate §3 ledger entry;
  not filed here, this document only claims the §5.20 row.

### 1.6 The registration method's own routing

`ImageRegistrationMethod::AddITKObserver` (`sitkImageRegistrationMethod.cxx:1179-1189`)
sends `sitkIterationEvent` to **`m_ActiveOptimizer`** and everything else to the
registration `ProcessObject`. `RemoveITKObserver` mirrors it (`:1191-1202`).
Three facts fall out, all citable, all load-bearing for a port:

1. **`sitkAnyEvent` is not the union of the events.** An `AnyEvent` observer is
   attached to the registration process object, which never invokes
   `IterationEvent` — the optimizer does. So `AnyEvent` misses every optimizer
   iteration.
2. **`sitkEvent.h:57-59`'s promise that `MultiResolutionIterationEvent` also
   invokes the general `IterationEvent` does not hold here.** ITK's
   `CheckEvent` hierarchy would deliver it, but only to observers *on the
   registration object* — and `sitkIterationEvent` observers were routed away to
   the optimizer.
3. **`ImageRegistrationMethod::Abort()` is a no-op.** It sets
   `AbortGenerateDataOn` on the registration process object; neither
   `ImageRegistrationMethodv4` nor any `Optimizersv4` class reads
   `GetAbortGenerateData` (`rg -l GetAbortGenerateData Modules` — 19 files,
   none of them either). `StopRegistration()` is the only working cancellation.
4. **No `ProgressEvent` during registration.** `ImageRegistrationMethodv4`
   never calls `UpdateProgress` (`rg -n "UpdateProgress" itkImageRegistrationMethodv4.hxx`
   — zero hits), so `GetProgress()` stays `0.0` for the whole run.

### 1.7 The procedural API has no observers at all

Every SimpleITK free function is generated from
`ExpandTemplateGenerator/templates/ProceduralAPI.cxx.jinja:5-21`:

```
Image Median ( const Image& image1, ... ) {
  MedianImageFilter filter;
  filter.SetRadius( radius );
  return filter.Execute( image1 );
}
```

The filter object is a local; the caller never sees it, so **no command can be
attached to a procedural call.** Observing a filter upstream *requires* the
object interface (`MedianImageFilter f; f.AddCommand(...); f.Execute(img);`).

This is the single most important fact for scoping: `sitk-filters`' free
functions are the faithful port of the *procedural* API, and the procedural API
has no observer surface to be missing. Adding progress to them is a **new**
object-interface wave, not a parity fix.

---

## 2. This port's current shape

- `crates/sitk-registration/src/method.rs:1757-1818` — `execute(&self, fixed,
  moving, initial)` validates, builds the pyramid schedule
  (`level_schedule`), then loops levels calling `run_single_level`
  (`:2069`), threading the transform through and keeping the last level's
  diagnostics.
- `RegistrationResult<T>` (`method.rs:599-610`) already carries
  `metric_value`, `iterations`, `stop_reason`, `valid_points` — the
  post-hoc half of upstream's "Measurements".
- `run_single_level` builds the metric, scales, and estimator, then builds an
  `objective!()` closure that mutably borrows `transform` and hands it to
  `optimizer.optimize(...)` (`method.rs:2189-2201, 2227-2260`).
- Each optimizer owns its own loop. `GradientDescentOptimizer::optimize_with_lr`
  (`optimizer.rs:188-260`) is representative: per iteration it runs the
  convergence monitor, checks the iteration cap, computes `lr`, steps, re-evals,
  and checks the step tolerance. `OptimizerResult` (`optimizer.rs:91-102`) is
  `{parameters, value, iterations, stop_reason}`; `StopReason`
  (`optimizer.rs:69-87`) has six variants, none for user cancellation.
- `crates/sitk-filters` — 94 modules of free functions returning
  `Result<Image>`. No filter object, no shared state, nothing to attach to.
- Ledger §6 names the gap: "the observer/introspection surface
  (`GetOptimizerIteration`/`GetMetricValue`/…/`StopRegistration`, upstream
  `sitkCommand.h`/`sitkEvent.h`)".

---

## 3. Design decisions

### 3.1 Callback representation

Three candidate shapes:

**(a) Observers passed *into* `execute` — recommended.**

```rust
/// What an iteration callback learns. Every field is a loop local; the
/// callback never names the method, so there is no reentrancy.
#[non_exhaustive]
pub struct IterationContext<'a> {
    /// Zero-based resolution level (upstream `GetCurrentLevel`).
    pub level: usize,
    /// Steps taken by the current level's optimizer (upstream
    /// `GetOptimizerIteration`).
    pub iteration: usize,
    /// Metric value at `position` (upstream `GetMetricValue`).
    pub metric_value: f64,
    /// The current iterate (upstream `GetOptimizerPosition`).
    pub position: &'a [f64],
    /// Per-parameter scales in force (upstream `GetOptimizerScales`).
    pub scales: &'a [f64],
    /// `None` for optimizers that do not have one — mirrors upstream's unbound
    /// `m_pfGetOptimizerLearningRate`, except it says so in the type instead of
    /// returning `0.0`.
    pub learning_rate: Option<f64>,
    /// `None` unless convergence monitoring is on (upstream
    /// `GetOptimizerConvergenceValue`).
    pub convergence_value: Option<f64>,
}

/// The callback's answer, i.e. upstream's `StopRegistration()` inverted into a
/// return value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Control { Continue, Stop }

/// Observers for one `execute` call. Not `Clone`; borrows live for the call.
#[derive(Default)]
pub struct Observers<'cb> {
    on_start: Vec<Box<dyn FnMut() + 'cb>>,
    on_end: Vec<Box<dyn FnMut() + 'cb>>,
    on_multi_resolution_iteration: Vec<Box<dyn FnMut(&IterationContext<'_>) + 'cb>>,
    on_iteration: Vec<Box<dyn FnMut(&IterationContext<'_>) -> Control + 'cb>>,
}

impl<'cb> Observers<'cb> {
    pub fn on_iteration(&mut self, f: impl FnMut(&IterationContext<'_>) -> Control + 'cb) -> &mut Self;
    pub fn on_multi_resolution_iteration(&mut self, f: impl FnMut(&IterationContext<'_>) + 'cb) -> &mut Self;
    pub fn on_start(&mut self, f: impl FnMut() + 'cb) -> &mut Self;
    pub fn on_end(&mut self, f: impl FnMut() + 'cb) -> &mut Self;
}

impl ImageRegistrationMethod {
    pub fn execute_observed<T: ParametricTransform>(
        &self, fixed: &Image, moving: &Image, initial: T,
        observers: &mut Observers<'_>,
    ) -> Result<RegistrationResult<T>>;
}
```

`execute` keeps its signature and becomes
`self.execute_observed(fixed, moving, initial, &mut Observers::default())`.

Why this and not the others:

- **The classic borrow problem disappears by construction.** Upstream's callback
  takes no arguments and reaches back into the method
  (`method.GetOptimizerIteration()`) while the method is mid-`Execute`. In Rust
  that is `&self` inside a `&mut self` region, and — worse — the objective
  closure in `run_single_level` already holds `&mut transform`. Passing the
  measurements *into* the callback as a `&IterationContext` built from the
  optimizer's own loop locals means the callback never needs a handle on the
  method at all. No `RefCell`, no `Rc<Cell<…>>`, no split-borrow gymnastics on
  `self`.
- **Upstream's two UB clauses become unrepresentable.** "Delete a command during
  a callback" and "`RemoveAllCommands` while invoking" (`sitkProcessObject.h:265-266,
  284-288`) are exactly the aliasing the borrow checker rejects: `Observers` is
  uniquely borrowed for the duration of `execute_observed`, so no code can touch
  it from inside a callback. That is a structural closure of the whole defect
  family, not a runtime guard.
- **No `'static` bound.** A closure capturing `&mut log_file` or `&mut Vec<f64>`
  works, which is upstream's stack-allocated-`Command` use case
  (`sitkCommand.h:39-41`).
- **`StopRegistration` becomes a return value**, checked by the optimizer at the
  exact point ITK checks `m_Stop` (`itkGradientDescentOptimizerv4.hxx:146-152`).
  A callback cannot mutate the optimizer, so the "callback pokes the object it
  is being called from" hazard never arises.

**(b) `add_command`-stored boxes + `&mut self` execute** — closest to upstream's
spelling (`method.add_command(Event::Iteration, cmd)`), but the stored
`Box<dyn FnMut + 'static>` forces every callback to own its captures, or forces
a lifetime parameter onto `ImageRegistrationMethod<'cb>` which then infects
every field, `Default`, and every caller. It also re-opens the reentrancy
question: with observers stored on `self`, invoking them needs `&mut self` while
the level loop already holds `&self` borrows of the metric configuration. The
only way out is interior mutability — a runtime gate for a problem option (a)
does not have.

**(c) Channels** (`Sender<RegistrationEvent>`) — trivially `Send`, decoupled,
and wrong for this surface. `StopRegistration` has no return path, so
cancellation needs a second `AtomicBool` side channel; `position`/`scales` must
be cloned into every message; and the delivery is asynchronous, so a receiver
can no longer observe "the metric value at the iterate where it was evaluated",
which is precisely the guarantee ITK's commit `cf929f2ca7` exists to provide.
Worth offering later as a thin adapter *on top of* (a):
`observers.on_iteration(|c| { let _ = tx.send(c.into()); Control::Continue })`.

**Recommendation: (a).** It is the structural fix; (b) and (c) can both be built
from it, and neither can be reached from the other.

### 3.2 Where `valid_points` goes (a known phase-1 gap)

Upstream's `GetMetricNumberOfValidPoints()` is an active measurement
(`sitkImageRegistrationMethod.h:707-717`) because the ITK metric object is
shared and can be re-queried at any time. In this port, the count is produced by
`metric.evaluate()` (`metric.rs:658, 764`) and dropped on the floor by the
`objective!()` closure, which returns `(value, derivative)`.

The optimizer loop therefore *cannot* see it, and the objective closure cannot
hand it over: both are `&mut`-borrowed by the same `optimize` call, so no shared
local can bridge them without interior mutability.

The structural fix is to widen the objective's return type so the measurement
travels with the evaluation that produced it:

```rust
pub struct Evaluation { pub value: f64, pub derivative: Vec<f64>, pub valid_points: usize }
// F: FnMut(&[f64]) -> Evaluation
```

That touches every optimizer in `optimizer.rs`, `lbfgsb.rs`, `lbfgs2.rs`,
`gradient_free.rs`. **Phase 1 ships `IterationContext` without
`valid_points`** (the struct is `#[non_exhaustive]` precisely so the field can
be added), and the widening is phase 1b. Do not fake it with a `Cell`: a field
whose value depends on which closure last ran is the dual-meaning smell.

### 3.3 Abort semantics

**Registration.** `Control::Stop` from an iteration callback breaks the loop
before the next step, exactly where ITK breaks (`itkGradientDescentOptimizerv4.hxx:146-152`).
The run then completes normally: upstream's `StopOptimization()` makes
`registration->Update()` return, and `Execute` returns the transform at the
current iterate. So this is **`Ok`, not `Err`** — the signal lands in
`StopReason`:

```rust
pub enum StopReason { …, /// A `on_iteration` observer returned `Control::Stop`
                          /// (upstream `StopRegistration()` →
                          /// "StopOptimization() called from IterationEvent observer").
                          StoppedByObserver }
```

`ImageRegistrationMethod::Abort()` is **not ported**: it is a documented no-op
upstream for this class (§1.6.3). Porting a no-op would be a lie in the type
system.

Optimizers that cannot be stopped upstream (LBFGSB, LBFGS2, Amoeba —
`sitkImageRegistrationMethod.h:740-747`) must not silently honor `Control::Stop`
either. Two sub-choices, and phase 1 takes the second:
ignore the return value (upstream's `StopRegistration() -> false`), or refuse at
configuration time. **Take the upstream shape**: honor `Stop` where the loop
supports it; where it does not, ignore it and document per optimizer. Amoeba
additionally fires no `IterationEvent` at all upstream (§1.5), so this port must
not invoke `on_iteration` for it — otherwise the port's callback count diverges
from SimpleITK's for the same script.

**Filters.** There is nothing to abort: the free functions are the procedural
API and the procedural API has no observers (§1.7). If a future object-interface
wave adds them, the faithful mapping of a cooperative abort
(`itkProcessObject.cxx:1704-1708`) is `Ok(partial_image)` with progress forced
to `1.0` and content that upstream itself calls "valid but undefined"
(`sitkProcessObject.h:315-321`). Handing a Rust caller an `Ok` image of
undefined content is a footgun; the recommendation *when that wave happens* is
`Err(FilterError::Aborted)` and a §4 divergence row. That decision is out of
scope here and deliberately not part of §5.20.

### 3.4 Progress

Upstream's guarantee, restated from §1.5: a monotone `f32` in `[0,1]`, updated
about 100 times per filter by `ProgressReporter`, fired only on the update
thread, forced to `1.0` on abort. Nothing about count, timing, or interleaving
with `IterationEvent`.

**Registration reports no progress at all** upstream (§1.6.4). Phase 1 exposes
none. `on_multi_resolution_iteration` plus `on_iteration` give a caller strictly
more information than `GetProgress()` would.

**Filters that could report progress without restructuring** — those with an
explicit outer loop over a *known* iteration count, which is exactly ITK's
`FiniteDifferenceImageFilter::Halt` family (`elapsed / m_NumberOfIterations`,
`itkFiniteDifferenceImageFilter.hxx:210-214`):

`anisotropic_diffusion.rs`, `min_max_curvature_flow.rs`, `denoise.rs`
(which owns `curvature_flow`), `demons/{level_set_motion, compose, common,
symmetric, diffeomorphic, fast_symmetric}`, `level_set/anti_alias.rs`
and the level-set segmentation filters, `chan_vese.rs`, `patch_based_denoising.rs`,
`n4_bias_field.rs`, `slic.rs`,
`displacement_field/{invert, iterative_inverse}.rs`, and
`region_growing.rs`'s confidence-connected loop (fixed iteration count, and the
one ITK filter in this list that polls `GetAbortGenerateData` directly).

Separable filters (`recursive_gaussian`, `bspline_decomposition.rs`,
`distance.rs`) have a per-axis outer loop — that is `1/dim` granularity, coarser
than upstream's per-pixel, and would be a §4 divergence rather than parity.

**Filters that cannot** — every one-pass `Zip` kernel (arithmetic, `cast`,
threshold, morphology inner loops). Upstream's per-pixel `ProgressReporter`
means threading a reporter handle into each kernel: 94 modules, every signature,
for a number nobody reads. That is the phase-2 cost, and it is why progress is
not in phase 1.

---

## 4. Scope split

**Phase 1 — registration events (this design, one crate, no filter churn).**

New in `sitk-registration`: `IterationContext`, `Control`, `Observers`,
`StopReason::StoppedByObserver`, `execute_observed`. Hook sites, each mapped to
its upstream firing point:

| event | hook site in this port | upstream firing point |
|---|---|---|
| start | `method.rs::execute`, after validation, before the level loop (between `:1794` and `:1795`) | `itkProcessObject.cxx:1673` |
| multi-resolution iteration | `method.rs::execute`, top of each level iteration, before `prepare_level` (`:1796`) | `itkImageRegistrationMethodv4.hxx:318`, called per level at `:796-803` |
| iteration | inside each optimizer loop, after the convergence + iteration-cap checks and **before** the parameter update — for `GradientDescentOptimizer::optimize_with_lr` that is immediately before `optimizer.rs:235`'s `let lr = learning_rate_of(&grad);` | `itkGradientDescentOptimizerv4.hxx:144` |
| `Control::Stop` honored | same loop, immediately after the callback, before the step | `itkGradientDescentOptimizerv4.hxx:146-152` |
| end | `method.rs::execute`, after the level loop, before building `RegistrationResult` (between `:1810` and `:1812`) | `itkProcessObject.cxx:1711` |

Plumbing: each optimizer's `optimize*` gains a sibling taking
`&mut dyn FnMut(&IterationContext<'_>) -> Control`; the existing `optimize*`
delegates with a no-op. `run_single_level` gains `level: usize` (it already has
it) and the observer reference, and builds the `IterationContext` from
`(level, taken, value, &p, &scales, lr, monitor.convergence_value())`. The
`Amoeba` branch does not call the hook (§3.3). LBFGSB / LBFGS2 call it and
discard `Control::Stop`, documented per optimizer.

Everything phase 1 needs already exists as a local in the loops it touches. No
type in `sitk-core`, `sitk-transform` or `sitk-filters` changes.

**Phase 1b — `valid_points` as an active measurement.** Widen the objective to
return `Evaluation` (§3.2). Additive to `IterationContext` because it is
`#[non_exhaustive]`.

**Phase 2 — filter progress/abort.** Requires an object interface for filters
(`MedianImageFilter { radius }` with `execute(&img)`), because the procedural
form structurally cannot carry observers (§1.7). Then a `ProgressReporter`
equivalent threaded into 94 modules. Large, independent, and gated on §5.20's
answer to (a)/(b)/(c) since the callback shape must be shared.

**Never (with reasons in §5).** `sitkDeleteEvent`, `sitkUserEvent`,
`sitkAnyEvent`, `Abort()` on the registration method, `RemoveAllCommands`.

---

## 5. API parity table

`RM` = `ImageRegistrationMethod`. "phase" is when it lands under the recommended
option (a).

| upstream | citation | proposed Rust | phase | parity notes |
|---|---|---|---|---|
| `class Command` (virtual `Execute()`) | `sitkCommand.h:44-78` | *(none)* — `FnMut` closures | — | A no-argument virtual `Execute()` carrying no event/sender is strictly less information than a Rust closure with `&IterationContext`. A `Command` trait would add a vtable and a naming/lifetime protocol for zero capability. |
| `FunctionCommand` | `sitkFunctionCommand.h:33-75` | `impl FnMut(..) + 'cb` | 1 | Direct. The `void* clientData` overload (`:60-66`) has no Rust meaning — captures replace it. |
| `Command::{Get,Set}Name` | `sitkCommand.h:53-57` | *(none)* | — | Names exist upstream only to make `ProcessObject::ToString()` (`sitkProcessObject.cxx:156-161`) printable. Nothing else reads them. |
| `ObjectOwnedBase` lifetime linkage | `sitkObjectOwnedBase.h:47-118`, `sitkCommand.cxx:44-54` | *(none)* | — | Its entire job is to stop a dangling `Command*`. Borrowed `Observers<'cb>` makes the dangle unrepresentable (§3.1). |
| `ProcessObject::AddCommand(EventEnum, Command&)` | `sitkProcessObject.h:271-272` | `Observers::on_iteration/on_start/on_end/on_multi_resolution_iteration` | 1 | One method per event instead of an enum argument: it lets each callback have the signature its event can actually supply (`on_iteration` returns `Control`, `on_start` does not). Return value: upstream's `int` is documented "reserved for latter usage" (`:269`) and is always `0` (`sitkProcessObject.cxx:392`) — dropped. |
| `AddCommand(EventEnum, std::function<void()>)` | `sitkProcessObject.h:280-281` | same as above | 1 | The lambda overload *is* the shape being ported; the `Command` overload is not. |
| `RemoveAllCommands()` | `sitkProcessObject.h:289-290` | *(none)* | — | The observer set is a call argument, not object state. `execute_observed(.., &mut Observers::default())` is "no commands". Also removes the UB clause at `:286-288`. |
| `HasCommand(EventEnum)` | `sitkProcessObject.h:293-294` | *(none)* | — | Callers construct the `Observers`; they know. |
| `GetProgress()` | `sitkProcessObject.h:306-307` | *(none)* for `RM` | — | `ImageRegistrationMethodv4` never calls `UpdateProgress`, so upstream's value is `0.0` for the whole run (§1.6.4). For filters, phase 2. |
| `Abort()` | `sitkProcessObject.h:324-325` | *(none)* for `RM` | — | No-op upstream for this class: nothing in `RegistrationMethodsv4`/`Optimizersv4` reads `GetAbortGenerateData` (§1.6.3). |
| `sitkStartEvent` | `sitkEvent.h:51-53` | `Observers::on_start` | 1 | Once, before level 0. |
| `sitkEndEvent` | `sitkEvent.h:42-44` | `Observers::on_end` | 1 | Once, after the last level, on the success path. On `Err` this port does not call it; upstream's `EndEvent` is also skipped when `GenerateData` throws (`itkProcessObject.cxx:1693-1698` rethrows before `:1711`). Faithful. |
| `sitkIterationEvent` | `sitkEvent.h:45-47` | `Observers::on_iteration` | 1 | Fires per optimizer step, before the step, for every optimizer except Amoeba (§1.5). Upstream routes it to the optimizer, not the registration object (`sitkImageRegistrationMethod.cxx:1179-1189`). |
| `sitkMultiResolutionIterationEvent` | `sitkEvent.h:54-60` | `Observers::on_multi_resolution_iteration` | 1 | Per level, before that level's optimization. **Does not also invoke `on_iteration`**, despite `sitkEvent.h:57-59` — see §1.6.2 for why upstream's own routing breaks that promise. Documented divergence-from-the-doc-comment, parity with the code. |
| `sitkProgressEvent` | `sitkEvent.h:48-50` | *(none)* phase 1; filter-only phase 2 | 2 | Never fires for `RM` upstream (§1.6.4). |
| `sitkAbortEvent` | `sitkEvent.h:36-38` | *(none)* | — | Fires only when a multi-threader's `ProgressReporter` throws `ProcessAborted` (`itkProcessObject.cxx:1686-1692`). Not reachable for `RM`; for filters, this port has no `AbortGenerateData` flag to set. |
| `sitkDeleteEvent` | `sitkEvent.h:39-41` | *(none)* | — | Means "the internal `itk::ProcessObject` was destroyed". This port has no long-lived ITK object; `execute` builds and drops its state within one call. A `Drop` hook on `ImageRegistrationMethod` would fire at a different moment and mean a different thing. **Cannot be matched.** |
| `sitkUserEvent` | `sitkEvent.h:61-63` | *(none)* | — | An escape hatch for ITK classes that invoke `UserEvent`; none of the registration or optimizer classes do. Nothing to deliver. |
| `sitkAnyEvent` | `sitkEvent.h:33-35` | *(none)* | — | Upstream's is not "all events" for `RM` — it is attached to the registration object and misses every optimizer iteration (§1.6.1). Porting the name would port the bug. A caller wanting all events registers all four callbacks. |
| `RM::GetOptimizerIteration()` | `sitkImageRegistrationMethod.h:697-698`, `.cxx:570-578` | `IterationContext::iteration` | 1 | Active → passed in. Upstream's post-`Execute` fallback is `RegistrationResult::iterations`, which already exists. |
| `RM::GetMetricValue()` | `.h:705-706`, `.cxx:614-621` | `IterationContext::metric_value` | 1 | Same. Post-hoc: `RegistrationResult::metric_value`. |
| `RM::GetOptimizerPosition()` | `.h:699-700`, `.cxx:581-590` | `IterationContext::position: &[f64]` | 1 | Borrowed, not cloned. Upstream returns `{}` outside `Execute`; there is no outside here. |
| `RM::GetOptimizerLearningRate()` | `.h:701-702`, `.cxx:592-601` | `IterationContext::learning_rate: Option<f64>` | 1 | `None` where upstream's slot is unbound and the getter silently returns `0.0` (LBFGSB `:277`, LBFGS2 `:307`, Amoeba `:361`, Exhaustive, Powell). Divergence: `Option` instead of a sentinel `0.0`. |
| `RM::GetOptimizerConvergenceValue()` | `.h:703-704`, `.cxx:603-611` | `IterationContext::convergence_value: Option<f64>` | 1 | Bound by the four GD-family branches and by `OnePlusOneEvolutionary` (`GetFrobeniusNorm()`, `CreateOptimizer.cxx:424-426`). Same `Option`-vs-`0.0` divergence. |
| `RM::GetOptimizerScales()` | `.h:722-731`, `.cxx:634-646` | `IterationContext::scales: &[f64]` | 1 | Upstream: manual scales returned verbatim, estimated scales only during execution. Here the estimator runs before the loop, so the estimated values are in scope for every callback. Strictly more available than upstream, never less. |
| `RM::GetCurrentLevel()` | `.h:720-721`, `.cxx:649-656` | `IterationContext::level` | 1 | Upstream returns `0` outside `Execute` and clamps a `SizeValueType` overflow to `0` (`sitkImageRegistrationMethod.cxx:52-64`). `usize`, no clamp needed. |
| `RM::GetMetricNumberOfValidPoints()` | `.h:707-717`, `.cxx:624-631` | `IterationContext::valid_points` | **1b** | Blocked on widening the objective's return type (§3.2). Post-hoc value already exists as `RegistrationResult::valid_points`. |
| `RM::GetOptimizerStopConditionDescription()` | `.h:733-736`, `.cxx:549-557` | `RegistrationResult::stop_reason` (`StopReason`) | done | Already ported as an enum rather than a `std::string`. Upstream calls it "Measurement updated at the end of execution", i.e. not active. |
| `RM::StopRegistration()` | `.h:738-748`, `.cxx:559-568` | `Control::Stop` returned from `on_iteration` | 1 | Inverted from a method-call into a return value: a callback that cannot name the method cannot poke it. Upstream returns `false` for LBFGSB/LBFGS2/Amoeba; here those optimizers ignore `Stop` (documented per optimizer) and Amoeba never calls the hook. `Exhaustive` maps to `StopWalking()` upstream (`CreateOptimizer.cxx:334`) — same `Control::Stop` here. |

---

## 6. What a phase-1 implementer must not do

- Do not add `RefCell`/`Cell`/`Rc` to bridge the objective closure and the
  observer. If a measurement is not reachable from a loop local, widen the value
  the evaluation returns (§3.2).
- Do not invoke `on_iteration` for the Amoeba optimizer. Upstream fires no
  `IterationEvent` there; a port that fires one changes observable callback
  counts for identical scripts.
- Do not synthesize a `ProgressEvent` for registration. Upstream has none, and
  `iteration/number_of_iterations` is not what `GetProgress()` means.
- Do not port `Abort()` on `ImageRegistrationMethod` as anything, including a
  no-op that returns `Ok`.
- Do not name a callback slot `on_any_event`.
