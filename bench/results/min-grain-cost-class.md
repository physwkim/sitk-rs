# The floor is not one number: `MIN_GRAIN` per cost class

Written after Campaign S and **before** V4 was built or run. The S window
adjudicated `MIN_GRAIN` 2048 (D1) against 1024 (D2) and returned a **split
verdict**, and the split is the finding: neither constant is right, because a
single element-count floor cannot express a *work* floor across two cost classes.

## What S measured (5 clean rounds, 64³, tN, `foreign=none`)

`v2/v0` is `MIN_GRAIN` 1024 against 2048, paired within each round:

| op | primitive | `v2/v0` |
|---|---|---|
| `mean` | `par_map_window` → `fill_indexed` | **0.304** |
| `binary_dilate` | `par_map_window` → `fill_indexed` | **0.334** |
| `median` | `par_map_window` → `fill_indexed` | **0.476** |
| `gradient_magnitude` | `par_map_window` → `fill_indexed` | **0.641** |
| `otsu_threshold` | `bin_counts` / `min_max` (reduce) | 1.145 |
| `rescale_intensity` | `min_max` + flat map | 1.088 |
| `gradient_magnitude_recursive_gaussian` | `for_each_line_mut` | 1.056 |
| `signed_maurer_distance_map` | `for_each_line_mut` | 1.046 |
| `smoothing_recursive_gaussian` | `for_each_line_mut` | 1.046 |

The line does not fall between fast ops and slow ops, or between big buffers and
small ones. **It falls exactly on `fill_indexed`.** Every op that wins goes
through it; no op that regresses does.

## Why an element count cannot be the floor

`MIN_GRAIN` exists to keep the work in one task well above the cost of
dispatching one (order 1 µs for a rayon job). That is a floor on **work**. It is
written as a floor on **elements**, and the conversion factor between them is the
op's cost per element — which differs by about two orders of magnitude between
the two classes this module already distinguishes:

- **Indexed/stencil** (`fill_indexed`): a 3³–5³ window is 27–125 fused multiply-adds
  per element, ~50–100 ns each. A 1024-element task is **50–100 µs** of work —
  overhead is under 2% and the floor is not what protects it. What binds here is
  `TARGET_TASKS`: the floor's only job is to *not fight the target* at the
  smallest volume the port supports, which is `SMALL / TARGET_TASKS = 262144 / 256
  = 1024`.
- **Elementwise/reduce** (`fill_zip`, `for_each_mut`, `min_max`, `bin_counts`): one
  or two ops per element, well under a nanosecond each. A 1024-element task is
  **~1 µs** of work — the *same order as the dispatch it pays for*. Here the floor
  is the only thing standing between the pass and its own scheduling overhead, and
  1024 is below it.

So the premise I wrote in `min-grain-derivation.md` — *"at 1024 elements a task
carries ~1–10 µs of streaming work against ~sub-µs of scheduling overhead, so (a)
holds"* — is **true for the stencil class and false for the elementwise class**. I
wrote the falsification condition into P4 in as many words (*"if one regresses,
premise (a) is wrong"*), and it regressed, and it is.

This is the same split the module has already accepted once. `fill_indexed`'s doc
says it, about the leaf cap:

> The two fills serve two cost classes — indexed/stencil work that is expensive
> per element, and elementwise work that is nearly free per element — and the
> split policy follows from which one you are in, not from a tuned number.

`with_max_len(1)` is already per-class. The grain floor is not, and that is the
defect: one constant sitting between two classes, wrong for both — too low to
protect the cheap class, too high to feed the expensive one.

## V4 — the floor follows the class, like the leaf cap already does

    MIN_GRAIN_INDEXED = SMALL / TARGET_TASKS = 1024   // fill_indexed only
    MIN_GRAIN         = 2048                          // unchanged, everywhere else

Nothing else changes. This is not "1024 where it helped": it is the floor being
derived per class from the same overhead argument, evaluated with each class's own
cost per element — and the class boundary is one that already exists in the code,
not one drawn around the winners.

## Predictions for V4, before it runs

**P7 — V4 reproduces V2's win on exactly the four `fill_indexed` ops, and reproduces
it *closely*.** For those ops V4 emits the identical grain (1024) to V2, so the
decomposition is the same integers: `mean ≈ 0.30`, `binary_dilate ≈ 0.33`,
`median ≈ 0.48`, `gradient_magnitude ≈ 0.64`, each within noise of its V2 ratio.

**P8 — V4 is an *identity* on every op that does not reach `fill_indexed`, not merely
"no measurable change".** `MIN_GRAIN` is untouched on those paths, so the emitted
chunk boundaries are the same integers as V0. `otsu_threshold`, `rescale_intensity`,
`gmrg`, `srg`, `maurer` must come back at **1.00 ± noise**. If any of them moves
outside the noise band, then V4 is reaching a path I have not accounted for, and the
claim that the split falls on `fill_indexed` is wrong.

**P9 — V4 regresses nothing at 64³.** V2's three regressions (`otsu` 1.145,
`rescale` 1.088, `gmrg` 1.056) are all on non-`fill_indexed` paths and must vanish
by P8.

**P10 — V4 changes nothing at 256³ or 512³, for any op.** `grain(16.7M, 4096)`
clamps to the 4096 ceiling before any floor is consulted, so *no* floor — 1024,
2048, or per-class — can bind at or above `medium`. Already confirmed empirically
for V2 in Campaigns M and L; V4 inherits it by the same arithmetic.

## What this does not settle

The elementwise/reduce class may want a floor **larger** than 2048, not merely
"not 1024": at ~1 ns/element, holding scheduling overhead under 5% wants ~20 µs of
work per task, which is ~16 K elements, not 2 K. `otsu_threshold` at 64³ raises 128
reduce tasks of ~2 µs each and may be paying for most of them. That is a *derived
hypothesis and nothing more* — it is not measured here, I am not fitting a constant
to it, and I am not proposing it. It is the next question, stated so it does not get
smuggled into this answer.
