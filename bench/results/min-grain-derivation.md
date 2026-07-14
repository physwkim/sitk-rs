# `MIN_GRAIN`: which derivation is wrong, decided before the box was touched

Two derivations in the tree disagree about the grain at 64³, and a measurement
that simply picked the faster number would be a fit wearing a derivation's
clothes. So this names the false premise first. The sweep that follows adjudicates
between **two derived candidates**, 2048 and 1024; it does not nominate a third.

## The two claims

**D1 — the floor (`MIN_GRAIN = 2048`).** From its own doc: *"feeding a 96-worker
pool from [64³] needs a grain of at most 262144/96 = 2730, and 2048 is the largest
power of two under that. It raises 128 tasks."*

**D2 — the target (`TARGET_TASKS = 256`).** `grain(len) = clamp(len/256, MIN_GRAIN,
ceiling)`. At 64³ that asks for `262144/256 = 1024`. The floor overrides it, and
the pass gets 128 tasks where the rule asked for 256.

## D1 is the false one, and it is false twice

**Fault 1 — its objective does not control the makespan.** For `T` equal tasks on
`P` workers doing total work `W`:

    makespan(T) = (W / T) * ceil(T / P)

D1 optimises "every worker gets at least one task" (`T >= P`). That is not
sufficient, and the arithmetic is brutal at exactly the value D1 chose. On this
box (`P = 96`):

| T | waves | makespan | utilisation |
|---|---|---|---|
| 64 | 1 | W/64 | 0.67 |
| **128** (D1) | **2** | **W/64** | **0.67** |
| **256** (D2) | 3 | W/85 | 0.89 |

**`T = 128` has exactly the makespan of `T = 64`.** The second wave is 32 tasks
wide on a 96-wide pool, so two-thirds of the box idles through it. D1 raised the
task count from 64 to 128 and bought *nothing*, while believing it had closed the
defect. Its premise — "one task per worker is enough" — is the wrong objective;
the quantity that matters is `ceil(T/P)`, and it only improves when `T` grows past
the *next multiple* of `P`, or grows large enough that the quantisation washes out
(`U >= T/(T + P)`).

**Fault 2 — it is keyed on this box.** D1 divides by **96**: the worker count of
the machine the port was benchmarked on. `TARGET_TASKS`'s own doc forbids exactly
this — *"an upper bound on the worker count of any box this runs on — never a
reading of the running pool"* — and while D1's reading happens at authoring time
rather than run time (so it does not break the determinism contract), it makes the
constant a property of **this machine** rather than of the port. A 128-worker box
would want a different 2048, which is the tell of a fitted constant: it re-opens
the moment the hardware changes.

## Re-derivation, from what a floor is actually for

The two constants must own different things, and D1's error was letting the floor
own task count:

- **`TARGET_TASKS` owns the task count.** It is the design's stated upper bound on
  pool width (256), and `T >= 256` is what bounds the worst-case utilisation over
  *every* pool this could run on: `U >= T/(T + P) >= 0.5` for any `P <= 256`. This
  is the only one of the two that may mention pools.
- **`MIN_GRAIN` owns the per-task work.** Its job is to stop a short input raising
  tasks so small that rayon's per-task overhead (order 100 ns – 1 µs) is a large
  fraction of the work in one. That is a property of *work*, not of *pools*, and it
  must not be derived from a worker count at all.

So the floor is the largest power of two that (a) keeps per-task work well above
scheduling overhead and (b) **does not fight the target at the smallest volume the
port is designed for** — `doc/bench-spec.md`'s `small`, 64³:

    MIN_GRAIN = SMALL / TARGET_TASKS = 262144 / 256 = 1024

At 1024 elements a task carries ~1–10 µs of streaming work against ~sub-µs of
scheduling overhead, so (a) holds; the merged doc's own `otsu_threshold` 64³ sweep
is consistent — the curve is still *descending* at 1024 (2048 → 1.57 ms, 1024 →
1.38), so overhead has not begun to dominate there. And (b) holds by construction:
the floor now binds only **below** 64³, where a short input genuinely does not
contain 256 tasks' worth of parallelism, which is the case the floor exists for.

The floor stops being a function of the box. That is the structural point, and it
is why this is a re-derivation rather than a re-fit.

## What the sweep decides

It adjudicates D1 (2048) against D2 (1024) — both derived, one of them from a
premise I have just argued is false. If 1024 does **not** beat 2048 at 64³, then my
makespan model is wrong and D1's number survives on evidence rather than on its
derivation, and I will say so in those words.

## Where `MIN_GRAIN` can bind at all (integers, not opinion)

`grain(len, c) = clamp(len.div_ceil(256), MIN_GRAIN, c)`. The floor binds only when
`len.div_ceil(256) < MIN_GRAIN`, i.e. `len < 256 * MIN_GRAIN`:

| MIN_GRAIN | binds below | 64³ = 262 144 | 256³ = 16.7 M | 512³ = 134 M |
|---|---|---|---|---|
| 2048 (D1) | 524 288 | **binds** (grain 2048, T=128) | no | no |
| 1024 (D2) | 262 144 | **does not bind** (grain 1024, T=256 — the target, met exactly) | no | no |

So changing it is a **no-op at and above `medium` by the same integer argument as
the grain seam and the line grain**: at 256³ and 512³, `len/256` already exceeds
both candidates, the clamp never reaches the floor, and the emitted boundaries are
the same integers either way. Only 64³ (and `fft`'s padded sub-64³ buffers) can
move.
