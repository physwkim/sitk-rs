# The elementwise floor cannot be raised: the constraint is infeasible at 64³

Last round I derived a hypothesis and refused to act on it: *"the elementwise/reduce
floor may want to be **larger** than 2048 — at ~1 ns/element, holding dispatch under
5% wants ~20 µs of work per task, which is ~16 K elements, not 2 K."* Acting on it
now, as asked, and **before any clock**: it is refutable on integers, and it is
refuted. The floor stays at 2048. No box time is owed on this.

## The two constraints, written down together

A floor `F` fixes both the work per task and the number of tasks, because the volume
is fixed. At 64³ (`len = 262 144`):

| constraint | what it demands | in elements |
|---|---|---|
| **dispatch is amortised** — per-task work ≫ rayon's per-task cost `d` | `F · e ≥ 20 · d` | `F ≥ ~16 384` at `e ≈ 1 ns` |
| **the pool is fed** — a task count that can occupy the workers | `len / F ≥ P` | `F ≤ 2 730` at `P = 96` |

`F ≥ 16384` and `F ≤ 2730` have **no solution**. The two constraints are jointly
satisfiable only when

    len  ≥  P · 20 · d / e  ≈  96 · 16 384  =  1 572 864 elements

which is **6× larger than the smallest volume the port supports**. Below ~1.5 M
elements there is no floor that both amortises dispatch and fills the box: the volume
simply does not contain that much parallelism. So below that length the floor is not
choosing an optimum, it is **choosing which constraint to break**, and my hypothesis
amounted to breaking the one that costs more.

## Which one costs more, as a bound and not an opinion

Raising the floor to 16 384 gives `262144 / 16384 = 16` tasks. With `n < P` the
makespan is bounded below by the cost of a single task, whatever the schedule:

    makespan  ≥  c_max  =  W / 16

against `W / 96` for a full box — a **≥ 6× floor on the loss**, and no scheduling
cleverness can go under it, because 80 of the 96 workers have nothing to hold. The
dispatch overhead it was buying back is, by its own premise, 5% of the work. **A 5%
saving cannot pay a 6× penalty.** The hypothesis is dead, and it did not need a leg.

(The 64³ grain sweep recorded in `MIN_GRAIN`'s doc — 16 384 → 4.22 ms against 2048 →
1.57 ms, a 2.7× regression — is *consistent* with this direction, but I am not
resting the argument on it: I superseded that sweep last round as a single-shot on an
unpinned process shape, and it would be dishonest to promote it back to evidence now
that it agrees with me. The bound above stands on its own.)

## What this says about the floor that *is* there

2048 raises 128 tasks at 64³ — inside the feasible band (`F ≤ 2730`), and the largest
power of two there. That is D1's arithmetic, and **D1's number survives for the cheap
class**, on a constraint D1 never stated. What did not survive is D1's *reasoning*
(one task per worker is the objective) and D1's *scope* (one floor for the whole
module): the makespan argument that demolished the first and the cost-class argument
that demolished the second both still hold. A right number can sit on a wrong
derivation, and this one did.

So the cheap class's floor is not a free parameter waiting for a sweep. It is pinned
between `len/P` above and dispatch below, and at the volumes this port benchmarks the
upper constraint binds. The only thing that could move it is a much larger `d` than
the µs-order one assumed here — and `d` is measured by the probe in
`stencil-residual-prediction.md`, not chosen. If that probe returns a `d` above ~20 µs,
this page is wrong and I will reopen it.

## Prediction

**P17 — the elementwise floor is at its constrained optimum and no sweep will beat it
by more than the noise floor.** Specifically: at 64³, a 4096 floor (64 tasks, half the
pool idle by the bound above) and a 16 384 floor (16 tasks) both **regress**
`otsu_threshold` and `rescale_intensity`, and 1024 regresses them too — as already
measured, 1.14× in two independent windows. 2048 sits at the top of the feasible band
and is a local optimum bracketed on both sides. If any of those three beats 2048
outside noise, the feasibility argument above is wrong somewhere and I want to know
where.
