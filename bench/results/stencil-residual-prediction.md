# The 2.5× residual: what it cannot be, and what I predict it is

Written before the probe was built or run, and before any clock this round. The
question is the one the last round left open: the equal-cost makespan model allows
**1.33×** when `mean` at 64³ goes from 128 tasks to 256, and the box returned
**3.3×**. A model that only *explains* the cells it was built on is a restatement of
the data, so this states the exclusions as bounds, the surviving mechanism as a
prediction, and names the cells the prediction is tested on — which are cells the
model is not allowed to see first.

## Two exclusions, proved rather than measured

**R2 — no packing model can reach 3.3×, for any heterogeneity, on any pool.**
For total work `W` cut into `n` tasks on `P` workers, every list schedule obeys

    max(W/P, c_max)  ≤  makespan  ≤  W/P + c_max,     c_max(n) = h · W/n

where `h` is the slowest task over the average one. Doubling `n` halves `c_max`, so

    ratio(n → 2n)  ≤  (W/P + c) / max(W/P, c/2)  ≤  2      for every h, every P.

Both corners are inside it: equal-cost wave quantisation (`h = 1`) gives 1.33×, and
a pure straggler (`h → ∞`) approaches 2×. **Heterogeneity cannot rescue the model** —
it inflates the numerator and denominator together. 3.3× is outside the family, so
**`W` is not constant across the grain, or the workers are not independent.** This
is what kills the model I used last round; it does not need a box to see.

**R3 — no grain-independent additive overhead can reach it either.** With
`T = ceil(n/P)·(W/n) + w` and `w` fixed, `w` *compresses* the ratio toward 1. From
the published `t1` leg at 256³ (`mean` 1966.6 ms → 117 ns/voxel), `W(64³) ≈ 30.7 ms`,
so the compute term at `P = 96` is at most ~0.5 ms while the cell measures 14.0 ms.
Fitting `w` to V0 gives `w ≈ 13.5 ms`, which then predicts **13.9 ms for V4**. The
box says 4.3 ms. Pool wake-up, allocation, page-faulting the output — anything that
costs the same whatever the grain — is excluded by this line.

## What survives

Whatever it is, it must make **the effective parallelism a function of the grain**.
The candidate I am naming, because it is a real behaviour of the scheduler and not a
property I have invented for the occasion:

**H1 — worker participation.** Rayon's workers sleep, and are woken in proportion to
the jobs the split tree makes *stealable*. A coarse grain exposes fewer leaves, so
fewer workers are tickled awake before the pass is over — and a pass this short
(single-digit ms) can finish with most of the pool still asleep. The pool is then not
96 wide; it is however wide it woke. Halving the grain doubles the leaves, wakes more
of the box, and the win is not a scheduling improvement at all — it is a *larger
machine*.

H1 is outside R2/R3 precisely because `P_effective`, not `W` or `w`, is what moves.

**H2 — grain-dependent throughput** (locality/bandwidth: the aggregate live working
set `P × (chunk + halo)` crossing a cache level). Also outside R2/R3, because it
makes `W` itself a function of the grain.

**H3 — something I have not thought of.** Named so the probe's null is reportable.

## The probe, and why it needs no clock

H1 is a statement about **which threads run chunks**, not about how long they take.
So it is measured with counters: record `rayon::current_thread_index()` per chunk
inside `fill_indexed` — the real function, on the real `mean` at 64³ — and count the
distinct workers that participated and how many chunks each ran. A contended box
perturbs a clock; it does not turn a woken worker into a sleeping one, so this
survives the sibling panel's gate running beside it.

H2 makes no prediction about participation: it says the box is fully awake and each
worker is simply slower.

## Predictions, before the probe runs

**P11 — under the 2048 grain (128 chunks), `mean` at 64³ runs on *far fewer than 96*
distinct workers.** Central estimate **20–40**; the band that confirms H1 is anything
under 64. If the participation count comes back at 90+, **H1 is dead** and I will say
so — the pool was awake and the residual is H2 or H3.

**P12 — under the 1024 grain (256 chunks), participation is substantially higher than
under 2048**, and the ratio of participations is the same order as the ratio of the
times (3.3×). If both grains wake the same number of workers, H1 is dead by the same
line.

**P13 — the chunks-per-worker histogram under 2048 is heavily skewed**, with a few
workers running many chunks each, rather than 128 chunks spread ~1.3 each over 96
workers. A flat histogram over a small worker set would mean the pool was capped, not
starved, and points at the pool, not the split tree.

## The held-out test — the part that makes this a model and not a restatement

If P11–P13 confirm H1, the model is `T = ceil(n/P_eff(n)) · (W/n + d)`, with
`P_eff(n)` read off the probe. It is then **fitted on `mean` alone**, and must predict
cells it has never seen. Committed now, graded when the box lands:

**P14 — `P_eff` is a property of the split tree and the pass duration, not of the op.**
So `median` (a *longer* pass at 64³, 15.1 ms) must show **higher** participation than
`gradient_magnitude` (2.6 ms, the shortest) at the same grain. Concretely: I predict
`gradient_magnitude` has the *lowest* participation of the four stencil ops at the
2048 grain, and `median` the highest.

**P15 — the win must vanish at `P = 64`, if and only if H1 is false.** At 64 workers,
128 tasks is a *perfect two-wave fit* — a packing model predicts the 1024 grain buys
nothing there. H1 predicts the opposite: the 1024 grain still wins at `t64`, because
the mechanism is how many workers woke, not how the tasks packed. **This is the cell
that separates the two families, and it is a thread count no leg in this port has
ever run.**

**P16 — at `t1` the two grains are within noise of each other.** One worker cannot
have a participation problem, and the chunk decomposition does not change the work.
If `t1` shows a 3.3× gap, then H1 *and* H2-as-concurrency are both wrong: the grain
would be changing single-thread throughput, and the parallel win is inherited rather
than caused. `t1` legs are cheap and this is the first thing to run.

## What I am not doing

I am not sweeping the grain to find a better number. The grain is fixed by the
cost-class derivation already merged; this round is about *why* it paid, and the only
constant that could come out of it is `d` (rayon's per-task dispatch cost), which is
a property of the scheduler and is measured, not chosen.
