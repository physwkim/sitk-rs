# Where is the idleness? Predicted before the timeline probe

`wall = busy / occupancy`, and occupancy is 20–38 of 96 on a warm pool with 256
independent leaves. "The workers are idle" is not yet "the workers are idle **here**".
This names where, and what would refute it, before the clock.

## The candidates, and what each must survive

A mechanism does not get to explain the idleness unless it also survives **the win it is
being used to explain**: 256 chunks beat 128 by 3.16× at `t96`.

**H4 — arrival stagger (the pool ramps).** Rayon's workers sleep between passes; a pass
this short (a few ms) may spend most of its life waking the box. Idleness would then be
*at the front*: a worker contributes nothing until it wakes, and the pass ends before the
pool is full. Survives the win: with a finer grain the tail chunk is half as long, so the
wall ends sooner relative to the ramp, and a *larger fraction of the pass* runs at high
occupancy.

**H5 — split-tree starvation.** `par_chunks_mut(g).with_max_len(1)` splits binary, and a
worker exposes a job only by splitting on its own path; if exposure is slower than
demand, woken workers find nothing to steal and idle **mid-pass**. Survives the win
trivially (more leaves = more exposure).

**H6 — the tail.** Workers finish, find nothing left, and the wall is set by the last
long chunk. Idleness is **at the back**. This is a packing effect and is therefore
capped at 2× by R2 — so it **cannot be the whole story**, but it can be part of it.

**H7 — steal contention on a hot deque.** *Already in trouble before the probe runs:*
more tasks means more steals, so it predicts 256 chunks are **worse** than 128. The box
says 3.16× better. **H7 is refuted by the win itself**, and I predict the timeline shows
no gap growth from 128 to 256 chunks. If it does, both this and the grain result need
re-reading.

## Predictions

**P23 — arrival is staggered over milliseconds, and that is the bulk of the idleness.**
The distribution of each worker's *first* chunk start should be spread across a large
fraction of the wall: I predict the **90th-percentile arrival lands past 40% of the
wall**, and that the concurrency profile *rises through the pass* rather than plateauing
early. **Falsifier: if every worker's first chunk starts within the first 10% of the
wall, H4 is dead** and the idleness is mid-pass (H5) or at the back (H6).

**P24 — a worker that has arrived stays busy.** Inter-chunk gaps for an *already-active*
worker should be small (median gap < 10% of a chunk). **Falsifier: large mid-pass gaps
for active workers → H5, split-tree starvation, and the fix would be the split policy,
not the grain.**

**P25 — the tail is one chunk long, and it is not the mechanism.** The interval from
"90% of chunks complete" to "last chunk complete" should be ≈ one chunk duration —
roughly 2× longer in ms at the coarse grain, because its chunk is 2× longer. Real, but
bounded by R2 and far too small to be the 3.16×.

**P26 — occupancy is bounded by arrivals, not by work.** At any instant before the tail,
the number of *unstarted* chunks should be large (work was available; workers were not).
**Falsifier: chunks run out mid-pass → the decomposition is too coarse in a packing
sense, which R2 says cannot pay 3.16×.**

## What I expect to conclude

That the 64³ stencil pass is **too short to fill this box**: it spends its life waking
workers, and the grain's real service is that a finer chunk lets the pass end *later
relative to the ramp* and with a shorter tail — a bigger machine for the same work. If
that is right, the *structural* fix is not a smaller grain at all: it is not paying the
ramp per pass. I will not propose one until the probe says which.
