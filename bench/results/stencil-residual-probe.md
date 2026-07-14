# The participation probe: H1 is dead, and the residual is a *rate*

Graded against `stencil-residual-prediction.md`, which was committed (`c76e3ac`)
before the probe existed. Counters, not clocks — so these numbers are valid despite
the sibling panel's gate running beside them, and no `cargo bench` was run.

## Method

A temporary `probe_hit()` inside **`fill_indexed` itself** — the real function on the
real ops, not a lookalike — recording `rayon::current_thread_index()` once per chunk.
The real `mean`, `median` and `gradient_magnitude` at 64³, 5 reps each, at the two
grains (two builds: `MIN_GRAIN_INDEXED` 1024 and 2048). One untimed warm pass first,
so allocator and pool state are not what is being read. The probe was reverted; it is
not in the tree.

## What it found

| grain | chunks | `mean` workers | `median` workers | `gradient_magnitude` workers | max chunks on one worker |
|---|---|---|---|---|---|
| **2048** (V0) | 128 | 88–96 | 87–95 | 80–94 | **2–5** |
| **1024** (V4) | 256 | 90–96 | 87–95 | 87–95 | **5–8** |

96 workers in the pool. At `t64`, both grains put 58–64 workers to work; at `t1`, one
worker takes all 256 chunks, as it must.

## Grades

**P11 — FAILED.** I predicted the coarse grain would run on **20–40** distinct
workers, and wrote that "if the participation count comes back at 90+, H1 is dead."
It came back at **88–96**. The pool was already awake.

**P12 — FAILED.** Participation is the same at both grains (~90 of 96). It does not
scale with the 3.3× the times moved.

**P13 — FAILED.** The coarse histogram is *flat*, not skewed: 128 chunks over ~90
workers, **max 2–3 each** against a mean of 1.3. No worker is holding a tail.

**H1 is refuted.** Rayon wakes the box for both grains. The win is not a larger
machine.

## And this closes the packing family from the other side

R2 bounded any packing model at 2× *a priori*. The probe now says the same thing
*a posteriori*, from the observed schedules rather than from a bound: the coarse pass
puts at most **2–3** chunks on the busiest worker, and the fine pass at most **5–8**
chunks of half the cost. Those two schedules predict makespans within ~1.2× of each
other. The box says **3.3×**. The schedule is not what changed.

## What is left, and why it must be the memory system

The work is grain-independent, and that is code, not conjecture:

- `window_at` chooses the interior vs boundary path **per pixel**
  (`window_view(center, i, boundary)` — "on the interior path the boundary buffer is
  never even written"), not per chunk. A coarse chunk and a fine chunk do the identical
  work per element.
- Per-task state is `O(1)`: one `Cursor`, one `WindowScratch`, allocated once per task.
  A finer grain pays *more* of that, not less — the wrong direction to explain a win.

So: same work, same schedule, same number of awake workers, and a 3.3× difference in
wall time. The only term left in `T = W / (P · rate)` is **`rate`** — the workers are
going faster per element under the fine grain. That is a memory-system property
(locality, TLB, NUMA placement, or bandwidth under a different access footprint), and
it is the one thing on the list that a counter cannot settle and a clock can.

## The experiment this leaves, sharpened

**P16 is now the whole question, and it is cheap.** At `t1` there is no contention and
no schedule: if the two grains differ at `t1`, the grain is changing *single-thread*
throughput (locality — a chunk's footprint against L1/L2), and the parallel win is
inherited. If they agree at `t1`, the effect only exists when 96 workers stream
concurrently, and it is *contention* (shared L3 / memory controller / NUMA), which the
per-task footprint alone cannot explain and a thread sweep can.

`t1` at 64³ is a handful of milliseconds per leg. **This is the first thing to run
when the box lands, and it splits the remaining space in half on one leg.**

Second, and only if `t1` agrees: a thread sweep (`t8`, `t24`, `t48`, `t96`) at both
grains. Contention predicts the ratio *grows with the thread count* — it should be
~1.0 at `t8` and open up as the box fills. Locality predicts it is flat in `P`.

## What I got wrong, and what it cost

I predicted the wrong mechanism, in a document written to be graded, and the probe
killed it in one run. What the exclusions bought is that the space is now genuinely
small: packing is out by a bound *and* by observation, participation is out by
counters, work is out by code. Nothing in `T = W / (P · rate)` is left but `rate`.
The prediction failed; the round did not.
