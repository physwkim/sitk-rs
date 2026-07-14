# The cost-class split: predicted before the clock

The straggler is the defect (`occupancy-result.md`): `c_max/wall` = 0.93–1.00, and the
straggling chunk is the `z = 0` plane, where every voxel takes the boundary window path at
~1.0 µs against ~130 ns interior. `MIN_GRAIN_INDEXED` helps only because halving `g` halves
that chunk. This is the fix that removes the family, and these are the numbers it must hit.

## The design, and why it carries no fitted constant

A flat chunker cuts `0..len` into equal-**count** pieces, but the cost per index is not
uniform: a row whose window can overhang in any dimension above 0 takes the checked path
for *every* pixel; any other row takes the interior path except `radius[0]` pixels at each
end. That is **two cost classes, uniform within each**, and it is a fact about the walk,
not about a benchmark — so the decomposition can be built from `size` and `radius` alone.

The rule: **each cost class is split into its own `TARGET_TASKS` chunks.** Then

    c_max = max(W_checked, W_mixed) / TARGET_TASKS  ≤  W / TARGET_TASKS

and the 8× ratio between the classes **never appears in the code**. There is no cost
constant to derive, and therefore none to fit — the classes are counted, not weighted.
Chunks stay contiguous index ranges, so element `i` is still computed by exactly one task
from `i` alone.

## Predictions

**P28 — the straggler dies.** At 64³/`t96`, `c_max/wall` falls from **0.93–1.00** to
**≤ 0.25**, and occupancy rises from 29–50 to **≥ 70 of 96** for `mean` and `median`.
`gradient_magnitude` will not reach 70: its wall is heading for its own ~0.9 ms ramp floor,
which this change does not touch. **Falsifier: `c_max/wall` stays above 0.5** — then the
chunks are still not cost-homogeneous and I have mis-modelled the classes.

**P29 — `MIN_GRAIN_INDEXED` stops mattering, and I will delete it.** With the split in, the
stencil family no longer reaches the flat floor at all, so I revert `fill_indexed` to the
old global `MIN_GRAIN = 2048` and **remove the constant from the tree**. The 64³/`tN` walls
must be **at least as good as today's fine-grain walls** with the floor gone. **Falsifier:
reverting the floor loses the win** — then the split is not the mechanism, it is a second
one, and I say so rather than keep the constant to hide it.

**P30 — bit-neutral by construction.** The split changes *which task* computes a voxel,
never *how*: `window_at` picks interior/checked per pixel, and the partition is a pure
re-grouping of those same per-pixel decisions. Every checksum unchanged on every op at
every size; `bit_parity` 18/18 at 1/4/48/96. **Falsifier: any checksum moves — then it is
not a re-grouping and I stop.** This is not a hope; the guard is that no `f(i)` and no `i`
changes, only chunk boundaries, which is the same argument that lets `grain` be tuned.

**P31 — 256³ and 512³ do not move** (within ±5%). The shell is 4.6% of a 256³ volume and
2.3% of a 512³ one, and at those sizes `c_max` was never the makespan — `W/P` was. There is
almost nothing to win, and **a large move at these sizes would mean I have changed the work,
not the schedule**, which P30 forbids. This is the pin that catches a fix that "helps" by
breaking something.

**P32 — the residual, settled by subtraction.** The probe recovered only 1.30–1.36× of the
bench's 3.16×, and I refused to claim the straggler explained all of it. **If the mechanism
is complete, the split recovers the full win against the V0 baseline**: `mean` 64³/`tN`
from 14.1 ms to **≤ 4.5 ms** with the floor *removed*. If it lands short — say 6–8 ms —
then the difference between that and 4.47 ms is a **second mechanism**, isolated by
subtraction, and it becomes the next question. Either outcome is a result; only a claim
without the number is not.

## What would make me stop

If P30's falsifier fires — any moved checksum — I stop and revert, because a decomposition
that changes a bit is not a decomposition change. Nothing below that is worth a wrong bit.
