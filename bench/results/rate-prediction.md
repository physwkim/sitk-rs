# Locality or contention: the prediction, before the clock

Written before any leg of this round. The residual is now known to live in `rate`
(`stencil-residual-probe.md`: work is grain-independent by code, the schedule is flat
by counters, the pool is awake at both grains, and packing is capped at 2× by R2). The
two survivors differ in *where* the rate is lost, and one leg separates them.

## The discriminator

At `t1` there is **no contention and no schedule**, and — this is the point — the
*global access order is identical at both grains*: one worker walks chunk 0, then
chunk 1, and within each chunk `i` ascends, so the sequence of addresses touched is
the same sequence the serial loop touches, whatever the chunk size. The only thing a
finer grain adds at `t1` is **more per-chunk setup** (one `Cursor::seek` and one
`WindowScratch` per chunk instead of per two).

- **Grains differ at `t1`** → the grain changes *single-thread* throughput. Locality.
  The parallel win is then inherited, not caused, and the fix would be about chunk
  footprint against the private caches.
- **Grains agree at `t1`** → the effect exists only when 96 workers stream
  concurrently. Contention. The fix would be about what they share.

## Predictions

**P18 — the grains AGREE at `t1`, within the small-cell noise floor, and if anything
the 1024 grain is *marginally slower*.** It pays 256 chunk setups where 2048 pays 128,
and it can buy nothing back: a single worker's address stream is identical either way.
Central estimate `t1(1024)/t1(2048)` = **1.00–1.03**. **Falsifier: if 1024 is faster at
`t1` by more than the noise floor, I am wrong and the mechanism is locality** — the
parallel win is inherited and the whole "contention" line is dead.

**P19 — `W(64³)` comes in BELOW the 30.7 ms I extrapolated, and the extrapolation was
begging the question.** The 117 ns/voxel came from the 256³ `t1` leg, where the input
is 64 MB and streams from memory; at 64³ the input is 1 MB and is cache-resident, so
per-voxel cost must *fall*. Central estimate for `mean` at 64³ `t1`: **15–25 ms**
(60–95 ns/voxel). This is the number the port has never had, and it is why the leg is
run rather than extrapolated.

**P20 — the ratio grows with `P`.** Contention's signature: at `t8` the 96-way sharing
does not exist, so the grains should be within noise there; the gap should open as the
box fills, reaching ~3× only near `t96`. **Falsifier: a ratio flat in `P`** — which,
combined with P18's null at `t1`, would mean neither survivor is right and I am back to
H3.

**P21 — at `t64` the fine grain still wins, by ≥1.5×.** This is the cell the packing
family cannot have: on 64 workers, 128 tasks is a *perfect two-wave fit* with zero idle
quantisation, so every packing model predicts **no win at all** there. It is the
held-out cell named last round, and no leg in this port has ever run `t64`.

**P22 — under the coarse grain, 96 workers barely beat one.** If `W(64³) ≈ 20 ms` and
the coarse `tN` cell is 14.0 ms, the speedup of the whole box over serial is **~1.4×**.
That is the real shape of the defect, and it says the 64³ stencil pass was not "somewhat
under-parallel" — it was *nearly serial*. **Falsifier: coarse-grain speedup above 5×**,
which would mean my `W` estimate is badly wrong and P19 with it.

## If it is contention, "the memory system" is not an answer

Bandwidth saturation and cache-line sharing are different mechanisms with different
fixes, and I will not stop at the family name:

- **Bandwidth/L3 capacity.** Then the gap must track the *aggregate concurrent
  footprint*. At 2048, 96 workers cover 196 608 contiguous elements at any instant; at
  1024 they cover 98 304 — half the span, so their stencil halos overlap more and the
  live input set is smaller. Signature: the gap tracks `P × grain`, so it should also
  appear at `t96` with a *larger* grain, and shrink at `t48` where the span halves.
- **Line sharing / RFO traffic at chunk edges.** Then it must depend on chunk
  *alignment*, not span — and a 1024-element f32 chunk is 4 KB while 2048 is 8 KB, both
  64-byte aligned, so this predicts **no effect**, which is why I do not expect it.

The thread sweep at both grains distinguishes these, because they disagree about what
happens between `t48` and `t96`. If neither shape appears, I will say the mechanism is
unidentified rather than name one that fits.
