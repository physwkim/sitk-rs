# Three levers, one window: what each is predicted to be worth

Written before the first leg. P1–P3 are carried over verbatim from
`line-grain-prediction.md` (they were written before the line grain was merged and
are now graded); P4–P6 are new and are graded in the same window. Same model
throughout: `makespan(T) = (W/T) * ceil(T/P)`, `P = 96`.

## The levers, and why they must be separated

| variant | lever | binds where |
|---|---|---|
| **V0** | `main` with the **line grain reverted** (fixed `GRAIN` block run) | — the baseline |
| **V1** | `main` — the **line grain** (`grain(len)` block run) | 64³ only, 6 shapes |
| **V2** | `main` + **`MIN_GRAIN` 2048 → 1024** | 64³ only, every pass |
| **V3** | `main` + **`with_max_len(1)`** on the line pass's block path | every size, line ops |

V1 and V2 land on **the same cells at the same size** (64³ line ops), which is why
one window and a common baseline is the only comparison that can attribute a move.
V1−V0 isolates the line grain; V2−V1 isolates the floor; V3−V1 isolates the leaf
cap.

## Predictions

**P1 — nothing at 256³ or 512³ moves under V1.** The emitted block runs are the same
integers. *If a medium or large cell moves outside the noise band, the integer proof
is wrong and the line grain must be reverted, not tuned.* That sentence stands.

**P2 — `smoothing_recursive_gaussian`, `gradient_magnitude_recursive_gaussian`,
`signed_maurer_distance_map` at 64³ do not move under V1. 0%, within noise.** Their
one binding shape goes 64 → 128 tasks, which the model says is a wash at `P = 96`.

**P3 — `fft_convolution` at 64³ is the only cell that moves under V1.** Its padded
passes go 43 → 86 and 35 → 70 tasks, both inside one wave, so those passes should
roughly halve. Op-level central estimate **15%**, band >5% and <2×.

**P4 — V2 improves *every* 64³ op that goes through the map or reduce grain, and
regresses none.** `T` goes 128 → 256; makespan `W/64 → W/85`. Ceiling on the
parallel portion is **1.33×**; op-level, the share that is not the parallel pass
dilutes it. Central estimate **10–25%** on the map/reduce-bound small cells
(`otsu_threshold`, `mean`, `gradient_magnitude`, `rescale_intensity`), less on ops
whose 64³ time is dominated by allocation or a serial stage. **No 64³ op regresses**
— if one does, the floor is protecting against a scheduling overhead I have argued
is negligible at 1024 elements, and my re-derivation's premise (a) is wrong.

**P5 — V2 moves nothing at 256³ or 512³.** The floor cannot bind above 262 144
elements (see `min-grain-derivation.md`), so this is the same integer no-op. Same
consequence if violated: the proof is wrong, revert.

**P6 — V3 moves nothing, at any size. Low confidence** — this is the one I expect to
learn from rather than confirm. The leaf cap was measured to matter on the *map*
path (`mean`, 256³), where adaptive splitting left one worker holding a large
unsplit leaf. The line pass already raises 256–32 768 chunks at medium and large, so
I do not expect a starved split tree. If it *does* move something, it should be
`srg`/`gmrg`/`maurer` at 256³/512³ — which are exactly the cells the whole grain
family cannot reach, and it would be the one lever that touches them.

## What I expect the window to conclude

That the line grain (V1) is worth nothing on the cells it was aimed at and something
on `fft` small; that the floor (V2) is the lever that actually moves 64³; and that
the two together are the whole of what task-count arithmetic can buy on this box —
after which `srg`/`gmrg` at medium and large are a memory-bandwidth or split-tree
problem, not a grain problem.
