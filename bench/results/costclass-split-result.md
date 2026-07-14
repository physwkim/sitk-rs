# The cost-class split: the straggler is dead, and the constant is out of the tree

Graded against `costclass-split-prediction.md`, committed (`ef42c62`) before the code
existed. Box exclusively mine; every kept round `foreign=none`, atomic rounds, load trace
in `costclass/legs.log`.

## What shipped

`NeighborhoodIterator::cost_runs` names the two cost classes the walk actually has — a
**checked row** (every pixel materializes the window) and a **mixed row** (interior except
a fixed `2*radius[0]` pixels) — and `parallel::map_indexed_init_by_cost` gives **each class
its own `TARGET_TASKS` split**. That bounds `c_max` at `max(W_class)/TARGET_TASKS` without
the code ever comparing the two costs, so there is no constant to fit. `MIN_GRAIN_INDEXED`
is deleted.

## The numbers, 64³/`tN`, paired within round

| op | pre-R3 baseline | merged main | **split** | vs main | vs baseline |
|---|---|---|---|---|---|
| `mean` | 14.34 | 5.20 | **2.61** | **0.50×** | **0.18×** |
| `binary_dilate` | 26.86 | 16.52 | **4.28** | **0.26×** | **0.16×** |
| `median` | 16.88 | 7.28 | **5.35** | **0.74×** | **0.32×** |
| `gradient_magnitude` | 2.46 | 1.60 | **1.07** | **0.67×** | **0.43×** |

Reproduced across four independent campaigns (SPLIT-S, SPLIT-S3, P29, FINAL), the last on
the fully gated source. `t1` legs are flat (0.99–1.00) except `mean` at **1.08** — the
split pays ~2.7× more per-task setup, and with no parallelism to hide it that shows.

256³: **no regressions** — `gradient_magnitude` 0.68×, `mean` 0.89×, `binary_dilate` 0.94×,
`median` 0.96×.

## Grades

**P28 — HELD in mechanism, FAILED on my own thresholds.** `c_max/wall` falls from
**0.93–1.00** to **0.15** (`gm`), **0.27** (`mean`), **0.44** (`median`) — the makespan is
no longer one chunk, which was the whole point. But I predicted **≤0.25** and occupancy
**≥70** for `mean` and `median`: occupancy is **56.8** (`mean`, from 29.5) and **71.2**
(`median`, from 50.0), so `mean` misses the bar I set. Chunks are still heterogeneous by
3.1–4.0× (`c_max/c_mean`), which the class model says they should not be, and I do not yet
know why. The prediction failed at the number even where it held at the mechanism.

**P29 — HELD, and the constant is gone.** With the split in, re-adding the 1024 floor
changes nothing for the stencil family (0.98–1.03 in every campaign). A 4-way control
(both floors × both chunkers, one op per process) on the ops that *still* use the flat
fill puts them within 1% — `signed_maurer_distance_map` 2.87/2.89/2.88/2.90 ms — and
`otsu_threshold` in fact **prefers** the 2048 floor it now gets back (0.87 vs 0.97), which
is the same regression that split the floors in Round 3. The floor was a workaround; it is
out.

**P30 — HELD.** `bit_parity` **18/18** at 1/4/48/96, every checksum unchanged,
**3417/3417** workspace. The split changes which task computes a voxel, never how.

**P31 — FAILED, and it caught two real bugs before the user did.** I predicted 256³ moves
by less than ±5%. It moved by **up to 32%** (`gm` 0.68×) — in my favour, but the prediction
was wrong. More importantly the *first* version of this fix **regressed `binary_dilate`
1.41× at 256³**, and the second regressed it **3.5×**, and P31 is what caught both:
1. I dropped the `GRAIN` **ceiling** along with the floor. A class total of 16.7 M asks for
   a 62 500-element chunk, 15× what the ceiling allows. Cost-homogeneous chunks that large
   are worse than heterogeneous small ones.
2. I claimed `with_max_len(1)` was unnecessary "because the leaves are built here". That
   was **wrong**: rayon groups the task *list* into jobs adaptively, so one worker holds a
   long run of tasks and executes it alone — the very straggler this change removes,
   reintroduced one level up. Without the cap: `binary_dilate` **3.5×**, `mean` **2.2×**
   slower than the flat chunker.

**P32 — HELD, and the residual is closed.** I predicted the split would recover the full
bench win (`mean` 64³ from 14.1 ms to ≤4.5 ms) if the straggler was the whole mechanism.
It lands at **2.61 ms** — *better* than the grain's best and 5.5× the baseline. The gap I
refused to explain last round (the probe reproduced only 1.30–1.36× of the bench's 3.16×)
is not a second mechanism: removing the straggler outright pays **more** than halving it
did, which is what a single mechanism looks like when you stop treating it and start
deleting it.

## What I nearly reported, and did not

A multi-op campaign showed the deleted floor costing `otsu`, `fft_convolution` and
`signed_maurer_distance_map` **1.85–1.95×** — paired within round, tight bands, three
rounds. It would have been a headline. **It is an artifact.** A 4-way control on
`signed_maurer_distance_map` alone put all four binaries within 1% (2.87–2.90 ms), and the
same binary that read **6.03 ms** in one campaign read **2.89 ms** in another. A binary
raced against a *copy of itself* agrees to 2–4%, so this is not build noise: it is the
**op-set inside the bench process** changing an op's time by up to 2× at the 64³ cell.
Every number in this document that I claim is either (a) a stencil op, reproduced across
four campaigns, or (b) from a 4-way single-op control. The rest I am throwing away.

## UNFIXED

- **The 64³ cross-campaign instability itself** — up to ~2× for `otsu` / `fft` /
  `signed_maurer_distance_map` depending on which ops share the process. It makes any
  single-campaign small-cell comparison of a *non-stencil* op untrustworthy, and nobody has
  looked at it. It did not touch the stencil results (four campaigns agree), but it is a
  live hazard for the next person who benchmarks at 64³.
- **`c_max/c_mean` is still 3.1–4.0×** after the split. The class model predicts ~1, so
  either the classes are not as uniform as the model says, or contention inflates a chunk's
  cost by when it runs. `mean`'s occupancy (56.8 of 96) is capped by whatever that is.
- **The ~2× memory tax** — measured, flat, grain-independent, unattributed (bandwidth vs
  shared-cache). Per instruction, not chased.
- **`gradient_magnitude` is now ramp-bound**: at a 1.66 ms wall, the ~0.9 ms pool wake-up
  is over half the pass. No grain and no partition can touch that; it needs the pass not to
  pay a ramp.

## One process failure worth recording

Mid-round I ran `git checkout -- crates/sitk-core/src/parallel.rs` to revert a temporary
probe, and **deleted the implementation with it** — the probe was in the same file. I
rebuilt it from the turn's own record, re-gated, and re-measured on the restored source
(the FINAL campaign above exists because of this, not in spite of it). No result in this
document comes from the pre-loss build.
