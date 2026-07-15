# What this benchmark cannot measure

Read this before trusting any number in [`../../doc/bench-results.md`](../../doc/bench-results.md).
Everything below was established the hard way over Rounds 8–9 of the harness
investigation; this page is the index to those limits so nobody re-derives them. The
evidence lives in the three result docs cited inline — this is a map, not a re-run.

## 1. The ramp/transient family — timing near a fresh process is not steady state

There are **three** startup transients on this box, and they do not share a sign, so
"warm up more" is not a blanket fix — you have to know which one an op sits in.

- **The rust box-ramp *penalizes*.** After the box idles, the first ~2 s of a 96-thread
  pass runs up to 70% slow and decays. The old harness's 500 ms warm-up left it inside
  the measurement, and the cell order `for size { for op { t1 then tN } }` ran a
  seconds-long serial leg immediately before every parallel leg — cooling 95 of 96
  cores on purpose. It inflated 64³ `tN` by up to 2.02×. Fixed by `WARM_UP_MS = 3 s`.
  Evidence: [`harness-instability-result.md`](harness-instability-result.md).
- **The ITK per-process transient *flatters*.** ITK's first calls in a fresh process are
  *fast* and the cost climbs to a plateau; the old C++ harness's single warm-up call
  measured the fast phase. It flattered 64³ ITK by up to 2.4×. It is per-process and
  **survives quiet-gating** — no box-warming removes it, only a warm-up that outlasts it.
  Evidence: [`small-64-rebuilt.md`](small-64-rebuilt.md).
- **A second ITK transient at 512³ has the *opposite* sign.** `rescale_intensity` at 512³
  is misinflated with the old harness reporting it *slower*, not faster (old/new 1.15,
  inverted, disjoint bands) — so the amortization model that explained the first two is
  incomplete. Named, not explained. Evidence: [`itk-transient-result.md`](itk-transient-result.md).

**The rule that falls out, both languages:** a fresh-process timing of a sub-plateau op
is not a steady-state number. If ten samples fit inside the first few seconds of a cold
process, you are timing the startup, not the op. The only defense is a warm-up measured
against the box's own ramp (3 s here), not a fixed sample or call count.

## 2. The page-backing bimodality — a box-wide state the protocol refuses rather than guesses

Some ops read 2× apart between runs, in *runs* of fast-then-slow legs, not as a
per-launch coin flip. What it is **not**: not NUMA (the kernel's `numa_miss`/`numa_foreign`
counters are zero on plain legs, and forcing 43% of pages remote with
`numactl --interleave=all` leaves the 2× standing), not clock (fast and slow legs run
within 2% of each other), not the allocator mmap/trim threshold (pinning both to 1 GiB
does not flip it), not heap layout (the mode flips with the process shape held fixed).
What it **is**: a box-wide state that flips after the first heavy post-idle pass and
persists across separate processes for minutes — the fast mode is the *idle* box — with a
**minor-fault signature** (slow legs take ~1.4× the faults per iteration and burn more
CPU-seconds, i.e. more work, while `MemFree`/`Cached`/`compact_stall` stay flat). It is
the same family as the ~2× memory tax carried in UNFIXED since Round 6. Localized to
page-backing granularity; not closed, because closing it needs `/sys` writes and a chase
that was explicitly out of scope. Evidence: [`itk-transient-result.md`](itk-transient-result.md) §P37.

**Operational consequence:** `gradient_magnitude_recursive_gaussian`, `fft_convolution`,
and `signed_maurer_distance_map` (ITK side) have **no certifiable 64³ number**, and one op
per process does *not* avoid the mode. The protocol **refuses** these cells — prints them
as refused with the measured spread — rather than publishing a guess. A refused cell is a
result: "this box cannot resolve this cell," not a missing one.

## 3. What `run_protocol.py` guarantees — and what it does not

Guarantees: **one op per process** (kills the multi-op op-set interaction that flipped
short cells 2×), a **during-leg quiet gate** (`LegWatch` samples foreign load *while* the
leg runs, not only at its edges — an earlier edges-only version wrongly refused five 64³
cells that are tight when re-taken), and **refusal above the noise floor** (a cell whose
launch-to-launch spread exceeds the floor is not certified). Publishes the median of ≥6
launches with the spread beside it.

Does **not** guarantee: escape from the §2 bimodality — it doesn't, and it refuses those
cells instead. And it does **not** certify `t1` — every `t1` column on both harnesses at
every size was never retaken under the protocol and is soft. Evidence:
[`harness-instability-result.md`](harness-instability-result.md) §"The protocol".

## 4. The noise floors, as measured

Within-campaign launch-to-launch spread under the protocol, from two independent
campaigns per op:

| size | floor |
|---|---|
| 64³ | **1.13×** |
| 256³ | **1.08×** |
| 512³ | **1.15×** |

Cross-campaign median disagreement was ≤5.1%. **A ratio inside the floor for its size is
unresolved — full stop.** It is not "parity" and not "a tie"; it is a number the box
cannot distinguish from 1.0. Several rows the old document published as ties (`gmrg`
large 1.01×, `gradient_magnitude` large 1.02×, `smoothing_recursive_gaussian` medium
1.02×) are inside the floor and establish nothing. Evidence:
[`harness-instability-result.md`](harness-instability-result.md) §"The noise floor, measured".
