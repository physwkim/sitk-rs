# Why does the same binary, on the same op, read 2.89 ms and 6.03 ms?

Written before any leg. The observation that forces this round: `signed_maurer_distance_map`
at 64³/`tN`, **the same binary**, one op per process both times, quiet box both times —
**6.03 ms** in one campaign and **2.89 ms** in another. Within a campaign the legs are tight
(paired ratios came back as `[1.85, 1.85]`), which is what made the artifact look like a
result. So the variance is **between process launches**, not within them, and it is large
enough (2×) to have manufactured a headline.

Every ratio in `doc/bench-results.md` that compares two numbers taken in different campaigns
is suspect until this is closed.

## The box, as facts rather than guesses

Xeon Gold 6542Y ×2 — **2 sockets, 2 NUMA nodes**, 48 physical cores / 96 threads, governor
`schedutil`, **800 MHz to 4.1 GHz** (a 5× clock range), THP on `madvise`. The bench pool is
96 workers, so every pass spans both sockets.

## Candidates, each with the counter that kills it

**H10 — the allocator/page-fault lottery (my primary).** Each bench iteration allocates a
fresh output volume (1–2 MB at 64³) and drops it. glibc serves an allocation that large by
`mmap` when it is above the mmap threshold, and **that threshold is dynamic**: it rises only
once glibc has *seen* a large mmapped block freed. So a process either (a) settles into
heap reuse — the buffer is recycled, its pages stay mapped, and iteration cost is the op —
or (b) never settles, and every iteration pays `mmap` + a fresh page fault per 4 KB page
(≈512 faults per 2 MB buffer) + `munmap`. Which one a process lands in depends on its
allocation history, which is exactly what "which ops are in this binary" changes. **This is
bimodal per process and stable within one, which is the observed shape.**
*Discriminator: minor page faults, a counter, not a clock.* A slow leg must show
**order-of-magnitude more minor faults** than a fast one. **Falsifier: fault counts equal
across fast and slow legs → H10 is dead**, and I would rather learn that from `getrusage`
than from a benchmark.

**H9 — frequency/thermal history.** `schedutil` over a 5× clock range, and a leg's neighbours
in time determine what clock it starts at and settles to. *Discriminator: mean core MHz
sampled during the leg.* If H9 is the mechanism, the leg's ms must **correlate with its own
measured MHz** (I predict |r| > 0.7 if it is the cause). **Falsifier: slow and fast legs run
at the same clock.**

**H8 — NUMA first-touch placement.** The input buffer is first-touched by the main thread
and lands on one node; 96 workers then read it from both. A process is one placement draw.
*Discriminator: `numactl --interleave=all`, which removes the draw.* If placement is the
mechanism, interleaving **collapses the spread**. If it is not, interleaving changes the mean
and leaves the spread. Note this predicts *at most* a remote-vs-local latency penalty, which
is well under 2× on this part — **so H8 is the weakest of the three and I say so now**, before
it has a chance to look prescient.

**H11 — op-set interaction per se.** Already refuted in part: the 6.03/2.89 pair were *both*
single-op processes. Whatever this is, it survives one op per process, so "one op per process"
cannot be the fix on its own.

## Predictions

**P33 — the spread reproduces under a fixed protocol.** 20 launches of the *same* binary on
the *same* op, identical gaps, quiet box: the max/min ratio comes back **≥1.5×**, and the
distribution is **bimodal**, not a smear. **Falsifier: a tight unimodal distribution → the
variance is driven by something about the campaign I have not modelled, and I have to go
looking again rather than declare a mechanism.**

**P34 — the fast/slow split is visible in minor page faults, not in the clock.** Slow legs
show ≥10× the minor faults of fast legs; MHz differs by less than 20% between them.
**Falsifier: faults equal, MHz correlated → it is H9, and the fix is a frequency protocol,
not an allocator one.**

**P35 — pinning the allocator's behaviour collapses the spread.** With
`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=…` (or `MALLOC_MMAP_THRESHOLD_`) set high enough
that the output buffer never leaves the heap, 20 launches agree **within 10%**, max/min.
**Falsifier: the spread survives → H10 is wrong even if the fault counts pointed at it, and
the protocol must come from whatever survives.**

## What the deliverable is

Not an explanation — a **protocol**, and a proof of it: *the same binary, measured twice
under the protocol, agrees inside the noise floor.* Then, and only then, `doc/bench-results.md`
gets audited against it. I do not edit that file (the main panel owns it); I will produce the
audit as a list of which numbers survive the protocol and which cannot be reproduced, and
hand it over.
