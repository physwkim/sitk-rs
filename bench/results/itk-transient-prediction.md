# Three predictions, written before the clock

## 1. The ITK transient should shrink with volume — and may be nil at 512³

The transient is **per-process and climbing**: ITK's first calls in a fresh process
are fast and the cost rises to a plateau over the first few seconds of work. A
benchmark that runs longer therefore *amortizes* it rather than escaping it, so the
published inflation should fall as the per-call cost rises.

**P36 — the defect scales with (transient / measurement duration).** The transient is
~2–3 s of process time. A 64³ ITK op costs 0.6–37 ms, so its ten samples land wholly
inside the transient — hence the measured 0.41–1.02 (up to 2.4× flattery). At 256³
an op costs 40–4400 ms and at 512³ 100–30 000 ms, so the ten samples increasingly
outlast it. **Predicted: `|1 − old/new|` at 256³ mostly ≤ 0.3 and concentrated in the
cheapest ops; at 512³ ≤ 0.1 for every op, i.e. inside the noise floor.**
**Falsifier: any 512³ op with disjoint old/new bands and a ratio outside 0.90–1.10.**
If that fires, the defect is not amortization and I have the mechanism wrong.

**A structural exclusion, derived rather than measured.** The fixed warm-up is
`do { fn(); } while (elapsed < 3 s)`. For any op whose *single call* already costs
≥ 3 s, that loop runs **exactly one call** — which is precisely what the old harness
did. **The two binaries are then the same code path and cannot differ.** From the
published ITK figures that excludes, by construction and not by measurement:

- 256³: `connected_component` (4352 ms/call)
- 512³: `connected_component` (30 180), `binary_dilate` (14 812), `median` (3486),
  `fft_convolution` (3327)

These are not "skipped for time" — there is nothing there to measure. Every other op
at both sizes gets measured, paired, one op per process.

## 2. The `gmrg`/`fft` bimodality: runs across *processes* is the clue

Six solo legs of `gmrg` on a quiet box: `7.8 9.0 7.3 | 15.3 17.4 16.7` — fast run,
then slow run. Each leg is **a separate process**, so whatever persists is persisting
**in the box, not in the process**. That kills every per-process explanation at once,
including the ones I already excluded for the multi-op version (heap layout, allocator
threshold, per-process thread placement) — and it means I must not assume the solo
mode inherits those refutations. It is a *slowly drifting box state*.

**P37 — it is NUMA page placement driven by node free-memory imbalance.** This box has
two nodes with **43 GB free on node0 and 25 GB on node1**. When a node's free list is
depleted the kernel falls back to the other node, so a process's pages land remote —
and that state drifts over minutes and persists across process launches, which is
exactly a *run* of slow legs rather than a coin flip. A bandwidth-bound stencil op
reading remote memory at 2× the latency is the observed factor.
*Discriminator: `numa_hit` vs `numa_foreign`/`other_node` per leg from
`/sys/devices/system/node/node*/numastat`, and `numactl --interleave=all`, which
removes the fallback by construction.* **Predicted: slow legs show a materially higher
other-node fraction, and interleaving collapses the spread below the 1.15× floor.**
**Falsifier: the other-node counters are flat across fast and slow legs, or
interleaving leaves the 2× spread standing → not NUMA, and I say so rather than
reach for the next story.**

## 3. The C++ per-sample serial checksum

`main.cxx:135` runs a single-threaded `fnv1a64` over the whole output **between every
sample**, outside the timed region. At 512³ (537 MB) that is hundreds of ms of
one-thread work between parallel samples — the same shape as the defect just fixed:
**a serial leg cooling the box before a parallel measurement**. "No resolvable effect"
was also true of the ITK transient until it was looked at with a 40-sample trace.

**P38 — it inflates the ITK 512³ `tN` numbers.** A binary that checksums once instead
of per sample should measure the *same* op **faster** at 512³, by the ramp cost the
cooling reintroduces. **Predicted: a resolvable effect, ≥ 1.15× on at least one
bandwidth-bound 512³ op.** **Falsifier: every op's bands overlap → the checksum is
too short relative to a 512³ pass to cool anything, and I price it at zero and say
so.** Either way it gets a number, not a mention.

## What would make me stop

If P36's falsifier fires at 512³, the amortization model is wrong and I do not get to
publish a "shrinks with volume" story with an exception attached to it.
