# The ITK transient at 256³/512³, the bimodality, and the checksum — graded

Graded against `itk-transient-prediction.md`, committed (`9325bb9`) before any leg.
Box mine except where a round is voided for foreign load; every kept leg gated on
`/proc/stat` busy cores. Paired old/new is the same binary source differing only in
`WARM_UP_MS`; one op per process throughout.

## P36 — the transient shrinks with volume, but it is NOT nil at 512³. FALSIFIED.

The amortization model was right in its shape and wrong in its strong claim. The
defect does shrink — but two 512³ ops have disjoint bands outside the predicted
0.90–1.10, and one of them is *inverted*.

### 256³ (`old/new < 1.00` ⇒ ITK reported faster than it is)

| op | old/new | reading |
|---|---|---|
| `gradient_magnitude` | **0.77** | ITK flattered **1.30×**, four campaigns agree (0.73–0.84), bands disjoint |
| `otsu_threshold` | 0.74 | signal present but its ITK arm is itself unstable (1.35× within-arm); soft |
| `rescale_intensity` | 1.14 | marginal, at the floor |
| `signed_maurer_distance_map` | 1.02 | nil — the earlier 0.78 was an n=1 artifact, gone at n=3 |
| `smoothing_recursive_gaussian`, `discrete_gaussian`, `median`, `mean`, `gmrg`, `binary_dilate` | 0.96–1.04 | nil |

`connected_component` was **not measured** (see the structural exclusion below).

### 512³

| op | old/new | reading |
|---|---|---|
| `rescale_intensity` | **1.15** | disjoint bands, **inverted** — the old harness reported ITK *slower*. 3 rounds: old [261.5, 262.4, 269.3] / new [223.1, 228.0, 229.1] |
| `otsu_threshold` | 0.88 | its new arm spans 1.34× on its own; unresolved, not a clean signal |
| `gradient_magnitude`, `gmrg`, `smoothing_recursive_gaussian`, `discrete_gaussian`, `mean`, `signed_maurer_distance_map` | 0.99–1.01 | nil |

**P36's falsifier fired: `rescale_intensity` at 512³ is outside 0.90–1.10 with disjoint
bands.** So volume does not drive the defect to zero, and the mechanism is not pure
amortization of a climbing transient. `rescale_intensity` is the *cheapest* op (a
single pass, ~235 ms at 512³) yet it is the one that misinflates at the *largest*
size — the opposite of what a climbing-transient-amortized-by-duration predicts, and
the inversion (ITK measured *slower* on one warm-up call than on many) says the first
call is *slow* here, not fast. There are at least two transients with opposite signs
and the amortization story only described one. I am not going to name the second from
this evidence; what I can say is falsifiable and it failed.

### The structural exclusion held exactly

Predicted from the fixed warm-up loop (`do { fn(); } while (elapsed < 3 s)`): any op
whose single call already costs ≥ 3 s runs exactly one warm-up call in **both**
binaries and cannot differ. Confirmed — `connected_component` (256³ and 512³),
`binary_dilate`, `median`, `fft_convolution` (512³) are the same code path on both
arms. Not measured because there is nothing to measure, not because of budget.

## P37 — the bimodality is NOT NUMA. FALSIFIED, cleanly.

`gmrg` solo, 5 paired legs, plain (A) vs `numactl --interleave=all` (B), with the
system NUMA counters over each leg:

```
leg0 A  ms= 6.887  other_node=       12  numa_miss=0  numa_foreign=0
leg1 A  ms=17.366  other_node=       30  numa_miss=0  numa_foreign=0
leg2 A  ms=17.762  other_node=       64  numa_miss=0  numa_foreign=0
leg0 B  ms= 6.846  other_node=  101761  (interleave: 43% of pages remote by design)
leg2 B  ms=13.491  other_node=  101405
```

**Both falsifier clauses fired.** `numa_miss` and `numa_foreign` are **zero** on every
plain leg — the kernel never fell back to the far node, so the free-memory imbalance
hypothesis is simply wrong about what the kernel did. And `--interleave=all`, which
*does* put 43% of pages on the remote node, leaves the 2× spread standing (B spread
1.99×). Remote memory is not the mechanism; it is not even a contributor worth 5%.

### What the bimodality actually is — as far as cheap discriminators reach

I excluded, for the *solo* mode specifically and not by inheritance: clock (the 5.66 ms
and 2.93 ms legs ran within 2% of each other), NUMA (above), allocator mmap/trim
threshold (pinning both to 1 GiB does not flip it — `gmrg` A/B medians 16.79 vs 16.97).
What survives, from counters I already had:

- **The fast mode is the *idle* box.** After a 180 s idle, `gmrg` leg 0 is **7.14 ms**,
  and legs 1–7 are 14–18 ms. The box flips to the slow state after the first heavy
  pass and **stays there for minutes, across separate processes** — which is why the
  legs come in runs, not as a coin flip.
- **The slow mode re-faults.** Slow legs take **2367–2975 minor faults per iteration**
  against 1328–2278 fast, and burn 485–596 CPU-seconds against 411–434 — measurably
  *more work*, not the same work run slower. `pgalloc` tracks it (fast legs 292–316k,
  slow 363–445k).
- **It is not the free-memory or cache level.** `MemFree`, `MemAvailable`, `Cached` and
  `compact_stall` are flat to four digits across fast and slow legs.

That localizes it to **page-backing granularity** — a box-wide state (THP is `madvise`
here; the first heavy 96-thread pass fragments the free lists and huge-page backing is
not restored for minutes) that changes the fault cost of the next process's buffers.
It is the **same family as the "~2× memory tax" carried in UNFIXED since Round 6**, now
with a fault-count signature attached. Nailing THP-vs-khugepaged-vs-something requires
`/sys` writes I do not have and a chase I was told not to run. **I am stopping at
"box-wide page-backing state, per-process candidates excluded, fault-count confirmed"
and leaving it in UNFIXED**, which is the honest boundary.

The consequence for the harness is unchanged: three 64³ ops (`gmrg`, `fft_convolution`,
and `signed_maurer` on the ITK side) have no certifiable number because this mode is
larger than the noise floor, and one op per process does **not** avoid it — my Round-8
claim that it did was wrong, and this is the measurement that retires it.

## P38 — the per-sample checksum costs nothing resolvable. FALSIFIED (as predicted).

Same binary, `SITK_BENCH_NO_CHECKSUM` on/off, 512³, paired, one op per process. An n=3
pass on `mean` teased at C/N **1.16**, but it was one high C leg over a short sample;
at **n=5** it collapses:

```
mean  C legs [478.6, 483.6, 485.9, 489.8, 572.7]  N legs [478.1, 481.3, 486.1, 488.9, 503.0]  C/N=1.00
```

`signed_maurer` 0.86 and `discrete_gaussian` 1.02 both have overlapping bands.
**Bands overlap ⇒ the checksum is too short relative to a 512³ pass to cool anything.
Priced at zero.** The 537 MB single-threaded fnv1a64 is real work, but a 512³ `mean`
pass is long enough to re-warm before the next sample, so it does not reach the timed
region.

The probe (an env-gated skip in `timeFilter`) was reverted after measuring — a negative
result does not justify a permanent flag that disables a correctness check. `main.cxx`
carries only the warm-up fix.

## Verdict for the document

- **256³ ITK is safe except `gradient_magnitude`.** That row's published 0.64× is built
  on an ITK number flattered 1.30×; corrected it moves to roughly **0.43×** — a bigger
  win. `otsu_threshold` at 256³ is soft (unstable ITK arm), not a clean correction.
- **512³ ITK is safe except `rescale_intensity`.** Its published 0.41× is built on an
  ITK number that the paired test finds inflated the *other* way (old/new 1.15,
  inverted), so the corrected ratio is *smaller* — around **0.35×**, a slightly bigger
  win. `otsu_threshold` at 512³ is unresolved.
- **Every other 256³ and 512³ row tested is within the noise floor of old/new = 1.0**
  and stands. Ten of twelve at each size were tested; the untested ones are the
  ≥3 s-per-call ops, which are excluded by construction, so **coverage at 256³/512³ is
  now complete** for ops that can differ.

## UNFIXED

- **The bimodality (`gmrg`, `fft_convolution`).** Box-wide page-backing state, flips
  after the first post-idle pass, persists across processes for minutes. Per-process
  candidates (clock, NUMA, allocator threshold, heap layout) all excluded; localized to
  minor-fault/page-backing cost. Same family as the Round-6 memory tax. Not chased
  further per standing instruction. Three 64³ cells remain uncertifiable because of it.
- **The second ITK transient with opposite sign** (`rescale_intensity` misinflates at
  512³, inverted). Named, not explained; one paired signal, not a mechanism.
- **`otsu_threshold`** is soft at both 256³ and 512³ from its own within-arm instability,
  independent of the warm-up question.
- **All `t1` columns** remain untested for the transient, both harnesses, every size.
