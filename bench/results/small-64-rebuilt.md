# The 64³ table, rebuilt — and the ITK column was flattered, not penalised

The retracted table is replaced here. Both columns are measured under the same
protocol (`bench/run_protocol.py`): **one op per process, quiet gate, warm-up long
enough to outlast the transient, median of ≥6 launches, and a cell whose
launch-to-launch spread exceeds the 1.15× noise floor is REFUSED rather than
published.** Rust is merged main; ITK is 6.0 with the C++ harness's warm-up fixed
(below). A refused cell is a result and is printed as one.

## The finding that matters: ITK's 64³ numbers were *under*-reported by up to 2.4×

`bench/cpp/main.cxx` warmed up with **one call**. That is not a warm-up at 64³,
where an ITK op costs 0.6–37 ms — and the consequence is the opposite of the Rust
side's. ITK's first calls in a fresh process are **fast**, and the cost *climbs* to a
plateau. The old harness's ten samples measured the transient. `gradient_magnitude`,
64³/`tN`, forty consecutive samples, old binary:

```
2.33 1.32 0.95 0.90 0.93 1.25 0.97 1.44 1.28 1.43 1.50 2.10 1.46 1.70 1.33 1.47
1.60 2.25 1.41 1.31 2.03 2.64 2.28 2.32 2.38 2.35 2.29 2.29 1.42 1.53 2.69 2.52
2.78 2.83 1.51 2.72 2.36 2.32 2.65 2.35
```

It climbs from 0.90 to a ~2.4 ms plateau. The fixed binary sits at that plateau from
sample 0 (`2.66 2.33 2.34 2.31 2.35 …`). The published ITK number was **1.01 ms**;
the sustained cost is **2.43 ms**. Same shape on `discrete_gaussian` (3.2 → 7.0) and
`otsu_threshold` (2.1 → 3.3).

It is a **fresh-process transient, not a cold box**: the old binary measures fast in
every round *regardless of whether the new binary ran immediately before it*, so the
box being hot does not remove it. No quiet-gating or box-warming can — only a warm-up
that outlasts it.

**The fix** (`WARM_UP_MS = 3000`, `main.cxx`): warm up until 3 s of wall time has been
spent, not for a fixed *count* of calls. Same 3 s the Rust harness uses, derived from
the same measured ramp — not tuned until an ITK number came out somewhere pleasant.
An expensive 512³ cell still gets exactly one warm-up call, which is what the loop
condition already gives it, so the >120 s skip gate is unchanged.

### ITK-side inflation, quantified (old/new, paired, one op per process)

`old/new < 1.00` means **the frozen C++ harness reported ITK FASTER than it is.**

| op, 64³ `tN` | old (1 call) | new (3 s) | old/new | ITK was flattered by |
|---|---|---|---|---|
| `gradient_magnitude` | 1.006 | 2.426 | **0.41** | **2.4×** |
| `discrete_gaussian` | 3.499 | 7.157 | **0.49** | **2.0×** |
| `otsu_threshold` | 2.404 | 3.817 | 0.63 | 1.6× |
| `mean` | 4.164 | 5.833 | 0.71 | 1.4× |
| `gmrg` | 8.283 | 11.671 | 0.71 | 1.4× |
| `smoothing_recursive_gaussian` | 2.117 | 2.880 | 0.73 | 1.4× |
| `signed_maurer_distance_map` | 25.485 | 34.200 | 0.75 | 1.3× |
| `median` | 25.551 | 28.984 | 0.88 | 1.1× |
| `fft_convolution` | 23.565 | 26.612 | 0.89 | 1.1× |
| `rescale_intensity` | 0.913 | 1.021 | 0.89 | 1.1× |
| `connected_component` | 135.18 | 139.17 | 0.97 | — |
| `binary_dilate` | 36.947 | 36.175 | 1.02 | — |

**Both columns of the old 64³ table were wrong, in opposite directions**: the port was
inflated up to 2.02× and ITK was deflated up to 2.4×, so a published ratio could be
off by ~5×. That is the whole explanation of the 64³ "losses".

## The rebuilt table — 64³, `tN`, merged main vs ITK 6.0

| op, 64³ `tN` | **rust ms** (spread) | **itk ms** (spread) | **rust/itk** | status |
|---|---|---|---|---|
| `connected_component` | 11.637 (1.01×) | 142.757 (1.11×) | **0.08×** | certified |
| `binary_dilate` | 3.200 (1.09×) | 36.274 (1.01×) | **0.09×** | certified |
| `median` | 3.941 (1.09×) | 29.018 (1.14×) | **0.14×** | certified |
| `gradient_magnitude` | 0.515 (1.11×) | 2.458 (1.07×) | **0.21×** | certified |
| `otsu_threshold` | 0.886 (1.07×) | 3.855 (1.13×) | **0.23×** | certified |
| `discrete_gaussian` | 1.980 (1.04×) | 7.536 (1.05×) | **0.26×** | certified |
| `mean` | 1.664 (1.02×) | 6.020 (1.11×) | **0.28×** | certified |
| `rescale_intensity` | 0.420 (1.03×) | 1.054 (1.14×) | **0.40×** | certified |
| `smoothing_recursive_gaussian` | 2.066 (1.07×) | 2.883 (1.04×) | **0.72×** | certified |
| `signed_maurer_distance_map` | 2.718 (1.08×) | *refused* (1.22×) | — | **ITK column refused** |
| `gradient_magnitude_recursive_gaussian` | *refused* (2.39×) | 11.953 (1.04×) | — | **rust column refused** |
| `fft_convolution` | *refused* (1.84×) | *refused* (1.32×) | — | **both refused** |

**Every certified cell is a win, 0.08×–0.72×.** The old table published `mean` as a
**2.82× loss**; it is a **0.28× win** — the loss was the two harness defects, plus the
cost-class split, which is in merged main and is real code, not measurement. The old
table's other 64³ "losses" (`gradient_magnitude` 1.05×, `gmrg` 1.19×,
`smoothing_recursive_gaussian` 1.34×, `fft_convolution` 1.41×) do not survive either:
the two that can be certified are wins (0.21×, 0.72×) and the other two are refused.

### The refusals, and what they mean

- **`gmrg` (rust), spread 2.39×.** Six solo legs on a quiet box: `7.8 9.0 7.3 | 15.3
  17.4 16.7`. Two modes a factor of two apart, and here they came in *runs* — the
  first three legs fast, the last three slow — so it is not a per-launch coin flip
  either. This is the unexplained 2× mode from `harness-instability-result.md`,
  showing up **solo**, which one op per process was supposed to avoid. It does not.
- **`fft_convolution` (both columns).** Rust 12.4 vs 22.6–22.9; ITK spread 1.32×.
  Same shape.
- **`signed_maurer_distance_map` (ITK column), spread 1.22×.** Marginal; more legs
  would probably settle it. Rust's own column for this op is certified at 2.718.

A refused cell is not a missing number, it is a measured statement: **this box cannot
resolve this cell, and anyone quoting a single figure for it is quoting noise.**

## The C++ cell-order defect: it does not exist there — but a different one does

You asked whether `for size { for op { t1 then tN } }` exists in the C++ harness. **It
does not.** `--config t1|tN` is a *process-level* flag
(`SetGlobalDefaultNumberOfThreads` at startup, `main.cxx:223`), so a C++ process runs
one config only and no serial leg is ever interleaved before a parallel one. The Rust
harness's defect is Rust's alone.

The C++ harness has a *different* serial interleave, and it is worth recording: the
per-sample output checksum (`fnv1a64` over the whole output, `main.cxx:135`) runs
single-threaded **between every sample**, outside the timed region. At 512³ that is
hundreds of ms of one-thread work between each parallel sample. It did not produce a
resolvable effect at large (below), so I am naming it, not claiming it.

## Verdict on the medium and large ITK columns

**256³ — not safe for `gradient_magnitude`.** Paired old/new, three rounds each, bands
do not overlap: old `[35.4, 35.6, 38.0]`, new `[46.5, 49.0, 49.2]` → **old/new 0.73**,
ITK under-reported by **1.37×**. The published medium ratio (`gradient_magnitude`
0.64×) is built on the flattered ITK number and must be retaken; corrected against the
new ITK figure it moves to roughly **0.43×** — a bigger win, not a smaller one.
`otsu_threshold` at 256³ is **unresolved** (0.69 in one campaign, 1.03 in another);
`discrete_gaussian` 0.94, `mean` 0.99, `smoothing_recursive_gaussian` 0.99,
`rescale_intensity` 1.17 are inside or near the floor. **The other six 256³ ops were
not tested.**

**512³ — no resolvable defect on the two ops tested.** `gradient_magnitude` old/new
**1.00**, `otsu_threshold` **0.91**, three rounds each, bands overlap. On this
evidence the 512³ ITK column stands. **I tested two of twelve ops and I will not
generalise from that** — the honest statement is that the defect is size-dependent
(crushing at 64³, real for one op at 256³, absent at 512³ where tested), and the
remaining 512³ ops are untested, not cleared.

## UNFIXED

- **The 2× mode is not explained, and it is not avoided either.** One op per process
  removes it for most ops but **not for `gmrg` or `fft_convolution`**, which are
  bimodal solo on a quiet box. My previous round's claim that solo measurement escapes
  it was too strong, and this is where it breaks.
- **`bench/run_protocol.py` gated only at the leg's edges**, so a sibling `cargo` that
  started and finished inside a leg was invisible. It reported five 64³ cells as
  unstable (spread up to 1.96×) that are tight to 1.02–1.11× when re-taken. Fixed
  (`LegWatch` samples during the leg); the five cells above are the re-taken ones.
- **Nine of twelve 256³ ITK ops, and ten of twelve 512³ ITK ops, are untested** for
  the warm-up defect.
- **`t1` columns are untested** on both harnesses at every size.
