# Three-lever window — provenance, and the load trace beside the numbers

These NDJSON legs are **A/B variant twins, not a published sweep.** They live in a
subdirectory for the same reason `twin-r4/` does: a `bench/results/*.ndjson` glob
must not be able to pull a variant leg into a published comparison. Nothing here is
a `doc/bench-results.md` number.

## The variants

Each is a separately-built `bench_ops` binary off the same `main`, differing in one
constant or one call:

| | lever |
|---|---|
| `v0` | `main` with the **line grain reverted** to the fixed `GRAIN` block run — the baseline |
| `v1` | `main` — the line grain (`grain(len)` block run), merged in the previous round |
| `v2` | `main` + **`MIN_GRAIN` 2048 → 1024** globally (the D2 candidate) |
| `v3` | `main` + **`with_max_len(1)`** on the line pass's block path |
| `v4` | `main` + the **cost-class floor**: `MIN_GRAIN_INDEXED = 1024` on `fill_indexed` only |

## Campaigns

| | ops | size | rounds kept | what it grades |
|---|---|---|---|---|
| `S` | all 12 | 64³ | 5 (v0,v1,v2,v3) | P2, P3, P4 |
| `S2` | all 12 | 64³ | 5 (v0,v2,v4) | P7, P8, P9 |
| `M` | the 4 line-pass ops | 256³ | 3 (v0,v1,v2,v3) | P1, P5, P6 |
| `L` | `gmrg`, `maurer` | 512³ | 3 (v0,v1,v2,v3) | P1, P5, P6 |

`tN` (96 threads) throughout. A ratio is **paired within a round** — one round is one
process shape — and the band quoted in the report is the min/max of the per-round
ratios, never a ratio of pooled means.

## The load trace, and why loadavg is not it

The box was **not** exclusively mine. Three foreign loads ran during the window: the
sitk panel's `device_pipeline` test loop, a sibling's `cargo`/`clippy`/`nextest`
correctness gate, and — for most of the night — a **C++ ITK build in a different
caucus session** (`cc1plus`, ~25 cores), which nobody in this session controls.

So every leg was gated and sampled:

- **Gate.** No leg starts until 3 consecutive 5 s samples show `/proc/stat` busy
  cores **< 3.0** *and* zero foreign processes (`cargo`, `nextest`, `rustc`,
  `cc1plus`, `nvcc`, `device_pipeline`, `mattes*`, any other `bench_ops`).
- **Sampler.** `load.ndjson` in each campaign directory records busy cores and the
  foreign-process list every 2 s, for the whole campaign, including the waits.
- **Atomic rounds.** The four legs of a round are only comparable inside one process
  shape, so **one contaminated leg voids the whole round** — see `legs.log`, where
  every voided round is named with the process that voided it. Campaign M voided 7
  rounds to keep 3.
- Every kept leg reads `clean` in `legs.log`. `busy_max` on a kept leg is 55–95
  cores: that is *my own* 96-thread pool, which is the point of measuring `tN`.

`uptime`'s loadavg is **not** the gate, and cannot be: on this box it reads 18–21
with nothing running and 325–345 while busy, which `bench/results/README.md` already
records ("is not used anywhere and must not be"). `/proc/stat` busy cores is the
measure that file trusts, and it is what gates here. This substitution is deliberate
and is not the gate that was asked for by name.

## Bit-identity

The harness writes an `output_checksum` per cell. Across all 20 legs of S, all 15 of
S2, all 12 of M and all 12 of L, **every op has exactly one checksum class** — no
variant, at any size, moved a bit. The grain is a scheduling knob and this is the
evidence, not the argument.
