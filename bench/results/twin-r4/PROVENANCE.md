# The r3→r4 twin: is the `Stencil`/`WindowView` refactor a regression?

Raw data for the question "did `gradient_magnitude` medium/large and
`discrete_gaussian` large really get slower between the r3 and r4 sweeps, or did
the box move under them?"

**These are not sweep results and must never be quoted as any.** They live in a
subdirectory precisely so a `bench/results/*.ndjson` glob cannot reach them: they
are an A/B of two *trees*, and mixing legs into a published comparison would
silently overwrite rows by `(op, size, harness, config)` key.

## The two trees

| tree | commit | what it is |
|---|---|---|
| `pre` | `fd2b372` | the tree the published **r3** sweep ran on — before the grain seam and before the `Stencil`/`WindowView` refactor |
| `post` | `main` (`946846c`) | the tree the published **r4** sweep ran on |

Both binaries were built from those trees plus one commit that is common to both:
an env-var allow-list (`SITK_BENCH_OPS` / `SITK_BENCH_SIZES` /
`SITK_BENCH_CONFIGS`) that gates *which* cells run and nothing inside one. With
the variables unset — which is how all four `full-leg*` files were produced — the
harness is the published harness, cell for cell. The bench harness source is
otherwise byte-identical between the two trees (`git diff fd2b372 main --
crates/sitk-filters/benches` is empty), so the timed path is the same code on
both sides of the seam.

## The legs

| file | tree | wall |
|---|---|---|
| `full-leg1-post-main.ndjson` | post | 899 s |
| `full-leg2-pre-fd2b372.ndjson` | pre | 866 s |
| `full-leg3-pre-fd2b372.ndjson` | pre | 865 s |
| `full-leg4-post-main.ndjson` | post | 888 s |

Full 12-op sweeps, run **post / pre / pre / post** (ABBA, so a monotonic drift in
the box cancels rather than loading onto one tree), back to back, 18:01–19:00 on
2026-07-14. Their ~15 min wall time matches the published legs', which is the
check that this is the same path and not a shortened one.

`subset-12-rounds.ndjson` is a *different* process shape — 6 pre + 6 post rounds
of a 2-op subset (`gradient_magnitude`, `discrete_gaussian`), ABBA. It is kept
because it is the evidence for how much the *shape* alone moves a number: the
identical `main` binary measures `gradient_magnitude` large at **155 ms** in the
2-op process and **98 ms** in the 12-op sweep. Any comparison across process
shapes is void, and this file is why we know that.

## Output checksums

Every cell carries the same `output_checksum` on both trees — the refactor is
bit-identical, as it claimed to be, and this is the four-leg confirmation of it.
