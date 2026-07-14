# C++ (ITK) benchmark harness

Implements the C++ side of `doc/bench-spec.md`. One binary, 12 ops, three
volume sizes, two thread configs (`t1`, `tN`); `gpu` is not applicable here.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # ITK_DIR defaults to the
cmake --build build -j                           # reference verify-build
```

`ITK_DIR` defaults to `/home/stevek/work/ITK-worktrees/verify-build`
(ITK 6.0, static). Override with `-DITK_DIR=...`.

## Run

```sh
./build/sitk_bench_cpp --config tN --samples 10 --out results_tN.ndjson
./build/sitk_bench_cpp --config t1 --samples 10 --out results_t1.ndjson
cat results_*.ndjson > results_cpp.ndjson
```

Flags: `--seed` (default 42 — must match the Rust harness), `--samples`
(default 10), `--config t1|tN`, `--size small|medium|large` (repeatable,
default all), `--op <key>` (repeatable, default all), `--out <path>`.

NDJSON goes to `--out` (or stdout); per-sample timings and the input
checksums go to stderr.

## Notes on the contract

- Thread config is set with `MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1)`
  for `t1`, before any filter is constructed. Run `t1` and `tN` as separate
  processes.
- The input volume is generated once per size, outside every timed region.
  Binary/label inputs are the same volume thresholded at `>= 500.0`.
- Every sample constructs a **fresh filter** and calls `Update()`. A new
  filter has no cached output, so the pipeline is forced to execute in full;
  only `Update()` is inside the timer. The input `itk::Image` has no upstream
  source, so nothing is regenerated inside the timed region.
- `large` is skipped with `"skipped": "too slow"` when the (untimed) warmup
  call exceeds 120 s, per the spec's `> 120 s` rule.
