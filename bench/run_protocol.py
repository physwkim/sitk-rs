#!/usr/bin/env python3
"""Run the Rust bench under the measurement protocol.

Without this driver a number from `bench_ops` is not comparable to a number from
another run of `bench_ops`. Three things break it, all measured in
`bench/results/harness-instability-result.md`:

1. **The box ramps.** After the box idles, the first ~2 s of a 96-thread pass runs
   up to 70% slow and decays. Criterion's warm-up used to be 500 ms, so the ramp
   landed inside the measurement. Fixed at source (`WARM_UP_MS` in `bench_ops.rs`):
   this driver does not have to do anything about it, but every number below is
   only reproducible on a binary that carries that fix.
2. **The op-set moves an op by 2×.** In a twelve-op process, short `tN` cells at
   64³ flip bimodally between two modes a factor of two apart -- `rescale_intensity`
   {0.42, 0.89 ms}, `discrete_gaussian` {2.0, 3.9}, `gmrg` {7.1, 14.3} -- and which
   mode a cell lands in changes run to run. Measured **solo**, the same cells are
   stable to 1.05-1.12x and always land on the fast mode. So: **one op per
   process.** That is what this driver does and it is why it exists.
3. **The box is shared.** `/proc/loadavg` reads 18-21 on an idle box here and
   cannot gate anything; real busy cores come from `/proc/stat`. A leg that ran
   beside a `cargo`/`nextest`/CUDA process is dropped, not averaged in.

Noise floor under this protocol, from two independent campaigns per op
(`harness-instability-result.md`): within-campaign spread <=1.13x at 643, <=1.08x
at 2563, <=1.15x at 5123; cross-campaign medians agree to <=5.1%. **A ratio below
1.15x is not a result.**

usage:
    python3 bench/run_protocol.py --out bench/results/rust-protocol.ndjson
    python3 bench/run_protocol.py --ops mean,median --sizes small --repeats 6
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NDJSON = os.path.join(ROOT, "target", "bench-results-rust.ndjson")
FOREIGN = ("cargo", "nextest", "rustc", "cc1plus", "nvcc", "device_pipeline", "mattes")

OPS = ["binary_dilate", "connected_component", "discrete_gaussian", "fft_convolution",
       "gradient_magnitude", "gradient_magnitude_recursive_gaussian", "mean", "median",
       "otsu_threshold", "rescale_intensity", "signed_maurer_distance_map",
       "smoothing_recursive_gaussian"]


def busy_cores(window=5.0):
    def snap():
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(v), v[3] + v[4]
    t0, i0 = snap()
    time.sleep(window)
    t1, i1 = snap()
    return ((t1 - t0) - (i1 - i0)) / (t1 - t0) * os.cpu_count()


def foreign():
    ps = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True).stdout
    return sorted({c.strip() for c in ps.splitlines()[1:]
                   if any(c.strip().startswith(f) for f in FOREIGN)})


class LegWatch(threading.Thread):
    """Watches for foreign load *during* a leg, not just before and after it.

    Gating only at the leg's edges is not a gate: a sibling `cargo`/`nextest` that
    starts and finishes inside the leg is invisible to it, and the leg is published.
    That is not hypothetical -- it happened while this file was being written, and
    it reported five 64^3 cells as unstable (spread up to 1.96x) that are tight to
    1.02-1.11x when re-taken with this watch armed. A leg that shared the box with a
    foreign process at ANY 2 s sample is dropped.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.seen = set()

    def run(self):
        while not self.stop.wait(2.0):
            self.seen.update(foreign())


def wait_quiet(deadline, need=2):
    streak = 0
    while streak < need:
        if time.time() > deadline:
            return False
        streak = streak + 1 if (busy_cores() < 3.0 and not foreign()) else 0
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=None, help="bench_ops binary (default: cargo-built release)")
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--configs", default="t1,tN")
    ap.add_argument("--repeats", type=int, default=6, help="launches per (op, size); the median leg is published")
    ap.add_argument("--budget", type=float, default=7200.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    binary = args.binary
    if binary is None:
        subprocess.run(["cargo", "build", "--release", "--bench", "bench_ops", "-p", "sitk-filters"],
                       cwd=ROOT, check=True)
        cands = [os.path.join(ROOT, "target", "release", "deps", f)
                 for f in os.listdir(os.path.join(ROOT, "target", "release", "deps"))
                 if f.startswith("bench_ops-") and not f.endswith(".d")]
        binary = max(cands, key=os.path.getmtime)

    deadline = time.time() + args.budget
    cells, dropped = {}, []
    for size in args.sizes.split(","):
        for op in args.ops.split(","):
            for k in range(args.repeats):
                if not wait_quiet(deadline):
                    print("NO QUIET WINDOW — stopping", file=sys.stderr)
                    sys.exit(1)
                env = dict(os.environ)
                env["SITK_BENCH_OPS"] = op          # ONE op per process. See (2) above.
                env["SITK_BENCH_SIZES"] = size
                env["SITK_BENCH_CONFIGS"] = args.configs
                before = foreign()
                watch = LegWatch()
                watch.start()
                p = subprocess.run([binary], env=env, cwd=ROOT, capture_output=True, text=True)
                watch.stop.set()
                watch.join(timeout=3)
                if p.returncode != 0:
                    print(p.stderr[-2000:], file=sys.stderr)
                    sys.exit(1)
                dirty = sorted(set(before) | watch.seen | set(foreign()))
                if dirty:
                    dropped.append((op, size, k, dirty))
                    print(f"  drop {op} {size} leg {k}: foreign={dirty}", flush=True)
                    continue
                for row in map(json.loads, open(NDJSON)):
                    cells.setdefault((row["op"], row["size"], row["config"]), []).append(row)
                print(f"  {op} {size} leg {k} ok", flush=True)

    out = []
    for key in sorted(cells):
        legs = cells[key]
        timed = [r for r in legs if r["ms_median"] is not None]
        if not timed:
            out.append(legs[0])          # a skipped/projected row: publish it as it stands
            continue
        med = statistics.median(r["ms_median"] for r in timed)
        pick = min(timed, key=lambda r: abs(r["ms_median"] - med))
        lo = min(r["ms_median"] for r in timed)
        hi = max(r["ms_median"] for r in timed)
        print(f"{key[0]:38s} {key[1]:6s} {key[2]:3s}  n={len(timed)}  "
              f"median={med:9.3f}  spread={hi / lo:.2f}x"
              f"{'   *** ABOVE THE 1.15x NOISE FLOOR — do not quote this cell' if hi / lo > 1.15 else ''}")
        out.append(pick)

    if args.out:
        with open(args.out, "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        print(f"\nwrote {len(out)} rows to {args.out}")
    if dropped:
        print(f"{len(dropped)} legs dropped for foreign load (listed above); they are not in the output")


if __name__ == "__main__":
    main()
