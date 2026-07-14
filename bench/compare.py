#!/usr/bin/env python3
"""Merge the harnesses' NDJSON and print the comparison table.

Reads every .ndjson given on the command line (rust + cpp + gpu rows all
land in the same stream, per doc/bench-spec.md) and emits one row per
(op, size).

Two things this refuses to do quietly:

- If the same (op, size) has a different `input_checksum` between the
  rust and cpp harnesses, the comparison for that op is VOID and the row
  says so instead of printing a ratio. That equality is the entire basis
  of the comparison.
- A `skipped` measurement prints its reason string. It never becomes a
  blank cell that reads like "no difference".
"""

import json
import sys
from collections import defaultdict

SIZES = ["small", "medium", "large"]


def main(paths):
    # (op, size, harness, config) -> row
    rows = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows[(d["op"], d["size"], d["harness"], d["config"])] = d

    ops = sorted({k[0] for k in rows})
    checksum_conflicts = []

    for size in SIZES:
        present = [op for op in ops if any((op, size, h, c) in rows for h in ("rust", "cpp") for c in ("t1", "tN", "gpu"))]
        if not present:
            continue
        print(f"\n## {size}\n")
        print(f"{'op':<38} {'rust t1':>10} {'cpp t1':>10} {'ratio':>7}   "
              f"{'rust tN':>10} {'cpp tN':>10} {'ratio':>7}   {'gpu':>9}")
        print("-" * 116)
        for op in present:
            cells = {}
            for h, c in (("rust", "t1"), ("cpp", "t1"), ("rust", "tN"), ("cpp", "tN"), ("rust", "gpu")):
                d = rows.get((op, size, h, c))
                cells[(h, c)] = d

            # Checksum equality gate — the basis of the whole comparison.
            sums = {}
            for h in ("rust", "cpp"):
                for c in ("t1", "tN"):
                    d = cells.get((h, c))
                    if d and not d.get("skipped") and d.get("input_checksum"):
                        sums.setdefault(h, set()).add(str(d["input_checksum"]).lower())
            if "rust" in sums and "cpp" in sums and sums["rust"] != sums["cpp"]:
                checksum_conflicts.append((op, size, sums["rust"], sums["cpp"]))
                print(f"{op:<38} !! INPUT CHECKSUM MISMATCH — comparison VOID for this op")
                continue

            def fmt(d):
                if d is None:
                    return "     —    "
                if d.get("skipped"):
                    return "   skip   "
                return f"{d['ms_median']:10.1f}"

            def ratio(a, b):
                if a is None or b is None or a.get("skipped") or b.get("skipped"):
                    return "     — "
                if not b["ms_median"]:
                    return "     — "
                r = a["ms_median"] / b["ms_median"]
                return f"{r:6.2f}x"

            r1, c1 = cells[("rust", "t1")], cells[("cpp", "t1")]
            rn, cn = cells[("rust", "tN")], cells[("cpp", "tN")]
            g = cells[("rust", "gpu")]
            print(f"{op:<38} {fmt(r1)} {fmt(c1)} {ratio(r1, c1)}   "
                  f"{fmt(rn)} {fmt(cn)} {ratio(rn, cn)}   {fmt(g)}")

        # Surface every skip reason rather than letting "skip" stand alone.
        notes = []
        for op in present:
            for h, c in (("rust", "t1"), ("rust", "tN"), ("cpp", "t1"), ("cpp", "tN"), ("rust", "gpu")):
                d = rows.get((op, size, h, c))
                if d and d.get("skipped"):
                    notes.append(f"  {op} [{h} {c}]: {d['skipped']}")
        if notes:
            print("\nskipped:")
            for n in notes:
                print(n)

    print("\nratio = rust / cpp. > 1.00x means the PORT IS SLOWER than ITK.")
    if checksum_conflicts:
        print("\n!! CHECKSUM CONFLICTS — these ops did not receive the same input in both")
        print("   harnesses, so their numbers are not comparable:")
        for op, size, r, c in checksum_conflicts:
            print(f"   {op} [{size}]: rust={sorted(r)} cpp={sorted(c)}")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
