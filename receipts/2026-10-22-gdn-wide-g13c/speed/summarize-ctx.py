#!/usr/bin/env python3
"""Summarizes a m2-ctx.sh log: per prompt length, the median decode and prefill of each arm, the median of the per-round paired decode change of new against base, and the digests.
usage: sum-ctx.py LOG"""
import collections
import re
import statistics as S
import sys

rows = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    m = re.search(r"CTX9? ctx (\d+) round (\d+) (\w+) rc=(\d+) prompt (\d+) decode ([\d.]+) prefill ([\d.]+) ttft ([\d.]+) sha (\w+)", line)
    if m:
        rows[(int(m[1]), int(m[5]))][(m[3], int(m[2]))] = (float(m[6]), float(m[7]), m[9], int(m[4]))
for (nominal, real), v in sorted(rows.items()):
    rounds = sorted({r for (_, r) in v})
    pair = [100 * (v[("new", r)][0] / v[("base", r)][0] - 1) for r in rounds if ("new", r) in v and ("base", r) in v]
    ppair = [100 * (v[("new", r)][1] / v[("base", r)][1] - 1) for r in rounds if ("new", r) in v and ("base", r) in v]
    print(f"nominal {nominal} real prompt {real}:")
    for arm in ("base", "new"):
        d = [x[0] for (a, _), x in v.items() if a == arm]
        p = [x[1] for (a, _), x in v.items() if a == arm]
        sh = collections.Counter(x[2] for (a, _), x in v.items() if a == arm)
        rc = {x[3] for (a, _), x in v.items() if a == arm}
        print(f"  {arm}: decode median {S.median(d):.2f} tok/s (min {min(d):.2f} max {max(d):.2f}), prefill median {S.median(p):.1f}, digests {dict(sh)}, rc {rc}")
    print(f"  new against base per round decode {[round(x, 2) for x in pair]} median {S.median(pair):+.2f}%; prefill median {S.median(ppair):+.2f}%")
