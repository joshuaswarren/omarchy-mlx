#!/usr/bin/env python3
"""DispatchFuse: gated-vs-baseline dispatch deltas, computed on-host."""
import sys
from collections import Counter

sys.path.insert(0, "/var/tmp/dfuse")
from analyze_w1 import kernel_names, load  # noqa: E402

d = sys.argv[1]
names = kernel_names("/var/tmp/dfuse-build/overlay/mlx/backend/omarchy/compute.h")


def cls(ev):
    c = Counter()
    for r in ev:
        kn = names[r["e"]] if r["e"] < len(names) else f"enum{r['e']}"
        c[(r["p"], kn)] += 1
    return c


for model, base, cand in (("4b", "prof-4b-d64.ndjson", "prof-4b-d64-rn2.ndjson"),
                          ("9b", "prof-9b-d64.ndjson", "prof-9b-d64-gdn2.ndjson")):
    b, c = load(f"{d}/{base}"), load(f"{d}/{cand}")
    delta = len(c) - len(b)
    print(f"== {model}: baseline {len(b)} | gated {len(c)} | delta {delta:+d} "
          f"({delta/64:+.2f}/decode-token)")
    cb, cc = cls(b), cls(c)
    diffs = {}
    for k in set(cb) | set(cc):
        v = cc[k] - cb[k]
        if v:
            diffs[k] = v
    for k, v in sorted(diffs.items(), key=lambda kv: kv[1]):
        print(f"   {v:+7d}  {k[0]:22s} {k[1]}")
