#!/usr/bin/env python3
"""Merge vkbw CSV files into markdown tables.

Files named LABEL-runN.csv are grouped by LABEL (one label per chip or machine). Each cell shows the median of the per-run
medians and, in brackets, the lowest and highest per-run median. With exactly two labels a ratio column is added.
Usage: summarize.py results/*.csv
"""
import csv
import re
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit(__doc__)

groups = defaultdict(list)  # label -> list of csv paths
for path in sys.argv[1:]:
    groups[re.sub(r"-run\d+\.csv$", "", path.rsplit("/", 1)[-1])].append(path)
labels = sorted(groups)

cells = defaultdict(lambda: defaultdict(list))  # key -> label -> [gbs per run]
bad = []
for label in labels:
    for path in groups[label]:
        with open(path) as f:
            rows = [r for r in f if not r.startswith("#")]
        for r in csv.DictReader(rows):
            key = (r["suite"], r["op"], int(r["width_bytes"]), int(r["wg"]), r["subgroup"], int(r["size_mib"]), int(r["stride_bytes"]))
            cells[key][label].append(float(r["gbs_median"]))
            if r["check"] != "ok":
                bad.append((label, key))

def fmt(vals):
    if not vals:
        return "-"
    return f"{statistics.median(vals):.1f} [{min(vals):.1f}-{max(vals):.1f}]"

def table(title, keys, head, cols):
    if not keys:
        return
    print(f"\n### {title}\n")
    print("| " + " | ".join(head + labels + (["ratio"] if len(labels) == 2 else [])) + " |")
    print("|" + "---|" * (len(head) + len(labels) + (1 if len(labels) == 2 else 0)))
    for k in keys:
        row = [str(c) for c in cols(k)] + [fmt(cells[k].get(l, [])) for l in labels]
        if len(labels) == 2:
            a, b = (statistics.median(cells[k][l]) if cells[k].get(l) else None for l in labels)
            row.append(f"{b / a:.2f}" if a and b else "-")
        print("| " + " | ".join(row) + " |")

print("GB/s = 1e9 bytes per second of useful bytes (read, plus written for copy). Cell: median of run medians [min-max of run medians].")
for suite, op in (("A", "read"), ("A", "copy")):
    ks = sorted((k for k in cells if k[0] == suite and k[1] == op), key=lambda k: (k[5], k[2], k[6]))
    table(f"Suite A {op}: width x size x lane stride (workgroup 128, default subgroup)", ks,
          ["size MiB", "load B", "stride B"], lambda k: (k[5], k[2], k[6]))
ks = sorted((k for k in cells if k[0] == "B"), key=lambda k: (k[2], k[3], k[4] != "default", int(k[4]) if k[4] != "default" else 0))
table("Suite B read: workgroup and subgroup size (coalesced, largest size)", ks, ["load B", "workgroup", "subgroup"], lambda k: (k[2], k[3], k[4]))
if bad:
    print(f"\nCHECK FAILED in {len(bad)} cases, for example {bad[0]}. Do not trust those rows.")
    sys.exit(1)
