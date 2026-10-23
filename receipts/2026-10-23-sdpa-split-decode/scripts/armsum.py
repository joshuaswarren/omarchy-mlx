"""Per-round paired summary of an ARMS decode log read from stdin. usage: armsum.py [REFERENCE_ARM]  (default base)."""
import re
import statistics as st
import sys

d = {}
for line in sys.stdin:
    m = re.match(r"ARMS round (\d+) (\w+) rc=(\d+) decode ([\d.]+).*sha (\w+) load1=([\d.]+)", line)
    if m:
        d.setdefault(m[2], []).append((int(m[1]), float(m[4]), m[5], float(m[6]), int(m[3])))
ref = sys.argv[1] if len(sys.argv) > 1 else "base"
base = {r: v for r, v, *_ in d[ref]}
for arm, rows in d.items():
    pct = [(v / base[r] - 1) * 100 for r, v, *_ in rows]
    print(arm, [v for _, v, *_ in rows], "sha", sorted({x[2] for x in rows}), "rc", sorted({x[4] for x in rows}),
          "maxload", max(x[3] for x in rows), "median tok/s", st.median(v for _, v, *_ in rows),
          f"vs {ref} % per round", [round(p, 2) for p in pct], "median", round(st.median(pct), 2))
