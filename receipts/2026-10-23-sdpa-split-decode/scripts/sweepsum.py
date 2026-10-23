"""Break-even table of an sdpa_bench3 log: python3 sweepsum.py LOG. Chain and independent us per call, default arm against SPLIT=1 arm."""
import re
import sys

arm = None
d = {}
for line in open(sys.argv[1]):
    m = re.match(r"SDPA-RUN arm (\w+)", line)
    if m:
        arm = m[1]
    m = re.match(r"SDPA kv (\d+): independent ([\d.]+) us .*chain ([\d.]+) us", line)
    if m:
        d.setdefault(arm, {})[int(m[1])] = (float(m[2]), float(m[3]))
print("kv  chain default  chain split  ratio   indep default  indep split  ratio")
for k in sorted(d["default"]):
    (di, dc), (si, sc) = d["default"][k], d["split"][k]
    print(f"{k:5d} {dc:9.1f} {sc:9.1f} {dc / sc:6.2f}   {di:9.1f} {si:9.1f} {di / si:6.2f}")
