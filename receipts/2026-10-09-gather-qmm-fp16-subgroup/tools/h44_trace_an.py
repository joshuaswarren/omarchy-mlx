"""Per-arm kernel names of the gqmm_probe trace. usage: h44_trace_an.py <trace.err> <compute.h of the wheel's source commit>
Splits on '@@ARM <dtype> <arm>' ... '@@END', maps DISPATCH kernel ids to ComputeKernel names with the given enum header."""
import re
import sys
from collections import Counter

names, on = [], False
for line in open(sys.argv[2]):
    if "enum class ComputeKernel" in line:
        on = True
        continue
    if on:
        if line.strip().startswith("};"):
            break
        m = re.fullmatch(r"([A-Za-z0-9_]+),?", re.sub(r"//.*", "", line).strip())
        if m:
            names.append(m[1])
cur, arms = None, {}
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"@@ARM (\S+) (\S+)", line)
    if m:
        cur = (m[1], m[2])
        arms[cur] = Counter()
        continue
    if line.startswith("@@END"):
        cur = None
        continue
    if cur:
        k = re.search(r"DISPATCH kernel=(\d+) count=(\d+) gx=(\d+) gy=(\d+) gz=(\d+)", line)
        if k:
            arms[cur][(names[int(k[1])] if int(k[1]) < len(names) else int(k[1]), int(k[3]) * int(k[4]) * int(k[5]))] += 1
for arm, c in arms.items():
    print(arm[0], arm[1], "dispatches", sum(c.values()))
    for (name, wg), n in c.most_common(8):
        print(f"    {name} workgroups {wg} x{n}")
