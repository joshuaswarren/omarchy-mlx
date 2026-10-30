"""Print a run-length encoded slice of the dispatch sequence of a MLX_OMARCHY_GPU_PROFILE ndjson, plus per-(kernel, n) counts for the whole file.
  seq-slice.py PROFILE COMPUTE_H START COUNT
d records carry e (kernel enum), n (element count), op (operation code) and s (submission id); file order is dispatch order within a submission."""
import json
import re
import sys
from collections import Counter


def names(path):
    src = open(path).read()
    i = src.index("enum class ComputeKernel")
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", src[j + 1:k], flags=re.S))
    out, val = {}, -1
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"(\w+)\s*(?:=\s*(\d+))?$", item)
        val = int(m.group(2)) if m.group(2) else val + 1
        out[val] = m.group(1)
    return out


nm = names(sys.argv[2])
start, count = int(sys.argv[3]), int(sys.argv[4])
seq = []
tot = Counter()
with open(sys.argv[1]) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("k") == "d":
            key = (nm.get(r["e"], f"e{r['e']}"), r.get("n"), r.get("op"))
            tot[key] += 1
            seq.append(key)
print("dispatches", len(seq))
print("top (kernel, n, op) over the whole file:")
for key, c in tot.most_common(14):
    print(f"  {c:8d}  {key}")
print(f"slice {start}..{start + count} run-length:")
prev, run = None, 0
for key in seq[start:start + count] + [None]:
    if key == prev:
        run += 1
        continue
    if prev is not None:
        print(f"  x{run:<4d} {prev}")
    prev, run = key, 1
