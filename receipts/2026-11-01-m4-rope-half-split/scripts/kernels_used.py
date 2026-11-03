"""Kernel names and dispatch counts in an MLX_OMARCHY_GPU_PROFILE stream.
usage: kernels_used.py <profile.jsonl[.gz]> <compute.h> [name-substring ...]"""
import collections
import gzip
import json
import sys

sys.path.insert(0, ".")
from profile_analyze import parse_kernel_names  # noqa: E402

names = parse_kernel_names(sys.argv[2])
want = sys.argv[3:]
op = gzip.open if sys.argv[1].endswith(".gz") else open
c = collections.Counter()
with op(sys.argv[1], "rt") as f:
    for line in f:
        d = json.loads(line)
        if d.get("k") == "d":
            c[names[d["e"]] if d["e"] < len(names) else f"id{d['e']}"] += 1
        elif d.get("k") == "end":
            print("end:", json.dumps(d))
for k, n in sorted(c.items(), key=lambda kv: -kv[1]):
    if not want or any(w in k for w in want):
        print(f"{n:7d} {k}")
