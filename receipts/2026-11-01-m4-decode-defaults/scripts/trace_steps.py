"""Per-decode-step dispatch table from an MLX_OMARCHY_GPU_PROFILE stream.

usage: trace_steps.py <profile.jsonl[.gz]> <compute.h>
Groups dispatch records by submission, keeps the decode submissions whose
records are all present (the most common dispatch count among submissions
after the first three), and prints, per kernel, dispatches per decode step
and GPU microseconds per step (t1 - t0, mean over those steps).
"""
import collections
import gzip
import json
import sys

sys.path.insert(0, ".")
from profile_analyze import parse_kernel_names  # noqa: E402

path, header = sys.argv[1], sys.argv[2]
names = parse_kernel_names(header)
op = gzip.open if path.endswith(".gz") else open
by_sub = collections.defaultdict(list)
end = None
with op(path, "rt") as f:
    for line in f:
        d = json.loads(line)
        if d.get("k") == "d":
            by_sub[d["s"]].append(d)
        elif d.get("k") == "end":
            end = d
counts = collections.Counter(len(v) for s, v in by_sub.items() if s > 3)
step_n = counts.most_common(1)[0][0]
steps = [v for s, v in by_sub.items() if s > 3 and len(v) == step_n]
per = collections.defaultdict(lambda: [0, 0.0])
for v in steps:
    for d in v:
        k = names[d["e"]] if d["e"] < len(names) else f"id{d['e']}"
        per[k][0] += 1
        per[k][1] += (d["t1"] - d["t0"]) / 1000.0
tot = sum(x[1] for x in per.values()) / len(steps)
print(f"end record: {json.dumps(end)}")
print(f"decode steps with all records: {len(steps)}; dispatches per step: {step_n}; "
      f"GPU busy per step: {tot / 1000:.2f} ms")
print(f"{'kernel':34s} {'per step':>9s} {'us/step':>9s} {'us/disp':>8s} {'share':>6s}")
for k, (n, us) in sorted(per.items(), key=lambda kv: -kv[1][1]):
    print(f"{k:34s} {n / len(steps):9.1f} {us / len(steps):9.1f} {us / n:8.1f} {us / len(steps) / tot:6.1%}")
