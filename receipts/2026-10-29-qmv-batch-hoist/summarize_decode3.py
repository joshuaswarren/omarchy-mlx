"""Summarize a bgdecode3 log: median ms/step per (B, arm) over the triples, ratios vs A, ids hash agreement."""
import json
import re
import statistics as st
import sys

arm = None
B = None
runs = {}
for line in open(sys.argv[1]):
    m = re.match(r"-- B=(\d+) triple (\d+) arm (\w)", line)
    if m:
        B, arm = int(m.group(1)), m.group(3)
    elif line.startswith("BGDECODE"):
        d = json.loads(line[len("BGDECODE"):])
        runs.setdefault((B, arm), []).append((d["median_ms_per_step"], d["ids_sha"]))
for b in (2, 3, 4):
    ids = {i for a in "ABC" for _, i in runs.get((b, a), [])}
    med = {a: st.median(v for v, _ in runs.get((b, a), [(float("nan"), "")])) for a in "ABC"}
    mean = {a: st.mean(v for v, _ in runs.get((b, a), [(float("nan"), "")])) for a in "ABC"}
    n = {a: len(runs.get((b, a), [])) for a in "ABC"}
    print(f"B={b} runs {n} ids {sorted(ids)}")
    for a in "ABC":
        print(f"  {a}: median {med[a]:.2f} mean {mean[a]:.2f} ms/step  speedup vs A: median {med['A']/med[a]:.3f}x mean {mean['A']/mean[a]:.3f}x  all {[v for v,_ in runs.get((b,a),[])]}")
