"""Compare two shape_sweep.py outputs cell by cell. Usage: shape_sweep_cmp.py main.jsonl cand.jsonl"""
import json
import sys


def load(p):
    with open(p) as f:
        return {r["id"]: r["sha"] for r in map(json.loads, f) if r["k"] == "cell"}


m, c = load(sys.argv[1]), load(sys.argv[2])
bad = sorted(k for k in m.keys() & c.keys() if m[k] != c[k])
print(f"cells main {len(m)} cand {len(c)} common {len(m.keys() & c.keys())} mismatches {len(bad)}")
for k in bad[:20]:
    print("DIFF", k, m[k], c[k])
sys.exit(1 if bad or len(m) != len(c) else 0)
