"""Compare two shape_sweep.py outputs cell by cell. Usage: shape_sweep_cmp.py main.jsonl cand.jsonl"""
import json
import sys


def load(p):
    with open(p) as f:
        return {r["id"]: r["sha"] for r in map(json.loads, f) if r["k"] == "cell"}


m, c = load(sys.argv[1]), load(sys.argv[2])
missing, extra = sorted(m.keys() - c.keys()), sorted(c.keys() - m.keys())
bad = sorted(k for k in m.keys() & c.keys() if m[k] != c[k])
print(f"cells main {len(m)} cand {len(c)} common {len(m.keys() & c.keys())} mismatches {len(bad)} missing {len(missing)} extra {len(extra)}")
for k in bad[:20]:
    print("DIFF", k, m[k], c[k])
for k in missing[:20]:
    print("MISSING in cand", k)
for k in extra[:20]:
    print("EXTRA in cand", k)
sys.exit(1 if bad or missing or extra else 0)
