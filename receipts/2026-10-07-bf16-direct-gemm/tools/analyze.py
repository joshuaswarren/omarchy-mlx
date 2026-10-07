"""Throwaway: H13-H16 three-arm analysis for one chip dir (res-h1316/<chip>)."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])


def load(kind, arm, rep):
    rows = [json.loads(x) for x in (d / f"h1316-{kind}-{arm}-r{rep}.jsonl").read_text().splitlines() if x.strip()]
    if kind == "mb":
        return {f"{r['cell']}:{r.get('dtype', '').split('.')[-1]}": r.get("tflops", r.get("gb_s"))
                for r in rows if r["k"] == "time"}
    return {r["cell"]: r for r in rows if r["k"] == "cell"}


reps = (1, 2, 3)
print(f"== {d.name}: rowcheck hashes (C1, C2 vs M)")
for rep in reps:
    m, c1, c2 = (load("rc", a, rep) for a in ("M", "C1", "C2"))
    bad = [(a, k) for a, c in (("C1", c1), ("C2", c2)) for k in m if c[k]["sha"] != m[k]["sha"]]
    print(f"r{rep}: {len(m)} cells, mismatches {bad or 'none'}")

print("== metal_baseline TFLOP/s, % vs M per rep (C1 | C2)")
mb = {(a, r): load("mb", a, r) for a in ("M", "C1", "C2") for r in reps}
for cell in mb[("M", 1)]:
    pc = lambda a, cell=cell: " ".join(f"{100 * (mb[(a, r)][cell] / mb[('M', r)][cell] - 1):+.1f}" for r in reps)
    ms = " ".join(f"{mb[('M', r)][cell]:.3f}" for r in reps)
    print(f"{cell:24s} M {ms} | C1 {pc('C1')} | C2 {pc('C2')}")

print("== rowcheck median_ms: C1 vs M, C2 vs C1 (% speedup per rep)")
rc = {(a, r): load("rc", a, r) for a in ("M", "C1", "C2") for r in reps}
for cell in rc[("M", 1)]:
    sp = lambda a, b, cell=cell: " ".join(f"{100 * (rc[(b, r)][cell]['median_ms'] / rc[(a, r)][cell]['median_ms'] - 1):+.1f}"
                               for r in reps)
    ms = " ".join(f"{rc[('C1', r)][cell]['median_ms']:.2f}" for r in reps)
    print(f"{cell:10s} C1ms {ms} | C1/M {sp('C1', 'M')} | C2/C1 {sp('C2', 'C1')}")
