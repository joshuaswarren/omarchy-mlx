"""Summarize the rope_bits.py timing lines in ab.py JSONL logs.

summarize.py skips rope_bits.py records, because the probe prints its
timings as text ("<dtype> freqs=<bool> q+k pair <x> us"), not as a JSON
result. This reads those lines from each record's stdout_tail. For each form
it prints every process's value in run order with its arm, then per arm the
mean and the min to max, then B/A of the means. Arm A is the first arm in
the log. A record with rc != 0 stops the summary.

usage: rope_summary.py <log.jsonl>...
"""
import json
import re
import sys
from collections import OrderedDict

LINE = re.compile(r"^(\S+) freqs=(True|False)\s+q\+k pair\s+([0-9.]+) us$")

forms = OrderedDict()
arms = []
for path in sys.argv[1:]:
    for n, line in enumerate(open(path), 1):
        r = json.loads(line)
        if r["rc"] != 0:
            sys.exit(f"{path}:{n}: rc={r['rc']} for arm {r['arm']}; not summarized")
        if r["arm"] not in arms:
            arms.append(r["arm"])
        seen = 0
        for out in r["stdout_tail"]:
            m = LINE.match(out.strip())
            if m:
                dtype = m.group(1).split(".")[-1]
                form = f"{dtype} {'freqs' if m.group(2) == 'True' else 'base'}"
                forms.setdefault(form, []).append((r["arm"], r["stamp"], float(m.group(3))))
                seen += 1
        if seen == 0:
            sys.exit(f"{path}:{n}: no rope_bits.py timing line in stdout_tail")

if len(arms) != 2:
    sys.exit(f"expected two arms, found {arms}")
a, b = arms
print(f"A = {a}, B = {b}; host wall time per decode q+k pair (us), 200 pairs in one eval")
for form, rows in forms.items():
    print(f"## {form}")
    print("  per process, run order: " + ", ".join(f"{arm} {v:.1f}" for arm, _, v in rows))
    means = {}
    for arm in (a, b):
        vals = [v for x, _, v in rows if x == arm]
        means[arm] = sum(vals) / len(vals)
        print(f"  {arm}: n={len(vals)} mean {means[arm]:.1f} (min {min(vals):.1f}, max {max(vals):.1f})")
    print(f"  B/A {means[b] / means[a]:.3f}")
