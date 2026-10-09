"""Summarise a cf_ticket.sh select/gate log. usage: cf_select_an.py <log> (the lines 'arm L=...: {json}')."""
import json
import re
import statistics
import sys

rows = {}
for line in open(sys.argv[1]):
    m = re.match(r"(composed|rowp\d|flash-[a-z0-9]+) L=\S+: (\{.*)", line)
    if not m:
        print(line.rstrip()[:160])
        continue
    d = json.loads(m.group(2))
    for L, v in d["L"].items():
        rows.setdefault((int(L), m.group(1)), []).append((v["ms_median"], v["max_err_head0"], v["mean_err_head0"]))
for L in sorted({k[0] for k in rows}):
    base = statistics.median([x[0] for x in rows[(L, "composed")]])
    for arm in sorted({k[1] for k in rows if k[0] == L}, key=lambda a: (a != "composed", a)):
        v = rows[(L, arm)]
        ms = [x[0] for x in v]
        print(f"L={L} {arm:12s} reps {len(ms)} ms {ms} median {statistics.median(ms):.3f} ratio-to-composed "
              f"{statistics.median(ms) / base:.3f} spread {100 * (max(ms) / min(ms) - 1):.1f}% "
              f"maxerr {v[0][1]:.2e} meanerr {v[0][2]:.2e}")
