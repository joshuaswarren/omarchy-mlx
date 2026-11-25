#!/usr/bin/env python3
"""PAR-99 table v2: per-kernel GPU time + achieved bandwidth using the
Honeykrisp dt1 attribution (w88-corrected): dt1 = t1[i] - t1[i-1] within the
same submit serial s - each dispatch's share of its submit's GPU interval,
not the raw tick pair (whose pairing double-counts the first and drops the
last dispatch of every submit).

Same inputs as v1: prof NDJSON, markers NDJSON, compute.h.
"""
import json
import re
import sys

prof_path, marker_path, compute_h = sys.argv[1], sys.argv[2], sys.argv[3]

names = []
inside = False
for line in open(compute_h):
    if "enum class ComputeKernel" in line:
        inside = True
        continue
    if inside:
        m = re.match(r"^\s{2}(\w+),\s*(?://.*)?$", line)
        if m:
            names.append(m.group(1))
        elif "};" in line:
            break

markers = [json.loads(l) for l in open(marker_path) if l.strip()]
ds = [m["t"] for m in markers if m.get("p") == "decode_start"]
dd = [m["t"] for m in markers if m.get("p") == "decode_done"]
if not ds or not dd:
    print("NO_MARKER decode", [m.get("p") for m in markers])
    sys.exit(3)
lo, hi = ds[0], dd[0]

# per submit serial: ordered dispatch list
subs = {}
order = []
for line in open(prof_path, errors="ignore"):
    if '"k":"d"' not in line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    s = r.get("s")
    q0 = None
    subs.setdefault(s, []).append(r)
    if s is not None and s not in [o[0] for o in order]:
        order.append((s, True))

queue = {}
for line in open(prof_path, errors="ignore"):
    if '"k":"q"' in line:
        try:
            r = json.loads(line)
        except Exception:
            continue
        queue[r["s"]] = r.get("queue_t0")

agg = {}
total = 0
nd = 0
for s, dispatches in subs.items():
    q0 = queue.get(s)
    if q0 is None or not (lo < q0 <= hi):
        continue
    # dt1: each dispatch's interval ends at its own t1 and starts at the
    # previous dispatch's t1 within the same submit (Honeykrisp timestamps
    # are written by the same submit's tail)
    prev_t1 = None
    for r in dispatches:
        t1 = r.get("t1")
        if t1 is None:
            continue
        ns = (t1 - prev_t1) if prev_t1 is not None else 0
        prev_t1 = t1
        key = (names[r["e"]] if r["e"] < len(names) else f"kernel_{r['e']}", r.get("op", "?"))
        a = agg.setdefault(key, [0, 0])
        a[0] += ns
        a[1] += 1
        total += ns
        nd += 1

print("# PAR-99 v2 (dt1 attribution, w88-corrected): 4B decode window, jw16 T6001 G14X")
print("# submits=%d dispatches=%d total_kernel_ns=%d" % (len([s for s in subs if queue.get(s) and lo < queue[s] <= hi]), nd, total))
print("# kernel\tcount\ttotal_us\tshare_pct")
for (k, op), (ns, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
    share = 100.0 * ns / total if total else 0.0
    print("%s\t%s\t%d\t%d\t%.2f" % (k, op, c, ns, share))
