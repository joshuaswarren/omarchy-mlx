#!/usr/bin/env python3
"""Per-token GPU busy/idle from MLX_OMARCHY_GPU_PROFILE buffer mode (DISPATCH=0).

busy = union of command-buffer [t0, t1] GPU intervals whose host begin time falls in the
steady decode window (from the first to the last delivered token); idle = GPU span - busy.
usage: busy_split.py <profile.ndjson> <windows.jsonl>
"""
import json, sys
prof, win = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(win)][1:]
t1s = [r["t1"] for r in rows]
lo, hi, n = t1s[0], t1s[-1], len(t1s) - 1
period, bufs, end = 1.0, [], {}
for line in open(prof):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("k") == "meta":
        period = r["period_ns"]
    elif r.get("k") == "b" and "t0" in r and lo <= r["t"] < hi:
        bufs.append((r["t0"], r["t1"]))
    elif r.get("k") == "end":
        end = r
bufs.sort()
busy, cur0, cur1 = 0, None, None
for a, b in bufs:
    if cur1 is None or a > cur1:
        if cur1 is not None:
            busy += cur1 - cur0
        cur0, cur1 = a, b
    else:
        cur1 = max(cur1, b)
if cur1 is not None:
    busy += cur1 - cur0
span = (bufs[-1][1] - bufs[0][0]) if bufs else 0
wall_ms = (hi - lo) / 1e6 / n
busy_ms, span_ms = busy * period / 1e6 / n, span * period / 1e6 / n
print(f"tokens={n} buffers_per_token={len(bufs)/n:.2f} wall_ms={wall_ms:.3f} gpu_busy_ms={busy_ms:.3f} "
      f"gpu_idle_ms={span_ms - busy_ms:.3f} busy_frac={busy_ms/wall_ms:.3f} dispatches_total={end.get('dispatches_total')}")
