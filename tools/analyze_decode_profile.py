#!/usr/bin/env python3
"""Per-token decode profile analyzer.

Inputs:
  --profile  MLX_OMARCHY_GPU_PROFILE NDJSON (events: meta/b/d/s/q/j/end)
  --windows  driver jsonl (meta line + per-token t0/t1)
  --compute-h  overlay/mlx/backend/omarchy/compute.h (kernel enum names)

Method:
  - Windows: token i window = [t1[i-1], t1[i]] (steady state: everything
    the host+GPU did to produce token i, from the moment token i-1 was
    delivered). Token 0 uses [meta.start, t1[0]] and is reported separately.
  - d-records carry no host clock, only the submission id "s". Map each
    submission id to a window via its "s" event host time; dispatches
    inherit that window. Submissions whose s-time falls before the first
    window are warmup/prefill and are excluded.
  - Kernel GPU time = (t1-t0) GPU ticks * meta.period_ns. Profiler build
    inserts barriers per dispatch, inflating absolute times ~10-30%; treat
    kernel times as RELATIVE shares, take wall/dispatch/submit/join counts
    as exact.
"""
import argparse
import json
import re
import statistics


def kernel_names(path):
    names = []
    for line in open(path):
        m = re.match(r"\s+(\w+)\s*,\s*$", line)
        if m and not line.strip().startswith("//"):
            names.append(m.group(1))
        else:
            m2 = re.match(r"\s+(\w+)\s*=\s*\d+", line)
            if m2:
                names.append(m2.group(1))
    return names


ap = argparse.ArgumentParser()
ap.add_argument("--profile", required=True)
ap.add_argument("--windows", required=True)
ap.add_argument("--compute-h", required=True)
ap.add_argument("--out", default="")
a = ap.parse_args()

names = kernel_names(a.compute_h)

meta = None
wins = []
for line in open(a.windows):
    r = json.loads(line)
    if "prompt" in r:
        meta = r
    else:
        wins.append(r)

start = meta["start_monotonic"]
edges = [start] + [w["t1"] for w in wins]
period_ns = 1.0
valid_bits = 64

# Pass 1: read all events; keep s/j/b with host times, collect d by sub.
subs = {}          # sub -> {"t": s-event host time, "d": [drec]}
joins = []
begins = []
n_dropped = 0
endrec = None
for line in open(a.profile):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    k = r.get("k")
    if k == "meta":
        period_ns = r["period_ns"]
        valid_bits = r["valid_bits"]
    elif k == "s":
        subs.setdefault(r["s"], {"t": r["t"], "d": []})
    elif k == "d":
        subs.setdefault(r["s"], {"t": None, "d": []})["d"].append(r)
    elif k == "j":
        joins.append(r)
    elif k == "b":
        begins.append(r)
    elif k == "end":
        endrec = r

# Assign submissions to token windows by submit host time.
def win_of(t):
    if t is None:
        return -1
    for i in range(len(edges) - 1):
        if edges[i] <= t < edges[i + 1]:
            return i
    return -2  # after last token / before start handled by -1 sentinel order


rows = []
for i in range(len(wins)):
    rows.append({"wall_ns": edges[i + 1] - edges[i], "disp": 0, "sub": 0,
                 "join": 0, "gpu_ns": 0.0, "join_wait_ns": 0,
                 "submit_host_ns": 0, "kern": {}, "kc": {}})

unassigned = {"disp": 0, "gpu_ns": 0.0}
for sid, s in subs.items():
    w = win_of(s["t"])
    for d in s["d"]:
        if w < 0:
            unassigned["disp"] += 1
            continue
        r = rows[w]
        r["disp"] += 1
        t0, t1 = d.get("t0"), d.get("t1")
        gpu = 0.0
        if t0 is not None and t1 is not None and t1 >= t0:
            gpu = (t1 - t0) * period_ns
        r["gpu_ns"] += gpu
        kn = names[d["e"]] if d["e"] < len(names) else f"e{d['e']}"
        key = f"{kn}|n={d['n']}|op={d['op']}"
        r["kern"][key] = r["kern"].get(key, 0.0) + gpu
        r["kc"][key] = r["kc"].get(key, 0) + 1

for j in joins:
    w = win_of(j["t"])
    if w >= 0:
        rows[w]["join"] += 1
        rows[w]["join_wait_ns"] += j.get("wait", 0)

# Count submits per window directly from s events.
for sid, s in subs.items():
    w = win_of(s["t"])
    if w >= 0:
        rows[w]["sub"] += 1
        rows[w]["submit_host_ns"] += s.get("dur", 0)

steady = rows[4:] if len(rows) > 8 else rows[1:]
sumr = lambda k: sum(r[k] for r in steady)
med = lambda k: statistics.median([r[k] for r in steady])
wall_ms = sumr("wall_ns") / 1e6
gpu_ms = sumr("gpu_ns") / 1e6

# Top kernels aggregated over steady tokens.
agg = {}
aggc = {}
for r in steady:
    for k, v in r["kern"].items():
        agg[k] = agg.get(k, 0.0) + v
    for k, v in r["kc"].items():
        aggc[k] = aggc.get(k, 0) + v

out = {
    "period_ns": period_ns,
    "valid_bits": valid_bits,
    "tokens": len(wins),
    "steady_tokens": len(steady),
    "steady": {
        "wall_ms_total": round(wall_ms, 3),
        "wall_ms_median": round(med("wall_ns") / 1e6, 3),
        "tok_s_from_wall": round(1000.0 * (len(steady)) / wall_ms, 2) if wall_ms else 0,
        "gpu_ms_total": round(gpu_ms, 3),
        "gpu_ms_median": round(med("gpu_ns") / 1e6, 3),
        "gpu_busy_fraction": round(gpu_ms / wall_ms, 3) if wall_ms else 0,
        "dispatches_per_token_median": med("disp"),
        "submits_per_token_median": med("sub"),
        "joins_per_token_median": med("join"),
        "join_wait_ms_median": round(med("join_wait_ns") / 1e6, 3),
        "submit_host_ms_median": round(med("submit_host_ns") / 1e6, 3),
    },
    "token0": {"wall_ms": round(rows[0]["wall_ns"] / 1e6, 3),
               "disp": rows[0]["disp"], "sub": rows[0]["sub"]} if rows else None,
    "unassigned_dispatches": unassigned["disp"],
    "end": endrec,
    "top_kernels_by_gpu_ms": sorted(
        [(k, round(v / 1e6, 3), aggc[k]) for k, v in agg.items()], key=lambda x: -x[1]
    )[:20],
    "top_kernels_by_count": sorted(
        [(k, n, round(agg.get(k, 0.0) / 1e6, 3)) for k, n in aggc.items()], key=lambda x: -x[1]
    )[:20],
    "per_token": [
        {"i": i + 4, "wall_ms": round(r["wall_ns"] / 1e6, 3), "gpu_ms": round(r["gpu_ns"] / 1e6, 3),
         "disp": r["disp"], "sub": r["sub"], "join": r["join"]}
        for i, r in enumerate(rows)
    ],
}
js = json.dumps(out, indent=1)
if a.out:
    open(a.out, "w").write(js)
print(js)
