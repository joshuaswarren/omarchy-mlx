#!/usr/bin/env python3
"""Parse DRM scheduler ftrace files (gpu_scheduler:drm_sched_job_{queue,run,done})
for the target python PID only, computing both busy = done - max(run, prev_done)
(PR #454 / G16S formula) and raw run->done duration (PR #575 / G13G formula),
split into all / prefill / decode phases. Never emits other processes.

Usage: parse_ftrace.py <drv_file> <trace_file>
"""
import json
import re
import sys

drv_path, trace_path = sys.argv[1:3]
drv = open(drv_path).read()
m_pid = re.search(r"^PID (\d+)", drv, re.M)
m_win = re.search(r"^WINDOW (\S+) (\S+)", drv, re.M)
m_ft = re.search(r"^FIRSTTOK (\S+) gen_tps=(\S+)", drv, re.M)
m_dec = re.search(r"^DECODE (\d+) (\d+) (\d+) ids_sha256=(\S+)", drv, re.M)
m_prov = re.search(r"^(provenance: .+)$", drv, re.M)

target_pid = m_pid.group(1) if m_pid else None
t0, t1 = map(float, m_win.groups())
tfirst = float(m_ft.group(1))
gen_tps = float(m_ft.group(2))
n_tok = int(m_dec.group(3))
ids_sha = m_dec.group(4)

rx = re.compile(
    r"^\s*(\S+)-(\d+)\s+\[\d+\].*?\s(\d+\.\d+): drm_sched_job_(queue|run|done): (.*)$"
)

jobs = {}
with open(trace_path, "r", errors="replace") as f:
    for line in f:
        r = rx.match(line)
        if not r:
            continue
        comm, pid, ts_s, ev, rest = r.groups()
        ts = float(ts_s)
        # Match fence or id (kernels vary slightly: fence= or id=)
        fm = re.search(r"fence=(\S+)", rest) or re.search(r"id=(\S+)", rest)
        if not fm:
            continue
        key = fm.group(1).rstrip(",")
        j = jobs.setdefault(key, {})
        j[ev] = ts
        if ev == "queue":
            j["comm"] = comm
            j["pid"] = pid
            rm = re.search(r"ring=([^,\s]+)", rest)
            if rm:
                j["ring"] = rm.group(1)

# Filter to jobs queued by our python process (or comm starting with python if pid absent)
# Note: drm_sched_job_queue runs in the submitting thread, so its pid may be a worker thread
# inside the python process; match comm starting with "python" (or thread comm inside python).
# Check how comms look in the window first.
all_in_win = [
    j for j in jobs.values()
    if {"queue", "run", "done", "comm"} <= j.keys() and t0 <= j["queue"] <= t1
]
py = sorted(
    [j for j in all_in_win if j["comm"].startswith("python") or j.get("pid") == target_pid],
    key=lambda j: j["done"],
)
prev = 0.0
for j in py:
    j["busy_ms"] = (j["done"] - max(j["run"], prev)) * 1e3
    j["run_done_ms"] = (j["done"] - j["run"]) * 1e3
    j["queue_done_ms"] = (j["done"] - j["queue"]) * 1e3
    prev = j["done"]

def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))], 3)

def stats(sel, field):
    v = [j[field] for j in sel]
    if not v:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": len(v),
        "p50": pct(v, 50),
        "p95": pct(v, 95),
        "p99": pct(v, 99),
        "max": round(max(v), 3),
    }

prefill = [j for j in py if j["queue"] < tfirst]
decode = [j for j in py if j["queue"] >= tfirst]

out = {
    "provenance": m_prov.group(1) if m_prov else None,
    "tokens": n_tok,
    "gen_tps": gen_tps,
    "ids_sha256": ids_sha,
    "all_jobs_in_window": len(all_in_win),
    "python_jobs_in_window": len(py),
    "phases": {
        "all": {
            "busy_ms": stats(py, "busy_ms"),
            "run_done_ms": stats(py, "run_done_ms"),
            "queue_done_ms": stats(py, "queue_done_ms"),
        },
        "prefill": {
            "busy_ms": stats(prefill, "busy_ms"),
            "run_done_ms": stats(prefill, "run_done_ms"),
            "queue_done_ms": stats(prefill, "queue_done_ms"),
        },
        "decode": {
            "busy_ms": stats(decode, "busy_ms"),
            "run_done_ms": stats(decode, "run_done_ms"),
            "queue_done_ms": stats(decode, "queue_done_ms"),
        },
    },
}
print(json.dumps(out, indent=2))
