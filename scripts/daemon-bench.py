#!/usr/bin/env python3
"""Run 20 sequential transcribes against the daemon-attached ANE worker.
Writes one summary line per invocation; never raises so a single bad
transcribe does not abort the bench."""
import hashlib, json, os, sys, time
from pathlib import Path

VENV_PY = "/var/tmp/MesaParity/venv/bin/python3.14"
CLI = "/var/tmp/MesaParity/venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-parakeet"
SOCK = "/tmp/pkwarm-h000/ane.sock"
SUMMARY = "/tmp/pkwarm-h000/daemon-bench/summary.txt"
BENCH_DIR = Path("/tmp/pkwarm-h000/daemon-bench")
BENCH_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MLX_OMARCHY_PK_KEEP_WORKER"] = "1"
os.environ["MLX_OMARCHY_ANE_SOCK"] = SOCK

n = 20
results = []
for i in range(1, n + 1):
    out = BENCH_DIR / f"r{i}"
    out.mkdir(parents=True, exist_ok=True)
    wall_start = time.monotonic()
    rc = os.system(
        f"flock -w 600 /tmp/m1-gpu.lock {VENV_PY} {CLI} transcribe -o {out} "
        f"> {out}.log 2>&1")
    wall_ms = (time.monotonic() - wall_start) * 1000
    report_path = out / "transcribe-report.json"
    if not report_path.is_file():
        results.append({"i": i, "rc": rc, "error": "no report"})
        continue
    r = json.loads(report_path.read_text())
    tsha = hashlib.sha256(r["transcript"].encode()).hexdigest()[:16]
    results.append({
        "i": i,
        "rc": rc,
        "wall_ms": round(wall_ms, 1),
        "tsha": tsha,
        "ane_exec_ms": r["ane"]["exec_ms"],
        "session_open_ms": r["ane"]["session"]["open_ms"],
        "session_reused": r["ane"]["session"]["reused"],
        "worker_starts": r["ane"]["worker_starts"],
        "pipeline_ms": r["timing"]["total_pipeline_ms"],
        "cpu_tensor_events": r["execution"]["cpu_tensor_events"],
        "emissions": r["execution"]["chain_slots_used"],
        "status": r["status"],
        "checks_failed": r.get("verification", {}).get("checks_failed", []),
    })

with open(SUMMARY, "w") as fh:
    fh.write("i  rc wall_ms   tsha            open  exec  pipe  wker reused cpu  em status checks\n")
    for row in results:
        fh.write(
            f"{row.get('i',0):>2}  {row.get('rc',0):>3} "
            f"{row.get('wall_ms',0):>7.1f}  "
            f"{row.get('tsha','')}  "
            f"{row.get('session_open_ms',0):>4.0f}  "
            f"{row.get('ane_exec_ms',0):>4.0f}  "
            f"{row.get('pipeline_ms',0):>4.0f}  "
            f"{row.get('worker_starts',0):>4}  "
            f"{row.get('session_reused',False)!s:>5}  "
            f"{row.get('cpu_tensor_events',0):>3}  "
            f"{row.get('emissions',0):>3}  "
            f"{row.get('status','?')}  "
            f"{','.join(row.get('checks_failed',[])) or '-'}\n")
print(open(SUMMARY).read())