"""Interleaved A/B of `mlx-omarchy-parakeet transcribe` per-call wall:
daemon-attached (MLX_OMARCHY_PK_KEEP_WORKER=1) vs private worker, ABAB
order on one boot. Each call is a fresh client process; wall is measured
from process launch to exit. load1 and PSI cpu avg10 are recorded
before every call. Writes ab-summary.json beside the per-call outputs.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

VENV_PY = "/var/tmp/pkwarm-venv/bin/python3.14"
CLI = "/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-parakeet"
OUT = Path("/tmp/pkwarm-h000/ab")
SOCK = "/tmp/pkwarm-h000/ane.sock"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def idle_state():
    load1 = float(Path("/proc/loadavg").read_text().split()[0])
    some = Path("/proc/pressure/cpu").read_text().splitlines()[0]
    avg10 = float(some.split()[1].split("=")[1])
    return load1, avg10


def run(arm, i):
    dest = OUT / f"{arm}-r{i}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "transcribe-report.json").unlink(missing_ok=True)
    env = os.environ.copy()
    env.pop("MLX_OMARCHY_PK_KEEP_WORKER", None)
    env.pop("MLX_OMARCHY_ANE_SOCK", None)
    if arm == "daemon":
        env["MLX_OMARCHY_PK_KEEP_WORKER"] = "1"
        env["MLX_OMARCHY_ANE_SOCK"] = SOCK
    load1, psi = idle_state()
    start = time.monotonic()
    proc = subprocess.run([VENV_PY, CLI, "transcribe", "-o", str(dest)],
                          env=env, capture_output=True, timeout=120)
    wall_ms = (time.monotonic() - start) * 1000
    (dest / "stderr.log").write_bytes(proc.stderr or b"")
    row = {"arm": arm, "i": i, "rc": proc.returncode,
           "wall_ms": round(wall_ms, 1), "load1": load1, "psi_avg10": psi}
    rp = dest / "transcribe-report.json"
    if rp.is_file():
        r = json.loads(rp.read_text())
        row.update({
            "tsha": hashlib.sha256(r["transcript"].encode()).hexdigest(),
            "status": r["status"],
            "checks_failed": [c["check"] for c in r["verification"]["checks"]
                              if not c["pass"]],
            "emissions": len(json.loads(
                (dest / "token_ids.json").read_text())["token_ids"]),
            "transport": r["ane"]["session"].get("transport"),
            "session_open_ms": r["ane"]["session"]["open_ms"],
            "ane_exec_ms": r["ane"]["exec_ms"],
            "pipeline_ms": r["timing"]["total_pipeline_ms"],
            "cpu_events": r["execution"]["cpu_tensor_events"],
            "stages": {s["stage"]: s["wall_ms"] for s in r["stages"]},
        })
    return row


rows = []
for i in range(1, N + 1):
    for arm in ("daemon", "private"):
        row = run(arm, i)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in (
            "arm", "i", "rc", "wall_ms", "transport", "session_open_ms",
            "pipeline_ms", "status", "load1", "psi_avg10")}))
        sys.stdout.flush()


def summarize(arm):
    good = [r for r in rows if r["arm"] == arm and r.get("status") == "match"]
    def stat(values):
        values = sorted(values)
        p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
        return {"median": round(statistics.median(values), 1),
                "p95": round(p95, 1), "n": len(values)}
    out = {"wall_ms": stat([r["wall_ms"] for r in good]),
           "pipeline_ms": stat([r["pipeline_ms"] for r in good]),
           "session_open_ms": stat([r["session_open_ms"] for r in good]),
           "ane_exec_ms": stat([r["ane_exec_ms"] for r in good])}
    for stage in good[0]["stages"] if good else []:
        out[stage] = stat([r["stages"][stage] for r in good])
    out["transports"] = sorted({r.get("transport") for r in good})
    out["tshas"] = sorted({r["tsha"] for r in good})
    out["emissions"] = sorted({r["emissions"] for r in good})
    out["checks_failed"] = sorted({c for r in good for c in r["checks_failed"]})
    out["cpu_events"] = sorted({r["cpu_events"] for r in good})
    out["failures"] = [r for r in rows if r["arm"] == arm
                       and r.get("status") != "match"]
    return out


summary = {"daemon": summarize("daemon"), "private": summarize("private"),
           "rows": rows}
(OUT / "ab-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "failures"}
                  for k, v in summary.items() if k != "rows"}, indent=1))
