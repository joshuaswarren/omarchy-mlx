"""Wall-time bench: N invocations with or without the daemon.
Reports per-invocation wall (client process start to JSON out) so
the L3 saving is visible end-to-end. Per-invocation transcribe stdout
is captured to <out>/stdout.log so failures can be diagnosed."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VENV_PY = "/var/tmp/pkwarm-venv/bin/python3.14"
CLI = "/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-parakeet"
OUT = Path("/tmp/pkwarm-h000")
SOCK = "/tmp/pkwarm-h000/ane.sock"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ARM = sys.argv[2] if len(sys.argv) > 2 else "daemon"


def run_one(label, env, dest):
    dest.mkdir(parents=True, exist_ok=True)
    stale = dest / "transcribe-report.json"
    stale.unlink(missing_ok=True)
    full_env = os.environ.copy()
    full_env.update(env)
    start = time.monotonic()
    rc = subprocess.run(
        [VENV_PY, CLI, "transcribe", "-o", str(dest)],
        env=full_env,
        capture_output=True,
        timeout=60,
    )
    wall_ms = (time.monotonic() - start) * 1000
    log_path = dest / "stdout.log"
    log_path.write_bytes(rc.stdout or b"")
    err_path = dest / "stderr.log"
    err_path.write_bytes(rc.stderr or b"")
    rp = dest / "transcribe-report.json"
    if not rp.is_file():
        return {
            "label": label, "rc": rc.returncode, "wall_ms": wall_ms,
            "error": "no report",
        }
    r = json.loads(rp.read_text())
    return {
        "label": label,
        "rc": rc.returncode,
        "wall_ms": wall_ms,
        "tsha": hashlib.sha256(r["transcript"].encode()).hexdigest()[:16],
        "status": r["status"],
        "transport": r["ane"]["session"].get("transport"),
        "session_open_ms": r["ane"]["session"]["open_ms"],
        "session_reused": r["ane"]["session"]["reused"],
        "ane_exec_ms": r["ane"]["exec_ms"],
        "pipeline_ms": r["timing"]["total_pipeline_ms"],
        "stages": {s["stage"]: s["wall_ms"] for s in r["stages"]},
        "checks_failed": [
            c["check"] for c in r["verification"]["checks"] if not c["pass"]
        ],
        "worker_starts": r["ane"]["worker_starts"],
        "cpu_events": r["execution"]["cpu_tensor_events"],
        "emissions": r["execution"]["chain_slots_used"],
    }


out_base = OUT / ("wall-" + ARM)
out_base.mkdir(parents=True, exist_ok=True)
summary = []
for i in range(1, N + 1):
    if ARM == "daemon":
        env = {
            "MLX_OMARCHY_PK_KEEP_WORKER": "1",
            "MLX_OMARCHY_ANE_SOCK": SOCK,
        }
    else:
        env = {}
    res = run_one(f"{ARM}-{i}", env, out_base / f"r{i}")
    summary.append(res)
    print(json.dumps(res))
    sys.stdout.flush()

(out_base / "summary.json").write_text(json.dumps(summary, indent=2))