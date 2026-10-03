#!/usr/bin/env python3
"""Quick 2-invocation test against the daemon, to validate the
session-alive-across-clients fix without paying for the full 20-pass
bench. Each invocation writes its own transcribe-report.json to
/tmp/pkwarm-h000/t-<i>/."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VENV_PY = "/var/tmp/pkwarm-venv/bin/python3.14"
CLI = "/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-parakeet"
SOCK = "/tmp/pkwarm-h000/ane.sock"
OUT = Path("/tmp/pkwarm-h000")

os.environ["MLX_OMARCHY_PK_KEEP_WORKER"] = "1"
os.environ["MLX_OMARCHY_ANE_SOCK"] = SOCK

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
print(f"running {n} invocations against daemon at {SOCK}")
for i in range(1, n + 1):
    out = OUT / f"t-{i}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    start = time.monotonic()
    rc = subprocess.run(
        [VENV_PY, CLI, "transcribe", "-o", str(out)],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    rp = out / "transcribe-report.json"
    if not rp.is_file():
        print(f"r{i} ec={rc.returncode} wall={elapsed_ms:.0f}ms NO_REPORT")
        print(f"  log: {log}")
        continue
    r = json.loads(rp.read_text())
    tsha = hashlib.sha256(r["transcript"].encode()).hexdigest()[:16]
    print(
        f"r{i} ec={rc.returncode} wall={elapsed_ms:.0f}ms "
        f"sha={tsha} status={r['status']} "
        f"session={r['ane']['session']['open_ms']:.0f}ms open / "
        f"exec={r['ane']['exec_ms']:.0f}ms / "
        f"pipe={r['timing']['total_pipeline_ms']:.0f}ms / "
        f"wker={r['ane']['worker_starts']} / "
        f"cpu_events={r['execution']['cpu_tensor_events']} / "
        f"emissions={r['execution']['chain_slots_used']}"
    )
    if r["status"] != "match":
        print(f"  FAILED checks: {r['verification']['checks_failed']}")