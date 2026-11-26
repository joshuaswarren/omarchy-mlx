#!/bin/bash
# LayaHead2 final gate: ONE run of the frozen held-out suite with the winning
# configuration, same harness as the 2026-09-30 evaluation, plus the 100-call
# dev and held-out warm-latency runs (routing_latency.py shape).
# usage: run_final.sh OUT_DIR SERVE SUITE_HELD SUITE_DEV MODEL PORT [EXTRA_SRV_ARGS...]
set -euo pipefail

OUT=$1; SERVE=$2; HELD=$3; DEV=$4; MODEL=$5; PORT=$6
shift 6
EXTRA_SRV=("$@")

PY=${PY:-"$HOME/.local/share/mlx-omarchy/venv/bin/python"}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

state() {
  {
    date -u +%FT%TZ
    echo "boot $(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime $(cat /proc/uptime)"
    echo "load $(cat /proc/loadavg)"
    cat /proc/pressure/cpu
  } > "$OUT/host-$1.txt"
}

RESV="$HOME/.local/share/mlx-omarchy/reservations.json"
if [ -f "$RESV" ]; then
  RESV="$RESV" PYTHONPATH="$SERVE" "$PY" - <<'PYEOF' 2>/dev/null || true
import json, os
from pathlib import Path
p = Path(os.environ["RESV"])
if p.exists():
    d = json.loads(p.read_text())
    if "laya-mlx" in d:
        del d["laya-mlx"]
        p.write_text(json.dumps(d))
PYEOF
fi

state before
PYTHONPATH="$SERVE" "$PY" -m mlx_omarchy_laya.server --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype float16 --max-questions 8 \
  "${EXTRA_SRV[@]}" > "$OUT/server.log" 2>&1 &
WPID=$!
cleanup() {
  kill -TERM "$WPID" 2>/dev/null || true
  sleep 1
  kill -KILL "$WPID" 2>/dev/null || true
}
trap cleanup EXIT
up=0
for i in $(seq 1 60); do
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then up=1; break; fi
  sleep 2
done
[ "$up" = 1 ] || { echo "server failed" >&2; exit 4; }

# 1) held-out raw answers (resumable), gate harness, deadline 2.0 as the gate.
PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/dev_sweep_run.py" \
  --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
  --suite "$HELD" --deadline-seconds 2.0 --out "$OUT/held-raw.json"

# 2) held-out warm latency, 5 warmups + 100 timed calls.
PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/routing_latency.py" \
  --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
  --suite "$HELD" --out "$OUT/latency-warm-100-held.json"

# 3) dev warm latency, same shape (the p95 <= 250 dev criterion).
PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/routing_latency.py" \
  --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
  --suite "$DEV" --out "$OUT/latency-warm-100-dev.json"

"$PY" -c 'import mlx.core as mx; print(mx.__version__)' > "$OUT/wheel.txt"
state after
ps -eo pid,ppid,etime,cmd > "$OUT/ps-end.txt"
sync
echo "[final] DONE"
