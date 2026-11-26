#!/bin/bash
# LayaHead2 ticket 1: inventory + phase decomposition + resident-worker HTTP
# probe. Resumable NDJSON; safe to re-run inside a fresh gpu-turn ticket.
# usage: run_ticket1.sh OUT_DIR SERVE_DIR SUITE MODEL_DIR [PORT]
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; PORT=${5:-50517}
PY=${PY:-"$HOME/.local/share/mlx-omarchy/venv/bin/python"}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

state() {  # $1 = label
  {
    date -u +%FT%TZ
    echo "boot $(cat /proc/sys/kernel/random/boot_id)"
    uname -r
    echo "uptime $(cat /proc/uptime)"
    echo "load $(cat /proc/loadavg)"
    cat /proc/pressure/cpu
    echo "others $(ps -eo pid,cmd | grep -cE 'python|mlx' || true)"
  } > "$OUT/host-$1.txt"
}

# Clear a stale laya-mlx reservation whose owner died with a prior ticket.
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

state inventory
"$PY" -c 'import mlx.core as mx; print(mx.__version__)' > "$OUT/wheel.txt" 2>&1
env | grep -E '^MLX_OMARCHY' > "$OUT/env.txt" || echo "no MLX_OMARCHY env" > "$OUT/env.txt"
(cd "$SERVE/.." && PYTHONPATH="$SERVE" "$PY" "$SCRIPTS/mlx_provenance.py" \
  > "$OUT/provenance.txt") || true

# --- phase decomposition (standalone process, default env as installed) ------
PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/head_probe.py" phases \
  --serve "$SERVE" --model "$MODEL" --suite "$SUITE" --out "$OUT/phases.ndjson" \
  --n 40 --warmup 3 --arm cap-default

# --- resident worker, gate pipeline shape ------------------------------------
PYTHONPATH="$SERVE" "$PY" -m mlx_omarchy_laya.server --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype float16 --max-questions 8 \
  > "$OUT/server.log" 2>&1 &
WORKER_PID=$!
cleanup() {
  kill -TERM "$WORKER_PID" 2>/dev/null || true
  sleep 1
  kill -KILL "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT

up=0
for i in $(seq 1 60); do
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    up=1; echo "server up after ${i}s"; break
  fi
  sleep 2
done
[ "$up" = 1 ] || { echo "server failed to come up" >&2; exit 4; }

PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/head_probe.py" http \
  --serve "$SERVE" --url "http://127.0.0.1:${PORT}/v1/decisions" \
  --suite "$SUITE" --out "$OUT/http-capdefault.ndjson" \
  --n 40 --warmup 5 --arm cap-default --server-pid "$WORKER_PID"

state after-http
ps -eo pid,ppid,etime,cmd > "$OUT/ps-end.txt"
sync
echo "[ticket1] DONE"
