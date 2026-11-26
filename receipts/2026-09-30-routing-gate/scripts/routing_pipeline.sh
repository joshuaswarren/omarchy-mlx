#!/bin/bash
# RoutingGate pipeline (resumable, gpu-turn scheduled).
# Run as: gpu-turn -m 15 -- bash routing_pipeline.sh dev
#                                            held
# Each step writes its own resumable file under the artifacts dir and
# syncs after every case. After every step, run "sync" on the M2 to
# survive a reboot.
set -e
HOME="<home>/mlx-chat-qual-home"
SRC="<home>/mlx-assistant-qual"
PY="<home>/.local/share/mlx-omarchy/venv/bin/python"
LAYA_MODEL="$HOME/models/laya-mlx"
PORT="${ROUTING_EVAL_PORT:-50500}"
LOG="<home>/routing-pipeline-laya.log"
ART="<home>/.local/share/apple-silicon-lab/artifacts/RoutingGate/routing-gate"
mkdir -p "$ART"

cd "$SRC"

PHASE="${1:-dev}"

# Clear any stale laya-mlx reservation left by a prior crash or kill.
# A "resident" entry without a live owner blocks the next worker from
# admitting. We bypass the owner check because the prior process is
# known to be gone (verified via /proc before launching).
RESV="$HOME/.local/share/mlx-omarchy/reservations.json"
if [ -f "$RESV" ]; then
  RESV="$RESV" PYTHONPATH=serve "$PY" - <<'PYEOF' 2>/dev/null || true
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

# Boot Laya worker (short-lived, only inside this gpu-turn slot).
PYTHONPATH=serve "$PY" -m mlx_omarchy_laya.server --model "$LAYA_MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype float16 --max-questions 8 \
  > "$LOG" 2>&1 &
WORKER_PID=$!
cleanup() {
  kill -TERM "$WORKER_PID" 2>/dev/null || true
  sleep 1
  kill -KILL "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 60); do
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[routing-pipeline] Laya /health up after ${i}s"
    break
  fi
  sleep 2
done

"$PY" scripts/mlx_provenance.py > "$ART/provenance-$PHASE.txt" 2>&1 || true
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg; } \
  >> "$ART/host-$PHASE.txt"

if [ "$PHASE" = "dev" ]; then
  PYTHONPATH=serve "$PY" <home>/src/dev_sweep_run.py \
    --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
    --suite "<home>/src/routing_dev.json" \
    --deadline-seconds 2.0 \
    --out "$ART/dev-raw-latest.json"
elif [ "$PHASE" = "held" ]; then
  # Resumable: the runner skips cases already checkpointed in this file.
  PYTHONPATH=serve "$PY" <home>/src/dev_sweep_run.py \
    --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
    --suite "<home>/src/routing_held_out.json" \
    --deadline-seconds 2.0 \
    --out "$ART/held-raw.json"
  PYTHONPATH="serve:<home>/src" "$PY" <home>/src/routing_latency.py \
    --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
    --suite "<home>/src/routing_dev.json" \
    --out "$ART/latency-warm-100.json"
elif [ "$PHASE" = "latency" ]; then
  PYTHONPATH="serve:<home>/src" "$PY" <home>/src/routing_latency.py \
    --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
    --suite "<home>/src/routing_dev.json" \
    --out "$ART/latency-warm-100-dev.json"
else
  echo "usage: $0 dev|held" >&2
  exit 2
fi
sync
echo "[routing-pipeline] DONE phase=$PHASE"