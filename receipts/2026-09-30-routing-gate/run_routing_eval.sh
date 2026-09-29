#!/bin/bash
set -e
# Start Laya worker under flock /tmp/m2-gpu.lock, wait for /health, run the
# held-out routing suite evaluation, then stop the worker.
HOME="<qual-home>"
SRC="<repo>"
PY="<venv>/bin/python"
LAYA_MODEL="$HOME/models/laya-mlx"
PORT="${ROUTING_EVAL_PORT:-50500}"
LOG="<log>/routing-eval-laya.log"
OUT="<notebook>/RoutingGate/routing-gate/eval-$(date -u +%Y%m%dT%H%M%SZ).json"
mkdir -p "$(dirname "$OUT")"

cd "$SRC"
PYTHONPATH=serve flock /tmp/m2-gpu.lock \
  setsid nohup "$PY" -m mlx_omarchy_laya.server --model "$LAYA_MODEL" \
    --host 127.0.0.1 --port "$PORT" --dtype float16 --max-questions 8 \
    > "$LOG" 2>&1 < /dev/null &
WORKER_PID=$!
echo "Laya worker pid=$WORKER_PID port=$PORT"

# Wait for /health
for i in $(seq 1 60); do
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Laya /health up after ${i}s"; break
  fi
  sleep 2
done

# Run evaluation under the same lock to keep the GPU exclusive
PYTHONPATH=serve flock /tmp/m2-gpu.lock \
  "$PY" "<home>/src/routing-evaluate.py" \
    --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
    --deadline-seconds 2.0 \
    --out "$OUT"

# Stop the Laya worker
kill -TERM "$WORKER_PID" 2>/dev/null || true
sleep 1
kill -KILL "$WORKER_PID" 2>/dev/null || true
echo "wrote $OUT"