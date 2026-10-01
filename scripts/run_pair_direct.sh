#!/usr/bin/env bash
# Manual pair launcher + scripted session measurer.
# Spawns chat + laya workers like the assistant does, then runs a real
# chat/completion/decision/chat session and samples RSS/PSS for the whole
# tree. Cheaper than the full assistant path; uses the same workers.
#
# Usage: run_pair_direct.sh <chat_model_path> <laya_path> <label>
set -uo pipefail

CHAT_MODEL_PATH=${1:?chat model path required}
LAYA_PATH=${2:?laya path required}
LABEL=${3:?label required}
RUN_ID="$(date -u +%Y-%m-%dT%H-%MZ)-$LABEL"
ART=${ARTIFACT_DIR:-<home>/.local/share/apple-silicon-lab/artifacts/PairGates/$RUN_ID}
mkdir -p "$ART"

CHAT_LOG="$ART/chat.log"
LAYA_LOG="$ART/laya.log"
TRACE_LOG="$ART/trace.log"
CHAT_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
LAYA_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "chat port: $CHAT_PORT laya port: $LAYA_PORT"

# Start chat server (mlx-lm shim).
MLX_OMARCHY_TRACE_DISPATCH=1 \
PYTHONPATH=<home>/agents/PairGates/worktree/serve \
<home>/.local/share/mlx-omarchy/venv/bin/python -u \
  <home>/agents/PairGates/worktree/serve/mlx_omarchy_serve/_mlxlm_server.py \
  --model "$CHAT_MODEL_PATH" \
  --host 127.0.0.1 --port "$CHAT_PORT" \
  --max-tokens 512 --decode-concurrency 1 --prompt-concurrency 1 \
  --prompt-cache-size 0 \
  >>"$CHAT_LOG" 2>>"$TRACE_LOG" &
CHAT_PID=$!
echo "chat pid: $CHAT_PID"

# Start laya server.
MLX_OMARCHY_TRACE_DISPATCH=1 \
PYTHONPATH=<home>/agents/PairGates/worktree/serve \
<home>/.local/share/mlx-omarchy/venv/bin/python -u \
  -c "from mlx_omarchy_laya.server import serve_main; import sys; sys.argv=['laya','--model',sys.argv[1],'--host','127.0.0.1','--port',sys.argv[2],'--max-questions','8']; serve_main(sys.argv[3:])" \
  -- "$LAYA_PATH" "$LAYA_PORT" \
  >>"$LAYA_LOG" 2>>"$TRACE_LOG" &
LAYA_PID=$!
echo "laya pid: $LAYA_PID"

# Trap to clean up on exit
trap 'echo "killing chat and laya"; kill -KILL -"$CHAT_PID" -"$LAYA_PID" 2>/dev/null || true' EXIT

# Wait for chat to be healthy (port responds)
CHAT_DEADLINE=$((SECONDS + 600))
while [ $SECONDS -lt $CHAT_DEADLINE ]; do
  if curl -s "http://127.0.0.1:$CHAT_PORT/v1/models" 2>/dev/null | grep -q "object"; then
    echo "chat ready"
    break
  fi
  sleep 2
done
if ! curl -s "http://127.0.0.1:$CHAT_PORT/v1/models" 2>/dev/null | grep -q "object"; then
  echo "CHAT FAILED to start" >&2
  tail -50 "$CHAT_LOG" >&2 || true
  exit 1
fi

# Wait for laya to be healthy
while [ $SECONDS -lt $CHAT_DEADLINE ]; do
  if curl -s -X POST -H "Content-Type: application/json" -d '{}' "http://127.0.0.1:$LAYA_PORT/v1/decisions" 2>/dev/null; then
    echo "laya ready"
    break
  fi
  sleep 2
done

# Run the scripted session via pair_measure.py.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH=<home>/agents/PairGates/worktree/serve \
<home>/.local/share/mlx-omarchy/venv/bin/python -u \
  "$SCRIPT_DIR/pair_direct_measure.py" \
  --chat-url "http://127.0.0.1:$CHAT_PORT" \
  --laya-url "http://127.0.0.1:$LAYA_PORT" \
  --chat-pid "$CHAT_PID" \
  --laya-pid "$LAYA_PID" \
  --label "$LABEL" \
  --out "$ART/measure.json"

# After measurements, count trace lines.
DISPATCH_LINES=$(grep -c "\[rtmod\] DISPATCH" "$TRACE_LOG" 2>/dev/null || echo 0)
MAX_COUNT=$(grep -oE "count=[0-9]+" "$TRACE_LOG" | awk -F= '{print $2}' | sort -n | tail -1 2>/dev/null || echo 0)
echo "DISPATCH lines: $DISPATCH_LINES, max running count: ${MAX_COUNT:-0}"
{
  echo "dispatch_lines=$DISPATCH_LINES"
  echo "max_running_count=${MAX_COUNT:-0}"
} > "$ART/trace-summary.txt"

echo "DONE: $ART"
ls -la "$ART"