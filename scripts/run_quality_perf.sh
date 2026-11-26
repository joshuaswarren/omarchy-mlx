#!/usr/bin/env bash
# Like run_pair.sh but for the Quality idle-GPU perf set.
set -euo pipefail
PAIR_ID=${1:?pair id required}
HOME_DIR=${2:?home dir required}
LABEL=${3:?label required}
RUN_ID="$(date -u +%Y-%m-%dT%H-%MZ)-$LABEL"
ART=${ARTIFACT_DIR:-<home>/.local/share/apple-silicon-lab/artifacts/PairGates/$RUN_ID}
mkdir -p "$ART"

GPU_USERS=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
if [ -n "$GPU_USERS" ]; then
  echo "WARN: renderD128 in use by pids: $GPU_USERS"
fi
LOAD_BEFORE=$(cat /proc/loadavg)
echo "loadavg before: $LOAD_BEFORE"

LOG="$ART/assistant.log"
TRACE_LOG="$ART/trace.log"
MEASURE_JSON="$ART/measure.json"
PERF_JSON="$ART/quality-perf.json"
COUNTERS_BEFORE="$ART/counters-before.json"
COUNTERS_AFTER="$ART/counters-after.json"
RUNTIME_JSON="$HOME_DIR/assistant/application.json"
INFO=<home>/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-info

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "chosen port: $PORT"

MLX_OMARCHY_TRACE_DISPATCH=1 \
  PYTHONPATH="<home>/agents/PairGates/worktree/scripts:<home>/agents/PairGates/worktree/serve" \
  <home>/.local/share/mlx-omarchy/venv/bin/python \
  -u -m mlx_omarchy_assistant \
    --home "$HOME_DIR" --port "$PORT" \
    --pair "$PAIR_ID" --yes --no-browser \
    >>"$LOG" 2>>"$TRACE_LOG" &
APP_PID=$!
echo "$APP_PID" >"$ART/assistant.pid"
trap 'kill -KILL -"$APP_PID" 2>/dev/null || true; rm -f "$ART/assistant.pid"' EXIT
echo "started assistant pid=$APP_PID"

DEADLINE=$((SECONDS + 1500))
while [ $SECONDS -lt $DEADLINE ]; do
  if [ -s "$RUNTIME_JSON" ]; then
    PORT_ACTUAL=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['port'])" "$RUNTIME_JSON")
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_ACTUAL/api/status" | grep -q 200; then
      break
    fi
  fi
  sleep 2
done
if [ ! -s "$RUNTIME_JSON" ]; then
  echo "FAILED: assistant never wrote application.json" >&2
  tail -200 "$LOG" >&2 || true
  exit 1
fi
COOKIE=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['cookie'])" "$RUNTIME_JSON")
CSRF=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['csrf'])" "$RUNTIME_JSON")
echo "assistant ready on port $PORT_ACTUAL"

while [ $SECONDS -lt $DEADLINE ]; do
  STATE=$(curl -s -H "Cookie: $COOKIE" -H "X-Assistant-CSRF: $CSRF" \
              -H "Origin: http://127.0.0.1:$PORT_ACTUAL" \
              "http://127.0.0.1:$PORT_ACTUAL/api/status" \
           | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('setup',{}).get('state') or d.get('state','?'))" 2>/dev/null || echo '?')
  if [ "$STATE" = "complete" ]; then echo "setup complete"; break; fi
  if [ "$STATE" = "error" ]; then echo "setup errored"; tail -200 "$LOG" >&2 || true; exit 2; fi
  sleep 2
done

CHAT_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-chat.log"
DECISION_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-decision.log"
CHAT_PORT=""
while [ $SECONDS -lt $DEADLINE ]; do
  if [ -s "$CHAT_LOG" ]; then
    CHAT_PORT=$(grep -oE "Starting httpd at [0-9.]+ on port ([0-9]+)" "$CHAT_LOG" | head -1 | grep -oE "[0-9]+$" || true)
    if [ -n "$CHAT_PORT" ] && curl -s "http://127.0.0.1:$CHAT_PORT/v1/models" 2>/dev/null | grep -q "object"; then
      echo "chat worker ready on port $CHAT_PORT"
      break
    fi
  fi
  sleep 2
done
if [ -z "$CHAT_PORT" ]; then
  echo "FAILED: chat worker never bound its port" >&2
  tail -50 "$CHAT_LOG" >&2 || true
  exit 3
fi
DECISION_PORT=""
if [ -s "$DECISION_LOG" ]; then
  DECISION_PORT=$(grep -oE "Starting httpd at [0-9.]+ on port ([0-9]+)" "$DECISION_LOG" | head -1 | grep -oE "[0-9]+$" || true)
  echo "decision worker ready on port ${DECISION_PORT:-?}"
fi

# Warmup completion (fence).
curl -s -X POST "http://127.0.0.1:$CHAT_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
  >/dev/null || true
echo "warmup complete"

$INFO --json >"$COUNTERS_BEFORE" || true

# Run the per-phase memory harness.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="<home>/agents/PairGates/worktree/scripts:<home>/agents/PairGates/worktree/serve" \
  <home>/.local/share/mlx-omarchy/venv/bin/python \
  -u "$SCRIPT_DIR/pair_measure.py" \
  --assistant-app-json "$RUNTIME_JSON" \
  --assistant-pid "$APP_PID" \
  --chat-port "$CHAT_PORT" \
  --decision-port "${DECISION_PORT:-0}" \
  --chat-log "$CHAT_LOG" \
  --decision-log "$DECISION_LOG" \
  --label "$LABEL-memory" \
  --out "$MEASURE_JSON"

# Run the Quality idle-GPU perf harness.
PYTHONPATH="<home>/agents/PairGates/worktree/scripts:<home>/agents/PairGates/worktree/serve" \
  <home>/.local/share/mlx-omarchy/venv/bin/python \
  -u "$SCRIPT_DIR/quality_perf.py" \
  --assistant-app-json "$RUNTIME_JSON" \
  --chat-port "$CHAT_PORT" \
  --chat-log "$CHAT_LOG" \
  --decision-log "$DECISION_LOG" \
  --label "$LABEL-perf" \
  --out "$PERF_JSON"

$INFO --json >"$COUNTERS_AFTER" || true

count_dispatch() { grep -c "\[rtmod\] DISPATCH" "$1" 2>/dev/null || echo 0; }
max_count() { grep -oE "count=[0-9]+" "$1" 2>/dev/null | awk -F= '{print $2}' | sort -n | tail -1 2>/dev/null || echo 0; }
DISPATCH_ASSISTANT=$(count_dispatch "$TRACE_LOG")
DISPATCH_CHAT=$(count_dispatch "$CHAT_LOG")
DISPATCH_DECISION=$(count_dispatch "$DECISION_LOG")
MAX_ASSISTANT=$(max_count "$TRACE_LOG")
MAX_CHAT=$(max_count "$CHAT_LOG")
MAX_DECISION=$(max_count "$DECISION_LOG")

LOAD_AFTER=$(cat /proc/loadavg)
GPU_USERS_AFTER=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)

{
  echo "dispatch_lines_assistant=$DISPATCH_ASSISTANT"
  echo "dispatch_lines_chat=$DISPATCH_CHAT"
  echo "dispatch_lines_decision=$DISPATCH_DECISION"
  echo "max_running_count_assistant=${MAX_ASSISTANT:-0}"
  echo "max_running_count_chat=${MAX_CHAT:-0}"
  echo "max_running_count_decision=${MAX_DECISION:-0}"
  echo "loadavg_before=$LOAD_BEFORE"
  echo "loadavg_after=$LOAD_AFTER"
  echo "fuser_after=$GPU_USERS_AFTER"
} > "$ART/trace-summary.txt"

echo "DONE: $ART"
ls -la "$ART"