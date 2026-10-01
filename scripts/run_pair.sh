#!/usr/bin/env bash
# Run one PairGates measurement:
#   run_pair.sh <pair_id> <home_dir> <label>
# pair_id is 'everyday' or 'quality'. home_dir is the assistant home
# (a directory with an assistant/ subdir).
set -euo pipefail
PAIR_ID=${1:?pair id required}
HOME_DIR=${2:?home dir required}
LABEL=${3:?label required}
RUN_ID="$(date -u +%Y-%m-%dT%H-%MZ)-$LABEL"
ART=${ARTIFACT_DIR:-<home>/.local/share/apple-silicon-lab/artifacts/PairGates/$RUN_ID}
mkdir -p "$ART"

# Check fuser state (warn if anything else is on the GPU).
GPU_USERS=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
if [ -n "$GPU_USERS" ]; then
  echo "WARN: renderD128 in use by pids: $GPU_USERS"
fi

LOG="$ART/assistant.log"
TRACE_LOG="$ART/trace.log"
MEASURE_JSON="$ART/measure.json"
COUNTERS_BEFORE="$ART/counters-before.json"
COUNTERS_AFTER="$ART/counters-after.json"
RUNTIME_JSON="$HOME_DIR/assistant/application.json"
INFO=<home>/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-info

# Pick a random free port.
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "chosen port: $PORT"

# Start the assistant.
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
echo "started assistant pid=$APP_PID log=$LOG trace=$TRACE_LOG"

# Wait for application.json and /api/status to come up.
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
echo "assistant ready on port $PORT_ACTUAL"

# Pull cookie and csrf from application.json for authenticated status calls.
COOKIE=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['cookie'])" "$RUNTIME_JSON")
CSRF=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['csrf'])" "$RUNTIME_JSON")

# Wait for setup to complete (status -> state=complete).
echo "waiting for setup completion..."
while [ $SECONDS -lt $DEADLINE ]; do
  STATE=$(curl -s -H "Cookie: $COOKIE" -H "X-Assistant-CSRF: $CSRF" \
              -H "Origin: http://127.0.0.1:$PORT_ACTUAL" \
              "http://127.0.0.1:$PORT_ACTUAL/api/status" \
           | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('setup',{}).get('state') or d.get('state','?'))" 2>/dev/null || echo '?')
  if [ "$STATE" = "complete" ]; then
    echo "setup complete"
    break
  fi
  if [ "$STATE" = "error" ]; then
    echo "setup errored"
    curl -s -H "Cookie: $COOKIE" -H "X-Assistant-CSRF: $CSRF" \
         -H "Origin: http://127.0.0.1:$PORT_ACTUAL" \
         "http://127.0.0.1:$PORT_ACTUAL/api/status" | python3 -m json.tool
    tail -200 "$LOG" >&2 || true
    exit 2
  fi
  sleep 2
done

# Fence: the chat worker must have bound its port before sampling memory.
# The chat worker writes to $HOME_DIR/assistant/logs/<pair>-chat.log once
# it is listening; we parse the port and probe it with a real completion
# to confirm the model is loaded.
CHAT_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-chat.log"
DECISION_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-decision.log"
CHAT_PORT=""
DECISION_PORT=""
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

# Find the decision worker port from its log.
if [ -s "$DECISION_LOG" ]; then
  DECISION_PORT=$(grep -oE "Starting httpd at [0-9.]+ on port ([0-9]+)" "$DECISION_LOG" | head -1 | grep -oE "[0-9]+$" || true)
  if [ -n "$DECISION_PORT" ]; then
    echo "decision worker ready on port $DECISION_PORT"
  fi
fi

# Drive one warmup completion through the chat worker so the model is
# fully loaded before memory sampling begins. This guarantees the chat
# worker is in the descendant tree at peak.
curl -s -X POST "http://127.0.0.1:$CHAT_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
  >/dev/null || true
echo "warmup complete"

# Kick off the decision-path capture in the background; the harness runs
# in parallel. The probe waits on the Laya worker port and POSTs a
# single /v1/decisions call, capturing dispatch lines from the worker's
# stderr log.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DECISION_TRACE_DIR="$ART/decision-trace"
mkdir -p "$DECISION_TRACE_DIR"
bash "$SCRIPT_DIR/capture_decision_trace.sh" "$PAIR_ID" "$HOME_DIR" "$DECISION_TRACE_DIR" \
  >"$DECISION_TRACE_DIR/capture.log" 2>&1 &
DECISION_TRACE_PID=$!
echo "decision trace probe pid=$DECISION_TRACE_PID"

# Snapshot counters before measurement.
$INFO --json >"$COUNTERS_BEFORE" || true

# Run the measurement harness.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="<home>/agents/PairGates/worktree/scripts:<home>/agents/PairGates/worktree/serve" \
  <home>/.local/share/mlx-omarchy/venv/bin/python \
  -u "$SCRIPT_DIR/pair_measure.py" \
  --assistant-app-json "$RUNTIME_JSON" \
  --assistant-pid "$APP_PID" \
  --chat-port "$CHAT_PORT" \
  --decision-port "${DECISION_PORT:-0}" \
  --chat-log "$TRACE_CHAT_LOG" \
  --decision-log "$TRACE_DECISION_LOG" \
  --label "$LABEL" \
  --out "$MEASURE_JSON"

# Snapshot counters after.
$INFO --json >"$COUNTERS_AFTER" || true

# Trace counts from the assistant's stderr (trace.log), chat worker log,
# and decision worker log. Split per-worker so we can attribute lines.
TRACE_CHAT_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-chat.log"
TRACE_DECISION_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-decision.log"

count_dispatch() {
    grep -c "\[rtmod\] DISPATCH" "$1" 2>/dev/null || echo 0
}
max_count() {
    grep -oE "count=[0-9]+" "$1" 2>/dev/null | awk -F= '{print $2}' | sort -n | tail -1 2>/dev/null || echo 0
}

DISPATCH_ASSISTANT=$(count_dispatch "$TRACE_LOG")
DISPATCH_CHAT=$(count_dispatch "$TRACE_CHAT_LOG")
DISPATCH_DECISION=$(count_dispatch "$TRACE_DECISION_LOG")
echo "DISPATCH lines: assistant=$DISPATCH_ASSISTANT chat=$DISPATCH_CHAT decision=$DISPATCH_DECISION"

MAX_ASSISTANT=$(max_count "$TRACE_LOG")
MAX_CHAT=$(max_count "$TRACE_CHAT_LOG")
MAX_DECISION=$(max_count "$TRACE_DECISION_LOG")
echo "max count: assistant=${MAX_ASSISTANT:-0} chat=${MAX_CHAT:-0} decision=${MAX_DECISION:-0}"

{
  echo "dispatch_lines_assistant=$DISPATCH_ASSISTANT"
  echo "dispatch_lines_chat=$DISPATCH_CHAT"
  echo "dispatch_lines_decision=$DISPATCH_DECISION"
  echo "max_running_count_assistant=${MAX_ASSISTANT:-0}"
  echo "max_running_count_chat=${MAX_CHAT:-0}"
  echo "max_running_count_decision=${MAX_DECISION:-0}"
  if [ -f "$DECISION_TRACE_DIR/decision-trace-summary.txt" ]; then
    cat "$DECISION_TRACE_DIR/decision-trace-summary.txt" >> "$ART/trace-summary.txt"
  fi
} > "$ART/trace-summary.txt"

echo "DONE: $ART"
ls -la "$ART"