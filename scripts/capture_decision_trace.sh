#!/usr/bin/env bash
# Wait for a running pair's Laya worker to come up, then drive one
# /v1/decisions call and capture the dispatch lines. Designed to be
# run inside the gpu-turn ticket alongside the main run_pair.sh flow.
#
# Usage: capture_decision_trace.sh <pair_id> <home_dir> <out_dir>
set -euo pipefail
PAIR_ID=${1:?pair id required}
HOME_DIR=${2:?home dir required}
OUT=${3:?out dir required}
DECISION_LOG="$HOME_DIR/assistant/logs/${PAIR_ID}-decision.log"

mkdir -p "$OUT"
echo "Waiting for decision worker to bind port (log $DECISION_LOG)..."
DEADLINE=$((SECONDS + 1500))
DECISION_PORT=""
while [ $SECONDS -lt $DEADLINE ]; do
  if [ -s "$DECISION_LOG" ]; then
    DECISION_PORT=$(grep -oE "Starting httpd at [0-9.]+ on port ([0-9]+)" "$DECISION_LOG" | head -1 | grep -oE "[0-9]+$" || true)
    if [ -n "$DECISION_PORT" ]; then
      # Probe: GET /v1/models
      if curl -s -m 3 "http://127.0.0.1:$DECISION_PORT/v1/models" 2>/dev/null | grep -q "object" || \
         curl -s -m 3 "http://127.0.0.1:$DECISION_PORT/healthz" 2>/dev/null; then
        echo "decision worker bound on port $DECISION_PORT"
        break
      fi
    fi
  fi
  sleep 2
done
if [ -z "$DECISION_PORT" ]; then
  echo "FAILED: decision worker never bound" >&2
  exit 1
fi

# Snapshot the dispatch line count BEFORE the request.
BEFORE=$(grep -c "\[rtmod\] DISPATCH" "$DECISION_LOG" 2>/dev/null || echo 0)
echo "decision dispatch lines before: $BEFORE"

# Drive a single decision request.
echo "POST /v1/decisions ..."
curl -s -X POST "http://127.0.0.1:$DECISION_PORT/v1/decisions" \
  -H "Content-Type: application/json" \
  -d '{"questions":[{"id":"q1","question":"Pick the bitter one: espresso | tea | juice","options":[{"id":"a","label":"espresso"},{"id":"b","label":"tea"},{"id":"c","label":"juice"}],"criteria":"bitterness"}]}' \
  | tee "$OUT/decision-response.json"
echo

# Snapshot AFTER.
sleep 1
AFTER=$(grep -c "\[rtmod\] DISPATCH" "$DECISION_LOG" 2>/dev/null || echo 0)
echo "decision dispatch lines after: $AFTER"
echo "delta: $((AFTER - BEFORE))"

{
  echo "decision_port=$DECISION_PORT"
  echo "decision_dispatch_lines_before=$BEFORE"
  echo "decision_dispatch_lines_after=$AFTER"
  echo "decision_dispatch_lines_delta=$((AFTER - BEFORE))"
  echo "max_running_count=$(grep -oE 'count=[0-9]+' $DECISION_LOG 2>/dev/null | awk -F= '{print $2}' | sort -n | tail -1 2>/dev/null || echo 0)"
} > "$OUT/decision-trace-summary.txt"

cat "$OUT/decision-trace-summary.txt"