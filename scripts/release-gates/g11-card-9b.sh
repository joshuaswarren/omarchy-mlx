#!/usr/bin/env bash
# Gate 10 — card smoke on the 9B: the g3 card check (SSE stream visibility)
# against a fresh everyday-pair (9B) home. Proves the card path still works
# with stream-time card promotion.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
# LOCKER: the binary the setsid+execvp launcher runs. Raw mode wraps the
# server in flock; inside a gpu-turn ticket (GPU_TURN_TICKET=1) the ticket
# already holds the lock and nesting flock deadlocks, so exec timeout only.
LOCKER=(flock -x -w 900); [[ "${GPU_TURN_TICKET:-}" == 1 ]] && LOCKER=(timeout)
LOG="$LOG_DIR/g11-card-9b.log"
RUNNER="$GATES_DIR/gate3-card-runner.py"
ASSIST="$GATE_ROOT/${TAG}-assist-9b-card"

gate_refuse_existing "$ASSIST"
mkdir -p "$ASSIST"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"

python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  "${LOCKER[@]}" timeout -k 60 2400 \
  env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" HF_HOME="$HF_CACHE" \
  "$GATE_HOME/.local/bin/mlx-omarchy-chat" --home "$ASSIST" --no-browser --pair everyday --yes \
  >"$LOG.server" 2>&1 &
PID=$!
python3 "$RUNNER" "$ASSIST" "$LOG" cards
RC=$?
if ! grep -q "CARD_CHECK PASS" "$LOG"; then RC=1; fi

PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' \
  "$ASSIST/assistant/application.json" 2>/dev/null)
if [[ -n "$PORT" ]]; then
  if LISTENERS=$(fuser -n tcp "$PORT" 2>/dev/null); then
    for SERVER_PID in $LISTENERS; do
      PGID=$(python3 -c 'import pathlib,sys; print((pathlib.Path("/proc")/sys.argv[1]/"stat").read_text().split()[4])' "$SERVER_PID" 2>/dev/null)
      [[ -z "$PGID" ]] || { kill -TERM -- "-$PGID" 2>/dev/null; sleep 5; kill -KILL -- "-$PGID" 2>/dev/null; }
      kill -TERM "$SERVER_PID" 2>/dev/null
    done
  fi
fi
kill -TERM -- "-$PID" 2>/dev/null
wait "$PID" 2>/dev/null || :
if [[ -n "$PORT" ]]; then
  if FUSER_OUT=$(fuser -n tcp "$PORT" 2>&1); then FUSER_RC=0; else FUSER_RC=$?; fi
  gate_log "$LOG" "fuser_port=$PORT exit=$FUSER_RC output=$FUSER_OUT"
  [[ $FUSER_RC -eq 1 && -z "$FUSER_OUT" ]] || RC=1
fi
gate_log "$LOG" "GATE11_EXIT $RC"
exit "$RC"
