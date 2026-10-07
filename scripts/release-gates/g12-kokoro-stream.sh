#!/usr/bin/env bash
# Gate 12 — STREAMED Kokoro default path (a6729cdfa): voice-enabled fresh
# setup, primer wait, then a warm-up + a measured multi-sentence read-aloud
# over the real /api/speak stream: warm TTFA <= 1.5 s, zero starved chunk
# arrivals (gap-free playout), zero error events, done received. Requires
# an idle machine. No env override anywhere — the unset default path.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
# LOCKER: the binary the setsid+execvp launcher runs. Raw mode wraps the
# server in flock; inside a gpu-turn ticket (GPU_TURN_TICKET=1) the ticket
# already holds the lock and nesting flock deadlocks, so exec the inner
# command with no wrapper (LOCKER empty).
LOCKER=(flock -x -w 900); [[ "${GPU_TURN_TICKET:-}" == 1 ]] && LOCKER=()
LOG="$LOG_DIR/g12-kokoro-stream.log"
ASSIST="$GATE_ROOT/${TAG}-assist-stream"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"
gate_log "$LOG" "loadavg_before=$(cat /proc/loadavg)"

gate_refuse_existing "$ASSIST"
mkdir -p "$ASSIST"

python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  "${LOCKER[@]}" timeout -k 60 2400 \
  env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" HF_HOME="$HF_CACHE" \
  "$GATE_HOME/.local/bin/mlx-omarchy-chat" --home "$ASSIST" --no-browser --pair everyday --yes \
  >"$LOG.server" 2>&1 &
PID=$!
python3 "$GATES_DIR/g12-kokoro-stream-driver.py" "$ASSIST" "$LOG"
RC=$?

PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' \
  "$ASSIST/assistant/application.json" 2>/dev/null)
if [[ -n "$PORT" ]]; then
  if LISTENERS=$(fuser -n tcp "$PORT" 2>/dev/null); then
    for SERVER_PID in $LISTENERS; do
      PGID=$(python3 -c 'import pathlib,sys; print((pathlib.Path("/proc")/sys.argv[1]/"stat").read_text().split()[4])' "$SERVER_PID" 2>/dev/null)
      [[ -z "$PGID" ]] || kill -TERM -- "-$PGID" 2>/dev/null
      kill -TERM "$SERVER_PID" 2>/dev/null
    done
    sleep 5
    kill -KILL -- "-${PGID:-0}" 2>/dev/null
    sleep 1
  fi
fi
kill -TERM -- "-$PID" 2>/dev/null
wait "$PID" 2>/dev/null || :
if [[ -n "$PORT" ]]; then
  if FUSER_OUT=$(fuser -n tcp "$PORT" 2>&1); then FUSER_RC=0; else FUSER_RC=$?; fi
  gate_log "$LOG" "fuser_port=$PORT exit=$FUSER_RC output=$FUSER_OUT"
  [[ $FUSER_RC -eq 1 && -z "$FUSER_OUT" ]] || RC=1
fi
gate_log "$LOG" "GATE12_EXIT $RC"
exit "$RC"
