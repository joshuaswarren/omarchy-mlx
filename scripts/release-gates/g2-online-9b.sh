#!/usr/bin/env bash
# Gate 2 — fresh assistant HOME + EMPTY HF cache, online 9B.
# Listener cleanup folded in (process-group kill + fuser re-verify; a leftover
# listener fails the gate unless cleared).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g2-online-9b.log"
PROBE="$GATES_DIR/gate-probe.py"

gate_refuse_existing "$HF_CACHE" "$ASSIST_9B"
mkdir -p "$HF_CACHE" "$ASSIST_9B"
: > "$LOG"
gate_begin "$LOG"
gate_log "$LOG" "df_before $(df -BG "$GATE_ROOT" | tail -1 | awk '{print $4}')"
gate_log "$LOG" "hf_bytes_before $(du -sb "$HF_CACHE" | cut -f1)"

python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  flock -x -w 300 "$GPU_LOCK" timeout -k 60 2400 \
  env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" HF_HOME="$HF_CACHE" \
  "$GATE_HOME/.local/bin/mlx-omarchy-chat" --home "$ASSIST_9B" --no-browser --pair everyday --yes \
  >"$LOG.server" 2>&1 &
PID=$!
python3 "$PROBE" "$ASSIST_9B" "$LOG" online
RC=$?
if ! grep -q "OUTBOUND_DENIED False" "$LOG"; then RC=1; fi

# Listener cleanup: kill the server's process group, then prove the port is free.
PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' \
  "$ASSIST_9B/assistant/application.json" 2>/dev/null)
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
gate_log "$LOG" "GATE2_ONLINE_EXIT $RC"
gate_log "$LOG" "df_after $(df -BG "$GATE_ROOT" | tail -1 | awk '{print $4}')"
gate_log "$LOG" "hf_bytes_after $(du -sb "$HF_CACHE" | cut -f1)"
exit "$RC"
