#!/usr/bin/env bash
# Gate 3 — fresh 4B assistant home; card visibility via the corrected SSE
# runner (default since v0.7.12). Listener cleanup folded in.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g3-online-4b.log"
RUNNER="$GATES_DIR/gate3-card-runner.py"

gate_refuse_existing "$ASSIST_4B"
mkdir -p "$ASSIST_4B"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"
gate_log "$LOG" "df_before $(df -BG "$GATE_ROOT" | tail -1 | awk '{print $4}')"

python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  flock -x -w 300 "$GPU_LOCK" timeout -k 60 2400 \
  env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" HF_HOME="$HF_CACHE" \
  "$GATE_HOME/.local/bin/mlx-omarchy-chat" --home "$ASSIST_4B" --no-browser --pair compact --yes \
  >"$LOG.server" 2>&1 &
PID=$!
python3 "$RUNNER" "$ASSIST_4B" "$LOG"
RC=$?
if ! grep -q "OUTBOUND_DENIED False" "$LOG" || ! grep -q "CARD_CHECK PASS" "$LOG"; then RC=1; fi

PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' \
  "$ASSIST_4B/assistant/application.json" 2>/dev/null)
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
gate_log "$LOG" "GATE3_ONLINE_EXIT $RC"
gate_log "$LOG" "hf_bytes_after $(du -sb "$HF_CACHE" | cut -f1)"
exit "$RC"
