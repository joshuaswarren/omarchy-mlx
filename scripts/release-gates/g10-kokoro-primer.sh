#!/usr/bin/env bash
# Gate 9 — Kokoro primer + serve-path TTFA on the M2 (idle).
# After a fresh setup: voice.synthesis.primed must be true within 60 s and
# the first /api/speak TTFA must be <= 2.0 s (the 1.5 s-target TTFA work:
# primer + first-segment split). Must run on an idle machine — any load
# invalidates the TTFA number.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g10-kokoro-primer.log"
ASSIST="$GATE_ROOT/${TAG}-assist-primer"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"
gate_log "$LOG" "loadavg_before=$(cat /proc/loadavg)"

gate_refuse_existing "$ASSIST"
mkdir -p "$ASSIST"

python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  flock -x -w 900 "$GPU_LOCK" timeout -k 60 2400 \
  env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" HF_HOME="$HF_CACHE" \
  "$GATE_HOME/.local/bin/mlx-omarchy-chat" --home "$ASSIST" --no-browser --pair everyday --yes \
  >"$LOG.server" 2>&1 &
PID=$!
python3 "$GATES_DIR/g10-kokoro-primer-driver.py" "$ASSIST" "$LOG"
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
gate_log "$LOG" "GATE10_EXIT $RC"
exit "$RC"
