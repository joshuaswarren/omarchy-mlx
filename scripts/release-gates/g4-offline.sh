#!/usr/bin/env bash
# Gate 4 — offline restart: app AND probe inside the same unshare -n -r
# namespace with loopback up (corrected procedure as default).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g4-offline.log"
PROBE="$GATES_DIR/gate-probe.py"

gate_refuse_existing "$ASSIST_OFF"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"

gate_lock -x -w 300 unshare -n -r bash -c '
set -uo pipefail
ip link set lo up
export PATH="'"$GATE_INSTALL_PATH"'"
env -i PATH="$PATH" HOME="'"$GATE_HOME"'" HF_HOME="'"$HF_CACHE"'" \
  HF_HUB_OFFLINE=1 MLX_OMARCHY_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "'"$GATE_HOME"'/.local/bin/mlx-omarchy-chat" --home "'"$ASSIST_OFF"'" --no-browser --pair everyday --yes \
  >"'"$LOG"'.server" 2>&1 &
SRV=$!
python3 "'"$PROBE"'" "'"$ASSIST_OFF"'" "'"$LOG"'" offline
RC=$?
kill -TERM "$SRV" 2>/dev/null
kill -TERM -- "-$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null || :
exit $RC
'
RC=$?
gate_log "$LOG" "GATE4_OFFLINE_EXIT $RC"
exit "$RC"
