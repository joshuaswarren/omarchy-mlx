#!/usr/bin/env bash
# Gate 14 — routing default smoke (M2, idle): fresh everyday (9B) homes.
#   leg A (default): routing must be ON by default — a structured-decision
#     chat turn is answered by the routed head (not the main pair), head
#     ms recorded;
#   leg B (kill switch): MLX_OMARCHY_ROUTING=0 on a second home — the
#     same turn is answered by the main pair.
# PRECONDITION: the routing-default commit (8b1226da6 or equivalent) must
# be in the cut; without it the default-ON leg cannot pass and this gate
# is expected to fail — do not ship a cut that claims routing default ON
# without this gate green.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g14-routing.log"
ASSIST_A="$GATE_ROOT/${TAG}-assist-route-on"
ASSIST_B="$GATE_ROOT/${TAG}-assist-route-off"
: > "$LOG"
gate_begin "$LOG"

run_leg() { # run_leg <home-dir> [KEY=VALUE ...]  -> background pid
  local home_dir="$1"; shift
  gate_refuse_existing "$home_dir"
  mkdir -p "$home_dir"
  python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    flock -x -w 900 "$GPU_LOCK" timeout -k 60 2400 \
    env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" HF_HOME="$HF_CACHE" \
    "$@" "$GATE_HOME/.local/bin/mlx-omarchy-chat" --home "$home_dir" --no-browser --pair everyday --yes \
    >"$LOG.server-$(basename "$home_dir")" 2>&1 &
  echo $!
}

kill_leg() { # kill_leg <home-dir> <pid>
  local home="$1" pid="$2" port
  port=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' \
    "$home/assistant/application.json" 2>/dev/null)
  [ -n "$port" ] && fuser -k -n tcp "$port" 2>/dev/null
  kill -TERM -- "-$pid" 2>/dev/null
  kill -TERM "$pid" 2>/dev/null
  sleep 3
  kill -KILL -- "-$pid" 2>/dev/null
  sleep 1
  if [ -n "$port" ] && fuser -n tcp "$port" >/dev/null 2>&1; then
    gate_log "$LOG" "LISTENER_LEFT port=$port"
  fi
}

RC=0

echo "== leg A: default (routing expected ON) ==" | tee -a "$LOG"
PA=$(run_leg "$ASSIST_A")
sleep 5
OUT_A=$(python3 "$GATES_DIR/g14-routing-driver.py" "$ASSIST_A" "$LOG")
echo "$OUT_A" | tee -a "$LOG"
MODEL_A=$(echo "$OUT_A" | grep "^MODEL " | awk '{print $2}')
kill_leg "$ASSIST_A" "$PA"

echo "== leg B: MLX_OMARCHY_ROUTING=0 (routing expected OFF) ==" | tee -a "$LOG"
PB=$(run_leg "$ASSIST_B" MLX_OMARCHY_ROUTING=0)
sleep 5
OUT_B=$(python3 "$GATES_DIR/g14-routing-driver.py" "$ASSIST_B" "$LOG")
echo "$OUT_B" | tee -a "$LOG"
MODEL_B=$(echo "$OUT_B" | grep "^MODEL " | awk '{print $2}')
kill_leg "$ASSIST_B" "$PB"

gate_log "$LOG" "MODEL_A $MODEL_A"
gate_log "$LOG" "MODEL_B $MODEL_B"
if [[ -n "$MODEL_A" && "$MODEL_A" != *"everyday"* && "$MODEL_A" != "unrecorded" ]]; then
  gate_log "$LOG" "ROUTE_ON PASS (routed head answered: $MODEL_A)"
else
  gate_log "$LOG" "ROUTE_ON FAIL (model_a=${MODEL_A:-none} — routing default not shipped, or the head is not the laya lane, or the answer record carries no model)"
  RC=1
fi
if [[ -z "$MODEL_B" || "$MODEL_B" == *"everyday"* ]]; then
  gate_log "$LOG" "KILL_SWITCH PASS (main pair answered: ${MODEL_B:-main-pair})"
else
  gate_log "$LOG" "KILL_SWITCH FAIL (model_b=$MODEL_B — the kill switch did not stop routing)"
  RC=1
fi
gate_log "$LOG" "GATE14_EXIT $RC"
exit "$RC"
