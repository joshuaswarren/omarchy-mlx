#!/usr/bin/env bash
# PairGates memory ticket v2 (heartbeat on, max_tokens=700 for cards,
# per-phase validity check). Usage:
#   ticket_memory_v2.sh <pair_id> <home> <label> <artifact_dir> <minutes>
set -uo pipefail
PAIR_ID=${1:?pair id required}
HOME_DIR=${2:?home dir required}
LABEL=${3:?label required}
ART_DIR=${4:?artifact dir required}
MAXM=${5:-15}
WORKTREE=$HOME/agents/PairGates/main-checkout
VENV=$HOME/.local/share/mlx-omarchy/venv/bin/python

mkdir -p "$ART_DIR"
mkdir -p "$HOME_DIR/assistant/logs" "$HOME_DIR/assistant/pair-locks" "$HOME_DIR/models"

LOAD_BEFORE=$(cat /proc/loadavg)
uname_r=$(uname -r)
GPU_USERS_BEFORE=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
echo "loadavg_before=$LOAD_BEFORE uname_r=$uname_r fuser_before=$GPU_USERS_BEFORE" | tee "$ART_DIR/launch.log"

PYTHONPATH=$WORKTREE/scripts:$WORKTREE/serve \
  MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 \
  $VENV -u $WORKTREE/scripts/pair_memory_v2.py \
  --pair-id "$PAIR_ID" --home "$HOME_DIR" \
  --repo-serve $WORKTREE/serve \
  --label "$LABEL" \
  --out "$ART_DIR/measure.json" \
  2>&1 | tee -a "$ART_DIR/harness.log"
RC=$?
echo "pair_memory_v2 rc=$RC" | tee -a "$ART_DIR/launch.log"

pkill -KILL -f "mlx_omarchy_assistant" 2>/dev/null || true
pkill -KILL -f "_mlxlm_server" 2>/dev/null || true
pkill -KILL -f "mlx_omarchy_laya" 2>/dev/null || true

LOAD_AFTER=$(cat /proc/loadavg)
GPU_USERS_AFTER=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
echo "loadavg_after=$LOAD_AFTER fuser_after=$GPU_USERS_AFTER" | tee -a "$ART_DIR/launch.log"

if [ -s "$ART_DIR/measure.json" ]; then
    touch "$ART_DIR/COMPLETE"
    echo "DONE: $ART_DIR"
    exit 0
else
    echo "FAILED: no measure.json"
    exit 1
fi