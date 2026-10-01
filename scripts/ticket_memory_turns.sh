#!/usr/bin/env bash
# PairGates memory/perf ticket via pair_memory_turns.py (assistant API).
# Small-ticket testing per Main: compact4b first (~2 min load), then the
# 9B, then the 27B (long ticket).
set -uo pipefail

PAIR_ID=${1:?pair id required}
HOME_DIR=${2:?home dir required}
LABEL=${3:?label required}
ART_DIR=${4:?artifact dir required}
MAXM=${5:-15}
WORKTREE=<home>/agents/PairGates/main-checkout
VENV=<home>/.local/share/mlx-omarchy/venv/bin/python

mkdir -p "$ART_DIR"
python3 /tmp/reset_pairgates_home.py "$HOME_DIR" 2>/dev/null || true

LOAD_BEFORE=$(cat /proc/loadavg)
echo "loadavg_before=$LOAD_BEFORE" | tee "$ART_DIR/launch.log"
uname -r | tee -a "$ART_DIR/launch.log"
GPU_USERS_BEFORE=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
echo "fuser_before=$GPU_USERS_BEFORE" | tee -a "$ART_DIR/launch.log"

PYTHONPATH=$WORKTREE/scripts:$WORKTREE/serve \
  MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 \
  $VENV -u $WORKTREE/scripts/pair_memory_turns.py \
  --pair "$PAIR_ID" --home "$HOME_DIR" \
  --repo-serve $WORKTREE/serve \
  --label "$LABEL" \
  --out "$ART_DIR/measure.json" \
  2>&1 | tee -a "$ART_DIR/harness.log"
RC=$?
echo "pair_memory_turns rc=$RC" | tee -a "$ART_DIR/launch.log"

# Clean up anything left resident.
pkill -KILL -f "mlx_omarchy_assistant.*PairGates" 2>/dev/null || true
pkill -KILL -f "_mlxlm_server.*PairGates" 2>/dev/null || true
pkill -KILL -f "mlx_omarchy_laya.*PairGates" 2>/dev/null || true

LOAD_AFTER=$(cat /proc/loadavg)
GPU_USERS_AFTER=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
echo "loadavg_after=$LOAD_AFTER fuser_after=$GPU_USERS_AFTER" | tee -a "$ART_DIR/launch.log"

if [ -s "$ART_DIR/measure.json" ]; then
    touch "$ART_DIR/COMPLETE"
    echo "DONE: $ART_DIR"
else
    echo "FAILED: no measure.json written"
    exit 1
fi