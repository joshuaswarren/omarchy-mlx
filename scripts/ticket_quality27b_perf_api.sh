#!/usr/bin/env bash
# quality27b-perf ticket via the assistant HTTP API (resumes the saved
# pair home if possible, skips 27B setup).
set -uo pipefail

LABEL=${1:-quality27b-perf}
ART=${2:-<home>/.local/share/apple-silicon-lab/artifacts/PairGates/2026-10-01T10-55Z-quality27b-perf}
MAXM=${3:-30}
WORKTREE=<home>/agents/PairGates/main-checkout
HOME_DIR=<home>/agents/PairGates/homes/quality27b
VENV=<home>/.local/share/mlx-omarchy/venv/bin/python

mkdir -p "$ART"

LOAD_BEFORE=$(cat /proc/loadavg)
echo "loadavg_before=$LOAD_BEFORE" | tee "$ART/launch.log"
uname -r | tee -a "$ART/launch.log"
GPU_USERS_BEFORE=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
echo "fuser_before=$GPU_USERS_BEFORE" | tee -a "$ART/launch.log"

PYTHONPATH=$WORKTREE/scripts:$WORKTREE/serve \
  MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 \
  $VENV -u $WORKTREE/scripts/quality27b_perf_api.py \
  --home "$HOME_DIR" \
  --repo-serve $WORKTREE/serve \
  --out "$ART/quality27b-perf.json" \
  2>&1 | tee -a "$ART/harness.log"
RC=$?
echo "quality27b_perf_api rc=$RC" | tee -a "$ART/launch.log"

pkill -KILL -f "mlx_omarchy_assistant.*PairGates" 2>/dev/null || true
pkill -KILL -f "_mlxlm_server.*PairGates" 2>/dev/null || true
pkill -KILL -f "mlx_omarchy_laya.*PairGates" 2>/dev/null || true

LOAD_AFTER=$(cat /proc/loadavg)
GPU_USERS_AFTER=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
echo "loadavg_after=$LOAD_AFTER fuser_after=$GPU_USERS_AFTER" | tee -a "$ART/launch.log"

if [ -s "$ART/quality27b-perf.json" ]; then
    touch "$ART/COMPLETE"
    echo "DONE: $ART"
else
    echo "FAILED: no quality27b-perf.json"
    exit 1
fi