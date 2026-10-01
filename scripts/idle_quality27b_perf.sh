#!/usr/bin/env bash
# PairGates idle-GPU Quality 27B perf rerun. HELD until Main announces
# a quiet render node: aborts unless fuser /dev/dri/renderD128 is
# empty before AND after (after killing only MY own leftovers),
# and records loadavg either way.
# Usage: idle_quality27b_perf.sh <artifact_dir> <minutes>
set -uo pipefail
ART=${1:?artifact dir required}
MAXM=${2:-45}
WORKTREE=$HOME/agents/PairGates/main-checkout
VENV=$HOME/.local/share/mlx-omarchy/venv/bin/python
HOME_DIR=$HOME/mlx-assistant-runs/quality-home
source "$WORKTREE/scripts/ticket_guard.sh"

mkdir -p "$ART"

# Kill MY leftovers from earlier aborted tickets first, then gate.
guard_kill_leftovers "$HOME_DIR"
GPU_USERS=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
LOAD_BEFORE=$(cat /proc/loadavg)
uname_r=$(uname -r)
{
    echo "loadavg_before=$LOAD_BEFORE uname_r=$uname_r fuser_before=$GPU_USERS"
    if [ -n "$GPU_USERS" ]; then
        echo "REFUSED: render node busy with non-PairGates work (fuser=$GPU_USERS); idle-only run"
        exit 1
    fi
} | tee "$ART/launch.log"

# Watchdog: kill the 27B assistant tree if this ticket dies on ANY path.
guard_start_watchdog "$HOME_DIR"

# Cooperative cleanup on normal exits.
trap 'ps -eo pid,cmd | grep -E "mlx_omarchy_assistant.*quality-home|_mlxlm_server.*quality-home|mlx_omarchy_laya.*quality-home" | awk "{print \$1}" | xargs -r kill -9 2>/dev/null' EXIT

PYTHONPATH=$WORKTREE/scripts:$WORKTREE/serve \
  MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 \
  $VENV -u $WORKTREE/scripts/quality27b_perf_api.py \
  --home "$HOME_DIR" \
  --repo-serve $WORKTREE/serve \
  --label "quality27b-idle" \
  --out "$ART/quality27b-idle.json" \
  2>&1 | tee -a "$ART/harness.log"
RC=${PIPESTATUS[0]}
echo "quality27b_perf_api rc=$RC" | tee -a "$ART/launch.log"

guard_kill_leftovers "$HOME_DIR"
LOAD_AFTER=$(cat /proc/loadavg)
GPU_USERS_AFTER=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
{
    echo "loadavg_after=$LOAD_AFTER"
    if [ -n "$GPU_USERS_AFTER" ]; then
        echo "WARNING: render node busy after run (fuser=$GPU_USERS_AFTER); timings are NOT idle-clean"
    else
        echo "fuser_after= (empty)"
    fi
} | tee -a "$ART/launch.log"

if [ -s "$ART/quality27b-idle.json" ] && [ -z "$GPU_USERS_AFTER" ]; then
    touch "$ART/COMPLETE"
    echo "DONE: $ART"
    exit 0
fi
echo "FAILED: no clean idle result"
exit 1
