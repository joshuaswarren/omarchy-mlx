#!/usr/bin/env bash
# PairGates per-pair card timing on ONE boot (4B, 9B, 27B), quiet window.
# Usage: ticket_card_timing.sh <artifact_dir> <minutes>
set -uo pipefail
ART=${1:?artifact dir required}
MAXM=${2:-30}
WORKTREE=$HOME/agents/PairGates/main-checkout
VENV=$HOME/.local/share/mlx-omarchy/venv/bin/python
H=$HOME/agents/PairGates/homes
source "$WORKTREE/scripts/ticket_guard.sh"

mkdir -p "$ART"

# Kill MY leftovers from any earlier aborted ticket first, then gate.
guard_kill_leftovers "$H/compact4b" "$H/everyday9b" "$H/quality27b"
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

# Watchdog: if this ticket dies on ANY path (incl. SIGKILL of the
# wrapper), kill the whole assistant tree for every home we touch.
guard_start_watchdog "$H/compact4b" "$H/everyday9b" "$H/quality27b"

# Cooperative cleanup still runs on normal exits (EXIT fires on TERM).
trap 'ps -eo pid,cmd | grep -E "mlx_omarchy_assistant.*agents/PairGates|_mlxlm_server.*agents/PairGates|mlx_omarchy_laya.*agents/PairGates" | awk "{print \$1}" | xargs -r kill -9 2>/dev/null' EXIT

run_pair() {
    local pair_id=$1 home_dir=$2 label=$3
    PYTHONPATH=$WORKTREE/scripts:$WORKTREE/serve \
      MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 \
      $VENV -u $WORKTREE/scripts/card_timing_boot.py \
      --pair "$pair_id" --home "$home_dir" \
      --repo-serve $WORKTREE/serve \
      --label "$label" \
      --out "$ART/card-$label.json" \
      2>&1 | tee -a "$ART/harness.log"
}

run_pair compact "$H/compact4b" compact4b
run_pair everyday "$H/everyday9b" everyday9b
run_pair quality "$H/quality27b" quality27b

guard_kill_leftovers "$H/compact4b" "$H/everyday9b" "$H/quality27b"
GPU_USERS_AFTER=$(fuser /dev/dri/renderD128 2>/dev/null | tr -d ' ' || true)
LOAD_AFTER=$(cat /proc/loadavg)
{
    echo "loadavg_after=$LOAD_AFTER"
    if [ -n "$GPU_USERS_AFTER" ]; then
        echo "INVALID: render node busy after run (fuser=$GPU_USERS_AFTER)"
    else
        echo "fuser_after= (empty)"
    fi
} | tee -a "$ART/launch.log"

OK=1
for lbl in compact4b everyday9b quality27b; do
    [ -s "$ART/card-$lbl.json" ] || OK=0
done
if [ "$OK" = "1" ] && [ -z "$GPU_USERS_AFTER" ]; then
    touch "$ART/COMPLETE"
    echo "DONE: $ART"
    exit 0
fi
echo "FAILED: incomplete or contaminated"
exit 1
