#!/bin/bash
# LayaHead2 ticket 7: rope-table lever A/B. Same cap wheel (private venv),
# cap40000 both arms, float16; arms differ only by serve tree: base
# (per-layer cos/sin, 56 trig-gate joins per forward) vs rope (tables built
# once at load). Round 1 sweeps the 154-case dev set per arm.
# usage: run_ticket7.sh OUT_DIR SERVE_BASE SERVE_ROPE SUITE MODEL PORT
set -euo pipefail

OUT=$1; BASE=$2; ROPE=$3; SUITE=$4; MODEL=$5; PORT=${6:-50518}
VENV=$HOME/scratch/laya2/venv-cap
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
VPY="$VENV/bin/python"
"$VPY" -c 'import mlx.core as mx; print(mx.__version__)' > "$OUT/wheel-cap.txt"
grep -q '4b2929a' "$OUT/wheel-cap.txt" || { echo "wrong wheel stamp" >&2; exit 5; }
if cmp -s "$BASE/mlx_omarchy_laya/model.py" "$ROPE/mlx_omarchy_laya/model.py"; then
  echo "arms identical: model.py does not differ" >&2; exit 6
fi

export PY="$VPY"
export SWEEP=1
"$SCRIPTS/run_ab.sh" "$OUT" "$BASE" "$SUITE" "$MODEL" "$PORT" 2 30 \
  "base:LAYA_SERVE=$BASE,MLX_OMARCHY_BATCH_WORK=40000" \
  "rope:LAYA_SERVE=$ROPE,MLX_OMARCHY_BATCH_WORK=40000"
sync
echo "[ticket7] DONE"
