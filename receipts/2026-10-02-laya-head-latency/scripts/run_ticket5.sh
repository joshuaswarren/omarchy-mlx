#!/bin/bash
# LayaHead2 ticket 5: worker dtype A/B on the cap wheel (private venv from
# ticket 3), every arm at MLX_OMARCHY_BATCH_WORK=40000. float16 is the
# shipped control; bfloat16 and float32 reach the cooperative-matrix matmul
# route that float16 never takes. Round 1 also sweeps the 154-case dev set
# per arm: the decision-equality gate.
# usage: run_ticket5.sh OUT_DIR SERVE SUITE MODEL PORT
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; PORT=${5:-50518}
VENV=$HOME/scratch/laya2/venv-cap
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

state() {
  {
    date -u +%FT%TZ
    echo "boot $(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime $(cat /proc/uptime)"
    echo "load $(cat /proc/loadavg)"
    cat /proc/pressure/cpu
  } > "$OUT/host-$1.txt"
}
state start
VPY="$VENV/bin/python"
"$VPY" -c 'import mlx.core as mx; print(mx.__version__)' > "$OUT/wheel-cap.txt"
grep -q '4b2929a' "$OUT/wheel-cap.txt" || { echo "wrong wheel stamp" >&2; exit 5; }

export PY="$VPY"
export SWEEP=1
"$SCRIPTS/run_ab.sh" "$OUT" "$SERVE" "$SUITE" "$MODEL" "$PORT" 2 25 \
  "f16:LAYA_DTYPE=float16,MLX_OMARCHY_BATCH_WORK=40000" \
  "bf16:LAYA_DTYPE=bfloat16,MLX_OMARCHY_BATCH_WORK=40000" \
  "f32:LAYA_DTYPE=float32,MLX_OMARCHY_BATCH_WORK=40000"

state end
sync
echo "[ticket5] DONE"
