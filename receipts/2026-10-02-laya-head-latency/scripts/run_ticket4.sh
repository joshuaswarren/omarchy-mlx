#!/bin/bash
# LayaHead2 ticket 4: cap dose refinement (10k/20k/80k) around the 40000
# sweet spot, plus in-process screens of eager-cap40000 and
# compiled-cap40000 interaction on the same cap wheel (private venv, reused
# from ticket 3).
# usage: run_ticket4.sh OUT_DIR SERVE SUITE MODEL WHEEL PORT
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; WHEEL=$5; PORT=${6:-50518}
SHARED_SP=$HOME/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages
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

# 1) in-process: eager with cap40000, then compiled with cap40000.
export MLX_OMARCHY_BATCH_WORK=40000
PYTHONPATH="$SERVE:$SCRIPTS" "$VPY" "$SCRIPTS/head_probe.py" phases \
  --serve "$SERVE" --model "$MODEL" --suite "$SUITE" --out "$OUT/phases-capint.ndjson" \
  --n 12 --warmup 3 --arm eager-cap40000
PYTHONPATH="$SERVE:$SCRIPTS" "$VPY" "$SCRIPTS/head_probe.py" phases --compile \
  --serve "$SERVE" --model "$MODEL" --suite "$SUITE" --out "$OUT/phases-capint.ndjson" \
  --n 15 --warmup 0 --arm compiled-cap40000
unset MLX_OMARCHY_BATCH_WORK

# 2) dose refinement, one round, three arms, n=30 each.
export PY="$VPY"
"$SCRIPTS/run_ab.sh" "$OUT" "$SERVE" "$SUITE" "$MODEL" "$PORT" 1 30 \
  "cap10000:MLX_OMARCHY_BATCH_WORK=10000" \
  "cap20000:MLX_OMARCHY_BATCH_WORK=20000" \
  "cap80000:MLX_OMARCHY_BATCH_WORK=80000"

state end
sync
echo "[ticket4] DONE"
