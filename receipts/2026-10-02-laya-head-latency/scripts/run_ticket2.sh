#!/bin/bash
# LayaHead2 ticket 2: mx.compile steady-state screen vs eager control on the
# same fixed text (single shape), same process conditions, plus a short
# eager T-scaling probe at ~256 and ~512 tokens to confirm the fixed-cost
# character. NDJSON resumable; answers must equal the shipped engine path.
# usage: run_ticket2.sh OUT_DIR SERVE SUITE MODEL_DIR LONGTEXT [PORT]
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; LONGTEXT=$5
PY=${PY:-"$HOME/.local/share/mlx-omarchy/venv/bin/python"}
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

state screen-start
"$PY" -c 'import mlx.core as mx; print(mx.__version__)' > "$OUT/wheel.txt"

# 1) eager control on the screen text (n=12, single shape via --compile? no:
#    eager control uses --fixed-text so the only difference is compilation).
PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/head_probe.py" phases \
  --serve "$SERVE" --model "$MODEL" --suite "$LONGTEXT" --out "$OUT/phases-eager.ndjson" \
  --n 12 --warmup 3 --arm eager-ctrl

# 2) compiled steady state on the same text.
PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/head_probe.py" phases --compile \
  --serve "$SERVE" --model "$MODEL" --suite "$LONGTEXT" --out "$OUT/phases-compiled.ndjson" \
  --n 30 --warmup 0 --arm compiled

# 3) eager T-scaling probe: three lengths (~1x, ~2.5x, ~5x the screen text).
#    $6 = length suite JSON; skip when not provided.
if [ $# -gt 5 ] && [ -n "$6" ]; then
  PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/head_probe.py" phases \
    --serve "$SERVE" --model "$MODEL" --suite "$6" --out "$OUT/phases-tscaling.ndjson" \
    --n 9 --warmup 3 --arm tscaling
fi

state screen-end
sync
echo "[ticket2] DONE"
