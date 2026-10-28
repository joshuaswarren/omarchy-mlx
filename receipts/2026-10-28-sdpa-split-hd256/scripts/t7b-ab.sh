#!/usr/bin/env bash
# Wrapper (13-inch M1): A/B of the split-K decode attention, ONE build (side venv ~/t7/venv-main), arms interleaved A,B,A,B,A,B.
#   A = MLX_OMARCHY_SDPA_DECODE_SPLIT=0 (split route off: the fused kernel), B = variable unset (the route's default on this part).
# Cells (GPU, no download, HF_HUB_OFFLINE=1):
#   t7b-ab.sh 9B8K   mlx-community/Qwen3.5-9B-4bit at 8192, 3 pairs (cap 30 min, expect 20)
#   t7b-ab.sh 2B32K  SiddhJagani/Qwen3.8-2B-mlx-4Bit at 32768, 3 pairs (cap 30 min, expect 20)
#   t7b-ab.sh 2B8K   same model at 8192, 3 pairs (cap 20 min, expect 8)
#   t7b-ab.sh BENCH  attention micro-benchmark, bf16, both arms (cap 15 min, expect 6)
# Output: ~/t7/results/ab-<cell>-<UTC>.log (one line per run: arm, prefill and decode tok/s) and ab-<cell>-<arm>-<pair>-<UTC>.json
set -u
CELL=${1:?cell}
PY="${AB_VENV:-$HOME/t7/venv-main}/bin/python"
RUNNER="$HOME/t7/t7b-mlx-omarchy.py"
HOST_LABEL="MacBook Pro 13-inch (M1, 16 GB)"
OUT="$HOME/t7/results"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1
STAMP=$(date -u +%Y%m%dT%H%MZ)
LOG="$OUT/ab-$CELL-$STAMP.log"
echo "ab cell $CELL $STAMP load $(cut -d' ' -f1 /proc/loadavg) mlx $("$PY" -c 'import mlx.core as m; print(m.__file__)')" | tee "$LOG"
arm_env() {
  if [ "$1" = A ]; then export MLX_OMARCHY_SDPA_DECODE_SPLIT=0; else unset MLX_OMARCHY_SDPA_DECODE_SPLIT; fi
}
if [ "$CELL" = BENCH ]; then
  for arm in A B; do
    arm_env "$arm"
    echo "== arm $arm" | tee -a "$LOG"
    SDPA_BENCH_DTYPES=bf16 timeout 400 "$PY" "$HOME/t7/sdpa-decode-bench-bf16.py" 512 2048 8192 16384 32768 2>&1 | grep -E "^(PROVENANCE|ROOF|SDPA|DONE)" | tee -a "$LOG"
  done
  exit 0
fi
case "$CELL" in
  9B8K) MODEL=mlx-community/Qwen3.5-9B-4bit; SIZE=8192 ;;
  2B32K) MODEL=SiddhJagani/Qwen3.8-2B-mlx-4Bit; SIZE=32768 ;;
  2B8K) MODEL=SiddhJagani/Qwen3.8-2B-mlx-4Bit; SIZE=8192 ;;
  *) echo "unknown $CELL" >&2; exit 64 ;;
esac
for pair in 1 2 3; do
  for arm in A B; do
    arm_env "$arm"
    J="$OUT/ab-$CELL-$arm-$pair-$STAMP.json"
    timeout 1500 "$PY" "$RUNNER" --model "$MODEL" --host-label "$HOST_LABEL" --sizes "$SIZE" --rounds 1 --pause 20 --out "$J" 2>&1 | grep -E "^run round|Error" | sed "s/^/pair $pair arm $arm: /" | tee -a "$LOG"
  done
done
