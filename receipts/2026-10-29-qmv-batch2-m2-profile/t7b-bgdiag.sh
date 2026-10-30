#!/usr/bin/env bash
# Wrapper: batch decode A/B/C (mlx_lm BatchGenerator, B = 2, 3, 4 rows, greedy, 64 tokens, Qwen3.8-2B 4-bit) on one Mac.
#   t7b-bgdiag.sh hostB|m2|hostC
# A = ~/t7/venv-diag-s (a736744a, shipped kernel), B = ~/t7/venv-diag-h (42b30a2d, whole-word hoist); diagnostics wheels run unprofiled (no MLX_OMARCHY_GPU_PROFILE).
# Three interleaved pairs per B (the order rotates); every run is its own process; the row ids hash must be equal in all runs of one B.
# GPU, no download, cap 25 min (expect 8). Apple Honeykrisp ICD only.
set -u
HOST=${1:?host}
case "$HOST" in
  hostB) MODEL="$HOME/scratch/model/q38-2b" ;;
  m2|hostC) MODEL=$(ls -d "$HOME"/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/* | head -1) ;;
  *) echo "host?"; exit 64 ;;
esac
OUT="$HOME/t7/results"; mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%MZ); LOG="$OUT/bgdiag-$HOST-$STAMP.log"
echo "bgdiag $HOST $STAMP load $(cut -d' ' -f1 /proc/loadavg) kernel $(uname -r)" | tee "$LOG"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/asahi_icd.json HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
for B in 2 3 4; do
  for pair in 1 2 3; do
    case $pair in 1) ORDER="A B" ;; 2) ORDER="B A" ;; 3) ORDER="A B" ;; esac
    for arm in $ORDER; do
      case $arm in A) PY="$HOME/t7/venv-diag-s/bin/python" ;; B) PY="$HOME/t7/venv-diag-h/bin/python" ;; esac
      echo "-- B=$B pair $pair arm $arm" | tee -a "$LOG"
      timeout 240 "$PY" "$HOME/t7/bg_decode_ab.py" "$MODEL" "$B" 64 5 2>&1 | grep -E "^(PROVENANCE|BGDECODE)|Error|Traceback" | cut -c1-300 | tee -a "$LOG"
    done
  done
done
echo "done $(date -u +%T)" | tee -a "$LOG"
