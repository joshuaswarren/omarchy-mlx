#!/usr/bin/env bash
# Wrapper (13-inch M1): pre-registered check H3 (short contexts do not regress), one build (venv-main), arms A (split route off) and B (default),
# decode-profile-h3.py at contexts 128 and 256 for the 2B and the 9B, interleaved A,B,A,B per model. GPU, cap 20 min (expect 8). No download.
set -u
PY="${AB_VENV:-$HOME/t7/venv-main}/bin/python"
OUT="$HOME/t7/results"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1
STAMP=$(date -u +%Y%m%dT%H%MZ)
LOG="$OUT/h3-$STAMP.log"
echo "H3 short-context check $STAMP load $(cut -d' ' -f1 /proc/loadavg)" | tee "$LOG"
for model in SiddhJagani/Qwen3.8-2B-mlx-4Bit mlx-community/Qwen3.5-9B-4bit; do
  for rep in 1 2; do
    for arm in A B; do
      if [ "$arm" = A ]; then export MLX_OMARCHY_SDPA_DECODE_SPLIT=0; else unset MLX_OMARCHY_SDPA_DECODE_SPLIT; fi
      echo "== $model rep $rep arm $arm" | tee -a "$LOG"
      timeout 420 "$PY" "$HOME/t7/decode-profile-h3.py" "$model" 128 256 2>&1 | grep -E "^(DECODE .*tok/s \(|PROVENANCE|Traceback|[A-Za-z]*Error)" | cut -c1-260 | tee -a "$LOG"
    done
  done
done
