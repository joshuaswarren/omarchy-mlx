#!/usr/bin/env bash
# Wrapper (13-inch M1): greedy token comparison, split route off (A) vs default (B), one build (venv-main), for the 2B and the 9B.
#   t7b-tokcmp.sh      GPU, cap 15 min (expect 4): prints agreement of 48 generated token ids per model. No download.
set -u
PY="${AB_VENV:-$HOME/t7/venv-main}/bin/python"
OUT="$HOME/t7/results"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1
STAMP=$(date -u +%Y%m%dT%H%MZ)
LOG="$OUT/tokcmp-$STAMP.log"
echo "token compare $STAMP load $(cut -d' ' -f1 /proc/loadavg)" | tee "$LOG"
for model in SiddhJagani/Qwen3.8-2B-mlx-4Bit mlx-community/Qwen3.5-9B-4bit; do
  for arm in A B; do
    if [ "$arm" = A ]; then export MLX_OMARCHY_SDPA_DECODE_SPLIT=0; else unset MLX_OMARCHY_SDPA_DECODE_SPLIT; fi
    timeout 400 "$PY" "$HOME/t7/token-compare.py" "$model" 48 > "$OUT/tokcmp-$arm-$STAMP.out" 2>&1
    echo "$model arm $arm rc=$? $(grep -m1 PROMPT_TOKENS "$OUT/tokcmp-$arm-$STAMP.out" | cut -c1-60)" | tee -a "$LOG"
    grep -E "^(TOKENS|TEXT)" "$OUT/tokcmp-$arm-$STAMP.out" | sed "s|^|$arm: |" | cut -c1-400 | tee -a "$LOG"
    cp "$OUT/tokcmp-$arm-$STAMP.out" "$OUT/tokcmp-$(basename "$model")-$arm-$STAMP.out"
  done
  a=$(grep -m1 "^TOKENS" "$OUT/tokcmp-$(basename "$model")-A-$STAMP.out" | cut -d' ' -f2-)
  b=$(grep -m1 "^TOKENS" "$OUT/tokcmp-$(basename "$model")-B-$STAMP.out" | cut -d' ' -f2-)
  python3 - "$a" "$b" <<'PY' | tee -a "$LOG"
import sys
a, b = sys.argv[1].split(), sys.argv[2].split()
same = sum(1 for x, y in zip(a, b) if x == y)
first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
print(f"AGREE {same} of {max(len(a), len(b))} token ids; first difference at index {first}")
PY
done
