#!/usr/bin/env bash
# Wrapper (13-inch M1): teacher-forced decode NLL, split route off (A) vs default (B), one build (venv-main), 2B and 9B,
# 3000 prompt tokens then 400 scored decode steps of real text (the repository docs). GPU, cap 30 min (expect 12). No download.
set -u
PY="$HOME/t7/venv-main/bin/python"
OUT="$HOME/t7/results"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1
STAMP=$(date -u +%Y%m%dT%H%MZ)
LOG="$OUT/nll-$STAMP.log"
GLOB="$HOME/scratch/w7q-sdpa/repo/docs/*.md"
echo "NLL check $STAMP load $(cut -d' ' -f1 /proc/loadavg)" | tee "$LOG"
for model in SiddhJagani/Qwen3.8-2B-mlx-4Bit mlx-community/Qwen3.5-9B-4bit; do
  for arm in A B; do
    if [ "$arm" = A ]; then export MLX_OMARCHY_SDPA_DECODE_SPLIT=0; else unset MLX_OMARCHY_SDPA_DECODE_SPLIT; fi
    f="$OUT/nll-$(basename "$model")-$arm-$STAMP.out"
    timeout 900 "$PY" "$HOME/t7/nll-compare.py" "$model" 3000 400 "$GLOB" > "$f" 2>&1
    echo "$model arm $arm rc=$?" | tee -a "$LOG"
    grep -E "^(NLL |NLL_SETUP|Traceback|[A-Za-z]*Error|AssertionError)" "$f" | cut -c1-240 | tee -a "$LOG"
  done
  a=$(grep -m1 "^NLLVEC" "$OUT/nll-$(basename "$model")-A-$STAMP.out" | cut -d' ' -f2-)
  b=$(grep -m1 "^NLLVEC" "$OUT/nll-$(basename "$model")-B-$STAMP.out" | cut -d' ' -f2-)
  python3 - "$a" "$b" <<'PY' | tee -a "$LOG"
import sys
a = [float(x) for x in sys.argv[1].split()]
b = [float(x) for x in sys.argv[2].split()]
n = min(len(a), len(b))
d = [y - x for x, y in zip(a, b)]
mean = sum(d) / n if n else float("nan")
worst = max((abs(v) for v in d), default=float("nan"))
print(f"DELTA B minus A: mean {mean:+.6f} nats/token over {n} steps, largest per-step |delta| {worst:.4f}")
PY
done
