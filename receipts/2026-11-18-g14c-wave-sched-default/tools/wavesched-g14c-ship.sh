#!/usr/bin/env bash
# G14C WAVE_SCHED ship gate: 3 decode runs per arm, step logprobs on 2 models.
set -euo pipefail
export MLX_DISABLE_COMPILE=1
PY=/usr/lib/omarchy-mlx/venv/bin/python
BD="$HOME/scratch/w7Q/bench_decode.py"
TOOLS="$HOME/scratch/w7Q/g14c-wave-sched-ship"
SC="$HOME/scratch/w7Q"
OUT="${OUT:-$HOME/scratch/w7Q/g14c-wave-sched-ship/run-$(date -u +%Y%m%dT%H%MZ)}"
mkdir -p "$OUT"
# Packaged hosts have no standalone .whl; gdn_logits omits --wheel (installed RECORD only).
PROMPT='Tell me about the history of computing in three short paragraphs: the early mechanical era, the transistor era, and the modern GPU era.'
M4B="${MODEL_4B:-$(find "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)}"
M2B="${MODEL_2B:-$(
  find "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.8-2B-4bit/snapshots" \
    "$HOME/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1
)}"
if ! test -d "$M4B" || ! test -d "$M2B"; then
  echo "missing models M4B=$M4B M2B=$M2B"
  exit 2
fi
run_bench() {
  local model="$1" tag="$2" envpair="$3" i="$4"
  ( unset MLX_OMARCHY_WAVE_SCHED; eval "export $envpair"
     "$PY" "$BD" --model "$model" --prompt "$PROMPT" --tokens 128 --temp 0 --seed 0 --warmup-tokens 8
  ) 2>&1 | tee "$OUT/bench-${tag}-r${i}.log" | grep -E '^decode |generated_ids'
}
for model_key in 4B 2B; do
  eval "model=\$M${model_key}"
  for i in 1 2 3; do
    run_bench "$model" "${model_key}-off" "MLX_OMARCHY_WAVE_SCHED=0" "$i"
    run_bench "$model" "${model_key}-on" "MLX_OMARCHY_WAVE_SCHED=1" "$i"
  done
  for arm in off on; do
    envpair="MLX_OMARCHY_WAVE_SCHED=$([ "$arm" = on ] && echo 1 || echo 0)"
    ( unset MLX_OMARCHY_WAVE_SCHED; eval "export $envpair"
      "$PY" "$TOOLS/gdn_logits.py" --model "$model" --prompt "$PROMPT" --tokens 128 --full-steps 32 \
        --out "$OUT/logits-${model_key}-${arm}" --scripts "$SC"
    ) | tee "$OUT/logits-${model_key}-${arm}.log"
  done
  "$PY" "$TOOLS/gdn_numcmp.py" "$OUT/logits-${model_key}-off" "$OUT/logits-${model_key}-on" | tee "$OUT/numcmp-${model_key}.txt"
done
echo "SHIP_OUT=$OUT"
