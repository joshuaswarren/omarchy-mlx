#!/usr/bin/env bash
# Real-model first-forward repro on the M2 omarchy wheel.
set -u
PY=$HOME/v0.7.30-gates/v0.7.31-gate-home-3/.local/share/mlx-omarchy/venv/bin/python
MODEL=$HOME/.cache/huggingface/hub/models--SiddhJagani--Qwen3.5-9B-MLX-4bit/snapshots
MODEL=$(ls -d "$MODEL"/*/ 2>/dev/null | head -1)
echo "MODEL=$MODEL"
echo "=== small prompt (first prefill + 8 decode steps)"
timeout -k 20 240 "$PY" -m mlx_lm generate --model "$MODEL" \
  --prompt "Hello" --max-tokens 4 --temp 0 2>&1 | tail -15
echo "rc=$?"
