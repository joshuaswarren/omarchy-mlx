#!/usr/bin/env bash
# BF16 decode dispatch census window (lock held by caller).
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
PY=.work/venv-bf16dec/bin/python
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

export MLX_OMARCHY_GPU_PROFILE=/tmp/bf16dec-census-short.jsonl
"$PY" scripts/profile_generate.py \
  --model ~/models/Qwen2.5-0.5B-Instruct-bf16-mlx \
  --prompt "Hi" --max-tokens 32 --temp 0 --seed 0 \
  --markers /tmp/bf16dec-census-short-markers.jsonl \
  > /tmp/bf16dec-census-short.log 2>&1

"$PY" - <<'PYEOF'
import json
m = json.load(open("scripts/bench_matrix.json"))
e = m["prompts"]["ctx1024"]
parts = [e["base"]] + [
    "%s Entry %d of %d." % (e["item"], i, e["items"])
    for i in range(1, e["items"] + 1)]
open("/tmp/ctx1024-prompt.txt", "w").write(" ".join(parts))
PYEOF

export MLX_OMARCHY_GPU_PROFILE=/tmp/bf16dec-census-ctx.jsonl
"$PY" scripts/profile_generate.py \
  --model ~/models/Qwen2.5-0.5B-Instruct-bf16-mlx \
  --prompt "$(cat /tmp/ctx1024-prompt.txt)" --max-tokens 32 --temp 0 --seed 0 \
  --markers /tmp/bf16dec-census-ctx-markers.jsonl \
  > /tmp/bf16dec-census-ctx.log 2>&1

echo CENSUS_RUNS_DONE
