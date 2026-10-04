#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: per-chip policy deploy verify (jw16 = G14).
set -euo pipefail
PY=/var/tmp/v072-venv-fused/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
M2B=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/deploy-verify-pc
mkdir -p "$O"
"$PY" -c "import mlx.core as mx; print('serving:', mx.__version__)"

run() { # model depth label
  timeout 600 "$PY" "$B" --model "$1" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$2" --prefill-tokens 512 --label "pc-$3" --out "$O/pc-$3.json" \
    > "$O/pc-$3.log" 2> "$O/pc-$3.err"
  echo "verify $3: $(grep -o '"ordered_records_sha256[^,}]*' "$O/pc-$3.json") rate=$(python3 -c "import json;print(round(json.load(open('$O/pc-$3.json'))['decode_tok_rate']['median'],2))")"
}

run "$M9B" 64 9b-d64
run "$M9B" 512 9b-d512
run "$M2B" 64 2b-d64
run "$M2B" 512 2b-d512
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
