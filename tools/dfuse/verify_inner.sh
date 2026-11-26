#!/usr/bin/env bash
# Runs INSIDE a gpuwin window. Serving venv, NO gate env: default-ON verify.
set -euo pipefail
PY=/var/tmp/v072-venv-fused/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
HF=${HOME}/.cache/huggingface/hub
M4B=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)
M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/deploy-verify
mkdir -p "$O"
test -z "${MLX_OMARCHY_ROPE_NORM_FUSE:-}" || { echo "gate env set"; exit 1; }

run() { # model depth label
  timeout 300 "$PY" "$B" --model "$1" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$2" --prefill-tokens 512 --label "dv-$3" --out "$O/dv-$3.json" \
    > "$O/dv-$3.log" 2> "$O/dv-$3.err"
  echo "deploy-verify $3: $(grep -o '"ordered_records_sha256[^,}]*' "$O/dv-$3.json")"
}

run "$M4B" 64 4b-d64
run "$M4B" 512 4b-d512
run "$M2B" 64 2b-d64
run "$M2B" 512 2b-d512
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
