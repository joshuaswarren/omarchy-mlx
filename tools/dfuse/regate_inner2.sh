#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: 4B default-ON + 2B cross-check digest runs.
set -euo pipefail
C=/var/tmp/dfuse-cand/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
HF=${HOME}/.cache/huggingface/hub
M4B=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)
M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/regate
mkdir -p "$O"

run() { # py model depth label
  timeout 300 "$1" "$B" --model "$2" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$3" --prefill-tokens 512 --label "$4" --out "$O/$4.json" \
    > "$O/$4.log" 2> "$O/$4.err"
  echo "regate $4: $(grep -o '"ordered_records_sha256[^,}]*' "$O/$4.json")"
}

run "$C" "$M4B" 64 regate-4b-d64
run "$C" "$M2B" 64 regate-2b-d64
run "$C" "$M4B" 512 regate-4b-d512
run "$C" "$M2B" 512 regate-2b-d512
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
