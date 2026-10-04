#!/usr/bin/env bash
# Production-pin check: 2B 5-pass cells vs the standing Fuse6 pins.
set -euo pipefail
C=/var/tmp/dfuse-cand/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M2B=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/regate/pins
mkdir -p "$O"

for depth in 64 128 256 512; do
  timeout 600 "$C" "$B" --model "$M2B" --prompts "$P" --limit 1 --warmup 1 \
    --passes 5 --new-tokens "$depth" --prefill-tokens 512 --label "pin2b-d$depth" \
    --out "$O/pin2b-d$depth.json" > "$O/pin2b-d$depth.log" 2> "$O/pin2b-d$depth.err"
  echo "2B d$depth 5-pass: $(grep -o '"ordered_records_sha256[^,}]*' "$O/pin2b-d$depth.json")"
done
