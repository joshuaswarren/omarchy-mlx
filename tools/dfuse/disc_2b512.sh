#!/usr/bin/env bash
# Discriminator: 2B d512 digest on the UNTOUCHED serving wheel (b581d5c).
set -euo pipefail
PY=/var/tmp/v072-venv-fused/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
HF=${HOME}/.cache/huggingface/hub
M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/regate
mkdir -p "$O"
"$PY" -c "import mlx.core as mx; print('serving wheel:', mx.__version__)"
timeout 300 "$PY" "$B" --model "$M2B" --prompts "$P" --limit 1 --warmup 1 \
  --passes 1 --new-tokens 512 --prefill-tokens 512 --label disc-2b-d512 \
  --out "$O/disc-2b-d512.json" > "$O/disc-2b-d512.log" 2> "$O/disc-2b-d512.err"
grep -o '"ordered_records_sha256[^,}]*' "$O/disc-2b-d512.json"
