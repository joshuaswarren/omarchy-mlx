#!/usr/bin/env bash
# Discriminator 2: 2B d512 digest on the diag wheel (my commits, OLD base b8adbc966, no main qmm change).
set -euo pipefail
C=/var/tmp/dfuse-venv/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M2B=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/regate
mkdir -p "$O"
"$C" -c "import mlx.core as mx; print('diag wheel:', mx.__version__)"
timeout 300 "$C" "$B" --model "$M2B" --prompts "$P" --limit 1 --warmup 1 \
  --passes 1 --new-tokens 512 --prefill-tokens 512 --label disc2-2b-d512 \
  --out "$O/disc2-2b-d512.json" > "$O/disc2.log" 2> "$O/disc2.err"
grep -o '"ordered_records_sha256[^,}]*' "$O/disc2-2b-d512.json"
