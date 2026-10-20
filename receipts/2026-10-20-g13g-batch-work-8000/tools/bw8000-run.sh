#!/bin/bash
# 13-inch M1 TIMING: BATCH_WORK 5000 vs 8000 on the v0.7.33 stack, ledger LLM step, for the 2B and the Qwen3-4B (5 pairs each). Waits for load1 <= 1 up to 8 min.
set -u
D=$WORK
cd "$D" || exit 1
TAG=$(date -u +%m%dT%H%M%S)
for _ in $(seq 1 24); do [ "$(cut -d. -f1 /proc/loadavg)" -le 1 ] && break; sleep 20; done
echo "BW8000 start $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
MODEL_SUBDIR=qwen3_5-4bit bash ledgerab.sh "$D/s7-2b-$TAG" "bw5000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=5000" "bw8000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=8000" 5
MODEL_SUBDIR=qwen3-4b-4bit bash ledgerab.sh "$D/s7-4b-$TAG" "bw5000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=5000" "bw8000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=8000" 5
