#!/bin/bash
# 13-inch M1 TIMING: the PR wheel (default budget 8000, no env) against the v0.7.33 release wheel (default 5000), ledger LLM step, Qwen3.8-2B and Qwen3-4B, 5 pairs each.
# Aborts (rc 3) if load1 stays above 1.3 for 8 minutes.
set -u
D=$WORK
cd "$D" || exit 1
TAG=$(date -u +%m%dT%H%M%S)
for _ in $(seq 1 24); do awk '{exit !($1<=1.3)}' /proc/loadavg && break; sleep 20; done
awk '{exit !($1<=1.3)}' /proc/loadavg || { echo "PR-AB ABORT load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 3; }
echo "PR-AB start $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
MODEL_SUBDIR=qwen3_5-4bit bash ledgerab.sh "$D/s9-2b-$TAG" "rel33,venv-release,mlx-lm-series-release,MLX_AB_UNUSED=1" "pr,venv-pr,mlx-lm-series-release,MLX_AB_UNUSED=1" 5
MODEL_SUBDIR=qwen3-4b-4bit bash ledgerab.sh "$D/s9-4b-$TAG" "rel33,venv-release,mlx-lm-series-release,MLX_AB_UNUSED=1" "pr,venv-pr,mlx-lm-series-release,MLX_AB_UNUSED=1" 5
