#!/bin/bash
# 13-inch M1 TIMING: BATCH_WORK 5000 vs 8000 on the Qwen3-4B (ledger step, 5 pairs). Aborts (rc 3) if load1 stays above 1.3 for 8 minutes, so a loaded machine cannot produce a number.
set -u
D=$WORK
cd "$D" || exit 1
TAG=$(date -u +%m%dT%H%M%S)
for _ in $(seq 1 24); do awk '{exit !($1<=1.3)}' /proc/loadavg && break; sleep 20; done
awk '{exit !($1<=1.3)}' /proc/loadavg || { echo "BW8000-4B ABORT load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 3; }
echo "BW8000-4B start $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
MODEL_SUBDIR=qwen3-4b-4bit bash ledgerab.sh "$D/s8-4b-$TAG" "bw5000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=5000" "bw8000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=8000" 5
