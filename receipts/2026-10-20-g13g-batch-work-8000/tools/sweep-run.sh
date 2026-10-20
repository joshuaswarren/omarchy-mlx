#!/bin/bash
# 13-inch M1 TIMING BATCH_WORK sweep, sequential, for the guard backlog: 5000 vs 10000, 15000 vs 20000 on venv-release (v0.7.33 stack). Needs a quiet SoC.
set -u
D=$WORK
cd "$D" || exit 1
TAG=$(date -u +%m%dT%H%M%S)
for _ in $(seq 1 24); do [ "$(cut -d. -f1 /proc/loadavg)" -le 1 ] && break; sleep 20; done
echo "SWEEP start $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
bash ledgerab.sh "$D/s6-bw5000-10000-$TAG" "bw5000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=5000" "bw10000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=10000" 5
bash ledgerab.sh "$D/s6-bw15000-20000-$TAG" "bw15000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=15000" "bw20000,venv-release,mlx-lm-series-release,MLX_OMARCHY_BATCH_WORK=20000" 5
