#!/usr/bin/env bash
# HwProbe incremental wheel build on the M2 (private copy of the od tree).
# Invoked detached from the dev box; logs to /var/tmp/hwprobe-wheel/build-hwprobe.log.
set -euo pipefail
cd /var/tmp/hwprobe-wheel
exec nice -n 19 ionice -c3 env \
  CMAKE_BUILD_PARALLEL_LEVEL=3 \
  DEV_RELEASE=1 \
  MLX_OMARCHY_WHOLE_BUNDLE_DIR=/var/tmp/od-distributed-wheel-20261004/.work/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole \
  bash scripts/build-wheel.sh