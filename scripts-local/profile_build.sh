#!/usr/bin/env bash
# Profile build (MLX_OMARCHY_GPU_PROFILING=ON) for GDN prefill measurement.
# NOT for performance; carries GPU-profiler dispatch barriers.
set -euo pipefail
ROOT="~/.config/superpowers/worktrees/mlx-omarchy/GdnPrefill"
WORK="$ROOT/.work"
BUILD="$WORK/build-profile"
mkdir -p "$BUILD"
cd "$BUILD"
cmake -G Ninja "$WORK/mlx" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_OMARCHY_GPU_PROFILING=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_SHARED_LIBS=OFF \
  -DBUILD_SHARED_LIBS=OFF
ninja -C "$BUILD" -j8
