#!/usr/bin/env bash
# RopeNormBatch dev-box doctest: build omarchy_fast_ops_tests with tests ON
# and run the new per-batch offset case on llvmpipe
# (MLX_OMARCHY_ALLOW_NON_APPLE=1). Scoped, nice'd, single target.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .work/mlx ]; then
  ./scripts/prepare-mlx.sh
fi

if [ ! -f .work/build-doctest/build.ninja ]; then
  cmake -S .work/mlx -B .work/build-doctest -G Ninja \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DCMAKE_BUILD_TYPE=Release
fi

cmake --build .work/build-doctest --target omarchy_fast_ops_tests -j4

BIN=.work/build-doctest/overlay/tests/omarchy/omarchy_fast_ops_tests
if [ ! -x "$BIN" ]; then
  BIN=$(find .work/build-doctest -name omarchy_fast_ops_tests -type f | head -1)
fi
echo "binary: $BIN"

echo "== full fast-ops suite (includes the two rope_rms_norm cases)"
MLX_OMARCHY_ALLOW_NON_APPLE=1 "$BIN"
echo "doctest-suite-ok"
