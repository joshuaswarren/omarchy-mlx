#!/usr/bin/env bash
# Host-only schema, adapter, and ANEC header validation gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${MLX_OMARCHY_BUILD_DIR:-$ROOT/.work/build}"

python3 "$ROOT/overlay/tests/omarchy/ane/test_h13_package_to_bundle.py"
"$ROOT/scripts/prepare-mlx.sh" >/dev/null

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S "$ROOT/.work/mlx" -B "$BUILD_DIR" \
    -DMLX_BUILD_OMARCHY=ON \
    -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_TESTS=ON \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF
fi

cmake --build "$BUILD_DIR" --target omarchy_ane_bundle_tests -j4
"$BUILD_DIR/tests/omarchy/omarchy_ane_bundle_tests"
echo "[receipt] schema-3 adapter and bundle validation passed on $(uname -m)"
