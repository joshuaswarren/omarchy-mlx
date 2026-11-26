#!/usr/bin/env bash
# Build the Omarchy runtime slice and run its focused tests.
# This is the U2 runtime gate: tools/ci/run-omarchy-runtime.sh
#
# Release-equivalent configuration: the CPU tensor backend is compiled out,
# so any CPU primitive dispatch is impossible by construction.
#
# On a non-Omarchy development machine (no Apple GPU), export
# MLX_OMARCHY_ALLOW_NON_APPLE=1 to exercise the runtime against a desktop or
# software Vulkan driver. Receipts from such runs must record that the
# device is a development device, not Honeykrisp.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${MLX_OMARCHY_BUILD_DIR:-$ROOT/build-omarchy}"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  "$@"

cmake --build "$BUILD_DIR" --target \
  omarchy_runtime_tests \
  omarchy_copy_offset_tests \
  mlx-omarchy-info \
  -j

echo "== omarchy_runtime_tests =="
"$BUILD_DIR/tests/omarchy/omarchy_runtime_tests"

echo "== omarchy_copy_offset_tests =="
"$BUILD_DIR/tests/omarchy/omarchy_copy_offset_tests"

echo "== mlx-omarchy-info =="
"$BUILD_DIR/tools/mlx-omarchy-info/mlx-omarchy-info"

echo "== mlx-omarchy-info --trace-smoke =="
"$BUILD_DIR/tools/mlx-omarchy-info/mlx-omarchy-info" --trace-smoke

echo "== mlx-omarchy-info --json (receipt) =="
"$BUILD_DIR/tools/mlx-omarchy-info/mlx-omarchy-info" --json
