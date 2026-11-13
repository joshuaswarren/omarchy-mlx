#!/bin/bash
# usage: [REPO=<checkout>] [GATE_SED=<sed script applied to the copied device.h>] gate-harness.sh old|new
# Compiles device.h (origin/main for "old", the working tree for "new") together with the body of the width-256 gate TEST_CASE from
# overlay/tests/omarchy/test_sdpa_split.cpp, against a stub mlx/api.h, and runs it. Needs g++ and the Vulkan headers; no GPU.
set -u
mode="$1"
repo=${REPO:-$PWD}
work=/tmp/sdpa-gate-$mode
mkdir -p "$work/inc/mlx/backend/omarchy" "$work/inc/mlx"
printf '#define MLX_API\n' > "$work/inc/mlx/api.h"
if [ "$mode" = old ]; then
  git -C "$repo" show origin/main:overlay/mlx/backend/omarchy/device.h > "$work/inc/mlx/backend/omarchy/device.h"
else
  cp "$repo/overlay/mlx/backend/omarchy/device.h" "$work/inc/mlx/backend/omarchy/device.h"
  if [ -n "${GATE_SED:-}" ]; then
    sed -i "$GATE_SED" "$work/inc/mlx/backend/omarchy/device.h"
    cmp -s "$repo/overlay/mlx/backend/omarchy/device.h" "$work/inc/mlx/backend/omarchy/device.h" && echo "MUTATION DID NOT APPLY"
  fi
fi
start=$(grep -n 'the width-256 value-phase variant needs group 4, V alignment' "$repo/overlay/tests/omarchy/test_sdpa_split.cpp" | head -1 | cut -d: -f1)
{
  echo '#include <cstdio>'
  echo '#include <string>'
  echo '#include "mlx/backend/omarchy/device.h"'
  echo 'using namespace mlx::core;'; echo 'static int fails = 0, checks = 0;'
  echo '#define CHECK(x) do { ++checks; if (!(x)) { ++fails; std::printf("FAIL line %d: %s\n", __LINE__, #x); } } while (0)'
  echo '#define CHECK_FALSE(x) CHECK(!(x))'
  echo '#define TEST_CASE(name) static void gate()'
  sed -n "${start},\$p" "$repo/overlay/tests/omarchy/test_sdpa_split.cpp" | sed -n '1,/^}/p' | sed "1s/^TEST_CASE(.*/static void gate() {/"
  echo 'int main() { gate(); std::printf("checks=%d fails=%d\n", checks, fails); return fails != 0; }'
} > "$work/gate.cpp"
rm -f "$work/gate"
(cd "$work" && g++ -std=c++20 -Iinc gate.cpp -o gate 2>&1 | grep -E 'error' | head -3; test -x gate && ./gate)
