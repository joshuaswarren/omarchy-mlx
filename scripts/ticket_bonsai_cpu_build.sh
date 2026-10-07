#!/usr/bin/env bash
# BonsaiBuild ticket 1 (CPU, run under: fill-run -- nice -n 19 ...): build
# libmlx.so + the omarchy ctest targets from origin/main and run the three
# dispatch-touching suites. No GPU, no wheel, no bundle needed.
# Usage (on jw16): scripts/ticket_bonsai_cpu_build.sh [tree-dir]
set -euo pipefail

TREE=${1:-/var/tmp/bonsai-cpu}
LOG="$TREE/ticket.log"
mkdir -p "$TREE"
exec > >(tee -a "$LOG") 2>&1

echo "== bonsai cpu build ticket start $(date -u +%FT%TZ)"
echo "tree=$TREE"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "kernel=$(uname -r)"
echo "uname=$(uname -a)"

if [ ! -d "$TREE/repo/.git" ]; then
  git clone https://github.com/joshuaswarren/omarchy-mlx.git "$TREE/repo"
fi
git -C "$TREE/repo" fetch origin
git -C "$TREE/repo" checkout --detach origin/main
COMMIT=$(git -C "$TREE/repo" rev-parse HEAD)
echo "commit=$COMMIT"

cd "$TREE/repo"
export CMAKE_INCLUDE_PATH=/usr/include/openblas
bash scripts/prepare-mlx.sh

cmake -S .work/mlx -B .work/build -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build .work/build --target mlx \
  omarchy_primitive_tests omarchy_runtime_tests omarchy_matmul_family_tests

echo "== ctest =="
ctest --test-dir .work/build -R \
  "omarchy_(primitive|runtime|matmul_family)_tests" --output-on-failure

echo "== bonsai cpu build ticket PASS commit=$COMMIT $(date -u +%FT%TZ)"
