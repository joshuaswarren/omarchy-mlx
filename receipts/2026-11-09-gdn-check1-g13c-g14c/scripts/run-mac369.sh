#!/usr/bin/env bash
# Issue 369 check-1 run on a lab Mac (G13C on the 16-inch, G14C on the 14-inch M2 Max).
# Fetches branch w7q/issue369-check1, prepares the pinned MLX tree, builds
# omarchy_gdn_gqa_bf16a_tests, runs the new check-1 case. No file outside
# the output directory and the worktree changes.
set -euo pipefail
O=${1:?out dir}
W=$HOME/work/i369
[ -e "$O" ] && { echo "abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "$(hostname) $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
R=$HOME/src/mlx-omarchy
if [ ! -d "$R/.git" ]; then R=$HOME/work/mlx-omarchy; fi
if [ ! -d "$R/.git" ]; then
  say "cloning omarchy-mlx into $R"
  mkdir -p "$HOME/src"
  git clone -q https://github.com/joshuaswarren/omarchy-mlx.git "$R"
fi
cd "$R"
git fetch -q origin main
git worktree remove --force "$W" 2>/dev/null || true
git worktree add -q "$W" origin/main
say "tree origin/main $(git -C "$W" rev-parse --short HEAD)"
cd "$W"
bash scripts/prepare-mlx.sh > "$O/prep.log" 2>&1
mkdir -p build369
cmake -S .work/mlx -B build369 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_OMARCHY_GPU_PROFILING=OFF -DMLX_USE_CCACHE=OFF > "$O/cmake.log" 2>&1
cmake --build build369 --parallel 8 --target omarchy_gdn_gqa_bf16a_tests > "$O/build.log" 2>&1
BIN=$(find build369 -name omarchy_gdn_gqa_bf16a_tests -type f | head -1)
say "built $BIN $(sha256sum "$BIN" | cut -c1-16)"
"$BIN" --test-case='*per-op fp64*' | tee "$O/check1.out"
say "end rc=$? $(date -u +%FT%TZ)"
