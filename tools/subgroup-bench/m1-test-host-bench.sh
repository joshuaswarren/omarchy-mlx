#!/usr/bin/env bash
# Build + run the subgroup-vs-tree microbenchmark on m1-test-host.
#
# Reads PROTOCOL.md for the full rationale. This script exists so the
# Main session (and any reviewer who inherits it) gets one concrete
# command and one concrete log path.
#
# Exit path: set -euo pipefail. A glslc compile failure must NOT print
# BENCH_DONE (which it used to, because the brace pipeline returned
# tee's status). pipefail makes the pipeline return the bench's exit
# code, and -e turns the script's status into nonzero. The bench
# itself dies on compile failure; pipefail then propagates that.
set -euo pipefail

# Always run from the worktree root so the relative shader paths
# resolve. The bench prints stderr from glslc/glslangValidator if
# either step fails; do not silence it.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.." || exit 1
WORKTREE="$(pwd)"

# 1. Ensure the staged MLX is fresh; the overlay tree holds the kernels
#    we measure, not the staged ones. (Skipped if .work/mlx already
#    present from the prior session.)
if [[ ! -d .work/mlx ]]; then
  bash scripts/prepare-mlx.sh
fi

# 2. Build the harness. The shaders are compiled by the program
#    itself at startup (see tools/subgroup-bench/bench.cpp), passing
#    --target-env=vulkan1.3 because the subgroup extensions need
#    SPIR-V 1.3.
g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/subgroup-bench tools/subgroup-bench/bench.cpp

LOG=/tmp/subgroup-bench.log
# pipefail: a failure inside the brace group propagates. The bench
# exits nonzero on shader compile failure (see die in bench.cpp), so
# tee returning 0 alone is not enough.
{
  echo "=== bench ==="
  /tmp/subgroup-bench
  echo "=== bench --quick ==="
  /tmp/subgroup-bench --quick
} 2>&1 | tee "$LOG"

echo "BENCH_DONE log=$LOG"
