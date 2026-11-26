#!/bin/bash
# Three-dispatch dependent-chain microbench, M1 window. Run under
#   flock -w 2400 /tmp/m1-gpu.lock tools/chain-dep-bench/run-m1.sh
# from the repo root. Driver selection via environment (VK_DRIVER_FILES).
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
host="$(hostname)"
echo "host=$host commit=$commit stamp=$stamp VK_DRIVER_FILES=${VK_DRIVER_FILES:-<default>}"
g++ -std=c++17 -O2 -o /tmp/chain-dep-bench tools/chain-dep-bench/bench.cpp -ldl
/tmp/chain-dep-bench
echo "CHAIN_DEP_DONE"
