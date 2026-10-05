#!/usr/bin/env bash
# Dev-box skeleton check on lavapipe: algorithm + correctness only, tiny W
# (lavapipe workgroups are software threads; large co-resident grids are
# deadlock-prone). NO performance claims come from this host.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/gridbarrier-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
g++ -std=c++17 -O2 -o /tmp/gridbarrier-bench tools/gridbarrier-bench/bench.cpp
export GB_DEV=${GB_DEV:-llvmpipe}
{
  echo "dev skeleton $stamp GB_DEV=$GB_DEV"
  /tmp/gridbarrier-bench probe 32
  /tmp/gridbarrier-bench stress 32 2 20000 plain
  /tmp/gridbarrier-bench stress 32 4 20000 plain
} 2>&1 | tee "tools/gridbarrier-bench/out/dev-$stamp.ndjson"
