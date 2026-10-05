#!/usr/bin/env bash
# H290 hardware leg: op-level fused MLP-tail A/B (3 production dispatches vs
# 1 persistent dispatch with 2 in-kernel grid barriers), bit-exact at the
# 2B decode tail shapes, plus the occupancy probe that gates G.
# Run from the checkout root under the GPU lock / gpu-turn ticket:
#   flock -w 2400 /tmp/m1-gpu.lock timeout 900 \
#     bash tools/gridbarrier-bench/tail/run-tail-hw.sh
# Safety: the fused kernel's barrier spin is bounded at 2^15 polls (~10 ms,
# well under the 40 ms firmware cl_context_switch_timeout, HkTurnover input);
# the probe escalates G and stops at the first timeout flag; the exactness
# phase stops at the first timeout flag.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p tools/gridbarrier-bench/tail/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
{
  echo "host=$(hostname) commit=$commit stamp=$stamp icd=${VK_DRIVER_FILES:-<default>}"
  uname -r
  uptime
} | tee "tools/gridbarrier-bench/tail/out/tail-$stamp.session.txt"

g++ -std=c++17 -O2 -o /tmp/gbb-probe tools/gridbarrier-bench/bench.cpp
g++ -std=c++17 -O2 -o /tmp/bench-tail tools/gridbarrier-bench/tail/bench_tail.cpp
O=tools/gridbarrier-bench/tail/out/tail-$stamp.ndjson

# 1. Occupancy ceiling at local 256 (the fused kernel's workgroup size):
#    escalates and stops at the first timeout; the largest clean G is the
#    measured ceiling for this class.
GB_GS="${GB_PROBE_GS:-16,32,48,64,80,96,112,128,160,192,224,256}" \
  /tmp/gbb-probe probe 256 2>&1 | tee -a "$O"

# 2. Fused-tail A/B at a conservative G (override with GB_G once the probe
#    leg has measured the ceiling on this chip).
GB_G="${GB_G:-32}" GB_SETS="${GB_SETS:-16}" GB_ROUNDS="${GB_ROUNDS:-256}" \
  /tmp/bench-tail 2>&1 | tee -a "$O"
echo "RUN_TAIL_DONE" | tee -a "tools/gridbarrier-bench/tail/out/tail-$stamp.session.txt"
