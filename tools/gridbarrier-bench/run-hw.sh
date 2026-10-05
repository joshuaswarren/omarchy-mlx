#!/usr/bin/env bash
# Hardware leg: occupancy probe + barrier-cost sweep + 1e7-crossing stress.
# Run from the checkout root under the GPU lock and a hard timeout, on an
# idle host, e.g.:
#   flock -w 2400 /tmp/m1-gpu.lock timeout 900 \
#     bash tools/gridbarrier-bench/run-hw.sh
# Safety: every kernel spin is bounded (2^17 polls ~ 40 ms -> timeout flag,
# clean exit); a non-co-resident grid reports a timeout instead of hanging.
# After the run: journalctl -k | grep -i 'agx\|GPU timeout' and
# systemctl --failed are recorded by the caller.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/gridbarrier-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
{
  echo "host=$(hostname) commit=$commit stamp=$stamp icd=${VK_DRIVER_FILES:-<default>}"
  uname -r
  uptime
} | tee "tools/gridbarrier-bench/out/hw-$stamp.session.txt"
g++ -std=c++17 -O2 -o /tmp/gridbarrier-bench tools/gridbarrier-bench/bench.cpp
B=/tmp/gridbarrier-bench
O=tools/gridbarrier-bench/out/hw-$stamp.ndjson
# Occupancy + cost: 32- and 128-thread workgroups.
$B probe 32   2>&1 | tee -a "$O"
$B probe 128  2>&1 | tee -a "$O"
# Correctness stress: >=1e7 barrier crossings per cell at the largest
# completing G of each class, plain bindings (the crux), then coherent.
$B stress 128 64  156250 plain    2>&1 | tee -a "$O"   # 1.0e7 crossings
$B stress 128 64  156250 coherent 2>&1 | tee -a "$O"
$B stress 32  128 78125 plain     2>&1 | tee -a "$O"   # 1.0e7 crossings
$B stress 32  128 78125 coherent  2>&1 | tee -a "$O"
echo "RUN_HW_DONE" | tee -a "tools/gridbarrier-bench/out/hw-$stamp.session.txt"
