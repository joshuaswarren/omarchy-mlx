#!/usr/bin/env bash
# Per-dispatch floor microbench, M1 window. Run under
#   flock -w 2400 /tmp/m1-gpu.lock timeout 900 bash tools/dispatch-floor-bench/run-m1.sh
# from the checkout root. Produces NDJSON logs under tools/dispatch-floor-bench/out/.
# Driver selection is inherited from the environment: default = system fork,
# VK_DRIVER_FILES=<icd.json> selects stock or worktree-built drivers,
# HK_PERFTEST=... selects driver perftest modes. The session log records both.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/dispatch-floor-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
host="$(hostname)"

echo "host=$host commit=$commit stamp=$stamp" \
  | tee "tools/dispatch-floor-bench/out/session-$stamp.txt"
echo "VK_DRIVER_FILES=${VK_DRIVER_FILES:-<default>} HK_PERFTEST=${HK_PERFTEST:-<unset>}" \
  | tee -a "tools/dispatch-floor-bench/out/session-$stamp.txt"
vulkaninfo 2>/dev/null | grep -m1 deviceName \
  | tee -a "tools/dispatch-floor-bench/out/session-$stamp.txt" || true

g++ -std=c++17 -O2 -o /tmp/dispatch-floor-bench tools/dispatch-floor-bench/bench.cpp

# Two independent legs for repeatability.
/tmp/dispatch-floor-bench 2>&1 \
  | tee "tools/dispatch-floor-bench/out/bench-a-$stamp.ndjson"
/tmp/dispatch-floor-bench 2>&1 \
  | tee "tools/dispatch-floor-bench/out/bench-b-$stamp.ndjson"

echo "RUN_M1_DONE" | tee -a "tools/dispatch-floor-bench/out/session-$stamp.txt"
