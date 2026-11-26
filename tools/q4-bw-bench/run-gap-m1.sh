#!/usr/bin/env bash
# Decode in-model gap bench, M1 window. Run under ONE top-level flock:
#   flock -w 7200 /tmp/m1-gpu.lock timeout 2400 bash tools/q4-bw-bench/run-gap-m1.sh
# Timing-only arms; no driver swap; never merges.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || cat tools/q4-bw-bench/out/TREECOMMIT 2>/dev/null || echo not-git)"
host="$(hostname)"

echo "host=$host commit=$commit stamp=$stamp" | tee "tools/q4-bw-bench/out/gap-session-$stamp.txt"
echo "-- shader provenance --" | tee -a "tools/q4-bw-bench/out/gap-session-$stamp.txt"
sha256sum tools/q4-bw-bench/shaders/*.comp | tee -a "tools/q4-bw-bench/out/gap-session-$stamp.txt"
# The frozen base copy must equal the production shader for the measurement
# to be about the shipped kernel.
diff -q tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
    | tee -a "tools/q4-bw-bench/out/gap-session-$stamp.txt" || {
  echo "FATAL: qmm_vec_base.comp drifted from the production shader" \
      | tee -a "tools/q4-bw-bench/out/gap-session-$stamp.txt"
  exit 1
}

echo "quiet gate..." | tee -a "tools/q4-bw-bench/out/gap-session-$stamp.txt"
until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do sleep 5; done

g++ -std=c++17 -O2 -o /tmp/q4-bw-bench tools/q4-bw-bench/bench.cpp
/tmp/q4-bw-bench --gap 2>&1 | tee "tools/q4-bw-bench/out/gap-a-$stamp.ndjson"
sleep 10
/tmp/q4-bw-bench --gap 2>&1 | tee "tools/q4-bw-bench/out/gap-b-$stamp.ndjson"

echo "RUN_GAP_M1_DONE" | tee -a "tools/q4-bw-bench/out/gap-session-$stamp.txt"
