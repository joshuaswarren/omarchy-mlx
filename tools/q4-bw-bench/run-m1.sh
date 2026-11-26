#!/usr/bin/env bash
# Q4 decode GEMV bandwidth bench, M1 window. Run under
#   flock -w 2400 /tmp/m1-gpu.lock timeout 900 bash tools/q4-bw-bench/run-m1.sh
# from the checkout root. Produces NDJSON logs under tools/q4-bw-bench/out/.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
host="$(hostname)"

echo "host=$host commit=$commit stamp=$stamp" | tee "tools/q4-bw-bench/out/session-$stamp.txt"
echo "-- shader provenance --" | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
sha256sum tools/q4-bw-bench/shaders/*.comp | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
# The base copy must equal the production shader for the measurement to
# be about the shipped kernel.
diff -q tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
    | tee -a "tools/q4-bw-bench/out/session-$stamp.txt" || {
  echo "FATAL: qmm_vec_base.comp drifted from the production shader" \
      | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
  exit 1
}

g++ -std=c++17 -O2 -o /tmp/q4-bw-bench tools/q4-bw-bench/bench.cpp

# Two independent legs for repeatability; each leg times base, unroll,
# loadfirst, and wg128 on identical input buffers with bit-exactness
# compares against the base.
/tmp/q4-bw-bench 2>&1 | tee "tools/q4-bw-bench/out/bench-a-$stamp.ndjson"
/tmp/q4-bw-bench 2>&1 | tee "tools/q4-bw-bench/out/bench-b-$stamp.ndjson"

echo "RUN_M1_DONE" | tee -a "tools/q4-bw-bench/out/session-$stamp.txt"
