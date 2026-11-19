#!/usr/bin/env bash
# Q4 decode GEMV bandwidth bench, G17C (M5 Max) window: the ceiling-table
# shape (n=320, k=2560, affine group 32) with the production general
# affine kernel as base against the 40 WG x 320-thread ceiling mapping in
# shaders/qmm_vec_cand_m5.comp. Run under
#   flock -w 2400 /tmp/m1-gpu.lock timeout 900 bash tools/q4-bw-bench/run-m5.sh
# from the checkout root (an independent /tmp/m1-gpu.lock per physical
# host). Produces NDJSON logs under tools/q4-bw-bench/out/.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
host="$(hostname)"

echo "host=$host commit=$commit stamp=$stamp" | tee "tools/q4-bw-bench/out/session-m5-$stamp.txt"
echo "-- shader provenance --" | tee -a "tools/q4-bw-bench/out/session-m5-$stamp.txt"
sha256sum tools/q4-bw-bench/shaders/*.comp | tee -a "tools/q4-bw-bench/out/session-m5-$stamp.txt"
# The frozen base copy must equal the production shader for the base
# side to be about the shipped kernel.
diff -q tools/q4-bw-bench/shaders/qmm_vec_base.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
    | tee -a "tools/q4-bw-bench/out/session-m5-$stamp.txt" || {
  echo "FATAL: qmm_vec_base.comp drifted from the production shader" \
      | tee -a "tools/q4-bw-bench/out/session-m5-$stamp.txt"
  exit 1
}

g++ -std=c++17 -O2 -o /tmp/q4-bw-bench-m5 tools/q4-bw-bench/bench.cpp

# Two independent legs for repeatability; each leg times base and the m5
# ceiling mapping on identical input buffers with bit-exactness compares
# against the base. The subgroup base requires subgroupSize == 32 (the
# G17C); on a screening box run --m5 --tree directly instead.
/tmp/q4-bw-bench-m5 --m5 2>&1 | tee "tools/q4-bw-bench/out/bench-m5-a-$stamp.ndjson"
/tmp/q4-bw-bench-m5 --m5 2>&1 | tee "tools/q4-bw-bench/out/bench-m5-b-$stamp.ndjson"

echo "RUN_M5_DONE" | tee -a "tools/q4-bw-bench/out/session-m5-$stamp.txt"
