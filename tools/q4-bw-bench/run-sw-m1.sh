#!/usr/bin/env bash
# SwiGLU dedicated-tile screen, M1 Max window (M1MD-2). Run under
#   flock -w 2400 /tmp/m1-gpu.lock timeout 900 bash tools/q4-bw-bench/run-sw-m1.sh
# from the checkout root. Produces NDJSON logs under tools/q4-bw-bench/out/.
# Arms: production r2s8w2 (base), r1s8w2, r2s4w2, r2s8w2h (scale-first
# hoist) on the 4B gate+up geometry (N=9728,K=2560) plus the 2B pair;
# bit-exactness compare of every side vs base.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
commit="$(git rev-parse --short=7 HEAD 2>/dev/null || echo not-git)"
host="$(hostname)"
echo "host=$host commit=$commit stamp=$stamp mode=swiglu" \
    | tee "tools/q4-bw-bench/out/session-sw-$stamp.txt"
sha256sum tools/q4-bw-bench/shaders/qmm_vec_swiglu.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
    | tee -a "tools/q4-bw-bench/out/session-sw-$stamp.txt"
g++ -std=c++17 -O2 -o /tmp/q4-bw-bench-sw tools/q4-bw-bench/bench.cpp
# Pin the internal Apple GPU: the bench picks the first enumerated device
# and jw16's B60 eGPU sorts first.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/asahi_icd.json
/tmp/q4-bw-bench-sw --swiglu 2>&1 | tee "tools/q4-bw-bench/out/sw-bench-a-$stamp.ndjson"
/tmp/q4-bw-bench-sw --swiglu 2>&1 | tee "tools/q4-bw-bench/out/sw-bench-b-$stamp.ndjson"
echo "RUN_SW_M1_DONE" | tee -a "tools/q4-bw-bench/out/session-sw-$stamp.txt"
