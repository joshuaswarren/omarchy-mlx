#!/usr/bin/env bash
# H137b pair A/B on a second chip (M2/G14C primary): does the larger core
# count give the REAL production GEMV kernel enough co-residency (>= the 256
# row-tile workgroups of a 2048x2048 Q4 GEMV at 128 threads) that the
# persistent form stops losing memory-level parallelism? H137b on G13G:
# bit-exact everywhere but +25 us/pair at the <=80-WG residency ceiling.
# Safety: escalating G, one short submission per set, stop at first timeout
# flag (2^17-poll bounded spin), never crosses the driver watchdog.
# Run under the GPU lock / a gpu-turn ticket, from the checkout root:
#   bash tools/gridbarrier-bench/pair/run-pair.sh
set -euo pipefail
cd "$(dirname "$0")"
dir="$PWD"
bash pair-gen.sh /tmp/pair_fused.comp
g++ -std=c++17 -O2 -o /tmp/pair-bench pair-bench.cpp
PAIR_GS="${PAIR_GS:-64,128,192,256,320,384}" \
  /tmp/pair-bench /tmp/pair_fused.comp "$dir/shaders/qmm_vec_base.comp"
