#!/usr/bin/env bash
# Q4 decode GEMV bandwidth bench, 2B decode shapes (Qwen3.8-2B), bf16
# x-load mix. Run under ONE top-level flock from any directory containing
# tools/q4-bw-bench:
#   flock -w 7200 /tmp/m1-gpu.lock timeout 2400 bash tools/q4-bw-bench/run-gap-2b.sh
# Timing-only arms; no driver swap; never merges. Unlike run-gap-m1.sh
# this does NOT diff the base copy against the checkout's production
# shader (hosts may hold different branch states); shader provenance is
# the sha256 line instead.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tools/q4-bw-bench/out
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
host="$(hostname)"

echo "host=$host stamp=$stamp" | tee "tools/q4-bw-bench/out/gap2b-session-$stamp.txt"
echo "-- shader provenance --" | tee -a "tools/q4-bw-bench/out/gap2b-session-$stamp.txt"
sha256sum tools/q4-bw-bench/shaders/*.comp | tee -a "tools/q4-bw-bench/out/gap2b-session-$stamp.txt"

echo "quiet gate..." | tee -a "tools/q4-bw-bench/out/gap2b-session-$stamp.txt"
until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do sleep 5; done

g++ -std=c++17 -O2 -o /tmp/q4-bw-bench2b tools/q4-bw-bench/bench.cpp
/tmp/q4-bw-bench2b --gap --2b 2>&1 | tee "tools/q4-bw-bench/out/gap2b-a-$stamp.ndjson"
sleep 10
/tmp/q4-bw-bench2b --gap --2b 2>&1 | tee "tools/q4-bw-bench/out/gap2b-b-$stamp.ndjson"

echo "RUN_GAP_2B_DONE" | tee -a "tools/q4-bw-bench/out/gap2b-session-$stamp.txt"
