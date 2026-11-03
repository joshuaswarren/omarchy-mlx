#!/bin/bash
# Receipt-grade decode A/B for the narrowed #316 (p1n f333327) against main a69d9e6: the
# receipt's protocol (decode_bench.py <repo> 64 4, two interleaved pairs per model, the idle
# wait: load below 0.5 and 20 s idle before every process). Resumable per model.
set -uo pipefail
U=$HOME/gpu/upr; O=$U/gate/out/speed316n; mkdir -p $O; cd $U/probes
for m in granite-4.0-1b-4bit Qwen2.5-1.5B-Instruct-4bit Llama-3.2-3B-Instruct-4bit Qwen2.5-0.5B-Instruct-4bit; do
  [ -s $O/done-$m ] && { echo "skip $m"; continue; }
  python3 ab.py $O/perf-p1n-$m.jsonl 2 main p1n -- decode_bench.py mlx-community/$m 64 4 || { echo "AB_FAILED $m"; exit 3; }
  date -u +%FT%TZ > $O/done-$m
done
echo "SPEED316N_DONE $(date -u +%FT%TZ)"
