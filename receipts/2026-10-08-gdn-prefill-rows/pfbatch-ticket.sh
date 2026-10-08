#!/bin/bash
# GPU ticket body (MAXMIN 6; jw16 G13C first). Batched prefill, the c4 lever the oMLX profile points at: oMLX prefills
# concurrent requests one at a time (c4: four 454-token prefills = about half the request wall). With the QmmBatch fix the
# [B>1, T>1] prefill rows are exact, so a batched [4, T] pass is allowed. Does it beat 4 sequential [1, T] passes?
# pfbatch.py: T = 128, 454 (the bench prompt), 1024; two processes (mirrored) so a drift shows.
# Rule (pre-registered): per T, ratio = batched4_s / seq4_s. BATCHING WINS at T iff ratio < 0.95 in BOTH runs; LOSES iff
# ratio > 1.05 in both; else NEUTRAL. Rows must be exact (row_max_abs_diff <= 0.5 nats, the bf16 floor seen before) or
# VOID. Expected from the flat prefill tok/s across prompt lengths (467/459/440 on jwm1): NEUTRAL (EST), because the
# coopmat kernel already saturates at M = 512. If so, concurrent prefill is compute-bound and the c4 lever is prefill
# speed and interleaving for latency, not batching for throughput.
set -u
B=~/.local/share/coreglass
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json HF_HUB_OFFLINE=1
export PYTHONPATH=$B/mlx-lm-series-main/lib/python3.14/site-packages
PY=${PYV:-$B/venv-main/bin/python}
for run in 1 2; do
  sleep ${COOL:-3}
  $PY $B/prof/pfbatch.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFTS:-128 454 1024} 2>&1 \
    | grep -E '^PFBATCH|Error' | sed "s/^/@@pb run$run /"
done
echo "@@pb done $(date -u +%T)"
