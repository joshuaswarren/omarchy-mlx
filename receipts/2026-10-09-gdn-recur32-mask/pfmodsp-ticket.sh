#!/bin/bash
# GPU ticket body (MAXMIN 8; M2: sleep 60 first, COOL=60). Module profile of a padded batched prefill (pfmodsp.py): which op
# type of the real mlx_lm.server prefill (BatchGenerator, left padding, mask) costs more batched than as four single runs.
# Arm qgpr (main now contains the fix), two runs. Lengths: PFL (default 454 417 363 304; jwm1: 120 110 90 70).
# Rule: the one in pfmodsp.py (largest excess_ms is the next fix; within 10% of the single sum is cleared).
# Refuses to start under 6 GiB MemAvailable.
set -u
B=~/.local/share/coreglass
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt 6291456 ]; then
  echo "@@pm SKIPPED MemAvailable ${avail_kb} kB < 6 GiB"
  exit 0
fi
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json HF_HUB_OFFLINE=1
export PYTHONPATH=$B/mlx-lm-series-main/lib/python3.14/site-packages
for run in 1 2; do
  sleep ${COOL:-3}
  raw=$B/build/pfmodsp-raw-$$-$run.txt
  $B/venv-qgpr/bin/python $B/prof/pfmodsp.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFL:-454 417 363 304} >> "$raw" 2>&1
  echo "@@pm run$run rc=$? memavail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  grep -E '^PFMODSP|Error' "$raw" | sed "s/^/@@pm run$run /"
  grep -q '^PFMODSP' "$raw" || tail -6 "$raw" | cut -c1-200 | sed "s/^/@@pm run$run raw: /"
done
echo "@@pm done $(date -u +%T)"
