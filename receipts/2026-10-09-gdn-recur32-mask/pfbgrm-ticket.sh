#!/bin/bash
# GPU ticket body (MAXMIN 10; M2: sleep 60 first, COOL=60). Padded batched prefill (the real mlx_lm.server prefill: four prompts of
# different lengths) on the merged fix (qgpr) against the masked-recur32 branch (qrm), mirrored qgpr qrm qrm qgpr, then pfbg2
# (which row differs, margin) on qrm. Rule: rm_accept.py (pre-registered before the wheel was built). Lengths PFL (default
# 454 417 363 304; jwm1: 120 110 90 70). Refuses under 6 GiB MemAvailable. Every step is a short submit: the slowest, the
# old padded route, is 10.7 s for the whole four-row prefill on jw16 (well under the 20 s per-submit rule).
set -u
B=~/.local/share/coreglass
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt 6291456 ]; then echo "@@pg SKIPPED MemAvailable ${avail_kb} kB < 6 GiB"; exit 0; fi
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json HF_HUB_OFFLINE=1
export PYTHONPATH=$B/mlx-lm-series-main/lib/python3.14/site-packages
for arm in qgpr qrm qrm qgpr; do
  sleep ${COOL:-3}
  raw=$B/build/pfbg-raw-$$-$arm.txt
  $B/venv-$arm/bin/python $B/prof/pfbg.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFL:-454 417 363 304} > "$raw" 2>&1
  echo "@@pg $arm rc=$?"
  grep -E '^PFBG|Error' "$raw" | sed "s/^/@@pg $arm /"
  grep -q '^PFBG' "$raw" || tail -6 "$raw" | cut -c1-200 | sed "s/^/@@pg $arm raw: /"
done
for run in 1 2; do
  sleep ${COOL:-3}
  $B/venv-qrm/bin/python $B/prof/pfbg2.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFL:-454 417 363 304} 2>&1 | grep -E '^PFBG2|Error' | sed "s/^/@@pg2 qrm run$run /"
done
echo "@@pg done $(date -u +%T)"
