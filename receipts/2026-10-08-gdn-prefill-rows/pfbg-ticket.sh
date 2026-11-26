#!/bin/bash
# GPU ticket body (MAXMIN 8; M2: sleep 60 first, COOL=60). Audit probe for the next batch-gated slow path: the REAL
# mlx_lm.server prefill (BatchGenerator, four prompts of different lengths = left padding + masks) against the same four
# prompts one at a time. Arms (mirrored): qbase (bug), qgpr (the fix), qgpr, qbase. Lengths: PFL (default the bench prompt
# lengths 454 417 363 304; jwm1: 120 110 90 70, because the full [4, T, 248320] logits exhaust its device memory).
# Rule (pre-registered): per arm ratio = batched_s / seq_s. After the fix the batched path should be near parity; if
# qgpr's ratio is > 1.3 the masked/padded batched prefill still falls off a route and the module profile of that case is
# the next audit step (pfmods on a padded batch). first_tokens_equal must be true or the arm is void.
set -u
B=~/.local/share/coreglass
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json HF_HUB_OFFLINE=1
export PYTHONPATH=$B/mlx-lm-series-main/lib/python3.14/site-packages
for arm in qbase qgpr qgpr qbase; do
  sleep ${COOL:-3}
  raw=$B/build/pfbg-raw-$$-$arm.txt
  $B/venv-$arm/bin/python $B/prof/pfbg.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFL:-454 417 363 304} > "$raw" 2>&1
  echo "@@pg $arm rc=$?"
  grep -E '^PFBG|Error' "$raw" | sed "s/^/@@pg $arm /"
  grep -q '^PFBG' "$raw" || tail -6 "$raw" | cut -c1-200 | sed "s/^/@@pg $arm raw: /"
done
echo "@@pg done $(date -u +%T)"
