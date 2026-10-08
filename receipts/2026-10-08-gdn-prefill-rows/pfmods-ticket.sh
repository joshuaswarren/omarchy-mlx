#!/bin/bash
# GPU ticket body (MAXMIN 5). pfbatch found the batched [4, T] prefill 5.6-7.6x SLOWER than four sequential [1, T]
# passes on jw16 (G13C, 12.6 prerelease): T128 2.58 s vs 0.466, T454 9.64 vs 1.44, T1024 22.0 vs 2.91, rows identical,
# row 0 not bit-equal to the sequential pass. pfmods.py times every sublayer type at B=1 and B=4 (eval + sync around each)
# to name the module (or op) that falls off its fast route when the batch dimension is 4. T via PFT (default 454; jwm1,
# which fails with device-memory exhaustion on the [4, T, 248320] logits at T >= 443, runs PFT=128).
# Rule: the module type whose B=4 time exceeds 4x its B=1 time by the largest absolute amount is the fix target.
# Pre-registered; no decision beyond the ranking.
set -u
B=~/.local/share/coreglass
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json HF_HUB_OFFLINE=1
export PYTHONPATH=$B/mlx-lm-series-main/lib/python3.14/site-packages
PY=${PYV:-$B/venv-main/bin/python}
sleep ${COOL:-3}
$PY $B/prof/pfmods.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFT:-454} 2>&1 | grep -E '^PFMODS|Error' | sed 's/^/@@pm /'
echo "@@pm done $(date -u +%T)"
