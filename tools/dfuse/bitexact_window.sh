#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: bit-exact state-injection bisect (tiled, v0.7.26 venv).
set -euo pipefail
C=/var/tmp/gdubar-v0726-venv/bin/python
M9B=$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
O=/var/tmp/dfuse/bitexact
mkdir -p "$O"
env -u MLX_OMARCHY_GDN_RAW_REPEAT -u MLX_OMARCHY_GDN_DECODE_TILE \
  "$C" "${BISECT_PY:-/var/tmp/dfuse/bitexact_bisect.py}" "$M9B" "$O" 12 1 > "$O/bisect.log" 2>&1 || { echo BISECT-FAILED; tail -8 "$O/bisect.log"; exit 1; }
grep -c DIFF "$O/bisect.log" || true
ls -la "$O"/failing-operands.npz 2>/dev/null || true
