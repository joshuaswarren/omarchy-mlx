#!/usr/bin/env bash
# Window A part 2: arms (baseline gemv attn cast bf16fast).
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
OUT=receipts-work/2026-09-11-bf16-decode-attribution
RUNPY=$PWD/.work/venv-run-bf16dec/bin/python
ABLPY=$PWD/.work/venv-ablate-bf16dec/bin/python
BASE=$HOME/src/mlx-bf16dec-base/dist/mlx_omarchy-0.32.2.dev202609111524+a5b8c4a-cp314-cp314-linux_aarch64.whl
ABLDIST=dist/mlx_omarchy-0.32.2.dev202609111520+d2ef0db-cp314-cp314-linux_aarch64.whl
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

timeout 3300 python3 /tmp/run_arms_bf16.py \
  --root "$PWD" \
  --python "$RUNPY" \
  --ablate-python "$ABLPY" \
  --wheel "$BASE" \
  --ablate-wheel "$ABLDIST" \
  --arms baseline gemv attn cast bf16fast \
  --out "$OUT/arms"
echo WINDOW_A_DONE
