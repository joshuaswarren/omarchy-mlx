#!/usr/bin/env bash
# Window B: remaining attribution arms + vec-exclusion probe + col16 screen.
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
OUT=receipts-work/2026-09-11-bf16-decode-attribution
RUNPY=$PWD/.work/venv-run-bf16dec/bin/python
ABLPY=$PWD/.work/venv-ablate-bf16dec/bin/python
CTRL=$PWD/.work/venv-bf16dec/bin/python
SCR=$PWD/.work/venv-screen/bin/python
BASE=$HOME/src/mlx-bf16dec-base/dist/mlx_omarchy-0.32.2.dev202609111524+a5b8c4a-cp314-cp314-linux_aarch64.whl
ABLDIST=dist/mlx_omarchy-0.32.2.dev202609111520+d2ef0db-cp314-cp314-linux_aarch64.whl
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

# ---- 1. remaining arms ----
timeout 2700 python3 receipts-work/2026-09-11-bf16-decode-attribution/run_arms_bf16.py \
  --root "$PWD" \
  --python "$RUNPY" \
  --ablate-python "$ABLPY" \
  --wheel "$BASE" \
  --ablate-wheel "$ABLDIST" \
  --arms ewise copy rope rms swiglu sampler all \
  --out "$OUT/arms"

# ---- 2. vec-exclusion probe (in-model tile vs vec) ----
VEC_PROBE_OUT="$OUT/probe/vec_exclusion.jsonl" \
  $CTRL python3 /tmp/vec_exclusion_probe.py | tee "$OUT/probe/vec_exclusion.txt"

# ---- 3. col16 screen: base span=4 vs span=16/64/512, wall rounds ----
mkdir -p "$OUT/screen"
SCREEN_LABELS=("base:4" "span16:16" "span64:64" "span512:512")
for pair in "${SCREEN_LABELS[@]}"; do
  label="${pair%%:*}"; span="${pair##*:}"
  if [ "$label" = base ]; then
    $SCR python3 /tmp/vec_screen.py --label base --rounds 24 \
      --out "$OUT/screen/base.ndjson" --digest-out "$OUT/screen/digests.ndjson"
  else
    MLX_OMARCHY_VEC_WG_SPAN=$span \
    $SCR python3 /tmp/vec_screen.py --label "$label" --rounds 24 \
      --out "$OUT/screen/$label.ndjson" --digest-out "$OUT/screen/digests.ndjson"
  fi
done
echo WINDOW_B_DONE
