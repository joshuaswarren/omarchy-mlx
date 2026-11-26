#!/usr/bin/env bash
# Window C: vec-exclusion probe + col16 span screen (arms are complete).
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
OUT=receipts-work/2026-09-11-bf16-decode-attribution
mkdir -p "$OUT/probe" "$OUT/screen"
CTRL=$PWD/.work/venv-bf16dec/bin/python
SCR=$PWD/.work/venv-screen/bin/python
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

VEC_PROBE_OUT="$OUT/probe/vec_exclusion.jsonl" \
  "$CTRL" /tmp/vec_exclusion_probe.py 2>&1 | tee "$OUT/probe/vec_exclusion.txt"

for pair in base:4 span16:16 span64:64 span512:512; do
  label="${pair%%:*}"; span="${pair##*:}"
  if [ "$label" = base ]; then
    "$SCR" /tmp/vec_screen.py --label base --rounds 24 \
      --out "$OUT/screen/base.ndjson" \
      --digest-out "$OUT/screen/digests.ndjson"
  else
    MLX_OMARCHY_VEC_WG_SPAN=$span \
    "$SCR" /tmp/vec_screen.py --label "$label" --rounds 24 \
      --out "$OUT/screen/$label.ndjson" \
      --digest-out "$OUT/screen/digests.ndjson"
  fi
done
echo WINDOW_C_DONE
