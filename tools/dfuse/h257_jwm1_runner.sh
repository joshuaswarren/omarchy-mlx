#!/usr/bin/env bash
# H257 owner-bar bundle runner for jwm1 (G13G, real older Mesa).
# w71: set the env knobs, run under the lock. Produces the three owner-bar
# measurements for the 9B GDU raw route (fused perrow_pf vs composed).
#
# Knobs:
#   H257_PY       venv python with the v0.7.26+ wheel (route default ON)
#   H257_MODEL    9B model dir (Qwen3.5-9B MLX 4-bit)
#   H257_PROMPTS10 10-prompt jsonl ({"text": ...} per line)
#   H257_OUT      output dir
# Gates: no JW16_MAINTENANCE-style flag, uptime>=360, PSI cpu some avg10=0.
set -euo pipefail
PY=${H257_PY:?set H257_PY to the venv python}
MODEL=${H257_MODEL:?set H257_MODEL to the 9B model dir}
export H257_PROMPTS10=${H257_PROMPTS10:-${HOME}/bench-scripts/qwen38-2b-prompts-10.jsonl}
OUT=${H257_OUT:?set H257_OUT}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)

test ! -e /var/tmp/JW16_MAINTENANCE || true   # jwm1 flag name may differ
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "uptime gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }

echo "== (1) fp64 state/output error, both paths (captured operands)"
env -u MLX_OMARCHY_GDN_RAW_REPEAT MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF=0 \
  "$PY" "$HERE/gdu_fp64_probe.py" --capture "$OUT/gdu-operands.npz" > "$OUT/capture.log" 2>&1 || echo "capture FAILED"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 \
  "$PY" "$HERE/gdu_fp64_probe.py" --compare "$OUT/gdu-operands.npz" "$OUT/fp64.json" \
  > "$OUT/fp64.log" 2>&1 || echo "fp64 FAILED"
cat "$OUT/fp64.json" 2>/dev/null || tail -5 "$OUT/fp64.log"

echo "== (2) free-run 10x512, composed top-2 gap vs bf16 ULP at each first divergence"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 \
  "$PY" "$HERE/free_run_gaps.py" "$MODEL" "$OUT/free-run-gaps.json" 10 512 \
  > "$OUT/gaps.log" 2>&1 || echo "gaps FAILED"
python3 - <<'PYEOF'
import json
r = json.load(open(f"{__import__('os').environ['H257_OUT']}/free-run-gaps.json"))
print("mean_identity_pct:", r["mean_identity_pct"])
print("exact_match_pct_overall:", r["exact_match_pct_overall"])
print("all_divergences_within_one_bf16_ulp:", r["all_divergences_within_one_bf16_ulp"])
for row in r["rows"]:
    if row["first_divergence"] is not None:
        print(row)
PYEOF

echo "== (3) S=1 decode-route-sensitive PPL"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 \
  "$PY" "$HERE/ppl_s1.py" "$MODEL" "$OUT/ppl-s1.json" > "$OUT/ppl.log" 2>&1 || echo "ppl FAILED"
cat "$OUT/ppl-s1.json" 2>/dev/null || tail -5 "$OUT/ppl.log"

echo "== done =="
