#!/usr/bin/env bash
# DrainFix A/B inner: decode cells for base (ctl) vs lookahead (on/stack/neutral) arms.
# Runs INSIDE a gpuwin window. Controlled by env:
#   OUT, DF_PLAN - semicolon-separated specs "label:depth:arm[:passes]" (arm=ctl|on|stack|neutral)
# Gates: uptime>=360s, load1<0.5, PSI cpu some avg10=0.00.
set -euo pipefail
OUT=${OUT:?missing OUT}
DF_PLAN=${DF_PLAN:?missing DF_PLAN}
PY=${DF_PY:-/var/tmp/v072-venv-fused/bin/python}
CANDPY=${DF_CAND_PY:-}
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
MODEL=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d' ' -f1 /proc/loadavg)
load_int=${load1%%.*}; load_frac=${load1#*.}00
(( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf 'measurement_host=%s uptime_s=%s load1=%s psi=%s\n' "$(hostname)" "$uptime_s" "$load1" "$psi" | tee "$OUT/gates.txt"
printf 'DF_PLAN=%s\n' "$DF_PLAN" >> "$OUT/gates.txt"
"$PY" -m pip show mlx-omarchy | sed -n '1,2p' > "$OUT/package.txt" 2>/dev/null || true
if [ -n "$CANDPY" ]; then
  "$CANDPY" -m pip show mlx-omarchy | sed -n '1,2p' > "$OUT/candidate-package.txt" 2>/dev/null || true
fi

run_one() { # label depth arm passes
  local label=$1 depth=$2 arm=$3 passes=${4:-1} py extra=()
  case "$arm" in
    # ctl = serving venv, no env. All other arms run the candidate venv (patched
    # generate.py); "neutral" keeps the gate OFF to prove the patch machinery is inert.
    ctl)     py="$PY" ;;
    on)      py="${CANDPY:?on arm needs DF_CAND_PY}"; extra=(MLX_OMARCHY_DECODE_LOOKAHEAD=2) ;;
    stack)   py="${CANDPY:?stack arm needs DF_CAND_PY}"; extra=(MLX_OMARCHY_DECODE_LOOKAHEAD=2 HK_SUBMIT_POLL_US=10000) ;;
    stack20) py="${CANDPY:?stack arm needs DF_CAND_PY}"; extra=(MLX_OMARCHY_DECODE_LOOKAHEAD=2 HK_SUBMIT_POLL_US=20000) ;;
    neutral) py="${CANDPY:?neutral arm needs DF_CAND_PY}"; extra=(MLX_OMARCHY_DECODE_LOOKAHEAD=0) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  if [ "$py" = "$CANDPY" ]; then
    # prove the process really loads the patched module with the intended gate
    env "${extra[@]}" "$py" - <<'PYEOF'
import importlib, inspect, os
g = importlib.import_module("mlx_lm.generate")
src = inspect.getsource(g.generate_step)
assert "MLX_OMARCHY_DECODE_LOOKAHEAD" in src, "patch not loaded"
print("patch-active gate =", os.environ.get("MLX_OMARCHY_DECODE_LOOKAHEAD", "<unset>"))
PYEOF
  fi
  env "${extra[@]}" MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 MLX_OMARCHY_GDN_F16_STATE=0 \
    "$py" "$BENCH" --model "$MODEL" --prompts "$PROMPTS" --limit 1 --new-tokens "$depth" \
    --prefill-tokens 0 --warmup 1 --passes "$passes" --label "$label" --out "$OUT/$label.json"
  "$PY" - "$OUT/$label.json" <<'PYEOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("df_pins", "/var/tmp/drainfix/pins.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
d = json.load(open(sys.argv[1]))
proto = d.get("protocol") or {}
key = (proto.get("new_tokens"), proto.get("passes", 1))
dig = d.get("ordered_records_sha256")
pin = mod.PINS.get(key)
if pin is None:
    print(f"digest UNPINNED depth={key} value={dig}")
elif pin != dig:
    raise SystemExit(f"DIGEST-MISMATCH depth={key} got={dig}")
else:
    print(f"digest OK depth={key}")
PYEOF
  echo "ran $label depth=$depth arm=$arm passes=$passes"
}

IFS=';' read -ra SPECS <<< "$DF_PLAN"
for spec in "${SPECS[@]}"; do
  IFS=':' read -r label depth arm passes <<< "$spec"
  run_one "$label" "$depth" "$arm" "${passes:-1}"
done
echo "== plan done =="
