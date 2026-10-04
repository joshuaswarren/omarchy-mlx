#!/usr/bin/env bash
# DispatchFuse A/B inner: decode cells for ctl (serving) vs cand (patched venv,
# rope-norm gate on) vs neu (patched venv, gate off). Runs INSIDE a gpuwin window.
# Env: DF_OUT (results dir), DF_PLAN "label:depth:arm[:passes]" ;-separated.
set -euo pipefail
DF_OUT=${DF_OUT:?missing DF_OUT}
DF_PLAN=${DF_PLAN:?missing DF_PLAN}
PY=/var/tmp/v072-venv-fused/bin/python
CANDPY=/var/tmp/dfuse-cand/bin/python
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf "host=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" "$uptime_s" "$load1" "$psi" | tee "$DF_OUT/gates.txt"
printf "DF_PLAN=%s\n" "$DF_PLAN" >> "$DF_OUT/gates.txt"
"$PY" -m pip show mlx-omarchy | sed -n 2p > "$DF_OUT/ctl-stamp.txt"
"$CANDPY" -m pip show mlx-omarchy | sed -n 2p > "$DF_OUT/cand-stamp.txt"
cmp -s "$DF_OUT/ctl-stamp.txt" "$DF_OUT/cand-stamp.txt" && { echo "STAMP COLLISION"; exit 1; }

run_one() { # label depth arm passes model
  local label=$1 depth=$2 arm=$3 passes=${4:-1} model=$5 py extra=()
  case "$arm" in
    ctl) py="$PY" ;;
    on)  py="$CANDPY"; extra=(MLX_OMARCHY_ROPE_NORM_FUSE=1) ;;
    neu) py="$CANDPY" ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  if [ "$py" = "$CANDPY" ]; then
    env "${extra[@]}" "$py" - <<'PYEOF'
import importlib, inspect, os
m = importlib.import_module("mlx_lm.models.qwen3")
src = inspect.getsource(m)
assert "MLX_OMARCHY_ROPE_NORM_FUSE" in src, "qwen3 patch not loaded"
print("patch-active gate =", os.environ.get("MLX_OMARCHY_ROPE_NORM_FUSE", "<unset>"))
PYEOF
  fi
  if [ "$py" = "$CANDPY" ]; then
    env "${extra[@]}" "$py" - <<'PYEOF'
import mlx.core as mx
print("cand-wheel", mx.__version__)
PYEOF
  fi
  local out="$DF_OUT/$label"
  timeout 300 env "${extra[@]}" \
    "$py" "$BENCH" --model "$model" --prompts "$PROMPTS" --limit 1 --warmup 1 \
      --passes "$passes" --new-tokens "$depth" --prefill-tokens 512 \
      --label "$label" --out "$out.json" > "$out.log" 2> "$out.err" \
    || { echo "RUN FAILED $label rc=$?"; return 1; }
  echo "done $label $(grep -o '"ordered_records_sha256[^,}]*' "$out.json" | head -1)"
}

M4B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)

IFS=';' read -ra SPECS <<< "$DF_PLAN"
for spec in "${SPECS[@]}"; do
  IFS=':' read -ra f <<< "$spec"
  case "${f[2]}" in
    ctl4b) run_one "${f[0]}" "${f[1]}" ctl "${f[3]:-1}" "$M4B" ;;
    on4b)  run_one "${f[0]}" "${f[1]}" on  "${f[3]:-1}" "$M4B" ;;
    neu4b) run_one "${f[0]}" "${f[1]}" neu "${f[3]:-1}" "$M4B" ;;
    *) echo "bad spec $spec"; exit 2 ;;
  esac
done

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc" | tee -a "$DF_OUT/gates.txt"
