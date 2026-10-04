#!/usr/bin/env bash
# Fuse2B A/B inner: decode cells for ctl (serving) vs cand (fused wheel,
# MLX_OMARCHY_GDN_CONV_DELTA=1) vs neu (cand wheel, gate off = bit-identical
# dispatch behavior to serving modulo the wheel). Runs INSIDE a gpuwin window.
# Env: DF_OUT (results dir), DF_PLAN "label:depth:arm" ;-separated.
set -euo pipefail
DF_OUT=${DF_OUT:?missing DF_OUT}
DF_PLAN=${DF_PLAN:?missing DF_PLAN}
PY=${DF_CTL_PY:-/var/tmp/v072-venv-fused/bin/python}
CANDPY=/var/tmp/fuse2b-cand/bin/python
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
HF=${HOME}/.cache/huggingface/hub

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf "host=%s boot=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" | tee "$DF_OUT/gates.txt"
printf "DF_PLAN=%s\n" "$DF_PLAN" >> "$DF_OUT/gates.txt"
"$PY" -m pip show mlx-omarchy | sed -n 2p > "$DF_OUT/ctl-stamp.txt"
"$CANDPY" -m pip show mlx-omarchy | sed -n 2p > "$DF_OUT/cand-stamp.txt"
cat "$DF_OUT/ctl-stamp.txt" "$DF_OUT/cand-stamp.txt"
cmp -s "$DF_OUT/ctl-stamp.txt" "$DF_OUT/cand-stamp.txt" && { echo "STAMP COLLISION"; exit 1; }

M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
test -d "$M2B" || { echo "missing 2B snapshot"; exit 1; }

run_one() { # label depth arm passes
  local label=$1 depth=$2 arm=$3 passes=${4:-1}
  local py extra=()
  case "$arm" in
    ctl*) py="$PY" ;;
    on*)  py="$CANDPY"; extra=(MLX_OMARCHY_GDN_CONV_DELTA=1) ;;
    neu*) py="$CANDPY" ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  if [ "$py" = "$CANDPY" ]; then
    env -u MLX_OMARCHY_GDN_CONV_DELTA "${extra[@]}" "$py" - <<'PYEOF'
import importlib, inspect, os
q5 = importlib.import_module("mlx_lm.models.qwen3_5")
src = inspect.getsource(q5)
assert "MLX_OMARCHY_GDN_CONV_DELTA" in src, "conv-delta patch not loaded"
if os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA") == "1":
    import mlx.core as mx
    assert hasattr(mx.fast, "gdn_conv_delta_update"), "fast primitive missing"
print("patch-active ok: GDN_CONV_DELTA=", os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA", "-"))
PYEOF
  fi
  local out="$DF_OUT/$label"
  timeout 600 env -u MLX_OMARCHY_GDN_CONV_DELTA "${extra[@]}" \
    "$py" "$BENCH" --model "$M2B" --prompts "$PROMPTS" --limit 1 --warmup 1 \
      --passes "$passes" --new-tokens "$depth" --prefill-tokens 512 \
      --label "$label" --out "$out.json" > "$out.log" 2> "$out.err" \
    || { echo "RUN FAILED $label rc=$?"; return 1; }
  echo "done $label $(grep -o '"ordered_records_sha256[^,}]*' "$out.json" | head -1)"
}

IFS=';' read -ra SPECS <<< "$DF_PLAN"
for spec in "${SPECS[@]}"; do
  IFS=':' read -ra f <<< "$spec"
  run_one "${f[0]}" "${f[1]}" "${f[2]}" "${f[3]:-1}"
done

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc" | tee -a "$DF_OUT/gates.txt"
