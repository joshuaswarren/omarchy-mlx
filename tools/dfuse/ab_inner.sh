#!/usr/bin/env bash
# DispatchFuse A/B inner: decode cells for ctl (serving) vs cand (patched venv)
# vs neu (patched venv, gates off). Runs INSIDE a gpuwin window.
# Env: DF_OUT (results dir), DF_PLAN "label:depth:arm[:passes]" ;-separated.
# Arms: ctl4b/on4b/neu4b (rope-norm fold), ctl9b/on9b/neu9b (GDN raw repeat),
# ctl2b/on2b/neu2b (rope-norm gate flip on the already-patched qwen3_next).
set -euo pipefail
DF_OUT=${DF_OUT:?missing DF_OUT}
DF_PLAN=${DF_PLAN:?missing DF_PLAN}
PY=${DF_CTL_PY:-/var/tmp/v072-venv-fused/bin/python}
CANDPY=${DF_CAND_PY:-/var/tmp/dfuse-cand/bin/python}
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
printf "host=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" "$uptime_s" "$load1" "$psi" | tee "$DF_OUT/gates.txt"
printf "DF_PLAN=%s\n" "$DF_PLAN" >> "$DF_OUT/gates.txt"
"$PY" -m pip show mlx-omarchy | sed -n 2p > "$DF_OUT/ctl-stamp.txt"
"$CANDPY" -m pip show mlx-omarchy | sed -n 2p > "$DF_OUT/cand-stamp.txt"
cmp -s "$DF_OUT/ctl-stamp.txt" "$DF_OUT/cand-stamp.txt" && { echo "STAMP COLLISION"; exit 1; }
PROV=${DF_PROV:-/var/tmp/dfuse-build/scripts/mlx_provenance.py}
"$PY" "$PROV" > "$DF_OUT/ctl-provenance.txt" 2>&1 || { echo "ctl provenance refused"; exit 1; }
"$CANDPY" "$PROV" > "$DF_OUT/cand-provenance.txt" 2>&1 || { echo "cand provenance refused"; exit 1; }

M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
M4B=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)
M9B=$(ls -d $HF/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)

run_one() { # label depth arm passes model gate...
  local label=$1 depth=$2 arm=$3 passes=${4:-1} model=$5; shift 5
  local py extra=()
  case "$arm" in
    ctl*) py="$PY" ;;
    on*)  py="$CANDPY"; extra=("$@") ;;
    neu*) py="$CANDPY" ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  if [ "$py" = "$CANDPY" ] && [ "${#extra[@]}" -gt 0 ]; then
    env "${extra[@]}" "$py" - <<'PYEOF'
import importlib, inspect, os
g = importlib.import_module("mlx_lm.models.gated_delta")
q3 = importlib.import_module("mlx_lm.models.qwen3")
qn = importlib.import_module("mlx_lm.models.qwen3_next")
for name, mod in (("gated_delta", g), ("qwen3", q3), ("qwen3_next", qn)):
    src = inspect.getsource(mod)
    if os.environ.get("MLX_OMARCHY_ROPE_NORM_FUSE") == "1" and name in ("qwen3", "qwen3_next"):
        assert "MLX_OMARCHY_ROPE_NORM_FUSE" in src, f"{name} rope-norm patch not loaded"
    if os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT") == "1" and name == "gated_delta":
        assert "MLX_OMARCHY_GDN_RAW_REPEAT" in src, "gdn raw-repeat patch not loaded"
print("patch-active gates ok:",
      os.environ.get("MLX_OMARCHY_ROPE_NORM_FUSE", "-"),
      os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "-"))
PYEOF
  fi
  local out="$DF_OUT/$label"
  timeout 600 env "${extra[@]}" \
    "$py" "$BENCH" --model "$model" --prompts "$PROMPTS" --limit 1 --warmup 1 \
      --passes "$passes" --new-tokens "$depth" --prefill-tokens 512 \
      --label "$label" --out "$out.json" > "$out.log" 2> "$out.err" \
    || { echo "RUN FAILED $label rc=$?"; return 1; }
  echo "done $label $(grep -o '"ordered_records_sha256[^,}]*' "$out.json" | head -1)"
}

IFS=';' read -ra SPECS <<< "$DF_PLAN"
for spec in "${SPECS[@]}"; do
  IFS=':' read -ra f <<< "$spec"
  case "${f[2]}" in
    ctl4b) run_one "${f[0]}" "${f[1]}" ctl "${f[3]:-1}" "$M4B" ;;
    on4b)  run_one "${f[0]}" "${f[1]}" on  "${f[3]:-1}" "$M4B" MLX_OMARCHY_ROPE_NORM_FUSE=1 ;;
    neu4b) run_one "${f[0]}" "${f[1]}" neu "${f[3]:-1}" "$M4B" ;;
    ctl9b) run_one "${f[0]}" "${f[1]}" ctl "${f[3]:-1}" "$M9B" ;;
    on9b)  run_one "${f[0]}" "${f[1]}" on  "${f[3]:-1}" "$M9B" MLX_OMARCHY_GDN_RAW_REPEAT=1 ;;
    neu9b) run_one "${f[0]}" "${f[1]}" neu "${f[3]:-1}" "$M9B" ;;
    off9b) run_one "${f[0]}" "${f[1]}" on  "${f[3]:-1}" "$M9B" MLX_OMARCHY_GDN_RAW_REPEAT=0 ;;
    ctl2b) run_one "${f[0]}" "${f[1]}" ctl "${f[3]:-1}" "$M2B" ;;
    on2b)  run_one "${f[0]}" "${f[1]}" on  "${f[3]:-1}" "$M2B" MLX_OMARCHY_ROPE_NORM_FUSE=1 ;;
    neu2b) run_one "${f[0]}" "${f[1]}" neu "${f[3]:-1}" "$M2B" ;;
    *) echo "bad spec $spec"; exit 2 ;;
  esac
done

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc" | tee -a "$DF_OUT/gates.txt"
