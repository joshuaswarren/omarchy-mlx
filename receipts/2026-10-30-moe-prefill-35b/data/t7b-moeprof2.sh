#!/usr/bin/env bash
# 16-inch M1 Max: Qwen3.6-35B-A3B 4-bit prefill with the venv's own mlx_lm (U0 = pristine 0.31.3, no omarchy serve patches: the first profile ran on this)
# against the same mlx_lm with the current patch series applied (F = ~/scratch/mlxlm-series, scripts/apply-mlx-lm-patches.sh of 42b30a2d on pristine 0.31.3).
# Same diagnostics wheel (42b30a2d) in every run, one process per run, two interleaved rounds U0 F U0 F (unprofiled), then one profiled F run.
# Each run: 512-token warmup chunk, then 2048 tokens of real text in chunks of 512 (as t7b-moeprof.sh).
#   t7b-moeprof2.sh hostB
# GPU, HF offline, no download, cap 25 min (expect 12).
set -u
HOST=${1:?host}
[ "$HOST" = hostB ] || { echo "hostB only"; exit 64; }
PY="$HOME/t7/venv-diag-h/bin/python"
MODEL=$(find "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots -mindepth 1 -maxdepth 1 | head -1)
STAMP=$(date -u +%Y%m%dT%H%M%SZ); D="$HOME/t7/results/moeprof2-$STAMP"; mkdir -p "$D"; LOG="$D/log.txt"
echo "moeprof2 $HOST $STAMP load $(cut -d' ' -f1 /proc/loadavg) kernel $(uname -r) free $(df -BG --output=avail "$HOME" | tail -1)" | tee "$LOG"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/asahi_icd.json HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
PP_F="$HOME/scratch/mlxlm-series"
"$PY" -c 'import importlib.metadata as md; import mlx_lm; print("wheel", md.version("mlx_omarchy"), "U0 mlx_lm", mlx_lm.__file__)' | tee -a "$LOG"
PYTHONPATH="$PP_F" "$PY" -c 'import mlx_lm; print("F  mlx_lm", mlx_lm.__file__)' | tee -a "$LOG"
run() { # name pythonpath [env...]
  local name=$1 pp=$2; shift 2
  echo "-- $name $(date -u +%T)" | tee -a "$LOG"
  env PYTHONPATH="$pp" "$@" timeout 900 "$PY" "$HOME/t7/prefill_profile_driver.py" --model "$MODEL" --text "$HOME/t7/prefill-text.txt" \
    --tokens 2048 --step 512 --warmup-tokens 512 --out "$D/$name-windows.jsonl" 2>&1 | grep -E "^prefill|Error|Traceback|\[prof\]|text too short" | cut -c1-200 | tee -a "$LOG"
}
run U0-r1 "" MLX_OMARCHY_X=1
run F-r1 "$PP_F" MLX_OMARCHY_X=1
run U0-r2 "" MLX_OMARCHY_X=1
run F-r2 "$PP_F" MLX_OMARCHY_X=1
run FP "$PP_F" MLX_OMARCHY_GPU_PROFILE="$D/FP.ndjson" MLX_OMARCHY_GPU_PROFILE_LABEL=FP
python3 "$HOME/t7/b2prof_summary.py" one --profile "$D/FP.ndjson" --windows "$D/FP-windows.jsonl" --compute-h "$HOME/t7/compute.h" --out "$D/FP-sum.json" 2>&1 | head -16 | tee -a "$LOG"
rm -f "$D/FP.ndjson.keep"
for n in U0-r1 F-r1 U0-r2 F-r2; do python3 "$HOME/t7/b2prof_summary.py" wall --windows "$D/$n-windows.jsonl" --out "$D/$n-wall.json" 2>&1 | sed "s/^/$n /" | tee -a "$LOG"; done
echo "done $(date -u +%T)" | tee -a "$LOG"
