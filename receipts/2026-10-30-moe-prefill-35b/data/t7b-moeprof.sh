#!/usr/bin/env bash
# 16-inch M1 Max: where does Qwen3.6-35B-A3B 4-bit spend a prefill chunk? Diagnostics wheel of 42b30a2d (main after #303; the Python-only and docstring
# commits since then do not touch the backend). One process per run, model loaded from the local HF cache, mlx_lm 0.31.3 from the venv.
#   t7b-moeprof.sh hostB
# Runs (each: 512-token warmup chunk, then 2048 tokens of real text in chunks of 512; one bracketed window per chunk):
#   P  = MLX_OMARCHY_GPU_PROFILE on  (per-dispatch GPU timestamps: dispatch counts and GPU ms by kernel name; the profiler adds a barrier per dispatch)
#   U  = unprofiled                  (prefill tokens per second)
#   T0 = unprofiled with MLX_OMARCHY_GATHER_QMM_TILE=0 (is the sorted-expert tile route taken at all?)
# GPU, no download (HF offline), cap 25 min (expect 14).
set -u
HOST=${1:?host}
[ "$HOST" = hostB ] || { echo "hostB only"; exit 64; }
PY="$HOME/t7/venv-diag-h/bin/python"
MODEL=$(find "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots -mindepth 1 -maxdepth 1 | head -1)
STAMP=$(date -u +%Y%m%dT%H%M%SZ); D="$HOME/t7/results/moeprof-$STAMP"; mkdir -p "$D"; LOG="$D/log.txt"
echo "moeprof $HOST $STAMP load $(cut -d' ' -f1 /proc/loadavg) kernel $(uname -r) free $(df -BG --output=avail "$HOME" | tail -1) mem $(free -g | sed -n 2p | tr -s ' ')" | tee "$LOG"
echo "model $(basename "$MODEL")" | tee -a "$LOG"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/asahi_icd.json HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
"$PY" -c 'import importlib.metadata as md; print("wheel", md.version("mlx_omarchy"))' | tee -a "$LOG"
run() { # name env...
  local name=$1; shift
  echo "-- $name $(date -u +%T)" | tee -a "$LOG"
  env "$@" MLX_OMARCHY_GPU_PROFILE_LABEL="$name" timeout 900 "$PY" "$HOME/t7/prefill_profile_driver.py" --model "$MODEL" --text "$HOME/t7/prefill-text.txt" \
    --tokens 2048 --step 512 --warmup-tokens 512 --out "$D/$name-windows.jsonl" 2>&1 | grep -E "^prefill|Error|Traceback|\[prof\]|text too short" | cut -c1-200 | tee -a "$LOG"
}
run P MLX_OMARCHY_GPU_PROFILE="$D/P.ndjson"
python3 "$HOME/t7/b2prof_summary.py" one --profile "$D/P.ndjson" --windows "$D/P-windows.jsonl" --compute-h "$HOME/t7/compute.h" --out "$D/P-sum.json" 2>&1 | head -16 | tee -a "$LOG"
ls -la "$D/P.ndjson" | awk '{print "profile bytes", $5}' | tee -a "$LOG"
run U MLX_OMARCHY_NOPROFILE=1
run T0 MLX_OMARCHY_GATHER_QMM_TILE=0
for n in U T0; do python3 "$HOME/t7/b2prof_summary.py" wall --windows "$D/$n-windows.jsonl" --out "$D/$n-wall.json" 2>&1 | sed "s/^/$n /" | tee -a "$LOG"; done
echo "done $(date -u +%T)" | tee -a "$LOG"
