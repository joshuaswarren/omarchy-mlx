#!/usr/bin/env bash
# docs/numerics-gate.md for MLX_OMARCHY_QMV_SMALL_T on the parity-contract 2B model
# (SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b, docs/parity-contract.md), ten contract prompts
# (token ids from receipts-work/vocab-prune/out/contract-ctl-r1.json), 512 teacher tokens each.
#
#   GPU arms: the candidate wheel in ~/.cache/omarchy-mlx-fix/venv; route off (deployed) and
#             MLX_OMARCHY_QMV_SMALL_T=6 (candidate); text read in 4-token chunks (5 runs each)
#             and 6-token chunks (1 run each), so every pass is a short pass
#   reference: stock upstream MLX 0.32.3 CPU + mlx-lm 0.31.3, fp32, in ~/.cache/hk/ref-venv
#   check 1:  perop_error.py (fp64 numpy) on the candidate wheel
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$HOME/Projects/omarchy-mlx-fix
model=$HOME/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
record=$repo/receipts-work/vocab-prune/out/contract-ctl-r1.json
gpu_py=$HOME/.cache/omarchy-mlx-fix/venv/bin/python
ref_py=$HOME/.cache/hk/ref-venv/bin/python
out=$HOME/.cache/omarchy-mlx-fix/gate
lock=$HOME/.local/state/mseries-ai/worker.lock
export VK_DRIVER_FILES=$HOME/.local/share/mlx-omarchy/vulkan/honeykrisp_icd.aarch64.json
mkdir -p "$out"
gate() { "$1" "$here/qwen38_chunked_gate.py" "${@:2}"; }

echo "== model files"
( cd "$model" && sha256sum model.safetensors tokenizer.json config.json ) | tee "$out/model-sha256.txt"
grep -q '^b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863 ' "$out/model-sha256.txt"
"$gpu_py" -c 'import mlx.core as mx; print("candidate wheel", mx.__version__)' 2>/dev/null | tee "$out/wheel.txt"

echo "== teacher (route off, free-running greedy, 10 x 512)"
[[ -s $out/teacher.json ]] || flock "$lock" env -u MLX_OMARCHY_QMV_SMALL_T \
  "$gpu_py" "$here/qwen38_chunked_gate.py" teacher --model "$model" --record "$record" --out "$out/teacher.json"

echo "== reference: stock MLX CPU fp32, 5 parallel parts (about 35 minutes)"
if [[ ! -s $out/ref.json ]]; then
  pids=()
  for part in 0,1 2,3 4,5 6,7 8,9; do
    "$ref_py" "$here/qwen38_chunked_gate.py" measure --model "$model" --teacher "$out/teacher.json" \
      --chunk 64 --cpu --only "$part" --out "$out/ref-part-${part/,/-}.json" > "$out/ref-part-${part/,/-}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  gate "$ref_py" merge --model "$model" --parts "$out"/ref-part-*.json --out "$out/ref.json"
fi

echo "== GPU arms"
for r in 1 2 3 4 5; do
  for arm in base cand; do
    f=$out/$arm-c4-r$r.json; [[ -s $f ]] && continue
    t=""; [[ $arm == cand ]] && t=6
    flock "$lock" env MLX_OMARCHY_QMV_SMALL_T="$t" "$gpu_py" "$here/qwen38_chunked_gate.py" measure \
      --model "$model" --teacher "$out/teacher.json" --chunk 4 --out "$f"
  done
done
for arm in base cand; do
  f=$out/$arm-c6-r1.json; [[ -s $f ]] && continue
  t=""; [[ $arm == cand ]] && t=6
  flock "$lock" env MLX_OMARCHY_QMV_SMALL_T="$t" "$gpu_py" "$here/qwen38_chunked_gate.py" measure \
    --model "$model" --teacher "$out/teacher.json" --chunk 6 --out "$f"
done

echo "== harness check: S=1 PPL vs one-pass prefill PPL (route off)"
flock "$lock" env -u MLX_OMARCHY_QMV_SMALL_T "$gpu_py" "$here/qwen38_chunked_gate.py" validate \
  --model "$model" --teacher "$out/teacher.json" --out "$out/validate.json"

echo "== check 1: per-op error against fp64"
flock "$lock" env VK_DRIVER_FILES="$VK_DRIVER_FILES" "$gpu_py" "$here/perop_error.py" "$out/perop-error.json" | tail -3

echo "== checks 2 and 3"
python3 "$here/analyze_gate.py" "$out" "$repo" | tee "$out/gate.txt"
echo "== old vs new (compare_tf9.py, information only)"
PYTHONPATH=$repo/tools/dfuse python3 "$repo/tools/dfuse/compare_tf9.py" "$out/base-c4-r1.json" "$out/cand-c4-r1.json" \
  | tee "$out/compare_tf9.txt" | head -20
echo "Done: $out"
