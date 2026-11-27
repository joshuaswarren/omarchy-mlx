#!/usr/bin/env bash
# R2 A/B: 5 pairs, A=legacy env (MLX_DISABLE_COMPILE=1) vs B=compile tape on.
# ABBA pattern decorrelates drift. One info arm: B + batch_work 40000.
set -u
MLXMODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
PY=/usr/lib/omarchy-mlx/venv/bin/python
DRV=/home/joshuawarren/src/omarchy-mlx/tools/profile_decode_driver.py
OUT=/home/joshuawarren/scratch/wB1-r2
ICD=/usr/lib/nvidia-egpu-615.78.08/share/vulkan/icd.d/nvidia_icd.json

run() { # $1=tag $2..=env
  local tag=$1; shift
  env "$@" VK_ICD_FILENAMES=$ICD MLX_OMARCHY_ALLOW_NON_APPLE=1 \
    MLX_OMARCHY_VRAM_ACTIVATIONS=1 MLX_OMARCHY_VRAM_WEIGHTS=$MLXMODEL \
    HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
    $PY -B $DRV --model $MLXMODEL --prompt-tokens 512 --tokens 64 --warmup-tokens 8 \
    --out $OUT/ab-$tag.jsonl > $OUT/ab-$tag.log 2>&1
  python3 - "$OUT/ab-$tag.jsonl" "$tag" <<PY
import json,statistics,hashlib
rows=[json.loads(l) for l in open("$OUT/ab-$tag.jsonl")]
toks=rows[1:]
rates=[1e9/(t["t1"]-t["t0"]) for t in toks if t.get("t1") and t.get("t0")]
ids=hashlib.sha256(",".join(str(t["tok"]) for t in toks).encode()).hexdigest()[:16]
print("$tag",round(statistics.median(rates),2),"tok_s","n",len(toks),"ids",ids)
PY
}

run a1 MLX_DISABLE_COMPILE=1
run b1
run b2
run a2 MLX_DISABLE_COMPILE=1
run a3 MLX_DISABLE_COMPILE=1
run b3
run b4
run a4 MLX_DISABLE_COMPILE=1
run a5 MLX_DISABLE_COMPILE=1
run b5
run b6bw MLX_OMARCHY_BATCH_WORK=40000
