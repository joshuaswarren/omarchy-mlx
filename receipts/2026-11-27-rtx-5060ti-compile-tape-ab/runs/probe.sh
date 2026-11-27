#!/usr/bin/env bash
# R2 probe (wB1): (a) does llama-server CUDA engage CUDA graphs on the 5060 Ti;
# (b) MLX-Vulkan decode baseline + per-call fixed cost on the NVIDIA ICD.
set -u
CBIN=/home/joshuawarren/scratch/serve-ab/llama-cuda-build/bin
GGUF=/home/joshuawarren/scratch/serve-ab/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
MLXMODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
PY=/usr/lib/omarchy-mlx/venv/bin/python
DRV=/home/joshuawarren/src/omarchy-mlx/tools/profile_decode_driver.py
OUT=/home/joshuawarren/scratch/wB1-r2
PORT=8157

srv_arm() { # $1=label $2..=extra env
  local label=$1; shift
  local logf=$OUT/srv-$label.log
  env "$@" LD_LIBRARY_PATH=$CBIN \
    $CBIN/llama-server -m $GGUF --port $PORT --host 127.0.0.1 \
    -ngl 99 -c 2048 --parallel 1 -fa on --no-warmup >$logf 2>&1 &
  local srv=$!
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
  curl -s http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" \
    -d "{\"prompt\":\"The history of France begins in antiquity.\",\"n_predict\":128,\"temperature\":0,\"cache_prompt\":false}" \
    > $OUT/resp-$label.json
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  echo "=== cuda arm $label ==="
  python3 - "$OUT/resp-$label.json" <<PY
import json
d=json.load(open("$OUT/resp-$label.json"))
t=d.get("timings",{})
print("predicted_ms",t.get("predicted_ms"),"predicted_per_second",t.get("predicted_per_second"),"n",t.get("predicted_n"))
PY
  grep -icE "graph" $logf; grep -iE "graph.*(capture|reuse|disabl|fail)|reused" $logf | tail -6
}

mlx_arm() { # $1=label $2..=extra env
  local label=$1; shift
  echo "=== mlx arm $label ==="
  env "$@" VK_ICD_FILENAMES=/usr/lib/nvidia-egpu-615.78.08/share/vulkan/icd.d/nvidia_icd.json \
    MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_VRAM_ACTIVATIONS=1 \
    MLX_OMARCHY_VRAM_WEIGHTS=$MLXMODEL \
    HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
    $PY -B $DRV --model $MLXMODEL --prompt-tokens 512 --tokens 64 --warmup-tokens 8 \
    --out $OUT/mlx-$label.jsonl > $OUT/mlx-$label.log 2>&1
  python3 - "$OUT/mlx-$label.jsonl" <<PY
import json,statistics,hashlib
rows=[json.loads(l) for l in open("$OUT/mlx-$label.jsonl")]
meta,toks=rows[0],rows[1:]
rates=[1e9/(t["t1"]-t["t0"]) for t in toks if t.get("t1") and t.get("t0")]
ids=hashlib.sha256(",".join(str(t["tok"]) for t in toks).encode()).hexdigest()[:16]
print("median_tok_s",round(statistics.median(rates),2),"n",len(toks),"digest",ids)
print("meta",{k:v for k,v in meta.items() if k in ("wheel","device","commit")})
PY
}

srv_arm graphs_on
srv_arm graphs_off GGML_CUDA_DISABLE_GRAPHS=1
srv_arm gopt GGML_CUDA_GRAPH_OPT=1
mlx_arm base MLX_DISABLE_COMPILE=1
mlx_arm bw MLX_OMARCHY_BATCH_WORK=1 MLX_DISABLE_COMPILE=1
# compile-tape arms: the fused chain only engages on the compiled path
mlx_arm compile
mlx_arm compile_nofc MLX_OMARCHY_FUSED_CHAIN=0
