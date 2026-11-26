#!/usr/bin/env bash
# Deploy-verify: serving venv, NO env vars (fold default ON), digest pins.
set -euo pipefail
PY=/var/tmp/v072-venv-fused/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
HF=${HOME}/.cache/huggingface/hub
M4B=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)
M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/deploy-verify
mkdir -p "$O"
env | grep -q MLX_OMARCHY_ROPE_NORM_FUSE && { echo "gate env must be unset"; exit 1; }

run() { # model depth label
  timeout 300 "$PY" "$B" --model "$1" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$2" --prefill-tokens 512 --label "$3" --out "$O/$3.json" \
    > "$O/$3.log" 2> "$O/$3.err"
  echo "deploy-verify $3: $(grep -o '"ordered_records_sha256[^,}]*' "$O/$3.json")"
}

sudo journalctl --flush; sync
bash /var/tmp/appbar/gpuwin.sh "timeout 800 bash /var/tmp/dfuse/verify_inner.sh"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
curl -s -m 8 http://127.0.0.1:8002/health && echo " health-ok"
