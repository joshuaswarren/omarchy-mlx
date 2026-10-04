#!/usr/bin/env bash
# ppl probe: 9B mean-NLL with composed (gate off) vs fused (gate on).
set -euo pipefail
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
OLD=/var/tmp/v072-venv-fused.pre-20261004T000642/bin/python
NEW=/var/tmp/dfuse-cand/bin/python
O=/var/tmp/dfuse/ppl
mkdir -p "$O"
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$OLD" /var/tmp/dfuse/ppl_probe.py "$M9B" "$P" > "$O/ppl-composed.json"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 "$NEW" /var/tmp/dfuse/ppl_probe.py "$M9B" "$P" > "$O/ppl-fused.json"
echo "composed: $(cat $O/ppl-composed.json)"
echo "fused:    $(cat $O/ppl-fused.json)"
