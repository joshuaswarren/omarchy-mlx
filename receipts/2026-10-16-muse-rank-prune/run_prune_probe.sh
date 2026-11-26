#!/bin/bash
# run_prune_probe.sh: four CPU ranks on one machine, prune off then on, one shared checkpoint file.
# Needs: MUSE_SERVE_DIR (this repo's serve directory), OMARCHY_CLUSTER_DIR (an omarchy-cluster checkout),
# a Python with stock mlx, mlx-lm and numpy as PYTHON (default python3), and mlx.launch on PATH or as MLX_LAUNCH.
set -u
: "${MUSE_SERVE_DIR:?set MUSE_SERVE_DIR}" "${OMARCHY_CLUSTER_DIR:?set OMARCHY_CLUSTER_DIR}"
PYTHON=${PYTHON:-python3}
LAUNCH=${MLX_LAUNCH:-mlx.launch}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
CKPT=$WORK/tiny.safetensors
for P in 0 1; do
  timeout 100 "$LAUNCH" --backend ring --hosts 127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
    --env PYTHONUNBUFFERED=1 --env MLX_OMARCHY_PIPELINE_PRUNE=$P \
    --env MUSE_SERVE_DIR="$MUSE_SERVE_DIR" --env OMARCHY_CLUSTER_DIR="$OMARCHY_CLUSTER_DIR" \
    "$PYTHON" "$HERE/prune_probe.py" "$CKPT" > "$WORK/prune$P.log" 2>&1
  echo "== PRUNE=$P (log $WORK/prune$P.log)"
  grep -E "^rank|Error|Traceback" "$WORK/prune$P.log" | sort | cut -c1-230
done
