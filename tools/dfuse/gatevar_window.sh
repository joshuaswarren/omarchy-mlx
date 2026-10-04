#!/usr/bin/env bash
# Runs INSIDE one gpuwin window: per-call replay bisect for each gate variant.
# usage: gatevar_window.sh <venv-python> <bisect.py> <nstep> <prompt...>
set -uo pipefail
PY="$1"; BIS="$2"; NSTEP="$3"; shift 3
M9B=$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
"$PY" -c "import mlx.core as mx; print('wheel', mx.__version__)"
for VAR in 0 1 2; do
  for P in "$@"; do
    O=/var/tmp/dfuse/gatevar/v$VAR-p$P
    mkdir -p "$O"
    env -u MLX_OMARCHY_GDN_RAW_REPEAT MLX_OMARCHY_GDN_GATE_VARIANT=$VAR \
      timeout 240 "$PY" "$BIS" "$M9B" "$O" "$NSTEP" "$P" > "$O/bisect.log" 2>&1
    rc=$?
    echo "v$VAR p$P rc=$rc ok=$(grep -c '^OK' "$O/bisect.log") diff=$(grep -c '^DIFF' "$O/bisect.log") $(grep -o 'identical=[A-Za-z]*' "$O/bisect.log" | tail -1) $(grep -o 'CONTROL[^(]*' "$O/bisect.log")"
  done
done
