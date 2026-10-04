#!/usr/bin/env bash
# Runs INSIDE one gpuwin window: 2B (bf16 A_log, flag bit 9 off) greedy-id
# digests on the reference wheel vs the candidate wheel, tiled and perrow.
# usage: nonreg_window.sh <ref-python> <cand-python> <tools-dir> <tokens> <prompt...>
set -uo pipefail
REF="$1"; CAND="$2"; TOOLS="$3"; N="$4"; shift 4
M2B=$(ls -d "$HOME"/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B*/snapshots/*/ | head -1)
for ARM in tiled perrow_pf; do
  case $ARM in
    tiled) ENVS=() ;;
    perrow_pf) ENVS=(MLX_OMARCHY_GDN_DECODE_TILE=0) ;;
  esac
  for SIDE in ref cand; do
    PY=$REF; [ "$SIDE" = cand ] && PY=$CAND
    echo "== $ARM $SIDE $("$PY" -c 'import mlx.core as mx; print(mx.__version__)')"
    env "${ENVS[@]}" timeout 200 "$PY" "$TOOLS/ids_digest.py" "$M2B" "$N" "$@" 2>&1 | grep '^p'
  done
done
