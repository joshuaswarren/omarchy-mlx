#!/usr/bin/env bash
# Runs INSIDE one gpuwin window: per-call fused-vs-composed replay on each
# decode kernel arm (tiled = jw16 default; perrow_pf / perrow = the G13
# kernels forced on this part), then optionally the 10x512 free-run bar.
# usage: arms_window.sh <venv-python> <tools-dir> <nstep> <freerun 0|1> <prompt...>
set -uo pipefail
PY="$1"; TOOLS="$2"; NSTEP="$3"; FREERUN="$4"; shift 4
M9B=$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
"$PY" -c "import mlx.core as mx; print('wheel', mx.__version__)"
for ARM in tiled perrow_pf perrow; do
  case $ARM in
    tiled) ENVS=() ;;
    perrow_pf) ENVS=(MLX_OMARCHY_GDN_DECODE_TILE=0) ;;
    perrow) ENVS=(MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF=0) ;;
  esac
  for P in "$@"; do
    O=/var/tmp/dfuse/arms/$ARM-p$P
    mkdir -p "$O"
    env -u MLX_OMARCHY_GDN_RAW_REPEAT "${ENVS[@]}" \
      timeout 240 "$PY" "$TOOLS/bitexact_bisect.py" "$M9B" "$O" "$NSTEP" "$P" > "$O/bisect.log" 2>&1
    echo "$ARM p$P rc=$? ok=$(grep -c '^OK' "$O/bisect.log") diff=$(grep -c '^DIFF' "$O/bisect.log") $(grep -o 'identical=[A-Za-z]*' "$O/bisect.log" | tail -1) $(grep -o 'CONTROL[^(]*' "$O/bisect.log")"
  done
done
if [ "$FREERUN" = 1 ]; then
  env -u MLX_OMARCHY_GDN_RAW_REPEAT timeout 700 "$PY" "$TOOLS/free_run_gaps.py" "$M9B" \
    /var/tmp/dfuse/arms/free-run-gaps.json 10 512 > /var/tmp/dfuse/arms/free-run.log 2>&1
  echo "freerun rc=$?"; tail -2 /var/tmp/dfuse/arms/free-run.log | cut -c1-600
fi
