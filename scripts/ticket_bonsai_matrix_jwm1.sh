#!/usr/bin/env bash
# BonsaiBuild full-matrix ticket: sync to origin/main, rebuild the wheel
# only if dist does not already carry the current HEAD, then run the whole
# Bonsai parity matrix (ticket_bonsai_gpu_jwm1.sh). Rerun-safe.
# Usage (on jwm1): bash ticket_bonsai_matrix_jwm1.sh [tree-dir]
set -euo pipefail

TREE=${1:-$HOME/bonsai-build}
REPO="$TREE/repo"
LOG="$TREE/matrix.log"
exec > >(tee -a "$LOG") 2>&1

echo "== bonsai matrix ticket start $(date -u +%FT%TZ)"
git -C "$REPO" fetch origin
git -C "$REPO" checkout --detach -q origin/main
HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
echo "head=$HEAD_SHA"

cd "$REPO"
CURRENT=$(ls -t dist/mlx_omarchy-*.whl 2>/dev/null | head -1 || true)
if [ -z "$CURRENT" ] || [[ "$CURRENT" != *"$HEAD_SHA"* ]]; then
  echo "== building wheel for $HEAD_SHA (dist stale: ${CURRENT:-none})"
  DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR="$TREE/whole-bundle" \
    CMAKE_BUILD_PARALLEL_LEVEL=4 bash scripts/build-wheel.sh
else
  echo "== dist wheel already carries $HEAD_SHA"
fi

bash "$TREE/ticket_bonsai_gpu_jwm1.sh" "$TREE"
echo "== bonsai matrix ticket done $(date -u +%FT%TZ)"
