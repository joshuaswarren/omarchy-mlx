#!/usr/bin/env bash
# FamGlmDsa incremental M2 wheel build (Main's shortcut recipe, no
# prepare-mlx: it would rm -rf the staged tree and its build objects).
# Runs ON the M2. Log: /var/tmp/FamGlmDsa-wheel/build-dsa.log
set -euo pipefail
COPY=/var/tmp/FamGlmDsa-wheel
PATCH="$HOME/dsa-build/mlx-fast-dsa-indexer.patch"
OVERLAY="$HOME/dsa-build/overlay"

if [ ! -d "$COPY" ]; then
  cp -a /var/tmp/od-distributed-wheel-20261004 "$COPY"
fi
cd "$COPY"

# 1. Apply the DSA patch to the already-staged, already-patched tree.
patch -d .work/mlx -p1 --forward --fuzz=0 < "$PATCH"

# 2. Copy the changed omarchy overlay files into the staged tree.
cp "$OVERLAY/shaders/dsa_indexer.comp" .work/mlx/shaders/ 2>/dev/null || true
find "$OVERLAY" -type f | while read -r f; do
  rel=${f#"$OVERLAY"/}
  mkdir -p ".work/mlx/$(dirname "$rel")"
  cp "$f" ".work/mlx/$rel"
done
# Keep provenance copies at the repo root too.
mkdir -p patches
cp "$PATCH" patches/
if [ -d overlay ]; then cp -r "$OVERLAY"/. overlay/; else cp -r "$OVERLAY" overlay; fi

# 3. Incremental wheel build from the staged tree (venv-build from the
# od copy carries the pinned build deps).
PY=.work/venv-build/bin/python
[ -x "$PY" ] || PY=python3
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_OMARCHY_ANE_DEVICE=ON"
export CMAKE_BUILD_PARALLEL_LEVEL=3
export DEV_RELEASE=1
echo "== incremental wheel build ($(date -u)) =="
nice -n 19 ionice -c3 "$PY" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir dist .work/mlx
echo "BUILD DONE ($(date -u))"
ls -la dist/
