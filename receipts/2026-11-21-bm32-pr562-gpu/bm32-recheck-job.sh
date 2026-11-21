#!/usr/bin/env bash
# PR 562 (w85/steel-gemm-port) recheck ticket r4: fresh worktree at the
# tested commit, wheel build with pinned bundle+licence env, fresh venv,
# module provenance, then the nemotron BM32 KernelRecheck row.
# $1 = receipt path. Re-runnable.
set -euo pipefail
SHA=7871b7e2b1b22419fa46e46f0236e9c55dea3544
BRANCH=w85/steel-gemm-port
RECEIPT="$1"
mkdir -p "$(dirname "$RECEIPT")"
exec > >(tee "$RECEIPT") 2>&1
echo "ticket-start $(date -u +%FT%TZ) sha=$SHA host=$(hostname)"
uname -a
export MLX_OMARCHY_WHOLE_BUNDLE_DIR="$HOME/scratch/w7m-r49/bundle/parakeet-encoder-whole"
export MLX_OMARCHY_GCC_LICENSE_DIR="$HOME/scratch/w7m-r49/licenses"
WORK=/home/joshuawarren/scratch/w85/bm32-recheck-r4
cd /home/joshuawarren/src/omarchy-mlx
git fetch origin "$BRANCH"
git worktree remove --force "$WORK/tree" 2>/dev/null || true
rm -rf "$WORK"
mkdir -p "$WORK"
git worktree add "$WORK/tree" "$SHA"
cd "$WORK/tree"
test "$(git rev-parse HEAD)" = "$SHA"
echo "checkout: $(git rev-parse HEAD)"
./scripts/prepare-mlx.sh
./scripts/build-wheel.sh
ls -la dist/
python3 -m venv --system-site-packages "$WORK/venv"
"$WORK/venv/bin/pip" install --no-input "$WORK"/tree/dist/*.whl numpy 2>&1 | tail -3
"$WORK/venv/bin/python" -c "import mlx.core as mx; print('MODULE', mx.__file__); print('VERSION', mx.__version__)"
cd "$WORK/tree"
"$WORK/venv/bin/python" -m tools.kernel_recheck.run --gpu --kernel nemotron_bm32_steel_linear_nt --timeout 120
echo "ticket-end $(date -u +%FT%TZ)"
