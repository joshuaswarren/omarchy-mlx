#!/bin/bash
# Build the omarchy-mlx wheel of main at aaef3b6c (the wide GDN decode kernel, PR 252) on the 16-inch M1 Max, with the work directory and ccache of the daily ledger
# (a rebuild takes minutes), and unpack it into tree-main2. CPU build only: no GPU kernel runs. Output: ~/.local/share/coreglass/build/dist-main2/ and ~/work/tree-main2.
set -euo pipefail
sha=aaef3b6c95832c9cfcfcced3bd37d50b9bdc0c3e
B=$HOME/.local/share/coreglass
S=$B/build
src=$S/omarchy-mlx-main2
tree=$HOME/work/tree-main2
test ! -e "$tree" || { echo "tree exists: $tree"; exit 1; }
mkdir -p "$S/dist-main2"
[ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin main
git -C "$src" reset -q --hard "$sha"
[ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
echo "building omarchy-mlx $sha start $(date -u +%FT%TZ) kernel $(uname -r) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) load1 $(cut -d' ' -f1 /proc/loadavg)"
(cd "$src" && nice -n 19 env DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR="$B/whole-bundle" MLX_OMARCHY_WORK_DIR="$S/work" \
  bash scripts/build-wheel.sh) > "$S/wheel-main2.log" 2>&1 || { echo "wheel build FAILED, tail:"; tail -20 "$S/wheel-main2.log"; exit 1; }
mv -f "$src"/dist/mlx_omarchy-*.whl "$S/dist-main2/"
wheel=$(find "$S/dist-main2" -name 'mlx_omarchy-*.whl' | head -1)
case "$wheel" in *diag*) echo "wheel name has a diag stamp: $wheel"; exit 1 ;; esac
echo "wheel $(basename "$wheel") sha256 $(sha256sum "$wheel" | cut -c1-64)"
mkdir "$tree"
"$B/venv-main/bin/python" -m zipfile -e "$wheel" "$tree"
test -f "$tree/mlx/lib/libmlx.so"
echo "tree libmlx $(sha256sum "$tree/mlx/lib/libmlx.so" | cut -c1-16)"
echo "PF_WIDE strings $(strings -a "$tree/mlx/lib/libmlx.so" | grep -c MLX_OMARCHY_GDN_PF_WIDE || true)"
nprof=$(strings -a "$tree/mlx/lib/libmlx.so" | grep -c 'MLX_OMARCHY_GPU_PROFILE' || true)
[ "$nprof" -eq 0 ] || { echo "GPU profiling harness is compiled into this libmlx"; exit 1; }
echo "done $(date -u +%FT%TZ) wheel rc 0"
