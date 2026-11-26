#!/bin/bash
# Build the omarchy-mlx wheel of the branch sdpa-split-bf16 at 6943e37a (split-K decode SDPA as the per-part default) on the Apple M2 Max machine in a fresh clone (src-final) and unpack it into tree-final.
# CPU only: no GPU kernel runs, no measurement. No file outside src-final and tree-final changes.
set -euo pipefail
sha=6943e37ad95f1c28e385ad4c5dc4c4384ad62254
D=$HOME/work
src=$D/src-final
tree=$D/tree-final
bundle=$HOME/.local/share/coreglass/whole-bundle
echo "final wheel start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) sha $sha load1=$(cut -d' ' -f1 /proc/loadavg)"
test ! -e "$tree" && test ! -e "$src"
[ "$(sha256sum "$bundle/manifest.json" | cut -c1-64)" = 08769793f8ee3299381f499bf90537d620e23635000a6dbc54b9b5c6a55a54ab ] || { echo "bundle manifest differs"; exit 1; }
[ "$(sha256sum "$bundle/program-0.anec" | cut -c1-64)" = 13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453 ] || { echo "bundle program differs"; exit 1; }
git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin sdpa-split-bf16
git -C "$src" reset -q --hard "$sha"
[ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
cd "$src"
nice -n 19 env DEV_RELEASE=1 MLX_OMARCHY_WORK_DIR="$src/.work" MLX_OMARCHY_WHOLE_BUNDLE_DIR="$bundle" bash scripts/build-wheel.sh > "$src/wheel.log" 2>&1 \
  || { echo "wheel build FAILED, tail:"; tail -20 "$src/wheel.log"; exit 1; }
wheel=$(find "$src/dist" -name 'mlx_omarchy-*.whl' | head -1)
echo "wheel $(basename "$wheel") sha256 $(sha256sum "$wheel" | cut -c1-64)"
case "$wheel" in *diag*) echo "wheel name has a diag stamp: $wheel"; exit 1 ;; esac
mkdir "$tree"
"$HOME/.local/share/coreglass/venv-main/bin/python" -m zipfile -e "$wheel" "$tree"
test -f "$tree/mlx/lib/libmlx.so"
echo "tree libmlx $(sha256sum "$tree/mlx/lib/libmlx.so" | cut -c1-16)"
nprof=$(strings -a "$tree/mlx/lib/libmlx.so" | grep -c 'MLX_OMARCHY_GPU_PROFILE' || true)
[ "$nprof" -eq 0 ] || { echo "GPU profiling harness is compiled into this libmlx"; exit 1; }
echo "done $(date -u +%FT%TZ) wheel rc 0"
