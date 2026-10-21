#!/bin/bash
# Decode PR final head on the 14-inch M2 Max: build the mlx-omarchy wheel and the two doctest binaries of branch w7b/gdn-bf16a-hk-eq-hv-g14c at one pinned commit
# in a fresh clone (src-pr3), unpack the wheel into tree-pr3, make venv-pr3 from the step1 venv with that wheel, run the doctests, then the 128-token log-prob
# comparison on Qwen3.8-2B 4-bit of the plain prefetch kernel (MLX_OMARCHY_GDN_PF_WIDE=0) against the new default (wide prefetch kernel), and of the tiled kernel
# against the new default. The build is CPU only; the doctests and the comparisons use the GPU. No file outside src-pr3, tree-pr3, venv-pr3 and pr3-num-* changes.
set -euo pipefail
sha=356c2df282513057385df5fdc747983e77232fd4
D=$HOME/work
src=$D/src-pr3
tree=$D/tree-pr3
vp=$D/venv-pr3
bundle=$D/whole-bundle
tag=$(date -u +%m%dT%H%M%S)
echo "PR3 start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) sha $sha load1=$(cut -d' ' -f1 /proc/loadavg)"
test ! -e "$tree" && test ! -e "$src" && test ! -e "$vp" || { echo "PR3 abort: an output directory exists"; exit 1; }
[ "$(sha256sum "$bundle/manifest.json" | cut -c1-64)" = 08769793f8ee3299381f499bf90537d620e23635000a6dbc54b9b5c6a55a54ab ] || { echo "bundle manifest differs"; exit 1; }
[ "$(sha256sum "$bundle/program-0.anec" | cut -c1-64)" = 13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453 ] || { echo "bundle program differs"; exit 1; }
git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin w7b/gdn-bf16a-hk-eq-hv-g14c
git -C "$src" reset -q --hard "$sha"
[ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
cd "$src"
nice -n 19 env DEV_RELEASE=1 MLX_OMARCHY_WORK_DIR="$src/.work" MLX_OMARCHY_WHOLE_BUNDLE_DIR="$bundle" bash scripts/build-wheel.sh > "$src/wheel.log" 2>&1 \
  || { echo "wheel build FAILED, tail:"; tail -20 "$src/wheel.log"; exit 1; }
wheel=$(ls "$src"/dist/mlx_omarchy-*.whl)
echo "PR3 wheel $(basename "$wheel") sha256 $(sha256sum "$wheel" | cut -c1-64)"
case "$wheel" in *diag*) echo "wheel name has a diag stamp: $wheel"; exit 1 ;; esac
mkdir "$tree"
"$HOME/.local/share/coreglass/venv-main/bin/python" -m zipfile -e "$wheel" "$tree"
test -d "$tree/mlx"
echo "PR3 tree libmlx $(sha256sum "$tree/mlx/lib/libmlx.so" | cut -c1-16)"
nprof=$(strings -a "$tree/mlx/lib/libmlx.so" | grep -c 'MLX_OMARCHY_GPU_PROFILE' || true)
[ "$nprof" -eq 0 ] || { echo "GPU profiling harness is compiled into this libmlx"; exit 1; }
cp -a "$HOME/work/base-venv" "$vp"
for f in "$vp"/bin/*; do
  if [ -f "$f" ] && [ ! -L "$f" ] && head -c 2 "$f" | grep -q '#!'; then sed -i "1s|^#!.*/python3\\?\$|#!$vp/bin/python3|" "$f"; fi
done
"$vp/bin/python" -m pip install -q --no-deps --force-reinstall "$wheel"
echo "PR3 venv mx $("$vp/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx $(sha256sum "$vp"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
work="$src/.work/mlx"
(cd "$work" && nice -n 19 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas . > "$src/tests-cmake.log" 2>&1 \
  && nice -n 19 cmake --build . --target omarchy_gdn_gqa_bf16a_tests omarchy_runtime_tests -j8 > "$src/tests-build.log" 2>&1) || { echo "PR3 tests build FAILED"; tail -15 "$src/tests-cmake.log" "$src/tests-build.log"; exit 1; }
echo "PR3 tests built $(date -u +%FT%TZ)"
set +e
for t in omarchy_gdn_gqa_bf16a_tests omarchy_runtime_tests; do
  bin=$(find "$work" -name "$t" -type f -executable | head -1)
  echo "PR3 doctest $t $(sha256sum "$bin" | cut -c1-16)"
  env -i HOME="$HOME" PATH=/usr/bin:/bin VK_DRIVER_FILES="$HOME/.local/share/coreglass/vulkan-main/honeykrisp_icd.aarch64.json" timeout 280 "$bin" 2>&1 | grep -a -E 'test cases|assertions|Status|ERROR|gdn_pf_wide|gdn_bf16a_untiled_default\] ratio 1|FAILED' | tail -12
  echo "PR3 doctest $t rc=${PIPESTATUS[0]}"
done
export VP=$vp WHL=$wheel SC=$D/refill
M=$HOME/.local/share/coreglass/models/qwen3_5-4bit
bash "$D/refill/armab.sh" num "$D/pr3-num-plain-$tag" "$M" MLX_OMARCHY_GDN_PF_WIDE=0 MLX_AB_UNUSED=1 2>&1 | grep -a -E '^AB (start|model|wheel|numerics|census|compare)|ids_equal|max_abs|top1|argmax|bound|VACUITY|mean' | cut -c1-200
bash "$D/refill/armab.sh" num "$D/pr3-num-tile-$tag" "$M" MLX_OMARCHY_GDN_DECODE_TILE=1 MLX_AB_UNUSED=1 2>&1 | grep -a -E '^AB (start|model|wheel|numerics|census|compare)|ids_equal|max_abs|top1|argmax|bound|VACUITY|mean' | cut -c1-200
echo "PR3 end $(date -u +%FT%TZ)"
