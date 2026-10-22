#!/bin/bash
# Decode PR for G13C at equal head counts: build the mlx-omarchy wheel and the two doctest binaries of branch w7b/gdn-wide-g13c-equal-heads at one pinned commit on the 14-inch M2 Max
# in a fresh clone (src-g13m2), unpack the wheel into tree-g13m2, and run the doctests on this machine's GPU (Apple M2 Max, G14C): the G14C expectations of the changed rule.
# No measurement. No file outside src-g13m2 and tree-g13m2 changes.
set -euo pipefail
sha=ba12383f5c1c44fb2d69692bf79864b271c2d7e9
D=$HOME/work
src=$D/src-g13m2
tree=$D/tree-g13m2
bundle=$D/whole-bundle
echo "G13M2 start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) sha $sha load1=$(cut -d' ' -f1 /proc/loadavg)"
test ! -e "$tree" && test ! -e "$src" || { echo "G13M2 abort: an output directory exists"; exit 1; }
[ "$(sha256sum "$bundle/manifest.json" | cut -c1-64)" = 08769793f8ee3299381f499bf90537d620e23635000a6dbc54b9b5c6a55a54ab ] || { echo "bundle manifest differs"; exit 1; }
[ "$(sha256sum "$bundle/program-0.anec" | cut -c1-64)" = 13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453 ] || { echo "bundle program differs"; exit 1; }
git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin w7b/gdn-wide-g13c-equal-heads
git -C "$src" reset -q --hard "$sha"
[ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
cd "$src"
nice -n 19 env DEV_RELEASE=1 MLX_OMARCHY_WORK_DIR="$src/.work" MLX_OMARCHY_WHOLE_BUNDLE_DIR="$bundle" bash scripts/build-wheel.sh > "$src/wheel.log" 2>&1 \
  || { echo "wheel build FAILED, tail:"; tail -20 "$src/wheel.log"; exit 1; }
wheel=$(ls "$src"/dist/mlx_omarchy-*.whl)
echo "G13M2 wheel $(basename "$wheel") sha256 $(sha256sum "$wheel" | cut -c1-64)"
case "$wheel" in *diag*) echo "wheel name has a diag stamp: $wheel"; exit 1 ;; esac
mkdir "$tree"
"$HOME/.local/share/coreglass/venv-main/bin/python" -m zipfile -e "$wheel" "$tree"
test -d "$tree/mlx"
echo "G13M2 tree libmlx $(sha256sum "$tree/mlx/lib/libmlx.so" | cut -c1-16)"
nprof=$(strings -a "$tree/mlx/lib/libmlx.so" | grep -c 'MLX_OMARCHY_GPU_PROFILE' || true)
[ "$nprof" -eq 0 ] || { echo "GPU profiling harness is compiled into this libmlx"; exit 1; }
work="$src/.work/mlx"
(cd "$work" && nice -n 19 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas . > "$src/tests-cmake.log" 2>&1 \
  && nice -n 19 cmake --build . --target omarchy_gdn_gqa_bf16a_tests omarchy_runtime_tests -j8 > "$src/tests-build.log" 2>&1) || { echo "G13M2 tests build FAILED"; tail -15 "$src/tests-cmake.log" "$src/tests-build.log"; exit 1; }
echo "G13M2 tests built $(date -u +%FT%TZ)"
set +e
for t in omarchy_gdn_gqa_bf16a_tests omarchy_runtime_tests; do
  bin=$(find "$work" -name "$t" -type f -executable | head -1)
  echo "G13M2 doctest $t $(sha256sum "$bin" | cut -c1-16)"
  env -i HOME="$HOME" PATH=/usr/bin:/bin VK_DRIVER_FILES="$HOME/.local/share/coreglass/vulkan-main/honeykrisp_icd.aarch64.json" timeout 280 "$bin" 2>&1 | grep -a -E 'test cases|assertions|Status|ERROR|gdn_pf_wide|gdn_bf16a_untiled_default\] ratio 1|FAILED' | tail -12
  echo "G13M2 doctest $t rc=${PIPESTATUS[0]}"
done
echo "G13M2 end $(date -u +%FT%TZ)"
