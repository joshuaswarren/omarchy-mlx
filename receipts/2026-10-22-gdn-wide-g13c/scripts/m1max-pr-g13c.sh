#!/bin/bash
# Decode PR for G13C at equal head counts, on the 16-inch M1 Max: build the mlx-omarchy wheel of branch w7b/gdn-wide-g13c-equal-heads at one pinned commit (a private work directory in the clone, the shared ccache),
# unpack it into tree-g13c, build the two doctest binaries in the same work tree, run them on this machine's GPU (Apple M1 Max, G13C) and print their exit codes.
# No measurement. No file outside the build clone (with its private work directory), tree-g13c and the dist directory changes; the ledger work tree is not touched.
set -uo pipefail
sha=ba12383f5c1c44fb2d69692bf79864b271c2d7e9
B=$HOME/.local/share/coreglass
S=$B/build
src=$S/omarchy-mlx-g13c
tree=$HOME/work/tree-g13c
work=$src/.work/mlx
echo "PRG13 start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) sha $sha load1=$(cut -d' ' -f1 /proc/loadavg)"
test ! -e "$tree" || { echo "tree exists: $tree"; exit 1; }
mkdir -p "$S/dist-g13c"
[ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src" || exit 1
git -C "$src" fetch -q origin w7b/gdn-wide-g13c-equal-heads || exit 1
git -C "$src" reset -q --hard "$sha" || exit 1
[ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
(cd "$src" && nice -n 19 env DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR="$B/whole-bundle" MLX_OMARCHY_WORK_DIR="$src/.work" bash scripts/build-wheel.sh) > "$S/wheel-g13c.log" 2>&1 \
  || { echo "wheel build FAILED, tail:"; tail -20 "$S/wheel-g13c.log"; exit 1; }
mv -f "$src"/dist/mlx_omarchy-*.whl "$S/dist-g13c/"
wheel=$(find "$S/dist-g13c" -name 'mlx_omarchy-*.whl' | head -1)
case "$wheel" in *diag*) echo "wheel name has a diag stamp: $wheel"; exit 1 ;; esac
echo "PRG13 wheel $(basename "$wheel") sha256 $(sha256sum "$wheel" | cut -c1-64)"
mkdir "$tree"
"$B/venv-main/bin/python" -m zipfile -e "$wheel" "$tree"
test -f "$tree/mlx/lib/libmlx.so" || exit 1
echo "PRG13 tree libmlx $(sha256sum "$tree/mlx/lib/libmlx.so" | cut -c1-16)"
nprof=$(strings -a "$tree/mlx/lib/libmlx.so" | grep -c 'MLX_OMARCHY_GPU_PROFILE' || true)
[ "$nprof" -eq 0 ] || { echo "GPU profiling harness is compiled into this libmlx"; exit 1; }
(cd "$work" && nice -n 19 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas . > "$S/tests-g13c-cmake.log" 2>&1 \
  && nice -n 19 cmake --build . --target omarchy_gdn_gqa_bf16a_tests omarchy_runtime_tests -j8 > "$S/tests-g13c-build.log" 2>&1) \
  || { echo "PRG13 tests build FAILED"; tail -15 "$S/tests-g13c-cmake.log" "$S/tests-g13c-build.log"; exit 1; }
echo "PRG13 tests built $(date -u +%FT%TZ)"
for t in omarchy_gdn_gqa_bf16a_tests omarchy_runtime_tests; do
  bin=$(find "$work" -name "$t" -type f -executable | head -1)
  echo "PRG13 doctest $t $(sha256sum "$bin" | cut -c1-16)"
  env -i HOME="$HOME" PATH=/usr/bin:/bin VK_DRIVER_FILES="$B/vulkan-main/honeykrisp_icd.aarch64.json" timeout 280 "$bin" 2>&1 | grep -a -E 'test cases|assertions|Status|ERROR|gdn_pf_wide|gdn_bf16a_untiled_default\] ratio 1|FAILED' | tail -12
  echo "PRG13 doctest $t rc=${PIPESTATUS[0]}"
done
echo "PRG13 end $(date -u +%FT%TZ)"
