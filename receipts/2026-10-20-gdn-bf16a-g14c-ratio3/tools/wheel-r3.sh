#!/bin/bash
# CPU-only build on the 16-inch M1 Max of omarchy-mlx at one commit: the wheel, and (WITH_TESTS=1) the omarchy_gdn_gqa_bf16a_tests binary.
# Same clone, work dir and ccache as the daily ledger build (the daily ledger build script). No GPU kernel runs. Output: ~/.local/share/coreglass/build/dist-r3/<tag>/
# usage: wheel-r3.sh TAG BRANCH SHA [WITH_TESTS]
set -euo pipefail
tag=${1:?tag}
branch=${2:?branch}
sha=${3:?commit sha}
tests=${4:-0}
B=$HOME/.local/share/coreglass
S=$B/build
src=$S/omarchy-mlx-r3
out=$S/dist-r3/$tag
[ ! -e "$out" ] || { echo "ABORT: $out exists"; exit 1; }
mkdir -p "$out"
[ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin "$branch"
git -C "$src" reset -q --hard "$sha"
[ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
git -C "$src" status --short | head -3
echo "building omarchy-mlx $sha ($tag) start $(date -u +%FT%TZ) kernel $(uname -r) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id)"
(cd "$src" && ionice -c3 nice -n 19 env DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR="$B/whole-bundle" MLX_OMARCHY_WORK_DIR="$S/work" \
  bash scripts/build-wheel.sh) > "$out/wheel.log" 2>&1 || { echo "wheel build FAILED, tail:"; tail -20 "$out/wheel.log"; exit 1; }
mv -f "$src"/dist/mlx_omarchy-*.whl "$out/"
ls -l "$out/"*.whl
sha256sum "$out"/mlx_omarchy-*.whl
echo "wheel done $(date -u +%FT%TZ)"
if [ "$tests" = 1 ]; then
  tree=$S/work/mlx
  [ -d "$tree" ] || { echo "no build tree $tree"; exit 1; }
  cd "$tree"
  ionice -c3 nice -n 19 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas . > "$out/tests-cmake.log" 2>&1 || { echo "cmake FAILED"; tail -10 "$out/tests-cmake.log"; exit 1; }
  ionice -c3 nice -n 19 cmake --build . --target omarchy_gdn_gqa_bf16a_tests -j8 > "$out/tests-build.log" 2>&1 || { echo "test build FAILED"; tail -20 "$out/tests-build.log"; exit 1; }
  bin=$(find "$tree" -name omarchy_gdn_gqa_bf16a_tests -type f -executable | head -1)
  cp "$bin" "$out/omarchy_gdn_gqa_bf16a_tests"
  echo "test binary $out/omarchy_gdn_gqa_bf16a_tests $(sha256sum "$out/omarchy_gdn_gqa_bf16a_tests" | cut -c1-64)"
fi
echo "done $(date -u +%FT%TZ)"
