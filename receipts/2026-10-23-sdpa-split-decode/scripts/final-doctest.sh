#!/bin/bash
# Split-K decode SDPA device test on the Apple M2 Max: build omarchy_sdpa_split_tests and omarchy_sdpa_decode_fused_tests at one pinned commit in a private source tree, then run both binaries
# (the split route against a float64 reference and the fused kernel, dispatch counts, uncovered shapes; the existing fused-decode tests unchanged).
# Stage build is CPU only and incremental (re-run it after a timeout); stage run uses the GPU and writes nothing outside OUT.
# usage: final-doctest.sh build | run OUT
set -euo pipefail
sha=6943e37ad95f1c28e385ad4c5dc4c4384ad62254
src=$HOME/work/src-vtail
stage=${1:?build|run}
export VK_DRIVER_FILES=/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json
echo "stage $stage boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) sha $sha start $(date -u +%FT%TZ) load1 $(cut -d' ' -f1 /proc/loadavg)"
if [ "$stage" = build ]; then
  [ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
  git -C "$src" fetch -q origin sdpa-split-bf16
  git -C "$src" reset -q --hard "$sha"
  [ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || { echo "HEAD is not $sha"; exit 1; }
  cd "$src"
  nice -n 19 ./scripts/prepare-mlx.sh > "$src/prepare.log" 2>&1 || { echo "prepare FAILED"; tail -5 "$src/prepare.log"; exit 1; }
  [ -f .work/build/build.ninja ] || nice -n 19 cmake -S .work/mlx -B .work/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INCLUDE_PATH=/usr/include/openblas -DOMARCHY_GLSLANG=/usr/bin/glslangValidator -DOMARCHY_GLSLC=/usr/bin/glslc \
    > "$src/cmake.log" 2>&1 || { echo "cmake FAILED"; tail -8 "$src/cmake.log"; exit 1; }
  nice -n 19 cmake --build .work/build --target omarchy_sdpa_split_tests omarchy_sdpa_decode_fused_tests -j12 > "$src/build.log" 2>&1 \
    || { echo "build FAILED"; tail -25 "$src/build.log"; exit 1; }
  sha256sum .work/build/tests/omarchy/omarchy_sdpa_split_tests .work/build/tests/omarchy/omarchy_sdpa_decode_fused_tests | cut -c1-64
  printf '%s\n' "$sha" > "$src/built-sha"
  echo "done $(date -u +%FT%TZ) build rc 0"
else
  out=${2:?out dir}
  test ! -e "$out"
  mkdir -p "$out"
  { [ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] && [ "$(cat "$src/built-sha")" = "$sha" ]; } || { echo "binary is not a build of $sha"; exit 1; }
  run_one() {
    local name=$1 bin one
    bin=$src/.work/build/tests/omarchy/$name
    sha256sum "$bin" | tee -a "$out/bin.sha256"
    set +e
    timeout 900 "$bin" > "$out/$name.txt" 2>&1
    one=$?
    set -e
    tail -6 "$out/$name.txt"
    echo "$name rc $one"
    if grep -q '^Skipping' "$out/$name.txt"; then
      echo "NOT CONFIRMED $name (a GPU case printed Skipping: no Vulkan device was used)"
      return 1
    elif grep -Eq 'test cases: +[0-9]+ \| +[0-9]+ passed \| +0 failed' "$out/$name.txt" && grep -Eq 'assertions: +[0-9]+ \| +[0-9]+ passed \| +0 failed' "$out/$name.txt"; then
      echo "ALL CASES RAN, 0 FAILED, no case skipped: $name $(grep -E 'assertions:' "$out/$name.txt" | head -1)"
      return "$one"
    fi
    echo "NOT CONFIRMED $name (no all-passed summary)"
    return 1
  }
  rc=0
  run_one omarchy_sdpa_split_tests || rc=1
  run_one omarchy_sdpa_decode_fused_tests || rc=1
  echo "done $(date -u +%FT%TZ) rc $rc"
  exit $rc
fi
