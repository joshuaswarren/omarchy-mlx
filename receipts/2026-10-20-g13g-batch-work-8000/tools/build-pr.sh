#!/bin/bash
# build-pr.sh BRANCH SHA : on the 16-inch, build the wheel of the PR commit and the omarchy_runtime_tests binary (CPU only; the budget doctests are pure).
set -euo pipefail
BR=${1:?branch}; SHA=${2:?sha}
B=$HOME/.local/share/coreglass
S=$B/build
src=$S/omarchy-mlx-bis
mkdir -p "$S/dist-bis"
[ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin "$BR"
git -C "$src" reset -q --hard "$SHA"
short=$(git -C "$src" rev-parse --short=7 HEAD)
echo "building $short start $(date -u +%FT%TZ)"
(cd "$src" && nice -n 10 env DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR="$B/whole-bundle" MLX_OMARCHY_WORK_DIR="$S/work" bash scripts/build-wheel.sh) > "$S/wheel-pr-$short.log" 2>&1 || { tail -15 "$S/wheel-pr-$short.log"; exit 1; }
mv -f "$src"/dist/mlx_omarchy-*.whl "$S/dist-bis/"
ls "$S/dist-bis/"mlx_omarchy-*"+$short-"*.whl
sha256sum "$S/dist-bis/"mlx_omarchy-*"+$short-"*.whl | cut -c1-80
cd "$S/work/mlx"
nice -n 10 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas . > "$S/tests-pr-cmake.log" 2>&1 || { tail -10 "$S/tests-pr-cmake.log"; exit 1; }
nice -n 10 cmake --build . --target omarchy_runtime_tests -j8 > "$S/tests-pr-build.log" 2>&1 || { tail -20 "$S/tests-pr-build.log"; exit 1; }
bin=$(find "$S/work/mlx" -name omarchy_runtime_tests -type f -executable | head -1)
echo "test binary $bin $(sha256sum "$bin" | cut -c1-64)"
"$bin" --test-case='submission-cap default resolves per calibrated board,work budget resolves per device name, not per process' 2>&1 | tail -8
echo "done $(date -u +%FT%TZ)"
