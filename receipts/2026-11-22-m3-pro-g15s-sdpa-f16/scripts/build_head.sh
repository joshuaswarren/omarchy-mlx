#!/usr/bin/env bash
# Head build of 4e3b83d2a: prepare-mlx, Release C++ test tree (standing battery,
# capability-sim, sdpa_decode_fused, info tool), then the Release wheel with the
# repo's release CMake args (scripts/wheel-cmake-args.sh, ANE worker off as in
# the base wheel). Fresh .work per build; 6 jobs, nice 10.
set -euo pipefail
L=${LANE}
R=${REPO}
VB=${BASE_BUILD}/venv-build/bin/python
cd "$R"
test "$(git rev-parse --short=9 HEAD)" = "$1"
test -z "$(git status --porcelain --untracked-files=no)"
echo "loadavg $(cat /proc/loadavg)"; free -m | sed -n 2,3p
scripts/prepare-mlx.sh
INC="-DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
rm -rf .work/mlx/build-head
cmake -S .work/mlx -B .work/mlx/build-head -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DMLX_OMARCHY_GPU_PROFILING=OFF $INC
T=""
for s in runtime primitive matmul_family fast_ops kv_ops indexing_ops reduce_ops shape_ops linalg_ops copy_offset distributed compiled_tape fft_ops fft_general eig_ops take_fill conv complex_ops select_layout fast_regression scatter_determinism eq_math fused_chain error_contract ane_bundle capability_sim sdpa_decode_fused; do
  T="$T omarchy_${s}_tests"
done
nice -n 10 cmake --build .work/mlx/build-head -j6 --target $T mlx-omarchy-info
echo "TESTS_BUILT $(date -u +%FT%TZ)"
source scripts/wheel-cmake-args.sh
export CMAKE_ARGS="$CMAKE_ARGS $INC" CMAKE_BUILD_PARALLEL_LEVEL=6 CMAKE_GENERATOR=Ninja
unset DEV_RELEASE PYPI_RELEASE MLX_OMARCHY_LOCAL_VERSION
echo "CMAKE_ARGS=$CMAKE_ARGS"
rm -rf "$L/dist-head"; mkdir -p "$L/dist-head"
nice -n 10 "$VB" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$L/dist-head" "$R/.work/mlx"
ls "$L/dist-head"
echo "WHEEL_BUILT $(date -u +%FT%TZ)"
