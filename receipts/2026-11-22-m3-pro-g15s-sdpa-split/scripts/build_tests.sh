#!/usr/bin/env bash
# build_tests.sh SIDE: Release C++ test tree in scratch/wt-SIDE (after build_wheel.sh prepared .work/mlx there; fresh build dir):
# the standing battery (AGENTS.md § Test rules), capability simulation, sdpa_split and the info tool. Run under /tmp/m1-gpu.lock.
set -euo pipefail
L=${LANE}; W=$L/scratch/wt-$1
cd $W; echo "commit $(git rev-parse HEAD) loadavg $(cat /proc/loadavg)"
INC="-DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
rm -rf .work/mlx/build-tests
cmake -S .work/mlx -B .work/mlx/build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DMLX_OMARCHY_GPU_PROFILING=OFF $INC
T=""
for s in runtime primitive matmul_family fast_ops kv_ops indexing_ops reduce_ops shape_ops linalg_ops copy_offset distributed compiled_tape fft_ops fft_general eig_ops take_fill conv complex_ops select_layout fast_regression scatter_determinism eq_math fused_chain error_contract ane_bundle capability_sim sdpa_split; do
  T="$T omarchy_${s}_tests"
done
nice -n 10 cmake --build .work/mlx/build-tests -j6 --target $T mlx-omarchy-info
echo "TESTS_BUILT $(date -u +%FT%TZ)"
