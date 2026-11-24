#!/usr/bin/env bash
# build_rebased_tests.sh COMMIT: Release C++ test tree of the standing battery, capability simulation and sdpa_split at COMMIT in
# scratch/wt-rebased (fresh .work, prepare-mlx.sh, fresh build dir), plus the info tool. Run under /tmp/m1-gpu.lock.
set -euo pipefail
L=${LANE}
R=${REPO}
W=$L/scratch/wt-rebased
[ -d $W ] || git -C $R worktree add --detach $W $1
git -C $W checkout -q --detach $1; echo "commit $(git -C $W rev-parse HEAD) loadavg $(cat /proc/loadavg)"
cd $W; rm -rf .work; scripts/prepare-mlx.sh > /dev/null
cmake -S .work/mlx -B .work/mlx/build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DMLX_OMARCHY_GPU_PROFILING=OFF \
  -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas > /dev/null
T=""
for s in runtime primitive matmul_family fast_ops kv_ops indexing_ops reduce_ops shape_ops linalg_ops copy_offset distributed compiled_tape fft_ops fft_general eig_ops take_fill conv complex_ops select_layout fast_regression scatter_determinism eq_math fused_chain error_contract ane_bundle capability_sim sdpa_split; do
  T="$T omarchy_${s}_tests"
done
nice -n 10 cmake --build .work/mlx/build-tests -j6 --target $T mlx-omarchy-info
echo "TESTS_BUILT $(date -u +%FT%TZ)"
