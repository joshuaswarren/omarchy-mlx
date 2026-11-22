#!/usr/bin/env bash
# Before-proof tree: commit 8304e5dd3 with only the new test_sdpa_decode_fused.cpp from $1 copied in; builds that one test target.
set -euo pipefail
L=${LANE}
R=${REPO}
W=$L/scratch/wt-before
[ -d $W ] || git -C $R worktree add --detach $W 8304e5dd3
git -C $W checkout -q --detach 8304e5dd3
git -C $R show $1:overlay/tests/omarchy/test_sdpa_decode_fused.cpp > $W/overlay/tests/omarchy/test_sdpa_decode_fused.cpp
git -C $W status --porcelain
cd $W; scripts/prepare-mlx.sh > $L/logs/prepare-before.log 2>&1
cmake -S .work/mlx -B .work/mlx/build-before -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DMLX_OMARCHY_GPU_PROFILING=OFF -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas > /dev/null
nice -n 10 cmake --build .work/mlx/build-before -j6 --target omarchy_sdpa_decode_fused_tests > $L/logs/build-before-tests.log 2>&1
ls -la .work/mlx/build-before/tests/omarchy/omarchy_sdpa_decode_fused_tests; echo BEFORE_BUILT
