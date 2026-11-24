#!/usr/bin/env bash
# build_split_test.sh COMMIT: Release build of omarchy_sdpa_split_tests at COMMIT in scratch/wt-tests (fresh .work). Run under /tmp/m1-gpu.lock.
set -euo pipefail
L=${LANE}
R=${REPO}
W=$L/scratch/wt-tests
[ -d $W ] || git -C $R worktree add --detach $W $1
git -C $W checkout -q --detach $1; echo "commit $(git -C $W rev-parse HEAD) loadavg $(cat /proc/loadavg)"
cd $W; rm -rf .work; scripts/prepare-mlx.sh > /dev/null
cmake -S .work/mlx -B .work/mlx/build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON -DMLX_OMARCHY_GPU_PROFILING=OFF \
  -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas > /dev/null
nice -n 10 cmake --build .work/mlx/build-tests -j6 --target omarchy_sdpa_split_tests
echo "SPLIT_TEST_BUILT $(date -u +%FT%TZ)"
