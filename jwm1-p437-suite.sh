#!/usr/bin/env bash
# Whole omarchy test suite (minus ane_runtime) at w7q/p437-rebase head.
set -uo pipefail
S=$HOME/scratch
rm -f $S/w7q-p437suite.flag
export MLX_OMARCHY_WORK_DIR=$S/w7q-p437-suite-work
cd $S/w7q-p437-src || exit 1
git fetch -q origin w7q/p437-rebase
git checkout -q --detach origin/w7q/p437-rebase
bash scripts/prepare-mlx.sh > $S/w7q-p437suite-prepare.log 2>&1 || { echo "prepare failed" > $S/w7q-p437suite-status.txt; echo done > $S/w7q-p437suite.flag; exit 1; }
B=$S/w7q-p437-suite-b
cmake -S "$MLX_OMARCHY_WORK_DIR/mlx" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DOMARCHY_MLX_SYSTEM_PREFIX=/usr/lib/omarchy-mlx -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas \
  > $S/w7q-p437suite-cfg.log 2>&1 || { echo "configure failed" > $S/w7q-p437suite-status.txt; echo done > $S/w7q-p437suite.flag; exit 1; }
nice -n 10 cmake --build "$B" -j4 -k > $S/w7q-p437suite-build.log 2>&1
echo "build rc=$?" > $S/w7q-p437suite-status.txt
cd "$B" || exit 1
~/bin/gpu-turn -m 20 -- bash -c "ctest -E ane_runtime --output-on-failure > $S/w7q-p437suite-ctest.log 2>&1"; rc=$?
echo "ctest rc=$rc" >> $S/w7q-p437suite-status.txt
tail -n 5 $S/w7q-p437suite-ctest.log >> $S/w7q-p437suite-status.txt
echo done > $S/w7q-p437suite.flag
