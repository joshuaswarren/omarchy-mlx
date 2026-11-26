#!/usr/bin/env bash
# Fail-before control only: the pin case built at w7q/p437-failbefore
# (old f16-A selector), run under the f32-only capability sim. The pin
# must FAIL there, proving the fixed branch's pass is load-bearing.
set -uo pipefail
# no set -e: run steps keep going past a failing command so later steps record status
S=$HOME/scratch
rm -f $S/w7q-p437fb.flag
SRCFB=$S/w7q-p437fb-src
cd $S/w7q-rows-build || exit 1
git fetch -q origin w7q/p437-failbefore
if [ ! -d "$SRCFB" ]; then
  git worktree add -q --detach "$SRCFB" origin/w7q/p437-failbefore || { echo "worktree add failed" > $S/w7q-p437fb-status.txt; echo done > $S/w7q-p437fb.flag; exit 1; }
fi
cd "$SRCFB" || exit 1
git checkout -q --detach origin/w7q/p437-failbefore
git log --oneline -1 > $S/w7q-p437fb-head.txt
export MLX_OMARCHY_WORK_DIR=$S/w7q-p437fb-work
bash scripts/prepare-mlx.sh > $S/w7q-p437fb-prepare.log 2>&1 || { echo "prepare failed" > $S/w7q-p437fb-status.txt; echo done > $S/w7q-p437fb.flag; exit 1; }
BFB=$S/w7q-p437fb-b
cmake -S "$MLX_OMARCHY_WORK_DIR/mlx" -B "$BFB" -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DOMARCHY_MLX_SYSTEM_PREFIX=/usr/lib/omarchy-mlx \
  -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas \
  > $S/w7q-p437fb-cfg.log 2>&1 || { echo "configure failed" > $S/w7q-p437fb-status.txt; echo done > $S/w7q-p437fb.flag; exit 1; }
nice -n 10 cmake --build "$BFB" -j4 --target omarchy_matmul_family_tests > $S/w7q-p437fb-build.log 2>&1
echo "failbefore build rc=$?" > $S/w7q-p437fb-status.txt
cd "$BFB" || exit 1
MLX_OMARCHY_CAPS_SIM=m1-honeykrisp-fork ~/bin/gpu-turn -m 20 -- ctest -R omarchy_matmul_family_tests > $S/w7q-p437fb-ctest.log 2>&1
echo "failbefore sim ctest rc=$?" >> $S/w7q-p437fb-status.txt
grep -E "f16 qmm prefill dispatches|FAILED|Status:" $S/w7q-p437fb-ctest.log | tail -n 6 >> $S/w7q-p437fb-status.txt
echo done > $S/w7q-p437fb.flag
