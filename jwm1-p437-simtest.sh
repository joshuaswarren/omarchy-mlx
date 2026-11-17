#!/usr/bin/env bash
# Fail-before / pass-after for the f16 coopmat pin, under the f32-only
# capability sim, then the fixed branch on real hardware.
set -uo pipefail
# no set -e: run steps keep going past a failing command so later steps record status
S=$HOME/scratch
rm -f $S/w7q-p437.flag
export MLX_OMARCHY_WORK_DIR=$S/w7q-p437-work
cd $S/w7q-p437-src || exit 1
git fetch -q origin w7q/p437-rebase w7q/p437-failbefore
git checkout -q --detach FETCH_HEAD
bash scripts/prepare-mlx.sh > $S/w7q-p437-prepare.log 2>&1 || { echo "prepare failed" > $S/w7q-p437-sim-status.txt; echo done > $S/w7q-p437.flag; exit 1; }
B=$S/w7q-p437-b
nice -n 10 cmake --build $B -j4 --target omarchy_matmul_family_tests > $S/w7q-p437-simbuild.log 2>&1
echo "fixed build rc=$?" > $S/w7q-p437-sim-status.txt
cd $B
MLX_OMARCHY_CAPS_SIM=m1-honeykrisp-fork ~/bin/gpu-turn -m 20 -- ctest -R omarchy_matmul_family_tests > $S/w7q-p437-sim-ctest.log 2>&1
echo "fixed sim ctest rc=$?" >> $S/w7q-p437-sim-status.txt
tail -n 4 $S/w7q-p437-sim-ctest.log >> $S/w7q-p437-sim-status.txt
# fail-before control: stage the control branch into its own tree and build
SRCFB=$S/w7q-p437fb-src
if [ ! -d "$SRCFB" ]; then
  git worktree add -q --detach "$SRCFB" origin/w7q/p437-failbefore || exit 1
fi
cd "$SRCFB" || exit 1
git log --oneline -1 > $S/w7q-p437fb-head.txt
export MLX_OMARCHY_WORK_DIR=$S/w7q-p437fb-work
bash scripts/prepare-mlx.sh > $S/w7q-p437fb-prepare.log 2>&1 || { echo "prepare failed" >> $S/w7q-p437-sim-status.txt; echo done > $S/w7q-p437.flag; exit 1; }
BFB=$S/w7q-p437fb-b
cmake -S "$MLX_OMARCHY_WORK_DIR/mlx" -B "$BFB" -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DOMARCHY_MLX_SYSTEM_PREFIX=/usr/lib/omarchy-mlx \
  > $S/w7q-p437fb-configure.log 2>&1 || { echo "fb configure failed" >> $S/w7q-p437-sim-status.txt; echo done > $S/w7q-p437.flag; exit 1; }
nice -n 10 cmake --build "$BFB" -j4 --target omarchy_matmul_family_tests > $S/w7q-p437fb-build.log 2>&1
echo "failbefore build rc=$?" >> $S/w7q-p437-sim-status.txt
cd "$BFB"
MLX_OMARCHY_CAPS_SIM=m1-honeykrisp-fork ~/bin/gpu-turn -m 20 -- ctest -R omarchy_matmul_family_tests > $S/w7q-p437fb-ctest.log 2>&1
echo "failbefore sim ctest rc=$?" >> $S/w7q-p437-sim-status.txt
grep -E "f16 qmm prefill dispatches|FAILED|Status:" $S/w7q-p437fb-ctest.log | tail -n 5 >> $S/w7q-p437-sim-status.txt
# fixed branch on real hardware, no sim
cd $B
~/bin/gpu-turn -m 20 -- ctest -R omarchy_matmul_family_tests > $S/w7q-p437-hw-ctest.log 2>&1
echo "fixed hw ctest rc=$?" >> $S/w7q-p437-sim-status.txt
tail -n 3 $S/w7q-p437-hw-ctest.log >> $S/w7q-p437-sim-status.txt
echo done > $S/w7q-p437.flag
