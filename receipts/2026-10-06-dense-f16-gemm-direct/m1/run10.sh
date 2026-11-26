#!/bin/bash
# MatmulGap H2b on jwm1: the new focused test at the tip, the capability
# simulation profile matrix, and a dispatch trace of the MLX-level check.
R=/var/tmp/w7q-h9
exec >> $R/run10.log 2>&1
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id)"
cd $R/src || exit 1
git fetch -q $R/w7q.bundle "refs/heads/agent/matmul-gap:refs/heads/w2" && git checkout -q w2 || { echo FETCH-FAIL; exit 3; }
echo "staged $(git log --oneline -1 | cut -c1-100)"
scripts/prepare-mlx.sh > $R/prep10.log 2>&1 || { echo PREPARE-FAIL; exit 3; }
nice -n 10 cmake --build build-tests -j 8 --target omarchy_matmul_family_tests omarchy_capability_sim_tests > $R/build10.log 2>&1 || { echo BUILD-FAIL; tail -20 $R/build10.log; exit 3; }
exec 9>/tmp/m1-gpu.lock
flock -w 1800 9 || { echo LOCK-TIMEOUT; exit 2; }
echo "lock $(date -u +%FT%TZ)"
T=build-tests/tests/omarchy
timeout -s KILL 900 $T/omarchy_matmul_family_tests > $R/out/matmul_family_tip.log 2>&1
echo "matmul_family tip rc=$? $(grep -E 'test cases:' $R/out/matmul_family_tip.log | tail -1)"
for p in $(sed -n 's/^profiles: //p' $R/out/omarchy_capability_sim_tests.log); do
  MLX_OMARCHY_CAPS_SIM=$p timeout -s KILL 900 $T/omarchy_capability_sim_tests "$p" > $R/out/capsim-$p.log 2>&1
  echo "capsim $p rc=$? $(grep -E 'test cases:' $R/out/capsim-$p.log | tail -1)"
done
timeout -s KILL 900 $T/omarchy_capability_sim_tests > $R/out/capsim-noprofile.log 2>&1
echo "capsim (real caps) rc=$? $(grep -E 'test cases:|usage' $R/out/capsim-noprofile.log | tail -1)"
MLX_OMARCHY_TRACE_DISPATCH=1 $R/venv/bin/python $R/mlxcheck.py > $R/out/mlxcheck-trace.log 2>&1
echo "trace lines $(wc -l < $R/out/mlxcheck-trace.log)"
echo "end $(date -u +%FT%TZ)"
