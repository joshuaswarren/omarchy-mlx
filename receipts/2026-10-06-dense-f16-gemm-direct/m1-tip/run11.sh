#!/bin/bash
# MatmulGap H2c on jwm1: rebuild the rebased branch tip (wheel + suites),
# then, under the GPU lock, the touched suites, the MLX-level route check,
# the 20 s matmul loop per route, and the standalone edge-shape bit check.
R=/var/tmp/w7q-h11
mkdir -p $R/out
exec >> $R/run.log 2>&1
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id)"
cd $R || exit 1
[ -d src ] && rm -r src
git init -q src
cd src || exit 1
git fetch -q /var/tmp/h335/src "+refs/heads/*:refs/remotes/base/*"
git fetch -q $R/w7q.bundle "refs/heads/agent/matmul-gap:refs/heads/w" && git checkout -q w || { echo FETCH-FAIL; exit 3; }
echo "staged $(git log --oneline -1 | cut -c1-100)"
scripts/prepare-mlx.sh > $R/prep.log 2>&1 || { echo PREPARE-FAIL; exit 3; }
export MLX_OMARCHY_WHOLE_BUNDLE_DIR=/var/tmp/encoder-whole-jwm1/bundle
export CMAKE_BUILD_PARALLEL_LEVEL=8
nice -n 10 scripts/build-wheel.sh > $R/build.log 2>&1 || { echo BUILD-FAIL; tail -20 $R/build.log; exit 3; }
sha256sum dist/*.whl
python3 -m venv $R/venv
$R/venv/bin/pip -q install numpy || { echo PIP-FAIL; exit 3; }
$R/venv/bin/pip -q install --no-deps --force-reinstall dist/*.whl || { echo WHL-FAIL; exit 3; }
cmake -S .work/mlx -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_USE_CCACHE=ON -DBUILD_SHARED_LIBS=OFF > $R/cfg-tests.log 2>&1 || { echo CFG-FAIL; exit 3; }
SUITES="omarchy_runtime_tests omarchy_primitive_tests omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_indexing_ops_tests"
nice -n 10 cmake --build build-tests -j 8 --target $SUITES omarchy_capability_sim_tests > $R/build-tests.log 2>&1 || { echo TESTBUILD-FAIL; exit 3; }
echo "built $(date -u +%FT%TZ)"
exec 9>/tmp/m1-gpu.lock
flock -w 1800 9 || { echo LOCK-TIMEOUT; exit 2; }
echo "lock $(date -u +%FT%TZ)"
T=build-tests/tests/omarchy
for t in $SUITES; do
  timeout -s KILL 900 $T/$t > $R/out/$t.log 2>&1
  echo "$t rc=$? $(grep -E 'test cases:' $R/out/$t.log | tail -1)"
done
for p in m1-honeykrisp-fork no-cooperative-matrix; do
  MLX_OMARCHY_CAPS_SIM=$p timeout -s KILL 900 $T/omarchy_capability_sim_tests "$p" > $R/out/capsim-$p.log 2>&1
  echo "capsim $p rc=$? $(grep -E 'test cases:' $R/out/capsim-$p.log | tail -1)"
done
PY=$R/venv/bin/python
echo "wheel $($PY -c 'import mlx.core as mx; print(mx.__version__)')"
$PY /var/tmp/w7q-h9/mlxcheck.py > $R/out/mlxcheck-direct.json 2>&1
MLX_OMARCHY_NO_COOPMAT=1 $PY /var/tmp/w7q-h9/mlxcheck.py > $R/out/mlxcheck-noco.json 2>&1
sleep 8
echo "mm direct $($PY /var/tmp/w7q-h9/mm_trace.py 20 2>&1 | tail -1 | cut -c1-40)"
sleep 8
echo "mm noco   $(MLX_OMARCHY_NO_COOPMAT=1 $PY /var/tmp/w7q-h9/mm_trace.py 20 2>&1 | tail -1 | cut -c1-40)"
cd /var/tmp/w7q-gemm || exit 1
export VK_ICD_FILENAMES=/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json
D=$R/src/overlay/mlx/backend/omarchy/shaders/matmul_coopmat_direct.comp
RB=$R/src/overlay/mlx/backend/omarchy/shaders/matmul_rb.comp
for mnk in 100,72,64 32,32,8 33,34,16 130,200,40 64,64,4096 1000,520,88 97,4096,512; do
  for f in 0 1 4 5; do
    at=$(( (f >> 2) & 1 )); bt=$(( f & 1 )); m=${mnk%%,*}
    if [ $at = 1 ] && [ $((m % 2)) = 1 ]; then continue; fi
    r=$(timeout 60 ./gemm-bench --dtype f16 --mnk $mnk --samples 1000000 --reps 1 --rounds 1 \
      --side rb=$RB:64:64:$f --side d=$D:64:64:$f:-DA_T=$at,-DB_T=$bt 2>&1 | grep '"side":"d"' | grep accuracy)
    echo "edge $mnk flags=$f $r" >> $R/out/edge.log
  done
done
echo "edge cases $(wc -l < $R/out/edge.log), nonzero bits $(grep -vc '"bits_vs_rb":0}' $R/out/edge.log)"
echo "end $(date -u +%FT%TZ)"
