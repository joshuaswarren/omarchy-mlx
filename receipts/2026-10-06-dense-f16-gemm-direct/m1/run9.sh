#!/bin/bash
# MatmulGap H2 on jwm1: build agent/matmul-gap (wheel + test suites) and run
# the standing M1 battery, all under the shared GPU lock.
R=/var/tmp/w7q-h9
mkdir -p $R/out
exec >> $R/run.log 2>&1
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id)"
exec 9>/tmp/m1-gpu.lock
flock -w 1800 9 || { echo LOCK-TIMEOUT; exit 2; }
echo "lock $(date -u +%FT%TZ)"
if [ ! -e $R/built-ok ]; then
  cd $R || exit 1
  [ -d src ] && rm -r src
  git init -q src
  cd src || exit 1
  git fetch -q /var/tmp/h335/src "+refs/heads/*:refs/remotes/base/*"
  git fetch -q $R/w7q.bundle "refs/heads/agent/matmul-gap:refs/heads/w" 2>&1 | tail -1
  git checkout -q w 2>&1 | tail -1
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
    -DMLX_USE_CCACHE=ON -DBUILD_SHARED_LIBS=OFF > $R/cfg-tests.log 2>&1 || { echo CFG-FAIL; tail -20 $R/cfg-tests.log; exit 3; }
  touch $R/built-ok
fi
cd $R/src || exit 1
SUITES="omarchy_runtime_tests omarchy_primitive_tests omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_kv_ops_tests omarchy_indexing_ops_tests omarchy_reduce_ops_tests omarchy_shape_ops_tests omarchy_linalg_ops_tests omarchy_copy_offset_tests omarchy_distributed_tests omarchy_compiled_tape_tests omarchy_fft_ops_tests omarchy_fft_general_tests omarchy_eig_ops_tests omarchy_take_fill_tests omarchy_conv_tests omarchy_complex_ops_tests omarchy_select_layout_tests omarchy_fast_regression_tests omarchy_scatter_determinism_tests omarchy_eq_math_tests omarchy_fused_chain_tests omarchy_error_contract_tests omarchy_ane_bundle_tests omarchy_capability_sim_tests"
nice -n 10 cmake --build build-tests -j 8 --target $SUITES > $R/build-tests.log 2>&1 || { echo TESTBUILD-FAIL; grep -m5 -B2 -A8 "error" $R/build-tests.log; exit 3; }
echo "tests built $(date -u +%FT%TZ)"
for t in $SUITES; do
  bin=$(find build-tests -name "$t" -type f -perm -u+x | head -1)
  if [ -z "$bin" ]; then echo "$t MISSING"; continue; fi
  timeout -s KILL 900 "$bin" > $R/out/$t.log 2>&1
  rc=$?
  echo "$t rc=$rc $(grep -E 'test cases:|All tests passed' $R/out/$t.log | tail -1)"
done
echo "battery end $(date -u +%FT%TZ)"
PY=$R/venv/bin/python
echo "wheel $($PY -c 'import mlx.core as mx; print(mx.__version__)')"
$PY $R/mlxcheck.py > $R/out/mlxcheck-direct.json 2>&1
MLX_OMARCHY_NO_COOPMAT=1 $PY $R/mlxcheck.py > $R/out/mlxcheck-noco.json 2>&1
for i in 1 2 3; do
  sleep 8
  echo "mm direct #$i $($PY $R/mm_trace.py 20 2>&1 | tail -1)"
  sleep 8
  echo "mm noco   #$i $(MLX_OMARCHY_NO_COOPMAT=1 $PY $R/mm_trace.py 20 2>&1 | tail -1)"
done
echo "end $(date -u +%FT%TZ)"
