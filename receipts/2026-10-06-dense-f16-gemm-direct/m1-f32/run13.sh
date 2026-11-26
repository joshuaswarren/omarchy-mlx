#!/bin/bash
# MatmulGap H3b on jwm1: rebuild agent/matmul-gap (f32 direct route), then
# under the GPU lock the touched suites, capability simulation, the
# multi-dtype MLX check against the main-tip wheel, a dispatch-trace
# kernel census, and f32 linear-layer m/n sweeps against main.
R=/var/tmp/w7q-h13
mkdir -p $R/out
exec >> $R/run.log 2>&1
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id)"
cd $R || exit 1
[ -d src ] && rm -r src
git init -q src
cd src || exit 1
git fetch -q /var/tmp/h335/src "+refs/heads/*:refs/remotes/base/*"
if ! git fetch -q $R/w7q.bundle "refs/heads/agent/matmul-gap:refs/heads/w"; then echo FETCH-FAIL; exit 3; fi
git checkout -q w
echo "staged $(git log --oneline -1 | cut -c1-100)"
scripts/prepare-mlx.sh > $R/prep.log 2>&1 || { echo PREPARE-FAIL; exit 3; }
export MLX_OMARCHY_WHOLE_BUNDLE_DIR=/var/tmp/encoder-whole-jwm1/bundle
export CMAKE_BUILD_PARALLEL_LEVEL=8
nice -n 10 scripts/build-wheel.sh > $R/build.log 2>&1 || { echo BUILD-FAIL; grep -m5 -B3 -A10 "error" $R/build.log; exit 3; }
sha256sum dist/*.whl
python3 -m venv $R/venv
$R/venv/bin/pip -q install numpy || { echo PIP-FAIL; exit 3; }
$R/venv/bin/pip -q install --no-deps --force-reinstall dist/*.whl || { echo WHL-FAIL; exit 3; }
cmake -S .work/mlx -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_USE_CCACHE=ON -DBUILD_SHARED_LIBS=OFF > $R/cfg-tests.log 2>&1 || { echo CFG-FAIL; exit 3; }
SUITES="omarchy_runtime_tests omarchy_primitive_tests omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_indexing_ops_tests omarchy_linalg_ops_tests omarchy_conv_tests omarchy_fused_chain_tests omarchy_compiled_tape_tests"
nice -n 10 cmake --build build-tests -j 8 --target $SUITES omarchy_capability_sim_tests > $R/build-tests.log 2>&1 || { echo TESTBUILD-FAIL; grep -m5 -B3 -A10 "error" $R/build-tests.log; exit 3; }
echo "built $(date -u +%FT%TZ)"
exec 9>/tmp/m1-gpu.lock
flock -w 1800 9 || { echo LOCK-TIMEOUT; exit 2; }
echo "lock $(date -u +%FT%TZ)"
T=build-tests/tests/omarchy
for t in $SUITES; do
  timeout -s KILL 900 "$T/$t" > "$R/out/$t.log" 2>&1
  echo "$t rc=$? $(grep -E 'test cases:' "$R/out/$t.log" | tail -1)"
done
for p in m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix m1-g13-legacy; do
  MLX_OMARCHY_CAPS_SIM=$p timeout -s KILL 900 $T/omarchy_capability_sim_tests "$p" > "$R/out/capsim-$p.log" 2>&1
  echo "capsim $p rc=$? $(grep -E 'test cases:' "$R/out/capsim-$p.log" | tail -1)"
done
PY=$R/venv/bin/python
echo "wheel $($PY -c 'import mlx.core as mx; print(mx.__version__)')"
C=/var/tmp/w7q-h12/mlxcheck2.py
M=/var/tmp/w7q-h11/venv/bin/python
$PY $C > $R/out/mlxcheck2-direct.json 2>&1
$M $C > $R/out/mlxcheck2-main.json 2>&1
MLX_OMARCHY_TRACE_DISPATCH=1 $PY $C 2> $R/out/trace.log > /dev/null
grep -o "DISPATCH kernel=[0-9]*" $R/out/trace.log | sort | uniq -c > $R/out/trace-kernels.txt
rm -f $R/out/trace.log
for o in nt nn; do
  $M /var/tmp/w7q-h12/ms.py float32 $o > $R/out/ms32-$o-main.json 2>&1
  $PY /var/tmp/w7q-h12/ms.py float32 $o > $R/out/ms32-$o-new.json 2>&1
done
echo "end $(date -u +%FT%TZ)"
