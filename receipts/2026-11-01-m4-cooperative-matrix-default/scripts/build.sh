#!/bin/bash
# usage: build.sh <variant> <ref> <diag 0|1> [test targets...]
# Fresh worktree, prepare-mlx.sh, fresh CMake tree, stamped wheel; then the
# named doctest targets in the same build tree (same libmlx).
set -uo pipefail
V=$1; REF=$2; DIAG=$3; shift 3; TESTS="$*"
U=$HOME/gpu/upr; WT=$U/wt-$V
mkdir -p $U/logs $U/dist
exec >> $U/logs/build-$V.log 2>&1
echo "start $(date -u +%FT%TZ) variant=$V ref=$REF diag=$DIAG"
if [ "${TESTS_ONLY:-0}" = 1 ]; then cd $WT; C=$(git rev-parse --short=7 HEAD); export PATH=$U/venv-build/bin:$PATH CPLUS_INCLUDE_PATH=$HOME/gpu/vkinc C_INCLUDE_PATH=$HOME/gpu/vkinc; else
rm -rf $WT; git -C $U/repo.git worktree prune
git -C $U/repo.git worktree add --detach $WT $REF || exit 2
cd $WT
C=$(git rev-parse --short=7 HEAD); echo "commit $C $(git rev-parse HEAD)"
mkdir -p .work; cp $HOME/gpu/omarchy-mlx/.work/mlx-*.tar.gz .work/
./scripts/prepare-mlx.sh || exit 3
export PATH=$U/venv-build/bin:$PATH CPLUS_INCLUDE_PATH=$HOME/gpu/vkinc C_INCLUDE_PATH=$HOME/gpu/vkinc CMAKE_BUILD_PARALLEL_LEVEL=10 DEV_RELEASE=1
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DMLX_OMARCHY_ANE_DEVICE=ON -DVulkan_INCLUDE_DIR=$HOME/gpu/vkinc"
if [ "$DIAG" = 1 ]; then
  export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_GPU_PROFILING=ON" MLX_OMARCHY_LOCAL_VERSION=diag.$C
else
  export MLX_OMARCHY_LOCAL_VERSION=$C
fi
echo "glslc: $(glslc --version | head -1)"
rm -rf .work/mlx/build $U/dist/$V
t0=$(date +%s)
python -m pip wheel --no-build-isolation --no-deps --wheel-dir $U/dist/$V .work/mlx || exit 4
echo "wheel_seconds=$(( $(date +%s) - t0 ))"; ls -la $U/dist/$V
fi
if [ -n "$TESTS" ]; then
  # Doctest battery: a separate static-library CMake tree from the same
  # prepared source (the CI configuration; the wheel's shared libmlx hides
  # the internal symbols the tests call). Fresh tree, no cache reuse.
  BT=$WT/.work/build-tests
  rm -rf $BT
  t0=$(date +%s)
  cmake -S .work/mlx -B $BT -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DMLX_OMARCHY_ANE_DEVICE=ON -DVulkan_INCLUDE_DIR=$HOME/gpu/vkinc $( [ "$DIAG" = 1 ] && echo -DMLX_OMARCHY_GPU_PROFILING=ON ) > $U/logs/cmake-tests-$V.log 2>&1 || exit 5
  cmake --build $BT --target $TESTS -j 10 || exit 6
  echo "tests_seconds=$(( $(date +%s) - t0 ))"
  ln -sfn $BT/tests/omarchy $U/tests-$V
fi
echo "BUILD_OK $(date -u +%FT%TZ)"
