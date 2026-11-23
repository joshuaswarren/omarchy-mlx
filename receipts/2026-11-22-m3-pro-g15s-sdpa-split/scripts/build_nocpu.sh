#!/usr/bin/env bash
# build_nocpu.sh: CPU-disabled Release wheel (docs/parity-contract.md 2.1) of scratch/wt-head, local version nocpu.<short9>, and venv-nocpu. Run under /tmp/m1-gpu.lock.
set -euo pipefail
L=${LANE}; W=$L/scratch/wt-head
VB=${BUILD_VENV}/bin/python
cd $W; C=$(git rev-parse --short=9 HEAD); echo "commit $C loadavg $(cat /proc/loadavg)"
export MLX_OMARCHY_WORK_DIR="$W/.work/parity-nocpu" MLX_OMARCHY_LOCAL_VERSION="nocpu.$C" DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=6 CMAKE_GENERATOR=Ninja
rm -rf "$MLX_OMARCHY_WORK_DIR"
scripts/prepare-mlx.sh
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_OMARCHY_GPU_PROFILING=OFF -DCMAKE_INSTALL_LIBDIR=lib -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
echo "CMAKE_ARGS=$CMAKE_ARGS"
rm -rf $L/dist-nocpu; mkdir -p $L/dist-nocpu
nice -n 10 "$VB" -m pip wheel -v --no-build-isolation --no-deps --wheel-dir $L/dist-nocpu "$MLX_OMARCHY_WORK_DIR/mlx"
echo "no_cpu objects:"; find "$MLX_OMARCHY_WORK_DIR/mlx/build" -path '*backend/no_cpu*' -name '*.o' | sed 's|.*/mlx/backend/|backend/|' | sort
echo "cpu objects:"; find "$MLX_OMARCHY_WORK_DIR/mlx/build" -path '*backend/cpu/*' -name '*.o' | sed 's|.*/mlx/backend/|backend/|' | sort
rm -rf $L/venv-nocpu; python3 -m venv $L/venv-nocpu
$L/venv-nocpu/bin/pip install -q -r $L/evidence/venv-base-freeze-nomlx.txt
$L/venv-nocpu/bin/pip install -q --no-deps $(ls $L/dist-nocpu/*.whl) mlx-lm==0.31.3
bash $W/scripts/apply-mlx-lm-patches.sh $L/venv-nocpu
$L/venv-nocpu/bin/python $W/scripts/mlx_dist_alias.py --self
ls $L/dist-nocpu; echo NOCPU_BUILT
