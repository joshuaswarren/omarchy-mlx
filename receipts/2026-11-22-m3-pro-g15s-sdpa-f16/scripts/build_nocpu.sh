#!/usr/bin/env bash
# docs/parity-contract.md 2.1: CPU-disabled Release wheel of code commit $1 (run under /tmp/m1-gpu.lock).
set -euo pipefail
L=${LANE}
R=${REPO}
VB=${BASE_BUILD}/venv-build/bin/python
cd "$R"
test "$(git rev-parse --short=9 HEAD)" = "$1"
git diff --quiet "$1" -- overlay patches mlx.lock scripts/prepare-mlx.sh
echo "loadavg $(cat /proc/loadavg)"
export MLX_OMARCHY_WORK_DIR="$R/.work/parity-nocpu-$1"
export MLX_OMARCHY_LOCAL_VERSION="nocpu.$1"
export DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=6 CMAKE_GENERATOR=Ninja
rm -rf "$MLX_OMARCHY_WORK_DIR"
scripts/prepare-mlx.sh
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_GPU_PROFILING=OFF -DCMAKE_INSTALL_LIBDIR=lib"
export CMAKE_ARGS="$CMAKE_ARGS -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
echo "CMAKE_ARGS=$CMAKE_ARGS"
rm -rf "$L/dist-nocpu"; mkdir -p "$L/dist-nocpu"
nice -n 10 "$VB" -m pip wheel -v --no-build-isolation --no-deps --wheel-dir "$L/dist-nocpu" "$MLX_OMARCHY_WORK_DIR/mlx"
B=$(ls -d "$MLX_OMARCHY_WORK_DIR"/mlx/build/temp* 2>/dev/null | head -1)
echo "no_cpu objects:"; find "$MLX_OMARCHY_WORK_DIR/mlx/build" -path '*backend/no_cpu*' -name '*.o' | sed 's|.*/mlx/backend/|backend/|' | sort
echo "cpu objects:"; find "$MLX_OMARCHY_WORK_DIR/mlx/build" -path '*backend/cpu/*' -name '*.o' | sed 's|.*/mlx/backend/|backend/|' | sort
# venv-nocpu: venv-head's packages with the nocpu wheel
rm -rf "$L/venv-nocpu"; python3 -m venv "$L/venv-nocpu"
"$L/venv-nocpu/bin/pip" install -q -r "$L/evidence/venv-base-freeze-nomlx.txt"
"$L/venv-nocpu/bin/pip" install -q --no-deps "$(ls $L/dist-nocpu/*.whl)" mlx-lm==0.31.3
bash "$R/install.sh" --patch-venv "$L/venv-nocpu"
ls "$L/dist-nocpu"; echo NOCPU_BUILT
