#!/bin/bash
# Diagnostics (GPU profiling harness ON) build of omarchy-mlx at a tag, for submission-budget calibration only.
set -euo pipefail
TAG=${1:-v0.7.41}
D=$HOME/mlxpr/src; mkdir -p $D; cd $D
[ -d omarchy-mlx/.git ] || git clone -q https://github.com/joshuaswarren/omarchy-mlx.git
cd omarchy-mlx; git fetch -q --tags origin; git checkout -q $TAG
bash scripts/prepare-mlx.sh
python3.14 -m venv .work/venv-diagbuild
. .work/venv-diagbuild/bin/activate
pip -q install 'setuptools>=80' typing_extensions 'cmake>=3.25' nanobind==2.10.2 wheel 2>&1 | tail -2 || true
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DMLX_OMARCHY_GPU_PROFILING=ON"
export DEV_RELEASE=1 MLX_OMARCHY_LOCAL_VERSION=diag.$(git rev-parse --short=8 HEAD)
export CMAKE_BUILD_PARALLEL_LEVEL=8
pip wheel --no-build-isolation --no-deps --wheel-dir $HOME/mlxpr/dist-diag .work/mlx
ls -la $HOME/mlxpr/dist-diag
