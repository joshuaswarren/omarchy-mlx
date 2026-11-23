#!/usr/bin/env bash
# build_wheel.sh SIDE COMMIT: Release wheel of COMMIT in its own scratch worktree (fresh .work, fresh cmake cache),
# with the repo's release CMake args, then venv-SIDE = base packages + this wheel + mlx-lm 0.31.3 + the checkout's scripts/apply-mlx-lm-patches.sh and mlx_dist_alias.py.
# Run under /tmp/m1-gpu.lock. 6 jobs, nice 10.
set -euo pipefail
L=${LANE}
R=${REPO}
VB=${BUILD_VENV}/bin/python
PB=${BASE_VENV}/bin/python
SIDE=$1; C=$2; W=$L/scratch/wt-$SIDE
echo "loadavg $(cat /proc/loadavg)"; free -m | sed -n 2,3p
[ -d $W ] || git -C $R worktree add --detach $W $C
git -C $W checkout -q --detach $C
test -z "$(git -C $W status --porcelain --untracked-files=no)"
echo "side $SIDE commit $(git -C $W rev-parse HEAD)"
cd $W
rm -rf .work
scripts/prepare-mlx.sh
INC="-DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
source scripts/wheel-cmake-args.sh
export CMAKE_ARGS="$CMAKE_ARGS $INC" CMAKE_BUILD_PARALLEL_LEVEL=6 CMAKE_GENERATOR=Ninja
unset DEV_RELEASE PYPI_RELEASE MLX_OMARCHY_LOCAL_VERSION
echo "CMAKE_ARGS=$CMAKE_ARGS"
rm -rf $L/dist-$SIDE; mkdir -p $L/dist-$SIDE
nice -n 10 "$VB" -m pip wheel --no-build-isolation --no-deps --wheel-dir $L/dist-$SIDE $W/.work/mlx
ls $L/dist-$SIDE
[ -f $L/evidence/venv-base-freeze-nomlx.txt ] || $PB -m pip freeze | grep -v -E '^(mlx|mlx-lm|mlx-omarchy)[ =@]' > $L/evidence/venv-base-freeze-nomlx.txt
rm -rf $L/venv-$SIDE; python3 -m venv $L/venv-$SIDE
$L/venv-$SIDE/bin/pip install -q -r $L/evidence/venv-base-freeze-nomlx.txt
$L/venv-$SIDE/bin/pip install -q --no-deps $(ls $L/dist-$SIDE/*.whl) mlx-lm==0.31.3
bash $W/scripts/apply-mlx-lm-patches.sh $L/venv-$SIDE
$L/venv-$SIDE/bin/python $W/scripts/mlx_dist_alias.py --self
$L/venv-$SIDE/bin/pip freeze > $L/evidence/venv-$SIDE-freeze.txt
echo "WHEEL_BUILT $SIDE $(date -u +%FT%TZ)"
