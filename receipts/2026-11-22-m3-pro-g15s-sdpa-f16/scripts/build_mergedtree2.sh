#!/usr/bin/env bash
# Release wheels of newest green main ($1) and of head ($2) merged locally onto it (no push), each in its own scratch worktree.
set -euo pipefail
L=${LANE}
R=${REPO}
VB=${BASE_BUILD}/venv-build/bin/python
MAIN=$1; HEAD_SHA=$2; SFX=$3
S=$L/scratch; mkdir -p $S
git -C $R worktree add --detach $S/wt-main$SFX $MAIN
git -C $R worktree add --detach $S/wt-merged$SFX $HEAD_SHA
git -C $S/wt-merged$SFX -c user.name="${GIT_NAME:?}" -c user.email="${GIT_EMAIL:?}" -c commit.gpgsign=false merge --no-edit -q $MAIN
echo "merged tree commit $(git -C $S/wt-merged$SFX rev-parse --short=9 HEAD) tree $(git -C $S/wt-merged$SFX rev-parse HEAD^{tree})"
INC="-DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
for side in main merged; do
  W=$S/wt-$side$SFX; cd $W
  echo "== $side $(git rev-parse --short=9 HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
  scripts/prepare-mlx.sh > $L/logs/prepare-$side$SFX.log 2>&1
  source scripts/wheel-cmake-args.sh
  export CMAKE_ARGS="$CMAKE_ARGS $INC" CMAKE_BUILD_PARALLEL_LEVEL=6 CMAKE_GENERATOR=Ninja
  unset DEV_RELEASE PYPI_RELEASE MLX_OMARCHY_LOCAL_VERSION
  rm -rf $L/dist-$side$SFX; mkdir -p $L/dist-$side$SFX
  nice -n 10 "$VB" -m pip wheel --no-build-isolation --no-deps --wheel-dir $L/dist-$side$SFX $W/.work/mlx > $L/logs/wheel-$side$SFX.log 2>&1
  ls $L/dist-$side$SFX
  rm -rf $L/venv-$side$SFX; python3 -m venv $L/venv-$side$SFX
  $L/venv-$side$SFX/bin/pip install -q -r $L/evidence/venv-base-freeze-nomlx.txt
  $L/venv-$side$SFX/bin/pip install -q --no-deps $(ls $L/dist-$side$SFX/*.whl) mlx-lm==0.31.3
  bash $R/install.sh --patch-venv $L/venv-$side$SFX > $L/logs/patch-venv-$side$SFX$SFX.log 2>&1
done
echo MERGEDTREE_BUILT
