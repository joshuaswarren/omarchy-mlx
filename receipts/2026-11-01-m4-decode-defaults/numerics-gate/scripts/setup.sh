#!/bin/bash
# Recreate ~/gpu/upr after the restore: bare repo, build venv, then build the
# four release wheels and two profiling wheels in sequence, then their venvs.
set -uo pipefail
U=$HOME/gpu/upr
mkdir -p $U/logs
exec >> $U/logs/setup.log 2>&1
echo "setup start $(date -u +%FT%TZ)"
if [ ! -d $U/repo.git ]; then
  git clone -q --bare https://github.com/techmore/omarchy-mlx.git $U/repo.git || exit 2
fi
git -C $U/repo.git fetch -q https://github.com/joshuaswarren/omarchy-mlx.git main:refs/heads/upstream-main || exit 2
for c in a69d9e6 34d0153 91d9f8a 89b54a0; do git -C $U/repo.git cat-file -e $c^{commit} || { echo "missing $c"; exit 2; }; done
if [ ! -x $U/venv-build/bin/python ]; then
  python3 -m venv --system-site-packages $U/venv-build
  $U/venv-build/bin/pip install -q setuptools==84.2.1 cmake==4.4.4 typing_extensions==4.16.0 || exit 2
fi
printf 'main a69d9e6\np1 34d0153\np2 91d9f8a\np6 89b54a0\ndmain a69d9e6\ndp1 34d0153\n' > $U/commits.txt
for spec in main:a69d9e6:0 p1:34d0153:0 p2:91d9f8a:0 p6:89b54a0:0 dmain:a69d9e6:1 dp1:34d0153:1; do
  IFS=: read v ref diag <<< "$spec"
  if ls $U/dist/$v/mlx_omarchy-*.whl >/dev/null 2>&1; then echo "have $v"; else
    bash $U/probes/build.sh $v $ref $diag; grep -q BUILD_OK $U/logs/build-$v.log || { echo "build $v failed"; exit 3; }
  fi
  [ -x $U/venv-$v/bin/python ] || bash $U/probes/mkvenv.sh $v || { echo "venv $v failed"; exit 4; }
  echo "ready $v $(date -u +%FT%TZ)"
done
echo "SETUP_DONE $(date -u +%FT%TZ)"
