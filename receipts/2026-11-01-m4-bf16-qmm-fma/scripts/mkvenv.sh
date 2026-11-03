#!/bin/bash
# usage: mkvenv.sh <variant>: fresh runtime venv with that variant's wheel,
# mlx-lm 0.31.3 (no deps) and the pinned runtime requirements.
set -e
U=$HOME/gpu/upr; V=$1
rm -rf $U/venv-$V
python3 -m venv $U/venv-$V
$U/venv-$V/bin/pip install -q -r $U/probes/runtime-reqs.txt
$U/venv-$V/bin/pip install -q --no-deps $U/dist/$V/mlx_omarchy-*.whl mlx-lm==0.31.3
$U/venv-$V/bin/pip list 2>/dev/null | grep -iE "^mlx"
