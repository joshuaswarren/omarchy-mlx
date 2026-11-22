#!/usr/bin/env bash
# venv-head: same packages as the base venv (g15s-wave-sched/venv-base), with the head wheel, then install.sh --patch-venv.
set -euo pipefail
L=<lane>
R=<repo>
PB=<base-build>/venv-base/bin/python
WH=$(ls $L/dist-head/*.whl)
rm -rf $L/venv-head
python3 -m venv $L/venv-head
$PB -m pip freeze | grep -v -E '^(mlx|mlx-lm|mlx-omarchy)[ =@]' > $L/evidence/venv-base-freeze-nomlx.txt
$L/venv-head/bin/pip install -q -r $L/evidence/venv-base-freeze-nomlx.txt
$L/venv-head/bin/pip install -q --no-deps "$WH" mlx-lm==0.31.3
bash $R/install.sh --patch-venv $L/venv-head
$L/venv-head/bin/pip freeze > $L/evidence/venv-head-freeze.txt
$PB -m pip freeze > $L/evidence/venv-base-freeze.txt
diff <(grep -v -E '^mlx' $L/evidence/venv-base-freeze.txt) <(grep -v -E '^mlx' $L/evidence/venv-head-freeze.txt) && echo "FREEZE_EQUAL_EXCEPT_MLX"
grep -E '^mlx' $L/evidence/venv-base-freeze.txt $L/evidence/venv-head-freeze.txt
