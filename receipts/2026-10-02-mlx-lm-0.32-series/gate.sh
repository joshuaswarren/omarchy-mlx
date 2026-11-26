set -u
hostname; uname -r
D="$HOME/.local/share/coreglass"
PY="$D/venv/bin/python"
M="$D/models/qwen3_5-4bit"
S="$D/omlx-omarchy-series/lib/python3.14/site-packages"
LOCK="$1"
sha256sum "$M/model.safetensors" | cut -c1-16
(cd "$D/omarchy-mlx-src" && sha256sum patches/mlx-lm-0.32/*.patch scripts/patch-mlx-lm-qknorm.py | sha256sum | cut -c1-16)
flock -w 120 "$LOCK" sh -c "
  $PY /tmp/cg-series-gate.py $M upstream 2>&1 | grep '^{'
  PYTHONPATH=$S $PY /tmp/cg-series-gate.py $M series 2>&1 | grep '^{'
  MLX_OMARCHY_GDN_QKNORM_FUSE=0 PYTHONPATH=$S $PY /tmp/cg-series-gate.py $M series-qknorm-off 2>&1 | grep '^{'
  $PY /tmp/cg-series-gate.py $M upstream-repeat 2>&1 | grep '^{'
  PYTHONPATH=$S $PY /tmp/cg-series-gate.py $M series-repeat 2>&1 | grep '^{'
"
