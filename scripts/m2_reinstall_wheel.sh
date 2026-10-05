#!/usr/bin/env bash
# Recreate the HwProbe private venv on the M2 from the shared venv and
# install the HwProbe-built wheel exactly once. Run ON the M2.
set -euo pipefail
VENV=/tmp/hwprobe-venv
WHEEL=/var/tmp/hwprobe-wheel/dist/stamped-od/mlx_omarchy-0.32.4.dev202610050017+b8af62c-cp314-cp314-linux_aarch64.whl

python3 - <<'PY'
import shutil
shutil.rmtree('/tmp/hwprobe-venv', ignore_errors=True)
print('old venv removed')
PY

cp -a /var/tmp/shared-omarchy-venv "$VENV"
"$VENV/bin/pip" install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -3

echo '--- verify ---'
ls -d "$VENV"/lib/python*/site-packages/mlx_omarchy-*
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
import mlx
print('version:', version('mlx-omarchy'))
import mlx.core as mx
print('mx import ok; mlx path:', list(mlx.__path__))
PY