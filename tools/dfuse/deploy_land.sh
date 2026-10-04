#!/usr/bin/env bash
# DispatchFuse deploy: final wheel into the serving venv + patch set, then verify.
set -euo pipefail
WHEEL=/var/tmp/dfuse-build/dist/mlx_omarchy-0.32.4.dev202610040439+dfuse.15547ef8f-cp314-cp314-linux_aarch64.whl
test -e "$WHEEL" || { echo "missing wheel"; exit 1; }
sha256sum "$WHEEL"
bash /var/tmp/appbar/deploy_wheel.sh "$WHEEL" /var/tmp/v072-venv-fused
PY=/var/tmp/v072-venv-fused/bin/python
python3 /var/tmp/dfuse/patch-mlx-lm-qwen3-rope-norm.py /var/tmp/v072-venv-fused
python3 /var/tmp/dfuse/patch-mlx-lm-gdn-raw-repeat.py /var/tmp/v072-venv-fused
"$PY" -c "import mlx.core as mx; print('serving now:', mx.__version__)"
"$PY" - <<'PYEOF'
import importlib, inspect
q3 = importlib.import_module("mlx_lm.models.qwen3")
src = inspect.getsource(q3)
assert 'MLX_OMARCHY_ROPE_NORM_FUSE", "1"' in src, "default-ON fold not in serving qwen3.py"
print("serving qwen3 fold default ON: verified")
PYEOF
