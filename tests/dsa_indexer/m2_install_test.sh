#!/usr/bin/env bash
# FamGlmDsa: install the v2 native wheel into a PRIVATE venv (shebang-safe
# per the 2026-10-05 hazard broadcast) and run the native parity harness.
set -euo pipefail
COPY=/var/tmp/FamGlmDsa-wheel
PRIV=/var/tmp/FamGlmDsa-venv
WHEEL=$(ls "$COPY"/dist/mlx_omarchy-*.whl | head -1)

if [ ! -d "$PRIV" ]; then
  cp -a /var/tmp/shared-omarchy-venv "$PRIV"
fi
# Rewrite bin/ shebangs so pip cannot reach the shared venv.
"$PRIV"/bin/python -m venv --upgrade "$PRIV"
grep -l '/var/tmp/shared-omarchy-venv' "$PRIV"/bin/* 2>/dev/null | while read -r f; do
  sed -i '1s|.*|#!'"$PRIV"'/bin/python3|' "$f"
done

"$PRIV"/bin/python -m pip install --no-deps --force-reinstall "$WHEEL"
"$PRIV"/bin/python - <<'PYEOF'
import mlx.core as mx, sys
print("mx file:", mx.__file__)
print("mx version:", mx.__version__)
assert "FamGlmDsa-venv" in mx.__file__, mx.__file__
print("private-venv install VERIFIED")
PYEOF
echo "wheel: $WHEEL"
sha256sum "$WHEEL"
sha256sum "$PRIV"/lib/python3.14/site-packages/mlx/lib/libmlx.so 2>/dev/null || true
echo "INSTALL DONE (run native parity under gpu-turn separately)"
