#!/bin/bash
# DecodeFuse3: refresh the candidate venv from the serving base (named paths only).
set -euo pipefail
VENV=/var/tmp/fuse3-venv
[ -d "$VENV" ] && rm -r "$VENV"
cp -a /var/tmp/v072-venv-fused "$VENV"
"$VENV/bin/python3" -m pip install --force-reinstall --no-deps -q /var/tmp/fuse3-build/dist/*.whl
"$VENV/bin/python3" -m pip list 2>/dev/null | grep -i mlx-omarchy
bash /var/tmp/fuse3-build/scripts/apply-mlx-lm-patches.sh "$VENV" 2>&1 | tail -14
