#!/usr/bin/env bash
# DrainFix: build the candidate venv (clone of serving venv, ONE patched file) — CPU only.
# DecodeBw lesson applied: the clone's pip is NEVER used; nothing installs into serving.
set -euo pipefail
SRC=/var/tmp/v072-venv-fused
DST=/var/tmp/drainfix/venv-cand
TOOLS=/var/tmp/drainfix/tools
free_g=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc '0-9')
test "$free_g" -ge 6 || { echo "low disk: ${free_g}G free"; exit 1; }
if [ -e "$DST" ]; then echo "candidate venv exists, patching in place"; else
  cp -a "$SRC" "$DST"
fi
GEN="$DST"/lib/python3.14/site-packages/mlx_lm/generate.py
test -e "$GEN"
"$TOOLS/patch_generate_step.py" "$GEN"
"$DST/bin/python" - <<'EOF'
import importlib
import inspect
import mlx.core as mx
g = importlib.import_module("mlx_lm.generate")
src = inspect.getsource(g.generate_step)
assert "MLX_OMARCHY_DECODE_LOOKAHEAD" in src, "patch not active in candidate"
assert hasattr(mx, "async_eval")
print("candidate generate.py:", g.__file__)
print("candidate OK, async_eval present")
EOF
echo "== provenance: serving =="
nice -n 19 /var/tmp/v072-venv-fused/bin/python /var/tmp/drainfix/tools/mlx_provenance.py 2>&1 | tail -4
echo "== provenance: candidate =="
nice -n 19 "$DST/bin/python" /var/tmp/drainfix/tools/mlx_provenance.py 2>&1 | tail -4
sha256sum "$SRC/lib/python3.14/site-packages/mlx_lm/generate.py" "$GEN"
