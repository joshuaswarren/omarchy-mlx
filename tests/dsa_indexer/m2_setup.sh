#!/usr/bin/env bash
# M2-side setup + verify for the FamGlmDsa correctness ticket.
set -u
cd ~/dsa-fam-correctness/tests/dsa_indexer
# Remove the core dump and pycache that rode along with the scp.
for f in core.[0-9]*; do
  [ -e "$f" ] && rm -f -- "$f"
done
if [ -d __pycache__ ]; then rm -r __pycache__; fi
~/dsa-fam-correctness/venv/bin/python - <<'PYEOF'
import mlx.core as mx
print("mlx ok:", mx.metal.is_available(), mx.default_device())
PYEOF
