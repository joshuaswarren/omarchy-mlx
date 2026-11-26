#!/usr/bin/env bash
# Relocate the HwProbe private venv from /tmp (tmpfs, wiped by the
# 00:45Z-03:00Z agx_stats reboots) to /var/tmp (disk-backed). Run ON the M2.
set -euo pipefail
if [ -d /var/tmp/hwprobe-venv ]; then
  echo "target already exists; refusing"
  exit 1
fi
cp -a /tmp/hwprobe-venv /var/tmp/hwprobe-venv
python3 - <<'PY'
import shutil
shutil.rmtree('/tmp/hwprobe-venv')
print('tmp copy removed')
PY
/var/tmp/hwprobe-venv/bin/python - <<'PY'
from importlib.metadata import version
import mlx.core as mx
print('relocated venv ok; version:', version('mlx-omarchy'))
PY
df -h /var/tmp | tail -1