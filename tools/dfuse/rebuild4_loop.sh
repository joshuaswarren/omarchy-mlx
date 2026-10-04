#!/usr/bin/env bash
# On-box diag rebuild loop for the perrow fix (survives route flaps).
set -uo pipefail
LOG=/var/tmp/dfuse/rebuild4.log
for attempt in 1 2 3 4 5 6; do
  echo "=== rebuild4 attempt $attempt $(date -u +%FT%TZ)" > "$LOG"
  if cd /var/tmp/dfuse-build && DIAG=1 bash /var/tmp/appbar/build_wheel_venv.sh \
      /var/tmp/dfuse-build dfuse.032a69328 \
      /var/tmp/dfuse-venv /var/tmp/v072-venv-fused >> "$LOG" 2>&1; then
    /var/tmp/dfuse-venv/bin/python -c "import mlx.core as mx; print('diag:', mx.__version__)" >> "$LOG" 2>&1
    echo "REBUILD-COMPLETE" >> "$LOG"
    exit 0
  fi
  echo "attempt $attempt failed" >> "$LOG"
  sleep 60
done
echo "REBUILD-EXHAUSTED" >> "$LOG"
exit 1
