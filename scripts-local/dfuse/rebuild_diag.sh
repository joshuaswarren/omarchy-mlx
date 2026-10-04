#!/usr/bin/env bash
# Rebuild diag wheel (order-matched walks) + captured-operand bit-identity probe.
set -uo pipefail
ssh -o ConnectTimeout=8 jw16mbp1-linux 'bash /var/tmp/dfuse/build_launch.sh; for i in $(seq 1 240); do pgrep -f "[d]fuse/build_launch" >/dev/null || break; sleep 5; done; /var/tmp/dfuse-venv/bin/python -c "import mlx.core as mx; print(\"diag:\", mx.__version__)"'
