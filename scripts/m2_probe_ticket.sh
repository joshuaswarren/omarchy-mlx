#!/usr/bin/env bash
# HwProbe queue-jump probe ticket (approved by Main 00:38Z 2026-10-05).
# 2-minute gpu-turn slot; probe itself is ~3 s of GPU.
#
# Env overrides (defaults fit the shared M2 lane convention):
#   GPU_TURN  path to the lane's gpu-turn wrapper (default: gpu-turn on PATH)
#   RUN_HOME  HOME the probe process runs with (default: from `logname`)
set -euo pipefail
VENV=${VENV:-/var/tmp/hwprobe-venv}
GPU_TURN=${GPU_TURN:-$(command -v gpu-turn)}
RUN_HOME=${RUN_HOME:-$(logname)}
cd /var/tmp
setsid nohup "$GPU_TURN" -m 2 -- \
  env HOME="$RUN_HOME" \
  "$VENV/bin/python" /var/tmp/hwprobe-wheel/scripts/probe_device_info.py \
  > /var/tmp/hwprobe-probe-post-change.json \
  2> /var/tmp/hwprobe-probe-post-change.err < /dev/null &
echo "launched pid $!"