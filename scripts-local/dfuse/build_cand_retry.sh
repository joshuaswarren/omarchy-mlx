#!/usr/bin/env bash
# DispatchFuse: retry the cand-wheel build across reboots, then patch + verify.
set -uo pipefail
for attempt in 1 2 3 4 5 6; do
  echo "=== cand build attempt $attempt $(date -u +%FT%TZ)"
  if ssh -o ConnectTimeout=8 jw16mbp1-linux '
    while [ $(cut -d. -f1 /proc/uptime) -lt 380 ]; do sleep 15; done
    test ! -e /var/tmp/JW16_MAINTENANCE || { echo MAINTENANCE-FLAG; exit 1; }
    bash /var/tmp/dfuse/build_cand_launch.sh
    for i in $(seq 1 240); do pgrep -f "[d]fuse/build_cand_launch" >/dev/null || break; sleep 5; done
    pgrep -f "[b]uild-wheel.sh" >/dev/null && { echo STILL-BUILDING; exit 3; }
    /var/tmp/dfuse-cand/bin/python -c "import mlx.core as mx; print(\"cand:\", mx.__version__)"
    python3 /var/tmp/dfuse/patch-mlx-lm-qwen3-rope-norm.py /var/tmp/dfuse-cand
    python3 /var/tmp/dfuse/patch-mlx-lm-gdn-raw-repeat.py /var/tmp/dfuse-cand
  '; then
    echo "CAND BUILD COMPLETE"
    exit 0
  fi
  echo "attempt $attempt failed; waiting for host"
  sleep 90
done
echo "CAND BUILD EXHAUSTED"
exit 1
