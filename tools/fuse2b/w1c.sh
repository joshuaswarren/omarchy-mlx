#!/usr/bin/env bash
# Fuse2B W1c outer.
set -euo pipefail
BUILD=/var/tmp/fuse2b-build
test ! -e /var/tmp/JW16_MAINTENANCE
bash -n "$BUILD/tools/fuse2b/w1c_inner.sh"
CMD="bash $BUILD/tools/fuse2b/w1c_inner.sh"
bash /var/tmp/appbar/gpuwin.sh "$CMD" 2>&1 | tail -22
svc=$(systemctl is-active llm-inference || true)
echo "outer check llm-inference: $svc"
if [ "$svc" != active ]; then bash /var/tmp/appbar/gpuwin.sh true; systemctl is-active llm-inference; fi
