#!/usr/bin/env bash
# Fuse2B W2 outer: dispatch-count proof window.
set -euo pipefail
BUILD=/var/tmp/fuse2b-build
test ! -e /var/tmp/JW16_MAINTENANCE
bash -n "$BUILD/tools/fuse2b/w2_profile_inner.sh"
CMD="bash $BUILD/tools/fuse2b/w2_profile_inner.sh"
bash /var/tmp/appbar/gpuwin.sh "$CMD" 2>&1 | tail -18
svc=$(systemctl is-active llm-inference || true)
echo "outer check llm-inference: $svc"
if [ "$svc" != active ]; then bash /var/tmp/appbar/gpuwin.sh true; systemctl is-active llm-inference; fi
