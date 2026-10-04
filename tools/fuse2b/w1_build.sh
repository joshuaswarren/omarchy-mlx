#!/usr/bin/env bash
# Fuse2B W1 outer: stage scripts, run the build window via gpuwin, mirror artifacts.
set -euo pipefail
LANE=/var/tmp/fuse2b
TOOLS=$LANE
BUILD=/var/tmp/fuse2b-build
mkdir -p "$LANE"
test ! -e /var/tmp/JW16_MAINTENANCE
for f in /var/tmp/appbar/gpuwin.sh /var/tmp/appbar/build_wheel_venv.sh \
         "$BUILD/scripts/apply-mlx-lm-patches.sh" "$BUILD/scripts/mlx_provenance.py" \
         "$BUILD/tools/fuse2b/gdn_conv_delta_bitcheck.py" /var/tmp/v072-venv-fused/bin/python; do
  test -e "$f" || { echo "missing prerequisite: $f"; exit 1; }
done
bash -n "$BUILD/tools/fuse2b/w1_build_inner.sh"
CMD="bash $BUILD/tools/fuse2b/w1_build_inner.sh"
bash /var/tmp/appbar/gpuwin.sh "$CMD" 2>&1 | tail -25
svc=$(systemctl is-active llm-inference || true)
echo "outer check llm-inference: $svc"
if [ "$svc" != active ]; then bash /var/tmp/appbar/gpuwin.sh true; systemctl is-active llm-inference; fi
