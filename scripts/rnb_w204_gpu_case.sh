#!/usr/bin/env bash
# width-204 rope_rms_norm cell: fused vs composed vs fp64 on this host's GPU.
# Golden-clone a warm tree, build the test target (gives libmlx.a), compile
# the probe against it, run, log, then delete the clone.
#   bash rnb_w204_gpu_case.sh
set -euo pipefail
TAG=rnb-w204
if pgrep -f '[c]c1plus|[c]make|[n]inja' > /dev/null; then
  echo "BUILD BUSY: another build is running; aborting without starting one." >&2
  exit 3
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
D=/var/tmp/${TAG}-tree
rm -rf "$D" 2>/dev/null || true
if [ -d /var/tmp/golden-wheel ]; then
  bash /var/tmp/golden-clone-tree.sh "$D"
  SRC="$D/.work/mlx"
else
  echo "FATAL: no /var/tmp/golden-wheel to clone; prepare a tree first" >&2
  exit 2
fi
cd "$SRC"
cmake -DMLX_BUILD_TESTS=ON . > "/var/tmp/${TAG}-cmake.log" 2>&1
nice -n 10 ninja -j4 omarchy_fast_ops_tests > "/var/tmp/${TAG}-build.log" 2>&1
# link the probe with the same libs the test target uses (minus libmlx.a)
LIBS=$(grep -A6 "^build tests/omarchy/omarchy_fast_ops_tests" build.ninja \
  | tr ' ' '\n' | grep -E '\.so$|\.a$' | grep -v 'libmlx.a' | sort -u || true)
g++ -std=gnu++20 -O2 -I "$SRC" "$HERE/rnb_w204_probe.cpp" \
  "$SRC/libmlx.a" $LIBS -lpthread -ldl -o "/var/tmp/${TAG}-probe" 2>&1 | head -5
echo "== probe on $(uname -m) GPU:"
"/var/tmp/${TAG}-probe" 2>&1
echo "== done; results above; cleaning clone"
cd /var/tmp
python3 - <<'PY'
import shutil
shutil.rmtree("/var/tmp/rnb-w204-tree", ignore_errors=True)
print("clone deleted")
PY
