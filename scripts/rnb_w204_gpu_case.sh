#!/usr/bin/env bash
# width-204 rope_rms_norm cell: fused vs composed vs fp64 on this host's GPU.
# Golden-clone a warm tree, build the test target (gives libmlx.a), compile
# the probe against it, run, then delete the clone.
#   bash rnb_w204_gpu_case.sh
set -euo pipefail
TAG=rnb-w204
if pgrep -f '[c]c1plus|[c]make|[n]inja' > /dev/null; then
  echo "BUILD BUSY: another build is running; aborting without starting one." >&2
  exit 3
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
D=/var/tmp/${TAG}-tree
if [ -d /var/tmp/golden-wheel ]; then
  bash /var/tmp/golden-clone-tree.sh "$D"
  SRC="$D/.work/mlx"
else
  echo "FATAL: no /var/tmp/golden-wheel to clone; prepare a tree first" >&2
  exit 2
fi
cd "$SRC"
cmake -DMLX_BUILD_TESTS=ON . > "/var/tmp/${TAG}-cmake.log" 2>&1
nice -n 10 make -j4 omarchy_fast_ops_tests > "/var/tmp/${TAG}-build.log" 2>&1
GUF=$(find "$SRC" -name libgguflib.a | head -1 || true)
# link ladder: blas/lapack first, then without
if ! g++ -std=gnu++20 -O2 -I "$SRC" "$HERE/rnb_w204_probe.cpp" \
  "$SRC/libmlx.a" $GUF -llapack -lblas -lpthread -ldl \
  -o "/var/tmp/${TAG}-probe" > "/var/tmp/${TAG}-link.log" 2>&1; then
  g++ -std=gnu++20 -O2 -I "$SRC" "$HERE/rnb_w204_probe.cpp" \
    "$SRC/libmlx.a" $GUF -lpthread -ldl -o "/var/tmp/${TAG}-probe" \
    > "/var/tmp/${TAG}-link.log" 2>&1
fi
echo "== probe on $(uname -m) GPU:"
"/var/tmp/${TAG}-probe" 2>&1
echo "== done; results above; cleaning clone"
cd /var/tmp
python3 - <<'PY'
import shutil
shutil.rmtree("/var/tmp/rnb-w204-tree", ignore_errors=True)
print("clone deleted")
PY
