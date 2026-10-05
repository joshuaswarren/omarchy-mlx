#!/usr/bin/env bash
# width-204 rope_rms_norm cell: fused vs composed vs fp64 on this host's GPU.
# Golden-clone a warm tree, build the test target (gives libmlx.a), compile
# the probe against it, run, then delete the clone.
#   bash rnb_w204_gpu_case.sh
set -euo pipefail
TAG=rnb-w204
# No local busy guard: the gpu-turn ticket IS the serialization; an abort
# here just loses the FIFO slot under contention.
HERE="$(cd "$(dirname "$0")" && pwd)"
D=/var/tmp/${TAG}-tree
python3 - <<'PY'
import shutil, os
p = "/var/tmp/rnb-w204-tree"
if os.path.isdir(p):
    shutil.rmtree(p, ignore_errors=True)
    print("pre-cleaned stale clone")
PY
if [ -d /var/tmp/golden-wheel ]; then
  bash /var/tmp/golden-clone-tree.sh "$D"
  SRC="$D/.work/mlx"
else
  echo "FATAL: no /var/tmp/golden-wheel to clone; prepare a tree first" >&2
  exit 2
fi
# golden's staged tree predates current main; re-point and re-prepare so the
# probe tests the current patches (Main: b226895dc fixed fast.h).
cd "$D"
git fetch origin main 2>&1 | tail -1
git checkout -q -f origin/main
echo "tree at $(git rev-parse --short=7 HEAD)"
./scripts/prepare-mlx.sh
cd "$SRC"
cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release . \
  > "/var/tmp/${TAG}-cmake.log" 2>&1
nice -n 10 make -j4 omarchy_fast_ops_tests > "/var/tmp/${TAG}-build.log" 2>&1
GUF=$(find "$SRC" -name libgguflib.a | head -1 || true)
# link ladder: blas/lapack first, then openblas (Arch/T6021 golden trees:
# libblas lacks cblas; openblas provides it), then bare
if ! g++ -std=gnu++20 -O2 -I "$SRC" "$HERE/rnb_w204_probe.cpp" \
  "$SRC/libmlx.a" $GUF -llapack -lblas -lpthread -ldl \
  -o "/var/tmp/${TAG}-probe" > "/var/tmp/${TAG}-link.log" 2>&1; then
  if ! g++ -std=gnu++20 -O2 -I "$SRC" "$HERE/rnb_w204_probe.cpp" \
    "$SRC/libmlx.a" $GUF -lopenblas -lpthread -ldl \
    -o "/var/tmp/${TAG}-probe" >> "/var/tmp/${TAG}-link.log" 2>&1; then
    g++ -std=gnu++20 -O2 -I "$SRC" "$HERE/rnb_w204_probe.cpp" \
      "$SRC/libmlx.a" $GUF -lpthread -ldl -o "/var/tmp/${TAG}-probe" \
      >> "/var/tmp/${TAG}-link.log" 2>&1
  fi
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
