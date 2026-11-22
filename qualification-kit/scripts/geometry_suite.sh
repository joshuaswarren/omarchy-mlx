#!/usr/bin/env bash
# Geometry suite (G504 contract) for the M5 Linux-MLX kit.
# Usage: geometry_suite.sh OUT_DIR
# The geometry tests are the C++ omarchy_qmm_geometry_tests executable (overlay/tests/omarchy),
# which wheels do not ship. Build it from the wheel's source commit and set
# OMARCHY_QMM_GEOMETRY_TESTS to its path. Without it, the suite records "skip" with the reason
# and exits 0, so a wheel-only install does not fail the kit here.
set -euo pipefail
OUT=${1:?out dir}
bin=${OMARCHY_QMM_GEOMETRY_TESTS:-}
write() { # status
  python3 - "$OUT" "$1" "$bin" <<'PY'
import json, os, socket, sys
out, status, binary = sys.argv[1:]
try:
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()
except OSError:
    boot = "unknown"
detail = {"binary": binary}
if status == "skip":
    detail["reason"] = ("omarchy_qmm_geometry_tests is a C++ test binary that wheels do not ship; set "
                        "OMARCHY_QMM_GEOMETRY_TESTS to a build of it from the wheel's source commit")
else:
    log = os.path.join(out, "geometry.log")
    lines = open(log, errors="replace").read().splitlines()
    detail.update(log="geometry.log", summary=lines[-1] if lines else "")
json.dump({"suite": "geometry", "status": status, "boot_id": boot, "host": socket.gethostname(),
           "detail": detail}, open(os.path.join(out, "geometry.json"), "w"))
PY
  echo "suite=geometry status=$1"
}
if [ -z "$bin" ] || [ ! -x "$bin" ]; then
  write skip
  exit 0
fi
if "$bin" > "$OUT/geometry.log" 2>&1; then
  write pass
else
  write fail
  exit 1
fi
