#!/usr/bin/env bash
# M5 Linux-MLX qualification kit (G510 run; staged by G518).
# Usage: m5-g510-run-kit.sh <WHEEL> <WHEEL_SHA256> <ICD_JSON> <ICD_SO_SHA256> <OUT_DIR>
#   ICD_JSON: the G509 private ICD manifest (not the system /usr/share/vulkan/icd.d one).
# Rule (#547): a run counts only if the G17C (0x6050) device behind THIS ICD did the work.
# Any CPU device (llvmpipe/lavapipe/CPU stream) or the system ICD fails the run.
# Each suite writes OUT_DIR/<suite>.json; the script stops at the first failing suite.
set -euo pipefail
WHEEL=${1:?wheel}; WHEEL_SHA=${2:?wheel sha256}; ICD=${3:?G509 ICD json}; ICD_SHA=${4:?ICD .so sha256}; OUT=${5:?out dir}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
suite() { # name status detail-json
  printf '{"suite":"%s","status":"%s","boot_id":"%s","host":"%s","detail":%s}\n' "$1" "$2" \
    "$(cat /proc/sys/kernel/random/boot_id)" "$(uname -n)" "$3" > "$OUT/$1.json"
  echo "suite=$1 status=$2"; [ "$2" = pass ] || [ "$2" = skip ] || exit 1
}

# 1. wheel pin
got=$(sha256sum "$WHEEL" | cut -c1-64)
[ "$got" = "$WHEEL_SHA" ] || suite wheel-pin fail "{\"want\":\"$WHEEL_SHA\",\"got\":\"$got\"}"
python3 -m venv "$OUT/venv"
"$OUT/venv/bin/pip" install -q --no-deps "$WHEEL"
suite wheel-pin pass "{\"wheel\":\"$(basename "$WHEEL")\",\"sha256\":\"$got\"}"

# 2. ICD pin: the manifest must name a library outside the system ICD dirs, with the pinned sha
lib=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["ICD"]["library_path"])' "$ICD")
case "$lib" in /usr/share/vulkan/*|/etc/vulkan/*|/usr/lib/libvulkan_*|libvulkan_*) suite icd-pin fail "{\"reason\":\"system-icd\",\"library\":\"$lib\"}";; esac
[ -f "$lib" ] || suite icd-pin fail "{\"reason\":\"library-missing\",\"library\":\"$lib\"}"
lsha=$(sha256sum "$lib" | cut -c1-64)
[ "$lsha" = "$ICD_SHA" ] || suite icd-pin fail "{\"reason\":\"sha-mismatch\",\"want\":\"$ICD_SHA\",\"got\":\"$lsha\"}"
suite icd-pin pass "{\"manifest\":\"$ICD\",\"library\":\"$lib\",\"sha256\":\"$lsha\"}"

# Only the G509 ICD is visible from here on: no system ICD can be picked up.
export VK_ICD_FILENAMES="$ICD" VK_DRIVER_FILES="$ICD"
unset VK_ADD_DRIVER_FILES MLX_OMARCHY_ALLOW_CPU 2>/dev/null || true

# 3. G17C admission: Vulkan sees exactly one device, it is 0x6050, and it is not a CPU device
command -v vulkaninfo >/dev/null || suite g17c-admission fail '{"reason":"vulkaninfo-missing"}'
vulkaninfo --summary > "$OUT/vulkaninfo.txt" 2>&1 || true
ndev=$(grep -c 'deviceName' "$OUT/vulkaninfo.txt" || true)
devid=$(grep -m1 -o 'deviceID *= *0x[0-9a-fA-F]*' "$OUT/vulkaninfo.txt" | grep -o '0x[0-9a-fA-F]*' || true)
dtype=$(grep -m1 -o 'deviceType *= *[A-Z_]*' "$OUT/vulkaninfo.txt" | awk '{print $NF}' || true)
dname=$(grep -m1 'deviceName' "$OUT/vulkaninfo.txt" | sed 's/.*= *//' || true)
if [ "$ndev" != 1 ] || [ "${devid,,}" != 0x6050 ] || [ "$dtype" = PHYSICAL_DEVICE_TYPE_CPU ] || echo "$dname" | grep -qiE 'llvmpipe|lavapipe|swiftshader'; then
  suite g17c-admission fail "{\"devices\":\"$ndev\",\"deviceID\":\"$devid\",\"deviceType\":\"$dtype\",\"deviceName\":\"$dname\"}"
fi
suite g17c-admission pass "{\"deviceID\":\"$devid\",\"deviceType\":\"$dtype\",\"deviceName\":\"$dname\"}"

# 4. MLX on that device: default device is GPU, a kernel runs, and the process mapped the pinned ICD library
"$OUT/venv/bin/python" - "$lib" "$OUT/mlx-device.json" <<'PY'
import json, sys, mlx.core as mx
lib, out = sys.argv[1], sys.argv[2]
dev = str(mx.default_device())
a = mx.random.normal((512, 512)); b = (a @ a.T).sum(); mx.eval(b)
maps = open("/proc/self/maps").read()
ok = "gpu" in dev.lower() and lib in maps and "llvmpipe" not in maps and "lavapipe" not in maps
json.dump({"suite": "mlx-device", "status": "pass" if ok else "fail", "default_device": dev,
           "icd_mapped": lib in maps, "cpu_icd_mapped": ("llvmpipe" in maps or "lavapipe" in maps),
           "checksum": float(b)}, open(out, "w"))
sys.exit(0 if ok else 1)
PY
echo "suite=mlx-device status=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$OUT/mlx-device.json")"

# 5. geometry tests (G504 contract), run under the same ICD-only environment. The tests are the
# C++ omarchy_qmm_geometry_tests binary, which wheels do not ship; see geometry_suite.sh.
bash "$(dirname "$0")/geometry_suite.sh" "$OUT"

echo "ALL SUITES PASS OR SKIPPED (see geometry.json): the 5-pair tok/s A/B (G510 step 3) may run with this environment"
