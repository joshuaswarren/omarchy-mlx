#!/usr/bin/env bash
# Gate 7a — packaged-ICD fixture contract via the OMARCHY_MLX_SYSTEM_PREFIX
# seam, against the gate-1 install and the 7b staged prefix.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g7a-packaged-icd.log"
DEST="/tmp/${TAG}-sysinst"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"

VDST="$DEST/usr/lib/omarchy-mlx/vulkan"
mkdir -p "$VDST"
# the packaged honeykrisp ICD is the only ICD an Omarchy M+ image has;
# the stock-Mesa asahi default guaranteed a driver-sha mismatch here
cp "${SYSTEM_ICD:-/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json}" "$VDST/honeykrisp_icd.aarch64.json"
DRIVER_SHA=$(env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" \
  "$GATE_HOME/.local/bin/mlx-omarchy-info" --json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["driver_sha"])')
printf '%s\n' "$DRIVER_SHA" > "$VDST/mesa-git-sha"
gate_log "$LOG" "fixture_driver_sha $DRIVER_SHA"

env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" \
  OMARCHY_MLX_SYSTEM_PREFIX="$DEST/usr/lib/omarchy-mlx" \
  "$GATE_HOME/.local/bin/mlx-omarchy-info" --json 2>>"$LOG" \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
for k in ("icd_source", "icd_path", "expected_sha", "expected_sha_source", "driver_sha"):
    print(k, "=", d.get(k))
ok = d.get("icd_source") == "packaged" and d.get("expected_sha") == d.get("driver_sha") != ""
print("ICD_CONTRACT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
' | tee -a "$LOG"
RC=${PIPESTATUS[0]}
gate_log "$LOG" "GATE7A_EXIT $RC"
exit "$RC"
