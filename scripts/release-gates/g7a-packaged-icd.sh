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
# The packaged ICD is the only ICD an Omarchy M+ image has; the stock-Mesa
# asahi default guaranteed a driver-sha mismatch here. Aurora renames the
# driver to Borealis (Ryan 2026-11-20), so the packaged file may carry
# either name: publish it under whatever name the package installed.
# Borealis comes first, as in the runtime (packaged_icd_names() in
# honeykrisp_identity.h, PACKAGED_ICD_NAMES in serve/mlx_omarchy_paths.py),
# so the gate checks the file the runtime loads. SYSTEM_ICD wins when it
# names a file; SYSTEM_ICD_DIR moves the search.
: "${SYSTEM_ICD_DIR:=/usr/lib/omarchy-mlx/vulkan}"
SRC_ICD="${SYSTEM_ICD:-$SYSTEM_ICD_DIR/honeykrisp_icd.aarch64.json}"
if { [ -z "${SYSTEM_ICD:-}" ] || [ ! -f "$SRC_ICD" ]; } \
    && [ -f "$SYSTEM_ICD_DIR/borealis_icd.aarch64.json" ]; then
  SRC_ICD="$SYSTEM_ICD_DIR/borealis_icd.aarch64.json"
fi
cp "$SRC_ICD" "$VDST/$(basename "$SRC_ICD")"
# A rerun with the same TAG reuses $DEST: after the copy, drop the other
# name an earlier run published, so the runtime resolves this run's pick.
for name in borealis_icd.aarch64.json honeykrisp_icd.aarch64.json; do
  [ "$name" = "$(basename "$SRC_ICD")" ] || rm -f "$VDST/$name"
done
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
# pipefail: non-zero when mlx-omarchy-info or the contract check fails.
RC=$?
gate_log "$LOG" "GATE7A_EXIT $RC"
exit "$RC"
