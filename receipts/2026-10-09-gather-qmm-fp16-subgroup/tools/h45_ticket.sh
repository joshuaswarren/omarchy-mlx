#!/bin/bash
# MatmulGap H45 (one gpu-turn ticket, no lock call inside, about 12 min): identity, the family tests (gather_qmm subgroup cases, fp16 and
# bf16), then the decode probe v2 on the OLD wheel (venv-head) and the NEW wheel (venv-gq). Single log with the identity beside the
# doctest lines: $H/h45-<host>.log. Env: CHIPLABEL.
set -u
H=${WORK_DIR:?set WORK_DIR to the directory with the venvs, the test tree, the probe scripts and the logs}
D=$HOME/.local/share/coreglass/vulkan-v3-6543eeb7df
export VK_ICD_FILENAMES=$D/honeykrisp_icd.aarch64.json VK_DRIVER_FILES=$D/honeykrisp_icd.aarch64.json MESA_SHADER_CACHE_DISABLE=1
HOST=$(hostname)
LOG=$H/h45-$HOST.log
BIN=$H/tbuild-gq/tests/omarchy/omarchy_matmul_family_tests
for f in "$BIN" "$H/venv-gq/bin/python" "$H/venv-head/bin/python"; do
  if [ ! -x "$f" ]; then echo "missing $f" | tee "$LOG"; exit 2; fi
done
shopt -s nullglob
LIBS=("$D"/libvulkan_*.so)
shopt -u nullglob
if [ "${#LIBS[@]}" -ne 1 ]; then echo "driver library: expected 1 match, found ${#LIBS[@]}" | tee "$LOG"; exit 2; fi
{
  echo "start $(date -u +%FT%TZ) chip ${CHIPLABEL:-unlabelled} kernel $(uname -r) load $(cut -d' ' -f1 /proc/loadavg)"
  echo "device $("$H/venv-gq/bin/python" -c 'import mlx.core as mx; print(mx.device_info().get("device_name"))' 2>&1 | tail -1)"
  echo "vulkaninfo: $(vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|driverVersion' | head -4 | tr -s ' ' | tr '\n' ';')"
  echo "driver library sha256 $(sha256sum "${LIBS[0]}" | cut -c1-16)"
  echo "family test binary sha256 $(sha256sum "$BIN" | cut -c1-16)"
  for v in venv-head venv-gq; do
    echo "$v: $("$H/$v/bin/python" "$H/mlx_provenance.py" 2> /dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["dist_version"], "verified="+d["verified"], "version_match="+str(d["version_match"]))' 2>&1 | tail -1)"
  done
} | tee "$LOG"
FULL=$H/h45-family-$HOST.log
timeout -k 20 900 "$BIN" -tc='gather qmm*' > "$FULL" 2>&1
echo "family (gather qmm cases) rc=$? $(grep -E '^\[doctest\] (test cases|assertions|Status)' "$FULL" | tr '\n' ' ')" | tee -a "$LOG"
grep -E 'gather-qmm-sub' "$FULL" | tee -a "$LOG" > /dev/null
for v in venv-head venv-gq; do
  sleep "${GAP:-30}"
  echo "probe $v" | tee -a "$LOG"
  timeout -k 20 280 "$H/$v/bin/python" "$H/gqmm_probe2.py" 40 60 2> "$H/h45-$HOST-$v.err" | tee -a "$LOG"
done
echo "end $(date -u +%FT%TZ)" | tee -a "$LOG"
