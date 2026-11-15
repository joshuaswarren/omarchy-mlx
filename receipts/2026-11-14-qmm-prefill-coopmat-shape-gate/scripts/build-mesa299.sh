#!/bin/bash
# MLXCOOPFIX: build the mesa #299 (coopmat/agx3b-cmat @ b70f6dc, coopmat default-on
# across the AGX3B ISA) asahi Vulkan driver NATIVELY on the J616s (M4 Pro, aarch64)
# and stage it as a user-space overlay ICD. NDEBUG release (asserts OFF), the
# shipping-style build: a mismatched coopmat shape fails pipeline creation with
# VK_ERROR_UNKNOWN rather than aborting. No install, no reboot.
# Launched detached; writes build-mesa.log and MESA-DONE (ok) / MESA-FAIL (error).
set -u
W=/var/tmp/mlxcoopfix; SRC=$W/mesa; B=$W/mesa-build; LOG=$W/build-mesa.log
WANT=b70f6dc9d68   # #299 tip (prefix; reset verifies)
mkdir -p $W; : > $LOG; rm -f $W/MESA-DONE $W/MESA-FAIL
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
fail(){ log "FAIL: $*"; touch $W/MESA-FAIL; exit 1; }

log "mesa299 overlay build on $(uname -n) $(uname -m)"
if [ ! -d $SRC/.git ]; then
  git clone --filter=blob:none https://github.com/aurora-silicon/mesa.git $SRC >>$LOG 2>&1 || fail "clone"
fi
git -C $SRC fetch origin coopmat/agx3b-cmat >>$LOG 2>&1 || fail "fetch"
git -C $SRC checkout -f coopmat/agx3b-cmat >>$LOG 2>&1 || fail "checkout"
git -C $SRC reset --hard $WANT >>$LOG 2>&1 || fail "reset $WANT"
HAVE=$(git -C $SRC rev-parse --short HEAD)
log "mesa HEAD=$HAVE"; git -C $SRC log --oneline -1 | tee -a $LOG
if [ ! -d $B ]; then
  meson setup $B $SRC -Dvulkan-drivers=asahi -Dgallium-drivers=asahi \
    -Dbuildtype=release -Db_ndebug=true >>$LOG 2>&1 || fail "meson setup"
fi
ninja -C $B src/asahi/vulkan/libvulkan_asahi.so >>$LOG 2>&1 || fail "ninja"
ICD=$(find $B -name 'libvulkan_asahi.so' | head -1)
[ -n "$ICD" ] && [ -f "$ICD" ] || fail "no libvulkan_asahi.so"
chmod -R a+rX $B $SRC 2>/dev/null
# Name satisfies omarchy is_honeykrisp_icd() (MLXCOOP 10:22 note).
OV=$W/honeykrisp_icd.aarch64.json
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.4.0"}}\n' "$ICD" > $OV
chmod a+r $OV
log "overlay ICD = $ICD"
log "overlay json = $OV"
log "MESA-DONE"
touch $W/MESA-DONE
