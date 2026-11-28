#!/usr/bin/env bash
# m3-tester-kit r1 — read-only base-M3 ANE preflight.
#
# Reports what the omarchy-ane bring-up ladder would face on this machine
# WITHOUT touching hardware: no insmod, no dmesg writes, no root, no
# writes outside the report file. Every check is a read of /proc, /sys,
# /etc or a --check-style probe. Check names, the chip->module map and the
# refusal wording mirror scripts/m3m4_kit.sh's opt-in ANE stage (which
# mirrors omarchy-ane ane/h15 and ane/h16).
#
# Usage:
#   bash ane-preflight.sh            # prints the report
#   ANE_ROOT=/fake/root bash ane-preflight.sh   # tests/chroot re-root
#
# Exit 0 when no REFUSED line prints; exit 1 otherwise. A refused line is
# a fact about this machine's ANE readiness, not a kit failure.
set -euo pipefail

# ANE_ROOT re-roots /proc, /etc and /sys for tests; "/" = the real fs.
ROOT="${ANE_ROOT:-/}"
ROOT="${ROOT%/}"
fails=0

# <name> <0=ok|other> <detail>; records REFUSED lines, never stops early.
report() {
  if [ "$2" = "0" ]; then
    echo "ANE preflight $1: PASS${3:+ ($3)}"
  else
    echo "ANE preflight $1: REFUSED ($3)"
    fails=$((fails + 1))
  fi
}

info() { echo "ANE preflight $1: $2"; }

soc="$(tr '\0' '\n' < "$ROOT/proc/device-tree/compatible" 2>/dev/null |
  grep -E '^apple,t[0-9]+$' | head -1 || true)"
soc="${soc#apple,}"
info soc "${soc:-unknown}"

case "$soc" in
  t8122 | t6030 | t6031 | t6034) module="ane_h15" ;;
  t8132 | t6040 | t6041) module="ane_h16" ;;
  *) module="" ;;
esac
report chip-map "$( [ -n "$module" ] && echo 0 || echo 1 )" \
  "no omarchy-ane bring-up module for '${soc:-unknown}' (M3 family -> ane_h15, M4 -> ane_h16)"

loaded=""
for m in ane ane_t6021 ane_h15 ane_h16; do
  if [ -d "$ROOT/sys/module/$m" ]; then loaded="$m"; break; fi
done
report no-ane-module "$( [ -z "$loaded" ] && echo 0 || echo 1 )" \
  "module '$loaded' already loaded; bring-up stages refuse to continue past it — reboot clears"

key=""
[ "$module" = "ane_h15" ] && key="ane-h15-experimental"
[ "$module" = "ane_h16" ] && key="ane-h16-experimental"
optin="$ROOT/etc/omarchy-mac-boot/dtb-overlays.opt-in"
if [ -n "$key" ]; then
  rc=1
  [ -f "$optin" ] && grep -qx "$key" "$optin" && rc=0
  report opt-in-key "$rc" \
    "line '$key' not in $optin — add it and reboot before any bring-up stage"
else
  info opt-in-key "no opt-in key applies (no bring-up module for this chip)"
fi

fwrc=0
if [ "$module" = "ane_h16" ]; then
  if command -v omarchy-ane-firmware-fetch >/dev/null 2>&1; then
    omarchy-ane-firmware-fetch --check --root "$ROOT" >/dev/null 2>&1 || fwrc=1
  else
    fwrc=1
  fi
  report pinned-firmware "$fwrc" \
    "the h16 module accepts only the pinned leto/aether bytes — install omarchy-ane-firmware-fetch and fetch them"
else
  info pinned-firmware "not checked (h15 has no firmware pin yet; its boot stage takes fw_path= explicitly)"
fi

ko=""
for c in "$ROOT/usr/lib/omarchy-ane/${module:-ane_h15}.ko" \
         "$ROOT/usr/lib/modules/$(uname -r)/extra/${module:-ane_h15}.ko"; do
  [ -r "$c" ] && ko="$c" && break
done
if [ -n "$ko" ]; then
  ver="$(modinfo -F vermagic "$ko" 2>/dev/null | head -1 || true)"
  rc=1
  [ -n "$ver" ] && [ "${ver%% *}" = "$(uname -r)" ] && rc=0
  report ko-vermagic "$rc" \
    "$ko vermagic '${ver:-unreadable}' vs running '$(uname -r)'"
else
  info ko-file "no ${module:-ane_h15}.ko found under /usr/lib/omarchy-ane or modules/extra — build it before staging (expected on a base install)"
fi

echo
if [ "$fails" -eq 0 ]; then
  echo "ANE preflight: no refusals — this base M3 is bring-up ready."
else
  echo "ANE preflight: $fails refusal(s) above; each names its fix."
fi
exit "$([ "$fails" -eq 0 ] && echo 0 || echo 1)"
