#!/bin/bash
# Warn when /etc/default/update-m1n1 pins the device tree list with a DTBS=
# setting. update-m1n1 then builds boot.bin from that list only, so package
# device tree overlays (/usr/lib/omarchy-mac-boot/dtb-overlays, selected by
# /etc/omarchy-mac-boot/dtb-overlays.opt-in) are silently ignored. This is
# what the aurora-sep installer (Touch ID kernels) writes.
# Read-only: prints a warning and nothing else. Run before you rely on any
# opt-in overlay (gpu-pstate-*, the omarchy-ane node) or after you change the
# opt-in file. See docs/gpu-base-pstate.md.
# usage: check-dtbs-override.sh [CONFIG]
#   exit 0 - no DTBS override; the opt-in overlay path is live
#   exit 1 - a non-empty DTBS= is present; overlays are ignored
#   exit 2 - bad usage, or the config exists but cannot be read
set -u
if [ $# -gt 1 ]; then
	echo "usage: $0 [CONFIG]" >&2
	exit 2
fi
conf=${1:-/etc/default/update-m1n1}
trim() { # strip leading and trailing whitespace from $1 (prints result)
	set -- "$1"
	s=${1#"${1%%[![:space:]]*}"}
	s=${s%"${s##*[![:space:]]}"}
	printf '%s' "$s"
}
if [ ! -f "$conf" ]; then
	if [ -e "$conf" ]; then
		echo "cannot read $conf: not a regular file" >&2
		exit 2
	fi
	echo "no $conf; no DTBS override; the opt-in overlay path is live"
	exit 0
fi
found=""
lineno=0
while IFS= read -r cur || [ -n "${cur:-}" ]; do
	lineno=$((lineno + 1))
	cur=$(trim "$cur")
	case $cur in
	export[[:space:]]*) cur=$(trim "${cur#export}") ;;
	esac
	case $cur in
	DTBS=*)
		value=$(trim "${cur#*=}")
		value=${value#\"}
		value=${value%\"}
		value=${value#\'}
		value=${value%\'}
		if [ -n "$value" ]; then
			found=$cur
			break
		fi
		;;
	esac
done <"$conf" || { echo "cannot read $conf" >&2; exit 2; }
if [ -z "$found" ]; then
	echo "no non-empty DTBS= in $conf; the opt-in overlay path is live"
	exit 0
fi
echo "DTBS override found in $conf, line $lineno:"
echo "    $found"
echo "WARNING: update-m1n1 builds boot.bin from this list only. Package device"
echo "tree overlays in /usr/lib/omarchy-mac-boot/dtb-overlays (opt-in keys in"
echo "/etc/omarchy-mac-boot/dtb-overlays.opt-in: gpu-pstate-*, the omarchy-ane"
echo "node) are silently ignored. An opt-in line plus a reboot changes nothing."
echo "Before you turn an overlay on, or to check why it did not take effect,"
echo "see docs/gpu-base-pstate.md (\"If DTBS is set\")."
exit 1
