#!/bin/bash
# Compile the device tree overlays in packaging/dt for a package recipe:
#   packaging/dt/PREFIX-NAME.dts  ->  DESTDIR/usr/share/omarchy-platform/dtb-overlays/PREFIX/omarchy-NAME.dtbo
# PREFIX selects the board device trees by file name (t8103 = every t8103-*.dtb).
# The directory is the one omarchy-ane installs into and omarchy-mac-boot /
# omarchy-ane-dt apply from. An overlay whose root has the string
# "omarchy,opt-in" is applied only when that string is a line of
# /etc/omarchy-platform/dtb-overlays.opt-in; the owner writes that file.
# For a recipe: packaging/build-dtbo.sh "$pkgdir"   (needs dtc and fdtget)
set -euo pipefail
shopt -s nullglob
destdir=${1:?usage: build-dtbo.sh DESTDIR}
here=$(dirname -- "$(realpath -- "$0")")
overlay_dir=usr/share/omarchy-platform/dtb-overlays
n=0
for src in "$here"/dt/*.dts; do
	base=${src##*/}
	base=${base%.dts}
	prefix=${base%%-*}
	name=${base#"$prefix"-}
	[[ $prefix != "$base" && -n $name ]] || { echo "build-dtbo: $base.dts: not named PREFIX-NAME.dts" >&2; exit 1; }
	out=$destdir/$overlay_dir/$prefix/omarchy-$name.dtbo
	install -d "${out%/*}"
	dtc -q -@ -I dts -O dtb -o "$out" "$src"
	fdtget -t s "$out" / omarchy,opt-in >/dev/null 2>&1 || { echo "build-dtbo: $base.dts: no omarchy,opt-in root string; overlays from this package are opt-in only" >&2; exit 1; }
	echo "build-dtbo: $base.dts -> /${out#"$destdir"/}"
	n=$((n + 1))
done
[[ $n -gt 0 ]] || { echo "build-dtbo: no overlays in $here/dt" >&2; exit 1; }
