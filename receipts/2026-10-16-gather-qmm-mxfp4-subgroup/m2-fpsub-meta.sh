#!/bin/bash
# m2-fpsub-meta.sh: the metadata the mxfp4 subgroup receipt lacks (PR 214 review): scripts/mlx_provenance.py against the wheel and the unpacked tree the numerics ran on, the installed packages
# (Mesa, the Vulkan driver, kernel and firmware extractor), the Vulkan device summary if vulkaninfo exists, and a device reopen in a fresh process. No benchmark, no model.
set -uo pipefail
W=${WORK:?set WORK to the working directory}
S=${PYENV:?set PYENV to a directory with venv/bin/python}
WHL=$(find "$W/fpsub-wheel" -maxdepth 1 -name 'mlx_omarchy-*+2f8fc40*-cp314-cp314-linux_aarch64.whl' | head -1)
TREE=$W/fpsub-tree
O=$W/fpsub-meta-m2-1
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "FPMETA $*" | tee -a "$L"; }
[ -n "$WHL" ] && [ -d "$TREE" ] || { say "ABORT no wheel or tree"; exit 1; }
say "start $(date -u +%FT%TZ) kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -c1-64)"
say "packages: $(pacman -Q mesa vulkan-asahi-omarchy linux-aurora asahi-fwextract 2>/dev/null | tr '\n' ';')"
if command -v vulkaninfo > /dev/null 2>&1; then
  vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|driverInfo|apiVersion|driverVersion" | head -8 | while IFS= read -r ln; do say "vulkaninfo: $ln"; done
else
  say "vulkaninfo: not installed"
fi
say "provenance script sha256 $(sha256sum "$W/mlx_provenance.py" | cut -c1-64)"
PYTHONPATH="$TREE" timeout 120 "$S/venv/bin/python" "$W/mlx_provenance.py" --expect-wheel "$WHL" > "$O/prov.json" 2> "$O/prov.err"
say "provenance rc=$?"
python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print({k:d.get(k) for k in ('dist','verified','mismatch','dist_version','mx_version','version_match','expect_wheel')})" "$O/prov.json" 2>&1 | while IFS= read -r ln; do say "provenance: $ln"; done
PYTHONPATH="$TREE" timeout 120 "$S/venv/bin/python" -c "import mlx.core as mx; y=mx.ones((4,))+1; mx.eval(y); print('reopen', mx.default_device(), y.tolist())" > "$O/reopen.out" 2>&1
say "device reopen rc=$? $(tail -1 "$O/reopen.out")"
say "DONE out=$O"
