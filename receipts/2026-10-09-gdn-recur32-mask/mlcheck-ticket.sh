#!/bin/bash
# GPU ticket body (MAXMIN 6; M2: sleep 60 first, COOL=60). Maskless prefill A/B, control qgpr against qrm (the masked-recur32 change),
# mirrored qgpr qrm qrm qgpr, one fresh process per arm. Rule: mlcheck.py docstring (fixed before the wheel was built), scored by
# mlcheck_decide.py. Each call is one submit of a few ms (T 128 and 512, one head group). Refuses under 6 GiB MemAvailable.
set -u
B=~/.local/share/coreglass
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt 6291456 ]; then echo "@@ml SKIPPED MemAvailable ${avail_kb} kB < 6 GiB"; exit 0; fi
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json
unset MLX_OMARCHY_GDN_RECUR32 MLX_OMARCHY_NO_COOPMAT_GDN
for arm in qgpr ${CHG:-qrm} ${CHG:-qrm} qgpr; do
  sleep ${COOL:-3}
  $B/venv-$arm/bin/python $B/prof/mlcheck.py 2>&1 | grep -E '^MLCHECK|Error' | sed "s/^/@@ml ${arm/qrm3/qrm} /"
done
echo "@@ml done $(date -u +%T)"
