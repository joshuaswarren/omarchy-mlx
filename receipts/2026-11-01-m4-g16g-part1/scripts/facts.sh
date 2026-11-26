#!/bin/bash
# Host, driver, firmware and model identity for the receipts. Prints only
# public identifiers: no host name, user name, serial, address or home path.
H=$HOME
red() { sed -e "s#$H#~#g" -e "s#$(hostname)#<host>#g" -e "s#$(whoami)#<user>#g"; }
echo "date_utc $(date -u +%FT%TZ)"
echo "machine $(tr -d '\0' < /proc/device-tree/model) ($(tr '\0' ' ' < /proc/device-tree/compatible))"
echo "memory_total_gib $(awk '/MemTotal/{printf "%.1f", $2/2^20}' /proc/meminfo)  cpus $(nproc) online $(cat /sys/devices/system/cpu/online)"
echo "os $(. /etc/os-release; echo $PRETTY_NAME)"
echo "kernel $(uname -r) $(uname -v)"
echo "gpu_kernel_driver $(journalctl -k -b --no-pager -q | grep -m1 -oE 'asahi_m4 [^:]+: G16: GPU ID=.*')"
echo "gpu_boot_firmware $(journalctl -k -b --no-pager -q | grep -m1 -oE 'boot firmware [^ ]+')"
echo "gpu_driver_hw_set $(journalctl -k -b --no-pager -q | grep -oE 'hardware data \+0x[0-9a-f]+ = 0x[0-9a-f]+' | tr '\n' ';')"
echo "pmp_bound_devices $(ls /sys/bus/platform/drivers/apple_pmp 2>/dev/null | grep -c '@')"
L=$H/gpu/mesa-install/lib/libvulkan_asahi.so
echo "mesa_lib $(strings $L | grep -m1 -oE 'Mesa [0-9.]+-devel \(git-[0-9a-f]+\)') sha256=$(sha256sum $L | cut -c1-16)"
echo "mesa_source $(git -C $H/gpu/gravity-mesa log -1 --format='%h %s' $(strings $L | grep -m1 -oE 'git-[0-9a-f]+' | cut -d- -f2)) (branch m4-honeykrisp-compute)"
echo "glslc $(glslc --version | head -1)"
echo "python $(python3 --version)"
echo "env AGX_APPLE9_MAX_SHARED=32768 MLX_OMARCHY_BATCH_WORK=120000"
VK_ICD_FILENAMES=$H/gpu/mesa-install/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$H/gpu/mesa-install/lib AGX_APPLE9_MAX_SHARED=32768 \
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|driverInfo|apiVersion|driverVersion|conformanceVersion" | head -8 | sed 's/^/vulkan /'
VK_ICD_FILENAMES=$H/gpu/mesa-install/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$H/gpu/mesa-install/lib \
  vulkaninfo 2>/dev/null | grep -m3 -E "VK_KHR_cooperative_matrix" | sed 's/^/vulkan_ext /'
for d in $H/.cache/huggingface/hub/models--*; do
  repo=$(basename $d | sed 's/^models--//; s/--/\//')
  snap=$(ls -d $d/snapshots/* | head -1)
  rev=$(basename $snap)
  q=$(python3 -c "import json,sys; c=json.load(open('$snap/config.json')); q=c.get('quantization') or c.get('quantization_config'); print(json.dumps(q, sort_keys=True) if q else 'none', c.get('torch_dtype', c.get('dtype','?')))" 2>/dev/null)
  for f in $snap/*.safetensors; do
    echo "model $repo rev=${rev:0:12} $(basename $f) sha256=$(sha256sum $f 2>/dev/null | cut -c1-64) quant=$q"
  done
done
