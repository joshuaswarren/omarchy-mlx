#!/bin/bash
# GPU ticket body (MAXMIN 8) for G13C (jw16). recur32 is default-on only for G13G (g13_legacy_part); on G13C a masked row takes the
# two-pass snapshot scan (device test: 2 dispatches against 1 maskless), so the masked-recur32 change (qrm) does nothing there by
# default (pfbgrm jw16: qgpr 1.638/1.643, qrm 1.628/1.646). Does MLX_OMARCHY_GDN_RECUR32=2 (recur32 for every T >= 2, masked too)
# fix the padded batched prefill on G13C? Arms (mirrored): qrm default, qrm env=2, qrm env=2, qrm default. Per arm ratio =
# batched_s / seq_s of pfbg.py; first_tokens_equal must be true or the arm is void.
# Rule (pre-registered with this ticket): env=2 median ratio <= 1.15 AND <= 0.85 x the default median => a masked-only recur32
# default for G13C is justified (a separate change; the maskless route stays coopmat). Otherwise no change for G13C.
# Refuses under 6 GiB MemAvailable. Slowest submit is the two-pass scan (about 10 s for the whole four-row prefill).
set -u
B=~/.local/share/coreglass
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt 6291456 ]; then echo "@@pe SKIPPED MemAvailable ${avail_kb} kB < 6 GiB"; exit 0; fi
export VK_DRIVER_FILES=$B/vulkan-6543eeb7df/honeykrisp_icd.aarch64.json HF_HUB_OFFLINE=1
export PYTHONPATH=$B/mlx-lm-series-main/lib/python3.14/site-packages
for arm in default env2 env2 default; do
  sleep ${COOL:-3}
  raw=$B/build/pfbgenv-raw-$$-$arm.txt
  if [ $arm = env2 ]; then export MLX_OMARCHY_GDN_RECUR32=2; else unset MLX_OMARCHY_GDN_RECUR32; fi
  $B/venv-qrm/bin/python $B/prof/pfbg.py $B/models/qwen3_5-4bit $B/omlx-bench/omlx-prompt.txt ${PFL:-454 417 363 304} > "$raw" 2>&1
  echo "@@pe $arm rc=$?"
  grep -E '^PFBG|Error' "$raw" | sed "s/^/@@pe $arm /"
done
unset MLX_OMARCHY_GDN_RECUR32
echo "@@pe done $(date -u +%T)"
