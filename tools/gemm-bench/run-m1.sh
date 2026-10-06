#!/usr/bin/env bash
# Run gemm-bench on an M1-family host under the shared GPU lock.
# usage: run-m1.sh <tag> <gemm-bench args...>; output also in out/<tag>.log
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
tag=$1
shift
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json}
# jwm1 drops the GPU into a low performance state whenever the CPU complex
# may enter its deepest idle state (H322/H325); a zero-latency PM QoS
# request held for the whole run keeps every arm in the same state. Hosts
# without access to the node (the M1 Max shows no such effect, H328) run
# without it, and the log says so.
qos=held
if ! { exec 3>/dev/cpu_dma_latency; } 2>/dev/null; then
  qos=unavailable
else
  printf '\x00\x00\x00\x00' >&3
fi
{
  echo "# $(date -u +%FT%TZ) tag=$tag boot=$(cat /proc/sys/kernel/random/boot_id) load=$(cut -d' ' -f1-3 /proc/loadavg) icd=$VK_ICD_FILENAMES cpu_dma_latency=$qos"
  echo "# cmd: ./gemm-bench $*"
  rc=0
  flock -w 1800 /tmp/m1-gpu.lock timeout -s KILL 300 ./gemm-bench "$@" 2>&1 || rc=$?
  echo "# end $(date -u +%FT%TZ) rc=$rc"
} | tee "out/$tag.log"
