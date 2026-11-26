#!/bin/bash
# MatmulGap H40 ticket (one gpu-turn ticket, no lock call inside, about 14 min; GAP=60 on the M2 Max):
#   1. output hashes of the flash route, old (venv-cf, +3c767b30) and new (venv-cf2, tile order reversed), L 512 / 1000 / 1024 / 2048
#   2. device test of the new build (tbuild-cf2): tolerance gate, 3-run identity
#   3. micro (H372 harness, causal), L 512 / 1024 / 2048, 3 repeats, arms alternated: composed, old, new, GAP seconds idle before each
# Output: $H/h40-<host>.log
set -u
H=$WORK
D=$PRIVATE_ICD_DIR
export VK_ICD_FILENAMES=$D/honeykrisp_icd.aarch64.json MESA_SHADER_CACHE_DISABLE=1
HOST=$(hostname)
LOG=$H/h40-$HOST.log
for f in "$H/venv-cf/bin/python" "$H/venv-cf2/bin/python" "$H/tbuild-cf2/tests/omarchy/omarchy_sdpa_prefill_flash_tests"; do
  if [ ! -x "$f" ]; then echo "missing $f" | tee "$LOG"; exit 2; fi
done
echo "start $(date -u +%FT%TZ) host $HOST kernel $(uname -r) load $(cut -d' ' -f1 /proc/loadavg)" | tee "$LOG"
for v in venv-cf venv-cf2; do
  echo "hash $v: $("$H/$v/bin/python" "$H/sdpa_hash.py" 2>&1 | tail -1 | cut -c1-600)" | tee -a "$LOG"
done
FULL=$H/h40-bringup-$HOST.log
timeout -k 20 800 "$H/tbuild-cf2/tests/omarchy/omarchy_sdpa_prefill_flash_tests" -tc='causal coopmat flash*' > "$FULL" 2>&1
echo "device test rc=$? $(grep -E '^\[doctest\] (test cases|assertions)' "$FULL" | tr '\n' ' ')" | tee -a "$LOG"
micro() { # label venv env; env is a placeholder word for env(1) when the arm sets nothing (X40=1 is deliberately unused)
  sleep "${GAP:-30}"
  echo "$1 L=512,1024,2048: $(env "$3" timeout 200 "$H/$2/bin/python" "$H/sdpa_micro.py" causal 512,1024,2048 2>&1 | tail -1 | cut -c1-900)" | tee -a "$LOG"
}
for rep in 1 2 3; do
  micro composed venv-cf X40=1
  micro flash-old venv-cf MLX_OMARCHY_SDPA_CAUSAL_FLASH=1
  micro flash-new venv-cf2 MLX_OMARCHY_SDPA_CAUSAL_FLASH=1
done
echo "end $(date -u +%FT%TZ)" | tee -a "$LOG"
