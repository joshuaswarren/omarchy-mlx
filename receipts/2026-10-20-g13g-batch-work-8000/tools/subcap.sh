#!/bin/bash
# subcap.sh LABEL MODEL_SUBDIR : submission-size calibration on the 13-inch M1 with the diagnostics wheel (3b24ad4 + profiling), budgets 5000..40000.
# scripts/subcap_calibrate.py of 3b24ad416 drives one process per budget and prints p50/p95/p99/max ms per submission and decode tok/s.
set -u
B=$HOME/.local/share/coreglass
D=$WORK
for _ in $(seq 1 24); do [ "$(cut -d. -f1 /proc/loadavg)" -le 1 ] && break; sleep 20; done
echo "SUBCAP start $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) label $1 model $2"
env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES=$B/vulkan-release/honeykrisp_icd.aarch64.json \
  timeout 1500 "$B/venv-diag/bin/python" "$D/subcap_calibrate.py" --model "$B/models/$2" --tokens 64 --prefill-tokens 512 \
  --budgets 5000,8000,10000,12000,15000,20000,40000 --profile "$D/subcap-$1.jsonl" --venv-python "$B/venv-diag/bin/python" > "$D/subcap-$1.out" 2>&1
echo "SUBCAP rc=$? end $(date -u +%T)"
tail -40 "$D/subcap-$1.out" | cut -c1-200
