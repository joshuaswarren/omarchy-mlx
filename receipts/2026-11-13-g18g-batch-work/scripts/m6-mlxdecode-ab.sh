#!/bin/bash
# usage: m6-mlxdecode-ab.sh 2b|4b   (one m6 queue ticket, about 16 min)
# Interleaved A/B of MLX_OMARCHY_BATCH_WORK on the M6 GPU: for the one model the legs run  default, 20000, 40000, default, 20000, 40000  (120 s each),
# so a clock or thermal drift hits both arms. libmlx has no calibrated work-group budget row for G18G, so the default is the 5000 fallback.
# Per leg: decode tok/s (m6-mlxdecode.py: 512-token prompt, 128 tokens per rep) and, concurrently, the issue #19 frame-pacing proxy
# (submit-latency: a 60 Hz client, one tiny submit per frame, host latency to fence; 90 s). Also one probe-alone baseline of 20 s.
# No logged-in compositor session is driven here: the probe is the stand-in, as in the omarchy-mlx submission-cap receipts.
set -u
K=${1:?2b|4b}
W=$HOME/scratch/w7g
V=$HOME/scratch/mlx-venv/venv
H=$HOME/.cache/huggingface/hub
export VK_DRIVER_FILES=$W/icd/asahi_icd.json
PAT='G18_APP_JOB_FAILED|G18_APP_FAIL|G18_RECOVERY|G18_JOB_WATCHDOG|G18_APP_STRANDED|submit failed|admission rejected'
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) ko=$(sha256sum "$(modinfo -F filename asahi)" | cut -c1-8) icd=$(sha256sum "$W/icd/libvulkan_asahi.so" | cut -c1-8) probe=$(sha256sum "$W/submit-latency" | cut -c1-8) $(date -u +%FT%TZ)"
test "$(hostname)" = m6 || exit 2
B0=$(sudo dmesg | grep -acE "$PAT")
mkdir -p "$W/m6-mlxdecode"
echo "== probe alone $(date -u +%T)"
"$W/submit-latency" 20 2>&1 | grep -a 'submit-latency:'
case "$K" in 2b) M1=models--SiddhJagani--Qwen3.8-2B-mlx-4Bit;; 4b) M1=models--mlx-community--Qwen3-4B-Instruct-2507-4bit;; *) exit 2;; esac
for m in $M1; do
  SNAP=$(echo "$H/$m"/snapshots/*)
  for arm in default 20000 40000 default 20000 40000; do
    echo "== $m work=$arm $(date -u +%T)"
    F=$W/m6-mlxdecode/flag-$$-$arm
    rm -f "$F"
    if [ "$arm" = default ]; then unset MLX_OMARCHY_BATCH_WORK; else export MLX_OMARCHY_BATCH_WORK=$arm; fi
    FLAGFILE=$F timeout 400 "$V/bin/python" "$W/m6-mlxdecode.py" "$SNAP" 120 "$W/m6-mlxdecode/ab-$m-$arm-$(date -u +%H%M%S).json" 2>&1 | grep -aE 'SUMMARY|Error|error' | tail -n 3 &
    PY=$!
    for _ in $(seq 1 120); do [ -f "$F" ] && break; sleep 1; done
    sleep 5
    echo "probe: $("$W/submit-latency" 90 2>&1 | grep -a 'submit-latency:')"
    wait $PY
    unset MLX_OMARCHY_BATCH_WORK
  done
done
B1=$(sudo dmesg | grep -acE "$PAT")
echo "fail_lines_new=$((B1 - B0))"
echo "end $(date -u +%T)"
