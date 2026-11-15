#!/bin/bash
# usage: m6-mlxdecode-wk.sh 2b|4b SECONDS_PER_LEG   (one m6 queue ticket, about 17-20 min)
# MLX decode tok/s on the M6 GPU with NO frame probe running: MLX_OMARCHY_BATCH_WORK default (5000 fallback), 20000, 40000; legs interleaved
# default,20000,40000 x3. Each rep prints a hash of its 128 greedy token ids: equal hashes across budgets = same output.
# 512-token prompt, 128 tokens per rep, stock ICD ee5fbac9, same venv build as the M2 rows.
set -u
K=${1:?2b|4b}; S=${2:-70}
W=$HOME/scratch/w7g
V=$HOME/scratch/mlx-venv/venv
H=$HOME/.cache/huggingface/hub
export VK_DRIVER_FILES=$W/icd/asahi_icd.json
case "$K" in 2b) M=models--SiddhJagani--Qwen3.8-2B-mlx-4Bit;; 4b) M=models--mlx-community--Qwen3-4B-Instruct-2507-4bit;; *) exit 2;; esac
SNAP=$(echo "$H/$M"/snapshots/*)
PAT='G18_APP_JOB_FAILED|G18_APP_FAIL|G18_RECOVERY|G18_JOB_WATCHDOG|G18_APP_STRANDED|submit failed|admission rejected'
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) ko=$(sha256sum "$(modinfo -F filename asahi)" | cut -c1-8) icd=$(sha256sum "$W/icd/libvulkan_asahi.so" | cut -c1-8) $(date -u +%FT%TZ)"
test "$(hostname)" = m6 || exit 2
B0=$(sudo dmesg | grep -acE "$PAT")
mkdir -p "$W/m6-mlxdecode"
END=$((SECONDS + 1320))
for r in 1 2 3; do
  for arm in default 20000 40000; do
    [ $SECONDS -lt $END ] || { echo "deadline: stopped before r$r $arm"; break 2; }
    echo "== $K r$r work=$arm $(date -u +%T)"
    if [ "$arm" = default ]; then unset MLX_OMARCHY_BATCH_WORK; else export MLX_OMARCHY_BATCH_WORK=$arm; fi
    timeout 400 "$V/bin/python" "$W/m6-mlxdecode2.py" "$SNAP" "$S" "$W/m6-mlxdecode/wk-$K-$arm-r$r-$(date -u +%H%M%S).json" 2>&1 | grep -aE '^rep|SUMMARY|Error|error|Traceback' | tail -n 6
    echo "rc=${PIPESTATUS[0]}"
    unset MLX_OMARCHY_BATCH_WORK
  done
done
B1=$(sudo dmesg | grep -acE "$PAT")
echo "fail_lines_new=$((B1 - B0))"
echo "end $(date -u +%T)"
