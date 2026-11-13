#!/usr/bin/env bash
# M2 Linux (G14C): per-submission time on Qwen3-4B decode with the diagnostics wheel of #422 head 213366c9a (GPU profiling harness on),
# scripts/subcap_calibrate.py, budgets 40000, 50000, 60000, 80000, 120000, 64 decode tokens after a 512-token prefill, one process per budget.
# Ryan's rule (omarchy-mlx #407): p50 <= 6 ms, p99 <= about 40 ms, no sample at 78 ms.
#   w9c-linux-subcap.sh OUTDIR
set -uo pipefail
test "$(hostname)" = jwm2-linux
test ! -f /var/tmp/ane-hold-m2
OUT=$1
Q=$HOME/scratch/w9c-quietrows
H=$HOME/.cache/huggingface/hub
V=$HOME/scratch/w9c-g14cdiag-venv/venv
export VK_DRIVER_FILES=$HOME/scratch/w9c-mesa-main/asahi_icd.main.json
mkdir -p "$OUT"
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) load=$(cut -d' ' -f1 /proc/loadavg)" | tee "$OUT/run.log"
SNAP=$(echo "$H"/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*)
timeout 1500 "$V/bin/python" "$Q/subcap_calibrate.py" --model "$SNAP" --tokens 64 --prefill-tokens 512 \
  --budgets 40000,50000,60000,80000,120000 --profile "$OUT/subcap.jsonl" --venv-python "$V/bin/python" > "$OUT/subcap.out" 2>&1
echo "SUBCAP rc=$?" | tee -a "$OUT/run.log"
tail -n 30 "$OUT/subcap.out" | cut -c1-200 | tee -a "$OUT/run.log"
echo SUBCAP_DONE | tee -a "$OUT/run.log"
