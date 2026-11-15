#!/bin/bash
# usage: m6-barcount.sh 2b|4b|4bx|both|smoke [PROCS]   (m6, one queue ticket)
# the barcount tracer (barcount.py) (libmlx trace counters: barriers_emitted/skipped, dispatches, submits per decode step) on the M6 GPU, default env.
# 2b/4b: stock ICD. 4bx: Qwen3 4B once per ICD build (stock, a5, pr251-head), PROCS processes each, to see whether the graph depends on the Mesa build.
# Same venv build and models as the M2 rows. Stops starting new processes after 21 min so the 25 min ticket cap holds.
set -u
K=${1:?2b|4b|4bx|smoke}; N=${2:-5}
W=$HOME/scratch/w7g
V=$HOME/scratch/mlx-venv/venv
Q=$HOME/scratch/quiet
H=$HOME/.cache/huggingface/hub
PAT='G18_APP_JOB_FAILED|G18_APP_FAIL|G18_RECOVERY|G18_JOB_WATCHDOG|G18_APP_STRANDED|submit failed|admission rejected'
echo "ENV MLX_OMARCHY_BATCH_WORK=${MLX_OMARCHY_BATCH_WORK:-unset}"
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) ko=$(sha256sum "$(modinfo -F filename asahi)" | cut -c1-8) $(date -u +%FT%TZ)"
test "$(hostname)" = m6 || exit 2
B0=$(sudo dmesg | grep -acE "$PAT")
OUT=$W/m6-barcount-$K; mkdir -p "$OUT"
END=$((SECONDS + 1260))
run_model() { # model-dir-name tag icd-dir
  local SNAP; SNAP=$(echo "$H/$1"/snapshots/*)
  export VK_DRIVER_FILES=$W/$3/asahi_icd.json
  echo "-- icd $3 sha=$(sha256sum "$W/$3/libvulkan_asahi.so" | cut -c1-8)"
  for i in $(seq 1 "$N"); do
    [ $SECONDS -lt $END ] || { echo "deadline: stopped before $2 p$i"; return; }
    echo "== $2 p$i $(date -u +%T)"
    timeout 600 "$V/bin/python" "$Q/barcount.py" "$SNAP" "$OUT/$2-p$i.json" 2>&1 | tail -n 1
    echo "rc=${PIPESTATUS[0]}"
    sleep 30
  done
}
case "$K" in
  smoke)
    export VK_DRIVER_FILES=$W/icd/asahi_icd.json
    timeout 300 "$V/bin/python" -c 'import mlx.core as mx; print(mx.__version__, mx.default_device()); a=mx.ones((1024,1024)); b=(a@a).sum(); mx.eval(b); print(float(b))'
    echo "rc=$?";;
  2b) run_model models--SiddhJagani--Qwen3.8-2B-mlx-4Bit 2b icd;;
  4b) run_model models--mlx-community--Qwen3-4B-Instruct-2507-4bit 4b icd;;
  4bx)
    for d in icd icd-fix3f-a5 icd-pr251-head; do
      run_model models--mlx-community--Qwen3-4B-Instruct-2507-4bit "4b-$d" "$d"
    done;;
  both) run_model models--SiddhJagani--Qwen3.8-2B-mlx-4Bit 2b icd; run_model models--mlx-community--Qwen3-4B-Instruct-2507-4bit 4b icd;;
  *) exit 2;;
esac
B1=$(sudo dmesg | grep -acE "$PAT")
echo "fail_lines_new=$((B1 - B0))"
echo "end $(date -u +%T)"
