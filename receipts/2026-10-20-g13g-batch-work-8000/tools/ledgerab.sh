#!/bin/bash
# ledgerab.sh OUT ARM_A ARM_B ROUNDS : paired A/B of the coreglass ledger LLM step (llmstep.py, Qwen3.8-2B 4-bit, 477 prompt tokens, 128 generated)
# on the 13-inch M1. An ARM is NAME,VENV,SERIES,ENV... where VENV is a venv dir under ~/.local/share/coreglass, SERIES the mlx-lm series dir
# (or - for none) and ENV zero or more NAME=VALUE words. Fresh process per sample, arms alternated, one process warms up inside llmstep.
# Same driver as the ledger's release stack (vulkan-release). The ledger reports the median of 3; this prints every sample.
set -uo pipefail
O=${1:?out}; A=${2:?arm A}; B=${3:?arm B}; N=${4:-5}
B0=$HOME/.local/share/coreglass
STEP=$B0/ledger/llmstep.py
MODEL=$B0/models/${MODEL_SUBDIR:-qwen3_5-4bit}
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "LAB $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "driver $(readlink "$B0/vulkan-release") libvulkan $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16) model $(sha256sum "$MODEL/model.safetensors" | cut -c1-16)"
IFS=, read -r NA VA SA EA <<< "$A"; IFS=, read -r NB VB SB EB <<< "$B"
for v in "$VA" "$VB"; do say "venv $v mx=$("$B0/$v/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx=$(sha256sum "$B0/$v"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"; done
run() {
  local name=$1 venv=$2 series=$3 extra=$4 i=$5
  local pp=(); [ "$series" != - ] && pp=("PYTHONPATH=$B0/$series/lib/python3.14/site-packages")
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-release/honeykrisp_icd.aarch64.json" "${pp[@]}" $extra \
    timeout 600 "$B0/$venv/bin/python" "$STEP" "$MODEL" 477 128 > "$O/$name-$i.out" 2> "$O/$name-$i.err"
  local rc=$?
  say "sample $i $name rc=$rc $(grep -a coreglass_result "$O/$name-$i.out" | python3 -c "import sys,json;r=json.loads(sys.stdin.read())['coreglass_result'];print('decode %.1f prefill %.1f ttft %.1f sha %s' % (r['decode_tok_s'], r['prefill_tok_s'], r['ttft_ms'], r['tokens_sha']))" 2>/dev/null) load1=$(cut -d' ' -f1 /proc/loadavg)"
}
for i in $(seq 1 "$N"); do
  if [ $((i % 2)) = 1 ]; then
    run "$NA" "$VA" "$SA" "${EA//;/ }" "$i"; run "$NB" "$VB" "$SB" "${EB//;/ }" "$i"
  else
    run "$NB" "$VB" "$SB" "${EB//;/ }" "$i"; run "$NA" "$VA" "$SA" "${EA//;/ }" "$i"
  fi
done
say "end $(date -u +%FT%TZ)"
