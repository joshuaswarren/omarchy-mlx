#!/bin/bash
# swscreen.sh OUT ROUNDS MODELDIR SET : decode speed of the model in MODELDIR on this Mac with the ledger stack (venv-main wheel, mlx-lm series main, driver vulkan-main),
# one arm per environment switch against the default (arm base), interleaved: every round runs every arm once, the order rotated per round.
# The step is the coreglass ledger LLM step (llmstep.py: 477 prompt tokens, 128 generated, one process per sample).
# Prints every sample with its tokens digest. No arm changes a file; each arm is one environment variable.
set -uo pipefail
O=${1:?out}; N=${2:-3}; MODEL=${3:?model dir}; SET=${4:?arm set wide|dense|tail}
B0=$HOME/.local/share/coreglass
STEP=$HOME/work/llmstep.py
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "ARMS $*" | tee -a "$L"; }
awk '{exit !($1<=1.3)}' /proc/loadavg || { say "ABORT load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 3; }
case "$SET" in
  wide) ARMS=(
    "base||"
    "tile1||MLX_OMARCHY_GDN_DECODE_TILE=1"
    "plain||MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_WIDE=0"
    "wide||MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_WIDE=1"
  ) ;;
  dense) ARMS=(
    "base||"
    "sdpabf16||MLX_OMARCHY_SDPA_BF16_FAST=1"
    "wave||MLX_OMARCHY_WAVE_SCHED=1"
    "defer||MLX_OMARCHY_DEFER_COMMIT=1"
    "qmvmulti||MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=1"
    "outfold1||MLX_OMARCHY_OUTGATE_FOLD=1"
    "nobarrier||MLX_OMARCHY_GATED_BARRIERS=0"
    "noword||MLX_OMARCHY_QMM_VEC_Q4_WORD=0"
    "sdpa0||MLX_OMARCHY_SDPA_DECODE_NATIVE=0"
    "argc||MLX_OMARCHY_ARGREDUCE_CHUNKED=1"
    "lsec||MLX_OMARCHY_LSE_CHUNKED=1"
  ) ;;
  tail) ARMS=(
    "base||"
    "argc||MLX_OMARCHY_ARGREDUCE_CHUNKED=1"
    "lsec||MLX_OMARCHY_LSE_CHUNKED=1"
    "both||MLX_OMARCHY_ARGREDUCE_CHUNKED=1 MLX_OMARCHY_LSE_CHUNKED=1"
  ) ;;
  sdpa) ARMS=(
    "base|$HOME/work/tree-sdpa|MLX_OMARCHY_ARGREDUCE_CHUNKED=0 MLX_OMARCHY_LSE_CHUNKED=0"
    "split|$HOME/work/tree-sdpa|MLX_OMARCHY_ARGREDUCE_CHUNKED=0 MLX_OMARCHY_LSE_CHUNKED=0 MLX_OMARCHY_SDPA_DECODE_SPLIT=1"
    "ledger||"
  ) ;;
  *) echo "ABORT unknown arm set $SET"; exit 2 ;;
esac
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) gov=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane/parameters/call_settle_us /sys/module/ane_t6021/parameters/call_settle_us 2>/dev/null | head -1) set $SET"
say "llmstep $(sha256sum "$STEP" | cut -c1-16)"
say "driver $(readlink "$B0/vulkan-main") model $(sha256sum "$MODEL/model.safetensors.index.json" | cut -c1-16) mx=$("$B0/venv-main/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx=$(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
n=${#ARMS[@]}
for r in $(seq 1 "$N"); do
  for k in $(seq 0 $((n - 1))); do
    idx=$(( (k + (r - 1) * 1) % n ))
    IFS='|' read -r name tree extra <<< "${ARMS[$idx]}"
    pp="$B0/mlx-lm-series-main/lib/python3.14/site-packages"
    [ -n "$tree" ] && pp="$tree:$pp"
    # shellcheck disable=SC2086
    env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
      PYTHONPATH="$pp" $extra \
      timeout 400 "$B0/venv-main/bin/python" "$STEP" "$MODEL" 477 128 > "$O/$name-$r.out" 2> "$O/$name-$r.err"
    rc=$?
    say "round $r $name rc=$rc $(grep -a coreglass_result "$O/$name-$r.out" | python3 -c "import sys,json;r=json.loads(sys.stdin.read())['coreglass_result'];print('decode %.2f prefill %.1f ttft %.1f sha %s' % (r['decode_tok_s'], r['prefill_tok_s'], r['ttft_ms'], r['tokens_sha']))" 2>/dev/null || echo NOPARSE) load1=$(cut -d' ' -f1 /proc/loadavg)"
  done
done
say "end $(date -u +%FT%TZ)"
