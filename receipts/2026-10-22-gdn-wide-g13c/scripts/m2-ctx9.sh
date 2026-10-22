#!/bin/bash
# m2-ctx9.sh OUT ROUNDS : decode speed of Qwen3.5-9B 4-bit on the 14-inch M2 Max against the prompt length, main stack (venv-main wheel a04423e, mlx-lm series main, driver vulkan-main)
# against the wheel of main at aaef3b6c (tree-main2, default environment: the wide GDN decode kernel). The step is the coreglass ledger LLM step (llmstep.py: PROMPT prompt tokens,
# 128 generated, one process per sample). Prompt lengths 477 and 2048; for each length ROUNDS rounds, both arms once per round, the order swapped every round.
# Prints every sample with its tokens digest. No arm changes a file; the new arm differs by the wheel on PYTHONPATH.
set -uo pipefail
O=${1:?out}; N=${2:-3}
B0=$HOME/.local/share/coreglass
STEP=$HOME/work/llmstep.py
MODEL=$B0/models/qwen3_5-9b-4bit
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "CTX9 $*" | tee -a "$L"; }
awk '{exit !($1<=1.3)}' /proc/loadavg || { say "ABORT load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 3; }
TREE=$HOME/work/tree-main2
ARMS=(
  "base|"
  "new|$TREE"
)
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) gov=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2>/dev/null)"
say "llmstep $(sha256sum "$STEP" | cut -c1-16)"
say "driver $(readlink "$B0/vulkan-main") model $(sha256sum "$MODEL/model.safetensors.index.json" | cut -c1-16) mx=$("$B0/venv-main/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx=$(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
say "new tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) mx=$(PYTHONPATH="$TREE" "$B0/venv-main/bin/python" -c 'import mlx.core as m; print(m.__version__)')"
for P in 477 2048; do
  for r in $(seq 1 "$N"); do
    for k in 0 1; do
      idx=$(( (k + r - 1) % 2 ))
      IFS='|' read -r name tree <<< "${ARMS[$idx]}"
      pp="$B0/mlx-lm-series-main/lib/python3.14/site-packages"
      [ -n "$tree" ] && pp="$tree:$pp"
      env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
        PYTHONPATH="$pp" \
        timeout 300 "$B0/venv-main/bin/python" "$STEP" "$MODEL" "$P" 128 > "$O/$name-$P-$r.out" 2> "$O/$name-$P-$r.err"
      rc=$?
      say "ctx $P round $r $name rc=$rc $(grep -a coreglass_result "$O/$name-$P-$r.out" | python3 -c "import sys,json;r=json.loads(sys.stdin.read())['coreglass_result'];print('prompt %d decode %.2f prefill %.1f ttft %.1f sha %s' % (r['prompt_tokens'], r['decode_tok_s'], r['prefill_tok_s'], r['ttft_ms'], r['tokens_sha']))" 2>/dev/null || echo NOPARSE) load1=$(cut -d' ' -f1 /proc/loadavg)"
    done
  done
done
say "end $(date -u +%FT%TZ)"
