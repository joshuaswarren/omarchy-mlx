#!/bin/bash
# Greedy generation of the ledger-step prompt (PTOK prompt tokens, 128 generated) on this Mac with the split-K final tree: arm default (flags unset), arm split0 (MLX_OMARCHY_SDPA_DECODE_SPLIT=0), arm ref32 (float32 model, SPLIT=0),
# token ids and top-1/top-2 gaps saved per arm; gencmp.py reports the first divergence. Untimed. Writes only OUT.   usage: gen-div.sh OUT MODELDIR PTOK
set -uo pipefail
O=${1:?out}; M=${2:?model dir}; PT=${3:?prompt tokens}
D=$HOME/work; B0=$HOME/.local/share/coreglass; TREE=$D/tree-final
[ -e "$O" ] && { echo "GDIV abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "GDIV $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) prompt $PT"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) gen_tokens $(sha256sum "$D/gen_tokens.py" | cut -c1-16) gencmp $(sha256sum "$D/gencmp.py" | cut -c1-16) driver $(readlink "$B0/vulkan-main")"
run() { # name env dtype
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
    PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" $2 \
    timeout 1500 "$B0/venv-main/bin/python" "$D/gen_tokens.py" "$M" "$PT" 128 "$O/$1.json" $3 > "$O/$1.out" 2> "$O/$1.err"
  say "arm $1 rc=$? $(grep -a 'GEN done' "$O/$1.out")"
}
run default "MLX_AB_UNUSED=1" ""
run split0 "MLX_OMARCHY_SDPA_DECODE_SPLIT=0" ""
run ref32 "MLX_OMARCHY_SDPA_DECODE_SPLIT=0" "float32"
"$B0/venv-main/bin/python" "$D/gencmp.py" "$O/default.json" "$O/split0.json" "$O/ref32.json" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
