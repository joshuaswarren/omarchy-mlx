#!/bin/bash
# Numerics gate for the decode PR on the 14-inch M2 Max, Qwen3.8-2B 4-bit, PR wheel 7a09bac in venv-pr.
# S=1 teacher-forced capture (tfref.py) of 4 text segments of 160 tokens on the GPU with the tiled kernel (old default),
# the new default, the new default again (noise floor), and the wide kernel, then the same capture on the CPU in float32
# as the independent reference. tfcmp.py prints agreement, NLL and log-prob error of each GPU arm against the CPU reference.
# A CPU smoke test of 8 tokens runs first so a CPU path failure shows within a minute. No file outside the output directory changes.
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
BUDGET=${2:-1200}
vp=$D/venv-pr
M=$HOME/.local/share/coreglass/models/qwen3_5-4bit
T=$D/tfref-text.txt
[ -e "$O" ] && { echo "GATE abort $O exists"; exit 1; }
mkdir -p "$O"
export HF_HUB_OFFLINE=1
say() { echo "GATE $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg) cpu budget ${BUDGET}s"
say "venv mx $("$vp/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx $(sha256sum "$vp"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) tfref $(sha256sum "$D/tfref.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) text $(sha256sum "$T" | cut -c1-16)"
run() { # name device dtype budget env...
  local name=$1 dev=$2 dty=$3 bud=$4
  shift 4
  env "$@" timeout 3000 "$vp/bin/python" "$D/tfref.py" --model "$M" --text "$T" --device "$dev" --dtype "$dty" --nseg 4 --maxtok 160 --budget "$bud" --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFREF done' "$O/$name.out" | cut -c1-80) segments $(ls "$O/$name" 2> /dev/null | wc -l)"
}
env HF_HUB_OFFLINE=1 timeout 300 "$vp/bin/python" "$D/tfref.py" --model "$M" --text "$T" --device cpu --dtype float32 --nseg 1 --maxtok 8 --out "$O/smoke" > "$O/smoke.out" 2> "$O/smoke.err"
say "cpu smoke rc=$? $(grep -a 'TFREF seg' "$O/smoke.out" | cut -c1-90) $(tail -n 1 "$O/smoke.err" | cut -c1-160)"
run tile1 gpu native 1e9 MLX_OMARCHY_GDN_DECODE_TILE=1
run plain gpu native 1e9 MLX_AB_UNUSED=1
run plain2 gpu native 1e9 MLX_AB_UNUSED=1
run wide gpu native 1e9 MLX_OMARCHY_GDN_PF_WIDE=1
run cpu cpu float32 "$BUDGET" MLX_AB_UNUSED=1
python3 "$D/tfcmp.py" "$O/cpu" tile1="$O/tile1" plain="$O/plain" plain2="$O/plain2" wide="$O/wide" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
