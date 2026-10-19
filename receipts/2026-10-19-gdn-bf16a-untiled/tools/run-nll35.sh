#!/bin/bash
# Qwen3.6-35B-A3B NLL accept-not-worse: arms D (default), U (MLX_OMARCHY_GDN_DECODE_TILE=0), D2 (default again). See PREREG.md.
set -uo pipefail
D=$HOME/scratch/w7b-gqmm/nll35
B0=$HOME/scratch/w7q/step1-49e6b3c1b
VP=$B0/venv
O=$D/out-1
MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "NLL35 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "nll_pair $(sha256sum "$D/nll_pair.py" | cut -c1-16) nll_cmp $(sha256sum "$D/nll_cmp.py" | cut -c1-16) text $(sha256sum "$D/text-AB-e7c41800.txt" | cut -c1-16)"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
for arm in D U D2; do
  case $arm in D|D2) envv=(MLX_AB_UNUSED=1) ;; U) envv=(MLX_OMARCHY_GDN_DECODE_TILE=0) ;; esac
  say "== arm $arm env ${envv[*]} $(date -u +%T)"
  env "${envv[@]}" timeout 3000 "$VP/bin/python" "$D/nll_pair.py" "$MODEL" "$D/text-AB-e7c41800.txt" "$O/$arm.npz" --prefixes 128 --first 0 \
    --expect-libmlx 960f0f358a2aba25 > "$O/$arm.out" 2> "$O/$arm.err"
  say "arm $arm rc=$? $(grep -a 'NLL mean' "$O/$arm.out" | cut -c1-200)"
done
say "compare U minus D"
python3 "$D/nll_cmp.py" "$O/U.npz" "$O/D.npz" --label "U minus D" 2>&1 | tee -a "$L"
say "compare D2 minus D (determinism)"
python3 "$D/nll_cmp.py" "$O/D2.npz" "$O/D.npz" --label "D2 minus D" 2>&1 | tee -a "$L"
say "end $(date -u +%FT%TZ)"
