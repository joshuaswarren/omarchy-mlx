#!/bin/bash
# Qwen3.8-27B-4bit NLL accept-not-worse on the 16-inch M1 Max. See PREREG-nll27.md.
# usage: run-nll27.sh D0|U0|D1|U1|D2 OUTDIR   (OUTDIR must not exist)
set -uo pipefail
PART=${1:?D0|U0|D1|U1|D2}
O=${2:?output dir}
N=$HOME/scratch/w7b-gqmm/nll35
B0=$HOME/scratch/w7q/step1-49e6b3c1b
VP=$B0/venv
MODEL=mlx-community/Qwen3.8-27B-4bit
case $PART in
  D0) arm=D; first=0; count=64 ;;
  U0) arm=U; first=0; count=64 ;;
  D1) arm=D; first=64; count=64 ;;
  U1) arm=U; first=64; count=64 ;;
  D2) arm=D; first=0; count=32 ;;
  *) echo "bad part"; exit 2 ;;
esac
[ ! -e "$O" ] || { echo "ABORT: $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "NLL27 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) part=$PART arm=$arm first=$first count=$count kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "nll_pair $(sha256sum "$N/nll_pair.py" | cut -c1-16) nll_cmp $(sha256sum "$N/nll_cmp.py" | cut -c1-16) text $(sha256sum "$N/text-AB-e7c41800.txt" | cut -c1-16)"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
case $arm in D) envv=(MLX_AB_UNUSED=1) ;; U) envv=(MLX_OMARCHY_GDN_DECODE_TILE=0) ;; esac
say "== arm $arm env ${envv[*]} $(date -u +%T)"
env "${envv[@]}" timeout 1700 "$VP/bin/python" "$N/nll_pair.py" "$MODEL" "$N/text-AB-e7c41800.txt" "$O/$PART.npz" --prefixes $count --first $first \
  --expect-libmlx 960f0f358a2aba25 > "$O/$PART.out" 2> "$O/$PART.err"
say "part $PART rc=$? $(grep -a 'NLL mean' "$O/$PART.out" | cut -c1-200)"
say "end $(date -u +%FT%TZ)"
