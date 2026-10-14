#!/bin/bash
# Scoreboard v3 ticket body: runsb.sh <tsv> <tag> <python> <mlxlm-tree> <keys...>
# Weights of a repo that is NOT in ~/.cache/huggingface before the ticket are downloaded just in time and purged by score.sh (JIT mode, disk gate 80%/30 GB).
# A key whose repo was already cached runs with JIT off, so score.sh never deletes someone else's weights (the H402 mistake).
# Rows are timed; ANE overlap is recorded in each row's flags, not blocked.
S=$HOME/scratch/w71/score
H=$HOME/.cache/huggingface/hub
cd "$S" || exit 1
TSV=$1 TAG=$2 PYB=$3 TREE=$4
shift 4
L=$S/sb-$TAG.log
if [ -s "$L" ]; then mv -f "$L" "$L.old-$(date -u +%H%M%S)"; fi
ANE_B0=$(awk '/^busy_ns/{print $2}' /sys/class/accel/accel0/device/ane_stats 2> /dev/null); ANE_T0=$(date +%s.%N)
CPUF=/sys/devices/system/cpu/cpufreq
gov_orig=$(cat $CPUF/policy0/scaling_governor 2> /dev/null)
if [ -n "${SB_GOV:-}" ] && [ "$SB_GOV" != "$gov_orig" ]; then
  for p in $CPUF/policy*; do echo "$SB_GOV" | sudo -n tee "$p/scaling_governor" > /dev/null; done
  trap 'for p in $CPUF/policy*; do echo "$gov_orig" | sudo -n tee "$p/scaling_governor" > /dev/null; done' EXIT
fi
echo "SB-$TAG governor=$(cat $CPUF/policy0/scaling_governor 2> /dev/null) (found $gov_orig, SB_GOV=${SB_GOV:-unset})" >> "$L"
hw() { for z in /sys/class/hwmon/hwmon*; do [ "$(cat "$z/name" 2> /dev/null)" = macsmc_hwmon ] && echo "power_W=$(awk "{printf \"%.1f\", \$1/1000000}" "$z/power1_input" 2> /dev/null) fan1=$(cat "$z/fan1_input" 2> /dev/null) fan2=$(cat "$z/fan2_input" 2> /dev/null)"; done; }
echo "SB-$TAG mem_avail_mb=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) oom_kills=$(dmesg 2>/dev/null | grep -c "Out of memory") boot_id=$(cat /proc/sys/kernel/random/boot_id) gpu_timeouts=$(dmesg 2>/dev/null | grep -c "Job timed out")" >> "$L"
echo "SB-$TAG $(date -u +%T) start keys=$* tree=$TREE df=$(df -h /home | tail -1 | awk '{print $4" free "$5}') load1=$(cut -d' ' -f1 /proc/loadavg) hw: $(hw) anes=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)" >> "$L"
echo "SB-$TAG icd_env SB_DRIVER_FILES=${SB_DRIVER_FILES:-} SB_ICD=${SB_ICD:-} SB_SYSTEM=${SB_SYSTEM:-} HK_PERFTEST=${HK_PERFTEST:-} icd_lib_sha256=$( [ -n "${SB_DRIVER_FILES:-}" ] && sha256sum "$(dirname "$SB_DRIVER_FILES")/libvulkan_asahi.so" 2>/dev/null | cut -c1-24)" >> "$L"
if [ -n "${SB_SYSTEM:-}" ]; then echo "SB-$TAG system-vk: $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | tr -s " ") lib=$(sha256sum /usr/lib/libvulkan_asahi.so 2>/dev/null | cut -c1-12) pkg=$(pacman -Q vulkan-asahi-omarchy 2>/dev/null | tr " " "=")" >> "$L"; fi
[ -x "$S/llama-linux/bin/llama-bench" ] && export LLAMA_BENCH=$S/llama-linux/bin/llama-bench
export SCORE_TSV=$TSV SCORE_MLXLM=$TREE SCORE_CAP=1500 SCORE_PY=$PYB
cached=() fresh=()
for k in "$@"; do
  repo=$(awk -F'\t' -v k="$k" '$1==k {print $3}' "$S/$TSV")
  if [ -d "$H/models--${repo//\//--}" ]; then cached+=("$k"); else fresh+=("$k"); fi
done
echo "SB-$TAG cached(kept)=${cached[*]:-none} fresh(JIT purge)=${fresh[*]:-none}" >> "$L"
sleep 20
n0=$(wc -l < "$S/score.log" 2> /dev/null || echo 0)
if [ ${#fresh[@]} -gt 0 ]; then
  SCORE_JIT=1 SCORE_ONLY="${fresh[*]}" SCORE_TAG=-$TAG bash "$S/score.sh" run >> "$L" 2>&1
fi
if [ ${#cached[@]} -gt 0 ]; then
  SCORE_JIT=0 SCORE_ONLY="${cached[*]}" SCORE_TAG=-$TAG-c bash "$S/score.sh" run >> "$L" 2>&1
fi
for k in "$@"; do r=$(tail -n +$((n0 + 1)) "$S/score.log" 2> /dev/null | grep -h "^SCORE|$k|" | tail -1 | cut -c1-330); echo "SB-$TAG ${r:-NO-NEW-ROW (score.sh wrote no row for $k; stale score.log lines are never replayed)}" >> "$L"; done
mkdir -p "$S/keep-$TAG"; for k in "$@"; do for f in "llama-$k.json" "llama-$k.err" "out-$TAG/$k.json" "out-$TAG/$k.log" "out-$TAG-c/$k.json" "out-$TAG-c/$k.log"; do [ -f "$S/$f" ] && cp "$S/$f" "$S/keep-$TAG/$(echo "$f" | tr / _)" 2> /dev/null; done; done
echo "SB-$TAG mem_avail_mb=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) oom_kills=$(dmesg 2>/dev/null | grep -c "Out of memory") gpu_timeouts=$(dmesg 2>/dev/null | grep -c "Job timed out")" >> "$L"
echo "SB-$TAG $(date -u +%T) DONE df=$(df -h /home | tail -1 | awk '{print $4" free "$5}') anes_end=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true) ane_busy_pct=$(awk -v b0="${ANE_B0:-}" -v t0="$ANE_T0" -v t1="$(date +%s.%N)" '/^busy_ns/ && b0 != "" {printf "%.1f", ($2 - b0) / ((t1 - t0) * 1e7)}' /sys/class/accel/accel0/device/ane_stats 2> /dev/null || true)" >> "$L"
