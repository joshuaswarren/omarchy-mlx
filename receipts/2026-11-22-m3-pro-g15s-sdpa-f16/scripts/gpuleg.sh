#!/usr/bin/env bash
# gpuleg.sh NAME TIMEOUT_S CMD... — one bounded GPU leg under /tmp/m1-gpu.lock.
# Writes logs/NAME.log (command output) and evidence/NAME.state (before/after device state).
set -uo pipefail
LANE=${LANE}
name=$1; to=$2; shift 2
log=$LANE/logs/$name.log; st=$LANE/evidence/$name.state
unset MLX_OMARCHY_ALLOW_NON_APPLE MLX_OMARCHY_CAPS_SIM VK_DRIVER_FILES VK_ICD_FILENAMES
snap() {
  echo "time $(date -u +%FT%TZ)"
  echo "loadavg $(cat /proc/loadavg)"
  free -m | awk 'NR==2{print "mem_used_mb "$3" mem_avail_mb "$7} NR==3{print "swap_used_mb "$3}'
  for f in /sys/class/hwmon/hwmon*/temp*_input; do
    [ -r "$f" ] || continue
    lbl_f="${f%_input}_label"
    if [ -r "$lbl_f" ]; then
      lbl=$(cat "$lbl_f" | tr ' ' _)
    else
      lbl=$(basename "$(dirname "$f")")_$(basename "$f" _input)
    fi
    echo "temp $lbl $(( $(cat "$f") / 1000 ))C"
  done | head -n 16
  echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "ac_online $(cat /sys/class/power_supply/macsmc-ac/online 2>/dev/null || echo unknown)"
  echo "gpu_kernel_lines_this_boot $(journalctl -k -b --no-pager 2>/dev/null | grep -cE 'asahi-g15|292400000.gpu')"
  timeout --foreground 30 vulkaninfo --summary < /dev/null 2>/dev/null | grep -E '^\s+(deviceName|driverID|driverInfo|apiVersion|conformanceVersion)' | sed 's/^\s*/vk /'
}
exec 9>/tmp/m1-gpu.lock
if ! flock -w 5400 -x 9; then echo "lock busy" >&2; exit 97; fi
# Timed legs (WAIT_LOAD=1): wait inside the lock until the 1-minute load is under 1.0 (up to 900 s); a leg that never gets there is marked VOID.
load_note=""
if [ "${WAIT_LOAD:-0}" = 1 ]; then
  for _ in $(seq 180); do awk '{exit !($1 < 1.0)}' /proc/loadavg && break; sleep 5; done
  l1=$(cut -d' ' -f1 /proc/loadavg); load_note="load1_at_start $l1"
  awk '{exit !($1 < 1.0)}' /proc/loadavg || load_note="$load_note VOID(load>=1.0)"
fi
since=$(date '+%Y-%m-%d %H:%M:%S')
{ echo "== before"; snap; echo "cmd $*"; echo "timeout_s $to"; [ -n "$load_note" ] && echo "$load_note"; } > "$st"
start=$(date +%s.%N)
timeout --foreground --kill-after=30s "$to" "$@" < /dev/null > "$log" 2>&1
rc=$?
end=$(date +%s.%N)
{ echo "== after"; echo "exit $rc"; echo "wall_s $(awk -v a=$start -v b=$end 'BEGIN{printf "%.2f", b-a}')"; snap;
  echo "== new GPU kernel lines"; journalctl -k --since "$since" --no-pager 2>/dev/null | grep -E 'asahi|292400000.gpu|drm' || echo "(none)"; } >> "$st"
echo "$name exit=$rc ${load_note} wall=$(awk -v a=$start -v b=$end 'BEGIN{printf "%.2f", b-a}')s"
exit $rc
