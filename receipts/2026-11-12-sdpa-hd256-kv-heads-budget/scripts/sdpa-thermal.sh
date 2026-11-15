#!/bin/bash
# Identity and thermal snapshot: run before and after each row. Values are recorded, not gated.
echo "time_utc $(date -u +%FT%TZ)"
echo "kernel $(uname -r)"
echo "boot_id $(cat /proc/sys/kernel/random/boot_id)"
echo "dt_model $(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
echo "dt_compatible $(tr '\0' ' ' < /proc/device-tree/compatible 2>/dev/null)"
for h in /sys/class/hwmon/hwmon*; do
  n=$(cat "$h/name" 2>/dev/null)
  for t in "$h"/temp*_input; do
    [ -r "$t" ] && echo "temp $n $(basename "$t") $(cat "$t")"
  done
done
for p in /sys/class/power_supply/*; do
  [ -r "$p/online" ] && echo "power $(basename "$p") online=$(cat "$p/online")"
done
echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "swap_used_kb $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)"
