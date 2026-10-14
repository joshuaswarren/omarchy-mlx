#!/bin/bash
# row-sampler.sh start|stop OUTFILE : 1 Hz CSV of every power, thermal, clock and ANE counter readable on the M1-class laptops (read-only sysfs).
# start: runs in the background (setsid), writes OUTFILE and OUTFILE.pid; stop: kills it. One row per second, epoch seconds in column 1.
# Columns: epoch, sys_power_uW, ac_in_uW, rail38_uW, heatpipe_uW, fan1_rpm, fan2_rpm, nand_mC, batt_hotspot_mC, charger_mC, wifi_mC,
#          cpu_khz_<policy>... (one per cpufreq policy), loadavg1, ane_busy_ns, ane_jobs.
# No die or per-block power sensors exist on this kernel's SMC hwmon (checked on the 16-inch M1 Max: no debugfs keys); heatpipe power is the closest
# package-power proxy and the fans and the board temperatures are the thermal proxies.
set -u
cmd=${1:?start|stop}; out=${2:?outfile}
case "$cmd" in
stop) [ -s "$out.pid" ] && kill "$(cat "$out.pid")" 2> /dev/null; rm -f "$out.pid"; echo "stopped $out ($(($(wc -l < "$out") - 1)) rows)"; exit 0 ;;
start) ;;
*) echo "usage"; exit 2 ;;
esac
HW=$(grep -l macsmc_hwmon /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
STATS=/sys/class/accel/accel0/device/ane_stats
POL=$(ls -d /sys/devices/system/cpu/cpufreq/policy*)
hdr="epoch,sys_power_uW,ac_in_uW,rail38_uW,heatpipe_uW,fan1_rpm,fan2_rpm,nand_mC,batt_hotspot_mC,charger_mC,wifi_mC"
for p in $POL; do hdr="$hdr,khz_$(basename "$p")"; done
echo "$hdr,loadavg1,ane_busy_ns,ane_jobs" > "$out"
body() {
	while [ "$(date +%s)" -lt "$END" ]; do
		line=$(date +%s.%N | cut -c1-14)
		for f in power1_input power2_input power3_input power4_input fan1_input fan2_input temp1_input temp2_input temp3_input temp4_input; do
			line="$line,$(cat "$HW/$f" 2> /dev/null)"
		done
		for p in $POL; do line="$line,$(cat "$p/scaling_cur_freq" 2> /dev/null)"; done
		line="$line,$(cut -d' ' -f1 /proc/loadavg),$(awk '/^busy_ns/{b=$2}/^jobs/{j=$2}END{printf "%s,%s", b, j}' "$STATS" 2> /dev/null)"
		echo "$line" >> "$out"
		sleep 1
	done
}
export -f body 2> /dev/null
END=$(($(date +%s) + ${MAXS:-3600}))
setsid bash -c 'for f in /proc/self/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2> /dev/null && eval "exec $n>&-"; done; '"$(declare -f body); HW=$HW; STATS=$STATS; POL='$POL'; out=$out; END=$END; body" > /dev/null 2>&1 < /dev/null &
echo $! > "$out.pid"
echo "started $out pid $(cat "$out.pid")"
