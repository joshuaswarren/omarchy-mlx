#!/bin/bash
# idlepow.sh: what does the performance governor cost at IDLE? Four windows of WIN seconds with nothing running, governors flipped between them.
# Order P S S P (the two windows of a governor bracket the other, so a slow drift shows). The first SKIP seconds of every window are discarded (the flip
# transient and the previous state). A 1 Hz sampler (power, clocks, fan, temperatures) runs for the whole block; idlepow_analyze.py reads it.
# The start governors are restored on any exit and verified. Needs sudo -n for the governor write, as runsb.sh does.
# env: CPUF GOV_SUDO GOV_SAMP WIN(180) SKIP(40) GOV_MAXLOAD(0.5) GOV_LOADWAIT(300) STOPFILE SERVE_PAT (the serving-process pattern; a test overrides it)
set -u
CPUF=${GOV_CPUF:-/sys/devices/system/cpu/cpufreq}
SUDO=${GOV_SUDO:-sudo -n}
WIN=${WIN:-180}
SKIP=${SKIP:-40}
STOPFILE=${STOPFILE:-$HOME/.local/state/idle-guard/ane-stop}
HERE=$(cd "$(dirname "$0")" && pwd)
T=idlepow-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/out-$T
mkdir -p "$OUT"
L=$OUT/block.log
SAMP=${GOV_SAMP:-$WORK/sampler/row-sampler.sh}
CSV=${GOV_CSV:-$WORK/sampler/rows-$T.csv}
say() { echo "IDLEPOW $T $(date -u +%T) $*" | tee -a "$L"; }
die() { say "ABORT: $*"; exit 1; }
declare -A orig
for p in "$CPUF"/policy*; do orig[$p]=$(cat "$p/scaling_governor"); done
restore() {
	for p in "${!orig[@]}"; do echo "${orig[$p]}" | $SUDO tee "$p/scaling_governor" > /dev/null; done
	say "restored governors: $(for p in "${!orig[@]}"; do printf '%s=%s ' "$(basename "$p")" "$(cat "$p/scaling_governor")"; done)"
	for p in "${!orig[@]}"; do [ "$(cat "$p/scaling_governor")" = "${orig[$p]}" ] || say "RESTORE MISMATCH: $(basename "$p") is $(cat "$p/scaling_governor"), started as ${orig[$p]}"; done
	[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" > /dev/null 2>&1
}
trap restore EXIT
trap 'exit 130' INT TERM
gov_set() {
	for p in "$CPUF"/policy*; do echo "$1" | $SUDO tee "$p/scaling_governor" > /dev/null || die "cannot write governor $1"; done
	for p in "$CPUF"/policy*; do [ "$(cat "$p/scaling_governor")" = "$1" ] || die "governor read-back of $(basename "$p") is not $1"; done
}
nane=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)
[ "${nane:-0}" -eq 0 ] || die "$nane ANE process(es) running"
[ -e "$STOPFILE" ] || die "the ANE filler stop file $STOPFILE is absent: the filler could start during the block"
SERVE_PAT=${SERVE_PAT:-"[l]itellm|[l]lama-server|[m]lx_omarchy_serve|[m]lx_lm.server|[v]llm"}
busy=$(pgrep -fc "$SERVE_PAT" || true)
[ "${busy:-0}" -eq 0 ] || die "$busy serving process(es) running"
LOADWAIT=${GOV_LOADWAIT:-300}
waited=0
while :; do
	load=$(cut -d' ' -f1 /proc/loadavg)
	awk -v l="$load" -v m="${GOV_MAXLOAD:-0.5}" 'BEGIN{exit !(l < m)}' && break
	[ "$waited" -lt "$LOADWAIT" ] || die "load1 $load is not under ${GOV_MAXLOAD:-0.5} after waiting ${waited}s"
	sleep 10
	waited=$((waited + 10))
done
[ -f "$HERE/idlepow_analyze.py" ] || die "missing idlepow_analyze.py next to the script"
[ -f "$SAMP" ] || die "no sampler $SAMP"
say "start boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) load1 $load window ${WIN}s skip ${SKIP}s ac_online=$(cat /sys/class/power_supply/macsmc-ac/online 2> /dev/null) sessions=$(loginctl list-sessions --no-legend 2> /dev/null | wc -l)"
say "found governors: $(for p in "${!orig[@]}"; do printf '%s=%s ' "$(basename "$p")" "${orig[$p]}"; done)"
say "sha256 prefixes: script $(sha256sum "$0" | cut -c1-16) analyze $(sha256sum "$HERE/idlepow_analyze.py" | cut -c1-16) sampler $(sha256sum "$SAMP" | cut -c1-16)"
bash "$SAMP" start "$CSV" >> "$L" 2>&1
say "sampler csv $CSV"
i=0
for a in P S S P; do
	i=$((i + 1))
	case $a in S) gov_set schedutil ;; P) gov_set performance ;; esac
	t0=$(date +%s)
	say "window $i arm $a governors=$(cat "$CPUF"/policy*/scaling_governor | sort -u | tr '\n' ',') start_epoch=$t0"
	sleep "$WIN"
	say "window $i arm $a end_epoch=$(date +%s) load1=$(cut -d' ' -f1 /proc/loadavg)"
done
bash "$SAMP" stop "$CSV" >> "$L" 2>&1
python3 "$HERE/idlepow_analyze.py" "$CSV" "$L" "$SKIP" 2>&1 | tee -a "$L"
say "DONE"
