#!/bin/bash
# clampab.sh KEY MODEL: which threads must carry uclamp_min for the decode gain? One model, four placements, one boot, Latin square.
#   N none; T clamp the calling thread only AFTER mlx and mlx_lm are imported (what the serve shim does today);
#   B clamp before anything starts (same as uclampset -m 1024); A clamp every thread of the process after the imports.
# All arms run under schedutil (the stock governor; the clamp only acts under it) and the script restores the governors it found, on any exit.
# Each run is scripts/bench_decode.py (pinned length, exact token-id digest) started through run_clamped.py, which also prints the table of
# threads and their util_min before and after the run. No root beyond the governor write through the existing sudo. Nothing is installed.
# env: S (score dir) TREE PP (full PYTHONPATH, default tree plus deps) PYB TOKENS ROUNDS(4) GOV_SETTLE GOV_DRY GOV_CPUF GOV_SUDO GOV_MAXLOAD GOV_SAMP
set -u
KEY=${1:?KEY (a short label such as e4b or 9b)}
MODEL=${2:?MODEL (a Hugging Face repo id in the local cache)}
S=${GOV_S:-$WORK/score}
CPUF=${GOV_CPUF:-/sys/devices/system/cpu/cpufreq}
SUDO=${GOV_SUDO:-sudo -n}
TREE=${TREE:-main81}
PYB=${PYB:-$S/venv/bin/python}
TOKENS=${TOKENS:-128}
# One fixed prompt for every run of every arm (bench_decode.py refuses to start without one; EOS is suppressed, so it always generates TOKENS tokens).
PROMPT="Write a short story about a lighthouse keeper who finds a message in a bottle."
ROUNDS=${ROUNDS:-4}
SETTLE=${GOV_SETTLE:-15}
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH:-$HERE/bench_decode.py}
T=clamp-$KEY-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/out-$T
mkdir -p "$OUT"
L=$OUT/ticket.log
SAMP=${GOV_SAMP:-$WORK/sampler/row-sampler.sh}
CSV=${GOV_CSV:-$WORK/sampler/rows-$T.csv}
say() { echo "CLAMP $T $(date -u +%T) $*" | tee -a "$L"; }
die() { say "ABORT: $*"; exit 1; }

declare -A orig
for p in "$CPUF"/policy*; do orig[$p]=$(cat "$p/scaling_governor"); done
restore() {
	for p in "${!orig[@]}"; do echo "${orig[$p]}" | $SUDO tee "$p/scaling_governor" > /dev/null; done
	say "restored governors: $(for p in "${!orig[@]}"; do printf '%s=%s ' "$(basename "$p")" "$(cat "$p/scaling_governor")"; done)"
	[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" > /dev/null 2>&1
}
trap restore EXIT
trap 'exit 130' INT TERM
gov_set() {
	for p in "$CPUF"/policy*; do echo "$1" | $SUDO tee "$p/scaling_governor" > /dev/null || die "cannot write governor $1"; done
	for p in "$CPUF"/policy*; do [ "$(cat "$p/scaling_governor")" = "$1" ] || die "governor read-back of $(basename "$p") is not $1"; done
}

nane=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)
[ "${nane:-0}" -eq 0 ] || die "$nane ANE process(es) running; the filler must be off"
# The load average decays slowly after the previous ticket, so wait for it (at most GOV_LOADWAIT seconds) before refusing.
LOADWAIT=${GOV_LOADWAIT:-300}
waited=0
while :; do
	load=$(cut -d' ' -f1 /proc/loadavg)
	awk -v l="$load" -v m="${GOV_MAXLOAD:-0.5}" 'BEGIN{exit !(l < m)}' && break
	[ "$waited" -lt "$LOADWAIT" ] || die "load1 $load is not under ${GOV_MAXLOAD:-0.5} after waiting ${waited}s"
	sleep 10
	waited=$((waited + 10))
done
[ "$waited" -eq 0 ] || say "waited ${waited}s for load1 to settle (now $load)"
[ -f "$BENCH" ] || die "no bench_decode.py"
(cd "$HERE" && "$PYB" -c 'import caps_sim_guard, mlx_provenance') > /dev/null 2>&1 || die "bench_decode.py needs caps_sim_guard.py and mlx_provenance.py next to it in $HERE"
[ -n "${PP:-}" ] || [ -d "$S/$TREE/mlx" ] || die "no tree $S/$TREE"
export PYTHONPATH="${PP:-$S/$TREE:$S/mlxlm-latest-deps}"
export HF_HUB_OFFLINE=1 MESA_SHADER_CACHE_DISABLE=1
unset VK_ICD_FILENAMES VK_DRIVER_FILES
say "start key $KEY model $MODEL boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) load1 $load tree $TREE tokens $TOKENS rounds $ROUNDS"
say "found governors: $(for p in "${!orig[@]}"; do printf '%s=%s ' "$(basename "$p")" "${orig[$p]}"; done)"
say "sha256 prefixes: script $(sha256sum "$0" | cut -c1-16) bench_decode $(sha256sum "$BENCH" | cut -c1-16) run_clamped $(sha256sum "$HERE/run_clamped.py" | cut -c1-16)"
TREEDIR=${PP:+${PP%%:*}}
TREEDIR=${TREEDIR:-$S/$TREE}
say "libmlx $(sha256sum "$TREEDIR/mlx/lib/libmlx.so" 2> /dev/null | cut -c1-16) from $TREEDIR"
[ -f "$SAMP" ] && { bash "$SAMP" start "$CSV" >> "$L" 2>&1; say "sampler csv $CSV"; }

ORDERS=("N T B A" "T B A N" "B A N T" "A N T B")
mode_of() { case $1 in N) echo none ;; T) echo thread ;; B) echo before ;; A) echo all ;; esac; }
gov_set schedutil
r=0
while [ "$r" -lt "$ROUNDS" ]; do
	for a in ${ORDERS[$((r % 4))]}; do
		sleep "$SETTLE"
		f=$OUT/run-$a$((r + 1))
		t0=$(date +%s)
		say "round $((r + 1)) arm $a mode $(mode_of "$a") governors=$(cat "$CPUF"/policy*/scaling_governor | sort -u | tr '\n' ',') load1=$(cut -d' ' -f1 /proc/loadavg) start_epoch=$t0"
		timeout 900 "$PYB" "$HERE/run_clamped.py" "$(mode_of "$a")" "$BENCH" --model "$MODEL" --prompt "$PROMPT" --tokens "$TOKENS" --temp 0 --seed 0 > "$f.out" 2> "$f.err"
		rc=$?
		t1=$(date +%s)
		say "round $((r + 1)) arm $a end_epoch=$t1 rc=$rc $(grep -h 'decode .* tok/s over' "$f.out" | head -1) $(grep -h 'generated_ids' "$f.out" | head -1 | cut -c1-90)"
		[ "$rc" -eq 0 ] || die "round $((r + 1)) arm $a exited $rc, stopping the ticket: $(grep -v '^CLAMP' "$f.err" | tail -2 | tr '\n' ' ' | cut -c1-300)"
		grep -h '^CLAMP' "$f.err" | cut -c1-360 >> "$L"
	done
	r=$((r + 1))
done
[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" >> "$L" 2>&1
python3 "$HERE/clampab_analyze.py" "$OUT" "$ROUNDS" "$CSV" "$L" 2>&1 | tee -a "$L"
say "DONE"
