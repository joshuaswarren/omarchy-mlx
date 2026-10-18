#!/bin/bash
# serveab7.sh KEY MODEL WEIGHTS_GIB: does the SERVER users run (mlx-omarchy serve) get the performance governor gain? One model, two arms, one boot.
#   S  CPU governor schedutil (the stock governor), MLX_OMARCHY_UCLAMP_MIN unset (the shim clamps to 1024 by default)
#   P  CPU governor performance, the same environment
# Each run starts the real server with the user's command line, sends one warm-up request (the model loads on the first request),
# prints the util_min table of the server's threads, then REQS request pairs (32 and 128 tokens, not streamed). Order S P | P S | S P | P S. The governor is set (sudo -n, read back) before each run; the start values are restored on any exit and verified.
# The governors are forced to schedutil (the stock governor; the clamp only acts under it) and restored on any exit.
# env: SERVE_CMD ("mlx-omarchy -m mlx_omarchy_serve"), S, CPUF, GOV_* (as clampab.sh), ROUNDS(4), REQS(3), TOKENS(128), PORT0(18700), READY(300)
set -u
KEY=${1:?KEY (a short label such as e4b)}
MODEL=${2:?MODEL (a Hugging Face repo id in the local cache, or a model directory)}
WGIB=${3:?WEIGHTS_GIB (the checkpoint size, for the memory plan of a model that is not in the catalog)}
CPUF=${GOV_CPUF:-/sys/devices/system/cpu/cpufreq}
SUDO=${GOV_SUDO:-sudo -n}
SERVE_CMD=${SERVE_CMD:-mlx-omarchy -m mlx_omarchy_serve}
ROUNDS=${ROUNDS:-4}
REQS=${REQS:-3}
TOKENS=${TOKENS:-128}
PORT0=${PORT0:-18700}
READY=${READY:-300}
SETTLE=${GOV_SETTLE:-15}
# One fixed prompt for every request of every arm.
PROMPT="Write a short story about a lighthouse keeper who finds a message in a bottle."
HERE=$(cd "$(dirname "$0")" && pwd)
T=serve7-$KEY-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/out-$T
mkdir -p "$OUT"
L=$OUT/ticket.log
SAMP=${GOV_SAMP:-$WORK/sampler/row-sampler.sh}
CSV=${GOV_CSV:-$WORK/sampler/rows-$T.csv}
say() { echo "SERVEAB7 $T $(date -u +%T) $*" | tee -a "$L"; }
die() { say "ABORT: $*"; exit 1; }

declare -A orig
for p in "$CPUF"/policy*; do orig[$p]=$(cat "$p/scaling_governor"); done
SP=""
stop_server() {
	[ -n "$SP" ] && { kill -- -"$SP" 2> /dev/null || kill "$SP" 2> /dev/null; wait "$SP" 2> /dev/null; }
	SP=""
}
restore() {
	stop_server
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
[ "${nane:-0}" -eq 0 ] || die "$nane ANE process(es) running; the filler must be off"
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
for f in serve_client.py thread_table.py serveab7_analyze.py; do [ -f "$HERE/$f" ] || die "missing $f next to the script"; done
export HF_HUB_OFFLINE=1
say "start key $KEY model $MODEL weights_gib $WGIB boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) load1 $load tokens $TOKENS reqs $REQS rounds $ROUNDS"
say "serve command: $SERVE_CMD serve --offline --yes --port PORT --context 4096 --weights-gib $WGIB $MODEL"
say "package: $(pacman -Q omarchy-mlx 2> /dev/null | tr '\n' ' ')"
say "found governors: $(for p in "${!orig[@]}"; do printf '%s=%s ' "$(basename "$p")" "${orig[$p]}"; done)"
say "sha256 prefixes: script $(sha256sum "$0" | cut -c1-16) client $(sha256sum "$HERE/serve_client.py" | cut -c1-16) threads $(sha256sum "$HERE/thread_table.py" | cut -c1-16)"
[ -f "$SAMP" ] && { bash "$SAMP" start "$CSV" >> "$L" 2>&1; say "sampler csv $CSV"; }

ORDERS=("S P" "P S" "S P" "P S")
r=0
idx=0
while [ "$r" -lt "$ROUNDS" ]; do
	for a in ${ORDERS[$((r % 4))]}; do
		case $a in S) gov_set schedutil ;; P) gov_set performance ;; esac
		sleep "$SETTLE"
		idx=$((idx + 1))
		port=$((PORT0 + idx))
		f=$OUT/run-$a$((r + 1))
		ss -ltn 2> /dev/null | grep -q ":$port " && die "port $port is already in use"
		t0=$(date +%s)
		say "round $((r + 1)) arm $a governors=$(cat "$CPUF"/policy*/scaling_governor | sort -u | tr '\n' ',') load1=$(cut -d' ' -f1 /proc/loadavg) port $port start_epoch=$t0"
		setsid env -u MLX_OMARCHY_UCLAMP_MIN $SERVE_CMD serve --offline --yes --port "$port" --context 4096 --weights-gib "$WGIB" "$MODEL" > "$f.serve.log" 2>&1 &
		SP=$!
		ok=0
		for ((i = 0; i < READY; i += 2)); do
			sleep 2
			kill -0 "$SP" 2> /dev/null || break
			curl -fs "http://127.0.0.1:$port/v1/models" -o /dev/null 2> /dev/null && { ok=1; break; }
		done
		[ "$ok" -eq 1 ] || die "round $((r + 1)) arm $a: the server did not answer on port $port: $(grep -v '^$' "$f.serve.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
		python3 "$HERE/serve_client.py" "$port" 8 "$PROMPT" > "$f.warm" 2>&1 || die "round $((r + 1)) arm $a: the warm-up request failed: $(cut -c1-300 "$f.warm")"
		spid=$(pgrep -f "_mlxlm_server.*--port $port" | head -1)
		say "round $((r + 1)) arm $a shim line: $(grep -h 'uclamp' "$f.serve.log" | head -1 | cut -c1-160)"
		[ -n "$spid" ] || die "round $((r + 1)) arm $a: no server process found for port $port"
		say "round $((r + 1)) arm $a $(python3 "$HERE/thread_table.py" "$spid" | cut -c1-300)"
		for ((q = 1; q <= REQS; q++)); do
			python3 "$HERE/serve_client.py" "$port" "$TOKENS" "$PROMPT" > "$f.req$q" 2>&1
			rc=$?
			say "round $((r + 1)) arm $a req $q rc=$rc $(cut -c1-250 "$f.req$q")"
			[ "$rc" -eq 0 ] || die "round $((r + 1)) arm $a request $q failed, stopping the ticket"
		done
		say "round $((r + 1)) arm $a governors at the end: $(cat "$CPUF"/policy*/scaling_governor | sort -u | tr '\n' ',')"
		say "round $((r + 1)) arm $a end_epoch=$(date +%s)"
		stop_server
	done
	r=$((r + 1))
done
[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" >> "$L" 2>&1
python3 "$HERE/serveab7_analyze.py" "$OUT" "$ROUNDS" "$REQS" "$CSV" "$L" 2>&1 | tee -a "$L"
say "DONE"
