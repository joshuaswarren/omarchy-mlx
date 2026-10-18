#!/bin/bash
# fixab_ab.sh KEY MODEL_DIR WEIGHTS_GIB: BEFORE and AFTER the two fixes (rotating-cache mask, per-token logprob read), 13-inch, gemma-4-E4B.
#   A    bench_decode, tree main81 (anchor)
#   G    mlx_lm.server generator in-process, no HTTP, tree main81            (before)
#   GF   the same on tree main81-fix                                        (after)
#   H    the real server (python -m mlx_omarchy_serve serve), tree main81     (before)
#   HF   the real server, tree main81-fix                                     (after)
# main81-fix = main81 with the two patches applied to a copy of its mlx_lm (patch-mlx-lm-rotkv-maskless.py, mlx-lm-server-skip-logprob-read.patch);
# the mlx wheel part is the same files (symlinks). Same prompt, 128 tokens, greedy, governor performance READ not written, ANE filler off.
# G-type arms and server arms run REQS request pairs per process; the server rate is the wall difference of a TOKENS//4 and a TOKENS request
# (serve_client.py, non-streamed). ROUNDS rounds, each a rotation of the five arms. REGISTERED RULE: fixab_analyze.py header.
# No governor write, no root, no kill except the one server this script starts (its own process group). env: PYB, ROUNDS(3), REQS(2), TOKENS(128), PORT0(18900)
set -u
KEY=${1:?KEY}
MODEL=${2:?MODEL_DIR}
WGIB=${3:?WEIGHTS_GIB}
S=${GOV_S:-$WORK/score}
CPUF=${GOV_CPUF:-/sys/devices/system/cpu/cpufreq}
PYB=${PYB:-$S/venv/bin/python}
SERVEPKG=${SERVEPKG:-$WORK/serve-pkg/serve}
ROUNDS=${ROUNDS:-3}
REQS=${REQS:-2}
TOKENS=${TOKENS:-128}
PORT0=${PORT0:-18900}
READY=${READY:-300}
SETTLE=${GOV_SETTLE:-15}
PROMPT="Write a short story about a lighthouse keeper who finds a message in a bottle."
HERE=$(cd "$(dirname "$0")" && pwd)
T=fixab-$KEY-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/out-$T
mkdir -p "$OUT"
L=$OUT/ticket.log
SAMP=${GOV_SAMP:-/var/tmp/sampler/row-sampler.sh}
CSV=${GOV_CSV:-/var/tmp/sampler/rows-$T.csv}
say() { echo "FIXAB $T $(date -u +%T) $*" | tee -a "$L"; }
die() { say "ABORT: $*"; exit 1; }
SP=""
stop_server() {
	[ -n "$SP" ] && { kill -- -"$SP" 2> /dev/null || kill "$SP" 2> /dev/null; wait "$SP" 2> /dev/null; }
	SP=""
}
cleanup() { stop_server; [ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" > /dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 130' INT TERM

for p in "$CPUF"/policy*; do [ "$(cat "$p/scaling_governor")" = performance ] || die "governor of $(basename "$p") is $(cat "$p/scaling_governor"), not performance"; done
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
for f in bench_decode.py run_clamped.py mlx_provenance.py caps_sim_guard.py servegen_inproc.py serve_client.py thread_table.py fixab_analyze.py; do
	[ -f "$HERE/$f" ] || die "missing $f next to the script"
done
[ -d "$MODEL" ] || die "model directory $MODEL does not exist"
for t in main81 main81-fix; do [ -d "$S/$t/mlx" ] && [ -d "$S/$t/mlx_lm" ] || die "no tree $S/$t"; done
tree_env() { echo "$S/$1:$S/mlxlm-latest-deps:$SERVEPKG"; }
export HF_HUB_OFFLINE=1 MESA_SHADER_CACHE_DISABLE=1 MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 PYTHONDONTWRITEBYTECODE=1
unset VK_ICD_FILENAMES VK_DRIVER_FILES MLX_OMARCHY_UCLAMP_MIN
say "start key $KEY model $MODEL weights_gib $WGIB boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) load1 $load tokens $TOKENS reqs $REQS rounds $ROUNDS"
say "governors: $(cat "$CPUF"/policy*/scaling_governor | sort | uniq -c | tr '\n' ' ') (read only)"
for t in main81 main81-fix; do
	say "tree $t libmlx $(sha256sum "$S/$t/mlx/lib/libmlx.so" 2> /dev/null | cut -c1-16) cache.py $(sha256sum "$S/$t/mlx_lm/models/cache.py" | cut -c1-16) server.py $(sha256sum "$S/$t/mlx_lm/server.py" | cut -c1-16) rotkv marker $(grep -c MLX_OMARCHY_ROTKV_MASKLESS "$S/$t/mlx_lm/models/cache.py") logprob marker $(grep -c MLX_OMARCHY_SERVER_LOGPROB_READ "$S/$t/mlx_lm/server.py")"
done
say "sha256 prefixes: script $(sha256sum "$0" | cut -c1-16)$(for f in bench_decode.py run_clamped.py servegen_inproc.py serve_client.py thread_table.py fixab_analyze.py; do printf ' %s %s' "$f" "$(sha256sum "$HERE/$f" | cut -c1-16)"; done)"
[ -f "$SAMP" ] && { bash "$SAMP" start "$CSV" >> "$L" 2>&1; say "sampler csv $CSV"; }

read -ra AR <<< "${ARMS:-A G GF H HF}"
rot() { local i n=${#AR[@]}; for ((i = 0; i < n; i++)); do printf '%s ' "${AR[$(((i + $1) % n))]}"; done; }
idx=0
r=0
while [ "$r" -lt "$ROUNDS" ]; do
	for a in $(rot "$r"); do
		sleep "$SETTLE"
		idx=$((idx + 1))
		f=$OUT/run-$a$((r + 1))
		t0=$(date +%s)
		say "round $((r + 1)) arm $a load1=$(cut -d' ' -f1 /proc/loadavg) start_epoch=$t0"
		case $a in
			A)
				env PYTHONPATH="$(tree_env main81)" timeout 900 "$PYB" "$HERE/run_clamped.py" thread "$HERE/bench_decode.py" --model "$MODEL" --prompt "$PROMPT" --tokens "$TOKENS" --temp 0 --seed 0 > "$f.out" 2> "$f.err"
				rc=$?
				say "round $((r + 1)) arm A rc=$rc $(grep -h 'decode .* tok/s over' "$f.out" | head -1) $(grep -h 'generated_ids' "$f.out" | head -1 | cut -c1-90)"
				[ "$rc" -eq 0 ] || die "round $((r + 1)) arm A exited $rc: $(grep -v '^CLAMP' "$f.err" | tail -2 | tr '\n' ' ' | cut -c1-300)"
				;;
			G | GF)
				tree=main81
				[ "$a" = GF ] && tree=main81-fix
				env PYTHONPATH="$(tree_env $tree)" MLX_OMARCHY_SERVE_CONTEXT_LIMIT=4096 timeout 900 "$PYB" "$HERE/servegen_inproc.py" G "$MODEL" "$REQS" "$TOKENS" "$PROMPT" > "$f.out" 2> "$f.err"
				rc=$?
				grep -h -E '^(provenance|ARM)' "$f.out" | cut -c1-300 >> "$L"
				say "round $((r + 1)) arm $a tree $tree rc=$rc pairs $(grep -c '"pair"' "$f.out")"
				[ "$rc" -eq 0 ] || die "round $((r + 1)) arm $a exited $rc: $(grep -v 'INFO -' "$f.err" | tail -2 | tr '\n' ' ' | cut -c1-300)"
				;;
			H | HF)
				tree=main81
				[ "$a" = HF ] && tree=main81-fix
				port=$((PORT0 + idx))
				ss -ltn 2> /dev/null | grep -q ":$port " && die "port $port is already in use"
				setsid env PYTHONPATH="$(tree_env $tree)" "$PYB" -m mlx_omarchy_serve serve --offline --yes --port "$port" --context 4096 --weights-gib "$WGIB" "$MODEL" > "$f.serve.log" 2>&1 &
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
				[ -n "$spid" ] || die "round $((r + 1)) arm $a: no server process found for port $port"
				say "round $((r + 1)) arm $a tree $tree $(python3 "$HERE/thread_table.py" "$spid" | cut -c1-200)"
				for ((q = 1; q <= REQS; q++)); do
					python3 "$HERE/serve_client.py" "$port" "$TOKENS" "$PROMPT" > "$f.req$q" 2>&1
					rc=$?
					say "round $((r + 1)) arm $a req $q rc=$rc $(cut -c1-250 "$f.req$q")"
					[ "$rc" -eq 0 ] || die "round $((r + 1)) arm $a request $q failed, stopping the ticket"
				done
				stop_server
				;;
		esac
		say "round $((r + 1)) arm $a end_epoch=$(date +%s)"
	done
	r=$((r + 1))
done
[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" >> "$L" 2>&1
python3 "$HERE/fixab_analyze.py" "$OUT" "$ROUNDS" "$REQS" "$CSV" "$L" 2>&1 | tee -a "$L"
say "DONE"
